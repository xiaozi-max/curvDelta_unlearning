"""VILA LoRA initialization (arXiv:2508.21300).

This module is **preprocessing-only**: it computes Fisher importance for LoRA
parameters and rewrites ``lora_A``, ``lora_B`` and ``base_layer.weight`` so
that ``W = residual + scaling · B @ A``. The unlearn / eval pipeline never
imports this module — it just loads the residual base and the VILA-init
adapter from disk like any ordinary PEFT checkpoint.

See ``scripts/prepare_vila_init.py`` for the end-to-end driver.
"""

import logging
import os
from functools import reduce
from typing import Dict, Optional

from tqdm import tqdm
import torch
from torch import nn

logger = logging.getLogger(__name__)

_LORA_A_TOKEN = ".lora_A"
_LORA_B_TOKEN = ".lora_B"


def _normalize_param_name(name: str) -> str:
    return name.replace("module.", "")


def _module_layer_name(lora_a_name: str) -> str:
    """Map a ``...Q.lora_A.default.weight``-style name to ``...Q``."""
    name = _normalize_param_name(lora_a_name)
    return name.replace(".lora_A.default.weight", "")


def _get_module_by_name(model, dotted: str):
    return reduce(getattr, dotted.split("."), model)


def compute_fisher_from_grad_stats(
    grad_sum: torch.Tensor,
    sq_grad_sum: torch.Tensor,
    cnt: torch.Tensor,
) -> torch.Tensor:
    """Variance estimator: ``E[g^2] - (E[g])^2``, clamped to be non-negative."""
    cnt = cnt.to(sq_grad_sum.dtype).to(sq_grad_sum.device)
    mean_sq = sq_grad_sum / cnt
    mean = grad_sum / cnt.to(grad_sum.device)
    return torch.clamp(mean_sq - mean.pow(2), min=0)


@torch.no_grad()
def apply_vila_init(
    model,
    importance_payload: Dict,
    eps: float = 1e-5,
    rank: Optional[int] = None,
    rebalance: bool = True,
):
    """Rewrite LoRA weights using the VILA initialization scheme.

    Args:
        model: A PEFT-LoRA-wrapped model.
        importance_payload: dict produced by ``compute_vila_importance``
            (keys ``f_cnt``, ``r_cnt``, ``importance_f``, ``importance_r``).
        eps: Numerical floor used both in the F/R ratio and the row-importance
            inversion. Matches the original VILA implementation.
        rank: Override for the LoRA rank used in ``svd_lowrank``. If ``None``
            (default), inferred from each LoRA layer's ``r``.
        rebalance: If True (default), per-rank renormalize ``B`` and ``A`` so
            that ``||B[:, j]||_2 == ||A[j, :]||_2`` for every rank ``j``. The
            product ``B @ A`` (and therefore the residual / model output) is
            preserved exactly. The raw VILA formula multiplies columns of
            ``B`` by ``1/(row_imp + eps)``, which can inflate column norms by
            10²-10³× when ``row_imp`` has near-zero rows. Adam-style optimizers
            apply unit-magnitude per-parameter updates, so an unbalanced (B,A)
            pair makes ``B @ ΔA`` dominate ``ΔB @ A`` by the same factor.
            Rebalancing is a coordinate change in the rank-r subspace that
            recovers the same scale geometry vanilla LoRA enjoys.

    The function mutates ``model`` in-place. After this call the caller is
    expected to ``save_pretrained`` both the residual base and the adapter.
    """
    f_cnt = importance_payload["f_cnt"]
    r_cnt = importance_payload["r_cnt"]
    importance_f: Dict[str, torch.Tensor] = importance_payload["importance_f"]
    importance_r: Dict[str, torch.Tensor] = importance_payload["importance_r"]

    f_cnt_t = torch.as_tensor(f_cnt, dtype=torch.float32)
    r_cnt_t = torch.as_tensor(r_cnt, dtype=torch.float32)

    layer_importances: Dict[str, torch.Tensor] = {}
    for name in importance_f.keys():
        if _LORA_B_TOKEN in name or _LORA_A_TOKEN not in name:
            continue
        lora_b_name = name.replace(_LORA_A_TOKEN, _LORA_B_TOKEN)
        if lora_b_name not in importance_f:
            logger.warning(f"VILA: missing lora_B counterpart for {name}; skipping.")
            continue

        imp_a_f = importance_f[name].to(torch.float32)
        imp_b_f = importance_f[lora_b_name].to(torch.float32)
        imp_a_r = importance_r[name].to(torch.float32)
        imp_b_r = importance_r[lora_b_name].to(torch.float32)

        imp_f_BA = imp_b_f @ imp_a_f
        imp_r_BA = imp_b_r @ imp_a_r

        # ratio = (imp_f_BA / f_cnt_t) / (eps + (imp_r_BA / r_cnt_t))
        ratio = imp_f_BA / (eps + imp_r_BA)
        layer_importances[_module_layer_name(name)] = ratio

    if not layer_importances:
        raise RuntimeError("VILA: importance payload had no usable lora_A/lora_B pairs.")

    augmented = 0
    for layer_name, importance in layer_importances.items():
        try:
            lora_A_dict = _get_module_by_name(model, layer_name + ".lora_A")
            lora_B_dict = _get_module_by_name(model, layer_name + ".lora_B")
            base_layer = _get_module_by_name(model, layer_name + ".base_layer")
            scaling_dict = _get_module_by_name(model, layer_name + ".scaling")
        except AttributeError:
            logger.warning(f"VILA: cannot locate LoRA submodules under '{layer_name}'.")
            continue

        adapter_name = next(iter(lora_A_dict.keys()))
        scaling = scaling_dict[adapter_name]

        weight_device = base_layer.weight.device
        if weight_device.type == "cuda":
            compute_device = weight_device
        elif torch.cuda.is_available():
            local_rank = int(os.environ.get("LOCAL_RANK", "0"))
            local_rank = min(local_rank, torch.cuda.device_count() - 1)
            compute_device = torch.device(f"cuda:{local_rank}")
        else:
            compute_device = weight_device
        dtype = base_layer.weight.dtype

        W = base_layer.weight.data.to(device=compute_device, dtype=torch.float32)
        importance = importance.to(device=compute_device, dtype=torch.float32)

        layer_rank = rank or lora_A_dict[adapter_name].weight.shape[0]
        layer_rank = min(layer_rank, min(W.shape))

        row_importance = importance.sum(dim=1).clamp(min=0).sqrt()  # [out]
        weighted_W = row_importance.unsqueeze(1) * W  # [out, in]
        U, S, V = torch.svd_lowrank(weighted_W, q=layer_rank)

        S_scaled = S / scaling
        new_lora_A = (V * torch.sqrt(S_scaled)).t().contiguous()  # [r, in]
        new_lora_B = (
            (1.0 / (row_importance + eps)).unsqueeze(1) * (U * torch.sqrt(S_scaled))
        ).contiguous()  # [out, r]

        if rebalance:
            b_norm = new_lora_B.norm(dim=0).clamp_min(1e-12)  # [r]
            a_norm = new_lora_A.norm(dim=1).clamp_min(1e-12)  # [r]
            scale = (b_norm / a_norm).sqrt()
            new_lora_B = new_lora_B / scale  # [out, r]
            new_lora_A = new_lora_A * scale.unsqueeze(1)  # [r, in]

        new_residual = W - scaling * (new_lora_B @ new_lora_A)

        lora_A_dict[adapter_name].weight.data = new_lora_A.to(
            device=weight_device, dtype=dtype
        )
        lora_B_dict[adapter_name].weight.data = new_lora_B.to(
            device=weight_device, dtype=dtype
        )
        base_layer.weight.data = new_residual.to(device=weight_device, dtype=dtype)
        augmented += 1

    logger.info(f"VILA: initialized {augmented}/{len(layer_importances)} LoRA layers.")
    return model


# ---------------------------------------------------------------------------
# Phase 1 — Importance computation
# ---------------------------------------------------------------------------


def _init_lora_gaussian(model, sigma: float) -> None:
    """Override default PEFT init with ``Normal(0, sigma)`` for lora_A/lora_B."""
    for name, param in model.named_parameters():
        if _LORA_A_TOKEN in name or _LORA_B_TOKEN in name:
            torch.nn.init.normal_(param, mean=0.0, std=sigma)


def compute_vila_importance(
    model,
    forget_loader,
    retain_loader,
    *,
    sigma: float = 0.05,
    num_epochs: int = 1,
    max_steps: Optional[int] = None,
    device: Optional[torch.device] = None,
):
    """Run the VILA Fisher-importance accumulation loop.

    Both loaders are expected to yield batches with keys ``input_ids``,
    ``attention_mask``, ``labels``. The model must already have LoRA adapters
    attached. Only ``lora_A`` / ``lora_B`` parameters require gradients.

    The number of steps per epoch is ``min(len(forget_loader),
    len(retain_loader))`` — running ``max(...)`` with cycling causes the
    smaller loader (typically forget) to be repeated ~10× per epoch on TOFU
    forget10, inflating wall time and biasing forget statistics.
    """
    device = device or next(model.parameters()).device

    for name, param in model.named_parameters():
        param.requires_grad = (_LORA_A_TOKEN in name) or (_LORA_B_TOKEN in name)
    _init_lora_gaussian(model, sigma)

    lora_params = {
        n: p for n, p in model.named_parameters()
        if (_LORA_A_TOKEN in n) or (_LORA_B_TOKEN in n)
    }
    lora_names = list(lora_params.keys())
    grad_f = {n: torch.zeros_like(p, dtype=torch.float32) for n, p in lora_params.items()}
    grad_r = {n: torch.zeros_like(p, dtype=torch.float32) for n, p in lora_params.items()}
    sq_grad_f = {n: torch.zeros_like(p, dtype=torch.float32) for n, p in lora_params.items()}
    sq_grad_r = {n: torch.zeros_like(p, dtype=torch.float32) for n, p in lora_params.items()}

    f_cnt = torch.zeros((), dtype=torch.float32, device=device)
    r_cnt = torch.zeros((), dtype=torch.float32, device=device)

    model.train()

    def _accumulate(batch, grad_buf, sq_buf):
        inputs = {
            "input_ids": batch["input_ids"].to(device, non_blocking=True),
            "attention_mask": batch["attention_mask"].to(device, non_blocking=True),
            "labels": batch["labels"].to(device, non_blocking=True),
        }
        model.zero_grad(set_to_none=True)
        outputs = model(**inputs)
        outputs.loss.backward()
        cnt = (inputs["labels"] != -100).sum().to(torch.float32)
        for n, p in lora_params.items():
            if p.grad is not None:
                g = p.grad.detach().to(torch.float32)
                grad_buf[n].add_(g * cnt)
                sq_buf[n].add_(g.pow(2) * cnt)
        return cnt

    for epoch in range(num_epochs):
        logger.info(f"VILA importance epoch {epoch + 1}/{num_epochs}")
        forget_iter = iter(forget_loader)
        retain_iter = iter(retain_loader)
        steps = min(len(forget_loader), len(retain_loader))
        if max_steps is not None:
            steps = min(steps, max_steps)
        pbar = tqdm(
            range(steps),
            desc=f"  epoch {epoch + 1}/{num_epochs}",
            disable=None,
            dynamic_ncols=True,
        )
        for step in pbar:
            try:
                f_batch = next(forget_iter)
            except StopIteration:
                forget_iter = iter(forget_loader)
                f_batch = next(forget_iter)
            try:
                r_batch = next(retain_iter)
            except StopIteration:
                retain_iter = iter(retain_loader)
                r_batch = next(retain_iter)

            f_cnt = f_cnt + _accumulate(f_batch, grad_f, sq_grad_f)
            r_cnt = r_cnt + _accumulate(r_batch, grad_r, sq_grad_r)

            pbar.set_postfix({"f_cnt": int(f_cnt.item()), "r_cnt": int(r_cnt.item())})

    importance_f = {}
    importance_r = {}
    for n in lora_names:
        importance_f[n] = compute_fisher_from_grad_stats(
            grad_f[n], sq_grad_f[n], f_cnt
        ).cpu()
        importance_r[n] = compute_fisher_from_grad_stats(
            grad_r[n], sq_grad_r[n], r_cnt
        ).cpu()

    return {
        "f_cnt": f_cnt.detach().cpu(),
        "r_cnt": r_cnt.detach().cpu(),
        "importance_f": importance_f,
        "importance_r": importance_r,
    }


__all__ = ["apply_vila_init", "compute_vila_importance"]
