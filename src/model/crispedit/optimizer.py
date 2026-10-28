"""Projected AdamW for CrispEdit.

Identical semantics to ``torch.optim.AdamW`` (including decoupled weight
decay), except that 2-D target-weight gradients are projected into a
retain-derived low-curvature subspace immediately before ``step()``.

When ``offload_state_to_cpu`` is set, the fp32 ``exp_avg``/``exp_avg_sq``
states live on CPU (paged per parameter inside ``step()``), which removes the
large full-rank AdamW state from VRAM — the same trick bnb's paged optimizers
use, without the 8-bit quantization.
"""

from __future__ import annotations

import math
from typing import Dict

import torch

from model.crispedit import project_matrix_gradient


class CrispEditAdamW(torch.optim.AdamW):
    """AdamW whose 2-D target gradients are projected before each step.

    ``projection_cache_map`` maps each target ``torch.nn.Parameter`` to its
    projection cache ``{"Ua", "Ub", "M"}`` (see
    :func:`model.crispedit.projection_cache_from_covariances`). Parameters not
    present in the map, and non-2-D gradients, follow ordinary AdamW.

    The cache is runtime state. Rebuild it for the current model parameters
    when constructing the optimizer before loading an optimizer checkpoint.
    """

    def __init__(
        self,
        params,
        projection_cache_map: Dict[torch.nn.Parameter, Dict[str, torch.Tensor]] = None,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
        amsgrad=False,
        offload_state_to_cpu: bool = False,
        **kwargs,
    ):
        if projection_cache_map is None:
            projection_cache_map = {}
        self._projection_cache_map = projection_cache_map
        self._offload_state_to_cpu = offload_state_to_cpu
        super().__init__(
            params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
            **kwargs,
        )

    def reset_cache(self, new_projection_cache_map) -> None:
        """Replace the projection cache and project existing ``exp_avg``.

        Mirrors CrispEdit's ``ProjectedAdam.reset_cache``: only 2-D ``exp_avg``
        buffers are projected; step counters and ``exp_avg_sq`` are left
        untouched and the optimizer itself is not reset.
        """
        if new_projection_cache_map is None:
            new_projection_cache_map = {}
        self._projection_cache_map = new_projection_cache_map
        for group in self.param_groups:
            for p in group["params"]:
                if p not in self.state:
                    continue
                cache = new_projection_cache_map.get(p)
                if cache is None:
                    continue
                state = self.state[p]
                if "exp_avg" in state:
                    m = state["exp_avg"]
                    if m.ndim == 2:
                        m.copy_(project_matrix_gradient(m, cache))

    @torch.no_grad()
    def step(self, closure=None):
        """Project 2-D target gradients, then run the AdamW update."""
        if not self._offload_state_to_cpu:
            for group in self.param_groups:
                for p in group["params"]:
                    if p.grad is None or p.grad.ndim != 2:
                        continue
                    cache = self._projection_cache_map.get(p)
                    if cache is not None:
                        p.grad.copy_(project_matrix_gradient(p.grad, cache))
            return super().step(closure)

        # CPU-offloaded path: per-parameter staging keeps at most one param's
        # m/v on the GPU at a time.
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                cache = self._projection_cache_map.get(p)
                if p.grad.ndim == 2 and cache is not None:
                    p.grad.copy_(project_matrix_gradient(p.grad, cache))
                self._offload_adam_step(p, group)
        return loss

    def _offload_adam_step(self, p, group) -> None:
        """AdamW update with fp32 m/v kept on CPU (staged to ``p.device``)."""
        state = self.state[p]
        if not state:
            state["step"] = torch.tensor(0.0)
            state["exp_avg"] = torch.zeros(p.numel(), dtype=torch.float32).view_as(p)
            state["exp_avg_sq"] = torch.zeros(p.numel(), dtype=torch.float32).view_as(p)
        m = state["exp_avg"].to(p.device)
        v = state["exp_avg_sq"].to(p.device)

        step_count = float(state["step"].item()) + 1.0
        state["step"].fill_(step_count)
        b1, b2 = group["betas"]
        bias_correction1 = 1.0 - b1**step_count
        bias_correction2 = 1.0 - b2**step_count
        lr = group["lr"]

        # Decoupled weight decay, then the AdamW update (matches torch).
        if group["weight_decay"] != 0:
            p.mul_(1.0 - lr * group["weight_decay"])
        m.mul_(b1).add_(p.grad, alpha=1.0 - b1)
        v.mul_(b2).addcmul_(p.grad, p.grad, value=1.0 - b2)
        denom = v.sqrt().div_(math.sqrt(bias_correction2)).add_(group["eps"])
        p.addcdiv_(m, denom, value=-lr / bias_correction1)

        state["exp_avg"] = m.to("cpu")
        state["exp_avg_sq"] = v.to("cpu")


__all__ = ["CrispEditAdamW"]
