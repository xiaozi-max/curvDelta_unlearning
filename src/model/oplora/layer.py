import logging

import torch
import torch.nn.functional as F
from peft.tuners.lora.layer import LoraLayer, LoraVariant

logger = logging.getLogger(__name__)


def compute_weight_basis(
    weight: torch.Tensor, k: int, *, niter: int = 4
) -> tuple[torch.Tensor, torch.Tensor]:
    """Top-k left/right singular vectors of a frozen base weight.

    ``torch.svd_lowrank`` runs in float32. Returned bases match ``weight``'s
    dtype and device and have shape ``(d_out, k_eff)`` and ``(d_in, k_eff)``.
    """
    if weight.ndim != 2:
        raise ValueError(f"OPLoRA SVD expects a 2D weight, got {tuple(weight.shape)}")
    if int(k) <= 0:
        raise ValueError(f"oplora_k must be positive, got {k}")

    d_out, d_in = weight.shape
    k_eff = min(int(k), d_out, d_in)
    if k_eff != int(k):
        logger.info(
            "OPLoRA clamped k from %s to %s for weight shape %s",
            k,
            k_eff,
            tuple(weight.shape),
        )

    work = weight.detach().float()
    U, _S, V = torch.svd_lowrank(work, q=k_eff, niter=niter)
    del work
    return U[:, :k_eff].to(dtype=weight.dtype), V[:, :k_eff].to(dtype=weight.dtype)


class OPLora(LoraVariant):
    """Bilateral orthogonal complement of the base weight's top-k SVD.

    Forward (parameter-level, ZeRO-2):

        A_eff = A - (A V_k) V_k^T
        B_eff = B - U_k (U_k^T B)
        h = W_0 x + (alpha/r) B_eff A_eff x

    ``project_lora_weights_`` copies ``A_eff`` / ``B_eff`` into the live
    parameters. A forward-only projection does not constrain AdamW or the
    saved adapter.
    """

    @staticmethod
    def init(module: LoraLayer, adapter_name: str, **kwargs) -> None:
        base_weight = (
            module.base_layer.weight if hasattr(module, "base_layer") else module.weight
        )
        module.register_buffer(
            f"oplora_V_{adapter_name}",
            base_weight.new_zeros((module.in_features, 0)),
            persistent=False,
        )
        module.register_buffer(
            f"oplora_U_{adapter_name}",
            base_weight.new_zeros((module.out_features, 0)),
            persistent=False,
        )

    @staticmethod
    def _bases(module: LoraLayer, adapter_name: str, dtype: torch.dtype):
        V = getattr(module, f"oplora_V_{adapter_name}")
        U = getattr(module, f"oplora_U_{adapter_name}")
        if V.dtype != dtype:
            V = V.to(dtype=dtype)
        if U.dtype != dtype:
            U = U.to(dtype=dtype)
        return U, V

    @staticmethod
    def _reject_sharded(weight_A: torch.Tensor, weight_B: torch.Tensor) -> None:
        if hasattr(weight_A, "ds_id") or hasattr(weight_B, "ds_id"):
            raise RuntimeError(
                "OPLoRA requires full LoRA parameters (ZeRO-2). "
                "ZeRO-3 shards the adapter, so the projection cannot be folded in."
            )

    @staticmethod
    def forward(
        module: LoraLayer,
        active_adapter: str,
        x: torch.Tensor,
        result: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        lora_A = module.lora_A[active_adapter]
        lora_B = module.lora_B[active_adapter]
        OPLora._reject_sharded(lora_A.weight, lora_B.weight)
        dropout = module.lora_dropout[active_adapter]
        scaling = module.scaling[active_adapter]
        A_eff, B_eff = OPLora.folded_lora_weights(module, active_adapter, clone=False)
        compute_dtype = lora_A.weight.dtype
        x_in = dropout(x)
        if x_in.dtype != compute_dtype:
            x_in = x_in.to(compute_dtype)
        h = F.linear(F.linear(x_in, A_eff), B_eff)
        return result + (h * scaling).to(result.dtype)

    @staticmethod
    def folded_lora_weights(
        module: LoraLayer, active_adapter: str, *, clone: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``A P_R`` and ``P_L B`` without writing the live parameters."""
        weight_A = module.lora_A[active_adapter].weight
        weight_B = module.lora_B[active_adapter].weight
        U, V = OPLora._bases(module, active_adapter, weight_A.dtype)

        if V.shape[1] > 0:
            A_eff = weight_A - (weight_A @ V) @ V.T
        else:
            A_eff = weight_A
        if U.shape[1] > 0:
            B_eff = weight_B - U @ (U.T @ weight_B)
        else:
            B_eff = weight_B

        if clone:
            A_eff = A_eff.detach().clone()
            B_eff = B_eff.detach().clone()
        return A_eff, B_eff

    @staticmethod
    def project_lora_weights_(module: LoraLayer, active_adapter: str) -> None:
        """Copy ``A_eff`` / ``B_eff`` into ``lora_A`` / ``lora_B``."""
        weight_A = module.lora_A[active_adapter].weight
        weight_B = module.lora_B[active_adapter].weight
        OPLora._reject_sharded(weight_A, weight_B)
        A_eff, B_eff = OPLora.folded_lora_weights(module, active_adapter, clone=False)
        with torch.no_grad():
            if A_eff is not weight_A:
                weight_A.copy_(A_eff)
            if B_eff is not weight_B:
                weight_B.copy_(B_eff)

    @staticmethod
    def _compute_delta_weight(module: LoraLayer, active_adapter: str) -> torch.Tensor:
        OPLora._reject_sharded(
            module.lora_A[active_adapter].weight, module.lora_B[active_adapter].weight
        )
        A_eff, B_eff = OPLora.folded_lora_weights(module, active_adapter, clone=False)
        return (B_eff @ A_eff) * module.scaling[active_adapter]

    @staticmethod
    def merge_safe(
        module: LoraLayer, active_adapter: str, orig_weight: torch.Tensor
    ) -> torch.Tensor:
        delta = OPLora._compute_delta_weight(module, active_adapter)
        return orig_weight + delta.to(orig_weight.dtype)

    @staticmethod
    def merge_unsafe(
        module: LoraLayer, active_adapter: str, orig_weight: torch.Tensor
    ) -> None:
        delta = OPLora._compute_delta_weight(module, active_adapter)
        orig_weight.data += delta.to(orig_weight.dtype)

    @staticmethod
    def unmerge(
        module: LoraLayer, active_adapter: str, orig_weight: torch.Tensor
    ) -> torch.Tensor:
        delta = OPLora._compute_delta_weight(module, active_adapter)
        return orig_weight - delta.to(orig_weight.dtype)
