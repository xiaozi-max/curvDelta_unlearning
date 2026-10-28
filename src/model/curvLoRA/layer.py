import torch
import torch.nn.functional as F
from peft.tuners.lora.layer import LoraLayer, LoraVariant


class curvLoRA(LoraVariant):
    """LoRA variant that applies bilateral low-curvature projection.

    Mathematically the forward pass computes:
        h = (I - Ū_out Ū_out^T) B A (I - Ū_in Ū_in^T) x

    where Ū_in and Ū_out are frozen high-curvature eigenvector bases stored
    as buffers on the LoRA module, and A, B are the standard learnable LoRA
    matrices.

    Implementation note: when LoRA parameters are locally available, the
    projections are folded into the LoRA parameters each forward
    (parameter-level, O(r·d·k) per layer), avoiding two B·T·d-sized matmuls
    plus their saved activations. The trainer also writes ``A_eff`` / ``B_eff``
    back into ``lora_A`` / ``lora_B`` after each optimizer step so a vanilla
    PEFT save stores the projected adapter. Under DeepSpeed ZeRO-3 the LoRA
    weights are sharded and only materialised inside the wrapped ``nn.Linear``
    forward, so the implementation falls back to the original token-level
    projection in that case (detected via ``hasattr(weight, "ds_id")``).

    Dropout placement: in the parameter-level path dropout is applied to
    ``x`` (matching standard PEFT LoRA's ``lora_B(lora_A(dropout(x)))``); the
    token-level fallback applies dropout to the projected input ``P_in x``,
    matching the previous implementation. With ``lora_dropout=0`` both forms
    are identical.

    The merged delta weight is:
        ΔW = (I - Ū_out Ū_out^T) @ B @ A @ (I - Ū_in Ū_in^T) * scaling
    """

    @staticmethod
    def init(module: LoraLayer, adapter_name: str, **kwargs) -> None:
        base_weight = (
            module.base_layer.weight if hasattr(module, "base_layer") else module.weight
        )
        module.register_buffer(
            f"U_in_bar_{adapter_name}",
            base_weight.new_zeros((module.in_features, 0), dtype=base_weight.dtype),
        )
        module.register_buffer(
            f"U_out_bar_{adapter_name}",
            base_weight.new_zeros((module.out_features, 0), dtype=base_weight.dtype),
        )

    @staticmethod
    def forward(
        module: LoraLayer,
        active_adapter: str,
        x: torch.Tensor,
        result: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        U_in_bar = getattr(module, f"U_in_bar_{active_adapter}")
        U_out_bar = getattr(module, f"U_out_bar_{active_adapter}")
        lora_A = module.lora_A[active_adapter]
        lora_B = module.lora_B[active_adapter]
        dropout = module.lora_dropout[active_adapter]
        scaling = module.scaling[active_adapter]

        # DeepSpeed ZeRO-3 partitions LoRA parameters; the full weight is only
        # materialised inside the wrapped nn.Linear's forward via gather hooks.
        # We can only fold projections into the LoRA parameters when the full
        # weight is locally available; otherwise fall back to the token-level form.
        weight_A = lora_A.weight
        weight_B = lora_B.weight
        sharded = hasattr(weight_A, "ds_id") or hasattr(weight_B, "ds_id")

        if sharded:
            # Token-level form: keep nn.Linear forwards so DeepSpeed gather hooks fire.
            U_in_bar_x = U_in_bar.to(dtype=x.dtype) if U_in_bar.dtype != x.dtype else U_in_bar
            U_out_bar_x = U_out_bar.to(dtype=x.dtype) if U_out_bar.dtype != x.dtype else U_out_bar
            if U_in_bar_x.shape[1] > 0:
                x_proj = x - (x @ U_in_bar_x) @ U_in_bar_x.T
            else:
                x_proj = x
            h = lora_B(lora_A(dropout(x_proj)))
            if U_out_bar_x.shape[1] > 0:
                h = h - (h @ U_out_bar_x) @ U_out_bar_x.T
            return result + h * scaling

        # Parameter-level form: fold projections into A_eff / B_eff once per
        # forward (cost O(r·d·k), independent of batch×seq), avoiding the two
        # B·T·d-sized projection matmuls and their saved activations.
        A_eff, B_eff = curvLoRA.folded_lora_weights(
            module, active_adapter, clone=False
        )
        compute_dtype = weight_A.dtype
        x_in = dropout(x)
        if x_in.dtype != compute_dtype:
            x_in = x_in.to(compute_dtype)
        h = F.linear(F.linear(x_in, A_eff), B_eff)
        return result + (h * scaling).to(result.dtype)

    @staticmethod
    def folded_lora_weights(
        module: LoraLayer, active_adapter: str, *, clone: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``A @ P_in`` and ``P_out @ B`` without mutating live weights.

        ``A_eff = A - (A Ū_in) Ū_in^T``, ``B_eff = B - Ū_out (Ū_out^T B)``.
        Empty bases leave the original weight unchanged.
        """
        U_in_bar = getattr(module, f"U_in_bar_{active_adapter}")
        U_out_bar = getattr(module, f"U_out_bar_{active_adapter}")
        weight_A = module.lora_A[active_adapter].weight
        weight_B = module.lora_B[active_adapter].weight
        compute_dtype = weight_A.dtype
        if U_in_bar.dtype != compute_dtype:
            U_in_bar = U_in_bar.to(dtype=compute_dtype)
        if U_out_bar.dtype != compute_dtype:
            U_out_bar = U_out_bar.to(dtype=compute_dtype)

        if U_in_bar.shape[1] > 0:
            A_eff = weight_A - (weight_A @ U_in_bar) @ U_in_bar.T
        else:
            A_eff = weight_A
        if U_out_bar.shape[1] > 0:
            B_eff = weight_B - U_out_bar @ (U_out_bar.T @ weight_B)
        else:
            B_eff = weight_B

        if clone:
            A_eff = A_eff.detach().clone()
            B_eff = B_eff.detach().clone()
        return A_eff, B_eff

    @staticmethod
    def project_lora_weights_(module: LoraLayer, active_adapter: str) -> None:
        """Write ``A_eff`` / ``B_eff`` back into the live LoRA parameters.

        Used after each optimizer step so a later vanilla PEFT save stores the
        projected adapter. Sharded ZeRO-3 weights are left untouched; the
        token-level forward still applies the projection in that case.
        """
        weight_A = module.lora_A[active_adapter].weight
        weight_B = module.lora_B[active_adapter].weight
        if hasattr(weight_A, "ds_id") or hasattr(weight_B, "ds_id"):
            return
        A_eff, B_eff = curvLoRA.folded_lora_weights(
            module, active_adapter, clone=False
        )
        with torch.no_grad():
            if A_eff is not weight_A:
                weight_A.copy_(A_eff)
            if B_eff is not weight_B:
                weight_B.copy_(B_eff)

    @staticmethod
    def compute_effective_delta_weight(
        module: LoraLayer, active_adapter: str, *, fp32: bool = False
    ) -> torch.Tensor:
        """Compute (I - Ū_out Ū_out^T) @ B @ A @ (I - Ū_in Ū_in^T) * scaling.

        When ``fp32=True``, the returned tensor is always float32 regardless of
        the native weight dtype, which is useful for norm computations.
        """
        U_in_bar = getattr(module, f"U_in_bar_{active_adapter}")
        U_out_bar = getattr(module, f"U_out_bar_{active_adapter}")
        weight_A = module.lora_A[active_adapter].weight
        weight_B = module.lora_B[active_adapter].weight
        scaling = module.scaling[active_adapter]

        device = weight_B.device
        dtype = weight_B.dtype

        cast_to_fp32 = fp32 or (
            device.type == "cpu"
            and (dtype == torch.float16 or dtype == torch.bfloat16)
        )
        if cast_to_fp32:
            weight_A = weight_A.float()
            weight_B = weight_B.float()
            U_in_bar = U_in_bar.float()
            U_out_bar = U_out_bar.float()
        else:
            U_in_bar = U_in_bar.to(dtype=dtype)
            U_out_bar = U_out_bar.to(dtype=dtype)

        # BA: (d_out, d_in)
        BA = weight_B @ weight_A

        # Left projection: (I - Ū_out Ū_out^T) @ BA
        P_out_BA = BA - U_out_bar @ (U_out_bar.T @ BA)

        # Right projection: P_out_BA @ (I - Ū_in Ū_in^T)
        delta = P_out_BA - (P_out_BA @ U_in_bar) @ U_in_bar.T

        delta = delta * scaling

        if cast_to_fp32 and not fp32 and dtype != torch.float32:
            delta = delta.to(dtype=dtype)

        return delta

    @staticmethod
    def effective_delta_weight_norm(
        module: LoraLayer, active_adapter: str
    ) -> torch.Tensor:
        """Return ||ΔW_eff||_F (a scalar tensor) computed in float32."""
        delta = curvLoRA.compute_effective_delta_weight(
            module, active_adapter, fp32=True
        )
        return torch.linalg.vector_norm(delta)

    @staticmethod
    def _compute_delta_weight(module: LoraLayer, active_adapter: str) -> torch.Tensor:
        """Compatibility wrapper — see ``compute_effective_delta_weight``."""
        return curvLoRA.compute_effective_delta_weight(
            module, active_adapter, fp32=False
        )

    @staticmethod
    def merge_safe(
        module: LoraLayer, active_adapter: str, orig_weight: torch.Tensor
    ) -> torch.Tensor:
        delta = curvLoRA._compute_delta_weight(module, active_adapter)
        return orig_weight + delta.to(orig_weight.dtype)

    @staticmethod
    def merge_unsafe(
        module: LoraLayer, active_adapter: str, orig_weight: torch.Tensor
    ) -> None:
        delta = curvLoRA._compute_delta_weight(module, active_adapter)
        orig_weight.data += delta.to(orig_weight.dtype)

    @staticmethod
    def unmerge(
        module: LoraLayer, active_adapter: str, orig_weight: torch.Tensor
    ) -> torch.Tensor:
        delta = curvLoRA._compute_delta_weight(module, active_adapter)
        return orig_weight - delta.to(orig_weight.dtype)
