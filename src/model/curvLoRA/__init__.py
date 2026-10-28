import hashlib
import logging
from typing import Optional

import torch

from model.curvLoRA.layer import curvLoRA

logger = logging.getLogger(__name__)


def apply_curvLoRA(
    model,
    adapter_name: Optional[str] = None,
):
    from peft.tuners.lora.layer import Linear as PeftLoraLinear

    augmented = 0
    total_lora = 0

    for module_name, module in model.named_modules():
        if not isinstance(module, PeftLoraLinear):
            continue

        total_lora += 1
        active = adapter_name or _get_active_adapter(module)
        module.lora_variant[active] = curvLoRA()
        curvLoRA.init(module, active)
        augmented += 1

    logger.info("curvLoRA: augmented %s/%s LoRA layers", augmented, total_lora)

    return model


def _get_active_adapter(module) -> Optional[str]:
    if hasattr(module, "active_adapters") and module.active_adapters:
        return module.active_adapters[0]
    if hasattr(module, "_active_adapter"):
        return module._active_adapter
    return None


def get_energy_threshold(eigenvalues: torch.Tensor, energy_ratio: float) -> float:
    total_energy = eigenvalues.sum()
    if total_energy == 0:
        return 0.0

    sorted_eigvals, _ = torch.sort(eigenvalues, descending=True)
    cumulative_energy = torch.cumsum(sorted_eigvals, dim=0)
    energy_ratios = cumulative_energy / total_energy

    rank = torch.searchsorted(energy_ratios, energy_ratio).item() + 1
    rank = min(rank, len(sorted_eigvals))
    return sorted_eigvals[rank - 1].item()


def _factor_diagnostics(factor: torch.Tensor, eigenvalues: Optional[torch.Tensor] = None) -> str:
    with torch.no_grad():
        finite_mask = torch.isfinite(factor)
        finite_values = factor[finite_mask]
        if finite_values.numel() == 0:
            finite_min = finite_max = finite_mean = float("nan")
        else:
            finite_min = finite_values.min().item()
            finite_max = finite_values.max().item()
            finite_mean = finite_values.mean().item()

        all_zero = bool(torch.all(factor == 0).item())
        rows = min(4, factor.shape[0])
        cols = min(4, factor.shape[1])
        diag_sample = factor.diagonal()[:4].detach().cpu().tolist()
        top_left = factor[:rows, :cols].detach().cpu().tolist()

        parts = [
            f"shape={tuple(factor.shape)}",
            f"dtype={factor.dtype}",
            f"device={factor.device}",
            f"all_zero={all_zero}",
            f"finite={bool(finite_mask.all().item())}",
            f"finite_min={finite_min:.6e}",
            f"finite_max={finite_max:.6e}",
            f"finite_mean={finite_mean:.6e}",
            f"diag_sample={diag_sample}",
            f"top_left_{rows}x{cols}={top_left}",
        ]

        if eigenvalues is not None:
            eig = eigenvalues.detach()
            zero_count = int((eig == 0).sum().item())
            parts.extend(
                [
                    f"eigen_min={eig.min().item():.6e}",
                    f"eigen_max={eig.max().item():.6e}",
                    f"eigen_zero_count={zero_count}",
                    f"eigen_sample={eig[:4].cpu().tolist()}",
                ]
            )

        return ", ".join(parts)


def _symmetric_eigh(
    factor: torch.Tensor,
    factor_name: Optional[str] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    label = factor_name or "<unnamed KFAC factor>"
    factor = factor.float()
    factor = 0.5 * (factor + factor.mT)

    if not torch.isfinite(factor).all():
        raise RuntimeError(
            f"Non-finite KFAC factor for {label}: {_factor_diagnostics(factor)}"
        )

    try:
        return torch.linalg.eigh(factor)
    except torch._C._LinAlgError:
        logger.warning(
            "KFAC eigendecomposition failed for %s; retrying in float64. %s",
            label,
            _factor_diagnostics(factor),
            exc_info=True,
        )

    try:
        eigenvalues, eigenvectors = torch.linalg.eigh(factor.double())
    except torch._C._LinAlgError:
        logger.exception(
            "KFAC eigendecomposition also failed in float64 for %s. %s",
            label,
            _factor_diagnostics(factor),
        )
        raise

    logger.warning(
        "KFAC eigendecomposition succeeded in float64 for %s. %s",
        label,
        _factor_diagnostics(factor, eigenvalues),
    )
    return eigenvalues.float(), eigenvectors.float()


def compute_projection_cache(
    factor: torch.Tensor,
    energy_ratio: float = 0.9,
    factor_name: Optional[str] = None,
) -> torch.Tensor:
    eigenvalues, eigenvectors = _symmetric_eigh(factor, factor_name)
    eigenvalues = eigenvalues.clamp(min=0)
    if eigenvalues.sum() == 0:
        return eigenvectors[:, :0]

    threshold = get_energy_threshold(eigenvalues, energy_ratio)
    basis = eigenvectors[:, eigenvalues >= threshold]

    return basis


def compute_projection_cache_bottom(
    factor: torch.Tensor,
    energy_ratio: float = 0.9,
    factor_name: Optional[str] = None,
) -> torch.Tensor:
    """Select bottom-k eigenvectors where k matches the top-k selection rank.

    ``torch.linalg.eigh`` returns eigenvalues in ascending order, so the
    bottom-k eigenvectors are the first k columns.
    """
    eigenvalues, eigenvectors = _symmetric_eigh(factor, factor_name)
    eigenvalues = eigenvalues.clamp(min=0)
    if eigenvalues.sum() == 0:
        return eigenvectors[:, :0]

    threshold = get_energy_threshold(eigenvalues, energy_ratio)
    k = int((eigenvalues >= threshold).sum().item())

    return eigenvectors[:, :k]


def random_orthogonal_basis(d: int, k: int, seed: int, device: torch.device) -> torch.Tensor:
    """Generate a random orthonormal basis of shape ``(d, k)``.

    Uses a CPU generator seeded deterministically so all ranks produce the
    same matrix without an explicit broadcast.
    """
    if k == 0:
        return torch.zeros(d, 0, dtype=torch.float32, device=device)
    gen = torch.Generator()
    gen.manual_seed(seed)
    Z = torch.randn(d, k, generator=gen, dtype=torch.float32)
    Q, _ = torch.linalg.qr(Z)
    return Q.to(device)


def name_step_seed(name: str, step: int) -> int:
    """Return a deterministic non-negative int seed from (name, step)."""
    key = f"{name}::{step}".encode()
    return int(hashlib.sha256(key).hexdigest()[:8], 16) % (2**31)


__all__ = [
    "apply_curvLoRA",
    "compute_projection_cache",
    "compute_projection_cache_bottom",
    "get_energy_threshold",
    "name_step_seed",
    "random_orthogonal_basis",
]
