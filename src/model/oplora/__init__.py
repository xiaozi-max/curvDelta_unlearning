import logging
from typing import Optional

from peft.tuners.lora.layer import Linear as PeftLoraLinear

from model.oplora.layer import OPLora, compute_weight_basis

logger = logging.getLogger(__name__)


def _replace_buffer(module, name: str, value) -> None:
    if name in module._buffers:
        delattr(module, name)
    module.register_buffer(name, value, persistent=False)


def _get_active_adapter(module) -> Optional[str]:
    if hasattr(module, "active_adapters") and module.active_adapters:
        return module.active_adapters[0]
    if hasattr(module, "_active_adapter"):
        return module._active_adapter
    return None


def apply_oplora(model, *, k: int, niter: int = 4, adapter_name: Optional[str] = None):
    """Attach OPLoRA and cache each target layer's top-k SVD basis."""
    if int(k) <= 0:
        raise ValueError(f"oplora_k must be positive, got {k}")

    augmented = 0
    total_lora = 0
    for module in model.modules():
        if not isinstance(module, PeftLoraLinear):
            continue
        total_lora += 1
        active = adapter_name or _get_active_adapter(module)
        if active is None:
            raise RuntimeError("OPLoRA could not resolve the active LoRA adapter name")
        module.lora_variant[active] = OPLora()
        OPLora.init(module, active)
        base_weight = module.get_base_layer().weight
        U, V = compute_weight_basis(base_weight, k, niter=niter)
        _replace_buffer(module, f"oplora_U_{active}", U)
        _replace_buffer(module, f"oplora_V_{active}", V)
        augmented += 1

    logger.info(
        "OPLoRA: cached SVD bases on %s/%s LoRA layers (requested k=%s)",
        augmented,
        total_lora,
        k,
    )
    model._oplora_k = int(k)
    return model


__all__ = ["OPLora", "apply_oplora", "compute_weight_basis"]
