from __future__ import annotations

import logging
from typing import Optional

from model.curvDelta.layer import curvDelta
from model.dense import DenseConfig, DenseLinear, get_dense_model


logger = logging.getLogger(__name__)


def _get_active_adapter(module) -> Optional[str]:
    if hasattr(module, "active_adapters") and module.active_adapters:
        return module.active_adapters[0]
    if hasattr(module, "_active_adapter"):
        return module._active_adapter
    return None


def apply_curvDelta(model, adapter_name: Optional[str] = None):
    """Attach curvDelta to the currently active Dense adapter."""

    augmented = 0
    total_dense = 0
    for _, module in model.named_modules():
        if not isinstance(module, DenseLinear):
            continue

        total_dense += 1
        active = adapter_name or _get_active_adapter(module)
        if active is None or active not in module.dense_delta:
            continue
        module.dense_variant[active] = curvDelta()
        curvDelta.init(module, active)
        augmented += 1

    logger.info("curvDelta: augmented %s/%s Dense layers", augmented, total_dense)
    return model


def get_curvDelta_model(
    model,
    config: DenseConfig,
    adapter_name: str = "default",
    low_cpu_mem_usage: bool = False,
):
    peft_model = get_dense_model(
        model,
        config,
        adapter_name=adapter_name,
        low_cpu_mem_usage=low_cpu_mem_usage,
    )
    return apply_curvDelta(peft_model, adapter_name=adapter_name)


__all__ = [
    "curvDelta",
    "apply_curvDelta",
    "get_curvDelta_model",
]
