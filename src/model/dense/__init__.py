from __future__ import annotations

from model.dense.config import DENSE_PEFT_TYPE, DenseConfig
from model.dense.layer import DenseLayer, DenseLinear, DenseVariant
from model.dense.model import DenseModel, DensePeftModelForCausalLM


def register_dense_peft_method() -> None:
    """Register the repository-local Dense tuner in PEFT's dispatch maps."""

    from peft.mapping import (
        PEFT_TYPE_TO_CONFIG_MAPPING,
        PEFT_TYPE_TO_PREFIX_MAPPING,
        PEFT_TYPE_TO_TUNER_MAPPING,
    )

    entries = (
        (PEFT_TYPE_TO_CONFIG_MAPPING, DENSE_PEFT_TYPE, DenseConfig),
        (PEFT_TYPE_TO_TUNER_MAPPING, DENSE_PEFT_TYPE, DenseModel),
        (PEFT_TYPE_TO_PREFIX_MAPPING, DENSE_PEFT_TYPE, DenseModel.prefix),
    )
    for mapping, key, value in entries:
        existing = mapping.get(key)
        if existing is not None and existing is not value:
            raise RuntimeError(
                f"PEFT mapping {key!r} is already registered to {existing!r}."
            )
        mapping[key] = value


def get_dense_model(
    model,
    config: DenseConfig,
    adapter_name: str = "default",
    low_cpu_mem_usage: bool = False,
):
    register_dense_peft_method()
    return DensePeftModelForCausalLM(
        model,
        config,
        adapter_name=adapter_name,
        autocast_adapter_dtype=False,
        low_cpu_mem_usage=low_cpu_mem_usage,
    )


register_dense_peft_method()


__all__ = [
    "DENSE_PEFT_TYPE",
    "DenseConfig",
    "DenseLayer",
    "DenseLinear",
    "DenseModel",
    "DensePeftModelForCausalLM",
    "DenseVariant",
    "get_dense_model",
    "register_dense_peft_method",
]
