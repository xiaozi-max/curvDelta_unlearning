from __future__ import annotations

import warnings
from typing import Any, Optional

import torch
import torch.nn.functional as F
from torch import nn

from peft.tuners.tuners_utils import BaseTunerLayer, check_adapters_to_merge


class DenseVariant:
    """Extension point for variants that transform a dense adapter update."""

    @staticmethod
    def init(module: "DenseLayer", adapter_name: str, **kwargs) -> None:
        raise NotImplementedError

    @staticmethod
    def get_delta_weight(
        module: "DenseLayer", adapter_name: str, delta: torch.Tensor
    ) -> torch.Tensor:
        raise NotImplementedError

    @staticmethod
    def forward(
        module: "DenseLayer",
        active_adapter: str,
        x: torch.Tensor,
        result: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        raise NotImplementedError


class DenseLayer(BaseTunerLayer):
    """Base class for layers containing full-rank additive adapters."""

    adapter_layer_names = ("dense_delta",)
    other_param_names = ()

    def __init__(self, base_layer: nn.Module, **kwargs) -> None:
        self.base_layer = base_layer
        self.dense_delta = nn.ParameterDict({})
        self.dense_variant: dict[str, DenseVariant] = {}
        self._disable_adapters = False
        self.merged_adapters = []
        self.kwargs = kwargs

        base_layer = self.get_base_layer()
        if not isinstance(base_layer, nn.Linear):
            raise ValueError(
                f"Unsupported layer type {type(base_layer)}. Dense supports torch.nn.Linear only."
            )
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        base_layer.requires_grad_(False)

    def update_layer(
        self,
        adapter_name: str,
        inference_mode: bool = False,
        **kwargs,
    ) -> None:
        base_weight = self.get_base_layer().weight
        self.dense_delta[adapter_name] = nn.Parameter(
            base_weight.new_zeros(base_weight.shape), requires_grad=True
        )
        self._move_adapter_to_device_of_base_layer(adapter_name)
        self.set_adapter(self.active_adapters)
        if inference_mode:
            self.dense_delta[adapter_name].requires_grad_(False)

    def get_delta_weight(
        self, adapter_name: str, delta: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        delta = self.dense_delta[adapter_name] if delta is None else delta
        variant = self.dense_variant.get(adapter_name)
        return variant.get_delta_weight(self, adapter_name, delta) if variant else delta

    def merge(
        self,
        safe_merge: bool = False,
        adapter_names: Optional[list[str]] = None,
    ) -> None:
        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            return

        for active_adapter in adapter_names:
            if active_adapter not in self.dense_delta:
                continue

            base_layer = self.get_base_layer()
            delta = self.get_delta_weight(active_adapter)
            if safe_merge:
                orig_weight = base_layer.weight.data.clone() + delta.to(base_layer.weight.dtype)
                if not torch.isfinite(orig_weight).all():
                    raise ValueError(
                        "NaNs or Infs detected while merging Dense adapter "
                        f"{active_adapter!r}."
                    )
                base_layer.weight.data = orig_weight
            else:
                base_layer.weight.data += delta.to(base_layer.weight.dtype)

            self.merged_adapters.append(active_adapter)

    def unmerge(self) -> None:
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return

        while self.merged_adapters:
            active_adapter = self.merged_adapters.pop()
            if active_adapter not in self.dense_delta:
                continue
            weight = self.get_base_layer().weight
            weight.data -= self.get_delta_weight(active_adapter).to(weight.dtype)

    def _check_forward_args(self, adapter_names) -> None:
        if adapter_names is not None:
            raise NotImplementedError(
                "Dense adapters do not support per-example adapter_names."
            )

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        adapter_names = kwargs.pop("adapter_names", None)
        self._check_forward_args(adapter_names)
        previous_dtype = x.dtype

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            result = self.base_layer(x, *args, **kwargs)
            result_dtype = result.dtype
            for active_adapter in self.active_adapters:
                if active_adapter not in self.dense_delta:
                    continue
                if active_adapter in self.dense_variant:
                    result = self.dense_variant[active_adapter].forward(
                        self,
                        active_adapter=active_adapter,
                        x=x,
                        result=result,
                        **kwargs,
                    )
                else:
                    delta = self.dense_delta[active_adapter]
                    result = result + F.linear(x.to(delta.dtype), delta).to(
                        result_dtype
                    )
            result = result.to(result_dtype)

        return result.to(previous_dtype)

    def supports_lora_conversion(self, adapter_name: str = "default") -> bool:
        return True

    def __repr__(self) -> str:
        return "dense." + super().__repr__()


class DenseLinear(nn.Module, DenseLayer):
    def __init__(
        self,
        base_layer: nn.Linear,
        adapter_name: str,
        inference_mode: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        DenseLayer.__init__(self, base_layer, **kwargs)
        self._active_adapter = adapter_name
        self.update_layer(adapter_name, inference_mode=inference_mode)

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        return DenseLayer.forward(self, x, *args, **kwargs)

    def __repr__(self) -> str:
        return "dense." + nn.Module.__repr__(self)
