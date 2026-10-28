import torch
import torch.nn.functional as F

from model.dense.layer import DenseLayer, DenseVariant


class curvDelta(DenseVariant):
    """Dense variant with bilateral low-curvature projection."""

    @staticmethod
    def init(module: DenseLayer, adapter_name: str, **kwargs) -> None:
        base_weight = module.get_base_layer().weight
        module.register_buffer(
            f"U_in_bar_{adapter_name}",
            base_weight.new_zeros((module.in_features, 0)),
            persistent=False,
        )
        module.register_buffer(
            f"U_out_bar_{adapter_name}",
            base_weight.new_zeros((module.out_features, 0)),
            persistent=False,
        )

    @staticmethod
    def get_delta_weight(
        module: DenseLayer, adapter_name: str, delta: torch.Tensor
    ) -> torch.Tensor:
        U_in_bar = getattr(module, f"U_in_bar_{adapter_name}").to(delta)
        U_out_bar = getattr(module, f"U_out_bar_{adapter_name}").to(delta)

        projected = delta
        if U_out_bar.shape[1] > 0:
            projected = projected - U_out_bar @ (U_out_bar.T @ projected)
        if U_in_bar.shape[1] > 0:
            projected = projected - (projected @ U_in_bar) @ U_in_bar.T
        return projected

    @staticmethod
    def forward(
        module: DenseLayer,
        active_adapter: str,
        x: torch.Tensor,
        result: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        delta = module.dense_delta[active_adapter]
        U_in_bar = getattr(module, f"U_in_bar_{active_adapter}").to(delta)
        U_out_bar = getattr(module, f"U_out_bar_{active_adapter}").to(delta)

        x_projected = x.to(delta.dtype)
        if U_in_bar.shape[1] > 0:
            x_projected = x_projected - (x_projected @ U_in_bar) @ U_in_bar.T

        update = F.linear(x_projected, delta)
        if U_out_bar.shape[1] > 0:
            update = update - (update @ U_out_bar) @ U_out_bar.T

        return result + update.to(result.dtype)
