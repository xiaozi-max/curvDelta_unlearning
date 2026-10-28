"""Write OPLoRA's complement projection into the live LoRA parameters."""

from transformers import TrainerCallback

from model.oplora.layer import OPLora


def project_oplora_adapters(model) -> None:
    """Fold ``P_L`` / ``P_R`` into every OPLoRA adapter on ``model``."""
    for module in model.modules():
        variants = getattr(module, "lora_variant", None)
        if not variants:
            continue
        for adapter_name, variant in variants.items():
            if isinstance(variant, OPLora) and hasattr(module, "lora_A"):
                OPLora.project_lora_weights_(module, adapter_name)


class OPLoraCallback(TrainerCallback):
    """Project after the optimizer step so AdamW and the checkpoint see ``A_eff`` / ``B_eff``."""

    def on_optimizer_step(self, args, state, control, **kwargs):
        project_oplora_adapters(kwargs["model"])

    def on_train_end(self, args, state, control, **kwargs):
        project_oplora_adapters(kwargs["model"])
