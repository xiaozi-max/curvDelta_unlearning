from __future__ import annotations

import torch
from peft import PeftModelForCausalLM

from peft.tuners.tuners_utils import BaseTuner, BaseTunerLayer, check_target_module_exists
from peft.utils import TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING

from model.dense.layer import DenseLayer, DenseLinear


class DenseModel(BaseTuner):
    """PEFT tuner that replaces selected Linear layers with DenseLinear."""

    prefix = "dense_"
    tuner_layer_cls = DenseLayer
    target_module_mapping = TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            if name == "model":
                raise
            return getattr(self.model, name)

    @staticmethod
    def _check_target_module_exists(config, key):
        return check_target_module_exists(config, key)

    @staticmethod
    def _prepare_adapter_config(config, model_config):
        if config.target_modules is None:
            targets = TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING.get(model_config["model_type"])
            if targets is None:
                raise ValueError("Please specify target_modules for this model architecture.")
            config.target_modules = set(targets)
        return config

    def _mark_only_adapters_as_trainable(self, model):
        for name, parameter in model.named_parameters():
            if self.prefix not in name:
                parameter.requires_grad_(False)

    def enable_adapter_layers(self):
        for module in self.model.modules():
            if isinstance(module, DenseLayer):
                module.enable_adapters(True)

    def disable_adapter_layers(self):
        for module in self.model.modules():
            if isinstance(module, DenseLayer):
                module.enable_adapters(False)

    def set_adapter(self, adapter_name):
        for module in self.model.modules():
            if isinstance(module, DenseLayer):
                if module.merged:
                    module.unmerge()
                module.set_adapter(adapter_name)
        self.active_adapter = adapter_name

    def _replace_module(self, parent, child_name, new_module, child):
        setattr(parent, child_name, new_module)

    def merge_and_unload(self, progressbar=False, safe_merge=False, adapter_names=None):
        for name, module in list(self.model.named_modules()):
            if isinstance(module, DenseLinear):
                module.merge(safe_merge=safe_merge, adapter_names=adapter_names)
                parent_name, _, child_name = name.rpartition(".")
                parent = self.model.get_submodule(parent_name)
                setattr(parent, child_name, module.get_base_layer())
        return self.model

    def _create_and_replace(
        self,
        dense_config,
        adapter_name,
        target,
        target_name,
        parent,
        current_key,
        **optional_kwargs,
    ) -> None:
        if current_key is None:
            raise ValueError("Current key cannot be None.")

        if isinstance(target, DenseLayer):
            target.update_layer(
                adapter_name,
                inference_mode=dense_config.inference_mode,
            )
            return

        new_module = self._create_new_module(
            adapter_name,
            target,
            inference_mode=dense_config.inference_mode,
        )
        if adapter_name != self.active_adapter:
            new_module.requires_grad_(False)
        self._replace_module(parent, target_name, new_module, target)

    @staticmethod
    def _create_new_module(
        adapter_name: str,
        target: torch.nn.Module,
        inference_mode: bool = False,
    ) -> DenseLinear:
        if isinstance(target, BaseTunerLayer):
            target_base_layer = target.get_base_layer()
        else:
            target_base_layer = target

        if not isinstance(target_base_layer, torch.nn.Linear):
            raise ValueError(
                f"Target module {target} is not supported. Dense supports torch.nn.Linear only."
            )
        if isinstance(target, BaseTunerLayer) and not isinstance(target, DenseLayer):
            raise ValueError("Dense cannot be mixed with another PEFT tuner layer.")

        return DenseLinear(
            target,
            adapter_name,
            inference_mode=inference_mode,
        )

    def _cast_adapter_dtype(
        self, adapter_name: str, autocast_adapter_dtype: bool = True
    ) -> None:
        # Dense delta W intentionally follows the base weight dtype.
        return None


class DensePeftModelForCausalLM(PeftModelForCausalLM):
    """Causal-LM PEFT wrapper that saves a merged full model."""

    def _merged_base_state_dict(self, state_dict=None):
        source = self.state_dict() if state_dict is None else state_dict
        dense_modules = {
            name: module
            for name, module in self.base_model.model.named_modules()
            if isinstance(module, DenseLinear)
        }
        module_name = next(iter(dense_modules))
        suffix = f"{module_name}.base_layer.weight"
        root = next(key[: -len(suffix)] for key in source if key.endswith(suffix))

        merged = {}
        for key, value in source.items():
            if not key.startswith(root):
                continue
            plain_key = key[len(root) :]
            if ".dense_delta." in plain_key or ".U_in_bar_" in plain_key or ".U_out_bar_" in plain_key:
                continue
            merged[plain_key.replace(".base_layer.", ".")] = value

        for name, module in dense_modules.items():
            weight = source[f"{root}{name}.base_layer.weight"].clone()
            if not module.merged:
                for adapter_name in module.active_adapters:
                    delta = source[f"{root}{name}.dense_delta.{adapter_name}"]
                    weight += module.get_delta_weight(adapter_name, delta).to(weight.dtype)
            merged[f"{name}.weight"] = weight
        return merged

    def save_pretrained(
        self,
        save_directory,
        safe_serialization: bool = True,
        is_main_process: bool = True,
        state_dict=None,
        **kwargs,
    ):
        return self.base_model.model.save_pretrained(
            save_directory,
            state_dict=self._merged_base_state_dict(state_dict),
            safe_serialization=safe_serialization,
            is_main_process=is_main_process,
            **kwargs,
        )
