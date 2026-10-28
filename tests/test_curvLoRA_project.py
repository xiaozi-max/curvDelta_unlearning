import importlib.util
import os
import sys
from pathlib import Path

from peft import LoraConfig, PeftModel, get_peft_model
from safetensors.torch import load_file
import torch
from transformers import LlamaConfig, LlamaForCausalLM


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from model.curvLoRA import apply_curvLoRA  # noqa: E402
from model.curvLoRA.layer import curvLoRA  # noqa: E402

_CRV_PATH = Path(__file__).resolve().parents[1] / "src" / "trainer" / "curvature.py"
_CRV_SPEC = importlib.util.spec_from_file_location(
    "trainer_curvature_for_lora_project_test", _CRV_PATH
)
_CRV_MODULE = importlib.util.module_from_spec(_CRV_SPEC)
assert _CRV_SPEC.loader is not None
_CRV_SPEC.loader.exec_module(_CRV_MODULE)
CurvatureTrainerMixin = _CRV_MODULE.CurvatureTrainerMixin


def _tiny_llama():
    config = LlamaConfig(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=32,
        tie_word_embeddings=False,
    )
    return LlamaForCausalLM(config)


def _wrap_curvLoRA(model):
    config = LoraConfig(
        r=4,
        lora_alpha=8,
        lora_dropout=0.0,
        target_modules=["q_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    return apply_curvLoRA(get_peft_model(model, config))


def _q_proj(peft_model):
    return peft_model.base_model.model.model.layers[0].self_attn.q_proj


def _fill_adapter_and_projections(layer, *, k_in=2, k_out=2):
    with torch.no_grad():
        layer.lora_A["default"].weight.normal_(std=0.05)
        layer.lora_B["default"].weight.normal_(std=0.05)
        if k_in > 0:
            layer._buffers["U_in_bar_default"] = torch.eye(
                layer.in_features, dtype=layer.lora_A["default"].weight.dtype
            )[:, :k_in]
        if k_out > 0:
            layer._buffers["U_out_bar_default"] = torch.eye(
                layer.out_features, dtype=layer.lora_B["default"].weight.dtype
            )[:, :k_out]


def test_project_lora_weights_writes_a_eff_b_eff_into_parameters():
    torch.manual_seed(0)
    model = _wrap_curvLoRA(_tiny_llama())
    layer = _q_proj(model)
    _fill_adapter_and_projections(layer)
    model.eval()

    A_before = layer.lora_A["default"].weight.detach().clone()
    B_before = layer.lora_B["default"].weight.detach().clone()
    A_eff, B_eff = curvLoRA.folded_lora_weights(layer, "default")
    assert not torch.equal(A_eff, A_before)
    assert not torch.equal(B_eff, B_before)

    inputs = torch.tensor([[1, 2, 3, 4]])
    expected = model(input_ids=inputs).logits

    curvLoRA.project_lora_weights_(layer, "default")
    assert torch.allclose(layer.lora_A["default"].weight, A_eff, atol=1e-6)
    assert torch.allclose(layer.lora_B["default"].weight, B_eff, atol=1e-6)
    assert torch.allclose(model(input_ids=inputs).logits, expected, atol=1e-5)

    x = torch.randn(2, layer.in_features)
    base = layer.base_layer(x)
    projected_fwd = curvLoRA.forward(layer, "default", x, base)
    plain = base + (
        torch.nn.functional.linear(
            torch.nn.functional.linear(x, layer.lora_A["default"].weight),
            layer.lora_B["default"].weight,
        )
        * layer.scaling["default"]
    )
    assert torch.allclose(projected_fwd, plain, atol=1e-5)


def test_inplace_project_lets_vanilla_peft_save_match_training(tmp_path):
    torch.manual_seed(1)
    base = _tiny_llama()
    base_state = {key: value.detach().clone() for key, value in base.state_dict().items()}
    model = _wrap_curvLoRA(base)
    layer = _q_proj(model)
    _fill_adapter_and_projections(layer)
    model.eval()

    inputs = torch.tensor([[1, 2, 3, 4]])
    expected = model(input_ids=inputs).logits
    curvLoRA.project_lora_weights_(layer, "default")
    A_saved = layer.lora_A["default"].weight.detach().clone()
    B_saved = layer.lora_B["default"].weight.detach().clone()

    model.save_pretrained(tmp_path)
    assert torch.equal(layer.lora_A["default"].weight, A_saved)
    assert torch.equal(layer.lora_B["default"].weight, B_saved)

    weight_path = tmp_path / "adapter_model.safetensors"
    if not weight_path.exists():
        weight_path = tmp_path / "adapter_model.bin"
    saved = (
        load_file(weight_path)
        if weight_path.suffix == ".safetensors"
        else torch.load(weight_path, map_location="cpu")
    )
    assert not any("U_in_bar" in key or "U_out_bar" in key for key in saved)
    a_key = next(key for key in saved if "lora_A" in key and key.endswith(".weight"))
    b_key = next(key for key in saved if "lora_B" in key and key.endswith(".weight"))
    assert torch.allclose(saved[a_key].cpu(), A_saved.cpu(), atol=1e-6)
    assert torch.allclose(saved[b_key].cpu(), B_saved.cpu(), atol=1e-6)

    reload_base = _tiny_llama()
    reload_base.load_state_dict(base_state)
    loaded = PeftModel.from_pretrained(reload_base, tmp_path)
    loaded.eval()
    assert torch.allclose(loaded(input_ids=inputs).logits, expected, atol=1e-5)


def test_apply_projections_retracts_lora_weights():
    torch.manual_seed(2)
    model = _wrap_curvLoRA(_tiny_llama())
    layer = _q_proj(model)
    with torch.no_grad():
        layer.lora_A["default"].weight.normal_(std=0.05)
        layer.lora_B["default"].weight.normal_(std=0.05)
    U_in = torch.eye(layer.in_features)[:, :2]
    U_out = torch.eye(layer.out_features)[:, :1]
    A_before = layer.lora_A["default"].weight.detach().clone()

    mixin = CurvatureTrainerMixin
    trainer = type("Trainer", (), {})()
    trainer.args = type("Args", (), {"world_size": 1})()
    trainer._target_modules = {"q_proj": layer}
    trainer._adapter_names = {"q_proj": "default"}
    trainer._local_projections = {"q_proj": (U_in, U_out)}
    trainer._adapter_reference = mixin._adapter_reference
    trainer._project_lora_adapter_weights = mixin._project_lora_adapter_weights.__get__(
        trainer
    )
    mixin._apply_projections(trainer)

    assert torch.equal(layer.U_in_bar_default, U_in.to(layer.U_in_bar_default))
    assert not torch.equal(layer.lora_A["default"].weight, A_before)
    A_eff, B_eff = curvLoRA.folded_lora_weights(layer, "default")
    assert torch.allclose(layer.lora_A["default"].weight, A_eff)
    assert torch.allclose(layer.lora_B["default"].weight, B_eff)
