import os
import sys

from omegaconf import OmegaConf
import pytest
from safetensors.torch import load_file
import torch
import torch.nn.functional as F
from transformers import LlamaConfig, LlamaForCausalLM


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from model import apply_peft  # noqa: E402
from model.curvDelta import get_curvDelta_model  # noqa: E402
from model.curvDelta.layer import curvDelta  # noqa: E402
from model.dense import DenseConfig, DenseLinear, get_dense_model  # noqa: E402


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


def _dense_config():
    return DenseConfig(target_modules=["q_proj"], task_type="CAUSAL_LM")


def _q_proj(peft_model):
    return peft_model.base_model.model.model.layers[0].self_attn.q_proj


def test_dense_linear_zero_init_gradient_and_merge_roundtrip():
    torch.manual_seed(0)
    base = torch.nn.Linear(4, 3, bias=False)
    base_weight = base.weight.detach().clone()
    layer = DenseLinear(base, "default")
    x = torch.randn(2, 4)

    assert torch.equal(layer.dense_delta["default"], torch.zeros_like(base_weight))
    assert torch.allclose(layer(x), F.linear(x, base_weight))

    with torch.no_grad():
        layer.dense_delta["default"].copy_(torch.randn_like(base_weight))
    expected = F.linear(x, base_weight + layer.dense_delta["default"])
    assert torch.allclose(layer(x), expected)

    layer(x).sum().backward()
    assert layer.dense_delta["default"].grad is not None
    assert base.weight.grad is None

    with pytest.raises(NotImplementedError, match="adapter_names"):
        layer(x, adapter_names=["default", "default"])

    layer.merge(safe_merge=True)
    assert torch.allclose(layer(x), expected)
    layer.unmerge()
    assert torch.allclose(base.weight, base_weight)


def test_curvDelta_forward_matches_explicit_projected_delta():
    torch.manual_seed(1)
    base = torch.nn.Linear(4, 3, bias=False)
    layer = DenseLinear(base, "default")
    layer.dense_variant["default"] = curvDelta()
    curvDelta.init(layer, "default")

    with torch.no_grad():
        layer.dense_delta["default"].copy_(torch.randn(3, 4))
        layer._buffers["U_in_bar_default"] = torch.tensor([[1.0], [0.0], [0.0], [0.0]])
        layer._buffers["U_out_bar_default"] = torch.tensor([[0.0], [1.0], [0.0]])

    x = torch.randn(2, 4)
    delta = layer.dense_delta["default"]
    U_in = layer.U_in_bar_default
    U_out = layer.U_out_bar_default
    projected = delta - U_out @ (U_out.T @ delta)
    projected = projected - (projected @ U_in) @ U_in.T
    expected = F.linear(x, base.weight + projected)

    assert torch.allclose(layer(x), expected, atol=1e-6)
    layer.merge(safe_merge=True)
    assert torch.allclose(layer(x), expected, atol=1e-6)
    layer.unmerge()


def test_plain_dense_save_exports_merged_full_model_non_destructively(tmp_path):
    torch.manual_seed(2)
    model = get_dense_model(_tiny_llama(), _dense_config())
    layer = _q_proj(model)
    assert isinstance(layer, DenseLinear)
    assert all(
        ("dense_delta" in name) == parameter.requires_grad
        for name, parameter in model.named_parameters()
    )

    with torch.no_grad():
        layer.dense_delta["default"].normal_(std=0.02)
    inputs = torch.tensor([[1, 2, 3, 4]])
    model.eval()
    expected = model(input_ids=inputs).logits
    original_delta = layer.dense_delta["default"].detach().clone()

    model.save_pretrained(tmp_path)
    assert (tmp_path / "config.json").exists()
    assert not (tmp_path / "adapter_config.json").exists()
    assert not any("dense_delta" in key for key in load_file(tmp_path / "model.safetensors"))
    assert torch.equal(layer.dense_delta["default"], original_delta)
    assert torch.allclose(model(input_ids=inputs).logits, expected, atol=1e-6)

    loaded = LlamaForCausalLM.from_pretrained(tmp_path)
    loaded.eval()
    actual = loaded(input_ids=inputs).logits
    assert torch.allclose(actual, expected, atol=1e-6)


def test_plain_dense_multiple_active_adapters_are_additive():
    torch.manual_seed(4)
    base = torch.nn.Linear(4, 3, bias=False)
    base_weight = base.weight.detach().clone()
    layer = DenseLinear(base, "default")
    layer.update_layer("second")
    with torch.no_grad():
        layer.dense_delta["default"].normal_()
        layer.dense_delta["second"].normal_()
    layer.set_adapter(["default", "second"])

    x = torch.randn(2, 4)
    expected_weight = (
        base_weight + layer.dense_delta["default"] + layer.dense_delta["second"]
    )
    assert torch.allclose(layer(x), F.linear(x, expected_weight), atol=1e-6)


def test_curvDelta_save_exports_merged_full_model_non_destructively(tmp_path):
    torch.manual_seed(3)
    model = get_curvDelta_model(_tiny_llama(), _dense_config())
    layer = _q_proj(model)
    assert isinstance(layer, DenseLinear)
    assert "default" in layer.dense_variant

    with torch.no_grad():
        layer.dense_delta["default"].normal_(std=0.02)
        layer._buffers["U_in_bar_default"] = torch.eye(layer.in_features)[:, :2]
        layer._buffers["U_out_bar_default"] = torch.eye(layer.out_features)[:, :2]

    inputs = torch.tensor([[1, 2, 3, 4]])
    model.eval()
    expected = model(input_ids=inputs).logits
    original_delta = layer.dense_delta["default"].detach().clone()

    model.save_pretrained(tmp_path, state_dict=model.state_dict())
    assert (tmp_path / "config.json").exists()
    assert not (tmp_path / "adapter_config.json").exists()
    assert not any("U_in_bar" in key or "U_out_bar" in key for key in model.state_dict())
    saved_keys = load_file(tmp_path / "model.safetensors")
    assert not any("dense_delta" in key or "U_in_bar" in key or "U_out_bar" in key for key in saved_keys)
    assert "default" in layer.dense_delta
    assert torch.equal(layer.dense_delta["default"], original_delta)
    assert torch.allclose(model(input_ids=inputs).logits, expected, atol=1e-6)

    loaded = LlamaForCausalLM.from_pretrained(tmp_path)
    loaded.eval()
    actual = loaded(input_ids=inputs).logits
    assert torch.allclose(actual, expected, atol=1e-6)


def test_curvDelta_merge_and_unload_returns_plain_model():
    torch.manual_seed(5)
    model = get_curvDelta_model(_tiny_llama(), _dense_config())
    layer = _q_proj(model)
    with torch.no_grad():
        layer.dense_delta["default"].normal_(std=0.02)
        layer._buffers["U_in_bar_default"] = torch.eye(layer.in_features)[:, :1]
        layer._buffers["U_out_bar_default"] = torch.eye(layer.out_features)[:, :1]

    inputs = torch.tensor([[1, 2, 3]])
    model.eval()
    expected = model(input_ids=inputs).logits
    merged = model.merge_and_unload(safe_merge=True)

    assert not any(isinstance(module, DenseLinear) for module in merged.modules())
    merged.eval()
    assert torch.allclose(merged(input_ids=inputs).logits, expected, atol=1e-6)


def test_curvDelta_is_attached_only_to_initial_adapter():
    model = get_curvDelta_model(_tiny_llama(), _dense_config())
    layer = _q_proj(model)
    assert set(layer.dense_variant) == {"default"}


def test_apply_peft_dispatches_curvDelta():
    config = OmegaConf.create(
        {
            "peft_type": "curvDelta",
            "target_modules": ["q_proj"],
            "task_type": "CAUSAL_LM",
        }
    )
    model = apply_peft(_tiny_llama(), config)
    layer = _q_proj(model)
    assert isinstance(layer, DenseLinear)
    assert set(layer.dense_variant) == {"default"}

    model.add_adapter("second", _dense_config())
    assert "second" in layer.dense_delta
    assert set(layer.dense_variant) == {"default"}


def test_dense_rejects_unsupported_configuration():
    with pytest.raises(ValueError, match="bias"):
        DenseConfig(target_modules=["q_proj"], task_type="CAUSAL_LM", bias="all")
    with pytest.raises(ValueError, match="modules_to_save"):
        DenseConfig(
            target_modules=["q_proj"],
            task_type="CAUSAL_LM",
            modules_to_save=["lm_head"],
        )
