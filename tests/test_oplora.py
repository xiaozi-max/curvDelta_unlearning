import importlib.util
import os
import sys
from pathlib import Path

from peft import LoraConfig, PeftModel, get_peft_model
from safetensors.torch import load_file
import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from model.oplora import apply_oplora, compute_weight_basis  # noqa: E402
from model.oplora.layer import OPLora  # noqa: E402

_OPLORA_TRAINER = Path(__file__).resolve().parents[1] / "src" / "trainer" / "oplora.py"
_SPEC = importlib.util.spec_from_file_location("oplora_callback_under_test", _OPLORA_TRAINER)
_MODULE = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_MODULE)
OPLoraCallback = _MODULE.OPLoraCallback


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


def _wrap(model, k=2):
    config = LoraConfig(
        r=4,
        lora_alpha=8,
        lora_dropout=0.0,
        target_modules=["q_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    return apply_oplora(get_peft_model(model, config), k=k)


def _q_proj(peft_model):
    return peft_model.base_model.model.model.layers[0].self_attn.q_proj


def test_svd_basis_is_top_k_and_orthonormal():
    torch.manual_seed(0)
    W = torch.randn(12, 8)
    U, V = compute_weight_basis(W, k=3)
    assert U.shape == (12, 3) and V.shape == (8, 3)
    assert U.dtype == W.dtype
    assert torch.allclose(U.T @ U, torch.eye(3), atol=1e-4)
    assert torch.allclose(V.T @ V, torch.eye(3), atol=1e-4)
    _exact_U, S, _exact_Vh = torch.linalg.svd(W, full_matrices=False)
    captured = torch.linalg.norm(U.float().T @ W @ V.float())
    assert captured >= S[:3].norm() * 0.95


def test_k_is_clamped_and_nonpositive_k_errors():
    W = torch.randn(6, 4)
    U, V = compute_weight_basis(W, k=100)
    assert U.shape[1] == 4 and V.shape[1] == 4
    try:
        compute_weight_basis(W, k=0)
    except ValueError:
        return
    raise AssertionError("k=0 should raise")


def test_forward_matches_explicit_projection_and_kills_top_singular_vectors():
    torch.manual_seed(0)
    model = _wrap(_tiny_llama(), k=2)
    layer = _q_proj(model)
    model.eval()
    with torch.no_grad():
        layer.lora_A["default"].weight.normal_(std=0.2)
        layer.lora_B["default"].weight.normal_(std=0.2)

    U = layer.oplora_U_default
    V = layer.oplora_V_default
    A = layer.lora_A["default"].weight
    B = layer.lora_B["default"].weight
    scaling = layer.scaling["default"]
    A_eff = A - (A @ V) @ V.T
    B_eff = B - U @ (U.T @ B)

    x = torch.randn(2, 3, layer.in_features)
    base = layer.base_layer(x)
    got = OPLora.forward(layer, "default", x, base)
    expected = base + torch.nn.functional.linear(
        torch.nn.functional.linear(x, A_eff), B_eff
    ) * scaling
    assert torch.allclose(got, expected, atol=1e-5)

    delta = (B_eff @ A_eff) * scaling
    assert torch.allclose(delta @ V, torch.zeros_like(delta @ V), atol=1e-5)
    assert torch.allclose(U.T @ delta, torch.zeros_like(U.T @ delta), atol=1e-5)


def test_callback_writes_projection_into_parameters():
    torch.manual_seed(0)
    model = _wrap(_tiny_llama(), k=2)
    layer = _q_proj(model)
    with torch.no_grad():
        layer.lora_A["default"].weight.normal_(std=0.2)
        layer.lora_B["default"].weight.normal_(std=0.2)
    A_before = layer.lora_A["default"].weight.detach().clone()
    B_before = layer.lora_B["default"].weight.detach().clone()

    OPLoraCallback().on_optimizer_step(None, None, None, model=model)

    A = layer.lora_A["default"].weight
    B = layer.lora_B["default"].weight
    assert not torch.allclose(A, A_before)
    assert not torch.allclose(B, B_before)
    V = layer.oplora_V_default
    U = layer.oplora_U_default
    assert torch.allclose(A @ V, torch.zeros_like(A @ V), atol=1e-5)
    assert torch.allclose(U.T @ B, torch.zeros_like(U.T @ B), atol=1e-5)


def test_sharded_weight_is_rejected():
    model = _wrap(_tiny_llama(), k=2)
    layer = _q_proj(model)
    layer.lora_A["default"].weight.ds_id = 0
    try:
        OPLora.project_lora_weights_(layer, "default")
    except RuntimeError as exc:
        assert "ZeRO-2" in str(exc)
        return
    raise AssertionError("sharded adapter should raise")


def test_projected_adapter_reloads_as_vanilla_lora(tmp_path):
    torch.manual_seed(1)
    base = _tiny_llama()
    base_state = {key: value.detach().clone() for key, value in base.state_dict().items()}
    model = _wrap(base, k=2)
    layer = _q_proj(model)
    model.eval()
    with torch.no_grad():
        layer.lora_A["default"].weight.normal_(std=0.2)
        layer.lora_B["default"].weight.normal_(std=0.2)
    A_before = layer.lora_A["default"].weight.detach().clone()
    OPLora.project_lora_weights_(layer, "default")
    assert not torch.allclose(layer.lora_A["default"].weight, A_before)

    inputs = torch.tensor([[1, 2, 3, 4]])
    expected = model(input_ids=inputs).logits
    model.save_pretrained(tmp_path)

    weight_path = tmp_path / "adapter_model.safetensors"
    if not weight_path.exists():
        weight_path = tmp_path / "adapter_model.bin"
    saved = (
        load_file(weight_path)
        if weight_path.suffix == ".safetensors"
        else torch.load(weight_path, map_location="cpu", weights_only=True)
    )
    assert not any("oplora_U" in key or "oplora_V" in key for key in saved)

    reload_base = _tiny_llama()
    reload_base.load_state_dict(base_state)
    loaded = PeftModel.from_pretrained(reload_base, tmp_path)
    loaded.eval()
    assert torch.allclose(loaded(input_ids=inputs).logits, expected, atol=1e-5)


@pytest.mark.parametrize("safe_merge", [False, True])
def test_merge_and_unload_preserves_effective_forward(safe_merge):
    torch.manual_seed(11)
    model = _wrap(_tiny_llama(), k=2).eval()
    layer = _q_proj(model)
    with torch.no_grad():
        layer.lora_A["default"].weight.normal_(std=0.2)
        layer.lora_B["default"].weight.normal_(std=0.2)
    x = torch.randn(2, 3, layer.in_features)
    expected_layer = layer(x).detach()
    assert not torch.allclose(expected_layer, layer.base_layer(x))
    tokens = torch.tensor([[1, 2, 3, 4]])
    expected_logits = model(input_ids=tokens).logits.detach()
    original_dtype = layer.base_layer.weight.dtype

    merged = model.merge_and_unload(safe_merge=safe_merge).eval()
    merged_layer = merged.model.layers[0].self_attn.q_proj

    assert merged_layer.weight.dtype == original_dtype
    assert torch.allclose(merged_layer(x), expected_layer, atol=1e-6)
    assert torch.allclose(merged(input_ids=tokens).logits, expected_logits, atol=1e-6)


@pytest.mark.parametrize("safe_merge", [False, True])
def test_merge_unmerge_restores_base_and_effective_forward(safe_merge):
    torch.manual_seed(12)
    layer = _q_proj(_wrap(_tiny_llama(), k=2).eval())
    with torch.no_grad():
        layer.lora_A["default"].weight.normal_(std=0.2)
        layer.lora_B["default"].weight.normal_(std=0.2)
    original_weight = layer.base_layer.weight.detach().clone()
    x = torch.randn(2, 3, layer.in_features)
    expected = layer(x).detach()

    layer.merge(safe_merge=safe_merge)
    assert torch.allclose(layer(x), expected, atol=1e-6)
    assert not torch.allclose(layer.base_layer.weight, original_weight)
    layer.unmerge()

    assert not layer.merged
    assert layer.base_layer.weight.dtype == original_weight.dtype
    assert torch.allclose(layer.base_layer.weight, original_weight, atol=1e-6)
    assert torch.allclose(layer(x), expected, atol=1e-6)
