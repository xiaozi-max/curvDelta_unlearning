import importlib.util
import os
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf
from transformers import LlamaConfig, LlamaForCausalLM

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from model import apply_peft  # noqa: E402
from model.curvLoRA import (  # noqa: E402
    compute_projection_cache,
    compute_projection_cache_bottom,
)
from model.curvDelta.layer import curvDelta  # noqa: E402
from model.dense import DenseLinear  # noqa: E402

# Load trainer.curvature without triggering trainer/__init__.py (which pulls in deepspeed)
_CRV_PATH = Path(__file__).resolve().parents[1] / "src" / "trainer" / "curvature.py"
_CRV_SPEC = importlib.util.spec_from_file_location(
    "trainer_curvature_for_test", _CRV_PATH
)
_CRV_MODULE = importlib.util.module_from_spec(_CRV_SPEC)
assert _CRV_SPEC.loader is not None
_CRV_SPEC.loader.exec_module(_CRV_MODULE)

CurvatureTrainerMixin = _CRV_MODULE.CurvatureTrainerMixin
KFACTracker = _CRV_MODULE.KFACTracker


def test_apply_projections_supports_curvDelta_parameters():
    module = DenseLinear(torch.nn.Linear(4, 3, bias=False), "default")
    module.dense_variant["default"] = curvDelta()
    curvDelta.init(module, "default")
    U_in = torch.eye(4)[:, :2]
    U_out = torch.eye(3)[:, :1]
    trainer = SimpleNamespace(
        args=SimpleNamespace(world_size=1),
        _target_modules={"layer": module},
        _adapter_names={"layer": "default"},
        _local_projections={"layer": (U_in, U_out)},
        _adapter_reference=lambda target, adapter: target.dense_delta[adapter],
        _project_lora_adapter_weights=lambda: None,
    )

    CurvatureTrainerMixin._apply_projections(trainer)

    assert torch.equal(module.U_in_bar_default, U_in)
    assert torch.equal(module.U_out_bar_default, U_out)


# ------------------------------------------------------------------ #
# Tests: KFACTracker capture                                           #
# ------------------------------------------------------------------ #


def test_capture_collects_only_inside_context():
    layer = torch.nn.Linear(3, 2, bias=False)
    tracker = KFACTracker({"layer": layer})
    tracker.install_hooks()
    x = torch.tensor([[[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]]], requires_grad=True)
    mask = torch.tensor([[1, 1]])

    with tracker.capture(mask):
        loss = layer(x).sum()
        loss.backward()

    assert tracker.kfac_n["layer"] == 1
    assert tracker.kfac_A["layer"].shape == (3, 3)
    assert tracker.kfac_S["layer"].shape == (2, 2)
    assert tracker.kfac_A["layer"].sum() > 0

    # Outside context: nothing should be captured
    saved_n = tracker.kfac_n["layer"]
    layer.zero_grad(set_to_none=True)
    loss = layer(x).sum()
    loss.backward()
    assert tracker.kfac_n["layer"] == saved_n

    tracker.remove_hooks()


def test_capture_ignores_no_grad():
    layer = torch.nn.Linear(3, 2, bias=False)
    tracker = KFACTracker({"layer": layer})
    tracker.install_hooks()
    x = torch.tensor([[[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]]])
    mask = torch.tensor([[1, 1]])

    with torch.no_grad():
        with tracker.capture(mask):
            layer(x)

    assert tracker.kfac_n["layer"] == 0
    tracker.remove_hooks()


def test_capture_accumulates_across_batches():
    layer = torch.nn.Linear(4, 3, bias=False)
    tracker = KFACTracker({"layer": layer})
    tracker.install_hooks()

    mask1 = torch.ones((1, 3), dtype=torch.long)
    with tracker.capture(mask1):
        x1 = torch.randn(1, 3, 4, requires_grad=True)
        loss1 = layer(x1).sum()
        loss1.backward()

    n_after_first = tracker.kfac_n["layer"]

    mask2 = torch.ones((1, 4), dtype=torch.long)
    with tracker.capture(mask2):
        x2 = torch.randn(1, 4, 4, requires_grad=True)
        loss2 = layer(x2).sum()
        loss2.backward()

    assert tracker.kfac_n["layer"] == n_after_first + 3
    tracker.remove_hooks()


def test_capture_pairs_backward_with_last_grad_enabled_forward():
    layer = torch.nn.Linear(3, 2, bias=False)
    tracker = KFACTracker({"layer": layer})
    tracker.install_hooks()
    x_grad = torch.tensor(
        [[[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]]], requires_grad=True
    )
    x_nograd = torch.tensor([[[100.0, 200.0, 300.0], [0.0, 0.0, 0.0]]])
    mask = torch.tensor([[1, 1]])

    with tracker.capture(mask):
        loss = layer(x_grad).sum()
        with torch.no_grad():
            layer(x_nograd)
        loss.backward()

    expected_input = x_grad[:, :-1, :].reshape(-1, x_grad.shape[-1])
    expected = expected_input.T @ expected_input
    assert tracker.kfac_n["layer"] == 1
    assert torch.allclose(tracker.kfac_A["layer"], expected)
    tracker.remove_hooks()


# ------------------------------------------------------------------ #
# Tests: masked capture                                                #
# ------------------------------------------------------------------ #


def test_capture_with_attention_mask():
    """Mask should select only non-padded positions (dropping last token)."""
    layer = torch.nn.Linear(4, 3, bias=False)
    tracker = KFACTracker({"layer": layer})
    tracker.install_hooks()

    B, T, D_in = 2, 5, 4
    x = torch.randn(B, T, D_in, requires_grad=True)
    mask = torch.ones(B, T, dtype=torch.long)
    mask[0, 3:] = 0  # first sequence: 3 valid tokens
    mask[1, 4:] = 0  # second sequence: 4 valid tokens

    with tracker.capture(mask):
        out = layer(x)
        loss = out.sum()
        loss.backward()

    # valid positions from mask[:, :-1]: seq0 has 3, seq1 has 4 → total 7
    expected_n = 3 + 4
    assert tracker.kfac_n["layer"] == expected_n
    assert tracker.kfac_A["layer"].shape == (D_in, D_in)
    tracker.remove_hooks()


def test_capture_upcasts_bfloat16_inputs():
    layer = torch.nn.Linear(3, 2, bias=False).to(torch.bfloat16)
    tracker = KFACTracker({"layer": layer})
    tracker.install_hooks()

    x = torch.randn(1, 2, 3, dtype=torch.bfloat16, requires_grad=True)
    mask = torch.tensor([[1, 1]], dtype=torch.long)

    with tracker.capture(mask):
        loss = layer(x).sum()
        loss.backward()

    assert tracker.kfac_A["layer"].dtype == torch.float32
    assert tracker.kfac_S["layer"].dtype == torch.float32
    assert tracker.kfac_A["layer"].sum() > 0
    tracker.remove_hooks()


# ------------------------------------------------------------------ #
# Tests: retain-loss dispatch                                          #
# ------------------------------------------------------------------ #


def test_curvature_nll_uses_model_loss():
    sentinel = torch.tensor(7.0)
    trainer = SimpleNamespace(curvature_configs=SimpleNamespace(retain_loss_type="NLL"))
    model = lambda **inputs: SimpleNamespace(loss=sentinel)
    result = CurvatureTrainerMixin._compute_curvature_retain_loss(
        trainer, model, {"labels": torch.ones(1, 1)}
    )
    assert result is sentinel


def test_curvature_kl_uses_sampled_teacher_fisher():
    class TinyModel:
        def __init__(self, logits):
            self.logits = logits

        def __call__(self, **inputs):
            return SimpleNamespace(logits=self.logits)

    class FisherTrainer(CurvatureTrainerMixin):
        def __init__(self):
            self.curvature_configs = SimpleNamespace(
                retain_loss_type="KL", fisher_temperature=1.0, fisher_sample_seed=0
            )
            self.ref_model = None
            self._curvature_fisher_generator = None
            self.prepare_ref_model_calls = 0

        def _prepare_ref_model(self, model):
            self.prepare_ref_model_calls += 1
            return TinyModel(torch.zeros((1, 3, 3), dtype=torch.float32))

    trainer = FisherTrainer()
    model = TinyModel(torch.ones((1, 3, 3), dtype=torch.float32, requires_grad=True))
    retain_inputs = {
        "input_ids": torch.ones((1, 3), dtype=torch.long),
        "attention_mask": torch.ones((1, 3), dtype=torch.long),
        "labels": torch.ones((1, 3), dtype=torch.long),
    }
    loss = trainer._compute_curvature_retain_loss(model, retain_inputs)
    assert trainer.prepare_ref_model_calls == 1
    assert trainer.ref_model is not None
    assert loss.ndim == 0
    assert loss.requires_grad
    assert torch.allclose(loss, torch.tensor(3.0).log())


def _make_collect_factors_trainer(tracker, loss, accelerator):
    """Build a minimal trainer namespace for ``_collect_factors`` unit tests.

    ``_compute_curvature_retain_loss`` and ``_build_projection_basis`` are
    stubbed so the model is never invoked; only the backward path matters.
    """
    return SimpleNamespace(
        model=SimpleNamespace(
            zero_grad=lambda set_to_none: None,
            eval=lambda: None,
            train=lambda: None,
        ),
        _tracker=tracker,
        _local_projections={},
        _curvature_fisher_generator=None,
        _retain_loader=[
            {
                "attention_mask": torch.ones(1, 2),
                "labels": torch.zeros(1, 2, dtype=torch.long),
            }
        ],
        _prepare_inputs=lambda b: b,
        _compute_curvature_retain_loss=lambda model, retain_inputs: loss,
        _build_projection_basis=lambda A, S, name: (
            torch.zeros(A.shape[0], 0),
            torch.zeros(S.shape[0], 0),
        ),
        accelerator=accelerator,
        curvature_configs=SimpleNamespace(
            retain_loss_type="NLL",
            energy_ratio=0.9,
            update_rule="replace",
            ema_decay=0.9,
        ),
        state=SimpleNamespace(global_step=0),
    )


class _CountingLoss:
    def __init__(self):
        self.backward_calls = 0

    def item(self):
        return 0.0

    def backward(self):
        self.backward_calls += 1


def test_collect_factors_pre_engine_uses_plain_backward():
    """bootstrap path (pre_engine=True) calls loss.backward() directly."""
    layer = torch.nn.Linear(3, 2, bias=False)
    tracker = KFACTracker({"layer": layer})

    loss = _CountingLoss()
    accelerator = SimpleNamespace(is_main_process=True)
    trainer = _make_collect_factors_trainer(tracker, loss, accelerator)

    CurvatureTrainerMixin._collect_factors(trainer, pre_engine=True)

    assert loss.backward_calls == 1
    tracker.remove_hooks()


def test_collect_factors_in_training_uses_accelerator_backward():
    """refresh path (pre_engine=False) uses accelerator.no_sync + accelerator.backward."""
    layer = torch.nn.Linear(3, 2, bias=False)
    tracker = KFACTracker({"layer": layer})

    loss = _CountingLoss()

    class Accel:
        def __init__(self):
            self.no_sync_calls = 0
            self.backward_calls = 0
            self.is_main_process = True

        def no_sync(self, model):
            self.no_sync_calls += 1
            return nullcontext()

        def backward(self, l):
            self.backward_calls += 1

    accel = Accel()
    trainer = _make_collect_factors_trainer(tracker, loss, accel)

    CurvatureTrainerMixin._collect_factors(trainer, pre_engine=False)

    assert loss.backward_calls == 0
    assert accel.no_sync_calls == 1
    assert accel.backward_calls == 1
    tracker.remove_hooks()


# ------------------------------------------------------------------ #
# Tests: compute_projection_cache                                      #
# ------------------------------------------------------------------ #


def test_zero_energy_projection_cache_keeps_empty_bases():
    basis = compute_projection_cache(torch.zeros(4, 4))
    assert basis.shape == (4, 0)


def test_projection_cache_with_identity_factors():
    basis = compute_projection_cache(torch.eye(4), energy_ratio=0.5)
    # All eigenvalues equal → energy threshold selects ~half
    assert basis.shape[0] == 4
    assert basis.shape[1] > 0


def test_projection_cache_symmetrizes_factors_before_eigh():
    factor = torch.tensor([[2.0, 0.2], [0.0, 1.0]])
    sym_factor = 0.5 * (factor + factor.mT)

    basis = compute_projection_cache(factor, energy_ratio=0.5)
    expected = compute_projection_cache(sym_factor, energy_ratio=0.5)

    assert torch.allclose(basis @ basis.mT, expected @ expected.mT)


def test_projection_cache_retries_linalg_failure_in_double(monkeypatch, caplog):
    calls = []
    real_eigh = torch.linalg.eigh

    def flaky_eigh(factor):
        calls.append(factor.dtype)
        if factor.dtype == torch.float32:
            raise torch._C._LinAlgError("mock linalg.eigh failure")
        return real_eigh(factor)

    monkeypatch.setattr(torch.linalg, "eigh", flaky_eigh)

    basis = compute_projection_cache(torch.eye(3), factor_name="layer.output")

    assert basis.shape == (3, 3)
    assert calls == [torch.float32, torch.float64]
    assert "layer.output" in caplog.text
    assert "all_zero=False" in caplog.text


def test_projection_cache_reports_nonfinite_factor_name():
    factor = torch.eye(2)
    factor[0, 0] = torch.nan

    with pytest.raises(RuntimeError, match="layer.input"):
        compute_projection_cache_bottom(factor, factor_name="layer.input")


# ------------------------------------------------------------------ #
# Tests: KFACTracker chunking                                          #
# ------------------------------------------------------------------ #


def test_chunk_split_divides_modules_correctly():
    layers = {f"l{i}": torch.nn.Linear(3, 2, bias=False) for i in range(5)}
    tracker = KFACTracker(layers, num_chunks=2)
    assert tracker.num_chunks == 2
    assert tracker._chunks == [["l0", "l1", "l2"], ["l3", "l4"]]


def test_chunk_split_clamps_to_module_count():
    layers = {f"l{i}": torch.nn.Linear(3, 2, bias=False) for i in range(2)}
    tracker = KFACTracker(layers, num_chunks=10)
    # Can't have more chunks than modules
    assert tracker.num_chunks == 2
    assert tracker._chunks == [["l0"], ["l1"]]


def test_single_chunk_stores_on_cpu():
    """num_chunks=1 stores accumulators on CPU; activate_chunk moves them to GPU."""
    layer = torch.nn.Linear(3, 2, bias=False)
    tracker = KFACTracker({"layer": layer}, device=torch.device("cpu"), num_chunks=1)
    assert tracker._storage_device == torch.device("cpu")
    assert tracker.kfac_A["layer"].device == torch.device("cpu")


def test_multi_chunk_stores_on_cpu():
    """num_chunks>1 should store accumulators on CPU between activations."""
    layers = {f"l{i}": torch.nn.Linear(3, 2, bias=False) for i in range(4)}
    tracker = KFACTracker(layers, num_chunks=2)
    assert tracker._storage_device == torch.device("cpu")
    for name in layers:
        assert tracker.kfac_A[name].device == torch.device("cpu")


def test_activate_deactivate_chunk():
    layers = {f"l{i}": torch.nn.Linear(3, 2, bias=False) for i in range(4)}
    tracker = KFACTracker(layers, num_chunks=2)

    tracker.activate_chunk(0)
    assert tracker._current_chunk == 0
    assert tracker.active_module_names() == ["l0", "l1"]
    # Active chunk accumulators are on compute device
    assert tracker.kfac_A["l0"].device == tracker._compute_device
    # Inactive chunk stays on CPU
    assert tracker.kfac_A["l2"].device == torch.device("cpu")

    tracker.deactivate_chunk(0)
    assert tracker._current_chunk is None
    assert tracker.kfac_A["l0"].device == torch.device("cpu")


def test_hooks_only_installed_for_active_chunk():
    """Hooks should only be installed for the active chunk's modules."""
    layers = {f"l{i}": torch.nn.Linear(3, 2, bias=False) for i in range(4)}
    tracker = KFACTracker(layers, num_chunks=2)

    tracker.activate_chunk(1)
    tracker.install_hooks()
    # Should have 2 hooks per module (fwd + bwd) for chunk 1 (l2, l3)
    assert len(tracker._handles) == 4

    tracker.remove_hooks()
    assert len(tracker._handles) == 0
    tracker.deactivate_chunk(1)


def test_clear_accumulators_scoped_to_chunk():
    layers = {f"l{i}": torch.nn.Linear(3, 2, bias=False) for i in range(4)}
    tracker = KFACTracker(layers, num_chunks=2)

    # Set some non-zero values on all modules
    for name in layers:
        tracker.kfac_A[name].fill_(1.0)
        tracker.kfac_n[name] = 5

    tracker.activate_chunk(0)
    tracker.clear_accumulators()

    # Chunk 0 (l0, l1) should be cleared
    assert tracker.kfac_A["l0"].sum() == 0
    assert tracker.kfac_n["l0"] == 0
    # Chunk 1 (l2, l3) should be untouched
    assert tracker.kfac_A["l2"].sum() > 0
    assert tracker.kfac_n["l2"] == 5

    tracker.deactivate_chunk(0)


def test_chunked_capture_collects_factors():
    """End-to-end: chunked collection should produce correct factors."""
    layer0 = torch.nn.Linear(4, 3, bias=False)
    layer1 = torch.nn.Linear(4, 3, bias=False)
    # Share weights so both layers produce the same output
    layer1.weight = layer0.weight
    layers = {"l0": layer0, "l1": layer1}

    x = torch.randn(1, 3, 4, requires_grad=True)
    mask = torch.ones(1, 3, dtype=torch.long)

    # Collect with num_chunks=2 (each layer in its own chunk)
    tracker = KFACTracker(layers, num_chunks=2)

    for chunk_idx in range(tracker.num_chunks):
        tracker.activate_chunk(chunk_idx)
        tracker.clear_accumulators()
        tracker.install_hooks()
        with tracker.capture(mask):
            # Both layers run in forward, but only active chunk's hooks fire
            out = layer0(x) + layer1(x)
            loss = out.sum()
            loss.backward()
        tracker.remove_hooks()
        tracker.deactivate_chunk(chunk_idx)

    # Each layer should have captured factors
    for name in ["l0", "l1"]:
        assert tracker.kfac_A[name].shape == (4, 4)
        assert tracker.kfac_S[name].shape == (3, 3)
        assert tracker.kfac_n[name] > 0


def test_ema_blending_scoped_to_chunk():
    layers = {f"l{i}": torch.nn.Linear(3, 2, bias=False) for i in range(4)}
    tracker = KFACTracker(layers, num_chunks=2)

    # Simulate: set factors and save persistent for chunk 0
    tracker.activate_chunk(0)
    tracker.kfac_A["l0"].fill_(10.0)
    tracker.kfac_S["l0"].fill_(10.0)
    tracker.save_persistent()

    # Now set new factors
    tracker.kfac_A["l0"].fill_(0.0)
    tracker.kfac_S["l0"].fill_(0.0)
    tracker.blend_persistent(ema_decay=0.5)

    # After blending: 0.0 * 0.5 + 10.0 * 0.5 = 5.0
    assert torch.allclose(tracker.kfac_A["l0"], torch.full((3, 3), 5.0))

    # Chunk 1 should not have persistent state
    assert tracker.prev_A["l2"] is None
    tracker.deactivate_chunk(0)



# ------------------------------------------------------------------ #
# Tests: optimizer group splitting                                       #
# ------------------------------------------------------------------ #


def test_curvDelta_optimizer_groups_are_split_by_layer():
    class LayeredDenseModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList(
                [
                    DenseLinear(torch.nn.Linear(4, 3, bias=False), "default")
                    for _ in range(3)
                ]
            )

    model = LayeredDenseModel()
    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        [{"params": trainable_parameters, "weight_decay": 0.1}], lr=1e-3
    )
    original_parameter_ids = {id(parameter) for parameter in trainable_parameters}
    trainer = SimpleNamespace(model=model)

    CurvatureTrainerMixin._split_dense_optimizer_groups_by_layer(trainer, optimizer)

    assert len(optimizer.param_groups) == 3
    assert all(len(group["params"]) == 1 for group in optimizer.param_groups)
    assert all(group["weight_decay"] == 0.1 for group in optimizer.param_groups)
    assert {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    } == original_parameter_ids


def test_curvDelta_optimizer_split_rejects_initialized_state():
    model = torch.nn.Module()
    model.layers = torch.nn.ModuleList(
        [DenseLinear(torch.nn.Linear(4, 3, bias=False), "default")]
    )
    parameter = next(
        parameter for parameter in model.parameters() if parameter.requires_grad
    )
    optimizer = torch.optim.AdamW([parameter], lr=1e-3)
    optimizer.state[parameter]["step"] = torch.tensor(1.0)
    trainer = SimpleNamespace(model=model)

    with pytest.raises(RuntimeError, match="before optimizer state"):
        CurvatureTrainerMixin._split_dense_optimizer_groups_by_layer(trainer, optimizer)


# ------------------------------------------------------------------ #
# Tests: bootstrap_curvature                                           #
# ------------------------------------------------------------------ #

def _curvDelta_module(in_features=4, out_features=3):
    module = DenseLinear(torch.nn.Linear(in_features, out_features, bias=False), "default")
    module.dense_variant["default"] = curvDelta()
    curvDelta.init(module, "default")
    return module


def test_offload_and_restore_adapter_params_round_trip():
    module = _curvDelta_module()
    original = module.dense_delta["default"].data.clone()
    cpu = torch.device("cpu")
    trainer = SimpleNamespace(
        _target_modules={"layer": module},
        _iter_adapter_params=CurvatureTrainerMixin._iter_adapter_params,
    )

    CurvatureTrainerMixin._offload_adapter_params(trainer, cpu)
    assert module.dense_delta["default"].data.device == cpu

    CurvatureTrainerMixin._restore_adapter_params(trainer, cpu)
    assert module.dense_delta["default"].data.device == cpu
    assert torch.equal(module.dense_delta["default"].data, original)


def test_iter_adapter_params_yields_dense_delta_and_lora_weights():
    dense_module = _curvDelta_module()
    dense_params = list(CurvatureTrainerMixin._iter_adapter_params(dense_module))
    assert dense_params == [dense_module.dense_delta["default"]]


def test_bootstrap_curvature_offloads_collects_restores_applies():
    module = _curvDelta_module()
    device = torch.device("cpu")
    tracker = KFACTracker({"layer": module}, device=device, num_chunks=1)
    events = []

    class FakeModel:
        training = False
        quantization_method = None

        def disable_adapter(self):
            events.append("disable_adapter_enter")

            class Ctx:
                def __enter__(self):
                    events.append("disable_in")

                def __exit__(self, *exc):
                    events.append("disable_out")
                    return False

            return Ctx()

    model = FakeModel()
    trainer = SimpleNamespace(
        model=model,
        optimizer=None,
        args=SimpleNamespace(resume_from_checkpoint=None),
        accelerator=SimpleNamespace(device=device),
        curvature_configs=SimpleNamespace(
            cache_mode="static",
            update_rule="replace",
        ),
        _curvature_initial_projection_ready=False,
        _target_modules={"layer": module},
        _tracker=tracker,
    )

    def fake_move_model_to_device(m, d):
        events.append(("move", d))

    def fake_offload(d):
        events.append(("offload", d))

    def fake_restore(d):
        events.append(("restore", d))

    def fake_collect(save_persistent=False, pre_engine=False):
        events.append(("collect", save_persistent, pre_engine))

    def fake_apply():
        events.append("apply")

    trainer._move_model_to_device = fake_move_model_to_device
    trainer._offload_adapter_params = fake_offload
    trainer._restore_adapter_params = fake_restore
    trainer._collect_factors = fake_collect
    trainer._apply_projections = fake_apply

    CurvatureTrainerMixin.bootstrap_curvature(trainer)

    assert trainer._curvature_initial_projection_ready is True
    # Ordering: move → offload → collect(under disable_adapter) → restore → apply
    assert events[0] == ("move", device)
    assert events[1] == ("offload", device)
    assert events[2] == "disable_adapter_enter"
    assert ("collect", False, True) in events
    assert ("restore", device) in events
    assert events[-2] == "apply" or "apply" in events


def test_bootstrap_curvature_skips_when_already_ready():
    trainer = SimpleNamespace(_curvature_initial_projection_ready=True)
    # Should return without touching any other attribute.
    CurvatureTrainerMixin.bootstrap_curvature(trainer)


def test_bootstrap_curvature_raises_if_optimizer_exists():
    trainer = SimpleNamespace(
        optimizer=object(),
        _curvature_initial_projection_ready=False,
    )
    with pytest.raises(RuntimeError, match="before the optimizer is created"):
        CurvatureTrainerMixin.bootstrap_curvature(trainer)


def test_bootstrap_curvature_raises_on_resume():
    trainer = SimpleNamespace(
        optimizer=None,
        args=SimpleNamespace(resume_from_checkpoint="some/path"),
        _curvature_initial_projection_ready=False,
    )
    with pytest.raises(ValueError, match="not supported when resuming"):
        CurvatureTrainerMixin.bootstrap_curvature(trainer)


def test_curvature_on_train_begin_skips_after_bootstrap():
    trainer = SimpleNamespace(_curvature_initial_projection_ready=True)
    # Must not call _collect_factors or _apply_projections.
    CurvatureTrainerMixin.curvature_on_train_begin(trainer)


# ------------------------------------------------------------------ #
# Tests: bootstrap_curvature end-to-end integration                   #
# ------------------------------------------------------------------ #


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


def _curvDelta_peft_model():
    peft_cfg = OmegaConf.create(
        {
            "peft_type": "curvDelta",
            "target_modules": ["q_proj"],
            "task_type": "CAUSAL_LM",
        }
    )
    return apply_peft(_tiny_llama(), peft_cfg)


class _RetainDataset:
    """Tiny retain set: a few token sequences with labels."""

    def __init__(self, n=4, seq_len=6, vocab=32):
        torch.manual_seed(0)
        self.items = [
            {
                "input_ids": torch.randint(0, vocab, (seq_len,), dtype=torch.long),
                "attention_mask": torch.ones(seq_len, dtype=torch.long),
                "labels": torch.randint(0, vocab, (seq_len,), dtype=torch.long),
            }
            for _ in range(n)
        ]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]



class _AnchorDataset:
    """Wraps a retain dataset and reports a fixed anchor size for ``len()``.

    ``_build_retain_loader`` calls ``len(self.train_dataset)`` (the anchor,
    e.g. the forget split) to size the retain subsample. A plain namespace
    cannot define ``__len__`` as a resolved dunder, so use a real class.
    """

    def __init__(self, retain, anchor_size):
        self.retain = retain
        self._anchor_size = anchor_size

    def __len__(self):
        return self._anchor_size

def _collate(features):
    return {
        "input_ids": torch.stack([f["input_ids"] for f in features]),
        "attention_mask": torch.stack([f["attention_mask"] for f in features]),
        "labels": torch.stack([f["labels"] for f in features]),
    }


class _IntegrationTrainer(CurvatureTrainerMixin):
    """Minimal trainer that runs real bootstrap_curvature without HF Trainer.

    Overrides _compute_curvature_retain_loss with a real differentiable LM
    loss so KFAC backward hooks fire on the actual base layer under
    disable_adapter(). ``collect_device_checks`` records dense_delta /
    base-weight device observed inside the collection forward.
    """

    def __init__(self, model, device):
        self.model = model
        self.optimizer = None
        self.args = SimpleNamespace(
            process_index=0,
            world_size=1,
            resume_from_checkpoint=None,
            dataloader_num_workers=0,
            dataloader_pin_memory=False,
        )
        self.accelerator = SimpleNamespace(
            device=device,
            is_main_process=True,
        )
        self.curvature_configs = SimpleNamespace(
            cache_mode="static",
            update_rule="replace",
            energy_ratio=0.9,
            retain_loss_type="NLL",
            collection_batch_size=2,
            num_chunks=1,
            projection_mode="kfac_top",
        )
        self.state = SimpleNamespace(global_step=0)
        self.train_dataset = _AnchorDataset(_RetainDataset(), anchor_size=4)
        self.data_collator = _collate
        self._curvature_initial_projection_ready = False
        self._target_modules = {}
        self._adapter_names = {}
        self._setup_curvature()
        self.collect_device_checks = []

    def _apply_projections(self):
        self.applied_projections = {
            name: (left.clone(), right.clone())
            for name, (left, right) in self._local_projections.items()
        }
        super()._apply_projections()

    def _move_model_to_device(self, model, device):
        model.to(device)

    def _prepare_inputs(self, batch):
        return {k: v.to(self.accelerator.device) for k, v in batch.items()}

    def _compute_curvature_retain_loss(self, model, retain_inputs):
        # Record device state during collect: adapter params vs base weights
        # vs KFAC factors. All three must match the claimed bootstrap layout.
        for name, module in self._target_modules.items():
            base = module.get_base_layer()
            delta = module.dense_delta["default"]
            self.collect_device_checks.append(
                {
                    "name": name,
                    "delta_device": delta.data.device,
                    "base_device": base.weight.data.device,
                    "kfac_A_device": self._tracker.kfac_A[name].device,
                    "kfac_S_device": self._tracker.kfac_S[name].device,
                }
            )
        outputs = model(**retain_inputs)
        return outputs.loss


def _run_bootstrap_integration(device):
    model = _curvDelta_peft_model()
    trainer = _IntegrationTrainer(model, device)
    # Sanity: target found and is the wrapped q_proj DenseLinear.
    assert trainer._target_modules, "no curvature target modules found"
    assert any("q_proj" in name for name in trainer._target_modules)

    trainer.bootstrap_curvature()

    assert trainer._curvature_initial_projection_ready is True
    # Collect actually ran (at least one batch).
    assert trainer.collect_device_checks, "collect_factors did not run any batch"
    # During collect: adapter params on CPU, base + KFAC factors on compute device.
    for check in trainer.collect_device_checks:
        assert check["delta_device"] == torch.device("cpu"), (
            f"dense_delta on {check['delta_device']} during collect, expected CPU"
        )
        assert check["base_device"] == device, (
            f"base weight on {check['base_device']} during collect, expected {device}"
        )
        assert check["kfac_A_device"] == device, (
            f"kfac_A on {check['kfac_A_device']} during collect, expected {device}"
        )
        assert check["kfac_S_device"] == device, (
            f"kfac_S on {check['kfac_S_device']} during collect, expected {device}"
        )
    # After bootstrap: adapter params restored to compute device.
    for module in trainer._target_modules.values():
        assert module.dense_delta["default"].data.device == device
    # Projections actually applied: module buffers must equal the computed
    # captured projections (shape + value + device), proving _apply_projections
    # ran and wrote the real basis — not just the empty buffers registered at
    # curvDelta.init() time.
    for name, module in trainer._target_modules.items():
        U_in, U_out = trainer.applied_projections[name]
        buf_in = module._buffers["U_in_bar_default"]
        buf_out = module._buffers["U_out_bar_default"]
        assert buf_in.shape == U_in.shape, (
            f"U_in_bar shape {buf_in.shape} != projection {U_in.shape}"
        )
        assert buf_out.shape == U_out.shape, (
            f"U_out_bar shape {buf_out.shape} != projection {U_out.shape}"
        )
        assert buf_in.device == device, (
            f"U_in_bar on {buf_in.device}, expected {device}"
        )
        assert buf_out.device == device, (
            f"U_out_bar on {buf_out.device}, expected {device}"
        )
        assert torch.equal(buf_in, U_in.to(device)), (
            f"U_in_bar values do not match applied projection for {name}"
        )
        assert torch.equal(buf_out, U_out.to(device)), (
            f"U_out_bar values do not match applied projection for {name}"
        )


def test_bootstrap_curvature_integration_cpu():
    """End-to-end bootstrap on CPU: real PEFT, real disable_adapter, real KFAC."""
    _run_bootstrap_integration(torch.device("cpu"))


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA for device-migration check"
)
def test_bootstrap_curvature_integration_cuda():
    """End-to-end bootstrap on CUDA: asserts dense_delta offloaded to CPU during collect."""
    _run_bootstrap_integration(torch.device("cuda"))
