import importlib.util
from io import BytesIO
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import pytest
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset
from transformers import LlamaConfig, LlamaForCausalLM

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from model.crispedit import (  # noqa: E402
    project_matrix_gradient,
    projection_cache_from_covariances,
    resolve_target_modules,
)
from model.crispedit.optimizer import CrispEditAdamW  # noqa: E402
from model.crispedit.stats import (  # noqa: E402
    build_projection_cache,
    collect_kfac_covariances,
)

# Load trainer.crispedit without triggering trainer/__init__.py (deepspeed).
_CRISPEDIT_PATH = Path(__file__).resolve().parents[1] / "src" / "trainer" / "crispedit.py"
_CRISPEDIT_SPEC = importlib.util.spec_from_file_location(
    "trainer_crispedit_for_test", _CRISPEDIT_PATH
)
_CRISPEDIT_MODULE = importlib.util.module_from_spec(_CRISPEDIT_SPEC)
assert _CRISPEDIT_SPEC.loader is not None
_CRISPEDIT_SPEC.loader.exec_module(_CRISPEDIT_MODULE)

CrispEditTrainerMixin = _CRISPEDIT_MODULE.CrispEditTrainerMixin


# ------------------------------------------------------------------ #
# Helpers                                                              #
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


def _down_proj(model):
    return model.model.layers[0].mlp.down_proj


def _random_ortho(n, seed):
    gen = torch.Generator().manual_seed(seed)
    z = torch.randn(n, n, generator=gen)
    q, _ = torch.linalg.qr(z)
    return q


class _RetainDataset(Dataset):
    def __len__(self):
        return 4

    def __getitem__(self, idx):
        ids = torch.tensor([1, 2, 3, 4, 5, 6])
        labels = torch.tensor([-100, -100, -100, 4, 5, 6])
        return {
            "input_ids": ids,
            "labels": labels,
            "attention_mask": torch.ones(6, dtype=torch.long),
        }


def _collate(features):
    return {
        "input_ids": torch.stack([f["input_ids"] for f in features]),
        "attention_mask": torch.stack([f["attention_mask"] for f in features]),
        "labels": torch.stack([f["labels"] for f in features]),
    }


class _TrainDataset:
    def __init__(self, retain):
        self.retain = retain


class _BaseTrainer:
    def __init__(self, model, dataset, collator, args, accelerator):
        self.model = model
        self.train_dataset = dataset
        self.data_collator = collator
        self.args = args
        self.accelerator = accelerator
        self.optimizer = None
        self.callbacks = []

    def add_callback(self, callback):
        self.callbacks.append(callback)

    def _move_model_to_device(self, model, device):
        model.to(device)


def _crispedit_configs():
    return {
        "layers": [],
        "energy_threshold": 0.9,
        "retain_n_samples": 4,
        "retain_factor_dtype": "float32",
        "collection_batch_size": 2,
        "num_chunks": 1,
        "weight_decay": 0.01,
        "refresh_on_weight_change": False,
        "weight_change_threshold": 0.1,
    }


_PEFT_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def _trainer_args():
    return SimpleNamespace(
        learning_rate=0.01,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        weight_decay=0.0,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
    )


# ------------------------------------------------------------------ #
# Config                                                               #
# ------------------------------------------------------------------ #


def test_crispedit_default_config_has_required_fields():
    path = Path(__file__).resolve().parents[1] / "configs" / "crispedit" / "default.yaml"
    cfg = OmegaConf.load(path)
    cc = cfg.crispedit_configs
    for field in [
        "layers",
        "energy_threshold",
        "retain_n_samples",
        "retain_factor_dtype",
        "collection_batch_size",
        "num_chunks",
        "weight_decay",
        "refresh_on_weight_change",
        "weight_change_threshold",
    ]:
        assert field in cc, f"missing field {field!r}"
    assert cc.energy_threshold == 0.9


def test_peft_crispedit_config_declares_target_set():
    path = Path(__file__).resolve().parents[1] / "configs" / "peft" / "crispedit.yaml"
    cfg = OmegaConf.load(path)
    assert cfg.peft_type == "crispedit"
    assert list(cfg.target_modules) == _PEFT_TARGET_MODULES


# ------------------------------------------------------------------ #
# Target resolution                                                    #
# ------------------------------------------------------------------ #


def test_resolve_target_modules_all_layers():
    model = _tiny_llama()
    resolved = resolve_target_modules(model, ["down_proj"])
    assert list(resolved) == ["model.layers.0.mlp.down_proj"]
    assert resolved["model.layers.0.mlp.down_proj"].weight.shape == (8, 16)


def test_resolve_target_modules_multi_module():
    model = _tiny_llama()
    resolved = resolve_target_modules(model, _PEFT_TARGET_MODULES)
    expected = {
        *(f"model.layers.0.self_attn.{m}" for m in ["q_proj", "k_proj", "v_proj", "o_proj"]),
        *(f"model.layers.0.mlp.{m}" for m in ["gate_proj", "up_proj", "down_proj"]),
    }
    assert set(resolved) == expected


def test_resolve_target_modules_respects_layer_filter():
    config = LlamaConfig(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=32,
        tie_word_embeddings=False,
    )
    model = LlamaForCausalLM(config)
    resolved = resolve_target_modules(model, ["down_proj"], layers=[1])
    assert list(resolved) == ["model.layers.1.mlp.down_proj"]


# ------------------------------------------------------------------ #
# Projection math                                                      #
# ------------------------------------------------------------------ #


def test_projection_cache_low_curvature_mask_direction():
    A = torch.diag(torch.tensor([4.0, 1.0, 0.1]))
    B = torch.diag(torch.tensor([9.0, 1.0]))
    cache = projection_cache_from_covariances(A, B, energy_threshold=0.9)
    M = cache["M"]
    # Sa ascending = [0.1, 1.0, 4.0]; Sb ascending = [1.0, 9.0]
    # Kronecker eigenvalues = outer([0.1,1,4],[1,9]) = [[0.1,0.9],[1,9],[4,36]]
    # top-90% cutoff = 4.0 -> keep entries strictly below it.
    assert M[0, 0].item() is True  # 0.1 (lowest curvature) retained
    assert M[0, 1].item() is True  # 0.9 retained
    assert M[1, 0].item() is True  # 1.0 retained
    assert M[2, 0].item() is False  # 4.0 cut off
    assert M[1, 1].item() is False  # 9.0 cut off
    assert M[2, 1].item() is False  # 36.0 cut off
    assert M.shape == (3, 2)


def test_projection_cache_eigenvectors_are_orthogonal():
    A = torch.diag(torch.tensor([4.0, 1.0, 0.1]))
    B = torch.diag(torch.tensor([9.0, 1.0]))
    cache = projection_cache_from_covariances(A, B, energy_threshold=0.9)
    assert torch.allclose(cache["Ua"].T @ cache["Ua"], torch.eye(3), atol=1e-5)
    assert torch.allclose(cache["Ub"].T @ cache["Ub"], torch.eye(2), atol=1e-5)


def test_projection_cache_degenerate_covariance_is_unconstrained():
    A = torch.zeros(3, 3)
    B = torch.zeros(2, 2)
    cache = projection_cache_from_covariances(A, B, energy_threshold=0.9)
    assert cache["M"].all()


def test_project_matrix_gradient_matches_formula():
    torch.manual_seed(0)
    Ua = _random_ortho(3, 1)
    Ub = _random_ortho(2, 2)
    M = torch.tensor([[True, True], [True, False], [False, False]])
    cache = {"Ua": Ua, "Ub": Ub, "M": M}
    grad = torch.randn(2, 3)
    expected = Ub @ ((Ub.T @ grad @ Ua) * M.T) @ Ua.T
    assert torch.allclose(project_matrix_gradient(grad, cache), expected)
    assert project_matrix_gradient(grad, cache).shape == (2, 3)
    assert project_matrix_gradient(grad, cache).dtype == grad.dtype


# ------------------------------------------------------------------ #
# Optimizer                                                            #
# ------------------------------------------------------------------ #


def test_optimizer_matches_adamw_on_projected_gradient():
    torch.manual_seed(1)
    Ua = _random_ortho(3, 1)
    Ub = _random_ortho(2, 2)
    M = torch.tensor([[True, True], [True, False], [False, False]])
    cache = {"Ua": Ua, "Ub": Ub, "M": M}

    w = torch.nn.Parameter(torch.randn(2, 3))
    w_ref = torch.nn.Parameter(w.detach().clone())
    g = torch.randn(2, 3)
    w.grad = g.clone()
    w_ref.grad = project_matrix_gradient(g, cache).clone()

    opt = CrispEditAdamW(
        [w],
        projection_cache_map={w: cache},
        lr=0.1,
        betas=(0.9, 0.999),
        weight_decay=0.01,
    )
    opt_ref = torch.optim.AdamW([w_ref], lr=0.1, betas=(0.9, 0.999), weight_decay=0.01)
    opt.step()
    opt_ref.step()
    assert torch.allclose(w, w_ref)


def test_optimizer_applies_weight_decay_on_zero_gradient():
    w = torch.nn.Parameter(torch.ones(2, 3))
    opt = CrispEditAdamW([w], projection_cache_map={}, lr=0.1, weight_decay=0.1)
    w.grad = torch.zeros(2, 3)
    opt.step()
    # decoupled weight decay: p -= lr * wd * p = 0.1 * 0.1 * 1.0
    assert torch.allclose(w, torch.full((2, 3), 0.99))


def test_optimizer_leaves_non_target_parameters_ordinary_adamw():
    torch.manual_seed(2)
    Ua = _random_ortho(3, 1)
    Ub = _random_ortho(2, 2)
    M = torch.tensor([[True, False], [False, True], [True, True]])
    cache = {"Ua": Ua, "Ub": Ub, "M": M}

    w_target = torch.nn.Parameter(torch.randn(2, 3))
    w_other = torch.nn.Parameter(torch.randn(2, 3))
    w_other_ref = torch.nn.Parameter(w_other.detach().clone())

    w_target.grad = torch.randn(2, 3)
    w_other.grad = torch.randn(2, 3)
    w_other_ref.grad = w_other.grad.clone()

    opt = CrispEditAdamW(
        [w_target, w_other],
        projection_cache_map={w_target: cache},
        lr=0.1,
        weight_decay=0.0,
    )
    opt_ref = torch.optim.AdamW([w_other_ref], lr=0.1, weight_decay=0.0)
    opt.step()
    opt_ref.step()
    assert torch.allclose(w_other, w_other_ref)


def test_optimizer_reset_cache_projects_exp_avg_without_reset():
    torch.manual_seed(3)
    Ua = _random_ortho(3, 1)
    Ub = _random_ortho(2, 2)
    cache1 = {"Ua": Ua, "Ub": Ub, "M": torch.ones(3, 2, dtype=torch.bool)}

    w = torch.nn.Parameter(torch.randn(2, 3))
    opt = CrispEditAdamW([w], projection_cache_map={w: cache1}, lr=0.1, weight_decay=0.0)
    w.grad = torch.randn(2, 3)
    opt.step()

    state = opt.state[w]
    exp_avg_sq_before = state["exp_avg_sq"].clone()
    exp_avg_before = state["exp_avg"].clone()

    M2 = torch.tensor([[True, False], [False, False], [False, False]])
    cache2 = {"Ua": Ua, "Ub": Ub, "M": M2}
    opt.reset_cache({w: cache2})

    assert torch.allclose(state["exp_avg"], project_matrix_gradient(exp_avg_before, cache2))
    assert torch.equal(state["exp_avg_sq"], exp_avg_sq_before)
    assert "step" in state


def test_optimizer_offload_matches_torch_adamw():
    torch.manual_seed(4)
    w = torch.nn.Parameter(torch.randn(2, 3))
    ref = torch.nn.Parameter(w.detach().clone())
    opt = CrispEditAdamW(
        [w], projection_cache_map={}, lr=0.01, betas=(0.9, 0.999),
        eps=1e-8, weight_decay=0.1, offload_state_to_cpu=True,
    )
    ref_opt = torch.optim.AdamW([ref], lr=0.01, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.1)
    for _ in range(5):
        g = torch.randn(2, 3)
        w.grad = g.clone()
        ref.grad = g.clone()
        opt.step()
        ref_opt.step()
    assert torch.allclose(w, ref, atol=1e-6)
    # fp32 states are paged to CPU.
    assert opt.state[w]["exp_avg"].device.type == "cpu"
    assert opt.state[w]["exp_avg"].dtype == torch.float32


@pytest.mark.parametrize("offload_state_to_cpu", [False, True])
def test_optimizer_state_roundtrip_preserves_projection(offload_state_to_cpu):
    cache = {"Ua": torch.eye(2), "Ub": torch.eye(2), "M": torch.zeros(2, 2, dtype=torch.bool)}
    protected = torch.nn.Parameter(torch.ones(2, 2))
    ordinary = torch.nn.Parameter(torch.ones(2, 2))
    optimizer = CrispEditAdamW(
        [protected, ordinary], projection_cache_map={protected: cache},
        lr=0.1, weight_decay=0.0, offload_state_to_cpu=offload_state_to_cpu,
    )
    protected.grad = torch.ones_like(protected)
    ordinary.grad = torch.ones_like(ordinary)
    optimizer.step()
    checkpoint = BytesIO()
    torch.save(optimizer.state_dict(), checkpoint)
    checkpoint.seek(0)

    resumed_protected = torch.nn.Parameter(protected.detach().clone())
    resumed_ordinary = torch.nn.Parameter(ordinary.detach().clone())
    resumed = CrispEditAdamW(
        [resumed_protected, resumed_ordinary], projection_cache_map={resumed_protected: cache},
        lr=0.1, weight_decay=0.0, offload_state_to_cpu=offload_state_to_cpu,
    )
    resumed.load_state_dict(torch.load(checkpoint, weights_only=True))
    for parameter in [protected, ordinary, resumed_protected, resumed_ordinary]:
        parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    resumed.step()

    assert torch.equal(resumed_protected, torch.ones_like(resumed_protected))
    assert torch.allclose(resumed_ordinary, ordinary)
    assert resumed.state[resumed_protected]["step"].item() == 2
    assert all("projection_cache_map" not in group for group in resumed.state_dict()["param_groups"])


def test_optimizer_reset_cache_keeps_runtime_map_out_of_state_dict():
    weight = torch.nn.Parameter(torch.ones(2, 2))
    optimizer = CrispEditAdamW([weight])
    cache = {"Ua": torch.eye(2), "Ub": torch.eye(2), "M": torch.zeros(2, 2, dtype=torch.bool)}
    optimizer.reset_cache({weight: cache})
    assert all("projection_cache_map" not in group for group in optimizer.state_dict()["param_groups"])


# ------------------------------------------------------------------ #
# Collection                                                           #
# ------------------------------------------------------------------ #


def test_collect_kfac_covariances_counts_valid_tokens_and_restores_state():
    model = _tiny_llama()
    target = {"model.layers.0.mlp.down_proj": _down_proj(model)}
    loader = DataLoader(_RetainDataset(), batch_size=2, collate_fn=_collate)

    result = collect_kfac_covariances(
        model, loader, target, torch.device("cpu"), torch.float32
    )

    name = "model.layers.0.mlp.down_proj"
    A, B, n = result[name]
    assert A.shape == (16, 16)
    assert B.shape == (8, 8)
    # 4 samples x 3 valid shifted tokens = 12
    assert n == 12
    assert torch.isfinite(A).all()
    assert torch.isfinite(B).all()

    # requires_grad state is restored to its pre-collection value.
    assert _down_proj(model).weight.requires_grad


def test_build_projection_cache_maps_target_weight():
    model = _tiny_llama()
    name = "model.layers.0.mlp.down_proj"
    A = torch.eye(16)
    B = torch.eye(8)
    cache = build_projection_cache(model, {name: (A, B, 10)}, 0.9)
    weight = _down_proj(model).weight
    assert set(cache.keys()) == {weight}
    assert set(cache[weight].keys()) == {"Ua", "Ub", "M"}


# ------------------------------------------------------------------ #
# Lifecycle                                                            #
# ------------------------------------------------------------------ #


class _CrispEditTrainer(CrispEditTrainerMixin, _BaseTrainer):
    pass


def test_lifecycle_rejects_deepspeed_before_bootstrap():
    model = _tiny_llama()
    model._crispedit_target_modules = ["down_proj"]
    trainer = _CrispEditTrainer(
        crispedit_configs=_crispedit_configs(),
        model=model,
        dataset=_TrainDataset(_RetainDataset()),
        collator=_collate,
        args=_trainer_args(),
        accelerator=SimpleNamespace(device=torch.device("cpu"), is_main_process=True),
    )
    trainer.is_deepspeed_enabled = True

    with pytest.raises(RuntimeError, match="ordinary DDP or single-process"):
        trainer.bootstrap_crispedit()


def test_lifecycle_bootstrap_freezes_non_target_and_builds_optimizer():
    model = _tiny_llama()
    model._crispedit_target_modules = ["down_proj"]
    dataset = _TrainDataset(_RetainDataset())
    accelerator = SimpleNamespace(device=torch.device("cpu"), is_main_process=True)
    trainer = _CrispEditTrainer(
        crispedit_configs=_crispedit_configs(),
        model=model,
        dataset=dataset,
        collator=_collate,
        args=_trainer_args(),
        accelerator=accelerator,
    )

    # Target module resolved during construction.
    assert "model.layers.0.mlp.down_proj" in trainer._target_modules

    # Optimizer not yet created.
    assert trainer.optimizer is None

    trainer.bootstrap_crispedit()

    # Only target weight is trainable; everything else frozen.
    target_weight = _down_proj(model).weight
    assert target_weight.requires_grad
    for name, p in model.named_parameters():
        if p is target_weight:
            continue
        assert not p.requires_grad, f"{name} should be frozen"

    assert len(trainer._crispedit_projection_cache) == 1

    optimizer = trainer.create_optimizer()
    assert isinstance(optimizer, CrispEditAdamW)
    assert len(optimizer.param_groups[0]["params"]) == 1
    assert optimizer.param_groups[0]["params"][0] is target_weight

    # Idempotent bootstrap.
    trainer.bootstrap_crispedit()
    assert len(trainer._crispedit_projection_cache) == 1


def test_lifecycle_composes_mixin_before_trainer():
    composed = type("GradDiff", (CrispEditTrainerMixin, _BaseTrainer), {})
    model = _tiny_llama()
    model._crispedit_target_modules = ["down_proj"]
    dataset = _TrainDataset(_RetainDataset())
    accelerator = SimpleNamespace(device=torch.device("cpu"), is_main_process=True)
    trainer = composed(
        crispedit_configs=_crispedit_configs(),
        model=model,
        dataset=dataset,
        collator=_collate,
        args=_trainer_args(),
        accelerator=accelerator,
    )
    assert isinstance(trainer, CrispEditTrainerMixin)
    assert isinstance(trainer, _BaseTrainer)


def test_lifecycle_multi_module_targets_are_trainable():
    model = _tiny_llama()
    model._crispedit_target_modules = list(_PEFT_TARGET_MODULES)
    dataset = _TrainDataset(_RetainDataset())
    accelerator = SimpleNamespace(device=torch.device("cpu"), is_main_process=True)
    trainer = _CrispEditTrainer(
        crispedit_configs=_crispedit_configs(),
        model=model,
        dataset=dataset,
        collator=_collate,
        args=_trainer_args(),
        accelerator=accelerator,
    )
    assert len(trainer._target_modules) == 7
    trainer.bootstrap_crispedit()
    assert len(trainer._crispedit_projection_cache) == 7
    target_ids = {id(p) for p in trainer._crispedit_target_weights.values()}
    for name, p in model.named_parameters():
        assert p.requires_grad == (id(p) in target_ids), name
    optimizer = trainer.create_optimizer()
    assert isinstance(optimizer, CrispEditAdamW)
    assert len(optimizer.param_groups[0]["params"]) == 7
