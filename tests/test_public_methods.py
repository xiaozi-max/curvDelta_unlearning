"""Public release names and adapter/trainer dispatch contracts."""

from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

from omegaconf import OmegaConf
import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


@pytest.fixture
def trainer_module(monkeypatch):
    # The inherited RMU implementation requires DeepSpeed even for CPU-only imports.
    rmu = ModuleType("trainer.unlearn.rmu")
    rmu.RMU = type("RMU", (), {})
    monkeypatch.setitem(sys.modules, "trainer.unlearn.rmu", rmu)
    import trainer
    return trainer


def tiny_model():
    return LlamaForCausalLM(LlamaConfig(
        vocab_size=32, hidden_size=8, intermediate_size=16,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
    ))


@pytest.mark.parametrize("method,variant", [
    ("curvLoRA", "curvLoRA"), ("curvDelta", "curvDelta"),
    ("lora", None), ("dense", None), ("oplora", "OPLora"),
])
def test_public_adapter_dispatch(method, variant):
    import model
    assert hasattr(model, "apply_peft"), "Public adapter dispatch is missing"
    cfg = OmegaConf.load(ROOT / "configs" / "peft" / f"{method}.yaml")
    cfg.target_modules = ["q_proj"]
    if "r" in cfg:
        cfg.r, cfg.lora_alpha = 2, 4
    if "oplora_k" in cfg:
        cfg.oplora_k = 2
    adapted = model.apply_peft(tiny_model(), cfg)
    layer = adapted.base_model.model.model.layers[0].self_attn.q_proj
    variants = getattr(layer, "lora_variant", getattr(layer, "dense_variant", {}))
    assert (type(variants["default"]).__name__ if variants else None) == variant
    output = adapted(input_ids=torch.tensor([[1, 2, 3]]), labels=torch.tensor([[1, 2, 3]]))
    assert torch.isfinite(output.loss)
    output.loss.backward()


def test_crispedit_dispatch_keeps_base_model():
    import model
    assert hasattr(model, "apply_peft"), "Public adapter dispatch is missing"
    base = tiny_model()
    adapted = model.apply_peft(base, OmegaConf.load(ROOT / "configs/peft/crispedit.yaml"))
    assert adapted is base
    assert "q_proj" in base._crispedit_target_modules


@pytest.mark.parametrize("method", ["curvLoRA", "curvDelta"])
def test_public_method_config_names(method):
    path = ROOT / "configs" / method / "default.yaml"
    assert path.exists(), f"Missing public config group: {method}"
    assert f"{method}_configs" in OmegaConf.load(path)
    for entry in ("train.yaml", "unlearn.yaml"):
        assert any(f"/{method}" in str(item) for item in OmegaConf.load(ROOT / "configs" / entry).defaults)


@pytest.mark.parametrize("method", ["curvLoRA", "curvDelta"])
def test_public_trainer_dispatch(method, monkeypatch, trainer_module):
    trainer = trainer_module
    from trainer.curvature import CurvatureTrainerMixin

    class RecordingTrainer:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.callbacks = []

        def add_callback(self, callback):
            self.callbacks.append(callback)

    monkeypatch.setitem(trainer.TRAINER_REGISTRY, "RecordingTrainer", RecordingTrainer)
    monkeypatch.setattr(trainer, "load_trainer_args", lambda *args: SimpleNamespace(output_dir="unused"))
    monkeypatch.setattr(CurvatureTrainerMixin, "_setup_curvature", lambda self: None)
    cfg = OmegaConf.create({"handler": "RecordingTrainer", "args": {},
                            "method_args": {f"{method}_configs": {"cache_mode": "static"}}})
    loaded, _ = trainer.load_trainer(cfg, model=SimpleNamespace())
    assert isinstance(loaded, CurvatureTrainerMixin)
    assert loaded.curvature_configs.cache_mode == "static"
    assert loaded.callbacks


def test_public_trainer_rejects_mixed_parameterizations(monkeypatch, trainer_module):
    trainer = trainer_module
    monkeypatch.setattr(trainer, "load_trainer_args", lambda *args: SimpleNamespace(output_dir="unused"))
    cfg = OmegaConf.create({"handler": "GradDiff", "args": {}, "method_args": {
        "curvLoRA_configs": {}, "crispedit_configs": {}}})
    with pytest.raises(ValueError, match="cannot be combined"):
        trainer.load_trainer(cfg, model=SimpleNamespace())


@pytest.mark.parametrize("backend,callback_name", [
    ("crispedit", "CrispEditCallback"), ("oplora", "OPLoraCallback"),
])
def test_baseline_trainer_dispatch(backend, callback_name, monkeypatch, trainer_module):
    from trainer.crispedit import CrispEditTrainerMixin
    trainer = trainer_module

    class RecordingTrainer:
        def __init__(self, **kwargs):
            self.callbacks = []

        def add_callback(self, callback):
            self.callbacks.append(callback)

    monkeypatch.setitem(trainer.TRAINER_REGISTRY, "RecordingTrainer", RecordingTrainer)
    monkeypatch.setattr(trainer, "load_trainer_args", lambda *args: SimpleNamespace(output_dir="unused"))
    monkeypatch.setattr(CrispEditTrainerMixin, "_setup_crispedit", lambda self: None)
    cfg = OmegaConf.create({"handler": "RecordingTrainer", "args": {}, "method_args":
                            {"crispedit_configs": {}} if backend == "crispedit" else {}})
    model = SimpleNamespace(_oplora_k=2) if backend == "oplora" else SimpleNamespace()
    loaded, _ = trainer.load_trainer(cfg, model=model)
    assert any(type(callback).__name__ == callback_name for callback in loaded.callbacks)


def test_vila_reference_uses_original_checkpoint(monkeypatch, trainer_module):
    from trainer.unlearn.grad_diff import GradDiff
    from transformers import AutoModelForCausalLM
    original = tiny_model()
    loaded = []

    def from_pretrained(path, **kwargs):
        loaded.append(path)
        return original

    monkeypatch.setattr(AutoModelForCausalLM, "from_pretrained", from_pretrained)
    owner = SimpleNamespace(
        _ref_model_path="original-checkpoint", is_deepspeed_enabled=False,
        accelerator=SimpleNamespace(device=torch.device("cpu"), prepare_model=lambda model, **kwargs: model),
    )
    reference = GradDiff._prepare_ref_model(owner, tiny_model())
    assert loaded == ["original-checkpoint"]
    assert reference is original
    assert not reference.training
    assert all(not parameter.requires_grad for parameter in reference.parameters())


@pytest.mark.parametrize("method", ["curvLoRA", "curvDelta"])
def test_public_hydra_config_composes(method):
    from hydra import compose, initialize_config_dir
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "configs")):
        cfg = compose(config_name="unlearn", overrides=[f"peft={method}", f"{method}=default", "task_name=smoke"])
    assert cfg.peft.peft_type == method
    assert cfg.trainer.method_args[f"{method}_configs"].energy_ratio == 0.9


def test_vila_initialization_preserves_model_output():
    from model import apply_peft
    from model.lora_init import apply_vila_init
    cfg = OmegaConf.create({"peft_type": "lora", "r": 2, "lora_alpha": 4,
                           "lora_dropout": 0.0, "target_modules": ["q_proj"], "task_type": "CAUSAL_LM"})
    adapted = apply_peft(tiny_model(), cfg).eval()
    importance = {name: torch.ones_like(param) for name, param in adapted.named_parameters() if "lora_" in name}
    tokens = torch.tensor([[1, 2, 3]])
    expected = adapted(input_ids=tokens).logits.detach()
    apply_vila_init(adapted, {"f_cnt": 1, "r_cnt": 1, "importance_f": importance, "importance_r": importance})
    assert torch.allclose(adapted(input_ids=tokens).logits, expected, atol=1e-6)


def test_public_curvLoRA_config_excludes_unsupported_trust_region():
    cfg = OmegaConf.load(ROOT / "configs/curvLoRA/default.yaml")
    assert "relative_trust_region" not in cfg.curvLoRA_configs
