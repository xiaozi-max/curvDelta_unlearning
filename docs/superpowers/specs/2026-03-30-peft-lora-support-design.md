# PEFT LoRA Support for Unlearning

## Problem

The repository currently supports full-model training for all unlearning methods. There is no parameter-efficient fine-tuning (PEFT) support. Adding LoRA support enables:

- Significantly reduced memory usage during unlearning (only adapter weights are trained)
- Faster training (fewer parameters to update)
- Smaller saved artifacts (adapter-only saves are ~50MB vs ~14GB for 7B models)
- A foundation for future LoRA variant research

## Scope

- **In scope**: PEFT LoRA for all 11 unlearning methods, DeepSpeed compatibility, adapter and merged model saving, extensible architecture for future LoRA variants
- **Out of scope**: QLoRA, Prefix Tuning, IA3, other PEFT methods; finetuning mode LoRA; loading pre-trained adapters for continued unlearning

## Approach: Model-Layer PEFT Integration + Adapter-Aware Training

PEFT wrapping happens at model loading time (`get_model()`). The trainer layer gains adapter-aware reference model handling and save logic. Configuration is opt-in via Hydra defaults.

---

## 1. Model Loading & PEFT Wrapping

**File**: `src/model/__init__.py`

### Changes

Add optional PEFT wrapping to `get_model()`:

1. After loading the base model and tokenizer, check for `peft_args` in config.
2. If present, call new `apply_peft(model, peft_args)` helper:
   - Constructs `peft.LoraConfig` from the args dict.
   - Calls `peft.get_peft_model(model, lora_config)`.
   - Calls `model.enable_input_require_grads()` for gradient checkpointing compatibility.
   - Returns the PEFT-wrapped model.
3. If `peft_args` is absent, behavior is unchanged (backward compatible).

Add utility function `is_peft_model(model)`:
- Returns `isinstance(model, peft.PeftModel)` (with import guard).
- Used by trainer and evaluator to detect PEFT models.

### Extensibility

Future LoRA variants can:
- Register new model wrapper functions via a PEFT registry or by replacing `apply_peft()`.
- Use the same config pattern (`configs/peft/variant.yaml`).

---

## 2. Reference Model Optimization

**File**: `src/trainer/unlearn/grad_diff.py` (core), affects all methods using `_prepare_ref_model()`

### Key Insight

In LoRA training, base model weights are frozen. The reference model (representing pre-unlearning behavior) is simply the base model with adapters disabled. No `deepcopy` is needed.

### Changes

Override `_prepare_ref_model()` in `GradDiff`:

```python
def _prepare_ref_model(self, model):
    if is_peft_model(model):
        return model  # ref_model IS the model; use adapter disable for ref behavior
    # Original deepcopy path for non-LoRA
    ref_model = copy.deepcopy(model).to(self.accelerator.device)
    ref_model.eval()
    if self.is_deepspeed_enabled:
        ref_model = self._prepare_deepspeed(ref_model)
    else:
        ref_model = self.accelerator.prepare_model(ref_model, evaluation_mode=True)
    return ref_model
```

Add `_disable_adapter_context()` to `GradDiff` (or `UnlearnTrainer`):

```python
def _disable_adapter_context(self, model):
    if is_peft_model(model) and model is self.model:
        return model.disable_adapter()
    return contextlib.nullcontext()
```

All `compute_loss` methods that call `ref_model` wrap those calls:

```python
with self._disable_adapter_context(self.ref_model):
    ref_outputs = self.ref_model(**inputs)
```

### Memory Savings

For a 7B model: ~14GB saved (no full model deepcopy).

### Affected Methods

All methods inheriting `_prepare_ref_model()`: GradDiff, DPO, NPO, SimNPO, UNDIAL, WGA, SatImp, PDU, RMU.

GradAscent and CEU don't use reference models — no changes needed.

---

## 3. Model Saving & Loading

**Files**: `src/trainer/base.py`, `src/train.py`

### Save Strategy

New config option `save_peft_merge` (boolean, default `false`) in `trainer.args`:

- `false`: Save LoRA adapter only via `model.save_pretrained()`. Produces adapter config + weights (~50MB).
- `true`: Merge adapters into base model via `model.merge_and_unload()`, then save full model.

### Changes

Override `save_model()` in `FinetuneTrainer`:

```python
def save_model(self, output_dir=None):
    if is_peft_model(self.model):
        output_dir = output_dir or self.args.output_dir
        unwrapped = self.accelerator.unwrap_model(self.model)
        if getattr(self.args, 'save_peft_merge', False):
            merged = unwrapped.merge_and_unload()
            merged.save_pretrained(output_dir)
        else:
            unwrapped.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
    else:
        super().save_model(output_dir)
```

### Checkpoint Saving

HuggingFace Trainer's `_save_checkpoint` already detects `PeftModel` and saves only adapter weights. No changes needed.

### Evaluation Loading

For `src/eval.py`:
- Merged models: load directly, no changes.
- Adapter-only saves: add `peft_model_path` support to `get_model()` — loads base model first, then loads adapter via `PeftModel.from_pretrained()`.

---

## 4. Hydra Configuration System

### New Files

**`configs/peft/lora.yaml`**:
```yaml
r: 16
lora_alpha: 32
lora_dropout: 0.05
target_modules:
  - q_proj
  - v_proj
  - k_proj
  - o_proj
bias: "none"
task_type: "CAUSAL_LM"
```

### Modified Files

**`configs/unlearn.yaml`** — add optional peft defaults:
```yaml
defaults:
  - _self_
  - override /model: ...
  - override /trainer: ...
  - optional /peft: null
```

**`configs/trainer/finetune.yaml`** — add save option:
```yaml
args:
  # ... existing args ...
  save_peft_merge: false
```

### CLI Usage

```bash
# Without LoRA (default, backward compatible)
python src/train.py --config-name=unlearn.yaml experiment=unlearn/tofu/default

# With LoRA
python src/train.py --config-name=unlearn.yaml experiment=unlearn/tofu/default \
  peft=lora peft.r=8 peft.lora_alpha=16

# With LoRA, save merged model
python src/train.py --config-name=unlearn.yaml experiment=unlearn/tofu/default \
  peft=lora trainer.args.save_peft_merge=true
```

### Extensibility

Future LoRA variants: create `configs/peft/your_variant.yaml` and use `peft=your_variant` on CLI.

---

## 5. DeepSpeed Compatibility

### PEFT + DeepSpeed Integration

PEFT natively supports DeepSpeed via `accelerate`. Key behaviors:

- ZeRO-3: base model parameters are sharded; LoRA adapter parameters (small) are replicated.
- `accelerate` handles PEFT model distribution automatically.
- `_prepare_deepspeed()` is skipped for ref models in LoRA mode (ref_model is model itself).

### RMU Special Handling

RMU accesses intermediate layers via `_get_matching_module()` and uses `isinstance(model, deepspeed.DeepSpeedEngine)` to unwrap.

With LoRA, model structure becomes `DeepSpeedEngine(PeftModel(LlamaForCausalLM))`. Modify `_get_matching_module()` to unwrap both DeepSpeed and PEFT layers:

```python
def _get_matching_module(self, model, module_regex):
    unwrapped = model
    if isinstance(unwrapped, deepspeed.DeepSpeedEngine):
        unwrapped = unwrapped.module
    if is_peft_model(unwrapped):
        unwrapped = unwrapped.base_model.model
    # ... existing regex matching on unwrapped ...
```

### RMU Parameter Names

PEFT wrapping adds `base_model.model.` prefix to parameter names. RMU's `trainable_params_regex` patterns need adaptation. In LoRA mode, RMU should operate on LoRA parameters (not base model parameters). Document this in RMU's config.

---

## 6. Edge Cases

### Gradient Checkpointing

PEFT models require `model.enable_input_require_grads()` for gradient checkpointing. Handle in `apply_peft()` after `get_peft_model()`.

### Parameter Name Changes

PEFT wrapping changes `named_parameters()` output. All code that filters by parameter name (RMU's `trainable_params_regex`) must account for the `base_model.model.` prefix or use PEFT's `get_model_status()`.

### Evaluation

`model.eval()` disables dropout but keeps LoRA adapters active. No changes needed for evaluation.

### Tokenizer

No impact. Tokenizer is independent of PEFT wrapping.

---

## 7. Dependencies

Add to `requirements.txt`:
```
peft>=0.11.0
```

Compatible with existing: transformers 4.51.3, torch 2.4.1, accelerate 0.34.2, deepspeed 0.15.4.

---

## 8. Files Modified Summary

| File | Change |
|------|--------|
| `src/model/__init__.py` | Add `apply_peft()`, `is_peft_model()`, modify `get_model()` |
| `src/trainer/base.py` | Override `save_model()` for PEFT |
| `src/trainer/unlearn/base.py` | Add `_disable_adapter_context()` |
| `src/trainer/unlearn/grad_diff.py` | Modify `_prepare_ref_model()` for LoRA |
| `src/trainer/unlearn/rmu.py` | Adapt `_get_matching_module()` for PEFT unwrapping |
| `src/train.py` | Pass peft config to `get_model()` |
| `src/eval.py` | Support loading adapter-only saves |
| `configs/peft/lora.yaml` | New: default LoRA config |
| `configs/unlearn.yaml` | Add optional peft defaults |
| `configs/trainer/finetune.yaml` | Add `save_peft_merge` option |
| `requirements.txt` | Add `peft>=0.11.0` |

Methods requiring `compute_loss` changes (add `_disable_adapter_context` wrapping around ref_model calls):
- `src/trainer/unlearn/grad_diff.py` (retain loss with KL)
- `src/trainer/unlearn/dpo.py`
- `src/trainer/unlearn/npo.py`
- `src/trainer/unlearn/undial.py`
- `src/trainer/unlearn/wga.py`
- `src/trainer/unlearn/satimp.py`
- `src/trainer/unlearn/rmu.py`
