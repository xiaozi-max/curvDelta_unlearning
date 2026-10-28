"""End-to-end VILA preprocessing.

Produces a directory containing both the residualized base weights and a
standard PEFT adapter initialized with the VILA scheme. The unlearn / eval
pipeline then loads it like any ordinary LoRA checkpoint via the existing
``model.peft_model_path`` mechanism.

Pipeline::

    1. Load original base + LoRA wrapper (uses the same `peft` Hydra group).
    2. Compute per-LoRA-parameter Fisher importance on forget+retain.
    3. Run weighted SVD to get VILA-init A, B (with per-rank rebalance).
    4. Replace base_layer.weight with residual W - scaling·B@A.
    5. Save:
         <out>/adapter_config.json, adapter_model.safetensors  # VILA-init adapter
         <out>/tokenizer.*                                     # tokenizer (mirrored)
         <out>/base/{config.json, model.safetensors, tokenizer.*}  # residual base

       ``adapter_config.json`` records ``base_model_name_or_path`` as the
       absolute path to ``<out>/base`` so downstream consumers do not have to
       do any path arithmetic. The base lives in a subdirectory so that
       loading it with ``AutoModelForCausalLM.from_pretrained(<out>/base)``
       sees standard HF weights and no adapter sidecar.

Usage::

    python scripts/prepare_vila_init.py \\
        experiment=unlearn/tofu/default \\
        model=Llama-3.2-3B-Instruct \\
        model.model_args.pretrained_model_name_or_path=$ROOT_DIR/models/.../tofu_full \\
        forget_split=forget10 retain_split=retain90 \\
        peft=lora peft.r=8 peft.lora_alpha=16 peft.lora_dropout=0 \\
        lora_init=vila lora_init.path=saves/vila_init/tofu_3B_f10_r8

After this, train with::

    src/train.py ... \\
        model.model_args.pretrained_model_name_or_path=saves/vila_init/tofu_3B_f10_r8/base \\
        model.tokenizer_args.pretrained_model_name_or_path=saves/vila_init/tofu_3B_f10_r8/base \\
        +model.peft_model_path=saves/vila_init/tofu_3B_f10_r8 \\
        trainer=NPO \\
        +trainer.method_args.ref_model_path=$ORIGINAL_BASE

The ``+`` prefix is required by Hydra to *add* fields that are not present in
the default config (``model.peft_model_path`` and
``trainer.method_args.ref_model_path``).
"""

import json
import logging
import os
import sys
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader

# Make ``src/`` importable when invoked as ``python scripts/...``
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from data import get_collators, get_data  # noqa: E402
from model import get_model  # noqa: E402
from model.lora_init.vila import (  # noqa: E402
    apply_vila_init,
    compute_vila_importance,
)
from trainer.utils import seed_everything  # noqa: E402

logger = logging.getLogger(__name__)


def _save_vila_directory(model, tokenizer, out_dir: str) -> None:
    """Save the residualized base + VILA-init adapter.

    Layout:
      <out_dir>/
        adapter_config.json      (base_model_name_or_path = abs path to base/)
        adapter_model.safetensors
        tokenizer files...
        base/
          config.json
          model*.safetensors
          tokenizer files...

    The base lives in a subdirectory so that
    ``AutoModelForCausalLM.from_pretrained(<out_dir>/base)`` sees a clean
    HF checkpoint with no adapter sidecar. ``base_model_name_or_path`` is
    written as an absolute path so consumers do not need to know where the
    adapter was originally produced.

    The PEFT model is first asked to save its adapter (which writes
    ``adapter_config.json`` and ``adapter_model.safetensors``). The base is
    then obtained by calling ``model.unload()``, which removes the PEFT
    wrappers and restores plain ``nn.Linear`` modules holding the residual
    ``base_layer.weight``. This is required so the saved base has standard
    HF state-dict keys (``...q_proj.weight``) instead of PEFT-wrapped ones
    (``...q_proj.base_layer.weight``).
    """
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    base_dir = os.path.join(out_dir, "base")
    os.makedirs(base_dir, exist_ok=True)
    tokenizer.save_pretrained(out_dir)
    tokenizer.save_pretrained(base_dir)
    model.save_pretrained(out_dir, safe_serialization=True)
    base = model.unload()
    base.save_pretrained(base_dir, safe_serialization=True)
    cfg_path = os.path.join(out_dir, "adapter_config.json")
    with open(cfg_path) as f:
        cfg = json.load(f)
    cfg["base_model_name_or_path"] = base_dir
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)


@hydra.main(version_base=None, config_path="../configs", config_name="unlearn.yaml")
def main(cfg: DictConfig):
    seed_everything(cfg.get("seed", 0))

    lora_init_cfg = cfg.get("lora_init", None)
    assert lora_init_cfg is not None, (
        "Pass `lora_init=vila` so this script can read precompute hyperparameters "
        "(sigma, num_epochs, max_steps, batch_size, eps, rebalance) and the output path."
    )
    assert lora_init_cfg.handler == "vila", (
        f"prepare_vila_init.py expects `lora_init.handler=vila`, got "
        f"{lora_init_cfg.handler!r}."
    )

    peft_cfg = cfg.get("peft", None)
    assert peft_cfg is not None, "VILA preprocessing requires a peft (LoRA) config."

    out_dir = lora_init_cfg.path
    os.makedirs(out_dir, exist_ok=True)

    model_cfg = cfg.model
    template_args = model_cfg.template_args

    model, tokenizer = get_model(model_cfg, peft_cfg=peft_cfg, is_trainable=True)
    if torch.cuda.is_available():
        model = model.to("cuda")
    logger.info(f"Model loaded on {model.device}. Loading dataset ...")

    data = get_data(cfg.data, mode="train", tokenizer=tokenizer, template_args=template_args)
    collator = get_collators(cfg.collator, tokenizer=tokenizer)

    forget_ds, retain_ds = data["forget"], data["retain"]
    logger.info(f"forget set size: {len(forget_ds)} | retain set size: {len(retain_ds)}")

    bs = lora_init_cfg.batch_size
    forget_loader = DataLoader(forget_ds, batch_size=bs, shuffle=True, collate_fn=collator)
    retain_loader = DataLoader(retain_ds, batch_size=bs, shuffle=True, collate_fn=collator)

    importance = compute_vila_importance(
        model,
        forget_loader,
        retain_loader,
        sigma=lora_init_cfg.sigma,
        num_epochs=lora_init_cfg.num_epochs,
        max_steps=lora_init_cfg.get("max_steps", None),
    )
    logger.info(
        f"Computed importance: f_cnt={int(importance['f_cnt'])}, "
        f"r_cnt={int(importance['r_cnt'])}, #tensors={len(importance['importance_f'])}"
    )

    apply_vila_init(
        model,
        importance,
        eps=lora_init_cfg.eps,
        rebalance=lora_init_cfg.get("rebalance", True),
    )

    _save_vila_directory(model, tokenizer, out_dir)
    logger.info(f"Saved VILA-init checkpoint to {out_dir}")


if __name__ == "__main__":
    main()
