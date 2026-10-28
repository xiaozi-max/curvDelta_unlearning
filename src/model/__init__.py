from transformers import AutoModelForCausalLM, AutoTokenizer
from omegaconf import DictConfig, OmegaConf, open_dict
from typing import Dict, Any
import os
import torch
import logging
from model.probe import ProbedLlamaForCausalLM

hf_home = os.getenv("HF_HOME", default=None)

logger = logging.getLogger(__name__)

MODEL_REGISTRY: Dict[str, Any] = {}


def _register_model(model_class):
    MODEL_REGISTRY[model_class.__name__] = model_class


def is_peft_model(model):
    try:
        from peft import PeftModel

        return isinstance(model, PeftModel)
    except ImportError:
        return False


def apply_peft(model, peft_cfg):
    peft_dict = OmegaConf.to_container(peft_cfg, resolve=True)
    peft_type = peft_dict.pop("peft_type")
    peft_dict.pop("curvLoRA_configs", None)
    peft_dict.pop("curvDelta_configs", None)
    oplora_k = peft_dict.pop("oplora_k", None)
    oplora_niter = peft_dict.pop("oplora_niter", 4)

    if peft_type == "crispedit":
        # No adapter: CrispEdit updates base weights directly. Keep the target
        # module names on the model so CrispEditTrainerMixin can resolve them.
        model._crispedit_target_modules = list(peft_dict["target_modules"])
        return model

    if peft_type in {"lora", "curvLoRA", "oplora"}:
        from peft import LoraConfig, get_peft_model

        peft_config = LoraConfig(**peft_dict)
        model = get_peft_model(model, peft_config)

        if peft_type == "curvLoRA":
            from model.curvLoRA import apply_curvLoRA

            model = apply_curvLoRA(model)
        elif peft_type == "oplora":
            if oplora_k is None:
                raise ValueError("peft_type oplora requires oplora_k")
            from model.oplora import apply_oplora

            model = apply_oplora(model, k=int(oplora_k), niter=int(oplora_niter))
    elif peft_type in {"dense", "curvDelta"}:
        from model.dense import DenseConfig, get_dense_model

        peft_config = DenseConfig(**peft_dict)
        if peft_type == "curvDelta":
            from model.curvDelta import get_curvDelta_model

            model = get_curvDelta_model(model, peft_config)
        else:
            model = get_dense_model(model, peft_config)
    else:
        raise ValueError(f"Unknown peft_type: {peft_type!r}")

    model.enable_input_require_grads()
    model.print_trainable_parameters()
    return model


def get_dtype(model_args):
    with open_dict(model_args):
        torch_dtype = model_args.pop("torch_dtype", None)
    if model_args.get("attn_implementation", None) == "flash_attention_2":
        # This check handles https://github.com/Dao-AILab/flash-attention/blob/7153673c1a3c7753c38e4c10ef2c98a02be5f778/flash_attn/flash_attn_triton.py#L820
        # If you want to run at other precisions consider running "training or inference using
        # Automatic Mixed-Precision via the `with torch.autocast(device_type='torch_device'):`
        # decorator" or using an attn_implementation compatible with the precision in the model
        # config.
        assert torch_dtype in ["float16", "bfloat16"], ValueError(
            f"Invalid torch_dtype '{torch_dtype}' for the requested attention "
            f"implementation: 'flash_attention_2'. Supported types are 'float16' "
            f"and 'bfloat16'."
        )
    if torch_dtype == "float16":
        return torch.float16
    elif torch_dtype == "bfloat16":
        return torch.bfloat16
    return torch.float32


def get_model(model_cfg: DictConfig, peft_cfg: DictConfig = None, is_trainable: bool = False):
    """Load a (base, tokenizer) pair, optionally wrapped in a PEFT adapter.

    Adapter selection:

    1. ``model_cfg.peft_model_path`` set → load that adapter on top of the
       base from ``model_args.pretrained_model_name_or_path``.
    2. Otherwise if ``peft_cfg`` is not None → freshly wrap the base via
       ``apply_peft`` (training-from-scratch path).
    3. Otherwise → return the plain base model.

    For VILA-init checkpoints produced by ``scripts/prepare_vila_init.py``
    (layout: ``<vila_dir>/{adapter_config.json, base/}``), point
    ``model_args.pretrained_model_name_or_path`` and ``tokenizer_args`` at
    ``<vila_dir>/base`` and set ``model_cfg.peft_model_path=<vila_dir>``;
    no special branch is needed here.

    ``is_trainable`` is forwarded to ``PeftModel.from_pretrained`` so the
    adapter is loaded with grad-tracking enabled at training time.
    """
    assert model_cfg is not None and model_cfg.model_args is not None, ValueError(
        "Model config not found or model_args absent in configs/model."
    )
    model_args = model_cfg.model_args
    tokenizer_args = model_cfg.tokenizer_args
    torch_dtype = get_dtype(model_args)
    model_handler = model_cfg.get("model_handler", "AutoModelForCausalLM")
    model_cls = MODEL_REGISTRY[model_handler]
    with open_dict(model_args):
        model_path = model_args.pop("pretrained_model_name_or_path", None)
    try:
        model = model_cls.from_pretrained(
            pretrained_model_name_or_path=model_path,
            torch_dtype=torch_dtype,
            **model_args,
            cache_dir=hf_home,
        )
    except Exception as e:
        logger.warning(f"Model {model_path} requested with {model_args}")
        raise ValueError(
            f"Error {e} while fetching model using {model_cls.__name__}.from_pretrained()."
        )

    peft_model_path = model_cfg.get("peft_model_path", None)
    if peft_model_path is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(
            model, peft_model_path, is_trainable=is_trainable
        )
        if is_trainable:
            model.enable_input_require_grads()
        logger.info(f"Loaded PEFT adapter from {peft_model_path}")
    elif peft_cfg is not None:
        model = apply_peft(model, peft_cfg)

    tokenizer = get_tokenizer(tokenizer_args)
    return model, tokenizer


def _add_or_replace_eos_token(tokenizer, eos_token: str) -> None:
    is_added = tokenizer.eos_token_id is None
    num_added_tokens = tokenizer.add_special_tokens({"eos_token": eos_token})

    if is_added:
        logger.info("Add eos token: {}".format(tokenizer.eos_token))
    else:
        logger.info("Replace eos token: {}".format(tokenizer.eos_token))

    if num_added_tokens > 0:
        logger.info("New tokens have been added, make sure `resize_vocab` is True.")


def get_tokenizer(tokenizer_cfg: DictConfig):
    try:
        tokenizer = AutoTokenizer.from_pretrained(**tokenizer_cfg, cache_dir=hf_home)
    except Exception as e:
        error_message = (
            f"{'--' * 40}\n"
            f"Error {e} fetching tokenizer using AutoTokenizer.\n"
            f"Tokenizer requested from path: {tokenizer_cfg.get('pretrained_model_name_or_path', None)}\n"
            f"Full tokenizer config: {tokenizer_cfg}\n"
            f"{'--' * 40}"
        )
        raise RuntimeError(error_message)

    if tokenizer.eos_token_id is None:
        logger.info("replacing eos_token with <|endoftext|>")
        _add_or_replace_eos_token(tokenizer, eos_token="<|endoftext|>")

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Setting pad_token as eos token: {}".format(tokenizer.pad_token))

    # Qwen3 flash attention requires left padding.
    # Detect by checking for Qwen2Tokenizer (used by Qwen3-4B-Instruct-2507).
    if "Qwen2Tokenizer" in type(tokenizer).__name__:
        if tokenizer.padding_side != "left":
            logger.info(
                "Setting tokenizer.padding_side to 'left' (required by Qwen3 flash attention)"
            )
            tokenizer.padding_side = "left"

    return tokenizer


# register models
_register_model(AutoModelForCausalLM)
_register_model(ProbedLlamaForCausalLM)
