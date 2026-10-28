"""LoRA initialization helpers (preprocessing-only).

Currently only VILA (arXiv:2508.21300) is implemented. See
``scripts/prepare_vila_init.py`` for the end-to-end driver and
``src/model/lora_init/vila.py`` for the algorithm.

Nothing in this package is imported by the train / eval runtime: the
preprocessing script writes a self-contained directory containing the
residual base weights and a standard PEFT adapter, which the unlearn /
eval pipeline loads like any other LoRA checkpoint.
"""

from model.lora_init.vila import apply_vila_init, compute_vila_importance

__all__ = ["apply_vila_init", "compute_vila_importance"]
