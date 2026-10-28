"""CrispEdit trainer mixin: lifecycle for direct base-weight updates.

Composes with any registered trainer to (1) collect retain-set K-FAC
statistics before the optimizer is created, (2) freeze every non-target
parameter, and (3) install a projected ``CrispEditAdamW`` that filters
target-layer gradients into a retain-derived low-curvature subspace. The
selected unlearning trainer's ``compute_loss`` is untouched.
"""

from __future__ import annotations

import logging
from typing import Dict

import torch
from torch.utils.data import DataLoader, SequentialSampler, Subset
from transformers import TrainerCallback

from model.crispedit import resolve_target_modules
from model.crispedit.optimizer import CrispEditAdamW
from model.crispedit.stats import build_projection_cache, collect_kfac_covariances

logger = logging.getLogger(__name__)


class CrispEditCallback(TrainerCallback):
    """Periodically check for weight change and refresh the projection cache."""

    def __init__(self, trainer):
        self.trainer = trainer

    def on_epoch_end(self, args, state, control, **kwargs):
        self.trainer.refresh_crispedit()


class CrispEditTrainerMixin:
    """Lifecycle mixin for CrispEdit direct base-weight unlearning updates."""

    def __init__(self, crispedit_configs=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if crispedit_configs is None:
            raise ValueError("CrispEdit trainer configuration is required.")
        self.crispedit_configs = crispedit_configs
        self._crispedit_projection_cache: Dict = {}
        self._crispedit_bootstrapped = False
        self._crispedit_target_weights: Dict[str, torch.nn.Parameter] = {}
        self._crispedit_ref_weights_cpu: Dict[str, torch.Tensor] = {}
        self.add_callback(CrispEditCallback(self))
        self._setup_crispedit()

    # ---- setup ----------------------------------------------------------

    def _setup_crispedit(self) -> None:
        target_modules = getattr(self.model, "_crispedit_target_modules", None)
        if not target_modules:
            raise ValueError("CrispEdit requires peft=crispedit (target_modules) to be set.")
        layers = list(self.crispedit_configs.get("layers", []))
        self._num_chunks = int(self.crispedit_configs.get("num_chunks", 1))
        self._target_modules = resolve_target_modules(self.model, target_modules, layers)
        self._retain_loader = self._build_retain_loader()
        self._crispedit_target_weights = {n: m.weight for n, m in self._target_modules.items()}

    def _build_retain_loader(self) -> DataLoader:
        retain_dataset = self.train_dataset.retain
        retain_n_samples = self.crispedit_configs.get("retain_n_samples", None)
        if (
            retain_n_samples is not None
            and retain_n_samples > 0
            and retain_n_samples < len(retain_dataset)
        ):
            indices = torch.randperm(
                len(retain_dataset), generator=torch.Generator().manual_seed(0)
            ).tolist()[: retain_n_samples]
            retain_dataset = Subset(retain_dataset, indices)

        sampler = SequentialSampler(retain_dataset)
        return DataLoader(
            retain_dataset,
            batch_size=self.crispedit_configs.get("collection_batch_size", 1),
            sampler=sampler,
            collate_fn=self.data_collator,
            drop_last=False,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

    # ---- bootstrap ------------------------------------------------------

    def bootstrap_crispedit(self) -> None:
        """Collect retain K-FAC and build the projection cache before training.

        Idempotent. Must run before the optimizer is created (i.e. before
        ``trainer.train()``). Called from ``src/train.py``.
        """
        if self._crispedit_bootstrapped:
            return
        if getattr(self, "is_deepspeed_enabled", False):
            raise RuntimeError(
                "CrispEdit is incompatible with DeepSpeed ZeRO because ZeRO "
                "replaces the original matrix parameters with flat optimizer "
                "partitions. Use ordinary DDP or single-process execution."
            )
        if self.optimizer is not None:
            raise RuntimeError(
                "bootstrap_crispedit() must run before the optimizer is created."
            )
        if getattr(self.model, "quantization_method", None) is None:
            self._move_model_to_device(self.model, self.accelerator.device)

        self._freeze_non_target_parameters()

        dtype = self.crispedit_configs.get("retain_factor_dtype", "float32")
        covariance_cache = collect_kfac_covariances(
            self.model,
            self._retain_loader,
            self._target_modules,
            self.accelerator.device,
            dtype,
            num_chunks=self._num_chunks,
        )
        energy_threshold = self.crispedit_configs.get("energy_threshold", 0.9)
        self._crispedit_projection_cache = build_projection_cache(
            self.model, covariance_cache, energy_threshold
        )
        self._snapshot_target_weights()
        self._crispedit_bootstrapped = True

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info(
            "CrispEdit bootstrap complete: retain K-FAC collected and "
            "projection cache built for %d target modules.",
            len(self._target_modules),
        )

    def _freeze_non_target_parameters(self) -> None:
        target_ids = {id(w) for w in self._crispedit_target_weights.values()}
        for p in self.model.parameters():
            p.requires_grad_(id(p) in target_ids)

    def _snapshot_target_weights(self) -> None:
        self._crispedit_ref_weights_cpu = {
            name: w.detach().cpu().clone()
            for name, w in self._crispedit_target_weights.items()
        }

    # ---- optimizer ------------------------------------------------------

    def create_optimizer(self, *args, **kwargs):
        """Build a projected ``CrispEditAdamW`` over the target weights only."""
        if not self._crispedit_bootstrapped:
            raise RuntimeError(
                "CrispEdit optimizer requires bootstrap_crispedit() to have run."
            )
        lr = self.args.learning_rate
        betas = (self.args.adam_beta1, self.args.adam_beta2)
        eps = self.args.adam_epsilon
        weight_decay = self.crispedit_configs.get(
            "weight_decay", getattr(self.args, "weight_decay", 0.0)
        )
        params = list(self._crispedit_projection_cache.keys())
        optimizer = CrispEditAdamW(
            params,
            projection_cache_map=self._crispedit_projection_cache,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            offload_state_to_cpu=bool(
                self.crispedit_configs.get("offload_optimizer_state", False)
            ),
        )
        self.optimizer = optimizer
        return optimizer

    # ---- refresh --------------------------------------------------------

    def refresh_crispedit(self) -> None:
        """Optionally recollect statistics and reset the projection cache.

        Recollects only when ``refresh_on_weight_change`` is enabled and the
        maximum relative target-weight change exceeds
        ``weight_change_threshold``; then calls ``optimizer.reset_cache(...)``
        so ``exp_avg`` is re-projected exactly as in CrispEdit.
        """
        if not self.crispedit_configs.get("refresh_on_weight_change", False):
            return
        threshold = self.crispedit_configs.get("weight_change_threshold", 0.1)
        if not self._target_weights_changed(threshold):
            return

        dtype = self.crispedit_configs.get("retain_factor_dtype", "float32")
        covariance_cache = collect_kfac_covariances(
            self.model,
            self._retain_loader,
            self._target_modules,
            self.accelerator.device,
            dtype,
            num_chunks=self._num_chunks,
        )
        energy_threshold = self.crispedit_configs.get("energy_threshold", 0.9)
        new_cache = build_projection_cache(self.model, covariance_cache, energy_threshold)
        self._crispedit_projection_cache = new_cache
        if self.optimizer is not None:
            self.optimizer.reset_cache(new_cache)
        self._snapshot_target_weights()

    def _target_weights_changed(self, threshold: float) -> bool:
        for name, w in self._crispedit_target_weights.items():
            ref = self._crispedit_ref_weights_cpu[name].to(w.device)
            denom = ref.norm()
            if denom == 0:
                continue
            change = (w.detach() - ref).norm() / denom
            if change > threshold:
                return True
        return False


__all__ = ["CrispEditTrainerMixin"]
