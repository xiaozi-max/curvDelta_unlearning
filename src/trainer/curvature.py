import logging
import re
from collections import defaultdict
from contextlib import contextmanager
from typing import Dict, List, Optional

from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, SequentialSampler, Subset
from transformers import TrainerCallback

from model.curvLoRA import (
    compute_projection_cache,
    compute_projection_cache_bottom,
    name_step_seed,
    random_orthogonal_basis,
)
from model.curvLoRA.layer import curvLoRA

logger = logging.getLogger(__name__)


# ====================================================================== #
# KFACTracker — centralised hook & accumulator state                      #
# ====================================================================== #


class KFACTracker:
    """Captures KFAC factors via forward/backward hooks.

    All state (saved activations, accumulators, EMA history) lives here.
    Supports chunked processing: when ``num_chunks > 1``, modules are split
    into groups and only the active chunk's accumulators reside on GPU.

    Typical usage::

        tracker = KFACTracker(target_modules)
        tracker.install_hooks()
        for batch in loader:
            with tracker.capture(batch["attention_mask"]):
                loss = model(**batch).loss
                loss.backward()
        tracker.remove_hooks()

    Chunked usage::

        tracker = KFACTracker(target_modules, device=gpu, num_chunks=3)
        for i in range(tracker.num_chunks):
            tracker.activate_chunk(i)
            tracker.clear_accumulators()
            tracker.install_hooks()
            for batch in loader:
                with tracker.capture(mask):
                    loss.backward()
            tracker.remove_hooks()
            # ... compute projections for this chunk ...
            tracker.deactivate_chunk(i)

    For KL-style losses where a ``no_grad()`` reference forward goes through
    the same modules, the ``capture`` guard skips that pass automatically
    (``torch.is_grad_enabled()`` is False).
    """

    def __init__(
        self,
        target_modules: Dict[str, torch.nn.Module],
        device: torch.device = None,
        num_chunks: int = 1,
    ):
        self.target_modules = target_modules
        self._compute_device = device or torch.device("cpu")
        # Factors always live on CPU between collections; only the active
        # chunk is moved to the compute device by ``activate_chunk``.
        self._storage_device = torch.device("cpu")
        self._handles: list = []
        self._capturing = False
        self._current_mask: Optional[torch.Tensor] = None
        self._saved_acts: Dict[str, list] = {n: [] for n in target_modules}

        # KFAC accumulators (on storage device; moved to compute device per chunk)
        self.kfac_A: Dict[str, torch.Tensor] = {}
        self.kfac_S: Dict[str, torch.Tensor] = {}
        self.kfac_n: Dict[str, int] = {}

        # Persistent factors for EMA blending
        self.prev_A: Dict[str, Optional[torch.Tensor]] = {}
        self.prev_S: Dict[str, Optional[torch.Tensor]] = {}

        for name, module in target_modules.items():
            self.kfac_A[name] = torch.zeros(
                module.in_features,
                module.in_features,
                dtype=torch.float32,
                device=self._storage_device,
            )
            self.kfac_S[name] = torch.zeros(
                module.out_features,
                module.out_features,
                dtype=torch.float32,
                device=self._storage_device,
            )
            self.prev_A[name] = None
            self.prev_S[name] = None
            self.kfac_n[name] = 0

        # Chunk bookkeeping
        module_names = list(target_modules.keys())
        num_chunks = min(num_chunks, max(len(module_names), 1))
        self._chunks: List[List[str]] = []
        if module_names:
            chunk_size = (len(module_names) + num_chunks - 1) // num_chunks
            for i in range(0, len(module_names), chunk_size):
                self._chunks.append(module_names[i : i + chunk_size])
        self._current_chunk: Optional[int] = None

    # ---- chunk lifecycle --------------------------------------------------

    @property
    def num_chunks(self) -> int:
        return len(self._chunks)

    def activate_chunk(self, chunk_idx: int) -> None:
        """Move the chunk's accumulators (and EMA state) to compute device."""
        self._current_chunk = chunk_idx
        self._move_chunk_state(chunk_idx, self._compute_device)

    def deactivate_chunk(self, chunk_idx: int) -> None:
        """Move the chunk's state back to storage device and remove hooks."""
        self.remove_hooks()
        self._move_chunk_state(chunk_idx, self._storage_device)
        self._current_chunk = None

    # Clear cache after moving chunk to CPU to reduce fragmentation
        if self._compute_device.type == "cuda":
            torch.cuda.empty_cache()

    def _move_chunk_state(self, chunk_idx: int, device: torch.device) -> None:
        for name in self._chunks[chunk_idx]:
            self.kfac_A[name] = self.kfac_A[name].to(device)
            self.kfac_S[name] = self.kfac_S[name].to(device)
            if self.prev_A[name] is not None:
                self.prev_A[name] = self.prev_A[name].to(device)
                self.prev_S[name] = self.prev_S[name].to(device)

    def active_module_names(self) -> list:
        """Return module names for the currently active chunk (or all if unchunked)."""
        if self._current_chunk is None:
            return list(self.target_modules.keys())
        return self._chunks[self._current_chunk]

    # ---- hook management ------------------------------------------------

    def install_hooks(self) -> None:
        for name in self.active_module_names():
            module = self.target_modules[name]
            self._handles.append(
                module.register_forward_pre_hook(self._make_fwd_hook(name))
            )
            self._handles.append(
                module.register_full_backward_pre_hook(self._make_bwd_hook(name))
            )

    def remove_hooks(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()
        self._clear_transient_state()

    # ---- capture context ------------------------------------------------

    @contextmanager
    def capture(self, mask: torch.Tensor):
        """Enable KFAC capture for a forward + backward pass.

        The ``no_grad()`` guard inside the hooks ensures that reference-model
        forwards (e.g. KL divergence) going through the same base modules are
        silently skipped.
        """
        self._current_mask = mask
        self._capturing = True
        try:
            yield
        finally:
            self._capturing = False
            self._current_mask = None
            for acts in self._saved_acts.values():
                acts.clear()

    # ---- accumulators ---------------------------------------------------

    def clear_accumulators(self) -> None:
        for name in self.active_module_names():
            self.kfac_A[name].zero_()
            self.kfac_S[name].zero_()
            self.kfac_n[name] = 0

    # ---- persistent factor management (EMA blending) --------------------

    def save_persistent(self) -> None:
        for name in self.active_module_names():
            self.prev_A[name] = self.kfac_A[name].clone()
            self.prev_S[name] = self.kfac_S[name].clone()

    def blend_persistent(self, ema_decay: float) -> None:
        for name in self.active_module_names():
            self.kfac_A[name].mul_(1.0 - ema_decay).add_(self.prev_A[name], alpha=ema_decay)
            self.kfac_S[name].mul_(1.0 - ema_decay).add_(self.prev_S[name], alpha=ema_decay)

    # ---- internal hooks -------------------------------------------------

    def _make_fwd_hook(self, name: str):
        def hook(module, args):
            if not self._capturing or not torch.is_grad_enabled():
                return
            self._saved_acts[name].append(args[0].detach())

        return hook

    def _make_bwd_hook(self, name: str):
        def hook(module, grad_output):
            if not self._capturing:
                return
            acts = self._saved_acts[name].pop()
            grads = grad_output[0].detach()
            valid_mask = self._current_mask[:, :-1].bool()
            acts_f = acts[:, :-1, :][valid_mask].to(dtype=torch.float32)
            grads_f = grads[:, :-1, :][valid_mask].to(dtype=torch.float32)
            self.kfac_A[name].addmm_(acts_f.T, acts_f)
            self.kfac_S[name].addmm_(grads_f.T, grads_f)
            self.kfac_n[name] += acts_f.shape[0]

        return hook

    def _clear_transient_state(self) -> None:
        for acts in self._saved_acts.values():
            acts.clear()


class CurvatureCallback(TrainerCallback):
    def __init__(self, trainer):
        self.trainer = trainer

    def _should_refresh(self, counter: int, unit: str) -> bool:
        configs = self.trainer.curvature_configs
        if configs.cache_mode == "static":
            return False
        if configs.update_unit != unit:
            return False
        if counter == 0 or counter % configs.update_interval != 0:
            return False
        return True

    def on_train_begin(self, args, state, control, **kwargs):
        self.trainer.curvature_on_train_begin()

    def on_epoch_begin(self, args, state, control, **kwargs):
        if self._should_refresh(int(state.epoch), "epochs"):
            self.trainer.curvature_refresh()

    def on_step_begin(self, args, state, control, **kwargs):
        if self._should_refresh(state.global_step, "steps"):
            self.trainer.curvature_refresh()

    def on_optimizer_step(self, args, state, control, **kwargs):
        self.trainer._project_lora_adapter_weights()

    def on_train_end(self, args, state, control, **kwargs):
        self.trainer.curvature_on_train_end()


class CurvatureTrainerMixin:
    """Mixin that manages KFAC collection and curvature projection updates."""

    def __init__(
        self,
        curvLoRA_configs=None,
        curvDelta_configs=None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.add_callback(CurvatureCallback(self))

        if curvLoRA_configs is not None and curvDelta_configs is not None:
            raise ValueError(
                "Specify only one of curvLoRA_configs and curvDelta_configs."
            )
        self.curvature_configs = curvLoRA_configs or curvDelta_configs
        if self.curvature_configs is None:
            raise ValueError("Curvature trainer configuration is required.")
        self._target_modules: Dict[str, torch.nn.Module] = {}
        self._adapter_names: Dict[str, str] = {}
        # Set before the concrete unlearning trainer initializes so
        # bootstrap can run from train.py before trainer.train().
        self._curvature_initial_projection_ready = False
        self._setup_curvature()

    def _setup_curvature(self) -> None:
        for name, module in self.model.named_modules():
            variants = getattr(module, "lora_variant", None) or getattr(module, "dense_variant", None)
            if not variants:
                continue
            self._target_modules[name] = module
            self._adapter_names[name] = next(iter(variants))

        # Assign modules round-robin to ranks for distributed factor collection.
        # Each rank only tracks its assigned modules (on GPU), while all ranks
        # process the full retain dataset.
        module_names = list(self._target_modules.keys())
        rank = self.args.process_index
        world_size = self.args.world_size
        assigned_modules = {
            name: self._target_modules[name]
            for i, name in enumerate(module_names)
            if i % world_size == rank
        }

        self._tracker = KFACTracker(
            assigned_modules,
            device=self.accelerator.device,
            num_chunks=getattr(self.curvature_configs, "num_chunks", 1),
        )
        self._local_projections: Dict[str, tuple] = {}
        self._retain_loader = self._build_retain_loader()
        self._curvature_fisher_generator: Optional[torch.Generator] = None

    def _build_retain_loader(self) -> DataLoader:
        retain_dataset = self.train_dataset.retain

        # Subsample retain to retain_ratio × anchor size.
        # retain_ratio=1.0 (default) reproduces the original anchor-equal budget.
        anchor_size = len(self.train_dataset)
        retain_ratio = float(getattr(self.curvature_configs, "retain_ratio", 1.0))
        target_size = int(round(anchor_size * retain_ratio))

        full_indices = torch.randperm(
            len(retain_dataset), generator=torch.Generator().manual_seed(0)
        ).tolist()

        if target_size >= len(retain_dataset):
            if target_size > len(retain_dataset):
                logger.warning(
                    "retain_ratio=%.4f requests %d samples but retain set has only %d; "
                    "capping to full retain set.",
                    retain_ratio,
                    target_size,
                    len(retain_dataset),
                )
            indices = full_indices
        else:
            # Prefix of the permutation so smaller-ratio sets are strict subsets.
            indices = full_indices[:target_size]

        retain_dataset = Subset(retain_dataset, indices)

        # All ranks process the full dataset; each rank only collects KFAC
        # factors for its round-robin-assigned module subset on GPU.
        sampler = SequentialSampler(retain_dataset)

        dataloader = DataLoader(
            retain_dataset,
            batch_size=self.curvature_configs.collection_batch_size,
            sampler=sampler,
            collate_fn=self.data_collator,
            drop_last=False,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

        return dataloader

    def _get_curvature_fisher_generator(self, device: torch.device) -> torch.Generator:
        generator = self._curvature_fisher_generator
        if generator is None or generator.device != device:
            generator = torch.Generator(device=device)
            generator.manual_seed(int(self.curvature_configs.fisher_sample_seed))
            self._curvature_fisher_generator = generator
        return generator

    def _curvature_valid_target_mask(self, retain_inputs) -> torch.Tensor:
        if "labels" in retain_inputs:
            return retain_inputs["labels"][..., 1:].ne(-100)
        return retain_inputs["attention_mask"][..., 1:].bool()

    def _compute_curvature_kl_fisher_loss(self, model, retain_inputs):
        temperature = float(self.curvature_configs.fisher_temperature)
        if temperature <= 0:
            raise ValueError("curvature fisher_temperature must be > 0")

        if getattr(self, "ref_model", None) is None:
            self.ref_model = self._prepare_ref_model(model)

        with torch.no_grad():
            ref_outputs = self.ref_model(**retain_inputs)
            ref_logits = ref_outputs.logits[..., :-1, :] / temperature
            ref_probs = F.softmax(ref_logits, dim=-1)

        outputs = model(**retain_inputs)
        student_logits = outputs.logits[..., :-1, :]
        valid_mask = self._curvature_valid_target_mask(retain_inputs).to(
            student_logits.device
        )
        flat_ref_probs = ref_probs.to(student_logits.device)[valid_mask]
        generator = self._get_curvature_fisher_generator(student_logits.device)
        sampled_targets = torch.multinomial(
            flat_ref_probs.float(),
            num_samples=1,
            replacement=True,
            generator=generator,
        ).squeeze(-1)
        flat_student_logits = student_logits[valid_mask]
        return F.cross_entropy(flat_student_logits, sampled_targets, reduction="mean")

    def _compute_curvature_retain_loss(self, model, retain_inputs):
        retain_loss_type = self.curvature_configs.retain_loss_type
        if retain_loss_type == "KL":
            return self._compute_curvature_kl_fisher_loss(model, retain_inputs)
        if retain_loss_type == "NLL":
            return model(**retain_inputs).loss
        raise ValueError(f"Unsupported curvature retain_loss_type: {retain_loss_type!r}")

    def bootstrap_curvature(self) -> None:
        """Collect KFAC and apply projections before ``trainer.train()``.

        Called from ``train.py`` after the trainer is constructed and before
        ``trainer.train()`` so the optimizer/DeepSpeed engine is not yet
        initialized. This lets the collection pass run on the raw eager PEFT
        model with a plain ``loss.backward()``, avoiding any DeepSpeed
        lifecycle coupling.

        Sequence (per rank):
          1. Move the model to the rank device (Trainer defers this until
             ``deepspeed_init``; ZeRO-2 does not shard parameters, so placing
             them now is safe and lets KFAC hooks see on-device inputs).
          2. Under ``disable_adapter()`` and with adapter parameters moved
             to CPU, the retain pass collects KFAC on the base layer only —
             the collection peak is ``base + KFAC chunk`` with no adapter,
             optimizer, or gradient storage.
          3. Restore adapter parameters to the device and apply projections.
        """
        if self._curvature_initial_projection_ready:
            return
        if self.optimizer is not None:
            raise RuntimeError(
                "bootstrap_curvature() must run before the optimizer is created."
            )
        resume_from = getattr(self.args, "resume_from_checkpoint", None)
        if resume_from is not None:
            # U_in_bar/U_out_bar are persistent=False buffers and are not
            # restored from a Trainer checkpoint. Bootstrapping from the
            # base model here would recompute projections from the wrong
            # weights and silently train with an inconsistent projection.
            raise ValueError(
                "Curvature bootstrap is not supported when resuming from a "
                "Trainer checkpoint (projection buffers are not persisted). "
                "Start a fresh run or persist/restore the projection cache "
                "explicitly before enabling resume."
            )

        if getattr(self.model, "quantization_method", None) is None:
            self._move_model_to_device(self.model, self.accelerator.device)

        self._offload_adapter_params(torch.device("cpu"))
        try:
            with self.model.disable_adapter():
                is_ema = (
                    self.curvature_configs.cache_mode == "refresh"
                    and self.curvature_configs.update_rule == "ema"
                )
                self._collect_factors(
                    save_persistent=is_ema, pre_engine=True
                )
        finally:
            self._restore_adapter_params(self.accelerator.device)

        self._apply_projections()
        self._curvature_initial_projection_ready = True

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info(
            "Curvature bootstrap complete: KFAC collected and projections "
            "applied before trainer.train()."
        )

    def _offload_adapter_params(self, device: torch.device) -> None:
        """Move all curvature adapter parameters to ``device`` (collection phase)."""
        for module in self._target_modules.values():
            for param in self._iter_adapter_params(module):
                param.data = param.data.to(device)

    def _restore_adapter_params(self, device: torch.device) -> None:
        """Move all curvature adapter parameters back to ``device`` after collection."""
        for module in self._target_modules.values():
            for param in self._iter_adapter_params(module):
                param.data = param.data.to(device)

    @staticmethod
    def _iter_adapter_params(module: torch.nn.Module):
        """Yield trainable curvature parameters for one target module."""
        if hasattr(module, "lora_A"):
            for adapter in module.lora_A.values():
                yield adapter.weight
        if hasattr(module, "lora_B"):
            for adapter in module.lora_B.values():
                yield adapter.weight
        if hasattr(module, "dense_delta"):
            for adapter in module.dense_delta.values():
                yield adapter

    def create_optimizer(self, *args, **kwargs):
        """Build the optimizer, then split curvDelta groups by Transformer layer.

        DeepSpeed ZeRO-2 allocates flat buffers (``bit16_groups_flat``, fp32 master
        partitions, gradient partitions) **per param group**. With a single group
        covering all ``dense_delta`` parameters, these buffers are one huge
        contiguous allocation each (~6.4 GiB for a 3B model), which fragments the
        CUDA caching allocator and causes OOM on subsequent large allocations
        (``copy_grads_in_partition`` during backward, ``flatten().to(fp32)``
        during step). Splitting into one group per Transformer layer makes each
        buffer ~200 MiB, reducing fragmentation and keeping per-group step()
        allocations small enough to fit in available memory.
        """
        optimizer = super().create_optimizer(*args, **kwargs)

        if (
            hasattr(optimizer, "param_groups")
            and self._uses_curvDelta()
            and not getattr(self, "_dense_optimizer_groups_split", False)
        ):
            self._split_dense_optimizer_groups_by_layer(optimizer)
            self._dense_optimizer_groups_split = True

        return optimizer

    def _uses_curvDelta(self) -> bool:
        return any(
            "dense_delta" in name and parameter.requires_grad
            for name, parameter in self.model.named_parameters()
        )

    def _split_dense_optimizer_groups_by_layer(self, optimizer) -> None:
        """Split large Dense groups so ZeRO-2 never flattens all layers at once."""
        if optimizer.state:
            raise RuntimeError(
                "curvDelta optimizer groups must be split before optimizer "
                "state is initialized."
            )

        parameter_names = {
            id(parameter): name for name, parameter in self.model.named_parameters()
        }
        original_parameter_ids = [
            id(parameter)
            for group in optimizer.param_groups
            for parameter in group["params"]
        ]
        new_groups = []

        for original_group in optimizer.param_groups:
            group_options = {
                key: value for key, value in original_group.items() if key != "params"
            }
            layer_buckets = defaultdict(list)
            other_parameters = []

            for parameter in original_group["params"]:
                name = parameter_names.get(id(parameter), "")
                match = re.search(r"(?:^|\.)layers\.(\d+)\.", name)
                if match is None:
                    other_parameters.append(parameter)
                else:
                    layer_buckets[int(match.group(1))].append(parameter)

            for layer_index in sorted(layer_buckets):
                new_groups.append(
                    {
                        **group_options,
                        "params": layer_buckets[layer_index],
                    }
                )
            if other_parameters:
                new_groups.append(
                    {
                        **group_options,
                        "params": other_parameters,
                    }
                )

        new_parameter_ids = [
            id(parameter) for group in new_groups for parameter in group["params"]
        ]
        if sorted(original_parameter_ids) != sorted(new_parameter_ids):
            raise RuntimeError(
                "Optimizer parameters changed while splitting curvDelta groups."
            )
        if not new_groups:
            raise RuntimeError(
                "curvDelta optimizer group splitting produced no groups."
            )

        optimizer.param_groups = new_groups
        largest_group_numel = max(
            sum(parameter.numel() for parameter in group["params"])
            for group in new_groups
        )
        largest_group_bytes = max(
            sum(
                parameter.numel() * parameter.element_size()
                for parameter in group["params"]
            )
            for group in new_groups
        )
        logger.info(
            "Split curvDelta optimizer into %d Transformer-layer groups; "
            "largest group has %.2fM parameters (%.2f MiB in parameter dtype).",
            len(new_groups),
            largest_group_numel / 1e6,
            largest_group_bytes / 2**20,
        )

    def curvature_on_train_begin(self):
        if self._curvature_initial_projection_ready:
            return
        is_ema = (
            self.curvature_configs.cache_mode == "refresh"
            and self.curvature_configs.update_rule == "ema"
        )
        self._collect_factors(save_persistent=is_ema)

        # Clear cache to reduce fragmentation after collection phase
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self._apply_projections()

        # Clear cache again before training begins
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def curvature_refresh(self) -> None:
        # Clear cache BEFORE collecting to flush fragmented blocks from training
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if self.curvature_configs.update_rule == "replace":
            self._collect_factors()
            self._apply_projections()
        elif self.curvature_configs.update_rule == "ema":
            self._collect_factors(blend_ema=True, save_persistent=True)
            self._apply_projections()
        else:
            raise ValueError(
                f"Unknown curvature update rule: {self.curvature_configs.update_rule}"
            )

    def curvature_on_train_end(self):
        self._project_lora_adapter_weights()
        self._tracker.remove_hooks()

    def _project_lora_adapter_weights(self) -> None:
        """Write ``P_out B`` / ``A P_in`` into live LoRA parameters.

        Dense adapters are skipped: their constraint lives in the variant
        forward / ``get_delta_weight`` path, and they already merge on save.
        """
        for name, module in self._target_modules.items():
            if not hasattr(module, "lora_A"):
                continue
            curvLoRA.project_lora_weights_(module, self._adapter_names[name])

    def _collect_factors(
        self,
        blend_ema: bool = False,
        save_persistent: bool = False,
        pre_engine: bool = False,
    ) -> None:
        """Collect KFAC factors, optionally in chunks to reduce GPU memory.

        When ``num_chunks > 1``, iterates over module chunks: each chunk's
        accumulators are moved to GPU, a full retain pass collects factors,
        projections are computed and stored on CPU, then GPU memory is freed
        before the next chunk.

        Args:
            blend_ema: If True, blend current factors with persistent EMA state.
            save_persistent: If True, snapshot factors for future EMA blending.
            pre_engine: If True, the accelerator/DeepSpeed engine is not yet
                initialized (bootstrap path): call ``loss.backward()`` directly
                on the unwrapped model. If False (in-training refresh), use the
                HEAD path ``accelerator.no_sync`` + ``accelerator.backward``.
        """
        tracker = self._tracker
        self._local_projections = {}
        is_main = self.accelerator.is_main_process
        energy_ratio = self.curvature_configs.energy_ratio
        if self.curvature_configs.retain_loss_type == "KL":
            logger.info("Using sampled teacher CE Fisher for KL curvature collection.")

        for chunk_idx in range(tracker.num_chunks):
            self._curvature_fisher_generator = None
            tracker.activate_chunk(chunk_idx)
            tracker.clear_accumulators()
            tracker.install_hooks()
            self.model.eval()

            chunk_names = tracker.active_module_names()
            desc = (
                f"Collecting KFAC factors (chunk {chunk_idx + 1}/{tracker.num_chunks})"
                if tracker.num_chunks > 1
                else "Collecting KFAC factors"
            )
            pbar = tqdm(
                self._retain_loader,
                desc=desc,
                disable=not is_main,
                dynamic_ncols=True,
            )
            for batch in pbar:
                batch = self._prepare_inputs(batch)
                self.model.zero_grad(set_to_none=True)
                with tracker.capture(batch["attention_mask"]):
                    loss = self._compute_curvature_retain_loss(
                        model=self.model,
                        retain_inputs=batch,
                    )
                    if pre_engine:
                        loss.backward()
                    else:
                        with self.accelerator.no_sync(self.model):
                            self.accelerator.backward(loss)
                self.model.zero_grad(set_to_none=True)
                pbar.set_postfix({"retain_loss": f"{loss.item():.4f}"})

            tracker.remove_hooks()
            self.model.train()

            # Normalize factors in-place.
            for name in chunk_names:
                tracker.kfac_A[name] /= tracker.kfac_n[name]
                tracker.kfac_S[name] /= tracker.kfac_n[name]

            # EMA blending for this chunk (factors must be on GPU).
            if blend_ema:
                tracker.blend_persistent(self.curvature_configs.ema_decay)

            # Compute projections for this chunk on GPU, store on CPU.
            for name in chunk_names:
                U_in, U_out = self._build_projection_basis(
                    tracker.kfac_A[name], tracker.kfac_S[name], name
                )
                self._local_projections[name] = (U_in.cpu(), U_out.cpu())

            if save_persistent:
                tracker.save_persistent()

            tracker.deactivate_chunk(chunk_idx)

    def _build_projection_basis(
        self, A: torch.Tensor, S: torch.Tensor, name: str
    ):
        """Compute ``(U_in, U_out)`` tensors for one module according to ``projection_mode``.

        For ``random``, the top K-FAC basis is computed first to determine the
        realized rank per side, then replaced with a deterministic random
        orthonormal basis of the same rank.
        """
        energy_ratio = self.curvature_configs.energy_ratio
        mode = getattr(self.curvature_configs, "projection_mode", "kfac_top")
        input_factor_name = f"{name}.input"
        output_factor_name = f"{name}.output"

        if mode == "kfac_top":
            U_in = compute_projection_cache(
                A, energy_ratio=energy_ratio, factor_name=input_factor_name
            )
            U_out = compute_projection_cache(
                S, energy_ratio=energy_ratio, factor_name=output_factor_name
            )

        elif mode == "kfac_bottom":
            U_in = compute_projection_cache_bottom(
                A, energy_ratio=energy_ratio, factor_name=input_factor_name
            )
            U_out = compute_projection_cache_bottom(
                S, energy_ratio=energy_ratio, factor_name=output_factor_name
            )

        elif mode == "random":
            U_in_top = compute_projection_cache(
                A, energy_ratio=energy_ratio, factor_name=input_factor_name
            )
            U_out_top = compute_projection_cache(
                S, energy_ratio=energy_ratio, factor_name=output_factor_name
            )
            step = getattr(self.state, "global_step", 0)
            seed_in = name_step_seed(name + "::in", step)
            seed_out = name_step_seed(name + "::out", step)
            U_in = random_orthogonal_basis(A.shape[0], U_in_top.shape[1], seed_in, A.device)
            U_out = random_orthogonal_basis(S.shape[0], U_out_top.shape[1], seed_out, S.device)

        elif mode == "input_only":
            U_in = compute_projection_cache(
                A, energy_ratio=energy_ratio, factor_name=input_factor_name
            )
            U_out = torch.zeros(S.shape[0], 0, dtype=U_in.dtype, device=A.device)

        elif mode == "output_only":
            U_in = torch.zeros(A.shape[0], 0, dtype=torch.float32, device=A.device)
            U_out = compute_projection_cache(
                S, energy_ratio=energy_ratio, factor_name=output_factor_name
            )

        else:
            raise ValueError(f"Unknown projection_mode: {mode!r}")

        return U_in, U_out

    def _apply_projections(self) -> None:
        """Set projection buffers on curvature adapter modules.

        Single-GPU: reads directly from ``_local_projections``.
        Multi-GPU: broadcasts via all-reduce so every rank holds identical buffers.

        In both cases projections are converted to the target dtype immediately
        and intermediate fp32 tensors are freed per-module to keep peak
        memory low on VRAM-constrained GPUs (e.g. L40 48 GB with 7B models).
        """
        if self.args.world_size > 1:
            self._distributed_apply_projections()
        else:
            for name, module in self._target_modules.items():
                adapter_name = self._adapter_names[name]
                U_in_bar, U_out_bar = self._local_projections[name]
                ref = self._adapter_reference(module, adapter_name)
                module._buffers[f"U_in_bar_{adapter_name}"] = U_in_bar.to(ref)
                module._buffers[f"U_out_bar_{adapter_name}"] = U_out_bar.to(ref)
                del self._local_projections[name]
            self._local_projections.clear()
        self._project_lora_adapter_weights()

    @staticmethod
    def _adapter_reference(module: torch.nn.Module, adapter_name: str) -> torch.Tensor:
        """Return a parameter used only to infer device/dtype for projection buffers."""
        if hasattr(module, "lora_A"):
            return module.lora_A[adapter_name].weight
        if hasattr(module, "dense_delta"):
            return module.dense_delta[adapter_name]
        raise TypeError(f"Unsupported curvature adapter module: {type(module).__name__}")

    def _distributed_apply_projections(self) -> None:
        """Broadcast pre-computed projections to all ranks, writing buffers directly.

        Each rank computed projections for its round-robin-assigned modules
        during ``_collect_factors()``.  This method broadcasts them so every
        rank holds identical projection buffers.  fp32 intermediates are
        converted to the target dtype immediately and freed per-module to
        avoid holding all projections in fp32 simultaneously.
        """
        device = self.accelerator.device
        rank = self.args.process_index
        world_size = self.args.world_size
        module_names = list(self._target_modules.keys())

        # --- Phase 1: all-reduce projection shapes ---
        num_modules = len(module_names)
        shapes = torch.zeros(num_modules, 2, dtype=torch.long, device=device)
        for i, name in enumerate(module_names):
            if i % world_size == rank:
                U_in, U_out = self._local_projections[name]
                shapes[i, 0] = U_in.shape[1]
                shapes[i, 1] = U_out.shape[1]
        shapes = self.accelerator.reduce(shapes, reduction="sum")

        # --- Phase 2: share projections via all-reduce, write buffers directly ---
        for i, name in enumerate(module_names):
            owner = i % world_size
            module = self._target_modules[name]
            adapter_name = self._adapter_names[name]
            ref = self._adapter_reference(module, adapter_name)
            in_cols = int(shapes[i, 0].item())
            out_cols = int(shapes[i, 1].item())

            if owner == rank:
                buf_in = self._local_projections[name][0].to(
                    device=device, dtype=torch.float32
                ).contiguous()
            else:
                buf_in = torch.zeros(
                    module.in_features, in_cols, dtype=torch.float32, device=device
                )
            buf_in = self.accelerator.reduce(buf_in, reduction="sum").to(ref)
            module._buffers[f"U_in_bar_{adapter_name}"] = buf_in

            if owner == rank:
                buf_out = self._local_projections[name][1].to(
                    device=device, dtype=torch.float32
                ).contiguous()
            else:
                buf_out = torch.zeros(
                    module.out_features, out_cols, dtype=torch.float32, device=device
                )
            buf_out = self.accelerator.reduce(buf_out, reduction="sum").to(ref)
            module._buffers[f"U_out_bar_{adapter_name}"] = buf_out

            # Free the fp32 source immediately (only the owner rank had it).
            if owner == rank:
                del self._local_projections[name]

        self._local_projections.clear()
