import contextlib
import logging
import random
from dataclasses import replace
from functools import cached_property
from typing import Any, Dict, Generator, Iterable, Literal, Optional, Tuple, Union, Set, TYPE_CHECKING
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass, fields as dataclass_fields
from math import cos, pi

import numpy as np
import torch
import torch.distributed as dist
import torch.distributed.checkpoint.state_dict as dist_cp_sd
import torch.nn as nn
from torch.distributed import DeviceMesh
from torch.distributed.checkpoint.metadata import Metadata
from torch.distributed.fsdp import FSDPModule
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.tensor import DTensor
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Optimizer

try:
    # olmo_core's ``Transformer.apply_ddp`` uses the composable ``replicate()`` API,
    # which does not wrap the model in a ``DistributedDataParallel`` instance — it
    # swaps the model's class to a dynamically created subclass of this marker type.
    from torch.distributed._composable.replicate import DDP as _ComposableDDP
except ImportError:  # pragma: no cover - depends on torch version
    _ComposableDDP = None

from olmo_core.data.utils import get_labels, split_batch
from olmo_core.distributed.checkpoint import (
    merge_state_dicts,
    prune_state_dict,
    swap_param_keys,
)
from olmo_core.distributed.parallel import (
    build_world_mesh,
    get_dp_process_group,
)
from olmo_core.distributed.utils import (
    get_local_tensor,
    get_reduce_divide_factor,
    get_world_size,
    is_distributed,
)
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.float8 import Float8Config
from olmo_core.config import Config
from olmo_core.nn.lm_head import LMOutputWithLoss
from olmo_core.nn.transformer import Transformer
from olmo_core.nn.transformer.config import TransformerActivationCheckpointingMode
from olmo_core.optim.scheduler import _linear_warmup as _lr_linear_warmup
from olmo_core.optim.scheduler import _linear_decay as _lr_linear_decay
from olmo_core.optim import OptimConfig, SkipStepOptimizer
from olmo_core.optim.scheduler import Scheduler
from olmo_core.utils import gc_cuda, get_default_device, log_once, move_to_device

from ...common import ReduceType
from ..train_module import EvalBatchSpec, TrainModule
from .common import parallelize_model
from .config import (
    TransformerActivationCheckpointingConfig,
    TransformerContextParallelConfig,
    TransformerDataParallelConfig,
    TransformerExpertParallelConfig,
    TransformerTensorParallelConfig,
    TransformerSAMConfig,
)

log = logging.getLogger(__name__)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from olmo_core.train import Trainer


# ---------------------------------------------------------------------------
# RNG state helpers — ensure identical dropout masks between ascent and
# descent forward passes by capturing and restoring full RNG state.
# ---------------------------------------------------------------------------

def _get_current_cuda_device() -> Optional[int]:
    """
    Return the CUDA device index for this process.
    Assumes the process has already selected its device
    (e.g. via torch.cuda.set_device(local_rank)).
    """
    if not torch.cuda.is_available():
        return None
    return torch.cuda.current_device()


def get_rng_state() -> Dict[str, Any]:
    """
    Capture RNG state for:
      - Python random
      - NumPy
      - PyTorch CPU
      - PyTorch CUDA on the current process's current GPU only
    """
    cuda_device = _get_current_cuda_device()

    state: Dict[str, Any] = {
        "python_random": random.getstate(),
        "numpy_random": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": None,
        "cuda_device": cuda_device,
        "distributed_rank": dist.get_rank() if dist.is_available() and dist.is_initialized() else None,
    }

    if cuda_device is not None:
        state["torch_cuda"] = torch.cuda.get_rng_state(cuda_device)

    return state


def set_rng_state(state: Dict[str, Any]) -> None:
    """
    Restore RNG state saved by get_rng_state().
    Restores CUDA RNG state onto the current process's current GPU.
    """
    random.setstate(state["python_random"])
    np.random.set_state(state["numpy_random"])
    torch.set_rng_state(state["torch_cpu"])

    saved_cuda_state = state.get("torch_cuda", None)
    if saved_cuda_state is not None:
        if not torch.cuda.is_available():
            raise RuntimeError("Saved CUDA RNG state exists, but CUDA is not available now.")

        current_device = _get_current_cuda_device()
        saved_device = state.get("cuda_device", None)

        if saved_device is not None and current_device != saved_device:
            raise RuntimeError(
                f"Current CUDA device ({current_device}) does not match "
                f"saved CUDA device ({saved_device}). Make sure each rank "
                f"sets its correct local device before restoring RNG state."
            )

        torch.cuda.set_rng_state(saved_cuda_state, device=current_device)


def _nested_placeholder_from_keys(keys: Iterable[str], prefix: str) -> Optional[Dict[str, Any]]:
    """
    Build a nested dict of ``None`` placeholders for every flattened checkpoint key under
    ``prefix`` (e.g. ``"sam_scheduler.state.warmup"``), so that the checkpoint loader will
    read those entries back. Returns ``None`` if the checkpoint has no such keys.
    """
    out: Optional[Dict[str, Any]] = None
    for key in keys:
        if not key.startswith(prefix + "."):
            continue
        if out is None:
            out = {}
        node = out
        parts = key[len(prefix) + 1 :].split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = None
    return out


@dataclass
class SAMScheduler(Config, metaclass=ABCMeta):
    """
    Schedule for SAM rho, similar to LR schedulers but specialized to set sam_config.rho.
    """

    decay_alpha: float = 0.1
    warmup: Optional[int] = None
    warmup_steps: Optional[int] = None  # deprecated alias for 'warmup'
    # Internal, saved with checkpoints to ensure consistent semantics on resume.
    _initial_rho: Optional[float] = None

    def __post_init__(self):
        if self.warmup is None and self.warmup_steps is not None:
            self.warmup = self.warmup_steps
            self.warmup_steps = None
        if not (0.0 <= self.decay_alpha <= 1.0):
            raise OLMoConfigurationError("'decay_alpha' must be in [0, 1].")

    @abstractmethod
    def get_rho(self, initial_rho: float, current: int, t_max: int) -> float:
        raise NotImplementedError

    def set_rho(self, sam_config: "TransformerSAMConfig", trainer: "Trainer") -> float:  # type: ignore[name-defined]
        """
        Update and return the current rho using the schedule. Mutates sam_config.rho.
        """
        if self._initial_rho is None:
            self._initial_rho = float(sam_config.rho)
        # Mirror LR scheduler step semantics (by steps)
        current = int(trainer.global_step)
        t_max = int(trainer.max_steps)
        rho = self.get_rho(self._initial_rho, current, t_max)
        sam_config.rho = float(rho)
        return float(rho)

    # Serialization helpers for checkpoints.
    def as_state(self) -> Dict[str, Any]:
        # Serialize every dataclass field (including subclass fields like
        # SAMConstantScheduler.decay) so the schedule round-trips through checkpoints.
        return {f.name: getattr(self, f.name) for f in dataclass_fields(self)}

    @classmethod
    def from_state(cls, state: Dict[str, Any]) -> "SAMScheduler":
        inst = cls()  # type: ignore[call-arg]
        for k, v in state.items():
            setattr(inst, k, v)
        return inst


@dataclass
class SAMConstantScheduler(SAMScheduler):
    """
    Constant rho with optional warmup and optional linear decay at the end to decay_alpha * rho0.
    """

    decay: Optional[int] = None
    decay_steps: Optional[int] = None  # deprecated alias for 'decay'

    def __post_init__(self):
        super().__post_init__()
        if self.decay is None and self.decay_steps is not None:
            self.decay = self.decay_steps
            self.decay_steps = None
        if self.decay is not None and self.decay < 0:
            raise OLMoConfigurationError("'decay' must be >= 0.")

    def get_rho(self, initial_rho: float, current: int, t_max: int) -> float:
        # The warmup/decay helpers assert their min value is strictly below initial_rho,
        # so short-circuit the degenerate cases (rho == 0, decay_alpha == 1).
        if initial_rho <= 0:
            return 0.0

        warmup = int(self.warmup or 0)
        # Warmup from 0 -> initial_rho
        if warmup > 0 and current <= warmup:
            return float(_lr_linear_warmup(initial_rho, current, warmup, 0.0))

        # No decay configured: stay constant after warmup
        if self.decay is None or self.decay == 0:
            return float(initial_rho)

        decay = int(self.decay)
        start_decay_at = max(t_max - decay, warmup)
        if current >= start_decay_at:
            # Linear decay to decay_alpha * initial_rho at step t_max
            eta_min = float(initial_rho * self.decay_alpha)
            if eta_min >= initial_rho:
                return float(initial_rho)
            step_from_end = max(t_max - current, 0)
            return float(_lr_linear_decay(initial_rho, step_from_end, decay, eta_min))

        return float(initial_rho)


@dataclass
class SAMCosineScheduler(SAMScheduler):
    """
    Cosine decay from initial_rho to decay_alpha * initial_rho with optional warmup (0 -> initial_rho).
    """

    def get_rho(self, initial_rho: float, current: int, t_max: int) -> float:
        if initial_rho <= 0:
            return 0.0

        warmup = int(self.warmup or 0)
        eta_min = float(initial_rho * self.decay_alpha)

        if warmup > 0 and current < warmup:
            return float(_lr_linear_warmup(initial_rho, current, warmup, 0.0))

        if current >= t_max:
            return float(eta_min)

        # Cosine over remaining steps after warmup
        current_adj = max(current - warmup, 0)
        t_max_adj = max(t_max - warmup, 1)
        return float(eta_min + (initial_rho - eta_min) * (1 + cos(pi * current_adj / t_max_adj)) / 2.0)


class TransformerSAMTrainModule(TrainModule):
    """
    A :class:`TrainModule` for any :class:`~olmo_core.nn.transformer.Transformer` model
    implementation provided by this library.

    .. tip::
        Use the :class:`TransformerTrainModuleConfig` to easily configure and build
        :class:`TransformerTrainModule` instances.

    :param model: The :class:`~olmo_core.nn.transformer.Transformer` model to train.
    :param optim: The corresponding optimizer config.
    :param rank_microbatch_size: The microbatch size *in tokens* per rank,
        i.e. the number of tokens to process at a time from each rank.

        .. note:: This must evenly divide into the global batch size by a factor of the data
            parallel world size. If this is less than the global batch divided by the data
            parallel world size then gradient accumulation is used.
    :param max_sequence_length: The maximum expected sequence length during training and evaluation.
    :param compile_model: Whether to compile to the model.
    :param float8_config: Float8 configuration for the model.
    :param dp_config: Data parallel configuration for the model.
    :param tp_config: Tensor parallel configuration for the model.
    :param cp_config: Context parallel configuration for the model.
    :param ac_config: Activation checkpointing configuration for the model.
    :param z_loss_multiplier: Use Z-loss with this multiplier.
    :param autocast_precision: Enable AMP with this data type.
    :param max_grad_norm: Clip gradient norms to this value.
    :param scheduler: Optional learning rate scheduler for the optimizer.
    :param device: The device to train on.
    :param state_dict_save_opts: Can be used to override the state dict options used
        when saving a checkpoint.
    :param state_dict_load_opts: Can be used to override the state dict options used
        when loading a checkpoint.
    :param load_key_mapping: Can be used to load a checkpoint where certain parameter have different names.
        This dictionary should map current keys to keys in the checkpoint to be loaded.
    """

    def __init__(
        self,
        model: Transformer,
        optim: OptimConfig,
        rank_microbatch_size: int,
        max_sequence_length: int,
        compile_model: bool = False,
        float8_config: Optional[Float8Config] = None,
        dp_config: Optional[TransformerDataParallelConfig] = None,
        tp_config: Optional[TransformerTensorParallelConfig] = None,
        cp_config: Optional[TransformerContextParallelConfig] = None,
        ep_config: Optional[TransformerExpertParallelConfig] = None,
        ac_config: Optional[TransformerActivationCheckpointingConfig] = None,
        sam_config: Optional[TransformerSAMConfig] = None,
        z_loss_multiplier: Optional[float] = None,
        autocast_precision: Optional[torch.dtype] = None,
        max_grad_norm: Optional[float] = None,
        sam_scheduler: Optional[SAMScheduler] = None,
        scheduler: Optional[Scheduler] = None,
        device: Optional[torch.device] = None,
        state_dict_save_opts: Optional[dist_cp_sd.StateDictOptions] = None,
        state_dict_load_opts: Optional[dist_cp_sd.StateDictOptions] = None,
        load_key_mapping: Optional[Dict[str, str]] = None,
        label_ignore_index: int = -100,
    ):
        super().__init__()

        # Validate some options.
        if rank_microbatch_size % max_sequence_length != 0:
            raise OLMoConfigurationError(
                f"'rank_microbatch_size' ({rank_microbatch_size:,d} tokens) must be divisible by "
                f"'max_sequence_length' ({max_sequence_length:,d} tokens)"
            )

        # SAM perturbs local weights and reads local gradients directly, which is only
        # correct when every rank holds full, plain-tensor parameters. Restrict to data
        # parallelism until the DTensor (TP/CP/EP) cases are worked through.
        if tp_config is not None or cp_config is not None or ep_config is not None:
            raise OLMoConfigurationError(
                "TransformerSAMTrainModule only supports data parallelism; "
                "tensor/context/expert parallel configs are not supported."
            )

        # Build world mesh.
        self.device = device or get_default_device()
        self.world_mesh: Optional[DeviceMesh] = None
        if is_distributed():
            self.world_mesh = build_world_mesh(
                dp=dp_config, tp=tp_config, cp=cp_config, ep=ep_config, device_type=self.device.type
            )
            log.info(f"Data parallel world size = {get_world_size(self.dp_process_group):,d}")
        elif (
            dp_config is not None
            or tp_config is not None
            or ep_config is not None
            or cp_config is not None
        ):
            raise OLMoConfigurationError(
                "Training parallelism configs are only valid for distributed training"
            )

        if (
            ac_config is not None
            and ac_config.mode == TransformerActivationCheckpointingMode.budget
            and not compile_model
        ):
            raise OLMoConfigurationError(
                "Activation checkpointing with 'budget' mode requires compilation to be enabled"
            )

        # Parallelize model.
        self.model = parallelize_model(
            model,
            world_mesh=self.world_mesh,
            device=self.device,
            max_sequence_length=max_sequence_length,
            rank_microbatch_size=rank_microbatch_size,
            compile_model=compile_model,
            float8_config=float8_config,
            dp_config=dp_config,
            tp_config=tp_config,
            cp_config=cp_config,
            ep_config=ep_config,
            ac_config=ac_config,
        )
        self._model_mode: Optional[Literal["train", "eval"]] = None

        # SAM does not currently support FSDP. FSDP's gradient reduce-scatter during
        # backward prevents independent per-GPU perturbations required for m-SAM.
        # Use DDP (DistributedDataParallel) instead.
        if isinstance(self.model, (FSDP, FSDPModule)):
            raise OLMoConfigurationError(
                "TransformerSAMTrainModule does not currently support FSDP/FSDP2. "
                "FSDP's gradient reduce-scatter during backward prevents the independent "
                "per-GPU gradient computation required for m-SAM. Use DDP instead by setting "
                "dp_config to a DDP-based configuration."
            )

        self._dp_config = dp_config
        self._cp_config = cp_config
        self._tp_config = tp_config
        self._ep_config = ep_config
        self.label_ignore_index = label_ignore_index
        self.z_loss_multiplier = z_loss_multiplier
        self.rank_microbatch_size = rank_microbatch_size
        self.max_sequence_length = max_sequence_length
        self.autocast_precision = autocast_precision
        self.max_grad_norm = max_grad_norm
        self.scheduler = scheduler
        self.state_dict_save_opts = state_dict_save_opts or dist_cp_sd.StateDictOptions(
            flatten_optimizer_state_dict=True, cpu_offload=True
        )
        self.state_dict_load_opts = state_dict_load_opts or dist_cp_sd.StateDictOptions(
            flatten_optimizer_state_dict=True, strict=True
        )
        self.load_key_mapping = load_key_mapping
        # Keep a private copy: pre_train() fills in 'm' and the rho scheduler mutates
        # 'rho' in place, which shouldn't leak into the caller's config object.
        self.sam_config = replace(sam_config) if sam_config is not None else TransformerSAMConfig()
        # Build allowed parameter set for SAM perturbation if filtering is requested.
        self._sam_allowed_param_ids: Optional[Set[int]] = self._build_sam_allowed_param_ids()

        # SAM rho scheduler
        self.sam_scheduler: Optional[SAMScheduler] = sam_scheduler

        # Build optimizer(s).
        log.info("Building optimizer...")
        self.optim: Optimizer = optim.build(self.model, strict=True)

    @property
    def dp_process_group(self) -> Optional[dist.ProcessGroup]:
        return None if self.world_mesh is None else get_dp_process_group(self.world_mesh)

    @property
    def eval_batch_spec(self) -> EvalBatchSpec:
        return EvalBatchSpec(
            self.rank_microbatch_size,
            max_sequence_length=self.max_sequence_length,
            #  fixed_sequence_length=self.tp_enabled,
        )

    @property
    def dp_config(self) -> Optional[TransformerDataParallelConfig]:
        return self._dp_config

    @property
    def tp_enabled(self) -> bool:
        return self._tp_config is not None

    @property
    def cp_enabled(self) -> bool:
        return self._cp_config is not None

    @property
    def ep_enabled(self) -> bool:
        return self._ep_config is not None

    @cached_property
    def world_size(self) -> int:
        return get_world_size()

    @cached_property
    def _reduce_divide_factor(self) -> float:
        return get_reduce_divide_factor(self.world_size)

    def pre_train(self):
        # Validate batch size.
        # NOTE: we run this in `pre_train()` instead of, say, `on_attach()` because callbacks
        # like `BatchSizeScheduler` may change the global batch size after the module is attached.
        dp_ws = get_world_size(self.trainer.dp_process_group)
        if self.trainer.global_batch_size % (self.rank_microbatch_size * dp_ws) != 0:
            raise OLMoConfigurationError(
                f"global batch size ({self.trainer.global_batch_size:,d}) must be divisible by "
                f"micro-batch size ({self.rank_microbatch_size:,d}) x DP world size ({dp_ws})"
            )

        device_batch_size = self.trainer.global_batch_size // dp_ws

        # Validate / infer m for SAM.
        # m = per-GPU batch size (in tokens) used to compute each SAM perturbation.
        #   - When m == device_batch_size: standard SAM (one perturbation per GPU per step).
        #   - When m < device_batch_size: m-SAM — the device batch is split into
        #     (device_batch_size / m) SAM groups, each getting its own perturbation.
        #   - m must be a multiple of rank_microbatch_size and must evenly divide
        #     the device batch size.
        if self.sam_config.m is None:
            self.sam_config.m = device_batch_size
        m = self.sam_config.m
        if m < self.rank_microbatch_size:
            raise OLMoConfigurationError(
                f"SAM perturbation batch size m ({m:,d} tokens) must be >= "
                f"rank_microbatch_size ({self.rank_microbatch_size:,d} tokens)"
            )
        if m > device_batch_size:
            raise OLMoConfigurationError(
                f"SAM perturbation batch size m ({m:,d} tokens) must be <= "
                f"device_batch_size ({device_batch_size:,d} tokens)"
            )
        if m % self.rank_microbatch_size != 0:
            raise OLMoConfigurationError(
                f"SAM perturbation batch size m ({m:,d} tokens) must be divisible by "
                f"rank_microbatch_size ({self.rank_microbatch_size:,d} tokens)"
            )
        if device_batch_size % m != 0:
            raise OLMoConfigurationError(
                f"Device batch size ({device_batch_size:,d} tokens) must be divisible by "
                f"SAM perturbation batch size m ({m:,d} tokens)"
            )

        sam_group_size = m // self.rank_microbatch_size
        num_sam_groups = device_batch_size // m
        log.info(
            f"SAM config: m={m:,d} tokens, {sam_group_size} micro-batch(es) per perturbation, "
            f"{num_sam_groups} perturbation(s) per device per step"
        )

    def state_dict(self, *, optim: Optional[bool] = None) -> Dict[str, Any]:
        if optim is None:
            optim = True
        return self._get_state_dict(self.state_dict_save_opts, optim=optim)

    def state_dict_to_load(
        self, metadata: Metadata, *, optim: Optional[bool] = None
    ) -> Dict[str, Any]:
        has_optim_state: bool = False
        for key in metadata.state_dict_metadata.keys():
            if key.startswith("optim."):
                has_optim_state = True
                break

        if optim is None:
            if not has_optim_state:
                log.warning("No optimizer state found in checkpoint")
                optim = False
            else:
                optim = True

        load_opts = self.state_dict_load_opts
        if optim:
            if not has_optim_state:
                raise RuntimeError(
                    "Checkpoint does not contain optimizer state, but 'optim=True' was requested"
                )

            if "optim.param_groups.0.params" in metadata.state_dict_metadata:
                # unflattened optimizer state
                if load_opts.flatten_optimizer_state_dict:
                    log.warning(
                        "Loading checkpoint with an unflattened optimizer state even though "
                        "'flatten_optimizer_state_dict=True' in train module's 'state_dict_load_opts', "
                        "automatically switching to 'flatten_optimizer_state_dict=False'."
                    )
                    load_opts = replace(load_opts, flatten_optimizer_state_dict=False)
            else:
                # flattened optimizer state
                if not load_opts.flatten_optimizer_state_dict:
                    log.warning(
                        "Loading checkpoint with a flattened optimizer state even though "
                        "'flatten_optimizer_state_dict=False' in train module's 'state_dict_load_opts', "
                        "automatically switching to 'flatten_optimizer_state_dict=True'."
                    )
                    load_opts = replace(load_opts, flatten_optimizer_state_dict=True)

        state_dict = self._get_state_dict(load_opts, optim=optim)
        # The save planner flattens nested dicts, so the SAM scheduler appears in the
        # checkpoint metadata as 'sam_scheduler.class', 'sam_scheduler.state.<field>', etc.
        # Reconstruct a matching nested placeholder so the loader reads those entries back
        # (an empty dict would create no read items and load nothing).
        if self.sam_scheduler is not None:
            sam_placeholder = _nested_placeholder_from_keys(
                metadata.state_dict_metadata.keys(), prefix="sam_scheduler"
            )
            if sam_placeholder is not None:
                state_dict["sam_scheduler"] = sam_placeholder
        if self.load_key_mapping is not None:
            swap_param_keys(state_dict, self.load_key_mapping, metadata=metadata)

        if not load_opts.strict:
            # Remove any keys in the 'state_dict' that are not present in the checkpoint.
            pruned_keys = prune_state_dict(state_dict, set(metadata.state_dict_metadata.keys()))
            if pruned_keys:
                log.warning(f"Checkpoint is missing the following keys: {pruned_keys}")

        return state_dict

    def state_dict_to_save(self, *, optim: Optional[bool] = None) -> Dict[str, Any]:
        if optim is None:
            optim = True
        return self._get_state_dict(self.state_dict_save_opts, optim=optim)

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        load_optim = "optim" in state_dict

        if self.load_key_mapping is not None:
            swap_param_keys(state_dict, self.load_key_mapping, reverse=True, quiet=True)

        # NOTE: `dist_cp_sd.set_(model|optimizer)_state_dict()` doesn't respect `strict=False`
        # option with missing keys, so we have to handle that on our own.
        if not self.state_dict_load_opts.strict:
            flatten_optimizer_state_dict = (
                False if not load_optim else ("state" not in state_dict["optim"])
            )
            load_opts = replace(
                self.state_dict_load_opts, flatten_optimizer_state_dict=flatten_optimizer_state_dict
            )
            full_state_dict = self._get_state_dict(load_opts, optim=load_optim)
            merge_state_dicts(state_dict, full_state_dict)

        dist_cp_sd.set_model_state_dict(
            self.model,
            state_dict["model"],
            options=self.state_dict_load_opts,
        )
        gc_cuda()
        if load_optim:
            dist_cp_sd.set_optimizer_state_dict(
                self.model,
                self.optim,
                state_dict["optim"],
                options=self.state_dict_load_opts,
            )
            gc_cuda()
        # Load SAM scheduler state if present. The configured scheduler always wins on
        # which scheduler runs; the checkpoint only restores its internal state.
        sam_sd = state_dict.get("sam_scheduler", None)
        if isinstance(sam_sd, dict) and "class" in sam_sd and "state" in sam_sd:
            if self.sam_scheduler is None:
                log.warning(
                    "Checkpoint contains SAM scheduler state but no SAM scheduler is "
                    "configured, ignoring it"
                )
            else:
                cls = self.sam_scheduler.__class__
                cls_path = f"{cls.__module__}.{cls.__name__}"
                if sam_sd["class"] != cls_path:
                    log.warning(
                        f"Checkpoint SAM scheduler class ({sam_sd['class']}) does not match "
                        f"the configured one ({cls_path}), keeping the configured scheduler's "
                        "fresh state"
                    )
                else:
                    for k, v in sam_sd["state"].items():
                        if hasattr(self.sam_scheduler, k):
                            setattr(self.sam_scheduler, k, v)

    def train_batch(self, batch: Dict[str, Any], dry_run: bool = False):
        # Set model to train mode if it isn't already.
        self._set_model_mode("train")

        # Generate labels.
        if "labels" not in batch:
            batch["labels"] = get_labels(batch, label_ignore_index=self.label_ignore_index)

        # Record how many instances are going to be skipped (masked out).
        if (instance_mask := batch.get("instance_mask")) is not None and not dry_run:
            self.record_metric(
                "train/masked instances (%)", (~instance_mask).float().mean(), ReduceType.mean
            )

        # Calculate and record how many tokens are going to be used in the loss.
        batch_num_tokens = batch["labels"].numel()
        batch_num_tokens_for_loss = move_to_device(
            (batch["labels"] != self.label_ignore_index).sum(), self.device
        )
        self.record_metric(
            "train/masked labels (%)",
            (batch_num_tokens - batch_num_tokens_for_loss) / batch_num_tokens,
            ReduceType.mean,
        )

        # Batch losses to record.
        ce_batch_loss = move_to_device(torch.tensor(0.0), self.device)
        perturbed_ce_batch_loss = move_to_device(torch.tensor(0.0), self.device)
        z_batch_loss: Optional[torch.Tensor] = None
        if self.z_loss_multiplier is not None:
            z_batch_loss = move_to_device(torch.tensor(0.0), self.device)

        # Split into micro-batches.
        if self.rank_microbatch_size < (seq_len := batch["input_ids"].shape[1]):
            raise RuntimeError(
                f"Microbatch size ({self.rank_microbatch_size}) is too small relative to sequence length ({seq_len})"
            )
        micro_batches = split_batch(batch, self.rank_microbatch_size // seq_len)
        num_micro_batches = len(micro_batches)

        # Update rho from scheduler before computing perturbation.
        if self.sam_scheduler is not None and not dry_run:
            self.sam_scheduler.set_rho(self.sam_config, self.trainer)

        # Precompute perturbation config.
        norm_mode = (self.sam_config.normalization or "global").lower()
        rho = torch.tensor(self.sam_config.rho, device=self.device)

        # SAM group size: how many micro-batches per SAM perturbation.
        # m = per-GPU batch size (in tokens) for computing each perturbation.
        assert self.sam_config.m is not None, "sam_config.m must be set before train_batch (see pre_train)"
        sam_group_size = self.sam_config.m // self.rank_microbatch_size
        if num_micro_batches % sam_group_size != 0:
            raise RuntimeError(
                f"Number of micro-batches ({num_micro_batches}) must be divisible by the SAM "
                f"group size ({sam_group_size} micro-batches, m={self.sam_config.m:,d} tokens), "
                "otherwise trailing micro-batches would be silently dropped. This can happen "
                "when the global batch size changes mid-run (e.g. via a batch size scheduler) "
                "to a value incompatible with 'sam_config.m'."
            )
        num_sam_groups = num_micro_batches // sam_group_size

        # SAM algorithm (with optional m-sharding):
        # For each SAM group of ``sam_group_size`` micro-batches:
        #   1. Save RNG state and accumulated descent grads
        #   2. ASCENT: forward+backward without DP sync over the group (accumulate
        #      ascent gradients across micro-batches in the group)
        #   3. Compute perturbation eps = rho * grad / ||grad|| and perturb weights
        #   4. Restore descent grads and RNG state (so descent sees identical dropout)
        #   5. DESCENT: forward+backward at perturbed weights (DDP all-reduce fires
        #      only on the very last micro-batch of the entire device batch)
        #   6. Restore original weights
        # Descent gradients accumulate across all SAM groups. DDP all-reduce fires
        # only on the very last micro-batch of the entire device batch.
        descent_mb_counter = 0
        for group_idx in range(num_sam_groups):
            group_start = group_idx * sam_group_size
            group_end = group_start + sam_group_size

            # --- Save accumulated descent grads and zero for clean ascent ---
            # On the first group saved_descent_grads is empty (p.grad is None).
            # On subsequent groups it preserves the accumulated descent gradients
            # so the ascent backward produces a clean, uncontaminated gradient.
            saved_descent_grads: Dict[int, torch.Tensor] = {}
            for p in self.model.parameters():
                if p.grad is not None:
                    saved_descent_grads[id(p)] = p.grad
                    p.grad = None

            # --- Save RNG state for replay during descent ---
            rng_state = get_rng_state()

            # --- ASCENT: accumulate gradients over this SAM group without DP sync ---
            group_num_tokens_for_loss = move_to_device(torch.tensor(0.0), self.device)
            with self._grad_sync_disabled():
                for mb_idx in range(group_start, group_end):
                    input_ids, labels, model_kwargs = self._prepare_batch(
                        micro_batches[mb_idx], keep_keys=True
                    )
                    group_num_tokens_for_loss += (labels != self.label_ignore_index).sum()
                    _, loss, ce_loss, _ = self.model_forward(
                        input_ids,
                        labels=labels,
                        ignore_index=self.label_ignore_index,
                        loss_reduction="sum",
                        z_loss_multiplier=self.z_loss_multiplier,
                        loss_div_factor=batch_num_tokens_for_loss,
                        return_logits=False,
                        **model_kwargs,
                    )
                    ce_batch_loss += get_local_tensor(ce_loss.detach())
                    del ce_loss
                    loss.backward()

            # --- Compute and apply perturbation from accumulated ascent gradient ---
            scale_global: torch.Tensor = rho
            if norm_mode == "global":
                gnorm = self._grad_global_norm()
                scale_global = (rho / (gnorm + self.sam_config.eps)).to(self.device)
            # The ascent loss was divided by the full device batch's token count while the
            # gradient only covers this group's tokens, so for the unnormalized mode rescale
            # to make the perturbation rho times the group-mean gradient, independent of how
            # many SAM groups the device batch is split into. The normalized modes cancel
            # any scalar factor.
            scale_unnormalized: torch.Tensor = rho
            if norm_mode == "none":
                scale_unnormalized = rho * (
                    batch_num_tokens_for_loss / group_num_tokens_for_loss.clamp(min=1.0)
                )
            # Save original weights for an exact restore. Perturb-then-subtract leaves
            # floating point rounding residue on every weight each step, which is
            # significant for pure-bf16 parameters and would let DDP ranks slowly drift
            # apart, since each rank perturbs differently and DDP never re-syncs weights.
            perturbations: list[Tuple[nn.Parameter, torch.Tensor]] = []
            for p in self.model.parameters():
                if p.grad is None:
                    continue
                if self._sam_allowed_param_ids is not None and id(p) not in self._sam_allowed_param_ids:
                    continue
                if norm_mode == "none":
                    scale_p = scale_unnormalized
                elif norm_mode == "layer":
                    p_norm = p.grad.detach().norm(2)
                    scale_p = (rho / (p_norm + self.sam_config.eps)).to(self.device)
                elif norm_mode == "global":
                    scale_p = scale_global
                else:
                    raise OLMoConfigurationError(f"Invalid SAM normalization mode: {norm_mode}")
                eps_w = p.grad.detach() * scale_p.to(dtype=p.dtype)
                perturbations.append((p, p.data.clone()))
                p.data.add_(eps_w)

            # --- Restore accumulated descent grads (discard ascent grads) ---
            for p in self.model.parameters():
                if id(p) in saved_descent_grads:
                    p.grad = saved_descent_grads[id(p)]
                else:
                    p.grad = None
            del saved_descent_grads

            # --- Restore RNG state so descent sees identical dropout masks ---
            set_rng_state(rng_state)
            del rng_state

            # --- DESCENT: forward+backward at perturbed weights ---
            # DDP all-reduce fires only on the very last micro-batch of the entire
            # device batch (descent_mb_counter == num_micro_batches - 1).
            for mb_idx in range(group_start, group_end):
                with self._train_microbatch_context(descent_mb_counter, num_micro_batches):
                    input_ids, labels, model_kwargs = self._prepare_batch(micro_batches[mb_idx])

                    _, loss, ce_loss, z_loss = self.model_forward(
                        input_ids,
                        labels=labels,
                        ignore_index=self.label_ignore_index,
                        loss_reduction="sum",
                        z_loss_multiplier=self.z_loss_multiplier,
                        loss_div_factor=batch_num_tokens_for_loss,
                        return_logits=False,
                        **model_kwargs,
                    )

                    perturbed_ce_batch_loss += get_local_tensor(ce_loss.detach())
                    del ce_loss
                    if z_batch_loss is not None:
                        assert z_loss is not None
                        z_batch_loss += get_local_tensor(z_loss.detach())
                        del z_loss

                    loss.backward()

                descent_mb_counter += 1

            # --- Restore original weights exactly ---
            for p, original_data in perturbations:
                p.data.copy_(original_data)
            del perturbations

        del batch  # In case this helps with memory utilization.

        self.model.post_batch(dry_run=dry_run)

        if dry_run:
            self.model.reset_auxiliary_metrics()
            return

        # Record loss metrics.
        if isinstance(self.optim, SkipStepOptimizer):
            # Need to reduce the loss right away for the SkipStepOptimizer.
            if is_distributed():
                ce_batch_loss.div_(self._reduce_divide_factor)
                dist.all_reduce(ce_batch_loss)
                ce_batch_loss.div_(self.world_size)
                ce_batch_loss.mul_(self._reduce_divide_factor)
            self.record_ce_loss(ce_batch_loss)
            if is_distributed():
                perturbed_ce_batch_loss.div_(self._reduce_divide_factor)
                dist.all_reduce(perturbed_ce_batch_loss)
                perturbed_ce_batch_loss.div_(self.world_size)
                perturbed_ce_batch_loss.mul_(self._reduce_divide_factor)
            self.record_metric("Perturbed CE Loss", perturbed_ce_batch_loss, namespace="train")
            self.optim.latest_loss = ce_batch_loss
        else:
            self.record_ce_loss(ce_batch_loss, ReduceType.mean)
            self.record_metric("Perturbed CE Loss", perturbed_ce_batch_loss, ReduceType.mean, namespace="train")
        if z_batch_loss is not None:
            assert self.z_loss_multiplier is not None
            self.record_metric(
                "Z loss",
                z_batch_loss,
                ReduceType.mean,
                namespace="train",
            )
            self.record_metric(
                "Z loss unscaled",
                z_batch_loss / self.z_loss_multiplier,
                ReduceType.mean,
                namespace="train",
            )

        # And additional metrics.
        for metric_name, (metric_val, reduction) in self.model.compute_auxiliary_metrics(
            reset=True
        ).items():
            self.record_metric(
                metric_name,
                metric_val,
                reduction,
                namespace="train",
            )

    def eval_batch(
        self,
        batch: Dict[str, Any],
        labels: Optional[torch.Tensor] = None,
        return_logits: bool = True,
    ) -> Union[torch.Tensor, LMOutputWithLoss]:
        # TODO: (epwalsh) Currently all of our evaluators require the full logits locally,
        # but when we're using CP/TP we usually can't materialize the full logits locally (due to OOMs).
        # However we could at least support in-loop PPL evals with a little work in the evaluator
        # code to handle the sharded logits.
        if self.cp_enabled:
            raise RuntimeError(
                f"{self.__class__.__name__}.eval_batch() does not support context parallelism yet, "
                "please disable in-loop evals"
            )
        if self.tp_enabled:
            raise RuntimeError(
                f"{self.__class__.__name__}.eval_batch() does not support tensor parallelism yet, "
                "please disable in-loop evals"
            )

        input_ids, labels, model_kwargs = self._prepare_batch(batch, labels)

        self._set_model_mode("eval")

        with self._eval_batch_context():
            return self.model_forward(
                input_ids,
                labels=labels,
                ignore_index=self.label_ignore_index,
                loss_reduction="none",
                return_logits=return_logits,
                **model_kwargs,
            )

    def optim_step(self):
        # Maybe clip gradients.
        if self.max_grad_norm is not None:
            grad_norm = self._clip_grad_norm(self.max_grad_norm)
            # NOTE: grad norm is already reduced over ranks, so we set `reduce_type` to `None`.
            self.trainer.record_metric(
                "total grad norm", grad_norm, reduce_type=None, namespace="optim"
            )
            if isinstance(self.optim, SkipStepOptimizer):
                self.optim.latest_grad_norm = grad_norm

        # Maybe adjust learning rate.
        if self.scheduler is not None:
            for group_idx, group in enumerate(self.optim.param_groups):
                new_lr = self.scheduler.set_lr(group, self.trainer)
                self.trainer.record_metric(f"LR (group {group_idx})", new_lr, namespace="optim")

        # Log SAM rho similar to LR so it's surfaced in WandB
        if self.sam_scheduler is not None:
            self.trainer.record_metric("SAM rho", float(self.sam_config.rho), namespace="optim")

        # Step optimizer.
        self.optim.step()
        if isinstance(self.optim, SkipStepOptimizer):
            self.record_metric("step skipped", self.optim.step_skipped, namespace="optim")

        self.model.post_optim_step()

    def zero_grads(self):
        self.optim.zero_grad(set_to_none=True)

    def model_forward(
        self, input_ids: torch.Tensor, labels: Optional[torch.Tensor] = None, **kwargs
    ) -> Union[torch.Tensor, LMOutputWithLoss]:
        """
        Run a forward pass on a micro-batch, returning the logits.
        """
        with self._model_forward_context():
            return self.model(input_ids, labels=labels, **kwargs)

    def num_flops_per_token(self, seq_len: int) -> int:
        return 2 * self.model.num_flops_per_token(seq_len)

    @contextlib.contextmanager
    def _train_microbatch_context(
        self, micro_batch_idx: int, num_micro_batches: int
    ) -> Generator[None, None, None]:
        # Only sync gradients on the final micro-batch of the device batch.
        is_last_mb = micro_batch_idx == num_micro_batches - 1
        with contextlib.ExitStack() as stack:
            if not is_last_mb:
                stack.enter_context(self._grad_sync_disabled())
            yield

    @contextlib.contextmanager
    def _eval_batch_context(self) -> Generator[None, None, None]:
        with contextlib.ExitStack() as stack:
            stack.enter_context(torch.no_grad())
            yield

    @contextlib.contextmanager
    def _model_forward_context(self) -> Generator[None, None, None]:
        with contextlib.ExitStack() as stack:
            if self.autocast_precision is not None:
                stack.enter_context(torch.autocast(self.device.type, dtype=self.autocast_precision))
            yield

    @contextlib.contextmanager
    def _grad_sync_disabled(self) -> Generator[None, None, None]:
        """
        Disable data-parallel gradient sync for forwards/backwards run within this context.

        olmo_core's ``apply_ddp`` uses the composable ``replicate()`` API, which swaps the
        model's class to a marker type that is NOT a ``DistributedDataParallel`` instance,
        so an ``isinstance(self.model, DDP)`` check alone never matches and gradients would
        be all-reduced on every backward — silently turning the per-rank (m-)SAM
        perturbations into globally-averaged ones. The classic wrapper is still handled in
        case it is ever used.
        """
        if isinstance(self.model, DDP):
            with self.model.no_sync():
                yield
        elif _ComposableDDP is not None and isinstance(self.model, _ComposableDDP):
            self.model.set_requires_gradient_sync(False)
            try:
                yield
            finally:
                self.model.set_requires_gradient_sync(True)
        else:
            yield

    def _build_sam_allowed_param_ids(self) -> Optional[Set[int]]:
        types_str = (self.sam_config.sam_parameter_types or "").strip()
        if not types_str:
            return None
        type_names = {t.strip().lower() for t in types_str.split(",") if t.strip()}
        allowed: Set[int] = set()
        for module in self.model.modules():
            if module.__class__.__name__.lower() in type_names:
                for p in module.parameters(recurse=False):
                    allowed.add(id(p))
        return allowed

    def _get_state_dict(
        self, sd_options: dist_cp_sd.StateDictOptions, optim: bool = True
    ) -> Dict[str, Any]:
        state_dict: Dict[str, Any] = {
            "model": dist_cp_sd.get_model_state_dict(self.model, options=sd_options),
        }
        if optim:
            state_dict["optim"] = dist_cp_sd.get_optimizer_state_dict(
                self.model, self.optim, options=sd_options
            )
        # Persist SAM scheduler if present.
        if self.sam_scheduler is not None:
            # Serialize by class path and state for simple, robust loading.
            cls = self.sam_scheduler.__class__
            state_dict["sam_scheduler"] = {
                "class": f"{cls.__module__}.{cls.__name__}",
                "state": self.sam_scheduler.as_state(),
            }
        return state_dict

    def _clip_grad_norm(
        self, max_grad_norm: float, norm_type: float = 2.0, foreach: Optional[bool] = None
    ) -> torch.Tensor:
        parameters = [p for p in self.model.parameters()]
        grads = [p.grad for p in parameters if p.grad is not None]

        total_norm = nn.utils.get_total_norm(
            grads, norm_type=norm_type, error_if_nonfinite=False, foreach=foreach
        )

        # A DTensor norm has partial placements and must be reduced to get the true value.
        if isinstance(total_norm, DTensor):
            total_norm = total_norm.full_tensor()

        torch.nn.utils.clip_grads_with_norm_(parameters, max_grad_norm, total_norm, foreach=foreach)
        return total_norm

    def _grad_global_norm(
        self, norm_type: float = 2.0, foreach: Optional[bool] = None
    ) -> torch.Tensor:
        # When sam_parameter_types restricts which parameters receive perturbations,
        # compute the gradient norm only over those parameters so the perturbation
        # magnitude is not diluted by gradients from non-perturbed parameters.
        allowed = self._sam_allowed_param_ids
        if allowed is not None:
            grads = [p.grad for p in self.model.parameters() if p.grad is not None and id(p) in allowed]
        else:
            grads = [p.grad for p in self.model.parameters() if p.grad is not None]
        if not grads:
            return torch.tensor(0.0, device=self.device)
        total_norm = nn.utils.get_total_norm(
            grads, norm_type=norm_type, error_if_nonfinite=False, foreach=foreach
        )
        if isinstance(total_norm, DTensor):
            total_norm = total_norm.full_tensor()
        return total_norm

    def _prepare_batch(
        self,
        batch: Dict[str, Any],
        labels: Optional[torch.Tensor] = None,
        *,
        keep_keys: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, Any]]:
        if keep_keys:
            input_ids = batch["input_ids"]
            labels_val = labels if labels is not None else batch.get("labels", None)
            model_kwargs = {k: v for k, v in batch.items() if k not in ("input_ids", "labels")}
        else:
            input_ids = batch.pop("input_ids")
            labels_val = labels if labels is not None else batch.pop("labels", None)
            model_kwargs = batch
        if "doc_lens" in batch and "max_doc_lens" in batch:
            log_once(log, "intra-document masking enabled")
        return input_ids, labels_val, model_kwargs

    def _set_model_mode(self, mode: Literal["train", "eval"]):
        if self._model_mode != mode:
            if mode == "train":
                self.model.train()
            elif mode == "eval":
                self.model.eval()
            else:
                raise ValueError(f"Invalid model mode: {mode}")
            self._model_mode = mode
