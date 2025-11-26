import contextlib
import logging
from dataclasses import replace
from functools import cached_property
from typing import Any, Dict, Generator, Literal, Optional, Tuple, Union, Set, TYPE_CHECKING
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from importlib import import_module
from math import cos, pi

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

from olmo_core.data.utils import get_labels, split_batch
from olmo_core.distributed.checkpoint import (
    merge_state_dicts,
    prune_state_dict,
    swap_param_keys,
)
from olmo_core.distributed.parallel import (
    DataParallelType,
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
        if self.decay_alpha < 0:
            raise OLMoConfigurationError("'decay_alpha' must be >= 0.")

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
        return {
            "decay_alpha": self.decay_alpha,
            "warmup": self.warmup,
            "_initial_rho": self._initial_rho,
        }

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
            step_from_end = max(t_max - current, 0)
            return float(_lr_linear_decay(initial_rho, step_from_end, decay, eta_min))

        return float(initial_rho)


@dataclass
class SAMCosineScheduler(SAMScheduler):
    """
    Cosine decay from initial_rho to decay_alpha * initial_rho with optional warmup (0 -> initial_rho).
    """

    def get_rho(self, initial_rho: float, current: int, t_max: int) -> float:
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
        self.sam_config = sam_config or TransformerSAMConfig()
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
        # Validate / infer m for m-SAM.
        if self.sam_config.m is None:
            self.sam_config.m = dp_ws
        elif self.sam_config.m != dp_ws:
            raise OLMoConfigurationError(
                f"For m-SAM, 'm' ({self.sam_config.m}) must equal DP world size ({dp_ws}) when "
                "sam_micro_batch equals the device batch size"
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
        # Add placeholder for SAM scheduler if present in checkpoint.
        if "sam_scheduler" in metadata.state_dict_metadata:
            state_dict["sam_scheduler"] = {}
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
        # Load SAM scheduler if present
        sam_sd = state_dict.get("sam_scheduler", None)
        if isinstance(sam_sd, dict) and "class" in sam_sd and "state" in sam_sd:
            try:
                cls_path: str = sam_sd["class"]
                mod_name, cls_name = cls_path.rsplit(".", 1)
                mod = import_module(mod_name)
                cls = getattr(mod, cls_name)
                if isinstance(self.sam_scheduler, cls):
                    # Update existing instance state
                    for k, v in sam_sd["state"].items():
                        setattr(self.sam_scheduler, k, v)
                else:
                    self.sam_scheduler = cls.from_state(sam_sd["state"])
            except Exception as e:
                log.warning(f"Failed to load SAM scheduler from checkpoint ({e}), continuing without it")

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
        initial_ce_batch_loss = move_to_device(torch.tensor(0.0), self.device)
        ascent_ce_batch_loss = move_to_device(torch.tensor(0.0), self.device)
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

        # Update rho from scheduler before computing perturbation
        if self.sam_scheduler is not None and not dry_run:
            self.sam_scheduler.set_rho(self.sam_config, self.trainer)

        # m-SAM ascent: compute local gradient without DP sync and perturb parameters.
        perturbations: Optional[list[Tuple[nn.Parameter, torch.Tensor]]] = []
        with self._sam_no_sync_context():
            for micro_batch_idx, micro_batch in enumerate(micro_batches):
                # No-sync context prevents gradient reduction across ranks here.
                input_ids, labels, model_kwargs = self._prepare_batch(micro_batch, keep_keys=True)
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
                initial_ce_batch_loss += get_local_tensor(ce_loss.detach())
                del ce_loss
                loss.backward()

        # Compute normalization scalars and apply perturbation based on config.
        norm_mode = (self.sam_config.normalization or "global").lower()
        rho = torch.tensor(self.sam_config.rho, device=self.device)
        scale_global: torch.Tensor = rho
        if norm_mode == "global":
            gnorm = self._grad_global_norm()
            scale_global = (rho / (gnorm + self.sam_config.eps)).to(self.device)
        perturbations = []
        for p in self.model.parameters():
            if p.grad is None:
                continue
            if self._sam_allowed_param_ids is not None and id(p) not in self._sam_allowed_param_ids:
                continue
            if norm_mode == "none":
                scale_p = rho
            elif norm_mode == "layer":
                p_norm = p.grad.detach().norm(2)
                scale_p = (rho / (p_norm + self.sam_config.eps)).to(self.device)
            elif norm_mode == "global":
                scale_p = scale_global
            else:
                raise OLMoConfigurationError(f"Invalid SAM normalization mode: {norm_mode}")
            eps_w = p.grad.detach() * scale_p.to(dtype=p.dtype)
            p.data.add_(eps_w)
            perturbations.append((p, eps_w))

        # Train one micro-batch at a time.
        for micro_batch_idx, micro_batch in enumerate(micro_batches):
            with self._train_microbatch_context(micro_batch_idx, num_micro_batches):
                input_ids, labels, model_kwargs = self._prepare_batch(micro_batch)

                # Run forward pass, get losses.
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

                # Update total batch CE and Z loss.
                ascent_ce_batch_loss += get_local_tensor(ce_loss.detach())
                del ce_loss
                if z_batch_loss is not None:
                    assert z_loss is not None
                    z_batch_loss += get_local_tensor(z_loss.detach())
                    del z_loss

                # Run backward pass.
                loss.backward()

        # Restore parameters (remove perturbation).
        if perturbations:
            for p, eps_w in perturbations:
                p.data.sub_(eps_w)
            perturbations.clear()

        del batch  # In case this helps with memory utilization.

        self.model.post_batch(dry_run=dry_run)

        if dry_run:
            self.model.reset_auxiliary_metrics()
            return

        # Record loss metrics.
        if isinstance(self.optim, SkipStepOptimizer):
            # Need to reduce the loss right away for the SkipStepOptimizer.
            if is_distributed():
                initial_ce_batch_loss.div_(self._reduce_divide_factor)
                dist.all_reduce(initial_ce_batch_loss)
                initial_ce_batch_loss.div_(self.world_size)
                initial_ce_batch_loss.mul_(self._reduce_divide_factor)
            self.record_ce_loss(initial_ce_batch_loss)
            # Reduce ascent CE loss the same way
            if is_distributed():
                ascent_ce_batch_loss.div_(self._reduce_divide_factor)
                dist.all_reduce(ascent_ce_batch_loss)
                ascent_ce_batch_loss.div_(self.world_size)
                ascent_ce_batch_loss.mul_(self._reduce_divide_factor)
            self.record_metric("Ascent CE Loss", ascent_ce_batch_loss, namespace="train")
            self.optim.latest_loss = initial_ce_batch_loss
        else:
            self.record_ce_loss(initial_ce_batch_loss, ReduceType.mean)
            self.record_metric("Ascent CE Loss", ascent_ce_batch_loss, ReduceType.mean, namespace="train")
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
        self, batch: Dict[str, Any], labels: Optional[torch.Tensor] = None
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
        is_last_mb = micro_batch_idx == num_micro_batches - 1
        with contextlib.ExitStack() as stack:
            if isinstance(self.model, FSDPModule):
                assert self.dp_config is not None
                # On the last backward FSDP waits on pending gradient reduction and clears internal data
                # data structures for backward prefetching.
                self.model.set_is_last_backward(is_last_mb)
                # For HSDP we can delay the gradients all-reduce until the final micro-batch.
                if self.dp_config.name == DataParallelType.hsdp:
                    self.model.set_requires_all_reduce(is_last_mb)
            elif isinstance(self.model, DDP):
                # For DDP, only sync gradients on the final micro-batch.
                if not is_last_mb:
                    stack.enter_context(self.model.no_sync())

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
    def _sam_no_sync_context(self) -> Generator[None, None, None]:
        with contextlib.ExitStack() as stack:
            if isinstance(self.model, DDP):
                stack.enter_context(self.model.no_sync())
                yield
                return
            if isinstance(self.model, FSDPModule) and hasattr(self.model, "set_requires_all_reduce"):
                self.model.set_requires_all_reduce(False)
                try:
                    yield
                finally:
                    self.model.set_requires_all_reduce(True)
                return
            yield

    def _build_sam_allowed_param_ids(self) -> Optional[Set[int]]:
        types_str = (self.sam_config.sam_parameter_types or "").strip()
        if not types_str:
            return None
        if isinstance(self.model, (FSDP, FSDPModule)):
            raise OLMoConfigurationError(
                "sam_parameter_types is not supported with FSDP-wrapped models due to parameter flattening"
            )
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
        if isinstance(self.model, FSDP):
            return self.model.clip_grad_norm_(max_grad_norm)

        # Adapted from https://github.com/pytorch/torchtitan/blob/2a4437014e66bcf88a3f0419b816266e6326d539/torchtitan/utils.py#L348

        parameters = [p for p in self.model.parameters()]
        grads = [p.grad for p in parameters if p.grad is not None]

        total_norm = nn.utils.get_total_norm(
            grads, norm_type=norm_type, error_if_nonfinite=False, foreach=foreach
        )

        # If total_norm is a DTensor, the placements must be `torch.distributed._tensor.ops.math_ops._NormPartial`.
        # We can simply reduce the DTensor to get the total norm in this tensor's process group
        # and then convert it to a local tensor.
        # NOTE: It has two purposes:
        #       1. to make sure the total norm is computed correctly when PP is used (see below)
        #       2. to return a reduced total_norm tensor whose .item() would return the correct value
        if isinstance(total_norm, DTensor):
            # Will reach here if any non-PP parallelism is used.
            # If only using PP, total_norm will be a local tensor.
            total_norm = total_norm.full_tensor()

        torch.nn.utils.clip_grads_with_norm_(parameters, max_grad_norm, total_norm, foreach=foreach)
        return total_norm

    def _grad_global_norm(
        self, norm_type: float = 2.0, foreach: Optional[bool] = None
    ) -> torch.Tensor:
        parameters = [p for p in self.model.parameters()]
        grads = [p.grad for p in parameters if p.grad is not None]
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
