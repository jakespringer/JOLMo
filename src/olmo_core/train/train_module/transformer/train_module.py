import contextlib
import logging
from dataclasses import replace
from functools import cached_property
from typing import Any, Dict, Generator, Literal, Optional, Tuple, Union

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
from olmo_core.nn.lm_head import LMOutputWithLoss
from olmo_core.nn.peft import PEFTConfig
from olmo_core.nn.transformer import Transformer
from olmo_core.nn.transformer.config import TransformerActivationCheckpointingMode
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
    TransformerEMAConfig,
    TransformerExpertParallelConfig,
    TransformerTensorParallelConfig,
)

log = logging.getLogger(__name__)


class TransformerTrainModule(TrainModule):
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
        z_loss_multiplier: Optional[float] = None,
        autocast_precision: Optional[torch.dtype] = None,
        max_grad_norm: Optional[float] = None,
        scheduler: Optional[Scheduler] = None,
        device: Optional[torch.device] = None,
        state_dict_save_opts: Optional[dist_cp_sd.StateDictOptions] = None,
        state_dict_load_opts: Optional[dist_cp_sd.StateDictOptions] = None,
        load_key_mapping: Optional[Dict[str, str]] = None,
        label_ignore_index: int = -100,
        ema: Optional[TransformerEMAConfig] = None,
        peft: Optional[PEFTConfig] = None,
    ):
        super().__init__()

        # Validate some options.
        if rank_microbatch_size % max_sequence_length != 0:
            raise OLMoConfigurationError(
                f"'rank_microbatch_size' ({rank_microbatch_size:,d} tokens) must be divisible by "
                f"'max_sequence_length' ({max_sequence_length:,d} tokens)"
            )

        # Apply PEFT model transforms BEFORE parallelization. This lets
        # FSDP/TP/AC/compile see the post-LoRA module graph and shard or
        # parallelize the added adapter modules just like the original
        # weights. Frozen base parameters are picked up by
        # OptimConfig.build_groups (it skips requires_grad=False) when
        # the optimizer is built below.
        #
        # LoRA's design uses property delegation on ``LoRALinear.weight``
        # and a no-op ``reset_parameters`` on its ``A``/``B`` branches so
        # the model's standard init pass leaves adapter values intact.
        # If a future PEFT method needs a more aggressive skip (e.g.,
        # leaving entire components uninitialized), wire a predicate
        # through ``Transformer.init_weights`` at that time.
        self.peft = peft
        peft_key_mapping: Optional[Dict[str, str]] = None
        if self.peft is not None:
            self.peft.apply_model_transforms(model)
            peft_key_mapping = self.peft.checkpoint_key_mapping(model)

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
        # Merge any user-provided load_key_mapping with the mapping
        # contributed by PEFT model transforms (e.g. LoRA renames
        # ``*.w_q.weight`` → ``*.w_q.base.weight``). Conflicts here
        # already raised inside ``PEFTConfig.checkpoint_key_mapping``.
        if peft_key_mapping:
            merged_mapping: Dict[str, str] = dict(peft_key_mapping)
            if load_key_mapping:
                # User-provided overrides PEFT defaults.
                merged_mapping.update(load_key_mapping)
            self.load_key_mapping = merged_mapping
        else:
            self.load_key_mapping = load_key_mapping

        # Build optimizer(s).
        log.info("Building optimizer...")
        self.optim: Optimizer = optim.build(self.model, strict=True)

        # Weight EMA configuration. Shadow tensors are allocated *lazily* on first
        # use, so that when training continues from a base model loaded after
        # `__init__` (via `load_path`), the EMAs are seeded from the loaded base
        # weights rather than the random-init weights present at `__init__` time.
        # Lazy allocation also correctly handles checkpoint resume: a load that
        # provides EMA state populates the shadows on first allocation, while a
        # load that does not populate them lets training-time updates start from
        # the just-loaded model state.
        self.ema_config: Optional[TransformerEMAConfig] = ema
        self._ema_shadows: List[Dict[str, torch.Tensor]] = []
        self._ema_step_counter: int = 0
        self._ema_loaded_from_checkpoint: bool = False
        if self.ema_config is not None:
            log.info(
                f"Configured {len(self.ema_config.decays)} weight EMA shadow(s) "
                f"with decays={list(self.ema_config.decays)}, "
                f"offload_to_cpu={self.ema_config.offload_to_cpu}; "
                "shadows will be allocated on first use."
            )

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

    def on_attach(self):
        # Auto-install the PEFT gradient callback when any gradient
        # transforms are configured. The callback runs every transform
        # at ``pre_optim_step`` (before grad clipping inside
        # ``optim_step``). This is a no-op when ``peft`` is None or has
        # no gradient transforms.
        if self.peft is not None:
            self.peft.install_gradient_callback(self.trainer)

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

        # Ensure EMA shadows are allocated before training starts. In the no-load
        # path this is the first allocation; in the load path :meth:`load_state_dict`
        # has already allocated and seeded them, so this is a no-op.
        if self.ema_config is not None:
            self._ensure_ema_shadows_allocated()

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

        # Drop EMA template entries that aren't present in this checkpoint, so that
        # resuming from a pre-EMA (or differently-configured-EMA) checkpoint succeeds
        # under strict load mode. Missing shadows will retain their initial values.
        if self.ema_config is not None:
            checkpoint_keys = metadata.state_dict_metadata.keys()
            for i in range(len(self.ema_config.decays)):
                top_key = f"ema_{i}"
                prefix = f"{top_key}."
                if not any(k.startswith(prefix) for k in checkpoint_keys):
                    state_dict.pop(top_key, None)
                    log.warning(
                        f"Checkpoint does not contain EMA shadow '{top_key}'; "
                        "the live shadow will be left at its initial value."
                    )

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

        # Capture which EMA keys were present in the *input* state_dict — i.e., came
        # from the actual checkpoint, not from a strict=False merge below. We will
        # consume those entries after the model/optim load.
        ema_input_keys: Dict[int, Dict[str, torch.Tensor]] = {}
        if self.ema_config is not None:
            for i in range(len(self.ema_config.decays)):
                key = f"ema_{i}"
                ema_sd = state_dict.get(key)
                if ema_sd:
                    ema_input_keys[i] = ema_sd

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

        # Load EMA shadows if both the train module and the checkpoint have them.
        # Use the keys captured *before* the strict=False merge above so that EMA
        # template entries re-added by `merge_state_dicts` don't masquerade as real
        # checkpoint data. For any EMA index *not* present in the checkpoint, re-seed
        # the shadow from the live model — which has just been populated above by
        # `set_model_state_dict`, so the new shadow will track the just-loaded
        # weights. Doing this here (rather than in `pre_train`) ensures that any
        # checkpoint write between this load and `train_module.pre_train` (e.g. an
        # explicit step-0 pre-train save) sees consistent EMA state.
        if self.ema_config is not None:
            had_any = False
            for i in range(len(self.ema_config.decays)):
                if i in ema_input_keys:
                    self._load_shadow_from_state(i, ema_input_keys[i])
                    had_any = True
                else:
                    if self.ema_config.init_from_model and self._ema_shadows:
                        log.warning(
                            f"Checkpoint is missing EMA shadow 'ema_{i}'; "
                            "re-seeding it from the just-loaded model parameters."
                        )
                        # Replace storage; this captures the post-load model state.
                        self._ema_shadows[i] = self._init_ema_shadow()
                    else:
                        log.warning(
                            f"Checkpoint is missing EMA shadow 'ema_{i}'; "
                            "leaving its initial value unchanged "
                            f"(init_from_model={self.ema_config.init_from_model})."
                        )
            if had_any:
                self._ema_loaded_from_checkpoint = True

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
                ce_batch_loss += get_local_tensor(ce_loss.detach())
                del ce_loss
                if z_batch_loss is not None:
                    assert z_loss is not None
                    z_batch_loss += get_local_tensor(z_loss.detach())
                    del z_loss

                # Run backward pass.
                loss.backward()

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
            self.optim.latest_loss = ce_batch_loss
        else:
            self.record_ce_loss(ce_batch_loss, ReduceType.mean)
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

        # Step optimizer.
        self.optim.step()
        if isinstance(self.optim, SkipStepOptimizer):
            self.record_metric("step skipped", self.optim.step_skipped, namespace="optim")

        # Update weight EMA shadows (no-op if no EMA configured).
        self._ema_update()

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
        return self.model.num_flops_per_token(seq_len)

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
        if self.ema_config is not None:
            self._ensure_ema_shadows_allocated()
            for i in range(len(self.ema_config.decays)):
                state_dict[f"ema_{i}"] = self._wrap_shadow_as_dtensor_state(i)
        return state_dict

    # ------------------------------------------------------------------
    # Weight EMA support.
    # ------------------------------------------------------------------

    def _ema_target_device(self) -> torch.device:
        assert self.ema_config is not None
        return torch.device("cpu") if self.ema_config.offload_to_cpu else self.device

    def _init_ema_shadow(self) -> Dict[str, torch.Tensor]:
        """Allocate one EMA shadow dict mapping parameter name -> local-shard tensor."""
        assert self.ema_config is not None
        target_device = self._ema_target_device()
        shadow: Dict[str, torch.Tensor] = {}
        for name, p in self.model.named_parameters():
            local = get_local_tensor(p).detach()
            if self.ema_config.init_from_model:
                shadow[name] = local.to(device=target_device, copy=True)
            else:
                shadow[name] = torch.zeros_like(local, device=target_device)
        return shadow

    def _ensure_ema_shadows_allocated(self) -> None:
        """
        Lazily allocate EMA shadow tensors from the current model state. Called from
        the first-use entry points: state-dict construction, EMA update, swap, and
        :meth:`pre_train`. Subsequent calls are no-ops.
        """
        if self.ema_config is None or self._ema_shadows:
            return
        self._ema_shadows = [self._init_ema_shadow() for _ in self.ema_config.decays]
        shadow_bytes = sum(
            t.numel() * t.element_size() for t in self._ema_shadows[0].values()
        )
        log.info(
            f"Allocated {len(self._ema_shadows)} weight EMA shadow(s) "
            f"(per-shadow local size = {shadow_bytes / (1024 ** 2):.1f} MiB)"
        )

    @torch.no_grad()
    def _ema_update(self) -> None:
        """Apply ``shadow <- decay * shadow + (1 - decay) * param`` to all EMA shadows."""
        if self.ema_config is None:
            return
        self._ensure_ema_shadows_allocated()
        self._ema_step_counter += 1
        if self._ema_step_counter % self.ema_config.update_every_n_steps != 0:
            return
        for decay, shadow in zip(self.ema_config.decays, self._ema_shadows):
            weight = 1.0 - decay
            for name, p in self.model.named_parameters():
                live_local = get_local_tensor(p).detach()
                tgt = shadow[name]
                if tgt.device != live_local.device:
                    # Cross-device path (typically GPU -> CPU offload). Use a
                    # synchronous copy to avoid races with the subsequent host-side
                    # lerp_. The CPU-offload path is already bandwidth-bound, so the
                    # extra sync is negligible.
                    live_local = live_local.to(tgt.device)
                if tgt.dtype != live_local.dtype:
                    live_local = live_local.to(dtype=tgt.dtype)
                tgt.lerp_(live_local, weight)

    @contextlib.contextmanager
    def use_ema_params(self, ema_idx: int) -> Generator[None, None, None]:
        """
        Temporarily swap the live model's parameter shards with EMA shadow ``ema_idx``.
        The original parameters are restored when the context manager exits, even on
        exceptions. Memory cost while the block is active is one extra copy of the
        local parameter shards on the live device.
        """
        if self.ema_config is None:
            raise RuntimeError("No EMA shadows are configured on this train module")
        self._ensure_ema_shadows_allocated()
        if not (0 <= ema_idx < len(self._ema_shadows)):
            raise IndexError(
                f"EMA index {ema_idx} out of range (have {len(self._ema_shadows)} shadows)"
            )
        shadow = self._ema_shadows[ema_idx]
        saved: Dict[str, torch.Tensor] = {}
        # NOTE: model parameters typically have ``requires_grad=True``, so their
        # ``_local_tensor`` is a leaf variable that autograd refuses to mutate via
        # ``.copy_()``. We bypass this by writing to ``.data`` (the standard PyTorch
        # idiom for in-place parameter mutation outside of autograd) and also wrap
        # the whole swap in ``torch.no_grad()`` for good measure.
        try:
            with torch.no_grad():
                for name, p in self.model.named_parameters():
                    live_local = get_local_tensor(p)
                    saved[name] = live_local.detach().clone()
                    src = shadow[name]
                    # CPU -> GPU path for offloaded shadows; safe to be non-blocking
                    # because the subsequent copy runs on the same CUDA stream.
                    if src.device != live_local.device:
                        src = src.to(live_local.device, non_blocking=True)
                    if src.dtype != live_local.dtype:
                        src = src.to(dtype=live_local.dtype)
                    live_local.data.copy_(src)
            yield
        finally:
            with torch.no_grad():
                for name, p in self.model.named_parameters():
                    if name in saved:
                        get_local_tensor(p).data.copy_(saved[name])

    def _wrap_shadow_as_dtensor_state(self, ema_idx: int) -> Dict[str, torch.Tensor]:
        """
        Build a ``{param_name: tensor}`` state dict matching the live model's structure,
        wrapping each shadow shard so it shares mesh/placement with the corresponding
        live parameter. Used at save time so distributed checkpointing handles EMA
        sharding via the same path as the model state.

        Each entry is a *clone* of the live shadow storage (never an alias). This
        matters for async checkpoint saves: they may continue reading from the
        returned tensors after this method returns, while training continues to
        update the live shadows in :meth:`_ema_update`. Aliasing would race.
        """
        assert self.ema_config is not None
        shadow = self._ema_shadows[ema_idx]
        out: Dict[str, torch.Tensor] = {}
        for name, p in self.model.named_parameters():
            local = shadow[name]
            # Always clone to break aliasing with live storage. For an on-GPU shadow
            # this is a same-device clone; for a CPU-offloaded shadow it's a
            # CPU->GPU copy needed to match the live param's device mesh.
            staged = local.detach().to(p.device, copy=True)
            if isinstance(p, DTensor):
                out[name] = DTensor.from_local(
                    staged,
                    device_mesh=p.device_mesh,
                    placements=p.placements,
                )
            else:
                out[name] = staged
        return out

    def _load_shadow_from_state(
        self, ema_idx: int, sd: Dict[str, torch.Tensor]
    ) -> None:
        """
        Inverse of :meth:`_wrap_shadow_as_dtensor_state`. Copies the loaded
        per-parameter tensors back into the EMA shadow storage, honoring the
        configured device placement.
        """
        assert self.ema_config is not None
        target_device = self._ema_target_device()
        shadow = self._ema_shadows[ema_idx]
        missing: List[str] = []
        for name, _ in self.model.named_parameters():
            if name not in sd:
                missing.append(name)
                continue
            loaded = sd[name]
            local = get_local_tensor(loaded).detach().to(target_device)
            shadow[name] = local
        if missing:
            log.warning(
                f"EMA shadow {ema_idx} missing {len(missing)} parameter(s) at load "
                f"(first few: {missing[:3]}); those entries are unchanged."
            )
        self._ema_loaded_from_checkpoint = True

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

    def _prepare_batch(
        self, batch: Dict[str, Any], labels: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, Any]]:
        input_ids = batch.pop("input_ids")
        labels = labels if labels is not None else batch.pop("labels", None)
        if "doc_lens" in batch and "max_doc_lens" in batch:
            log_once(log, "intra-document masking enabled")
        return input_ids, labels, batch

    def _set_model_mode(self, mode: Literal["train", "eval"]):
        if self._model_mode != mode:
            if mode == "train":
                self.model.train()
            elif mode == "eval":
                self.model.eval()
            else:
                raise ValueError(f"Invalid model mode: {mode}")
            self._model_mode = mode
