"""
Pre-training with an auxiliary representation-matching loss against a frozen
teacher.

The student has two heads: the standard vocab LM head (inherited from
:class:`TransformerTrainModule`), and a trainable Linear ``rep_head`` that
projects the student's last-block hidden state into the teacher's embedding
space. The rep loss is ``|| rep_head(h_student) - h_teacher ||^2``
(optionally L2-normalized).

``rep_head`` uses its own dedicated ``AdamW`` optimizer (``rep_optim``),
decoupled from the student's optimizer. This sidesteps DCP's model-scoped
optimizer-state handling, avoids Muon sub-optimizer partitioning issues, and
gives rep_head a clean round-trip through the checkpoint machinery. See
``JOLMo/notes/teacher-training.md`` for the full design.
"""

import contextlib
import logging
from dataclasses import dataclass
from typing import Any, Dict, Generator, List, Optional

import torch
import torch.distributed as dist
import torch.distributed.checkpoint.state_dict as dist_cp_sd
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.checkpoint.metadata import Metadata
from torch.nn.parallel import DistributedDataParallel as DDP

from olmo_core.config import Config
from olmo_core.data.utils import get_labels, split_batch
from olmo_core.distributed.utils import get_local_tensor, is_distributed
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.nn.transformer import NormalizedTransformer, Transformer
from olmo_core.optim import SkipStepOptimizer
from olmo_core.utils import move_to_device

from ...common import ReduceType
from .teacher import TeacherModel, TeacherModelConfig
from .train_module import TransformerTrainModule

log = logging.getLogger(__name__)

__all__ = [
    "RepLossConfig",
    "TransformerRepLossTrainModule",
]


@dataclass
class RepLossConfig(Config):
    """Representation-loss configuration for
    :class:`TransformerRepLossTrainModule`."""

    rep_loss_weight: float = 1.0
    """Weight on the per-token L2 rep-matching loss. Default 1.0 is a
    conservative starting point; inspect the initial ``train/rep loss`` metric
    in wandb and tune."""

    ce_loss_weight: float = 1.0
    """Weight on the base LM loss. See :class:`DistillConfig.ce_loss_weight`
    for the z_loss interaction."""

    normalize: bool = False
    """If True, L2-normalize student projection and teacher embedding before
    L2 (cosine-style). Default False = raw L2."""

    rep_head_layer_norm: bool = False
    """If True, insert a LayerNorm before the rep_head Linear."""

    rep_lr: float = 3.0e-4
    """Learning rate for the dedicated ``rep_optim`` (AdamW). Not tied to the
    student optimizer's LR. 3e-4 is standard for a lightweight regression
    head."""


class TransformerRepLossTrainModule(TransformerTrainModule):
    """
    Pre-training with an auxiliary representation-matching loss.

    Inherits FSDP / student-optimizer / scheduler / checkpointing from
    :class:`TransformerTrainModule`. Adds a dedicated ``rep_head`` submodule
    and ``rep_optim`` optimizer.

    :param model: The student :class:`~olmo_core.nn.transformer.Transformer`.
    :param teacher: Teacher model configuration (built inside ``__init__``).
    :param rep_loss: Representation-loss hyperparameters.
    :param kwargs: Forwarded to ``TransformerTrainModule.__init__``.
    """

    def __init__(
        self,
        model: Transformer,
        teacher: TeacherModelConfig,
        rep_loss: RepLossConfig,
        **kwargs,
    ):
        for k in ("tp_config", "cp_config", "pp_config"):
            if kwargs.get(k) is not None:
                raise OLMoConfigurationError(
                    f"TransformerRepLossTrainModule does not support {k} in v1"
                )

        # Check for nGPT / EMA BEFORE super().__init__ so we fail fast without
        # building the student. We cannot inspect self.ema_config here because
        # it's set in super(); see below for the EMA reject.
        if isinstance(model, NormalizedTransformer):
            raise OLMoConfigurationError(
                "RepLossTrainModule does not support NormalizedTransformer "
                "(nGPT) students: last-block output is hypersphere-normalized, "
                "making L2 to raw teacher embeddings geometrically ill-posed."
            )

        super().__init__(model=model, **kwargs)
        self.rep_loss_config = rep_loss

        # EMA reject: base EMA iterates self.model.named_parameters() only,
        # so rep_head would be silently excluded from shadow updates. Rather
        # than ship a half-EMAed model, reject up front.
        if self.ema_config is not None:
            raise OLMoConfigurationError(
                "Weight EMA is not supported with TransformerRepLossTrainModule "
                "(EMA shadows do not cover rep_head)."
            )

        if self.z_loss_multiplier is not None and rep_loss.ce_loss_weight != 1.0:
            raise OLMoConfigurationError(
                "ce_loss_weight != 1.0 is not supported with z_loss enabled "
                "(z_loss would be silently rescaled)."
            )

        self.teacher: TeacherModel = teacher.build(
            student_world_mesh=self.world_mesh,
            device=self.device,
            student_max_sequence_length=self.max_sequence_length,
            rank_microbatch_size=self.rank_microbatch_size,
        )

        # Vocab-size pre-flight: rep_loss feeds the student's input_ids to
        # the teacher, so mismatched tokenizers mean token id k indexes
        # unrelated teacher embeddings. Catches the gross-mismatch case
        # (different vocab sizes) at init rather than mid-training. Same-
        # vocab-size-but-different-mapping is still a silent footgun —
        # the caller (JolmoModel) is responsible for ensuring tokenizer
        # compatibility.
        if self.teacher.vocab_size != self.model.vocab_size:
            raise OLMoConfigurationError(
                "rep_loss requires the teacher and student to use "
                f"compatible tokenizers. Student vocab_size="
                f"{self.model.vocab_size}, teacher vocab_size="
                f"{self.teacher.vocab_size}."
            )

        if (
            self.teacher.trained_sequence_length is not None
            and self.teacher.trained_sequence_length < self.max_sequence_length
        ):
            raise OLMoConfigurationError(
                f"Teacher trained at seq_len="
                f"{self.teacher.trained_sequence_length}, but student uses "
                f"{self.max_sequence_length}."
            )

        # Build rep_head: Linear (optionally preceded by LayerNorm). fp32
        # storage; autocast handles bf16 matmul internally.
        d_student = self.model.d_model
        d_teacher = self.teacher.d_model
        rep_core: nn.Module = nn.Linear(
            d_student, d_teacher, bias=True, dtype=torch.float32,
        ).to(self.device)
        if rep_loss.rep_head_layer_norm:
            rep_core = nn.Sequential(
                nn.LayerNorm(d_student, dtype=torch.float32).to(self.device),
                rep_core,
            )

        # DDP-wrap in distributed mode. DDP's __init__ broadcasts rank-0
        # weights, giving us identical init across ranks without manual
        # broadcasting. DDP also all-reduces grads during backward.
        if is_distributed() and self.dp_process_group is not None:
            self.rep_head: nn.Module = DDP(
                rep_core,
                process_group=self.dp_process_group,
                broadcast_buffers=False,
            )
        else:
            self.rep_head = rep_core

        # Dedicated AdamW for rep_head. Decoupled from the student's
        # optimizer — works regardless of whether the student uses AdamW,
        # Muon, etc.
        rep_lr = rep_loss.rep_lr
        self.rep_optim = torch.optim.AdamW(
            list(self.rep_head.parameters()),
            lr=rep_lr,
            betas=(0.9, 0.95),
            weight_decay=0.0,
        )
        # Set initial_lr on the param group so LR schedulers (which read
        # group["initial_lr"] as the base) work with it.
        for g in self.rep_optim.param_groups:
            g.setdefault("initial_lr", rep_lr)

        # Install forward hook on the student's last block to capture hidden
        # states. Always-capture + eager-clear in train_batch + finally-guard
        # on exception.
        self._student_hidden_cache: List[Optional[torch.Tensor]] = [None]
        student_base = (
            self.model.module if isinstance(self.model, DDP) else self.model
        )
        last_block = student_base.blocks[str(student_base.n_layers - 1)]

        def _capture(_module, _inp, out):
            self._student_hidden_cache[0] = out

        self._rep_hook_handle = last_block.register_forward_hook(_capture)

        log.info(f"RepLossConfig: {rep_loss}")
        log.info(
            f"rep_head params: "
            f"{sum(p.numel() for p in self.rep_head.parameters())} "
            f"(d_student={d_student}, d_teacher={d_teacher}, rep_lr={rep_lr})"
        )

    def _rep_head_module(self) -> nn.Module:
        """Return the unwrapped ``rep_head`` (``DDP.module`` if DDP-wrapped)."""
        return self.rep_head.module if isinstance(self.rep_head, DDP) else self.rep_head

    @contextlib.contextmanager
    def _train_microbatch_context(
        self, micro_batch_idx: int, num_micro_batches: int
    ) -> Generator[None, None, None]:
        # Base handles student FSDP/DDP no_sync. For DDP-wrapped rep_head we
        # additionally gate its all-reduce to fire only on the final
        # microbatch, matching the student's accumulation semantics.
        with super()._train_microbatch_context(micro_batch_idx, num_micro_batches):
            is_last_mb = micro_batch_idx == num_micro_batches - 1
            if isinstance(self.rep_head, DDP) and not is_last_mb:
                with self.rep_head.no_sync():
                    yield
            else:
                yield

    def train_batch(self, batch: Dict[str, Any], dry_run: bool = False):
        self._set_model_mode("train")

        if "labels" not in batch:
            batch["labels"] = get_labels(
                batch, label_ignore_index=self.label_ignore_index
            )

        if (instance_mask := batch.get("instance_mask")) is not None and not dry_run:
            self.record_metric(
                "train/masked instances (%)",
                (~instance_mask).float().mean(),
                ReduceType.mean,
            )

        batch_num_tokens = batch["labels"].numel()
        batch_num_tokens_for_loss = move_to_device(
            (batch["labels"] != self.label_ignore_index).sum(), self.device
        )
        self.record_metric(
            "train/masked labels (%)",
            (batch_num_tokens - batch_num_tokens_for_loss) / batch_num_tokens,
            ReduceType.mean,
        )

        ce_batch_loss = move_to_device(torch.tensor(0.0), self.device)
        rep_batch_loss = move_to_device(torch.tensor(0.0), self.device)
        z_batch_loss: Optional[torch.Tensor] = None
        if self.z_loss_multiplier is not None:
            z_batch_loss = move_to_device(torch.tensor(0.0), self.device)

        if self.rank_microbatch_size < (seq_len := batch["input_ids"].shape[1]):
            raise RuntimeError(
                f"Microbatch size ({self.rank_microbatch_size}) is too small "
                f"relative to sequence length ({seq_len})"
            )
        micro_batches = split_batch(batch, self.rank_microbatch_size // seq_len)
        num_micro_batches = len(micro_batches)

        for micro_batch_idx, micro_batch in enumerate(micro_batches):
            with self._train_microbatch_context(micro_batch_idx, num_micro_batches):
                input_ids, labels, model_kwargs = self._prepare_batch(micro_batch)
                teacher_kwargs = {
                    k: model_kwargs[k]
                    for k in ("doc_lens", "max_doc_lens", "cache_leftpad")
                    if k in model_kwargs
                }

                # Teacher first.
                t_emb = self.teacher.hidden_states(input_ids, **teacher_kwargs)

                try:
                    output = self.model_forward(
                        input_ids,
                        labels=labels,
                        ignore_index=self.label_ignore_index,
                        loss_reduction="sum",
                        z_loss_multiplier=self.z_loss_multiplier,
                        loss_div_factor=batch_num_tokens_for_loss,
                        return_logits=False,
                        **model_kwargs,
                    )
                    h = self._student_hidden_cache[0]
                    self._student_hidden_cache[0] = None
                    if h is None:
                        raise RuntimeError(
                            "Student forward did not populate the hidden-state "
                            "cache (hook may have been removed)."
                        )

                    loss = output.loss
                    ce_loss = output.ce_loss
                    z_loss = output.z_loss

                    live_mask = labels != self.label_ignore_index
                    # Run rep_head inside the same autocast context the
                    # student forward uses, so s_emb is bf16 (not fp32)
                    # — otherwise we materialize a fp32 (B, S, d_teacher)
                    # activation saved for backward. The fp32 upcast below
                    # applies only to the small live-token slice.
                    with self._model_forward_context():
                        s_emb = self.rep_head(h)

                    # Boolean-index first to limit the fp32 copy to live
                    # tokens only. The explicit ``.float()`` ensures the L2
                    # computation runs in fp32 regardless of the student's or
                    # teacher's autocast / FSDP dtypes — the teacher can run
                    # in a completely different dtype than the student (e.g.
                    # bf16 teacher with fp32 student, or vice versa) and the
                    # subtraction + pow(2) + sum are guaranteed to be in fp32.
                    s_live = s_emb[live_mask].to(torch.float32)
                    t_live = t_emb[live_mask].detach().to(torch.float32)
                    if self.rep_loss_config.normalize:
                        s_live = F.normalize(s_live, dim=-1)
                        t_live = F.normalize(t_live, dim=-1)
                    # L2 loss computed in fp32:
                    #   rep_sum_i = sum_d (s_live[i,d] - t_live[i,d])^2
                    #   rep_sum   = sum_i rep_sum_i
                    rep_sum = (s_live - t_live).pow(2).sum(dim=-1).sum()
                    # rep_contrib is fp32 / (int64 batch_num_tokens_for_loss)
                    # → fp32. Accumulated per-microbatch across the batch.
                    rep_contrib = rep_sum / batch_num_tokens_for_loss
                    rep_batch_loss += rep_contrib.detach()

                    total = (
                        self.rep_loss_config.ce_loss_weight * loss
                        + self.rep_loss_config.rep_loss_weight * rep_contrib
                    )
                    ce_batch_loss += get_local_tensor(ce_loss.detach())
                    if z_batch_loss is not None and z_loss is not None:
                        z_batch_loss += get_local_tensor(z_loss.detach())

                    total.backward()
                finally:
                    # Clear cache on exit (including exceptional exits) to
                    # avoid holding a grad-tracking reference that would
                    # block FSDP reshard / autograd graph release.
                    self._student_hidden_cache[0] = None

                del output, t_emb, h, total, loss

        del batch

        self.model.post_batch(dry_run=dry_run)
        if dry_run:
            self.model.reset_auxiliary_metrics()
            return

        if isinstance(self.optim, SkipStepOptimizer):
            if is_distributed():
                ce_batch_loss.div_(self._reduce_divide_factor)
                dist.all_reduce(ce_batch_loss)
                ce_batch_loss.div_(self.world_size)
                ce_batch_loss.mul_(self._reduce_divide_factor)
            self.record_ce_loss(ce_batch_loss)
            self.optim.latest_loss = ce_batch_loss
        else:
            self.record_ce_loss(ce_batch_loss, ReduceType.mean)

        self.record_metric(
            "rep loss", rep_batch_loss, ReduceType.mean, namespace="train"
        )
        if z_batch_loss is not None:
            assert self.z_loss_multiplier is not None
            self.record_metric(
                "Z loss", z_batch_loss, ReduceType.mean, namespace="train"
            )
            self.record_metric(
                "Z loss unscaled",
                z_batch_loss / self.z_loss_multiplier,
                ReduceType.mean,
                namespace="train",
            )

        for metric_name, (metric_val, reduction) in self.model.compute_auxiliary_metrics(
            reset=True
        ).items():
            self.record_metric(metric_name, metric_val, reduction, namespace="train")

    def num_flops_per_token(self, seq_len: int) -> int:
        """Total FLOPs/token across student (fwd+bwd) and teacher (fwd)."""
        return sum(self.num_flops_per_token_parts(seq_len).values())

    def num_flops_per_token_parts(self, seq_len: int) -> Dict[str, int]:
        """Per-model FLOPs/token breakdown used by
        :class:`~olmo_core.train.callbacks.SpeedMonitorCallback` to log
        an MFU metric per model. Student counts forward + backward;
        teacher is forward-only (frozen)."""
        return {
            "student": self.model.num_flops_per_token(seq_len),
            "teacher": self.teacher.num_flops_per_token(seq_len),
        }

    def optim_step(self):
        # Base handles: grad clip (via our override, which also clips
        # rep_head), scheduler on self.optim groups, step self.optim, EMA
        # update (no-op when EMA is disabled — which it always is here),
        # post_optim_step on self.model.
        super().optim_step()
        # Additionally schedule and step rep_optim.
        if self.scheduler is not None:
            for group in self.rep_optim.param_groups:
                new_lr = self.scheduler.set_lr(group, self.trainer)
                self.trainer.record_metric(
                    "LR (rep_head)", new_lr, namespace="optim"
                )
        # rep_optim is plain AdamW — SkipStep semantics do NOT apply. A
        # SkipStepOptimizer on self.optim may skip the student update; we
        # always step rep_head. Acceptable for a small regression head.
        self.rep_optim.step()

    def zero_grads(self):
        super().zero_grads()
        self.rep_optim.zero_grad(set_to_none=True)

    def _clip_grad_norm(
        self, max_grad_norm: float, norm_type: float = 2.0,
        foreach: Optional[bool] = None,
    ) -> torch.Tensor:
        # Student is clipped by the base (FSDP-aware in the FSDP path;
        # generic otherwise).
        student_norm = super()._clip_grad_norm(
            max_grad_norm, norm_type=norm_type, foreach=foreach
        )
        # Clip rep_head separately to the same max_grad_norm. DDP has
        # already averaged rep_head grads by this point, so every rank
        # sees the same local norm.
        rep_params = [p for p in self.rep_head.parameters() if p.grad is not None]
        if rep_params:
            rep_norm = torch.nn.utils.clip_grad_norm_(
                rep_params,
                max_grad_norm,
                norm_type=norm_type,
                error_if_nonfinite=False,
                foreach=foreach,
            )
            self.trainer.record_metric(
                "rep_head grad norm",
                rep_norm,
                reduce_type=None,
                namespace="optim",
            )
        # Report the student's norm as "total grad norm" (matches existing
        # semantics; rep_head norm is a separate metric).
        return student_norm

    # ------------------------------------------------------------------
    # Checkpointing — round-trip rep_head weights AND rep_optim moments.
    # ------------------------------------------------------------------

    def _get_state_dict(
        self, sd_options: dist_cp_sd.StateDictOptions, optim: bool = True
    ) -> Dict[str, Any]:
        sd = super()._get_state_dict(sd_options, optim=optim)
        rep_mod = self._rep_head_module()
        sd["rep_head"] = dist_cp_sd.get_model_state_dict(
            rep_mod, options=sd_options
        )
        if optim:
            sd["rep_optim"] = dist_cp_sd.get_optimizer_state_dict(
                rep_mod, self.rep_optim, options=sd_options,
            )
        return sd

    def state_dict_to_load(
        self, metadata: Metadata, *, optim: Optional[bool] = None
    ) -> Dict[str, Any]:
        # ``super().state_dict_to_load`` dispatches back through
        # ``_get_state_dict`` on this subclass, so the returned ``sd``
        # already carries ``"rep_head"`` and ``"rep_optim"`` template
        # entries. We re-affirm them with a fresh template when the
        # checkpoint contains the corresponding keys (matches layout of
        # save_dict), and actively POP them when the checkpoint does
        # NOT contain them. The pop path is the "first resume from a
        # non-rep_loss checkpoint" case described in the design doc:
        # rep_head stays at its DDP-broadcast init, rep_optim starts
        # fresh — we must not pass stale templates to ``dist_cp.load``.
        sd = super().state_dict_to_load(metadata, optim=optim)
        rep_mod = self._rep_head_module()
        load_opts = self.state_dict_load_opts
        has_rep_head = any(
            k.startswith("rep_head.") for k in metadata.state_dict_metadata
        )
        has_rep_optim = any(
            k.startswith("rep_optim.") for k in metadata.state_dict_metadata
        )
        if has_rep_head:
            sd["rep_head"] = dist_cp_sd.get_model_state_dict(
                rep_mod, options=load_opts
            )
        else:
            sd.pop("rep_head", None)
            log.warning(
                "Checkpoint does not contain 'rep_head.*' keys; rep_head "
                "will keep its DDP-broadcast initial weights."
            )
        if optim is not False and has_rep_optim:
            sd["rep_optim"] = dist_cp_sd.get_optimizer_state_dict(
                rep_mod, self.rep_optim, options=load_opts,
            )
        else:
            sd.pop("rep_optim", None)
        return sd

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        rep_head_sd = state_dict.pop("rep_head", None)
        rep_optim_sd = state_dict.pop("rep_optim", None)
        super().load_state_dict(state_dict)
        rep_mod = self._rep_head_module()
        if rep_head_sd is not None:
            dist_cp_sd.set_model_state_dict(
                rep_mod, rep_head_sd, options=self.state_dict_load_opts,
            )
        if rep_optim_sd is not None:
            dist_cp_sd.set_optimizer_state_dict(
                rep_mod, self.rep_optim, rep_optim_sd,
                options=self.state_dict_load_opts,
            )
