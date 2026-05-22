"""
Pre-training with knowledge distillation from a frozen teacher model.

Extends :class:`TransformerTrainModule` with KL divergence between the
student's and teacher's next-token distributions. Supports forward KL,
reverse KL, or a sum of both. See ``JOLMo/notes/teacher-training.md`` for
the full design.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F

from olmo_core.data.utils import get_labels, split_batch
from olmo_core.distributed.utils import get_local_tensor, is_distributed
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.config import Config
from olmo_core.nn.lm_head import (
    LMLossImplementation,
    LMOutputWithLoss,
    NormalizedLMHead,
)
from olmo_core.nn.transformer import Transformer
from olmo_core.optim import SkipStepOptimizer
from olmo_core.utils import move_to_device

from ...common import ReduceType
from .teacher import TeacherModel, TeacherModelConfig
from .train_module import TransformerTrainModule

log = logging.getLogger(__name__)

__all__ = [
    "DistillConfig",
    "TransformerDistillTrainModule",
    "kl_per_token",
]


def kl_per_token(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    *,
    direction: Literal["forward", "reverse"],
    temperature: float,
) -> torch.Tensor:
    """
    Return per-token KL of shape ``(B, S)``. Caller applies any masking and
    reduction.

    :param student_logits: ``(B, S, V)`` grad-tracking tensor (typically bf16).
    :param teacher_logits: ``(B, S, V)`` detached tensor (typically bf16).
    :param direction: ``"forward"`` for ``KL(teacher || student)``, ``"reverse"``
        for ``KL(student || teacher)``.
    :param temperature: Softmax temperature applied to both logit tensors.
    """
    T = temperature
    # dtype=fp32 promotes *inside* the log_softmax kernel; the full-V fp32
    # input copy is avoided, but the fp32 output (B,S,V) is saved for backward.
    s_log = F.log_softmax(student_logits / T, dim=-1, dtype=torch.float32)
    t_log = F.log_softmax(teacher_logits / T, dim=-1, dtype=torch.float32)
    if direction == "forward":
        # KL(teacher || student) = sum_v exp(t_log) * (t_log - s_log)
        kl_elem = F.kl_div(s_log, t_log, reduction="none", log_target=True)
    elif direction == "reverse":
        # KL(student || teacher) = sum_v exp(s_log) * (s_log - t_log)
        kl_elem = F.kl_div(t_log, s_log, reduction="none", log_target=True)
    else:
        raise ValueError(f"Unknown KL direction: {direction!r}")
    # Hinton's T^2 convention: compensates the ~1/T factor introduced into
    # the softmax gradient. Applied to both directions.
    return kl_elem.sum(dim=-1) * (T ** 2)


@dataclass
class DistillConfig(Config):
    """Logit-distillation configuration for :class:`TransformerDistillTrainModule`."""

    forward_kl_weight: float = 1.0
    """Weight on ``KL(teacher || student)``."""

    reverse_kl_weight: float = 0.0
    """Weight on ``KL(student || teacher)``. See :attr:`reverse_kl_warmup_steps`
    and :attr:`allow_reverse_kl_from_scratch`."""

    temperature: float = 1.0
    """Softmax temperature for both logits before the KL."""

    ce_loss_weight: float = 1.0
    """Weight on the base LM loss (CE, or CE + z_loss when
    ``z_loss_multiplier`` is set). The combination
    ``ce_loss_weight != 1.0`` AND ``z_loss_multiplier != None`` is rejected
    in ``__init__`` to avoid silently rescaling z-loss stabilization."""

    allow_reverse_kl_from_scratch: bool = False
    """If True, disables the reverse-KL divergence guard. Use only when
    the student starts from a warm checkpoint."""

    reverse_kl_warmup_steps: Optional[int] = None
    """If set, ``reverse_kl_weight`` is ramped linearly from 0 to its full
    value over this many global steps. Setting warmup_steps also suppresses
    the reverse-KL guard."""


class TransformerDistillTrainModule(TransformerTrainModule):
    """
    Pre-training with KL distillation.

    Inherits all FSDP / optimizer / scheduler / checkpointing behavior from
    :class:`TransformerTrainModule` — only ``__init__`` and ``train_batch``
    are overridden.

    :param model: The student :class:`~olmo_core.nn.transformer.Transformer`.
    :param teacher: Teacher model configuration (built inside ``__init__``).
    :param distill: Distillation hyperparameters.
    :param kwargs: Forwarded to ``TransformerTrainModule.__init__``.
    """

    def __init__(
        self,
        model: Transformer,
        teacher: TeacherModelConfig,
        distill: DistillConfig,
        **kwargs,
    ):
        # Reject unsupported parallelism before building the student.
        for k in ("tp_config", "cp_config", "pp_config"):
            if kwargs.get(k) is not None:
                raise OLMoConfigurationError(
                    f"TransformerDistillTrainModule does not support {k} in v1"
                )

        super().__init__(model=model, **kwargs)
        self.distill_config = distill

        self.teacher: TeacherModel = teacher.build(
            student_world_mesh=self.world_mesh,
            device=self.device,
            student_max_sequence_length=self.max_sequence_length,
            rank_microbatch_size=self.rank_microbatch_size,
        )

        # Vocab match.
        if self.teacher.vocab_size != self.model.vocab_size:
            raise OLMoConfigurationError(
                f"Distill requires matching vocab; teacher="
                f"{self.teacher.vocab_size}, student="
                f"{self.model.vocab_size}"
            )

        # LM head checks.
        if isinstance(self.model.lm_head, NormalizedLMHead):
            raise OLMoConfigurationError(
                "Distill does not support NormalizedLMHead (sz scaling "
                "conflicts with KL temperature)."
            )
        if self.model.lm_head.loss_implementation != LMLossImplementation.default:
            raise OLMoConfigurationError(
                "Distill requires lm_head.loss_implementation='default' "
                "(fused-linear does not return logits)."
            )

        # Teacher context window.
        if (
            self.teacher.trained_sequence_length is not None
            and self.teacher.trained_sequence_length < self.max_sequence_length
        ):
            raise OLMoConfigurationError(
                f"Teacher trained at seq_len="
                f"{self.teacher.trained_sequence_length}, but student uses "
                f"{self.max_sequence_length}. RoPE extrapolation would produce "
                f"meaningless teacher signal."
            )

        # Reverse-KL divergence guard. Suppressed if a warmup is configured
        # or the explicit opt-out is set.
        if (
            distill.reverse_kl_weight > 0
            and distill.reverse_kl_warmup_steps is None
            and not distill.allow_reverse_kl_from_scratch
            and distill.forward_kl_weight < 2.0 * distill.reverse_kl_weight
        ):
            raise OLMoConfigurationError(
                "Reverse-KL with forward_kl_weight < 2 * reverse_kl_weight "
                "diverges from a random student. Either (a) increase "
                "forward_kl_weight, (b) set reverse_kl_warmup_steps to ramp "
                "reverse KL from zero, or (c) set "
                "allow_reverse_kl_from_scratch=True (warm-start only)."
            )

        # z_loss + ce_loss_weight is silently broken — z_loss is folded into
        # the base model_forward loss and would be rescaled exactly when it
        # matters most (early training).
        if self.z_loss_multiplier is not None and distill.ce_loss_weight != 1.0:
            raise OLMoConfigurationError(
                "ce_loss_weight != 1.0 is not supported with z_loss enabled "
                "(z_loss would be silently rescaled). Set z_loss_multiplier=None "
                "or ce_loss_weight=1.0."
            )

        log.info(f"DistillConfig: {distill}")
        V = self.model.vocab_size
        logit_gib = self.rank_microbatch_size * V * 2 / 2**30  # bf16 per tensor
        # Realistic peak during combined KL+CE backward:
        #   2× bf16 logits (student + teacher)          : 2 * logit_gib
        #   2× fp32 log-softmax saved for KL backward   : 4 * logit_gib
        #   1× fp32 logits copy saved inside CE backward
        #       (cross_entropy_loss does logits.float())
        #       + 1× fp32 log_softmax saved by F.cross_entropy
        #                                               : 4 * logit_gib
        # ≈ 10× logit_gib.
        log.info(
            f"Distill memory note: peak during backward ~= "
            f"{logit_gib * 10:.1f} GiB per microbatch for V={V}, S="
            f"{self.max_sequence_length}, tokens="
            f"{self.rank_microbatch_size}. For V~100K on 80GB GPUs, "
            f"rank_microbatch_size=4*seq_len is a safe default. Lower if OOM."
        )

    def train_batch(self, batch: Dict[str, Any], dry_run: bool = False):
        self._set_model_mode("train")

        if "labels" not in batch:
            batch["labels"] = get_labels(
                batch, label_ignore_index=self.label_ignore_index
            )

        # Preserve base metric: masked instances %.
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
        kl_fw_batch = move_to_device(torch.tensor(0.0), self.device)
        kl_rv_batch = move_to_device(torch.tensor(0.0), self.device)
        z_batch_loss: Optional[torch.Tensor] = None
        if self.z_loss_multiplier is not None:
            z_batch_loss = move_to_device(torch.tensor(0.0), self.device)

        # Effective reverse KL weight (linear warmup if configured).
        rv_w = self.distill_config.reverse_kl_weight
        if (
            rv_w > 0
            and self.distill_config.reverse_kl_warmup_steps is not None
            and not dry_run
        ):
            ramp = min(
                1.0,
                self.trainer.global_step
                / max(1, self.distill_config.reverse_kl_warmup_steps),
            )
            rv_w = rv_w * ramp

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
                # Forward intra-document masking kwargs to the teacher so its
                # attention respects the same document boundaries as the student.
                teacher_kwargs = {
                    k: model_kwargs[k]
                    for k in ("doc_lens", "max_doc_lens", "cache_leftpad")
                    if k in model_kwargs
                }

                # Teacher forward first — activations free immediately.
                teacher_logits = self.teacher.logits(input_ids, **teacher_kwargs)

                # Student forward.
                output = self.model_forward(
                    input_ids,
                    labels=labels,
                    ignore_index=self.label_ignore_index,
                    loss_reduction="sum",
                    z_loss_multiplier=self.z_loss_multiplier,
                    loss_div_factor=batch_num_tokens_for_loss,
                    return_logits=True,
                    **model_kwargs,
                )
                assert isinstance(output, LMOutputWithLoss)
                student_logits = output.logits
                assert student_logits is not None, (
                    "Distill requires return_logits=True; LM head did not "
                    "return logits."
                )
                loss = output.loss
                ce_loss = output.ce_loss
                z_loss = output.z_loss

                live_mask = labels != self.label_ignore_index

                kl_term = torch.zeros((), device=self.device)
                if self.distill_config.forward_kl_weight > 0:
                    kl_fw = kl_per_token(
                        student_logits,
                        teacher_logits.detach(),
                        direction="forward",
                        temperature=self.distill_config.temperature,
                    )
                    s = (kl_fw * live_mask).sum()
                    kl_term = kl_term + self.distill_config.forward_kl_weight * s
                    kl_fw_batch += (s / live_mask.sum().clamp(min=1)).detach()
                if rv_w > 0:
                    kl_rv = kl_per_token(
                        student_logits,
                        teacher_logits.detach(),
                        direction="reverse",
                        temperature=self.distill_config.temperature,
                    )
                    s = (kl_rv * live_mask).sum()
                    kl_term = kl_term + rv_w * s
                    kl_rv_batch += (s / live_mask.sum().clamp(min=1)).detach()

                # Put KL on the same per-token footing CE was scaled to via
                # loss_div_factor inside model_forward.
                kl_term = kl_term / batch_num_tokens_for_loss

                total = self.distill_config.ce_loss_weight * loss + kl_term
                ce_batch_loss += get_local_tensor(ce_loss.detach())
                if z_batch_loss is not None and z_loss is not None:
                    z_batch_loss += get_local_tensor(z_loss.detach())

                total.backward()

                # Release Python refs so the CUDA caching allocator can reuse
                # blocks on the next microbatch. Autograd already freed saved-
                # for-backward tensors during .backward().
                del output, student_logits, teacher_logits, total, loss

        del batch

        self.model.post_batch(dry_run=dry_run)
        if dry_run:
            self.model.reset_auxiliary_metrics()
            return

        # CE loss metric (with SkipStep early all-reduce if applicable).
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

        if self.distill_config.forward_kl_weight > 0:
            self.record_metric(
                "KL forward", kl_fw_batch, ReduceType.mean, namespace="train"
            )
        if self.distill_config.reverse_kl_weight > 0:
            self.record_metric(
                "KL reverse", kl_rv_batch, ReduceType.mean, namespace="train"
            )
            if self.distill_config.reverse_kl_warmup_steps is not None:
                # Record the effective (warmup-ramped) weight as a scalar
                # metric — same on every rank, so no reduction.
                self.record_metric(
                    "KL reverse effective weight",
                    float(rv_w),
                    namespace="train",
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
