"""
Random per-element gradient masking — a concrete
:class:`~olmo_core.nn.peft.base.GradientMaskTransform`.

Each parameter element's gradient is multiplied by an independent
Bernoulli(``fraction_trainable``) sample. The mask is generated once
(deterministically from the seed) and held for the run.

See ``notes/parameter-efficient-training.md`` (§6).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import torch

from olmo_core.exceptions import OLMoConfigurationError

from ..base import GradientMaskTransform


# Default exclusions: MoE routers are tiny but disproportionately
# important to keep fully trainable. Embeddings and the LM head are
# also commonly safer to leave untouched. Users can override.
_DEFAULT_EXCLUDES: List[str] = [
    "*.feed_forward_moe.router.*",
]


@dataclass
class RandomGradientMaskConfig(GradientMaskTransform):
    """Mask each gradient element independently with probability
    ``1 - fraction_trainable`` of being zeroed.

    Example::

        RandomGradientMaskConfig(fraction_trainable=0.1, seed=42)

    will leave roughly 10% of the gradients of every ``nn.Linear``
    weight matrix in effect, and zero the rest. The realized fraction
    has Bernoulli variance — for large tensors it's essentially
    exactly ``fraction_trainable``.

    .. note::
        By default, gradient masking does not fully freeze masked
        entries — stateful optimizers (AdamW's weight decay, residual
        momentum, Lion's sign update, Muon's orthogonalization, ...)
        continue to move ``p`` even when ``grad=0``. For
        optimizer-agnostic guaranteed-constant mask=0 entries, set
        ``enforce_strict_freeze=True`` (inherited from
        :class:`~olmo_core.nn.peft.GradientMaskTransform`). This takes
        a one-time snapshot of the masked parameters at
        :meth:`pre_train` and restores ``mask=0`` positions after every
        ``optim_step``. Memory cost: one extra copy of the masked
        parameters per rank (offload to CPU via
        ``offload_strict_freeze_to_cpu=True`` if needed).

    .. note::
        **Interaction with LoRA**: when ``RandomGradientMaskConfig`` is
        composed with ``LoRAConfig``, the LoRA branch matrices
        (``*.lora_A.weight``, ``*.lora_B.weight``) are *eligible* — they
        live inside ``_LoRABranchLinear`` (an ``nn.Linear`` subclass) and
        their gradients flow during training. The frozen base linears
        are automatically skipped because they have ``requires_grad=False``.
        If you want adapters fully trained and don't want this masking
        applied to them, add ``"*.lora_A.weight"`` and ``"*.lora_B.weight"``
        to :attr:`exclude_fqn_patterns`.
    """

    fraction_trainable: float = 0.1
    """Bernoulli probability that any given parameter element is
    trainable (mask=1). Range ``(0, 1]``."""

    exclude_fqn_patterns: List[str] = field(
        default_factory=lambda: list(_DEFAULT_EXCLUDES)
    )

    def __post_init__(self) -> None:
        if not (0.0 < self.fraction_trainable <= 1.0):
            raise OLMoConfigurationError(
                "RandomGradientMaskConfig 'fraction_trainable' must be in (0, 1]"
            )

    def generate_mask(
        self,
        param_fqn: str,
        global_shape: torch.Size,
        generator: torch.Generator,
    ) -> torch.Tensor:
        del param_fqn
        probs = torch.rand(global_shape, generator=generator, dtype=torch.float32)
        return probs < self.fraction_trainable  # bool, CPU


__all__ = ["RandomGradientMaskConfig"]
