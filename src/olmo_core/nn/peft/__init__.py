"""
Parameter-efficient training (PEFT) for JOLMo.

See ``notes/parameter-efficient-training.md`` for the full design.

Two orthogonal abstractions:

  - :class:`ModelTransform` — build-time mutation of an ``nn.Module``
    (replace/add submodules, set ``requires_grad``). LoRA, BitFit, IA³,
    head-only, etc.
  - :class:`GradientTransform` — per-step in-place mutation of
    ``param.grad`` after backward, before optim. Random masking,
    Fisher masking, gradient noise, etc. A sugar subclass
    :class:`GradientMaskTransform` factors out the "build a per-param
    mask and multiply" pattern.

The user-facing container is :class:`PEFTConfig`, which is plumbed
through :class:`~olmo_core.train.train_module.TransformerTrainModuleConfig.peft`.
"""

from .base import (
    GradientMaskTransform,
    GradientTransform,
    ModelTransform,
    ModelTransformReport,
    PEFTConfig,
    TrainabilityReport,
    stable_hash_u64,
)
from .freezing import (
    BitFitTransform,
    FreezeFQNPatternsTransform,
    HeadOnlyTransform,
    TopNBlocksTransform,
)
from .gradient_mask import RandomGradientMaskConfig
from .lora import LORA_TARGET_PRESETS, LoRAConfig, LoRALinear

__all__ = [
    # Abstractions.
    "ModelTransform",
    "GradientTransform",
    "GradientMaskTransform",
    "PEFTConfig",
    "ModelTransformReport",
    "TrainabilityReport",
    "stable_hash_u64",
    # Freezing primitives.
    "FreezeFQNPatternsTransform",
    "BitFitTransform",
    "HeadOnlyTransform",
    "TopNBlocksTransform",
    # LoRA.
    "LoRAConfig",
    "LoRALinear",
    "LORA_TARGET_PRESETS",
    # Gradient mask.
    "RandomGradientMaskConfig",
]
