"""
Toy transforms for testing the PEFT abstraction.

These exist so that abstraction-level tests don't have to depend on
LoRA or random masking.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import torch
import torch.nn as nn

from .base import GradientTransform, ModelTransform, ModelTransformReport


@dataclass
class _NoopModelTransform(ModelTransform):
    """Validates ``ModelTransform`` plumbing without changing anything."""

    note: str = "noop"

    def apply(self, model: nn.Module) -> ModelTransformReport:
        del model
        return ModelTransformReport(transform_name=type(self).__name__, notes=self.note)


@dataclass
class _RecordingGradTransform(GradientTransform):
    """Records the parameter FQNs whose gradients it sees each step.

    Useful for tests that need to assert ordering / coverage."""

    seen_fqns_per_step: List[List[str]] = field(default_factory=list)
    pre_train_called: bool = False

    def pre_train(self, model: nn.Module) -> None:
        del model
        self.pre_train_called = True

    def transform(self, model: nn.Module) -> None:
        step_seen: List[str] = []
        for fqn, p in model.named_parameters():
            if p.grad is not None:
                step_seen.append(fqn)
        self.seen_fqns_per_step.append(step_seen)
