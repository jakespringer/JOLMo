"""
Auto-installed dispatcher for PEFT gradient transforms.

This callback is installed by
:class:`olmo_core.train.train_module.TransformerTrainModule` when
``peft.gradient_transforms`` is non-empty. The user does not register
it manually.

It runs every registered
:class:`~olmo_core.nn.peft.base.GradientTransform` at ``pre_optim_step``,
in declared order, before grad clipping inside ``optim_step``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, List

from ...nn.peft.base import GradientTransform
from .callback import Callback


@dataclass
class PEFTGradientCallback(Callback):
    """Dispatch ``GradientTransform.transform`` on every step.

    :param transforms: Declared order of gradient transforms to run.
    """

    priority: ClassVar[int] = 50
    """Run before grad-clipping inside ``optim_step`` (callbacks fire
    at ``pre_optim_step``, before the train module's ``optim_step``
    method runs)."""

    transforms: List[GradientTransform] = field(default_factory=list)

    def pre_train(self) -> None:
        model = self.trainer.train_module.model
        for tx in self.transforms:
            tx.pre_train(model)

    def pre_optim_step(self) -> None:
        model = self.trainer.train_module.model
        for tx in self.transforms:
            tx.transform(model)

    def post_train_batch(self) -> None:
        # Fires after ``train_module.optim_step()`` and
        # ``train_module.zero_grads()``. We use it to run any
        # transform's ``post_optim_step`` hook — e.g., the
        # strict-freeze parameter restore in
        # :class:`~olmo_core.nn.peft.GradientMaskTransform`.
        model = self.trainer.train_module.model
        for tx in self.transforms:
            tx.post_optim_step(model)

    def state_dict(self) -> Dict[str, Any]:
        return {f"tx_{i}": tx.state_dict() for i, tx in enumerate(self.transforms)}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        for i, tx in enumerate(self.transforms):
            tx.load_state_dict(state_dict.get(f"tx_{i}", {}))
