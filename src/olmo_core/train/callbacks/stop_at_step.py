"""Cancel training when ``trainer.global_step`` reaches a specific value.

Useful for terminating a run early without modifying ``max_duration`` (and
hence the scheduler's ``t_max``). The canonical use case is the second leg
of a chained training run that must continue a parent's LR schedule on a
different dataset: the parent sets ``max_duration`` to the full anneal
length but stops early at the dataset-switch point so the LR curve at
that boundary matches what the child run will see.
"""

from dataclasses import dataclass
from typing import ClassVar, Optional

from .callback import Callback


@dataclass
class StopAtStepCallback(Callback):
    """Cancel the trainer once it reaches ``stop_at_step``.

    Lower priority than the default checkpointer (priority 1), so a checkpoint
    scheduled for the same step is written *before* the cancel takes effect.
    """

    # ClassVar so the dataclass machinery doesn't try to make this an init arg
    # (matches the base ``Callback.priority`` declaration).
    priority: ClassVar[int] = 0

    stop_at_step: Optional[int] = None
    """The (inclusive) global step at which to cancel training. ``None`` disables the callback."""

    def post_step(self):
        if self.stop_at_step is None:
            return
        if self.step >= self.stop_at_step:
            # ``no_sync=True``: post_step runs from every rank simultaneously,
            # so we set the cancel flag locally on each rank without entering
            # the bookkeeping op (which would deadlock when called from all
            # ranks at the same point in the loop).
            self.trainer.cancel_run(
                f"canceled by StopAtStepCallback (stop_at_step={self.stop_at_step})",
                no_sync=True,
            )
