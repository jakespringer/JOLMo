"""Cancel training once a tracked validation metric stops improving.

Watches ``metric_name`` (e.g. ``eval/lm/dclm/CE loss``) as the evaluator reports it
each eval period, via the :meth:`~Callback.log_metrics` hook, which receives the
reduced, rank-consistent metrics for each step. The callback keeps the running best
value; after ``patience`` consecutive evals with no new best, it cancels the run.

This makes the run length self-determining: pair it with a horizon-free LR schedule
(e.g. :class:`~olmo_core.optim.AlphaInvSqrtWithWarmup`) and a generous ``max_duration``
cap, and training stops as soon as the validation loss has clearly bottomed out.

Cancellation uses ``no_sync=False`` (the default). ``log_metrics`` may run in the
async bookkeeping thread and can lag the main loop by a few steps, so we must *not*
``barrier()`` here (which is what ``no_sync=True`` does). Instead ``cancel_run`` just
records the intent; the trainer's ``training_complete`` check runs ``check_if_canceled``
collectively every ``cancel_check_interval`` steps and stops all ranks cleanly.
"""

from dataclasses import dataclass
from typing import ClassVar, Dict, Optional

from .callback import Callback


@dataclass
class EarlyStoppingCallback(Callback):
    """Cancel the run when ``metric_name`` stops improving for ``patience`` evals.

    ``metric_name`` is the fully-qualified metric key the evaluator logs, e.g.
    ``"eval/lm/dclm/CE loss"``. ``None`` (the default) disables the callback.
    """

    # ClassVar so the dataclass machinery doesn't treat this as an init arg
    # (matches the base ``Callback.priority`` declaration).
    priority: ClassVar[int] = 0

    metric_name: Optional[str] = None
    """The metric to watch, e.g. ``"eval/lm/dclm/CE loss"``. ``None`` disables the callback."""

    patience: int = 3
    """Number of consecutive evals with no new best allowed before canceling."""

    mode: str = "min"
    """``"min"`` if lower is better (a loss), ``"max"`` if higher is better."""

    min_delta: float = 0.0
    """An eval only counts as an improvement if it beats the best by more than this."""

    def pre_train(self):
        # Runtime counters live as plain instance attributes (not dataclass fields) so
        # they don't need to be serializable by the config machinery.
        self._best: Optional[float] = None
        self._num_bad = 0
        self._last_step = -1

    def log_metrics(self, step: int, metrics: Dict[str, float]):
        if self.metric_name is None or self.metric_name not in metrics:
            return
        # log_metrics can be called for a previous step and, in principle, more than
        # once; only act on strictly newer eval steps.
        if step <= getattr(self, "_last_step", -1):
            return
        self._last_step = step

        value = float(metrics[self.metric_name])
        if self._best is None:
            improved = True
        elif self.mode == "min":
            improved = value < self._best - self.min_delta
        else:
            improved = value > self._best + self.min_delta

        if improved:
            self._best = value
            self._num_bad = 0
        else:
            self._num_bad += 1
            if self._num_bad >= self.patience:
                self.trainer.cancel_run(
                    f"early stopping: '{self.metric_name}' did not beat "
                    f"{self._best:.4f} for {self.patience} consecutive evals"
                )
