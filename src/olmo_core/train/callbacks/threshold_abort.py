"""Cancel training early when a tracked metric is clearly worse than a known
incumbent — the "doomed cell" abort for hyperparameter searches.

Unlike :class:`EarlyStoppingCallback` (which waits for the run's *own* progress to
stall), this callback compares against an externally supplied ``threshold``: as
soon as an eval at or after ``check_after_step`` reports ``metric_name`` on the
wrong side of ``threshold``, the run is canceled. Intended use: when a search
already has an incumbent optimum, boundary cells that are far behind it at the
first eval (empirically, losers of the apple ladder trailed by >0.05 nats at 1/3
epoch) get cut at ~1/3 of their budget instead of running to completion.

Same cancellation discipline as early stopping: ``log_metrics`` may run in the
async bookkeeping thread, so ``cancel_run`` only records the intent and the
trainer stops all ranks collectively at the next ``cancel_check_interval``.
"""

from dataclasses import dataclass
from typing import ClassVar, Dict, Optional

from .callback import Callback


@dataclass
class ThresholdAbortCallback(Callback):
    """Cancel the run if ``metric_name`` is worse than ``threshold`` at any eval
    from ``check_after_step`` onward.

    ``metric_name`` is the fully-qualified metric key the evaluator logs, e.g.
    ``"eval/lm/dclm/CE loss"``. ``None`` (the default) disables the callback.
    """

    priority: ClassVar[int] = 0

    metric_name: Optional[str] = None
    """The metric to watch. ``None`` disables the callback."""

    threshold: Optional[float] = None
    """Abort when the metric is worse than this. ``None`` disables the callback."""

    check_after_step: int = 0
    """Only evals at ``step >= check_after_step`` can trigger the abort (evals
    during warmup/early transients are ignored)."""

    mode: str = "min"
    """``"min"``: abort when metric > threshold (a loss); ``"max"``: abort when
    metric < threshold."""

    def log_metrics(self, step: int, metrics: Dict[str, float]):
        if self.metric_name is None or self.threshold is None:
            return
        if self.metric_name not in metrics or step < self.check_after_step:
            return
        value = float(metrics[self.metric_name])
        doomed = value > self.threshold if self.mode == "min" else value < self.threshold
        if doomed:
            self.trainer.cancel_run(
                f"threshold abort: '{self.metric_name}' = {value:.4f} is worse than "
                f"{self.threshold:.4f} at step {step} (>= {self.check_after_step})"
            )
