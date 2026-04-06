import logging
from dataclasses import dataclass, field
from typing import Dict, Optional

import torch

from olmo_core.exceptions import OLMoConfigurationError

from ..train_module import TransformerTrainModule
from .evaluator_callback import EvaluatorCallback

log = logging.getLogger(__name__)


def _variant_label_for_decay(decay: float) -> str:
    """Human-readable label for an EMA decay value (e.g. 0.999 -> ``"0.999"``)."""
    return f"{decay:g}"


@dataclass
class EMAEvaluatorCallback(EvaluatorCallback):
    """
    Extends :class:`EvaluatorCallback` to additionally evaluate the model under each
    EMA shadow configured on the train module. Metrics for the live model are reported
    under the usual ``eval/...`` prefix; each EMA variant is reported under
    ``eval/ema_<decay>/...``.

    If :data:`track_metric` is set to a canonical metric name (e.g.
    ``"eval/lm/CrossEntropyLoss"``), the latest scalar value for *each* variant
    (live + each EMA) is cached in :data:`_latest_metric_per_variant` so that
    :class:`~.ema_checkpointer.EMACheckpointerCallback` can pick the best variant.
    """

    track_metric: Optional[str] = None
    """
    Canonical metric name (without any EMA prefix) to remember per variant. The base
    name should match what the live evaluator emits, e.g. ``"eval/lm/CrossEntropyLoss"``.
    """

    track_metric_mode: str = "min"
    """``"min"`` or ``"max"``."""

    # Updated by :meth:`_record_metric`. Maps variant label ("live", "ema_0.999", ...)
    # to the most recent observed scalar for ``track_metric``. ``init=False`` so it
    # is not exposed as a constructor argument and cannot be set from YAML.
    _latest_metric_per_variant: Dict[str, float] = field(
        default_factory=dict, init=False, repr=False
    )

    def post_attach(self):
        super().post_attach()
        tm = self.trainer.train_module
        if not isinstance(tm, TransformerTrainModule) or tm.ema_config is None:
            raise OLMoConfigurationError(
                "EMAEvaluatorCallback requires a TransformerTrainModule with an `ema` config"
            )
        if self.track_metric_mode not in ("min", "max"):
            raise OLMoConfigurationError(
                f"track_metric_mode must be 'min' or 'max', got {self.track_metric_mode!r}"
            )

    def _perform_eval(self):
        tm = self.trainer.train_module
        assert isinstance(tm, TransformerTrainModule) and tm.ema_config is not None

        # 1. Live model: same metric prefix as the parent class so existing wandb
        #    panels are unchanged.
        self._run_evaluators(metric_prefix="eval", variant_label="live")

        # 2. Each EMA, swapped in temporarily.
        for i, decay in enumerate(tm.ema_config.decays):
            label = _variant_label_for_decay(decay)
            variant = f"ema_{label}"
            with tm.use_ema_params(i):
                self._run_evaluators(
                    metric_prefix=f"eval/ema_{label}",
                    variant_label=variant,
                )

    def _record_metric(
        self, full_name: str, value: torch.Tensor, *, variant_label: str
    ) -> None:
        # Always forward to the trainer for normal metric reporting.
        self.trainer.record_metric(full_name, value)

        if self.track_metric is None:
            return

        # Map back to the canonical (variant-prefix-free) metric name so the user can
        # specify the metric in its 'eval/<evaluator>/<name>' form regardless of
        # which variant is currently being evaluated.
        if variant_label == "live":
            canonical = full_name
        else:
            # variant_label is "ema_<label>"; the metric prefix is "eval/ema_<label>".
            canonical = full_name.replace(f"eval/{variant_label}/", "eval/", 1)

        if canonical == self.track_metric:
            try:
                self._latest_metric_per_variant[variant_label] = float(value.item())
            except Exception:  # pragma: no cover - defensive
                log.warning(
                    f"Could not coerce tracked metric '{full_name}' to float; skipping."
                )
