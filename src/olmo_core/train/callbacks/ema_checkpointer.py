import logging
from dataclasses import dataclass
from typing import Any, ClassVar, Dict, Optional

from olmo_core.config import StrEnum
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.io import join_path

from ..train_module import TransformerTrainModule
from .callback import Callback
from .config_saver import ConfigSaverCallback
from .ema_evaluator_callback import EMAEvaluatorCallback

log = logging.getLogger(__name__)


class EMASaveMode(StrEnum):
    """
    How :class:`EMACheckpointerCallback` writes EMA copies at the end of training.
    """

    none = "none"
    """Do not write any EMA copies. The default :class:`CheckpointerCallback`
    behavior is unchanged."""

    all = "all"
    """At end of training, write the live model and every EMA shadow into separate
    ``final-<label>`` subfolders next to the standard ``final/`` checkpoint."""

    best = "best"
    """At end of training, save only the variant (live or any EMA) with the best
    value of :data:`EMACheckpointerCallback.track_metric` observed during training."""


@dataclass
class EMACheckpointerCallback(Callback):
    """
    Saves end-of-training checkpoints for one or more EMA variants of the model.

    This callback runs *in addition to* the standard :class:`CheckpointerCallback` —
    it does not replace it. Per-step interval saves are still handled by the
    standard checkpointer; this one only writes extra ``final-<variant>/``
    directories at :meth:`post_train`.

    Save modes:

    - ``"none"``: do nothing. The default; preserves backwards-compatible behavior.
    - ``"all"``: write ``final-live/`` (the live model) plus ``final-ema_<decay>/``
      for each configured EMA.
    - ``"best"``: track :data:`track_metric` across all variants over the run and
      save only the best one. Requires :class:`EMAEvaluatorCallback` to be present
      in the trainer's callbacks (auto-promoted by
      :class:`LMEvaluatorCallbackConfig` when EMAs are configured).
    """

    # Lower priority => runs later. We want to run *after* both CheckpointerCallback
    # (priority 1) and EvaluatorCallback (default priority 0), so that on eval steps
    # the EMAEvaluatorCallback has already populated its latest metric cache by the
    # time our post_step reads it.
    priority: ClassVar[int] = -1

    save_mode: EMASaveMode = EMASaveMode.none
    """How to handle final EMA checkpoint output."""

    save_live_model: bool = False
    """If True, also write the live model to ``final-live/`` in ``"all"`` mode. The
    live model is already saved to ``final/`` by the standard CheckpointerCallback,
    so this defaults to False to avoid duplication on disk."""

    track_metric: Optional[str] = None
    """Required for ``save_mode='best'``. The canonical metric name (with no EMA
    prefix) to optimize, e.g. ``"eval/lm/CrossEntropyLoss"``."""

    track_metric_mode: str = "min"
    """``"min"`` or ``"max"``."""

    enabled: bool = True

    # Bookkeeping (populated during post_step in 'best' mode). Mirrors the
    # CheckpointerCallback pattern of using plain dataclass fields with default
    # values for private state. Persisted across resume via state_dict / load_state_dict.
    _best_variant: Optional[str] = None
    _best_value: Optional[float] = None

    def state_dict(self) -> Dict[str, Any]:
        return {
            "best_variant": self._best_variant,
            "best_value": self._best_value,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self._best_variant = state_dict.get("best_variant")
        self._best_value = state_dict.get("best_value")

    def post_attach(self):
        if not self.enabled:
            return
        if self.save_mode == EMASaveMode.none:
            return
        if self.track_metric_mode not in ("min", "max"):
            raise OLMoConfigurationError(
                f"track_metric_mode must be 'min' or 'max', got {self.track_metric_mode!r}"
            )
        if self.save_mode == EMASaveMode.best and self.track_metric is None:
            raise OLMoConfigurationError(
                "EMACheckpointerCallback save_mode='best' requires 'track_metric' to be set"
            )
        tm = self.trainer.train_module
        if not isinstance(tm, TransformerTrainModule) or tm.ema_config is None:
            raise OLMoConfigurationError(
                "EMACheckpointerCallback requires a TransformerTrainModule with an `ema` config"
            )

    def pre_train(self):
        if not self.enabled or self.save_mode != EMASaveMode.best:
            return
        if self._find_ema_evaluator_callback() is None:
            raise OLMoConfigurationError(
                "EMACheckpointerCallback save_mode='best' requires an EMAEvaluatorCallback "
                "to be present in the trainer's callbacks (so that per-variant validation "
                "metrics can be tracked). Configure validation_datasets and ensure the "
                "evaluator gets auto-promoted by setting `ema_track_metric` on "
                "LMEvaluatorCallbackConfig."
            )

    # ------------------------------------------------------------------
    # 'best' mode bookkeeping.
    # ------------------------------------------------------------------

    def post_step(self):
        if not self.enabled or self.save_mode != EMASaveMode.best:
            return
        ema_eval = self._find_ema_evaluator_callback()
        if ema_eval is None or not ema_eval._latest_metric_per_variant:
            return
        for variant, value in ema_eval._latest_metric_per_variant.items():
            if self._is_better(value):
                if self._best_variant != variant or self._best_value != value:
                    log.info(
                        f"EMACheckpointerCallback: new best variant '{variant}' "
                        f"with {self.track_metric}={value}"
                    )
                self._best_variant = variant
                self._best_value = value

    # ------------------------------------------------------------------
    # End-of-training save.
    # ------------------------------------------------------------------

    def post_train(self):
        if not self.enabled or self.save_mode == EMASaveMode.none:
            return

        tm = self.trainer.train_module
        assert isinstance(tm, TransformerTrainModule) and tm.ema_config is not None
        save_folder = self.trainer.save_folder

        if self.save_mode == EMASaveMode.all:
            if self.save_live_model:
                self._save_variant("live", ema_idx=None, save_folder=save_folder)
            for i, decay in enumerate(tm.ema_config.decays):
                label = f"ema_{decay:g}"
                self._save_variant(label, ema_idx=i, save_folder=save_folder)
        elif self.save_mode == EMASaveMode.best:
            if self._best_variant is None:
                log.warning(
                    "EMACheckpointerCallback save_mode='best': no value of "
                    f"track_metric '{self.track_metric}' was observed during "
                    "training; nothing will be saved."
                )
                return
            log.info(
                f"EMACheckpointerCallback save_mode='best': saving variant "
                f"'{self._best_variant}' with {self.track_metric}={self._best_value}"
            )
            ema_idx = self._variant_to_ema_idx(self._best_variant)
            self._save_variant(self._best_variant, ema_idx=ema_idx, save_folder=save_folder)

    # ------------------------------------------------------------------
    # Helpers.
    # ------------------------------------------------------------------

    def _save_variant(
        self, label: str, *, ema_idx: Optional[int], save_folder: str
    ) -> None:
        """
        Save one variant under ``<save_folder>/final-<label>/`` using the trainer's
        existing :class:`Checkpointer`. ``ema_idx=None`` saves the live model.
        """
        out_dir = join_path(save_folder, f"final-{label}")
        log.info(f"Saving EMA variant '{label}' to '{out_dir}'...")
        train_module = self.trainer.train_module
        checkpointer = self.trainer.checkpointer
        if ema_idx is None:
            checkpointer.save(out_dir, train_module, self.trainer.state_dict())
        else:
            with train_module.use_ema_params(ema_idx):
                checkpointer.save(out_dir, train_module, self.trainer.state_dict())

        # Selectively notify *only* ConfigSaverCallback so that the variant
        # directory ends up with the same `config.json` and `data_paths.txt` files
        # that the standard checkpointer's saves get. We deliberately do NOT fan
        # out to all callbacks (which is what CheckpointerCallback's post_train
        # does), because callbacks like BeakerCallback and SlackNotifierCallback
        # treat `post_checkpoint_saved` as "the canonical latest checkpoint" and
        # would be confused (or spammy) for out-of-band EMA variants.
        for cb in self.trainer.callbacks.values():
            if isinstance(cb, ConfigSaverCallback):
                cb.post_checkpoint_saved(out_dir)

    def _find_ema_evaluator_callback(self) -> Optional[EMAEvaluatorCallback]:
        for cb in self.trainer.callbacks.values():
            if isinstance(cb, EMAEvaluatorCallback):
                return cb
        return None

    def _is_better(self, value: float) -> bool:
        if self._best_value is None:
            return True
        if self.track_metric_mode == "min":
            return value < self._best_value
        return value > self._best_value

    def _variant_to_ema_idx(self, variant: str) -> Optional[int]:
        if variant == "live":
            return None
        if not variant.startswith("ema_"):
            raise ValueError(f"Unrecognized EMA variant label '{variant}'")
        decay_str = variant[len("ema_"):]
        decays = self.trainer.train_module.ema_config.decays  # type: ignore[union-attr]
        for i, d in enumerate(decays):
            if f"{d:g}" == decay_str:
                return i
        raise ValueError(
            f"Could not map variant '{variant}' back to an EMA index "
            f"(known decays: {list(decays)})"
        )
