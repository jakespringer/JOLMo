import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

from olmo_core.distributed.utils import get_rank

from .callback import Callback

log = logging.getLogger(__name__)


@dataclass
class MetricsJsonlCallback(Callback):
    """
    Appends every logged metrics dict to a JSONL file, one row per logged step
    (``{"_step": step, **metrics}``), from rank 0. A dependency-free local
    alternative to :class:`WandBCallback`: the full metric history (train loss,
    eval losses, LR, throughput, ...) lands on disk next to the checkpoints,
    so downstream tooling can read it without any external service.

    .. note::
        ``path`` must be on a local filesystem. If unset, it defaults to
        ``{save_folder}/metrics.jsonl``, which therefore requires a local
        ``save_folder``.
    """

    enabled: bool = True
    """
    Set to false to disable this callback.
    """

    path: Optional[str] = None
    """
    The JSONL file to append to. Defaults to ``{save_folder}/metrics.jsonl``.
    """

    _file = None

    def pre_train(self):
        if self.enabled and get_rank() == 0:
            target = Path(self.path) if self.path else Path(self.trainer.save_folder) / "metrics.jsonl"
            target.parent.mkdir(parents=True, exist_ok=True)
            # Append so a resumed run extends the same history; duplicate steps
            # are the reader's job to reconcile (last row wins).
            self._file = open(target, "a")
            log.info(f"Logging metrics to '{target}'")

    def log_metrics(self, step: int, metrics: Dict[str, float]):
        if self._file is not None:
            self._file.write(json.dumps({"_step": step, **metrics}, default=float) + "\n")
            self._file.flush()

    def post_train(self):
        self.close()

    def close(self):
        if self._file is not None:
            self._file.close()
            self._file = None
