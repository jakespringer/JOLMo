"""
Trainer :class:`Callback` implementations.
"""

from .batch_size_scheduler import BatchSizeSchedulerCallback
from .beaker import BeakerCallback
from .callback import Callback, CallbackConfig
from .checkpointer import CheckpointerCallback, CheckpointRemovalStrategy
from .comet import CometCallback, CometNotificationSetting
from .config_saver import ConfigSaverCallback
from .console_logger import ConsoleLoggerCallback
from .early_stopping import EarlyStoppingCallback
from .ema_checkpointer import EMACheckpointerCallback, EMASaveMode
from .ema_evaluator_callback import EMAEvaluatorCallback
from .evaluator_callback import (
    DownstreamEvaluatorCallbackConfig,
    EvaluatorCallback,
    LMEvaluatorCallbackConfig,
)
from .garbage_collector import GarbageCollectorCallback
from .gpu_memory_monitor import GPUMemoryMonitorCallback
from .list_checkpointer import ListCheckpointerCallback
from .monkey_patcher import MonkeyPatcherCallback
from .peft_gradient import PEFTGradientCallback
from .profiler import ProfilerCallback
from .sequence_length_scheduler import SequenceLengthSchedulerCallback
from .slack_notifier import SlackNotificationSetting, SlackNotifierCallback
from .speed_monitor import SpeedMonitorCallback
from .stop_at_step import StopAtStepCallback
from .wandb import WandBCallback

__all__ = [
    "Callback",
    "CallbackConfig",
    "CheckpointerCallback",
    "CheckpointRemovalStrategy",
    "CometCallback",
    "CometNotificationSetting",
    "ConfigSaverCallback",
    "ConsoleLoggerCallback",
    "EarlyStoppingCallback",
    "EvaluatorCallback",
    "LMEvaluatorCallbackConfig",
    "DownstreamEvaluatorCallbackConfig",
    "EMAEvaluatorCallback",
    "EMACheckpointerCallback",
    "EMASaveMode",
    "GarbageCollectorCallback",
    "GPUMemoryMonitorCallback",
    "ProfilerCallback",
    "SlackNotifierCallback",
    "SlackNotificationSetting",
    "SequenceLengthSchedulerCallback",
    "SpeedMonitorCallback",
    "WandBCallback",
    "BeakerCallback",
    "BatchSizeSchedulerCallback",
    "MonkeyPatcherCallback",
    "ListCheckpointerCallback",
    "PEFTGradientCallback",
]

__doc__ += "\n"
for name in __all__[2:]:
    if name.endswith("Callback"):
        __doc__ += f"- :class:`{name}`\n"
