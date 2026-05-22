import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

from olmo_core.data import (
    NumpyDatasetConfig,
    NumpyPaddedFSLDataset,
    NumpyVSLDatasetConfig,
    TextDataLoaderBase,
    TokenizerConfig,
)
from olmo_core.data.utils import get_labels
from olmo_core.distributed.utils import get_rank, get_world_size, is_distributed
from olmo_core.eval import Evaluator
from olmo_core.eval.lm_evaluator import LMEvaluator
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.nn.lm_head import LMOutputWithLoss
from olmo_core.utils import (
    cuda_sync_debug_mode,
    format_float,
    gc_cuda,
    get_default_device,
    move_to_device,
)

from ..common import Duration, MetricMergeStrategy
from ..train_module import EvalBatchSizeUnit, EvalBatchSpec, TransformerTrainModule
from ..train_module.transformer.sam_train_module import TransformerSAMTrainModule
from .callback import Callback, CallbackConfig

if TYPE_CHECKING:
    from olmo_eval import HFTokenizer

    from ..trainer import Trainer

log = logging.getLogger(__name__)


@dataclass
class EvaluatorCallback(Callback):
    """
    Runs in-loop evaluations for a :class:`~olmo_core.train.train_module.TransformerTrainModule`
    or :class:`~olmo_core.train.train_module.transformer.sam_train_module.TransformerSAMTrainModule`
    periodically during training.
    """

    evaluators: List[Evaluator] = field(default_factory=list)
    """
    The evaluators to run.
    """

    eval_interval: int = 1000
    """
    The interval (in steps) with which to run the evaluators.
    """

    eval_steps: Optional[List[int]] = None
    """
    An explicit list of training steps at which to run the evaluators, in addition
    to the regular ``eval_interval``. Mirrors :attr:`CheckpointerCallback.save_steps`.
    """

    eval_on_startup: bool = False
    """
    Whether to run an evaluation when the trainer starts up.
    """

    cancel_after_first_eval: bool = False
    """
    If ``True``, cancel the run after running evals for the first time.
    This combined with ``eval_on_startup=True`` is useful if you just want to run in-loop evals
    without training any longer.
    """

    eval_duration: Duration = field(default_factory=lambda: Duration.epochs(1))
    """
    The duration to run each evaluator for.
    """

    log_interval: int = 5
    """
    How often to log eval progress to the console during an eval loop.
    """

    def post_attach(self):
        if not isinstance(self.trainer.train_module, (TransformerTrainModule, TransformerSAMTrainModule)):
            raise OLMoConfigurationError(
                f"'{self.__class__.__name__}' requires a transformer-compatible train module"
            )

    def pre_train(self):
        if self.eval_on_startup:
            self._perform_eval()

    def post_step(self):
        if self.step == 1 and self._has_future_evals():
            # Run a baseline eval after a single training step so the run surfaces
            # any eval-path issues (e.g. OOM) immediately rather than waiting for
            # the first periodic eval.
            self._perform_eval()
            return
        if self.step <= 1:
            return
        periodic = self.eval_interval > 0 and self.step % self.eval_interval == 0
        explicit = self.eval_steps is not None and self.step in self.eval_steps
        if not (periodic or explicit):
            return

        self._perform_eval()

    def _has_future_evals(self) -> bool:
        """True iff at least one eval would fire at some step in ``(1, max_steps]``.

        The step-1 eval is only worth running if some later eval is also going to
        run during training — otherwise we'd be doing an eval that the user
        never asked for.
        """
        try:
            max_steps = self.trainer.max_steps
        except Exception:
            # If max_steps can't be determined yet, assume future evals may run.
            return True
        if self.eval_interval > 0 and self.eval_interval <= max_steps:
            return True
        if self.eval_steps:
            for s in self.eval_steps:
                if 1 < s <= max_steps:
                    return True
        return False

    def _record_metric(
        self, full_name: str, value: torch.Tensor, *, variant_label: str
    ) -> None:
        """
        Hook for subclasses to intercept metric recording. The base class simply
        forwards to ``self.trainer.record_metric``.
        """
        del variant_label  # base class ignores
        self.trainer.record_metric(full_name, value)

    def _perform_eval(self):
        self._run_evaluators(metric_prefix="eval", variant_label="live")

    def _run_evaluators(self, *, metric_prefix: str, variant_label: str):
        """
        Run all configured evaluators once and emit metrics under
        ``f"{metric_prefix}/{evaluator.name}/{name}"``.

        ``variant_label`` is an opaque tag passed to :meth:`_record_metric` so
        subclasses can distinguish e.g. live vs. EMA evaluations. The base class
        ignores it.
        """
        # Put model in eval train mode.
        # TODO: make sure grads will be zeroed at this point
        #  self.trainer.optim.zero_grad(set_to_none=True)
        #  self.trainer.model.eval()
        dp_world_size = get_world_size(self.trainer.dp_process_group)

        # Release the training forward/backward's fragmented CUDA reservations back
        # to the allocator pool before allocating eval-time forward activations
        # (esp. the full (B, T, V) logits tensor). Without this, the allocator can
        # OOM even when enough memory is theoretically free but fragmented.
        gc_cuda()

        evaluator_times = []
        evaluator_names = []
        evaluator_bs = []

        for evaluator in self.evaluators:
            log.info(f"Running {evaluator.name} evals ({variant_label})...")
            start_time = time.monotonic()
            evaluator.reset_metrics()
            eval_step = 0
            eval_tokens = 0
            # If the evaluator only needs per-token CE loss, skip returning the
            # full logits tensor — eliminates one B*T*V allocation per iteration
            # (e.g. 16 GiB for 32K tokens × 128K vocab in fp32).
            return_logits = getattr(evaluator, "needs_logits", True)
            for batch in evaluator:
                eval_step += 1
                eval_tokens += batch["input_ids"].numel() * dp_world_size

                batch = move_to_device(batch, get_default_device())
                with torch.no_grad():
                    # Run forward pass, get logits and un-reduced CE loss.
                    labels = get_labels(batch)
                    output = self.trainer.train_module.eval_batch(
                        batch, labels=labels, return_logits=return_logits
                    )
                    assert isinstance(output, LMOutputWithLoss)
                    logits, _, ce_loss, _ = output

                    # NOTE: might have host-device syncs here but that's okay.
                    with cuda_sync_debug_mode(0):
                        evaluator.update_metrics(batch, ce_loss, logits)

                # Drop references so the previous iteration's logits / ce_loss /
                # LMOutputWithLoss don't live into the *next* forward pass. Without
                # this, peak memory briefly doubles when the next eval_batch call
                # allocates a new logits tensor while the old one is still bound.
                del output, logits, ce_loss, batch, labels

                if self.eval_duration.due(step=eval_step, tokens=eval_tokens, epoch=1):
                    self._log_progress(evaluator, eval_step)
                    break

                if eval_step % self.log_interval == 0 or eval_step == evaluator.total_batches:
                    self._log_progress(evaluator, eval_step)

            # NOTE: going to have a host-device sync here but that's okay. It's only once
            # per evaluator.
            metrics_str = []
            evaluation_names = []
            with cuda_sync_debug_mode(0):
                metrics = evaluator.compute_metrics()
                for name, value in metrics.items():
                    evaluation_names.append(name)
                    metrics_str.append(f"    {name}={format_float(value.item())}")
                    full_name = f"{metric_prefix}/{evaluator.name}/{name}"
                    self._record_metric(full_name, value, variant_label=variant_label)

            evaluator_times.append(time.monotonic() - start_time)
            evaluator_names.append(evaluation_names)
            evaluator_bs.append(eval_step)

            log.info(
                f"Finished {evaluator.name} evals ({variant_label}) in "
                f"{time.monotonic() - start_time:.1f} seconds. Metrics:\n"
                + "\n".join(metrics_str)
            )

        # Sort by evaluator_times in ascending order
        sorted_evaluators = sorted(
            zip(evaluator_names, evaluator_bs, evaluator_times), key=lambda x: x[2]
        )

        # Record evaluation speed.
        eval_speeds = []
        for names, bs, t in sorted_evaluators:
            name = names[0]
            eval_speeds.append(f"    {name} (+variants): {t:.1f} sec ({bs} batches)")
        total_time = sum(evaluator_times)
        total_bs = sum(int(bs) if bs is not None else 0 for bs in evaluator_bs)
        eval_speeds.append(
            f"    Total evaluation time: {total_time:.1f} seconds ({total_bs} batches)"
        )
        log.info("Evaluation speed:\n" + "\n".join(eval_speeds))

        self.trainer.record_metric(
            "throughput/in-loop eval time (s)", total_time, merge_strategy=MetricMergeStrategy.sum
        )
        self.trainer.record_metric(
            "throughput/in-loop eval batches", total_bs, merge_strategy=MetricMergeStrategy.sum
        )

        if self.cancel_after_first_eval:
            self.trainer.cancel_run(
                "canceled from evaluator callback since 'cancel_after_first_eval' is set",
                no_sync=True,  # 'no_sync' because we're calling this from all ranks at the same time.
            )

    def _log_progress(self, evaluator: Evaluator, eval_step: int):
        if evaluator.total_batches is not None:
            log.info(f"[eval={evaluator.name},step={eval_step}/{evaluator.total_batches}]")
        else:
            log.info(f"[eval={evaluator.name},step={eval_step}]")


def _build_evaluator_callback(
    trainer: "Trainer",
    *,
    evaluators: List[Evaluator],
    eval_interval: int,
    log_interval: int,
    eval_on_startup: bool,
    cancel_after_first_eval: bool,
    eval_duration: Duration,
    eval_steps: Optional[List[int]] = None,
    ema_track_metric: Optional[str] = None,
    ema_track_metric_mode: str = "min",
) -> Callback:
    """
    Build either an :class:`EvaluatorCallback` or its EMA-aware subclass, depending
    on whether the attached train module has any EMA shadows configured.
    """
    tm = trainer.train_module
    ema_config = getattr(tm, "ema_config", None)
    if ema_config is not None:
        # Local import to avoid a circular import (ema_evaluator_callback imports
        # this module).
        from .ema_evaluator_callback import EMAEvaluatorCallback

        return EMAEvaluatorCallback(
            evaluators=evaluators,
            eval_interval=eval_interval,
            eval_steps=eval_steps,
            log_interval=log_interval,
            eval_on_startup=eval_on_startup,
            cancel_after_first_eval=cancel_after_first_eval,
            eval_duration=eval_duration,
            track_metric=ema_track_metric,
            track_metric_mode=ema_track_metric_mode,
        )
    return EvaluatorCallback(
        evaluators=evaluators,
        eval_interval=eval_interval,
        eval_steps=eval_steps,
        log_interval=log_interval,
        eval_on_startup=eval_on_startup,
        cancel_after_first_eval=cancel_after_first_eval,
        eval_duration=eval_duration,
    )


@dataclass
class LMEvaluatorCallbackConfig(CallbackConfig):
    eval_dataset: NumpyDatasetConfig
    eval_interval: int = 1000
    eval_steps: Optional[List[int]] = None
    eval_on_startup: bool = False
    cancel_after_first_eval: bool = False
    eval_duration: Duration = field(default_factory=lambda: Duration.epochs(1))
    log_interval: int = 5
    enabled: bool = True
    ema_track_metric: Optional[str] = None
    """
    If the train module has weight EMAs configured, the resulting evaluator callback
    is auto-promoted to :class:`~.ema_evaluator_callback.EMAEvaluatorCallback`. Set
    this to a canonical metric name (e.g. ``"eval/lm/CrossEntropyLoss"``) to have it
    cached per variant for use by :class:`~.ema_checkpointer.EMACheckpointerCallback`
    in ``"best"`` save mode.
    """
    ema_track_metric_mode: str = "min"

    def build(self, trainer: "Trainer") -> Optional[Callback]:
        if not self.enabled:
            return None

        if isinstance(self.eval_dataset, NumpyVSLDatasetConfig):
            dataset_max_sequence_length = self.eval_dataset.max_sequence_length
        else:
            assert hasattr(self.eval_dataset, "sequence_length")
            dataset_max_sequence_length = self.eval_dataset.sequence_length

        batch_spec = trainer.train_module.eval_batch_spec
        if (
            batch_spec.max_sequence_length is not None
            and dataset_max_sequence_length > batch_spec.max_sequence_length
        ):
            raise OLMoConfigurationError(
                f"The maximum sequence length for the LM eval dataset ({dataset_max_sequence_length:,d} tokens) "
                f"is too long for the train module's maximum eval sequence length ({batch_spec.max_sequence_length:,d} tokens)"
            )

        global_eval_batch_size: int
        if batch_spec.batch_size_unit == EvalBatchSizeUnit.tokens:
            global_eval_batch_size = batch_spec.rank_batch_size * get_world_size(
                trainer.dp_process_group
            )
        elif batch_spec.batch_size_unit == EvalBatchSizeUnit.instances:
            global_eval_batch_size = (
                batch_spec.rank_batch_size
                * dataset_max_sequence_length
                * get_world_size(trainer.dp_process_group)
            )
        else:
            raise NotImplementedError(batch_spec.batch_size_unit)

        dataset = self.eval_dataset.build()
        if not isinstance(dataset, NumpyPaddedFSLDataset):
            raise OLMoConfigurationError(
                f"Expected a padded FSL dataset, got '{dataset.__class__.__name__}' instead"
            )

        if not isinstance(trainer.data_loader, TextDataLoaderBase):
            raise OLMoConfigurationError(
                f"Expected a text-based data loader, got '{dataset.__class__.__name__}' instead"
            )

        evaluator = LMEvaluator.from_numpy_dataset(
            dataset,
            name="lm",
            global_batch_size=global_eval_batch_size,
            collator=trainer.data_loader.collator,
            device=trainer.device,
            dp_process_group=trainer.dp_process_group,
        )
        return _build_evaluator_callback(
            trainer,
            evaluators=[evaluator],
            eval_interval=self.eval_interval,
            eval_steps=self.eval_steps,
            log_interval=self.log_interval,
            eval_on_startup=self.eval_on_startup,
            cancel_after_first_eval=self.cancel_after_first_eval,
            eval_duration=self.eval_duration,
            ema_track_metric=self.ema_track_metric,
            ema_track_metric_mode=self.ema_track_metric_mode,
        )


class DownstreamEvaluator(Evaluator):
    metric_type_to_label = {
        "f1_v1": "F1 score",
        "acc_v1": "accuracy",
        "len_norm_v1": "length-normalized accuracy",
        "pmi_dc_v1": "PMI-DC accuracy",
        "ce_loss_v1": "CE loss",
        "bpb_v1": "BPB",
        "soft_v1": "soft loss",
        "soft_log_v1": "log soft loss",
        "f1_v2": "F1 score v2",
        "acc_v2": "accuracy v2",
        "len_norm_v2": "length-normalized accuracy v2",
        "pmi_dc_v2": "PMI-DC accuracy v2",
        "ce_loss_v2": "CE loss v2",
        "bpb_v2": "BPB v2",
        "soft_v2": "soft loss v2",
        "soft_log_v2": "log soft loss v2",
    }

    def __init__(
        self,
        *,
        name: str,
        task: str,
        batch_spec: EvalBatchSpec,
        tokenizer: "HFTokenizer",
        device: Optional[torch.device] = None,
        dp_process_group: Optional[dist.ProcessGroup] = None,
    ):
        from olmo_eval import ICLMetric, ICLMultiChoiceTaskDataset, build_task

        task_dataset: ICLMultiChoiceTaskDataset
        if batch_spec.fixed_sequence_length:
            assert batch_spec.max_sequence_length is not None
            task_dataset = build_task(
                task, tokenizer, model_ctx_len=batch_spec.max_sequence_length, fixed_ctx_len=True
            )
        elif batch_spec.max_sequence_length is not None:
            task_dataset = build_task(task, tokenizer, model_ctx_len=batch_spec.max_sequence_length)
        else:
            task_dataset = build_task(task, tokenizer)

        self.label = task
        self.task = task_dataset
        self.metric = ICLMetric(metric_type=self.task.metric_type).to(
            device or get_default_device()
        )
        sampler: Optional[DistributedSampler] = None
        if is_distributed():
            sampler = DistributedSampler(
                self.task,  # type: ignore
                drop_last=False,
                shuffle=False,
                num_replicas=get_world_size(dp_process_group),
                rank=get_rank(dp_process_group),
            )

        if (
            batch_spec.max_sequence_length is not None
            and self.task.max_sequence_length > batch_spec.max_sequence_length
        ):
            raise OLMoConfigurationError(
                f"The maximum sequence length for downstream eval task '{task}' ({self.task.max_sequence_length:,d} tokens) "
                f"is too long for the train module's maximum eval sequence length ({batch_spec.max_sequence_length:,d} tokens)"
            )

        rank_batch_size_instances: int
        if batch_spec.batch_size_unit == EvalBatchSizeUnit.instances:
            rank_batch_size_instances = batch_spec.rank_batch_size
        elif batch_spec.batch_size_unit == EvalBatchSizeUnit.tokens:
            if batch_spec.fixed_sequence_length:
                assert batch_spec.max_sequence_length is not None
                if batch_spec.rank_batch_size % batch_spec.max_sequence_length != 0:
                    raise OLMoConfigurationError(
                        f"The eval batch size ({batch_spec.rank_batch_size} tokens) must be divisible "
                        f"by the maximum eval sequence length ({batch_spec.max_sequence_length:,d} tokens)"
                    )
                rank_batch_size_instances = (
                    batch_spec.rank_batch_size // batch_spec.max_sequence_length
                )
            else:
                rank_batch_size_instances = (
                    batch_spec.rank_batch_size // self.task.max_sequence_length
                )
        else:
            raise NotImplementedError(batch_spec.batch_size_unit)

        log.info(
            f"Using per-rank batch size of {rank_batch_size_instances} instances "
            f"for downstream eval task '{task}' with max sequence length {self.task.max_sequence_length:,d} tokens"
        )

        data_loader = DataLoader(
            self.task,  # type: ignore
            batch_size=rank_batch_size_instances,
            collate_fn=self.task.collate_fn,
            drop_last=False,
            shuffle=False,
            num_workers=0,
            sampler=sampler,
        )

        super().__init__(name=name, batches=data_loader, device=device)

    def update_metrics(
        self, batch: Dict[str, Any], ce_loss: Optional[torch.Tensor], logits: Optional[torch.Tensor]
    ) -> None:
        del ce_loss
        self.metric.update(batch, logits)

    def compute_metrics(self) -> Dict[str, torch.Tensor]:
        metric_type_to_value = self.metric.compute()
        outputs = {}
        for metric_type, value in metric_type_to_value.items():
            key = f"{self.label} ({self.metric_type_to_label[metric_type]})"
            outputs[key] = value
        return outputs

    def reset_metrics(self) -> None:
        self.metric.reset()


@dataclass
class DownstreamEvaluatorCallbackConfig(CallbackConfig):
    tasks: List[str]
    tokenizer: TokenizerConfig
    eval_interval: int = 1000
    eval_duration: Duration = field(default_factory=lambda: Duration.epochs(1))
    eval_on_startup: bool = False
    cancel_after_first_eval: bool = False
    log_interval: int = 5
    enabled: bool = True
    ema_track_metric: Optional[str] = None
    """See :data:`LMEvaluatorCallbackConfig.ema_track_metric`."""
    ema_track_metric_mode: str = "min"

    def build(self, trainer: "Trainer") -> Optional[Callback]:
        if not self.enabled:
            return None

        from olmo_eval import HFTokenizer

        if self.tokenizer.identifier is None:
            raise OLMoConfigurationError(
                "Tokenizer 'identifier' required to build a concrete tokenizer"
            )

        tokenizer = HFTokenizer(
            self.tokenizer.identifier,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
        )

        evaluators: List[Evaluator] = []
        for task in sorted(self.tasks):
            evaluators.append(
                DownstreamEvaluator(
                    name="downstream",
                    task=task,
                    batch_spec=trainer.train_module.eval_batch_spec,
                    tokenizer=tokenizer,
                    device=trainer.device,
                    dp_process_group=trainer.dp_process_group,
                )
            )

        return _build_evaluator_callback(
            trainer,
            evaluators=evaluators,
            eval_interval=self.eval_interval,
            eval_on_startup=self.eval_on_startup,
            cancel_after_first_eval=self.cancel_after_first_eval,
            log_interval=self.log_interval,
            eval_duration=self.eval_duration,
            ema_track_metric=self.ema_track_metric,
            ema_track_metric_mode=self.ema_track_metric_mode,
        )
