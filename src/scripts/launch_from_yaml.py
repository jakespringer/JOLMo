import argparse
import logging
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import torch

from olmo_core.config import Config
from olmo_core.data import NumpyDataLoaderConfig, NumpyDatasetConfig, NumpyPaddedFSLDatasetConfig
from olmo_core.distributed.utils import get_rank
from olmo_core.io import is_url
from olmo_core.nn.transformer import TransformerConfig, Transformer
from olmo_core.train import (
    TrainerConfig,
    prepare_training_environment,
    teardown_training_environment,
)
from olmo_core.train.callbacks import ConfigSaverCallback, WandBCallback, LMEvaluatorCallbackConfig
from olmo_core.train.train_module import TrainModule, TransformerTrainModuleConfig
from olmo_core.train.train_module.transformer.config import TransformerSAMConfig
from olmo_core.train.train_module.transformer.sam_train_module import (
    SAMScheduler,
    TransformerSAMTrainModule,
)
from olmo_core.train.trainer import Trainer
from olmo_core.train.common import Duration
from olmo_core.utils import seed_all

log = logging.getLogger(__name__)


@dataclass
class WandBSettings(Config):
    enabled: bool = False
    name: Optional[str] = None
    project: Optional[str] = None
    entity: Optional[str] = None
    group: Optional[str] = None
    tags: Optional[List[str]] = None
    notes: Optional[str] = None
    id: Optional[str] = None
    resume: Optional[str] = None  # e.g. "allow" or "must"


@dataclass
class YamlExperimentConfig(Config):
    # Either provide full model config or a factory + args
    model: Optional[TransformerConfig] = None
    model_factory: Optional[str] = None
    model_factory_args: Optional[Dict[str, Any]] = None

    dataset: Optional[NumpyDatasetConfig] = None
    data_loader: Optional[NumpyDataLoaderConfig] = None

    # "normal" or "sam"
    train_module_type: str = "normal"
    train_module: Optional[TransformerTrainModuleConfig] = None
    # SAM-only config (optional)
    sam: Optional[TransformerSAMConfig] = None  # populated dynamically if available
    sam_scheduler: Optional[SAMScheduler] = None  # schedule for sam rho, if provided

    trainer: Optional[TrainerConfig] = None
    wandb: Optional[WandBSettings] = None
    # Optional mapping of validation dataset label -> list of paths. If provided,
    # an LM evaluator will be added that reports metrics per label.
    validation_datasets: Optional[Dict[str, List[str]]] = None
    # Optional eval interval for validation datasets. Defaults to 1000 if not set.
    validation_eval_interval: Optional[int] = None

    init_seed: int = 12536
    load_path: Optional[str] = None
    n_tokens: Optional[int] = None


def _build_model(cfg: YamlExperimentConfig) -> Transformer:
    if cfg.model is not None:
        return cfg.model.build(init_device="meta")
    if cfg.model_factory:
        factory = getattr(TransformerConfig, cfg.model_factory, None)
        if factory is None:
            raise ValueError(f"Unknown model factory: {cfg.model_factory}")
        kwargs = cfg.model_factory_args or {}
        model_conf = factory(**kwargs)
        return model_conf.build(init_device="meta")
    raise ValueError("You must provide either 'model' or 'model_factory'.")


def _build_train_module_normal(
    cfg: YamlExperimentConfig, model: Transformer
) -> TrainModule:
    assert cfg.train_module is not None
    return cfg.train_module.build(model)


def _build_train_module_sam(
    cfg: YamlExperimentConfig, model: Transformer
) -> TrainModule:
    assert cfg.train_module is not None
    try:
        import torch.distributed.checkpoint.state_dict as dist_cp_sd  # type: ignore
    except Exception:  # pragma: no cover
        dist_cp_sd = None  # type: ignore
    from olmo_core.config import DType

    # Mirror TransformerTrainModuleConfig.build(...) behavior while routing to SAM module.
    kwargs = cfg.train_module.as_dict(exclude_none=True, recurse=False)
    if (autocast_precision := kwargs.pop("autocast_precision", None)) is not None:
        kwargs["autocast_precision"] = cast(DType, autocast_precision).as_pt()
    if (state_dict_save_opts := kwargs.pop("state_dict_save_opts", None)) is not None:
        if dist_cp_sd is not None:
            kwargs["state_dict_save_opts"] = dist_cp_sd.StateDictOptions(**state_dict_save_opts)  # type: ignore
    if (state_dict_load_opts := kwargs.pop("state_dict_load_opts", None)) is not None:
        if dist_cp_sd is not None:
            kwargs["state_dict_load_opts"] = dist_cp_sd.StateDictOptions(**state_dict_load_opts)  # type: ignore

    sam_cfg = cfg.sam
    return TransformerSAMTrainModule(model=model, sam_config=sam_cfg, sam_scheduler=cfg.sam_scheduler, **kwargs)


def _ensure_wandb_callback(
    trainer: Trainer,
    cfg: YamlExperimentConfig,
) -> None:
    settings = cfg.wandb or WandBSettings(enabled=False)
    if not settings.enabled:
        return

    # If a wandb callback already exists, augment it; otherwise add a new one.
    existing_cb: Optional[WandBCallback] = None
    for name, cb in trainer.callbacks.items():
        if isinstance(cb, WandBCallback):
            existing_cb = cast(WandBCallback, cb)
            break

    if existing_cb is None:
        new_cb = WandBCallback(
            enabled=True,
            name=settings.name,
            project=settings.project,
            entity=settings.entity,
            group=settings.group,
            tags=settings.tags,
            notes=settings.notes,
        )
        if hasattr(new_cb, "id") and settings.id is not None:
            setattr(new_cb, "id", settings.id)
        if hasattr(new_cb, "resume") and settings.resume is not None:
            setattr(new_cb, "resume", settings.resume)
        trainer.add_callback("wandb", new_cb)
    else:
        if settings.name is not None:
            existing_cb.name = settings.name
        if settings.project is not None:
            existing_cb.project = settings.project
        if settings.entity is not None:
            existing_cb.entity = settings.entity
        if settings.group is not None:
            existing_cb.group = settings.group
        if settings.tags is not None:
            existing_cb.tags = settings.tags
        if settings.notes is not None:
            existing_cb.notes = settings.notes
        if hasattr(existing_cb, "id") and settings.id is not None:
            setattr(existing_cb, "id", settings.id)
        if hasattr(existing_cb, "resume") and settings.resume is not None:
            setattr(existing_cb, "resume", settings.resume)


def _ensure_validation_callbacks(
    trainer: Trainer,
    cfg: YamlExperimentConfig,
) -> None:
    """
    If 'validation_datasets' are provided in the YAML, attach an LM evaluator that
    evaluates each dataset individually (via per-file metadata labels).
    """
    if not cfg.validation_datasets:
        return

    assert cfg.dataset is not None
    assert cfg.train_module is not None

    # Derive sequence length and tokenizer for eval dataset from training config.
    seq_len = getattr(cfg.dataset, "sequence_length", None)
    if seq_len is None:
        seq_len = getattr(cfg.train_module, "max_sequence_length", None)
    if seq_len is None:
        raise ValueError("Cannot determine sequence length for validation datasets")

    tokenizer = getattr(cfg.dataset, "tokenizer", None)
    if tokenizer is None:
        raise ValueError("Training dataset must specify a tokenizer to run validation")

    work_dir = getattr(cfg.dataset, "work_dir", None)

    val_paths: List[str] = []
    metadata: List[Dict[str, Any]] = []
    for label, paths in cfg.validation_datasets.items():
        for p in paths:
            val_paths.append(p)
            metadata.append({"label": label})

    eval_dataset_cfg = NumpyPaddedFSLDatasetConfig(
        sequence_length=seq_len,  # type: ignore[arg-type]
        tokenizer=tokenizer,  # type: ignore[arg-type]
        work_dir=work_dir,  # type: ignore[arg-type]
    )
    # Inherit base fields after construction to satisfy some linters/type-checkers.
    eval_dataset_cfg.paths = val_paths
    eval_dataset_cfg.metadata = metadata

    eval_cb_cfg = LMEvaluatorCallbackConfig(
        eval_dataset=eval_dataset_cfg,
        eval_interval=cfg.validation_eval_interval or 1000,
    )
    cb = eval_cb_cfg.build(trainer)
    if cb is not None:
        trainer.add_callback("lm_eval_validation", cb)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=sys.argv[0],
        usage=f"python {sys.argv[0]} CONFIG.yaml [OPTIONS...] [CONFIG_OVERRIDES...]",
        description="Launch training from a YAML config.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("config", type=str, help="Path/URL to YAML config.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the parsed config and exit.",
    )
    return parser


def main():
    opts, overrides = _parse_args()
    cfg = YamlExperimentConfig.from_file(opts.config, overrides=overrides)

    missing = [
        name
        for name, val in dict(
            dataset=cfg.dataset,
            data_loader=cfg.data_loader,
            train_module=cfg.train_module,
            trainer=cfg.trainer,
        ).items()
        if val is None
    ]
    if missing:
        raise ValueError(f"Missing required config sections: {', '.join(missing)}")

    assert cfg.dataset is not None
    assert cfg.data_loader is not None
    assert cfg.train_module is not None
    assert cfg.trainer is not None

    if opts.dry_run and get_rank() == 0:
        print(cfg)
        return

    prepare_training_environment(shared_filesystem=not is_url(cfg.trainer.save_folder))

    seed_all(cfg.init_seed)

    model = _build_model(cfg)
    if cfg.train_module_type.lower() == "sam":
        train_module = _build_train_module_sam(cfg, model)
    else:
        train_module = _build_train_module_normal(cfg, model)

    dataset = cfg.dataset.build()
    data_loader = cfg.data_loader.build(dataset, dp_process_group=train_module.dp_process_group)
    trainer = cfg.trainer.build(train_module, data_loader)

    # Override max_duration if n_tokens is specified
    if cfg.n_tokens is not None:
        trainer.max_duration = Duration.tokens(cfg.n_tokens)

    # Save config for W&B and checkpoints.
    for callback in trainer.callbacks.values():
        if isinstance(callback, ConfigSaverCallback):
            callback.config = cfg.as_config_dict()
            break

    # Ensure W&B callback is present and configured with id/resume semantics.
    _ensure_wandb_callback(trainer, cfg)
    # Optionally attach validation evaluators if provided.
    _ensure_validation_callbacks(trainer, cfg)

    # Auto-load latest checkpoint if present; else optionally init-from `load_path` (model weights only).
    loaded = False
    if not trainer.no_checkpoints:
        loaded = trainer.maybe_load_checkpoint()
    if not loaded and cfg.load_path:
        log.info(f"Loading checkpoint from '{cfg.load_path}' to initialize weights only...")
        trainer.load_checkpoint(cfg.load_path, load_trainer_state=False, load_optim_state=False)

    trainer.fit()

    teardown_training_environment()


def _parse_args() -> Tuple[argparse.Namespace, List[str]]:
    parser = _parser()
    opts, overrides = parser.parse_known_args()
    return opts, overrides


if __name__ == "__main__":
    main()


