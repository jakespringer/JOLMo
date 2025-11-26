### Launching runs, configs, and run directory layout

How configs are passed
- Training scripts in this repo build an `ExperimentConfig` (model, data, train module, trainer) and accept CLI overrides in dot-notation.
- Typical entry point uses `olmo_core.script_utils.main`, which parses a few standard flags and forwards overrides to your builder.

```38:85:/usr0/home/jspringe/projects/JOLMo/src/olmo_core/script_utils.py
def get_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=sys.argv[0],
        usage=f"python {sys.argv[0]} [OPTIONS...] [CONFIG_OVERRIDES...]",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--name", type=str, help="A name to assign the run for logging.")
    parser.add_argument("--sequence-length", type=int, default=None, ...)
    parser.add_argument("--data-root", type=str, default="http://olmo-data.org", ...)
    parser.add_argument("--save-folder", type=str, required=True, help="A local or remote directory to save checkpoints to.")
    parser.add_argument("--work-dir", type=str, help="A local directory to use as a working directory for dataset preprocessing.")
    parser.add_argument("--dry-run", action="store_true", help="Print the config and exit.")
    return parser
```

What the main runner does

```101:146:/usr0/home/jspringe/projects/JOLMo/src/olmo_core/script_utils.py
def main(
    config_builder: Callable[[argparse.Namespace, List[str]], ExperimentConfig],
    parser: Optional[argparse.ArgumentParser] = None,
) -> None:
    opts, overrides = _parse_args(parser)
    ...
    prepare_training_environment(shared_filesystem=not is_url(opts.save_folder))
    ...
    model = config.model.build(init_device="meta")
    train_module = config.train_module.build(model)
    dataset = config.dataset.build()
    data_loader = config.data_loader.build(dataset, dp_process_group=train_module.dp_process_group)
    trainer = config.trainer.build(train_module, data_loader)
    ...
    trainer.fit()
    teardown_training_environment()
```

Launching
- Local single node:
  - `python path/to/script.py --save-folder=/tmp/run1 --dry-run` to print the config
  - `torchrun --nproc-per-node=8 path/to/script.py --save-folder=/shared/run1 [OVERRIDES...]`
- Multi-node: see the torchrun example in multi_node.md.
- Official example scripts live under `src/examples/*/train.py` and `src/scripts/train/*`.

Example: building a minimal config in a script

```142:193:/usr0/home/jspringe/projects/JOLMo/src/examples/llm/train.py
def build_config(opts, overrides: List[str]) -> ExperimentConfig:
    ...
    tokenizer_config = TokenizerConfig.gpt2()
    ...
    model_config = factory(
        vocab_size=tokenizer_config.padded_vocab_size(),
    )
    ...
    dataset_config = NumpyFSLDatasetConfig(
        paths=DATA_PATHS,
        sequence_length=opts.sequence_length,
        tokenizer=tokenizer_config,
        work_dir=work_dir,
    )
    data_loader_config = NumpyDataLoaderConfig(
        global_batch_size=256 * 1024,
        seed=0,
        num_workers=4,
    )
    train_module_config = TransformerTrainModuleConfig(
        rank_microbatch_size=16 * 1024,
        max_sequence_length=opts.sequence_length,
        optim=AdamWConfig(lr=1e-3, group_overrides=[...]),
        compile_model=True,
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.fsdp, param_dtype=DType.bfloat16, reduce_dtype=DType.float32
        ),
        max_grad_norm=1.0,
        scheduler=CosWithWarmup(warmup_steps=100),
    )
```

Expected run directory structure
- You specify one `save_folder` for the run (local path or remote URL like `s3://...` or `gs://...`).
- Checkpoints are saved as subdirectories named by step, plus a small metadata file.

```891:905:/usr0/home/jspringe/projects/JOLMo/src/olmo_core/train/trainer.py
    def save_checkpoint(self) -> PathOrStr:
        """
        Save a checkpoint for the current step to the :data:`save_folder`.
        """
        dirname = self.checkpointer.checkpoint_dirname(self.global_step)
        path = join_path(self.save_folder, dirname)
        log.info(f"Saving checkpoint for step {self.global_step} to '{path}'...")
        self.checkpointer.save(path, self.train_module, cast(Dict[str, Any], self.state_dict()))
        ...
```

Layout of a single checkpoint (local folder case)
- `save_folder/step{N}/.metadata.json` – run metadata
- `save_folder/step{N}/train/rank{R}.pt` – trainer state for each rank
- `save_folder/step{N}/model_and_optim/.metadata` – state-dict metadata
- `save_folder/step{N}/model_and_optim/*` – sharded model and optimizer state

Notes
- If saving to a local folder across nodes, ensure a shared filesystem is used (set `OLMO_SHARED_FS=1` or pass `shared_filesystem=True` when preparing the environment). The trainer enforces this.
- You can resume: point `--save-folder` at the same directory; the trainer will load the latest checkpoint if present. Or set `load_path` to a different checkpoint location.


