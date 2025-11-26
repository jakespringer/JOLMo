### Multi-node training and efficiency

How distributed is initialized

```85:139:/usr0/home/jspringe/projects/JOLMo/src/olmo_core/train/__init__.py
def prepare_training_environment(
    *,
    seed: Optional[int] = None,
    backend: Optional[str] = "cpu:gloo,cuda:nccl",
    timeout: timedelta = timedelta(minutes=30),
    log_filter_type: Optional[LogFilterType] = None,
    shared_filesystem: Optional[bool] = None,
):
    """
    Prepare the environment for training, including setting up the distributed process group
    for distributed training.
    ...
    """
    ...
    # Initialize process group.
    if backend is not None:
        init_distributed(backend=backend, timeout=timeout, shared_filesytem=shared_filesystem)
    else:
        torch.set_default_device(get_default_device())
    ...
```

How data/tensor/context parallel is applied to the model

```116:145:/usr0/home/jspringe/projects/JOLMo/src/olmo_core/train/train_module/transformer/common.py
    # Maybe shard/replicate according to data parallel config.
    if dp_config is not None:
        assert world_mesh is not None
        dp_mesh = get_dp_model_mesh(world_mesh)
        param_dtype = dp_config.param_dtype.as_pt() if dp_config.param_dtype is not None else None
        if dp_config.name in (DataParallelType.fsdp, DataParallelType.hsdp):
            for m in model_parts:
                ...
                m.apply_fsdp(
                    dp_mesh=dp_mesh,
                    param_dtype=param_dtype,
                    reduce_dtype=dp_config.reduce_dtype.as_pt(),
                    wrapping_strategy=dp_config.wrapping_strategy,
                    pp_enabled=pp_enabled,
                    prefetch_factor=dp_config.prefetch_factor,
                )
            log.info(f"Applied FSDP to the model with {get_device_mesh_info(dp_mesh)}")
        elif dp_config.name == DataParallelType.ddp:
            for m in model_parts:
                ...
                m.apply_ddp(dp_mesh=dp_mesh, compile_enabled=compile_model, param_dtype=param_dtype)
            log.info(f"Applied DDP to the model with {get_device_mesh_info(dp_mesh)}")
        else:
            raise NotImplementedError(dp_config.name)
```

Quick start: torchrun
- Single node (8 GPUs):
  - `torchrun --nproc-per-node=8 path/to/your_script.py --save-folder=/shared/run1 [OVERRIDES...]`
- Multi-node (2 nodes x 8 GPUs):
  - `torchrun --nnodes=2 --nproc-per-node=8 --node_rank=$RANK --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:29400 path/to/your_script.py --save-folder=/shared/run1 [OVERRIDES...]`

Recommended parallel configs (from the full guide)

```298:336:/usr0/home/jspringe/projects/JOLMo/docs/source/guides/all_in_one_for_researchers.md
### Guidelines
...
- For models with 1B or more parameters you should use FSDP instead of DDP.
  This can be configured by setting the {class}`dp_config <olmo_core.train.train_module.TransformerDataParallelConfig>` option as follows:
  ```python
  TransformerTrainModule(
      dp_config=TransformerDataParallelConfig(name="fsdp", param_dtype="bfloat16"),
      ...
  )
  ```
  Equivalently you can set the `dp_config` via command-line overrides like this:
  ```
  --train_module.dp_config='{name: fsdp, param_dtype: bfloat16}'
  ```
  Depending on the size of your model, the number of nodes you're training on, and the data center bandwidth, you may also want to try HSDP instead of FSDP:
  ```python
  TransformerTrainModule(
      dp_config=TransformerDataParallelConfig(name="hsdp", param_dtype="bfloat16"),
      ...
  )
  ```
...
```

Minimal efficiency checklist
- Compile the model: set `compile_model=True` in `TransformerTrainModuleConfig`.
- Prefer FSDP/HSDP for ≥1B params; set `param_dtype=bfloat16`, `reduce_dtype=float32`.
- Increase `rank_microbatch_size` to the max that fits; rely on gradient accumulation automatically when `global_batch_size` is larger.
- If you OOM, enable activation checkpointing with budget mode and tune the budget.
- Keep a CPU-only backend (Gloo) alongside NCCL for async bookkeeping (default if you leave the backend string to `"cpu:gloo,cuda:nccl"`).
- Increase `TrainerConfig.metrics_collect_interval` if logging becomes a bottleneck.
- For shared local folders across nodes, set `OLMO_SHARED_FS=1` or configure `shared_filesystem=True` in `prepare_training_environment`.

Setting dp_config in a script (example)

```177:190:/usr0/home/jspringe/projects/JOLMo/src/examples/llm/train.py
    train_module_config = TransformerTrainModuleConfig(
        rank_microbatch_size=16 * 1024,  # NOTE: this is specified in tokens, not instances
        max_sequence_length=opts.sequence_length,
        optim=AdamWConfig(
            lr=1e-3,
            group_overrides=[
                OptimGroupOverride(params=["embeddings.weight"], opts=dict(weight_decay=0.0))
            ],
        ),
        compile_model=True,
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.fsdp, param_dtype=DType.bfloat16, reduce_dtype=DType.float32
        ),
        max_grad_norm=1.0,
        scheduler=CosWithWarmup(warmup_steps=100),
    )
```


