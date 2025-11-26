### Using Apache Spark for data loading (efficient integration)

There are two practical ways to use Spark with this training stack:
- Recommended: use Spark offline to build token-ID shards on storage, then train with the built-in numpy-backed dataset/data loader (fastest, simplest).
- Advanced: implement a custom DataLoader that streams from Spark at train time (only if you truly cannot precompute).

Why precompute?
- The built-in data pipeline is optimized for reading large contiguous numpy arrays with deterministic per-epoch shuffles, DP-aware slicing, and efficient range reads from local/NFS/S3/GS. Precomputing lets you fully leverage those paths without changing training code.

Key integration points
- Datasets and data loaders:

```44:66:/usr0/home/jspringe/projects/JOLMo/src/olmo_core/data/data_loader.py
class DataLoaderBase(ABC):
    """
    An abstract base class for data loaders used by the :class:`~olmo_core.train.Trainer`.
    ...
```

```307:323:/usr0/home/jspringe/projects/JOLMo/src/olmo_core/data/data_loader.py
class NumpyDataLoaderBase(TextDataLoaderBase):
    """
    A distributed, deterministic, stateful data loader base class for use with
    :class:`~olmo_core.data.numpy_dataset.NumpyDatasetBase` dataset classes.
    """
```

```1115:1130:/usr0/home/jspringe/projects/JOLMo/src/olmo_core/data/data_loader.py
        data_loader = NumpyDataLoaderBase.wrap_numpy_dataset(
            dataset,
            global_batch_size=self.global_batch_size,
            collator=collator or DataCollator(pad_token_id=dataset.pad_token_id),
            work_dir=self.work_dir or dataset.work_dir,
            dp_world_size=get_world_size(dp_process_group),
            dp_rank=get_rank(dp_process_group),
            fs_local_rank=get_fs_local_rank(),
            seed=self.seed,
            num_threads=self.num_threads,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            target_device_type=self.target_device_type or get_default_device().type,
            ignore_fingerprint_mismatch=self.ignore_fingerprint_mismatch,
        )
```

### Option A (recommended): Precompute numpy shards with Spark

Goal: Write a handful of large `.npy` files with token IDs (dtype must match dataset’s expected dtype), with EOS tokens separating documents. Then point `NumpyFSLDatasetConfig` (or `NumpyVSLDatasetConfig`) at those files.

Minimal pipeline sketch (Spark)
- Tokenize texts into arrays of token IDs; append EOS to each doc.
- Coalesce to moderately large partitions (e.g., 1–4GB output files).
- In `mapPartitions`, concatenate token arrays and write a single `.npy` per partition to a temp local path; upload to shared storage (NFS/S3/GS).
- Collect the list of written shard paths for your training config.

Notes
- DType must match the dataset dtype. If unspecified, the dataset picks it from vocab size:

```2372:2386:/usr0/home/jspringe/projects/JOLMo/src/olmo_core/data/numpy_dataset.py
    def get_dtype(self) -> NumpyUIntTypes:
        if self.dtype is not None:
            return NumpyDatasetDType(self.dtype).as_np_dtype()
        for dtype in (NumpyDatasetDType.uint8, NumpyDatasetDType.uint16, NumpyDatasetDType.uint32, NumpyDatasetDType.uint64):
            if (self.tokenizer.vocab_size - 1) <= np.iinfo(dtype.as_np_dtype()).max:
                log.info(f"Assuming dtype '{dtype}' based on vocab size")
                return dtype.as_np_dtype()
        raise ValueError("vocab size too big!")
```

- Document boundaries: Simply include EOS between docs. The dataset can infer document indices; if local, it may infer directly from the `.npy` array, otherwise it uses sidecar metadata if available.
- You can pass many shards (the dataset supports multiple paths).

Sample config (fixed sequence length)

```142:176:/usr0/home/jspringe/projects/JOLMo/src/examples/llm/train.py
def build_config(opts, overrides: List[str]) -> ExperimentConfig:
    tokenizer_config = TokenizerConfig.gpt2()
    ...
    dataset_config = NumpyFSLDatasetConfig(
        paths=DATA_PATHS,              # <- your Spark-produced .npy shards (local/NFS/S3/GS)
        sequence_length=opts.sequence_length,
        tokenizer=tokenizer_config,
        work_dir=work_dir,             # fast local SSD directory used for cache/indices
    )
    data_loader_config = NumpyDataLoaderConfig(
        global_batch_size=256 * 1024,  # in tokens
        seed=0,
        num_workers=4,
    )
```

Collation and batching
- The provided `DataCollator` pads/truncates and builds masks:

```24:31:/usr0/home/jspringe/projects/JOLMo/src/olmo_core/data/collator.py
@dataclass
class DataCollator:
    """
    The default data collator used by :class:`~olmo_core.data.data_loader.TextDataLoaderBase` subclasses.
    """
    pad_token_id: int
```

Efficiency checklist
- Produce few large shards (avoid tiny files).
- Ensure EOS between documents.
- Align dtype with dataset config.
- Put shards where training has good bandwidth/latency (shared NFS, or S3/GS with high-throughput range reads).
- Set `work_dir` to a fast local SSD for caching indices/buckets.
- Increase `num_workers`, `num_threads`, and `prefetch_factor` in `NumpyDataLoaderConfig` if the loader becomes the bottleneck.

### Option B (advanced): Stream from Spark via a custom DataLoader

Only do this if you cannot precompute. You’ll need to implement a subclass of `TextDataLoaderBase` (or `DataLoaderBase`) that:
- Returns exactly `rank_batch_size` tokens per local rank per batch.
- Is deterministic per epoch and checkpointable (stateful).
- Slices/partitions deterministically across DP ranks and workers.

Required methods (for a `DataLoaderBase` subclass)

```148:168:/usr0/home/jspringe/projects/JOLMo/src/olmo_core/data/data_loader.py
    @property
    @abstractmethod
    def total_batches(self) -> Optional[int]:
        ...
    def __iter__(self) -> Iterator[Dict[str, Any]]:
        ...
    @abstractmethod
    def state_dict(self) -> Dict[str, Any]: ...
    @abstractmethod
    def load_state_dict(self, state_dict: Dict[str, Any]): ...
    @abstractmethod
    def reshuffle(self, epoch: Optional[int] = None, **kwargs): ...
    @abstractmethod
    def _iter_batches(self) -> Iterable[Dict[str, Any]]: ...
    @abstractmethod
    def get_mock_batch(self) -> Dict[str, Any]: ...
```

Guidelines for a Spark-backed loader
- Partitioning: derive a stable global ordering for (epoch, seed), then deterministically assign example indices to DP ranks and to data-loader workers (see how the numpy loader slices by DP rank and worker in its `_get_local_instance_indices`).
- Batching: accumulate items into token-batches of exactly `rank_batch_size` (tokens, not instances).
- State: save `seed`, `epoch`, `batches_processed` (and any Spark offsets/cursors) in `state_dict()`; restore them in `load_state_dict()`.
- Collation: reuse `DataCollator` (pad to max length within batch).

Reference logic to emulate
- DP/worker slicing and batching:

```697:725:/usr0/home/jspringe/projects/JOLMo/src/olmo_core/data/data_loader.py
    def _get_local_instance_indices(self, indices: np.ndarray) -> Iterable[int]:
        ...
        # Slice batches by data loader worker rank ...
        if (worker_info := self.worker_info) is not None:
            indices = indices[worker_info.id :: worker_info.num_workers]
        # Finally slice batches into micro batches for the local DP rank.
        indices = indices[:, self.dp_rank :: self.dp_world_size].reshape((-1,))
        return indices
```

Minimal training changes
- None for Option A (precompute). Just point dataset paths at your Spark outputs.
- For Option B, register your custom DataLoader in your script (build it instead of `NumpyDataLoaderConfig.build()`), but keep the trainer and model unchanged.

When to choose Option B
- You must stream data that cannot be materialized (strict governance), or you need just-in-time transforms that are impractical to precompute. Expect more engineering to meet determinism and throughput targets.


