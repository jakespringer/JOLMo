### Add a new optimizer (SAM example)

This codebase builds optimizers from config objects and wires them into the training loop automatically.

- The train module constructs the optimizer from your config:

```191:195:/usr0/home/jspringe/projects/JOLMo/src/olmo_core/train/train_module/transformer/train_module.py
        # Build optimizer(s).
        log.info("Building optimizer...")
        self.optim: Optimizer = optim.build(self.model, strict=True)
```

- Optimizers are specified via `OptimConfig` subclasses:

```41:79:/usr0/home/jspringe/projects/JOLMo/src/olmo_core/optim/config.py
@dataclass
class OptimConfig(Config, Generic[Opt], metaclass=ABCMeta):
    """
    Base class for :class:`~torch.optim.Optimizer` configs.
    """

    group_overrides: Optional[List[OptimGroupOverride]] = None
    """
    Use this to pull out groups parameters into a separate param groups with their own options.
    """

    compile: bool = False
    """
    Compile the optimizer step.
    """
```

- Example: built-in AdamW config:

```233:249:/usr0/home/jspringe/projects/JOLMo/src/olmo_core/optim/adam.py
@dataclass
class AdamWConfig(OptimConfig):  # NOTE: omagaconf doesn't like "OptimConfig[torch.optim.AdamW]"
    """
    Configuration class for building an :class:`torch.optim.AdamW` optimizer.
    """
    ...
    @classmethod
    def optimizer(cls) -> Type[torch.optim.AdamW]:
        return torch.optim.AdamW
```

Minimal steps to add a new optimizer
- 1) Implement the runtime optimizer (subclass `torch.optim.Optimizer` if you need custom stepping), or reuse a PyTorch optimizer.
- 2) Add a config class `MyOptimConfig(OptimConfig)` that returns your optimizer class from `optimizer()`.
- 3) Optionally export it from `olmo_core/optim/__init__.py` for convenience.
- 4) Use it in `TransformerTrainModuleConfig(optim=MyOptimConfig(...))`.

Where to put things
- Runtime optimizer: `olmo_core/optim/my_optim.py`
- Config: same file; export via `olmo_core/optim/__init__.py`

Using group overrides and compilation
- You can create param groups by name patterns with `OptimGroupOverride`, e.g., to set `weight_decay=0` for embeddings.
- Setting `compile=True` on your `OptimConfig` compiles `optim.step()` (experimental).

```174:225:/usr0/home/jspringe/projects/JOLMo/src/olmo_core/optim/config.py
        optim: torch.optim.Optimizer = self.optimizer()(
            self.build_groups(model, strict=strict), **kwargs
        )
        ...
        if self.compile:
            log.info("Compiling optimizer step...")
            optim.step = torch.compile(optim.step)
        ...
        return cast(Opt, optim)
```

Implementing Sharpness-Aware Minimization (SAM)
SAM requires a two-pass update: (1) compute grads and perturb weights; (2) recompute loss under perturbation; (3) restore weights and apply base update. There are two practical integration strategies:

- Strategy A (recommended): Implement SAM as a trainer callback that wraps any base optimizer. No core code edits are required because callbacks run right after backward and before the optimizer step.
  - Hooks available:
    - `pre_step(batch)`: called before the batch is processed; capture the batch.
    - `pre_optim_step()`: called after backward, before `optim.step()`; do SAM’s perturbation, second forward+backward, then restore.

Skeleton callback (minimal)

```python
from dataclasses import dataclass
import torch
from olmo_core.train.callbacks import Callback
from olmo_core.utils import move_to_device

@dataclass
class SAMCallback(Callback):
    rho: float = 0.05

    def __post_init__(self):
        self._cached_batch = None

    def pre_step(self, batch):
        # Keep the batch to reuse for the second forward/backward
        self._cached_batch = batch

    @torch.no_grad()
    def _grad_norm(self):
        total = torch.zeros(1, device=self.trainer.device, dtype=torch.float32)
        for group in self.trainer.train_module.optim.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    total += p.grad.detach().to(dtype=torch.float32).pow(2).sum()
        return total.sqrt()

    def pre_optim_step(self):
        tm = self.trainer.train_module
        optim = tm.optim
        batch = self._cached_batch
        if batch is None:
            return

        # 1) Compute raw grad norm and build perturbations
        with torch.no_grad():
            gn = self._grad_norm()
            if not torch.isfinite(gn) or gn == 0:
                return
            scale = self.rho / (gn + 1e-12)
            for g in optim.param_groups:
                for p in g["params"]:
                    if p.grad is None:
                        continue
                    e = p.grad * scale
                    p.add_(e)
                    optim.state[p]["_sam_perturb"] = e

        # 2) Second forward/backward under perturbation (re-run microbatches)
        #    This mirrors tm.train_batch’s forward/backward at a high level.
        #    For large batches you may split again like tm does to avoid OOM.
        input_ids = batch["input_ids"]
        labels = batch.get("labels")
        loss_out = tm.model_forward(input_ids, labels=labels, loss_reduction="sum")
        if isinstance(loss_out, tuple):
            _, loss, *_ = loss_out
        else:
            loss = loss_out
        loss.backward()

        # 3) Restore weights (leaves grads from the second backward intact)
        with torch.no_grad():
            for g in optim.param_groups:
                for p in g["params"]:
                    e = optim.state[p].pop("_sam_perturb", None)
                    if e is not None:
                        p.sub_(e)
```

Wire the callback into a run:

```12:35:/usr0/home/jspringe/projects/JOLMo/src/olmo_core/train/config.py
@dataclass
class TrainerConfig(Config):
    ...
    callbacks: Dict[str, Callback] = field(default_factory=dict)
    ...
    def with_callback(self, name: str, callback: Callback) -> "TrainerConfig":
        ...
```

Example usage (any training script that builds an `ExperimentConfig`):

```python
from olmo_core.train import TrainerConfig
from olmo_core.train.train_module.transformer.config import TransformerTrainModuleConfig
from olmo_core.optim import AdamWConfig, CosWithWarmup

train_module = TransformerTrainModuleConfig(
    rank_microbatch_size=16*1024,
    max_sequence_length=2048,
    optim=AdamWConfig(lr=1e-3),
    scheduler=CosWithWarmup(warmup=100),
    compile_model=True,
)
trainer = TrainerConfig(save_folder="/path/to/run").with_callback("sam", SAMCallback(rho=0.05))
```

- Strategy B (advanced): Implement a SAM optimizer wrapper and a corresponding `SAMConfig(OptimConfig)`. This is straightforward for single-pass optimizers, but canonical SAM requires a second forward/backward. If you choose this route you still need a driver (closure or a callback) to run the second pass because the core loop calls `optim.step()` without a closure. Using the callback approach above avoids modifying the trainer.

Notes
- Gradient clipping and schedulers continue to work as usual. They happen in the trainer before `optim.step()`.
- For multi-microbatch training, re-run the second forward/backward in microbatches to match memory use.


