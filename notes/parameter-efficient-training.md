# Parameter-Efficient Training: Design Specification

**Status**: implemented (initial PR landed). See ``src/olmo_core/nn/peft/``.

**Implementation deltas vs. this spec** (kept honest; spec is the
design source, code is the source of truth):

- **Discriminator / registry**: the spec proposed a custom
  ``register_model_transform`` decorator with a per-class ``name``
  field. The implementation instead relies on OLMo-core's existing
  ``Config._CLASS_`` mechanism (cf.
  ``olmo_core.config.Config.as_config_dict`` and ``Config.from_dict``)
  which already handles heterogeneous-list YAML round-trip via the
  module-qualified class name. No additional registry needed; the
  round-trip is validated by ``test_peft_config_round_trip_via_class_name_discriminator``.
- **``ModelTransform.init_weights_skip``**: spec'd as an optional
  protocol hook. Removed from v1 because the only built-in
  ``ModelTransform`` (LoRA) doesn't need it — ``LoRALinear`` uses
  ``@property`` weight/bias delegation plus a no-op
  ``reset_parameters`` on ``_LoRABranchLinear`` to leave adapter
  values intact through the standard init pass. When a future PEFT
  method needs a different init policy, wire the predicate through
  ``Transformer.init_weights`` at that time.
- **TP-aware LoRA, FP8+LoRA edge cases**: tested only on FSDP paths.
  TP is rejected at config-build time via
  ``LoRAConfig.assert_compatible``. FP8+LoRA: ``_LoRABranchLinear``
  carries ``_peft_no_fp8 = True``; the FP8 filter honors it. The
  ``base`` linear *is* FP8'd (by design).
- **Strict freeze for gradient masking**: spec §2.5 / §6.4 noted that
  pure gradient masking does *not* keep mask=0 entries constant under
  stateful optimizers (AdamW weight decay, residual momentum, Lion's
  sign update, Muon's orthogonalization). Implemented:
  ``GradientMaskTransform.enforce_strict_freeze: bool = False``. When
  on, ``pre_train`` snapshots the local shard of every masked
  parameter and a new ``post_optim_step`` hook (fired in
  ``PEFTGradientCallback.post_train_batch``) restores mask=0 positions
  after every optimizer step. Optimizer-agnostic by construction: it
  operates in parameter space, not gradient space. Memory cost: one
  local-shard-sized copy per masked parameter (offloadable to CPU via
  ``offload_strict_freeze_to_cpu=True``). Works under FSDP / TP / CP /
  EP / PP because the snapshot is always local-shard-shaped via
  ``get_local_tensor`` — same code path as the existing grad-masking
  ``transform``.

**Scope**: a unified abstraction for parameter-efficient training (PEFT)
in JOLMo, plus two concrete instances:

1. **LoRA adapters** (a `ModelTransform`).
2. **Random gradient masking** (a `GradientMaskTransform`).

The point is the abstraction. LoRA and random masking are the first two
instances; the design must accommodate BitFit, IA³, prefix tuning,
head-only finetuning, Fisher masking, magnitude masking, and structured
masking without revisiting it.

---

## 0. Design principles

1. **No invasive changes to model code.** `Transformer`,
   `TransformerBlock`, `Attention`, `FeedForward`, and `lm_head` are not
   modified. All PEFT effects are transforms applied to the constructed
   model and to its gradients.
2. **Configuration follows the existing `Config` pattern.** Every
   transform is a `Config` dataclass with a discriminator field so
   heterogeneous lists round-trip through YAML/CLI.
3. **Determinism.** Random choices (LoRA `A` init, mask generation) are
   reproducible from a seed — bit-identical across machines, DP world
   sizes, and DTensor placements.
4. **Frozen ≠ removed.** Anything we freeze keeps `requires_grad=False`.
   `OptimConfig.build_groups` already filters these out and warns on
   group-override patterns that match them. We rely on this.
5. **One field on `TransformerTrainModuleConfig`.** A new optional
   `peft: PEFTConfig`. Adding new PEFT methods does not add new
   top-level config fields.

---

## 1. Design alternatives considered

The user's question is essentially: *what is the shape of the
abstraction?* Worth doing the exercise explicitly, because the wrong
shape compounds.

### 1.1 Alternatives

**A. Two abstractions, two lists** (the recommendation; see §2).
`ModelTransform` (build-time mutation + `requires_grad` decisions) and
`GradientTransform` (per-step `.grad` mutation) as separate abstract
bases. `PEFTConfig` carries one list of each.

**B. One unified `PEFTMethod` with both lifecycle hooks** (no-op by
default). One list of methods on `PEFTConfig`. Every method overrides
whichever hooks it needs.

**C. Just `Callback`s.** No new abstractions; PEFT methods are
`Callback` subclasses with a new `pre_build_model` hook to mutate the
model before `parallelize_model`.

**D. Builder / Plan DSL.** `PEFTPlan().add_lora(...).add_mask(...)`
procedurally produces transforms. `.finalize()` returns a declarative
config object.

**E. `PEFTTrainModule` subclasses.** A new `TransformerLoRATrainModule`
(etc.) per PEFT method, mirroring how `TransformerSAMTrainModule` and
`TransformerDistillTrainModule` already live alongside
`TransformerTrainModule`.

**F. Per-parameter `ParameterPolicy`.** Each parameter has an associated
policy object (`Frozen`, `MaskedGrad(mask_fn)`, `Trainable`). The PEFT
system builds and applies a parameter→policy map.

**G. One unified abstraction, methods declare lifecycle hooks via
attributes.** Like (B) but the abstraction has only one base class with
optional methods; loose-typed.

### 1.2 Evaluation criteria

| Criterion | What it means |
|---|---|
| **Lifecycle correctness** | Model mutation must run *before* `parallelize_model`; grad mutation *after* backward, before optim. |
| **Type safety / discoverability** | Compile-time / IDE-time signal of what hooks exist and what they do. |
| **Composability** | Can a run use two PEFT methods (LoRA + mask, BitFit + Fisher mask) without manual glue? |
| **Boilerplate per new method** | How many lines of "no-op overrides" does a method-author write? |
| **Serialization** | Round-trip through OLMo-core YAML/CLI overrides. |
| **Fit for catalog** | How cleanly do BitFit, IA³, magnitude mask, Fisher mask, etc., slot in? |
| **Surface area** | New concepts the user must learn. |

### 1.3 Comparison

| | A (two lists) | B (unified hooks) | C (callbacks) | D (builder) | E (subclasses) | F (policy) | G (loose hooks) |
|---|---|---|---|---|---|---|---|
| Lifecycle correctness | ✅ | ✅ | ❌ — no pre-build hook | ✅ | ✅ | ✅ | ✅ |
| Type safety | ✅ | ⚠️ — both phases on every method | ✅ | ❌ — builder isn't typed | ✅ | ⚠️ | ❌ |
| Composability | ✅ | ✅ | ✅ | ✅ | ❌ — multi-method = MI | ⚠️ | ✅ |
| Boilerplate | low | medium (no-op overrides) | low | medium | high | high | low |
| Serialization | ✅ | ✅ | ✅ | ❌ | ✅ | ⚠️ | ✅ |
| Fit for catalog | ✅ — phases match methods | ✅ | ⚠️ | ⚠️ | ❌ | ❌ — LoRA adds modules | ⚠️ |
| Surface area | 2 bases + sugar | 1 base | 0 (reuse Callback) | builder + node | 1 class per method | policy zoo | 1 base |

### 1.4 Why each loses

- **(B)** collapses two genuinely-distinct lifecycle phases into one
  type. Every method ends up with `def transform_gradients(...): pass`
  or `def apply_to_model(...): pass`. It also blurs ordering rules
  (which methods' `apply_to_model` runs first vs. last when their
  `transform_gradients` run interleaved? — answer is messy).
  The orthogonality of (A) is not just a notational nicety; it's the
  thing that makes the catalog (§7) work.
- **(C)** would be lovely if `Callback` had a `pre_build_model` hook.
  Adding one to `Callback` is possible but it expands the callback
  surface for a use case that's distinct from the others (logging,
  checkpointing, evaluation). And once you've added it, a "LoRA
  Callback" mixed in a list with `WandbCallback` and `ConfigSaverCallback`
  is conceptually wrong — those are about *training loop integration*,
  PEFT is about *what the model and training are*. Don't conflate.
- **(D)** is fine ergonomically but breaks the OLMo-core declarative
  config tradition. YAML/CLI overrides have to target some structured
  state; a procedural builder produces opaque output.
- **(E)** scales as O(PEFT_methods × train_module_variants). With SAM,
  distill, rep-loss already there, plus LoRA, BitFit, IA³, etc., that's
  a Cartesian explosion. And combining two PEFT methods in one run
  requires multiple inheritance.
- **(F)** can't represent "add a LoRA adapter module here." That's an
  *architectural* change, not a parameter-property change. Policies are
  fine for the gradient-mask side of the problem, but they're not the
  right primitive for the model-mutation side.
- **(G)** is (B) minus type safety. Strictly worse than (B).

### 1.5 Why (A) wins

The two abstractions correspond to two *physically different lifecycle
phases* with two *physically different runtimes*:
- One runs once, on `nn.Module` references, on the main process,
  before any device placement or sharding decisions.
- One runs every step, on DTensor `.grad` shards, after backward.

Forcing them into one abstraction is overgeneralization. Letting them be
two is *honest* about the structure.

The risk of (A) — "what about methods that need both?" — is real but
small. Two cases:
- **Independent**: LoRA + random masking (mask only adapter grads).
  Two entries in two lists; they don't share state. Clean.
- **Coupled**: a hypothetical method whose mask depends on adapter
  identities. Even here, the `GradientTransform` can look up the model
  by FQN at `pre_train` time and discover what was added. No need for
  shared abstraction.

What I'd worried about — "GaLore-style methods" — turns out to want a
*third* abstraction (`OptimizerStateTransform`), not a unified one. And
GaLore-class methods are better expressed as custom optimizers
(§8), since the existing `OptimConfig` pattern already accommodates
custom optimizer classes. So no third abstraction.

### 1.6 Refinements added during the alternatives pass

These came out of the comparison and got promoted to the design:

1. **`PEFTConfig` is an orchestrator, not a passive bag.** It owns the
   "apply model transforms in order, build the aggregate skip-predicate,
   install the gradient callback" logic, so `TransformerTrainModule`
   only has to call two methods. (Comes from §1.3 surface-area concern:
   keep `TransformerTrainModule` thin.)
2. **`FreezeFQNPatternsTransform` is a public primitive.** BitFit,
   HeadOnly, TopNBlocks are 10-line wrappers around it. Most "freeze a
   pattern of FQNs" methods compile to this. (Comes from §1.4 — (A)
   risks proliferation of one-trick `ModelTransform`s. Factoring the
   shared pattern out keeps the count down.)
3. **Explicit `ModelTransform` contract.** A short list of "MUST /
   SHOULD" rules in §2.1. Without them, transform-authors can violate
   assumptions (e.g., re-init parameters, mutate the parallelized
   model) and produce subtle bugs.
4. **`TrainabilityReport` aggregates `ModelTransformReport`s** and is
   logged at INFO level after all transforms apply. The user sees one
   summary of what's trainable, not N transform-specific summaries.

---

## 2. The abstractions

### 2.1 `ModelTransform`

Defined in [`src/olmo_core/nn/peft/base.py`](../src/olmo_core/nn/peft/base.py).

```python
@dataclass
class ModelTransform(Config, metaclass=ABCMeta):
    """A build-time, in-place mutation of an ``nn.Module``.

    A ``ModelTransform`` may:
      - replace existing submodules,
      - add new submodules,
      - set ``requires_grad`` on parameters (this is how it declares
        what is trainable under this PEFT method).

    Contract (rely on these; the abstraction breaks otherwise):

      MUST:
        - mutate ``model`` in place; do not return a new model.
        - be deterministic given its config (any randomness derives
          from an explicit ``seed`` field on the config).
        - be applied on the unparallelized model, before any
          ``apply_tp``/``apply_fsdp``/``apply_pp`` call.
        - leave ``model.embeddings`` and ``model.lm_head`` accessible
          (rest of codebase expects them).

      MUST NOT:
        - call ``model.init_weights`` or otherwise re-initialize
          parameters owned by the model author.
        - assume idempotence; the orchestrator guarantees the
          transform is applied exactly once.

      SHOULD:
        - log a ``ModelTransformReport`` describing what it did.
        - implement ``init_weights_skip`` if it adds parameters with
          non-trivial init that must survive the model-author's init
          pass.
        - implement ``checkpoint_key_mapping`` if it renames parameters
          (so old checkpoints still load).
    """
    name: ClassVar[str] = "(abstract)"   # discriminator for serialization

    @abstractmethod
    def apply(self, model: nn.Module) -> "ModelTransformReport": ...

    def init_weights_skip(
        self, model: nn.Module
    ) -> Optional[Callable[[str], bool]]:
        return None

    def checkpoint_key_mapping(
        self, model: nn.Module
    ) -> Optional[Dict[str, str]]:
        return None

    def assert_compatible(
        self, train_module_config: "TransformerTrainModuleConfig"
    ) -> None:
        """Optional. Raise OLMoConfigurationError on incompatible combos
        (e.g., LoRA + TP in v1). Called at config-build time so the
        user sees the error before any training resources are
        allocated."""
        return None
```

`ModelTransformReport`:

```python
@dataclass(frozen=True)
class ModelTransformReport:
    transform_name: str
    modules_replaced: int = 0
    modules_added:    int = 0
    params_frozen:    int = 0
    params_added:     int = 0
    target_fqns:      Tuple[str, ...] = ()
    notes:            str = ""
    def summary(self) -> str: ...
```

### 2.2 `GradientTransform`

```python
@dataclass
class GradientTransform(Config, metaclass=ABCMeta):
    """A per-step in-place mutation of ``param.grad`` for every parameter
    in the model. Runs after backward and before grad clipping.

    Stateful but the state is allocated lazily in ``pre_train`` (after
    parallelization is final). State is NOT checkpointed by default;
    override ``state_dict``/``load_state_dict`` if your transform has
    training-time state that must round-trip.

    Multiple ``GradientTransform``s compose in the order declared in
    ``PEFTConfig.gradient_transforms``.
    """
    name: ClassVar[str] = "(abstract)"

    @abstractmethod
    def pre_train(self, model: nn.Module) -> None: ...

    @abstractmethod
    def transform(self, model: nn.Module) -> None: ...

    def state_dict(self) -> Dict[str, Any]: return {}
    def load_state_dict(self, sd: Dict[str, Any]) -> None: del sd
```

### 2.3 `GradientMaskTransform` (sugar layer)

A large class of useful gradient transforms have one shape: "per-param
binary mask, computed once, multiplied into `.grad` every step." Factor
it.

```python
@dataclass
class GradientMaskTransform(GradientTransform, metaclass=ABCMeta):
    """Base for any GradientTransform that acts as a per-parameter
    multiplicative mask. Subclasses implement ``generate_mask`` only.
    The base handles selection, deterministic seeding, DTensor-correct
    sharding, storage, and per-step application.
    """
    name: ClassVar[str] = "(abstract)"

    target_module_types:  List[str] = field(default_factory=lambda: ["Linear"])
    target_param_kinds:   List[str] = field(default_factory=lambda: ["weight"])
    exclude_fqn_patterns: List[str] = field(default_factory=list)
    seed: int = 0

    @abstractmethod
    def generate_mask(
        self, param_fqn: str, global_shape: torch.Size, generator: torch.Generator
    ) -> torch.Tensor:
        """Return a CPU mask tensor (bool or float) of shape
        ``global_shape``. ``generator`` is seeded from
        ``(self.seed, param_fqn)`` and is the only source of randomness
        the subclass should use."""

    # Provided by the base; see §3.3 for what they do:
    def pre_train(self, model: nn.Module) -> None: ...
    def transform(self, model: nn.Module) -> None: ...
```

Concrete masks (random, magnitude, Fisher, structured) implement only
`generate_mask`. Methods that aren't shaped like a mask (gradient noise,
gradient centralization, LayerDrop) extend `GradientTransform` directly.

### 2.4 `PEFTConfig` — orchestrator

```python
@dataclass
class PEFTConfig(Config):
    model_transforms:    List[ModelTransform]    = field(default_factory=list)
    gradient_transforms: List[GradientTransform] = field(default_factory=list)

    def assert_compatible(self, tm_config) -> None:
        for tx in self.model_transforms:    tx.assert_compatible(tm_config)
        # (gradient_transforms don't currently need this hook; add if needed)

    def apply_model_transforms(
        self, model: nn.Module
    ) -> "TrainabilityReport":
        """Run every model transform in declared order. Returns the
        aggregate trainability report (logged at INFO by the train
        module). Idempotency: re-running this on an already-transformed
        model is a programmer error (asserts)."""

    def init_weights_skip(
        self, model: nn.Module
    ) -> Optional[Callable[[str], bool]]:
        """OR of per-transform skip predicates."""

    def checkpoint_key_mapping(
        self, model: nn.Module
    ) -> Dict[str, str]:
        """Merge per-transform key mappings; raise on conflicts."""

    def install_gradient_callback(
        self, trainer: "Trainer"
    ) -> None:
        """Auto-installs a single PEFTGradientCallback that runs all
        gradient_transforms in declared order at pre_optim_step."""
```

`TrainabilityReport`:

```python
@dataclass(frozen=True)
class TrainabilityReport:
    per_transform: Tuple[ModelTransformReport, ...]
    total_params:        int
    trainable_params:    int
    added_params:        int
    frozen_params:       int
    def summary(self) -> str: ...   # human-readable single line
```

User code:

```python
peft = PEFTConfig(
    model_transforms=[
        LoRAConfig(r=8, alpha=16, target_modules=["q", "v"]),
    ],
    gradient_transforms=[
        RandomGradientMaskConfig(fraction_trainable=0.1, seed=42),
    ],
)
```

### 2.5 Why a discriminator-based registry (not closed `Union[...]`)

`List[ModelTransform]` must round-trip through YAML. Two ways to do
that:

- **Closed union**: `List[Union[LoRAConfig, BitFitConfig, ...]]`. Adds
  every new transform to the union. Out-of-tree transforms impossible.
- **Discriminator + registry**: each subclass declares
  `name: ClassVar[str]`; a module-level dict maps name → class. The
  config deserializer reads `{"name": "lora", ...}` and dispatches.
  Out-of-tree transforms register via decorator.

OLMo-core already uses the discriminator pattern (cf. `FeedForwardType`,
`AttentionType`). Use it here. The two-line addition in `base.py` is:

```python
_MODEL_TRANSFORM_REGISTRY: Dict[str, Type[ModelTransform]] = {}
def register_model_transform(cls):
    _MODEL_TRANSFORM_REGISTRY[cls.name] = cls; return cls
```

(and analogous for `GradientTransform`). The `Config.from_dict` /
`as_dict` paths consult the registry to find the right concrete class.

---

## 3. Supporting infrastructure (new files & primitives)

These are not LoRA-specific; they exist to make PEFT method authoring
trivial.

### 3.1 `FreezeFQNPatternsTransform` — primitive for freezing-only methods

[`src/olmo_core/nn/peft/freezing.py`](../src/olmo_core/nn/peft/freezing.py)

```python
@register_model_transform
@dataclass
class FreezeFQNPatternsTransform(ModelTransform):
    """Set requires_grad=False on every parameter matching any glob
    in ``freeze_patterns`` unless it also matches any glob in
    ``except_patterns``. Does not add or replace modules.
    """
    name: ClassVar[str] = "freeze_fqn_patterns"
    freeze_patterns: List[str]
    except_patterns: List[str] = field(default_factory=list)

    def apply(self, model): ...   # ~15 lines: walk, fnmatch, freeze, report
```

Then BitFit, HeadOnly, TopNBlocks are ~10 lines each:

```python
@register_model_transform
@dataclass
class BitFitTransform(FreezeFQNPatternsTransform):
    name: ClassVar[str] = "bitfit"
    freeze_patterns: List[str] = field(default_factory=lambda: ["*"])
    except_patterns: List[str] = field(default_factory=lambda: ["*.bias"])

@register_model_transform
@dataclass
class HeadOnlyTransform(FreezeFQNPatternsTransform):
    name: ClassVar[str] = "head_only"
    freeze_patterns: List[str] = field(default_factory=lambda: ["*"])
    except_patterns: List[str] = field(default_factory=lambda: ["lm_head.*"])
```

This is the kind of factoring the alternatives analysis (§1.3) flagged:
the abstraction is only worth its cost if it eliminates per-method
boilerplate. `FreezeFQNPatternsTransform` is the primitive that does
that for the freezing family.

### 3.2 `ParameterMaskRegistry` — private storage for `GradientMaskTransform`

[`src/olmo_core/nn/peft/gradient_mask/_registry.py`](../src/olmo_core/nn/peft/gradient_mask/_registry.py)

Internal data structure: `Dict[param_fqn, torch.Tensor]` where each
tensor is the local-shard-shaped mask on the param's device, plus
helpers `build(model, mask_transform)` and `apply(model)`. Used by
`GradientMaskTransform`'s `pre_train`/`transform`; concrete mask
subclasses never touch it.

Lives in a private (underscored) module because it's an implementation
detail; promote it if a sibling abstraction needs it later.

### 3.3 `PEFTGradientCallback` — auto-installed dispatcher

[`src/olmo_core/train/callbacks/peft_gradient.py`](../src/olmo_core/train/callbacks/peft_gradient.py)

```python
@dataclass
class PEFTGradientCallback(Callback):
    """Auto-installed by TransformerTrainModule when
    peft.gradient_transforms is non-empty. Runs every registered
    GradientTransform at pre_optim_step in declared order.
    Aggregates state-dicts so checkpoints round-trip correctly."""
    priority: ClassVar[int] = 50   # runs before grad clipping inside optim_step

    _transforms: List[GradientTransform] = field(default_factory=list)

    def pre_train(self):
        for tx in self._transforms:
            tx.pre_train(self.trainer.train_module.model)

    def pre_optim_step(self):
        m = self.trainer.train_module.model
        for tx in self._transforms:
            tx.transform(m)

    def state_dict(self):
        return {f"tx_{i}": tx.state_dict() for i, tx in enumerate(self._transforms)}

    def load_state_dict(self, sd):
        for i, tx in enumerate(self._transforms):
            tx.load_state_dict(sd.get(f"tx_{i}", {}))
```

Note: this is exactly the case (C) "PEFT methods are just callbacks"
*almost* gets right. The difference: the callback is internal plumbing,
not the public API. The user authors `GradientTransform`s; the system
auto-wraps them.

### 3.4 `stable_hash_u64` — deterministic per-FQN seed derivation

`base.py`:

```python
def stable_hash_u64(global_seed: int, fqn: str) -> int:
    """SHA-256 of f'{seed}:{fqn}'.encode(), first 8 bytes, big-endian uint64.
    Explicitly NOT Python's hash() (randomized per process)."""
```

Used by every `GradientMaskTransform` and by `LoRALinear` for adapter
init.

### 3.5 `_local_shard_of` — DTensor-aware global→local slicer

`base.py`:

```python
def _local_shard_of(
    global_tensor: torch.Tensor, like: torch.Tensor
) -> torch.Tensor:
    """If ``like`` is a DTensor, return the rank-local shard of
    ``global_tensor`` under ``like``'s device_mesh and placements.
    If ``like`` is a plain Tensor, return ``global_tensor``.
    Implementation: ``distribute_tensor(global_tensor, mesh,
    placements).to_local()``."""
```

Used wherever a mask/state needs to align with a parameter's shard.

### 3.6 `peft.testing` — toy transforms for abstraction tests

[`src/olmo_core/nn/peft/testing.py`](../src/olmo_core/nn/peft/testing.py)

Two minimal in-tree transforms that exist *only* to test the abstraction
itself:
- `_NoopModelTransform`: validates `apply` plumbing without changing
  anything.
- `_FullGradMaskTransform`: trivial mask = "everything trainable",
  validates `GradientMaskTransform` plumbing without doing real work.

Without these, the abstraction tests have to depend on LoRA or random
masking, which couples PR-level testing. Cheap insurance.

---

## 4. Lifecycle integration

### 4.1 Hook into `TransformerTrainModule`

One new field on the config, and one new field on the module:

```python
@dataclass
class TransformerTrainModuleConfig(Config):
    ...
    peft: Optional[PEFTConfig] = None

    def build(self, model, device=None):
        if self.peft is not None:
            self.peft.assert_compatible(self)   # raises if e.g. LoRA + TP
        ...
        return TransformerTrainModule(
            model=model,
            peft=self.peft,
            ...
        )
```

And the train module:

```python
def __init__(self, model, ..., peft: Optional[PEFTConfig] = None):
    super().__init__()
    # validation ...

    # PEFT: apply model transforms BEFORE parallelize_model.
    self.peft = peft
    if peft is not None:
        report = peft.apply_model_transforms(model)
        log.info(report.summary())

    # Parallelize. Init-weights now uses a skip predicate so that
    # transform-owned parameters keep their post-apply init.
    self.model = parallelize_model(
        model,
        ...
        init_weights_skip=(peft.init_weights_skip(model) if peft else None),
    )

    # Optimizer built over post-transform params; frozen params filtered
    # out automatically by OptimConfig.build_groups.
    self.optim = optim.build(self.model, strict=True)

def on_attach(self):
    super().on_attach()
    if self.peft is not None:
        self.peft.install_gradient_callback(self.trainer)
```

Three corollary code touches:

1. `init_weights` in
   [`src/olmo_core/nn/transformer/init.py`](../src/olmo_core/nn/transformer/init.py)
   grows an `init_weights_skip: Optional[Callable[[str], bool]] = None`
   parameter, passed through to wherever individual tensors are
   reset.
2. `Transformer.init_weights(...)` and `parallelize_model(...)` thread
   the same predicate through.
3. The existing `load_key_mapping` field on
   `TransformerTrainModuleConfig` is merged with
   `peft.checkpoint_key_mapping(model)` before being applied.

### 4.2 Ordering & composition

- Within `model_transforms`: applied in declared list order. A later
  transform sees the model produced by earlier ones. (Example pitfall:
  BitFit *after* LoRA re-enables biases on the now-frozen LoRA base
  linears. The summary log shows this; the user reorders if it's
  wrong.)
- Within `gradient_transforms`: applied in declared list order each
  step, on top of the same `param.grad`. (Example: `[noise, mask]` adds
  then masks; `[mask, noise]` masks then adds noise — different
  behavior.)
- Across the two lists: model transforms always run first (build time);
  gradient transforms always run after backward each step.

Surface this prominently in `PEFTConfig`'s docstring.

---

## 5. LoRA — concrete `ModelTransform`

Module: [`src/olmo_core/nn/peft/lora.py`](../src/olmo_core/nn/peft/lora.py).

### 5.1 Config

```python
@register_model_transform
@dataclass
class LoRAConfig(ModelTransform):
    name: ClassVar[str] = "lora"

    r: int = 8
    alpha: float = 16.0
    dropout: float = 0.0
    target_modules: List[str] = field(default_factory=list)
    target_fqn_patterns: Optional[List[str]] = None
    modules_to_save:     Optional[List[str]] = None
    init_seed: int = 0
    use_rslora: bool = False

    def apply(self, model): ...
    def init_weights_skip(self, model): ...
    def checkpoint_key_mapping(self, model): ...
    def assert_compatible(self, tm_config):
        if tm_config.tp_config is not None:
            raise OLMoConfigurationError(
                "LoRA + tensor parallelism is not supported in v1; "
                "drop tp_config or wait for the v2 LoRALinear.apply_tp."
            )
```

`target_modules` resolves through:

```python
LORA_TARGET_PRESETS: Dict[str, str] = {
    "q":         "blocks.*.attention.w_q",
    "k":         "blocks.*.attention.w_k",
    "v":         "blocks.*.attention.w_v",
    "o":         "blocks.*.attention.w_out",
    "qkv":       "blocks.*.attention.w_qkv",
    "gate_proj": "blocks.*.feed_forward.w1",
    "down_proj": "blocks.*.feed_forward.w2",
    "up_proj":   "blocks.*.feed_forward.w3",
    "lm_head":   "lm_head.w_out",
}
```

### 5.2 `LoRALinear`

```python
class LoRALinear(nn.Module):
    """Drop-in nn.Linear replacement. Output shape and dtype match.

    Parameter FQNs after wrap:
        before:  blocks.0.attention.w_q.weight
        after:   blocks.0.attention.w_q.base.weight    (frozen)
                 blocks.0.attention.w_q.lora_A.weight  (trainable)
                 blocks.0.attention.w_q.lora_B.weight  (trainable)
    """
    base:    nn.Linear
    lora_A:  nn.Linear   # bias=False, (r, in_features)
    lora_B:  nn.Linear   # bias=False, (out_features, r)
    dropout: nn.Module
    scaling: float       # alpha/r (or alpha/sqrt(r) for rsLoRA)

    def forward(self, x):
        return self.base(x) + self.scaling * self.lora_B(self.lora_A(self.dropout(x)))
```

Init: Kaiming(`A`) + zero(`B`), CPU `torch.Generator` seeded from
`stable_hash_u64(init_seed, fqn)` — bit-identical across machines.

### 5.3 `apply()`

1. Walk `named_modules()`. For each FQN matching any resolved glob:
   - Assert it's `nn.Linear`. Replace on parent with `LoRALinear`.
2. Walk `named_parameters()`. Set `requires_grad`:
   - `True` for `*.lora_A.*`, `*.lora_B.*`.
   - Unchanged for matches of `modules_to_save`.
   - `False` otherwise.
3. Return `ModelTransformReport`.

### 5.4 Parallelism

- **FSDP / DDP / AC**: no special handling. FSDP2 shards `LoRALinear`'s
  children transparently. Frozen-base grads skipped.
- **TP**: out of scope v1; `assert_compatible` raises. v2 adds
  `LoRALinear.apply_tp(plan, mesh, ...)` and dispatches from
  `Attention.apply_tp` / `FeedForward.apply_tp`.
- **FP8**: update the FP8 walker (under
  [`src/olmo_core/float8/`](../src/olmo_core/float8/)) to recurse into
  `LoRALinear` and swap only `.base`, leaving `lora_A` / `lora_B` in
  bf16.
- **PP / CP / EP**: no special handling.

### 5.5 Checkpointing

Adapter params round-trip through DCP without changes.
`checkpoint_key_mapping` renames `*.w_q.base.weight ↔ *.w_q.weight` so a
pre-LoRA checkpoint loads cleanly; missing adapter keys retain
post-`apply` init (strict=False warns).

---

## 6. Random gradient masking — concrete `GradientMaskTransform`

Module:
[`src/olmo_core/nn/peft/gradient_mask/random.py`](../src/olmo_core/nn/peft/gradient_mask/random.py).

### 6.1 Config

```python
@register_gradient_transform
@dataclass
class RandomGradientMaskConfig(GradientMaskTransform):
    name: ClassVar[str] = "random_gradient_mask"

    fraction_trainable: float = 0.1
    # Inherits target_module_types, target_param_kinds,
    # exclude_fqn_patterns, seed from GradientMaskTransform.

    def __post_init__(self):
        if not (0.0 < self.fraction_trainable <= 1.0):
            raise OLMoConfigurationError("'fraction_trainable' must be in (0, 1]")

    def generate_mask(self, fqn, global_shape, generator):
        probs = torch.rand(global_shape, generator=generator, dtype=torch.float32)
        return probs < self.fraction_trainable   # bool, CPU
```

The whole subclass is ~10 lines. That's the test of the
`GradientMaskTransform` abstraction.

### 6.2 What the base provides

```python
class GradientMaskTransform(GradientTransform, metaclass=ABCMeta):
    def pre_train(self, model):
        self._registry = ParameterMaskRegistry()
        for fqn, p in model.named_parameters():
            if not self._matches(fqn, p, model): continue
            seed = stable_hash_u64(self.seed, fqn)
            gen = torch.Generator(device="cpu").manual_seed(seed)
            global_mask = self.generate_mask(fqn, p.shape, gen)
            local_mask = _local_shard_of(global_mask, p)
            self._registry[fqn] = local_mask.to(p.device)
            del global_mask   # not retained

    def transform(self, model):
        for fqn, p in model.named_parameters():
            mask = self._registry.get(fqn)
            if mask is None or p.grad is None: continue
            get_local_tensor(p.grad).mul_(mask)
```

### 6.3 Determinism properties (worth asserting in tests)

- **World-size invariance**: same seed, same model, two different DP
  world sizes → concatenated local shards reconstruct the same global
  mask, bit-identically.
- **Cross-platform**: same seed, same model, two different CPU
  architectures → same global mask. CPU `torch.Generator` + SHA-256
  seeds.
- **No-leak**: `exclude_fqn_patterns` matches → no registry entry; grad
  untouched.

### 6.4 Interaction with optimizer state

Gradient masking does **not** fully freeze masked entries under AdamW:
weight decay shrinks `p` even when `grad=0`. This is documented behavior
in `GradientMaskTransform`'s docstring. Workarounds: `weight_decay=0`
(common in PEFT runs), or combine with a `ModelTransform` that freezes
whole parameters.

---

## 7. What the abstraction buys you — catalog of follow-ons

| Method | Abstraction | Estimated LoC | Notes |
|---|---|---|---|
| **BitFit** | `FreezeFQNPatternsTransform` subclass | ~10 | freeze=`["*"]`, except=`["*.bias"]` |
| **HeadOnly** | `FreezeFQNPatternsTransform` subclass | ~10 | except=`["lm_head.*"]` |
| **TopNBlocks** | `FreezeFQNPatternsTransform` subclass | ~15 | computes except-pattern from N |
| **IA³** | new `ModelTransform`, new `IA3Module` | ~120 | mirrors LoRA's pattern |
| **Prefix/Prompt tuning** | new `ModelTransform` + forward-hook | ~150 | freezes everything else |
| **Random mask** | `GradientMaskTransform` subclass | ~10 | §6 |
| **Magnitude mask** | `GradientMaskTransform` subclass | ~15 | ranks by \|w\|, keeps top-k% |
| **Fisher mask** | `GradientTransform` subclass | ~60 | EMA Fisher → crystallize → mask; needs `state_dict` |
| **Structured mask** | `GradientMaskTransform` subclass | ~20 | whole-row/col/block |
| **LayerDrop** | `GradientTransform` subclass | ~30 | per-step random block zeroing |
| **Gradient noise** | `GradientTransform` subclass | ~20 | scaled Gaussian additive |
| **Gradient centralization** | `GradientTransform` subclass | ~15 | subtract per-tensor mean |

The point of this table: every method on it has a clear home in one of
two (or three) abstractions, with no special-casing. That's the
acceptance test of the design.

---

## 8. Intentionally **not** covered

- **GaLore-style projected optimizer state.** Belongs in a custom
  `OptimConfig` / `Optimizer`, not a `GradientTransform` (the
  projection geometry must be shared with the optimizer's moments).
- **QLoRA / 4-bit base weights.** Composes at the FP8 hook; the
  quantization is its own concern.
- **Adapter merge / `merge_and_unload`.** Inference utility, not a
  training-time abstraction.
- **HuggingFace `peft` import/export.** A converter on top of LoRA's
  FQN convention.
- **Mask schedules / curriculum PEFT.** Possible via a future
  `post_step` hook on `GradientTransform`; YAGNI for v1.

---

## 9. File layout

```
src/olmo_core/nn/peft/
    __init__.py                  # public re-exports
    base.py                      # ModelTransform, GradientTransform,
                                 #   GradientMaskTransform, PEFTConfig,
                                 #   ModelTransformReport, TrainabilityReport,
                                 #   stable_hash_u64, _local_shard_of,
                                 #   registry/decorator machinery
    freezing.py                  # FreezeFQNPatternsTransform,
                                 #   BitFitTransform, HeadOnlyTransform,
                                 #   TopNBlocksTransform
    lora.py                      # LoRAConfig, LoRALinear,
                                 #   LORA_TARGET_PRESETS
    gradient_mask/
        __init__.py
        _registry.py             # ParameterMaskRegistry (private)
        random.py                # RandomGradientMaskConfig
    testing.py                   # toy transforms for abstraction tests

src/olmo_core/train/callbacks/
    peft_gradient.py             # PEFTGradientCallback (auto-installed)

src/test/nn/peft/
    test_base.py                 # abstraction round-trip, registry,
                                 #   stable_hash, _local_shard_of
    test_freezing.py             # FreezeFQN + BitFit + HeadOnly
    test_lora.py                 # apply, init, checkpoint mapping
    test_gradient_mask_random.py # determinism, world-size invariance
```

`TransformerTrainModuleConfig` gains one field. `TransformerTrainModule`
gains one pre-parallelize call and one auto-installed callback. The
init pass in `nn/transformer/init.py` gains an optional `skip` parameter.
That's the entire intrusion into existing code.

---

## 10. Implementation phasing

1. **Abstractions PR.** `peft/base.py`,
   `train/callbacks/peft_gradient.py`, the `init_weights_skip`
   parameter in `init.py`, and the `peft` field on the train-module
   config. `peft/testing.py` for two toy transforms. Tests of the
   abstraction itself. No real PEFT method yet.
2. **Freezing primitives PR.** `peft/freezing.py` with
   `FreezeFQNPatternsTransform`, BitFit, HeadOnly. Cheap, tests
   composability without LoRA.
3. **LoRA PR.** `peft/lora.py` + the init-skip predicate + the FP8 hook
   change. FSDP integration tests. `assert_compatible` raises on TP.
   Checkpoint round-trip test.
4. **Random masking PR.** `peft/gradient_mask/random.py`. Determinism
   property tests, world-size-invariance test.
5. **Follow-ups** (each independent, no abstraction changes): TP-aware
   LoRA, magnitude mask, Fisher mask, structured mask, IA³, HF `peft`
   converter.

---

## 11. Open questions worth a decision now

- **Default `LoRAConfig.target_modules`** — recommend `["q", "k", "v",
  "o"]` (attention-only; original-paper recipe; standard `peft`
  default).
- **Discriminator default `name`** — for the abstract bases, use
  `"(abstract)"` and have the registry decorator refuse to register it.
  Prevents accidental "this base class is concrete" bugs.
- **`GradientMaskTransform` semantics under non-default
  `apply_to_grad_clipping=False`** — v1 always runs masks before
  clipping (because callbacks fire before `optim_step`); document the
  field as accepted-but-only-True-supported, and defer real support to
  when a user asks for it (would require a `post_clip` hook on the train
  module).
- **MoE router default exclusion** — recommend including
  `"*.feed_forward_moe.router.*"` in `RandomGradientMaskConfig`'s
  default `exclude_fqn_patterns`. Routers are tiny but
  disproportionately important; make the default safe, override
  explicit.
- **In-place vs. copy semantics for `ModelTransform.apply`** — in place.
  Document; 32B-scale runs cannot afford deep-copy.

---

## 12. Test inventory (recommended for the abstractions PR)

These tests are the contract for the abstractions. The PR is not done
until they all pass.

1. `test_registry_roundtrip`: `LoRAConfig(...).as_dict()` →
   `Config.from_dict(...)` → equal config.
2. `test_register_disallows_abstract`: `register_model_transform` on a
   class with `name == "(abstract)"` raises.
3. `test_apply_idempotency_guard`: applying the same `PEFTConfig`
   twice to the same model asserts (or logs and refuses).
4. `test_skip_predicate_or`: two model transforms each contributing a
   skip predicate combine via OR; the init pass respects both.
5. `test_checkpoint_mapping_conflict`: two model transforms mapping
   the same current-fqn to different checkpoint-fqns raise at config
   apply time.
6. `test_gradient_callback_order`: with two `GradientTransform`s, the
   second sees the gradient produced by the first.
7. `test_gradient_callback_state_dict`: round-trip through DCP.
8. `test_trainability_report_aggregation`: two model transforms
   producing different `params_frozen` counts aggregate correctly in
   `TrainabilityReport`.
9. `test_local_shard_of_under_fsdp`: `_local_shard_of` produces the
   correct slice under `Shard(0)`, `Shard(1)`, `Replicate`,
   multi-dim mesh. (Property tests with toy DTensor inputs.)
10. `test_stable_hash_u64_deterministic`: same `(seed, fqn)` → same
    output across Python invocations and platforms.

These tests catch the failure modes the abstractions exist to prevent.
If they don't catch a real bug later, they're documenting intent at
least.
