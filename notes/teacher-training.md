# Teacher-based Training: Simple Design & Implementation Plan

Scope: two new train modules for JOLMo that use a **frozen teacher model** during
training. We want something much simpler than `contrastive-learning.md` — no
GradCache, no similarity matrices, no all-gather gymnastics. Just a generic
teacher abstraction and two concrete loss setups.

The two train modules:

1. **`TransformerDistillTrainModule`** — standard autoregressive pretraining plus
   KL divergence between student and teacher next-token distributions. Teacher is
   a language model. Supports forward KL, reverse KL, or a sum of both.
2. **`TransformerRepLossTrainModule`** — standard autoregressive pretraining
   (head #1) plus an L2 regression loss between the student's *second* LM head
   output and the teacher's per-token embedding (head #1 is the normal vocab
   head; head #2 projects the student's last hidden state to the teacher's
   embedding space). Teacher is an embedding model.

Both modes follow the same design principles:

- **Follow the SAM pattern**: new train module alongside `TransformerTrainModule`
  (see [sam_train_module.py](../src/olmo_core/train/train_module/transformer/sam_train_module.py)).
  The `Trainer` class itself does not change.
- **No new Trainer**: "trainer" in the user's phrasing means *train module*.
  `Trainer.fit()` is already generic over `TrainModule`.
- **No GradCache**: everything is computed per-microbatch, student and teacher
  forward on the *same* microbatch, losses summed, one backward. Teacher stays
  in `torch.no_grad()` + `eval()`.
- **Teachers are JOLMo `Transformer` instances loaded from a checkpoint**, wrapped
  thinly. HuggingFace support is explicitly out-of-scope for v1; we can add it
  later if we need it. This keeps the teacher abstraction small and lets us
  reuse `parallelize_model` with FSDP for the teacher.

---

## 1. TeacherModel abstraction

A single file [`teacher.py`](../src/olmo_core/train/train_module/transformer/teacher.py)
that provides a small wrapper + config. The wrapper is the object that every
teacher-using train module holds.

```python
# teacher.py

@dataclass
class TeacherModelConfig(Config):
    """
    Configuration for a frozen JOLMo teacher model.

    The teacher is built with the same parallelization primitives as the
    student, shares the student's world mesh, and is held in eval mode with
    requires_grad=False on all parameters.
    """

    model: TransformerConfig
    """Full transformer config for the teacher."""

    checkpoint_path: str
    """Path or URL to the teacher's model-state checkpoint (unsharded or distributed)."""

    dp_config: Optional[TransformerDataParallelConfig] = None
    """
    FSDP/DDP config for the teacher. If None, the teacher is replicated on every
    rank (OK for ≤1B teachers with ~2GB bf16 weights). For larger teachers,
    specify FSDP.
    """

    autocast_precision: Optional[DType] = None
    """Autocast dtype for teacher forward. Usually bfloat16."""

    compile: bool = False
    """Whether to torch.compile the teacher's forward."""

    def build(
        self,
        *,
        world_mesh: Optional[DeviceMesh],
        device: torch.device,
        max_sequence_length: int,
        rank_microbatch_size: int,
    ) -> "TeacherModel":
        ...


class TeacherModel:
    """
    Holds a frozen Transformer and exposes two forward APIs.

    Loading flow (build()):
      1. model_config.build(init_device="meta")
      2. parallelize_model(..., dp_config=self.dp_config) — same helper the
         student uses; gets FSDP wrapping if requested
      3. load checkpoint via olmo_core.distributed.checkpoint utilities
      4. model.eval(); for p in model.parameters(): p.requires_grad_(False)
      5. optionally torch.compile(model)
    """

    model: Transformer
    device: torch.device
    autocast_precision: Optional[torch.dtype]

    @torch.no_grad()
    def logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Returns (B, S, vocab_size) logits. Uses return_logits=True."""

    @torch.no_grad()
    def hidden_states(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Returns the last hidden state (pre-LM-head), shape (B, S, d_teacher).
        Implemented via a forward hook registered on the final transformer block
        — the same trick the contrastive doc describes, but with no GradCache
        complications since we only ever run the teacher in no_grad.
        """
```

Implementation notes:

- The hook for `hidden_states` captures the output of `model.blocks[str(n_layers-1)]`
  inside a `_hidden_state_cache` list, runs `self.model(input_ids)` to populate it,
  then returns the cached tensor. Because we're inside `torch.no_grad()`, there's
  no graph.
- FSDP wrapping works with hooks — the hook fires after the block's `__call__`
  has already gathered parameters and produced output.
- For v1 we require that the teacher use a `Transformer` model (i.e. JOLMo
  architecture) loaded from a JOLMo checkpoint. If/when we need HF teachers,
  we add a subclass.
- `build()` is called from the train module's `__init__`, *after* the student
  has been parallelized, using the student's `world_mesh` so both live on the
  same GPUs.
- Teacher parameters are NOT in any optimizer. Teacher weights are NOT saved in
  any checkpoint — on resume, the teacher is rebuilt from config + its
  `checkpoint_path`.

---

## 2. DistillTrainModule (KL distillation)

File: [`distill_train_module.py`](../src/olmo_core/train/train_module/transformer/distill_train_module.py).

Extends `TransformerTrainModule`. All FSDP/optimizer/scheduler/checkpointing
machinery is inherited. We only override `__init__` (to build the teacher and
stash config) and `train_batch`.

### 2.1 Config

```python
@dataclass
class DistillConfig(Config):
    """Logit-distillation config for DistillTrainModule."""
    forward_kl_weight: float = 1.0      # weight on KL(teacher || student)
    reverse_kl_weight: float = 0.0      # weight on KL(student || teacher)
    temperature: float = 1.0            # temperature applied to both logits
    ce_loss_weight: float = 1.0         # weight on the normal CE loss
```

Setting one of `forward_kl_weight` / `reverse_kl_weight` to 0 disables that
term; both nonzero means the loss is `ce + fw_w * fwKL + rv_w * rvKL`. The
teacher vocab must equal the student vocab (assertion in `__init__`).

### 2.2 KL loss utilities

Live in `distill_train_module.py` near the class:

```python
def kl_from_logits(
    student_logits: Tensor,     # (N, V), N = num live tokens
    teacher_logits: Tensor,     # (N, V), detached
    *,
    direction: Literal["forward", "reverse"],
    temperature: float,
) -> Tensor:
    T = temperature
    s_log = F.log_softmax(student_logits / T, dim=-1)
    t_log = F.log_softmax(teacher_logits / T, dim=-1)
    if direction == "forward":
        # KL(teacher || student) = sum_v p_t * (log p_t - log p_s)
        t = t_log.exp()
        kl = (t * (t_log - s_log)).sum(-1)
    else:
        s = s_log.exp()
        kl = (s * (s_log - t_log)).sum(-1)
    return (kl.mean()) * (T ** 2)
```

Student logits are masked to live tokens (labels != ignore_index) before being
passed in — this avoids paying KL on padding / masked positions.

### 2.3 `__init__`

```python
class TransformerDistillTrainModule(TransformerTrainModule):
    def __init__(
        self,
        model: Transformer,
        teacher: TeacherModelConfig,
        distill: DistillConfig,
        **kwargs,             # everything TransformerTrainModule takes
    ):
        super().__init__(model=model, **kwargs)
        self.distill_config = distill
        self.teacher = teacher.build(
            world_mesh=self.world_mesh,
            device=self.device,
            max_sequence_length=self.max_sequence_length,
            rank_microbatch_size=self.rank_microbatch_size,
        )
        assert self.teacher.model.vocab_size == self.model.vocab_size, (
            "Distill requires same vocab for teacher and student"
        )
```

### 2.4 `train_batch`

Copy `TransformerTrainModule.train_batch()` verbatim (it's ~80 lines), and
inside the microbatch loop modify the student forward and add the KL term.

Key changes from the base implementation:

1. **Call `model_forward(..., return_logits=True)`**. The fused-linear loss path
   doesn't materialize logits, so we can't use it here. The rest of the LM head
   still runs normally. Memory cost: `B_micro * S * V * 2 bytes` extra per
   microbatch in bf16. For V=100K, S=2048, B_micro=32 this is ~13 GB — so
   users may need a smaller `rank_microbatch_size` with distillation.
2. **Teacher forward** inside `torch.no_grad()`, same `input_ids`.
3. **KL loss** on live tokens only; add to `loss` *before* `loss.backward()`.

Sketch of the microbatch body:

```python
_, loss, ce_loss, z_loss = self.model_forward(
    input_ids, labels=labels,
    ignore_index=self.label_ignore_index,
    loss_reduction="sum",
    z_loss_multiplier=self.z_loss_multiplier,
    loss_div_factor=batch_num_tokens_for_loss,
    return_logits=True,
    **model_kwargs,
)
student_logits = _get_logits_from_output(...)  # from model_forward return

# Teacher forward (no grad)
teacher_logits = self.teacher.logits(input_ids)  # (B_micro, S, V)

# Mask and flatten to live-token rows
live_mask = (labels != self.label_ignore_index)  # (B_micro, S)
s_flat = student_logits[live_mask]               # (N, V)
t_flat = teacher_logits[live_mask].detach()      # (N, V)

# Weighted CE + KL
kl_total = torch.zeros((), device=self.device)
if self.distill_config.forward_kl_weight > 0:
    fw = kl_from_logits(s_flat, t_flat,
                        direction="forward",
                        temperature=self.distill_config.temperature)
    kl_total = kl_total + self.distill_config.forward_kl_weight * fw
    fw_running += fw.detach()
if self.distill_config.reverse_kl_weight > 0:
    rv = kl_from_logits(s_flat, t_flat,
                        direction="reverse",
                        temperature=self.distill_config.temperature)
    kl_total = kl_total + self.distill_config.reverse_kl_weight * rv
    rv_running += rv.detach()

# Scale per-microbatch so total divides by batch_num_tokens_for_loss, matching
# how CE is scaled inside model_forward via loss_div_factor. This keeps the KL
# and CE on the same per-token footing regardless of microbatch size.
kl_total = kl_total * (live_mask.sum() / batch_num_tokens_for_loss)

loss = (self.distill_config.ce_loss_weight * loss) + kl_total
ce_batch_loss += get_local_tensor(ce_loss.detach())
loss.backward()
```

After the microbatch loop, record `train/KL forward`, `train/KL reverse` via
`self.record_metric(..., ReduceType.mean)` (same pattern as the base class's
`record_ce_loss`).

### 2.5 Checkpointing

Inherit `state_dict` / `load_state_dict` from the base class unchanged. Teacher
has no state to save (frozen + rebuilt from config). If we ever want to stash
the teacher config in the checkpoint for debugging, we can override
`_get_state_dict` to add a non-tensor `"teacher_config"` entry — but it's
purely informational.

### 2.6 Logit-distillation pitfalls

- **LMHead loss_implementation**: with `DistillTrainModule`, the LM head must
  use `loss_implementation=default` (not `fused_linear`). We assert this in
  `__init__`.
- **Autocast / dtype**: student logits are typically bf16 under autocast; the
  KL is computed in whatever dtype they come out as. `kl_from_logits` should
  cast inputs to fp32 before the softmaxes to avoid underflow when
  `temperature` is large. Tiny cost, large numerical safety margin.
- **TP/CP**: initial implementation disallows `tp_config`/`cp_config` (raise
  in `__init__`) — mirrors the SAM module's restriction. The student logits
  end up sharded under TP, which complicates the KL computation. Add support
  later if needed.

---

## 3. RepLossTrainModule (student-emb matches teacher-emb)

File: [`rep_loss_train_module.py`](../src/olmo_core/train/train_module/transformer/rep_loss_train_module.py).

Same pattern as `DistillTrainModule`: extends `TransformerTrainModule`,
overrides `__init__` and `train_batch`. The complication here is the **second
LM head** on the student that projects the last hidden state into the
teacher's embedding space.

### 3.1 Second LM head

Add a trainable `nn.Linear(d_student, d_teacher, bias=False)` (optionally with a
pre-norm) as a submodule on the train module, not inside the student `Transformer`
itself — the student's `Transformer` is frozen-in-structure after
`parallelize_model` runs, and we want to keep the teacher-specific head isolated
from the student's vocab head and from any HF export path.

Submodule name: `self.rep_head`. It lives on the same device as the student.
Its parameters are added to the optimizer via a separate param group after
`super().__init__(...)` finishes:

```python
self.rep_head = nn.Linear(
    student.d_model, teacher_d_model, bias=False, dtype=...
).to(self.device)
self.optim.add_param_group({
    "params": list(self.rep_head.parameters()),
    "lr": ...,      # same LR as the main group by default; configurable
    "weight_decay": 0.0,
    "initial_lr": ...,
})
```

Because `self.rep_head` is a regular `nn.Module`, `named_parameters()` at
checkpoint time picks it up (via our override of `_get_state_dict`, see §3.5).

**FSDP for the rep head**: for small d_teacher×d_student (a few million params
even for a big teacher) we don't bother sharding the rep head — it's replicated
per rank, and gradient sync happens via the manual all-reduce that DDP/FSDP
would normally do. Since `rep_head` is not inside the FSDP-wrapped student, its
gradients need explicit all-reduce when `is_distributed()`. We handle that in
`optim_step()` by all-reducing `rep_head.parameters()` grads before calling
`super().optim_step()`. (Alternative: wrap `rep_head` with DDP. Leaving that as
a follow-up — for a few-million-param head the manual all-reduce is cheaper
than the extra wrapping machinery.)

### 3.2 Extracting the student's last hidden state

Same forward-hook trick as in `TeacherModel.hidden_states`, registered on the
student's final block in `__init__`. Stored in `self._student_hidden_cache =
[None]`. The hook fires on every forward (including evals) but we only consume
it in `train_batch`, so that's fine.

Under FSDP2, register the hook on the inner module *after* `parallelize_model`
wraps it. The hook sees the output of the wrapped block post-all-gather, which
is what we want.

### 3.3 Config

```python
@dataclass
class RepLossConfig(Config):
    """Representation-loss config."""
    rep_loss_weight: float = 1.0
    """Weight on the L2 representation-matching loss."""

    ce_loss_weight: float = 1.0
    """Weight on the normal CE loss."""

    normalize: bool = False
    """
    If True, L2-normalize both student and teacher embeddings before computing
    the L2 distance (cosine-style). If False, plain L2 on raw vectors.
    """

    rep_head_layer_norm: bool = False
    """If True, apply a LayerNorm before the rep_head linear."""
```

The teacher's embedding dimension comes from `teacher_config.model.d_model` and
is what we size `rep_head` against.

### 3.4 `train_batch`

Inside the microbatch loop, after the student forward pass, the hook has
populated `self._student_hidden_cache[0]` with shape `(B_micro, S, d_student)`.
Apply `rep_head`, compare to teacher embeddings on live tokens:

```python
# Student forward (writes into _student_hidden_cache via hook)
_, loss, ce_loss, z_loss = self.model_forward(
    input_ids, labels=labels,
    ignore_index=self.label_ignore_index,
    loss_reduction="sum",
    z_loss_multiplier=self.z_loss_multiplier,
    loss_div_factor=batch_num_tokens_for_loss,
    return_logits=False,       # fused-linear is fine here
    **model_kwargs,
)
h = self._student_hidden_cache[0]           # (B_micro, S, d_student)

# Project to teacher dim
s_emb = self.rep_head(h)                    # (B_micro, S, d_teacher)

# Teacher embeddings (no grad)
t_emb = self.teacher.hidden_states(input_ids)  # (B_micro, S, d_teacher)

# Mask to live tokens
live_mask = (labels != self.label_ignore_index)  # (B_micro, S)
s_live = s_emb[live_mask]                   # (N, d_teacher)
t_live = t_emb[live_mask].detach()          # (N, d_teacher)

if self.rep_loss_config.normalize:
    s_live = F.normalize(s_live, dim=-1)
    t_live = F.normalize(t_live, dim=-1)

# Mean squared L2 distance per token, then scale like CE
rep_loss_sum = (s_live - t_live).pow(2).sum(dim=-1).sum()
rep_loss = rep_loss_sum / batch_num_tokens_for_loss

loss = (self.rep_loss_config.ce_loss_weight * loss) + \
       (self.rep_loss_config.rep_loss_weight * rep_loss)

ce_batch_loss += get_local_tensor(ce_loss.detach())
rep_batch_loss += rep_loss.detach()
loss.backward()
```

After the loop, record `train/rep loss` with `ReduceType.mean`.

### 3.5 Checkpointing additions

Override `_get_state_dict` / `load_state_dict` to round-trip `rep_head`:

```python
def _get_state_dict(self, sd_options, optim=True):
    sd = super()._get_state_dict(sd_options, optim=optim)
    sd["rep_head"] = self.rep_head.state_dict()
    return sd

def load_state_dict(self, state_dict):
    rep_state = state_dict.pop("rep_head", None)
    super().load_state_dict(state_dict)
    if rep_state is not None:
        self.rep_head.load_state_dict(rep_state)
```

The `optim` state dict already includes the rep_head parameters' optimizer
states because they live in `self.optim` as an extra param group.

### 3.6 Gradient all-reduce for `rep_head`

Override `optim_step()`:

```python
def optim_step(self):
    if is_distributed() and self.dp_process_group is not None:
        for p in self.rep_head.parameters():
            if p.grad is not None:
                dist.all_reduce(p.grad, op=dist.ReduceOp.AVG,
                                group=self.dp_process_group)
    super().optim_step()
```

This runs after `train_batch()` has accumulated grads across all microbatches
on every rank. The student's grads were already all-reduced by FSDP/DDP during
backward; we only need to handle `rep_head` here.

### 3.7 Rep-loss pitfalls

- **Teacher `d_model` mismatch**: if the teacher is, say, 2560-dim and the
  student is 1024-dim, `rep_head` is `Linear(1024, 2560)` = ~2.6M params. Tiny,
  but on 70B-class teachers (~8192-dim) it grows to ~8M — still fine.
- **Last-block output ≠ post-final-norm**: the hook captures the block output,
  *before* the LM head's `layer_norm` (if any). That's what we want —
  representations from the backbone, independent of the vocab head. Make sure
  this matches whichever layer in the teacher `hidden_states()` returns (it
  does: teacher's hook is also on the final block).
- **Autocast dtype**: both student hidden states and teacher embeddings will
  be bf16 under autocast. L2 in bf16 is noisy; cast to fp32 before the
  subtraction.
- **Numerical scale**: L2 distance grows with `d_teacher`. For d=2048, random
  pairs have squared-L2 ≈ `2*d*Var` ≈ hundreds or thousands. Pick
  `rep_loss_weight` accordingly — starting point ~0.01, adjust after seeing
  initial values in wandb.

---

## 4. Wiring into the launch path

### 4.1 Register new `train_module_type`s

In [`launch_from_yaml.py`](../src/scripts/launch_from_yaml.py), add fields on
`YamlExperimentConfig`:

```python
teacher: Optional[TeacherModelConfig] = None
distill: Optional[DistillConfig] = None
rep_loss: Optional[RepLossConfig] = None
```

and two new builder functions that mirror `_build_train_module_sam`:

```python
def _build_train_module_distill(cfg, model) -> TrainModule:
    kwargs = _common_train_module_kwargs(cfg)  # extract from cfg.train_module
    if cfg.teacher is None or cfg.distill is None:
        raise ValueError("train_module_type=distill requires 'teacher' and 'distill' sections")
    return TransformerDistillTrainModule(
        model=model, teacher=cfg.teacher, distill=cfg.distill, **kwargs,
    )

def _build_train_module_rep_loss(cfg, model) -> TrainModule:
    kwargs = _common_train_module_kwargs(cfg)
    if cfg.teacher is None or cfg.rep_loss is None:
        raise ValueError("train_module_type=rep_loss requires 'teacher' and 'rep_loss' sections")
    return TransformerRepLossTrainModule(
        model=model, teacher=cfg.teacher, rep_loss=cfg.rep_loss, **kwargs,
    )
```

Extract the autocast/state-dict-opts handling currently inlined in
`_build_train_module_sam` into a private helper `_common_train_module_kwargs`
to avoid a third copy.

Route in `main()`:

```python
tmt = cfg.train_module_type.lower()
if tmt == "sam":
    train_module = _build_train_module_sam(cfg, model)
elif tmt == "distill":
    train_module = _build_train_module_distill(cfg, model)
elif tmt == "rep_loss":
    train_module = _build_train_module_rep_loss(cfg, model)
else:
    train_module = _build_train_module_normal(cfg, model)
```

### 4.2 Exports

[`JOLMo/src/olmo_core/train/train_module/__init__.py`](../src/olmo_core/train/train_module/__init__.py)
gets `TransformerDistillTrainModule`, `TransformerRepLossTrainModule`,
`DistillConfig`, `RepLossConfig`, `TeacherModel`, `TeacherModelConfig`.

### 4.3 `PretrainedModel` in `mixture-pretraining`

Two subclasses in
[`mixture_pretraining_stages/unordered/training.py`](../../mixture-pretraining/mixture_pretraining_stages/unordered/training.py):

```python
@dataclass(frozen=True)
class DistillPretrainedModel(PretrainedModel):
    teacher_model_type: str = "1B"              # "610M", "1B", "2.5B"
    teacher_checkpoint_path: str = ""
    teacher_dp_fsdp: bool = True
    forward_kl_weight: float = 1.0
    reverse_kl_weight: float = 0.0
    kl_temperature: float = 1.0
    ce_loss_weight: float = 1.0

    def _build_yaml_config(self, ...):
        cfg = super()._build_yaml_config(...)
        cfg["train_module_type"] = "distill"
        cfg["teacher"] = {"_CLASS_": ..., ...}
        cfg["distill"] = {"_CLASS_": ..., ...}
        # Force the LM head to use non-fused loss, since distill needs logits
        cfg["model"]["lm_head"]["loss_implementation"] = "default"
        return cfg


@dataclass(frozen=True)
class RepLossPretrainedModel(PretrainedModel):
    teacher_model_type: str = "1B"
    teacher_checkpoint_path: str = ""
    teacher_dp_fsdp: bool = True
    rep_loss_weight: float = 0.01
    ce_loss_weight: float = 1.0
    rep_normalize: bool = False
    rep_head_layer_norm: bool = False

    def _build_yaml_config(self, ...):
        cfg = super()._build_yaml_config(...)
        cfg["train_module_type"] = "rep_loss"
        cfg["teacher"] = {"_CLASS_": ..., ...}
        cfg["rep_loss"] = {"_CLASS_": ..., ...}
        # fused-linear loss is fine for rep_loss (no logits needed)
        return cfg
```

---

## 5. Example YAML fragments

Distillation:

```yaml
train_module_type: distill
teacher:
  _CLASS_: olmo_core.train.train_module.transformer.teacher.TeacherModelConfig
  model:
    _CLASS_: olmo_core.nn.transformer.config.TransformerConfig
    d_model: 1280
    n_layers: 30
    # ... full 1B config ...
  checkpoint_path: /path/to/teacher-1B/checkpoint
  autocast_precision: bfloat16
  dp_config:
    _CLASS_: olmo_core.train.train_module.transformer.config.TransformerDataParallelConfig
    name: fsdp
    param_dtype: bfloat16
    reduce_dtype: float32
distill:
  _CLASS_: olmo_core.train.train_module.transformer.distill_train_module.DistillConfig
  forward_kl_weight: 1.0
  reverse_kl_weight: 0.0
  temperature: 1.0
  ce_loss_weight: 1.0
```

Rep-loss:

```yaml
train_module_type: rep_loss
teacher:
  _CLASS_: olmo_core.train.train_module.transformer.teacher.TeacherModelConfig
  model:
    _CLASS_: olmo_core.nn.transformer.config.TransformerConfig
    # ... full teacher config ...
  checkpoint_path: /path/to/teacher/checkpoint
  autocast_precision: bfloat16
  dp_config:
    _CLASS_: olmo_core.train.train_module.transformer.config.TransformerDataParallelConfig
    name: fsdp
    param_dtype: bfloat16
    reduce_dtype: float32
rep_loss:
  _CLASS_: olmo_core.train.train_module.transformer.rep_loss_train_module.RepLossConfig
  rep_loss_weight: 0.01
  ce_loss_weight: 1.0
  normalize: false
  rep_head_layer_norm: false
```

---

## 6. File structure summary

```
JOLMo/src/olmo_core/train/train_module/transformer/
  teacher.py                     (new) TeacherModelConfig, TeacherModel
  distill_train_module.py        (new) DistillConfig, TransformerDistillTrainModule
  rep_loss_train_module.py       (new) RepLossConfig, TransformerRepLossTrainModule

JOLMo/src/olmo_core/train/train_module/__init__.py
  + export new classes

JOLMo/src/scripts/launch_from_yaml.py
  + YamlExperimentConfig fields
  + _build_train_module_distill / _build_train_module_rep_loss
  + main() routing

mixture_pretraining_stages/unordered/training.py
  + DistillPretrainedModel
  + RepLossPretrainedModel
```

---

## 7. Implementation checklist

1. [ ] `teacher.py`: `TeacherModelConfig`, `TeacherModel`, hook-based
       `hidden_states`, checkpoint load, FSDP wrap.
2. [ ] `distill_train_module.py`: `DistillConfig`, `kl_from_logits`,
       `TransformerDistillTrainModule.__init__` + `train_batch`, metric
       recording.
3. [ ] `rep_loss_train_module.py`: `RepLossConfig`, hook on student final
       block, `rep_head` linear + param group, `train_batch`, `optim_step`
       grad all-reduce, state dict round-trip.
4. [ ] Export new classes from the train_module `__init__.py`.
5. [ ] Wire into `launch_from_yaml.py` (extract `_common_train_module_kwargs`,
       add two builders + routing).
6. [ ] Add `DistillPretrainedModel` and `RepLossPretrainedModel` subclasses.
7. [ ] Dry-run a tiny student + tiny teacher (e.g. 160M + 160M) to verify the
       full launch path.
8. [ ] Single-GPU smoke test for both modules.
9. [ ] Multi-GPU FSDP test (student FSDP + teacher FSDP).

---

## 8. Explicitly out of scope for v1

- HuggingFace teachers (we can add an `HFTeacherModel` subclass later).
- GradCache / pairwise similarity / contrastive matrix objectives — if we ever
  want that, it's a separate train module, see `contrastive-learning.md`.
- Cross-tokenizer / cross-vocab distillation (we assume vocab match and assert
  it).
- Tensor/context/pipeline parallelism for the student in these modules. Start
  with DP (DDP or FSDP). Error out if TP/CP/PP configured.
- Precomputed offline teacher logits — saves compute for very large teachers
  but complicates the data pipeline. Add only when we have a teacher big
  enough that running it online is painful.
- Per-layer or multi-layer rep matching (we use only the last block's output).
