# Teacher-based Training: Simple Design & Implementation Plan

Scope: two new train modules for JOLMo that use a **frozen teacher model**
during training. Much simpler than `contrastive-learning.md` — no GradCache,
no similarity matrices, no all-gather gymnastics. Just a generic teacher
abstraction and two concrete loss setups.

The two train modules:

1. **`TransformerDistillTrainModule`** — standard autoregressive pretraining
   plus KL divergence between student and teacher next-token distributions.
2. **`TransformerRepLossTrainModule`** — standard autoregressive pretraining
   (head #1) plus an L2 regression loss between the student's *second* LM
   head output and the teacher's per-token embedding.

Design principles:
- **Follow the SAM pattern**: new train module alongside
  `TransformerTrainModule`. `Trainer` itself does not change.
- **No GradCache**: per-microbatch student + teacher forward on the *same*
  tokens; losses summed; one backward. Teacher stays in `torch.no_grad()` +
  `eval()` + `requires_grad=False`.
- **Teachers are JOLMo `Transformer` instances** loaded from a distributed
  checkpoint. HF teachers are out-of-scope for v1.
- **rep_head uses its own dedicated AdamW optimizer**, decoupled from the
  student's optimizer. This sidesteps DCP's model-scoped optimizer-state
  handling, avoids Muon-vs-rep_head wiring issues, and gives rep_head a
  clean round-trip through the checkpoint machinery.

---

## 1. TeacherModel abstraction

File: [`teacher.py`](../src/olmo_core/train/train_module/transformer/teacher.py).

```python
@dataclass
class TeacherModelConfig(Config):
    model: TransformerConfig
    """Full transformer config for the teacher."""

    checkpoint_path: str
    """Path/URL to the teacher's model-state distributed checkpoint."""

    dp_config: Optional[TransformerDataParallelConfig] = None
    """FSDP/DDP config for the teacher. None → replicated per rank. Default
    reshard_after_forward=True is automatic via parallelize_model (confirmed
    at nn/transformer/model.py:757 for non-PP)."""

    autocast_precision: Optional[DType] = None
    """Autocast dtype for teacher forward (typically bfloat16)."""

    compile: bool = False
    """Whether to torch.compile the teacher's forward."""

    trained_sequence_length: Optional[int] = None
    """Sequence length the teacher was TRAINED at. Used to reject configs
    where student max_sequence_length > teacher_trained_sequence_length
    (RoPE extrapolation produces meaningless teacher signal). If None, the
    check is skipped — callers should set this explicitly."""
```

The field name is deliberately **not** `max_sequence_length` because
`TeacherModel.build()` is also called with a `max_sequence_length` kwarg
(the student's seq-len, passed to `parallelize_model` for FSDP/activation
sizing). Using the same name would collide on `TeacherModel` attribute
assignment and make the RoPE guard vacuous.

### 1.1 `TeacherModel.build` — concrete loading sequence

```python
class TeacherModel:
    model: Transformer
    device: torch.device
    autocast_precision: Optional[torch.dtype]
    trained_sequence_length: Optional[int]
    _hidden_state_cache: List[Optional[torch.Tensor]]
```

Build flow inside `TeacherModelConfig.build(student_world_mesh, device,
student_max_sequence_length, rank_microbatch_size)`:

1. `model = self.model.build(init_device="meta")`.
2. `parallelize_model(model, world_mesh=student_world_mesh,
   device=device, max_sequence_length=student_max_sequence_length,
   rank_microbatch_size=rank_microbatch_size,
   compile_model=self.compile, dp_config=self.dp_config)`. Explicit pass of
   `max_sequence_length` and `rank_microbatch_size` matches the student
   call (train_module.py:158-171). `ac_config` / `float8_config` /
   `tp_config` / `cp_config` / `ep_config` are None by design: teacher is
   inference-only.
3. `load_model_and_optim_state(self.checkpoint_path, model, optim=None)`
   from `olmo_core.distributed.checkpoint` overwrites the random init.
   `optim=None` is supported (confirmed at checkpoint/__init__.py:290-381).
   `gc_cuda()` after to release the random-init storage.
4. `model.eval()`.
5. `for p in model.parameters(): p.requires_grad_(False)`.
6. `assert all(not p.requires_grad for p in model.parameters())`.
7. Mesh-consistency assert: if `student_world_mesh is not None`, the
   teacher's resulting `dp_process_group` must be the same
   `dist.ProcessGroup` object as the student's.
8. Construct the `TeacherModel`, storing
   `trained_sequence_length=self.trained_sequence_length` (NOT the student
   kwarg). Register the last-block forward hook (§1.2).

### 1.2 Hidden-state capture and autocast

```python
@contextlib.contextmanager
def _infer_ctx(self):
    with torch.no_grad():
        if self.autocast_precision is not None:
            with torch.autocast(self.device.type, dtype=self.autocast_precision):
                yield
        else:
            yield

def logits(self, input_ids, **kwargs) -> Tensor:
    """(B, S, vocab_size). kwargs forwards doc_lens/max_doc_lens/
    cache_leftpad so teacher respects the same intra-document masking as
    the student."""
    with self._infer_ctx():
        return self.model(input_ids, return_logits=True, **kwargs)

def hidden_states(self, input_ids, **kwargs) -> Tensor:
    """(B, S, d_teacher), captured via forward hook on the last block."""
    with self._infer_ctx():
        self.model(input_ids, **kwargs)
    h = self._hidden_state_cache[0]
    self._hidden_state_cache[0] = None
    return h
```

Hook registration in `build()`, after parallelize_model:
```python
last_block = model.blocks[str(model.n_layers - 1)]
last_block.register_forward_hook(
    lambda m, inp, out: self._hidden_state_cache.__setitem__(0, out)
)
```

**Intra-document masking is critical.** Callers must forward
`doc_lens` / `max_doc_lens` / `cache_leftpad` to the teacher; otherwise
teacher attention crosses packed-document boundaries while the student
doesn't. The train modules do this explicitly (§2.4, §3.4).

### 1.3 Ordering: teacher-before-student

Teacher forward runs *first* in every microbatch. Teacher is `no_grad`;
activations and parameters reshard immediately after forward. This keeps
teacher peak from overlapping with the student's saved-for-backward
activations.

---

## 2. DistillTrainModule

File: [`distill_train_module.py`](../src/olmo_core/train/train_module/transformer/distill_train_module.py).

Extends `TransformerTrainModule`. Overrides `__init__` and `train_batch`.

### 2.1 Config

```python
@dataclass
class DistillConfig(Config):
    forward_kl_weight: float = 1.0
    """Weight on KL(teacher || student)."""

    reverse_kl_weight: float = 0.0
    """Weight on KL(student || teacher). See allow_reverse_kl_from_scratch
    and reverse_kl_warmup_steps."""

    temperature: float = 1.0
    """Softmax temperature for both logits before KL."""

    ce_loss_weight: float = 1.0
    """Weight on the base LM loss (CE, or CE + z_loss when z_loss_multiplier
    is set). Setting this != 1.0 with z_loss enabled is REJECTED in
    __init__ — see §2.3."""

    allow_reverse_kl_from_scratch: bool = False
    """If True, disables the reverse-KL divergence guard. See §2.3."""

    reverse_kl_warmup_steps: Optional[int] = None
    """If set, reverse_kl_weight is ramped linearly from 0 to its full value
    over this many steps. This is the preferred way to combine reverse KL
    with a random student: setting warmup_steps suppresses the guard."""
```

Final per-token loss:
`ce_loss_weight * LM_loss + forward_kl_weight * KL_fw + effective_reverse_kl_weight * KL_rv`
where `effective_reverse_kl_weight` is the warmup-ramped value.

**Resume drift**: `DistillConfig` and `teacher.checkpoint_path` are rebuilt
from YAML every run and NOT persisted. `__init__` logs the full config.
Persisting a hash is a follow-up.

**Eval**: `eval_batch` inherits unchanged — validation reports student CE
only. This is a known v1 limitation. **Users comparing distill runs to a
baseline should rely on downstream evals, not validation CE** (a well-
distilled student may have slightly worse standalone CE while performing
better on real tasks).

### 2.2 KL loss utility

```python
def kl_per_token(
    student_logits: Tensor,     # (B, S, V), grad-tracking, typically bf16
    teacher_logits: Tensor,     # (B, S, V), detached, typically bf16
    *,
    direction: Literal["forward", "reverse"],
    temperature: float,
) -> Tensor:
    """Returns per-token KL of shape (B, S). Caller masks and reduces."""
    T = temperature
    # dtype=fp32 promotes inside the kernel; the fp32 OUTPUT (B,S,V) is
    # saved by autograd for backward. This is unavoidable for stability.
    s_log = F.log_softmax(student_logits / T, dim=-1, dtype=torch.float32)
    t_log = F.log_softmax(teacher_logits / T, dim=-1, dtype=torch.float32)
    if direction == "forward":
        kl_elem = F.kl_div(s_log, t_log, reduction="none", log_target=True)
    else:
        kl_elem = F.kl_div(t_log, s_log, reduction="none", log_target=True)
    # T^2 scaling is Hinton's convention. The gradient-magnitude argument
    # (~1/T through softmax chain rule) applies to both directions.
    return kl_elem.sum(dim=-1) * (T ** 2)
```

Masking is post-softmax on the `(B, S)` result, not pre-softmax on
`(B, S, V)` logits (the latter is a multi-GB copy for V~100K).

### 2.3 `__init__`

```python
class TransformerDistillTrainModule(TransformerTrainModule):
    def __init__(
        self,
        model: Transformer,
        teacher: TeacherModelConfig,
        distill: DistillConfig,
        **kwargs,
    ):
        for k in ("tp_config", "cp_config", "pp_config"):
            if kwargs.get(k) is not None:
                raise OLMoConfigurationError(
                    f"TransformerDistillTrainModule does not support {k} in v1"
                )
        super().__init__(model=model, **kwargs)
        self.distill_config = distill

        self.teacher = teacher.build(
            student_world_mesh=self.world_mesh,
            device=self.device,
            student_max_sequence_length=self.max_sequence_length,
            rank_microbatch_size=self.rank_microbatch_size,
        )

        # Vocab match.
        if self.teacher.model.vocab_size != self.model.vocab_size:
            raise OLMoConfigurationError(
                f"Distill requires matching vocab; teacher="
                f"{self.teacher.model.vocab_size}, student="
                f"{self.model.vocab_size}"
            )

        # LM head checks.
        if isinstance(self.model.lm_head, NormalizedLMHead):
            raise OLMoConfigurationError(
                "Distill does not support NormalizedLMHead (sz scaling "
                "conflicts with KL temperature)."
            )
        if self.model.lm_head.loss_implementation != LMLossImplementation.default:
            raise OLMoConfigurationError(
                "Distill requires loss_implementation='default' (fused-linear "
                "does not return logits)."
            )

        # Teacher context window.
        if (self.teacher.trained_sequence_length is not None
            and self.teacher.trained_sequence_length < self.max_sequence_length):
            raise OLMoConfigurationError(
                f"Teacher trained at seq_len={self.teacher.trained_sequence_length}; "
                f"student uses {self.max_sequence_length}. RoPE extrapolation "
                f"produces meaningless teacher signal."
            )

        # Reverse-KL guard. If a warmup is configured OR the opt-out is set,
        # we allow any ratio. Otherwise require forward to dominate by 2x.
        if (distill.reverse_kl_weight > 0
            and distill.reverse_kl_warmup_steps is None
            and not distill.allow_reverse_kl_from_scratch
            and distill.forward_kl_weight < 2.0 * distill.reverse_kl_weight):
            raise OLMoConfigurationError(
                "Reverse-KL with forward_kl_weight < 2 * reverse_kl_weight "
                "diverges from a random student (p_t → 0 regions). Either "
                "(a) set forward_kl_weight >= 2 * reverse_kl_weight, "
                "(b) set reverse_kl_warmup_steps > 0 to ramp reverse KL from "
                "zero, or (c) if warm-starting, set "
                "allow_reverse_kl_from_scratch=True."
            )

        # z_loss × ce_loss_weight is silently broken (z_loss gets rescaled
        # exactly when it matters most, early training). Reject.
        if self.z_loss_multiplier is not None and distill.ce_loss_weight != 1.0:
            raise OLMoConfigurationError(
                "ce_loss_weight != 1.0 is not supported with z_loss enabled; "
                "z_loss is folded into the base model_forward loss and would "
                "be silently rescaled. Set z_loss_multiplier=None or "
                "ce_loss_weight=1.0."
            )

        log.info(f"DistillConfig: {distill}")
        V = self.model.vocab_size
        logit_gib = self.rank_microbatch_size * V * 2 / 2**30
        # Realistic peak during KL+CE backward, bf16 student logits + bf16
        # teacher logits + two fp32 log-softmax outputs saved for KL backward
        # + one fp32 logits copy saved inside cross_entropy_loss (confirmed
        # at nn/functional/cross_entropy_loss.py:35 — `logits.float()`
        # saved for backward) + one fp32 log_softmax saved by F.cross_entropy
        # itself. Roughly 10× the bf16 logit size.
        log.info(
            f"Distill memory note: peak during backward ≈ "
            f"{logit_gib * 10:.1f} GiB per microbatch. For V~100K, S=2048 on "
            f"80GB GPUs, rank_microbatch_size=4×S (B_µ=4) is the safe "
            f"default. Lower if OOM."
        )
```

### 2.4 `train_batch`

Copy `TransformerTrainModule.train_batch` as a starting point. Preserve
every existing metric record (instance_mask %, masked-labels %, auxiliary
metrics). Honor `dry_run=True` via the base's post-batch early return.

```python
def train_batch(self, batch, dry_run=False):
    self._set_model_mode("train")
    if "labels" not in batch:
        batch["labels"] = get_labels(batch, label_ignore_index=self.label_ignore_index)
    # ... base instance_mask / masked-labels-% recording ...

    batch_num_tokens_for_loss = move_to_device(
        (batch["labels"] != self.label_ignore_index).sum(), self.device
    )
    ce_batch_loss = move_to_device(torch.tensor(0.0), self.device)
    kl_fw_batch   = move_to_device(torch.tensor(0.0), self.device)
    kl_rv_batch   = move_to_device(torch.tensor(0.0), self.device)
    z_batch_loss  = (move_to_device(torch.tensor(0.0), self.device)
                    if self.z_loss_multiplier is not None else None)

    # Effective reverse KL weight with optional linear warmup.
    rv_w = self.distill_config.reverse_kl_weight
    if (rv_w > 0 and self.distill_config.reverse_kl_warmup_steps is not None):
        ramp = min(1.0, self.trainer.global_step /
                   max(1, self.distill_config.reverse_kl_warmup_steps))
        rv_w = rv_w * ramp

    seq_len = batch["input_ids"].shape[1]
    micro_batches = split_batch(batch, self.rank_microbatch_size // seq_len)
    num_micro_batches = len(micro_batches)

    for i, micro_batch in enumerate(micro_batches):
        with self._train_microbatch_context(i, num_micro_batches):
            input_ids, labels, model_kwargs = self._prepare_batch(micro_batch)
            teacher_kwargs = {
                k: model_kwargs[k]
                for k in ("doc_lens", "max_doc_lens", "cache_leftpad")
                if k in model_kwargs
            }

            # Teacher first.
            teacher_logits = self.teacher.logits(input_ids, **teacher_kwargs)

            # Student.
            output = self.model_forward(
                input_ids, labels=labels,
                ignore_index=self.label_ignore_index,
                loss_reduction="sum",
                z_loss_multiplier=self.z_loss_multiplier,
                loss_div_factor=batch_num_tokens_for_loss,
                return_logits=True,
                **model_kwargs,
            )
            student_logits = output.logits
            loss, ce_loss, z_loss = output.loss, output.ce_loss, output.z_loss

            live_mask = (labels != self.label_ignore_index)

            kl_term = torch.zeros((), device=self.device)
            if self.distill_config.forward_kl_weight > 0:
                kl_fw = kl_per_token(student_logits, teacher_logits.detach(),
                                     direction="forward",
                                     temperature=self.distill_config.temperature)
                s = (kl_fw * live_mask).sum()
                kl_term = kl_term + self.distill_config.forward_kl_weight * s
                kl_fw_batch += (s / live_mask.sum().clamp(min=1)).detach()
            if rv_w > 0:
                kl_rv = kl_per_token(student_logits, teacher_logits.detach(),
                                     direction="reverse",
                                     temperature=self.distill_config.temperature)
                s = (kl_rv * live_mask).sum()
                kl_term = kl_term + rv_w * s
                kl_rv_batch += (s / live_mask.sum().clamp(min=1)).detach()

            kl_term = kl_term / batch_num_tokens_for_loss

            total = self.distill_config.ce_loss_weight * loss + kl_term
            ce_batch_loss += get_local_tensor(ce_loss.detach())
            if z_batch_loss is not None and z_loss is not None:
                z_batch_loss += get_local_tensor(z_loss.detach())

            total.backward()

            # Release Python name-bindings so the CUDA caching allocator can
            # reuse blocks on the next microbatch. Autograd freed saved-for-
            # backward tensors during .backward().
            del output, student_logits, teacher_logits, total, loss

    self.model.post_batch(dry_run=dry_run)
    if dry_run:
        self.model.reset_auxiliary_metrics()
        return

    # SkipStep support (base pattern, train_module.py:502-510).
    if isinstance(self.optim, SkipStepOptimizer):
        if is_distributed():
            ce_batch_loss.div_(self._reduce_divide_factor)
            dist.all_reduce(ce_batch_loss)
            ce_batch_loss.div_(self.world_size)
            ce_batch_loss.mul_(self._reduce_divide_factor)
        self.record_ce_loss(ce_batch_loss)
        self.optim.latest_loss = ce_batch_loss
    else:
        self.record_ce_loss(ce_batch_loss, ReduceType.mean)

    if self.distill_config.forward_kl_weight > 0:
        self.record_metric("KL forward", kl_fw_batch, ReduceType.mean, namespace="train")
    if self.distill_config.reverse_kl_weight > 0:
        self.record_metric("KL reverse", kl_rv_batch, ReduceType.mean, namespace="train")
        if self.distill_config.reverse_kl_warmup_steps is not None:
            self.record_metric("KL reverse effective weight", rv_w, namespace="train")
    if z_batch_loss is not None:
        self.record_metric("Z loss", z_batch_loss, ReduceType.mean, namespace="train")
        self.record_metric("Z loss unscaled",
                           z_batch_loss / self.z_loss_multiplier,
                           ReduceType.mean, namespace="train")

    for name, (val, reduction) in self.model.compute_auxiliary_metrics(reset=True).items():
        self.record_metric(name, val, reduction, namespace="train")
```

### 2.5 Checkpointing

Inherit unchanged. Teacher has no state to persist; rebuilt from config on
resume.

---

## 3. RepLossTrainModule

File: [`rep_loss_train_module.py`](../src/olmo_core/train/train_module/transformer/rep_loss_train_module.py).

Same pattern as DistillTrainModule. The novel pieces are a **second
trainable LM head** and **a dedicated AdamW optimizer for it**.

### 3.1 `rep_head` + `rep_optim`

`rep_head` is `nn.Linear(d_student, d_teacher, bias=True)`, optionally
preceded by a `LayerNorm`. It lives **outside** `self.model` and has
**its own AdamW optimizer** `self.rep_optim`. This decoupling is the
key simplification over earlier revisions:

- `self.optim` sees only the student's parameters (FSDP-aware, DCP
  handles it natively).
- `self.rep_optim` sees only rep_head's parameters (plain tensors,
  DDP-averaged gradients, standard torch state_dict round-trip).
- Works correctly under any student optimizer (AdamW / Muon / etc.) —
  rep_head never goes through Muon's sub-optimizer partitioning.
- On resume, if the checkpoint was saved by a rep_loss run, both model
  weights and rep_optim moments round-trip. If the checkpoint was saved
  by a non-rep_loss run, rep_optim initializes fresh (no mismatch crash).

```python
class TransformerRepLossTrainModule(TransformerTrainModule):
    def __init__(self, model, teacher: TeacherModelConfig,
                 rep_loss: RepLossConfig, **kwargs):
        for k in ("tp_config", "cp_config", "pp_config"):
            if kwargs.get(k) is not None:
                raise OLMoConfigurationError(
                    f"TransformerRepLossTrainModule does not support {k} in v1"
                )
        super().__init__(model=model, **kwargs)
        self.rep_loss_config = rep_loss

        # nGPT rejects: last-block output is hypersphere-normalized
        # (NormalizedTransformerBlock.forward applies l2_normalize at the
        # output; confirmed at nn/transformer/block.py:325-341). Raw L2 to
        # a teacher embedding is geometrically ill-posed.
        if isinstance(self.model, NormalizedTransformer):
            raise OLMoConfigurationError(
                "RepLossTrainModule does not support NormalizedTransformer "
                "(nGPT) students."
            )

        # EMA would silently exclude rep_head (base EMA iterates
        # self.model.named_parameters() only). Reject rather than ship a
        # half-EMAed model.
        if self.ema_config is not None:
            raise OLMoConfigurationError(
                "Weight EMA is not supported with TransformerRepLossTrainModule "
                "(EMA shadows do not cover rep_head)."
            )

        if self.z_loss_multiplier is not None and rep_loss.ce_loss_weight != 1.0:
            raise OLMoConfigurationError(
                "ce_loss_weight != 1.0 is not supported with z_loss enabled."
            )

        self.teacher = teacher.build(
            student_world_mesh=self.world_mesh, device=self.device,
            student_max_sequence_length=self.max_sequence_length,
            rank_microbatch_size=self.rank_microbatch_size,
        )

        if (self.teacher.trained_sequence_length is not None
            and self.teacher.trained_sequence_length < self.max_sequence_length):
            raise OLMoConfigurationError(
                f"Teacher trained at seq_len={self.teacher.trained_sequence_length}; "
                f"student uses {self.max_sequence_length}."
            )

        # Build rep_head. Construction is on CPU, then .to(device). DDP
        # broadcasts rank-0 weights to all other ranks at wrap time.
        d_teacher = self.teacher.model.d_model
        rep_core = nn.Linear(
            self.model.d_model, d_teacher, bias=True, dtype=torch.float32,
        ).to(self.device)
        if rep_loss.rep_head_layer_norm:
            rep_core = nn.Sequential(
                nn.LayerNorm(self.model.d_model, dtype=torch.float32).to(self.device),
                rep_core,
            )

        if is_distributed() and self.dp_process_group is not None:
            self.rep_head = DDP(
                rep_core, process_group=self.dp_process_group,
                broadcast_buffers=False,
            )
        else:
            self.rep_head = rep_core

        # Dedicated AdamW for rep_head. Betas/weight_decay are rep-head-specific
        # — rep_head is a single small Linear and tracks teacher hidden states,
        # so we don't want its schedule tied to student-weight-decay policies.
        rep_lr = rep_loss.rep_lr
        self.rep_optim = torch.optim.AdamW(
            self.rep_head.parameters(),
            lr=rep_lr,
            betas=(0.9, 0.95),
            weight_decay=0.0,
        )
        # Set initial_lr on the param group so the LR scheduler sees it
        # (LR schedulers in JOLMo use `group["initial_lr"]` as the base).
        for g in self.rep_optim.param_groups:
            g.setdefault("initial_lr", rep_lr)

        # Install the student hidden-state hook.
        self._student_hidden_cache: List[Optional[torch.Tensor]] = [None]
        base = self.model.module if isinstance(self.model, DDP) else self.model
        last_block = base.blocks[str(base.n_layers - 1)]
        self._rep_hook_handle = last_block.register_forward_hook(
            lambda m, inp, out: self._student_hidden_cache.__setitem__(0, out)
        )

        log.info(f"RepLossConfig: {rep_loss}")
        log.info(
            f"rep_head params: {sum(p.numel() for p in self.rep_head.parameters())} "
            f"(d_student={self.model.d_model}, d_teacher={d_teacher}, "
            f"rep_lr={rep_lr})"
        )

    def _rep_head_module(self) -> nn.Module:
        """Unwraps DDP if wrapped."""
        return self.rep_head.module if isinstance(self.rep_head, DDP) else self.rep_head
```

### 3.2 Config

```python
@dataclass
class RepLossConfig(Config):
    rep_loss_weight: float = 1.0
    """Weight on the per-token L2 rep-matching loss. Default 1.0 is a
    conservative starting point; __init__ logs the expected initial
    magnitude so users can tune. Typical step-0 rep_loss with
    normalize=False is O(d_teacher × Var(teacher)) ~ 1e3; divide by
    rank_world_size and by live-token count to get the reported metric.
    Tune from wandb."""

    ce_loss_weight: float = 1.0

    normalize: bool = False
    """If True, L2-normalize student projection and teacher embedding
    before the L2 (cosine-style). Default False = raw L2."""

    rep_head_layer_norm: bool = False
    """If True, insert a LayerNorm before the rep_head Linear."""

    rep_lr: float = 3.0e-4
    """Learning rate for rep_optim (the dedicated AdamW for rep_head).
    Not tied to the student optimizer's LR. 3e-4 is standard for a
    lightweight regression head."""
```

### 3.3 `_train_microbatch_context` override — rep_head no_sync

DDP's default behavior is to all-reduce gradients on every backward. With
gradient accumulation across microbatches, this inflates rep_head's
gradient by `num_microbatches` (and wastes bandwidth). Base
`_train_microbatch_context` already handles FSDP/DDP student no_sync; we
extend it for rep_head:

```python
@contextlib.contextmanager
def _train_microbatch_context(self, micro_batch_idx, num_micro_batches):
    with super()._train_microbatch_context(micro_batch_idx, num_micro_batches):
        is_last_mb = micro_batch_idx == num_micro_batches - 1
        if isinstance(self.rep_head, DDP) and not is_last_mb:
            with self.rep_head.no_sync():
                yield
        else:
            yield
```

### 3.4 `train_batch`

```python
def train_batch(self, batch, dry_run=False):
    self._set_model_mode("train")
    if "labels" not in batch:
        batch["labels"] = get_labels(batch, label_ignore_index=self.label_ignore_index)
    # ... base instance_mask / masked-labels-% recording ...

    batch_num_tokens_for_loss = move_to_device(
        (batch["labels"] != self.label_ignore_index).sum(), self.device
    )
    ce_batch_loss  = move_to_device(torch.tensor(0.0), self.device)
    rep_batch_loss = move_to_device(torch.tensor(0.0), self.device)
    z_batch_loss   = (move_to_device(torch.tensor(0.0), self.device)
                     if self.z_loss_multiplier is not None else None)

    seq_len = batch["input_ids"].shape[1]
    micro_batches = split_batch(batch, self.rank_microbatch_size // seq_len)
    num_micro_batches = len(micro_batches)

    for i, micro_batch in enumerate(micro_batches):
        with self._train_microbatch_context(i, num_micro_batches):
            input_ids, labels, model_kwargs = self._prepare_batch(micro_batch)
            teacher_kwargs = {
                k: model_kwargs[k]
                for k in ("doc_lens", "max_doc_lens", "cache_leftpad")
                if k in model_kwargs
            }

            t_emb = self.teacher.hidden_states(input_ids, **teacher_kwargs)

            try:
                output = self.model_forward(
                    input_ids, labels=labels,
                    ignore_index=self.label_ignore_index,
                    loss_reduction="sum",
                    z_loss_multiplier=self.z_loss_multiplier,
                    loss_div_factor=batch_num_tokens_for_loss,
                    return_logits=False,
                    **model_kwargs,
                )
                h = self._student_hidden_cache[0]
                self._student_hidden_cache[0] = None

                loss, ce_loss, z_loss = output.loss, output.ce_loss, output.z_loss

                live_mask = (labels != self.label_ignore_index)
                s_emb = self.rep_head(h)
                s_live = s_emb[live_mask].float()
                t_live = t_emb[live_mask].detach().float()
                if self.rep_loss_config.normalize:
                    s_live = F.normalize(s_live, dim=-1)
                    t_live = F.normalize(t_live, dim=-1)
                rep_sum = (s_live - t_live).pow(2).sum(dim=-1).sum()
                rep_contrib = rep_sum / batch_num_tokens_for_loss
                rep_batch_loss += rep_contrib.detach()

                total = (self.rep_loss_config.ce_loss_weight * loss
                         + self.rep_loss_config.rep_loss_weight * rep_contrib)
                ce_batch_loss += get_local_tensor(ce_loss.detach())
                if z_batch_loss is not None and z_loss is not None:
                    z_batch_loss += get_local_tensor(z_loss.detach())

                total.backward()
            finally:
                self._student_hidden_cache[0] = None

    self.model.post_batch(dry_run=dry_run)
    if dry_run:
        self.model.reset_auxiliary_metrics()
        return

    if isinstance(self.optim, SkipStepOptimizer):
        if is_distributed():
            ce_batch_loss.div_(self._reduce_divide_factor)
            dist.all_reduce(ce_batch_loss)
            ce_batch_loss.div_(self.world_size)
            ce_batch_loss.mul_(self._reduce_divide_factor)
        self.record_ce_loss(ce_batch_loss)
        self.optim.latest_loss = ce_batch_loss
    else:
        self.record_ce_loss(ce_batch_loss, ReduceType.mean)

    self.record_metric("rep loss", rep_batch_loss, ReduceType.mean, namespace="train")
    if z_batch_loss is not None:
        self.record_metric("Z loss", z_batch_loss, ReduceType.mean, namespace="train")

    for name, (val, reduction) in self.model.compute_auxiliary_metrics(reset=True).items():
        self.record_metric(name, val, reduction, namespace="train")
```

### 3.5 `optim_step` and `zero_grads` — both optimizers

The base's `optim_step` applies the scheduler to `self.optim.param_groups`
and calls `self.optim.step()`. We extend to also drive `self.rep_optim`.

```python
def optim_step(self):
    # Base: grad clip (our override covers both model + rep_head), schedule
    # self.optim groups, step self.optim, post_optim_step on model.
    super().optim_step()
    # Also schedule + step rep_optim.
    if self.scheduler is not None:
        for group in self.rep_optim.param_groups:
            new_lr = self.scheduler.set_lr(group, self.trainer)
            self.trainer.record_metric("LR (rep_head)", new_lr, namespace="optim")
    self.rep_optim.step()
    # Note: rep_optim does not participate in SkipStep semantics. A
    # SkipStepOptimizer on self.optim may skip the student update; rep_optim
    # still steps. For rep_head (a small regression head), this is benign.

def zero_grads(self):
    super().zero_grads()
    self.rep_optim.zero_grad(set_to_none=True)
```

### 3.6 Gradient clipping

The base `_clip_grad_norm` clips `self.model.parameters()` only. We clip
rep_head separately to the same `max_grad_norm` and log its norm as a
distinct metric:

```python
def _clip_grad_norm(self, max_grad_norm, norm_type=2.0, foreach=None):
    student_norm = super()._clip_grad_norm(max_grad_norm, norm_type=norm_type,
                                           foreach=foreach)
    rep_params = [p for p in self.rep_head.parameters() if p.grad is not None]
    if rep_params:
        rep_norm = torch.nn.utils.clip_grad_norm_(
            rep_params, max_grad_norm, norm_type=norm_type,
            error_if_nonfinite=False, foreach=foreach,
        )
        self.trainer.record_metric(
            "rep_head grad norm", rep_norm, reduce_type=None, namespace="optim",
        )
    return student_norm
```

### 3.7 Checkpointing — rep_head weights + rep_optim state

Because `rep_head` is outside `self.model` and `rep_optim` is outside
`self.optim`, neither is captured by the base's `_get_state_dict`. We
round-trip both explicitly via a side-channel, using `dist_cp_sd` so the
machinery matches the rest of the checkpoint:

```python
def _get_state_dict(self, sd_options, optim=True):
    sd = super()._get_state_dict(sd_options, optim=optim)
    rep_mod = self._rep_head_module()
    sd["rep_head"] = dist_cp_sd.get_model_state_dict(rep_mod, options=sd_options)
    if optim:
        sd["rep_optim"] = dist_cp_sd.get_optimizer_state_dict(
            rep_mod, self.rep_optim, options=sd_options,
        )
    return sd

def state_dict_to_load(self, metadata, *, optim=None):
    sd = super().state_dict_to_load(metadata, optim=optim)
    rep_mod = self._rep_head_module()
    load_opts = self.state_dict_load_opts
    if any(k.startswith("rep_head.") for k in metadata.state_dict_metadata):
        sd["rep_head"] = dist_cp_sd.get_model_state_dict(rep_mod, options=load_opts)
    if optim is not False and any(
        k.startswith("rep_optim.") for k in metadata.state_dict_metadata
    ):
        sd["rep_optim"] = dist_cp_sd.get_optimizer_state_dict(
            rep_mod, self.rep_optim, options=load_opts,
        )
    return sd

def load_state_dict(self, state_dict):
    rep_head_sd = state_dict.pop("rep_head", None)
    rep_optim_sd = state_dict.pop("rep_optim", None)
    super().load_state_dict(state_dict)
    rep_mod = self._rep_head_module()
    if rep_head_sd is not None:
        dist_cp_sd.set_model_state_dict(
            rep_mod, rep_head_sd, options=self.state_dict_load_opts,
        )
    if rep_optim_sd is not None:
        dist_cp_sd.set_optimizer_state_dict(
            rep_mod, self.rep_optim, rep_optim_sd,
            options=self.state_dict_load_opts,
        )
```

`dist_cp_sd.get_optimizer_state_dict(rep_mod, self.rep_optim)` succeeds
because all of `rep_optim`'s parameters ARE in `rep_mod.named_parameters()`
(rep_optim was constructed from exactly those params). This is the
specific bug that broke earlier revisions' attempt to put rep_head in
`self.optim`.

**First-resume from a non-rep_loss checkpoint**: the checkpoint has no
`rep_head.*` or `rep_optim.*` keys. `state_dict_to_load` skips both
templates; `load_state_dict` pops None for both; `super().load_state_dict`
loads the student weights and student optim cleanly. rep_head stays at
its DDP-broadcast init; rep_optim's state stays empty. Training proceeds
with a fresh rep_head. This is the supported path for "warm-start a
rep_loss run from a pretrained student checkpoint" — use
`JolmoModel.base_model` with `reload_optimizer=True` if desired; the
student's optim state round-trips, and rep_optim starts fresh.

**Unshard / HF export**: `unshard.py` filters by `"model."` / `"optim."`
prefixes — top-level `rep_head.*` and `rep_optim.*` keys are silently
dropped. Intended: HF export is student-only.

### 3.8 Pitfalls

- **Last-block output is pre-`lm_head.norm`** on both teacher and student
  — symmetric by construction. Note this is NOT the teacher's "canonical"
  embedding space (post-norm); rep_head is regressing to the
  pre-final-norm residual stream. Acceptable for an auxiliary signal.
- **Memory delta vs. base**: teacher forward activations + `B_µ * S *
  d_teacher * 2B` cached hidden states. No V factor.
- **Autocast dtype**: student h and teacher t_emb are bf16; we index
  then `.float()` on the live-token slice only.
- **rep_loss magnitude at step 0**: `__init__` logs the expected order of
  magnitude. A 50-100× imbalance with CE is a signal to retune
  `rep_loss_weight`.
- **No SkipStep on rep_optim**: rep_head steps even when the student's
  SkipStep skips. For a small regression head this is fine.

---

## 4. Wiring into the launch path

### 4.1 `YamlExperimentConfig` + builders

In [`launch_from_yaml.py`](../src/scripts/launch_from_yaml.py):

```python
@dataclass
class YamlExperimentConfig(Config):
    # ... existing fields ...
    teacher: Optional[TeacherModelConfig] = None
    distill: Optional[DistillConfig] = None
    rep_loss: Optional[RepLossConfig] = None


def _common_train_module_kwargs(cfg: YamlExperimentConfig) -> Dict[str, Any]:
    """Shared autocast / state-dict-opts conversion — extracted from
    _build_train_module_sam for reuse."""
    kwargs = cfg.train_module.as_dict(exclude_none=True, recurse=False)
    if (prec := kwargs.pop("autocast_precision", None)) is not None:
        kwargs["autocast_precision"] = cast(DType, prec).as_pt()
    if (save_opts := kwargs.pop("state_dict_save_opts", None)) is not None:
        kwargs["state_dict_save_opts"] = dist_cp_sd.StateDictOptions(**save_opts)
    if (load_opts := kwargs.pop("state_dict_load_opts", None)) is not None:
        kwargs["state_dict_load_opts"] = dist_cp_sd.StateDictOptions(**load_opts)
    return kwargs


def _build_train_module_distill(cfg, model):
    if cfg.teacher is None or cfg.distill is None:
        raise OLMoConfigurationError(
            "train_module_type=distill requires 'teacher' and 'distill' sections"
        )
    return TransformerDistillTrainModule(
        model=model, teacher=cfg.teacher, distill=cfg.distill,
        **_common_train_module_kwargs(cfg),
    )


def _build_train_module_rep_loss(cfg, model):
    if cfg.teacher is None or cfg.rep_loss is None:
        raise OLMoConfigurationError(
            "train_module_type=rep_loss requires 'teacher' and 'rep_loss' sections"
        )
    return TransformerRepLossTrainModule(
        model=model, teacher=cfg.teacher, rep_loss=cfg.rep_loss,
        **_common_train_module_kwargs(cfg),
    )


# In main():
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

Refactor `_build_train_module_sam` to use `_common_train_module_kwargs`.

### 4.2 Exports

In `JOLMo/src/olmo_core/train/train_module/__init__.py`: export
`TransformerDistillTrainModule`, `TransformerRepLossTrainModule`,
`DistillConfig`, `RepLossConfig`, `TeacherModel`, `TeacherModelConfig`.

### 4.3 `JolmoModel` in `mixture-pretraining`

Relevant existing code at
`mixture_pretraining_stages/training.py`:
- `JolmoModel` dataclass (line 356).
- `MODEL_ARCHS` dict (line 31).
- `_build_model_spec(model_type, vocab_size)` (line 93).
- `_tokenizer_config(tokenizer_id)` (line 47).
- `_build_tokenizer_yaml(tokenizer_id)` (line 59).
- `trainer_type: Literal["standard", "sam"]` (line 411).

Extend:

```python
@dataclass(frozen=True)
class JolmoModel(Artifact):
    # ... existing fields ...
    trainer_type: Literal["standard", "sam", "distill", "rep_loss"] = "standard"

    # --- Teacher-based training (distill / rep_loss only) ---
    teacher_model_type: Optional[str] = None
    teacher_checkpoint_path: Optional[str] = None
    teacher_tokenizer: Optional[str] = None
    teacher_dp_fsdp: bool = True
    teacher_autocast_precision: str = "bfloat16"
    teacher_trained_sequence_length: Optional[int] = None
    """Sequence length the teacher was trained at. Required to reject
    configurations where the student seq len exceeds the teacher's
    (RoPE extrapolation produces meaningless teacher signal)."""

    # --- Distill-only fields ---
    forward_kl_weight: float = 1.0
    reverse_kl_weight: float = 0.0
    reverse_kl_warmup_steps: Optional[int] = None
    allow_reverse_kl_from_scratch: bool = False
    kl_temperature: float = 1.0
    distill_ce_loss_weight: float = 1.0

    # --- RepLoss-only fields ---
    rep_loss_weight: float = 1.0
    rep_ce_loss_weight: float = 1.0
    rep_normalize: bool = False
    rep_head_layer_norm: bool = False
    rep_lr: float = 3.0e-4
```

Validation in `__post_init__`:

```python
if self.trainer_type in ("distill", "rep_loss"):
    if not self.teacher_model_type:
        raise ValueError(f"trainer_type='{self.trainer_type}' requires teacher_model_type")
    if self.teacher_model_type not in MODEL_ARCHS:
        raise ValueError(f"teacher_model_type='{self.teacher_model_type}' not in MODEL_ARCHS")
    if not self.teacher_checkpoint_path:
        raise ValueError(f"trainer_type='{self.trainer_type}' requires teacher_checkpoint_path")
    # Tokenizer identity: distill feeds teacher logits through the KL (needs
    # exact tokenizer match). rep_loss feeds input_ids — if the two
    # tokenizers differ, id k indexes unrelated teacher embeddings, which
    # is noise. Same check applies to both modes.
    if self.teacher_tokenizer is not None and self.teacher_tokenizer != self.tokenizer:
        raise ValueError(
            f"trainer_type='{self.trainer_type}' requires the teacher and "
            f"student to use the same tokenizer. Got student={self.tokenizer!r}, "
            f"teacher={self.teacher_tokenizer!r}. Leave teacher_tokenizer=None "
            f"to inherit from student."
        )

# rep_loss does not currently support Weight EMA.
if self.trainer_type == "rep_loss" and self.ema_decays:
    raise ValueError(
        "ema_decays is not currently supported with trainer_type='rep_loss'"
    )
```

In `_build_yaml_config`, after the existing train_module block:

```python
if self.trainer_type in ("distill", "rep_loss"):
    teacher_tokenizer = self.teacher_tokenizer or self.tokenizer
    teacher_vocab = _tokenizer_config(teacher_tokenizer).padded_vocab_size()
    cfg["teacher"] = {
        "_CLASS_": "olmo_core.train.train_module.transformer.teacher.TeacherModelConfig",
        "model": _build_model_spec(self.teacher_model_type, teacher_vocab),
        "checkpoint_path": self.teacher_checkpoint_path,
        "autocast_precision": self.teacher_autocast_precision,
        "trained_sequence_length": self.teacher_trained_sequence_length,
        "dp_config": (
            {
                "_CLASS_": "olmo_core.train.train_module.transformer.config.TransformerDataParallelConfig",
                "name": "fsdp",
                "param_dtype": self.dp_param_dtype,
                "reduce_dtype": self.dp_reduce_dtype,
            } if self.teacher_dp_fsdp else None
        ),
    }

if self.trainer_type == "distill":
    cfg["train_module_type"] = "distill"
    cfg["distill"] = {
        "_CLASS_": "olmo_core.train.train_module.transformer.distill_train_module.DistillConfig",
        "forward_kl_weight": self.forward_kl_weight,
        "reverse_kl_weight": self.reverse_kl_weight,
        "reverse_kl_warmup_steps": self.reverse_kl_warmup_steps,
        "temperature": self.kl_temperature,
        "ce_loss_weight": self.distill_ce_loss_weight,
        "allow_reverse_kl_from_scratch": self.allow_reverse_kl_from_scratch,
    }
    # Distill requires the non-fused LM loss path.
    cfg["model"]["lm_head"]["loss_implementation"] = "default"
elif self.trainer_type == "rep_loss":
    cfg["train_module_type"] = "rep_loss"
    cfg["rep_loss"] = {
        "_CLASS_": "olmo_core.train.train_module.transformer.rep_loss_train_module.RepLossConfig",
        "rep_loss_weight": self.rep_loss_weight,
        "ce_loss_weight": self.rep_ce_loss_weight,
        "normalize": self.rep_normalize,
        "rep_head_layer_norm": self.rep_head_layer_norm,
        "rep_lr": self.rep_lr,
    }
```

**Teacher checkpoint staging**: `construct()` already stages training data
from GCS via `_download_chunk_dirs`. The teacher checkpoint needs the
same treatment — add an explicit
`builder.download_from_gs(teacher_gs_dir, teacher_local_dir, directory=True)`
call (single-node; all processes on the node share the staged path) and
substitute the local path into `cfg["teacher"]["checkpoint_path"]`. For
multi-node expansion, switch to a rank-0-per-node stage + barrier.

---

## 5. Example YAML

```yaml
train_module_type: distill
teacher:
  _CLASS_: olmo_core.train.train_module.transformer.teacher.TeacherModelConfig
  model:
    _CLASS_: olmo_core.nn.transformer.config.TransformerConfig
    # ... full config from _build_model_spec(teacher_model_type, vocab_size) ...
  checkpoint_path: /local/path/to/teacher/checkpoint
  autocast_precision: bfloat16
  trained_sequence_length: 2048
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

rep_loss is identical in shape; replace `distill:` with `rep_loss:`.

---

## 6. File structure

```
JOLMo/src/olmo_core/train/train_module/transformer/
  teacher.py                     (new)
  distill_train_module.py        (new) DistillConfig, kl_per_token, TransformerDistillTrainModule
  rep_loss_train_module.py       (new) RepLossConfig, TransformerRepLossTrainModule

JOLMo/src/olmo_core/train/train_module/__init__.py
  + exports

JOLMo/src/scripts/launch_from_yaml.py
  + YamlExperimentConfig fields (teacher, distill, rep_loss)
  + _common_train_module_kwargs helper (refactors SAM builder)
  + _build_train_module_distill / _build_train_module_rep_loss
  + main() routing

mixture_pretraining_stages/training.py
  + extend JolmoModel.trainer_type literal
  + teacher_* / distill-only / rep_loss-only fields
  + __post_init__ validation (teacher_model_type, tokenizer identity, EMA reject)
  + _build_yaml_config branches
  + teacher-checkpoint staging in construct()
```

---

## 7. Implementation checklist

1. [ ] `teacher.py`:
   - `TeacherModelConfig` with `trained_sequence_length` field (NOT
     `max_sequence_length`).
   - `TeacherModel` stores `trained_sequence_length` as an attribute
     distinct from the student-seq-len kwarg passed to build().
   - `build()`: meta → `parallelize_model(..., max_sequence_length=student_seq,
     rank_microbatch_size=..., compile_model=self.compile, dp_config=...)`
     → `load_model_and_optim_state(path, model, optim=None)` → `gc_cuda()`
     → `eval()` → freeze → requires_grad=False assert → mesh identity assert.
   - `logits(input_ids, **kwargs)` and `hidden_states(input_ids, **kwargs)`
     wrap forward in `torch.no_grad()` + `torch.autocast(...)`, forwarding
     `doc_lens`/`max_doc_lens`/`cache_leftpad`.
   - Forward hook on `model.blocks[str(n_layers - 1)]`.
2. [ ] `distill_train_module.py`:
   - `DistillConfig` with `reverse_kl_warmup_steps`.
   - `kl_per_token` using `F.log_softmax(..., dtype=torch.float32)`,
     `F.kl_div(log_target=True)`, `T**2` scaling both directions.
   - `__init__`: TP/CP/PP reject, vocab match, `NormalizedLMHead` reject,
     `loss_implementation=default`, teacher-trained-seq-len assert,
     reverse-KL guard (unless warmup or opt-out), z_loss × `ce_loss_weight
     != 1.0` reject, memory note (~10× logit_gib, recommend B_µ=4 default).
   - `train_batch`: teacher-first with intra-doc kwargs forwarded, reverse-KL
     warmup ramp using `trainer.global_step`, full-tensor KL + post-softmax
     mask, dry_run early return, SkipStep early all-reduce, preserved base
     metrics.
3. [ ] `rep_loss_train_module.py`:
   - `RepLossConfig` with `rep_lr` field (default 3e-4), `rep_loss_weight`
     default 1.0.
   - `__init__`: TP/CP/PP reject, nGPT reject, EMA reject, z_loss ×
     `ce_loss_weight != 1.0` reject, teacher-trained-seq-len assert.
   - `rep_head = nn.Linear(d_s, d_t, bias=True)` optionally prefixed by
     LayerNorm, DDP-wrapped in distributed mode.
   - **Dedicated `self.rep_optim = torch.optim.AdamW(rep_head.parameters(),
     lr=rep_lr, betas=(0.9, 0.95), weight_decay=0.0)`**; set `initial_lr`
     on its param group.
   - `_rep_head_module()` property (unwraps DDP).
   - Forward hook on student's last block; cache cleared in try/finally.
   - `_train_microbatch_context` override: `rep_head.no_sync()` for
     non-last microbatches.
   - `train_batch`: teacher-first with intra-doc kwargs forwarded,
     index-then-cast on live tokens, preserved base metrics, dry_run,
     SkipStep.
   - `optim_step` override: call `super().optim_step()`, then schedule +
     step `self.rep_optim`, record `LR (rep_head)` metric.
   - `zero_grads` override: call `super().zero_grads()`, then
     `self.rep_optim.zero_grad(set_to_none=True)`.
   - `_clip_grad_norm` override: student via super(), rep_head separately.
   - `_get_state_dict` / `state_dict_to_load` / `load_state_dict`: save
     and restore both `sd["rep_head"]` (via `get_model_state_dict(rep_mod)`)
     and `sd["rep_optim"]` (via
     `get_optimizer_state_dict(rep_mod, self.rep_optim)`).
4. [ ] Exports in `train_module/__init__.py`.
5. [ ] `launch_from_yaml.py`: `_common_train_module_kwargs`, two new
   builders, main() routing, new `YamlExperimentConfig` fields.
6. [ ] `mixture_pretraining_stages/training.py`:
   - Extend `trainer_type` literal.
   - Add `teacher_*`, distill, rep_loss fields.
   - `__post_init__`: teacher fields required when `trainer_type` in
     `("distill", "rep_loss")`; tokenizer identity for both modes;
     `ema_decays` rejected for `rep_loss`.
   - `_build_yaml_config` branches.
   - Teacher-checkpoint staging via `builder.download_from_gs(...,
     directory=True)` in `construct()`.
7. [ ] Tests:
   - `--dry-run` prints config and exits for both new `train_module_type`s.
   - Single-GPU 160M × 160M smoke for ~10 steps.
   - Multi-GPU FSDP student + FSDP teacher + DDP rep_head. Assert step-0
     identical rep_head state across ranks.
   - Resume test for rep_loss: rep_head WEIGHTS and rep_optim MOMENTS both
     round-trip exactly. Verify identical gradients after a single step
     on resumed vs. continuous runs.
   - First-resume from a non-rep_loss checkpoint: student state loads,
     rep_head starts at broadcast init, rep_optim starts fresh, training
     proceeds.
   - Intra-document-masking: with packed data + doc_lens in the batch,
     verify the teacher forward sees them (e.g., by hooking the teacher's
     attention module).
   - Reverse-KL warmup: verify the `KL reverse effective weight` metric
     ramps 0 → configured_weight over the configured steps.
   - Muon + rep_loss: smoke-test that rep_head parameters actually update
     (their values change after a step).
   - dry_run=True records no metrics.

---

## 8. Explicitly out of scope for v1

- HuggingFace teachers.
- GradCache / similarity-matrix objectives.
- Cross-tokenizer / cross-vocab teachers.
- TP / CP / PP for the student.
- Offline precomputed teacher logits.
- Per-layer or multi-layer rep matching.
- Combined distill + rep-loss.
- Eval-time KL or rep-loss logging.
- Persisting `DistillConfig` / `RepLossConfig` / teacher-checkpoint hash
  for drift detection on resume.
- Learnable temperature.
- Chunked or fused-linear-KL kernels.
- `unshard.py` / HF-export handling of rep_head (intentionally dropped).
- Weight EMA for `TransformerRepLossTrainModule`.
- Running the teacher at a longer context window than it was trained at
  (rejected).
- SkipStep semantics for rep_optim (rep_optim steps unconditionally).
- Multi-node teacher-checkpoint staging with rank-0-per-node
  broadcast (single-node stage is assumed).
