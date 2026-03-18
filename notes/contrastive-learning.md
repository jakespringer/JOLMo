# Distillation Pre-training: Implementation Plan

This document describes the design and implementation plan for adding two distillation
methods to JOLMo pre-training: **logit distillation** and **contrastive (similarity-based)
distillation**, plus a **combined objective** that uses both simultaneously.

---

## 1. Overview & Goals

We want to add three training modes to the JOLMo pre-training framework:

1. **Logit distillation**: Train a student LM to match the next-token probability distribution
   of a teacher model, using KL divergence on logits.
2. **Contrastive distillation**: Train a student LM so that its token-level pairwise similarity
   structure matches that of a contrastively-trained teacher model, using cross-entropy on
   similarity matrices.
3. **Combined objective**: Both logit and contrastive distillation simultaneously, with a
   tunable weight `lambda` controlling the contrastive contribution.

All modes also maintain the standard cross-entropy (CE) language modeling loss against
ground-truth labels. The final loss is:

```
L = L_CE + alpha * L_logit_distill + lambda * L_contrastive_distill
```

where `alpha` and `lambda` are configurable weights (either can be 0 to disable that term).

### Design Principles

- **Minimal Trainer changes**: The `Trainer` class does not need modification. All distillation
  logic lives in a new `TrainModule` subclass, following the pattern established by
  `TransformerSAMTrainModule`.
- **Flexible teacher abstraction**: Teachers can be JOLMo `Transformer` models, HuggingFace
  models, or any `nn.Module` with a defined interface. Teachers are configured in a separate
  YAML section with their own dtype, checkpoint path, and distribution strategy.
- **Separate batch sizes**: The teacher can have its own microbatch size (for inference
  chunking), and the contrastive loss can operate on a configurable subset of the batch.
- **GradCache for contrastive loss**: The pairwise similarity matrix over all tokens requires
  all representations simultaneously. GradCache decouples the loss-level backward from the
  encoder backward, allowing sub-batch gradient accumulation without approximation.

---

## 2. Architecture

### 2.1 High-Level Design

```
                    ┌──────────────────────────────┐
                    │    YamlExperimentConfig       │
                    │  train_module_type: "distill" │
                    │  distillation: {...}          │
                    │  logit_teacher: {...}         │
                    │  contrastive_teacher: {...}   │
                    └──────────┬───────────────────┘
                               │
                               ▼
                    ┌──────────────────────────────┐
                    │ _build_train_module_distill() │
                    │  in launch_from_yaml.py       │
                    └──────────┬───────────────────┘
                               │
            ┌──────────────────┼──────────────────────┐
            ▼                  ▼                       ▼
   ┌─────────────┐  ┌──────────────────┐  ┌────────────────────┐
   │ Student      │  │ LogitTeacher     │  │ ContrastiveTeacher │
   │ (Transformer │  │ (frozen model,   │  │ (frozen model,     │
   │  + optimizer │  │  produces logits)│  │  produces per-token│
   │  + scheduler)│  │                  │  │  embeddings)       │
   └──────┬──────┘  └──────────────────┘  └────────────────────┘
          │
          ▼
   ┌─────────────────────────────┐
   │ DistillationTrainModule     │
   │  extends TransformerTrain-  │
   │  Module, overrides          │
   │  train_batch()              │
   └─────────────────────────────┘
```

### 2.2 Teacher Model Abstraction

The teacher model abstraction needs to support:
- JOLMo `Transformer` models (loaded from checkpoint)
- HuggingFace models (loaded via `AutoModelForCausalLM` or `AutoModel`)
- Any `nn.Module` that conforms to a simple interface
- Optional distribution over multiple GPUs (FSDP, DDP)
- Separate microbatch size for inference chunking

#### TeacherModel Wrapper

```python
class TeacherModel:
    """
    Wraps a frozen model for use as a distillation teacher.
    Provides a unified interface for logit and embedding extraction.
    """
    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def get_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Returns (batch_size, seq_len, vocab_size) logits."""
        ...

    @torch.no_grad()
    def get_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Returns (batch_size, seq_len, d_embedding) per-token embeddings."""
        ...
```

#### TeacherModelConfig

```python
@dataclass
class TeacherModelConfig(Config):
    """Base configuration for a teacher model."""

    # ---- Model specification (choose one) ----
    # Option A: JOLMo Transformer
    model_config: Optional[TransformerConfig] = None

    # Option B: HuggingFace model
    hf_model_name: Optional[str] = None

    # ---- Checkpoint ----
    checkpoint_path: Optional[str] = None

    # ---- Inference settings ----
    dtype: DType = DType.bfloat16
    compile: bool = False

    # ---- Distribution strategy ----
    # If the teacher is large, it may need its own FSDP/DDP config.
    # If None, the teacher is replicated on each rank (suitable for
    # small teachers that fit on a single GPU).
    dp_config: Optional[TransformerDataParallelConfig] = None

    # ---- Batching ----
    # Teacher's own microbatch size (in tokens). If None, uses the
    # student's rank_microbatch_size. Useful when the teacher is much
    # larger and needs smaller forward-pass chunks.
    rank_microbatch_size: Optional[int] = None

    def build(self, device: torch.device) -> TeacherModel:
        """Build, load, freeze, and optionally distribute the teacher model."""
        if self.model_config is not None:
            model = self.model_config.build(init_device="meta")
            # Load checkpoint...
            # Optionally parallelize with dp_config...
        elif self.hf_model_name is not None:
            from transformers import AutoModelForCausalLM, AutoModel
            model = AutoModelForCausalLM.from_pretrained(
                self.hf_model_name,
                torch_dtype=self.dtype.as_pt(),
            )
            # Optionally wrap with FSDP...
        else:
            raise ValueError("Must provide model_config or hf_model_name")

        return TeacherModel(model, device)
```

**Key design choice**: The teacher model is loaded, frozen, and potentially FSDP-wrapped
*separately* from the student model. Both reside on the same set of GPUs. FSDP shards each
model independently, so the per-GPU memory is `(student_shard + teacher_shard + optimizer_state)`.

**Why same GPUs (not separate GPU groups)**: Separate GPU groups would require a custom
`ProcessGroup` setup and cross-group communication for logits/embeddings. This adds significant
complexity. Since the teacher is frozen (no optimizer state), its memory footprint is just the
model weights, which is typically manageable. For a bf16 teacher:
- 1B params = ~2 GB per GPU (with 8-way FSDP sharding)
- 7B params = ~14 GB per GPU (with 8-way FSDP sharding) → ~1.75 GB per GPU
- 70B params = ~140 GB → ~17.5 GB per GPU (8-way)

For very large teachers (70B+), separate GPU groups or offline precomputation of logits is
recommended. This can be added as a future enhancement.

### 2.3 DistillationTrainModule

The core implementation is a new `DistillationTrainModule` that extends `TransformerTrainModule`.

**Why extend TransformerTrainModule (not standalone like SAM)?**

The SAM train module duplicates most of `TransformerTrainModule`'s code. This is because SAM
needs fundamentally different gradient management (ascent/descent with perturbation). For
distillation, the changes are more localized:
- The `train_batch()` method needs additional loss terms
- Everything else (parallelization, optimizer, scheduler, checkpointing) is inherited unchanged

By extending `TransformerTrainModule`, we avoid ~400 lines of duplicated code and automatically
benefit from any future improvements to the base class.

**However**, there is one important requirement: we need access to hidden states (the
representation before the LM head) for contrastive distillation. This requires either:

1. **Forward hook on the last transformer block** (preferred — no model changes needed)
2. **Adding an `encode()` method to Transformer** (cleaner API but requires model change)
3. **Adding `return_hidden_states` flag to `Transformer.forward()`** (also requires model change)

**Recommendation: Use approach (1) for the initial implementation, with approach (2) or (3)
available as a future cleanup.**

Hook-based hidden state extraction:

```python
class DistillationTrainModule(TransformerTrainModule):
    def __init__(self, model, ..., contrastive_teacher=None, ...):
        super().__init__(model, ...)

        # Register hook to capture hidden states before LM head
        if contrastive_teacher is not None:
            self._hidden_state_cache = [None]
            base_model = model.module if isinstance(model, DDP) else model
            # After parallelization, get the last block
            last_block_name = str(base_model.n_layers - 1)
            last_block = base_model.blocks[last_block_name]

            def capture_hidden_state(module, input, output):
                self._hidden_state_cache[0] = output

            self._hidden_hook = last_block.register_forward_hook(capture_hidden_state)
```

After any call to `self.model(input_ids, ...)`, `self._hidden_state_cache[0]` contains the
hidden state tensor of shape `(batch_size, seq_len, d_model)`.

**Important**: The hook fires regardless of `torch.no_grad()`. In no-grad mode, the captured
tensor has no computation graph (which is what we want for GradCache step 1). In grad mode,
the captured tensor IS part of the computation graph (needed for the surrogate loss in
GradCache step 4).

**FSDP compatibility**: With FSDP2 (FSDPModule), submodules are individually FSDP-wrapped.
The hook is registered on the inner module, and FSDP's parameter management still works
correctly because the hook fires during the module's `__call__`.

**torch.compile compatibility**: Forward hooks are supported by `torch.compile` as of
PyTorch 2.1+. The hook simply captures a reference to the output tensor without modifying the
computation, so it should not interfere with compiled execution.

**Alternative: `encode()` method (future cleanup)**:

If we later want a cleaner API, we can add to `Transformer`:

```python
class Transformer(nn.Module):
    def encode(self, input_ids, **kwargs) -> torch.Tensor:
        """Forward through embeddings and blocks; return hidden states before LM head."""
        h = self.embeddings(input_ids)
        for block in self.blocks.values():
            h = block(h, **kwargs)
        return h

    def forward(self, input_ids, *, labels=None, ...):
        h = self.encode(input_ids, ...)
        return self.lm_head(h, labels=labels, ...)
```

The problem: with FSDP wrapping, `encode()` is a custom method (not `forward()`), so FSDP's
parameter-gathering hooks might not fire. However, with FSDP2's per-submodule wrapping, each
call to `self.embeddings(...)` and `block(...)` individually triggers FSDP, so this should
work. Needs testing.

### 2.4 Configuration System

New config classes:

```python
@dataclass
class LogitDistillationConfig(Config):
    """Configuration for logit-based distillation."""
    weight: float = 1.0              # alpha in the loss equation
    temperature: float = 1.0         # Temperature for softening logits
    kl_direction: str = "forward"    # "forward" or "reverse"
    # "forward": KL(teacher || student) — mean-seeking, simpler
    # "reverse": KL(student || teacher) — mode-seeking


@dataclass
class ContrastiveDistillationConfig(Config):
    """Configuration for contrastive (similarity-based) distillation."""
    weight: float = 1.0              # lambda in the loss equation
    temperature: float = 0.1         # Temperature for similarity softmax
    kl_direction: str = "forward"    # "forward" or "reverse"

    # Batch size for contrastive loss (in tokens). Must divide the
    # global_batch_size. Determines how many tokens participate in the
    # similarity matrix. E.g., if global_batch_size=524288 and
    # contrastive_batch_size=65536, the similarity matrix is computed
    # over 65536/seq_len = 32 examples (with seq_len=2048).
    contrastive_batch_size: int

    # GradCache sub-batch size for the no-grad encoder forward.
    # Controls memory during GradCache step 1. Smaller = less memory.
    # Must divide contrastive_batch_size.
    gradcache_chunk_size: Optional[int] = None  # Defaults to rank_microbatch_size

    # Student layer to extract hidden states from.
    # -1 = last hidden state (before LM head). Default and recommended.
    student_layer: int = -1

    # Projection head: project student hidden states to teacher embedding
    # dimension. Required if student d_model != teacher d_embedding.
    # If None and dimensions match, no projection is used.
    projection_dim: Optional[int] = None

    # Whether to mask the self-similarity diagonal in the similarity
    # matrix. Recommended: True (self-similarity is trivially 1.0 and
    # provides no learning signal).
    mask_diagonal: bool = True


@dataclass
class DistillationConfig(Config):
    """Top-level distillation configuration."""
    logit: Optional[LogitDistillationConfig] = None
    contrastive: Optional[ContrastiveDistillationConfig] = None
    logit_teacher: Optional[TeacherModelConfig] = None
    contrastive_teacher: Optional[TeacherModelConfig] = None
```

### 2.5 Integration Points

#### launch_from_yaml.py

Add a new `train_module_type: "distill"` and corresponding builder:

```python
# In YamlExperimentConfig:
distillation: Optional[DistillationConfig] = None
logit_teacher: Optional[TeacherModelConfig] = None
contrastive_teacher: Optional[TeacherModelConfig] = None

# New builder function:
def _build_train_module_distill(cfg, model):
    kwargs = cfg.train_module.as_dict(exclude_none=True, recurse=False)
    # ... handle autocast_precision, state_dict_opts (same as normal) ...

    # Build teacher models
    logit_teacher = None
    if cfg.logit_teacher is not None:
        logit_teacher = cfg.logit_teacher.build(device)

    contrastive_teacher = None
    if cfg.contrastive_teacher is not None:
        contrastive_teacher = cfg.contrastive_teacher.build(device)

    return DistillationTrainModule(
        model=model,
        logit_teacher=logit_teacher,
        contrastive_teacher=contrastive_teacher,
        distillation_config=cfg.distillation,
        **kwargs,
    )

# In main():
if cfg.train_module_type.lower() == "distill":
    train_module = _build_train_module_distill(cfg, model)
```

#### PretrainedModel artifact (mixture-pretraining)

Create a `DistillationPretrainedModel` subclass:

```python
@dataclass(frozen=True)
class DistillationPretrainedModel(PretrainedModel):
    # Logit distillation
    logit_teacher_path: Optional[str] = None
    logit_teacher_model_type: Optional[str] = None  # "610M", "1B", etc.
    logit_teacher_hf_name: Optional[str] = None
    logit_distill_weight: float = 1.0
    logit_distill_temperature: float = 1.0
    logit_distill_kl_direction: str = "forward"

    # Contrastive distillation
    contrastive_teacher_path: Optional[str] = None
    contrastive_teacher_hf_name: Optional[str] = None
    contrastive_distill_weight: float = 1.0
    contrastive_distill_temperature: float = 0.1
    contrastive_batch_size: Optional[int] = None  # in tokens

    def _build_yaml_config(self, ...):
        cfg = super()._build_yaml_config(...)
        cfg["train_module_type"] = "distill"
        cfg["distillation"] = { ... }
        if self.logit_teacher_path or self.logit_teacher_hf_name:
            cfg["logit_teacher"] = { ... }
        if self.contrastive_teacher_path or self.contrastive_teacher_hf_name:
            cfg["contrastive_teacher"] = { ... }
        return cfg
```

---

## 3. Logit Distillation: Implementation Details

### 3.1 Algorithm

For each batch, the train step for logit distillation is:

```
1. For each microbatch:
     a. Run teacher forward (no grad) → teacher_logits: (B_micro, S, V)
     b. Run student forward (with grad) → student_logits: (B_micro, S, V), ce_loss
     c. Compute KL divergence between softened distributions
     d. total_loss = ce_loss + alpha * kl_loss
     e. total_loss.backward()
2. optim_step()
3. zero_grads()
```

The teacher and student process the SAME data. The teacher's microbatch size may differ
from the student's (teacher may need smaller chunks if it's larger). In that case, the
teacher processes the microbatch in sub-chunks:

```python
# In train_batch, for each student microbatch:
student_input_ids = microbatch["input_ids"]  # (B_micro, S)

# Teacher may need smaller sub-batches
teacher_mbs = self.teacher_microbatch_size or student_input_ids.shape[0]
teacher_logits_chunks = []
with torch.no_grad():
    for chunk_start in range(0, student_input_ids.shape[0], teacher_mbs):
        chunk = student_input_ids[chunk_start:chunk_start + teacher_mbs]
        teacher_logits_chunks.append(self.logit_teacher.get_logits(chunk))
teacher_logits = torch.cat(teacher_logits_chunks, dim=0)  # (B_micro, S, V)
```

### 3.2 Loss Computation

#### Forward KL: KL(teacher || student)

The student learns to cover the teacher's distribution. This is the standard choice for
pre-training distillation (Minitron, Peng et al. 2024).

```python
def forward_kl_loss(student_logits, teacher_logits, temperature=1.0):
    """
    Forward KL: KL(teacher || student).
    Equivalent to cross-entropy(student, softmax(teacher)) up to a constant.

    student_logits: (B, S, V)  — student's raw logits
    teacher_logits: (B, S, V)  — teacher's raw logits (detached)
    """
    T = temperature
    # Soften both distributions
    teacher_probs = F.softmax(teacher_logits / T, dim=-1)       # (B, S, V)
    student_log_probs = F.log_softmax(student_logits / T, dim=-1)  # (B, S, V)

    # KL(teacher || student) = sum_v teacher * (log teacher - log student)
    # = -sum_v teacher * log student + const
    # The cross-entropy form (ignoring the constant entropy of teacher):
    loss = -(teacher_probs * student_log_probs).sum(-1)  # (B, S)

    # T^2 scaling to preserve gradient magnitudes when T > 1
    loss = loss.mean() * (T ** 2)
    return loss
```

#### Reverse KL: KL(student || teacher)

The student concentrates on the teacher's high-probability modes.

```python
def reverse_kl_loss(student_logits, teacher_logits, temperature=1.0):
    """
    Reverse KL: KL(student || teacher).
    Gradient flows through student_logits via the student_probs term.
    """
    T = temperature
    student_log_probs = F.log_softmax(student_logits / T, dim=-1)
    teacher_log_probs = F.log_softmax(teacher_logits / T, dim=-1)
    student_probs = F.softmax(student_logits / T, dim=-1)

    # KL(student || teacher) = sum_v student * (log student - log teacher)
    loss = (student_probs * (student_log_probs - teacher_log_probs)).sum(-1)  # (B, S)
    loss = loss.mean() * (T ** 2)
    return loss
```

**Note on T^2 scaling**: When temperature T > 1, the softmax outputs are flatter, and the
gradients of KL divergence are scaled down by approximately 1/T^2. Multiplying by T^2
compensates for this, keeping gradient magnitudes roughly constant across temperatures.
The standard Hinton (2015) distillation uses this scaling. For T=1, the factor is 1 (no effect).

**Practical recommendation (from literature)**: For LLM pretraining distillation, lower
temperatures (T=1 or T=2) generally work best. The next-token distribution in language
models is often already peaked, and excessive softening destroys signal. Start with T=1
(forward KL) as the default.

### 3.3 Token Masking in KL Loss

The CE loss already handles `label_ignore_index` (typically -100) to mask padding tokens.
The KL loss should use the same mask:

```python
# Mask out tokens where labels == ignore_index
label_mask = (labels != self.label_ignore_index)  # (B, S)
kl_loss_per_token = kl_loss_per_token * label_mask.float()
kl_loss = kl_loss_per_token.sum() / label_mask.sum()
```

### 3.4 Vocabulary Mismatch

If the teacher and student use different tokenizers/vocabularies, standard token-level KL
divergence is not directly applicable. For the initial implementation, **we assume the same
vocabulary**. If different vocabularies are needed later, the recommended approach is:

- **Optimal transport / Wasserstein distance**: Compares distributions in a continuous
  embedding space rather than requiring aligned vocabulary indices.
- **Universal distillation via approximate likelihood matching** (ICLR 2026): Matches
  sequence-level likelihoods rather than token-level distributions.
- **DSKD (Dual-Space Knowledge Distillation)**: Aligns through a unified output space.

These are significantly more complex and can be added as future work.

---

## 4. Contrastive Distillation: Implementation Details

### 4.1 Algorithm Overview

The contrastive distillation loss operates on pairwise token similarities:

1. Extract per-token embeddings from both teacher and student for a batch of C examples.
2. Flatten to `(C * L, D)` where L = sequence length, D = embedding dimension.
3. Compute cosine similarity matrices: `(C*L, C*L)` for both teacher and student.
4. Divide by temperature τ to get "logits."
5. Apply cross-entropy (treating teacher similarities as soft targets).

The key challenge: computing the similarity matrix requires ALL token embeddings
simultaneously (not just per-microbatch). GradCache solves this by decoupling the loss
backward from the encoder backward.

### 4.2 Similarity Matrix & Contrastive Loss

```python
def compute_similarity_matrix(embeddings, temperature):
    """
    embeddings: (N, D) where N = num_tokens across all examples
    Returns: (N, N) pairwise cosine similarity / temperature
    """
    # L2 normalize
    normed = F.normalize(embeddings, p=2, dim=-1)  # (N, D)
    # Pairwise cosine similarity
    sims = normed @ normed.T  # (N, N)
    # Scale by temperature
    sims = sims / temperature
    return sims


def contrastive_distill_loss(student_sims, teacher_sims, kl_direction="forward",
                              mask_diagonal=True):
    """
    student_sims: (N, N) — student similarity "logits"
    teacher_sims: (N, N) — teacher similarity "logits"
    """
    if mask_diagonal:
        # Self-similarity is trivially 1.0/T; mask it out
        mask = torch.eye(student_sims.size(0), device=student_sims.device).bool()
        student_sims = student_sims.masked_fill(mask, float('-inf'))
        teacher_sims = teacher_sims.masked_fill(mask, float('-inf'))

    if kl_direction == "forward":
        # Forward KL: teacher is reference
        teacher_probs = F.softmax(teacher_sims.detach(), dim=-1)      # (N, N)
        student_log_probs = F.log_softmax(student_sims, dim=-1)       # (N, N)
        loss = -(teacher_probs * student_log_probs).sum(-1).mean()    # Scalar
    elif kl_direction == "reverse":
        # Reverse KL: student is reference
        student_probs = F.softmax(student_sims, dim=-1)
        student_log_probs = F.log_softmax(student_sims, dim=-1)
        teacher_log_probs = F.log_softmax(teacher_sims.detach(), dim=-1)
        loss = (student_probs * (student_log_probs - teacher_log_probs)).sum(-1).mean()
    else:
        raise ValueError(f"Unknown kl_direction: {kl_direction}")

    return loss
```

**Key details**:

- **Cosine similarity**: We normalize embeddings to unit length first. This makes the
  similarity invariant to the magnitude of embeddings, focusing purely on directional
  agreement.
- **Temperature**: Divides the raw cosine similarities (which are in [-1, 1]) before
  softmax. Lower τ → sharper distribution (more confident about which tokens are similar).
  Typical values: 0.05-0.2. This is analogous to the temperature in standard contrastive
  learning (e.g., SimCLR uses τ=0.1).
- **Diagonal masking**: The diagonal of the cosine similarity matrix is always 1.0 (before
  temperature scaling). Including it biases the softmax heavily toward the self-similarity
  entry, drowning out the cross-token relationships we care about. Masking it (setting to
  -inf before softmax) is standard practice in contrastive learning.
- **Row-wise cross-entropy**: Each row of the similarity matrix defines a distribution over
  "which other tokens is this token similar to?" We match this distribution between teacher
  and student. This is equivalent to treating each token as a query and all other tokens as
  keys, with the teacher providing soft labels for the correct similarity structure.

### 4.3 GradCache Integration

The similarity matrix requires ALL token embeddings from the contrastive batch. Without
GradCache, this means holding the full computation graph for all examples simultaneously,
which is memory-prohibitive for large batches.

GradCache splits the backward pass into two parts:
1. **Loss → representations** (requires full batch, but only operates on small embedding
   tensors, not the full encoder graph)
2. **Representations → encoder parameters** (can be done sub-batch by sub-batch once
   representation gradients are known)

#### Full GradCache Algorithm for Contrastive Distillation

```python
def _compute_contrastive_gradients(self, batch):
    """
    GradCache steps 1-3: Compute and cache gradients of the contrastive loss
    with respect to the student's hidden states.

    Returns: cached_grad tensor of shape (C*L, d_model)
    """
    contrastive_batch = extract_contrastive_subset(batch)
    chunk_size = self.gradcache_chunk_size or self.rank_microbatch_size

    # ---- Step 1: No-grad forward through student ----
    # Cache hidden states and RNG states for exact replay.
    student_h_chunks = []
    rng_states = []

    with torch.no_grad():
        for chunk in split_batch(contrastive_batch, chunk_size // seq_len):
            # Save RNG state for exact replay in step 4
            rng_states.append(torch.cuda.get_rng_state())
            input_ids = chunk["input_ids"]
            # Forward through student model (hook captures hidden state)
            self.model(input_ids)
            h = self._hidden_state_cache[0]  # (chunk_B, S, d_model)
            student_h_chunks.append(h)

    student_h = torch.cat(student_h_chunks, dim=0)  # (C, S, d_model)
    student_h_flat = student_h.reshape(-1, student_h.size(-1))  # (C*S, d_model)

    # ---- Step 1b: No-grad forward through contrastive teacher ----
    teacher_e_chunks = []
    teacher_chunk_size = (
        self.contrastive_teacher_microbatch_size or chunk_size
    )

    with torch.no_grad():
        for chunk in split_batch(contrastive_batch, teacher_chunk_size // seq_len):
            e = self.contrastive_teacher.get_embeddings(chunk["input_ids"])
            teacher_e_chunks.append(e)

    teacher_e = torch.cat(teacher_e_chunks, dim=0)  # (C, S, d_teacher)
    teacher_e_flat = teacher_e.reshape(-1, teacher_e.size(-1))  # (C*S, d_teacher)

    # ---- Step 2: Compute loss on detached representations ----
    student_h_leaf = student_h_flat.detach().requires_grad_(True)  # (C*S, d_model)
    teacher_e_detach = teacher_e_flat.detach()                      # (C*S, d_teacher)

    # Optional: project student hidden states to teacher dimension
    if self.projection_head is not None:
        student_proj = self.projection_head(student_h_leaf)  # (C*S, d_proj)
    else:
        student_proj = student_h_leaf                         # (C*S, d_model)

    # Compute similarity matrices
    student_sims = compute_similarity_matrix(student_proj, self.contrastive_temperature)
    teacher_sims = compute_similarity_matrix(teacher_e_detach, self.contrastive_temperature)

    # Compute contrastive loss (scaled by weight lambda)
    loss = self.contrastive_weight * contrastive_distill_loss(
        student_sims, teacher_sims,
        kl_direction=self.contrastive_kl_direction,
        mask_diagonal=self.mask_diagonal,
    )

    # ---- Step 3: Backprop to get representation gradients ----
    loss.backward()

    # Cache the gradient w.r.t. student hidden states.
    # If projection head is used, the gradient is already chain-ruled through it:
    #   d(loss)/d(student_h_leaf) = d(loss)/d(student_proj) @ d(student_proj)/d(student_h_leaf)
    cached_grad = student_h_leaf.grad.detach().clone()  # (C*S, d_model)

    # NOTE: The projection head's .grad is also accumulated from this backward.
    # These gradients will persist and be included in the optimizer step.

    # Clean up large tensors
    del student_h_leaf, student_sims, teacher_sims, student_proj, loss

    return cached_grad, rng_states
```

#### Important: Projection Head Gradient Handling

The projection head (if used) is a trainable `nn.Module` whose parameters need to be updated.
Its gradients are computed during GradCache step 3 (the `loss.backward()` call above) because
`student_proj = self.projection_head(student_h_leaf)` is part of the computation graph.

These gradients accumulate in `self.projection_head.parameters()` and persist through the
main training loop. The optimizer (which includes all model parameters) picks them up during
`optim_step()`.

**Critical**: The projection head must be included in the optimizer's parameter groups. Since
it's separate from the main Transformer model, we need to add it explicitly:

```python
# In DistillationTrainModule.__init__:
if self.projection_head is not None:
    # Add projection head parameters to the optimizer
    # Option A: Separate param group with its own LR
    self.optim.add_param_group({
        "params": list(self.projection_head.parameters()),
        "lr": optim_config.lr,  # Or a separate LR
    })
    # Option B: Include in model before optimizer construction
    # (requires architectural change)
```

### 4.4 RNG State Management

GradCache requires two forward passes through the student encoder: one without grad (step 1)
and one with grad (step 4). If the model uses dropout or other stochastic elements, the
hidden states from both passes must be identical. Otherwise, the cached gradients would be
incorrect.

**Solution**: Save the CUDA RNG state before each no-grad sub-batch forward, and restore it
before the corresponding with-grad forward in step 4.

```python
# Step 1: Save RNG state before each chunk
rng_state = torch.cuda.get_rng_state()
student_h = model(chunk)  # no grad
rng_states.append(rng_state)

# Step 4: Restore RNG state before replay
torch.cuda.set_rng_state(rng_states[chunk_idx])
student_h = model(chunk)  # with grad — same dropout pattern
```

**Alternative**: If the model doesn't use dropout during pretraining (which is common for
modern LLMs like LLaMA, OLMo), RNG management is unnecessary. The JOLMo Transformer
should be checked for stochastic elements during training.

### 4.5 Memory Analysis for Contrastive Loss

For a contrastive batch of C examples with sequence length L:

| Component | Size | Example (C=32, L=2048, d=1024) |
|-----------|------|--------------------------------|
| Student hidden states (cached) | C * L * d * 4 bytes (fp32) | 256 MB |
| Teacher embeddings (cached) | C * L * d_t * 4 bytes | 256 MB (if d_t = d) |
| Similarity matrix (student) | (C*L)^2 * 4 bytes | 16 GB |
| Similarity matrix (teacher) | (C*L)^2 * 4 bytes | 16 GB |
| Cached gradients | C * L * d * 4 bytes | 256 MB |

**The similarity matrix is the bottleneck.** For C=32, L=2048: N = 65,536 tokens.
The N×N matrix is 65536^2 * 4 bytes ≈ 16 GB. This is too large for a single GPU.

**Solutions**:

1. **Reduce the contrastive batch size.** For C=4, L=2048: N = 8,192, matrix = 256 MB. Very
   feasible. For C=8: N = 16,384, matrix = 1 GB. Still feasible.

2. **Tiled similarity computation.** Compute the similarity matrix in tiles/blocks:
   ```
   for i_tile in range(0, N, tile_size):
       for j_tile in range(0, N, tile_size):
           sims_tile = student_normed[i_tile:i_end] @ student_normed[j_tile:j_end].T
           # Accumulate loss/gradients from this tile
   ```
   This reduces memory from O(N^2) to O(tile_size * N). The gradient accumulation on the
   embeddings is correct because the cross-entropy loss decomposes row-wise, and each tile
   contributes to a subset of rows.

3. **Use bf16 for similarity matrices.** Halves the memory. The cosine similarities are in
   [-1/T, 1/T], well within bf16 range.

**Recommendation**: Start with small contrastive batch sizes (C=4 to C=8, giving N=8K to
16K, matrix size 256MB to 1GB). If larger batches are needed, implement tiled computation.

### 4.6 Tiled Similarity Computation (for large contrastive batches)

For when the full N×N similarity matrix doesn't fit in memory:

```python
def tiled_contrastive_loss(student_h_leaf, teacher_e, temperature,
                           kl_direction, tile_size=4096, mask_diagonal=True):
    """
    Compute contrastive loss without materializing the full N×N matrix.

    The loss decomposes row-wise:
      L = (1/N) * sum_i CE(student_sims[i,:], softmax(teacher_sims[i,:]))

    For each row tile (a block of query tokens), we compute the full row
    of similarities against ALL key tokens, compute the row's contribution
    to the loss, and accumulate gradients on student_h_leaf.
    """
    N = student_h_leaf.size(0)
    student_normed = F.normalize(student_h_leaf, p=2, dim=-1)
    teacher_normed = F.normalize(teacher_e.detach(), p=2, dim=-1)

    total_loss = 0.0
    for i_start in range(0, N, tile_size):
        i_end = min(i_start + tile_size, N)

        # Full rows of similarity for this tile of queries
        student_sims_row = (student_normed[i_start:i_end] @ student_normed.T) / temperature
        teacher_sims_row = (teacher_normed[i_start:i_end] @ teacher_normed.T) / temperature

        if mask_diagonal:
            # Mask diagonal entries within this tile
            for k in range(i_end - i_start):
                global_idx = i_start + k
                student_sims_row[k, global_idx] = float('-inf')
                teacher_sims_row[k, global_idx] = float('-inf')

        # Compute loss for this tile of rows
        if kl_direction == "forward":
            teacher_probs = F.softmax(teacher_sims_row, dim=-1)
            student_log_probs = F.log_softmax(student_sims_row, dim=-1)
            tile_loss = -(teacher_probs * student_log_probs).sum(-1).sum()
        # ... (reverse KL similar)

        total_loss = total_loss + tile_loss

    total_loss = total_loss / N
    return total_loss
```

This approach:
- Memory: O(tile_size × N) instead of O(N²)
- Correctness: Exact (not an approximation) because each row computes the full softmax
  over ALL N keys.
- Gradient: `total_loss.backward()` correctly accumulates gradients on `student_h_leaf`
  across all tiles (PyTorch accumulates gradients from multiple backward calls).

**Important subtlety**: Each tile's backward pass through `student_normed[i_start:i_end] @ student_normed.T`
produces gradients on ALL entries of `student_normed` (because the T/transpose side includes
all tokens). So the gradient on `student_h_leaf` is accumulated across tiles, which is correct.

### 4.7 Multi-GPU Considerations for Contrastive Loss

In a distributed setting with data parallelism, each GPU processes a shard of the batch.
For the contrastive loss, we want the similarity matrix to span ALL tokens across ALL
GPUs (not just the local shard), because more tokens = more negatives = richer signal.

**Approach: all-gather representations before computing similarities.**

After GradCache step 1 (each GPU has its local hidden states), do an all-gather to collect
all representations globally:

```python
# Each GPU has local_student_h of shape (local_C * L, d)
# All-gather to get global representations
if is_distributed():
    gathered = [torch.empty_like(local_student_h) for _ in range(world_size)]
    dist.all_gather(gathered, local_student_h)
    global_student_h = torch.cat(gathered, dim=0)  # (global_C * L, d)
else:
    global_student_h = local_student_h
```

Then compute the similarity matrix on the global representations. Each GPU computes the
SAME loss (using the same global similarity matrix), but during GradCache step 3, each GPU
only caches gradients for its LOCAL representations:

```python
# After loss.backward() on the global similarity matrix:
# student_h_leaf.grad has shape (global_C * L, d)
# Each GPU extracts its local portion:
local_cached_grad = student_h_leaf.grad[local_start:local_end].clone()
```

This is the approach described in the GradCache paper's multi-GPU extension. The only extra
communication is the all-gather of representations, which is small compared to gradient
all-reduce.

**Note**: For simplicity, the initial implementation can skip multi-GPU gathering and compute
the similarity matrix using only the local GPU's tokens. This is a valid approximation when
each GPU has enough tokens. Multi-GPU gathering can be added as an optimization.

---

## 5. Combined Objective: Unified Train Step

### 5.1 Full Algorithm

The combined `train_batch()` method handles CE loss, logit distillation, and contrastive
distillation in a single training step:

```python
def train_batch(self, batch, dry_run=False):
    self._set_model_mode("train")

    # Generate labels
    if "labels" not in batch:
        batch["labels"] = get_labels(batch, label_ignore_index=self.label_ignore_index)

    batch_num_tokens_for_loss = (batch["labels"] != self.label_ignore_index).sum()

    # ================================================================
    # Phase 1: Contrastive GradCache (if enabled)
    # ================================================================
    # Computes gradients of contrastive loss w.r.t. student hidden states
    # and caches them for injection during the main forward/backward pass.
    cached_contrastive_grad = None
    contrastive_rng_states = None

    if self.contrastive_teacher is not None and self.contrastive_weight > 0:
        cached_contrastive_grad, contrastive_rng_states = \
            self._compute_contrastive_gradients(batch)

    # ================================================================
    # Phase 2: Main forward/backward loop with gradient accumulation
    # ================================================================
    ce_batch_loss = torch.tensor(0.0, device=self.device)
    kl_batch_loss = torch.tensor(0.0, device=self.device)

    micro_batches = split_batch(batch, self.rank_microbatch_size // seq_len)
    num_micro_batches = len(micro_batches)
    contrastive_token_offset = 0  # Track position in cached_contrastive_grad

    for micro_batch_idx, micro_batch in enumerate(micro_batches):
        with self._train_microbatch_context(micro_batch_idx, num_micro_batches):
            input_ids, labels, model_kwargs = self._prepare_batch(micro_batch)

            # ---- Student forward pass ----
            # Need return_logits=True for logit distillation
            _, loss, ce_loss, z_loss = self.model_forward(
                input_ids, labels=labels,
                ignore_index=self.label_ignore_index,
                loss_reduction="sum",
                loss_div_factor=batch_num_tokens_for_loss,
                return_logits=(self.logit_teacher is not None),
                z_loss_multiplier=self.z_loss_multiplier,
                **model_kwargs,
            )

            ce_batch_loss += ce_loss.detach()

            # ---- Logit distillation loss ----
            if self.logit_teacher is not None and self.logit_weight > 0:
                # Get teacher logits (may chunk for large teachers)
                teacher_logits = self._get_teacher_logits(input_ids)
                student_logits = self._hidden_state_cache[0]  # Or from model output

                if self.logit_kl_direction == "forward":
                    kl_loss = forward_kl_loss(
                        student_logits, teacher_logits.detach(),
                        temperature=self.logit_temperature,
                    )
                else:
                    kl_loss = reverse_kl_loss(
                        student_logits, teacher_logits.detach(),
                        temperature=self.logit_temperature,
                    )

                kl_batch_loss += kl_loss.detach()
                loss = loss + self.logit_weight * kl_loss

            # ---- Contrastive surrogate loss ----
            if cached_contrastive_grad is not None:
                # Determine how many tokens from this microbatch are in the
                # contrastive subset
                mb_num_tokens = input_ids.numel()
                contrastive_end = contrastive_token_offset + mb_num_tokens

                if contrastive_token_offset < cached_contrastive_grad.size(0):
                    # This microbatch has some contrastive examples
                    actual_end = min(contrastive_end, cached_contrastive_grad.size(0))
                    grad_chunk = cached_contrastive_grad[
                        contrastive_token_offset:actual_end
                    ]  # (chunk_tokens, d_model)

                    # Get hidden state from hook (captured during model forward above)
                    h = self._hidden_state_cache[0]  # (B_micro, S, d_model)
                    h_flat = h.reshape(-1, h.size(-1))
                    h_contrastive = h_flat[:grad_chunk.size(0)]

                    # Surrogate loss: injects cached contrastive gradients
                    # into the encoder's computation graph.
                    # d(surrogate)/d(h) = grad_chunk (detached), which is
                    # exactly d(contrastive_loss)/d(h) from GradCache step 3.
                    surrogate = (h_contrastive * grad_chunk.detach()).sum()
                    loss = loss + surrogate

                contrastive_token_offset = contrastive_end

            # ---- Backward pass ----
            loss.backward()

    # Record metrics
    self.record_ce_loss(ce_batch_loss, ReduceType.mean)
    if self.logit_teacher is not None:
        self.record_metric("KL loss", kl_batch_loss, ReduceType.mean, namespace="train")
    if self.contrastive_teacher is not None:
        self.record_metric(
            "Contrastive loss", <contrastive_loss_value>,
            ReduceType.mean, namespace="train"
        )
```

### 5.2 Surrogate Loss: Why It Works

The surrogate loss `(h * cached_grad).sum()` deserves careful explanation:

1. In GradCache step 3, we computed `d(L_contrastive) / d(h)` = `cached_grad` where `h`
   are the student's hidden states (before the LM head).

2. In the main forward pass, the model produces hidden states `h'` (same values as `h`
   because model weights haven't changed and RNG is restored).

3. The surrogate `S = sum(h' * cached_grad)` has gradient `dS/dh' = cached_grad`.

4. Adding `S` to the total loss means `d(total_loss)/dh' = d(ce_loss)/dh' + cached_grad`.

5. Backpropagating through the encoder gives:
   ```
   d(total_loss)/d(params) = d(ce_loss)/d(params) + cached_grad @ d(h')/d(params)
                             = d(ce_loss)/d(params) + d(L_contrastive)/d(params)
   ```
   which is exactly the gradient of `L_CE + L_contrastive` w.r.t. encoder parameters.

**This is exact, not an approximation** — it produces the same gradient as computing both
losses jointly in a single forward/backward pass, but without needing to hold all
embeddings in the computation graph simultaneously.

### 5.3 Loss Weighting & Scaling

The three loss components have different scales:

- **CE loss**: Typically ~2-4 nats/token for LLM pretraining. Already normalized by
  `batch_num_tokens_for_loss`.
- **KL loss**: Scale depends on temperature and how different teacher/student are. At T=1,
  KL ≈ CE initially (before training). The T^2 factor keeps gradients stable.
- **Contrastive loss**: Cross-entropy over an N-token "vocabulary" (where N = C*L). Can be
  large (log(N) ≈ 9 for N=8192). Normalize by the number of query tokens (the `.mean()`
  in the contrastive loss).

The weights `alpha` (logit) and `lambda` (contrastive) should be tuned. Starting points:
- `alpha = 1.0`: Equal weight to CE and KL loss.
- `lambda = 0.1 to 1.0`: Contrastive loss is an auxiliary signal; start small and increase.

Logging all three loss components separately (via `record_metric`) is essential for tuning.

### 5.4 Important: Logit Distillation Requires Logits from Student

The standard `TransformerTrainModule` calls `model_forward(... return_logits=False)` when
using the fused linear+loss path (LigerKernel), which is more memory-efficient but does not
return logits.

For logit distillation, we **must** have the student's logits to compute KL divergence.
This means:
- `return_logits=True` must be set when calling `model_forward`
- The fused linear+loss path cannot be used (since it doesn't materialize logits)
- This increases memory usage by `batch_size * seq_len * vocab_size * 2 bytes` (bf16)
- For B=4, S=2048, V=100K: ~1.6 GB extra per microbatch

**Mitigation**: Use smaller microbatch sizes to offset the extra memory. Or implement a
chunked KL computation that avoids materializing the full logit tensor.

Alternatively, for logit distillation, we can use the default (non-fused) LM head which
already returns logits.

---

## 6. Distributed Training Considerations

### 6.1 FSDP/DDP for Teacher Model

The teacher model needs to be on the same GPUs as the student. For distribution:

**Small teacher (fits on a single GPU)**:
- No distribution needed. Each GPU holds a full copy.
- Memory: ~2× model size in bf16 (e.g., 1B model ≈ 2 GB per GPU).

**Medium teacher (needs sharding)**:
- Wrap with FSDP independently of the student.
- Each GPU holds a shard. FSDP handles all-gather during forward.
- The teacher's FSDP wrapping is independent of the student's.

**Large teacher (70B+)**:
- Even with 8-way FSDP, ~17.5 GB per GPU just for teacher weights.
- Combined with student + optimizer, may exceed GPU memory.
- **Solutions**: (a) Offload teacher to CPU between forward passes, (b) use tensor
  parallelism for the teacher, (c) precompute teacher outputs offline.
- Offline precomputation is recommended for very large teachers.

**Implementation for FSDP teacher**:

```python
# In TeacherModelConfig.build():
if self.dp_config is not None:
    from olmo_core.train.train_module.transformer.common import parallelize_model
    teacher_model = parallelize_model(
        teacher_model,
        world_mesh=student_world_mesh,  # Share the same mesh
        device=device,
        dp_config=self.dp_config,
        # No TP/CP/PP for the teacher (simpler)
    )
```

### 6.2 Memory Budget Analysis

For a typical setup: 610M student, 1B teacher (logit), on 8 GPUs with FSDP:

| Component | Per-GPU Memory (bf16) |
|-----------|-----------------------|
| Student model (FSDP shard) | ~150 MB |
| Student optimizer (AdamW, 2 states) | ~300 MB |
| Teacher model (FSDP shard) | ~250 MB |
| Student activations (1 microbatch) | ~500 MB - 2 GB |
| Teacher activations (1 microbatch) | ~500 MB - 2 GB |
| Logits (student, 1 microbatch) | ~800 MB (with V=100K) |
| Logits (teacher, 1 microbatch) | ~800 MB |
| **Total estimate** | **~3-6 GB** |

This is well within 80 GB A100/H100 budget. The bottleneck is activation memory, which
scales with microbatch size.

For contrastive distillation with C=8 examples, L=2048:

| Component | Memory |
|-----------|--------|
| Student hidden states (cached, bf16) | C*L*d*2 = 8*2048*1024*2 = 32 MB |
| Teacher embeddings (cached, bf16) | 32 MB |
| Similarity matrix (fp32) | (C*L)^2*4 = 16384^2*4 = 1 GB |
| Cached gradients (fp32) | 32 MB |
| **Total contrastive overhead** | **~1.1 GB** |

Very manageable. The similarity matrix dominates. For C=16: ~4 GB. For C=32: ~16 GB
(may need tiling).

### 6.3 Throughput Considerations

- **Logit distillation**: Adds one teacher forward pass per microbatch. For a 1B teacher
  vs 610M student, this roughly doubles the forward-pass time but does NOT increase
  backward-pass time (teacher is frozen). Net overhead: ~50-70% wall-clock time increase.

- **Contrastive distillation with GradCache**: Adds one extra student forward pass (no-grad,
  for GradCache step 1) plus one teacher forward pass. The no-grad forward is ~50% cheaper
  than a grad-enabled forward (no activation storage). Net overhead: ~30-50% wall-clock
  time increase over standard training.

- **Combined**: Both overheads stack. Expect ~80-120% wall-clock increase over standard
  training.

**Optimization opportunities**:
- Use `torch.compile` for both student and teacher (significant speedup from kernel fusion).
- Use bf16 autocast for teacher forward passes (no accuracy impact since teacher is frozen).
- Pipeline the teacher forward with the student backward (requires async execution).

---

## 7. Configuration & YAML Format

### Example YAML for Logit Distillation Only

```yaml
train_module_type: distill

distillation:
  _CLASS_: olmo_core.train.train_module.transformer.config.DistillationConfig
  logit:
    _CLASS_: olmo_core.train.train_module.transformer.config.LogitDistillationConfig
    weight: 1.0
    temperature: 1.0
    kl_direction: forward

logit_teacher:
  _CLASS_: olmo_core.train.train_module.transformer.config.TeacherModelConfig
  hf_model_name: allenai/OLMo-2-0425-1B
  dtype: bfloat16
  rank_microbatch_size: 32768  # Smaller than student's if teacher is larger
```

### Example YAML for Contrastive Distillation Only

```yaml
train_module_type: distill

distillation:
  _CLASS_: olmo_core.train.train_module.transformer.config.DistillationConfig
  contrastive:
    _CLASS_: olmo_core.train.train_module.transformer.config.ContrastiveDistillationConfig
    weight: 0.5
    temperature: 0.1
    kl_direction: forward
    contrastive_batch_size: 16384  # 8 examples * 2048 seq_len
    gradcache_chunk_size: 4096
    mask_diagonal: true

contrastive_teacher:
  _CLASS_: olmo_core.train.train_module.transformer.config.TeacherModelConfig
  hf_model_name: sentence-transformers/all-MiniLM-L6-v2
  dtype: bfloat16
```

### Example YAML for Combined Objective

```yaml
train_module_type: distill

distillation:
  _CLASS_: olmo_core.train.train_module.transformer.config.DistillationConfig
  logit:
    _CLASS_: olmo_core.train.train_module.transformer.config.LogitDistillationConfig
    weight: 1.0
    temperature: 2.0
    kl_direction: forward
  contrastive:
    _CLASS_: olmo_core.train.train_module.transformer.config.ContrastiveDistillationConfig
    weight: 0.1
    temperature: 0.1
    kl_direction: forward
    contrastive_batch_size: 16384
    gradcache_chunk_size: 4096
    mask_diagonal: true

logit_teacher:
  _CLASS_: olmo_core.train.train_module.transformer.config.TeacherModelConfig
  model_config:
    _CLASS_: olmo_core.nn.transformer.config.TransformerConfig
    d_model: 1280
    n_layers: 30
    # ... full model config ...
  checkpoint_path: /path/to/teacher/checkpoint
  dtype: bfloat16
  dp_config:
    _CLASS_: olmo_core.train.train_module.transformer.config.TransformerDataParallelConfig
    name: fsdp
    param_dtype: bfloat16
    reduce_dtype: float32

contrastive_teacher:
  _CLASS_: olmo_core.train.train_module.transformer.config.TeacherModelConfig
  hf_model_name: nomic-ai/nomic-embed-text-v1.5
  dtype: bfloat16
```

---

## 8. File Structure & Changes

### New Files

```
JOLMo/src/olmo_core/train/train_module/transformer/
  distillation_train_module.py    # DistillationTrainModule
  teacher.py                      # TeacherModel, TeacherModelConfig
```

### Modified Files

```
JOLMo/src/olmo_core/train/train_module/transformer/config.py
  + DistillationConfig
  + LogitDistillationConfig
  + ContrastiveDistillationConfig
  + TeacherModelConfig (or in teacher.py)

JOLMo/src/olmo_core/train/train_module/__init__.py
  + Export new classes

JOLMo/src/scripts/launch_from_yaml.py
  + YamlExperimentConfig fields for distillation
  + _build_train_module_distill() function
  + train_module_type == "distill" routing in main()

mixture_pretraining_stages/unordered/training.py
  + DistillationPretrainedModel subclass (or extend PretrainedModel)
```

### Checklist

1. [ ] Create `teacher.py` with `TeacherModel` wrapper and `TeacherModelConfig`
2. [ ] Create `distillation_train_module.py` with `DistillationTrainModule`
3. [ ] Implement `forward_kl_loss()` and `reverse_kl_loss()` utilities
4. [ ] Implement `_compute_contrastive_gradients()` (GradCache steps 1-3)
5. [ ] Implement `compute_similarity_matrix()` and `contrastive_distill_loss()`
6. [ ] Implement the combined `train_batch()` with surrogate loss injection
7. [ ] Add projection head support (optional, for dimension mismatch)
8. [ ] Add configs to `config.py` (`DistillationConfig`, etc.)
9. [ ] Export new classes from `__init__.py`
10. [ ] Add `_build_train_module_distill()` to `launch_from_yaml.py`
11. [ ] Add `train_module_type: "distill"` routing
12. [ ] Add distillation fields to `YamlExperimentConfig`
13. [ ] Create `DistillationPretrainedModel` in `training.py`
14. [ ] Test with `--dry-run` (catches OOMs, validates config)
15. [ ] Test logit distillation alone (single GPU)
16. [ ] Test contrastive distillation alone (single GPU)
17. [ ] Test combined objective (single GPU)
18. [ ] Test multi-GPU (FSDP for both student and teacher)

---

## 9. Considerations & Potential Pitfalls

### 9.1 Fused Linear+Loss Incompatibility

The JOLMo `LMHead` supports a `fused_linear` loss implementation (using Liger-Kernel) that
avoids materializing the full logit tensor. This is more memory-efficient but **does not
return logits**.

For logit distillation, we MUST have student logits. This means:
- When `logit_teacher` is configured, force `return_logits=True` in model forward.
- This implicitly disables the fused path.
- Document that logit distillation uses more memory than standard training due to
  materialized logits.

### 9.2 Contrastive Batch Must Be First in the Data Batch

The contrastive loss operates on a subset of the batch. The implementation assumes this
subset consists of the **first C examples** in the batch (simplest to implement). Since
the data loader shuffles data randomly, any contiguous subset is a random sample.

The contrastive batch size (in tokens) must:
- Divide the global batch size
- Be a multiple of `sequence_length` (whole examples only)
- Be >= `rank_microbatch_size` (at least one microbatch worth)

### 9.3 Gradient Accumulation Interaction

With gradient accumulation (multiple microbatches per optimizer step), the contrastive
GradCache runs ONCE per `train_batch()` call (on the contrastive subset), and its gradients
are injected into the appropriate microbatches during the main loop.

**Critical**: The contrastive GradCache uses the model weights as they are at the START of
`train_batch()`. The main forward/backward loop also uses the same weights (no optimizer
step within `train_batch()`). So the cached gradients are consistent.

### 9.4 Checkpoint Management

The teacher model is frozen and never checkpointed as part of training state. Only the
student model + optimizer + scheduler are checkpointed. This means:
- Teacher does not appear in `state_dict()`
- On resume, the teacher is rebuilt from its config and checkpoint path
- The projection head (if used) DOES appear in `state_dict()` (it has trainable params)

```python
def state_dict(self, *, optim=None):
    sd = super().state_dict(optim=optim)
    if self.projection_head is not None:
        sd["projection_head"] = self.projection_head.state_dict()
    return sd

def load_state_dict(self, state_dict):
    if "projection_head" in state_dict and self.projection_head is not None:
        self.projection_head.load_state_dict(state_dict.pop("projection_head"))
    super().load_state_dict(state_dict)
```

### 9.5 Teacher Model Parallelism Interaction with Student

Both models share the same `world_mesh` (same set of GPUs). Each is FSDP-wrapped
independently. This works because PyTorch FSDP supports multiple independent FSDP instances
on the same process group.

**However**, there's a subtlety: if the student uses tensor parallelism (TP) or pipeline
parallelism (PP), the teacher should NOT (for simplicity). The teacher's distribution should
be limited to FSDP/DDP within the data-parallel dimension of the world mesh.

If the student uses TP, the student's logits might be sharded across TP ranks. The KL loss
computation needs to handle this (or materialize full logits first, which `return_logits=True`
already does in the existing codebase for non-TP eval paths).

### 9.6 Numerical Stability

- **Cosine similarity + low temperature**: `cos_sim / 0.05 = 20 * cos_sim`. Since cos_sim
  is in [-1, 1], the scaled values are in [-20, 20]. This is fine for softmax (no overflow
  risk with fp32).
- **KL divergence with temperature**: When T is large, softmax outputs become very flat, and
  `log(flat_probs)` can have large magnitude. The T^2 scaling compensates. Use fp32 for the
  loss computation even if the model runs in bf16.
- **Log-softmax vs softmax + log**: Always use `F.log_softmax` (numerically stable) rather
  than `F.softmax(...).log()` (numerically unstable).

### 9.7 What If the Contrastive Teacher Has a Different Tokenizer?

If the contrastive teacher uses a different tokenizer than the student, the token-level
embeddings won't align. For example, the teacher might split "unbelievable" into
["un", "believable"] (2 tokens) while the student has it as ["unbeliev", "able"] (also 2
tokens, but different boundaries).

**For the initial implementation, assume the same tokenizer.** If different tokenizers are
needed:
- Decode to text, re-encode with each tokenizer
- Use character-level or word-level alignment to map token embeddings
- Or aggregate to word-level embeddings before computing similarities

This is a significant additional complexity and should be a separate follow-up.

### 9.8 Scaling the Contrastive Loss Gradient Through the Surrogate

The surrogate `S = (h * cached_grad).sum()` injects `cached_grad` as `dS/dh`. But we want
the gradient to be `d(lambda * L_contrastive)/dh = lambda * cached_grad`. If the contrastive
loss was already scaled by `lambda` during GradCache step 3, then `cached_grad` already
includes the lambda factor. Make sure NOT to multiply by lambda again in the surrogate.

If the contrastive loss was NOT scaled by lambda during step 3, then the surrogate should
be `lambda * S`. Pick one convention and be consistent.

**Recommended convention**: Scale during GradCache step 3:
```python
loss = self.contrastive_weight * contrastive_distill_loss(...)
loss.backward()
# cached_grad now includes lambda scaling
```
Then the surrogate is just `S = (h * cached_grad).sum()` without additional scaling.

### 9.9 Dry Run Considerations

The Trainer calls `train_batch(batch, dry_run=True)` before training to detect OOMs and
trigger `torch.compile`. The dry run should:
- Run the teacher forward (to detect teacher OOM)
- Run the contrastive GradCache pipeline (to detect similarity matrix OOM)
- NOT record metrics or update any state

### 9.10 Temperature as a Learnable Parameter?

Some distillation papers make the temperature learnable. For simplicity, we keep it as a
fixed hyperparameter. It can be made learnable later by:
- Making `temperature` an `nn.Parameter`
- Including it in the optimizer's parameter groups
- Adding it to the state dict for checkpointing

---

## 10. Open Questions & Future Work

1. **Offline teacher logits**: For very large teachers, precomputing logits offline
   (potentially with top-K sparsification) is more practical. This would require a separate
   data pipeline that reads both tokens and precomputed logits.

2. **Adaptive KL (AKL)**: The AKL approach dynamically weights forward vs reverse KL based
   on the current alignment between teacher and student. This could improve convergence.

3. **Separate GPU groups for teacher**: Running the teacher on a dedicated set of GPUs
   with cross-GPU communication. More memory-efficient but significantly more complex.

4. **Tiled similarity computation**: For large contrastive batches (C > 16), implement
   tiled similarity computation to avoid materializing the full N×N matrix.

5. **Cross-tokenizer distillation**: Support different tokenizers between teacher and
   student using optimal transport or likelihood matching.

6. **Progressive distillation**: Gradually reduce the teacher's influence over training
   (annealing `alpha` and `lambda` to 0), so the student learns to stand on its own.

7. **Layer-wise contrastive matching**: Instead of using only the last hidden state, match
   intermediate layer representations between teacher and student (requires choosing which
   layers to match).

8. **Sequence-level vs token-level distillation**: For open-ended generation quality,
   sequence-level reverse KL (MiniLLM-style, using policy gradient) may be beneficial but
   requires RLHF-like infrastructure.

---

## Appendix A: GradCache Pseudocode (Standalone Reference)

```python
def gradcache_contrastive_step(
    student_encoder,       # The model (or model-up-to-hidden-state)
    contrastive_teacher,   # Frozen teacher that outputs embeddings
    batch,                 # Dict with input_ids, labels, etc.
    contrastive_batch_size,
    gradcache_chunk_size,
    temperature,
    projection_head=None,
    kl_direction="forward",
    mask_diagonal=True,
):
    """
    Full GradCache procedure for contrastive distillation.
    Returns: cached_grad tensor and list of RNG states.
    """
    seq_len = batch["input_ids"].shape[1]
    contrastive_examples = contrastive_batch_size // seq_len
    contrastive_batch = {k: v[:contrastive_examples] for k, v in batch.items()}

    # ============================================================
    # STEP 1: No-grad forward through student → cache hidden states
    # ============================================================
    student_h_list = []
    rng_states = []

    with torch.no_grad():
        chunks = split_batch(contrastive_batch, gradcache_chunk_size // seq_len)
        for chunk in chunks:
            rng_states.append(torch.cuda.get_rng_state())
            h = student_encoder(chunk["input_ids"])  # (B_chunk, S, d)
            student_h_list.append(h)

    student_h = torch.cat(student_h_list)       # (C, S, d)
    student_h_flat = student_h.reshape(-1, student_h.size(-1))  # (C*S, d)

    # ============================================================
    # STEP 1b: No-grad forward through teacher → cache embeddings
    # ============================================================
    teacher_e_list = []

    with torch.no_grad():
        for chunk in split_batch(contrastive_batch, gradcache_chunk_size // seq_len):
            e = contrastive_teacher.get_embeddings(chunk["input_ids"])
            teacher_e_list.append(e)

    teacher_e = torch.cat(teacher_e_list)
    teacher_e_flat = teacher_e.reshape(-1, teacher_e.size(-1))

    # ============================================================
    # STEP 2: Compute contrastive loss on detached representations
    # ============================================================
    student_h_leaf = student_h_flat.detach().requires_grad_(True)
    teacher_e_detach = teacher_e_flat.detach()

    # Optional projection
    if projection_head is not None:
        student_proj = projection_head(student_h_leaf)
    else:
        student_proj = student_h_leaf

    # Similarity matrices
    student_sims = compute_similarity_matrix(student_proj, temperature)
    teacher_sims = compute_similarity_matrix(teacher_e_detach, temperature)

    # Loss
    loss = contrastive_distill_loss(
        student_sims, teacher_sims,
        kl_direction=kl_direction,
        mask_diagonal=mask_diagonal,
    )

    # ============================================================
    # STEP 3: Backward to get representation gradients
    # ============================================================
    loss.backward()
    cached_grad = student_h_leaf.grad.detach().clone()

    # Reshape back to (C, S, d) for alignment with microbatches
    cached_grad = cached_grad.reshape(student_h.shape)

    # Clean up
    del student_h_leaf, student_sims, teacher_sims, student_proj, loss

    return cached_grad, rng_states
```

## Appendix B: Contrastive Loss with All-Gather for Multi-GPU

```python
def contrastive_loss_distributed(
    local_student_h,    # (local_C * L, d) — this GPU's student hidden states
    local_teacher_e,    # (local_C * L, d_t) — this GPU's teacher embeddings
    temperature,
    kl_direction,
    mask_diagonal,
):
    """
    Compute contrastive loss using all-gathered representations from all GPUs.
    Each GPU gets the full similarity matrix but only back-propagates to its
    local representations.
    """
    world_size = dist.get_world_size()

    # All-gather student representations
    all_student_h = [torch.empty_like(local_student_h) for _ in range(world_size)]
    dist.all_gather(all_student_h, local_student_h)

    # The local portion needs grad; remote portions are detached
    all_student_h[dist.get_rank()] = local_student_h  # keep grad
    for i in range(world_size):
        if i != dist.get_rank():
            all_student_h[i] = all_student_h[i].detach()

    global_student_h = torch.cat(all_student_h, dim=0)  # (global_N, d)

    # All-gather teacher representations (no grad needed)
    all_teacher_e = [torch.empty_like(local_teacher_e) for _ in range(world_size)]
    dist.all_gather(all_teacher_e, local_teacher_e)
    global_teacher_e = torch.cat(all_teacher_e, dim=0).detach()

    # Compute global similarity matrices
    student_sims = compute_similarity_matrix(global_student_h, temperature)
    teacher_sims = compute_similarity_matrix(global_teacher_e, temperature)

    # Compute loss (gradient only flows to local_student_h)
    loss = contrastive_distill_loss(student_sims, teacher_sims, kl_direction, mask_diagonal)

    return loss
```
