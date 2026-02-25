# Master Plan: Implementing New Optimizers in JOLMo

> **IMPORTANT: Implement ONE optimizer at a time from the checklist below.** After completing each optimizer, update this file to check it off and record any learnings in the Agent State section. Read the individual plan file for the optimizer you are implementing before starting.

## How to Use This Plan

1. Pick the next unchecked optimizer from the checklist below.
2. Read the corresponding `<optimizer>.md` plan file in this directory.
3. Read the skill files at `agent/skills/jolmo-optimizers.md` and `agent/skills/jolmo-pretraining.md` for the JOLMo framework conventions.
4. Implement the optimizer following the steps in the plan.
5. Write, but don't run any available tests to verify correctness.
6. Come back here, check off the completed optimizer, and update the Agent State section with anything you learned.

## Key Files Reference

| File | Purpose |
|------|---------|
| `JOLMo/src/olmo_core/optim/config.py` | `OptimConfig` base class — all optimizer configs extend this |
| `JOLMo/src/olmo_core/optim/__init__.py` | Public exports — add new optimizer imports and `__all__` entries here |
| `JOLMo/src/olmo_core/optim/adamw.py` | Reference for simple optimizer (AdamWConfig wraps torch.optim.AdamW) |
| `JOLMo/src/olmo_core/optim/lion.py` | Reference for custom optimizer (Lion class + LionConfig) |
| `JOLMo/src/olmo_core/optim/muon.py` | Reference for composite optimizer (Muon wraps torch.optim.Muon + AdamW, custom `build()`) |
| `mixture-pretraining/mixture_pretraining_stages/training.py` | Wire optimizer into experiment system (`_optimizer_class_name`, `_build_optimizer_spec`, `JolmoModel`) |

## Implementation Pattern Summary

For each optimizer, the implementation involves:

1. **Create optimizer file** (`JOLMo/src/olmo_core/optim/<name>.py`):
   - Optimizer class extending `torch.optim.Optimizer`
   - Config dataclass extending `OptimConfig` with `@classmethod optimizer()` returning the optimizer class
   - Config field names MUST match optimizer `__init__` kwargs exactly
   - Use `get_local_tensor()` for DTensor/FSDP compatibility
   - Use tensor step counters (not Python ints) to avoid host-device sync

2. **Export** from `JOLMo/src/olmo_core/optim/__init__.py`

3. **Wire into mixture-pretraining** (optional but recommended):
   - Add to `_optimizer_class_name()` in `training.py`
   - Add special case in `_build_optimizer_spec()` if needed
   - Update `JolmoModel.optimizer` Literal type

4. **Write tests** (in `JOLMo/src/test/optim/`)

---

## Implementation Checklist

### Tier 1: Simple (AdamW-like state, minimal code changes)

- [x] **NAdamW** — [nadamw.md](nadamw.md)
  - Difficulty: Easy
  - Same state as AdamW, one extra `.lerp()` call for Nesterov correction
  - No new hyperparameters to tune vs AdamW
  - Estimated lines: ~80 (optimizer) + ~20 (config)

- [x] **Cautious (C-AdamW)** — [cautious.md](cautious.md)
  - Difficulty: Easy
  - Same state as AdamW, one extra mask computation
  - No new hyperparameters to tune vs AdamW
  - Estimated lines: ~90 (optimizer) + ~20 (config)

### Tier 2: Moderate (extra state, new hyperparameters, but standard optimizer pattern)

- [x] **Mars** — [mars.md](mars.md)
  - Difficulty: Moderate
  - Extra `last_grad` buffer per parameter (+50% memory)
  - Variance-reduction correction + per-parameter gradient clipping
  - Different default hyperparameters than AdamW (higher LR, lower WD)
  - Separate handling of 1D vs 2D parameters
  - Estimated lines: ~150 (optimizer) + ~30 (config)

- [x] **Sophia** — [sophia.md](sophia.md)
  - Difficulty: Moderate-Hard
  - Same state as AdamW (exp_avg + hessian)
  - **Requires training loop modifications** for Hessian estimation (GNB or Hutchinson)
  - The optimizer itself is simple; the integration complexity is in the training loop
  - Estimated lines: ~100 (optimizer) + ~30 (config) + ~50 (training loop integration)

### Tier 3: Complex (composite optimizers, custom `build()`, matrix operations)

- [x] **Scion** — [scion.md](scion.md)
  - Difficulty: Hard
  - Composite optimizer like Muon: different update rules per parameter type
  - Custom `build()` method to classify parameters
  - Newton-Schulz iteration for 2D params (CUDA-only, bfloat16)
  - Frank-Wolfe constrained update (no standard weight decay)
  - Estimated lines: ~200 (optimizer) + ~80 (config with custom build)

- [x] **Soap** — [soap.md](soap.md)
  - Difficulty: Hard
  - Gram matrix tracking + eigenvector computation
  - Gradient rotation via tensordot projections
  - Periodic eigenbasis refresh via power iteration + QR
  - Complex re-projection logic when eigenbasis changes
  - Estimated lines: ~350 (optimizer) + ~30 (config)

- [x] **Kron (PSGD)** — [kron.md](kron.md)
  - Difficulty: Very Hard
  - Kronecker-factored preconditioner with triangular/diagonal factor management
  - Einsum expression generation and caching
  - Probabilistic update scheduling
  - Dimension merging for >2D tensors
  - Recommend vendoring from `kron_torch` rather than reimplementing
  - Estimated lines: ~500+ (optimizer + helpers) + ~40 (config)

---

## Agent State

> This section is for the implementing agent to record important learnings, gotchas, and decisions made during implementation. Update this after completing each optimizer.

### General Learnings

<!-- Update this as you implement optimizers -->
- The `OptimConfig.build()` default path passes all config fields (except `group_overrides`, `compile`, `fixed_fields`) as kwargs to the optimizer constructor. NAdamW has the same fields as AdamW so no special handling needed in `_build_optimizer_spec()`.
- The existing default path in `_build_optimizer_spec()` already handles `betas`, `lr`, `weight_decay`, and `group_overrides`. Only need to gate `fused=True` for torch built-in optimizers.
- For `step_size = lr / bias_correction1` where both can be tensors, use direct multiplication/division (tensor ops) rather than `addcdiv_` with `value=` which requires a Python scalar.

### Per-Optimizer Notes

#### NAdamW
<!-- Fill in after implementing -->
- Status: Complete
- Files created: `nadamw.py` (optimizer + config), `test/optim/nadamw_test.py` (5 tests)
- Files modified: `optim/__init__.py` (exports), `mixture_pretraining_stages/training.py` (wiring)
- Notes: Straightforward implementation. Same state as AdamW. The key difference is one non-in-place `.lerp()` call for Nesterov correction. No special handling needed in `_build_optimizer_spec` since field names match the default AdamW-style path (minus `fused`).

#### Cautious (C-AdamW)
<!-- Fill in after implementing -->
- Status: Complete
- Files created: `cautious.py` (optimizer + config), `test/optim/cautious_test.py` (6 tests)
- Files modified: `optim/__init__.py` (exports), `mixture_pretraining_stages/training.py` (wiring)
- Notes: Very straightforward — nearly identical to NAdamW structurally. The key difference is the sign-alignment mask `(exp_avg * grad > 0)` with mean normalization via `.clamp_(min=mask_eps)`. No special handling needed in `_build_optimizer_spec` since field names follow the default AdamW-style path (minus `fused`). The extra `mask_eps` field uses its default value from the config. Same state as AdamW (exp_avg, exp_avg_sq, step).

#### Mars
<!-- Fill in after implementing -->
- Status: Complete
- Files created: `mars.py` (optimizer + config), `test/optim/mars_test.py` (6 tests)
- Files modified: `optim/__init__.py` (exports), `mixture_pretraining_stages/training.py` (wiring)
- Notes: More complex than NAdamW/Cautious due to: (1) extra `last_grad` state buffer (+50% memory), (2) separate 1D vs 2D parameter handling with different betas/lr/wd, (3) variance-reduction correction `c_t = g + gamma * beta1/(1-beta1) * (g - g_prev)` with per-parameter L2 clipping. Uses coupled weight decay (L2 regularization) for 2D params per the MARS paper, but decoupled weight decay for 1D fallback. Added special case in `_build_optimizer_spec` to pass `gamma` field. The `gamma=0, optimize_1d=True, weight_decay=0` configuration reduces exactly to AdamW (verified in tests).

#### Sophia
<!-- Fill in after implementing -->
- Status: Complete
- Files created: `sophia.py` (optimizer + config), `test/optim/sophia_test.py` (6 tests)
- Files modified: `optim/__init__.py` (exports), `mixture_pretraining_stages/training.py` (wiring)
- Notes: The optimizer itself is straightforward — same state as AdamW (exp_avg + hessian replaces exp_avg_sq) with element-wise clipping via `ratio = (|m| / (rho * h + eps)).clamp_(max=1)`. The key design choice is that `hessian_update_interval` is accepted by `Sophia.__init__` but NOT placed in `defaults` (param groups) — it's stored as `self.hessian_update_interval` for the training loop to read. This allows `OptimConfig.build()` to pass it through cleanly. Added special case in `_build_optimizer_spec` to pass `rho` and `hessian_update_interval`. Without `update_hessian()` calls, hessian stays zero and Sophia degrades to `lr * sign(m)` (SignSGD with momentum). **Important**: Sophia requires training loop integration for the Hessian estimation (GNB or Hutchinson), which is NOT implemented here — only the optimizer and its `update_hessian()` / `update_hessian_from_estimates()` methods are provided. The training loop must call these periodically.

#### Scion
<!-- Fill in after implementing -->
- Status: Complete
- Files created: `scion.py` (optimizer + config), `test/optim/scion_test.py` (8 tests)
- Files modified: `optim/__init__.py` (exports), `mixture_pretraining_stages/training.py` (wiring)
- Notes: Follows the MuonConfig.build() composite pattern — custom `build()` classifies params by norm type (spectral/sign/bias_rms) and creates separate param groups. Key differences from Muon: (1) Scion is a single optimizer class, not a wrapper around two sub-optimizers — it handles all three norm types itself in `step()`, (2) uses Frank-Wolfe constrained update `(1 - lr) * theta - lr * scale * d` which provides implicit weight decay (no separate `weight_decay` param), (3) only stores one momentum buffer per param (no second moment), so ~50% less optimizer state than AdamW. The Newton-Schulz iteration is identical to Muon's (same coefficients, bfloat16, transpose trick). Scion uses `scale` per norm group instead of `lr` per optimizer type. Added special case in `_build_optimizer_spec` to pass `momentum`, `spectral_scale`, `sign_scale`, `bias_rms_scale` (no `betas`/`weight_decay` since Scion doesn't use them).

#### Soap
<!-- Fill in after implementing -->
- Status: Complete
- Files created: `soap.py` (optimizer + config), `test/optim/soap_test.py` (8 tests)
- Files modified: `optim/__init__.py` (exports), `mixture_pretraining_stages/training.py` (wiring)
- Notes: Most complex self-contained optimizer so far. Key design decisions: (1) First step is skipped — only initializes Gram matrices and computes initial eigenbasis via `eigh`; parameter update starts on step 2. (2) Eigenbasis refresh uses power iteration + QR every `precondition_frequency` steps, with eigenvalue-based reordering of `exp_avg_sq` to maintain consistency. (3) Momentum re-projection: before each eigenbasis update, `exp_avg` is projected back to original space, then re-projected into the (possibly new) eigenbasis. (4) 1D params skip preconditioning by default (`precondition_1d=False`) — they get Gram matrix entries as empty lists `[]` and projections just permute dims. (5) No custom `build()` needed — `shampoo_beta=None` is resolved to `betas[1]` in `Soap.__init__`, and all config fields pass through `OptimConfig.build()` cleanly. (6) Added special case in `_build_optimizer_spec` to pass `precondition_frequency`. (7) FSDP limitation: Gram matrices are computed from local shards, which is an approximation — a proper fix would require `all_reduce` on outer products (noted in plan as known limitation but not implemented).

#### Kron (PSGD)
<!-- Fill in after implementing -->
- Status: Complete
- Files created: `kron.py` (optimizer + config + vendored helpers), `test/optim/kron_test.py` (8 tests)
- Files modified: `optim/__init__.py` (exports), `mixture_pretraining_stages/training.py` (wiring)
- Notes: Most complex optimizer in the set. Vendored core logic from `kron_torch` with JOLMo adaptations: (1) `get_local_tensor()` on all state tensor accesses for DTensor compatibility, (2) tensor step counters to avoid host-device sync, (3) deterministic balance schedule (`_prob_step % 100 == 0`) instead of `random.Random` for distributed compatibility, (4) host-device-sync-free RMS clipping via `(1.1 / (rms + 1e-12)).clamp_(max=1.0)`, (5) `state_dict()`/`load_state_dict()` overrides to persist global `_prob_step` and `_update_counter` across checkpoints (prevents compute spike on resume). Key design: (a) einsum expressions are generated at init time in `_build_exprs()` and cached in per-parameter state; (b) `_apply_Q_inverse()` uses `torch.linalg.solve_triangular` in float32 for stability even with bf16 gradients; (c) dimension merging via `_merge_dims()` collapses >2D tensors into ≤2D for tractable Kronecker factorization; (d) `_ProbScheduler` anneals preconditioner update frequency from 100% to 3% of steps. Config fields pass through `OptimConfig.build()` cleanly — `precond_update_prob_*` fields are consumed by `Kron.__init__` for the scheduler but not placed in `defaults` (param groups). Added special case in `_build_optimizer_spec` to map `betas[0]` → `b1`. FSDP limitation: Kronecker factors are computed from local shards (shard-local approximation), same as Soap's Gram matrices — not empirically validated but acceptable starting point per the plan.

---

## Appendix: Quick Reference for Optimizer Characteristics

| Optimizer | Family | Extra State vs AdamW | Custom `build()`? | Training Loop Changes? | Memory Overhead |
|-----------|--------|---------------------|-------------------|----------------------|-----------------|
| NAdamW | Scalar (Nesterov) | None | No | No | 0% |
| Cautious | Scalar (masked) | None | No | No | 0% |
| Mars | Scalar (variance-reduced) | +1 buffer (last_grad) | No | No | +50% |
| Sophia | Hessian approximation | None (hessian replaces exp_avg_sq) | No | **Yes** (Hessian estimation) | 0% |
| Scion | Matrix-based | -1 buffer (no second moment) | **Yes** (param classification) | No | -50% |
| Soap | Matrix-based | +Gram matrices + eigenvectors | No | No | +O(m² + n²) |
| Kron | Matrix-based | +Q factors (triangular/diagonal) | No | No | +O(m² + n²) |

## Appendix: Recommended Implementation Order

The checklist above is ordered by difficulty. The recommended implementation order is:

1. **NAdamW** — easiest, builds confidence with the framework
2. **Cautious** — easy, similar pattern to NAdamW
3. **Mars** — first moderate optimizer, introduces extra state management
4. **Sophia** — introduces training loop integration complexity
5. **Scion** — first complex optimizer, introduces custom `build()` pattern (like Muon)
6. **Soap** — complex matrix operations but self-contained optimizer
7. **Kron** — most complex, recommend doing last (or vendoring from kron_torch)
