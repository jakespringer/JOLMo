# Kron (PSGD) Optimizer Implementation Plan

## Overview

**Kron** is the Kronecker-factored variant of PSGD (Preconditioned Stochastic Gradient Descent). It maintains per-tensor Kronecker preconditioner factors `Q_i` such that `P = Q^T Q` approximates the inverse Hessian. The preconditioner is updated probabilistically using random probes and a whitening criterion, then applied to the gradient for a preconditioned update.

**Paper references:**
- Li (2015/2018), "Preconditioned Stochastic Gradient Descent" (IEEE TNNLS)
- Li (2024), "Stochastic Hessian Fittings with Lie Groups" (arXiv:2402.11858)
- Wen et al. (2025), "Fantastic Pretraining Optimizers and Where to Find Them" (Algorithm 10)

**Reference implementations:**
- `kron_torch` (Evan Walters): https://github.com/evanatyourservice/kron_torch (`pip install kron-torch`)
- `psgd_torch` (Xi-Lin Li, original author): https://github.com/lixilinx/psgd_torch

**Complexity:** Very high — this is the most complex optimizer to implement. Involves Kronecker-factored preconditioners, einsum expressions for multi-dimensional tensors, triangular matrix updates, periodic balancing, probabilistic update scheduling, and dimension merging.

---

## Algorithm (Pseudocode)

```
Input: params θ, lr η, momentum β₁, weight_decay λ,
       precond_lr α, max_size_triangular, min_ndim_triangular

Init per param:
    μ = 0 (momentum buffer)
    For each dimension d of grad shape:
        if dim_size ≤ max_size_triangular and ndim ≥ min_ndim_triangular:
            Q_d = scale^(1/ndim) * I    (upper triangular)
        else:
            Q_d = scale^(1/ndim) * ones  (diagonal, stored as vector)
    update_counter = 0, step = 0

for each step:
    step += 1
    g = ∇L(θ)

    # 1. Momentum with bias correction
    μ = β₁ * μ + (1 - β₁) * g
    μ_hat = μ / (1 - β₁^step)

    # 2. Determine if preconditioner update happens
    p = schedule(step)          # anneals from 1.0 to 0.03
    update_counter += 1
    do_update = (update_counter >= 1/p)
    if do_update: update_counter = 0

    # 3. Optionally balance Q factors (~every 100 updates)
    if do_update and random() < 0.01 and ndim > 1:
        norms = [||Q_d||_inf for all d]
        geo_mean = exp(mean(log(norms)))
        Q_d *= geo_mean / norms[d]

    # 4. Update preconditioner (if scheduled)
    if do_update:
        G = μ_hat (or raw gradient)
        V = randn_like(G)
        G_noised = G + sqrt(eps) * mean(|G|) * V

        # Forward: A = Q₁ ⊗ Q₂ ⊗ ... @ G_noised (via einsum)
        A = einsum(exprA, Q₁, Q₂, ..., G_noised)

        # Inverse: conjB = (Q₁ ⊗ Q₂ ⊗ ...)⁻¹ @ V (via triangular solves)
        conjB = solve_kronecker_inverse(Q_factors, V)

        # Update each factor
        for each Q_d:
            term1 = einsum(exprG_d, A, A)        # ≈ A A^T along dim d
            term2 = einsum(exprG_d, conjB, conjB) # ≈ B^T B along dim d
            grad_Q = triu(term1 - term2)
            normalizer = spectral_norm_lb(term1 + term2)
            Q_d -= α * (grad_Q / normalizer) @ Q_d

    # 5. Precondition gradient
    #    pre_grad = (Q₁^T Q₁) ⊗ (Q₂^T Q₂) ⊗ ... @ μ_hat
    pre_grad = einsum(exprP, Q₁, Q₂, ..., Q₁, Q₂, ..., μ_hat)

    # 6. Clip update RMS
    rms = sqrt(mean(pre_grad²))
    if rms > 1.1:
        pre_grad *= 1.1 / (rms + 1e-12)

    # 7. Weight decay (decoupled, only for dim >= 2)
    if λ > 0 and ndim >= 2:
        pre_grad += λ * θ

    # 8. Parameter update
    θ -= η * pre_grad
```

---

## Implementation Strategy

Given the extreme complexity, the recommended approach is to **vendor the `kron_torch` implementation** and adapt it to JOLMo's patterns, rather than reimplementing from scratch.

### Step 1: Create `JOLMo/src/olmo_core/optim/kron.py`

**Strategy: Adapt the `kron_torch` implementation.**

The key adaptations needed:
1. Use `get_local_tensor()` for DTensor compatibility
2. Wrap in JOLMo's `OptimConfig` pattern
3. Ensure state tensors are properly initialized on the correct device
4. Handle the probabilistic update schedule as part of optimizer state

Core structure:

```python
import math
import string
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Type

import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer

from ..distributed.utils import get_local_tensor
from .config import OptimConfig

import logging
log = logging.getLogger(__name__)


class _ProbScheduler:
    """Exponential anneal with flat start for preconditioner update probability."""
    def __init__(self, max_prob=1.0, min_prob=0.03, decay=0.001, flat_start=500):
        self.max_prob = max_prob
        self.min_prob = min_prob
        self.decay = decay
        self.flat_start = flat_start

    def __call__(self, step: int) -> float:
        prob = self.max_prob * math.exp(-self.decay * max(0, step - self.flat_start))
        return max(self.min_prob, min(self.max_prob, prob))


# ─── Einsum expression builders (from kron_torch) ───
# These generate einsum strings at init time for efficient Kronecker ops.
# The full implementation should be vendored from kron_torch/kron.py.
# Key functions: _init_Q_exprs(), _update_precond(), _precond_grad(), _balance_Q()


class Kron(Optimizer):
    """
    PSGD Kron: Preconditioned SGD with Kronecker-factored preconditioner.

    Maintains per-tensor Kronecker factors Q such that P = Q^T Q
    approximates the inverse Hessian. Preconditioner updates are
    amortized via a probabilistic schedule.

    Reference:
        Li (2015), "Preconditioned Stochastic Gradient Descent"
        Li (2024), "Stochastic Hessian Fittings with Lie Groups"
    """

    def __init__(
        self,
        params,
        lr: float = 3e-4,
        b1: float = 0.9,
        weight_decay: float = 0.0,
        precond_lr: float = 0.1,
        precond_init_scale: float = 1.0,
        max_size_triangular: int = 8192,
        min_ndim_triangular: int = 2,
        memory_save_mode: Optional[str] = None,
        momentum_into_precond_update: bool = True,
        merge_dims: bool = True,
        precond_update_prob_max: float = 1.0,
        precond_update_prob_min: float = 0.03,
        precond_update_prob_decay: float = 0.001,
        precond_update_prob_flat_start: int = 500,
    ):
        defaults = dict(
            lr=lr, b1=b1, weight_decay=weight_decay,
            precond_lr=precond_lr,
            precond_init_scale=precond_init_scale,
            max_size_triangular=max_size_triangular,
            min_ndim_triangular=min_ndim_triangular,
            memory_save_mode=memory_save_mode,
            momentum_into_precond_update=momentum_into_precond_update,
            merge_dims=merge_dims,
        )
        super().__init__(params, defaults)
        self._prob_scheduler = _ProbScheduler(
            precond_update_prob_max, precond_update_prob_min,
            precond_update_prob_decay, precond_update_prob_flat_start,
        )
        self._prob_step = 0
        self._update_counter = 0
        # NOTE: Do NOT use random.Random for balance decisions — it is not
        # synchronized across distributed ranks. Use deterministic step-based
        # logic instead (e.g., balance every 100 preconditioner updates).

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            with torch.enable_grad():
                closure()

        # Determine if preconditioner update happens this step
        update_prob = self._prob_scheduler(self._prob_step)
        self._update_counter += 1
        do_update = self._update_counter >= 1.0 / update_prob
        if do_update:
            self._update_counter = 0
        self._prob_step += 1
        # Use deterministic step-based balance instead of random.Random
        # (random.Random is not synchronized across distributed ranks)
        balance = do_update and self._prob_step % 100 == 0

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = get_local_tensor(p.grad)
                p_local = get_local_tensor(p)
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = torch.zeros((), dtype=torch.float32, device=p.device)
                    # Merge dims for >2D tensors
                    merged_shape = _best_merge_shape(grad.shape) if group["merge_dims"] and grad.dim() > 2 else None
                    if merged_shape is not None:
                        state["merged_shape"] = merged_shape
                    work_grad = grad.view(*merged_shape) if merged_shape else grad
                    state["momentum_buffer"] = torch.zeros_like(work_grad)
                    state["Q"], state["exprs"] = _init_Q_exprs(
                        work_grad, group["precond_init_scale"],
                        group["max_size_triangular"],
                        group["min_ndim_triangular"],
                        group["memory_save_mode"],
                    )

                step = state["step"]
                step.add_(1)

                # Reshape if needed
                work_grad = grad.view(*state["merged_shape"]) if "merged_shape" in state else grad

                # Momentum with bias correction
                mu = state["momentum_buffer"]
                mu.mul_(group["b1"]).add_(work_grad, alpha=1.0 - group["b1"])
                mu_hat = mu / (1.0 - group["b1"] ** step)

                # Balance Q factors
                if work_grad.dim() > 1 and balance:
                    _balance_Q(state["Q"])

                # Update preconditioner
                if do_update:
                    signal = mu_hat if group["momentum_into_precond_update"] else work_grad
                    _update_precond(state["Q"], state["exprs"], signal, step=group["precond_lr"])

                # Precondition gradient
                pre_grad = _precond_grad(state["Q"], state["exprs"], mu_hat)

                # Clip update RMS (avoid Python if on GPU tensor for host-device sync)
                rms = pre_grad.pow(2).mean().sqrt()
                clip_scale = (1.1 / (rms + 1e-12)).clamp_(max=1.0)
                pre_grad.mul_(clip_scale)

                # Weight decay (decoupled, only for dim >= 2)
                # Use work_grad.dim() (local tensor) instead of p.dim() (may be DTensor)
                if group["weight_decay"] != 0 and work_grad.dim() >= 2:
                    pre_grad.add_(p_local.view(pre_grad.shape), alpha=group["weight_decay"])

                # Parameter update
                p_local.add_(pre_grad.view(p_local.shape), alpha=-group["lr"])


# ─── Helper functions to vendor from kron_torch ───
# The following functions need to be vendored/adapted from kron_torch/kron.py:
#
# _init_Q_exprs(grad, scale, max_size_tri, min_ndim_tri, memory_save_mode)
#   → Initializes Q factors (triangular or diagonal) and einsum expressions
#
# _update_precond(Q, exprs, grad, step)
#   → Updates Q factors using random probes and whitening criterion
#
# _precond_grad(Q, exprs, grad)
#   → Applies P = Q^T Q to the gradient via einsum
#
# _balance_Q(Q)
#   → Rebalances Q factors so their norms are equal (geometric mean)
#
# _best_merge_shape(shape)
#   → Finds optimal way to merge dimensions for >2D tensors


@dataclass
class KronConfig(OptimConfig):
    """
    Configuration class for building a :class:`Kron` (PSGD) optimizer.
    """

    lr: float = 3e-4
    b1: float = 0.9
    weight_decay: float = 0.0
    precond_lr: float = 0.1
    precond_init_scale: float = 1.0
    max_size_triangular: int = 8192
    min_ndim_triangular: int = 2
    memory_save_mode: Optional[str] = None
    momentum_into_precond_update: bool = True
    merge_dims: bool = True
    precond_update_prob_max: float = 1.0
    precond_update_prob_min: float = 0.03
    precond_update_prob_decay: float = 0.001
    precond_update_prob_flat_start: int = 500

    @classmethod
    def optimizer(cls) -> Type[Kron]:
        return Kron
```

### Step 2: Vendor helper functions from `kron_torch`

The core helper functions (`_init_Q_exprs`, `_update_precond`, `_precond_grad`, `_balance_Q`, `_best_merge_shape`) should be vendored from `kron_torch/kron.py`. These are ~400 lines of code that handle:
- Building einsum expression strings at init time
- Triangular vs diagonal preconditioner factor management
- Random probe generation and whitening-based updates
- Spectral norm estimation for normalizing updates

**This is the bulk of the implementation work.** The vendored code should be placed in helper functions within `kron.py` or a separate `kron_utils.py` file.

**When vendoring, apply these JOLMo-specific modifications:**
- Replace any `if tensor > threshold` patterns with `torch.clamp`/`torch.where` to avoid host-device sync
- Add `get_local_tensor()` calls on any state tensors accessed outside the main `step()` loop
- Replace `random.Random` usage with deterministic step-based logic for distributed compatibility
- Ensure `torch.linalg.solve_triangular` calls use float32 even if gradients are bf16
- Verify all einsum expressions operate on the correct (local shard) tensor shapes

### Step 3: Export from `__init__.py`

```python
from .kron import Kron, KronConfig

__all__ = [
    ...
    "Kron",
    "KronConfig",
]
```

### Step 4: Wire into mixture-pretraining

1. Add to `_optimizer_class_name()`:
```python
if name == "kron":
    return "olmo_core.optim.KronConfig"
```

2. Add special case in `_build_optimizer_spec()`:
```python
if cls == "olmo_core.optim.KronConfig":
    return {
        "_CLASS_": cls,
        "lr": lr,
        "b1": betas[0],
        "weight_decay": weight_decay,
        "group_overrides": [embedding_override],
    }
```

3. Update `JolmoModel.optimizer` Literal type.

### Step 5: Write tests

Create `JOLMo/src/test/optim/kron_test.py`:
- `test_config_builds_correctly()`
- `test_optimizer_step()` — verify parameters update
- `test_preconditioner_init()` — verify Q factors are initialized correctly
- `test_precond_update_schedule()` — verify the probabilistic schedule anneals

---

## Default Hyperparameters for LLM Pretraining

| Parameter | Value | Notes |
|-----------|-------|-------|
| lr | 1e-3 | ~3x higher than Adam |
| b1 | 0.9 | Same as Adam's beta1 |
| weight_decay | 0.5 | Much higher than Adam's 0.1 (per Wen et al.) |
| precond_lr | 0.1 | Must be ≥ 0.1 for bf16 stability |
| max_size_triangular | 8192 | Dims > this use diagonal preconditioner |
| min_ndim_triangular | 2 | 1D params get diagonal preconditioner |
| memory_save_mode | None | None=all triangular; 'one_diag' saves memory |
| precond_update_prob | 1.0 → 0.03 | Anneals over training |

---

## Memory & Compute Overhead

- **State per parameter (m x n matrix):**
  - Momentum buffer: m*n
  - Q₁ (triangular): m*(m+1)/2 (or m for diagonal)
  - Q₂ (triangular): n*(n+1)/2 (or n for diagonal)
  - Einsum expression strings (small)
- **Memory overhead vs AdamW:** Can be significantly higher for large dimensions (Q factors are O(m² + n²))
- **Compute:** Preconditioner update involves matrix multiplications, but amortized by probabilistic scheduling (drops to ~3% of steps)

---

## FSDP / Distributed Compatibility (Critical)

**The Kronecker factorization is fundamentally incompatible with FSDP gradient sharding** unless mitigated. This is the most important consideration for integrating Kron into JOLMo.

**The problem:** Under FSDP, each rank receives a shard of each parameter's gradient (e.g., shape `(m, n/W)` instead of `(m, n)`). The Kronecker factors `Q_i` are computed from gradient tensors reshaped from the **full** parameter shape. Computing them from a local shard produces mathematically different factors:

1. **Kronecker factor dimensions change:** For a `(m, n)` parameter sharded to `(m, n/W)`, `Q_1` would be `(m, m)` (correct) but `Q_2` would be `(n/W, n/W)` instead of `(n, n)`.
2. **Preconditioner updates use partial contractions:** Operations like `G @ Q_j^T @ Q_j @ G^T` on local shards give partial sums, not the full result.
3. **Einsum expressions depend on tensor shape:** Generated at init time from the local shard shape, which differs from the full shape.

**Mitigation options (choose one during implementation):**

1. **All-gather gradients for preconditioner updates only:** Before calling `_update_precond()`, gather the full gradient across FSDP ranks, compute the update on the full tensor, then discard. The preconditioned gradient application (`_precond_grad`) can still use local shards if Q factors are consistent across ranks. Cost: one all-gather per preconditioner update (amortized by probabilistic scheduling).

2. **Accept shard-local approximation:** Compute Kronecker factors from local shards. This produces per-rank approximations that may work "well enough" in practice (similar to how Muon applies Newton-Schulz to local shards). **This has NOT been empirically validated for Kron/PSGD** and should be tested carefully.

3. **Disable FSDP for Kron parameters:** Use FSDP only for data parallelism (gradient reduction) but not parameter sharding. This limits the model size that can be trained.

**Recommendation:** Start with option 2 (shard-local approximation) for simplicity, with logging to monitor preconditioner quality. Add all-gather (option 1) if convergence degrades.

---

## Key Pitfalls

1. **Vendoring kron_torch** — The helper functions are complex and tightly coupled. Vendor the entire module rather than rewriting piecemeal. Ensure proper attribution and license compliance (MIT license).

2. **Einsum expressions are generated at init** — The einsum strings depend on the tensor shape and are cached in state. If shapes change (unlikely in training), this will break.

3. **Precond update is probabilistic** — The `_update_counter` and `_prob_step` are global state (instance attributes, not per-parameter). They must be serialized for checkpoint compatibility. **Override `state_dict()` and `load_state_dict()`:**
   ```python
   def state_dict(self):
       d = super().state_dict()
       d["__kron_global__"] = {
           "_prob_step": self._prob_step,
           "_update_counter": self._update_counter,
       }
       return d

   def load_state_dict(self, state_dict):
       kron_global = state_dict.pop("__kron_global__", {})
       super().load_state_dict(state_dict)
       self._prob_step = kron_global.get("_prob_step", 0)
       self._update_counter = kron_global.get("_update_counter", 0)
   ```
   Without this, resuming from a checkpoint resets the update probability to 1.0 (max), causing a sudden compute spike.

4. **Triangular solve stability** — The inverse application through Q factors uses `torch.linalg.solve_triangular`. Ensure numerical stability by using float32 for preconditioner operations even if gradients are bf16.

5. **Memory can blow up** — For very large layers, the triangular Q factors can be huge. Use `max_size_triangular` to cap this, and `memory_save_mode='one_diag'` if needed.

6. **No bias correction on preconditioner** — Unlike Adam's v_t, the preconditioner has no bias correction. This is by design (the whitening criterion is scale-invariant).

7. **`get_local_tensor()` on state tensors after checkpoint load** — After `load_state_dict()`, state tensors (momentum_buffer, Q factors) may be reconstituted as DTensors by FSDP's distributed state dict machinery. Wrap ALL state tensor accesses with `get_local_tensor()` to handle this case, even if they were originally created as local tensors.

8. **`_ProbScheduler` uses Python `math.exp` and `max`** — These are not traceable by `torch.compile`. If `compile=True`, the probabilistic scheduling logic must either be computed outside the compiled region or rewritten with `torch` ops. Consider computing `do_update` and `balance` flags before entering the compiled parameter loop.

9. **Host-device sync in RMS clipping** — The original `kron_torch` uses `if rms > 1.1` which triggers a CUDA sync. The plan replaces this with `clip_scale = (1.1 / (rms + 1e-12)).clamp_(max=1.0); pre_grad.mul_(clip_scale)` to stay on-device. Apply the same pattern when vendoring.
