# SOAP Optimizer Implementation Plan

## Overview

**SOAP** (ShampoO with Adam in the Preconditioner's eigenbasis) rotates the gradient into the eigenbasis of Shampoo's Kronecker-factored preconditioner, runs Adam-style coordinate-wise updates in that rotated space, and rotates back. Eigenbases are refreshed periodically via power iteration + QR decomposition.

**Paper references:**
- Vyas et al. (2024), "SOAP: Improving and Stabilizing Shampoo using Adam" (NeurIPS 2024)
- Wen et al. (2025), "Fantastic Pretraining Optimizers and Where to Find Them" (Algorithm 11)

**Official code:** https://github.com/nikhilvyas/SOAP

**Complexity:** High — involves Gram matrix tracking, eigenvector computation (eigh at init, power iteration + QR for periodic refresh), gradient rotation (tensordot projections), and dimension merging for >2D tensors.

---

## Algorithm (Pseudocode)

```
Input: params W ∈ R^(m×n), lr η, betas (β₁, β₂), eps ε, weight_decay λ,
       precondition_frequency f, shampoo_beta β_s, max_precond_dim D

Init: L = 0 ∈ R^(m×m), R = 0 ∈ R^(n×n)  (Gram matrices)
      M = 0 ∈ R^(m×n), V = 0 ∈ R^(m×n)   (Adam moments in rotated space)
      Q_L = I, Q_R = I                      (eigenvector matrices)

for each step t:
    G = ∇L(W)

    # 1. Project gradient into eigenbasis
    G' = Q_L^T @ G @ Q_R

    # 2. Update moments (in rotated space)
    M = β₁ * M + (1 - β₁) * G'       (first moment, maintained in rotated space)
    V = β₂ * V + (1 - β₂) * G'^2     (second moment, element-wise)

    # 3. Bias correction
    bc1 = 1 - β₁^t
    bc2 = 1 - β₂^t
    step_size = η * sqrt(bc2) / bc1

    # 4. Adam update in rotated space
    N' = M / (sqrt(V) + ε)

    # 5. Project back to original space
    N = Q_L @ N' @ Q_R^T

    # 6. Parameter update
    W -= step_size * N + η * λ * W

    # 7. Update Gram matrices
    L = β_s * L + (1 - β_s) * G @ G^T
    R = β_s * R + (1 - β_s) * G^T @ G

    # 8. Refresh eigenbases (every f steps)
    if t mod f == 0:
        # Before refresh: project M back to original space
        M_orig = Q_L @ M @ Q_R^T

        # Power iteration + QR
        Q_L, _ = QR(L @ Q_L)
        Q_R, _ = QR(R @ Q_R)

        # Reorder V to match new eigenbasis ordering
        # ... (sort by estimated eigenvalue, reorder V accordingly)

        # Re-project M into new eigenbasis
        M = Q_L^T @ M_orig @ Q_R
```

**Key details for 1D and >2D params:**
- 1D params: skip preconditioning by default (`precondition_1d=False`), use plain Adam
- >2D params: merge dimensions until each dim ≤ `max_precond_dim` before computing Gram matrices
- Dimensions > `max_precond_dim`: skip that dimension's eigenbasis (identity projection)

---

## Implementation Steps

### Step 1: Create `JOLMo/src/olmo_core/optim/soap.py`

**Strategy: Adapt the official implementation from nikhilvyas/SOAP.**

The official implementation is ~350 lines and well-structured. Key adaptations:
1. Use `get_local_tensor()` for DTensor compatibility
2. Wrap in JOLMo's `OptimConfig` pattern
3. Ensure the first-step eigenbasis initialization is handled properly

```python
import math
from dataclasses import dataclass
from itertools import chain
from typing import Dict, List, Optional, Tuple, Type

import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer

from ..distributed.utils import get_local_tensor
from .config import OptimConfig

import logging
log = logging.getLogger(__name__)


class Soap(Optimizer):
    """
    SOAP: ShampoO with Adam in the Preconditioner's eigenbasis.

    Rotates gradients into Shampoo's eigenbasis, runs Adam there,
    and rotates back. Eigenbases are refreshed every `precondition_frequency` steps
    via power iteration + QR.

    Reference:
        Vyas et al. (2024), "SOAP: Improving and Stabilizing Shampoo using Adam"
    """

    def __init__(
        self,
        params,
        lr: float = 3e-3,
        betas: Tuple[float, float] = (0.95, 0.95),
        shampoo_beta: Optional[float] = None,
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        precondition_frequency: int = 10,
        max_precond_dim: int = 10000,
        merge_dims: bool = False,
        precondition_1d: bool = False,
        correct_bias: bool = True,
    ):
        if shampoo_beta is None:
            shampoo_beta = betas[1]
        defaults = dict(
            lr=lr, betas=betas, shampoo_beta=shampoo_beta,
            eps=eps, weight_decay=weight_decay,
            precondition_frequency=precondition_frequency,
            max_precond_dim=max_precond_dim,
            merge_dims=merge_dims,
            precondition_1d=precondition_1d,
            correct_bias=correct_bias,
        )
        super().__init__(params, defaults)

    def _merge_dims(self, grad, max_precond_dim):
        """Merge dimensions of gradient tensor until each dim <= max_precond_dim."""
        shape = grad.shape
        new_shape = []
        curr_shape = 1
        for sh in shape:
            temp_shape = curr_shape * sh
            if temp_shape > max_precond_dim:
                if curr_shape > 1:
                    new_shape.append(curr_shape)
                    curr_shape = sh
                else:
                    new_shape.append(sh)
                    curr_shape = 1
            else:
                curr_shape = temp_shape
        if curr_shape > 1 or len(new_shape) == 0:
            new_shape.append(curr_shape)
        return grad.reshape(new_shape)

    def _init_preconditioner(self, grad, state, group):
        """Initialize Gram matrices for each dimension."""
        state["GG"] = []
        precondition_1d = group["precondition_1d"]
        max_precond_dim = group["max_precond_dim"]

        if grad.dim() == 1:
            if not precondition_1d or grad.shape[0] > max_precond_dim:
                state["GG"].append([])
            else:
                state["GG"].append(torch.zeros(grad.shape[0], grad.shape[0], device=grad.device))
        else:
            work_grad = self._merge_dims(grad, max_precond_dim) if group["merge_dims"] else grad
            for sh in work_grad.shape:
                if sh > max_precond_dim:
                    state["GG"].append([])
                else:
                    state["GG"].append(torch.zeros(sh, sh, device=grad.device))

        state["Q"] = None
        state["precondition_frequency"] = group["precondition_frequency"]
        state["shampoo_beta"] = group["shampoo_beta"]  # Already resolved from None → betas[1] in __init__

    def _project(self, grad, state, group):
        """Project gradient into eigenbasis: G' = Q_L^T @ G @ Q_R."""
        original_shape = grad.shape
        if group["merge_dims"]:
            grad = self._merge_dims(grad, group["max_precond_dim"])
        for mat in state["Q"]:
            if len(mat) > 0:
                grad = torch.tensordot(grad, mat, dims=[[0], [0]])
            else:
                permute_order = list(range(1, len(grad.shape))) + [0]
                grad = grad.permute(permute_order)
        if group["merge_dims"]:
            grad = grad.reshape(original_shape)
        return grad

    def _project_back(self, grad, state, group):
        """Project from eigenbasis back: N = Q_L @ N' @ Q_R^T."""
        original_shape = grad.shape
        if group["merge_dims"]:
            grad = self._merge_dims(grad, group["max_precond_dim"])
        for mat in state["Q"]:
            if len(mat) > 0:
                grad = torch.tensordot(grad, mat, dims=[[0], [1]])
            else:
                permute_order = list(range(1, len(grad.shape))) + [0]
                grad = grad.permute(permute_order)
        if group["merge_dims"]:
            grad = grad.reshape(original_shape)
        return grad

    def _update_preconditioner(self, grad, state, group):
        """Update Gram matrices and refresh eigenbases periodically."""
        max_precond_dim = group["max_precond_dim"]
        precondition_1d = group["precondition_1d"]
        merge_dims = group["merge_dims"]

        # Before updating eigenbases, project exp_avg back to original space
        # Must use get_local_tensor() since state["exp_avg"] may be a DTensor
        if state["Q"] is not None:
            local_exp_avg = get_local_tensor(state["exp_avg"])
            state["exp_avg"] = self._project_back(local_exp_avg, state, group)

        # Update Gram matrices: L = β_s*L + (1-β_s)*G@G^T, etc.
        if grad.dim() == 1:
            if precondition_1d and grad.shape[0] <= max_precond_dim:
                state["GG"][0].lerp_(grad.unsqueeze(1) @ grad.unsqueeze(0), 1 - state["shampoo_beta"])
        else:
            work_grad = self._merge_dims(grad, max_precond_dim) if merge_dims else grad
            for idx, sh in enumerate(work_grad.shape):
                if sh <= max_precond_dim:
                    dims_except_idx = list(chain(range(idx), range(idx + 1, len(work_grad.shape))))
                    outer_product = torch.tensordot(work_grad, work_grad, dims=[dims_except_idx, dims_except_idx])
                    state["GG"][idx].lerp_(outer_product, 1 - state["shampoo_beta"])

        # Compute eigenbases
        if state["Q"] is None:
            state["Q"] = self._get_eigenbasis_eigh(state["GG"])
        elif state["step"] > 0 and state["step"] % state["precondition_frequency"] == 0:
            state["Q"] = self._get_eigenbasis_qr(state, max_precond_dim, merge_dims)

        # Re-project exp_avg into (possibly updated) eigenbasis
        if state["step"] > 0:
            local_exp_avg = get_local_tensor(state["exp_avg"])
            state["exp_avg"] = self._project(local_exp_avg, state, group)

    def _get_eigenbasis_eigh(self, GG):
        """Full eigendecomposition for initial eigenbasis."""
        result = []
        for m in GG:
            if len(m) == 0:
                result.append([])
                continue
            m_float = m.float()
            try:
                _, Q = torch.linalg.eigh(m_float + 1e-30 * torch.eye(m_float.shape[0], device=m_float.device))
            except Exception:
                _, Q = torch.linalg.eigh(m_float.double() + 1e-30 * torch.eye(m_float.shape[0], device=m_float.device))
                Q = Q.float()
            Q = torch.flip(Q, [1])  # descending eigenvalue order
            result.append(Q.to(dtype=m.dtype))
        return result

    def _get_eigenbasis_qr(self, state, max_precond_dim, merge_dims):
        """Power iteration + QR for eigenbasis refresh."""
        result = []
        exp_avg_sq = get_local_tensor(state["exp_avg_sq"])
        orig_shape = exp_avg_sq.shape
        if merge_dims:
            exp_avg_sq = self._merge_dims(exp_avg_sq, max_precond_dim)

        for ind, (m, o) in enumerate(zip(state["GG"], state["Q"])):
            if len(m) == 0:
                result.append([])
                continue
            m_float = m.float()
            o_float = o.float()

            # Sort by estimated eigenvalues
            est_eig = torch.diag(o_float.T @ m_float @ o_float)
            sort_idx = torch.argsort(est_eig, descending=True)
            exp_avg_sq = exp_avg_sq.index_select(ind, sort_idx)
            o_float = o_float[:, sort_idx]

            # Power iteration + QR
            power_iter = m_float @ o_float
            Q, _ = torch.linalg.qr(power_iter)
            result.append(Q.to(dtype=m.dtype))

        if merge_dims:
            exp_avg_sq = exp_avg_sq.reshape(orig_shape)
        state["exp_avg_sq"] = exp_avg_sq
        return result

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            with torch.enable_grad():
                closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = get_local_tensor(p.grad)
                p_local = get_local_tensor(p)
                state = self.state[p]

                if "step" not in state:
                    state["step"] = torch.zeros((), dtype=torch.float32, device=p.device)

                if "exp_avg" not in state:
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)

                if "Q" not in state:
                    self._init_preconditioner(grad, state, group)
                    self._update_preconditioner(grad, state, group)
                    continue  # skip first step (need eigenbasis)

                # Project gradient into eigenbasis
                grad_projected = self._project(grad, state, group)

                exp_avg = get_local_tensor(state["exp_avg"])
                exp_avg_sq = get_local_tensor(state["exp_avg_sq"])
                beta1, beta2 = group["betas"]
                step = state["step"]
                step.add_(1)

                # Adam updates in rotated space
                exp_avg.mul_(beta1).add_(grad_projected, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).add_(grad_projected.square(), alpha=1.0 - beta2)

                denom = exp_avg_sq.sqrt().add_(group["eps"])

                step_size = group["lr"]
                if group["correct_bias"]:
                    bc1 = 1.0 - beta1 ** step
                    bc2 = 1.0 - beta2 ** step
                    # Use bc2.sqrt() (not math.sqrt) since bc2 is a tensor
                    step_size = step_size * bc2.sqrt() / bc1

                # Project back to original space
                norm_grad = self._project_back(exp_avg / denom, state, group)

                # Parameter update
                # Note: step_size may be a tensor (from bias correction or compile=True),
                # so avoid alpha= arg which must be a Python number
                p_local.sub_(step_size * norm_grad)

                # Decoupled weight decay
                if group["weight_decay"] > 0:
                    p_local.sub_(group["lr"] * group["weight_decay"] * p_local)

                # Update Gram matrices & refresh eigenbases
                self._update_preconditioner(grad, state, group)


@dataclass
class SoapConfig(OptimConfig):
    """
    Configuration class for building a :class:`Soap` optimizer.
    """

    lr: float = 3e-3
    betas: Tuple[float, float] = (0.95, 0.95)
    shampoo_beta: Optional[float] = None
    """Beta for Gram matrix EMA. If None, defaults to betas[1]."""
    eps: float = 1e-8
    weight_decay: float = 0.01
    precondition_frequency: int = 10
    max_precond_dim: int = 10000
    merge_dims: bool = False
    precondition_1d: bool = False
    correct_bias: bool = True

    @classmethod
    def optimizer(cls) -> Type[Soap]:
        return Soap

    # No custom build() needed — Soap.__init__ resolves shampoo_beta=None → betas[1]
    # and the base OptimConfig.build() passes all config fields as kwargs correctly.
```

### Step 2: Export from `__init__.py`

```python
from .soap import Soap, SoapConfig

__all__ = [
    ...
    "Soap",
    "SoapConfig",
]
```

### Step 3: Wire into mixture-pretraining

1. Add to `_optimizer_class_name()`:
```python
if name == "soap":
    return "olmo_core.optim.SoapConfig"
```

2. Add special case in `_build_optimizer_spec()`:
```python
if cls == "olmo_core.optim.SoapConfig":
    return {
        "_CLASS_": cls,
        "lr": lr,
        "betas": list(betas),
        "weight_decay": weight_decay,
        "precondition_frequency": 10,
        "group_overrides": [embedding_override],
    }
```

3. Update `JolmoModel.optimizer` Literal type.

### Step 4: Write tests

Create `JOLMo/src/test/optim/soap_test.py`:
- `test_config_builds_correctly()`
- `test_optimizer_step()` — verify parameters update (note: first step is skipped)
- `test_eigenbasis_refresh()` — run enough steps to trigger QR refresh, verify Q changes
- `test_gram_matrix_update()` — verify Gram matrices are accumulating

---

## Default Hyperparameters for LLM Pretraining

| Parameter | Value | Notes |
|-----------|-------|-------|
| lr | 3e-3 | Higher than AdamW (~10x) |
| beta1 | 0.95 | Higher than AdamW's 0.9 |
| beta2 | 0.95 | Lower than AdamW's 0.999 |
| shampoo_beta | 0.95 (= beta2) | EMA for Gram matrices |
| eps | 1e-8 | Standard |
| weight_decay | 0.01 | Lower than AdamW's 0.1 |
| precondition_frequency | 10 | At batch size 2M; use 80 at batch size 256K |
| max_precond_dim | 10000 | Skip preconditioning for huge dims |
| precondition_1d | False | Skip biases/layernorms |

---

## Memory & Compute Overhead

For a parameter tensor of shape (m x n):
- **Gram matrices:** m² + n² floats (for L and R)
- **Eigenvectors:** m² + n² floats (for Q_L and Q_R)
- **Adam states:** 2 * m * n (for M and V, same as Adam)
- **Total:** 2(m² + n²) + 3mn vs AdamW's 3mn
- **Compute per step:** O(m²n + mn²) for projection operations
- **Every f steps:** O(m³ + n³) for power iteration + QR

---

## Key Pitfalls

1. **First step is skipped** — The official implementation skips the parameter update on step 1 (only initializes Gram matrices and eigenbasis). This is important for correctness.

2. **Momentum re-projection** — When eigenbases are refreshed, `exp_avg` must be projected back to original space, eigenbasis updated, then re-projected. Forgetting this corrupts the momentum state.

3. **V reordering during QR** — `exp_avg_sq` must be reordered to match the new eigenvalue ordering after eigenbasis refresh. This ensures Adam's second moment remains consistent.

4. **The `tensordot` projection** — For each mode of the tensor, contract with Q along the appropriate axis. Forward: `dims=[[0],[0]]` (Q^T @ grad). Back: `dims=[[0],[1]]` (Q @ grad). Getting this wrong silently produces wrong results.

5. **Numerical stability** — `torch.linalg.eigh` can fail on bf16 tensors. Always cast to float32 for eigen/QR operations, then cast back.

6. **DTensor compatibility** — Use `get_local_tensor()` on `p`, `p.grad`, and ALL state tensors (`exp_avg`, `exp_avg_sq`) before arithmetic or passing to `_project`/`_project_back`. Gram matrices and eigenvectors are created as local tensors (from `grad` which is already local), so they don't need `get_local_tensor()`.

7. **FSDP Gram matrices are shard-local (critical limitation)** — Under FSDP, `grad` is a local shard (e.g., shape `(m, n/W)` instead of `(m, n)`). The Gram matrix `G @ G^T` computed from a local shard is a **partial outer product**, not the full Gram matrix. Without an `all_reduce` across FSDP ranks, each rank computes a different Gram matrix → different eigenvectors → inconsistent gradient rotations → **incorrect updates that corrupt the model**.

   **To fix:** After computing each Gram matrix outer product in `_update_preconditioner`, call `torch.distributed.all_reduce(outer_product, op=ReduceOp.SUM)` to obtain the correct full Gram matrix. This requires access to the FSDP process group (passed at `build()` time or detected from the model). The all-reduce cost is `O(m² + n²)` per parameter per step, which is small compared to the gradient computation. Note: the eigenvector computation (eigh/QR) then operates on identical Gram matrices across all ranks, producing consistent eigenvectors.

   **Alternative:** Accept shard-local Gram matrices as an approximation. For large parameter shards, the local Gram matrix is a reasonable estimate of the full one (by the law of large numbers). However, this approximation has not been empirically validated for SOAP at scale and may degrade convergence.

8. **`compile=True` and tensor `lr`** — When `compile=True`, `group["lr"]` becomes a tensor. The bias-corrected `step_size` is then also a tensor. Avoid using it as `alpha=` or `value=` arguments to in-place ops. Use explicit tensor arithmetic (e.g., `p_local.sub_(step_size * norm_grad)`).
