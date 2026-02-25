# Mars Optimizer Implementation Plan

## Overview

**MARS** (Make vARiance Reduction Shine) combines variance-reduced gradient estimation (from the STORM family) with Adam-style preconditioned updates. The key idea: before feeding the gradient into Adam's momentum/second-moment machinery, compute a corrected gradient `c_t` that subtracts stochastic noise using the difference between current and previous gradients.

**Paper references:**
- Yuan et al. (2024), "MARS: Unleashing the Power of Variance Reduction for Training Large Models" (ICML 2025)
- Wen et al. (2025), "Fantastic Pretraining Optimizers and Where to Find Them" (Algorithm 5)

**Official code:** https://github.com/AGI-Arena/MARS

**Complexity:** Moderate — requires one extra gradient buffer per parameter (`last_grad`), and the variance-reduction correction + gradient clipping.

---

## Algorithm (Pseudocode)

```
Input: params θ₀, lr η, betas (β₁, β₂), gamma γ, eps ε, weight_decay λ
Init:  m₀ = 0, v₀ = 0, g_prev = 0, t = 0

for each step:
    t += 1
    g = ∇L(θ)

    # Variance-reduced gradient correction
    c = g + γ * (β₁ / (1 - β₁)) * (g - g_prev)

    # Gradient clipping on corrected gradient
    if ||c||₂ > 1:
        c = c / ||c||₂

    # Update first moment (momentum on corrected gradient)
    m = β₁ * m + (1 - β₁) * c

    # Update second moment (on corrected gradient, NOT raw gradient)
    v = β₂ * v + (1 - β₂) * c²

    # Bias correction
    bc1 = 1 - β₁^t
    bc2 = 1 - β₂^t

    # AdamW-style update with decoupled weight decay
    denom = (sqrt(v) / sqrt(bc2) + ε) * bc1
    θ -= η * (m / denom + λ * θ)

    # Store gradient for next step (approximate mode)
    g_prev = g
```

**Key details:**
- The correction factor `γ * β₁ / (1 - β₁)` ensures proper interaction with the momentum update
- `v_t` tracks EMA of `c_t²` (corrected gradient squared), NOT `g_t²`
- The L2-norm clipping is on the corrected gradient `c_t`, not the raw gradient
- With `gamma=0`, MARS reduces to standard AdamW
- **1D parameters** (biases, layernorms) should use plain AdamW (no variance reduction) by default

---

## Implementation Steps

### Step 1: Create `JOLMo/src/olmo_core/optim/mars.py`

```python
from dataclasses import dataclass
from typing import Tuple, Type

import torch
from torch.optim.optimizer import Optimizer

from ..distributed.utils import get_local_tensor
from .config import OptimConfig


class Mars(Optimizer):
    """
    MARS: Make vARiance Reduction Shine.

    Variance-reduced AdamW variant that uses the STORM technique to
    correct gradient estimates before feeding them into Adam-style updates.

    Reference:
        Yuan et al. (2024), "MARS: Unleashing the Power of Variance
        Reduction for Training Large Models"
    """

    def __init__(
        self,
        params,
        lr: float = 3e-3,
        betas: Tuple[float, float] = (0.95, 0.99),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        gamma: float = 0.025,
        optimize_1d: bool = False,
        lr_1d_factor: float = 0.5,
        betas_1d: Tuple[float, float] = (0.9, 0.95),
        weight_decay_1d: float = 0.1,
    ):
        assert lr >= 0.0
        assert all(0.0 <= beta <= 1.0 for beta in betas)
        defaults = dict(
            lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
            gamma=gamma, optimize_1d=optimize_1d,
            lr_1d_factor=lr_1d_factor, betas_1d=betas_1d,
            weight_decay_1d=weight_decay_1d,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            with torch.enable_grad():
                closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            gamma = group["gamma"]
            optimize_1d = group["optimize_1d"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = get_local_tensor(p.grad)
                p_local = get_local_tensor(p)
                is_2d = (grad.dim() >= 2)

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = torch.zeros((), dtype=torch.float32, device=p.device)
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                    state["last_grad"] = torch.zeros_like(p)

                exp_avg = get_local_tensor(state["exp_avg"])
                exp_avg_sq = get_local_tensor(state["exp_avg_sq"])
                last_grad = get_local_tensor(state["last_grad"])
                step = state["step"]
                step.add_(1)

                if is_2d or optimize_1d:
                    # --- MARS variance-reduced correction ---
                    # c_t = g_t + gamma * (beta1 / (1 - beta1)) * (g_t - g_prev)
                    correction = gamma * (beta1 / (1.0 - beta1))
                    c_t = grad + correction * (grad - last_grad)

                    # Gradient clipping on corrected gradient
                    # Use clamp instead of Python if to avoid host-device sync
                    c_t_norm = c_t.norm().clamp_(min=1.0)
                    c_t = c_t / c_t_norm

                    # First moment update
                    exp_avg.mul_(beta1).add_(c_t, alpha=1.0 - beta1)

                    # Second moment update (on corrected gradient)
                    exp_avg_sq.mul_(beta2).addcmul_(c_t, c_t, value=1.0 - beta2)

                    # Bias correction (tensor ops to avoid host-device sync)
                    bias_correction1 = 1 - beta1 ** step
                    bias_correction2 = 1 - beta2 ** step

                    denom = (exp_avg_sq.sqrt() / bias_correction2.sqrt()).add_(eps)

                    # NOTE: MARS uses coupled weight decay (L2 regularization in the
                    # update, not decoupled). This follows the official MARS paper and
                    # implementation, but differs from JOLMo's usual decoupled pattern.
                    update = exp_avg / (denom * bias_correction1)
                    p_local.add_(update + weight_decay * p_local, alpha=-lr)
                else:
                    # Fallback: plain AdamW for 1D params (decoupled weight decay)
                    beta1_1d, beta2_1d = group["betas_1d"]
                    wd_1d = group["weight_decay_1d"]
                    lr_1d = lr * group["lr_1d_factor"]

                    # Decoupled weight decay for 1D params
                    if wd_1d != 0:
                        p_local.mul_(1.0 - lr_1d * wd_1d)

                    exp_avg.mul_(beta1_1d).add_(grad, alpha=1.0 - beta1_1d)
                    exp_avg_sq.mul_(beta2_1d).addcmul_(grad, grad, value=1.0 - beta2_1d)

                    bias_correction1 = 1 - beta1_1d ** step
                    bias_correction2 = 1 - beta2_1d ** step

                    denom = (exp_avg_sq.sqrt() / bias_correction2.sqrt()).add_(eps)

                    # Use explicit tensor ops — bias_correction1 is a tensor,
                    # and value= args to addcdiv_ must be Python numbers
                    step_size_1d = lr_1d / bias_correction1
                    p_local.add_(-(step_size_1d * exp_avg / denom))

                # Store gradient for next step
                last_grad.copy_(grad)


@dataclass
class MarsConfig(OptimConfig):
    """
    Configuration class for building a :class:`Mars` optimizer.
    """

    lr: float = 3e-3
    betas: Tuple[float, float] = (0.95, 0.99)
    eps: float = 1e-8
    weight_decay: float = 0.01
    gamma: float = 0.025
    optimize_1d: bool = False
    lr_1d_factor: float = 0.5
    betas_1d: Tuple[float, float] = (0.9, 0.95)
    weight_decay_1d: float = 0.1

    @classmethod
    def optimizer(cls) -> Type[Mars]:
        return Mars
```

### Step 2: Export from `__init__.py`

Add to `JOLMo/src/olmo_core/optim/__init__.py`:

```python
from .mars import Mars, MarsConfig

__all__ = [
    ...
    "Mars",
    "MarsConfig",
]
```

### Step 3: Wire into mixture-pretraining

In `mixture-pretraining/mixture_pretraining_stages/training.py`:

1. Add to `_optimizer_class_name()`:
```python
if name == "mars":
    return "olmo_core.optim.MarsConfig"
```

2. Update `_build_optimizer_spec()` — MARS has different default hyperparameters than AdamW. Add a special case:
```python
if cls == "olmo_core.optim.MarsConfig":
    return {
        "_CLASS_": cls,
        "lr": lr,                    # Note: MARS uses higher LR (~3e-3 to 6e-3)
        "betas": list(betas),
        "weight_decay": weight_decay,
        "gamma": 0.025,
        "group_overrides": [embedding_override],
    }
```

3. Update `JolmoModel.optimizer` Literal type.

### Step 4: Write tests

Create `JOLMo/src/test/optim/mars_test.py`:
- `test_config_builds_correctly()`
- `test_config_with_group_overrides()`
- `test_optimizer_step()` — verify parameters update
- `test_gamma_zero_matches_adamw()` — with `gamma=0`, MARS should behave like AdamW
- `test_variance_reduction_active()` — verify `last_grad` is populated after a step

---

## Default Hyperparameters for LLM Pretraining

| Parameter | Value | Notes |
|-----------|-------|-------|
| lr | 3e-3 to 6e-3 | ~10x higher than AdamW |
| beta1 | 0.95 | Higher than AdamW's 0.9 |
| beta2 | 0.99 | Lower than Adam's 0.999 |
| eps | 1e-8 | Standard |
| weight_decay | 0.01 | 10x lower than AdamW's 0.1 |
| gamma | 0.025 | Variance reduction strength; robust in [0.005, 0.025] |
| optimize_1d | False | 1D params use plain AdamW |
| lr_1d_factor | 0.5 | LR for 1D params = lr * lr_1d_factor |
| betas_1d | (0.9, 0.95) | Betas for 1D params |
| weight_decay_1d | 0.1 | Weight decay for 1D params |

**Important:** MARS uses significantly different hyperparameters than AdamW. The higher LR and lower weight decay are because the variance-reduced correction + clipping produces a more stable update direction.

---

## Memory & Compute Overhead

- **State per parameter:** `exp_avg`, `exp_avg_sq`, `last_grad`, `step` — 3 buffers + scalar (vs AdamW's 2 buffers)
- **Memory overhead vs AdamW:** +50% optimizer state (one extra buffer for `last_grad`)
- **Compute overhead:** Negligible — one subtraction, one multiply, one norm, and one conditional division per step

---

## Key Pitfalls

1. **The correction formula order matters** — `(grad - last_grad).mul_(gamma * beta1 / (1 - beta1)).add_(grad)` modifies the difference in-place. Be careful with in-place ops on grad tensors.

2. **L2 clipping is per-parameter** — each parameter's corrected gradient is independently clipped to unit norm. This is different from the global gradient clipping done by the training loop.

3. **Second moment tracks corrected gradients** — `v_t = EMA(c_t²)`, NOT `EMA(g_t²)`. This is critical for correctness.

4. **1D parameters need separate handling** — The official implementation falls back to plain AdamW for biases and layernorms (1D params) by default. This is controlled by `optimize_1d`.

5. **DTensor compatibility** — use `get_local_tensor()` on all tensors, following JOLMo conventions.
