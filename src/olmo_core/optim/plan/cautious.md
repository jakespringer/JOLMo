# Cautious Optimizer (C-AdamW) Implementation Plan

## Overview

**Cautious AdamW (C-AdamW)** adds a single modification to AdamW: a sign-alignment mask that zeros out update coordinates where the momentum and current gradient disagree in sign. The mask is normalized to preserve update magnitude. This can be described as "one line of code" added to any momentum optimizer.

**Paper references:**
- Liang et al. (2024), "Cautious Optimizers: Improving Training with One Line of Code" (arXiv:2411.16085)
- Wen et al. (2025), "Fantastic Pretraining Optimizers and Where to Find Them" (Algorithm 7)

**Reference implementations:**
- `timm/optim/adamw.py` — Ross Wightman's `caution` flag in pytorch-image-models
- `kyleliang919/C-Optim` — official repository

**Complexity:** Very simple — identical state to AdamW, one extra mask computation per step.

---

## Algorithm (Pseudocode)

```
Input: params θ₀, lr η, betas (β₁, β₂), eps ε, weight_decay λ, mask_eps ξ
Init:  m₀ = 0, v₀ = 0, t = 0

for each step:
    t += 1
    g = ∇L(θ)

    # Decoupled weight decay
    θ *= (1 - η * λ)

    # Update biased first moment
    m = β₁ * m + (1 - β₁) * g

    # Update biased second moment
    v = β₂ * v + (1 - β₂) * g²

    # Bias corrections
    bc1 = 1 - β₁^t
    bc2 = 1 - β₂^t
    step_size = η / bc1

    # Denominator
    denom = sqrt(v / bc2) + ε

    # ═══ CAUTIOUS MODIFICATION (the "one line") ═══
    # Sign-alignment mask: keep only coordinates where
    # momentum and gradient agree in sign
    mask = (m * g > 0).float()
    # Normalize to preserve update magnitude
    mask /= clamp(mean(mask), min=ξ)
    # Apply mask to momentum
    m_masked = m * mask
    # ═══════════════════════════════════════════════

    # Parameter update
    θ -= step_size * m_masked / denom
```

**Key insight:** Since `denom` is always positive, `sign(m / denom) = sign(m)`. So the mask `(m * g > 0)` is equivalent to checking if the full update direction and gradient agree, which is computationally cheaper.

---

## Implementation Steps

### Step 1: Create `JOLMo/src/olmo_core/optim/cautious.py`

```python
from dataclasses import dataclass
from typing import Tuple, Type

import torch
from torch.optim.optimizer import Optimizer

from ..distributed.utils import get_local_tensor
from .config import OptimConfig


class CautiousAdamW(Optimizer):
    """
    Cautious AdamW: AdamW with sign-alignment masking.

    Zeros out update coordinates where the momentum and current gradient
    disagree in sign, with normalization to preserve update magnitude.

    Reference:
        Liang et al. (2024), "Cautious Optimizers: Improving Training
        with One Line of Code"
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
        weight_decay: float = 0.1,
        mask_eps: float = 1e-3,
    ):
        assert lr >= 0.0
        assert all(0.0 <= beta <= 1.0 for beta in betas)
        defaults = dict(
            lr=lr, betas=betas, eps=eps,
            weight_decay=weight_decay, mask_eps=mask_eps,
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
            mask_eps = group["mask_eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = get_local_tensor(p.grad)
                p_local = get_local_tensor(p)

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = torch.zeros((), dtype=torch.float32, device=p.device)
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)

                exp_avg = get_local_tensor(state["exp_avg"])
                exp_avg_sq = get_local_tensor(state["exp_avg_sq"])
                step = state["step"]
                step.add_(1)

                # Decoupled weight decay
                if weight_decay != 0:
                    p_local.mul_(1.0 - lr * weight_decay)

                # Update biased first moment
                exp_avg.lerp_(grad, 1.0 - beta1)

                # Update biased second moment
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                # Bias correction (all tensor ops to avoid host-device sync)
                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step
                step_size = lr / bias_correction1

                # Denominator
                denom = (exp_avg_sq.sqrt() / bias_correction2.sqrt()).add_(eps)

                # ═══ CAUTIOUS MASK ═══
                mask = (exp_avg * grad > 0).to(grad.dtype)
                mask.div_(mask.mean().clamp_(min=mask_eps))
                masked_exp_avg = exp_avg * mask
                # ═════════════════════

                # Parameter update (use explicit tensor ops — step_size is a tensor,
                # and value= args to addcdiv_ must be Python numbers)
                p_local.add_(-step_size * (masked_exp_avg / denom))


@dataclass
class CautiousAdamWConfig(OptimConfig):
    """
    Configuration class for building a :class:`CautiousAdamW` optimizer.
    """

    lr: float = 1e-3
    betas: Tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    weight_decay: float = 0.1
    mask_eps: float = 1e-3

    @classmethod
    def optimizer(cls) -> Type[CautiousAdamW]:
        return CautiousAdamW
```

### Step 2: Export from `__init__.py`

Add to `JOLMo/src/olmo_core/optim/__init__.py`:

```python
from .cautious import CautiousAdamW, CautiousAdamWConfig

__all__ = [
    ...
    "CautiousAdamW",
    "CautiousAdamWConfig",
]
```

### Step 3: Wire into mixture-pretraining

In `mixture-pretraining/mixture_pretraining_stages/training.py`:

1. Add to `_optimizer_class_name()`:
```python
if name == "cautious":
    return "olmo_core.optim.CautiousAdamWConfig"
```

2. `_build_optimizer_spec()` — C-AdamW uses the same fields as AdamW plus `mask_eps`. The default path works; `mask_eps` will use its default value. No `fused` flag needed.

3. Update `JolmoModel.optimizer` Literal type.

### Step 4: Write tests

Create `JOLMo/src/test/optim/cautious_test.py`:
- `test_config_builds_correctly()`
- `test_config_with_group_overrides()`
- `test_optimizer_step()` — verify parameters update
- `test_mask_zeros_disagreeing_coords()` — manually set momentum and gradient to disagree on some coords, verify those coords are zeroed in the update
- `test_mask_normalization()` — verify the mask mean normalization preserves expected magnitude

---

## Default Hyperparameters for LLM Pretraining

| Parameter | Value | Notes |
|-----------|-------|-------|
| lr | 3e-4 | Same as AdamW |
| beta1 | 0.9 | Same as AdamW |
| beta2 | 0.95 | Same as AdamW for LLMs |
| eps | 1e-8 | Standard |
| weight_decay | 0.1 | Same as AdamW |
| mask_eps | 1e-3 | Clamp floor for mask mean; fixed in all impls |

**C-AdamW introduces zero new hyperparameters that need tuning** — use identical AdamW settings.

---

## Memory & Compute Overhead

- **State per parameter:** `exp_avg`, `exp_avg_sq`, `step` — identical to AdamW (2 buffers + scalar)
- **Extra compute per step:** One element-wise comparison, one mean, one clamp, one multiply — negligible
- **Memory overhead vs AdamW:** Zero (mask is a temporary tensor)

---

## Key Pitfalls

1. **The mask is computed on biased momentum** — the timm implementation applies the mask to `exp_avg` (biased first moment), not the bias-corrected `m_hat`. This works because the sign of the momentum doesn't change with bias correction (scalar positive divisor).

2. **Mask normalization clamp** — `mask.mean().clamp_(min=1e-3)` prevents division by zero when all coordinates are masked (which shouldn't happen in practice but guards against edge cases).

3. **DTensor compatibility** — the official C-Optim repo has special handling for DTensor: `mask = (exp_avg.full_tensor() * grad.full_tensor() > 0)`. In JOLMo, using `get_local_tensor()` on both before computing the mask should suffice.

4. **The mask is a temporary** — do NOT store it in state. It's recomputed every step from the current momentum and gradient.
