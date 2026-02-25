# NAdamW Optimizer Implementation Plan

## Overview

**NAdamW** = NAdam (Nesterov-corrected Adam) + decoupled weight decay. The only algorithmic difference from AdamW is replacing the bias-corrected first moment `m_hat_t` in the update numerator with a Nesterov "look-ahead" estimate: `beta1 * m_hat_t + (1-beta1) * g_t / (1 - beta1^t)`.

**Paper references:**
- Dozat (2016), "Incorporating Nesterov Momentum into Adam"
- Loshchilov & Hutter (2019), "Decoupled Weight Decay Regularization"
- Wen et al. (2025), "Fantastic Pretraining Optimizers and Where to Find Them" (Algorithm 2)

**Reference implementations:**
- `torch.optim.NAdam` with `decoupled_weight_decay=True` (PyTorch 2.2+)
- `timm/optim/nadamw.py` from huggingface/pytorch-image-models (Ross Wightman)
- MLCommons AlgoPerf baseline

**Complexity:** Simple — same state as AdamW (exp_avg, exp_avg_sq, step), one additional non-in-place `.lerp()` call.

---

## Algorithm (Pseudocode)

```
Input: params θ₀, lr η, betas (β₁, β₂), eps ε, weight_decay λ
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

    # Nesterov correction: apply EMA formula AGAIN (non-in-place)
    # This computes: β₁ * m + (1 - β₁) * g  (the "look-ahead")
    m_corrected = m.lerp(g, 1 - β₁)     # ← KEY DIFFERENCE from AdamW

    # Denominator
    denom = sqrt(v / bc2) + ε

    # Update
    θ -= step_size * m_corrected / denom
```

The critical insight: `exp_avg.lerp(grad, 1 - beta1)` applied a second time (without storing back) computes `beta1 * m_t + (1 - beta1) * g_t`, which after bias correction gives the Nesterov-corrected momentum.

---

## Implementation Steps

### Step 1: Create `JOLMo/src/olmo_core/optim/nadamw.py`

```python
from dataclasses import dataclass
from typing import Tuple, Type

import torch
from torch.optim.optimizer import Optimizer

from ..distributed.utils import get_local_tensor
from .config import OptimConfig


class NAdamW(Optimizer):
    """
    NAdamW: Adam with Nesterov momentum and decoupled weight decay.

    References:
        - Dozat (2016), "Incorporating Nesterov Momentum into Adam"
        - Loshchilov & Hutter (2019), "Decoupled Weight Decay Regularization"
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ):
        assert lr >= 0.0
        assert all(0.0 <= beta <= 1.0 for beta in betas)
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
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

                # Bias corrections (all tensor ops to avoid host-device sync)
                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step

                step_size = lr / bias_correction1

                # Nesterov correction: apply EMA formula AGAIN (non-in-place)
                exp_avg_corrected = exp_avg.lerp(grad, 1.0 - beta1)

                # Denominator: sqrt(v_hat) + eps
                denom = (exp_avg_sq.sqrt() / bias_correction2.sqrt()).add_(eps)

                # Parameter update (use explicit tensor ops — step_size is a tensor,
                # and value= args to addcdiv_ must be Python numbers)
                p_local.add_(-step_size * (exp_avg_corrected / denom))


@dataclass
class NAdamWConfig(OptimConfig):
    """
    Configuration class for building a :class:`NAdamW` optimizer.
    """

    lr: float = 1e-3
    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.01

    @classmethod
    def optimizer(cls) -> Type[NAdamW]:
        return NAdamW
```

### Step 2: Export from `__init__.py`

Add to `JOLMo/src/olmo_core/optim/__init__.py`:

```python
from .nadamw import NAdamW, NAdamWConfig

__all__ = [
    ...
    "NAdamW",
    "NAdamWConfig",
]
```

### Step 3: Wire into mixture-pretraining

In `mixture-pretraining/mixture_pretraining_stages/training.py`:

1. Add to `_optimizer_class_name()`:
```python
if name == "nadamw":
    return "olmo_core.optim.NAdamWConfig"
```

2. Update `_build_optimizer_spec()` — NAdamW uses the same fields as AdamW (`lr`, `betas`, `eps`, `weight_decay`), so the existing default path works. No `fused` flag needed.

3. Update `JolmoModel.optimizer` Literal type:
```python
optimizer: Literal["adamw", "adam", "muon", "nadamw"] = "adamw"
```

### Step 4: Write tests

Create `JOLMo/src/test/optim/nadamw_test.py` following the pattern in the skill guide:
- `test_config_builds_correctly()` — verify NAdamWConfig creates NAdamW
- `test_config_with_group_overrides()` — verify param group overrides work
- `test_optimizer_step()` — verify parameters update and differ from initial
- `test_nadamw_differs_from_adamw()` — run both on same model/batch, verify updates differ (Nesterov correction is active)

---

## Default Hyperparameters for LLM Pretraining

| Parameter | Value | Notes |
|-----------|-------|-------|
| lr | 3e-4 | Same as AdamW |
| beta1 | 0.9 | Same as AdamW |
| beta2 | 0.95 | Same as AdamW for LLMs |
| eps | 1e-8 | Standard |
| weight_decay | 0.1 | Standard |
| warmup | 2000 steps | Linear warmup |
| schedule | Cosine decay | Standard |

NAdamW uses identical hyperparameters to AdamW — no tuning changes needed.

---

## Memory & Compute Overhead

- **State per parameter:** `exp_avg`, `exp_avg_sq`, `step` — identical to AdamW (2 buffers + scalar)
- **Extra compute per step:** One non-in-place `.lerp()` call (negligible)
- **Memory overhead vs AdamW:** Zero

---

## Key Pitfalls

1. **The `.lerp()` must be non-in-place** — `exp_avg.lerp(grad, 1 - beta1)` (returns new tensor) NOT `exp_avg.lerp_(grad, 1 - beta1)` (modifies exp_avg). The stored momentum `exp_avg` must remain unchanged; only the corrected version used for the update is different.

2. **DTensor compatibility** — use `get_local_tensor()` on `p`, `p.grad`, and state tensors, following the pattern in `SkipStepAdamW._step()`.

3. **Step counting** — use a tensor on device (not Python int) to avoid host-device sync, following the JOLMo convention.
