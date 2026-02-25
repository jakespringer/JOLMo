# Sophia Optimizer Implementation Plan

## Overview

**Sophia** (Second-order Clipped Stochastic Optimization) uses momentum + periodic diagonal Hessian estimation (via Hutchinson HVP or Gauss-Newton-Bartlett) + element-wise clipping. The clipping ensures bounded worst-case updates even with noisy Hessian estimates.

**Paper references:**
- Liu et al. (2023/2024), "Sophia: A Scalable Stochastic Second-Order Optimizer for Language Model Pre-training" (ICLR 2024)
- Wen et al. (2025), "Fantastic Pretraining Optimizers and Where to Find Them" (Algorithm 4)

**Official code:** https://github.com/Liuhong99/Sophia

**Complexity:** Moderate for the optimizer itself, but **requires training loop modifications** for the Hessian estimation step (extra forward+backward pass every k steps).

---

## Algorithm (Pseudocode)

```
Input: params θ, lr η, betas (β₁, β₂), rho ρ, eps ε, weight_decay λ, k (Hessian interval)
Init:  m₀ = 0, h₀ = 0, t = 0

for each step t:
    g = ∇L(θ)

    # Gradient momentum
    m = β₁ * m + (1 - β₁) * g

    # Periodic Hessian estimation (every k steps)
    if t mod k == 0:
        if using GNB:
            # Forward pass, sample y_hat ~ softmax(logits), backward on resampled loss
            h_hat = grad_sampled²                # element-wise squared gradient
        elif using Hutchinson:
            # Forward + backward with create_graph=True
            u ~ Rademacher or N(0,I)
            Hu = ∇(⟨∇L, u⟩)                    # Hessian-vector product
            h_hat = u * Hu                       # diagonal estimate

        h = β₂ * h + (1 - β₂) * h_hat

    # Decoupled weight decay
    θ *= (1 - η * λ)

    # Clipped Hessian-preconditioned update
    #   ratio = clamp(|m| / (ρ * h + ε), max=1)
    #   θ -= η * sign(m) * ratio
    ratio = (m.abs() / (ρ * h + ε)).clamp_(max=1.0)
    θ -= η * sign(m) * ratio
```

**Key insight:** When `h` is large (high curvature), `|m| / (ρ * h)` is small and the update is like Newton's method. When `h` is small or zero (flat direction), the clipping activates at 1.0, and the update reduces to `η * sign(m)` — essentially SignSGD. This graceful fallback is what makes Sophia robust.

---

## Implementation Steps

### Step 1: Create `JOLMo/src/olmo_core/optim/sophia.py`

```python
from dataclasses import dataclass
from typing import Tuple, Type

import torch
from torch.optim.optimizer import Optimizer

from ..distributed.utils import get_local_tensor
from .config import OptimConfig


class Sophia(Optimizer):
    """
    Sophia: Second-order Clipped Stochastic Optimization.

    Uses gradient momentum + periodic diagonal Hessian estimation
    + element-wise clipping of the Hessian-preconditioned update.

    The Hessian estimation requires an external call to `update_hessian()`
    or `update_hessian_from_estimates()` every k steps from the training loop.

    Reference:
        Liu et al. (2024), "Sophia: A Scalable Stochastic Second-Order
        Optimizer for Language Model Pre-training"
    """

    def __init__(
        self,
        params,
        lr: float = 6e-4,
        betas: Tuple[float, float] = (0.965, 0.99),
        rho: float = 0.04,
        weight_decay: float = 0.1,
        eps: float = 1e-15,
        hessian_update_interval: int = 10,
    ):
        assert lr >= 0.0
        assert all(0.0 <= beta < 1.0 for beta in betas)
        assert rho > 0.0
        # Store for external use by training loop / callback
        self.hessian_update_interval = hessian_update_interval
        defaults = dict(lr=lr, betas=betas, rho=rho, weight_decay=weight_decay, eps=eps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def update_hessian(self):
        """
        Update diagonal Hessian estimate using the Gauss-Newton-Bartlett estimator.

        Call this AFTER a backward pass on the resampled-label loss.
        The gradients in p.grad should be from the GNB forward/backward pass.
        """
        for group in self.param_groups:
            _, beta2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = get_local_tensor(p.grad)
                state = self.state[p]
                self._init_state(state, p)

                hessian = get_local_tensor(state["hessian"])
                hessian.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

    @torch.no_grad()
    def update_hessian_from_estimates(self, estimates):
        """
        Update diagonal Hessian estimate from precomputed Hutchinson estimates.

        Args:
            estimates: list of tensors, one per parameter, containing u * (H @ u).
        """
        idx = 0
        for group in self.param_groups:
            _, beta2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                self._init_state(state, p)

                hessian = get_local_tensor(state["hessian"])
                h_hat = estimates[idx]
                hessian.mul_(beta2).add_(h_hat, alpha=1.0 - beta2)
                idx += 1

    def _init_state(self, state, p):
        if len(state) == 0:
            state["step"] = torch.zeros((), dtype=torch.float32, device=p.device)
            state["exp_avg"] = torch.zeros_like(p)
            state["hessian"] = torch.zeros_like(p)

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            with torch.enable_grad():
                closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, _ = group["betas"]
            rho = group["rho"]
            weight_decay = group["weight_decay"]
            eps = group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = get_local_tensor(p.grad)
                p_local = get_local_tensor(p)
                state = self.state[p]
                self._init_state(state, p)

                exp_avg = get_local_tensor(state["exp_avg"])
                hessian = get_local_tensor(state["hessian"])
                step = state["step"]
                step.add_(1)

                # Decoupled weight decay
                if weight_decay != 0:
                    p_local.mul_(1.0 - lr * weight_decay)

                # Update gradient EMA
                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)

                # Clipped Hessian-preconditioned update:
                #   ratio = min(|m| / (rho * h + eps), 1)
                #   update = sign(m) * ratio
                ratio = (exp_avg.abs() / (rho * hessian + eps)).clamp_(max=1.0)
                p_local.addcmul_(exp_avg.sign(), ratio, value=-lr)


@dataclass
class SophiaConfig(OptimConfig):
    """
    Configuration class for building a :class:`Sophia` optimizer.

    Note: Sophia requires training loop modifications to call
    ``optimizer.update_hessian()`` every ``hessian_update_interval`` steps
    after a GNB-style backward pass. See the plan document for details.

    The ``hessian_update_interval`` is passed through to the optimizer constructor
    and stored as ``optimizer.hessian_update_interval`` for the training loop / callback
    to read. This avoids needing a custom ``build()`` override.
    """

    lr: float = 6e-4
    betas: Tuple[float, float] = (0.965, 0.99)
    rho: float = 0.04
    weight_decay: float = 0.1
    eps: float = 1e-15
    hessian_update_interval: int = 10

    @classmethod
    def optimizer(cls) -> Type[Sophia]:
        return Sophia
```

### Step 2: Training Loop Integration

**This is the critical part that differs from all other optimizers.** Sophia needs a second forward+backward pass every `k` steps for Hessian estimation.

**Option A: GNB estimator (recommended, simpler)**

Modify `TransformerTrainModule.optim_step()` or add a callback:

```python
# In the training loop, every k steps AFTER the normal optim_step():
if isinstance(self.optim, Sophia) and step % self.optim.hessian_update_interval == 0:
    self.optim.zero_grad(set_to_none=True)

    # Get a batch (can reuse current or fetch new)
    X, Y = current_batch

    # Forward pass
    with torch.no_grad():
        logits = self.model(X)

    # Sample labels from model's softmax distribution
    samp_dist = torch.distributions.Categorical(logits=logits)
    y_sample = samp_dist.sample()

    # Recompute logits WITH gradients
    logits = self.model(X)
    loss_sampled = F.cross_entropy(logits.view(-1, logits.size(-1)), y_sample.view(-1))
    loss_sampled.backward()

    # Update Hessian EMA
    self.optim.update_hessian()
    self.optim.zero_grad(set_to_none=True)
```

**Option B: Implement as a callback**

Create a `SophiaHessianCallback` that hooks into `post_train_batch()`:

```python
class SophiaHessianCallback(Callback):
    def __init__(self, interval: int = 10):
        self.interval = interval

    def post_train_batch(self):
        if self.trainer.global_step % self.interval != 0:
            return
        if not isinstance(self.trainer.train_module.optim, Sophia):
            return

        # Perform GNB Hessian estimation
        # (requires access to the model and current batch)
        ...
```

**Option C: Override TransformerTrainModule (most invasive)**

Create `TransformerSophiaTrainModule` extending `TransformerTrainModule` that overrides `train_batch()` to include the Hessian estimation step.

**Recommended approach: Option A or B.** Option A is simplest for a first implementation. The Hessian update adds ~5% overhead (amortized over k=10 steps).

### Step 3: Export from `__init__.py`

```python
from .sophia import Sophia, SophiaConfig

__all__ = [
    ...
    "Sophia",
    "SophiaConfig",
]
```

### Step 4: Wire into mixture-pretraining

1. Add to `_optimizer_class_name()`:
```python
if name == "sophia":
    return "olmo_core.optim.SophiaConfig"
```

2. Add special case in `_build_optimizer_spec()`:
```python
if cls == "olmo_core.optim.SophiaConfig":
    return {
        "_CLASS_": cls,
        "lr": lr,
        "rho": 0.04,
        "weight_decay": weight_decay,
        "hessian_update_interval": 10,
        "group_overrides": [embedding_override],
    }
```

3. Update `JolmoModel.optimizer` Literal type.

### Step 5: Write tests

Create `JOLMo/src/test/optim/sophia_test.py`:
- `test_config_builds_correctly()`
- `test_optimizer_step_without_hessian()` — verify it works (degrades to sign-based updates)
- `test_hessian_update_gnb()` — manually call `update_hessian()` and verify `state["hessian"]` is populated
- `test_clipping_behavior()` — set up scenarios where clipping activates vs doesn't

---

## Default Hyperparameters for LLM Pretraining

| Parameter | 125M | 355M | 770M | Notes |
|-----------|------|------|------|-------|
| lr | 6e-4 | 7e-4 | 3e-4 | Similar to AdamW |
| beta1 | 0.965 | 0.965 | 0.965 | Higher than Adam's 0.9 |
| beta2 | 0.99 | 0.99 | 0.99 | For Hessian EMA |
| rho | 0.05 | 0.08 | 0.05 | Clipping threshold |
| weight_decay | 0.2 | 0.2 | 0.2 | Higher than AdamW's 0.1 |
| eps | 1e-15 | 1e-15 | 1e-15 | Very small for numerical stability |
| k (interval) | 10 | 10 | 10 | Hessian update frequency |
| grad_clip | 1.0 | 1.0 | 1.0 | Standard |

---

## Memory & Compute Overhead

- **State per parameter:** `exp_avg`, `hessian`, `step` — 2 buffers + scalar (same as AdamW)
- **Memory overhead vs AdamW:** Zero additional optimizer state
- **Compute overhead:** ~30% average for GNB (2 forward + 1 backward for Hessian estimation every k=10 steps, amortized: ~3x cost / 10 = 30%). The "~5%" cited elsewhere is an underestimate.
- **The Hessian estimation cost** is amortized: only 1 in every k steps does the extra pass

---

## Key Pitfalls

1. **Training loop modification required** — Unlike all other optimizers in this plan, Sophia cannot be a pure drop-in replacement. The Hessian estimation requires an extra forward+backward pass integrated into the training loop. This is the main implementation challenge.

2. **GNB estimator needs logits** — The GNB method samples labels from the model's own softmax distribution, requiring access to logits (not just the loss). This means the Hessian callback needs access to the model and batch.

3. **Batch size scaling** — In the official code, `rho * bs` appears in the denominator because the GNB estimate scales with batch size. When implementing in JOLMo, ensure proper scaling by the total batch size (tokens, not sequences).

4. **The `hessian_update_interval` is accepted by `Sophia.__init__`** — It is NOT placed in `defaults` (param groups) — it's stored as `self.hessian_update_interval` on the optimizer instance for the training loop / callback to read. This allows the base `OptimConfig.build()` to work without a custom override (the field passes through `as_dict()` → kwargs cleanly).

5. **Hessian can be negative** — Hutchinson estimates can produce negative values. The `rho * h + eps` in the denominator handles this gracefully (negative h makes the denominator small, triggering clipping). GNB always produces non-negative estimates.

6. **"Fantastic Pretraining Optimizers" finding** — Wen et al. (2025) found Sophia "does not offer significant speedup over AdamW for models under 0.5B." This is worth noting but doesn't affect implementation.
