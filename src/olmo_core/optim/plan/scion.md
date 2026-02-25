# Scion Optimizer Implementation Plan

## Overview

**Scion** (Stochastic Conditional descent with Operator Norms) is a Frank-Wolfe-style optimizer that uses operator-norm-based Linear Minimization Oracles (LMOs) to produce update directions. For 2D weight matrices it uses Newton-Schulz orthogonalization (same as Muon); for embeddings/output it uses sign normalization; for 1D params it uses RMS normalization.

**Paper references:**
- Pethick et al. (2025), "Training Deep Learning Models with Norm-Constrained LMOs" (arXiv:2502.07529)
- Wen et al. (2025), "Fantastic Pretraining Optimizers and Where to Find Them" (Algorithm 9)

**Official code:** https://github.com/LIONS-EPFL/scion

**Complexity:** High — requires a composite optimizer (like Muon) with different update rules per parameter type, Newton-Schulz iteration for matrix params, and a custom `build()` method to classify parameters. Also requires norm-aware initialization.

---

## Algorithm (Pseudocode)

```
Input: params {θᵢ}, lr γ, momentum β ∈ (0,1], per-layer scales {Rᵢ}, per-layer norms
Init:  mᵢ = 0 for all layers

for each step:
    g = ∇L(θ)

    for each parameter group i:
        # 1. Momentum update (EMA of gradients)
        mᵢ = (1 - β) * mᵢ + β * gᵢ

        # 2. Compute LMO direction based on norm type
        if norm == Spectral (2D hidden weight):
            dᵢ = NewtonSchulz5(mᵢ.reshape(rows, cols), steps=5)
            dᵢ *= sqrt(d_out / d_in)           # scaling factor
        elif norm == Sign (embedding/output):
            dᵢ = sign(mᵢ) / fan_in
        elif norm == BiasRMS (1D params):
            dᵢ = mᵢ / RMS(mᵢ)

        # 3. Constrained Frank-Wolfe step
        θᵢ = (1 - γ) * θᵢ - γ * Rᵢ * dᵢ

function NewtonSchulz5(G, steps=5):
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.bfloat16()
    if rows > cols: X = X.T
    X = X / (||X||_F + 1e-7)
    for i in 1..steps:
        A = X @ X.T
        B = b*A + c*A@A
        X = a*X + B@X
    if rows > cols: X = X.T
    return X
```

**Key differences from Muon:**
- Scion uses a constrained update: `(1 - γ) * θ + ...` which shrinks weights toward zero (Frank-Wolfe)
- Scion applies norm-specific LMOs to ALL parameter types (not just hidden 2D)
- Scion uses a `scale` parameter per group instead of an adaptive learning rate
- Scion does NOT use Adam-style second moment — it's purely momentum + projection

---

## Implementation Steps

### Step 1: Create `JOLMo/src/olmo_core/optim/scion.py`

This follows the composite optimizer pattern from `muon.py` — classify params into groups, apply different update rules per group.

```python
import math
from dataclasses import dataclass
from fnmatch import fnmatch
from typing import Any, Dict, List, Optional, Set, Tuple, Type, cast

import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer

from ..distributed.utils import get_local_tensor
from ..utils import move_to_device
from .config import INITIAL_LR_FIELD, LR_FIELD, OptimConfig

import logging
log = logging.getLogger(__name__)


def _newton_schulz5(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """
    Newton-Schulz iteration to approximate the orthogonal polar factor UV^T.
    """
    assert G.dim() == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(0) > G.size(1):
        X = X.T
    X = X / (X.norm() + 1e-7)
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X


class Scion(Optimizer):
    """
    Scion optimizer: Frank-Wolfe style with operator-norm LMOs.

    Uses Newton-Schulz orthogonalization for 2D hidden weight matrices,
    sign normalization for embeddings/output, and RMS normalization for 1D params.

    Each parameter group must have:
        _norm: str   -- "spectral", "sign", or "bias_rms"
        scale: float -- constraint radius R for this group
    """

    def __init__(
        self,
        params,
        *,
        lr: float = 2.44e-4,
        momentum: float = 0.1,
        ns_steps: int = 5,
    ):
        defaults: Dict[str, Any] = dict(lr=lr, momentum=momentum, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            ns_steps = group.get("ns_steps", 5)
            norm_type = group.get("_norm", "spectral")
            scale = group.get("scale", 1.0)

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = get_local_tensor(p.grad)
                p_local = get_local_tensor(p)
                state = self.state[p]

                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(p)

                buf = get_local_tensor(state["momentum_buffer"])

                # Momentum (EMA): buf = (1 - momentum) * buf + momentum * grad
                buf.mul_(1.0 - momentum).add_(grad, alpha=momentum)

                # Compute LMO direction based on norm type
                if norm_type == "spectral":
                    d = _newton_schulz5(buf.reshape(buf.size(0), -1), steps=ns_steps)
                    d = d.view(buf.shape)
                    d_out, d_in = buf.size(0), buf.numel() // buf.size(0)
                    d.mul_(math.sqrt(d_out / d_in))
                elif norm_type == "sign":
                    d_in = buf.size(-1)
                    d = torch.sign(buf) / d_in
                elif norm_type == "bias_rms":
                    rms = torch.sqrt(buf.pow(2).mean() + 1e-8)
                    d = buf / rms
                else:
                    raise ValueError(f"Unknown norm type: {norm_type}")

                # Constrained Frank-Wolfe step:
                # theta = (1 - lr) * theta - lr * scale * d
                # Note: lr may be a tensor when compile=True, so avoid alpha= arg
                p_local.mul_(1.0 - lr)
                p_local.sub_(lr * (scale * d))

        return loss


@dataclass
class ScionConfig(OptimConfig):
    """
    Configuration class for building a composite :class:`Scion` optimizer.

    Uses Scion (Newton-Schulz) for 2D hidden weight matrices,
    sign normalization for embeddings/output, and RMS normalization for 1D params.
    """

    lr: float = 2.44e-4
    momentum: float = 0.1
    ns_steps: int = 5
    spectral_scale: float = 50.0
    sign_scale: float = 3000.0
    bias_rms_scale: float = 50.0

    @classmethod
    def optimizer(cls) -> Type[Scion]:
        return Scion

    def _classify_param(self, name: str, param: torch.Tensor) -> str:
        """
        Classify parameter into norm type.

        Rules:
        1. Non-2D params → bias_rms (norms, biases, scalars)
        2. Embeddings → sign
        3. LM head → sign
        4. All other 2D params → spectral
        """
        if param.ndim < 2:
            return "bias_rms"
        if fnmatch(name, "embeddings.*"):
            return "sign"
        if fnmatch(name, "lm_head.*"):
            return "sign"
        return "spectral"

    def build(self, model: nn.Module, strict: bool = True) -> Scion:
        """
        Build the Scion optimizer with norm-specific parameter groups.

        Follows the MuonConfig.build() pattern: first call build_groups()
        to handle group_overrides, then split each group by norm type.
        """
        # Build a set of param IDs for each norm type
        norm_type_by_id: Dict[int, str] = {}
        for name, p in model.named_parameters():
            if p.requires_grad:
                norm_type_by_id[id(p)] = self._classify_param(name, p)

        scale_map = {
            "spectral": self.spectral_scale,
            "sign": self.sign_scale,
            "bias_rms": self.bias_rms_scale,
        }

        # Get base param groups (handles group_overrides)
        base_groups = self.build_groups(model, strict=strict)

        # Split each base group by norm type
        final_groups: List[Dict[str, Any]] = []
        for group in base_groups:
            params = group.pop("params")
            group_opts = dict(group)  # remaining opts from overrides

            # Partition params by norm type
            by_norm: Dict[str, List[torch.Tensor]] = {}
            for p in params:
                nt = norm_type_by_id.get(id(p), "bias_rms")
                by_norm.setdefault(nt, []).append(p)

            for norm_type, norm_params in by_norm.items():
                g: Dict[str, Any] = {"params": norm_params, "_norm": norm_type}
                g.update(group_opts)
                g.setdefault("scale", scale_map[norm_type])
                g.setdefault("lr", self.lr)
                g.setdefault("momentum", self.momentum)
                g.setdefault("ns_steps", self.ns_steps)
                final_groups.append(g)

        if not final_groups:
            raise ValueError("No trainable parameters found when building Scion optimizer.")

        # Create optimizer
        optim = Scion(final_groups, lr=self.lr, momentum=self.momentum, ns_steps=self.ns_steps)

        # Set initial_lr on each group
        fixed_fields_per_group: List[Dict[str, Any]] = [{} for _ in optim.param_groups]
        for fixed_fields, group in zip(fixed_fields_per_group, optim.param_groups):
            lr = group.get(LR_FIELD)
            if lr is not None:
                if self.compile:
                    group[LR_FIELD] = move_to_device(torch.tensor(lr), self.device)
                else:
                    group[LR_FIELD] = lr
                group.setdefault(INITIAL_LR_FIELD, lr)
            for k in self.fixed_fields:
                if k in group:
                    fixed_fields[k] = group[k]

        # Log group info
        log.info(
            f"Building {self.optimizer().__name__} optimizer with "
            f"{len(optim.param_groups)} param group(s)..."
        )
        for g_idx, group in enumerate(optim.param_groups):
            norm_type = group.get("_norm", "unknown")
            group_fields_list = "\n - ".join(
                [f"{k}: {v}" for k, v in group.items() if k != "params"]
            )
            log.info(
                f"Group {g_idx} ({norm_type}), {len(group['params'])} parameter(s):"
                f"\n - {group_fields_list}"
            )

        if self.compile:
            log.info("Compiling optimizer step...")
            optim.step = torch.compile(optim.step)

        # Register hook to reset fixed fields after checkpoint load
        def reset_fixed_fields(opt: torch.optim.Optimizer):
            for ff, group in zip(fixed_fields_per_group, opt.param_groups):
                group.update(ff)

        optim.register_load_state_dict_post_hook(reset_fixed_fields)

        return cast(Scion, optim)
```

### Step 2: Export from `__init__.py`

```python
from .scion import Scion, ScionConfig

__all__ = [
    ...
    "Scion",
    "ScionConfig",
]
```

### Step 3: Wire into mixture-pretraining

1. Add to `_optimizer_class_name()`:
```python
if name == "scion":
    return "olmo_core.optim.ScionConfig"
```

2. Add special case in `_build_optimizer_spec()` since Scion has different fields:
```python
if cls == "olmo_core.optim.ScionConfig":
    return {
        "_CLASS_": cls,
        "lr": lr,
        "momentum": 0.1,
        "spectral_scale": 50.0,
        "sign_scale": 3000.0,
        "bias_rms_scale": 50.0,
        "group_overrides": [embedding_override],
    }
```

3. Update `JolmoModel.optimizer` Literal type.

### Step 4: Write tests

Create `JOLMo/src/test/optim/scion_test.py`:
- `test_config_builds_correctly()`
- `test_param_classification()` — verify embedding/lm_head/hidden params are classified correctly
- `test_optimizer_step()` — verify parameters update
- `test_newton_schulz_orthogonality()` — verify the NS iteration produces approximately orthogonal output

---

## Default Hyperparameters for LLM Pretraining

| Parameter | Value | Notes |
|-----------|-------|-------|
| lr | 2^-12 ≈ 2.44e-4 | Learning rate (gamma in paper) |
| momentum | 0.1 | EMA parameter (1 - traditional momentum 0.9) |
| spectral_scale | 50.0 | Constraint radius for hidden layers |
| sign_scale | 3000.0 | Constraint radius for embedding/output (50 * 60) |
| bias_rms_scale | 50.0 | Constraint radius for bias/norm params |
| ns_steps | 5 | Newton-Schulz iterations |

---

## Memory & Compute Overhead

- **State per parameter:** `momentum_buffer` — 1 buffer (vs AdamW's 2 buffers). Lower memory!
- **Compute per step:** Newton-Schulz iteration (5 matrix multiplications) per 2D param — similar to Muon
- **Memory overhead vs AdamW:** ~50% less optimizer state (no second moment)

---

## Key Pitfalls

1. **Constrained vs unconstrained** — The Frank-Wolfe `(1 - lr)` weight shrinkage is essential. Without it, the optimizer can diverge. This acts as implicit weight decay.

2. **No separate weight_decay parameter** — Scion's constrained formulation provides its own regularization through the `(1 - lr)` shrinkage. Do NOT add separate weight decay on top (or if you do, be very careful about double-regularization).

3. **Custom `build()` required** — Like Muon, Scion needs a custom `build()` to classify parameters into norm groups. Follow the `MuonConfig.build()` pattern.

4. **Newton-Schulz requires CUDA** — The `@torch.compile` decorator and bfloat16 operations require a CUDA device. Add appropriate guards.

5. **Scale parameter is critical** — The `scale` (constraint radius R) must be tuned per parameter type. The defaults (50.0 for hidden, 3000.0 for output) are from the nanoGPT experiments.

6. **Norm-aware initialization** — The original Scion paper recommends special initialization (orthogonal for spectral params, sign for sign params). This may need a separate initialization step or can be handled externally. For now, use standard initialization and note this as a future improvement.

7. **FSDP shard approximation** — Under FSDP2, parameters are sharded and `get_local_tensor()` returns a shard (e.g., `(m, n/W)` instead of `(m, n)`). The Newton-Schulz iteration computes the polar factor of the **local shard**, not a shard of the polar factor of the full matrix. This is a known approximation — PyTorch's built-in `torch.optim.Muon` uses the same approach. The approximation works well in practice because the Newton-Schulz iteration is applied to normalized matrices and the per-shard result is a reasonable surrogate. No all-reduce is needed for the Newton-Schulz computation.

8. **`compile=True` and `lr` as tensor** — When `compile=True`, the base `build()` converts `group["lr"]` to a tensor. The step function must avoid using `lr` in `alpha=` or `value=` arguments to in-place ops (which require Python numbers). Use explicit tensor arithmetic instead (e.g., `p_local.sub_(lr * x)` rather than `p_local.add_(x, alpha=-lr)`).
