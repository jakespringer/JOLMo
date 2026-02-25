"""
PSGD Kron: Preconditioned SGD with Kronecker-factored preconditioner.

Vendored and adapted from ``kron_torch`` (Evan Walters, MIT License).
https://github.com/evanatyourservice/kron_torch

Adaptations for JOLMo:
- ``get_local_tensor()`` for DTensor / FSDP compatibility
- Tensor step counters (no host-device sync)
- Deterministic balance schedule (no ``random.Random``)
- Host-device-sync-free RMS clipping
- ``state_dict`` / ``load_state_dict`` overrides for global scheduler state
"""

import logging
import math
import string
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Type

import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer

from ..distributed.utils import get_local_tensor
from .config import OptimConfig

log = logging.getLogger(__name__)


# ─── Probability scheduler ──────────────────────────────────────────────────


class _ProbScheduler:
    """Exponential anneal with flat start for preconditioner update probability."""

    def __init__(
        self,
        max_prob: float = 1.0,
        min_prob: float = 0.03,
        decay: float = 0.001,
        flat_start: int = 500,
    ):
        self.max_prob = max_prob
        self.min_prob = min_prob
        self.decay = decay
        self.flat_start = flat_start

    def __call__(self, step: int) -> float:
        prob = self.max_prob * math.exp(-self.decay * max(0, step - self.flat_start))
        return max(self.min_prob, min(self.max_prob, prob))


# ─── Einsum expression builders ─────────────────────────────────────────────

# Letters for einsum subscripts (lowercase a-z).
_LETTERS = string.ascii_lowercase


def _init_Q_exprs(
    grad: torch.Tensor,
    scale: float,
    max_size_triangular: int,
    min_ndim_triangular: int,
    memory_save_mode: Optional[str],
) -> Tuple[List[torch.Tensor], Dict[str, str]]:
    """Initialize Kronecker Q factors and pre-compute einsum expression strings.

    Returns ``(Q_list, exprs)`` where *exprs* contains cached einsum strings for
    the forward application (``"A"``), the preconditioned-gradient step (``"P"``),
    and per-factor gradient terms (``"Gk_<k>"`` for each factor *k*).
    """
    ndim = grad.dim()
    shape = grad.shape

    # Decide which dims get triangular vs diagonal factors.
    if memory_save_mode is None:
        # All dims that qualify get triangular.
        triangular = [
            sh <= max_size_triangular and ndim >= min_ndim_triangular
            for sh in shape
        ]
    elif memory_save_mode == "one_diag":
        # The largest dim is diagonal, rest triangular (if they qualify).
        max_idx = max(range(ndim), key=lambda i: shape[i])
        triangular = [
            (i != max_idx and sh <= max_size_triangular and ndim >= min_ndim_triangular)
            for i, sh in enumerate(shape)
        ]
    else:
        triangular = [False] * ndim

    # Build Q factors.
    Q: List[torch.Tensor] = []
    factor_scale = scale ** (1.0 / max(ndim, 1))
    for i, sh in enumerate(shape):
        if triangular[i]:
            # Upper-triangular factor stored as full matrix; only triu is used.
            q = torch.eye(sh, device=grad.device, dtype=grad.dtype) * factor_scale
            Q.append(q)
        else:
            # Diagonal factor stored as a vector.
            Q.append(torch.ones(sh, device=grad.device, dtype=grad.dtype) * factor_scale)

    # Build einsum expressions.
    exprs = _build_exprs(shape, triangular)
    return Q, exprs


def _build_exprs(shape: Tuple[int, ...], triangular: List[bool]) -> Dict[str, str]:
    """Build all einsum expression strings needed by the optimizer."""
    ndim = len(shape)
    # Subscripts for the gradient tensor: first `ndim` letters.
    grad_subs = list(_LETTERS[:ndim])
    # Subscripts for Q factor rows (new indices).
    q_row_subs = list(_LETTERS[ndim : 2 * ndim])

    exprs: Dict[str, str] = {}

    # --- expr "A": forward application  A = Q_0 ⊗ Q_1 ⊗ ... @ G ---
    # For triangular: Q_k has subscripts (q_row_k, grad_k).
    # For diagonal:   Q_k has subscript  (grad_k,) — element-wise multiply.
    q_inputs: List[str] = []
    out_subs = list(grad_subs)
    for k in range(ndim):
        if triangular[k]:
            q_inputs.append(f"{q_row_subs[k]}{grad_subs[k]}")
            out_subs[k] = q_row_subs[k]
        else:
            q_inputs.append(f"{grad_subs[k]}")
    lhs = ",".join(q_inputs + ["".join(grad_subs)])
    rhs = "".join(out_subs)
    exprs["A"] = f"{lhs}->{rhs}"

    # --- expr "P": preconditioned gradient  P = (Q^T Q)_kron @ G ---
    # P = Q_0^T Q_0 ⊗ Q_1^T Q_1 ⊗ ... @ G
    # This needs Q factors listed twice (once for Q^T, once for Q).
    qt_inputs: List[str] = []
    q2_inputs: List[str] = []
    p_out_subs = list(grad_subs)
    # We need extra indices for the "middle" contraction.
    mid_subs = list(_LETTERS[2 * ndim : 3 * ndim])
    for k in range(ndim):
        if triangular[k]:
            # Q_k^T has subs (grad_k, mid_k), Q_k has subs (mid_k, q_row_k).
            # Then contraction over mid_k gives (Q^T Q)_{grad_k, q_row_k}.
            qt_inputs.append(f"{mid_subs[k]}{q_row_subs[k]}")
            q2_inputs.append(f"{mid_subs[k]}{grad_subs[k]}")
            p_out_subs[k] = q_row_subs[k]
        else:
            # Diagonal: Q_k^T Q_k = diag(q_k^2), applied element-wise.
            qt_inputs.append(f"{grad_subs[k]}")
            q2_inputs.append(f"{grad_subs[k]}")
    p_lhs = ",".join(qt_inputs + q2_inputs + ["".join(grad_subs)])
    p_rhs = "".join(p_out_subs)
    exprs["P"] = f"{p_lhs}->{p_rhs}"

    # --- expr "Gk_<k>": per-factor gradient terms ---
    # For the update of Q_k, we need:
    #   term = contract A (or conjB) over all dims except k, squared.
    # Specifically: term1_k = einsum over A*A contracting all dims except k.
    a_subs = list(out_subs)  # subscripts of A (output of forward)
    for k in range(ndim):
        # Compute the outer product along dimension k of A with itself.
        # Use A with subs `a_subs` and a copy with different sub for dim k.
        extra = _LETTERS[3 * ndim + k]
        a2_subs = list(a_subs)
        a2_subs[k] = extra
        contract_lhs = "".join(a_subs) + "," + "".join(a2_subs)
        contract_rhs = a_subs[k] + extra
        exprs[f"Gk_{k}"] = f"{contract_lhs}->{contract_rhs}"

    return exprs


# ─── Core operations ────────────────────────────────────────────────────────


def _balance_Q(Q: List[torch.Tensor]) -> None:
    """Re-balance Q factors so their norms are approximately equal (geometric mean)."""
    if len(Q) < 2:
        return
    norms = []
    for q in Q:
        if q.dim() == 2:
            norms.append(q.abs().amax().clamp(min=1e-8))
        else:
            norms.append(q.abs().amax().clamp(min=1e-8))
    log_norms = torch.stack([n.log() for n in norms])
    geo_mean = log_norms.mean().exp()
    for i, q in enumerate(Q):
        q.mul_(geo_mean / norms[i])


def _update_precond(
    Q: List[torch.Tensor],
    exprs: Dict[str, str],
    grad: torch.Tensor,
    step: float,
) -> None:
    """Update Kronecker Q factors using a random probe and the whitening criterion.

    Args:
        Q: List of Q factors (triangular matrices or diagonal vectors).
        exprs: Pre-computed einsum expression strings.
        grad: The signal to use (bias-corrected momentum or raw gradient).
        step: Preconditioner learning rate (``precond_lr``).
    """
    dtype = grad.dtype
    device = grad.device

    # Random probe vector.
    V = torch.randn_like(grad)

    # Noise the signal for the whitening criterion.
    noise_scale = grad.abs().mean().sqrt().clamp(min=1e-12)
    G_noised = grad + noise_scale * V

    # Forward: A = (Q_0 ⊗ Q_1 ⊗ ...) @ G_noised
    A = _apply_Q_forward(Q, exprs, G_noised)

    # Inverse: conjB = (Q_0 ⊗ Q_1 ⊗ ...)^{-1} @ V
    conjB = _apply_Q_inverse(Q, exprs, V)

    # Update each factor.
    for k, q in enumerate(Q):
        expr_k = exprs[f"Gk_{k}"]
        term1 = torch.einsum(expr_k, A, A)       # ~ A A^T along dim k
        term2 = torch.einsum(expr_k, conjB, conjB)  # ~ B^T B along dim k

        # Normalizer: lower bound on spectral norm of (term1 + term2).
        normalizer = (term1 + term2).abs().amax().clamp(min=1e-12)

        if q.dim() == 2:
            # Triangular factor: gradient is triu(term1 - term2).
            grad_Q = torch.triu(term1 - term2)
            q.sub_((step / normalizer) * (grad_Q @ q))
        else:
            # Diagonal factor: element-wise update.
            grad_Q = torch.diag(term1 - term2) if term1.dim() == 2 else (term1 - term2)
            q.sub_((step / normalizer) * (grad_Q * q))


def _apply_Q_forward(
    Q: List[torch.Tensor],
    exprs: Dict[str, str],
    G: torch.Tensor,
) -> torch.Tensor:
    """Apply Q_0 ⊗ Q_1 ⊗ ... to G via einsum."""
    # Build operands list for the "A" expression.
    operands: List[torch.Tensor] = []
    for q in Q:
        if q.dim() == 2:
            operands.append(q)
        else:
            operands.append(q)
    operands.append(G)
    return torch.einsum(exprs["A"], *operands)


def _apply_Q_inverse(
    Q: List[torch.Tensor],
    exprs: Dict[str, str],
    V: torch.Tensor,
) -> torch.Tensor:
    """Apply (Q_0 ⊗ Q_1 ⊗ ...)^{-1} to V via sequential triangular solves.

    For Kronecker products: (A ⊗ B)^{-1} = A^{-1} ⊗ B^{-1}.
    For triangular Q_k: solve Q_k @ X = V along dimension k.
    For diagonal Q_k: divide by Q_k element-wise along dimension k.
    """
    result = V.clone()
    ndim = V.dim()
    for k, q in enumerate(Q):
        if q.dim() == 2:
            # Triangular solve along dimension k.
            # Move dim k to the last position, solve, move back.
            result = result.movedim(k, -1).to(torch.float32)
            q_f32 = q.to(torch.float32)
            # solve_triangular: Q_k @ X = result  =>  X = Q_k^{-1} @ result
            result = torch.linalg.solve_triangular(
                q_f32.unsqueeze(0).expand(result.shape[:-1] + q_f32.shape[:1]),
                result.unsqueeze(-1),
                upper=True,
            ).squeeze(-1)
            result = result.to(V.dtype).movedim(-1, k)
        else:
            # Diagonal: divide element-wise.
            # Expand q to broadcast along dimension k.
            shape = [1] * ndim
            shape[k] = q.shape[0]
            result = result / q.view(shape).clamp(min=1e-12)
    return result


def _precond_grad(
    Q: List[torch.Tensor],
    exprs: Dict[str, str],
    grad: torch.Tensor,
) -> torch.Tensor:
    """Apply the full preconditioner P = (Q^T Q)_kron to the gradient."""
    operands: List[torch.Tensor] = []
    # First pass: Q^T factors.
    for q in Q:
        operands.append(q)
    # Second pass: Q factors.
    for q in Q:
        operands.append(q)
    operands.append(grad)
    return torch.einsum(exprs["P"], *operands)


# ─── Dimension merging ──────────────────────────────────────────────────────


def _merge_dims(shape: Tuple[int, ...], max_size: int = 8192) -> Optional[Tuple[int, ...]]:
    """Merge adjacent dimensions of >2D tensors to get a 2D (or smaller) shape.

    Returns ``None`` if no merging is needed (already ≤ 2D).
    """
    if len(shape) <= 2:
        return None
    new_shape: List[int] = []
    curr = 1
    for s in shape:
        if curr * s <= max_size:
            curr *= s
        else:
            if curr > 1:
                new_shape.append(curr)
            curr = s
    if curr > 1 or len(new_shape) == 0:
        new_shape.append(curr)
    merged = tuple(new_shape)
    if merged == shape:
        return None
    return merged


# ─── Optimizer ───────────────────────────────────────────────────────────────


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
            lr=lr,
            b1=b1,
            weight_decay=weight_decay,
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
            precond_update_prob_max,
            precond_update_prob_min,
            precond_update_prob_decay,
            precond_update_prob_flat_start,
        )
        self._prob_step = 0
        self._update_counter = 0.0

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
        self._update_counter = kron_global.get("_update_counter", 0.0)

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            with torch.enable_grad():
                closure()

        # Determine if preconditioner update happens this step.
        update_prob = self._prob_scheduler(self._prob_step)
        self._update_counter += 1.0
        do_update = self._update_counter >= 1.0 / update_prob
        if do_update:
            self._update_counter = 0.0
        self._prob_step += 1
        # Deterministic balance: every 100 preconditioner updates.
        balance = do_update and self._prob_step % 100 == 0

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = get_local_tensor(p.grad)
                p_local = get_local_tensor(p)
                state = self.state[p]

                # State initialization.
                if len(state) == 0:
                    state["step"] = torch.zeros((), dtype=torch.float32, device=p.device)
                    # Merge dims for >2D tensors.
                    merged = _merge_dims(grad.shape) if group["merge_dims"] else None
                    if merged is not None:
                        state["merged_shape"] = merged
                    work_grad = grad.view(*merged) if merged else grad
                    state["momentum_buffer"] = torch.zeros_like(work_grad)
                    state["Q"], state["exprs"] = _init_Q_exprs(
                        work_grad,
                        group["precond_init_scale"],
                        group["max_size_triangular"],
                        group["min_ndim_triangular"],
                        group["memory_save_mode"],
                    )

                step_t = state["step"]
                step_t.add_(1)

                # Reshape if needed.
                merged_shape = state.get("merged_shape")
                work_grad = grad.view(*merged_shape) if merged_shape is not None else grad

                # Momentum with bias correction.
                mu = get_local_tensor(state["momentum_buffer"])
                b1 = group["b1"]
                mu.mul_(b1).add_(work_grad, alpha=1.0 - b1)
                mu_hat = mu / (1.0 - b1 ** step_t)

                Q = state["Q"]
                Q = [get_local_tensor(q) for q in Q]
                state["Q"] = Q

                # Balance Q factors.
                if work_grad.dim() > 1 and balance:
                    _balance_Q(Q)

                # Update preconditioner.
                if do_update:
                    signal = mu_hat if group["momentum_into_precond_update"] else work_grad
                    _update_precond(Q, state["exprs"], signal, step=group["precond_lr"])

                # Precondition gradient.
                pre_grad = _precond_grad(Q, state["exprs"], mu_hat)

                # Clip update RMS (no host-device sync).
                rms = pre_grad.pow(2).mean().sqrt()
                clip_scale = (1.1 / (rms + 1e-12)).clamp_(max=1.0)
                pre_grad.mul_(clip_scale)

                # Weight decay (decoupled, only for dim >= 2).
                if group["weight_decay"] != 0 and work_grad.dim() >= 2:
                    pre_grad.add_(p_local.view(pre_grad.shape), alpha=group["weight_decay"])

                # Parameter update.
                p_local.add_(pre_grad.view(p_local.shape), alpha=-group["lr"])


# ─── Config ──────────────────────────────────────────────────────────────────


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
