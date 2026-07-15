"""Polylogarithm cross entropy:  L_s(p) = Li_s(1 - p),  p = softmax prob of true class.

``s = 1`` recovers ordinary cross entropy exactly (``Li_1(1 - p) = -log p``) and is
dispatched to :func:`~olmo_core.nn.functional.cross_entropy_loss`'s ``F.cross_entropy``
path. For ``s in (0, 0.99]`` the polylog is evaluated with precomputed coefficients
(pure elementwise polynomial arithmetic at runtime, no special functions):

  u = 1 - p.  Two convergence regimes, split at u = 0.5:

  Branch A (u <= 0.5, i.e. easy examples, p >= 0.5):
      Li_s(u) = sum_{k>=1} u^k / k^s          -- direct series, Horner form.
      Tail error ~ u^{K+1}, so K=48 terms => ~1e-15 at the u=0.5 boundary.

  Branch B (u > 0.5, i.e. hard examples, where the direct series needs ~1/p terms):
      Li_s(e^mu) = Gamma(1-s) * (-mu)^{s-1} + sum_{k>=0} zeta(s-k) mu^k / k!,
      with mu = ln u = log1p(-p) in (-0.694, 0].  Terms shrink like (|mu|/2pi)^k
      ~ 0.11^k, so K=24 terms => ~1e-15.  Valid for all s != 1, 2, 3, ...
      The singular Gamma term IS the p^{s-1} divergence of the loss, computed in
      log space; ``p_min`` clamps the true-class probability, bounding both the
      loss and the per-example gradient weight (w <= ~Gamma(2-s) * p_min^{s-1}).

Gradient: a custom ``autograd.Function`` using the polylog ladder
    d/dz_j Li_s(1 - p_y) = w_s(p_y) * (p_j - 1[j == y]),
    w_s(p) = p * Li_{s-1}(1-p) / (1-p),   with w_s(1) = 1  (w == 1 everywhere at s=1),
so backward is one fused formula with the same structure as ``F.cross_entropy``'s.

Memory: like the ``default`` loss implementation (and unlike ``fused_linear``), this
materializes the full ``(N, vocab_size)`` logits. The forward saves only the fp32
log-probs tensor for backward (probs are recomputed there and reused in place as the
gradient buffer), so retained state matches the default CE path; per-token fp64
intermediates are ``O(N)`` and negligible.

Coefficients (~73 floats per ``s``) are plain Python numbers computed once per ``s``
(LRU-cached); the Horner loops broadcast scalars against tensors, so there are no
buffers and no device/dtype concerns. The required zeta values are computed with a
dependency-free Borwein eta-series algorithm (plus the reflection formula for
negative arguments) — no mpmath/scipy.
"""

import functools
import math
from fractions import Fraction
from typing import List, Literal, Optional, Tuple

import torch
import torch.nn.functional as F

from .cross_entropy_loss import cross_entropy_loss

__all__ = ["polylog_cross_entropy_loss"]


def _riemann_zeta(x: float) -> float:
    """zeta(x) for real x != 1, stdlib only. Borwein's eta-series algorithm for
    x > 0 (error ~ (3 + sqrt(8))^-n, far below fp64 at n=50); the functional
    equation  zeta(x) = 2^x pi^{x-1} sin(pi x/2) Gamma(1-x) zeta(1-x)  maps x < 0
    to 1 - x > 1. Init-time only (never in the training loop)."""
    if x == 1.0:
        raise ValueError("zeta has a pole at 1")
    if x == 0.0:
        return -0.5
    if x < 0.0:
        # Trivial zeros: sin(pi x / 2) is exactly 0 in exact arithmetic.
        if x == int(x) and int(x) % 2 == 0:
            return 0.0
        return (
            (2.0**x)
            * (math.pi ** (x - 1.0))
            * math.sin(math.pi * x / 2.0)
            * math.gamma(1.0 - x)
            * _riemann_zeta(1.0 - x)
        )
    n = 50
    # d_k = n * sum_{i<=k} (n+i-1)! 4^i / ((n-i)! (2i)!), exact rational arithmetic
    # (individual terms are not integers, only suitable combinations are).
    d: List[Fraction] = []
    acc = Fraction(0)
    for i in range(n + 1):
        acc += Fraction(
            math.factorial(n + i - 1) * 4**i, math.factorial(n - i) * math.factorial(2 * i)
        )
        d.append(n * acc)
    eta = (
        -sum((-1) ** k * float(d[k] - d[n]) / (k + 1) ** x for k in range(n)) / float(d[n])
    )
    return eta / (1.0 - 2.0 ** (1.0 - x))


@functools.lru_cache(maxsize=None)
def _polylog_coeffs(
    s: float, K_a: int = 48, K_b: int = 24
) -> Tuple[Tuple[Tuple[float, ...], Tuple[float, ...], float], ...]:
    """Coefficients for Li_s (the loss) and Li_{s-1} (the gradient weight):
    per order, ``(a, z, g)`` = direct-series terms ``k^-order``, expansion terms
    ``zeta(order-k)/k!``, and ``Gamma(1-order)``."""

    def pack(order: float) -> Tuple[Tuple[float, ...], Tuple[float, ...], float]:
        a = tuple(k**-order for k in range(1, K_a + 1))
        z = tuple(_riemann_zeta(order - k) / math.factorial(k) for k in range(K_b + 1))
        return a, z, math.gamma(1.0 - order)

    return pack(s), pack(s - 1.0)


def _li(
    u: torch.Tensor,
    negmu: torch.Tensor,
    a: Tuple[float, ...],
    z: Tuple[float, ...],
    g: float,
    order: float,
) -> torch.Tensor:
    """Li_order(u) for u in [0, 1), fp64 in / fp64 out. ``negmu = -ln(u)`` is
    precomputed by the caller (from log1p for accuracy near u=1)."""
    hard = u > 0.5

    # Branch A -- mask u BEFORE use so the unused branch can't emit NaN/inf.
    ua = torch.where(hard, torch.zeros_like(u), u)
    acc = torch.zeros_like(ua)
    for ck in reversed(a):  # Horner: sum a_k u^k = u*(a1 + u*(a2 + ...))
        acc = ua * (ck + acc)

    # Branch B: mu = ln u, safe value where unused.
    nm = torch.where(hard, negmu, torch.ones_like(negmu))
    mu = -nm
    poly = torch.zeros_like(mu)
    for ck in reversed(z):  # sum zeta(order-k) mu^k / k!
        poly = ck + mu * poly
    sing = g * torch.exp((order - 1.0) * torch.log(nm))
    return torch.where(hard, sing + poly, acc)


class _PolylogCE(torch.autograd.Function):
    """Per-token polylog CE. Returns ``(loss, ce)`` vectors of shape ``(N,)`` with
    exact zeros at masked positions; ``ce`` is the true (unclamped) cross entropy,
    marked non-differentiable (logging only)."""

    @staticmethod
    def forward(ctx, logits, safe_labels, mask, s, coeffs, log_pmin):
        (a_s, z_s, g_s), (a_g, z_g, g_g) = coeffs
        logp = torch.log_softmax(logits, dim=-1)
        lp_raw = logp.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
        ce = torch.where(mask, -lp_raw, torch.zeros_like(lp_raw))

        lp_y = lp_raw.clamp_min(log_pmin).double()
        p = torch.exp(lp_y)
        u = -torch.expm1(lp_y)  # 1 - p, exact near p=1
        # -mu = -log(1-p); log1p is exact down to tiny p, and the p_min clamp rules
        # out p==0. (clamp max avoids log1p(-1) = -inf when p rounds to exactly 1;
        # that value only feeds the masked-off Branch B anyway.)
        negmu = -torch.log1p(-p.clamp(max=1 - 1e-16))

        li = _li(u, negmu, a_s, z_s, g_s, s)
        loss = torch.where(mask, li.to(logits.dtype), torch.zeros_like(lp_raw))

        ctx.save_for_backward(logp, safe_labels, mask, p, u, negmu)
        ctx.meta = (s, a_g, z_g, g_g)
        ctx.mark_non_differentiable(ce)
        return loss, ce

    @staticmethod
    def backward(ctx, grad_out, _grad_ce):
        logp, safe_labels, mask, p, u, negmu = ctx.saved_tensors
        s, a_g, z_g, g_g = ctx.meta
        # w_s(p) = p * Li_{s-1}(u) / u, with the exact limit w = 1 as u -> 0
        # (Branch A's leading term is u * 1).
        li = _li(u, negmu, a_g, z_g, g_g, s - 1.0)
        w = torch.where(u > 1e-12, p * li / u.clamp_min(1e-300), torch.ones_like(u))
        w = (grad_out.double() * w).to(logp.dtype) * mask
        grad = logp.exp()  # probs, recomputed here and reused as the grad buffer
        grad.scatter_add_(
            -1,
            safe_labels.unsqueeze(-1),
            -torch.ones_like(safe_labels, dtype=logp.dtype).unsqueeze(-1),
        )
        grad.mul_(w.unsqueeze(-1))
        return grad, None, None, None, None, None


def _validate_s(s: float) -> None:
    if not (0.0 < s <= 1.0):
        raise ValueError(f"polylog 's' must be in (0, 1], got {s}")
    if 0.99 < s < 1.0:
        raise ValueError(
            f"polylog 's' must be <= 0.99 or exactly 1 (the expansion has a pole at 1), got {s}"
        )


@torch._dynamo.disable()
def polylog_cross_entropy_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    s: float,
    p_min: float = 1e-6,
    ignore_index: int = -100,
    reduction: Literal["mean", "sum", "none"] = "mean",
    compute_z_loss: bool = False,
    z_loss_multiplier: float = 1e-4,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
    """
    Polylog cross entropy ``Li_s(1 - p_true)`` with the same masking/reduction semantics
    as :func:`cross_entropy_loss`, plus a detached true-CE byproduct for logging.

    :param logits: Predicted unnormalized logits with shape ``(N, vocab_size)``.
    :param labels: Ground truth class indices with shape ``(N,)``.
    :param s: The polylog order, in ``(0, 0.99]`` or exactly ``1.0`` (== ordinary CE).
    :param p_min: Clamp on the true-class probability, bounding the ``p^(s-1)`` blowup
        of the loss and of the per-example gradient weight. Choose jointly with ``s``.
    :param ignore_index: Target value that is ignored (zero loss and zero gradient).
    :param reduction: "none", "mean" (over non-ignored targets), or "sum".
    :param compute_z_loss: Compute the softmax auxiliary loss as well.
    :param z_loss_multiplier: The multiplier to apply to the z-loss.

    :returns: ``(loss, z_loss, ce_loss)`` where ``ce_loss`` is the detached true cross
        entropy at the same reduction (identical to ``loss`` when ``s == 1``).
    """
    _validate_s(s)
    logits = logits.float()

    if s == 1.0:
        loss, z_loss = cross_entropy_loss(
            logits,
            labels,
            ignore_index=ignore_index,
            reduction=reduction,
            compute_z_loss=compute_z_loss,
            z_loss_multiplier=z_loss_multiplier,
        )
        return loss, z_loss, loss.detach()

    mask = labels != ignore_index
    safe_labels = labels.masked_fill(~mask, 0)
    loss, ce = _PolylogCE.apply(
        logits, safe_labels, mask, s, _polylog_coeffs(s), math.log(p_min)
    )
    if reduction == "mean":
        denom = mask.sum()
        loss = loss.sum() / denom
        ce = ce.sum() / denom
    elif reduction == "sum":
        loss = loss.sum()
        ce = ce.sum()
    elif reduction != "none":
        raise ValueError(reduction)

    z_loss: Optional[torch.Tensor] = None
    if compute_z_loss:
        # Mirrors cross_entropy_loss exactly (including the unmasked "none" case).
        z_squared = logits.logsumexp(-1).pow(2)
        if reduction == "mean":
            z_squared = (z_squared * mask).sum() / mask.sum()
        elif reduction == "sum":
            z_squared = (z_squared * mask).sum()
        z_loss = z_loss_multiplier * z_squared

    return loss, z_loss, ce.detach()
