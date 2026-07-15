import math

import pytest
import torch
import torch.nn.functional as F

from olmo_core.nn.functional import cross_entropy_loss, polylog_cross_entropy_loss
from olmo_core.nn.functional.polylog_cross_entropy_loss import (
    _polylog_coeffs,
    _PolylogCE,
    _riemann_zeta,
)
from olmo_core.testing import DEVICES


def test_riemann_zeta():
    torch.testing.assert_close(_riemann_zeta(2.0), math.pi**2 / 6, rtol=1e-12, atol=0.0)
    torch.testing.assert_close(_riemann_zeta(0.0), -0.5, rtol=0.0, atol=0.0)
    torch.testing.assert_close(_riemann_zeta(-1.0), -1.0 / 12.0, rtol=1e-12, atol=0.0)
    assert _riemann_zeta(-2.0) == 0.0
    # The regime the coefficients actually use: zeta(s - k) for s in (0, 1), k = 0..24.
    torch.testing.assert_close(_riemann_zeta(0.5), -1.4603545088095868, rtol=1e-12, atol=0.0)
    mpmath = pytest.importorskip("mpmath")
    for x in (0.5, 0.99, -0.5, -3.5, -23.5, 26.0):
        torch.testing.assert_close(
            _riemann_zeta(x), float(mpmath.zeta(x)), rtol=1e-10, atol=1e-300
        )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("reduction", ["sum", "mean", "none"])
def test_s_equals_one_matches_cross_entropy(device, reduction):
    vocab_size = 512
    N = 64
    logits = torch.randn(N, vocab_size, device=device)
    labels = torch.randint(0, vocab_size, (N,), device=device)
    labels[::7] = -100

    loss, z_loss, ce_loss = polylog_cross_entropy_loss(
        logits, labels, s=1.0, reduction=reduction, compute_z_loss=True
    )
    ref_loss, ref_z = cross_entropy_loss(
        logits, labels, reduction=reduction, compute_z_loss=True
    )
    torch.testing.assert_close(loss, ref_loss)
    torch.testing.assert_close(z_loss, ref_z)
    torch.testing.assert_close(ce_loss, ref_loss)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("s", [0.5, 0.99])
@pytest.mark.parametrize("reduction", ["sum", "mean", "none"])
def test_masking_and_ce_byproduct(device, s, reduction):
    vocab_size = 512
    N = 64
    logits = torch.randn(N, vocab_size, device=device, requires_grad=True)
    labels = torch.randint(0, vocab_size, (N,), device=device)

    loss, z_loss, ce_loss = polylog_cross_entropy_loss(
        logits, labels, s=s, reduction=reduction, compute_z_loss=True
    )
    ref_ce, ref_z = cross_entropy_loss(logits, labels, reduction=reduction, compute_z_loss=True)
    # The ce byproduct is the true cross entropy, detached.
    assert not ce_loss.requires_grad
    torch.testing.assert_close(ce_loss, ref_ce.detach())
    torch.testing.assert_close(z_loss, ref_z)

    # Adding fully-masked rows changes nothing (and their grads are exactly zero).
    logits_padded = torch.cat([logits.detach(), torch.randn(3, vocab_size, device=device)])
    logits_padded.requires_grad_(True)
    labels_padded = torch.cat([labels, torch.tensor([-100] * 3, device=device)])
    loss1, _, ce1 = polylog_cross_entropy_loss(
        logits_padded, labels_padded, s=s, reduction=reduction
    )
    if reduction == "none":
        torch.testing.assert_close(loss1[:N], loss)
        assert (loss1[N:] == 0).all() and (ce1[N:] == 0).all()
        loss1.sum().backward()
    else:
        torch.testing.assert_close(loss1, loss)
        torch.testing.assert_close(ce1, ce_loss)
        loss1.backward()
    assert logits_padded.grad is not None
    assert (logits_padded.grad[N:] == 0).all()
    assert logits_padded.grad[:N].abs().sum() > 0


@pytest.mark.parametrize("s", [0.5, 0.9])
def test_gradcheck(s):
    torch.manual_seed(0)
    logits = torch.randn(8, 16, dtype=torch.float64, requires_grad=True)
    labels = torch.randint(0, 16, (8,))
    mask = labels != -100
    coeffs = _polylog_coeffs(s)

    def fn(x):
        loss, _ = _PolylogCE.apply(x, labels, mask, s, coeffs, math.log(1e-6))
        return loss

    assert torch.autograd.gradcheck(fn, (logits,))


@pytest.mark.parametrize("s", [0.5, 0.75, 0.99])
def test_values_match_mpmath_polylog(s):
    mpmath = pytest.importorskip("mpmath")
    torch.manual_seed(0)
    # Spread true-class probabilities across both branches (p >= 0.5 and p < 0.5,
    # including near the p_min clamp).
    logits = torch.zeros(7, 4, dtype=torch.float64)
    logits[:, 0] = torch.tensor([8.0, 2.0, 0.5, 0.0, -2.0, -8.0, -20.0])
    labels = torch.zeros(7, dtype=torch.long)

    loss, _, _ = polylog_cross_entropy_loss(logits, labels, s=s, reduction="none")

    p_min = 1e-6
    logp = torch.log_softmax(logits.float(), dim=-1)[:, 0].double().clamp_min(math.log(p_min))
    for i in range(7):
        expected = float(mpmath.polylog(s, 1.0 - math.exp(float(logp[i]))))
        torch.testing.assert_close(float(loss[i]), expected, rtol=1e-6, atol=1e-9)


def test_gradient_weight_formula():
    # d/dz Li_s(1-p_y) = w_s(p_y) * (softmax - onehot), w_s(p) = p Li_{s-1}(1-p)/(1-p).
    mpmath = pytest.importorskip("mpmath")
    s = 0.5
    logits = torch.tensor([[2.0, 0.0, -1.0]], dtype=torch.float64, requires_grad=True)
    labels = torch.tensor([0])
    loss, _, _ = polylog_cross_entropy_loss(logits, labels, s=s, reduction="sum")
    loss.backward()

    probs = torch.softmax(logits.detach().float(), dim=-1).double()
    p = float(probs[0, 0])
    w = p * float(mpmath.polylog(s - 1.0, 1.0 - p)) / (1.0 - p)
    onehot = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64)
    torch.testing.assert_close(
        logits.grad.double(), w * (probs - onehot), rtol=1e-5, atol=1e-8
    )


def test_invalid_s():
    logits = torch.randn(4, 8)
    labels = torch.randint(0, 8, (4,))
    for s in (0.0, -0.5, 1.5, 0.995):
        with pytest.raises(ValueError):
            polylog_cross_entropy_loss(logits, labels, s=s)
