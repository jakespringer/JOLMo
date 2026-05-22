"""
Tests for :class:`RandomGradientMaskConfig`.

Determinism and selection-filter behavior. World-size-invariance is
implicitly exercised by the determinism tests since the global mask is
generated identically; the FSDP-shard slicing path is tested separately
when distributed infrastructure is available.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from olmo_core.nn.peft import RandomGradientMaskConfig
from olmo_core.nn.peft.base import stable_hash_u64


def _toy_model() -> nn.Module:
    return nn.Sequential(
        nn.Linear(8, 16),
        nn.ReLU(),
        nn.Linear(16, 4),
    )


def _setup_registry(model, cfg):
    cfg.pre_train(model)
    return cfg._registry


def test_random_mask_pre_train_creates_registry():
    model = _toy_model()
    cfg = RandomGradientMaskConfig(fraction_trainable=0.5, seed=42)
    registry = _setup_registry(model, cfg)

    # Linear weights should be in the registry.
    keys = set(registry.keys())
    assert "0.weight" in keys
    assert "2.weight" in keys
    # Biases are not (default target_param_kinds=["weight"]).
    assert "0.bias" not in keys
    assert "2.bias" not in keys


def test_random_mask_realized_fraction_is_close_to_target():
    # Larger tensor → tighter concentration around the target fraction.
    model = nn.Linear(256, 256)
    cfg = RandomGradientMaskConfig(fraction_trainable=0.3, seed=1)
    cfg.pre_train(model)
    mask = cfg._registry["weight"]
    realized = mask.float().mean().item()
    assert abs(realized - 0.3) < 0.02


def test_random_mask_is_deterministic_under_same_seed():
    model_a = _toy_model()
    model_b = _toy_model()
    cfg_a = RandomGradientMaskConfig(fraction_trainable=0.4, seed=42)
    cfg_b = RandomGradientMaskConfig(fraction_trainable=0.4, seed=42)
    cfg_a.pre_train(model_a)
    cfg_b.pre_train(model_b)
    for key in cfg_a._registry.keys():
        ma = cfg_a._registry[key]
        mb = cfg_b._registry[key]
        assert torch.equal(ma, mb), f"mismatch on {key}"


def test_random_mask_changes_with_seed():
    model = _toy_model()
    cfg_a = RandomGradientMaskConfig(fraction_trainable=0.4, seed=1)
    cfg_b = RandomGradientMaskConfig(fraction_trainable=0.4, seed=2)
    cfg_a.pre_train(_toy_model())
    cfg_b.pre_train(_toy_model())
    for key in cfg_a._registry.keys():
        assert not torch.equal(
            cfg_a._registry[key], cfg_b._registry[key]
        ), f"unexpected equality on {key}"


def test_random_mask_zeroes_gradients_at_mask_zero_positions():
    model = nn.Linear(8, 4, bias=False)
    cfg = RandomGradientMaskConfig(fraction_trainable=0.5, seed=1)
    cfg.pre_train(model)

    # Synthesize a gradient of all ones; after transform, only mask=1
    # positions retain the value, mask=0 positions become 0.
    model.weight.grad = torch.ones_like(model.weight)
    cfg.transform(model)

    mask = cfg._registry["weight"]
    expected = mask.to(model.weight.grad.dtype)
    assert torch.equal(model.weight.grad, expected)


def test_random_mask_skips_frozen_parameters():
    model = _toy_model()
    # Freeze the first linear's weight.
    model[0].weight.requires_grad = False

    cfg = RandomGradientMaskConfig(fraction_trainable=0.3, seed=0)
    cfg.pre_train(model)

    keys = set(cfg._registry.keys())
    # Frozen param should NOT have a mask entry.
    assert "0.weight" not in keys
    # Other linear weight still has one.
    assert "2.weight" in keys


def test_random_mask_excludes_via_fqn_pattern():
    model = _toy_model()
    cfg = RandomGradientMaskConfig(
        fraction_trainable=0.3,
        seed=0,
        exclude_fqn_patterns=["0.*"],
    )
    cfg.pre_train(model)
    keys = set(cfg._registry.keys())
    assert "0.weight" not in keys
    assert "2.weight" in keys


def test_random_mask_target_param_kinds_can_include_bias():
    model = _toy_model()
    cfg = RandomGradientMaskConfig(
        fraction_trainable=0.3,
        seed=0,
        target_param_kinds=["weight", "bias"],
        exclude_fqn_patterns=[],  # override default MoE-router exclude
    )
    cfg.pre_train(model)
    keys = set(cfg._registry.keys())
    assert "0.weight" in keys
    assert "0.bias" in keys


def test_random_mask_derived_seed_differs_per_param():
    # Two different param FQNs under the same global seed should
    # produce different masks (otherwise every layer would have the
    # same pattern).
    model = _toy_model()
    cfg = RandomGradientMaskConfig(fraction_trainable=0.5, seed=42)
    cfg.pre_train(model)
    h0 = stable_hash_u64(42, "0.weight")
    h2 = stable_hash_u64(42, "2.weight")
    assert h0 != h2


# ---------------------------------------------------------------------------
# Strict-freeze: snapshot + post-step restore.
# ---------------------------------------------------------------------------


def test_strict_freeze_off_by_default_means_no_snapshot():
    model = _toy_model()
    cfg = RandomGradientMaskConfig(fraction_trainable=0.5, seed=0)
    cfg.pre_train(model)
    assert cfg._snapshots is None


def test_strict_freeze_allocates_snapshot_per_masked_param():
    model = _toy_model()
    cfg = RandomGradientMaskConfig(
        fraction_trainable=0.5, seed=0, enforce_strict_freeze=True
    )
    cfg.pre_train(model)
    assert cfg._snapshots is not None
    # Every masked param has a snapshot of the same local shape.
    for fqn, mask in cfg._registry.items():
        snap = cfg._snapshots[fqn]
        # Snapshot mirrors the parameter's local shape (=global, no
        # DTensor in this test).
        p = dict(model.named_parameters())[fqn]
        assert snap.shape == p.shape
        # Mask and snapshot live on the same device (no offload).
        assert snap.device == mask.device


def test_strict_freeze_offload_puts_snapshot_on_cpu():
    model = _toy_model()
    cfg = RandomGradientMaskConfig(
        fraction_trainable=0.5,
        seed=0,
        enforce_strict_freeze=True,
        offload_strict_freeze_to_cpu=True,
    )
    cfg.pre_train(model)
    for snap in cfg._snapshots.values():
        assert snap.device == torch.device("cpu")


def _step_with_mask_and_restore(model: nn.Module, cfg, optim: torch.optim.Optimizer):
    """Mimic a single trainer step under the
    :class:`PEFTGradientCallback`: backward → pre_optim_step (mask grad)
    → optim.step → zero_grads → post_train_batch (post_optim_step)."""
    # Fake a backward pass: synthesize a nonzero gradient on every
    # trainable parameter. We use a known constant so the optimizer step
    # is deterministic.
    for p in model.parameters():
        if p.requires_grad:
            p.grad = torch.ones_like(p)
    cfg.transform(model)   # equivalent to pre_optim_step
    optim.step()
    optim.zero_grad(set_to_none=True)
    cfg.post_optim_step(model)


def test_strict_freeze_keeps_masked_entries_constant_under_adamw():
    """The acid test for issue #2 of the user's question: under AdamW
    with weight_decay > 0, masked entries should NOT drift even after
    many steps."""
    torch.manual_seed(0)
    model = nn.Linear(8, 4, bias=False)
    cfg = RandomGradientMaskConfig(
        fraction_trainable=0.5,
        seed=42,
        enforce_strict_freeze=True,
    )
    cfg.pre_train(model)
    mask = cfg._registry["weight"].bool()
    initial = model.weight.detach().clone()

    optim = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=0.1)
    for _ in range(20):
        _step_with_mask_and_restore(model, cfg, optim)

    # mask=0 positions are pinned at their initial values.
    assert torch.allclose(
        model.weight.detach()[~mask],
        initial[~mask],
        rtol=0.0,
        atol=0.0,
    )
    # mask=1 positions have moved (sanity: we're actually training).
    assert not torch.allclose(model.weight.detach()[mask], initial[mask])


def test_strict_freeze_keeps_masked_entries_constant_under_sgd_momentum():
    """SGD+momentum: residual momentum would normally cause drift in
    mask=0 entries (the user's general optimizer concern). Strict freeze
    eliminates it."""
    torch.manual_seed(0)
    model = nn.Linear(8, 4, bias=False)
    cfg = RandomGradientMaskConfig(
        fraction_trainable=0.5, seed=7, enforce_strict_freeze=True
    )
    cfg.pre_train(model)
    mask = cfg._registry["weight"].bool()
    initial = model.weight.detach().clone()

    optim = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9, weight_decay=0.1)
    for _ in range(50):
        _step_with_mask_and_restore(model, cfg, optim)

    assert torch.equal(model.weight.detach()[~mask], initial[~mask])


def test_strict_freeze_off_drifts_under_adamw_weight_decay():
    """Negative control: without strict freeze, AdamW weight decay
    *does* shrink masked entries (documented in the spec)."""
    torch.manual_seed(0)
    model = nn.Linear(8, 4, bias=False)
    cfg = RandomGradientMaskConfig(
        fraction_trainable=0.5, seed=42, enforce_strict_freeze=False
    )
    cfg.pre_train(model)
    mask = cfg._registry["weight"].bool()
    initial = model.weight.detach().clone()

    optim = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=0.1)
    for _ in range(20):
        _step_with_mask_and_restore(model, cfg, optim)

    # Without strict freeze, mask=0 entries drift (weight decay
    # shrinks them toward zero).
    assert not torch.equal(model.weight.detach()[~mask], initial[~mask])


def test_strict_freeze_offload_produces_identical_restore():
    """CPU offload must produce identical restored values to the
    device-resident path."""
    torch.manual_seed(0)
    model_a = nn.Linear(8, 4, bias=False)
    model_b = nn.Linear(8, 4, bias=False)
    model_b.load_state_dict(model_a.state_dict())  # same initial weights

    cfg_a = RandomGradientMaskConfig(
        fraction_trainable=0.5, seed=1, enforce_strict_freeze=True
    )
    cfg_b = RandomGradientMaskConfig(
        fraction_trainable=0.5,
        seed=1,
        enforce_strict_freeze=True,
        offload_strict_freeze_to_cpu=True,
    )
    cfg_a.pre_train(model_a)
    cfg_b.pre_train(model_b)

    optim_a = torch.optim.AdamW(model_a.parameters(), lr=1e-2, weight_decay=0.1)
    optim_b = torch.optim.AdamW(model_b.parameters(), lr=1e-2, weight_decay=0.1)
    for _ in range(10):
        _step_with_mask_and_restore(model_a, cfg_a, optim_a)
        _step_with_mask_and_restore(model_b, cfg_b, optim_b)

    assert torch.allclose(model_a.weight, model_b.weight, atol=0.0, rtol=0.0)


def test_strict_freeze_seamless_mid_run_activation():
    """End-to-end of the scenario the user described: train normally
    for N steps with AdamW (no mask), then 'flip on' strict-freeze
    masking. After the flip, the still-trainable entries continue to
    update normally from their inherited optimizer state; the now-frozen
    entries stay exactly at the flip-time values."""
    torch.manual_seed(0)
    model = nn.Linear(8, 4, bias=False)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=0.1)

    # Phase 1: 5 normal steps. No mask, no freeze. Optimizer accumulates
    # m, v.
    for _ in range(5):
        for p in model.parameters():
            p.grad = torch.ones_like(p)
        optim.step()
        optim.zero_grad(set_to_none=True)

    # Snapshot of parameters at the boundary.
    pre_flip_params = model.weight.detach().clone()
    # Capture m, v at the flip — used to prove the optimizer wasn't
    # reset by activating the mask.
    pre_flip_m = optim.state[model.weight]["exp_avg"].clone()
    pre_flip_v = optim.state[model.weight]["exp_avg_sq"].clone()
    assert (pre_flip_m.abs() > 0).all()  # sanity: phase 1 did accumulate

    # Phase 2: activate the mask + strict freeze. Inherit optimizer
    # state — we are NOT clearing m, v.
    cfg = RandomGradientMaskConfig(
        fraction_trainable=0.3, seed=99, enforce_strict_freeze=True
    )
    cfg.pre_train(model)
    mask = cfg._registry["weight"].bool()

    for _ in range(15):
        _step_with_mask_and_restore(model, cfg, optim)

    # 1. mask=0 entries are EXACTLY at their flip-time values.
    assert torch.equal(model.weight.detach()[~mask], pre_flip_params[~mask])

    # 2. mask=1 entries moved (we're actually still training them).
    assert not torch.allclose(model.weight.detach()[mask], pre_flip_params[mask])

    # 3. Optimizer state was inherited, not reset. At β2=0.999 and 15
    #    steps, v at mask=0 positions decays by at most ~1.5%, so v
    #    must still match within 2% of its pre-flip value (proving
    #    nothing zeroed it at the flip).
    post_v = optim.state[model.weight]["exp_avg_sq"]
    assert torch.allclose(
        post_v[~mask], pre_flip_v[~mask] * (0.999 ** 15), rtol=1e-3, atol=1e-7
    )
