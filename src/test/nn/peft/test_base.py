"""
Tests for the PEFT abstraction itself.

These tests don't depend on LoRA or random masking — they validate the
contract of ``ModelTransform`` / ``GradientTransform`` /
``GradientMaskTransform`` / ``PEFTConfig`` via the toy transforms in
``olmo_core.nn.peft.testing``.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from olmo_core.nn.peft import (
    PEFTConfig,
    stable_hash_u64,
)
from olmo_core.nn.peft.testing import _NoopModelTransform, _RecordingGradTransform


# ---------------------------------------------------------------------------
# Utilities.
# ---------------------------------------------------------------------------


def _make_toy_model() -> nn.Module:
    """Tiny stand-in for a real model used in abstraction tests."""
    return nn.Sequential(
        nn.Linear(8, 16),
        nn.ReLU(),
        nn.Linear(16, 4),
    )


# ---------------------------------------------------------------------------
# stable_hash_u64.
# ---------------------------------------------------------------------------


def test_stable_hash_u64_deterministic():
    # Same input → same output, every time.
    assert stable_hash_u64(0, "blocks.0.attention.w_q") == stable_hash_u64(
        0, "blocks.0.attention.w_q"
    )
    assert stable_hash_u64(42, "x") != stable_hash_u64(43, "x")
    assert stable_hash_u64(42, "x") != stable_hash_u64(42, "y")


def test_stable_hash_u64_fits_uint64():
    h = stable_hash_u64(123, "some.long.parameter.name")
    assert 0 <= h < (1 << 64)


# ---------------------------------------------------------------------------
# ModelTransform / PEFTConfig orchestration.
# ---------------------------------------------------------------------------


def test_apply_model_transforms_runs_in_order():
    model = _make_toy_model()
    peft = PEFTConfig(
        model_transforms=[
            _NoopModelTransform(note="first"),
            _NoopModelTransform(note="second"),
        ]
    )
    report = peft.apply_model_transforms(model)
    assert len(report.per_transform) == 2
    assert report.per_transform[0].notes == "first"
    assert report.per_transform[1].notes == "second"


def test_apply_model_transforms_idempotency_guard():
    model = _make_toy_model()
    peft = PEFTConfig(model_transforms=[_NoopModelTransform()])
    peft.apply_model_transforms(model)
    with pytest.raises(RuntimeError, match="already had PEFT applied"):
        peft.apply_model_transforms(model)


def test_trainability_report_counts_match_named_parameters():
    model = _make_toy_model()
    peft = PEFTConfig(model_transforms=[_NoopModelTransform()])
    report = peft.apply_model_transforms(model)

    expected_total = sum(p.numel() for p in model.parameters())
    expected_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    assert report.total_params == expected_total
    assert report.trainable_params == expected_trainable
    assert report.frozen_params == expected_total - expected_trainable


def test_checkpoint_key_mapping_conflict_raises():
    from olmo_core.exceptions import OLMoConfigurationError

    model = _make_toy_model()

    class _MapA(_NoopModelTransform):
        def checkpoint_key_mapping(self, model):
            return {"current": "old_a"}

    class _MapB(_NoopModelTransform):
        def checkpoint_key_mapping(self, model):
            return {"current": "old_b"}

    peft = PEFTConfig(model_transforms=[_MapA(), _MapB()])
    peft.apply_model_transforms(model)
    with pytest.raises(OLMoConfigurationError, match="Conflicting checkpoint key mappings"):
        peft.checkpoint_key_mapping(model)


# ---------------------------------------------------------------------------
# GradientTransform orchestration.
# ---------------------------------------------------------------------------


def test_recording_gradient_transform_round_trip():
    model = _make_toy_model()
    tx = _RecordingGradTransform()
    assert not tx.pre_train_called
    tx.pre_train(model)
    assert tx.pre_train_called

    # Fabricate gradients on every parameter.
    for p in model.parameters():
        p.grad = torch.zeros_like(p)

    tx.transform(model)
    assert len(tx.seen_fqns_per_step) == 1
    seen = tx.seen_fqns_per_step[0]
    expected = [name for name, _ in model.named_parameters()]
    assert seen == expected


def test_state_dict_round_trip_default_is_empty():
    tx = _RecordingGradTransform()
    sd = tx.state_dict()
    assert sd == {}
    tx.load_state_dict(sd)


# ---------------------------------------------------------------------------
# YAML / dict round-trip via OLMo-core's _CLASS_ discriminator.
# ---------------------------------------------------------------------------


def test_peft_config_round_trip_via_class_name_discriminator():
    """``PEFTConfig`` with concrete transforms in its lists must
    survive ``as_config_dict`` → ``from_dict`` via OLMo-core's
    ``_CLASS_`` discriminator (the same mechanism used for
    heterogeneous callback lists, etc.). This is the YAML-loading
    contract."""
    from olmo_core.nn.peft import LoRAConfig, RandomGradientMaskConfig

    original = PEFTConfig(
        model_transforms=[LoRAConfig(r=8, alpha=16, target_modules=["q", "v"])],
        gradient_transforms=[
            RandomGradientMaskConfig(fraction_trainable=0.1, seed=42)
        ],
    )
    encoded = original.as_config_dict()  # _CLASS_ markers included
    restored = PEFTConfig.from_dict(encoded)

    assert len(restored.model_transforms) == 1
    assert isinstance(restored.model_transforms[0], LoRAConfig)
    assert restored.model_transforms[0].r == 8
    assert restored.model_transforms[0].alpha == 16
    assert restored.model_transforms[0].target_modules == ["q", "v"]

    assert len(restored.gradient_transforms) == 1
    assert isinstance(restored.gradient_transforms[0], RandomGradientMaskConfig)
    assert restored.gradient_transforms[0].fraction_trainable == 0.1
    assert restored.gradient_transforms[0].seed == 42
