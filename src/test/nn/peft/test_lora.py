"""
Tests for the LoRA ``ModelTransform``.

These tests construct a toy model whose FQN layout matches JOLMo's
:class:`~olmo_core.nn.transformer.Transformer` (``blocks.<idx>.attention.w_q``
etc.) so the LORA presets fire as they would on a real model.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn

from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.nn.peft import LoRAConfig, LoRALinear


# ---------------------------------------------------------------------------
# Fixtures.
# ---------------------------------------------------------------------------


def _toy_transformer_like(n_blocks: int = 2, d_model: int = 16) -> nn.Module:
    class _Attn(nn.Module):
        def __init__(self):
            super().__init__()
            self.w_q = nn.Linear(d_model, d_model)
            self.w_k = nn.Linear(d_model, d_model)
            self.w_v = nn.Linear(d_model, d_model)
            self.w_out = nn.Linear(d_model, d_model)

    class _FF(nn.Module):
        def __init__(self):
            super().__init__()
            self.w1 = nn.Linear(d_model, d_model * 4)
            self.w2 = nn.Linear(d_model * 4, d_model)
            self.w3 = nn.Linear(d_model, d_model * 4)

    class _Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.attention = _Attn()
            self.feed_forward = _FF()

    class _LMHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.w_out = nn.Linear(d_model, 32)

    class _Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.embeddings = nn.Embedding(32, d_model)
            self.blocks = nn.ModuleDict(
                {str(i): _Block() for i in range(n_blocks)}
            )
            self.lm_head = _LMHead()

    return _Model()


# ---------------------------------------------------------------------------
# Config validation.
# ---------------------------------------------------------------------------


def test_lora_config_rejects_zero_rank():
    with pytest.raises(OLMoConfigurationError):
        LoRAConfig(r=0, target_modules=["q"])


def test_lora_config_rejects_empty_targets():
    with pytest.raises(OLMoConfigurationError):
        LoRAConfig(r=8)


def test_lora_config_rejects_bad_dropout():
    with pytest.raises(OLMoConfigurationError):
        LoRAConfig(r=8, dropout=1.5, target_modules=["q"])


def test_lora_config_unknown_preset_raises():
    model = _toy_transformer_like()
    cfg = LoRAConfig(r=4, target_modules=["nonexistent_category"])
    with pytest.raises(OLMoConfigurationError, match="Unknown LoRA target category"):
        cfg.apply(model)


def test_lora_assert_compatible_blocks_tp():
    class _StubTrainModuleConfig:
        tp_config = object()

    cfg = LoRAConfig(r=4, target_modules=["q"])
    with pytest.raises(OLMoConfigurationError, match="LoRA \\+ tensor parallelism"):
        cfg.assert_compatible(_StubTrainModuleConfig())


# ---------------------------------------------------------------------------
# Module replacement.
# ---------------------------------------------------------------------------


def test_lora_replaces_matched_linears_with_lora_linear():
    model = _toy_transformer_like(n_blocks=2)
    LoRAConfig(r=4, alpha=8, target_modules=["q", "v"]).apply(model)

    for i in range(2):
        assert isinstance(model.blocks[str(i)].attention.w_q, LoRALinear)
        assert isinstance(model.blocks[str(i)].attention.w_v, LoRALinear)
        # Untargeted modules are unchanged.
        assert not isinstance(model.blocks[str(i)].attention.w_k, LoRALinear)
        assert not isinstance(model.blocks[str(i)].attention.w_out, LoRALinear)


def test_lora_freezes_non_adapter_params():
    model = _toy_transformer_like(n_blocks=2)
    LoRAConfig(r=4, target_modules=["q", "v"]).apply(model)

    for name, p in model.named_parameters():
        if ".lora_A." in name or ".lora_B." in name:
            assert p.requires_grad, name
        else:
            assert not p.requires_grad, name


def test_lora_modules_to_save_keeps_lm_head_trainable():
    model = _toy_transformer_like()
    LoRAConfig(
        r=4,
        target_modules=["q"],
        modules_to_save=["lm_head.*"],
    ).apply(model)
    lm_head_params = [
        (n, p)
        for n, p in model.named_parameters()
        if n.startswith("lm_head.")
    ]
    assert lm_head_params  # exists
    for name, p in lm_head_params:
        assert p.requires_grad, name


def test_lora_target_fqn_patterns_overrides_presets():
    model = _toy_transformer_like()
    cfg = LoRAConfig(
        r=4,
        target_modules=["q"],  # should be IGNORED
        target_fqn_patterns=["blocks.*.attention.w_out"],
    )
    cfg.apply(model)
    assert isinstance(model.blocks["0"].attention.w_out, LoRALinear)
    assert not isinstance(model.blocks["0"].attention.w_q, LoRALinear)


def test_lora_apply_raises_if_no_match():
    model = _toy_transformer_like()
    with pytest.raises(OLMoConfigurationError, match="zero modules"):
        LoRAConfig(r=4, target_fqn_patterns=["this.does.not.exist"]).apply(model)


# ---------------------------------------------------------------------------
# Initialization.
# ---------------------------------------------------------------------------


def test_lora_branches_initialized_correctly():
    model = _toy_transformer_like()
    LoRAConfig(r=4, alpha=8, target_modules=["q"]).apply(model)

    # Trigger reset_parameters as the model init pass would do.
    for module in model.modules():
        if hasattr(module, "reset_parameters"):
            module.reset_parameters()

    w_q = model.blocks["0"].attention.w_q
    assert isinstance(w_q, LoRALinear)
    # lora_B is zero-init.
    assert torch.allclose(w_q.lora_B.weight, torch.zeros_like(w_q.lora_B.weight))
    # lora_A is non-zero (Kaiming uniform).
    assert not torch.allclose(w_q.lora_A.weight, torch.zeros_like(w_q.lora_A.weight))


def test_lora_initial_forward_equals_base_forward():
    """At step 0, lora_B is zero, so the LoRA delta contributes 0."""
    model = _toy_transformer_like()
    LoRAConfig(r=4, alpha=8, target_modules=["q"]).apply(model)
    for module in model.modules():
        if hasattr(module, "reset_parameters"):
            module.reset_parameters()

    w_q = model.blocks["0"].attention.w_q
    x = torch.randn(2, w_q.in_features)
    out = w_q(x)
    base_out = w_q.base(x)
    assert torch.allclose(out, base_out, atol=1e-6, rtol=1e-5)


# ---------------------------------------------------------------------------
# Checkpoint key mapping.
# ---------------------------------------------------------------------------


def test_checkpoint_key_mapping_renames_base_to_original():
    model = _toy_transformer_like()
    cfg = LoRAConfig(r=4, target_modules=["q"])
    cfg.apply(model)
    mapping = cfg.checkpoint_key_mapping(model)
    assert mapping is not None

    # Every wrapped module should produce a base.weight mapping.
    assert (
        mapping.get("blocks.0.attention.w_q.base.weight")
        == "blocks.0.attention.w_q.weight"
    )
    # bias too (default nn.Linear has bias=True).
    assert (
        mapping.get("blocks.0.attention.w_q.base.bias")
        == "blocks.0.attention.w_q.bias"
    )
    # Unwrapped modules should not appear.
    assert "blocks.0.attention.w_k.base.weight" not in mapping


# ---------------------------------------------------------------------------
# rsLoRA scaling.
# ---------------------------------------------------------------------------


def test_rslora_scaling():
    model = _toy_transformer_like()
    LoRAConfig(r=16, alpha=8, target_modules=["q"], use_rslora=True).apply(model)
    w_q = model.blocks["0"].attention.w_q
    assert isinstance(w_q, LoRALinear)
    expected = 8.0 / math.sqrt(16)
    assert math.isclose(w_q.scaling, expected, rel_tol=1e-6)


def test_standard_lora_scaling():
    model = _toy_transformer_like()
    LoRAConfig(r=16, alpha=8, target_modules=["q"]).apply(model)
    w_q = model.blocks["0"].attention.w_q
    assert isinstance(w_q, LoRALinear)
    assert math.isclose(w_q.scaling, 8.0 / 16.0, rel_tol=1e-6)


# ---------------------------------------------------------------------------
# weight / bias property delegation.
# ---------------------------------------------------------------------------


def test_lora_linear_weight_property_delegates_to_base():
    model = _toy_transformer_like()
    LoRAConfig(r=4, target_modules=["q"]).apply(model)
    w_q = model.blocks["0"].attention.w_q
    assert isinstance(w_q, LoRALinear)
    assert w_q.weight is w_q.base.weight
    assert w_q.bias is w_q.base.bias
