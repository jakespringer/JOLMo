"""
Tests for the freezing primitives (BitFit, HeadOnly, TopNBlocks,
FreezeFQNPatternsTransform).
"""

from __future__ import annotations

import torch.nn as nn

from olmo_core.nn.peft import (
    BitFitTransform,
    FreezeFQNPatternsTransform,
    HeadOnlyTransform,
    TopNBlocksTransform,
)


def _toy_model_with_blocks() -> nn.Module:
    """Three-block stand-in matching JOLMo's ``Transformer.blocks`` shape."""

    class _Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.attention = nn.Linear(8, 8)
            self.feed_forward = nn.Linear(8, 8)

    class _Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.embeddings = nn.Embedding(64, 8)
            self.blocks = nn.ModuleDict({"0": _Block(), "1": _Block(), "2": _Block()})
            self.lm_head = nn.Sequential(nn.Linear(8, 64))

    return _Model()


def _trainable_fqns(model: nn.Module) -> set:
    return {name for name, p in model.named_parameters() if p.requires_grad}


def test_freeze_fqn_patterns_freezes_nothing_by_default():
    model = _toy_model_with_blocks()
    before = _trainable_fqns(model)
    FreezeFQNPatternsTransform().apply(model)
    after = _trainable_fqns(model)
    assert before == after


def test_freeze_fqn_patterns_freezes_matching_keeps_excepted():
    model = _toy_model_with_blocks()
    FreezeFQNPatternsTransform(
        freeze_patterns=["blocks.*"],
        except_patterns=["blocks.0.*"],
    ).apply(model)
    trainable = _trainable_fqns(model)
    # blocks.0.* still trainable; blocks.1.* and blocks.2.* frozen.
    assert any(n.startswith("blocks.0.") for n in trainable)
    assert not any(n.startswith("blocks.1.") for n in trainable)
    assert not any(n.startswith("blocks.2.") for n in trainable)
    # embeddings and lm_head untouched (no freeze pattern matched).
    assert any(n.startswith("embeddings") for n in trainable)
    assert any(n.startswith("lm_head") for n in trainable)


def test_bitfit_keeps_only_biases_trainable():
    model = _toy_model_with_blocks()
    BitFitTransform().apply(model)
    trainable = _trainable_fqns(model)
    assert trainable  # at least the biases
    for name in trainable:
        assert name.endswith(".bias"), name


def test_head_only_keeps_only_lm_head_trainable():
    model = _toy_model_with_blocks()
    HeadOnlyTransform().apply(model)
    trainable = _trainable_fqns(model)
    assert trainable
    for name in trainable:
        assert name.startswith("lm_head."), name


def test_top_n_blocks_keeps_last_n_plus_lm_head():
    model = _toy_model_with_blocks()
    TopNBlocksTransform(n=1, keep_lm_head=True).apply(model)
    trainable = _trainable_fqns(model)
    # The last block (index 2) is trainable, blocks 0 and 1 are frozen.
    assert any(n.startswith("blocks.2.") for n in trainable)
    assert not any(n.startswith("blocks.0.") for n in trainable)
    assert not any(n.startswith("blocks.1.") for n in trainable)
    assert any(n.startswith("lm_head.") for n in trainable)
    assert not any(n.startswith("embeddings") for n in trainable)


def test_top_n_blocks_n_zero_freezes_everything_except_lm_head():
    model = _toy_model_with_blocks()
    TopNBlocksTransform(n=0, keep_lm_head=True).apply(model)
    trainable = _trainable_fqns(model)
    assert all(n.startswith("lm_head.") for n in trainable)


def test_top_n_blocks_keep_lm_head_false():
    model = _toy_model_with_blocks()
    TopNBlocksTransform(n=1, keep_lm_head=False).apply(model)
    trainable = _trainable_fqns(model)
    assert any(n.startswith("blocks.2.") for n in trainable)
    assert not any(n.startswith("lm_head.") for n in trainable)
