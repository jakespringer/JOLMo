"""
Freezing primitives — ``ModelTransform`` instances whose only action is
to set ``requires_grad=False`` on parameters matching a pattern.

Methods that compile to "freeze a set of FQNs" (BitFit, head-only,
last-N blocks, "everything but the embeddings", etc.) are thin
wrappers around :class:`FreezeFQNPatternsTransform`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fnmatch import fnmatch
from typing import List

import torch.nn as nn

from .base import ModelTransform, ModelTransformReport


def _matches_any(name: str, patterns: List[str]) -> bool:
    return any(fnmatch(name, pat) for pat in patterns)


@dataclass
class FreezeFQNPatternsTransform(ModelTransform):
    """Set ``requires_grad=False`` on every parameter whose FQN matches
    any glob in :attr:`freeze_patterns` **unless** it also matches any
    glob in :attr:`except_patterns`.

    Does not add or replace modules; does not rename parameters; does
    not require any checkpoint key mapping.
    """

    freeze_patterns: List[str] = field(default_factory=list)
    """``fnmatch`` globs over parameter FQNs (e.g. ``"*"``,
    ``"blocks.0.*"``, ``"embeddings.*"``). Empty list = freeze nothing
    (no-op)."""

    except_patterns: List[str] = field(default_factory=list)
    """``fnmatch`` globs over parameter FQNs to leave trainable even if
    they match :attr:`freeze_patterns`."""

    def apply(self, model: nn.Module) -> ModelTransformReport:
        frozen = 0
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if not _matches_any(name, self.freeze_patterns):
                continue
            if _matches_any(name, self.except_patterns):
                continue
            p.requires_grad = False
            frozen += p.numel()

        return ModelTransformReport(
            transform_name=type(self).__name__,
            params_frozen=frozen,
        )


@dataclass
class BitFitTransform(FreezeFQNPatternsTransform):
    """BitFit: freeze every parameter except biases.

    Reference: Ben-Zaken et al., "BitFit: Simple Parameter-efficient
    Fine-tuning for Transformer-based Masked Language-models", 2021.
    """

    freeze_patterns: List[str] = field(default_factory=lambda: ["*"])
    except_patterns: List[str] = field(default_factory=lambda: ["*.bias"])


@dataclass
class HeadOnlyTransform(FreezeFQNPatternsTransform):
    """Freeze every parameter except those in ``lm_head.*``.

    Useful for "linear probe" style training on top of a frozen
    encoder."""

    freeze_patterns: List[str] = field(default_factory=lambda: ["*"])
    except_patterns: List[str] = field(default_factory=lambda: ["lm_head.*"])


@dataclass
class TopNBlocksTransform(ModelTransform):
    """Freeze every parameter except those in the last ``n`` transformer
    blocks (and, optionally, the LM head).

    Block FQN convention: ``blocks.<idx>.*`` (matches JOLMo
    :class:`Transformer`).
    """

    n: int = 1
    """Number of trailing blocks to keep trainable."""

    keep_lm_head: bool = True
    """If ``True`` (default), also keep ``lm_head.*`` trainable."""

    def __post_init__(self) -> None:
        from ...exceptions import OLMoConfigurationError

        if self.n < 0:
            raise OLMoConfigurationError("TopNBlocksTransform.n must be >= 0")

    def apply(self, model: nn.Module) -> ModelTransformReport:
        # Discover block indices from the model (avoids hardcoding a count).
        indices: List[int] = []
        if hasattr(model, "blocks") and isinstance(model.blocks, nn.ModuleDict):
            for k in model.blocks.keys():
                try:
                    indices.append(int(k))
                except ValueError:
                    continue
        indices.sort()

        if self.n == 0:
            keep_indices: List[int] = []
        else:
            keep_indices = indices[-self.n :]

        keep_patterns = [f"blocks.{i}.*" for i in keep_indices]
        if self.keep_lm_head:
            keep_patterns.append("lm_head.*")

        frozen = 0
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if _matches_any(name, keep_patterns):
                continue
            p.requires_grad = False
            frozen += p.numel()

        return ModelTransformReport(
            transform_name=type(self).__name__,
            params_frozen=frozen,
            notes=f"kept_blocks={keep_indices}, keep_lm_head={self.keep_lm_head}",
        )


__all__ = [
    "FreezeFQNPatternsTransform",
    "BitFitTransform",
    "HeadOnlyTransform",
    "TopNBlocksTransform",
]
