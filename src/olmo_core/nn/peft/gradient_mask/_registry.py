"""
Private storage for :class:`GradientMaskTransform` masks.

A thin typed wrapper around ``Dict[str, torch.Tensor]``. The dict maps
parameter FQNs to local-shard-shaped mask tensors. Used internally by
:class:`GradientMaskTransform`; concrete mask subclasses never touch it.

Kept private (``_registry``) so callers don't accumulate dependencies
on it. Promote to public if and when a non-mask gradient transform
wants the same storage.
"""

from __future__ import annotations

from typing import Dict, Iterator, Optional

import torch


class ParameterMaskRegistry:
    """``{parameter_fqn: local_mask}`` storage.

    Behaves like a dict; iteration order matches insertion order.
    """

    __slots__ = ("_masks",)

    def __init__(self) -> None:
        self._masks: Dict[str, torch.Tensor] = {}

    def __setitem__(self, fqn: str, mask: torch.Tensor) -> None:
        self._masks[fqn] = mask

    def __getitem__(self, fqn: str) -> torch.Tensor:
        return self._masks[fqn]

    def get(self, fqn: str, default: Optional[torch.Tensor] = None) -> Optional[torch.Tensor]:
        return self._masks.get(fqn, default)

    def __contains__(self, fqn: str) -> bool:
        return fqn in self._masks

    def __len__(self) -> int:
        return len(self._masks)

    def keys(self) -> Iterator[str]:
        return iter(self._masks.keys())

    def values(self) -> Iterator[torch.Tensor]:
        return iter(self._masks.values())

    def items(self) -> Iterator:
        return iter(self._masks.items())
