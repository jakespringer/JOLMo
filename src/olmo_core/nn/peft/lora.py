"""
LoRA — a :class:`ModelTransform` that adapts selected ``nn.Linear``
modules with a frozen base plus a trainable low-rank delta.

Modeled on the HuggingFace ``peft`` library's API; categories are
JOLMo-native names (``q``, ``k``, ``v``, ``o``, ``qkv``, ``gate_proj``,
``down_proj``, ``up_proj``, ``lm_head``) that resolve to module-FQN
globs over the JOLMo :class:`~olmo_core.nn.transformer.Transformer`
layout.

Parameter FQN convention after wrap:

    before:  blocks.0.attention.w_q.weight
    after:   blocks.0.attention.w_q.base.weight    (frozen)
             blocks.0.attention.w_q.lora_A.weight  (trainable)
             blocks.0.attention.w_q.lora_B.weight  (trainable)

See ``notes/parameter-efficient-training.md`` (§5).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from fnmatch import fnmatch
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from ...exceptions import OLMoConfigurationError
from .base import ModelTransform, ModelTransformReport, stable_hash_u64

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Target presets.
# ---------------------------------------------------------------------------


LORA_TARGET_PRESETS: Dict[str, str] = {
    "q":         "blocks.*.attention.w_q",
    "k":         "blocks.*.attention.w_k",
    "v":         "blocks.*.attention.w_v",
    "o":         "blocks.*.attention.w_out",
    "qkv":       "blocks.*.attention.w_qkv",     # FusedAttention
    "gate_proj": "blocks.*.feed_forward.w1",
    "down_proj": "blocks.*.feed_forward.w2",
    "up_proj":   "blocks.*.feed_forward.w3",
    "lm_head":   "lm_head.w_out",
}


# ---------------------------------------------------------------------------
# Module implementations.
# ---------------------------------------------------------------------------


class _LoRABranchLinear(nn.Linear):
    """A no-op-``reset_parameters`` ``nn.Linear`` used for LoRA's
    ``A`` / ``B`` branches.

    The standard model init pass calls ``reset_parameters`` on every
    descendant module; this subclass refuses to overwrite its
    parameters. :class:`LoRALinear` initializes its own branches in
    :meth:`LoRALinear.reset_parameters`.

    Also carries a class-level marker ``_peft_no_fp8`` so the Float8
    swap walker skips these (the LoRA branches should stay in
    bf16/fp32 while the base may be FP8'd).
    """

    _peft_no_fp8 = True

    def reset_parameters(self) -> None:  # noqa: D401 — intentional no-op
        # LoRALinear.reset_parameters handles initialization; without
        # this override the Kaiming/zero scheme would be clobbered.
        pass


class LoRALinear(nn.Module):
    """Drop-in replacement for ``nn.Linear`` with a frozen base weight
    and a trainable rank-decomposed delta.

    Forward: ``y = base(x) + scaling * lora_B(lora_A(dropout(x)))``.

    Children:
      - ``base`` — the original ``nn.Linear``; ``requires_grad=False``
        on both weight and (optional) bias.
      - ``lora_A`` — :class:`_LoRABranchLinear`, no bias, shape ``(r, in)``.
      - ``lora_B`` — :class:`_LoRABranchLinear`, no bias, shape ``(out, r)``.

    Convenience attributes:
      - ``weight``, ``bias`` — properties delegating to ``base``. This
        lets the model's init pass (``init_method._init_linear``) treat
        a :class:`LoRALinear` as if it were a plain ``nn.Linear``,
        initializing only the base weight while leaving the LoRA
        branches alone.
    """

    def __init__(
        self,
        original: nn.Linear,
        *,
        r: int,
        alpha: float,
        dropout: float,
        init_seed: int,
        fqn: str,
        use_rslora: bool = False,
    ) -> None:
        super().__init__()
        self.in_features = original.in_features
        self.out_features = original.out_features
        self.r = r
        self.alpha = alpha
        self.use_rslora = use_rslora
        self._init_seed = init_seed
        self._fqn = fqn

        # Adopt the original linear as our frozen base. Note: we do NOT
        # detach or clone; this preserves any meta-device / partial init.
        self.base = original
        self.base.weight.requires_grad = False
        if self.base.bias is not None:
            self.base.bias.requires_grad = False

        # LoRA branches. Match dtype/device of the base so meta-init
        # plus to_empty() works uniformly.
        device = self.base.weight.device
        dtype = self.base.weight.dtype
        self.lora_A = _LoRABranchLinear(
            self.in_features, self.r, bias=False, device=device, dtype=dtype
        )
        self.lora_B = _LoRABranchLinear(
            self.r, self.out_features, bias=False, device=device, dtype=dtype
        )

        if use_rslora:
            self.scaling: float = float(alpha) / math.sqrt(r)
        else:
            self.scaling = float(alpha) / float(r)

        self.dropout: nn.Module = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    # ------------------------------------------------------------------
    # Initialization. Called from the model init pass via
    # ``module.reset_parameters()``.
    # ------------------------------------------------------------------

    def reset_parameters(self) -> None:
        """Kaiming-uniform init for ``lora_A`` (``a=sqrt(5)``, matching
        ``peft``/``loralib`` defaults), zero init for ``lora_B``.

        The generator is pegged to the LoRA branch's local-tensor
        device (matching JOLMo's standard init convention). This is
        bit-deterministic across runs on the same hardware family; for
        true cross-platform determinism the user can hash a CPU-seeded
        init and copy it in post-hoc (out of scope for v1).
        """
        from ...distributed.utils import get_local_tensor
        from ..transformer.init import _apply_init  # local: avoid import cycle

        # Per-FQN derived seed; fits in int64.
        derived = stable_hash_u64(self._init_seed, self._fqn) % (2**63)
        gen_device = get_local_tensor(self.lora_A.weight).device
        gen = torch.Generator(device=gen_device).manual_seed(int(derived))

        _apply_init(
            nn.init.kaiming_uniform_,
            self.lora_A.weight,
            a=math.sqrt(5),
            generator=gen,
        )
        _apply_init(nn.init.zeros_, self.lora_B.weight)

    # ------------------------------------------------------------------
    # Forward.
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        # NOTE: dropout is applied to the input of the LoRA branch
        # (matching peft/loralib), not to the base output.
        lora_out = self.lora_B(self.lora_A(self.dropout(x)))
        return base_out + self.scaling * lora_out

    # ------------------------------------------------------------------
    # nn.Linear-like surface for init/normalize-matrix code paths.
    # ------------------------------------------------------------------

    @property
    def weight(self) -> torch.Tensor:
        return self.base.weight

    @property
    def bias(self) -> Optional[torch.Tensor]:
        return self.base.bias

    def extra_repr(self) -> str:
        return (
            f"in={self.in_features}, out={self.out_features}, r={self.r}, "
            f"alpha={self.alpha}, scaling={self.scaling:.4g}"
        )


# ---------------------------------------------------------------------------
# Config (the ModelTransform).
# ---------------------------------------------------------------------------


@dataclass
class LoRAConfig(ModelTransform):
    """LoRA adapters for a JOLMo :class:`Transformer`.

    Reference: Hu et al., "LoRA: Low-Rank Adaptation of Large Language
    Models", ICLR 2022.

    Example::

        LoRAConfig(r=8, alpha=16, target_modules=["q", "v"])

    will adapt every ``blocks.*.attention.w_q`` and
    ``blocks.*.attention.w_v`` module.
    """

    r: int = 8
    """LoRA rank. Each targeted ``nn.Linear(in, out)`` gains parameters
    ``A: (r, in)`` and ``B: (out, r)``."""

    alpha: float = 16.0
    """LoRA alpha. Adapter contribution is scaled by ``alpha/r``
    (or ``alpha/sqrt(r)`` when :attr:`use_rslora` is True)."""

    dropout: float = 0.0
    """Dropout on the LoRA branch input. Matches peft's
    ``lora_dropout``."""

    target_modules: List[str] = field(default_factory=list)
    """Logical weight categories to adapt. Resolved through
    :data:`LORA_TARGET_PRESETS`. See the module docstring for the
    available categories. Must be non-empty unless
    :attr:`target_fqn_patterns` is set."""

    target_fqn_patterns: Optional[List[str]] = None
    """Optional escape hatch: explicit ``fnmatch`` globs over MODULE
    FQNs (e.g. ``"blocks.*.attention.w_q"``). When non-empty,
    *replaces* :attr:`target_modules` resolution."""

    modules_to_save: Optional[List[str]] = None
    """``fnmatch`` globs over PARAMETER FQNs that should remain
    trainable alongside the LoRA adapters (e.g. ``["lm_head.*"]``).
    Mirrors peft's ``modules_to_save``."""

    init_seed: int = 0
    """Seed used to initialize ``lora_A`` (Kaiming). ``lora_B`` is
    always zero-init."""

    use_rslora: bool = False
    """If True, scale by ``alpha/sqrt(r)`` instead of ``alpha/r``
    (rank-stabilized LoRA, https://arxiv.org/abs/2312.03732)."""

    def __post_init__(self) -> None:
        if self.r <= 0:
            raise OLMoConfigurationError("LoRA 'r' must be > 0")
        if self.alpha <= 0:
            raise OLMoConfigurationError("LoRA 'alpha' must be > 0")
        if not (0.0 <= self.dropout < 1.0):
            raise OLMoConfigurationError("LoRA 'dropout' must be in [0, 1)")
        if not self.target_modules and not self.target_fqn_patterns:
            raise OLMoConfigurationError(
                "LoRA requires at least one of 'target_modules' or 'target_fqn_patterns'"
            )

    # ------------------------------------------------------------------
    # ModelTransform contract.
    # ------------------------------------------------------------------

    def assert_compatible(self, train_module_config) -> None:
        # Block LoRA + TP for v1 (see notes/parameter-efficient-training.md §5.4).
        tp_cfg = getattr(train_module_config, "tp_config", None)
        if tp_cfg is not None:
            raise OLMoConfigurationError(
                "LoRA + tensor parallelism is not supported in v1; "
                "drop tp_config or wait for the v2 LoRALinear.apply_tp."
            )

    def apply(self, model: nn.Module) -> ModelTransformReport:
        matched_fqns = self._resolve_target_fqns(model)

        # Replace each matched module with a LoRALinear in place.
        for fqn in matched_fqns:
            parent_fqn, _, leaf_name = fqn.rpartition(".")
            parent = model.get_submodule(parent_fqn) if parent_fqn else model
            original = getattr(parent, leaf_name)
            if not isinstance(original, nn.Linear):
                raise OLMoConfigurationError(
                    f"LoRA target {fqn!r} is a {type(original).__name__}, not nn.Linear"
                )
            wrapped = LoRALinear(
                original,
                r=self.r,
                alpha=self.alpha,
                dropout=self.dropout,
                init_seed=self.init_seed,
                fqn=fqn,
                use_rslora=self.use_rslora,
            )
            setattr(parent, leaf_name, wrapped)

        # Walk params and set requires_grad. Rules:
        #   - *.lora_A.*, *.lora_B.* → trainable
        #   - matches modules_to_save → unchanged (whatever the model author set)
        #   - else → frozen
        msave = list(self.modules_to_save or [])
        params_added = 0
        params_frozen = 0
        params_added_fqns: List[str] = []

        for name, p in model.named_parameters():
            if _is_lora_branch_param(name):
                if not p.requires_grad:
                    # Should always be the case for newly-added params; here
                    # for explicitness.
                    p.requires_grad = True
                params_added += p.numel()
                params_added_fqns.append(name)
            elif msave and any(fnmatch(name, pat) for pat in msave):
                # leave as-is
                pass
            else:
                if p.requires_grad:
                    p.requires_grad = False
                    params_frozen += p.numel()

        return ModelTransformReport(
            transform_name=type(self).__name__,
            modules_replaced=len(matched_fqns),
            params_added=params_added,
            params_frozen=params_frozen,
            target_fqns=tuple(matched_fqns),
            notes=(
                f"r={self.r}, alpha={self.alpha}, scaling="
                f"{(self.alpha / math.sqrt(self.r)) if self.use_rslora else (self.alpha / self.r):.4g}"
            ),
        )

    def checkpoint_key_mapping(self, model: nn.Module) -> Optional[Dict[str, str]]:
        """Map post-wrap parameter FQNs back to pre-wrap names so a
        pre-LoRA checkpoint loads.

        Example::

            {"blocks.0.attention.w_q.base.weight": "blocks.0.attention.w_q.weight"}

        Adapter params are absent from such checkpoints and retain
        their :meth:`LoRALinear.reset_parameters` init.
        """
        # We resolve targets from the model — which, by this point in the
        # lifecycle, has already had LoRA applied (otherwise there'd be
        # no .base attribute to alias). Discover them by walking for
        # LoRALinear instances directly.
        mapping: Dict[str, str] = {}
        for fqn, m in model.named_modules():
            if isinstance(m, LoRALinear):
                # base.weight / base.bias → original weight / bias
                mapping[f"{fqn}.base.weight"] = f"{fqn}.weight"
                if m.base.bias is not None:
                    mapping[f"{fqn}.base.bias"] = f"{fqn}.bias"
        return mapping or None

    # ------------------------------------------------------------------
    # Target resolution.
    # ------------------------------------------------------------------

    def _resolve_target_fqns(self, model: nn.Module) -> List[str]:
        """Resolve user-facing target categories / patterns into a
        deterministic list of MODULE FQNs in the model. Returns FQNs in
        the order ``named_modules`` yields them."""
        if self.target_fqn_patterns:
            patterns = list(self.target_fqn_patterns)
        else:
            patterns = []
            for cat in self.target_modules:
                if cat not in LORA_TARGET_PRESETS:
                    raise OLMoConfigurationError(
                        f"Unknown LoRA target category {cat!r}. "
                        f"Known: {sorted(LORA_TARGET_PRESETS.keys())}"
                    )
                patterns.append(LORA_TARGET_PRESETS[cat])

        matched: List[str] = []
        seen: set = set()
        for fqn, _ in model.named_modules():
            if not fqn:
                continue
            if any(fnmatch(fqn, pat) for pat in patterns):
                if fqn not in seen:
                    matched.append(fqn)
                    seen.add(fqn)

        if not matched:
            raise OLMoConfigurationError(
                "LoRA target resolution produced zero modules. "
                f"Patterns: {patterns}. Check 'target_modules' against your model."
            )
        return matched


def _is_lora_branch_param(fqn: str) -> bool:
    """True if ``fqn`` names a parameter inside a LoRA ``A`` or ``B``
    branch (i.e. ``*.lora_A.*`` or ``*.lora_B.*``)."""
    return ".lora_A." in fqn or ".lora_B." in fqn


__all__ = [
    "LoRAConfig",
    "LoRALinear",
    "LORA_TARGET_PRESETS",
]
