"""
Foundational abstractions for parameter-efficient training (PEFT).

See ``notes/parameter-efficient-training.md`` for the design rationale.

This module defines:

  - :class:`ModelTransform` — a build-time mutation of an ``nn.Module``.
  - :class:`GradientTransform` — a per-step mutation of ``param.grad``.
  - :class:`GradientMaskTransform` — sugar base for multiplicative-mask
    gradient transforms.
  - :class:`PEFTConfig` — the user-facing container that orchestrates
    both kinds of transform.
  - :class:`ModelTransformReport`, :class:`TrainabilityReport` —
    structured logs of what changed.
  - :func:`stable_hash_u64`, :func:`_local_shard_of` — shared utilities.

The intent is that LoRA, BitFit, IA³, random masking, Fisher masking,
etc., are all expressible as concrete subclasses with no further changes
to this module.
"""

from __future__ import annotations

import hashlib
import logging
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass, field
from fnmatch import fnmatch
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Optional,
    Tuple,
)

import torch
import torch.nn as nn

from ...config import Config

if TYPE_CHECKING:
    from ...train.trainer import Trainer

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Utilities.
# ---------------------------------------------------------------------------


def stable_hash_u64(global_seed: int, key: str) -> int:
    """Deterministic 64-bit hash of ``(global_seed, key)``.

    Implementation: SHA-256 of ``f"{global_seed}:{key}".encode()``,
    first 8 bytes interpreted as big-endian unsigned integer.

    Bit-identical across Python invocations, platforms, and processes
    — explicitly *not* Python's built-in ``hash`` (which is randomized
    per-process via ``PYTHONHASHSEED``).
    """
    digest = hashlib.sha256(f"{int(global_seed)}:{key}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def _local_shard_of(global_tensor: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    """Return the rank-local shard of ``global_tensor`` shaped to match
    the placement of ``like``.

    If ``like`` is a plain ``Tensor`` (no parallelism), the full
    ``global_tensor`` is returned. If ``like`` is a ``DTensor``, the
    local shard under ``like``'s ``device_mesh`` / ``placements`` is
    returned.
    """
    from ...distributed.utils import distribute_like, get_local_tensor

    return get_local_tensor(distribute_like(like, global_tensor))


def _matches_any(name: str, patterns: List[str]) -> bool:
    return any(fnmatch(name, pat) for pat in patterns)


# ---------------------------------------------------------------------------
# Reports.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelTransformReport:
    """Structured log of what a single :class:`ModelTransform` did."""

    transform_name: str
    modules_replaced: int = 0
    modules_added: int = 0
    params_frozen: int = 0
    params_added: int = 0
    target_fqns: Tuple[str, ...] = ()
    notes: str = ""

    def summary(self) -> str:
        parts = [f"{self.transform_name}"]
        if self.modules_replaced:
            parts.append(f"modules_replaced={self.modules_replaced}")
        if self.modules_added:
            parts.append(f"modules_added={self.modules_added}")
        if self.params_added:
            parts.append(f"params_added={self.params_added:,d}")
        if self.params_frozen:
            parts.append(f"params_frozen={self.params_frozen:,d}")
        if self.notes:
            parts.append(self.notes)
        return " | ".join(parts)


@dataclass(frozen=True)
class TrainabilityReport:
    """Aggregate report covering every applied :class:`ModelTransform`."""

    per_transform: Tuple[ModelTransformReport, ...]
    total_params: int
    trainable_params: int
    added_params: int
    frozen_params: int

    @property
    def trainable_fraction(self) -> float:
        return 0.0 if self.total_params == 0 else self.trainable_params / self.total_params

    def summary(self) -> str:
        tx_names = [r.transform_name for r in self.per_transform]
        pct = 100.0 * self.trainable_fraction
        return (
            f"Trainability: trainable={self.trainable_params:,d} / total={self.total_params:,d} "
            f"({pct:.2f}%), frozen={self.frozen_params:,d}, added={self.added_params:,d}; "
            f"transforms=[{', '.join(tx_names)}]"
        )


# ---------------------------------------------------------------------------
# Abstractions.
# ---------------------------------------------------------------------------


@dataclass
class ModelTransform(Config, metaclass=ABCMeta):
    """A build-time, in-place mutation of an ``nn.Module``.

    A :class:`ModelTransform` may replace submodules, add submodules,
    and set ``requires_grad`` on parameters. It declares which params
    are trainable under the corresponding PEFT method.

    Contract:

      MUST:
        - mutate ``model`` in place; do not return a new model.
        - be deterministic given its config; any randomness derives
          from an explicit ``seed``-style field on the subclass.
        - be applied on the *unparallelized* model, before any
          ``apply_tp`` / ``apply_fsdp`` / ``apply_pp`` calls.
        - leave ``model.embeddings`` and ``model.lm_head`` accessible
          (the rest of the codebase expects them).

      MUST NOT:
        - call ``model.init_weights``.
        - assume idempotence; the orchestrator guarantees each
          transform runs exactly once per model.

      SHOULD:
        - return a :class:`ModelTransformReport` describing what
          happened.
        - implement :meth:`checkpoint_key_mapping` if it renames
          parameters, so pre-transform checkpoints still load.
        - implement :meth:`assert_compatible` to fail fast on
          unsupported combinations (e.g. LoRA + TP in v1).
        - rely on per-module wrapping tricks (e.g. property
          delegation, no-op ``reset_parameters`` on adapter branches)
          to make adapter parameters survive the model's standard
          init pass, rather than relying on the init pass to skip
          them. The transformer's ``init_weights`` does not currently
          accept a skip predicate.
    """

    @abstractmethod
    def apply(self, model: nn.Module) -> ModelTransformReport:
        raise NotImplementedError

    def checkpoint_key_mapping(self, model: nn.Module) -> Optional[Dict[str, str]]:
        """Optional. Mapping ``{current_fqn: checkpoint_fqn}`` so a
        pre-transform checkpoint loads cleanly."""
        del model
        return None

    def assert_compatible(self, train_module_config: Any) -> None:
        """Optional. Raise :class:`OLMoConfigurationError` for
        incompatible config combinations. Called once at config-build
        time so the user sees the error before any training resources
        are allocated."""
        del train_module_config


@dataclass
class GradientTransform(Config, metaclass=ABCMeta):
    """A per-step, in-place mutation of ``param.grad``.

    Runs after the backward pass and before the optimizer step
    (specifically: before gradient clipping inside
    ``optim_step``). Stateful but lazily-allocated in
    :meth:`pre_train`, after parallelization is final.

    Multiple :class:`GradientTransform` instances compose in the order
    declared in :attr:`PEFTConfig.gradient_transforms`.
    """

    @abstractmethod
    def pre_train(self, model: nn.Module) -> None:
        """Allocate per-parameter state. Called once, after
        parallelization, before the first training step."""
        raise NotImplementedError

    @abstractmethod
    def transform(self, model: nn.Module) -> None:
        """Modify ``param.grad`` in place. Called every step.

        Implementations should skip parameters with ``p.grad is None``
        and parameters they don't own."""
        raise NotImplementedError

    def post_optim_step(self, model: nn.Module) -> None:
        """Optional hook fired *after* ``optimizer.step()`` has run and
        gradients have been zeroed, in the trainer's ``post_train_batch``
        phase. Default no-op.

        Use this to apply parameter-space corrections that depend on the
        post-step parameter values (e.g., the strict-freeze restore in
        :class:`GradientMaskTransform`).
        """
        del model

    def state_dict(self) -> Dict[str, Any]:
        """State to checkpoint. Defaults to empty (stateless)."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Restore from a state dict. Defaults to no-op."""
        del state_dict


@dataclass
class GradientMaskTransform(GradientTransform, metaclass=ABCMeta):
    """Sugar base for any :class:`GradientTransform` whose action is a
    per-parameter multiplicative mask.

    Subclasses implement :meth:`generate_mask` only. The base handles:

      - parameter selection (module-type / param-kind / FQN-exclusion
        filters),
      - deterministic seeding (one CPU ``Generator`` per parameter,
        seeded from ``(self.seed, param_fqn)``),
      - DTensor-correct sharding (slicing the global mask under the
        param's ``device_mesh`` / ``placements``),
      - storage of local-shard-shaped masks on the param's device,
      - per-step application (``local_grad.mul_(local_mask)``).

    Concrete masks (random, magnitude, Fisher, structured) are
    ~10-line subclasses.

    .. note::
        By default, gradient masking does **not** fully freeze masked
        entries under stateful optimizers: AdamW's weight decay still
        shrinks ``p``, Adam's residual momentum still produces updates,
        Lion's sign-of-momentum continues, Muon's orthogonalization
        couples masked and unmasked entries through the spectral
        update, and so on. The gradient is only one input to the
        update — preconditioners, momentum, and weight decay all act
        on parameters directly.

        Set :attr:`enforce_strict_freeze` to ``True`` to get optimizer-
        agnostic guaranteed-constant mask=0 entries via a parameter
        snapshot taken at the start of training. This works under any
        optimizer because it operates in parameter space, not gradient
        space.
    """

    target_module_types: List[str] = field(default_factory=lambda: ["Linear"])
    """Module class names (``type(m).__name__``) whose parameters are
    eligible. Default ``["Linear"]`` covers all weight matrices."""

    target_param_kinds: List[str] = field(default_factory=lambda: ["weight"])
    """Within an eligible module, only parameters whose leaf attribute
    name is in this list are masked. Default ``["weight"]`` leaves
    biases alone."""

    exclude_fqn_patterns: List[str] = field(default_factory=list)
    """``fnmatch`` globs over parameter FQNs that should *not* be masked
    even when the module-type filter matches. Use to protect e.g. the
    LM head or MoE routers."""

    seed: int = 0
    """Global seed. Per-parameter seeds are derived as
    ``stable_hash_u64(seed, param_fqn)``."""

    enforce_strict_freeze: bool = False
    """If ``True``, after each optimizer step, restore parameter values
    at ``mask=0`` positions to a snapshot taken at the start of
    training. This makes the freeze *optimizer-agnostic* — entries
    stay exactly constant regardless of weight decay, residual
    momentum, sign updates, orthogonalization, or any other update
    rule.

    Memory cost: one full local-shard-shaped tensor per masked
    parameter, same dtype as the parameter, allocated in
    :meth:`pre_train` and held for the run. Equivalent in size to one
    extra copy of the masked parameters per rank. Works uniformly
    under FSDP / TP / CP / EP / PP because the snapshot is always
    sized to the local shard (we go through ``get_local_tensor`` /
    ``_local_shard_of``)."""

    offload_strict_freeze_to_cpu: bool = False
    """When :attr:`enforce_strict_freeze` is ``True``, store the
    snapshot on host memory instead of the parameter's device. Trades
    GPU memory for one device-side copy of the full snapshot per
    optimizer step. The copy is bandwidth-bound and typically much
    smaller than the optimizer step's own work, but profile it on
    your hardware before committing."""

    @abstractmethod
    def generate_mask(
        self,
        param_fqn: str,
        global_shape: torch.Size,
        generator: torch.Generator,
    ) -> torch.Tensor:
        """Return a CPU mask tensor of shape ``global_shape``.

        ``generator`` is already seeded deterministically from
        ``(self.seed, param_fqn)``. It is the only source of randomness
        the subclass should use; *do not* read other RNG state.

        The returned tensor may be ``bool`` or a float dtype; the base
        class stores it as-is and multiplies it into gradients each step.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Base-provided plumbing.
    # ------------------------------------------------------------------

    def pre_train(self, model: nn.Module) -> None:
        from ...distributed.utils import get_local_tensor
        from .gradient_mask._registry import ParameterMaskRegistry

        registry = ParameterMaskRegistry()
        eligible: Dict[str, nn.Module] = self._eligible_modules(model)

        # If strict freeze is enabled we also keep a full snapshot of the
        # local shard of every masked parameter. This is what gives us
        # the optimizer-agnostic freeze: each step we restore mask=0
        # positions to their snapshot value, undoing any drift the
        # optimizer introduced (weight decay, residual momentum, Lion
        # sign updates, Muon orthogonalization, ...). The snapshot is
        # always local-shard-shaped — ``get_local_tensor`` peels
        # DTensors regardless of FSDP / TP / CP / EP / PP placement, so
        # the same code is correct under every parallelism strategy.
        snapshots: Optional[Dict[str, torch.Tensor]] = (
            {} if self.enforce_strict_freeze else None
        )

        for fqn, p in model.named_parameters():
            if not self._param_is_eligible(fqn, p, eligible):
                continue

            derived_seed = stable_hash_u64(self.seed, fqn)
            gen = torch.Generator(device="cpu").manual_seed(derived_seed)

            global_shape = p.shape  # DTensor.shape returns the global shape
            global_mask = self.generate_mask(fqn, global_shape, gen)
            if tuple(global_mask.shape) != tuple(global_shape):
                raise RuntimeError(
                    f"generate_mask for {fqn!r} returned shape {tuple(global_mask.shape)}, "
                    f"expected {tuple(global_shape)}"
                )

            local_mask = _local_shard_of(global_mask, p).detach().to(p.device)
            registry[fqn] = local_mask
            del global_mask  # do not retain

            if snapshots is not None:
                # Snapshot the local-shard view of the parameter's data,
                # detached + cloned so subsequent optimizer steps don't
                # mutate it. ``.detach()`` is critical: parameters
                # require_grad=True and a bare ``.clone()`` would build
                # an autograd graph.
                local_p = get_local_tensor(p.data).detach().clone()
                if self.offload_strict_freeze_to_cpu:
                    local_p = local_p.to("cpu")
                snapshots[fqn] = local_p

        self._registry = registry
        self._snapshots: Optional[Dict[str, torch.Tensor]] = snapshots

        n_masked = len(registry)
        n_elements = sum(m.numel() for m in registry.values())
        log.info(
            f"{type(self).__name__}: masked {n_masked} parameter(s) "
            f"covering {n_elements:,d} local elements"
        )
        if snapshots is not None:
            snap_bytes = sum(t.numel() * t.element_size() for t in snapshots.values())
            where = "CPU" if self.offload_strict_freeze_to_cpu else "device"
            log.info(
                f"{type(self).__name__}: strict-freeze snapshot allocated on {where} "
                f"({snap_bytes / (1024 ** 2):.1f} MiB local)"
            )

    def transform(self, model: nn.Module) -> None:
        from ...distributed.utils import get_local_tensor

        registry = getattr(self, "_registry", None)
        if registry is None:  # pre_train not run; safe no-op
            return

        for fqn, p in model.named_parameters():
            mask = registry.get(fqn)
            if mask is None or p.grad is None:
                continue
            get_local_tensor(p.grad).mul_(mask)

    def post_optim_step(self, model: nn.Module) -> None:
        """Restore mask=0 positions to the snapshot taken in
        :meth:`pre_train`. No-op when :attr:`enforce_strict_freeze` is
        ``False``.

        Runs in the trainer's ``post_train_batch`` phase via
        :class:`~olmo_core.train.callbacks.peft_gradient.PEFTGradientCallback`,
        after ``optim_step`` and ``zero_grads`` have completed and
        before any logging / checkpoint callback observes the
        parameters (since ``PEFTGradientCallback.priority=50`` runs
        ahead of the default-priority callbacks).
        """
        from ...distributed.utils import get_local_tensor

        registry = getattr(self, "_registry", None)
        snapshots = getattr(self, "_snapshots", None)
        if registry is None or snapshots is None:
            return

        with torch.no_grad():
            for fqn, p in model.named_parameters():
                mask = registry.get(fqn)
                snap = snapshots.get(fqn)
                if mask is None or snap is None:
                    continue
                local_p = get_local_tensor(p.data)

                # CPU-offload path: copy the snapshot to the parameter's
                # device. ``non_blocking=True`` lets the copy overlap
                # with the subsequent torch.where on a separate stream
                # (when the caller has one).
                src = snap
                if src.device != local_p.device:
                    src = src.to(local_p.device, non_blocking=True)
                if src.dtype != local_p.dtype:
                    src = src.to(dtype=local_p.dtype)

                # ``mask`` is bool / 0-1; ``torch.where`` with a bool
                # condition dispatches to a fused kernel. Result is
                # copied back into the parameter's local shard in
                # place, preserving the storage identity FSDP relies on.
                local_p.copy_(torch.where(mask.bool(), local_p, src))

    # ------------------------------------------------------------------
    # Selection helpers.
    # ------------------------------------------------------------------

    def _eligible_modules(self, model: nn.Module) -> Dict[str, nn.Module]:
        """Map from module FQN → module, for modules matching
        ``target_module_types`` (by ``type(m).__name__`` OR by MRO ancestor
        name, so subclasses count).

        Includes the root module (FQN ``""``) when it matches — handles
        flat models like ``model = nn.Linear(...)``."""
        wanted = set(self.target_module_types)
        out: Dict[str, nn.Module] = {}
        for fqn, m in model.named_modules():
            mro_names = {cls.__name__ for cls in type(m).__mro__}
            if wanted & mro_names:
                out[fqn] = m
        return out

    def _param_is_eligible(
        self,
        param_fqn: str,
        param: torch.Tensor,
        eligible_modules: Dict[str, nn.Module],
    ) -> bool:
        # Skip frozen parameters: nothing to mask, and ``p.grad`` will be None.
        if not param.requires_grad:
            return False

        # FQN of the form "a.b.c.weight" → module FQN "a.b.c", leaf "weight".
        module_fqn, _, leaf = param_fqn.rpartition(".")
        if leaf not in self.target_param_kinds:
            return False
        if module_fqn not in eligible_modules:
            return False
        if _matches_any(param_fqn, self.exclude_fqn_patterns):
            return False
        return True


# ---------------------------------------------------------------------------
# Orchestrator config.
# ---------------------------------------------------------------------------


@dataclass
class PEFTConfig(Config):
    """User-facing container that orchestrates PEFT for a training run.

    Two lists:

      - :attr:`model_transforms` — applied once, in declared order,
        before parallelization. Each entry is a :class:`ModelTransform`.
      - :attr:`gradient_transforms` — wrapped into a single
        auto-installed :class:`~olmo_core.train.callbacks.peft_gradient.PEFTGradientCallback`
        which runs them in declared order at every ``pre_optim_step``.

    Composition semantics:

      - Within either list, **declared order matters**. A later model
        transform sees the model produced by earlier ones; a later
        gradient transform sees gradients produced by earlier ones.
      - Across lists: model transforms always run first (build time);
        gradient transforms run every step thereafter.
    """

    model_transforms: List[ModelTransform] = field(default_factory=list)
    gradient_transforms: List[GradientTransform] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Compatibility / orchestration entry points.
    # ------------------------------------------------------------------

    def assert_compatible(self, train_module_config: Any) -> None:
        """Run every model transform's compatibility check. Called once
        at config-build time."""
        for tx in self.model_transforms:
            tx.assert_compatible(train_module_config)

    def apply_model_transforms(self, model: nn.Module) -> TrainabilityReport:
        """Run every model transform in declared order. Returns an
        aggregated :class:`TrainabilityReport`."""
        if getattr(model, "_peft_applied", False):
            raise RuntimeError(
                "PEFTConfig.apply_model_transforms was called on a model "
                "that has already had PEFT applied; this is a programmer error."
            )

        per_transform: List[ModelTransformReport] = []
        for tx in self.model_transforms:
            report = tx.apply(model)
            log.info(report.summary())
            per_transform.append(report)

        total_params = 0
        trainable_params = 0
        for _, p in model.named_parameters():
            total_params += p.numel()
            if p.requires_grad:
                trainable_params += p.numel()
        added_params = sum(r.params_added for r in per_transform)
        frozen_params = total_params - trainable_params

        # Mark so we don't double-apply.
        try:
            model._peft_applied = True  # type: ignore[assignment]
        except Exception:
            pass

        report = TrainabilityReport(
            per_transform=tuple(per_transform),
            total_params=total_params,
            trainable_params=trainable_params,
            added_params=added_params,
            frozen_params=frozen_params,
        )
        log.info(report.summary())
        return report

    def checkpoint_key_mapping(self, model: nn.Module) -> Dict[str, str]:
        """Merge every model transform's key mapping. Raises on
        conflicts (same current_fqn → different checkpoint_fqns)."""
        from ...exceptions import OLMoConfigurationError

        merged: Dict[str, str] = {}
        for tx in self.model_transforms:
            mapping = tx.checkpoint_key_mapping(model)
            if not mapping:
                continue
            for cur, ckpt in mapping.items():
                if cur in merged and merged[cur] != ckpt:
                    raise OLMoConfigurationError(
                        f"Conflicting checkpoint key mappings for {cur!r}: "
                        f"{merged[cur]!r} vs. {ckpt!r}. Reorder or reconcile."
                    )
                merged[cur] = ckpt
        return merged

    def install_gradient_callback(self, trainer: "Trainer") -> None:
        """Auto-install a single :class:`PEFTGradientCallback` carrying
        all gradient transforms in declared order. No-op when there are
        no gradient transforms."""
        if not self.gradient_transforms:
            return

        from ...train.callbacks.peft_gradient import PEFTGradientCallback

        trainer.add_callback(
            "peft_gradient",
            PEFTGradientCallback(transforms=list(self.gradient_transforms)),
        )


__all__ = [
    "ModelTransform",
    "GradientTransform",
    "GradientMaskTransform",
    "PEFTConfig",
    "ModelTransformReport",
    "TrainabilityReport",
    "stable_hash_u64",
]
