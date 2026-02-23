import logging
from dataclasses import dataclass
from fnmatch import fnmatch
from typing import Any, Dict, List, Optional, Set, Tuple, Type, cast

import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer

from ..utils import move_to_device
from .config import INITIAL_LR_FIELD, LR_FIELD, OptimConfig

log = logging.getLogger(__name__)


class Muon(Optimizer):
    """
    Composite optimizer that wraps :class:`torch.optim.Muon` for 2D hidden-layer
    weight matrices and :class:`torch.optim.AdamW` for everything else (embeddings,
    norms, biases, LM head).

    Param groups must be tagged with ``_muon: bool`` to indicate which sub-optimizer
    handles them. This is done automatically by :meth:`MuonConfig.build`.

    Internally, the two sub-optimizers **share** this wrapper's ``self.state`` dict
    and reference the same param group dict objects, so:

    - The LR scheduler can set ``group["lr"]`` on the wrapper's param groups and
      the sub-optimizers see the change immediately.
    - Checkpointing reads from the wrapper's ``state``/``param_groups`` and gets
      all optimizer state from both sub-optimizers.
    - ``load_state_dict`` on the wrapper updates the shared dicts in-place, so
      sub-optimizers see the loaded state.
    """

    def __init__(
        self,
        params,
        *,
        lr: float = 0.02,
        weight_decay: float = 0.0,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        ns_coefficients: Tuple[float, float, float] = (3.4445, -4.775, 2.0315),
        eps: float = 1e-7,
        adjust_lr_fn: Optional[str] = None,
        adamw_lr: float = 3e-4,
        adamw_betas: Tuple[float, float] = (0.9, 0.95),
        adamw_eps: float = 1e-8,
        adamw_weight_decay: float = 0.1,
        adamw_fused: bool = True,
    ):
        # Validate adjust_lr_fn before anything else.
        if adjust_lr_fn is not None and adjust_lr_fn not in ("original", "match_rms_adamw"):
            raise ValueError(
                f"adjust_lr_fn must be None, 'original', or 'match_rms_adamw', got {adjust_lr_fn!r}"
            )

        # Use lr and weight_decay as defaults (they'll be overridden per-group by
        # MuonConfig.build, but this satisfies Optimizer's requirement for defaults).
        defaults: Dict[str, Any] = dict(lr=lr, weight_decay=weight_decay)
        super().__init__(params, defaults)

        # Partition param groups by _muon tag.
        muon_groups = [g for g in self.param_groups if g.get("_muon", False)]
        adamw_groups = [g for g in self.param_groups if not g.get("_muon", False)]

        # Validate: Muon groups must contain only 2D parameters.
        for group in muon_groups:
            for p in group["params"]:
                if p.ndim != 2:
                    raise ValueError(
                        f"Muon group contains a parameter with ndim={p.ndim} (shape {tuple(p.shape)}). "
                        "Muon requires all parameters in _muon=True groups to be 2D."
                    )

        # --- Build torch.optim.Muon sub-optimizer ---
        muon_params = [p for g in muon_groups for p in g["params"]]
        if muon_params:
            self._muon = torch.optim.Muon(
                muon_params,
                lr=lr,
                weight_decay=weight_decay,
                momentum=momentum,
                nesterov=nesterov,
                ns_coefficients=ns_coefficients,
                eps=eps,
                ns_steps=ns_steps,
                adjust_lr_fn=adjust_lr_fn,
            )
            # Copy any per-group defaults that torch.optim.Muon added (momentum,
            # nesterov, ns_coefficients, etc.) into our wrapper's muon group dicts.
            # This ensures the sub-optimizer's step() can find them when we share
            # param_groups below.
            if self._muon.param_groups:
                muon_defaults = {
                    k: v
                    for k, v in self._muon.param_groups[0].items()
                    if k not in ("params", "lr", "weight_decay")
                }
                for g in muon_groups:
                    for k, v in muon_defaults.items():
                        g.setdefault(k, v)

            # Share state and param_groups with the wrapper.
            self._muon.state = self.state
            self._muon.param_groups = muon_groups
        else:
            self._muon = None

        # --- Build torch.optim.AdamW sub-optimizer ---
        adamw_params = [p for g in adamw_groups for p in g["params"]]
        if adamw_params:
            self._adamw = torch.optim.AdamW(
                adamw_params,
                lr=adamw_lr,
                betas=adamw_betas,
                eps=adamw_eps,
                weight_decay=adamw_weight_decay,
                fused=adamw_fused,
            )
            # Copy AdamW-specific per-group defaults into our wrapper's adamw groups.
            if self._adamw.param_groups:
                adamw_defaults = {
                    k: v
                    for k, v in self._adamw.param_groups[0].items()
                    if k not in ("params", "lr", "weight_decay")
                }
                for g in adamw_groups:
                    for k, v in adamw_defaults.items():
                        g.setdefault(k, v)

            # Share state and param_groups with the wrapper.
            self._adamw.state = self.state
            self._adamw.param_groups = adamw_groups
        else:
            self._adamw = None

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        if self._muon is not None:
            self._muon.step()
        if self._adamw is not None:
            self._adamw.step()

        return loss

    def zero_grad(self, set_to_none: bool = True) -> None:
        # Inherited Optimizer.zero_grad iterates self.param_groups (all groups).
        super().zero_grad(set_to_none=set_to_none)


@dataclass
class MuonConfig(OptimConfig):
    """
    Configuration class for building a composite :class:`Muon` optimizer.

    Uses :class:`torch.optim.Muon` (Newton-Schulz orthogonalization) for 2D hidden
    weight matrices, and :class:`torch.optim.AdamW` for everything else (embeddings,
    norms, biases, LM head).
    """

    # --- Muon hyperparameters (2D hidden weight matrices) ---
    lr: float = 0.02
    weight_decay: float = 0.0
    momentum: float = 0.95
    nesterov: bool = True
    ns_steps: int = 5
    ns_coefficients: Tuple[float, float, float] = (3.4445, -4.775, 2.0315)
    eps: float = 1e-7
    adjust_lr_fn: Optional[str] = None

    # --- AdamW hyperparameters (embeddings, norms, biases, LM head) ---
    adamw_lr: float = 3e-4
    adamw_betas: Tuple[float, float] = (0.9, 0.95)
    adamw_eps: float = 1e-8
    adamw_weight_decay: float = 0.1
    adamw_fused: bool = True

    # --- Param classification ---
    muon_on_lm_head: bool = False

    @classmethod
    def optimizer(cls) -> Type[Muon]:
        return Muon

    def _is_muon_param(self, name: str, param: torch.Tensor) -> bool:
        """
        Decide whether a parameter should be optimized by Muon (True) or AdamW (False).

        Rules:
        1. Non-2D params always go to AdamW (norms, biases, scalars).
        2. Embeddings always go to AdamW (even though they are 2D).
        3. LM head goes to AdamW by default (configurable via ``muon_on_lm_head``).
        4. All other 2D params go to Muon.
        """
        if param.ndim != 2:
            return False
        if fnmatch(name, "embeddings.*"):
            return False
        if not self.muon_on_lm_head and fnmatch(name, "lm_head.*"):
            return False
        return True

    def build(self, model: nn.Module, strict: bool = True) -> Muon:
        """
        Build the composite Muon optimizer.

        Overrides :meth:`OptimConfig.build` because we need to:
        1. Classify parameters into Muon vs AdamW before creating groups.
        2. Tag groups with ``_muon`` and set different ``lr`` values per type.
        3. Pass different kwargs to the internal sub-optimizers.
        """
        # --- Step 1: Build a set of param tensors that should use Muon ---
        muon_param_set: Set[int] = set()
        for name, p in model.named_parameters():
            if p.requires_grad and self._is_muon_param(name, p):
                muon_param_set.add(id(p))

        # --- Step 2: Get base param groups from OptimConfig (handles group_overrides) ---
        base_groups = self.build_groups(model, strict=strict)

        # --- Step 3: Split each base group into muon / adamw subgroups ---
        final_groups: List[Dict[str, Any]] = []

        for group in base_groups:
            params = group.pop("params")
            group_opts = dict(group)  # remaining opts from overrides (e.g. weight_decay: 0.0)

            muon_params = [p for p in params if id(p) in muon_param_set]
            adamw_params = [p for p in params if id(p) not in muon_param_set]

            if muon_params:
                muon_group: Dict[str, Any] = {"params": muon_params, "_muon": True}
                muon_group.update(group_opts)
                # Set muon defaults, but let explicit overrides take precedence.
                muon_group.setdefault("lr", self.lr)
                muon_group.setdefault("weight_decay", self.weight_decay)
                final_groups.append(muon_group)

            if adamw_params:
                adamw_group: Dict[str, Any] = {"params": adamw_params, "_muon": False}
                adamw_group.update(group_opts)
                # Set adamw defaults, but let explicit overrides take precedence.
                adamw_group.setdefault("lr", self.adamw_lr)
                adamw_group.setdefault("weight_decay", self.adamw_weight_decay)
                final_groups.append(adamw_group)

        if not final_groups:
            raise ValueError("No trainable parameters found when building Muon optimizer.")

        # --- Step 4: Create the composite optimizer ---
        optim = Muon(
            final_groups,
            lr=self.lr,
            weight_decay=self.weight_decay,
            momentum=self.momentum,
            nesterov=self.nesterov,
            ns_steps=self.ns_steps,
            ns_coefficients=self.ns_coefficients,
            eps=self.eps,
            adjust_lr_fn=self.adjust_lr_fn,
            adamw_lr=self.adamw_lr,
            adamw_betas=self.adamw_betas,
            adamw_eps=self.adamw_eps,
            adamw_weight_decay=self.adamw_weight_decay,
            adamw_fused=self.adamw_fused,
        )

        # --- Step 5: Set initial_lr on each group (same pattern as base build()) ---
        fixed_fields_per_group: List[Dict[str, Any]] = [{} for _ in optim.param_groups]
        for fixed_fields, group in zip(fixed_fields_per_group, optim.param_groups):
            lr: Optional[float] = None
            if LR_FIELD in group:
                lr = group[LR_FIELD]
            if lr is not None:
                if self.compile:
                    group[LR_FIELD] = move_to_device(torch.tensor(lr), self.device)
                else:
                    group[LR_FIELD] = lr
                group.setdefault(INITIAL_LR_FIELD, lr)

            for k in self.fixed_fields:
                if k in group:
                    fixed_fields[k] = group[k]

        # --- Step 6: Log group info ---
        log.info(
            f"Building {self.optimizer().__name__} optimizer with {len(optim.param_groups)} param group(s)..."
        )
        for g_idx, group in enumerate(optim.param_groups):
            group_fields_list = "\n - ".join(
                [f"{k}: {v}" for k, v in group.items() if k != "params"]
            )
            group_type = "Muon" if group.get("_muon") else "AdamW"
            if group_fields_list:
                log.info(
                    f"Group {g_idx} ({group_type}), {len(group['params'])} parameter(s):\n - {group_fields_list}"
                )
            else:
                log.info(f"Group {g_idx} ({group_type}), {len(group['params'])} parameter(s)")

        # --- Step 7: Optionally compile ---
        if self.compile:
            log.info("Compiling optimizer step...")
            optim.step = torch.compile(optim.step)

        # --- Step 8: Register hook to reset fixed fields after checkpoint load ---
        def reset_fixed_fields(opt: torch.optim.Optimizer):
            for ff, group in zip(fixed_fields_per_group, opt.param_groups):
                group.update(ff)

        optim.register_load_state_dict_post_hook(reset_fixed_fields)

        return cast(Muon, optim)
