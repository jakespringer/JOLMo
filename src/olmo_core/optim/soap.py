import logging
from dataclasses import dataclass
from itertools import chain
from typing import List, Optional, Tuple, Type

import torch
from torch.optim.optimizer import Optimizer

from ..distributed.utils import get_local_tensor
from .config import OptimConfig

log = logging.getLogger(__name__)


class Soap(Optimizer):
    """
    SOAP: ShampoO with Adam in the Preconditioner's eigenbasis.

    Rotates gradients into Shampoo's eigenbasis, runs Adam there,
    and rotates back. Eigenbases are refreshed every ``precondition_frequency`` steps
    via power iteration + QR.

    Reference:
        Vyas et al. (2024), "SOAP: Improving and Stabilizing Shampoo using Adam"
    """

    def __init__(
        self,
        params,
        lr: float = 3e-3,
        betas: Tuple[float, float] = (0.95, 0.95),
        shampoo_beta: Optional[float] = None,
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        precondition_frequency: int = 10,
        max_precond_dim: int = 10000,
        merge_dims: bool = False,
        precondition_1d: bool = False,
        correct_bias: bool = True,
    ):
        if shampoo_beta is None:
            shampoo_beta = betas[1]
        defaults = dict(
            lr=lr,
            betas=betas,
            shampoo_beta=shampoo_beta,
            eps=eps,
            weight_decay=weight_decay,
            precondition_frequency=precondition_frequency,
            max_precond_dim=max_precond_dim,
            merge_dims=merge_dims,
            precondition_1d=precondition_1d,
            correct_bias=correct_bias,
        )
        super().__init__(params, defaults)

    def _merge_dims(self, grad: torch.Tensor, max_precond_dim: int) -> torch.Tensor:
        """Merge dimensions of gradient tensor until each dim <= max_precond_dim."""
        shape = grad.shape
        new_shape: List[int] = []
        curr_shape = 1
        for sh in shape:
            temp_shape = curr_shape * sh
            if temp_shape > max_precond_dim:
                if curr_shape > 1:
                    new_shape.append(curr_shape)
                    curr_shape = sh
                else:
                    new_shape.append(sh)
                    curr_shape = 1
            else:
                curr_shape = temp_shape
        if curr_shape > 1 or len(new_shape) == 0:
            new_shape.append(curr_shape)
        return grad.reshape(new_shape)

    def _init_preconditioner(self, grad: torch.Tensor, state: dict, group: dict) -> None:
        """Initialize Gram matrices for each dimension."""
        state["GG"] = []
        precondition_1d = group["precondition_1d"]
        max_precond_dim = group["max_precond_dim"]

        if grad.dim() == 1:
            if not precondition_1d or grad.shape[0] > max_precond_dim:
                state["GG"].append([])
            else:
                state["GG"].append(torch.zeros(grad.shape[0], grad.shape[0], device=grad.device))
        else:
            work_grad = self._merge_dims(grad, max_precond_dim) if group["merge_dims"] else grad
            for sh in work_grad.shape:
                if sh > max_precond_dim:
                    state["GG"].append([])
                else:
                    state["GG"].append(torch.zeros(sh, sh, device=grad.device))

        state["Q"] = None
        state["precondition_frequency"] = group["precondition_frequency"]
        state["shampoo_beta"] = group["shampoo_beta"]

    def _project(self, grad: torch.Tensor, state: dict, group: dict) -> torch.Tensor:
        """Project gradient into eigenbasis: G' = Q_L^T @ G @ Q_R."""
        original_shape = grad.shape
        if group["merge_dims"]:
            grad = self._merge_dims(grad, group["max_precond_dim"])
        for mat in state["Q"]:
            if len(mat) > 0:
                grad = torch.tensordot(grad, mat, dims=[[0], [0]])
            else:
                permute_order = list(range(1, len(grad.shape))) + [0]
                grad = grad.permute(permute_order)
        if group["merge_dims"]:
            grad = grad.reshape(original_shape)
        return grad

    def _project_back(self, grad: torch.Tensor, state: dict, group: dict) -> torch.Tensor:
        """Project from eigenbasis back: N = Q_L @ N' @ Q_R^T."""
        original_shape = grad.shape
        if group["merge_dims"]:
            grad = self._merge_dims(grad, group["max_precond_dim"])
        for mat in state["Q"]:
            if len(mat) > 0:
                grad = torch.tensordot(grad, mat, dims=[[0], [1]])
            else:
                permute_order = list(range(1, len(grad.shape))) + [0]
                grad = grad.permute(permute_order)
        if group["merge_dims"]:
            grad = grad.reshape(original_shape)
        return grad

    def _update_preconditioner(self, grad: torch.Tensor, state: dict, group: dict) -> None:
        """Update Gram matrices and refresh eigenbases periodically."""
        max_precond_dim = group["max_precond_dim"]
        precondition_1d = group["precondition_1d"]
        merge_dims = group["merge_dims"]

        # Before updating eigenbases, project exp_avg back to original space
        if state["Q"] is not None:
            local_exp_avg = get_local_tensor(state["exp_avg"])
            state["exp_avg"] = self._project_back(local_exp_avg, state, group)

        # Update Gram matrices: L = beta_s*L + (1-beta_s)*G@G^T, etc.
        if grad.dim() == 1:
            if precondition_1d and grad.shape[0] <= max_precond_dim:
                state["GG"][0].lerp_(grad.unsqueeze(1) @ grad.unsqueeze(0), 1 - state["shampoo_beta"])
        else:
            work_grad = self._merge_dims(grad, max_precond_dim) if merge_dims else grad
            for idx, sh in enumerate(work_grad.shape):
                if sh <= max_precond_dim:
                    dims_except_idx = list(chain(range(idx), range(idx + 1, len(work_grad.shape))))
                    outer_product = torch.tensordot(work_grad, work_grad, dims=[dims_except_idx, dims_except_idx])
                    state["GG"][idx].lerp_(outer_product, 1 - state["shampoo_beta"])

        # Compute eigenbases
        if state["Q"] is None:
            state["Q"] = self._get_eigenbasis_eigh(state["GG"])
        elif state["step"] > 0 and state["step"] % state["precondition_frequency"] == 0:
            state["Q"] = self._get_eigenbasis_qr(state, max_precond_dim, merge_dims)

        # Re-project exp_avg into (possibly updated) eigenbasis
        if state["step"] > 0:
            local_exp_avg = get_local_tensor(state["exp_avg"])
            state["exp_avg"] = self._project(local_exp_avg, state, group)

    def _get_eigenbasis_eigh(self, GG: list) -> list:
        """Full eigendecomposition for initial eigenbasis."""
        result = []
        for m in GG:
            if len(m) == 0:
                result.append([])
                continue
            m_float = m.float()
            try:
                _, Q = torch.linalg.eigh(m_float + 1e-30 * torch.eye(m_float.shape[0], device=m_float.device))
            except Exception:
                _, Q = torch.linalg.eigh(
                    m_float.double() + 1e-30 * torch.eye(m_float.shape[0], device=m_float.device, dtype=torch.float64)
                )
                Q = Q.float()
            Q = torch.flip(Q, [1])  # descending eigenvalue order
            result.append(Q.to(dtype=m.dtype))
        return result

    def _get_eigenbasis_qr(self, state: dict, max_precond_dim: int, merge_dims: bool) -> list:
        """Power iteration + QR for eigenbasis refresh."""
        result = []
        exp_avg_sq = get_local_tensor(state["exp_avg_sq"])
        orig_shape = exp_avg_sq.shape
        if merge_dims:
            exp_avg_sq = self._merge_dims(exp_avg_sq, max_precond_dim)

        for ind, (m, o) in enumerate(zip(state["GG"], state["Q"])):
            if len(m) == 0:
                result.append([])
                continue
            m_float = m.float()
            o_float = o.float()

            # Sort by estimated eigenvalues
            est_eig = torch.diag(o_float.T @ m_float @ o_float)
            sort_idx = torch.argsort(est_eig, descending=True)
            exp_avg_sq = exp_avg_sq.index_select(ind, sort_idx)
            o_float = o_float[:, sort_idx]

            # Power iteration + QR
            power_iter = m_float @ o_float
            Q, _ = torch.linalg.qr(power_iter)
            result.append(Q.to(dtype=m.dtype))

        if merge_dims:
            exp_avg_sq = exp_avg_sq.reshape(orig_shape)
        state["exp_avg_sq"] = exp_avg_sq
        return result

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            with torch.enable_grad():
                closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = get_local_tensor(p.grad)
                p_local = get_local_tensor(p)
                state = self.state[p]

                if "step" not in state:
                    state["step"] = torch.zeros((), dtype=torch.float32, device=p.device)

                if "exp_avg" not in state:
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)

                if "Q" not in state:
                    self._init_preconditioner(grad, state, group)
                    self._update_preconditioner(grad, state, group)
                    continue  # skip first step (need eigenbasis)

                # Project gradient into eigenbasis
                grad_projected = self._project(grad, state, group)

                exp_avg = get_local_tensor(state["exp_avg"])
                exp_avg_sq = get_local_tensor(state["exp_avg_sq"])
                beta1, beta2 = group["betas"]
                step = state["step"]
                step.add_(1)

                # Adam updates in rotated space
                exp_avg.mul_(beta1).add_(grad_projected, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).add_(grad_projected.square(), alpha=1.0 - beta2)

                denom = exp_avg_sq.sqrt().add_(group["eps"])

                step_size = group["lr"]
                if group["correct_bias"]:
                    bc1 = 1.0 - beta1 ** step
                    bc2 = 1.0 - beta2 ** step
                    step_size = step_size * bc2.sqrt() / bc1

                # Project back to original space
                norm_grad = self._project_back(exp_avg / denom, state, group)

                # Parameter update
                p_local.sub_(step_size * norm_grad)

                # Decoupled weight decay
                if group["weight_decay"] > 0:
                    p_local.sub_(group["lr"] * group["weight_decay"] * p_local)

                # Update Gram matrices & refresh eigenbases
                self._update_preconditioner(grad, state, group)


@dataclass
class SoapConfig(OptimConfig):
    """
    Configuration class for building a :class:`Soap` optimizer.
    """

    lr: float = 3e-3
    betas: Tuple[float, float] = (0.95, 0.95)
    shampoo_beta: Optional[float] = None
    """Beta for Gram matrix EMA. If None, defaults to betas[1]."""
    eps: float = 1e-8
    weight_decay: float = 0.01
    precondition_frequency: int = 10
    max_precond_dim: int = 10000
    merge_dims: bool = False
    precondition_1d: bool = False
    correct_bias: bool = True

    @classmethod
    def optimizer(cls) -> Type[Soap]:
        return Soap
