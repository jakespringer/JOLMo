from dataclasses import dataclass
from typing import Tuple, Type

import torch
from torch.optim.optimizer import Optimizer

from ..distributed.utils import get_local_tensor
from .config import OptimConfig


class Sophia(Optimizer):
    """
    Sophia: Second-order Clipped Stochastic Optimization.

    Uses gradient momentum + periodic diagonal Hessian estimation
    + element-wise clipping of the Hessian-preconditioned update.

    The Hessian estimation requires an external call to :meth:`update_hessian`
    or :meth:`update_hessian_from_estimates` every ``hessian_update_interval``
    steps from the training loop.

    Reference:
        Liu et al. (2024), "Sophia: A Scalable Stochastic Second-Order
        Optimizer for Language Model Pre-training"
    """

    def __init__(
        self,
        params,
        lr: float = 6e-4,
        betas: Tuple[float, float] = (0.965, 0.99),
        rho: float = 0.04,
        weight_decay: float = 0.1,
        eps: float = 1e-15,
        hessian_update_interval: int = 10,
    ):
        assert lr >= 0.0
        assert all(0.0 <= beta < 1.0 for beta in betas)
        assert rho > 0.0
        # Store for external use by training loop / callback
        self.hessian_update_interval = hessian_update_interval
        defaults = dict(lr=lr, betas=betas, rho=rho, weight_decay=weight_decay, eps=eps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def update_hessian(self):
        """
        Update diagonal Hessian estimate using the Gauss-Newton-Bartlett estimator.

        Call this AFTER a backward pass on the resampled-label loss.
        The gradients in ``p.grad`` should be from the GNB forward/backward pass.
        """
        for group in self.param_groups:
            _, beta2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = get_local_tensor(p.grad)
                state = self.state[p]
                self._init_state(state, p)

                hessian = get_local_tensor(state["hessian"])
                hessian.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

    @torch.no_grad()
    def update_hessian_from_estimates(self, estimates):
        """
        Update diagonal Hessian estimate from precomputed Hutchinson estimates.

        Args:
            estimates: list of tensors, one per parameter, containing ``u * (H @ u)``.
        """
        idx = 0
        for group in self.param_groups:
            _, beta2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                self._init_state(state, p)

                hessian = get_local_tensor(state["hessian"])
                h_hat = estimates[idx]
                hessian.mul_(beta2).add_(h_hat, alpha=1.0 - beta2)
                idx += 1

    def _init_state(self, state, p):
        if len(state) == 0:
            state["step"] = torch.zeros((), dtype=torch.float32, device=p.device)
            state["exp_avg"] = torch.zeros_like(p)
            state["hessian"] = torch.zeros_like(p)

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            with torch.enable_grad():
                closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, _ = group["betas"]
            rho = group["rho"]
            weight_decay = group["weight_decay"]
            eps = group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = get_local_tensor(p.grad)
                p_local = get_local_tensor(p)
                state = self.state[p]
                self._init_state(state, p)

                exp_avg = get_local_tensor(state["exp_avg"])
                hessian = get_local_tensor(state["hessian"])
                step = state["step"]
                step.add_(1)

                # Decoupled weight decay
                if weight_decay != 0:
                    p_local.mul_(1.0 - lr * weight_decay)

                # Update gradient EMA
                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)

                # Clipped Hessian-preconditioned update:
                #   ratio = min(|m| / (rho * h + eps), 1)
                #   update = sign(m) * ratio
                ratio = (exp_avg.abs() / (rho * hessian + eps)).clamp_(max=1.0)
                p_local.addcmul_(exp_avg.sign(), ratio, value=-lr)


@dataclass
class SophiaConfig(OptimConfig):
    """
    Configuration class for building a :class:`Sophia` optimizer.

    Note: Sophia requires training loop modifications to call
    ``optimizer.update_hessian()`` every ``hessian_update_interval`` steps
    after a GNB-style backward pass. The ``hessian_update_interval`` is
    passed through to the optimizer constructor and stored as
    ``optimizer.hessian_update_interval`` for the training loop to read.
    """

    lr: float = 6e-4
    betas: Tuple[float, float] = (0.965, 0.99)
    rho: float = 0.04
    weight_decay: float = 0.1
    eps: float = 1e-15
    hessian_update_interval: int = 10

    @classmethod
    def optimizer(cls) -> Type[Sophia]:
        return Sophia
