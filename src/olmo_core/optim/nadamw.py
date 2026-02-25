from dataclasses import dataclass
from typing import Tuple, Type

import torch
from torch.optim.optimizer import Optimizer

from ..distributed.utils import get_local_tensor
from .config import OptimConfig


class NAdamW(Optimizer):
    """
    NAdamW: Adam with Nesterov momentum and decoupled weight decay.

    References:
        - Dozat (2016), "Incorporating Nesterov Momentum into Adam"
        - Loshchilov & Hutter (2019), "Decoupled Weight Decay Regularization"
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ):
        assert lr >= 0.0
        assert all(0.0 <= beta <= 1.0 for beta in betas)
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            with torch.enable_grad():
                closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = get_local_tensor(p.grad)
                p_local = get_local_tensor(p)

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = torch.zeros((), dtype=torch.float32, device=p.device)
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)

                exp_avg = get_local_tensor(state["exp_avg"])
                exp_avg_sq = get_local_tensor(state["exp_avg_sq"])
                step = state["step"]
                step.add_(1)

                # Decoupled weight decay
                if weight_decay != 0:
                    p_local.mul_(1.0 - lr * weight_decay)

                # Update biased first moment
                exp_avg.lerp_(grad, 1.0 - beta1)

                # Update biased second moment
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                # Bias corrections (all tensor ops to avoid host-device sync)
                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step

                step_size = lr / bias_correction1

                # Nesterov correction: apply EMA formula AGAIN (non-in-place)
                exp_avg_corrected = exp_avg.lerp(grad, 1.0 - beta1)

                # Denominator: sqrt(v / bc2) + eps
                denom = (exp_avg_sq.sqrt() / bias_correction2.sqrt()).add_(eps)

                # Parameter update
                p_local.add_(-step_size * (exp_avg_corrected / denom))


@dataclass
class NAdamWConfig(OptimConfig):
    """
    Configuration class for building a :class:`NAdamW` optimizer.
    """

    lr: float = 1e-3
    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.01

    @classmethod
    def optimizer(cls) -> Type[NAdamW]:
        return NAdamW
