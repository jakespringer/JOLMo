from dataclasses import dataclass
from typing import Tuple, Type

import torch
from torch.optim.optimizer import Optimizer

from ..distributed.utils import get_local_tensor
from .config import OptimConfig


class Mars(Optimizer):
    """
    MARS: Make vARiance Reduction Shine.

    Variance-reduced AdamW variant that uses the STORM technique to
    correct gradient estimates before feeding them into Adam-style updates.

    Reference:
        Yuan et al. (2024), "MARS: Unleashing the Power of Variance
        Reduction for Training Large Models"
    """

    def __init__(
        self,
        params,
        lr: float = 3e-3,
        betas: Tuple[float, float] = (0.95, 0.99),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        gamma: float = 0.025,
        optimize_1d: bool = False,
        lr_1d_factor: float = 0.5,
        betas_1d: Tuple[float, float] = (0.9, 0.95),
        weight_decay_1d: float = 0.1,
    ):
        assert lr >= 0.0
        assert all(0.0 <= beta <= 1.0 for beta in betas)
        defaults = dict(
            lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
            gamma=gamma, optimize_1d=optimize_1d,
            lr_1d_factor=lr_1d_factor, betas_1d=betas_1d,
            weight_decay_1d=weight_decay_1d,
        )
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
            gamma = group["gamma"]
            optimize_1d = group["optimize_1d"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = get_local_tensor(p.grad)
                p_local = get_local_tensor(p)
                is_2d = (grad.dim() >= 2)

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = torch.zeros((), dtype=torch.float32, device=p.device)
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                    state["last_grad"] = torch.zeros_like(p)

                exp_avg = get_local_tensor(state["exp_avg"])
                exp_avg_sq = get_local_tensor(state["exp_avg_sq"])
                last_grad = get_local_tensor(state["last_grad"])
                step = state["step"]
                step.add_(1)

                if is_2d or optimize_1d:
                    # --- MARS variance-reduced correction ---
                    # c_t = g_t + gamma * (beta1 / (1 - beta1)) * (g_t - g_prev)
                    correction = gamma * (beta1 / (1.0 - beta1))
                    c_t = grad + correction * (grad - last_grad)

                    # Gradient clipping on corrected gradient
                    # Use clamp instead of Python if to avoid host-device sync
                    c_t_norm = c_t.norm().clamp_(min=1.0)
                    c_t = c_t / c_t_norm

                    # First moment update
                    exp_avg.mul_(beta1).add_(c_t, alpha=1.0 - beta1)

                    # Second moment update (on corrected gradient)
                    exp_avg_sq.mul_(beta2).addcmul_(c_t, c_t, value=1.0 - beta2)

                    # Bias correction (tensor ops to avoid host-device sync)
                    bias_correction1 = 1 - beta1 ** step
                    bias_correction2 = 1 - beta2 ** step

                    denom = (exp_avg_sq.sqrt() / bias_correction2.sqrt()).add_(eps)

                    # NOTE: MARS uses coupled weight decay (L2 regularization in the
                    # update, not decoupled). This follows the official MARS paper and
                    # implementation, but differs from JOLMo's usual decoupled pattern.
                    update = exp_avg / (denom * bias_correction1)
                    p_local.add_(update + weight_decay * p_local, alpha=-lr)
                else:
                    # Fallback: plain AdamW for 1D params (decoupled weight decay)
                    beta1_1d, beta2_1d = group["betas_1d"]
                    wd_1d = group["weight_decay_1d"]
                    lr_1d = lr * group["lr_1d_factor"]

                    # Decoupled weight decay for 1D params
                    if wd_1d != 0:
                        p_local.mul_(1.0 - lr_1d * wd_1d)

                    exp_avg.mul_(beta1_1d).add_(grad, alpha=1.0 - beta1_1d)
                    exp_avg_sq.mul_(beta2_1d).addcmul_(grad, grad, value=1.0 - beta2_1d)

                    bias_correction1 = 1 - beta1_1d ** step
                    bias_correction2 = 1 - beta2_1d ** step

                    denom = (exp_avg_sq.sqrt() / bias_correction2.sqrt()).add_(eps)

                    # Use explicit tensor ops — bias_correction1 is a tensor,
                    # and value= args to addcdiv_ must be Python numbers
                    step_size_1d = lr_1d / bias_correction1
                    p_local.add_(-(step_size_1d * exp_avg / denom))

                # Store gradient for next step
                last_grad.copy_(grad)


@dataclass
class MarsConfig(OptimConfig):
    """
    Configuration class for building a :class:`Mars` optimizer.
    """

    lr: float = 3e-3
    betas: Tuple[float, float] = (0.95, 0.99)
    eps: float = 1e-8
    weight_decay: float = 0.01
    gamma: float = 0.025
    optimize_1d: bool = False
    lr_1d_factor: float = 0.5
    betas_1d: Tuple[float, float] = (0.9, 0.95)
    weight_decay_1d: float = 0.1

    @classmethod
    def optimizer(cls) -> Type[Mars]:
        return Mars
