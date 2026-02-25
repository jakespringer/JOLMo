from dataclasses import dataclass
from typing import Tuple, Type

import torch
from torch.optim.optimizer import Optimizer

from ..distributed.utils import get_local_tensor
from .config import OptimConfig


class CautiousAdamW(Optimizer):
    """
    Cautious AdamW: AdamW with sign-alignment masking.

    Zeros out update coordinates where the momentum and current gradient
    disagree in sign, with normalization to preserve update magnitude.

    Reference:
        Liang et al. (2024), "Cautious Optimizers: Improving Training
        with One Line of Code"
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
        weight_decay: float = 0.1,
        mask_eps: float = 1e-3,
    ):
        assert lr >= 0.0
        assert all(0.0 <= beta <= 1.0 for beta in betas)
        defaults = dict(
            lr=lr, betas=betas, eps=eps,
            weight_decay=weight_decay, mask_eps=mask_eps,
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
            mask_eps = group["mask_eps"]

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

                # Denominator
                denom = (exp_avg_sq.sqrt() / bias_correction2.sqrt()).add_(eps)

                # Cautious mask: keep only coordinates where momentum
                # and gradient agree in sign
                mask = (exp_avg * grad > 0).to(grad.dtype)
                mask.div_(mask.mean().clamp_(min=mask_eps))
                masked_exp_avg = exp_avg * mask

                # Parameter update (use explicit tensor ops — step_size is a tensor,
                # and value= args to addcdiv_ must be Python numbers)
                p_local.add_(-step_size * (masked_exp_avg / denom))


@dataclass
class CautiousAdamWConfig(OptimConfig):
    """
    Configuration class for building a :class:`CautiousAdamW` optimizer.
    """

    lr: float = 1e-3
    betas: Tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    weight_decay: float = 0.1
    mask_eps: float = 1e-3

    @classmethod
    def optimizer(cls) -> Type[CautiousAdamW]:
        return CautiousAdamW
