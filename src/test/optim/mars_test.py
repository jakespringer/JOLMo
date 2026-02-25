import copy

import pytest
import torch
import torch.nn as nn

from olmo_core.distributed.checkpoint import (
    load_model_and_optim_state,
    save_model_and_optim_state,
)
from olmo_core.optim import AdamWConfig, Mars, MarsConfig, OptimGroupOverride
from olmo_core.testing import DEVICES


class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.wte = nn.Embedding(1024, 16)
        self.fc1 = nn.Linear(16, 32)
        self.fc2 = nn.Linear(32, 16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.wte(x)
        x = self.fc1(x)
        x = torch.relu(x)
        return self.fc2(x)


def test_config_builds_correctly():
    config = MarsConfig(lr=3e-3)
    model = MyModel()
    optim = config.build(model)
    assert isinstance(optim, Mars)
    assert len(optim.param_groups) == 1
    for group in optim.param_groups:
        assert "initial_lr" in group


def test_config_with_group_overrides():
    config = MarsConfig(
        group_overrides=[OptimGroupOverride(params=["wte.*"], opts=dict(weight_decay=0.0))]
    )
    model = MyModel()
    optim = config.build(model)
    assert isinstance(optim, Mars)
    assert len(optim.param_groups) == 2
    assert optim.param_groups[0]["weight_decay"] == 0.0
    assert len(optim.param_groups[0]["params"]) == 1
    assert len(optim.param_groups[1]["params"]) == len(list(model.parameters())) - 1

    for group in optim.param_groups:
        assert "initial_lr" in group


@pytest.mark.parametrize("device", DEVICES)
def test_optimizer_step(device: torch.device):
    config = MarsConfig()
    model = MyModel().train().to(device)
    optim = config.build(model)

    initial_params = {n: p.clone() for n, p in model.named_parameters()}

    optim.zero_grad(set_to_none=True)
    model(torch.randint(0, 1024, (2, 8), device=device).int()).sum().backward()
    optim.step()

    any_changed = False
    for n, p in model.named_parameters():
        if not torch.equal(p, initial_params[n]):
            any_changed = True
    assert any_changed


@pytest.mark.parametrize("device", DEVICES)
def test_gamma_zero_matches_adamw(device: torch.device):
    """With gamma=0 and optimize_1d=True, MARS should behave similarly to AdamW on 2D params.

    Note: MARS uses coupled weight decay (L2) while AdamW uses decoupled weight decay,
    so exact match is only expected with weight_decay=0.
    """
    torch.manual_seed(42)

    model_mars = MyModel().to(device)
    model_adamw = copy.deepcopy(model_mars).to(device)

    cfg_common = dict(lr=1e-3, betas=(0.9, 0.999), weight_decay=0.0, eps=1e-8)
    optim_mars = MarsConfig(
        gamma=0.0, optimize_1d=True,
        lr_1d_factor=1.0, betas_1d=(0.9, 0.999), weight_decay_1d=0.0,
        **cfg_common,  # type: ignore[arg-type]
    ).build(model_mars)
    optim_adamw = AdamWConfig(fused=False, **cfg_common).build(model_adamw)  # type: ignore[arg-type]

    for _ in range(3):
        inp = torch.randint(0, 128, (4, 8), device=device)

        optim_mars.zero_grad(set_to_none=True)
        model_mars(inp).sum().backward()
        optim_mars.step()

        optim_adamw.zero_grad(set_to_none=True)
        model_adamw(inp).sum().backward()
        optim_adamw.step()

    # With gamma=0, optimize_1d=True, and weight_decay=0, MARS reduces to AdamW
    for p_m, p_a in zip(model_mars.parameters(), model_adamw.parameters()):
        torch.testing.assert_close(p_m, p_a, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("device", DEVICES)
def test_variance_reduction_active(device: torch.device):
    """Verify last_grad is populated after a step (variance reduction state is maintained)."""
    config = MarsConfig()
    model = MyModel().train().to(device)
    optim = config.build(model)

    optim.zero_grad(set_to_none=True)
    model(torch.randint(0, 1024, (2, 8), device=device).int()).sum().backward()
    optim.step()

    # Check that last_grad has been stored for at least one parameter
    any_last_grad_nonzero = False
    for p in model.parameters():
        if p in optim.state and "last_grad" in optim.state[p]:
            if optim.state[p]["last_grad"].any():
                any_last_grad_nonzero = True
    assert any_last_grad_nonzero, "last_grad should be nonzero after a step"


@pytest.mark.parametrize("device", DEVICES)
def test_checkpoint_roundtrip(device: torch.device, tmp_path):
    config = MarsConfig()
    model = MyModel().train().to(device)
    optim = config.build(model)

    # Take a step.
    optim.zero_grad(set_to_none=True)
    model(torch.randint(0, 1024, (2, 8), device=device).int()).sum().backward()
    optim.step()

    # Save and then restore a checkpoint, and make sure fixed fields reset.
    for group in optim.param_groups:
        group["initial_lr"] = 1e-8
    save_model_and_optim_state(tmp_path, model, optim)
    load_model_and_optim_state(tmp_path, model, optim)
    for group in optim.param_groups:
        assert group["initial_lr"] == config.lr
