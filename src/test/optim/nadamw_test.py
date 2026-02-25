import copy

import pytest
import torch
import torch.nn as nn

from olmo_core.distributed.checkpoint import (
    load_model_and_optim_state,
    save_model_and_optim_state,
)
from olmo_core.optim import AdamWConfig, NAdamW, NAdamWConfig, OptimGroupOverride
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
    config = NAdamWConfig(lr=1e-3)
    model = MyModel()
    optim = config.build(model)
    assert isinstance(optim, NAdamW)
    assert len(optim.param_groups) == 1
    for group in optim.param_groups:
        assert "initial_lr" in group


def test_config_with_group_overrides():
    config = NAdamWConfig(
        group_overrides=[OptimGroupOverride(params=["wte.*"], opts=dict(weight_decay=0.0))]
    )
    model = MyModel()
    optim = config.build(model)
    assert isinstance(optim, NAdamW)
    assert len(optim.param_groups) == 2
    assert optim.param_groups[0]["weight_decay"] == 0.0
    assert len(optim.param_groups[0]["params"]) == 1
    assert len(optim.param_groups[1]["params"]) == len(list(model.parameters())) - 1

    for group in optim.param_groups:
        assert "initial_lr" in group


@pytest.mark.parametrize("device", DEVICES)
def test_optimizer_step(device: torch.device):
    config = NAdamWConfig()
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
def test_checkpoint_roundtrip(device: torch.device, tmp_path):
    config = NAdamWConfig()
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


@pytest.mark.parametrize("device", DEVICES)
def test_nadamw_differs_from_adamw(device: torch.device):
    """NAdamW and AdamW should produce different updates (Nesterov correction is active)."""
    torch.manual_seed(42)

    model_nadamw = MyModel().to(device)
    model_adamw = copy.deepcopy(model_nadamw).to(device)

    cfg_common = dict(lr=1e-3, betas=(0.9, 0.999), weight_decay=0.01)
    optim_nadamw = NAdamWConfig(**cfg_common).build(model_nadamw)  # type: ignore[arg-type]
    optim_adamw = AdamWConfig(fused=False, **cfg_common).build(model_adamw)  # type: ignore[arg-type]

    for _ in range(3):
        inp = torch.randint(0, 128, (4, 8), device=device)

        optim_nadamw.zero_grad(set_to_none=True)
        model_nadamw(inp).sum().backward()
        optim_nadamw.step()

        optim_adamw.zero_grad(set_to_none=True)
        model_adamw(inp).sum().backward()
        optim_adamw.step()

    # After multiple steps, parameters should differ
    any_differ = False
    for p_n, p_a in zip(model_nadamw.parameters(), model_adamw.parameters()):
        if not torch.equal(p_n, p_a):
            any_differ = True
    assert any_differ, "NAdamW and AdamW produced identical updates — Nesterov correction may not be active"
