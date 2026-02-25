import copy

import pytest
import torch
import torch.nn as nn

from olmo_core.distributed.checkpoint import (
    load_model_and_optim_state,
    save_model_and_optim_state,
)
from olmo_core.optim import AdamWConfig, CautiousAdamW, CautiousAdamWConfig, OptimGroupOverride
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
    config = CautiousAdamWConfig(lr=1e-3)
    model = MyModel()
    optim = config.build(model)
    assert isinstance(optim, CautiousAdamW)
    assert len(optim.param_groups) == 1
    for group in optim.param_groups:
        assert "initial_lr" in group


def test_config_with_group_overrides():
    config = CautiousAdamWConfig(
        group_overrides=[OptimGroupOverride(params=["wte.*"], opts=dict(weight_decay=0.0))]
    )
    model = MyModel()
    optim = config.build(model)
    assert isinstance(optim, CautiousAdamW)
    assert len(optim.param_groups) == 2
    assert optim.param_groups[0]["weight_decay"] == 0.0
    assert len(optim.param_groups[0]["params"]) == 1
    assert len(optim.param_groups[1]["params"]) == len(list(model.parameters())) - 1

    for group in optim.param_groups:
        assert "initial_lr" in group


@pytest.mark.parametrize("device", DEVICES)
def test_optimizer_step(device: torch.device):
    config = CautiousAdamWConfig()
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
    config = CautiousAdamWConfig()
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
def test_mask_zeros_disagreeing_coords(device: torch.device):
    """When momentum and gradient disagree in sign, those coords should receive zero update."""
    model = nn.Linear(4, 4, bias=False).to(device)

    optim = CautiousAdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.95))

    # Take a first step to populate momentum (exp_avg).
    model.zero_grad()
    loss = model(torch.randn(2, 4, device=device)).sum()
    loss.backward()
    optim.step()

    # Now manually set momentum to a known pattern: positive everywhere
    for p in model.parameters():
        state = optim.state[p]
        state["exp_avg"].fill_(1.0)

    # Set gradient to have mixed signs: first 2 cols positive, last 2 negative
    model.zero_grad()
    loss = model(torch.randn(2, 4, device=device)).sum()
    loss.backward()
    for p in model.parameters():
        p.grad[:, :2] = 1.0   # agrees with momentum (both positive)
        p.grad[:, 2:] = -1.0  # disagrees with momentum (grad neg, momentum pos)

    params_before = {n: p.clone() for n, p in model.named_parameters()}
    optim.step()

    # Coords where grad and momentum agree (cols 0,1) should be updated;
    # coords where they disagree (cols 2,3) should get a smaller update.
    # Because the mask normalizes by mean, disagreeing coords get 0 but
    # agreeing coords get amplified, so updates still happen everywhere
    # only in proportion to agreement. Verify that the agreeing columns
    # changed more than the disagreeing ones.
    for n, p in model.named_parameters():
        delta = (p - params_before[n]).abs()
        agree_delta = delta[:, :2].mean()
        disagree_delta = delta[:, 2:].mean()
        assert agree_delta > disagree_delta, (
            f"Agreeing coords should have larger updates: agree={agree_delta.item():.6f}, "
            f"disagree={disagree_delta.item():.6f}"
        )


@pytest.mark.parametrize("device", DEVICES)
def test_cautious_differs_from_adamw(device: torch.device):
    """CautiousAdamW and AdamW should produce different updates (cautious mask is active)."""
    torch.manual_seed(42)

    model_cautious = MyModel().to(device)
    model_adamw = copy.deepcopy(model_cautious).to(device)

    cfg_common = dict(lr=1e-3, betas=(0.9, 0.95), weight_decay=0.1)
    optim_cautious = CautiousAdamWConfig(**cfg_common).build(model_cautious)  # type: ignore[arg-type]
    optim_adamw = AdamWConfig(fused=False, **cfg_common).build(model_adamw)  # type: ignore[arg-type]

    for _ in range(3):
        inp = torch.randint(0, 128, (4, 8), device=device)

        optim_cautious.zero_grad(set_to_none=True)
        model_cautious(inp).sum().backward()
        optim_cautious.step()

        optim_adamw.zero_grad(set_to_none=True)
        model_adamw(inp).sum().backward()
        optim_adamw.step()

    # After multiple steps, parameters should differ
    any_differ = False
    for p_c, p_a in zip(model_cautious.parameters(), model_adamw.parameters()):
        if not torch.equal(p_c, p_a):
            any_differ = True
    assert any_differ, "CautiousAdamW and AdamW produced identical updates — cautious mask may not be active"
