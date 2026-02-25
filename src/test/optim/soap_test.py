import torch
import torch.nn as nn

from olmo_core.optim import SoapConfig, OptimGroupOverride
from olmo_core.optim.soap import Soap


class SoapTestModel(nn.Module):
    """Small model for testing SOAP optimizer."""

    def __init__(self):
        super().__init__()
        self.wte = nn.Embedding(128, 16)
        self.fc1 = nn.Linear(16, 32, bias=False)
        self.fc2 = nn.Linear(32, 16, bias=False)
        self.ln = nn.LayerNorm(16)
        self.head = nn.Linear(16, 128, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.wte(x)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = self.ln(x)
        return self.head(x)


def test_config_builds_correctly():
    """Test that SoapConfig.build() creates the right optimizer type."""
    config = SoapConfig(lr=1e-3)
    model = SoapTestModel()
    optim = config.build(model)

    assert isinstance(optim, Soap)
    assert len(optim.param_groups) == 1

    for group in optim.param_groups:
        assert "initial_lr" in group


def test_config_with_group_overrides():
    """Test parameter group overrides work."""
    config = SoapConfig(
        group_overrides=[OptimGroupOverride(params=["wte.*"], opts=dict(weight_decay=0.0))]
    )
    model = SoapTestModel()
    optim = config.build(model)

    assert len(optim.param_groups) == 2
    assert optim.param_groups[0]["weight_decay"] == 0.0


def test_optimizer_step():
    """Test that the optimizer updates parameters (note: first step is skipped)."""
    config = SoapConfig()
    model = SoapTestModel().train()
    optim = config.build(model)

    # Step 1: first step is skipped (only initializes Gram matrices and eigenbasis)
    optim.zero_grad(set_to_none=True)
    model(torch.randint(0, 128, (2, 8))).sum().backward()
    initial_params = {n: p.clone() for n, p in model.named_parameters()}
    optim.step()

    # After step 1, params should NOT have changed (first step is skipped).
    for n, p in model.named_parameters():
        assert torch.equal(p, initial_params[n]), f"{n} changed on first step (should be skipped)"

    # Step 2: now parameters should update.
    initial_params = {n: p.clone() for n, p in model.named_parameters()}
    optim.zero_grad(set_to_none=True)
    model(torch.randint(0, 128, (2, 8))).sum().backward()
    optim.step()

    any_changed = False
    for n, p in model.named_parameters():
        if not torch.equal(p, initial_params[n]):
            any_changed = True
            break
    assert any_changed, "No parameters were updated after second optim.step()"


def test_gram_matrix_update():
    """Test that Gram matrices are initialized and accumulating."""
    config = SoapConfig()
    model = SoapTestModel().train()
    optim = config.build(model)

    # Take one step to initialize state.
    optim.zero_grad(set_to_none=True)
    model(torch.randint(0, 128, (2, 8))).sum().backward()
    optim.step()

    # Verify that Gram matrices exist and are non-zero for 2D params.
    for p in model.parameters():
        if p.dim() >= 2 and p in optim.state:
            state = optim.state[p]
            assert "GG" in state, "Gram matrices not initialized"
            for gg in state["GG"]:
                if len(gg) > 0:
                    assert gg.norm().item() > 0, "Gram matrix is still all zeros after a step"


def test_eigenbasis_refresh():
    """Test that eigenbases are refreshed after precondition_frequency steps."""
    config = SoapConfig(precondition_frequency=3)
    model = SoapTestModel().train()
    optim = config.build(model)

    # Take enough steps to trigger at least one QR refresh (step 1 skipped + 3 more = step 3).
    for i in range(5):
        optim.zero_grad(set_to_none=True)
        model(torch.randint(0, 128, (2, 8))).sum().backward()
        optim.step()

    # Verify Q matrices exist and are non-identity for 2D params.
    for p in model.parameters():
        if p.dim() >= 2 and p in optim.state:
            state = optim.state[p]
            assert state["Q"] is not None, "Eigenbasis Q not set"
            for q in state["Q"]:
                if len(q) > 0:
                    # Q should not be identity after several steps with non-zero gradients.
                    I = torch.eye(q.shape[0], device=q.device, dtype=q.dtype)
                    assert not torch.allclose(q, I, atol=1e-2), "Q is still identity after refresh"


def test_multiple_steps():
    """Test that multiple consecutive steps work without error."""
    config = SoapConfig()
    model = SoapTestModel().train()
    optim = config.build(model)

    for _ in range(10):
        optim.zero_grad(set_to_none=True)
        model(torch.randint(0, 128, (2, 8))).sum().backward()
        optim.step()


def test_1d_params_skip_preconditioning():
    """Test that 1D params skip preconditioning when precondition_1d=False (default)."""
    config = SoapConfig(precondition_1d=False)
    model = SoapTestModel().train()
    optim = config.build(model)

    # Take steps to populate state.
    for _ in range(3):
        optim.zero_grad(set_to_none=True)
        model(torch.randint(0, 128, (2, 8))).sum().backward()
        optim.step()

    # Check that 1D params have empty Gram matrices (no preconditioning).
    for p in model.parameters():
        if p.dim() == 1 and p in optim.state:
            state = optim.state[p]
            if "GG" in state:
                for gg in state["GG"]:
                    assert len(gg) == 0, "1D param should have empty Gram matrix when precondition_1d=False"


def test_state_dict_roundtrip():
    """Test that state_dict / load_state_dict preserves optimizer state."""
    config = SoapConfig()
    model = SoapTestModel().train()
    optim = config.build(model)

    # Take steps to populate state.
    for _ in range(3):
        optim.zero_grad(set_to_none=True)
        model(torch.randint(0, 128, (2, 8))).sum().backward()
        optim.step()

    assert len(optim.state) > 0

    sd = optim.state_dict()
    assert "state" in sd
    assert "param_groups" in sd

    # Mess up initial_lr, then load and verify it's restored.
    original_initial_lrs = [g["initial_lr"] for g in optim.param_groups]
    for g in optim.param_groups:
        g["initial_lr"] = 1e-8

    optim.load_state_dict(sd)

    for g, expected_lr in zip(optim.param_groups, original_initial_lrs):
        assert g["initial_lr"] == expected_lr
