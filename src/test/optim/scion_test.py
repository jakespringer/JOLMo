import pytest
import torch
import torch.nn as nn

from olmo_core.optim import ScionConfig, OptimGroupOverride
from olmo_core.optim.scion import Scion, _newton_schulz5


# Newton-Schulz runs in bfloat16, typically requires CUDA.
requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for Scion")


class ScionTestModel(nn.Module):
    """
    Small model with a mix of param types:
    - embedding (2D but should go to sign norm)
    - two linear layers (2D hidden matrices -> spectral norm / Newton-Schulz)
    - a layer norm (1D -> bias_rms norm)
    - an lm_head-style output projection (2D but should go to sign norm)
    """

    def __init__(self):
        super().__init__()
        self.embeddings = nn.Embedding(128, 32)
        self.fc1 = nn.Linear(32, 64, bias=False)
        self.fc2 = nn.Linear(64, 32, bias=False)
        self.norm = nn.LayerNorm(32)
        self.lm_head = nn.Linear(32, 128, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embeddings(x)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = self.norm(x)
        return self.lm_head(x)


def test_scion_param_classification():
    """Test that params are classified into the correct norm groups."""
    config = ScionConfig()
    model = ScionTestModel()

    expected: dict = {}
    for name, p in model.named_parameters():
        expected[name] = config._classify_param(name, p)

    # Embeddings and lm_head should be sign.
    assert expected["embeddings.weight"] == "sign"
    assert expected["lm_head.weight"] == "sign"

    # Hidden linear layers should be spectral.
    assert expected["fc1.weight"] == "spectral"
    assert expected["fc2.weight"] == "spectral"

    # LayerNorm params should be bias_rms.
    assert expected["norm.weight"] == "bias_rms"
    assert expected["norm.bias"] == "bias_rms"


def test_scion_config_builds_correctly():
    """Test that ScionConfig.build() creates the right optimizer with correct groups."""
    config = ScionConfig()
    model = ScionTestModel()
    optim = config.build(model)

    assert isinstance(optim, Scion)

    # Check that initial_lr is set on every group.
    for group in optim.param_groups:
        assert "initial_lr" in group

    # Check we have all three norm types.
    norm_types_found = set()
    for group in optim.param_groups:
        norm_types_found.add(group.get("_norm"))

    assert "spectral" in norm_types_found
    assert "sign" in norm_types_found
    assert "bias_rms" in norm_types_found

    # Check scales are set correctly.
    for group in optim.param_groups:
        norm = group["_norm"]
        if norm == "spectral":
            assert group["scale"] == config.spectral_scale
        elif norm == "sign":
            assert group["scale"] == config.sign_scale
        elif norm == "bias_rms":
            assert group["scale"] == config.bias_rms_scale


def test_scion_with_group_overrides():
    """Test that group_overrides work correctly with norm-type splitting."""
    config = ScionConfig(
        group_overrides=[
            OptimGroupOverride(params=["embeddings.weight"], opts=dict(scale=999.0))
        ],
    )
    model = ScionTestModel()
    optim = config.build(model)

    # Find the embedding group — it should be sign norm with overridden scale.
    emb_id = id(model.embeddings.weight)
    found = False
    for g in optim.param_groups:
        for p in g["params"]:
            if id(p) == emb_id:
                assert g.get("_norm") == "sign", "embeddings should be in sign group"
                assert g["scale"] == 999.0, "Override should set scale to 999.0"
                found = True
    assert found, "Embedding param not found in any group"


@requires_cuda
def test_scion_step():
    """Test that the optimizer actually updates parameters on CUDA."""
    device = torch.device("cuda")
    config = ScionConfig()
    model = ScionTestModel().train().to(device)
    optim = config.build(model)

    # Record initial param values.
    initial_params = {n: p.clone() for n, p in model.named_parameters()}

    # Forward-backward-step.
    optim.zero_grad(set_to_none=True)
    inp = torch.randint(0, 128, (2, 8), device=device)
    model(inp).sum().backward()
    optim.step()

    # Verify at least some params changed.
    any_changed = False
    for n, p in model.named_parameters():
        if not torch.equal(p, initial_params[n]):
            any_changed = True
            break
    assert any_changed, "No parameters were updated after optim.step()"


@requires_cuda
def test_scion_frank_wolfe_shrinkage():
    """Test that the Frank-Wolfe (1 - lr) shrinkage is applied to weights."""
    device = torch.device("cuda")
    lr = 0.1
    config = ScionConfig(lr=lr)
    model = ScionTestModel().train().to(device)
    optim = config.build(model)

    # Record initial norms.
    initial_norms = {n: p.norm().item() for n, p in model.named_parameters()}

    # Take a step with zero gradients — only the (1-lr) shrinkage should apply.
    optim.zero_grad(set_to_none=True)
    # Manually set gradients to zero (not None).
    for p in model.parameters():
        p.grad = torch.zeros_like(p)
    optim.step()

    # All params should have shrunk by factor (1-lr).
    for n, p in model.named_parameters():
        expected_norm = initial_norms[n] * (1.0 - lr)
        actual_norm = p.norm().item()
        assert abs(actual_norm - expected_norm) < 1e-3 * max(expected_norm, 1e-6), (
            f"{n}: expected norm {expected_norm:.6f}, got {actual_norm:.6f}"
        )


@requires_cuda
def test_newton_schulz_orthogonality():
    """Test that Newton-Schulz iteration produces approximately orthogonal output."""
    device = torch.device("cuda")
    G = torch.randn(32, 64, device=device)
    X = _newton_schulz5(G, steps=5)

    # X should be approximately orthogonal: X @ X^T ≈ I (up to scaling).
    # The NS iteration targets the polar factor, so X X^T should be close to I
    # for the smaller dimension.
    if G.size(0) <= G.size(1):
        product = X @ X.T
    else:
        product = X.T @ X

    I = torch.eye(product.size(0), device=device, dtype=product.dtype)
    diff = (product - I).float().norm().item()
    assert diff < 0.5, f"Newton-Schulz output not approximately orthogonal: ||XX^T - I|| = {diff:.4f}"


@requires_cuda
def test_scion_multiple_steps():
    """Test that multiple consecutive steps work without error."""
    device = torch.device("cuda")
    config = ScionConfig()
    model = ScionTestModel().train().to(device)
    optim = config.build(model)

    for _ in range(5):
        optim.zero_grad(set_to_none=True)
        inp = torch.randint(0, 128, (2, 8), device=device)
        model(inp).sum().backward()
        optim.step()


@requires_cuda
def test_scion_state_dict_roundtrip():
    """Test that state_dict / load_state_dict preserves optimizer state."""
    device = torch.device("cuda")
    config = ScionConfig()
    model = ScionTestModel().train().to(device)
    optim = config.build(model)

    # Take a step to populate state.
    optim.zero_grad(set_to_none=True)
    inp = torch.randint(0, 128, (2, 8), device=device)
    model(inp).sum().backward()
    optim.step()

    # Verify state is populated.
    assert len(optim.state) > 0

    # Save and load state dict.
    sd = optim.state_dict()
    assert "state" in sd
    assert "param_groups" in sd

    # Mess up initial_lr, then load and verify it's restored.
    original_initial_lrs = [g["initial_lr"] for g in optim.param_groups]
    for g in optim.param_groups:
        g["initial_lr"] = 1e-8

    optim.load_state_dict(sd)

    # Fixed fields (initial_lr) should be restored to pre-load values by the hook.
    for g, expected_lr in zip(optim.param_groups, original_initial_lrs):
        assert g["initial_lr"] == expected_lr
