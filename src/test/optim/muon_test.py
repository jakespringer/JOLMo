import pytest
import torch
import torch.nn as nn

from olmo_core.optim import MuonConfig, OptimGroupOverride
from olmo_core.optim.muon import Muon


# torch.optim.Muon requires bfloat16 (Newton-Schulz runs in bf16) and typically CUDA.
requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for Muon")
requires_torch_muon = pytest.mark.skipif(
    not hasattr(torch.optim, "Muon"), reason="torch.optim.Muon not available in this PyTorch version"
)


class MuonTestModel(nn.Module):
    """
    Small model with a mix of param types:
    - embedding (2D but should go to AdamW)
    - two linear layers (2D hidden matrices → Muon)
    - a layer norm (1D → AdamW)
    - an lm_head-style output projection (2D but should default to AdamW)
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


@requires_torch_muon
def test_muon_config_builds_correctly():
    """Test that MuonConfig.build() creates the right optimizer with correct groups."""
    config = MuonConfig(adamw_fused=False)
    model = MuonTestModel()
    optim = config.build(model)

    assert isinstance(optim, Muon)

    # Check that initial_lr is set on every group.
    for group in optim.param_groups:
        assert "initial_lr" in group

    # Identify muon vs adamw groups.
    muon_groups = [g for g in optim.param_groups if g.get("_muon")]
    adamw_groups = [g for g in optim.param_groups if not g.get("_muon")]

    assert len(muon_groups) >= 1
    assert len(adamw_groups) >= 1

    # All Muon params must be 2D.
    for g in muon_groups:
        for p in g["params"]:
            assert p.ndim == 2

    # Muon groups should have muon LR, adamw groups should have adamw LR.
    for g in muon_groups:
        assert g["initial_lr"] == config.lr
    for g in adamw_groups:
        assert g["initial_lr"] == config.adamw_lr


@requires_torch_muon
def test_muon_param_classification():
    """Test that params are classified into the correct optimizer groups."""
    config = MuonConfig(adamw_fused=False)
    model = MuonTestModel()

    # Collect what should be Muon vs AdamW based on the classification rules.
    expected_muon_params = set()
    expected_adamw_params = set()
    for name, p in model.named_parameters():
        if p.ndim != 2:
            expected_adamw_params.add(id(p))
        elif name.startswith("embeddings."):
            expected_adamw_params.add(id(p))
        elif name.startswith("lm_head."):
            expected_adamw_params.add(id(p))
        else:
            expected_muon_params.add(id(p))

    optim = config.build(model)

    actual_muon_params = set()
    actual_adamw_params = set()
    for g in optim.param_groups:
        for p in g["params"]:
            if g.get("_muon"):
                actual_muon_params.add(id(p))
            else:
                actual_adamw_params.add(id(p))

    assert actual_muon_params == expected_muon_params
    assert actual_adamw_params == expected_adamw_params

    # Specifically check: fc1.weight and fc2.weight should be Muon.
    fc1_id = id(model.fc1.weight)
    fc2_id = id(model.fc2.weight)
    assert fc1_id in actual_muon_params
    assert fc2_id in actual_muon_params

    # Specifically check: embeddings.weight and lm_head.weight should be AdamW.
    emb_id = id(model.embeddings.weight)
    lm_id = id(model.lm_head.weight)
    assert emb_id in actual_adamw_params
    assert lm_id in actual_adamw_params


@requires_torch_muon
def test_muon_on_lm_head():
    """Test that muon_on_lm_head=True sends the LM head to Muon."""
    config = MuonConfig(muon_on_lm_head=True, adamw_fused=False)
    model = MuonTestModel()
    optim = config.build(model)

    lm_head_id = id(model.lm_head.weight)
    for g in optim.param_groups:
        for p in g["params"]:
            if id(p) == lm_head_id:
                assert g.get("_muon") is True, "lm_head.weight should be in a Muon group"


@requires_torch_muon
def test_muon_with_group_overrides():
    """Test that group_overrides work correctly with the Muon/AdamW split."""
    config = MuonConfig(
        adamw_fused=False,
        group_overrides=[
            OptimGroupOverride(params=["embeddings.weight"], opts=dict(weight_decay=0.0))
        ],
    )
    model = MuonTestModel()
    optim = config.build(model)

    # Find the embedding group — it should be AdamW with weight_decay=0.0.
    emb_id = id(model.embeddings.weight)
    found = False
    for g in optim.param_groups:
        for p in g["params"]:
            if id(p) == emb_id:
                assert g.get("_muon") is False, "embeddings should be in AdamW group"
                assert g["weight_decay"] == 0.0, "Override should set weight_decay to 0.0"
                found = True
    assert found, "Embedding param not found in any group"


@requires_cuda
@requires_torch_muon
def test_muon_step():
    """Test that the optimizer actually updates parameters on CUDA."""
    device = torch.device("cuda")
    config = MuonConfig(adamw_fused=True)
    model = MuonTestModel().train().to(device)
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
@requires_torch_muon
def test_muon_state_dict_roundtrip():
    """Test that state_dict / load_state_dict preserves optimizer state."""
    device = torch.device("cuda")
    config = MuonConfig(adamw_fused=True)
    model = MuonTestModel().train().to(device)
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


@requires_cuda
@requires_torch_muon
def test_muon_lr_update_visible_to_sub_optimizers():
    """Test that LR changes on wrapper param_groups are visible to sub-optimizers."""
    device = torch.device("cuda")
    config = MuonConfig(adamw_fused=True)
    model = MuonTestModel().train().to(device)
    optim = config.build(model)

    # Simulate what the scheduler does: update lr on each group.
    new_lr = 0.0042
    for g in optim.param_groups:
        g["lr"] = new_lr

    # Verify sub-optimizers see the updated LR.
    if optim._muon is not None:
        for g in optim._muon.param_groups:
            assert g["lr"] == new_lr
    if optim._adamw is not None:
        for g in optim._adamw.param_groups:
            assert g["lr"] == new_lr


@requires_torch_muon
def test_muon_rejects_non_2d_in_muon_group():
    """Test that passing non-2D params with _muon=True raises."""
    bias_param = nn.Parameter(torch.randn(16))
    with pytest.raises(ValueError, match="ndim=1"):
        Muon(
            [{"params": [bias_param], "_muon": True}],
            lr=0.02,
            weight_decay=0.0,
            adamw_fused=False,
        )


class EmbeddingOnlyModel(nn.Module):
    """Model with only embeddings and norms — no 2D hidden matrices for Muon."""

    def __init__(self):
        super().__init__()
        self.embeddings = nn.Embedding(64, 16)
        self.norm = nn.LayerNorm(16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.embeddings(x))


@requires_torch_muon
def test_muon_config_no_muon_params():
    """Test that a model with no 2D hidden params still builds (AdamW only)."""
    model = EmbeddingOnlyModel()
    config = MuonConfig(adamw_fused=False)
    optim = config.build(model)

    assert isinstance(optim, Muon)
    assert optim._muon is None
    assert optim._adamw is not None

    muon_groups = [g for g in optim.param_groups if g.get("_muon")]
    assert len(muon_groups) == 0


@requires_cuda
@requires_torch_muon
def test_muon_multiple_steps():
    """Test that multiple consecutive steps work without error."""
    device = torch.device("cuda")
    config = MuonConfig(adamw_fused=True)
    model = MuonTestModel().train().to(device)
    optim = config.build(model)

    for _ in range(5):
        optim.zero_grad(set_to_none=True)
        inp = torch.randint(0, 128, (2, 8), device=device)
        model(inp).sum().backward()
        optim.step()
