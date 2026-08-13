"""CPU correctness and identity tests for the MTS T1 MSTA variant."""

import json
from pathlib import Path

import pytest
import torch

from src.dataset.canonical_periodic import build_canonical_periodic_topology
from src.dataset.dataloader import mips_trimer_collate
from src.modules.mips_local_graph import (
    MIPSLocalGraphEncoder,
    MSTAMIPSLocalAttention,
    MSTAMIPSLocalLayer,
    add_function_preserving_t1_parameters,
    topology_attention_identity,
)


def _batch(smiles="*CCO*"):
    return mips_trimer_collate([build_canonical_periodic_topology(smiles)])


def _models():
    torch.manual_seed(11)
    t0 = MIPSLocalGraphEncoder().eval()
    t1 = MIPSLocalGraphEncoder(topology_attention_variant="msta_last2").eval()
    t1.load_state_dict(add_function_preserving_t1_parameters(t0.state_dict()), strict=True)
    return t0, t1


def test_only_final_two_layers_are_msta():
    model = MIPSLocalGraphEncoder(topology_attention_variant="msta_last2")
    assert [isinstance(model.layers[i], MSTAMIPSLocalLayer) for i in range(6)] == [
        False, False, False, False, True, True
    ]
    assert model.model_identity == "T1"


def test_msta_masks_and_branch_softmax_keep_relation_rows():
    _, model = _models()
    batch = _batch()
    with torch.no_grad():
        model._forward_impl(batch, use_star=False, use_geometry=False, use_md=False)
    attention = model.layers[-1].attention.last_attention_branches
    assert attention is not None
    assert attention["local"].size(0) == batch.lga_edge_index.size(1)
    assert attention["context"].size(0) == batch.lga_edge_index.size(1)
    assert int(batch.lga_edge_index.size(1)) > int(
        torch.unique(batch.lga_edge_index, dim=1).size(1)
    )  # canonical relation multiplicity is retained
    assert set(batch.lga_spd[attention["local_mask"]].tolist()) <= {0, 1}
    assert set(batch.lga_spd[attention["context_mask"]].tolist()) <= {0, 1, 2}
    for name in ("local", "context"):
        weights = attention[name]
        mask = attention[name + "_mask"]
        sums = torch.zeros(batch.mips_x.size(0), model.num_heads)
        sums.scatter_add_(
            0,
            batch.lga_edge_index[1].unsqueeze(-1).expand_as(weights),
            weights * mask.unsqueeze(-1).to(weights.dtype),
        )
        assert torch.allclose(sums, torch.ones_like(sums), atol=2e-5, rtol=2e-5)
    diagnostic = model.layers[-1].attention
    diagnostic.diagnostic_capture = True
    with torch.no_grad():
        model._forward_impl(batch, use_star=False, use_geometry=False, use_md=False)
    row = diagnostic.last_diagnostic
    assert row["finite"] is True
    for name in ("local_attention_entropy", "context_attention_entropy"):
        summary = row[name]
        assert summary["count"] > 0
        assert summary["finite"] is True
        assert summary["range_valid"] is True
        assert 0.0 <= summary["mean"] <= 1.0


def test_msta_entropy_is_finite_for_masked_spd2_and_per_head():
    torch.manual_seed(17)
    attention = MSTAMIPSLocalAttention(dim=32, num_heads=4, dropout=0.0).eval()
    attention.diagnostic_capture = True
    # One target has self/one-hop/two-hop rows.  Local excludes SPD=2 while
    # context includes it; the entropy summary must still contain every head.
    x = torch.randn(1, 32)
    edge_index = torch.tensor([[0, 0, 0], [0, 0, 0]], dtype=torch.long)
    spd = torch.tensor([0, 1, 2], dtype=torch.long)
    with torch.no_grad():
        attention(x, edge_index, torch.zeros(3, 4), spd)
    branches = attention.last_attention_branches
    assert branches["local_mask"].tolist() == [True, True, False]
    assert branches["context_mask"].tolist() == [True, True, True]
    row = attention.last_diagnostic
    assert row["finite"] is True
    assert row["local_attention_entropy"]["count"] == 4
    assert row["context_attention_entropy"]["count"] == 4
    for name in ("local_attention_entropy", "context_attention_entropy"):
        summary = row[name]
        assert summary["range_valid"] is True
        assert 0.0 <= summary["mean"] <= 1.0


def test_msta_entropy_n_one_is_zero_and_complete():
    torch.manual_seed(19)
    attention = MSTAMIPSLocalAttention(dim=16, num_heads=2, dropout=0.0).eval()
    attention.diagnostic_capture = True
    x = torch.randn(1, 16)
    edge_index = torch.tensor([[0], [0]], dtype=torch.long)
    with torch.no_grad():
        attention(x, edge_index, torch.zeros(1, 2), torch.tensor([0]))
    row = attention.last_diagnostic
    assert row["finite"] is True
    for name in ("local_attention_entropy", "context_attention_entropy"):
        summary = row[name]
        assert summary["count"] == 2
        assert summary["mean"] == pytest.approx(0.0)
        assert summary["range_valid"] is True


def test_function_preserving_initialization_is_exact_at_eval():
    t0, t1 = _models()
    batch = _batch()
    with torch.no_grad():
        graph0, nodes0 = t0._forward_impl(
            batch, use_star=False, use_geometry=False, use_md=False
        )
        graph1, nodes1 = t1._forward_impl(
            batch, use_star=False, use_geometry=False, use_md=False
        )
    assert torch.allclose(nodes0, nodes1, atol=1e-6, rtol=1e-6)
    assert torch.allclose(graph0, graph1, atol=1e-6, rtol=1e-6)


def test_local_output_is_zero_bias_free_and_gets_finite_gradient():
    model = MIPSLocalGraphEncoder(topology_attention_variant="msta_last2")
    layer = model.layers[-1].attention
    assert layer.local_output.bias is None
    assert torch.count_nonzero(layer.local_output.weight) == 0
    batch = _batch()
    model.train()
    model.zero_grad(set_to_none=True)
    graph, _ = model._forward_impl(
        batch, use_star=False, use_geometry=False, use_md=False
    )
    graph.square().mean().backward()
    assert layer.local_output.weight.grad is not None
    assert torch.isfinite(layer.local_output.weight.grad).all()
    assert torch.count_nonzero(layer.local_output.weight.grad) > 0


def test_identity_contract_and_t0_legacy_default():
    assert topology_attention_identity("o8")["model_identity"] == "T0"
    identity = topology_attention_identity("msta_last2")
    assert identity["model_identity"] == "T1"
    assert identity["msta_layer_indices"] == [4, 5]
    with pytest.raises(ValueError):
        topology_attention_identity("unknown")


def test_resolved_t1_config_is_separate_from_t0():
    root = Path(__file__).resolve().parents[1]
    default = json.loads((root / "configs/mts/default.json").read_text())
    t0 = json.loads(
        (root / "configs/mts/experiments/T0_o8_pretrain20k_matched_v1.json").read_text()
    )
    t1 = json.loads(
        (root / "configs/mts/experiments/T1_msta_readiness.json").read_text()
    )
    assert default["topology_attention_variant"] == "msta_last2"
    assert t0["topology_attention_variant"] == "o8"
    assert t1["topology_attention_variant"] == "msta_last2"
    assert t1["msta_local_spd"] == [0, 1]
    assert t1["msta_context_spd"] == [0, 1, 2]
