"""CPU correctness tests for the retained MSTA variant."""

import torch

from src.dataset.canonical_periodic import build_canonical_periodic_topology
from src.dataset.dataloader import mips_trimer_collate
from src.modules.mips_local_graph import (
    MIPSLocalGraphEncoder,
    MSTAMIPSLocalLayer,
)


def _batch(smiles="*CCO*"):
    return mips_trimer_collate([build_canonical_periodic_topology(smiles)])


def test_only_final_two_layers_are_msta():
    model = MIPSLocalGraphEncoder(topology_attention_variant="msta_last2")
    assert [isinstance(model.layers[i], MSTAMIPSLocalLayer) for i in range(6)] == [
        False, False, False, False, True, True
    ]


def test_msta_masks_and_branch_softmax_keep_relation_rows():
    model = MIPSLocalGraphEncoder(topology_attention_variant="msta_last2").eval()
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
