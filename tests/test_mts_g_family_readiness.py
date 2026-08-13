"""Focused G-family causal and identity tests (no formal training)."""

import torch

from scripts.mts_g_family_readiness_smoke import _sample
from src.dataset.dataloader import mips_trimer_collate
from src.modules.mips_local_graph import MIPSLocalGraphEncoder


def _model(arm):
    torch.manual_seed(42)
    return MIPSLocalGraphEncoder(
        topology_attention_variant="msta_last2",
        graph_geometry_mode=arm,
        g_family_arm=arm,
        use_star_rbf=False,
        use_mcl=False,
    ).eval()


def test_g0_g1_g2_g3_share_step0_forward():
    values = []
    for arm in ("g0", "g1", "g2", "g3"):
        values.append(_model(arm)(mips_trimer_collate([_sample(1, arm), _sample(2, arm)]))[0])
    for value in values[1:]:
        assert torch.allclose(values[0], value, atol=1e-6, rtol=1e-6)


def test_g0_bypass_and_geometry_gradients():
    for arm in ("g0", "g1", "g2", "g3"):
        model = _model(arm).train()
        output, _ = model(mips_trimer_collate([_sample(1, arm), _sample(2, arm)]))
        output.square().mean().backward()
        geometry_grads = [p.grad for p in model.relation_geometry_bias.parameters() if p.grad is not None]
        if arm == "g0":
            assert not geometry_grads
        else:
            assert geometry_grads
            assert all(torch.isfinite(value).all() for value in geometry_grads)


def test_zero_init_projection_then_second_step_path_encoder_gradient():
    model = _model("g1").train()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
    batch = mips_trimer_collate([_sample(1, "g1"), _sample(2, "g1")])
    first, _ = model(batch)
    first.square().mean().backward()
    projection_grad = model.relation_geometry_bias.relation_projection.weight.grad
    assert projection_grad is not None
    assert torch.isfinite(projection_grad).all()
    assert torch.count_nonzero(projection_grad).item() > 0
    assert all(
        parameter.grad is None or torch.count_nonzero(parameter.grad).item() == 0
        for parameter in model.relation_geometry_bias.path_mlp.parameters()
    )
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    second, _ = model(batch)
    second.square().mean().backward()
    path_grads = [
        parameter.grad
        for parameter in model.relation_geometry_bias.path_mlp.parameters()
        if parameter.grad is not None
    ]
    assert path_grads
    assert all(torch.isfinite(value).all() for value in path_grads)
    assert any(torch.count_nonzero(value).item() > 0 for value in path_grads)


def test_collate_preserves_global_relation_rows_and_path_offsets():
    batch = mips_trimer_collate([_sample(1, "g2"), _sample(2, "g2")])
    assert batch.mts_relation_geometry_relation_row.tolist() == [3, 7]
    assert batch.mts_relation_geometry_path_offsets.tolist() == [0, 1, 2]
