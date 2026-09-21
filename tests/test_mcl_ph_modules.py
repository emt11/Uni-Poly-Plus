"""Architecture contract tests for the MCL-PH branch (sections 4, 5.2, 8, 9.3).

A small hand-built batch is used so that the declared structures -- dense versus
Top-2 routing, per-expert parameter independence, the fusion fallback, the
shared atom head and the strict deployment round trip -- are observable without
opening a frozen cache.  Every forward/backward here is a real model call and is
counted as such.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
import torch
from torch_geometric.data import Data

from src.dataset.mcl_ph_view import DESCRIPTOR_COLUMNS, ROUTER_RADII
from src.modules.mcl_ph import (FUSION_MODES, MCLPHEncoder, TopologyRouter, balance_term,
                                deployment_package, load_deployment)
from src.modules.mcl_ph_pretrain import MCLPHPretrainer

GRAPHS = 2
CANONICAL = (2, 1)
HEAVY = (3, 3)
BONDS = ((0, 1), (0, 2))


def build_batch(*, readout_valid=(True, True), geometry_valid=(True, True),
                heavy_edges=(3, 3)):
    """One tiny two-graph batch with the declared MCL-PH fields."""
    batch = Data()
    canonical_total = sum(CANONICAL)
    heavy_total = sum(HEAVY)
    canonical_graph = torch.cat([torch.full((count,), index, dtype=torch.long)
                                 for index, count in enumerate(CANONICAL)])
    heavy_graph = torch.cat([torch.full((count,), index, dtype=torch.long)
                             for index, count in enumerate(HEAVY)])
    batch.canonical_graph_index = canonical_graph
    batch.mips_x = torch.zeros((canonical_total, 137))
    batch.mips_x[torch.arange(canonical_total), torch.arange(canonical_total) % 6] = 1.0
    batch.mips_backbone_mask = torch.zeros(canonical_total, dtype=torch.bool)
    relations = 2 * canonical_total
    batch.lga_edge_index = torch.stack([
        torch.arange(relations) % canonical_total,
        (torch.arange(relations) + 1) % canonical_total])
    batch.lga_spd = torch.ones(relations, dtype=torch.long)
    batch.lga_relation_mask = torch.zeros(relations, dtype=torch.bool)
    batch.lga_path_index = torch.zeros((relations, 3), dtype=torch.long)
    batch.lga_path_mask = torch.zeros((relations, 3), dtype=torch.bool)
    batch.bond_path_features = torch.zeros((relations, 2, 14))
    batch.bond_path_mask = torch.zeros((relations, 2), dtype=torch.bool)
    batch.graph_available = torch.ones(GRAPHS, dtype=torch.bool)
    batch.mcl_z = torch.tensor([5, 6, 8, 6, 6, 7], dtype=torch.long)
    batch.mcl_charge = torch.tensor([3, 3, 3, 4, 3, 2], dtype=torch.long)
    batch.mcl_aromatic = torch.tensor([0, 1, 0, 0, 0, 0], dtype=torch.long)
    batch.mcl_atom_batch = heavy_graph
    batch.mcl_central_index = torch.tensor([0, 1, 3], dtype=torch.long)
    batch.mcl_readout_valid = torch.tensor(readout_valid, dtype=torch.bool)
    batch.mcl_geometry_valid = torch.tensor(geometry_valid, dtype=torch.bool)
    batch.mcl_bond_index = torch.tensor([[0, 1], [1, 2], [3, 4]], dtype=torch.long)
    ends, scales, types, distances = [], [], [], []
    offset = 0
    for index, count in enumerate(heavy_edges):
        local = torch.tensor([[0, 1], [1, 2], [0, 2]], dtype=torch.long)[:count]
        for slot, cutoff in enumerate((2.0, 3.0, 4.0)):
            for left, right in local.tolist():
                if slot > index or left == right:
                    continue
                ends.extend([(left + offset, right + offset), (right + offset, left + offset)])
                scales.extend([slot, slot])
                distances.extend([1.0 + 0.4 * slot, 1.0 + 0.4 * slot])
                types.extend([0, 0])
        offset += count
    batch.mcl_edge_index = (torch.tensor(ends, dtype=torch.long).t().contiguous()
                            if ends else torch.zeros((2, 0), dtype=torch.long))
    batch.mcl_edge_scale = torch.tensor(scales, dtype=torch.long)
    batch.mcl_edge_distance = torch.tensor(distances)
    batch.mcl_edge_type = torch.tensor(types, dtype=torch.long)
    trajectory = torch.zeros((GRAPHS, len(ROUTER_RADII), DESCRIPTOR_COLUMNS))
    trajectory[0, :, 0] = 4.0
    trajectory[1, :, 3] = 0.5
    batch.mcl_trajectory = trajectory
    # Heavy-atom view coordinates whose distances equal the declared raw targets,
    # so the diagnostic copy-error baseline of this fixture is exactly zero.
    batch.mcl_pos = torch.tensor([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [0.0, 1.2, 0.0],
                                  [0.0, 0.0, 5.0], [1.6, 0.0, 5.0], [3.4, 0.0, 5.0]])
    return batch


def build_labels(batch, *, geometric=True):
    """Supervision of a two-graph batch.

    ``geometric=False`` mirrors the data contract of a graph without a valid
    geometry: it keeps the geometric mask target but carries no length, angle or
    non-bond target at all.
    """
    canonical_total = int(batch.mips_x.size(0))
    empty_pairs = torch.zeros((0, 2), dtype=torch.long)
    empty_values = torch.zeros(0)
    labels = {
        'atom_mask': torch.tensor([True, False, True], dtype=torch.bool),
        'atom_label': torch.tensor([4, 5, 6], dtype=torch.long),
        'mcl_length_pair': torch.tensor([[0, 1], [3, 4]], dtype=torch.long),
        'mcl_length_target': torch.tensor([0.4, -0.2]),
        'mcl_length_raw': torch.tensor([1.5, 1.6]),
        'mcl_length_graph': torch.tensor([0, 1], dtype=torch.long),
        'mcl_angle_pair': torch.tensor([[0, 1]], dtype=torch.long),
        'mcl_angle_target': torch.tensor([-0.5]),
        'mcl_angle_graph': torch.tensor([0], dtype=torch.long),
        'mcl_nonbond_pair': torch.tensor([[0, 2], [3, 5]], dtype=torch.long),
        'mcl_nonbond_slot': torch.tensor([0, 2], dtype=torch.long),
        'mcl_nonbond_target': torch.tensor([0.1, -0.3]),
        'mcl_nonbond_raw': torch.tensor([1.2, 3.4]),
        'mcl_nonbond_graph': torch.tensor([0, 1], dtype=torch.long),
    }
    if not geometric:
        labels.update(
            mcl_length_pair=empty_pairs, mcl_length_target=empty_values,
            mcl_length_raw=empty_values, mcl_length_graph=torch.zeros(0, dtype=torch.long),
            mcl_angle_pair=empty_pairs, mcl_angle_target=empty_values,
            mcl_angle_graph=torch.zeros(0, dtype=torch.long),
            mcl_nonbond_pair=empty_pairs, mcl_nonbond_slot=torch.zeros(0, dtype=torch.long),
            mcl_nonbond_target=empty_values, mcl_nonbond_raw=empty_values,
            mcl_nonbond_graph=torch.zeros(0, dtype=torch.long))
    del canonical_total
    return labels


def make_pretrainer(mode='gate', **kwargs):
    torch.manual_seed(0)
    return MCLPHPretrainer(mode, **kwargs)


# ---------------------------------------------------------------------------
# Section 5.2 -- router
# ---------------------------------------------------------------------------

def test_dense_routing_is_a_full_simplex_distribution():
    router = TopologyRouter()
    router.configure(TopologyRouter.DENSE)
    weights = router(torch.zeros((5, len(ROUTER_RADII) * DESCRIPTOR_COLUMNS)))['weights']
    assert torch.allclose(weights.sum(-1), torch.ones(5))
    assert bool((weights > 0).all())


def test_top_two_routing_keeps_exactly_two_scales_and_normalises():
    router = TopologyRouter()
    router.configure(TopologyRouter.TOP2)
    trajectory = torch.zeros((4, len(ROUTER_RADII) * DESCRIPTOR_COLUMNS))
    trajectory[0, 0] = 3.0
    routed = router(trajectory)
    weights = routed['weights']
    selected = (weights > 0).sum(-1)
    assert selected.tolist() == [2, 2, 2, 2]
    assert torch.allclose(weights.sum(-1), torch.ones(4))
    assert routed['top_k'].shape == (4, 2)


def test_dense_to_top_two_switch_happens_after_the_declared_update():
    router = TopologyRouter(dense_updates=500)
    assert router.routing_mode_for_step(1) == TopologyRouter.DENSE
    assert router.routing_mode_for_step(500) == TopologyRouter.DENSE
    assert router.routing_mode_for_step(501) == TopologyRouter.TOP2


def test_tied_logits_select_scales_in_ascending_cutoff_order():
    router = TopologyRouter()
    router.configure(TopologyRouter.TOP2)
    with torch.no_grad():
        for parameter in router.parameters():
            parameter.zero_()
    routed = router(torch.zeros((1, len(ROUTER_RADII) * DESCRIPTOR_COLUMNS)))
    assert routed['top_k'].tolist() == [[0, 1]]
    assert torch.allclose(routed['weights'], torch.tensor([[0.5, 0.5, 0.0]]))


def test_each_expert_owns_independent_parameters():
    branch = make_pretrainer('gate').encoder.branch
    states = [expert.state_dict() for expert in branch.experts]
    for name in states[0]:
        identities = {id(states[index][name]) for index in range(3)}
        assert len(identities) == 3, f'{name} is shared between experts'
    weight = 'element.weight'
    for left, right in ((0, 1), (0, 2), (1, 2)):
        assert not torch.equal(states[left][weight], states[right][weight]), \
            'the experts must start from different tensors'
    assert len({float(expert.cutoff) for expert in branch.experts}) == 3


def test_balance_term_is_zero_for_uniform_and_two_for_degenerate_routing():
    uniform = torch.full((4, 3), 1.0 / 3.0)
    value, empty = balance_term(uniform, torch.ones(4, dtype=torch.bool))
    assert not empty and float(value) == pytest.approx(0.0, abs=1e-6)
    degenerate = torch.tensor([[1.0, 0.0, 0.0]] * 4)
    value, _ = balance_term(degenerate, torch.ones(4, dtype=torch.bool))
    assert float(value) == pytest.approx(2.0, abs=1e-6)


def test_balance_term_without_any_valid_graph_is_a_finite_differentiable_zero():
    probabilities = torch.full((3, 3), 1.0 / 3.0, requires_grad=True)
    value, empty = balance_term(probabilities, torch.zeros(3, dtype=torch.bool))
    assert empty
    assert torch.isfinite(value)
    value.backward()
    assert probabilities.grad is not None
    assert float(probabilities.grad.abs().sum()) == 0.0


def test_disabled_experts_still_receive_the_geometry_gradient():
    model = make_pretrainer('gate')
    batch, labels = build_batch(), build_labels(build_batch())
    report = model(batch, labels)
    model.objective(report).backward()
    # Only the geometry objective trains the experts, and it never passes
    # through alpha, so every expert must carry a gradient.
    for index, expert in enumerate(model.encoder.branch.experts):
        gradient = expert.element.weight.grad
        assert gradient is not None and float(gradient.abs().sum()) > 0, f'expert {index}'


def test_the_router_receives_a_task_gradient():
    model = make_pretrainer('gate')
    batch, labels = build_batch(), build_labels(build_batch())
    report = model(batch, labels)
    model.objective(report).backward()
    gradient = model.encoder.branch.router.net[0].weight.grad
    assert gradient is not None and float(gradient.abs().sum()) > 0


# ---------------------------------------------------------------------------
# Section 8 -- fusion
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('mode', FUSION_MODES)
def test_every_fusion_mode_runs_and_keeps_the_atom_width(mode):
    model = make_pretrainer(mode)
    batch, labels = build_batch(), build_labels(build_batch())
    report = model(batch, labels)
    assert report['fused'].shape == report['atom_states'].shape
    assert report['fused'].shape[-1] == 512
    assert torch.isfinite(model.objective(report))


@pytest.mark.parametrize('mode', ('cat', 'gate', 'xattn'))
def test_an_invalid_readout_returns_the_two_dimensional_state_unchanged(mode):
    model = make_pretrainer(mode)
    batch = build_batch(readout_valid=(False, False), geometry_valid=(False, False))
    labels = build_labels(batch)
    report = model(batch, labels)
    assert torch.equal(report['fused'], report['atom_states']), \
        'the fallback must be the identity, not a zero tensor pushed through a bias'


@pytest.mark.parametrize('mode', ('cat', 'gate', 'xattn'))
def test_a_partly_invalid_batch_fuses_only_the_valid_graph(mode):
    model = make_pretrainer(mode)
    batch = build_batch(readout_valid=(True, False), geometry_valid=(True, False))
    labels = build_labels(batch)
    report = model(batch, labels)
    second = batch.canonical_graph_index == 1
    assert torch.equal(report['fused'][second], report['atom_states'][second])
    assert not torch.equal(report['fused'][~second], report['atom_states'][~second])


def test_cross_attention_degenerates_to_a_single_key_projection():
    model = make_pretrainer('xattn')
    batch = build_batch(readout_valid=(True, True))
    batch.mcl_central_index = torch.tensor([0, 1, 3], dtype=torch.long)
    labels = build_labels(batch)
    report = model(batch, labels)
    assert torch.isfinite(report['fused']).all()
    assert not torch.equal(report['fused'], report['atom_states'])


def test_the_two_atom_objectives_share_one_classification_head():
    model = make_pretrainer('gate')
    heads = [name for name, module in model.named_modules() if isinstance(module, torch.nn.Linear)
             and module.out_features == 101]
    assert heads == ['atom_head'], 'exactly one 512 -> 101 head may exist'


# ---------------------------------------------------------------------------
# Section 9.3 -- deployment
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('mode', ('cat', 'gate', 'xattn'))
def test_deployment_round_trips_strictly(mode):
    source_model = make_pretrainer(mode)
    batch = build_batch()
    package = deployment_package(source_model.encoder, 2, source={'cohort_hash': 'x'})
    restored = MCLPHEncoder(mode)
    load_deployment(restored, package, 2, expected_fusion=mode)
    assert restored.branch.router.mode == TopologyRouter.TOP2
    # Both sides run the same routing mode: comparing a dense source with a
    # Top-2 deployment would not be a parity check.
    source_model.encoder.branch.router.configure(TopologyRouter.TOP2)
    source_model.eval()
    restored.eval()
    labels = build_labels(batch)
    with torch.no_grad():
        first = source_model.encoder.fuse(source_model.encoder.encode(batch))
        second = restored.fuse(restored.encode(batch))
    assert torch.allclose(first, second, atol=1e-6)


def test_deployment_rejects_a_legacy_bundle_and_a_wrong_fusion():
    source_model = make_pretrainer('gate')
    package = deployment_package(source_model.encoder, 2)
    legacy = dict(package, architecture='O8-BondPath-GalformerTrimer-Hop2')
    with pytest.raises(ValueError):
        load_deployment(MCLPHEncoder('gate'), legacy, 2)
    with pytest.raises(ValueError):
        load_deployment(MCLPHEncoder('cat'), package, 2)
    with pytest.raises(ValueError):
        load_deployment(MCLPHEncoder('gate'), package, 3)
    wrong_route = dict(package, training_route='dual_glt')
    with pytest.raises(ValueError):
        load_deployment(MCLPHEncoder('gate'), wrong_route, 2)
    dense_inference = dict(package, router={'dense_updates': 500, 'top_k': 2})
    with pytest.raises(ValueError):
        load_deployment(MCLPHEncoder('gate'), dense_inference, 2)
