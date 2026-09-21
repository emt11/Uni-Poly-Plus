"""Family-B model contract: routing, messages, arms, deployment and gradients.

Everything here runs on a tiny synthetic batch that follows the real packed
contract (all-atom spatial graph + heavy-atom GLT bond rows), so it needs no
frozen cache, no GPU and no cohort.
"""
import math

import numpy as np
import pytest
import torch

from src.dataset.glt_ph import radius_grid
from src.dataset.glt_ph_fusion_inputs import stat_profile
from src.modules.glt_galformer_ph import GLTGalPH
from src.modules.glt_ph_fusion_candidates import (GLTFusionB, R2DModel,
                                                  ConditionalProfileEncoder,
                                                  SpatialMessageBlock, FusionDownstream,
                                                  FusionPretrainer, fusion_deployment_package,
                                                  load_fusion_deployment, MESSAGE_INPUT_DIM,
                                                  SPATIAL_WRITE_SCALE)
from scripts.pretrain_glt_ph_fusion import decay_split

ATOMS = 4
BONDS = 3
GRAPH_COUNT = 2


class Batch:
    """Attribute bag mirroring the packed batch the runners produce."""


def _spatial(positions, radii=(2.5, 4.0, 6.0)):
    edge, scale, distance, bonded = [], [], [], []
    for index, radius in enumerate(radii):
        for left in range(positions.size(0)):
            for right in range(positions.size(0)):
                if left == right:
                    continue
                value = float((positions[left] - positions[right]).norm())
                if value <= radius:
                    edge.append((left, right))
                    scale.append(index)
                    distance.append(value)
                    bonded.append(abs(left - right) == 1)
    return (torch.tensor(edge, dtype=torch.long).t().contiguous(),
            torch.tensor(scale, dtype=torch.long),
            torch.tensor(distance, dtype=torch.float32),
            torch.tensor(bonded, dtype=torch.bool))


def build_batch(seed=0, graphs=GRAPH_COUNT, profile='STAT'):
    generator = torch.Generator().manual_seed(seed)
    atoms = graphs * ATOMS
    bonds = graphs * BONDS
    positions = torch.cat([torch.tensor([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0],
                                         [3.0, 0.0, 0.0], [4.5, 0.0, 0.0]])
                           for _ in range(graphs)])
    batch = Batch()
    batch.mips_x = torch.rand((atoms, 137), generator=generator)
    batch.mips_x[:, 0] = 1.0
    batch.mips_backbone_mask = torch.zeros(atoms, dtype=torch.bool)
    pairs = [(left, right) for left in range(atoms) for right in range(atoms)
             if left != right and left // ATOMS == right // ATOMS]
    batch.lga_edge_index = torch.tensor(pairs, dtype=torch.long).t().contiguous()
    hops = torch.tensor([abs(left - right) for left, right in pairs], dtype=torch.long)
    batch.lga_spd = torch.where(hops == 3, torch.zeros_like(hops), hops)
    batch.lga_path_index = torch.tensor([(left, right, -1) for left, right in pairs])
    batch.lga_path_mask = torch.tensor([[True, True, False]] * len(pairs))
    batch.lga_path_shift = torch.zeros((len(pairs), 3), dtype=torch.long)
    batch.lga_source_image_shift = torch.zeros((len(pairs), 3), dtype=torch.long)
    batch.lga_relation_mask = torch.zeros(len(pairs), dtype=torch.bool)
    batch.bond_path_features = torch.zeros((len(pairs), 2, 14))
    batch.bond_path_mask = torch.zeros((len(pairs), 2), dtype=torch.bool)
    batch.mask2d_rows = torch.zeros(0, dtype=torch.long)
    batch.mask2d_policy = torch.zeros(0, dtype=torch.long)
    batch.mask2d_donor_atoms = torch.zeros(0, dtype=torch.long)
    batch.mask3d_rows = torch.zeros(0, dtype=torch.long)
    batch.mask3d_policy = torch.zeros(0, dtype=torch.long)
    batch.mask3d_donor_atoms = torch.zeros(0, dtype=torch.long)
    batch.canonical_graph_index = torch.arange(atoms) // ATOMS
    batch.bond_batch = torch.arange(bonds) // BONDS
    batch.graph_available = torch.ones(graphs, dtype=torch.bool)
    batch.geometry_valid = torch.ones(graphs, dtype=torch.bool)
    batch.bond_z_a = torch.full((bonds,), 6, dtype=torch.long)
    batch.bond_z_b = torch.full((bonds,), 6, dtype=torch.long)
    batch.bond_type = torch.ones(bonds, dtype=torch.long)
    batch.bond_distance = torch.full((bonds,), 1.5)
    batch.bond_center = torch.ones(bonds, dtype=torch.bool)
    left = torch.tensor([graph * ATOMS + offset for graph in range(graphs)
                         for offset in range(BONDS)])
    batch.bond_atom_index = torch.stack([left, left + 1])
    paths, angles, masks, sources, targets, groups, flags = [], [], [], [], [], [], []
    for graph in range(graphs):
        base = graph * BONDS
        for index in range(BONDS):
            paths.append([base + index, base + index, -1])
            angles.append([0.0, 0.0])
            masks.append([True, False])
            sources.append(base + index)
            targets.append(base + index)
            groups.append(len(groups))
            flags.append(True)
        for left, right in ((0, 1), (1, 0), (1, 2), (2, 1)):
            paths.append([base + left, base + right, -1])
            angles.append([math.pi / 3.0, 0.0])
            masks.append([True, False])
            sources.append(base + left)
            targets.append(base + right)
            groups.append(len(groups))
            flags.append(False)
    batch.line_path = torch.tensor(paths, dtype=torch.long)
    batch.line_angle = torch.tensor(angles, dtype=torch.float32)
    batch.line_mask = torch.tensor(masks, dtype=torch.bool)
    batch.line_source = torch.tensor(sources, dtype=torch.long)
    batch.line_target = torch.tensor(targets, dtype=torch.long)
    batch.line_path_group = torch.tensor(groups, dtype=torch.long)
    batch.line_is_self = torch.tensor(flags, dtype=torch.bool)
    edge_index, scale, distance, bonded = _spatial(positions)
    batch.spatial_edge_index = edge_index
    batch.spatial_scale = scale
    batch.spatial_distance = distance
    batch.spatial_bonded = bonded
    batch.atom_batch = torch.arange(atoms) // ATOMS
    batch.profile_input = _profile(profile, positions, atoms, graphs)
    batch.profile_valid = torch.ones(graphs, dtype=torch.bool)
    batch.profile_source = ['test'] * graphs
    return batch


def _profile(kind, positions, atoms, graphs):
    if kind == 'CONST':
        value = np.full((3, 32), 0.25, dtype=np.float32)
        return torch.tensor(np.stack([value] * graphs))
    numbers = np.full(atoms, 6, dtype=np.int64)
    rows = [stat_profile(positions[graph * ATOMS:(graph + 1) * ATOMS], numbers[:ATOMS])
            for graph in range(graphs)]
    return torch.tensor(np.stack(rows))


def stats():
    return {'mean': np.full((3, 32), 0.25, dtype=np.float32),
            'std': np.full((3, 32), 0.5, dtype=np.float32)}


def test_conditional_encoder_shapes_and_zero_initialised_router():
    encoder = ConditionalProfileEncoder()
    profile = torch.zeros((2, 3, 32))
    profile[:, :, 5] = 1.0
    out = encoder(profile)
    assert tuple(out.shape) == (2, 128)
    block = SpatialMessageBlock()
    routing = torch.softmax(block.router(torch.zeros((3, 128))), dim=-1)
    assert torch.allclose(routing, torch.full_like(routing, 1 / 3), atol=1e-6)
    assert tuple(block.scale_embedding.shape) == (3, 128)
    assert MESSAGE_INPUT_DIM == 2 * 128 + 33 + 128


def test_conditional_encoder_patchify_follows_the_radius_order():
    encoder = ConditionalProfileEncoder()
    profile = torch.arange(3 * 32, dtype=torch.float32).reshape(1, 3, 32)
    patches = encoder.patchify(profile)
    assert tuple(patches.shape) == (1, 8, 12)
    # patch 0 holds radii 0-3 of every channel, channel-major
    assert patches[0, 0, :4].tolist() == [0, 1, 2, 3]
    assert patches[0, 0, 4:8].tolist() == [32, 33, 34, 35]
    assert patches[0, 1, :4].tolist() == [4, 5, 6, 7]


def test_empty_and_missing_geometry_are_zero_and_finite():
    block = SpatialMessageBlock().eval()
    states = torch.randn((BONDS, 512))
    data = Batch()
    data.atom_batch = torch.zeros(ATOMS, dtype=torch.long)
    data.bond_atom_index = torch.stack([torch.arange(BONDS), torch.arange(BONDS) + 1])
    data.spatial_edge_index = torch.zeros((2, 0), dtype=torch.long)
    data.spatial_scale = torch.zeros(0, dtype=torch.long)
    data.spatial_distance = torch.zeros(0)
    data.spatial_bonded = torch.zeros(0, dtype=torch.bool)
    with torch.no_grad():
        updated, metrics = block(states, BONDS, data, torch.zeros((1, 128)))
    assert torch.equal(updated, states)              # no edges: exactly no update
    assert metrics['spatial_edge_count'] == 0
    assert all(value == 0.0 for value in metrics['message_norms'])


def test_router_counterfactual_changes_routing_and_restores_bitwise():
    torch.manual_seed(3)
    model = GLTFusionB(arm='STAT', profile_stats=stats()).eval()
    batch = build_batch()
    with torch.no_grad():
        baseline = model(batch)['g3'].clone()
        block = model.spatial[0]
        original = block.router[-1].weight.detach().clone()
        # A small random counterfactual: adding a constant could not change the
        # softmax of an all-zero head, so the perturbation must be structured.
        noise = torch.randn(block.router[-1].weight.shape,
                            generator=torch.Generator().manual_seed(21))
        block.router[-1].weight.add_(noise * 0.05)
        changed = model(batch)['spatial_metrics'][0]['routing_mean']
        assert max(abs(value - 1 / 3) for value in changed) > 0
        block.router[-1].weight.copy_(original)
        restored = model(batch)['g3']
    assert torch.equal(restored, baseline)      # restored bit-exactly


def test_block_update_is_permutation_equivariant():
    torch.manual_seed(4)
    block = SpatialMessageBlock().eval()
    data = build_batch()
    states = torch.randn((data.bond_atom_index.size(1), 512))
    conditional = torch.randn((GRAPH_COUNT, 128))
    atoms = int(data.atom_batch.numel())
    with torch.no_grad():
        baseline, _ = block(states, states.size(0), data, conditional)
        permutation = torch.randperm(atoms, generator=torch.Generator().manual_seed(11))
        inverse = torch.empty_like(permutation)
        inverse[permutation] = torch.arange(atoms)
        moved = Batch()
        moved.atom_batch = data.atom_batch[permutation]
        moved.bond_atom_index = inverse[data.bond_atom_index]
        moved.spatial_edge_index = inverse[data.spatial_edge_index]
        moved.spatial_distance = data.spatial_distance.clone()
        moved.spatial_bonded = data.spatial_bonded.clone()
        moved.spatial_scale = data.spatial_scale.clone()
        permuted, _ = block(states, states.size(0), moved, conditional)
    # the same physical bond receives the same message under any atom labelling
    assert torch.allclose(baseline, permuted, atol=1e-6)


def test_three_arms_share_initial_tensors_and_differ_only_in_input():
    shared = stats()

    def build(arm):
        # The runners seed the global streams once and then apply the common
        # initialization artifact; here the same stream position stands in for
        # both, so any arm-specific tensor difference is a real difference.
        torch.manual_seed(20260924)
        return GLTFusionB(arm=arm, profile_stats=shared)

    arms = {arm: build(arm).eval() for arm in ('CONST', 'STAT', 'PH')}
    reference = arms['CONST'].state_dict()
    for name, value in reference.items():
        for arm in ('STAT', 'PH'):
            assert torch.equal(value, arms[arm].state_dict()[name]), name
    batch = build_batch(profile='STAT')
    other = build_batch(profile='CONST')
    assert not torch.allclose(batch.profile_input, other.profile_input)
    with torch.no_grad():
        same = [arms[arm](batch)['g3'] for arm in ('CONST', 'STAT', 'PH')]
        # Uniform initial routing is the declared contract: with a zero router
        # head the conditional input cannot move the trunk yet.
        assert torch.equal(same[0], same[1])
        assert torch.equal(same[1], arms['STAT'](other)['g3'])
        for arm in ('CONST', 'STAT', 'PH'):
            block = arms[arm].spatial[0]
            noise = torch.randn(block.router[-1].weight.shape,
                                generator=torch.Generator().manual_seed(22))
            block.router[-1].weight.add_(noise * 0.05)   # counterfactual: open the router
        opened = {arm: arms[arm](batch) for arm in ('CONST', 'STAT', 'PH')}
        opened_other = arms['STAT'](other)
    assert torch.equal(opened['CONST']['g3'], opened['STAT']['g3'])
    # now the declared conditional input is what separates the arms
    assert not np.allclose(opened['STAT']['spatial_metrics'][0]['routing_mean'],
                           opened_other['spatial_metrics'][0]['routing_mean'])
    assert not torch.allclose(opened['STAT']['g3'], opened_other['g3'])


def test_block_runs_under_bf16_autocast():
    """Autocast must not mix bf16 messages with an fp32 accumulator."""
    torch.manual_seed(31)
    model = GLTFusionB(arm='STAT', profile_stats=stats()).eval()
    batch = build_batch(profile='STAT')
    with torch.no_grad():
        reference = model(batch)['g3'].float()
        with torch.autocast('cpu', dtype=torch.bfloat16):
            mixed = model(batch)['g3'].float()
    assert torch.isfinite(reference).all() and torch.isfinite(mixed).all()


def test_forward_returns_the_declared_keys_and_keeps_cls_rows_untouched():
    torch.manual_seed(5)
    model = GLTFusionB(arm='PH', profile_stats=stats()).eval()
    batch = build_batch(profile='STAT')
    with torch.no_grad():
        out = model(batch)
    for key in ('g2', 'g3', 'cls2', 'cls3', 'atom_states', 'bond_states', 'line3d_valid'):
        assert key in out
    assert out['bond_states'].size(0) == int(batch.bond_distance.numel())
    assert out['cls3'].size(0) == GRAPH_COUNT
    metrics = out['spatial_metrics']
    assert [row['after_layer'] for row in metrics] == [3, 5]
    assert all(row['spatial_edge_count'] > 0 for row in metrics)
    # the write-back never subtracts or rescales the trunk states
    assert SPATIAL_WRITE_SCALE == 0.1
    for row in metrics:
        assert row['bond_update_relative'] >= 0.0
        assert len(row['routing_mean']) == 3
        assert sum(row['routing_mean']) == pytest.approx(1.0, abs=1e-5)


def test_optimizer_group_and_weight_decay_contract():
    model = GLTFusionB(arm='STAT', profile_stats=stats())
    decay, no_decay = decay_split(model)
    decay_names = {name for name, _ in decay}
    no_decay_names = {name for name, _ in no_decay}
    assert 'spatial.0.message.0.weight' in no_decay_names      # PH-side module: no decay
    assert 'conditional.patch.0.weight' in no_decay_names
    assert 'o8.layers.0.attention.qkv.weight' in decay_names
    assert 'o8.layers.0.norm1.weight' in no_decay_names        # normalisation
    assert 'glt.endpoint.bias' in no_decay_names               # bias
    assert not (decay_names & no_decay_names)
    assert not any(name.startswith('ph_') for name in list(decay_names) + list(no_decay_names))


def test_deployment_roundtrip_is_strict():
    torch.manual_seed(6)
    model = GLTFusionB(arm='PH', profile_stats=stats())
    trainer = FusionPretrainer(model, objective='R0')
    package = fusion_deployment_package(trainer, 4, arm='PH')
    assert not any(name.startswith(('head_2d.', 'head_3d.', 'cl_proj2.', 'cl_proj3.'))
                   for name in package['state_dict'])
    target = GLTFusionB(arm='PH', profile_stats=stats())
    load_fusion_deployment(target, package, expected_step=4)
    with pytest.raises(ValueError):
        load_fusion_deployment(target, package, expected_step=5)
    broken = dict(package, architecture='something-else')
    with pytest.raises(ValueError):
        load_fusion_deployment(target, broken, expected_step=4)
    missing = dict(package, state_dict={name: value for name, value in
                                        list(package['state_dict'].items())[1:]})
    with pytest.raises(ValueError):
        load_fusion_deployment(target, missing, expected_step=4)


def test_r2d_reference_has_no_geometry_branch_and_supports_2d_only():
    torch.manual_seed(7)
    model = R2DModel().eval()
    names = [name for name, _ in model.named_parameters()]
    assert not any(name.startswith('glt') for name in names)
    assert not any('ph_' in name for name in names)
    batch = build_batch()
    with torch.no_grad():
        out = model(batch)
    assert tuple(out['cls2'].shape) == (GRAPH_COUNT, 512)
    head = FusionDownstream(model, readout='2D_ONLY')
    with torch.no_grad():
        prediction, aux = head(batch)
    assert tuple(prediction.shape) == (GRAPH_COUNT, 1)
    assert aux['readout'] == '2D_ONLY'


def test_new_modules_receive_finite_nonzero_gradients():
    torch.manual_seed(8)
    model = GLTFusionB(arm='PH', profile_stats=stats())
    batch = build_batch(profile='STAT')
    out = model(batch)
    out['g3'].sum().backward()

    def norms(prefix):
        return [parameter.grad.norm().item() for name, parameter in model.named_parameters()
                if name.startswith(prefix) and parameter.grad is not None]

    # The spatial block is on the loss path from step 0.
    assert any(value > 0 for value in norms('spatial.'))
    assert all(np.isfinite(value) for value in norms('spatial.'))
    # The conditional encoder sits behind a zero-initialised router head, so its
    # first-step gradient may legitimately be zero; after one real update the
    # declared 'finite non-zero task gradient' must hold for it too.
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    model(batch)['g3'].sum().backward()
    assert any(value > 0 for value in norms('conditional.'))
    assert all(np.isfinite(value) for value in norms('conditional.'))


def test_r0_reference_trunk_is_the_original_class():
    torch.manual_seed(9)
    model = GLTGalPH('cls', None).eval()
    batch = build_batch()
    with torch.no_grad():
        out = model(batch)
    assert out['cls2'] is not None and out['cls3'] is not None
    assert tuple(out['g3'].shape) == (GRAPH_COUNT, 512)
    assert model.ph_mode is None
    assert not any('ph' in name for name, _ in model.named_parameters())
