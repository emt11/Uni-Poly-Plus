"""Family-B input contract: spatial graphs, STAT, keys, conditional inputs."""
import numpy as np
import pytest
import torch

from src.dataset.glt_ph import PH_BINS, PH_CHANNELS, radius_grid
from src.dataset.glt_ph_fusion_inputs import (ARMS, SPATIAL_RADII, conditional_input,
                                              edge_rbf, heavy_projection, physical_bond_table,
                                              rbf_centers, rbf_width, spatial_edges,
                                              spatial_edge_count, standardize, stat_profile,
                                              _pair_keys)


class Trimer:
    """Minimal frozen-Trimer stand-in with the cache's own field names."""

    def __init__(self, positions, numbers, edges, bond_types, heavy=None, base=None,
                 offsets=None):
        self.trimer_pos = np.asarray(positions, dtype=np.float32)
        self.trimer_atomic_number = np.asarray(numbers, dtype=np.int64)
        self.trimer_edge_index = np.asarray(edges, dtype=np.int64).reshape(2, -1)
        self.trimer_bond_type = np.asarray(bond_types, dtype=np.int64)
        self.trimer_base_ru_atom_id = np.asarray(
            base if base is not None else range(len(numbers)), dtype=np.int64)
        self.trimer_ru_offset = np.asarray(
            offsets if offsets is not None else [0] * len(numbers), dtype=np.int64)
        if heavy is not None:
            self.trimer_heavy_indices = np.asarray(heavy, dtype=np.int64)
        self.trimer_geometry_valid = True


class Topology:
    def __init__(self, canonical_to_base):
        self.canonical_to_trimer_base_atom_id = np.asarray(canonical_to_base, dtype=np.int64)


def test_rbf_grid_is_equally_spaced_over_the_declared_range():
    centers = rbf_centers()
    assert centers.shape == (32,)
    assert centers[0] == 0.0 and centers[-1] == 6.0
    assert np.allclose(np.diff(centers), rbf_width())
    features = edge_rbf(torch.tensor([0.0, 6.0]), torch.tensor([True, False]))
    assert tuple(features.shape) == (2, 33)
    assert features[0, 0] > features[0, 1]          # nearest centre wins at d=0
    assert features[:, -1].tolist() == [1.0, 0.0]


def test_spatial_graphs_are_nested_symmetric_and_self_free():
    positions = torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [5.0, 0.0, 0.0],
                              [9.0, 0.0, 0.0]])
    # pairs: (0,1)=2.0 (0,2)=5.0 (0,3)=9.0 (1,2)=3.0 (1,3)=7.0 (2,3)=4.0
    graphs = spatial_edges(positions)
    assert [int(graph.size(1)) for graph in graphs] == [2, 6, 8]
    sets = [{(int(a), int(b)) for a, b in zip(graph[0], graph[1])} for graph in graphs]
    assert sets[0] <= sets[1] <= sets[2]
    for edges in sets:
        assert all(a != b for a, b in edges)
        assert all((b, a) in edges for a, b in edges)
    # d <= r is inclusive: the 4 A pair is in the 4 A graph and not in the 2.5 A one.
    assert (1, 2) in sets[1] and (1, 2) not in sets[0]
    assert (0, 3) not in sets[2]
    assert spatial_edge_count(positions) == [int(graph.size(1)) for graph in graphs]


def test_critical_radius_belongs_to_the_inner_graph_only():
    positions = torch.tensor([[0.0, 0.0, 0.0], [float(SPATIAL_RADII[0]), 0.0, 0.0]])
    inner = spatial_edges(positions, radii=(SPATIAL_RADII[0],))[0]
    assert int(inner.size(1)) == 2                     # exactly at the radius: included
    outside = spatial_edges(positions, radii=(SPATIAL_RADII[0] - 1e-3,))[0]
    assert int(outside.size(1)) == 0


def test_stat_channels_match_a_brute_force_reference():
    positions = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    numbers = torch.tensor([6, 1, 8])
    profile = stat_profile(positions, numbers)
    grid = radius_grid()
    heavy_pairs = np.array([3.0])                       # heavy atoms 0 (C) and 2 (O)
    all_pairs = np.array([1.0, 3.0, 2.0])               # every atom pair, H included
    for channel, pairs in ((0, heavy_pairs), (1, all_pairs)):
        expected = np.searchsorted(np.sort(pairs), grid, side='right') / pairs.size
        assert np.allclose(profile[channel], expected, atol=1e-6)
    # independent per-atom degree variance of the heavy cutoff graph
    points = np.asarray(positions)[np.asarray(numbers) > 1]
    distance = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)
    degree = (distance[None, :, :] <= grid[:, None, None]).sum(-1) - 1
    variance = ((degree - degree.mean(axis=1, keepdims=True)) ** 2).mean(axis=1)
    assert np.allclose(profile[2], variance / (points.shape[0] - 1) ** 2, atol=1e-6)


def test_stat_handles_degenerate_point_sets():
    positions = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    numbers = torch.tensor([1, 1])                     # no heavy atoms at all
    profile = stat_profile(positions, numbers)
    assert tuple(profile.shape) == (PH_CHANNELS, PH_BINS)
    assert not profile[0].any() and not profile[2].any()
    assert profile[1].max() == 1.0                     # the single all-atom pair


def test_heavy_projection_keeps_the_all_atom_index_space():
    positions = np.zeros((4, 3), dtype=np.float32)
    trimer = Trimer(positions, [6, 1, 8, 1], [[0], [2]], [1], heavy=[0, 2])
    projection = heavy_projection(trimer)
    assert projection['heavy_indices'].tolist() == [0, 2]
    assert projection['heavy_of_all'].tolist() == [0, -1, 1, -1]
    assert projection['edge_heavy'].tolist() == [[0], [1]]
    assert projection['positions'].size(0) == 4        # all-atom coordinates kept


def test_physical_bond_table_follows_the_frozen_row_order():
    positions = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [3.0, 0.0, 0.0],
                          [4.5, 0.0, 0.0]], dtype=np.float32)
    # heavy atoms interleaved with hydrogens: local index != all-atom index
    trimer = Trimer(positions, [6, 1, 6, 6], [[0, 2], [2, 3]], [1, 1], heavy=[0, 2, 3],
                    base=[0, 1, 2, 3], offsets=[0, 1, -1, 0])
    topology = Topology([0, 2, 3])                     # canonical -> base of heavy atoms
    table = physical_bond_table(trimer, topology)
    assert table['index'].tolist() == [[0, 1], [1, 2]]
    # the same bonds expressed in the all-atom space the spatial graph uses
    assert table['atom_index'].tolist() == [[0, 2], [2, 3]]
    assert table['center'].tolist() == [False, False]   # offsets 1/2 are not central
    assert table['code'].tolist() == [1, 1]


def test_spatial_relations_are_invariant_to_rigid_motion():
    # no pair sits near a radius boundary, so a rigid motion cannot flip a
    # membership decision through floating-point rounding
    positions = torch.tensor([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [3.0, 0.0, 0.0],
                              [0.0, 2.2, 0.0], [0.0, 0.0, 1.7]])
    angle = 0.7
    rotation = torch.tensor([[np.cos(angle), -np.sin(angle), 0.0],
                             [np.sin(angle), np.cos(angle), 0.0],
                             [0.0, 0.0, 1.0]], dtype=torch.float32)
    mirror = torch.diag(torch.tensor([1.0, -1.0, -1.0], dtype=torch.float32))
    moved = positions @ (mirror @ rotation).t() + torch.tensor([3.0, -2.0, 5.0])
    base = spatial_edges(positions)
    other = spatial_edges(moved)
    for left, right in zip(base, other):
        assert left.shape == right.shape
        assert torch.equal(left.sort(1).values, right.sort(1).values)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
def test_edge_features_follow_the_input_device():
    """A GPU batch must not be combined with a CPU basis (r5's failure mode)."""
    distance = torch.tensor([1.0, 2.0], device='cuda')
    bonded = torch.tensor([True, False], device='cuda')
    features = edge_rbf(distance, bonded)
    assert features.device.type == 'cuda'
    # and the same values as the CPU path
    cpu = edge_rbf(distance.cpu(), bonded.cpu())
    assert torch.allclose(features.cpu(), cpu, atol=1e-6)


def test_pair_keys_survive_wide_atom_indices():
    pair = torch.tensor([[3, 0], [0, 3]])
    keys = _pair_keys(pair, 4)
    assert keys[0] == keys[1]


def test_conditional_inputs_are_declared_and_ph_falls_back_to_const():
    positions = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    numbers = torch.tensor([6, 6, 8])
    const = np.zeros((PH_CHANNELS, PH_BINS), dtype=np.float32)
    const[0] = 0.5
    profile = np.ones((PH_CHANNELS, PH_BINS), dtype=np.float32)
    value, valid, source = conditional_input(positions, numbers, profile, True, 'CONST', const)
    assert source == 'p_train_mean' and valid and torch.equal(value, torch.as_tensor(const))
    value, valid, source = conditional_input(positions, numbers, profile, True, 'STAT', const)
    assert source == 'sample_spatial_stat' and valid
    assert not torch.equal(value, torch.as_tensor(const))
    value, valid, source = conditional_input(positions, numbers, profile, True, 'PH', const)
    assert source == 'sample_own_frozen' and valid and torch.equal(value, torch.as_tensor(profile))
    value, valid, source = conditional_input(positions, numbers, profile, False, 'PH', const)
    assert source == 'p_train_mean_fallback' and not valid
    assert torch.equal(value, torch.as_tensor(const))
    with pytest.raises(ValueError):
        conditional_input(positions, numbers, profile, True, 'NOPE', const)
    assert set(ARMS) == {'CONST', 'STAT', 'PH'}


def test_standardisation_makes_the_constant_input_exactly_zero():
    mean = np.full((PH_CHANNELS, PH_BINS), 0.25, dtype=np.float32)
    std = np.full((PH_CHANNELS, PH_BINS), 0.5, dtype=np.float32)
    scaled = standardize(mean, mean, std)
    assert torch.equal(scaled, torch.zeros_like(scaled))
    with pytest.raises(ValueError):
        standardize(mean, mean, np.zeros_like(std))
