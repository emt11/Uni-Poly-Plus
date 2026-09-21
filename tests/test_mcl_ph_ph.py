"""Definition tests for the MCL-PH topology trajectory (section 5.1).

The acceptance reference is deliberately independent of the production code:
the Betti numbers come from explicit rank computations of the two boundary
matrices over F2, and the Wiener/efficiency columns come from ``networkx``
shortest paths.  No model is constructed and no frozen cache is opened.
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from scripts.audit_mcl_ph_p0 import betti_by_rank, ph_fixtures, wiener_by_networkx
from src.dataset import mcl_ph_view as view

FIXTURES = dict(ph_fixtures())


def test_grid_and_constants_are_the_declared_ones():
    assert view.ROUTER_RADII == tuple(1.5 + 0.1 * index for index in range(31))
    assert view.VR_MAX_EDGE == 4.5
    assert view.EXPERT_CUTOFFS == (2.0, 3.0, 4.0)
    assert view.DESCRIPTOR_COLUMNS == 5


@pytest.mark.parametrize('name', sorted(FIXTURES))
def test_betti_columns_match_an_independent_rank_reference(name):
    points = FIXTURES[name]
    trajectory = view.five_descriptors(points)
    pairs = view.persistence_pairs(points)
    normalizer = max(1, len(pairs[1]))
    for index, radius in enumerate(view.ROUTER_RADII):
        expected_zero, expected_one = betti_by_rank(points, radius)
        assert trajectory[index, 3] * float(points.shape[0]) == pytest.approx(expected_zero)
        assert trajectory[index, 4] * normalizer == pytest.approx(expected_one)


@pytest.mark.parametrize('name', sorted(FIXTURES))
def test_graph_columns_match_an_independent_shortest_path_reference(name):
    points = FIXTURES[name]
    trajectory = view.five_descriptors(points)
    for index, radius in enumerate(view.ROUTER_RADII):
        wiener, efficiency = wiener_by_networkx(points, radius)
        assert trajectory[index, 1] == pytest.approx(wiener, abs=1e-12)
        assert trajectory[index, 2] == pytest.approx(efficiency, abs=1e-12)


def test_isolated_point_has_one_component_and_no_edges():
    trajectory = view.five_descriptors(np.zeros((1, 3), dtype=np.float64))
    assert trajectory[:, 0].tolist() == [0.0] * 31
    assert trajectory[:, 1].tolist() == [0.0] * 31
    assert trajectory[:, 2].tolist() == [0.0] * 31
    assert trajectory[:, 3].tolist() == [1.0] * 31
    assert trajectory[:, 4].tolist() == [0.0] * 31


def test_isolated_points_are_never_connected_beyond_the_cutoff():
    trajectory = view.five_descriptors(FIXTURES['isolated_points'])
    assert trajectory[:, 0].tolist() == [0.0] * 31
    assert np.allclose(trajectory[:, 3], 1.0)


def test_tree_has_a_monotone_betti_zero_and_no_filling_triangle():
    points = FIXTURES['chain_tree']
    pairs = view.persistence_pairs(points)
    assert pairs[1] == []
    trajectory = view.five_descriptors(points)
    # A four-node path is the maximally spread tree, so the component-wise
    # Wiener column saturates at 1 while the radius graph is still a path and
    # falls back to 0 once every pair is adjacent.
    assert trajectory[0, 1] == pytest.approx(1.0)
    assert trajectory[-1, 1] == pytest.approx(0.0)
    assert trajectory[-1, 3] == pytest.approx(0.25)


def test_a_filling_triangle_is_a_zero_length_interval():
    points = FIXTURES['triangle_fill']
    pairs = view.persistence_pairs(points)
    # All three edges and the triangle share one filtration value, so the H1
    # class is born and dies at the same radius and must not be counted.
    assert pairs[1] == []
    assert view.five_descriptors(points)[:, 4].tolist() == [0.0] * 31


def test_square_ring_is_filled_by_its_diagonal():
    points = FIXTURES['square_with_diagonal']
    pairs = view.persistence_pairs(points)
    assert len(pairs[1]) == 1
    birth, death = pairs[1][0]
    assert birth == pytest.approx(1.0)
    assert death == pytest.approx(math.sqrt(2.0))


def test_disconnected_graph_keeps_zero_efficiency_between_components():
    points = FIXTURES['two_clusters']
    trajectory = view.five_descriptors(points)
    assert np.isfinite(trajectory).all()
    wiener, efficiency = wiener_by_networkx(points, view.ROUTER_RADII[-1])
    assert trajectory[-1, 1] == pytest.approx(wiener, abs=1e-12)
    assert trajectory[-1, 2] == pytest.approx(efficiency, abs=1e-12)


def test_right_censored_h1_is_kept_and_normalised_by_its_own_count():
    points = FIXTURES['square_censored']
    pairs = view.persistence_pairs(points)
    assert len(pairs[1]) == 1
    birth, death = pairs[1][0]
    assert birth == pytest.approx(3.5)
    assert not math.isfinite(death), 'the filling diagonal lies beyond the cutoff'
    trajectory = view.five_descriptors(points)
    # Exactly one positive-length H1 interval exists, so the column is the
    # active indicator itself and the last grid point must still be active.
    assert trajectory[-1, 4] == pytest.approx(1.0)
    assert trajectory[0, 4] == pytest.approx(0.0)


def test_an_empty_cloud_is_not_a_valid_topology_input():
    with pytest.raises(ValueError):
        view.five_descriptors(np.zeros((0, 3), dtype=np.float64))


def test_zero_length_intervals_are_dropped_not_counted():
    points = FIXTURES['coincident']
    pairs = view.persistence_pairs(points)
    assert all(birth < death for birth, death in pairs[0] + pairs[1])


def test_coordinates_and_distances_are_float64():
    points = np.asarray(FIXTURES['micro_ring'], dtype=np.float32)
    matrix = view.distance_matrix(points)
    assert matrix.dtype == np.float64
    assert view.five_descriptors(points).dtype == np.float64


def test_descriptors_are_translation_rotation_and_permutation_invariant():
    points = FIXTURES['micro_ring']
    baseline = view.five_descriptors(points)
    shifted = view.five_descriptors(points + np.asarray([3.1, -2.2, 0.7]))
    assert np.allclose(baseline, shifted, atol=1e-12)
    angle = 0.63
    rotation = np.asarray([[math.cos(angle), -math.sin(angle), 0.0],
                           [math.sin(angle), math.cos(angle), 0.0],
                           [0.0, 0.0, 1.0]])
    rotated = view.five_descriptors(points @ rotation.T)
    assert np.allclose(baseline, rotated, atol=1e-12)
    permuted = view.five_descriptors(points[[3, 0, 5, 1, 4, 2]])
    assert np.allclose(baseline, permuted, atol=1e-12)


def test_descriptors_are_reflection_invariant_at_the_declared_boundary():
    """Reflection is a distance-preserving map, so the trajectory must not move."""
    points = FIXTURES['micro_ring']
    mirrored = points.copy()
    mirrored[:, 2] *= -1.0
    assert np.allclose(view.five_descriptors(points), view.five_descriptors(mirrored),
                       atol=1e-12)


# ---------------------------------------------------------------------------
# Section 5.1 column 0: the declared 2/n Randic normalisation
# ---------------------------------------------------------------------------

def randic_reference(adjacency):
    """Hand formula ``2/n * sum_edges 1/sqrt(deg(u) deg(v))`` of section 5.1.

    Written from the contract text alone: an explicit edge enumeration over the
    degree list, with no production helper involved.
    """
    count = int(adjacency.shape[0])
    if count <= 0:
        return 0.0
    degree = [int(adjacency[row].sum()) for row in range(count)]
    total = 0.0
    for row in range(count):
        for column in range(row + 1, count):
            if adjacency[row, column] and degree[row] > 0 and degree[column] > 0:
                total += 1.0 / math.sqrt(degree[row] * degree[column])
    return 2.0 * total / float(count)


def _adjacency(pairs, count):
    graph = np.zeros((count, count), dtype=np.float64)
    for left, right in pairs:
        graph[left, right] = graph[right, left] = 1.0
    return graph


RANDIC_CASES = {
    # name: (edge list, node count, hand-computed value)
    'single_edge': ([(0, 1)], 2, 1.0),
    'complete_triangle': ([(0, 1), (0, 2), (1, 2)], 3, 1.0),
    'triangle_plus_isolated': ([(0, 1), (0, 2), (1, 2)], 4, 0.75),
    # degrees 1,2,2,1 -> 1/sqrt(1*2) + 1/sqrt(2*2) + 1/sqrt(2*1)
    'path_of_four': ([(0, 1), (1, 2), (2, 3)], 4,
                     2.0 / 4.0 * (1.0 / math.sqrt(2.0) + 0.5 + 1.0 / math.sqrt(2.0))),
    'two_disjoint_edges': ([(0, 1), (2, 3)], 4, 1.0),
    'star_of_four': ([(0, 1), (0, 2), (0, 3)], 4, 2.0 / 4.0 * 3.0 / math.sqrt(3.0)),
    'no_edge_at_all': ([], 5, 0.0),
    'isolated_pair_only': ([(0, 1)], 6, 2.0 / 6.0),
}


@pytest.mark.parametrize('name', sorted(RANDIC_CASES))
def test_randic_column_matches_hand_computed_values(name):
    pairs, count, expected = RANDIC_CASES[name]
    adjacency = _adjacency(pairs, count)
    assert view._randic(adjacency) == pytest.approx(expected, abs=1e-12)
    assert view._randic(adjacency) == pytest.approx(randic_reference(adjacency), abs=1e-12)


def test_randic_column_is_bounded_by_one_on_every_simple_graph():
    """``2/n * sum_edges 1/sqrt(d_u d_v) <= 1`` by AM-GM, for any simple graph."""
    generator = np.random.default_rng(20260921)
    for count in range(2, 12):
        for _ in range(20):
            raw = generator.random((count, count)) < 0.4
            adjacency = np.triu(raw, 1).astype(np.float64)
            adjacency = adjacency + adjacency.T
            value = view._randic(adjacency)
            assert 0.0 <= value <= 1.0 + 1e-12


@pytest.mark.parametrize('name', sorted(FIXTURES))
def test_every_descriptor_column_stays_in_its_declared_range(name):
    """All five columns are normalised into [0,1] on any real point cloud.

    The pre-r2 column 0 was the unnormalised degree sum, which violates this
    range; the invariant is what makes the router's input scale independent of
    the molecule size.
    """
    trajectory = view.five_descriptors(FIXTURES[name])
    assert trajectory.shape == (len(view.ROUTER_RADII), 5)
    assert np.isfinite(trajectory).all()
    assert (trajectory >= -1e-12).all()
    assert (trajectory <= 1.0 + 1e-12).all(), trajectory.max()


@pytest.mark.parametrize('name', sorted(FIXTURES))
def test_column_order_is_the_declared_one(name):
    """Each column is checked against the reference of its own definition."""
    points = FIXTURES[name]
    trajectory = view.five_descriptors(points)
    cloud = view._as_point_cloud(points)
    distance = view.distance_matrix(cloud)
    pairs = view.persistence_pairs(points)
    normalizer = max(1, len(pairs[1]))
    for index, radius in enumerate(view.ROUTER_RADII):
        adjacency = view._radius_graph(distance, radius)
        assert trajectory[index, 0] == pytest.approx(randic_reference(adjacency), abs=1e-12)
        wiener, efficiency = wiener_by_networkx(points, radius)
        assert trajectory[index, 1] == pytest.approx(wiener, abs=1e-12)
        assert trajectory[index, 2] == pytest.approx(efficiency, abs=1e-12)
        zero, one = betti_by_rank(points, radius)
        assert trajectory[index, 3] == pytest.approx(zero / float(points.shape[0]), abs=1e-12)
        assert trajectory[index, 4] == pytest.approx(one / normalizer, abs=1e-12)


def test_randic_is_not_the_unnormalised_degree_sum():
    """A size-dependent column would make the router input scale with the graph."""
    adjacency = _adjacency([(left, right) for left in range(6) for right in range(left + 1, 6)], 6)
    assert view._randic(adjacency) == pytest.approx(1.0, abs=1e-12)
    assert view._randic(adjacency) < 6.0  # K6 has 15 edges: raw sum would be 7.5
