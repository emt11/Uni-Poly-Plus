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
