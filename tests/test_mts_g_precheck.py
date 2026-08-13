from types import SimpleNamespace

import torch

from scripts.analyze_mts_g_precheck import (
    _prepare_trimer,
    _new_aggregate,
    _new_stratum,
    _merge_aggregate,
    _variance_rows,
    enumerate_two_edge_paths,
    path_geometry,
    stratum_key,
)


def test_enumerates_all_two_edge_paths_without_collapsing_multiplicity():
    paths = enumerate_two_edge_paths(
        (0, 0),
        (3, 0),
        [(0, 1), (1, 3), (0, 2), (2, 3)],
        -99,
        -98,
    )
    assert paths == [((0, 0), (1, 0), (3, 0)), ((0, 0), (2, 0), (3, 0))]


def test_signed_shift_is_part_of_stratum_key():
    left = stratum_key(6, 8, 6, (1, 2), -1)
    right = stratum_key(6, 8, 6, (1, 2), 1)
    assert left != right
    assert left[2] == (1, 2)


def test_path_geometry_uses_real_bonds_and_known_cosine():
    info = {
        "state_to_local": {(0, 0): 0, (1, 0): 1, (2, 0): 2},
        "positions": torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        "atomic": torch.tensor([6, 6, 8]),
        "bonds": {(0, 1): {1}, (1, 2): {2}},
    }
    geometry, reason = path_geometry(((0, 0), (1, 0), (2, 0)), info)
    assert reason is None
    assert abs(geometry["distance"] - 2**0.5) < 1e-6
    assert abs(geometry["cosine"]) < 1e-6
    assert geometry["bond_types"] == (1, 2)


def test_path_geometry_reports_offset_and_nonfinite_failures():
    info = {
        "state_to_local": {(0, 0): 0, (1, 0): 1},
        "positions": torch.zeros((2, 3)),
        "atomic": torch.tensor([6, 6]),
        "bonds": {(0, 1): {1}},
    }
    assert path_geometry(((0, 1), (1, 0), (0, 0)), info)[1] == "ru_offset_uncovered"
    info["positions"][0, 0] = float("nan")
    assert path_geometry(((0, 0), (1, 0), (0, 0)), info)[1] == "degenerate_geometry"


def test_canonical_trimer_mapping_and_atomic_identity_are_explicitly_checked():
    topology = SimpleNamespace(
        z=torch.tensor([6, 8]),
        canonical_to_trimer_base_atom_id=torch.tensor([0, 1]),
    )
    trimer = SimpleNamespace(
        trimer_geometry_valid=True,
        trimer_geometry_is_3d=True,
        trimer_2d_fallback=False,
        trimer_pos=torch.zeros((6, 3)),
        trimer_atomic_number=torch.tensor([6, 8, 6, 8, 6, 8]),
        trimer_base_ru_atom_id=torch.tensor([0, 1, 0, 1, 0, 1]),
        trimer_ru_offset=torch.tensor([-1, -1, 0, 0, 1, 1]),
        trimer_central_ru_mask=torch.tensor([False, False, True, True, False, False]),
        mips_to_trimer_central_index=torch.tensor([2, 3]),
        trimer_edge_index=torch.tensor([[0, 1], [1, 0]]),
        trimer_bond_type=torch.tensor([1, 1]),
    )
    info, reason = _prepare_trimer(trimer, topology)
    assert reason is None
    assert info["state_to_local"][(1, 0)] == 3
    trimer.mips_to_trimer_central_index = torch.tensor([3, 2])
    assert _prepare_trimer(trimer, topology)[1] == "canonical_trimer_mapping_mismatch"


def test_conditional_variance_matches_hand_calculation_and_support_gate():
    aggregate = _new_aggregate()
    aggregate["strata"] = {
        "a": {**_new_stratum(), "path_count": 2, "path_sum": 1.0, "path_sumsq": 0.5, "relation_count": 2, "relation_sum": 1.0, "relation_sumsq": 0.5},
        "b": {**_new_stratum(), "path_count": 1, "path_sum": -1.0, "path_sumsq": 1.0, "relation_count": 1, "relation_sum": -1.0, "relation_sumsq": 1.0},
    }
    rows = _variance_rows(aggregate)
    path_n2 = next(row for row in rows if row["unit"] == "path" and row["support_n"] == 2)
    assert path_n2["supported_count"] == 2
    assert path_n2["within_variance"] >= 0.0
    assert path_n2["between_variance"] >= 0.0
    assert path_n2["within_over_total"] <= 1.0


def test_aggregate_merge_is_worker_order_independent():
    left = _new_aggregate()
    right = _new_aggregate()
    left["sample_count"] = 2
    left["relation_count"] = 3
    right["sample_count"] = 1
    right["relation_count"] = 4
    left["strata"]["x"] = {**_new_stratum(), "path_count": 2, "path_sum": 0.5, "path_sumsq": 0.25, "relation_count": 2, "relation_sum": 0.5, "relation_sumsq": 0.25}
    right["strata"]["x"] = {**_new_stratum(), "path_count": 1, "path_sum": -0.5, "path_sumsq": 0.25, "relation_count": 1, "relation_sum": -0.5, "relation_sumsq": 0.25}
    forward = _new_aggregate()
    reverse = _new_aggregate()
    _merge_aggregate(forward, left)
    _merge_aggregate(forward, right)
    _merge_aggregate(reverse, right)
    _merge_aggregate(reverse, left)
    assert forward["sample_count"] == reverse["sample_count"] == 3
    assert forward["relation_count"] == reverse["relation_count"] == 7
    assert forward["strata"]["x"] == reverse["strata"]["x"]
