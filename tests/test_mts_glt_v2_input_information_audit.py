import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/analyze_mts_glt_v2_input_information_v1.py"
SPEC = importlib.util.spec_from_file_location("mts_glt_v2_input_audit", SCRIPT)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def test_relation_occurrence_uses_matched_physical_distance_slots():
    tokens = {
        "token_atom_a": np.asarray([0, 1]),
        "token_atom_b": np.asarray([1, 2]),
        "token_shift": np.asarray([1, 0]),
        "token_endpoint_z_a": np.asarray([6, 6]),
        "token_endpoint_z_b": np.asarray([6, 8]),
        "token_observation_distances": np.asarray([
            [1.4, 1.5, 0.0],
            [1.2, 1.3, 1.4],
        ], dtype=np.float32),
        "token_observation_count": np.asarray([2, 3]),
    }
    relations = {
        "relation_source": np.asarray([0]),
        "relation_target": np.asarray([1]),
        "relation_center_atom": np.asarray([1]),
        "relation_multiplicity": np.asarray([2]),
        "relation_observation_angles": np.asarray([[2.0, 2.1, 0.0]], dtype=np.float32),
        "relation_observation_count": np.asarray([2]),
        "relation_valid": np.asarray([True]),
        "relation_is_fallback": np.asarray([False]),
    }

    rows = audit.relation_occurrences(tokens, relations, 17)

    assert rows.shape == (2,)
    np.testing.assert_allclose(rows["d_source"], [1.4, 1.5])
    np.testing.assert_allclose(rows["d_target"], [1.3, 1.4])
    np.testing.assert_allclose(rows["angle"], [2.0, 2.1])
    assert rows["sample_id"].tolist() == [17, 17]


def test_o8_trace_distinguishes_topology_from_bond_chemistry():
    trace = audit._o8_trace()
    rows = {row["feature"]: row for row in trace["features"]}

    assert trace["o8_explicit_bond_chemistry"] == "NO"
    bond_type = rows["bond_type(single/double/triple/aromatic)"]
    assert bond_type["present_in_batch"]
    assert not bond_type["passed_to_o8"]
    assert not bond_type["used_in_forward"]
    assert rows["bond_existence/topological path"]["used_in_forward"]


def test_metadata_reports_exact_periodic_redundancies():
    tokens = {
        "token_shift": np.asarray([0, 0, 1]),
        "token_cross": np.asarray([False, False, True]),
        "token_observation_count": np.asarray([3, 3, 2]),
        "distance_variance_norm": np.asarray([0.1, 0.2, 0.3]),
    }
    relations = {
        "relation_cross": np.asarray([False, True]),
        "relation_observation_count": np.asarray([3, 2]),
        "relation_multiplicity": np.asarray([3, 2]),
        "angle_variance_norm": np.asarray([0.2, 0.4]),
    }

    report = audit._metadata_report(tokens, relations)

    assert report["exact_redundancies"]["distance_count_equals_3_minus_abs_shift"]
    assert report["exact_redundancies"]["angle_count_equals_stored_multiplicity"]
