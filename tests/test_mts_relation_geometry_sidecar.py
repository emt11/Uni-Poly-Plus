from types import SimpleNamespace

import numpy as np
import torch

from src.dataset.mts_relation_geometry import (
    REASON_TO_CODE,
    build_sample_record,
    enumerate_two_edge_paths,
    path_geometry,
    prepare_trimer,
)


def _topology(*, graph_available=True):
    return SimpleNamespace(
        lga_edge_index=torch.tensor([[0], [2]], dtype=torch.long),
        lga_spd=torch.tensor([2], dtype=torch.long),
        lga_source_image_shift=torch.tensor([0], dtype=torch.long),
        atomic_numbers=torch.tensor([6, 8, 7], dtype=torch.long),
        ru_edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
        ru_left_boundary=0,
        ru_right_boundary=2,
        graph_available=graph_available,
        canonical_to_trimer_base_atom_id=torch.tensor([0, 1, 2], dtype=torch.long),
    )


def _trimer(*, valid=True, fallback=False, atomic=None):
    return SimpleNamespace(
        trimer_geometry_valid=valid,
        trimer_geometry_is_3d=torch.tensor(valid and not fallback),
        trimer_2d_fallback=torch.tensor(fallback),
        trimer_pos=torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 1.0, 0.0]]),
        trimer_atomic_number=torch.tensor(atomic or [6, 8, 7], dtype=torch.long),
        trimer_base_ru_atom_id=torch.tensor([0, 1, 2], dtype=torch.long),
        trimer_ru_offset=torch.tensor([0, 0, 0], dtype=torch.long),
        trimer_central_ru_mask=torch.tensor([True, True, True]),
        mips_to_trimer_central_index=torch.tensor([0, 1, 2], dtype=torch.long),
        trimer_edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
        trimer_bond_type=torch.tensor([1, 1], dtype=torch.long),
    )


def test_sidecar_keeps_all_paths_and_geometry_values():
    record = build_sample_record(b"k" * 32, _topology(), _trimer())
    arrays = record["arrays"]
    assert arrays["relation_row"] == [0]
    assert arrays["relation_num_shortest_paths"] == [1]
    assert arrays["relation_path_offsets"] == [0, 1]
    assert arrays["path_intermediate_canonical_id"] == [1]
    assert arrays["path_geometry_valid"] == [True]
    assert arrays["path_invalid_reason_code"] == [0]
    assert np.isfinite(arrays["path_cos_angle"]).all()
    assert arrays["relation_geometry_valid"] == [True]


def test_invalid_geometry_retains_topology_path_with_reason_code():
    record = build_sample_record(b"k" * 32, _topology(), _trimer(valid=False, fallback=True))
    arrays = record["arrays"]
    assert arrays["relation_num_shortest_paths"] == [1]
    assert arrays["path_geometry_valid"] == [False]
    assert arrays["path_invalid_reason_code"] == [REASON_TO_CODE["2d_fallback"]]
    assert arrays["relation_invalid_reason_code"] == [REASON_TO_CODE["2d_fallback"]]
    assert arrays["path_cos_angle"] == [0.0]


def test_identity_mismatch_is_not_silently_mapped():
    trimer = _trimer(atomic=[6, 6, 7])
    prepared, reason = prepare_trimer(trimer, _topology())
    assert prepared is None
    assert reason == "canonical_trimer_atomic_mismatch"


def test_signed_shift_and_real_bond_semantics():
    paths = enumerate_two_edge_paths((2, -1), (0, -1), [(0, 1), (1, 2)], 0, 2)
    assert paths == [((2, -1), (1, -1), (0, -1))]
    # A direct local path uses two real bonds and has no Star edge.
    trimer = _trimer()
    prepared, reason = prepare_trimer(trimer, _topology())
    assert reason is None
    geometry, reason = path_geometry(((0, 0), (1, 0), (2, 0)), prepared)
    assert reason is None
    assert geometry["bond_types"] == (1, 1)


def test_graph_unavailable_has_no_false_geometry():
    record = build_sample_record(b"k" * 32, _topology(graph_available=False), _trimer())
    arrays = record["arrays"]
    assert arrays["relation_row"] == [0]
    assert arrays["relation_num_shortest_paths"] == [0]
    assert arrays["relation_geometry_valid"] == [False]
    assert arrays["relation_invalid_reason_code"] == [REASON_TO_CODE["graph_unavailable"]]
