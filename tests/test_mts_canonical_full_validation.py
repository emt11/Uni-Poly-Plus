import torch
from torch_geometric.data import Data

from src.dataset.mts_cache_integrity import _validate_trimer_record, _validate_topology_record


def _pair():
    topology = Data(
        mips_x=torch.zeros((2, 137)),
        canonical_atom_id=torch.arange(2),
        canonical_ru_atom_index=torch.arange(2),
        canonical_to_trimer_base_atom_id=torch.arange(2),
        lga_edge_index=torch.tensor([[0, 1], [0, 1]]),
        lga_spd=torch.tensor([0, 0]),
        lga_path_index=torch.tensor([[0], [1]]),
        lga_path_mask=torch.ones((2, 1), dtype=torch.bool),
        lga_path_shift=torch.zeros((2, 1), dtype=torch.long),
        graph_available=True,
        z=torch.tensor([6, 6]),
        mts_canonical_periodic=True,
        ru_edge_index=torch.tensor([[0, 1], [1, 0]]),
        ru_bond_type=torch.tensor([1, 1]),
        ru_left_boundary=0,
        ru_right_boundary=1,
        connection_bond_policy="matching_attachment_type",
        repeat_metadata={"attachment_bond_type": 1},
    )
    trimer = Data(
        trimer_pos=torch.zeros((6, 3)),
        trimer_atomic_number=torch.tensor([6, 6] * 3),
        trimer_ru_offset=torch.tensor([-1, -1, 0, 0, 1, 1]),
        trimer_base_ru_atom_id=torch.tensor([0, 1] * 3),
        trimer_central_ru_mask=torch.tensor([False, False, True, True, False, False]),
        trimer_edge_index=torch.tensor(
            [
                [0, 1, 2, 3, 4, 5, 1, 2, 3, 4],
                [1, 0, 3, 2, 5, 4, 2, 1, 4, 3],
            ]
        ),
        trimer_bond_type=torch.ones(10, dtype=torch.long),
        trimer_bond_aromatic=torch.zeros(10, dtype=torch.bool),
        mips_to_trimer_central_index=torch.tensor([2, 3]),
        trimer_geometry_valid=True,
        trimer_geometry_is_3d=True,
        trimer_2d_fallback=False,
        trimer_mcl_schema="mips-trimer-scage-trimer-v8",
        trimer_mcl_schema_version=8,
        trimer_formal_charge=torch.zeros(6, dtype=torch.long),
        trimer_is_aromatic=torch.zeros(6, dtype=torch.bool),
        trimer_chiral_tag=torch.zeros(6, dtype=torch.long),
        trimer_attachment_role=torch.zeros(6, dtype=torch.long),
        trimer_internal_degree=torch.ones(6, dtype=torch.long),
    )
    return topology, trimer


def test_edge_bond_length_mismatch_is_structural_failure():
    topology, trimer = _pair()
    trimer.trimer_bond_type = torch.ones(9, dtype=torch.long)
    failures = _validate_trimer_record(topology, trimer, "*CC*")
    assert "edge_count_bond_type_count" in failures


def test_invalid_mapping_and_nonfinite_valid_geometry_fail():
    topology, trimer = _pair()
    trimer.trimer_geometry_valid = True
    trimer.trimer_pos[2, 0] = float("nan")
    trimer.mips_to_trimer_central_index = torch.tensor([0, 99])
    failures = _validate_trimer_record(topology, trimer, "*CC*")
    assert "trimer_mapping" in failures
    assert "trimer_nonfinite_coordinates" in failures


def test_topology_relation_endpoint_is_hard_failure():
    topology, _ = _pair()
    topology.lga_edge_index[0, 0] = 9
    assert "topology_relation_endpoint" in _validate_topology_record(topology)


def test_graph_unavailable_placeholder_does_not_require_identity_graph():
    topology = Data(
        mips_x=torch.zeros((2, 137)),
        graph_available=False,
        mts_canonical_periodic=True,
    )
    trimer = Data(
        trimer_pos=torch.zeros((6, 3)),
        trimer_geometry_valid=False,
        trimer_geometry_is_3d=False,
        trimer_2d_fallback=False,
        trimer_mcl_schema="mips-trimer-scage-trimer-v8",
        trimer_mcl_schema_version=8,
    )
    assert _validate_trimer_record(topology, trimer, "must-not-be-parsed") == []


def test_invalid_geometry_checks_mapping_but_not_bond_graph():
    topology, trimer = _pair()
    trimer.trimer_geometry_valid = False
    trimer.trimer_edge_index = torch.empty((2, 0), dtype=torch.long)
    trimer.trimer_bond_type = torch.empty(0, dtype=torch.long)
    trimer.trimer_bond_aromatic = torch.empty(0, dtype=torch.bool)
    assert _validate_trimer_record(topology, trimer, "must-not-be-parsed") == []


def test_aromatic_kekule_form_is_accepted_without_smiles_parse():
    topology, trimer = _pair()
    topology.ru_bond_type = torch.tensor([4, 4])
    trimer.trimer_bond_type[:6] = torch.tensor([1, 1, 2, 2, 1, 1])
    assert "trimer_bond_graph" not in _validate_trimer_record(
        topology, trimer, "must-not-be-parsed"
    )


def test_legacy_all_double_aromatic_graph_is_normalized():
    topology, trimer = _pair()
    # Historical canonical Topology rows omitted aromatic flags and persisted
    # the internal RU as all-double; the Trimer contains the equivalent
    # single/double/aromatic Kekule representation.
    topology.ru_bond_type = torch.tensor([2, 2])
    trimer.trimer_bond_aromatic[2] = True
    assert "trimer_bond_graph" not in _validate_trimer_record(
        topology, trimer, "must-not-be-parsed"
    )


def test_aromatic_serialization_accepts_legacy_triple_code():
    topology, trimer = _pair()
    topology.ru_bond_type = torch.tensor([3, 3])
    trimer.trimer_bond_type[:2] = torch.tensor([4, 4])
    trimer.trimer_bond_aromatic[:2] = True
    assert "trimer_bond_graph" not in _validate_trimer_record(
        topology, trimer, "must-not-be-parsed"
    )
