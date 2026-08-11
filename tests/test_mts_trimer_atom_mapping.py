import torch

from src.dataset.canonical_periodic import find_base_atom_mapping, find_atom_graph_mapping


def test_base_atom_mapping_is_deterministic_and_excludes_dummy_sites():
    first = find_base_atom_mapping("*CC*", "*CC*")
    second = find_base_atom_mapping("*CC*", "*CC*")
    assert first.dtype == torch.long
    assert first.tolist() == [0, 1]
    assert torch.equal(first, second)


def test_graph_mapping_rejects_incompatible_atom_identity():
    try:
        find_atom_graph_mapping("*CC*", "*CO*")
    except ValueError as exc:
        assert "mapping failed" in str(exc)
    else:
        raise AssertionError("incompatible atom identity was accepted")
