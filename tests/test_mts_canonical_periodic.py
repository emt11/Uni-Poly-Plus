"""Focused correctness checks for the canonical periodic MTS path."""

import pytest
import torch

from src.dataset.canonical_periodic import (
    build_canonical_periodic_topology,
)


def test_lifted_rows_keep_shifts_and_symmetric_polymer_mask():
    topology = build_canonical_periodic_topology("*C(*)C(=O)OCC(C)(C)C")
    assert topology.mips_x.shape[1] == 137
    assert topology.mips_x.size(0) == topology.mips_backbone_mask.numel()
    assert not hasattr(topology, "ru_copy_index")
    relation = topology.lga_edge_index
    shifts = topology.lga_source_image_shift
    assert relation.size(1) == shifts.numel()
    # Shared-boundary *C has canonical self relations at non-zero shifts.
    self_shifts = shifts[(relation[0] == 0) & (relation[1] == 0)]
    assert {-1, 0, 1}.issubset(set(int(value) for value in self_shifts))
    polymer = topology.polymer_link_mask
    assert torch.equal(polymer, topology.lga_star_edge_mask)
    assert int(polymer.sum()) == 2
    assert torch.all(topology.lga_spd <= 2)


def test_degenerate_dummy_pair_reports_unavailable_graph_instead_of_keyerror():
    with pytest.raises(ValueError, match="attachment neighbor"):
        build_canonical_periodic_topology("*=*")
