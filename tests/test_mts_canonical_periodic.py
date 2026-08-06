"""Focused correctness checks for the canonical periodic MTS path."""

import pytest
import torch

from src.dataset.canonical_periodic import (
    build_canonical_periodic_topology,
    build_corrected_explicit_k_ru_reference,
    migrate_explicit_topology_to_canonical,
)
from src.dataset.dataloader import mips_trimer_collate
from src.dataset.dataset import _compute_ru_base_layer, _compute_topology_layer
from src.dataset.graph_data import (
    MIPSLocalConfig,
    attach_mips_local_lga,
    attach_polymerized_mips_atom_features,
    build_mips_data_object,
    build_mips_local_structure,
)
from src.dataset.trimer_mcl import attach_finite_trimer_mcl
from src.modules.mips_local_graph import MIPSLocalGraphEncoder


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


@pytest.mark.parametrize(
    ("smiles", "repeat_factor"),
    [
        ("*CC*", 1),
        ("*CCO*", 2),
        ("*CC(=O)OCC*", 3),
        ("*C(*)C(=O)OCC(C)(C)C", 4),
        ("*c1ccccc1*", 7),
    ],
)
def test_corrected_reference_mapping_for_repeat_factors(smiles, repeat_factor):
    topology = build_canonical_periodic_topology(smiles)
    reference = build_corrected_explicit_k_ru_reference(
        topology, repeat_factor
    )
    n = int(topology.mips_x.size(0))
    relation = topology.lga_edge_index.long()
    shifts = topology.lga_source_image_shift.long()
    lifted = reference.lga_edge_index.long()
    assert lifted.size(1) == int(repeat_factor) * relation.size(1)
    for copy in range(int(repeat_factor)):
        start = copy * relation.size(1)
        stop = start + relation.size(1)
        expected_source = relation[0] + (
            (copy + shifts) % int(repeat_factor)
        ) * n
        expected_target = relation[1] + copy * n
        assert torch.equal(lifted[0, start:stop], expected_source)
        assert torch.equal(lifted[1, start:stop], expected_target)
    assert torch.equal(
        reference.mips_x.view(int(repeat_factor), n, -1),
        topology.mips_x.unsqueeze(0).expand(int(repeat_factor), -1, -1),
    )


def test_corrected_explicit_translation_reference_and_model_equivalence():
    topology = build_canonical_periodic_topology("*CCO*")
    canonical = mips_trimer_collate([topology])
    explicit = build_corrected_explicit_k_ru_reference(topology, 7)
    model = MIPSLocalGraphEncoder().eval()
    with torch.no_grad():
        canonical_graph, canonical_nodes = model._forward_impl(
            canonical, use_star=False, use_geometry=False, use_md=False
        )
        explicit_graph, explicit_nodes = model._forward_impl(
            explicit, use_star=False, use_geometry=False, use_md=False
        )
    n = topology.mips_x.size(0)
    assert torch.allclose(
        explicit_nodes.reshape(7, n, -1),
        canonical_nodes.unsqueeze(0),
        atol=1e-5,
        rtol=1e-5,
    )
    assert torch.allclose(canonical_graph, explicit_graph, atol=1e-5, rtol=1e-5)


def test_translation_reference_includes_star_rbf_and_mcl():
    smiles = "*CCO*"
    topology = build_canonical_periodic_topology(smiles)
    attach_finite_trimer_mcl(topology, smiles)
    topology.mips_md = torch.zeros(200)
    topology.mips_md_valid = torch.tensor(False)
    canonical = mips_trimer_collate([topology])
    model = MIPSLocalGraphEncoder().eval()
    model.trimer_mcl.geometry_gate.data.fill_(0.2)
    model.star_distance_bias.projection.weight.data.fill_(0.1)
    explicit = build_corrected_explicit_k_ru_reference(topology, 4)
    with torch.no_grad():
        canonical_graph, canonical_nodes = model._forward_impl(canonical)
        explicit_graph, explicit_nodes = model._forward_impl(explicit)
    n = topology.mips_x.size(0)
    assert torch.allclose(
        explicit_nodes.reshape(4, n, -1),
        canonical_nodes.unsqueeze(0),
        atol=1e-5,
        rtol=1e-5,
    )
    assert torch.allclose(canonical_graph, explicit_graph, atol=1e-5, rtol=1e-5)


def test_native_topology_and_one_record_migration():
    smiles = "*CCO*"
    ru_base = _compute_ru_base_layer(smiles)
    native = _compute_topology_layer(smiles, ru_base, max_hops=2)
    old_structure = build_mips_local_structure(
        smiles, config=MIPSLocalConfig(max_hops=2)
    )
    old = build_mips_data_object(
        old_structure["structure_mol"], old_structure["backbone_info"]
    )
    attach_mips_local_lga(old, old_structure, config=MIPSLocalConfig(max_hops=2))
    attach_polymerized_mips_atom_features(old, smiles)
    migrated = migrate_explicit_topology_to_canonical(
        old, ru_base, verify_features=True
    )
    assert torch.equal(native.mips_x, migrated.mips_x)
    assert torch.equal(native.mips_backbone_mask, migrated.mips_backbone_mask)
    assert torch.equal(native.lga_edge_index, migrated.lga_edge_index)
    assert migrated.mips_local_lga_schema_version == 2
