import copy

import pytest
import torch
from rdkit import Chem
from rdkit.Chem import AllChem

from src.dataset.dataloader import custom_collate, mips_trimer_collate
from src.dataset.mts_star_rbf_v2 import build_star_rbf_v2_sample
from src.dataset import trimer_mcl as trimer_mcl_module
from src.dataset.dataset import (
    _attach_mips_descriptors,
    _compute_smiles_features_from_config,
    _mips_source_star_sub_molecule,
)
from src.modules.mips_local_graph import MIPSLocalGraphEncoder


def graph_data(smiles="*CCO*", *, trimer_max_heavy_atoms=384):
    data = _compute_smiles_features_from_config(
        smiles,
        "./pretrained_models/encoders/PubChem10M_SMILES_BPE_450k",
        32, "star_linking", "repeat_unit", "disabled",
        graph_encoder_type="scage",
        mips_core="paper_corrected",
        mips_max_hops=2,
        mips_use_descriptors=True,
        spatial_mode="trimer_scage",
        graph_geometry_mode="trimer_scage_mcl",
        trimer_num_candidates=4,
        trimer_max_heavy_atoms=trimer_max_heavy_atoms,
    )
    record = build_star_rbf_v2_sample(b"k" * 32, data, data)
    relations, pairs = record["relations"], record["pairs"]
    data.mts_star_v2_relation_row = torch.tensor(
        [item["row"] for item in relations], dtype=torch.long
    )
    data.mts_star_v2_relation_pair_index = torch.tensor(
        [item["pair_index"] for item in relations], dtype=torch.long
    )
    data.mts_star_v2_relation_spd = torch.tensor(
        [item["spd"] for item in relations], dtype=torch.long
    )
    data.mts_star_v2_pair_observation_distances = torch.tensor(
        [item["distances"] for item in pairs], dtype=torch.float
    )
    data.mts_star_v2_pair_observation_count = torch.tensor(
        [item["observation_count"] for item in pairs], dtype=torch.long
    )
    data.mts_star_v2_pair_valid = torch.tensor(
        [item["valid"] for item in pairs], dtype=torch.bool
    )
    data.mts_star_v2_pair_geometry_source = torch.tensor(
        [item["geometry_source"] for item in pairs], dtype=torch.long
    )
    data.mts_star_v2_sidecar_artifact = "a" * 64
    data.mts_star_v2_model_semantic_hash = "b" * 64
    data.mts_star_v2_rbf_upper = 6.0
    data.y = torch.zeros(1)
    return data


def test_fixed_architecture_and_incoming_attention():
    batch = custom_collate([graph_data()])
    encoder = MIPSLocalGraphEncoder().eval()
    graph, nodes = encoder(batch)
    assert encoder.architecture_name == "MIPS-Trimer-SCAGE"
    assert encoder.max_hops == 2
    assert len(encoder.layers) == 6
    assert len(encoder.trimer_mcl.layers) == 2
    assert not any("atom_pair" in key for key in encoder.state_dict())
    assert not hasattr(batch, "mips_atom_pair_3d")
    assert not hasattr(batch, "input_ids_smiles")
    assert not hasattr(batch, "fp")
    assert not hasattr(batch, "cell")
    assert not hasattr(batch, "pbc")
    assert graph.shape == (1, 512)
    assert nodes.shape == (batch.num_nodes, 512)
    # Both new residual gates are zero at initialization.
    assert torch.allclose(graph[0], nodes.mean(0), atol=1e-6)
    target = batch.lga_edge_index[1]
    attention = encoder.layers[-1].attention.last_attention
    for node in torch.unique(target):
        assert torch.allclose(
            attention[target == node].sum(0), torch.ones(8), atol=1e-6
        )


def test_production_mips_collate_excludes_legacy_modalities():
    batch = mips_trimer_collate([graph_data("*CCO*"), graph_data("*CCCCCCC*")])
    assert batch.mips_x.size(1) == 137
    assert batch.trimer_pos.ndim == 2 and batch.trimer_pos.size(1) == 3
    assert batch.mips_to_trimer_central_index.numel() == batch.x.size(0)
    assert batch.batch.numel() == batch.x.size(0)
    assert not hasattr(batch, "input_ids_smiles")
    assert not hasattr(batch, "fp")
    assert not hasattr(batch, "cell")
    assert not hasattr(batch, "pbc")
    # LMDB records are unbatched, so the collator must evaluate the shared
    # MCL predicate before it creates the explicit PyG graph-batch vectors.
    assert bool(batch.mcl_valid.any())


def test_short_ru_mapping_and_polymerized_features():
    data = graph_data("*CCO*")
    assert data.mips_boundary_distance > 5
    assert int(data.mips_repeat_factor) == 3
    mapped = data.mips_to_trimer_central_index
    assert mapped.unique().numel() == 3
    assert data.trimer_central_ru_mask[mapped].all()
    assert torch.equal(
        data.trimer_base_ru_atom_id[mapped],
        data.canonical_ru_atom_index,
    )
    assert data.mips_atom_feature_source == (
        "topology_only_trimer_central_ru"
    )
    for canonical_id in torch.unique(data.canonical_ru_atom_index):
        rows = data.mips_x[data.canonical_ru_atom_index == canonical_id]
        assert torch.equal(rows, rows[:1].expand_as(rows))


def test_star_relation_geometry_and_direct_edge_mask():
    data = graph_data("*CCO*")
    assert data.star_3d_valid
    metadata_edges = data.trimer_edge_index[:, ::2].T
    positions = data.trimer_pos
    # Exactly two directed LGA relations are direct virtual Star edges.
    assert int(data.lga_star_edge_mask.sum()) == 2
    assert bool((data.lga_spd[data.lga_star_edge_mask] == 1).all())
    assert float(data.star_3d_asymmetry) <= 0.15
    assert 0.0 < float(data.star_3d_distance) <= 3.0
    assert metadata_edges.numel() > 0 and torch.isfinite(positions).all()
    reversed_data = graph_data("*OCC*")
    assert torch.equal(
        data.star_3d_distance, reversed_data.star_3d_distance
    )

    batch = custom_collate([data])
    encoder = MIPSLocalGraphEncoder().eval()
    bias = encoder.star_distance_bias.forward_periodic_relation_v2(
        batch, torch.float32
    )
    assert torch.equal(bias, torch.zeros_like(bias))
    encoder.star_distance_bias.projection.weight.data.fill_(0.1)
    bias = encoder.star_distance_bias.forward_periodic_relation_v2(
        batch, torch.float32
    )
    assert bool((bias[batch.lga_star_edge_mask] != 0).all())
    relation_source = batch.mts_star_v2_pair_geometry_source[
        batch.mts_star_v2_relation_pair_index
    ]
    # v2 encodes every valid periodic relation (not only direct Star links),
    # while the audited true-self relation remains an exact zero bias.
    assert torch.equal(
        bias[relation_source == 1],
        torch.zeros_like(bias[relation_source == 1]),
    )


def test_md200_is_2d_only_and_invalid_is_exact_fallback():
    mol = Chem.MolFromSmiles("*[Si]CCO*")
    completed = _mips_source_star_sub_molecule(mol)
    assert completed.GetNumConformers() == 0
    assert all(atom.GetAtomicNum() != 0 for atom in completed.GetAtoms())
    data = graph_data()
    _attach_mips_descriptors(data, mol)
    assert data.mips_md.shape == (200,)
    assert data.mips_descriptor_schema_version == 5
    assert not hasattr(data, "mips_atom_pair_3d")

    batch = custom_collate([data])
    encoder = MIPSLocalGraphEncoder().eval()
    encoder.md_residual.gate.data.fill_(0.3)
    invalid = copy.deepcopy(batch)
    invalid.mips_md_valid.zero_()
    invalid.mips_md.fill_(123.0)
    with torch.no_grad():
        expected, _ = encoder._forward_impl(invalid, use_md=False)
        observed, _ = encoder(invalid)
    assert torch.equal(expected, observed)


def test_geometry_fallback_and_stage_boundaries():
    batch = custom_collate([graph_data()])
    encoder = MIPSLocalGraphEncoder().eval()
    encoder.trimer_mcl.geometry_gate.data.fill_(0.2)
    encoder.star_distance_bias.projection.weight.data.fill_(0.1)
    star_invalid = copy.deepcopy(batch)
    star_invalid.mts_star_v2_pair_valid.zero_()
    with torch.no_grad():
        expected, expected_nodes = encoder._forward_impl(
            star_invalid, use_star=False, use_geometry=True, use_md=True
        )
        observed, observed_nodes = encoder(star_invalid)
    assert torch.equal(expected, observed)
    assert torch.equal(expected_nodes, observed_nodes)

    geometry_invalid = copy.deepcopy(batch)
    geometry_invalid.trimer_geometry_valid.zero_()
    with torch.no_grad():
        expected, expected_nodes = encoder._forward_impl(
            geometry_invalid, use_star=True, use_geometry=False, use_md=True
        )
        observed, observed_nodes = encoder(geometry_invalid)
    assert torch.equal(expected, observed)
    assert torch.equal(expected_nodes, observed_nodes)

    masked = batch.x.clone()
    masked[0] = 0
    with torch.no_grad():
        topology, _ = encoder.forward_with_x(batch, masked)
        adapted, _ = encoder.forward_geometry_with_x(batch, masked)
    assert torch.isfinite(topology).all()
    assert torch.isfinite(adapted).all()


def test_2d_flags_never_enable_trimer_mcl():
    batch = custom_collate([graph_data()])
    encoder = MIPSLocalGraphEncoder().eval()
    encoder.trimer_mcl.geometry_gate.data.fill_(0.4)
    invalid_2d = copy.deepcopy(batch)
    invalid_2d.trimer_geometry_valid.fill_(True)
    invalid_2d.trimer_geometry_is_3d.fill_(False)
    invalid_2d.trimer_2d_fallback.fill_(True)
    with torch.no_grad():
        expected, expected_nodes = encoder._forward_impl(
            invalid_2d, use_geometry=False, use_md=True
        )
        observed, observed_nodes = encoder(invalid_2d)
    assert torch.equal(expected, observed)
    assert torch.equal(expected_nodes, observed_nodes)


def test_mcl_rigid_motion_invariance():
    batch = custom_collate([graph_data("*CCCCCCC*")])
    encoder = MIPSLocalGraphEncoder().eval()
    encoder.trimer_mcl.geometry_gate.data.fill_(0.2)
    with torch.no_grad():
        reference, _ = encoder(batch)
    moved = copy.deepcopy(batch)
    rotation = torch.tensor([
        [0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    moved.trimer_pos = moved.trimer_pos @ rotation.T + torch.tensor(
        [3.0, -2.0, 5.0]
    )
    with torch.no_grad():
        transformed, _ = encoder(moved)
    assert torch.allclose(reference, transformed, atol=1e-6, rtol=1e-6)


def test_targeted_etkdg_retry_only_runs_after_fast_failure(monkeypatch):
    calls = []

    def fake_attempt(
        molecule, *, num_candidates, seed, use_random_coords,
        max_iterations,
    ):
        calls.append((
            num_candidates, seed, use_random_coords, max_iterations
        ))
        return molecule, ([] if len(calls) == 1 else [7])

    monkeypatch.setattr(
        trimer_mcl_module, "_embed_attempt", fake_attempt
    )
    molecule = Chem.MolFromSmiles("CC")
    _, conformer_ids = (
        trimer_mcl_module._embed_with_targeted_retry(
            molecule, num_candidates=4, seed=123
        )
    )
    assert conformer_ids == [7]
    assert calls[0] == (4, 123, False, 42)
    assert calls[1][0] == 2
    assert calls[1][2:] == (True, 200)
    assert calls[1][1] != calls[0][1]

    success_calls = []

    def successful_fast_attempt(
        molecule, *, num_candidates, seed, use_random_coords,
        max_iterations,
    ):
        success_calls.append((
            num_candidates, seed, use_random_coords, max_iterations
        ))
        return molecule, [3]

    monkeypatch.setattr(
        trimer_mcl_module, "_embed_attempt", successful_fast_attempt
    )
    _, conformer_ids = (
        trimer_mcl_module._embed_with_targeted_retry(
            molecule, num_candidates=4, seed=123
        )
    )
    assert conformer_ids == [3]
    assert success_calls == [(4, 123, False, 42)]


def test_mmff_relax_selects_lowest_finite_energy(monkeypatch):
    """New protocol: selects lowest finite post-relax energy for normal mol."""
    from src.dataset.trimer_mcl import (
        _MMFFConformerSelection,
        _optimize_mmff_and_select_lowest_finite,
    )
    molecule = Chem.MolFromSmiles("CCO")
    molecule = Chem.AddHs(molecule)
    AllChem.EmbedMultipleConfs(molecule, numConfs=2, params=AllChem.ETKDGv3())
    molecule = Chem.RemoveHs(molecule)

    selection = _optimize_mmff_and_select_lowest_finite(
        molecule, [0, 1], max_iterations=200,
    )
    assert isinstance(selection, _MMFFConformerSelection)
    assert torch.isfinite(torch.tensor(selection.energy))
    assert selection.conf_id in (0, 1)

    # MMFF parameters missing triggers explicit error.
    monkeypatch.setattr(
        trimer_mcl_module.AllChem,
        "MMFFGetMoleculeProperties",
        lambda *args, **kwargs: None,
    )
    with pytest.raises(ValueError, match="mmff94_parameters_unavailable"):
        _optimize_mmff_and_select_lowest_finite(
            molecule, [0, 1], max_iterations=200,
        )


def test_large_trimer_keeps_2d_diagnostics_but_disables_mcl():
    data = graph_data("*CCO*", trimer_max_heavy_atoms=1)
    assert not bool(data.trimer_geometry_valid)
    assert not bool(data.trimer_geometry_is_3d)
    assert bool(data.trimer_2d_fallback)
    assert data.trimer_geometry_source == (
        "rdkit_2d_large_molecule_mcl_disabled"
    )
    assert data.trimer_failure_code == "large_trimer_2d_mcl_disabled"
    assert torch.equal(
        data.trimer_pos[:, 2], torch.zeros_like(data.trimer_pos[:, 2])
    )
    assert not bool(data.star_3d_valid)
    assert float(data.star_3d_distance) == 0.0

    batch = custom_collate([data])
    encoder = MIPSLocalGraphEncoder().eval()
    encoder.trimer_mcl.geometry_gate.data.fill_(0.2)
    encoder.star_distance_bias.projection.weight.data.fill_(0.1)
    with torch.no_grad():
        expected_graph, expected_nodes = encoder._forward_impl(
            batch, use_geometry=False
        )
        graph, nodes = encoder(batch)
    assert torch.equal(graph, expected_graph)
    assert torch.equal(nodes, expected_nodes)
