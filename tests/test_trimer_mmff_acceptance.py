"""Contract tests for the explicit-H single-conformer Trimer protocol."""

import pytest
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from torch_geometric.data import Data

from src.dataset.dataset import _compute_ru_base_layer, _compute_topology_layer
from src.dataset.graph_data import build_periodic_multimer_mol
from src.dataset.trimer_mcl import (
    TrimerContractError, _ExpectedGeometryFailure, _attach_placeholder,
    _round_seed, attach_finite_trimer_mcl,
    audit_double_bond_stereo_coordinates,
    audit_tetrahedral_stereo_coordinates,
)


def _topology(smiles="*CCO*"):
    ru = _compute_ru_base_layer(smiles)
    top = _compute_topology_layer(smiles, ru, max_hops=2)
    top.smiles = smiles
    return top


def _embedded_copy(molecule, count=2, seed=17):
    result = Chem.Mol(molecule)
    result.RemoveAllConformers()
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    params.useRandomCoords = True
    params.enforceChirality = True
    params.maxIterations = 200
    params.pruneRmsThresh = -1.0
    ids = list(AllChem.EmbedMultipleConfs(result, numConfs=count, params=params))
    assert len(ids) == count
    return result, [int(value) for value in ids]


def test_round_seed_is_schema_deterministic_and_round_specific():
    assert _round_seed("abc", 0) == _round_seed("abc", 0)
    assert _round_seed("abc", 0) != _round_seed("abc", 1)


def test_formal_embedding_uses_randomcoords_and_stops_after_success(monkeypatch):
    from src.dataset import trimer_mcl as module
    calls = []
    original = module._embed_attempt

    def wrapped(*args, **kwargs):
        calls.append(dict(kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "_embed_attempt", wrapped)
    result = attach_finite_trimer_mcl(
        _topology(), "*CCO*", num_candidates=2, max_rounds=2,
        timeout_seconds=60, sample_key="randomcoords-contract",
    )
    assert result.trimer_geometry_valid
    assert len(calls) == 1
    assert calls[0]["use_random_coords"] is True
    assert calls[0]["max_iterations"] == 200
    assert result.generation_diagnostics["num_rounds"] == 1
    candidate_rows = result.generation_diagnostics["rounds"][0]["candidates"]
    assert len(candidate_rows) == 2
    assert all(row["finite_pre_mmff"] for row in candidate_rows)
    assert all(row["double_bond_stereo_pre_pass"] is True for row in candidate_rows)
    assert all(row["tetra_stereo_pre_pass"] is True for row in candidate_rows)
    assert all(row["mmff_energy_finite"] for row in candidate_rows)
    assert all(row["finite_post_mmff"] for row in candidate_rows)
    assert all(row["double_bond_stereo_post_f64_pass"] is True for row in candidate_rows)
    assert all(row["tetra_stereo_post_f32_pass"] is True for row in candidate_rows)
    assert all(row["final_valid"] for row in candidate_rows)
    assert result.generation_diagnostics["stereo_time"] >= 0


def test_second_round_runs_only_when_first_has_no_final_valid_candidate(monkeypatch):
    from src.dataset import trimer_mcl as module
    calls = []
    original = module._embed_attempt

    def first_empty(molecule, **kwargs):
        calls.append(dict(kwargs))
        if len(calls) == 1:
            empty = Chem.Mol(molecule)
            empty.RemoveAllConformers()
            return empty, []
        return original(molecule, **kwargs)

    monkeypatch.setattr(module, "_embed_attempt", first_empty)
    result = attach_finite_trimer_mcl(
        _topology(), "*CCO*", num_candidates=1, max_rounds=2,
        timeout_seconds=60, sample_key="two-round-contract",
    )
    assert result.trimer_geometry_valid
    assert len(calls) == 2
    assert calls[0]["seed"] != calls[1]["seed"]
    assert all(call["use_random_coords"] is True for call in calls)


def test_successful_round_processes_all_candidates_and_prefers_converged(monkeypatch):
    from src.dataset import trimer_mcl as module
    original_embed = module._embed_attempt
    optimize_calls = []

    def embed_two(molecule, **kwargs):
        return original_embed(
            molecule, num_candidates=2, seed=kwargs["seed"],
            use_random_coords=True, max_iterations=200,
        )

    def optimize(_mol, *, confId, **_kwargs):
        optimize_calls.append(int(confId))
        return 1 if int(confId) == 0 else 0

    monkeypatch.setattr(module, "_embed_attempt", embed_two)
    monkeypatch.setattr(AllChem, "MMFFOptimizeMolecule", optimize)
    monkeypatch.setattr(
        module, "_calculate_mmff_energy",
        lambda _mol, _props, conf_id: 1.0 if int(conf_id) == 0 else 10.0,
    )
    result = attach_finite_trimer_mcl(
        _topology(), "*CCO*", num_candidates=2, max_rounds=2,
        sample_key="convergence-priority",
    )
    assert optimize_calls == [0, 1]
    assert result.selected_converged is True
    assert result.trimer_conformer_candidate_id == 1
    assert result.trimer_conformer_energy == 10.0


def test_finite_nonconverged_candidate_is_explicit_fallback(monkeypatch):
    monkeypatch.setattr(AllChem, "MMFFOptimizeMolecule", lambda *args, **kwargs: 1)
    result = attach_finite_trimer_mcl(
        _topology(), "*CCO*", num_candidates=1, max_rounds=1,
        sample_key="nonconverged-fallback",
    )
    assert result.trimer_geometry_valid
    assert result.selected_converged is False
    assert result.generation_diagnostics["rounds"][0]["candidates"][0]["mmff_status"] == 1


def test_mmff_unsupported_stops_before_embedding(monkeypatch):
    from src.dataset import trimer_mcl as module
    monkeypatch.setattr(AllChem, "MMFFGetMoleculeProperties", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        module, "_embed_attempt",
        lambda *args, **kwargs: pytest.fail("embedding must not run"),
    )
    result = attach_finite_trimer_mcl(_topology(), "*CCO*")
    assert not result.trimer_geometry_valid
    assert result.search_stop_reason == "MMFF_UNSUPPORTED"
    assert result.generation_diagnostics["num_rounds"] == 0
    assert result.trimer_pos.shape == (0, 3)
    assert result.trimer_atomic_numbers.numel() == 0
    assert result.trimer_edge_index.shape == (2, 0)


def test_all_atom_payload_keeps_hydrogens_and_physical_bonds():
    topology = _topology()
    result = attach_finite_trimer_mcl(
        topology, "*CCO*", num_candidates=1, max_rounds=1,
        sample_key="all-atom-contract",
    )
    assert result.trimer_geometry_valid
    count = result.trimer_pos.size(0)
    assert result.trimer_pos.shape == (count, 3)
    assert result.trimer_pos.dtype == torch.float32
    assert result.trimer_atomic_numbers.numel() == count
    assert torch.equal(result.trimer_atomic_numbers, result.trimer_atomic_number)
    assert bool((result.trimer_atomic_numbers == 1).any())
    assert result.trimer_atom_id.tolist() == list(range(count))
    assert torch.equal(
        result.trimer_heavy_indices,
        torch.nonzero(result.trimer_heavy_mask, as_tuple=False).flatten(),
    )
    hydrogens = torch.nonzero(~result.trimer_heavy_mask, as_tuple=False).flatten()
    parents = result.h_parent_heavy_index[hydrogens]
    assert bool(result.trimer_heavy_mask[parents].all())
    assert torch.equal(result.trimer_ru_offset[hydrogens], result.trimer_ru_offset[parents])
    assert result.trimer_bonds.size(1) * 2 == result.trimer_edge_index.size(1)
    assert result.trimer_bond_types.numel() == result.trimer_bonds.size(1)
    assert bool(result.trimer_heavy_mask[result.o8_to_trimer_atom].all())
    assert torch.equal(
        result.o8_heavy_mask, result.trimer_heavy_mask[result.o8_to_trimer_atom]
    )
    assert result.multi_conformer is False
    assert not hasattr(result, "conformer_positions")
    from src.dataset.mts_cache_integrity import _validate_trimer_record
    assert _validate_trimer_record(topology, result) == []


def test_equivalent_noncanonical_input_uses_same_canonical_identity():
    kwargs = dict(
        num_candidates=2, max_rounds=1, timeout_seconds=60,
        sample_key="canonical-source-identity",
    )
    first = attach_finite_trimer_mcl(_topology("*CCO*"), "*CCO*", **kwargs)
    second = attach_finite_trimer_mcl(_topology("*CCO*"), "*OCC*", **kwargs)
    assert first.trimer_geometry_valid and second.trimer_geometry_valid
    assert torch.equal(first.trimer_atomic_numbers, second.trimer_atomic_numbers)
    assert torch.equal(first.trimer_edge_index, second.trimer_edge_index)
    assert torch.equal(first.trimer_pos, second.trimer_pos)


def test_explicit_ez_is_retained_and_coordinate_audit_is_reorder_invariant():
    mol = Chem.AddHs(Chem.MolFromSmiles("F/C=C/Cl"))
    assert AllChem.EmbedMolecule(mol, AllChem.ETKDGv3()) == 0
    heavy = Chem.RemoveHs(mol)
    xyz = torch.tensor(heavy.GetConformer().GetPositions(), dtype=torch.float32)
    assert audit_double_bond_stereo_coordinates(heavy, xyz) == 1
    order = list(reversed(range(heavy.GetNumAtoms())))
    reordered = Chem.RenumberAtoms(heavy, order)
    assert audit_double_bond_stereo_coordinates(reordered, xyz[order]) == 1


def test_tetrahedral_stereo_accepts_declared_geometry_and_rejects_mirror():
    mol = Chem.AddHs(Chem.MolFromSmiles("N[C@@H](C)C(=O)O"))
    embedded, ids = _embedded_copy(mol, count=1)
    xyz = torch.tensor(embedded.GetConformer(ids[0]).GetPositions())
    assert audit_tetrahedral_stereo_coordinates(embedded, xyz) == 1
    mirrored = xyz.clone()
    mirrored[:, 0] *= -1
    with pytest.raises(_ExpectedGeometryFailure, match="TETRAHEDRAL_STEREO_MISMATCH"):
        audit_tetrahedral_stereo_coordinates(embedded, mirrored)


def test_tetrahedral_audit_is_atom_renumbering_invariant():
    mol = Chem.AddHs(Chem.MolFromSmiles("F[C@](Cl)(Br)I"))
    embedded, ids = _embedded_copy(mol, count=1)
    xyz = torch.tensor(embedded.GetConformer(ids[0]).GetPositions())
    assert audit_tetrahedral_stereo_coordinates(embedded, xyz) == 1
    order = list(reversed(range(embedded.GetNumAtoms())))
    reordered = Chem.RenumberAtoms(embedded, order)
    assert audit_tetrahedral_stereo_coordinates(reordered, xyz[order]) == 1


def test_unspecified_tetrahedral_stereo_is_not_rejected():
    mol = Chem.AddHs(Chem.MolFromSmiles("CC(O)C"))
    embedded, ids = _embedded_copy(mol, count=1)
    xyz = torch.tensor(embedded.GetConformer(ids[0]).GetPositions())
    mirrored = xyz.clone()
    mirrored[:, 0] *= -1
    assert audit_tetrahedral_stereo_coordinates(embedded, xyz) == 0
    assert audit_tetrahedral_stereo_coordinates(embedded, mirrored) == 0


def test_declared_tetrahedral_stereo_survives_source_to_all_atom_trimer():
    smiles = "*N[C@@H](C)C(=O)O*"
    topology = _topology(smiles)
    result = attach_finite_trimer_mcl(
        topology, smiles, num_candidates=2, max_rounds=2,
        sample_key="source-to-trimer-tetrahedral",
    )
    assert result.trimer_geometry_valid
    heavy, _ = build_periodic_multimer_mol(
        Chem.MolFromSmiles(Chem.MolToSmiles(Chem.MolFromSmiles(smiles), True)),
        3, close_periodic=False,
    )
    all_atom = Chem.AddHs(heavy)
    assert audit_tetrahedral_stereo_coordinates(
        all_atom, result.trimer_pos
    ) == 3
    from scripts.validate_dual_glt import audit_frozen_stereo
    assert audit_frozen_stereo(
        result, smiles, topology=topology, return_details=True
    ) == {'double_bonds': 0, 'tetrahedral_centers': 3, 'total': 3}


def test_fatal_identity_mapping_is_not_downgraded():
    data = _topology()
    data.canonical_to_trimer_base_atom_id = torch.full_like(
        data.canonical_ru_atom_index, 99
    )
    with pytest.raises(TrimerContractError):
        attach_finite_trimer_mcl(data, "*CCO*", max_rounds=0)


def test_total_embedding_failure_is_single_conformer_unavailable(monkeypatch):
    from src.dataset import trimer_mcl as module
    monkeypatch.setattr(
        module, "_embed_attempt", lambda mol, **kwargs: (Chem.Mol(mol), [])
    )
    result = attach_finite_trimer_mcl(
        _topology(), "*CCO*", num_candidates=8, max_rounds=2,
    )
    assert not result.trimer_geometry_valid
    assert result.search_stop_reason == "ETKDG_NO_VALID_CONFORMER"
    assert result.trimer_pos.shape == (0, 3)
    assert result.trimer_atomic_numbers.numel() == 0
    assert result.trimer_edge_index.shape == (2, 0)
    assert result.generation_diagnostics["num_candidates_requested"] == 16


def test_unknown_embed_exception_remains_fatal(monkeypatch):
    from src.dataset import trimer_mcl as module

    def explode(*args, **kwargs):
        raise RuntimeError("unexpected-worker-error")

    monkeypatch.setattr(module, "_embed_attempt", explode)
    with pytest.raises(RuntimeError, match="unexpected-worker-error"):
        attach_finite_trimer_mcl(_topology(), "*CCO*", max_rounds=1)


def test_single_conformer_lmdb_roundtrip(tmp_path):
    from src.dataset.lmdb_cache import LmdbLayerStore, LmdbLayerWriter
    record = attach_finite_trimer_mcl(
        _topology(), "*CCO*", num_candidates=1, max_rounds=1,
        sample_key="roundtrip",
    )
    root = tmp_path / "single"
    writer = LmdbLayerWriter(root, {"schema": "test-trimer-single"})
    key = b"x" * 32
    writer.add(key, record)
    store = writer.finalize()
    store.close()
    reopened = LmdbLayerStore(root)
    try:
        loaded = reopened[key]
        assert loaded.trimer_pos.ndim == 2
        assert torch.equal(loaded.trimer_pos, record.trimer_pos)
        assert torch.equal(loaded.h_parent_heavy_index, record.h_parent_heavy_index)
        assert not hasattr(loaded, "conformer_positions")
    finally:
        reopened.close()


def test_actual_generation_replay_is_deterministic():
    kwargs = dict(
        num_candidates=2, max_rounds=2, timeout_seconds=60,
        sample_key="deterministic-test-key",
    )
    first = attach_finite_trimer_mcl(_topology(), "*CCO*", **kwargs)
    second = attach_finite_trimer_mcl(_topology(), "*CCO*", **kwargs)
    assert first.trimer_geometry_valid == second.trimer_geometry_valid
    assert first.trimer_conformer_round_id == second.trimer_conformer_round_id
    assert first.trimer_conformer_candidate_id == second.trimer_conformer_candidate_id
    assert first.selected_converged == second.selected_converged
    assert first.trimer_conformer_energy == pytest.approx(
        second.trimer_conformer_energy, abs=1e-6
    )
    assert torch.allclose(first.trimer_pos, second.trimer_pos, atol=1e-6, rtol=0)


def test_over_384_heavy_atoms_enters_randomcoords_without_2d(monkeypatch):
    from src.dataset import trimer_mcl as module
    smiles = "*" + "C" * 129 + "*"
    source = Chem.MolFromSmiles(smiles)
    trimer, metadata = build_periodic_multimer_mol(source, 3, close_periodic=False)
    assert trimer.GetNumAtoms() > 384
    base_count = int(metadata["base_atom_count"])
    data = Data(
        num_nodes=base_count,
        graph_available=True,
        canonical_ru_atom_index=torch.arange(base_count),
        canonical_to_trimer_base_atom_id=torch.arange(base_count),
        z=torch.tensor([atom.GetAtomicNum() for atom in source.GetAtoms()
                        if atom.GetAtomicNum() > 0]),
    )
    called = {}

    def bounded_failure(molecule, **kwargs):
        called.update(kwargs)
        empty = Chem.Mol(molecule)
        empty.RemoveAllConformers()
        return empty, []

    monkeypatch.setattr(module, "_embed_attempt", bounded_failure)
    monkeypatch.setattr(
        AllChem, "Compute2DCoords",
        lambda *args, **kwargs: pytest.fail("2-D fallback called"),
    )
    result = attach_finite_trimer_mcl(
        data, source, num_candidates=8, max_rounds=1, timeout_seconds=60,
    )
    assert called["use_random_coords"] is True
    assert called["max_iterations"] == 200
    assert not result.trimer_geometry_valid
    assert not bool(result.trimer_2d_fallback)
