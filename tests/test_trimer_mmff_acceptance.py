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
        _topology(), "*CCO*", num_candidates=4, max_rounds=2,
        timeout_seconds=60, sample_key="randomcoords-contract",
    )
    assert result.trimer_geometry_valid
    # first-valid early stop: exactly ONE candidate embed (candidate 0)
    assert len(calls) == 1
    assert calls[0]["use_random_coords"] is True
    assert calls[0]["max_iterations"] == 200
    assert calls[0]["num_candidates"] == 1
    assert result.generation_diagnostics["num_rounds"] == 1
    candidate_rows = result.generation_diagnostics["rounds"][0]["candidates"]
    assert len(candidate_rows) == 1
    assert candidate_rows[0]["candidate_id"] == 0
    assert candidate_rows[0]["finite_pre_mmff"] is True
    assert candidate_rows[0]["double_bond_stereo_pre_pass"] is True
    assert candidate_rows[0]["tetra_stereo_pre_pass"] is True
    assert candidate_rows[0]["mmff_energy_finite"] is True
    assert candidate_rows[0]["finite_post_mmff"] is True
    assert candidate_rows[0]["double_bond_stereo_post_f64_pass"] is True
    assert candidate_rows[0]["tetra_stereo_post_f32_pass"] is True
    assert candidate_rows[0]["final_valid"] is True
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
    assert result.trimer_conformer_round_id == 1
    assert result.trimer_conformer_candidate_id == 0


def test_first_valid_accepts_candidate0_and_never_ranks_energy(monkeypatch):
    """First-valid protocol: candidate 0 is accepted even when a later
    candidate would be 'better' (converged / lower energy); the remaining
    candidates are never attempted and no energy comparator runs."""
    from src.dataset import trimer_mcl as module
    optimize_calls = []
    energy_calls = []

    def optimize(_mol, *, confId, **_kwargs):
        optimize_calls.append(int(confId))
        return 1  # candidate 0 is NON-converged

    def energy(_mol, _props, conf_id):
        energy_calls.append(int(conf_id))
        return 100.0  # deliberately terrible energy

    monkeypatch.setattr(AllChem, "MMFFOptimizeMolecule", optimize)
    monkeypatch.setattr(module, "_calculate_mmff_energy", energy)
    result = attach_finite_trimer_mcl(
        _topology(), "*CCO*", num_candidates=4, max_rounds=2,
        sample_key="first-valid-contract",
    )
    # candidate 0 accepted despite being non-converged with high energy
    assert result.trimer_geometry_valid
    assert result.trimer_conformer_candidate_id == 0
    assert result.trimer_conformer_round_id == 0
    assert result.selected_converged is False
    assert result.trimer_conformer_energy == 100.0
    # early stop: candidate 1/2/3 and round 1 never executed
    assert optimize_calls == [0]
    assert energy_calls == [0]
    rows = result.generation_diagnostics["rounds"][0]["candidates"]
    assert len(rows) == 1
    assert result.generation_diagnostics["num_rounds"] == 1


def test_finite_nonconverged_candidate_is_explicit_fallback(monkeypatch):
    monkeypatch.setattr(AllChem, "MMFFOptimizeMolecule", lambda *args, **kwargs: 1)
    result = attach_finite_trimer_mcl(
        _topology(), "*CCO*", num_candidates=1, max_rounds=1,
        sample_key="nonconverged-fallback",
    )
    assert result.trimer_geometry_valid
    assert result.selected_converged is False
    assert result.generation_diagnostics["rounds"][0]["candidates"][0]["mmff_status"] == 1


def test_mmff_unsupported_is_ordinary_rejection(monkeypatch):
    from src.dataset import trimer_mcl as module
    monkeypatch.setattr(AllChem, "MMFFGetMoleculeProperties", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        module, "_embed_attempt",
        lambda *args, **kwargs: pytest.fail("embedding must not run"),
    )
    with pytest.raises(module.TrimerGeometryRejection, match="MMFF_UNSUPPORTED"):
        attach_finite_trimer_mcl(_topology(), "*CCO*")


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


def test_open_h_caps_preserve_charged_aromatic_attachment_ru_bonds():
    """A terminal aromatic [n+] needs an explicit H-cap declaration.

    This is the real pilot structure that previously made only RU+1 lose
    aromaticity during sanitization and triggered the internal-bond contract.
    """
    smiles = (
        "*COc1ccc(CSC(=O)c2ccc(C(=O)OCCCCOC(=O)"
        "c3ccc[n+](*)c3)cc2)cc1"
    )
    trimer, metadata = build_periodic_multimer_mol(
        smiles, num_repeat_units=3, close_periodic=False
    )

    unit_bonds = []
    for unit in metadata["unit_atoms"]:
        reverse = {atom_index: base_id for base_id, atom_index in enumerate(unit)}
        unit_bonds.append({
            (
                min(reverse[bond.GetBeginAtomIdx()], reverse[bond.GetEndAtomIdx()]),
                max(reverse[bond.GetBeginAtomIdx()], reverse[bond.GetEndAtomIdx()]),
                str(bond.GetBondType()),
                bool(bond.GetIsAromatic()),
            )
            for bond in trimer.GetBonds()
            if bond.GetBeginAtomIdx() in reverse
            and bond.GetEndAtomIdx() in reverse
        })
    assert unit_bonds[0] == unit_bonds[1] == unit_bonds[2]

    right_terminal = trimer.GetAtomWithIdx(metadata["unit_right_boundaries"][-1])
    assert right_terminal.GetSymbol() == "N"
    assert right_terminal.GetFormalCharge() == 1
    assert right_terminal.GetIsAromatic()
    assert right_terminal.GetNumExplicitHs() == 1
    assert metadata["terminal_cap_hydrogens"] == [
        {
            "side": "left",
            "atom": metadata["unit_left_boundaries"][0],
            "count": 1,
        },
        {
            "side": "right",
            "atom": metadata["unit_right_boundaries"][-1],
            "count": 1,
        },
    ]
    with_hydrogens = Chem.AddHs(trimer)
    capped_nitrogen = with_hydrogens.GetAtomWithIdx(
        metadata["unit_right_boundaries"][-1]
    )
    assert sum(
        neighbor.GetAtomicNum() == 1 for neighbor in capped_nitrogen.GetNeighbors()
    ) == 1


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


def test_total_embedding_failure_is_ordinary_rejection(monkeypatch):
    from src.dataset import trimer_mcl as module
    monkeypatch.setattr(
        module, "_embed_attempt", lambda mol, **kwargs: (Chem.Mol(mol), [])
    )
    with pytest.raises(module.TrimerGeometryRejection) as excinfo:
        attach_finite_trimer_mcl(
            _topology(), "*CCO*", num_candidates=4, max_rounds=2,
        )
    rejection = excinfo.value
    assert rejection.code == "ETKDG_NO_VALID_CONFORMER"
    assert rejection.candidate_attempts == 8      # 4 + 4
    assert rejection.round_reached == 1
    assert rejection.last_embed_failure == "ETKDG_NO_VALID_CONFORMER"


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
    with pytest.raises(module.TrimerGeometryRejection):
        attach_finite_trimer_mcl(
            data, source, num_candidates=4, max_rounds=1, timeout_seconds=60,
        )
    assert called["use_random_coords"] is True
    assert called["max_iterations"] == 200
