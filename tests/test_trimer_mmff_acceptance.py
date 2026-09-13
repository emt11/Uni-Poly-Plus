"""Stage-A contract tests for deterministic multi-conformer Trimers."""

import copy

import pytest
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from torch_geometric.data import Data

from src.dataset.dataset import _compute_ru_base_layer, _compute_topology_layer
from src.dataset.graph_data import build_periodic_multimer_mol
from src.dataset.trimer_mcl import (
    TrimerContractError, _EnsembleCandidate, _attach_placeholder, _deduplicate,
    _round_seed, attach_finite_trimer_mcl,
    audit_double_bond_stereo_coordinates, fixed_identity_rmsd,
    iter_conformers, select_conformer,
)


def _topology(smiles="*CCO*"):
    ru = _compute_ru_base_layer(smiles)
    top = _compute_topology_layer(smiles, ru, max_hops=2)
    top.smiles = smiles
    return top


def _ensemble(k, n=5):
    record = Data()
    record.multi_conformer = True
    record.num_conformers = k
    record.conformer_positions = torch.arange(k * n * 3, dtype=torch.float32).reshape(k, n, 3)
    record.conformer_energies = torch.arange(k, dtype=torch.float32)
    record.conformer_round_ids = torch.zeros(k, dtype=torch.long)
    record.conformer_candidate_ids = torch.arange(k, dtype=torch.long)
    record.trimer_edge_index = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]])
    record.trimer_base_ru_atom_id = torch.arange(n)
    record.trimer_ru_offset = torch.tensor([-1, 0, 0, 1, 1])
    return record


@pytest.mark.parametrize("k", range(5))
def test_ensemble_k_shapes_and_explicit_selection(k):
    record = _ensemble(k)
    before = copy.deepcopy(record)
    assert len(list(iter_conformers(record))) == k
    assert torch.equal(record.conformer_positions, before.conformer_positions)
    assert not hasattr(record, "trimer_pos")
    if k:
        view = select_conformer(record, k - 1)
        assert view.trimer_pos.shape == (5, 3)
        assert view.selected_conformer_index == k - 1
    else:
        with pytest.raises(IndexError):
            select_conformer(record, 0)


def test_old_single_conformer_record_is_not_silently_upgraded():
    with pytest.raises(ValueError, match="not a multi-conformer"):
        select_conformer(Data(trimer_pos=torch.zeros(3, 3)), 0)


def test_round_seed_is_deterministic_and_round_specific():
    assert _round_seed("abc", 1) == _round_seed("abc", 1)
    assert _round_seed("abc", 1) != _round_seed("abc", 2)


def test_stable_energy_round_candidate_sorting_and_dedup():
    base = torch.tensor([[0., 0., 0.], [1., 0., 0.], [0., 2., 0.], [0., 0., 3.]])
    distinct = base.clone(); distinct[3, 0] = 4.0
    candidates = [
        _EnsembleCandidate(distinct, 1.0, 1, 0),
        _EnsembleCandidate(base + 8.0, 1.0, 0, 2),
        _EnsembleCandidate(base, 1.0, 0, 1),
    ]
    kept, duplicates = _deduplicate(candidates, target=4, threshold=0.3)
    assert [(c.round_id, c.candidate_id) for c in kept] == [(0, 1), (1, 0)]
    assert duplicates == 1


def test_fixed_identity_rmsd_translation_rotation_invariant():
    xyz = torch.tensor([[0., 0., 0.], [1., 0., 0.], [0., 2., 0.], [0., 0., 3.]])
    rotation = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    assert fixed_identity_rmsd(xyz, xyz @ rotation + 4.0) < 1e-6


def test_fixed_identity_rmsd_forbids_reflection_and_permutation():
    xyz = torch.tensor([[0., 0., 0.], [1., 0., 0.], [0., 2., 0.], [0., 0., 3.]])
    reflected = xyz.clone(); reflected[:, 2] *= -1
    assert fixed_identity_rmsd(xyz, reflected) > 0.3
    assert fixed_identity_rmsd(xyz, xyz[[3, 1, 2, 0]]) > 0.3


def test_fixed_identity_rmsd_forbids_ru_swap():
    xyz = torch.tensor([[0., 0., 0.], [1., 0., 0.], [3., 1., 0.], [7., 1., 2.]])
    assert fixed_identity_rmsd(xyz, xyz[[2, 3, 0, 1]]) > 0.3


def test_explicit_ez_is_retained_in_all_preservable_ru_copies():
    mol, _ = build_periodic_multimer_mol("*C/C=C/C*", 3, close_periodic=False)
    defined = [b for b in mol.GetBonds() if b.GetStereo() in {
        Chem.BondStereo.STEREOE, Chem.BondStereo.STEREOZ,
        Chem.BondStereo.STEREOCIS, Chem.BondStereo.STEREOTRANS}]
    assert len(defined) == 3
    assert all(len(tuple(b.GetStereoAtoms())) == 2 for b in defined)


def test_real_ensemble_has_no_2d_or_implicit_conformer_zero():
    result = attach_finite_trimer_mcl(
        _topology("*CCO*"), "*CCO*", num_candidates=2, max_rounds=1,
        target_conformers=1, timeout_seconds=60)
    assert not bool(result.trimer_2d_fallback)
    assert result.conformer_positions.dtype == torch.float32
    assert not hasattr(result, "trimer_pos")
    if result.num_conformers:
        assert select_conformer(result, 0).trimer_pos.ndim == 2


def test_large_threshold_argument_does_not_trigger_2d(monkeypatch):
    called = {"embed": False}
    from src.dataset import trimer_mcl as module
    original = module._embed_attempt
    def wrapped(*args, **kwargs):
        called["embed"] = True
        return original(*args, **kwargs)
    monkeypatch.setattr(module, "_embed_attempt", wrapped)
    result = attach_finite_trimer_mcl(
        _topology("*CCO*"), "*CCO*", max_heavy_atoms=1,
        num_candidates=1, max_rounds=1, target_conformers=1)
    assert called["embed"]
    assert not bool(result.trimer_2d_fallback)


def test_fatal_identity_mapping_is_not_downgraded():
    data = _topology("*CCO*")
    data.canonical_to_trimer_base_atom_id = torch.full_like(
        data.canonical_ru_atom_index, 99)
    with pytest.raises(TrimerContractError):
        attach_finite_trimer_mcl(data, "*CCO*", max_rounds=0)


def test_k0_placeholder_has_empty_ensemble_and_no_2d():
    data = Data(num_nodes=0)
    _attach_placeholder(data, "ETKDG_NO_VALID_CONFORMER", 1)
    assert data.conformer_positions.shape == (0, 0, 3)
    assert data.num_conformers == 0
    assert not bool(data.trimer_2d_fallback)


def test_stereo_audit_accepts_rdkit_ez_geometry_and_atom_reorder():
    from rdkit.Chem import AllChem
    mol = Chem.AddHs(Chem.MolFromSmiles("F/C=C/Cl"))
    assert AllChem.EmbedMolecule(mol, AllChem.ETKDGv3()) == 0
    heavy = Chem.RemoveHs(mol)
    xyz = torch.tensor(heavy.GetConformer().GetPositions(), dtype=torch.float32)
    assert audit_double_bond_stereo_coordinates(heavy, xyz) == 1
    order = list(reversed(range(heavy.GetNumAtoms())))
    reordered = Chem.RenumberAtoms(heavy, order)
    assert audit_double_bond_stereo_coordinates(reordered, xyz[order]) == 1


def test_etkdg_total_failure_is_k0_not_fatal(monkeypatch):
    from src.dataset import trimer_mcl as module
    monkeypatch.setattr(module, "_embed_attempt", lambda mol, **kwargs: (Chem.Mol(mol), []))
    record = attach_finite_trimer_mcl(
        _topology("*CCO*"), "*CCO*", max_rounds=1, target_conformers=4)
    assert record.num_conformers == 0
    assert record.search_stop_reason == "ETKDG_NO_VALID_CONFORMER"
    assert record.conformer_positions.shape[1] == record.trimer_atomic_number.numel()


def test_unknown_embed_exception_remains_fatal(monkeypatch):
    from src.dataset import trimer_mcl as module
    def explode(*args, **kwargs):
        raise RuntimeError("unexpected-worker-error")
    monkeypatch.setattr(module, "_embed_attempt", explode)
    with pytest.raises(RuntimeError, match="unexpected-worker-error"):
        attach_finite_trimer_mcl(_topology("*CCO*"), "*CCO*", max_rounds=1)


def test_timeout_preserves_completed_state(monkeypatch):
    from src.dataset import trimer_mcl as module
    ticks = iter((0.0, 241.0, 242.0))
    monkeypatch.setattr(module.time, "monotonic", lambda: next(ticks))
    record = attach_finite_trimer_mcl(_topology("*CCO*"), "*CCO*", max_rounds=4)
    assert record.num_conformers == 0
    assert record.search_stop_reason == "timeout"


def test_multi_conformer_lmdb_roundtrip(tmp_path):
    from src.dataset.lmdb_cache import LmdbLayerStore, LmdbLayerWriter
    root = tmp_path / "ensemble"
    meta = {"schema": "test-trimer-ensemble", "multi_conformer": True}
    writer = LmdbLayerWriter(root, meta)
    key = b"x" * 32
    writer.add(key, _ensemble(2))
    store = writer.finalize()
    store.close()
    reopened = LmdbLayerStore(root)
    try:
        record = reopened[key]
        assert record.num_conformers == 2
        assert torch.equal(select_conformer(record, 1).trimer_pos,
                           record.conformer_positions[1])
    finally:
        reopened.close()


def test_actual_generation_replay_is_deterministic():
    kwargs = dict(num_candidates=2, max_rounds=2, target_conformers=2,
                  rmsd_threshold=0.3, timeout_seconds=60,
                  sample_key="deterministic-test-key")
    first = attach_finite_trimer_mcl(_topology("*CCO*"), "*CCO*", **kwargs)
    second = attach_finite_trimer_mcl(_topology("*CCO*"), "*CCO*", **kwargs)
    assert first.num_conformers == second.num_conformers
    assert first.generation_diagnostics["num_candidates_embedded"] == second.generation_diagnostics["num_candidates_embedded"]
    assert torch.equal(first.conformer_round_ids, second.conformer_round_ids)
    assert torch.equal(first.conformer_candidate_ids, second.conformer_candidate_ids)
    assert torch.allclose(first.conformer_energies, second.conformer_energies,
                          atol=1e-6, rtol=0)
    assert torch.allclose(first.conformer_positions, second.conformer_positions,
                          atol=1e-6, rtol=0)


def test_over_384_heavy_atoms_enters_real_etkdg_without_2d(monkeypatch):
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
        z=torch.tensor([source.GetAtomWithIdx(i).GetAtomicNum()
                        for i in range(source.GetNumAtoms())
                        if source.GetAtomWithIdx(i).GetAtomicNum() > 0]),
    )
    called = {"actual_etkdg": False}
    original = module._embed_attempt

    def actual_then_bounded_failure(molecule, **kwargs):
        called["actual_etkdg"] = True
        embedded, _ = original(
            molecule, num_candidates=1, seed=kwargs["seed"],
            use_random_coords=False, max_iterations=1)
        embedded.RemoveAllConformers()
        return embedded, []

    monkeypatch.setattr(module, "_embed_attempt", actual_then_bounded_failure)
    monkeypatch.setattr(AllChem, "Compute2DCoords",
                        lambda *args, **kwargs: pytest.fail("2-D fallback called"))
    result = attach_finite_trimer_mcl(
        data, source, num_candidates=8, max_rounds=1,
        target_conformers=4, timeout_seconds=60)
    assert called["actual_etkdg"]
    assert result.num_conformers == 0
    assert result.search_stop_reason == "ETKDG_NO_VALID_CONFORMER"
    assert not bool(result.trimer_2d_fallback)
