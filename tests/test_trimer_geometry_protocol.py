"""First-valid Trimer geometry protocol tests.

Covers: candidate budget (4/round, 2 rounds, max 8 attempts), early-stop
first-valid selection (A-D), energy-ranking removal, the shared 60 s
deadline, geometry-failure exclusion in the LMDB builder (no tombstone
records, rejection ledger, manifest accounting) and hard contract stops.
"""

import json
import sys
from pathlib import Path

import pytest
import torch
from rdkit import Chem

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset import trimer_mcl as module
from src.dataset.dataset import _compute_ru_base_layer, _compute_topology_layer
from src.dataset.trimer_mcl import (
    TRIMER_CANDIDATES_PER_ROUND,
    TRIMER_MAX_ROUNDS,
    TRIMER_SAMPLE_TIMEOUT_SECONDS,
    TrimerContractError,
    TrimerGeometryRejection,
    attach_finite_trimer_mcl,
)


def _topology(smiles="*CCO*"):
    ru = _compute_ru_base_layer(smiles)
    top = _compute_topology_layer(smiles, ru, max_hops=2)
    top.smiles = smiles
    return top


def test_protocol_constants():
    assert TRIMER_CANDIDATES_PER_ROUND == 4
    assert TRIMER_MAX_ROUNDS == 2
    assert TRIMER_CANDIDATES_PER_ROUND * TRIMER_MAX_ROUNDS == 8
    assert TRIMER_SAMPLE_TIMEOUT_SECONDS == 60.0
    with pytest.raises(ValueError):
        attach_finite_trimer_mcl(_topology(), "*CCO*", num_candidates=8)


def test_case_a_round0_candidate0_valid_stops_everything(monkeypatch):
    calls = []
    original = module._embed_attempt

    def wrapped(*args, **kwargs):
        calls.append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "_embed_attempt", wrapped)
    result = attach_finite_trimer_mcl(
        _topology(), "*CCO*", num_candidates=4, max_rounds=2,
        sample_key="case-a",
    )
    assert result.trimer_geometry_valid
    assert len(calls) == 1                       # attempts == 1
    assert result.trimer_conformer_candidate_id == 0
    assert result.trimer_conformer_round_id == 0
    assert result.generation_diagnostics["num_rounds"] == 1
    rows = result.generation_diagnostics["rounds"][0]["candidates"]
    assert len(rows) == 1                        # candidate 1/2/3 never attempted
    assert rows[0]["candidate_id"] == 0


def test_case_b_candidate0_fails_candidate1_valid(monkeypatch):
    calls = {"finite": 0}
    original_finite = module._conformer_coordinates_are_finite_3d

    def finite(*args, **kwargs):
        calls["finite"] += 1
        return calls["finite"] > 1               # candidate 0 pre-check fails

    monkeypatch.setattr(module, "_conformer_coordinates_are_finite_3d", finite)
    result = attach_finite_trimer_mcl(
        _topology(), "*CCO*", num_candidates=4, max_rounds=2,
        sample_key="case-b",
    )
    assert result.trimer_geometry_valid
    assert result.trimer_conformer_candidate_id == 1
    assert result.trimer_conformer_round_id == 0
    rows = result.generation_diagnostics["rounds"][0]["candidates"]
    assert [row["candidate_id"] for row in rows] == [0, 1]
    assert rows[0]["rejection"] == "PRE_MMFF_NONFINITE"
    assert rows[1]["final_valid"] is True
    assert len(rows) == 2                        # attempts == 2, early stop


def test_case_c_round0_all_fail_round1_candidate0_valid(monkeypatch):
    calls = {"finite": 0}
    original_finite = module._conformer_coordinates_are_finite_3d

    def finite(*args, **kwargs):
        calls["finite"] += 1
        return calls["finite"] > 4               # round0 pre-checks all fail

    monkeypatch.setattr(module, "_conformer_coordinates_are_finite_3d", finite)
    result = attach_finite_trimer_mcl(
        _topology(), "*CCO*", num_candidates=4, max_rounds=2,
        sample_key="case-c",
    )
    assert result.trimer_geometry_valid
    assert result.trimer_conformer_round_id == 1
    assert result.trimer_conformer_candidate_id == 0
    rows = [
        row for round_rows in result.generation_diagnostics["rounds"]
        for row in round_rows["candidates"]
    ]
    assert len(rows) == 5                        # attempts == 5
    assert result.generation_diagnostics["num_rounds"] == 2


def test_case_d_all_eight_fail_is_rejection_without_record(monkeypatch):
    monkeypatch.setattr(
        module, "_conformer_coordinates_are_finite_3d", lambda *a, **k: False
    )
    data = _topology()
    with pytest.raises(TrimerGeometryRejection) as excinfo:
        attach_finite_trimer_mcl(
            data, "*CCO*", num_candidates=4, max_rounds=2,
            sample_key="case-d",
        )
    rejection = excinfo.value
    assert rejection.candidate_attempts == 8
    assert rejection.code == "NO_VALID_CONFORMER"
    # no valid record was attached: the carrier keeps only the initial
    # "not_built" placeholder state and is never serialized by the builder
    assert bool(data.trimer_geometry_valid) is False
    assert data.search_stop_reason == "not_built"


def test_energy_ranking_is_gone_candidate0_wins(monkeypatch):
    optimize_calls = []

    def optimize(_mol, *, confId, **_kwargs):
        optimize_calls.append(int(confId))
        return 1  # non-converged

    monkeypatch.setattr(module.AllChem, "MMFFOptimizeMolecule", optimize)
    monkeypatch.setattr(
        module, "_calculate_mmff_energy",
        lambda _m, _p, conf_id: 1000.0,  # terrible energy, still accepted
    )
    result = attach_finite_trimer_mcl(
        _topology(), "*CCO*", num_candidates=4, max_rounds=2,
        sample_key="energy-removed",
    )
    assert result.trimer_geometry_valid
    assert result.trimer_conformer_candidate_id == 0
    assert result.selected_converged is False
    assert optimize_calls == [0]                 # no further candidates


def test_shared_deadline_is_not_per_round(monkeypatch):
    class FakeClock:
        def __init__(self):
            self.now = 0.0

        def __call__(self):
            return self.now

    clock = FakeClock()
    monkeypatch.setattr(module.time, "monotonic", clock)
    real_embed = module._embed_attempt

    def embed(*args, **kwargs):
        # candidate 0 consumes 50 s, candidate 1 pushes past the 60 s budget
        clock.now = 61.0 if clock.now >= 50.0 else 50.0
        return real_embed(*args, **kwargs)

    monkeypatch.setattr(module, "_embed_attempt", embed)
    monkeypatch.setattr(
        module, "_conformer_coordinates_are_finite_3d", lambda *a, **k: False
    )
    data = _topology()
    with pytest.raises(TrimerGeometryRejection) as excinfo:
        attach_finite_trimer_mcl(
            data, "*CCO*", num_candidates=4, max_rounds=2,
            timeout_seconds=60.0, sample_key="shared-deadline",
        )
    rejection = excinfo.value
    assert rejection.code == "TIMEOUT"
    assert rejection.round_reached == 0
    assert rejection.candidate_attempts == 2     # round 1 never started


def test_geometry_failure_exclusion_in_builder(monkeypatch, tmp_path):
    """Ordinary geometry failure -> key absent from Trimer LMDB, present in
    the rejection ledger, absent from accepted accounting; topology record
    still exists."""
    from torch_geometric.data import Data

    from src.dataset.dataset import UniDataset
    from src.dataset.lmdb_cache import LmdbLayerStore, sample_key_from_smiles

    good, bad = "*CC*", "*CCC*"
    # the builder hands attach_finite_trimer_mcl the canonical MONOMER mol,
    # so the patch matches on the monomer identity
    bad_identity = Chem.MolToSmiles(Chem.MolFromSmiles(bad), canonical=True)
    real_attach = module.attach_finite_trimer_mcl

    def patched_attach(data, smiles, **kwargs):
        identity = (
            kwargs.get("sample_key")
            if isinstance(smiles, str)
            else Chem.MolToSmiles(smiles, canonical=True)
        )
        if identity == bad_identity:
            raise TrimerGeometryRejection(
                "NO_VALID_CONFORMER", round_reached=1, candidate_attempts=8,
                elapsed_seconds=1.0,
            )
        return real_attach(data, smiles, **kwargs)

    monkeypatch.setattr(
        __import__("src.dataset.dataset", fromlist=["attach_finite_trimer_mcl"]),
        "attach_finite_trimer_mcl", patched_attach,
    )

    (tmp_path / "raw").mkdir(parents=True)
    (tmp_path / "raw" / "tiny.csv").write_text(
        "smiles,y\n" + good + ",1.0\n" + bad + ",1.0\n", encoding="utf-8"
    )
    dataset = UniDataset(
        root=str(tmp_path), dataset="tiny", smiles_model_name="",
        graph_encoder_type="mips_trimer_scage", graph_input="star_linking",
        use_feature_cache=True, feature_source_dataset="tiny",
        fp_mode="disabled", cache_layers="ru_base,topology,trimer",
        cache_validate="sample", feature_cache_workers=0,
        mips_core="paper_corrected", mips_max_hops=2,
        mips_use_descriptors=True, mips_descriptor_protocol="source_star_sub",
        spatial_mode="trimer_scage", graph_geometry_mode="trimer_scage_mcl",
        topology_representation="canonical_lifted", trimer_num_candidates=4,
        trimer_max_heavy_atoms=384, modalities=("graph",),
        experiment_id="exclusion-test", feature_config_hash="manual",
        require_frozen_store=False,
    )
    cache_root = tmp_path / "processed" / "mips_trimer_scage"
    trimer_roots = list((cache_root / "trimer").glob("*/"))
    topology_roots = list((cache_root / "topology").glob("*/"))
    assert len(trimer_roots) == 1 and len(topology_roots) == 1

    trimer_store = LmdbLayerStore(trimer_roots[0])
    topology_store = LmdbLayerStore(topology_roots[0])
    try:
        good_key = sample_key_from_smiles(good)
        bad_key = sample_key_from_smiles(bad)
        # accepted sample present with valid geometry
        record = trimer_store[good_key]
        assert bool(record.trimer_geometry_valid)
        # rejected sample: NO tombstone record at all
        assert bad_key not in trimer_store
        # topology records are unaffected by geometry exclusion
        assert good_key in topology_store and bad_key in topology_store
    finally:
        trimer_store.close()
        topology_store.close()

    manifest = json.loads(
        (trimer_roots[0] / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["source_count"] == 2
    assert manifest["accepted_count"] == 1
    assert manifest["rejected_count"] == 1
    assert manifest["source_count"] == manifest["accepted_count"] + manifest["rejected_count"]
    assert manifest["rejection_counts"] == {"NO_VALID_CONFORMER": 1}
    ledger = [
        json.loads(line)
        for line in (trimer_roots[0] / "rejections.jsonl")
        .read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(ledger) == 1
    assert ledger[0]["sample_key"] == sample_key_from_smiles(bad).hex()
    assert ledger[0]["failure_code"] == "NO_VALID_CONFORMER"
    assert ledger[0]["candidate_attempts"] == 8
    # the geometry-eligible cohort excludes the rejected key
    eligible = [bool(record.trimer_geometry_valid)]
    assert eligible == [True]


def test_contract_error_hard_stops_build(monkeypatch, tmp_path):
    """Identity corruption must abort the build: no .done, no .frozen."""
    from src.dataset.dataset import UniDataset

    bad_identity = Chem.MolToSmiles(Chem.MolFromSmiles("*CCC*"), canonical=True)
    real_attach = module.attach_finite_trimer_mcl

    def patched_attach(data, smiles, **kwargs):
        identity = (
            kwargs.get("sample_key")
            if isinstance(smiles, str)
            else Chem.MolToSmiles(smiles, canonical=True)
        )
        if identity == bad_identity:
            raise TrimerContractError("RU_MAPPING_CORRUPTION")
        return real_attach(data, smiles, **kwargs)

    monkeypatch.setattr(
        __import__("src.dataset.dataset", fromlist=["attach_finite_trimer_mcl"]),
        "attach_finite_trimer_mcl", patched_attach,
    )

    (tmp_path / "raw").mkdir(parents=True)
    (tmp_path / "raw" / "tiny.csv").write_text(
        "smiles,y\n*CC*,1.0\n*CCC*,1.0\n", encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="RU_MAPPING_CORRUPTION"):
        UniDataset(
            root=str(tmp_path), dataset="tiny", smiles_model_name="",
            graph_encoder_type="mips_trimer_scage", graph_input="star_linking",
            use_feature_cache=True, feature_source_dataset="tiny",
            fp_mode="disabled", cache_layers="ru_base,topology,trimer",
            cache_validate="sample", feature_cache_workers=0,
            mips_core="paper_corrected", mips_max_hops=2,
            mips_use_descriptors=True, mips_descriptor_protocol="source_star_sub",
            spatial_mode="trimer_scage", graph_geometry_mode="trimer_scage_mcl",
            topology_representation="canonical_lifted", trimer_num_candidates=4,
            trimer_max_heavy_atoms=384, modalities=("graph",),
            experiment_id="contract-test", feature_config_hash="manual",
            require_frozen_store=False,
        )
    cache_root = tmp_path / "processed" / "mips_trimer_scage"
    trimer_roots = list((cache_root / "trimer").glob("*/"))
    if trimer_roots:
        assert not (trimer_roots[0] / ".done").exists()
        assert not (trimer_roots[0] / ".frozen").exists()
