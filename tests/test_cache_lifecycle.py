import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch_geometric.data import Data

from scripts.build_mts_cache import (
    _audit_freeze_publish,
    _metadata,
    _run_trimer,
)
from src.dataset.cache_lifecycle import (
    CacheLifecycleError,
    PublishedCacheDataset,
    StagingWriter,
    append_jsonl,
    artifact_identity,
    bundle_identity,
    json_hash,
    load_source_rows,
    sidecar_binding,
    snapshot_tree,
    validate_sidecar_binding,
)
from src.dataset.cache_spec import (
    ROUTE_BUILD_SPECS,
    TRIMER_BUILD_SPEC,
)
from src.dataset.mts_target_contract import make_trimer_metadata
from src.dataset.trimer_mcl import TrimerContractError, attach_finite_trimer_mcl


def _ru(normalized="*CC*"):
    return Data(
        normalized_polymer_smiles=normalized,
        ru_chemistry_valid=True,
        ru_base_valid=True,
        ru_base_failure_code="",
        ru_atomic_number=torch.tensor([6, 6]),
        ru_edge_index=torch.tensor([[0, 1], [1, 0]]),
        ru_bond_type=torch.ones(2, dtype=torch.long),
    )


def _topology():
    return Data(
        mips_x=torch.zeros((2, 137)),
        z=torch.tensor([6, 6]),
        edge_index=torch.tensor([[0, 1], [1, 0]]),
        canonical_ru_atom_index=torch.arange(2),
        graph_available=True,
        lga_edge_index=torch.tensor([[0, 1], [0, 1]]),
        lga_path_index=torch.tensor([[0, -1, -1], [1, -1, -1]]),
    )


def _trimer():
    return Data(
        trimer_geometry_valid=True,
        trimer_geometry_is_3d=torch.tensor(True),
        trimer_2d_fallback=torch.tensor(False),
        trimer_pos=torch.tensor([[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]]),
        trimer_atomic_number=torch.tensor([6, 6]),
        trimer_edge_index=torch.tensor([[0, 1], [1, 0]]),
        trimer_bond_type=torch.ones(2, dtype=torch.long),
        trimer_central_ru_mask=torch.tensor([True, True]),
        mips_to_trimer_central_index=torch.tensor([0, 1]),
        o8_heavy_mask=torch.tensor([True, True]),
        o8_heavy_indices=torch.tensor([0, 1]),
        trimer_heavy_mask=torch.tensor([True, True]),
        trimer_heavy_indices=torch.tensor([0, 1]),
        star_3d_distance=torch.tensor(1.4),
        star_3d_valid=torch.tensor(True),
    )


def _world(tmp_path):
    source_hash = "1" * 64
    artifact_hashes = {}
    artifact_hashes["ru_base"] = artifact_identity("ru_base", source_hash, {})
    artifact_hashes["topology"] = artifact_identity(
        "topology", source_hash, {"ru_base": artifact_hashes["ru_base"]}
    )
    artifact_hashes["trimer"] = artifact_identity(
        "trimer", source_hash, {"ru_base": artifact_hashes["ru_base"]}
    )
    bundle_hash = bundle_identity(source_hash, artifact_hashes)
    staging = tmp_path / "builds" / f"{bundle_hash}.staging"
    final = tmp_path / "builds" / bundle_hash
    staging.mkdir(parents=True)
    (staging / "source").mkdir()
    source_manifest = {"source_manifest_hash": source_hash}
    (staging / "source" / "manifest.json").write_text(
        json.dumps(source_manifest), encoding="utf-8"
    )
    parents = {
        "ru_base": {},
        "topology": {"ru_base": artifact_hashes["ru_base"]},
        "trimer": {"ru_base": artifact_hashes["ru_base"]},
    }
    metadata = {
        layer: _metadata(layer, artifact_hashes[layer], source_hash, parents[layer])
        for layer in ("ru_base", "topology", "trimer")
    }
    return source_manifest, artifact_hashes, metadata, staging, final


def test_build_spec_is_generator_and_metadata_source(monkeypatch):
    metadata = make_trimer_metadata(
        ru_base_hash="a" * 64,
        topology_hash="b" * 64,
        rdkit_version="test",
    )
    assert metadata["build_spec"] == TRIMER_BUILD_SPEC
    assert set(TRIMER_BUILD_SPEC["parents"]) == {"ru_base"}
    assert (
        TRIMER_BUILD_SPEC["parameters"]["seed_policy"]["geometry_seed_spec"]
        ["terminal_capping"]
        == "missing_seam_bond_order_hydrogen_equivalents"
    )
    data = Data(num_nodes=1, graph_available=False)
    with pytest.raises(ValueError, match="must come from build_spec"):
        attach_finite_trimer_mcl(
            data, "*C*", build_spec=TRIMER_BUILD_SPEC, num_candidates=3
        )


def test_pilot_source_manifest_reports_invalid_candidates(tmp_path):
    source = tmp_path / "source.csv"
    source.write_text("SMILES,value\nnot-a-smiles,0\n*CC*,1\n*CC*,2\n*CO*,3\n")
    rows, manifest = load_source_rows(source, 2)
    assert [row["source_smiles"] for row in rows] == ["*CC*", "*CO*"]
    selection = manifest["selection"]
    assert selection["invalid_candidates_excluded"] == 1
    assert selection["duplicate_candidates_excluded"] == 1
    assert selection["raw_candidates_examined"] == 4


def test_lifecycle_audit_freeze_publish_and_zero_write(tmp_path):
    source, hashes, metadata, staging, final = _world(tmp_path)
    keys = [bytes([value]) * 32 for value in (1, 2)]
    rows = [
        {"sample_key": key.hex(), "source_smiles": "*CC*",
         "normalized_smiles": "*CC*", "source_row": index}
        for index, key in enumerate(keys)
    ]
    for layer, values in (
        ("ru_base", [_ru(), _ru()]),
        ("topology", [_topology(), _topology()]),
        ("trimer", [_trimer()]),
    ):
        writer = StagingWriter(staging / layer, metadata[layer])
        try:
            for key, value in zip(keys, values):
                assert writer.put(key, value)[0]
        finally:
            writer.close()
    append_jsonl(staging / "trimer" / "rejections.jsonl", {
        "sample_key": keys[1].hex(), "failure_code": "MMFF_UNSUPPORTED",
        "candidate_attempts": 0, "elapsed_seconds": 0.01,
        "round_reached": -1,
    })
    for key, status in zip(keys, ("accepted", "rejected")):
        append_jsonl(staging / "trimer" / "runtime.jsonl", {
            "sample_key": key.hex(), "status": status,
            "elapsed_seconds": 0.01, "candidate_attempts": 1,
            "round_reached": 0, "record_bytes": 1 if status == "accepted" else 0,
            "writer_seconds": 0.001 if status == "accepted" else 0.0,
        })
    store, manifests, rejections = _audit_freeze_publish(
        tmp_path, staging, final, rows, source, hashes, metadata
    )
    assert not staging.exists()
    assert manifests["trimer"]["accepted_count"] == 1
    assert manifests["trimer"]["rejected_count"] == 1
    assert set(rejections) == {keys[1]}
    assert set(store["artifacts"]["trimer"]["parents"]) == {"ru_base"}
    marker = json.loads((final / "trimer" / ".frozen").read_text())
    manifest = json.loads((final / "trimer" / "manifest.json").read_text())
    assert marker == {"manifest_hash": json_hash(manifest)}

    before = snapshot_tree(tmp_path)
    dataset = PublishedCacheDataset(tmp_path)
    try:
        assert len(dataset) == 1
        item = dataset[0]
        assert item.mips_x.shape == (2, 137)
        assert "x" not in item.keys()
        assert "trimer_atomic_numbers" not in item.keys()
    finally:
        dataset.close()
    assert snapshot_tree(tmp_path) == before


def test_sidecar_binding_rejects_any_parent_or_cohort_change():
    parent = {"artifact_hash": "a" * 64}
    spec = {"feature": "bond_angle", "radius": 2}
    metadata = sidecar_binding(parent, "b" * 64, spec)
    validate_sidecar_binding(metadata, parent, "b" * 64, spec)
    with pytest.raises(CacheLifecycleError):
        validate_sidecar_binding(metadata, {"artifact_hash": "c" * 64}, "b" * 64, spec)
    with pytest.raises(CacheLifecycleError):
        validate_sidecar_binding(metadata, parent, "d" * 64, spec)


def test_multiworker_contract_error_is_not_rejection(tmp_path):
    source, hashes, metadata, staging, _ = _world(tmp_path)
    key = b"z" * 32
    writer = StagingWriter(staging / "ru_base", metadata["ru_base"])
    try:
        writer.put(key, _ru("not-a-smiles"))
    finally:
        writer.close()
    rows = [{
        "sample_key": key.hex(), "source_smiles": "not-a-smiles",
        "normalized_smiles": "not-a-smiles", "source_row": 0,
    }]
    with pytest.raises(TrimerContractError):
        _run_trimer(staging, rows, metadata, workers=2, interrupt_after=0)
    assert (staging / "trimer" / "rejections.jsonl").read_text() == ""
    assert not (staging / "trimer" / ".frozen").exists()
