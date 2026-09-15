import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from src.dataset.cache_lifecycle import CacheLifecycleError, json_hash
from src.dataset.glt_dual_cache import load_active_dual_store, load_dual_cohort
from src.dataset.lmdb_cache import sample_key_from_normalized


def _active_store(tmp_path: Path):
    bundle_hash = "a" * 64
    artifacts = {}
    bundle_artifacts = {}
    for index, layer in enumerate(("ru_base", "topology", "trimer"), 1):
        spec = {"artifact_type": layer, "parameters": {"marker": index}}
        spec_hash = hashlib.sha256(json.dumps(
            spec, sort_keys=True, separators=(",", ":")
        ).encode()).hexdigest()
        artifact_hash = str(index) * 64
        manifest_hash = str(index + 3) * 64
        artifacts[layer] = {
            "path": f"builds/{bundle_hash}/{layer}",
            "build_spec": spec,
            "build_spec_hash": spec_hash,
            "artifact_hash": artifact_hash,
            "manifest_hash": manifest_hash,
            "record_count": 1,
        }
        bundle_artifacts[layer] = {
            "artifact_hash": artifact_hash, "manifest_hash": manifest_hash,
        }
        (tmp_path / "builds" / bundle_hash / layer).mkdir(parents=True)
    source_hash = "f" * 64
    store = {
        "bundle_hash": bundle_hash,
        "source": {"path": f"builds/{bundle_hash}/source",
                   "source_manifest_hash": source_hash, "source_count": 1},
        "artifacts": artifacts,
    }
    (tmp_path / "store.json").write_text(json.dumps(store), encoding="utf-8")
    (tmp_path / "builds" / bundle_hash / "bundle_manifest.json").write_text(
        json.dumps({"bundle_hash": bundle_hash,
                    "source_manifest_hash": source_hash,
                    "artifacts": bundle_artifacts}), encoding="utf-8"
    )
    return store


def _cohort(tmp_path: Path, store: dict):
    root = tmp_path / "cohort"
    root.mkdir()
    normalized = "*CC(C)*"
    key = sample_key_from_normalized(normalized)
    keys = np.frombuffer(key, dtype=np.uint8).reshape(1, 32)
    np.save(root / "keys.npy", keys)
    row = {
        "sample_key": key.hex(), "source_row": 7,
        "source_smiles": "*C(C)C*", "normalized_smiles": normalized,
    }
    (root / "records.jsonl").write_text(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {
        "artifact_type": "glt_dual_formal_cohort",
        "identity_policy": "unique_structure",
        "ordering_policy": "source_manifest_order_filtered_by_topology_and_trimer_acceptance",
        "sample_count": 1, "duplicate_count": 0,
        "ordered_sample_key_hash": hashlib.sha256(key).hexdigest(),
        "main_bundle_hash": store["bundle_hash"],
        "source_manifest_hash": store["source"]["source_manifest_hash"],
        "topology_manifest_hash": store["artifacts"]["topology"]["manifest_hash"],
        "trimer_manifest_hash": store["artifacts"]["trimer"]["manifest_hash"],
        "keys_file_sha256": digest(root / "keys.npy"),
        "records_file_sha256": digest(root / "records.jsonl"),
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (root / ".frozen").write_text(
        json.dumps({"manifest_hash": json_hash(manifest)}), encoding="utf-8"
    )
    return root


def test_active_bundle_is_selected_only_by_store_json(tmp_path):
    store = _active_store(tmp_path)
    assert load_active_dual_store(tmp_path)["bundle_hash"] == store["bundle_hash"]
    binding = store["artifacts"]["trimer"]
    binding["path"] = "builds/latest-looking/trimer"
    (tmp_path / "store.json").write_text(json.dumps(store), encoding="utf-8")
    with pytest.raises(CacheLifecycleError, match="active bundle"):
        load_active_dual_store(tmp_path)


def test_cohort_uses_explicit_normalized_identity_and_all_bindings(tmp_path):
    store = _active_store(tmp_path)
    root = _cohort(tmp_path, store)
    cohort = load_dual_cohort(root, tmp_path)
    assert cohort["records"][0]["source_smiles"] != cohort["records"][0]["normalized_smiles"]
    assert bytes(cohort["keys_array"][0]) == sample_key_from_normalized(
        cohort["records"][0]["normalized_smiles"]
    )
    manifest = json.loads((root / "manifest.json").read_text())
    manifest["trimer_manifest_hash"] = "0" * 64
    (root / "manifest.json").write_text(json.dumps(manifest))
    (root / ".frozen").write_text(
        json.dumps({"manifest_hash": json_hash(manifest)})
    )
    with pytest.raises(CacheLifecycleError, match="trimer_manifest_hash"):
        load_dual_cohort(root, tmp_path)


def test_cohort_tampering_is_rejected(tmp_path):
    store = _active_store(tmp_path)
    root = _cohort(tmp_path, store)
    with (root / "records.jsonl").open("a", encoding="utf-8") as handle:
        handle.write("{}\n")
    with pytest.raises(CacheLifecycleError, match="records file hash"):
        load_dual_cohort(root, tmp_path)
