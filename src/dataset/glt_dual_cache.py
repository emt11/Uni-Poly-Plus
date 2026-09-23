"""Strict frozen-bundle and cohort reader for the current dual GLT route.

The training path is deliberately read-only::

    store.json -> active bundle -> layer manifest/.frozen -> readonly LMDB
               -> frozen dual cohort -> deterministic sample order

No directory discovery, cache repair, migration, SMILES re-canonicalisation
fallback, or online sample filtering is permitted here.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .cache_lifecycle import CacheLifecycleError, ReadonlyArtifact, json_hash
from .cache_spec import validate_store
from .lmdb_cache import sample_key_from_normalized


COHORT_REQUIRED_FIELDS = (
    "sample_key", "source_smiles", "normalized_smiles",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ordered_key_hash(keys) -> str:
    digest = hashlib.sha256()
    for key in keys:
        raw = bytes(key)
        if len(raw) != 32:
            raise CacheLifecycleError("dual cohort keys must contain 32 bytes")
        digest.update(raw)
    return digest.hexdigest()


def load_active_dual_store(cache_root) -> dict:
    """Validate the one explicitly selected active bundle.

    The bundle is selected only by ``store.json``.  Every layer path must be
    inside ``builds/<bundle_hash>/<layer>`` and the bundle manifest must bind
    the exact source and per-layer identities recorded by the store.
    """

    cache_root = Path(cache_root).resolve()
    store_path = cache_root / "store.json"
    if not store_path.is_file():
        raise CacheLifecycleError(f"dual cache store is missing: {store_path}")
    store = validate_store(json.loads(store_path.read_text(encoding="utf-8")))
    if set(store.get("artifacts", {})) != {"ru_base", "topology", "trimer"}:
        raise CacheLifecycleError(
            "dual route requires exactly ru_base/topology/trimer bindings"
        )
    bundle_hash = str(store.get("bundle_hash", ""))
    if len(bundle_hash) != 64:
        raise CacheLifecycleError("store.json has no valid active bundle_hash")
    bundle_root = (cache_root / "builds" / bundle_hash).resolve()
    if cache_root not in bundle_root.parents or not bundle_root.is_dir():
        raise CacheLifecycleError("active bundle directory is missing or escapes cache root")
    for layer, binding in store["artifacts"].items():
        expected = (bundle_root / layer).resolve()
        observed = (cache_root / str(binding["path"])).resolve()
        if observed != expected:
            raise CacheLifecycleError(
                f"store binding {layer} does not belong to active bundle"
            )
    bundle_path = bundle_root / "bundle_manifest.json"
    if not bundle_path.is_file():
        raise CacheLifecycleError("active bundle has no bundle_manifest.json")
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    if bundle.get("bundle_hash") != bundle_hash:
        raise CacheLifecycleError("bundle manifest hash binding mismatch")
    if bundle.get("source_manifest_hash") != store.get("source", {}).get(
        "source_manifest_hash"
    ):
        raise CacheLifecycleError("bundle/source manifest binding mismatch")
    if set(bundle.get("artifacts", {})) != {"ru_base", "topology", "trimer"}:
        raise CacheLifecycleError("bundle manifest layer set is incomplete")
    for layer, binding in store["artifacts"].items():
        recorded = bundle["artifacts"][layer]
        if recorded != {
            "artifact_hash": binding["artifact_hash"],
            "manifest_hash": binding["manifest_hash"],
        }:
            raise CacheLifecycleError(f"bundle/store {layer} identity mismatch")
    return store


class DualFrozenBundle:
    """The topology and Trimer layers from one active published bundle."""

    def __init__(self, cache_root, *, expected_bundle_hash=None):
        self.cache_root = Path(cache_root).resolve()
        self.store = load_active_dual_store(self.cache_root)
        self.bundle_hash = str(self.store["bundle_hash"])
        if (
            expected_bundle_hash is not None
            and str(expected_bundle_hash) != self.bundle_hash
        ):
            raise CacheLifecycleError(
                "cohort is bound to a different active main bundle"
            )
        self.topology = self.trimer = None
        try:
            self.topology = ReadonlyArtifact(
                self.cache_root, self.store["artifacts"]["topology"]
            )
            self.trimer = ReadonlyArtifact(
                self.cache_root, self.store["artifacts"]["trimer"]
            )
        except BaseException:
            self.close()
            raise

    def close(self):
        topology, trimer = self.topology, self.trimer
        self.topology = self.trimer = None
        try:
            if topology is not None:
                topology.close()
        finally:
            if trimer is not None:
                trimer.close()


def load_dual_cohort(cohort_root, cache_root, *, verify_files=True) -> dict:
    """Load and validate an immutable ordered dual cohort manifest."""

    cohort_root = Path(cohort_root).resolve()
    manifest_path = cohort_root / "manifest.json"
    frozen_path = cohort_root / ".frozen"
    keys_path = cohort_root / "keys.npy"
    records_path = cohort_root / "records.jsonl"
    for path in (manifest_path, frozen_path, keys_path, records_path):
        if not path.is_file():
            raise CacheLifecycleError(f"dual cohort file is missing: {path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    if frozen != {"manifest_hash": json_hash(manifest)}:
        raise CacheLifecycleError("dual cohort .frozen/manifest mismatch")
    store = load_active_dual_store(cache_root)
    expected_bindings = {
        "main_bundle_hash": store["bundle_hash"],
        "source_manifest_hash": store["source"]["source_manifest_hash"],
        "topology_manifest_hash": store["artifacts"]["topology"]["manifest_hash"],
        "trimer_manifest_hash": store["artifacts"]["trimer"]["manifest_hash"],
    }
    for name, expected in expected_bindings.items():
        if manifest.get(name) != expected:
            raise CacheLifecycleError(f"dual cohort {name} binding mismatch")
    if manifest.get("ordering_policy") not in {
        "source_manifest_order_filtered_by_topology_and_trimer_acceptance",
        "task_order_then_original_row_preserving_fixed_property_rows",
    }:
        raise CacheLifecycleError("unsupported dual cohort ordering policy")
    if verify_files:
        if sha256_file(keys_path) != manifest.get("keys_file_sha256"):
            raise CacheLifecycleError("dual cohort keys file hash mismatch")
        if sha256_file(records_path) != manifest.get("records_file_sha256"):
            raise CacheLifecycleError("dual cohort records file hash mismatch")
    keys = np.load(keys_path, mmap_mode="r")
    if keys.dtype != np.uint8 or keys.ndim != 2 or keys.shape[1] != 32:
        raise CacheLifecycleError("dual cohort keys.npy must be uint8 [N,32]")
    count = int(keys.shape[0])
    if count != int(manifest.get("sample_count", -1)):
        raise CacheLifecycleError("dual cohort sample count mismatch")
    if ordered_key_hash(keys) != manifest.get("ordered_sample_key_hash"):
        raise CacheLifecycleError("dual cohort ordered key hash mismatch")
    records = []
    identity_policy = manifest.get("identity_policy")
    if identity_policy not in {"unique_structure", "dataset_rows_with_reuse"}:
        raise CacheLifecycleError("unsupported dual cohort identity policy")
    seen = set()
    with records_path.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if index >= count:
                raise CacheLifecycleError("dual cohort has extra records")
            row = json.loads(line)
            missing = [name for name in COHORT_REQUIRED_FIELDS if name not in row]
            if "source_row" not in row and "original_row" not in row:
                missing.append("source_row_or_original_row")
            if missing:
                raise CacheLifecycleError(
                    f"dual cohort record {index} is missing fields: {missing}"
                )
            key = bytes.fromhex(str(row["sample_key"]))
            if key != bytes(np.asarray(keys[index], dtype=np.uint8)):
                raise CacheLifecycleError("dual cohort record/key order mismatch")
            if key != sample_key_from_normalized(str(row["normalized_smiles"])):
                raise CacheLifecycleError("dual cohort normalized identity mismatch")
            if key in seen and identity_policy == "unique_structure":
                raise CacheLifecycleError("dual cohort contains duplicate identities")
            seen.add(key)
            records.append(row)
    if len(records) != count:
        raise CacheLifecycleError("dual cohort records are truncated")
    observed_duplicates = count - len(seen)
    if int(manifest.get("duplicate_count", -1)) != observed_duplicates:
        raise CacheLifecycleError("dual cohort duplicate count mismatch")
    return {
        "root": str(cohort_root), "manifest": manifest,
        "manifest_hash": json_hash(manifest),
        "keys_array": keys, "records": records,
    }


def _load_selected_jsonl_records(records_path, selected_indices, expected_count):
    """Decode only selected JSONL records; leave every other line opaque.

    The file is binary-read so skipped rows, including their serialized labels,
    are never UTF-8-decoded or JSON-decoded. The caller may still validate the
    immutable file digest independently without materializing record values.
    """
    selected_indices = tuple(selected_indices)
    if any(isinstance(value, bool) or not isinstance(value, int)
           for value in selected_indices):
        raise CacheLifecycleError("selected cohort indices must be integers")
    if tuple(sorted(set(selected_indices))) != selected_indices:
        raise CacheLifecycleError("selected cohort indices must be sorted and unique")
    if any(value < 0 or value >= int(expected_count) for value in selected_indices):
        raise CacheLifecycleError("selected cohort index is out of range")

    wanted = set(selected_indices)
    selected = {}
    observed = 0
    with Path(records_path).open("rb") as handle:
        for index, line in enumerate(handle):
            if index >= int(expected_count):
                raise CacheLifecycleError("dual cohort has extra records")
            observed = index + 1
            if index in wanted:
                try:
                    selected[index] = json.loads(line)
                except (UnicodeDecodeError, ValueError) as error:
                    raise CacheLifecycleError(
                        f"selected dual cohort record {index} is invalid JSON"
                    ) from error
    if observed != int(expected_count):
        raise CacheLifecycleError("dual cohort records are truncated")
    if len(selected) != len(selected_indices):
        raise CacheLifecycleError("selected dual cohort records are incomplete")
    return selected


def load_dual_cohort_task_rows(
    cohort_root,
    cache_root,
    *,
    task,
    row_indices,
    expected_task_rows,
    expected_split_sha256,
):
    """Load labels and records only for explicitly selected task row indices.

    This path is for the frozen downstream-row cohort. It checks the frozen
    manifests, bundle bindings, full-file digests, row/key identity and live
    task counts, but JSON-decodes only selected train/validation records. The
    other cohort rows remain opaque bytes and are used only for file integrity
    and line-count validation.
    """
    cohort_root = Path(cohort_root).resolve()
    cache_root = Path(cache_root).resolve()
    manifest_path = cohort_root / "manifest.json"
    frozen_path = cohort_root / ".frozen"
    keys_path = cohort_root / "keys.npy"
    records_path = cohort_root / "records.jsonl"
    for path in (manifest_path, frozen_path, keys_path, records_path):
        if not path.is_file():
            raise CacheLifecycleError(f"dual cohort file is missing: {path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    manifest_hash = json_hash(manifest)
    if frozen != {"manifest_hash": manifest_hash}:
        raise CacheLifecycleError("dual cohort .frozen/manifest mismatch")

    store = load_active_dual_store(cache_root)
    expected_bindings = {
        "main_bundle_hash": store["bundle_hash"],
        "source_manifest_hash": store["source"]["source_manifest_hash"],
        "topology_manifest_hash": store["artifacts"]["topology"]["manifest_hash"],
        "trimer_manifest_hash": store["artifacts"]["trimer"]["manifest_hash"],
    }
    for name, expected in expected_bindings.items():
        if manifest.get(name) != expected:
            raise CacheLifecycleError(f"dual cohort {name} binding mismatch")

    if manifest.get("identity_policy") != "dataset_rows_with_reuse":
        raise CacheLifecycleError("selected task rows require dataset_rows_with_reuse")
    if manifest.get("ordering_policy") != \
            "task_order_then_original_row_preserving_fixed_property_rows":
        raise CacheLifecycleError("selected task rows require the frozen task-row ordering")

    union_root = cohort_root.parent / "union_outer5_inner20"
    union_manifest_path = union_root / "manifest.json"
    if not union_manifest_path.is_file():
        raise CacheLifecycleError("downstream union manifest is missing")
    union_manifest = json.loads(union_manifest_path.read_text(encoding="utf-8"))
    if json_hash(union_manifest) != manifest.get("union_manifest_hash"):
        raise CacheLifecycleError("downstream cohort/union manifest binding mismatch")
    task_order = union_manifest.get("task_order")
    task_counts = manifest.get("task_counts")
    if not isinstance(task_order, list) or len(task_order) != len(set(task_order)):
        raise CacheLifecycleError("downstream union task order is invalid")
    if set(task_order) != set(task_counts or {}):
        raise CacheLifecycleError("downstream union/cohort task sets differ")
    if union_manifest.get("task_counts") != task_counts:
        raise CacheLifecycleError("downstream union/cohort task counts differ")
    sample_count = int(manifest.get("sample_count", -1))
    if sample_count != int(union_manifest.get("row_count", -2)) \
            or sum(int(task_counts[name]) for name in task_order) != sample_count:
        raise CacheLifecycleError("downstream task counts do not cover the frozen cohort")
    task = str(task)
    if task not in task_counts:
        raise CacheLifecycleError(f"downstream cohort has no task={task}")
    task_count = int(task_counts[task])
    if task_count != int(expected_task_rows):
        raise CacheLifecycleError(
            f"live cohort task rows {task_count} != split rows {int(expected_task_rows)}"
        )
    split_hash = (manifest.get("split_manifest_sha256") or {}).get(task)
    union_split_hash = (union_manifest.get("split_manifest_sha256") or {}).get(task)
    if split_hash != str(expected_split_sha256) or union_split_hash != split_hash:
        raise CacheLifecycleError(f"frozen cohort split identity differs for task={task}")

    selected_task_indices = tuple(row_indices)
    if any(isinstance(value, bool) or not isinstance(value, int)
           for value in selected_task_indices):
        raise CacheLifecycleError("selected task row indices must be integers")
    if tuple(sorted(set(selected_task_indices))) != selected_task_indices:
        raise CacheLifecycleError("selected task row indices must be sorted and unique")
    if any(value < 0 or value >= task_count for value in selected_task_indices):
        raise CacheLifecycleError("selected task row index is out of range")

    if sha256_file(keys_path) != manifest.get("keys_file_sha256"):
        raise CacheLifecycleError("dual cohort keys file hash mismatch")
    if sha256_file(records_path) != manifest.get("records_file_sha256"):
        raise CacheLifecycleError("dual cohort records file hash mismatch")
    keys = np.load(keys_path, mmap_mode="r")
    if keys.dtype != np.uint8 or keys.ndim != 2 or keys.shape[1] != 32:
        raise CacheLifecycleError("dual cohort keys.npy must be uint8 [N,32]")
    if keys.shape[0] != sample_count:
        raise CacheLifecycleError("dual cohort sample count mismatch")
    if ordered_key_hash(keys) != manifest.get("ordered_sample_key_hash"):
        raise CacheLifecycleError("dual cohort ordered key hash mismatch")
    if sample_count - len({bytes(np.asarray(row, dtype=np.uint8)) for row in keys}) \
            != int(manifest.get("duplicate_count", -1)):
        raise CacheLifecycleError("dual cohort duplicate count mismatch")

    task_offset = sum(int(task_counts[name]) for name in task_order[:task_order.index(task)])
    global_indices = tuple(task_offset + index for index in selected_task_indices)
    selected_records = _load_selected_jsonl_records(
        records_path, global_indices, expected_count=sample_count)
    rows = []
    for local_index, global_index in zip(selected_task_indices, global_indices):
        row = selected_records[global_index]
        required = ("task", "original_row", "label", "sample_key", "source_smiles",
                    "normalized_smiles")
        missing = [name for name in required if name not in row]
        if missing:
            raise CacheLifecycleError(
                f"selected downstream record {global_index} is missing fields: {missing}"
            )
        if row["task"] != task or row["original_row"] != local_index:
            raise CacheLifecycleError("selected downstream task/row identity mismatch")
        try:
            key = bytes.fromhex(str(row["sample_key"]))
            label = float(row["label"])
        except (TypeError, ValueError) as error:
            raise CacheLifecycleError("selected downstream key/label is invalid") from error
        if len(key) != 32 or key != bytes(np.asarray(keys[global_index], dtype=np.uint8)):
            raise CacheLifecycleError("selected downstream record/key order mismatch")
        if key != sample_key_from_normalized(str(row["normalized_smiles"])):
            raise CacheLifecycleError("selected downstream normalized identity mismatch")
        if isinstance(row["label"], bool) or not math.isfinite(label):
            raise CacheLifecycleError("selected downstream label is nonfinite or boolean")
        row["label"] = label
        rows.append(row)

    return {
        "root": str(cohort_root), "manifest": manifest,
        "manifest_hash": manifest_hash, "keys_array": keys, "records": rows,
        "selected_task_indices": list(selected_task_indices),
    }
