#!/usr/bin/env python3
"""Build and strictly validate the read-only MTS relation-geometry sidecar.

The builder consumes only frozen Topology/Trimer LMDB records and cohort
manifests.  It uses worker processes for reads, while the parent is the only
process that writes staging/final arrays.  No Dataset, model or cache writer
is opened by this command.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.audit_mips_trimer_cache import _specs  # noqa: E402
from src.dataset.lmdb_cache import LmdbLayerStore, load_cohort  # noqa: E402
from src.dataset.mips_trimer_contract import (  # noqa: E402
    CACHE_RELATION_GEOMETRY_SCHEMA,
    RELATION_GEOMETRY_BUILDER_VERSION,
)
from src.dataset.mts_relation_geometry import (  # noqa: E402
    CODE_TO_REASON,
    INVALID_REASON_CODES,
    array_dtypes,
    build_sample_record,
)


_WORKER_TOPOLOGY = None
_WORKER_TRIMER = None

ARRAY_NAMES = (
    "sample_keys",
    "sample_relation_offsets",
    "relation_row",
    "relation_source_canonical_id",
    "relation_target_canonical_id",
    "relation_signed_source_shift",
    "relation_geometry_valid",
    "relation_invalid_reason_code",
    "relation_num_shortest_paths",
    "relation_path_offsets",
    "relation_endpoint_distance",
    "path_intermediate_canonical_id",
    "path_intermediate_ru_shift",
    "path_source_trimer_atom_id",
    "path_intermediate_trimer_atom_id",
    "path_target_trimer_atom_id",
    "path_bond_type_left",
    "path_bond_type_right",
    "path_geometry_valid",
    "path_invalid_reason_code",
    "path_cos_angle",
)
_DTYPES = {
    "sample_keys": np.dtype("u1"),
    "sample_relation_offsets": np.dtype("<i8"),
    **array_dtypes(),
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_digest(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def _atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _cohort_read_only(cache_root: Path, name: str):
    pointer = cache_root / "cohorts" / str(name) / "current.json"
    if not pointer.is_file():
        raise RuntimeError(f"cohort pointer is missing: {pointer}")
    pointer_payload = json.loads(pointer.read_text(encoding="utf-8"))
    cohort_dir = pointer.parent / str(pointer_payload["cohort_hash"])
    manifest_path = cohort_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("integrity_schema") != "mips-trimer-scage-cohort-manifest-v2":
        raise RuntimeError(f"read-only sidecar builder refuses a cohort needing manifest upgrade: {name}")
    return load_cohort(cohort_dir, load_text=False, verify_integrity=True)


def _cache_identity(project_root: Path):
    specs = _specs(project_root)
    identity = {}
    for name in ("topology", "trimer"):
        root = Path(specs[name]["root"])
        required = [root / ".done", root / ".frozen", root / "metadata.json", root / "manifest.json"]
        if not all(path.is_file() for path in required):
            raise RuntimeError(f"frozen {name} cache is incomplete: {root}")
        identity[name] = {
            "root": str(root),
            "schema": specs[name]["meta"].get("schema"),
            "feature_config_hash": specs[name]["meta"].get("feature_config_hash"),
            "done_artifact_hash": (root / ".done").read_text(encoding="utf-8").strip(),
            "done_file_sha256": _sha256_file(root / ".done"),
            "frozen_file_sha256": _sha256_file(root / ".frozen"),
            "metadata_file_sha256": _sha256_file(root / "metadata.json"),
            "manifest_file_sha256": _sha256_file(root / "manifest.json"),
        }
    return identity


def _identity_binding(identity, cohort):
    manifest = cohort["manifest"]
    payload = {
        "schema": CACHE_RELATION_GEOMETRY_SCHEMA,
        "builder_version": RELATION_GEOMETRY_BUILDER_VERSION,
        "cohort": manifest["dataset_name"],
        "cohort_hash": manifest["cohort_hash"],
        "ordered_sample_key_hash": manifest["ordered_sample_key_hash"],
        "record_count": len(cohort["keys"]),
        "topology": identity["topology"],
        "trimer": identity["trimer"],
    }
    return _json_digest(payload), payload


def _init_worker(topology_root: str, trimer_root: str):
    global _WORKER_TOPOLOGY, _WORKER_TRIMER
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    _WORKER_TOPOLOGY = LmdbLayerStore(topology_root, require_done=True)
    _WORKER_TRIMER = LmdbLayerStore(trimer_root, require_done=True)


def _empty_record(key: bytes):
    arrays = {
        name: []
        for name in (
            "relation_row", "relation_source_canonical_id", "relation_target_canonical_id",
            "relation_signed_source_shift", "relation_geometry_valid", "relation_invalid_reason_code",
            "relation_num_shortest_paths", "relation_endpoint_distance",
            "path_intermediate_canonical_id", "path_intermediate_ru_shift",
            "path_source_trimer_atom_id", "path_intermediate_trimer_atom_id",
            "path_target_trimer_atom_id", "path_bond_type_left", "path_bond_type_right",
            "path_geometry_valid", "path_invalid_reason_code", "path_cos_angle",
        )
    }
    arrays["relation_path_offsets"] = [0]
    return {
        "sample_key": bytes(key),
        "arrays": arrays,
        "sample_reasons": ["record_read_error"],
        "graph_available": False,
        "geometry_3d": False,
    }


def _chunk_worker(keys):
    records = []
    for raw_key in keys:
        key = bytes(raw_key)
        try:
            records.append(build_sample_record(key, _WORKER_TOPOLOGY[key], _WORKER_TRIMER[key]))
        except Exception:
            records.append(_empty_record(key))
    return records


def _array_hashes(root: Path, shapes):
    files = {}
    for name in ARRAY_NAMES:
        path = root / f"{name}.npy"
        files[name] = {
            "path": path.name,
            "sha256": _sha256_file(path),
            "dtype": np.dtype(_DTYPES[name]).str,
            "shape": list(shapes[name]),
            "bytes": int(path.stat().st_size),
        }
    return files


def _raw_append(handle, value, dtype):
    array = np.asarray(value, dtype=dtype)
    if array.size:
        handle.write(array.tobytes(order="C"))


def _npy_from_raw(raw: Path, output: Path, dtype, shape):
    output_tmp = output.with_suffix(output.suffix + ".tmp")
    mmap = np.lib.format.open_memmap(output_tmp, mode="w+", dtype=dtype, shape=shape)
    total = int(np.prod(shape, dtype=np.int64))
    offset = 0
    itemsize = np.dtype(dtype).itemsize
    with raw.open("rb") as handle:
        while offset < total:
            count = min(total - offset, max(1, (8 << 20) // max(itemsize, 1)))
            values = np.fromfile(handle, dtype=dtype, count=count)
            if values.size != count:
                del mmap
                output_tmp.unlink(missing_ok=True)
                raise RuntimeError(f"staging array length mismatch: {raw}")
            mmap.reshape(-1)[offset : offset + count] = values
            offset += count
    mmap.flush()
    del mmap
    os.replace(output_tmp, output)


def _record_stats(record, stats):
    arrays = record["arrays"]
    relation_valid = np.asarray(arrays["relation_geometry_valid"], dtype=np.bool_)
    path_valid = np.asarray(arrays["path_geometry_valid"], dtype=np.bool_)
    stats["sample_count"] += 1
    stats["graph_available_samples"] += int(record["graph_available"])
    stats["geometry_3d_samples"] += int(record["geometry_3d"])
    stats["relation_count"] += int(relation_valid.size)
    stats["relation_complete_geometry_count"] += int(relation_valid.sum())
    stats["path_count"] += int(path_valid.size)
    stats["valid_path_count"] += int(path_valid.sum())
    for reason in record["sample_reasons"]:
        stats["sample_invalid_reasons"][str(reason)] += 1
    for code in np.asarray(arrays["relation_invalid_reason_code"], dtype=np.int16).tolist():
        if int(code):
            stats["relation_invalid_reasons"][CODE_TO_REASON.get(int(code), f"code_{int(code)}")] += 1
    for code in np.asarray(arrays["path_invalid_reason_code"], dtype=np.int16).tolist():
        if int(code):
            stats["path_invalid_reasons"][CODE_TO_REASON.get(int(code), f"code_{int(code)}")] += 1


def _new_stats():
    return {
        "sample_count": 0,
        "graph_available_samples": 0,
        "geometry_3d_samples": 0,
        "relation_count": 0,
        "relation_complete_geometry_count": 0,
        "path_count": 0,
        "valid_path_count": 0,
        "sample_invalid_reasons": collections.Counter(),
        "relation_invalid_reasons": collections.Counter(),
        "path_invalid_reasons": collections.Counter(),
    }


def _json_stats(stats):
    output = dict(stats)
    for name in ("sample_invalid_reasons", "relation_invalid_reasons", "path_invalid_reasons"):
        output[name] = dict(sorted(stats[name].items()))
    output["relation_geometry_complete_rate"] = (
        stats["relation_complete_geometry_count"] / stats["relation_count"]
        if stats["relation_count"] else None
    )
    output["path_geometry_valid_rate"] = (
        stats["valid_path_count"] / stats["path_count"] if stats["path_count"] else None
    )
    return output


def _strict_validate(root: Path, *, cohort=None, source_identity=None):
    """Validate one published sidecar without modifying it."""

    root = Path(root)
    metadata_path = root / "metadata.json"
    if not metadata_path.is_file() or not (root / ".done").is_file() or not (root / ".frozen").is_file():
        raise RuntimeError(f"sidecar is incomplete: {root}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("schema") != CACHE_RELATION_GEOMETRY_SCHEMA:
        raise RuntimeError("unsupported relation-geometry sidecar schema")
    if int(metadata.get("builder_version", -1)) != RELATION_GEOMETRY_BUILDER_VERSION:
        raise RuntimeError("relation-geometry builder version mismatch")
    if metadata.get("selection") not in {"full_cohort", "smoke_prefix"}:
        raise RuntimeError("unsupported sidecar selection")
    artifact = str(metadata.get("artifact_hash", ""))
    if artifact != (root / ".done").read_text(encoding="utf-8").strip():
        raise RuntimeError("sidecar .done artifact hash mismatch")
    frozen = json.loads((root / ".frozen").read_text(encoding="utf-8"))
    if frozen.get("artifact_hash") != artifact or frozen.get("schema") != CACHE_RELATION_GEOMETRY_SCHEMA:
        raise RuntimeError("sidecar .frozen binding mismatch")
    expected_artifact = _json_digest({key: value for key, value in metadata.items() if key != "artifact_hash"})
    if expected_artifact != artifact:
        raise RuntimeError("sidecar metadata artifact hash mismatch")
    if source_identity is not None:
        for name in ("topology", "trimer"):
            if metadata["source_identity"].get(name) != source_identity.get(name):
                raise RuntimeError(f"sidecar source identity mismatch: {name}")
    if cohort is not None:
        manifest = cohort["manifest"]
        if metadata.get("cohort_hash") != manifest.get("cohort_hash"):
            raise RuntimeError("sidecar cohort hash mismatch")
        if metadata.get("ordered_sample_key_hash") != manifest.get("ordered_sample_key_hash"):
            raise RuntimeError("sidecar ordered sample-key hash mismatch")
        if int(metadata.get("record_count", -1)) != len(cohort["keys"]):
            raise RuntimeError("sidecar record count mismatch")
    shapes = {name: tuple(item["shape"]) for name, item in metadata.get("arrays", {}).items()}
    if set(shapes) != set(ARRAY_NAMES):
        raise RuntimeError("sidecar array manifest is incomplete")
    for name in ARRAY_NAMES:
        path = root / f"{name}.npy"
        if not path.is_file():
            raise RuntimeError(f"sidecar array missing: {name}")
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        observed = metadata["arrays"][name]
        if np.dtype(array.dtype).str != str(observed["dtype"]) or list(array.shape) != list(observed["shape"]):
            raise RuntimeError(f"sidecar array dtype/shape mismatch: {name}")
        if _sha256_file(path) != observed.get("sha256"):
            raise RuntimeError(f"sidecar array hash mismatch: {name}")
        if array.dtype.kind == "f" and not bool(np.isfinite(array).all()):
            raise RuntimeError(f"sidecar non-finite array: {name}")
    keys = np.load(root / "sample_keys.npy", mmap_mode="r", allow_pickle=False)
    sample_offsets = np.load(root / "sample_relation_offsets.npy", mmap_mode="r", allow_pickle=False)
    relation_offsets = np.load(root / "relation_path_offsets.npy", mmap_mode="r", allow_pickle=False)
    relation_count = int(shapes["relation_row"][0])
    path_count = int(shapes["path_intermediate_canonical_id"][0])
    if keys.ndim != 2 or keys.shape[1] != 32 or sample_offsets.shape != (keys.shape[0] + 1,):
        raise RuntimeError("sidecar sample offset shape mismatch")
    if relation_offsets.shape != (relation_count + 1,):
        raise RuntimeError("sidecar relation offset shape mismatch")
    if int(sample_offsets[0]) != 0 or int(sample_offsets[-1]) != relation_count:
        raise RuntimeError("sidecar sample offset boundary mismatch")
    if int(relation_offsets[0]) != 0 or int(relation_offsets[-1]) != path_count:
        raise RuntimeError("sidecar relation offset boundary mismatch")
    if not bool(np.all(sample_offsets[1:] >= sample_offsets[:-1])) or not bool(np.all(relation_offsets[1:] >= relation_offsets[:-1])):
        raise RuntimeError("sidecar offsets are not monotonic")
    relation_num = np.load(root / "relation_num_shortest_paths.npy", mmap_mode="r", allow_pickle=False)
    if not bool(np.array_equal(relation_num.astype(np.int64), np.diff(relation_offsets))):
        raise RuntimeError("sidecar relation/path count mismatch")
    relation_valid = np.load(root / "relation_geometry_valid.npy", mmap_mode="r", allow_pickle=False)
    relation_codes = np.load(root / "relation_invalid_reason_code.npy", mmap_mode="r", allow_pickle=False)
    path_valid = np.load(root / "path_geometry_valid.npy", mmap_mode="r", allow_pickle=False)
    path_codes = np.load(root / "path_invalid_reason_code.npy", mmap_mode="r", allow_pickle=False)
    if bool(np.any(relation_valid & (relation_codes != 0))) or bool(np.any((~relation_valid) & (relation_codes == 0))):
        raise RuntimeError("relation validity/reason code mismatch")
    if bool(np.any(path_valid & (path_codes != 0))) or bool(np.any((~path_valid) & (path_codes == 0))):
        raise RuntimeError("path validity/reason code mismatch")
    if cohort is not None:
        expected_keys = np.asarray(cohort["keys"], dtype=np.uint8)
        if not bool(np.array_equal(keys, expected_keys)):
            raise RuntimeError("sidecar sample-key order mismatch")
    return metadata


class RelationGeometrySidecar:
    """Small read-only row reader used by tests and future Dataset work."""

    def __init__(self, root: Path, *, cohort=None, source_identity=None):
        self.root = Path(root)
        self.metadata = _strict_validate(self.root, cohort=cohort, source_identity=source_identity)
        self.arrays = {
            name: np.load(self.root / f"{name}.npy", mmap_mode="r", allow_pickle=False)
            for name in ARRAY_NAMES
        }

    def __len__(self):
        return int(self.arrays["sample_keys"].shape[0])

    def row(self, index: int):
        index = int(index)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        relation_start = int(self.arrays["sample_relation_offsets"][index])
        relation_end = int(self.arrays["sample_relation_offsets"][index + 1])
        path_start = int(self.arrays["relation_path_offsets"][relation_start])
        path_end = int(self.arrays["relation_path_offsets"][relation_end])
        return {
            "sample_key": np.asarray(self.arrays["sample_keys"][index]),
            "relations": {name: np.asarray(self.arrays[name][relation_start:relation_end]) for name in ARRAY_NAMES if name.startswith("relation_") and name != "relation_path_offsets"},
            "paths": {name: np.asarray(self.arrays[name][path_start:path_end]) for name in ARRAY_NAMES if name.startswith("path_")},
        }


def _build_one(
    *,
    project_root: Path,
    cache_root: Path,
    cohort_name: str,
    sidecar_root: Path,
    report_root: Path,
    workers: int,
    chunk_size: int,
    max_samples: int | None = None,
    output_selection: str = "full_cohort",
):
    cohort = _cohort_read_only(cache_root, cohort_name)
    identity = _cache_identity(project_root)
    source_hash, binding = _identity_binding(identity, cohort)
    keys = [bytes(row) for row in np.asarray(cohort["keys"], dtype=np.uint8)]
    if max_samples is not None:
        keys = keys[: int(max_samples)]
        binding = dict(binding)
        binding["selection"] = output_selection
    else:
        binding["selection"] = "full_cohort"
    # Selection is part of the content identity so a smoke prefix can never
    # occupy the full-cohort artifact path.
    source_hash = _json_digest(binding)
    final_root = sidecar_root / source_hash / cohort_name / str(cohort["manifest"]["cohort_hash"])
    if final_root.exists():
        if (final_root / ".done").is_file():
            _strict_validate(final_root, cohort=cohort if max_samples is None else None, source_identity=identity)
            return {"cohort": cohort_name, "root": str(final_root), "reused": True, "record_count": len(keys)}
        raise RuntimeError(f"same-identity sidecar directory is incomplete; refusing overwrite: {final_root}")
    final_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{cohort_name}.staging-", dir=str(final_root.parent)))
    handles = {}
    try:
        for name in ARRAY_NAMES:
            handles[name] = (staging / f"{name}.bin").open("wb")
        # Offset arrays start with a zero sentinel.
        _raw_append(handles["sample_relation_offsets"], [0], _DTYPES["sample_relation_offsets"])
        _raw_append(handles["relation_path_offsets"], [0], _DTYPES["relation_path_offsets"])
        stats = _new_stats()
        relation_total = 0
        path_total = 0
        chunks = [keys[start : start + int(chunk_size)] for start in range(0, len(keys), int(chunk_size))]
        started = time.monotonic()
        if int(workers) > 1 and chunks:
            with ProcessPoolExecutor(
                max_workers=int(workers),
                initializer=_init_worker,
                initargs=(identity["topology"]["root"], identity["trimer"]["root"]),
            ) as executor:
                iterator = executor.map(_chunk_worker, chunks)
                records_iter = enumerate(iterator, start=1)
                for completed, records in records_iter:
                    for record in records:
                        arrays = record["arrays"]
                        _raw_append(handles["sample_keys"], np.frombuffer(record["sample_key"], dtype=np.uint8), _DTYPES["sample_keys"])
                        for name in ARRAY_NAMES:
                            if name in {"sample_keys", "sample_relation_offsets", "relation_path_offsets"}:
                                continue
                            _raw_append(handles[name], arrays[name], _DTYPES[name])
                        relation_total += len(arrays["relation_row"])
                        prior_path_total = path_total
                        path_total += len(arrays["path_intermediate_canonical_id"])
                        _raw_append(handles["sample_relation_offsets"], [relation_total], _DTYPES["sample_relation_offsets"])
                        _raw_append(
                            handles["relation_path_offsets"],
                            np.asarray(arrays["relation_path_offsets"][1:], dtype=np.int64) + prior_path_total,
                            _DTYPES["relation_path_offsets"],
                        )
                        _record_stats(record, stats)
                    if completed == 1 or completed == len(chunks) or completed % max(1, len(chunks) // 20) == 0:
                        print(f"[sidecar] cohort={cohort_name} chunks={completed}/{len(chunks)} samples={stats['sample_count']} relations={relation_total} paths={path_total}", flush=True)
        else:
            _init_worker(identity["topology"]["root"], identity["trimer"]["root"])
            try:
                for completed, chunk in enumerate(chunks, start=1):
                    for record in _chunk_worker(chunk):
                        arrays = record["arrays"]
                        _raw_append(handles["sample_keys"], np.frombuffer(record["sample_key"], dtype=np.uint8), _DTYPES["sample_keys"])
                        for name in ARRAY_NAMES:
                            if name in {"sample_keys", "sample_relation_offsets", "relation_path_offsets"}:
                                continue
                            _raw_append(handles[name], arrays[name], _DTYPES[name])
                        relation_count = len(arrays["relation_row"])
                        path_count = len(arrays["path_intermediate_canonical_id"])
                        relation_total += relation_count
                        path_total += path_count
                        _raw_append(handles["sample_relation_offsets"], [relation_total], _DTYPES["sample_relation_offsets"])
                        _raw_append(handles["relation_path_offsets"], np.asarray(arrays["relation_path_offsets"][1:], dtype=np.int64) + (path_total - path_count), _DTYPES["relation_path_offsets"])
                        _record_stats(record, stats)
                    if completed == 1 or completed == len(chunks) or completed % max(1, len(chunks) // 20) == 0:
                        print(f"[sidecar] cohort={cohort_name} chunks={completed}/{len(chunks)} samples={stats['sample_count']} relations={relation_total} paths={path_total}", flush=True)
            finally:
                if _WORKER_TOPOLOGY is not None:
                    _WORKER_TOPOLOGY.close()
                if _WORKER_TRIMER is not None:
                    _WORKER_TRIMER.close()
        elapsed = max(time.monotonic() - started, 1e-9)
        # The multiprocessing branch needs the exact per-relation offsets.  A
        # compact offset log is built from each record while writing; the
        # branch above is replaced by the common helper below on the next run.
        for handle in handles.values():
            handle.flush()
            handle.close()
        handles = {}
        # Convert the raw streams to immutable .npy files.
        shapes = {
            "sample_keys": (len(keys), 32),
            "sample_relation_offsets": (len(keys) + 1,),
            "relation_path_offsets": (relation_total + 1,),
            "relation_row": (relation_total,),
            "relation_source_canonical_id": (relation_total,),
            "relation_target_canonical_id": (relation_total,),
            "relation_signed_source_shift": (relation_total,),
            "relation_geometry_valid": (relation_total,),
            "relation_invalid_reason_code": (relation_total,),
            "relation_num_shortest_paths": (relation_total,),
            "relation_endpoint_distance": (relation_total,),
            "path_intermediate_canonical_id": (path_total,),
            "path_intermediate_ru_shift": (path_total,),
            "path_source_trimer_atom_id": (path_total,),
            "path_intermediate_trimer_atom_id": (path_total,),
            "path_target_trimer_atom_id": (path_total,),
            "path_bond_type_left": (path_total,),
            "path_bond_type_right": (path_total,),
            "path_geometry_valid": (path_total,),
            "path_invalid_reason_code": (path_total,),
            "path_cos_angle": (path_total,),
        }
        for name in ARRAY_NAMES:
            _npy_from_raw(staging / f"{name}.bin", staging / f"{name}.npy", _DTYPES[name], shapes[name])
        # Publish only after every array has a complete header and payload.
        final_root.mkdir(parents=True, exist_ok=False)
        for name in ARRAY_NAMES:
            os.replace(staging / f"{name}.npy", final_root / f"{name}.npy")
        stats_json = _json_stats(stats)
        metadata = {
            "schema": CACHE_RELATION_GEOMETRY_SCHEMA,
            "builder_version": RELATION_GEOMETRY_BUILDER_VERSION,
            "cohort": cohort_name,
            "cohort_hash": cohort["manifest"]["cohort_hash"],
            "ordered_sample_key_hash": cohort["manifest"]["ordered_sample_key_hash"],
            "record_count": len(keys),
            "selection": output_selection,
            "source_identity": identity,
            "source_binding_hash": source_hash,
            "invalid_reason_codes": dict(INVALID_REASON_CODES),
            "stats": stats_json,
            "arrays": _array_hashes(final_root, shapes),
            "runtime": {"elapsed_seconds": elapsed, "samples_per_second": len(keys) / elapsed, "workers": int(workers), "chunk_size": int(chunk_size), "gpu_used": False},
        }
        metadata["artifact_hash"] = _json_digest({key: value for key, value in metadata.items() if key != "artifact_hash"})
        _atomic_json(final_root / "metadata.json", metadata)
        _atomic_json(final_root / "manifest.json", {"schema": CACHE_RELATION_GEOMETRY_SCHEMA, "artifact_hash": metadata["artifact_hash"], "arrays": metadata["arrays"], "record_count": len(keys), "cohort_hash": metadata["cohort_hash"]})
        (final_root / ".done.tmp").write_text(metadata["artifact_hash"] + "\n", encoding="utf-8")
        os.replace(final_root / ".done.tmp", final_root / ".done")
        _atomic_json(final_root / ".frozen", {"schema": CACHE_RELATION_GEOMETRY_SCHEMA, "artifact_hash": metadata["artifact_hash"], "source_binding_hash": source_hash, "topology_done_artifact_id": identity["topology"]["done_artifact_hash"], "trimer_done_artifact_id": identity["trimer"]["done_artifact_hash"]})
        # Reopen strict read-only before declaring the build successful.
        _strict_validate(final_root, cohort=cohort if max_samples is None else None, source_identity=identity)
        return {"cohort": cohort_name, "root": str(final_root), "reused": False, "record_count": len(keys), "relation_count": relation_total, "path_count": path_total, "stats": stats_json, "elapsed_seconds": elapsed, "artifact_hash": metadata["artifact_hash"]}
    finally:
        for handle in handles.values():
            try:
                handle.close()
            except Exception:
                pass
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)


def _load_cohorts(cache_root, names):
    return [(name, _cohort_read_only(cache_root, name)) for name in names]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohorts", default="PI1M_v2 downstream_union")
    parser.add_argument("--cache-root", default="data/processed/mips_trimer_scage")
    parser.add_argument("--sidecar-root", default="data/processed/mips_trimer_scage/relation_geometry")
    parser.add_argument("--report-root", default="results/mts_multiscale_topology/g_prep_relation_geometry_v1")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--max-samples", type=int, default=0, help="test-only prefix; never used for a full artifact")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    names = [item for item in str(args.cohorts).replace(",", " ").split() if item]
    if set(names) != {"PI1M_v2", "downstream_union"}:
        raise SystemExit("--cohorts must contain exactly PI1M_v2 and downstream_union")
    if int(args.workers) < 0 or int(args.chunk_size) < 1:
        raise SystemExit("workers must be nonnegative and chunk-size must be positive")
    cache_root = (PROJECT_ROOT / args.cache_root).resolve()
    sidecar_root = (PROJECT_ROOT / args.sidecar_root).resolve()
    report_root = (PROJECT_ROOT / args.report_root).resolve()
    report_root.mkdir(parents=True, exist_ok=True)
    identity_before = _cache_identity(PROJECT_ROOT)
    outputs = []
    for name in ("PI1M_v2", "downstream_union"):
        output = _build_one(
            project_root=PROJECT_ROOT,
            cache_root=cache_root,
            cohort_name=name,
            sidecar_root=sidecar_root,
            report_root=report_root,
            workers=int(args.workers),
            chunk_size=int(args.chunk_size),
            max_samples=(int(args.max_samples) if args.smoke and int(args.max_samples) > 0 else None),
            output_selection=("smoke_prefix" if args.smoke else "full_cohort"),
        )
        outputs.append(output)
    identity_after = _cache_identity(PROJECT_ROOT)
    if identity_before != identity_after:
        raise RuntimeError("source cache identity changed during sidecar build")
    _atomic_json(report_root / "source_integrity_before.json", identity_before)
    _atomic_json(report_root / "source_integrity_after.json", identity_after)
    validation = []
    for output in outputs:
        cohort = _cohort_read_only(cache_root, output["cohort"])
        metadata = _strict_validate(
            Path(output["root"]), cohort=cohort, source_identity=identity_after
        )
        validation.append(
            {
                "cohort": output["cohort"],
                "root": output["root"],
                "strict_validate": True,
                "record_count": int(metadata["record_count"]),
                "relation_count": int(metadata["stats"]["relation_count"]),
                "path_count": int(metadata["stats"]["path_count"]),
                "relation_geometry_complete_rate": metadata["stats"][
                    "relation_geometry_complete_rate"
                ],
                "path_geometry_valid_rate": metadata["stats"][
                    "path_geometry_valid_rate"
                ],
                "artifact_hash": metadata["artifact_hash"],
            }
        )
    _atomic_json(
        report_root / "validation_summary.json",
        {
            "schema": CACHE_RELATION_GEOMETRY_SCHEMA,
            "builder_version": RELATION_GEOMETRY_BUILDER_VERSION,
            "strict_validate_all": True,
            "source_identity_unchanged": True,
            "cohorts": validation,
        },
    )
    _atomic_json(report_root / "build_summary.json", {"schema": CACHE_RELATION_GEOMETRY_SCHEMA, "builder_version": RELATION_GEOMETRY_BUILDER_VERSION, "outputs": outputs, "source_identity_unchanged": True, "selection": "smoke_prefix" if args.smoke else "full_cohort"})
    print(json.dumps({"outputs": outputs, "source_identity_unchanged": True}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
