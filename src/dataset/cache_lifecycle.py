"""Minimal offline lifecycle for the rebuilt MTS cache.

The only supported flow is::

    source -> staging build -> audit -> freeze -> atomic publish -> readonly

There is intentionally no migration, repair, legacy discovery, or online
materialization in this module.
"""

from __future__ import annotations

import csv
import fcntl
import hashlib
import io
import json
import os
import pickle
import signal
import time
from collections import Counter
from pathlib import Path

import lmdb
import numpy as np
import torch
from rdkit import Chem
from torch.utils.data import Dataset
from torch_geometric.data import Data

from .cache_spec import (
    RECORD_FIELDS,
    ROUTE_BUILD_SPECS,
    build_spec_hash,
    canonical_json,
)
from .lmdb_cache import normalize_polymer_smiles, sample_key_from_normalized
from .trimer_mcl import (
    TrimerContractError,
    TrimerGeometryRejection,
    attach_finite_trimer_mcl,
)


class CacheLifecycleError(RuntimeError):
    pass


class CacheAuditError(CacheLifecycleError):
    pass


class IntentionalBuildInterrupt(KeyboardInterrupt):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_hash(value) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def ordered_key_hash(keys) -> str:
    digest = hashlib.sha256()
    for key in keys:
        raw = bytes(key)
        if len(raw) != 32:
            raise CacheLifecycleError("sample keys must contain 32 bytes")
        digest.update(raw)
    return digest.hexdigest()


def atomic_json(path: Path, value) -> None:
    path = Path(path)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def serialize_record(key: bytes, data: Data) -> bytes:
    buffer = io.BytesIO()
    torch.save(
        {"sample_key": bytes(key), "data": data}, buffer,
        pickle_protocol=pickle.HIGHEST_PROTOCOL,
    )
    return buffer.getvalue()


def deserialize_record(payload: bytes, key: bytes) -> Data:
    record = torch.load(
        io.BytesIO(payload), map_location="cpu", weights_only=False
    )
    if set(record) != {"sample_key", "data"}:
        raise CacheAuditError("cache record envelope contains unsupported fields")
    if bytes(record["sample_key"]) != bytes(key):
        raise CacheAuditError("cache record sample_key mismatch")
    data = record["data"]
    if not isinstance(data, Data):
        raise CacheAuditError("cache record payload is not PyG Data")
    return data


def select_record_fields(layer: str, data: Data, *, optional_fields=()) -> Data:
    output = Data()
    for name in (*RECORD_FIELDS[layer], *tuple(optional_fields)):
        if hasattr(data, name):
            output[name] = getattr(data, name)
    if layer == "topology":
        output.num_nodes = int(torch.as_tensor(output.mips_x).size(0))
    return output


def load_source_rows(source_csv: Path, limit: int) -> tuple[list[dict], dict]:
    """Materialize a fixed pilot cohort of unique canonical identities.

    Invalid raw candidates are outside the pilot source manifest and are
    counted explicitly.  Once a row enters this manifest, any identity drift
    or corruption is a hard stop in every later phase.
    """

    source_csv = Path(source_csv).resolve()
    records = []
    seen = set()
    invalid_candidates = []
    duplicate_candidates = 0
    with source_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if not header:
            raise CacheLifecycleError("source CSV is empty")
        for source_row, row in enumerate(reader):
            if not row:
                continue
            source_smiles = str(row[0]).strip()
            normalized, valid = normalize_polymer_smiles(source_smiles)
            if not valid:
                invalid_candidates.append(int(source_row))
                continue
            key = sample_key_from_normalized(normalized)
            if key in seen:
                duplicate_candidates += 1
                continue
            seen.add(key)
            records.append({
                "sample_key": key.hex(),
                "source_smiles": source_smiles,
                "normalized_smiles": normalized,
                "source_row": int(source_row),
            })
            if int(limit) and len(records) >= int(limit):
                break
    full_source = int(limit) == 0
    if not full_source and len(records) != int(limit):
        raise CacheLifecycleError(
            f"source provided {len(records)} unique rows, expected {limit}"
        )
    records_digest = hashlib.sha256(
        "".join(canonical_json(row) + "\n" for row in records).encode("utf-8")
    ).hexdigest()
    manifest_body = {
        "source_csv": str(source_csv),
        "source_csv_sha256": sha256_file(source_csv),
        "selection": {
            "policy": (
                "full_unique_valid_canonical_production" if full_source
                else "first_unique_valid_canonical_pilot"
            ),
            "limit": int(limit),
            "raw_candidates_examined": (
                int(records[-1]["source_row"]) + 1 if records else 0
            ),
            "invalid_candidates_excluded": len(invalid_candidates),
            "invalid_candidate_rows": invalid_candidates,
            "duplicate_candidates_excluded": int(duplicate_candidates),
        },
        "source_count": len(records),
        "ordered_source_key_hash": ordered_key_hash(
            bytes.fromhex(row["sample_key"]) for row in records
        ),
        "records_digest": records_digest,
    }
    manifest = {**manifest_body, "source_manifest_hash": json_hash(manifest_body)}
    return records, manifest


def artifact_identity(layer: str, source_manifest_hash: str, parents: dict,
                      *, build_spec=None) -> str:
    spec = build_spec or ROUTE_BUILD_SPECS[layer]
    return json_hash({
        "artifact_type": str(layer),
        "build_spec_hash": build_spec_hash(spec),
        "source_manifest_hash": str(source_manifest_hash),
        "parents": dict(sorted(parents.items())),
    })


def bundle_identity(source_manifest_hash: str, artifact_hashes: dict) -> str:
    return json_hash({
        "source_manifest_hash": str(source_manifest_hash),
        "artifacts": dict(sorted(artifact_hashes.items())),
    })


class StagingWriter:
    """One resumable LMDB writer confined to a bundle ``.staging`` root."""

    def __init__(self, root: Path, metadata: dict, map_size=1024 ** 3):
        self.root = Path(root)
        if not any(part.endswith(".staging") for part in self.root.parts):
            raise CacheLifecycleError("writer target is not inside .staging")
        self.root.mkdir(parents=True, exist_ok=True)
        if (self.root / ".frozen").exists():
            raise CacheLifecycleError("cannot open a frozen staging artifact")
        self.metadata_path = self.root / "metadata.json"
        if self.metadata_path.exists():
            observed = json.loads(self.metadata_path.read_text(encoding="utf-8"))
            if observed != metadata:
                raise CacheLifecycleError("staging metadata mismatch on resume")
        else:
            atomic_json(self.metadata_path, metadata)
        self._lock = (self.root / ".writer.lock").open("a+")
        try:
            fcntl.flock(self._lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            self._lock.close()
            raise CacheLifecycleError("staging writer is already active") from exc
        data_root = self.root / "data.lmdb"
        data_root.mkdir(exist_ok=True)
        data_file = data_root / "data.mdb"
        initial = max(int(map_size), int(data_file.stat().st_size * 2) if data_file.exists() else 0)
        self.environment = lmdb.open(
            str(data_root), subdir=True, map_size=max(initial, 64 * 1024 ** 2),
            readonly=False, lock=True, readahead=False, meminit=False,
            map_async=False, sync=True, metasync=True, max_readers=512,
        )

    def __contains__(self, key: bytes) -> bool:
        with self.environment.begin(write=False) as txn:
            return txn.get(bytes(key)) is not None

    def get(self, key: bytes) -> Data:
        with self.environment.begin(write=False) as txn:
            payload = txn.get(bytes(key))
        if payload is None:
            raise KeyError(bytes(key).hex())
        return deserialize_record(bytes(payload), bytes(key))

    def put(self, key: bytes, data: Data) -> tuple[bool, int, float]:
        payload = serialize_record(bytes(key), data)
        started = time.monotonic()
        while True:
            try:
                with self.environment.begin(write=True) as txn:
                    inserted = txn.put(bytes(key), payload, overwrite=False)
                break
            except lmdb.MapFullError:
                current = int(self.environment.info()["map_size"])
                self.environment.set_mapsize(current * 2)
        return bool(inserted), len(payload), time.monotonic() - started

    def count(self) -> int:
        return int(self.environment.stat()["entries"])

    def sync(self) -> None:
        self.environment.sync(True)

    def close(self) -> None:
        if getattr(self, "environment", None) is not None:
            self.environment.sync(True)
            self.environment.close()
            self.environment = None
        if getattr(self, "_lock", None) is not None:
            fcntl.flock(self._lock.fileno(), fcntl.LOCK_UN)
            self._lock.close()
            self._lock = None


class ReadonlyArtifact:
    """Strict published-artifact reader; it has no write-capable operation."""

    def __init__(self, cache_root: Path, binding: dict):
        self.cache_root = Path(cache_root).resolve()
        self.binding = dict(binding)
        self.root = (self.cache_root / binding["path"]).resolve()
        if self.cache_root not in self.root.parents:
            raise CacheLifecycleError("artifact path escapes cache root")
        if any(part.endswith(".staging") for part in self.root.parts):
            raise CacheLifecycleError("training reader refuses .staging")
        metadata_path = self.root / "metadata.json"
        manifest_path = self.root / "manifest.json"
        frozen_path = self.root / ".frozen"
        if not all(path.is_file() for path in (metadata_path, manifest_path, frozen_path)):
            raise CacheLifecycleError(f"artifact is not frozen: {self.root}")
        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
        manifest_hash = json_hash(self.manifest)
        if set(frozen) != {"manifest_hash"} or frozen["manifest_hash"] != manifest_hash:
            raise CacheLifecycleError(".frozen does not bind the final manifest")
        if binding.get("manifest_hash") != manifest_hash:
            raise CacheLifecycleError("store binding manifest hash mismatch")
        if self.metadata.get("artifact_hash") != binding.get("artifact_hash"):
            raise CacheLifecycleError("store binding artifact hash mismatch")
        if self.metadata.get("build_spec") != binding.get("build_spec"):
            raise CacheLifecycleError("store binding build_spec mismatch")
        if self.metadata.get("build_spec_hash") != build_spec_hash(
            self.metadata["build_spec"]
        ):
            raise CacheLifecycleError("artifact build_spec hash mismatch")
        for name in (
            "artifact_hash", "build_spec_hash", "source_manifest_hash",
            "parents", "record_count",
        ):
            if self.manifest.get(name) != binding.get(name):
                raise CacheLifecycleError(f"store/manifest {name} mismatch")
        self._environment = lmdb.open(
            str(self.root / "data.lmdb"), subdir=True, readonly=True, lock=False,
            readahead=False, meminit=False, max_readers=512,
        )
        self._transaction = self._environment.begin(buffers=False)

    def __contains__(self, key) -> bool:
        return self._transaction.get(bytes(key)) is not None

    def __getitem__(self, key) -> Data:
        key = bytes(key)
        payload = self._transaction.get(key)
        if payload is None:
            raise KeyError(key.hex())
        return deserialize_record(bytes(payload), key)

    def close(self):
        if self._transaction is not None:
            self._transaction.abort()
            self._transaction = None
        if self._environment is not None:
            self._environment.close()
            self._environment = None


class PublishedCacheDataset(Dataset):
    """Geometry-accepted readonly lookup used by pilot and future training."""

    def __init__(self, cache_root):
        self.cache_root = Path(cache_root).resolve()
        store_path = self.cache_root / "store.json"
        store = json.loads(store_path.read_text(encoding="utf-8"))
        if set(store.get("artifacts", {})) != {"ru_base", "topology", "trimer"}:
            raise CacheLifecycleError("active store is incomplete")
        self.store = store
        self.topology = ReadonlyArtifact(self.cache_root, store["artifacts"]["topology"])
        self.trimer = ReadonlyArtifact(self.cache_root, store["artifacts"]["trimer"])
        accepted_path = self.trimer.root / "accepted_keys.npy"
        self.keys = np.load(accepted_path, mmap_mode="r")
        if self.keys.dtype != np.uint8 or self.keys.ndim != 2 or self.keys.shape[1] != 32:
            raise CacheLifecycleError("accepted key array is invalid")

    def __len__(self):
        return int(self.keys.shape[0])

    def __getitem__(self, index):
        key = bytes(np.asarray(self.keys[int(index)], dtype=np.uint8))
        topology = self.topology[key]
        trimer = self.trimer[key]
        output = Data()
        for source in (topology, trimer):
            for name in source.keys():
                if name in output:
                    raise CacheLifecycleError(f"duplicate published field: {name}")
                output[name] = source[name]
        output.num_nodes = int(torch.as_tensor(output.mips_x).size(0))
        return output

    def close(self):
        self.topology.close()
        self.trimer.close()


def sidecar_binding(parent_binding: dict, ordered_accepted_key_hash: str,
                    sidecar_build_spec: dict) -> dict:
    return {
        "parent_artifact_hash": str(parent_binding["artifact_hash"]),
        "ordered_accepted_key_hash": str(ordered_accepted_key_hash),
        "sidecar_build_spec": sidecar_build_spec,
        "sidecar_build_spec_hash": json_hash(sidecar_build_spec),
    }


def validate_sidecar_binding(metadata: dict, parent_binding: dict,
                             ordered_accepted_key_hash: str,
                             sidecar_build_spec: dict) -> None:
    expected = sidecar_binding(
        parent_binding, ordered_accepted_key_hash, sidecar_build_spec
    )
    if metadata != expected:
        raise CacheLifecycleError("sidecar parent/cohort/build-spec mismatch")


def read_rejections(path: Path) -> dict[bytes, dict]:
    output = {}
    if not Path(path).is_file():
        return output
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if set(row) - {
                "sample_key", "failure_code", "candidate_attempts",
                "elapsed_seconds", "round_reached", "source_row",
                "canonical_smiles", "layer", "status", "exception_type",
                "exception_message", "worker_pid", "traceback_hash",
            }:
                raise CacheLifecycleError("rejection ledger has unsupported fields")
            key = bytes.fromhex(row["sample_key"])
            if key in output:
                raise CacheLifecycleError(
                    f"duplicate rejection key at line {line_number}"
                )
            output[key] = row
    return output


def append_jsonl(path: Path, row: dict) -> None:
    payload = (canonical_json(row) + "\n").encode("utf-8")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        written = os.write(fd, payload)
        if written != len(payload):
            raise CacheLifecycleError("short JSONL append")
        os.fsync(fd)
    finally:
        os.close(fd)


def snapshot_tree(root: Path) -> dict:
    root = Path(root)
    return {
        str(path.relative_to(root)): (path.stat().st_size, path.stat().st_mtime_ns)
        for path in sorted(root.rglob("*")) if path.is_file()
    }


def failure_distribution(rejections: dict) -> dict:
    return dict(sorted(Counter(
        str(row["failure_code"]) for row in rejections.values()
    ).items()))
