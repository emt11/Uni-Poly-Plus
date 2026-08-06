"""Content-addressed LMDB cache for the graph-only MTS route.

The public MIPS implementation uses one LMDB lookup per base graph and mmap'ed
NumPy matrices for fixed-width descriptors.  This module keeps those useful
properties while adding versioned metadata, atomic completion markers,
resumable single-writer construction, and cohort manifests.
"""

from __future__ import annotations

import copy
import fcntl
import hashlib
import io
import json
import os
import pickle
import shutil
import threading
import time
from collections import OrderedDict
from pathlib import Path

import lmdb
import numpy as np
import pandas as pd
import torch
from rdkit import Chem, rdBase
from torch_geometric.data import Data
from .mips_trimer_contract import (
    CACHE_LAYOUT_SCHEMA,
    TRIMER_LMDB_SCHEMA as CONTRACT_TRIMER_LMDB_SCHEMA,
    TOPOLOGY_LMDB_SCHEMA as CONTRACT_TOPOLOGY_LMDB_SCHEMA,
)


RU_BASE_SCHEMA = "mips-trimer-scage-ru-base-v2"
TOPOLOGY_LMDB_SCHEMA = CONTRACT_TOPOLOGY_LMDB_SCHEMA
TRIMER_LMDB_SCHEMA = CONTRACT_TRIMER_LMDB_SCHEMA
MD200_LMDB_SCHEMA = "mips-trimer-scage-md200-lmdb-v1"
MD200_ARRAY_SCHEMA = "mips-trimer-scage-md200-array-v1"
MCL_THRESHOLDS_ARRAY_SCHEMA = "mips-trimer-scage-mcl-thresholds-array-v1"
COHORT_SCHEMA = "mips-trimer-scage-cohort-v1"
COHORT_INTEGRITY_SCHEMA = "mips-trimer-scage-cohort-manifest-v2"

_INITIAL_MAP_SIZE = 64 * 1024 ** 3
_MAX_MAP_SIZE = 1024 * 1024 ** 3

# python-lmdb rejects opening the same environment twice in one process when
# the handles are created with slightly different flags.  A training process
# can legitimately hold one read-only layer per task/fold, all pointing at
# the same content-addressed topology/trimer root.  Share read handles by
# absolute path and keep a small reference count instead of reopening the
# environment for every Dataset instance.  The registry is process-local;
# forked DataLoader workers get their own handles through ``__getstate__``.
_READ_ENV_REGISTRY = {}
_READ_ENV_REGISTRY_LOCK = threading.RLock()


def _json_digest(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _sha256_file(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_polymer_smiles(smiles: str) -> tuple[str, bool]:
    """Return a deterministic cache identity without dropping invalid rows."""

    source = str(smiles).strip()
    try:
        molecule = Chem.MolFromSmiles(source)
    except Exception:
        return f"INVALID::{source}", False
    if molecule is None:
        return f"INVALID::{source}", False
    return Chem.MolToSmiles(molecule, canonical=True), True


def sample_key_from_normalized(normalized_smiles: str) -> bytes:
    return hashlib.sha256(str(normalized_smiles).encode("utf-8")).digest()


def sample_key_from_smiles(smiles: str) -> bytes:
    normalized, _ = normalize_polymer_smiles(smiles)
    return sample_key_from_normalized(normalized)


def coerce_sample_key(value) -> bytes:
    if isinstance(value, np.ndarray):
        array = np.asarray(value, dtype=np.uint8).reshape(-1)
        if array.size == 32:
            return array.tobytes()
    if isinstance(value, memoryview):
        value = value.tobytes()
    if isinstance(value, bytearray):
        value = bytes(value)
    if isinstance(value, bytes):
        if len(value) != 32:
            raise ValueError("MIPS cache sample keys must contain 32 bytes")
        return value
    text = str(value)
    if len(text) == 64:
        try:
            return bytes.fromhex(text)
        except ValueError:
            pass
    return sample_key_from_smiles(text)


def _atomic_json(path, value) -> None:
    path = str(path)
    temporary = f"{path}.tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(value, handle, sort_keys=True, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


def _serialize_record(schema: str, key: bytes, data: Data) -> bytes:
    buffer = io.BytesIO()
    torch.save(
        {
            "layout_schema": CACHE_LAYOUT_SCHEMA,
            "content_schema": str(schema),
            "sample_key": bytes(key),
            "data": data,
        },
        buffer,
        pickle_protocol=pickle.HIGHEST_PROTOCOL,
    )
    return buffer.getvalue()


def _deserialize_record(payload: bytes, schema: str, key: bytes) -> Data:
    record = torch.load(
        io.BytesIO(payload), map_location="cpu", weights_only=False
    )
    if record.get("layout_schema") != CACHE_LAYOUT_SCHEMA:
        raise RuntimeError("LMDB feature record has an incompatible layout schema")
    if record.get("content_schema") != str(schema):
        raise RuntimeError("LMDB feature record has an incompatible content schema")
    if bytes(record.get("sample_key", b"")) != bytes(key):
        raise RuntimeError("LMDB feature record key mismatch")
    data = record.get("data")
    if not isinstance(data, Data):
        raise RuntimeError("LMDB feature record does not contain a PyG Data object")
    return data


class LmdbLayerStore:
    """Read-only, one-record-at-a-time immutable layer store."""

    def __init__(self, root, expected_meta=None, require_done=True):
        self.root = str(root)
        # Initialize handles before any validation can raise so destruction of
        # a partially constructed reader is always safe.
        self._environment = None
        self._transaction = None
        self._pid = None
        self._registry_key = None
        self.data_path = os.path.join(self.root, "data.lmdb")
        self.metadata_path = os.path.join(self.root, "metadata.json")
        self.manifest_path = os.path.join(self.root, "manifest.json")
        self.done_path = os.path.join(self.root, ".done")
        if not os.path.isfile(self.metadata_path):
            raise RuntimeError(f"LMDB layer metadata is missing: {self.root}")
        with open(self.metadata_path, encoding="utf-8") as handle:
            self.meta = json.load(handle)
        if self.meta.get("cache_layout_schema") != CACHE_LAYOUT_SCHEMA:
            raise RuntimeError(f"incompatible LMDB cache layout: {self.root}")
        if expected_meta is not None and self.meta != expected_meta:
            raise RuntimeError(f"LMDB layer metadata mismatch: {self.root}")
        self.schema = str(self.meta["schema"])
        if self.schema in {
            "mips-trimer-scage-topology-lmdb-v1",
            "mips-trimer-scage-topology-lmdb-v2",
        } or self.meta.get("feature_content_schema") == (
            "mips-trimer-scage-feature-v4"
        ):
            raise RuntimeError(
                "legacy explicit MTS topology cache is rejected; rebuild "
                "with mts-canonical-periodic-topology-lmdb-v1"
            )
        if require_done:
            self._validate_done()

    def _validate_done(self):
        if not os.path.isfile(self.manifest_path) or not os.path.isfile(self.done_path):
            raise RuntimeError(f"incomplete LMDB feature layer: {self.root}")
        with open(self.manifest_path, encoding="utf-8") as handle:
            manifest = json.load(handle)
        with open(self.done_path, encoding="utf-8") as handle:
            observed = handle.read().strip()
        if observed != _json_digest(manifest):
            raise RuntimeError(f"LMDB completion hash mismatch: {self.root}")
        if manifest.get("metadata_hash") != _json_digest(self.meta):
            raise RuntimeError(f"LMDB metadata hash mismatch: {self.root}")
        self.manifest = manifest

    def _connect(self):
        pid = os.getpid()
        if self._environment is None or self._pid != pid:
            self.close()
            registry_key = (pid, os.path.realpath(self.data_path))
            with _READ_ENV_REGISTRY_LOCK:
                entry = _READ_ENV_REGISTRY.get(registry_key)
                if entry is None:
                    environment = lmdb.open(
                        self.data_path,
                        subdir=True,
                        readonly=True,
                        lock=False,
                        readahead=False,
                        meminit=False,
                        max_readers=2048,
                    )
                    entry = [environment, 0]
                    _READ_ENV_REGISTRY[registry_key] = entry
                entry[1] += 1
                self._environment = entry[0]
                self._registry_key = registry_key
            self._transaction = self._environment.begin(buffers=False)
            self._pid = pid
        return self._transaction

    def renew(self):
        if self._environment is not None:
            if self._transaction is not None:
                self._transaction.abort()
            self._transaction = self._environment.begin(buffers=False)

    def close(self):
        if self._transaction is not None:
            self._transaction.abort()
        if self._environment is not None:
            registry_key = self._registry_key
            with _READ_ENV_REGISTRY_LOCK:
                entry = _READ_ENV_REGISTRY.get(registry_key)
                if entry is not None and entry[0] is self._environment:
                    entry[1] -= 1
                    if entry[1] <= 0:
                        _READ_ENV_REGISTRY.pop(registry_key, None)
                        entry[0].close()
                else:
                    # This branch is only expected after a fork or an
                    # abnormal interpreter shutdown; do not leak a handle.
                    self._environment.close()
        self._transaction = None
        self._environment = None
        self._pid = None
        self._registry_key = None

    def __del__(self):
        self.close()

    def __getstate__(self):
        return {"root": self.root}

    def __setstate__(self, state):
        self.__init__(state["root"])

    def __len__(self):
        if hasattr(self, "manifest"):
            return int(self.manifest["count"])
        return int(self._connect().stat()["entries"])

    def __contains__(self, key):
        key = coerce_sample_key(key)
        # Cursor.set_key checks key presence without materialising the large
        # serialized PyG value.  A million membership checks must not grow the
        # Python allocator by the full topology cache size.
        return bool(self._connect().cursor().set_key(key))

    def missing_mask(self, keys, *, cohort_hash=None):
        """Return a compact uint8 mask without materialising missing keys."""

        # A completed layer records the cohort identities it was built for.
        # If the requested cohort is one of those identities and the record
        # counts agree, the manifest itself proves coverage; avoid touching
        # every large LMDB value just to rediscover that fact.
        if (
            cohort_hash is not None
            and hasattr(self, "manifest")
            and str(cohort_hash) in set(self.manifest.get("cohort_hashes", []))
            and int(self.manifest.get("count", -1)) == len(keys)
        ):
            return np.zeros(len(keys), dtype=np.uint8)
        keys = list(keys)
        mask = np.zeros(len(keys), dtype=np.uint8)
        transaction = self._connect()
        cursor = transaction.cursor()
        for index, key in enumerate(keys):
            if not cursor.set_key(coerce_sample_key(key)):
                mask[index] = 1
        return mask

    def get_raw(self, key):
        key = coerce_sample_key(key)
        value = self._connect().get(key)
        if value is None:
            raise KeyError(key.hex())
        return bytes(value)

    def __getitem__(self, key):
        key = coerce_sample_key(key)
        return _deserialize_record(self.get_raw(key), self.schema, key)


class LmdbLayerWriter:
    """Resumable single-writer with bounded transaction loss."""

    def __init__(
        self,
        root,
        meta,
        *,
        commit_size=128,
        commit_seconds=30.0,
        rebuild=False,
    ):
        self.root = str(root)
        self._writer_lock_handle = None
        self._writer_lock_path = os.path.join(self.root, ".writer.lock")
        self._lifecycle_lock_handle = None
        self._lifecycle_lock_path = os.path.join(
            os.path.dirname(os.path.dirname(self.root)), ".lifecycle.lock"
        )
        self.meta = copy.deepcopy(meta)
        self.meta["cache_layout_schema"] = CACHE_LAYOUT_SCHEMA
        self.schema = str(self.meta["schema"])
        if self.schema in {
            "mips-trimer-scage-topology-lmdb-v1",
            "mips-trimer-scage-topology-lmdb-v2",
        } or self.meta.get("feature_content_schema") == (
            "mips-trimer-scage-feature-v4"
        ):
            raise RuntimeError(
                "legacy explicit MTS topology cannot be written; use "
                "mts-canonical-periodic-topology-lmdb-v1"
            )
        self._acquire_lifecycle_lock()
        self._acquire_writer_lock()
        self.commit_size = max(1, int(commit_size))
        self.commit_seconds = max(1.0, float(commit_seconds))
        try:
            frozen_path = os.path.join(self.root, ".frozen")
            if os.path.isfile(frozen_path):
                raise RuntimeError(
                    f"LMDB cache is frozen and cannot be written: {self.root}"
                )
            if rebuild and os.path.isdir(self.root):
                # Keep the lock file itself outside the destructive operation.
                for entry in os.listdir(self.root):
                    if entry not in {".writer.lock", ".frozen"}:
                        path = os.path.join(self.root, entry)
                        if os.path.isdir(path):
                            shutil.rmtree(path)
                        else:
                            os.remove(path)
            os.makedirs(self.root, exist_ok=True)
            self.data_path = os.path.join(self.root, "data.lmdb")
            self.metadata_path = os.path.join(self.root, "metadata.json")
            self.manifest_path = os.path.join(self.root, "manifest.json")
            self.done_path = os.path.join(self.root, ".done")
            self.previous_cohort_hashes = []
            if os.path.isfile(self.manifest_path):
                try:
                    with open(self.manifest_path, encoding="utf-8") as handle:
                        previous_manifest = json.load(handle)
                    self.previous_cohort_hashes = list(
                        previous_manifest.get("cohort_hashes", [])
                    )
                except Exception:
                    self.previous_cohort_hashes = []
            if os.path.isfile(self.metadata_path):
                with open(self.metadata_path, encoding="utf-8") as handle:
                    observed = json.load(handle)
                if observed != self.meta:
                    raise RuntimeError(f"incomplete LMDB layer metadata mismatch: {self.root}")
            else:
                _atomic_json(self.metadata_path, self.meta)
            if os.path.isfile(self.done_path):
                os.remove(self.done_path)
            os.makedirs(self.data_path, exist_ok=True)
            existing_data = os.path.join(self.data_path, "data.mdb")
            existing_size = (
                os.path.getsize(existing_data)
                if os.path.isfile(existing_data) else 0
            )
            initial_map_size = min(
                _MAX_MAP_SIZE,
                max(_INITIAL_MAP_SIZE, existing_size * 2),
            )
            self.environment = lmdb.open(
                self.data_path,
                subdir=True,
                map_size=initial_map_size,
                readonly=False,
                lock=True,
                readahead=False,
                meminit=False,
                map_async=False,
                sync=True,
                metasync=True,
                max_readers=2048,
            )
            self.buffer = []
            self.buffer_keys = set()
            self.last_commit = time.monotonic()
            self.inserted = 0
        except BaseException:
            self._release_writer_lock()
            self._release_lifecycle_lock()
            raise

    def _acquire_lifecycle_lock(self):
        os.makedirs(os.path.dirname(self._lifecycle_lock_path), exist_ok=True)
        handle = open(self._lifecycle_lock_path, "a+", encoding="utf-8")
        try:
            # Writers share the lifecycle lock; the finalizer takes it
            # exclusively for validation/freeze.  This closes the
            # check-then-freeze race without serializing independent layers.
            fcntl.flock(handle.fileno(), fcntl.LOCK_SH)
        except (BlockingIOError, OSError) as exc:
            handle.close()
            raise RuntimeError(
                f"cache lifecycle is being finalized: {self.root}"
            ) from exc
        self._lifecycle_lock_handle = handle

    def _release_lifecycle_lock(self):
        handle = getattr(self, "_lifecycle_lock_handle", None)
        if handle is None:
            return
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass
        try:
            handle.close()
        except OSError:
            pass
        self._lifecycle_lock_handle = None

    def _acquire_writer_lock(self):
        os.makedirs(self.root, exist_ok=True)
        handle = open(self._writer_lock_path, "a+", encoding="utf-8")
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (BlockingIOError, OSError) as exc:
            handle.close()
            raise RuntimeError(
                f"LMDB writer lock is held by another process: {self.root}"
            ) from exc
        handle.seek(0)
        handle.truncate()
        handle.write(json.dumps({
            "pid": os.getpid(),
            "started_at": time.time(),
            "root": self.root,
        }, sort_keys=True) + "\n")
        handle.flush()
        self._writer_lock_handle = handle

    def _release_writer_lock(self):
        handle = getattr(self, "_writer_lock_handle", None)
        if handle is None:
            return
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass
        try:
            handle.close()
        except OSError:
            pass
        self._writer_lock_handle = None

    def __contains__(self, key):
        key = coerce_sample_key(key)
        if key in self.buffer_keys:
            return True
        with self.environment.begin(write=False) as transaction:
            return transaction.get(key) is not None

    def missing(self, keys):
        output = []
        with self.environment.begin(write=False) as transaction:
            for key in keys:
                key = coerce_sample_key(key)
                if key not in self.buffer_keys and transaction.get(key) is None:
                    output.append(key)
        return output

    def missing_mask(self, keys):
        """Return a compact uint8 mask without materialising a key set."""

        keys = list(keys)
        mask = np.zeros(len(keys), dtype=np.uint8)
        with self.environment.begin(write=False) as transaction:
            for index, key in enumerate(keys):
                key = coerce_sample_key(key)
                if key not in self.buffer_keys and transaction.get(key) is None:
                    mask[index] = 1
        return mask

    def get(self, key):
        """Read one already committed record while the single writer is open."""

        key = coerce_sample_key(key)
        for buffered_key, payload in reversed(self.buffer):
            if buffered_key == key:
                return _deserialize_record(payload, self.schema, key)
        with self.environment.begin(write=False) as transaction:
            value = transaction.get(key)
        if value is None:
            raise KeyError(key.hex())
        return _deserialize_record(bytes(value), self.schema, key)

    def add(self, key, data):
        key = coerce_sample_key(key)
        if key in self:
            return False
        self.buffer.append((key, _serialize_record(self.schema, key, data)))
        self.buffer_keys.add(key)
        if (
            len(self.buffer) >= self.commit_size
            or time.monotonic() - self.last_commit >= self.commit_seconds
        ):
            self.flush()
        return True

    def _write_buffer(self):
        while True:
            try:
                with self.environment.begin(write=True) as transaction:
                    for key, value in self.buffer:
                        transaction.put(key, value, overwrite=False)
                return
            except lmdb.MapFullError:
                current = int(self.environment.info()["map_size"])
                target = min(_MAX_MAP_SIZE, max(current * 2, current + _INITIAL_MAP_SIZE))
                if target <= current:
                    raise RuntimeError(
                        f"LMDB layer exceeded maximum map size: {self.root}"
                    )
                self.environment.set_mapsize(target)

    def flush(self):
        if not self.buffer:
            return
        self._write_buffer()
        self.inserted += len(self.buffer)
        self.buffer = []
        self.buffer_keys.clear()
        self.environment.sync(True)
        self.last_commit = time.monotonic()

    def finalize(self, *, cohort_hashes=None, failure_count=0):
        self.flush()
        count = int(self.environment.stat()["entries"])
        transaction_id = int(self.environment.info()["last_txnid"])
        self.environment.sync(True)
        manifest = {
            "cache_layout_schema": CACHE_LAYOUT_SCHEMA,
            "schema": self.schema,
            "metadata_hash": _json_digest(self.meta),
            "count": count,
            "failure_count": int(failure_count),
            "last_transaction_id": transaction_id,
            "cohort_hashes": sorted(set(
                self.previous_cohort_hashes + list(cohort_hashes or [])
            )),
            "completed_at": time.time(),
        }
        _atomic_json(self.manifest_path, manifest)
        temporary = f"{self.done_path}.tmp"
        with open(temporary, "w", encoding="utf-8") as handle:
            handle.write(_json_digest(manifest) + "\n")
        os.replace(temporary, self.done_path)
        self.environment.close()
        self.environment = None
        store = LmdbLayerStore(self.root, expected_meta=self.meta)
        self._release_writer_lock()
        self._release_lifecycle_lock()
        return store

    def close(self):
        if self.environment is not None:
            self.environment.close()
            self.environment = None
        self._release_writer_lock()
        self._release_lifecycle_lock()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def _cohort_integrity_fields(cohort_dir, manifest):
    """Return immutable file-integrity fields without changing cohort identity.

    The cohort hash intentionally remains the v1 key-order identity.  These
    fields are a separate manifest contract so adding them does not invalidate
    the already-built PI1M_v2 cache roots.
    """
    cohort_dir = Path(cohort_dir)
    arrays = {
        "chemistry_valid.npy": np.load(
            cohort_dir / "chemistry_valid.npy", mmap_mode="r"
        ),
        "sample_keys.npy": np.load(
            cohort_dir / "sample_keys.npy", mmap_mode="r"
        ),
        "source_row_keys.npy": np.load(
            cohort_dir / "source_row_keys.npy", mmap_mode="r"
        ),
    }
    expected = {
        "chemistry_valid.npy": {
            "sha256": _sha256_file(cohort_dir / "chemistry_valid.npy"),
            "shape": list(arrays["chemistry_valid.npy"].shape),
            "dtype": str(arrays["chemistry_valid.npy"].dtype),
        },
        "sample_keys.npy": {
            "sha256": _sha256_file(cohort_dir / "sample_keys.npy"),
            "shape": list(arrays["sample_keys.npy"].shape),
            "dtype": str(arrays["sample_keys.npy"].dtype),
        },
        "source_row_keys.npy": {
            "sha256": _sha256_file(cohort_dir / "source_row_keys.npy"),
            "shape": list(arrays["source_row_keys.npy"].shape),
            "dtype": str(arrays["source_row_keys.npy"].dtype),
        },
    }
    text_files = {}
    for name in ("smiles.txt", "normalized_smiles.txt"):
        path = cohort_dir / name
        text_files[name] = {
            "sha256": _sha256_file(path),
            "bytes": int(path.stat().st_size),
        }
    return {
        "integrity_schema": COHORT_INTEGRITY_SCHEMA,
        "source_csv_sha256": manifest.get("source_csv_sha256"),
        "text_files": text_files,
        "arrays": expected,
    }


def _upgrade_cohort_manifest_integrity(
    cohort_dir, manifest, *, verify=True
):
    """Atomically add manifest-v2 integrity fields while preserving hash/id."""
    if (
        not verify
        and manifest.get("integrity_schema") == COHORT_INTEGRITY_SCHEMA
        and manifest.get("text_files")
        and manifest.get("arrays")
    ):
        return manifest
    integrity = _cohort_integrity_fields(cohort_dir, manifest)
    stable = {
        "integrity_schema": manifest.get("integrity_schema"),
        "source_csv_sha256": manifest.get("source_csv_sha256"),
        "text_files": manifest.get("text_files"),
        "arrays": manifest.get("arrays"),
    }
    expected = {
        "integrity_schema": integrity["integrity_schema"],
        "source_csv_sha256": integrity["source_csv_sha256"],
        "text_files": integrity["text_files"],
        "arrays": integrity["arrays"],
    }
    if stable != expected:
        manifest = {**manifest, **expected}
        _atomic_json(Path(cohort_dir) / "manifest.json", manifest)
    return manifest


def build_or_load_cohort(
    cache_root, dataset_name, source_csv, *, load_text=True, verify_integrity=True
):
    """Create an ordered, label-independent cohort manifest."""

    source_csv = str(source_csv)
    pointer_dir = Path(cache_root) / "cohorts" / str(dataset_name)
    pointer_dir.mkdir(parents=True, exist_ok=True)
    pointer_path = pointer_dir / "current.json"
    if pointer_path.is_file():
        with open(pointer_path, encoding="utf-8") as handle:
            pointer = json.load(handle)
        manifest_path = pointer_dir / pointer["cohort_hash"] / "manifest.json"
        if manifest_path.is_file():
            with open(manifest_path, encoding="utf-8") as handle:
                manifest = json.load(handle)
            # Frozen graph-only training has already verified the source and
            # all cohort artifacts in the bundle.  Avoid rescanning a
            # million-row CSV on every DDP rank during a read-only reopen;
            # cache construction and integrity-checked callers still compute
            # the source hash below.
            source_hash = (
                _sha256_file(source_csv) if verify_integrity else None
            )
            if (
                manifest.get("schema") == COHORT_SCHEMA
                and (
                    not verify_integrity
                    or manifest.get("source_csv_sha256") == source_hash
                )
            ):
                try:
                    return load_cohort(
                        pointer_dir / pointer["cohort_hash"],
                        load_text=load_text,
                        verify_integrity=verify_integrity,
                    )
                except (FileNotFoundError, RuntimeError, ValueError):
                    pass

    # A missing/stale pointer must be rebuilt from the source CSV.  This is
    # intentionally after the fast frozen-cache path above so normal training
    # never pays this scan.
    source_hash = _sha256_file(source_csv)

    frame = pd.read_csv(source_csv)
    if frame.shape[1] < 1:
        raise ValueError(f"cohort CSV has no SMILES column: {source_csv}")
    raw_smiles = frame.iloc[:, 0].astype(str).str.strip().tolist()
    unique = []
    keys = []
    normalized_values = []
    chemistry_valid = []
    row_keys = []
    seen = set()
    started = time.monotonic()
    last_report = started
    valid_count = 0
    invalid_count = 0
    # Invalid P/Si structures are represented in the cohort and handled by
    # unavailable records later; their per-row RDKit diagnostics must not
    # drown out the aggregate million-row progress log.
    blocker = rdBase.BlockLogs()
    try:
        for row_index, smiles in enumerate(raw_smiles, start=1):
            normalized, valid = normalize_polymer_smiles(smiles)
            valid_count += int(valid)
            invalid_count += int(not valid)
            key = sample_key_from_normalized(normalized)
            row_keys.append(key)
            if key not in seen:
                seen.add(key)
                unique.append(smiles)
                keys.append(key)
                normalized_values.append(normalized)
                chemistry_valid.append(bool(valid))
            now = time.monotonic()
            if now - last_report >= 5.0:
                elapsed = max(now - started, 1e-9)
                rate = row_index / elapsed
                eta = (len(raw_smiles) - row_index) / max(rate, 1e-9)
                print(
                    "[cohort] "
                    f"completed={row_index}/{len(raw_smiles)} "
                    f"unique={len(unique)} valid={valid_count} "
                    f"invalid={invalid_count} rate={rate:.1f}/s "
                    f"eta={eta / 60.0:.1f}m",
                    flush=True,
                )
                last_report = now
    finally:
        del blocker
    elapsed = max(time.monotonic() - started, 1e-9)
    print(
        "[cohort] "
        f"completed={len(raw_smiles)}/{len(raw_smiles)} "
        f"unique={len(unique)} valid={valid_count} invalid={invalid_count} "
        f"rate={len(raw_smiles) / elapsed:.1f}/s",
        flush=True,
    )
    key_matrix = np.frombuffer(b"".join(keys), dtype=np.uint8).reshape(-1, 32)
    row_key_matrix = np.frombuffer(
        b"".join(row_keys), dtype=np.uint8
    ).reshape(-1, 32)
    ordered_key_hash = hashlib.sha256(key_matrix.tobytes()).hexdigest()
    source_row_key_hash = hashlib.sha256(row_key_matrix.tobytes()).hexdigest()
    cohort_hash = hashlib.sha256(
        f"{COHORT_SCHEMA}:{ordered_key_hash}:{source_row_key_hash}".encode("utf-8")
    ).hexdigest()
    cohort_dir = pointer_dir / cohort_hash
    cohort_dir.mkdir(parents=True, exist_ok=True)
    keys_tmp = cohort_dir / "sample_keys.npy.tmp"
    with open(keys_tmp, "wb") as handle:
        np.save(handle, key_matrix, allow_pickle=False)
    os.replace(keys_tmp, cohort_dir / "sample_keys.npy")
    row_keys_tmp = cohort_dir / "source_row_keys.npy.tmp"
    with open(row_keys_tmp, "wb") as handle:
        np.save(handle, row_key_matrix, allow_pickle=False)
    os.replace(row_keys_tmp, cohort_dir / "source_row_keys.npy")
    for name, values in (
        ("smiles.txt", unique),
        ("normalized_smiles.txt", normalized_values),
    ):
        temporary = cohort_dir / f"{name}.tmp"
        with open(temporary, "w", encoding="utf-8") as handle:
            for value in values:
                handle.write(json.dumps(value, ensure_ascii=False) + "\n")
        os.replace(temporary, cohort_dir / name)
    valid_tmp = cohort_dir / "chemistry_valid.npy.tmp"
    with open(valid_tmp, "wb") as handle:
        np.save(handle, np.asarray(chemistry_valid, dtype=np.bool_), allow_pickle=False)
    os.replace(valid_tmp, cohort_dir / "chemistry_valid.npy")
    manifest = {
        "schema": COHORT_SCHEMA,
        "dataset_name": str(dataset_name),
        "source_csv": os.path.abspath(source_csv),
        "source_csv_sha256": source_hash,
        "source_row_count": len(raw_smiles),
        "unique_count": len(unique),
        "ordered_sample_key_hash": ordered_key_hash,
        "source_row_key_hash": source_row_key_hash,
        "cohort_hash": cohort_hash,
    }
    manifest.update(_cohort_integrity_fields(cohort_dir, manifest))
    _atomic_json(cohort_dir / "manifest.json", manifest)
    _atomic_json(pointer_path, {"cohort_hash": cohort_hash})
    return load_cohort(
        cohort_dir, load_text=load_text, verify_integrity=verify_integrity
    )


def load_cohort(cohort_dir, *, load_text=True, verify_integrity=True):
    cohort_dir = Path(cohort_dir)
    with open(cohort_dir / "manifest.json", encoding="utf-8") as handle:
        manifest = json.load(handle)
    # Upgrade old v1 manifests in place.  The cohort identity is the key
    # ordering/hash and is deliberately unchanged by this integrity metadata.
    manifest = _upgrade_cohort_manifest_integrity(
        cohort_dir, manifest, verify=verify_integrity
    )
    keys_array = np.load(cohort_dir / "sample_keys.npy", mmap_mode="r")
    row_keys_array = np.load(
        cohort_dir / "source_row_keys.npy", mmap_mode="r"
    )
    if keys_array.shape != (int(manifest["unique_count"]), 32):
        raise RuntimeError("cohort sample key matrix has an invalid shape")
    if verify_integrity and hashlib.sha256(
        np.asarray(keys_array).tobytes()
    ).hexdigest() != manifest["ordered_sample_key_hash"]:
        raise RuntimeError("cohort ordered sample key hash mismatch")
    if row_keys_array.shape != (int(manifest["source_row_count"]), 32):
        raise RuntimeError("cohort source-row key matrix has an invalid shape")
    if verify_integrity and hashlib.sha256(
        np.asarray(row_keys_array).tobytes()
    ).hexdigest() != manifest["source_row_key_hash"]:
        raise RuntimeError("cohort source-row key hash mismatch")
    valid = np.load(cohort_dir / "chemistry_valid.npy", mmap_mode="r")
    if valid.shape != (len(keys_array),) or valid.dtype != np.bool_:
        raise RuntimeError("cohort chemistry_valid array has an invalid shape/dtype")
    smiles = normalized = None
    if load_text:
        with open(cohort_dir / "smiles.txt", encoding="utf-8") as handle:
            smiles = [json.loads(line) for line in handle if line.strip()]
        with open(cohort_dir / "normalized_smiles.txt", encoding="utf-8") as handle:
            normalized = [json.loads(line) for line in handle if line.strip()]
        if len(smiles) != len(keys_array) or len(normalized) != len(keys_array):
            raise RuntimeError("cohort text/key lengths differ")
    return {
        "root": str(cohort_dir),
        "manifest": manifest,
        "keys_array": keys_array,
        "row_keys_array": row_keys_array,
        # Keep the ordered key matrix mmap-backed.  Callers that need an LMDB
        # key pass an individual uint8 row; ``coerce_sample_key`` converts it
        # without materialising a million Python ``bytes`` objects.
        "keys": keys_array,
        "smiles": smiles,
        "normalized_smiles": normalized,
        "chemistry_valid": valid,
    }


def materialize_md200_array(cohort, descriptor_store, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    values_path = output_dir / "md200.npy"
    valid_path = output_dir / "md200_valid.npy"
    metadata_path = output_dir / "md200_metadata.json"
    expected = {
        "schema": MD200_ARRAY_SCHEMA,
        "cohort_hash": cohort["manifest"]["cohort_hash"],
        "ordered_sample_key_hash": cohort["manifest"]["ordered_sample_key_hash"],
        "descriptor_feature_config_hash": descriptor_store.meta[
            "feature_config_hash"
        ],
        "shape": [len(cohort["keys"]), 200],
        "dtype": "float32",
    }
    if metadata_path.is_file() and values_path.is_file() and valid_path.is_file():
        with open(metadata_path, encoding="utf-8") as handle:
            observed = json.load(handle)
        stable_observed = {
            key: observed.get(key) for key in expected
        }
        if stable_observed == expected:
            return str(values_path), str(valid_path), observed

    values_tmp = output_dir / "md200.npy.tmp"
    valid_tmp = output_dir / "md200_valid.npy.tmp"
    values = np.lib.format.open_memmap(
        values_tmp, mode="w+", dtype=np.float32, shape=(len(cohort["keys"]), 200)
    )
    validity = np.lib.format.open_memmap(
        valid_tmp, mode="w+", dtype=np.bool_, shape=(len(cohort["keys"]),)
    )
    for index, key in enumerate(cohort["keys"]):
        data = descriptor_store[key]
        vector = getattr(data, "mips_md", None)
        if not torch.is_tensor(vector) or vector.numel() != 200:
            raise RuntimeError(f"invalid MD200 cache row for {bytes(key).hex()}")
        values[index] = vector.detach().cpu().float().numpy()
        validity[index] = bool(getattr(data, "mips_md_valid", False))
    values.flush()
    validity.flush()
    del values, validity
    os.replace(values_tmp, values_path)
    os.replace(valid_tmp, valid_path)
    metadata = {
        **expected,
        "values_sha256": _sha256_file(values_path),
        "valid_sha256": _sha256_file(valid_path),
        "created_at": time.time(),
    }
    _atomic_json(metadata_path, metadata)
    return str(values_path), str(valid_path), metadata


def compute_mcl_thresholds(data):
    """Compute the two immutable Trimer hard-mask thresholds for one record.

    The result is deliberately a tiny ``float32[2]`` value.  Invalid or
    non-3-D records receive NaNs and remain exact O8 fallbacks; no distance
    matrix is retained in the cache.
    """
    invalid = np.full((2,), np.nan, dtype=np.float32)
    try:
        if not bool(getattr(data, "trimer_geometry_valid", False)):
            return invalid
        if not bool(getattr(data, "trimer_geometry_is_3d", False)):
            return invalid
        if bool(getattr(data, "trimer_2d_fallback", False)):
            return invalid
        positions = torch.as_tensor(getattr(data, "trimer_pos"))
        if positions.ndim != 2 or positions.size(-1) != 3:
            return invalid
        positions = positions.float()
        if positions.size(0) < 2 or not bool(torch.isfinite(positions).all()):
            return invalid
        distances = torch.pdist(positions)
        if not distances.numel() or not bool(torch.isfinite(distances).all()):
            return invalid
        values = torch.quantile(
            distances, distances.new_tensor((0.20, 0.50))
        ).cpu().numpy().astype(np.float32, copy=False)
        return values if np.isfinite(values).all() else invalid
    except (AttributeError, IndexError, RuntimeError, TypeError, ValueError):
        return invalid


def materialize_mcl_threshold_array(cohort, thresholds, trimer_root):
    """Atomically persist ``[N,2]`` thresholds in cohort order.

    This is a derived accelerator, not a replacement for the Trimer LMDB.  It
    is content-bound to the ordered cohort and Trimer ``.done`` artifact so a
    stale array is ignored by readers and rejected by validation.
    """
    cohort_root = Path(cohort["root"])
    values_path = cohort_root / "mcl_thresholds.npy"
    metadata_path = cohort_root / "mcl_thresholds_metadata.json"
    done_path = Path(trimer_root) / ".done"
    if not done_path.is_file():
        raise RuntimeError("cannot materialize MCL thresholds without Trimer .done")
    trimer_artifact_hash = done_path.read_text(encoding="utf-8").strip()
    thresholds = np.asarray(thresholds, dtype=np.float32)
    expected = {
        "schema": MCL_THRESHOLDS_ARRAY_SCHEMA,
        "cohort_hash": cohort["manifest"]["cohort_hash"],
        "ordered_sample_key_hash": cohort["manifest"]["ordered_sample_key_hash"],
        "trimer_artifact_hash": trimer_artifact_hash,
        "shape": list(thresholds.shape),
        "dtype": "float32",
    }
    if thresholds.ndim != 2 or thresholds.shape[1] != 2:
        raise ValueError("MCL thresholds must have shape [N,2]")
    if values_path.is_file() and metadata_path.is_file():
        try:
            observed = json.loads(metadata_path.read_text(encoding="utf-8"))
            stable = {key: observed.get(key) for key in expected}
            if stable == expected:
                loaded = np.load(values_path, mmap_mode="r")
                if tuple(loaded.shape) == tuple(thresholds.shape):
                    return str(values_path), observed
        except (OSError, ValueError, json.JSONDecodeError):
            pass
    cohort_root.mkdir(parents=True, exist_ok=True)
    temporary = values_path.with_suffix(values_path.suffix + ".tmp")
    array = np.lib.format.open_memmap(
        temporary, mode="w+", dtype=np.float32, shape=thresholds.shape
    )
    array[:] = thresholds
    array.flush()
    del array
    os.replace(temporary, values_path)
    metadata = {
        **expected,
        "values_sha256": _sha256_file(values_path),
        "created_at": time.time(),
    }
    _atomic_json(metadata_path, metadata)
    return str(values_path), metadata


class LmdbFeatureStore:
    """Compose O8, optional Trimer, and mmap'ed MD200 by content key."""

    schema = CACHE_LAYOUT_SCHEMA

    def __init__(
        self,
        *,
        topology_root,
        cohort,
        trimer_root=None,
        md200_path=None,
        md200_valid_path=None,
        angle_cache_root=None,
    ):
        self.roots = {"topology": str(topology_root)}
        self.topology = LmdbLayerStore(topology_root)
        self.trimer = None
        if trimer_root is not None:
            self.roots["trimer"] = str(trimer_root)
            self.trimer = LmdbLayerStore(trimer_root)
        self.cohort = cohort
        self.keys = cohort["keys_array"]
        self.mcl_thresholds = None
        self.mcl_thresholds_path = None
        # Read-side memoization for the two MCL hard-mask thresholds.  A
        # finalized Stage-3 cohort can use the persisted mmap below; the
        # graph-only stages use this bounded fallback without changing the
        # immutable LMDB records.
        # Thresholds are a derived read-side memo, not cache content.  Keep a
        # bounded LRU so a million-row pretraining cohort cannot accumulate a
        # million Python dictionary entries in every DDP rank.
        self._mcl_threshold_cache = OrderedDict()
        self._mcl_threshold_cache_limit = 4096
        threshold_path = Path(cohort["root"]) / "mcl_thresholds.npy"
        threshold_meta_path = Path(cohort["root"]) / "mcl_thresholds_metadata.json"
        # Thresholds are an independent, read-only Trimer artifact.  Stage 2
        # does not load MD200, but it must still use this mmap; coupling the
        # threshold lookup to ``md200_path`` silently caused an expensive
        # pdist/quantile recomputation for every Stage-2 sample.
        if (
            threshold_path.is_file()
            and threshold_meta_path.is_file()
            and self.trimer is not None
        ):
            try:
                threshold_meta = json.loads(
                    threshold_meta_path.read_text(encoding="utf-8")
                )
                trimer_done = Path(self.roots["trimer"]) / ".done"
                trimer_hash = trimer_done.read_text(encoding="utf-8").strip()
                if (
                    threshold_meta.get("schema") == MCL_THRESHOLDS_ARRAY_SCHEMA
                    and threshold_meta.get("cohort_hash")
                    == cohort["manifest"]["cohort_hash"]
                    and threshold_meta.get("ordered_sample_key_hash")
                    == cohort["manifest"]["ordered_sample_key_hash"]
                    and threshold_meta.get("trimer_artifact_hash") == trimer_hash
                    and tuple(threshold_meta.get("shape", ()))
                    == (len(self.keys), 2)
                ):
                    candidate = np.load(threshold_path, mmap_mode="r")
                    if tuple(candidate.shape) == (len(self.keys), 2):
                        self.mcl_thresholds = candidate
                        self.mcl_thresholds_path = str(threshold_path)
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                self.mcl_thresholds = None
        self.md200_path = md200_path
        self.md200_valid_path = md200_valid_path
        self.md200 = (
            np.load(md200_path, mmap_mode="r") if md200_path is not None else None
        )
        self.md200_valid = (
            np.load(md200_valid_path, mmap_mode="r")
            if md200_valid_path is not None else None
        )
        self.angle_cache_root = str(angle_cache_root) if angle_cache_root is not None else None
        self.angle_offsets = None
        self.angle_indices = None
        self.angle_bins = None
        self.angle_cos = None
        self.angle_valid = None
        self.angle_class_counts = None
        self.angle_validation_mask = None
        self.angle_cache_metadata = None
        self.angle_cache_artifact_hash = None
        if self.angle_cache_root is not None:
            angle_root = Path(self.angle_cache_root)
            metadata_path = angle_root / "metadata.json"
            try:
                if metadata_path.is_file():
                    angle_meta = json.loads(metadata_path.read_text(encoding="utf-8"))
                    done_path = angle_root / ".done"
                    frozen_path = angle_root / ".frozen"
                    angle_schema = angle_meta.get("schema")
                    if (
                        angle_schema not in {
                            "mts-trimer-bond-angle-cache-v1",
                            "mts-angle-continuous-cache-v1",
                        }
                        or not done_path.is_file()
                        or not frozen_path.is_file()
                    ):
                        raise RuntimeError("MTS angle cache is not frozen")
                    expected = {
                        "cohort_hash": cohort["manifest"]["cohort_hash"],
                        "ordered_sample_key_hash": cohort["manifest"]["ordered_sample_key_hash"],
                    }
                    if self.trimer is not None:
                        trimer_done = Path(self.roots["trimer"]) / ".done"
                        if not trimer_done.is_file():
                            raise RuntimeError("Trimer artifact marker is missing")
                        expected["trimer_artifact_hash"] = trimer_done.read_text(
                            encoding="utf-8"
                        ).strip()
                    if all(angle_meta.get(key) == value for key, value in expected.items()):
                        self.angle_offsets = np.load(angle_root / "angle_offsets.npy", mmap_mode="r")
                        self.angle_indices = np.load(angle_root / "angle_indices.npy", mmap_mode="r")
                        if angle_schema == "mts-trimer-bond-angle-cache-v1":
                            self.angle_bins = np.load(
                                angle_root / "angle_bins.npy", mmap_mode="r"
                            )
                            self.angle_valid = np.load(
                                angle_root / "angle_valid.npy", mmap_mode="r"
                            )
                            counts_path = angle_root / "angle_class_counts.npy"
                            if counts_path.is_file():
                                self.angle_class_counts = np.load(
                                    counts_path, mmap_mode="r"
                                )
                        else:
                            self.angle_cos = np.load(
                                angle_root / "angle_cos.npy", mmap_mode="r"
                            )
                            # The validation mask controls pretraining model
                            # selection, not geometric validity.  A record is
                            # angle-valid when the frozen cache contains at
                            # least one true-bond triplet; MCL validity is
                            # checked independently by the collator/model.
                            self.angle_valid = np.asarray(
                                self.angle_offsets[1:] > self.angle_offsets[:-1],
                                dtype=np.bool_,
                            )
                            self.angle_validation_mask = np.load(
                                angle_root / "validation_mask.npy", mmap_mode="r"
                            )
                        common_invalid = (
                            self.angle_offsets.shape != (len(self.keys) + 1,)
                            or self.angle_indices.ndim != 2
                            or self.angle_indices.shape[1] != 3
                            or self.angle_valid.shape != (len(self.keys),)
                        )
                        categorical_invalid = (
                            angle_schema == "mts-trimer-bond-angle-cache-v1"
                            and (
                                self.angle_bins.shape
                                != (self.angle_indices.shape[0],)
                                or self.angle_class_counts is None
                                or self.angle_class_counts.shape != (20,)
                                or self.angle_class_counts.dtype != np.int64
                            )
                        )
                        continuous_invalid = (
                            angle_schema == "mts-angle-continuous-cache-v1"
                            and (
                                self.angle_cos.shape
                                != (self.angle_indices.shape[0],)
                                or self.angle_cos.dtype != np.float32
                            )
                        )
                        if common_invalid or categorical_invalid or continuous_invalid:
                            raise RuntimeError("MTS angle cache shape mismatch")
                        self.angle_cache_metadata = angle_meta
                        self.angle_cache_artifact_hash = done_path.read_text(
                            encoding="utf-8"
                        ).strip()
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                self.angle_offsets = None
                self.angle_indices = None
                self.angle_bins = None
                self.angle_cos = None
                self.angle_valid = None
                self.angle_class_counts = None
                self.angle_validation_mask = None
                self.angle_cache_artifact_hash = None
        if self.md200 is not None:
            if self.md200.ndim != 2 or self.md200.shape[1] != 200:
                raise RuntimeError("MD200 mmap must have shape [N,200]")
            if self.md200.dtype != np.float32:
                raise RuntimeError("MD200 mmap must use float32")
        if self.md200_valid is not None:
            if self.md200_valid.ndim != 1 or self.md200_valid.shape[0] != len(self.keys):
                raise RuntimeError("MD200 validity mmap has an invalid shape")
            if self.md200_valid.dtype != np.bool_:
                raise RuntimeError("MD200 validity mmap must use bool")
        # MD200 lookup uses a compact sorted permutation rather than a
        # million-entry Python dictionary.  Do not even build that index for
        # graph-only Stage 1/1.5: those stages never request MD200 and should
        # pay neither the sort time nor the extra resident array.
        self._sorted_key_order = None
        self._sorted_key_view = None
        if (
            self.md200 is not None
            or self.md200_valid is not None
            or self.mcl_thresholds is not None
            or self.angle_offsets is not None
        ):
            key_view = np.asarray(self.keys, dtype=np.uint8).view("S32").reshape(-1)
            self._sorted_key_order = np.argsort(key_view, kind="mergesort")
            self._sorted_key_view = key_view[self._sorted_key_order]

    def __len__(self):
        return len(self.keys)

    def close(self):
        self.topology.close()
        if self.trimer is not None:
            self.trimer.close()
        # NumPy memmaps close when their backing mmap objects are collected.
        self.md200 = None
        self.md200_valid = None
        self.mcl_thresholds = None
        self.angle_offsets = None
        self.angle_indices = None
        self.angle_bins = None
        self.angle_cos = None
        self.angle_valid = None
        self.angle_class_counts = None
        self.angle_validation_mask = None
        self.angle_cache_artifact_hash = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __contains__(self, key):
        key = coerce_sample_key(key)
        if key not in self.topology:
            return False
        return self.trimer is None or key in self.trimer

    def _row_for_key(self, key):
        if self._sorted_key_view is None:
            return None
        # ``keys_sorted`` is a fixed-width byte-string array.  Construct the
        # lookup scalar directly from the 32-byte digest; converting through a
        # uint8 array and reshaping to one element is incorrect (it has 32
        # elements and raises for normal SHA256 keys).
        key = np.frombuffer(bytes(key), dtype=np.dtype("S32"), count=1)[0]
        position = int(np.searchsorted(self._sorted_key_view, key))
        if (
            position >= int(self._sorted_key_view.size)
            or self._sorted_key_view[position] != key
        ):
            return None
        return int(self._sorted_key_order[position])

    def __getitem__(self, key):
        key = coerce_sample_key(key)
        merged = copy.copy(self.topology[key])
        if self.trimer is not None:
            layer = self.trimer[key]
            for name in layer.keys():
                if name in merged:
                    raise RuntimeError(f"duplicate LMDB cache field: {name}")
                merged[name] = layer[name]
            if not hasattr(merged, "trimer_mcl_thresholds"):
                row = self._row_for_key(key) if self.mcl_thresholds is not None else None
                if row is not None:
                    thresholds = torch.from_numpy(
                        np.asarray(self.mcl_thresholds[row]).copy()
                    ).float()
                else:
                    cache_key = bytes(key)
                    thresholds = self._mcl_threshold_cache.get(cache_key)
                if thresholds is None:
                    positions = getattr(merged, "trimer_pos", None)
                    if positions is not None and torch.as_tensor(positions).ndim == 2:
                        positions = torch.as_tensor(positions).float()
                        if positions.size(0) >= 2 and bool(torch.isfinite(positions).all()):
                            distances = torch.pdist(positions)
                            thresholds = torch.quantile(
                                distances,
                                distances.new_tensor((0.20, 0.50)),
                            ).cpu()
                        else:
                            thresholds = torch.full((2,), float("nan"))
                    else:
                        thresholds = torch.full((2,), float("nan"))
                    self._mcl_threshold_cache[cache_key] = thresholds
                elif row is None:
                    self._mcl_threshold_cache.move_to_end(cache_key)
                while len(self._mcl_threshold_cache) > self._mcl_threshold_cache_limit:
                    self._mcl_threshold_cache.popitem(last=False)
                merged.trimer_mcl_thresholds = thresholds.clone()
        if self.md200 is not None:
            row = self._row_for_key(key)
            if row is None:
                raise KeyError(key.hex())
            merged.mips_md = torch.from_numpy(
                np.asarray(self.md200[row]).copy()
            ).float()
            merged.mips_md_valid = bool(self.md200_valid[row])
            merged.mips_descriptor_source = "source_star_sub"
            merged.mips_descriptor_optimizer = "not_applicable_2d"
            merged.mips_descriptor_schema_version = 5
            merged.descriptor_failure_code = (
                "" if merged.mips_md_valid else "md200_content_cache_invalid"
            )
        if self.angle_offsets is not None:
            row = self._row_for_key(key)
            if row is None:
                raise KeyError(key.hex())
            start = int(self.angle_offsets[row])
            end = int(self.angle_offsets[row + 1])
            merged.trimer_angle_index = torch.from_numpy(
                np.asarray(self.angle_indices[start:end], dtype=np.int64).copy()
            )
            if self.angle_bins is not None:
                merged.trimer_angle_bins = torch.from_numpy(
                    np.asarray(self.angle_bins[start:end], dtype=np.int64).copy()
                )
            if self.angle_cos is not None:
                merged.trimer_angle_cos = torch.from_numpy(
                    np.asarray(self.angle_cos[start:end], dtype=np.float32).copy()
                )
            merged.trimer_angle_valid = bool(self.angle_valid[row])
            merged.trimer_angle_cache_schema = str(
                self.angle_cache_metadata.get("schema", "")
            )
            merged.trimer_angle_cache_artifact_hash = str(
                self.angle_cache_metadata.get("trimer_artifact_hash", "")
            )
        return merged

    def values(self):
        for key in self.keys:
            yield self[key]
