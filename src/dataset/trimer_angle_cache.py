"""Derived finite-Trimer bond-angle targets for MTS joint pretraining.

The immutable Trimer LMDB remains the source of coordinates and real bonds.
This module only materializes compact, row-ordered angle indices/bins; it never
reruns ETKDG or MMFF and never treats a Star-Linking edge as a chemical bond.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from .mips_trimer_contract import CACHE_BOND_ANGLE_SCHEMA


ANGLE_CACHE_BUILDER_VERSION = 1
ANGLE_BIN_COUNT = 20


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def angle_cache_root(trimer_root, cohort_hash: str) -> Path:
    return Path(trimer_root) / "derived" / "bond_angle" / str(cohort_hash)


def _angle_bin(value: float) -> int:
    value = min(float(np.pi), max(0.0, float(value)))
    return min(ANGLE_BIN_COUNT - 1, int(np.floor(value / np.pi * ANGLE_BIN_COUNT)))


def _record_angles(data):
    """Return local [neighbor,center,neighbor] indices and bins for one Data."""
    geometry_valid = bool(getattr(data, "trimer_geometry_valid", False))
    is_3d = bool(torch.as_tensor(getattr(data, "trimer_geometry_is_3d", False)).item())
    fallback = bool(torch.as_tensor(getattr(data, "trimer_2d_fallback", False)).item())
    positions = getattr(data, "trimer_pos", None)
    central_mask = getattr(data, "trimer_central_ru_mask", None)
    edge_index = getattr(data, "trimer_edge_index", None)
    if not geometry_valid or not is_3d or fallback:
        return np.empty((0, 3), dtype=np.int32), np.empty((0,), dtype=np.uint8), False
    if positions is None or central_mask is None or edge_index is None:
        return np.empty((0, 3), dtype=np.int32), np.empty((0,), dtype=np.uint8), False
    positions = torch.as_tensor(positions).float()
    central_mask = torch.as_tensor(central_mask).bool().reshape(-1)
    edge_index = torch.as_tensor(edge_index).long()
    if positions.ndim != 2 or positions.size(-1) != 3 or not bool(torch.isfinite(positions).all()):
        return np.empty((0, 3), dtype=np.int32), np.empty((0,), dtype=np.uint8), False
    if edge_index.ndim != 2 or edge_index.size(0) != 2:
        return np.empty((0, 3), dtype=np.int32), np.empty((0,), dtype=np.uint8), False

    # The Trimer cache stores both directions of each real bond.  Deduplicate
    # by unordered local atom pair; no virtual Star edge is present here.
    neighbors = [[] for _ in range(int(positions.size(0)))]
    seen = set()
    for left, right in edge_index.t().tolist():
        left, right = int(left), int(right)
        if left == right or left < 0 or right < 0 or left >= len(neighbors) or right >= len(neighbors):
            continue
        pair = (min(left, right), max(left, right))
        if pair in seen:
            continue
        seen.add(pair)
        neighbors[left].append(right)
        neighbors[right].append(left)

    base_ids = torch.as_tensor(
        getattr(data, "trimer_base_ru_atom_id", torch.arange(len(neighbors)))
    ).long().reshape(-1)
    offsets = torch.as_tensor(
        getattr(data, "trimer_ru_offset", torch.zeros(len(neighbors)))
    ).long().reshape(-1)
    triples, bins = [], []
    for center in torch.nonzero(central_mask, as_tuple=False).flatten().tolist():
        endpoint_order = sorted(
            set(neighbors[int(center)]),
            key=lambda atom: (int(offsets[atom]), int(base_ids[atom]), int(atom)),
        )
        for first_pos in range(len(endpoint_order)):
            for second_pos in range(first_pos + 1, len(endpoint_order)):
                left = int(endpoint_order[first_pos])
                right = int(endpoint_order[second_pos])
                v_left = positions[left] - positions[int(center)]
                v_right = positions[right] - positions[int(center)]
                denominator = torch.linalg.vector_norm(v_left) * torch.linalg.vector_norm(v_right)
                if not bool(torch.isfinite(denominator)) or float(denominator) <= 1e-8:
                    continue
                cosine = torch.clamp(torch.dot(v_left, v_right) / denominator, -1.0, 1.0)
                angle = float(torch.acos(cosine).item())
                if not np.isfinite(angle):
                    continue
                triples.append((left, int(center), right))
                bins.append(_angle_bin(angle))
    return (
        np.asarray(triples, dtype=np.int32).reshape(-1, 3),
        np.asarray(bins, dtype=np.uint8),
        True,
    )


def validate_angle_cache(root, *, cohort_hash, ordered_key_hash, trimer_artifact_hash, record_count):
    root = Path(root)
    metadata_path = root / "metadata.json"
    required = [
        root / "angle_offsets.npy",
        root / "angle_indices.npy",
        root / "angle_bins.npy",
        root / "angle_valid.npy",
        root / "angle_class_counts.npy",
        metadata_path,
        root / ".done",
        root / ".frozen",
    ]
    if not all(path.is_file() for path in required):
        raise RuntimeError(f"MTS angle cache is incomplete: {root}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("schema") != CACHE_BOND_ANGLE_SCHEMA:
        raise RuntimeError("incompatible MTS angle cache schema")
    for key, expected in {
        "cohort_hash": str(cohort_hash),
        "ordered_sample_key_hash": str(ordered_key_hash),
        "trimer_artifact_hash": str(trimer_artifact_hash),
        "record_count": int(record_count),
    }.items():
        if metadata.get(key) != expected:
            raise RuntimeError(f"MTS angle cache {key} mismatch")
    metadata_artifact = hashlib.sha256(
        json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    done_artifact = (root / ".done").read_text(encoding="utf-8").strip()
    if done_artifact != metadata_artifact:
        raise RuntimeError("MTS angle cache .done does not match metadata")
    frozen = json.loads((root / ".frozen").read_text(encoding="utf-8"))
    if (
        frozen.get("schema") != CACHE_BOND_ANGLE_SCHEMA
        or frozen.get("artifact_hash") != done_artifact
    ):
        raise RuntimeError("MTS angle cache frozen marker is stale")
    offsets = np.load(root / "angle_offsets.npy", mmap_mode="r")
    indices = np.load(root / "angle_indices.npy", mmap_mode="r")
    bins = np.load(root / "angle_bins.npy", mmap_mode="r")
    valid = np.load(root / "angle_valid.npy", mmap_mode="r")
    counts = np.load(root / "angle_class_counts.npy", mmap_mode="r")
    if offsets.shape != (record_count + 1,) or offsets.dtype != np.int64:
        raise RuntimeError("invalid MTS angle offsets")
    if indices.ndim != 2 or indices.shape[1] != 3 or indices.dtype != np.int32:
        raise RuntimeError("invalid MTS angle indices")
    if bins.shape != (indices.shape[0],) or bins.dtype != np.uint8:
        raise RuntimeError("invalid MTS angle bins")
    if valid.shape != (record_count,) or valid.dtype != np.bool_:
        raise RuntimeError("invalid MTS angle validity")
    if counts.shape != (ANGLE_BIN_COUNT,) or counts.dtype != np.int64:
        raise RuntimeError("invalid MTS angle class counts")
    if int(offsets[0]) != 0 or int(offsets[-1]) != int(indices.shape[0]):
        raise RuntimeError("MTS angle offsets are not contiguous")
    if indices.size and (int(bins.min()) < 0 or int(bins.max()) >= ANGLE_BIN_COUNT):
        raise RuntimeError("MTS angle bin outside 0..19")
    for filename in (
        "angle_offsets.npy", "angle_indices.npy", "angle_bins.npy",
        "angle_valid.npy", "angle_class_counts.npy",
    ):
        digest = _sha256_file(root / filename)
        if metadata.get("files", {}).get(filename) != digest:
            raise RuntimeError(f"MTS angle cache file hash mismatch: {filename}")
    return metadata


def build_angle_cache(cohort, trimer_store, trimer_root, trimer_artifact_hash):
    """Build or reuse a row-ordered angle cache from an immutable Trimer store."""
    manifest = cohort["manifest"]
    root = angle_cache_root(trimer_root, manifest["cohort_hash"])
    root.mkdir(parents=True, exist_ok=True)
    try:
        metadata = validate_angle_cache(
            root,
            cohort_hash=manifest["cohort_hash"],
            ordered_key_hash=manifest["ordered_sample_key_hash"],
            trimer_artifact_hash=trimer_artifact_hash,
            record_count=len(cohort["keys"]),
        )
        return root, metadata
    except Exception:
        pass

    offsets = np.zeros(len(cohort["keys"]) + 1, dtype=np.int64)
    valid = np.zeros(len(cohort["keys"]), dtype=np.bool_)
    counts = np.zeros(ANGLE_BIN_COUNT, dtype=np.int64)
    started = time.monotonic()
    # Do not retain one Python object per angle: a million-row cohort can
    # contain several million targets.  Append compact records to raw
    # temporary streams, then materialize the final NPY arrays in one bounded
    # pass.  If interrupted, the temporary streams are harmless and the next
    # invocation rebuilds them before publishing .done.
    indices_raw = root / "angle_indices.raw.tmp"
    bins_raw = root / "angle_bins.raw.tmp"
    angle_count = 0
    with indices_raw.open("wb") as indices_handle, bins_raw.open("wb") as bins_handle:
        for row, key in enumerate(cohort["keys"]):
            data = trimer_store[key]
            local_indices, local_bins, geometry_available = _record_angles(data)
            valid[row] = bool(geometry_available and len(local_bins) > 0)
            if len(local_indices):
                local_indices.astype(np.int32, copy=False).tofile(indices_handle)
                local_bins.astype(np.uint8, copy=False).tofile(bins_handle)
                counts += np.bincount(
                    local_bins.astype(np.int64), minlength=ANGLE_BIN_COUNT
                )
                angle_count += int(len(local_bins))
            offsets[row + 1] = angle_count
            if row and row % 5000 == 0:
                elapsed = max(time.monotonic() - started, 1e-9)
                print(
                    f"[mts-angle-cache] {row}/{len(cohort['keys'])} "
                    f"angles={angle_count} samples/s={row/elapsed:.1f}",
                    flush=True,
                )

    # Small arrays are written normally; the two potentially large arrays are
    # filled from the raw streams in chunks so peak RAM is independent of the
    # total number of angle targets.
    fixed_files = {
        "angle_offsets.npy": offsets,
        "angle_valid.npy": valid,
        "angle_class_counts.npy": counts,
    }
    temporary_paths = {}
    for filename, array in fixed_files.items():
        destination = root / filename
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        with temporary.open("wb") as handle:
            np.save(handle, array)
        temporary_paths[filename] = temporary

    def materialize_raw(raw_path, destination, dtype, shape, chunk_values=1 << 20):
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        if int(np.prod(shape)) == 0:
            with temporary.open("wb") as handle:
                np.save(handle, np.empty(shape, dtype=dtype))
        else:
            mmap = np.lib.format.open_memmap(
                temporary, mode="w+", dtype=dtype, shape=shape
            )
            cursor = 0
            with raw_path.open("rb") as handle:
                while cursor < int(np.prod(shape)):
                    count = min(chunk_values, int(np.prod(shape)) - cursor)
                    chunk = np.fromfile(handle, dtype=dtype, count=count)
                    if chunk.size != count:
                        raise RuntimeError(f"truncated MTS angle stream: {raw_path}")
                    mmap.reshape(-1)[cursor:cursor + count] = chunk
                    cursor += count
            mmap.flush()
            del mmap
        temporary_paths[destination.name] = temporary

    materialize_raw(
        indices_raw, root / "angle_indices.npy", np.int32, (angle_count, 3)
    )
    materialize_raw(
        bins_raw, root / "angle_bins.npy", np.uint8, (angle_count,)
    )
    for filename, temporary in temporary_paths.items():
        os.replace(temporary, root / filename)
    indices_raw.unlink(missing_ok=True)
    bins_raw.unlink(missing_ok=True)
    metadata = {
        "schema": CACHE_BOND_ANGLE_SCHEMA,
        "builder_version": ANGLE_CACHE_BUILDER_VERSION,
        "cohort_hash": manifest["cohort_hash"],
        "ordered_sample_key_hash": manifest["ordered_sample_key_hash"],
        "trimer_artifact_hash": str(trimer_artifact_hash),
        "record_count": len(cohort["keys"]),
        "angle_count": int(angle_count),
        "angle_bin_count": ANGLE_BIN_COUNT,
        "center_policy": "central_ru_only",
        "bond_policy": "trimer_real_bonds_only",
        "bin_range": [0.0, float(np.pi)],
        "created_at": time.time(),
        "files": {
            filename: _sha256_file(root / filename)
            for filename in (
                "angle_offsets.npy", "angle_indices.npy", "angle_bins.npy",
                "angle_valid.npy", "angle_class_counts.npy",
            )
        },
    }
    _atomic_json(root / "metadata.json", metadata)
    artifact_hash = hashlib.sha256(
        json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    (root / ".done.tmp").write_text(artifact_hash + "\n", encoding="utf-8")
    os.replace(root / ".done.tmp", root / ".done")
    _atomic_json(root / ".frozen", {"schema": CACHE_BOND_ANGLE_SCHEMA, "artifact_hash": artifact_hash, "metadata_hash": artifact_hash})
    validate_angle_cache(
        root,
        cohort_hash=manifest["cohort_hash"],
        ordered_key_hash=manifest["ordered_sample_key_hash"],
        trimer_artifact_hash=trimer_artifact_hash,
        record_count=len(cohort["keys"]),
    )
    return root, metadata
