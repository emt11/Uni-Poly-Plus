"""Persistent static records for the dual GLT training route.

This module only materializes deterministic chemistry/connectivity derived from
the frozen Topology/Trimer records.  Coordinate-dependent distances and angles
are deliberately computed by :func:`materialize_dual_geometry` at read time.
"""

from __future__ import annotations

from collections import OrderedDict
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from .cache_lifecycle import CacheLifecycleError, atomic_json, json_hash
from .glt_dual_cache import ordered_key_hash
from .periodic_line_glt_complete import (
    build_complete_trimer_glt_sample, empty_complete_trimer_row,
)


STATIC_FORMAT = "glt-dual-static-v1"
# Bounded chunk-mapping cache.  The default keeps the original behaviour; a
# larger capacity is a benchmark candidate and must be requested explicitly.
CHUNK_CACHE_CAPACITY = 2
TARGET_FORMAT = "glt-dual-pretrain-targets-v1"
STATIC_FIELDS = (
    "bond_path_features", "bond_path_mask", "token_pos_index_a",
    "token_pos_index_b", "token_z_a", "token_z_b", "token_bond_type",
    "token_center_mask", "line_source", "line_target", "line_path",
    "line_path_mask", "line_path_group", "line_is_self",
    "angle_pos_triplet", "distance_token_index", "angle_pairs",
)
TARGET_FIELDS = ("brics_groups", "fingerprint_packed")


def _as_numpy(value, dtype=None):
    array = np.asarray(value)
    return array.astype(dtype, copy=False) if dtype is not None else array


def _projected_position_index(trimer):
    """Return (base, offset)->original trimer_pos index for heavy endpoints."""
    positions = torch.as_tensor(trimer.trimer_pos)
    base = torch.as_tensor(trimer.trimer_base_ru_atom_id).long().reshape(-1)
    offsets = torch.as_tensor(trimer.trimer_ru_offset).long().reshape(-1)
    heavy = getattr(trimer, "trimer_heavy_indices", None)
    if heavy is None:
        heavy_indices = torch.arange(positions.size(0), dtype=torch.long)
    else:
        heavy_indices = torch.as_tensor(heavy).long().reshape(-1)
    if base.numel() != positions.size(0) or offsets.numel() != positions.size(0):
        raise ValueError("static trimer identity dimensions disagree")
    if heavy_indices.numel() == 0 or heavy_indices.numel() > positions.size(0):
        raise ValueError("static trimer heavy identity is empty")
    if heavy_indices.max() >= positions.size(0) or heavy_indices.min() < 0:
        raise ValueError("static trimer heavy index is out of range")
    # ``trimer_heavy_indices`` contains indices into the all-atom coordinate
    # array; do not confuse its enumeration position with the original atom
    # identity (explicit H/D entries make those orders diverge).
    projected = {(int(base[original]), int(offsets[original])): int(original)
                 for original in heavy_indices.tolist()}
    if len(projected) != len(heavy_indices):
        raise ValueError("static trimer physical identity is not unique")
    return projected


def _token_position_indices(tokens, topology, trimer):
    mapping = torch.as_tensor(
        topology.canonical_to_trimer_base_atom_id, dtype=torch.long
    ).reshape(-1).tolist()
    projected = _projected_position_index(trimer)
    positions_a, positions_b = [], []
    for canonical_a, canonical_b, shift_a, shift_b in zip(
        tokens["token_atom_a"], tokens["token_atom_b"],
        tokens["token_anchor_q_a"], tokens["token_anchor_q_b"],
    ):
        try:
            positions_a.append(projected[(mapping[int(canonical_a)], int(shift_a))])
            positions_b.append(projected[(mapping[int(canonical_b)], int(shift_b))])
        except (IndexError, KeyError) as exc:
            raise ValueError("static token endpoint identity is not mapped") from exc
    return np.asarray(positions_a, dtype=np.int32), np.asarray(positions_b, dtype=np.int32)


def _angle_triplets(paths, path_mask, self_flags, token_a, token_b):
    triplets = np.full((len(paths), 2, 3), -1, dtype=np.int32)
    endpoint_sets = [
        {int(token_a[index]), int(token_b[index])}
        for index in range(len(token_a))
    ]
    for path_index, path in enumerate(paths):
        if bool(self_flags[path_index]):
            continue
        for hop in range(2):
            if not bool(path_mask[path_index, hop]):
                continue
            source, target = int(path[hop]), int(path[hop + 1])
            shared = endpoint_sets[source] & endpoint_sets[target]
            if len(shared) != 1:
                raise ValueError("static line relation has ambiguous shared endpoint")
            center = next(iter(shared))
            source_outer = next(iter(endpoint_sets[source] - {center}))
            target_outer = next(iter(endpoint_sets[target] - {center}))
            triplets[path_index, hop] = (source_outer, center, target_outer)
    return triplets


def build_dual_static(topology, trimer, smiles):
    """Build one deterministic static row using the existing reference path."""
    from .glt_dual import bond_paths, two_hop_paths

    from .canonical_periodic import resolve_normalized_identity
    identity = resolve_normalized_identity(topology, smiles, require_fields=True)
    path_features, path_mask = bond_paths(topology, smiles, identity=identity)
    try:
        row = build_complete_trimer_glt_sample(
            topology, trimer, smiles, identity=identity
        )
    except Exception:
        # Preserve contract errors; this only gives the caller a single error
        # boundary and never converts one into an invalid geometry row.
        raise
    if row["geometry_valid"]:
        paths = two_hop_paths(row)
    else:
        if row["invalid_reason"] not in {
            "geometry_invalid", "non_3d_geometry", "2d_fallback",
            "trimer_coordinates_invalid", "bond_distance_invalid",
        }:
            raise ValueError("static Trimer contract failure: " + str(row["invalid_reason"]))
        paths = two_hop_paths(empty_complete_trimer_row(row["invalid_reason"]))
    tokens = row["tokens"]
    token_position_a, token_position_b = _token_position_indices(tokens, topology, trimer) \
        if row["geometry_valid"] else (np.empty(0, np.int32), np.empty(0, np.int32))
    line_path = _as_numpy(paths["path"], np.int32)
    line_path_mask = _as_numpy(paths["mask"], bool)
    line_is_self = _as_numpy(paths["is_self"], bool)
    line_path_group = _as_numpy(paths["path_group"], np.int32)
    token_a = _as_numpy(tokens["token_atom_a"], np.int32)
    token_b = _as_numpy(tokens["token_atom_b"], np.int32)
    angle_triplets = _angle_triplets(
        line_path, line_path_mask, line_is_self, token_position_a, token_position_b
    ) if row["geometry_valid"] else np.empty((0, 2, 3), np.int32)
    centers = _as_numpy(tokens["token_center_internal"], bool)
    angle_pairs = []
    for index, path in enumerate(line_path):
        if bool(line_is_self[index]) or int(line_path_mask[index].sum()) != 1:
            continue
        left, right = int(path[0]), int(path[1])
        if left < right and bool(centers[left]) and bool(centers[right]):
            angle_pairs.append((left, right))
    return {
        "geometry_valid": bool(row["geometry_valid"]),
        "geometry_invalid_reason": str(row["invalid_reason"]),
        "bond_path_features": _as_numpy(path_features, np.float32),
        "bond_path_mask": _as_numpy(path_mask, bool),
        "token_pos_index_a": token_position_a,
        "token_pos_index_b": token_position_b,
        "token_z_a": _as_numpy(tokens["token_endpoint_z_a"], np.uint8),
        "token_z_b": _as_numpy(tokens["token_endpoint_z_b"], np.uint8),
        "token_bond_type": _as_numpy(tokens["token_bond_type"], np.uint8),
        "token_center_mask": centers,
        "line_source": _as_numpy(paths["source"], np.int32),
        "line_target": _as_numpy(paths["target"], np.int32),
        "line_path": line_path,
        "line_path_mask": line_path_mask,
        "line_path_group": line_path_group,
        "line_is_self": line_is_self,
        "angle_pos_triplet": angle_triplets,
        "distance_token_index": np.flatnonzero(centers).astype(np.int32),
        "angle_pairs": np.asarray(angle_pairs, dtype=np.int32).reshape(-1, 2),
    }


def build_pretrain_target(smiles):
    """Return packed deterministic BRICS groups and Morgan bits."""
    from .glt_dual_pretrain import chemical_targets

    groups, fingerprint = chemical_targets(str(smiles))
    bits = torch.as_tensor(fingerprint).detach().cpu().numpy().astype(np.uint8)
    if bits.shape != (2048,) or not np.isin(bits, (0, 1)).all():
        raise ValueError("pretrain fingerprint has invalid shape or bits")
    return {
        "brics_groups": tuple(tuple(int(atom) for atom in group) for group in groups),
        "fingerprint_packed": np.packbits(bits, bitorder="little").astype(np.uint8),
    }


def materialize_dual_geometry(static, positions):
    """Materialize dynamic distances/angles from frozen coordinates."""
    positions = torch.as_tensor(positions).float()
    if not bool(static["geometry_valid"]):
        return {
            "geometry_valid": False,
            "bond_distance": torch.empty(0),
            "line_angle": torch.empty((0, 2)),
        }
    endpoint_a = torch.from_numpy(np.array(static["token_pos_index_a"], dtype=np.int64, copy=True))
    endpoint_b = torch.from_numpy(np.array(static["token_pos_index_b"], dtype=np.int64, copy=True))
    if endpoint_a.numel() != endpoint_b.numel() or endpoint_a.numel() == 0:
        raise ValueError("static token endpoints are empty or mismatched")
    if endpoint_a.max() >= positions.size(0) or endpoint_b.max() >= positions.size(0):
        raise ValueError("static token endpoint exceeds Trimer coordinate count")
    distances = torch.linalg.vector_norm(positions[endpoint_a] - positions[endpoint_b], dim=-1)
    triplets = torch.from_numpy(np.array(static["angle_pos_triplet"], dtype=np.int64, copy=True))
    mask = torch.from_numpy(np.array(static["line_path_mask"], dtype=bool, copy=True))
    angles = torch.zeros(mask.shape, dtype=positions.dtype)
    valid_triplet = mask & (triplets[..., 0] >= 0)
    if bool(valid_triplet.any()):
        selected = triplets[valid_triplet]
        first = positions[selected[:, 0]] - positions[selected[:, 1]]
        second = positions[selected[:, 2]] - positions[selected[:, 1]]
        denominator = torch.linalg.vector_norm(first, dim=-1) * torch.linalg.vector_norm(second, dim=-1)
        if not bool(torch.isfinite(denominator).all()) or bool((denominator <= 0).any()):
            raise ValueError("static angle triplet is degenerate")
        cosine = torch.clamp((first * second).sum(-1) / denominator, -1.0, 1.0)
        values = torch.acos(cosine)
        if not bool(torch.isfinite(values).all()):
            raise ValueError("static angle materialization is nonfinite")
        angles[valid_triplet] = values
    if not bool(torch.isfinite(distances).all()) or bool((distances <= 0).any()):
        raise ValueError("static bond distance is invalid")
    return {"geometry_valid": True, "bond_distance": distances, "line_angle": angles}


def _empty_array(name):
    shapes = {
        "bond_path_features": (0, 2, 14), "bond_path_mask": (0, 2),
        "token_pos_index_a": (0,), "token_pos_index_b": (0,),
        "token_z_a": (0,), "token_z_b": (0,), "token_bond_type": (0,),
        "token_center_mask": (0,), "line_source": (0,), "line_target": (0,),
        "line_path": (0, 3), "line_path_mask": (0, 2),
        "line_path_group": (0,), "line_is_self": (0,),
        "angle_pos_triplet": (0, 2, 3), "distance_token_index": (0,),
        "angle_pairs": (0, 2),
    }
    dtypes = {
        "bond_path_features": np.float32, "bond_path_mask": bool,
        "token_pos_index_a": np.int32, "token_pos_index_b": np.int32,
        "token_z_a": np.uint8, "token_z_b": np.uint8,
        "token_bond_type": np.uint8, "token_center_mask": bool,
        "line_source": np.int32, "line_target": np.int32,
        "line_path": np.int32, "line_path_mask": bool,
        "line_path_group": np.int32, "line_is_self": bool,
        "angle_pos_triplet": np.int32, "distance_token_index": np.int32,
        "angle_pairs": np.int32,
    }
    return np.empty(shapes[name], dtype=dtypes[name])


def _pack_rows(rows, target=False):
    fields = TARGET_FIELDS if target else STATIC_FIELDS
    offsets_name = "group_offsets" if target else None
    packed = {}
    if target:
        group_ptr, atom_ptr, atoms, fingerprints = [0], [0], [], []
        for row in rows:
            for group in row["brics_groups"]:
                atoms.extend(group)
                atom_ptr.append(len(atoms))
            group_ptr.append(len(atom_ptr) - 1)
            fingerprints.append(row["fingerprint_packed"])
        packed.update(
            brics_sample_group_ptr=np.asarray(group_ptr, dtype=np.int64),
            brics_group_atom_ptr=np.asarray(atom_ptr, dtype=np.int64),
            brics_atom_index=np.asarray(atoms, dtype=np.int32),
            fingerprint_packed=np.stack(fingerprints).astype(np.uint8)
            if fingerprints else np.empty((0, 256), dtype=np.uint8),
        )
        return packed
    for name in fields:
        values = [np.asarray(row[name]) for row in rows]
        if name in {"bond_path_features", "bond_path_mask"}:
            packed[name] = np.concatenate(values, axis=0) if values else _empty_array(name)
        else:
            packed[name] = np.concatenate(values, axis=0) if values else _empty_array(name)
    packed["geometry_valid"] = np.asarray(
        [bool(row["geometry_valid"]) for row in rows], dtype=bool
    )
    packed["geometry_invalid_reason"] = np.asarray(
        [str(row["geometry_invalid_reason"]) for row in rows], dtype="<U128"
    )
    for name in ("bond_path", "token", "line_relation", "line_path", "distance_token", "angle_pair"):
        source = {
            "bond_path": "bond_path_features", "token": "token_pos_index_a",
            "line_relation": "line_source", "line_path": "line_path",
            "distance_token": "distance_token_index", "angle_pair": "angle_pairs",
        }[name]
        lengths = [len(row[source]) for row in rows]
        packed[name + "_offsets"] = np.asarray([0, *np.cumsum(lengths)], dtype=np.int64)
    return packed


def _quarantine(path, quarantine_root):
    """Move an interrupted leftover aside instead of deleting it."""

    quarantine_root = Path(quarantine_root)
    quarantine_root.mkdir(parents=True, exist_ok=True)
    target = quarantine_root / f"{Path(path).name}.attempt_{int(time.time() * 1000)}"
    os.replace(path, target)
    return target


def write_chunk(root, start, rows, *, targets=False, quarantine_root=None):
    """Write one complete chunk atomically.

    The payload is written into a private temporary directory inside this
    staging root and only then renamed to its final name, so a chunk that
    exists together with ``.complete`` is always a finished chunk.  An
    interrupted leftover is quarantined only when ``quarantine_root`` is given,
    i.e. when the caller has confirmed the staging belongs to this build; a
    completed chunk is never overwritten.
    """

    root = Path(root)
    chunks = root / "chunks"
    chunks.mkdir(parents=True, exist_ok=True)
    chunk = chunks / f"chunk_{start:08d}"
    temp = chunks / f".tmp_chunk_{start:08d}"
    if chunk.exists():
        if (chunk / ".complete").is_file():
            raise FileExistsError(f"refusing to overwrite completed static chunk: {chunk}")
        if quarantine_root is None:
            raise FileExistsError(f"refusing to overwrite static chunk: {chunk}")
        _quarantine(chunk, quarantine_root)

    # A process can be interrupted after the completion marker is durable but
    # before the final directory rename.  A valid, current-owned temporary
    # chunk is safe to promote; an incomplete or corrupt one is only moved to
    # the caller's quarantine area, never silently treated as complete.
    if temp.exists():
        if (temp / ".complete").is_file():
            try:
                observed = json.loads((temp / "manifest.json").read_text(encoding="utf-8"))
                if (int(observed.get("start", -1)) != int(start)
                        or int(observed.get("count", -1)) != len(rows)
                        or bool(observed.get("target")) != bool(targets)):
                    raise CacheLifecycleError("temporary chunk identity does not match request")
                load_chunk_payload(temp, observed, targets=targets)
            except Exception as exc:
                if quarantine_root is None:
                    raise CacheLifecycleError(
                        f"completed temporary chunk is invalid: {temp}") from exc
                _quarantine(temp, quarantine_root)
            else:
                os.replace(temp, chunk)
                return observed
        else:
            if quarantine_root is None:
                raise FileExistsError(f"refusing to overwrite static chunk: {temp}")
            _quarantine(temp, quarantine_root)
    temp.mkdir(parents=True)
    packed = _pack_rows(rows, target=targets)
    for name, value in packed.items():
        np.save(temp / f"{name}.npy", value)
    manifest = {"start": int(start), "count": len(rows), "target": bool(targets),
                "arrays": {name: {"shape": list(value.shape), "dtype": str(value.dtype)}
                           for name, value in packed.items()}}
    atomic_json(temp / "manifest.json", manifest)
    (temp / ".complete").write_text("complete\n", encoding="utf-8")
    os.replace(temp, chunk)
    return manifest


SAMPLE_OFFSET_TABLES = (
    "bond_path_offsets", "token_offsets", "line_relation_offsets",
    "line_path_offsets", "distance_token_offsets", "angle_pair_offsets",
    "brics_sample_group_ptr",
)

RAGGED_PAYLOADS = (
    ("bond_path_features", "bond_path_offsets"),
    ("bond_path_mask", "bond_path_offsets"),
    ("token_pos_index_a", "token_offsets"),
    ("token_pos_index_b", "token_offsets"),
    ("token_z_a", "token_offsets"),
    ("token_z_b", "token_offsets"),
    ("token_bond_type", "token_offsets"),
    ("token_center_mask", "token_offsets"),
    ("line_source", "line_relation_offsets"),
    ("line_target", "line_relation_offsets"),
    ("line_path", "line_path_offsets"),
    ("line_path_mask", "line_path_offsets"),
    ("line_path_group", "line_path_offsets"),
    ("line_is_self", "line_path_offsets"),
    ("angle_pos_triplet", "line_path_offsets"),
    ("distance_token_index", "distance_token_offsets"),
    ("angle_pairs", "angle_pair_offsets"),
    ("brics_atom_index", "brics_group_atom_ptr"),
)


def load_chunk_payload(chunk, item, *, targets=False):
    """Load one chunk's arrays and verify its recorded payload contract.

    Only headers and the offset tables are inspected: this is deliberately not
    a content hash, but a missing file, a truncated array, a shape/dtype change
    or an offset table that does not partition the payload must not pass.
    """

    recorded = item.get("arrays")
    if not isinstance(recorded, dict) or not recorded:
        raise CacheLifecycleError(f"chunk manifest records no arrays: {chunk}")
    if targets:
        required = {
            "brics_sample_group_ptr", "brics_group_atom_ptr", "brics_atom_index",
            "fingerprint_packed",
        }
    else:
        required = set(STATIC_FIELDS) | {
            "geometry_valid", "geometry_invalid_reason", "bond_path_offsets",
            "token_offsets", "line_relation_offsets", "line_path_offsets",
            "distance_token_offsets", "angle_pair_offsets",
        }
    missing = sorted(required - set(recorded))
    if missing:
        raise CacheLifecycleError(f"chunk manifest lacks required arrays: {chunk}: {missing}")
    arrays = {}
    for name, expected in sorted(recorded.items()):
        path = Path(chunk) / f"{name}.npy"
        if not path.is_file():
            raise CacheLifecycleError(f"chunk payload file is missing: {path}")
        try:
            value = np.load(path, mmap_mode="r")
        except Exception as exc:
            raise CacheLifecycleError(f"chunk payload cannot be read: {path}") from exc
        if list(value.shape) != list(expected.get("shape", [])):
            raise CacheLifecycleError(f"chunk payload shape mismatch: {path}")
        if str(value.dtype) != str(expected.get("dtype")):
            raise CacheLifecycleError(f"chunk payload dtype mismatch: {path}")
        arrays[name] = value
    count = int(item["count"])
    ragged_names = {name for name, _ in RAGGED_PAYLOADS}
    # Sample-level offset tables carry one more entry than rows.  Payload-level
    # pointers (for example the BRICS group->atom table) are instead checked
    # against the array they partition, through RAGGED_PAYLOADS below.
    for name in SAMPLE_OFFSET_TABLES:
        if name not in arrays:
            continue
        table = np.asarray(arrays[name])
        if int(table.shape[0]) != count + 1:
            raise CacheLifecycleError(f"chunk sample offset table shape mismatch: {name}")
        if int(table[0]) != 0 or np.any(np.diff(table) < 0):
            raise CacheLifecycleError(f"chunk offset table is not a monotone partition: {name}")
    for name, value in arrays.items():
        if name.endswith("offsets") or name.endswith("_ptr") or name in ragged_names:
            continue
        if value.ndim == 0 or int(value.shape[0]) != count:
            raise CacheLifecycleError(f"chunk payload row count mismatch: {name}")
    for name, value in arrays.items():
        if not name.endswith("_ptr") or name in SAMPLE_OFFSET_TABLES:
            continue
        table = np.asarray(value)
        if int(table[0]) != 0 or np.any(np.diff(table) < 0):
            raise CacheLifecycleError(f"chunk pointer table is not a monotone partition: {name}")
    for name, offset_name in RAGGED_PAYLOADS:
        if name not in arrays or offset_name not in arrays:
            continue
        table = np.asarray(arrays[offset_name])
        if int(table[-1]) != int(arrays[name].shape[0]):
            raise CacheLifecycleError(
                f"chunk ragged payload length does not match its offsets: {name}")
    if targets:
        for name in ("brics_sample_group_ptr", "fingerprint_packed"):
            if name not in arrays:
                raise CacheLifecycleError(f"target chunk lacks pretrain targets: {chunk}")
    elif "geometry_valid" not in arrays or "geometry_invalid_reason" not in arrays:
        raise CacheLifecycleError(f"static chunk lacks geometry flags: {chunk}")
    return arrays


class _ChunkReader:
    def __init__(self, root, manifest, *, targets=False, cache_capacity=CHUNK_CACHE_CAPACITY):
        self.root = Path(root)
        self.manifest = manifest
        self.targets = bool(targets)
        self.cache_capacity = max(1, int(cache_capacity))
        self.starts = np.asarray([int(item["start"]) for item in manifest["chunks"]], dtype=np.int64)
        self._cache = OrderedDict()

    def _load(self, chunk_id):
        if chunk_id in self._cache:
            self._cache.move_to_end(chunk_id)
            return self._cache[chunk_id]
        item = self.manifest["chunks"][chunk_id]
        chunk = self.root / item["path"]
        arrays = load_chunk_payload(chunk, item, targets=self.targets)
        self._cache[chunk_id] = arrays
        while len(self._cache) > self.cache_capacity:
            self._cache.popitem(last=False)
        return arrays

    def get(self, index):
        index = int(index)
        chunk_id = int(np.searchsorted(self.starts, index, side="right") - 1)
        if chunk_id < 0:
            raise IndexError(index)
        item = self.manifest["chunks"][chunk_id]
        local = index - int(item["start"])
        if local < 0 or local >= int(item["count"]):
            raise IndexError(index)
        arrays = self._load(chunk_id)
        result = {}
        if "geometry_valid" in arrays:
            ragged = {
                "bond_path_features": "bond_path_offsets", "bond_path_mask": "bond_path_offsets",
                "token_pos_index_a": "token_offsets", "token_pos_index_b": "token_offsets",
                "token_z_a": "token_offsets", "token_z_b": "token_offsets",
                "token_bond_type": "token_offsets", "token_center_mask": "token_offsets",
                "line_source": "line_relation_offsets", "line_target": "line_relation_offsets",
                "line_path": "line_path_offsets", "line_path_mask": "line_path_offsets",
                "line_path_group": "line_path_offsets", "line_is_self": "line_path_offsets",
                "angle_pos_triplet": "line_path_offsets", "distance_token_index": "distance_token_offsets",
                "angle_pairs": "angle_pair_offsets",
            }
            for name, offset_name in ragged.items():
                offsets = arrays[offset_name]
                left, right = int(offsets[local]), int(offsets[local + 1])
                result[name] = np.asarray(arrays[name][left:right])
            result["geometry_valid"] = bool(arrays["geometry_valid"][local])
            result["geometry_invalid_reason"] = str(arrays["geometry_invalid_reason"][local])
        else:
            result["brics_groups"] = [
                np.asarray(arrays["brics_atom_index"][int(arrays["brics_group_atom_ptr"][group]):
                    int(arrays["brics_group_atom_ptr"][group + 1])], dtype=np.int32)
                for group in range(int(arrays["brics_sample_group_ptr"][local]),
                                   int(arrays["brics_sample_group_ptr"][local + 1]))
            ]
            result["fingerprint_packed"] = np.asarray(arrays["fingerprint_packed"][local])
        return result


class DualStaticCache:
    """Read a frozen chunked static artifact by cohort index."""
    def __init__(self, root, *, expected_format=STATIC_FORMAT,
                 parent_bundle_hash=None, cohort_manifest_hash=None,
                 ordered_keys_hash=None, chunk_cache_capacity=CHUNK_CACHE_CAPACITY):
        self.root = Path(root).resolve()
        manifest_path, frozen_path, keys_path = (
            self.root / "manifest.json", self.root / ".frozen", self.root / "sample_keys.npy"
        )
        if not all(path.is_file() for path in (manifest_path, frozen_path, keys_path)):
            raise CacheLifecycleError("static cache is incomplete: " + str(self.root))
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.manifest_hash = json_hash(self.manifest)
        frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
        if frozen != {"manifest_hash": self.manifest_hash}:
            raise CacheLifecycleError("static cache .frozen/manifest mismatch")
        if self.manifest.get("format") != expected_format:
            raise CacheLifecycleError("static cache format mismatch")
        for name, expected in (("parent_bundle_hash", parent_bundle_hash),
                               ("cohort_manifest_hash", cohort_manifest_hash),
                               ("ordered_sample_key_hash", ordered_keys_hash)):
            if expected is not None and self.manifest.get(name) != expected:
                raise CacheLifecycleError("static cache binding mismatch: " + name)
        self.sample_keys = np.load(keys_path, mmap_mode="r")
        if self.sample_keys.shape != (int(self.manifest["sample_count"]), 32):
            raise CacheLifecycleError("static cache sample key shape mismatch")
        observed_key_hash = ordered_key_hash(self.sample_keys)
        if observed_key_hash != self.manifest.get("ordered_sample_key_hash"):
            raise CacheLifecycleError("static cache ordered sample-key hash mismatch")
        if len(np.unique(self.sample_keys, axis=0)) != self.sample_keys.shape[0]:
            raise CacheLifecycleError("static cache contains duplicate sample keys")
        self._key_to_index = {bytes(row): index for index, row in enumerate(self.sample_keys)}
        expected_start = 0
        for item in self.manifest.get("chunks", []):
            start, count = int(item.get("start", -1)), int(item.get("count", -1))
            if start != expected_start or count < 0 or not (self.root / str(item.get("path", "")) / ".complete").is_file():
                raise CacheLifecycleError("static cache chunk coverage is incomplete")
            expected_start += count
        if expected_start != int(self.manifest["sample_count"]):
            raise CacheLifecycleError("static cache chunk count mismatch")
        self._reader = _ChunkReader(self.root, self.manifest,
                                    targets=self.manifest.get("format") == TARGET_FORMAT,
                                    cache_capacity=chunk_cache_capacity)

    def __len__(self):
        return int(self.manifest["sample_count"])

    def get(self, index):
        return self._reader.get(index)

    def index_for_key(self, key):
        raw = bytes(key)
        try:
            return int(self._key_to_index[raw])
        except KeyError as exc:
            raise CacheLifecycleError("static cache has no requested sample key") from exc

    def get_by_key(self, key):
        return self.get(self.index_for_key(key))

    def close(self):
        self._reader._cache.clear()
        self._key_to_index.clear()


class PretrainTargetsCache(DualStaticCache):
    def __init__(self, root, **kwargs):
        super().__init__(root, expected_format=TARGET_FORMAT, **kwargs)


def load_static_caches(static_root, target_root=None, **bindings):
    static = DualStaticCache(static_root, **bindings)
    target = PretrainTargetsCache(target_root, **bindings) if target_root else None
    return static, target


__all__ = [
    "CHUNK_CACHE_CAPACITY", "STATIC_FORMAT", "TARGET_FORMAT", "STATIC_FIELDS", "TARGET_FIELDS",
    "build_dual_static", "build_pretrain_target", "materialize_dual_geometry",
    "write_chunk", "DualStaticCache", "PretrainTargetsCache", "load_static_caches",
]
