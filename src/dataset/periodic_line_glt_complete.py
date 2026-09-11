"""Complete-Trimer GLT records with Galformer-style bond chemistry.

This module is an independent sidecar contract for the complete finite Trimer
branch.  It consumes the already-frozen Trimer coordinates and the canonical
O8 topology; it never generates coordinates and never mutates either cache.
Each physical Trimer bond is one token (normally ``3N+2`` tokens), while
relations are the directed one-hop line-graph edges induced by a shared
physical atom.
"""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem

from .graph_data import build_periodic_multimer_mol
from .glt_bond_chemistry import (
    bond_feature_vector, bond_type_index, bond_stereo_index,
    BOND_FEATURE_DIM, STEREO_VALUES, STEREO_TO_INDEX, STEREO_UNKNOWN,
NUM_STEREO_TYPES,
)


SIDECAR_SCHEMA = "mts-periodic-line-glt-complete-v1"
BUILDER_VERSION = 1
MAX_ATOMIC_NUMBER = 100
ELEMENT_CLASSES = 101  # Z=1..100 plus unknown

BOND_TYPE_SINGLE = 0
BOND_TYPE_DOUBLE = 1
BOND_TYPE_TRIPLE = 2
BOND_TYPE_AROMATIC = 3
BOND_TYPE_UNKNOWN = 4
NUM_BOND_TYPES = 5


def _bond_chemistry(smiles: str):
    """Read chemistry from the same open three-RU RDKit molecule as Trimer."""

    molecule, metadata = build_periodic_multimer_mol(
        smiles, num_repeat_units=3, close_periodic=False
    )
    base_count = int(metadata["base_atom_count"])
    chemistry = {}
    for bond in molecule.GetBonds():
        a = int(bond.GetBeginAtomIdx())
        b = int(bond.GetEndAtomIdx())
        state_a = (a % base_count, a // base_count - 1)
        state_b = (b % base_count, b // base_count - 1)
        key = tuple(sorted((state_a, state_b)))
        chemistry[key] = {
            "bond_type": bond_type_index(bond),
            "stereo": bond_stereo_index(bond),
            "conjugated": int(bool(bond.GetIsConjugated())),
            "ring": int(bool(bond.IsInRing())),
            "features": bond_feature_vector(bond),
        }
    return chemistry


def _angle(positions: torch.Tensor, outer_a: int, center: int, outer_b: int):
    first = positions[int(outer_a)] - positions[int(center)]
    second = positions[int(outer_b)] - positions[int(center)]
    denominator = torch.linalg.vector_norm(first) * torch.linalg.vector_norm(second)
    if not bool(torch.isfinite(denominator)) or float(denominator) <= 0.0:
        return None
    cosine = torch.clamp(torch.dot(first, second) / denominator, -1.0, 1.0)
    value = torch.acos(cosine)
    return float(value) if bool(torch.isfinite(value)) else None


def _empty_tokens():
    return {
        "token_atom_a": np.empty(0, dtype=np.int64),
        "token_atom_b": np.empty(0, dtype=np.int64),
        "token_shift": np.empty(0, dtype=np.int16),
        "token_endpoint_z_a": np.empty(0, dtype=np.int16),
        "token_endpoint_z_b": np.empty(0, dtype=np.int16),
        "token_distance": np.empty(0, dtype=np.float32),
        "token_bond_type": np.empty(0, dtype=np.int8),
        "token_stereo": np.empty(0, dtype=np.int8),
        "token_conjugated": np.empty(0, dtype=np.int8),
        "token_ring": np.empty(0, dtype=np.int8),
        "token_bond_features": np.empty((0, BOND_FEATURE_DIM), dtype=np.float32),
        "token_anchor_q_a": np.empty(0, dtype=np.int8),
        "token_anchor_q_b": np.empty(0, dtype=np.int8),
        "token_valid": np.empty(0, dtype=bool),
        "token_center_internal": np.empty(0, dtype=bool),
    }


def _empty_relations():
    return {
        "relation_source": np.empty(0, dtype=np.int32),
        "relation_target": np.empty(0, dtype=np.int32),
        "relation_center_atom": np.empty(0, dtype=np.int32),
        "relation_source_image_shift": np.empty(0, dtype=np.int16),
        "relation_angle": np.empty(0, dtype=np.float32),
        "relation_valid": np.empty(0, dtype=bool),
    }


def empty_complete_trimer_row(reason: str = "invalid"):
    return {
        "schema": SIDECAR_SCHEMA,
        "geometry_valid": False,
        "invalid_reason": str(reason),
        "tokens": _empty_tokens(),
        "relations": _empty_relations(),
    }


def build_complete_trimer_glt_sample(topology, trimer, smiles: str):
    """Build one complete-Trimer row from frozen coordinates and topology.

    The row contains every unique physical edge in ``trimer_edge_index``.  A
    directed edge table is accepted (as produced by the cache) and collapsed
    only at the physical-bond level; distinct periodic copies remain distinct
    tokens.  Chemistry is reparsed from the open Trimer RDKit graph and is
    checked against the frozen bond type code.
    """

    if not bool(getattr(trimer, "trimer_geometry_valid", False)):
        return empty_complete_trimer_row("geometry_invalid")
    if not bool(getattr(trimer, "trimer_geometry_is_3d", False)):
        return empty_complete_trimer_row("non_3d_geometry")
    if bool(getattr(trimer, "trimer_2d_fallback", False)):
        return empty_complete_trimer_row("2d_fallback")

    raw_positions = getattr(trimer, "trimer_pos", None)
    raw_atomic = getattr(trimer, "trimer_atomic_number", None)
    raw_base = getattr(
        trimer, "trimer_base_ru_atom_id",
        getattr(trimer, "trimer_base_ru_atom_index", None),
    )
    raw_offsets = getattr(trimer, "trimer_ru_offset", None)
    raw_edge = getattr(trimer, "trimer_edge_index", None)
    raw_bond_codes = getattr(trimer, "trimer_bond_type", None)
    raw_mapping = getattr(topology, "canonical_to_trimer_base_atom_id", None)
    raw_topology_z = getattr(
        topology, "atomic_numbers", getattr(topology, "z", None)
    )
    if any(value is None for value in (
        raw_positions, raw_atomic, raw_base, raw_offsets, raw_edge,
        raw_bond_codes, raw_mapping, raw_topology_z,
    )):
        return empty_complete_trimer_row("trimer_identity_missing")
    positions = torch.as_tensor(raw_positions).float()
    atomic = torch.as_tensor(raw_atomic).long()
    base = torch.as_tensor(raw_base).long()
    offsets = torch.as_tensor(raw_offsets).long()
    edge = torch.as_tensor(raw_edge).long()
    bond_codes = torch.as_tensor(raw_bond_codes).long().reshape(-1)
    mapping = torch.as_tensor(raw_mapping).long().reshape(-1)
    topology_z = torch.as_tensor(raw_topology_z).long().reshape(-1)

    if (
        positions.ndim != 2 or positions.size(1) != 3
        or not bool(torch.isfinite(positions).all())
        or atomic.ndim != 1 or atomic.numel() != positions.size(0)
        or base.ndim != 1 or base.numel() != positions.size(0)
        or offsets.ndim != 1 or offsets.numel() != positions.size(0)
        or edge.ndim != 2 or edge.size(0) != 2
        or bond_codes.numel() != edge.size(1)
        or mapping.ndim != 1 or topology_z.ndim != 1
        or mapping.numel() != topology_z.numel()
    ):
        return empty_complete_trimer_row("trimer_identity_missing")
    if edge.numel() and (
        int(edge.min()) < 0 or int(edge.max()) >= positions.size(0)
    ):
        return empty_complete_trimer_row("real_bond_graph_invalid")

    base_to_canonical = {}
    for canonical_id, base_id in enumerate(mapping.tolist()):
        base_id = int(base_id)
        if base_id in base_to_canonical:
            return empty_complete_trimer_row("canonical_base_duplicate")
        base_to_canonical[base_id] = int(canonical_id)
        state = torch.nonzero(
            (base == base_id) & (offsets == 0), as_tuple=False
        ).flatten()
        if state.numel() != 1 or int(atomic[state[0]]) != int(topology_z[canonical_id]):
            return empty_complete_trimer_row("canonical_trimer_mapping_mismatch")

    try:
        chemistry = _bond_chemistry(str(smiles))
    except Exception as exc:
        return empty_complete_trimer_row(f"bond_chemistry_parse:{type(exc).__name__}")

    edge_codes = {}
    for column in range(int(edge.size(1))):
        left, right = (int(edge[0, column]), int(edge[1, column]))
        if left == right:
            return empty_complete_trimer_row("real_bond_self_loop")
        key = (min(left, right), max(left, right))
        code = int(bond_codes[column])
        if key in edge_codes and edge_codes[key] != code:
            return empty_complete_trimer_row("real_bond_type_conflict")
        edge_codes[key] = code

    records = []
    for (local_a, local_b), frozen_code in sorted(edge_codes.items()):
        base_a, base_b = int(base[local_a]), int(base[local_b])
        if base_a not in base_to_canonical or base_b not in base_to_canonical:
            return empty_complete_trimer_row("canonical_endpoint_missing")
        canonical_a, canonical_b = base_to_canonical[base_a], base_to_canonical[base_b]
        q_a, q_b = int(offsets[local_a]), int(offsets[local_b])
        key = tuple(sorted(((base_a, q_a), (base_b, q_b))))
        attrs = chemistry.get(key)
        if attrs is None:
            return empty_complete_trimer_row("bond_chemistry_missing")
        if int(frozen_code) - 1 != int(attrs["bond_type"]):
            return empty_complete_trimer_row("bond_type_mapping_mismatch")
        if (canonical_b, q_b, local_b) < (canonical_a, q_a, local_a):
            local_a, local_b = local_b, local_a
            base_a, base_b = base_b, base_a
            canonical_a, canonical_b = canonical_b, canonical_a
            q_a, q_b = q_b, q_a
        distance = torch.linalg.vector_norm(positions[local_a] - positions[local_b])
        if not bool(torch.isfinite(distance)) or float(distance) <= 0.0:
            return empty_complete_trimer_row("bond_distance_invalid")
        records.append({
            "local_a": local_a, "local_b": local_b,
            "atom_a": canonical_a, "atom_b": canonical_b,
            "q_a": q_a, "q_b": q_b,
            "z_a": int(atomic[local_a]), "z_b": int(atomic[local_b]),
            "distance": float(distance), "attrs": attrs,
            "center": q_a == 0 and q_b == 0,
        })

    records.sort(key=lambda row: (
        not row["center"], row["atom_a"], row["atom_b"],
        row["q_a"], row["q_b"], row["local_a"], row["local_b"],
    ))
    token_by_local = {}
    for token_id, row in enumerate(records):
        token_by_local[row["local_a"]] = token_by_local.get(row["local_a"], []) + [(token_id, row["local_b"])]
        token_by_local[row["local_b"]] = token_by_local.get(row["local_b"], []) + [(token_id, row["local_a"])]

    token_output = {
        "token_atom_a": np.asarray([row["atom_a"] for row in records], dtype=np.int64),
        "token_atom_b": np.asarray([row["atom_b"] for row in records], dtype=np.int64),
        "token_shift": np.asarray([row["q_b"] - row["q_a"] for row in records], dtype=np.int16),
        "token_endpoint_z_a": np.asarray([row["z_a"] if 1 <= row["z_a"] <= MAX_ATOMIC_NUMBER else 0 for row in records], dtype=np.int16),
        "token_endpoint_z_b": np.asarray([row["z_b"] if 1 <= row["z_b"] <= MAX_ATOMIC_NUMBER else 0 for row in records], dtype=np.int16),
        "token_distance": np.asarray([row["distance"] for row in records], dtype=np.float32),
        "token_bond_type": np.asarray([row["attrs"]["bond_type"] for row in records], dtype=np.int8),
        "token_stereo": np.asarray([row["attrs"]["stereo"] for row in records], dtype=np.int8),
        "token_conjugated": np.asarray([row["attrs"]["conjugated"] for row in records], dtype=np.int8),
        "token_ring": np.asarray([row["attrs"]["ring"] for row in records], dtype=np.int8),
        "token_bond_features": np.asarray([row["attrs"]["features"] for row in records], dtype=np.float32).reshape(-1, BOND_FEATURE_DIM),
        "token_anchor_q_a": np.asarray([row["q_a"] for row in records], dtype=np.int8),
        "token_anchor_q_b": np.asarray([row["q_b"] for row in records], dtype=np.int8),
        "token_valid": np.ones(len(records), dtype=bool),
        "token_center_internal": np.asarray([row["center"] for row in records], dtype=bool),
    }

    relation_rows = []
    seen = set()
    for center_local, incident in sorted(token_by_local.items()):
        center_base = int(base[center_local])
        center_canonical = base_to_canonical.get(center_base)
        if center_canonical is None:
            return empty_complete_trimer_row("relation_center_mapping_missing")
        center_q = int(offsets[center_local])
        for source_id, source_outer in incident:
            for target_id, target_outer in incident:
                if source_id == target_id:
                    continue
                key = (int(source_id), int(target_id), int(center_local))
                if key in seen:
                    continue
                seen.add(key)
                angle = _angle(positions, source_outer, center_local, target_outer)
                relation_rows.append({
                    "relation_source": int(source_id),
                    "relation_target": int(target_id),
                    "relation_center_atom": int(center_canonical),
                    "relation_source_image_shift": int(offsets[source_outer]) - center_q,
                    "relation_angle": 0.0 if angle is None else angle,
                    "relation_valid": angle is not None,
                })
    relation_output = {
        name: np.asarray([row[name] for row in relation_rows], dtype=(
            np.float32 if name == "relation_angle" else
            bool if name == "relation_valid" else
            np.int16 if name == "relation_source_image_shift" else np.int32
        ))
        for name in _empty_relations()
    }
    if not relation_rows:
        relation_output = _empty_relations()
    return {
        "schema": SIDECAR_SCHEMA,
        "geometry_valid": bool(records) and bool(token_output["token_valid"].all()),
        "invalid_reason": "",
        "tokens": token_output,
        "relations": relation_output,
    }


class CompleteTrimerGLTSidecar:
    """Read a packed complete-Trimer sidecar without copying its arrays."""

    def __init__(self, root, *, build_key_index=True):
        self.root = Path(root)
        self.metadata = json.loads((self.root / "metadata.json").read_text())
        if self.metadata.get("schema") != SIDECAR_SCHEMA:
            raise ValueError("complete Trimer GLT sidecar schema mismatch")
        self.sample_keys = np.load(self.root / "sample_keys.npy", mmap_mode="r")
        self.token_offsets = np.load(self.root / "token_offsets.npy", mmap_mode="r")
        self.relation_offsets = np.load(self.root / "relation_offsets.npy", mmap_mode="r")
        self.geometry_valid = np.load(self.root / "geometry_valid.npy", mmap_mode="r")
        if self.sample_keys.ndim != 2 or self.sample_keys.shape[1] != 32:
            raise ValueError("complete-Trimer sidecar sample keys must be [N,32]")
        if (
            self.token_offsets.ndim != 1
            or self.relation_offsets.ndim != 1
            or self.geometry_valid.ndim != 1
            or self.token_offsets.size != len(self.sample_keys) + 1
            or self.relation_offsets.size != len(self.sample_keys) + 1
            or self.geometry_valid.size != len(self.sample_keys)
        ):
            raise ValueError("complete-Trimer sidecar row offsets are inconsistent")
        self._arrays = {
            path.stem: np.load(path, mmap_mode="r")
            for path in self.root.glob("*.npy")
            if path.stem not in {"sample_keys", "token_offsets", "relation_offsets", "geometry_valid"}
        }
        required = set(_empty_tokens()) | set(_empty_relations())
        missing = sorted(required - set(self._arrays))
        if missing:
            raise ValueError(
                "complete-Trimer sidecar is missing required fields: "
                + ",".join(missing)
            )
        if self.metadata.get("bond_feature_dim") != BOND_FEATURE_DIM:
            raise ValueError("complete-Trimer sidecar bond feature dimension mismatch")
        self._key_to_index = (
            {bytes(row): index for index, row in enumerate(self.sample_keys)}
            if build_key_index else None
        )

    def __len__(self):
        return int(self.sample_keys.shape[0])

    def index_for_key(self, key, row_hint=None):
        key = bytes(key)
        if row_hint is not None and bytes(self.sample_keys[int(row_hint)]) == key:
            return int(row_hint)
        if self._key_to_index is None:
            raise KeyError("sidecar key index disabled and row_hint did not match")
        return self._key_to_index[key]

    def model_row(self, index):
        index = int(index)
        token_start, token_end = self.token_offsets[index:index + 2]
        relation_start, relation_end = self.relation_offsets[index:index + 2]
        tokens = {
            name: value[int(token_start):int(token_end)]
            for name, value in self._arrays.items() if name.startswith("token_")
        }
        relations = {
            name: value[int(relation_start):int(relation_end)]
            for name, value in self._arrays.items() if name.startswith("relation_")
        }
        return {
            "geometry_valid": bool(self.geometry_valid[index]),
            "tokens": tokens, "relations": relations,
        }


def write_complete_trimer_sidecar(root, sample_keys, rows):
    """Write a packed complete-Trimer sidecar and refuse existing targets."""

    root = Path(root)
    if root.exists():
        raise FileExistsError(f"refusing to overwrite sidecar: {root}")
    sample_keys = list(sample_keys)
    rows = list(rows)
    if len(sample_keys) != len(rows):
        raise ValueError("complete-Trimer sidecar keys/rows length mismatch")
    root.mkdir(parents=True)
    token_offsets, relation_offsets = [0], [0]
    for row in rows:
        token_offsets.append(token_offsets[-1] + len(row["tokens"]["token_atom_a"]))
        relation_offsets.append(relation_offsets[-1] + len(row["relations"]["relation_source"]))
    key_rows = [np.frombuffer(bytes(key), dtype=np.uint8) for key in sample_keys]
    if key_rows:
        key_array = np.stack(key_rows, axis=0).astype(np.uint8, copy=False)
    else:
        key_array = np.empty((0, 32), dtype=np.uint8)
    if key_array.ndim != 2 or key_array.shape[1] != 32:
        raise ValueError("complete-Trimer sidecar keys must be 32-byte rows")
    np.save(root / "sample_keys.npy", key_array)
    np.save(root / "token_offsets.npy", np.asarray(token_offsets, dtype=np.int64))
    np.save(root / "relation_offsets.npy", np.asarray(relation_offsets, dtype=np.int64))
    np.save(root / "geometry_valid.npy", np.asarray([row["geometry_valid"] for row in rows], dtype=bool))
    token_names = tuple(_empty_tokens())
    relation_names = tuple(_empty_relations())
    for name in token_names:
        values = [np.asarray(row["tokens"][name]) for row in rows]
        np.save(root / f"{name}.npy", np.concatenate(values, axis=0) if values else _empty_tokens()[name])
    for name in relation_names:
        values = [np.asarray(row["relations"][name]) for row in rows]
        np.save(root / f"{name}.npy", np.concatenate(values, axis=0) if values else _empty_relations()[name])
    (root / "metadata.json").write_text(json.dumps({
        "schema": SIDECAR_SCHEMA,
        "builder_version": BUILDER_VERSION,
        "sample_count": len(rows),
        "bond_feature_dim": BOND_FEATURE_DIM,
        "geometry_semantics": "complete_trimer_physical_bonds",
        "endpoint_element_classes": ELEMENT_CLASSES,
        "stereo_classes": NUM_STEREO_TYPES,
    }, indent=2, sort_keys=True) + "\n")
    (root / ".done").write_text("complete\n")


__all__ = [
    "SIDECAR_SCHEMA", "BUILDER_VERSION", "BOND_FEATURE_DIM",
    "ELEMENT_CLASSES", "MAX_ATOMIC_NUMBER", "BOND_TYPE_UNKNOWN",
    "STEREO_UNKNOWN", "bond_type_index", "bond_stereo_index",
    "bond_feature_vector", "empty_complete_trimer_row",
    "build_complete_trimer_glt_sample", "CompleteTrimerGLTSidecar",
    "write_complete_trimer_sidecar",
]
