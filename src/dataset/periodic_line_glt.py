"""Periodic chemical-bond line graph geometry for MTS-GLT-v1.

The builders in this module are pure: they consume the existing canonical
Topology and open-Trimer records and never mutate either cache.  A line token
is an undirected periodic chemical bond, canonicalised under endpoint reversal
and a global RU translation.  Bond type is deliberately returned only as a
masked-line label/QC field; model inputs are endpoint atom types and geometry.
"""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import torch

from .mts_star_rbf_v2 import prepare_topology, prepare_trimer


SIDECAR_SCHEMA = "mts-periodic-line-glt-v1"
BUILDER_VERSION = 1
MAX_ATOMIC_NUMBER = 100
NUM_BOND_TYPES = 6

ARRAY_DTYPES = {
    "sample_keys": np.dtype("u1"),
    "sample_token_offsets": np.dtype("<i8"),
    "sample_relation_offsets": np.dtype("<i8"),
    "graph_geometry_valid": np.dtype("<?"),
    "token_atom_a": np.dtype("<i2"),
    "token_atom_b": np.dtype("<i2"),
    "token_shift": np.dtype("<i2"),
    "token_endpoint_z_a": np.dtype("<i2"),
    "token_endpoint_z_b": np.dtype("<i2"),
    "token_bond_type": np.dtype("<i2"),
    "token_label": np.dtype("<i4"),
    "token_observation_distances": np.dtype("<f4"),
    "token_observation_count": np.dtype("u1"),
    "token_valid": np.dtype("<?"),
    "relation_source": np.dtype("<i4"),
    "relation_target": np.dtype("<i4"),
    "relation_center_atom": np.dtype("<i4"),
    "relation_multiplicity": np.dtype("<i2"),
    "relation_observation_angles": np.dtype("<f4"),
    "relation_observation_count": np.dtype("u1"),
    "relation_valid": np.dtype("<?"),
    "relation_is_fallback": np.dtype("<?"),
}
ARRAY_NAMES = tuple(ARRAY_DTYPES)


def canonical_line_token(u: int, q_u: int, v: int, q_v: int):
    """Return ``(atom_a, atom_b, relative_shift)`` for an undirected bond."""
    direct = (int(u), int(v), int(q_v) - int(q_u))
    inverse = (int(v), int(u), int(q_u) - int(q_v))
    return min(direct, inverse)


def masked_line_label(z_a: int, bond_type: int, z_b: int) -> int:
    """Compact identity for ``(min Z, bond type, max Z)``.

    The triangular endpoint-pair rank avoids a sparse 3-D classifier while
    preserving the full requested label identity.  Bond type never enters the
    encoder input.
    """
    left, right = sorted((int(z_a), int(z_b)))
    if not (0 <= left <= right <= MAX_ATOMIC_NUMBER):
        raise ValueError(f"atomic number outside masked-line vocabulary: {left}, {right}")
    bond = int(bond_type)
    if not (0 <= bond < NUM_BOND_TYPES):
        raise ValueError(f"bond type outside masked-line vocabulary: {bond}")
    pair_rank = right * (right + 1) // 2 + left
    return pair_rank * NUM_BOND_TYPES + bond


def _canonical_to_base(topology, atom: int) -> int:
    mapping = torch.as_tensor(topology.canonical_to_trimer_base_atom_id).long()
    return int(mapping[int(atom)])


def _local(trimer_info, topology, atom: int, shift: int):
    return trimer_info["state_to_local"].get(
        (_canonical_to_base(topology, atom), int(shift))
    )


def _bond_type_for_instance(trimer_info, topology, u, q_u, v, q_v):
    left = _local(trimer_info, topology, u, q_u)
    right = _local(trimer_info, topology, v, q_v)
    if left is None or right is None:
        return None
    values = trimer_info["bonds"].get((min(left, right), max(left, right)))
    if values is None or len(values) != 1:
        return None
    return next(iter(values))


def _distance(trimer_info, topology, u, q_u, v, q_v):
    left = _local(trimer_info, topology, u, q_u)
    right = _local(trimer_info, topology, v, q_v)
    if left is None or right is None:
        return None
    value = torch.linalg.vector_norm(
        trimer_info["positions"][left] - trimer_info["positions"][right]
    )
    if not bool(torch.isfinite(value)) or float(value) <= 0.0:
        return None
    return float(value)


def _angle(trimer_info, topology, u, q_u, center, q_c, w, q_w):
    indices = [
        _local(trimer_info, topology, u, q_u),
        _local(trimer_info, topology, center, q_c),
        _local(trimer_info, topology, w, q_w),
    ]
    if any(index is None for index in indices):
        return None
    p_u, p_c, p_w = (trimer_info["positions"][index] for index in indices)
    first, second = p_u - p_c, p_w - p_c
    denominator = torch.linalg.vector_norm(first) * torch.linalg.vector_norm(second)
    if not bool(torch.isfinite(denominator)) or float(denominator) <= 0.0:
        return None
    cosine = torch.clamp(torch.dot(first, second) / denominator, -1.0, 1.0)
    value = torch.acos(cosine)
    if not bool(torch.isfinite(value)):
        return None
    return float(value)


def _internal_bonds(topology):
    edges = torch.as_tensor(topology.ru_edge_index).long()
    types = torch.as_tensor(topology.ru_bond_type).long().reshape(-1)
    if edges.ndim != 2 or edges.size(0) != 2 or edges.size(1) != types.numel():
        raise ValueError("invalid RU chemical-bond table")
    observed = {}
    for column in range(edges.size(1)):
        u, v = int(edges[0, column]), int(edges[1, column])
        if u == v:
            raise ValueError("self chemical bond is invalid")
        key = (min(u, v), max(u, v))
        observed.setdefault(key, set()).add(int(types[column]))
    result = []
    for (u, v), values in sorted(observed.items()):
        if len(values) != 1:
            raise ValueError(f"conflicting RU bond types for {(u, v)}")
        result.append((u, 0, v, 0, next(iter(values))))
    return result


def _token_instances(token):
    u, v, shift = token
    return [
        (u, translation, v, shift + translation)
        for translation in range(-1, 2)
        if -1 <= translation <= 1 and -1 <= shift + translation <= 1
    ]


def build_periodic_line_sample(key: bytes, topology, trimer):
    """Build one periodic bond-token graph and its Trimer observations."""
    prepared, topology_reason = prepare_topology(topology)
    if prepared is None or topology_reason is not None:
        return {
            "sample_key": bytes(key), "geometry_valid": False,
            "tokens": [], "relations": [],
        }

    z = prepared["z"]
    bond_specs = _internal_bonds(topology)
    # The only inter-RU chemical bond in the linear two-attachment contract.
    bond_specs.append((prepared["right"], 0, prepared["left"], 1, None))
    token_specs = {}
    for u, q_u, v, q_v, bond_type in bond_specs:
        token = canonical_line_token(u, q_u, v, q_v)
        if token in token_specs:
            raise ValueError(f"duplicate periodic line token: {token}")
        token_specs[token] = bond_type

    trimer_info, _ = prepare_trimer(trimer, topology)
    geometry_valid = bool(trimer_info is not None)
    tokens = []
    token_index = {token: index for index, token in enumerate(sorted(token_specs))}
    for token in sorted(token_specs):
        u, v, shift = token
        instances = _token_instances(token)
        expected_count = 3 if shift == 0 else 2
        if len(instances) != expected_count:
            raise AssertionError("periodic line observation count contract failed")
        instance_types = set()
        missing_physical_bond = False
        distances = []
        if geometry_valid:
            for a, q_a, b, q_b in instances:
                observed_type = _bond_type_for_instance(
                    trimer_info, topology, a, q_a, b, q_b
                )
                if observed_type is not None:
                    instance_types.add(int(observed_type))
                else:
                    missing_physical_bond = True
                distance = _distance(trimer_info, topology, a, q_a, b, q_b)
                if distance is not None:
                    distances.append(distance)
        stored_type = token_specs[token]
        if geometry_valid and missing_physical_bond:
            raise ValueError(f"missing physical Trimer bond instance: {token}")
        if stored_type is not None:
            # Topology uses its own normalized aromatic-bond vocabulary while
            # the Trimer retains RDKit's raw aromatic code.  Internal labels
            # therefore come from Topology; Trimer only proves the bond and
            # supplies coordinates.  This field remains label/QC-only.
            bond_type = int(stored_type)
        else:
            if len(instance_types) != 1:
                if geometry_valid:
                    raise ValueError(f"conflicting periodic boundary bond type: {token}")
                instance_types = {0}
            bond_type = next(iter(instance_types))
        token_valid = geometry_valid and len(distances) == expected_count
        padded = (distances + [0.0, 0.0, 0.0])[:3]
        z_a, z_b = int(z[u]), int(z[v])
        tokens.append({
            "key": token,
            "atom_a": u,
            "atom_b": v,
            "shift": shift,
            "z_a": z_a,
            "z_b": z_b,
            "bond_type": bond_type,
            "label": masked_line_label(z_a, bond_type, z_b),
            "distances": padded,
            "observation_count": len(distances),
            "valid": token_valid,
        })

    # Enumerate actual bond instances in the open Trimer.  Group translation-
    # equivalent angles by canonical token endpoints and centre atom identity.
    incident = defaultdict(list)
    for token in sorted(token_specs):
        for u, q_u, v, q_v in _token_instances(token):
            token_id = token_index[token]
            incident[(u, q_u)].append((token_id, v, q_v))
            incident[(v, q_v)].append((token_id, u, q_u))

    grouped = defaultdict(list)
    for (center, q_c), members in incident.items():
        for left in range(len(members)):
            for right in range(left + 1, len(members)):
                src, outer_a, q_a = members[left]
                dst, outer_b, q_b = members[right]
                # Normalise the physical angle under a global RU translation.
                normalized = (outer_a, q_a - q_c, center, outer_b, q_b - q_c)
                inverse = (outer_b, q_b - q_c, center, outer_a, q_a - q_c)
                geometry_key = min(normalized, inverse)
                relation_key = (min(src, dst), max(src, dst), geometry_key)
                value = None
                if geometry_valid:
                    value = _angle(
                        trimer_info, topology,
                        outer_a, q_a, center, q_c, outer_b, q_b,
                    )
                grouped[relation_key].append(value)

    relations = []
    real_neighbors = set()
    for (low, high, geometry_key), observations in sorted(grouped.items()):
        valid_values = [value for value in observations if value is not None]
        span = max(geometry_key[1], 0, geometry_key[4]) - min(
            geometry_key[1], 0, geometry_key[4]
        )
        expected_count = 3 - span
        if not (0 <= span <= 2) or len(observations) != expected_count:
            raise ValueError(
                f"periodic angle observation multiplicity mismatch: "
                f"span={span}, expected={expected_count}, observed={len(observations)}"
            )
        padded = (valid_values + [0.0, 0.0, 0.0])[:3]
        valid = geometry_valid and len(valid_values) == expected_count
        directions = [(low, high)] if low == high else [(low, high), (high, low)]
        for source, target in directions:
            relations.append({
                "source": source,
                "target": target,
                "center_atom": int(geometry_key[2]),
                "multiplicity": expected_count,
                "angles": padded,
                "observation_count": len(valid_values),
                "valid": valid,
                "fallback": False,
            })
            real_neighbors.add(source)

    for token_id in range(len(tokens)):
        if token_id not in real_neighbors:
            relations.append({
                "source": token_id, "target": token_id,
                "center_atom": -1, "multiplicity": 1,
                "angles": [0.0, 0.0, 0.0], "observation_count": 0,
                "valid": False, "fallback": True,
            })
    relations.sort(key=lambda item: (
        item["target"], item["source"], item["center_atom"], item["fallback"]
    ))
    return {
        "sample_key": bytes(key),
        "geometry_valid": geometry_valid and all(token["valid"] for token in tokens),
        "tokens": tokens,
        "relations": relations,
    }


class PeriodicLineGLTSidecar:
    """Lightweight read-only mmap reader for the GLT sidecar."""

    def __init__(self, root):
        self.root = Path(root).resolve()
        metadata_path = self.root / "metadata.json"
        if not metadata_path.is_file() or not (self.root / ".done").is_file():
            raise RuntimeError(f"periodic line GLT sidecar is incomplete: {self.root}")
        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if self.metadata.get("schema") != SIDECAR_SCHEMA:
            raise RuntimeError("unsupported periodic line GLT sidecar schema")
        if int(self.metadata.get("builder_version", -1)) != BUILDER_VERSION:
            raise RuntimeError("periodic line GLT builder version mismatch")
        if set(self.metadata.get("arrays", {})) != set(ARRAY_NAMES):
            raise RuntimeError("periodic line GLT array manifest mismatch")
        self.arrays = {}
        for name in ARRAY_NAMES:
            value = np.load(self.root / f"{name}.npy", mmap_mode="r", allow_pickle=False)
            spec = self.metadata["arrays"][name]
            if list(value.shape) != list(spec["shape"]) or np.dtype(value.dtype).str != spec["dtype"]:
                raise RuntimeError(f"periodic line GLT shape/dtype mismatch: {name}")
            self.arrays[name] = value
        count = int(self.arrays["sample_keys"].shape[0])
        if self.arrays["sample_keys"].shape != (count, 32):
            raise RuntimeError("periodic line GLT sample-key shape mismatch")
        self._key_to_row = None

    def __len__(self):
        return int(self.arrays["sample_keys"].shape[0])

    def index_for_key(self, key, row_hint=None):
        raw = np.frombuffer(bytes(key), dtype=np.uint8)
        if row_hint is not None and 0 <= int(row_hint) < len(self):
            if np.array_equal(self.arrays["sample_keys"][int(row_hint)], raw):
                return int(row_hint)
        if self._key_to_row is None:
            self._key_to_row = {
                bytes(row): index for index, row in enumerate(self.arrays["sample_keys"])
            }
        if bytes(key) not in self._key_to_row:
            raise KeyError("sample key absent from periodic line GLT sidecar")
        return self._key_to_row[bytes(key)]

    def model_row(self, index):
        index = int(index)
        ts, te = (int(self.arrays["sample_token_offsets"][index + i]) for i in (0, 1))
        rs, re = (int(self.arrays["sample_relation_offsets"][index + i]) for i in (0, 1))
        token_names = (
            "token_atom_a", "token_atom_b", "token_shift",
            "token_endpoint_z_a", "token_endpoint_z_b", "token_label",
            "token_observation_distances", "token_observation_count", "token_valid",
        )
        relation_names = (
            "relation_source", "relation_target", "relation_center_atom",
            "relation_multiplicity", "relation_observation_angles",
            "relation_observation_count", "relation_valid", "relation_is_fallback",
        )
        return {
            "geometry_valid": bool(self.arrays["graph_geometry_valid"][index]),
            "tokens": {name: np.asarray(self.arrays[name][ts:te]) for name in token_names},
            "relations": {name: np.asarray(self.arrays[name][rs:re]) for name in relation_names},
        }

    def qc_row(self, index):
        row = self.model_row(index)
        ts, te = (int(self.arrays["sample_token_offsets"][int(index) + i]) for i in (0, 1))
        row["tokens"]["token_bond_type"] = np.asarray(
            self.arrays["token_bond_type"][ts:te]
        )
        return row


__all__ = [
    "ARRAY_DTYPES", "ARRAY_NAMES", "BUILDER_VERSION", "SIDECAR_SCHEMA",
    "NUM_BOND_TYPES", "MAX_ATOMIC_NUMBER", "PeriodicLineGLTSidecar",
    "build_periodic_line_sample", "canonical_line_token", "masked_line_label",
]
