"""Central-RU periodic line geometry for MTS-GLT-GraphGate-v1.

This module is independent from the frozen GLT-v1 sidecar.  Observation slots
retain their RU identity; runtime validity is derived from the central policy
instead of inheriting the legacy graph-level geometry flag.
"""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import torch

from .mts_star_rbf_v2 import prepare_topology, prepare_trimer
from .periodic_line_glt import (
    MAX_ATOMIC_NUMBER,
    NUM_BOND_TYPES,
    canonical_line_token,
    masked_line_label,
)


SIDECAR_SCHEMA = "mts-periodic-line-glt-central-v1"
BUILDER_VERSION = 1

ARRAY_DTYPES = {
    "sample_keys": np.dtype("u1"),
    "sample_token_offsets": np.dtype("<i8"),
    "sample_relation_offsets": np.dtype("<i8"),
    "graph_base_mapping_valid": np.dtype("<?"),
    "graph_runtime_valid": np.dtype("<?"),
    "graph_invalid_reason": np.dtype("u1"),
    "token_atom_a": np.dtype("<i2"),
    "token_atom_b": np.dtype("<i2"),
    "token_shift": np.dtype("<i2"),
    "token_endpoint_z_a": np.dtype("<i2"),
    "token_endpoint_z_b": np.dtype("<i2"),
    "token_bond_type": np.dtype("<i2"),
    "token_label": np.dtype("<i4"),
    "token_observation_distances": np.dtype("<f4"),
    "token_observation_valid": np.dtype("<?"),
    "token_observation_translation": np.dtype("i1"),
    "token_observation_q_a": np.dtype("i1"),
    "token_observation_q_b": np.dtype("i1"),
    "token_observation_count": np.dtype("u1"),
    "token_runtime_valid": np.dtype("<?"),
    "relation_source": np.dtype("<i4"),
    "relation_target": np.dtype("<i4"),
    "relation_center_atom": np.dtype("<i4"),
    "relation_outer_offset_a": np.dtype("i1"),
    "relation_outer_offset_b": np.dtype("i1"),
    "relation_span": np.dtype("u1"),
    "relation_multiplicity": np.dtype("<i2"),
    "relation_observation_angles": np.dtype("<f4"),
    "relation_observation_valid": np.dtype("<?"),
    "relation_observation_translation": np.dtype("i1"),
    "relation_observation_count": np.dtype("u1"),
    "relation_runtime_valid": np.dtype("<?"),
    "relation_is_fallback": np.dtype("<?"),
}
ARRAY_NAMES = tuple(ARRAY_DTYPES)

INVALID_NONE = 0
INVALID_BASE_MAPPING = 1
INVALID_DISTANCE = 2
INVALID_ANGLE = 3


def _canonical_to_base(topology, atom: int) -> int:
    mapping = torch.as_tensor(topology.canonical_to_trimer_base_atom_id).long()
    return int(mapping[int(atom)])


def _local(trimer_info, topology, atom: int, shift: int):
    return trimer_info["state_to_local"].get(
        (_canonical_to_base(topology, atom), int(shift))
    )


def _bond_type(trimer_info, topology, u, q_u, v, q_v):
    a, b = _local(trimer_info, topology, u, q_u), _local(trimer_info, topology, v, q_v)
    if a is None or b is None:
        return None
    values = trimer_info["bonds"].get((min(a, b), max(a, b)))
    return next(iter(values)) if values is not None and len(values) == 1 else None


def _distance(trimer_info, topology, u, q_u, v, q_v):
    a, b = _local(trimer_info, topology, u, q_u), _local(trimer_info, topology, v, q_v)
    if a is None or b is None:
        return None
    value = torch.linalg.vector_norm(trimer_info["positions"][a] - trimer_info["positions"][b])
    return float(value) if bool(torch.isfinite(value)) and float(value) > 0 else None


def _angle(trimer_info, topology, u, q_u, center, q_c, w, q_w):
    indices = [_local(trimer_info, topology, atom, shift) for atom, shift in (
        (u, q_u), (center, q_c), (w, q_w)
    )]
    if any(index is None for index in indices):
        return None
    p_u, p_c, p_w = (trimer_info["positions"][index] for index in indices)
    first, second = p_u - p_c, p_w - p_c
    denominator = torch.linalg.vector_norm(first) * torch.linalg.vector_norm(second)
    if not bool(torch.isfinite(denominator)) or float(denominator) <= 0:
        return None
    value = torch.acos(torch.clamp(torch.dot(first, second) / denominator, -1.0, 1.0))
    return float(value) if bool(torch.isfinite(value)) else None


def _internal_bonds(topology):
    edges = torch.as_tensor(topology.ru_edge_index).long()
    types = torch.as_tensor(topology.ru_bond_type).long().reshape(-1)
    observed = defaultdict(set)
    for column in range(edges.size(1)):
        u, v = int(edges[0, column]), int(edges[1, column])
        if u == v:
            raise ValueError("self chemical bond is invalid")
        observed[(min(u, v), max(u, v))].add(int(types[column]))
    result = []
    for (u, v), values in sorted(observed.items()):
        if len(values) != 1:
            raise ValueError(f"conflicting RU bond types for {(u, v)}")
        result.append((u, 0, v, 0, next(iter(values))))
    return result


def _instances(token):
    u, v, shift = token
    if abs(int(shift)) > 1:
        raise ValueError(f"real line token crosses more than one RU: {token}")
    return [
        (u, translation, v, shift + translation, translation)
        for translation in range(-1, 2)
        if -1 <= translation <= 1 and -1 <= shift + translation <= 1
    ]


def _slots(observations, translations, width=3):
    by_translation = {int(t): value for t, value in observations}
    values, valid, out_translations = [], [], []
    for translation in translations:
        value = by_translation.get(int(translation))
        values.append(0.0 if value is None else float(value))
        valid.append(value is not None)
        out_translations.append(int(translation))
    while len(values) < width:
        values.append(0.0)
        valid.append(False)
        out_translations.append(0)
    return values[:width], valid[:width], out_translations[:width]


def build_periodic_line_central_sample(key: bytes, topology, trimer):
    prepared, topology_reason = prepare_topology(topology)
    if prepared is None or topology_reason is not None:
        return {
            "sample_key": bytes(key), "base_mapping_valid": False,
            "runtime_valid": False, "invalid_reason": INVALID_BASE_MAPPING,
            "tokens": [], "relations": [],
        }

    z = prepared["z"]
    specs = _internal_bonds(topology)
    specs.append((prepared["right"], 0, prepared["left"], 1, None))
    token_specs = {}
    for u, q_u, v, q_v, bond_type in specs:
        token = canonical_line_token(u, q_u, v, q_v)
        if abs(int(token[2])) > 1:
            raise ValueError(f"invalid periodic chemical bond shift: {token}")
        if token in token_specs:
            raise ValueError(f"duplicate periodic line token: {token}")
        token_specs[token] = bond_type

    trimer_info, _ = prepare_trimer(trimer, topology)
    base_valid = trimer_info is not None
    token_index = {token: index for index, token in enumerate(sorted(token_specs))}
    tokens = []
    for token in sorted(token_specs):
        u, v, shift = token
        instances = _instances(token)
        required_translations = [0] if shift == 0 else sorted(item[4] for item in instances)
        observations, observed_types = [], set()
        if base_valid:
            for a, q_a, b, q_b, translation in instances:
                observed_type = _bond_type(trimer_info, topology, a, q_a, b, q_b)
                if observed_type is None:
                    raise ValueError(f"missing physical Trimer bond instance: {token}")
                observed_types.add(int(observed_type))
                observations.append((translation, _distance(trimer_info, topology, a, q_a, b, q_b)))
        values, mask, translations = _slots(observations, required_translations)
        q_a = translations.copy()
        q_b = [int(value) + int(shift) for value in translations]
        stored_type = token_specs[token]
        if stored_type is None:
            bond_type = next(iter(observed_types)) if len(observed_types) == 1 else 0
        else:
            bond_type = int(stored_type)
        required_count = len(required_translations)
        runtime_valid = bool(base_valid and all(mask[:required_count]))
        z_a, z_b = int(z[u]), int(z[v])
        tokens.append({
            "key": token, "atom_a": u, "atom_b": v, "shift": shift,
            "z_a": z_a, "z_b": z_b, "bond_type": bond_type,
            "label": masked_line_label(z_a, bond_type, z_b),
            "distances": values, "observation_valid": mask,
            "translations": translations, "q_a": q_a, "q_b": q_b,
            "observation_count": sum(mask), "runtime_valid": runtime_valid,
        })

    incident = defaultdict(list)
    for token in sorted(token_specs):
        for u, q_u, v, q_v, _ in _instances(token):
            token_id = token_index[token]
            incident[(u, q_u)].append((token_id, v, q_v))
            incident[(v, q_v)].append((token_id, u, q_u))

    grouped = defaultdict(list)
    for (center, q_c), members in incident.items():
        for left in range(len(members)):
            for right in range(left + 1, len(members)):
                source, outer_a, q_a = members[left]
                target, outer_b, q_b = members[right]
                direct = (outer_a, q_a - q_c, center, outer_b, q_b - q_c)
                inverse = (outer_b, q_b - q_c, center, outer_a, q_a - q_c)
                geometry_key = min(direct, inverse)
                relation_key = (min(source, target), max(source, target), geometry_key)
                value = _angle(trimer_info, topology, outer_a, q_a, center, q_c, outer_b, q_b) if base_valid else None
                grouped[relation_key].append((q_c, value))

    relations, real_sources = [], set()
    for (low, high, geometry_key), observations in sorted(grouped.items()):
        delta_a, delta_b = int(geometry_key[1]), int(geometry_key[4])
        span = max(delta_a, 0, delta_b) - min(delta_a, 0, delta_b)
        if span not in (0, 1, 2):
            raise ValueError(f"invalid canonical angle span: {span}")
        required_translations = {0: [0], 1: sorted(t for t, _ in observations), 2: [0]}[span]
        expected = 1 if span in (0, 2) else 2
        if len(set(required_translations)) != expected:
            raise ValueError(f"angle policy multiplicity mismatch for span={span}")
        values, mask, translations = _slots(observations, required_translations)
        runtime_valid = bool(base_valid and all(mask[:expected]))
        directions = [(low, high)] if low == high else [(low, high), (high, low)]
        for source, target in directions:
            relations.append({
                "source": source, "target": target,
                "center_atom": int(geometry_key[2]),
                "outer_offset_a": delta_a, "outer_offset_b": delta_b,
                "span": span, "multiplicity": expected,
                "angles": values, "observation_valid": mask,
                "translations": translations, "observation_count": sum(mask),
                "runtime_valid": runtime_valid, "fallback": False,
            })
            real_sources.add(source)

    for token_id in range(len(tokens)):
        if token_id not in real_sources:
            relations.append({
                "source": token_id, "target": token_id, "center_atom": -1,
                "outer_offset_a": 0, "outer_offset_b": 0, "span": 0,
                "multiplicity": 1, "angles": [0.0] * 3,
                "observation_valid": [False] * 3, "translations": [0] * 3,
                "observation_count": 0, "runtime_valid": True, "fallback": True,
            })
    relations.sort(key=lambda item: (item["target"], item["source"], item["center_atom"], item["fallback"]))
    distance_valid = all(item["runtime_valid"] for item in tokens)
    angle_valid = all(item["runtime_valid"] for item in relations if not item["fallback"])
    runtime_valid = bool(base_valid and tokens and distance_valid and angle_valid)
    reason = INVALID_NONE if runtime_valid else (
        INVALID_BASE_MAPPING if not base_valid else INVALID_DISTANCE if not distance_valid else INVALID_ANGLE
    )
    return {
        "sample_key": bytes(key), "base_mapping_valid": bool(base_valid),
        "runtime_valid": runtime_valid, "invalid_reason": reason,
        "tokens": tokens, "relations": relations,
    }


class PeriodicLineGLTCentralSidecar:
    def __init__(self, root):
        self.root = Path(root).resolve()
        metadata_path = self.root / "metadata.json"
        if not metadata_path.is_file() or not (self.root / ".done").is_file():
            raise RuntimeError(f"central periodic line sidecar is incomplete: {self.root}")
        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if self.metadata.get("schema") != SIDECAR_SCHEMA:
            raise RuntimeError("unsupported central periodic line sidecar schema")
        self.arrays = {}
        for name in ARRAY_NAMES:
            value = np.load(self.root / f"{name}.npy", mmap_mode="r", allow_pickle=False)
            spec = self.metadata["arrays"][name]
            if list(value.shape) != list(spec["shape"]) or np.dtype(value.dtype).str != spec["dtype"]:
                raise RuntimeError(f"central periodic line shape/dtype mismatch: {name}")
            self.arrays[name] = value
        self._key_to_row = None

    def __len__(self):
        return int(self.arrays["sample_keys"].shape[0])

    def index_for_key(self, key, row_hint=None):
        raw = np.frombuffer(bytes(key), dtype=np.uint8)
        if row_hint is not None and 0 <= int(row_hint) < len(self) and np.array_equal(self.arrays["sample_keys"][int(row_hint)], raw):
            return int(row_hint)
        if self._key_to_row is None:
            self._key_to_row = {bytes(row): i for i, row in enumerate(self.arrays["sample_keys"])}
        if bytes(key) not in self._key_to_row:
            raise KeyError("sample key absent from central periodic line sidecar")
        return self._key_to_row[bytes(key)]

    def model_row(self, index):
        index = int(index)
        ts, te = (int(self.arrays["sample_token_offsets"][index + i]) for i in (0, 1))
        rs, re = (int(self.arrays["sample_relation_offsets"][index + i]) for i in (0, 1))
        token_names = tuple(name for name in ARRAY_NAMES if name.startswith("token_"))
        relation_names = tuple(name for name in ARRAY_NAMES if name.startswith("relation_"))
        return {
            "geometry_valid": bool(self.arrays["graph_runtime_valid"][index]),
            "base_mapping_valid": bool(self.arrays["graph_base_mapping_valid"][index]),
            "invalid_reason": int(self.arrays["graph_invalid_reason"][index]),
            "tokens": {name: np.asarray(self.arrays[name][ts:te]) for name in token_names},
            "relations": {name: np.asarray(self.arrays[name][rs:re]) for name in relation_names},
        }

    def qc_row(self, index):
        return self.model_row(index)


__all__ = [
    "ARRAY_DTYPES", "ARRAY_NAMES", "BUILDER_VERSION", "SIDECAR_SCHEMA",
    "INVALID_NONE", "INVALID_BASE_MAPPING", "INVALID_DISTANCE", "INVALID_ANGLE",
    "PeriodicLineGLTCentralSidecar", "build_periodic_line_central_sample",
]
