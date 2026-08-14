"""Relation-wise periodic Trimer geometry for Star-RBF v2.

The pure builders in this module consume already loaded, frozen Topology and
Trimer records.  They never write a cache and never use 2-D coordinates.
"""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from .mips_trimer_contract import (
    STAR_RBF_V2_BUILDER_VERSION,
    STAR_RBF_V2_SIDECAR_SCHEMA,
)
GEOMETRY_SOURCE = {
    "invalid": 0,
    "trivial_self_no_bias": 1,
    "central_direct": 2,
    "adjacent_dual": 3,
    "outer_trimer_direct": 4,
}
INVALID_REASON = {
    "valid": 0,
    "graph_unavailable": 1,
    "topology_invalid": 2,
    "geometry_invalid": 3,
    "ru_offset_uncovered": 4,
    "nonfinite_distance": 5,
    "nonpositive_distance": 6,
}

ARRAY_DTYPES = {
    "sample_keys": np.dtype("u1"),
    "sample_relation_offsets": np.dtype("<i8"),
    "sample_pair_offsets": np.dtype("<i8"),
    "relation_row": np.dtype("<i4"),
    "relation_pair_index": np.dtype("<i4"),
    "relation_spd": np.dtype("<i2"),
    "pair_key_src": np.dtype("<i4"),
    "pair_key_dst": np.dtype("<i4"),
    "pair_key_shift": np.dtype("<i2"),
    "pair_spd": np.dtype("<i2"),
    "pair_relation_multiplicity": np.dtype("<i2"),
    "pair_path_signature_hash": np.dtype("u1"),
    "pair_observation_distances": np.dtype("<f4"),
    "pair_observation_count": np.dtype("u1"),
    "pair_valid": np.dtype("<?"),
    "pair_invalid_reason_code": np.dtype("<i2"),
    "pair_geometry_source": np.dtype("u1"),
    "pair_absolute_asymmetry": np.dtype("<f4"),
    "pair_relative_asymmetry": np.dtype("<f4"),
}
ARRAY_NAMES = tuple(ARRAY_DTYPES)


def _as_bool(value, default=False) -> bool:
    if value is None:
        return bool(default)
    try:
        return bool(torch.as_tensor(value).reshape(-1)[0].item())
    except (IndexError, RuntimeError, TypeError, ValueError):
        return bool(value)


def _tensor(value, *, dtype=None):
    if value is None:
        return None
    return torch.as_tensor(value, dtype=dtype)


def _internal_edges(topology):
    edge = _tensor(getattr(topology, "ru_edge_index", None), dtype=torch.long)
    if edge is None or edge.ndim != 2 or edge.size(0) != 2:
        edge = _tensor(getattr(topology, "edge_index", None), dtype=torch.long)
    if edge is None or edge.ndim != 2 or edge.size(0) != 2:
        return None
    unique = set()
    for left, right in edge.t().tolist():
        left, right = int(left), int(right)
        if left == right:
            continue
        unique.add((min(left, right), max(left, right)))
    return sorted(unique)


def prepare_topology(topology):
    """Validate the topology fields required by Star-RBF v2."""

    edge = _tensor(getattr(topology, "lga_edge_index", None), dtype=torch.long)
    spd = _tensor(getattr(topology, "lga_spd", None), dtype=torch.long)
    shift = _tensor(getattr(topology, "lga_source_image_shift", None), dtype=torch.long)
    z = _tensor(
        getattr(topology, "atomic_numbers", getattr(topology, "z", None)),
        dtype=torch.long,
    )
    left = getattr(topology, "ru_left_boundary", None)
    right = getattr(topology, "ru_right_boundary", None)
    internal = _internal_edges(topology)
    if (
        edge is None
        or edge.ndim != 2
        or edge.size(0) != 2
        or spd is None
        or spd.ndim != 1
        or spd.numel() != edge.size(1)
        or shift is None
        or shift.numel() != edge.size(1)
        or z is None
        or z.ndim != 1
        or left is None
        or right is None
        or internal is None
    ):
        return None, "topology_relation_fields_invalid"
    if edge.numel() and (int(edge.min()) < 0 or int(edge.max()) >= int(z.numel())):
        return None, "topology_relation_endpoint_invalid"
    if int(left) < 0 or int(right) < 0 or int(left) >= int(z.numel()) or int(right) >= int(z.numel()):
        return None, "topology_relation_endpoint_invalid"
    return {
        "edge": edge,
        "spd": spd,
        "shift": shift.reshape(-1),
        "z": z.reshape(-1),
        "left": int(left),
        "right": int(right),
        "internal_edges": internal,
    }, None


def prepare_trimer(trimer, topology=None):
    """Validate explicit central-RU identity and 3-D geometry fields."""

    if not _as_bool(getattr(trimer, "trimer_geometry_valid", False)):
        if _as_bool(getattr(trimer, "trimer_2d_fallback", False)):
            return None, "2d_fallback"
        if not _as_bool(getattr(trimer, "trimer_geometry_is_3d", False)):
            return None, "non_3d_geometry"
        return None, "geometry_invalid"
    if not _as_bool(getattr(trimer, "trimer_geometry_is_3d", False)):
        return None, "non_3d_geometry"
    if _as_bool(getattr(trimer, "trimer_2d_fallback", False)):
        return None, "2d_fallback"
    positions = _tensor(getattr(trimer, "trimer_pos", None))
    atomic = _tensor(getattr(trimer, "trimer_atomic_number", None), dtype=torch.long)
    base = _tensor(getattr(trimer, "trimer_base_ru_atom_id", None), dtype=torch.long)
    offsets = _tensor(getattr(trimer, "trimer_ru_offset", None), dtype=torch.long)
    central = _tensor(getattr(trimer, "trimer_central_ru_mask", None), dtype=torch.bool)
    mapping = _tensor(getattr(trimer, "mips_to_trimer_central_index", None), dtype=torch.long)
    edge = _tensor(getattr(trimer, "trimer_edge_index", None), dtype=torch.long)
    bond = _tensor(getattr(trimer, "trimer_bond_type", None), dtype=torch.long)
    if positions is None or positions.ndim != 2 or positions.size(1) != 3:
        return None, "trimer_identity_missing"
    if not bool(torch.isfinite(positions).all()):
        return None, "nonfinite_coordinates"
    if any(value is None for value in (atomic, base, offsets, central, mapping, edge, bond)):
        return None, "trimer_identity_missing"
    if (
        atomic.ndim != 1
        or base.ndim != 1
        or offsets.ndim != 1
        or central.ndim != 1
        or atomic.numel() != positions.size(0)
        or base.numel() != positions.size(0)
        or offsets.numel() != positions.size(0)
        or central.numel() != positions.size(0)
        or mapping.ndim != 1
        or bool((mapping < 0).any())
        or bool((mapping >= positions.size(0)).any())
        or not bool(central[mapping].all())
        or edge.ndim != 2
        or edge.size(0) != 2
        or bond.ndim != 1
        or bond.numel() != edge.size(1)
    ):
        return None, "mapping_invalid"
    state_to_local = {}
    for index in range(int(positions.size(0))):
        state = (int(base[index]), int(offsets[index]))
        if state in state_to_local:
            return None, "trimer_identity_duplicate"
        state_to_local[state] = index
    if topology is not None:
        topology_z = _tensor(
            getattr(topology, "atomic_numbers", getattr(topology, "z", None)),
            dtype=torch.long,
        )
        canonical_to_trimer = _tensor(
            getattr(
                topology,
                "canonical_to_trimer_base_atom_id",
                getattr(topology, "canonical_to_trimer_base_atom_index", None),
            ),
            dtype=torch.long,
        )
        if (
            topology_z is None
            or canonical_to_trimer is None
            or canonical_to_trimer.numel() != topology_z.numel()
            or mapping.numel() != topology_z.numel()
        ):
            return None, "canonical_trimer_identity_missing"
        for canonical_id in range(int(topology_z.numel())):
            base_id = int(canonical_to_trimer[canonical_id])
            local = state_to_local.get((base_id, 0))
            if local is None or int(mapping[canonical_id]) != int(local):
                return None, "canonical_trimer_mapping_mismatch"
            if int(atomic[local]) != int(topology_z[canonical_id]):
                return None, "canonical_trimer_atomic_mismatch"
    bonds = {}
    for column in range(int(edge.size(1))):
        left, right = int(edge[0, column]), int(edge[1, column])
        if (
            left == right
            or left < 0
            or right < 0
            or left >= positions.size(0)
            or right >= positions.size(0)
        ):
            return None, "real_bond_graph_invalid"
        pair = (min(left, right), max(left, right))
        bonds.setdefault(pair, set()).add(int(bond[column]))
    return {
        "positions": positions.float(),
        "atomic": atomic.reshape(-1),
        "state_to_local": state_to_local,
        "bonds": bonds,
    }, None


def periodic_pair_key(a: int, b: int, shift: int) -> tuple[int, int, int]:
    relation = (int(a), int(b), int(shift))
    inverse = (int(b), int(a), -int(shift))
    return min(relation, inverse)


def rbf_upper_from_distances(distances, grid: float = 0.25) -> float:
    values = np.asarray(distances, dtype=np.float64)
    values = values[np.isfinite(values) & (values > 0)]
    if not values.size:
        raise ValueError("cannot derive Star-RBF v2 upper without valid distances")
    p999 = float(np.percentile(values, 99.9))
    return float(max(3.0, math.ceil(p999 / grid) * grid + grid))


def _path_member(topology, row: int):
    index = torch.as_tensor(topology.lga_path_index)[row].reshape(-1)
    shift = torch.as_tensor(topology.lga_path_shift)[row].reshape(-1)
    mask = torch.as_tensor(topology.lga_path_mask)[row].bool().reshape(-1)
    return [
        [int(index[i]), int(shift[i])]
        for i in range(mask.numel()) if bool(mask[i])
    ]


def _signature(members) -> bytes:
    normalized = sorted(
        members, key=lambda item: (
            tuple(item["relation"]),
            tuple(tuple(value) for value in item["path"]),
        )
    )
    payload = json.dumps(normalized, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).digest()


def _distance(trimer_info, topology, atom_a: int, ru_a: int, atom_b: int, ru_b: int):
    mapping = torch.as_tensor(topology.canonical_to_trimer_base_atom_id).long()
    state = trimer_info["state_to_local"]
    left = state.get((int(mapping[int(atom_a)]), int(ru_a)))
    right = state.get((int(mapping[int(atom_b)]), int(ru_b)))
    if left is None or right is None:
        return None, "ru_offset_uncovered"
    value = torch.linalg.vector_norm(
        trimer_info["positions"][int(left)] - trimer_info["positions"][int(right)]
    )
    if not bool(torch.isfinite(value)):
        return None, "nonfinite_distance"
    value = float(value.item())
    if value <= 0.0:
        return None, "nonpositive_distance"
    return value, None


def build_star_rbf_v2_sample(key: bytes, topology, trimer):
    """Build all SPD<=2 relation-to-periodic-pair geometry for one sample."""

    prepared, topology_reason = prepare_topology(topology)
    trimer_info, trimer_reason = prepare_trimer(trimer, topology)
    graph_available = bool(getattr(topology, "graph_available", False))
    if prepared is None:
        return {"sample_key": bytes(key), "relations": [], "pairs": []}

    relation_members = defaultdict(list)
    for row in range(int(prepared["edge"].size(1))):
        spd = int(prepared["spd"][row])
        shift = int(prepared["shift"][row])
        if spd <= 2 and abs(shift) > 2:
            raise ValueError(
                f"Star-RBF v2 topology invariant violated at row {row}: "
                f"SPD={spd}, shift={shift}"
            )
        if spd > 2:
            continue
        # lga target=a, source=b.
        b = int(prepared["edge"][0, row])
        a = int(prepared["edge"][1, row])
        relation = (a, b, shift)
        pair_key = periodic_pair_key(*relation)
        relation_members[pair_key].append({
            "row": row,
            "relation": relation,
            "spd": spd,
            "path": _path_member(topology, row),
        })

    relations = []
    pairs = []
    for pair_index, pair_key in enumerate(sorted(relation_members)):
        members = relation_members[pair_key]
        identities = [tuple(member["relation"]) for member in members]
        if len(set(identities)) != len(identities):
            raise ValueError(f"duplicate Star-RBF v2 directed member: {pair_key}")
        spds = {int(member["spd"]) for member in members}
        if len(spds) != 1:
            raise ValueError(f"Star-RBF v2 pair SPD conflict: {pair_key}")
        expected = 1 if pair_key == (pair_key[1], pair_key[0], -pair_key[2]) else 2
        if len(members) != expected:
            raise ValueError(
                f"Star-RBF v2 inverse multiplicity mismatch for {pair_key}: "
                f"expected {expected}, observed {len(members)}"
            )
        if expected == 2:
            first = identities[0]
            if (first[1], first[0], -first[2]) not in identities:
                raise ValueError(f"Star-RBF v2 inverse relation missing: {pair_key}")

        a, b, shift = pair_key
        # Orient geometry to a positive shift without changing lookup identity.
        if shift < 0:
            a, b, shift = b, a, -shift
        spd = next(iter(spds))
        distances = []
        source = GEOMETRY_SOURCE["invalid"]
        reason = None
        if not graph_available:
            reason = "graph_unavailable"
        elif topology_reason:
            reason = "topology_invalid"
        elif a == b and shift == 0 and spd == 0:
            source = GEOMETRY_SOURCE["trivial_self_no_bias"]
        elif trimer_info is None:
            reason = "geometry_invalid"
        elif shift == 0:
            source = GEOMETRY_SOURCE["central_direct"]
            value, reason = _distance(trimer_info, topology, a, 0, b, 0)
            if value is not None:
                distances = [value]
        elif shift == 1:
            source = GEOMETRY_SOURCE["adjacent_dual"]
            left, left_reason = _distance(trimer_info, topology, a, -1, b, 0)
            right, right_reason = _distance(trimer_info, topology, a, 0, b, 1)
            reason = left_reason or right_reason
            if reason is None:
                distances = [left, right]
        elif shift == 2:
            source = GEOMETRY_SOURCE["outer_trimer_direct"]
            value, reason = _distance(trimer_info, topology, a, -1, b, 1)
            if value is not None:
                distances = [value]
        else:
            reason = "topology_invalid"

        trivial = source == GEOMETRY_SOURCE["trivial_self_no_bias"]
        valid = bool(trivial or (distances and reason is None))
        padded = (distances + [0.0, 0.0])[:2]
        abs_asym = abs(distances[0] - distances[1]) if len(distances) == 2 else 0.0
        rel_asym = abs_asym / ((sum(distances) / 2.0) + 1e-8) if len(distances) == 2 else 0.0
        signature_members = [
            {"relation": list(member["relation"]), "path": member["path"]}
            for member in members
        ]
        pairs.append({
            "key": pair_key,
            "spd": spd,
            "multiplicity": len(members),
            "path_signature_hash": _signature(signature_members),
            "distances": padded,
            "observation_count": len(distances),
            "valid": valid,
            "reason_code": INVALID_REASON["valid" if valid else (reason or "geometry_invalid")],
            "geometry_source": source,
            "absolute_asymmetry": abs_asym,
            "relative_asymmetry": rel_asym,
        })
        relations.extend({
            "row": member["row"], "pair_index": pair_index, "spd": member["spd"]
        } for member in members)

    relations.sort(key=lambda item: item["row"])
    return {"sample_key": bytes(key), "relations": relations, "pairs": pairs}


class StarRBFV2Sidecar:
    """Lightweight read-only mmap reader for a frozen v2 sidecar.

    Startup validates only the immutable layout declaration.  Expensive
    content/QC checks belong to the explicit offline sidecar QC command; the
    model reader validates the currently requested row on first access.
    """

    def __init__(self, root):
        self.root = Path(root).resolve()
        metadata_path = self.root / "metadata.json"
        if not all((self.root / name).is_file() for name in ("metadata.json", ".done", ".frozen")):
            raise RuntimeError(f"Star-RBF v2 sidecar is incomplete: {self.root}")
        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if self.metadata.get("schema") != STAR_RBF_V2_SIDECAR_SCHEMA:
            raise RuntimeError("unsupported Star-RBF v2 sidecar schema")
        if int(self.metadata.get("builder_version", -1)) != STAR_RBF_V2_BUILDER_VERSION:
            raise RuntimeError("Star-RBF v2 builder version mismatch")
        frozen = json.loads((self.root / ".frozen").read_text(encoding="utf-8"))
        if frozen.get("schema") != STAR_RBF_V2_SIDECAR_SCHEMA:
            raise RuntimeError("Star-RBF v2 frozen schema mismatch")
        if set(self.metadata.get("arrays", {})) != set(ARRAY_NAMES):
            raise RuntimeError("Star-RBF v2 array manifest mismatch")
        self.arrays = {}
        for name in ARRAY_NAMES:
            path = self.root / f"{name}.npy"
            value = np.load(path, mmap_mode="r", allow_pickle=False)
            spec = self.metadata["arrays"][name]
            if list(value.shape) != list(spec["shape"]) or np.dtype(value.dtype).str != spec["dtype"]:
                raise RuntimeError(f"Star-RBF v2 array shape/dtype mismatch: {name}")
            self.arrays[name] = value
        sample_count = int(self.arrays["sample_keys"].shape[0])
        relation_count = int(self.arrays["relation_row"].shape[0])
        pair_count = int(self.arrays["pair_valid"].shape[0])
        if self.arrays["sample_keys"].shape != (sample_count, 32):
            raise RuntimeError("Star-RBF v2 sample-key shape mismatch")
        for name in ("sample_relation_offsets", "sample_pair_offsets"):
            offsets = self.arrays[name]
            expected_end = relation_count if name == "sample_relation_offsets" else pair_count
            if offsets.shape != (sample_count + 1,) or int(offsets[0]) != 0 or int(offsets[-1]) != expected_end:
                raise RuntimeError(f"Star-RBF v2 offset boundary mismatch: {name}")
        if any(self.arrays[name].shape != (relation_count,) for name in (
            "relation_row", "relation_pair_index", "relation_spd"
        )):
            raise RuntimeError("Star-RBF v2 relation array length mismatch")
        if self.arrays["pair_observation_distances"].shape != (pair_count, 2):
            raise RuntimeError("Star-RBF v2 distance array shape mismatch")
        if self.arrays["pair_path_signature_hash"].shape != (pair_count, 32):
            raise RuntimeError("Star-RBF v2 path-signature shape mismatch")
        self.rbf_upper = float(self.metadata["rbf"]["upper"])
        self._key_to_row = None

    def __len__(self):
        return int(self.arrays["sample_keys"].shape[0])

    def index_for_key(self, key, row_hint=None):
        raw = np.frombuffer(bytes(key), dtype=np.uint8)
        if row_hint is not None and 0 <= int(row_hint) < len(self):
            if np.array_equal(self.arrays["sample_keys"][int(row_hint)], raw):
                return int(row_hint)
        if self._key_to_row is None:
            self._key_to_row = {bytes(row): i for i, row in enumerate(self.arrays["sample_keys"])}
        if bytes(key) not in self._key_to_row:
            raise KeyError("sample key absent from Star-RBF v2 sidecar")
        return self._key_to_row[bytes(key)]

    def model_row(self, index):
        index = int(index)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        rs, re = (int(self.arrays["sample_relation_offsets"][index + off]) for off in (0, 1))
        ps, pe = (int(self.arrays["sample_pair_offsets"][index + off]) for off in (0, 1))
        relation_count = int(self.arrays["relation_row"].shape[0])
        pair_count = int(self.arrays["pair_valid"].shape[0])
        if rs < 0 or re < rs or re > relation_count or ps < 0 or pe < ps or pe > pair_count:
            raise IndexError(f"Star-RBF v2 row offsets are invalid: {index}")
        pair_indices = np.asarray(self.arrays["relation_pair_index"][rs:re], dtype=np.int64)
        if pair_indices.size and (
            int(pair_indices.min()) < ps or int(pair_indices.max()) >= pe
        ):
            raise RuntimeError(f"Star-RBF v2 relation pair index is outside row {index}")
        counts = np.asarray(self.arrays["pair_observation_count"][ps:pe], dtype=np.int64)
        if counts.size and int(counts.max()) > 2:
            raise RuntimeError(f"Star-RBF v2 observation count exceeds two in row {index}")
        return {
            "relations": {name: np.asarray(self.arrays[name][rs:re]) for name in (
                "relation_row", "relation_pair_index", "relation_spd")},
            "pairs": {name: np.asarray(self.arrays[name][ps:pe]) for name in (
                "pair_observation_distances", "pair_observation_count",
                "pair_valid", "pair_geometry_source")},
        }

    def qc_row(self, index):
        """Return the complete row, including fields used only by offline QC."""
        index = int(index)
        model = self.model_row(index)
        ps = int(self.arrays["sample_pair_offsets"][index])
        pe = int(self.arrays["sample_pair_offsets"][index + 1])
        model["pairs"].update({
            name: np.asarray(self.arrays[name][ps:pe])
            for name in ARRAY_NAMES if name.startswith("pair_")
            and name not in model["pairs"]
        })
        return model

    def row(self, index):
        """Compatibility alias; new model code should call ``model_row``."""
        return self.model_row(index)


__all__ = [
    "ARRAY_DTYPES", "ARRAY_NAMES", "GEOMETRY_SOURCE", "INVALID_REASON",
    "StarRBFV2Sidecar", "build_star_rbf_v2_sample", "periodic_pair_key",
    "rbf_upper_from_distances",
]
