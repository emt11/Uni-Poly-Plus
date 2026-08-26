"""Periodic non-bonded spatial contacts for MTS-GLT-v2-MSContact-v1."""

from __future__ import annotations

from collections import Counter, deque
import json
from pathlib import Path

import numpy as np
import torch

from .canonical_periodic import _neighbors
from .periodic_line_glt import (
    _distance,
    _token_instances,
    canonical_line_token,
    prepare_topology,
    prepare_trimer,
)


SIDECAR_SCHEMA = "mts-periodic-spatial-contact-v1"
BUILDER_VERSION = 1
SPATIAL_SPD_MIN = 4
SPATIAL_CORE_CUTOFF = 4.0
SPATIAL_OUTER_CUTOFF = 5.0

ARRAY_DTYPES = {
    "sample_keys": np.dtype("u1"),
    "sample_pair_offsets": np.dtype("<i8"),
    "graph_valid": np.dtype("<?"),
    "sample_atom_count": np.dtype("<i2"),
    "pair_atom_a": np.dtype("<i2"),
    "pair_atom_b": np.dtype("<i2"),
    "pair_shift": np.dtype("<i2"),
    "pair_observation_distances": np.dtype("<f4"),
    "pair_observation_valid": np.dtype("<?"),
    "pair_observation_count": np.dtype("u1"),
    "pair_shell_id": np.dtype("u1"),
    "pair_periodic_self": np.dtype("<?"),
    "pair_valid": np.dtype("<?"),
    # QC-only arrays; model_row deliberately does not expose them.
    "pair_spd": np.dtype("<i2"),
    "pair_raw_mean_distance": np.dtype("<f4"),
    "pair_raw_variance": np.dtype("<f4"),
}
ARRAY_NAMES = tuple(ARRAY_DTYPES)


def _lifted_spd_maps(topology, max_abs_shift: int):
    atom_count = int(topology.canonical_atom_count)
    edges = np.asarray(topology.ru_edge_index.detach().cpu(), dtype=np.int64)
    internal = tuple(sorted({
        (min(int(a), int(b)), max(int(a), int(b)))
        for a, b in edges.T.tolist() if int(a) != int(b)
    }))
    left = int(topology.ru_left_boundary)
    right = int(topology.ru_right_boundary)
    shift_limit = max(4, int(max_abs_shift) + 2)
    output = {}
    for source in range(atom_count):
        start = (source, 0)
        distances = {start: 0}
        queue = deque([start])
        while queue:
            state = queue.popleft()
            for neighbor in _neighbors(state, internal, left, right):
                neighbor = (int(neighbor[0]), int(neighbor[1]))
                if abs(neighbor[1]) > shift_limit or neighbor in distances:
                    continue
                distances[neighbor] = distances[state] + 1
                queue.append(neighbor)
        output[source] = distances
    return output


def build_spatial_contact_sample(key: bytes, topology, trimer):
    """Build one filtered SPD>=4, mean-distance<=5A canonical pair table."""
    prepared, topology_reason = prepare_topology(topology)
    trimer_info, trimer_reason = prepare_trimer(trimer, topology)
    if prepared is None or topology_reason is not None or trimer_info is None:
        return {"sample_key": bytes(key), "graph_valid": False, "atom_count": 0, "pairs": []}
    if trimer_reason is not None:
        return {"sample_key": bytes(key), "graph_valid": False, "atom_count": 0, "pairs": []}

    mapping = np.asarray(
        topology.canonical_to_trimer_base_atom_id.detach().cpu(), dtype=np.int64
    )
    atom_count = int(topology.canonical_atom_count)
    if mapping.size != atom_count or len(set(mapping.tolist())) != atom_count:
        return {"sample_key": bytes(key), "graph_valid": False, "atom_count": atom_count, "pairs": []}
    base_to_canonical = {int(base): i for i, base in enumerate(mapping.tolist())}
    offsets = {index: set() for index in range(atom_count)}
    for (base, shift), _local in trimer_info["state_to_local"].items():
        canonical = base_to_canonical.get(int(base))
        if canonical is None:
            return {"sample_key": bytes(key), "graph_valid": False, "atom_count": atom_count, "pairs": []}
        offsets[canonical].add(int(shift))

    raw_counts = Counter()
    for atom_a in range(atom_count):
        left_offsets = sorted(offsets[atom_a])
        for atom_b in range(atom_a, atom_count):
            right_offsets = sorted(offsets[atom_b])
            if atom_a == atom_b:
                instances = (
                    (q_a, q_b) for pos, q_a in enumerate(left_offsets)
                    for q_b in left_offsets[pos + 1:]
                )
            else:
                instances = (
                    (q_a, q_b) for q_a in left_offsets for q_b in right_offsets
                )
            for q_a, q_b in instances:
                raw_counts[canonical_line_token(atom_a, q_a, atom_b, q_b)] += 1

    max_shift = max((abs(int(token[2])) for token in raw_counts), default=0)
    spd_maps = _lifted_spd_maps(topology, max_shift)
    pairs = []
    for token in sorted(raw_counts):
        atom_a, atom_b, shift = map(int, token)
        instances = _token_instances(token)
        if len(instances) != int(raw_counts[token]):
            raise ValueError("spatial contact observation multiplicity mismatch")
        distances = []
        for left, q_left, right, q_right in instances:
            value = _distance(trimer_info, topology, left, q_left, right, q_right)
            if value is None:
                return {"sample_key": bytes(key), "graph_valid": False, "atom_count": atom_count, "pairs": []}
            distances.append(float(value))
        values = np.asarray(distances, dtype=np.float64)
        spd = spd_maps.get(atom_a, {}).get((atom_b, shift))
        if spd is None or int(spd) < SPATIAL_SPD_MIN:
            continue
        mean = float(values.mean())
        if mean > SPATIAL_OUTER_CUTOFF:
            continue
        shell = 0 if mean <= SPATIAL_CORE_CUTOFF else 1
        padded = np.zeros(3, dtype=np.float32)
        valid_mask = np.zeros(3, dtype=bool)
        padded[: values.size] = values.astype(np.float32)
        valid_mask[: values.size] = True
        pairs.append({
            "atom_a": atom_a,
            "atom_b": atom_b,
            "shift": shift,
            "distances": padded,
            "observation_valid": valid_mask,
            "observation_count": int(values.size),
            "shell_id": shell,
            "periodic_self": bool(atom_a == atom_b and shift != 0),
            "valid": True,
            "spd": int(spd),
            "raw_mean_distance": mean,
            "raw_variance": float(values.var()),
        })
    return {"sample_key": bytes(key), "graph_valid": True, "atom_count": atom_count, "pairs": pairs}


class PeriodicSpatialContactSidecar:
    """Lightweight mmap reader; QC-only fields stay off the training row."""

    def __init__(self, root):
        self.root = Path(root).resolve()
        if not (self.root / "metadata.json").is_file() or not (self.root / ".done").is_file():
            raise RuntimeError(f"periodic spatial sidecar is incomplete: {self.root}")
        self.metadata = json.loads((self.root / "metadata.json").read_text())
        if self.metadata.get("schema") != SIDECAR_SCHEMA:
            raise RuntimeError("unsupported periodic spatial sidecar schema")
        self.arrays = {}
        for name in ARRAY_NAMES:
            value = np.load(self.root / f"{name}.npy", mmap_mode="r", allow_pickle=False)
            spec = self.metadata["arrays"][name]
            if list(value.shape) != list(spec["shape"]) or np.dtype(value.dtype).str != spec["dtype"]:
                raise RuntimeError(f"periodic spatial sidecar shape/dtype mismatch: {name}")
            self.arrays[name] = value
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
        try:
            return self._key_to_row[bytes(key)]
        except KeyError as error:
            raise KeyError("sample key absent from periodic spatial sidecar") from error

    def model_row(self, index):
        index = int(index)
        start, end = (
            int(self.arrays["sample_pair_offsets"][index + offset]) for offset in (0, 1)
        )
        names = (
            "pair_atom_a", "pair_atom_b", "pair_shift",
            "pair_observation_distances", "pair_observation_valid",
            "pair_observation_count", "pair_shell_id", "pair_periodic_self",
            "pair_valid",
        )
        return {
            "graph_valid": bool(self.arrays["graph_valid"][index]),
            "pairs": {name: np.asarray(self.arrays[name][start:end]) for name in names},
        }

    def qc_row(self, index):
        index = int(index)
        start, end = (
            int(self.arrays["sample_pair_offsets"][index + offset]) for offset in (0, 1)
        )
        return {
            **self.model_row(index),
            "qc": {
                name: np.asarray(self.arrays[name][start:end])
                for name in ("pair_spd", "pair_raw_mean_distance", "pair_raw_variance")
            },
        }


__all__ = [
    "ARRAY_DTYPES", "ARRAY_NAMES", "BUILDER_VERSION", "SIDECAR_SCHEMA",
    "PeriodicSpatialContactSidecar", "build_spatial_contact_sample",
]
