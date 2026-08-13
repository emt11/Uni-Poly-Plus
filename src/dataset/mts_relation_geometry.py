"""Read-only relation-geometry sidecar core for the MTS canonical cache.

This module is deliberately independent of the Dataset and model code.  It
contains the single implementation of lifted two-edge path enumeration,
canonical-to-Trimer identity validation and real-bond 3-D geometry used by
the G-precheck and the sidecar builder.  It never opens an LMDB writer.
"""

from __future__ import annotations

from collections import OrderedDict
import hashlib
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch

from .mips_trimer_contract import (
    CACHE_RELATION_GEOMETRY_SCHEMA,
    RELATION_GEOMETRY_BUILDER_VERSION,
)


# Integer codes are part of the sidecar contract.  Keep the order stable: a
# human-readable mapping is stored in metadata and readers must not infer
# meaning from the sentinel values in the payload arrays.
INVALID_REASON_CODES = OrderedDict(
    [
        ("valid", 0),
        ("graph_unavailable", 1),
        ("topology_relation_fields_invalid", 2),
        ("topology_relation_endpoint_invalid", 3),
        ("2d_fallback", 4),
        ("non_3d_geometry", 5),
        ("geometry_invalid", 6),
        ("nonfinite_coordinates", 7),
        ("trimer_identity_missing", 8),
        ("mapping_invalid", 9),
        ("trimer_identity_duplicate", 10),
        ("canonical_trimer_identity_missing", 11),
        ("canonical_trimer_mapping_mismatch", 12),
        ("canonical_trimer_atomic_mismatch", 13),
        ("real_bond_graph_invalid", 14),
        ("ru_offset_uncovered", 15),
        ("real_bond_missing", 16),
        ("real_bond_ambiguous", 17),
        ("degenerate_geometry", 18),
        ("nonfinite_geometry", 19),
        ("no_two_edge_paths", 20),
        ("no_valid_geometry", 21),
        ("partial_path_geometry", 22),
        ("endpoint_distance_invalid", 23),
        ("record_read_error", 24),
    ]
)
REASON_TO_CODE = dict(INVALID_REASON_CODES)
CODE_TO_REASON = {value: key for key, value in INVALID_REASON_CODES.items()}

# The persisted names are intentionally duplicated here instead of importing
# the builder script.  Dataset workers must be able to open a sidecar without
# importing a command-line module (and without opening an LMDB writer).
RELATION_GEOMETRY_ARRAY_NAMES = (
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


def reason_code(reason: str | None) -> int:
    """Return a stable payload code, treating ``None`` as valid."""

    return int(REASON_TO_CODE.get(str(reason or "valid"), REASON_TO_CODE["record_read_error"]))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class RelationGeometrySidecar:
    """Strict, read-only mmap reader for a frozen relation-geometry artifact.

    The reader is deliberately small enough to be constructed in Dataset
    workers.  It validates the immutable identity and all array boundaries at
    startup, then returns NumPy views for one sample row.  No cache or sidecar
    file is ever written by this class.
    """

    def __init__(
        self,
        root,
        *,
        expected_cohort=None,
        expected_cohort_hash=None,
        expected_artifact_hash=None,
        expected_ordered_sample_key_hash=None,
        expected_source_artifacts=None,
        verify_array_hashes=True,
    ):
        self.root = Path(root).resolve()
        metadata_path = self.root / "metadata.json"
        done_path = self.root / ".done"
        frozen_path = self.root / ".frozen"
        if not (metadata_path.is_file() and done_path.is_file() and frozen_path.is_file()):
            raise RuntimeError(f"relation-geometry sidecar is incomplete: {self.root}")
        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if self.metadata.get("schema") != CACHE_RELATION_GEOMETRY_SCHEMA:
            raise RuntimeError("unsupported relation-geometry sidecar schema")
        if int(self.metadata.get("builder_version", -1)) != int(RELATION_GEOMETRY_BUILDER_VERSION):
            raise RuntimeError("relation-geometry sidecar builder version mismatch")
        artifact = str(self.metadata.get("artifact_hash", ""))
        if expected_artifact_hash is not None and artifact != str(expected_artifact_hash):
            raise RuntimeError("relation-geometry sidecar artifact hash mismatch")
        if len(artifact) != 64 or done_path.read_text(encoding="utf-8").strip() != artifact:
            raise RuntimeError("relation-geometry sidecar .done binding mismatch")
        frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
        if frozen.get("schema") != CACHE_RELATION_GEOMETRY_SCHEMA or frozen.get("artifact_hash") != artifact:
            raise RuntimeError("relation-geometry sidecar .frozen binding mismatch")
        expected_artifact = hashlib.sha256(
            json.dumps(
                {key: value for key, value in self.metadata.items() if key != "artifact_hash"},
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
        if expected_artifact != artifact:
            raise RuntimeError("relation-geometry sidecar metadata hash mismatch")
        if expected_cohort_hash is not None and str(self.metadata.get("cohort_hash")) != str(expected_cohort_hash):
            raise RuntimeError("relation-geometry sidecar cohort hash mismatch")
        if expected_cohort is not None and str(self.metadata.get("cohort")) != str(expected_cohort):
            raise RuntimeError("relation-geometry sidecar cohort name mismatch")
        if expected_ordered_sample_key_hash is not None and str(self.metadata.get("ordered_sample_key_hash")) != str(expected_ordered_sample_key_hash):
            raise RuntimeError("relation-geometry sidecar ordered-key hash mismatch")
        if expected_source_artifacts:
            identity = self.metadata.get("source_identity", {})
            for name, expected in expected_source_artifacts.items():
                observed = identity.get(name, {})
                if isinstance(expected, dict):
                    for key, value in expected.items():
                        if value is not None and observed.get(key) != value:
                            raise RuntimeError(f"relation-geometry sidecar source identity mismatch: {name}.{key}")
                elif expected is not None and observed.get("done_artifact_hash") != expected:
                    raise RuntimeError(f"relation-geometry sidecar source identity mismatch: {name}")
        arrays_meta = self.metadata.get("arrays", {})
        if set(arrays_meta) != set(RELATION_GEOMETRY_ARRAY_NAMES):
            raise RuntimeError("relation-geometry sidecar array manifest is incomplete")
        self.arrays = {}
        for name in RELATION_GEOMETRY_ARRAY_NAMES:
            path = self.root / f"{name}.npy"
            if not path.is_file():
                raise RuntimeError(f"relation-geometry sidecar array missing: {name}")
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            spec = arrays_meta[name]
            if np.dtype(array.dtype).str != str(spec.get("dtype")) or list(array.shape) != list(spec.get("shape", [])):
                raise RuntimeError(f"relation-geometry sidecar array shape/dtype mismatch: {name}")
            if verify_array_hashes and _sha256_file(path) != str(spec.get("sha256")):
                raise RuntimeError(f"relation-geometry sidecar array hash mismatch: {name}")
            if array.dtype.kind == "f" and not bool(np.isfinite(array).all()):
                raise RuntimeError(f"relation-geometry sidecar non-finite array: {name}")
            self.arrays[name] = array
        keys = self.arrays["sample_keys"]
        sample_offsets = self.arrays["sample_relation_offsets"]
        relation_offsets = self.arrays["relation_path_offsets"]
        relation_count = int(self.arrays["relation_row"].shape[0])
        path_count = int(self.arrays["path_cos_angle"].shape[0])
        if keys.ndim != 2 or keys.shape[1] != 32 or sample_offsets.shape != (keys.shape[0] + 1,):
            raise RuntimeError("relation-geometry sidecar sample offset shape mismatch")
        if relation_offsets.shape != (relation_count + 1,) or int(sample_offsets[-1]) != relation_count or int(relation_offsets[-1]) != path_count:
            raise RuntimeError("relation-geometry sidecar offset boundary mismatch")
        if not bool(np.all(sample_offsets[1:] >= sample_offsets[:-1])) or not bool(np.all(relation_offsets[1:] >= relation_offsets[:-1])):
            raise RuntimeError("relation-geometry sidecar offsets are not monotonic")
        if not bool(np.array_equal(self.arrays["relation_num_shortest_paths"].astype(np.int64), np.diff(relation_offsets))):
            raise RuntimeError("relation-geometry sidecar path multiplicity mismatch")
        relation_valid = self.arrays["relation_geometry_valid"]
        relation_codes = self.arrays["relation_invalid_reason_code"]
        path_valid = self.arrays["path_geometry_valid"]
        path_codes = self.arrays["path_invalid_reason_code"]
        if bool(np.any(relation_valid & (relation_codes != 0))) or bool(np.any((~relation_valid) & (relation_codes == 0))):
            raise RuntimeError("relation-geometry sidecar relation mask/reason mismatch")
        if bool(np.any(path_valid & (path_codes != 0))) or bool(np.any((~path_valid) & (path_codes == 0))):
            raise RuntimeError("relation-geometry sidecar path mask/reason mismatch")
        self._key_to_row = None

    def __len__(self):
        return int(self.arrays["sample_keys"].shape[0])

    @property
    def artifact_hash(self):
        return str(self.metadata["artifact_hash"])

    @property
    def cohort_hash(self):
        return str(self.metadata["cohort_hash"])

    def row_for_key(self, key, *, row_hint=None):
        key = bytes(key)
        if row_hint is not None and 0 <= int(row_hint) < len(self):
            candidate = bytes(np.asarray(self.arrays["sample_keys"][int(row_hint)], dtype=np.uint8).tobytes())
            if candidate == key:
                return self.row(int(row_hint))
        if self._key_to_row is None:
            # Downstream cohorts are small; for a million-row PI cohort the
            # Dataset supplies a row hint from the immutable cohort manifest.
            if len(self) > 200000:
                raise KeyError("sidecar lookup requires the immutable cohort row hint")
            self._key_to_row = {
                bytes(np.asarray(raw, dtype=np.uint8).tobytes()): index
                for index, raw in enumerate(self.arrays["sample_keys"])
            }
        try:
            return self.row(self._key_to_row[key])
        except KeyError as exc:
            raise KeyError(f"sample key is absent from relation-geometry sidecar: {key.hex()}") from exc

    def index_for_key(self, key, *, row_hint=None):
        key = bytes(key)
        if row_hint is not None and 0 <= int(row_hint) < len(self):
            candidate = bytes(np.asarray(self.arrays["sample_keys"][int(row_hint)], dtype=np.uint8).tobytes())
            if candidate == key:
                return int(row_hint)
        if self._key_to_row is None:
            if len(self) > 200000:
                raise KeyError("sidecar lookup requires the immutable cohort row hint")
            self._key_to_row = {
                bytes(np.asarray(raw, dtype=np.uint8).tobytes()): index
                for index, raw in enumerate(self.arrays["sample_keys"])
            }
        if key not in self._key_to_row:
            raise KeyError(f"sample key is absent from relation-geometry sidecar: {key.hex()}")
        return int(self._key_to_row[key])

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
            "relations": {
                **{
                    name: np.asarray(self.arrays[name][relation_start:relation_end])
                    for name in RELATION_GEOMETRY_ARRAY_NAMES
                    if name.startswith("relation_") and name != "relation_path_offsets"
                },
                "relation_path_offsets": np.asarray(
                    self.arrays["relation_path_offsets"][relation_start:relation_end + 1],
                    dtype=np.int64,
                ) - path_start,
            },
            "paths": {
                name: np.asarray(self.arrays[name][path_start:path_end])
                for name in RELATION_GEOMETRY_ARRAY_NAMES if name.startswith("path_")
            },
        }


class RelationGeometryPermutation:
    """Read-only relation-index permutation bound to one sidecar artifact."""

    SCHEMA = "mts-relation-geometry-permutation-v1"

    def __init__(
        self, root, *, source_sidecar=None, expected_cohort_hash=None,
        expected_artifact_hash=None,
    ):
        self.root = Path(root).resolve()
        metadata_path = self.root / "metadata.json"
        if not metadata_path.is_file() or not (self.root / ".done").is_file() or not (self.root / ".frozen").is_file():
            raise RuntimeError(f"G3 permutation artifact is incomplete: {self.root}")
        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if self.metadata.get("schema") != self.SCHEMA:
            raise RuntimeError("unsupported G3 permutation schema")
        artifact = str(self.metadata.get("artifact_hash", ""))
        if expected_artifact_hash is not None and artifact != str(expected_artifact_hash):
            raise RuntimeError("G3 permutation artifact hash mismatch")
        if (self.root / ".done").read_text(encoding="utf-8").strip() != artifact:
            raise RuntimeError("G3 permutation .done binding mismatch")
        expected_artifact = hashlib.sha256(
            json.dumps(
                {key: value for key, value in self.metadata.items() if key != "artifact_hash"},
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
        if expected_artifact != artifact:
            raise RuntimeError("G3 permutation metadata hash mismatch")
        frozen = json.loads((self.root / ".frozen").read_text(encoding="utf-8"))
        if frozen.get("schema") != self.SCHEMA or frozen.get("artifact_hash") != artifact:
            raise RuntimeError("G3 permutation .frozen binding mismatch")
        if expected_cohort_hash is not None and str(self.metadata.get("cohort_hash")) != str(expected_cohort_hash):
            raise RuntimeError("G3 permutation cohort hash mismatch")
        if source_sidecar is not None and str(self.metadata.get("source_sidecar_artifact")) != str(source_sidecar):
            raise RuntimeError("G3 permutation source sidecar mismatch")
        path = self.root / "relation_permutation.npy"
        if not path.is_file():
            raise RuntimeError("G3 permutation payload is missing")
        self.relation_permutation = np.load(path, mmap_mode="r", allow_pickle=False)
        if self.relation_permutation.ndim != 1 or np.dtype(self.relation_permutation.dtype).kind not in "iu":
            raise RuntimeError("G3 permutation payload shape/dtype mismatch")
        spec = self.metadata.get("array", {})
        if list(self.relation_permutation.shape) != list(spec.get("shape", [])) or _sha256_file(path) != str(spec.get("sha256")):
            raise RuntimeError("G3 permutation payload hash mismatch")
        relation_count = int(self.metadata.get("relation_count", -1))
        if relation_count != int(self.relation_permutation.size):
            raise RuntimeError("G3 permutation relation count mismatch")
        if relation_count:
            if int(self.relation_permutation.min()) < 0 or int(self.relation_permutation.max()) >= relation_count:
                raise RuntimeError("G3 permutation payload index is out of bounds")
            if not np.array_equal(np.sort(np.asarray(self.relation_permutation)), np.arange(relation_count, dtype=np.int64)):
                raise RuntimeError("G3 permutation payload is not bijective")

    def permutation_for(self, relation_indices):
        indices = np.asarray(relation_indices, dtype=np.int64)
        if indices.size and (int(indices.min()) < 0 or int(indices.max()) >= self.relation_permutation.size):
            raise IndexError("G3 permutation relation index is out of bounds")
        return np.asarray(self.relation_permutation[indices], dtype=np.int64)


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


def enumerate_two_edge_paths(
    source: tuple[int, int],
    target: tuple[int, int],
    internal_edges: Sequence[tuple[int, int]],
    left: int,
    right: int,
):
    """Enumerate all distinct state sequences of exactly two real edges.

    Internal edges stay in one RU.  The two attachment edges lift the RU
    image by one.  A set removes only duplicate state sequences; distinct
    source shifts and multiplicity therefore remain distinct paths.
    """

    def neighbours(state):
        atom, shift = int(state[0]), int(state[1])
        for begin, end in internal_edges:
            begin, end = int(begin), int(end)
            if atom == begin:
                yield (end, shift)
            elif atom == end:
                yield (begin, shift)
        if atom == int(right):
            yield (int(left), shift + 1)
        if atom == int(left):
            yield (int(right), shift - 1)

    source = (int(source[0]), int(source[1]))
    target = (int(target[0]), int(target[1]))
    paths = []
    seen = set()
    for middle in neighbours(source):
        for observed_target in neighbours(middle):
            if observed_target != target:
                continue
            path = (source, tuple(middle), target)
            if path not in seen:
                seen.add(path)
                paths.append(path)
    return paths


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
    """Validate the fields needed for lifted topology path enumeration."""

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


def map_state_to_trimer(state, state_to_local: Mapping[tuple[int, int], int]):
    return state_to_local.get((int(state[0]), int(state[1])))


def path_geometry(path, trimer_info):
    """Return finite geometry and real bond types for one lifted path."""

    local = [map_state_to_trimer(state, trimer_info["state_to_local"]) for state in path]
    if any(index is None for index in local):
        return None, "ru_offset_uncovered"
    source, middle, target = (int(value) for value in local)
    pair_left = (min(source, middle), max(source, middle))
    pair_right = (min(middle, target), max(middle, target))
    left_codes = trimer_info["bonds"].get(pair_left)
    right_codes = trimer_info["bonds"].get(pair_right)
    if not left_codes or not right_codes:
        return None, "real_bond_missing"
    if len(left_codes) != 1 or len(right_codes) != 1:
        return None, "real_bond_ambiguous"
    positions = trimer_info["positions"]
    source_pos, middle_pos, target_pos = positions[source], positions[middle], positions[target]
    vector_left = source_pos - middle_pos
    vector_right = target_pos - middle_pos
    distance = torch.linalg.vector_norm(source_pos - target_pos)
    denominator = torch.linalg.vector_norm(vector_left) * torch.linalg.vector_norm(vector_right)
    if (
        not bool(torch.isfinite(distance))
        or not bool(torch.isfinite(denominator))
        or float(denominator) <= 1e-8
    ):
        return None, "degenerate_geometry"
    cosine = torch.clamp(torch.dot(vector_left, vector_right) / denominator, -1.0, 1.0)
    if not bool(torch.isfinite(cosine)):
        return None, "nonfinite_geometry"
    return {
        "distance": float(distance.item()),
        "cosine": float(cosine.item()),
        "bond_types": (next(iter(left_codes)), next(iter(right_codes))),
        "source_local": source,
        "middle_local": middle,
        "target_local": target,
    }, None


def _endpoint_distance(path, trimer_info):
    """Compute endpoint distance even when the path's real bonds are invalid."""

    if trimer_info is None:
        return None, "geometry_invalid"
    source = map_state_to_trimer(path[0], trimer_info["state_to_local"])
    target = map_state_to_trimer(path[-1], trimer_info["state_to_local"])
    if source is None or target is None:
        return None, "ru_offset_uncovered"
    positions = trimer_info["positions"]
    distance = torch.linalg.vector_norm(positions[int(source)] - positions[int(target)])
    if not bool(torch.isfinite(distance)):
        return None, "nonfinite_geometry"
    return float(distance.item()), None


def _empty_arrays():
    return {
        "relation_row": [],
        "relation_source_canonical_id": [],
        "relation_target_canonical_id": [],
        "relation_signed_source_shift": [],
        "relation_geometry_valid": [],
        "relation_invalid_reason_code": [],
        "relation_num_shortest_paths": [],
        "relation_path_offsets": [0],
        "relation_endpoint_distance": [],
        "path_intermediate_canonical_id": [],
        "path_intermediate_ru_shift": [],
        "path_source_trimer_atom_id": [],
        "path_intermediate_trimer_atom_id": [],
        "path_target_trimer_atom_id": [],
        "path_bond_type_left": [],
        "path_bond_type_right": [],
        "path_geometry_valid": [],
        "path_invalid_reason_code": [],
        "path_cos_angle": [],
    }


def build_sample_record(key: bytes, topology, trimer):
    """Build one sample's relation/path payload without writing anything.

    The payload retains every topology shortest path.  Geometry-invalid
    records use explicit integer/finite sentinels plus masks and reason codes;
    no 2-D fallback coordinate can enter a valid geometry value.
    """

    arrays = _empty_arrays()
    prepared, topology_reason = prepare_topology(topology)
    trimer_info, trimer_reason = prepare_trimer(trimer, topology)
    graph_available = _as_bool(getattr(topology, "graph_available", False))
    sample_reasons = []
    if not graph_available:
        sample_reasons.append("graph_unavailable")
    if topology_reason:
        sample_reasons.append(topology_reason)
    if trimer_reason:
        sample_reasons.append(trimer_reason)
    if prepared is None:
        return {
            "sample_key": bytes(key),
            "arrays": arrays,
            "sample_reasons": sample_reasons,
            "graph_available": graph_available,
            "geometry_3d": bool(trimer_info is not None),
        }
    rows = torch.nonzero(prepared["spd"] == 2, as_tuple=False).flatten().tolist()
    for row in rows:
        row = int(row)
        source_id = int(prepared["edge"][0, row])
        target_id = int(prepared["edge"][1, row])
        signed_shift = int(prepared["shift"][row])
        source = (source_id, signed_shift)
        target = (target_id, 0)
        paths = []
        relation_reason = None
        if not graph_available:
            relation_reason = "graph_unavailable"
        elif topology_reason:
            relation_reason = topology_reason
        else:
            paths = enumerate_two_edge_paths(
                source,
                target,
                prepared["internal_edges"],
                prepared["left"],
                prepared["right"],
            )
            if not paths:
                relation_reason = "no_two_edge_paths"
        relation_path_start = len(arrays["path_intermediate_canonical_id"])
        path_valid_count = 0
        path_invalid_reasons = []
        endpoint_distance = 0.0
        endpoint_distance_reason = None
        if paths and trimer_info is not None:
            endpoint_distance, endpoint_distance_reason = _endpoint_distance(paths[0], trimer_info)
            if endpoint_distance_reason:
                endpoint_distance = 0.0
        elif paths:
            endpoint_distance_reason = trimer_reason or "geometry_invalid"
        for path in paths:
            middle_id, middle_shift = int(path[1][0]), int(path[1][1])
            local_ids = [-1, -1, -1]
            bond_types = [-1, -1]
            path_reason = trimer_reason or relation_reason
            geometry = None
            if trimer_info is not None:
                local = [map_state_to_trimer(state, trimer_info["state_to_local"]) for state in path]
                local_ids = [int(item) if item is not None else -1 for item in local]
                geometry, path_reason = path_geometry(path, trimer_info)
                if geometry is not None:
                    bond_types = [int(geometry["bond_types"][0]), int(geometry["bond_types"][1])]
                    path_valid_count += 1
                    path_reason = None
            if path_reason is None and geometry is None:
                path_reason = "no_valid_geometry"
            if path_reason:
                path_invalid_reasons.append(path_reason)
            arrays["path_intermediate_canonical_id"].append(middle_id)
            arrays["path_intermediate_ru_shift"].append(middle_shift)
            arrays["path_source_trimer_atom_id"].append(local_ids[0])
            arrays["path_intermediate_trimer_atom_id"].append(local_ids[1])
            arrays["path_target_trimer_atom_id"].append(local_ids[2])
            arrays["path_bond_type_left"].append(bond_types[0])
            arrays["path_bond_type_right"].append(bond_types[1])
            arrays["path_geometry_valid"].append(bool(geometry is not None))
            arrays["path_invalid_reason_code"].append(reason_code(path_reason))
            arrays["path_cos_angle"].append(float(geometry["cosine"]) if geometry is not None else 0.0)
        relation_path_end = len(arrays["path_intermediate_canonical_id"])
        arrays["relation_row"].append(row)
        arrays["relation_source_canonical_id"].append(source_id)
        arrays["relation_target_canonical_id"].append(target_id)
        arrays["relation_signed_source_shift"].append(signed_shift)
        arrays["relation_num_shortest_paths"].append(len(paths))
        arrays["relation_path_offsets"].append(relation_path_end)
        arrays["relation_endpoint_distance"].append(float(endpoint_distance))
        valid = bool(
            paths
            and trimer_info is not None
            and path_valid_count == len(paths)
            and endpoint_distance_reason is None
        )
        if valid:
            relation_reason = None
        elif relation_reason is None:
            if path_invalid_reasons:
                relation_reason = (
                    "partial_path_geometry"
                    if path_valid_count and path_valid_count < len(paths)
                    else path_invalid_reasons[0]
                )
            elif endpoint_distance_reason:
                relation_reason = "endpoint_distance_invalid"
            else:
                relation_reason = "no_valid_geometry"
        arrays["relation_geometry_valid"].append(valid)
        arrays["relation_invalid_reason_code"].append(reason_code(relation_reason))
    arrays["relation_path_offsets"] = list(arrays["relation_path_offsets"])
    return {
        "sample_key": bytes(key),
        "arrays": arrays,
        "sample_reasons": sample_reasons,
        "graph_available": graph_available,
        "geometry_3d": bool(trimer_info is not None),
    }


def array_dtypes():
    """Return the canonical dtype for each persisted array."""

    return {
        "relation_row": np.dtype("<i4"),
        "relation_source_canonical_id": np.dtype("<i4"),
        "relation_target_canonical_id": np.dtype("<i4"),
        "relation_signed_source_shift": np.dtype("<i2"),
        "relation_geometry_valid": np.dtype("<?"),
        "relation_invalid_reason_code": np.dtype("<i2"),
        "relation_num_shortest_paths": np.dtype("<i4"),
        "relation_path_offsets": np.dtype("<i8"),
        "relation_endpoint_distance": np.dtype("<f4"),
        "path_intermediate_canonical_id": np.dtype("<i4"),
        "path_intermediate_ru_shift": np.dtype("<i2"),
        "path_source_trimer_atom_id": np.dtype("<i4"),
        "path_intermediate_trimer_atom_id": np.dtype("<i4"),
        "path_target_trimer_atom_id": np.dtype("<i4"),
        "path_bond_type_left": np.dtype("<i2"),
        "path_bond_type_right": np.dtype("<i2"),
        "path_geometry_valid": np.dtype("<?"),
        "path_invalid_reason_code": np.dtype("<i2"),
        "path_cos_angle": np.dtype("<f4"),
    }


__all__ = [
    "CACHE_RELATION_GEOMETRY_SCHEMA",
    "RELATION_GEOMETRY_BUILDER_VERSION",
    "INVALID_REASON_CODES",
    "CODE_TO_REASON",
    "REASON_TO_CODE",
    "RELATION_GEOMETRY_ARRAY_NAMES",
    "RelationGeometrySidecar",
    "RelationGeometryPermutation",
    "array_dtypes",
    "build_sample_record",
    "enumerate_two_edge_paths",
    "map_state_to_trimer",
    "path_geometry",
    "prepare_topology",
    "prepare_trimer",
    "reason_code",
]
