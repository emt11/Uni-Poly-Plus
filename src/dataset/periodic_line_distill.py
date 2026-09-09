"""N+1/N+2 center-RU line graphs derived from immutable GLT sidecars."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .periodic_line_glt import PeriodicLineGLTSidecar
from .periodic_line_glt_image import PeriodicLineImageSidecar


SCHEMA_PREFIX = "mts-periodic-line-distill-v1-"


class PeriodicLineDistillSidecar(PeriodicLineImageSidecar):
    def __init__(self, root):
        self.root = Path(root)
        self.metadata = json.loads((self.root / "metadata.json").read_text())
        if self.metadata.get("schema") not in {
            SCHEMA_PREFIX + "n_plus_1", SCHEMA_PREFIX + "n_plus_2",
        }:
            raise ValueError("periodic line distillation sidecar schema mismatch")
        self.sample_keys = np.load(self.root / "sample_keys.npy", mmap_mode="r")
        self.token_offsets = np.load(self.root / "token_offsets.npy", mmap_mode="r")
        self.relation_offsets = np.load(self.root / "relation_offsets.npy", mmap_mode="r")
        self.geometry_valid = np.load(self.root / "geometry_valid.npy", mmap_mode="r")
        self._arrays = {
            path.stem: np.load(path, mmap_mode="r")
            for path in self.root.glob("*.npy")
            if path.stem not in {"sample_keys", "token_offsets", "relation_offsets", "geometry_valid"}
        }
        self._key_to_index = {
            bytes(row): index for index, row in enumerate(self.sample_keys)
        }


def _side(q_a: int, q_b: int) -> int:
    if min(q_a, q_b) < 0 and max(q_a, q_b) == 0:
        return -1
    if min(q_a, q_b) == 0 and max(q_a, q_b) > 0:
        return 1
    raise ValueError(f"cross-RU instance is not center anchored: {(q_a, q_b)}")


def transform_row(image, moments, version: str):
    """Transform one aligned v1/image-v1 record without touching coordinates."""
    if version not in {"n_plus_1", "n_plus_2"}:
        raise ValueError(version)
    token = image["tokens"]
    old = moments["tokens"]
    count = len(token["token_atom_a"])
    if not image["geometry_valid"] or not count:
        return _empty_row()
    for name in ("token_atom_a", "token_atom_b", "token_shift"):
        if not np.array_equal(token[name], old[name]):
            raise ValueError(f"aligned GLT sidecars disagree on {name}")
    cross = np.flatnonzero(np.asarray(token["token_shift"]) != 0)
    internal = np.flatnonzero(np.asarray(token["token_shift"]) == 0)
    if len(cross) != 1 or not len(internal):
        return _empty_row()
    cross_id = int(cross[0])
    shift = int(token["token_shift"][cross_id])
    observations = np.asarray(old["token_observation_distances"][cross_id])
    observation_count = int(old["token_observation_count"][cross_id])
    instances = [
        (q, q + shift)
        for q in range(-1, 2)
        if -1 <= q + shift <= 1 and (q == 0 or q + shift == 0)
    ]
    if observation_count != 2 or len(instances) != 2:
        return _empty_row()
    cross_distance = {_side(*instance): float(observations[i]) for i, instance in enumerate(instances)}
    if set(cross_distance) != {-1, 1} or not all(np.isfinite(list(cross_distance.values()))):
        return _empty_row()

    source_ids = list(map(int, internal))
    old_to_new = {old_id: new_id for new_id, old_id in enumerate(source_ids)}
    token_sources = list(source_ids)
    roles = [0] * len(source_ids)
    if version == "n_plus_1":
        shared = len(token_sources)
        token_sources.append(cross_id)
        roles.append(2)
        cross_map = {-1: shared, 1: shared}
    else:
        left = len(token_sources)
        right = left + 1
        token_sources.extend([cross_id, cross_id])
        roles.extend([-1, 1])
        cross_map = {-1: left, 1: right}

    output = {}
    copy_names = (
        "token_atom_a", "token_atom_b", "token_endpoint_z_a", "token_endpoint_z_b",
        "token_bond_type", "token_stereo", "token_conjugated", "token_valid",
    )
    for name in copy_names:
        output[name] = np.asarray(token[name])[token_sources]
    output["token_shift"] = np.asarray(roles, dtype=np.int16)
    output["token_center_internal"] = np.asarray([role == 0 for role in roles], dtype=bool)
    distances = [float(token["token_distance"][old_id]) for old_id in source_ids]
    distances.extend(
        [0.5 * (cross_distance[-1] + cross_distance[1])]
        if version == "n_plus_1" else [cross_distance[-1], cross_distance[1]]
    )
    output["token_distance"] = np.asarray(distances, dtype=np.float32)
    output["token_anchor_q_a"] = np.zeros(len(token_sources), dtype=np.int8)
    output["token_anchor_q_b"] = np.asarray(roles, dtype=np.int8)

    relations = image["relations"]
    records = []
    for idx in range(len(relations["relation_source"])):
        if not bool(relations["relation_valid"][idx]):
            continue
        source_old = int(relations["relation_source"][idx])
        target_old = int(relations["relation_target"][idx])
        center = int(relations["relation_center_atom"][idx])
        # Image-v1 targets have one explicit anchor.  Retain only relations
        # whose shared atom is physically in RU(0), then add the reverse edge.
        # Internal target anchors are always in RU(0).  Cross-target rows can
        # be peripheral and become ambiguous when both cross endpoints share
        # the same canonical atom.  Drop them here: the exact reverse edge is
        # reconstructed below from every center-internal target row.
        if target_old == cross_id:
            continue
        target_new = old_to_new[target_old]
        target_physical = 0
        if source_old == cross_id:
            q_a = int(relations["relation_source_image_shift"][idx])
            source_side = _side(q_a, q_a + shift)
            source_new = cross_map[source_side]
            source_physical = source_side
        else:
            source_new = old_to_new[source_old]
            source_physical = 0
        angle = float(relations["relation_angle"][idx])
        physical_tag = (source_physical, target_physical)
        records.append((source_new, target_new, center, angle, physical_tag))
        records.append((target_new, source_new, center, angle, physical_tag[::-1]))

    # Existing image rows may already contain the reverse direction.  Remove
    # only the same physical directed triplet; N+1 keeps left/right multiplicity.
    unique = {}
    for source, target, center, angle, physical in records:
        key = (source, target, center, physical)
        previous = unique.get(key)
        if previous is not None and abs(previous - angle) > 5e-4:
            raise ValueError("duplicate physical angle identity disagrees")
        unique.setdefault(key, angle)
    ordered = sorted(unique.items())
    relation_output = {
        "relation_source": np.asarray([key[0] for key, _ in ordered], dtype=np.int32),
        "relation_target": np.asarray([key[1] for key, _ in ordered], dtype=np.int32),
        "relation_center_atom": np.asarray([key[2] for key, _ in ordered], dtype=np.int32),
        "relation_source_image_shift": np.asarray([key[3][0] for key, _ in ordered], dtype=np.int16),
        "relation_angle": np.asarray([value for _, value in ordered], dtype=np.float32),
        "relation_valid": np.ones(len(ordered), dtype=bool),
    }
    valid = bool(output["token_valid"].all() and len(internal))
    return {"geometry_valid": valid, "tokens": output, "relations": relation_output}


def _empty_row():
    token_int = (
        "token_atom_a", "token_atom_b", "token_shift", "token_endpoint_z_a",
        "token_endpoint_z_b", "token_bond_type", "token_stereo", "token_conjugated",
        "token_anchor_q_a", "token_anchor_q_b",
    )
    relation_int = (
        "relation_source", "relation_target", "relation_center_atom",
        "relation_source_image_shift",
    )
    return {
        "geometry_valid": False,
        "tokens": {
            **{name: np.empty(0, dtype=np.int64) for name in token_int},
            "token_distance": np.empty(0, dtype=np.float32),
            "token_valid": np.empty(0, dtype=bool),
            "token_center_internal": np.empty(0, dtype=bool),
        },
        "relations": {
            **{name: np.empty(0, dtype=np.int64) for name in relation_int},
            "relation_angle": np.empty(0, dtype=np.float32),
            "relation_valid": np.empty(0, dtype=bool),
        },
    }


__all__ = ["PeriodicLineDistillSidecar", "SCHEMA_PREFIX", "transform_row"]
