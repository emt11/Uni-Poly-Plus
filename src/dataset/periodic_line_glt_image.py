"""Center-anchored periodic line graph records for MTS-GLT-v3.

Unlike :mod:`periodic_line_glt`, this schema stores exactly one physical
distance for every canonical bond and exactly one physical angle for every
directed source-image -> center-target relation.  It never computes moments
over translated Trimer copies and never mutates the topology or geometry
objects supplied by the frozen caches.
"""

from __future__ import annotations

from collections import defaultdict
import json
import math
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem

from .graph_data import build_periodic_multimer_mol
from .periodic_line_glt import (
    MAX_ATOMIC_NUMBER,
    _internal_bonds,
    _prepare_topology,
    _prepare_trimer,
    canonical_line_token,
)


SIDECAR_SCHEMA = "mts-periodic-line-glt-image-v1"
BUILDER_VERSION = 1

BOND_TYPE_OTHER = 4
BOND_TYPE_MASK = 5
NUM_BOND_TYPE_TARGETS = 5

STEREO_VALUES = (
    Chem.rdchem.BondStereo.STEREONONE,
    Chem.rdchem.BondStereo.STEREOANY,
    Chem.rdchem.BondStereo.STEREOZ,
    Chem.rdchem.BondStereo.STEREOE,
    Chem.rdchem.BondStereo.STEREOCIS,
    Chem.rdchem.BondStereo.STEREOTRANS,
)
STEREO_TO_INDEX = {value: index for index, value in enumerate(STEREO_VALUES)}
STEREO_MASK = len(STEREO_VALUES)


def bond_type_index(bond: Chem.Bond) -> int:
    value = bond.GetBondType()
    return {
        Chem.rdchem.BondType.SINGLE: 0,
        Chem.rdchem.BondType.DOUBLE: 1,
        Chem.rdchem.BondType.TRIPLE: 2,
        Chem.rdchem.BondType.AROMATIC: 3,
    }.get(value, BOND_TYPE_OTHER)


def bond_stereo_index(bond: Chem.Bond) -> int:
    return STEREO_TO_INDEX.get(bond.GetStereo(), 0)


def _anchor_instance(token):
    """Choose one in-Trimer instance, preferring endpoint A in center RU."""
    u, v, shift = (int(value) for value in token)
    candidates = [
        (u, translation, v, translation + shift)
        for translation in range(-1, 2)
        if -1 <= translation + shift <= 1
        and (translation == 0 or translation + shift == 0)
    ]
    if not candidates:
        raise ValueError(f"canonical line has no center-anchored image: {token}")
    return min(
        candidates,
        key=lambda item: (
            0 if item[1] == 0 else 1,
            abs(item[1]) + abs(item[3]),
            item[1], item[3], item[0], item[2],
        ),
    )


def _physical_bond_chemistry(smiles: str):
    """Return bond chemistry keyed by ``(base_a, ru_a, base_b, ru_b)``."""
    molecule, metadata = build_periodic_multimer_mol(
        smiles, num_repeat_units=3, close_periodic=False
    )
    base_count = int(metadata["base_atom_count"])
    chemistry = {}
    for bond in molecule.GetBonds():
        a, b = int(bond.GetBeginAtomIdx()), int(bond.GetEndAtomIdx())
        state_a, state_b = (a % base_count, a // base_count - 1), (
            b % base_count, b // base_count - 1
        )
        key = tuple(sorted((state_a, state_b)))
        chemistry[key] = (
            bond_type_index(bond),
            bond_stereo_index(bond),
            int(bool(bond.GetIsConjugated())),
        )
    return chemistry


def _distance(info, topology, instance):
    u, q_u, v, q_v = instance
    mapping = torch.as_tensor(topology.canonical_to_trimer_base_atom_id).long()
    left = info["state_to_local"].get((int(mapping[u]), int(q_u)))
    right = info["state_to_local"].get((int(mapping[v]), int(q_v)))
    if left is None or right is None:
        return None
    value = torch.linalg.vector_norm(info["positions"][left] - info["positions"][right])
    return float(value) if bool(torch.isfinite(value)) and float(value) > 0 else None


def _angle(info, topology, outer_a, center, outer_b):
    mapping = torch.as_tensor(topology.canonical_to_trimer_base_atom_id).long()
    indices = [
        info["state_to_local"].get((int(mapping[atom]), int(shift)))
        for atom, shift in (outer_a, center, outer_b)
    ]
    if any(value is None for value in indices):
        return None
    p_a, p_c, p_b = (info["positions"][value] for value in indices)
    first, second = p_a - p_c, p_b - p_c
    denominator = torch.linalg.vector_norm(first) * torch.linalg.vector_norm(second)
    if not bool(torch.isfinite(denominator)) or float(denominator) <= 0:
        return None
    cosine = torch.clamp(torch.dot(first, second) / denominator, -1.0, 1.0)
    return float(torch.acos(cosine))


def build_periodic_line_image_sample(topology, trimer, smiles: str):
    """Build one v3 sidecar row from existing O8 and frozen Trimer records."""
    topology_info, topology_error = _prepare_topology(topology)
    trimer_info, trimer_error = _prepare_trimer(trimer, topology)
    if topology_info is None or trimer_info is None:
        return empty_image_row(topology_error or trimer_error or "invalid")

    bonds = list(_internal_bonds(topology))
    bonds.append((topology_info["right"], 0, topology_info["left"], 1, None))
    token_types = {}
    for u, q_u, v, q_v, bond_type in bonds:
        token = canonical_line_token(u, q_u, v, q_v)
        token_types[token] = bond_type
    tokens = sorted(token_types)
    token_index = {token: index for index, token in enumerate(tokens)}
    chemistry = _physical_bond_chemistry(str(smiles))
    canonical_to_base = torch.as_tensor(
        topology.canonical_to_trimer_base_atom_id
    ).long()

    output = {
        "token_atom_a": [], "token_atom_b": [], "token_shift": [],
        "token_endpoint_z_a": [], "token_endpoint_z_b": [],
        "token_distance": [], "token_bond_type": [], "token_stereo": [],
        "token_conjugated": [], "token_anchor_q_a": [], "token_anchor_q_b": [],
        "token_valid": [],
    }
    anchors = []
    z = topology_info["z"]
    for token in tokens:
        instance = _anchor_instance(token)
        anchors.append(instance)
        u, q_u, v, q_v = instance
        base_a, base_b = int(canonical_to_base[u]), int(canonical_to_base[v])
        key = tuple(sorted(((base_a, q_u), (base_b, q_v))))
        attrs = chemistry.get(key)
        distance = _distance(trimer_info, topology, instance)
        output["token_atom_a"].append(token[0])
        output["token_atom_b"].append(token[1])
        output["token_shift"].append(token[2])
        clean_z_a, clean_z_b = int(z[token[0]]), int(z[token[1]])
        output["token_endpoint_z_a"].append(
            clean_z_a if 1 <= clean_z_a <= MAX_ATOMIC_NUMBER else 0
        )
        output["token_endpoint_z_b"].append(
            clean_z_b if 1 <= clean_z_b <= MAX_ATOMIC_NUMBER else 0
        )
        output["token_distance"].append(0.0 if distance is None else distance)
        output["token_bond_type"].append(0 if attrs is None else attrs[0])
        output["token_stereo"].append(0 if attrs is None else attrs[1])
        output["token_conjugated"].append(0 if attrs is None else attrs[2])
        output["token_anchor_q_a"].append(q_u)
        output["token_anchor_q_b"].append(q_v)
        output["token_valid"].append(distance is not None and attrs is not None)

    # Enumerate every physical occurrence of every canonical line token.
    incident = defaultdict(list)
    for token_id, token in enumerate(tokens):
        u, v, shift = token
        for q_u in range(-1, 2):
            q_v = q_u + shift
            if not -1 <= q_v <= 1:
                continue
            instance = (u, q_u, v, q_v)
            if _distance(trimer_info, topology, instance) is None:
                continue
            incident[(u, q_u)].append((token_id, (v, q_v), q_u))
            incident[(v, q_v)].append((token_id, (u, q_u), q_u))

    relations = {
        "relation_source": [], "relation_target": [],
        "relation_center_atom": [], "relation_source_image_shift": [],
        "relation_angle": [], "relation_valid": [],
    }
    seen = set()
    for target_id, anchor in enumerate(anchors):
        u, q_u, v, q_v = anchor
        for center, outer_target in (((u, q_u), (v, q_v)), ((v, q_v), (u, q_u))):
            for source_id, outer_source, source_translation in incident[center]:
                if source_id == target_id:
                    continue
                key = (source_id, target_id, center, outer_source, outer_target)
                if key in seen:
                    continue
                seen.add(key)
                angle = _angle(trimer_info, topology, outer_source, center, outer_target)
                relations["relation_source"].append(source_id)
                relations["relation_target"].append(target_id)
                relations["relation_center_atom"].append(center[0])
                relations["relation_source_image_shift"].append(
                    int(source_translation - center[1])
                )
                relations["relation_angle"].append(0.0 if angle is None else angle)
                relations["relation_valid"].append(angle is not None)

    return {
        "schema": SIDECAR_SCHEMA,
        "geometry_valid": bool(output["token_valid"]) and all(output["token_valid"]),
        "invalid_reason": "",
        "tokens": {name: np.asarray(value) for name, value in output.items()},
        "relations": {name: np.asarray(value) for name, value in relations.items()},
    }


def empty_image_row(reason="invalid"):
    token_int = (
        "token_atom_a", "token_atom_b", "token_shift", "token_endpoint_z_a",
        "token_endpoint_z_b", "token_bond_type", "token_stereo",
        "token_conjugated", "token_anchor_q_a", "token_anchor_q_b",
    )
    relation_int = (
        "relation_source", "relation_target", "relation_center_atom",
        "relation_source_image_shift",
    )
    return {
        "schema": SIDECAR_SCHEMA, "geometry_valid": False,
        "invalid_reason": str(reason),
        "tokens": {
            **{name: np.empty(0, dtype=np.int64) for name in token_int},
            "token_distance": np.empty(0, dtype=np.float32),
            "token_valid": np.empty(0, dtype=bool),
        },
        "relations": {
            **{name: np.empty(0, dtype=np.int64) for name in relation_int},
            "relation_angle": np.empty(0, dtype=np.float32),
            "relation_valid": np.empty(0, dtype=bool),
        },
    }


class PeriodicLineImageSidecar:
    """Read a packed v3 sidecar written by ``write_image_sidecar``."""

    def __init__(self, root):
        self.root = Path(root)
        self.metadata = json.loads((self.root / "metadata.json").read_text())
        if self.metadata.get("schema") != SIDECAR_SCHEMA:
            raise ValueError("periodic line image sidecar schema mismatch")
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

    def __len__(self):
        return int(self.sample_keys.shape[0])

    def index_for_key(self, key, row_hint=None):
        key = bytes(key)
        if row_hint is not None and bytes(self.sample_keys[int(row_hint)]) == key:
            return int(row_hint)
        return self._key_to_index[key]

    def model_row(self, index):
        index = int(index)
        t0, t1 = self.token_offsets[index:index + 2]
        r0, r1 = self.relation_offsets[index:index + 2]
        tokens = {name: value[int(t0):int(t1)] for name, value in self._arrays.items() if name.startswith("token_")}
        relations = {name: value[int(r0):int(r1)] for name, value in self._arrays.items() if name.startswith("relation_")}
        return {"geometry_valid": bool(self.geometry_valid[index]), "tokens": tokens, "relations": relations}


def write_image_sidecar(root, sample_keys, rows):
    """Write a new packed sidecar directory; existing destinations are refused."""
    root = Path(root)
    if root.exists():
        raise FileExistsError(f"refusing to overwrite sidecar: {root}")
    root.mkdir(parents=True)
    token_names = tuple(rows[0]["tokens"]) if rows else tuple(empty_image_row()["tokens"])
    relation_names = tuple(rows[0]["relations"]) if rows else tuple(empty_image_row()["relations"])
    token_offsets, relation_offsets = [0], [0]
    for row in rows:
        token_offsets.append(token_offsets[-1] + len(row["tokens"]["token_atom_a"]))
        relation_offsets.append(relation_offsets[-1] + len(row["relations"]["relation_source"]))
    keys = np.asarray([np.frombuffer(bytes(key), dtype=np.uint8) for key in sample_keys], dtype=np.uint8)
    np.save(root / "sample_keys.npy", keys)
    np.save(root / "token_offsets.npy", np.asarray(token_offsets, dtype=np.int64))
    np.save(root / "relation_offsets.npy", np.asarray(relation_offsets, dtype=np.int64))
    np.save(root / "geometry_valid.npy", np.asarray([row["geometry_valid"] for row in rows], dtype=bool))
    for name in token_names:
        np.save(root / f"{name}.npy", np.concatenate([np.asarray(row["tokens"][name]) for row in rows]) if rows else np.asarray(empty_image_row()["tokens"][name]))
    for name in relation_names:
        np.save(root / f"{name}.npy", np.concatenate([np.asarray(row["relations"][name]) for row in rows]) if rows else np.asarray(empty_image_row()["relations"][name]))
    (root / "metadata.json").write_text(json.dumps({"schema": SIDECAR_SCHEMA, "builder_version": BUILDER_VERSION, "sample_count": len(rows)}, indent=2, sort_keys=True) + "\n")
    (root / ".done").write_text("complete\n")


__all__ = [
    "SIDECAR_SCHEMA", "BOND_TYPE_MASK", "NUM_BOND_TYPE_TARGETS",
    "STEREO_MASK", "build_periodic_line_image_sample",
    "PeriodicLineImageSidecar", "write_image_sidecar",
]
