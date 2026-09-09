"""Geometry-revision-2 N+1/N+2 line graphs built from frozen Trimers.

Relations are enumerated from distinct physical bonds incident on an atom in
the center repeat unit.  Canonical state equality never removes a physical
relation; this is important for polymers whose two connection sites are the
same atom.
"""

from __future__ import annotations

import numpy as np
import torch

from .periodic_line_glt import (
    MAX_ATOMIC_NUMBER, _angle, _distance, _internal_bonds,
    _prepare_topology, _prepare_trimer, canonical_line_token,
)
from .periodic_line_glt_image import _physical_bond_chemistry
from .periodic_line_distill import _empty_row


SCHEMA_PREFIX = "mts-periodic-line-distill-v2-"
GEOMETRY_REVISION = 2


def _chemistry(chemistry, topology, u, q_u, v, q_v):
    mapping = torch.as_tensor(topology.canonical_to_trimer_base_atom_id).long()
    key = tuple(sorted(((int(mapping[u]), int(q_u)), (int(mapping[v]), int(q_v)))))
    return chemistry.get(key)


def _token_row(topology, atom_a, atom_b, role, distance, chemistry, anchor):
    z = torch.as_tensor(getattr(topology, "atomic_numbers", topology.z)).long()
    z_a, z_b = int(z[atom_a]), int(z[atom_b])
    return {
        "token_atom_a": atom_a, "token_atom_b": atom_b,
        "token_shift": role,
        "token_endpoint_z_a": z_a if 1 <= z_a <= MAX_ATOMIC_NUMBER else 0,
        "token_endpoint_z_b": z_b if 1 <= z_b <= MAX_ATOMIC_NUMBER else 0,
        "token_distance": distance,
        "token_bond_type": chemistry[0], "token_stereo": chemistry[1],
        "token_conjugated": chemistry[2],
        "token_anchor_q_a": anchor[0], "token_anchor_q_b": anchor[1],
        "token_valid": True, "token_center_internal": role == 0,
    }


def _row_chemistry_lookup(row):
    tokens = row["tokens"]
    return {
        (
            int(tokens["token_atom_a"][i]), int(tokens["token_atom_b"][i]),
            int(tokens["token_shift"][i]),
        ): (
            int(tokens["token_bond_type"][i]), int(tokens["token_stereo"][i]),
            int(tokens["token_conjugated"][i]),
        )
        for i in range(len(tokens["token_atom_a"]))
        if bool(tokens["token_valid"][i])
    }


def build_periodic_line_distill_v2_sample(
    topology, trimer, smiles: str, version: str, *, chemistry_row=None,
):
    if version not in {"n_plus_1", "n_plus_2"}:
        raise ValueError(version)
    topology_info, topology_error = _prepare_topology(topology)
    trimer_info, trimer_error = _prepare_trimer(trimer, topology)
    if topology_info is None or trimer_info is None:
        return _empty_row()
    chemistry = None if chemistry_row is not None else _physical_bond_chemistry(str(smiles))
    chemistry_lookup = _row_chemistry_lookup(chemistry_row) if chemistry_row is not None else None
    internal = list(_internal_bonds(topology))
    left, right = topology_info["left"], topology_info["right"]

    tokens = []
    physical = []
    # One center-RU instance for every internal canonical bond.
    for physical_id, (u, _qu, v, _qv, _bond) in enumerate(internal):
        distance = _distance(trimer_info, topology, u, 0, v, 0)
        attrs = (
            chemistry_lookup.get(canonical_line_token(u, 0, v, 0))
            if chemistry_lookup is not None
            else _chemistry(chemistry, topology, u, 0, v, 0)
        )
        if distance is None or attrs is None:
            return _empty_row()
        state = len(tokens)
        tokens.append(_token_row(topology, u, v, 0, distance, attrs, (0, 0)))
        physical.append((physical_id, state, (u, 0), (v, 0)))

    left_distance = _distance(trimer_info, topology, right, -1, left, 0)
    right_distance = _distance(trimer_info, topology, right, 0, left, 1)
    if chemistry_lookup is not None:
        left_chem = right_chem = chemistry_lookup.get(
            canonical_line_token(right, 0, left, 1)
        )
    else:
        left_chem = _chemistry(chemistry, topology, right, -1, left, 0)
        right_chem = _chemistry(chemistry, topology, right, 0, left, 1)
    if any(value is None for value in (left_distance, right_distance, left_chem, right_chem)):
        return _empty_row()
    if tuple(left_chem) != tuple(right_chem):
        raise ValueError("left/right physical cross-bond chemistry differs")
    base_physical = len(internal)
    if version == "n_plus_1":
        cross_state = len(tokens)
        tokens.append(_token_row(
            topology, right, left, 2,
            0.5 * (left_distance + right_distance), left_chem, (0, 1),
        ))
        physical.extend([
            (base_physical, cross_state, (right, -1), (left, 0)),
            (base_physical + 1, cross_state, (right, 0), (left, 1)),
        ])
    else:
        left_state, right_state = len(tokens), len(tokens) + 1
        tokens.extend([
            _token_row(topology, right, left, -1, left_distance, left_chem, (-1, 0)),
            _token_row(topology, right, left, 1, right_distance, right_chem, (0, 1)),
        ])
        physical.extend([
            (base_physical, left_state, (right, -1), (left, 0)),
            (base_physical + 1, right_state, (right, 0), (left, 1)),
        ])

    # Re-index every physical bond by each center-RU endpoint.  Each ordered
    # pair is a separate message, even when both map to one N+1 state.
    incident = {}
    for physical_id, state, endpoint_a, endpoint_b in physical:
        for center, outer in ((endpoint_a, endpoint_b), (endpoint_b, endpoint_a)):
            if center[1] == 0:
                incident.setdefault(center[0], []).append(
                    (physical_id, state, outer)
                )
    relations = []
    seen_physical = set()
    for center_atom, bonds in sorted(incident.items()):
        for source_id, source_state, outer_source in bonds:
            for target_id, target_state, outer_target in bonds:
                if source_id == target_id:
                    continue
                physical_key = (source_id, target_id, center_atom)
                if physical_key in seen_physical:
                    continue
                seen_physical.add(physical_key)
                angle = _angle(
                    trimer_info, topology,
                    outer_source[0], outer_source[1], center_atom, 0,
                    outer_target[0], outer_target[1],
                )
                if angle is None:
                    continue
                relations.append({
                    "relation_source": source_state,
                    "relation_target": target_state,
                    "relation_center_atom": center_atom,
                    "relation_source_image_shift": outer_source[1],
                    "relation_angle": angle,
                    "relation_valid": True,
                })

    token_names = tuple(tokens[0]) if tokens else tuple(_empty_row()["tokens"])
    relation_names = (
        tuple(relations[0]) if relations else tuple(_empty_row()["relations"])
    )
    token_output = {
        name: np.asarray([row[name] for row in tokens]) for name in token_names
    }
    relation_output = {
        name: np.asarray([row[name] for row in relations]) for name in relation_names
    }
    # Explicit dtypes match the packed sidecar/collate contract.
    for name in token_output:
        token_output[name] = token_output[name].astype(
            np.float32 if name == "token_distance" else
            bool if name in {"token_valid", "token_center_internal"} else np.int64
        )
    for name in relation_output:
        relation_output[name] = relation_output[name].astype(
            np.float32 if name == "relation_angle" else
            bool if name == "relation_valid" else np.int64
        )
    return {
        "geometry_valid": bool(tokens and all(row["token_valid"] for row in tokens)),
        "tokens": token_output, "relations": relation_output,
    }


def collapse_n_plus_2_to_n_plus_1(row):
    """Collapse the two physical boundary states without collapsing messages."""
    if not row["geometry_valid"]:
        return _empty_row()
    tokens = row["tokens"]
    center = np.asarray(tokens["token_center_internal"], dtype=bool)
    n = int(center.sum())
    if len(center) != n + 2 or not center[:n].all() or center[n:].any():
        raise ValueError("N+2 token ordering is invalid")
    output = {name: np.asarray(value[:n + 1]).copy() for name, value in tokens.items()}
    output["token_atom_a"][n] = tokens["token_atom_a"][n]
    output["token_atom_b"][n] = tokens["token_atom_b"][n]
    output["token_shift"][n] = 2
    output["token_distance"][n] = 0.5 * (
        float(tokens["token_distance"][n]) + float(tokens["token_distance"][n + 1])
    )
    output["token_anchor_q_a"][n] = 0
    output["token_anchor_q_b"][n] = 1
    relations = {name: np.asarray(value).copy() for name, value in row["relations"].items()}
    relations["relation_source"][relations["relation_source"] >= n] = n
    relations["relation_target"][relations["relation_target"] >= n] = n
    return {"geometry_valid": True, "tokens": output, "relations": relations}


def build_periodic_line_distill_v2_pair(topology, trimer, smiles: str, *, chemistry_row=None):
    two = build_periodic_line_distill_v2_sample(
        topology, trimer, smiles, "n_plus_2", chemistry_row=chemistry_row,
    )
    return {"n_plus_2": two, "n_plus_1": collapse_n_plus_2_to_n_plus_1(two)}


__all__ = [
    "GEOMETRY_REVISION", "SCHEMA_PREFIX",
    "build_periodic_line_distill_v2_sample", "build_periodic_line_distill_v2_pair",
    "collapse_n_plus_2_to_n_plus_1",
]
