"""Runtime reflection-invariant torsions for the frozen periodic-line sidecar.

The helper adds geometry observations to an individual ``Data`` item.  It does
not alter line tokens or line relations: every torsion is attached to an
already-existing directed 1-hop line relation.
"""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations

import torch


TORSION_EPS = 1e-8


def _token_instances(atom_a, atom_b, shift):
    return [
        (int(atom_a), translation, int(atom_b), int(shift) + translation)
        for translation in range(-1, 2)
        if -1 <= translation <= 1
        and -1 <= int(shift) + translation <= 1
    ]


def reflection_invariant_dihedral_cosine(p_a, p_b, p_c, p_d, eps=TORSION_EPS):
    """Return ``cos(phi)`` or ``None`` for a numerically undefined torsion."""
    axis_ab = p_b - p_a
    axis_bc = p_c - p_b
    axis_cd = p_d - p_c
    normal_abc = torch.cross(axis_ab, axis_bc, dim=-1)
    normal_bcd = torch.cross(axis_bc, axis_cd, dim=-1)
    left = torch.linalg.vector_norm(normal_abc)
    right = torch.linalg.vector_norm(normal_bcd)
    if float(left) <= float(eps) or float(right) <= float(eps):
        return None
    value = torch.dot(normal_abc, normal_bcd) / (left * right)
    if not bool(torch.isfinite(value)):
        return None
    return float(value.clamp(-1.0, 1.0))


def canonical_torsion_quadruplet(atoms):
    minimum = min(offset for _, offset in atoms)
    direct = tuple((int(atom), int(offset) - minimum) for atom, offset in atoms)
    reverse = tuple(reversed(direct))
    return min(direct, reverse)


def build_periodic_torsion_fields(data):
    """Attach flat torsion observations aligned to existing relation rows."""
    token_a = torch.as_tensor(data.glt_token_atom_a).long().cpu()
    token_b = torch.as_tensor(data.glt_token_atom_b).long().cpu()
    token_shift = torch.as_tensor(data.glt_token_shift).long().cpu()
    relation_source = torch.as_tensor(data.glt_relation_source).long().cpu()
    relation_target = torch.as_tensor(data.glt_relation_target).long().cpu()
    relation_center = torch.as_tensor(data.glt_relation_center_atom).long().cpu()
    fallback = torch.as_tensor(data.glt_relation_is_fallback).bool().cpu()
    relation_count = int(relation_source.numel())

    incident = defaultdict(list)
    for token in range(int(token_a.numel())):
        for a, q_a, b, q_b in _token_instances(
            token_a[token], token_b[token], token_shift[token]
        ):
            incident[(a, q_a)].append((token, b, q_b))
            incident[(b, q_b)].append((token, a, q_a))

    grouped = defaultdict(list)
    for (center, q_center), members in incident.items():
        for first, second in combinations(members, 2):
            token_first, outer_first, q_first = first
            token_second, outer_second, q_second = second
            normalized = (
                outer_first, q_first - q_center, center,
                outer_second, q_second - q_center,
            )
            inverse = (
                outer_second, q_second - q_center, center,
                outer_first, q_first - q_center,
            )
            geometry_key = min(normalized, inverse)
            key = (min(token_first, token_second), max(token_first, token_second), geometry_key)
            grouped[key].append({
                'center': (center, q_center),
                'members': (first, second),
            })

    descriptors = []
    for (low, high, geometry_key), occurrences in sorted(grouped.items()):
        directions = [(low, high)] if low == high else [(low, high), (high, low)]
        for source, target in directions:
            descriptors.append({
                'source': source, 'target': target,
                'center': int(geometry_key[2]),
                'geometry_key': geometry_key,
                'occurrences': occurrences,
            })
    descriptors.sort(key=lambda item: (
        item['target'], item['source'], item['center'], False
    ))
    real_rows = torch.nonzero(~fallback, as_tuple=False).flatten().tolist()
    if len(descriptors) != len(real_rows):
        raise ValueError(
            'torsion relation reconstruction mismatch: '
            f'{len(descriptors)} descriptors for {len(real_rows)} relations'
        )
    for row, descriptor in zip(real_rows, descriptors):
        observed = (
            int(relation_source[row]), int(relation_target[row]),
            int(relation_center[row]),
        )
        expected = (
            descriptor['source'], descriptor['target'], descriptor['center']
        )
        if observed != expected:
            raise ValueError(
                f'torsion directed-relation alignment mismatch: {observed} != {expected}'
            )

    positions = torch.as_tensor(data.trimer_pos).float().cpu()
    base = torch.as_tensor(data.trimer_base_ru_atom_id).long().cpu()
    offsets = torch.as_tensor(data.trimer_ru_offset).long().cpu()
    coordinates = {
        (int(atom), int(offset)): positions[index]
        for index, (atom, offset) in enumerate(zip(base.tolist(), offsets.tolist()))
    }
    geometry_valid = bool(getattr(data, 'trimer_geometry_valid', False))
    values, rows = [], []
    counts = torch.zeros(relation_count, dtype=torch.long)
    for row, descriptor in zip(real_rows, descriptors):
        if not geometry_valid:
            continue
        seen = set()
        source_token = descriptor['source']
        target_token = descriptor['target']
        for occurrence in descriptor['occurrences']:
            center = occurrence['center']
            members = occurrence['members']
            orientations = []
            if source_token != target_token:
                source_member = next(item for item in members if item[0] == source_token)
                target_member = next(item for item in members if item[0] == target_token)
                orientations.append((target_member, source_member))
            else:
                orientations.append((members[0], members[1]))
                orientations.append((members[1], members[0]))
            for target_member, source_member in orientations:
                atom_a = (int(target_member[1]), int(target_member[2]))
                atom_b = (int(center[0]), int(center[1]))
                atom_c = (int(source_member[1]), int(source_member[2]))
                for extension_token, atom_d_id, atom_d_offset in incident[atom_c]:
                    if int(extension_token) == int(source_token):
                        continue
                    atom_d = (int(atom_d_id), int(atom_d_offset))
                    quadruplet = (atom_a, atom_b, atom_c, atom_d)
                    key = canonical_torsion_quadruplet(quadruplet)
                    if key in seen or any(atom not in coordinates for atom in quadruplet):
                        continue
                    value = reflection_invariant_dihedral_cosine(
                        *(coordinates[atom] for atom in quadruplet)
                    )
                    if value is None:
                        continue
                    seen.add(key)
                    values.append(value)
                    rows.append(row)
        counts[row] = len(seen)

    data.glt_torsion_observation_value = torch.tensor(values, dtype=torch.float32)
    data.glt_torsion_observation_relation = torch.tensor(rows, dtype=torch.long)
    data.glt_relation_torsion_count = counts
    data.glt_relation_torsion_covered = counts > 0
    data.glt_relation_torsion_source_cross_ru = torch.zeros(relation_count, dtype=torch.bool)
    if relation_count:
        valid_source = relation_source.clamp(0, max(int(token_shift.numel()) - 1, 0))
        data.glt_relation_torsion_source_cross_ru = token_shift[valid_source].abs() > 0
    data.glt_graph_torsion_covered = bool((counts > 0).any())
    return data


__all__ = [
    'TORSION_EPS', 'build_periodic_torsion_fields',
    'canonical_torsion_quadruplet',
    'reflection_invariant_dihedral_cosine',
]
