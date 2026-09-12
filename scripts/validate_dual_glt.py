#!/usr/bin/env python3
"""At most two real frozen records, CPU only, no optimizer/cache writes.

Run on a supported environment only. Example:
python scripts/validate_dual_glt.py --topology-root PATH --trimer-root PATH \
    --sample HEXKEY 'P-SMILES' --sample HEXKEY_N0 'P-SMILES_N0'
"""
import argparse
import gc
import copy
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from rdkit import Chem
from torch.utils.data import DataLoader
from src.dataset.glt_dual import FrozenDualLayerSource, DualGLTDataset, dual_glt_collate, build_dual_sample
from src.dataset.graph_data import build_periodic_multimer_mol
from src.dataset.glt_bond_chemistry import bond_feature_vector
from src.dataset.canonical_periodic import resolve_normalized_identity


def check_result(status, reason='', **details):
    return dict(status=status, reason=reason, **details)


def _json_safe(value):
    """Convert report values to finite JSON primitives without hiding failure."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if torch.is_tensor(value):
        return _json_safe(value.detach().cpu().tolist())
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return str(value)


AUDIT_CHECK_NAMES = (
    'record', 'chemistry', 'source_stereo', 'connection_policy',
    'star_linking', 'translation_chemistry', 'seam_stereo', 'identity',
    'connection_geometry', 'stereo_coordinates', 'model_input',
)


class _AuditStop(Exception):
    """Unwind a completed audit without bypassing report finalization."""

    def __init__(self, code):
        super().__init__(int(code))
        self.code = int(code)


def _bond_dir_after_endpoint_mapping(source_bond, copied_bond, mapped_begin, mapped_end):
    """Compare RDKit directional-bond enums in the copied bond's orientation.

    ``BondDir`` is orientation-sensitive: a canonical reparse may reverse a
    bond's begin/end indices while retaining the same E/Z chemistry.  The
    slash directions therefore need the corresponding endpoint reversal
    before being compared; a raw enum equality would reject an equivalent
    non-canonical P-SMILES.
    """
    direction = source_bond.GetBondDir()
    if (
        copied_bond.GetBeginAtomIdx() == int(mapped_begin)
        and copied_bond.GetEndAtomIdx() == int(mapped_end)
    ):
        return direction
    if (
        copied_bond.GetBeginAtomIdx() == int(mapped_end)
        and copied_bond.GetEndAtomIdx() == int(mapped_begin)
    ):
        return {
            Chem.BondDir.ENDUPRIGHT: Chem.BondDir.ENDDOWNRIGHT,
            Chem.BondDir.ENDDOWNRIGHT: Chem.BondDir.ENDUPRIGHT,
        }.get(direction, direction)
    return None


def blank_record(key, smiles, status, reason):
    """Return a complete per-record report when opening/auditing cannot start."""
    checks = {
        name: check_result('NOT_RUN', reason) for name in AUDIT_CHECK_NAMES
    }
    checks['record'] = check_result(status, reason)
    return dict(key=bytes(key).hex(), smiles=str(smiles), checks=checks, connections=[])


def audit_source_stereo(
    source, mol, meta, source_to_normalized_base=None,
    source_to_normalized_atom=None, normalized_molecule=None,
):
    """Compare source stereo metadata with every physical normalized copy.

    The source atom indices are provenance only.  ``source_to_normalized_base``
    maps source non-dummy ranks to the normalized RU ranks used by ``meta``;
    ``source_to_normalized_atom`` and ``normalized_molecule`` additionally
    resolve dummy-site side order and notation-level BondDir changes after a
    canonical reparse.  Omitting them is retained for callers whose molecule
    and reconstruction were built from the same atom order.  A terminal dummy
    reference has no finite physical neighbour and is therefore expected to become
    ``STEREONONE``; selecting another explicit substituent would change the
    chemical meaning and is not considered preservation.
    """
    real = [a.GetIdx() for a in source.GetAtoms() if a.GetAtomicNum() != 0]
    dummies = [a.GetIdx() for a in source.GetAtoms() if a.GetAtomicNum() == 0]
    if len(dummies) != 2 or any(
        source.GetAtomWithIdx(i).GetDegree() != 1 for i in dummies
    ):
        raise ValueError('source must have two degree-one attachment atoms')
    base = {old: i for i, old in enumerate(real)}
    if source_to_normalized_base is None:
        source_to_normalized_base = torch.arange(len(real), dtype=torch.long)
    source_to_normalized_base = torch.as_tensor(
        source_to_normalized_base, dtype=torch.long
    ).reshape(-1)
    if source_to_normalized_base.numel() != len(real) or sorted(
        source_to_normalized_base.tolist()
    ) != list(range(len(real))):
        raise ValueError('source/normalized stereo mapping is not a permutation')
    if source_to_normalized_atom is None:
        source_to_normalized_atom = torch.arange(
            source.GetNumAtoms(), dtype=torch.long
        )
    source_to_normalized_atom = torch.as_tensor(
        source_to_normalized_atom, dtype=torch.long
    ).reshape(-1)
    if (
        source_to_normalized_atom.numel() != source.GetNumAtoms()
        or normalized_molecule is not None
        and source_to_normalized_atom.numel() != normalized_molecule.GetNumAtoms()
        or sorted(source_to_normalized_atom.tolist())
        != list(range(source_to_normalized_atom.numel()))
    ):
        raise ValueError('source/normalized full stereo mapping is incomplete')
    if normalized_molecule is None:
        normalized_molecule = source
    normalized_dummies = [
        atom.GetIdx() for atom in normalized_molecule.GetAtoms()
        if atom.GetAtomicNum() == 0
    ]
    if len(normalized_dummies) != 2:
        raise ValueError('normalized source must have two attachment atoms')
    normalized_base = [
        atom.GetIdx() for atom in normalized_molecule.GetAtoms()
        if atom.GetAtomicNum() != 0
    ]
    normalized_rank = {atom: rank for rank, atom in enumerate(normalized_base)}
    normalized_dummy_side = {
        int(atom): side for side, atom in enumerate(normalized_dummies)
    }
    normalized_boundary = [
        normalized_molecule.GetAtomWithIdx(i).GetNeighbors()[0].GetIdx()
        for i in normalized_dummies
    ]
    units = meta['unit_atoms']
    signs = {
        Chem.BondStereo.STEREOE: -1,
        Chem.BondStereo.STEREOTRANS: -1,
        Chem.BondStereo.STEREOZ: 1,
        Chem.BondStereo.STEREOCIS: 1,
    }
    failures, snapshots = [], []
    retained = terminal_undefined = center_specified = 0
    for original in source.GetBonds():
        a, b = original.GetBeginAtomIdx(), original.GetEndAtomIdx()
        snapshots.append(dict(
            atoms=[a, b], bond_type=str(original.GetBondType()),
            stereo=str(original.GetStereo()), references=list(original.GetStereoAtoms()),
            direction=str(original.GetBondDir()),
        ))
        if a not in base or b not in base:
            continue  # attachment bonds become physical seams
        a_norm = int(source_to_normalized_base[base[a]])
        b_norm = int(source_to_normalized_base[base[b]])
        normalized_bond = normalized_molecule.GetBondBetweenAtoms(
            int(source_to_normalized_atom[a]), int(source_to_normalized_atom[b])
        )
        if normalized_bond is None:
            failures.append(dict(
                source_bond=[a, b], reason='missing normalized source bond'
            ))
            continue
        expected_stereo = normalized_bond.GetStereo()
        if expected_stereo != original.GetStereo():
            failures.append(dict(
                source_bond=[a, b], reason='source/normalized stereo differs',
                source=str(original.GetStereo()),
                normalized=str(expected_stereo),
            ))
        if original.GetStereo() in signs and expected_stereo not in signs:
            failures.append(dict(
                source_bond=[a, b], reason='normalized stereo is not defined'
            ))
            continue
        if original.GetStereo() in signs:
            center_specified += 1
        for q, unit in enumerate(units):
            expected_begin = normalized_rank[normalized_bond.GetBeginAtomIdx()]
            expected_end = normalized_rank[normalized_bond.GetEndAtomIdx()]
            i, j = unit[a_norm], unit[b_norm]
            copied = mol.GetBondBetweenAtoms(i, j)
            tag = dict(source_bond=[a, b], unit=q, physical_bond=[i, j])
            if copied is None:
                failures.append(dict(**tag, reason='missing copied bond'))
                continue
            expected_direction = _bond_dir_after_endpoint_mapping(
                normalized_bond,
                copied,
                unit[expected_begin],
                unit[expected_end],
            )
            if expected_direction is None or copied.GetBondDir() != expected_direction:
                failures.append(dict(**tag, reason='BondDir changed'))
            if original.GetStereo() not in signs:
                if copied.GetStereo() != original.GetStereo():
                    failures.append(dict(
                        **tag, reason='unspecified/unknown stereo changed'
                    ))
                continue
            expected = []
            unresolved = False
            refs = list(original.GetStereoAtoms())
            if len(refs) != 2:
                unresolved = True
            else:
                for ref in refs:
                    if ref in base:
                        expected.append(unit[int(source_to_normalized_base[base[ref]])])
                        continue
                    if ref not in dummies:
                        unresolved = True
                        break
                    normalized_ref = int(source_to_normalized_atom[ref])
                    if normalized_ref not in normalized_dummy_side:
                        unresolved = True
                        break
                    side = normalized_dummy_side[normalized_ref]
                    neighbor_unit = q - 1 if side == 0 else q + 1
                    if not 0 <= neighbor_unit < len(units):
                        unresolved = True
                        expected.append(None)
                        continue
                    expected.append(
                        units[neighbor_unit][
                            normalized_rank[normalized_boundary[1 - side]]
                        ]
                    )
            if unresolved or len(expected) != 2 or any(value is None for value in expected):
                terminal_undefined += 1
                if copied.GetStereo() != Chem.BondStereo.STEREONONE:
                    failures.append(dict(
                        **tag, reason='terminal bond no longer has defined stereo'
                    ))
                continue
            substituents = [
                {
                    n.GetIdx() for n in mol.GetAtomWithIdx(end).GetNeighbors()
                    if n.GetIdx() != other
                }
                for end, other in ((i, j), (j, i))
            ]
            actual = list(copied.GetStereoAtoms())
            if copied.GetBeginAtomIdx() != i:
                actual.reverse()
            if copied.GetStereo() not in signs or len(actual) != 2:
                failures.append(dict(**tag, reason='preservable stereo lost'))
                continue
            if any(ref not in neighbors for ref, neighbors in zip(actual, substituents)):
                failures.append(dict(
                    **tag, reason='stereo reference is not a real neighbor'
                ))
                continue
            parity = sum(ref != wanted for ref, wanted in zip(actual, expected)) % 2
            if signs[copied.GetStereo()] != signs[expected_stereo] * (
                -1 if parity else 1
            ):
                failures.append(dict(**tag, reason='relative stereo changed'))
            else:
                retained += 1
    return check_result(
        'ANOMALY' if failures else 'PASS', failures=failures,
        source_bonds=snapshots, retained_copies=retained,
        terminal_undefined=terminal_undefined,
        center_specified=center_specified,
        expected_center_bonds=sum(
            b.GetBeginAtomIdx() in base and b.GetEndAtomIdx() in base
            for b in source.GetBonds()
        ),
    )


def audit_record(topology, trimer, smiles):
    """Audit the supplied record only; never generate or modify coordinates."""
    report = dict(smiles=smiles,
                  checks={name: check_result('NOT_RUN', 'prerequisite unavailable')
                          for name in AUDIT_CHECK_NAMES},
                  connections=[])
    checks = report['checks']
    try:
        identity = resolve_normalized_identity(
            topology, smiles, require_fields=hasattr(topology, 'mips_x')
        )
        source = identity['source_molecule']
        construction_smiles = identity['normalized_smiles']
        mol, meta = build_periodic_multimer_mol(
            construction_smiles, 3, close_periodic=False
        )
        report['identity_mapping'] = {
            'normalized_smiles': construction_smiles,
            'source_to_normalized_atom': identity['source_to_normalized_atom'].tolist(),
            'source_to_normalized_base': identity['source_to_normalized_base'].tolist(),
            'attachments': identity['attachments'],
        }
    except ValueError as exc:
        checks['record'] = check_result('ANOMALY', str(exc))
        checks['chemistry'] = check_result('ANOMALY', str(exc))
        checks['identity'] = checks['stereo_coordinates'] = check_result('NOT_RUN', 'chemistry unavailable')
        return report
    checks['chemistry'] = check_result('PASS')
    checks['record'] = check_result('PASS')
    try:
        checks['source_stereo'] = audit_source_stereo(
            source, mol, meta, identity['source_to_normalized_base'],
            identity['source_to_normalized_atom'], identity['normalized_molecule'],
        )
    except ValueError as exc:
        checks['source_stereo'] = check_result('ANOMALY', str(exc))
    report['connection_policy'] = {name: meta[name] for name in (
        'attachment_bond_type_left', 'attachment_bond_type_right', 'connection_bond_policy')}
    report['connection_policy']['actual_type'] = str(meta['attachment_bond_type'])
    report['connection_policy']['declared_left'] = str(
        meta['attachment_bond_type_left']
    )
    report['connection_policy']['declared_right'] = str(
        meta['attachment_bond_type_right']
    )
    policy = meta['connection_bond_policy']
    checks['connection_policy'] = check_result(
        'PASS' if policy == 'matching_attachment_type' else 'REVIEW',
        '' if policy == 'matching_attachment_type' else 'policy requires review; not proof of invalid chemistry')
    checks['star_linking'] = check_result(
        'NOT_APPLICABLE',
        'legacy single-RU Star graph is not an input to this route; physical seams audited below',
    )
    size = meta['base_atom_count']
    normalized_base = [
        atom.GetIdx() for atom in identity['normalized_molecule'].GetAtoms()
        if atom.GetAtomicNum() != 0
    ]
    normalized_rank = {atom: rank for rank, atom in enumerate(normalized_base)}
    chemistry = {}
    differences, terminal_differences = [], []
    source_features = {}
    for bond in identity['normalized_molecule'].GetBonds():
        begin, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if begin not in normalized_rank or end not in normalized_rank:
            continue
        key = tuple(sorted((normalized_rank[begin], normalized_rank[end])))
        source_features[key] = bond_feature_vector(bond)
    terminal_unset = set()
    for item in meta.get('terminal_stereo_unset', []):
        bond = tuple(int(value) for value in item.get('bond', ()))
        if len(bond) != 2 or bond[0] not in normalized_rank or bond[1] not in normalized_rank:
            continue
        terminal_unset.add((
            int(item['unit']),
            tuple(sorted((normalized_rank[bond[0]], normalized_rank[bond[1]]))),
        ))
    for key, expected_feature in source_features.items():
        a_rank, b_rank = key
        for unit_id, unit in enumerate(meta['unit_atoms']):
            copied = mol.GetBondBetweenAtoms(unit[a_rank], unit[b_rank])
            if copied is None:
                differences.append(dict(identity=list(key), unit=unit_id, reason='missing internal copy'))
                continue
            observed_feature = bond_feature_vector(copied)
            if np.array_equal(expected_feature, observed_feature):
                continue
            physical_key = tuple(sorted((int(unit[a_rank]), int(unit[b_rank]))))
            if unit_id != 1 and not np.array_equal(expected_feature[:5], observed_feature[:5]):
                differences.append(dict(
                    identity=list(key), unit=unit_id,
                    physical_bond=list(physical_key),
                    expected=expected_feature.tolist(),
                    observed=observed_feature.tolist(),
                    reason='terminal bond type changed',
                ))
            elif unit_id != 1:
                terminal_differences.append(dict(
                    identity=list(key), unit=unit_id,
                    physical_bond=list(physical_key),
                    expected=expected_feature.tolist(),
                    observed=observed_feature.tolist(),
                    reason=(
                        'terminal stereo intentionally unspecified'
                        if (unit_id, key) in terminal_unset
                        else 'finite-terminal conjugation/ring/stereo differs'
                    ),
                ))
            else:
                differences.append(dict(
                    identity=list(key), unit=unit_id,
                    physical_bond=list(physical_key),
                    expected=expected_feature.tolist(),
                    observed=observed_feature.tolist(),
                ))
    checks['translation_chemistry'] = check_result(
        'ANOMALY' if differences else ('REVIEW' if terminal_differences else 'PASS'),
        'terminal copies may lose finite-chain stereo; no periodic E/Z inferred'
        if terminal_differences and not differences else '',
        differences=differences, terminal_stereo_unset=terminal_differences,
    )
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        q_i, q_j = i // size - 1, j // size - 1
        if q_i == q_j:
            continue
        report['connections'].append(dict(
            atoms=[i, j],
            identities=[
                {'base': int(i % size), 'q': int(q_i)},
                {'base': int(j % size), 'q': int(q_j)},
            ],
            period_shift=int(q_j - q_i),
            is_physical=True,
            is_periodic_closing=False,
            bond_type=str(bond.GetBondType()),
            stereo=str(bond.GetStereo()),
            conjugation=bool(bond.GetIsConjugated()),
            ring=bool(bond.IsInRing()),
            distance=None, angles=[], geometry_status='NOT_RUN',
        ))
    unspecified = [edge['atoms'] for edge in report['connections']
                   if edge['bond_type'] == 'DOUBLE' and edge['stereo'] in ('STEREONONE', 'STEREOANY')]
    defined_seams = [edge['atoms'] for edge in report['connections']
                     if edge['bond_type'] == 'DOUBLE' and edge['stereo'] not in ('STEREONONE', 'STEREOANY')]
    if defined_seams:
        checks['seam_stereo'] = check_result(
            'ANOMALY', 'new cross-RU double bond carries an unsupported E/Z claim',
            bonds=defined_seams, unspecified=unspecified,
        )
    else:
        checks['seam_stereo'] = check_result(
            'REVIEW' if unspecified else 'NOT_APPLICABLE',
            'cross-RU double bond stereo unspecified; no E/Z inferred' if unspecified else 'no unspecified seam double bond',
            bonds=unspecified,
        )
    from src.dataset.periodic_line_glt_complete import build_complete_trimer_glt_sample
    try:
        row = build_complete_trimer_glt_sample(
            topology, trimer, smiles, identity=identity
        )
    except (KeyError, RuntimeError, ValueError) as exc:
        checks['identity'] = check_result('ANOMALY', str(exc))
        checks['stereo_coordinates'] = check_result(
            'NOT_RUN', 'identity/geometry unavailable'
        )
        checks['model_input'] = check_result(
            'NOT_RUN', 'identity/geometry unavailable'
        )
        return report
    checks['identity'] = check_result('PASS' if row['geometry_valid'] else 'ANOMALY', row['invalid_reason'])
    if not row['geometry_valid']:
        checks['stereo_coordinates'] = check_result('NOT_RUN', 'invalid frozen identity/geometry')
        return report
    base = getattr(trimer, 'trimer_base_ru_atom_id', None)
    if base is None:
        base = trimer.trimer_base_ru_atom_index
    identities = [(int(b), int(q)) for b, q in zip(base, trimer.trimer_ru_offset)]
    expected = {(i, q) for i in range(size) for q in (-1, 0, 1)}
    if len(identities) != len(expected) or set(identities) != expected:
        checks['identity'] = check_result('ANOMALY', 'physical atom identities are not a bijection')
        checks['stereo_coordinates'] = check_result('NOT_RUN', 'invalid identity')
        return report
    lookup = {identity: i for i, identity in enumerate(identities)}
    def point(i):
        return trimer.trimer_pos[lookup[i % size, i // size - 1]].double()
    for atom in mol.GetAtoms():
        i = atom.GetIdx()
        if int(trimer.trimer_atomic_number[lookup[i % size, i // size - 1]]) != atom.GetAtomicNum():
            checks['identity'] = check_result('ANOMALY', 'physical element mismatch')
            checks['stereo_coordinates'] = check_result('NOT_RUN', 'invalid identity')
            return report
    for edge in report['connections']:
        i, j = edge['atoms']
        edge['distance'] = float((point(i) - point(j)).norm())
        edge['geometry_status'] = 'PASS' if edge['distance'] > 0 and math.isfinite(edge['distance']) else 'ANOMALY'
        for center, partner in ((i, j), (j, i)):
            for neighbor in mol.GetAtomWithIdx(center).GetNeighbors():
                k = neighbor.GetIdx()
                if k == partner:
                    continue
                u, v = point(k) - point(center), point(partner) - point(center)
                denominator = float(u.norm() * v.norm())
                angle = float(torch.acos((u.dot(v) / denominator).clamp(-1, 1))) if denominator > 0 else float('nan')
                valid = math.isfinite(angle)
                edge['angles'].append(dict(atoms=[k, center, partner], radians=angle if valid else None))
                if not valid:
                    edge['geometry_status'] = 'ANOMALY'
    checks['connection_geometry'] = check_result(
        'PASS' if report['connections'] and all(
            e['geometry_status'] == 'PASS' for e in report['connections']
        ) else ('NOT_APPLICABLE' if not report['connections'] else 'ANOMALY')
    )
    if checks['source_stereo']['status'] == 'PASS':
        try:
            count = audit_frozen_stereo(
                trimer, smiles, topology=topology, identity=identity
            )
            checks['stereo_coordinates'] = check_result('PASS' if count else 'NOT_APPLICABLE', checked_bonds=count)
        except ValueError as exc:
            checks['stereo_coordinates'] = check_result('ANOMALY', str(exc))
    try:
        item = build_dual_sample(topology, trimer, smiles)
        checks['model_input'] = check_result('PASS' if item.geometry_valid else 'ANOMALY',
                                            item.geometry_invalid_reason, center_bonds=int(item.bond_center.sum()))
    except ValueError as exc:
        checks['model_input'] = check_result('ANOMALY', str(exc))
    return report


def audit_frozen_stereo(trimer, smiles, *, topology=None, identity=None):
    """Check every retained specified E/Z copy against frozen coordinates.

    The chemical graph is rebuilt from the normalized identity, while the
    coordinate lookup is taken only from the immutable Trimer identity table.
    No coordinate is generated or repaired here.  Terminal finite-chain copies
    whose dummy reference was intentionally removed are ``STEREONONE`` and are
    therefore not counted as retained stereo.

    Only projected side agreement is checked, with no angular-quality cutoff.
    Opposite signs fail; an exactly orthogonal projection is indeterminate.
    """
    if identity is None and topology is not None:
        identity = resolve_normalized_identity(
            topology, smiles, require_fields=hasattr(topology, 'mips_x')
        )
    if identity is None:
        source = Chem.MolFromSmiles(str(smiles))
        if source is None:
            raise ValueError('invalid source P-SMILES')
        normalized = Chem.MolFromSmiles(Chem.MolToSmiles(source, canonical=True))
    else:
        normalized = identity['normalized_molecule']
    mol, meta = build_periodic_multimer_mol(
        normalized, 3, close_periodic=False
    )

    raw_positions = getattr(trimer, 'trimer_pos', None)
    positions = torch.as_tensor(raw_positions).double() if raw_positions is not None else None
    base = getattr(trimer, 'trimer_base_ru_atom_id', None)
    if base is None:
        base = getattr(trimer, 'trimer_base_ru_atom_index', None)
    offsets = getattr(trimer, 'trimer_ru_offset', None)
    atomic = getattr(trimer, 'trimer_atomic_number', None)
    if base is None or offsets is None or atomic is None:
        raise ValueError('frozen Trimer identity fields are missing')
    base = torch.as_tensor(base, dtype=torch.long).reshape(-1)
    offsets = torch.as_tensor(offsets, dtype=torch.long).reshape(-1)
    atomic = torch.as_tensor(atomic, dtype=torch.long).reshape(-1)
    if (
        positions is None or positions.ndim != 2 or positions.size(1) != 3
        or not bool(torch.isfinite(positions).all())
        or base.numel() != positions.size(0)
        or offsets.numel() != positions.size(0)
        or atomic.numel() != positions.size(0)
    ):
        raise ValueError('frozen Trimer identity/coordinates have invalid shape')
    size = int(meta['base_atom_count'])
    expected_states = {(base_id, q) for q in (-1, 0, 1) for base_id in range(size)}
    states = {(int(base_id), int(q)) for base_id, q in zip(base, offsets)}
    if states != expected_states or len(states) != positions.size(0):
        raise ValueError('frozen Trimer physical identities are not a bijection')
    lookup = {
        (int(base_id), int(q)): int(index)
        for index, (base_id, q) in enumerate(zip(base.tolist(), offsets.tolist()))
    }
    state_by_mol_index = {}
    for q, unit in enumerate(meta['unit_atoms']):
        for base_id, atom_index in enumerate(unit):
            state_by_mol_index[int(atom_index)] = (base_id, q - 1)

    def point(atom_index):
        try:
            state = state_by_mol_index[int(atom_index)]
            return positions[lookup[state]]
        except KeyError as exc:
            raise ValueError(f'missing frozen coordinate identity for atom {atom_index}') from exc

    retained = 0
    failures = []
    for bond in mol.GetBonds():
        stereo = bond.GetStereo()
        if stereo not in (
            Chem.BondStereo.STEREOE, Chem.BondStereo.STEREOZ,
            Chem.BondStereo.STEREOCIS, Chem.BondStereo.STEREOTRANS,
        ):
            continue
        i, j = int(bond.GetBeginAtomIdx()), int(bond.GetEndAtomIdx())
        refs = list(bond.GetStereoAtoms())
        if len(refs) != 2:
            failures.append(f'missing stereo references at {(i, j)}')
            continue
        if mol.GetBondBetweenAtoms(i, refs[0]) is None or mol.GetBondBetweenAtoms(j, refs[1]) is None:
            failures.append(f'stereo reference is not a neighbor at {(i, j)}')
            continue
        try:
            axis = point(j) - point(i)
        except ValueError as exc:
            failures.append(str(exc))
            continue
        axis_norm = torch.linalg.vector_norm(axis)
        if not bool(torch.isfinite(axis_norm)) or float(axis_norm) <= 1e-12:
            failures.append(f'degenerate stereo axis at {(i, j)}')
            continue
        axis = axis / axis_norm
        try:
            u, v = point(refs[0]) - point(i), point(refs[1]) - point(j)
        except ValueError as exc:
            failures.append(str(exc))
            continue
        u = u - torch.dot(u, axis) * axis
        v = v - torch.dot(v, axis) * axis
        denominator = torch.linalg.vector_norm(u) * torch.linalg.vector_norm(v)
        if not bool(torch.isfinite(denominator)) or float(denominator) <= 1e-12:
            failures.append(f'degenerate stereo projection at {(i, j)}')
            continue
        cosine = torch.dot(u, v) / denominator
        expected_sign = -1 if stereo in (
            Chem.BondStereo.STEREOE, Chem.BondStereo.STEREOTRANS
        ) else 1
        if not bool(torch.isfinite(cosine)):
            failures.append(f'nonfinite stereo projection cosine at {(i, j)}')
            continue
        signed_cosine = float(expected_sign * cosine)
        if signed_cosine == 0.0:
            failures.append(f'indeterminate orthogonal stereo projection at {(i, j)}')
            continue
        if signed_cosine < 0.0:
            failures.append(
                f'frozen coordinates disagree with specified bond stereo at {(i, j)}; '
                f'signed_cosine={signed_cosine}; cache unchanged'
            )
            continue
        retained += 1
    if failures:
        raise ValueError(
            f'{len(failures)} frozen stereo coordinate checks failed; '
            + '; '.join(failures[:8])
        )
    return retained


def renumber_frozen(topology, trimer, smiles):
    changed = copy.deepcopy(trimer)
    order = torch.arange(trimer.trimer_pos.size(0) - 1, -1, -1)
    inverse = torch.argsort(order)
    for name in ('trimer_pos', 'trimer_atomic_number', 'trimer_base_ru_atom_id',
                 'trimer_base_ru_atom_index', 'trimer_ru_offset'):
        value = getattr(trimer, name, None)
        if value is not None:
            setattr(changed, name, value[order])
    changed.trimer_edge_index = inverse[trimer.trimer_edge_index]
    return build_dual_sample(topology, changed, smiles)


def fixture_coverage(entries):
    """Validate supplied records without requiring a special E/Z + N=0 pair.

    Report the edge-case coverage actually present. Synthetic regression tests
    cover N=0/stereo separately; their absence is not a real-smoke failure.
    """
    roles = []
    for entry in entries:
        checks = entry['checks']
        source = checks.get('source_stereo', {})
        model = checks.get('model_input', {})
        expected, actual = source.get('expected_center_bonds'), model.get('center_bonds')
        role = None
        if source.get('status') == 'PASS' and model.get('status') == 'PASS' and expected == actual:
            if expected == 0:
                role = 'n0'
            elif expected is not None and expected > 0:
                role = 'ordinary_ez' if source.get('center_specified', 0) > 0 else 'ordinary'
        roles.append(role)
    passed = 1 <= len(roles) <= 2 and all(role is not None for role in roles)
    return check_result('PASS' if passed else 'ANOMALY',
        '' if passed else 'requires 1-2 valid records with matching source/model center counts',
        roles=roles, has_n0='n0' in roles, has_center_stereo='ordinary_ez' in roles)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--topology-root', required=True)
    parser.add_argument('--trimer-root', required=True)
    parser.add_argument('--sample', action='append', nargs=2, metavar=('HEXKEY', 'PSMILES'), required=True)
    parser.add_argument('--audit-only', action='store_true', help='inspect data without constructing a model')
    parser.add_argument('--report-json', type=Path, help='new report path; existing files are not overwritten')
    args = parser.parse_args()
    if not 1 <= len(args.sample) <= 2:
        parser.error('provide one or two real frozen records')
    if args.report_json is not None and args.report_json.exists():
        parser.error('report path already exists; choose a new path')
    torch.set_num_threads(1)
    try:
        samples = [(bytes.fromhex(key), smiles) for key, smiles in args.sample]
        if any(len(key) != 32 for key, _ in samples):
            raise ValueError('sample keys must have 32 bytes')
        if len({key for key, _ in samples}) != len(samples):
            raise ValueError('provide distinct frozen records')
    except ValueError as exc:
        parser.error(str(exc))
    report = dict(scope='only the explicitly supplied 1-2 records; not full-cache coverage',
                  samples=[], model=check_result('NOT_RUN', 'not executed'), outcome='NOT_RUN')
    source = None
    exit_code = 0
    try:
        try:
            source = FrozenDualLayerSource(args.topology_root, args.trimer_root, samples)
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            report['samples'] = [
                blank_record(key, smiles, 'ANOMALY', f'frozen layers unavailable: {exc}')
                for key, smiles in samples
            ]
            report['fixture_coverage'] = fixture_coverage(report['samples'])
            report['model'] = check_result('NOT_RUN', 'not executed: frozen layers unavailable')
            report['outcome'] = 'DATA_ANOMALY'
            report['error'] = f'{type(exc).__name__}: {exc}'
            raise _AuditStop(1)
        except Exception as exc:
            report['samples'] = [
                blank_record(key, smiles, 'SCRIPT_ERROR', f'frozen layer open failed: {exc}')
                for key, smiles in samples
            ]
            report['fixture_coverage'] = fixture_coverage(report['samples'])
            report['model'] = check_result('NOT_RUN', 'not executed: script error')
            report['outcome'] = 'SCRIPT_ERROR'
            report['error'] = f'{type(exc).__name__}: {exc}'
            raise _AuditStop(2)
        records = []
        for index, (key, smiles) in enumerate(samples):
            try:
                record = source[index]
            except (FileNotFoundError, KeyError, RuntimeError, ValueError) as exc:
                report['samples'].append(
                    blank_record(key, smiles, 'ANOMALY', f'frozen record unavailable: {exc}')
                )
                continue
            records.append(record)
            try:
                entry = audit_record(*record)
            except ValueError as exc:
                entry = blank_record(key, smiles, 'ANOMALY', str(exc))
            except Exception as exc:
                entry = blank_record(key, smiles, 'SCRIPT_ERROR', str(exc))
            entry['key'] = key.hex()
            report['samples'].append(entry)
        report['fixture_coverage'] = fixture_coverage(report['samples'])
        anomalies = any(c['status'] == 'ANOMALY' for e in report['samples'] for c in e['checks'].values())
        script_errors = any(c['status'] == 'SCRIPT_ERROR' for e in report['samples'] for c in e['checks'].values())
        if script_errors:
            report['outcome'] = 'SCRIPT_ERROR'
            exit_code = 2
            raise _AuditStop(exit_code)
        if anomalies or report['fixture_coverage']['status'] != 'PASS':
            report['outcome'] = 'DATA_ANOMALY'
            report['model'] = check_result('NOT_RUN', 'not executed: data anomaly')
            exit_code = 1
            raise _AuditStop(exit_code)
        reviews = any(c['status'] == 'REVIEW' for e in report['samples'] for c in e['checks'].values())
        report['outcome'] = 'REVIEW' if reviews else 'PASS'
        if args.audit_only:
            report['model'] = check_result('NOT_RUN', '--audit-only: no model constructed')
            raise _AuditStop(exit_code)
        report['model'] = check_result('RUNNING')
        from src.modules.glt_dual import build_dual_glt_model
        dataset = DualGLTDataset(source)
        batch = next(iter(DataLoader(dataset, batch_size=2, num_workers=0, collate_fn=dual_glt_collate)))
        centers = torch.bincount(batch.bond_batch[batch.bond_center], minlength=len(records))
        if not batch.geometry_valid.all() or not (centers > 0).any():
            raise RuntimeError('model smoke needs at least one valid graph with center bonds')
        checked = sum(entry['checks']['stereo_coordinates'].get('checked_bonds', 0)
                      for entry in report['samples'])
        renumbered = dual_glt_collate([renumber_frozen(*record) for record in records])
        if not renumbered.geometry_valid.all():
            raise RuntimeError('renumbering invalidated real geometry')
        for mode in ('concat', 'kfuse'):
            torch.manual_seed(42)
            model = build_dual_glt_model(mode, dropout=0).cpu()
            prediction = model(batch)
            torch.testing.assert_close(prediction, model(renumbered), atol=1e-5, rtol=1e-5)
            print({'mode': mode, 'stereo_bonds_checked': checked, 'renumbering': 'PASS'}, file=sys.stderr)
            if prediction.shape != (len(records), 1) or not torch.isfinite(prediction).all():
                raise RuntimeError('invalid predictions')
            targets = torch.linspace(0.3, -0.7, len(records)).unsqueeze(-1)
            (prediction - targets).square().mean().backward()
            names = ['o8.bond_bias.position', 'o8.layers.0.attention.qkv.weight',
                     'glt.endpoint.weight', 'glt.distance_projection.weight',
                     'glt.angle_bias.heads.2.weight', 'glt.layers.0.attention.qkv.weight',
                     'predictor.0.weight']
            if mode == 'kfuse':
                names.append('kfuse.v_proj.glt3d.weight')
                torch.testing.assert_close(model.kfuse.last_attention_weights,
                                           torch.ones_like(model.kfuse.last_attention_weights))
            gradients = {}
            for name in names:
                grad = model.get_parameter(name).grad
                if grad is None or not torch.isfinite(grad).all() or not grad.abs().sum() > 0:
                    raise RuntimeError(f'missing/nonfinite gradient: {name}')
                gradients[name] = float(grad.abs().sum())
            for parameter in model.parameters():
                if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                    raise RuntimeError('nonfinite gradient')
            print({'mode': mode, 'status': 'CPU_FORWARD_BACKWARD_PASS',
                   'centers': centers.tolist(), 'gradients': gradients}, file=sys.stderr)
            del model, prediction
            gc.collect()
        report['model'] = check_result('PASS', 'both fusion modes: CPU forward/backward and renumbering')
    except _AuditStop as stop:
        exit_code = stop.code
    except (FileNotFoundError, KeyError) as exc:
        missing_record = isinstance(exc, FileNotFoundError)
        exit_code = 1 if missing_record else 2
        report['outcome'] = 'DATA_ANOMALY' if missing_record else 'SCRIPT_ERROR'
        if report['model']['status'] == 'RUNNING':
            report['model'] = check_result('FAIL', str(exc))
        report['error'] = f'{type(exc).__name__}: {exc}'
    except Exception as exc:
        exit_code = 2
        running = report['model']['status'] == 'RUNNING'
        report['outcome'] = ('MODEL_FAILURE' if running and isinstance(exc, (
            AssertionError, RuntimeError, ValueError, FloatingPointError)) else 'SCRIPT_ERROR')
        if running:
            report['model'] = check_result('FAIL', str(exc))
        report['error'] = f'{type(exc).__name__}: {exc}'
        import traceback
        traceback.print_exc()
    finally:
        finalization_errors = []
        if 'fixture_coverage' not in report:
            try:
                report['fixture_coverage'] = fixture_coverage(report['samples'])
            except Exception as exc:
                finalization_errors.append(
                    f'fixture_coverage: {type(exc).__name__}: {exc}'
                )
                report['fixture_coverage'] = check_result(
                    'ANOMALY', 'fixture coverage could not be computed'
                )
                report['outcome'] = 'SCRIPT_ERROR'
                report['finalization_errors'] = list(finalization_errors)
        if source is not None:
            try:
                source.close()
            except Exception as exc:
                finalization_errors.append(f'close: {type(exc).__name__}: {exc}')
                report['outcome'] = 'SCRIPT_ERROR'
                report['finalization_errors'] = finalization_errors
        try:
            report_text = json.dumps(
                report, ensure_ascii=False, indent=2, allow_nan=False
            )
        except Exception as exc:
            finalization_errors.append(
                f'report_serialization: {type(exc).__name__}: {exc}'
            )
            report['outcome'] = 'SCRIPT_ERROR'
            report['finalization_errors'] = list(finalization_errors)
            report_text = json.dumps(
                _json_safe(report), ensure_ascii=False, indent=2, allow_nan=False
            )
        if args.report_json is not None:
            try:
                with args.report_json.open('x', encoding='utf-8') as stream:
                    stream.write(report_text)
                    stream.write('\n')
            except Exception as exc:
                finalization_errors.append(f'report: {type(exc).__name__}: {exc}')
                report['outcome'] = 'SCRIPT_ERROR'
                report['finalization_errors'] = list(finalization_errors)
        try:
            stdout_text = json.dumps(
                report, ensure_ascii=False, allow_nan=False
            )
        except Exception as exc:
            finalization_errors.append(
                f'stdout_serialization: {type(exc).__name__}: {exc}'
            )
            report['outcome'] = 'SCRIPT_ERROR'
            report['finalization_errors'] = list(finalization_errors)
            stdout_text = json.dumps(
                _json_safe(report), ensure_ascii=False, allow_nan=False
            )
        print(stdout_text)
        if finalization_errors:
            # A close/report failure is a script failure, but it must not
            # replace the complete stdout JSON with an uncaught traceback or
            # downgrade the exit status to the data-anomaly code.
            exit_code = 2
    return exit_code


if __name__ == '__main__':
    sys.exit(main())
