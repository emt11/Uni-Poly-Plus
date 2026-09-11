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

import torch
from rdkit import Chem
from torch.utils.data import DataLoader
from src.dataset.glt_dual import FrozenDualLayerSource, DualGLTDataset, dual_glt_collate, build_dual_sample
from src.dataset.graph_data import build_periodic_multimer_mol
from src.dataset.glt_bond_chemistry import bond_feature_vector


def check_result(status, reason='', **details):
    return dict(status=status, reason=reason, **details)


def audit_source_stereo(source, mol, meta):
    """Compare pristine input metadata with ALL copies, independently of copying.

    Compare the relative substituent convention, allowing a terminal implicit-H
    replacement to reverse that convention. This is metadata preservation, not
    a claim that CIP priorities of the finite molecule equal those of P-SMILES.
    """
    real = [a.GetIdx() for a in source.GetAtoms() if a.GetAtomicNum() != 0]
    dummies = [a.GetIdx() for a in source.GetAtoms() if a.GetAtomicNum() == 0]
    if len(dummies) != 2 or any(source.GetAtomWithIdx(i).GetDegree() != 1 for i in dummies):
        raise ValueError('source must have two degree-one attachment atoms')
    base = {old: i for i, old in enumerate(real)}
    boundary = [source.GetAtomWithIdx(i).GetNeighbors()[0].GetIdx() for i in dummies]
    units = meta['unit_atoms']
    signs = {Chem.BondStereo.STEREOE: -1, Chem.BondStereo.STEREOTRANS: -1,
             Chem.BondStereo.STEREOZ: 1, Chem.BondStereo.STEREOCIS: 1}
    failures, snapshots = [], []
    retained = terminal_undefined = center_specified = 0
    for original in source.GetBonds():
        a, b = original.GetBeginAtomIdx(), original.GetEndAtomIdx()
        snapshots.append(dict(atoms=[a, b], bond_type=str(original.GetBondType()),
            stereo=str(original.GetStereo()), references=list(original.GetStereoAtoms()),
            direction=str(original.GetBondDir())))
        if a not in base or b not in base:
            continue  # attachment bonds are replaced by new physical seams
        if original.GetStereo() in signs:
            center_specified += 1
        for q, unit in enumerate(units):
            i, j = unit[base[a]], unit[base[b]]
            copied = mol.GetBondBetweenAtoms(i, j)
            tag = dict(source_bond=[a, b], unit=q, physical_bond=[i, j])
            if copied is None:
                failures.append(dict(**tag, reason='missing copied bond'))
                continue
            if copied.GetBondDir() != original.GetBondDir():
                failures.append(dict(**tag, reason='BondDir changed'))
            if original.GetStereo() not in signs:
                if copied.GetStereo() != original.GetStereo():
                    failures.append(dict(**tag, reason='unspecified/unknown stereo changed'))
                continue
            # Each expected source substituent maps either to a real neighbor
            # or to a terminal implicit H. Do not use builder reference maps.
            expected = []
            for ref in original.GetStereoAtoms():
                if ref in base:
                    expected.append(unit[base[ref]])
                else:
                    side = dummies.index(ref)
                    neighbor_unit = q - 1 if side == 0 else q + 1
                    expected.append(units[neighbor_unit][base[boundary[1 - side]]]
                                    if 0 <= neighbor_unit < len(units) else None)
            if len(expected) != 2:
                failures.append(dict(**tag, reason='source stereo lacks two references'))
                continue
            substituents = [{n.GetIdx() for n in mol.GetAtomWithIdx(end).GetNeighbors()
                             if n.GetIdx() != other} for end, other in ((i, j), (j, i))]
            if any(ref is None and not neighbors for ref, neighbors in zip(expected, substituents)):
                terminal_undefined += 1
                if copied.GetStereo() != Chem.BondStereo.STEREONONE:
                    failures.append(dict(**tag, reason='terminal bond no longer has defined stereo'))
                continue
            actual = list(copied.GetStereoAtoms())
            if copied.GetBeginAtomIdx() != i:
                actual.reverse()
            if copied.GetStereo() not in signs or len(actual) != 2:
                failures.append(dict(**tag, reason='preservable stereo lost'))
                continue
            if any(ref not in neighbors for ref, neighbors in zip(actual, substituents)):
                failures.append(dict(**tag, reason='stereo reference is not a real neighbor'))
                continue
            parity = sum(ref != wanted for ref, wanted in zip(actual, expected)) % 2
            if signs[copied.GetStereo()] != signs[original.GetStereo()] * (-1 if parity else 1):
                failures.append(dict(**tag, reason='relative stereo changed'))
            else:
                retained += 1
    return check_result('ANOMALY' if failures else 'PASS', failures=failures,
        source_bonds=snapshots, retained_copies=retained, terminal_undefined=terminal_undefined,
        center_specified=center_specified,
        expected_center_bonds=sum(b.GetBeginAtomIdx() in base and b.GetEndAtomIdx() in base
                                  for b in source.GetBonds()))


def audit_record(topology, trimer, smiles):
    """Audit the supplied record only; never generate or modify coordinates."""
    names = ('chemistry', 'source_stereo', 'connection_policy', 'translation_chemistry',
             'seam_stereo', 'identity', 'connection_geometry', 'stereo_coordinates', 'model_input')
    report = dict(smiles=smiles,
                  checks={name: check_result('NOT_RUN', 'prerequisite unavailable') for name in names},
                  connections=[])
    checks = report['checks']
    try:
        source = Chem.MolFromSmiles(smiles)
        if source is None:
            raise ValueError('invalid source P-SMILES')
        mol, meta = build_periodic_multimer_mol(smiles, 3, close_periodic=False)
    except ValueError as exc:
        checks['chemistry'] = check_result('ANOMALY', str(exc))
        checks['identity'] = checks['stereo_coordinates'] = check_result('NOT_RUN', 'chemistry unavailable')
        return report
    checks['chemistry'] = check_result('PASS')
    checks['source_stereo'] = audit_source_stereo(source, mol, meta)
    report['connection_policy'] = {name: meta[name] for name in (
        'attachment_bond_type_left', 'attachment_bond_type_right', 'connection_bond_policy')}
    report['connection_policy']['actual_type'] = str(meta['attachment_bond_type'])
    policy = meta['connection_bond_policy']
    checks['connection_policy'] = check_result(
        'PASS' if policy == 'matching_attachment_type' else 'REVIEW',
        '' if policy == 'matching_attachment_type' else 'policy requires review; not proof of invalid chemistry')
    size = meta['base_atom_count']
    chemistry = {}
    differences = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        key = min((i % size, j % size, j // size - i // size),
                  (j % size, i % size, i // size - j // size))
        feature = bond_feature_vector(bond).tolist()
        if key in chemistry and chemistry[key] != feature:
            differences.append(dict(identity=list(key), bond=[i, j], features=feature))
        chemistry.setdefault(key, feature)
        if i // size != j // size:
            report['connections'].append(dict(atoms=[i, j], bond_type=str(bond.GetBondType()),
                stereo=str(bond.GetStereo()), conjugation=bool(bond.GetIsConjugated()),
                ring=bool(bond.IsInRing()), distance=None, angles=[], geometry_status='NOT_RUN'))
    checks['translation_chemistry'] = check_result('REVIEW' if differences else 'PASS',
        'finite terminal copies are not periodic 2D representatives' if differences else '',
        differences=differences)
    unspecified = [edge['atoms'] for edge in report['connections']
                   if edge['bond_type'] == 'DOUBLE' and edge['stereo'] in ('STEREONONE', 'STEREOANY')]
    checks['seam_stereo'] = check_result('REVIEW' if unspecified else 'NOT_APPLICABLE',
        'cross-RU double bond stereo unspecified; no E/Z inferred' if unspecified else 'no unspecified seam double bond',
        bonds=unspecified)
    from src.dataset.periodic_line_glt_complete import build_complete_trimer_glt_sample
    row = build_complete_trimer_glt_sample(topology, trimer, smiles)
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
        'PASS' if all(e['geometry_status'] == 'PASS' for e in report['connections']) else 'ANOMALY')
    if checks['source_stereo']['status'] == 'PASS':
        try:
            count = audit_frozen_stereo(trimer, smiles)
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


def audit_frozen_stereo(trimer, smiles):
    """Check every retained specified E/Z against existing coordinates."""
    mol, meta = build_periodic_multimer_mol(smiles, 3, close_periodic=False)
    base = getattr(trimer, 'trimer_base_ru_atom_id', None)
    if base is None:
        base = trimer.trimer_base_ru_atom_index
    lookup = {(int(b), int(q)): i for i, (b, q) in enumerate(zip(base, trimer.trimer_ru_offset))}
    size = meta['base_atom_count']
    positions = trimer.trimer_pos.double()
    def point(index):
        return positions[lookup[index % size, index // size - 1]]
    checked = 0
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        stereo = bond.GetStereo()
        if stereo not in (Chem.BondStereo.STEREOE, Chem.BondStereo.STEREOZ,
                          Chem.BondStereo.STEREOCIS, Chem.BondStereo.STEREOTRANS):
            continue
        refs = list(bond.GetStereoAtoms())
        if len(refs) != 2:
            raise ValueError(f'missing stereo references at {(i, j)}')
        left, right = refs
        axis = point(j) - point(i)
        axis_norm = axis.norm()
        if not torch.isfinite(axis_norm) or axis_norm <= 1e-12:
            raise ValueError(f'degenerate stereo axis at {(i, j)}')
        axis = axis / axis_norm
        u, v = point(left) - point(i), point(right) - point(j)
        u, v = u - u.dot(axis) * axis, v - v.dot(axis) * axis
        denominator = u.norm() * v.norm()
        if not torch.isfinite(denominator) or denominator <= 1e-12:
            raise ValueError(f'degenerate stereo projection at {(i, j)}')
        cosine = u.dot(v) / denominator
        expected_sign = -1 if stereo in (Chem.BondStereo.STEREOE, Chem.BondStereo.STEREOTRANS) else 1
        if not torch.isfinite(cosine) or expected_sign * cosine < 0.5:
            raise ValueError(f'frozen coordinates disagree with specified bond stereo at {(i, j)}; cache unchanged')
        checked += 1
    return checked


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
    """Bind roles to each record and independently count source internal bonds."""
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
            elif expected is not None and expected > 0 and source.get('center_specified', 0) > 0:
                role = 'ordinary_ez'
        roles.append(role)
    passed = len(roles) == 2 and set(roles) == {'n0', 'ordinary_ez'}
    return check_result('PASS' if passed else 'ANOMALY',
        '' if passed else 'requires ordinary central E/Z and independently verified real N=0', roles=roles)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--topology-root', required=True)
    parser.add_argument('--trimer-root', required=True)
    parser.add_argument('--sample', action='append', nargs=2, metavar=('HEXKEY', 'PSMILES'), required=True)
    parser.add_argument('--audit-only', action='store_true', help='inspect data without constructing a model')
    parser.add_argument('--report-json', type=Path, help='new report path; existing files are not overwritten')
    args = parser.parse_args()
    if len(args.sample) != 2:
        parser.error('provide exactly two records: one ordinary and one real N=0')
    if args.report_json is not None and args.report_json.exists():
        parser.error('report path already exists; choose a new path')
    torch.set_num_threads(1)
    try:
        samples = [(bytes.fromhex(key), smiles) for key, smiles in args.sample]
        if any(len(key) != 32 for key, _ in samples):
            raise ValueError('sample keys must have 32 bytes')
        if len({key for key, _ in samples}) != 2:
            raise ValueError('provide two distinct frozen records')
    except ValueError as exc:
        parser.error(str(exc))
    report = dict(scope='only the two explicitly supplied records; not full-cache coverage',
                  samples=[], model=check_result('NOT_RUN'), outcome='NOT_RUN')
    source = None
    exit_code = 0
    try:
        try:
            source = FrozenDualLayerSource(args.topology_root, args.trimer_root, samples)
        except ValueError as exc:
            report['outcome'] = 'DATA_ANOMALY'
            report['error'] = f'{type(exc).__name__}: {exc}'
            return 1
        records = []
        for index, (key, smiles) in enumerate(samples):
            try:
                record = source[index]
            except KeyError as exc:
                report['samples'].append(dict(key=key.hex(), smiles=smiles,
                    checks={'record': check_result('ANOMALY', f'missing frozen record: {exc}')}))
                continue
            records.append(record)
            entry = audit_record(*record)
            entry['key'] = key.hex()
            report['samples'].append(entry)
        report['fixture_coverage'] = fixture_coverage(report['samples'])
        anomalies = any(c['status'] == 'ANOMALY' for e in report['samples'] for c in e['checks'].values())
        if anomalies or report['fixture_coverage']['status'] != 'PASS':
            report['outcome'] = 'DATA_ANOMALY'
            exit_code = 1
            return exit_code
        reviews = any(c['status'] == 'REVIEW' for e in report['samples'] for c in e['checks'].values())
        report['outcome'] = 'REVIEW' if reviews else 'PASS'
        if args.audit_only:
            report['model'] = check_result('NOT_RUN', '--audit-only: no model constructed')
            return exit_code
        report['model'] = check_result('RUNNING')
        from src.modules.glt_dual import build_dual_glt_model
        dataset = DualGLTDataset(source)
        batch = next(iter(DataLoader(dataset, batch_size=2, num_workers=0, collate_fn=dual_glt_collate)))
        centers = torch.bincount(batch.bond_batch[batch.bond_center], minlength=2)
        if not batch.geometry_valid.all() or not (centers == 0).any() or not (centers > 0).any():
            raise RuntimeError('fixtures must contain valid ordinary and N=0 geometry')
        checked = sum(audit_frozen_stereo(trimer, smiles) for _, trimer, smiles in records)
        if checked == 0:
            raise RuntimeError('ordinary real fixture must contain specified central E/Z; do not claim stereo coverage')
        renumbered = dual_glt_collate([renumber_frozen(*record) for record in records])
        if not renumbered.geometry_valid.all():
            raise RuntimeError('renumbering invalidated real geometry')
        for mode in ('concat', 'kfuse'):
            torch.manual_seed(42)
            model = build_dual_glt_model(mode, dropout=0).cpu()
            prediction = model(batch)
            torch.testing.assert_close(prediction, model(renumbered), atol=1e-5, rtol=1e-5)
            print({'mode': mode, 'stereo_bonds_checked': checked, 'renumbering': 'PASS'}, file=sys.stderr)
            if prediction.shape != (2, 1) or not torch.isfinite(prediction).all():
                raise RuntimeError('invalid predictions')
            (prediction - torch.tensor([[0.3], [-0.7]])).square().mean().backward()
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
        if source is not None:
            try:
                source.close()
            except Exception as exc:
                finalization_errors.append(f'close: {type(exc).__name__}: {exc}')
                report['outcome'] = 'SCRIPT_ERROR'
                report['finalization_errors'] = finalization_errors
        if args.report_json is not None:
            try:
                with args.report_json.open('x', encoding='utf-8') as stream:
                    json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
            except Exception as exc:
                finalization_errors.append(f'report: {type(exc).__name__}: {exc}')
                report['outcome'] = 'SCRIPT_ERROR'
                report['finalization_errors'] = finalization_errors
        print(json.dumps(report, ensure_ascii=False, allow_nan=False))
        if finalization_errors:
            raise RuntimeError('; '.join(finalization_errors))
    return exit_code


if __name__ == '__main__':
    sys.exit(main())
