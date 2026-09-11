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


def audit_record(topology, trimer, smiles):
    """Audit the supplied record only; never generate or modify coordinates."""
    report = dict(smiles=smiles, checks={}, connections=[])
    checks = report['checks']
    try:
        mol, meta = build_periodic_multimer_mol(smiles, 3, close_periodic=False)
    except ValueError as exc:
        checks['chemistry'] = check_result('ANOMALY', str(exc))
        checks['identity'] = checks['stereo_coordinates'] = check_result('NOT_RUN', 'chemistry unavailable')
        return report
    report['connection_policy'] = {name: meta[name] for name in (
        'attachment_bond_type_left', 'attachment_bond_type_right', 'connection_bond_policy')}
    report['connection_policy']['actual_type'] = str(meta['attachment_bond_type'])
    policy = meta['connection_bond_policy']
    checks['connection_policy'] = check_result(
        'PASS' if policy == 'matching_attachment_type' else 'ANOMALY',
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
    checks['translation_chemistry'] = check_result('ANOMALY' if differences else 'PASS', differences=differences)
    unspecified = [edge['atoms'] for edge in report['connections']
                   if edge['bond_type'] == 'DOUBLE' and edge['stereo'] in ('STEREONONE', 'STEREOANY')]
    checks['seam_stereo'] = check_result('ANOMALY' if unspecified else 'NOT_APPLICABLE',
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
        left, right = bond.GetStereoAtoms()
        axis = point(j) - point(i)
        axis = axis / axis.norm().clamp_min(1e-12)
        u, v = point(left) - point(i), point(right) - point(j)
        u, v = u - u.dot(axis) * axis, v - v.dot(axis) * axis
        cosine = u.dot(v) / (u.norm() * v.norm()).clamp_min(1e-12)
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
    except ValueError as exc:
        parser.error(str(exc))
    report = dict(scope='only the two explicitly supplied records; not full-cache coverage',
                  samples=[], model=check_result('NOT_RUN'), outcome='NOT_RUN')
    source = None
    exit_code = 0
    try:
        source = FrozenDualLayerSource(args.topology_root, args.trimer_root, samples)
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
        counts = [e['checks'].get('model_input', {}).get('center_bonds') for e in report['samples']]
        coverage = len(records) == 2 and 0 in counts and any(n is not None and n > 0 for n in counts)
        # Ordinary sample must exercise explicitly specified central stereo.
        central_stereo = False
        for _, _, smiles in records:
            mol, meta = build_periodic_multimer_mol(smiles, 3, close_periodic=False)
            center = set(meta['unit_atoms'][1])
            central_stereo |= any(b.GetBeginAtomIdx() in center and b.GetEndAtomIdx() in center
                and b.GetStereo() in (Chem.BondStereo.STEREOE, Chem.BondStereo.STEREOZ,
                                     Chem.BondStereo.STEREOCIS, Chem.BondStereo.STEREOTRANS)
                for b in mol.GetBonds())
        report['fixture_coverage'] = check_result('PASS' if coverage and central_stereo else 'ANOMALY',
            '' if coverage and central_stereo else 'requires ordinary central E/Z and real N=0; no synthetic replacement')
        anomalies = any(c['status'] == 'ANOMALY' for e in report['samples'] for c in e['checks'].values())
        if anomalies or not coverage or not central_stereo:
            report['outcome'] = 'DATA_ANOMALY'
            exit_code = 1
            return exit_code
        report['outcome'] = 'PASS'
        if args.audit_only:
            report['model'] = check_result('NOT_RUN', '--audit-only: no model constructed')
            return exit_code
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
            print({'mode': mode, 'stereo_bonds_checked': checked, 'renumbering': 'PASS'})
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
                   'centers': centers.tolist(), 'gradients': gradients})
            del model, prediction
            gc.collect()
        report['model'] = check_result('PASS', 'both fusion modes: CPU forward/backward and renumbering')
    except (FileNotFoundError, KeyError) as exc:
        exit_code = 1
        report['outcome'] = 'DATA_ANOMALY'
        report['error'] = f'{type(exc).__name__}: {exc}'
    except Exception as exc:
        exit_code = 2
        report['outcome'] = 'SCRIPT_ERROR'
        report['error'] = f'{type(exc).__name__}: {exc}'
        import traceback
        traceback.print_exc()
    finally:
        if source is not None:
            source.close()
        print(json.dumps(report, ensure_ascii=False, allow_nan=False))
        if args.report_json is not None:
            with args.report_json.open('x', encoding='utf-8') as stream:
                json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
    return exit_code


if __name__ == '__main__':
    sys.exit(main())
