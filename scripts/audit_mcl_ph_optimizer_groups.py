#!/usr/bin/env python3
"""Read-only supplementary audit of downstream units written before r2.

The r1 fine-tuning runner wrote ``optimizer_groups`` into ``run.json`` and into
``best.pt`` but not into ``metrics.json``, so the strict aggregator cannot accept
those units.  The r2 fix only changes future writes: this script never rewrites a
unit, never re-runs training and never fabricates a raw field.  It reads the
unit's own records, checks that the value is identical in the two places that do
carry it, cross-checks the fields all three records share, and writes one
separate report stating exactly which value is missing, where its evidence
lives, and whether that evidence agrees.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from scripts.finetune_mcl_ph import ARMS, unit_directory

GROUP_NAMES = {'backbone', 'backbone_no_decay', 'head', 'head_no_decay'}
SHARED_FIELDS = ('arm', 'task', 'fold', 'stage', 'executed_epochs', 'best_epoch',
                 'best_validation_r2', 'pretrain_step', 'pretrain_package_sha256',
                 'optimizer_updates')


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _groups_problems(groups):
    problems = []
    if not isinstance(groups, list) or not groups:
        return ['the record carries no parameter group']
    names = {group.get('name') for group in groups}
    if names - GROUP_NAMES:
        problems.append('undeclared group names: ' + ','.join(sorted(names - GROUP_NAMES)))
    for group in groups:
        if int(group.get('num_parameters', 0)) <= 0:
            problems.append(f"group {group.get('name')!r} has no trainable parameter")
        if int(group.get('num_tensors', 0)) <= 0:
            problems.append(f"group {group.get('name')!r} has no tensor")
    return problems


def audit_unit(root, arm, task, fold):
    directory = unit_directory(root, arm, task, fold)
    problems, evidence = [], {}
    paths = {name: directory / name for name in ('run.json', 'metrics.json', 'best.pt')}
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        return {'arm': arm, 'task': task, 'fold': int(fold), 'status': 'INCOMPLETE_UNIT',
                'problems': ['missing artifacts: ' + ','.join(sorted(missing))]}
    run = json.loads(paths['run.json'].read_text(encoding='utf-8'))
    metrics = json.loads(paths['metrics.json'].read_text(encoding='utf-8'))
    checkpoint = torch.load(paths['best.pt'], map_location='cpu', weights_only=False)
    evidence['run_json_sha256'] = sha256_file(paths['run.json'])
    evidence['metrics_json_sha256'] = sha256_file(paths['metrics.json'])
    evidence['best_pt_sha256'] = sha256_file(paths['best.pt'])
    evidence['metrics_json_has_optimizer_groups'] = 'optimizer_groups' in metrics
    run_groups = run.get('optimizer_groups')
    checkpoint_groups = checkpoint.get('optimizer_groups')
    if run_groups in (None, []) and checkpoint_groups in (None, []):
        return {'arm': arm, 'task': task, 'fold': int(fold), 'status': 'NO_EVIDENCE',
                'problems': ['neither run.json nor best.pt carries the parameter groups'],
                'evidence': evidence}
    problems += _groups_problems(run_groups)
    problems += _groups_problems(checkpoint_groups)
    if run_groups != checkpoint_groups:
        problems.append('run.json and best.pt disagree on the parameter groups')
    for name in SHARED_FIELDS:
        left, right = run.get(name), metrics.get(name)
        if name == 'fold':
            left, right = int(left), int(right)
        if name == 'best_validation_r2':
            if not (isinstance(left, (int, float)) and isinstance(right, (int, float))
                    and abs(float(left) - float(right)) <= 1e-12):
                problems.append(f'{name} disagrees between run.json and metrics.json')
            continue
        if left != right:
            problems.append(f'{name} disagrees between run.json and metrics.json')
    for name in ('arm', 'task', 'fold'):
        value = checkpoint.get(name)
        if name == 'fold':
            value = int(value) if value is not None else None
        if value != run.get(name):
            problems.append(f'{name} disagrees between best.pt and run.json')
    for name in ('best_epoch', 'executed_epochs', 'optimizer_updates'):
        if name == 'optimizer_updates':
            continue
        if int(checkpoint.get(name, -1)) != int(metrics.get(name, -2)):
            problems.append(f'{name} disagrees between best.pt and metrics.json')
    status = 'COMPLETE_FROM_OWN_RECORDS' if not problems else 'INCONSISTENT'
    return {'arm': arm, 'task': task, 'fold': int(fold), 'status': status,
            'problems': problems, 'evidence': evidence,
            'missing_field': 'metrics.json:optimizer_groups',
            'value_from_run_json': run_groups,
            'value_from_best_pt': checkpoint_groups,
            'values_identical': run_groups == checkpoint_groups}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--arms', nargs='+', default=list(ARMS))
    parser.add_argument('--tasks', nargs='+', default=['xc'])
    parser.add_argument('--folds', nargs='+', type=int, default=[0])
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    units = [audit_unit(args.root, arm, task, fold) for arm in args.arms
             for task in args.tasks for fold in args.folds]
    complete = [unit for unit in units if unit['status'] == 'COMPLETE_FROM_OWN_RECORDS']
    payload = {
        'plan': 'MCL-PH-20260921-01/r2',
        'purpose': 'read-only evidence for the optimizer_groups field the r1 runner did not '
                   'write into metrics.json',
        'root': str(Path(args.root).resolve()),
        'units': units, 'units_checked': len(units),
        'units_complete_from_own_records': len(complete),
        'units_inconsistent': len(units) - len(complete),
        'acceptance': 'PARTIAL' if len(complete) != len(ARMS) * len(args.tasks) * len(args.folds)
                      else 'PASS',
        'metrics_json_modified': False,
        'training_rerun': False,
        'note': 'the value is reported, not injected: metrics.json of every unit is left '
                'exactly as the r1 run wrote it',
    }
    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + '\n'
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(text, encoding='utf-8')
    print(text, end='')
    if payload['units_inconsistent']:
        raise SystemExit(4)
    if payload['acceptance'] != 'PASS':
        raise SystemExit(5)


if __name__ == '__main__':
    main()
