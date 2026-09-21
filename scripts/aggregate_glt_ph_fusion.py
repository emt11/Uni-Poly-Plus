#!/usr/bin/env python3
"""Aggregate GLT-PH fusion fine-tuning units and apply the pre-registered gates.

The gates are the plan's own numbers and are never adjusted to a result: a
candidate is promoted only when it clears both controls (section 8).  Units are
rejected rather than partially aggregated when they are incomplete, carry an
outer-test field, are missing the checkpoint identity, or hold non-finite
metrics.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

from src.training.glt_dual_runtime import write_json

ARMS = ('R0', 'R2D', 'CONST', 'STAT', 'PH', 'CURRENT')
GAIN_THRESHOLD = 0.005
XC_THRESHOLD = 0.01
RISK_THRESHOLD = 0.01
CONTROLS = ('R0', 'CONST')
REPORT_KEYS = ('plan', 'arm', 'task', 'fold', 'readout', 'best_validation_r2',
               'best_epoch', 'epochs_run', 'epochs_configured', 'validation_only',
               'outer_test', 'complete', 'stage_completed', 'checkpoint', 'mode')


def load_unit(path):
    payload = json.loads(Path(path).read_text(encoding='utf-8'))
    missing = [key for key in REPORT_KEYS if key not in payload]
    if missing:
        raise ValueError(f'{path}: unit is missing {missing}')
    if payload['complete'] is not True or payload.get('stage_completed') != 'diagnostics':
        raise ValueError(f'{path}: unit is not a completed run (complete/stage)')
    if payload['outer_test'] != 'NOT_RUN' or payload['validation_only'] is not True:
        raise ValueError(f'{path}: unit is not validation-only')
    value = float(payload['best_validation_r2'])
    if not np.isfinite(value):
        raise ValueError(f'{path}: non-finite validation R2')
    if int(payload['epochs_run']) > int(payload['epochs_configured']):
        raise ValueError(f'{path}: epochs_run exceeds the configured budget')
    if not payload.get('checkpoint'):
        raise ValueError(f'{path}: unit records no checkpoint identity')
    return payload


def collect(root):
    units = {}
    for path in sorted(Path(root).glob('*/*/fold*/metrics.json')):
        payload = load_unit(path)
        key = (payload['arm'], payload['task'], int(payload['fold']))
        if key in units:
            raise ValueError(f'duplicate unit {key}')
        payload['path'] = str(path)
        units[key] = payload
    return units


def arm_summary(units, arm):
    per_task, per_fold = {}, {}
    for (unit_arm, task, fold), payload in sorted(units.items()):
        if unit_arm != arm:
            continue
        per_fold[f'{task}/fold{fold}'] = float(payload['best_validation_r2'])
        per_task.setdefault(task, []).append(float(payload['best_validation_r2']))
    task_means = {task: float(np.mean(values)) for task, values in sorted(per_task.items())}
    macro = float(np.mean(list(task_means.values()))) if task_means else float('nan')
    return {'per_fold': per_fold, 'per_task': task_means, 'macro': macro,
            'epochs_used': sum(int(units[key]['epochs_run']) for key in units
                               if key[0] == arm)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, help='fine-tuning results root')
    parser.add_argument('--output', required=True)
    parser.add_argument('--reference', default='R0')
    parser.add_argument('--candidate', default='PH')
    args = parser.parse_args()
    units = collect(args.root)
    if not units:
        raise SystemExit('no complete unit found under the results root')
    summary = {arm: arm_summary(units, arm) for arm in ARMS
               if any(key[0] == arm for key in units)}
    report = {'plan': 'GLT-PH-END2END-20260920-01', 'units': len(units),
              'arms': summary,
              'criteria': {'gain_threshold': GAIN_THRESHOLD,
                           'xc_threshold': XC_THRESHOLD,
                           'risk_threshold': RISK_THRESHOLD,
                           'note': ('engineering screen, not a significance test; '
                                    'two folds per task are not a significance test')}}
    deltas = {}
    candidate = summary.get(args.candidate)
    if candidate:
        for control in CONTROLS:
            other = summary.get(control)
            if not other:
                continue
            per_task = {task: candidate['per_task'].get(task, float('nan'))
                        - other['per_task'].get(task, float('nan'))
                        for task in sorted(set(candidate['per_task']) | set(other['per_task']))}
            deltas[f'{args.candidate}_MINUS_{control}'] = {
                'macro': candidate['macro'] - other['macro'], 'per_task': per_task}
        for name, value in deltas.items():
            control = name.split('_MINUS_')[1]
            other = summary.get(control)
            if not other:
                continue
            risk = [task for task, delta in value['per_task'].items()
                    if delta < -RISK_THRESHOLD]
            value['risk_flags'] = risk
            value['macro_ok'] = bool(value['macro'] >= GAIN_THRESHOLD)
            xc_delta = value['per_task'].get('xc')
            value['xc_mean'] = None if xc_delta is None else float(xc_delta)
            folds = {}
            for index in (0, 1):
                left = candidate['per_fold'].get(f'{args.candidate}/xc/fold{index}')
                right = other['per_fold'].get(f'{control}/xc/fold{index}')
                if left is not None and right is not None:
                    folds[str(index)] = float(left - right)
            value['xc_folds'] = folds
            # An XC claim needs both folds present and non-degrading.
            value['xc_ok'] = bool(xc_delta is not None and xc_delta >= XC_THRESHOLD
                                  and len(folds) == 2
                                  and all(item > 0 for item in folds.values()))
    report['deltas'] = deltas
    if deltas:
        verdicts = [name for name, value in deltas.items()
                    if value.get('macro_ok') and value.get('xc_ok')]
        report['VERDICT'] = ('EXPLORATION_CANDIDATE' if verdicts == list(deltas)
                             else 'NO_CANDIDATE')
        report['passing_against'] = [name.split('_MINUS_')[1] for name in verdicts]
    else:
        report['VERDICT'] = 'INCOMPLETE'
    write_json(args.output, report)
    print(json.dumps({'units': len(units), 'arms': sorted(summary),
                      'VERDICT': report['VERDICT'],
                      'macro': {arm: summary[arm]['macro'] for arm in summary}}, indent=1),
          flush=True)


if __name__ == '__main__':
    main()
