#!/usr/bin/env python3
"""Aggregate the PH retention development units and apply the §7 science gate.

Reads ``results/glt_galph_ph_retention_20260920/p2/<group>/<task>/fold<k>/metrics.json``
for the three groups, reports per-unit and per-task numbers, the three deltas and
the pre-registered engineering criteria.  Any unit that touched the outer test is
refused.
"""
import argparse
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

GROUPS = ('F_OFF', 'F_CONST', 'F_REAL')
CONTROLS = ('F_OFF', 'F_CONST')
TASKS = ('xc', 'eps', 'eat')
FOLDS = (0, 1)
GAIN_THRESHOLD = 0.005
XC_THRESHOLD = 0.01
RISK_THRESHOLD = 0.01


def load_units(root):
    units = {}
    for group in GROUPS:
        for task in TASKS:
            for fold in FOLDS:
                path = Path(root) / group / task / f'fold{fold}' / 'metrics.json'
                if not path.is_file():
                    raise FileNotFoundError(f'missing development unit: {path}')
                row = json.loads(path.read_text(encoding='utf-8'))
                if row.get('outer_test') != 'NOT_RUN':
                    raise ValueError(f'unit touched the outer test: {path}')
                if row.get('protocol') != 'ph_retention_development':
                    raise ValueError(f'unit was not a development run: {path}')
                units[(group, task, fold)] = row
    return units


def summarize(root):
    units = load_units(root)
    r2 = {key: float(row['best_validation_r2']) for key, row in units.items()}
    per_task = {group: {task: statistics.fmean(r2[(group, task, fold)] for fold in FOLDS)
                        for task in TASKS} for group in GROUPS}
    overall = {group: statistics.fmean(per_task[group][task] for task in TASKS)
               for group in GROUPS}
    deltas = {}
    for name, left, right in (('F_REAL_MINUS_F_OFF', 'F_REAL', 'F_OFF'),
                              ('F_REAL_MINUS_F_CONST', 'F_REAL', 'F_CONST'),
                              ('F_CONST_MINUS_F_OFF', 'F_CONST', 'F_OFF')):
        deltas[name] = {
            'mean': overall[left] - overall[right],
            'per_task': {task: per_task[left][task] - per_task[right][task] for task in TASKS},
            'per_fold': {f'{task}/fold{fold}': r2[(left, task, fold)] - r2[(right, task, fold)]
                         for task in TASKS for fold in FOLDS},
        }
    # §7 engineering criteria (thresholds, not significance)
    mean_ok = all(deltas[f'F_REAL_MINUS_{control}']['mean'] >= GAIN_THRESHOLD
                  for control in CONTROLS)
    xc = {control: {'mean': deltas[f'F_REAL_MINUS_{control}']['per_task']['xc'],
                    'folds': [deltas[f'F_REAL_MINUS_{control}']['per_fold'][f'xc/fold{fold}']
                              for fold in FOLDS]}
          for control in CONTROLS}
    xc_ok = all(xc[control]['mean'] >= XC_THRESHOLD
                and all(value >= 0 for value in xc[control]['folds']) for control in CONTROLS)
    risk = sorted({f'{task} vs {control}' for control in CONTROLS for task in TASKS
                   if deltas[f'F_REAL_MINUS_{control}']['per_task'][task] < -RISK_THRESHOLD})
    improved = [task for task in TASKS
                if all(deltas[f'F_REAL_MINUS_{control}']['per_task'][task] > 0
                       for control in CONTROLS)]
    verdict = ('STOP: F_REAL does not reach the three-task gain threshold against both '
               'controls' if not mean_ok else
               'XC_CLAIM_SUPPORTED' if xc_ok else
               'GAIN_WITHOUT_XC_CLAIM' if improved else 'GAIN_NOT_TASK_CONSISTENT')
    return {
        'units': {f'{key[0]}/{key[1]}/fold{key[2]}': {
            'best_validation_r2': r2[key], 'best_epoch': int(units[key]['best_epoch']),
            'wall_seconds': float(units[key]['wall_seconds']),
            'coverage': units[key].get('coverage'),
            'diagnostics': units[key].get('diagnostics'),
            'checkpoint_sha256': units[key].get('checkpoint_sha256'),
            'ph_encoder_version': units[key].get('ph_encoder_version'),
            'trainable_parameter_count': units[key].get('trainable_parameter_count'),
            'frozen_training_only_heads': units[key].get('frozen_training_only_heads'),
        } for key in sorted(units)},
        'arm_mean_r2': overall, 'arm_per_task_r2': per_task,
        'arm_per_fold_r2': {f'{group}/{task}/fold{fold}': r2[(group, task, fold)]
                            for group in GROUPS for task in TASKS for fold in FOLDS},
        'deltas': deltas,
        'criteria': {'gain_threshold': GAIN_THRESHOLD, 'xc_threshold': XC_THRESHOLD,
                     'risk_threshold': RISK_THRESHOLD,
                     'three_task_gain_vs_both_controls': bool(mean_ok),
                     'xc_claim_supported': bool(xc_ok),
                     'improved_tasks': improved,
                     'risk_flags': risk,
                     'note': ('POSITIVE/NA means mean deltas over six development units; two '
                              'folds per task are not a significance test')},
        'VERDICT': verdict,
        'EPS_ONLY_CANDIDATE': bool(improved == ['eps']),
        'best_epoch_at_budget_end': sorted(
            f'{key[0]}/{key[1]}/fold{key[2]}' for key in units
            if int(units[key]['best_epoch']) >= int(units[key]['epochs_configured'])),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--development', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    payload = summarize(args.development)
    Path(args.output).write_text(json.dumps(payload, indent=2), encoding='utf-8')
    print(json.dumps({key: payload[key] for key in
                      ('arm_mean_r2', 'criteria', 'VERDICT', 'EPS_ONLY_CANDIDATE',
                       'best_epoch_at_budget_end')}, indent=2))


if __name__ == '__main__':
    main()
