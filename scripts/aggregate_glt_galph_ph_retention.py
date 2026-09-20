#!/usr/bin/env python3
"""Aggregate the PH retention development units and apply the §7 science gate.

Reads ``results/glt_galph_ph_retention_20260920/p3/<group>/<task>/fold<k>/metrics.json``
for the three groups, reports per-unit and per-task numbers, the three deltas and
the pre-registered engineering criteria.  A unit is refused unless its directory
group/task/fold, protocol, new C1 deployment identity (the committed record both
the runner and this script read), finite metric, coverage, readout and epoch
budget all agree.  Any unit that touched the outer test is refused, and a
non-empty ``risk_flags`` verdict is always ``NEEDS_REVIEW`` (待审查) rather than
any promotion.
"""
import argparse
import json
import math
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.modules.glt_galph_checkpoint_identity import (DEFAULT_IDENTITY,
                                                       frozen_checkpoint_sha256,
                                                       load_identity)

GROUPS = ('F_OFF', 'F_CONST', 'F_REAL')
CONTROLS = ('F_OFF', 'F_CONST')
TASKS = ('xc', 'eps', 'eat')
FOLDS = (0, 1)
GAIN_THRESHOLD = 0.005
XC_THRESHOLD = 0.01
RISK_THRESHOLD = 0.01
DEVELOPMENT_EPOCH_CAP = 30


def frozen_c1_sha256(identity_path=DEFAULT_IDENTITY):
    return frozen_checkpoint_sha256(identity_path)


def load_units(root, checkpoint_sha256, identity):
    units = {}
    for group in GROUPS:
        for task in TASKS:
            for fold in FOLDS:
                path = Path(root) / group / task / f'fold{fold}' / 'metrics.json'
                if not path.is_file():
                    raise FileNotFoundError(f'missing development unit: {path}')
                row = json.loads(path.read_text(encoding='utf-8'))
                if row.get('outer_test') != 'NOT_RUN' or row.get('validation_only') is not True:
                    raise ValueError(f'unit touched the outer test: {path}')
                if row.get('protocol') != 'ph_retention_development':
                    raise ValueError(f'unit was not a development run: {path}')
                identity_fields = (row.get('group'), row.get('task'), int(row.get('fold', -1)))
                if identity_fields != (group, task, fold):
                    raise ValueError(f'unit identity {identity_fields} does not match {path}')
                if row.get('checkpoint_sha256') != checkpoint_sha256:
                    raise ValueError(f'unit did not run on the recorded deployment: {path}')
                recorded = row.get('checkpoint_identity')
                if not isinstance(recorded, dict) or \
                        recorded.get('record_path') != identity['record_path']:
                    raise ValueError(f'unit does not name this round\'s identity record: {path}')
                for field, expected in (('ph_encoder_version', identity['ph_encoder_version']),
                                        ('pretrain_summary_mode', identity['summary_mode']),
                                        ('pretrain_ph_mode', identity['ph_mode']),
                                        ('readout', 'DUAL'), ('adaptation', 'full')):
                    if row.get(field) != expected:
                        raise ValueError(f'unit {field}={row.get(field)!r} is not {expected!r}: {path}')
                if row.get('updates_incomplete'):
                    raise ValueError(f'unit is an incomplete bounded run: {path}')
                coverage = row.get('coverage')
                if not isinstance(coverage, dict) or int(coverage.get('missing', 1)) != 0 \
                        or int(coverage.get('samples', 0)) <= 0 \
                        or int(coverage.get('valid', 0)) + int(coverage.get('invalid', 0)) \
                        != int(coverage.get('samples', -1)):
                    raise ValueError(f'unit coverage is not a complete fold: {path}')
                r2 = row.get('best_validation_r2')
                if r2 is None or not math.isfinite(float(r2)):
                    raise ValueError(f'unit has a non-finite best_validation_r2: {path}')
                epoch, configured = int(row.get('best_epoch', -1)), \
                    int(row.get('epochs_configured', -1))
                if configured <= 0 or configured > DEVELOPMENT_EPOCH_CAP:
                    raise ValueError(f'unit epoch budget is outside the contract: {path}')
                if not 0 <= epoch <= configured:
                    raise ValueError(f'unit best_epoch {epoch} is outside its budget: {path}')
                ran = int(row.get('epochs_run', -1))
                if not epoch <= ran <= configured:
                    raise ValueError(f'unit epochs_run {ran} is inconsistent with its budget: {path}')
                diagnostics = row.get('diagnostics')
                if not isinstance(diagnostics, dict) or diagnostics.get('batches', 0) <= 0:
                    raise ValueError(f'unit has no PH diagnostics: {path}')
                if not math.isfinite(float(diagnostics.get('residual_relative_norm', float('nan')))):
                    raise ValueError(f'unit has a non-finite PH residual: {path}')
                units[(group, task, fold)] = row
    return units


def summarize(root, identity_path=DEFAULT_IDENTITY):
    identity = load_identity(identity_path)
    units = load_units(root, identity['sha256'], identity)
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
    verdict = ('NEEDS_REVIEW' if risk else
               'STOP: F_REAL does not reach the three-task gain threshold against both '
               'controls' if not mean_ok else
               'XC_CLAIM_SUPPORTED' if xc_ok else
               'GAIN_WITHOUT_XC_CLAIM' if improved else 'GAIN_NOT_TASK_CONSISTENT')
    return {
        'checkpoint_identity': {key: identity[key] for key in
                                ('record', 'record_path', 'checkpoint', 'sha256', 'step',
                                 'summary_mode', 'ph_mode', 'ph_encoder_version')},
        'units': {f'{key[0]}/{key[1]}/fold{key[2]}': {
            'best_validation_r2': r2[key], 'best_epoch': int(units[key]['best_epoch']),
            'epochs_run': int(units[key].get('epochs_run', -1)),
            'epochs_configured': int(units[key].get('epochs_configured', -1)),
            'wall_seconds': float(units[key]['wall_seconds']),
            'coverage': units[key].get('coverage'),
            'diagnostics': units[key].get('diagnostics'),
            'validation_r2_history': units[key].get('validation_r2_history'),
            'checkpoint_sha256': units[key].get('checkpoint_sha256'),
            'ph_encoder_version': units[key].get('ph_encoder_version'),
            'trainable_parameter_count': units[key].get('trainable_parameter_count'),
            'frozen_training_only_heads': units[key].get('frozen_training_only_heads'),
        } for key in sorted(units)},
        'arm_mean_r2': overall, 'arm_per_task_r2': per_task,
        'arm_per_fold_r2': {f'{group}/{task}/fold{fold}': r2[(group, task, fold)]
                            for group in GROUPS for task in TASKS for fold in FOLDS},
        'arm_gate_tanh': {group: statistics.fmean(
            float(units[(group, task, fold)]['diagnostics']['retention_gate_tanh'])
            for task in TASKS for fold in FOLDS) for group in GROUPS},
        'arm_residual_relative_norm': {group: statistics.fmean(
            float(units[(group, task, fold)]['diagnostics']['residual_relative_norm'])
            for task in TASKS for fold in FOLDS) for group in GROUPS},
        'arm_ph_valid_fraction': {group: statistics.fmean(
            float(units[(group, task, fold)]['diagnostics']['ph_valid_fraction'])
            for task in TASKS for fold in FOLDS) for group in GROUPS},
        'arm_epochs_used': {group: sum(
            int(units[(group, task, fold)].get('epochs_run', 0))
            for task in TASKS for fold in FOLDS) for group in GROUPS},
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
        'verdict_note': ('risk_flags is not empty: 交回审查，不自动晋级' if risk
                         else 'no risk flag raised'),
        'risk_flags': risk,
        'EPS_ONLY_CANDIDATE': bool(improved == ['eps']),
        'best_epoch_at_budget_end': sorted(
            f'{key[0]}/{key[1]}/fold{key[2]}' for key in units
            if int(units[key]['best_epoch']) >= int(units[key]['epochs_configured'])),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--development', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--identity', default=DEFAULT_IDENTITY)
    args = parser.parse_args()
    payload = summarize(args.development, args.identity)
    Path(args.output).write_text(json.dumps(payload, indent=2), encoding='utf-8')
    print(json.dumps({key: payload[key] for key in
                      ('arm_mean_r2', 'criteria', 'VERDICT', 'verdict_note',
                       'risk_flags', 'EPS_ONLY_CANDIDATE',
                       'best_epoch_at_budget_end')}, indent=2))


if __name__ == '__main__':
    main()
