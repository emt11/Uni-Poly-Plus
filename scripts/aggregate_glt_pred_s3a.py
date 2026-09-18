#!/usr/bin/env python3
"""Aggregate and apply the pre-registered GLT-PRED S3a selection gate."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


TASKS = ('xc', 'eps', 'eat')
FOLDS = (0, 1)
POLICIES = ('full', 'head', 'lora', 'ridge')
RIDGE_ALPHAS = (0.1, 1.0, 10.0, 100.0)
ORDER = {'ridge': 0, 'head': 1, 'lora': 2}


def _alpha_token(alpha):
    return str(alpha).replace('.', 'p')


def _unit(output, policy, task, fold, alpha=None):
    name = f'{task}_fold{int(fold)}'
    if alpha is not None:
        name += f'_alpha{_alpha_token(alpha)}'
    return Path(output) / 'runs' / policy / name


def _finite(value):
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _read_unit(output, policy, task, fold, alpha=None):
    unit = _unit(output, policy, task, fold, alpha)
    run = json.loads((unit / 'run.json').read_text(encoding='utf-8'))
    summary = json.loads((unit / 'summary.json').read_text(encoding='utf-8'))
    runtime = json.loads((unit / 'runtime.json').read_text(encoding='utf-8'))
    metrics_path = unit / task / f'fold{int(fold)}' / 'metrics.json'
    metrics = json.loads(metrics_path.read_text(encoding='utf-8'))
    expected_protocol = 'outer5_inner20_development'
    for name, record in (('run', run), ('summary', summary), ('runtime', runtime)):
        if record.get('protocol') != expected_protocol:
            raise ValueError(f'{unit}: {name}.protocol is not development')
        if record.get('outer_test') != 'NOT_RUN':
            raise ValueError(f'{unit}: {name}.outer_test is not NOT_RUN')
    if not run.get('development') or not summary.get('development') or not runtime.get('development'):
        raise ValueError(f'{unit}: development flag missing')
    if run.get('adaptation') != policy or runtime.get('adaptation') != policy:
        raise ValueError(f'{unit}: adaptation identity mismatch')
    if metrics.get('task') != task or int(metrics.get('fold', -1)) != int(fold):
        raise ValueError(f'{unit}: task/fold identity mismatch')
    if metrics.get('outer_test') != 'NOT_RUN' or not metrics.get('validation_only'):
        raise ValueError(f'{unit}: metrics are not validation-only')
    r2 = metrics.get('validation_r2', metrics.get('best_validation_r2'))
    mae = metrics.get('validation_mae')
    rmse = metrics.get('validation_rmse')
    required = {'r2': r2, 'mae': mae, 'rmse': rmse,
                'train_sample_count': metrics.get('train_sample_count'),
                'validation_sample_count': metrics.get('validation_sample_count'),
                'trainable_parameter_count': metrics.get('trainable_parameter_count'),
                'total_parameter_count': metrics.get('total_parameter_count'),
                'wall_seconds': metrics.get('wall_seconds')}
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise ValueError(f'{unit}: missing metrics {missing}')
    if any(not _finite(value) for name, value in required.items()
           if name in {'r2', 'mae', 'rmse', 'wall_seconds'}):
        raise ValueError(f'{unit}: nonfinite validation metrics')
    if any(int(required[name]) < 0 for name in ('train_sample_count', 'validation_sample_count',
                                                'trainable_parameter_count', 'total_parameter_count')):
        raise ValueError(f'{unit}: negative count')
    return {
        'policy': policy, 'task': task, 'fold': int(fold),
        'alpha': float(alpha) if alpha is not None else None,
        'validation_r2': float(r2), 'validation_mae': float(mae),
        'validation_rmse': float(rmse),
        'train_sample_count': int(required['train_sample_count']),
        'validation_sample_count': int(required['validation_sample_count']),
        'trainable_parameter_count': int(required['trainable_parameter_count']),
        'total_parameter_count': int(required['total_parameter_count']),
        'wall_seconds': float(required['wall_seconds']),
        'unit': str(unit), 'metrics_path': str(metrics_path),
    }


def _mean(rows, field):
    return float(sum(row[field] for row in rows) / len(rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--summary-json', required=True)
    parser.add_argument('--report-md', required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    rows = []
    for policy in ('full', 'head', 'lora'):
        for task in TASKS:
            for fold in FOLDS:
                rows.append(_read_unit(output, policy, task, fold))
    ridge_all = []
    for task in TASKS:
        for fold in FOLDS:
            for alpha in RIDGE_ALPHAS:
                ridge_all.append(_read_unit(output, 'ridge', task, fold, alpha))
    selected_ridge = []
    selected_alphas = {}
    for task in TASKS:
        for fold in FOLDS:
            options = [row for row in ridge_all if row['task'] == task and row['fold'] == fold]
            selected = sorted(options, key=lambda row: (-row['validation_r2'], row['alpha']))[0]
            selected_ridge.append(selected)
            selected_alphas[f'{task}/fold{fold}'] = selected['alpha']
    rows.extend(selected_ridge)

    by_policy_task = {
        (policy, task): [row for row in rows if row['policy'] == policy and row['task'] == task]
        for policy in POLICIES for task in TASKS
    }
    for key, values in by_policy_task.items():
        if len(values) != 2:
            raise ValueError(f'incomplete selected rows for {key}: {len(values)}')

    full = {task: by_policy_task[('full', task)] for task in TASKS}
    policy_summary = {}
    for policy in POLICIES:
        policy_summary[policy] = {}
        for task in TASKS:
            values = by_policy_task[(policy, task)]
            policy_summary[policy][task] = {
                'mean_validation_r2': _mean(values, 'validation_r2'),
                'mean_validation_mae': _mean(values, 'validation_mae'),
                'mean_validation_rmse': _mean(values, 'validation_rmse'),
                'mean_trainable_parameter_count': _mean(values, 'trainable_parameter_count'),
                'mean_total_parameter_count': _mean(values, 'total_parameter_count'),
                'folds': values,
            }

    eligibility = {}
    for policy in ('head', 'lora', 'ridge'):
        deltas = {task: [candidate['validation_r2'] - baseline['validation_r2']
                         for candidate, baseline in zip(
                             by_policy_task[(policy, task)], full[task])]
                  for task in TASKS}
        mean_delta = {task: float(sum(values) / len(values)) for task, values in deltas.items()}
        eligible = (mean_delta['xc'] >= 0.01
                    and min(deltas['xc']) >= -0.03
                    and mean_delta['eps'] >= -0.01
                    and mean_delta['eat'] >= -0.01)
        eligibility[policy] = {
            'eligible': bool(eligible), 'delta_by_task_and_fold': deltas,
            'mean_delta_by_task': mean_delta,
            'min_xc_fold_delta': float(min(deltas['xc'])),
        }

    eligible = [policy for policy in ('head', 'lora', 'ridge') if eligibility[policy]['eligible']]
    if not eligible:
        selected = 'full'
        reason = 'No HEAD/LoRA/RIDGE candidate met all pre-registered validation gates; retain FULL.'
    else:
        def selection_key(policy):
            info = eligibility[policy]
            trainable = policy_summary[policy]['xc']['mean_trainable_parameter_count']
            return (info['mean_delta_by_task']['xc'], info['min_xc_fold_delta'],
                    -trainable, -ORDER[policy])
        selected = max(eligible, key=selection_key)
        reason = ('Selected the eligible policy by mean XC delta, then worst-fold XC delta, '
                  'then fewer trainable parameters, then RIDGE/HEAD/LORA fixed order.')

    summary = {
        'scope': 'GLT-PRED S3a downstream adaptation development selection',
        'status': 'PASS',
        'expected_neural_units': 18, 'completed_neural_units': 18,
        'expected_ridge_fits': 24, 'completed_ridge_fits': len(ridge_all),
        'tasks': list(TASKS), 'folds': list(FOLDS),
        'adaptations': list(POLICIES), 'ridge_alphas': list(RIDGE_ALPHAS),
        'policy_summary': policy_summary, 'eligibility': eligibility,
        'ridge_selected_alphas': selected_alphas,
        'selected_adaptation': selected, 'selection_reason': reason,
        'outer_test_accessed': False, 'oof_run': False,
        'formal_pretrain_run': False, 's3b_started': False,
        'active_cache_modified': False, 'plan_md_modified': False,
        'development_only': True,
        'all_unit_rows': rows, 'all_ridge_fits': ridge_all,
    }
    preregistration = json.loads(
        (output / 'pre_registration.json').read_text(encoding='utf-8'))
    for key in ('base_commit', 'implementation_commit', 'reference_checkpoint',
                'reference_checkpoint_sha256'):
        if key not in preregistration:
            raise ValueError(f'pre-registration missing {key}')
    summary.update({
        'BASE_COMMIT': preregistration['base_commit'],
        'S3A_IMPLEMENTATION_COMMIT': preregistration['implementation_commit'],
        'REFERENCE_CHECKPOINT': preregistration['reference_checkpoint'],
        'REFERENCE_CHECKPOINT_SHA256': preregistration['reference_checkpoint_sha256'],
        'NEURAL_UNITS_EXPECTED': 18,
        'NEURAL_UNITS_COMPLETE': 18,
        'RIDGE_FITS_EXPECTED': 24,
        'RIDGE_FITS_COMPLETE': 24,
        'SELECTED_ADAPTATION': selected,
        'SELECTION_REASON': reason,
        'OUTER_TEST_ACCESSED': 'NO',
        'OOF_RUN': 'NO',
        'FORMAL_PRETRAIN_RUN': 'NO',
        'S3B_STARTED': 'NO',
        'ACTIVE_CACHE_MODIFIED': 'NO',
        'PLAN_MD_MODIFIED': 'NO',
        'FINAL_STATUS': 'WAITING_FOR_CODEX_REVIEW',
    })
    for policy in POLICIES:
        prefix = policy.upper()
        for task in TASKS:
            summary[f'{prefix}_{task.upper()}_MEAN_VALID_R2'] = (
                policy_summary[policy][task]['mean_validation_r2'])
        if policy != 'full':
            summary[f'{prefix}_ELIGIBLE'] = eligibility[policy]['eligible']
    summary['RIDGE_SELECTED_ALPHAS'] = selected_alphas
    target = Path(args.summary_json)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    report_lines = [
        '# GLT-PRED S3a development selection', '',
        f"- status: `{summary['status']}`",
        f"- neural units: `{summary['completed_neural_units']}/{summary['expected_neural_units']}`",
        f"- ridge fits: `{summary['completed_ridge_fits']}/{summary['expected_ridge_fits']}`",
        f"- selected adaptation: `{selected}`",
        f"- outer-test accessed: `{summary['outer_test_accessed']}`", '',
        '## Validation means', '',
        '| policy | XC R² | EPS R² | EAT R² | eligible |',
        '|---|---:|---:|---:|---|',
    ]
    for policy in POLICIES:
        report_lines.append('| {0} | {1:.6f} | {2:.6f} | {3:.6f} | {4} |'.format(
            policy,
            policy_summary[policy]['xc']['mean_validation_r2'],
            policy_summary[policy]['eps']['mean_validation_r2'],
            policy_summary[policy]['eat']['mean_validation_r2'],
            'baseline' if policy == 'full' else eligibility[policy]['eligible']))
    report_lines.extend(['', f'**Selection reason:** {reason}', '',
                         'This is development-fold validation-only evidence, not OOF or blind test evidence.'])
    Path(args.report_md).write_text('\n'.join(report_lines) + '\n', encoding='utf-8')
    print(json.dumps({
        'status': summary['status'], 'selected_adaptation': selected,
        'neural_units': f"{summary['completed_neural_units']}/{summary['expected_neural_units']}",
        'ridge_fits': f"{summary['completed_ridge_fits']}/{summary['expected_ridge_fits']}",
    }, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
