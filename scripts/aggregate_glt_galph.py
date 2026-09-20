#!/usr/bin/env python3
"""Aggregate the GLT-GALPH P2 telemetry and the P3 development deltas.

``--pretrain`` summarises the four 5k runs (per-arm loss terms, throughput, peak
memory, deployment identity).  ``--development`` reads the 24 development units
(4 arms x {xc,eps,eat} x folds {0,1}) and reports the four core deltas
N1-N0, C1-C0, C0-N0 and C1-N1.  Every development metrics file must state
``outer_test=NOT_RUN``; anything else is refused, because this round must not
touch the outer test set.
"""
import argparse
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ARMS = ('N0', 'N1', 'C0', 'C1')
TASKS = ('xc', 'eps', 'eat')
FOLDS = (0, 1)
SAMPLE_EVERY = 500


def _records(path):
    rows = []
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def summarize_pretrain(root):
    report = {}
    for arm in ARMS:
        folder = Path(root) / arm.lower() / 'pretrain'
        records = _records(folder / 'records_rank0.jsonl')
        if not records:
            report[arm] = {'status': 'MISSING', 'path': str(folder)}
            continue
        steps = [int(row['step']) for row in records]
        sampled = [row for row in records if int(row['step']) % SAMPLE_EVERY == 0
                   or row is records[-1]]
        deploy = folder / 'deploy_05000.pt'
        run = json.loads((folder / 'run.json').read_text(encoding='utf-8'))
        seconds = [float(row['step_seconds']) for row in records]
        report[arm] = {
            'status': 'COMPLETE' if steps[-1] == 5000 else 'PARTIAL',
            'steps': [steps[0], steps[-1]],
            'losses_at_steps': {str(row['step']): [round(float(v), 4) for v in row['losses']]
                                for row in sampled},
            'global_counts_last': [int(v) for v in records[-1]['global_counts']],
            'step_seconds_mean': float(statistics.fmean(seconds)),
            'updates_per_hour': float(3600.0 / statistics.fmean(seconds)),
            'peak_gpu_memory_gib': max(float(row.get('peak_gpu_memory_gib', 0.0))
                                       for row in records),
            'deployment': str(deploy),
            'deployment_exists': deploy.is_file(),
            'common_init_sha256': run['identity']['common_init_artifact_sha256'],
            'common_state_sha256': None,
            'arm': run['identity']['arm'],
            'optimizer': run['identity']['optimizer'],
            'lr': run['identity']['config']['lr'],
            'weight_decay': run['identity']['config']['weight_decay'],
            'warmup_steps': run['identity']['config']['warmup_steps'],
            'schedule_total_steps': run['identity']['config']['schedule_total_steps'],
        }
        if deploy.is_file():
            import hashlib
            import torch
            package = torch.load(deploy, map_location='cpu', weights_only=False)
            report[arm]['deployment_step'] = int(package['step'])
            report[arm]['deployment_summary_mode'] = package['summary_mode']
            report[arm]['deployment_ph_mode'] = package['ph_mode']
            report[arm]['deployment_ph_encoder_version'] = package.get('ph_encoder_version')
            report[arm]['deployment_tensors'] = len(package['state_dict'])
            report[arm]['deployment_bytes'] = deploy.stat().st_size
            report[arm]['deployment_sha256'] = hashlib.sha256(
                deploy.read_bytes()).hexdigest()
    return report


def version_freeze(pretrain, attribution_path):
    """The r3 freeze record: SHAs, PH encoder version and training commits."""
    import hashlib
    attribution = {}
    path = Path(attribution_path)
    if path.is_file():
        attribution = json.loads(path.read_text(encoding='utf-8'))
    return {
        'common_init': {
            'path': 'results/glt_galph_20260920/p1/common_init_galph_v1.pt',
            'sha256': hashlib.sha256(Path(
                'results/glt_galph_20260920/p1/common_init_galph_v1.pt').read_bytes()).hexdigest(),
        },
        'ph_encoder_version': 'scale-interaction-v2',
        'deployments': {arm: {'path': row.get('deployment'),
                              'step': row.get('deployment_step'),
                              'sha256': row.get('deployment_sha256'),
                              'ph_encoder_version': row.get('deployment_ph_encoder_version')}
                        for arm, row in pretrain.items()},
        'training_commits': attribution.get('training_commits', {}),
        'training_commit_evidence': attribution.get('evidence', {}),
    }


def summarize_development(root):
    units, missing = {}, []
    for arm in ARMS:
        for task in TASKS:
            for fold in FOLDS:
                path = Path(root) / arm.lower() / task / f'fold{fold}' / 'metrics.json'
                if not path.is_file():
                    missing.append(str(path))
                    continue
                row = json.loads(path.read_text(encoding='utf-8'))
                if row.get('outer_test') != 'NOT_RUN':
                    raise ValueError(f'development unit touched outer test: {path}')
                units[f'{arm}/{task}/fold{fold}'] = {
                    'best_validation_r2': float(row['best_validation_r2']),
                    'best_epoch': int(row['best_epoch']),
                    'validation_r2_final_epoch': float(row['validation_r2']),
                    'wall_seconds': float(row.get('wall_seconds', float('nan'))),
                    'readout': row['readout'],
                    'pretrain_step': int(row['pretrain_step']),
                    'pretrain_ph_mode': str(row['pretrain_ph_mode']),
                    'trainable_parameter_count': int(row['trainable_parameter_count']),
                }
    if missing:
        raise FileNotFoundError(f'{len(missing)} development units are missing: {missing[:4]}')
    arm_mean = {arm: statistics.fmean(units[f'{arm}/{task}/fold{fold}']['best_validation_r2']
                                      for task in TASKS for fold in FOLDS)
                for arm in ARMS}
    per_task = {arm: {task: statistics.fmean(units[f'{arm}/{task}/fold{fold}']['best_validation_r2']
                                             for fold in FOLDS) for task in TASKS}
                for arm in ARMS}
    per_fold = {arm: {f'{task}/fold{fold}': units[f'{arm}/{task}/fold{fold}']['best_validation_r2']
                      for task in TASKS for fold in FOLDS} for arm in ARMS}
    deltas = {}
    for name, left, right in (('N1_MINUS_N0', 'N1', 'N0'), ('C1_MINUS_C0', 'C1', 'C0'),
                              ('C0_MINUS_N0', 'C0', 'N0'), ('C1_MINUS_N1', 'C1', 'N1')):
        deltas[name] = {
            'mean': arm_mean[left] - arm_mean[right],
            'per_task': {task: per_task[left][task] - per_task[right][task] for task in TASKS},
            'per_fold': {key: per_fold[left][key] - per_fold[right][key]
                         for key in per_fold[left]},
        }
    best = max(ARMS, key=lambda arm: arm_mean[arm])
    return {
        'units': units, 'arm_mean_r2': arm_mean, 'arm_per_task_r2': per_task,
        'arm_per_fold_r2': per_fold, 'deltas': deltas,
        'PH_N_ROUTE': 'POSITIVE' if deltas['N1_MINUS_N0']['mean'] > 0 else 'NOT_ESTABLISHED',
        'PH_C_ROUTE': 'POSITIVE' if deltas['C1_MINUS_C0']['mean'] > 0 else 'NOT_ESTABLISHED',
        'CLS_VALUE': 'POSITIVE' if deltas['C0_MINUS_N0']['mean'] > 0 else 'NOT_ESTABLISHED',
        'CLS_VALUE_UNDER_PH': ('POSITIVE' if deltas['C1_MINUS_N1']['mean'] > 0
                               else 'NOT_ESTABLISHED'),
        'SELECTED_ROUTE': best,
        'selection_rule': 'highest mean best_validation_r2 over the six development units',
        'criterion_note': ('POSITIVE means the mean development-fold delta over six units '
                           'is above zero; two folds per task are not a significance test'),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pretrain')
    parser.add_argument('--development')
    parser.add_argument('--commit-attribution',
                        default='results/glt_galph_20260920/commit_attribution.json')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    payload = {}
    if args.pretrain:
        payload['pretrain'] = summarize_pretrain(args.pretrain)
        payload['versions'] = version_freeze(payload['pretrain'], args.commit_attribution)
    if args.development:
        payload['development'] = summarize_development(args.development)
    Path(args.output).write_text(json.dumps(payload, indent=2), encoding='utf-8')
    print(json.dumps(payload, indent=2))


if __name__ == '__main__':
    main()
