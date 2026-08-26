#!/usr/bin/env python3
"""Build the strict paired B/S3/X23 screening report."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/mts/glt_v2_x23_conditioning_screen_v1.json"


def _atomic(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.tmp.{os.getpid()}')
    try:
        tmp.write_text(text, encoding='utf-8')
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def _unit(root, task, fold):
    shard = root / 'shards' / '42' / task / f'fold_{fold}.csv'
    prediction = root / 'predictions' / '42' / task / f'fold_{fold}.npz'
    frame = pd.read_csv(shard)
    if len(frame) != 1:
        raise RuntimeError(f'expected one shard row: {shard}')
    row = frame.iloc[0]
    with np.load(prediction, allow_pickle=False) as payload:
        pred = {
            'y_true': np.asarray(payload['y_true']),
            'sample_indices': np.asarray(payload['sample_indices']),
            'metadata': json.loads(str(np.asarray(payload['metadata']).item())),
        }
    value = float(row['avg_test_r2'])
    if not np.isfinite(value):
        raise RuntimeError(f'non-finite R2: {shard}')
    return row, pred, value


def _stats(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        'mean': float(values.mean()), 'median': float(np.median(values)),
        'p25': float(np.quantile(values, .25)),
        'p75': float(np.quantile(values, .75)),
        'min': float(values.min()), 'max': float(values.max()),
        'positive_folds': int((values > 0).sum()),
    }


def main():
    config = json.loads(CONFIG_PATH.read_text(encoding='utf-8'))
    output = ROOT / config['output_root']
    roots = {
        'B': ROOT / config['baseline_root'],
        'S3': output / 's3', 'X23': output / 'x23',
    }
    modes = {
        'B': 'o8_glt_atom', 'S3': 'o8_glt_atom_self3d',
        'X23': 'o8_glt_atom_x23',
    }
    rows = []
    diagnostics = []
    for task in config['tasks']:
        for fold in config['folds']:
            units = {name: _unit(root, task, fold) for name, root in roots.items()}
            reference_pred = units['B'][1]
            fold_seed = int(reference_pred['metadata']['fold_seed'])
            for name, (row, pred, _) in units.items():
                if str(row['checkpoint_path']) != str((ROOT / config['checkpoint']).resolve()):
                    raise RuntimeError(f'{name} checkpoint mismatch: {task}/fold{fold}')
                if str(row['mts_glt_mode']) != modes[name]:
                    raise RuntimeError(f'{name} mode mismatch: {task}/fold{fold}')
                if str(row['mts_glt_fusion_strategy']) != 'legacy_zero':
                    raise RuntimeError(f'{name} is not Warm0: {task}/fold{fold}')
                if str(row['amp_dtype']) != 'fp32' or int(row['train_batch_size']) != 32:
                    raise RuntimeError(f'{name} downstream contract mismatch')
                if not np.array_equal(pred['sample_indices'], reference_pred['sample_indices']):
                    raise RuntimeError(f'{name} sample indices mismatch')
                if not np.array_equal(pred['y_true'], reference_pred['y_true']):
                    raise RuntimeError(f'{name} targets mismatch')
                if int(pred['metadata']['fold_seed']) != fold_seed:
                    raise RuntimeError(f'{name} fold seed mismatch')
            b, s3, x23 = (units[name][2] for name in ('B', 'S3', 'X23'))
            rows.append({
                'task': task, 'fold': int(fold), 'fold_seed': fold_seed,
                'B': b, 'S3': s3, 'X23': x23,
                'DeltaCapacity': s3 - b, 'DeltaCross': x23 - s3,
                'DeltaTotal': x23 - b,
            })
            for arm in ('s3', 'x23'):
                diag_path = output / 'interaction_units' / arm / task / f'fold_{fold}.json'
                checkpoint_path = output / 'checkpoints' / arm / task / f'fold_{fold}.pth'
                diag = json.loads(diag_path.read_text(encoding='utf-8'))
                payload = torch.load(checkpoint_path, map_location='cpu')
                expected_mode = modes['S3' if arm == 's3' else 'X23']
                if payload.get('mts_glt_mode') != expected_mode:
                    raise RuntimeError(f'best checkpoint mode mismatch: {checkpoint_path}')
                diagnostics.append({'arm': arm, **diag})
    folds = pd.DataFrame(rows)
    task_rows = []
    for task in config['tasks']:
        subset = folds[folds.task == task]
        task_rows.append({
            'task': task,
            **{column: float(subset[column].mean()) for column in (
                'B', 'S3', 'X23', 'DeltaCapacity', 'DeltaCross', 'DeltaTotal'
            )},
        })
    tasks = pd.DataFrame(task_rows)
    contrast_summary = {}
    for column in ('DeltaCapacity', 'DeltaCross', 'DeltaTotal'):
        contrast_summary[column] = {
            'macro3': float(tasks[column].mean()),
            'median_task': float(tasks[column].median()),
            'positive_tasks': int((tasks[column] > 0).sum()),
            **_stats(folds[column]),
        }
    cross = contrast_summary['DeltaCross']
    total = contrast_summary['DeltaTotal']
    cross_decision = (
        'POSITIVE' if cross['macro3'] > 0 and cross['median_task'] > 0
        and cross['positive_tasks'] >= 2 else 'NOT_ESTABLISHED'
    )
    total_decision = (
        'GO_candidate' if total['macro3'] > 0 and total['median_task'] > 0
        and total['positive_tasks'] >= 2 else 'STOP_candidate'
    )
    diag_frame = pd.DataFrame(diagnostics)
    x23_diag = diag_frame[diag_frame.arm == 'x23']
    summary = {
        'schema': config['schema'], 'baseline': config['baseline'],
        'checkpoint': str((ROOT / config['checkpoint']).resolve()),
        'checkpoint_step': 5000, 'tasks': config['tasks'],
        'folds': config['folds'], 'seed': 42,
        'protocol': config['evaluation_protocol'],
        'reused_b_runs': 9, 'new_s3_runs': 9, 'new_x23_runs': 9,
        'failed_runs': 0, 'contrasts': contrast_summary,
        'cross_conditioning_decision': cross_decision,
        'x23_total_decision': total_decision,
        'mean_x23_gate_magnitude': float(x23_diag.mean_abs_tanh_g23.mean()),
        'mean_x23_update_norm_ratio': float(x23_diag.mean_update_norm_ratio.mean()),
        'independent_blind_test': False,
    }
    manifest = {
        'schema': 'mts-glt-v2-x23-conditioning-manifest-v1',
        'config': str(CONFIG_PATH.relative_to(ROOT)),
        'scientific_variable': 'condition source: detached h3 versus detached h2',
        'baseline_source': str(roots['B'].relative_to(ROOT)),
        's3_source': str(roots['S3'].relative_to(ROOT)),
        'x23_source': str(roots['X23'].relative_to(ROOT)),
        'paired_units_verified': 9,
    }
    output.mkdir(parents=True, exist_ok=True)
    folds.to_csv(output / 'per_fold_results.csv', index=False)
    tasks.to_csv(output / 'task_results.csv', index=False)
    diag_frame.to_csv(output / 'interaction_diagnostics.csv', index=False)
    _atomic(output / 'x23_screen_summary.json', json.dumps(summary, indent=2, sort_keys=True) + '\n')
    _atomic(output / 'run_manifest.json', json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    print(json.dumps(summary, sort_keys=True))


if __name__ == '__main__':
    main()
