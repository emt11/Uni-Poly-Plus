#!/usr/bin/env python3
"""Strict paired B/SL/X2L reporter."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / 'configs/mts/glt_v2_x2l_line_conditioning_screen_v1.json'


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
    shard = root / 'shards/42' / task / f'fold_{fold}.csv'
    prediction = root / 'predictions/42' / task / f'fold_{fold}.npz'
    frame = pd.read_csv(shard)
    if len(frame) != 1:
        raise RuntimeError(f'expected one row: {shard}')
    row = frame.iloc[0]
    with np.load(prediction, allow_pickle=False) as payload:
        pred = {
            'y_true': np.asarray(payload['y_true']),
            'sample_indices': np.asarray(payload['sample_indices']),
            'metadata': json.loads(str(np.asarray(payload['metadata']).item())),
        }
    score = float(row['avg_test_r2'])
    if not np.isfinite(score):
        raise RuntimeError(f'non-finite score: {shard}')
    return row, pred, score


def _distribution(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        'fold_mean': float(values.mean()),
        'fold_median': float(np.median(values)),
        'fold_p25': float(np.quantile(values, .25)),
        'fold_p75': float(np.quantile(values, .75)),
        'fold_min': float(values.min()), 'fold_max': float(values.max()),
        'positive_folds': int((values > 0).sum()),
    }


def main():
    config = json.loads(CONFIG_PATH.read_text(encoding='utf-8'))
    output = ROOT / config['output_root']
    roots = {'B': ROOT / config['baseline_root'], 'SL': output / 'sl', 'X2L': output / 'x2l'}
    modes = {'B': 'o8_glt_atom', 'SL': 'o8_glt_atom_line_self', 'X2L': 'o8_glt_atom_line_x2l'}
    rows, diagnostics = [], []
    for task in config['tasks']:
        for fold in config['folds']:
            units = {name: _unit(root, task, fold) for name, root in roots.items()}
            reference = units['B'][1]
            fold_seed = int(reference['metadata']['fold_seed'])
            for name, (row, pred, _) in units.items():
                expected_checkpoint = str((ROOT / config['checkpoint']).resolve())
                if str(row['checkpoint_path']) != expected_checkpoint:
                    raise RuntimeError(f'{name} checkpoint mismatch')
                if str(row['mts_glt_mode']) != modes[name]:
                    raise RuntimeError(f'{name} mode mismatch')
                if str(row['mts_glt_fusion_strategy']) != 'legacy_zero':
                    raise RuntimeError(f'{name} is not Warm0')
                if str(row['amp_dtype']) != 'fp32' or int(row['train_batch_size']) != 32:
                    raise RuntimeError(f'{name} training contract mismatch')
                if not np.array_equal(pred['sample_indices'], reference['sample_indices']):
                    raise RuntimeError(f'{name} sample indices mismatch')
                if not np.array_equal(pred['y_true'], reference['y_true']):
                    raise RuntimeError(f'{name} targets mismatch')
                if int(pred['metadata']['fold_seed']) != fold_seed:
                    raise RuntimeError(f'{name} fold seed mismatch')
            b, sl, x2l = (units[name][2] for name in ('B', 'SL', 'X2L'))
            rows.append({
                'task': task, 'fold': int(fold), 'fold_seed': fold_seed,
                'B': b, 'SL': sl, 'X2L': x2l,
                'CapacityLine': sl - b, 'CrossLine': x2l - sl,
                'TotalLine': x2l - b,
            })
            for arm, label in (('sl', 'SL'), ('x2l', 'X2L')):
                diag_path = output / 'line_units' / arm / task / f'fold_{fold}.json'
                checkpoint_path = output / 'checkpoints' / arm / task / f'fold_{fold}.pth'
                diag = json.loads(diag_path.read_text(encoding='utf-8'))
                payload = torch.load(checkpoint_path, map_location='cpu')
                if payload.get('mts_glt_mode') != modes[label]:
                    raise RuntimeError(f'checkpoint mode mismatch: {checkpoint_path}')
                diagnostics.append({'arm': arm, **diag})
    folds = pd.DataFrame(rows)
    task_rows = []
    columns = ('B', 'SL', 'X2L', 'CapacityLine', 'CrossLine', 'TotalLine')
    for task in config['tasks']:
        subset = folds[folds.task == task]
        task_rows.append({'task': task, **{key: float(subset[key].mean()) for key in columns}})
    tasks = pd.DataFrame(task_rows)
    contrasts = {}
    for key in ('CapacityLine', 'CrossLine', 'TotalLine'):
        contrasts[key] = {
            'macro3': float(tasks[key].mean()),
            'median_task': float(tasks[key].median()),
            'positive_tasks': int((tasks[key] > 0).sum()),
            **_distribution(folds[key]),
        }
    cross, total = contrasts['CrossLine'], contrasts['TotalLine']
    cross_positive = (
        cross['macro3'] > 0 and cross['median_task'] > 0
        and cross['positive_tasks'] >= 2 and cross['positive_folds'] >= 5
    )
    total_positive = (
        total['macro3'] > 0 and total['median_task'] > 0
        and total['positive_tasks'] >= 2 and total['positive_folds'] >= 5
    )
    sanity = json.loads((output / 'gradient_sanity.json').read_text(encoding='utf-8'))
    diag = pd.DataFrame(diagnostics)
    sl_diag, x2l_diag = diag[diag.arm == 'sl'], diag[diag.arm == 'x2l']
    summary = {
        'schema': config['schema'], 'baseline': config['baseline'],
        'checkpoint': str((ROOT / config['checkpoint']).resolve()),
        'checkpoint_step': 5000, 'tasks': config['tasks'], 'folds': config['folds'],
        'seed': 42, 'protocol': config['evaluation_protocol'],
        'reused_b_runs': 9, 'new_sl_runs': 9, 'new_x2l_runs': 9, 'failed_runs': 0,
        'new_parameter_count_sl': int(sanity['sl']['parameter_count']),
        'new_parameter_count_x2l': int(sanity['x2l']['parameter_count']),
        'step0_equivalence_pass': bool(sanity['step0_equivalence_pass']),
        'step0_wgamma_gradient_pass': bool(sanity['step0_wgamma_gradient_pass']),
        'contrasts': contrasts,
        'cross_line_decision': 'POSITIVE_SCREEN' if cross_positive else 'NOT_ESTABLISHED',
        'x2l_total_decision': 'GO_candidate' if total_positive else 'STOP_candidate',
        'mean_sl_abs_modulation': float(sl_diag.mean_abs_modulation.mean()),
        'mean_x2l_abs_modulation': float(x2l_diag.mean_abs_modulation.mean()),
        'mean_sl_relative_token_change': float(sl_diag.mean_relative_token_change.mean()),
        'mean_x2l_relative_token_change': float(x2l_diag.mean_relative_token_change.mean()),
        'independent_blind_test': False,
    }
    manifest = {
        'schema': 'mts-glt-v2-x2l-line-conditioning-manifest-v1',
        'config': str(CONFIG_PATH.relative_to(ROOT)),
        'baseline_source': str(roots['B'].relative_to(ROOT)),
        'sl_source': str(roots['SL'].relative_to(ROOT)),
        'x2l_source': str(roots['X2L'].relative_to(ROOT)),
        'scientific_variable': 'detached self line versus symmetric detached O8 endpoints',
        'paired_units_verified': 9,
    }
    folds.to_csv(output / 'per_fold_results.csv', index=False)
    tasks.to_csv(output / 'task_results.csv', index=False)
    diag.to_csv(output / 'line_conditioning_diagnostics.csv', index=False)
    _atomic(output / 'x2l_line_screen_summary.json', json.dumps(summary, indent=2, sort_keys=True) + '\n')
    _atomic(output / 'run_manifest.json', json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    print(json.dumps(summary, sort_keys=True))


if __name__ == '__main__':
    main()
