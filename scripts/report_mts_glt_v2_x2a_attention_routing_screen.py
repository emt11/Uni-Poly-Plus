#!/usr/bin/env python3
"""Strict paired B/SA/X2A attention-routing reporter."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / 'configs/mts/glt_v2_x2a_attention_routing_screen_v1.json'


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
    roots = {'B': ROOT / config['baseline_root'], 'SA': output / 'sa', 'X2A': output / 'x2a'}
    modes = {'B': 'o8_glt_atom', 'SA': 'o8_glt_atom_attn_self', 'X2A': 'o8_glt_atom_attn_x2a'}
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
            b, sa, x2a = (units[name][2] for name in ('B', 'SA', 'X2A'))
            rows.append({
                'task': task, 'fold': int(fold), 'fold_seed': fold_seed,
                'B': b, 'SA': sa, 'X2A': x2a,
                'CapacityAttn': sa - b, 'CrossAttn': x2a - sa,
                'TotalAttn': x2a - b,
            })
            for arm, label in (('sa', 'SA'), ('x2a', 'X2A')):
                diag_path = output / 'attention_units' / arm / task / f'fold_{fold}.json'
                checkpoint_path = output / 'checkpoints' / arm / task / f'fold_{fold}.pth'
                diag = json.loads(diag_path.read_text(encoding='utf-8'))
                payload = torch.load(checkpoint_path, map_location='cpu')
                if payload.get('mts_glt_mode') != modes[label]:
                    raise RuntimeError(f'checkpoint mode mismatch: {checkpoint_path}')
                if not bool(diag.get('value_path_unchanged')):
                    raise RuntimeError(f'V-path diagnostic failed: {diag_path}')
                diagnostics.append({'arm': arm, **diag})
    folds = pd.DataFrame(rows)
    columns = ('B', 'SA', 'X2A', 'CapacityAttn', 'CrossAttn', 'TotalAttn')
    tasks = pd.DataFrame([
        {'task': task, **{
            key: float(folds[folds.task == task][key].mean()) for key in columns
        }} for task in config['tasks']
    ])
    contrasts = {}
    for key in ('CapacityAttn', 'CrossAttn', 'TotalAttn'):
        contrasts[key] = {
            'macro3': float(tasks[key].mean()),
            'median_task': float(tasks[key].median()),
            'positive_tasks': int((tasks[key] > 0).sum()),
            **_distribution(folds[key]),
        }
    cross, total = contrasts['CrossAttn'], contrasts['TotalAttn']
    positive = lambda item: (
        item['macro3'] > 0 and item['median_task'] > 0
        and item['positive_tasks'] >= 2 and item['positive_folds'] >= 5
    )
    sanity = json.loads((output / 'gradient_sanity.json').read_text(encoding='utf-8'))
    diag = pd.DataFrame(diagnostics)
    sa_diag, x2a_diag = diag[diag.arm == 'sa'], diag[diag.arm == 'x2a']
    capacity = contrasts['CapacityAttn']
    summary = {
        'schema': config['schema'], 'baseline': config['baseline'],
        'checkpoint': str((ROOT / config['checkpoint']).resolve()),
        'checkpoint_step': 5000, 'tasks': config['tasks'], 'folds': config['folds'],
        'seed': 42, 'protocol': config['evaluation_protocol'],
        'reused_b_runs': 9, 'new_sa_runs': 9, 'new_x2a_runs': 9, 'failed_runs': 0,
        'new_parameter_count_sa': int(sanity['sa']['parameter_count']),
        'new_parameter_count_x2a': int(sanity['x2a']['parameter_count']),
        'step0_equivalence_pass': bool(sanity['step0_equivalence_pass']),
        'first_backward_gradient_pass': bool(sanity['first_backward_gradient_pass']),
        'value_path_unchanged_pass': bool(sanity['value_path_unchanged_pass']),
        'angle_bias_unchanged_pass': bool(sanity['angle_bias_unchanged_pass']),
        'endpoint_swap_pass': bool(sanity['endpoint_swap_pass']),
        'macro3_capacity_attn': capacity['macro3'],
        'median_capacity_attn': capacity['median_task'],
        'positive_capacity_tasks': capacity['positive_tasks'],
        'positive_capacity_folds': capacity['positive_folds'],
        'macro3_cross_attn': cross['macro3'],
        'median_cross_attn': cross['median_task'],
        'positive_cross_tasks': cross['positive_tasks'],
        'positive_cross_folds': cross['positive_folds'],
        'macro3_total_attn': total['macro3'],
        'median_total_attn': total['median_task'],
        'positive_total_tasks': total['positive_tasks'],
        'positive_total_folds': total['positive_folds'],
        'contrasts': contrasts,
        'cross_attn_decision': 'POSITIVE_SCREEN' if positive(cross) else 'NOT_ESTABLISHED',
        'x2a_total_decision': 'GO_candidate' if positive(total) else 'STOP_candidate',
        'mean_sa_abs_gamma': float(sa_diag.mean_abs_gamma.mean()),
        'mean_x2a_abs_gamma': float(x2a_diag.mean_abs_gamma.mean()),
        'mean_sa_attention_change': float(sa_diag.mean_attention_probability_change.mean()),
        'mean_x2a_attention_change': float(x2a_diag.mean_attention_probability_change.mean()),
        'mean_sa_attention_kl': float(sa_diag.mean_attention_kl.mean()),
        'mean_x2a_attention_kl': float(x2a_diag.mean_attention_kl.mean()),
        'independent_blind_test': False,
    }
    manifest = {
        'schema': 'mts-glt-v2-x2a-attention-routing-manifest-v1',
        'config': str(CONFIG_PATH.relative_to(ROOT)),
        'baseline_source': str(roots['B'].relative_to(ROOT)),
        'sa_source': str(roots['SA'].relative_to(ROOT)),
        'x2a_source': str(roots['X2A'].relative_to(ROOT)),
        'scientific_variable': 'detached self line versus symmetric detached O8 endpoints for layer-1 Q/K only',
        'paired_units_verified': 9,
    }
    output.mkdir(parents=True, exist_ok=True)
    folds.to_csv(output / 'per_fold_results.csv', index=False)
    tasks.to_csv(output / 'task_results.csv', index=False)
    diag.to_csv(output / 'attention_routing_diagnostics.csv', index=False)
    _atomic(output / 'x2a_attention_screen_summary.json', json.dumps(summary, indent=2, sort_keys=True) + '\n')
    _atomic(output / 'run_manifest.json', json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    print(json.dumps(summary, sort_keys=True))


if __name__ == '__main__':
    main()
