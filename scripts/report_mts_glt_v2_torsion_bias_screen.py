#!/usr/bin/env python3
"""Strict paired B/TC/TG torsion-bias reporter."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / 'configs/mts/glt_v2_torsion_bias_screen_v1.json'


def _atomic(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.tmp.{os.getpid()}')
    try:
        tmp.write_text(text, encoding='utf-8'); os.replace(tmp, path)
    finally:
        if tmp.exists(): tmp.unlink()


def _unit(root, task, fold):
    shard = root / 'shards/42' / task / f'fold_{fold}.csv'
    prediction = root / 'predictions/42' / task / f'fold_{fold}.npz'
    frame = pd.read_csv(shard)
    if len(frame) != 1: raise RuntimeError(f'expected one row: {shard}')
    row = frame.iloc[0]
    with np.load(prediction, allow_pickle=False) as payload:
        pred = {
            'y_true': np.asarray(payload['y_true']),
            'sample_indices': np.asarray(payload['sample_indices']),
            'metadata': json.loads(str(np.asarray(payload['metadata']).item())),
        }
    score = float(row['avg_test_r2'])
    if not np.isfinite(score): raise RuntimeError(f'non-finite score: {shard}')
    return row, pred, score


def _distribution(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        'fold_mean': float(values.mean()), 'fold_median': float(np.median(values)),
        'fold_p25': float(np.quantile(values, .25)),
        'fold_p75': float(np.quantile(values, .75)),
        'fold_min': float(values.min()), 'fold_max': float(values.max()),
        'positive_folds': int((values > 0).sum()),
    }


def main():
    config = json.loads(CONFIG_PATH.read_text(encoding='utf-8'))
    output = ROOT / config['output_root']
    roots = {'B': ROOT / config['baseline_root'], 'TC': output / 'tc', 'TG': output / 'tg'}
    modes = {'B': 'o8_glt_atom', 'TC': 'o8_glt_atom_torsion_count', 'TG': 'o8_glt_atom_torsion'}
    rows, diagnostics = [], []
    for task in config['tasks']:
        for fold in config['folds']:
            units = {name: _unit(root, task, fold) for name, root in roots.items()}
            reference = units['B'][1]
            fold_seed = int(reference['metadata']['fold_seed'])
            for name, (row, pred, _) in units.items():
                if str(row['checkpoint_path']) != str((ROOT / config['checkpoint']).resolve()):
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
            b, tc, tg = (units[name][2] for name in ('B', 'TC', 'TG'))
            rows.append({
                'task': task, 'fold': int(fold), 'fold_seed': fold_seed,
                'B': b, 'TC': tc, 'TG': tg,
                'CountEffect': tc - b, 'GeometryEffect': tg - tc,
                'TotalTorsion': tg - b,
            })
            for arm, label in (('tc', 'TC'), ('tg', 'TG')):
                diag_path = output / 'torsion_units' / arm / task / f'fold_{fold}.json'
                checkpoint_path = output / 'checkpoints' / arm / task / f'fold_{fold}.pth'
                diag = json.loads(diag_path.read_text(encoding='utf-8'))
                payload = torch.load(checkpoint_path, map_location='cpu')
                if payload.get('mts_glt_mode') != modes[label]:
                    raise RuntimeError(f'checkpoint mode mismatch: {checkpoint_path}')
                diagnostics.append({'arm': arm, **diag})
    folds = pd.DataFrame(rows)
    columns = ('B', 'TC', 'TG', 'CountEffect', 'GeometryEffect', 'TotalTorsion')
    tasks = pd.DataFrame([
        {'task': task, **{key: float(folds[folds.task == task][key].mean()) for key in columns}}
        for task in config['tasks']
    ])
    contrasts = {}
    for key in ('CountEffect', 'GeometryEffect', 'TotalTorsion'):
        contrasts[key] = {
            'macro3': float(tasks[key].mean()),
            'median_task': float(tasks[key].median()),
            'positive_tasks': int((tasks[key] > 0).sum()),
            **_distribution(folds[key]),
        }
    positive = lambda item: (
        item['macro3'] > 0 and item['median_task'] > 0
        and item['positive_tasks'] >= 2 and item['positive_folds'] >= 5
    )
    geometry, total = contrasts['GeometryEffect'], contrasts['TotalTorsion']
    sanity = json.loads((output / 'geometry_sanity.json').read_text(encoding='utf-8'))
    coverage = json.loads((output / 'torsion_coverage.json').read_text(encoding='utf-8'))
    diag = pd.DataFrame(diagnostics)
    tg_diag = diag[diag.arm == 'tg']
    summary = {
        'schema': config['schema'], 'baseline': config['baseline'],
        'checkpoint': str((ROOT / config['checkpoint']).resolve()),
        'checkpoint_step': 5000, 'tasks': config['tasks'], 'folds': config['folds'],
        'seed': 42, 'protocol': config['evaluation_protocol'],
        'reused_b_runs': 9, 'new_tc_runs': 9, 'new_tg_runs': 9, 'failed_runs': 0,
        'torsion_relation_coverage': coverage['torsion_relation_coverage'],
        'mean_torsion_observations': coverage['mean_torsion_observations'],
        'median_torsion_observations': coverage['median_torsion_observations'],
        'internal_torsion_coverage': coverage['internal_torsion_coverage'],
        'cross_ru_torsion_coverage': coverage['cross_ru_torsion_coverage'],
        'new_parameter_count_tc': sanity['new_parameter_count_tc'],
        'new_parameter_count_tg': sanity['new_parameter_count_tg'],
        'step0_parity_pass': sanity['step0_parity_pass'],
        'geometry_invariance_pass': sanity['geometry_invariance_pass'],
        'canonicalization_pass': sanity['canonicalization_pass'],
        'macro3_count_effect': contrasts['CountEffect']['macro3'],
        'median_count_effect': contrasts['CountEffect']['median_task'],
        'positive_count_tasks': contrasts['CountEffect']['positive_tasks'],
        'positive_count_folds': contrasts['CountEffect']['positive_folds'],
        'macro3_geometry_effect': geometry['macro3'],
        'median_geometry_effect': geometry['median_task'],
        'positive_geometry_tasks': geometry['positive_tasks'],
        'positive_geometry_folds': geometry['positive_folds'],
        'macro3_total_torsion': total['macro3'],
        'median_total_torsion': total['median_task'],
        'positive_total_tasks': total['positive_tasks'],
        'positive_total_folds': total['positive_folds'],
        'torsion_geometry_decision': 'POSITIVE_SCREEN' if positive(geometry) else 'NOT_ESTABLISHED',
        'torsion_total_decision': 'GO_candidate' if positive(total) else 'STOP_candidate',
        'mean_torsion_bias': float(tg_diag.mean_abs_torsion_bias.mean()),
        'mean_angle_bias': float(tg_diag.mean_abs_angle_bias.mean()),
        'torsion_angle_bias_ratio': float(tg_diag.torsion_angle_bias_ratio.mean()),
        'contrasts': contrasts, 'independent_blind_test': False,
    }
    manifest = {
        'schema': 'mts-glt-v2-torsion-bias-manifest-v1',
        'config': str(CONFIG_PATH.relative_to(ROOT)),
        'count_multiplicity_contract': 'count only; no synthetic multiplicity statistic',
        'reflection_representation': 'cos(phi)',
        'message_graph': 'unchanged strict real-bond line 1-hop',
        'paired_units_verified': 9,
    }
    output.mkdir(parents=True, exist_ok=True)
    folds.to_csv(output / 'per_fold_results.csv', index=False)
    tasks.to_csv(output / 'task_results.csv', index=False)
    diag.to_csv(output / 'torsion_diagnostics.csv', index=False)
    _atomic(output / 'torsion_screen_summary.json', json.dumps(summary, indent=2, sort_keys=True) + '\n')
    _atomic(output / 'run_manifest.json', json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    print(json.dumps(summary, sort_keys=True))


if __name__ == '__main__':
    main()
