#!/usr/bin/env python3
"""Aggregate isolated GLT dual five-fold task shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.training.glt_dual_runtime import TASKS, write_json


def aggregate(root, output, *, tasks=TASKS):
    root, output = Path(root).resolve(), Path(output).resolve()
    rows, summary = [], {}
    for task in tasks:
        fold_metrics, all_targets, all_predictions, all_indices = [], [], [], []
        for fold in range(5):
            metrics_path = root / f'{task}_fold{fold}' / task / f'fold{fold}' / 'metrics.json'
            predictions_path = root / f'{task}_fold{fold}' / task / f'fold{fold}' / 'predictions.csv'
            if not metrics_path.is_file() or not predictions_path.is_file():
                raise FileNotFoundError(f'missing formal shard output: {metrics_path}')
            metrics = json.loads(metrics_path.read_text(encoding='utf-8'))
            if metrics.get('protocol') != 'outer5_inner20' or metrics.get('task') != task or int(metrics.get('fold', -1)) != fold:
                raise ValueError(f'formal shard metadata mismatch: {metrics_path}')
            prediction = pd.read_csv(predictions_path)
            required = {'row_index', 'target', 'prediction'}
            if not required.issubset(prediction.columns) or prediction['row_index'].duplicated().any():
                raise ValueError(f'formal shard prediction table is malformed: {predictions_path}')
            all_indices.extend(prediction['row_index'].astype(int).tolist())
            if not np.isfinite(prediction[['target', 'prediction']].to_numpy()).all():
                raise ValueError(f'formal shard contains nonfinite prediction: {predictions_path}')
            fold_metrics.append(metrics)
            all_targets.append(prediction['target'].to_numpy(float))
            all_predictions.append(prediction['prediction'].to_numpy(float))
            rows.append(metrics)
        if sorted(all_indices) != list(range(len(all_indices))):
            raise ValueError(f'formal shard OOF coverage is not exactly once for task={task}')
        targets = np.concatenate(all_targets)
        predictions = np.concatenate(all_predictions)
        summary[task] = {
            'test_r2': {'mean': float(np.mean([row['test_r2'] for row in fold_metrics])),
                        'std': float(np.std([row['test_r2'] for row in fold_metrics], ddof=1))},
            'test_mae': {'mean': float(np.mean([row['test_mae'] for row in fold_metrics])),
                         'std': float(np.std([row['test_mae'] for row in fold_metrics], ddof=1))},
            'test_rmse': {'mean': float(np.mean([row['test_rmse'] for row in fold_metrics])),
                          'std': float(np.std([row['test_rmse'] for row in fold_metrics], ddof=1))},
            'pooled_oof': {
                'r2': float(r2_score(targets, predictions)),
                'mae': float(mean_absolute_error(targets, predictions)),
                'rmse': float(np.sqrt(mean_squared_error(targets, predictions))),
            },
            'fold_count': 5,
        }
        task_dir = output / task
        task_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({'row_index': all_indices, 'target': targets, 'prediction': predictions}).to_csv(task_dir / 'pooled_shard_predictions.csv', index=False)
    if set(summary) != set(tasks):
        raise ValueError('formal aggregation task coverage mismatch')
    result = {
        'status': 'PASS',
        'protocol': 'outer5_inner20',
        'task_count': len(tasks), 'fold_count': len(tasks) * 5,
        'tasks': summary,
        'macro8_r2': float(np.mean([summary[task]['test_r2']['mean'] for task in tasks])) if len(tasks) == 8 else None,
        'interpretation': 'fixed five-fold development evaluation; not independent blind evidence',
    }
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output / 'all_fold_metrics.csv', index=False)
    write_json(output / 'summary.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--task', action='append', dest='tasks')
    args = parser.parse_args()
    tasks = list(args.tasks) if args.tasks else list(TASKS)
    result = aggregate(args.root, args.output, tasks=tasks)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
