#!/usr/bin/env python3
"""Aggregate the S5 formal outer-test comparison: B_NONE vs B_FP.

Validates all 80 shards (provenance, exactly-once marker, checkpoint identity,
row-index/target alignment, exact one-pass OOF coverage per task), then
computes per-task fold metrics with ddof=0, pooled-OOF metrics, macro8 from
fold-mean R2, paired fold deltas and the pre-registered formal gate.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ARMS = ('b_fp', 'b_none')
TASKS = ('eat', 'eea', 'egb', 'egc', 'ei', 'eps', 'nc', 'xc')
FOLDS = (0, 1, 2, 3, 4)
CHECKPOINTS = {
    'b_fp': 'results/glt_pred_20260918/s3b_formal/b_fp/pretrain/deploy_05000.pt',
    'b_none': 'results/glt_pred_20260918/s3b_formal/b_none/pretrain/deploy_05000.pt',
}


def _mean(values):
    return float(np.mean(values))


def _std(values):
    return float(np.std(values, ddof=0))


def load_shard(root, arm, task, fold):
    folder = root / arm / task / f'fold{fold}'
    # The launcher passes the shard output as <root>/<arm>/<task>/fold<k>;
    # finetune then nests its per-task layout one level deeper inside.
    inner = folder / task / f'fold{fold}'
    metrics = json.loads((inner / 'metrics.json').read_text(encoding='utf-8'))
    assert metrics['protocol'] == 'outer5_inner20_formal_shard', (arm, task, fold)
    assert metrics['formal_shard'] is True, (arm, task, fold)
    assert metrics['adaptation'] == 'full', (arm, task, fold)
    assert int(metrics['deployment_step']) == 5000, (arm, task, fold)
    assert metrics['outer_test'] == 'RUN_ONCE', (arm, task, fold)
    assert metrics['task'] == task and int(metrics['fold']) == fold, (arm, task, fold)
    run = json.loads((folder / 'run.json').read_text(encoding='utf-8'))
    command = run['command']
    checkpoint = command[command.index('--checkpoint') + 1]
    assert Path(checkpoint).resolve() == Path(CHECKPOINTS[arm]).resolve(), (arm, checkpoint)
    marker = json.loads((inner / 'outer_test_access.json').read_text(encoding='utf-8'))
    assert marker['status'] == 'COMPLETED', (arm, task, fold, marker)
    assert marker['task'] == task and int(marker['fold']) == fold
    assert marker['arm'] == arm, (arm, marker)
    assert int(marker['fold']) == int(metrics['fold'])
    predictions = pd.read_csv(inner / 'predictions.csv')
    assert predictions['row_index'].is_unique, (arm, task, fold)
    assert {'row_index', 'target', 'prediction'} <= set(predictions.columns)
    return metrics, predictions


def pooled_metrics(frame):
    return {
        'r2': float(r2_score(frame['target'].to_numpy(), frame['prediction'].to_numpy())),
        'mae': float(mean_absolute_error(frame['target'].to_numpy(), frame['prediction'].to_numpy())),
        'rmse': float(np.sqrt(mean_squared_error(frame['target'].to_numpy(),
                                                frame['prediction'].to_numpy()))),
        'samples': int(len(frame)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='results/glt_pred_20260918/s5_formal')
    parser.add_argument('--raw-root', default='data/raw')
    parser.add_argument('--output-root', default='results/glt_pred_20260918/s5_formal')
    args = parser.parse_args()
    root = Path(args.root)
    summary = {'units': 0, 'arms': {}, 'paired': {}, 'macro8': {}, 'gate': {}}

    for arm in ARMS:
        per_task = {}
        for task in TASKS:
            folds = {}
            frames = []
            for fold in FOLDS:
                metrics, predictions = load_shard(root, arm, task, fold)
                folds[fold] = {'r2': float(metrics['test_r2']),
                               'mae': float(metrics['test_mae']),
                               'rmse': float(metrics['test_rmse'])}
                frames.append(predictions)
                summary['units'] += 1
            r2_values = [folds[fold]['r2'] for fold in FOLDS]
            mae_values = [folds[fold]['mae'] for fold in FOLDS]
            rmse_values = [folds[fold]['rmse'] for fold in FOLDS]
            oof = pd.concat(frames, ignore_index=True)
            labels = pd.read_csv(Path(args.raw_root) / f'smi_{task}.csv')
            label_column = labels.columns[1]
            expected_targets = labels[label_column].to_numpy(dtype=np.float64)
            row_index = oof['row_index'].to_numpy()
            assert len(row_index) == len(labels), (arm, task, len(row_index), len(labels))
            assert sorted(row_index.tolist()) == list(range(len(labels))), (arm, task)
            # targets pass through float32 batch tensors, so the round trip
            # carries ~1e-5 relative error on large-valued properties.
            order = np.argsort(row_index)
            assert np.allclose(oof['target'].to_numpy()[order],
                               expected_targets[row_index[order]],
                               rtol=1e-5, atol=1e-4), (arm, task)
            per_task[task] = {
                'folds': {str(fold): folds[fold] for fold in FOLDS},
                'r2_mean': _mean(r2_values), 'r2_std': _std(r2_values),
                'mae_mean': _mean(mae_values), 'mae_std': _std(mae_values),
                'rmse_mean': _mean(rmse_values), 'rmse_std': _std(rmse_values),
                'pooled_oof': pooled_metrics(oof),
            }
        summary['arms'][arm] = per_task
        summary['macro8'][arm] = _mean([per_task[task]['r2_mean'] for task in TASKS])

    # Cross-arm identity: identical fold test sets and targets.
    for task in TASKS:
        for fold in FOLDS:
            left = pd.read_csv(root / 'b_fp' / task / f'fold{fold}' / task / f'fold{fold}' / 'predictions.csv')
            right = pd.read_csv(root / 'b_none' / task / f'fold{fold}' / task / f'fold{fold}' / 'predictions.csv')
            assert np.array_equal(left['row_index'].to_numpy(), right['row_index'].to_numpy()), (task, fold)
            assert np.allclose(left['target'].to_numpy(), right['target'].to_numpy(),
                               rtol=1e-6, atol=1e-6), (task, fold)

    paired = {}
    for task in TASKS:
        deltas = {fold: summary['arms']['b_none'][task]['folds'][str(fold)]['r2']
                  - summary['arms']['b_fp'][task]['folds'][str(fold)]['r2']
                  for fold in FOLDS}
        paired[task] = {
            'fold_deltas': {str(fold): round(deltas[fold], 6) for fold in FOLDS},
            'delta_mean': _mean(list(deltas.values())),
        }
    summary['paired'] = paired

    xc = paired['xc']
    xc_positive = sum(1 for fold in FOLDS if xc['fold_deltas'][str(fold)] > 0)
    macro_delta = summary['macro8']['b_none'] - summary['macro8']['b_fp']
    xc_condition = xc['delta_mean'] > 0 and xc_positive >= 3
    protection = macro_delta >= -0.005
    if xc_condition and protection:
        verdict, selected = 'PASS', 'B_NONE'
    elif xc_condition:
        verdict, selected = 'XC_SPECIFIC_ONLY', 'B_FP'
    else:
        verdict, selected = 'FAIL', 'B_FP'
    summary['gate'] = {
        'xc_delta_folds': xc['fold_deltas'],
        'xc_delta_mean': xc['delta_mean'],
        'xc_positive_fold_count': xc_positive,
        'xc_condition': bool(xc_condition),
        'macro8_delta': macro_delta,
        'macro8_protection_ge_-0.005': bool(protection),
        'B_NONE_FORMAL_GATE': verdict,
        'FINAL_SELECTED_MODEL': selected,
    }

    output = Path(args.output_root)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'summary.json').write_text(
        json.dumps(summary, indent=1, sort_keys=False) + '\n', encoding='utf-8')

    lines = ['# S5 正式 outer-test 汇总（B_NONE vs B_FP）', '',
             f"units: {summary['units']}/80", '',
             '## 每 task R²（5 fold mean ± std, ddof=0）', '',
             '| task | B_FP | B_NONE | delta(mean) |', '| --- | --- | --- | --- |']
    for task in TASKS:
        left = summary['arms']['b_fp'][task]
        right = summary['arms']['b_none'][task]
        lines.append(f"| {task} | {left['r2_mean']:.6f} ± {left['r2_std']:.6f} | "
                     f"{right['r2_mean']:.6f} ± {right['r2_std']:.6f} | "
                     f"{paired[task]['delta_mean']:+.6f} |")
    lines += ['', '## pooled OOF R²', '',
              '| task | B_FP | B_NONE |', '| --- | --- | --- |']
    for task in TASKS:
        lines.append(f"| {task} | {summary['arms']['b_fp'][task]['pooled_oof']['r2']:.6f} | "
                     f"{summary['arms']['b_none'][task]['pooled_oof']['r2']:.6f} |")
    lines += ['', '## macro8 与正式判定', '',
              f"- B_FP_MACRO8_R2 = {summary['macro8']['b_fp']:.6f}",
              f"- B_NONE_MACRO8_R2 = {summary['macro8']['b_none']:.6f}",
              f"- MACRO8_DELTA = {macro_delta:+.6f}",
              f"- XC_DELTA_MEAN = {xc['delta_mean']:+.6f}, positive folds = {xc_positive}/5",
              f"- B_NONE_FORMAL_GATE = **{verdict}**",
              f"- FINAL_SELECTED_MODEL = **{selected}**", '']
    (output / 'report.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps({'units': summary['units'],
                      'macro8': summary['macro8'],
                      'xc_delta_mean': xc['delta_mean'],
                      'xc_positive_fold_count': xc_positive,
                      'macro8_delta': macro_delta,
                      'gate': verdict,
                      'selected': selected}, indent=1))


if __name__ == '__main__':
    main()
