#!/usr/bin/env python3
"""D1 representation comparison for GLT-3D-GAIN-20260921-01.

Compares the frozen B_FP deployment representations with the per-unit S5 best
representations: per-dimension variance, participation-ratio effective rank,
the z2/z3 cosine coupling, and the deploy-vs-finetuned drift of the same rows.
Read-only over the saved ``feat_*`` archives; train/validation only.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

TASKS = ('xc', 'eps', 'eat')
FOLDS = (0, 1)


def effective_rank(values):
    """Participation ratio of the covariance spectrum: (sum l)^2 / sum l^2."""
    centered = np.asarray(values, dtype=np.float64)
    centered = centered - centered.mean(0, keepdims=True)
    covariance = centered.T @ centered / max(1, centered.shape[0] - 1)
    eigenvalues = np.linalg.eigvalsh(covariance)
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    total = eigenvalues.sum()
    if total <= 0:
        return 0.0
    return float(total ** 2 / np.square(eigenvalues).sum())


def coupling(left, right):
    """Mean absolute cosine between paired rows and between unpaired rows."""
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    numerator = (left * right).sum(1)
    denominator = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    paired = np.abs(numerator / np.maximum(denominator, 1e-12))
    roll = np.roll(np.arange(len(left)), 1)
    numerator = (left * right[roll]).sum(1)
    denominator = np.linalg.norm(left, axis=1) * np.linalg.norm(right[roll], axis=1)
    unpaired = np.abs(numerator / np.maximum(denominator, 1e-12))
    return {'paired_mean_abs_cosine': float(paired.mean()),
            'unpaired_mean_abs_cosine': float(unpaired.mean()),
            'paired_max': float(paired.max())}


def describe(values):
    values = np.asarray(values, dtype=np.float64)
    norms = np.linalg.norm(values, axis=1)
    return {
        'rows': int(values.shape[0]),
        'dimensions': int(values.shape[1]),
        'mean_dimension_variance': float(values.var(axis=0, ddof=1).mean()),
        'total_variance': float(values.var(axis=0, ddof=1).sum()),
        'mean_norm': float(norms.mean()),
        'exact_zero_rows': int((np.abs(values).sum(1) == 0).sum()),
        'participation_ratio_rank': effective_rank(values),
    }


def load(root, task, fold, split):
    with np.load(Path(root) / f'{task}_fold{fold}_{split}.npz', allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def drift(reference, other):
    """Cosine similarity between the same rows of two representations."""
    reference = np.asarray(reference, dtype=np.float64)
    other = np.asarray(other, dtype=np.float64)
    if reference.shape != other.shape:
        raise ValueError('drift comparison requires identical row sets')
    numerator = (reference * other).sum(1)
    denominator = np.linalg.norm(reference, axis=1) * np.linalg.norm(other, axis=1)
    cosine = numerator / np.maximum(denominator, 1e-12)
    relative = np.linalg.norm(other - reference, axis=1) / np.maximum(
        np.linalg.norm(reference, axis=1), 1e-12)
    return {'mean_cosine': float(cosine.mean()), 'min_cosine': float(cosine.min()),
            'mean_relative_drift': float(relative.mean()),
            'max_relative_drift': float(relative.max())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, help='D1 output root holding feat_* dirs')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    root = Path(args.root)

    report = {'tasks': {}, 'note': ('train/validation only; outer-test is never read. '
                                    'Participation ratio rank is (sum lambda)^2/sum(lambda^2) '
                                    'of the covariance spectrum.')}
    for task in TASKS:
        for fold in FOLDS:
            unit = {}
            for split in ('train', 'validation'):
                deploy = load(root / 'feat_deploy', task, fold, split)
                finetuned = load(root / 'feat_finetuned', task, fold, split)
                if not np.array_equal(deploy['sample_key'], finetuned['sample_key']):
                    raise ValueError(f'{task}/fold{fold}/{split}: row identity mismatch')
                entry = {
                    'deploy': {'z2': describe(deploy['z2']), 'z3': describe(deploy['z3']),
                               'graph_3d_raw': describe(deploy['graph_3d']),
                               'z2_z3_coupling': coupling(deploy['z2'], deploy['z3'])},
                    'finetuned': {'z2': describe(finetuned['z2']), 'z3': describe(finetuned['z3']),
                                  'graph_3d_raw': describe(finetuned['graph_3d']),
                                  'z2_z3_coupling': coupling(finetuned['z2'], finetuned['z3'])},
                    'finetuned_vs_deploy': {
                        'z2': drift(deploy['z2'], finetuned['z2']),
                        'z3': drift(deploy['z3'], finetuned['z3']),
                        'graph_3d_raw': drift(deploy['graph_3d'], finetuned['graph_3d'])},
                    'valid_rows': int(deploy['geometry_valid'].sum()),
                    'center_bond_zero_rows': int((deploy['center_bond_count'] == 0).sum()),
                }
                unit[split] = entry
                d = entry['deploy']
                print(json.dumps({
                    'unit': f'{task}/fold{fold}', 'split': split,
                    'deploy_z2_var': d['z2']['mean_dimension_variance'],
                    'deploy_z3_var': d['z3']['mean_dimension_variance'],
                    'deploy_z2_rank': d['z2']['participation_ratio_rank'],
                    'deploy_z3_rank': d['z3']['participation_ratio_rank'],
                    'coupling': d['z2_z3_coupling']['paired_mean_abs_cosine'],
                    'finetuned_z3_rank': entry['finetuned']['z3']['participation_ratio_rank'],
                    'z3_drift_cosine': entry['finetuned_vs_deploy']['z3']['mean_cosine'],
                }), flush=True)
            report['tasks'][f'{task}/fold{fold}'] = unit

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({'status': 'PASS', 'output': str(output)}))


if __name__ == '__main__':
    main()
