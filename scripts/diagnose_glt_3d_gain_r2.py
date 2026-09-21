#!/usr/bin/env python3
"""r2 numerical-robustness rerun of the D1 CPU linear probes.

Only the numerical protocol changes relative to r1:

* every feature matrix and label vector is cast to ``float64`` before any
  scaler is fitted;
* Ridge uses ``solver="svd"`` with ``fit_intercept=True``;
* the alpha grid is the fixed nine-point set, and the fit budget is checked
  **before every single fit** rather than after the loop.

The probe definitions themselves are unchanged: ``L2``/``L3``/``L23``/``L23-S``,
the within-split permutation (seed ``42 + fold``), and the residual probe
(three-fold OOF 2D baseline at ``alpha=1``, validation scored against the
full-train ``alpha=1`` baseline).

CPU only.  No model, no GPU, no outer-test, no re-encoding.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

TASKS = ('xc', 'eps', 'eat')
FOLDS = (0, 1)
ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0, 1000000.0)
LEGACY_ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0)
THREAD_KEYS = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
               'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')


class FitBudget:
    """Counts every real-data Ridge fit and refuses to start one over budget."""

    def __init__(self, limit):
        self.limit = int(limit)
        self.count = 0
        self.by_category = {}

    def spend(self, category):
        if self.count >= self.limit:
            raise RuntimeError(
                f'ridge fit budget exhausted before a {category} fit: '
                f'{self.count}/{self.limit}')
        self.count += 1
        self.by_category[category] = self.by_category.get(category, 0) + 1

    def report(self):
        return {'executed': self.count, 'allowed': self.limit,
                'by_category': dict(sorted(self.by_category.items()))}


def fit_ridge(x_train, y_train, x_validation, alpha, budget, category):
    """Train-only standardisation, SVD Ridge, prediction in original units."""
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    x_train = np.asarray(x_train, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64).reshape(-1)
    x_validation = np.asarray(x_validation, dtype=np.float64)
    if x_train.ndim != 2 or x_validation.ndim != 2:
        raise ValueError('ridge inputs must be two-dimensional')
    if x_train.shape[1] != x_validation.shape[1]:
        raise ValueError('train/validation feature width mismatch')
    if x_train.shape[0] != y_train.shape[0]:
        raise ValueError('train feature/label row count mismatch')
    budget.spend(category)
    feature_scaler = StandardScaler().fit(x_train)
    label_scaler = StandardScaler().fit(y_train.reshape(-1, 1))
    model = Ridge(alpha=float(alpha), fit_intercept=True, solver='svd')
    model.fit(feature_scaler.transform(x_train),
              label_scaler.transform(y_train.reshape(-1, 1)).ravel())
    scaled = model.predict(feature_scaler.transform(x_validation)).reshape(-1, 1)
    return label_scaler.inverse_transform(scaled).reshape(-1)


def r2_original(predicted, actual):
    from sklearn.metrics import r2_score
    return float(r2_score(np.asarray(actual, dtype=np.float64).reshape(-1, 1),
                          np.asarray(predicted, dtype=np.float64).reshape(-1, 1)))


def load_feature(base, task, fold, split):
    with np.load(Path(base) / f'{task}_fold{fold}_{split}.npz', allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def select(candidates):
    """Pick the best alpha; record whether the winner sits on a grid boundary."""
    best = max(candidates.items(), key=lambda item: item[1]['validation_r2'])
    values = [item[1]['validation_r2'] for item in candidates.items()]
    ordered = sorted(candidates, key=lambda key: float(key))
    boundary = 'interior'
    if float(best[0]) == float(ordered[0]):
        boundary = 'lower_bound'
    elif float(best[0]) == float(ordered[-1]):
        boundary = 'upper_bound'
    return {'selected_alpha': float(best[0]),
            'validation_r2': best[1]['validation_r2'],
            'boundary': boundary,
            'alpha_spread': float(max(values) - min(values))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--feature-root', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--fit-budget', type=int, default=294)
    args = parser.parse_args()

    started = time.perf_counter()
    report = {
        'feature_root': str(Path(args.feature_root).resolve()),
        'alphas': list(ALPHAS), 'legacy_alphas': list(LEGACY_ALPHAS),
        'ridge': {'solver': 'svd', 'fit_intercept': True, 'dtype': 'float64',
                  'note': 'features and labels are cast to float64 before scaling'},
        'permutation_note': ('z3 shuffle is drawn inside each split with seed 42 + fold; '
                             'no label use, no cross-split exchange'),
        'residual_definition': ('KFold(3, shuffle=True, random_state=42+fold) OOF from an '
                               'alpha=1 2D Ridge; validation prediction is the full-train '
                               'alpha=1 2D Ridge plus the 3D residual prediction'),
        'thread_settings': {key: os.environ.get(key) for key in THREAD_KEYS},
        'units': {}, 'fit_budget': {}, 'timing': {},
    }
    budget = FitBudget(args.fit_budget)
    base = Path(args.feature_root)

    for task in TASKS:
        for fold in FOLDS:
            unit_key = f'{task}/fold{fold}'
            train = load_feature(base, task, fold, 'train')
            validation = load_feature(base, task, fold, 'validation')
            y_train = np.asarray(train['label'], dtype=np.float64).reshape(-1)
            y_validation = np.asarray(validation['label'], dtype=np.float64).reshape(-1)

            perm_train = np.random.default_rng(42 + fold).permutation(len(y_train))
            perm_validation = np.random.default_rng(42 + fold).permutation(len(y_validation))

            families = {
                'L2': (train['z2'], validation['z2']),
                'L3': (train['z3'], validation['z3']),
                'L23': (np.concatenate([train['z2'], train['z3']], -1),
                        np.concatenate([validation['z2'], validation['z3']], -1)),
                'L23-S': (np.concatenate([train['z2'], train['z3'][perm_train]], -1),
                          np.concatenate([validation['z2'], validation['z3'][perm_validation]], -1)),
            }
            unit = {}
            for name, (x_train, x_validation) in families.items():
                x_train = np.asarray(x_train, dtype=np.float64)
                x_validation = np.asarray(x_validation, dtype=np.float64)
                candidates = {}
                for alpha in ALPHAS:
                    predicted = fit_ridge(x_train, y_train, x_validation, alpha,
                                          budget, name)
                    candidates[repr(float(alpha))] = {
                        'alpha': float(alpha),
                        'validation_r2': r2_original(predicted, y_validation)}
                unit[name] = {**select(candidates), 'candidates': candidates,
                              'feature_dim': int(x_train.shape[1]),
                              'feature_dtype_in': str(train['z2'].dtype)}
                print(json.dumps({'unit': unit_key, 'probe': name,
                                  'r2': unit[name]['validation_r2'],
                                  'alpha': unit[name]['selected_alpha'],
                                  'boundary': unit[name]['boundary']}), flush=True)

            from sklearn.model_selection import KFold
            oof = np.zeros(len(y_train), dtype=np.float64)
            filled = np.zeros(len(y_train), dtype=np.int64)
            for inner_train, inner_test in KFold(3, shuffle=True,
                                                 random_state=42 + fold).split(train['z2']):
                predicted = fit_ridge(train['z2'][inner_train], y_train[inner_train],
                                      train['z2'][inner_test], 1.0, budget, 'oof_baseline')
                oof[inner_test] = predicted
                filled[inner_test] += 1
            if not bool((filled == 1).all()):
                raise ValueError('OOF rows were not filled exactly once per inner fold')
            residual = y_train - oof
            baseline_prediction = fit_ridge(train['z2'], y_train, validation['z2'],
                                            1.0, budget, 'full_baseline')
            baseline_r2 = r2_original(baseline_prediction, y_validation)
            residual_candidates = {}
            for alpha in ALPHAS:
                residual_prediction = fit_ridge(train['z3'], residual, validation['z3'],
                                                alpha, budget, 'residual')
                residual_candidates[repr(float(alpha))] = {
                    'alpha': float(alpha),
                    'validation_r2': r2_original(baseline_prediction + residual_prediction,
                                                 y_validation)}
            best_residual = select(residual_candidates)
            unit['residual'] = {
                'baseline_alpha': 1.0,
                'baseline_validation_r2': baseline_r2,
                'oof_alpha': 1.0,
                'oof_inner_folds': 3,
                'oof_rows_filled_once': bool((filled == 1).all()),
                'candidates': residual_candidates,
                'selected_alpha': best_residual['selected_alpha'],
                'validation_r2': best_residual['validation_r2'],
                'boundary': best_residual['boundary'],
                'delta_vs_alpha1_baseline': best_residual['validation_r2'] - baseline_r2,
            }
            print(json.dumps({'unit': unit_key, 'probe': 'residual',
                              'r2': best_residual['validation_r2'],
                              'baseline': baseline_r2,
                              'delta': unit['residual']['delta_vs_alpha1_baseline'],
                              'alpha': best_residual['selected_alpha'],
                              'boundary': best_residual['boundary']}), flush=True)
            report['units'][unit_key] = unit

    report['fit_budget'] = budget.report()
    report['timing'] = {'wall_seconds': float(time.perf_counter() - started)}
    if budget.count > args.fit_budget:
        raise RuntimeError('ridge fit budget exceeded')
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({'status': 'PASS', 'output': str(output),
                      'fits': budget.count, 'allowed': args.fit_budget,
                      'wall_seconds': report['timing']['wall_seconds']}))


if __name__ == '__main__':
    main()
