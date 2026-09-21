#!/usr/bin/env python3
"""Model-free P0 revision for plan MCL-PH-20260921-01/r2: the Randic column.

The r1 P0 pass measured the five router descriptor columns with the unnormalised
Randic sum, so its reported column 0 ran to ~80 and the declared [0,1] range did
not hold.  ``src/dataset/mcl_ph_view.py`` now implements the declared formula
``2/n * sum_edges 1/sqrt(deg(u) deg(v))``; this script recomputes *only* the
statistics that fix changes, over the identical 4096-sample P_train set (same
ordering policy, same seed and noise sigma) and writes a separate report.

Nothing else is recomputed: the length / non-bond normalization statistics, the
mapping totals, the memory projection and the PH-definition fixtures do not
depend on the Randic column and keep their r1 values.  The frozen r1 artifacts
are never modified.

Budget contract: CPU only, zero model calls, zero optimizer updates.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from scripts.audit_mcl_ph_p0 import (DEFAULT_CACHE, DEFAULT_COHORT, DEFAULT_CONFIG,  # noqa: E402
                                     DEFAULT_STATIC, STATISTICS_SAMPLES, _sha256_keys,
                                     ordered_key_positions, open_source)
from src.dataset import mcl_ph_view as view  # noqa: E402
from src.training.glt_dual_runtime import (IndexedFrozenDualSource,  # noqa: E402
                                           load_sample_index_artifact)

FROZEN_REPORT = 'results/mcl_ph_20260921/p0/audit.json'
FROZEN_STATISTICS = 'results/mcl_ph_20260921/p0/statistics.npz'
DEFAULT_OUTPUT = 'results/mcl_ph_20260921/p0/statistics_randic_revision.json'
DEFAULT_NPZ = 'results/mcl_ph_20260921/p0/statistics_randic_revision.npz'
COLUMNS = ('randic', 'wiener', 'efficiency', 'betti0_per_atom', 'betti1_per_edge')
BUDGET_SECONDS = 30 * 60


def _column_summary(stack):
    """Per-column min/max/mean/std over every sample and radius."""
    summary = {}
    for index, name in enumerate(COLUMNS):
        values = stack[:, :, index]
        summary[name] = {
            'min': float(values.min()), 'max': float(values.max()),
            'mean': float(values.mean()), 'std': float(values.std()),
            'within_declared_range': bool(values.min() >= 0.0 and values.max() <= 1.0),
        }
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cohort-root', default=DEFAULT_COHORT)
    parser.add_argument('--cache-root', default=DEFAULT_CACHE)
    parser.add_argument('--dual-static-root', default=DEFAULT_STATIC)
    parser.add_argument('--config', default=DEFAULT_CONFIG)
    parser.add_argument('--output', default=DEFAULT_OUTPUT)
    parser.add_argument('--npz-output', default=DEFAULT_NPZ)
    parser.add_argument('--frozen-report', default=FROZEN_REPORT)
    parser.add_argument('--frozen-statistics', default=FROZEN_STATISTICS)
    parser.add_argument('--statistics-samples', type=int, default=STATISTICS_SAMPLES)
    parser.add_argument('--budget-seconds', type=float, default=BUDGET_SECONDS)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--sigma', type=float, default=0.03)
    args = parser.parse_args()
    started = time.perf_counter()
    config = json.loads(Path(args.config).read_text(encoding='utf-8'))
    split = load_sample_index_artifact(config['sample_index_artifact'],
                                       config['sample_index_split'])
    source, _ = open_source(args.cohort_root, args.cache_root,
                            dual_static_root=args.dual_static_root)
    subset = IndexedFrozenDualSource(source, split['indices'])
    order, keys = ordered_key_positions(subset, int(args.statistics_samples))
    frozen = json.loads(Path(args.frozen_report).read_text(encoding='utf-8'))
    report = {
        'plan': 'MCL-PH-20260921-01/r2', 'phase': 'P0 revision',
        'subject': 'the Randic descriptor column of the P_train statistics',
        'command': sys.argv, 'devices': 'cpu', 'model_calls': 0,
        'optimizer_updates': 0, 'budget_seconds': float(args.budget_seconds),
        'inputs': {
            'cohort_root': str(Path(args.cohort_root).resolve()),
            'config': str(Path(args.config).resolve()),
            'config_sha256': hashlib.sha256(Path(args.config).read_bytes()).hexdigest(),
            'sample_index_sha256': split['sha256'],
            'sample_index_split': split['split'],
            'seed': int(args.seed), 'sigma': float(args.sigma),
        },
        'sample_set': {
            'ordering_policy': 'first N of the P_train split sorted by raw 32-byte key order',
            'selected': len(order), 'first_key': bytes(keys[order[0]]).hex(),
            'last_key': bytes(keys[order[-1]]).hex(),
            'ordered_key_sha256': None,  # filled below
        },
        'supersedes': {
            'report_fields': [
                'statistics.router_mean', 'statistics.router_std',
            ],
            'report_values': {
                'router_mean': frozen['statistics']['router_mean'],
                'router_std': frozen['statistics']['router_std'],
            },
            'statistics_artifact_fields': ['router_mean', 'router_std'],
            'not_superseded': [
                'statistics.length', 'statistics.nonbond', 'statistics.target_counts',
                'statistics.nonbond_bin_histogram', 'statistics.real_zero_center_bond_keys',
                'mapping', 'memory', 'ph_definition', 'identity_source',
                'cost_projection',
            ],
            'reason': 'the length / non-bond normalization, the mapping totals, the '
                      'memory projection and the PH fixtures are computed from raw '
                      'distances and graph structure, none of which the Randic fix '
                      'touches',
            'consumer_note': 'router_mean / router_std are recorded evidence: neither '
                             'the data path nor the model reads them (they are loaded '
                             'into the statistics record and never indexed outside it), '
                             'so no training normalization depended on the stale column',
        },
    }
    report['sample_set']['ordered_key_sha256'] = _sha256_keys(keys, order)
    frozen_sha = frozen['sample_sets'].get('ordered_key_sha256')
    report['sample_set']['matches_frozen_sample_set'] = bool(
        frozen_sha is None or frozen_sha == report['sample_set']['ordered_key_sha256'])
    stack, partial = [], False
    loop_started = time.perf_counter()
    for position in order:
        if time.perf_counter() - started > args.budget_seconds:
            partial = True
            break
        trimer = subset[position][1]
        key = bytes(subset.samples[position][0]).hex()
        reference = view.reference_view(trimer, key, sigma=args.sigma, seed=args.seed)
        field = view.build_trimer_view(reference)
        stack.append(view.five_descriptors(field.positions.numpy()).astype(np.float32))
    matrix = (np.stack(stack) if stack else
              np.zeros((0, len(view.ROUTER_RADII), view.DESCRIPTOR_COLUMNS), dtype=np.float32))
    report['statistics'] = {
        'samples_requested': int(len(order)), 'samples_used': int(matrix.shape[0]),
        'partial': partial, 'seconds': time.perf_counter() - loop_started,
        'total_seconds': time.perf_counter() - started,
        'router_mean': matrix.mean(axis=0).tolist() if matrix.shape[0] else None,
        'router_std': matrix.std(axis=0).tolist() if matrix.shape[0] else None,
        'columns': _column_summary(matrix) if matrix.shape[0] else None,
        'declared_ranges': {
            'randic': [0.0, 1.0], 'wiener': [0.0, 1.0], 'efficiency': [0.0, 1.0],
            'betti0_per_atom': [0.0, 1.0], 'betti1_per_edge': [0.0, 1.0]},
        'router_mean_range': ([float(matrix.mean(axis=0).min()),
                               float(matrix.mean(axis=0).max())]
                              if matrix.shape[0] else None),
        'radii': list(view.ROUTER_RADII),
    }
    report['status'] = 'PARTIAL' if partial else 'PASS'
    report['outer_test'] = 'NOT_RUN'
    report['frozen_artifacts_modified'] = False
    if matrix.shape[0]:
        np.savez(args.npz_output,
                 router_mean=np.asarray(matrix.mean(axis=0), dtype=np.float32),
                 router_std=np.asarray(matrix.std(axis=0), dtype=np.float32),
                 samples=np.asarray(matrix.shape[0], dtype=np.int64),
                 seed=np.asarray(args.seed, dtype=np.int64),
                 sigma=np.asarray(args.sigma, dtype=np.float64),
                 ordered_key_sha256=np.asarray(report['sample_set']['ordered_key_sha256']))
        report['npz_output'] = str(args.npz_output)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, sort_keys=True,
                                      allow_nan=False) + '\n', encoding='utf-8')
    summary = {name: (round(value['min'], 6), round(value['max'], 6))
               for name, value in (report['statistics']['columns'] or {}).items()}
    print(json.dumps({'status': report['status'],
                      'seconds': round(report['statistics']['seconds'], 1),
                      'samples_used': report['statistics']['samples_used'],
                      'column_ranges': summary,
                      'router_mean_range': report['statistics']['router_mean_range'],
                      'output': str(destination)}, ensure_ascii=False), flush=True)
    if partial:
        raise SystemExit(3)


if __name__ == '__main__':
    main()
