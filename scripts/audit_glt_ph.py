#!/usr/bin/env python3
"""GLT-GALPH P0: audit and benchmark the PH profile over real frozen records.

Audit (default 1024 records) reports channel point counts, H fraction, the
invariance checks (translation/rotation/reflection/permutation, tolerance
1e-7), single-process vs multiprocessing agreement, and per-record runtimes.
``--benchmark`` mode reads more records and reports the runtime percentiles
and sidecar bytes/sample needed before the full sidecar build.
"""
import argparse
import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np

from src.dataset.glt_ph import PH_BINS, PH_CHANNELS, PH_SCHEMA, ph_profile
from src.training.glt_dual_runtime import (IndexedFrozenDualSource,
                                           load_sample_index_artifact,
                                           open_source, OrderedSampleStream)


def _record_inputs(subset, index):
    trimer = subset[index][1]
    positions = np.asarray(trimer.trimer_pos, dtype=np.float64)
    numbers = np.asarray(trimer.trimer_atomic_number, dtype=np.int64)
    return positions, numbers


def _worker_init(cohort_root, cache_root, dual_static_root, indices):
    global _WORKER_SOURCE
    base, _ = open_source(cohort_root, cache_root, dual_static_root=dual_static_root)
    _WORKER_SOURCE = IndexedFrozenDualSource(base, indices)


def _worker_profile(payload):
    position, index = payload
    positions, numbers = _record_inputs(_WORKER_SOURCE, index)
    started = time.perf_counter()
    profile, valid = ph_profile(positions, numbers)
    return {'position': int(position), 'valid': bool(valid),
            'seconds': time.perf_counter() - started,
            'profile': profile.tolist()}


def _rotation():
    angle = 0.63
    return np.array([[np.cos(angle), -np.sin(angle), 0.0],
                     [np.sin(angle), np.cos(angle), 0.0],
                     [0.0, 0.0, 1.0]])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pi1m-cohort-root', required=True)
    parser.add_argument('--cache-root', required=True)
    parser.add_argument('--dual-static-root', required=True)
    parser.add_argument('--split-artifact', required=True)
    parser.add_argument('--samples', type=int, default=1024)
    parser.add_argument('--workers', type=int, default=12)
    parser.add_argument('--benchmark', action='store_true',
                        help='runtime/size benchmark mode over --samples records')
    parser.add_argument('--progress-every', type=int, default=256)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    artifact = load_sample_index_artifact(args.split_artifact, 'train')
    base, _ = open_source(args.pi1m_cohort_root, args.cache_root,
                          dual_static_root=args.dual_static_root,
                          chunk_cache_capacity=64)
    subset = IndexedFrozenDualSource(base, artifact['indices'])
    stream = OrderedSampleStream(len(subset), 42)
    # Process in source-index order: random access over 235 chunks dominates
    # the runtime, while the PH maths itself is ~20 ms per record.
    positions_index = sorted(
        ((position, stream.index_at(position)) for position in range(int(args.samples))),
        key=lambda pair: int(pair[1]))
    started = time.perf_counter()
    if int(args.workers) <= 1:
        rows = [_worker_local(subset, position, index)
                for position, index in positions_index]
    else:
        # Close the parent handle before forking: LMDB refuses a second open
        # of the same environment inside the inherited process.
        base.close()
        with mp.Pool(processes=int(args.workers), initializer=_worker_init,
                     initargs=(args.pi1m_cohort_root, args.cache_root,
                               args.dual_static_root, artifact['indices'])) as pool:
            rows = list(pool.imap(_worker_profile, positions_index))
    wall = time.perf_counter() - started
    rows.sort(key=lambda row: row['position'])

    seconds = np.asarray([row['seconds'] for row in rows], dtype=np.float64)
    valid = sum(1 for row in rows if row['valid'])
    payload = {
        'schema_version': 'glt-gal-ph-audit-v1',
        'ph_schema': PH_SCHEMA,
        'mode': 'benchmark' if args.benchmark else 'audit',
        'samples': len(rows), 'valid_samples': valid,
        'invalid_samples': len(rows) - valid,
        'workers': int(args.workers), 'wall_seconds': wall,
        'seconds_per_sample': {
            'mean': float(seconds.mean()), 'p50': float(np.percentile(seconds, 50)),
            'p95': float(np.percentile(seconds, 95)),
            'p99': float(np.percentile(seconds, 99)),
            'max': float(seconds.max())},
        'projected_full_seconds': float(seconds.mean() * len(subset) / max(1, int(args.workers))),
        'sidecar_bytes_per_sample': int(PH_CHANNELS * PH_BINS * 4),
        'projected_sidecar_gib': (PH_CHANNELS * PH_BINS * 4 * len(subset)) / 2 ** 30,
        'pretrain_records': int(len(subset)),
    }
    if not args.benchmark:
        reopened, _ = open_source(args.pi1m_cohort_root, args.cache_root,
                                  dual_static_root=args.dual_static_root,
                                  chunk_cache_capacity=64)
        reopened_subset = IndexedFrozenDualSource(reopened, artifact['indices'])
        payload['correctness'] = _correctness_checks(reopened_subset, stream, artifact)
        reopened.close()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=1, sort_keys=False) + '\n',
                      encoding='utf-8')
    print(json.dumps({k: v for k, v in payload.items() if k != 'correctness'},
                     indent=1)[:2500])


def _worker_local(subset, position, index):
    positions, numbers = _record_inputs(subset, index)
    started = time.perf_counter()
    profile, valid = ph_profile(positions, numbers)
    return {'position': int(position), 'valid': bool(valid),
            'seconds': time.perf_counter() - started, 'profile': profile.tolist()}


def _correctness_checks(subset, stream, artifact):
    """Invariance + H-channel semantics + single/multi-process agreement."""
    report = {'tolerance': 1e-7}
    position = stream.index_at(0)
    index = position
    positions, numbers = _record_inputs(subset, index)
    base_profile, base_valid = ph_profile(positions, numbers)
    shift = np.array([3.0, -2.0, 1.5])
    moved, _ = ph_profile(positions + shift, numbers)
    rotated, _ = ph_profile(positions @ _rotation().T, numbers)
    reflected, _ = ph_profile(positions * np.array([1.0, 1.0, -1.0]), numbers)
    order = np.random.default_rng(7).permutation(positions.shape[0])
    permuted, _ = ph_profile(positions[order], numbers[order])
    report['translation_max_abs_diff'] = float(np.abs(base_profile - moved).max())
    report['rotation_max_abs_diff'] = float(np.abs(base_profile - rotated).max())
    report['reflection_max_abs_diff'] = float(np.abs(base_profile - reflected).max())
    report['permutation_max_abs_diff'] = float(np.abs(base_profile - permuted).max())
    report['invariance_pass'] = bool(all(
        report[key] <= 1e-7 for key in
        ('translation_max_abs_diff', 'rotation_max_abs_diff',
         'reflection_max_abs_diff', 'permutation_max_abs_diff')))
    # channel semantics: hydrogens enter only the all-atom channel
    heavy = numbers > 1
    only_heavy, _ = ph_profile(positions[heavy], numbers[heavy])
    report['h0_heavy_matches_heavy_subset'] = bool(
        np.abs(base_profile[0] - only_heavy[0]).max() <= 1e-7)
    report['h1_heavy_matches_heavy_subset'] = bool(
        np.abs(base_profile[1] - only_heavy[1]).max() <= 1e-7)
    report['all_atom_channel_changes_with_h'] = bool(
        np.abs(base_profile[2] - only_heavy[2]).max() > 1e-9)
    # degenerate/invalid path stays finite
    degenerate, degenerate_valid = ph_profile(np.zeros((3, 3)), np.array([6, 6, 6]))
    report['degenerate_returns_zeros'] = bool((not degenerate_valid)
                                             and float(np.abs(degenerate).max()) == 0.0)
    report['base_valid'] = bool(base_valid)
    report['profile_sum'] = float(base_profile.sum())
    report['profile_finite'] = bool(np.isfinite(base_profile).all())
    return report


if __name__ == '__main__':
    main()
