#!/usr/bin/env python3
"""Benchmark read-only cached preparation on real cohort records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.dataset.cache_lifecycle import atomic_json, zero_write_snapshot
from src.dataset.glt_dual import build_dual_sample
from src.dataset.glt_dual_cache import DualFrozenBundle, load_dual_cohort
from src.dataset.glt_dual_static import DualStaticCache, PretrainTargetsCache


def summary(values):
    values = sorted(values)
    return {
        'count': len(values),
        'mean_ms': statistics.fmean(values) * 1000,
        'p50_ms': statistics.median(values) * 1000,
        'p90_ms': values[min(len(values) - 1, int(0.9 * len(values)))] * 1000,
        'p99_ms': values[min(len(values) - 1, int(0.99 * len(values)))] * 1000,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache-root', required=True)
    parser.add_argument('--cohort-root', required=True)
    parser.add_argument('--static-root', required=True)
    parser.add_argument('--target-root')
    parser.add_argument('--samples', type=int, default=10000)
    parser.add_argument('--report-json', required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    cache_root = Path(args.cache_root).resolve()
    before = zero_write_snapshot(cache_root)
    cohort = load_dual_cohort(args.cohort_root, cache_root)
    bundle = DualFrozenBundle(cache_root, expected_bundle_hash=cohort['manifest']['main_bundle_hash'])
    static = DualStaticCache(
        args.static_root,
        parent_bundle_hash=cohort['manifest']['main_bundle_hash'],
        cohort_manifest_hash=cohort['manifest_hash'],
    )
    target = PretrainTargetsCache(
        args.target_root,
        parent_bundle_hash=cohort['manifest']['main_bundle_hash'],
        cohort_manifest_hash=cohort['manifest_hash'],
    ) if args.target_root else None
    try:
        count = min(int(args.samples), len(static))
        indices = list(range(count))
        timings, valid = [], 0
        started = time.perf_counter()
        for index in indices:
            row = cohort['records'][index]
            key = bytes.fromhex(str(row['sample_key']))
            topology, trimer = bundle.topology[key], bundle.trimer[key]
            smiles = row['source_smiles']
            row_started = time.perf_counter()
            item = build_dual_sample(
                topology, trimer, smiles, static=static.get(index)
            )
            timings.append(time.perf_counter() - row_started)
            valid += int(item.geometry_valid)
        elapsed = time.perf_counter() - started
        static_manifest_hash = static.manifest_hash
        target_manifest_hash = target.manifest_hash if target is not None else None
    finally:
        bundle.close()
        static.close()
        if target is not None:
            target.close()
    if zero_write_snapshot(cache_root) != before:
        raise RuntimeError('cached benchmark modified frozen main bundle')
    result = {
        'status': 'PASS',
        'scope': 'real cached static sample preparation only; no model forward',
        'samples': count,
        'geometry_valid_count': valid,
        'elapsed_seconds': elapsed,
        'samples_per_second': count / max(elapsed, 1e-12),
        'latency': summary(timings),
        'static_manifest_hash': static_manifest_hash,
        'target_manifest_hash': target_manifest_hash,
        'static_bytes': sum(path.stat().st_size for path in Path(args.static_root).rglob('*') if path.is_file()),
        'target_bytes': sum(path.stat().st_size for path in Path(args.target_root).rglob('*') if path.is_file()) if args.target_root else None,
        'main_cache_zero_write': True,
    }
    atomic_json(Path(args.report_json), result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
