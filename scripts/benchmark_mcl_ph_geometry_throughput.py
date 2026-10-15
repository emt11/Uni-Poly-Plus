#!/usr/bin/env python3
"""Bounded parallel throughput probe for deterministic Trimer generation."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import statistics
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PI1M = ROOT / 'data/processed/glt_dual_v2/pi1m/cohort_30f17b59bc5862a1/records.jsonl'


def worker_init():
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)


def generate(item):
    from src.dataset.canonical_periodic import build_canonical_periodic_topology
    from src.dataset.trimer_mcl import (TrimerContractError, TrimerGeometryRejection,
                                        attach_finite_trimer_mcl)
    index, smiles, key = item
    started = time.perf_counter()
    try:
        topology = build_canonical_periodic_topology(smiles)
        trimer = attach_finite_trimer_mcl(topology, smiles, sample_key=key)
        if not bool(trimer.trimer_geometry_valid):
            raise ValueError('generator returned invalid geometry without rejection')
        status, reason = 'PASS', ''
    except (TrimerGeometryRejection, TrimerContractError) as error:
        status, reason = type(error).__name__, str(error)
    return {'source_index': index, 'status': status, 'reason': reason,
            'wall_seconds': time.perf_counter() - started}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples', type=int, default=256)
    parser.add_argument('--workers', type=int, default=16)
    args = parser.parse_args()
    if not 1 <= args.samples <= 1024 or not 1 <= args.workers <= 64:
        raise ValueError('probe is limited to <=1024 samples and <=64 workers')
    rng = np.random.default_rng(42)
    selected = set(map(int, rng.choice(959588, size=args.samples, replace=False)))
    items = []
    with PI1M.open(encoding='utf-8') as stream:
        for index, line in enumerate(stream):
            if index in selected:
                row = json.loads(line)
                items.append((index, row['source_smiles'], row['sample_key']))
    if len(items) != args.samples:
        raise ValueError('PI1M source count differs from probe contract')
    started = time.perf_counter()
    results = []
    with ProcessPoolExecutor(max_workers=args.workers, initializer=worker_init) as pool:
        futures = [pool.submit(generate, item) for item in items]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(json.dumps(result), flush=True)
    elapsed = time.perf_counter() - started
    durations = sorted(row['wall_seconds'] for row in results)
    summary = {'samples': len(results), 'workers': args.workers,
               'passes': sum(row['status'] == 'PASS' for row in results),
               'failures': [row for row in results if row['status'] != 'PASS'],
               'wall_seconds': elapsed,
               'throughput_per_second': len(results) / elapsed,
               'sample_time_mean': statistics.mean(durations),
               'sample_time_median': statistics.median(durations),
               'sample_time_p95': durations[int(0.95 * (len(durations) - 1))],
               'sample_time_max': durations[-1]}
    print(json.dumps({'summary': summary}), flush=True)


if __name__ == '__main__':
    main()
