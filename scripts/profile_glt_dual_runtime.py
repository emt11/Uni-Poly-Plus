#!/usr/bin/env python3
"""Read-only timing profile of current frozen dual-GLT sample preparation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.dataset.cache_lifecycle import snapshot_tree
from src.dataset.glt_dual import build_dual_sample
from src.dataset.glt_dual_pretrain import chemical_targets, prepare_pretrain_sample
from src.training.glt_dual_runtime import open_source, write_json


def _summary(values):
    ordered = sorted(values)
    count = len(ordered)
    return {
        "count": count,
        "total_seconds": float(sum(ordered)),
        "mean_ms": float(statistics.fmean(ordered) * 1000),
        "median_ms": float(statistics.median(ordered) * 1000),
        "p95_ms": float(ordered[min(count - 1, int(0.95 * count))] * 1000),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--cohort-root", required=True)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260915)
    parser.add_argument("--report-json", required=True)
    args = parser.parse_args()
    if args.samples <= 0:
        parser.error("--samples must be positive")
    torch.set_num_threads(1)
    cache_root = Path(args.cache_root).resolve()
    before = snapshot_tree(cache_root)
    opened = time.perf_counter()
    source, _ = open_source(args.cohort_root, cache_root)
    open_seconds = time.perf_counter() - opened
    try:
        count = min(args.samples, len(source))
        # Even spacing is deterministic and avoids profiling only one source region.
        indices = [int(i * len(source) / count) for i in range(count)]
        records, read_times = [], []
        for index in indices:
            started = time.perf_counter()
            records.append(source[index])
            read_times.append(time.perf_counter() - started)

        # Warm only the bounded in-process BRICS/fingerprint memoization.  This
        # is not a persistent sidecar and is part of the existing runtime.
        for topology, _, smiles in records:
            normalized = str(topology.normalized_canonical_smiles)
            chemical_targets(normalized)

        clean_times, prepare_times = [], []
        geometry_valid = 0
        for local, record in enumerate(records):
            started = time.perf_counter()
            clean = build_dual_sample(*record)
            clean_times.append(time.perf_counter() - started)
            geometry_valid += int(clean.geometry_valid)
            started = time.perf_counter()
            prepare_pretrain_sample(
                *record,
                seed=args.seed,
                key=source.samples[indices[local]][0].hex(),
                position=local,
            )
            prepare_times.append(time.perf_counter() - started)
    finally:
        source.close()
    zero_write = snapshot_tree(cache_root) == before
    if not zero_write:
        raise RuntimeError("runtime profile modified the frozen cache")
    clean = _summary(clean_times)
    prepared = _summary(prepare_times)
    # 5,000 updates at the current formal global batch of 1008 consume
    # 5,040,000 samples, or this many source-cohort equivalents.
    formal_samples = 5000 * 1008
    cohort_passes = formal_samples / len(indices) * count / len(source)
    report = {
        "status": "PASS",
        "scope": "read-only current runtime profile; no model forward",
        "cache_root": str(cache_root),
        "cohort_root": str(Path(args.cohort_root).resolve()),
        "cohort_count": len(source),
        "profile_count": count,
        "cohort_load_seconds": open_seconds,
        "source_read": _summary(read_times),
        "build_dual_sample_clean": clean,
        "prepare_pretrain_sample_warm_targets": prepared,
        "geometry_valid_count": geometry_valid,
        "formal_5000_step_sample_count": formal_samples,
        "formal_5000_step_cohort_passes": formal_samples / len(source),
        "repeated_static_builds_per_valid_pretrain_sample": 2,
        "cache_zero_write": True,
    }
    write_json(args.report_json, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
