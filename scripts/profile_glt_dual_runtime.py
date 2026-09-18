#!/usr/bin/env python3
"""Read-only timing profile of current frozen dual-GLT sample preparation."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import resource
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.dataset.cache_lifecycle import zero_write_snapshot
from src.dataset.glt_dual import build_dual_sample
from src.dataset.glt_dual_pretrain import prepare_pretrain_sample, pretrain_collate
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


def _current_rss_bytes():
    """Return the parent process RSS without depending on psutil."""

    try:
        for line in Path('/proc/self/status').read_text(encoding='utf-8').splitlines():
            if line.startswith('VmRSS:'):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError):
        return None
    return None


def _size_summary(values):
    values = [int(value) for value in values]
    if not values:
        return {'count': 0, 'min': None, 'median': None, 'p95': None, 'max': None}
    ordered = sorted(values)
    return {
        'count': len(ordered), 'min': ordered[0],
        'median': int(statistics.median(ordered)),
        'p95': ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))],
        'max': ordered[-1],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--cohort-root", required=True)
    parser.add_argument("--dual-static-root",
                        help="optional training-ready dual_static_v1 artifact")
    parser.add_argument("--pretrain-target-root",
                        help="optional pretrain_targets_v1 artifact")
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260915)
    parser.add_argument("--report-json", required=True)
    args = parser.parse_args()
    if args.samples <= 0:
        parser.error("--samples must be positive")
    torch.set_num_threads(1)
    cache_root = Path(args.cache_root).resolve()
    before = zero_write_snapshot(cache_root)
    source = None
    error = None
    opened = time.perf_counter()
    try:
        source, _ = open_source(
            args.cohort_root, cache_root,
            dual_static_root=args.dual_static_root,
            pretrain_target_root=args.pretrain_target_root,
        )
        open_seconds = time.perf_counter() - opened
        count = min(args.samples, len(source))
        if count <= 0:
            raise ValueError('profile source is empty')
        # A fixed random permutation avoids a biased prefix/periodic sample
        # while keeping the exact order reproducible and shared by every phase.
        generator = torch.Generator().manual_seed(args.seed)
        indices = torch.randperm(len(source), generator=generator)[:count].tolist()
        sample_keys = [source.samples[int(index)][0].hex() for index in indices]
        phase = {name: [] for name in (
            'source_read', 'static_read', 'target_read', 'clean_prepare',
            'noisy_prepare', 'collate', 'complete_sample')}
        graph_sizes = {'nodes': [], 'bonds': [], 'relations': [], 'valid_relation_steps': []}
        geometry_valid = 0
        for local, raw_index in enumerate(indices):
            index = int(raw_index)
            started = time.perf_counter()
            record = source[index]
            after_source = time.perf_counter()
            static_started = time.perf_counter()
            static = source.static_for(index) if args.dual_static_root else None
            after_static = time.perf_counter()
            target_started = time.perf_counter()
            target = source.target_for(index) if args.pretrain_target_root else None
            after_target = time.perf_counter()
            clean_started = time.perf_counter()
            clean = build_dual_sample(*record, static=static)
            after_clean = time.perf_counter()
            prepared_started = time.perf_counter()
            noisy, labels = prepare_pretrain_sample(
                *record,
                seed=args.seed,
                key=source.samples[index][0].hex(),
                position=local,
                static=static,
                target=target,
            )
            after_prepare = time.perf_counter()
            collate_started = time.perf_counter()
            pretrain_collate([(noisy, labels)])
            after_collate = time.perf_counter()
            phase['source_read'].append(after_source - started)
            phase['static_read'].append(after_static - static_started)
            phase['target_read'].append(after_target - target_started)
            phase['clean_prepare'].append(after_clean - clean_started)
            phase['noisy_prepare'].append(after_prepare - prepared_started)
            phase['collate'].append(after_collate - collate_started)
            phase['complete_sample'].append(after_collate - started)
            geometry_valid += int(clean.geometry_valid)
            graph_sizes['nodes'].append(int(clean.mips_x.size(0)))
            graph_sizes['bonds'].append(int(clean.bond_distance.numel()))
            graph_sizes['relations'].append(int(clean.line_path.size(0)))
            graph_sizes['valid_relation_steps'].append(int(clean.line_mask.sum()))
        cache_zero_write = zero_write_snapshot(cache_root) == before
        if not cache_zero_write:
            raise RuntimeError('runtime profile modified the frozen cache')
        report = {
            'status': 'PASS',
            'scope': 'read-only current runtime profile; no model forward',
            'cache_root': str(cache_root),
            'cohort_root': str(Path(args.cohort_root).resolve()),
            'dual_static_root': (str(Path(args.dual_static_root).resolve())
                                 if args.dual_static_root else None),
            'pretrain_target_root': (str(Path(args.pretrain_target_root).resolve())
                                     if args.pretrain_target_root else None),
            'cohort_count': len(source),
            'profile_count': count,
            'seed': int(args.seed),
            'sample_indices': [int(index) for index in indices],
            'sample_keys': sample_keys,
            'cohort_load_seconds': open_seconds,
            'phases': {name: _summary(values) for name, values in phase.items()},
            'geometry_valid_count': geometry_valid,
            'geometry_valid_fraction': geometry_valid / count,
            'graph_sizes': {name: _size_summary(values) for name, values in graph_sizes.items()},
            'resource': {
                'pid': os.getpid(),
                'rss_bytes_at_end': _current_rss_bytes(),
                'ru_maxrss_kib': int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
            },
            'formal_5000_step_sample_count': 5000 * 1008,
            'formal_5000_step_cohort_passes': (5000 * 1008) / len(source),
            'cache_zero_write': True,
        }
    except Exception as exc:
        error = exc
    finally:
        if source is not None:
            source.close()
    if error is not None:
        report = {
            'status': 'FAILED', 'scope': 'read-only current runtime profile; no model forward',
            'cache_root': str(cache_root), 'profile_count': 0,
            'error_type': type(error).__name__, 'error': str(error),
            'cache_zero_write': zero_write_snapshot(cache_root) == before,
        }
        write_json(args.report_json, report)
        raise error.with_traceback(error.__traceback__)
    write_json(args.report_json, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
