#!/usr/bin/env python3
"""Recompute the fixed 1,024-record FGR coverage with true-Trimer pairs."""

import argparse
import json
import math
from collections import Counter
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch

from src.dataset.glt_dual_pretrain import fgr_pairs
from src.training.glt_dual_runtime import open_source


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cohort-root', required=True)
    parser.add_argument('--cache-root', required=True)
    parser.add_argument('--dual-static-root', required=True)
    parser.add_argument('--historical-audit', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    historical = json.loads(Path(args.historical_audit).read_text(encoding='utf-8'))
    selected = historical.get('sampled_records', [])
    if len(selected) != 1024:
        raise ValueError('historical audit does not contain exactly 1,024 selected records')
    source, _ = open_source(
        args.cohort_root, args.cache_root, dual_static_root=args.dual_static_root,
        pretrain_target_root=None,
    )
    pair_counts, spd2_counts, spd3_counts = [], [], []
    geometry_valid_count = 0
    fgr_valid_graph_count = 0
    no_pair_count = 0
    ru_sizes = Counter()
    details = []
    try:
        for item in selected:
            index = int(item['index'])
            key, smiles = source.samples[index]
            topology, trimer, source_smiles = source[index]
            if key.hex() != str(item['sample_key']):
                raise ValueError(f'coverage sample key mismatch at index {index}')
            if str(source_smiles) != str(smiles):
                raise ValueError(f'coverage source identity mismatch at index {index}')
            identity = None
            try:
                pair_index, target, spd = fgr_pairs(
                    topology, trimer, source_smiles, identity=identity,
                    seed=args.seed, key=key.hex(), position=index,
                    mu=0.0, sigma=1.0, max_pairs=32,
                )
            except ValueError as exc:
                # A strict identity/mapping error is a contract failure, not a
                # geometry fallback; let it abort the coverage run.
                raise RuntimeError(f'FGR contract failure at index {index}: {exc}') from exc
            geometry_valid = bool(getattr(trimer, 'trimer_geometry_valid', False))
            geometry_valid_count += int(geometry_valid)
            pair_count = int(pair_index.size(0))
            pair_counts.append(pair_count)
            spd2 = int((spd == 2).sum())
            spd3 = int((spd == 3).sum())
            spd2_counts.append(spd2)
            spd3_counts.append(spd3)
            if geometry_valid and pair_count:
                fgr_valid_graph_count += 1
            if pair_count == 0:
                no_pair_count += 1
            ru_sizes[str(int(topology.mips_x.size(0)))] += 1
            details.append({
                'index': index, 'sample_key': key.hex(),
                'geometry_valid': geometry_valid, 'pair_count': pair_count,
                'spd2_count': spd2, 'spd3_count': spd3,
            })
    finally:
        source.close()
    coverage = fgr_valid_graph_count / len(selected)
    report = {
        'status': 'PASS', 'seed': int(args.seed), 'selected_count': len(selected),
        'geometry_valid_count': geometry_valid_count,
        'fgr_valid_graph_count': fgr_valid_graph_count,
        'no_pair_count': no_pair_count, 'coverage_fraction': coverage,
        'coverage_gate': 'PASS' if coverage >= 0.50 else 'FAIL',
        'spd2_graph_count': sum(value > 0 for value in spd2_counts),
        'spd3_graph_count': sum(value > 0 for value in spd3_counts),
        'spd2_pair_total': sum(spd2_counts), 'spd3_pair_total': sum(spd3_counts),
        'pair_count_mean': statistics.fmean(pair_counts) if pair_counts else 0.0,
        'pair_count_p50': statistics.median(pair_counts) if pair_counts else 0.0,
        'pair_count_p95': sorted(pair_counts)[min(len(pair_counts) - 1, math.ceil(.95 * len(pair_counts)) - 1)] if pair_counts else 0,
        'pair_count_max': max(pair_counts) if pair_counts else 0,
        'ru_size_distribution': dict(sorted(ru_sizes.items(), key=lambda item: int(item[0]))),
        'historical_result': {
            'path': str(Path(args.historical_audit).resolve()),
            'coverage_fraction': historical.get('target_coverage', {}).get('fgr_coverage_fraction'),
            'label': 'historical invalid implementation result',
        },
        'outer_test_accessed': False, 'active_cache_modified': False,
        'records': details,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False), encoding='utf-8')
    print(json.dumps({key: report[key] for key in ('status', 'selected_count', 'coverage_fraction', 'coverage_gate')}, sort_keys=True))


if __name__ == '__main__':
    main()
