#!/usr/bin/env python3
"""Compare a small offline F2/noisy trajectory cache to the online path."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.dataset.mcl_ph_view import prepare_mcl_ph_sample
from src.dataset.mcl_ph_trajectory_cache import (MCLPHTrajectoryCache, require_direct_source_rows,
                                                cache_identity, direct_source_identity)
from src.training.glt_dual_runtime import IndexedFrozenDualSource, OrderedSampleStream, load_sample_index_artifact, open_source


def audit(args):
    if args.report.exists():
        raise FileExistsError(args.report)
    split = load_sample_index_artifact(args.sample_index, 'train')
    base, _ = open_source(args.source_root, args.cache_root)
    source = IndexedFrozenDualSource(base, split['indices'])
    cache = MCLPHTrajectoryCache(args.cache, verify_checksums=True)
    if cache.direct:
        cache.require_identity(cache_identity(seed=args.seed, noise_sigma=args.sigma,
            mask_ratio=args.mask_ratio, global_batch=args.global_batch,
            sample_index_split='train', direct_source=direct_source_identity(source)))
        require_direct_source_rows(args.cache, source)
    count = int(args.positions)
    cache.require_positions(count)
    structural_shards = 0
    if args.full_structure:
        if cache.manifest.get('complete') is not True:
            raise ValueError('formal cache manifest is not complete=true')
        cache.require_positions(len(cache))
        for first, rows in cache._bounds:
            cache[first]
            structural_shards += 1
    stream = OrderedSampleStream(len(source), int(args.seed))
    checked = 0
    try:
        for position in range(count):
            index = stream.index_at(position)
            topology, trimer, smiles = source[index]
            key = source.samples[index][0].hex()
            data, _ = prepare_mcl_ph_sample(
                topology, trimer, smiles, seed=int(args.seed), key=key,
                position=position, sigma=float(args.sigma), ratio=float(args.mask_ratio),
                static=source.static_for(index), statistics=None)
            expected = np.asarray(data.mcl_trajectory.numpy(), dtype=np.float32)
            actual = np.asarray(cache[position], dtype=np.float32)
            if expected.shape != actual.shape or not np.array_equal(expected, actual):
                raise ValueError(f'cache trajectory differs at position {position}')
            checked += 1
    finally:
        cache.close(); source.close()
    result = {'conclusion': 'PASS', 'positions': checked,
              'cache_root': str(args.cache.resolve()),
              'total_positions': len(cache),
              'manifest_complete': cache.manifest.get('complete'),
              'structural_finite_shards': structural_shards,
              'seed': int(args.seed), 'sigma': float(args.sigma),
              'mask_ratio': float(args.mask_ratio), 'f2': 'gudhi coefficient field 2',
              'comparison': 'online prepare_mcl_ph_sample exact float32 trajectory'}
    args.report.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--cache-root', type=Path, default=Path('.'))
    parser.add_argument('--sample-index', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--positions', type=int, default=32)
    parser.add_argument('--full-structure', action='store_true')
    parser.add_argument('--global-batch', type=int, default=1008)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--sigma', type=float, default=0.03)
    parser.add_argument('--mask-ratio', type=float, default=0.30)
    audit(parser.parse_args())
