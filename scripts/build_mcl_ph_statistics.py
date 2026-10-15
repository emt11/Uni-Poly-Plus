#!/usr/bin/env python3
"""Fit fixed P_train statistics with bounded memory and one source per worker."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.dataset.mcl_ph_view import prepare_mcl_ph_sample
from src.training.glt_dual_runtime import load_sample_index_artifact, open_source


def moments(values):
    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError('statistics inputs contain NaN/Inf')
    if len(values) == 0:
        return None
    mean = values.mean(axis=0)
    return len(values), mean, ((values - mean) ** 2).sum(axis=0)


def merge(left, right):
    if left is None:
        return right
    if right is None:
        return left
    n, mean, m2 = left
    count, other, other_m2 = right
    delta = other - mean
    total = n + count
    return total, mean + delta * (count / total), m2 + other_m2 + delta**2 * (n * count / total)


def initialize(source_root, cache_root):
    global SOURCE
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    SOURCE, _ = open_source(source_root, cache_root)


def chunk(payload):
    indices, seed, sigma, ratio = payload
    accum = {'router': None, 'length': None, 'nonbond': None}
    valid = []
    for index in indices:
        if SOURCE.entries[int(index)]['pretrain_role'] != 'train':
            raise ValueError(f'statistics attempted a non-train source row: {index}')
        record = SOURCE.record(int(index))
        if not bool(record.static['geometry_valid']):
            continue
        key = SOURCE.samples[int(index)][0].hex()
        sample, labels = prepare_mcl_ph_sample(
            record.topology, record.trimer, SOURCE.entries[int(index)]['source_smiles'],
            seed=seed, key=key, position=int(index), sigma=sigma, ratio=ratio,
            static=record.static, statistics=None)
        accum['router'] = merge(accum['router'], moments(sample.mcl_trajectory.numpy()[None]))
        for family, field in [('length', 'mcl_length_raw'), ('nonbond', 'mcl_nonbond_raw')]:
            values = labels[field].numpy().astype(np.float64)
            if not np.isfinite(values).all() or np.any(values < 0):
                raise ValueError(f'invalid {family} targets at source row {index}')
            accum[family] = merge(accum[family], moments(np.log1p(values)))
        valid.append(index)
    return accum, valid


def build(args):
    if args.output.exists() or args.output.with_suffix('.json').exists():
        raise FileExistsError(args.output)
    split = load_sample_index_artifact(args.sample_index, 'train')
    train = np.asarray(split['indices'], dtype=np.int64)
    if train.ndim != 1 or train.size == 0 or np.any(train[:-1] >= train[1:]):
        raise ValueError('train source indices must be sorted and nonempty')
    if args.limit:
        train = train[:args.limit]
    tasks = [(train[start:start + args.chunk_size].tolist(), args.seed, args.sigma, args.mask_ratio)
             for start in range(0, len(train), args.chunk_size)]
    accum = {'router': None, 'length': None, 'nonbond': None}
    valid = []
    # Fixed waves bound pending futures; each worker opens the source once.
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context('spawn'),
                             initializer=initialize,
                             initargs=(str(args.source_root), str(args.cache_root))) as pool:
        for start in range(0, len(tasks), args.workers * 2):
            futures = [pool.submit(chunk, task) for task in tasks[start:start + args.workers * 2]]
            for future in futures:
                partial, indices = future.result()
                for name in accum:
                    accum[name] = merge(accum[name], partial[name])
                valid.extend(indices)
            print(json.dumps({'chunks_completed': min(start + args.workers * 2, len(tasks)),
                              'chunks_total': len(tasks), 'valid_rows': len(valid)}), flush=True)
    if any(value is None for value in accum.values()):
        raise ValueError('P_train produced no legal finite geometry targets')
    if valid != sorted(set(valid)):
        raise ValueError('worker results changed source order or duplicated rows')
    payload = {}
    for name, (count, mean, m2) in accum.items():
        scale = np.maximum(np.sqrt(np.maximum(m2 / count, 0)), 1e-6)
        if name == 'router':
            payload.update(router_mean=mean.astype(np.float32), router_std=scale.astype(np.float32))
        else:
            payload[name + '_mu'] = np.asarray([mean], dtype=np.float64)
            payload[name + '_sigma'] = np.asarray([scale], dtype=np.float64)
    payload.update(samples=np.asarray([len(valid)], dtype=np.int64),
                   seed=np.asarray([args.seed], dtype=np.int64), sigma=np.asarray([args.sigma]),
                   identity_contract=np.asarray('mcl-ph-train-statistics-direct-v1'),
                   fit_source_indices=train,
                   valid_source_indices=np.asarray(valid, dtype=np.int64))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, **payload)
    report = {'conclusion': 'PASS', 'scope': 'probe' if args.limit else 'full',
              'source_root': str(args.source_root.resolve()),
              'train_source_rows': int(train.size), 'valid_geometry_rows': len(valid),
              'length_targets': accum['length'][0], 'nonbond_targets': accum['nonbond'][0],
              'workers': args.workers, 'chunk_size': args.chunk_size,
              'seed': args.seed, 'sigma': args.sigma, 'mask_ratio': args.mask_ratio,
              'position_rule': 'immutable source_index, as in the fitting recipe',
              'sample_index': str(args.sample_index.resolve()),
              'artifact': str(args.output.resolve()), 'identity_contract': 'direct',
              'aggregation': 'float64 population moments, target-weighted geometry; no validation rows'}
    args.output.with_suffix('.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--cache-root', type=Path, default=Path('.'))
    parser.add_argument('--sample-index', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, choices=range(1, 25), default=8)
    parser.add_argument('--chunk-size', type=int, default=128)
    parser.add_argument('--limit', type=int, default=0, help='probe prefix only; not a formal statistics artifact')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--sigma', type=float, default=0.03)
    parser.add_argument('--mask-ratio', type=float, default=0.30)
    args = parser.parse_args()
    if args.chunk_size <= 0 or args.limit < 0:
        parser.error('invalid chunk size or probe limit')
    build(args)
