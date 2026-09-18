#!/usr/bin/env python3
"""Scan the full P_train stream for ALIGN pool validity metadata only."""
import argparse
import json
import multiprocessing as mp
import time
from collections import Counter
from pathlib import Path

import numpy as np

from src.training.glt_dual_runtime import load_sample_index_artifact, open_source, OrderedSampleStream


_WORKER_SOURCE = None
_WORKER_INDICES = None
_WORKER_STREAM = None


def _worker_init(cohort_root, cache_root, dual_static_root, indices):
    """Each worker opens its own read-only view and replays the same stream."""

    global _WORKER_SOURCE, _WORKER_INDICES, _WORKER_STREAM
    _WORKER_SOURCE, _ = open_source(cohort_root, cache_root, dual_static_root=dual_static_root)
    _WORKER_INDICES = np.asarray(indices, dtype=np.int64)
    _WORKER_STREAM = OrderedSampleStream(_WORKER_INDICES.size, 42)


def _scan_microsteps(source, indices, stream, start, end):
    """Per-microstep pool composition for positions [start, end) of the stream."""

    records = source.cohort['records']
    rows = []
    geometry_invalid_samples = 0
    no_center_samples = 0
    for offset in range(start, end, 252):
        identities = []
        keep_flags = []
        for position in range(offset, min(offset + 252, indices.size)):
            logical_index = stream.index_at(position)
            source_index = int(indices[logical_index])
            row = records[source_index]
            static = source.static_for(source_index)
            geometry = bool(static['geometry_valid'])
            centers = np.asarray(static['token_center_mask'], dtype=bool).reshape(-1)
            if not geometry:
                geometry_invalid_samples += 1
            elif not bool(centers.any()):
                no_center_samples += 1
            identities.append(str(row['normalized_smiles']))
            keep_flags.append(bool(geometry and centers.any()))
        valid_ids = [identity for identity, keep in zip(identities, keep_flags) if keep]
        counts = Counter(valid_ids)
        pair_count = len(valid_ids)
        rows.append((pair_count, len(counts),
                     sum(pair_count - count for count in counts.values())))
    return rows, geometry_invalid_samples, no_center_samples


def _worker_chunk(payload):
    ordinal, start, end = payload
    rows, geometry_invalid, no_center = _scan_microsteps(
        _WORKER_SOURCE, _WORKER_INDICES, _WORKER_STREAM, start, end)
    return {'ordinal': int(ordinal), 'rows': rows,
            'geometry_invalid': int(geometry_invalid), 'no_center': int(no_center)}


def _summary(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {'mean': 0.0, 'p50': 0.0, 'p95': 0.0, 'min': 0, 'max': 0}
    return {'mean': float(values.mean()),
            'p50': float(np.percentile(values, 50, method='nearest')),
            'p95': float(np.percentile(values, 95, method='nearest')),
            'min': int(values.min()), 'max': int(values.max())}


def build(args):
    artifact = load_sample_index_artifact(args.split_artifact, 'train')
    source, _ = open_source(args.pi1m_cohort_root, args.cache_root,
                            dual_static_root=args.dual_static_root)
    indices = artifact['indices']
    if len(source) != int(artifact['payload']['pi1m_record_count']):
        raise ValueError('ALIGN diagnostics base cohort count mismatch')
    stream = OrderedSampleStream(len(indices), 42)
    valid_pairs = []
    distinct_valid = []
    negative_counts = []
    no_negative = 0
    all_invalid = 0
    geometry_invalid_samples = 0
    no_center_samples = 0
    key_to_identity = {}
    identity_to_key = {}
    microstep_count = 0
    started = time.perf_counter()
    try:
        records = source.cohort['records']
        for source_index in indices.tolist():
            row = records[int(source_index)]
            key = str(row['sample_key'])
            identity = str(row['normalized_smiles'])
            if key_to_identity.setdefault(key, identity) != identity:
                raise ValueError('sample_key maps to multiple normalized identities')
            if identity_to_key.setdefault(identity, key) != key:
                raise ValueError('normalized identity maps to multiple sample keys')
        if len(key_to_identity) != len(identity_to_key) or len(identity_to_key) != len(indices):
            raise ValueError('P_train sample_key/normalized_identity is not bijective')
        spans = [(ordinal, start, min(start + 252 * int(args.chunk_microsteps), len(indices)))
                 for ordinal, start in enumerate(
                     range(0, len(indices), 252 * int(args.chunk_microsteps)))]
        chunk_rows = []
        if int(args.workers) == 1:
            for ordinal, start, end in spans:
                rows, geometry_invalid, no_center = _scan_microsteps(
                    source, indices, stream, start, end)
                chunk_rows.append((ordinal, rows, geometry_invalid, no_center))
                if (ordinal + 1) % int(args.progress_every) == 0:
                    print(json.dumps({'chunks': ordinal + 1, 'total_chunks': len(spans),
                                      'processed': min(end, len(indices)),
                                      'total': len(indices),
                                      'elapsed_seconds': time.perf_counter() - started}),
                          flush=True)
        else:
            source.close()
            source = None
            with mp.Pool(processes=int(args.workers), initializer=_worker_init,
                         initargs=(args.pi1m_cohort_root, args.cache_root,
                                   args.dual_static_root, indices)) as pool:
                for ordinal, result in enumerate(pool.imap(_worker_chunk, spans), 1):
                    chunk_rows.append((result['ordinal'], result['rows'],
                                       result['geometry_invalid'], result['no_center']))
                    if ordinal % int(args.progress_every) == 0:
                        print(json.dumps({'chunks': ordinal, 'total_chunks': len(spans),
                                          'processed': min(ordinal * 252 * int(args.chunk_microsteps),
                                                           len(indices)),
                                          'total': len(indices),
                                          'elapsed_seconds': time.perf_counter() - started}),
                              flush=True)
        for ordinal, rows, geometry_invalid, no_center in sorted(chunk_rows):
            for pair_count, distinct_count, negatives in rows:
                valid_pairs.append(pair_count)
                distinct_valid.append(distinct_count)
                negative_counts.append(negatives)
                no_negative += int(negatives == 0)
                all_invalid += int(pair_count == 0)
                microstep_count += 1
            geometry_invalid_samples += int(geometry_invalid)
            no_center_samples += int(no_center)
    finally:
        if source is not None:
            source.close()
    payload = {
        'seed': 42,
        'world_size': 3,
        'microbatch': 84,
        'distributed_microbatch': 252,
        'p_train_count': int(len(indices)),
        'microstep_count': int(microstep_count),
        'valid_pair_count': _summary(valid_pairs),
        'distinct_valid_identity': _summary(distinct_valid),
        'negative_count': _summary(negative_counts),
        'no_negative_microstep_count': int(no_negative),
        'no_negative_microstep_fraction': float(no_negative / max(1, microstep_count)),
        'all_invalid_microstep_count': int(all_invalid),
        'geometry_invalid_sample_count': int(geometry_invalid_samples),
        'no_center_bond_sample_count': int(no_center_samples),
        'sample_key_identity_bijection': True,
        'identity_source': 'cohort.normalized_smiles',
        'elapsed_seconds': float(time.perf_counter() - started),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2,
                                 sort_keys=True, allow_nan=False) + '\n', encoding='utf-8')
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pi1m-cohort-root', required=True)
    parser.add_argument('--cache-root', required=True)
    parser.add_argument('--dual-static-root', required=True)
    parser.add_argument('--split-artifact', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--progress-every', type=int, default=100)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--chunk-microsteps', type=int, default=64)
    args = parser.parse_args()
    if args.progress_every <= 0:
        raise ValueError('--progress-every must be positive')
    if args.workers <= 0 or args.chunk_microsteps <= 0:
        raise ValueError('--workers and --chunk-microsteps must be positive')
    print(json.dumps(build(args), ensure_ascii=False, indent=2, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
