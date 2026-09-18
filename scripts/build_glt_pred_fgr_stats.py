#!/usr/bin/env python3
"""Fit the full-P_train FGR log-distance normalization statistics."""
import argparse
import hashlib
import json
import math
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np

from src.dataset.canonical_periodic import resolve_normalized_identity
from src.dataset.glt_dual_pretrain import fgr_true_trimer_candidates
from src.training.glt_dual_runtime import load_sample_index_artifact, open_source


_WORKER_SOURCE = None


def _worker_init(cohort_root, cache_root):
    global _WORKER_SOURCE
    _WORKER_SOURCE, _ = open_source(cohort_root, cache_root)


def _worker_chunk(payload):
    ordinal, source_indices = payload
    count = graph_count = geometry_valid_count = graphs_with_candidate = 0
    spd2_count = spd3_count = 0
    mean = 0.0
    m2 = 0.0
    minimum = math.inf
    maximum = -math.inf
    for source_index in source_indices:
        topology, trimer, smiles = _WORKER_SOURCE[int(source_index)]
        graph_count += 1
        if bool(getattr(trimer, 'trimer_geometry_valid', False)):
            geometry_valid_count += 1
        identity = resolve_normalized_identity(topology, smiles, require_fields=True)
        _, distances, spd = fgr_true_trimer_candidates(
            topology, trimer, identity=identity)
        if int(distances.numel()):
            graphs_with_candidate += 1
            spd2_count += int((spd == 2).sum())
            spd3_count += int((spd == 3).sum())
            for value in np.log1p(distances.detach().cpu().numpy().astype(np.float64)).tolist():
                value = float(value)
                count += 1
                delta = value - mean
                mean += delta / count
                m2 += delta * (value - mean)
                minimum = min(minimum, value)
                maximum = max(maximum, value)
    return {'ordinal': int(ordinal), 'count': count, 'graph_count': graph_count,
            'geometry_valid_count': geometry_valid_count,
            'graphs_with_candidate': graphs_with_candidate, 'spd2_count': spd2_count,
            'spd3_count': spd3_count, 'mean': mean, 'm2': m2,
            'min': minimum, 'max': maximum}


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def build(args):
    artifact = load_sample_index_artifact(args.split_artifact, 'train')
    started = time.perf_counter()
    indices = artifact['indices'].tolist()
    if args.limit:
        indices = indices[:int(args.limit)]
    chunks = [(ordinal, indices[start:start + int(args.chunk_size)])
              for ordinal, start in enumerate(range(0, len(indices), int(args.chunk_size)))]
    if int(args.workers) == 1:
        _worker_init(args.pi1m_cohort_root, args.cache_root)
        results = []
        for ordinal, chunk in chunks:
            results.append(_worker_chunk((ordinal, chunk)))
            if (ordinal + 1) % int(args.progress_every) == 0:
                print(json.dumps({'chunks': ordinal + 1, 'total_chunks': len(chunks),
                                  'processed': min((ordinal + 1) * int(args.chunk_size), len(indices)),
                                  'elapsed_seconds': time.perf_counter() - started}), flush=True)
    else:
        # ``imap`` preserves chunk order.  Each worker opens the frozen bundle
        # read-only and returns FP64 Welford state; the parent combines states
        # in that fixed order, so worker scheduling cannot alter the result.
        with mp.Pool(processes=int(args.workers), initializer=_worker_init,
                     initargs=(args.pi1m_cohort_root, args.cache_root)) as pool:
            results = []
            for ordinal, result in enumerate(pool.imap(_worker_chunk, chunks), 1):
                results.append(result)
                if ordinal % int(args.progress_every) == 0:
                    print(json.dumps({'chunks': ordinal, 'total_chunks': len(chunks),
                                      'processed': min(ordinal * int(args.chunk_size), len(indices)),
                                      'elapsed_seconds': time.perf_counter() - started}), flush=True)
    count = graph_count = geometry_valid_count = graphs_with_candidate = 0
    spd2_count = spd3_count = 0
    mean = 0.0
    m2 = 0.0
    minimum = math.inf
    maximum = -math.inf
    for result in results:
        chunk_count = int(result['count'])
        if chunk_count:
            if count == 0:
                mean, m2 = float(result['mean']), float(result['m2'])
            else:
                delta = float(result['mean']) - mean
                total = count + chunk_count
                m2 += float(result['m2']) + delta * delta * count * chunk_count / total
                mean += delta * chunk_count / total
            count += chunk_count
            minimum = min(minimum, float(result['min']))
            maximum = max(maximum, float(result['max']))
        graph_count += int(result['graph_count'])
        geometry_valid_count += int(result['geometry_valid_count'])
        graphs_with_candidate += int(result['graphs_with_candidate'])
        spd2_count += int(result['spd2_count'])
        spd3_count += int(result['spd3_count'])
    if count == 0:
        raise ValueError('P_train contains no valid FGR candidates')
    raw_sigma = math.sqrt(max(0.0, m2 / count))
    sigma = max(raw_sigma, 1e-8)
    payload = {
        'graphs_scanned': int(graph_count),
        'graphs_with_candidates': int(graphs_with_candidate),
        'candidate_count': int(count),
        'spd2_count': int(spd2_count),
        'spd3_count': int(spd3_count),
        'mu': float(mean),
        'sigma': float(sigma),
        'full_p_train_scan': not bool(args.limit),
        'split_artifact_sha256': artifact['sha256'],
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
    parser.add_argument('--split-artifact', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--progress-every', type=int, default=10000)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--chunk-size', type=int, default=1000)
    parser.add_argument('--limit', type=int, default=0,
                        help='diagnostic-only prefix limit; omit for the required full scan')
    args = parser.parse_args()
    if args.progress_every <= 0:
        raise ValueError('--progress-every must be positive')
    if args.workers <= 0 or args.chunk_size <= 0:
        raise ValueError('--workers and --chunk-size must be positive')
    if args.limit < 0:
        raise ValueError('--limit must be non-negative')
    print(json.dumps(build(args), ensure_ascii=False, indent=2, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
