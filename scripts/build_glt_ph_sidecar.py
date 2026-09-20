#!/usr/bin/env python3
"""Build the PH sidecar for the whole P_train manifest (GLT-GALPH P0).

Layout (schema glt-ph-h0h1-v1):
  sample_keys.npy  [N,32] uint8      sample key bytes, P_train manifest order
  ph_profile.npy   [N,3,32] float32
  ph_valid.npy     [N] bool
  metadata.json                       counts, schema, runtime, source identity
  .done                               written last; absence means incomplete

Workers process contiguous blocks in source-index order (the chunk cache makes
sequential reads cheap) and merge into the final arrays in the parent.  The
sidecar is read-only during training with ``mmap_mode='r'``.
"""
import argparse
import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np

from src.dataset.glt_ph import (PH_BINS, PH_CHANNELS, PH_RADIUS_MAX,
                               PH_RADIUS_MIN, PH_SCHEMA, ph_profile)
from src.training.glt_dual_runtime import (IndexedFrozenDualSource,
                                           load_sample_index_artifact,
                                           open_source, OrderedSampleStream)


def _worker_init(cohort_root, cache_root, dual_static_root, indices):
    global _WORKER_SOURCE, _WORKER_STREAM
    base, _ = open_source(cohort_root, cache_root,
                          dual_static_root=dual_static_root,
                          chunk_cache_capacity=64)
    _WORKER_SOURCE = IndexedFrozenDualSource(base, indices)
    _WORKER_STREAM = OrderedSampleStream(len(_WORKER_SOURCE), 42)


_WORKER_STREAM = None


def _worker_block(payload):
    block_id, positions = payload
    profiles = np.zeros((len(positions), PH_CHANNELS, PH_BINS), dtype=np.float32)
    valid = np.zeros((len(positions),), dtype=bool)
    keys = np.zeros((len(positions), 32), dtype=np.uint8)
    started = time.perf_counter()
    for offset, position in enumerate(positions):
        index = int(_WORKER_STREAM.index_at(int(position)))
        key, _ = _WORKER_SOURCE.samples[index]
        trimer = _WORKER_SOURCE[index][1]
        keys[offset] = np.frombuffer(key, dtype=np.uint8)[:32]
        profile, ok = ph_profile(np.asarray(trimer.trimer_pos, dtype=np.float64),
                                 np.asarray(trimer.trimer_atomic_number, dtype=np.int64))
        profiles[offset] = profile
        valid[offset] = ok
    return {'block': int(block_id), 'profiles': profiles, 'valid': valid,
            'keys': keys, 'seconds': time.perf_counter() - started}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pi1m-cohort-root', required=True)
    parser.add_argument('--cache-root', required=True)
    parser.add_argument('--dual-static-root', required=True)
    parser.add_argument('--split-artifact', required=True)
    parser.add_argument('--output-root', required=True)
    parser.add_argument('--workers', type=int, default=48)
    parser.add_argument('--blocks', type=int, default=96)
    parser.add_argument('--limit', type=int, default=0,
                        help='debug: only the first N manifest positions')
    args = parser.parse_args()

    artifact = load_sample_index_artifact(args.split_artifact, 'train')
    total = int(len(artifact['indices']))
    if int(args.limit) > 0:
        total = min(total, int(args.limit))
    output = Path(args.output_root)
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    block_size = max(1, (total + int(args.blocks) - 1) // int(args.blocks))
    blocks = [(block_id, list(range(start, min(start + block_size, total))))
              for block_id, start in enumerate(range(0, total, block_size))]
    profiles = np.zeros((total, PH_CHANNELS, PH_BINS), dtype=np.float32)
    valid = np.zeros((total,), dtype=bool)
    keys = np.zeros((total, 32), dtype=np.uint8)
    base, _ = open_source(args.pi1m_cohort_root, args.cache_root,
                          dual_static_root=args.dual_static_root,
                          chunk_cache_capacity=64)
    del base  # workers open their own handle; parent does not read data here
    with mp.Pool(processes=int(args.workers), initializer=_worker_init,
                 initargs=(args.pi1m_cohort_root, args.cache_root,
                           args.dual_static_root, artifact['indices'])) as pool:
        done = 0
        for result in pool.imap_unordered(_worker_block, blocks):
            positions = blocks[result['block']][1]
            profiles[positions] = result['profiles']
            valid[positions] = result['valid']
            keys[positions] = result['keys']
            done += 1
            if done % 8 == 0 or done == len(blocks):
                print(json.dumps({'blocks': done, 'total_blocks': len(blocks),
                                  'records': int(valid.size),
                                  'elapsed_seconds': time.perf_counter() - started}),
                      flush=True)
    np.save(output / 'sample_keys.npy', keys)
    np.save(output / 'ph_profile.npy', profiles)
    np.save(output / 'ph_valid.npy', valid)
    metadata = {
        'schema_version': PH_SCHEMA,
        'records': int(total), 'valid_records': int(valid.sum()),
        'invalid_records': int((~valid).sum()),
        'channels': PH_CHANNELS, 'bins': PH_BINS,
        'radius': [PH_RADIUS_MIN, PH_RADIUS_MAX],
        'channel_semantics': ['heavy_H0', 'heavy_H1', 'all_atom_H0'],
        'source_split_artifact': str(Path(args.split_artifact).resolve()),
        'split_artifact_sha256': artifact['sha256'],
        'workers': int(args.workers), 'blocks': len(blocks),
        'build_seconds': time.perf_counter() - started,
        'bytes': int(keys.nbytes + profiles.nbytes + valid.nbytes),
    }
    (output / 'metadata.json').write_text(
        json.dumps(metadata, indent=1, sort_keys=False) + '\n', encoding='utf-8')
    (output / '.done').write_text('ok\n', encoding='utf-8')
    print(json.dumps(metadata, indent=1))


if __name__ == '__main__':
    main()
