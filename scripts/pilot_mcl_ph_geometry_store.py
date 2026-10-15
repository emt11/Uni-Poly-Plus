#!/usr/bin/env python3
"""Bounded, resumable single-writer pilot for rebuilt PI1M Trimer geometry."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset.cache_lifecycle import StagingWriter, select_record_fields
from src.dataset.glt_dual import build_dual_sample
from src.dataset.glt_dual_static import DualStaticCache
from src.dataset.lmdb_cache import sample_key_from_normalized

SOURCE = ROOT / 'data/processed/glt_dual_v2/pi1m/cohort_30f17b59bc5862a1/records.jsonl'
STATIC = ROOT / 'data/processed/glt_dual_v2/pi1m/dual_static_v1'


def worker_init():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)


def selected_rows(count):
    selected = set(map(int, np.random.default_rng(42).choice(959588, size=count, replace=False)))
    rows = []
    with SOURCE.open(encoding='utf-8') as stream:
        for index, line in enumerate(stream):
            if index in selected:
                row = json.loads(line)
                if sample_key_from_normalized(row['normalized_smiles']).hex() != row['sample_key']:
                    raise ValueError(f'PI1M row {index} identity mismatch')
                rows.append((index, row['source_smiles'], row['sample_key']))
    if len(rows) != count:
        raise ValueError('PI1M source row count differs from pilot contract')
    return rows


def generate(row):
    from src.dataset.canonical_periodic import build_canonical_periodic_topology
    from src.dataset.trimer_mcl import attach_finite_trimer_mcl

    index, smiles, key = row
    start = time.perf_counter()
    topology = build_canonical_periodic_topology(smiles)
    trimer = attach_finite_trimer_mcl(topology, smiles, sample_key=key)
    if not bool(trimer.trimer_geometry_valid):
        raise ValueError(f'row {index}: generator returned invalid geometry')
    return row, topology, select_record_fields('trimer', trimer), time.perf_counter() - start


def verify_record(expected, actual, key):
    if set(expected.keys()) != set(actual.keys()):
        raise ValueError(f'{key}: stored geometry field set changed')
    for name in expected.keys():
        left, right = expected[name], actual[name]
        if isinstance(left, torch.Tensor):
            if not torch.equal(left, right):
                raise ValueError(f'{key}: stored {name} differs')
        elif left != right:
            raise ValueError(f'{key}: stored {name} differs')
    if not torch.isfinite(actual.trimer_pos).all():
        raise ValueError(f'{key}: stored coordinates are nonfinite')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--samples', type=int, default=256)
    parser.add_argument('--workers', type=int, default=16)
    args = parser.parse_args()
    if not 1 <= args.samples <= 256 or not 1 <= args.workers <= 16:
        raise ValueError('pilot limit is 256 samples and 16 workers')
    output = args.output.resolve()
    if output.exists() and not (output / 'metadata.json').is_file():
        raise FileExistsError(f'pilot output exists without resumable metadata: {output}')
    rows = selected_rows(args.samples)
    metadata = {'schema': 'mcl-ph-next-trimer-pilot-v1', 'selection': 'PI1M rng42',
                'source': str(SOURCE), 'rows': [{'source_index': i, 'sample_key': k}
                                               for i, _, k in rows]}
    started = time.perf_counter()
    inserted = resumed = bytes_written = static_passes = 0
    sizes = []
    writer = StagingWriter(output, metadata, map_size=64 * 1024 ** 2)
    cache = None
    try:
        cache = DualStaticCache(STATIC)
        with ProcessPoolExecutor(max_workers=args.workers, initializer=worker_init) as pool:
            for row, topology, trimer, duration in pool.map(generate, rows):
                index, smiles, key_hex = row
                key = bytes.fromhex(key_hex)
                if key in writer:
                    verify_record(trimer, writer.get(key), key_hex)
                    resumed += 1
                    size = 0
                else:
                    did_insert, size, _ = writer.put(key, trimer)
                    if not did_insert:
                        raise RuntimeError(f'{key_hex}: duplicate writer insert')
                    verify_record(trimer, writer.get(key), key_hex)
                    inserted += 1
                    bytes_written += size
                static = cache.get_by_key(key)
                dual = build_dual_sample(topology, trimer, smiles, static=static)
                if not bool(dual.geometry_valid) or not torch.isfinite(dual.bond_distance).all():
                    raise ValueError(f'{key_hex}: static geometry interface invalid')
                static_passes += 1
                sizes.append(size)
                print(json.dumps({'source_index': index, 'status': 'PASS',
                                  'generated_seconds': duration, 'payload_bytes': size,
                                  'atoms': int(trimer.trimer_pos.size(0))}), flush=True)
        writer.sync()
        disk = (output / 'data.lmdb' / 'data.mdb').stat().st_size
        elapsed = time.perf_counter() - started
        summary = {'samples': args.samples, 'workers': args.workers,
                   'inserted': inserted, 'resumed': resumed,
                   'static_interface_passes': static_passes,
                   'record_count': writer.count(), 'payload_bytes': bytes_written,
                   'lmdb_disk_bytes': disk, 'wall_seconds': elapsed,
                   'throughput_per_second': args.samples / elapsed,
                   'payload_bytes_per_inserted': bytes_written / inserted if inserted else 0,
                   'selection': 'PI1M rng42; same rows as compute-only probe'}
        (output / 'pilot_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        print(json.dumps({'summary': summary}), flush=True)
    finally:
        if cache is not None:
            cache.close()
        writer.close()


if __name__ == '__main__':
    main()
