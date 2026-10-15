#!/usr/bin/env python3
"""Benchmark PI1M publication compute plus one temporary LMDB writer.

Only already-frozen geometry rows are read; no conformer generation occurs.
Each worker-count trial writes an isolated temporary staging root.
"""
import argparse
import json
import multiprocessing as mp
from pathlib import Path
import shutil
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import publish_mcl_ph_pi1m_direct as publisher
from src.dataset.cache_lifecycle import StagingWriter


def trial(geometry, items, workers, root):
    started = time.perf_counter()
    publisher.initialize(geometry)
    writer = StagingWriter(root, {'schema': 'mcl-ph-publish-benchmark-v1',
                                  'coordinate_generation': False,
                                  'workers': int(workers)})
    counts = {}
    pool = mp.get_context('spawn').Pool(int(workers), initializer=publisher.initialize,
                                        initargs=(geometry,))
    try:
        for index, key, entry, error in pool.imap(publisher.materialize, items, chunksize=4):
            if error:
                raise RuntimeError(f'{index}/{key.hex()}: {error}')
            inserted, _, _ = writer.put(key, entry)
            if not inserted:
                raise RuntimeError('duplicate benchmark identity')
            counts[entry.final_geometry_status] = counts.get(entry.final_geometry_status, 0) + 1
        pool.close(); pool.join(); pool = None
    finally:
        if pool is not None:
            pool.terminate(); pool.join()
        writer.close()
        publisher.SOURCE_ENV.close(); publisher.SOURCE_DB.close()
    elapsed = time.perf_counter() - started
    return {'workers': int(workers), 'rows': len(items), 'seconds': elapsed,
            'rows_per_second': len(items) / max(elapsed, 1e-9), 'status': counts,
            'root': str(root)}


def main(args):
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    indices = list(publisher.source_rows('pi1m', limit=int(args.rows)))
    if len(indices) != int(args.rows):
        raise ValueError('benchmark probe did not resolve requested rows')
    trials = []
    for workers in args.workers:
        root = Path(tempfile.mkdtemp(prefix=f'mcl_pi1m_pub_{workers}_',
                                     suffix='.staging', dir=str(args.temp_root.resolve())))
        trials.append(trial(args.geometry.resolve(), indices, workers, root))
    result = {'conclusion': 'PASS', 'geometry': str(args.geometry.resolve()),
              'rows': int(args.rows), 'coordinate_generation': False,
              'trials': trials}
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--geometry', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--temp-root', type=Path, default=Path('/tmp'))
    parser.add_argument('--rows', type=int, default=256)
    parser.add_argument('--workers', type=int, nargs='+', default=[24, 48, 56])
    main(parser.parse_args())
