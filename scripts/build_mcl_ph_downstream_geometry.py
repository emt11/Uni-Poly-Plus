#!/usr/bin/env python3
"""Assess all unique downstream structures with the unchanged Trimer rule."""

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import json
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset.cache_lifecycle import StagingWriter
from src.dataset.glt_dual import build_dual_sample
from src.dataset.glt_dual_static import DualStaticCache
from src.dataset.lmdb_cache import sample_key_from_normalized
from src.dataset.trimer_mcl import TrimerGeometryRejection
from scripts.pilot_mcl_ph_geometry_store import generate, verify_record, worker_init

SOURCE = ROOT / 'data/processed/glt_dual_v2/downstream/cohort_1545eda5a8f6a1_paper5_outer/records.jsonl'
STATIC = ROOT / 'data/processed/glt_dual_v2/downstream/dual_static_v1'


def unique_rows():
    rows = []
    seen = set()
    with SOURCE.open(encoding='utf-8') as stream:
        for index, line in enumerate(stream):
            row = json.loads(line)
            key_hex = str(row['sample_key'])
            if sample_key_from_normalized(row['normalized_smiles']).hex() != key_hex:
                raise ValueError(f'downstream source row {index}: identity mismatch')
            if key_hex not in seen:
                seen.add(key_hex)
                rows.append((index, row['source_smiles'], key_hex))
    if len(rows) != 3655:
        raise ValueError(f'downstream unique identity count changed: {len(rows)}')
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=32)
    args = parser.parse_args()
    if not 1 <= args.workers <= 64:
        raise ValueError('downstream assessment workers must be in [1,64]')
    output = args.output.resolve()
    if output.exists() and not (output / 'metadata.json').is_file():
        raise FileExistsError(f'nonresumable output exists: {output}')
    rows = unique_rows()
    metadata = {'schema': 'mcl-ph-next-downstream-trimer-assessment-v1',
                'source': str(SOURCE), 'static': str(STATIC),
                'source_rows': 6265, 'unique_structures': len(rows),
                'generator': 'TRIMER_BUILD_SPEC', 'selection': 'first row per sample_key'}
    writer = StagingWriter(output, metadata, map_size=64 * 1024 ** 2)
    cache = None
    started = time.perf_counter()
    inserted = resumed = verified = 0
    failures = []
    try:
        cache = DualStaticCache(STATIC)
        pending_rows = iter(rows)
        pending = {}
        with ProcessPoolExecutor(max_workers=args.workers, initializer=worker_init) as pool:
            for _ in range(2 * args.workers):
                row = next(pending_rows, None)
                if row is None:
                    break
                pending[pool.submit(generate, row)] = row
            while pending:
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    source_row = pending.pop(future)
                    index, smiles, key_hex = source_row
                    try:
                        row, topology, trimer, duration = future.result()
                    except TrimerGeometryRejection as error:
                        failure = {'source_index': index, 'sample_key': key_hex,
                                   'status': 'GEOMETRY_REJECTION', 'reason': str(error)}
                        failures.append(failure)
                        print(json.dumps(failure), flush=True)
                    else:
                        key = bytes.fromhex(key_hex)
                        if key in writer:
                            verify_record(trimer, writer.get(key), key_hex)
                            resumed += 1
                        else:
                            did_insert, _, _ = writer.put(key, trimer)
                            if not did_insert:
                                raise RuntimeError(f'{key_hex}: duplicate writer insert')
                            verify_record(trimer, writer.get(key), key_hex)
                            inserted += 1
                        static = cache.get_by_key(key)
                        dual = build_dual_sample(topology, trimer, smiles, static=static)
                        if not bool(dual.geometry_valid) or not torch.isfinite(dual.bond_distance).all():
                            raise ValueError(f'{key_hex}: static geometry interface invalid')
                        verified += 1
                        print(json.dumps({'source_index': index, 'status': 'PASS',
                                          'generated_seconds': duration,
                                          'atoms': int(trimer.trimer_pos.size(0))}), flush=True)
                        if inserted % 256 == 0:
                            writer.sync()
                    next_row = next(pending_rows, None)
                    if next_row is not None:
                        pending[pool.submit(generate, next_row)] = next_row
        writer.sync()
        summary = {'source_rows': 6265, 'unique_structures': len(rows),
                   'inserted': inserted, 'resumed': resumed, 'verified': verified,
                   'geometry_rejections': len(failures), 'failures': failures,
                   'record_count': writer.count(),
                   'lmdb_disk_bytes': (output / 'data.lmdb' / 'data.mdb').stat().st_size,
                   'wall_seconds': time.perf_counter() - started,
                   'workers': args.workers}
        (output / 'assessment_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        print(json.dumps({'summary': summary}), flush=True)
    finally:
        if cache is not None:
            cache.close()
        writer.close()


if __name__ == '__main__':
    main()
