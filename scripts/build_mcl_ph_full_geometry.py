#!/usr/bin/env python3
"""Build the new PI1M or downstream Trimer staging cache, with a durable row ledger.

Ordinary conformer rejection is recorded per identity and does not stop the run.
The main lane uses the fixed worker count; only TIMEOUT is retried after the
main pool has exited, in a bounded CPU pool. The source and index
contract is checked by row content, without trusting a filename or digest.
"""

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import json
import os
from pathlib import Path
import sqlite3
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset.cache_lifecycle import StagingWriter, atomic_json, select_record_fields
from src.dataset.cache_spec import TRIMER_BUILD_SPEC
from src.dataset.glt_dual import build_dual_sample
from src.dataset.glt_dual_static import DualStaticCache
from src.dataset.lmdb_cache import sample_key_from_normalized
from src.dataset.trimer_mcl import TrimerGeometryRejection
from scripts.pilot_mcl_ph_geometry_store import verify_record, worker_init

INDEX = ROOT / 'results/mcl_ph_next_20260929/p0/direct_indices_v1'
PI1M = ROOT / 'data/processed/glt_dual_v2/pi1m/cohort_30f17b59bc5862a1/records.jsonl'
DOWNSTREAM = ROOT / 'data/processed/glt_dual_v2/downstream/cohort_1545eda5a8f6a1_paper5_outer/records.jsonl'
STATIC = ROOT / 'data/processed/glt_dual_v2'
EXPECTED = {'pi1m': 959359, 'downstream': 3655}


def source_rows(kind, limit=0):
    """Yield (source index, original SMILES, sample key); validate every selected row."""
    if kind == 'pi1m':
        indices = np.load(INDEX / 'pretrain_indices.npz')
        train = indices['train_source_indices']
        validation = indices['validation_source_indices']
        fixed = indices['fixed_validation_source_indices']
        if (len(train), len(validation), len(fixed)) != (911391, 47968, 1024):
            raise ValueError('PI1M index counts changed')
        if np.intersect1d(train, validation).size or not np.isin(fixed, validation).all():
            raise ValueError('PI1M split is not disjoint or fixed validation changed')
        for name, values in (('train', train), ('validation', validation), ('fixed', fixed)):
            if (values.min() < 0 or values.max() >= 959588 or
                    np.any(values[:-1] >= values[1:])):
                raise ValueError(f'PI1M {name} indices are not sorted unique valid rows')
        with (INDEX / 'downstream_rows.jsonl').open(encoding='utf-8') as handle:
            excluded = {json.loads(line)['normalized_smiles'] for line in handle}
        if len(excluded) != 3655:
            raise ValueError('downstream exclusion identity count changed')
        selected = np.zeros(959588, dtype=bool)
        selected[train] = True
        selected[validation] = True
        source, total = PI1M, 959588
    else:
        source, total = DOWNSTREAM, 6265
        direct = []
        with (INDEX / 'downstream_rows.jsonl').open(encoding='utf-8') as handle:
            direct = [json.loads(line) for line in handle]
        if len(direct) != total:
            raise ValueError('downstream direct index row count changed')
    seen = set()
    yielded = 0
    with source.open(encoding='utf-8') as handle:
        for index, line in enumerate(handle):
            if index >= total:
                raise ValueError(f'{kind}: source has extra rows')
            row = json.loads(line)
            normalized = row['normalized_smiles']
            key = row['sample_key']
            if sample_key_from_normalized(normalized).hex() != key:
                raise ValueError(f'{kind} row {index}: sample key differs from structure')
            if kind == 'downstream':
                expected = direct[index]
                if (expected['source_index'] != index or
                        expected['normalized_smiles'] != normalized or
                        expected['task'] != row['task'] or
                        expected['original_row'] != row['original_row']):
                    raise ValueError(f'downstream row {index}: direct index mismatch')
                if key in seen:
                    continue
                seen.add(key)
            elif not selected[index]:
                if normalized not in excluded:
                    raise ValueError(f'PI1M row {index}: eligible identity excluded by index')
                continue
            elif normalized in excluded:
                raise ValueError(f'PI1M row {index}: downstream identity entered pretraining')
            yielded += 1
            if not limit or yielded <= limit:
                yield index, row['source_smiles'], key
        if index + 1 != total:
            raise ValueError(f'{kind}: source ended at {index + 1}, expected {total}')
    if yielded != EXPECTED[kind]:
        raise ValueError(f'{kind}: selected {yielded}, expected {EXPECTED[kind]}')
    if kind == 'downstream' and len(seen) != EXPECTED[kind]:
        raise ValueError('downstream unique identity count changed')


def compute(row):
    from src.dataset.canonical_periodic import build_canonical_periodic_topology
    from src.dataset.trimer_mcl import attach_finite_trimer_mcl
    index, smiles, key = row
    started = time.perf_counter()
    topology = None
    try:
        topology = build_canonical_periodic_topology(smiles)
        trimer = attach_finite_trimer_mcl(topology, smiles, sample_key=key)
    except TrimerGeometryRejection as exc:
        return {'row': row, 'status': exc.code, 'seconds': exc.elapsed_seconds,
                'failure': {**exc.ledger_entry(key, str(getattr(topology, 'normalized_canonical_smiles', ''))),
                            'source_smiles': smiles}, 'topology': None, 'trimer': None}
    except Exception as exc:
        return {'row': row, 'status': 'GENERATOR_ERROR',
                'seconds': time.perf_counter() - started,
                'failure': {'error_type': type(exc).__name__, 'reason': str(exc)[:2000],
                            'source_smiles': smiles}, 'topology': None, 'trimer': None}
    seconds = time.perf_counter() - started
    if not bool(trimer.trimer_geometry_valid):
        raise ValueError(f'{index}: generator returned invalid geometry without rejection')
    qc = {'candidate_id': int(trimer.trimer_conformer_candidate_id),
          'round_id': int(trimer.trimer_conformer_round_id),
          'selected_converged': bool(trimer.selected_converged),
          'energy': float(trimer.trimer_conformer_energy),
          'seed': int(trimer.trimer_conformer_seed),
          'atoms': int(trimer.trimer_pos.size(0)),
          'explicit_h_atoms': int(torch.as_tensor(trimer.trimer_atomic_number).eq(1).sum()),
          'diagnostics': trimer.generation_diagnostics}
    if not np.isfinite(qc['energy']):
        raise ValueError(f'{row[2]}: selected conformer energy is nonfinite')
    return {'row': row, 'status': 'PASS', 'seconds': seconds,
            'qc': qc, 'topology': topology, 'trimer': trimer}


def ledger_connect(path):
    connection = sqlite3.connect(path)
    connection.execute('PRAGMA journal_mode=WAL')
    connection.execute('PRAGMA synchronous=FULL')
    connection.execute('''CREATE TABLE IF NOT EXISTS rows (
      source_index INTEGER PRIMARY KEY, sample_key TEXT NOT NULL UNIQUE,
      main_status TEXT NOT NULL, main_seconds REAL NOT NULL,
      main_detail TEXT NOT NULL, retry_status TEXT, retry_seconds REAL,
      retry_detail TEXT, final_status TEXT NOT NULL, coordinate_lane TEXT)''')
    connection.commit()
    return connection


def store_result(result, lane, writer, cache, db, retry_workers=None):
    index, smiles, key_hex = result['row']
    status = result['status']
    detail = result.get('qc', result.get('failure', {}))
    if lane == 'retry':
        if retry_workers is None:
            raise ValueError('retry worker count is required for provenance')
        detail['retry_workers'] = retry_workers
        detail['retry_worker_pid'] = result['worker_pid']
    if status == 'PASS':
        key = bytes.fromhex(key_hex)
        trimer = result['trimer']
        stored = select_record_fields('trimer', trimer)
        if key in writer:
            verify_record(stored, writer.get(key), key_hex)
        else:
            inserted, _, _ = writer.put(key, stored)
            if not inserted:
                raise RuntimeError(f'{key_hex}: duplicate LMDB insert')
            verify_record(stored, writer.get(key), key_hex)
        static = cache.get_by_key(key)
        if bool(static['geometry_valid']):
            dual = build_dual_sample(result['topology'], trimer, smiles, static=static)
            if not bool(dual.geometry_valid) or not torch.isfinite(dual.bond_distance).all():
                raise ValueError(f'{key_hex}: static geometry interface mismatch')
            detail['static_interface'] = 'PASS'
        else:
            # The old static source may reject a structure recovered by the
            # new timeout protocol. Record this for the subsequent rebuild.
            detail['static_interface'] = 'OLD_STATIC_INVALID_REBUILD_REQUIRED'
    payload = json.dumps(detail, separators=(',', ':'), allow_nan=False)
    if lane == 'main':
        db.execute('''INSERT INTO rows VALUES (?, ?, ?, ?, ?, NULL, NULL, NULL, ?, ?)''',
                   (index, key_hex, status, result['seconds'], payload, status,
                    'main' if status == 'PASS' else None))
    else:
        db.execute('''UPDATE rows SET retry_status=?, retry_seconds=?, retry_detail=?,
                      final_status=?, coordinate_lane=?
                      WHERE source_index=? AND sample_key=? AND main_status='TIMEOUT' AND retry_status IS NULL''',
                   (status, result['seconds'], payload, status,
                    'retry' if status == 'PASS' else None, index, key_hex))
        if db.execute('SELECT changes()').fetchone()[0] != 1:
            raise ValueError(f'{key_hex}: retry ledger state changed')
    db.commit()
    print(json.dumps({'source_index': index, 'sample_key': key_hex,
                      'lane': lane, 'status': status, 'seconds': result['seconds'],
                      **({'worker_pid': result['worker_pid'], 'retry_workers': retry_workers}
                         if lane == 'retry' else {})}), flush=True)


def retry_compute(row):
    result = compute(row)
    result['worker_pid'] = os.getpid()
    return result


def build_metadata(args):
    return {'schema': 'mcl-ph-next-full-trimer-v1', 'kind': args.kind,
            'source': str(PI1M if args.kind == 'pi1m' else DOWNSTREAM),
            'direct_index': str(INDEX), 'static': str(STATIC / args.kind / 'dual_static_v1'),
            'source_count': EXPECTED[args.kind], 'limit': args.limit,
            'main_workers': args.workers,
            'retry': f'TIMEOUT only; {args.retry_workers} workers after main',
            'retry_workers': args.retry_workers, 'build_spec': TRIMER_BUILD_SPEC}


def parallel_retry(args, output, db, writer, cache):
    timeouts = {int(i): key for i, key in db.execute(
        "SELECT source_index, sample_key FROM rows WHERE main_status='TIMEOUT' AND retry_status IS NULL")}
    retry_target = db.execute("SELECT COUNT(*) FROM rows WHERE main_status='TIMEOUT'").fetchone()[0]
    initial_done = retry_target - len(timeouts)
    started = time.monotonic()
    processed = 0

    def report(phase):
        elapsed = time.monotonic() - started
        remaining = len(timeouts) - processed
        rate = processed / elapsed if elapsed else 0.0
        status = {'phase': phase, 'retry_workers': args.retry_workers,
                  'done': initial_done + processed, 'target': retry_target,
                  'completed_this_run': processed, 'wall_seconds': elapsed,
                  'observed_rows_per_second': rate,
                  'remaining_seconds_at_observed_rate':
                      0.0 if not remaining else (remaining / rate if rate else None)}
        atomic_json(output / 'retry_progress.json', status)
        print(json.dumps({'retry_progress': status}), flush=True)

    def retry_rows():
        matched = 0
        for row in source_rows(args.kind, args.limit):
            if row[0] not in timeouts:
                continue
            if row[2] != timeouts[row[0]]:
                raise ValueError(f'{row[0]}: timeout retry identity differs')
            matched += 1
            yield row
        if matched != len(timeouts):
            raise ValueError('not every timeout identity was found in source')

    report('retry_start')
    if timeouts:
        submitted = iter(retry_rows())
        pending = {}
        with ProcessPoolExecutor(max_workers=args.retry_workers, initializer=worker_init) as pool:
            for _ in range(args.retry_workers * 2):
                row = next(submitted, None)
                if row is None:
                    break
                pending[pool.submit(retry_compute, row)] = row
            while pending:
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    row = pending.pop(future)
                    result = future.result()
                    if result['row'] != row:
                        raise ValueError(f'{row[0]}: retry worker row identity differs')
                    store_result(result, 'retry', writer, cache, db, args.retry_workers)
                    processed += 1
                    if processed % args.retry_progress_every == 0:
                        writer.sync()
                        report('retry')
                    next_row = next(submitted, None)
                    if next_row is not None:
                        pending[pool.submit(retry_compute, next_row)] = next_row
    writer.sync()
    if db.execute("SELECT COUNT(*) FROM rows WHERE main_status='TIMEOUT' AND retry_status IS NULL").fetchone()[0]:
        raise ValueError('unresolved timeout identities remain after retry')
    report('retry_finished')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kind', choices=EXPECTED, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=56)
    parser.add_argument('--retry-workers', type=int, default=56)
    parser.add_argument('--retry-only', action='store_true',
                        help='Resume only TIMEOUT rows after the main writer has closed')
    parser.add_argument('--retry-progress-every', type=int, default=10)
    parser.add_argument('--limit', type=int, default=0, help='Only for isolated smoke output')
    parser.add_argument('--progress-every', type=int, default=10000)
    args = parser.parse_args()
    if (not 1 <= args.workers <= 64 or not 1 <= args.retry_workers <= 64 or
            args.limit < 0 or args.progress_every < 1 or args.retry_progress_every < 1):
        parser.error('invalid workers, limit or progress interval')
    output = args.output.resolve()
    if output.exists() and not (output / 'metadata.json').is_file():
        raise FileExistsError(f'output exists without resumable metadata: {output}')
    if args.retry_only and not (output / 'metadata.json').is_file():
        raise FileNotFoundError('retry-only requires an existing main staging cache')
    metadata = build_metadata(args)
    legacy = {key: value for key, value in metadata.items() if key != 'retry_workers'}
    legacy['retry'] = 'TIMEOUT only; one worker after main'
    # Only this exact legacy contract is eligible for the explicit retry-only
    # handoff. Other source/schema/build-spec mismatches still fail closed.
    migration = (args.retry_only and
                 json.loads((output / 'metadata.json').read_text()) == legacy)
    writer = StagingWriter(output, legacy if migration else metadata, map_size=64 * 1024**3)
    db = None
    cache = None
    started = time.monotonic()
    try:
        db = ledger_connect(output / 'rows.sqlite')
        if args.retry_only:
            target = min(EXPECTED[args.kind], args.limit) if args.limit else EXPECTED[args.kind]
            count = db.execute('SELECT COUNT(*) FROM rows').fetchone()[0]
            passes = db.execute("SELECT COUNT(*) FROM rows WHERE final_status='PASS'").fetchone()[0]
            if count != target or writer.count() != passes:
                raise ValueError('retry-only requires a complete main ledger and matching LMDB count')
            if migration:
                # Migration is performed while holding the exclusive writer
                # lock; preserve the original contract and every prior result.
                atomic_json(output / 'metadata.before_parallel_retry.json', legacy)
                atomic_json(output / 'retry_policy_change.json', {
                    'schema': 'mcl-ph-retry-policy-change-v1',
                    'changed_at_unix': time.time(),
                    'authorization': '2026-09-30 user requested multi-CPU timeout retry',
                    'previous_retry': legacy['retry'], 'retry_workers': args.retry_workers,
                    'main_rows_preserved': count,
                    'previously_retried_rows': db.execute(
                        'SELECT COUNT(*) FROM rows WHERE retry_status IS NOT NULL').fetchone()[0]})
                atomic_json(output / 'metadata.json', metadata)
            previous = json.loads((output / 'progress.json').read_text()) if (output / 'progress.json').is_file() else {}
            started -= float(previous.get('wall_seconds', 0))
            cache = DualStaticCache(STATIC / args.kind / 'dual_static_v1')
            parallel_retry(args, output, db, writer, cache)
            report_progress(args, output, db, started, 'finished')
            return
        cache = DualStaticCache(STATIC / args.kind / 'dual_static_v1')
        existing = {int(i): (key, status) for i, key, status in db.execute(
            'SELECT source_index, sample_key, main_status FROM rows')}
        def pending_rows():
            for row in source_rows(args.kind, args.limit):
                index, _, key_hex = row
                if index in existing:
                    if existing[index][0] != key_hex:
                        raise ValueError(f'{index}: resumed key differs from source')
                    if existing[index][1] == 'PASS':
                        if bytes.fromhex(key_hex) not in writer:
                            raise ValueError(f'{index}: ledger PASS has no LMDB record')
                        if not torch.isfinite(writer.get(bytes.fromhex(key_hex)).trimer_pos).all():
                            raise ValueError(f'{index}: resumed LMDB coordinates are nonfinite')
                    continue
                yield row
        submitted = iter(pending_rows())
        pending = {}
        processed = 0
        with ProcessPoolExecutor(max_workers=args.workers, initializer=worker_init) as pool:
            for _ in range(args.workers * 2):
                row = next(submitted, None)
                if row is None:
                    break
                pending[pool.submit(compute, row)] = row
            while pending:
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    row = pending.pop(future)
                    result = future.result()  # Contract/schema errors must stop the writer.
                    if result['row'] != row:
                        raise ValueError(f'{row[0]}: worker row identity differs')
                    store_result(result, 'main', writer, cache, db)
                    processed += 1
                    if processed % args.progress_every == 0:
                        writer.sync()
                        report_progress(args, output, db, started, 'main')
                    next_row = next(submitted, None)
                    if next_row is not None:
                        pending[pool.submit(compute, next_row)] = next_row
        writer.sync()
        report_progress(args, output, db, started, 'main_complete')
        parallel_retry(args, output, db, writer, cache)
        report_progress(args, output, db, started, 'finished')
    finally:
        if cache is not None:
            cache.close()
        if db is not None:
            db.close()
        writer.close()


def report_progress(args, output, db, started, phase):
    statuses = dict(db.execute('SELECT final_status, COUNT(*) FROM rows GROUP BY final_status'))
    total = sum(statuses.values())
    target = min(EXPECTED[args.kind], args.limit) if args.limit else EXPECTED[args.kind]
    elapsed = time.monotonic() - started
    main_mean, main_max = db.execute(
        'SELECT AVG(main_seconds), MAX(main_seconds) FROM rows').fetchone()
    throughput = total / elapsed if elapsed else 0.0
    report = {'phase': phase, 'kind': args.kind, 'done': total, 'target': target,
              'final_statuses': statuses, 'wall_seconds': elapsed,
              'main_seconds_mean': main_mean, 'main_seconds_max': main_max,
              'observed_rows_per_second': throughput,
              'remaining_seconds_at_observed_rate':
                  (target - total) / throughput if throughput else None,
              'lmdb_disk_bytes': (output / 'data.lmdb/data.mdb').stat().st_size}
    (output / 'progress.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'progress': report}), flush=True)
    if phase == 'finished' and total != target:
        raise ValueError(f'ledger has {total} rows, expected {target}')


if __name__ == '__main__':
    main()
