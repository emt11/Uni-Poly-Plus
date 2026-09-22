#!/usr/bin/env python3
"""Build the offline noisy-topology trajectory cache of the MCL-PH route (r10R2).

One row per *presentation position* of the declared pre-training schedule, never
per molecule: position ``p`` is resolved through the same ``OrderedSampleStream``
and the same shared noisy-view draw the training run uses, and the stored value
is the ``[31, 5]`` float32 trajectory the online path would compute for it.  The
build is CPU-only and shard-parallel; a completed shard is never rebuilt, and an
incomplete shard is rebuilt from its start rather than partially trusted.

The script deliberately stops before everything the training path does after the
trajectory: no expert relations, no non-bond candidates, no model, no forward or
backward pass.  Those stay online.
"""
import argparse
from datetime import datetime, timezone
import json
import multiprocessing as mp
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataset.mcl_ph_view import DESCRIPTOR_COLUMNS, ROUTER_RADII
from src.dataset.mcl_ph_trajectory_cache import (CACHE_SCHEMA, DESCRIPTOR_DTYPE,
                                                 MANIFEST_NAME, SHARDS_DIRNAME,
                                                 TrajectoryCacheBuilder, cache_identity,
                                                 create_shard_files, describe_shard,
                                                 position_chunks, shard_bounds,
                                                 sha256_file, write_manifest)

BUILDER_CLASS = TrajectoryCacheBuilder
_WORKER = {}


def _now():
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def _builder_commit():
    try:
        return subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=Path(__file__).resolve().parents[1],
                              capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _resident_bytes():
    try:
        with open('/proc/self/statm', encoding='utf-8') as handle:
            pages = int(handle.read().split()[1])
        return pages * os.sysconf('SC_PAGE_SIZE')
    except (OSError, ValueError, IndexError):
        return None


def _child_max_rss_bytes():
    return int(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss) * 1024


def _init_worker(payload):
    _WORKER['builder'] = BUILDER_CLASS(**payload)


def _run_chunk(task):
    shard, first, rows = (int(task[0]), int(task[1]), int(task[2]))
    started = time.perf_counter()
    _WORKER['builder'].build_chunk(shard, first, rows)
    return shard, first, rows, time.perf_counter() - started


def _load_manifest(root):
    path = Path(root) / MANIFEST_NAME
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding='utf-8'))


def _keep_existing_shards(root, manifest, shard_size, total_positions, log):
    """Completed shards of a previous build, after verifying them on disk."""
    bounds = shard_bounds(shard_size, total_positions)
    kept, problems = [], []
    for entry in (manifest or {}).get('shards', []):
        index = int(entry['shard'])
        first, rows = bounds[index]
        path = Path(root) / SHARDS_DIRNAME / str(entry['file'])
        if int(entry['first_position']) != first or int(entry['rows']) != rows:
            problems.append(f'shard {index} disagrees with the declared layout')
        elif not path.is_file():
            problems.append(f'shard {index} file is missing')
        elif sha256_file(path) != str(entry['sha256']):
            problems.append(f'shard {index} does not match its recorded hash')
        else:
            kept.append(entry)
    if problems:
        log(f'=== DISCARDED {len(problems)} recorded shard(s): ' + '; '.join(problems[:5])
            + (' ...' if len(problems) > 5 else ''))
    return kept


def build_cache(*, builder_payload, root, shard_size, total_positions, chunk_size, workers,
                requested_total_positions, max_updates, log=print, progress_seconds=30.0,
                resume=True):
    """Fill every missing shard of one cache; returns the finished manifest."""
    root = Path(root)
    bounds = shard_bounds(shard_size, total_positions)
    (root / SHARDS_DIRNAME).mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    manifest = _load_manifest(root)
    completed = (_keep_existing_shards(root, manifest, shard_size, total_positions, log)
                 if resume else [])
    if not resume and manifest is not None:
        for entry in manifest.get('shards', []):
            (root / SHARDS_DIRNAME / str(entry['file'])).unlink(missing_ok=True)
        completed = []
    create_shard_files(root, shard_size, total_positions)
    done = {int(entry['shard']): entry for entry in completed}
    manifest = dict(schema=CACHE_SCHEMA, dtype=DESCRIPTOR_DTYPE.__name__,
                    shape_per_item=[len(ROUTER_RADII), DESCRIPTOR_COLUMNS],
                    total_positions=int(total_positions), shard_size=int(shard_size),
                    requested_total_positions=int(requested_total_positions),
                    max_updates=int(max_updates),
                    router_radii=[float(value) for value in ROUTER_RADII],
                    descriptor_columns=DESCRIPTOR_COLUMNS,
                    builder='scripts/build_mcl_ph_trajectory_cache.py',
                    builder_commit=_builder_commit(),
                    created_at=(manifest or {}).get('created_at', _now()),
                    updated_at=_now(), complete=False,
                    shards=sorted(done.values(), key=lambda row: row['shard']))
    manifest.update(builder_payload['identity'])
    write_manifest(root, manifest)
    tasks = [task for task in position_chunks(shard_size, total_positions, chunk_size)
             if task[0] not in done]
    pending_rows = sum(task[2] for task in tasks)
    log(f'=== CACHE BUILD START {_now()} root={root} positions={total_positions} '
        f'shards={len(bounds)} chunk={chunk_size} workers={workers} '
        f'todo_chunks={len(tasks)} todo_rows={pending_rows} '
        f'reused_shards={len(done)}')
    if not tasks:
        manifest['complete'] = len(done) == len(bounds)
        manifest['updated_at'] = _now()
        write_manifest(root, manifest)
        log(f'=== CACHE BUILD NOTHING TO DO {_now()} shards={len(done)}')
        return manifest
    clock = {'rows': 0, 'chunks': 0, 'shards': 0, 'last_log': started, 'window_start': started,
             'window_rows': 0}
    remaining = {index: sum(task[2] for task in tasks if task[0] == index) for index, _ in bounds}
    context = mp.get_context('fork')
    pool = context.Pool(max(1, int(workers)), initializer=_init_worker,
                        initargs=(builder_payload,))
    try:
        for shard, first, rows, seconds in pool.imap_unordered(_run_chunk, tasks, chunksize=1):
            clock['rows'] += rows
            clock['chunks'] += 1
            clock['window_rows'] += rows
            remaining[shard] -= rows
            if remaining[shard] == 0:
                entry = describe_shard(root, shard, bounds[shard][0], bounds[shard][1])
                done[shard] = entry
                manifest['shards'] = sorted(done.values(), key=lambda row: row['shard'])
                manifest['completed_shards'] = len(done)
                manifest['updated_at'] = _now()
                write_manifest(root, manifest)
                clock['shards'] += 1
            now = time.perf_counter()
            if now - clock['last_log'] >= float(progress_seconds):
                elapsed = now - started
                window = now - clock['window_start']
                rate = clock['window_rows'] / window if window > 0 else 0.0
                overall = clock['rows'] / elapsed if elapsed > 0 else 0.0
                left = (pending_rows - clock['rows']) / overall if overall > 0 else float('inf')
                log(f'--- {_now()} rows={clock["rows"]}/{pending_rows} '
                    f'({100.0 * clock["rows"] / max(1, pending_rows):.2f}%) '
                    f'shards_done={clock["shards"]} chunk_s={seconds:.1f} '
                    f'rate={rate:.1f}/s overall={overall:.1f}/s eta={left / 60.0:.1f}min '
                    f'load1={os.getloadavg()[0]:.1f} rss={(_resident_bytes() or 0) / 2**30:.2f}GiB '
                    f'child_max_rss={_child_max_rss_bytes() / 2**30:.2f}GiB')
                clock['last_log'] = now
                clock['window_start'] = now
                clock['window_rows'] = 0
    finally:
        pool.close()
        pool.join()
    elapsed = time.perf_counter() - started
    complete = len(done) == len(bounds)
    manifest['shards'] = sorted(done.values(), key=lambda row: row['shard'])
    manifest['completed_shards'] = len(done)
    manifest['complete'] = complete
    manifest['updated_at'] = _now()
    manifest['build_seconds'] = round(elapsed, 3)
    manifest['build_workers'] = int(workers)
    manifest['build_chunk_size'] = int(chunk_size)
    write_manifest(root, manifest)
    log(f'=== CACHE BUILD END {_now()} rows={clock["rows"]} shards={len(done)}/{len(bounds)} '
        f'complete={complete} seconds={elapsed:.1f} '
        f'rate={clock["rows"] / elapsed if elapsed > 0 else 0.0:.1f}/s '
        f'load1={os.getloadavg()[0]:.1f} child_max_rss={_child_max_rss_bytes() / 2**30:.2f}GiB')
    if not complete:
        missing = sorted(set(range(len(bounds))) - set(done))
        raise SystemExit(f'the cache is incomplete; missing shards {missing[:8]}')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    for name in ('cohort-root', 'cache-root', 'dual-static-root', 'sample-index-artifact',
                 'output'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--sample-index-split', default='train')
    parser.add_argument('--max-updates', type=int, default=5000)
    parser.add_argument('--global-batch', type=int, default=1008)
    parser.add_argument('--shard-size', type=int, default=100_000)
    parser.add_argument('--chunk-size', type=int, default=10_000)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--limit-positions', type=int, default=0,
                        help='build only the first N positions (benchmark and gate builds)')
    parser.add_argument('--no-resume', action='store_true',
                        help='discard recorded shards and rebuild every position')
    parser.add_argument('--progress-seconds', type=float, default=30.0)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding='utf-8'))
    requested = int(args.max_updates) * int(args.global_batch)
    total = requested if not args.limit_positions else min(requested, int(args.limit_positions))
    payload = dict(cohort_root=args.cohort_root, cache_root=args.cache_root,
                   dual_static_root=args.dual_static_root,
                   sample_index_artifact=args.sample_index_artifact,
                   sample_index_split=args.sample_index_split, output=str(args.output),
                   shard_size=int(args.shard_size), total_positions=int(total),
                   seed=int(config['seed']), sigma=float(config['noise_sigma']),
                   ratio=float(config['atom_mask_ratio']), identity={})
    # The cohort and static hashes live in the frozen readers, so the identity is
    # completed by one probe process before any worker starts.
    probe = BUILDER_CLASS(**payload)
    try:
        payload['identity'] = cache_identity(
            seed=int(config['seed']), noise_sigma=float(config['noise_sigma']),
            mask_ratio=float(config['atom_mask_ratio']), global_batch=int(args.global_batch),
            sample_index_split=args.sample_index_split,
            sample_index_artifact_sha256=probe.split_sha256,
            cohort_manifest_hash=probe.source.cohort['manifest_hash'],
            dual_static_manifest_hash=probe.source.static_cache.manifest_hash)
        print(f'=== IDENTITY {json.dumps(payload["identity"], sort_keys=True)}')
    finally:
        probe.close()
    build_cache(builder_payload=payload, root=args.output, shard_size=int(args.shard_size),
                total_positions=int(total), chunk_size=int(args.chunk_size),
                workers=int(args.workers), requested_total_positions=int(requested),
                max_updates=int(args.max_updates), resume=not args.no_resume,
                progress_seconds=float(args.progress_seconds))


if __name__ == '__main__':
    main()
