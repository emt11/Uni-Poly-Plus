#!/usr/bin/env python3
"""Build the offline noisy-topology trajectory cache of the MCL-PH route (r10R2).

One row per *presentation position* of the declared pre-training schedule, never
per molecule: position ``p`` is resolved through the same ``OrderedSampleStream``
and the same shared noisy-view draw the training run uses, and the stored value
is the ``[31, 5]`` float32 trajectory the online path would compute for it.  The
build is CPU-only and shard-parallel; a completed segment is never rebuilt, and
an incomplete one is rebuilt from its start rather than partially trusted.

The script deliberately stops before everything the training path does after the
trajectory: no expert relations, no non-bond candidates, no model, no forward or
backward pass.  Those stay online.

Progress is tracked through the segment markers the workers write, not through
the pool's result stream, and the pool is always torn down with ``terminate()``:
a worker that fails is reported instead of derailing the driver, and a driver
that fails still leaves a cache the next invocation can resume from.
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
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataset.mcl_ph_view import DESCRIPTOR_COLUMNS, ROUTER_RADII
from src.dataset.mcl_ph_trajectory_cache import (BUILD_STATE_DIRNAME, CACHE_SCHEMA,
                                                 DESCRIPTOR_DTYPE, MANIFEST_NAME,
                                                 SEGMENTS_DIRNAME, SHARDS_DIRNAME,
                                                 TrajectoryCacheBuilder,
                                                 adopt_written_segments, cache_identity,
                                                 create_shard_files, describe_shard,
                                                 position_chunks, read_segment_markers,
                                                 segment_key, shard_bounds, shard_paths,
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


def _run_segment(task):
    """Build one segment; a failure is returned as a record, never raised.

    A raising task would come back through the pool as an exception in the
    driver, which can abandon the build after hours of work; returning the
    failure keeps the driver's loop and the segment markers in charge.
    """
    shard, first, rows = (int(task[0]), int(task[1]), int(task[2]))
    started = time.perf_counter()
    try:
        _WORKER['builder'].build_chunk(shard, first, rows)
    except Exception as exc:                                   # noqa: BLE001 - reported
        return {'shard': shard, 'first_position': first, 'rows': rows, 'ok': False,
                'seconds': time.perf_counter() - started, 'error': f'{type(exc).__name__}: {exc}',
                'traceback': traceback.format_exc()[-4000:]}
    return {'shard': shard, 'first_position': first, 'rows': rows, 'ok': True,
            'seconds': time.perf_counter() - started, 'error': None, 'traceback': None}


def _load_manifest(root):
    path = Path(root) / MANIFEST_NAME
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding='utf-8'))


def _reset_cache(root, shard_size, total_positions, log):
    """Forget every shard file and segment marker of a previous build."""
    for path in shard_paths(root, shard_size, total_positions):
        path.unlink(missing_ok=True)
    directory = Path(root) / BUILD_STATE_DIRNAME / SEGMENTS_DIRNAME
    if directory.is_dir():
        for path in directory.glob('*.json'):
            path.unlink(missing_ok=True)
    log(f'=== DISCARDED the recorded shards and markers of a previous build')


def build_cache(*, builder_payload, root, shard_size, total_positions, chunk_size, workers,
                requested_total_positions, max_updates, log=print, progress_seconds=30.0,
                resume=True, adopt_written=False, poll_seconds=5.0, stall_seconds=900.0,
                max_requeues=2):
    """Fill every unbuilt segment of one cache; returns the finished manifest.

    Completion is read from the segment markers on disk.  The worker result
    stream only carries failure reports, so a driver that stops receiving
    results still finishes the build and writes its manifest, and every failure
    is named with its ``position`` instead of ending the run silently.
    """
    root = Path(root)
    bounds = shard_bounds(shard_size, total_positions)
    (root / SHARDS_DIRNAME).mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    manifest = _load_manifest(root)
    if not resume:
        _reset_cache(root, shard_size, total_positions, log)
        manifest = None
    create_shard_files(root, shard_size, total_positions)
    adopted = None
    if adopt_written:
        adopted = adopt_written_segments(root, shard_size, total_positions,
                                         chunk_size=chunk_size, log=log)
    tasks = position_chunks(shard_size, total_positions, chunk_size)
    expected = {}
    for shard, first, rows in tasks:
        expected.setdefault(int(shard), {})[int(first)] = int(rows)
    markers = read_segment_markers(root)
    done, problems = {}, []
    for entry in (manifest or {}).get('shards', []):
        index = int(entry['shard'])
        first, rows = bounds[index]
        path = root / SHARDS_DIRNAME / str(entry['file'])
        marked = {key for key in markers if key[0] == index}
        if int(entry['first_position']) != first or int(entry['rows']) != rows:
            problems.append(f'shard {index} disagrees with the declared layout')
        elif not path.is_file():
            problems.append(f'shard {index} file is missing')
        elif len(marked) < len(expected.get(index, {})):
            problems.append(f'shard {index} has not every segment marked as written')
        elif sha256_file(path) != str(entry['sha256']):
            problems.append(f'shard {index} does not match its recorded hash')
        else:
            done[index] = entry
    if problems:
        log(f'=== DISCARDED {len(problems)} recorded shard(s): ' + '; '.join(problems[:5])
            + (' ...' if len(problems) > 5 else ''))
    recovered = []
    for index in sorted(expected):
        if index in done:
            continue
        if all(segment_key(index, first) in markers and
               int(markers[segment_key(index, first)]['rows']) == int(rows)
               for first, rows in expected[index].items()):
            done[index] = describe_shard(root, index, *bounds[index])
            recovered.append(index)
    if recovered:
        log(f'=== RECOVERED {len(recovered)} shard(s) from their segment markers: '
            f'{recovered[:8]}' + (' ...' if len(recovered) > 8 else ''))

    def publish(complete, extra=None):
        payload = dict(schema=CACHE_SCHEMA, dtype=DESCRIPTOR_DTYPE.__name__,
                       shape_per_item=[len(ROUTER_RADII), DESCRIPTOR_COLUMNS],
                       total_positions=int(total_positions), shard_size=int(shard_size),
                       requested_total_positions=int(requested_total_positions),
                       max_updates=int(max_updates),
                       router_radii=[float(value) for value in ROUTER_RADII],
                       descriptor_columns=DESCRIPTOR_COLUMNS,
                       builder='scripts/build_mcl_ph_trajectory_cache.py',
                       builder_commit=_builder_commit(),
                       created_at=(manifest or {}).get('created_at', _now()),
                       updated_at=_now(), complete=bool(complete),
                       completed_shards=len(done), adopted_segments=(
                           int(adopted['adopted']) if adopted is not None else 0),
                       shards=sorted(done.values(), key=lambda row: row['shard']))
        payload.update(builder_payload['identity'])
        if extra:
            payload.update(extra)
        write_manifest(root, payload)
        return payload

    publish(False)
    pending = [task for task in tasks if segment_key(task[0], task[1]) not in markers]
    pending_rows = sum(int(task[2]) for task in pending)
    log(f'=== CACHE BUILD START {_now()} root={root} positions={total_positions} '
        f'shards={len(bounds)} chunk={chunk_size} workers={workers} '
        f'todo_segments={len(pending)} todo_rows={pending_rows} reused_shards={len(done)}')
    if not pending:
        final = publish(len(done) == len(bounds),
                        {'build_seconds': round(time.perf_counter() - started, 3),
                         'build_workers': int(workers), 'build_chunk_size': int(chunk_size)})
        log(f'=== CACHE BUILD NOTHING TO DO {_now()} shards={len(done)}')
        return final

    clock = {'rows': 0, 'segments': 0, 'shards': 0, 'last_log': time.monotonic()}
    failures, lost = {}, []

    def on_segment(record):
        if record.get('ok'):
            return
        key = segment_key(record['shard'], record['first_position'])
        failures[key] = record.get('error')
        log(f'--- SEGMENT FAILED {_now()} shard={record["shard"]} '
            f'first_position={record["first_position"]} rows={record["rows"]} '
            f'{record["error"]}')

    def on_lost(exc):
        lost.append(f'{type(exc).__name__}: {exc}')

    context = mp.get_context('fork')
    pool, requeues, outstanding, last_progress = None, 0, {}, time.monotonic()
    try:
        while True:
            if pending and pool is None:
                pool = context.Pool(max(1, int(workers)), initializer=_init_worker,
                                    initargs=(builder_payload,))
                for task in pending:
                    pool.apply_async(_run_segment, (task,), callback=on_segment,
                                     error_callback=on_lost)
                outstanding = {segment_key(task[0], task[1]): task for task in pending}
                pending, last_progress = [], time.monotonic()
            time.sleep(max(0.2, float(poll_seconds)))
            markers = read_segment_markers(root)
            fresh = [key for key in outstanding if key in markers]
            for key in fresh:
                del outstanding[key]
                clock['rows'] += int(markers[key]['rows'])
                clock['segments'] += 1
            if fresh:
                last_progress = time.monotonic()
            for index in sorted(expected):
                if index in done:
                    continue
                if all(segment_key(index, first) in markers and
                       int(markers[segment_key(index, first)]['rows']) == int(rows)
                       for first, rows in expected[index].items()):
                    done[index] = describe_shard(root, index, *bounds[index])
                    clock['shards'] += 1
                    publish(False)
            now = time.monotonic()
            if now - clock['last_log'] >= float(progress_seconds):
                window = now - clock['last_log']
                log(f'--- {_now()} rows={clock["rows"]}/{pending_rows} '
                    f'({100.0 * clock["rows"] / max(1, pending_rows):.2f}%) '
                    f'segments={clock["segments"]}/{len(outstanding) + clock["segments"]} '
                    f'shards_done={clock["shards"]} rate={clock["rows"] / window:.1f}/s '
                    f'outstanding={len(outstanding)} requeues={requeues} '
                    f'load1={os.getloadavg()[0]:.1f} rss={(_resident_bytes() or 0) / 2**30:.2f}GiB '
                    f'child_max_rss={_child_max_rss_bytes() / 2**30:.2f}GiB')
                clock['last_log'] = now
            if not outstanding:
                break
            if now - last_progress > float(stall_seconds):
                requeues += 1
                missing = sorted(outstanding)
                log(f'--- STALLED {_now()} no new segment for {stall_seconds:.0f}s; '
                    f'{len(missing)} segment(s) outstanding, requeue {requeues}/'
                    f'{max_requeues}: {missing[:8]}' + (' ...' if len(missing) > 8 else ''))
                pool.terminate()
                pool.join()
                pool = None
                if requeues > int(max_requeues):
                    raise RuntimeError(
                        f'the cache build stalled: {len(missing)} segment(s) still unbuilt '
                        f'after {max_requeues} requeue(s), first {missing[:8]}; '
                        f'last reported failure: {failures.get(missing[0]) or lost[-1:] or None}')
                pending = [task for _, task in sorted(outstanding.items())]
                outstanding = {}
    finally:
        if pool is not None:
            pool.terminate()
            pool.join()
    complete = len(done) == len(bounds)
    missing_shards = sorted(set(range(len(bounds))) - set(done))
    final = publish(complete, {'build_seconds': round(time.perf_counter() - started, 3),
                               'build_workers': int(workers),
                               'build_chunk_size': int(chunk_size)})
    log(f'=== CACHE BUILD END {_now()} rows={clock["rows"]} shards={len(done)}/{len(bounds)} '
        f'complete={complete} seconds={time.perf_counter() - started:.1f} '
        f'rate={clock["rows"] / max(1e-9, time.perf_counter() - started):.1f}/s '
        f'load1={os.getloadavg()[0]:.1f} child_max_rss={_child_max_rss_bytes() / 2**30:.2f}GiB '
        f'failed_segments={len(failures)} lost_results={len(lost)}')
    if failures:
        for key in sorted(failures)[:5]:
            log(f'=== FAILED SEGMENT shard={key[0]} first_position={key[1]} {failures[key]}')
    if not complete:
        raise SystemExit(f'the cache is incomplete; missing shards {missing_shards[:8]} '
                         f'({len(missing_shards)} of {len(bounds)})')
    return final


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
    parser.add_argument('--chunk-size', type=int, default=10_000,
                        help='rows per work unit; also the granularity of completion state')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--limit-positions', type=int, default=0,
                        help='build only the first N positions (benchmark and gate builds)')
    parser.add_argument('--no-resume', action='store_true',
                        help='discard recorded shards and rebuild every position')
    parser.add_argument('--adopt-written-segments', action='store_true',
                        help='mark the segments whose rows are already fully written; used to '
                             'resume a build whose driver lost its own bookkeeping')
    parser.add_argument('--poll-seconds', type=float, default=5.0)
    parser.add_argument('--stall-seconds', type=float, default=900.0)
    parser.add_argument('--max-requeues', type=int, default=2)
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
                adopt_written=bool(args.adopt_written_segments),
                poll_seconds=float(args.poll_seconds), stall_seconds=float(args.stall_seconds),
                max_requeues=int(args.max_requeues),
                progress_seconds=float(args.progress_seconds))


if __name__ == '__main__':
    main()
