#!/usr/bin/env python3
"""Run isolated formal task/fold shards in a bounded GPU grid."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.finetune_glt_dual import TASKS, fixed_manifest


def _validate_gpu_slots(gpus):
    gpus = [str(gpu) for gpu in gpus]
    if not 1 <= len(gpus) <= 4:
        raise ValueError('grid supports one to four GPU slots')
    if len(set(gpus)) != len(gpus):
        raise ValueError('grid requires distinct GPU slots; duplicate --gpu values are not allowed')
    return gpus


def _build_shard_command(args, task, fold, unit):
    mode = '--smoke' if getattr(args, 'smoke', False) else '--formal-shard'
    return [
        sys.executable, 'scripts/finetune_glt_dual.py',
        '--config', args.config, '--checkpoint', args.checkpoint,
        '--raw-root', args.raw_root, '--cohort-root', args.cohort_root,
        '--cache-root', args.cache_root, '--dual-static-root', args.dual_static_root,
        '--clean-cache-gib', format(args.clean_cache_gib, 'g'),
        '--output', str(unit), '--split-root', args.split_root,
        '--task', task, '--fold', str(fold), mode,
        *(['--timing'] if getattr(args, 'timing', False) else []),
    ]


def _completed_unit_matches(unit, task, fold, smoke):
    """Validate a resumable unit's mode and identity before skipping it."""

    unit = Path(unit)
    paths = {name: unit / f'{name}.json' for name in ('run', 'summary', 'runtime')}
    if not all(path.is_file() for path in paths.values()):
        return False
    try:
        records = {name: json.loads(path.read_text(encoding='utf-8'))
                   for name, path in paths.items()}
    except (OSError, ValueError):
        return False
    expected_protocol = 'outer5_inner20_smoke' if smoke else 'outer5_inner20_formal_shard'
    expected_outer = 'NOT_RUN' if smoke else 'RUN'
    expected_summary_outer = 'NOT_RUN' if smoke else 'RUN_ONCE'
    run, summary, runtime = records['run'], records['summary'], records['runtime']
    mismatches = []
    for name, record in (('run', run), ('runtime', runtime)):
        if record.get('protocol') != expected_protocol:
            mismatches.append(f'{name}.protocol={record.get("protocol")!r}')
        if bool(record.get('smoke')) != bool(smoke):
            mismatches.append(f'{name}.smoke={record.get("smoke")!r}')
        if bool(record.get('formal_shard')) != (not smoke):
            mismatches.append(f'{name}.formal_shard={record.get("formal_shard")!r}')
        if record.get('selected_tasks') != [task]:
            mismatches.append(f'{name}.selected_tasks={record.get("selected_tasks")!r}')
        if record.get('selected_folds') != [int(fold)]:
            mismatches.append(f'{name}.selected_folds={record.get("selected_folds")!r}')
        if record.get('outer_test') != expected_outer:
            mismatches.append(f'{name}.outer_test={record.get("outer_test")!r}')
    if summary.get('protocol') != expected_protocol:
        mismatches.append(f'summary.protocol={summary.get("protocol")!r}')
    if bool(summary.get('smoke')) != bool(smoke):
        mismatches.append(f'summary.smoke={summary.get("smoke")!r}')
    if bool(summary.get('formal_shard')) != (not smoke):
        mismatches.append(f'summary.formal_shard={summary.get("formal_shard")!r}')
    if summary.get('outer_test') != expected_summary_outer:
        mismatches.append(f'summary.outer_test={summary.get("outer_test")!r}')
    if runtime.get('status') != 'PASS':
        mismatches.append(f'runtime.status={runtime.get("status")!r}')
    task_record = summary.get('tasks', {}).get(task) if isinstance(summary.get('tasks'), dict) else None
    if not isinstance(task_record, dict):
        mismatches.append('summary.tasks does not contain the selected task')
    elif smoke and (not task_record.get('smoke') or task_record.get('outer_test') != 'NOT_RUN'):
        mismatches.append('summary task is not a validation-only smoke')
    elif not smoke and (not task_record.get('formal_shard') or task_record.get('outer_test') != 'RUN_ONCE'):
        mismatches.append('summary task is not a formal shard')
    if mismatches:
        raise RuntimeError(
            f'refusing to resume incompatible grid unit {unit}: ' + '; '.join(mismatches))
    return True


def _launch_with_boundary(launch, task, fold, gpu):
    # Start the controller boundary before calling the launch helper.  This
    # includes log creation, Popen, and any helper-side setup in the measured
    # launch-to-exit interval.
    launch_started = time.monotonic()
    process, handle, log_path = launch(task, fold, gpu)
    process_ready_observed = time.monotonic()
    handle.write(
        f'GRID_LAUNCH_STARTED_MONOTONIC={launch_started}\n'
        # Keep the historical field as a compatibility alias, but document
        # that it now means the pre-launch controller boundary.
        f'GRID_LAUNCH_MONOTONIC={launch_started}\n'
        f'GRID_PROCESS_READY_OBSERVED_MONOTONIC={process_ready_observed}\n')
    handle.flush()
    return (task, fold, process, handle, log_path, launch_started,
            process_ready_observed)


def _finish_child(item, *, code=None, exited=None):
    """Record a child after ``poll`` observed completion, then reap it.

    ``exited`` is intentionally supplied by the polling loop.  Calling
    ``wait`` is only resource reaping and must not define the exit timestamp.
    """

    task, fold, process, handle, log_path, launch_started, process_ready = item
    if code is None:
        code = process.poll()
        if code is None:
            raise RuntimeError('cannot finish a child before poll observes exit')
        exited = time.monotonic()
    elif exited is None:
        raise RuntimeError('a polled child requires its observation timestamp')
    # Reap only after the observation timestamp has been captured.
    process.wait()
    elapsed = exited - launch_started
    handle.write(
        f'GRID_EXIT_OBSERVED_MONOTONIC={exited}\n'
        # Compatibility alias; this is controller observation time, not an
        # operating-system process-exit timestamp.
        f'GRID_EXIT_MONOTONIC={exited}\n'
        f'GRID_LAUNCH_STARTED_MONOTONIC={launch_started}\n'
        f'GRID_PROCESS_READY_OBSERVED_MONOTONIC={process_ready}\n'
        f'GRID_LAUNCH_TO_EXIT_SECONDS={elapsed}\n'
        f'EXIT_CODE={code}\n')
    handle.close()
    return {'task': task, 'fold': fold, 'log': str(log_path),
            'exit_code': code,
            'launch_started_monotonic': float(launch_started),
            'process_ready_observed_monotonic': float(process_ready),
            'exit_observed_monotonic': float(exited),
            'launch_to_exit_seconds': float(elapsed)}


def _drain_active(active, *, poll_interval):
    """Poll all owned children until they finish, closing every handle."""

    rows = []
    while active:
        observed_any = False
        for key, item in list(active.items()):
            code = item[2].poll()
            if code is None:
                continue
            observed = time.monotonic()
            rows.append(_finish_child(item, code=code, exited=observed))
            del active[key]
            observed_any = True
        if active and not observed_any:
            time.sleep(float(poll_interval))
    return rows


def _run_dynamic_jobs(jobs, gpus, launch, *, poll_interval=0.05):
    """Run isolated shards with a free-slot queue instead of batch barriers.

    ``launch(task, fold, gpu)`` returns ``(process, handle, log_path)``.  A
    completed process immediately frees its own GPU slot; a failed shard stops
    new dispatch, while already-running shards are allowed to finish and are
    still closed/recorded before the error is raised.
    """

    pending = list(jobs)
    free = list(gpus)
    active = {}
    completed, failures = [], []

    def start_available():
        while pending and free and not failures:
            task, fold = pending.pop(0)
            gpu = free.pop(0)
            try:
                item = _launch_with_boundary(launch, task, fold, gpu)
            except BaseException:
                free.insert(0, gpu)
                completed.extend(_drain_active(active, poll_interval=poll_interval))
                active.clear()
                raise
            active[gpu] = item

    start_available()
    while active:
        finished = []
        for gpu, item in list(active.items()):
            task, fold, process, handle, log_path, _, _ = item
            code = process.poll()
            if code is None:
                continue
            # Capture the controller observation boundary immediately after
            # poll reports completion; wait() below only reaps the process.
            row = _finish_child(item, code=code, exited=time.monotonic())
            del active[gpu]
            free.append(gpu)
            finished.append(gpu)
            if code != 0:
                failures.append(row)
            else:
                completed.append(row)
        if failures:
            # Do not dispatch pending work after a failure.  Existing child
            # processes remain owned and are waited/closed, so no orphaned
            # shard is left behind.
            if active:
                time.sleep(float(poll_interval))
            continue
        if finished:
            free.sort(key=str)
            start_available()
        else:
            time.sleep(float(poll_interval))

    if failures:
        raise RuntimeError('formal shard failed; no new shards dispatched: '
                           + json.dumps({'failures': failures,
                                         'completed': completed,
                                         'pending': pending}, sort_keys=True))
    return completed


def _run_batched_jobs(jobs, gpus, launch, *, poll_interval=0.05):
    """Reference scheduler: wait for a whole GPU batch before dispatching more."""

    completed = []
    for offset in range(0, len(jobs), len(gpus)):
        active = []
        try:
            for task, fold in jobs[offset:offset + len(gpus)]:
                gpu = gpus[len(active)]
                active.append(_launch_with_boundary(launch, task, fold, gpu))
        except BaseException:
            _drain_active({index: item for index, item in enumerate(active)},
                          poll_interval=poll_interval)
            raise
        # Keep the batch barrier, but harvest all children concurrently.  A
        # short child may therefore be observed before an earlier long child;
        # no next batch is launched until this active list is empty.
        observed_rows = []
        while active:
            observed_any = False
            for index, item in list(enumerate(active)):
                code = item[2].poll()
                if code is None:
                    continue
                observed = time.monotonic()
                observed_rows.append((index, _finish_child(
                    item, code=code, exited=observed)))
                active[index] = None
                observed_any = True
            active = [item for item in active if item is not None]
            if active and not observed_any:
                time.sleep(float(poll_interval))
        rows = [row for _, row in sorted(observed_rows, key=lambda pair: pair[0])]
        failures = [row for row in rows if row['exit_code']]
        completed.extend(row for row in rows if not row['exit_code'])
        if failures:
            raise RuntimeError('batched smoke shard failed: ' +
                               json.dumps({'failures': failures}, sort_keys=True))
    return completed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'checkpoint', 'raw-root', 'cohort-root', 'cache-root', 'dual-static-root', 'output'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--split-root', default='data/splits/mips_outer5_inner20')
    parser.add_argument('--task', action='append', dest='tasks')
    parser.add_argument('--fold', action='append', type=int, dest='folds',
                        help='select fold(s); required with --smoke')
    parser.add_argument('--smoke', action='store_true',
                        help='run selected task/fold units as validation-only smokes')
    parser.add_argument('--batch-wait', action='store_true',
                        help='smoke reference scheduler: wait for each GPU batch before dispatching more')
    parser.add_argument('--gpu', action='append', dest='gpus', required=True,
                        help='one CUDA device per concurrent shard; repeat up to 4')
    parser.add_argument('--log-root', required=True)
    parser.add_argument('--clean-cache-gib', type=float, default=0.0,
                        help='forwarded process-local clean cache capacity in GiB (default: 0)')
    parser.add_argument('--timing', action='store_true',
                        help='forward per-epoch timing to each child shard')
    parser.add_argument('--resume', action='store_true',
                        help='continue an existing grid root: skip units that already have a '
                             'completed summary.json, and refuse to touch partial units')
    args = parser.parse_args()
    tasks = list(args.tasks) if args.tasks else list(TASKS)
    if sorted(set(tasks)) != sorted(tasks) or any(task not in TASKS for task in tasks):
        raise ValueError('unknown or duplicate task selection')
    folds = [int(value) for value in args.folds] if args.folds else list(range(5))
    if any(value < 0 or value >= 5 for value in folds) or len(set(folds)) != len(folds):
        raise ValueError('unknown or duplicate fold selection')
    if args.smoke:
        if not args.tasks or not args.folds:
            raise ValueError('--smoke requires explicit --task and --fold selections')
    elif args.folds:
        raise ValueError('--fold selection requires --smoke')
    if not math.isfinite(args.clean_cache_gib) or args.clean_cache_gib < 0:
        raise ValueError('--clean-cache-gib must be finite and non-negative')
    gpus = _validate_gpu_slots(args.gpus)
    if args.batch_wait and (not args.smoke or len(gpus) < 2):
        raise ValueError('--batch-wait requires --smoke with at least two GPU slots')
    output = Path(args.output).resolve()
    log_root = Path(args.log_root).resolve()
    if args.resume:
        if not output.is_dir():
            raise ValueError('--resume requires an existing grid root')
    else:
        output.mkdir(parents=True, exist_ok=False)
    log_root.mkdir(parents=True, exist_ok=True)
    # Materialize fixed manifests serially before concurrent shards access them.
    for task in tasks:
        fixed_manifest(task, Path(args.raw_root) / f'smi_{task}.csv',
                       Path(args.split_root) / f'{task}.json')
    jobs = [(task, fold) for task in tasks for fold in folds]
    skipped, partial = [], []
    if args.resume:
        pending = []
        for task, fold in jobs:
            unit = output / f'{task}_fold{fold}'
            if not unit.exists():
                pending.append((task, fold))
                continue
            summary_path = unit / 'summary.json'
            complete = False
            if summary_path.is_file():
                complete = _completed_unit_matches(unit, task, fold, args.smoke)
            if complete:
                skipped.append(f'{task}_fold{fold}')
            else:
                partial.append(str(unit))
        if partial:
            raise RuntimeError('partial grid units present; inspect before retrying: '
                               + ', '.join(partial))
        jobs = pending
    def launch(task, fold, gpu):
        unit = output / f'{task}_fold{fold}'
        log_path = log_root / f'{task}_fold{fold}.log'
        command = _build_shard_command(args, task, fold, unit)
        handle = log_path.open('w', encoding='utf-8')
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
        handle.write('COMMAND=' + ' '.join(command) + '\nCUDA_VISIBLE_DEVICES=' + gpu + '\n')
        handle.flush()
        process = subprocess.Popen(command, cwd=Path(__file__).resolve().parents[1],
                                   env=env, stdout=handle, stderr=subprocess.STDOUT)
        return process, handle, log_path

    completed = (_run_batched_jobs(jobs, gpus, launch) if args.batch_wait
                 else _run_dynamic_jobs(jobs, gpus, launch))
    print({'status': 'PASS', 'scheduler': 'batched' if args.batch_wait else 'dynamic',
           'smoke': bool(args.smoke), 'completed': completed, 'count': len(completed),
           'skipped_existing': skipped})


if __name__ == '__main__':
    main()
