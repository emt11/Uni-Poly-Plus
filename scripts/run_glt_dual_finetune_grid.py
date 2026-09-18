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

    def drain_active():
        """Wait for already-owned children and close their log handles."""
        for gpu, (task, fold, process, handle, log_path) in list(active.items()):
            code = process.wait()
            handle.write(f'EXIT_CODE={code}\n')
            handle.close()
            completed.append({'task': task, 'fold': fold, 'log': str(log_path),
                              'exit_code': code})
            del active[gpu]

    def start_available():
        while pending and free and not failures:
            task, fold = pending.pop(0)
            gpu = free.pop(0)
            try:
                process, handle, log_path = launch(task, fold, gpu)
            except BaseException:
                free.insert(0, gpu)
                drain_active()
                raise
            active[gpu] = (task, fold, process, handle, log_path)

    start_available()
    while active:
        finished = []
        for gpu, (task, fold, process, handle, log_path) in list(active.items()):
            code = process.poll()
            if code is None:
                continue
            handle.write(f'EXIT_CODE={code}\n')
            handle.close()
            del active[gpu]
            free.append(gpu)
            finished.append(gpu)
            if code != 0:
                failures.append({'task': task, 'fold': fold, 'exit_code': code,
                                 'log': str(log_path)})
            else:
                completed.append({'task': task, 'fold': fold, 'log': str(log_path)})
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


def _run_batched_jobs(jobs, gpus, launch):
    """Reference scheduler: wait for a whole GPU batch before dispatching more."""

    completed = []
    for offset in range(0, len(jobs), len(gpus)):
        active = []
        try:
            for task, fold in jobs[offset:offset + len(gpus)]:
                gpu = gpus[len(active)]
                process, handle, log_path = launch(task, fold, gpu)
                active.append((task, fold, process, handle, log_path))
        except BaseException:
            for _, _, process, handle, _ in active:
                code = process.wait()
                handle.write(f'EXIT_CODE={code}\n')
                handle.close()
            raise
        failures = []
        for task, fold, process, handle, log_path in active:
            code = process.wait()
            handle.write(f'EXIT_CODE={code}\n')
            handle.close()
            row = {'task': task, 'fold': fold, 'log': str(log_path), 'exit_code': code}
            if code:
                failures.append(row)
            else:
                completed.append(row)
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
                try:
                    complete = bool(json.loads(summary_path.read_text(encoding='utf-8')).get('tasks'))
                except ValueError:
                    complete = False
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
