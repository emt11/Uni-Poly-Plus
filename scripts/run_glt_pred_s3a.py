#!/usr/bin/env python3
"""Run the bounded GLT-PRED S3a development adaptation matrix.

This is deliberately a small controller for the fixed 3-task x 2-fold matrix;
it is not a general training workflow.  Every child is validation-only
(``--development``), and a child failure stops new dispatch while already
running children are reaped and recorded.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]


def _gpu_slots(values):
    gpus = [str(value) for value in values]
    if not 1 <= len(gpus) <= 4:
        raise ValueError('S3a supports one to four GPU slots')
    if len(set(gpus)) != len(gpus):
        raise ValueError('S3a requires distinct GPU slots')
    return gpus


def _alpha_token(alpha):
    return str(alpha).replace('.', 'p')


def _unit_root(output, policy, task, fold, alpha=None):
    name = f'{task}_fold{int(fold)}'
    if alpha is not None:
        name += f'_alpha{_alpha_token(alpha)}'
    return Path(output) / 'runs' / policy / name


def _jobs(output, tasks, folds, adaptations, ridge_alphas):
    jobs = []
    for policy in adaptations:
        for task in tasks:
            for fold in folds:
                if policy == 'ridge':
                    for alpha in ridge_alphas:
                        jobs.append(dict(policy=policy, task=task, fold=int(fold),
                                         alpha=float(alpha),
                                         unit=_unit_root(output, policy, task, fold, alpha)))
                else:
                    jobs.append(dict(policy=policy, task=task, fold=int(fold),
                                     alpha=None,
                                     unit=_unit_root(output, policy, task, fold)))
    return jobs


def _command(args, job):
    command = [
        sys.executable, 'scripts/finetune_glt_dual.py',
        '--config', args.config, '--checkpoint', args.checkpoint,
        '--raw-root', args.raw_root, '--split-root', args.split_root,
        '--cohort-root', args.cohort_root, '--cache-root', args.cache_root,
        '--dual-static-root', args.dual_static_root,
        '--task', job['task'], '--fold', str(job['fold']), '--development',
        '--adaptation', job['policy'], '--clean-cache-gib', format(args.clean_cache_gib, 'g'),
        '--timing', '--output', str(job['unit']),
    ]
    if job['policy'] == 'lora':
        command.extend(['--lora-rank', str(args.lora_rank),
                        '--lora-alpha', format(args.lora_alpha, 'g'),
                        '--lora-dropout', format(args.lora_dropout, 'g')])
    elif job['policy'] == 'ridge':
        command.extend(['--ridge-alpha', format(job['alpha'], 'g')])
    return command


def _start(args, job, gpu, log_path):
    command = _command(args, job)
    handle = Path(log_path).open('w', encoding='utf-8')
    handle.write('COMMAND=' + ' '.join(command) + '\n')
    handle.write('CUDA_VISIBLE_DEVICES=' + str(gpu) + '\n')
    handle.flush()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
    process = subprocess.Popen(command, cwd=ROOT, env=env,
                               stdout=handle, stderr=subprocess.STDOUT)
    return process, handle, command


def _run(args, jobs, gpus, log_root):
    pending = list(jobs)
    active = {}
    free = list(gpus)
    completed, failures = [], []
    while pending and free:
        job = pending.pop(0)
        gpu = free.pop(0)
        log_path = Path(log_root) / (
            f"{job['policy']}_{job['task']}_fold{job['fold']}"
            + (f"_alpha{_alpha_token(job['alpha'])}" if job['alpha'] is not None else '')
            + '.log')
        started = time.monotonic()
        process, handle, command = _start(args, job, gpu, log_path)
        active[gpu] = dict(job=job, gpu=gpu, process=process, handle=handle,
                           log=str(log_path), command=command, started=started)

    while active:
        observed = False
        for gpu, item in list(active.items()):
            code = item['process'].poll()
            if code is None:
                continue
            item['observed'] = time.monotonic()
            item['process'].wait()
            item['handle'].write(
                f"S3A_EXIT_OBSERVED_MONOTONIC={item['observed']}\n"
                f"S3A_WALL_SECONDS={item['observed'] - item['started']}\n"
                f"EXIT_CODE={code}\n")
            item['handle'].close()
            row = {
                'policy': item['job']['policy'], 'task': item['job']['task'],
                'fold': int(item['job']['fold']), 'alpha': item['job']['alpha'],
                'unit': str(item['job']['unit']), 'gpu': str(gpu), 'log': item['log'],
                'exit_code': int(code),
                'wall_seconds': float(item['observed'] - item['started']),
            }
            if code != 0:
                failures.append(row)
            else:
                completed.append(row)
            del active[gpu]
            free.append(gpu)
            observed = True
        if failures:
            # No new unit is dispatched after the first failure.  Remaining
            # children finish so their logs and exit codes are preserved.
            if active:
                time.sleep(0.5)
            continue
        while pending and free:
            job = pending.pop(0)
            gpu = free.pop(0)
            log_path = Path(log_root) / (
                f"{job['policy']}_{job['task']}_fold{job['fold']}"
                + (f"_alpha{_alpha_token(job['alpha'])}" if job['alpha'] is not None else '')
                + '.log')
            started = time.monotonic()
            process, handle, command = _start(args, job, gpu, log_path)
            active[gpu] = dict(job=job, gpu=gpu, process=process, handle=handle,
                               log=str(log_path), command=command, started=started)
        if active and not observed:
            time.sleep(0.5)
    return completed, failures, pending


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--raw-root', required=True)
    parser.add_argument('--split-root', required=True)
    parser.add_argument('--cohort-root', required=True)
    parser.add_argument('--cache-root', required=True)
    parser.add_argument('--dual-static-root', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--log-root', required=True)
    parser.add_argument('--gpu', action='append', dest='gpus', required=True)
    parser.add_argument('--task', action='append', dest='tasks',
                        default=['xc', 'eps', 'eat'])
    parser.add_argument('--fold', action='append', type=int, dest='folds',
                        default=[0, 1])
    parser.add_argument('--adaptation', action='append', dest='adaptations',
                        default=['full', 'head', 'lora', 'ridge'])
    parser.add_argument('--ridge-alpha', action='append', type=float,
                        dest='ridge_alphas', default=[0.1, 1.0, 10.0, 100.0])
    parser.add_argument('--clean-cache-gib', type=float, default=4.0)
    parser.add_argument('--lora-rank', type=int, default=8)
    parser.add_argument('--lora-alpha', type=float, default=8.0)
    parser.add_argument('--lora-dropout', type=float, default=0.0)
    args = parser.parse_args()
    gpus = _gpu_slots(args.gpus)
    if len(set(args.tasks)) != len(args.tasks) or set(args.tasks) != {'xc', 'eps', 'eat'}:
        raise ValueError('S3a tasks must be exactly xc, eps and eat')
    if sorted(set(args.folds)) != [0, 1]:
        raise ValueError('S3a folds must be exactly 0 and 1')
    if set(args.adaptations) != {'full', 'head', 'lora', 'ridge'}:
        raise ValueError('S3a adaptations must be full, head, lora and ridge')
    if sorted(float(x) for x in args.ridge_alphas) != [0.1, 1.0, 10.0, 100.0]:
        raise ValueError('S3a Ridge alphas must be exactly 0.1, 1, 10 and 100')
    output = Path(args.output).resolve()
    if not output.is_dir():
        raise ValueError('S3a output root must already contain pre-registration')
    allowed = {'pre_registration.json', 'preflight.json'}
    existing = {path.name for path in output.iterdir()}
    if not existing.issubset(allowed):
        raise ValueError('S3a output root contains unexpected prior artifacts')
    log_root = Path(args.log_root).resolve()
    log_root.mkdir(parents=True, exist_ok=True)
    jobs = _jobs(output, args.tasks, args.folds, args.adaptations, args.ridge_alphas)
    if len(jobs) != 42:
        raise RuntimeError(f'unexpected S3a job count: {len(jobs)}')
    for job in jobs:
        if job['unit'].exists():
            raise FileExistsError(f'unit output already exists: {job["unit"]}')
    (output / 'controller_command.json').write_text(
        json.dumps({'command': sys.argv, 'jobs': [
            {'policy': job['policy'], 'task': job['task'], 'fold': job['fold'],
             'alpha': job['alpha'], 'unit': str(job['unit'])} for job in jobs]},
                   indent=2, sort_keys=True) + '\n', encoding='utf-8')
    completed, failures, pending = _run(args, jobs, gpus, log_root)
    report = {
        'status': 'PASS' if not failures and not pending else 'FAIL',
        'expected_units': 42,
        'completed_units': len(completed),
        'failure_count': len(failures),
        'pending_units': len(pending),
        'gpus': gpus,
        'completed': completed,
        'failures': failures,
        'pending': [{'policy': job['policy'], 'task': job['task'],
                     'fold': job['fold'], 'alpha': job['alpha']}
                    for job in pending],
    }
    (output / 'controller.json').write_text(
        json.dumps(report, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2, sort_keys=True))
    if report['status'] != 'PASS':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
