#!/usr/bin/env python3
"""Run isolated formal task/fold shards in a bounded GPU grid."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.finetune_glt_dual import TASKS, fixed_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'checkpoint', 'raw-root', 'cohort-root', 'cache-root', 'dual-static-root', 'output'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--split-root', default='data/splits/mips_outer5_inner20')
    parser.add_argument('--task', action='append', dest='tasks')
    parser.add_argument('--gpu', action='append', dest='gpus', required=True,
                        help='one CUDA device per concurrent shard; repeat up to 4')
    parser.add_argument('--log-root', required=True)
    args = parser.parse_args()
    tasks = list(args.tasks) if args.tasks else list(TASKS)
    if sorted(set(tasks)) != sorted(tasks) or any(task not in TASKS for task in tasks):
        raise ValueError('unknown or duplicate task selection')
    gpus = [str(gpu) for gpu in args.gpus]
    if not 1 <= len(gpus) <= 4:
        raise ValueError('grid supports one to four GPU slots')
    output = Path(args.output).resolve()
    log_root = Path(args.log_root).resolve()
    output.mkdir(parents=True, exist_ok=False)
    log_root.mkdir(parents=True, exist_ok=True)
    # Materialize fixed manifests serially before concurrent shards access them.
    for task in tasks:
        fixed_manifest(task, Path(args.raw_root) / f'smi_{task}.csv',
                       Path(args.split_root) / f'{task}.json')
    jobs = [(task, fold) for task in tasks for fold in range(5)]
    completed = []
    for offset in range(0, len(jobs), len(gpus)):
        batch = jobs[offset:offset + len(gpus)]
        running = []
        for (task, fold), gpu in zip(batch, gpus):
            unit = output / f'{task}_fold{fold}'
            log_path = log_root / f'{task}_fold{fold}.log'
            command = [
                sys.executable, 'scripts/finetune_glt_dual.py',
                '--config', args.config, '--checkpoint', args.checkpoint,
                '--raw-root', args.raw_root, '--cohort-root', args.cohort_root,
                '--cache-root', args.cache_root, '--dual-static-root', args.dual_static_root,
                '--output', str(unit), '--split-root', args.split_root,
                '--task', task, '--fold', str(fold), '--formal-shard',
            ]
            handle = log_path.open('w', encoding='utf-8')
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
            handle.write('COMMAND=' + ' '.join(command) + '\nCUDA_VISIBLE_DEVICES=' + gpu + '\n')
            handle.flush()
            process = subprocess.Popen(command, cwd=Path(__file__).resolve().parents[1],
                                       env=env, stdout=handle, stderr=subprocess.STDOUT)
            running.append((task, fold, process, handle, log_path))
        for task, fold, process, handle, log_path in running:
            code = process.wait()
            handle.write(f'EXIT_CODE={code}\n')
            handle.close()
            if code != 0:
                raise RuntimeError(f'formal shard failed: task={task} fold={fold}; log={log_path}')
            completed.append({'task': task, 'fold': fold, 'log': str(log_path)})
    print({'status': 'PASS', 'completed': completed, 'count': len(completed)})


if __name__ == '__main__':
    main()
