#!/usr/bin/env python3
"""Bounded GPU scheduler for the 24 D2 development units (GLT-3D-GAIN-20260921-01/r6).

One subprocess per arm/task/fold unit, at most one unit per GPU slot at a time,
in the same shape as the existing GLT grids.  The unit list is fixed and written
to ``schedule.json`` before anything runs: an existing schedule file or an
existing unit directory stops the launch rather than being reused.

A unit failure stops the whole grid from dispatching further units (units
already running finish, so their epochs are still accounted for) and the
scheduler exits with that unit's own exit code.  Nothing is retried here.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.finetune_glt_3d_gain_d2 import (FOLDS, SCHEDULE_TOTAL_EPOCHS, STAGE_EPOCH_LIMIT,
                                            TASKS, ARMS, unit_directory)

DEFAULT_CONFIG = 'configs/mts/glt_pred_s3b_b_fp.json'
DEFAULT_CHECKPOINT = 'results/glt_pred_20260918/s3b_formal/b_fp/pretrain/deploy_05000.pt'
DEFAULT_COHORT = 'data/processed/glt_dual_v2/downstream/cohort_1545eda5a8f6a1'
DEFAULT_CACHE = 'data/processed/mips_trimer_scage_downstream'
DEFAULT_STATIC = 'data/processed/glt_dual_v2/downstream/dual_static_v1'
DEFAULT_SPLIT = 'data/splits/mips_outer5_inner20'


def unit_list():
    """The 24 development units, in the fixed order arms -> tasks -> folds."""
    return [(arm, task, fold) for arm in ARMS for task in TASKS for fold in FOLDS]


def validate_gpus(gpus):
    values = [str(value) for value in gpus]
    if not 1 <= len(values) <= 4:
        raise ValueError('the development grid supports one to four GPU slots')
    if len(set(values)) != len(values):
        raise ValueError('GPU slots must be distinct')
    return values


def unit_command(args, arm, task, fold):
    return [sys.executable, 'scripts/finetune_glt_3d_gain_d2.py',
            '--arm', arm, '--stage', 'development', '--epochs', str(args.epochs),
            '--config', args.config, '--checkpoint', args.checkpoint,
            '--cohort-root', args.cohort_root, '--cache-root', args.cache_root,
            '--dual-static-root', args.dual_static_root, '--split-root', args.split_root,
            '--task', task, '--fold', str(fold), '--output', str(args.output)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpus', nargs='+', required=True)
    parser.add_argument('--config', default=DEFAULT_CONFIG)
    parser.add_argument('--checkpoint', default=DEFAULT_CHECKPOINT)
    parser.add_argument('--cohort-root', default=DEFAULT_COHORT)
    parser.add_argument('--cache-root', default=DEFAULT_CACHE)
    parser.add_argument('--dual-static-root', default=DEFAULT_STATIC)
    parser.add_argument('--split-root', default=DEFAULT_SPLIT)
    parser.add_argument('--output', required=True)
    parser.add_argument('--log-root', required=True)
    parser.add_argument('--epochs', type=int, default=SCHEDULE_TOTAL_EPOCHS)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    gpus = validate_gpus(args.gpus)
    if not 1 <= args.epochs <= STAGE_EPOCH_LIMIT['development']:
        raise ValueError(f'development allows 1..{STAGE_EPOCH_LIMIT["development"]} epochs')
    args.output = Path(args.output)
    args.log_root = Path(args.log_root)

    units = unit_list()
    schedule = {'protocol': 'glt_3d_gain_d2_development', 'epochs_per_unit': args.epochs,
                'patience_from_config': True, 'gpu_slots': gpus, 'unit_count': len(units),
                'checkpoint': args.checkpoint, 'config': args.config,
                'split_root': args.split_root, 'output_root': str(args.output),
                'units': [{'arm': arm, 'task': task, 'fold': fold,
                           'directory': str(unit_directory(args.output, arm, task, fold)),
                           'slot': index % len(gpus)}
                          for index, (arm, task, fold) in enumerate(units)]}
    if args.dry_run:
        print(json.dumps(schedule, ensure_ascii=False, indent=2))
        return

    schedule_path = args.output / 'schedule.json'
    if schedule_path.exists():
        raise FileExistsError(f'{schedule_path} already exists; this grid never reuses one')
    existing = [entry['directory'] for entry in schedule['units']
                if Path(entry['directory']).exists()]
    if existing:
        raise FileExistsError('unit directories already exist: ' + ','.join(existing[:4]))
    args.output.mkdir(parents=True, exist_ok=True)
    args.log_root.mkdir(parents=True, exist_ok=True)
    schedule_path.write_text(json.dumps(schedule, ensure_ascii=False, indent=2), encoding='utf-8')

    started = time.perf_counter()
    stop = threading.Event()
    results = {}

    def run_chain(slot, gpu, chain):
        for arm, task, fold in chain:
            entry = unit_directory(args.output, arm, task, fold)
            log_path = args.log_root / f'{arm}_{task}_fold{fold}.log'
            if stop.is_set():
                results[(arm, task, fold)] = {'arm': arm, 'task': task, 'fold': fold,
                                              'status': 'SKIPPED_AFTER_FAILURE'}
                continue
            command = unit_command(args, arm, task, fold)
            environment = {**os.environ, 'CUDA_VISIBLE_DEVICES': gpu}
            begin = time.perf_counter()
            with log_path.open('a', encoding='utf-8') as log:
                log.write(f'=== SLOT={slot} GPU={gpu} ARM={arm} TASK={task} FOLD={fold} '
                          f'START {time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())} ===\n')
                log.write('COMMAND ' + ' '.join(command) + '\n')
                log.flush()
                code = subprocess.run(command, env=environment, stdout=log,
                                      stderr=subprocess.STDOUT).returncode
                log.write(f'=== SLOT={slot} EXIT={code} SECONDS={time.perf_counter() - begin:.1f} '
                          f'{time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())} ===\n')
            results[(arm, task, fold)] = {
                'arm': arm, 'task': task, 'fold': fold, 'slot': slot, 'gpu': gpu,
                'status': 'PASS' if code == 0 else 'FAILED', 'exit_code': int(code),
                'log': str(log_path), 'directory': str(entry),
                'seconds': float(time.perf_counter() - begin)}
            print(json.dumps({**results[(arm, task, fold)], 'slot': slot}), flush=True)
            if code != 0:
                stop.set()

    chains = {index: [] for index in range(len(gpus))}
    for index, unit in enumerate(units):
        chains[index % len(gpus)].append(unit)
    threads = [threading.Thread(target=run_chain, args=(index, gpus[index], chains[index]))
               for index in range(len(gpus))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    ordered = [results.get(unit, {'arm': unit[0], 'task': unit[1], 'fold': unit[2],
                                  'status': 'NOT_DISPATCHED'}) for unit in units]
    failures = [record for record in ordered if record['status'] != 'PASS']
    runtime = {'protocol': 'glt_3d_gain_d2_development', 'gpu_slots': gpus,
               'scheduled_units': len(units), 'completed_units':
                   sum(1 for record in ordered if record['status'] == 'PASS'),
               'failed_units': len(failures), 'all_units_pass': not failures,
               'units': ordered, 'exit_code': int(failures[0]['exit_code'])
                   if failures and 'exit_code' in failures[0] else (0 if not failures else 1),
               'wall_seconds': float(time.perf_counter() - started)}
    (args.output / 'grid_runtime.json').write_text(
        json.dumps(runtime, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'status': 'PASS' if not failures else 'FAILED',
                      'completed': runtime['completed_units'],
                      'failed': runtime['failed_units'],
                      'wall_seconds': round(runtime['wall_seconds'], 1)}, ensure_ascii=False))
    if failures:
        raise SystemExit(runtime['exit_code'])


if __name__ == '__main__':
    main()
