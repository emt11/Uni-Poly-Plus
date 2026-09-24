#!/usr/bin/env python3
"""Four independent GPU workers for the authorized MCL-PH 8x5 units.

Each unit keeps its original single-GPU training contract.  Workers own
disjoint units and output paths; any failure stops scheduling further units.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
from threading import Event

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.aggregate_mcl_ph import check_unit
from scripts.aggregate_mcl_ph_8x5 import aggregate
from scripts.finetune_mcl_ph import ARMS, PERIODIC_TDL_EPOCHS, unit_directory
from scripts.run_mcl_ph_8x5 import accepted_from_prior, parse_packages, unit_list
from src.training.glt_dual_runtime import require_tmux, sha256_file


def distribute(units, devices=(0, 1, 2, 3)):
    if len(devices) != 4 or len(set(devices)) != 4:
        raise ValueError('four distinct physical GPUs are required')
    return {device: units[index::4] for index, device in enumerate(devices)}


def run_worker(device, units, args, output, log_root, packages, stopped):
    stage = 'full8x5_outer' if args.evaluation == 'outer_test' else 'full8x5'
    try:
        for arm, task, fold in units:
            if stopped.is_set():
                return
            name = f'{arm}_{task}_fold{fold}'
            command = [sys.executable, 'scripts/finetune_mcl_ph.py',
                       '--arm', arm, '--stage', stage, '--config', args.config,
                       '--checkpoint', str(packages[arm]), '--expected-pretrain-step', '5000',
                       '--cohort-root', args.cohort_root, '--cache-root', args.cache_root,
                       '--dual-static-root', args.dual_static_root,
                       '--split-root', args.split_root, '--statistics', args.statistics,
                       '--cohort-index', args.cohort_index, '--task', task,
                       '--fold', str(fold),
                       '--epochs', str(10 + PERIODIC_TDL_EPOCHS.get(task, 60)
                                       if args.finetune_strategy == 'periodic_tdl' else 30),
                       '--finetune-strategy', args.finetune_strategy,
                       '--output', str(output)]
            environment = os.environ.copy()
            environment['CUDA_VISIBLE_DEVICES'] = str(device)
            with (log_root / f'{name}.log').open('x', encoding='utf-8') as stream:
                stream.write(json.dumps({'command': command, 'physical_gpu': device,
                                         'CUDA_VISIBLE_DEVICES': str(device)}) + '\n')
                stream.flush()
                result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT,
                                        env=environment, check=False)
            (log_root / f'{name}.exit').write_text(f'{result.returncode}\n', encoding='utf-8')
            if result.returncode != 0:
                stopped.set()
                raise RuntimeError(f'{name} on GPU {device} exited {result.returncode}')
            _, problems = check_unit(output, arm, task, fold, stage=stage,
                                     expected_step=5000)
            if problems:
                (log_root / f'{name}.acceptance.json').write_text(
                    json.dumps({'status': 'FAILED', 'problems': problems}, indent=2) + '\n',
                    encoding='utf-8')
                stopped.set()
                raise RuntimeError(f'{name} on GPU {device} failed acceptance: {problems}')
            print(json.dumps({'unit': name, 'physical_gpu': device, 'status': 'PASS'}),
                  flush=True)
    except BaseException:
        stopped.set()
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arms', nargs='+', choices=ARMS, required=True)
    parser.add_argument('--package', action='append', required=True)
    for name in ('config', 'cohort-root', 'cache-root', 'dual-static-root', 'split-root',
                 'statistics', 'cohort-index', 'output', 'log-root'):
        parser.add_argument(f'--{name}', required=True)
    parser.add_argument('--reuse-root')
    parser.add_argument('--reuse-log-root')
    parser.add_argument('--retry-unit')
    parser.add_argument('--finetune-strategy', choices=('legacy', 'periodic_tdl'),
                        default='legacy')
    parser.add_argument('--evaluation', choices=('internal', 'outer_test'), default='internal')
    args = parser.parse_args()
    require_tmux()
    if args.evaluation == 'outer_test' and args.finetune_strategy != 'periodic_tdl':
        raise ValueError('outer-test evaluation requires periodic_tdl')
    if args.evaluation == 'outer_test' and (args.reuse_root or args.reuse_log_root):
        raise ValueError('outer-test campaign must train all units in a fresh root')
    units = unit_list(args.arms)
    packages = parse_packages(args.package, args.arms)
    output, log_root = Path(args.output).resolve(), Path(args.log_root).resolve()
    if bool(args.reuse_root) != bool(args.reuse_log_root):
        raise ValueError('reuse root and reuse log root must be supplied together')
    if args.retry_unit and not args.reuse_root:
        raise ValueError('retry unit requires a prior campaign')
    old_root = Path(args.reuse_root).resolve() if args.reuse_root else None
    old_logs = Path(args.reuse_log_root).resolve() if args.reuse_log_root else None
    if output.exists() or log_root.exists():
        raise FileExistsError('full8x5 output and log roots must both be new')
    for path in (args.config, args.statistics, args.cohort_index):
        if not Path(path).is_file():
            raise FileNotFoundError(path)
    hashes = {arm: sha256_file(path) for arm, path in packages.items()}
    accepted = (accepted_from_prior(old_root, old_logs, units, packages, hashes,
                                    args.retry_unit, args.finetune_strategy)
                if old_root else {})
    remaining = [(arm, task, fold) for arm, task, fold in units
                 if f'{arm}_{task}_fold{fold}' not in accepted]
    assignment = distribute(remaining)
    output.mkdir(parents=True, exist_ok=False)
    log_root.mkdir(parents=True, exist_ok=False)
    (log_root / 'launch.json').write_text(json.dumps({
        'arms': args.arms, 'units_expected': len(units),
        'epochs_per_unit_max': (70 if args.finetune_strategy == 'periodic_tdl' else 30),
        'finetune_strategy': args.finetune_strategy,
        'outer_test': 'RUN' if args.evaluation == 'outer_test' else 'NOT_RUN',
        'evaluation': args.evaluation, 'worker_count': 4,
        'packages': {arm: {'path': str(path), 'sha256': hashes[arm]}
                     for arm, path in packages.items()},
        'cohort_index': str(Path(args.cohort_index).resolve()),
        'reuse_root': str(old_root) if old_root else None,
        'reuse_log_root': str(old_logs) if old_logs else None,
        'reused_units': sorted(accepted), 'retry_unit': args.retry_unit,
        'gpu_assignments': {str(device): [f'{a}_{t}_fold{f}' for a, t, f in rows]
                            for device, rows in assignment.items()},
    }, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    for arm, task, fold in units:
        name = f'{arm}_{task}_fold{fold}'
        if name in accepted:
            target = unit_directory(output, arm, task, fold)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.symlink_to(accepted[name], target_is_directory=True)
    print(json.dumps({'reused': len(accepted), 'new': len(remaining),
                      'workers': 4}), flush=True)
    stopped = Event()
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(run_worker, device, rows, args, output, log_root,
                               packages, stopped)
                   for device, rows in assignment.items()]
        failures = []
        for future in futures:
            try:
                future.result()
            except BaseException as error:
                stopped.set()
                failures.append(str(error))
    if failures:
        raise RuntimeError('full8x5 workers stopped: ' + '; '.join(failures))
    stage = 'full8x5_outer' if args.evaluation == 'outer_test' else 'full8x5'
    payload = aggregate(output, arms=args.arms, expected_step=5000, stage=stage)
    filename = 'full8x5_test.json' if args.evaluation == 'outer_test' else 'full8x5_validation.json'
    (output / filename).write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + '\n',
        encoding='utf-8')
    if payload['status'] != 'PASS':
        raise SystemExit(4)
    print(json.dumps({'status': 'PASS', 'units_accepted': payload['units_accepted'],
                      'outer_test': payload['outer_test']}), flush=True)


if __name__ == '__main__':
    main()
