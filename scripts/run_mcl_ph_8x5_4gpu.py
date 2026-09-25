#!/usr/bin/env python3
"""Run only the paper-aligned MCL-PH five-task, five-fold campaign on four GPUs."""

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
from scripts.finetune_mcl_ph import (ARMS, FOLDS, PAPER_SPLIT_PROTOCOL,
                                     PAPER_TASKS, unit_directory)
from src.training.glt_dual_runtime import require_tmux, sha256_file


def parse_packages(specs):
    packages = {}
    for spec in specs:
        if '=' not in spec:
            raise ValueError('each --package must be arm=path')
        arm, path = spec.split('=', 1)
        if arm not in ARMS or arm in packages or not path:
            raise ValueError(f'undeclared or duplicate package arm: {arm}')
        packages[arm] = Path(path).resolve()
    if set(packages) != set(ARMS):
        raise ValueError('provide exactly one package for each of the five arms')
    for arm, path in packages.items():
        if not path.is_file():
            raise FileNotFoundError(f'{arm} package is missing: {path}')
    return packages


def unit_list():
    return [(arm, task, fold) for arm in ARMS for task in PAPER_TASKS for fold in FOLDS]


def distribute(units, devices=(0, 1, 2, 3)):
    if len(devices) != 4 or len(set(devices)) != 4:
        raise ValueError('four distinct physical GPUs are required')
    return {device: units[index::4] for index, device in enumerate(devices)}


def verify_splits(split_root, cohort_split_root):
    for task in PAPER_TASKS:
        path = Path(split_root) / f'{task}.json'
        manifest = json.loads(path.read_text(encoding='utf-8'))
        if manifest.get('protocol') != PAPER_SPLIT_PROTOCOL or manifest.get('task') != task:
            raise ValueError(f'{task}: not the official paper5 fold manifest')
        source = manifest['official_source']
        name = 'EPS' if task == 'eps' else task.capitalize()
        for filename, key in ((f'{name}_cleaned.csv', 'cleaned_csv_sha256'),
                              (f'{name}_folds.pkl', 'folds_pkl_sha256')):
            if sha256_file(Path(split_root) / 'source' / filename) != source[key]:
                raise ValueError(f'{task}: official source identity changed')
        old = json.loads((Path(cohort_split_root) / f'{task}.json').read_text())
        if (old.get('sample_order_sha256') != manifest.get('sample_order_sha256')
                or old.get('sample_count') != manifest.get('sample_count')):
            raise ValueError(f'{task}: official rows differ from frozen cohort rows')


def run_worker(device, units, args, output, log_root, packages, stopped):
    try:
        for arm, task, fold in units:
            if stopped.is_set():
                return
            name = f'{arm}_{task}_fold{fold}'
            command = [sys.executable, 'scripts/finetune_mcl_ph.py',
                       '--arm', arm, '--stage', 'paper5_outer', '--config', args.config,
                       '--checkpoint', str(packages[arm]), '--expected-pretrain-step', '5000',
                       '--cohort-root', args.cohort_root, '--cache-root', args.cache_root,
                       '--dual-static-root', args.dual_static_root,
                       '--split-root', args.split_root,
                       '--cohort-split-root', args.cohort_split_root,
                       '--statistics', args.statistics, '--cohort-index', args.cohort_index,
                       '--task', task, '--fold', str(fold), '--epochs', '70',
                       '--finetune-strategy', 'periodic_tdl', '--output', str(output)]
            environment = os.environ.copy()
            environment['CUDA_VISIBLE_DEVICES'] = str(device)
            with (log_root / f'{name}.log').open('x', encoding='utf-8') as stream:
                stream.write(json.dumps({'command': command, 'physical_gpu': device}) + '\n')
                stream.flush()
                result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT,
                                        env=environment, check=False)
            (log_root / f'{name}.exit').write_text(f'{result.returncode}\n', encoding='utf-8')
            if result.returncode != 0:
                stopped.set()
                raise RuntimeError(f'{name} on GPU {device} exited {result.returncode}')
            _, problems = check_unit(output, arm, task, fold, stage='paper5_outer',
                                     expected_step=5000)
            if problems:
                (log_root / f'{name}.acceptance.json').write_text(
                    json.dumps({'status': 'FAILED', 'problems': problems}, indent=2) + '\n')
                stopped.set()
                raise RuntimeError(f'{name} on GPU {device} failed acceptance: {problems}')
            print(json.dumps({'unit': name, 'physical_gpu': device, 'status': 'PASS'}),
                  flush=True)
    except BaseException:
        stopped.set()
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', action='append', required=True)
    for name in ('config', 'cohort-root', 'cache-root', 'dual-static-root', 'split-root',
                 'cohort-split-root', 'statistics', 'cohort-index', 'output', 'log-root'):
        parser.add_argument(f'--{name}', required=True)
    args = parser.parse_args()
    require_tmux()
    output, log_root = Path(args.output).resolve(), Path(args.log_root).resolve()
    if output.exists() or log_root.exists():
        raise FileExistsError('paper5 output and log roots must both be new')
    for path in (args.config, args.statistics, args.cohort_index):
        if not Path(path).is_file():
            raise FileNotFoundError(path)
    verify_splits(args.split_root, args.cohort_split_root)
    packages = parse_packages(args.package)
    hashes = {arm: sha256_file(path) for arm, path in packages.items()}
    units = unit_list()
    assignment = distribute(units)
    output.mkdir(parents=True, exist_ok=False)
    log_root.mkdir(parents=True, exist_ok=False)
    (log_root / 'launch.json').write_text(json.dumps({
        'arms': ARMS, 'tasks': PAPER_TASKS, 'folds': FOLDS,
        'units_expected': len(units), 'epochs_per_unit_max': 70,
        'finetune_strategy': 'periodic_tdl', 'stage': 'paper5_outer',
        'outer_test': 'RUN', 'worker_count': 4,
        'packages': {arm: {'path': str(path), 'sha256': hashes[arm]}
                     for arm, path in packages.items()},
        'split_root': str(Path(args.split_root).resolve()),
        'cohort_split_root': str(Path(args.cohort_split_root).resolve()),
        'cohort_index': str(Path(args.cohort_index).resolve()),
        'reused_units': [],
        'gpu_assignments': {str(device): [f'{a}_{t}_fold{f}' for a, t, f in rows]
                            for device, rows in assignment.items()},
    }, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    print(json.dumps({'reused': 0, 'new': len(units), 'workers': 4}), flush=True)
    stopped = Event()
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(run_worker, device, rows, args, output, log_root,
                               packages, stopped) for device, rows in assignment.items()]
        failures = []
        for future in futures:
            try:
                future.result()
            except BaseException as error:
                stopped.set()
                failures.append(str(error))
    if failures:
        raise RuntimeError('paper5 workers stopped: ' + '; '.join(failures))
    payload = aggregate(output, arms=ARMS, expected_step=5000, stage='paper5_outer')
    (output / 'paper5_test.json').write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + '\n',
        encoding='utf-8')
    if payload['status'] != 'PASS':
        raise SystemExit(4)
    print(json.dumps({'status': 'PASS', 'units_accepted': payload['units_accepted'],
                      'outer_test': 'RUN'}), flush=True)


if __name__ == '__main__':
    main()
