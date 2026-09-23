#!/usr/bin/env python3
"""Serial, stop-on-failure launcher for authorized MCL-PH 8x5 adaptation.

The caller must explicitly select arms and fresh output/log roots. This tool
does not supply experiment authorization or run outer-test evaluation.
"""

import argparse
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.aggregate_mcl_ph import check_unit
from scripts.aggregate_mcl_ph_8x5 import aggregate
from scripts.finetune_mcl_ph import ARMS, FOLDS, TASKS
from src.training.glt_dual_runtime import require_tmux, sha256_file


def parse_packages(specs, arms):
    packages = {}
    for spec in specs:
        if '=' not in spec:
            raise ValueError('each --package must be arm=path')
        arm, path = spec.split('=', 1)
        if arm not in arms or arm in packages or not path:
            raise ValueError(f'undeclared or duplicate package arm: {arm}')
        packages[arm] = Path(path).resolve()
    if set(packages) != set(arms):
        raise ValueError('provide exactly one --package for every requested arm')
    for arm, path in packages.items():
        if not path.is_file():
            raise FileNotFoundError(f'{arm} package is missing: {path}')
    return packages


def unit_list(arms):
    arms = tuple(arms)
    if not arms or len(set(arms)) != len(arms) or any(arm not in ARMS for arm in arms):
        raise ValueError('requested arms must be a nonempty unique subset')
    return [(arm, task, fold) for arm in arms for task in TASKS for fold in FOLDS]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arms', nargs='+', choices=ARMS, required=True)
    parser.add_argument('--package', action='append', required=True, help='arm=deployment_path')
    parser.add_argument('--config', required=True)
    parser.add_argument('--cohort-root', required=True)
    parser.add_argument('--cache-root', required=True)
    parser.add_argument('--dual-static-root', required=True)
    parser.add_argument('--split-root', required=True)
    parser.add_argument('--statistics', required=True)
    parser.add_argument('--cohort-index', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--log-root', required=True)
    parser.add_argument('--expected-pretrain-step', type=int, default=5000)
    args = parser.parse_args()
    units = unit_list(args.arms)
    if int(args.expected_pretrain_step) != 5000:
        raise ValueError('the formal MCL-PH adaptation requires step 5000')
    packages = parse_packages(args.package, args.arms)
    require_tmux()
    output, log_root = Path(args.output).resolve(), Path(args.log_root).resolve()
    if output.exists() or log_root.exists():
        raise FileExistsError('full8x5 output and log roots must both be new')
    for path in (args.config, args.statistics, args.cohort_index):
        if not Path(path).is_file():
            raise FileNotFoundError(path)
    package_hashes = {arm: sha256_file(path) for arm, path in packages.items()}
    output.mkdir(parents=True, exist_ok=False)
    log_root.mkdir(parents=True, exist_ok=False)
    (log_root / 'launch.json').write_text(json.dumps({
        'arms': args.arms, 'units_expected': len(units), 'epochs_per_unit_max': 30,
        'outer_test': 'NOT_RUN',
        'packages': {arm: {'path': str(path), 'sha256': package_hashes[arm]}
                     for arm, path in packages.items()},
        'cohort_index': str(Path(args.cohort_index).resolve()),
    }, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    for arm, task, fold in units:
        unit = f'{arm}_{task}_fold{fold}'
        command = [sys.executable, 'scripts/finetune_mcl_ph.py',
                   '--arm', arm, '--stage', 'full8x5', '--config', args.config,
                   '--checkpoint', str(packages[arm]),
                   '--expected-pretrain-step', '5000',
                   '--cohort-root', args.cohort_root, '--cache-root', args.cache_root,
                   '--dual-static-root', args.dual_static_root,
                   '--split-root', args.split_root,
                   '--statistics', args.statistics, '--cohort-index', args.cohort_index,
                   '--task', task, '--fold', str(fold), '--epochs', '30',
                   '--output', str(output)]
        with (log_root / f'{unit}.log').open('x', encoding='utf-8') as stream:
            stream.write(json.dumps({'command': command}) + '\n')
            stream.flush()
            result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT,
                                    check=False)
        (log_root / f'{unit}.exit').write_text(f'{result.returncode}\n', encoding='utf-8')
        if result.returncode != 0:
            raise SystemExit(result.returncode)
        _, problems = check_unit(output, arm, task, fold, stage='full8x5',
                                 expected_step=5000)
        if problems:
            (log_root / f'{unit}.acceptance.json').write_text(
                json.dumps({'status': 'FAILED', 'problems': problems}, indent=2) + '\n',
                encoding='utf-8')
            raise SystemExit(4)
    payload = aggregate(output, arms=args.arms, expected_step=5000)
    (output / 'full8x5_validation.json').write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + '\n',
        encoding='utf-8')
    if payload['status'] != 'PASS':
        raise SystemExit(4)
    print(json.dumps({'status': 'PASS', 'units_accepted': payload['units_accepted'],
                      'outer_test': 'NOT_RUN'}))


if __name__ == '__main__':
    main()
