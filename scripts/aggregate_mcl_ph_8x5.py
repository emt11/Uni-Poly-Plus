#!/usr/bin/env python3
"""Validate complete eight-task, five-fold MCL-PH adaptation per requested arm.

This reports inner-validation R2 only. It does not read outer-test labels or
predictions, select a P2 parent, or claim an independent blind-test result.
"""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.aggregate_mcl_ph import check_unit
from scripts.finetune_mcl_ph import ARMS, FOLDS, TASKS


def aggregate(root, *, arms, expected_step, stage='full8x5'):
    if stage not in ('full8x5', 'full8x5_outer'):
        raise ValueError('unknown five-fold aggregation stage')
    arms = tuple(arms)
    if not arms or len(set(arms)) != len(arms) or any(arm not in ARMS for arm in arms):
        raise ValueError('arms must be a nonempty unique subset of MCL-PH arms')
    accepted, rejected = [], []
    for arm in arms:
        for task in TASKS:
            for fold in FOLDS:
                record, problems = check_unit(root, arm, task, fold,
                                              stage=stage, expected_step=expected_step)
                if problems:
                    rejected.append({'arm': arm, 'task': task, 'fold': fold,
                                     'problems': problems})
                else:
                    accepted.append(record)
    status = 'PASS' if not rejected else 'INCOMPLETE'
    payload = {
        'stage': stage, 'status': status, 'root': str(Path(root).resolve()),
        'arms': list(arms), 'tasks': list(TASKS), 'folds': list(FOLDS),
        'units_expected': len(arms) * len(TASKS) * len(FOLDS),
        'units_accepted': len(accepted), 'units_rejected': len(rejected),
        'accepted': accepted, 'rejected': rejected,
        'outer_test': 'RUN' if stage == 'full8x5_outer' else 'NOT_RUN',
        'validation_r2': None, 'test_r2': None,
    }
    if rejected:
        return payload
    strategies = {record.get('finetune_strategy', 'legacy') for record in accepted}
    if len(strategies) != 1:
        payload['status'] = 'INCOMPLETE'
        payload['rejected'].append({'problems': ['mixed fine-tuning strategies']})
        payload['units_rejected'] += 1
        return payload
    payload['finetune_strategy'] = strategies.pop()
    if stage == 'full8x5_outer' and payload['finetune_strategy'] != 'periodic_tdl':
        payload['status'] = 'INCOMPLETE'
        payload['rejected'].append({'problems': ['outer-test requires periodic_tdl']})
        payload['units_rejected'] += 1
        return payload
    table = {}
    test_table = {}
    for arm in arms:
        packages = {record['pretrain_package_sha256'] for record in accepted
                    if record['arm'] == arm}
        if len(packages) != 1:
            payload['status'] = 'INCOMPLETE'
            payload['rejected'].append({'arm': arm, 'problems': ['pretrain package SHA varies']})
            payload['units_rejected'] += 1
            return payload
        task_rows = {}
        for task in TASKS:
            folds = [next(record['best_validation_r2'] for record in accepted
                          if record['arm'] == arm and record['task'] == task
                          and record['fold'] == fold) for fold in FOLDS]
            task_rows[task] = {'folds': folds, 'mean': sum(folds) / len(folds)}
        table[arm] = {
            'tasks': task_rows,
            'macro8': sum(row['mean'] for row in task_rows.values()) / len(TASKS),
            'pretrain_package_sha256': packages.pop(),
        }
        if stage == 'full8x5_outer':
            test_rows = {}
            for task in TASKS:
                folds = [next(record['test_r2'] for record in accepted
                              if record['arm'] == arm and record['task'] == task
                              and record['fold'] == fold) for fold in FOLDS]
                mean = sum(folds) / len(folds)
                test_rows[task] = {'folds': folds, 'mean': mean,
                                   'std': (sum((value - mean) ** 2 for value in folds)
                                           / len(folds)) ** 0.5}
            test_table[arm] = {'tasks': test_rows,
                               'macro8': sum(row['mean'] for row in test_rows.values()) / len(TASKS),
                               'pretrain_package_sha256': table[arm]['pretrain_package_sha256']}
    payload['validation_r2'] = table
    if stage == 'full8x5_outer':
        payload['test_r2'] = test_table
        payload['std_definition'] = 'population standard deviation over five fold R2 values'
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--arms', nargs='+', choices=ARMS, required=True)
    parser.add_argument('--expected-pretrain-step', type=int, required=True)
    parser.add_argument('--stage', choices=('full8x5', 'full8x5_outer'), default='full8x5')
    parser.add_argument('--output')
    args = parser.parse_args()
    payload = aggregate(args.root, arms=args.arms,
                        expected_step=args.expected_pretrain_step, stage=args.stage)
    filename = 'full8x5_test.json' if args.stage == 'full8x5_outer' else 'full8x5_validation.json'
    destination = Path(args.output) if args.output else Path(args.root) / filename
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True,
                                      allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({'status': payload['status'], 'units_accepted': payload['units_accepted'],
                      'units_expected': payload['units_expected']}))
    if payload['status'] != 'PASS':
        raise SystemExit(4)


if __name__ == '__main__':
    main()
