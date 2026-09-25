#!/usr/bin/env python3
"""Aggregate the five tasks aligned to Periodic-TDL's released five folds."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.aggregate_mcl_ph import check_unit
from scripts.finetune_mcl_ph import ARMS, FOLDS, PAPER_TASKS


def aggregate(root, *, arms, expected_step, stage='paper5_outer'):
    if stage != 'paper5_outer':
        raise ValueError('only paper5_outer aggregation is active')
    arms = tuple(arms)
    if not arms or len(set(arms)) != len(arms) or any(arm not in ARMS for arm in arms):
        raise ValueError('arms must be a nonempty unique subset of MCL-PH arms')
    accepted, rejected = [], []
    for arm in arms:
        for task in PAPER_TASKS:
            for fold in FOLDS:
                record, problems = check_unit(root, arm, task, fold,
                                              stage='paper5_outer', expected_step=expected_step)
                if problems:
                    rejected.append({'arm': arm, 'task': task, 'fold': fold,
                                     'problems': problems})
                else:
                    accepted.append(record)
    payload = {
        'stage': 'paper5_outer', 'status': 'PASS' if not rejected else 'INCOMPLETE',
        'root': str(Path(root).resolve()), 'arms': list(arms),
        'tasks': list(PAPER_TASKS), 'folds': list(FOLDS),
        'units_expected': len(arms) * len(PAPER_TASKS) * len(FOLDS),
        'units_accepted': len(accepted), 'units_rejected': len(rejected),
        'accepted': accepted, 'rejected': rejected, 'outer_test': 'RUN',
        'validation_r2': None, 'test_r2': None,
        'comparability_scope': ('released Periodic-TDL row identity, official outer folds, '
                                'released-code inner split; project model and geometry'),
    }
    if rejected:
        return payload
    if {record['finetune_strategy'] for record in accepted} != {'periodic_tdl'}:
        payload['status'] = 'INCOMPLETE'
        payload['rejected'].append({'problems': ['fine-tuning strategy differs']})
        payload['units_rejected'] += 1
        return payload
    payload['finetune_strategy'] = 'periodic_tdl'
    validation_table, test_table = {}, {}
    for arm in arms:
        arm_rows = [row for row in accepted if row['arm'] == arm]
        packages = {row['pretrain_package_sha256'] for row in arm_rows}
        if len(packages) != 1:
            payload['status'] = 'INCOMPLETE'
            payload['rejected'].append({'arm': arm, 'problems': ['pretrain package SHA varies']})
            payload['units_rejected'] += 1
            return payload
        validation_tasks, test_tasks = {}, {}
        for task in PAPER_TASKS:
            rows = [next(row for row in arm_rows if row['task'] == task and
                         row['fold'] == fold) for fold in FOLDS]
            validation = [row['best_validation_r2'] for row in rows]
            tests = [row['test_r2'] for row in rows]
            mean = sum(tests) / len(tests)
            validation_tasks[task] = {'folds': validation,
                                      'mean': sum(validation) / len(validation)}
            test_tasks[task] = {'folds': tests, 'mean': mean,
                                'std': (sum((value - mean)**2 for value in tests)
                                        / len(tests))**.5}
        package_sha = packages.pop()
        validation_table[arm] = {
            'tasks': validation_tasks,
            'macro5': sum(row['mean'] for row in validation_tasks.values()) / len(PAPER_TASKS),
            'pretrain_package_sha256': package_sha,
        }
        test_table[arm] = {
            'tasks': test_tasks,
            'macro5': sum(row['mean'] for row in test_tasks.values()) / len(PAPER_TASKS),
            'pretrain_package_sha256': package_sha,
        }
    payload['validation_r2'] = validation_table
    payload['test_r2'] = test_table
    payload['std_definition'] = 'population standard deviation over five fold R2 values'
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--arms', nargs='+', choices=ARMS, required=True)
    parser.add_argument('--expected-pretrain-step', type=int, required=True)
    parser.add_argument('--output')
    args = parser.parse_args()
    payload = aggregate(args.root, arms=args.arms,
                        expected_step=args.expected_pretrain_step)
    destination = Path(args.output) if args.output else Path(args.root) / 'paper5_test.json'
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True,
                                      allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({'status': payload['status'], 'units_accepted': payload['units_accepted'],
                      'units_expected': payload['units_expected']}))
    if payload['status'] != 'PASS':
        raise SystemExit(4)


if __name__ == '__main__':
    main()
