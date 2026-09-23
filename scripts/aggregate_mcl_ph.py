#!/usr/bin/env python3
"""Aggregate the MCL-PH downstream units (MCL-PH-20260921-01/r1).

A unit counts as complete only when every record it owns passes every check
below.  An existing directory, a trailing DONE line in a log, or a
``best.pt`` without its identity records is never accepted as evidence.
Partial products, non-finite metrics, a wrong stage/protocol/step, a smoke unit
reused as a development unit, and any unit whose ``outer_test`` is not
``NOT_RUN`` are rejected.  An incomplete set is reported as ``INCOMPLETE`` and
the process exits non-zero.
"""
import argparse
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from scripts.finetune_mcl_ph import (ARMS, FOLDS, MCL_FUSION, SCHEDULE_TOTAL_EPOCHS,
                                     SPLIT_PROTOCOL, TASKS, unit_directory)

REQUIRED_FILES = ('run.json', 'runtime.json', 'metrics.json', 'best.pt',
                  'validation_predictions.npz')
STAGE_EPOCH_LIMIT = {'smoke': 1, 'development': SCHEDULE_TOTAL_EPOCHS,
                     'full8x5': SCHEDULE_TOTAL_EPOCHS}
# The declared P1 smoke scope; anything smaller can be verified but is PARTIAL.
ACCEPTANCE_ARMS = tuple(ARMS)
ACCEPTANCE_TASKS = ('xc',)
ACCEPTANCE_FOLDS = (0,)


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def check_unit(root, arm, task, fold, *, stage, expected_step):
    """Validate one unit; returns (record, problems)."""
    problems = []
    directory = unit_directory(root, arm, task, fold)
    missing = [name for name in REQUIRED_FILES if not (directory / name).is_file()]
    if missing:
        return None, [f'missing artifacts: {",".join(missing)}']
    try:
        records = {name: json.loads((directory / name).read_text(encoding='utf-8'))
                   for name in REQUIRED_FILES if name.endswith('.json')}
    except (OSError, ValueError) as error:
        return None, [f'unreadable record: {type(error).__name__}: {error}']
    runtime, run, metrics = records['runtime.json'], records['run.json'], records['metrics.json']
    for name, record in (('runtime', runtime), ('run', run), ('metrics', metrics)):
        if record.get('arm') != arm or record.get('task') != task \
                or int(record.get('fold', -1)) != fold:
            problems.append(f'{name} identity does not match {arm}/{task}/fold{fold}')
    if runtime.get('status') != 'PASS':
        problems.append(f"runtime status is {runtime.get('status')!r}, not PASS")
    if runtime.get('exit_code') != 0:
        problems.append(f"runtime exit_code is {runtime.get('exit_code')!r}")
    for name, record in (('run', run), ('metrics', metrics)):
        if record.get('stage') != stage:
            problems.append(f"{name} stage is {record.get('stage')!r}, not {stage!r}")
        if record.get('protocol') != f'mcl_ph_{stage}':
            problems.append(f"{name} protocol is {record.get('protocol')!r}")
        if record.get('outer_test') != 'NOT_RUN':
            problems.append(f"{name} outer_test is {record.get('outer_test')!r}")
    if metrics.get('pretrain_step') != int(expected_step):
        problems.append(f"pretrain_step is {metrics.get('pretrain_step')!r}, "
                        f"expected {int(expected_step)}")
    if arm in MCL_FUSION and metrics.get('pretrained_route') != 'mcl_ph':
        problems.append('an MCL-PH arm must record the mcl_ph pre-training route')
    if arm in ('glt_ref', 'o8_only') and metrics.get('pretrained_route') != 'dual_glt':
        problems.append('a reference arm must record the dual_glt pre-training route')
    requested = metrics.get('requested_epochs')
    limit = STAGE_EPOCH_LIMIT[stage]
    if not isinstance(requested, int) or isinstance(requested, bool) \
            or not 1 <= requested <= limit:
        problems.append(f'requested_epochs {requested!r} is outside 1..{limit}')
    history = metrics.get('history')
    if not isinstance(history, list) or not history:
        problems.append('history is empty or not a list')
        history = []
    executed = metrics.get('executed_epochs')
    if executed != len(history):
        problems.append(f'executed_epochs {executed!r} != history length {len(history)}')
    if executed != run.get('executed_epochs'):
        problems.append('executed_epochs disagrees between run and metrics')
    steps = [record.get('training_steps') for record in history]
    if not all(isinstance(value, int) and value > 0 for value in steps):
        problems.append('history holds a non-positive training step count')
    elif metrics.get('optimizer_updates') != sum(steps):
        problems.append('optimizer_updates does not equal the sum of training steps')
    values = [record.get('train_loss') for record in history] + \
             [record.get('validation_loss') for record in history] + \
             [record.get('validation_r2') for record in history]
    if not all(_finite(value) for value in values):
        problems.append('a history entry is not finite')
    best_epoch = metrics.get('best_epoch')
    if not isinstance(best_epoch, int) or isinstance(best_epoch, bool) \
            or not 1 <= best_epoch <= max(1, len(history)):
        problems.append(f'best_epoch {best_epoch!r} is outside the executed epochs')
    elif not math.isclose(float(metrics.get('best_validation_r2', float('nan'))),
                          float(history[best_epoch - 1]['validation_r2']), rel_tol=0, abs_tol=1e-12):
        problems.append('best_validation_r2 disagrees with the selected epoch')
    elif history and abs(max(record['validation_r2'] for record in history)
                         - float(metrics['best_validation_r2'])) > 1e-12:
        problems.append('the selected epoch is not the best validation epoch')
    split = metrics.get('split') or {}
    if split.get('protocol') != SPLIT_PROTOCOL:
        problems.append(f"split protocol is {split.get('protocol')!r}")
    if split.get('validation_is_test') is not False:
        problems.append('the split does not declare validation_is_test=false')
    if split.get('outer_test') != 'NOT_RUN':
        problems.append('the split record does not keep outer_test NOT_RUN')
    for name in ('sets_disjoint', 'union_equals_full_cohort'):
        if split.get(name) is not True:
            problems.append(f'the split record does not assert {name}')
    if int(split.get('validation_rows', 0)) <= 0 or int(split.get('train_rows', 0)) <= 0:
        problems.append('the split record has an empty train or validation set')
    try:
        with np.load(directory / 'validation_predictions.npz', allow_pickle=False) as payload:
            truth = np.asarray(payload['y_true'], dtype=np.float64).reshape(-1)
            prediction = np.asarray(payload['y_pred'], dtype=np.float64).reshape(-1)
            keys = np.asarray(payload['sample_keys'])
            if truth.size != prediction.size or truth.size == 0:
                problems.append('validation predictions are empty or mismatched')
            elif not (np.isfinite(truth).all() and np.isfinite(prediction).all()):
                problems.append('validation predictions contain NaN/Inf')
            if keys.size != truth.size:
                problems.append('validation sample keys disagree with the predictions')
            if int(payload['best_epoch']) != best_epoch:
                problems.append('the saved predictions are not from the selected epoch')
            if str(payload['outer_test']) != 'NOT_RUN':
                problems.append('the saved predictions do not keep outer_test NOT_RUN')
    except (OSError, ValueError, KeyError) as error:
        problems.append(f'unreadable validation predictions: {type(error).__name__}: {error}')
    groups = metrics.get('optimizer_groups')
    if not isinstance(groups, list) or not groups:
        problems.append('the parameter groups were not recorded')
    elif sum(int(group.get('num_parameters', 0)) for group in groups) <= 0:
        problems.append('the parameter groups contain no trainable parameter')
    elif {group.get('name') for group in groups} - {'backbone', 'backbone_no_decay',
                                                    'head', 'head_no_decay'}:
        problems.append('the parameter groups use an undeclared name')
    record = {'arm': arm, 'task': task, 'fold': int(fold), 'stage': stage,
              'executed_epochs': executed, 'optimizer_updates': metrics.get('optimizer_updates'),
              'best_epoch': best_epoch, 'best_validation_r2': metrics.get('best_validation_r2'),
              'pretrain_step': metrics.get('pretrain_step'),
              'pretrain_package_sha256': metrics.get('pretrain_package_sha256')}
    return record, problems


def unit_list(arms, tasks, folds):
    return [(arm, task, fold) for arm in arms for task in tasks for fold in folds]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--stage', default='smoke', choices=tuple(STAGE_EPOCH_LIMIT))
    parser.add_argument('--expected-pretrain-step', type=int, required=True)
    parser.add_argument('--arms', nargs='+', default=list(ARMS))
    parser.add_argument('--tasks', nargs='+', default=list(ACCEPTANCE_TASKS))
    parser.add_argument('--folds', nargs='+', type=int, default=list(ACCEPTANCE_FOLDS))
    parser.add_argument('--output')
    args = parser.parse_args()
    if args.stage == 'development':
        raise SystemExit('development aggregation is not part of the P1 budget')
    units = unit_list(args.arms, args.tasks, args.folds)
    accepted, rejected = [], []
    for arm, task, fold in units:
        record, problems = check_unit(args.root, arm, task, fold, stage=args.stage,
                                      expected_step=args.expected_pretrain_step)
        if problems:
            rejected.append({'arm': arm, 'task': task, 'fold': int(fold), 'problems': problems})
        else:
            accepted.append(record)
    # P1 acceptance is fixed at every declared arm: a subset can be verified, but
    # it is reported as PARTIAL and never as acceptance, whatever it contains.
    scope = (list(args.arms) == list(ARMS) and list(args.tasks) == list(ACCEPTANCE_TASKS)
             and list(args.folds) == list(ACCEPTANCE_FOLDS))
    if rejected:
        status = 'INCOMPLETE'
    elif scope and len(accepted) == len(units):
        status = 'PASS'
    else:
        status = 'PARTIAL'
    payload = {'plan': 'MCL-PH-20260921-01/r2', 'phase': 'P1', 'stage': args.stage,
               'status': status, 'acceptance': status, 'root': str(Path(args.root).resolve()),
               'acceptance_arms': list(ARMS), 'requested_arms': list(args.arms),
               'acceptance_tasks': list(ACCEPTANCE_TASKS), 'requested_tasks': list(args.tasks),
               'acceptance_folds': list(ACCEPTANCE_FOLDS), 'requested_folds': list(args.folds),
               'units_expected': len(units), 'units_accepted': len(accepted),
               'units_rejected': len(rejected), 'accepted': accepted, 'rejected': rejected,
               'outer_test': 'NOT_RUN',
               'note': 'implementation availability only; these units do not rank the arms '
                       'and are not a performance comparison'}
    if status == 'PARTIAL':
        payload['partial_reason'] = ('the requested arm/task/fold set is smaller than the '
                                     'declared P1 acceptance scope')
    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + '\n'
    destination = Path(args.output) if args.output else Path(args.root) / 'aggregate.json'
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(text, encoding='utf-8')
    print(text, end='')
    if status == 'INCOMPLETE':
        raise SystemExit(4)
    if status == 'PARTIAL':
        raise SystemExit(5)


if __name__ == '__main__':
    main()
