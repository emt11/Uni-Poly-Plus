#!/usr/bin/env python3
"""Aggregate the MCL-PH P2 development units (MCL-PH-20260921-01/r10, P2).

Five arms x three tasks x two folds = 30 units, every one trained at
``stage=development`` from the P2 pre-training step (5000).  Unit validation is
the P1 aggregator's own checker (``scripts.aggregate_mcl_ph.check_unit``): this
tool never implements a second, weaker copy of it.

Only when the complete 30-unit set is accepted does the tool report

* each arm's per-task fold R^2, per-task mean and three-task macro mean;
* the matched deltas of the three MCL arms against O8_ONLY and GLT_REF;
* the pre-registered qualification gate, applied simultaneously against both
  baselines;
* the parent selection of the plan's section 10 (CAT is the default, GATE/XATTN
  may replace it only under the stated conditions, a Macro3 tie within 0.002 is
  an engineering tie and prefers GATE, an empty qualified set is
  ``NO_QUALIFIED_ARM``).

An incomplete set is reported as ``INCOMPLETE`` and computes no winner.  These
deltas are an engineering screening, not a significance test, and the outer test
is never read (every unit must record ``outer_test = NOT_RUN``).
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.aggregate_mcl_ph import check_unit

PLAN = 'MCL-PH-20260921-01/r10'
ARMS = ('glt_ref', 'o8_only', 'm_cat', 'm_gate', 'm_xattn')
TASKS = ('xc', 'eps', 'eat')
FOLDS = (0, 1)
STAGE = 'development'
CANDIDATE_ARMS = ('m_cat', 'm_gate', 'm_xattn')
REFERENCE_ARMS = ('o8_only', 'glt_ref')

# Section 9: a candidate must clear every clause against both baselines at once.
MACRO3_MIN_DELTA = 0.005
XC_MEAN_MIN_DELTA = 0.01
TASK_SACRIFICE_FLOOR = -0.01
# Section 10: replacing CAT, and the engineering tie band.
REPLACEMENT_MACRO3_MIN_DELTA = 0.002
TIE_MACRO3_DELTA = 0.002


def unit_list(arms=ARMS, tasks=TASKS, folds=FOLDS):
    return [(arm, task, fold) for arm in arms for task in tasks for fold in folds]


def collect(root, *, expected_step, arms=ARMS, tasks=TASKS, folds=FOLDS):
    """Run the shared unit checker over the requested scope."""
    accepted, rejected = [], []
    for arm, task, fold in unit_list(arms, tasks, folds):
        record, problems = check_unit(root, arm, task, fold, stage=STAGE,
                                      expected_step=expected_step)
        if problems:
            rejected.append({'arm': arm, 'task': task, 'fold': int(fold),
                             'problems': problems})
            continue
        runtime = json.loads((Path(root) / arm / task / f'fold{fold}'
                              / 'runtime.json').read_text(encoding='utf-8'))
        record = dict(record, status=runtime.get('status'),
                      exit_code=runtime.get('exit_code'))
        accepted.append(record)
    return accepted, rejected


def _mean(values):
    return sum(values) / len(values)


def arm_table(records):
    """Per arm: per-task fold R^2 + mean, and the macro mean over the tasks."""
    table = {}
    for arm in ARMS:
        rows = {(record['arm'], record['task'], record['fold']): record
                for record in records if record['arm'] == arm}
        entry = {}
        for task in TASKS:
            folds = {f'fold{fold}': float(rows[(arm, task, fold)]['best_validation_r2'])
                     for fold in FOLDS}
            entry[task] = dict(folds, mean=_mean(list(folds.values())))
        entry['macro3'] = _mean([entry[task]['mean'] for task in TASKS])
        table[arm] = entry
    return table


def matched_deltas(table):
    """Section 8: candidate minus baseline, never absolute R^2 alone."""
    deltas = {}
    for arm in CANDIDATE_ARMS:
        entry = {}
        for baseline in REFERENCE_ARMS:
            candidate, reference = table[arm], table[baseline]
            entry[f'vs_{baseline}'] = {
                'xc_fold0': candidate['xc']['fold0'] - reference['xc']['fold0'],
                'xc_fold1': candidate['xc']['fold1'] - reference['xc']['fold1'],
                'xc_mean': candidate['xc']['mean'] - reference['xc']['mean'],
                'eps_mean': candidate['eps']['mean'] - reference['eps']['mean'],
                'eat_mean': candidate['eat']['mean'] - reference['eat']['mean'],
                'macro3': candidate['macro3'] - reference['macro3'],
            }
        deltas[arm] = entry
    return deltas


def qualification(deltas):
    """Section 9: every clause holds against both baselines, or NOT_QUALIFIED."""
    report = {}
    for arm in CANDIDATE_ARMS:
        per_baseline, passed = {}, []
        for baseline in REFERENCE_ARMS:
            delta = deltas[arm][f'vs_{baseline}']
            checks = {
                'macro3': delta['macro3'] >= MACRO3_MIN_DELTA,
                'xc_mean': delta['xc_mean'] >= XC_MEAN_MIN_DELTA,
                'xc_fold0_positive': delta['xc_fold0'] > 0,
                'xc_fold1_positive': delta['xc_fold1'] > 0,
                # Locked contract: a task mean may drop by at most 0.01, so the
                # floor is inclusive -- exactly -0.01 still qualifies.
                'no_task_sacrifice': all(delta[f'{task}_mean'] >= TASK_SACRIFICE_FLOOR
                                         for task in TASKS),
            }
            checks['qualified'] = all(checks.values())
            per_baseline[f'vs_{baseline}'] = checks
            passed.append(checks['qualified'])
        report[arm] = dict(per_baseline, qualified=all(passed))
    return report


def _tie_preference(candidates, table):
    """Highest Macro3; within the tie band prefer GATE (simpler than XATTN)."""
    best = max(table[arm]['macro3'] for arm in candidates)
    tied = [arm for arm in candidates if best - table[arm]['macro3'] < TIE_MACRO3_DELTA]
    if 'm_gate' in tied:
        return 'm_gate', sorted(tied)
    return max(tied, key=lambda arm: table[arm]['macro3']), sorted(tied)


def selection(report, table):
    """Section 10: CAT by default, GATE/XATTN only on the stated conditions."""
    qualified = [arm for arm in CANDIDATE_ARMS if report[arm]['qualified']]
    payload = {'qualified_arms': qualified, 'replacement': {},
               'selection_status': None, 'selected_parent': None, 'reason': None}
    if not qualified:
        payload.update(selection_status='NO_QUALIFIED_ARM', selected_parent=None,
                       reason='no MCL arm clears the gate against both baselines; STOP')
        return payload
    if 'm_cat' in qualified:
        challengers = []
        for arm in ('m_gate', 'm_xattn'):
            if arm not in qualified:
                payload['replacement'][arm] = {'eligible': False,
                                               'reason': 'not qualified'}
                continue
            checks = {
                'macro3': (table[arm]['macro3'] - table['m_cat']['macro3'])
                          >= REPLACEMENT_MACRO3_MIN_DELTA,
                'xc_fold0_positive': (table[arm]['xc']['fold0'] - table['m_cat']['xc']['fold0']) > 0,
                'xc_fold1_positive': (table[arm]['xc']['fold1'] - table['m_cat']['xc']['fold1']) > 0,
            }
            checks['eligible'] = all(checks.values())
            payload['replacement'][arm] = checks
            if checks['eligible']:
                challengers.append(arm)
        if not challengers:
            payload.update(selection_status='SELECTED', selected_parent='m_cat',
                           reason='CAT is qualified and no challenger meets the '
                                  'replacement conditions')
            return payload
        chosen, tied = _tie_preference(challengers, table)
        payload.update(selection_status='SELECTED', selected_parent=chosen,
                       reason=f'{chosen} replaces CAT; macro3 tie band among '
                              f'{tied}, GATE preferred when inside it')
        return payload
    chosen, tied = _tie_preference(qualified, table)
    payload.update(selection_status='SELECTED', selected_parent=chosen,
                   reason=f'CAT is not qualified; selected from {qualified} '
                          f'(macro3 tie band {tied})')
    return payload


def aggregate(root, *, expected_step, arms=ARMS, tasks=TASKS, folds=FOLDS):
    accepted, rejected = collect(root, expected_step=expected_step, arms=arms,
                                 tasks=tasks, folds=folds)
    units = unit_list(arms, tasks, folds)
    scope = (list(arms) == list(ARMS) and list(tasks) == list(TASKS)
             and list(folds) == list(FOLDS))
    if rejected:
        status = 'INCOMPLETE'
    elif scope and len(accepted) == len(units):
        status = 'PASS'
    else:
        status = 'PARTIAL'
    payload = {
        'plan': PLAN, 'phase': 'P2', 'stage': STAGE, 'status': status,
        'acceptance': status, 'root': str(Path(root).resolve()),
        'acceptance_arms': list(ARMS), 'requested_arms': list(arms),
        'acceptance_tasks': list(TASKS), 'requested_tasks': list(tasks),
        'acceptance_folds': list(FOLDS), 'requested_folds': list(folds),
        'units_expected': len(units), 'units_accepted': len(accepted),
        'units_rejected': len(rejected), 'accepted': accepted, 'rejected': rejected,
        'outer_test': 'NOT_RUN', 'r2': None, 'deltas': None, 'qualification': None,
        'selection': {'selection_status': 'INCOMPLETE', 'selected_parent': None,
                      'reason': 'the 30-unit set is not complete; no winner is computed'},
        'note': 'P2 development screening only; the outer test is NOT_RUN and these '
                'deltas are not a significance test',
    }
    if status != 'PASS':
        if status == 'PARTIAL':
            payload['partial_reason'] = ('the requested arm/task/fold set is smaller '
                                         'than the declared P2 scope')
        return payload
    table = arm_table(accepted)
    deltas = matched_deltas(table)
    report = qualification(deltas)
    payload['r2'] = table
    payload['deltas'] = deltas
    payload['qualification'] = report
    payload['selection'] = selection(report, table)
    payload['boundary_risk_units'] = [
        {'arm': record['arm'], 'task': record['task'], 'fold': record['fold']}
        for record in accepted if record['best_epoch'] == record['executed_epochs']]
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--expected-pretrain-step', type=int, required=True)
    parser.add_argument('--arms', nargs='+', default=list(ARMS))
    parser.add_argument('--tasks', nargs='+', default=list(TASKS))
    parser.add_argument('--folds', nargs='+', type=int, default=list(FOLDS))
    parser.add_argument('--output')
    args = parser.parse_args()
    payload = aggregate(args.root, expected_step=args.expected_pretrain_step,
                        arms=args.arms, tasks=args.tasks, folds=args.folds)
    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + '\n'
    destination = Path(args.output) if args.output else Path(args.root) / 'p2_results.json'
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(text, encoding='utf-8')
    print(text, end='')
    if payload['status'] == 'INCOMPLETE':
        raise SystemExit(4)
    if payload['status'] == 'PARTIAL':
        raise SystemExit(5)


if __name__ == '__main__':
    main()
