#!/usr/bin/env python3
"""r2 analysis for GLT-3D-GAIN-20260921-01.

Separates three effects that the r2 rerun introduced, without refitting
anything:

1. **numerical only**    r1 (default solver, five alphas) vs r2 (SVD + float64,
                         same five alphas);
2. **grid only**         r2 five-point sub-grid vs r2 nine-point grid;
3. **combined**          r1 vs r2 nine-point.

Also reports the per-unit alpha sweep, boundary hits, the paired differences
and the fixed D1 gate.  Reads the two frozen JSON results; no data, no model.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

TASKS = ('xc', 'eps', 'eat')
FOLDS = (0, 1)
FAMILIES = ('L2', 'L3', 'L23', 'L23-S')
FIVE = (0.01, 0.1, 1.0, 10.0, 100.0)


def key(alpha):
    return repr(float(alpha))


def pick(candidates, alphas):
    """Best alpha restricted to a sub-grid, with its boundary status."""
    available = [(alpha, candidates[key(alpha)]['validation_r2'])
                 for alpha in alphas if key(alpha) in candidates]
    if not available:
        raise ValueError('sub-grid is not contained in the candidate set')
    best = max(available, key=lambda item: item[1])
    ordered = sorted(alpha for alpha, _ in available)
    boundary = 'interior'
    if float(best[0]) == float(ordered[0]):
        boundary = 'lower_bound'
    elif float(best[0]) == float(ordered[-1]):
        boundary = 'upper_bound'
    return {'selected_alpha': float(best[0]), 'validation_r2': float(best[1]),
            'boundary': boundary}


def per_task_macro(values):
    """values: {unit_key: float} -> per-task means and their macro average."""
    by_task = {}
    for task in TASKS:
        folds = [values[f'{task}/fold{fold}'] for fold in FOLDS if f'{task}/fold{fold}' in values]
        by_task[task] = float(np.mean(folds)) if folds else None
    present = [value for value in by_task.values() if value is not None]
    return {'per_task': by_task, 'macro3': float(np.mean(present)) if present else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--r1', required=True)
    parser.add_argument('--r2', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    r1 = json.loads(Path(args.r1).read_text(encoding='utf-8'))
    r2 = json.loads(Path(args.r2).read_text(encoding='utf-8'))
    alphas_nine = tuple(float(value) for value in r2['alphas'])
    if tuple(sorted(float(value) for value in r1['alphas'])) != FIVE:
        raise ValueError('r1 alpha set is not the expected five-point grid')

    report = {'r1_path': str(Path(args.r1).resolve()), 'r2_path': str(Path(args.r2).resolve()),
              'alphas_r1': list(FIVE), 'alphas_r2': list(alphas_nine), 'units': {}}

    r1_unit = {}
    r2_unit = {}
    r2_five = {}
    for task in TASKS:
        for fold in FOLDS:
            unit_key = f'{task}/fold{fold}'
            source = r1['units'][unit_key]
            target = r2['units'][unit_key]
            entry = {'sweep': {}, 'comparison': {}}
            current_five = {}
            for name in FAMILIES + ('residual',):
                candidates_r2 = target[name]['candidates']
                sweep = {str(alpha): {
                    'validation_r2': candidates_r2[key(alpha)]['validation_r2'],
                    'in_r1_grid': key(alpha) in source[name]['candidates'],
                    'r1_validation_r2': (source[name]['candidates'][key(alpha)]['validation_r2']
                                         if key(alpha) in source[name]['candidates'] else None)}
                    for alpha in alphas_nine}
                five = pick(candidates_r2, FIVE)
                nine = pick(candidates_r2, alphas_nine)
                entry['sweep'][name] = {
                    'per_alpha': sweep,
                    'r2_nine': nine, 'r2_five_subgrid': five,
                    'r1_five': {'selected_alpha': source[name]['selected_alpha'],
                                'validation_r2': source[name]['validation_r2']},
                }
                if name == 'residual':
                    baseline_r2 = target[name]['baseline_validation_r2']
                    baseline_r1 = source[name]['baseline_validation_r2']
                    entry['comparison']['residual'] = {
                        'r1_delta': source[name]['delta_vs_alpha1_baseline'],
                        'r2_five_delta': five['validation_r2'] - baseline_r2,
                        'r2_nine_delta': nine['validation_r2'] - baseline_r2,
                        'r1_baseline_r2': baseline_r1,
                        'r2_baseline_r2': baseline_r2,
                    }
                else:
                    entry['comparison'][name] = {
                        'r1_r2': source[name]['validation_r2'],
                        'r2_five_r2': five['validation_r2'],
                        'r2_nine_r2': nine['validation_r2'],
                        'numerical_only_delta': five['validation_r2'] - source[name]['validation_r2'],
                        'grid_only_delta': nine['validation_r2'] - five['validation_r2'],
                        'combined_delta': nine['validation_r2'] - source[name]['validation_r2'],
                    }
            current_five[unit_key] = entry
            r1_unit[unit_key] = {name: source[name]['validation_r2'] for name in FAMILIES}
            r2_unit[unit_key] = {name: target[name]['validation_r2'] for name in FAMILIES}
            r2_five[unit_key] = {name: entry['sweep'][name]['r2_five_subgrid']['validation_r2']
                                 for name in FAMILIES}
            report['units'][unit_key] = entry

    # Paired differences: L23-L2 and L23-L23S under each protocol.
    report['paired'] = {}
    for label, values in (('r1', r1_unit), ('r2_five', r2_five), ('r2_nine', r2_unit)):
        l23_minus_l2 = {unit: values[unit]['L23'] - values[unit]['L2'] for unit in values}
        l23_minus_l23s = {unit: values[unit]['L23'] - values[unit]['L23-S'] for unit in values}
        report['paired'][label] = {
            'L23_minus_L2': {'per_unit': l23_minus_l2, **per_task_macro(l23_minus_l2)},
            'L23_minus_L23S': {'per_unit': l23_minus_l23s, **per_task_macro(l23_minus_l23s)},
        }
    for label in ('r1', 'r2_five', 'r2_nine'):
        delta = {}
        for task in TASKS:
            for fold in FOLDS:
                unit = f'{task}/fold{fold}'
                if label == 'r1':
                    delta[unit] = r1['units'][unit]['residual']['delta_vs_alpha1_baseline']
                    continue
                entry = report['units'][unit]['sweep']['residual']
                selected = (entry['r2_five_subgrid'] if label == 'r2_five'
                            else entry['r2_nine'])
                baseline = r2['units'][unit]['residual']['baseline_validation_r2']
                delta[unit] = selected['validation_r2'] - baseline
        report['paired'][label]['residual'] = {'per_unit': delta, **per_task_macro(delta)}

    # Boundary accounting across the four probe families.
    boundaries = {}
    for label, alphas in (('r2_five', FIVE), ('r2_nine', alphas_nine)):
        counts = {}
        for name in FAMILIES + ('residual',):
            hits = []
            for unit in report['units']:
                selected = (report['units'][unit]['sweep'][name]['r2_five_subgrid']
                            if label == 'r2_five' else
                            report['units'][unit]['sweep'][name]['r2_nine'])
                hits.append(selected['boundary'])
            counts[name] = {'upper_bound': hits.count('upper_bound'),
                            'lower_bound': hits.count('lower_bound'),
                            'interior': hits.count('interior')}
        boundaries[label] = counts
    report['boundaries'] = boundaries

    # The D1 gate is fixed in advance and is not adjusted here.
    for label in ('r1', 'r2_five', 'r2_nine'):
        residual = report['paired'][label]['residual']
        l23 = report['paired'][label]['L23_minus_L2']
        xc_residual = [residual['per_unit'][f'xc/fold{fold}'] for fold in FOLDS]
        xc_l23 = [l23['per_unit'][f'xc/fold{fold}'] for fold in FOLDS]
        report.setdefault('gate', {})[label] = {
            'residual_macro3': residual['macro3'],
            'residual_xc_both_positive': bool(all(value > 0 for value in xc_residual)),
            'residual_pass': bool(residual['macro3'] >= 0.005
                                  and all(value > 0 for value in xc_residual)),
            'L23_minus_L2_macro3': l23['macro3'],
            'L23_minus_L2_xc_both_positive': bool(all(value > 0 for value in xc_l23)),
            'L23_minus_L2_pass': bool(l23['macro3'] >= 0.005
                                      and all(value > 0 for value in xc_l23)),
        }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({'status': 'PASS', 'output': str(output),
                      'gate': report['gate']}, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
