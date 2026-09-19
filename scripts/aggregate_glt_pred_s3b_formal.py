#!/usr/bin/env python3
"""Aggregate GLT-PRED S3b formal development results and apply the fixed gates.

Reads the four arms' pretrain runtime and 24 development unit metrics,
computes paired validation-R2 deltas, the fixed XC/EPS/EAT development gate,
the FGR/ALIGN increment verdicts and SELECTED_PARENT.  Emits summary.json and
report.md; a read-only aggregation, it never touches checkpoints.
"""
import argparse
import json
from pathlib import Path

ARMS = ('b_fp', 'b_none', 't_fgr', 't_align')
TASKS = ('xc', 'eps', 'eat')
FOLDS = (0, 1)
CANDIDATES = {'t_fgr': 'FGR', 't_align': 'ALIGN'}


def mean(values):
    return sum(values) / len(values)


def load_unit(root, arm, task, fold):
    path = root / arm / 'development' / task / f'fold{fold}' / 'metrics.json'
    d = json.loads(path.read_text(encoding='utf-8'))
    assert d['protocol'] == 'outer5_inner20_development', (arm, task, fold, d['protocol'])
    assert d['development'] is True and d['validation_only'] is True, (arm, task, fold)
    assert d['outer_test'] == 'NOT_RUN', (arm, task, fold, d['outer_test'])
    return d


def gate(candidate, reference, r2):
    """Fixed development gate: XC mean >= +0.01, each XC fold >= -0.03,
    EPS/EAT mean >= -0.01.  Higher R2 is better, deltas are candidate - reference."""
    xc_mean = r2[candidate]['xc']['mean'] - r2[reference]['xc']['mean']
    xc_folds = {fold: r2[candidate]['xc']['folds'][fold] - r2[reference]['xc']['folds'][fold]
                for fold in FOLDS}
    eps_mean = r2[candidate]['eps']['mean'] - r2[reference]['eps']['mean']
    eat_mean = r2[candidate]['eat']['mean'] - r2[reference]['eat']['mean']
    checks = {
        'xc_mean_delta': round(xc_mean, 6),
        'xc_mean_delta_ge_0.01': bool(xc_mean >= 0.01),
        'xc_fold_deltas': {str(f): round(v, 6) for f, v in xc_folds.items()},
        'xc_each_fold_ge_-0.03': bool(all(v >= -0.03 for v in xc_folds.values())),
        'eps_mean_delta': round(eps_mean, 6),
        'eps_mean_delta_ge_-0.01': bool(eps_mean >= -0.01),
        'eat_mean_delta': round(eat_mean, 6),
        'eat_mean_delta_ge_-0.01': bool(eat_mean >= -0.01),
    }
    checks['GATE'] = 'PASS' if (checks['xc_mean_delta_ge_0.01']
                                and checks['xc_each_fold_ge_-0.03']
                                and checks['eps_mean_delta_ge_-0.01']
                                and checks['eat_mean_delta_ge_-0.01']) else 'FAIL'
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='results/glt_pred_20260918/s3b_formal')
    args = parser.parse_args()
    root = Path(args.root)

    summary = {'pretrain': {}, 'units': {}, 'r2_validation': {}, 'gates': {},
               'increment': {}, 'selected_parent': None, 's3b_verdict': None}
    for arm in ARMS:
        runtime = json.loads((root / arm / 'pretrain' / 'runtime.json').read_text())
        summary['pretrain'][arm] = {
            'status': runtime['status'],
            'completed_steps': int(runtime['completed_steps']),
            'deploy_05000': str(root / arm / 'pretrain' / 'deploy_05000.pt'),
        }
        units = {}
        r2 = {}
        for task in TASKS:
            folds = {}
            for fold in FOLDS:
                d = load_unit(root, arm, task, fold)
                units[f'{task}_fold{fold}'] = {
                    'best_validation_r2': round(float(d['best_validation_r2']), 6),
                    'validation_mae': round(float(d['validation_mae']), 6),
                    'validation_rmse': round(float(d['validation_rmse']), 6),
                    'best_epoch': int(d['best_epoch']),
                    'wall_seconds': round(float(d['wall_seconds']), 1),
                    'trainable_parameter_count': int(d['trainable_parameter_count']),
                    'outer_test': d['outer_test'],
                }
                folds[fold] = float(d['best_validation_r2'])
            r2[task] = {'folds': folds, 'mean': mean(list(folds.values()))}
        summary['units'][arm] = units
        summary['r2_validation'][arm] = {
            task: {'fold0': round(r2[task]['folds'][0], 6),
                   'fold1': round(r2[task]['folds'][1], 6),
                   'mean': round(r2[task]['mean'], 6)}
            for task in TASKS}

    r2 = {arm: {task: {'folds': {0: summary['r2_validation'][arm][task]['fold0'],
                                 1: summary['r2_validation'][arm][task]['fold1']},
                       'mean': summary['r2_validation'][arm][task]['mean']}
                for task in TASKS}
          for arm in ARMS}

    comparisons = {
        'B_NONE_VS_B_FP': ('b_none', 'b_fp'),
        'T_FGR_VS_B_FP': ('t_fgr', 'b_fp'),
        'T_FGR_VS_B_NONE': ('t_fgr', 'b_none'),
        'T_ALIGN_VS_B_FP': ('t_align', 'b_fp'),
        'T_ALIGN_VS_B_NONE': ('t_align', 'b_none'),
    }
    for name, (cand, ref) in comparisons.items():
        summary['gates'][name] = gate(cand, ref, r2)

    for cand, label in CANDIDATES.items():
        vs_fp = summary['gates'][{'t_fgr': 'T_FGR_VS_B_FP', 't_align': 'T_ALIGN_VS_B_FP'}[cand]]
        vs_none = summary['gates'][{'t_fgr': 'T_FGR_VS_B_NONE', 't_align': 'T_ALIGN_VS_B_NONE'}[cand]]
        summary['increment'][f'{label}_INCREMENT'] = (
            'ESTABLISHED' if vs_fp['GATE'] == 'PASS' and vs_none['GATE'] == 'PASS'
            else 'NOT_ESTABLISHED')

    fgr_ok = summary['increment']['FGR_INCREMENT'] == 'ESTABLISHED'
    align_ok = summary['increment']['ALIGN_INCREMENT'] == 'ESTABLISHED'
    none_vs_fp_pass = summary['gates']['B_NONE_VS_B_FP']['GATE'] == 'PASS'
    if fgr_ok and align_ok:
        # Case B: higher mean XC validation R2 wins; if very close (<0.005),
        # compare the worst XC fold delta vs B_NONE; then EPS/EAT protection.
        xc_f, xc_a = r2['t_fgr']['xc']['mean'], r2['t_align']['xc']['mean']
        if abs(xc_f - xc_a) >= 0.005:
            pick = 't_fgr' if xc_f > xc_a else 't_align'
            reason = 'higher mean XC validation R2'
        else:
            worst_f = min(r2['t_fgr']['xc']['folds'][f] - r2['b_none']['xc']['folds'][f]
                          for f in FOLDS)
            worst_a = min(r2['t_align']['xc']['folds'][f] - r2['b_none']['xc']['folds'][f]
                          for f in FOLDS)
            if abs(worst_f - worst_a) > 1e-9:
                pick = 't_fgr' if worst_f > worst_a else 't_align'
                reason = 'worst XC fold delta vs B_NONE (XC means within 0.005)'
            else:
                prot_f = min(r2['t_fgr']['eps']['mean'] - r2['b_none']['eps']['mean'],
                             r2['t_fgr']['eat']['mean'] - r2['b_none']['eat']['mean'])
                prot_a = min(r2['t_align']['eps']['mean'] - r2['b_none']['eps']['mean'],
                             r2['t_align']['eat']['mean'] - r2['b_none']['eat']['mean'])
                pick = 't_fgr' if prot_f >= prot_a else 't_align'
                reason = 'EPS/EAT protection vs B_NONE (XC and worst-fold tied)'
        summary['selected_parent'] = pick
        summary['s3b_verdict'] = f'SELECTED_CANDIDATE ({reason})'
    elif fgr_ok or align_ok:
        summary['selected_parent'] = 't_fgr' if fgr_ok else 't_align'
        summary['s3b_verdict'] = 'SELECTED_CANDIDATE (only one candidate passed both gates)'
    elif none_vs_fp_pass:
        summary['selected_parent'] = 'b_none'
        summary['s3b_verdict'] = 'NEGATIVE / INCONCLUSIVE (B_NONE passes the gate vs B_FP)'
    else:
        summary['selected_parent'] = 'b_fp'
        summary['s3b_verdict'] = 'NEGATIVE / INCONCLUSIVE (B_NONE fails the gate vs B_FP)'

    out = root / 'summary.json'
    out.write_text(json.dumps(summary, indent=1, sort_keys=False) + '\n', encoding='utf-8')

    lines = ['# S3b 正式结果汇总（development validation R² 口径）', '',
             '## 预训练状态', '', '| arm | status | completed_steps |', '| --- | --- | ---: |']
    for arm in ARMS:
        p = summary['pretrain'][arm]
        lines.append(f"| {arm} | {p['status']} | {p['completed_steps']} |")
    lines += ['', '## 每 task R²（fold0 / fold1 / mean）', '',
              '| arm | XC | EPS | EAT |', '| --- | --- | --- | --- |']
    for arm in ARMS:
        cells = [' / '.join(str(summary['r2_validation'][arm][t][k]) for k in ('fold0', 'fold1', 'mean'))
                 for t in TASKS]
        lines.append(f"| {arm} | {cells[0]} | {cells[1]} | {cells[2]} |")
    lines += ['', '## 固定 development gate（candidate − reference，R²）', '']
    for name, checks in summary['gates'].items():
        lines.append(f"- **{name}**: GATE={checks['GATE']} "
                     f"(XC mean Δ={checks['xc_mean_delta']}, folds Δ={checks['xc_fold_deltas']}, "
                     f"EPS Δ={checks['eps_mean_delta']}, EAT Δ={checks['eat_mean_delta']})")
    lines += ['', '## 晋级判定', '',
              f"- FGR_INCREMENT = {summary['increment']['FGR_INCREMENT']}",
              f"- ALIGN_INCREMENT = {summary['increment']['ALIGN_INCREMENT']}",
              f"- SELECTED_PARENT = **{summary['selected_parent']}**",
              f"- 判定说明: {summary['s3b_verdict']}", '']
    (root / 'report.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps({'summary': str(out), 'report': str(root / 'report.md'),
                      'gates': {k: v['GATE'] for k, v in summary['gates'].items()},
                      'increment': summary['increment'],
                      'selected_parent': summary['selected_parent']}, indent=1))


if __name__ == '__main__':
    main()
