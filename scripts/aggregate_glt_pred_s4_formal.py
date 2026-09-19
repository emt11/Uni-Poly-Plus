#!/usr/bin/env python3
"""Aggregate GLT-PRED S4 (B_NONE + ENV/TORSION) development results.

Applies the same fixed development gate as S3b to
``S4_ENV - B_NONE``, ``S4_TOR_ON - S4_TOR_OFF`` and ``S4_TOR_ON - B_NONE``,
derives the increment verdicts and SELECTED_S4_PARENT, and writes
summary.json + report.md.  Read-only aggregation.
"""
import argparse
import json
from pathlib import Path

from scripts.aggregate_glt_pred_s3b_formal import FOLDS, TASKS, gate, mean

ARMS = ('s4_env', 's4_tor_off', 's4_tor_on')
REFERENCE = 'b_none'


def _command_checkpoint(root, arm):
    """The --checkpoint value recorded in this arm's development run.json."""
    run = json.loads((root / arm / 'development' / 'run.json').read_text(encoding='utf-8'))
    command = run['command']
    return command[command.index('--checkpoint') + 1]


def load_unit(root, arm, task, fold):
    path = root / arm / 'development' / task / f'fold{fold}' / 'metrics.json'
    d = json.loads(path.read_text(encoding='utf-8'))
    assert d['protocol'] == 'outer5_inner20_development', (arm, task, fold)
    assert d['development'] is True and d['validation_only'] is True, (arm, task, fold)
    assert d['outer_test'] == 'NOT_RUN', (arm, task, fold, d['outer_test'])
    assert d['adaptation'] == 'full', (arm, task, fold, d['adaptation'])
    assert int(d['deployment_step']) == 5000, (arm, task, fold, d['deployment_step'])
    return d


def load_reference_r2(s3b_summary):
    r2 = {}
    for task in TASKS:
        fold0 = float(s3b_summary['r2_validation'][REFERENCE][task]['fold0'])
        fold1 = float(s3b_summary['r2_validation'][REFERENCE][task]['fold1'])
        r2[task] = {'folds': {0: fold0, 1: fold1}, 'mean': (fold0 + fold1) / 2}
    return {REFERENCE: r2}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='results/glt_pred_20260918/s4_formal')
    parser.add_argument('--s3b-summary',
                        default='results/glt_pred_20260918/s3b_formal/summary.json')
    args = parser.parse_args()
    root = Path(args.root)
    s3b = json.loads(Path(args.s3b_summary).read_text(encoding='utf-8'))

    summary = {'pretrain': {}, 'units': {}, 'r2_validation': {},
               'reference_r2_validation': s3b['r2_validation'][REFERENCE],
               'gates': {}, 'increment': {}, 'selected_s4_parent': None,
               's4_verdict': None}
    r2 = load_reference_r2(s3b)
    for arm in ARMS:
        runtime = json.loads((root / arm / 'pretrain' / 'runtime.json').read_text())
        assert runtime['status'] == 'PASS', (arm, runtime['status'])
        assert int(runtime['completed_steps']) == 5000, (arm, runtime['completed_steps'])
        expected = str(root / arm / 'pretrain' / 'deploy_05000.pt')
        used = _command_checkpoint(root, arm)
        assert Path(used).resolve() == Path(expected).resolve(), (arm, used)
        summary['pretrain'][arm] = {
            'status': runtime['status'],
            'completed_steps': int(runtime['completed_steps']),
            'deploy_05000': expected,
            'finetune_checkpoint_used': used,
        }
        units = {}
        r2[arm] = {}
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
                    'outer_test': d['outer_test'],
                }
                folds[fold] = float(d['best_validation_r2'])
            r2[arm][task] = {'folds': folds, 'mean': mean(list(folds.values()))}
        summary['units'][arm] = units
        summary['r2_validation'][arm] = {
            task: {'fold0': round(r2[arm][task]['folds'][0], 6),
                   'fold1': round(r2[arm][task]['folds'][1], 6),
                   'mean': round(r2[arm][task]['mean'], 6)}
            for task in TASKS}

    for name, (cand, ref) in {'ENV_VS_B_NONE': ('s4_env', REFERENCE),
                              'TOR_ON_VS_TOR_OFF': ('s4_tor_on', 's4_tor_off'),
                              'TOR_ON_VS_B_NONE': ('s4_tor_on', REFERENCE)}.items():
        summary['gates'][name] = gate(cand, ref, r2)

    summary['increment']['ENV_INCREMENT'] = (
        'ESTABLISHED' if summary['gates']['ENV_VS_B_NONE']['GATE'] == 'PASS'
        else 'NOT_ESTABLISHED')
    summary['increment']['TORSION_INCREMENT'] = (
        'ESTABLISHED' if (summary['gates']['TOR_ON_VS_TOR_OFF']['GATE'] == 'PASS'
                          and summary['gates']['TOR_ON_VS_B_NONE']['GATE'] == 'PASS')
        else 'NOT_ESTABLISHED')
    env_ok = summary['increment']['ENV_INCREMENT'] == 'ESTABLISHED'
    tor_ok = summary['increment']['TORSION_INCREMENT'] == 'ESTABLISHED'
    if not env_ok and not tor_ok:
        summary['selected_s4_parent'] = REFERENCE
        summary['s4_verdict'] = 'NEGATIVE / INCONCLUSIVE (neither candidate established)'
    elif env_ok and not tor_ok:
        summary['selected_s4_parent'] = 's4_env'
        summary['s4_verdict'] = 'SELECTED_CANDIDATE (only ENV established)'
    elif tor_ok and not env_ok:
        summary['selected_s4_parent'] = 's4_tor_on'
        summary['s4_verdict'] = 'SELECTED_CANDIDATE (only TORSION established)'
    else:
        delta_env = r2['s4_env']['xc']['mean'] - r2[REFERENCE]['xc']['mean']
        delta_tor = r2['s4_tor_on']['xc']['mean'] - r2[REFERENCE]['xc']['mean']
        if abs(delta_env - delta_tor) >= 0.005:
            pick = 's4_env' if delta_env > delta_tor else 's4_tor_on'
            reason = 'higher XC mean delta vs B_NONE'
        else:
            worst_env = min(r2['s4_env']['xc']['folds'][f] - r2[REFERENCE]['xc']['folds'][f]
                            for f in FOLDS)
            worst_tor = min(r2['s4_tor_on']['xc']['folds'][f] - r2[REFERENCE]['xc']['folds'][f]
                            for f in FOLDS)
            if abs(worst_env - worst_tor) > 1e-9:
                pick = 's4_env' if worst_env > worst_tor else 's4_tor_on'
                reason = 'worst XC fold delta vs B_NONE (XC mean deltas within 0.005)'
            else:
                prot_env = min(r2['s4_env']['eps']['mean'] - r2[REFERENCE]['eps']['mean'],
                               r2['s4_env']['eat']['mean'] - r2[REFERENCE]['eat']['mean'])
                prot_tor = min(r2['s4_tor_on']['eps']['mean'] - r2[REFERENCE]['eps']['mean'],
                               r2['s4_tor_on']['eat']['mean'] - r2[REFERENCE]['eat']['mean'])
                pick = 's4_env' if prot_env >= prot_tor else 's4_tor_on'
                reason = 'EPS/EAT protection vs B_NONE (XC criteria tied)'
        summary['selected_s4_parent'] = pick
        summary['s4_verdict'] = f'SELECTED_CANDIDATE ({reason})'

    out = root / 'summary.json'
    out.write_text(json.dumps(summary, indent=1, sort_keys=False) + '\n', encoding='utf-8')

    lines = ['# S4 正式结果汇总（development validation R² 口径，reference = B_NONE）', '',
             '## 预训练状态', '', '| arm | status | completed_steps |', '| --- | --- | ---: |']
    for arm in ARMS:
        p = summary['pretrain'][arm]
        lines.append(f"| {arm} | {p['status']} | {p['completed_steps']} |")
    lines += ['', '## 每 task R²（fold0 / fold1 / mean）', '',
              '| arm | XC | EPS | EAT |', '| --- | --- | --- | --- |',
              f"| B_NONE (reference) | "
              f"{s3b['r2_validation'][REFERENCE]['xc']['fold0']} / "
              f"{s3b['r2_validation'][REFERENCE]['xc']['fold1']} / "
              f"{s3b['r2_validation'][REFERENCE]['xc']['mean']} | "
              f"{s3b['r2_validation'][REFERENCE]['eps']['fold0']} / "
              f"{s3b['r2_validation'][REFERENCE]['eps']['fold1']} / "
              f"{s3b['r2_validation'][REFERENCE]['eps']['mean']} | "
              f"{s3b['r2_validation'][REFERENCE]['eat']['fold0']} / "
              f"{s3b['r2_validation'][REFERENCE]['eat']['fold1']} / "
              f"{s3b['r2_validation'][REFERENCE]['eat']['mean']} |"]
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
              f"- ENV_INCREMENT = {summary['increment']['ENV_INCREMENT']}",
              f"- TORSION_INCREMENT = {summary['increment']['TORSION_INCREMENT']}",
              f"- SELECTED_S4_PARENT = **{summary['selected_s4_parent']}**",
              f"- 判定说明: {summary['s4_verdict']}", '']
    (root / 'report.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps({'summary': str(out), 'report': str(root / 'report.md'),
                      'gates': {k: v['GATE'] for k, v in summary['gates'].items()},
                      'increment': summary['increment'],
                      'selected_s4_parent': summary['selected_s4_parent']}, indent=1))


if __name__ == '__main__':
    main()
