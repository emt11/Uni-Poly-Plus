#!/usr/bin/env python3
"""Aggregate the 24 D2 development units (GLT-3D-GAIN-20260921-01/r6).

Only the 24 predefined arm/task/fold identities are accepted, and a unit counts
as complete only when its own records pass every check below — an existing
directory or a trailing DONE line in a log is never accepted as evidence:

* ``runtime.json`` says PASS and its identity matches the unit;
* stage is ``development`` and ``outer_test`` is ``NOT_RUN``;
* the recorded split matches the fixed ``outer5_inner20`` fold;
* the recorded config equals the frozen config file and the recorded command
  names the expected deployment package;
* ``requested_epochs <= 30``, ``executed_epochs`` equals the history length,
  ``optimizer_updates`` equals the sum of the per-epoch step counts, and
  ``best_epoch`` / ``best_validation_r2`` agree with the history;
* every required artifact exists and every metric is finite.

An incomplete set is reported as such: the aggregate is written with status
``INCOMPLETE`` and the process exits non-zero.
"""
import argparse
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.finetune_glt_3d_gain_d2 import (ARM_LRS, FOLDS, STAGE_EPOCH_LIMIT, SPLIT_PROTOCOL,
                                            TASKS, unit_directory)
from scripts.run_glt_3d_gain_d2_development import unit_list

REQUIRED_FILES = ('run.json', 'runtime.json', 'metrics.json', 'best.pt')
CANDIDATES = ('fnorm', 'fstable')
REFERENCES = ('fbase', 'f2d')
GATE = {'macro3': 0.005, 'xc_mean': 0.01, 'task_floor': -0.01}


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def check_unit(root, arm, task, fold, *, config, checkpoint, max_epochs):
    """Validate one unit's records; returns (record, problems)."""
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
        if record.get('arm') != arm or record.get('task') != task or int(record.get('fold', -1)) != fold:
            problems.append(f'{name} identity does not match {arm}/{task}/fold{fold}')
    if runtime.get('status') != 'PASS':
        problems.append(f"runtime status is {runtime.get('status')!r}, not PASS")
    for name, record in (('run', run), ('metrics', metrics)):
        if record.get('stage') != 'development':
            problems.append(f"{name} stage is {record.get('stage')!r}, not development")
    if metrics.get('outer_test') != 'NOT_RUN':
        problems.append(f"metrics outer_test is {metrics.get('outer_test')!r}")

    requested = metrics.get('requested_epochs')
    if not isinstance(requested, int) or not 1 <= requested <= max_epochs:
        problems.append(f'requested_epochs {requested!r} is outside 1..{max_epochs}')
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
        problems.append('per-epoch training_steps are missing or not positive integers')
        steps = []
    updates = metrics.get('optimizer_updates')
    if steps and updates != sum(steps):
        problems.append(f'optimizer_updates {updates!r} != sum of per-epoch steps {sum(steps)}')
    if updates != run.get('optimizer_updates'):
        problems.append('optimizer_updates disagrees between run and metrics')
    r2 = [record.get('validation_r2') for record in history]
    finite_history = bool(r2) and all(_finite(value) for value in r2)
    if r2 and not finite_history:
        problems.append('history holds a non-finite validation_r2')
    best_value = metrics.get('best_validation_r2')
    if not _finite(best_value):
        problems.append('best_validation_r2 is not finite')
    elif finite_history:
        if best_value != max(r2):
            problems.append('best_validation_r2 is not the history maximum')
        if metrics.get('best_epoch') != r2.index(max(r2)) + 1:
            problems.append('best_epoch is not the history argmax')
    for name in ('train_loss', 'validation_loss', 'validation_r2'):
        values = [record.get(name) for record in history]
        if not all(_finite(value) for value in values):
            problems.append(f'history holds a non-finite {name}')

    if run.get('config') != config:
        problems.append('the recorded config differs from the frozen config')
    if run.get('learning_rate_table') != ARM_LRS[arm]:
        problems.append('the recorded learning-rate table differs from the arm contract')
    command = run.get('command')
    if not isinstance(command, list) or '--checkpoint' not in command:
        problems.append('the recorded command does not name a deployment package')
    elif command[command.index('--checkpoint') + 1] != str(checkpoint):
        problems.append('the unit was not initialised from the expected deployment package')
    if '--stage' not in (command or []) or command[command.index('--stage') + 1] != 'development':
        problems.append('the recorded command is not a development run')

    split = metrics.get('split') or {}
    if (split.get('protocol') != SPLIT_PROTOCOL or split.get('task') != task
            or int(split.get('fold', -1)) != fold or split.get('sets_disjoint') is not True
            or split.get('union_equals_full_cohort') is not True):
        problems.append('the recorded split evidence does not match the fixed fold')
    if int(split.get('train_rows', -1)) + int(split.get('validation_rows', -1)) \
            + int(split.get('test_rows', -1)) != int(split.get('sample_count', -2)):
        problems.append('the recorded split does not cover its cohort')

    rng = metrics.get('rng') or {}
    if arm == 'f2d':
        if rng.get('glt_stream_seed') is not None:
            problems.append('f2d reports a private 3D stream')
    elif rng.get('glt_stream_seed') != 42 or not isinstance(rng.get('glt_stream_entries'), int) \
            or rng.get('glt_stream_entries', 0) < 1:
        problems.append('the dual arm does not report its isolated 3D stream')

    if problems:
        return None, problems
    return {'arm': arm, 'task': task, 'fold': fold,
            'best_validation_r2': float(metrics['best_validation_r2']),
            'best_epoch': int(metrics['best_epoch']),
            'executed_epochs': int(executed), 'optimizer_updates': int(updates),
            'requested_epochs': int(requested),
            'history_r2': [float(value) for value in r2],
            'hit_epoch_cap': int(metrics['best_epoch']) >= max_epochs,
            'directory': str(directory)}, []


def task_means(records, arm):
    values = {}
    for task in TASKS:
        scores = [records[(arm, task, fold)]['best_validation_r2'] for fold in FOLDS]
        values[task] = sum(scores) / len(scores)
    return values


def deltas(records, candidate, reference):
    left, right = task_means(records, candidate), task_means(records, reference)
    per_task = {task: left[task] - right[task] for task in TASKS}
    per_fold = {task: [records[(candidate, task, fold)]['best_validation_r2']
                       - records[(reference, task, fold)]['best_validation_r2']
                       for fold in FOLDS] for task in TASKS}
    return {'per_task_mean': per_task,
            'per_fold': per_fold,
            'macro3': sum(per_task.values()) / len(per_task)}


def evaluate_gate(records, candidate):
    """The pre-registered gate, applied separately against each reference."""
    results = {}
    for reference in REFERENCES:
        delta = deltas(records, candidate, reference)
        conditions = {
            'macro3_at_least_0.005': delta['macro3'] >= GATE['macro3'],
            'xc_mean_at_least_0.01': delta['per_task_mean']['xc'] >= GATE['xc_mean'],
            'xc_both_folds_positive': all(value > 0 for value in delta['per_fold']['xc']),
            'no_task_mean_below_-0.01': all(value >= GATE['task_floor']
                                            for value in delta['per_task_mean'].values()),
        }
        results[reference] = {'delta': delta, 'conditions': conditions,
                              'passed': all(conditions.values())}
    results['passed'] = all(results[reference]['passed'] for reference in REFERENCES)
    return results


def select_candidate(records):
    """Both candidates are reported; the pick follows the pre-registered order."""
    evaluated = {candidate: evaluate_gate(records, candidate) for candidate in CANDIDATES}
    passing = [candidate for candidate in CANDIDATES if evaluated[candidate]['passed']]
    picked = None
    if passing:
        def key(candidate):
            macro = evaluated[candidate]['fbase']['delta']['macro3']
            xc = evaluated[candidate]['fbase']['delta']['per_task_mean']['xc']
            simplicity = 0 if candidate == 'fnorm' else 1
            return (-macro, -xc, simplicity)
        picked = sorted(passing, key=key)[0]
    return {'evaluated': evaluated, 'passing': passing, 'picked': picked,
            'rule': ('macro3 increment vs FBASE first, then XC increment, then the simpler '
                     'FNORM; the unselected candidate is reported too')}


def _number(value):
    return f'{value:+.5f}' if value < 0 else f'{value:.5f}'


def write_report(path, payload):
    """Render the aggregate as markdown; an incomplete set is reported as such."""
    lines = ['# D2 开发比较聚合', '',
             f"- 状态：**{payload['status']}**（接受 {payload['accepted_units']}/"
             f"{payload['expected_units']} 个单元）",
             f"- 根目录：`{payload['root']}`",
             f"- 配置：`{payload['config']}`｜部署包：`{payload['checkpoint']}`", '']
    if payload['problems']:
        lines += ['## 未通过校验的单元', '', '| 单元 | 问题 |', '| --- | --- |']
        for unit, issues in sorted(payload['problems'].items()):
            lines.append(f'| {unit} | ' + '；'.join(issues) + ' |')
        lines += ['', '未通过校验的单元不进入任何比较；下表缺省。', '']
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text('\n'.join(lines) + '\n', encoding='utf-8')
        return
    comparison = payload['comparison']
    lines += ['## 逐单元 validation R²（最佳 checkpoint）', '',
              '| 单元 | R² | best epoch | executed epochs | updates | 30 上限 |', '| --- | --- | --- | --- | --- | --- |']
    for unit, record in payload['units'].items():
        lines.append(f"| {unit} | {record['best_validation_r2']:.6f} | {record['best_epoch']} | "
                     f"{record['executed_epochs']} | {record['optimizer_updates']} | "
                     f"{'是' if record['hit_epoch_cap'] else '否'} |")
    lines += ['', '## 逐任务两折均值', '', '| arm | xc | eps | eat | macro3 |', '| --- | --- | --- | --- | --- |']
    for arm, means in comparison['per_arm_task_mean_r2'].items():
        macro = sum(means.values()) / len(means)
        lines.append(f"| {arm} | {means['xc']:.6f} | {means['eps']:.6f} | {means['eat']:.6f} | {macro:.6f} |")
    for title, key in (('FBASE − F2D', 'fbase_minus_f2d'), ('FNORM − FBASE', 'fnorm_minus_fbase'),
                       ('FSTABLE − FNORM', 'fstable_minus_fnorm')):
        delta = comparison[key]
        lines += ['', f'## {title}', '',
                  '| 任务 | 两折均值差 | fold0 | fold1 |', '| --- | --- | --- | --- |']
        for task in TASKS:
            folds = delta['per_fold'][task]
            lines.append(f"| {task} | {_number(delta['per_task_mean'][task])} | "
                         f"{_number(folds[0])} | {_number(folds[1])} |")
        lines.append(f"| **macro3** | **{_number(delta['macro3'])}** | | |")
    lines += ['', '## 候选 vs FBASE / vs F2D', '',
              '| 候选 | 参照 | macro3 | xc 均值差 | xc fold0 | xc fold1 | 通过 |',
              '| --- | --- | --- | --- | --- | --- | --- |']
    for candidate in CANDIDATES:
        for reference in REFERENCES:
            entry = comparison['gate'][candidate][reference]
            delta = entry['delta']
            lines.append(f"| {candidate} | {reference} | {_number(delta['macro3'])} | "
                         f"{_number(delta['per_task_mean']['xc'])} | "
                         f"{_number(delta['per_fold']['xc'][0])} | "
                         f"{_number(delta['per_fold']['xc'][1])} | "
                         f"{'是' if entry['passed'] else '否'} |")
    lines += ['', '## 门槛条件明细', '', '| 候选 | 参照 | 条件 | 成立 |', '| --- | --- | --- | --- |']
    for candidate in CANDIDATES:
        for reference in REFERENCES:
            for condition, ok in comparison['gate'][candidate][reference]['conditions'].items():
                lines.append(f'| {candidate} | {reference} | {condition} | {"是" if ok else "否"} |')
    selection = comparison['selection']
    lines += ['', '## 选择', '',
              f"- 通过门槛的候选：{selection['passing'] or '无'}",
              f"- 建议候选：**{selection['picked'] or '无'}**",
              f"- 规则：{selection['rule']}",
              f"- best_epoch 触及 30 上限的单元：{comparison['epoch_cap_hits'] or '无'}", '',
              '## 限制', '', f"- {comparison['limits']}",
              '- 公共随机流的末态摘要可能因各臂早停长度不同而不同，不作为跨臂相等要求。', '']
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--report', required=True)
    args = parser.parse_args()
    root = Path(args.root)
    config = json.loads(Path(args.config).read_text(encoding='utf-8'))
    max_epochs = STAGE_EPOCH_LIMIT['development']

    records, problems = {}, {}
    for arm, task, fold in unit_list():
        record, issues = check_unit(root, arm, task, fold, config=config,
                                    checkpoint=args.checkpoint, max_epochs=max_epochs)
        if issues:
            problems[f'{arm}/{task}/fold{fold}'] = issues
        else:
            records[(arm, task, fold)] = record

    payload = {'protocol': 'glt_3d_gain_d2_development', 'root': str(root),
               'config': args.config, 'checkpoint': args.checkpoint,
               'expected_units': len(unit_list()), 'accepted_units': len(records),
               'problems': problems,
               'status': 'COMPLETE' if not problems else 'INCOMPLETE',
               'units': {f'{arm}/{task}/fold{fold}': record
                         for (arm, task, fold), record in sorted(records.items())}}
    if not problems:
        payload['comparison'] = {
            'per_arm_task_mean_r2': {arm: task_means(records, arm) for arm in
                                     ('f2d', 'fbase', 'fnorm', 'fstable')},
            'fbase_minus_f2d': deltas(records, 'fbase', 'f2d'),
            'fnorm_minus_fbase': deltas(records, 'fnorm', 'fbase'),
            'fstable_minus_fnorm': deltas(records, 'fstable', 'fnorm'),
            'candidates_vs_fbase': {candidate: deltas(records, candidate, 'fbase')
                                    for candidate in CANDIDATES},
            'candidates_vs_f2d': {candidate: deltas(records, candidate, 'f2d')
                                  for candidate in CANDIDATES},
            'gate': {candidate: evaluate_gate(records, candidate) for candidate in CANDIDATES},
            'selection': select_candidate(records),
            'epoch_cap_hits': [f'{arm}/{task}/fold{fold}'
                               for (arm, task, fold), record in sorted(records.items())
                               if record['hit_epoch_cap']],
            'limits': ('single seed, three tasks x two folds, development comparison; '
                       'not an independent blind test and not the formal eight-task result')}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    write_report(args.report, payload)
    print(json.dumps({'status': payload['status'], 'accepted': payload['accepted_units'],
                      'problems': len(problems)}, ensure_ascii=False))
    for unit, issues in sorted(problems.items()):
        print(f'  {unit}: ' + '; '.join(issues))
    if problems:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
