#!/usr/bin/env python3
"""N1: freeze all bests and summarize saved raw-target predictions once."""
import csv
import difflib
import math
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_glt_ft_raw_targets_unit import (
    OUT, LOG, REVIEW, U1, BASE, SPLIT, ARMS, TASKS, PARAMETERS, REFERENCE_ARMS,
    read, write, now, parent, head_path, ft_folder, target_contract, validate_config, validate_best)

REFERENCE = 0.7855913245367517
FILES = ('CURRENT_PLAN.md', 'scripts/run_glt_ft_raw_targets_unit.py',
         'scripts/run_glt_ft_raw_targets.py', 'scripts/select_glt_ft_raw_targets.py',
         'tests/test_glt_ft_raw_targets.py')


def macro_r2(tasks):
    if set(tasks) != set(TASKS):
        raise ValueError('all seven tasks required')
    arrays = [np.asarray(tasks[t], dtype=float) for t in TASKS]
    if any(a.shape != (5,) or not np.isfinite(a).all() for a in arrays):
        raise ValueError('five finite folds required')
    return float(np.mean([a.mean() for a in arrays]))


def freeze():
    import torch
    if (OUT / 'evaluation_before_test.json').exists():
        raise FileExistsError('matrix already frozen')
    configs = {a: validate_config(read(OUT / 'initialization' / f'{a}.json'), a) for a in ARMS}
    units = []
    for arm in ARMS:
        for task in TASKS:
            split = read(SPLIT / f'{task}.json')
            for fold in range(5):
                folder = ft_folder(arm, task, fold)
                metric = read(folder / 'metrics.json')
                entry = split['folds'][fold]
                count = 70 * ((len(entry['train_indices']) + 23) // 24)
                if (metric['status'] != 'PASS' or metric['outer_test'] != 'NOT_RUN'
                        or metric['epochs'] != 70 or metric['parameters'] != PARAMETERS[arm]
                        or metric['updates'] != {'attempted': count, 'completed': count}
                        or metric['modes'] != configs[arm]
                        or read(folder / 'target.json') != target_contract()):
                    raise ValueError('FT matrix incomplete or target contract changed')
                best = torch.load(folder / 'best.pt', map_location='cpu', weights_only=False)
                validate_best(best, arm, task, fold, configs[arm])
                history = best['history']
                if (len(history) != 70 or [h['epoch'] for h in history] != list(range(1, 71))
                        or [h['stage'] for h in history] != ['head'] * 10 + ['joint'] * 60
                        or any(h.get('target_transform') != 'identity' or h.get('ft_profile') != 'uniform_u1'
                               for h in history)
                        or sum(h['training_steps'] for h in history) != count
                        or best['optimizer_updates'] != count
                        or best['best_epoch'] != int(np.argmin([h['validation_rmse'] for h in history])) + 1
                        or any(not torch.isfinite(v).all() for v in best['state_dict'].values())):
                    raise ValueError('best/history/budget mismatch')
                units.append(dict(arm=arm, task=task, fold=fold, best=str(folder / 'best.pt'),
                                  parent=str(parent(arm)), head_initial=str(head_path(arm, task, fold)),
                                  best_epoch=best['best_epoch'], target_contract=target_contract(),
                                  prediction_path=configs[arm]['prediction_path'], input_dim=configs[arm]['input_dim'],
                                  test_indices=sorted(entry['test_indices'])))
    if len(units) != 70:
        raise ValueError('complete 70-unit matrix required')
    write(OUT / 'evaluation_before_test.json', dict(status='FROZEN', validation_units=70,
          units=units, configs=configs, selection_by_validation=True, outer_test='NOT_RUN', time=now()))


def check_reference_labels(raw, reference_truth, mean, scale):
    """Account only for the old FP32 standardize/inverse round trip."""
    raw = np.asarray(raw, dtype=np.float64)
    truth = np.asarray(reference_truth, dtype=np.float64)
    if raw.shape != truth.shape or not np.isfinite(raw).all() or not np.isfinite(truth).all():
        raise ValueError('reference raw label schema mismatch')
    mean, scale = float(np.asarray(mean).reshape(-1)[0]), float(np.asarray(scale).reshape(-1)[0])
    if not math.isfinite(mean) or not math.isfinite(scale) or scale <= 0:
        raise ValueError('reference scaler metadata invalid')
    bound = 8 * np.finfo(np.float32).eps * (np.abs(raw) + np.abs(raw - mean) + abs(mean) + abs(scale))
    if np.any(np.abs(raw - truth) > bound):
        raise ValueError('reference labels differ beyond FP32 round-trip bound')
    return float(np.max(np.abs(raw - truth)))


def papers():
    result = {}
    for line in (ROOT / 'SOTA_RESULT.md').read_text().splitlines():
        parts = [v.strip() for v in line.split('|')]
        if len(parts) == 5 and parts[1].lower() in TASKS and parts[3].startswith('**'):
            result[parts[1].lower()] = dict(model=parts[2], r2=float(parts[3].strip('*').split('±')[0]))
    if set(result) != set(TASKS):
        raise ValueError('paper matrix incomplete')
    return result


def summarize():
    from sklearn.metrics import r2_score
    frozen = read(OUT / 'evaluation_before_test.json')
    refs = read(U1 / 'final_review/recovery1/seven_task_results.json')['arms']
    historic = read(BASE / 'final_review/seven_task_results.json')['arms']['F_CE']
    paper = papers()
    details, rows = {}, []
    for arm in ARMS:
        tasks, values = {}, {}
        reference = refs[REFERENCE_ARMS[arm]]
        for task in TASKS:
            folds = []
            for fold in range(5):
                binding = next(u for u in frozen['units'] if (u['arm'], u['task'], u['fold']) == (arm, task, fold))
                folder = OUT / 'formal' / arm / task / f'fold{fold}'
                m = read(folder / 'metrics.json')
                ref_fold = next(f for f in reference['tasks'][task]['folds'] if f['fold'] == fold)
                with np.load(folder / 'test_predictions.npz', allow_pickle=False) as p:
                    if (p['indices'].tolist() != binding['test_indices']
                            or int(p['best_epoch']) != binding['best_epoch']
                            or str(p['target_transform']) != 'identity'
                            or any(k in p.files for k in ('scaler_mean', 'scaler_scale'))):
                        raise ValueError('formal raw-target binding mismatch')
                    truth, prediction, raw = p['y_true'], p['y_pred'], p['raw_labels']
                    if (truth.shape != prediction.shape or not np.isfinite(prediction).all()
                            or not np.array_equal(truth, raw.astype(np.float32).astype(np.float64))):
                        raise ValueError('formal arrays are not raw targets')
                    with np.load(ref_fold['predictions'], allow_pickle=False) as ref:
                        for key in ('indices', 'sample_keys', 'geometry_valid'):
                            if not np.array_equal(p[key], ref[key]):
                                raise ValueError('scaled reference identity/geometry mismatch: ' + key)
                        label_error = check_reference_labels(raw, ref['y_true'], ref['scaler_mean'], ref['scaler_scale'])
                    r2 = float(r2_score(truth, prediction))
                    rmse = float(np.sqrt(np.mean((truth - prediction) ** 2)))
                if (m['status'] != 'PASS' or m['outer_test'] != 'RUN' or m['parameters'] != PARAMETERS[arm]
                        or m['target_contract'] != target_contract() or m['modes'] != frozen['configs'][arm]
                        or not math.isfinite(r2) or not math.isfinite(rmse)
                        or m['r2'] != r2 or m['rmse'] != rmse):
                    raise ValueError('formal metrics/config mismatch')
                val = read(ft_folder(arm, task, fold) / 'metrics.json')
                old_val = read(U1 / 'finetune' / REFERENCE_ARMS[arm] / task / f'fold{fold}' / 'metrics.json')
                row = dict(arm=arm, task=task, fold=fold, r2=r2, rmse=rmse,
                           scaled_reference_r2=ref_fold['r2'], scaled_reference_rmse=ref_fold['rmse'],
                           r2_minus_scaled=r2 - ref_fold['r2'], rmse_minus_scaled=rmse - ref_fold['rmse'],
                           validation_r2=val['r2'], validation_rmse=val['rmse'],
                           validation_r2_minus_scaled=val['r2'] - old_val['r2'],
                           validation_rmse_minus_scaled=val['rmse'] - old_val['rmse'],
                           reference_truth_roundtrip_max_abs=label_error, best_epoch=binding['best_epoch'],
                           best=binding['best'], predictions=str(folder / 'test_predictions.npz'),
                           scaled_reference_predictions=ref_fold['predictions'])
                folds.append(row)
                rows.append(row)
            keys = ('r2', 'rmse', 'r2_minus_scaled', 'rmse_minus_scaled', 'validation_r2_minus_scaled',
                    'validation_rmse_minus_scaled')
            stats = {k: dict(mean=float(np.mean([f[k] for f in folds])),
                             std=float(np.std([f[k] for f in folds], ddof=0))) for k in keys}
            stats.update(folds=folds, paper=dict(paper[task], delta=stats['r2']['mean'] - paper[task]['r2']),
                         r2_minus_historic_f_ce=stats['r2']['mean'] - historic[task]['r2']['mean'])
            tasks[task], values[task] = stats, [f['r2'] for f in folds]
        macro = macro_r2(values)
        details[arm] = dict(tasks=tasks, macro_test_r2=macro,
                            macro_minus_scaled=macro - reference['macro_test_r2'],
                            macro_minus_historic_f_ce=macro - REFERENCE,
                            parameters=PARAMETERS[arm], configuration=frozen['configs'][arm],
                            qualifies=macro > REFERENCE + 0.01)
    gap = details['UA_2D']['macro_test_r2'] - details['F_CE']['macro_test_r2']
    scaled_gap = refs['UA_2D']['macro_test_r2'] - refs['U0_FUSED']['macro_test_r2']
    pairwise = dict(UA_2D_minus_F_CE_raw=gap, UA_2D_minus_F_CE_scaled=scaled_gap,
                    gap_change_under_raw_targets=gap - scaled_gap)
    result = dict(arms=details, pairwise=pairwise, consumption=read(OUT / 'consumption.json'),
                  scaled_references={a: refs[REFERENCE_ARMS[a]]['macro_test_r2'] for a in ARMS},
                  historic_f_ce_macro=REFERENCE,
                  protocol='seven tasks five folds seed42+fold, ddof=0; shared TEST development; paper numeric comparison only',
                  label_roundtrip='raw FP64 source retained; model labels cast FP32; old standardized truth checked with fixed FP32 bound')
    REVIEW.mkdir(parents=True, exist_ok=True)
    write(REVIEW / 'seven_task_results.json', result)
    with (REVIEW / 'formal_fold_details.csv').open('x', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ['# N1 原单位标签微调正式结果（待独立审查）', '',
             '七任务各五折，seed42+fold，std ddof=0，七任务等权宏平均。展示3位小数，计算与产物保留完整精度。', '',
             '|模型|原单位标签宏R²|缩放对照宏R²|不缩放−缩放|相对历史F_CE|', '|---|---:|---:|---:|---:|']
    for a in ARMS:
        x = details[a]
        lines.append(f"|{a}|{x['macro_test_r2']:.3f}|{result['scaled_references'][a]:.3f}|{x['macro_minus_scaled']:+.3f}|{x['macro_minus_historic_f_ce']:+.3f}|")
    for a in ARMS:
        lines += ['', f'## {a}', '', '|任务|R² mean±std|原单位RMSE mean±std|R²差：不缩放−缩放|论文最佳模型|论文R²|项目−论文|',
                  '|---|---:|---:|---:|---|---:|---:|']
        for t in TASKS:
            x = details[a]['tasks'][t]
            lines.append(f"|{t}|{x['r2']['mean']:.3f}±{x['r2']['std']:.3f}|{x['rmse']['mean']:.3f}±{x['rmse']['std']:.3f}|{x['r2_minus_scaled']['mean']:+.3f}|{x['paper']['model']}|{x['paper']['r2']:.3f}|{x['paper']['delta']:+.3f}|")
        lines += ['', '|任务|fold0 R²|fold1 R²|fold2 R²|fold3 R²|fold4 R²|', '|---|---:|---:|---:|---:|---:|']
        for t in TASKS:
            lines.append('|' + t + '|' + '|'.join(f"{f['r2']:.3f}" for f in details[a]['tasks'][t]['folds']) + '|')
    lines += ['', f'原单位标签条件UA_2D−F_CE宏差 {gap:+.3f}；缩放条件宏差 {scaled_gap:+.3f}。', '',
              'F_CE的匹配缩放参考是U0_FUSED，历史任务特定FT仅作数值参考。无标签缩放、中心化、log/logit、逆变换或head仿射转换；相同parent/head/split/统一FT预算。',
              '共享TEST已用于开发，非独立盲测；论文数据、split及统计协议等价性未完全核实，仅数值对照。跨模型比较同时改变PT目标、预测模态与容量，不能归因于单个因素。',
              '固定学习率与clip下，原单位MSE改变梯度尺度是本实验干预的一部分；不按TEST追加调参。工程验收与性能分别报告，不自动替换F_CE或部署I。', '',
              '原单位FP64标签另存raw_labels；模型使用其FP32值。旧缩放预测y_true存在FP32标准化往返误差，逐fold报告实际误差并按预定量化界限检查，未要求scaler一致。']
    (REVIEW / 'formal_results.md').write_text('\n'.join(lines) + '\n')


def capture_diff():
    diff = []
    for name in FILES:
        before = LOG / 'source_before' / name
        original = before.read_text() if before.exists() else ''
        diff.extend(difflib.unified_diff(original.splitlines(True), (ROOT / name).read_text().splitlines(True),
                                         fromfile='before/' + name, tofile='after/' + name))
    destination = REVIEW / 'actual_changes.diff'
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('x') as handle:
        handle.write(''.join(diff))


def handoff():
    execution = (read(OUT / 'execution.json') if (OUT / 'execution.json').exists()
                 else read(OUT / 'recovery1/failure.json') if (OUT / 'recovery1/failure.json').exists()
                 else read(OUT / 'failure.json'))
    status = execution['status']
    final = LOG / 'final'
    test = (final / 'stdout.log').read_text() if (final / 'stdout.log').exists() else '未执行'
    test_error = (final / 'stderr.log').read_text() if (final / 'stderr.log').exists() else ''
    report_path = REVIEW / 'formal_results.md'
    results = report_path.read_text() if report_path.exists() else '正式矩阵未完成，不可比较；未补跑。'
    report = ('\n\n## 36. N1 r7 当前实际执行交接（' + status + '）\n\n'
              + '授权：用户要求实现并执行F_CE／UA_2D不缩放标签微调。无新PT，复用原parent和原head，uniform_u1七任务五折。\n\n'
              + '实际状态：' + status + '；工程与性能分别报告，自检不替代独立审查。\n\n实际消耗：\n\n```json\n'
              + u1_json(read(OUT / 'consumption.json')) + '\n```\n\n唯一最终CPU测试：\n\n```text\n'
              + test.strip() + '\n```\n' + ('\n```text\n' + test_error.strip() + '\n```\n' if test_error else '')
              + '\n' + results + '\n\n产物results/glt_optimization_20261011/r7/N1/；日志logs/glt_optimization_20261011/r7/N1/；'
              + '当前控制器Uni-Poly:glt_opt_r7_N1_controller_resource_fix；原控制器与纠正资源时的故障日志均保留。best在finetune/<arm>/<task>/foldN/best.pt，正式预测在formal/<arm>/<task>/foldN/test_predictions.npz；'
              + 'target.json与evaluation_before_test.json绑定identity/no-fit。当前交接／增量／逐fold表在final_review/recovery1/；原final_review交接为调度中断快照，root failure.json保留为已记录的历史事故。\n\n'
              + '保留S4／E0／E1／Q等未实施路线，不自动启动；F_CE、部署I、PROJECT_HISTORY.md、SOTA_RESULT.md保持；未读取TODO.md。\n')
    if execution.get('error'):
        report += '\n停止原因：' + execution['error'] + '。现场failure.json及日志保留，不自动重试。\n'
    REVIEW.mkdir(parents=True, exist_ok=True)
    with (REVIEW / 'delivery.md').open('x') as handle:
        handle.write(report)
    path = ROOT / 'CURRENT_PLAN.md'
    plan = path.read_text().replace('**N1，执行中**', '**N1，' + status + '**')
    path.write_text(plan + report)


def u1_json(value):
    import json
    return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)


if __name__ == '__main__':
    from src.training.glt_dual_runtime import require_tmux
    require_tmux()
    if sys.argv[1:] == ['--summarize']:
        summarize()
    elif sys.argv[1:] == ['--handoff']:
        handoff()
    elif sys.argv[1:] == ['--capture-diff']:
        capture_diff()
    elif not sys.argv[1:]:
        freeze()
    else:
        raise ValueError('invalid selection arguments')
