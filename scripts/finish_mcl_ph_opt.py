#!/usr/bin/env python3
"""One final testing/evaluation phase; saved evidence, no hash verification."""
import gc
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_mcl_ph_opt import PLAN, OUT, LOG, PI, DOWN, SPLITS, STATISTICS, execution_paths
from scripts.finetune_mcl_ph import MCLPHCleanDataset, build_mcl_arm, mcl_ph_downstream_collate
from src.dataset.mcl_ph_view import (load_geometric_statistics, prepare_mcl_ph_sample,
                                     mcl_ph_collate)
from src.modules.mcl_ph_pretrain import MCLPHPretrainer
from src.modules.glt_adaptation import AdaptationAdapter
from src.training.glt_dual_runtime import (require_tmux, open_source, load_sample_index_artifact,
    pretrain_source_fields, pretrain_loader, move_labels)
from src.training.sota_ft_contract import read, key, reference
from src.training.sota_ft_runtime import finite, json_write
from src.utils import set_global_seed, evaluate, TargetScaler
REVISION = 'r2'


def score(y, prediction):
    y, prediction = np.asarray(y).reshape(-1), np.asarray(prediction).reshape(-1)
    if len(y) != len(prediction) or not np.isfinite(y).all() or not np.isfinite(prediction).all():
        raise ValueError('invalid saved prediction')
    residual = y - prediction
    return dict(r2=float(1 - np.sum(residual ** 2) / np.sum((y - y.mean()) ** 2)),
                rmse=float(np.sqrt(np.mean(residual ** 2))), mae=float(np.mean(np.abs(residual))))


def pretraining_consumption():
    """Account stopped updates from the trainer's actual stage/error records."""
    units = []
    for arm in ('M0', 'Mmix'):
        folder = OUT / 'R2/pretrain' / arm
        if not (folder / 'runtime.json').exists():
            continue
        runtime = read(folder / 'runtime.json')
        rank_steps, rank_attempts, errors = [], [], []
        for rank in range(4):
            path = folder / f'stages_rank{rank}.log'
            events = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
            rank_steps.append(max([r['step'] for r in events if r['stage'] in ('step', 'optimizer_update') and r['event']=='complete'] or [0]))
            rank_attempts.append(max([r['step'] for r in events if r['stage']=='step' and r['event']=='enter'] or [0]))
            path = folder / f'runtime_failure_rank{rank}.json'
            errors.append(read(path).get('error', '') if path.exists() else '')
        # This specific stop is raised after optimizer.step on every rank.
        router_stop = all('STOP_ROUTER_COLLAPSE' in error for error in errors)
        completed = min(rank_attempts) if router_stop else min(rank_steps)
        windows = []
        window_path = folder / 'router_windows.jsonl'
        if window_path.exists():
            windows = [json.loads(line) for line in window_path.read_text().splitlines()]
        log = LOG / ('R2_pretrain_' + arm) / 'stdout.log'
        text = log.read_text() if log.exists() else ''
        decoder = json.JSONDecoder(); position = 0
        while True:
            position = text.find('{"step":', position)
            if position < 0:
                break
            try:
                record, _ = decoder.raw_decode(text[position:])
            except ValueError:
                position += 1
                continue
            if not window_path.exists() and record.get('rank') == 0 and record.get('router_valid_window'):
                windows.append(dict(step=record['step'], **record['router_valid_window']))
            position += 1
        units.append(dict(arm=arm, status=runtime['status'],
            completed_optimizer_updates=completed, fully_completed_step_records=rank_steps,
            attempted_step_records=rank_attempts, errors=errors, router_stop=router_stop,
            stop_windows=windows[-3:], router_windows=windows, folder=str(folder)))
    return units


def task_rows(task):
    manifest = read(SPLITS / (task + '.json'))
    source, frame = open_source(DOWN, DOWN, task=task, dual_static_root=DOWN,
        selected_indices=list(range(manifest['sample_count'])), split_manifest=manifest,
        expected_task_rows=manifest['sample_count'], record_index_path=DOWN / (task + '_offsets.npy'))
    try:
        return manifest, frame['label'].to_numpy(dtype=float), [k.hex() for k, _ in source.samples]
    finally:
        source.close()


def review_ft(folder, task, fold, rows, sp=False):
    """Read best once; recompute the consumed prediction/scaler from actual rows."""
    if (folder / 'UNCOMMITTED').exists() or (folder / 'failure.json').exists():
        raise ValueError('incomplete unit ' + str(folder))
    m, runtime = read(folder / 'metrics.json'), read(folder / 'runtime.json')
    # finetune_mcl_ph writes completion to runtime; the SP producer also
    # includes status in metrics. Consume each producer's actual schema.
    if runtime['status'] != 'PASS' or (sp and m['status'] != 'PASS'):
        raise ValueError('failed unit ' + str(folder))
    if not sp and (m['task'], m['fold'], m['stage'], m['outer_test']) != (task, fold, 'development', 'NOT_RUN'):
        raise ValueError('development unit identity ' + str(folder))
    finite(m)
    history = m['history']
    expected_epochs = 3 if m.get('smoke_only') else 70
    if len(history) != expected_epochs:
        raise ValueError('epoch count ' + str(folder))
    best = min(history, key=lambda r:r['validation_rmse'])
    if best['epoch'] != m['best_epoch']:
        raise ValueError('best selection ' + str(folder))
    checkpoint = torch.load(folder / 'best.pt', map_location='cpu', weights_only=False)
    finite(checkpoint)
    if checkpoint.get('epoch', checkpoint.get('best_epoch')) != best['epoch']:
        raise ValueError('saved best epoch ' + str(folder))
    manifest, labels, keys = rows
    entry = manifest['folds'][fold]
    mean, scale = labels[entry['train_indices']].mean(), labels[entry['train_indices']].std()
    np.testing.assert_allclose(checkpoint['scaler_mean'], [mean], atol=1e-12, rtol=0)
    np.testing.assert_allclose(checkpoint['scaler_scale'], [scale], atol=1e-12, rtol=0)
    path = folder / ('single_validation.npz' if sp else 'validation_predictions.npz')
    with np.load(path, allow_pickle=False) as prediction:
        indices = prediction['indices' if sp else 'validation_indices'].tolist()
        if indices != entry['validation_indices'] or prediction['sample_keys'].tolist() != [keys[i] for i in indices]:
            raise ValueError('prediction rows ' + str(folder))
        np.testing.assert_allclose(prediction['y_true'].reshape(-1), labels[indices], rtol=2e-6, atol=2e-6)
        computed = score(prediction['y_true'], prediction['y_pred'])
        np.testing.assert_allclose(computed['rmse'], best['validation_rmse'], rtol=1e-6, atol=1e-8)
    updates = sum(r['training_steps'] for r in history)
    expected_updates = expected_epochs * int(np.ceil(len(entry['train_indices']) / 24))
    if updates != expected_updates:
        raise ValueError('update count ' + str(folder))
    if len(history) == 70 and [r['stage'] for r in history] != ['head'] * 10 + ['joint'] * 60:
        raise ValueError('training recipe ' + str(folder))
    return dict(folder=str(folder), task=task, fold=fold, epochs=len(history), updates=updates,
                best_epoch=best['epoch'], validation=computed), checkpoint


class FixedValidation:
    def __init__(self, source, statistics):
        self.source, self.statistics = source, statistics

    def __len__(self):
        return (len(self.source) + 83) // 84

    def __getitem__(self, batch):
        samples = []
        for index in range(batch * 84, min((batch + 1) * 84, len(self.source))):
            topology, trimer, smiles, static, _, identity = pretrain_source_fields(
                self.source, index, include_identity=True)
            samples.append(prepare_mcl_ph_sample(topology, trimer, smiles, seed=42,
                key=self.source.samples[index][0].hex(), position=index, static=static,
                identity=identity, statistics=self.statistics, sigma=.03, ratio=.3))
        return mcl_ph_collate(samples)


def review_pretrain(arm, statistics):
    folder = OUT / 'R2/pretrain' / arm
    runtime = read(folder / 'runtime.json')
    if runtime['status'] != 'PASS' or runtime['completed_steps'] != 5000:
        raise ValueError('pretraining incomplete ' + arm)
    resume = torch.load(folder / 'resume_05000.pt', map_location='cpu', weights_only=False)
    package = torch.load(folder / 'deploy_05000.pt', map_location='cpu', weights_only=False)
    finite(resume['model']); finite(resume['optimizer']); finite(package)
    if resume['step'] != 5000 or resume['next_position'] != 5040000 or len(resume['rng']) != 4:
        raise ValueError('pretraining position/RNG ' + arm)
    objective = 'expert' if arm == 'M0' else 'expert_mixed_50'
    if package['pretraining_objective']['local'] != objective or resume['identity']['config']['local_objective'] != objective:
        raise ValueError('objective metadata ' + arm)
    encoder = {k.removeprefix('encoder.'):v for k,v in resume['model'].items() if k.startswith('encoder.')}
    if encoder.keys() != package['state_dict'].keys() or any(not torch.equal(v, package['state_dict'][k]) for k,v in encoder.items()):
        raise ValueError('deployment differs from trained encoder ' + arm)
    for rank in range(4):
        events = [json.loads(line) for line in (folder / f'stages_rank{rank}.log').read_text().splitlines()]
        steps = [r['step'] for r in events if r['stage'] == 'step' and r['event'] == 'complete']
        if steps != list(range(1, 5001)):
            raise ValueError('rank update history ' + arm)
    model = MCLPHPretrainer('xattn', dropout=.1, local_objective=objective)
    model.load_state_dict(resume['model'], strict=True)
    model.set_router_mode('top2', 5000)
    model.cuda().eval()
    del resume, encoder
    indices = load_sample_index_artifact(ROOT / 'results/mcl_ph_next_20260929/p0/direct_indices_v1/summary.json', 'fixed_validation')['indices'].tolist()
    if len(indices) != 1024:
        raise ValueError('fixed validation count')
    source, _ = open_source(PI, PI, dual_static_root=PI, selected_indices=indices)
    processed = 0
    sums, counts = np.zeros(2), np.zeros(2)
    usage, valid = np.zeros(3), 0
    try:
        with torch.inference_mode():
            for data, labels in pretrain_loader(FixedValidation(source, statistics), workers=8, seed=42, rank=0):
                processed += int(data.graph_available.numel())
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    report = model(data.cuda(), move_labels(labels, torch.device('cuda')))
                finite(report['atom_sum']); finite(report['geo_sum'])
                sums += [float(report['atom_sum']), float(report['geo_sum'])]
                counts += [int(report['atom_count']), int(report['geo_count'])]
                flag = report['readout_valid'].bool()
                usage += torch.bincount(report['router_top_k'][flag].reshape(-1), minlength=3).cpu().numpy()
                valid += int(flag.sum())
        if processed != 1024:
            raise ValueError('fixed validation coverage ' + arm)
    finally:
        source.close()
        del model
        gc.collect(); torch.cuda.empty_cache()
    return dict(arm=arm, step=5000, updates=5000, objective=objective, evaluated_rows=processed,
                evaluation_calls=1, loss_means=(sums / np.maximum(counts, 1)).tolist(),
                valid_graph_hard_rates=(usage / max(1, valid)).tolist()), package


def predict_formal(task, fold, package, checkpoint, statistics, folder):
    """Single frozen best prediction; no training or repeated validation."""
    folder.mkdir(parents=True, exist_ok=False)
    manifest = read(SPLITS / (task + '.json'))
    indices = sorted(manifest['folds'][fold]['test_indices'])
    source, frame = open_source(DOWN, DOWN, task=task, dual_static_root=DOWN,
        selected_indices=indices, split_manifest=manifest,
        record_index_path=DOWN / (task + '_offsets.npy'), expected_task_rows=manifest['sample_count'])
    try:
        model = build_mcl_arm('xattn', package, expected_step=5000)
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        model.cuda().eval()
        raw = frame['label'].to_numpy(dtype=float)
        scaler_state = StandardScaler()
        scaler_state.mean_ = np.asarray(checkpoint['scaler_mean'])
        scaler_state.scale_ = np.asarray(checkpoint['scaler_scale'])
        scaler_state.var_ = scaler_state.scale_ ** 2
        scaler_state.n_features_in_ = 1
        scaler = TargetScaler(task, scaler_state, transform_mode='standard')
        dataset = MCLPHCleanDataset(source, raw, statistics)
        dataset.set_target_override(scaler.transform(raw).reshape(-1))
        loader = DataLoader(dataset, batch_size=24, num_workers=0, shuffle=False, collate_fn=mcl_ph_downstream_collate)
        _, _, truth, predicted = evaluate(AdaptationAdapter(model), loader, nn.MSELoss(), torch.device('cuda'), scaler, amp_dtype='fp32')
        result = score(truth, predicted)
        np.savez(folder / 'test_predictions.npz', y_true=np.asarray(truth).reshape(-1),
                 y_pred=np.asarray(predicted).reshape(-1), test_indices=np.asarray(indices),
                 sample_keys=np.asarray([k.hex() for k, _ in source.samples]),
                 best_epoch=checkpoint['best_epoch'], outer_test='RUN')
        json_write(folder / 'metrics.json', dict(result, task=task, fold=fold, optimizer_updates=0,
                   checkpoint_epoch=checkpoint['best_epoch'], samples=len(indices)))
        return result
    finally:
        source.close()
        gc.collect(); torch.cuda.empty_cache()


def closeout_saved_report(path):
    """Finish pending saved-result review; never run tests or model inference."""
    path = Path(path).resolve()
    if REVISION != 'r3' or path != OUT / 'final_review/report.json':
        raise ValueError('saved closeout is limited to this r3 final report')
    report = read(path)
    if (report['status'] != 'FAILED_FINAL_TEST' or report.get('error') != "KeyError: 'status'"
            or report['cpu']['exit_code'] != 0 or not report.get('paired_initial_state_exact')
            or {r['arm'] for r in report.get('pretraining', [])} != {'M0', 'Mmix'}
            or any(r['evaluated_rows'] != 1024 or r['evaluation_calls'] != 1 for r in report['pretraining'])
            or report['R2_selection']['eligible_for_final_review']):
        raise ValueError('saved evidence does not describe the completed r3 evaluations and reader-only failure')
    output = path.parent / 'report_closeout.json'
    if output.exists():
        raise FileExistsError('saved closeout already exists; no reentry')
    report['initial_final_failure'] = dict(report=str(path), error=report.pop('error'))
    report['closeout_scope'] = '修复metrics/runtime消费关系，仅完成未执行的保存预测审查；0新增测试／模型调用／推理'
    data = {task:task_rows(task) for task in ('eea', 'eps')}
    report['R2_units'] = []
    for folder, unit in read(OUT / 'R2/scope.json')['units'].items():
        result, _ = review_ft(Path(folder), unit['task'], unit['fold'], data[unit['task']])
        report['R2_units'].append(result)
    # Both tasks failed their predeclared gates: no candidate inference.
    json_write(path.parent / 'frozen_test_selection.json', dict(plan=PLAN, revision=REVISION,
        status='FROZEN', units=[], training_updates=0, reason='Eea confirmation and EPS screening rejected'))
    seven = read(ROOT / 'results/mcl_ph_opt_20261006/final_review/seven_task_results.json')
    reasons = dict(eea='未完成／不可比较：后三折门槛未通过；没有新正式预测',
                   eps='未完成／不可比较：前两折门槛未通过；没有新正式预测')
    for row in seven['rows']:
        row['new_candidate_formal'] = reasons.get(row['task'], row['new_candidate_formal'])
    seven.update(plan=PLAN, revision=REVISION, status='REUSED_VERIFIED_R2_RETAINED_RESULTS',
        candidate_test_inference=False, independent_review='非独立人员审查', project_history_archived=False)
    json_write(path.parent / 'seven_task_results.json', seven)
    report['seven_tasks'] = seven['rows']
    report['formal_tasks'], report['formal'] = [], {}
    report['new_consumption'] = dict(R1_starts=0, R1_epochs=0, R1_updates=0,
        R2_pretrain_starts=len(report['pretrain_execution']),
        R2_pretrain_updates=sum(r['completed_optimizer_updates'] for r in report['pretrain_execution']),
        R2_ft_starts=len(report['R2_units']), R2_ft_epochs=sum(r['epochs'] for r in report['R2_units']),
        R2_ft_updates=sum(r['updates'] for r in report['R2_units']),
        fixed_pretrain_validation_rows=sum(r['evaluated_rows'] for r in report['pretraining']),
        formal_test_units=0, smoke_starts=0, closeout_new_tests=0, closeout_new_model_calls=0)
    report['prior_updates'] = 3110
    report['cumulative_updates'] = 3110 + report['new_consumption']['R2_pretrain_updates'] + report['new_consumption']['R2_ft_updates']
    report['status'] = 'COMPLETE_SAVED_EVIDENCE_CLOSEOUT_CANDIDATES_REJECTED'
    json_write(output, report)
    json_write(OUT / 'execution_status.json', dict(plan=PLAN, revision=REVISION,
        status=report['status'], execution_finished=True, full_plan_completed=True,
        report=str(output), initial_final_failure=report['initial_final_failure'],
        new_consumption=report['new_consumption'], cumulative_updates=report['cumulative_updates'],
        independent_review='待独立审查', project_history_archived=False))
    print(json.dumps(dict(status=report['status'], new_consumption=report['new_consumption'],
                         cumulative_updates=report['cumulative_updates'])), flush=True)


def main():
    global OUT, LOG, REVISION
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--revision', choices=('r2', 'r3'), default='r2')
    parser.add_argument('--run-root', type=Path)
    parser.add_argument('--log-root', type=Path)
    parser.add_argument('--initial-state', type=Path)
    parser.add_argument('--closeout-saved-report', type=Path,
                        help='continue only pending saved-result review after the r3 metrics reader failure')
    args = parser.parse_args()
    OUT, LOG, _ = execution_paths(args.revision, args.run_root, args.log_root, args.initial_state)
    REVISION = args.revision
    require_tmux(); torch.set_num_threads(1)
    if args.closeout_saved_report:
        return closeout_saved_report(args.closeout_saved_report)
    set_global_seed(42)
    # Waiting does not run a test or consume a scientific batch.
    controller_exit = LOG / ('controller_R3/exit.code' if REVISION == 'r3' else 'controller_R2/exit.code')
    while not controller_exit.exists():
        time.sleep(5)
    root = OUT / 'final_review'
    root.mkdir(parents=True, exist_ok=False)
    report = dict(plan=PLAN, revision=REVISION, status='RUNNING', hashes=False,
                  testing_rounds=1, reviewer='Codex 实施后依据实际证据核对；非独立人员审查',
                  scope='本轮改动及实际消费产物；不重放历史全流程', formal_tasks=[])
    json_write(root / 'report.json', report)
    try:
        command = [sys.executable, '-m', 'pytest', '-q',
                   'tests/test_mcl_ph_opt_r3.py' if REVISION == 'r3' else 'tests/test_mcl_ph_opt.py']
        completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
        (root / 'cpu_stdout.log').write_text(completed.stdout)
        (root / 'cpu_stderr.log').write_text(completed.stderr)
        report['cpu'] = dict(command=command, exit_code=completed.returncode, stdout=completed.stdout)
        if completed.returncode:
            raise RuntimeError('final CPU tests failed; see cpu_stdout.log')
        data = {task:task_rows(task) for task in (('eea', 'eps') if REVISION == 'r3' else ('nc', 'eea', 'eps'))}
        report['R1_units'] = []
        for fold in (() if REVISION == 'r3' else (2, 3, 4)):
            folder = OUT / 'R1' / key('development', 'nc', fold, 'SP')
            result, state = review_ft(folder, 'nc', fold, data['nc'], True)
            initial = torch.load(folder / 'head_initial.pt', map_location='cpu', weights_only=False)
            ref_head = torch.load(reference('nc', fold) / 'head_initial.pt', map_location='cpu', weights_only=False)
            if initial.keys() != ref_head.keys() or any(not torch.equal(initial[k],v) for k,v in ref_head.items()):
                raise ValueError('R1 original head initialization differs')
            report['R1_units'].append(result)
        if REVISION != 'r3':
            report['R1_selection'] = read(OUT / 'R1/selection.json')
        report['pretrain_execution'] = pretraining_consumption()
        # Old F0 replay and saved Nc SP are the comparisons actually consumed.
        report['old_A3_consumed'] = []
        for fold in (() if REVISION == 'r3' else (0, 1)):
            for arm in ('F0', 'SP'):
                folder = ROOT / 'results/sota_ft_20261005' / key('development', 'nc', fold, arm)
                result, _ = review_ft(folder, 'nc', fold, data['nc'], True)
                if arm == 'F0':
                    baseline = read(reference('nc', fold) / 'metrics.json')['best_validation_rmse']
                    np.testing.assert_allclose(result['validation']['rmse'], baseline, rtol=.005, atol=0)
                report['old_A3_consumed'].append(result)
        packages, selected_states = {}, {}
        statistics = load_geometric_statistics(STATISTICS)
        r2_completed = int(controller_exit.read_text()) == 0
        if r2_completed:
            initial = [torch.load(OUT / 'R2/pretrain' / arm / 'initial_model.pt', map_location='cpu', weights_only=False) for arm in ('M0', 'Mmix')]
            if initial[0].keys() != initial[1].keys() or any(v.dtype != initial[1][k].dtype or v.shape != initial[1][k].shape or not torch.equal(v, initial[1][k]) for k,v in initial[0].items()):
                raise ValueError('paired full step0 differs')
            if REVISION == 'r3':
                source_initial = torch.load(args.initial_state, map_location='cpu', weights_only=False)
                if source_initial.keys() != initial[0].keys() or any(not torch.equal(v, initial[0][k]) for k,v in source_initial.items()):
                    raise ValueError('paired initial state differs from authorized step0')
                del source_initial
            report['paired_initial_state_exact'] = True
            del initial
            report['pretraining'] = []
            for arm in ('M0', 'Mmix'):
                result, packages[arm] = review_pretrain(arm, statistics)
                report['pretraining'].append(result)
            selection = read(OUT / 'R2/selection.json')
            report['R2_selection'] = selection
            report['R2_units'] = []
            scope = read(OUT / 'R2/scope.json')
            for path, unit in scope['units'].items():
                result, checkpoint = review_ft(Path(path), unit['task'], unit['fold'], data[unit['task']])
                report['R2_units'].append(result)
                if unit['task'] in selection['eligible_for_final_review'] and '/Mmix/' in path:
                    selected_states[unit['task'], unit['fold']] = checkpoint
            report['formal_tasks'] = selection['eligible_for_final_review']
        else:
            report['R2_execution'] = '停止：控制器非零退出；保留现场，不重试或预测该候选'
        # Freeze the complete test allowlist before any new test inference.
        allowed = [dict(task=task, fold=fold, arm='Mmix', head='single best')
                   for task in report['formal_tasks'] for fold in range(5)]
        json_write(root / 'frozen_test_selection.json', dict(plan=PLAN, status='FROZEN', units=allowed, training_updates=0))
        report['formal'] = {}
        for task in report['formal_tasks']:
            rows = []
            for fold in range(5):
                rows.append(predict_formal(task, fold, packages['Mmix'], selected_states[task, fold],
                            statistics, OUT / 'R4' / task / f'fold{fold}'))
            report['formal'][task] = rows
        seven = read(ROOT / ('results/mcl_ph_opt_20261006/final_review/seven_task_results.json'
                            if REVISION == 'r3' else 'results/sota_ft_20261005/seven_task_retained_results.json'))
        for row in seven['rows']:
            task = row['task']
            if task in report['formal']:
                values = report['formal'][task]
                row['project_r2'] = float(np.mean([v['r2'] for v in values]))
                row['fold_r2'] = [v['r2'] for v in values]
                row['fold_rmse'] = [v['rmse'] for v in values]
                row['prediction_paths'] = [str(OUT / 'R4' / task / f'fold{fold}/test_predictions.npz') for fold in range(5)]
                row['new_candidate_formal'] = '五折冻结 Mmix 正式预测完成'
            else:
                # Recompute the seven retained formal rows from their saved predictions once.
                values = []
                manifest, labels, keys = (None, None, None) if REVISION == 'r3' else (data.get(task) or task_rows(task))
                for fold, path in enumerate([] if REVISION == 'r3' else row['prediction_paths']):
                    with np.load(path, allow_pickle=False) as p:
                        indices = p['test_indices'].tolist()
                        if indices != sorted(manifest['folds'][fold]['test_indices']) or p['sample_keys'].tolist() != [keys[i] for i in indices]:
                            raise ValueError('retained test rows ' + task)
                        np.testing.assert_allclose(p['y_true'].reshape(-1), labels[indices], rtol=2e-6, atol=2e-6)
                        values.append(score(p['y_true'], p['y_pred']))
                if REVISION != 'r3':
                    row['project_r2'] = float(np.mean([v['r2'] for v in values]))
                row['new_candidate_formal'] = ('未完成／不可比较：Nc SP 后三折门槛未通过' if task == 'nc'
                    else '未完成／不可比较：Mmix 未晋级或受实际停止条件影响' if task in ('eea', 'eps')
                    else '未完成／不可比较：本轮无该任务新候选，保留已有正式预测')
            row['difference'] = row['project_r2'] - row['paper_r2']
        seven.update(plan=PLAN, revision=REVISION, status='FINAL_SAVED_PREDICTIONS_REVIEWED',
                     candidate_test_inference=bool(report['formal_tasks']), independent_review='非独立人员审查', project_history_archived=False)
        json_write(root / 'seven_task_results.json', seven)
        report['seven_tasks'] = seven['rows']
        report['status'] = 'PASS_FINAL_TESTS' if r2_completed else 'PASS_FINAL_TESTS_EXECUTION_STOPPED_R2'
        report['new_consumption'] = dict(R1_starts=0 if REVISION == 'r3' else 3,
            R1_epochs=0 if REVISION == 'r3' else 210, R1_updates=0 if REVISION == 'r3' else 2310,
            R2_pretrain_starts=len(report['pretrain_execution']),
            R2_pretrain_updates=sum(r['completed_optimizer_updates'] for r in report['pretrain_execution']),
            R2_ft_starts=len(report.get('R2_units', [])),
            R2_ft_epochs=sum(r['epochs'] for r in report.get('R2_units', [])),
            R2_ft_updates=sum(r['updates'] for r in report.get('R2_units', [])),
            fixed_pretrain_validation_rows=sum(r['evaluated_rows'] for r in report.get('pretraining', [])),
            formal_test_units=len(allowed), smoke_starts=0)
        report['prior_updates'] = 3110 if REVISION == 'r3' else 0
        report['cumulative_updates'] = report['prior_updates'] + sum(report['new_consumption'][k]
            for k in ('R1_updates', 'R2_pretrain_updates', 'R2_ft_updates'))
        json_write(root / 'report.json', report)
        json_write(OUT / 'execution_status.json', dict(plan=PLAN, revision=REVISION,
            status=report['status'], execution_finished=True, full_plan_completed=r2_completed,
            new_consumption=report['new_consumption'], cumulative_updates=report['cumulative_updates'],
            independent_review='待独立审查', project_history_archived=False))
        print(json.dumps(dict(status=report['status'], formal_tasks=report['formal_tasks'], consumption=report['new_consumption'])), flush=True)
    except BaseException as error:
        report.update(status='FAILED_FINAL_TEST', error=f'{type(error).__name__}: {error}')
        json_write(root / 'report.json', report)
        raise


if __name__ == '__main__':
    main()
