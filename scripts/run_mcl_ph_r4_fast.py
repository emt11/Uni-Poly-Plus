#!/usr/bin/env python3
"""R4 bounded campaign: two packs, validation gates, frozen seven-task evaluation.

Every trainer has its own tmux window and immutable log/output directory.
Failures stop this campaign; successful units are never retrained or retried.
"""
import argparse
from collections import deque
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.training.glt_dual_runtime import require_tmux

LOG_ROOT = ROOT / 'logs/mcl_ph_r4_20261002'
OUT_ROOT = ROOT / 'results/mcl_ph_r4_20261002'
PI1M = ROOT / 'data/processed/mcl_ph_r4_20260930/pi1m_direct_01.staging'
DOWNSTREAM = ROOT / 'data/processed/mcl_ph_r4_20260930/downstream_direct_01.staging'
STATISTICS = ROOT / 'data/processed/mcl_ph_r4_20261001/statistics_train_full.npz'
SPLITS = ROOT / 'data/splits/mips_paper5_outer'
TASKS = ('eea', 'egb', 'egc', 'ei', 'eps', 'nc', 'xc')
DEV_TASKS = ('ei', 'eps', 'nc')
PACK_NAMES = {'glt': 'r3_glt_ref_fast_ra_01', 'mcl': 'r3_m_xattn_fast_01'}


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    os.replace(temporary, path)


def finite(value):
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError('non-finite value in result')
    if isinstance(value, dict):
        for item in value.values():
            finite(item)
    if isinstance(value, (list, tuple)):
        for item in value:
            finite(item)


def epochs(task):
    return 60 if task == 'egc' else 70


def route_gate(gains, *, d2=False):
    gains = np.asarray(gains, dtype=np.float64)
    if gains.shape != ((3,) if d2 else (3, 2)) or not np.isfinite(gains).all():
        raise ValueError('incomplete/non-finite route comparison')
    mean = float(gains.mean())
    positive = int((gains > 0).sum())
    task_means = gains if d2 else gains.mean(axis=1)
    return dict(passed=bool(mean >= .01 and positive >= (2 if d2 else 4)
                            and np.min(task_means) >= -.02),
                mean_relative_rmse_gain=mean, positive_units=positive,
                task_mean_relative_gains=task_means.tolist(), gains=gains.tolist())


def choose_routes(values, retained):
    """Two-fold task means select routes; the combined six-unit gate is fixed."""
    order = ('glt_ref', 'o8_only', retained)
    selected = {task: min(order, key=lambda arm: np.mean(values[arm][task]))
                for task in DEV_TASKS}
    gains = [[(reference - candidate) / reference
              for reference, candidate in zip(values['glt_ref'][task], values[selected[task]][task])]
             for task in DEV_TASKS]
    gate = route_gate(gains)
    return (selected if gate['passed'] else dict.fromkeys(DEV_TASKS, 'glt_ref')), gate, selected


def parse_rank_progress(path):
    if not path.exists():
        return {}
    with path.open('rb') as handle:
        handle.seek(max(0, path.stat().st_size - 250000))
        text = handle.read().decode(errors='replace')
    decoder, ranks, timings = json.JSONDecoder(), {}, []
    for match in re.finditer(r'\{"step":', text):
        try:
            record, _ = decoder.raw_decode(text[match.start():])
        except ValueError:
            continue
        if 'rank' not in record:
            continue
        finite(record.get('losses'))
        ranks[str(record['rank'])] = max(ranks.get(str(record['rank']), 0), int(record['step']))
        seconds = record.get('step_seconds', record.get('cpu_observed_timing', {}).get('step_seconds'))
        if record['rank'] == 0 and seconds:
            timings.append(float(seconds))
    step = min(ranks.values()) if len(ranks) == 4 else 0
    seconds = float(np.mean(timings[-50:])) if timings else None
    return dict(rank_updates=ranks, all_rank_updates=step, target_updates=5000,
                recent_step_seconds=seconds,
                current_pack_remaining_hours=(5000-step)*seconds/3600 if seconds else None)


def write_shell(log, command, *, gpu=None):
    log.mkdir(parents=True, exist_ok=False)
    environment = 'export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1\n'
    if gpu is not None:
        environment += f'export CUDA_VISIBLE_DEVICES={int(gpu)}\n'
    script = ('#!/bin/bash\ncd ' + shlex.quote(str(ROOT)) + ' || exit 1\nD=' +
              shlex.quote(str(log)) + '\nif ! mkdir "$D/attempt.started"; then exit 90; fi\n'
              'pwd > "$D/cwd.txt"\ndate -u +%FT%TZ > "$D/started_at.txt"\n' + environment +
              shlex.join(command) + ' > "$D/stdout.log" 2> "$D/stderr.log"\ncode=$?\n'
              'printf "%s\\n" "$code" > "$D/exit.code"\n'
              'date -u +%FT%TZ > "$D/finished_at.txt"\nexit "$code"\n')
    (log / 'run.sh').write_text(script)
    (log / 'command.txt').write_text(shlex.join(command) + '\n')
    (log / 'cwd.txt').write_text(str(ROOT) + '\n')


def start_window(name, log):
    subprocess.run(['tmux', 'new-window', '-t', 'Uni-Poly', '-n', name,
                    'bash ' + shlex.quote(str(log / 'run.sh'))], check=True)


def audit_unit(folder, *, stage, arm, task, fold):
    """Recompute metrics from saved predictions and check actual fold identities."""
    from src.training.glt_dual_runtime import open_source
    from src.training.split_protocol import resolve_fold
    metrics = json.loads((folder / 'metrics.json').read_text())
    runtime = json.loads((folder / 'runtime.json').read_text())
    finite(metrics)
    if runtime.get('status') != 'PASS' or runtime.get('exit_code') != 0:
        raise ValueError('failed unit: ' + str(folder))
    for key, expected in dict(arm=arm, task=task, fold=fold, stage=stage,
                              executed_epochs=epochs(task), pretrain_step=5000,
                              source_identity_contract='direct',
                              target_transform='identity' if task == 'xc' else 'standard').items():
        if metrics.get(key) != expected:
            raise ValueError(f'unit contract mismatch {folder}: {key}')
    history = metrics['history']
    if len(history) != epochs(task) or not (folder / 'best.pt').is_file():
        raise ValueError('incomplete unit history/checkpoint')
    best = min(history, key=lambda entry: entry['validation_rmse'])
    if (best['epoch'] != metrics['best_epoch'] or
            best['validation_rmse'] != metrics['best_validation_rmse'] or
            sum(entry['training_steps'] for entry in history) != metrics['optimizer_updates']):
        raise ValueError('validation selection or optimizer update ledger differs')
    if [entry['stage'] for entry in history] != ['head']*10 + ['joint']*(epochs(task)-10):
        raise ValueError('head/joint training schedule differs')
    import torch
    state = torch.load(folder / 'best.pt', map_location='cpu', weights_only=False)
    for name, tensor in state['state_dict'].items():
        if not torch.isfinite(tensor).all():
            raise ValueError('non-finite best-state tensor: ' + name)
    for key in ('arm', 'task', 'fold', 'best_epoch', 'best_validation_rmse', 'executed_epochs',
                'optimizer_updates', 'target_transform'):
        if state[key] != metrics[key]:
            raise ValueError('best checkpoint metadata differs: ' + key)
    manifest = json.loads((SPLITS / f'{task}.json').read_text())
    _, validation, _ = resolve_fold(manifest, task, fold,
        cohort_rows=manifest['sample_count'], expected_protocol='paper5_outer')
    with np.load(folder / 'validation_predictions.npz', allow_pickle=False) as prediction:
        truth = prediction['y_true'].reshape(-1)
        predicted = prediction['y_pred'].reshape(-1)
        if (prediction['validation_indices'].tolist() != validation or
                len(truth) != len(validation) or not np.isfinite(truth).all() or
                not np.isfinite(predicted).all()):
            raise ValueError('validation prediction contract differs')
        rmse = float(np.sqrt(np.mean((predicted-truth)**2)))
        if not np.isclose(rmse, metrics['best_validation_rmse'], rtol=1e-6, atol=1e-7):
            raise ValueError('validation RMSE differs from saved predictions')
        selected = sorted(validation)
        source, frame = open_source(DOWNSTREAM, DOWNSTREAM, task=task,
            dual_static_root=DOWNSTREAM, selected_indices=selected,
            expected_task_rows=manifest['sample_count'],
            record_index_path=DOWNSTREAM / f'{task}_offsets.npy', split_manifest=manifest)
        try:
            keys = dict(zip(selected, [sample[0].hex() for sample in source.samples]))
            labels = dict(zip(selected, frame['label'].to_numpy(dtype=np.float64)))
            if prediction['sample_keys'].tolist() != [keys[i] for i in validation]:
                raise ValueError('validation keys differ from direct source')
            # Saved truth has the trainer's float32 standardized round trip.
            np.testing.assert_allclose(truth, [labels[i] for i in validation], rtol=2e-6, atol=2e-6)
        finally:
            source.close()
    if stage == 'development':
        if metrics['outer_test'] != 'NOT_RUN' or (folder / 'test_predictions.npz').exists():
            raise ValueError('development opened outer-test')
    else:
        indices = sorted(manifest['folds'][fold]['test_indices'])
        with np.load(folder / 'test_predictions.npz', allow_pickle=False) as prediction:
            truth, predicted = prediction['y_true'].reshape(-1), prediction['y_pred'].reshape(-1)
            if (prediction['test_indices'].tolist() != indices or len(truth) != len(indices)
                    or not np.isfinite(truth).all() or not np.isfinite(predicted).all()):
                raise ValueError('formal prediction contract differs')
            r2 = float(1-np.sum((predicted-truth)**2)/np.sum((truth-truth.mean())**2))
            if not np.isclose(r2, metrics['test_r2'], rtol=1e-6, atol=1e-7):
                raise ValueError('formal R2 differs from saved predictions')
            source, frame = open_source(DOWNSTREAM, DOWNSTREAM, task=task,
                dual_static_root=DOWNSTREAM, selected_indices=indices,
                expected_task_rows=manifest['sample_count'],
                record_index_path=DOWNSTREAM / f'{task}_offsets.npy', split_manifest=manifest)
            try:
                if prediction['sample_keys'].tolist() != [sample[0].hex() for sample in source.samples]:
                    raise ValueError('test keys differ from direct source')
                np.testing.assert_allclose(truth, frame['label'].to_numpy(), rtol=2e-6, atol=2e-6)
            finally:
                source.close()
    report = dict(status='PASS', audited_at=timestamp(), metric_recomputed=True,
                  actual_validation_identity=True, outer_test=metrics['outer_test'])
    atomic_json(folder / 'acceptance.json', report)
    return metrics


class Campaign:
    def __init__(self, name):
        self.log = LOG_ROOT / name
        self.out = OUT_ROOT / name
        self.log.mkdir(parents=True, exist_ok=False)
        self.out.mkdir(parents=True, exist_ok=False)
        self.units = {}
        self.state = dict(status='RUNNING', started_at=timestamp(), pid=os.getpid(),
                          scientific_pretrain_updates=10000, cancelled_pretrain_updates=835,
                          cancelled_inflight_updates='0..1 per cancelled attempt; not independently countable',
                          prior_smoke_pretrain_updates=2, units=[], running_units=[])
        self.publish()

    def publish(self, **changes):
        self.state.update(changes, updated_at=timestamp())
        self.state['units'] = list(self.units.values())
        atomic_json(self.log / 'status.json', self.state)

    def pretrain(self, route):
        name = PACK_NAMES[route]
        log, out = LOG_ROOT / name, OUT_ROOT / name
        self.publish(stage='R3', current_package=route, package_acceptance='PENDING',
                     progress={}, current_log=str(log))
        if route == 'glt':
            # User-authorized fast GLT was launched before controller creation.
            if not (log / 'attempt.started').is_dir():
                raise ValueError('expected single fast GLT attempt is absent')
            config = ROOT / 'results/mcl_ph_r4_20261001/resolved/glt_ref.json'
            (log / 'resolved_config.json').write_text(config.read_text())
        else:
            config = self.out / 'm_xattn_fast.json'
            value = json.loads((OUT_ROOT / 'resolved/m_xattn_cached.json').read_text())
            value['save_every'] = 1000  # operational safety; targets/sampling unchanged
            atomic_json(config, value)
            capacity = json.loads((LOG_ROOT / 'mcl_worker_capacity_01/report.json').read_text())
            if capacity.get('status') != 'PASS' or capacity['best_workers'] not in (8, 12, 16):
                raise ValueError('MCL worker capacity/parity probe is not verified')
            identity_log = LOG_ROOT / 'mcl_identity_speed_01'
            if ((identity_log / 'exit.code').read_text().strip() != '0' or
                    json.loads((identity_log / 'report.json').read_text()).get('status') != 'PASS'):
                raise ValueError('MCL validated-identity input parity is not verified')
            io_log = LOG_ROOT / 'mcl_mapping_io_01'
            if ((io_log / 'exit.code').read_text().strip() != '0' or
                    json.loads((io_log / 'report.json').read_text()).get('status') != 'PASS'):
                raise ValueError('mapping and readonly readahead input parity is not verified')
            workers = capacity['best_workers']
            command = ['torchrun', '--standalone', '--nproc_per_node=4',
                'scripts/pretrain_mcl_ph.py', '--config', str(config),
                '--cohort-root', str(PI1M), '--cache-root', str(PI1M),
                '--dual-static-root', str(PI1M), '--statistics', str(STATISTICS),
                '--shared-new-init', str(ROOT / 'results/mcl_ph_r4_20261001/shared_new_init_v2.pt'),
                '--output', str(out), '--prep-workers', str(workers), '--lmdb-readahead']
            write_shell(log, command)
            (log / 'resolved_config.json').write_text(config.read_text())
            start_window('mcl_r4_mcl_fast', log)
            self.publish(prep_workers_per_rank=workers, total_prep_workers=4*workers)
        while not (log / 'exit.code').exists():
            self.publish(stage='R3', current_package=route,
                         progress=parse_rank_progress(log / 'stdout.log'), current_log=str(log))
            time.sleep(10)
        if (log / 'exit.code').read_text().strip() != '0':
            raise RuntimeError('pretraining failed: ' + name)
        command = [sys.executable, 'scripts/audit_mcl_ph_fast_pretrain.py', name]
        (log / 'accept_command.txt').write_text(shlex.join(command)+'\n')
        with (log / 'accept.stdout.log').open('x') as stdout, (log / 'accept.stderr.log').open('x') as stderr:
            result = subprocess.run(command, cwd=ROOT, stdout=stdout, stderr=stderr)
        (log / 'accept.exit.code').write_text(str(result.returncode)+'\n')
        if result.returncode != 0:
            raise RuntimeError('pretraining acceptance failed: ' + name)
        self.publish(stage='R3', current_package=route, package_acceptance='PASS')
        return config

    def batch(self, stage, requested, *, selection=None):
        queue = deque(unit for unit in requested if (stage, *unit) not in self.units)
        active = {}
        while queue or active:
            while queue and len(active) < 4:
                arm, task, fold = queue.popleft()
                gpu = next(g for g in range(4) if g not in active)
                key = (stage, arm, task, fold)
                folder = self.out / stage / arm / task / f'fold{fold}'
                log = self.log / stage / arm / task / f'fold{fold}'
                route = 'glt' if arm in ('glt_ref', 'o8_only') else 'mcl'
                checkpoint = OUT_ROOT / PACK_NAMES[route] / 'deploy_05000.pt'
                command = [sys.executable, 'scripts/finetune_mcl_ph.py', '--arm', arm,
                    '--stage', 'development' if stage != 'paper5_outer' else 'paper5_outer',
                    '--config', str(self.configs[route]), '--checkpoint', str(checkpoint),
                    '--expected-pretrain-step', '5000', '--cohort-root', str(DOWNSTREAM),
                    '--cache-root', str(DOWNSTREAM), '--dual-static-root', str(DOWNSTREAM),
                    '--split-root', str(SPLITS), '--cohort-index', str(DOWNSTREAM / f'{task}_offsets.npy'),
                    '--statistics', str(STATISTICS), '--task', task, '--fold', str(fold),
                    '--epochs', str(epochs(task)), '--output', str(self.out / stage)]
                reused = self.units.get(('development', arm, task, fold)) if stage == 'paper5_outer' else None
                if selection:
                    command.extend(['--frozen-selection', str(selection)])
                if reused:
                    command.extend(['--reuse-development-unit', reused['folder']])
                write_shell(log, command, gpu=gpu)
                window = f'mcl_r4_ft_{gpu}_{arm}_{task}_{fold}'
                start_window(window, log)
                active[gpu] = dict(key=key, arm=arm, task=task, fold=fold, folder=str(folder),
                                   log=str(log), window=window, reused=bool(reused), started_at=timestamp())
            self.publish(unit_stage=stage, running_units=list(active.values()), pending_units=len(queue))
            for gpu, unit in list(active.items()):
                log = Path(unit['log'])
                if not (log / 'exit.code').exists():
                    continue
                if (log / 'exit.code').read_text().strip() != '0':
                    raise RuntimeError('fine-tuning failed: ' + str(unit))
                metrics = audit_unit(Path(unit['folder']),
                    stage='paper5_outer' if stage == 'paper5_outer' else 'development',
                    arm=unit['arm'], task=unit['task'], fold=unit['fold'])
                self.units[tuple(unit['key'])] = dict(unit, validation_rmse=metrics['best_validation_rmse'],
                    test_r2=metrics.get('test_r2'), executed_epochs=metrics['executed_epochs'],
                    optimizer_updates=(0 if unit['reused'] else metrics['optimizer_updates']),
                    acceptance='PASS', finished_at=timestamp())
                del active[gpu]
                atomic_json(self.log / 'units.json', list(self.units.values()))
            time.sleep(5 if active else .1)
        self.publish(running_units=[], pending_units=0)

    def rmse(self, arm, task, fold):
        return self.units[('development', arm, task, fold)]['validation_rmse']

    def run(self):
        # Local tests and actual clean-input parity must pass before fine-tuning.
        self.configs = dict(glt=self.pretrain('glt'), mcl=self.pretrain('mcl'))
        self.publish(current_package=None, current_log=None, progress=None,
                     pretrain_acceptance=dict(glt='PASS', mcl='PASS'))
        for name in ('fast_finetune_tests_01', 'clean_input_probe_01', 'fast_campaign_tests_02'):
            if (LOG_ROOT / name / 'exit.code').read_text().strip() != '0':
                raise ValueError('speed change verification failed: ' + name)
        self.publish(stage='D1')
        self.batch('development', [(arm, task, 0) for arm in ('m_o8_only', 'm_xattn')
                                   for task in ('ei', 'nc')])
        self.publish(stage='D2')
        self.batch('development', [(arm, task, 0) for arm in ('m_xattn', 'm_xattn_scale')
                                   for task in DEV_TASKS])
        d2 = route_gate([(self.rmse('m_xattn', task, 0)-self.rmse('m_xattn_scale', task, 0))
                         / self.rmse('m_xattn', task, 0) for task in DEV_TASKS], d2=True)
        retained = 'm_xattn_scale' if d2['passed'] else 'm_xattn'
        atomic_json(self.out / 'd2_selection.json', dict(d2, retained=retained))
        self.publish(stage='D3', d2_retained=retained)
        self.batch('development', [(arm, task, fold) for arm in ('glt_ref', 'o8_only', retained)
                                   for task in DEV_TASKS for fold in (0, 1)])
        dev_units = [unit for key, unit in self.units.items() if key[0] == 'development']
        if len(dev_units) > 23 or sum(unit['executed_epochs'] for unit in dev_units) > 1610:
            raise ValueError('development budget exceeded')
        values = {arm: {task: [self.rmse(arm, task, fold) for fold in (0, 1)] for task in DEV_TASKS}
                  for arm in ('glt_ref', 'o8_only', retained)}
        routes, d3, proposed = choose_routes(values, retained)
        final_routes = dict.fromkeys(TASKS, 'glt_ref')
        final_routes.update(routes, eea='m_xattn')
        formal = [(arm, task, fold) for task in TASKS
                  for arm in dict.fromkeys(('glt_ref', final_routes[task])) for fold in range(5)]
        if len(formal) > 55 or sum(epochs(task) for _, task, _ in formal) > 3800:
            raise ValueError('formal logical budget exceeded')
        selection = self.out / 'frozen_selection.json'
        atomic_json(selection, dict(status='FROZEN', frozen_at=timestamp(), d2=d2, d3=d3,
            d3_proposed_routes=proposed, final_routes=final_routes, reference='glt_ref',
            allowed_units=[f'{arm}/{task}/fold{fold}' for arm, task, fold in formal],
            validation_only=True, formal_units=len(formal),
            split='paper5_outer', seed='42+fold', target='standard; Xc identity',
            development_scope='Ei/EPS/Nc folds0/1; these folds are not an independent blind test'))
        self.publish(stage='R5', frozen_selection=str(selection), final_routes=final_routes)
        # Large tasks first avoids leaving the longest task as the last job.
        formal.sort(key=lambda unit: -json.loads((SPLITS / f'{unit[1]}.json').read_text())['sample_count'])
        self.batch('paper5_outer', formal, selection=selection)
        report = self.aggregate(final_routes)
        atomic_json(self.out / 'seven_task_results.json', report)
        self.publish(status='COMPLETED_PENDING_REVIEW', stage='R5', finished_at=timestamp(),
                     result=str(self.out / 'seven_task_results.json'),
                     formal_units=len(formal), development_units=len(dev_units))
        self.record_result(report)

    def aggregate(self, routes):
        paper = dict(eea=.934, egb=.923, egc=.911, ei=.839, eps=.787, nc=.856, xc=.423)
        rows = []
        for task in TASKS:
            reference = [self.units[('paper5_outer', 'glt_ref', task, fold)]['test_r2'] for fold in range(5)]
            candidate = [self.units[('paper5_outer', routes[task], task, fold)]['test_r2'] for fold in range(5)]
            rows.append(dict(task=task, selected_arm=routes[task], reference_fold_r2=reference,
                selected_fold_r2=candidate, reference_mean_r2=float(np.mean(reference)),
                selected_mean_r2=float(np.mean(candidate)), selected_std_r2=float(np.std(candidate)),
                paper_best_r2=paper[task], project_minus_paper=float(np.mean(candidate))-paper[task],
                paper_model='Periodic-TDL (+Chem, +RC)' if task in ('egb', 'egc') else 'Periodic-TDL',
                prediction_paths=[str(self.out / 'paper5_outer' / routes[task] / task /
                                      f'fold{fold}' / 'test_predictions.npz') for fold in range(5)]))
        return dict(status='PASS', rows=rows, completed_at=timestamp(),
            summary='five-fold arithmetic mean R2, one initialization seed 42+fold, not pooled R2',
            protocol_caveat='development folds were used for selection; no independent blind-test claim; '
                'paper dataset/split/pretraining equivalence unverified, numerical comparison only',
            frozen_selection=str(self.out / 'frozen_selection.json'),
            source=str(DOWNSTREAM), paper_source=str(ROOT / 'SOTA_RESULT.md'))

    def record_result(self, report):
        # Preserve all earlier handoff text; final independent human/Codex review
        # is recorded separately before any PROJECT_HISTORY architecture archive.
        plan = ROOT / 'CURRENT_PLAN.md'
        text = plan.read_text()
        begin = text.index('状态：')
        end = text.index('\n\n', begin)
        text = text[:begin] + ('状态：**待审查／快速预训练、开发筛选及七任务正式复评均已结束**。'
            '自动审计已按实际保存预测复算指标并核对身份；Codex需独立核对diff、日志、checkpoint与预算后归档。'
            f'最新结果 `{self.out / "seven_task_results.json"}`，详细记录见R4-19。') + text[end:]
        section = '\n\n## R4-19. 快速全流程完成，独立审查待完成\n\n'
        section += f'- 自动流程结束于 `{timestamp()}`；日志 `{self.log}`，结果 `{self.out}`。\n'
        section += '- 状态：待审查。实际预测已经复算指标并核对来源身份；尚待Codex独立核对完整diff、日志和预算后归档。\n'
        section += '- 七任务为五折R²算术均值、seed42+fold；开发折参与选择，不是独立盲测。论文协议等价性未核实，仅数值对照。\n\n'
        section += '|任务|冻结路线|项目R²|论文模型|论文R²|项目−论文|\n|---|---|---:|---|---:|---:|\n'
        for row in report['rows']:
            section += f"|{row['task']}|{row['selected_arm']}|{row['selected_mean_r2']:.3f}|{row['paper_model']}|{row['paper_best_r2']:.3f}|{row['project_minus_paper']:+.3f}|\n"
        plan.write_text(text + section)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', default='fast_campaign_01')
    args = parser.parse_args()
    require_tmux()
    os.chdir(ROOT)
    campaign = Campaign(args.name)
    try:
        campaign.run()
    except BaseException as error:
        campaign.publish(status='STOPPED', error=f'{type(error).__name__}: {error}',
                         stopped_at=timestamp(), automatic_retry=False)
        if campaign.state.get('stage') == 'R3':
            window = 'mcl_r4_glt_fast' if campaign.state.get('current_package') == 'glt' else 'mcl_r4_mcl_fast'
            subprocess.run(['tmux', 'send-keys', '-t', 'Uni-Poly:'+window, 'C-c'], check=False)
        # Stop only the formal trainers registered to this campaign.
        for unit in campaign.state.get('running_units', []):
            subprocess.run(['tmux', 'send-keys', '-t', 'Uni-Poly:'+unit['window'], 'C-c'], check=False)
        raise


if __name__ == '__main__':
    main()
