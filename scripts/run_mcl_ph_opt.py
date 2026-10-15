#!/usr/bin/env python3
"""MCL-PH optimization: bounded, fresh outputs; no hash or smoke gates."""
import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_mcl_ph_r4_fast import write_shell, start_window
from scripts.finetune_mcl_ph import compact_train_validation_indices
from src.training.glt_dual_runtime import open_source, require_tmux
from src.training.sota_ft_contract import DOWN, SPLITS, key, recipe, reference, read
from src.training.sota_ft_runtime import json_write

PLAN = 'MCL-PH-OPT-20261006-01'
OUT = ROOT / 'results/mcl_ph_opt_20261006'
LOG = ROOT / 'logs/mcl_ph_opt_20261006/execute_r2'
PI = ROOT / 'data/processed/mcl_ph_r4_20260930/pi1m_direct_01.staging'
STATISTICS = ROOT / 'data/processed/mcl_ph_r4_20261001/statistics_train_full.npz'
SHARED = ROOT / 'results/mcl_ph_r4_20261001/shared_new_init_v2.pt'
REVISION = 'r2'
INITIAL_STATE = None
WINDOW_PREFIX = 'mcl_opt_'


def execution_paths(revision, run_root, log_root, initial_state):
    if revision == 'r3':
        expected = (ROOT / 'results/mcl_ph_opt_20261006/r3', ROOT / 'logs/mcl_ph_opt_20261006/r3',
                    ROOT / 'results/mcl_ph_opt_20261006/R2/pretrain/M0/initial_model.pt')
        actual = tuple(Path(p).resolve() if p else None for p in (run_root, log_root, initial_state))
        if actual != expected:
            raise ValueError('r3 requires its authorized fresh roots and the saved r2 complete step0')
        return actual
    return (Path(run_root).resolve() if run_root else OUT,
            Path(log_root).resolve() if log_root else LOG,
            Path(initial_state).resolve() if initial_state else None)


def launch(name, command, gpu=None):
    log = LOG / name
    write_shell(log, command, gpu=gpu)
    start_window(WINDOW_PREFIX + name, log)
    return log


def jobs(units):
    """Independent folds own distinct files and GPU windows."""
    active = {}
    pending = list(units)
    while pending or active:
        for gpu in (0, 1, 2):
            if pending and gpu not in active:
                name, command = pending.pop(0)
                active[gpu] = (name, launch(name, command, gpu))
        for gpu, (name, log) in list(active.items()):
            exit_path = log / 'exit.code'
            if not exit_path.exists():
                continue
            code = int(exit_path.read_text())
            if code:
                # Stop only this controller's other training windows.
                for other_name, other_log in active.values():
                    if not (other_log / 'exit.code').exists():
                        subprocess.run(['tmux', 'send-keys', '-t',
                                        'Uni-Poly:' + WINDOW_PREFIX + other_name, 'C-c'], check=False)
                raise RuntimeError(f'{name} exited {code}; retained {log}')
            print('COMPLETED ' + name, flush=True)
            del active[gpu]
        if active:
            time.sleep(5)


def r1():
    root = OUT / 'R1'
    root.mkdir(parents=True, exist_ok=False)
    scope = dict(plan=PLAN, revision='r2', status='AUTHORIZED_R1',
                 authorization='用户：请你执行本次计划，不需要我授权，执行完毕',
                 units={}, source_metadata=read(DOWN / 'direct_source.json'),
                 budget=dict(starts=3, epochs=210, updates=2310),
                 smoke=False, hashes=False, final_test='one final stage')
    manifest = read(SPLITS / 'nc.json')
    for fold in (2, 3, 4):
        entry = manifest['folds'][fold]
        selected, train, val = compact_train_validation_indices(
            entry['train_indices'], entry['validation_indices'])
        source, frame = open_source(DOWN, DOWN, task='nc', dual_static_root=DOWN,
            selected_indices=selected, split_manifest=manifest,
            record_index_path=DOWN / 'nc_offsets.npy', expected_task_rows=manifest['sample_count'])
        try:
            labels = frame['label'].to_numpy(dtype=float)
            r = recipe('development', 'nc', 0, 'SP')
            r.update(plan=PLAN, revision='r2', fold=fold, seed=42 + fold,
                     head_initial=str(reference('nc', fold) / 'head_initial.pt'),
                     K=0, ensemble='disabled; single best only')
            scope['units'][key('development', 'nc', fold, 'SP')] = dict(
                recipe=r, reserved_epochs=70, reserved_updates=70 * math.ceil(len(train) / 24),
                indices=dict(train=entry['train_indices'], validation=entry['validation_indices'], selected=selected),
                validation_keys=[source.samples[i][0].hex() for i in val],
                scaler_mean=[float(labels[train].mean())], scaler_scale=[float(labels[train].std())])
        finally:
            source.close()
    json_write(root / 'scope.json', scope)
    jobs([(f'R1_nc_f{fold}', ['python', 'scripts/finetune_sota_ft.py',
           '--stage', 'development', '--task', 'nc', '--fold', str(fold), '--arm', 'SP',
           '--scope', str(root / 'scope.json')]) for fold in (2, 3, 4)])
    gains = []
    folds = []
    for fold in (2, 3, 4):
        candidate = read(root / key('development', 'nc', fold, 'SP') / 'metrics.json')['results']['single']['rmse']
        baseline = read(reference('nc', fold) / 'metrics.json')['best_validation_rmse']
        gain = (baseline - candidate) / baseline
        gains.append(gain)
        folds.append(dict(fold=fold, candidate_rmse=candidate, reference_rmse=baseline, relative_gain=gain))
    passed = float(np.mean(gains)) >= .01 and sum(g > 0 for g in gains) >= 2 and min(gains) >= -.02
    json_write(root / 'selection.json', dict(plan=PLAN, stage='R1', folds=folds,
               mean_relative_gain=float(np.mean(gains)), positive_folds=sum(g > 0 for g in gains),
               minimum_relative_gain=min(gains), passed=passed,
               decision='Nc SP-single eligible for final review' if passed else 'retain GLT',
               scientific_status='development validation; shared folds, not blind test'))
    print(json.dumps(dict(stage='R1', passed=passed, gains=gains)), flush=True)


def r2():
    root = OUT / 'R2'
    root.mkdir(parents=True, exist_ok=False)
    original = read(ROOT / 'results/mcl_ph_r4_20261002/fast_campaign_04/m_xattn_fast.json')
    scope = dict(plan=PLAN, revision=REVISION, status='AUTHORIZED', units={},
                 authorization='用户：请执行下一步计划',
                 initial_state=str(INITIAL_STATE) if INITIAL_STATE else None,
                 stop_on_router_low_util=REVISION != 'r3',
                 pretraining_budget=dict(starts=2, updates=10000),
                 screening_budget=dict(starts=8, epochs=560, updates=5880),
                 confirmation_budget_per_task=dict(starts=6, epochs=420, eea_updates=4200, eps_updates=4620),
                 smoke=False, hashes=False)
    configs = {}
    for arm, objective in (('M0', 'expert'), ('Mmix', 'expert_mixed_50')):
        config = dict(original, local_objective=objective, router_hard_min=.01,
                      save_initial_state=True, stop_on_router_low_util=REVISION != 'r3',
                      optimization_plan=PLAN, optimization_revision=REVISION)
        configs[arm] = root / (arm + '.json')
        json_write(configs[arm], config)
    json_write(root / 'scope.json', scope)
    # Both arms require four ranks; run sequentially and retain the same recipe.
    for arm in ('M0', 'Mmix'):
        command = ['torchrun', '--standalone', '--nproc_per_node=4', 'scripts/pretrain_mcl_ph.py',
                   '--config', str(configs[arm]), '--cohort-root', str(PI), '--cache-root', str(PI),
                   '--dual-static-root', str(PI), '--statistics', str(STATISTICS),
                   '--shared-new-init', str(SHARED), '--output', str(root / 'pretrain' / arm),
                   '--prep-workers', '16', '--lmdb-readahead']
        if INITIAL_STATE:
            command += ['--initial-state', str(INITIAL_STATE)]
        log = launch('R2_pretrain_' + arm, command)
        while not (log / 'exit.code').exists():
            time.sleep(5)
        code = int((log / 'exit.code').read_text())
        if code:
            raise RuntimeError(f'R2 {arm} pretraining exited {code}; no retry')
        print('COMPLETED R2_pretrain_' + arm, flush=True)

    def downstream(stage, tasks, folds):
        units = []
        for arm in ('M0', 'Mmix'):
            for task in tasks:
                for fold in folds:
                    output = root / stage / arm
                    package = root / 'pretrain' / arm / 'deploy_05000.pt'
                    folder = output / 'm_xattn' / task / f'fold{fold}'
                    scope['units'][str(folder.resolve())] = dict(task=task, fold=fold,
                        checkpoint=str(package.resolve()), config=str(configs[arm].resolve()), epochs=70,
                        source=str(DOWN.resolve()), split=str(SPLITS.resolve()))
                    units.append((f'R2_{stage}_{arm}_{task}_f{fold}', [
                        'python', 'scripts/finetune_mcl_ph.py', '--arm', 'm_xattn',
                        '--stage', 'development', '--config', str(configs[arm]),
                        '--checkpoint', str(package), '--expected-pretrain-step', '5000',
                        '--cohort-root', str(DOWN), '--cache-root', str(DOWN), '--dual-static-root', str(DOWN),
                        '--split-root', str(SPLITS), '--cohort-index', str(DOWN / (task + '_offsets.npy')),
                        '--statistics', str(STATISTICS), '--task', task, '--fold', str(fold),
                        '--epochs', '70', '--output', str(output), '--target-transform', 'standard',
                        '--experiment-plan', PLAN, '--optimization-scope', str(root / 'scope.json')]))
        json_write(root / 'scope.json', scope)
        jobs(units)

    def gate(task, stage, folds, confirmation=False):
        rows = []
        strong_root = (ROOT / 'results/mcl_ph_r4_20261002/fast_campaign_04/paper5_outer/m_xattn'
                       if task == 'eea' else ROOT / 'results/mcl_ph_r4_20261002/fast_campaign_04/paper5_outer/glt_ref')
        for fold in folds:
            values = {arm: read(root / stage / arm / 'm_xattn' / task / f'fold{fold}' / 'metrics.json')['best_validation_rmse']
                      for arm in ('M0', 'Mmix')}
            strong = read(strong_root / task / f'fold{fold}' / 'metrics.json')['best_validation_rmse']
            rows.append(dict(fold=fold, **values, strong=strong,
                        gain_M0=(values['M0'] - values['Mmix']) / values['M0'],
                        gain_strong=(strong - values['Mmix']) / strong))
        gains = {ref: [row['gain_' + ref] for row in rows] for ref in ('M0', 'strong')}
        passed = all(np.mean(values) >= (.01 if confirmation else .02)
                     and sum(v > 0 for v in values) >= 2
                     and (not confirmation or min(values) >= -.02) for values in gains.values())
        return dict(task=task, rows=rows, gains=gains,
                    mean_gains={k:float(np.mean(v)) for k,v in gains.items()}, passed=bool(passed))

    downstream('screening', ('eea', 'eps'), (0, 1))
    screening = {task: gate(task, 'screening', (0, 1)) for task in ('eea', 'eps')}
    json_write(root / 'screening_selection.json', screening)
    eligible = [task for task, result in screening.items() if result['passed']]
    confirmation = {}
    if eligible:
        downstream('confirmation', eligible, (2, 3, 4))
        confirmation = {task:gate(task, 'confirmation', (2, 3, 4), True) for task in eligible}
    json_write(root / 'selection.json', dict(plan=PLAN, screening=screening,
        confirmation=confirmation, eligible_for_final_review=[task for task,v in confirmation.items() if v['passed']],
        scientific_status='development validation; shared folds, not blind test'))
    print(json.dumps(dict(stage='R2', eligible_for_final_review=[task for task,v in confirmation.items() if v['passed']])), flush=True)


def main():
    global OUT, LOG, REVISION, INITIAL_STATE, WINDOW_PREFIX
    require_tmux()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('R1', 'R2'), required=True)
    parser.add_argument('--run-root', type=Path)
    parser.add_argument('--log-root', type=Path)
    parser.add_argument('--revision', choices=('r2', 'r3'), default='r2')
    parser.add_argument('--initial-state', type=Path)
    args = parser.parse_args()
    if args.revision == 'r3' and (args.stage != 'R2' or not all((args.run_root, args.log_root, args.initial_state))):
        parser.error('r3 requires R2 and explicit run/log roots and complete initial state')
    OUT, LOG, INITIAL_STATE = execution_paths(args.revision, args.run_root, args.log_root, args.initial_state)
    REVISION = args.revision
    WINDOW_PREFIX = 'mcl_opt_r3_' if REVISION == 'r3' else WINDOW_PREFIX
    status_path = OUT / 'execution_status.json'
    OUT.mkdir(parents=True, exist_ok=True)
    if status_path.exists():
        raise FileExistsError('execution status already exists; no automatic reentry')
    json_write(status_path, dict(plan=PLAN, revision=REVISION, status='RUNNING', full_plan_completed=False))
    try:
        (r1 if args.stage == 'R1' else r2)()
        json_write(status_path, dict(plan=PLAN, revision=REVISION, status='TRAINING_FINISHED', full_plan_completed=False))
    except BaseException as error:
        json_write(status_path, dict(plan=PLAN, revision=REVISION, status='STOPPED_ACTUAL_ERROR',
                                    error=f'{type(error).__name__}: {error}', full_plan_completed=False))
        raise


if __name__ == '__main__':
    main()
