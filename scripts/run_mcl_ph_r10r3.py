#!/usr/bin/env python3
"""Serial r10R3 execution after CAT, with fail-closed gates and no retries.

Run only in the Uni-Poly tmux session.  CAT is already launched separately; this
driver waits for its exit marker, validates each completed arm, then runs the
remaining authorized arms and development units.  There is no wall-clock limit.
"""
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.aggregate_mcl_ph import check_unit
from scripts.finetune_mcl_ph import (build_head, build_mcl_arm, build_o8_only,
                                     GLTReferenceArm)
from scripts.verify_mcl_ph_arm import verify
from src.modules.glt_dual_pretrain import load_deployment as load_dual_deployment

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'results/mcl_ph_20260921'
PRETRAIN = RESULTS / 'p2/pretrain'
DEVELOPMENT = RESULTS / 'p2/development_r10r3'
LOGS = ROOT / 'logs/mcl_ph_20260921'
STATISTICS = RESULTS / 'p0_r10r3/statistics.npz'
SHARED = PRETRAIN / 'shared_new_init.pt'
COMMON = ROOT / 'results/glt_pred_20260918/s3b_prep/common_init_v1.pt'
TRAJECTORY = ROOT / 'data/processed/mcl_ph_cache/p2_noisy_seed42_sigma003_step5000_v1'
GLT = PRETRAIN / 'glt_ref_r10r1/deploy_05000.pt'
ARM_MODES = {'cat': 'm_cat', 'gate': 'm_gate', 'xattn': 'm_xattn'}
PACKAGE_FOR = {
    'glt_ref': GLT, 'o8_only': GLT,
    **{mode: PRETRAIN / f'{name}_r10r3/deploy_05000.pt'
       for name, mode in ARM_MODES.items()},
}
STATUS = LOGS / 'r10r3_driver_status.json'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def record(stage, **details):
    payload = {'stage': stage, 'time_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ',
                                                        time.gmtime()), **details}
    temporary = STATUS.with_suffix('.tmp')
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
    os.replace(temporary, STATUS)
    print(json.dumps(payload, sort_keys=True), flush=True)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def check_shared_head(head):
    expected = build_head().state_dict()
    actual = head.state_dict()
    require(set(actual) == set(expected) and
            all(torch.equal(actual[name], expected[name]) for name in expected),
            'the 512-wide downstream head does not have the shared initial state')


def run_logged(command, log_path, exit_path):
    require(not log_path.exists() and not exit_path.exists(),
            f'output log or exit marker already exists: {log_path}')
    print('COMMAND', ' '.join(command), 'LOG', log_path, flush=True)
    with log_path.open('w') as handle:
        result = subprocess.run(command, cwd=ROOT, stdout=handle,
                                stderr=subprocess.STDOUT, check=False)
    exit_path.write_text(str(result.returncode) + '\n')
    require(result.returncode == 0, f'command failed with exit {result.returncode}: {log_path}')


def finished_cat():
    marker = LOGS / 'p2_cat_r10r3_pretrain.exit'
    record('WAIT_CAT', marker=str(marker))
    while not marker.is_file():
        time.sleep(30)
    require(marker.read_text().strip() == '0', f'CAT failed: {marker.read_text().strip()}')


def check_pretrain(arm, stats_sha, shared_sha, common_sha):
    folder = PRETRAIN / f'{arm}_r10r3'
    problems, evidence = verify(folder, 5000, True)
    require(not problems, f'{arm} verifier: {problems}')
    require(evidence['status'] == 'PASS', f'{arm} runtime is not PASS')
    for step in (1000, 2000, 3000, 4000, 5000):
        for kind in ('resume', 'deploy'):
            path = folder / f'{kind}_{step:05d}.pt'
            require(path.is_file() and path.stat().st_size > 0, f'missing {path}')
    identity = json.loads((folder / 'run.json').read_text())['identity']
    for key, value in {'world_size': 4, 'statistics_sha256': stats_sha,
                       'statistics_samples': 4096,
                       'shared_new_init_sha256': shared_sha,
                       'common_init_artifact_sha256': common_sha}.items():
        require(identity.get(key) == value, f'{arm} identity mismatch: {key}')
    config = identity['config']
    for key, value in {'fusion_mode': arm, 'microbatch': 84, 'accumulation': 3,
                       'global_batch': 1008, 'amp_dtype': 'bf16',
                       'max_optimizer_steps': 5000, 'save_every': 1000,
                       'router_dense_updates': 500, 'router_top_k_afterwards': 2}.items():
        require(config.get(key) == value, f'{arm} config mismatch: {key}')
    milestones = {}
    for line in (folder / 'steps.jsonl').read_text().splitlines():
        item = json.loads(line)
        step = int(item['step'])
        require(step not in milestones, f'{arm} repeated milestone {step}')
        milestones[step] = item
        for value in [*item['losses'].values(), item['grad_total_preclip'],
                      *item['grad_norms'].values()]:
            require(math.isfinite(float(value)), f'{arm} nonfinite metric at {step}')
    require(milestones[500]['router_mode'] == 'dense' and
            milestones[501]['router_mode'] == 'top2',
            f'{arm} router transition is wrong')
    require(all(step in milestones for step in (1000, 2000, 3000, 4000, 5000)),
            f'{arm} missing checkpoint milestone logs')
    package_path = folder / 'deploy_05000.pt'
    package = torch.load(package_path, map_location='cpu', weights_only=False)
    require(package.get('source', {}).get('statistics_sha256') == stats_sha,
            f'{arm} deployment statistics identity mismatch')
    require(package.get('source', {}).get('shared_new_init_sha256') == shared_sha,
            f'{arm} deployment initialization identity mismatch')
    model = build_mcl_arm(arm, package, expected_step=5000)
    check_shared_head(model.head)
    del model, package
    gc.collect()
    record(f'{arm.upper()}_PASS', package_sha256=sha(package_path),
           milestones=[1000, 2000, 3000, 4000, 5000], router='500:dense,501:top2')


def launch_pretrain(arm):
    folder = PRETRAIN / f'{arm}_r10r3'
    require(not folder.exists(), f'refusing to overwrite {folder}')
    config = ROOT / f'configs/mts/mcl_ph_{arm}.json'
    command = [sys.executable, '-m', 'torch.distributed.run', '--standalone',
               '--nproc_per_node=4', 'scripts/pretrain_mcl_ph.py',
               '--config', str(config), '--cohort-root',
               'data/processed/glt_dual_v2/pi1m/cohort_30f17b59bc5862a1',
               '--cache-root', 'data/processed/mips_trimer_scage',
               '--dual-static-root', 'data/processed/glt_dual_v2/pi1m/dual_static_v1',
               '--statistics', str(STATISTICS), '--shared-new-init', str(SHARED),
               '--trajectory-cache', str(TRAJECTORY), '--diagnostics',
               '--prep-workers', '12', '--stop-after-step', '5000',
               '--output', str(folder)]
    record(f'{arm.upper()}_START', command=command, output=str(folder))
    run_logged(command, LOGS / f'p2_{arm}_r10r3_pretrain.log',
               LOGS / f'p2_{arm}_r10r3_pretrain.exit')


def check_reference():
    folder = PRETRAIN / 'glt_ref_r10r1'
    problems, _ = verify(folder, 5000, False)
    require(not problems, f'GLT_REF verifier: {problems}')
    package = torch.load(GLT, map_location='cpu', weights_only=False)
    require(package.get('step') == 5000 and package.get('fusion_mode') == 'concat',
            'GLT_REF deployment identity mismatch')
    model = GLTReferenceArm(torsion=bool(package.get('torsion_modules', False)))
    load_dual_deployment(model.model, package, 5000)
    copied_model, copied = build_o8_only(
        package, expected_step=5000,
        torsion=bool(package.get('torsion_modules', False)))
    require(len(copied) > 0, 'O8_ONLY copied no parameters')
    check_shared_head(copied_model.head)
    del model, copied_model, package
    gc.collect()
    record('REFERENCE_PASS', glt_package_sha256=sha(GLT), o8_tensors=len(copied))


def check_development_unit(arm, task, fold, expected_sha):
    record_value, problems = check_unit(DEVELOPMENT, arm, task, fold,
                                        stage='development', expected_step=5000)
    require(not problems, f'{arm}/{task}/fold{fold} unit rejected: {problems}')
    require(record_value['pretrain_package_sha256'] == expected_sha,
            f'{arm}/{task}/fold{fold} pretrain SHA mismatch')
    folder = DEVELOPMENT / arm / task / f'fold{fold}'
    best = torch.load(folder / 'best.pt', map_location='cpu', weights_only=False)
    for key, value in {'arm': arm, 'task': task, 'fold': fold, 'stage': 'development',
                       'best_epoch': record_value['best_epoch']}.items():
        require(best.get(key) == value, f'{arm}/{task}/fold{fold} best.pt {key} mismatch')
    require(isinstance(best.get('state_dict'), dict) and best['state_dict'],
            f'{arm}/{task}/fold{fold} best.pt has no model state')
    del best
    gc.collect()
    record('UNIT_PASS', arm=arm, task=task, fold=fold,
           best_epoch=record_value['best_epoch'],
           epochs=record_value['executed_epochs'])


def launch_development():
    require(not DEVELOPMENT.exists(), f'refusing to reuse {DEVELOPMENT}')
    packages = {arm: sha(path) for arm, path in PACKAGE_FOR.items()}
    for arm in ('glt_ref', 'o8_only', 'm_cat', 'm_gate', 'm_xattn'):
        for task in ('xc', 'eps', 'eat'):
            for fold in (0, 1):
                package = PACKAGE_FOR[arm]
                command = [sys.executable, 'scripts/finetune_mcl_ph.py', '--arm', arm,
                           '--stage', 'development', '--config',
                           'configs/mts/mcl_ph_gate.json', '--checkpoint', str(package),
                           '--expected-pretrain-step', '5000', '--cohort-root',
                           'data/processed/glt_dual_v2/downstream/cohort_1545eda5a8f6a1',
                           '--cache-root', 'data/processed/mips_trimer_scage_downstream',
                           '--dual-static-root',
                           'data/processed/glt_dual_v2/downstream/dual_static_v1',
                           '--split-root', 'data/splits/mips_outer5_inner20',
                           '--statistics', str(STATISTICS), '--task', task,
                           '--fold', str(fold), '--epochs', '30',
                           '--output', str(DEVELOPMENT)]
                tag = f'p2_{arm}_{task}_fold{fold}_r10r3'
                record('UNIT_START', arm=arm, task=task, fold=fold,
                       package_sha256=packages[arm], command=command)
                run_logged(command, LOGS / f'{tag}.log', LOGS / f'{tag}.exit')
                check_development_unit(arm, task, fold, packages[arm])
    command = [sys.executable, 'scripts/aggregate_mcl_ph_p2.py', '--root',
               str(DEVELOPMENT), '--expected-pretrain-step', '5000']
    run_logged(command, LOGS / 'p2_aggregate_r10r3.log',
               LOGS / 'p2_aggregate_r10r3.exit')
    result = json.loads((DEVELOPMENT / 'p2_results.json').read_text())
    require(result.get('status') == 'PASS' and result.get('outer_test') == 'NOT_RUN',
            'P2 aggregate did not accept 30 development units')
    record('COMPLETE', units=30, aggregate=str(DEVELOPMENT / 'p2_results.json'),
           selection=result.get('selection'))


def main():
    require(os.environ.get('TMUX'), 'the r10R3 driver must run inside Uni-Poly tmux')
    require(STATISTICS.is_file() and SHARED.is_file(), 'P0 statistics or shared init missing')
    with np.load(STATISTICS, allow_pickle=False) as stats:
        require(int(stats['samples']) == 4096 and
                all(np.isfinite(stats[name]).all() for name in stats.files
                    if stats[name].dtype.kind in 'fi'), 'P0 statistics invalid')
    stats_sha, shared_sha, common_sha = sha(STATISTICS), sha(SHARED), sha(COMMON)
    require(not (PRETRAIN / 'gate_r10r3').exists() and
            not (PRETRAIN / 'xattn_r10r3').exists() and
            not DEVELOPMENT.exists(), 'a later output root already exists')
    finished_cat()
    check_pretrain('cat', stats_sha, shared_sha, common_sha)
    for arm in ('gate', 'xattn'):
        launch_pretrain(arm)
        check_pretrain(arm, stats_sha, shared_sha, common_sha)
    check_reference()
    launch_development()


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        record('STOPPED', error=f'{type(error).__name__}: {error}')
        raise
