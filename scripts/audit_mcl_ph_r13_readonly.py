#!/usr/bin/env python3
"""Read-only acceptance of the fixed historical r13 five-task campaign."""

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from scripts.aggregate_mcl_ph import check_unit
from src.dataset.cache_lifecycle import json_hash
from src.training.glt_dual_runtime import sha256_file

ROOT = Path(__file__).resolve().parents[1]
TASKS = ('eea', 'egb', 'ei', 'eps', 'nc')
ARMS = ('glt_ref', 'o8_only', 'm_cat', 'm_gate', 'm_xattn')
FOLDS = range(5)
RESULTS = ROOT / 'results/mcl_ph_20260921/p2/paper5_official_r13_1'
LOGS = ROOT / 'logs/mcl_ph_20260921/paper5_official_r13_1'
OFFICIAL = ROOT / 'configs/mts/periodic_tdl_official5'
COHORT = ROOT / 'data/processed/glt_dual_v2/downstream/cohort_1545eda5a8f6a1'
UNION = ROOT / 'data/processed/glt_dual_v2/downstream/union_paper5_outer'
PRETRAIN = ROOT / 'results/mcl_ph_20260921/p2/pretrain'
PACKAGES = {'glt_ref': 'glt_ref_r10r1', 'o8_only': 'glt_ref_r10r1',
            'm_cat': 'cat_r10r3', 'm_gate': 'gate_r10r3',
            'm_xattn': 'xattn_r10r3'}


def audit():
    problems = []
    cohort = json.loads((COHORT / 'manifest.json').read_text(encoding='utf-8'))
    union = json.loads((UNION / 'manifest.json').read_text(encoding='utf-8'))
    frozen = json.loads((COHORT / '.frozen').read_text(encoding='utf-8'))
    if frozen != {'manifest_hash': json_hash(cohort)}:
        raise ValueError('frozen cohort manifest seal mismatch')
    if sha256_file(COHORT / 'keys.npy') != cohort['keys_file_sha256']:
        raise ValueError('frozen cohort key table SHA mismatch')
    keys = np.load(COHORT / 'keys.npy', allow_pickle=False)
    if keys.shape != (cohort['sample_count'], 32):
        raise ValueError('frozen cohort key table shape differs from manifest')
    package_sha = {arm: sha256_file(PRETRAIN / folder / 'deploy_05000.pt')
                   for arm, folder in PACKAGES.items()}
    if (ROOT / 'logs/mcl_ph_20260921/paper5_official_r13_1_launcher.exit').read_text().strip() != '0':
        problems.append('historical launcher exit is not zero')
    checked = 0
    for task in TASKS:
        manifest_path = OFFICIAL / f'{task}.json'
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        if (manifest.get('protocol') != 'periodic_tdl_official5'
                or manifest.get('task') != task
                or manifest.get('sample_count') != cohort['task_counts'][task]):
            problems.append(f'{task}: historical manifest identity mismatch')
            continue
        old_sha = cohort['split_manifest_sha256'][task]
        # The current union manifest was replaced after r13. Historical audit
        # uses the frozen cohort's own binding and saved per-row key table.
        if union['task_counts'] != cohort['task_counts']:
            problems.append(f'{task}: current task counts differ from frozen cohort')
        order = union['task_order']
        offset = sum(cohort['task_counts'][name] for name in order[:order.index(task)])
        expected_keys = [bytes(row).hex() for row in keys[offset:offset + manifest['sample_count']]]
        for arm in ARMS:
            for fold in FOLDS:
                exit_file = LOGS / f'{arm}_{task}_fold{fold}.exit'
                if not exit_file.is_file() or exit_file.read_text().strip() != '0':
                    problems.append(f'{arm}/{task}/fold{fold}: unit exit is not zero')
                _, unit_problems = check_unit(
                    RESULTS, arm, task, fold, stage='paper5_outer', expected_step=5000,
                    historical_manifest=manifest_path,
                    expected_package_sha256=package_sha[arm],
                    expected_sample_keys=expected_keys)
                metrics = RESULTS / arm / task / f'fold{fold}' / 'metrics.json'
                if metrics.is_file() and json.loads(metrics.read_text()).get('cohort_split_sha256') != old_sha:
                    unit_problems.append('frozen cohort split SHA differs from unit')
                problems.extend(f'{arm}/{task}/fold{fold}: {item}' for item in unit_problems)
                checked += 1
    return {'contract': 'historical_r13_official5_readonly',
            'status': 'PASS' if not problems and checked == 125 else 'INCOMPLETE',
            'units_expected': 125, 'units_checked': checked,
            'tasks': TASKS, 'arms': ARMS, 'folds': list(FOLDS),
            'package_sha256': package_sha, 'problems': problems}


if __name__ == '__main__':
    report = audit()
    print(json.dumps(report, indent=2, allow_nan=False))
    if report['status'] != 'PASS':
        raise SystemExit(4)
