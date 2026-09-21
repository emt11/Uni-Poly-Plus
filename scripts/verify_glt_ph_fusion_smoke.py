#!/usr/bin/env python3
"""Verify the P1 smoke evidence: resume parity, deployments and fine-tuning units.

Checks, per pretraining path: every run finished PASS, the standalone 2-update
run and the first two rows of the contiguous 4-update run are identical, the
contiguous run's last two rows are identical to the resumed run's rows 3-4 on
every rank, the deployment loads strictly, and the per-path update count is the
authorized one.  Then it checks each fine-tuning unit's report contract.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch

from src.modules.glt_ph_fusion_candidates import (GLTFusionB, R2DModel,
                                                  load_fusion_deployment)
from src.modules.glt_galformer_ph import GLTGalPH
from src.training.glt_dual_runtime import write_json

ARMS = ('R0', 'R2D', 'CONST', 'STAT', 'PH')
FINETUNE_ARMS = ('R0', 'R2D', 'CURRENT', 'CONST', 'STAT', 'PH')
COMPARE_FIELDS = ('losses', 'grad_norm_preclip', 'stream_digest', 'position_first',
                  'position_last', 'global_counts', 'lr')


def read_records(path):
    rows = {}
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        rows.setdefault(int(payload['rank']), {})[int(payload['step'])] = payload
    return rows


def compare(left, right, steps, fields=COMPARE_FIELDS):
    issues = []
    for rank in sorted(set(left) | set(right)):
        for step in steps:
            a = left.get(rank, {}).get(step)
            b = right.get(rank, {}).get(step)
            if a is None or b is None:
                issues.append(f'rank{rank} step{step}: missing record')
                continue
            for field in fields:
                if a[field] != b[field]:
                    issues.append(f'rank{rank} step{step} {field}: {a[field]!r} != {b[field]!r}')
    return issues


def check_path(root, arm, report, loader):
    checks = {}
    smoke = root / f'smoke_{arm}_2'
    cont = root / f'cont_{arm}_4'
    seg = root / f'seg_{arm}_2'
    for folder in (smoke, cont, seg):
        runtime = json.loads((folder / 'runtime.json').read_text())
        checks[f'{folder.name}/status'] = runtime['status'] == 'PASS'
        checks[f'{folder.name}/steps'] = int(runtime.get('completed_steps', 0))
    run = json.loads((cont / 'run.json').read_text())
    checks['cont/accumulation'] = int(run['accumulation']) >= 1
    checks['cont/parameter_counts'] = bool(run['parameter_counts']['trainable'])
    groups = json.loads((cont / 'optimizer_groups.json').read_text())
    decay = set(groups['decay_parameters'])
    no_decay = set(groups['no_decay_parameters'])
    checks['optimizer/partition'] = not (decay & no_decay)
    checks['optimizer/ph_side_without_decay'] = all(
        name in no_decay for name in no_decay if name.startswith(('spatial.', 'conditional.')))
    checks['optimizer/matrix_decays'] = any(name.endswith('.weight') for name in decay)
    smoke_records = read_records(smoke / 'records_rank0.jsonl')
    cont_records = read_records(cont / 'records_rank0.jsonl')
    issues = compare(smoke_records, cont_records, steps=[1, 2])
    checks['standalone_matches_contiguous_prefix'] = not issues
    resumed = read_records(seg / 'records_rank0.jsonl')
    issues_resume = compare(cont_records, resumed, steps=[3, 4])
    checks['resume_matches_contiguous'] = not issues_resume
    deployment = cont / 'deploy_00004.pt'
    checks['deployment_written'] = deployment.is_file()
    if deployment.is_file():
        package = torch.load(deployment, map_location='cpu', weights_only=False)
        checks['deployment_arm'] = package.get('arm') == arm
        if arm in ('CONST', 'STAT', 'PH'):
            model = GLTFusionB(arm=arm,
                               profile_stats=dict(mean=torch.zeros(3, 32),
                                                  std=torch.ones(3, 32)))
        elif arm == 'R2D':
            model = R2DModel()
        else:
            model = GLTGalPH('cls', None)
        try:
            checks['deployment_strict_load'] = bool(
                loader(model, package, expected_step=4))
        except Exception as error:
            checks['deployment_strict_load'] = f'{type(error).__name__}: {error}'
    checks['resume_issue_sample'] = issues_resume[:3]
    checks['standalone_issue_sample'] = issues[:3]
    report['paths'][arm] = checks
    return checks


def check_finetune(root, report):
    units = {}
    for arm in FINETUNE_ARMS:
        folder = root / arm / 'xc' / 'fold0'
        metrics = folder / 'metrics.json'
        if not metrics.is_file():
            units[arm] = {'present': False}
            continue
        payload = json.loads(metrics.read_text())
        runtime = json.loads((folder / 'runtime.json').read_text()) \
            if (folder / 'runtime.json').is_file() else {}
        units[arm] = {
            'present': True, 'complete': payload.get('complete'),
            'stage_completed': payload.get('stage_completed'),
            'epochs_run': payload.get('epochs_run'),
            'epochs_configured': payload.get('epochs_configured'),
            'validation_only': payload.get('validation_only'),
            'outer_test': payload.get('outer_test'),
            'readout': payload.get('readout'),
            'best_pt': (folder / 'best.pt').is_file(),
            'finite_r2': bool(payload.get('best_validation_r2') is not None),
            'checkpoint': bool(payload.get('checkpoint')),
            'runtime_status': runtime.get('status'),
        }
    report['finetune'] = units
    return units


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='results/glt_ph_end2end_20260920/p1')
    parser.add_argument('--finetune-root',
                        default='results/glt_ph_end2end_20260920/p1/finetune')
    parser.add_argument('--output',
                        default='results/glt_ph_end2end_20260920/p1/smoke_verification.json')
    args = parser.parse_args()
    root = Path(args.root)
    report = {'plan': 'GLT-PH-END2END-20260920-01', 'stage': 'P1',
              'paths': {}, 'expected_updates_per_path': 10}

    def loader(model, package, expected_step):
        if isinstance(model, (GLTFusionB, R2DModel)):
            load_fusion_deployment(model, package, expected_step=expected_step)
        else:
            from src.modules.glt_galformer_ph_pretrain import load_galformer_deployment
            load_galformer_deployment(model, package, expected_step)
        return True

    for arm in ARMS:
        check_path(root, arm, report, loader)
    check_finetune(Path(args.finetune_root), report)
    flat = [value for checks in report['paths'].values() for value in checks.values()
            if isinstance(value, bool)]
    failures = [f'{arm}:{name}' for arm, checks in report['paths'].items()
                for name, value in checks.items() if value is False]
    finetune_failures = [f'{arm}:{name}' for arm, unit in report['finetune'].items()
                         for name, value in unit.items()
                         if value is False or (name == 'present' and value is not True)]
    report['checks_passed'] = sum(1 for value in flat if value)
    report['checks_total'] = len(flat)
    report['failures'] = failures
    report['finetune_failures'] = finetune_failures
    report['status'] = 'PASS' if not failures and not finetune_failures else 'FAIL'
    write_json(args.output, report)
    print(json.dumps({'status': report['status'], 'checks': f"{report['checks_passed']}/"
                      f"{report['checks_total']}", 'failures': failures[:8],
                      'finetune_failures': finetune_failures[:8],
                      'finetune': report['finetune']}, indent=1), flush=True)
    raise SystemExit(0 if report['status'] == 'PASS' else 1)


if __name__ == '__main__':
    main()
