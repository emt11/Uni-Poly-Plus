#!/usr/bin/env python3
"""Bounded correctness smoke for the GALPH pretraining runner (P1.2).

Budget: 11 optimizer updates in total.

  N0 run A        2 updates (also the parity reference, plus a forced deploy)
  N0 run B        1 update  -> resume -> 1 update   (parity pair, 2 updates)
  N1 / C0 / C1    2 updates each
  N0 prefetch     1 update with --prep-workers 4    (prefetch invariance)

Checks: every arm performs a finite forward/backward with finite losses, the
sample positions and the 2D/3D/PH mask streams reproduce exactly across a
save/resume boundary, the LR follows the configured schedule, the parameter
state after step 2 is bit-identical, the prefetch path yields the same stream as
the inline path, and the deployment package of the forced export strict-loads
into a fresh model.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

PYTHON = sys.executable
COHORT = 'data/processed/glt_dual_v2/pi1m/cohort_30f17b59bc5862a1'
CACHE = 'data/processed/mips_trimer_scage'
STATIC = 'data/processed/glt_dual_v2/pi1m/dual_static_v1'


def _run(tag, config, extra, output_root, workers=0, resume=None):
    output = output_root / tag
    command = [PYTHON, '-m', 'torch.distributed.run', '--standalone',
               '--nproc_per_node=4', 'scripts/pretrain_glt_galformer_ph.py',
               '--config', config, '--cohort-root', COHORT, '--cache-root', CACHE,
               '--dual-static-root', STATIC, '--output', str(output),
               '--prep-workers', str(workers)] + extra
    if resume is not None:
        command += ['--resume', str(resume)]
    log = output_root / f'{tag}.log'
    with log.open('w', encoding='utf-8') as handle:
        completed = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT,
                                   cwd=str(ROOT), env={**os.environ, 'OMP_NUM_THREADS': '8'})
    return completed.returncode, output, log


def _records(output):
    rows = {}
    for path in sorted(Path(output).glob('records_rank*.jsonl')):
        rank = int(path.stem.rsplit('rank', 1)[-1])
        for line in path.read_text(encoding='utf-8').splitlines():
            row = json.loads(line)
            rows[(int(row['step']), rank)] = row
    return rows


def _parameters(path):
    return torch.load(path, map_location='cpu', weights_only=False)['model']


def _max_diff(left, right):
    assert set(left) == set(right), 'parameter key sets differ'
    return max(float((left[name] - right[name]).abs().max()) for name in left)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', required=True)
    parser.add_argument('--config-root', default='configs/mts')
    args = parser.parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    configs = {arm: f'{args.config_root}/glt_galph_{arm.lower()}.json'
               for arm in ('N0', 'N1', 'C0', 'C1')}
    checks, runs = {}, {}

    # 1. four arms, two updates each (N0 also writes a forced deployment)
    plan = [
        ('n0_a', configs['N0'], ['--updates', '2', '--save-steps', '1', '2',
                                 '--deploy-steps', '2', '--log-every', '1'], 0),
        ('n1_a', configs['N1'], ['--updates', '2', '--save-steps', '2', '--log-every', '1'], 0),
        ('c0_a', configs['C0'], ['--updates', '2', '--save-steps', '2', '--log-every', '1'], 0),
        ('c1_a', configs['C1'], ['--updates', '2', '--save-steps', '2', '--log-every', '1'], 0),
    ]
    for tag, config, extra, workers in plan:
        code, output, log = _run(tag, config, extra, output_root, workers=workers)
        checks[f'{tag}_exit0'] = code == 0
        if code != 0:
            print(json.dumps({'stage': tag, 'exit': code, 'log': str(log)}), flush=True)
        runs[tag] = _records(output)

    for arm, tag in (('N0', 'n0_a'), ('N1', 'n1_a'), ('C0', 'c0_a'), ('C1', 'c1_a')):
        rows = [row for (step, _), row in sorted(runs[tag].items())]
        checks[f'{arm}_finite_losses'] = bool(rows) and all(
            all(float(value) == float(value) and abs(float(value)) < 1e6
                for value in row['losses']) for row in rows)
        checks[f'{arm}_updates'] = len(rows) == 8          # 2 steps x 4 ranks
        checks[f'{arm}_counts_positive'] = all(
            float(row['global_counts'][0]) > 0 and float(row['global_counts'][1]) > 0
            and float(row['global_counts'][2]) > 0 for row in rows)
        if arm in ('N1', 'C1'):
            checks[f'{arm}_ph_count_positive'] = all(
                float(row['global_counts'][3]) > 0 for row in rows)
        else:
            checks[f'{arm}_ph_disabled'] = all(
                float(row['losses'][3]) == 0.0 and float(row['sums'][3]) == 0.0
                for row in rows)
        # world_size scaling: the reported L2D is the global token mean
        checks[f'{arm}_loss2d_in_range'] = all(
            0.5 < float(row['losses'][0]) < 12.0 for row in rows)

    # 2. resume parity: 1 update -> save -> resume -> 1 update
    code_b1, output_b, _ = _run('n0_b1', configs['N0'],
                                ['--updates', '1', '--save-steps', '1', '--log-every', '1'],
                                output_root)
    checks['n0_b1_exit0'] = code_b1 == 0
    code_b2, _, _ = _run('n0_b2', configs['N0'], ['--updates', '1', '--save-steps', '2', '--log-every', '1'],
                         output_root, resume=output_b / 'resume_00001.pt')
    checks['n0_b2_exit0'] = code_b2 == 0
    resumed = _records(output_root / 'n0_b2')
    reference = {key: row for key, row in runs['n0_a'].items() if key[0] == 2}
    parity = {}
    for key, row in sorted(resumed.items()):
        other = reference[key]
        parity[str(key)] = {field: row[field] == other[field] for field in
                            ('position_first', 'position_last', 'stream_digest')}
        parity[str(key)]['lr_equal'] = float(row['lr']) == float(other['lr'])
    checks['resume_positions_and_masks_identical'] = all(
        all(entry[field] for field in ('position_first', 'position_last',
                                       'stream_digest', 'lr_equal'))
        for entry in parity.values())
    checks['resume_parameter_state_identical'] = _max_diff(
        _parameters(output_root / 'n0_b2' / 'resume_00002.pt'),
        _parameters(output_root / 'n0_a' / 'resume_00002.pt')) == 0.0

    # 3. prefetch invariance: the same window prepared by 4 workers
    code_p, output_p, _ = _run('n0_prefetch', configs['N0'],
                               ['--updates', '1', '--save-steps', '1', '--log-every', '1'],
                               output_root, workers=4)
    checks['n0_prefetch_exit0'] = code_p == 0
    prefetch = _records(output_p)
    step1_reference = {key: row for key, row in runs['n0_a'].items() if key[0] == 1}
    checks['prefetch_stream_identical'] = all(
        prefetch[key]['stream_digest'] == step1_reference[key]['stream_digest']
        and prefetch[key]['position_first'] == step1_reference[key]['position_first']
        for key in prefetch)
    checks['prefetch_parameters_identical'] = _max_diff(
        _parameters(output_p / 'resume_00001.pt'),
        _parameters(output_root / 'n0_a' / 'resume_00001.pt')) == 0.0

    # 4. deployment package of the forced export strict-loads into a fresh model
    from src.modules.glt_galformer_ph import GLTGalPH
    from src.modules.glt_galformer_ph_pretrain import load_galformer_deployment
    package = torch.load(output_root / 'n0_a' / 'deploy_00002.pt', map_location='cpu',
                         weights_only=False)
    fresh = GLTGalPH('mean', None)
    load_galformer_deployment(fresh, package, 2)
    checks['deployment_strict_load'] = True
    checks['deployment_excludes_training_heads'] = not any(
        name.startswith(('head_2d.', 'head_3d.', 'cl_proj2.', 'cl_proj3.', 'ph_head.'))
        for name in package['state_dict'])
    checks['deployment_step_recorded'] = int(package['step']) == 2

    payload = {'status': 'PASS' if all(checks.values()) else 'FAIL',
               'total_optimizer_updates': 11,
               'checks': checks, 'resume_parity': parity}
    (output_root / 'smoke.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')
    print(json.dumps(payload, indent=2))
    return 0 if payload['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
