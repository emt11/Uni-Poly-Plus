#!/usr/bin/env python3
"""Verify one recovery-smoke stage against the r6 checklist, per group.

Checks, per group: the launcher's real exit code, runtime PASS, the unit's
identity and budget, the presence and strict loadability of best.pt, the frozen
PH encoder, and (for the control groups) the input/representation separation.
Writes a JSON verdict and exits non-zero if any check fails.
"""
import argparse
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch

from src.modules.glt_galformer_ph import GLTGalPH
from src.modules.glt_galformer_ph_downstream import GalformerDownstream
from src.modules.glt_galformer_ph_pretrain import load_galformer_deployment
from src.modules.glt_galformer_ph_retention import GalformerPHDownstream
from src.modules.glt_galph_checkpoint_identity import load_identity

GROUPS = ('F_OFF', 'F_CONST', 'F_REAL')


def check_group(group, root, status_log, identity, expected_epochs, task, fold):
    folder = Path(root) / group / task / f'fold{fold}'
    checks, notes = {}, {}
    metrics_path, runtime_path = folder / 'metrics.json', Path(root) / group / 'runtime.json'
    checks['metrics_present'] = metrics_path.is_file()
    checks['runtime_present'] = runtime_path.is_file()
    if not (checks['metrics_present'] and checks['runtime_present']):
        return {'checks': checks, 'notes': notes}
    metrics = json.loads(metrics_path.read_text(encoding='utf-8'))
    runtime = json.loads(runtime_path.read_text(encoding='utf-8'))
    checks['runtime_pass'] = runtime.get('status') == 'PASS'
    checks['complete'] = metrics.get('complete') is True
    checks['stage_diagnostics'] = metrics.get('stage_completed') == 'diagnostics'
    checks['epochs_run'] = int(metrics.get('epochs_run', -1)) == expected_epochs
    checks['epochs_configured'] = int(metrics.get('epochs_configured', -1)) == expected_epochs
    checks['validation_only'] = metrics.get('validation_only') is True
    checks['outer_test_not_run'] = metrics.get('outer_test') == 'NOT_RUN'
    checks['identity_sha'] = metrics.get('checkpoint_sha256') == identity['sha256']
    recorded = metrics.get('checkpoint_identity') or {}
    checks['identity_record'] = recorded.get('record_path') == identity['record_path']
    checks['encoder_version'] = metrics.get('ph_encoder_version') == \
        identity['ph_encoder_version']
    checks['modes'] = (metrics.get('pretrain_summary_mode'), metrics.get('pretrain_ph_mode')) \
        == (identity['summary_mode'], identity['ph_mode'])
    checks['readout_adaptation'] = (metrics.get('readout'), metrics.get('adaptation')) \
        == ('DUAL', 'full')
    coverage = metrics.get('coverage') or {}
    checks['coverage_complete'] = int(coverage.get('missing', 1)) == 0 and \
        int(coverage.get('samples', 0)) > 0
    r2 = metrics.get('best_validation_r2')
    checks['metric_finite'] = r2 is not None and math.isfinite(float(r2))
    history = metrics.get('validation_r2_history') or []
    checks['history_len'] = len(history) == expected_epochs
    checks['history_finite'] = bool(history) and all(math.isfinite(float(v)) for v in history)
    diagnostics = metrics.get('diagnostics') or {}
    checks['diagnostics_present'] = bool(diagnostics.get('batches', 0)) and \
        math.isfinite(float(diagnostics.get('residual_relative_norm', float('nan'))))
    best_path = folder / 'best.pt'
    checks['best_present'] = best_path.is_file()
    if checks['best_present']:
        import scripts.finetune_glt_galformer_ph_retention as runner
        saved = torch.load(best_path, map_location='cpu', weights_only=False)
        checks['best_identity'] = saved.get('checkpoint_sha256') == identity['sha256'] and \
            saved.get('epochs_run') == expected_epochs
        package = torch.load(identity['checkpoint'], map_location='cpu', weights_only=False)
        encoder = GLTGalPH(str(package['summary_mode']), package.get('ph_mode'))
        load_galformer_deployment(encoder, package, int(identity['step']))
        model = GalformerPHDownstream(
            encoder, readout='DUAL',
            ph_residual=dict(F_OFF='off', F_CONST='const', F_REAL='real')[group])
        missing = model.load_state_dict(saved['state_dict'], strict=True)
        checks['best_strict_load'] = True
        model.freeze_training_only_heads()
        model.train()
        checks['ph_encoder_frozen'] = all(not p.requires_grad
                                          for p in model.encoder.ph_encoder.parameters())
        checks['ph_encoder_eval'] = not model.encoder.ph_encoder.training
        checks['gamma_zero_init_state_present'] = 'gamma' in saved['state_dict']
        notes['fusion_gate_mean'] = diagnostics.get('gate_mean')
        notes['retention_gate_tanh'] = diagnostics.get('retention_gate_tanh')
        notes['residual_relative_norm'] = diagnostics.get('residual_relative_norm')
        notes['residual_norm'] = diagnostics.get('residual_norm')
        notes['ph_valid_fraction'] = diagnostics.get('ph_valid_fraction')
        notes['profile_source'] = diagnostics.get('profile_source')
        notes['profile_cross_sample_spread'] = diagnostics.get('profile_cross_sample_spread')
        notes['profile_vs_const_max_abs'] = diagnostics.get('profile_vs_const_max_abs')
        notes['encoder_summary_own_vs_const_max_abs'] = diagnostics.get(
            'encoder_summary_own_vs_const_max_abs')
        notes['best_validation_r2'] = r2
        notes['validation_r2_history'] = history
        notes['wall_seconds'] = metrics.get('wall_seconds')
    text = Path(status_log).read_text(encoding='utf-8') if Path(status_log).is_file() else ''
    checks['launcher_exit_zero'] = f'EXIT group={group} code=0' in text
    return {'checks': checks, 'notes': notes}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--status-log', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--identity', default='configs/mts/glt_galph_c1_repair_5k_identity.json')
    parser.add_argument('--expected-epochs', type=int, default=1)
    parser.add_argument('--task', default='xc')
    parser.add_argument('--fold', type=int, default=0)
    args = parser.parse_args()
    identity = load_identity(args.identity)
    payload = {'identity_sha256': identity['sha256'], 'expected_epochs': args.expected_epochs,
               'groups': {}, 'failed_checks': []}
    for group in GROUPS:
        result = check_group(group, args.root, args.status_log, identity,
                             args.expected_epochs, args.task, args.fold)
        payload['groups'][group] = result
        failed = sorted(name for name, ok in result['checks'].items() if not ok)
        if failed:
            payload['failed_checks'].append({'group': group, 'checks': failed})
    diagnostics = {group: payload['groups'][group]['notes'] for group in GROUPS}
    payload['separation'] = {
        'f_const_input_sample_independent': diagnostics['F_CONST'].get(
            'profile_cross_sample_spread') == 0.0,
        'f_const_summary_equals_const': diagnostics['F_CONST'].get(
            'encoder_summary_own_vs_const_max_abs') == 0.0,
        'f_real_input_sample_specific': float(diagnostics['F_REAL'].get(
            'profile_cross_sample_spread') or 0.0) > 0.0,
        'f_real_summary_differs_from_const': float(diagnostics['F_REAL'].get(
            'encoder_summary_own_vs_const_max_abs') or 0.0) > 0.0,
        'f_off_residual_exactly_zero': diagnostics['F_OFF'].get('residual_norm') == 0.0 and
        diagnostics['F_OFF'].get('residual_relative_norm') == 0.0,
    }
    for name, ok in payload['separation'].items():
        if not ok:
            payload['failed_checks'].append({'group': 'separation', 'checks': [name]})
    payload['VERDICT'] = 'PASS' if not payload['failed_checks'] else 'FAIL'
    Path(args.output).write_text(json.dumps(payload, indent=2), encoding='utf-8')
    print(json.dumps({'VERDICT': payload['VERDICT'],
                      'failed_checks': payload['failed_checks'],
                      'separation': payload['separation'],
                      'notes': diagnostics}, indent=2, default=str))
    return 0 if payload['VERDICT'] == 'PASS' else 1


if __name__ == '__main__':
    sys.exit(main())
