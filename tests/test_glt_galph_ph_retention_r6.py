"""r6 regression tests: result ordering, the launcher and partial-artifact refusal.

Every test here works on fixtures and fake processes; none of them starts a
training run, loads a dataset or takes an optimizer step.

The behaviour under test is the r6 fix: the best checkpoint and the core metrics
are written before the optional PH diagnostics, a diagnostic failure keeps them
and is recorded as a failure (never a PASS), the aggregator refuses such a unit,
and the shell launcher keeps the real exit code and stops the chain.
"""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import scripts.finetune_glt_galformer_ph_retention as runner

REPO = Path(__file__).resolve().parents[1]
LAUNCHER = REPO / 'scripts' / 'run_glt_galph_retention.sh'
EPOCH1_CONFIG = 'configs/mts/glt_galph_downstream_dual_epoch1.json'
IDENTITY = 'configs/mts/glt_galph_c1_repair_5k_identity.json'


class _Model:
    """Just enough model for ph_diagnostics to run or to be made to fail."""

    class _PHEncoder:
        def __call__(self, profile, mask):
            return profile

        def summarize(self, tokens):
            return tokens.mean(1)

    def __init__(self):
        self.ph_residual = 'real'
        self.encoder = type('Encoder', (), {'ph_encoder': self._PHEncoder()})()
        self.last_ph_stats = {'gate_tanh': 0.0, 'residual_relative_norm': 0.0,
                              'residual_norm': 0.0, 'ph_valid_fraction': 1.0}

    def __call__(self, batch):
        return None, {'gate_mean': 0.5}

    def eval(self):
        return self


class _Batch:
    def __init__(self):
        self.ph_profile = torch.ones((2, 3, 32))
        self.ph_mask = torch.zeros((2, 8), dtype=torch.bool)

    def to(self, device, **kwargs):
        return self


def _unit():
    return {'group': 'F_REAL', 'task': 'xc', 'fold': 0,
            'protocol': 'ph_retention_development', 'outer_test': 'NOT_RUN',
            'validation_only': True, 'best_validation_r2': 0.42, 'best_epoch': 1,
            'epochs_configured': 1, 'epochs_run': 1, 'wall_seconds': 1.0}


def _payload():
    return {'state_dict': {'weight': torch.zeros(2)}, 'architecture': 'stub',
            'group': 'F_REAL', 'task': 'xc', 'fold': 0,
            'protocol': 'ph_retention_development', 'epochs_run': 1,
            'scaler_mean': [0.0], 'scaler_scale': [1.0]}


# ① a diagnostic failure keeps best.pt and the core metrics and never says PASS
def test_diagnostic_failure_keeps_checkpoint_and_core_metrics(tmp_path, monkeypatch):
    folder = tmp_path / 'xc' / 'fold0'
    folder.mkdir(parents=True)
    unit = _unit()

    def explode(*args, **kwargs):
        raise RuntimeError('device mismatch in diagnostics')
    monkeypatch.setattr(runner, 'ph_diagnostics', explode)
    with pytest.raises(RuntimeError, match='device mismatch'):
        runner.complete_unit(folder, unit, _payload(), _Model(), [_Batch()],
                             torch.device('cpu'), torch.zeros((3, 32)))
    assert (folder / 'best.pt').is_file(), 'the checkpoint must survive the failure'
    saved = torch.load(folder / 'best.pt', map_location='cpu', weights_only=False)
    assert saved['state_dict']['weight'].shape == (2,)
    assert saved['epochs_run'] == 1 and saved['group'] == 'F_REAL'
    written = json.loads((folder / 'metrics.json').read_text(encoding='utf-8'))
    assert written['complete'] is False
    assert written['stage_completed'] == 'selection'
    assert written['failure']['stage'] == 'ph_diagnostics'
    assert written['failure']['checkpoint_kept'] is True
    assert written['best_validation_r2'] == 0.42
    assert 'diagnostics' not in written
    # the failure path writes a FAIL runtime for the unit that died
    context = {'group': 'F_REAL', 'protocol': 'ph_retention_development', 'task': 'xc',
               'fold': 0, 'output': str(tmp_path), 'started': 0.0}
    path = runner.write_failure(context, RuntimeError('device mismatch in diagnostics'))
    runtime = json.loads(Path(path).read_text(encoding='utf-8'))
    assert runtime['status'] == 'FAIL'
    assert runtime['failure']['checkpoint_kept'] is True
    assert runtime['failure']['stage'] == 'ph_diagnostics'
    assert runtime['outer_test'] == 'NOT_RUN'


# ② the success path delivers a complete unit and a PASS runtime
def test_success_path_writes_complete_metrics(tmp_path):
    folder = tmp_path / 'xc' / 'fold0'
    folder.mkdir(parents=True)
    unit = runner.complete_unit(folder, _unit(), _payload(), _Model(), [_Batch()],
                                torch.device('cpu'), torch.zeros((3, 32)))
    written = json.loads((folder / 'metrics.json').read_text(encoding='utf-8'))
    assert written['complete'] is True and written['stage_completed'] == 'diagnostics'
    assert 'failure' not in written
    assert written['diagnostics']['batches'] == 1
    assert unit['complete'] is True
    assert (folder / 'best.pt').is_file()
    status, incomplete = runner.runtime_status([unit])
    assert status == 'PASS' and incomplete == []


# ③ the aggregator refuses a partial unit
def test_aggregator_refuses_partial_units(tmp_path):
    import scripts.aggregate_glt_galph_ph_retention as aggregator
    from src.modules.glt_galph_checkpoint_identity import load_identity
    identity = load_identity()
    recorded = {key: identity[key] for key in
                ('record', 'record_path', 'checkpoint', 'sha256', 'step',
                 'summary_mode', 'ph_mode', 'ph_encoder_version')}
    for group in ('F_OFF', 'F_CONST', 'F_REAL'):
        for task in ('xc', 'eps', 'eat'):
            for fold in (0, 1):
                folder = tmp_path / group / task / f'fold{fold}'
                folder.mkdir(parents=True)
                row = {'group': group, 'task': task, 'fold': fold,
                       'protocol': 'ph_retention_development', 'outer_test': 'NOT_RUN',
                       'validation_only': True, 'best_validation_r2': 0.5, 'best_epoch': 1,
                       'epochs_configured': 1, 'epochs_run': 1,
                       'checkpoint_sha256': identity['sha256'],
                       'checkpoint_identity': recorded,
                       'ph_encoder_version': identity['ph_encoder_version'],
                       'pretrain_summary_mode': identity['summary_mode'],
                       'pretrain_ph_mode': identity['ph_mode'],
                       'readout': 'DUAL', 'adaptation': 'full',
                       'coverage': {'samples': 4, 'valid': 4, 'invalid': 0, 'missing': 0},
                       'diagnostics': {'batches': 1, 'retention_gate_tanh': 0.0,
                                       'residual_relative_norm': 0.0,
                                       'ph_valid_fraction': 1.0},
                       'complete': True, 'stage_completed': 'diagnostics',
                       'wall_seconds': 1.0}
                if group == 'F_REAL' and task == 'eat' and fold == 1:
                    row['complete'] = False
                    row['stage_completed'] = 'selection'
                    row['failure'] = {'stage': 'ph_diagnostics', 'error': 'boom',
                                      'checkpoint_kept': True}
                (folder / 'metrics.json').write_text(json.dumps(row), encoding='utf-8')
    with pytest.raises(ValueError, match='not a completed development unit'):
        aggregator.summarize(tmp_path)


# ④ the launcher keeps the real exit code and stops the chain
def _launch(tmp_path, python_command, groups='F_OFF F_CONST F_REAL'):
    status = tmp_path / 'status.log'
    completed = subprocess.run(
        [str(LAUNCHER), '--output-root', str(tmp_path / 'out'),
         '--status-log', str(status), '--config', EPOCH1_CONFIG, '--protocol', 'smoke',
         '--groups', groups],
        cwd=str(REPO), capture_output=True, text=True,
        env=dict(os.environ, PYTHON=python_command))
    return completed, status.read_text(encoding='utf-8') if status.is_file() else ''


def test_launcher_propagates_failure_and_stops(tmp_path):
    completed, status = _launch(tmp_path, '/bin/false')
    assert completed.returncode == 1, 'the process exit code must be the real one'
    assert 'STOPPED_AFTER_FAILURE' in status
    assert 'ALL_DONE' not in status, 'a failed chain must never report ALL_DONE'
    assert status.count('EXIT ') == 1, 'later groups must not run after a failure'
    assert 'group=F_OFF code=1' in status


def test_launcher_runs_every_group_on_success(tmp_path):
    completed, status = _launch(tmp_path, '/bin/true')
    assert completed.returncode == 0
    assert status.count('EXIT ') == 3
    assert 'ALL_DONE' in status and 'STOPPED_AFTER_FAILURE' not in status


def test_launcher_rejects_unknown_protocol(tmp_path):
    completed = subprocess.run(
        [str(LAUNCHER), '--output-root', str(tmp_path / 'out'),
         '--status-log', str(tmp_path / 'status.log'), '--config', EPOCH1_CONFIG,
         '--protocol', 'production'], cwd=str(REPO), capture_output=True, text=True)
    assert completed.returncode == 2
    assert 'must be smoke or development' in completed.stderr
