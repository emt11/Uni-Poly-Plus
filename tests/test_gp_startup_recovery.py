"""CPU orchestration regressions; fixtures never launch trainers or data workers."""
import copy
import json
import shlex
import sys
from pathlib import Path

import pytest
import torch

from scripts import finish_gp_sota as finish
from scripts import run_gp_development_stage as development


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def startup(tmp_path, monkeypatch):
    out, log = tmp_path / 'results', tmp_path / 'logs'
    monkeypatch.setattr(finish, 'OUT', out)
    monkeypatch.setattr(finish, 'LOG', log)
    monkeypatch.setattr(finish, 'require_tmux', lambda: None)
    indices = list(range(1024))
    monkeypatch.setattr(finish, 'load_sample_index_artifact',
                        lambda *args: {'indices': torch.tensor(indices)})
    folder = finish.package('GP').parent
    folder.mkdir(parents=True)
    acceptance = dict(status='PASS', route='GP', step=5000, evaluated_rows=1024,
                      evaluation_calls_per_package=1, optimizer_updates=0,
                      complete_encoder_exact=True, step0_205_exact=True,
                      resume_finite=True, validation_total_loss_candidate_selection=False,
                      source_indices=indices)
    write_json(folder / 'acceptance.json', acceptance)
    write_json(folder / 'runtime.json', dict(status='PASS', step=5000, executed_updates=5000,
                                             start_step=0, world=4, recipe=finish.pretrain_recipe('GP')))
    write_json(folder / 'budget.json', dict(attempted_updates=5000, completed_updates=5000,
                                            total_step=5000, smoke=False))
    (folder / 'validation.writer').touch()
    weights = {'weight': torch.tensor([1., 2.])}
    saved = dict(schema='gp-sota-resume-v2', step=5000, next_position=5040000,
                 rng=[{} for _ in range(4)], model={'encoder.weight': weights['weight']},
                 contract=dict(recipe=finish.pretrain_recipe('GP'), smoke=False))
    deploy = dict(schema='gp-sota-deploy-v2', route='GP', step=5000, state_dict=weights)
    torch.save(saved, folder / 'resume_05000.pt')
    torch.save(deploy, finish.package('GP'))
    scope = out / 'scopes/P3a.json'
    write_json(scope, finish.development_scope('P3a', finish.units('GP', finish.PRIORITY, (0, 1))))
    commands = {
        'p2a_GP_01': ['torchrun', '--standalone', '--nproc_per_node=4',
                      'scripts/pretrain_glt_mcl_ph.py', '--route', 'GP',
                      '--output', str(folder), '--stop', '5000'],
        'p2a_GP_acceptance_01': ['python', 'scripts/check_gp_pretrain_package.py',
                                 '--folder', str(folder)],
        'P3a_01': ['python', 'scripts/run_gp_development_stage.py', '--scope', str(scope)],
    }
    for name, command in commands.items():
        p = log / name
        p.mkdir(parents=True)
        (p / 'command.txt').write_text(shlex.join(command))
        (p / 'exit.code').write_text('1' if name == 'P3a_01' else '0')
    (log / 'P3a_01/stderr.log').write_text(f'FileNotFoundError: {out / "development/P3a"}')
    return out, log, folder


def test_recovery_preflight_is_read_only_and_launches_nothing(startup, monkeypatch, capsys):
    out, log, _ = startup
    before = {p: p.read_bytes() for root in (out, log) for p in root.rglob('*') if p.is_file()}

    def forbidden(*args, **kwargs):
        pytest.fail('read-only preflight attempted to launch a stage or trainer')

    for name in ('run', 'pretrain', 'stage', 'start_window'):
        monkeypatch.setattr(finish, name, forbidden)
    monkeypatch.setattr(sys, 'argv', ['finish_gp_sota.py', '--check-p3a-startup'])
    finish.main()
    report = json.loads(capsys.readouterr().out)
    assert report['status'] == 'PASS_P3A_STARTUP_RECOVERY'
    assert (report['model_calls'], report['optimizer_updates'], report['new_validation_rows']) == (0, 0, 0)
    after = {p: p.read_bytes() for root in (out, log) for p in root.rglob('*') if p.is_file()}
    assert after == before


@pytest.mark.parametrize('damage', ['scope', 'validation_indices', 'nonfinite',
                                   'export', 'child_log', 'training_output', 'ft_ledger'])
def test_recovery_rejects_changed_contract_or_prior_training(startup, damage):
    out, log, folder = startup
    if damage == 'scope':
        path = out / 'scopes/P3a.json'
        scope = finish.read(path)
        scope['units'][0]['fold'] = 2
        write_json(path, scope)
    elif damage == 'validation_indices':
        path = folder / 'acceptance.json'
        acceptance = finish.read(path)
        acceptance['source_indices'][0] = 9000
        write_json(path, acceptance)
    elif damage in ('nonfinite', 'export'):
        path = finish.package('GP')
        deploy = torch.load(path, weights_only=False)
        deploy['state_dict']['weight'][0] = float('nan') if damage == 'nonfinite' else 9.
        torch.save(deploy, path)
    elif damage == 'child_log':
        (log / 'P3a_GP_ei_f0_01').mkdir()
    elif damage == 'training_output':
        (out / 'development/P3a').mkdir(parents=True)
    else:
        write_json(out / 'other/budget.json', dict(kind='finetune', stage='development', attempted_updates=0))
    with pytest.raises((AssertionError, ValueError)):
        finish.check_p3a_startup_recovery()


def test_startup_retry_uses_new_log_and_preserves_scope(startup, monkeypatch):
    out, log, _ = startup
    scope_path = out / 'scopes/P3a.json'
    original = scope_path.read_bytes()
    failure = (log / 'P3a_01/stderr.log').read_bytes()
    calls = []

    def fake_run(name, command, *, attempt):
        calls.append((name, command, attempt))
        write_json(out / 'development/P3a/report.json', dict(rows=[{'status': 'synthetic'}]))

    monkeypatch.setattr(finish, 'run', fake_run)
    finish.stage('P3a', finish.units('GP', finish.PRIORITY, (0, 1)), resume_startup=True)
    assert calls == [('P3a', ['python', 'scripts/run_gp_development_stage.py',
                               '--scope', str(scope_path)], 2)]
    assert scope_path.read_bytes() == original
    assert (log / 'P3a_01/stderr.log').read_bytes() == failure
    with pytest.raises(AssertionError, match='cannot reuse training outputs'):
        finish.stage('P3a', finish.units('GP', finish.PRIORITY, (0, 1)), resume_startup=True)
    assert len(calls) == 1


def test_resume_branch_skips_gp_training_and_acceptance(startup, monkeypatch):
    finish.check_p3a_startup_recovery()
    expected = finish.units('GP', finish.PRIORITY, (0, 1))

    class ReachedP3a(Exception):
        pass

    def fake_stage(name, units, *, resume_startup):
        assert name == 'P3a' and units == expected and resume_startup
        raise ReachedP3a

    def forbidden(*args, **kwargs):
        pytest.fail('GP training/acceptance rerun before P3a')

    monkeypatch.setattr(finish, 'stage', fake_stage)
    monkeypatch.setattr(finish, 'run', forbidden)
    monkeypatch.setattr(finish, 'pretrain', forbidden)
    monkeypatch.setattr(sys, 'argv', ['finish_gp_sota.py', '--from-gp', '--resume-p3a-startup'])
    with pytest.raises(ReachedP3a):
        finish.main()


def test_development_creates_parent_and_refuses_existing_outputs(tmp_path, monkeypatch):
    out, log = tmp_path / 'results', tmp_path / 'logs'
    monkeypatch.setattr(development, 'OUT', out)
    monkeypatch.setattr(development, 'LOG', log)
    monkeypatch.setattr(development, 'require_tmux', lambda: None)
    monkeypatch.setattr(development.time, 'sleep', lambda _: None)
    checkpoint = out / 'pretrain/GP/deploy_05000.pt'
    write_json(checkpoint.parent / 'acceptance.json', dict(status='PASS'))
    unit = dict(route='GP', task='ei', fold=0, recipe='F0')
    scope = out / 'scopes/P3a.json'
    write_json(scope, dict(name='P3a', units=[unit], allowed_units=['F0/GP/ei/fold0/seed42'],
                           packages={'GP': str(checkpoint)}))
    assert not (out / 'development').exists()
    calls = []

    def fake_write_shell(path, command, *, gpu):
        path.mkdir(parents=True, exist_ok=False)
        calls.append((command, gpu))

    def fake_start_window(name, path):
        (path / 'exit.code').write_text('0')

    monkeypatch.setattr(development, 'write_shell', fake_write_shell)
    monkeypatch.setattr(development, 'start_window', fake_start_window)
    monkeypatch.setattr(development, 'audit', lambda folder, unit: dict(unit=copy.deepcopy(unit)))
    monkeypatch.setattr(sys, 'argv', ['run_gp_development_stage.py', '--scope', str(scope)])
    development.main()
    report = finish.read(out / 'development/P3a/report.json')
    assert report['status'] == 'PASS' and report['rows'] == [{'unit': unit}]
    assert len(calls) == 1 and calls[0][1] == 0
    assert calls[0][0][0:2] == ['python', 'scripts/finetune_glt_mcl_ph.py']
    sentinel = out / 'development/P3a/do_not_overwrite'
    sentinel.write_text('preserve')
    with pytest.raises(FileExistsError):
        development.main()
    assert sentinel.read_text() == 'preserve' and len(calls) == 1
