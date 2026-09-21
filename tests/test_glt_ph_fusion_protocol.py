"""Protocol contract: launcher discipline, failure retention and unit gating.

These tests are pure fixtures and fake processes: they never start a model, a
GPU job or a training run.
"""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from scripts.aggregate_glt_ph_fusion import GAIN_THRESHOLD, load_unit
from scripts.finetune_glt_ph_fusion import ALL_ARMS, EPOCH_CAPS, arm_family
from scripts.pretrain_glt_ph_fusion import (ARM_OBJECTIVE, FAMILY_ARMS, arm_family as pretrain_family,
                                            decay_split)
from src.training.glt_dual_runtime import write_json

REPO = Path(__file__).resolve().parents[1]
LAUNCHER = REPO / 'scripts' / 'run_glt_ph_fusion.sh'


def run_launcher(tmp_path, steps):
    steps_file = tmp_path / 'steps.tsv'
    steps_file.write_text('\n'.join('\t'.join(item) for item in steps) + '\n')
    log_dir = tmp_path / 'logs'
    environment = dict(os.environ, STEPS_FILE=str(steps_file), LOG_DIR=str(log_dir))
    completed = subprocess.run(['bash', str(LAUNCHER)], env=environment,
                               capture_output=True, text=True)
    status = (log_dir / 'chain_status.log').read_text()
    return completed, status


def test_launcher_keeps_the_real_exit_code_and_stops_on_failure(tmp_path):
    completed, status = run_launcher(tmp_path, [
        ('first', 'exit 0'),
        ('second', 'exit 7'),
        ('third', 'echo should-not-run'),
    ])
    assert completed.returncode == 7
    assert 'EXIT tag=first code=0' in status
    assert 'EXIT tag=second code=7' in status
    assert 'STOPPED_AFTER_FAILURE' in status
    assert 'ALL_DONE' not in status
    assert not (tmp_path / 'logs' / 'third.log').exists()


def test_launcher_writes_all_done_only_on_full_success(tmp_path):
    completed, status = run_launcher(tmp_path, [
        ('one', 'echo ok'),
        ('two', 'true'),
    ])
    assert completed.returncode == 0
    assert status.strip().splitlines()[-1].startswith('ALL_DONE')
    assert 'STOPPED_AFTER_FAILURE' not in status


def test_launcher_records_the_command_and_log_path(tmp_path):
    completed, status = run_launcher(tmp_path, [('only', 'echo hello')])
    assert completed.returncode == 0
    assert 'command=echo hello' in status
    assert 'log=' in status
    assert 'hello' in (tmp_path / 'logs' / 'only.log').read_text()


def test_pretrain_failure_report_keeps_the_evidence(tmp_path):
    from scripts.pretrain_glt_ph_fusion import write_failure

    output = tmp_path / 'run'
    output.mkdir()
    (output / 'records_rank0.jsonl').write_text('{"step": 1}\n')
    (output / 'resume_00002.pt').write_bytes(b'checkpoint')
    write_failure({'output': output, 'arm': 'PH', 'stage': 'train:step2'}, ValueError('boom'))
    runtime = json.loads((output / 'runtime.json').read_text())
    assert runtime['status'] == 'FAIL'
    assert runtime['stage'] == 'train:step2'
    assert 'boom' in runtime['error']
    assert runtime['checkpoint_kept'] is True and runtime['metrics_written'] is True
    assert (output / 'resume_00002.pt').exists()


def test_finetune_failure_report_marks_the_stage(tmp_path):
    from scripts.finetune_glt_ph_fusion import write_failure

    output = tmp_path / 'finetune'
    (output / 'xc' / 'fold0').mkdir(parents=True)
    (output / 'xc' / 'fold0' / 'best.pt').write_bytes(b'checkpoint')
    write_failure({'output': output, 'arm': 'CONST', 'stage': 'diagnostics:xc:fold0'},
                  RuntimeError('device'))
    runtime = json.loads((output / 'runtime.json').read_text())
    assert runtime['status'] == 'FAIL' and runtime['stage'] == 'diagnostics:xc:fold0'
    assert runtime['checkpoint_kept'] is True
    assert 'best.pt' in runtime['checkpoints_written']


def test_aggregator_refuses_incomplete_or_leaky_units(tmp_path):
    folder = tmp_path / 'PH' / 'xc' / 'fold0'
    folder.mkdir(parents=True)
    unit = dict(plan='GLT-PH-END2END-20260920-01', arm='PH', task='xc', fold=0,
                readout='DUAL', best_validation_r2=0.3, best_epoch=1, epochs_run=1,
                epochs_configured=1, validation_only=True, outer_test='NOT_RUN',
                complete=True, stage_completed='diagnostics',
                checkpoint={'arm': 'PH'}, mode='smoke')
    path = folder / 'metrics.json'
    write_json(path, unit)
    assert load_unit(path)['arm'] == 'PH'
    for broken, reason in ((dict(unit, complete=False), 'complete'),
                           (dict(unit, outer_test='RUN'), 'validation-only'),
                           (dict(unit, validation_only=False), 'validation-only'),
                           (dict(unit, epochs_run=5), 'epochs'),
                           (dict(unit, checkpoint=None), 'checkpoint')):
        write_json(path, broken)
        with pytest.raises(ValueError, match=reason):
            load_unit(path)
    # the project writer refuses NaN outright; a corrupted artifact must still
    # be rejected instead of entering a macro average
    path.write_text(json.dumps(dict(unit, best_validation_r2=float('nan'))))
    with pytest.raises(ValueError, match='finite'):
        load_unit(path)


def test_arm_and_objective_tables_are_closed():
    assert set(ALL_ARMS) == {'R0', 'R2D', 'CURRENT', 'CONST', 'STAT', 'PH'}
    assert set(ARM_OBJECTIVE) == {'R0', 'R2D', 'CONST', 'STAT', 'PH'}
    assert FAMILY_ARMS['reference'] == ('R0', 'R2D')
    assert FAMILY_ARMS['B'] == ('CONST', 'STAT', 'PH')
    for arm in ('CONST', 'STAT', 'PH'):
        assert pretrain_family(arm) == 'B'
        assert arm_family(arm) == 'B'
    assert arm_family('CURRENT') == 'CURRENT'
    assert EPOCH_CAPS['smoke'] == 1 and EPOCH_CAPS['development'] == 30
    assert GAIN_THRESHOLD == 0.005


def test_decay_split_keeps_the_declared_weight_decay_contract():
    import torch
    from torch import nn

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.matrix = nn.Linear(4, 4)
            self.conditional = nn.Linear(4, 4)
            self.spatial = nn.ModuleList([nn.Linear(4, 4)])
            self.norm = nn.LayerNorm(4)

    decay, no_decay = decay_split(Model())
    decay_names = {name for name, _ in decay}
    no_decay_names = {name for name, _ in no_decay}
    assert decay_names == {'matrix.weight'}
    assert {'matrix.bias', 'conditional.weight', 'conditional.bias',
            'spatial.0.weight', 'norm.weight', 'norm.bias'} <= no_decay_names


def test_scripts_expose_the_declared_cli_surface():
    for script, required in (('pretrain_glt_ph_fusion.py', ('--arm', '--mode', '--updates')),
                             ('finetune_glt_ph_fusion.py', ('--arm', '--mode', '--checkpoint',
                                                            '--identity')),
                             ('aggregate_glt_ph_fusion.py', ('--root', '--output'))):
        completed = subprocess.run([sys.executable, str(REPO / 'scripts' / script), '--help'],
                                   capture_output=True, text=True)
        assert completed.returncode == 0, script
        for flag in required:
            assert flag in completed.stdout, (script, flag)
