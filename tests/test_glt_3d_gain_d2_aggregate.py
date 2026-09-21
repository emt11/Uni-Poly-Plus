"""Synthetic-JSON tests for the D2 development aggregator (r6).

No model is built and no data is read here: the units are written as small JSON
records shaped exactly like the ones the runner produces, and the production
validation path is exercised directly.
"""
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.aggregate_glt_3d_gain_d2_development import (  # noqa: E402
    CANDIDATES, check_unit, deltas, evaluate_gate, select_candidate,
)
from scripts.finetune_glt_3d_gain_d2 import ARM_LRS, TASKS, unit_directory  # noqa: E402
from scripts.run_glt_3d_gain_d2_development import unit_list  # noqa: E402

AGGREGATOR = ROOT / 'scripts/aggregate_glt_3d_gain_d2_development.py'
CONFIG = {'seed': 42, 'patience': 10, 'finetune_batch': 32, 'eval_batch': 64,
          'finetune_weight_decay': 0.02, 'encoder_lr': 1e-5, 'fusion_head_lr': 1e-4}
CHECKPOINT = 'results/glt_pred_20260918/s3b_formal/b_fp/pretrain/deploy_05000.pt'
EPOCHS = 3
STEPS = 9


def _payloads(arm, task, fold, *, base_r2=0.40):
    history = [{'epoch': index + 1, 'train_loss': 1.0 - 0.1 * index,
                'validation_loss': 1.0 - 0.1 * index,
                'validation_r2': base_r2 + 0.01 * index, 'training_steps': STEPS,
                'learning_rates': [1e-5, 1e-5, 1e-5, 1e-5]} for index in range(EPOCHS)]
    metrics = {'arm': arm, 'task': task, 'fold': fold, 'stage': 'development',
               'protocol': 'glt_3d_gain_d2_development', 'outer_test': 'NOT_RUN',
               'requested_epochs': 30, 'executed_epochs': EPOCHS,
               'optimizer_updates': EPOCHS * STEPS,
               'best_validation_r2': history[-1]['validation_r2'], 'best_epoch': EPOCHS,
               'stalled_epochs': 0, 'history': history,
               'train_sample_count': 276, 'validation_sample_count': 69,
               'scaler_fit_split': 'train',
               'split': {'protocol': 'outer5_inner20', 'fold': fold, 'task': task,
                         'validation_is_test': False, 'sample_count': 432,
                         'train_rows': 276, 'validation_rows': 69, 'test_rows': 87,
                         'sets_disjoint': True, 'union_equals_full_cohort': True,
                         'outer_test': 'NOT_RUN'},
               'rng': {'glt_stream_seed': None if arm == 'f2d' else 42,
                       'glt_stream_entries': 0 if arm == 'f2d' else EPOCHS,
                       'glt_stream_digest': None if arm == 'f2d' else 'deadbeef',
                       'ambient_digest_begin': 'a', 'ambient_digest_end': 'b'}}
    run = {'status': 'PASS', 'arm': arm, 'task': task, 'fold': fold, 'stage': 'development',
           'command': [sys.executable, 'scripts/finetune_glt_3d_gain_d2.py', '--arm', arm,
                       '--stage', 'development', '--checkpoint', CHECKPOINT, '--config', 'cfg'],
           'config': CONFIG, 'learning_rate_table': ARM_LRS[arm],
           'executed_epochs': EPOCHS, 'optimizer_updates': EPOCHS * STEPS,
           'split': metrics['split'], 'history': history}
    runtime = {'status': 'PASS', 'arm': arm, 'task': task, 'fold': fold,
               'stage': 'development', 'exit_code': 0}
    return run, runtime, metrics


def write_unit(root, arm, task, fold, *, drop=None, **overrides):
    run, runtime, metrics = _payloads(arm, task, fold)
    for name, change in overrides.items():
        target = {'run': run, 'runtime': runtime, 'metrics': metrics}[name.split('__')[0]]
        target[name.split('__')[1]] = change
    unit = unit_directory(root, arm, task, fold)
    unit.mkdir(parents=True, exist_ok=True)
    for name, payload in (('run.json', run), ('runtime.json', runtime), ('metrics.json', metrics)):
        (unit / name).write_text(json.dumps(payload), encoding='utf-8')
    if drop != 'best.pt':
        (unit / 'best.pt').write_bytes(b'not a real checkpoint')
    return unit


@pytest.fixture
def synthetic_root(tmp_path):
    config_path = tmp_path / 'config.json'
    config_path.write_text(json.dumps(CONFIG), encoding='utf-8')
    root = tmp_path / 'units'
    for arm, task, fold in unit_list():
        write_unit(root, arm, task, fold)
    return root, config_path


def test_complete_synthetic_set_is_accepted(synthetic_root, tmp_path):
    root, config_path = synthetic_root
    output, report = tmp_path / 'aggregate.json', tmp_path / 'report.md'
    done = subprocess.run([sys.executable, str(AGGREGATOR), '--root', str(root),
                           '--config', str(config_path), '--checkpoint', CHECKPOINT,
                           '--output', str(output), '--report', str(report)],
                          capture_output=True, text=True)
    assert done.returncode == 0, done.stdout + done.stderr
    payload = json.loads(output.read_text(encoding='utf-8'))
    assert payload['status'] == 'COMPLETE'
    assert payload['accepted_units'] == len(unit_list()) == 24
    assert set(payload['comparison']) >= {'fbase_minus_f2d', 'fnorm_minus_fbase',
                                          'fstable_minus_fnorm', 'candidates_vs_fbase',
                                          'candidates_vs_f2d', 'gate', 'selection'}
    assert payload['comparison']['epoch_cap_hits'] == []
    report = report.read_text(encoding='utf-8')
    assert 'FBASE − F2D' in report and 'FNORM − FBASE' in report and 'FSTABLE − FNORM' in report
    assert 'macro3' in report and '建议候选' in report and '限制' in report


def _history_with_nan():
    history = _payloads('fbase', 'xc', 0)[2]['history']
    history[1]['validation_r2'] = float('nan')
    return history


def test_every_predicate_rejects_its_broken_unit(synthetic_root):
    root, _ = synthetic_root
    cases = {
        'runtime status': ({'runtime__status': 'FAILED'}, 'not PASS'),
        'stage': ({'metrics__stage': 'smoke'}, 'stage'),
        'outer test': ({'metrics__outer_test': 'RUN'}, 'outer_test'),
        'requested epochs': ({'metrics__requested_epochs': 31}, 'requested_epochs'),
        'executed epochs': ({'metrics__executed_epochs': EPOCHS + 1}, 'history length'),
        'optimizer updates': ({'metrics__optimizer_updates': 2}, 'sum of per-epoch steps'),
        'best r2': ({'metrics__best_validation_r2': 0.99}, 'history maximum'),
        'best epoch': ({'metrics__best_epoch': 1}, 'argmax'),
        'non-finite best r2': ({'metrics__best_validation_r2': float('nan')}, 'not finite'),
        'non-finite history': ({'metrics__history': _history_with_nan()},
                               'non-finite validation_r2'),
        'config': ({'run__config': {'seed': 43}}, 'frozen config'),
        'learning rates': ({'run__learning_rate_table': {'o8': 1e-3}}, 'learning-rate table'),
        'deployment source': ({'run__command': ['x', '--stage', 'development',
                                                '--checkpoint', 'other.pt']}, 'deployment package'),
        'split protocol': ({'metrics__split': {'protocol': 'outer5_inner30', 'task': 'xc',
                                               'fold': 0, 'sets_disjoint': True,
                                               'union_equals_full_cohort': True,
                                               'train_rows': 276, 'validation_rows': 69,
                                               'test_rows': 87, 'sample_count': 432}},
                           'fixed fold'),
        'identity': ({'runtime__task': 'eps'}, 'identity'),
    }
    for name, (overrides, message) in cases.items():
        directory = unit_directory(root, 'fbase', 'xc', 0)
        for path in directory.iterdir():
            path.unlink()
        write_unit(root, 'fbase', 'xc', 0, **overrides)
        record, problems = check_unit(root, 'fbase', 'xc', 0, config=CONFIG,
                                      checkpoint=CHECKPOINT, max_epochs=30)
        assert record is None, name
        assert any(message in problem for problem in problems), (name, problems)


def test_missing_artifacts_and_stream_contract_are_rejected(synthetic_root):
    root, _ = synthetic_root
    directory = unit_directory(root, 'fbase', 'xc', 0)
    for path in directory.iterdir():
        path.unlink()
    write_unit(root, 'fbase', 'xc', 0, drop='best.pt')
    record, problems = check_unit(root, 'fbase', 'xc', 0, config=CONFIG,
                                  checkpoint=CHECKPOINT, max_epochs=30)
    assert record is None and any('missing artifacts' in problem for problem in problems)

    for path in directory.iterdir():
        path.unlink()
    write_unit(root, 'fbase', 'xc', 0, metrics__rng={'glt_stream_seed': 42, 'glt_stream_entries': 0})
    assert check_unit(root, 'fbase', 'xc', 0, config=CONFIG, checkpoint=CHECKPOINT,
                      max_epochs=30)[0] is None

    for path in directory.iterdir():
        path.unlink()
    write_unit(root, 'f2d', 'xc', 0, metrics__rng={'glt_stream_seed': 42})
    directory = unit_directory(root, 'f2d', 'xc', 0)
    assert check_unit(root, 'f2d', 'xc', 0, config=CONFIG, checkpoint=CHECKPOINT,
                      max_epochs=30)[0] is None


def test_incomplete_set_is_reported_and_fails(synthetic_root, tmp_path):
    root, config_path = synthetic_root
    unit = unit_directory(root, 'fstable', 'eat', 1)
    for path in unit.iterdir():
        path.unlink()
    unit.rmdir()
    output = tmp_path / 'aggregate.json'
    done = subprocess.run([sys.executable, str(AGGREGATOR), '--root', str(root),
                           '--config', str(config_path), '--checkpoint', CHECKPOINT,
                           '--output', str(output), '--report', str(tmp_path / 'r.md')],
                          capture_output=True, text=True)
    assert done.returncode == 1
    payload = json.loads(output.read_text(encoding='utf-8'))
    assert payload['status'] == 'INCOMPLETE'
    assert list(payload['problems']) == ['fstable/eat/fold1']
    assert 'comparison' not in payload


def _records(**per_arm_r2):
    """A complete record table whose arms carry the given per-task R²."""
    records = {}
    for arm, task, fold in unit_list():
        value = per_arm_r2.get(arm, {}).get(task, 0.40)
        records[(arm, task, fold)] = {'arm': arm, 'task': task, 'fold': fold,
                                      'best_validation_r2': value + 0.001 * fold,
                                      'best_epoch': 5, 'executed_epochs': 5,
                                      'optimizer_updates': 45, 'requested_epochs': 30,
                                      'hit_epoch_cap': False}
    return records


def test_deltas_and_gate_follow_the_pre_registered_thresholds():
    records = _records(f2d={task: 0.30 for task in TASKS},
                       fbase={task: 0.40 for task in TASKS},
                       fnorm={'xc': 0.45, 'eps': 0.40, 'eat': 0.40},
                       fstable={'xc': 0.42, 'eps': 0.40, 'eat': 0.40})
    assert deltas(records, 'fbase', 'f2d')['macro3'] == pytest.approx(0.10)
    gate = evaluate_gate(records, 'fnorm')
    assert gate['passed'] is True
    assert all(gate[reference]['passed'] for reference in ('fbase', 'f2d'))
    # A second, smaller margin: fstable still clears every condition, so both
    # candidates pass and the pre-registered order picks the larger macro3.
    smaller = evaluate_gate(records, 'fstable')
    assert smaller['passed'] is True
    selection = select_candidate(records)
    assert selection['passing'] == ['fnorm', 'fstable'] and selection['picked'] == 'fnorm'

    marginal = _records(f2d={task: 0.30 for task in TASKS},
                        fbase={task: 0.40 for task in TASKS},
                        fnorm={'xc': 0.407, 'eps': 0.40, 'eat': 0.40},
                        fstable={'xc': 0.40, 'eps': 0.40, 'eat': 0.40})
    gate = evaluate_gate(marginal, 'fnorm')           # +0.007 on XC and a 0.0023 macro3
    assert gate['passed'] is False
    assert gate['fbase']['conditions']['macro3_at_least_0.005'] is False
    assert gate['fbase']['conditions']['xc_mean_at_least_0.01'] is False


def test_gate_rejects_a_single_regressing_task_and_the_selection_order():
    records = _records(f2d={task: 0.30 for task in TASKS},
                       fbase={task: 0.40 for task in TASKS},
                       fnorm={'xc': 0.45, 'eps': 0.40, 'eat': 0.35},   # EAT drops 0.05
                       fstable={'xc': 0.46, 'eps': 0.40, 'eat': 0.40})
    assert evaluate_gate(records, 'fnorm')['passed'] is False
    assert evaluate_gate(records, 'fnorm')['fbase']['conditions']['no_task_mean_below_-0.01'] is False
    selection = select_candidate(records)
    assert selection['passing'] == ['fstable'] and selection['picked'] == 'fstable'

    ties = _records(f2d={task: 0.30 for task in TASKS},
                    fbase={task: 0.40 for task in TASKS},
                    fnorm={'xc': 0.45, 'eps': 0.41, 'eat': 0.40},
                    fstable={'xc': 0.45, 'eps': 0.41, 'eat': 0.40})
    selection = select_candidate(ties)
    assert selection['passing'] == list(CANDIDATES) and selection['picked'] == 'fnorm'
