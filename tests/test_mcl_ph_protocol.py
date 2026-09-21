"""Model-free contract tests for the MCL-PH run and delivery layer.

Covers the pieces that decide whether a unit may be believed at all: the smoke
launchers' failure semantics (exit-code propagation, no rerun after a failure,
completion marker only when every arm succeeded), the aggregator's rejection of
partial or inconsistent products, and the fixed ``outer5_inner20`` split
contract.  No model is instantiated and no forward or backward pass is run, so
this file consumes none of the P1 CPU model budget.
"""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from scripts.aggregate_mcl_ph import STAGE_EPOCH_LIMIT, check_unit  # noqa: E402
from scripts.finetune_glt_3d_gain_d2 import resolve_fold  # noqa: E402
from scripts.finetune_mcl_ph import ARMS, SPLIT_PROTOCOL, unit_directory  # noqa: E402

PRETRAIN_LAUNCHER = ROOT / 'scripts' / 'run_mcl_ph_pretrain_smoke.sh'
FINETUNE_LAUNCHER = ROOT / 'scripts' / 'run_mcl_ph_finetune_smoke.sh'
SPLIT_ROOT = ROOT / 'data' / 'splits' / 'mips_outer5_inner20'


# ---------------------------------------------------------------------------
# Stub launcher fixtures: a fake interpreter and a fake runner
# ---------------------------------------------------------------------------

STUB_TOOLCHAIN = '''#!/bin/bash
# Stand-in for `python -m torch.distributed.run`: drop the launcher flags and
# execute the runner script directly, so the launcher's failure contract can be
# exercised without starting a distributed job.
set -u
args=("$@")
i=0
while [ "$i" -lt "${#args[@]}" ]; do
  case "${args[$i]}" in
    -m|torch.distributed.run|--standalone|--nproc_per_node=*) i=$((i+1)) ;;
    *) break ;;
  esac
done
runner="${args[$i]}"
i=$((i+1))
exec "$runner" "${args[@]:$i}"
'''

STUB_RUNNER = '''#!/bin/bash
# Records every invocation and fails the one selected by STUB_FAIL_AT.
set -u
count=$(cat "$STUB_COUNT_FILE" 2>/dev/null || echo 0)
count=$((count + 1))
echo "$count" > "$STUB_COUNT_FILE"
printf '%s\\n' "$*" >> "$STUB_ARGV_FILE"
if [ "$count" -eq "${STUB_FAIL_AT:-0}" ]; then
  exit "${STUB_FAIL_CODE:-7}"
fi
exit 0
'''


def _write_script(path, body):
    path.write_text(body, encoding='utf-8')
    path.chmod(0o755)
    return path


def _stub_environment(tmp_path, *, fail_at=0, fail_code=7):
    interpreter = _write_script(tmp_path / 'interpreter.sh', STUB_TOOLCHAIN)
    runner = _write_script(tmp_path / 'runner.sh', STUB_RUNNER)
    count = tmp_path / 'count.txt'
    argv = tmp_path / 'argv.txt'
    environment = {
        **os.environ, 'PYTHON': str(interpreter), 'MCL_RUNNER': str(runner),
        'REF_RUNNER': str(runner), 'RUNNER': str(runner),
        'ARMS': 'cat gate xattn', 'UPDATES': '1',
        'OUTPUT': str(tmp_path / 'out'), 'LOG': str(tmp_path / 'launcher.log'),
        'STATISTICS': str(tmp_path / 'statistics.npz'),
        'STUB_COUNT_FILE': str(count), 'STUB_ARGV_FILE': str(argv),
        'STUB_FAIL_AT': str(fail_at), 'STUB_FAIL_CODE': str(fail_code)}
    return environment, count, argv


def _run_launcher(launcher, environment):
    return subprocess.run(['bash', str(launcher)], env=environment, cwd=str(ROOT),
                          capture_output=True, text=True, timeout=300)


# ---------------------------------------------------------------------------
# Launcher failure contract
# ---------------------------------------------------------------------------

def test_pretrain_launcher_propagates_the_failing_arm_and_stops(tmp_path):
    environment, count, argv = _stub_environment(tmp_path, fail_at=2, fail_code=7)
    completed = _run_launcher(PRETRAIN_LAUNCHER, environment)
    assert completed.returncode == 7, completed.stdout + completed.stderr
    log = (tmp_path / 'launcher.log').read_text(encoding='utf-8')
    assert '=== ARM=cat START' in log and '=== ARM=gate START' in log
    assert '=== ARM=gate EXIT=7' in log
    assert '=== ARM=xattn START' not in log, 'a failing arm must stop the launcher'
    assert 'no rerun within this budget' in log
    assert 'RUNNER DONE ALL_ARMS_OK' not in log
    assert count.read_text(encoding='utf-8').strip() == '2'


def test_pretrain_launcher_runs_every_arm_and_marks_completion(tmp_path):
    environment, count, argv = _stub_environment(tmp_path)
    completed = _run_launcher(PRETRAIN_LAUNCHER, environment)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    log = (tmp_path / 'launcher.log').read_text(encoding='utf-8')
    assert log.count('EXIT=0') == 3
    assert 'RUNNER DONE ALL_ARMS_OK' in log
    assert count.read_text(encoding='utf-8').strip() == '3'
    calls = argv.read_text(encoding='utf-8').splitlines()
    assert len(calls) == 3
    # The per-arm config and the fixed two-update budget reach the runner.
    assert 'configs/mts/mcl_ph_cat.json' in calls[0]
    assert 'configs/mts/mcl_ph_xattn.json' in calls[2]
    assert all('--stop-after-step 1' in call for call in calls)
    assert all('--shared-new-init' in call for call in calls)


def test_finetune_launcher_selects_the_package_of_each_arm_and_stops(tmp_path):
    environment, count, argv = _stub_environment(tmp_path, fail_at=3, fail_code=9)
    environment['ARMS'] = 'glt_ref o8_only m_cat m_gate m_xattn'
    environment['STEP'] = '2'
    completed = _run_launcher(FINETUNE_LAUNCHER, environment)
    assert completed.returncode == 9, completed.stdout + completed.stderr
    log = (tmp_path / 'launcher.log').read_text(encoding='utf-8')
    assert '=== ARM=m_cat EXIT=9' in log
    assert '=== ARM=m_gate START' not in log
    assert 'RUNNER DONE ALL_ARMS_OK' not in log
    calls = argv.read_text(encoding='utf-8').splitlines()
    assert len(calls) == 3
    # glt_ref and o8_only share the dual-GLT package; the MCL arms take their own.
    assert '--checkpoint results/mcl_ph_20260921/p1/pretrain/glt_ref/deploy_00002.pt' in calls[0]
    assert '--checkpoint results/mcl_ph_20260921/p1/pretrain/glt_ref/deploy_00002.pt' in calls[1]
    assert '--checkpoint results/mcl_ph_20260921/p1/pretrain/cat/deploy_00002.pt' in calls[2]
    assert all('--stage smoke' in call and '--epochs 1' in call for call in calls)
    assert all('--expected-pretrain-step 2' in call for call in calls)


def test_finetune_launcher_rejects_an_unknown_arm(tmp_path):
    environment, count, argv = _stub_environment(tmp_path)
    environment['ARMS'] = 'm_unknown'
    completed = _run_launcher(FINETUNE_LAUNCHER, environment)
    assert completed.returncode == 2
    log = (tmp_path / 'launcher.log').read_text(encoding='utf-8')
    assert 'unknown arm m_unknown' in log
    assert 'RUNNER DONE ALL_ARMS_OK' not in log
    assert not count.exists(), 'an unknown arm must not start a runner at all'


# ---------------------------------------------------------------------------
# Aggregator: a unit counts only when every record agrees
# ---------------------------------------------------------------------------

def _unit_records(arm='m_gate', task='xc', fold=0, stage='smoke'):
    history = [{'epoch': 1, 'train_loss': 1.5, 'validation_loss': 1.2, 'validation_r2': 0.25,
                'training_steps': 9, 'learning_rates': [1e-5, 1e-4]}]
    metrics = dict(
        arm=arm, task=task, fold=fold, stage=stage, protocol=f'mcl_ph_{stage}',
        outer_test='NOT_RUN',
        pretrained_route=('dual_glt' if arm in ('glt_ref', 'o8_only') else 'mcl_ph'),
        requested_epochs=1, executed_epochs=1, optimizer_updates=9,
        best_validation_r2=0.25, best_epoch=1, history=history,
        pretrain_step=2, pretrain_package_sha256='0' * 64,
        split=dict(protocol=SPLIT_PROTOCOL, validation_is_test=False, outer_test='NOT_RUN',
                   sets_disjoint=True, union_equals_full_cohort=True,
                   train_rows=276, validation_rows=69, test_rows=87),
        optimizer_groups=[{'name': 'backbone', 'num_parameters': 12},
                          {'name': 'backbone_no_decay', 'num_parameters': 3},
                          {'name': 'head', 'num_parameters': 4},
                          {'name': 'head_no_decay', 'num_parameters': 2}])
    run = {key: value for key, value in metrics.items() if key != 'optimizer_groups'}
    runtime = dict(status='PASS', arm=arm, task=task, fold=fold, stage=stage, exit_code=0)
    return metrics, run, runtime


def _write_unit(root, arm='m_gate', task='xc', fold=0, stage='smoke', mutate=None):
    metrics, run, runtime = _unit_records(arm, task, fold, stage)
    if mutate is not None:
        mutate(metrics, run, runtime)
    directory = unit_directory(root, arm, task, fold)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / 'metrics.json').write_text(json.dumps(metrics), encoding='utf-8')
    (directory / 'run.json').write_text(json.dumps(run), encoding='utf-8')
    (directory / 'runtime.json').write_text(json.dumps(runtime), encoding='utf-8')
    (directory / 'best.pt').write_bytes(b'stub-checkpoint')
    np.savez(directory / 'validation_predictions.npz',
             sample_keys=np.asarray(['aa', 'bb']), y_true=np.asarray([1.0, 2.0]),
             y_pred=np.asarray([1.1, 1.9]),
             best_epoch=np.asarray(metrics['best_epoch'], dtype=np.int64),
             split_protocol=np.asarray(SPLIT_PROTOCOL), outer_test=np.asarray('NOT_RUN'))
    return directory


def _aggregate(root, *, stage='smoke', step=2, arms=tuple(ARMS), output=None):
    destination = output or (Path(root) / 'aggregate.json')
    completed = subprocess.run(
        [sys.executable, str(ROOT / 'scripts' / 'aggregate_mcl_ph.py'), '--root', str(root),
         '--stage', stage, '--expected-pretrain-step', str(step), '--arms', *arms,
         '--tasks', 'xc', '--folds', '0', '--output', str(destination)],
        cwd=str(ROOT), capture_output=True, text=True, timeout=300)
    payload = json.loads(destination.read_text(encoding='utf-8')) if \
        Path(destination).is_file() else None
    return completed, payload


def _write_all_arms(root, stage='smoke'):
    for arm in ARMS:
        _write_unit(root, arm=arm, stage=stage)


def test_aggregator_accepts_only_the_full_five_arm_scope(tmp_path):
    _write_all_arms(tmp_path)
    completed, payload = _aggregate(tmp_path)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert payload['status'] == 'PASS' and payload['acceptance'] == 'PASS'
    assert payload['units_accepted'] == len(ARMS) == payload['units_expected']
    assert payload['rejected'] == [] and payload['outer_test'] == 'NOT_RUN'
    assert payload['acceptance_arms'] == list(ARMS)


def test_aggregator_marks_a_subset_scope_as_partial(tmp_path):
    'A verified subset is never reported as acceptance (section 10 P1).'
    _write_all_arms(tmp_path)
    completed, payload = _aggregate(tmp_path, arms=tuple(ARMS)[:4])
    assert completed.returncode == 5, completed.stdout + completed.stderr
    assert payload['status'] == 'PARTIAL' and payload['acceptance'] == 'PARTIAL'
    assert payload['units_accepted'] == 4 and payload['rejected'] == []
    assert payload['requested_arms'] == list(ARMS)[:4]
    assert payload['acceptance_arms'] == list(ARMS)
    assert 'smaller than' in payload['partial_reason']


def test_aggregator_rejects_a_conflicting_unit_inside_the_full_scope(tmp_path):
    _write_all_arms(tmp_path)
    unit = unit_directory(tmp_path, 'm_xattn', 'xc', 0)
    metrics = json.loads((unit / 'metrics.json').read_text(encoding='utf-8'))
    metrics['pretrain_step'] = 7
    (unit / 'metrics.json').write_text(json.dumps(metrics), encoding='utf-8')
    completed, payload = _aggregate(tmp_path)
    assert completed.returncode == 4
    assert payload['status'] == 'INCOMPLETE' and payload['units_accepted'] == len(ARMS) - 1
    assert payload['rejected'][0]['arm'] == 'm_xattn'
    assert 'pretrain_step' in ' '.join(payload['rejected'][0]['problems'])


def test_aggregator_rejects_a_missing_arm_inside_the_full_scope(tmp_path):
    _write_all_arms(tmp_path)
    import shutil

    shutil.rmtree(unit_directory(tmp_path, 'm_cat', 'xc', 0))
    completed, payload = _aggregate(tmp_path)
    assert completed.returncode == 4
    assert payload['status'] == 'INCOMPLETE'
    assert 'missing artifacts' in ' '.join(payload['rejected'][0]['problems'])


def test_aggregator_rejects_an_empty_unit_directory(tmp_path):
    unit_directory(tmp_path, 'm_gate', 'xc', 0).mkdir(parents=True)
    completed, payload = _aggregate(tmp_path)
    assert completed.returncode == 4
    assert payload['status'] == 'INCOMPLETE'
    assert 'missing artifacts' in ' '.join(payload['rejected'][0]['problems'])


def test_aggregator_rejects_a_failed_run_that_kept_its_directory(tmp_path):
    _write_unit(tmp_path, mutate=lambda metrics, run, runtime: runtime.update(
        status='FAILED', exit_code=1, error='RuntimeError: nonfinite validation metrics'))
    completed, payload = _aggregate(tmp_path, arms=('m_gate',))
    assert completed.returncode == 4
    problems = ' '.join(payload['rejected'][0]['problems'])
    assert 'runtime status' in problems and 'exit_code' in problems


def test_aggregator_rejects_partial_and_inconsistent_products(tmp_path):
    def wrong_stage(metrics, run, runtime):
        metrics['stage'] = run['stage'] = 'development'
        metrics['protocol'] = run['protocol'] = 'mcl_ph_development'

    def reused_smoke(metrics, run, runtime):
        metrics['requested_epochs'] = STAGE_EPOCH_LIMIT['smoke'] + 4

    def outer_test_leak(metrics, run, runtime):
        metrics['outer_test'] = 'RUN'

    def split_without_the_disjointness_assertion(metrics, run, runtime):
        metrics['split']['sets_disjoint'] = False

    def validation_is_test(metrics, run, runtime):
        metrics['split']['validation_is_test'] = True

    def non_finite_metric(metrics, run, runtime):
        metrics['history'][0]['validation_r2'] = float('nan')

    def wrong_route(metrics, run, runtime):
        metrics['pretrained_route'] = 'dual_glt'

    def wrong_pretrain_step(metrics, run, runtime):
        metrics['pretrain_step'] = 3

    def wrong_selected_epoch(metrics, run, runtime):
        metrics['best_epoch'] = 2
        metrics['best_validation_r2'] = 0.25

    def history_disagrees(metrics, run, runtime):
        metrics['executed_epochs'] = 2

    def undeclared_group(metrics, run, runtime):
        metrics['optimizer_groups'][0]['name'] = 'router'

    def identity_mismatch(metrics, run, runtime):
        run['fold'] = 1

    cases = [
        ('wrong_stage', wrong_stage, "stage is 'development'", None),
        ('reused_smoke', reused_smoke, 'outside 1..1', None),
        ('outer_test_leak', outer_test_leak, 'outer_test', None),
        ('split_claims_disjoint', split_without_the_disjointness_assertion, 'sets_disjoint', None),
        ('validation_is_test', validation_is_test, 'validation_is_test=false', None),
        ('non_finite_metric', non_finite_metric, 'not finite', None),
        ('wrong_route', wrong_route, 'mcl_ph pre-training route', None),
        ('wrong_pretrain_step', wrong_pretrain_step, 'pretrain_step', None),
        ('wrong_selected_epoch', wrong_selected_epoch, 'outside the executed epochs', None),
        ('history_disagrees', history_disagrees, 'executed_epochs', None),
        ('undeclared_group', undeclared_group, 'undeclared name', None),
        ('identity_mismatch', identity_mismatch, 'identity does not match', None),
        ('missing_checkpoint', None, 'missing artifacts', 'best.pt'),
        ('missing_predictions', None, 'missing artifacts', 'validation_predictions.npz'),
    ]
    for name, mutate, expected, drop in cases:
        root = tmp_path / name
        _write_unit(root, mutate=mutate)
        if drop is not None:
            (unit_directory(root, 'm_gate', 'xc', 0) / drop).unlink()
        completed, payload = _aggregate(root, arms=('m_gate',))
        assert payload is not None, f'{name}: no aggregate written\n{completed.stderr}'
        problems = ' '.join(payload['rejected'][0]['problems'])
        assert completed.returncode == 4, f'{name}: {completed.stdout}{completed.stderr}'
        assert payload['status'] == 'INCOMPLETE', name
        assert expected in problems, f'{name}: {problems}'


def test_aggregator_rejects_nan_validation_predictions(tmp_path):
    directory = _write_unit(tmp_path)
    np.savez(directory / 'validation_predictions.npz',
             sample_keys=np.asarray(['aa', 'bb']), y_true=np.asarray([1.0, 2.0]),
             y_pred=np.asarray([1.1, float('nan')]), best_epoch=np.asarray(1, dtype=np.int64),
             split_protocol=np.asarray(SPLIT_PROTOCOL), outer_test=np.asarray('NOT_RUN'))
    completed, payload = _aggregate(tmp_path, arms=('m_gate',))
    assert completed.returncode == 4
    assert 'NaN/Inf' in ' '.join(payload['rejected'][0]['problems'])


def test_aggregator_rejects_predictions_from_another_epoch(tmp_path):
    directory = _write_unit(tmp_path)
    np.savez(directory / 'validation_predictions.npz',
             sample_keys=np.asarray(['aa', 'bb']), y_true=np.asarray([1.0, 2.0]),
             y_pred=np.asarray([1.1, 1.9]), best_epoch=np.asarray(7, dtype=np.int64),
             split_protocol=np.asarray(SPLIT_PROTOCOL), outer_test=np.asarray('NOT_RUN'))
    completed, payload = _aggregate(tmp_path, arms=('m_gate',))
    assert completed.returncode == 4
    assert 'not from the selected epoch' in ' '.join(payload['rejected'][0]['problems'])


def test_aggregator_counts_every_missing_arm_of_the_set(tmp_path):
    _write_unit(tmp_path)
    completed, payload = _aggregate(tmp_path, arms=('glt_ref', 'o8_only', 'm_cat', 'm_gate',
                                                    'm_xattn'))
    assert completed.returncode == 4
    assert payload['units_expected'] == 5 and payload['units_accepted'] == 1
    assert payload['units_rejected'] == 4


def test_aggregator_refuses_development_aggregation(tmp_path):
    completed, payload = _aggregate(tmp_path, stage='development')
    assert completed.returncode != 0
    assert 'not part of the P1 budget' in completed.stdout + completed.stderr


def test_check_unit_reports_every_problem_instead_of_the_first(tmp_path):
    _write_unit(tmp_path, mutate=lambda metrics, run, runtime: (
        metrics.update(outer_test='RUN', pretrain_step=9),
        runtime.update(status='FAILED', exit_code=1)))
    _, problems = check_unit(tmp_path, 'm_gate', 'xc', 0, stage='smoke', expected_step=2)
    joined = ' '.join(problems)
    assert 'outer_test' in joined and 'pretrain_step' in joined and 'runtime status' in joined
    assert len(problems) >= 3


# ---------------------------------------------------------------------------
# The fixed split contract
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('task', ['xc', 'eps', 'eat'])
def test_the_fixed_split_keeps_the_outer_test_rows_out(task):
    manifest = json.loads((SPLIT_ROOT / f'{task}.json').read_text(encoding='utf-8'))
    count = int(manifest['sample_count'])
    for entry in manifest['folds']:
        train, validation, evidence = resolve_fold(manifest, task, entry['fold'],
                                                   cohort_rows=count)
        outer_test = set(entry['test_indices'])
        assert set(train) | set(validation) | outer_test == set(range(count))
        assert not (set(train) | set(validation)) & outer_test
        assert evidence['validation_is_test'] is False
        assert evidence['outer_test'] == 'NOT_RUN'
        assert evidence['protocol'] == SPLIT_PROTOCOL
        assert len(train) == evidence['train_rows'] > 0
        assert len(validation) == evidence['validation_rows'] > 0


def test_resolve_fold_rejects_a_broken_split_manifest():
    manifest = json.loads((SPLIT_ROOT / 'xc.json').read_text(encoding='utf-8'))
    count = int(manifest['sample_count'])

    def broken(**changes):
        copy = json.loads(json.dumps(manifest))
        copy.update(changes)
        return copy

    with pytest.raises(ValueError, match='protocol'):
        resolve_fold(broken(protocol='outer5_only'), 'xc', 0, cohort_rows=count)
    with pytest.raises(ValueError, match='validation_is_test'):
        resolve_fold(broken(validation_is_test=True), 'xc', 0, cohort_rows=count)
    with pytest.raises(ValueError, match='manifest task'):
        resolve_fold(broken(), 'eps', 0, cohort_rows=count)
    with pytest.raises(ValueError, match='no fold 7'):
        resolve_fold(broken(), 'xc', 7, cohort_rows=count)
    with pytest.raises(ValueError, match='differs from the fixed split'):
        resolve_fold(broken(), 'xc', 0, cohort_rows=count + 1)
    overlap = json.loads(json.dumps(manifest))
    overlap['folds'][0]['validation_indices'] = list(overlap['folds'][0]['validation_indices'])
    overlap['folds'][0]['validation_indices'][0] = overlap['folds'][0]['train_indices'][0]
    with pytest.raises(ValueError, match='overlap'):
        resolve_fold(overlap, 'xc', 0, cohort_rows=count)
    # One outer-test row is dropped from every set: the fold no longer covers
    # the cohort, so the manifest is refused instead of silently shrinking.
    short = json.loads(json.dumps(manifest))
    short['folds'][0]['test_indices'].pop()
    with pytest.raises(ValueError, match='cover'):
        resolve_fold(short, 'xc', 0, cohort_rows=count)


def test_unit_directory_refuses_undeclared_names(tmp_path):
    for arm, task, fold in (('m_unknown', 'xc', 0), ('m_gate', 'nc', 0), ('m_gate', 'xc', 3)):
        with pytest.raises(ValueError):
            unit_directory(tmp_path, arm, task, fold)


# ---------------------------------------------------------------------------
# Runner-side guards that must trip before any data or model work
# ---------------------------------------------------------------------------

def _run_finetune(extra, *, output, statistics=None):
    command = [sys.executable, str(ROOT / 'scripts' / 'finetune_mcl_ph.py'),
               '--arm', 'm_gate', '--stage', 'smoke', '--config',
               str(ROOT / 'configs' / 'mts' / 'mcl_ph_gate.json'), '--checkpoint',
               str(ROOT / 'results' / 'mcl_ph_20260921' / 'p1' / 'pretrain' / 'gate'
                   / 'deploy_00002.pt'),
               '--expected-pretrain-step', '2', '--cohort-root', str(ROOT / 'missing'),
               '--cache-root', str(ROOT / 'missing'), '--dual-static-root', str(ROOT / 'missing'),
               '--split-root', str(SPLIT_ROOT), '--task', 'xc', '--fold', '0',
               '--output', str(output)]
    if statistics is not None:
        command += ['--statistics', str(statistics)]
    return subprocess.run(command + extra, cwd=str(ROOT), capture_output=True, text=True,
                          timeout=300)


def test_the_smoke_stage_refuses_more_than_one_epoch(tmp_path):
    completed = _run_finetune(['--epochs', '2'], output=tmp_path / 'out')
    assert completed.returncode != 0
    assert 'exactly one epoch' in completed.stdout + completed.stderr
    assert not (tmp_path / 'out').exists(), 'a refused invocation must not leave a unit folder'


def test_the_development_stage_refuses_a_longer_budget_than_the_schedule(tmp_path):
    completed = _run_finetune(['--stage', 'development', '--epochs', '31'],
                              output=tmp_path / 'out')
    assert completed.returncode != 0
    assert 'development units allow 1..30 epochs' in completed.stdout + completed.stderr
    assert not (tmp_path / 'out').exists()


def test_the_mcl_arms_require_the_shared_statistics_artifact(tmp_path):
    completed = _run_finetune(['--epochs', '1'], output=tmp_path / 'out')
    assert completed.returncode != 0
    assert 'shared statistics artifact' in completed.stdout + completed.stderr
    assert not (tmp_path / 'out').exists()


def test_the_pretrain_runner_records_a_non_pass_runtime_on_failure(tmp_path):
    """The failure record of a run that reaches the output directory.

    The runner is launched the way production launches it (``torchrun``, the
    config's four ranks), so the failure at the missing statistics file happens
    *after* the output directory exists -- the only situation in which the
    contract promises a record on disk.  Rank 0 owns ``runtime.json`` and every
    rank writes its own error, and no ``run.json`` or success marker may appear.
    """
    if not os.environ.get('TMUX'):
        pytest.skip('the pre-training runner requires the logged tmux session')
    probe = subprocess.run(['tmux', 'display-message', '-p', '#S'], capture_output=True, text=True)
    if probe.stdout.strip() != 'Uni-Poly':
        pytest.skip('the pre-training runner requires tmux session Uni-Poly')
    import torch

    if torch.cuda.is_available() and torch.cuda.device_count() < 4:
        pytest.skip('the four-rank configuration needs four visible devices')
    config = tmp_path / 'gate.json'
    config.write_text((ROOT / 'configs' / 'mts' / 'mcl_ph_gate.json').read_text(encoding='utf-8'),
                      encoding='utf-8')
    output = tmp_path / 'out'
    completed = subprocess.run(
        [sys.executable, '-m', 'torch.distributed.run', '--standalone',
         '--nproc_per_node', '4', str(ROOT / 'scripts' / 'pretrain_mcl_ph.py'),
         '--config', str(config),
         '--cohort-root', str(ROOT / 'missing'), '--cache-root', str(ROOT / 'missing'),
         '--dual-static-root', str(ROOT / 'missing'), '--statistics', str(ROOT / 'missing'),
         '--shared-new-init', str(tmp_path / 'shared.pt'), '--output', str(output)],
        cwd=str(ROOT), capture_output=True, text=True, timeout=600)
    assert completed.returncode == 1, completed.stdout + completed.stderr
    runtime = json.loads((output / 'runtime.json').read_text(encoding='utf-8'))
    assert runtime['status'] == 'FAILED'
    assert runtime['exit_code'] == completed.returncode
    assert 'FileNotFoundError' in runtime['error']
    for rank in range(4):
        record = json.loads((output / f'runtime_failure_rank{rank}.json').read_text(
            encoding='utf-8'))
        assert record['status'] == 'FAILED', rank
        assert record['rank'] == rank, rank
    assert json.loads(completed.stdout.strip().splitlines()[-1])['status'] == 'FAILED'
    assert not (output / 'run.json').exists(), 'a failed run must not leave a run record'
    assert 'ALL_ARMS_OK' not in completed.stdout


def test_a_failure_before_the_output_directory_is_reported_on_stdout_only(tmp_path):
    """A preparation failure cannot leave a record: it has no directory to leave it in.

    Invoked outside ``torchrun`` the runner dies on the world-size guard before it
    creates the output directory.  What it must still do is exit non-zero, print a
    FAILED status, never print PASS or a done marker, and leave nothing behind that
    a later resume could mistake for a started run.
    """
    if not os.environ.get('TMUX'):
        pytest.skip('the pre-training runner requires the logged tmux session')
    probe = subprocess.run(['tmux', 'display-message', '-p', '#S'], capture_output=True, text=True)
    if probe.stdout.strip() != 'Uni-Poly':
        pytest.skip('the pre-training runner requires tmux session Uni-Poly')
    config = tmp_path / 'gate.json'
    config.write_text((ROOT / 'configs' / 'mts' / 'mcl_ph_gate.json').read_text(encoding='utf-8'),
                      encoding='utf-8')
    output = tmp_path / 'out'
    completed = subprocess.run(
        [sys.executable, str(ROOT / 'scripts' / 'pretrain_mcl_ph.py'), '--config', str(config),
         '--cohort-root', str(ROOT / 'missing'), '--cache-root', str(ROOT / 'missing'),
         '--dual-static-root', str(ROOT / 'missing'), '--statistics', str(ROOT / 'missing'),
         '--shared-new-init', str(tmp_path / 'shared.pt'), '--output', str(output)],
        cwd=str(ROOT), capture_output=True, text=True, timeout=600)
    assert completed.returncode == 1, completed.stdout + completed.stderr
    assert json.loads(completed.stdout.strip().splitlines()[0])['status'] == 'FAILED'
    assert 'world size' in completed.stdout + completed.stderr
    assert 'PASS' not in completed.stdout.replace('FAILED', '')
    assert 'ALL_ARMS_OK' not in completed.stdout
    assert not output.exists(), 'a refused invocation must not create the output directory'


def test_the_pretrain_runner_refuses_a_top1_router_config(tmp_path):
    if not os.environ.get('TMUX'):
        pytest.skip('the pre-training runner requires the logged tmux session')
    probe = subprocess.run(['tmux', 'display-message', '-p', '#S'], capture_output=True, text=True)
    if probe.stdout.strip() != 'Uni-Poly':
        pytest.skip('the pre-training runner requires tmux session Uni-Poly')
    config = json.loads((ROOT / 'configs' / 'mts' / 'mcl_ph_gate.json').read_text(encoding='utf-8'))
    config['router_top_k_afterwards'] = 1
    path = tmp_path / 'top1.json'
    path.write_text(json.dumps(config), encoding='utf-8')
    completed = subprocess.run(
        [sys.executable, str(ROOT / 'scripts' / 'pretrain_mcl_ph.py'), '--config', str(path),
         '--cohort-root', str(ROOT / 'missing'), '--cache-root', str(ROOT / 'missing'),
         '--dual-static-root', str(ROOT / 'missing'), '--statistics', str(ROOT / 'missing'),
         '--shared-new-init', str(tmp_path / 'shared.pt'), '--output', str(tmp_path / 'out')],
        cwd=str(ROOT), capture_output=True, text=True, timeout=300)
    assert completed.returncode != 0
    assert 'Top-1 routing is forbidden' in completed.stdout + completed.stderr
    assert not (tmp_path / 'out').exists()
