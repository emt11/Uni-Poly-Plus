"""r5 fixture: the completion contract, the degenerate diagnostics, the memory windows.

Everything here is model-free and budget-free on purpose (the r5 order allows at
most four CPU diagnostic/submodule forwards): the launcher contract is checked
with a stub runner and synthetic records, the diagnostics with the diagnostic
function itself and two tiny ``FusionGate`` forwards, the memory labels with a
direct call.  No training path, no dataset, no checkpoint is touched.
"""
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.verify_mcl_ph_arm import verify  # noqa: E402
from src.modules.mcl_ph import FusionGate, MCLPHBranch  # noqa: E402

STUB = ROOT / 'tests' / '_mcl_ph_r5_stub_runner.py'
LAUNCHER = ROOT / 'scripts' / 'run_mcl_ph_pretrain_smoke.sh'
SUPERVISOR = ROOT / 'tests' / '_mcl_ph_r5_stall_supervisor.py'


def synthetic_arm(directory, mode, step=2):
    """The record shapes a finished arm can leave behind."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / 'run.json').write_text(json.dumps({'stub': mode}), encoding='utf-8')
    (directory / 'steps.jsonl').write_text('{"step": 1}\n', encoding='utf-8')
    (directory / f'resume_{step:05d}.pt').write_bytes(b'resume')
    runtime = {'status': 'PASS', 'cleanup': 'complete', 'main_returned': True,
               'completed_steps': step}
    if mode == 'training_complete':
        runtime = {'status': 'TRAINING_COMPLETE', 'cleanup': 'pending',
                   'main_returned': False, 'completed_steps': step}
    else:
        (directory / f'deploy_{step:05d}.pt').write_bytes(b'deploy')
    (directory / 'runtime.json').write_text(json.dumps(runtime), encoding='utf-8')
    return directory


def test_a_finished_arm_is_accepted(tmp_path):
    problems, evidence = verify(synthetic_arm(tmp_path / 'cat', 'pass'), 2, True)
    assert problems == []
    assert evidence['status'] == 'PASS' and evidence['cleanup'] == 'complete'


def test_training_complete_without_cleanup_is_rejected(tmp_path):
    """Training and export finished, cleanup did not: the r3 shape, not a PASS."""
    problems, evidence = verify(synthetic_arm(tmp_path / 'cat', 'training_complete'), 2, True)
    assert any('not PASS' in problem for problem in problems)
    assert any("cleanup is 'pending'" in problem for problem in problems)
    assert any('missing file: deploy_00002.pt' in problem for problem in problems)
    assert evidence['status'] == 'TRAINING_COMPLETE'


def test_a_missing_product_is_rejected_even_with_a_passing_record(tmp_path):
    directory = synthetic_arm(tmp_path / 'cat', 'pass')
    (directory / 'deploy_00002.pt').unlink()
    problems, _evidence = verify(directory, 2, True)
    assert problems == ['missing file: deploy_00002.pt']


def test_wrong_update_count_is_rejected(tmp_path):
    problems, _evidence = verify(synthetic_arm(tmp_path / 'cat', 'pass'), 3, True)
    assert any('completed_steps' in problem for problem in problems)


def test_base_contract_accepts_the_reference_records_shape(tmp_path):
    """The r1 reference runner has no cleanup fields; its arm gets the base level."""
    directory = synthetic_arm(tmp_path / 'glt_ref', 'pass')
    runtime = json.loads((directory / 'runtime.json').read_text(encoding='utf-8'))
    runtime.pop('cleanup'), runtime.pop('main_returned')
    (directory / 'runtime.json').write_text(json.dumps(runtime), encoding='utf-8')
    assert verify(directory, 2, False)[0] == []
    assert any('cleanup' in problem for problem in verify(directory, 2, True)[0])


def launch_with_stub(tmp_path, mode, timeout_seconds=60):
    log = tmp_path / f'run_{mode}.log'
    output = tmp_path / f'out_{mode}'
    environment = dict(os.environ, ARMS='cat', UPDATES='2', NPROC='1',
                       MCL_RUNNER=str(STUB.relative_to(ROOT)), STUB_MODE=mode,
                       OUTPUT=str(output), LOG=str(log),
                       PYTHON=sys.executable)
    completed = subprocess.run(['bash', str(LAUNCHER)], cwd=ROOT, env=environment,
                               capture_output=True, text=True, timeout=timeout_seconds)
    return completed, log.read_text(encoding='utf-8')


def test_launcher_refuses_to_mark_complete_when_cleanup_never_finished(tmp_path):
    completed, log = launch_with_stub(tmp_path, 'training_complete')
    assert completed.returncode == 6, log
    assert 'ALL_ARMS_OK' not in log
    assert 'VERIFY=1' in log and 'ABORT after cat (verification 1)' in log


def test_launcher_refuses_to_mark_complete_when_the_process_failed(tmp_path):
    """torchrun masks the child's own exit code as 1; the marker stays unwritten."""
    completed, log = launch_with_stub(tmp_path, 'fail')
    assert completed.returncode == 1, log
    assert 'ARM=cat EXIT=1' in log and 'ABORT after cat (exit 1)' in log
    assert 'ALL_ARMS_OK' not in log and 'VERIFY=' not in log


def test_launcher_refuses_to_mark_complete_when_the_run_times_out(tmp_path):
    """An external timeout kills the launcher mid-arm: no marker, no promotion."""
    environment = dict(os.environ, ARMS='cat', UPDATES='2', NPROC='1',
                       MCL_RUNNER=str(STUB.relative_to(ROOT)), STUB_MODE='hang',
                       OUTPUT=str(tmp_path / 'out'), LOG=str(tmp_path / 'run.log'),
                       PYTHON=sys.executable)
    completed = subprocess.run(['timeout', '-k', '5', '12', 'bash', str(LAUNCHER)],
                               cwd=ROOT, env=environment, capture_output=True, text=True,
                               timeout=60)
    log = (tmp_path / 'run.log').read_text(encoding='utf-8')
    assert completed.returncode == 124, log
    assert 'ALL_ARMS_OK' not in log and 'ARM=cat EXIT=' not in log


def test_launcher_writes_the_marker_only_after_every_check_passed(tmp_path):
    completed, log = launch_with_stub(tmp_path, 'pass')
    assert completed.returncode == 0, log
    assert 'ALL_ARMS_OK' in log
    assert log.index('VERIFY=0') < log.index('ALL_ARMS_OK')


def test_supervisor_stops_a_stalled_tree_and_preserves_the_site(tmp_path):
    """180 s of no progress must stop this run's own tree, not a neighbour's."""
    stages = tmp_path / 'cat'
    stages.mkdir()
    (stages / 'stages_rank0.log').write_text('{"stage": "cleanup", "event": "enter"}\n',
                                             encoding='utf-8')
    child = subprocess.Popen([sys.executable, '-c',
                              'import time\ntime.sleep(120)'], cwd=str(tmp_path))
    time.sleep(1.5)
    report_path = tmp_path / 'stall_supervisor.json'
    completed = subprocess.run([sys.executable, str(SUPERVISOR), '--root-pid', str(child.pid),
                                '--stages-dir', str(stages), '--report', str(report_path),
                                '--poll-seconds', '0.5', '--stack-seconds', '1',
                                '--stop-seconds', '3', '--grace-seconds', '2'],
                               capture_output=True, text=True, timeout=60)
    report = json.loads(report_path.read_text(encoding='utf-8'))
    assert completed.returncode == 9, completed.stdout + completed.stderr
    assert report['intervened'] is True and report['status'] == 'STOPPED_BY_SUPERVISOR'
    assert str(child.pid) in [str(pid) for pid in report['tree']['signalled_terminate']]
    assert child.poll() is not None, 'the stalled child is still running'
    assert report['stage_tail']['stages_rank0.log'] == [
        '{"stage": "cleanup", "event": "enter"}']
    assert os.getpid() not in report['tree']['signalled_terminate']


def test_supervisor_leaves_a_finished_run_alone(tmp_path):
    stages = tmp_path / 'cat'
    stages.mkdir()
    (stages / 'stages_rank0.log').write_text('{"stage": "cleanup", "event": "complete"}\n',
                                             encoding='utf-8')
    child = subprocess.Popen([sys.executable, '-c', 'import time\ntime.sleep(2)'])
    completed = subprocess.run([sys.executable, str(SUPERVISOR), '--root-pid', str(child.pid),
                                '--stages-dir', str(stages),
                                '--report', str(tmp_path / 'report.json'),
                                '--poll-seconds', '0.5', '--stack-seconds', '1',
                                '--stop-seconds', '60'],
                               capture_output=True, text=True, timeout=60)
    assert completed.returncode == 0
    assert 'STOPPED_BY_SUPERVISOR' not in completed.stdout
    child.wait(timeout=30)


def routed_logits(graphs, mode='top2'):
    logits = torch.randn(graphs, 3)
    dense = torch.softmax(logits, dim=-1)
    order = (torch.argsort(logits, dim=-1, descending=True, stable=True)
             if mode == 'top2' else None)
    return {'logits': logits, 'dense': dense, 'weights': dense,
            'top_k': (order[:, :2] if order is not None else None), 'order': order}


def finite(value):
    if isinstance(value, list):
        return all(finite(item) for item in value)
    return value is None or (isinstance(value, float) and math.isfinite(value))


def test_router_diagnostics_single_graph_and_empty_batch():
    """One graph reports a finite 0 std; an empty batch reports NOT_APPLICABLE."""
    self_stub = SimpleNamespace(router=SimpleNamespace(mode='top2'))
    single = MCLPHBranch._diagnostics(self_stub, routed_logits(1), torch.ones(1, dtype=torch.bool), 1)
    assert single['graphs'] == 1
    assert single['router_statistics'] == 'population_std_correction_0'
    assert 'single graph' in single['router_statistics_note']
    assert finite(single['router_logits_std']) and single['router_logits_std'] == [0.0, 0.0, 0.0]
    assert finite(single['router_entropy_std']) and single['router_entropy_std'] == 0.0
    assert finite(single['router_logits_mean']) and finite(single['router_probability_std'])
    assert single['router_tie_rate'] in (0.0, 1.0) and single['router_hard_selection'] is not None

    empty = MCLPHBranch._diagnostics(self_stub, routed_logits(0),
                                     torch.zeros(0, dtype=torch.bool), 0)
    assert empty['graphs'] == 0
    assert empty['router_statistics'] == 'NOT_APPLICABLE_EMPTY_BATCH'
    for key in ('router_logits_mean', 'router_logits_std', 'router_logits_min',
                'router_logits_max', 'router_probability_mean', 'router_probability_std',
                'router_entropy_mean', 'router_entropy_std', 'router_entropy_min',
                'router_entropy_max', 'router_tie_count', 'router_tie_rate'):
        assert empty[key] is None, key


def test_fusion_gate_diagnostics_single_graph_and_empty_batch():
    """The gate's std follows the router's rule; an empty batch does not raise."""
    torch.manual_seed(0)
    gate = FusionGate(hidden=8, token=4)
    gate.collect_diagnostics = True
    hidden = torch.randn(1, 8)
    tokens = torch.randn(1, 4)
    with torch.no_grad():
        gate(hidden, tokens)
    single = gate.last_diagnostics
    assert single['gate_statistics'] == 'population_std_correction_0'
    assert single['gate_std'] == [0.0, 0.0, 0.0, 0.0] and finite(single['gate_min'])
    with torch.no_grad():
        gate(torch.zeros(0, 8), torch.zeros(0, 4))
    empty = gate.last_diagnostics
    assert empty['gate_statistics'] == 'NOT_APPLICABLE_EMPTY_BATCH'
    assert empty['gate_mean'] is None and empty['gate_std'] is None
    assert empty['gate_min'] is None and empty['gate_max'] is None


def test_memory_record_labels_its_two_windows_separately():
    """CUDA covers the step window since the reset; CPU covers the process lifetime."""
    from scripts.pretrain_mcl_ph import memory_record
    record = memory_record()
    assert record['units'] == 'bytes'
    assert record['cuda']['window'] == 'step_since_last_reset'
    assert record['cpu']['window'] == 'rank_process_lifetime'
    for key in ('peak_allocated_bytes', 'peak_reserved_bytes'):
        assert record['cuda'][key] == 'NOT_MEASURED' or isinstance(record['cuda'][key], int)
    assert record['cpu']['peak_rss_bytes'] == 'NOT_MEASURED' or \
        record['cpu']['peak_rss_bytes'] > 0
    assert record['cpu']['scope'] == 'rank_process_peak_rss_dataloader_workers_excluded'
    assert 'window' not in record, 'the two windows must not share one vague label'


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-q']))
