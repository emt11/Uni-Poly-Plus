"""r7A fixture: the two engineering repairs, model-free.

Fix A (`shared` lifecycle): a functional test of the denominator helper the
training entry calls, plus a source contract that the initialization identity
cannot be shadowed again.

Fix B (supervisor progress): a unit test of the marker itself, then isolated
fake runs that prove the stop rule follows real stage marks only — stack dumps
and plain log growth cannot postpone the stop, a real mark resets the timer, and
a normal exit is never reported as a stall.

No model, no dataset, no GPU, no training process: every child here is a plain
``time.sleep`` process created by the test, and only its own tree is signalled.
"""
import ast
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tests'))

import _mcl_ph_r5_stall_supervisor as supervisor  # noqa: E402
from scripts.pretrain_mcl_ph import reduce_update_denominators  # noqa: E402

RUNNER = ROOT / 'scripts' / 'pretrain_mcl_ph.py'
SUPERVISOR = ROOT / 'tests' / '_mcl_ph_r5_stall_supervisor.py'
SLEEPER = 'import time\ntime.sleep(120)'
MARK = '{"event": "enter", "rank": 0, "stage": "step", "step": %d}\n'


# --------------------------------------------------------------------------- A


def test_update_denominators_are_the_globally_summed_effective_graph_counts():
    """The helper the training entry calls: same value, same collective input."""
    seen = []

    def world4_sum(tensor):
        seen.append(float(tensor.reshape(-1)[0]))
        return tensor * 4

    assert reduce_update_denominators({'atom': 252, 'geometry': 252},
                                      world4_sum, 'cpu') == {'atom': 1008.0,
                                                             'geometry': 1008.0}
    assert seen == [252.0, 252.0], 'the collective must receive this rank\'s own count'
    assert reduce_update_denominators({'atom': 252, 'geometry': 251},
                                      world4_sum, 'cpu') == {'atom': 1008.0,
                                                             'geometry': 1004.0}


def bound_names(tree):
    """Every name this module binds anywhere, with the line numbers."""
    bound = {}

    def record(target):
        if target is None:
            return
        for node in ast.walk(target):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                bound.setdefault(node.id, []).append(node.lineno)

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                record(target)
        elif isinstance(node, (ast.AugAssign, ast.AnnAssign, ast.For)):
            record(node.target)
        elif isinstance(node, ast.withitem):
            record(node.optional_vars)
        elif isinstance(node, ast.comprehension):
            record(node.target)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            bound.setdefault(node.name, []).append(node.lineno)
        elif isinstance(node, ast.arg):
            bound.setdefault(node.arg, []).append(node.lineno)
    return bound


def test_the_initialization_identity_cannot_be_shadowed_again():
    tree = ast.parse(RUNNER.read_text(encoding='utf-8'))
    bound = bound_names(tree)
    assert 'shared' not in bound, f'`shared` is still bound at lines {bound.get("shared")}'
    assert len(bound.get('shared_init', [])) == 1, (
        'the initialization identity must be bound exactly once, '
        f'bound at {bound.get("shared_init")}')
    assignment = [node for node in ast.walk(tree) if isinstance(node, ast.Assign)
                  and any(isinstance(target, ast.Name) and target.id == 'shared_init'
                          for target in node.targets)]
    assert len(assignment) == 1
    value = assignment[0].value
    assert isinstance(value, ast.Call) and getattr(value.func, 'id', None) == 'apply_shared_init'


def test_the_deploy_metadata_reads_that_same_identity():
    tree = ast.parse(RUNNER.read_text(encoding='utf-8'))
    sources = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = getattr(node.func, 'id', None) or getattr(node.func, 'attr', None)
        if name != 'deployment_package':
            continue
        for keyword in node.keywords:
            if keyword.arg == 'source' and isinstance(keyword.value, ast.Dict):
                for key, item in zip(keyword.value.keys, keyword.value.values):
                    if getattr(key, 'value', None) == 'shared_new_init_sha256':
                        sources.append(item)
    assert len(sources) == 1, 'the deploy package must carry one initialization identity'
    item = sources[0]
    assert isinstance(item, ast.Subscript) and isinstance(item.value, ast.Name)
    assert item.value.id == 'shared_init' and getattr(item.slice, 'value', None) == 'sha256'


def test_the_denominator_reduction_touches_no_identity():
    tree = ast.parse(RUNNER.read_text(encoding='utf-8'))
    helper = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                  and node.name == 'reduce_update_denominators')
    bound = bound_names(helper)
    assert set(bound) == {'update_counts', 'global_sum', 'device', 'denominators',
                          'name', 'value', 'denominator_total'}, bound
    reads = {node.id for node in ast.walk(helper)
             if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)}
    assert 'shared' not in reads and 'shared_init' not in reads
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                and node.name == 'main')
    called = {getattr(node.func, 'id', None) or getattr(node.func, 'attr', None)
              for node in ast.walk(main) if isinstance(node, ast.Call)}
    assert 'reduce_update_denominators' in called, (
        'the tested helper must be the one the training entry calls')


# --------------------------------------------------------------------------- B


def test_progress_marker_counts_only_appended_stage_marks(tmp_path):
    stages = tmp_path / 'cat'
    stages.mkdir()
    stage_log = stages / 'stages_rank0.log'
    stage_log.write_text(MARK % 1, encoding='utf-8')
    stack = stages / 'stall_stack_rank0.txt'
    stack.write_text('dump\n', encoding='utf-8')
    first = supervisor.progress_marker(stages)
    stack.write_text('dump\ndump\n', encoding='utf-8')
    os.utime(stage_log)  # a bare mtime refresh is not progress either
    assert supervisor.progress_marker(stages) == first
    stage_log.write_text((MARK % 1) + (MARK % 2), encoding='utf-8')
    assert supervisor.progress_marker(stages) != first


def run_supervisor(stages, root_pid, report_path, log_path=None, **thresholds):
    command = [sys.executable, str(SUPERVISOR), '--root-pid', str(root_pid),
               '--stages-dir', str(stages), '--report', str(report_path)]
    if log_path is not None:
        command += ['--log', str(log_path)]
    for name, value in thresholds.items():
        command += ['--' + name.replace('_', '-'), str(value)]
    return subprocess.run(command, capture_output=True, text=True, timeout=60)


def test_growing_stack_dumps_and_logs_cannot_postpone_the_stop(tmp_path):
    """Fake growth is not progress: the stop still happens when the marks stop."""
    stages = tmp_path / 'cat'
    stages.mkdir()
    (stages / 'stages_rank0.log').write_text(MARK % 1, encoding='utf-8')
    stack = stages / 'stall_stack_rank0.txt'
    stack.write_text('dump\n', encoding='utf-8')
    run_log = tmp_path / 'run.log'
    run_log.write_text('launcher line\n', encoding='utf-8')
    child = subprocess.Popen([sys.executable, '-c', SLEEPER], cwd=str(tmp_path))
    bystander = subprocess.Popen([sys.executable, '-c', SLEEPER])
    noise = {'appends': 0}
    keep_writing = threading.Event()
    keep_writing.set()

    def write_noise():
        while keep_writing.is_set():
            with open(stack, 'a', encoding='utf-8') as handle:
                handle.write('dump\n')
            with open(run_log, 'a', encoding='utf-8') as handle:
                handle.write('noise\n')
            noise['appends'] += 2
            time.sleep(0.02)

    writer = threading.Thread(target=write_noise, daemon=True)
    writer.start()
    try:
        time.sleep(1.0)  # longer than the stop threshold below, with no real mark
        report_path = tmp_path / 'stall_supervisor.json'
        completed = run_supervisor(stages, child.pid, report_path, log_path=run_log,
                                   poll_seconds=0.1, stack_seconds=5, stop_seconds=0.8,
                                   grace_seconds=1)
        report = json.loads(report_path.read_text(encoding='utf-8'))
        # Read the neighbour's state here: the cleanup below terminates it on purpose.
        bystander_alive = bystander.poll() is None
    finally:
        keep_writing.clear()
        writer.join(timeout=5)
        bystander.terminate()
        bystander.wait(timeout=30)
    assert noise['appends'] > 10, 'the fixture must actually grow the fake logs'
    assert completed.returncode == 9, completed.stdout + completed.stderr
    assert report['intervened'] is True and report['status'] == 'STOPPED_BY_SUPERVISOR'
    assert report['silent_seconds_at_stop'] < 1.5, (
        'stack dumps or log growth postponed the stop: '
        f"silent_seconds_at_stop={report['silent_seconds_at_stop']}")
    assert child.poll() is not None, 'the stalled child is still running'
    assert bystander_alive, 'a process outside this tree was signalled'
    assert os.getpid() not in report['tree']['signalled_terminate']
    assert report['progress_at_stop'][0][1] == len(MARK % 1), (
        'the stage log never changed, so the marker must be the initial one')
    assert report['log'] == str(run_log), 'the log stays in the report, not in the timer'


def test_a_real_stage_mark_resets_the_silence_timer(tmp_path):
    """Marks keep the run alive; the stop lands one stop-window after the last one."""
    stages = tmp_path / 'cat'
    stages.mkdir()
    stage_log = stages / 'stages_rank0.log'
    stage_log.write_text(MARK % 1, encoding='utf-8')
    child = subprocess.Popen([sys.executable, '-c', SLEEPER], cwd=str(tmp_path))
    stop_seconds = 0.8
    marks = [1]
    alive_after_last_mark = []

    def append_marks():
        for step in range(2, 7):
            time.sleep(0.4)
            with open(stage_log, 'a', encoding='utf-8') as handle:
                handle.write(MARK % step)
            marks.append(step)
        # The child must still be running now: no mark may have been ignored and no
        # stop may have happened while marks were arriving.
        alive_after_last_mark.append(child.poll() is None)

    writer = threading.Thread(target=append_marks, daemon=True)
    writer.start()
    report_path = tmp_path / 'stall_supervisor.json'
    started = time.monotonic()
    completed = run_supervisor(stages, child.pid, report_path,
                               poll_seconds=0.1, stack_seconds=5,
                               stop_seconds=stop_seconds, grace_seconds=1)
    elapsed = time.monotonic() - started
    writer.join(timeout=10)
    report = json.loads(report_path.read_text(encoding='utf-8'))
    assert completed.returncode == 9, completed.stdout + completed.stderr
    assert alive_after_last_mark == [True], 'the run was stopped while marks were arriving'
    assert elapsed >= len(range(2, 7)) * 0.4, 'the monitored window was too short to matter'
    assert report['silent_seconds_at_stop'] >= stop_seconds - 0.05
    assert stage_log.read_text(encoding='utf-8').count('\n') == len(marks)
    assert child.poll() is not None


# --------------------------------------------------------------------------- D


def test_a_normal_exit_is_not_reported_as_a_stall(tmp_path):
    stages = tmp_path / 'cat'
    stages.mkdir()
    (stages / 'stages_rank0.log').write_text(MARK % 1, encoding='utf-8')
    child = subprocess.Popen([sys.executable, '-c', 'import time\ntime.sleep(0.7)'],
                             cwd=str(tmp_path))
    report_path = tmp_path / 'stall_supervisor.json'
    completed = run_supervisor(stages, child.pid, report_path,
                               poll_seconds=0.1, stack_seconds=5, stop_seconds=30)
    report = json.loads(report_path.read_text(encoding='utf-8'))
    child.wait(timeout=30)
    assert completed.returncode == 0
    assert report['status'] == 'ROOT_EXITED' and report['intervened'] is False
    assert 'tree' not in report and 'stopped_at_monotonic' not in report
    assert 'STOPPED_BY_SUPERVISOR' not in completed.stdout
    assert child.returncode == 0, 'the supervisor killed a process that exited by itself'


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-q']))
