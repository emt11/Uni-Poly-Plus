"""r4 zero-update fixture for the export forensics.

These tests never train: they exercise the stage log, the stall watchdog, the
memory record and the ordering that puts a failure on disk before any cleanup
can block.  The only process launches are the pre-training runner with missing
inputs (the runner fails while building its source), which consumes no data and
performs no forward, backward or optimizer step.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.pretrain_mcl_ph import StageLogger, memory_record  # noqa: E402


def test_stage_marks_are_flushed_immediately(tmp_path):
    stages = StageLogger(tmp_path, 3, stall_seconds=0)
    stages.mark('rng_gather', 'enter', step=2)
    stages.mark('rng_gather', 'complete', step=2)
    lines = (tmp_path / 'stages_rank3.log').read_text(encoding='utf-8').splitlines()
    assert [json.loads(line)['stage'] for line in lines] == ['rng_gather', 'rng_gather']
    first = json.loads(lines[0])
    assert first['rank'] == 3 and first['event'] == 'enter' and first['step'] == 2
    assert isinstance(first['pid'], int) and first['monotonic'] > 0
    # A process killed from outside cannot flush anything: the marks are already
    # on disk, which is the whole point of writing them here.
    stages.close()
    assert len((tmp_path / 'stages_rank3.log').read_text(encoding='utf-8').splitlines()) == 2


def test_tensor_progress_names_the_currently_running_copy(tmp_path):
    stages = StageLogger(tmp_path, 0, stall_seconds=0)
    stages.tensor_progress('tensor_start', 'encoder.o8.layers.0.weight')
    stages.tensor_progress('tensor_complete', 'encoder.o8.layers.0.weight')
    record = json.loads((tmp_path / 'stages_rank0.log').read_text(encoding='utf-8').splitlines()[0])
    assert record['stage'] == 'tensor:encoder.o8.layers.0.weight'
    assert record['event'] == 'tensor_start'
    stages.close()


def test_the_stall_watchdog_writes_this_ranks_stack(tmp_path):
    stages = StageLogger(tmp_path, 1, stall_seconds=1.0)
    stages.mark('resume_save', 'enter')
    time.sleep(2.5)
    stacks = (tmp_path / 'stall_stack_rank1.txt').read_text(encoding='utf-8')
    assert 'Thread' in stacks and 'test_the_stall_watchdog_writes_this_ranks_stack' in stacks
    assert list(tmp_path.glob('stall_stack_rank*')) == [tmp_path / 'stall_stack_rank1.txt']
    stages.mark('resume_save', 'complete')
    stages.close()


def test_memory_record_states_its_window_units_and_scope():
    record = memory_record()
    assert record['window'] == 'step' and record['units'] == 'bytes'
    for key in ('cuda_peak_allocated_bytes', 'cuda_peak_reserved_bytes'):
        value = record[key]
        assert value == 'NOT_MEASURED' or (isinstance(value, int) and value >= 0)
    assert record['cpu_peak_rss_bytes'] == 'NOT_MEASURED' or \
        isinstance(record['cpu_peak_rss_bytes'], int)
    # The CPU number covers this rank only; saying so is part of the record.
    assert 'dataloader_workers_excluded' in record['cpu_scope'] or \
        record['cpu_scope'] == 'NOT_MEASURED'


def _torchrun_prepare_failure(tmp_path, extra):
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
    command = [sys.executable, '-m', 'torch.distributed.run', '--standalone',
               '--nproc_per_node', '4', str(ROOT / 'scripts' / 'pretrain_mcl_ph.py'),
               '--config', str(config),
               '--cohort-root', str(ROOT / 'data/processed/glt_dual_v2/pi1m/cohort_30f17b59bc5862a1'),
               '--cache-root', str(ROOT / 'data/processed/mips_trimer_scage'),
               '--dual-static-root', str(ROOT / 'data/processed/glt_dual_v2/pi1m/dual_static_v1'),
               '--statistics', str(ROOT / 'results/mcl_ph_20260921/p0/statistics.npz'),
               '--shared-new-init', str(tmp_path / 'shared.pt')] + extra + \
        ['--output', str(output)]
    return subprocess.run(command, cwd=str(ROOT), capture_output=True, text=True, timeout=600)


def test_a_failure_inside_the_body_is_recorded_before_cleanup(tmp_path):
    """A failure past the setup must reach disk without waiting for the teardown.

    ``--sample-index-artifact`` points at a path that cannot exist, so the run
    fails inside the training body while the output directory already exists --
    the case where the record has to be written *before* ``source.close()`` and
    ``destroy_process_group()`` run.  The per-rank append-only log keeps that
    first record even though the outermost handler writes the same failure again.
    """
    completed = _torchrun_prepare_failure(
        tmp_path, ['--sample-index-artifact', str(tmp_path / 'missing-index.json')])
    output = tmp_path / 'out'
    assert completed.returncode == 1, completed.stdout + completed.stderr
    assert not (output / 'run.json').exists(), 'a failed run must not leave a run record'
    assert 'ALL_ARMS_OK' not in completed.stdout
    runtime = json.loads((output / 'runtime.json').read_text(encoding='utf-8'))
    assert runtime['status'] == 'FAILED'
    for rank in range(4):
        log = (output / f'failure_rank{rank}.log').read_text(encoding='utf-8').splitlines()
        records = [json.loads(line) for line in log]
        phases = [record['phase'] for record in records]
        assert 'before_cleanup' in phases, (rank, phases)
        assert phases[0] == 'before_cleanup', (rank, phases)
        for record in records:
            assert record['status'] == 'FAILED' and record['rank'] == rank
            assert record['error'].strip()
        rank_file = json.loads((output / f'runtime_failure_rank{rank}.json').read_text(
            encoding='utf-8'))
        assert rank_file['status'] == 'FAILED' and rank_file['rank'] == rank
