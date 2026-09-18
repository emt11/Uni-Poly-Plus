"""Bounded speed-path tests: process-local clean cache and dynamic shard slots."""

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Data

from scripts.run_glt_dual_finetune_grid import (
    _build_shard_command, _run_batched_jobs, _run_dynamic_jobs, _validate_gpu_slots,
)
from scripts.pretrain_glt_dual import _diagnostic_update, _time_summary
from src.training import glt_dual_runtime as runtime


class _FakeSource:
    samples = [(b'a' * 32, '*C*'), (b'b' * 32, '*C*')]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index][0], object(), self.samples[index][1]

    def static_for(self, index):
        return None


class _DuplicateSource:
    samples = [(b'same-key' * 4, '*C*'), (b'same-key' * 4, '*C*')]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return object(), object(), self.samples[index][1]

    def static_for(self, index):
        return None


class _IndexDataset(Dataset):
    def __len__(self):
        return 8

    def __getitem__(self, index):
        return index


def _grid_args(clean_cache_gib):
    return SimpleNamespace(
        config='config.json', checkpoint='checkpoint.pt', raw_root='raw',
        cohort_root='cohort', cache_root='cache', dual_static_root='static',
        clean_cache_gib=clean_cache_gib, split_root='split',
    )


def test_worker_iterator_then_restore_preserves_parent_torch_rng():
    torch.manual_seed(9271)
    saved = runtime.rng_state()
    reference = torch.rand(16)
    runtime.restore_rng(saved)
    loader = DataLoader(_IndexDataset(), batch_size=1, num_workers=2)
    iterator = iter(loader)
    next(iterator)
    runtime.restore_rng(saved)
    observed = torch.rand(16)
    assert torch.equal(observed, reference)
    del iterator, loader


def test_grid_rejects_duplicate_gpu_slots_before_launch():
    with pytest.raises(ValueError, match='distinct GPU slots'):
        _validate_gpu_slots(['1', '1'])
    assert _validate_gpu_slots(['1', '2']) == ['1', '2']


def test_grid_cli_rejects_duplicate_gpu_before_launch(tmp_path):
    output = tmp_path / 'grid'
    command = [
        sys.executable, 'scripts/run_glt_dual_finetune_grid.py',
        '--config', 'config.json', '--checkpoint', 'checkpoint.pt',
        '--raw-root', str(tmp_path), '--cohort-root', str(tmp_path),
        '--cache-root', str(tmp_path), '--dual-static-root', str(tmp_path),
        '--output', str(output), '--log-root', str(tmp_path / 'logs'),
        '--gpu', '1', '--gpu', '1',
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.returncode != 0
    assert 'distinct GPU slots' in result.stderr
    assert not output.exists()


def test_grid_forwards_clean_cache_capacity():
    for capacity, expected in ((0, '0'), (4, '4')):
        command = _build_shard_command(
            _grid_args(capacity), 'xc', 0, Path('out/xc_fold0'))
        marker = command.index('--clean-cache-gib')
        assert command[marker + 1] == expected


def test_grid_smoke_command_uses_validation_only_mode():
    args = _grid_args(4)
    args.smoke = True
    command = _build_shard_command(args, 'eat', 1, Path('out/eat_fold1'))
    assert '--smoke' in command
    assert '--formal-shard' not in command
    assert command[command.index('--clean-cache-gib') + 1] == '4'


def test_batched_scheduler_waits_for_each_batch(tmp_path):
    launch_order = []

    def launch(task, fold, gpu):
        launch_order.append((task, fold, gpu))
        handle = (tmp_path / f'{task}_{fold}.log').open('w', encoding='utf-8')
        process = subprocess.Popen([sys.executable, '-c', 'pass'], stdout=handle,
                                   stderr=subprocess.STDOUT)
        return process, handle, tmp_path / f'{task}_{fold}.log'

    completed = _run_batched_jobs([('task', 0), ('task', 1), ('task', 2)], ['0', '1'], launch)
    assert [(row['task'], row['fold']) for row in completed] == [('task', 0), ('task', 1), ('task', 2)]
    assert launch_order == [('task', 0, '0'), ('task', 1, '1'), ('task', 2, '0')]


def test_diagnostic_sampling_is_first_interval_and_explicit_save_steps():
    selected = [step for step in range(1, 26)
                if _diagnostic_update(step, 0, 20, [7, 23])]
    assert selected == [1, 7, 20, 23]
    resumed = [step for step in range(11, 31)
               if _diagnostic_update(step, 10, 20, [])]
    assert resumed == [11, 20]


def test_time_summary_is_finite_and_empty_is_explicit():
    summary = _time_summary([0.2, 0.1, 0.3])
    assert summary['count'] == 3
    assert summary['median_seconds'] == 0.2
    assert all(np.isfinite(summary[key]) for key in ('mean_seconds', 'median_seconds', 'p95_seconds'))
    empty = _time_summary([])
    assert empty == {'count': 0, 'mean_seconds': None, 'median_seconds': None, 'p95_seconds': None}


def test_clean_dataset_duplicate_key_keeps_row_labels(monkeypatch):
    calls = []

    def fake_build(*record, static=None):
        calls.append(record[2])
        return Data(x=torch.tensor([3.0]))

    monkeypatch.setattr(runtime, 'build_dual_sample', fake_build)
    dataset = runtime.CleanLabeledDataset(
        _DuplicateSource(), np.asarray([1.0, 2.0]), cache_capacity_bytes=1024,
    )
    row0, row1 = dataset[0], dataset[1]
    assert len(calls) == 1
    assert row0.y.tolist() == [1.0] and row1.y.tolist() == [2.0]
    dataset.set_target_override([11.0, 22.0])
    assert dataset[0].y.tolist() == [11.0]
    assert dataset[1].y.tolist() == [22.0]
    assert dataset.cache_stats()['hits'] == 3


def test_clean_dataset_bounded_eviction_rebuilds_correct_data(monkeypatch):
    calls = []

    def fake_build(*record, static=None):
        calls.append(record[2])
        return Data(x=torch.tensor([float(record[0][0])]))

    monkeypatch.setattr(runtime, 'build_dual_sample', fake_build)
    source = _FakeSource()
    cached = runtime.CleanLabeledDataset(
        source, np.asarray([1.0, 2.0]), cache_capacity_bytes=4,
    )
    first = cached[0]
    cached[1]
    rebuilt = cached[0]
    uncached = runtime.CleanLabeledDataset(source, np.asarray([1.0, 2.0]))[0]
    assert first.x.tolist() == [97.0]
    assert rebuilt.x.tolist() == [97.0]
    assert rebuilt.x.tolist() == uncached.x.tolist()
    stats = cached.cache_stats()
    assert stats['evictions'] == 2 and stats['entries'] == 1
    assert len(calls) == 4


def test_clean_dataset_cache_clones_and_keeps_target_override(monkeypatch):
    calls = []

    def fake_build(*record, static=None):
        calls.append(record[2])
        return Data(x=torch.tensor([float(len(calls))]))

    monkeypatch.setattr(runtime, 'build_dual_sample', fake_build)
    dataset = runtime.CleanLabeledDataset(
        _FakeSource(), np.asarray([1.0, 2.0]), cache_capacity_bytes=1024,
    )
    first = dataset[0]
    first.x[0] = 99
    dataset.set_target_override([10.0, 20.0])
    replay = dataset[0]
    assert calls == ['*C*']
    assert replay.x.tolist() == [1.0]
    assert replay.y.tolist() == [10.0]
    stats = dataset.cache_stats()
    assert stats['hits'] == 1 and stats['misses'] == 1
    assert stats['entries'] == 1 and stats['payload_bytes'] > 0


def _timed_child(start_path, done_path, duration):
    code = (
        'from pathlib import Path; import time; '
        f'Path({str(start_path)!r}).write_text(str(time.monotonic())); '
        f'time.sleep({float(duration)!r}); '
        f'Path({str(done_path)!r}).write_text(str(time.monotonic()))'
    )
    return [sys.executable, '-c', code]


def _status_child(start_path, done_path, duration, exit_code=0):
    code = (
        'from pathlib import Path; import time; '
        f'Path({str(start_path)!r}).write_text(str(time.monotonic())); '
        f'time.sleep({float(duration)!r}); '
        f'Path({str(done_path)!r}).write_text(str(time.monotonic())); '
        f'raise SystemExit({int(exit_code)})'
    )
    return [sys.executable, '-c', code]


def test_dynamic_grid_dispatches_free_slot_before_long_slot_finishes(tmp_path):
    durations = {0: 0.35, 1: 0.05, 2: 0.05}

    def launch(task, fold, gpu):
        del gpu
        start_path = tmp_path / f'{task}_{fold}.start'
        done_path = tmp_path / f'{task}_{fold}.done'
        handle = (tmp_path / f'{task}_{fold}.log').open('w', encoding='utf-8')
        process = subprocess.Popen(
            _timed_child(start_path, done_path, durations[fold]),
            stdout=handle, stderr=subprocess.STDOUT,
        )
        return process, handle, tmp_path / f'{task}_{fold}.log'

    completed = _run_dynamic_jobs(
        [('task', 0), ('task', 1), ('task', 2)], ['0', '1'], launch,
        poll_interval=0.01,
    )
    assert [(row['task'], row['fold']) for row in completed] == [
        ('task', 1), ('task', 2), ('task', 0),
    ]
    second_start = float((tmp_path / 'task_2.start').read_text())
    first_done = float((tmp_path / 'task_0.done').read_text())
    assert second_start < first_done


def test_dynamic_grid_stops_dispatch_after_failure(tmp_path):
    def launch(task, fold, gpu):
        del gpu
        start_path = tmp_path / f'{task}_{fold}.start'
        done_path = tmp_path / f'{task}_{fold}.done'
        handle = (tmp_path / f'{task}_{fold}.log').open('w', encoding='utf-8')
        exit_code = 7 if fold == 0 else 0
        duration = 0.0 if fold == 0 else 0.20
        process = subprocess.Popen(
            _status_child(start_path, done_path, duration, exit_code),
            stdout=handle, stderr=subprocess.STDOUT,
        )
        return process, handle, tmp_path / f'{task}_{fold}.log'

    try:
        _run_dynamic_jobs(
            [('task', 0), ('task', 1), ('task', 2)], ['0', '1'], launch,
            poll_interval=0.01,
        )
    except RuntimeError as exc:
        message = str(exc)
    else:
        raise AssertionError('failed shard must stop the dynamic grid')
    assert 'exit_code' in message and '7' in message
    assert (tmp_path / 'task_1.start').is_file()
    assert (tmp_path / 'task_1.done').is_file()
    assert not (tmp_path / 'task_2.start').exists()


def test_dynamic_grid_drains_owned_children_when_launch_raises(tmp_path):
    def launch(task, fold, gpu):
        del gpu
        if fold == 1:
            raise RuntimeError('synthetic launch failure')
        start_path = tmp_path / f'{task}_{fold}.start'
        done_path = tmp_path / f'{task}_{fold}.done'
        handle = (tmp_path / f'{task}_{fold}.log').open('w', encoding='utf-8')
        process = subprocess.Popen(
            _status_child(start_path, done_path, 0.05),
            stdout=handle, stderr=subprocess.STDOUT,
        )
        return process, handle, tmp_path / f'{task}_{fold}.log'

    with pytest.raises(RuntimeError, match='synthetic launch failure'):
        _run_dynamic_jobs([('task', 0), ('task', 1)], ['0', '1'], launch)
    assert (tmp_path / 'task_0.done').is_file()
    assert 'EXIT_CODE=0' in (tmp_path / 'task_0.log').read_text()
