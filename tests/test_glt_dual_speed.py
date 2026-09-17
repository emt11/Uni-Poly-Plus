"""Bounded speed-path tests: process-local clean cache and dynamic shard slots."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch_geometric.data import Data

from scripts.run_glt_dual_finetune_grid import _run_dynamic_jobs
from src.training import glt_dual_runtime as runtime


class _FakeSource:
    samples = [(b'a' * 32, '*C*'), (b'b' * 32, '*C*')]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return object(), object(), self.samples[index][1]

    def static_for(self, index):
        return None


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
