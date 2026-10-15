from argparse import Namespace
from concurrent.futures import ThreadPoolExecutor
import json
import time

import pytest

from scripts import build_mcl_ph_full_geometry as build
from scripts import handoff_mcl_ph_parallel_retry as handoff


def test_retry_drains_out_of_order_failures_and_skips_completed(tmp_path, monkeypatch):
    output = tmp_path / 'retry.staging'
    output.mkdir()
    db = build.ledger_connect(output / 'rows.sqlite')
    rows = [(i, 'smiles', f'{i:064x}') for i in range(4)]
    for i, _, key in rows:
        retry = 'TIMEOUT' if i == 3 else None
        db.execute('INSERT INTO rows VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
                   (i, key, 'TIMEOUT', 60, '{}', retry, 60 if retry else None,
                    '{}' if retry else None, 'TIMEOUT', None))
    db.commit()
    monkeypatch.setattr(build, 'source_rows', lambda *args: iter(rows))
    monkeypatch.setattr(build, 'ProcessPoolExecutor',
                        lambda max_workers, initializer: ThreadPoolExecutor(max_workers=max_workers))

    def compute(row):
        time.sleep(0.03 if row[0] == 0 else 0.001)
        return {'row': row, 'status': 'GENERATOR_ERROR' if row[0] == 1 else 'TIMEOUT',
                'seconds': 0.01, 'failure': {'reason': 'synthetic regression fixture'}}

    monkeypatch.setattr(build, 'compute', compute)
    writer = Namespace(sync=lambda: None)
    args = Namespace(kind='downstream', limit=4, retry_workers=2, retry_progress_every=1)
    build.parallel_retry(args, output, db, writer, None)
    results = db.execute('SELECT source_index, retry_status, retry_detail FROM rows').fetchall()
    assert [row[1] for row in results] == ['TIMEOUT', 'GENERATOR_ERROR', 'TIMEOUT', 'TIMEOUT']
    assert all(json.loads(row[2])['retry_workers'] == 2 for row in results[:3])
    assert results[3][2] == '{}'
    progress = json.loads((output / 'retry_progress.json').read_text())
    assert progress['done'] == progress['target'] == 4
    assert progress['completed_this_run'] == 3
    db.close()


def test_retry_rejects_source_identity_mismatch(tmp_path, monkeypatch):
    output = tmp_path / 'retry.staging'
    output.mkdir()
    db = build.ledger_connect(output / 'rows.sqlite')
    db.execute("INSERT INTO rows VALUES (0, 'original', 'TIMEOUT', 60, '{}', NULL, NULL, NULL, 'TIMEOUT', NULL)")
    db.commit()
    monkeypatch.setattr(build, 'source_rows', lambda *args: iter([(0, 'smiles', 'wrong')]))
    monkeypatch.setattr(build, 'ProcessPoolExecutor',
                        lambda max_workers, initializer: ThreadPoolExecutor(max_workers=max_workers))
    args = Namespace(kind='downstream', limit=1, retry_workers=2, retry_progress_every=1)
    with pytest.raises(ValueError, match='identity differs'):
        build.parallel_retry(args, output, db, Namespace(sync=lambda: None), None)
    assert db.execute('SELECT retry_status FROM rows').fetchone()[0] is None
    db.close()


def test_handoff_signals_only_after_durable_main_completion(tmp_path, monkeypatch):
    exit_path = tmp_path / 'exit.code'
    state = {'phase': 'main', 'done': 3, 'target': 4}
    signals = []
    monkeypatch.setattr(handoff, 'read_progress', lambda _: dict(state))
    command = ['python', 'scripts/build_mcl_ph_full_geometry.py', '--output', str(tmp_path)]
    monkeypatch.setattr(handoff, 'process_identity', lambda _: (42, command))
    monkeypatch.setattr(handoff.os, 'kill', lambda pid, signum: signals.append((pid, signum)))

    def advance(_):
        if state['phase'] == 'main':
            assert not signals
            state.update(phase='main_complete', done=4)
        else:
            assert len(signals) == 1
            exit_path.write_text('130\n')

    monkeypatch.setattr(handoff.time, 'sleep', advance)
    result = handoff.wait_and_stop_main(123, 42, tmp_path, exit_path, lambda *args, **kwargs: None)
    assert result['interrupted_for_handoff'] and result['legacy_exit_code'] == 130


def test_handoff_refuses_reused_pid(tmp_path, monkeypatch):
    monkeypatch.setattr(handoff, 'read_progress', lambda _: {'phase': 'main_complete', 'done': 4, 'target': 4})
    monkeypatch.setattr(handoff, 'process_identity', lambda _: (43, []))
    monkeypatch.setattr(handoff.os, 'kill', lambda *args: pytest.fail('must not signal a reused PID'))
    with pytest.raises(RuntimeError, match='reused'):
        handoff.wait_and_stop_main(123, 42, tmp_path, tmp_path / 'exit.code', lambda *args, **kwargs: None)
