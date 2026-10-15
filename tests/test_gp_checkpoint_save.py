"""Bounded four-rank CPU save scenes; no model or optimizer."""
import json
import os
import stat
import time
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import src.training.gp_checkpoint_save as saving

SCENARIOS = ['normal', 'slow', 'resume_fail', 'deploy_fail', 'guard',
             'metadata_A', 'metadata_B', 'metadata_C', 'precommit_read', 'precommit_stat',
             'initial_gate_stat', 'initial_metadata_stat']


def worker(rank, folder, scenario):
    folder = Path(folder)
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method='file://' + str(folder / 'rendezvous'),
                            rank=rank, world_size=4, timeout=timedelta(seconds=20))
    events = []
    active = None
    injected = False
    original_metadata = saving.write_metadata
    original_fsync = os.fsync
    original_verify = saving._verified_payload
    original_read = Path.read_text
    original_stat = Path.stat
    original_unlink = Path.unlink
    verifying = False
    published = False

    def verify(*args):
        nonlocal verifying
        assert (folder / 'save_uncommitted_00007.json').exists()
        events.append(dict(event='verification_start_with_gate', at=time.time()))
        verifying = True
        try:
            return original_verify(*args)
        finally:
            verifying = False

    def read_text(path, *args, **kwargs):
        nonlocal injected
        assert not published, 'filesystem read after success publication'
        if verifying and scenario == 'precommit_read' and not injected:
            injected = True
            fault('precommit metadata read')
        return original_read(path, *args, **kwargs)

    def path_stat(path, *args, **kwargs):
        nonlocal injected
        assert not published, 'filesystem stat after success publication'
        initial_name = 'save_uncommitted_00007.json' if scenario == 'initial_gate_stat' else 'save_result_00007.json'
        if scenario.startswith('initial_') and path.name == initial_name and not injected:
            injected = True
            fault('initial collision stat: ' + path.name)
        if verifying and scenario == 'precommit_stat' and path.name == 'resume_00007.pt' and not injected:
            injected = True
            fault('precommit checkpoint stat')
        return original_stat(path, *args, **kwargs)

    def unlink(path, *args, **kwargs):
        nonlocal published
        value = original_unlink(path, *args, **kwargs)
        if path.name == 'save_uncommitted_00007.json':
            published = True
            events.append(dict(event='success_published', at=time.time()))
        return value

    def fault(location):
        events.append(dict(event='injected_exception', location=location,
                           error='OSError: injected ' + location, at=time.time()))
        raise OSError('injected ' + location)

    def metadata(path, payload):
        nonlocal active, injected
        active = Path(path).name
        events.append(dict(event='metadata_start', path=active, at=time.time()))
        try:
            if scenario == 'metadata_B' and active == 'failure.json' and not injected:
                injected = True
                fault('failure.json commit after save_result success')
            if scenario == 'metadata_C' and injected and active in ('save_result_00007.json', 'failure.json'):
                fault('FAILED correction commit')
            original_metadata(path, payload)
            events.append(dict(event='metadata_done', path=active, at=time.time()))
        finally:
            active = None

    def fsync(descriptor):
        nonlocal injected
        if scenario in ('metadata_A', 'metadata_C') and not injected and active == 'save_result_00007.json' and stat.S_ISDIR(os.fstat(descriptor).st_mode):
            assert (folder / active).exists()  # os.replace really happened
            injected = True
            fault('save_result directory fsync after os.replace')
        return original_fsync(descriptor)

    if rank == 0:
        saving.write_metadata = metadata
        os.fsync = fsync
        saving._verified_payload = original_verify if scenario.startswith('initial_') else verify
        Path.read_text = read_text
        Path.stat = path_stat
        Path.unlink = unlink

    def write(path, kind):
        assert rank == 0
        events.append(dict(event=kind + '_start', at=time.time()))
        if scenario == 'slow' and kind == 'resume':
            time.sleep(1.5)
        temporary = path.with_suffix('.pt.tmp')
        if scenario == kind + '_fail':
            temporary.touch()
            fault(kind + ' write failure')
        torch.save(dict(step=7, parameter=torch.tensor([1., 2.]), kind=kind), temporary)
        os.replace(temporary, path)
        events.append(dict(event=kind + '_done', at=time.time()))

    stop = dict(step=7, window_rates=[0., 1., 1.], consecutive_low_windows=[3, 0, 0], threshold=.01) if scenario in ('guard', 'metadata_B') else None
    error = None
    simulated_updates = 0
    entered = time.time()
    try:
        result = saving.synchronized_save(folder, 7, lambda p: write(p, 'resume'), lambda p: write(p, 'deploy'), stop=stop)
        # Actual control-flow statement after save, unreachable on any failure/guard.
        simulated_updates += 1
        (folder / f'update_rank{rank}.txt').write_text(str(simulated_updates))
    except saving.CoordinatedCheckpointExit as caught:
        error = str(caught)
        result = json.loads(error)
    finally:
        if rank == 0:
            saving.write_metadata = original_metadata
            os.fsync = original_fsync
            saving._verified_payload = original_verify
            Path.read_text = original_read
            Path.stat = original_stat
            Path.unlink = original_unlink
    ended = time.time()
    disk = saving.read_save_result(folder, 7)
    (folder / f'rank{rank}.json').write_text(json.dumps(dict(rank=rank, entered=entered, ended=ended,
        result=result, disk_result=disk, error=error, events=events, simulated_updates=simulated_updates)))
    dist.destroy_process_group()


@pytest.mark.parametrize('scenario', SCENARIOS)
def test_four_rank_save(scenario):
    folder = Path(os.environ['GP_SAVE_TEST_ROOT']) / scenario
    folder.mkdir(parents=True, exist_ok=False)
    prior = dict(step=7, status='FAILED', complete=False, stop=None,
                 metadata_protocol='gp-save-gated-v1', error='prior attempt must stay unchanged')
    if scenario == 'initial_metadata_stat':
        (folder / 'save_result_00007.json').write_text(json.dumps(prior))
    context = mp.spawn(worker, args=(str(folder), scenario), nprocs=4, join=False)
    deadline = time.monotonic() + 45
    while not context.join(timeout=1):
        if time.monotonic() > deadline:
            for process in context.processes:
                if process.is_alive():
                    process.terminate()
            for process in context.processes:
                process.join(timeout=5)
            pytest.fail('four-rank save exceeded 45 second bound')
    assert all(not p.is_alive() and p.exitcode == 0 for p in context.processes)
    rows = [json.loads((folder / f'rank{rank}.json').read_text()) for rank in range(4)]
    result = rows[0]['result']
    assert all(row['result'] == result for row in rows)
    disk = saving.read_save_result(folder, 7)
    assert all(row['disk_result'] == disk == result['disk_result'] for row in rows)
    normal = scenario in ('normal', 'slow')
    guard = scenario == 'guard'
    assert result['status'] == ('PASS' if normal else 'STOPPED_ROUTER_GUARD' if guard else 'FAILED')
    assert disk['success_consumable'] == normal
    assert result['metadata_commit'] == ('COMMITTED' if normal or guard else 'UNCOMMITTED_UNVERIFIABLE')
    assert all(row['simulated_updates'] == int(normal) for row in rows)
    assert all((folder / f'update_rank{i}.txt').exists() == normal for i in range(4))
    assert all(bool(row['error']) == (not normal) for row in rows)
    if normal or guard:
        assert not (folder / 'save_uncommitted_00007.json').exists()
        meta = json.loads((folder / 'save_result_00007.json').read_text())
        assert meta == disk['observed_result']
        assert all(row['ended'] >= (folder / 'save_result_00007.json').stat().st_mtime for row in rows)
        if guard:
            assert all(row['ended'] >= (folder / 'failure.json').stat().st_mtime for row in rows)
    else:
        assert (folder / 'save_uncommitted_00007.json').exists() == (not scenario.startswith('initial_'))
        assert disk['status'] == 'UNCOMMITTED_UNVERIFIABLE'
    for kind in ('resume', 'deploy'):
        path = folder / f'{kind}_00007.pt'
        details = result['files'][path.name]
        assert details == dict(exists=path.exists(), bytes=path.stat().st_size if path.exists() else None)
        if path.exists():
            saved = torch.load(path, map_location='cpu', weights_only=True)
            assert saved['step'] == 7 and saved['kind'] == kind
            assert torch.equal(saved['parameter'], torch.tensor([1., 2.]))
    if scenario == 'slow':
        times = {v['event']:v['at'] for v in rows[0]['events']}
        assert times['resume_done'] - times['resume_start'] >= 1.5
        assert all(row['ended'] >= times['deploy_done'] for row in rows)
    if scenario.startswith('metadata_'):
        assert result['metadata_error']
        faults = [v for v in rows[0]['events'] if v['event'] == 'injected_exception']
        assert len(faults) == (2 if scenario == 'metadata_C' else 1)
        if scenario == 'metadata_C':
            assert result['correction_error']
            # Stale raw PASS survived failed correction, yet cannot be consumed.
            assert json.loads((folder / 'save_result_00007.json').read_text())['status'] == 'PASS'
            assert not disk['success_consumable']
        else:
            assert json.loads((folder / 'save_result_00007.json').read_text())['status'] == 'FAILED'
    if scenario.startswith('precommit_'):
        assert result['metadata_error']
        faults = [v for v in rows[0]['events'] if v['event'] == 'injected_exception']
        assert len(faults) == 1 and not any(v['event'] == 'success_published' for v in rows[0]['events'])
        assert json.loads((folder / 'save_result_00007.json').read_text())['status'] == 'FAILED'
    if scenario.startswith('initial_'):
        assert 'injected initial collision stat' in result['error']
        assert len([v for v in rows[0]['events'] if v['event'] == 'injected_exception']) == 1
        assert not any(v['event'] in ('resume_start','deploy_start','metadata_start') for v in rows[0]['events'])
        if scenario == 'initial_metadata_stat':
            assert json.loads((folder / 'save_result_00007.json').read_text()) == prior
        else:
            assert not (folder / 'save_result_00007.json').exists()
    if normal or guard:
        events = rows[0]['events']
        assert next(v['at'] for v in events if v['event'] == 'verification_start_with_gate') <= next(v['at'] for v in events if v['event'] == 'success_published')
    if scenario.endswith('_fail'):
        assert not (folder / 'deploy_00007.pt').exists()
        assert (folder / (scenario.removesuffix('_fail') + '_00007.pt.tmp')).exists()
    (folder / 'assertions.json').write_text(json.dumps(dict(status='PASS_CPU_SCENARIO', scene=scenario,
        ranks=4, child_exitcodes=[p.exitcode for p in context.processes], success_consumable=disk['success_consumable'],
        actual_simulated_updates=[row['simulated_updates'] for row in rows], disk_result=disk), indent=2))
