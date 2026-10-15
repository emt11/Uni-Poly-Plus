"""Final CPU simulations for exclusive continuation and bounded scheduling."""
import json
import fcntl
import pytest
from pathlib import Path
import scripts.run_fce_param_r4 as schedule
import scripts.run_fce_param as accounting


def test_slot_caps_unknown_pid_and_memory_boundaries():
    card=dict(free_mib=20000,apps=[dict(pid=100)])
    assert schedule.allowed_slots(2,{0},card,{100},4,40000)==[1,2,3]
    assert schedule.allowed_slots(3,set(range(8)),card,{100},8,40000)==[]
    assert schedule.allowed_slots(2,{0},card,{100},8,40000)==list(range(1,8))
    assert schedule.allowed_slots(0,set(),card,{100},8,40000)==[]
    assert schedule.allowed_slots(2,{0},card,{99},8,40000)==[]
    assert schedule.allowed_slots(2,{0},card,{100},8,32767)==[]
    assert schedule.allowed_slots(2,{0},dict(card,free_mib=12287),{100},8,40000)==[]
    assert schedule.allowed_slots(2,{0,1,2,3},card,{100},2,40000)==[]
    assert len(schedule.allowed_slots(2,set(),dict(card,apps=[]),set(),8,40000))+len(schedule.allowed_slots(3,set(),dict(card,apps=[]),set(),8,40000))==16


def test_upgrade_requires_normal_joint_observation_and_headroom():
    assert not schedule.may_upgrade(4,179.999,True,True,False,0)
    assert schedule.may_upgrade(4,180,True,True,False,0)
    assert schedule.may_upgrade(6,180,True,True,False,0)
    assert not schedule.may_upgrade(8,900,True,True,False,0)
    assert not schedule.may_upgrade(4,900,False,True,False,0)
    assert not schedule.may_upgrade(4,900,True,False,False,0)
    assert not schedule.may_upgrade(4,900,True,True,True,0)
    assert not schedule.may_upgrade(4,900,True,True,False,1)
    assert [schedule.lower_level(x) for x in (8,6,4,2,1)]==[6,4,2,1,1]


def test_success_inflight_and_failure_never_become_pending(tmp_path,monkeypatch):
    log=tmp_path/'logs';out=tmp_path/'out';monkeypatch.setattr(schedule,'LOG',log);monkeypatch.setattr(schedule,'OUT',out)
    assert schedule.unit_state('R0','eea',0)=='pending'
    folder=out/'finetune/R0/eea/fold0';folder.mkdir(parents=True)
    launch=log/'ft_R0_eea_0';launch.mkdir(parents=True)
    (launch/'pid.txt').write_text('123')
    monkeypatch.setattr(schedule,'process',lambda pid:dict(argv=['python','scripts/run_fce_param_unit.py'],cwd=str(schedule.ROOT)))
    assert schedule.unit_state('R0','eea',0)=='running'
    (launch/'exit.txt').write_text('1')
    assert schedule.unit_state('R0','eea',0)=='failed'
    (launch/'exit.txt').write_text('0');(launch/'execution.json').write_text(json.dumps(dict(exit_code=0)))
    (folder/'metrics.json').write_text(json.dumps(dict(status='PASS')))
    assert schedule.unit_state('R0','eea',0)=='inconsistent'
    for name in ('best.pt','scaler.json','progress.json','updates.json','validation_predictions.npz'):(folder/name).touch()
    assert schedule.unit_state('R0','eea',0)=='success'
    monkeypatch.setattr(schedule,'unit_state',lambda a,t,f,formal=False:'success' if f==0 else 'failed' if f==1 else 'pending')
    pending,failed=schedule.planned_queue(['R0'])
    assert len(pending)==21 and len(failed)==7
    assert all(x[2]>=2 for x in pending)


def test_existing_freeze_not_repeated_and_budget_not_reset(tmp_path,monkeypatch):
    out=tmp_path/'out';log=tmp_path/'logs';out.mkdir();log.mkdir()
    monkeypatch.setattr(schedule,'OUT',out);monkeypatch.setattr(accounting,'OUT',out);monkeypatch.setattr(accounting,'LOG',log)
    (out/'selection_before_test.json').write_text(json.dumps(dict(validation_units=140,frozen=True,candidate='B_LR')))
    def forbidden():raise AssertionError('duplicate freeze')
    assert schedule.frozen_candidate(forbidden)=='B_LR'
    folder=out/'finetune/R0/eea/fold0';folder.mkdir(parents=True)
    (folder/'updates.json').write_text(json.dumps(dict(attempted=700,completed=699)))
    (folder/'progress.json').write_text(json.dumps(dict(history=[{}]*70)))
    (folder/'failure.json').write_text('{}')
    command=log/'ft_R0_eea_0';command.mkdir();(command/'command.json').write_text('{}')
    c=accounting.consumption()
    assert (c['ft_starts'],c['ft_attempted_updates'],c['ft_completed_updates'],c['ft_epochs'],c['ft_failed'])==(1,700,699,70,1)
    (folder/'updates.json').write_text(json.dumps(dict(attempted=205801,completed=699)))
    with pytest.raises(RuntimeError):accounting.consumption()


def test_exclusive_file_lock(tmp_path):
    with (tmp_path/'lock').open('a+') as first,(tmp_path/'lock').open('a+') as second:
        fcntl.flock(first,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):fcntl.flock(second,fcntl.LOCK_EX|fcntl.LOCK_NB)


def test_unmapped_nvml_pid_stays_unknown_but_owned_workload_can_share(monkeypatch,tmp_path):
    card=dict(free_mib=30000,apps=[dict(pid=700)])
    proof=dict(namespace_pids=[1700],pid=1700,gpu=2,start_ticks=999)
    monkeypatch.setattr(schedule,'unit_identity',lambda *a,**k:proof)
    running={(2,0):(tmp_path,('R0','eea',0))}
    identity=schedule.ownership(2,card,running)
    assert identity['admitted'] and identity['allow_unmapped']
    assert identity['unmatched_nvml_pids']==[700]
    assert not identity['nvml_mapping_verified']
    assert identity['known_namespace_pids']==[1700]  # never fabricate 1700 -> 700
    assert schedule.allowed_slots(2,{0},card,{1700},4,40000,allow_unmapped=True)==[1,2,3]
    # Unknown occupancy never permits taking a busy card without an owned unit.
    assert not schedule.ownership(3,card,running)['admitted']
    assert schedule.allowed_slots(3,set(),card,set(),4,40000,allow_unmapped=True)==[]
    assert schedule.allowed_slots(2,{0},dict(card,free_mib=10000),{1700},4,40000,allow_unmapped=True)==[]
    monkeypatch.setattr(schedule,'unit_identity',lambda *a,**k:None)
    assert not schedule.ownership(2,card,running)['admitted']


def test_one_unverified_local_unit_blocks_admission(monkeypatch,tmp_path):
    proofs=iter([dict(namespace_pids=[100]),None])
    monkeypatch.setattr(schedule,'unit_identity',lambda *a,**k:next(proofs))
    state=schedule.ownership(2,dict(apps=[dict(pid=700)]),
        {(2,0):(tmp_path/'a',('R0','eea',0)),(2,1):(tmp_path/'b',('R0','eea',1))})
    assert not state['admitted'] and not state['allow_unmapped']


def test_local_identity_requires_exact_launch_gpu_and_open_logs(tmp_path,monkeypatch):
    folder=tmp_path/'ft_R0_eea_0';folder.mkdir()
    proc=tmp_path/'proc/123';(proc/'fd').mkdir(parents=True)
    argv=[schedule.PYTHON,'scripts/run_fce_param_unit.py','--arm','R0','--task','eea','--fold','0']
    command=dict(argv=argv,gpu=2,cwd=str(schedule.ROOT),environment={'CUDA_VISIBLE_DEVICES':'2'})
    (folder/'command.json').write_text(json.dumps(command));(folder/'pid.txt').write_text('123')
    info=dict(pid=123,namespace_pids=[123],argv=argv,cwd=str(schedule.ROOT),state='S')
    monkeypatch.setattr(schedule,'process',lambda pid:dict(info))
    for fd,name in [('1','stdout.log'),('2','stderr.log')]:
        (folder/name).touch();(proc/'fd'/fd).symlink_to(folder/name)
    (proc/'environ').write_bytes(b'CUDA_VISIBLE_DEVICES=2\0')
    (proc/'stat').write_text('123 (python) '+' '.join(['S']+['0']*18+['9876']))
    def check():return schedule.unit_identity(folder,('R0','eea',0),2,proc_root=tmp_path/'proc')
    assert check()['start_ticks']==9876
    (proc/'environ').write_bytes(b'CUDA_VISIBLE_DEVICES=3\0')
    assert check() is None
    (proc/'environ').write_bytes(b'CUDA_VISIBLE_DEVICES=2\0')
    info['argv']=argv[:-1]+['1'];assert check() is None
    info['argv']=argv;info['cwd']='/tmp';assert check() is None
    info['cwd']=str(schedule.ROOT)
    (proc/'fd/1').unlink();(proc/'fd/1').symlink_to(folder/'stderr.log')
    assert check() is None
