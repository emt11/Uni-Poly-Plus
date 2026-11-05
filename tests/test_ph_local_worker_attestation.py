"""Resource-ownership contracts only: no torch, GPU, model or real data."""
import copy
import json
from pathlib import Path
import pytest
from scripts.ph_local_worker_attestation import attest_worker


@pytest.fixture
def fixture(tmp_path):
    proc=tmp_path/'proc'; p=proc/'22'; (p/'fd').mkdir(parents=True)
    cwd=tmp_path/'workspace';cwd.mkdir();log=tmp_path/'logs';log.mkdir();unit=tmp_path/'unit';unit.mkdir()
    for f in ('stdout.log','stderr.log'):(log/f).touch()
    (p/'fd/1').symlink_to(log/'stdout.log');(p/'fd/2').symlink_to(log/'stderr.log');(p/'cwd').symlink_to(cwd)
    fields=['S']+['0']*18+['123']+['0']*5;(p/'stat').write_text('22 (python) '+' '.join(fields))
    argv=['/opt/conda/bin/python','-u','scripts/run_ua3d_ph_opt_r2.py','ft','--arm','P_PE','--task','eea','--fold','0']
    (p/'cmdline').write_bytes(('\0'.join(argv)+'\0').encode())
    env=dict(CUDA_VISIBLE_DEVICES='2',WORLD_SIZE='1',UA3M_PHYSICAL_GPU_UUID='GPU-test')
    (p/'environ').write_bytes(('\0'.join(k+'='+v for k,v in env.items())+'\0').encode())
    (proc/'21/task/21').mkdir(parents=True);(proc/'21/task/21/children').write_text('22')
    (unit/'consumption.json').write_text(json.dumps(dict(process=dict(pid=22,start_ticks=123))))
    (unit/'environment.json').write_text(json.dumps(dict(physical=dict(actual_uuid='GPU-test'))))
    (log/'process.json').write_text(json.dumps(dict(child=dict(pid=21,start_ticks=99))))
    j=dict(argv=['timeout','100s']+argv,env=env,cwd=str(cwd),log=str(log),consumption=str(unit/'consumption.json'))
    return proc,p,j


def test_positive_preserves_unknown_host_pid(fixture):
    proc,p,j=fixture;result=attest_worker(j,proc)
    assert result['attested'] and result['host_pid_mapping']=='UNKNOWN' and 'host_pid' not in result


@pytest.mark.parametrize('kind',['argv','cwd','environment','logs','start','lineage'])
def test_mismatch_fails_closed(fixture,kind):
    proc,p,j=fixture;j=copy.deepcopy(j)
    if kind=='argv':j['argv'][-1]='4'
    elif kind=='cwd':j['cwd']=str(p)
    elif kind=='environment':j['env']['CUDA_VISIBLE_DEVICES']='3'
    elif kind=='logs':j['log']=str(p)
    elif kind=='start':
        q=Path(j['consumption']);q.write_text(json.dumps(dict(process=dict(pid=22,start_ticks=124))))
    else:(proc/'21/task/21/children').write_text('')
    assert attest_worker(j,proc)['attested'] is False
