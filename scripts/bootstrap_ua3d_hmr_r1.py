#!/usr/bin/env python3
"""Standard-library launch, irreversible jobs and first-failure preservation."""
import argparse
import datetime
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/ua3d_hmr_20261103/r1'
LOG=ROOT/'logs/ua3d_hmr_20261103/r1'
SOURCES=('src/dataset/ua3d_hmr_r1.py','src/modules/ua3d_hmr_r1.py','src/training/ua3d_hmr_r1.py','scripts/prepare_ua3d_hmr_r1.py','scripts/run_ua3d_hmr_r1.py','scripts/bootstrap_ua3d_hmr_r1.py','configs/ua3d_hmr_r1_gpu2.json','tests/test_ua3d_hmr_r1.py')
THREADS={'CUDA_VISIBLE_DEVICES':'','WORLD_SIZE':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','NUMEXPR_NUM_THREADS':'1','VECLIB_MAXIMUM_THREADS':'1','PYTHONUNBUFFERED':'1','CUBLAS_WORKSPACE_CONFIG':':4096:8','PYTHONPATH':str(ROOT)}

def read(p):return json.loads(Path(p).read_text())
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_name(p.name+f'.tmp.{os.getpid()}');q.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n');os.replace(q,p)
def exclusive(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False)
def identity(pid=None):
    pid=pid or os.getpid();s=Path(f'/proc/{pid}/stat').read_text();return dict(pid=pid,start_ticks=int(s[s.rfind(')')+2:].split()[19]))
def parse_jobs(jobs):
    ids=set();counts={};intervals=[];wall=0
    for j in jobs:
        key=j['id']
        if key in ids:raise ValueError('duplicate job identity')
        ids.add(key)
        for k in ('reserved_unix','started_unix','ended_unix'):
            if k not in j:raise ValueError('missing time field '+k)
            if j[k] is not None and (not isinstance(j[k],(float,int)) or not math.isfinite(j[k])):raise ValueError('time type')
        if j['started_unix'] is not None and j['started_unix']<j['reserved_unix']:raise ValueError('start before reservation')
        if j['ended_unix'] is not None and (j['started_unix'] is None or j['ended_unix']<j['started_unix']):raise ValueError('end without start')
        c=counts.setdefault(j['kind'],dict(reserved=0,started=0,attempted=0,completed=0));c['reserved']+=1;c['started']+=int(j['started_unix'] is not None);c['attempted']+=int(j['started_unix'] is not None);c['completed']+=int(j['exit']==0)
        if j['started_unix'] is not None:
            end=j['ended_unix'] or time.time();wall+=end-j['started_unix']
            if j['gpu']:intervals.append((j['started_unix'],end))
    busy=0;last=0
    for a,b in sorted(intervals):busy+=max(0,b-max(a,last));last=max(last,b)
    return dict(counts=counts,process_wall_seconds=wall,physical_gpu_union_seconds=busy)
def state_for(jobs,expected):
    parse_jobs(jobs)
    return dict(status='NOT_RUN' if not jobs else 'FAILED' if any(j['exit'] not in (None,0) for j in jobs) else 'RUNNING' if any(j['exit'] is None for j in jobs) else 'PASSED' if len(jobs)==expected else 'PARTIAL',attempts=len(jobs),exit=next((j['exit'] for j in jobs if j['exit'] not in (None,0)),None) if any(j['exit']!=0 for j in jobs) else 0 if len(jobs)==expected else None)
def size(root):return sum(p.stat().st_size for p in Path(root).rglob('*') if p.is_file() and not p.is_symlink())
def fail(phase,error,code=1):
    if not (OUT/'first_failure.json').exists():exclusive(OUT/'first_failure.json',dict(phase=phase,error=error,exit=code,time=time.time()))
    write(OUT/'execution_manifest.json',dict(status='STOPPED',phase=phase,execution_complete=False,independent_review='PENDING',error_source='first_failure.json',exit=code))
def check_clock(preparation=False,reserve=2400):
    c=read(OUT/'authorization/clock.json');now=time.time()
    if now>=c['deadline_unix']-reserve or (preparation and now>=c['prepare_deadline_unix']):raise TimeoutError('deadline/preparation reserve')

def bootstrap():
    if not os.environ.get('TMUX'):raise RuntimeError('tmux required')
    lock=(OUT/'bootstrap.lock').open('x');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    env=dict(os.environ,**THREADS);phase='source_lock';rc=1
    try:
        check_clock(True)
        for name in SOURCES:
            target=OUT/'authorization/source_locked'/name;target.parent.mkdir(parents=True,exist_ok=True)
            with target.open('x') as f:f.write((ROOT/name).read_text())
        exclusive(OUT/'authorization/source_lock.json',dict(files=SOURCES,locked_unix=time.time(),no_hashes=True))
        python=OUT/'env/bin/python'
        for label,argv in [('venv',[sys.executable,'-m','venv','--system-site-packages',str(OUT/'env')]),('dependency_plan',[str(python),'-m','pip','install','--dry-run','--report',str(OUT/'authorization/pip_plan.json'),'--index-url','https://pypi.org/simple','--only-binary=:all:','scikit-image==0.25.2'])]:
            phase=label
            with (LOG/f'{label}.log').open('x') as f:result=subprocess.run(argv,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
            exclusive(OUT/'authorization'/f'{label}_exit.json',dict(argv=argv,cwd=str(ROOT),exit=result.returncode,time=time.time()))
            if result.returncode:raise RuntimeError(label+' failed')
        plan=read(OUT/'authorization/pip_plan.json');allowed={'scikit-image','imageio','tifffile','lazy-loader'}
        for item in plan['install']:
            if item['metadata']['name'].lower().replace('_','-') not in allowed:raise ValueError('dependency transaction would replace numerical stack')
            if not item['download_info']['url'].startswith('https://files.pythonhosted.org/'):raise ValueError('nonofficial wheel')
        phase='dependency_install';argv=[str(python),'-m','pip','install','--index-url','https://pypi.org/simple','--only-binary=:all:','--report',str(OUT/'authorization/pip_install.json'),'scikit-image==0.25.2']
        with (LOG/'dependency_install.log').open('x') as f:p=subprocess.run(argv,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
        exclusive(OUT/'authorization/dependency_install_exit.json',dict(exit=p.returncode,argv=argv,time=time.time()))
        if p.returncode:raise RuntimeError('install failed')
        phase='outer';cmd=[str(python),str(ROOT/'scripts/run_ua3d_hmr_r1.py'),'outer']
        with (LOG/'outer.log').open('x') as f:p=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True);write(OUT/'outer_process.json',dict(**identity(p.pid),argv=cmd,cwd=str(ROOT),env=THREADS,started_unix=time.time()));rc=p.wait()
        exclusive(OUT/'outer_exit.json',dict(exit=rc,ended_unix=time.time()))
        if rc:raise RuntimeError('outer exit '+str(rc))
        phase='final_delivery'
        # Finalize only after final CPU has really exited. No second suite.
        import importlib.util
        spec=importlib.util.spec_from_file_location('hmr_training',ROOT/'src/training/ua3d_hmr_r1.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);mod.deliver(None)
        rc=0
    except BaseException:
        rc=rc or 1;fail(phase,traceback.format_exc(),rc)
        try:
            import importlib.util
            spec=importlib.util.spec_from_file_location('hmr_training',ROOT/'src/training/ua3d_hmr_r1.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);mod.deliver(traceback.format_exc())
        except BaseException:write(OUT/'delivery_error.json',dict(error=traceback.format_exc()))
    finally:
        write(OUT/'bootstrap_exit.json',dict(exit=rc,ended_unix=time.time(),phase=phase));lock.close()
    return rc
if __name__=='__main__':sys.exit(bootstrap())
