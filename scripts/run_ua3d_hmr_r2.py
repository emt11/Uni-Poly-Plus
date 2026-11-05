#!/usr/bin/env python3
"""Real outer/controller/worker waits. Every scientific worker owns one identity."""
import argparse
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.bootstrap_ua3d_hmr_r2 import OUT,LOG,THREADS,read,write,exclusive,identity,check_clock

def resources():
    check_clock(True)
    if shutil.disk_usage(ROOT).free<128*1024**3:raise ValueError('startup free disk <128GiB')
    affinity=sorted(os.sched_getaffinity(0));quota=Path('/sys/fs/cgroup/cpu.max').read_text().split();effective=len(affinity) if quota[0]=='max' else min(len(affinity),max(1,math.floor(int(quota[0])/int(quota[1]))));R=max(2,math.ceil(.05*effective));mem={line.split(':')[0]:int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines() if line.split(':')[0] in ('MemAvailable','MemTotal')};pool=min(128*1024**3,.6*mem['MemAvailable']);workers=min(effective-R,math.floor(pool/1024**3))
    if workers<1:raise ValueError('insufficient CPU/memory')
    gpu=subprocess.run(['nvidia-smi','-i','2','--query-gpu=index,uuid,name,memory.total,memory.free','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout
    if 'GPU-31456dd5-a421-142b-644b-7a80f482dfb9' not in gpu:raise ValueError('GPU2 map changed')
    exclusive(OUT/'authorization/resources.json',dict(affinity=affinity,cpu_max=quota,C_eff=effective,R=R,M_pool=pool,P_geom=workers,memory=mem,disk_free=shutil.disk_usage(ROOT).free,gpu2=gpu,time=time.time(),threads=THREADS))
    versions={p:importlib.metadata.version(p) for p in ('numpy','scipy','torch','rdkit','scikit-image','imageio','tifffile','lazy-loader')}
    expected=read(ROOT/'results/ua3d_hmr_20261103/r1/authorization/dependencies.json')
    if versions!=expected:raise ValueError('read-only parent environment versions changed')
    exclusive(OUT/'authorization/dependencies.json',versions)
    exclusive(OUT/'authorization/environment_reuse.json',dict(python=sys.executable,parent_env=str(ROOT/'results/ua3d_hmr_20261103/r1/env'),installations=0,PYTHONDONTWRITEBYTECODE=os.environ.get('PYTHONDONTWRITEBYTECODE')))

def worker(args):
    phase=args.kind;log=Path(args.log);started=time.time();rc=1;usage=None
    exclusive(log/'process.json',dict(**identity(),started_unix=started,argv=sys.argv,cwd=str(ROOT),CVD=os.environ.get('CUDA_VISIBLE_DEVICES'),env={k:os.environ.get(k) for k in THREADS}))
    try:
        from src.training import ua3d_hmr_r2 as t
        if phase=='preflight':
            t.cpu_contract();rc=subprocess.run([sys.executable,str(ROOT/'tests/test_ua3d_hmr_r2.py'),'--preflight'],cwd=ROOT,env=os.environ).returncode
        elif phase=='prepare':
            from scripts.prepare_ua3d_hmr_r2 import prepare
            prepare();rc=0
        elif phase in ('diagnostic','ft'):t.fit_job(phase,args.arm,args.task,args.fold,args.repeat);rc=0
        elif phase=='test':t.formal_job(args.arm,args.task,args.fold);rc=0
        elif phase=='cpu':
            t.cpu_contract();rc=subprocess.run([sys.executable,str(ROOT/'tests/test_ua3d_hmr_r2.py'),'--final'],cwd=ROOT,env=os.environ).returncode
        else:raise ValueError('unknown worker kind')
    except BaseException:write(log/'failure.json',dict(error=traceback.format_exc(),phase=phase));traceback.print_exc();rc=1
    exclusive(log/'end.json',dict(exit=rc,started_unix=started,ended_unix=time.time()));return rc

def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['outer','controller','worker']);p.add_argument('--kind');p.add_argument('--arm',default='');p.add_argument('--task',default='');p.add_argument('--fold',type=int,default=0);p.add_argument('--repeat',default='');p.add_argument('--log');a=p.parse_args()
    if a.mode=='worker':return worker(a)
    if a.mode=='controller':
        try:
            from src.training.ua3d_hmr_r2 import Controller
            return Controller().run()
        except BaseException:
            from scripts.bootstrap_ua3d_hmr_r2 import fail
            fail('controller_import_initialize',traceback.format_exc());write(OUT/'controller_exit.json',dict(exit=1,ended_unix=time.time(),phase='import_initialize'));traceback.print_exc();return 1
    cmd=[sys.executable,str(Path(__file__).resolve()),'controller'];rc=1
    try:
        resources()
        with (LOG/'controller.log').open('x') as f:
            proc=subprocess.Popen(cmd,cwd=ROOT,env=dict(os.environ,**THREADS),stdout=f,stderr=subprocess.STDOUT,start_new_session=True);exclusive(OUT/'controller_process.json',dict(**identity(proc.pid),argv=cmd,cwd=str(ROOT),started_unix=time.time()));rc=proc.wait()
    except BaseException:
        from scripts.bootstrap_ua3d_hmr_r2 import fail
        fail('outer',traceback.format_exc());traceback.print_exc();rc=1
    if not (OUT/'controller_exit.json').exists():exclusive(OUT/'controller_exit.json',dict(exit=rc,ended_unix=time.time(),source='outer wait'))
    return rc
if __name__=='__main__':sys.exit(main())
