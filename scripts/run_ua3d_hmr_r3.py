#!/usr/bin/env python3
"""r3 outer and worker entry, using the frozen parent virtualenv."""
import argparse
import importlib.metadata
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
from scripts.bootstrap_ua3d_hmr_r3 import OUT,LOG,P1,P2,PYTHON,THREADS,read,write,exclusive,identity,wait_wrapper,check_clock,fail

def resources():
    check_clock(True)
    if shutil.disk_usage(ROOT).free<16*1024**3:raise ValueError('free disk <16GiB')
    affinity=sorted(os.sched_getaffinity(0));quota=Path('/sys/fs/cgroup/cpu.max').read_text().split();eff=len(affinity) if quota[0]=='max' else min(len(affinity),max(1,math.floor(int(quota[0])/int(quota[1]))));versions={p:importlib.metadata.version(p) for p in read(P1/'authorization/dependencies.json')}
    if versions!=read(P1/'authorization/dependencies.json') or Path(sys.executable)!=PYTHON:raise ValueError('parent venv entry/versions differ')
    exclusive(OUT/'authorization/resources.json',dict(affinity=affinity,C_eff=eff,cpu_max=quota,versions=versions,python=sys.executable,new_install=0,threads=THREADS,disk_free=shutil.disk_usage(ROOT).free,time=time.time()))

def worker(a):
    log=Path(a.log);started=time.time();rc=1;child=None
    exclusive(log/'process.json',dict(**identity(),started_unix=started,argv=sys.argv,cwd=str(ROOT),CVD=os.environ.get('CUDA_VISIBLE_DEVICES'),env={k:os.environ.get(k) for k in THREADS}))
    def terminate(sig,frame):
        if child is not None and child.poll() is None:child.terminate();child.wait()
        raise SystemExit(128+sig)
    signal.signal(signal.SIGTERM,terminate)
    try:
        if a.kind in ('preflight','cpu'):
            if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU suite CVD')
            flag='--preflight' if a.kind=='preflight' else '--final';child=subprocess.Popen([str(PYTHON),str(ROOT/'tests/test_ua3d_hmr_r3.py'),flag],cwd=ROOT,env=os.environ);rc=child.wait()
        else:
            from src.training import ua3d_hmr_r3 as t
            if a.kind=='ft':t.fit_job('ft',a.arm,a.task,a.fold)
            elif a.kind=='test':t.formal_job(a.arm,a.task,a.fold)
            else:raise ValueError('unauthorized worker kind')
            rc=0
    except BaseException:
        write(log/'failure.json',dict(error=traceback.format_exc(),phase=a.kind));traceback.print_exc();rc=1
    exclusive(log/'end.json',dict(exit=rc,started_unix=started,ended_unix=time.time()));return rc

def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['outer','controller','worker']);p.add_argument('--kind');p.add_argument('--arm',default='');p.add_argument('--task',default='');p.add_argument('--fold',type=int,default=0);p.add_argument('--log');a=p.parse_args()
    if a.mode=='worker':return worker(a)
    exclusive(OUT/f'{a.mode}_self_process.json',dict(**identity(),started_unix=time.time(),argv=sys.argv,cwd=str(ROOT)))
    try:
        if a.mode=='controller':
            from src.training.ua3d_hmr_r3 import Controller
            return Controller().run()
        resources();return wait_wrapper([str(PYTHON),str(Path(__file__).resolve()),'controller'],LOG/'controller.log',OUT,'controller',dict(os.environ,**THREADS))
    except BaseException:
        fail(a.mode+'_initialize_run',traceback.format_exc());traceback.print_exc();return 1
if __name__=='__main__':sys.exit(main())
