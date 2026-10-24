#!/usr/bin/env python3
"""Authorized r2 F0 only: 105 paired cached-feature adaptation units, one CPU suite."""
import argparse
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
import traceback
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.training.ua2d_fp_film import *

def launch(name,argv,gpu):
    dest=LOG/name;dest.mkdir(parents=True,exist_ok=False)
    window='film_r2_'+name
    write(dest/'command.json',dict(argv=argv,cwd=str(ROOT),gpu=gpu,window=window,time=now(),threads=4))
    script=dest/'run.sh'
    script.write_text('#!/bin/bash\ncd '+shlex.quote(str(ROOT))+' || exit 90\nexport CUDA_VISIBLE_DEVICES='+shlex.quote(gpu)+' OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4\n'+
        'pwd > '+shlex.quote(str(dest/'cwd.txt'))+'\ndate -Iseconds > '+shlex.quote(str(dest/'start.txt'))+'\n'+
        shlex.join(argv)+' > '+shlex.quote(str(dest/'stdout.log'))+' 2> '+shlex.quote(str(dest/'stderr.log'))+'\n'+
        'film_code=$?\nprintf "%s\\n" "$film_code" > '+shlex.quote(str(dest/'exit.txt'))+'\ndate -Iseconds > '+shlex.quote(str(dest/'end.txt'))+'\nexit "$film_code"\n')
    subprocess.run(['tmux','new-window','-d','-t','Uni-Poly','-n',window,'bash '+shlex.quote(str(script))],check=True)
    return dest/'exit.txt'

def resource_count(gpu):
    text=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True)
    return len([p for p in text.splitlines() if p.strip()])

def execute():
    if OUT.exists(): raise FileExistsError('r2 output root exists; no automatic resume/retry')
    OUT.mkdir(parents=True)
    write(OUT/'execution_manifest.json',dict(plan='UA2D-SOTA-20261018-01',revision='r2',round='F0',stage='development',
        commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),authorization='user current message: F0 NULL SHIFT FILM only plus one CPU suite',
        status='执行中',start=now(),pid=os.getpid(),cwd=str(ROOT),gpus=[2,3],per_gpu=2,recipe=RECIPE,limits=LIMITS,
        allowed_modes=MODES,feature_root=str(CACHE),trained_core=str(UA/'finetune/UA_2D'),logs=str(LOG),independent_review='NOT_RUN',
        old_controllers=dict(paused=[125308,127331],stopped_workers=[134866,136384,138199,138354]),outer_test='NOT_RUN'))
    budget();handoff('执行中','实现完成；仅 r2 F0 开发队列开始。CPU suite 尚未执行。')
    units=iter((m,t,f) for m in MODES for t in TASKS for f in range(5));running={};exhausted=False;failures=[];completed=[]
    try:
        while running or not exhausted:
            for slot,(path,m,t,f) in list(running.items()):
                if path.exists():
                    code=int(path.read_text());del running[slot]
                    completed.append(dict(mode=m,task=t,fold=f,exit=code))
                    if code:
                        failures.append(completed[-1]);exhausted=True
            for gpu in (2,3):
                for number in (0,1):
                    if exhausted or (gpu,number) in running: continue
                    # Actual dispatch decision, no benchmark or additional model forward.
                    local_pending=sum(1 for g,_ in running if g==gpu)
                    if max(resource_count(gpu),local_pending)>=2: continue
                    try: m,t,f=next(units)
                    except StopIteration: exhausted=True;break
                    args=[sys.executable,'scripts/run_ua2d_fp_film.py','--revision','r2','--round','F0','--stage','development','--mode',m,'--task',t,'--fold',str(f)]
                    path=launch(f'{m}_{t}_{f}',args,str(gpu));running[gpu,number]=(path,m,t,f)
            budget()
            write(OUT/'dispatch.json',dict(status='执行中',running=[dict(gpu=k[0],slot=k[1],mode=v[1],task=v[2],fold=v[3]) for k,v in running.items()],completed=completed,failures=failures,time=now(),retry=False))
            if running or not exhausted: time.sleep(2)
        result=summary()
        # The sole correctness suite is deferred until all authorized launched units stop.
        budget(cpu_tests=1)
        path=launch('final_cpu',['timeout','300s',sys.executable,'-m','pytest','-q','tests/test_ua2d_fp_film.py'],'')
        while not path.exists(): time.sleep(2)
        cpu=int(path.read_text());review=OUT/'final_review'
        (review/'final_cpu.log').write_text(path.with_name('stdout.log').read_text()+path.with_name('stderr.log').read_text())
        (review/'final_cpu_exit.txt').write_text(str(cpu)+'\n')
        status='需返修' if cpu else ('待审查' if result['complete'] else '阻断')
        manifest=read(OUT/'execution_manifest.json');manifest.update(status=status,end=now(),cpu_exit=cpu,completed_units=len(completed),failures=failures,controller_exit=0 if cpu==0 and result['complete'] else 1)
        write(OUT/'execution_manifest.json',manifest);budget(cpu_tests=1)
        handoff(status,f'F0 开发完成度 {len(list((OUT/"adaptation/F0").glob("*/*/fold*/metrics.json")))}/105；唯一 CPU suite exit={cpu}。等待独立审查，其他范围保持暂停。')
        if cpu or not result['complete']: raise RuntimeError('execution incomplete or final CPU failed; no retry')
    except BaseException as e:
        write(OUT/'failure.json',dict(error=repr(e),traceback=traceback.format_exc(),time=now(),retry=False))
        raise

def main():
    from src.training.glt_dual_runtime import require_tmux
    require_tmux()
    p=argparse.ArgumentParser()
    p.add_argument('--revision',required=True,choices=['r2']);p.add_argument('--round',required=True,choices=['F0'])
    p.add_argument('--stage',required=True,choices=['development']);p.add_argument('--mode',choices=MODES)
    p.add_argument('--task',choices=TASKS);p.add_argument('--fold',type=int,choices=range(5))
    args=p.parse_args()
    if args.mode is None:
        if args.task is not None or args.fold is not None: raise ValueError('incomplete unit')
        execute()
    elif args.task is not None and args.fold is not None: run_unit(args.mode,args.task,args.fold)
    else: raise ValueError('incomplete unit')
if __name__=='__main__': main()
