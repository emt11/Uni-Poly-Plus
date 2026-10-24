"""One bounded queue: two FT slots per physical GPU, no retry."""
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.training.ua2d_sota import *

def launch(name,args,gpu):
    folder=LOG/name;folder.mkdir(parents=True,exist_ok=False)
    command=[sys.executable,'scripts/run_ua2d_sota.py','--plan','UA2D-SOTA-20261018-01','--revision','r1','--stage','development',*args]
    write(folder/'command.json',dict(argv=command,cwd=str(ROOT),gpu=gpu,window='ua2d_sota_'+name,threads=4,time=now()))
    script=folder/'run.sh'
    script.write_text('#!/bin/bash\ncd '+shlex.quote(str(ROOT))+' || exit 90\nexport CUDA_VISIBLE_DEVICES='+str(gpu)+' OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4\n'+
      'date -Iseconds > '+shlex.quote(str(folder/'start.txt'))+'\n'+shlex.join(command)+' > '+shlex.quote(str(folder/'stdout.log'))+' 2> '+shlex.quote(str(folder/'stderr.log'))+'\n'+
      'ua_code=$?\nprintf "%s\\n" "$ua_code" > '+shlex.quote(str(folder/'exit.txt'))+'\ndate -Iseconds > '+shlex.quote(str(folder/'end.txt'))+'\nexit "$ua_code"\n')
    subprocess.run(['tmux','new-window','-d','-t','Uni-Poly','-n','ua2d_sota_'+name,'bash '+shlex.quote(str(script))],check=True)
    return folder/'exit.txt'

def matrix(arm):
    if (OUT/'development'/f'{arm}_dispatch.json').exists(): raise FileExistsError('no repeat matrix')
    units=iter((t,f) for t in TASKS for f in range(5));running={};done=[];failed=False;end=False
    write(OUT/'development'/f'{arm}_dispatch.json',dict(status='RUNNING',starts=0,time=now()))
    starts=0
    while running or not end:
        for slot,(path,t,f) in list(running.items()):
            if path.exists():
                code=int(path.read_text());done.append(dict(task=t,fold=f,exit=code));del running[slot]
                if code: failed=True;end=True
        for gpu in (2,3):
            for slot in (0,1):
                if end or (gpu,slot) in running: continue
                try: t,f=next(units)
                except StopIteration: end=True;break
                starts+=1
                path=launch(f'ft_{arm}_{t}_{f}',['--unit','ft','--arm',arm,'--task',t,'--fold',str(f)],gpu)
                running[gpu,slot]=(path,t,f)
        write(OUT/'development'/f'{arm}_dispatch.json',dict(status='RUNNING' if running else ('阻断' if failed else 'PASS'),starts=starts,completed=done,time=now(),retry=False))
        refresh_budget()
        if running: time.sleep(5)
    return not failed

if __name__=='__main__':
    from src.training.glt_dual_runtime import require_tmux
    require_tmux()
    for arm in sys.argv[1:]: matrix(arm)

def pt_launch(arm):
    folder=LOG/f'pt_{arm}';folder.mkdir(parents=True,exist_ok=False)
    args=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=2','scripts/run_ua2d_sota.py','--plan','UA2D-SOTA-20261018-01','--revision','r1','--stage','development','--unit','pt','--arm',arm]
    write(folder/'command.json',dict(argv=args,cwd=str(ROOT),gpus=[2,3],world=2,window='ua2d_sota_pt_'+arm,time=now()))
    script=folder/'run.sh'
    script.write_text('#!/bin/bash\ncd '+shlex.quote(str(ROOT))+' || exit 90\nexport CUDA_VISIBLE_DEVICES=2,3 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4\n'+
        'date -Iseconds > '+shlex.quote(str(folder/'start.txt'))+'\n'+shlex.join(args)+' > '+shlex.quote(str(folder/'stdout.log'))+' 2> '+shlex.quote(str(folder/'stderr.log'))+'\n'+
        'ua_code=$?\nprintf "%s\\n" "$ua_code" > '+shlex.quote(str(folder/'exit.txt'))+'\ndate -Iseconds > '+shlex.quote(str(folder/'end.txt'))+'\nexit "$ua_code"\n')
    subprocess.run(['tmux','new-window','-d','-t','Uni-Poly','-n','ua2d_sota_pt_'+arm,'bash '+shlex.quote(str(script))],check=True)
    return folder/'exit.txt'

def wait_exit(path):
    while not path.exists():
        refresh_budget();time.sleep(10)
    refresh_budget();return int(path.read_text())==0

def after_initial():
    if not wait_exit(LOG/'controller/exit.txt'): raise RuntimeError('initial FT controller failure')
    if len(list((OUT/'finetune/T0').glob('*/fold*/metrics.json')))!=35:
        write(OUT/'development/PT_blocked.json',dict(reason='T0 matching control incomplete',status='阻断',time=now()));return
    for arm in ('T_RAWCE','T_AL05','M_ALL4','M_MS_CONST','M_MS_PH'):
        if wait_exit(pt_launch(arm)): matrix(arm)
    write(OUT/'development/independent_matrix_end.json',dict(status='待审查',time=now(),formal='NOT_RUN'))

def integration_and_kernel():
    proposal=propose_integration()
    if proposal['run']: matrix('I')
    core=choose_core()['core']
    if core!='UA':
        wait_exit(launch('kernel_F1',['--unit','F1','--arm',core],2))
    f0=kernel_folder('F0')/'summary.json'
    if not f0.exists():
        write(OUT/'development/finalization_blocked.json',dict(status='阻断',reason='F0 failed and recovery not authorized; cannot complete global validation selection or freeze final system',core=core,time=now(),outer_test='NOT_RUN'))
        return False
    if core!='UA' and not (OUT/'kernel/F1/summary.json').exists():
        write(OUT/'development/finalization_blocked.json',dict(status='阻断',reason='F1 incomplete; no final system freeze',core=core,time=now(),outer_test='NOT_RUN'))
        return False
    return True
