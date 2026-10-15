#!/usr/bin/env python3
"""Fixed next-cycle dispatch, separate logs/writers, no automatic retries."""
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
import traceback
import datetime
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
OUT=ROOT/'results/glt_ft_generalization_20261008/r1'
LOG=ROOT/'logs/glt_ft_generalization_20261008/r1'
ARMS=('FT_PARTIAL','FT_LOW_LR','FT_REG')
TASKS=('eea','egb','egc','ei','eps','nc','xc')
PT_ARMS=('B0','T_RAW','T0','T0_A','K0','K_MD','K_MD_A','TK_MD','TK_MD_A','TK_MD_L','G_NORM','G_FGR')
def read(p):return json.loads(Path(p).read_text())
def write(p,v):
    p=Path(p);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n');os.replace(tmp,p)
def now():return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
def launch(name,command,gpus=(),timeout=None):
    folder=LOG/name;folder.mkdir(exist_ok=False)
    cmd=shlex.join(command);(folder/'command.txt').write_text(cmd+'\n');(folder/'cwd.txt').write_text(str(ROOT)+'\n')
    if timeout:cmd=f'timeout --signal=TERM {timeout}s '+cmd
    script=folder/'command.sh'
    script.write_text(f'''#!/bin/bash
cd {shlex.quote(str(ROOT))} || exit 90
export CUDA_VISIBLE_DEVICES={shlex.quote(','.join(map(str,gpus)))}
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
date -Iseconds > {shlex.quote(str(folder/'start.txt'))}
{cmd} > {shlex.quote(str(folder/'stdout.log'))} 2> {shlex.quote(str(folder/'stderr.log'))}
code=$?
printf '%s\\n' "$code" > {shlex.quote(str(folder/'exit.txt'))}
date -Iseconds > {shlex.quote(str(folder/'end.txt'))}
exit "$code"
''')
    subprocess.run(['tmux','new-window','-d','-t','Uni-Poly','-n','glt_ftg_r1_'+name,'bash '+shlex.quote(str(script))],check=True)
    return folder/'exit.txt'
def consumption():
    starts=list(LOG.glob('*/start.txt'))
    result=dict(pt_training_starts=0,pt_optimizer_updates=0,endpoint_starts=0,endpoint_completed=0,endpoint_presented=0,
        ft_starts=0,ft_completed=0,ft_epochs=0,ft_attempted_updates=0,ft_completed_updates=0,formal_starts=0,formal_completed=0,engineering_test_calls=0)
    for p in starts:
        name=p.parent.name;done=(p.parent/'exit.txt').exists() and int((p.parent/'exit.txt').read_text())==0
        if name.startswith('ptval_'):result['endpoint_starts']+=1;result['endpoint_completed']+=int(done)
        elif name.startswith('ft_'):result['ft_starts']+=1;result['ft_completed']+=int(done)
        elif name.startswith('formal_'):result['formal_starts']+=1;result['formal_completed']+=int(done)
        elif name=='final':result['engineering_test_calls']+=1
    for p in (OUT/'pt_endpoint').glob('*/progress_rank*.json'):result['endpoint_presented']+=read(p)['presented']
    for folder in (OUT/'finetune').glob('*/*/fold*'):
        p=next((folder/f for f in ('metrics.json','failure.json','progress.json') if (folder/f).exists()),None)
        if p:
            d=read(p);result['ft_attempted_updates']+=d['updates']['attempted'];result['ft_completed_updates']+=d['updates']['completed'];result['ft_epochs']+=d.get('epochs',d.get('completed_epochs',len(d.get('history',[]))))
    if result['endpoint_starts']>12 or result['endpoint_presented']>12288 or result['ft_starts']>105 or result['ft_epochs']>7200 or result['ft_attempted_updates']>154350 or result['formal_starts']>105 or result['engineering_test_calls']>1:raise RuntimeError('authorized budget exceeded')
    return result
def refresh():write(OUT/'consumption.json',consumption())
def wait(p):
    while not p.exists():refresh();time.sleep(5)
    refresh()
    if int(p.read_text()):raise RuntimeError(f'{p} exit={p.read_text().strip()}; no retry')
def free_gpus():
    busy={line.split(',')[0].strip() for line in subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True).splitlines() if line.strip()}
    return [int(line.split(',')[0]) for line in subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True).splitlines() if line.split(',')[1].strip() not in busy]
def units(arm,formal=False):
    pending=[(t,f) for t in TASKS for f in range(5)];active={};failed=None
    while pending or active:
        for gpu in free_gpus():
            if failed or not pending:break
            if gpu in active:continue
            task,fold=pending.pop(0);name=('formal_' if formal else 'ft_')+arm+'_'+task+'_f'+str(fold)
            folder=OUT/('formal' if formal else 'finetune')/arm/task/f'fold{fold}'
            if folder.exists():raise FileExistsError('existing unit: do not overwrite/restart '+str(folder))
            command=['python','scripts/run_glt_ft_generalization_unit.py','--arm',arm,'--task',task,'--fold',str(fold)]
            if formal:command.append('--formal')
            active[gpu]=(name,launch(name,command,[gpu]))
        for gpu,(name,p) in list(active.items()):
            if p.exists():
                code=int(p.read_text());print(f'{name} exit={code}',flush=True);del active[gpu]
                if code:failed=f'{name} exit={code}'
        refresh()
        if failed and not active:raise RuntimeError(failed+'; no retry')
        time.sleep(5)
def main():
    from src.training.glt_dual_runtime import require_tmux
    require_tmux();status=OUT/'execution_status.json'
    if status.exists():raise FileExistsError('controller previously started; no duplicate/retry')
    write(status,dict(status='执行中',plan='GLT-FT-GENERALIZATION-20261008-01/r1',started=now()))
    try:
        for arm in PT_ARMS:
            while len(free_gpus())!=4:time.sleep(5)
            print('endpoint '+arm,flush=True)
            wait(launch('ptval_'+arm,['torchrun','--standalone','--nproc_per_node=4','scripts/evaluate_glt_pt_endpoint.py','--arm',arm],range(4)))
        for arm in ARMS:units(arm)
        wait(launch('selection',['python','scripts/select_glt_ft_generalization.py','freeze']))
        wait(launch('final',['python','-m','pytest','-q','tests/test_glt_ft_generalization.py','--junitxml='+str(LOG/'final/junit.xml')],timeout=300))
        for arm in ARMS:units(arm,formal=True)
        wait(launch('summary',['python','scripts/select_glt_ft_generalization.py','summarize']))
        write(status,dict(status='待审查',completed=now(),consumption=consumption(),independent_review='未开始'))
    except BaseException as error:
        write(status,dict(status='阻断',time=now(),error=repr(error),traceback=traceback.format_exc(),no_retry=True));refresh();raise
if __name__=='__main__':main()
