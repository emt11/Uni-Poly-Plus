"""Unattended bounded FCE campaign; no re-runs of successful science units."""
import datetime
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time
import traceback
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_fce_param_unit import OUT,LOG,SPLIT,ARMS,TASKS,read,write,config_for,epochs
from src.training.glt_dual_runtime import require_tmux
PYTHON='/opt/conda/bin/python'
GPUS=(2,3)


def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()


def available(gpu):
    rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free','--format=csv,noheader,nounits'],text=True)
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True)
    row=next([v.strip() for v in s.split(',')] for s in rows.splitlines() if int(s.split(',')[0])==gpu)
    return int(row[2])>=12000 and not any(s.split(',')[0].strip()==row[1] for s in apps.splitlines() if s.strip())


def launch(name,argv,gpu=None):
    if gpu is not None and gpu not in GPUS:raise ValueError('physical GPU2/3 only')
    folder=LOG/name;folder.mkdir(parents=True,exist_ok=False)
    environment=dict(CUDA_VISIBLE_DEVICES='' if gpu is None else str(gpu),OMP_NUM_THREADS='16' if gpu is None else '4',
        OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1')
    write(folder/'command.json',dict(argv=argv,command=shlex.join(argv),cwd=str(ROOT),environment=environment,gpu=gpu))
    launcher='''import datetime,json,os,subprocess,sys,time
from pathlib import Path
f=Path(sys.argv[1]);j=json.loads((f/'command.json').read_text());start=time.monotonic()
j.update(start=datetime.datetime.now(datetime.timezone.utc).isoformat(),status='RUNNING')
(f/'execution.json').write_text(json.dumps(j,indent=2))
with (f/'stdout.log').open('xb') as out,(f/'stderr.log').open('xb') as err:
 p=subprocess.Popen(j['argv'],cwd=j['cwd'],env=dict(os.environ,**j['environment']),stdout=out,stderr=err,start_new_session=True)
 (f/'pid.txt').write_text(str(p.pid));rc=p.wait()
j.update(end=datetime.datetime.now(datetime.timezone.utc).isoformat(),elapsed_seconds=time.monotonic()-start,exit_code=rc,status='PASS' if rc==0 else 'FAILED')
(f/'execution.json').write_text(json.dumps(j,indent=2));(f/'exit.txt').write_text(str(rc));sys.exit(rc)
'''
    script=folder/'command.sh';script.write_text('cd '+shlex.quote(str(ROOT))+'\n'+shlex.join([PYTHON,'-c',launcher,str(folder)])+'\n')
    window='fce_param_r2_controller' if name=='controller' else 'fce_r2_'+name
    subprocess.run(['tmux','new-window','-d','-t','Uni-Poly','-n',window,'bash '+shlex.quote(str(script))],check=True)
    # Keep completed windows and real exits, including failures.
    subprocess.run(['tmux','set-option','-w','-t','Uni-Poly:'+window,'remain-on-exit','on'],check=True)
    return folder


def consumption():
    result=dict(ft_starts=0,ft_completed=0,ft_epochs=0,ft_attempted_updates=0,ft_completed_updates=0,
        ft_failed=0,formal_starts=0,formal_completed=0,cpu_test_starts=0,pt_starts=0)
    for f in LOG.glob('ft_*/command.json'):result['ft_starts']+=1
    for f in LOG.glob('test_*/command.json'):result['formal_starts']+=1
    result['cpu_test_starts']=len(list(LOG.glob('final_cpu*/command.json')))
    for f in (OUT/'finetune').glob('*/*/fold*'):
        if (f/'updates.json').exists():
            u=read(f/'updates.json');result['ft_attempted_updates']+=u['attempted'];result['ft_completed_updates']+=u['completed']
        if (f/'progress.json').exists():result['ft_epochs']+=len(read(f/'progress.json')['history'])
        result['ft_completed']+=int((f/'metrics.json').exists())
        result['ft_failed']+=int((f/'failure.json').exists())
    result['formal_completed']=len(list((OUT/'formal').glob('*/*/fold*/metrics.json')))
    limits=dict(ft_starts=140,ft_completed=140,ft_epochs=9600,ft_attempted_updates=205800,
        ft_completed_updates=205800,formal_starts=70,formal_completed=70,cpu_test_starts=2,pt_starts=0)
    if any(result[k]>v for k,v in limits.items()):raise RuntimeError('hard budget exceeded '+str(result))
    write(OUT/'consumption.json',result)
    return result


def snapshot(stage,running,waiting):
    c=consumption();durations=[];steps=0
    for path in LOG.glob('ft_*/execution.json'):
        x=read(path)
        if x.get('exit_code')==0:
            name=path.parent.name.split('_');task=name[-2];fold=int(name[-1])
            n=epochs(task)*math.ceil(len(read(SPLIT/(task+'.json'))['folds'][fold]['train_indices'])/24)
            durations.append(x['elapsed_seconds']);steps+=n
    eta=None if not steps else max(0,205800-c['ft_completed_updates'])*sum(durations)/steps/2
    value=dict(time=now(),stage=stage,consumption=c,running=running,waiting=waiting,
        conservative_ft_eta_seconds=None if eta is None else eta*1.3,
        eta_basis='normal completed FT walltime per update, two available GPUs; excludes resource waiting/formal/tests',
        status='执行中',independent_review='NOT_PERFORMED')
    write(OUT/'status.json',value)
    with (LOG/'controller/progress.jsonl').open('a') as f:f.write(json.dumps(value,ensure_ascii=False)+'\n')
    # Update one owned section; retain every older handoff verbatim.
    p=ROOT/'CURRENT_PLAN.md';text=p.read_text();marker='\n<!-- FCE_R2_LIVE -->\n'
    text=text.split(marker)[0]
    estimate='尚不可估计' if eta is None else f'{eta*1.3/3600:.2f}小时（不含等待／正式预测／测试）'
    p.write_text(text+marker+f'\n本轮实时记录：{now()}（UTC）；阶段{stage}，FT完成{c["ft_completed"]}/140、失败{c["ft_failed"]}、epochs {c["ft_epochs"]}/9600、attempted updates {c["ft_attempted_updates"]}/205800、正式预测{c["formal_completed"]}/70；剩余FT估计{estimate}。运行／等待及命令见results/fce_param_20261014/r2/status.json与日志controller/progress.jsonl。\n')


def matrix(arms,formal=False):
    queue=[(a,t,f) for a in arms for t in TASKS for f in range(5)];running={};failures=[]
    # Existing successful units are immutable and never re-run.
    stage='formal' if formal else 'finetune'
    queue=[u for u in queue if not (OUT/stage/u[0]/u[1]/f'fold{u[2]}'/'metrics.json').exists()]
    for u in list(queue):
        if (OUT/stage/u[0]/u[1]/f'fold{u[2]}'/'failure.json').exists():
            failures.append(u);queue.remove(u)
    while queue or running:
        for gpu,(folder,unit) in list(running.items()):
            if (folder/'exit.txt').exists():
                rc=int((folder/'exit.txt').read_text())
                if rc:failures.append(unit)
                del running[gpu]
        waiting=[]
        for gpu in GPUS:
            if gpu in running or not queue:continue
            if not available(gpu):waiting.append(gpu);continue
            arm,task,fold=queue.pop(0)
            argv=[PYTHON,'scripts/run_fce_param_unit.py','--arm',arm,'--task',task,'--fold',str(fold)]
            if formal:argv.append('--formal')
            name=('test_' if formal else 'ft_')+arm+'_'+task+'_'+str(fold)
            if (LOG/name).exists():
                failures.append((arm,task,fold));continue
            folder=launch(name,argv,gpu);running[gpu]=(folder,(arm,task,fold))
        snapshot(('正式预测' if formal else 'FT')+':'+','.join(arms),
            {str(g):dict(log=str(x[0]),unit=x[1]) for g,x in running.items()},waiting)
        if queue or running:time.sleep(30)
    return failures


def campaign():
    require_tmux();started=now();failed=[]
    # Fixed source copies document the actual runtime without hashes.
    for name in ['scripts/run_fce_param.py','scripts/run_fce_param_unit.py','scripts/select_fce_param.py',
                 'src/training/fce_param.py','tests/test_fce_param.py']:
        dest=OUT/'source_used'/name;dest.parent.mkdir(parents=True,exist_ok=True)
        if dest.exists():raise FileExistsError('runtime snapshot exists')
        shutil.copyfile(ROOT/name,dest)
    for arm in ARMS:
        for task in TASKS:write(OUT/'initialization'/arm/(task+'.json'),config_for(arm,task))
    planned_epochs=sum(epochs(t)*5 for t in TASKS)*4
    planned_updates=sum(epochs(t)*math.ceil(len(read(SPLIT/(t+'.json'))['folds'][f]['train_indices'])/24) for t in TASKS for f in range(5))*4
    write(OUT/'planned_budget.json',dict(ft=140,epochs=planned_epochs,updates=planned_updates,formal=70))
    if (planned_epochs,planned_updates)!=(9600,205800):raise ValueError('actual split budget differs from authorized plan')
    try:
        for arm in ARMS:failed+=matrix([arm])
        if failed:
            write(OUT/'blocked.json',dict(failed_units=failed,reason='no complete trusted continuation state; successful units kept; no retry budget expansion'))
            raise RuntimeError('incomplete FT matrix; selection and TEST prohibited')
        # Run slow CPU freeze/summary inside this controller tmux window.
        from scripts.select_fce_param import freeze,summarize
        candidate=freeze();failed=matrix(['R0',candidate],True)
        if failed:raise RuntimeError('incomplete formal predictions '+str(failed))
        test=launch('final_cpu',[ 'timeout','300s',PYTHON,'-m','pytest','-q','tests/test_fce_param.py'])
        while not (test/'exit.txt').exists():snapshot('最终CPU测试',{},[]);time.sleep(10)
        rc=int((test/'exit.txt').read_text());delivery=summarize()
        c=consumption();state='待审查' if rc==0 else '需返修'
        write(OUT/'completion.json',dict(status=state,start=started,end=now(),test_exit=rc,candidate=candidate,consumption=c,independent_review='NOT_PERFORMED'))
        p=ROOT/'CURRENT_PLAN.md';p.write_text(p.read_text()+f'\n\n## FCE r2 最终执行记录\n\n状态：**{state}**。四臂140 FT、70正式预测实际完成；唯一CPU测试真实exit {rc}。独立审查未进行，不归档PROJECT_HISTORY。实际预算：{json.dumps(c,ensure_ascii=False)}。\n\n'+delivery+'\n')
    except BaseException as e:
        c=consumption();write(OUT/'campaign_failure.json',dict(status='阻断',error=repr(e),traceback=traceback.format_exc(),time=now(),consumption=c))
        p=ROOT/'CURRENT_PLAN.md';p.write_text(p.read_text()+'\n\nFCE r2：阻断。'+repr(e)+'；现场results/fce_param_20261014/r2/campaign_failure.json。缺臂／缺折不进入选优，不判定有效。\n')
        raise


if __name__=='__main__':campaign()
