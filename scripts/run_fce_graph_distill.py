"""r15 bounded event-driven neural FT/formal orchestration; no retries."""
import argparse
import copy
import ctypes
import datetime
import difflib
import json
import math
import os
from pathlib import Path
import select
import shlex
import shutil
import subprocess
import sys
import time
import traceback
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from scripts.run_fce_graph_distill_unit import (ARMS,TASKS,OUT,LOG,R10,POLICY,read,write,
    config_for,epochs,source_binding,validate_checkpoint,validate_frozen,selected_history)
from scripts.run_fce_graph_distill_unit import MSE_POLICY
from src.training.fce_graph_distill import parent,pt_config_for,validate_parent,VALIDATION_STEPS,RESUME,TEACHER
from src.training.fce_kernel import bind_branch_arrays,metrics,blend_arrays
from src.training.glt_dual_runtime import require_tmux
PYTHON='/opt/conda/bin/python';GPUS=(2,3)
EDITED=('CURRENT_PLAN.md','src/modules/glt_dual_pretrain.py','src/training/fce_graph_distill.py',
    'scripts/pretrain_fce_graph_distill.py','scripts/run_fce_graph_distill.py',
    'scripts/run_fce_graph_distill_unit.py','tests/test_fce_graph_distill.py')
LIMITS=dict(pt_starts=2,pt_attempted_updates=10000,pt_completed_updates=10000,
    student_training_presentations=10080000,teacher_training_presentations=10080000,
    pt_validation_calls=6,student_validation_presentations=6144,teacher_validation_presentations=6144,
    deploy_packages=2,resume_packages=2,ft_starts=70,ft_completed=70,ft_epochs=4800,ft_attempted_updates=102900,
    ft_completed_updates=102900,validation_calls=4800,formal_starts=70,formal_completed=70,formal_outputs=140,
    cpu_test_starts=1,krr_fit=0,krr_forward=0,morgan_generation=0,ft_workers=0,extra_seeds=0,refit=0)


def now():return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
def check_budget(b):
    if set(b)!=set(LIMITS) or any(type(v) is not int or not 0<=v<=LIMITS[k] for k,v in b.items()):raise ValueError('graph-distill hard budget mismatch')
def require_stage(stage,b):
    check_budget(b)
    required=dict(pt_completed_updates=10000,student_training_presentations=10080000,teacher_training_presentations=10080000,
        pt_validation_calls=6,student_validation_presentations=6144,teacher_validation_presentations=6144,deploy_packages=2,resume_packages=2)
    if stage in ('FT','freeze','formal','CPU') and any(b[k]!=v for k,v in required.items()):raise ValueError('complete both PT arms required')
    if stage in ('freeze','formal','CPU') and any(b[k]!=v for k,v in dict(ft_completed=70,ft_epochs=4800,ft_attempted_updates=102900,ft_completed_updates=102900,validation_calls=4800).items()):raise ValueError('complete 70 FT matrix required')
    if stage=='CPU' and (b['formal_completed'],b['formal_outputs'])!=(70,140):raise ValueError('complete formal matrix required for CPU test')
def consumption():
    b={k:0 for k in LIMITS};b['ft_starts']=len(list(LOG.glob('ft_*/command.json')));b['formal_starts']=len(list(LOG.glob('test_*/command.json')));b['cpu_test_starts']=len(list(LOG.glob('final_cpu/command.json')))
    per_arm={}
    for arm in ARMS:
        b['pt_starts']+=int((LOG/('pt_'+arm)/'command.json').exists());folder=OUT/'pretrain'/arm;rank_records=[]
        for rank in (0,1):
            p=folder/f'failure_rank{rank}.json'
            if not p.exists():p=folder/f'consumption_rank{rank}.json'
            if p.exists():rank_records.append(read(p))
        per_arm[arm]=dict(rank_records=rank_records)
        if rank_records:
            b['pt_attempted_updates']+=max(r['attempted_updates'] for r in rank_records)
            b['pt_completed_updates']+=min(r['completed_updates'] for r in rank_records) if len(rank_records)==2 else 0
            for key in ('student_training_presentations','teacher_training_presentations'):b[key]+=sum(r[key] for r in rank_records)
        for step in VALIDATION_STEPS:
            val=OUT/'pt_validation'/arm/f'step{step:05d}'
            b['pt_validation_calls']+=int(val.exists())
            for rank in (0,1):
                p=val/f'progress_rank{rank}.json'
                if p.exists():
                    r=read(p);b['student_validation_presentations']+=r['student_presented'];b['teacher_validation_presentations']+=r['teacher_presented']
        b['deploy_packages']+=int((folder/'deploy_10000.pt').exists());b['resume_packages']+=int((folder/'resume_10000.pt').exists())
    for f in (OUT/'ft').glob('*/*/fold*'):
        if (f/'updates.json').exists():
            u=read(f/'updates.json');b['ft_attempted_updates']+=u['attempted'];b['ft_completed_updates']+=u['completed']
        if (f/'progress.json').exists():b['ft_epochs']+=len(read(f/'progress.json')['history'])
        b['ft_completed']+=int((f/'metrics.json').exists())
    b['validation_calls']=b['ft_epochs'];b['formal_completed']=len(list((OUT/'formal').glob('*/*/fold*/metrics.json')))
    b['formal_outputs']=len(list((OUT/'formal').glob('*/*/fold*/*predictions.npz')))
    write(OUT/'consumption.json',b);write(OUT/'pt_consumption_by_arm.json',per_arm);check_budget(b);return b



def launch(name,argv,gpu=None):
    require_tmux()
    if gpu is not None and gpu not in (*GPUS,"2,3"):raise ValueError('GPU2/3 only')
    folder=LOG/name;folder.mkdir(parents=True,exist_ok=False)
    environment=dict(CUDA_VISIBLE_DEVICES='' if gpu is None else str(gpu),OMP_NUM_THREADS='1' if gpu is None else '4',
        OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1')
    command=dict(argv=argv,command=shlex.join(argv),cwd=str(ROOT),environment=environment,gpu=gpu,window='fce_graph_distill_'+name)
    write(folder/'command.json',command)
    wrapper='''import datetime,json,os,subprocess,sys,time,traceback
from pathlib import Path
f=Path(sys.argv[1]);j=json.loads((f/'command.json').read_text());started=time.monotonic()
if any((f/n).exists() for n in ('execution.json','exit.txt','stdout.log','stderr.log')):raise FileExistsError('no replay')
j.update(start=datetime.datetime.now(datetime.timezone.utc).isoformat(),status='RUNNING',end=None,exit_code=None)
def save():
 p=f/'execution.tmp';p.write_text(json.dumps(j,indent=2)+'\\n');os.replace(p,f/'execution.json')
save();rc=1
try:
 with (f/'stdout.log').open('xb') as out,(f/'stderr.log').open('xb') as err:
  child=subprocess.Popen(j['argv'],cwd=j['cwd'],env=dict(os.environ,**j['environment']),stdout=out,stderr=err)
  j['pid']=child.pid;save();rc=child.wait()
except BaseException:j['error']=traceback.format_exc()
finally:
 j.update(end=datetime.datetime.now(datetime.timezone.utc).isoformat(),elapsed_seconds=time.monotonic()-started,exit_code=rc,status='EXITED');save();(f/'exit.txt').write_text(str(rc)+'\\n')
if f.name=='controller':subprocess.run([sys.executable,'scripts/run_fce_graph_distill.py','--start-finish'],cwd=j['cwd'],env=dict(os.environ,**j['environment']),check=False)
sys.exit(rc)
'''
    script=folder/'command.sh';script.write_text('cd '+shlex.quote(str(ROOT))+'\n'+shlex.join([PYTHON,'-c',wrapper,str(folder)])+'\n')
    subprocess.run(['tmux','new-window','-d','-t','Uni-Poly','-n',command['window'],
        'tmux set-option -w -t "$TMUX_PANE" remain-on-exit on\nexec bash '+shlex.quote(str(script))],check=True)
    return folder


def wait_original(folder):
    path=folder/'exit.txt';libc=ctypes.CDLL(None,use_errno=True);fd=libc.inotify_init1(os.O_CLOEXEC)
    if fd<0:raise OSError(ctypes.get_errno(),'inotify_init1')
    try:
        if libc.inotify_add_watch(fd,os.fsencode(folder),0x8|0x80)<0:raise OSError(ctypes.get_errno(),'inotify_add_watch')
        while not path.exists():os.read(fd,4096)
        return int(path.read_text())
    finally:os.close(fd)


def gpu_slots(gpu,running):
    uuids=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader,nounits'],text=True)
    uuid=next(s.split(',')[1].strip() for s in uuids.splitlines() if int(s.split(',')[0])==gpu)
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True)
    actual=sum(s.split(',')[0].strip()==uuid for s in apps.splitlines() if s.strip())
    own=[v for v in running.values() if v['gpu']==gpu]
    pending=sum(v['formal'] or not (OUT/'ft'/v['arm']/v['task']/f"fold{v['fold']}"/'progress.json').exists() for v in own)
    occupied=max(actual+pending,len(own));return max(0,4-occupied),dict(gpu=gpu,actual_contexts=actual,pending=pending,own=len(own),conservative_occupied=occupied)


def matrix(formal=False):
    # Largest work first reduces tail duration without changing any unit's seed.
    queue=sorted([(a,t,f) for a in ARMS for t in TASKS for f in range(5)],key=lambda x:epochs(x[1])*math.ceil(len(source_binding(*x)['reference']['train_indices'])/24),reverse=True)
    running={};failures=[];libc=ctypes.CDLL(None,use_errno=True);fd=libc.inotify_init1(os.O_CLOEXEC)
    if fd<0:raise OSError(ctypes.get_errno(),'inotify_init1')
    try:
        while queue or running:
            changed=False
            for name,v in list(running.items()):
                if (v['log']/'exit.txt').exists():
                    rc=int((v['log']/'exit.txt').read_text());changed=True
                    if rc:failures.append(dict(task=v['task'],fold=v['fold'],exit_code=rc,log=str(v['log'])))
                    del running[name]
            if failures and queue:
                write(OUT/('formal_stopped_queue.json' if formal else 'ft_stopped_queue.json'),dict(unstarted=queue,failures=failures));queue=[]
            if queue:
                for gpu in GPUS:
                    count,observed=gpu_slots(gpu,running)
                    for slot in range(count):
                        if not queue:break
                        arm,task,fold=queue.pop(0);name=('test_' if formal else 'ft_')+arm+'_'+task+'_'+str(fold)
                        folder=OUT/('formal' if formal else 'ft')/arm/task/f'fold{fold}'
                        if folder.exists() or (LOG/name).exists():raise FileExistsError('unit writer/overwrite conflict')
                        b=consumption();key='formal_starts' if formal else 'ft_starts'
                        if b[key]>=70:raise ValueError('start budget exhausted')
                        argv=[PYTHON,'scripts/run_fce_graph_distill_unit.py','--arm',arm,'--task',task,'--fold',str(fold)]+(['--formal'] if formal else [])
                        log=launch(name,argv,gpu);running[name]=dict(gpu=gpu,log=log,arm=arm,task=task,fold=fold,formal=formal);changed=True
                        if libc.inotify_add_watch(fd,os.fsencode(log),0x8|0x80)<0:raise OSError(ctypes.get_errno(),'inotify_add_watch')
                        with (LOG/'controller/scheduling.jsonl').open('a') as stream:stream.write(json.dumps(dict(time=now(),phase='formal' if formal else 'FT',launched=name,observed=observed,reserved_after=observed['conservative_occupied']+slot+1,own_running={str(g):sum(v['gpu']==g for v in running.values()) for g in GPUS}))+'\n')
            if changed:
                b=consumption();write(OUT/'status.json',dict(status='执行中',phase='formal' if formal else 'FT',consumption=b,running=[dict(arm=v['arm'],task=v['task'],fold=v['fold'],gpu=v['gpu']) for v in running.values()],queued=len(queue)))
                print('scheduler', 'formal' if formal else 'FT', 'completed',b['formal_completed' if formal else 'ft_completed'],'running',len(running),'waiting',len(queue),flush=True)
            if queue or running:
                # Completion-driven own jobs; timeout only permits required external resource scheduling.
                ready=select.select([fd],[],[],15 if queue else None)[0]
                if ready:os.read(fd,65536)
        if failures:raise RuntimeError('unit failures retained; no retries: '+str(failures))
    finally:os.close(fd)


def load_predictions(path):
    with np.load(path,allow_pickle=False) as v:return {k:v[k].copy() for k in v.files}

def freeze():
    require_stage('freeze',consumption());target=OUT/'selection_before_test.json'
    if target.exists() or (OUT/'formal').exists():raise FileExistsError('freeze/TEST overwrite risk')
    units=[]
    for arm in ARMS:
        for task in TASKS:
            for fold in range(5):
                folder=OUT/'ft'/arm/task/f'fold{fold}';meta=source_binding(arm,task,fold);cfg=meta['config'];entry=meta['manifest']['folds'][fold]
                m=read(folder/'metrics.json');history=read(folder/'progress.json')['history'];s=read(folder/'scaler.json');count=epochs(task)*math.ceil(len(entry['train_indices'])/24)
                if m['config']!=cfg or m['loss_policy']!=MSE_POLICY or m['status']!='PASS' or m['outer_test']!='NOT_RUN' or m['epochs']!=epochs(task) or m['updates']!=dict(attempted=count,completed=count):raise ValueError('FT completion/config/loss mismatch')
                chosen=selected_history(history)
                if len(history)!=epochs(task) or any(r['recipe']!=cfg['recipe'] or r['loss_policy']!=MSE_POLICY for r in history) or chosen['epoch']!=m['best_epoch'] or chosen['validation_rmse']!=m['rmse'] or chosen['validation_r2']!=m['r2']:
                    raise ValueError('raw neural RMSE best/history mismatch')
                if s!=meta['reference']['scaler']:raise ValueError('frozen train scaler mismatch')
                unit=dict(arm=arm,task=task,fold=fold,seed=42+fold,config=cfg,loss_policy=MSE_POLICY,adaptation_policy=None,parent=str(parent(arm)),head_initial=meta['reference']['head_initial'],
                    best=str(folder/'best.pt'),best_epoch=m['best_epoch'],best_stage=chosen['stage'],scaler=s,expected_updates=count,train_indices=entry['train_indices'],validation_indices=entry['validation_indices'],test_indices=sorted(entry['test_indices']),
                    reference=meta['reference'],fixed_krr=meta['fixed_krr'])
                best=torch.load(folder/'best.pt',map_location='cpu',weights_only=False);validate_checkpoint(best,unit)
                if any(not torch.isfinite(v).all() for v in best['state_dict'].values()):raise FloatingPointError('nonfinite neural best')
                del best
                predictions=load_predictions(folder/'validation_predictions.npz');own=dict(unit,alpha=unit['fixed_krr']['alpha'])
                bound=bind_branch_arrays(predictions,own,'validation',meta['identities'],('y_pred',))
                cached=load_predictions(unit['fixed_krr']['validation']);kr=bind_branch_arrays(cached,own,'validation',meta['identities'],('y_krr',))
                if not np.array_equal(bound['y_true'],kr['y_true']):raise ValueError('validation truth identity mismatch')
                if metrics(bound['y_true'],bound['y_pred'])!=dict(r2=m['r2'],rmse=m['rmse']):raise ValueError('best validation array metric mismatch')
                fusion=blend_arrays(bound['y_pred'],kr['y_krr'],.75)
                unit['validation_diagnostic']=dict(neural=metrics(bound['y_true'],bound['y_pred']),fused=metrics(bound['y_true'],fusion))
                write(folder/'fused_validation_diagnostic.json',dict(metrics=unit['validation_diagnostic'],krr_source=unit['fixed_krr']['validation'],arithmetic_only=True,best_unchanged=True))
                units.append(unit)
    value=dict(policy=POLICY,units=units,frozen=True,frozen_at=now(),budget_at_freeze=consumption(),reference_test_opened=False,source_r10=str(R10/'selection_before_test.json'))
    validate_frozen(value);write(target,value)
    return value


def report():
    frozen=validate_frozen(read(OUT/'selection_before_test.json'));require_stage('CPU',consumption())
    old=read(R10/'final_review/seven_task_results.json');old_rows={(r['task'],r['fold']):r for r in old['folds']}
    review=OUT/'final_review';review.mkdir(exist_ok=False)
    policies=('CONT_REF_neural','CONT_REF_fused','GRAPH_DISTILL_neural','GRAPH_DISTILL_fused','g05','r10')
    pairs=dict(graph_fused_minus_cont=('GRAPH_DISTILL_fused','CONT_REF_fused'),graph_fused_minus_r10=('GRAPH_DISTILL_fused','r10'),
        cont_fused_minus_r10=('CONT_REF_fused','r10'),graph_neural_minus_cont=('GRAPH_DISTILL_neural','CONT_REF_neural'),
        graph_neural_minus_g05=('GRAPH_DISTILL_neural','g05'),cont_neural_minus_g05=('CONT_REF_neural','g05'),
        cont_fused_minus_own_neural=('CONT_REF_fused','CONT_REF_neural'),graph_fused_minus_own_neural=('GRAPH_DISTILL_fused','GRAPH_DISTILL_neural'))
    units={(u['arm'],u['task'],u['fold']):u for u in frozen['units']};rows=[];index=[]
    for task in TASKS:
        for fold in range(5):
            reference=old_rows[(task,fold)];values={};validation={};best={};paths={};truth=None
            for arm in ARMS:
                u=units[(arm,task,fold)];meta=source_binding(arm,task,fold);folder=OUT/'formal'/arm/task/f'fold{fold}';m=read(folder/'metrics.json')
                if m['status']!='PASS' or m['config']!=u['config'] or m['loss_policy']!=MSE_POLICY or m['adaptation_policy'] is not None or m['outer_test']!='RUN' or m['best_epoch']!=u['best_epoch'] or m['neural_test_forward_calls']!=1:raise ValueError('formal policy/epoch/forward mismatch')
                own=dict(u,alpha=u['fixed_krr']['alpha'])
                for kind,file in [('neural','test_predictions.npz'),('fused','fused_predictions.npz')]:
                    label=arm+'_'+kind;path=folder/file;bound=bind_branch_arrays(load_predictions(path),own,'test',meta['identities'],('y_pred',))
                    if truth is None:truth=bound['y_true']
                    if not np.array_equal(truth,bound['y_true']):raise ValueError('paired two-arm TEST truth mismatch')
                    values[label]=metrics(truth,bound['y_pred']);validation[label]=u['validation_diagnostic'][kind];paths[label]=str(path)
                if values[arm+'_neural']!=dict(r2=m['r2'],rmse=m['rmse']) or values[arm+'_fused']!=m['fused']:raise ValueError('formal metric mismatch')
                best[arm]=dict(epoch=u['best_epoch'],stage=u['best_stage'])
                index.append(dict(arm=arm,task=task,fold=fold,parent=u['parent'],head_initial=u['head_initial'],best=u['best'],best_epoch=u['best_epoch'],best_stage=u['best_stage'],fixed_krr=u['fixed_krr'],neural_prediction=paths[arm+'_neural'],fused_prediction=paths[arm+'_fused']))
            for label,path in [('g05',reference['paths']['g05']),('r10',reference['output'])]:
                bound=bind_branch_arrays(load_predictions(path),own,'test',meta['identities'],('y_pred',))
                if not np.array_equal(truth,bound['y_true']):raise ValueError('reference TEST truth mismatch')
                values[label]=metrics(truth,bound['y_pred']);paths[label]=str(path)
            if values['g05']!=reference['metrics']['g05'] or values['r10']!=reference['metrics']['global']:raise ValueError('reference saved metric mismatch')
            rows.append(dict(task=task,fold=fold,metrics=values,validation=validation,best=best,paths=paths,
                differences={name:{m:values[a][m]-values[b][m] for m in ('r2','rmse')} for name,(a,b) in pairs.items()},
                test_minus_validation={p:{m:values[p][m]-validation[p][m] for m in ('r2','rmse')} for p in validation}))
    tasks={}
    for task in TASKS:
        selected=[r for r in rows if r['task']==task];paper=old['tasks'][task]['paper']
        stats={p:{m:dict(mean=float(np.mean([r['metrics'][p][m] for r in selected])),std=float(np.std([r['metrics'][p][m] for r in selected],ddof=0))) for m in ('r2','rmse')} for p in policies}
        differences={name:{m:stats[a][m]['mean']-stats[b][m]['mean'] for m in ('r2','rmse')} for name,(a,b) in pairs.items()}
        counts={}
        for name in pairs:
            gains=[r['differences'][name]['r2'] for r in selected]
            counts[name]=dict(improved=sum(x>0 for x in gains),degraded=sum(x<0 for x in gains),unchanged=sum(x==0 for x in gains),mean_delta_excluding_best_fold=float((sum(gains)-max(gains))/4))
        tasks[task]=dict(stats=stats,differences=differences,fold_counts=counts,paper=paper,minus_paper={p:stats[p]['r2']['mean']-paper['r2'] for p in policies})
    macro={p:float(np.mean([tasks[t]['stats'][p]['r2']['mean'] for t in TASKS])) for p in policies}
    macro_differences={name:macro[a]-macro[b] for name,(a,b) in pairs.items()};concentration={}
    for name in pairs:
        best_task=max(TASKS,key=lambda t:tasks[t]['differences'][name]['r2']);best_fold=max(rows,key=lambda r:r['differences'][name]['r2'])
        concentration[name]=dict(best_task=best_task,best_unit=[best_fold['task'],best_fold['fold']],macro_excluding_best_task=float(np.mean([tasks[t]['differences'][name]['r2'] for t in TASKS if t!=best_task])),mean_excluding_best_unit=float((sum(r['differences'][name]['r2'] for r in rows)-best_fold['differences'][name]['r2'])/34),
            degraded_tasks=[t for t in TASKS if tasks[t]['differences'][name]['r2']<0],improved_units=sum(r['differences'][name]['r2']>0 for r in rows),degraded_units=sum(r['differences'][name]['r2']<0 for r in rows))
    milestones=dict(mechanism=macro_differences['graph_fused_minus_cont']>=.010,project=macro_differences['graph_fused_minus_r10']>=.010)
    result=dict(policy=POLICY,tasks=tasks,macro_r2=macro,macro_differences=macro_differences,milestones=milestones,concentration=concentration,
        reached_paper={p:sum(tasks[t]['minus_paper'][p]>=0 for t in TASKS) for p in policies},folds=rows,
        best_stage_counts={a:{stage:sum(u['arm']==a and u['best_stage']==stage for u in frozen['units']) for stage in ('head','joint')} for a in ARMS},
        consumption=consumption(),aggregation='5 folds, seed42+fold, std ddof=0; equal seven-task R2 mean; no macro RMSE',protocol='development-exposed validation/TEST; paper protocol equivalence unverified',independent_review='NOT_PERFORMED')
    write(review/'seven_task_results.json',result);write(review/'model_prediction_index.json',index)
    import csv
    records=[]
    for r in rows:
        row=dict(task=r['task'],fold=r['fold'])
        for a,b in r['best'].items():row.update({a+'_best_epoch':b['epoch'],a+'_best_stage':b['stage']})
        for group in ('differences','test_minus_validation'):
            for name,vals in r[group].items():
                for m,v in vals.items():row[group+'_'+name+'_'+m]=v
        records.append(row)
    with (review/'fold_differences.csv').open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    lines=['状态：待审查；同一主体执行与自检，独立审查未进行。',
        'CONT_REF／GRAPH_DISTILL均从同一G05完整resume_05000接续5000步至累计10000，恢复Adam／两rank RNG／数据游标／原20000步LR轨迹；两臂均计算固定干净G05 teacher。唯一差异为graph恢复损失权重0与0.1×min(新增update/1000,1)。Student未增加参数。',
        '两臂分别七任务五折原R0全参数MSE，从各自固定累计10000 deploy与原head_initial开始；以原单位neural validation RMSE严格改善选best。全部70项全局冻结后才读取TEST；统一0.75新神经＋0.25 r10固定radius2 KRR／α。',
        'Teacher额外训练呈现10,080,000、validation呈现6,144；非零额外计算／内存，完整PT各步耗时及GPU峰值见pretrain日志。', '']
    for metric in ('r2','rmse'):
        lines+=['|任务|'+'|'.join(p+' '+('R²' if metric=='r2' else 'RMSE')+'±std' for p in policies)+'|','|---|'+'---:|'*len(policies)]
        for t in TASKS:lines.append('|'+t+'|'+'|'.join(f"{tasks[t]['stats'][p][metric]['mean']:.3f}±{tasks[t]['stats'][p][metric]['std']:.3f}" for p in policies)+'|')
        lines.append('')
    lines+=['|政策|Macro R²|论文数值达标数|','|---|---:|---:|']+[f"|{p}|{macro[p]:.3f}|{result['reached_paper'][p]}/7|" for p in policies]
    lines+=['','|比较|Δmacro R²|改善／退化单元|退化任务|去掉最大增益任务后差|去掉最大增益fold后差|','|---|---:|---:|---|---:|---:|']
    for name in pairs:
        c=concentration[name];lines.append(f"|{name}|{macro_differences[name]:+.3f}|{c['improved_units']}／{c['degraded_units']}|{c['degraded_tasks']}|{c['macro_excluding_best_task']:+.3f}|{c['mean_excluding_best_unit']:+.3f}|")
    lines+=['',f"机制+0.010门槛：{'达到' if milestones['mechanism'] else '未达到'}；项目+0.010门槛：{'达到' if milestones['project'] else '未达到'}。两者独立判定，均不自动晋级／部署。",'',
        '|任务|论文最佳模型|论文R²|'+'|'.join(p+'−论文' for p in policies)+'|','|---|---|---:|'+'---:|'*len(policies)]
    for t in TASKS:
        r=tasks[t];lines.append(f"|{t}|{r['paper']['model']}|{r['paper']['r2']:.3f}|"+'|'.join(f"{r['minus_paper'][p]:+.3f}" for p in policies)+'|')
    lines+=['','逐任务两类主差、各自神经−G05、退化与改善折数及去掉最大增益折的差见seven_task_results.json。全部逐fold差、validation→TEST差及best epoch／阶段见fold_differences.csv；所有模型／预测路径见model_prediction_index.json。',
        '融合−自身神经仅为预测政策差，不称interaction。每任务原单位RMSE单独报告，负差为改善，不跨单位平均。先原精度统计／判定后展示3位小数，std ddof=0。共享validation／TEST已开发暴露，论文数据／协议等价性未核实，仅探索性数值对照，不称独立盲测、稳定优势或公平超过论文。']
    (review/'delivery.md').write_text('\n'.join(lines)+'\n');return '\n'.join(lines)+'\n'



def update_plan(status,body):
    p=ROOT/'CURRENT_PLAN.md';s=p.read_text().replace('状态：**执行中（r15）**。','状态：**'+status+'（r15）**。',1)
    s=s.replace('状态：**执行中**。已收到当前用户第71—72节完整实现／实验授权；','状态：**'+status+'**。已收到当前用户第71—72节完整实现／实验授权；',1)
    p.write_text(s+'\n\n## 73. r15实际执行交接\n\n'+body+'\n')


def run_pt(arm):
    # Only required capacity scheduling; no separate monitoring process.
    while True:
        slots=[gpu_slots(g,{})[0] for g in GPUS]
        if slots==[4,4]:break
        time.sleep(15)
    write(OUT/'status.json',dict(status='执行中',phase='PT',arm=arm,consumption=consumption()))
    ranks=LOG/('pt_'+arm)/'ranks'
    argv=[PYTHON,'-m','torch.distributed.run','--standalone','--nnodes=1','--nproc_per_node=2','--max_restarts=0',
        '--log_dir',str(ranks),'--redirects=3','--tee=3','scripts/pretrain_fce_graph_distill.py','--arm',arm]
    job=launch('pt_'+arm,argv,gpu='2,3')
    if wait_original(job):raise RuntimeError('PT '+arm+' failed; no retry or next arm')
    for rank in (0,1):
        if read(OUT/'pretrain'/arm/f'execution_rank{rank}.json')['exit_code']!=0:raise RuntimeError('PT rank failed')
    validate_parent(torch.load(parent(arm),map_location='cpu',weights_only=False),arm)
    consumption()


def campaign():
    require_tmux();started=now()
    try:
        source=read(R10/'selection_before_test.json');units=source['r9_freeze']['units']
        planned_epochs=2*sum(epochs(u['task']) for u in units);planned_updates=2*sum(epochs(u['task'])*math.ceil(len(u['train_indices'])/24) for u in units)
        if len(units)!=35 or (planned_epochs,planned_updates)!=(4800,102900):raise ValueError('source FT budget differs from authorization')
        write(OUT/'planned_budget.json',LIMITS)
        for arm in ARMS:
            write(OUT/'pt_config'/f'{arm}.json',pt_config_for(arm));run_pt(arm)
        require_stage('FT',consumption())
        initialize=launch('initialize',[PYTHON,'scripts/run_fce_graph_distill_unit.py','--initialize'])
        if wait_original(initialize):raise RuntimeError('parent/reference initialization failed; no FT dispatch')
        matrix();require_stage('freeze',consumption());freeze();require_stage('formal',consumption());matrix(True)
        require_stage('CPU',consumption());delivery=report()
        test=launch('final_cpu',['timeout','300s',PYTHON,'-m','pytest','-q','tests/test_fce_graph_distill.py']);rc=wait_original(test)
        status='待审查' if rc==0 else '需返修';b=consumption()
        write(OUT/'completion.json',dict(status=status,start=started,end=now(),test_exit=rc,consumption=b,independent_review='NOT_PERFORMED'))
        update_plan(status,delivery+f'\n唯一CPU检查真实exit {rc}；预算'+json.dumps(b,ensure_ascii=False)+'。\n')
        write(OUT/'status.json',dict(status=status,phase='finished',consumption=b))
    except BaseException as error:
        try:b=consumption()
        except BaseException:b=read(OUT/'consumption.json') if (OUT/'consumption.json').exists() else {}
        write(OUT/'campaign_failure.json',dict(status='需返修',error=repr(error),traceback=traceback.format_exc(),time=now(),consumption=b))
        update_plan('需返修','异常停止：'+repr(error)+'；现场campaign_failure.json，不自动重跑或扩预算；不完整矩阵禁止正式评价。七任务均未完成／不可比较，未运行最终CPU测试。');raise


def setup():
    require_tmux()
    if OUT.exists() or LOG.exists():raise FileExistsError('new r15 roots must not exist; no replay')
    OUT.mkdir(parents=True);LOG.mkdir(parents=True);folder=LOG/'setup';folder.mkdir();started=now();rc=1
    write(folder/'command.json',dict(argv=sys.argv,cwd=str(ROOT),environment={k:os.environ.get(k) for k in ('CUDA_VISIBLE_DEVICES','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')},window='fce_graph_distill_setup'))
    (folder/'stdout.log').touch();(folder/'stderr.log').touch()
    try:
        for name in ('CURRENT_PLAN.md','src/modules/glt_dual_pretrain.py'):
            target=OUT/'source_before'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(Path('/tmp/fce_graph_distill_before_20261016')/name,target)
        for name in (*EDITED,'src/training/fce_param.py','src/dataset/glt_dual_pretrain.py','src/training/fce_kernel.py','scripts/run_fce_tune_t3b_unit.py','scripts/finetune_mcl_ph.py','src/utils.py','src/modules/glt_adaptation.py','src/training/glt_dual_runtime.py'):
            target=OUT/'source_used'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,target)
        launch('controller',[PYTHON,'scripts/run_fce_graph_distill.py','--controller']);rc=0
    except BaseException:
        with (folder/'stderr.log').open('a') as stream:traceback.print_exc(file=stream)
        raise
    finally:write(folder/'execution.json',dict(start=started,end=now(),exit_code=rc));(folder/'exit.txt').write_text(str(rc)+'\n')


def finish_delivery():
    require_tmux();review=OUT/'final_review';review.mkdir(exist_ok=True);records=[];intervals=[]
    for folder in sorted(LOG.iterdir()):
        if not folder.is_dir() or folder.name=='finish_delivery':continue
        record=read(folder/'execution.json') if (folder/'execution.json').exists() else dict(start=None,end=None,exit_code=None)
        record.update(log_dir=str(folder),exit_txt=int((folder/'exit.txt').read_text()) if (folder/'exit.txt').exists() else None);records.append(record)
        if folder.name.startswith(('ft_','test_')) and record.get('start') and record.get('end'):
            intervals.append(dict(gpu=record['gpu'],start=record['start'],end=record['end'],unit=folder.name))
    def peak(gpu=None):
        points=[]
        for v in intervals:
            if gpu is not None and v['gpu']!=gpu:continue
            points.extend([(datetime.datetime.fromisoformat(v['start']),1),(datetime.datetime.fromisoformat(v['end']),-1)])
        total=high=0
        for _,d in sorted(points,key=lambda x:(x[0],x[1])):total+=d;high=max(high,total)
        return high
    resource=dict(own_interval_peak={str(g):peak(g) for g in GPUS},own_total_peak=peak(),combined_observations=str(LOG/'controller/scheduling.jsonl'),note='own interval peaks; existing contexts and pending reservations included at dispatch, no preemption')
    write(review/'execution_exits.json',dict(records=records,resource=resource,own_finish_exit=str(LOG/'finish_delivery/exit.txt'),time=now()))
    for arm in ARMS:
        for rank in (0,1):
            path=OUT/'pretrain'/arm/f'execution_rank{rank}.json'
            records.append(dict(read(path) if path.exists() else dict(exit_code=None,note='未运行或未核实'),stage='PT rank',arm=arm,rank=rank))
    write(review/'execution_exits.json',dict(records=records,resource=resource,own_finish_exit=str(LOG/'finish_delivery/exit.txt'),time=now()))
    if not (review/'delivery.md').exists():
        (review/'delivery.md').write_text('状态：需返修；'+json.dumps(read(OUT/'campaign_failure.json'),ensure_ascii=False)+'\n七任务Eea、Egb、Egc、Ei、EPS、Nc、Xc：未完成／不可比较。\n')
    p=ROOT/'CURRENT_PLAN.md';s=p.read_text();s+='\n\nr15真实退出索引final_review/execution_exits.json；收尾自身真实exit见logs/fce_tune_20261016/graph_distill/finish_delivery/exit.txt，写本索引时不伪造尚未退出状态。实际并发记录：'+json.dumps(resource,ensure_ascii=False)+'。\n'
    s+='结果results/fce_tune_20261016/graph_distill/，日志logs/fce_tune_20261016/graph_distill/；tmux Uni-Poly:fce_graph_distill_*，增量final_review/graph_distill_only_changes.diff。无删除，无新增哈希验证，无模型历史／SOTA表改动，独立审查未进行。\n';p.write_text(s)
    for name in EDITED:
        target=OUT/'source_final'/name;target.parent.mkdir(parents=True,exist_ok=True)
        with target.open('xb') as stream:stream.write((ROOT/name).read_bytes())
    chunks=[]
    for name in EDITED:
        before=OUT/'source_before'/name;old=before.read_text().splitlines(True) if before.exists() else []
        chunks.extend(difflib.unified_diff(old,(ROOT/name).read_text().splitlines(True),fromfile='before/'+name,tofile='after/'+name))
    (review/'graph_distill_only_changes.diff').write_text(''.join(chunks))
    result=subprocess.run(['git','diff','--check','--','CURRENT_PLAN.md','src/modules/glt_dual_pretrain.py'],capture_output=True,text=True)
    write(review/'diff_format_check.json',dict(exit_code=result.returncode,stdout=result.stdout,stderr=result.stderr))
    print('exit index and incremental diff delivered',resource,flush=True)


if __name__=='__main__':
    require_tmux();p=argparse.ArgumentParser();p.add_argument('--setup',action='store_true');p.add_argument('--controller',action='store_true');p.add_argument('--finish-delivery',action='store_true');p.add_argument('--start-finish',action='store_true');a=p.parse_args()
    if a.setup:setup()
    elif a.controller:campaign()
    elif a.start_finish:launch('finish_delivery',[PYTHON,'scripts/run_fce_graph_distill.py','--finish-delivery'])
    elif a.finish_delivery:finish_delivery()
    else:p.error('explicit role required')
