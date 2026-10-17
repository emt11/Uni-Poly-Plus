"""FP graph-f1 bounded event-driven neural FT/formal orchestration; no retries."""
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
from scripts.run_fp_graph_fusion_unit import (ARMS,TASKS,OUT,LOG,R10,POLICY,read,write,
    config_for,epochs,source_binding,validate_checkpoint,validate_frozen,selected_history)
from scripts.run_fp_graph_fusion_unit import MSE_POLICY
from src.training.fp_graph_fusion import parent,PARENT,MODULE_INITIAL,input_policy,REVISION,validate_initial
from src.training.fce_kernel import bind_branch_arrays,metrics
import sqlite3
from src.training.glt_dual_runtime import require_tmux
PYTHON='/opt/conda/bin/python';GPUS=(2,3)
EDITED=('CURRENT_PLAN.md','src/modules/fp_graph_fusion.py','src/training/fp_graph_fusion.py','scripts/run_fp_graph_fusion.py','scripts/run_fp_graph_fusion_unit.py','tests/test_fp_graph_fusion.py')
LIMITS=dict(pt=0,pt_validation=0,endpoint=0,teacher_forward=0,teacher_updates=0,
    ft_starts=105,ft_completed=105,ft_epochs=7200,ft_attempted_updates=154350,ft_completed_updates=154350,
    validation_calls=7200,formal_starts=105,formal_completed=105,formal_outputs=105,
    materialization_ft_attempts=70500,materialization_ft_completed=70500,materialization_test_attempts=17625,materialization_test_completed=17625,
    fp_unique_generated=5875,fp_cache_hits=88125,cpu_test_starts=1,krr_fit=0,krr_forward=0,new_conformers=0,ft_workers=0,extra_seeds=0,refit=0)


def now():return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
def check_budget(b):
    if set(b)!=set(LIMITS) or any(type(v) is not int or not 0<=v<=LIMITS[k] for k,v in b.items()):raise ValueError('FP graph hard budget mismatch')
def require_stage(stage,b):
    check_budget(b)
    if stage in ('freeze','formal','CPU') and any(b[k]!=v for k,v in dict(ft_completed=105,ft_epochs=7200,ft_attempted_updates=154350,ft_completed_updates=154350,validation_calls=7200,materialization_ft_attempts=70500,materialization_ft_completed=70500).items()):raise ValueError('complete 105 FT matrix and materialization required')
    if stage=='CPU' and any(b[k]!=v for k,v in dict(formal_completed=105,formal_outputs=105,materialization_test_attempts=17625,materialization_test_completed=17625).items()):raise ValueError('complete formal matrix/materialization required for CPU test')
def consumption():
    b={k:0 for k in LIMITS};b['ft_starts']=len(list(LOG.glob('ft_*/command.json')));b['formal_starts']=len(list(LOG.glob('test_*/command.json')));b['cpu_test_starts']=len(list(LOG.glob('final_cpu/command.json')))
    for f in (OUT/'ft').glob('*/*/fold*'):
        if (f/'updates.json').exists():
            u=read(f/'updates.json');b['ft_attempted_updates']+=u['attempted'];b['ft_completed_updates']+=u['completed']
        if (f/'progress.json').exists():b['ft_epochs']+=len(read(f/'progress.json')['history'])
        b['ft_completed']+=int((f/'metrics.json').exists())
    for stage,key in [('ft','ft'),('formal','test')]:
        for p in (OUT/stage).glob('*/*/fold*/input_consumption.json'):
            r=read(p);b['materialization_'+key+'_attempts']+=r['attempted_rows'];b['materialization_'+key+'_completed']+=r['materialized_graphs']
    b['validation_calls']=b['ft_epochs'];b['formal_completed']=len(list((OUT/'formal').glob('*/*/fold*/metrics.json')))
    b['formal_outputs']=len(list((OUT/'formal').glob('*/*/fold*/*predictions.npz')))
    if (OUT/'fingerprints.sqlite').exists():
        with sqlite3.connect('file:'+str(OUT/'fingerprints.sqlite')+'?mode=ro',uri=True,timeout=120) as db:
            counts=db.execute('SELECT generated,hits FROM counts WHERE id=1').fetchone()
        if counts:b['fp_unique_generated'],b['fp_cache_hits']=map(int,counts)
    write(OUT/'consumption.json',b);check_budget(b);return b



def launch(name,argv,gpu=None):
    require_tmux()
    if gpu is not None and gpu not in (*GPUS,"2,3"):raise ValueError('GPU2/3 only')
    folder=LOG/name;folder.mkdir(parents=True,exist_ok=False)
    environment=dict(CUDA_VISIBLE_DEVICES='' if gpu is None else str(gpu),OMP_NUM_THREADS='1' if gpu is None else '4',
        OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1')
    command=dict(argv=argv,command=shlex.join(argv),cwd=str(ROOT),environment=environment,gpu=gpu,window='fp_graph_f1_'+name)
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
if f.name=='controller':subprocess.run([sys.executable,'scripts/run_fp_graph_fusion.py','--start-finish'],cwd=j['cwd'],env=dict(os.environ,**j['environment']),check=False)
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
                        if b[key]>=105:raise ValueError('start budget exhausted')
                        argv=[PYTHON,'scripts/run_fp_graph_fusion_unit.py','--arm',arm,'--task',task,'--fold',str(fold)]+(['--formal'] if formal else [])
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
                if len(history)!=epochs(task) or any(r['recipe']!=cfg['recipe'] or r['loss_policy']!=MSE_POLICY for r in history) or chosen['epoch']!=m['best_epoch'] or chosen['validation_rmse']!=m['rmse'] or chosen['validation_r2']!=m['r2']:raise ValueError('raw neural RMSE best/history mismatch')
                if s!=meta['reference']['scaler']:raise ValueError('frozen train scaler mismatch')
                unit=dict(arm=arm,task=task,fold=fold,seed=42+fold,config=cfg,loss_policy=MSE_POLICY,adaptation_policy=None,input_policy=input_policy(arm),module_initial=str(MODULE_INITIAL),fp_graph_revision=REVISION,optimizer_groups=read(folder/'parameter_groups.json'),gradient_paths=read(folder/'gradient_paths.json'),parent=str(parent(arm)),head_initial=meta['reference']['head_initial'],
                    best=str(folder/'best.pt'),best_epoch=m['best_epoch'],best_stage=chosen['stage'],scaler=s,expected_updates=count,train_indices=entry['train_indices'],validation_indices=entry['validation_indices'],test_indices=sorted(entry['test_indices']),reference=meta['reference'],reference_r10=meta['reference_r10'])
                best=torch.load(folder/'best.pt',map_location='cpu',weights_only=False);validate_checkpoint(best,unit)
                if any(not torch.isfinite(v).all() for v in best['state_dict'].values()):raise FloatingPointError('nonfinite best')
                del best
                bound=bind_branch_arrays(load_predictions(folder/'validation_predictions.npz'),unit,'validation',meta['identities'],('y_pred',))
                values=metrics(bound['y_true'],bound['y_pred'])
                if values!=dict(r2=m['r2'],rmse=m['rmse']):raise ValueError('best validation array metric mismatch')
                unit['validation_diagnostic']=values;units.append(unit)
    value=dict(policy=POLICY,units=units,frozen=True,frozen_at=now(),budget_at_freeze=consumption(),reference_test_opened=False,source_r10=str(R10/'selection_before_test.json'))
    validate_frozen(value);write(target,value);return value


def aggregate(rows,papers):
    policies=(*ARMS,'g05','r10');pairs={'real_minus_null':('FP_REAL_KFUSE','FP_NULL_KFUSE'),'real_minus_mean':('FP_REAL_KFUSE','FP_REAL_MEAN')}
    for a in ARMS:
        for b in ('g05','r10'):pairs[a+'_minus_'+b]=(a,b)
    for r in rows:
        r['differences']={name:{m:r['metrics'][a][m]-r['metrics'][b][m] for m in ('r2','rmse')} for name,(a,b) in pairs.items()}
        r['test_minus_validation']={p:{m:r['metrics'][p][m]-r['validation'][p][m] for m in ('r2','rmse')} for p in r['validation']}
    tasks={}
    for task in TASKS:
        selected=[r for r in rows if r['task']==task]
        if len(selected)!=5 or {r['fold'] for r in selected}!=set(range(5)):raise ValueError('five complete folds required for aggregate')
        stats={p:{m:dict(mean=float(np.mean([r['metrics'][p][m] for r in selected])),std=float(np.std([r['metrics'][p][m] for r in selected],ddof=0))) for m in ('r2','rmse')} for p in policies}
        differences={name:{m:stats[a][m]['mean']-stats[b][m]['mean'] for m in ('r2','rmse')} for name,(a,b) in pairs.items()};counts={}
        for name in pairs:
            gains=[r['differences'][name]['r2'] for r in selected]
            counts[name]=dict(improved=sum(x>0 for x in gains),degraded=sum(x<0 for x in gains),unchanged=sum(x==0 for x in gains),mean_delta_excluding_best_fold=float((sum(gains)-max(gains))/4))
        tasks[task]=dict(stats=stats,differences=differences,fold_counts=counts,paper=papers[task],minus_paper={p:stats[p]['r2']['mean']-papers[task]['r2'] for p in policies})
    macro={p:float(np.mean([tasks[t]['stats'][p]['r2']['mean'] for t in TASKS])) for p in policies};delta={name:macro[a]-macro[b] for name,(a,b) in pairs.items()};concentration={}
    for name in pairs:
        best_task=max(TASKS,key=lambda t:tasks[t]['differences'][name]['r2']);best_fold=max(rows,key=lambda r:r['differences'][name]['r2'])
        concentration[name]=dict(best_task=best_task,best_unit=[best_fold['task'],best_fold['fold']],macro_excluding_best_task=float(np.mean([tasks[t]['differences'][name]['r2'] for t in TASKS if t!=best_task])),mean_excluding_best_unit=float((sum(r['differences'][name]['r2'] for r in rows)-best_fold['differences'][name]['r2'])/34),degraded_tasks=[t for t in TASKS if tasks[t]['differences'][name]['r2']<0],improved_units=sum(r['differences'][name]['r2']>0 for r in rows),degraded_units=sum(r['differences'][name]['r2']<0 for r in rows))
    return dict(tasks=tasks,macro_r2=macro,macro_differences=delta,concentration=concentration,
        milestones=dict(mechanism=delta['real_minus_null']>=.010,project=delta['FP_REAL_KFUSE_minus_r10']>=.010),
        reached_paper={p:sum(tasks[t]['minus_paper'][p]>=0 for t in TASKS) for p in policies},folds=rows)


def report(status):
    frozen=validate_frozen(read(OUT/'selection_before_test.json'));require_stage('CPU',consumption())
    old=read(R10/'final_review/seven_task_results.json');old_rows={(r['task'],r['fold']):r for r in old['folds']}
    review=OUT/'final_review';review.mkdir(exist_ok=False);policies=(*ARMS,'g05','r10')
    units={(u['arm'],u['task'],u['fold']):u for u in frozen['units']};rows=[];index=[]
    for task in TASKS:
        for fold in range(5):
            reference=old_rows[(task,fold)];values={};validation={};best={};paths={};truth=None
            for arm in ARMS:
                u=units[(arm,task,fold)];meta=source_binding(arm,task,fold);folder=OUT/'formal'/arm/task/f'fold{fold}';m=read(folder/'metrics.json')
                if m['status']!='PASS' or m['config']!=u['config'] or m['loss_policy']!=MSE_POLICY or m['adaptation_policy'] is not None or m['outer_test']!='RUN' or m['best_epoch']!=u['best_epoch'] or m['neural_test_forward_calls']!=1:raise ValueError('formal policy/epoch/forward mismatch')
                path=folder/'test_predictions.npz';bound=bind_branch_arrays(load_predictions(path),u,'test',meta['identities'],('y_pred',))
                if truth is None:truth=bound['y_true']
                if not np.array_equal(truth,bound['y_true']):raise ValueError('paired TEST truth mismatch')
                values[arm]=metrics(truth,bound['y_pred']);validation[arm]=u['validation_diagnostic'];paths[arm]=str(path)
                if values[arm]!=dict(r2=m['r2'],rmse=m['rmse']):raise ValueError('formal metric mismatch')
                best[arm]=dict(epoch=u['best_epoch'],stage=u['best_stage'])
                index.append(dict(arm=arm,task=task,fold=fold,parent=u['parent'],head_initial=u['head_initial'],module_initial=u['module_initial'],best=u['best'],best_epoch=u['best_epoch'],best_stage=u['best_stage'],neural_prediction=str(path)))
            for label,path in [('g05',reference['paths']['g05']),('r10',reference['output'])]:
                bound=bind_branch_arrays(load_predictions(path),dict(u,alpha=u['reference_r10']['alpha']),'test',meta['identities'],('y_pred',))
                if not np.array_equal(truth,bound['y_true']):raise ValueError('reference TEST truth mismatch')
                values[label]=metrics(truth,bound['y_pred']);paths[label]=str(path)
            if values['g05']!=reference['metrics']['g05'] or values['r10']!=reference['metrics']['global']:raise ValueError('reference metric mismatch')
            rows.append(dict(task=task,fold=fold,metrics=values,validation=validation,best=best,paths=paths))
    result=aggregate(rows,{t:old['tasks'][t]['paper'] for t in TASKS});result.update(policy=POLICY,consumption=consumption(),aggregation='5 folds seed42+fold; std ddof0; seven-task equal macro R2, no macro RMSE',protocol='development-exposed validation/TEST; paper protocol equivalence unverified',independent_review='NOT_PERFORMED')
    write(review/'seven_task_results.json',result);write(review/'model_prediction_index.json',index)
    availability=[];groups=[]
    for u in frozen['units']:
        a,t,f=u['arm'],u['task'],u['fold'];groups.append(dict(arm=a,task=t,fold=f,groups=u['optimizer_groups'],gradient_paths=u['gradient_paths']))
        for stage in ('ft','formal'):availability.append(dict(arm=a,task=t,fold=f,stage=stage,**read(OUT/stage/a/t/f'fold{f}'/'input_consumption.json')))
    write(review/'input_availability.json',availability);write(review/'parameter_groups.json',groups)
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
    tasks=result['tasks'];lines=['状态：'+status+'；同一主体执行与自检，独立审查未进行。',
        'FP-GRAPH-20261017 / f1：三臂同原G05 step5000、原fold head及隔离CPU seed20261017共用融合初态；新增2033152、总41909496、head2558465参数。三个V全零；self／真实incident geometry／FP三候选，geometry含bias整体屏蔽。',
        '唯一主候选FP_REAL_KFUSE；FP_NULL_KFUSE全0.5，FP_REAL_MEAN严格有效模态均匀；三臂均计算Q/K/V。Morgan定义同旧radius2/2048 binary，输入按自身L2归一化。七任务五折原R0 MSE，head新融合与回归器一起训练、encoder冻结在线前向且禁用旧head cache；joint全参数。',
        '全部105项best/scaler/政策/parent/head/模块初态绑定全局冻结后才读本周期TEST。105正式神经单元、105新数组；G05/r10只引用已有结果，无新KRR或候选后融合。','']
    for metric in ('r2','rmse'):
        lines+=['|任务|'+'|'.join(p+' '+('R²' if metric=='r2' else 'RMSE')+'±std' for p in policies)+'|','|---|'+'---:|'*len(policies)]
        for t in TASKS:lines.append('|'+t+'|'+'|'.join(f"{tasks[t]['stats'][p][metric]['mean']:.3f}±{tasks[t]['stats'][p][metric]['std']:.3f}" for p in policies)+'|')
        lines.append('')
    lines+=['|政策|Macro R²|论文数值达标数|','|---|---:|---:|']+[f"|{p}|{result['macro_r2'][p]:.3f}|{result['reached_paper'][p]}/7|" for p in policies]
    lines+=['','|比较|Δmacro R²|改善／退化单元|退化任务|去掉最大增益任务后差|去掉最大增益fold后差|','|---|---:|---:|---|---:|---:|']
    for name,d in result['macro_differences'].items():
        c=result['concentration'][name];lines.append(f"|{name}|{d:+.3f}|{c['improved_units']}／{c['degraded_units']}|{c['degraded_tasks']}|{c['macro_excluding_best_task']:+.3f}|{c['mean_excluding_best_unit']:+.3f}|")
    milestones=result['milestones'];lines+=['',f"主候选相对NULL机制+0.010门槛：{'达到' if milestones['mechanism'] else '未达到'}；相对r10项目+0.010门槛：{'达到' if milestones['project'] else '未达到'}。MEAN对照单独报告，不按TEST切换主候选，不自动晋级或追加候选。",'',
        '|任务|论文最佳模型|论文R²|'+'|'.join(p+'−论文' for p in policies)+'|','|---|---|---:|'+'---:|'*len(policies)]
    for t in TASKS:
        r=tasks[t];lines.append(f"|{t}|{r['paper']['model']}|{r['paper']['r2']:.3f}|"+'|'.join(f"{r['minus_paper'][p]:+.3f}" for p in policies)+'|')
    lines+=['','逐任务及逐折退化、改善折数、validation→TEST差、去最大增益任务/fold统计见seven_task_results.json及fold_differences.csv；105模型/数组索引、输入计算与缓存命中、参数组及梯度路径分别见model_prediction_index.json、input_availability.json、parameter_groups.json。',
        '预算：'+json.dumps(result['consumption'],ensure_ascii=False),
        '共享validation／TEST已开发暴露，仅探索性数值比较；论文协议等价性未核实，不宣称公平比较或独立盲测。原精度计算后3位小数，std ddof0；RMSE为各任务原单位，负差为改善，不跨单位平均。attention不作为因果贡献，融合差不称interaction。第7节PT及其他候选未启动。']
    (review/'delivery.md').write_text('\n'.join(lines)+'\n');return '\n'.join(lines)+'\n'


def update_plan(status,body):
    p=ROOT/'CURRENT_PLAN.md';s=p.read_text().replace('状态：**执行中（FP_GRAPH_F1）**。','状态：**'+status+'（FP_GRAPH_F1）**。')
    p.write_text(s+'\n\n## 80. FP-GRAPH首轮实际执行交接\n\n'+body+'\n')


def campaign():
    require_tmux();started=now()
    try:
        source=read(R10/'selection_before_test.json');units=source['r9_freeze']['units']
        planned_epochs=3*sum(epochs(u['task']) for u in units);planned_updates=3*sum(epochs(u['task'])*math.ceil(len(u['train_indices'])/24) for u in units)
        n_ft=3*sum(len(u['train_indices'])+len(u['validation_indices']) for u in units);n_test=3*sum(len(u['test_indices']) for u in units)
        if len(units)!=35 or (planned_epochs,planned_updates,n_ft,n_test)!=(7200,154350,70500,17625):raise ValueError('source budgets differ from authorization')
        write(OUT/'planned_budget.json',LIMITS)
        initialize=launch('initialize',[PYTHON,'scripts/run_fp_graph_fusion_unit.py','--initialize'])
        if wait_original(initialize):raise RuntimeError('G05/module/reference initialization failed; no FT dispatch')
        matrix();require_stage('freeze',consumption());freeze();require_stage('formal',consumption());matrix(True)
        require_stage('CPU',consumption())
        test=launch('final_cpu',['timeout','300s',PYTHON,'-m','pytest','-q','tests/test_fp_graph_fusion.py']);rc=wait_original(test)
        status='待审查' if rc==0 else '需返修';b=consumption();delivery=report(status)
        write(OUT/'completion.json',dict(status=status,start=started,end=now(),test_exit=rc,consumption=b,independent_review='NOT_PERFORMED'))
        update_plan(status,delivery+f'\n唯一CPU检查真实exit {rc}；预算'+json.dumps(b,ensure_ascii=False)+'。\n')
        write(OUT/'status.json',dict(status=status,phase='finished',consumption=b))
    except BaseException as error:
        try:b=consumption()
        except BaseException:b=read(OUT/'consumption.json') if (OUT/'consumption.json').exists() else {}
        write(OUT/'campaign_failure.json',dict(status='需返修',error=repr(error),traceback=traceback.format_exc(),time=now(),consumption=b))
        update_plan('需返修','异常停止：'+repr(error)+'；现场campaign_failure.json，不自动重跑或扩预算。不完整候选矩阵不以历史结果补齐；未完成任务标未完成／不可比较。');raise


def setup():
    require_tmux()
    if OUT.exists() or LOG.exists():raise FileExistsError('new FP graph f1 roots must not exist; no replay')
    OUT.mkdir(parents=True);LOG.mkdir(parents=True);folder=LOG/'setup';folder.mkdir();started=now();rc=1
    write(folder/'command.json',dict(argv=sys.argv,cwd=str(ROOT),environment={k:os.environ.get(k) for k in ('CUDA_VISIBLE_DEVICES','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')},window='fp_graph_f1_setup'))
    (folder/'stdout.log').touch(exist_ok=True);(folder/'stderr.log').touch(exist_ok=True)
    try:
        for name in ('CURRENT_PLAN.md',):
            target=OUT/'source_before'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(Path('/tmp/fp_graph_f1_before_20261017')/name,target)
        for name in (*EDITED,'FP_GRAPH_FUSION_DESIGN.md','src/modules/glt_dual_pretrain.py','src/modules/original_mips_knowledge_fusion.py','src/training/xc_kernel.py','src/modules/glt_dual.py','src/dataset/glt_dual.py','src/training/fce_param.py','src/dataset/glt_dual_pretrain.py','src/training/fce_kernel.py','scripts/run_fce_tune_t3b_unit.py','scripts/finetune_mcl_ph.py','src/utils.py','src/modules/glt_adaptation.py','src/training/glt_dual_runtime.py'):
            target=OUT/'source_used'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,target)
        launch('controller',[PYTHON,'scripts/run_fp_graph_fusion.py','--controller']);rc=0
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
    if not (review/'delivery.md').exists():
        (review/'delivery.md').write_text('状态：需返修；'+json.dumps(read(OUT/'campaign_failure.json'),ensure_ascii=False)+'\n七任务Eea、Egb、Egc、Ei、EPS、Nc、Xc：未完成／不可比较。\n')
    p=ROOT/'CURRENT_PLAN.md';s=p.read_text();s+='\n\nFP graph f1真实退出索引final_review/execution_exits.json；收尾自身真实exit见logs/fp_graph_fusion_20261017/f1/finish_delivery/exit.txt，写本索引时不伪造尚未退出状态。实际并发记录：'+json.dumps(resource,ensure_ascii=False)+'。\n'
    s+='结果results/fp_graph_fusion_20261017/f1/，日志logs/fp_graph_fusion_20261017/f1/；tmux Uni-Poly:fp_graph_f1_*，增量final_review/fp_graph_f1_only_changes.diff。无删除，无新增哈希验证，无模型历史／SOTA表改动，独立审查未进行。\n';p.write_text(s)
    for name in EDITED:
        target=OUT/'source_final'/name;target.parent.mkdir(parents=True,exist_ok=True)
        with target.open('xb') as stream:stream.write((ROOT/name).read_bytes())
    chunks=[]
    for name in EDITED:
        before=OUT/'source_before'/name;old=before.read_text().splitlines(True) if before.exists() else []
        chunks.extend(difflib.unified_diff(old,(ROOT/name).read_text().splitlines(True),fromfile='before/'+name,tofile='after/'+name))
    (review/'fp_graph_f1_only_changes.diff').write_text(''.join(chunks))
    result=subprocess.run(['git','diff','--check','--','CURRENT_PLAN.md'],capture_output=True,text=True)
    write(review/'diff_format_check.json',dict(exit_code=result.returncode,stdout=result.stdout,stderr=result.stderr))
    print('exit index and incremental diff delivered',resource,flush=True)


if __name__=='__main__':
    require_tmux();p=argparse.ArgumentParser();p.add_argument('--setup',action='store_true');p.add_argument('--controller',action='store_true');p.add_argument('--finish-delivery',action='store_true');p.add_argument('--start-finish',action='store_true');a=p.parse_args()
    if a.setup:setup()
    elif a.controller:campaign()
    elif a.start_finish:launch('finish_delivery',[PYTHON,'scripts/run_fp_graph_fusion.py','--finish-delivery'])
    elif a.finish_delivery:finish_delivery()
    else:p.error('explicit role required')
