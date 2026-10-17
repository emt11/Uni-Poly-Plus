"""r14 bounded event-driven neural FT/formal orchestration; no retries."""
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
from scripts.run_fce_norm_unit import (ARM,TASKS,OUT,LOG,R10,PARENT,POLICY,read,write,
    config_for,epochs,source_binding,validate_checkpoint,validate_frozen,selected_history,validate_parameter_policy)
from scripts.run_fce_norm_unit import MSE_FT_LOSS_POLICY
from src.training.fce_param import NORM_ADAPTATION_POLICY
from src.training.fce_kernel import bind_branch_arrays,metrics,blend_arrays
from src.training.glt_dual_runtime import require_tmux
PYTHON='/opt/conda/bin/python';GPUS=(2,3)
EDITED=('CURRENT_PLAN.md','src/training/fce_param.py','scripts/run_fce_norm.py',
    'scripts/run_fce_norm_unit.py','tests/test_fce_norm.py')
LIMITS=dict(ft_starts=35,ft_completed=35,ft_epochs=2400,ft_attempted_updates=51450,
    ft_completed_updates=51450,validation_calls=2400,formal_starts=35,formal_completed=35,formal_outputs=70,
    cpu_test_starts=1,pt=0,krr_fit=0,krr_forward=0,morgan_generation=0,workers=0,extra_seeds=0,refit=0)


def now():return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
def check_budget(b):
    if set(b)!=set(LIMITS) or any(type(v) is not int or not 0<=v<=LIMITS[k] for k,v in b.items()):raise ValueError('LayerNorm hard budget mismatch')
def require_stage(stage,b):
    check_budget(b)
    if stage in ('freeze','formal','CPU') and any(b[k]!=v for k,v in dict(ft_completed=35,ft_epochs=2400,ft_attempted_updates=51450,ft_completed_updates=51450,validation_calls=2400).items()):raise ValueError('complete 35 FT matrix required')
    if stage=='CPU' and (b['formal_completed'],b['formal_outputs'])!=(35,70):raise ValueError('complete formal matrix required for CPU test')
def consumption():
    b={k:0 for k in LIMITS};b['ft_starts']=len(list(LOG.glob('ft_*/command.json')));b['formal_starts']=len(list(LOG.glob('test_*/command.json')));b['cpu_test_starts']=len(list(LOG.glob('final_cpu/command.json')))
    for f in (OUT/'finetune'/ARM).glob('*/fold*'):
        if (f/'updates.json').exists():
            u=read(f/'updates.json');b['ft_attempted_updates']+=u['attempted'];b['ft_completed_updates']+=u['completed']
        if (f/'progress.json').exists():b['ft_epochs']+=len(read(f/'progress.json')['history'])
        b['ft_completed']+=int((f/'metrics.json').exists())
    b['validation_calls']=b['ft_epochs'];b['formal_completed']=len(list((OUT/'formal'/ARM).glob('*/fold*/metrics.json')))
    b['formal_outputs']=len(list((OUT/'formal'/ARM).glob('*/fold*/*predictions.npz')))
    write(OUT/'consumption.json',b);check_budget(b);return b


def launch(name,argv,gpu=None):
    require_tmux()
    if gpu is not None and gpu not in GPUS:raise ValueError('GPU2/3 only')
    folder=LOG/name;folder.mkdir(parents=True,exist_ok=False)
    environment=dict(CUDA_VISIBLE_DEVICES='' if gpu is None else str(gpu),OMP_NUM_THREADS='1' if gpu is None else '4',
        OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1')
    command=dict(argv=argv,command=shlex.join(argv),cwd=str(ROOT),environment=environment,gpu=gpu,window='fce_norm_'+name)
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
if f.name=='controller':subprocess.run([sys.executable,'scripts/run_fce_norm.py','--start-finish'],cwd=j['cwd'],env=dict(os.environ,**j['environment']),check=False)
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
    pending=sum(v['formal'] or not (OUT/'finetune'/ARM/v['task']/f"fold{v['fold']}"/'progress.json').exists() for v in own)
    occupied=max(actual+pending,len(own));return max(0,4-occupied),dict(gpu=gpu,actual_contexts=actual,pending=pending,own=len(own),conservative_occupied=occupied)


def matrix(formal=False):
    # Largest work first reduces tail duration without changing any unit's seed.
    queue=sorted([(t,f) for t in TASKS for f in range(5)],key=lambda x:epochs(x[0])*math.ceil(len(source_binding(*x)['reference']['train_indices'])/24),reverse=True)
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
                        task,fold=queue.pop(0);name=('test_' if formal else 'ft_')+task+'_'+str(fold)
                        folder=OUT/('formal' if formal else 'finetune')/ARM/task/f'fold{fold}'
                        if folder.exists() or (LOG/name).exists():raise FileExistsError('unit writer/overwrite conflict')
                        b=consumption();key='formal_starts' if formal else 'ft_starts'
                        if b[key]>=35:raise ValueError('start budget exhausted')
                        argv=[PYTHON,'scripts/run_fce_norm_unit.py','--task',task,'--fold',str(fold)]+(['--formal'] if formal else [])
                        log=launch(name,argv,gpu);running[name]=dict(gpu=gpu,log=log,task=task,fold=fold,formal=formal);changed=True
                        if libc.inotify_add_watch(fd,os.fsencode(log),0x8|0x80)<0:raise OSError(ctypes.get_errno(),'inotify_add_watch')
                        with (LOG/'controller/scheduling.jsonl').open('a') as stream:stream.write(json.dumps(dict(time=now(),phase='formal' if formal else 'FT',launched=name,observed=observed,reserved_after=observed['conservative_occupied']+slot+1,own_running={str(g):sum(v['gpu']==g for v in running.values()) for g in GPUS}))+'\n')
            if changed:
                b=consumption();write(OUT/'status.json',dict(status='执行中',phase='formal' if formal else 'FT',consumption=b,running=[dict(task=v['task'],fold=v['fold'],gpu=v['gpu']) for v in running.values()],queued=len(queue)))
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
    for task in TASKS:
        for fold in range(5):
            folder=OUT/'finetune'/ARM/task/f'fold{fold}';meta=source_binding(task,fold);cfg=meta['config'];entry=meta['manifest']['folds'][fold]
            m=read(folder/'metrics.json');history=read(folder/'progress.json')['history'];s=read(folder/'scaler.json');count=epochs(task)*math.ceil(len(entry['train_indices'])/24)
            if m['config']!=cfg or m['loss_policy']!=MSE_FT_LOSS_POLICY or m['status']!='PASS' or m['outer_test']!='NOT_RUN' or m['epochs']!=epochs(task) or m['updates']!=dict(attempted=count,completed=count):raise ValueError('FT completion/config/loss mismatch')
            validate_parameter_policy(m['parameter_policy'])
            for row in history:
                if row['adaptation_policy']!=NORM_ADAPTATION_POLICY:raise ValueError('history adaptation mismatch')
            chosen=selected_history(history)
            if len(history)!=epochs(task) or any(r['recipe']!=cfg['recipe'] or r['loss_policy']!=MSE_FT_LOSS_POLICY for r in history) or chosen['epoch']!=m['best_epoch'] or chosen['validation_rmse']!=m['rmse'] or chosen['validation_r2']!=m['r2']:
                raise ValueError('raw neural RMSE best/history mismatch')
            if s!=meta['reference']['scaler']:raise ValueError('frozen train scaler mismatch')
            unit=dict(arm=ARM,task=task,fold=fold,seed=42+fold,config=cfg,loss_policy=copy.deepcopy(MSE_FT_LOSS_POLICY),adaptation_policy=copy.deepcopy(NORM_ADAPTATION_POLICY),parameter_policy=read(folder/'parameter_policy.json'),parent=str(PARENT),head_initial=meta['reference']['head_initial'],
                best=str(folder/'best.pt'),best_epoch=m['best_epoch'],best_stage=chosen['stage'],scaler=s,expected_updates=count,train_indices=entry['train_indices'],validation_indices=entry['validation_indices'],test_indices=sorted(entry['test_indices']),
                reference=meta['reference'],fixed_krr=meta['fixed_krr'])
            best=torch.load(folder/'best.pt',map_location='cpu',weights_only=False);validate_checkpoint(best,unit)
            if any(not torch.isfinite(v).all() for v in best['state_dict'].values()):raise FloatingPointError('nonfinite neural best')
            groups=best['optimizer_groups'];validate_parameter_policy(unit['parameter_policy'],groups)
            for stage in ('head','joint'):
                for group in groups[stage]:
                    expected_lr=cfg['recipe']['head_lr'] if stage=='head' else cfg['recipe']['joint_backbone_lr' if group['name']=='backbone' else 'joint_head_lr']
                    if group['lr']!=expected_lr or group['weight_decay']!=cfg['recipe']['weight_decay']:raise ValueError('original LR/WD mismatch')
            unit['optimizer_groups']=groups
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
    old_summary=read(R10/'final_review/seven_task_results.json');old_rows={(r['task'],r['fold']):r for r in old_summary['folds']}
    review=OUT/'final_review';review.mkdir(exist_ok=False);rows=[];paper={t:old_summary['tasks'][t]['paper'] for t in TASKS}
    for u in frozen['units']:
        task,fold=u['task'],u['fold'];meta=source_binding(task,fold);folder=OUT/'formal'/ARM/task/f'fold{fold}';m=read(folder/'metrics.json')
        if m['status']!='PASS' or m['config']!=u['config'] or m['loss_policy']!=MSE_FT_LOSS_POLICY or m['outer_test']!='RUN' or m['best_epoch']!=u['best_epoch'] or m['neural_test_forward_calls']!=1:raise ValueError('formal loss/best/forward binding mismatch')
        own=dict(u,alpha=u['fixed_krr']['alpha']);reference=old_rows[(task,fold)]
        bound={}
        for label,path in [('neural',folder/'test_predictions.npz'),('fused',folder/'fused_predictions.npz'),('g05',reference['paths']['g05']),('r10',reference['output'])]:
            p=load_predictions(path)
            # r10 alpha is metadata for the frozen, unchanged kernel branch.
            bound[label]=bind_branch_arrays(p,own,'test',meta['identities'],('y_pred',))
        truth=bound['neural']['y_true']
        if any(not np.array_equal(truth,bound[p]['y_true']) for p in bound):raise ValueError('paired formal truth mismatch')
        values={label:metrics(truth,p['y_pred']) for label,p in bound.items()}
        if values['neural']!=dict(r2=m['r2'],rmse=m['rmse']) or values['fused']!=m['fused'] or values['g05']!=reference['metrics']['g05'] or values['r10']!=reference['metrics']['global']:raise ValueError('formal metric/schema mismatch')
        record=dict(task=task,fold=fold,best_epoch=u['best_epoch'],best_stage=u['best_stage'],metrics=values,validation=u['validation_diagnostic'],
            fused_minus_r10={k:values['fused'][k]-values['r10'][k] for k in ('r2','rmse')},neural_minus_g05={k:values['neural'][k]-values['g05'][k] for k in ('r2','rmse')},
            new_fused_minus_neural={k:values['fused'][k]-values['neural'][k] for k in ('r2','rmse')},old_fused_minus_neural={k:values['r10'][k]-values['g05'][k] for k in ('r2','rmse')},
            test_minus_validation={label:{k:values[label][k]-u['validation_diagnostic'][label][k] for k in ('r2','rmse')} for label in ('neural','fused')},
            neural_path=str(folder/'test_predictions.npz'),fused_path=str(folder/'fused_predictions.npz'),krr_path=m['krr_path'])
        rows.append(record)
    policies=('fused','neural','r10','g05');tasks={}
    for task in TASKS:
        selected=[r for r in rows if r['task']==task];stats={p:{k:dict(mean=float(np.mean([r['metrics'][p][k] for r in selected])),std=float(np.std([r['metrics'][p][k] for r in selected],ddof=0))) for k in ('r2','rmse')} for p in policies}
        gains=[r['fused_minus_r10']['r2'] for r in selected]
        tasks[task]=dict(stats=stats,paper=paper[task],minus_paper={p:stats[p]['r2']['mean']-paper[task]['r2'] for p in policies},
            fused_minus_r10={k:stats['fused'][k]['mean']-stats['r10'][k]['mean'] for k in ('r2','rmse')},neural_minus_g05={k:stats['neural'][k]['mean']-stats['g05'][k]['mean'] for k in ('r2','rmse')},
            improved_folds=sum(x>0 for x in gains),degraded_folds=sum(x<0 for x in gains),mean_delta_excluding_best_fold=float((sum(gains)-max(gains))/4))
    macro={p:float(np.mean([tasks[t]['stats'][p]['r2']['mean'] for t in TASKS])) for p in policies};gain=macro['fused']-macro['r10']
    best_task=max(TASKS,key=lambda t:tasks[t]['fused_minus_r10']['r2']);best_unit=max(rows,key=lambda r:r['fused_minus_r10']['r2'])
    result=dict(policy=POLICY,tasks=tasks,macro_r2=macro,delta_macro_r2=gain,delta_neural_macro_r2=macro['neural']-macro['g05'],milestone=gain>=.010,
        reached_paper={p:sum(tasks[t]['stats'][p]['r2']['mean']>=paper[t]['r2'] for t in TASKS) for p in policies},
        degraded_fused_tasks=[t for t in TASKS if tasks[t]['fused_minus_r10']['r2']<0],degraded_neural_tasks=[t for t in TASKS if tasks[t]['neural_minus_g05']['r2']<0],
        concentration=dict(best_task=best_task,best_unit=[best_unit['task'],best_unit['fold']],macro_delta_excluding_best_task=float(np.mean([tasks[t]['fused_minus_r10']['r2'] for t in TASKS if t!=best_task])),mean_delta_excluding_best_unit=float((sum(r['fused_minus_r10']['r2'] for r in rows)-best_unit['fused_minus_r10']['r2'])/34)),
        best_epoch_distribution={stage:sum(u['best_stage']==stage for u in frozen['units']) for stage in ('head','joint')},folds=rows,consumption_at_science_end=consumption(),
        aggregation='five-fold arithmetic mean/std ddof=0; G05 seed42+fold; equal seven-task R2 macro',protocol='shared validation/TEST development exposed; paper equivalence unverified',independent_review='NOT_PERFORMED')
    write(review/'seven_task_results.json',result)
    with (review/'fold_differences.csv').open('x',newline='') as stream:
        fields=['task','fold','best_epoch','best_stage','fused_minus_r10_r2','fused_minus_r10_rmse','neural_minus_g05_r2','neural_minus_g05_rmse','neural_test_minus_validation_r2','neural_test_minus_validation_rmse','fused_test_minus_validation_r2','fused_test_minus_validation_rmse'];writer=__import__('csv').DictWriter(stream,fieldnames=fields);writer.writeheader()
        for r in rows:writer.writerow(dict(task=r['task'],fold=r['fold'],best_epoch=r['best_epoch'],best_stage=r['best_stage'],**{p+'_'+m:r[p][m] for p in ('fused_minus_r10','neural_minus_g05') for m in ('r2','rmse')},**{label+'_test_minus_validation_'+m:r['test_minus_validation'][label][m] for label in ('neural','fused') for m in ('r2','rmse')}))
    write(review/'model_prediction_index.json',[dict(task=u['task'],fold=u['fold'],best=u['best'],best_epoch=u['best_epoch'],best_stage=u['best_stage'],head_initial=u['head_initial'],parent=u['parent'],fixed_krr=u['fixed_krr'],neural_prediction=r['neural_path'],fused_prediction=r['fused_path']) for u,r in zip(frozen['units'],rows)])
    write(review/'parameter_groups.json',[dict(task=u['task'],fold=u['fold'],parameter_policy=u['parameter_policy'],optimizer_groups=u['optimizer_groups']) for u in frozen['units']])
    lines=['总参数39,876,344；head仅4张量／525,313；joint为52个encoder LN affine张量／26,624＋原head，共56张量／551,937；冻结encoder参数39,324,407。实际名与组见parameter_groups.json。冻结张量逐元素比较在最终唯一CPU检查内执行，结果见frozen_encoder_comparison.json。', '状态：待审查；同一主体执行与自检，独立审查未进行。',
        '唯一科学变化：全部七任务使用原MSE；joint仅更新实际nn.LayerNorm affine＋原head，head阶段不变。R0 recipe、G05 parent、原head_initial、seed42+fold、split和缓存语义不变；仍按原单位neural validation RMSE严格改善选best。',
        '固定复用r10 radius2 KRR／α及已有预测，0.75新神经＋0.25旧KRR；无新KRR拟合、KRR模型forward、Morgan生成或参数搜索。', '',
        '|任务|新融合R²±std|新神经R²±std|r10 R²±std|G05 R²±std|融合−r10|神经−G05|改善／退化fold（融合）|', '|---|---:|---:|---:|---:|---:|---:|---:|']
    for t in TASKS:
        r=tasks[t];s=r['stats'];lines.append('|'+t+'|'+'|'.join(f"{s[p]['r2']['mean']:.3f}±{s[p]['r2']['std']:.3f}" for p in policies)+f"|{r['fused_minus_r10']['r2']:+.3f}|{r['neural_minus_g05']['r2']:+.3f}|{r['improved_folds']}／{r['degraded_folds']}|")
    lines+=['','|任务|新融合RMSE±std|新神经RMSE±std|r10 RMSE±std|G05 RMSE±std|融合−r10 RMSE|神经−G05 RMSE|','|---|---:|---:|---:|---:|---:|---:|']
    for t in TASKS:
        r=tasks[t];s=r['stats'];lines.append('|'+t+'|'+'|'.join(f"{s[p]['rmse']['mean']:.3f}±{s[p]['rmse']['std']:.3f}" for p in policies)+f"|{r['fused_minus_r10']['rmse']:+.3f}|{r['neural_minus_g05']['rmse']:+.3f}|")
    lines+=['','|政策|Macro R²|论文数值达标数|','|---|---:|---:|']
    lines+=[f"|{p}|{macro[p]:.3f}|{result['reached_paper'][p]}/7|" for p in policies]
    lines+=['',f"主比较融合−r10 macro R² {gain:+.3f}；新神经−G05 {result['delta_neural_macro_r2']:+.3f}；+0.010进展里程碑：{'达到' if result['milestone'] else '未达到'}。",
        '退化任务（原精度R²）：融合−r10 '+str(result['degraded_fused_tasks'])+'；神经−G05 '+str(result['degraded_neural_tasks'])+'。',
        'Best阶段分布：'+str(result['best_epoch_distribution'])+'；各fold best epoch与阶段见fold_differences.csv。',
        f"最大增益任务{best_task}，去掉后macro差{result['concentration']['macro_delta_excluding_best_task']:+.3f}；最大增益单元{result['concentration']['best_unit']}，去掉后34单元平均差{result['concentration']['mean_delta_excluding_best_unit']:+.3f}。",'',
        '|任务|论文最佳模型|论文R²|新融合−论文|新神经−论文|r10−论文|G05−论文|','|---|---|---:|---:|---:|---:|---:|']
    for t in TASKS:
        r=tasks[t];lines.append(f"|{t}|{paper[t]['model']}|{paper[t]['r2']:.3f}|"+'|'.join(f"{r['minus_paper'][p]:+.3f}" for p in policies)+'|')
    lines+=['','每任务原单位RMSE单独报告，负差表示改善，不平均不同物理单位RMSE。五折mean/std ddof=0，原精度计算与门槛判定后展示3位小数。',
        '完整逐折两类主差、validation→TEST差、best epoch见fold_differences.csv；各自融合相对自身神经的差和全部原精度指标见seven_task_results.json，不称interaction。',
        '35项best／政策／recipe／parent／head／scaler／split／核绑定全部冻结后才读取TEST。共享validation／TEST开发暴露，论文协议等价性未核实，仅探索性对照，不称独立盲测、稳定优势或公平超过论文。']
    (review/'delivery.md').write_text('\n'.join(lines)+'\n');return '\n'.join(lines)+'\n'


def update_plan(status,body):
    p=ROOT/'CURRENT_PLAN.md';s=p.read_text().replace('状态：**执行中（r14）**。','状态：**'+status+'（r14）**。',1)
    s=s.replace('## 67. r14：七任务统一LayerNorm＋head微调（执行中）','## 67. r14：七任务统一LayerNorm＋head微调（'+status+'）',1)
    p.write_text(s+'\n\n## 69. r14执行交接\n\n'+body+'\n')


def campaign():
    require_tmux();started=now()
    try:
        source=read(R10/'selection_before_test.json');units=source['r9_freeze']['units']
        planned_epochs=sum(epochs(u['task']) for u in units);planned_updates=sum(epochs(u['task'])*math.ceil(len(u['train_indices'])/24) for u in units)
        if len(units)!=35 or (planned_epochs,planned_updates)!=(2400,51450):raise ValueError('source FT budget differs from authorization')
        write(OUT/'planned_budget.json',dict(ft=35,epochs=planned_epochs,updates=planned_updates,formal=35,outputs=70,new_pt=0,new_krr_fit=0,new_krr_forward=0,new_morgan=0))
        initialize=launch('initialize',[PYTHON,'scripts/run_fce_norm_unit.py','--initialize'])
        if wait_original(initialize):raise RuntimeError('parent/reference initialization failed; no FT dispatch')
        matrix();require_stage('freeze',consumption());freeze();require_stage('formal',consumption());matrix(True)
        require_stage('CPU',consumption());delivery=report()
        test=launch('final_cpu',['timeout','300s',PYTHON,'-m','pytest','-q','tests/test_fce_norm.py']);rc=wait_original(test)
        status='待审查' if rc==0 else '需返修';b=consumption()
        write(OUT/'completion.json',dict(status=status,start=started,end=now(),test_exit=rc,consumption=b,independent_review='NOT_PERFORMED'))
        update_plan(status,delivery+f'\n唯一CPU检查真实exit {rc}；预算'+json.dumps(b,ensure_ascii=False)+'。\n')
        write(OUT/'status.json',dict(status=status,phase='finished',consumption=b))
    except BaseException as error:
        try:b=consumption()
        except BaseException:b=read(OUT/'consumption.json') if (OUT/'consumption.json').exists() else {}
        write(OUT/'campaign_failure.json',dict(status='需返修',error=repr(error),traceback=traceback.format_exc(),time=now(),consumption=b))
        update_plan('需返修','异常停止：'+repr(error)+'；现场campaign_failure.json，不自动重跑或扩预算；不完整矩阵禁止正式评价。');raise


def setup():
    require_tmux()
    if OUT.exists() or LOG.exists():raise FileExistsError('new r14 roots must not exist; no replay')
    OUT.mkdir(parents=True);LOG.mkdir(parents=True);folder=LOG/'setup';folder.mkdir();started=now();rc=1
    write(folder/'command.json',dict(argv=sys.argv,cwd=str(ROOT),environment={k:os.environ.get(k) for k in ('CUDA_VISIBLE_DEVICES','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')},window='fce_norm_setup'))
    (folder/'stdout.log').touch();(folder/'stderr.log').touch()
    try:
        for name in ('CURRENT_PLAN.md','src/training/fce_param.py'):
            target=OUT/'source_before'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(Path('/tmp/fce_norm_before_20261016')/name,target)
        for name in (*EDITED,'src/training/fce_kernel.py','scripts/run_fce_tune_t3b_unit.py','scripts/finetune_mcl_ph.py','src/utils.py','src/modules/glt_adaptation.py','src/training/glt_dual_runtime.py'):
            target=OUT/'source_used'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,target)
        launch('controller',[PYTHON,'scripts/run_fce_norm.py','--controller']);rc=0
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
    comparison=read(review/'frozen_encoder_comparison.json') if (review/'frozen_encoder_comparison.json').exists() else {'status':'未完成'}
    with (review/'delivery.md').open('a') as stream:stream.write('\n最终唯一CPU检查冻结张量直接比较：'+json.dumps(comparison,ensure_ascii=False)+'\n')
    p=ROOT/'CURRENT_PLAN.md';s=p.read_text();s+='\n冻结encoder逐元素比较：'+json.dumps(comparison,ensure_ascii=False)+'\n';s+='\n\nr14真实退出索引final_review/execution_exits.json；收尾自身真实exit见logs/fce_tune_20261016/norm_ft/finish_delivery/exit.txt，写本索引时不伪造尚未退出状态。实际并发记录：'+json.dumps(resource,ensure_ascii=False)+'。\n'
    s+='结果results/fce_tune_20261016/norm_ft/，日志logs/fce_tune_20261016/norm_ft/；tmux Uni-Poly:fce_norm_*，增量final_review/norm_only_changes.diff。无删除，无新增哈希验证，无模型历史／SOTA表改动，独立审查未进行。\n';p.write_text(s)
    chunks=[]
    for name in EDITED:
        before=OUT/'source_before'/name;old=before.read_text().splitlines(True) if before.exists() else []
        chunks.extend(difflib.unified_diff(old,(ROOT/name).read_text().splitlines(True),fromfile='before/'+name,tofile='after/'+name))
    (review/'norm_only_changes.diff').write_text(''.join(chunks))
    result=subprocess.run(['git','diff','--check','--','CURRENT_PLAN.md','src/training/fce_param.py'],capture_output=True,text=True)
    write(review/'diff_format_check.json',dict(exit_code=result.returncode,stdout=result.stdout,stderr=result.stderr))
    print('exit index and incremental diff delivered',resource,flush=True)


if __name__=='__main__':
    require_tmux();p=argparse.ArgumentParser();p.add_argument('--setup',action='store_true');p.add_argument('--controller',action='store_true');p.add_argument('--finish-delivery',action='store_true');p.add_argument('--start-finish',action='store_true');a=p.parse_args()
    if a.setup:setup()
    elif a.controller:campaign()
    elif a.start_finish:launch('finish_delivery',[PYTHON,'scripts/run_fce_norm.py','--finish-delivery'])
    elif a.finish_delivery:finish_delivery()
    else:p.error('explicit role required')
