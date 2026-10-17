"""r10: one global validation-selected weight; arrays only, no model replay."""
import argparse
import copy
import csv
import datetime
import difflib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import traceback
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from src.training.fce_kernel import (TASKS,POLICY,GLOBAL_BLEND_POLICY,PhaseGate,
    GlobalBlendGate,bind_branch_arrays,metrics,new_directory,choose_alpha,FEATURE_SPEC)
from src.training.fce_param import (select_global_finetune_weight,blend_finetune_arrays,
    GLOBAL_FINETUNE_BLEND_POLICY)
OUT=ROOT/'results/fce_tune_20261016/global_blend'
LOG=ROOT/'logs/fce_tune_20261016/global_blend'
R9=ROOT/'results/fce_tune_20261016/all_task_kernel'
EDITED=('CURRENT_PLAN.md','src/training/fce_kernel.py','src/training/fce_param.py',
    'scripts/run_fce_global_blend.py','tests/test_fce_global_blend.py')
ENV=dict(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
    NUMEXPR_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1',BLIS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1')
PYTHON='/opt/conda/bin/python'
LIMITS=dict(validation_units=35,validation_scores=175,formal_units=35,new_outputs=35,reused_outputs=35,
    final_cpu_starts=1,pt=0,ft=0,fit=0,features=0,model_inference=0,gpu=0,workers=0,refit=0)


def now():return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
def read(p):return json.loads(Path(p).read_text())
def write(p,value,replace=False):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists() and not replace:raise FileExistsError(p)
    p.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def check_budget(b):
    if set(b)!=set(LIMITS) or any(type(v) is not int or not 0<=v<=LIMITS[k] for k,v in b.items()) or b['new_outputs']+b['reused_outputs']>35:raise ValueError('r10 budget exceeded')
def require_tmux():
    if not os.environ.get('TMUX') or subprocess.check_output(['tmux','display-message','-p','#S'],text=True).strip()!='Uni-Poly':raise RuntimeError('requires Uni-Poly tmux')
def require_environment():
    if any(os.environ.get(k)!=v for k,v in ENV.items()):raise ValueError('CPU environment mismatch')
def load_npz(path,gate=None,test=False):
    if test:
        if gate is None:raise ValueError('TEST requires global freeze')
        gate.require_test()
    with np.load(path,allow_pickle=False) as v:return {k:v[k].copy() for k in v.files}


def run():
    require_tmux();require_environment();started=now();budget={k:0 for k in LIMITS};phase='validation'
    def ledger():check_budget(budget);write(OUT/'consumption.json',budget,True)
    try:
        ledger();old=read(R9/'selection_before_test.json');old_gate=PhaseGate();old_gate.freeze(old)
        if GLOBAL_FINETUNE_BLEND_POLICY!=GLOBAL_BLEND_POLICY:raise ValueError('public FT policy mismatch')
        validation=[];sources=[]
        for u in old['units']:
            folder=Path(u['model']).parent/'validation'/f"alpha_{u['alpha']:g}"
            values=load_npz(folder/'predictions.npz');trial=read(folder/'metrics.json')
            if trial!=u['chosen_validation'] or choose_alpha(u['validation_trials'])!=trial:raise ValueError('r9 alpha/validation source mismatch')
            aligned=bind_branch_arrays(values,u,'validation',old['identities'][u['task']],('y_g05','y_krr','y_fused'))
            if not np.array_equal(aligned['y_fused'],.5*aligned['y_g05']+.5*aligned['y_krr']):raise ValueError('r9 fixed .5 validation policy mismatch')
            for label,key in [('reference','y_g05'),('branch','y_krr'),('fused','y_fused')]:
                actual=metrics(aligned['y_true'],aligned[key])
                if any(not np.isclose(actual[m],trial[label][m],rtol=1e-12,atol=1e-12) for m in actual):raise ValueError('validation metric/schema mismatch')
            validation.append(dict(task=u['task'],fold=u['fold'],y_true=aligned['y_true'],g05=aligned['y_g05'],krr=aligned['y_krr']))
            sources.append(dict(task=u['task'],fold=u['fold'],path=str(folder/'predictions.npz'),alpha=u['alpha'],model=u['model']))
            budget['validation_units']+=1;ledger()
        selected=select_global_finetune_weight(validation);budget['validation_scores']=175;ledger()
        payload=dict(selected,frozen=True,frozen_at=now(),r9_freeze=old,validation_sources=sources,budget_at_freeze=copy.deepcopy(budget),reference_test_opened=False)
        gate=GlobalBlendGate();gate.freeze(payload);write(OUT/'selection_before_test.json',payload)
        print('global weight frozen',payload['weight'],flush=True)
        phase='formal';gate.require_test()
        summary=read(R9/'final_review/seven_task_results.json')
        if summary['policy']!=POLICY:raise ValueError('r9 formal policy mismatch')
        records={(r['task'],r['fold']):r for r in summary['folds']}
        if len(summary['folds'])!=35 or set(records)!={(t,f) for t in TASKS for f in range(5)}:raise ValueError('r9 formal coverage mismatch')
        rows=[];weight=payload['weight']
        for u in old['units']:
            budget['formal_units']+=1;ledger();gate.require_test()
            key=(u['task'],u['fold']);r=records[key]
            if r['alpha']!=u['alpha'] or r['validation']!=u['chosen_validation']:raise ValueError('formal frozen alpha/validation mismatch')
            paths=dict(g05=r['reference_path'],krr=r['paths']['branch'],r9=r['paths']['fused']);bound={}
            for label,path in paths.items():
                bound[label]=bind_branch_arrays(load_npz(path,gate,test=True),u,'test',old['identities'][u['task']],('y_pred',))
            truth=bound['g05']['y_true'];g05=bound['g05']['y_pred'];krr=bound['krr']['y_pred'];half=bound['r9']['y_pred']
            if any(not np.array_equal(truth,bound[k]['y_true']) for k in bound):raise ValueError('TEST branch label mismatch')
            if not np.array_equal(half,.5*g05+.5*krr):raise ValueError('r9 TEST fixed .5 mismatch')
            equivalent={0.:'krr',.5:'r9',1.:'g05'}.get(weight)
            folder=new_directory(OUT/'formal'/u['task']/f"fold{u['fold']}")
            if equivalent is not None:
                prediction=bound[equivalent]['y_pred'];output=paths[equivalent]
                write(folder/'reused_prediction.json',dict(weight=weight,path=output,policy=GLOBAL_BLEND_POLICY,source=equivalent));budget['reused_outputs']+=1
            else:
                prediction=blend_finetune_arrays(g05,krr,gate);output=str(folder/'global_blend_predictions.npz')
                np.savez(output,indices=np.asarray(u['test_indices']),sample_keys=np.asarray([old['identities'][u['task']][str(i)]['sample_key'] for i in u['test_indices']]),
                    y_true=truth,y_pred=prediction,weight=weight,alpha=u['alpha'],scaler_mean=u['scaler']['mean'],scaler_scale=u['scaler']['scale'])
                budget['new_outputs']+=1
            values={label:metrics(truth,p) for label,p in [('g05',g05),('r9',half),('global',prediction)]}
            for label,prior in [('g05','reference'),('r9','fused')]:
                if any(not np.isclose(values[label][m],r[prior][m],rtol=1e-12,atol=1e-12) for m in values[label]):raise ValueError('formal metrics disagree with existing r9 record')
            record=dict(task=u['task'],fold=u['fold'],alpha=u['alpha'],weight=weight,metrics=values,paths=paths,output=output,reused=equivalent is not None,
                global_minus_r9={m:values['global'][m]-values['r9'][m] for m in ('r2','rmse')},global_minus_g05={m:values['global'][m]-values['g05'][m] for m in ('r2','rmse')})
            write(folder/'metrics.json',record);rows.append(record);ledger()
        phase='report';report(rows,selected,summary['paper'],budget)
        if budget['formal_units']!=35 or budget['new_outputs']+budget['reused_outputs']!=35:raise ValueError('incomplete formal matrix')
        budget['final_cpu_starts']+=1;ledger()
        test=launch('final_cpu',['timeout','300s',PYTHON,'-m','pytest','-q','tests/test_fce_global_blend.py'])
        rc=wait_exit(test);status='待审查' if rc==0 else '需返修'
        write(OUT/'completion.json',dict(status=status,start=started,end=now(),test_exit=rc,weight=weight,consumption=budget,independent_review='NOT_PERFORMED'))
        update_plan(status,(OUT/'final_review/delivery.md').read_text()+f'\n唯一CPU检查真实exit {rc}。\n')
    except BaseException as error:
        ledger();write(OUT/'failure.json',dict(phase=phase,error=repr(error),traceback=traceback.format_exc(),consumption=budget,time=now()))
        update_plan('需返修','执行停止：'+repr(error)+'；现场failure.json，不自动重跑或扩大预算。')
        raise


def update_plan(status,body):
    p=ROOT/'CURRENT_PLAN.md';s=p.read_text().replace('状态：**执行中（r10）**。','状态：**'+status+'（r10）**。',1)
    s=s.replace('## 51. r10下一步：七任务统一的validation全局融合权重（执行中）','## 51. r10：七任务统一的validation全局融合权重（'+status+'）',1)
    p.write_text(s+'\n\n## 53. r10执行交接\n\n'+body+'\n')


def report(rows,selection,paper,budget):
    folder=new_directory(OUT/'final_review');tasks={}
    for t in TASKS:
        selected=[r for r in rows if r['task']==t];stats={label:{m:dict(mean=float(np.mean([r['metrics'][label][m] for r in selected])),std=float(np.std([r['metrics'][label][m] for r in selected],ddof=0))) for m in ('r2','rmse')} for label in ('g05','r9','global')}
        delta={ref:{m:stats['global'][m]['mean']-stats[ref][m]['mean'] for m in ('r2','rmse')} for ref in ('r9','g05')}
        tasks[t]=dict(stats=stats,deltas=delta,paper=paper[t],minus_paper={label:stats[label]['r2']['mean']-paper[t]['r2'] for label in stats},
            improved_folds_vs_r9=sum(r['global_minus_r9']['r2']>0 for r in selected),degraded_folds_vs_r9=sum(r['global_minus_r9']['r2']<0 for r in selected))
    macro={label:{m:float(np.mean([tasks[t]['stats'][label][m]['mean'] for t in TASKS])) for m in ('r2','rmse')} for label in ('g05','r9','global')}
    result=dict(weight=selection['weight'],selection=selection,tasks=tasks,macro=macro,folds=rows,consumption_at_science_end=budget,
        degraded_tasks_vs_r9=[t for t in TASKS if tasks[t]['deltas']['r9']['r2']<0],degraded_tasks_vs_g05=[t for t in TASKS if tasks[t]['deltas']['g05']['r2']<0],
        aggregation='5 folds mean/std ddof=0; 7 equal tasks; G05 seed42+fold',protocol='shared TEST development exposed; paper equivalence unverified',independent_review='NOT_PERFORMED')
    write(folder/'seven_task_results.json',result)
    with (folder/'fold_differences.csv').open('x',newline='') as stream:
        fields=['task','fold','alpha','weight','global_minus_r9_r2','global_minus_r9_rmse','global_minus_g05_r2','global_minus_g05_rmse'];writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for r in rows:writer.writerow(dict(task=r['task'],fold=r['fold'],alpha=r['alpha'],weight=r['weight'],**{ref+'_'+m:r[ref][m] for ref in ('global_minus_r9','global_minus_g05') for m in ('r2','rmse')}))
    lines=['状态：待审查；同一主体执行与自检，独立审查未进行。',f"唯一全局G05权重w={selection['weight']:g}；七任务五折全部相同，不按任务回退。r9模型、α、split、scaler固定。",'',
        '|w|35单元validation R²等权均值|','|---|---:|']
    lines += [f"|{s['weight']:g}|{s['mean_validation_r2']:.3f}|" for s in selection['scores']]
    lines+=['','上表仅用于validation选择；以下TEST只评价冻结后的一个新政策，并与已有G05、r9固定0.5政策对照，未在TEST评价其他网格权重。','',
        '|任务|G05 R²±std|r9 0.5 R²±std|全局w R²±std|G05 RMSE±std|r9 RMSE±std|全局RMSE±std|全局−r9 R²|全局−G05 R²|','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for t in TASKS:
        s=tasks[t]['stats'];fmt=lambda label,m:f"{s[label][m]['mean']:.3f}±{s[label][m]['std']:.3f}"
        lines.append('|'+t+'|'+ '|'.join(fmt(label,m) for m in ('r2','rmse') for label in ('g05','r9','global'))+f"|{tasks[t]['deltas']['r9']['r2']:+.3f}|{tasks[t]['deltas']['g05']['r2']:+.3f}|")
    lines+=['','|政策|Macro R²|七任务RMSE均值（不同物理单位，仅描述）|','|---|---:|---:|']
    lines += [f"|{label}|{macro[label]['r2']:.3f}|{macro[label]['rmse']:.3f}|" for label in ('g05','r9','global')]
    lines += ['',f"全局−r9 macro R² {macro['global']['r2']-macro['r9']['r2']:+.3f}；全局−G05 {macro['global']['r2']-macro['g05']['r2']:+.3f}。",
        '按R²任务均值，较r9退化：'+str(result['degraded_tasks_vs_r9'])+'；较G05退化：'+str(result['degraded_tasks_vs_g05'])+'。','',
        '|任务|论文最佳模型|论文R²|G05−论文|r9−论文|全局−论文|全局−r9 RMSE|全局−G05 RMSE|改善／退化折数vs r9|','|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for t in TASKS:
        r=tasks[t];lines.append(f"|{t}|{paper[t]['model']}|{paper[t]['r2']:.3f}|{r['minus_paper']['g05']:+.3f}|{r['minus_paper']['r9']:+.3f}|{r['minus_paper']['global']:+.3f}|{r['deltas']['r9']['rmse']:+.3f}|{r['deltas']['g05']['rmse']:+.3f}|{r['improved_folds_vs_r9']}／{r['degraded_folds_vs_r9']}|")
    lines+=['','完整逐折差见fold_differences.csv，完整原精度指标、统计及路径见seven_task_results.json。原精度计算后展示3位小数，std ddof=0。',
        '共享validation／TEST已开发暴露；与论文数据、split、处理及统计协议等价性未核实，仅探索性数值对照，不称独立盲测或SOTA。',
        'PT／FT／fit／特征生成／模型推理／GPU／worker／refit均0；等价政策复用已有结果，否则只保存35组所选权重算术融合。实际预算见consumption.json；冻结selection_before_test.json。']
    (folder/'delivery.md').write_text('\n'.join(lines)+'\n')


def launch(name,argv):
    require_tmux();folder=new_directory(LOG/name)
    command=dict(argv=argv,cwd=str(ROOT),environment=ENV,window='fce_global_blend_'+name);write(folder/'command.json',command)
    wrapper='''import datetime,json,os,subprocess,sys,traceback
from pathlib import Path
p=Path(sys.argv[1]);c=json.loads((p/'command.json').read_text())
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def save(v):(p/'execution.json').write_text(json.dumps(v,indent=2)+'\\n')
v=dict(command=c,start=now(),end=None,exit=None,status='RUNNING');save(v);rc=1
try:
 with (p/'stdout.log').open('x') as out,(p/'stderr.log').open('x') as err:
  child=subprocess.Popen(c['argv'],cwd=c['cwd'],env=dict(os.environ,**c['environment']),stdout=out,stderr=err)
  v['pid']=child.pid;save(v);rc=child.wait()
except BaseException:v['error']=traceback.format_exc()
finally:
 v.update(end=now(),exit=rc,status='EXITED');save(v);(p/'exit.txt').write_text(str(rc)+'\\n')
if p.name=='controller':subprocess.run([sys.executable,'scripts/run_fce_global_blend.py','--launch-finish'],cwd=c['cwd'],env=dict(os.environ,**c['environment']),check=False)
sys.exit(rc)
'''
    commandline=shlex.join([PYTHON,'-c',wrapper,str(folder)])
    (folder/'command.sh').write_text('#!/bin/bash\ncd '+shlex.quote(str(ROOT))+'\n'+commandline+'\n')
    subprocess.run(['tmux','new-window','-d','-t','Uni-Poly','-n',command['window'],
        'tmux set-option -w -t "$TMUX_PANE" remain-on-exit on\nexec bash '+shlex.quote(str(folder/'command.sh'))],check=True)
    return folder


def wait_exit(folder):
    import ctypes
    folder=Path(folder);libc=ctypes.CDLL(None,use_errno=True);fd=libc.inotify_init1(os.O_CLOEXEC)
    if fd<0:raise OSError(ctypes.get_errno(),'inotify_init1')
    try:
        if libc.inotify_add_watch(fd,os.fsencode(folder),0x8|0x80)<0:raise OSError(ctypes.get_errno(),'inotify_add_watch')
        while not (folder/'exit.txt').exists():os.read(fd,4096)
        return int((folder/'exit.txt').read_text())
    finally:os.close(fd)


def setup():
    require_tmux();require_environment()
    if OUT.exists() or LOG.exists():raise FileExistsError('r10 roots already exist; no replay')
    new_directory(OUT);new_directory(LOG);folder=new_directory(LOG/'setup');started=now();rc=1
    write(folder/'command.json',dict(argv=sys.argv,cwd=str(ROOT),environment=ENV,window='fce_global_blend_setup'))
    (folder/'stdout.log').touch(exist_ok=False);(folder/'stderr.log').touch(exist_ok=False)
    try:
        for name in ('CURRENT_PLAN.md','src/training/fce_param.py','src/training/fce_kernel.py'):
            target=OUT/'source_before'/name;target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(Path('/tmp/fce_global_blend_before_20261016')/name,target)
        for name in (*EDITED,'src/training/xc_kernel.py'):
            target=OUT/'source_used'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,target)
        launch('controller',[PYTHON,'scripts/run_fce_global_blend.py','--run']);rc=0
    except BaseException:
        with (folder/'stderr.log').open('a') as stream:traceback.print_exc(file=stream)
        raise
    finally:
        write(folder/'execution.json',dict(start=started,end=now(),exit=rc,status='EXITED'));(folder/'exit.txt').write_text(str(rc)+'\n')


def finish():
    require_tmux();require_environment();review=OUT/'final_review';review.mkdir(exist_ok=True);records=[]
    for name in ('setup','controller','final_cpu'):
        folder=LOG/name
        if (folder/'execution.json').exists():
            record=read(folder/'execution.json');record.update(log_dir=str(folder),exit_txt=int((folder/'exit.txt').read_text()) if (folder/'exit.txt').exists() else None);records.append(record)
    write(review/'exit_index.json',dict(records=records,own_finish_exit=str(LOG/'finish_delivery/exit.txt'),time=now()))
    p=ROOT/'CURRENT_PLAN.md';s=p.read_text()
    s+='\n\nr10实际预算：'+json.dumps(read(OUT/'consumption.json'),ensure_ascii=False)+'。真实exit：'+str([(Path(r['log_dir']).name,r['exit_txt']) for r in records])+'。收尾自身真实exit见logs/fce_tune_20261016/global_blend/finish_delivery/exit.txt；本索引写入时不伪造尚未退出的收尾状态。完整退出索引final_review/exit_index.json。\n'
    s+='结果results/fce_tune_20261016/global_blend/，日志logs/fce_tune_20261016/global_blend/；tmux Uni-Poly:fce_global_blend_setup/controller/final_cpu/finish_delivery；本轮增量final_review/global_blend_only_changes.diff。全部旧实验及产物保留，没有删除或新增哈希验证，无独立审查。\n'
    p.write_text(s);changes=[]
    for name in EDITED:
        before=OUT/'source_before'/name;previous=before.read_text().splitlines(True) if before.exists() else []
        changes.extend(difflib.unified_diff(previous,(ROOT/name).read_text().splitlines(True),fromfile='before/'+name,tofile='after/'+name))
    (review/'global_blend_only_changes.diff').write_text(''.join(changes))
    checked=subprocess.run(['git','diff','--check','--','CURRENT_PLAN.md','src/training/fce_param.py'],cwd=ROOT,capture_output=True,text=True)
    write(review/'diff_format_check.json',dict(exit=checked.returncode,stdout=checked.stdout,stderr=checked.stderr))
    print('delivery ready; science/test exits:',[(Path(r['log_dir']).name,r['exit_txt']) for r in records],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--setup',action='store_true');p.add_argument('--run',action='store_true');p.add_argument('--finish',action='store_true');p.add_argument('--launch-finish',action='store_true');a=p.parse_args()
    if a.setup:setup()
    elif a.run:run()
    elif a.launch_finish:launch('finish_delivery',[PYTHON,'scripts/run_fce_global_blend.py','--finish'])
    elif a.finish:finish()
    else:p.error('explicit role required')
