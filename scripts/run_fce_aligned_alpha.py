"""r11: cached validation choices, necessary selected-alpha fits, then TEST."""
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
import rdkit,scipy,sklearn
from threadpoolctl import threadpool_info
from src.training.fce_kernel import (TASKS,ALPHAS,FEATURE_SPEC,ALIGNED_ALPHA_POLICY,
    GlobalBlendGate,AlignedAlphaGate,bind_branch_arrays,validate_model,metrics,new_directory)
from src.training.fce_param import (ALIGNED_FINETUNE_POLICY,select_aligned_finetune_alpha,
    restore_aligned_finetune_model,predict_aligned_finetune_predictions)
OUT=ROOT/'results/fce_tune_20261016/aligned_alpha'
LOG=ROOT/'logs/fce_tune_20261016/aligned_alpha'
R10=ROOT/'results/fce_tune_20261016/global_blend'
R9=ROOT/'results/fce_tune_20261016/all_task_kernel'
EDITED=('CURRENT_PLAN.md','src/training/fce_kernel.py','src/training/fce_param.py',
    'scripts/run_fce_aligned_alpha.py','tests/test_fce_aligned_alpha.py')
ENV=dict(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
    NUMEXPR_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1',BLIS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1')
PYTHON='/opt/conda/bin/python'
LIMITS=dict(validation_scores=105,selected_units=35,K=35,fit_attempts=35,fit_completed=35,
    reused_models=35,reused_coefficients=35,formal_starts=35,formal_completed=35,new_outputs=70,
    reused_formal=35,test_queries=5875,final_cpu_starts=1,pt=0,neural_ft=0,epochs=0,
    optimizer_updates=0,gpu=0,workers=0,refit=0,train_features=0,validation_forward=0)


def now():return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
def read(p):return json.loads(Path(p).read_text())
def write(p,value,replace=False):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists() and not replace:raise FileExistsError(p)
    p.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def check_budget(b):
    if set(b)!=set(LIMITS) or any(type(v) is not int or not 0<=v<=LIMITS[k] for k,v in b.items()):raise ValueError('r11 budget mismatch')
    if b['fit_attempts']>b['K'] or b['formal_starts']>b['K'] or b['new_outputs']>2*b['K'] or b['fit_completed']>b['fit_attempts'] or b['formal_completed']>b['formal_starts']:raise ValueError('r11 changed-only budget exceeded')
def require_tmux():
    if not os.environ.get('TMUX') or subprocess.check_output(['tmux','display-message','-p','#S'],text=True).strip()!='Uni-Poly':raise RuntimeError('requires Uni-Poly tmux')
def require_environment():
    if any(os.environ.get(k)!=v for k,v in ENV.items()):raise ValueError('CPU environment mismatch')
def load_npz(path,gate=None,test=False):
    if test:
        if gate is None:raise ValueError('TEST requires aligned freeze')
        gate.require_test()
    with np.load(path,allow_pickle=False) as v:return {k:v[k].copy() for k in v.files}


def run():
    require_tmux();require_environment();started=now();b={k:0 for k in LIMITS};phase='validation'
    def ledger():check_budget(b);write(OUT/'consumption.json',b,True)
    try:
        ledger();r10=read(R10/'selection_before_test.json');prior_gate=GlobalBlendGate();prior_gate.freeze(r10)
        if r10['weight']!=.75 or ALIGNED_FINETUNE_POLICY!=ALIGNED_ALPHA_POLICY:raise ValueError('fixed .75 public policy mismatch')
        versions=dict(rdkit=rdkit.__version__,sklearn=sklearn.__version__,numpy=np.__version__,scipy=scipy.__version__)
        old_versions=read(R9/'policy.json')['versions']
        if any(versions[k]!=old_versions[k] for k in versions):raise ValueError('numerical dependency versions changed')
        if any(p['num_threads']!=1 for p in threadpool_info()):raise ValueError('CPU thread mismatch')
        write(OUT/'policy.json',dict(policy=ALIGNED_ALPHA_POLICY,historical_model_feature_spec=FEATURE_SPEC,versions=versions,threadpools=threadpool_info()))
        units=[];old=r10['r9_freeze'];identities=old['identities']
        for prior in old['units']:
            caches={};sources=[]
            for alpha in ALPHAS:
                folder=Path(prior['model']).parent/'validation'/f'alpha_{alpha:g}'
                values=load_npz(folder/'predictions.npz');record=read(folder/'metrics.json')
                expected=next(t for t in prior['validation_trials'] if t['alpha']==alpha)
                if record!=expected:raise ValueError('cached validation trial metadata mismatch')
                own=dict(prior,alpha=alpha);aligned=bind_branch_arrays(values,own,'validation',identities[prior['task']],('y_g05','y_krr','y_fused'))
                if not np.array_equal(aligned['y_fused'],.5*aligned['y_g05']+.5*aligned['y_krr']):raise ValueError('historical .5 cache mismatch')
                for label,key in [('reference','y_g05'),('branch','y_krr'),('fused','y_fused')]:
                    actual=metrics(aligned['y_true'],aligned[key])
                    if any(not np.isclose(actual[m],record[label][m],rtol=1e-12,atol=1e-12) for m in actual):raise ValueError('cached score/source mismatch')
                caches[alpha]=values;sources.append(dict(alpha=alpha,path=str(folder/'predictions.npz')))
            choice=select_aligned_finetune_alpha(prior,caches,identities[prior['task']]);unit=copy.deepcopy(prior)
            unit.update(choice,source_model=prior['model'],validation_sources=sources,model_ready=False)
            units.append(unit);b['validation_scores']+=3;b['selected_units']+=1;ledger()
        b['K']=sum(u['changed'] for u in units);ledger()
        write(OUT/'alpha_choices_before_models.json',dict(K=b['K'],units=units,policy=ALIGNED_ALPHA_POLICY,time=now()))
        print('validation selected all 35; changed K=',b['K'],flush=True);phase='models'
        for u in units:
            folder=new_directory(OUT/'models'/u['task']/f"fold{u['fold']}")
            if not u['changed']:
                if not Path(u['model']).is_file():raise FileNotFoundError(u['model'])
                u.update(model_ready=True,model_origin='unchanged');b['reused_models']+=1
            else:
                prior=next(x for x in old['units'] if (x['task'],x['fold'])==(u['task'],u['fold']))
                saved=load_npz(prior['model']);cached=[];paths=[]
                # Only explicit model artifacts from the same saved training problem;
                # predictions.npz is never a coefficient source.
                for name in ('model.npz','selected_model.npz'):
                    path=Path(prior['model']).parent/'validation'/f"alpha_{u['alpha']:g}"/name
                    if path.is_file():cached.append(load_npz(path));paths.append(str(path))
                def on_fit(event):
                    b['fit_attempts' if event=='attempted' else 'fit_completed']+=1;ledger()
                model,origin=restore_aligned_finetune_model(saved,prior,u['alpha'],on_fit=on_fit,cached_models=cached)
                if origin=='reused_coefficients':u['model']=paths[0];b['reused_coefficients']+=1
                else:
                    u['model']=str(folder/'selected_model.npz');np.savez(u['model'],**model)
                u.update(model_ready=True,model_origin=origin,coefficient_cache_candidates=paths)
            write(folder/'binding.json',u);ledger()
        payload=dict(policy=ALIGNED_ALPHA_POLICY,K=b['K'],units=units,r10_freeze=r10,frozen=True,frozen_at=now(),
            budget_at_freeze=copy.deepcopy(b),identities=identities,versions=versions,reference_test_opened=False)
        gate=AlignedAlphaGate();gate.freeze(payload);write(OUT/'selection_before_test.json',payload);phase='formal'
        gate.require_test();prior_summary=read(R10/'final_review/seven_task_results.json')
        records={(r['task'],r['fold']):r for r in prior_summary['folds']}
        if set(records)!={(t,f) for t in TASKS for f in range(5)}:raise ValueError('r10 formal coverage mismatch')
        rows=[]
        for u in units:
            gate.require_test();r=records[(u['task'],u['fold'])]
            if r['alpha']!=u['old_alpha'] or r['weight']!=.75:raise ValueError('r10 cached alpha/weight mismatch')
            cached_output=load_npz(r['output'],gate,True)
            if 'weight' in cached_output and float(cached_output['weight'])!=.75:raise ValueError('r10 cached prediction weight mismatch')
            aligned=bind_branch_arrays(cached_output,dict(u,alpha=u['old_alpha']),'test',identities[u['task']],('y_pred',))
            truth=aligned['y_true'];reference=load_npz(r['paths']['g05'],gate,True)
            ref=bind_branch_arrays(reference,dict(u,alpha=u['old_alpha']),'test',identities[u['task']],('y_pred',))
            if not np.array_equal(truth,ref['y_true']):raise ValueError('r10/G05 truth mismatch')
            g05=ref['y_pred'];folder=new_directory(OUT/'formal'/u['task']/f"fold{u['fold']}")
            if u['changed']:
                b['formal_starts']+=1;ledger();model=load_npz(u['model'])
                def on_query():b['test_queries']+=1;ledger()
                yt,ref_pred,branch,prediction=predict_aligned_finetune_predictions(model,u,reference,identities[u['task']],gate,on_query=on_query)
                if not np.array_equal(yt,truth) or not np.array_equal(ref_pred,g05):raise ValueError('new prediction reference mismatch')
                paths={}
                for label,value in [('branch',branch),('r11',prediction)]:
                    path=folder/(label+'_predictions.npz');np.savez(path,indices=u['test_indices'],sample_keys=[identities[u['task']][str(i)]['sample_key'] for i in u['test_indices']],
                        y_true=truth,y_pred=value,y_g05=g05,alpha=u['alpha'],prediction_weight=.75,alpha_selection_weight=.75,scaler_mean=u['scaler']['mean'],scaler_scale=u['scaler']['scale'])
                    paths[label]=str(path);b['new_outputs']+=1;ledger()
                b['formal_completed']+=1
            else:
                prediction=aligned['y_pred'];paths=dict(r11=r['output'],branch='unchanged; no branch model forward')
                write(folder/'reused_prediction.json',dict(path=r['output'],alpha=u['alpha'],prediction_weight=.75));b['reused_formal']+=1
            values=copy.deepcopy(r['metrics']);values['r10']=values.pop('global');values['r11']=metrics(truth,prediction)
            if metrics(truth,aligned['y_pred'])!=values['r10'] or metrics(truth,g05)!=values['g05']:raise ValueError('r10/G05 cached metric mismatch')
            result=dict(task=u['task'],fold=u['fold'],old_alpha=u['old_alpha'],alpha=u['alpha'],changed=u['changed'],metrics=values,paths=paths,
                validation=u['chosen_validation'],delta={m:values['r11'][m]-values['r10'][m] for m in ('r2','rmse')},
                test_minus_validation={m:values['r11'][m]-u['chosen_validation']['fused'][m] for m in ('r2','rmse')})
            write(folder/'metrics.json',result);rows.append(result);ledger()
        if b['formal_completed']!=b['K'] or b['reused_formal']!=35-b['K'] or b['new_outputs']!=2*b['K']:raise ValueError('incomplete changed-only matrix')
        phase='report';report(rows,units,prior_summary,b)
        b['final_cpu_starts']+=1;ledger();test=launch('final_cpu',['timeout','300s',PYTHON,'-m','pytest','-q','tests/test_fce_aligned_alpha.py'])
        rc=wait_exit(test);status='待审查' if rc==0 else '需返修'
        write(OUT/'completion.json',dict(status=status,start=started,end=now(),test_exit=rc,consumption=b,independent_review='NOT_PERFORMED'))
        update_plan(status,(OUT/'final_review/delivery.md').read_text()+f'\n唯一CPU检查真实exit {rc}。\n')
    except BaseException as error:
        write(OUT/'consumption.json',b,True);write(OUT/'failure.json',dict(phase=phase,error=repr(error),traceback=traceback.format_exc(),consumption=b,time=now()))
        update_plan('需返修','执行停止：'+repr(error)+'；failure.json保留现场，不自动重跑或扩大预算。');raise


def update_plan(status,body):
    p=ROOT/'CURRENT_PLAN.md';s=p.read_text().replace('状态：**执行中（r11）**。','状态：**'+status+'（r11）**。',1)
    s=s.replace('## 55. r11：七任务统一对齐α选参目标（执行中）','## 55. r11：七任务统一对齐α选参目标（'+status+'）',1)
    p.write_text(s+'\n\n## 57. r11执行交接\n\n'+body+'\n')


def report(rows,units,prior,budget):
    folder=new_directory(OUT/'final_review');paper={t:prior['tasks'][t]['paper'] for t in TASKS};tasks={};policies=('r11','r10','r9','g05')
    for t in TASKS:
        selected=[r for r in rows if r['task']==t]
        stats={label:{m:dict(mean=float(np.mean([r['metrics'][label][m] for r in selected])),std=float(np.std([r['metrics'][label][m] for r in selected],ddof=0))) for m in ('r2','rmse')} for label in policies}
        gains=[r['delta']['r2'] for r in selected];best=max(gains)
        tasks[t]=dict(stats=stats,paper=paper[t],minus_paper={p:stats[p]['r2']['mean']-paper[t]['r2'] for p in policies},
            differences={p:{m:stats['r11'][m]['mean']-stats[p][m]['mean'] for m in ('r2','rmse')} for p in policies[1:]},
            changed_units=sum(u['task']==t and u['changed'] for u in units),improved_folds=sum(x>0 for x in gains),degraded_folds=sum(x<0 for x in gains),
            best_fold=max(selected,key=lambda r:r['delta']['r2'])['fold'],mean_delta_excluding_best_fold=float((sum(gains)-best)/4))
    macro={p:float(np.mean([tasks[t]['stats'][p]['r2']['mean'] for t in TASKS])) for p in policies};gain=macro['r11']-macro['r10']
    gains=[r['delta']['r2'] for r in rows];best_row=max(rows,key=lambda r:r['delta']['r2'])
    result=dict(policy=ALIGNED_ALPHA_POLICY,K=budget['K'],tasks=tasks,macro_r2=macro,delta_macro_r2=gain,milestone=gain>=.010,
        reached_paper={p:sum(tasks[t]['stats'][p]['r2']['mean']>=paper[t]['r2'] for t in TASKS) for p in policies},
        degraded_tasks={p:[t for t in TASKS if tasks[t]['differences'][p]['r2']<0] for p in policies[1:]},folds=rows,
        concentration=dict(best_task=best_row['task'],best_fold=best_row['fold'],mean_delta_excluding_best_unit=float((sum(gains)-best_row['delta']['r2'])/34)),
        consumption_at_science_end=copy.deepcopy(budget),aggregation='five-fold arithmetic mean/std ddof=0; seven equally weighted R2 task means; G05 seed42+fold',
        protocol='shared validation/TEST development exposed; paper equivalence unverified',independent_review='NOT_PERFORMED')
    write(folder/'seven_task_results.json',result)
    with (folder/'alpha_changes.csv').open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['task','fold','old_alpha','alpha','changed','model_origin','model']);writer.writeheader()
        for u in units:writer.writerow({k:u[k] for k in writer.fieldnames})
    with (folder/'fold_differences.csv').open('x',newline='') as f:
        fields=['task','fold','old_alpha','alpha','delta_r2','delta_rmse','validation_r2','validation_rmse','test_minus_validation_r2','test_minus_validation_rmse'];writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for r in rows:writer.writerow(dict(task=r['task'],fold=r['fold'],old_alpha=r['old_alpha'],alpha=r['alpha'],delta_r2=r['delta']['r2'],delta_rmse=r['delta']['rmse'],validation_r2=r['validation']['fused']['r2'],validation_rmse=r['validation']['fused']['rmse'],test_minus_validation_r2=r['test_minus_validation']['r2'],test_minus_validation_rmse=r['test_minus_validation']['rmse']))
    lines=['状态：待审查；同一主体执行与自检，独立审查未进行。',
        '七任务35单元统一固定w=0.75；唯一变化是α按0.75融合validation原单位RMSE选择，同值选大α，无task开关或TEST回退。',
        f"K={budget['K']}；新增fit attempted/completed={budget['fit_attempts']}/{budget['fit_completed']}，复用未变模型{budget['reused_models']}，复用变化α系数{budget['reused_coefficients']}；新正式预测{budget['formal_completed']}，复用{budget['reused_formal']}，新数组{budget['new_outputs']}组，TEST结构查询{budget['test_queries']}次。",
        'K=0时仅等价复用；本轮是否等价：'+str(budget['K']==0)+'。训练／validation特征生成、validation模型forward、PT、神经FT、epochs、updates、GPU、worker、train＋validation refit均0。',
        '', '|任务|r11 R²±std|r10 R²±std|r9 R²±std|G05 R²±std|r11−r10|变化α数|改善／退化折数|','|---|---:|---:|---:|---:|---:|---:|---:|']
    for t in TASKS:
        r=tasks[t];s=r['stats'];lines.append('|'+t+'|'+'|'.join(f"{s[p]['r2']['mean']:.3f}±{s[p]['r2']['std']:.3f}" for p in policies)+f"|{r['differences']['r10']['r2']:+.3f}|{r['changed_units']}|{r['improved_folds']}／{r['degraded_folds']}|")
    lines+=['','|任务|r11 RMSE±std|r10 RMSE±std|r9 RMSE±std|G05 RMSE±std|r11−r10 RMSE|去掉最大增益折的ΔR²|','|---|---:|---:|---:|---:|---:|---:|']
    for t in TASKS:
        r=tasks[t];s=r['stats'];lines.append('|'+t+'|'+'|'.join(f"{s[p]['rmse']['mean']:.3f}±{s[p]['rmse']['std']:.3f}" for p in policies)+f"|{r['differences']['r10']['rmse']:+.3f}|{r['mean_delta_excluding_best_fold']:+.3f}|")
    lines+=['','|政策|Macro R²|论文数值达标数|','|---|---:|---:|']
    lines+=[f"|{p}|{macro[p]:.3f}|{result['reached_paper'][p]}/7|" for p in policies]
    lines+=['',f"主比较r11−r10 macro R² {gain:+.3f}；+0.010进展里程碑：{'达到' if result['milestone'] else '未达到'}。",
        '退化任务（R²原精度）：'+json.dumps(result['degraded_tasks'],ensure_ascii=False)+'。',
        f"最大单元增益{best_row['task']}/fold{best_row['fold']}；去掉该单元后的34单元平均ΔR² {result['concentration']['mean_delta_excluding_best_unit']:+.3f}。",'',
        '|任务|论文最佳模型|论文R²|r11−论文|r10−论文|r9−论文|G05−论文|r11−r9 R²／RMSE|r11−G05 R²／RMSE|',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for t in TASKS:
        r=tasks[t];lines.append(f"|{t}|{paper[t]['model']}|{paper[t]['r2']:.3f}|"+'|'.join(f"{r['minus_paper'][p]:+.3f}" for p in policies)+f"|{r['differences']['r9']['r2']:+.3f}／{r['differences']['r9']['rmse']:+.3f}|{r['differences']['g05']['r2']:+.3f}／{r['differences']['g05']['rmse']:+.3f}|")
    lines+=['','每任务原单位RMSE单独报告，负差表示改善，不跨物理单位平均RMSE。原精度选参、统计与门槛判定后展示3位小数。',
        '105项validation分数与来源见alpha_choices_before_models.json；35项旧／新α见alpha_changes.csv；逐折及validation→TEST差见fold_differences.csv；全原精度四政策指标见seven_task_results.json。',
        '完整模型、split、scaler、身份和固定权重冻结见selection_before_test.json；旧FEATURE_SPEC中的0.5字段只作历史元数据，不是实际选参或预测权重。',
        '共享validation／TEST已开发暴露；论文数据、split、处理及统计协议等价性未核实，仅探索性数值对照，不称独立盲测或SOTA，不把validation提升当作TEST提升。']
    (folder/'delivery.md').write_text('\n'.join(lines)+'\n')


def launch(name,argv):
    require_tmux();folder=new_directory(LOG/name)
    command=dict(argv=argv,cwd=str(ROOT),environment=ENV,window='fce_aligned_alpha_'+name);write(folder/'command.json',command)
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
if p.name=='controller':subprocess.run([sys.executable,'scripts/run_fce_aligned_alpha.py','--launch-finish'],cwd=c['cwd'],env=dict(os.environ,**c['environment']),check=False)
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
    if OUT.exists() or LOG.exists():raise FileExistsError('r11 roots already exist; no replay')
    new_directory(OUT);new_directory(LOG);folder=new_directory(LOG/'setup');started=now();rc=1
    write(folder/'command.json',dict(argv=sys.argv,cwd=str(ROOT),environment=ENV,window='fce_aligned_alpha_setup'))
    (folder/'stdout.log').touch(exist_ok=False);(folder/'stderr.log').touch(exist_ok=False)
    try:
        for name in ('CURRENT_PLAN.md','src/training/fce_param.py','src/training/fce_kernel.py'):
            target=OUT/'source_before'/name;target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(Path('/tmp/fce_aligned_alpha_before_20261016')/name,target)
        for name in (*EDITED,'src/training/xc_kernel.py'):
            target=OUT/'source_used'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,target)
        launch('controller',[PYTHON,'scripts/run_fce_aligned_alpha.py','--run']);rc=0
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
    s+='\n\nr11实际预算：'+json.dumps(read(OUT/'consumption.json'),ensure_ascii=False)+'。真实exit：'+str([(Path(r['log_dir']).name,r['exit_txt']) for r in records])+'。收尾自身真实exit见logs/fce_tune_20261016/aligned_alpha/finish_delivery/exit.txt；本索引写入时不伪造尚未退出的收尾状态。完整退出索引final_review/exit_index.json。\n'
    s+='结果results/fce_tune_20261016/aligned_alpha/，日志logs/fce_tune_20261016/aligned_alpha/；tmux Uni-Poly:fce_aligned_alpha_setup/controller/final_cpu/finish_delivery；本轮增量final_review/aligned_alpha_only_changes.diff。全部旧实验及产物保留，没有删除或新增哈希验证，无独立审查。\n'
    p.write_text(s);changes=[]
    for name in EDITED:
        before=OUT/'source_before'/name;previous=before.read_text().splitlines(True) if before.exists() else []
        changes.extend(difflib.unified_diff(previous,(ROOT/name).read_text().splitlines(True),fromfile='before/'+name,tofile='after/'+name))
    (review/'aligned_alpha_only_changes.diff').write_text(''.join(changes))
    checked=subprocess.run(['git','diff','--check','--','CURRENT_PLAN.md','src/training/fce_param.py'],cwd=ROOT,capture_output=True,text=True)
    write(review/'diff_format_check.json',dict(exit=checked.returncode,stdout=checked.stdout,stderr=checked.stderr))
    print('delivery ready; science/test exits:',[(Path(r['log_dir']).name,r['exit_txt']) for r in records],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--setup',action='store_true');p.add_argument('--run',action='store_true');p.add_argument('--finish',action='store_true');p.add_argument('--launch-finish',action='store_true');a=p.parse_args()
    if a.setup:setup()
    elif a.run:run()
    elif a.launch_finish:launch('finish_delivery',[PYTHON,'scripts/run_fce_aligned_alpha.py','--finish'])
    elif a.finish:finish()
    else:p.error('explicit role required')
