"""r12: 35 new radius3 KRR models, fixed inherited alphas and global .75."""
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
from src.training.fce_kernel import (TASKS,RADIUS3_FEATURE_SPEC,RADIUS3_POLICY,
    Radius3Cache,Radius3Gate,AlignedAlphaGate,bind_branch_arrays,metrics,new_directory)
from src.training.fce_param import (RADIUS3_FINETUNE_POLICY,
    fit_radius3_finetune_predictions,predict_radius3_finetune_predictions)
OUT=ROOT/'results/fce_tune_20261016/radius3'
LOG=ROOT/'logs/fce_tune_20261016/radius3'
R11=ROOT/'results/fce_tune_20261016/aligned_alpha'
EDITED=('CURRENT_PLAN.md','src/training/fce_kernel.py','src/training/fce_param.py',
    'scripts/run_fce_radius3.py','tests/test_fce_radius3.py')
ENV=dict(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
    NUMEXPR_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1',BLIS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1')
PYTHON='/opt/conda/bin/python'
LIMITS=dict(fit_attempts=35,fit_completed=35,new_models=35,validation_reads=35,
    validation_predictions=35,validation_outputs=35,formal_starts=35,formal_completed=35,
    formal_outputs=70,feature_generations=5875,cache_hits=29375,train_queries=18786,
    validation_queries=4714,test_queries=5875,final_cpu_starts=1,pt=0,neural_ft=0,epochs=0,
    optimizer_updates=0,neural_inference=0,gpu=0,workers=0,refit=0)


def now():return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
def read(p):return json.loads(Path(p).read_text())
def write(p,value,replace=False):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists() and not replace:raise FileExistsError(p)
    p.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def check_budget(b):
    if set(b)!=set(LIMITS) or any(type(v) is not int or not 0<=v<=LIMITS[k] for k,v in b.items()):raise ValueError('r12 budget mismatch')
    if b['fit_completed']>b['fit_attempts'] or b['formal_completed']>b['formal_starts']:raise ValueError('r12 completion count mismatch')
def require_tmux():
    if not os.environ.get('TMUX') or subprocess.check_output(['tmux','display-message','-p','#S'],text=True).strip()!='Uni-Poly':raise RuntimeError('requires Uni-Poly tmux')
def require_environment():
    if any(os.environ.get(k)!=v for k,v in ENV.items()):raise ValueError('CPU environment mismatch')
def load_npz(path,gate=None,test=False):
    if test:
        if gate is None:raise ValueError('TEST requires radius3 freeze')
        gate.require_test()
    with np.load(path,allow_pickle=False) as v:return {k:v[k].copy() for k in v.files}


def save_predictions(path,u,truth,g05,branch,fusion,identities,role):
    if Path(path).exists():raise FileExistsError(path)
    indices=u[role+'_indices']
    np.savez(path,indices=np.asarray(indices),sample_keys=[identities[str(i)]['sample_key'] for i in indices],
        y_true=truth,y_g05=g05,y_krr=branch,y_pred=fusion,alpha=u['alpha'],prediction_weight=.75,
        feature_spec=json.dumps(RADIUS3_FEATURE_SPEC,sort_keys=True),scaler_mean=u['scaler']['mean'],scaler_scale=u['scaler']['scale'])


def run():
    require_tmux();require_environment();started=now();b={k:0 for k in LIMITS};phase='initialization'
    def ledger():check_budget(b);write(OUT/'consumption.json',b,True)
    try:
        ledger();prior=read(R11/'selection_before_test.json');old_gate=AlignedAlphaGate();old_gate.freeze(prior)
        if RADIUS3_FINETUNE_POLICY!=RADIUS3_POLICY:raise ValueError('radius3 public policy mismatch')
        versions=dict(rdkit=rdkit.__version__,sklearn=sklearn.__version__,numpy=np.__version__,scipy=scipy.__version__)
        if versions!=prior['versions']:raise ValueError('radius3 dependency versions mismatch')
        pools=threadpool_info()
        if any(p['num_threads']!=1 for p in pools):raise ValueError('CPU numerical thread count mismatch')
        if {r:sum(len(u[r+'_indices']) for u in prior['units']) for r in ('train','validation','test')}!={'train':18786,'validation':4714,'test':5875}:raise ValueError('source split role totals mismatch')
        write(OUT/'policy.json',dict(policy=RADIUS3_POLICY,versions=versions,threadpools=pools,alpha_origin=str(R11/'selection_before_test.json')))
        def on_query(role,hit):
            b[role+'_queries']+=1;b['cache_hits' if hit else 'feature_generations']+=1;check_budget(b)
        cache=Radius3Cache(on_query);identities=prior['identities'];units=[];phase='fit_validation'
        for source in prior['units']:
            u=copy.deepcopy(source);u.update(source_radius2_model=source['model'],radius3_model_ready=False,model_spec=RADIUS3_FEATURE_SPEC,prediction_weight=.75,
                alpha_origin=dict(path=str(R11/'selection_before_test.json'),task=source['task'],fold=source['fold'],alpha=source['alpha']))
            for role in ('train','validation'):
                for row in u[role+'_samples']:
                    if identities[u['task']][str(row['original_row'])]!={k:row[k] for k in ('sample_key','normalized_smiles')}:raise ValueError('inherited source identity mismatch')
            folder=new_directory(OUT/'models'/u['task']/f"fold{u['fold']}")
            def on_fit(event):b['fit_attempts' if event=='attempted' else 'fit_completed']+=1;ledger()
            model=fit_radius3_finetune_predictions(load_npz(source['model']),source,cache,on_fit=on_fit)
            u['model']=str(folder/'selected_model.npz');np.savez(u['model'],**model);b['new_models']+=1
            chosen=[s for s in source['validation_sources'] if s['alpha']==source['alpha']]
            if len(chosen)!=1:raise ValueError('inherited alpha validation path mismatch')
            old_validation=load_npz(chosen[0]['path']);b['validation_reads']+=1;ledger()
            aligned=bind_branch_arrays(old_validation,u,'validation',identities[u['task']],('y_g05',))
            actual=metrics(aligned['y_true'],aligned['y_g05'])
            if any(not np.isclose(actual[m],source['chosen_validation']['reference'][m],rtol=1e-12,atol=1e-12) for m in actual):raise ValueError('validation G05 reference mismatch')
            truth,g05,branch,fusion=predict_radius3_finetune_predictions(model,u,old_validation,identities[u['task']],cache,'validation')
            b['validation_predictions']+=1
            save_predictions(folder/'validation_predictions.npz',u,truth,g05,branch,fusion,identities[u['task']],'validation');b['validation_outputs']+=1
            u.update(radius3_model_ready=True,validation_source=chosen[0]['path'],validation_diagnostic=dict(reference=actual,branch=metrics(truth,branch),fused=metrics(truth,fusion)))
            write(folder/'binding.json',u);units.append(u);ledger()
            print('new radius3 fit and validation complete',u['task'],u['fold'],flush=True)
        if (b['fit_completed'],b['new_models'],b['validation_predictions'])!=(35,35,35):raise ValueError('all radius3 models required')
        frozen=dict(policy=RADIUS3_POLICY,units=units,r11_freeze=prior,identities=identities,versions=versions,frozen=True,frozen_at=now(),budget_at_freeze=copy.deepcopy(b),reference_test_opened=False)
        gate=Radius3Gate();gate.freeze(frozen);write(OUT/'selection_before_test.json',frozen);phase='formal'
        gate.require_test();old_summary=read(R11/'final_review/seven_task_results.json')
        old_rows={(r['task'],r['fold']):r for r in old_summary['folds']}
        r10_summary=read(ROOT/'results/fce_tune_20261016/global_blend/final_review/seven_task_results.json')
        r10_rows={(r['task'],r['fold']):r for r in r10_summary['folds']}
        rows=[]
        for u in units:
            gate.require_test();b['formal_starts']+=1;ledger();key=(u['task'],u['fold']);old=old_rows[key];old10=r10_rows[key]
            if old['alpha']!=u['alpha']:raise ValueError('inherited r11 alpha mismatch at TEST')
            reference=load_npz(old10['paths']['g05'],gate,True)
            truth,g05,branch,fusion=predict_radius3_finetune_predictions(load_npz(u['model']),u,reference,identities[u['task']],cache,'test',gate)
            p11=bind_branch_arrays(load_npz(old['paths']['r11'],gate,True),u,'test',identities[u['task']],('y_pred',))
            old10unit=dict(u,alpha=old10['alpha'])
            p10=bind_branch_arrays(load_npz(old10['output'],gate,True),old10unit,'test',identities[u['task']],('y_pred',))
            if not np.array_equal(truth,p11['y_true']) or not np.array_equal(truth,p10['y_true']):raise ValueError('formal policy truth mismatch')
            values={p:metrics(truth,v) for p,v in [('r12',fusion),('r11',p11['y_pred']),('r10',p10['y_pred'])]}
            if any(values[p]!=old['metrics'][p] for p in ('r11','r10')):raise ValueError('formal old policy metrics mismatch')
            folder=new_directory(OUT/'formal'/u['task']/f"fold{u['fold']}");paths={}
            for label,value in [('branch',branch),('r12',fusion)]:
                path=folder/(label+'_predictions.npz');save_predictions(path,u,truth,g05,branch,value,identities[u['task']],'test');paths[label]=str(path);b['formal_outputs']+=1;ledger()
            record=dict(task=u['task'],fold=u['fold'],alpha=u['alpha'],metrics=values,branch=metrics(truth,branch),paths=paths,
                validation=u['validation_diagnostic'],difference={p:{m:values['r12'][m]-values[p][m] for m in ('r2','rmse')} for p in ('r11','r10')},
                test_minus_validation={m:values['r12'][m]-u['validation_diagnostic']['fused'][m] for m in ('r2','rmse')})
            write(folder/'metrics.json',record);rows.append(record);b['formal_completed']+=1;ledger()
        if (b['fit_completed'],b['validation_outputs'],b['formal_completed'],b['formal_outputs'])!=(35,35,35,70):raise ValueError('incomplete radius3 matrix')
        phase='report';report(rows,units,old_summary,b)
        b['final_cpu_starts']+=1;ledger();test=launch('final_cpu',['timeout','300s',PYTHON,'-m','pytest','-q','tests/test_fce_radius3.py'])
        rc=wait_exit(test);status='待审查' if rc==0 else '需返修'
        write(OUT/'completion.json',dict(status=status,start=started,end=now(),test_exit=rc,consumption=b,independent_review='NOT_PERFORMED'))
        update_plan(status,(OUT/'final_review/delivery.md').read_text()+f'\n唯一CPU检查真实exit {rc}。\n')
    except BaseException as error:
        write(OUT/'consumption.json',b,True);write(OUT/'failure.json',dict(phase=phase,error=repr(error),traceback=traceback.format_exc(),consumption=b,time=now()))
        update_plan('需返修','执行停止：'+repr(error)+'；failure.json保留现场，不自动重跑或扩大预算。');raise


def update_plan(status,body):
    p=ROOT/'CURRENT_PLAN.md';s=p.read_text().replace('状态：**执行中（r12）**。','状态：**'+status+'（r12）**。',1)
    s=s.replace('## 59. r12：七任务统一Morgan半径表示对照（执行中）','## 59. r12：七任务统一Morgan半径表示对照（'+status+'）',1)
    s=s.replace('已收到本轮用户明确执行授权，状态执行中。实现及科学执行按第59节范围推进；初始预算消耗0，最终CPU检查仅在全部科学执行完成后一次。所有实际状态以新增日志和产物为据，不重跑第56节或旧周期。','本轮执行状态：'+status+'。实际科学消耗、最终唯一CPU检查及退出证据见第61节与新结果根；不重跑第56节或旧周期。',1)
    s+='\n\n## 61. r12执行交接\n\n'+body+'\n';p.write_text(s)


def report(rows,units,prior,budget):
    folder=new_directory(OUT/'final_review');paper={t:prior['tasks'][t]['paper'] for t in TASKS};tasks={};policies=('r12','r11','r10')
    for t in TASKS:
        selected=[r for r in rows if r['task']==t]
        stats={p:{m:dict(mean=float(np.mean([r['metrics'][p][m] for r in selected])),std=float(np.std([r['metrics'][p][m] for r in selected],ddof=0))) for m in ('r2','rmse')} for p in policies}
        gains=[r['difference']['r11']['r2'] for r in selected]
        tasks[t]=dict(stats=stats,paper=paper[t],minus_paper={p:stats[p]['r2']['mean']-paper[t]['r2'] for p in policies},
            differences={p:{m:stats['r12'][m]['mean']-stats[p][m]['mean'] for m in ('r2','rmse')} for p in ('r11','r10')},
            improved_folds=sum(x>0 for x in gains),degraded_folds=sum(x<0 for x in gains),mean_delta_excluding_best_fold=float((sum(gains)-max(gains))/4))
    macro={p:float(np.mean([tasks[t]['stats'][p]['r2']['mean'] for t in TASKS])) for p in policies};gain=macro['r12']-macro['r11']
    best_unit=max(rows,key=lambda r:r['difference']['r11']['r2']);best_task=max(TASKS,key=lambda t:tasks[t]['differences']['r11']['r2'])
    concentration=dict(best_task=best_task,best_unit=[best_unit['task'],best_unit['fold']],
        mean_delta_excluding_best_unit=float((sum(r['difference']['r11']['r2'] for r in rows)-best_unit['difference']['r11']['r2'])/34),
        macro_delta_excluding_best_task=float(np.mean([tasks[t]['differences']['r11']['r2'] for t in TASKS if t!=best_task])))
    result=dict(policy=RADIUS3_POLICY,tasks=tasks,macro_r2=macro,delta_macro_r2=gain,delta_macro_vs_r10=macro['r12']-macro['r10'],milestone=gain>=.010,
        reached_paper={p:sum(tasks[t]['stats'][p]['r2']['mean']>=paper[t]['r2'] for t in TASKS) for p in policies},
        degraded_tasks={p:[t for t in TASKS if tasks[t]['differences'][p]['r2']<0] for p in ('r11','r10')},
        improved_tasks={p:[t for t in TASKS if tasks[t]['differences'][p]['r2']>0] for p in ('r11','r10')},concentration=concentration,folds=rows,
        consumption_at_science_end=copy.deepcopy(budget),aggregation='five folds mean/std ddof=0, G05 seed42+fold; equal seven-task R2 macro',
        protocol='shared validation/TEST development exposed; paper equivalence unverified',independent_review='NOT_PERFORMED')
    write(folder/'seven_task_results.json',result)
    write(folder/'model_prediction_index.json',[dict(task=u['task'],fold=u['fold'],alpha=u['alpha'],alpha_origin=u['alpha_origin'],model=u['model'],
        validation=str(Path(u['model']).parent/'validation_predictions.npz'),formal=next(r['paths'] for r in rows if (r['task'],r['fold'])==(u['task'],u['fold']))) for u in units])
    with (folder/'fold_differences.csv').open('x',newline='') as f:
        fields=['task','fold','alpha','r12_minus_r11_r2','r12_minus_r11_rmse','r12_minus_r10_r2','r12_minus_r10_rmse','validation_r2','validation_rmse','test_minus_validation_r2','test_minus_validation_rmse'];w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for r in rows:w.writerow(dict(task=r['task'],fold=r['fold'],alpha=r['alpha'],**{f'r12_minus_{p}_{m}':r['difference'][p][m] for p in ('r11','r10') for m in ('r2','rmse')},
            validation_r2=r['validation']['fused']['r2'],validation_rmse=r['validation']['fused']['rmse'],test_minus_validation_r2=r['test_minus_validation']['r2'],test_minus_validation_rmse=r['test_minus_validation']['rmse']))
    lines=['状态：待审查；同一主体执行及自检，独立审查未进行。',
        '唯一变化Morgan radius2→3；2048位及其他特征设置不变，w=0.75、35项α固定继承r11，全任务同一公共接口。validation仅诊断，未选参。',
        f"新增fit attempted/completed={budget['fit_attempts']}/{budget['fit_completed']}，新模型{budget['new_models']}，validation读取／预测／数组各{budget['validation_outputs']}，正式单元{budget['formal_completed']}／正式数组{budget['formal_outputs']}。",
        f"角色查询train {budget['train_queries']}、validation {budget['validation_queries']}、TEST {budget['test_queries']}；实际结构生成{budget['feature_generations']}、缓存命中{budget['cache_hits']}。纯结构缓存无标签／模型／预测。PT／神经FT／epochs／updates／神经推理／GPU／worker／train＋validation refit均0。",'',
        '|任务|r12 R²±std|r11 R²±std|r10 R²±std|r12−r11 R²|r12−r10 R²|改善／退化折数vs r11|','|---|---:|---:|---:|---:|---:|---:|']
    for t in TASKS:
        r=tasks[t];s=r['stats'];lines.append('|'+t+'|'+'|'.join(f"{s[p]['r2']['mean']:.3f}±{s[p]['r2']['std']:.3f}" for p in policies)+f"|{r['differences']['r11']['r2']:+.3f}|{r['differences']['r10']['r2']:+.3f}|{r['improved_folds']}／{r['degraded_folds']}|")
    lines+=['','|任务|r12 RMSE±std|r11 RMSE±std|r10 RMSE±std|r12−r11 RMSE|r12−r10 RMSE|去掉最大增益fold后的ΔR²|','|---|---:|---:|---:|---:|---:|---:|']
    for t in TASKS:
        r=tasks[t];s=r['stats'];lines.append('|'+t+'|'+'|'.join(f"{s[p]['rmse']['mean']:.3f}±{s[p]['rmse']['std']:.3f}" for p in policies)+f"|{r['differences']['r11']['rmse']:+.3f}|{r['differences']['r10']['rmse']:+.3f}|{r['mean_delta_excluding_best_fold']:+.3f}|")
    lines+=['','|政策|Macro R²|论文数值达标数|','|---|---:|---:|']
    lines+=[f"|{p}|{macro[p]:.3f}|{result['reached_paper'][p]}/7|" for p in policies]
    lines+=['',f"r12−r11 macro R² {gain:+.3f}，r12−r10 {result['delta_macro_vs_r10']:+.3f}；+0.010进展里程碑：{'达到' if result['milestone'] else '未达到'}。",
        '改善任务：'+json.dumps(result['improved_tasks'],ensure_ascii=False)+'；退化任务：'+json.dumps(result['degraded_tasks'],ensure_ascii=False)+'。',
        f"最大增益单元{concentration['best_unit']}，去掉后其余34单元平均ΔR² {concentration['mean_delta_excluding_best_unit']:+.3f}；最大增益任务{best_task}，去掉后六任务macro差{concentration['macro_delta_excluding_best_task']:+.3f}。",'',
        '|任务|论文最佳模型|论文R²|r12−论文|r11−论文|r10−论文|','|---|---|---:|---:|---:|---:|']
    for t in TASKS:
        r=tasks[t];lines.append(f"|{t}|{paper[t]['model']}|{paper[t]['r2']:.3f}|"+'|'.join(f"{r['minus_paper'][p]:+.3f}" for p in policies)+'|')
    lines+=['','逐折与validation→TEST差见fold_differences.csv；35项继承α及来源、新模型、validation与正式预测路径见model_prediction_index.json；全局冻结selection_before_test.json。',
        '各任务RMSE使用原单位，负差表示改善，不跨单位平均RMSE。原精度统计及门槛判定后展示3位小数。',
        '共享validation／TEST已开发暴露，论文数据／split／评估统计口径等价性未核实，仅探索性数值对照，不宣称独立盲测、稳定优势或超过论文；无按任务回退或候选替换。']
    (folder/'delivery.md').write_text('\n'.join(lines)+'\n')


def launch(name,argv):
    require_tmux();folder=new_directory(LOG/name)
    command=dict(argv=argv,cwd=str(ROOT),environment=ENV,window='fce_radius3_'+name);write(folder/'command.json',command)
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
if p.name=='controller':subprocess.run([sys.executable,'scripts/run_fce_radius3.py','--launch-finish'],cwd=c['cwd'],env=dict(os.environ,**c['environment']),check=False)
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
    if OUT.exists() or LOG.exists():raise FileExistsError('r12 roots already exist; no replay')
    new_directory(OUT);new_directory(LOG);folder=new_directory(LOG/'setup');started=now();rc=1
    write(folder/'command.json',dict(argv=sys.argv,cwd=str(ROOT),environment=ENV,window='fce_radius3_setup'))
    (folder/'stdout.log').touch(exist_ok=False);(folder/'stderr.log').touch(exist_ok=False)
    try:
        for name in ('CURRENT_PLAN.md','src/training/fce_param.py','src/training/fce_kernel.py'):
            target=OUT/'source_before'/name;target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(Path('/tmp/fce_radius3_before_20261016')/name,target)
        for name in (*EDITED,'src/training/xc_kernel.py'):
            target=OUT/'source_used'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,target)
        launch('controller',[PYTHON,'scripts/run_fce_radius3.py','--run']);rc=0
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
    s+='\n\nr12实际预算：'+json.dumps(read(OUT/'consumption.json'),ensure_ascii=False)+'。真实exit：'+str([(Path(r['log_dir']).name,r['exit_txt']) for r in records])+'。收尾自身真实exit见logs/fce_tune_20261016/radius3/finish_delivery/exit.txt；本索引写入时不伪造尚未退出的收尾状态。完整退出索引final_review/exit_index.json。\n'
    s+='结果results/fce_tune_20261016/radius3/，日志logs/fce_tune_20261016/radius3/；tmux Uni-Poly:fce_radius3_setup/controller/final_cpu/finish_delivery；本轮增量final_review/radius3_only_changes.diff。全部旧实验及产物保留，没有删除或新增哈希验证，无独立审查。\n'
    p.write_text(s);changes=[]
    for name in EDITED:
        before=OUT/'source_before'/name;previous=before.read_text().splitlines(True) if before.exists() else []
        changes.extend(difflib.unified_diff(previous,(ROOT/name).read_text().splitlines(True),fromfile='before/'+name,tofile='after/'+name))
    (review/'radius3_only_changes.diff').write_text(''.join(changes))
    checked=subprocess.run(['git','diff','--check','--','CURRENT_PLAN.md','src/training/fce_param.py'],cwd=ROOT,capture_output=True,text=True)
    write(review/'diff_format_check.json',dict(exit=checked.returncode,stdout=checked.stdout,stderr=checked.stderr))
    print('delivery ready; science/test exits:',[(Path(r['log_dir']).name,r['exit_txt']) for r in records],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--setup',action='store_true');p.add_argument('--run',action='store_true');p.add_argument('--finish',action='store_true');p.add_argument('--launch-finish',action='store_true');a=p.parse_args()
    if a.setup:setup()
    elif a.run:run()
    elif a.launch_finish:launch('finish_delivery',[PYTHON,'scripts/run_fce_radius3.py','--finish'])
    elif a.finish:finish()
    else:p.error('explicit role required')
