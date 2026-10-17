"""r9: one uniform post-FT policy, 35 frozen units; reuse 10, compute 25."""
import argparse
import copy
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
import rdkit
import scipy
import sklearn
from threadpoolctl import threadpool_info
from src.training.fce_param import (TASKS,FINETUNE_PREDICTION_POLICY,
    fit_finetune_predictions,predict_finetune_predictions)
from src.training.fce_kernel import (ALPHAS,FEATURE_SPEC,POLICY,LIMITS,PhaseGate,
    check_budget,validate_split,validate_reference,validate_model,metadata,new_directory,
    align_reference,choose_alpha,metrics,scaler_values)

OUT=ROOT/'results/fce_tune_20261016/all_task_kernel'
LOG=ROOT/'logs/fce_tune_20261016/all_task_kernel'
REFERENCE=ROOT/'results/fce_tune_20261015/t3b'
SOURCE=ROOT/'data/processed/mcl_ph_r4_20260930/downstream_direct_01.staging'
SPLIT=ROOT/'data/splits/mips_paper5_outer'
REUSE=(ROOT/'results/fce_tune_20261016/xc_kernel',ROOT/'results/fce_tune_20261016/ei_kernel/recovery2')
PYTHON='/opt/conda/bin/python'
ENV=dict(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
    NUMEXPR_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1',BLIS_NUM_THREADS='1',
    PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1')
EDITED=('CURRENT_PLAN.md','src/training/fce_param.py','src/training/fce_kernel.py',
    'scripts/run_fce_all_kernel.py','tests/test_fce_all_kernel.py')


def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def read(p):return json.loads(Path(p).read_text())
def write(p,value,*,replace=False):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists() and not replace:raise FileExistsError(p)
    p.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def load_npz(p):
    with np.load(p,allow_pickle=False) as v:return {k:v[k].copy() for k in v.files}
def require_tmux():
    if not os.environ.get('TMUX') or subprocess.check_output(['tmux','display-message','-p','#S'],text=True).strip()!='Uni-Poly':
        raise RuntimeError('requires tmux Uni-Poly')
def require_environment():
    if any(os.environ.get(k)!=v for k,v in ENV.items()):raise ValueError('CPU environment mismatch')


def source_rows(task,indices,manifest,fold,role,gate=None):
    if role=='test':
        if gate is None:raise ValueError('TEST before global freeze')
        gate.require_test()
    if indices!=manifest['folds'][fold][role+'_indices']:raise ValueError('source/fold role mismatch')
    offsets=np.load(SOURCE/(task+'_offsets.npy'),allow_pickle=False)
    if offsets.dtype!=np.int64 or offsets.shape!=(manifest['sample_count'],):raise ValueError('source offsets mismatch')
    rows=[]
    with (SOURCE/(task+'.jsonl')).open('rb') as stream:
        for i in indices:
            stream.seek(int(offsets[i]));row=json.loads(stream.readline())
            if row['task']!=task or row['original_row']!=i or row['fold_membership'][fold]!={'fold':fold,'role':role}:
                raise ValueError('source identity/role mismatch')
            if not row['normalized_smiles'] or not row['sample_key'] or not np.isfinite(row['label']):raise ValueError('invalid source data')
            rows.append(row)
    return rows


def reference_binding(task,fold,entry,references):
    matches=[u for u in references if (u['task'],u['fold'],u['arm'])==(task,fold,'T3B_G05')]
    if len(matches)!=1:raise ValueError('missing fixed reference')
    unit=matches[0];folder=REFERENCE/'finetune/T3B_G05'/task/f'fold{fold}'
    scaler=read(folder/'scaler.json');record=read(folder/'metrics.json')
    validate_reference(unit,task,fold,entry,scaler,record)
    expected=dict(parent=str(REFERENCE/'pretrain/T3B_G05/deploy_05000.pt'),
        head_initial=str(ROOT/'results/glt_fusion_20261007/r1/finetune/B0'/task/f'fold{fold}'/'head_initial.pt'),best=str(folder/'best.pt'))
    if any(unit[k]!=v or not Path(v).is_file() for k,v in expected.items()):raise ValueError('reference checkpoint path mismatch')
    return unit,scaler


def reuse_registry(versions):
    registry={}
    for root in REUSE:
        frozen=read(root/'selection_before_test.json')
        if not frozen['frozen'] or frozen['feature_spec']!=FEATURE_SPEC or frozen['reference_root']!=str(REFERENCE) or frozen['selection_rule']!=POLICY['selection']:
            raise ValueError('reuse policy/reference mismatch')
        if any(frozen['versions'][k]!=versions[k] for k in ('rdkit','sklearn','numpy','scipy')):raise ValueError('reuse numerical dependency versions mismatch')
        for old in frozen['units']:
            task=old['reference']['task'];key=(task,old['fold'])
            if task not in TASKS or key in registry:raise ValueError('duplicate/unsupported reuse unit')
            registry[key]=(root,old,frozen['candidate'])
    if len(registry)!=10:raise ValueError('expected ten predeclared reuse units')
    return registry


def restore_unit(root,old,candidate,unit,train,validation,reference):
    for k in ('fold','alpha','scaler','reference','train_indices','validation_indices','test_indices'):
        if k!='alpha' and old[k]!=unit[k]:raise ValueError('reuse '+k+' mismatch')
    if old['fit_count']!=3 or choose_alpha(old['validation_trials'])!=old['chosen_validation'] or old['alpha']!=old['chosen_validation']['alpha']:
        raise ValueError('reuse selection mismatch')
    unit.update(alpha=old['alpha'],model=old['model'],fit_count=3,
        validation_trials=old['validation_trials'],chosen_validation=old['chosen_validation'],
        train_samples=metadata(train),validation_samples=metadata(validation),reuse_root=str(root),reuse_candidate=candidate)
    model=load_npz(unit['model']);validate_model(model,unit)
    if not np.array_equal(model['train_y'],np.asarray([r['label'] for r in train],dtype=np.float64)):
        raise ValueError('reuse training targets mismatch')
    truth,g05=align_reference(reference,validation,unit['scaler'])
    folder=Path(unit['model']).parent
    for trial in unit['validation_trials']:
        path=folder/'validation'/f"alpha_{trial['alpha']:g}";v=load_npz(path/'predictions.npz')
        if read(path/'metrics.json')!=trial or trial['fit_rows']!=len(train):raise ValueError('reuse validation record mismatch')
        projected=dict(v,y_pred=v['y_g05']);yt,ref=align_reference(projected,validation,unit['scaler'])
        if not np.array_equal(yt,truth) or not np.array_equal(ref,g05) or not np.array_equal(v['y_fused'],.5*v['y_g05']+.5*v['y_krr']):
            raise ValueError('reuse validation identity/fusion mismatch')
        for key,values in [('reference',v['y_g05']),('branch',v['y_krr']),('fused',v['y_fused'])]:
            actual=metrics(v['y_true'],values)
            if any(not np.isclose(actual[m],trial[key][m],rtol=1e-12,atol=1e-12) for m in actual):raise ValueError('reuse validation metrics mismatch')
    return unit


def run():
    require_tmux();require_environment();budget={k:0 for k in LIMITS};started=now();phase='initialization'
    def ledger():check_budget(budget);write(OUT/'consumption.json',budget,replace=True)
    try:
        if FINETUNE_PREDICTION_POLICY!=POLICY:raise ValueError('public FT policy diverged')
        versions=dict(rdkit=rdkit.__version__,sklearn=sklearn.__version__,numpy=np.__version__,scipy=scipy.__version__,threadpools=threadpool_info())
        if any(t['num_threads']!=1 for t in versions['threadpools']):raise ValueError('thread budget mismatch')
        write(OUT/'policy.json',dict(policy=POLICY,versions=versions,new_limits=LIMITS))
        registry=reuse_registry(versions);references=read(REFERENCE/'selection_before_test.json')['units']
        manifests={};units=[];identities={};gate=PhaseGate();ledger();phase='fit_validation'
        for task in TASKS:
            manifest=read(SPLIT/(task+'.json'));manifests[task]=manifest;features={};identities[task]={}
            if manifest['task']!=task or manifest['protocol']!='paper5_outer' or len(manifest['folds'])!=5:raise ValueError('manifest schema mismatch')
            for fold,entry in enumerate(manifest['folds']):
                validate_split(entry,manifest['sample_count'])
                if entry['fold']!=fold:raise ValueError('fold order mismatch')
                binding,scaler=reference_binding(task,fold,entry,references)
                train=source_rows(task,entry['train_indices'],manifest,fold,'train')
                validation=source_rows(task,entry['validation_indices'],manifest,fold,'validation')
                for row in train+validation:
                    key=str(row['original_row']);value={k:row[k] for k in ('sample_key','normalized_smiles')}
                    if key in identities[task] and identities[task][key]!=value:raise ValueError('conflicting source identities')
                    identities[task][key]=value
                ref=load_npz(REFERENCE/'finetune/T3B_G05'/task/f'fold{fold}'/'validation_predictions.npz')
                unit=dict(task=task,fold=fold,sample_count=manifest['sample_count'],
                    train_indices=entry['train_indices'],validation_indices=entry['validation_indices'],test_indices=entry['test_indices'],
                    reference=binding,scaler=scaler)
                folder=new_directory(OUT/'folds'/task/f'fold{fold}')
                if (task,fold) in registry:
                    root,old,candidate=registry[(task,fold)]
                    unit=restore_unit(root,old,candidate,unit,train,validation,ref)
                else:
                    def on_fit(kind):budget['fit_attempts' if kind=='attempted' else 'fit_completed']+=1;ledger()
                    def on_trial(alpha,values,record):
                        path=new_directory(folder/'validation'/f'alpha_{alpha:g}')
                        np.savez(path/'predictions.npz',**values);write(path/'metrics.json',record)
                    result=fit_finetune_predictions(train,validation,ref,scaler,on_fit=on_fit,on_trial=on_trial,features=features)
                    model=result.pop('model');modelpath=folder/'selected_model.npz';np.savez(modelpath,**model)
                    unit.update(**result,model=str(modelpath),fit_count=3,reuse_root=None)
                    validate_model(model,unit)
                write(folder/'selection.json',unit);units.append(unit)
                print(f"validation complete {task}/fold{fold}; new fits {budget['fit_completed']}/75; reused={bool(unit['reuse_root'])}",flush=True)
            if set(identities[task])!=set(map(str,range(manifest['sample_count']))):raise ValueError('source identity coverage mismatch')
        if len(units)!=35 or budget['fit_completed']!=75:raise ValueError('incomplete validation matrix')
        payload=dict(policy=copy.deepcopy(POLICY),units=units,identities=identities,frozen=True,frozen_at=now(),
            budget_at_freeze=copy.deepcopy(budget),manifests=manifests,reference_test_development_exposed=True)
        gate.freeze(payload);write(OUT/'selection_before_test.json',payload);phase='formal'
        formal=[]
        for u in gate.payload['units']:
            gate.require_test();task=u['task'];fold=u['fold'];scaler=u['scaler'];manifest=manifests[task]
            rows=source_rows(task,u['test_indices'],manifest,fold,'test',gate)
            for row in rows:
                if identities[task][str(row['original_row'])]!={k:row[k] for k in ('sample_key','normalized_smiles')}:raise ValueError('TEST identity differs from freeze')
            reference_path=REFERENCE/'formal/T3B_G05'/task/f'fold{fold}'/'test_predictions.npz'
            ref=load_npz(reference_path);truth,g05=align_reference(ref,rows,scaler)
            folder=new_directory(OUT/'formal'/task/f'fold{fold}')
            paths={}
            if u['reuse_root']:
                root=Path(u['reuse_root'])/'formal'/f'fold{fold}'
                for label,file in [('branch','KRR_SELECTED'),('fused',u['reuse_candidate'])]:paths[label]=str(root/(file+'_predictions.npz'))
                values={}
                for label,path in paths.items():
                    p=load_npz(path);yt,values[label]=align_reference(p,rows,scaler)
                    _,oldg05=align_reference(dict(p,y_pred=p['y_g05']),rows,scaler)
                    if not np.array_equal(yt,truth) or not np.array_equal(oldg05,g05) or float(p['alpha'])!=u['alpha']:raise ValueError('cached TEST/reference mismatch')
                branch=values['branch'];fusion=values['fused']
                if not np.array_equal(fusion,.5*g05+.5*branch):raise ValueError('cached fusion policy mismatch')
                budget['reused_units']+=1;budget['reused_prediction_groups']+=2;ledger()
                write(folder/'reused_predictions.json',paths)
            else:
                budget['formal_starts']+=1;ledger();model=validate_model(load_npz(u['model']),u)
                truth,g05,branch,fusion=predict_finetune_predictions(model,rows,ref,scaler)
                for label,pred in [('branch',branch),('fused',fusion)]:
                    path=folder/(label+'_predictions.npz')
                    np.savez(path,indices=np.asarray(u['test_indices']),sample_keys=np.asarray([r['sample_key'] for r in rows]),
                        y_true=truth,y_pred=pred,y_g05=g05,alpha=u['alpha'],scaler_mean=scaler['mean'],scaler_scale=scaler['scale'])
                    paths[label]=str(path);budget['prediction_groups']+=1;ledger()
                budget['formal_completed']+=1;ledger()
            record=dict(task=task,fold=fold,alpha=u['alpha'],reference=metrics(truth,g05),branch=metrics(truth,branch),
                fused=metrics(truth,fusion),validation=u['chosen_validation'],reused=bool(u['reuse_root']),paths=paths,reference_path=str(reference_path))
            write(folder/'metrics.json',record);formal.append(record)
        if (budget['formal_completed'],budget['reused_units'],budget['prediction_groups'],budget['reused_prediction_groups'])!=(25,10,50,20):raise ValueError('incomplete formal matrix')
        phase='report';report(formal,budget)
        budget['final_cpu_starts']+=1;ledger()
        test=launch('final_cpu',['timeout','300s',PYTHON,'-m','pytest','-q','tests/test_fce_all_kernel.py'])
        rc=wait_exit(test);status='待审查' if rc==0 else '需返修'
        write(OUT/'completion.json',dict(status=status,start=started,end=now(),test_exit=rc,consumption=budget,independent_review='NOT_PERFORMED'))
        p=ROOT/'CURRENT_PLAN.md';text=p.read_text().replace('状态：**执行中（r9）**。','状态：**'+status+'（r9；独立审查未进行）**。',1)
        text+='\n\n## 49. r9执行交接\n\n'+(OUT/'final_review/delivery.md').read_text()+f'\n唯一最终CPU检查exit {rc}；预算'+json.dumps(budget,ensure_ascii=False)+'。\n'
        p.write_text(text)
    except BaseException as error:
        ledger();write(OUT/'failure.json',dict(phase=phase,error=repr(error),traceback=traceback.format_exc(),consumption=budget,time=now()))
        p=ROOT/'CURRENT_PLAN.md';text=p.read_text().replace('状态：**执行中（r9）**。','状态：**需返修（r9）**。',1)
        p.write_text(text+'\n\nr9停止：'+repr(error)+'；现场'+str(OUT/'failure.json')+'；不自动重试或扩大预算。\n')
        raise


def paper_values():
    result={}
    for line in (ROOT/'SOTA_RESULT.md').read_text().splitlines():
        cols=[v.strip() for v in line.split('|')]
        if len(cols)==5 and cols[1].lower() in TASKS and cols[3].startswith('**'):
            result[cols[1].lower()]=dict(model=cols[2],r2=float(cols[3].strip('*').split('±')[0]))
    if set(result)!=set(TASKS):raise ValueError('missing paper reference')
    return result


def report(rows,budget):
    review=new_directory(OUT/'final_review');paper=paper_values();tasks={}
    for task in TASKS:
        selected=[r for r in rows if r['task']==task]
        stats={label:{metric:dict(mean=float(np.mean([r[label][metric] for r in selected])),
            std=float(np.std([r[label][metric] for r in selected],ddof=0))) for metric in ('r2','rmse')} for label in ('reference','branch','fused')}
        deltas=[r['fused']['r2']-r['reference']['r2'] for r in selected]
        tasks[task]=dict(stats=stats,delta_r2=stats['fused']['r2']['mean']-stats['reference']['r2']['mean'],
            delta_rmse=stats['fused']['rmse']['mean']-stats['reference']['rmse']['mean'],
            fused_minus_paper=stats['fused']['r2']['mean']-paper[task]['r2'],reference_minus_paper=stats['reference']['r2']['mean']-paper[task]['r2'],
            improved_folds=sum(v>0 for v in deltas),mean_delta_excluding_best_fold=float((sum(deltas)-max(deltas))/4),folds=selected)
    macro={label:float(np.mean([tasks[t]['stats'][label]['r2']['mean'] for t in TASKS])) for label in ('reference','branch','fused')}
    delta=macro['fused']-macro['reference'];reached={label:sum(tasks[t]['stats'][label]['r2']['mean']>=paper[t]['r2'] for t in TASKS) for label in ('reference','fused')}
    summary=dict(policy=POLICY,tasks=tasks,macro_r2=macro,delta_macro_r2=delta,milestone=delta>=.010,
        paper=paper,reached=reached,consumption_at_science_end=budget,folds=rows,
        aggregation='five fold arithmetic mean/std ddof=0; seven equal-weight tasks; reference seed42+fold',
        protocol='shared TEST development exposed; same policy reused units predeclared; paper equivalence unverified',independent_review='NOT_PERFORMED')
    write(review/'seven_task_results.json',summary)
    lines=['状态：待审查；独立审查未进行。七任务统一G05_MKRR50，全部启用同一策略，无task-specific开关。',
        f"Macro R²：G05 {macro['reference']:.3f} → 统一融合 {macro['fused']:.3f}，差{delta:+.3f}；+0.010里程碑{'达到' if delta>=.010 else '未达到'}。",
        f"论文数值达标：参考{reached['reference']}/7，统一融合{reached['fused']}/7。",'',
        '|任务|G05 R² mean±std|统一融合R² mean±std|ΔR²|G05 RMSE mean±std|融合RMSE mean±std|ΔRMSE|改善折|来源|',
        '|---|---:|---:|---:|---:|---:|---:|---:|---|']
    for task in TASKS:
        t=tasks[task];a=t['stats']['reference'];b=t['stats']['fused'];source='复用同政策5折' if all(r['reused'] for r in t['folds']) else '新增5折'
        lines.append(f"|{task}|{a['r2']['mean']:.3f}±{a['r2']['std']:.3f}|{b['r2']['mean']:.3f}±{b['r2']['std']:.3f}|{t['delta_r2']:+.3f}|{a['rmse']['mean']:.3f}±{a['rmse']['std']:.3f}|{b['rmse']['mean']:.3f}±{b['rmse']['std']:.3f}|{t['delta_rmse']:+.3f}|{t['improved_folds']}/5|{source}|")
    lines+=['','|任务|论文最佳模型|论文R²|参考−论文|融合−论文|','|---|---|---:|---:|---:|']
    for task in TASKS:
        t=tasks[task];p=paper[task];lines.append(f"|{task}|{p['model']}|{p['r2']:.3f}|{t['reference_minus_paper']:+.3f}|{t['fused_minus_paper']:+.3f}|")
    lines+=['','五折算术均值及std ddof=0，先原精度计算再展示3位小数；RMSE负差表示改善。G05神经训练原有配方未改；统一的是微调后结构分支及最终预测。',
        '新增75fit、25正式单元／50组预测；预先指定的Ei／Xc同一政策10单元／20组预测只读复用。完整策略共35单元，无按TEST择优混搭，无新的七任务神经FT。',
        '共享validation／TEST已开发暴露；论文数据／split／处理及统计协议等价性未核实，只作探索性数值对照。',
        '','|任务|fold|α|ΔR²|ΔRMSE|validation融合R²|TEST−validation R²|TEST−validation RMSE|','|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        v=r['validation']['fused'];lines.append(f"|{r['task']}|{r['fold']}|{r['alpha']:g}|{r['fused']['r2']-r['reference']['r2']:+.3f}|{r['fused']['rmse']-r['reference']['rmse']:+.3f}|{v['r2']:.3f}|{r['fused']['r2']-v['r2']:+.3f}|{r['fused']['rmse']-v['rmse']:+.3f}|")
    (review/'delivery.md').write_text('\n'.join(lines)+'\n')


def launch(name,argv):
    require_tmux();folder=new_directory(LOG/name)
    command=dict(argv=argv,cwd=str(ROOT),environment=ENV,window='fce_all_kernel_'+name);write(folder/'command.json',command)
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
if p.name=='controller':subprocess.run([sys.executable,'scripts/run_fce_all_kernel.py','--launch-finish'],cwd=c['cwd'],env=dict(os.environ,**c['environment']),check=False)
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
    if OUT.exists() or LOG.exists():raise FileExistsError('r9 roots already exist')
    new_directory(OUT);new_directory(LOG);folder=new_directory(LOG/'setup');started=now();rc=1
    write(folder/'command.json',dict(argv=sys.argv,cwd=str(ROOT),environment=ENV,window='fce_all_kernel_setup'))
    (folder/'stdout.log').touch(exist_ok=False);(folder/'stderr.log').touch(exist_ok=False)
    try:
        for name in ('CURRENT_PLAN.md','src/training/fce_param.py'):
            target=OUT/'source_before'/name;target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(Path('/tmp/fce_all_kernel_before_20261016')/name,target)
        for name in (*EDITED,'src/training/xc_kernel.py'):
            target=OUT/'source_used'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,target)
        launch('controller',[PYTHON,'scripts/run_fce_all_kernel.py','--run']);rc=0
    except BaseException:
        with (folder/'stderr.log').open('a') as stream:traceback.print_exc(file=stream)
        raise
    finally:
        record=dict(start=started,end=now(),exit=rc,status='EXITED')
        write(folder/'execution.json',record);(folder/'exit.txt').write_text(str(rc)+'\n')


def finish():
    require_tmux();require_environment();review=OUT/'final_review';review.mkdir(exist_ok=True);records=[]
    for name in ('setup','controller','final_cpu'):
        folder=LOG/name
        if (folder/'execution.json').exists():
            row=read(folder/'execution.json');row.update(stage=name,log_dir=str(folder),exit_txt=int((folder/'exit.txt').read_text()));records.append(row)
    write(review/'execution_exits.json',dict(records=records,own_finish_exit=str(LOG/'finish/exit.txt')))
    p=ROOT/'CURRENT_PLAN.md';text=p.read_text()
    if (OUT/'completion.json').exists():
        summary=read(review/'seven_task_results.json');status=read(OUT/'completion.json')['status']
        first='**本轮七任务共用同一微调后拟合、validation选参和正式预测逻辑，固定0.5 G05＋0.5 KRR，禁止按任务启停或按TEST拼接赢家。**'
        start=text.index(first);end=text.index('\n\n',start)
        text=text[:start]+f"**r9七任务统一比较已完成。** Macro R² {summary['macro_r2']['reference']:.3f} → {summary['macro_r2']['fused']:.3f}，差{summary['delta_macro_r2']:+.3f}；全任务同一政策，75新增fit、25新正式／10复用。最终CPU真实exit {read(OUT/'completion.json')['test_exit']}，状态{status}；不自动追加任务或候选，完整结果见第49节。"+text[end:]
        text=text.replace('## 48. r9执行提示词','## 48. r9执行提示词（历史记录，已消费预算不再执行）',1)
        p.write_text(text)
    changes=[]
    for name in EDITED:
        before=OUT/'source_before'/name;previous=before.read_text().splitlines(True) if before.exists() else []
        changes.extend(difflib.unified_diff(previous,(ROOT/name).read_text().splitlines(True),fromfile='before/'+name,tofile='after/'+name))
    (review/'actual_changes.diff').write_text(''.join(changes))
    checked=subprocess.run(['git','diff','--check','--',*EDITED],cwd=ROOT,capture_output=True,text=True)
    write(review/'diff_format_check.json',dict(exit=checked.returncode,stdout=checked.stdout,stderr=checked.stderr))
    print(json.dumps(dict(exits=[(r['stage'],r['exit_txt']) for r in records],completion=read(OUT/'completion.json') if (OUT/'completion.json').exists() else None),ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--setup',action='store_true');p.add_argument('--run',action='store_true');p.add_argument('--finish',action='store_true');p.add_argument('--launch-finish',action='store_true');a=p.parse_args()
    if a.setup:setup()
    elif a.run:run()
    elif a.finish:finish()
    elif a.launch_finish:launch('finish',[PYTHON,'scripts/run_fce_all_kernel.py','--finish'])
    else:p.error('explicit execution role required')
