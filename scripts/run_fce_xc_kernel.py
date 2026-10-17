"""r7: one CPU process, 15 fixed fits, global freeze, five formal units."""
import argparse
import contextlib
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
import sklearn
import scipy
from threadpoolctl import threadpool_info
from src.training.xc_kernel import (ALPHAS,CANDIDATE,REFERENCE_ARM,FEATURE_SPEC,LIMITS,
    generator,fingerprint,bit_array,restore_bits,bound_kernel,validate_kernel,scaler_values,
    standardize,fuse,metrics,choose_alpha,fit_once,predict_dual,align_reference,PhaseGate,
    read_predictions,check_budget,validate_freeze,validate_reference_protocol,finite)
OUT=ROOT/'results/fce_tune_20261016/xc_kernel'
LOG=ROOT/'logs/fce_tune_20261016/xc_kernel'
REFERENCE=ROOT/'results/fce_tune_20261015/t3b'
SOURCE=ROOT/'data/processed/mcl_ph_r4_20260930/downstream_direct_01.staging'
SPLIT=ROOT/'data/splits/mips_paper5_outer/xc.json'
ENV=dict(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
    NUMEXPR_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1',BLIS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1')
PYTHON='/opt/conda/bin/python'


def now():return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
def read(p):return json.loads(Path(p).read_text())
def write(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');os.replace(tmp,p)
def require_tmux():
    if not os.environ.get('TMUX') or subprocess.check_output(['tmux','display-message','-p','#S'],text=True).strip()!='Uni-Poly':raise RuntimeError('requires tmux Uni-Poly')
def require_environment():
    if any(os.environ.get(k)!=v for k,v in ENV.items()):raise ValueError('CPU-only single-thread environment mismatch')


def load_split():
    manifest=read(SPLIT)
    if manifest.get('task')!='xc' or manifest.get('protocol')!='paper5_outer' or manifest.get('sample_count')!=432 or len(manifest['folds'])!=5:raise ValueError('Xc split schema mismatch')
    for fold,entry in enumerate(manifest['folds']):
        if entry['fold']!=fold or len(entry['train_indices'])!=276:raise ValueError('Xc fold/train budget mismatch')
        ids=[i for role in ('train','validation','test') for i in entry[role+'_indices']]
        if len(ids)!=432 or set(ids)!=set(range(432)):raise ValueError('Xc split overlap/missing rows')
    return manifest


def source_rows(indices,manifest,fold,role,gate=None):
    if role=='test':
        if gate is None:raise ValueError('test source read before freeze')
        gate.require_test()
    entry=manifest['folds'][fold]
    if list(indices)!=entry[role+'_indices']:raise ValueError('source request not bound to own fold role')
    offsets=np.load(SOURCE/'xc_offsets.npy',allow_pickle=False)
    if offsets.dtype!=np.int64 or offsets.shape!=(432,):raise ValueError('source offsets schema mismatch')
    rows=[]
    with (SOURCE/'xc.jsonl').open('rb') as stream:
        for idx in indices:
            stream.seek(int(offsets[idx]));row=json.loads(stream.readline())
            if row['task']!='xc' or row['original_row']!=idx or not row.get('normalized_smiles') or len(bytes.fromhex(row['sample_key']))!=32:raise ValueError('source identity/schema mismatch')
            if row['fold_membership'][fold]!={'fold':fold,'role':role}:raise ValueError('source split role mismatch')
            if not np.isfinite(row['label']):raise FloatingPointError('nonfinite source target')
            rows.append(row)
    return rows


def reference_binding(fold,entry):
    frozen=read(REFERENCE/'selection_before_test.json')
    units=[u for u in frozen['units'] if (u['arm'],u['task'],u['fold'])==(REFERENCE_ARM,'xc',fold)]
    if len(units)!=1:raise ValueError('missing fixed G05 reference binding')
    unit=units[0];validate_reference_protocol(unit);cfg=unit['config'];pt=cfg['training_config'];recipe=cfg['recipe']
    expected_parent=str(REFERENCE/'pretrain/T3B_G05/deploy_05000.pt')
    expected_head=str(ROOT/'results/glt_fusion_20261007/r1/finetune/B0/xc'/f'fold{fold}'/'head_initial.pt')
    if unit['parent']!=expected_parent or cfg['parent']!=expected_parent or cfg['arm']!=REFERENCE_ARM or cfg['task']!='xc' or unit['seed']!=42+fold or unit['head_initial']!=expected_head:
        raise ValueError('reference parent/arm/head/seed mismatch')
    if pt.get('arm')!=REFERENCE_ARM or pt.get('revision')!='r5' or pt.get('loss_weights')!=[1,.5,0] or pt.get('schedule_total_steps')!=20000 or recipe.get('arm')!='R0':raise ValueError('reference G05/R0 protocol mismatch')
    for k in ('train_indices','validation_indices','test_indices'):
        expected=sorted(entry[k]) if k=='test_indices' else entry[k]
        if unit[k]!=expected:raise ValueError('reference split mismatch '+k)
    folder=REFERENCE/'finetune/T3B_G05/xc'/f'fold{fold}'
    scaler=read(folder/'scaler.json');m=read(folder/'metrics.json')
    if scaler!=unit['scaler'] or m['status']!='PASS' or m['config']!=cfg or m['best_epoch']!=unit['best_epoch'] or m['outer_test']!='NOT_RUN':raise ValueError('reference validation/scaler metadata mismatch')
    scaler_values(scaler)
    return unit,scaler


def feature_rows(rows,features,gen):
    metadata=[]
    for row in rows:
        idx=row['original_row']
        if idx not in features:
            fp,details=fingerprint(row['normalized_smiles'],gen);features[idx]=fp
        else:
            details=dict(on_bits=features[idx].GetNumOnBits())
        metadata.append(dict(original_row=idx,sample_key=row['sample_key'],normalized_smiles=row['normalized_smiles'],**details))
    return metadata


def save_predictions(path,rows,truth,g05,krr,scaler):
    path.parent.mkdir(parents=True,exist_ok=True)
    np.savez(path,indices=np.asarray([r['original_row'] for r in rows],dtype=np.int64),
        sample_keys=np.asarray([r['sample_key'] for r in rows]),y_true=truth,y_g05=g05,y_krr=krr,
        y_fused=fuse(g05,krr),scaler_mean=scaler['mean'],scaler_scale=scaler['scale'])


def run_science():
    require_tmux();require_environment();started=now();budget={k:0 for k in LIMITS};phase='initialization'
    def ledger():check_budget(budget);write(OUT/'consumption.json',budget)
    try:
        manifest=load_split();gen=generator();features={};units=[];gate=PhaseGate()
        versions=dict(rdkit=rdkit.__version__,sklearn=sklearn.__version__,numpy=np.__version__,scipy=scipy.__version__,threadpools=threadpool_info())
        write(OUT/'feature_spec.json',dict(spec=FEATURE_SPEC,generator_info=gen.GetInfoString(),versions=versions,environment=ENV))
        phase='fit_validation'
        for fold,entry in enumerate(manifest['folds']):
            folder=OUT/'folds'/f'fold{fold}';folder.mkdir(parents=True,exist_ok=False)
            binding,scaler=reference_binding(fold,entry)
            train=source_rows(entry['train_indices'],manifest,fold,'train');val=source_rows(entry['validation_indices'],manifest,fold,'validation')
            mu,sigma=scaler_values(scaler);train_y=finite([r['label'] for r in train])
            if not np.isclose(train_y.mean(),mu,rtol=1e-12,atol=1e-12) or not np.isclose(train_y.std(),sigma,rtol=1e-12,atol=1e-12):raise ValueError('source train targets/scaler mismatch')
            train_meta=feature_rows(train,features,gen);val_meta=feature_rows(val,features,gen)
            write(folder/'inputs.json',dict(fold=fold,train=train_meta,validation=val_meta,scaler=scaler,reference=binding,feature_spec=FEATURE_SPEC))
            pred=read_predictions(REFERENCE/'finetune/T3B_G05/xc'/f'fold{fold}'/'validation_predictions.npz')
            truth,g05=align_reference(pred,val,scaler)
            k_train=bound_kernel(features,entry['train_indices'],entry['train_indices'])
            k_val=bound_kernel(features,entry['validation_indices'],entry['train_indices'])
            z=standardize(train_y,scaler);trials=[];coefficients={}
            for alpha in ALPHAS:
                budget['fit_attempts']+=1;ledger()
                coef=fit_once(k_train,z,alpha);budget['fit_completed']+=1;ledger()
                coefficients[alpha]=coef
                krr=predict_dual(k_val,coef,scaler);fusion=fuse(g05,krr)
                trial=dict(alpha=alpha,branch=metrics(truth,krr),fused=metrics(truth,fusion),reference=metrics(truth,g05),fit_rows=276,fit_completed=True)
                path=folder/'validation'/f'alpha_{alpha:g}';save_predictions(path/'predictions.npz',val,truth,g05,krr,scaler);write(path/'metrics.json',trial);trials.append(trial)
            chosen=choose_alpha(trials);alpha=chosen['alpha'];model=folder/'selected_model.npz'
            np.savez(model,alpha=np.float64(alpha),dual_coef=coefficients[alpha],train_bits=bit_array([features[i] for i in entry['train_indices']]),
                train_indices=np.asarray(entry['train_indices'],dtype=np.int64),train_keys=np.asarray([r['sample_key'] for r in train]),
                train_normalized_smiles=np.asarray([r['normalized_smiles'] for r in train]),train_y=train_y,mean=np.float64(mu),scale=np.float64(sigma),feature_spec=json.dumps(FEATURE_SPEC,sort_keys=True))
            unit=dict(fold=fold,alpha=alpha,fit_count=3,model=str(model),train_indices=entry['train_indices'],
                validation_indices=entry['validation_indices'],test_indices=entry['test_indices'],train_samples=train_meta,
                scaler=scaler,reference=binding,validation_trials=trials,chosen_validation=chosen)
            write(folder/'selection.json',unit);units.append(unit)
        payload=dict(candidate=CANDIDATE,feature_spec=FEATURE_SPEC,units=units,frozen=True,frozen_at=now(),
            reference_root=str(REFERENCE),reference_test_development_exposed=True,versions=versions,
            split=manifest,source=str(SOURCE/'xc.jsonl'),selection_rule='own fold fused validation raw RMSE minimum; exact ties larger alpha',
            test_reference_opened=False,budget_at_freeze=dict(budget))
        validate_freeze(payload)
        if budget['fit_completed']!=15 or budget['formal_starts']!=0:raise ValueError('incomplete pre-TEST budget')
        write(OUT/'selection_before_test.json',payload);gate.freeze(read(OUT/'selection_before_test.json'))
        phase='formal';formal_rows=[]
        for u in gate.payload['units']:
            fold=u['fold'];entry=manifest['folds'][fold];scaler=u['scaler'];gate.require_test()
            budget['formal_starts']+=1;ledger()
            rows=source_rows(u['test_indices'],manifest,fold,'test',gate)
            with np.load(u['model'],allow_pickle=False) as saved:
                model={k:saved[k].copy() for k in saved.files}
            if float(model['alpha'])!=u['alpha'] or model['train_indices'].tolist()!=u['train_indices'] or model['train_keys'].tolist()!=[r['sample_key'] for r in u['train_samples']] or json.loads(str(model['feature_spec']))!=FEATURE_SPEC:
                raise ValueError('selected model/frozen train identity mismatch')
            if [float(model['mean'])]!=scaler['mean'] or [float(model['scale'])]!=scaler['scale']:raise ValueError('selected model scaler mismatch')
            frozen_features={idx:fp for idx,fp in zip(u['train_indices'],restore_bits(model['train_bits']))}
            feature_rows(rows,frozen_features,gen)
            k_test=bound_kernel(frozen_features,u['test_indices'],u['train_indices'])
            reference_path=REFERENCE/'formal/T3B_G05/xc'/f'fold{fold}'/'test_predictions.npz'
            pred=read_predictions(reference_path,gate=gate,test=True);truth,g05=align_reference(pred,rows,scaler)
            krr=predict_dual(k_test,model['dual_coef'],scaler);fusion=fuse(g05,krr)
            folder=OUT/'formal'/f'fold{fold}';folder.mkdir(parents=True,exist_ok=False)
            for label,value in (('KRR_SELECTED',krr),(CANDIDATE,fusion)):
                np.savez(folder/(label+'_predictions.npz'),indices=np.asarray(u['test_indices']),sample_keys=np.asarray([r['sample_key'] for r in rows]),
                    y_true=truth,y_pred=value,y_g05=g05,alpha=np.float64(u['alpha']),scaler_mean=scaler['mean'],scaler_scale=scaler['scale'])
                budget['prediction_groups']+=1;ledger()
            result=dict(fold=fold,alpha=u['alpha'],reference=metrics(truth,g05),branch=metrics(truth,krr),fused=metrics(truth,fusion),
                validation=u['chosen_validation'],krr_test_predict_calls=1,reference_path=str(reference_path),reference_reused=True)
            write(folder/'metrics.json',result);formal_rows.append(result);budget['formal_completed']+=1;ledger()
        phase='report';report(formal_rows,versions,budget)
        if (budget['fit_completed'],budget['formal_completed'],budget['prediction_groups'])!=(15,5,10):raise ValueError('incomplete scientific matrix')
        budget['final_cpu_starts']+=1;ledger()
        test=launch('final_cpu',['timeout','300s',PYTHON,'-m','pytest','-q','tests/test_fce_xc_kernel.py'])
        test_rc=wait_exit(test);state='待审查' if test_rc==0 else '需返修'
        write(OUT/'completion.json',dict(status=state,start=started,end=now(),test_exit=test_rc,consumption=budget,independent_review='NOT_PERFORMED'))
        p=ROOT/'CURRENT_PLAN.md';text=p.read_text().replace('状态：**执行中（r7）**。授权来源：','状态：**'+state+'**。授权来源：',1)
        text+='\n\n## 35. r7执行交接\n\n状态：**'+state+'**；唯一CPU检查真实exit '+str(test_rc)+'；同一主体自检，独立审查未进行。\n\n'+(OUT/'final_review/delivery.md').read_text();p.write_text(text)
        write(OUT/'status.json',dict(status=state,phase='finished',consumption=budget))
    except BaseException as error:
        ledger();write(OUT/'failure.json',dict(status='阻断',phase=phase,error=repr(error),traceback=traceback.format_exc(),consumption=budget,time=now()))
        p=ROOT/'CURRENT_PLAN.md';text=p.read_text().replace('状态：**执行中（r7）**。授权来源：','状态：**阻断**。授权来源：',1)
        p.write_text(text+'\n\nr7执行阻断：'+repr(error)+'；保留results/fce_tune_20261016/xc_kernel/failure.json和日志；不自动重试，不扩大预算。\n')
        raise


def report(rows,versions,budget):
    import csv
    review=OUT/'final_review';review.mkdir(exist_ok=False)
    stats={}
    for label in ('reference','branch','fused'):
        stats[label]={metric:dict(mean=float(np.mean([r[label][metric] for r in rows])),std=float(np.std([r[label][metric] for r in rows],ddof=0))) for metric in ('r2','rmse')}
    delta_r2=stats['fused']['r2']['mean']-stats['reference']['r2']['mean'];delta_rmse=stats['fused']['rmse']['mean']-stats['reference']['rmse']['mean']
    fold_rows=[]
    for r in rows:
        fold_rows.append(dict(fold=r['fold'],alpha=r['alpha'],reference_r2=r['reference']['r2'],branch_r2=r['branch']['r2'],fused_r2=r['fused']['r2'],
            delta_r2=r['fused']['r2']-r['reference']['r2'],reference_rmse=r['reference']['rmse'],branch_rmse=r['branch']['rmse'],fused_rmse=r['fused']['rmse'],
            delta_rmse=r['fused']['rmse']-r['reference']['rmse'],validation_fused_r2=r['validation']['fused']['r2'],validation_fused_rmse=r['validation']['fused']['rmse'],
            test_minus_validation_r2=r['fused']['r2']-r['validation']['fused']['r2'],test_minus_validation_rmse=r['fused']['rmse']-r['validation']['fused']['rmse']))
    gains=[r['delta_r2'] for r in fold_rows];best=max(gains)
    concentration=dict(r2_improved_folds=sum(x>0 for x in gains),rmse_improved_folds=sum(r['delta_rmse']<0 for r in fold_rows),
        best_fold=max(fold_rows,key=lambda r:r['delta_r2'])['fold'],mean_delta_excluding_best_fold=float((sum(gains)-best)/4))
    summary=dict(task='Xc',candidate=CANDIDATE,stats=stats,folds=fold_rows,delta_r2=delta_r2,delta_rmse=delta_rmse,
        fused_minus_paper=stats['fused']['r2']['mean']-.423,reference_minus_paper=stats['reference']['r2']['mean']-.423,
        branch_minus_paper=stats['branch']['r2']['mean']-.423,paper_r2=.423,paper_model='Periodic-TDL',
        paper_numeric_target_reached=stats['fused']['r2']['mean']>=.423,milestone=delta_r2>=.030,
        benefit_concentration=concentration,versions=versions,consumption_at_science_end=dict(budget),
        aggregation='five fold arithmetic means and std ddof=0; reference seed42+fold; no new seven-task macro',
        independent_review='NOT_PERFORMED',protocol='shared TEST development exposed; paper protocol equivalence unverified')
    write(review/'xc_results.json',summary)
    with (review/'fold_details.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(fold_rows[0]));w.writeheader();w.writerows(fold_rows)
    lines=['状态：待审查；独立审查未进行。唯一主要候选为XC_G05_MKRR50，固定原单位0.5 G05＋0.5所选KRR。',
        '仅Xc五折；均值为五折算术均值，std ddof=0。差值先用原精度计算，再展示三位小数。',
        '', '|方案|R² mean±std|RMSE mean±std|R²−论文0.423|','|---|---:|---:|---:|']
    for key,name in [('reference','G05_R0（复用）'),('branch','KRR_SELECTED（按融合validation选α后的诊断）'),('fused',CANDIDATE)]:
        s=stats[key];lines.append(f"|{name}|{s['r2']['mean']:.3f}±{s['r2']['std']:.3f}|{s['rmse']['mean']:.3f}±{s['rmse']['std']:.3f}|{s['r2']['mean']-.423:+.3f}|")
    lines+=['',f"融合−G05：R² {delta_r2:+.3f}，RMSE {delta_rmse:+.3f}。R²改善{concentration['r2_improved_folds']}/5折，RMSE改善{concentration['rmse_improved_folds']}/5折。",
        f"相对G05 +0.030进展里程碑：{'达到' if delta_r2>=.030 else '未达到'}；Xc论文数值目标：{'达到' if summary['paper_numeric_target_reached'] else '未达到'}。",
        f"最大增益fold{concentration['best_fold']}；去掉该fold后剩余四折平均R²差{concentration['mean_delta_excluding_best_fold']:+.3f}。",
        '', '|fold|α|G05 R²|KRR R²|融合R²|ΔR²|G05 RMSE|KRR RMSE|融合RMSE|ΔRMSE|validation融合R²|TEST−validation R²|TEST−validation RMSE|',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in fold_rows:
        lines.append(f"|{r['fold']}|{r['alpha']:g}|{r['reference_r2']:.3f}|{r['branch_r2']:.3f}|{r['fused_r2']:.3f}|{r['delta_r2']:+.3f}|{r['reference_rmse']:.3f}|{r['branch_rmse']:.3f}|{r['fused_rmse']:.3f}|{r['delta_rmse']:+.3f}|{r['validation_fused_r2']:.3f}|{r['test_minus_validation_r2']:+.3f}|{r['test_minus_validation_rmse']:+.3f}|")
    lines+=['', '|任务|本轮固定融合R²／RMSE|论文最佳模型|论文R²|本轮−论文|当前状态|','|---|---|---|---:|---|---|']
    for task,paper in [('Eea',.934),('Egb',.923),('Egc',.911),('Ei',.839),('EPS',.787),('Nc',.856)]:
        model='Periodic-TDL (+Chem, +RC)' if task in ('Egb','Egc') else 'Periodic-TDL'
        lines.append(f'|{task}|—|{model}|{paper:.3f}|不可比较|本轮未运行／不可比较，预算0|')
    lines.append(f"|Xc|{stats['fused']['r2']['mean']:.3f}±{stats['fused']['r2']['std']:.3f}／{stats['fused']['rmse']['mean']:.3f}±{stats['fused']['rmse']['std']:.3f}|Periodic-TDL|0.423|{summary['fused_minus_paper']:+.3f}|五折完整正式结果，待审查|")
    lines+=['', '本轮没有新的七任务macro。新增15次CPU fit，无refit，五fold各一次选中分支TEST预测并保存分支／融合两组输出，共10组；旧G05五组TEST仅复用。',
        'KRR分支只作诊断，不是对其独立选择α的baseline；不按TEST选择分支或融合赢家。',
        '结构特征直接来自原normalized_smiles，保留连接点；不补造立构或晶体信息。共享validation／TEST已参与开发，不是独立盲测；项目Xc432与论文历史记录430、split／协议等价性未核实，仅探索性数值对照。',
        '结果results/fce_tune_20261016/xc_kernel/；日志logs/fce_tune_20261016/xc_kernel/；冻结selection_before_test.json；正式预测formal/foldN/；tmux Uni-Poly:fce_xc_kernel_*。']
    (review/'delivery.md').write_text('\n'.join(lines)+'\n')


def launch(name, argv):
    require_tmux();folder=LOG/name;folder.mkdir(parents=True,exist_ok=False)
    command=dict(argv=argv,cwd=str(ROOT),environment=ENV,window='fce_xc_kernel_'+name)
    write(folder/'command.json',command)
    wrapper='''import json,os,subprocess,sys,datetime,traceback
from pathlib import Path
p=Path(sys.argv[1]); c=json.loads((p/'command.json').read_text())
def stamp():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def save(v):
 t=p/'execution.tmp';t.write_text(json.dumps(v,indent=2)+'\\n');os.replace(t,p/'execution.json')
if (p/'execution.json').exists() or (p/'exit.txt').exists():raise RuntimeError('execution replay forbidden')
v=dict(command=c,start=stamp(),end=None,exit=None,status='RUNNING');save(v);rc=1
try:
 with (p/'stdout.log').open('x') as out,(p/'stderr.log').open('x') as err:
  child=subprocess.Popen(c['argv'],cwd=c['cwd'],env=dict(os.environ,**c['environment']),stdout=out,stderr=err)
  v['pid']=child.pid;save(v);rc=child.wait()
except BaseException:
 v['error']=traceback.format_exc()
finally:
 v.update(end=stamp(),exit=rc,status='EXITED');save(v);(p/'exit.txt').write_text(str(rc)+'\\n')
if p.name=='controller':
 subprocess.run([sys.executable,'scripts/run_fce_xc_kernel.py','--launch-finish'],cwd=c['cwd'],env=dict(os.environ,**c['environment']),check=False)
sys.exit(rc)
'''
    cmd=shlex.join([PYTHON,'-c',wrapper,str(folder)])
    (folder/'command.sh').write_text('#!/bin/bash\ncd '+shlex.quote(str(ROOT))+'\n'+cmd+'\n')
    subprocess.run(['tmux','new-window','-d','-t','Uni-Poly','-n',command['window'],'tmux set-option -w remain-on-exit on\nexec bash '+shlex.quote(str(folder/'command.sh'))],check=True)
    return folder


def wait_exit(folder):
    # Completion notification, not a polling monitor.
    import ctypes
    folder=Path(folder);libc=ctypes.CDLL(None,use_errno=True)
    fd=libc.inotify_init1(os.O_CLOEXEC)
    if fd<0:raise OSError(ctypes.get_errno(),'inotify_init1')
    try:
        if libc.inotify_add_watch(fd,os.fsencode(folder),0x8|0x80)<0:raise OSError(ctypes.get_errno(),'inotify_add_watch')
        while not (folder/'exit.txt').exists():os.read(fd,4096)
        return int((folder/'exit.txt').read_text())
    finally:os.close(fd)


def setup():
    require_tmux();require_environment()
    if OUT.exists() or LOG.exists():raise FileExistsError('new r7 roots must not already exist')
    LOG.mkdir(parents=True);OUT.mkdir(parents=True)
    before=OUT/'source_before';before.mkdir()
    shutil.copyfile('/tmp/fce_xc_kernel_before_20261016/CURRENT_PLAN.md',before/'CURRENT_PLAN.md')
    used=OUT/'source_used';used.mkdir()
    for name in ('src/training/xc_kernel.py','scripts/run_fce_xc_kernel.py','tests/test_fce_xc_kernel.py'):
        dest=used/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,dest)
    write(LOG/'setup.json',dict(start=now(),cwd=str(ROOT),environment=ENV,command=sys.argv,tmux='Uni-Poly:fce_xc_kernel_setup'))
    launch('controller',[PYTHON,'scripts/run_fce_xc_kernel.py','--run'])
    write(LOG/'setup_exit.json',dict(exit=0,end=now(),meaning='setup completed; scientific controller has separate real exit'))


def finish():
    require_tmux();require_environment();review=OUT/'final_review';review.mkdir(exist_ok=True)
    index=[]
    for folder in sorted(LOG.iterdir()):
        if not folder.is_dir() or folder.name=='finish_delivery':continue
        record=read(folder/'execution.json') if (folder/'execution.json').exists() else dict(status='未核实',exit=None)
        record['log_dir']=str(folder);record['exit_txt']=int((folder/'exit.txt').read_text()) if (folder/'exit.txt').exists() else None
        index.append(record)
    write(review/'exit_index.json',dict(records=index,setup=read(LOG/'setup_exit.json'),
        own_finish_exit=str(LOG/'finish_delivery/exit.txt'),time=now()))
    plan=ROOT/'CURRENT_PLAN.md';text=plan.read_text()
    if (review/'xc_results.json').exists():
        summary=read(review/'xc_results.json');s=summary['stats']['fused']
        row=f"|Xc|{s['r2']['mean']:.3f}±{s['r2']['std']:.3f}／{s['rmse']['mean']:.3f}±{s['rmse']['std']:.3f}|Periodic-TDL|0.423|{summary['fused_minus_paper']:+.3f}|五折正式完成，待审查|"
        text=text.replace('|Xc|未完成|Periodic-TDL|0.423|不可比较|待执行；参考G05 R²0.371±0.157|',row)
    text+='\n\nr7实际预算：'+json.dumps(read(OUT/'consumption.json'),ensure_ascii=False)+'。\n'
    if (OUT/'selection_before_test.json').exists():
        frozen=read(OUT/'selection_before_test.json')
        text+='冻结时间：'+frozen['frozen_at']+'；五折所选α：'+str([u['alpha'] for u in frozen['units']])+'。全部15fit完成且正式单元启动0时冻结。\n'
    text+='真实退出：'+str([(Path(r['log_dir']).name,r['exit_txt']) for r in index])+'；setup exit0；收尾自身exit见logs/fce_tune_20261016/xc_kernel/finish_delivery/exit.txt（本索引写入时收尾尚未退出，不伪造exit）。完整退出索引final_review/exit_index.json；一次性CPU检查日志final_cpu/stdout.log及stderr.log；本轮增量diff final_review/xc_kernel_only_changes.diff。\n'
    plan.write_text(text)
    changes=[]
    paths=['CURRENT_PLAN.md','src/training/xc_kernel.py','scripts/run_fce_xc_kernel.py','tests/test_fce_xc_kernel.py']
    for name in paths:
        previous=(OUT/'source_before/CURRENT_PLAN.md').read_text().splitlines(True) if name=='CURRENT_PLAN.md' else []
        changes.extend(difflib.unified_diff(previous,(ROOT/name).read_text().splitlines(True),fromfile='before/'+name,tofile='after/'+name))
    (review/'xc_kernel_only_changes.diff').write_text(''.join(changes))
    full=subprocess.run(['git','diff','--','CURRENT_PLAN.md'],cwd=ROOT,capture_output=True,text=True,check=True).stdout
    (review/'tracked_plan_diff_including_preexisting_changes.diff').write_text(full)
    checked=subprocess.run(['git','diff','--check','--','CURRENT_PLAN.md'],cwd=ROOT,capture_output=True,text=True)
    write(review/'diff_format_check.json',dict(exit=checked.returncode,stdout=checked.stdout,stderr=checked.stderr))
    print(json.dumps(dict(exit_records=index,completion=read(OUT/'completion.json') if (OUT/'completion.json').exists() else None),ensure_ascii=False,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--setup',action='store_true');p.add_argument('--run',action='store_true');p.add_argument('--finish',action='store_true');p.add_argument('--launch-finish',action='store_true');a=p.parse_args()
    if a.setup:setup()
    elif a.run:run_science()
    elif a.launch_finish:launch('finish_delivery',[PYTHON,'scripts/run_fce_xc_kernel.py','--finish'])
    elif a.finish:finish()
    else:p.error('one explicit execution role required')
