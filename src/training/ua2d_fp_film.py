"""UA2D SOTA r2: frozen cached h, unchanged Morgan, paired NULL/SHIFT/FILM.

Only F0 development is executable in this authorization. No encoder forward,
new fingerprint generation, kernel fit, or TEST-label reader is provided.
"""
import copy
import datetime
import json
import math
import os
import sqlite3
from pathlib import Path
import numpy as np
import torch
from torch import nn
from src.training.xc_kernel import FEATURE_SPEC as KERNEL_SPEC, metrics

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/ua2d_sota_20261018/r2/fp_fusion'
LOG=ROOT/'logs/ua2d_sota_20261018/r2/fp_fusion'
UA=ROOT/'results/glt_optimization_20261011/r4/U1'
CACHE=ROOT/'results/ua2d_sota_20261018/r1/kernel/F0_recovery1'
SPLIT=ROOT/'data/splits/mips_paper5_outer'
DOWN=ROOT/'data/processed/mcl_ph_r4_20260930/downstream_direct_01.staging'
FP_OLD=ROOT/'results/fp_graph_fusion_20261017/f1/fingerprints.sqlite'
FP_SIDE=ROOT/'results/ua2d_sota_20261018/r1/fingerprints.sqlite'
MODES=('NULL','SHIFT','FILM')
TASKS=('eea','egb','egc','ei','eps','nc','xc')
FEATURE_SPEC={k:v for k,v in KERNEL_SPEC.items() if k not in ('kernel','returnDistance','kernel_dtype','g05_weight','krr_weight')}
LIMITS=dict(adaptation_starts=105,epochs=5250,cpu_tests=1,pt=0,krr=0,new_fp=0,ph=0,conformers=0,encoder_forward=0,formal_test=0)
RECIPE=dict(epochs=50,batch=24,drop_last=False,dtype='float32',workers=0,lr=.0003,weight_decay=.02,
    clip=5.,scheduler='CosineAnnealingLR',T_max=50,eta_min=.000003,head_dropout=.1,adapter_dropout=0.,seed='42+fold')

def now(): return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
def read(p): return json.loads(Path(p).read_text())
def write(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');os.replace(tmp,p)
def finite(x):
    x=np.asarray(x,dtype=np.float64)
    if not np.isfinite(x).all(): raise FloatingPointError('nonfinite array')
    return x
def folder(mode,task,fold): return OUT/'adaptation/F0'/mode/task/f'fold{fold}'
def base_folder(task,fold): return UA/'finetune/UA_2D'/task/f'fold{fold}'
def inverse(z,scaler): return finite(z)*scaler['scale'][0]+scaler['mean'][0]

def selected_indices(entry):
    train=entry['train_indices'];valid=entry['validation_indices']
    if not train or not valid or len(set(train+valid))!=len(train+valid): raise ValueError('train/validation overlap or empty')
    return train+valid

def validate_features(arrays,entry,keys):
    ids=selected_indices(entry)
    if set(arrays)!=set(('features','indices','sample_keys','train_count')): raise ValueError('expected raw h feature cache schema; no normalized u')
    if arrays['indices'].tolist()!=ids or int(arrays['train_count'])!=len(entry['train_indices']): raise ValueError('feature split order/count mismatch')
    if arrays['sample_keys'].astype(str).tolist()!=keys: raise ValueError('feature row/sample_key mismatch')
    h=arrays['features']
    if h.shape!=(len(ids),512) or not np.isfinite(h).all(): raise ValueError('raw h dimensions/nonfinite')
    return h.astype(np.float32)

class MorganReader:
    """Both historical databases are read-only; missing features stop the unit."""
    def __init__(self):
        inherited=read(ROOT/'results/fp_graph_fusion_20261017/f1/selection_before_test.json')['policy']['input_spec']['feature_spec']
        if inherited!=FEATURE_SPEC: raise ValueError('historical Morgan feature_spec mismatch')
        self.old=sqlite3.connect(FP_OLD.as_uri()+'?mode=ro',uri=True)
        self.side=sqlite3.connect(FP_SIDE.as_uri()+'?mode=ro',uri=True)
    def get(self,smiles):
        row=self.old.execute('SELECT bits FROM bits WHERE smiles=?',(smiles,)).fetchone()
        if row is None: row=self.side.execute('SELECT value FROM bits WHERE smiles=?',(smiles,)).fetchone()
        if row is None: raise ValueError('missing read-only Morgan feature: '+smiles)
        bits=np.frombuffer(row[0],dtype=np.uint8).copy()
        if bits.shape!=(2048,) or not np.isin(bits,[0,1]).all() or not bits.any(): raise ValueError('invalid Morgan binary vector')
        return bits.astype(np.float32)
    def close(self): self.old.close();self.side.close()

class FilmAdapter(nn.Module):
    def __init__(self,mode):
        super().__init__()
        if mode not in MODES: raise ValueError('invalid fusion mode')
        self.mode=mode
        with torch.random.fork_rng(devices=[]):
            torch.default_generator.manual_seed(20261018)
            self.input=nn.Linear(2048,32);self.output=nn.Linear(32,1024)
            nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)
    def condition(self,bits):
        if self.mode=='NULL': bits=torch.full_like(bits,.5)
        norm=bits.norm(dim=-1,keepdim=True)
        if not torch.isfinite(bits).all() or bool((norm<=0).any()): raise ValueError('invalid FP input norm/value')
        x=math.sqrt(512)*bits/norm
        values=self.output(torch.nn.functional.gelu(self.input(x)))
        gamma,beta=(.1*values.tanh()).chunk(2,dim=-1)
        return gamma,beta
    def forward(self,h,bits):
        gamma,beta=self.condition(bits)
        return h+beta if self.mode=='SHIFT' else (1+gamma)*h+beta

class CachedFusion(nn.Module):
    def __init__(self,mode,head_state):
        super().__init__();self.adapter=FilmAdapter(mode)
        with torch.random.fork_rng(devices=[]):
            self.head=nn.Sequential(nn.Linear(512,512),nn.GELU(),nn.Dropout(.1),nn.Linear(512,1))
        strict_restore(self.head,head_state)
    def forward(self,h,bits): return self.head(self.adapter(h,bits))

def strict_restore(module,state):
    current=module.state_dict()
    if set(state)!=set(current): raise ValueError('adaptation state names mismatch')
    for k,v in state.items():
        if v.shape!=current[k].shape or v.dtype!=current[k].dtype or not torch.isfinite(v).all(): raise ValueError('adaptation tensor mismatch '+k)
    module.load_state_dict(state,strict=True)

def load_unit(task,fold):
    """Normal execution loader selects train/validation before any labels are read."""
    from src.training.glt_dual_runtime import open_source
    if task not in TASKS or fold not in range(5): raise ValueError('unit outside authorized matrix')
    manifest=read(SPLIT/f'{task}.json');entry=manifest['folds'][fold];ids=selected_indices(entry)
    source,frame=open_source(DOWN,DOWN,task=task,selected_indices=sorted(ids),split_manifest=manifest,verify_key_digest=False)
    reader=None
    try:
        rows={int(r['original_row']):r for r in source.entries}
        if set(rows)!=set(ids): raise ValueError('development source selected rows mismatch')
        keys=[rows[i]['sample_key'] for i in ids]
        feature_path=CACHE/'features'/task/f'fold{fold}'/'train_validation.npz'
        with np.load(feature_path,allow_pickle=False) as p: arrays={k:p[k].copy() for k in p.files}
        h=validate_features(arrays,entry,keys)
        choice=read(CACHE/'GRAPH'/task/f'fold{fold}'/'selection.json')
        core=base_folder(task,fold)/'best.pt'
        if choice['arm']!='UA' or choice['encoder_best']!=str(core): raise ValueError('h encoder provenance mismatch')
        with np.load(CACHE/'GRAPH'/task/f'fold{fold}'/'model.npz',allow_pickle=False) as p:
            if json.loads(str(p['feature_spec']))!=KERNEL_SPEC: raise ValueError('feature cache Morgan schema mismatch')
        scaler=read(base_folder(task,fold)/'scaler.json')
        if scaler.get('fit')!='train' or scaler.get('transform')!='standard' or scaler['scale'][0]<=0: raise ValueError('scaler semantics mismatch')
        n=len(entry['train_indices']);y=finite([rows[i]['label'] for i in ids])
        if not np.isclose(y[:n].mean(),scaler['mean'][0],rtol=1e-12,atol=1e-12) or not np.isclose(y[:n].std(),scaler['scale'][0],rtol=1e-12,atol=1e-12): raise ValueError('saved scaler differs from inner-train')
        best=torch.load(core,map_location='cpu',weights_only=False)
        if (best['task'],best['fold'],best['seed'])!=(task,fold,42+fold) or best['scaler_mean']!=scaler['mean'] or best['scaler_scale']!=scaler['scale']: raise ValueError('trained core identity/scaler mismatch')
        head={k.removeprefix('head.'):v for k,v in best['state_dict'].items() if k.startswith('head.')}
        reader=MorganReader();bits=np.stack([reader.get(rows[i]['normalized_smiles']) for i in ids])
        with np.load(base_folder(task,fold)/'validation_predictions.npz',allow_pickle=False) as p:
            if p['indices'].tolist()!=entry['validation_indices'] or p['sample_keys'].astype(str).tolist()!=keys[n:]: raise ValueError('epoch0 validation identity mismatch')
            baseline_y=finite(p['y_true']).reshape(-1);baseline_pred=finite(p['y_pred']).reshape(-1)
        tolerance=8*np.finfo(np.float32).eps*(np.abs(y[n:])+abs(scaler['mean'][0])+scaler['scale'][0])
        if np.any(np.abs(baseline_y-y[n:])>tolerance): raise ValueError('baseline labels/source mismatch')
        metadata=dict(task=task,fold=fold,core=str(core),core_best_epoch=best['best_epoch'],feature_path=str(feature_path),
            train_indices=entry['train_indices'],validation_indices=entry['validation_indices'],sample_keys=keys,
            feature_kind='raw model.features norm2 output; no L2 normalization',feature_spec=FEATURE_SPEC,scaler=scaler)
        return dict(h=h,bits=bits,y=y,n=n,ids=ids,keys=keys,head=head,baseline_y=baseline_y,baseline_pred=baseline_pred,metadata=metadata)
    finally:
        if reader is not None: reader.close()
        source.close()

def save_best(path,model,cfg,best_epoch,metadata):
    from src.training.glt_dual_runtime import save_checkpoint
    save_checkpoint(path,dict(state_dict={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},config=cfg,
        metadata=metadata,best_epoch=best_epoch,no_adaptation_selected=best_epoch==0,frozen_components=['O8','readout','norm2'],
        encoder_weights_reference=metadata['core'],feature_spec=FEATURE_SPEC,outer_test='NOT_RUN'))

def run_unit(mode,task,fold):
    from src.utils import set_global_seed
    if os.environ.get('CUDA_VISIBLE_DEVICES') not in ('2','3'): raise ValueError('adaptation requires one physical GPU2/3')
    dest=folder(mode,task,fold);dest.mkdir(parents=True,exist_ok=False)
    cfg=dict(plan='UA2D-SOTA-20261018-01',revision='r2',round='F0',mode=mode,task=task,fold=fold,seed=42+fold,recipe=RECIPE)
    write(dest/'config.json',cfg);updates=dict(attempted=0,completed=0);history=[]
    write(dest/'updates.json',updates)
    try:
        data=load_unit(task,fold);n=data['n'];metadata=data['metadata'];scaler=metadata['scaler']
        write(dest/'feature_sources.json',metadata);write(dest/'scaler.json',scaler)
        model=CachedFusion(mode,data['head']).to('cuda:0');set_global_seed(42+fold)
        h=torch.from_numpy(data['h']).to('cuda:0');bits=torch.from_numpy(data['bits']).to('cuda:0')
        targets=torch.tensor((data['y']-scaler['mean'][0])/scaler['scale'][0],dtype=torch.float32,device='cuda:0')[:,None]
        batches=math.ceil(n/24);update_limit=50*batches
        write(dest/'planned_budget.json',dict(starts=1,epochs=50,updates=update_limit,train_rows=n,validation_rows=len(h)-n))
        initial=metrics(data['baseline_y'],data['baseline_pred']);best=dict(epoch=0,**initial);best_prediction=data['baseline_pred'].copy()
        save_best(dest/'best.pt',model,cfg,0,metadata)
        optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.02)
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=50,eta_min=3e-6)
        generator=torch.Generator().manual_seed(42+fold)
        # Index-only loader preserves identical batch permutations across the three modes.
        train=torch.utils.data.DataLoader(torch.arange(n),batch_size=24,shuffle=True,num_workers=0,drop_last=False,generator=generator)
        for epoch in range(1,51):
            model.train();losses=[]
            for index in train:
                index=index.to('cuda:0');optimizer.zero_grad(set_to_none=True)
                prediction=model(h[index],bits[index]);loss=nn.functional.mse_loss(prediction,targets[index])
                if not torch.isfinite(loss): raise FloatingPointError('nonfinite adaptation loss')
                loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True)
                if updates['attempted']>=update_limit: raise RuntimeError('adaptation update budget exhausted')
                updates['attempted']+=1;write(dest/'updates.json',updates)
                optimizer.step();updates['completed']+=1;write(dest/'updates.json',updates)
                losses.append(float(loss.detach()))
            scheduler.step();model.eval();pred=[]
            with torch.inference_mode():
                for start in range(n,len(h),24): pred.append(model(h[start:start+24],bits[start:start+24]).flatten().cpu().numpy())
            raw=inverse(np.concatenate(pred),scaler);current=metrics(data['baseline_y'],raw)
            record=dict(epoch=epoch,train_loss=float(np.mean(losses)),validation_r2=current['r2'],validation_rmse=current['rmse'],lr=optimizer.param_groups[0]['lr'])
            history.append(record);write(dest/'progress.json',dict(history=history,updates=updates,outer_test='NOT_RUN'))
            print(json.dumps(record),flush=True)
            if current['rmse']<best['rmse']:
                best=dict(epoch=epoch,**current);best_prediction=raw.copy();save_best(dest/'best.pt',model,cfg,epoch,metadata)
        if updates['attempted']!=update_limit or updates['completed']!=update_limit: raise ValueError('adaptation exact update mismatch')
        np.savez(dest/'validation_predictions.npz',indices=np.asarray(data['ids'][n:]),sample_keys=np.asarray(data['keys'][n:]),
            y_true=data['baseline_y'],y_pred=best_prediction,scaler_mean=scaler['mean'],scaler_scale=scaler['scale'],best_epoch=best['epoch'])
        write(dest/'metrics.json',dict(status='PASS',mode=mode,task=task,fold=fold,best_epoch=best['epoch'],no_adaptation_selected=best['epoch']==0,
            r2=best['r2'],rmse=best['rmse'],baseline=initial,epochs=50,updates=updates,outer_test='NOT_RUN'))
    except BaseException as e:
        write(dest/'failure.json',dict(error=repr(e),traceback=__import__('traceback').format_exc(),epochs=len(history),updates=updates,time=now(),retry=False));raise

def budget(cpu_tests=0):
    configs=list((OUT/'adaptation/F0').glob('*/*/fold*/config.json'))
    histories=[read(p) for p in (OUT/'adaptation/F0').glob('*/*/fold*/progress.json')]
    updates=[read(p) for p in (OUT/'adaptation/F0').glob('*/*/fold*/updates.json')]
    actual={k:0 for k in LIMITS};actual.update(adaptation_starts=len(configs),epochs=sum(len(p['history']) for p in histories),cpu_tests=cpu_tests)
    if any(actual[k]>v for k,v in LIMITS.items()): raise RuntimeError('r2 scientific budget exceeded')
    write(OUT/'budget.json',dict(actual=actual,limits=LIMITS,attempted_updates=sum(p['attempted'] for p in updates),completed_updates=sum(p['completed'] for p in updates),
        completed_units=len(list((OUT/'adaptation/F0').glob('*/*/fold*/metrics.json'))),old_r1_budget='preserved read-only; not reset or exchanged',time=now()))
    return actual

def summary():
    import csv
    values={};tasks={};all_complete=True;unit_records={}
    for mode in MODES:
        records=[];tasks[mode]={}
        for task in TASKS:
            folds=[]
            for fold in range(5):
                path=folder(mode,task,fold)/'metrics.json'
                if path.exists():
                    rec=read(path)
                    if rec['status']!='PASS' or rec['outer_test']!='NOT_RUN': raise ValueError('invalid validation record')
                    folds.append(rec);records.append(rec);unit_records[mode,task,fold]=rec
            complete=len(folds)==5
            tasks[mode][task]=dict(complete=complete,folds=folds)
            if complete:
                tasks[mode][task].update(r2=dict(mean=float(np.mean([f['r2'] for f in folds])),std=float(np.std([f['r2'] for f in folds]))),
                    rmse=dict(mean=float(np.mean([f['rmse'] for f in folds])),std=float(np.std([f['rmse'] for f in folds]))))
        if len(records)==35: values[mode]=float(np.mean([r['r2'] for r in records]))
        else: all_complete=False
    baseline={}
    for task in TASKS:
        baseline[task]=[read(base_folder(task,f)/'metrics.json') for f in range(5)]
    base_macro=float(np.mean([r['r2'] for rows in baseline.values() for r in rows]))
    mechanism=max(('SHIFT','FILM'),key=lambda m:values[m]) if all_complete else None
    paired=[];comparisons={}
    pairs=[('FILM','NULL'),('FILM','SHIFT'),('SHIFT','UA'),('FILM','UA'),('NULL','UA')]
    for candidate,reference in pairs:
        if candidate not in values or (reference!='UA' and reference not in values): continue
        ref=base_macro if reference=='UA' else values[reference]
        comparisons[candidate+'-'+reference]=dict(validation_macro_delta=values[candidate]-ref,development_threshold_met=values[candidate]-ref>=.010,
            formal_effectiveness='未完成／不可比较：未授权 TEST')
        for task in TASKS:
            for fold in range(5):
                a=unit_records[candidate,task,fold];b=baseline[task][fold] if reference=='UA' else unit_records[reference,task,fold]
                paired.append(dict(candidate=candidate,reference=reference,task=task,fold=fold,r2_delta=a['r2']-b['r2'],rmse_delta=a['rmse']-b['rmse']))
    result=dict(status='PASS' if all_complete else '阻断',complete=all_complete,validation_macro=values,baseline_validation_macro=base_macro,
        tasks=tasks,comparisons=comparisons,mechanism=mechanism,main_predeclared='FILM',tie_preference='SHIFT',
        eligible_for_future_system_selection=bool(mechanism and values[mechanism]-base_macro>=.010),
        future_test_authorized=False,final_system_frozen=False,outer_test='NOT_RUN',aggregation='35 fold R2 equal mean; task 5-fold std ddof=0',time=now())
    write(OUT/'development/validation_summary.json',result)
    write(OUT/'development/mechanism_selection.json',{k:result[k] for k in ('mechanism','main_predeclared','tie_preference','eligible_for_future_system_selection','validation_macro','baseline_validation_macro','future_test_authorized','final_system_frozen','time')})
    review=OUT/'final_review';review.mkdir(parents=True,exist_ok=True)
    with (review/'paired_deltas.csv').open('w') as h:
        writer=csv.DictWriter(h,fieldnames=['candidate','reference','task','fold','r2_delta','rmse_delta']);writer.writeheader();writer.writerows(paired)
    lines=['# r2 F0 validation 开发结果','',f'完整度：{len(unit_records)}/105。原 UA validation 宏平均 R²：{base_macro:.3f}。',
        '', '| Task | NULL R² | SHIFT R² | FILM R² |', '|---|---:|---:|---:|']
    for task in TASKS:
        row=[]
        for mode in MODES:
            value=tasks[mode][task]
            row.append(f'{value["r2"]["mean"]:.3f}±{value["r2"]["std"]:.3f}' if value['complete'] else '未完成')
        lines.append('| '+task+' | '+' | '.join(row)+' |')
    lines+=['| 宏平均 | '+' | '.join(f'{values[m]:.3f}' if m in values else '未完成' for m in MODES)+' |','',
        f'统一机制选择：{mechanism or "未完成，未选择"}。只按 validation 选择，不是最终系统冻结；F1／正式 TEST 均未授权。','']
    for pair,record in comparisons.items(): lines.append(f'- {pair} validation ΔR²：{record["validation_macro_delta"]:+.3f}。')
    lines+=['','NULL 控制继续训练 head 及全局适配的影响。FILM−NULL 是 FP 信息对照；FILM−SHIFT 是乘法组件消融。validation 过线不代表正式有效。']
    (review/'validation_report.md').write_text('\n'.join(lines)+'\n')
    paper={}
    import re
    text=(ROOT/'SOTA_RESULT.md').read_text().split('## 各任务最佳 R²',1)[1]
    for line in text.splitlines():
        columns=[c.strip() for c in line.split('|')]
        if len(columns)>3 and columns[1].lower() in TASKS:
            match=re.search(r'\*\*([0-9.]+)\s*±\s*([0-9.]+)\*\*',columns[3])
            if match: paper[columns[1].lower()]=dict(model=columns[2],r2=float(match[1]),std=float(match[2]))
    if set(paper)!=set(TASKS): raise ValueError('missing seven-task paper references')
    sota=['# r2 七任务正式 SOTA 对照','','本批仅授权 validation 开发，无正式 TEST。下列七项均不可作性能结论。',
        '','| Task | r2 TEST R² | 论文 R² | 模型 | 项目 − 论文 |','|---|---|---:|---|---|']
    for task in TASKS:
        p=paper[task];sota.append(f'| {task} | 未完成／不可比较 | {p["r2"]:.3f}±{p["std"]:.3f} | {p["model"]} | — |')
    sota+=['','项目保留 paper5_outer 既有 split，FT seed=42+fold，5 fold 等权、七任务等权；历史 TEST 参与过路线开发。与论文的数据、split、预训练语料和 seed 统计不保证一致，只能作未来数值对照。']
    (review/'seven_task_comparison.md').write_text('\n'.join(sota)+'\n')
    write(review/'seven_task_results.json',dict(status='未完成／不可比较',tasks={t:dict(status='未完成／不可比较',reason='正式TEST未授权',paper=paper[t],project_r2=None,delta=None) for t in TASKS},outer_test='NOT_RUN'))
    sources={}
    for p in (OUT/'adaptation/F0').glob('*/*/fold*/feature_sources.json'):
        item=read(p);sources[item['task']+f'/fold{item["fold"]}']=item
    write(OUT/'feature_sources.json',dict(units=sources,new_encoder_forward=0,new_fp=0,readonly_caches=[str(FP_OLD),str(FP_SIDE)],source_root=str(CACHE)))
    return result

def handoff(status,detail):
    import difflib
    files=['src/training/ua2d_fp_film.py','scripts/run_ua2d_fp_film.py','tests/test_ua2d_fp_film.py']
    diff=''
    for name in files:
        path=ROOT/name
        if path.exists(): diff+=''.join(difflib.unified_diff([],path.read_text().splitlines(True),fromfile='/dev/null',tofile='b/'+name))
    review=OUT/'final_review';review.mkdir(parents=True,exist_ok=True);(review/'implementation.diff').write_text(diff)
    path=ROOT/'CURRENT_PLAN.md';text=path.read_text();start=text.index('- 状态：');end=text.index('\n',start)
    text=text[:start]+f'- 状态：**{status}（r2 F0）**。用户当前消息已授权仅 NULL／SHIFT／FILM 开发及一次 CPU 专项测试；P/A/T/M/I、旧核、F1 与正式 TEST 保持暂停。'+text[end:]
    marker='### r2 F0 授权执行记录'
    if marker in text:
        a=text.index(marker);b=text.index('\n## 13.',a);text=text[:a]+text[b:]
    at=text.index('\n## 13.')
    b=read(OUT/'budget.json');a=b['actual']
    block=marker+'\n\n'+f'- 时间：{now()}；{detail}\n'
    block+='- 当前消息明确授权 r2 F0 105 starts／5250 epochs 及唯一 FP CPU suite；本地 dev／3503fbe780a5399410d6db6055305925765c3ff8。已有修改全部保留，未读取 TODO，未操作远程。\n'
    block+='- 执行前发现旧 r1 的控制器 125308、127331 仍运行，已 SIGSTOP 保持暂停；四个 P_HLR 在途进程 134866、136384、138199、138354 已 SIGTERM（保留已保存 best／进度，shell 记录真实退出）。未恢复旧控制器、重试或改写 r1 预算。新代码不写旧 r1 产物。\n'
    block+=f'- r2 实耗：适配 {a["adaptation_starts"]} starts／{a["epochs"]} epochs；updates attempted/completed={b["attempted_updates"]}/{b["completed_updates"]}；CPU suite {a["cpu_tests"]} 次；新增 PT／KRR／FP／PH／构象／encoder forward／正式 TEST 均 0。\n'
    block+='- 入口 `scripts/run_ua2d_fp_film.py`；主窗口 `Uni-Poly:ua2d_sota_fp_r2`；单元窗口 `film_r2_<mode>_<task>_<fold>`；CPU 窗口 `film_r2_final_cpu`。command／cwd／start／end／真实 exit／stdout／stderr 在 `logs/ua2d_sota_20261018/r2/fp_fusion/`；结果和预算在对应 results 根。\n'
    for name in ('validation_report.md','seven_task_comparison.md'):
        p=review/name
        if p.exists(): block+='\n'+p.read_text()+'\n'
    block+='- 仅执行方自检，不代表 Codex 独立审查通过；不归档 PROJECT_HISTORY.md。不生成 selection_before_test.json 或正式名单，当前只作开发机制选择。\n'
    path.write_text(text[:at]+'\n'+block+text[at:])

def restore_package(package,*,mode,task,fold):
    """Strict restoration for the cached h interface; core weights stay referenced."""
    expected=dict(plan='UA2D-SOTA-20261018-01',revision='r2',round='F0',mode=mode,task=task,fold=fold,seed=42+fold,recipe=RECIPE)
    if package.get('config')!=expected or mode not in MODES or task not in TASKS or fold not in range(5): raise ValueError('adaptation package configuration mismatch')
    if package.get('feature_spec')!=FEATURE_SPEC or package.get('outer_test')!='NOT_RUN': raise ValueError('adaptation package feature/stage mismatch')
    epoch=package.get('best_epoch')
    if type(epoch) is not int or not 0<=epoch<=50 or package.get('no_adaptation_selected')!=(epoch==0): raise ValueError('adaptation package epoch mismatch')
    if package.get('frozen_components')!=['O8','readout','norm2']: raise ValueError('adaptation package freeze mismatch')
    core=str(base_folder(task,fold)/'best.pt');metadata=package.get('metadata',{})
    if package.get('encoder_weights_reference')!=core or metadata.get('core')!=core: raise ValueError('adaptation package core mismatch')
    scaler=metadata.get('scaler',{})
    if scaler.get('fit')!='train' or scaler.get('transform')!='standard' or len(scaler.get('mean',[]))!=1 or len(scaler.get('scale',[]))!=1 or scaler['scale'][0]<=0:
        raise ValueError('adaptation package scaler mismatch')
    finite(scaler['mean']);finite(scaler['scale'])
    state=package['state_dict'];head={k.removeprefix('head.'):v for k,v in state.items() if k.startswith('head.')}
    model=CachedFusion(mode,head);strict_restore(model,state)
    return model
