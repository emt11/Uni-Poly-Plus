"""Explicit six-arm clean-FP campaign; importing starts no experiment."""
from collections import deque
from dataclasses import asdict
import datetime
import json
import math
import os
from pathlib import Path
import traceback
import numpy as np
import torch
from torch import nn
from src.modules.ua2d_cleanfp_pretrain import CleanConfig, CleanPretrainer, SCHEMA
from src.modules.glt_dual import strict_module_state
from src.training.ua2d_pretrain_fp import read, write, freeze, finite, now

ROOT=Path(__file__).resolve().parents[2]
PLAN='UA2D-CLEANFP-READOUT-DISTILL-20261019-01'
REVISION='r1'
OUT=ROOT/'results/ua2d_cleanfp_readout_distill_20261019/r1'
LOG=ROOT/'logs/ua2d_cleanfp_readout_distill_20261019/r1'
OLD=ROOT/'results/ua2d_pretrain_fp_20261019/fast_r4'
CACHE=OLD/'prepared_cache'
SOURCE=ROOT/'data/processed/mcl_ph_r4_20260930/pi1m_direct_01.staging'
INDEX=ROOT/'results/mcl_ph_next_20260929/p0/direct_indices_v1/summary.json'
DOWN=ROOT/'data/processed/mcl_ph_r4_20260930/downstream_direct_01.staging'
SPLIT=ROOT/'data/splits/mips_paper5_outer'
HEADS=ROOT/'results/glt_optimization_20261011/r4/U1/initialization/heads_2d'
TASKS=('eea','egb','egc','ei','eps','nc','xc')
ARMS=('G_ALL1','G_DYN','Q_ALL1','Q_DYN','QL_ALL1','QL_DYN')
LIMITS=dict(pt_starts=6,pt_updates=30000,pt_presentations=30240000,ft_starts=210,
    ft_epochs=14700,ft_updates=336000,validation_units=14700,test_starts=210,test_graphs=35250,
    fp_generated=0,cpu_suites=1)


def objective_config(arm):
    if arm not in ARMS:raise ValueError('arm outside fixed matrix')
    architecture, policy=arm.split('_')
    return asdict(CleanConfig(readout='mean' if architecture=='G' else 'query',
                              local=architecture=='QL',weight_policy=policy))


def parent_path(arm):
    objective_config(arm)
    return OUT/'pretrain'/arm/'deploy.pt'


def check_limits(counts):
    if set(counts)!=set(LIMITS) or any(type(counts[k]) is not int or not 0<=counts[k]<=v for k,v in LIMITS.items()):
        raise RuntimeError('campaign budget exceeded: '+repr(counts))


def validate_cache(path):
    from src.dataset.ua2d_fast_pretrain import CACHE_SCHEMA
    from src.training.ua2d_pretrain_fp import FP_SPEC
    meta=read(Path(path)/'metadata.json')
    if (meta.get('schema')!=CACHE_SCHEMA or meta.get('status')!='COMPLETE'
            or meta.get('source')!=str(SOURCE) or meta.get('index')!=str(INDEX)
            or meta.get('train_rows')!=911391 or meta.get('completed')!=911391):
        raise ValueError('prepared source/index/schema/rows mismatch')
    # Historical optimization metadata is not a chemical fingerprint parameter.
    excluded={'weight','head','loss'}
    chemical={k:v for k,v in FP_SPEC.items() if k not in excluded}
    actual={k:v for k,v in meta.get('spec',{}).items() if k not in excluded}
    if actual!=chemical:raise ValueError('Morgan chemical label definition mismatch')
    return meta


def initialize():
    from src.training.glt_dual_runtime import save_checkpoint
    from src.modules.ua2d_galformer_pretrain import SCHEMA as OLD_SCHEMA
    source=ROOT/'results/ua2d_pretrain_fp_20261019/r1/initialization'
    common=torch.load(source/'common_step0.pt',map_location='cpu',weights_only=False)
    heads=torch.load(source/'task_heads/V1.pt',map_location='cpu',weights_only=False)
    if (common.get('schema')!=OLD_SCHEMA or common.get('step')!=0 or common.get('seed')!=42
            or common.get('plan')!='UA2D-PRETRAIN-FP-20261019-01'
            or heads.get('schema')!=OLD_SCHEMA or heads.get('step')!=0 or heads.get('strategy')!='mask_align'):
        raise ValueError('unsupported historical random step0')
    folder=OUT/'initialization';folder.mkdir(exist_ok=False)
    torch.manual_seed(42)
    model=CleanPretrainer(CleanConfig())
    expected={k for k in model.state_dict() if not k.startswith(('line_head.','fp_head.'))}
    if set(common['state_dict'])!=expected:raise ValueError('common step0 module set mismatch')
    if set(heads['state_dict'])!={k for k in model.state_dict() if k.startswith('line_head.')}:
        raise ValueError('V1 head module set mismatch')
    state=dict(common['state_dict'],**heads['state_dict'])
    state.update({k:v for k,v in model.state_dict().items() if k.startswith('fp_head.')})
    strict_module_state(model,state)
    save_checkpoint(folder/'common_step0.pt',dict(schema=SCHEMA,role='common_step0',step=0,seed=42,plan=PLAN,
        source=str(source),state_dict=state))
    extra=CleanPretrainer(CleanConfig(readout='query',local=True))
    for name in ('readout','local_predictor'):
        save_checkpoint(folder/(name+'.pt'),dict(schema=SCHEMA,role=name,step=0,plan=PLAN,
            state_dict=getattr(extra,name).state_dict()))
    for arm in ARMS:freeze(OUT/'configs'/f'{arm}.json',objective_config(arm))
    write(folder/'summary.json',dict(schema=SCHEMA,plan=PLAN,common_source=str(source),
        common_seed=42,fp_seed=20261019,readout_seed=20261020,local_seed=20261021,trained_parent=False))


def load_initialization(model,path):
    path=Path(path)
    common=torch.load(path,map_location='cpu',weights_only=False)
    if (common.get('schema')!=SCHEMA or common.get('role')!='common_step0'
            or common.get('step')!=0 or common.get('seed')!=42 or common.get('plan')!=PLAN):
        raise ValueError('invalid new common step0')
    state=dict(common['state_dict'])
    for name,enabled in [('readout',model.config.readout=='query'),('local_predictor',model.config.local)]:
        if enabled:
            extra=torch.load(path.parent/(name+'.pt'),map_location='cpu',weights_only=False)
            if extra.get('schema')!=SCHEMA or extra.get('role')!=name or extra.get('step')!=0 or extra.get('plan')!=PLAN:
                raise ValueError('extra module initialization mismatch')
            state.update({name+'.'+k:v for k,v in extra['state_dict'].items()})
    strict_module_state(model,state)


class LossBalancer:
    """Detached global-window state; applied weights never change within a window."""
    def __init__(self,names,policy):
        if policy not in ('ALL1','DYN'):raise ValueError('invalid policy')
        self.names=tuple(names);self.policy=policy;self.weights=dict.fromkeys(names,1.)
        self.history=deque(maxlen=40);self.initial_sum=np.zeros(len(names),dtype=np.float64)
        self.baseline=None;self.observations=0;self.adjustments=0

    def observe(self,losses,counts):
        values=finite([losses[k] for k in self.names])
        if self.policy=='ALL1' or not all(counts[k]>0 for k in self.names):return
        self.observations+=1;self.history.append(values.copy())
        if self.observations<=200:self.initial_sum+=values
        if self.observations==200:self.baseline=self.initial_sum/200
        if self.observations<200 or (self.observations-200)%20:return
        history=np.stack(self.history)
        ratio=history[-20:].mean(0)/np.maximum(history[-40:-20].mean(0),1e-8)
        probability=np.exp(ratio-ratio.max());probability/=probability.sum()
        score=np.maximum(probability,1e-8)/np.maximum(self.baseline,1e-8)
        low,high=0.,1.
        while np.clip(high*score,.25,4.).sum()<len(self.names):high*=2
        for _ in range(80):
            middle=(low+high)/2
            if np.clip(middle*score,.25,4.).sum()<len(self.names):low=middle
            else:high=middle
        target=np.clip(((low+high)/2)*score,.25,4.)
        previous=np.array([self.weights[k] for k in self.names])
        self.weights=dict(zip(self.names,(.9*previous+.1*target).tolist()))
        self.adjustments+=1

    def state_dict(self):
        return dict(revision=SCHEMA,names=list(self.names),policy=self.policy,weights=self.weights,
            history=[x.tolist() for x in self.history],initial_sum=self.initial_sum.tolist(),
            baseline=None if self.baseline is None else self.baseline.tolist(),
            observations=self.observations,adjustments=self.adjustments)

    def load_state_dict(self,state):
        if state['revision']!=SCHEMA or state['names']!=list(self.names) or state['policy']!=self.policy:
            raise ValueError('balancer restore mismatch')
        weights=finite([state['weights'][k] for k in self.names])
        if (weights<.25).any() or (weights>4).any() or abs(weights.sum()-len(weights))>1e-7:
            raise ValueError('invalid saved dynamic weights')
        if self.policy=='ALL1' and not np.array_equal(weights,np.ones_like(weights)):
            raise ValueError('ALL1 state changed weights')
        self.weights=dict(zip(self.names,weights.tolist()))
        self.history=deque((finite(x) for x in state['history']),maxlen=40)
        self.initial_sum=finite(state['initial_sum']);self.baseline=None if state['baseline'] is None else finite(state['baseline'])
        self.observations=int(state['observations']);self.adjustments=int(state['adjustments'])


class Data(torch.utils.data.Dataset):
    def __init__(self,task,fold,formal=False):
        from src.training.glt_dual_runtime import open_source
        from scripts.finetune_mcl_ph import compact_train_validation_indices
        if task not in TASKS or fold not in range(5):raise ValueError('unit outside matrix')
        manifest=read(SPLIT/f'{task}.json');entry=manifest['folds'][fold]
        if manifest.get('validation_is_test') is not False:raise ValueError('validation/test separation invalid')
        if formal:
            frozen=read(OUT/'selection/final_frozen.json')
            if not frozen['frozen']:raise ValueError('TEST before freeze')
            self.indices=sorted(entry['test_indices']);self.train=[];self.validation=list(range(len(self.indices)))
        else:self.indices,self.train,self.validation=compact_train_validation_indices(entry['train_indices'],entry['validation_indices'])
        self.source,frame=open_source(DOWN,DOWN,task=task,selected_indices=self.indices,split_manifest=manifest,verify_key_digest=False)
        if frame.original_row.astype(int).tolist()!=self.indices:raise ValueError('selected source row mismatch')
        self.raw_targets=finite(frame.label.to_numpy());self.targets=self.raw_targets.copy();self.cache={}
        self.keys=np.asarray([s[0].hex() for s in self.source.samples])
    def __len__(self):return len(self.indices)
    def __getitem__(self,i):
        from src.training.glt_dual_runtime import pretrain_source_fields
        from src.dataset.glt_dual import build_glt_2d_sample
        from scripts.finetune_mcl_ph import clone_clean_record
        if i not in self.cache:
            top,_,text,static,_=pretrain_source_fields(self.source,i)
            self.cache[i]=build_glt_2d_sample(top,text,static=static)
        d=clone_clean_record(self.cache[i]);d.y=torch.tensor([self.targets[i]],dtype=torch.float32);return d
    def set_target_override(self,y):self.targets=np.asarray(y,dtype=np.float64)
    def close(self):self.source.close()


def loader(ds,ids,shuffle=False,seed=42):
    from src.dataset.glt_dual import glt_2d_collate
    return torch.utils.data.DataLoader(torch.utils.data.Subset(ds,ids),batch_size=24,shuffle=shuffle,
        num_workers=0,drop_last=False,collate_fn=glt_2d_collate,generator=torch.Generator().manual_seed(seed) if shuffle else None)


def unit_path(arm,task,fold,formal=False):return OUT/('formal' if formal else 'finetune')/arm/task/f'fold{fold}'


def unit_config(arm,task,fold):
    from scripts.run_glt_align_2d_unit import PROFILE
    return dict(plan=PLAN,revision=REVISION,arm=arm,task=task,fold=fold,seed=42+fold,
        parent=str(parent_path(arm)),head_initial=str(HEADS/task/f'fold{fold}.pt'),
        profile=PROFILE,split=str(SPLIT/f'{task}.json'),objectives=objective_config(arm),prediction_path='pure_2d')


def finetune(arm,task,fold,formal=False):
    from src.training.glt_dual_runtime import require_tmux,save_checkpoint
    from src.modules.ua2d_cleanfp_pretrain import restore_downstream
    from src.modules.glt_dual import strict_module_state
    from src.utils import set_global_seed,scale_targets,evaluate,train_epoch,TargetScaler
    from sklearn.preprocessing import StandardScaler
    import scripts.finetune_mcl_ph as old
    from src.modules.glt_adaptation import AdaptationAdapter
    require_tmux()
    if os.environ.get('CUDA_VISIBLE_DEVICES') not in ('2','3'):raise ValueError('worker must use physical GPU2 or GPU3 only')
    folder=unit_path(arm,task,fold,formal);folder.mkdir(parents=True,exist_ok=False)
    cfg=unit_config(arm,task,fold);ds=None;history=[]
    progress=dict(status='RUNNING',attempted_updates=0,completed_updates=0,epochs_started=0,completed_epochs=0,
        validation_started=0,validation_completed=0,test_graphs=0)
    def record():write(folder/'consumption.json',progress)
    record()
    try:
        set_global_seed(42+fold)
        package=torch.load(cfg['parent'],map_location='cpu',weights_only=False)
        if package.get('step')!=5000 or package.get('config')!=cfg['objectives'] or package.get('execution',{}).get('revision')!=REVISION:
            raise ValueError('PT parent/config/revision mismatch')
        head=torch.load(cfg['head_initial'],map_location='cpu',weights_only=False)
        model=restore_downstream(package,head)
        old.configure_periodic_tdl_dropout(model,task,profile='uniform_u1');model.to('cuda:0');del package
        if formal:
            selected=read(OUT/'selection/final_frozen.json')['units'][f'{arm}/{task}/{fold}']
            saved=torch.load(unit_path(arm,task,fold)/'best.pt',map_location='cpu',weights_only=False)
            scaler_json=read(unit_path(arm,task,fold)/'scaler.json')
            if selected['config']!=cfg or saved['config']!=cfg or selected['best']!=str(unit_path(arm,task,fold)/'best.pt') or selected['best_epoch']!=saved['best_epoch'] or selected['scaler']!=scaler_json or saved['scaler']!=scaler_json:
                raise ValueError('frozen task/fold/head/parent/scaler mismatch')
            strict_module_state(model,saved['state_dict']);best_epoch=saved['best_epoch']
        ds=Data(task,fold,formal)
        if formal:
            if ds.indices!=selected['test_indices']:raise ValueError('frozen TEST indices mismatch')
            scaler=TargetScaler(task,StandardScaler(),transform_mode='standard')
            scaler.scaler.mean_=np.asarray(scaler_json['mean']);scaler.scaler.scale_=np.asarray(scaler_json['scale'])
            scaler.scaler.var_=scaler.scaler.scale_**2;scaler.scaler.n_features_in_=1
            ds.set_target_override(scaler.transform(ds.raw_targets).reshape(-1))
        else:
            scaler=scale_targets(ds,task,train_indices=ds.train,transform_mode='standard')
            scaler_json=dict(mean=scaler.scaler.mean_.tolist(),scale=scaler.scaler.scale_.tolist(),fit='train',transform='standard')
            write(folder/'scaler.json',scaler_json);save_checkpoint(folder/'head_initial.pt',head)
        validation=loader(ds,ds.validation)
        if formal:
            # Record actual graph presentations at each evaluation batch.
            def observed():
                for batch in validation:
                    progress['test_graphs']+=len(batch.y);record();yield batch
            loss,r2,truth,pred=evaluate(model,observed(),nn.MSELoss(),'cuda:0',scaler=scaler)
            if not math.isfinite(loss):raise FloatingPointError('nonfinite formal loss')
        else:
            train=loader(ds,ds.train,True,42+fold);expected=70*len(train)
            set_global_seed(42+fold);best_r2=-math.inf;best_state=None
            class HeadOnly(AdaptationAdapter):
                def train(self,mode=True):
                    super().train(mode)
                    if mode:model.eval();model.head.train()
                    return self
            class NoStep:
                def step(self):pass
            def observe(kind):
                progress[kind+'_updates']+=1
                if progress['attempted_updates']>expected:raise RuntimeError('FT unit update budget exhausted')
                record()
            for stage,epochs in [('head',10),('joint',60)]:
                old.periodic_tdl_trainability(model,head_only=stage=='head')
                wrapped=HeadOnly(model) if stage=='head' else AdaptationAdapter(model)
                optimizer,groups=old.periodic_tdl_optimizer(model,task,head_only=stage=='head',profile='uniform_u1')
                optimizer.register_step_pre_hook(lambda opt,args,kwargs:observe('attempted'))
                optimizer.register_step_post_hook(lambda opt,args,kwargs:observe('completed'))
                scheduler=(torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=10,eta_min=3e-5) if stage=='head' else
                    torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=10,T_mult=1,eta_min=1e-6))
                for stage_epoch in range(epochs):
                    progress['epochs_started']+=1;record()
                    result=train_epoch(wrapped,train,nn.MSELoss(),optimizer,NoStep(),'cuda:0',amp_dtype='fp32',max_grad_norm=5.,fail_nonfinite=True)
                    scheduler.step();progress['validation_started']+=1;record()
                    val_loss,r2,y,p=evaluate(wrapped,validation,nn.MSELoss(),'cuda:0',scaler=scaler)
                    finite([val_loss,r2]);finite(y);finite(p)
                    progress['validation_completed']+=1;progress['completed_epochs']+=1
                    rec=dict(epoch=progress['completed_epochs'],stage=stage,stage_epoch=stage_epoch+1,train_loss=result[0],
                        validation_loss=val_loss,validation_r2=r2,learning_rates=[g['lr'] for g in optimizer.param_groups],optimizer_groups=groups)
                    history.append(rec);write(folder/'history.json',history);record();print(json.dumps(rec),flush=True)
                    if r2>best_r2:
                        best_r2=float(r2);best_epoch=progress['completed_epochs']
                        best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                        truth=np.asarray(y).copy();pred=np.asarray(p).copy()
                        save_checkpoint(folder/'best.pt',dict(state_dict=best_state,config=cfg,best_epoch=best_epoch,scaler=scaler_json,
                            task=task,fold=fold,seed=42+fold,parent=cfg['parent'],head_initial=cfg['head_initial'],outer_test='NOT_RUN'))
            if progress['completed_updates']!=expected or progress['attempted_updates']!=expected:raise ValueError('FT budget/count mismatch')
            r2=best_r2
        truth=finite(truth).reshape(-1);pred=finite(pred).reshape(-1)
        np.savez(folder/('test_predictions.npz' if formal else 'validation_predictions.npz'),y_true=truth,y_pred=pred,
            indices=np.asarray([ds.indices[i] for i in ds.validation]),sample_keys=ds.keys[ds.validation],best_epoch=best_epoch,
            scaler_mean=scaler_json['mean'],scaler_scale=scaler_json['scale'])
        from src.training.xc_kernel import metrics
        m=metrics(truth,pred);finite(list(m.values()))
        write(folder/'metrics.json',dict(status='COMPLETE',config=cfg,best_epoch=best_epoch,**m,outer_test='RUN' if formal else 'NOT_RUN'))
        progress['status']='COMPLETE';record()
    except BaseException as error:
        progress['status']='FAILED';record();write(folder/'failure.json',dict(error=repr(error),traceback=traceback.format_exc(),time=now(),retry=False));raise
    finally:
        if ds:ds.close()


def aggregate(arm,formal=False):
    rows={};units={}
    for task in TASKS:
        scores=[]
        for fold in range(5):
            folder=unit_path(arm,task,fold,formal);m=read(folder/'metrics.json');c=read(folder/'consumption.json')
            if m['status']!='COMPLETE' or c['status']!='COMPLETE' or m['config']!=unit_config(arm,task,fold):raise ValueError('incomplete unit')
            if not formal and (c['completed_epochs']!=70 or c['completed_updates']!=c['attempted_updates']):raise ValueError('incomplete FT')
            scores.append(float(m['r2']));units[f'{arm}/{task}/{fold}']=dict(metrics=str(folder/'metrics.json'),r2=m['r2'],
                best=str(unit_path(arm,task,fold)/'best.pt'),predictions=str(folder/('test_predictions.npz' if formal else 'validation_predictions.npz')),
                best_epoch=m['best_epoch'],config=m['config'],scaler=read(unit_path(arm,task,fold)/'scaler.json'))
        finite(scores);rows[task]=dict(folds=scores,mean=float(np.mean(scores)),std=float(np.std(scores,ddof=1)))
    return dict(tasks=rows,macro=float(np.mean([rows[t]['mean'] for t in TASKS])),units=units)


PAIRS=(('Q_ALL1','G_ALL1'),('Q_DYN','G_DYN'),('QL_ALL1','Q_ALL1'),('QL_DYN','Q_DYN'),
       ('G_DYN','G_ALL1'),('Q_DYN','Q_ALL1'),('QL_DYN','QL_ALL1'))


def freeze_final():
    values={a:aggregate(a) for a in ARMS}
    best=max(ARMS,key=lambda a:values[a]['macro'])
    delta=values[best]['macro']-values['G_ALL1']['macro']
    units={}
    for a in ARMS:
        for key,u in values[a]['units'].items():
            entry=read(SPLIT/f"{u['config']['task']}.json")['folds'][u['config']['fold']]
            units[key]=dict(u,test_indices=sorted(entry['test_indices']))
    freeze(OUT/'selection/final_frozen.json',dict(plan=PLAN,revision=REVISION,frozen=True,time=now(),
        baseline='G_ALL1',highest_validation=best,proposed=best if delta>=.010 else 'G_ALL1',
        delta_val=delta,validation_gate=delta>=.010,validation=values,units=units,formal_arms=list(ARMS),
        comparisons=[dict(candidate=a,reference=b,delta=values[a]['macro']-values[b]['macro']) for a,b in PAIRS],
        rule='full precision validation only; fixed tie order; TEST never reselects',budget=read(OUT/'budget.json')))


def reports(status,error=None):
    folder=OUT/'final_review';folder.mkdir(exist_ok=True)
    vals,tests={},{}
    for a in ARMS:
        for formal,target in [(False,vals),(True,tests)]:
            try:target[a]=aggregate(a,formal)
            except (FileNotFoundError,ValueError):pass
    comparisons=[]
    for a,b in PAIRS:
        row=dict(candidate=a,reference=b)
        if a in vals and b in vals:row['delta_validation']=vals[a]['macro']-vals[b]['macro']
        if a in tests and b in tests:row['delta_test']=tests[a]['macro']-tests[b]['macro']
        row['effective']=row.get('delta_validation',-math.inf)>=.010 and row.get('delta_test',-math.inf)>=.010
        comparisons.append(row)
    selected=read(OUT/'selection/final_frozen.json') if (OUT/'selection/final_frozen.json').exists() else None
    summary=dict(plan=PLAN,revision=REVISION,status=status,error=error,validation=vals,test=tests,
        comparisons=comparisons,budget=read(OUT/'budget.json'),independent_review=False)
    if selected:
        p=selected['proposed'];summary.update(proposed=p,highest_validation=selected['highest_validation'],
            validation_gate=selected['validation_gate'])
        if p in tests and 'G_ALL1' in tests:
            delta=tests[p]['macro']-tests['G_ALL1']['macro']
            summary.update(delta_test=delta,effective=selected['validation_gate'] and delta>=.010)
    write(folder/'results.json',summary)
    for filename,values in [('validation_report.md',vals),('test_report.md',tests)]:
        lines=[f'# {PLAN} / {REVISION}',f'状态：{status}；独立审查尚未进行。',
            '五折seed42–46与fold绑定；先任务内五折均值，再七任务等权；任务标准差ddof=1。',
            '| Arm | '+' | '.join(t.upper() for t in TASKS)+' | Macro R² |','|---|'+'---|'*8]
        for a in ARMS:
            cells=([f"{values[a]['tasks'][t]['mean']:.3f}±{values[a]['tasks'][t]['std']:.3f}" for t in TASKS]
                   +[f"{values[a]['macro']:.3f}"] if a in values else ['未完成／不可比较']*8)
            lines.append('| '+a+' | '+' | '.join(cells)+' |')
        lines.append('完整精度与路径见results.json；TEST不参与选择。')
        (folder/filename).write_text('\n'.join(lines)+'\n')
    references=dict(eea=(.934,'Periodic-TDL'),egb=(.923,'Periodic-TDL (+Chem,+RC)'),
        egc=(.911,'Periodic-TDL (+Chem,+RC)'),ei=(.839,'Periodic-TDL'),eps=(.787,'Periodic-TDL'),
        nc=(.856,'Periodic-TDL'),xc=(.423,'Periodic-TDL'))
    lines=['# 七任务 SOTA 数值对照','| Task | 论文模型 | 参考 | '+' | '.join(ARMS)+' |',
           '|---|---|---:|'+'---|'*6]
    for task,(ref,model) in references.items():
        cells=[]
        for a in ARMS:
            if a not in tests:cells.append('未完成／不可比较：无完整正式结果')
            else:
                v=tests[a]['tasks'][task]
                cells.append(f"{v['mean']:.3f}±{v['std']:.3f} / {v['mean']-ref:+.3f}")
        lines.append(f'| {task.upper()} | {model} | {ref:.3f} | '+' | '.join(cells)+' |')
    lines+=['差值=项目−论文；五折seed42–46，任务内均值及样本标准差ddof=1。',
        '论文为逐任务最佳包络；数据／split／预训练语料同一性未证明，只作数值对照，不宣称公平超过论文或独立盲测。',
        '正式预测：formal/{arm}/{task}/fold{fold}/test_predictions.npz；完整精度和日志／预算见results.json及execution_manifest.json。']
    (folder/'sota_comparison.md').write_text('\n'.join(lines)+'\n')
    lines=['# 预定成对比较','| Candidate | Reference | Δvalidation | ΔTEST | ≥0.010有效 |','|---|---|---:|---:|---|']
    for row in comparisons:
        dv=f"{row['delta_validation']:+.3f}" if 'delta_validation' in row else '未完成'
        dt=f"{row['delta_test']:+.3f}" if 'delta_test' in row else '未完成'
        lines.append(f"| {row['candidate']} | {row['reference']} | {dv} | {dt} | {row['effective']} |")
    (folder/'comparison_report.md').write_text('\n'.join(lines)+'\n')
