"""Independent r5 Stage0 execution; parent scientific code and inputs remain read-only."""
import json
import math
import os
from pathlib import Path
import time
import traceback
import numpy as np
import torch
from src.modules import ua3d_base_opt as m
from src.dataset import ua3m_paperfuse as d
from src.training import ua3d_spatial as parent
from src.training.ua3m_paperfuse import (read,write,exclusive,now,ignore_termination,seed_all,
    finite,metrics,process_identity,process_alive,scaler_validate,transform,summary,SOTA,
    require_tmux,save_checkpoint,pt_learning_rate,window_objective,save_predictions)
ROOT=Path(__file__).resolve().parents[2]
PLAN='UA3D-BASE-OPT-20261026-01';REVISION='r5';EXECUTION_REVISION='r5-stage0-once'
OUT=ROOT/'results/ua3d_base_opt_20261026/r5';LOG=ROOT/'logs/ua3d_base_opt_20261026/r5'
PARENT=parent.PARENT;DOWN=parent.DOWN;SPLIT=parent.SPLIT;PT_CACHE=parent.PT_CACHE;MAP_ROOT=parent.MAP_ROOT
INITIAL=OUT/'initialization/common_step0.pt'
ARMS=m.ARMS;TASKS=m.TASKS;ALLOWED_GPUS=(2,)
REFERENCE=0.795244745287872
LIMITS=dict(pt_starts=4,pt_updates=40000,pt_microbatches=240000,pt_presentations=40320000,
    ft_starts=140,ft_epochs=9800,ft_updates=224000,test_starts=140,test_graphs=23500,cpu_suites=1)
ZERO_COST=dict(fp_generation=0,geometry_generation=0,kernel_fit=0,fp_prediction=0,additional_seeds=0)

def parent_path(arm):return OUT/'pretrain'/arm/'deploy.pt'
def head_path(task,fold):return parent.head_path(task,fold)
def prepared_root():return PARENT
def base_path(task,fold):return parent.base_path(task,fold)
def unit_path(arm,task,fold,formal=False,mode='FULL'):
    if mode!='FULL':raise ValueError('probe not authorized')
    return OUT/('formal' if formal else 'finetune')/mode/arm/task/f'fold{fold}'

def config(arm,task,fold,mode='FULL'):
    if arm not in ARMS or task not in TASKS or type(fold) is not int or fold not in range(5) or mode!='FULL':raise ValueError('outside Stage0 matrix')
    return dict(plan=PLAN,revision=REVISION,arm=arm,mode=mode,task=task,fold=fold,seed=42+fold,
        parent=str(parent_path(arm)),head=str(head_path(task,fold)),input_cache=str(PARENT/'inputs'),
        scaler_reference=str(base_path(task,fold)/'scaler.json'),model=m.CONFIG,world_size=1,batch=24,
        epochs=70,head_epochs=10,precision='FP32',workers=0,
        loss='MSE+0.08*ALIGN_TMOL-0.08*gate_entropy',selection='strict original-unit validation RMSE')

def update_plan(status,phase,detail):
    path=ROOT/'CURRENT_PLAN.md';text=path.read_text()
    start='<!-- UA3D_BASE_OPT_R5_STATUS_START -->';end='<!-- UA3D_BASE_OPT_R5_STATUS_END -->'
    block=(start+'\n'+f'- 更新时间：{now()}；Stage0 状态：**{status}**；阶段：{phase}。\n'
        +'- 用户本轮固定提示词已明确授权 Stage0；Stage1—5 未授权。\n'
        +f'- {detail}\n- budget：`{OUT.relative_to(ROOT)}/budget.json`；日志：`{LOG.relative_to(ROOT)}`。\n'
        +'- independent_review=PENDING；最终交付待审查，不自动替换 BASE。\n'+end)
    if start in text:
        left,rest=text.split(start,1);_,right=rest.split(end,1);text=left+block+right
    else:text=text.rstrip()+'\n\n'+block+'\n'
    # Only current status wording changes; preserve the user's planning text.
    text=text.replace('状态：**待授权**','状态：**'+status+'**',2)
    path.write_text(text)
    write(OUT/'status.json',dict(status=status,phase=phase,detail=detail,time=now(),independent_review='PENDING'))

def check_inputs():
    pt=read(PT_CACHE/'metadata.json');mapping=read(MAP_ROOT/'metadata.json')
    if (pt.get('schema')!='ua2d_prepared_templates_r4' or pt.get('status')!='COMPLETE' or pt.get('train_rows')!=911391
        or mapping.get('status')!='COMPLETE' or mapping.get('records')!=910756
        or mapping.get('eligibility_semantics')!='logical_slot_not_source_row'):raise ValueError('PT cache/schema mismatch')
    cache=d.InputCache(PARENT/'inputs')
    try:d.validate_missing_inventory(read(PARENT/'inputs/geometry_missing.json'))
    finally:cache.close()
    original=read(PARENT/'inputs/original_splits.json')
    for task in TASKS:
        split=read(SPLIT/f'{task}.json')
        if split!=original[task]:raise ValueError('shared split changed')
        for fold in range(5):
            if not head_path(task,fold).is_file():raise FileNotFoundError(head_path(task,fold))
            if not (base_path(task,fold)/'scaler.json').is_file():raise FileNotFoundError('missing fixed scaler')
    return dict(pt=pt,mapping=mapping,splits=original)

def check_prepared():
    if not INITIAL.is_file() or not (OUT/'initialization/QFP_step0.pt').is_file():raise FileNotFoundError('cycle step0 missing')
    frozen=read(OUT/'identity.json')
    if frozen['plan']!=PLAN or frozen['revision']!=REVISION:raise ValueError('cycle identity mismatch')
    for task in TASKS:
        if read(SPLIT/f'{task}.json')!=frozen['inputs']['splits'][task]:raise ValueError('split identity mismatch')

def initialize_cycle():
    require_tmux()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU initialization must hide GPU')
    inputs=check_inputs();seed_all(42)
    model=m.BaseOptFuse('P_BASE',True)
    sources=[parent.INITIAL,parent.INITIAL.parent/'task_heads/V1.pt',parent.modal_path('BASE')]
    common,heads,modal=[torch.load(p,map_location='cpu',weights_only=False) for p in sources]
    evidence=m.native.load_initial(model,common,heads,modal)
    folder=OUT/'initialization';folder.mkdir(exist_ok=False)
    save_checkpoint(INITIAL,dict(plan=PLAN,revision=REVISION,step=0,seed=42,config=m.CONFIG,
        state_dict=model.state_dict(),sources=list(map(str,sources)),initialization=evidence))
    independent=m.BaseOptFuse('P_SYM_IND',True)
    save_checkpoint(folder/'QFP_step0.pt',dict(seed=m.QFP_SEED,state_dict=independent.QFP.state_dict()))
    del common,heads,modal,model,independent
    identity=dict(plan=PLAN,revision=REVISION,based_on='dev/2992e3cd095ba636c0f120251c43ca435ae3ea89 plus protected dirty CURRENT_PLAN.md and TODO.md',
        common_step0=str(INITIAL),sources=list(map(str,sources)),inputs=inputs,model=m.CONFIG,reference=REFERENCE,
        qfp_seed=m.QFP_SEED,ft_policy=m.CONFIG['ft_policy'],physical_gpu=2,world_size=1,workers=0,time=now())
    exclusive(OUT/'identity.json',identity)
    write(folder/'initialization.json',dict(**evidence,common_step0=str(INITIAL),qfp_seed=m.QFP_SEED,shared_parameters='strict same state_dict for all four arms'))

def frozen_unit(frozen,arm,task,fold,mode='FULL'):
    if frozen.get('plan')!=PLAN or frozen.get('revision')!=REVISION or frozen.get('frozen') is not True:raise ValueError('TEST before global freeze')
    chosen=frozen['units'][f'{mode}/{arm}/{task}/{fold}']
    if chosen['config']!=config(arm,task,fold,mode) or chosen['best']!=str(unit_path(arm,task,fold,mode=mode)/'best.pt'):raise ValueError('frozen estimator mismatch')
    return chosen

def unit(arm,task,fold,formal=False,physical_gpu=2):
    require_tmux()
    if physical_gpu not in ALLOWED_GPUS or os.environ.get('CUDA_VISIBLE_DEVICES')!=str(physical_gpu) or os.environ.get('WORLD_SIZE','1')!='1':raise ValueError('allocated physical GPU/world1 mismatch')
    torch.set_num_threads(2);check_prepared();cfg=config(arm,task,fold)
    folder=unit_path(arm,task,fold,formal);folder.mkdir(parents=True,exist_ok=False);write(folder/'config.json',cfg)
    state=dict(status='RUNNING',pid=os.getpid(),process=process_identity(os.getpid()),started=now(),
        epochs_started=0,completed_epochs=0,attempted_updates=0,completed_updates=0,train_graphs=0,
        validation_started=0,validation_graphs=0,test_graphs=0,test_started=0,physical_gpu=physical_gpu,gpu_uuid=os.environ.get('UA3M_PHYSICAL_GPU_UUID'),allowed_gpus=list(ALLOWED_GPUS),resource_revision=EXECUTION_REVISION,phase='initialize')
    def record():write(folder/'consumption.json',state)
    record();ds=None;started=time.monotonic()
    try:
        scaler=read(base_path(task,fold)/'scaler.json')
        if formal:
            frozen=read(OUT/'selection/final_frozen.json');selected=frozen_unit(frozen,arm,task,fold)
            saved=torch.load(selected['best'],map_location='cpu',weights_only=False)
            if any(saved[k]!=selected[k] for k in ('best_epoch','best_stage','scaler')):raise ValueError('frozen checkpoint/scaler mismatch')
            if scaler!=saved['scaler']:raise ValueError('BEST scaler mismatch')
            model=m.BaseOptFuse(arm,False)
            if saved.get('schema')!=m.FT_SCHEMA or saved.get('config')!=cfg:raise ValueError('FT checkpoint schema/config mismatch')
            m.strict_module_state(model,saved['state_dict']);best_epoch=saved['best_epoch'];best_stage=saved['best_stage'];del saved
        else:
            seed_all(42+fold)
            package=torch.load(parent_path(arm),map_location='cpu',weights_only=False)
            if package.get('arm')!=arm:raise ValueError('requested arm and PT endpoint differ')
            head=torch.load(head_path(task,fold),map_location='cpu',weights_only=False)
            model=m.initialize_ft(package,head);del package,head
        ds=d.Data(task,fold,formal,DOWN,SPLIT,prepared_root()/'inputs')
        if formal:
            if (selected['test_indices']!=ds.indices or selected['test_keys']!=ds.keys.tolist() or not np.array_equal(selected['test_targets'],ds.raw_targets)):raise ValueError('frozen TEST row/key/target mismatch')
        else:scaler_validate(ds.raw_targets,ds.train,scaler)
        ds.targets=transform(ds.raw_targets,scaler);write(folder/'scaler.json',scaler)
        model.to('cuda:0');validation=d.loader(ds,ds.validation)
        parameter_counts=dict(total=sum(p.numel() for p in model.parameters()),backbone=sum(p.numel() for n,p in model.named_parameters() if n.startswith(tuple(name+'.' for name in m.NATIVE))))
        write(folder/'parameters.json',parameter_counts)
        def evaluate():
            model.eval();truth=[];pred=[];gate_sum=np.zeros(3);gate_count=0
            with torch.inference_mode():
                for batch in validation:
                    batch=batch.to('cuda:0');p,weights,_=model.predict(batch)
                    if not torch.isfinite(p).all():raise FloatingPointError('nonfinite prediction')
                    truth.append(batch.y.view(-1).cpu().numpy());pred.append(p.cpu().numpy())
                    state['test_graphs' if formal else 'validation_graphs']+=len(p)
                    state['test_3d_graphs' if formal else 'validation_3d_graphs']=state.get('test_3d_graphs' if formal else 'validation_3d_graphs',0)+int(batch.availability[:,1].sum())
                    weights=weights.float().cpu().numpy();gate_sum+=weights.sum(axis=0);gate_count+=len(weights)
            y=transform(np.concatenate(truth),scaler,True);p=transform(np.concatenate(pred),scaler,True)
            return y,p,metrics(y,p),dict(mean_weights=(gate_sum/gate_count).tolist() if gate_count else None,samples=gate_count)
        if formal:
            state.update(phase='TEST',test_started=1);record()
            truth,prediction,result,diagnostics=evaluate()
            save_predictions(folder/'test_predictions.npz',ds,truth,prediction,best_epoch=best_epoch,best_stage=best_stage)
        else:
            train=d.loader(ds,ds.train,True,42+fold);expected=70*len(train)
            seed_all(42+fold);history=[];best_rmse=math.inf;train_tick=time.monotonic()
            for stage,epochs in (('head',10),('joint',60)):
                opt,groups=m.optimizer_ft(model,stage)
                scheduler=(torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=10,eta_min=3e-5) if stage=='head'
                    else torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt,T_0=10,T_mult=1,eta_min=1e-6))
                for stage_epoch in range(epochs):
                    tick=time.monotonic();state.update(phase=stage,epochs_started=state['epochs_started']+1,
                        epoch_open=True,epoch_expected_updates=len(train),epoch_expected_train=len(ds.train),epoch_expected_validation=len(ds.validation))
                    record();model.train();losses=[];rates=[g['lr'] for g in opt.param_groups]
                    for batch in train:
                        batch=batch.to('cuda:0');opt.zero_grad(set_to_none=True)
                        loss,diagnostics_train=model.training_loss(batch);output=batch.y.view(-1)
                        if not torch.isfinite(loss):raise FloatingPointError('nonfinite loss')
                        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True)
                        if state['attempted_updates']>=expected:raise RuntimeError('update allowance exhausted')
                        state['attempted_updates']+=1;state['train_graphs']+=len(output)
                        state['train_3d_graphs']=state.get('train_3d_graphs',0)+int(batch.availability[:,1].sum())
                        opt.step();state['completed_updates']+=1;losses.append(float(loss.detach()))
                    scheduler.step();state['validation_started']+=1
                    y,p,met,diag=evaluate();state['completed_epochs']+=1;epoch=state['completed_epochs']
                    if met['rmse']<best_rmse:
                        best_rmse=met['rmse'];best_epoch=epoch;best_stage=stage;result=met;diagnostics=diag
                        save_checkpoint(folder/'best.pt',dict(schema=m.FT_SCHEMA,architecture=m.CONFIG,config=cfg,
                            state_dict={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                            best_epoch=best_epoch,best_stage=best_stage,scaler=scaler,outer_test='NOT_RUN'))
                        save_predictions(folder/'validation_predictions.npz',ds,y,p,best_epoch=best_epoch,best_stage=best_stage)
                    rec=dict(epoch=epoch,stage=stage,stage_epoch=stage_epoch+1,train_loss=float(np.mean(losses)),
                        validation=met,training_components={k:float(v) for k,v in diagnostics_train.items()},learning_rates=rates,optimizer_groups=groups,modality_diagnostics=diag,
                        seconds=time.monotonic()-tick,elapsed_seconds=time.monotonic()-train_tick,gpu_peak_bytes=torch.cuda.max_memory_allocated())
                    history.append(rec);write(folder/'history.json',history);print(json.dumps(rec),flush=True)
                    state['epoch_open']=False;state['ft_epochs']=state['epochs_started'];state['ft_updates']=state['attempted_updates'];record()
            state['ft_epochs']=state['epochs_started'];state['ft_updates']=state['attempted_updates']
            if state['completed_epochs']!=70 or state['attempted_updates']!=expected or state['completed_updates']!=expected:
                raise ValueError('incomplete FT trajectory')
        write(folder/'metrics.json',dict(status='COMPLETE',config=cfg,best_epoch=best_epoch,best_stage=best_stage,scaler=scaler,
            **result,diagnostics=diagnostics,parameters=parameter_counts))
        state.update(status='COMPLETE',ended=now(),elapsed_seconds=time.monotonic()-started,gpu_peak_bytes=torch.cuda.max_memory_allocated(),
            best_epoch=best_epoch,best_stage=best_stage);record()
    except BaseException:
        ignore_termination()
        state['ft_epochs']=state['epochs_started'];state['ft_updates']=state['attempted_updates']
        state.update(status='FAILED',error=traceback.format_exc(),ended=now());record()
        write(folder/'failure.json',dict(error=state['error'],automatic_retry=False));raise
    finally:
        ignore_termination()
        if ds is not None:ds.close()



def aggregate(arm,formal=False,mode='FULL'):
    scores={task:[] for task in TASKS};units={}
    for task in TASKS:
        for fold in range(5):
            folder=unit_path(arm,task,fold,formal,mode);met=read(folder/'metrics.json');c=read(folder/'consumption.json');entry=read(SPLIT/f'{task}.json')['folds'][fold]
            if met.get('status')!='COMPLETE' or c.get('status')!='COMPLETE' or met['config']!=config(arm,task,fold,mode):raise ValueError('incomplete estimator')
            epochs=70 if mode=='FULL' else 30;updates=epochs*math.ceil(len(entry['train_indices'])/24)
            expected=dict(test_started=1,test_graphs=len(entry['test_indices'])) if formal else dict(epochs_started=epochs,completed_epochs=epochs,attempted_updates=updates,completed_updates=updates,train_graphs=epochs*len(entry['train_indices']),validation_started=epochs,validation_graphs=epochs*len(entry['validation_indices']))
            if any(c.get(k)!=v for k,v in expected.items()):raise ValueError('estimator scientific consumption mismatch')
            scores[task].append(met['r2']);ft=unit_path(arm,task,fold,mode=mode)
            units[f'{mode}/{arm}/{task}/{fold}']=dict(config=met['config'],best=str(ft/'best.pt'),best_epoch=met['best_epoch'],best_stage=met['best_stage'],scaler=met['scaler'],test_indices=sorted(entry['test_indices']),predictions=str(folder/('test_predictions.npz' if formal else 'validation_predictions.npz')))
    return dict(**summary(scores),units=units)


def nominate(validation,order=ARMS):
    if set(validation)!=set(order):raise ValueError('all four validation candidates required')
    finite([validation[a]['macro'] for a in order])
    # max() is stable; the second key makes the declared arm order explicit.
    return max(order,key=lambda a:(validation[a]['macro'],-order.index(a)))


def metrics(y,p):
    return parent.metrics(y,p)


def nominate(validation,order=ARMS):
    if set(validation)!=set(order):raise ValueError('complete validation candidates required')
    finite([validation[a]['macro'] for a in order]);return max(order,key=lambda a:(validation[a]['macro'],-order.index(a)))


def freeze():
    check_prepared()
    if (OUT/'selection/final_frozen.json').exists():raise FileExistsError('only one freeze')
    validation={a:aggregate(a) for a in ARMS};selected=nominate(validation)
    units={k:v for a in ARMS for k,v in validation[a]['units'].items()}
    if len(units)!=140:raise ValueError('all140 FULL estimators required')
    alignment={}
    for task in TASKS:
        for fold in range(5):
            ds=d.Data(task,fold,True,DOWN,SPLIT,PARENT/'inputs')
            try:
                alignment[f'{task}/{fold}']=dict(indices=ds.indices,keys=ds.keys.tolist(),targets=ds.raw_targets.tolist())
                for arm in ARMS:
                    u=units[f'FULL/{arm}/{task}/{fold}'];saved=torch.load(u['best'],map_location='cpu',weights_only=False)
                    if (saved['schema']!=m.FT_SCHEMA or saved['config']!=u['config']
                        or saved['scaler']!=u['scaler'] or saved['best_epoch']!=u['best_epoch']
                        or saved['best_stage']!=u['best_stage']):raise ValueError('BEST freeze identity mismatch')
                    u.update(test_keys=ds.keys.tolist(),test_targets=ds.raw_targets.tolist())
                    del saved
            finally:ds.close()
    exclusive(OUT/'selection/validation_selection.json',dict(plan=PLAN,revision=REVISION,validation=validation,
        PT_DIRECTION_SELECTED=selected,tie_order=list(ARMS),selection='equal seven-task validation macro, full precision'))
    exclusive(OUT/'selection/row_sample_key_alignment.json',alignment)
    frozen=dict(plan=PLAN,revision=REVISION,frozen=True,time=now(),PT_DIRECTION_SELECTED=selected,
        units=units,identity=read(OUT/'identity.json'),splits={task:read(SPLIT/f'{task}.json') for task in TASKS})
    exclusive(OUT/'selection/final_frozen.json',frozen)
    return frozen
