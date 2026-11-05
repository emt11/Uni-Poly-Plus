"""Independent head-p adapter. Parent scientific functions and all source roots are read only."""
import copy
import json
import math
import os
from pathlib import Path
import time
import traceback
import numpy as np
import torch
from src.modules import ua3d_base_opt_r6 as m
from src.dataset import ua3m_paperfuse as d
from src.training.ua3m_paperfuse import (read,write,exclusive,now,ignore_termination,seed_all,
    finite,metrics,process_identity,process_alive,scaler_validate,transform,summary,SOTA,
    require_tmux,save_checkpoint,save_predictions)
ROOT=Path(__file__).resolve().parents[2]
PLAN='UA3D-HEAD-DROPOUT-20261028-01';REVISION='r1';EXECUTION_REVISION='r1-head-p-only'
FT_SCHEMA='ua3d-head-dropout-best-r1'
OUT=ROOT/'results/ua3d_head_dropout_20261028/r1'
LOG=ROOT/'logs/ua3d_head_dropout_20261028/r1'
REF=ROOT/'results/ua3d_base_opt_20261026/r6/repairs/concurrency_r1'
REF_ID='F_PD_N1_F0';REFERENCE=.804675462276949
PARENT=ROOT/'results/ua3m_paperfuse_20261025/r1/repairs/gpu0123_restart1'
DOWN=ROOT/'data/processed/mcl_ph_r4_20260930/downstream_direct_01.staging'
SPLIT=ROOT/'data/splits/mips_paper5_outer'
SCALERS=ROOT/'results/ua2d_p10k_nofp_align_20261024/r1/finetune/D_A020'
TASKS=('eea','egb','egc','ei','eps','nc','xc');ALLOWED_GPUS=(2,)
SPEC=read(ROOT/'configs/ua3d_head_dropout_r1.json');LIMITS=SPEC['limits'];ORDER=list(SPEC['recipes'])

def resource_contract(kind):
    return dict(resource_revision=EXECUTION_REVISION,physical_gpus=[2],world_size=1,
        job_concurrency_limit=4 if kind in ('ft','test') else 1,pt_max_concurrency=0,
        ft_max_concurrency=4,test_max_concurrency=4,phase_exclusive=True,ft_test_workers_per_job=0)
def head_path(task,fold):return PARENT/'initialization/heads'/task/f'fold{fold}.pt'
def base_path(task,fold):return SCALERS/task/f'fold{fold}'
def prepared_root():return PARENT
def parent_path(arm):return REF/'pretrain'/REF_ID/'deploy.pt'
def registry():return read(OUT/'registry/recipes.json')
def get_recipe(arm):
    if arm not in ORDER:raise ValueError('outside fixed matrix')
    return registry()['recipes'][arm]
def unit_path(arm,task,fold,formal=False):
    root=REF if arm=='H50_BASE' else OUT
    return root/('formal' if formal else 'finetune')/(REF_ID if arm=='H50_BASE' else arm)/task/f'fold{fold}'
def config(arm,task,fold):
    if task not in TASKS or type(fold) is not int or fold not in range(5):raise ValueError('task/fold outside fixed matrix')
    if arm=='H50_BASE':return read(unit_path(arm,task,fold)/'config.json')
    return dict(plan=PLAN,revision=REVISION,recipe=get_recipe(arm),task=task,fold=fold,seed=42+fold,
        parent=str(parent_path(arm)),head=str(head_path(task,fold)),input_cache=str(PARENT/'inputs'),
        scaler_reference=str(base_path(task,fold)/'scaler.json'),model=m.CONFIG,world_size=1,batch=24,
        epochs=70,head_epochs=10,precision='FP32',workers=0,selection='strict original-unit validation RMSE',
        resources=resource_contract('ft'))
def update_plan(status,phase,detail):
    p=ROOT/'CURRENT_PLAN.md';text=p.read_text();a='<!-- UA3D_HEAD_DROPOUT_R1_STATUS_START -->';b='<!-- UA3D_HEAD_DROPOUT_R1_STATUS_END -->'
    left,rest=text.split(a,1);_,right=rest.split(b,1)
    p.write_text(left+a+'\n'+f'- 更新时间：{now()}；状态：{status}；阶段：{phase}。\n'
        +'- 用户2026-10-28明确授权本提示词执行；独立新周期，旧r6授权和结论保持。\n'
        +'- '+detail+'\n- GPU2 FT≤4／TEST≤4，阶段互斥；PT/prepare/生成0；最终CPU仅最后1次。\n'
        +f'- t0=2026-10-28T10:50:13+08:00；硬截止=2026-10-29T10:50:13+08:00；预算 `{OUT.relative_to(ROOT)}/budget.json`。\n'
        +'- independent_review=PENDING；共享开发评价；不自动替换BASE或归档为已验收。\n'+b+right)
    write(OUT/'status.json',dict(status=status,phase=phase,detail=detail,time=now(),independent_review='PENDING'))
def check_prepared():
    record=registry()
    if record.get('plan')!=PLAN or record.get('revision')!=REVISION:raise ValueError('registry identity mismatch')
    for task in TASKS:
        if read(SPLIT/f'{task}.json')!=record['splits'][task]:raise ValueError('split changed')
def gpu_contract():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='2' or os.environ.get('WORLD_SIZE')!='1':raise ValueError('GPU2/world1 mismatch')
    if torch.cuda.device_count()!=1:raise ValueError('exactly one visible CUDA device required')
    actual='GPU-'+str(torch.cuda.get_device_properties(0).uuid).removeprefix('GPU-')
    if actual!=os.environ.get('UA3M_PHYSICAL_GPU_UUID'):raise ValueError('physical/logical UUID mismatch')
    return dict(**resource_contract('ft'),physical_gpu=2,logical_device='cuda:0',gpu_uuid=actual)
def verify_runtime(model,cfg,eval_mode=False):
    ft=cfg['recipe']['ft']
    if ft!=dict(m.DEFAULT_FT,phead=SPEC['recipes'][cfg['recipe']['id']]):raise ValueError('non-head factor changed')
    if not isinstance(model.head[2],torch.nn.Dropout) or model.head[2].p!=ft['phead']:raise ValueError('head p mismatch')
    for name,expected in (('o8',0.),('encoder3d',0.),('fp_encoder',.2)):
        mods=[x for x in getattr(model,name).modules() if isinstance(x,torch.nn.Dropout)]
        if not mods or any(x.p!=expected for x in mods):raise ValueError('dropout mismatch '+name)
    if any(v.is_floating_point() and v.dtype!=torch.float32 for v in model.state_dict().values()):raise ValueError('non-FP32 model state')
    if eval_mode and any(x.training for x in model.modules()):raise ValueError('TEST must eval all modules')
def select(validation):
    if list(validation)!=ORDER:raise ValueError('incomplete validation pool')
    finite([v['macro'] for v in validation.values()])
    deltas={k:{task:validation[k]['tasks'][task]['mean']-validation['H50_BASE']['tasks'][task]['mean'] for task in TASKS} for k in ORDER}
    eligible=[k for k in ORDER if k=='H50_BASE' or all(x>=-.005 for x in deltas[k].values())]
    chosen=max(eligible,key=lambda k:(validation[k]['macro'],-ORDER.index(k)))
    return chosen,dict(eligible=eligible,task_deltas=deltas,ranking=sorted(ORDER,key=lambda k:(-validation[k]['macro'],ORDER.index(k))),rule='task delta>=-0.005; full precision macro; H50 then registration exact ties')
def frozen_unit(frozen,arm,task,fold):
    if arm=='H50_BASE':raise ValueError('alias cannot dispatch')
    if frozen.get('plan')!=PLAN or frozen.get('frozen') is not True:raise ValueError('early TEST')
    u=frozen['units'][f'{arm}/{task}/{fold}']
    if u['config']!=config(arm,task,fold) or u['best']!=str(unit_path(arm,task,fold)/'best.pt'):raise ValueError('frozen config mismatch')
    return u

def unit(arm,task,fold,formal=False,physical_gpu=2):
    require_tmux()
    if arm=='H50_BASE':raise ValueError('read-only alias cannot execute')
    if physical_gpu not in ALLOWED_GPUS or os.environ.get('CUDA_VISIBLE_DEVICES')!=str(physical_gpu) or os.environ.get('WORLD_SIZE','1')!='1':raise ValueError('allocated physical GPU/world1 mismatch')
    torch.set_num_threads(2);check_prepared();cfg=config(arm,task,fold)
    folder=unit_path(arm,task,fold,formal);folder.mkdir(parents=True,exist_ok=False);write(folder/'config.json',cfg)
    state=dict(status='RUNNING',pid=os.getpid(),process=process_identity(os.getpid()),started=now(),
        epochs_started=0,completed_epochs=0,attempted_updates=0,completed_updates=0,train_graphs=0,
        validation_started=0,validation_graphs=0,test_graphs=0,test_started=0,physical_gpu=physical_gpu,gpu_uuid=os.environ.get('UA3M_PHYSICAL_GPU_UUID'),allowed_gpus=list(ALLOWED_GPUS),resource_revision=EXECUTION_REVISION,phase='initialize')
    def record():write(folder/'consumption.json',state)
    record();ds=None;started=time.monotonic()
    try:
        state['actual_device']=gpu_contract();record()
        scaler=read(base_path(task,fold)/'scaler.json')
        if formal:
            frozen=read(OUT/'selection/final_frozen.json');selected=frozen_unit(frozen,arm,task,fold)
            saved=torch.load(selected['best'],map_location='cpu',weights_only=False)
            if any(saved[k]!=selected[k] for k in ('best_epoch','best_stage','scaler')):raise ValueError('frozen checkpoint/scaler mismatch')
            if scaler!=saved['scaler']:raise ValueError('BEST scaler mismatch')
            model=m.BaseOptFuse(False,ft=cfg['recipe']['ft'])
            if saved.get('schema')!=FT_SCHEMA or saved.get('config')!=cfg:raise ValueError('FT checkpoint schema/config mismatch')
            m.strict_module_state(model,saved['state_dict']);best_epoch=saved['best_epoch'];best_stage=saved['best_stage'];del saved
        else:
            seed_all(42+fold)
            package=torch.load(parent_path(arm),map_location='cpu',weights_only=False)
            head=torch.load(head_path(task,fold),map_location='cpu',weights_only=False)
            model=m.initialize_ft(package,head,cfg['recipe']);del package,head
        verify_runtime(model,cfg)
        ds=d.Data(task,fold,formal,DOWN,SPLIT,prepared_root()/'inputs')
        if formal:
            if (selected['test_indices']!=ds.indices or selected['test_keys']!=ds.keys.tolist()):raise ValueError('frozen TEST row/key/target mismatch')
        else:scaler_validate(ds.raw_targets,ds.train,scaler)
        ds.targets=transform(ds.raw_targets,scaler);write(folder/'scaler.json',scaler)
        if formal:
            old_identity=read(unit_path('H50_BASE',task,fold,True)/'identity_alignment.json')
            for key,value in (('indices',ds.indices),('sample_keys',ds.keys.tolist()),('raw_targets',ds.raw_targets.tolist()),('transformed_targets',ds.targets.tolist()),('scaler',scaler)):
                if old_identity[key]!=value:raise ValueError('paired raw target/scaler identity mismatch '+key)
        model.to('cuda:0');validation=d.loader(ds,ds.validation)
        parameter_counts=dict(total=sum(p.numel() for p in model.parameters()),backbone=sum(p.numel() for n,p in model.named_parameters() if n.startswith(tuple(name+'.' for name in m.NATIVE))))
        write(folder/'parameters.json',parameter_counts)
        def evaluate():
            model.eval();verify_runtime(model,cfg,True);state['evaluation_runtime']=model.runtime_contract();truth=[];pred=[];diag_sum={};gate_count=0
            with torch.inference_mode():
                for batch in validation:
                    batch=batch.to('cuda:0');p,weights,_=model.predict(batch)
                    if not torch.isfinite(p).all():raise FloatingPointError('nonfinite prediction')
                    truth.append(batch.y.view(-1).cpu().numpy());pred.append(p.cpu().numpy())
                    state['test_graphs' if formal else 'validation_graphs']+=len(p)
                    state['test_3d_graphs' if formal else 'validation_3d_graphs']=state.get('test_3d_graphs' if formal else 'validation_3d_graphs',0)+int(batch.availability[:,1].sum())
                    for key,value in model.last_diagnostics.items():diag_sum[key]=diag_sum.get(key,0)+value.detach().cpu().numpy()*len(p)
                    gate_count+=len(p)
            y=transform(np.concatenate(truth),scaler,True);p=transform(np.concatenate(pred),scaler,True)
            if not formal:
                with np.load(unit_path('H50_BASE',task,fold)/'validation_predictions.npz',allow_pickle=False) as ref:
                    if not np.array_equal(ref['indices'],np.array(ds.indices)[ds.validation]) or not np.array_equal(ref['sample_keys'],ds.keys[ds.validation]) or not np.array_equal(ref['y_true'],y):raise ValueError('paired validation row/key/target mismatch')
            return y,p,metrics(y,p),dict(values={k:np.asarray(v/gate_count).tolist() for k,v in diag_sum.items()},samples=gate_count)
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
                    else torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt,T_0=10,T_mult=1,eta_min=1e-6)
                    if cfg['recipe']['ft']['scheduler']=='restart'
                    else torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=60,eta_min=1e-6))
                for stage_epoch in range(epochs):
                    tick=time.monotonic();state.update(phase=stage,epochs_started=state['epochs_started']+1,
                        epoch_open=True,epoch_expected_updates=len(train),epoch_expected_train=len(ds.train),epoch_expected_validation=len(ds.validation))
                    record();model.train();losses=[];component_sum={};training_diag={};train_count=0;rates=[g['lr'] for g in opt.param_groups]
                    verify_runtime(model,cfg);runtime=model.runtime_contract()
                    for batch in train:
                        batch=batch.to('cuda:0');opt.zero_grad(set_to_none=True)
                        loss,diagnostics_train=model.training_loss(batch);output=batch.y.view(-1)
                        size=len(output);train_count+=size
                        for key,value in diagnostics_train.items():component_sum[key]=component_sum.get(key,0.)+float(value)*size
                        for key,value in model.last_diagnostics.items():training_diag[key]=training_diag.get(key,0)+value.detach().cpu().numpy()*size
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
                        save_checkpoint(folder/'best.pt',dict(schema=FT_SCHEMA,architecture=m.CONFIG,config=cfg,
                            state_dict={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                            best_epoch=best_epoch,best_stage=best_stage,scaler=scaler,outer_test='NOT_RUN'))
                        save_predictions(folder/'validation_predictions.npz',ds,y,p,best_epoch=best_epoch,best_stage=best_stage)
                    rec=dict(epoch=epoch,stage=stage,stage_epoch=stage_epoch+1,train_loss=float(np.mean(losses)),
                        validation=met,training_components={k:v/train_count for k,v in component_sum.items()},training_modality_diagnostics={k:np.asarray(v/train_count).tolist() for k,v in training_diag.items()},runtime_contract=runtime,learning_rates=rates,optimizer_groups=groups,modality_diagnostics=diag,
                        seconds=time.monotonic()-tick,elapsed_seconds=time.monotonic()-train_tick,gpu_peak_bytes=torch.cuda.max_memory_allocated())
                    history.append(rec);write(folder/'history.json',history);print(json.dumps(rec),flush=True)
                    state['epoch_open']=False;state['ft_epochs']=state['epochs_started'];state['ft_updates']=state['attempted_updates'];record()
            state['ft_epochs']=state['epochs_started'];state['ft_updates']=state['attempted_updates']
            if state['completed_epochs']!=70 or state['attempted_updates']!=expected or state['completed_updates']!=expected:
                raise ValueError('incomplete FT trajectory')
        if formal:
            with np.load(unit_path('H50_BASE',task,fold,True)/'test_predictions.npz',allow_pickle=False) as base:
                for key,value in (('indices',np.array(ds.indices)),('sample_keys',ds.keys),('y_true',truth)):
                    if not np.array_equal(base[key],value):raise ValueError('paired TEST row/key/target mismatch')
            write(folder/'identity_alignment.json',dict(indices=ds.indices,sample_keys=ds.keys.tolist(),raw_targets=ds.raw_targets.tolist(),
                transformed_targets=ds.targets.tolist(),frozen_source=selected['target_contract'],scaler=scaler,matched=True))
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


def aggregate(recipe_id,formal=False):
    scores={task:[] for task in TASKS};units={}
    for task in TASKS:
        for fold in range(5):
            folder=unit_path(recipe_id,task,fold,formal);met=read(folder/'metrics.json');c=read(folder/'consumption.json')
            entry=read(SPLIT/f'{task}.json')['folds'][fold]
            if met.get('status')!='COMPLETE' or c.get('status')!='COMPLETE' or met['config']!=config(recipe_id,task,fold):raise ValueError('incomplete estimator')
            updates=70*math.ceil(len(entry['train_indices'])/24)
            expected=dict(test_started=1,test_graphs=len(entry['test_indices'])) if formal else dict(epochs_started=70,
                completed_epochs=70,attempted_updates=updates,completed_updates=updates,train_graphs=70*len(entry['train_indices']),
                validation_started=70,validation_graphs=70*len(entry['validation_indices']),epoch_open=False)
            if any(c.get(k)!=v for k,v in expected.items()):raise ValueError('estimator scientific consumption mismatch')
            scores[task].append(met['r2']);ft=unit_path(recipe_id,task,fold)
            units[f'{recipe_id}/{task}/{fold}']=dict(config=met['config'],best=str(ft/'best.pt'),
                best_epoch=met['best_epoch'],best_stage=met['best_stage'],scaler=met['scaler'],
                scaler_path=str(ft/'scaler.json'),test_indices=sorted(entry['test_indices']),
                predictions=str(folder/('test_predictions.npz' if formal else 'validation_predictions.npz')))
    return dict(**summary(scores),units=units)

def register_sources():
    """Normal registration, no inference, no new initialization or prepare."""
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('registration must hide GPUs')
    source=read(REF/'registry/recipes.json')['recipes'][REF_ID]
    if source['ft']!=m.DEFAULT_FT or source['pt_id']!=REF_ID:raise ValueError('F recipe identity mismatch')
    pt=m.pt_config('P_CURRENT_SHARED',.1,0.)
    if source['pt']!=pt:raise ValueError('F PT recipe mismatch')
    prior=read(REF/'final_review/results.json')
    if (prior['formal'][REF_ID]['macro']!=REFERENCE or prior['validation'][REF_ID]['macro']!=SPEC['reference_validation']
        or prior['FINAL']!='E_P_SYM_SHARED' or prior['performance_pass'] is not False
        or read(REF/'controller_exit.json')['exit']!=0):raise ValueError('reference evidence mismatch')
    package=torch.load(parent_path('H50_BASE'),map_location='cpu',weights_only=False)
    m.validate_package(package,pt)
    if package.get('pt_id')!=REF_ID:raise ValueError('PT source id mismatch')
    for name,value in package['state_dict'].items():
        if not torch.is_tensor(value) or (value.is_floating_point() and (value.dtype!=torch.float32 or not torch.isfinite(value).all())):
            raise ValueError('PT dtype/finite mismatch '+name)
    del package
    cache=d.InputCache(PARENT/'inputs')
    try:rows=copy.deepcopy(cache.rows)
    finally:cache.close()
    d.validate_missing_inventory(read(PARENT/'inputs/geometry_missing.json'))
    original=read(PARENT/'inputs/original_splits.json');splits={};totals=dict(ft_updates=0,train_graphs=0,validation_graphs=0,test_graphs=0)
    aliases={};recipes={}
    for i,(arm,p) in enumerate(SPEC['recipes'].items()):
        recipes[arm]=dict(id=arm,pt_id=REF_ID,pt=pt,ft=dict(m.DEFAULT_FT,phead=p),registration=i,
            source_recipe=REF_ID,control='H50_BASE',alias=arm=='H50_BASE')
    for task in TASKS:
        split=read(SPLIT/f'{task}.json');splits[task]=split
        if split!=original[task] or split.get('validation_is_test') is not False:raise ValueError('split identity mismatch')
        for fold,e in enumerate(split['folds']):
            sets=[set(e[k]) for k in ('train_indices','validation_indices','test_indices')]
            if len(split['folds'])!=5 or any(sets[i]&sets[j] for i in range(3) for j in range(i)):raise ValueError('split overlap/count')
            totals['ft_updates']+=70*math.ceil(len(e['train_indices'])/24)
            totals['train_graphs']+=70*len(e['train_indices']);totals['validation_graphs']+=70*len(e['validation_indices']);totals['test_graphs']+=len(e['test_indices'])
            ft=unit_path('H50_BASE',task,fold);formal=unit_path('H50_BASE',task,fold,True)
            cfg=read(ft/'config.json');scaler=read(base_path(task,fold)/'scaler.json')
            expected=dict(plan='UA3D-BASE-OPT-20261026-01',revision='r6',recipe=source,task=task,fold=fold,seed=42+fold,
                parent=str(parent_path('H50_BASE')),head=str(head_path(task,fold)),input_cache=str(PARENT/'inputs'),
                scaler_reference=str(base_path(task,fold)/'scaler.json'),model=m.CONFIG,world_size=1,batch=24,epochs=70,head_epochs=10,
                precision='FP32',workers=0,selection='strict original-unit validation RMSE')
            if any(cfg.get(k)!=v for k,v in expected.items()):raise ValueError('control actual config differs')
            if cfg['resources']['physical_gpus']!=[2] or cfg['resources']['world_size']!=1:raise ValueError('control resources mismatch')
            for folder in (ft,formal):
                met=read(folder/'metrics.json');usage=read(folder/'consumption.json')
                if met['status']!='COMPLETE' or usage['status']!='COMPLETE' or met['config']!=cfg or read(folder/'config.json')!=cfg or met['scaler']!=scaler:
                    raise ValueError('control source incomplete/config/scaler mismatch')
            hist=read(ft/'history.json')
            if len(hist)!=70 or any(h['runtime_contract']['dropout']['head.2']['p']!=.5 for h in hist):raise ValueError('control runtime p mismatch')
            head=torch.load(head_path(task,fold),map_location='cpu',weights_only=False)
            small=m.native.property_head(.5);m.strict_module_state(small,head);del small,head
            saved=torch.load(ft/'best.pt',map_location='cpu',weights_only=False)
            if saved['schema']!=m.FT_SCHEMA or saved['config']!=cfg or saved['scaler']!=scaler:raise ValueError('control BEST mismatch')
            model=m.BaseOptFuse(False,ft=cfg['recipe']['ft']);m.strict_module_state(model,saved['state_dict']);del model,saved
            ds=d.Data(task,fold,False,DOWN,SPLIT,PARENT/'inputs')
            try:
                scaler_validate(ds.raw_targets,ds.train,scaler)
                with np.load(ft/'validation_predictions.npz',allow_pickle=False) as pred:
                    if not np.array_equal(pred['indices'],np.array(ds.indices)[ds.validation]) or not np.array_equal(pred['sample_keys'],ds.keys[ds.validation]):raise ValueError('control validation identity mismatch')
            finally:ds.close()
            aliases[f'H50_BASE/{task}/{fold}']=dict(ft=str(ft),formal=str(formal),config=cfg,head=str(head_path(task,fold)),scaler=str(base_path(task,fold)/'scaler.json'),new_ft=0,new_test=0)
    expected=dict(ft_updates=56000,train_graphs=1315020,validation_graphs=329980,test_graphs=5875)
    if totals!=expected:raise ValueError('per-recipe consumption mismatch '+repr(totals))
    exclusive(OUT/'registry/recipes.json',dict(plan=PLAN,revision=REVISION,recipes=recipes,aliases=aliases,splits=splits,rows=rows,per_recipe=totals,
        reference_root=str(REF),reference=REFERENCE,source_pt=str(parent_path('H50_BASE')),time=now(),test_targets_accessed=False))
    if aggregate('H50_BASE')['macro']!=SPEC['reference_validation'] or aggregate('H50_BASE',True)['macro']!=REFERENCE:raise ValueError('control aggregate mismatch')
    return dict(status='COMPLETE',time=now(),pt_new=0,prepare_new=0,control_aliases=35,source_pt=str(parent_path('H50_BASE')),no_forward=True)

def freeze():
    check_prepared();record=registry();validation={k:aggregate(k) for k in ORDER};chosen,rule=select(validation)
    units={key:u for value in validation.values() for key,u in value['units'].items()}
    target_contract=dict(source=str(DOWN),label='original property label',transform='fixed train-only scaler; inverse for metrics',
        truth_access='only inside corresponding formal TEST',precision='FP32')
    for task in TASKS:
        rows={x['row']:x for x in record['rows'][task]}
        for fold in range(5):
            indices=sorted(record['splits'][task]['folds'][fold]['test_indices']);keys=[rows[i]['key'] for i in indices]
            for arm in ORDER:
                u=units[f'{arm}/{task}/{fold}'];saved=torch.load(u['best'],map_location='cpu',weights_only=False)
                schema=m.FT_SCHEMA if arm=='H50_BASE' else FT_SCHEMA
                if (saved.get('schema')!=schema or saved['config']!=u['config'] or saved['scaler']!=u['scaler']
                    or saved['best_epoch']!=u['best_epoch'] or saved['best_stage']!=u['best_stage']):raise ValueError('frozen BEST mismatch')
                model=m.BaseOptFuse(False,ft=u['config']['recipe']['ft']);m.strict_module_state(model,saved['state_dict'])
                proxy=copy.deepcopy(u['config']);proxy['recipe']['id']=arm;verify_runtime(model,proxy)
                u.update(test_keys=keys,target_contract=target_contract,pt_source=str(parent_path(arm)),head_source=str(head_path(task,fold)),
                    p=SPEC['recipes'][arm],alias=arm=='H50_BASE',split_source=str(SPLIT/f'{task}.json'),
                    row_key_source=str(PARENT/'inputs/rows.json'),scaler_source=str(base_path(task,fold)/'scaler.json'),
                    formal_alias=str(unit_path(arm,task,fold,True)) if arm=='H50_BASE' else None)
                del model,saved
    if len(units)!=210:raise ValueError('frozen inventory incomplete')
    value=dict(plan=PLAN,revision=REVISION,frozen=True,time=now(),FINAL=chosen,FINAL_p=SPEC['recipes'][chosen],validation=validation,
        units=units,selection=rule,registry=record,test_targets_read=False,shared_development_evaluation=True)
    exclusive(OUT/'selection/final_frozen.json',value)
    return value
