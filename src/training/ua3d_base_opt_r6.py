"""r6 explicit source paths, fixed recipe registry, validation selection and one freeze."""
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
from src.training import ua3d_spatial as parent
from src.training.ua3m_paperfuse import (read,write,exclusive,now,ignore_termination,seed_all,
    finite,metrics,process_identity,process_alive,scaler_validate,transform,summary,SOTA,
    require_tmux,save_checkpoint,pt_learning_rate,window_objective,save_predictions)
ROOT=Path(__file__).resolve().parents[2]
PLAN='UA3D-BASE-OPT-20261026-01';REVISION='r6'
EXECUTION_REVISION=os.environ.get('UA3D_BASE_OPT_RESOURCE_REVISION','r6-once')
SOURCE=ROOT/'results/ua3d_base_opt_20261026/r6'
CONTINUATION=EXECUTION_REVISION=='gpu2-pt2-fttest4-c1'
if EXECUTION_REVISION not in ('r6-once','gpu2-pt2-fttest4-c1'):raise ValueError('unauthorized resource revision')
OUT=SOURCE/'repairs/concurrency_r1' if CONTINUATION else SOURCE
LOG=ROOT/'logs/ua3d_base_opt_20261026/r6'/('repairs/concurrency_r1' if CONTINUATION else '')
if CONTINUATION and os.environ.get('UA3D_BASE_OPT_EXECUTION_ROOT')!=str(OUT):raise ValueError('explicit continuation root mismatch')
PARENT=parent.PARENT;DOWN=parent.DOWN;SPLIT=parent.SPLIT;PT_CACHE=parent.PT_CACHE;MAP_ROOT=parent.MAP_ROOT
INITIAL=SOURCE/'initialization/common_step0.pt'
TASKS=m.TASKS;ALLOWED_GPUS=(2,);REFERENCE=.795244745287872
LIMITS=dict(initialize_starts=1,pt_starts=8,pt_updates=80000,pt_microbatches=480000,pt_presentations=80640000,
    ft_starts=805,ft_epochs=56350,ft_updates=1288000,test_starts=805,test_graphs=135125,cpu_suites=1)
if CONTINUATION:LIMITS.update(ft_starts=806,ft_epochs=56420,ft_updates=1288700)
ZERO_COST=dict(fp_generation=0,geometry_generation=0,kernel_fit=0,additional_seeds=0,output_fusion=0)
HIST_PT=ROOT/'results/ua3d_spatial_20261025/r4/pretrain/BASE/deploy.pt'
REFERENCE_FILE=ROOT/'results/ua3d_spatial_20261025/r4/final_review/results.json'
BASELINE='A_L_A8_B8'

def identity_path():return SOURCE/'identity.json'
def resource_contract(kind):
    limit=2 if kind=='pt' else 4 if kind in ('ft','test') else 1
    if not CONTINUATION:limit=1
    return dict(resource_revision=EXECUTION_REVISION,physical_gpus=[2],world_size=1,
        job_concurrency_limit=limit,pt_max_concurrency=2 if CONTINUATION else 1,
        ft_max_concurrency=4 if CONTINUATION else 1,test_max_concurrency=4 if CONTINUATION else 1,
        phase_exclusive=True,pt_workers_per_job=4,ft_test_workers_per_job=0)

def head_path(task,fold):return PARENT/'initialization/heads'/task/f'fold{fold}.pt'
def base_path(task,fold):return parent.base_path(task,fold)
def prepared_root():return PARENT
def registry():return read(OUT/'registry/recipes.json')
def get_recipe(recipe_id):
    record=registry();canonical=record['aliases'].get(recipe_id,recipe_id)
    if canonical not in record['recipes']:raise ValueError('unregistered recipe')
    return record['recipes'][canonical]
def parent_path(recipe_id):
    recipe=get_recipe(recipe_id)
    return HIST_PT if recipe['pt_id']=='HIST_R4_BASE' else OUT/'pretrain'/recipe['pt_id']/'deploy.pt'
def unit_path(recipe_id,task,fold,formal=False):
    return OUT/('formal' if formal else 'finetune')/recipe_id/task/f'fold{fold}'
def config(recipe_id,task,fold):
    recipe=get_recipe(recipe_id)
    if recipe['id']!=recipe_id or task not in TASKS or type(fold) is not int or fold not in range(5):raise ValueError('outside r6 matrix')
    return dict(plan=PLAN,revision=REVISION,recipe=recipe,task=task,fold=fold,seed=42+fold,
        parent=str(parent_path(recipe_id)),head=str(head_path(task,fold)),input_cache=str(PARENT/'inputs'),
        scaler_reference=str(base_path(task,fold)/'scaler.json'),model=m.CONFIG,world_size=1,batch=24,
        epochs=70,head_epochs=10,precision='FP32',workers=0,selection='strict original-unit validation RMSE',
        **(dict(resources=resource_contract('ft')) if CONTINUATION else {}))

def update_plan(status,phase,detail):
    path=ROOT/'CURRENT_PLAN.md';text=path.read_text()
    start='<!-- UA3D_BASE_OPT_R6_STATUS_START -->';end='<!-- UA3D_BASE_OPT_R6_STATUS_END -->'
    if CONTINUATION:
        start='<!-- UA3D_BASE_OPT_R6_CONCURRENCY_R1_STATUS_START -->';end='<!-- UA3D_BASE_OPT_R6_CONCURRENCY_R1_STATUS_END -->'
        if start not in text:text=text.replace('## 12.',start+'\n'+end+'\n\n## 12.',1)
    if start not in text or end not in text:raise ValueError('r6 status writer markers absent')
    block=(start+'\n'+f'- 更新时间：{now()}；状态：**{status}**；阶段：{phase}。\n'
        +('- 用户明确授权gpu2-pt2-fttest4-c1一次接续；GPU2 PT最多2路、FT/TEST最多4路，阶段互斥；保留原用户停止记录，不恢复r5。\n' if CONTINUATION else '- 用户2026-10-26已明确授权r6 StageA—G完整周期；GPU2单路串行，不恢复r5。\n')
        +f'- {detail}\n- 预算：`{OUT.relative_to(ROOT)}/budget.json`；日志：`{LOG.relative_to(ROOT)}`。\n'
        +'- independent_review=PENDING；共享开发评价，不自动部署、替换BASE或归档为已验收。\n'+end)
    left,rest=text.split(start,1);_,right=rest.split(end,1)
    path.write_text(left+block+right)
    write(OUT/'status.json',dict(status=status,phase=phase,detail=detail,time=now(),independent_review='PENDING'))

def check_inputs():
    pt=read(PT_CACHE/'metadata.json');mapping=read(MAP_ROOT/'metadata.json')
    if (pt.get('schema')!='ua2d_prepared_templates_r4' or pt.get('status')!='COMPLETE' or pt.get('train_rows')!=911391
        or mapping.get('status')!='COMPLETE' or mapping.get('records')!=910756
        or mapping.get('eligibility_semantics')!='logical_slot_not_source_row'):raise ValueError('PT cache/schema mismatch')
    cache=d.InputCache(PARENT/'inputs')
    try:
        d.validate_missing_inventory(read(PARENT/'inputs/geometry_missing.json'))
        rows=copy.deepcopy(cache.rows)
    finally:cache.close()
    original=read(PARENT/'inputs/original_splits.json')
    totals=dict(ft_updates=0,test_graphs=0)
    for task in TASKS:
        split=read(SPLIT/f'{task}.json')
        if split!=original[task] or split.get('validation_is_test') is not False:raise ValueError('shared split mismatch')
        for fold in range(5):
            entry=split['folds'][fold]
            sets=[set(entry[k]) for k in ('train_indices','validation_indices','test_indices')]
            if any(sets[i]&sets[j] for i in range(3) for j in range(i)):raise ValueError('split overlap')
            if not head_path(task,fold).is_file() or not (base_path(task,fold)/'scaler.json').is_file():raise FileNotFoundError('head/scaler absent')
            totals['ft_updates']+=70*math.ceil(len(entry['train_indices'])/24);totals['test_graphs']+=len(entry['test_indices'])
    if totals!=dict(ft_updates=56000,test_graphs=5875):raise ValueError('fixed per-recipe budget differs')
    ref=read(REFERENCE_FILE)['formal']['FULL']['BASE']
    if ref['macro']!=REFERENCE:raise ValueError('specified BASE reference differs')
    return dict(pt=pt,mapping=mapping,splits=original,rows=rows,per_recipe=totals,
        reference=ref,reference_source=str(REFERENCE_FILE),test_targets_accessed=False)

def check_prepared():
    identity=read(identity_path())
    if identity['plan']!=PLAN or identity['revision']!=REVISION or not INITIAL.is_file():raise ValueError('r6 initialization absent/mismatch')
    for task in TASKS:
        if read(SPLIT/f'{task}.json')!=identity['inputs']['splits'][task]:raise ValueError('split identity changed')

def gpu_contract():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='2' or os.environ.get('WORLD_SIZE')!='1':raise ValueError('GPU2/world1 env mismatch')
    if torch.cuda.device_count()!=1:raise ValueError('must expose exactly one CUDA device')
    actual='GPU-'+str(torch.cuda.get_device_properties(0).uuid).removeprefix('GPU-')
    expected=os.environ.get('UA3M_PHYSICAL_GPU_UUID')
    if actual!=expected:raise ValueError(f'physical/logical UUID mismatch: {actual} vs {expected}')
    return dict(**resource_contract(os.environ.get('UA3D_BASE_OPT_JOB_KIND','ft')),
        physical_gpu=2,logical_device='cuda:0',gpu_uuid=actual)

def initialize_cycle():
    if CONTINUATION:raise RuntimeError('continuation must reuse parent initialization; initialize/prepare forbidden')
    require_tmux()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU initialization must hide GPU')
    torch.set_num_threads(2);inputs=check_inputs();seed_all(42)
    model=m.BaseOptFuse(True)
    sources=[parent.INITIAL,parent.INITIAL.parent/'task_heads/V1.pt',PARENT/'initialization/TMOL3M_modal.pt']
    common,heads,modal=[torch.load(p,map_location='cpu',weights_only=False) for p in sources]
    evidence=m.native.load_initial(model,common,heads,modal)
    folder=OUT/'initialization';folder.mkdir(exist_ok=False)
    save_checkpoint(INITIAL,dict(plan=PLAN,revision=REVISION,step=0,seed=42,config=m.CONFIG,
        state_dict=model.state_dict(),sources=list(map(str,sources)),initialization=evidence))
    # QFP is a value-identical, independently parameterized copy of the loaded Q2.
    save_checkpoint(folder/'QFP_step0.pt',dict(source='common_step0.Q2',state_dict=copy.deepcopy(model.Q2.state_dict())))
    package=torch.load(HIST_PT,map_location='cpu',weights_only=False);m.spatial.validate_package(package)
    if package['arm']!='BASE':raise ValueError('r4 BASE endpoint mismatch')
    temp=m.BaseOptFuse(False);m.ft_state(package,temp)
    for task in TASKS:
        for fold in range(5):
            head=torch.load(head_path(task,fold),map_location='cpu',weights_only=False)
            m.strict_module_state(temp.head,head)
            ds=d.Data(task,fold,False,DOWN,SPLIT,PARENT/'inputs')
            try:scaler_validate(ds.raw_targets,ds.train,read(base_path(task,fold)/'scaler.json'))
            finally:ds.close()
    exclusive(OUT/'identity.json',dict(plan=PLAN,revision=REVISION,inputs=inputs,model=m.CONFIG,
        common_step0=str(INITIAL),sources=list(map(str,sources)),reference=REFERENCE,time=now(),
        physical_gpu=2,world_size=1,workers_pt=4,workers_ft_test=0,initialization=evidence))
    exclusive(OUT/'registry/recipes.json',dict(plan=PLAN,revision=REVISION,recipes={},aliases={},slots=[],selections={},pt={}))
    write(folder/'initialization.json',dict(**evidence,sources=list(map(str,sources)),head_units=35,
        train_only_scalers=35,test_targets_accessed=False,qfp='deepcopy loaded Q2'))

def signature(recipe):return dict(pt_id=recipe['pt_id'],pt=recipe['pt'],ft=recipe['ft'])
def register_recipe(slot,stage,pt_id='HIST_R4_BASE',pt=None,ft=None,parent_id=None,control=None):
    record=registry()
    if any(x['slot']==slot for x in record['slots']):raise ValueError('duplicate recipe slot')
    proposal=dict(id=slot,stage=stage,pt_id=pt_id,pt=copy.deepcopy(pt),ft=copy.deepcopy(ft if ft is not None else m.DEFAULT_FT),
        parent=parent_id,control=control,registration=len(record['slots']))
    canonical=next((k for k,v in record['recipes'].items() if signature(v)==signature(proposal)),None)
    if canonical is None:
        if len(record['recipes'])>=23:raise ValueError('unique recipe limit')
        canonical=slot;record['recipes'][slot]=proposal
    else:record['aliases'][slot]=canonical
    record['slots'].append(dict(slot=slot,stage=stage,canonical=canonical,alias=canonical!=slot,parent=parent_id,control=control))
    write(OUT/'registry/recipes.json',record)
    return canonical,canonical==slot

def register_pt(pt_id,pt):
    record=registry()
    if pt_id in record['pt'] or len(record['pt'])>=8:raise ValueError('PT registry conflict/limit')
    record['pt'][pt_id]=copy.deepcopy(pt);write(OUT/'registry/recipes.json',record)

def select(validation,order,control):
    if set(validation)!=set(order) or control not in validation:raise ValueError('selection pool/control mismatch')
    finite([v['macro'] for v in validation.values()])
    eligible=[key for key in order if all(validation[key]['tasks'][task]['mean']-validation[control]['tasks'][task]['mean']>=-.005 for task in TASKS)]
    priority=[control]+[key for key in order if key!=control]
    chosen=max(eligible,key=lambda key:(validation[key]['macro'],-priority.index(key)))
    return chosen,dict(control=control,order=order,eligible=eligible,rejected=[key for key in order if key not in eligible],
        rule='per-task mean delta >= -0.005; then full precision macro; control-first exact ties',
        task_deltas={key:{task:validation[key]['tasks'][task]['mean']-validation[control]['tasks'][task]['mean'] for task in TASKS} for key in order})

def select_stage(stage,order,control):
    order=list(dict.fromkeys(order));validation={key:aggregate(key) for key in order}
    chosen,rule=select(validation,order,control);record=registry()
    if stage in record['selections']:raise ValueError('stage already selected')
    record['selections'][stage]=dict(selected=chosen,**rule,time=now(),validation=validation)
    write(OUT/'registry/recipes.json',record);write(OUT/'selection'/f'{stage}.json',record['selections'][stage])
    return chosen

def frozen_unit(frozen,recipe_id,task,fold):
    if frozen.get('plan')!=PLAN or frozen.get('revision')!=REVISION or frozen.get('frozen') is not True:raise ValueError('TEST before global freeze')
    chosen=frozen['units'][f'{recipe_id}/{task}/{fold}']
    if chosen['config']!=config(recipe_id,task,fold) or chosen['best']!=str(unit_path(recipe_id,task,fold)/'best.pt'):raise ValueError('frozen estimator mismatch')
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
        state['actual_device']=gpu_contract();record()
        scaler=read(base_path(task,fold)/'scaler.json')
        if formal:
            frozen=read(OUT/'selection/final_frozen.json');selected=frozen_unit(frozen,arm,task,fold)
            saved=torch.load(selected['best'],map_location='cpu',weights_only=False)
            if any(saved[k]!=selected[k] for k in ('best_epoch','best_stage','scaler')):raise ValueError('frozen checkpoint/scaler mismatch')
            if scaler!=saved['scaler']:raise ValueError('BEST scaler mismatch')
            model=m.BaseOptFuse(False,ft=cfg['recipe']['ft'])
            if saved.get('schema')!=m.FT_SCHEMA or saved.get('config')!=cfg:raise ValueError('FT checkpoint schema/config mismatch')
            m.strict_module_state(model,saved['state_dict']);best_epoch=saved['best_epoch'];best_stage=saved['best_stage'];del saved
        else:
            seed_all(42+fold)
            package=torch.load(parent_path(arm),map_location='cpu',weights_only=False)
            head=torch.load(head_path(task,fold),map_location='cpu',weights_only=False)
            model=m.initialize_ft(package,head,cfg['recipe']);del package,head
        ds=d.Data(task,fold,formal,DOWN,SPLIT,prepared_root()/'inputs')
        if formal:
            if (selected['test_indices']!=ds.indices or selected['test_keys']!=ds.keys.tolist()):raise ValueError('frozen TEST row/key/target mismatch')
        else:scaler_validate(ds.raw_targets,ds.train,scaler)
        ds.targets=transform(ds.raw_targets,scaler);write(folder/'scaler.json',scaler)
        model.to('cuda:0');validation=d.loader(ds,ds.validation)
        parameter_counts=dict(total=sum(p.numel() for p in model.parameters()),backbone=sum(p.numel() for n,p in model.named_parameters() if n.startswith(tuple(name+'.' for name in m.NATIVE))))
        write(folder/'parameters.json',parameter_counts)
        def evaluate():
            model.eval();truth=[];pred=[];diag_sum={};gate_count=0
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
                    runtime=model.runtime_contract()
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
                        save_checkpoint(folder/'best.pt',dict(schema=m.FT_SCHEMA,architecture=m.CONFIG,config=cfg,
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

def freeze():
    check_prepared();record=registry()
    if set(record['selections'])!={'A','B','C','D','E','F','G_PARENT','G'}:raise ValueError('incomplete A-G validation')
    if (OUT/'selection/final_frozen.json').exists():raise FileExistsError('only one freeze')
    order=list(record['recipes']);validation={key:aggregate(key) for key in order}
    chosen,rule=select(validation,order,BASELINE)
    units={key:value for item in validation.values() for key,value in item['units'].items()}
    if len(units)!=35*len(order):raise ValueError('incomplete frozen estimator inventory')
    rows=read(identity_path())['inputs']['rows'];alignment={}
    target_contract=dict(source=str(DOWN),label='original property label',transform='fixed train-only scaler; inverse for metrics',
        truth_access='only inside the corresponding formal TEST unit',precision='FP32 forward; original-unit finite metrics')
    for task in TASKS:
        lookup={x['row']:x for x in rows[task]}
        for fold in range(5):
            entry=read(SPLIT/f'{task}.json')['folds'][fold];indices=sorted(entry['test_indices'])
            keys=[lookup[row]['key'] for row in indices]
            alignment[f'{task}/{fold}']=dict(indices=indices,keys=keys,target_contract=target_contract,test_truth_read=False)
            for recipe_id in order:
                u=units[f'{recipe_id}/{task}/{fold}'];saved=torch.load(u['best'],map_location='cpu',weights_only=False)
                if (saved['schema']!=m.FT_SCHEMA or saved['config']!=u['config'] or saved['scaler']!=u['scaler']
                    or saved['best_epoch']!=u['best_epoch'] or saved['best_stage']!=u['best_stage']):raise ValueError('BEST freeze identity mismatch')
                # Strict state checks without a forward; no new inference or TEST truth access.
                model=m.BaseOptFuse(False,ft=u['config']['recipe']['ft']);m.strict_module_state(model,saved['state_dict'])
                u.update(test_keys=keys,target_contract=target_contract,pt_source=str(parent_path(recipe_id)),
                    head_source=str(head_path(task,fold)))
                del saved,model
    exclusive(OUT/'selection/validation_selection.json',dict(plan=PLAN,revision=REVISION,FINAL=chosen,validation=validation,**rule))
    exclusive(OUT/'selection/row_sample_key_alignment.json',alignment)
    frozen=dict(plan=PLAN,revision=REVISION,frozen=True,time=now(),FINAL=chosen,units=units,
        registry=record,identity=read(identity_path()),initialization_source=str(identity_path()),target_contract=target_contract,
        splits={task:read(SPLIT/f'{task}.json') for task in TASKS},test_targets_read=False)
    exclusive(OUT/'selection/final_frozen.json',frozen)
    return frozen
