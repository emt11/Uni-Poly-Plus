"""Independent FT balance training, source manifests and single freeze. Parent scientific functions and all source roots are read only."""
import copy
import json
import math
import os
from pathlib import Path
import time
import traceback
import numpy as np
import torch
from src.modules import ua3d_ft_balance as m
from src.modules import ua3d_base_opt_r6 as original_model
from src.dataset import ua3m_paperfuse as d
from src.training.ua3m_paperfuse import (read,write,exclusive,now,ignore_termination,seed_all,
    finite,metrics,process_identity,process_alive,scaler_validate,transform,summary,SOTA,
    require_tmux,save_checkpoint,save_predictions)
ROOT=Path(__file__).resolve().parents[2]
PLAN='UA3D-FT-BALANCE-20261028-01';REVISION='r1';EXECUTION_REVISION='r1-ft-balance'
FT_SCHEMA='ua3d-ft-balance-best-r1'
OUT=ROOT/'results/ua3d_ft_balance_20261028/r1'
LOG=ROOT/'logs/ua3d_ft_balance_20261028/r1'
REF=ROOT/'results/ua3d_base_opt_20261026/r6/repairs/concurrency_r1'
REF_ID='F_PD_N1_F0';REFERENCE=.804675462276949
PARENT=ROOT/'results/ua3m_paperfuse_20261025/r1/repairs/gpu0123_restart1'
DOWN=ROOT/'data/processed/mcl_ph_r4_20260930/downstream_direct_01.staging'
SPLIT=ROOT/'data/splits/mips_paper5_outer'
SCALERS=ROOT/'results/ua2d_p10k_nofp_align_20261024/r1/finetune/D_A020'
REFT_REF=ROOT/'results/ua3d_base_reft_20261028/r1'
TASKS=('eea','egb','egc','ei','eps','nc','xc');ALLOWED_GPUS=(2,)
SPEC=read(ROOT/'configs/ua3d_ft_balance_r1.json');LIMITS=SPEC['limits'];ARMS=list(SPEC['recipes']);ALIASES=SPEC['aliases'];ORDER=ALIASES+ARMS

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
    if arm=='BASE_ORIGINAL':root,key=REF,REF_ID
    elif arm=='BASE_REFT_HISTORY':root,key=REFT_REF,'BASE_REFT'
    elif arm in ARMS:root,key=OUT,arm
    else:raise ValueError('unregistered estimator identity')
    return root/('formal' if formal else 'finetune')/key/task/f'fold{fold}'
def config(arm,task,fold):
    if task not in TASKS or type(fold) is not int or fold not in range(5):raise ValueError('task/fold outside fixed matrix')
    if arm in ALIASES:return read(unit_path(arm,task,fold)/'config.json')
    return dict(plan=PLAN,revision=REVISION,recipe=get_recipe(arm),task=task,fold=fold,seed=42+fold,
        parent=str(parent_path(arm)),head=str(head_path(task,fold)),input_cache=str(PARENT/'inputs'),
        scaler_reference=str(base_path(task,fold)/'scaler.json'),model=m.CONFIG,world_size=1,batch=24,
        epochs=70,head_epochs=10,precision='FP32',workers=0,selection='strict original-unit validation RMSE',
        resources=resource_contract('ft'),loss_schedule=get_recipe(arm)['loss_schedule'],lr_policy=get_recipe(arm)['lr_policy'])
def update_plan(status,phase,detail):
    p=ROOT/'CURRENT_PLAN.md';text=p.read_text();a='<!-- UA3D_FT_BALANCE_R1_STATUS_START -->';b='<!-- UA3D_FT_BALANCE_R1_STATUS_END -->'
    left,rest=text.split(a,1);_,right=rest.split(b,1)
    p.write_text(left+a+'\n'+f'- 更新时间：{now()}；状态：{status}；阶段：{phase}。\n'
        +'- 用户2026-10-28明确授权FT-BALANCE/r1完整固定周期；DSH执行；旧周期全部保留。\n'
        +'- '+detail+'\n- GPU2 FT≤4／TEST≤4，阶段互斥；PT/prepare/生成0；最终CPU仅最后1次。\n'
        +f'- t0=2026-10-28T16:37:39+08:00；硬截止=2026-10-29T16:37:39+08:00；预算 `{OUT.relative_to(ROOT)}/budget.json`。\n'
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
    if ft!=m.DEFAULT_FT:raise ValueError('non-head factor changed')
    if (model.loss_schedule,model.lr_policy)!=(cfg['loss_schedule'],cfg['lr_policy']):raise ValueError('runtime strategy/config mismatch')
    if not math.isclose(model.current_align,m.align_coefficient(cfg['loss_schedule'],model.global_epoch),rel_tol=0,abs_tol=1e-12):raise ValueError('ALIGN schedule mismatch')
    if not isinstance(model.head[2],torch.nn.Dropout) or model.head[2].p!=ft['phead']:raise ValueError('head p mismatch')
    for name,expected in (('o8',0.),('encoder3d',0.),('fp_encoder',.2)):
        mods=[x for x in getattr(model,name).modules() if isinstance(x,torch.nn.Dropout)]
        if not mods or any(x.p!=expected for x in mods):raise ValueError('dropout mismatch '+name)
    if any(v.is_floating_point() and v.dtype!=torch.float32 for v in model.state_dict().values()):raise ValueError('non-FP32 model state')
    if eval_mode and any(x.training for x in model.modules()):raise ValueError('TEST must eval all modules')
def select(validation):
    if list(validation)!=ORDER:raise ValueError('incomplete four-arm validation')
    finite([v['macro'] for v in validation.values()]);controls=('BASE_ORIGINAL','C0_BASE')
    threshold=max(validation[k]['macro'] for k in controls)+.005
    candidates=ARMS[1:];decisions={};eligible=[]
    for arm in candidates:
        deltas={control:{task:validation[arm]['tasks'][task]['mean']-validation[control]['tasks'][task]['mean'] for task in TASKS} for control in controls}
        protected=all(delta>=-.005 for values in deltas.values() for delta in values.values())
        ok=validation[arm]['macro']>=threshold and protected
        decisions[arm]=dict(macro_pass=validation[arm]['macro']>=threshold,task_protection_pass=protected,eligible=ok,task_deltas=deltas)
        if ok:eligible.append(arm)
    chosen=max(eligible,key=lambda k:(validation[k]['macro'],-candidates.index(k))) if eligible else 'BASE_KEEP'
    return chosen,dict(eligible=eligible,candidate_order=candidates,macro_threshold=threshold,decisions=decisions,
        rule='macro >= max(original BASE,C0)+0.005; each task >= both controls-0.005; full precision ranking; A,B,AB exact ties; otherwise BASE_KEEP')
def frozen_unit(frozen,arm,task,fold):
    if arm in ALIASES:raise ValueError('alias cannot dispatch')
    if frozen.get('plan')!=PLAN or frozen.get('frozen') is not True:raise ValueError('early TEST')
    u=frozen['units'][f'{arm}/{task}/{fold}']
    if u['config']!=config(arm,task,fold) or u['best']!=str(unit_path(arm,task,fold)/'best.pt'):raise ValueError('frozen config mismatch')
    return u

def unit(arm,task,fold,formal=False,physical_gpu=2):
    require_tmux()
    if arm in ALIASES:raise ValueError('read-only alias cannot execute')
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
            model=m.BalanceFuse(cfg['recipe']['ft'],cfg['loss_schedule'],cfg['lr_policy'])
            if saved.get('schema')!=FT_SCHEMA or saved.get('config')!=cfg:raise ValueError('FT checkpoint schema/config mismatch')
            m.strict_module_state(model,saved['state_dict']);best_epoch=saved['best_epoch'];best_stage=saved['best_stage'];saved_strategy=saved['strategy_metadata'];del saved
        else:
            seed_all(42+fold)
            package=torch.load(parent_path(arm),map_location='cpu',weights_only=False)
            head=torch.load(head_path(task,fold),map_location='cpu',weights_only=False)
            model=m.initialize_ft(package,head,cfg['recipe']);del package,head
        if formal:
            model.set_epoch(best_epoch);model.stage=best_stage
            if saved_strategy!=selected['strategy_metadata']:raise ValueError('frozen BEST strategy mismatch')
        verify_runtime(model,cfg)
        ds=d.Data(task,fold,formal,DOWN,SPLIT,prepared_root()/'inputs')
        if formal:
            if (selected['test_indices']!=ds.indices or selected['test_keys']!=ds.keys.tolist()):raise ValueError('frozen TEST row/key/target mismatch')
        else:scaler_validate(ds.raw_targets,ds.train,scaler)
        ds.targets=transform(ds.raw_targets,scaler);write(folder/'scaler.json',scaler)
        if formal:
            old_identity=read(unit_path('BASE_ORIGINAL',task,fold,True)/'identity_alignment.json')
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
                with np.load(unit_path('BASE_ORIGINAL',task,fold)/'validation_predictions.npz',allow_pickle=False) as ref:
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
                    model.set_epoch(state['completed_epochs']+1)
                    tick=time.monotonic();state.update(phase=stage,epochs_started=state['epochs_started']+1,
                        epoch_open=True,epoch_expected_updates=len(train),epoch_expected_train=len(ds.train),epoch_expected_validation=len(ds.validation))
                    record();model.train();losses=[];component_sum={};training_diag={};train_count=0;rates=[g['lr'] for g in opt.param_groups]
                    expected_rates=m.expected_learning_rates(stage,stage_epoch,cfg['lr_policy'])
                    if len(rates)!=len(expected_rates) or any(not math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-12) for a,b in zip(rates,expected_rates)):raise ValueError('actual scheduler LR mismatch')
                    verify_runtime(model,cfg);runtime=model.runtime_contract();actual_groups=m.optimizer_snapshot(model,opt)
                    if stage=='head' and any(getattr(model,n).training for n in m.NATIVE):raise ValueError('native mode in head stage')
                    if stage=='joint' and any(not getattr(model,n).training for n in m.NATIVE):raise ValueError('native mode in joint stage')
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
                            best_epoch=best_epoch,best_stage=best_stage,scaler=scaler,outer_test='NOT_RUN',strategy_metadata=dict(loss_schedule=cfg['loss_schedule'],lr_policy=cfg['lr_policy'],align_coefficient=model.current_align,global_epoch=epoch,optimizer_groups=actual_groups,learning_rates=rates)))
                        save_predictions(folder/'validation_predictions.npz',ds,y,p,best_epoch=best_epoch,best_stage=best_stage)
                    rec=dict(epoch=epoch,stage=stage,stage_epoch=stage_epoch+1,train_loss=float(np.mean(losses)),
                        validation=met,training_components={k:v/train_count for k,v in component_sum.items()},training_modality_diagnostics={k:np.asarray(v/train_count).tolist() for k,v in training_diag.items()},runtime_contract=runtime,learning_rates=rates,optimizer_groups=actual_groups,align_coefficient=model.current_align,entropy_coefficient=.08,modality_diagnostics=diag,
                        seconds=time.monotonic()-tick,elapsed_seconds=time.monotonic()-train_tick,gpu_peak_bytes=torch.cuda.max_memory_allocated())
                    history.append(rec);write(folder/'history.json',history);print(json.dumps(rec),flush=True)
                    state['epoch_open']=False;state['ft_epochs']=state['epochs_started'];state['ft_updates']=state['attempted_updates'];record()
            state['ft_epochs']=state['epochs_started'];state['ft_updates']=state['attempted_updates']
            if state['completed_epochs']!=70 or state['attempted_updates']!=expected or state['completed_updates']!=expected:
                raise ValueError('incomplete FT trajectory')
        if formal:
            with np.load(unit_path('BASE_ORIGINAL',task,fold,True)/'test_predictions.npz',allow_pickle=False) as base:
                for key,value in (('indices',np.array(ds.indices)),('sample_keys',ds.keys),('y_true',truth)):
                    if not np.array_equal(base[key],value):raise ValueError('paired TEST row/key/target mismatch')
            write(folder/'identity_alignment.json',dict(indices=ds.indices,sample_keys=ds.keys.tolist(),raw_targets=ds.raw_targets.tolist(),
                transformed_targets=ds.targets.tolist(),frozen_source=selected['target_contract'],scaler=scaler,matched=True))
        write(folder/'metrics.json',dict(status='COMPLETE',config=cfg,best_epoch=best_epoch,best_stage=best_stage,scaler=scaler,
            loss_schedule=cfg['loss_schedule'],lr_policy=cfg['lr_policy'],**result,diagnostics=diagnostics,parameters=parameter_counts))
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
    """One normal JSON/path registration; model/checkpoint loading occurs only in charged FT."""
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('registration must hide GPU')
    source=read(REF/'registry/recipes.json')['recipes'][REF_ID]
    pt=m.pt_config('P_CURRENT_SHARED',.1,0.)
    if source['pt_id']!=REF_ID or source['pt']!=pt or source['ft']!=m.DEFAULT_FT:raise ValueError('original F recipe differs')
    for root in (REF,REFT_REF):
        if read(root/'controller_exit.json')['exit']!=0:raise ValueError('historical source nonzero controller exit')
        b=read(root/'budget.json')
        key=REF_ID if root==REF else 'BASE_REFT'
        selected=[j for j in b['jobs'] if j.get('arm')==key and j['kind'] in ('ft','test')]
        if b['budget_violations'] or len(selected)!=70 or any(j['exit']!=0 for j in selected):raise ValueError('historical source estimator jobs incomplete or failed')
    prior=read(REF/'final_review/results.json');history=read(REFT_REF/'final_review/results.json')
    if prior['formal'][REF_ID]['macro']!=REFERENCE or prior['validation'][REF_ID]['macro']!=SPEC['reference_validation']:
        raise ValueError('original reference metrics mismatch')
    if prior['FINAL']!='E_P_SYM_SHARED' or prior['performance_pass'] is not False:raise ValueError('original r6 conclusions changed')
    if history['formal']['BASE_REFT']['macro']!=.8000156133174479 or history['FINAL']!='BASE_REFT':raise ValueError('historical REFT identity mismatch')
    if not parent_path('C0_BASE').is_file():raise FileNotFoundError('F PT endpoint absent')
    cache=d.InputCache(PARENT/'inputs')
    try:rows=copy.deepcopy(cache.rows)
    finally:cache.close()
    d.validate_missing_inventory(read(PARENT/'inputs/geometry_missing.json'))
    original=read(PARENT/'inputs/original_splits.json');splits={};totals=dict(ft_updates=0,train_graphs=0,validation_graphs=0,test_graphs=0)
    recipes={}
    for i,arm in enumerate(ORDER):
        policies=SPEC['recipes'][arm] if arm in ARMS else dict(loss_schedule='constant',lr_policy='original')
        recipes[arm]=dict(id=arm,pt_id=REF_ID,pt=pt,ft=copy.deepcopy(m.DEFAULT_FT),registration=i,source_recipe=REF_ID,
            control='C0_BASE',alias=arm in ALIASES,**policies)
    aliases={}
    for task in TASKS:
        split=read(SPLIT/f'{task}.json');splits[task]=split
        if split!=original[task] or split.get('validation_is_test') is not False or len(split['folds'])!=5:raise ValueError('split identity/count mismatch')
        for fold,e in enumerate(split['folds']):
            sets=[set(e[k]) for k in ('train_indices','validation_indices','test_indices')]
            if any(sets[i]&sets[j] for i in range(3) for j in range(i)):raise ValueError('split overlap')
            totals['ft_updates']+=70*math.ceil(len(e['train_indices'])/24);totals['train_graphs']+=70*len(e['train_indices'])
            totals['validation_graphs']+=70*len(e['validation_indices']);totals['test_graphs']+=len(e['test_indices'])
            scaler=read(base_path(task,fold)/'scaler.json')
            if not head_path(task,fold).is_file():raise FileNotFoundError('original fold head missing')
            for alias in ALIASES:
                ft=unit_path(alias,task,fold);formal=unit_path(alias,task,fold,True);cfg=read(ft/'config.json')
                required=dict(task=task,fold=fold,seed=42+fold,parent=str(parent_path(alias)),head=str(head_path(task,fold)),
                    input_cache=str(PARENT/'inputs'),scaler_reference=str(base_path(task,fold)/'scaler.json'),
                    model=m.CONFIG,world_size=1,batch=24,epochs=70,head_epochs=10,precision='FP32',workers=0)
                if any(cfg.get(k)!=v for k,v in required.items()) or cfg['recipe']['ft']!=m.DEFAULT_FT or cfg['recipe']['pt']!=pt or cfg['recipe']['pt_id']!=REF_ID:
                    raise ValueError('historical alias scientific config mismatch')
                for folder in (ft,formal):
                    met=read(folder/'metrics.json');usage=read(folder/'consumption.json')
                    if met['status']!='COMPLETE' or usage['status']!='COMPLETE' or met['config']!=cfg or read(folder/'config.json')!=cfg or met['scaler']!=scaler:
                        raise ValueError('historical alias incomplete/config/scaler mismatch')
                if not (ft/'best.pt').is_file():raise FileNotFoundError('alias BEST absent')
                aliases[f'{alias}/{task}/{fold}']=dict(ft=str(ft),formal=str(formal),config=cfg,new_ft=0,new_test=0,scaler_source=str(base_path(task,fold)/'scaler.json'),head_source=str(head_path(task,fold)))
    if totals!=dict(ft_updates=56000,train_graphs=1315020,validation_graphs=329980,test_graphs=5875):raise ValueError('per-arm budget mismatch')
    exclusive(OUT/'registry/recipes.json',dict(plan=PLAN,revision=REVISION,recipes=recipes,aliases=aliases,splits=splits,rows=rows,
        per_recipe=totals,reference=REFERENCE,reference_root=str(REF),history_root=str(REFT_REF),source_pt=str(parent_path('C0_BASE')),time=now(),model_constructed=False,checkpoint_loaded=False,test_targets_accessed=False))
    for alias,expected in (('BASE_ORIGINAL',REFERENCE),('BASE_REFT_HISTORY',.8000156133174479)):
        if aggregate(alias,True)['macro']!=expected:raise ValueError('historical alias aggregate mismatch')
    return dict(status='COMPLETE',time=now(),pt_new=0,prepare_new=0,alias_units=70,model_constructed=False,checkpoint_loaded=False,no_forward=True)

def freeze():
    check_prepared()
    if (OUT/'selection/final_frozen.json').exists():raise FileExistsError('only one immutable freeze')
    record=registry();validation={k:aggregate(k) for k in ORDER};chosen,rule=select(validation)
    units={key:u for value in validation.values() for key,u in value['units'].items()}
    target_contract=dict(source=str(DOWN),label='original property label',transform='fixed train-only scaler; inverse for metrics',truth_access='only inside corresponding formal TEST',precision='FP32')
    for task in TASKS:
        rows={x['row']:x for x in record['rows'][task]}
        for fold in range(5):
            indices=sorted(record['splits'][task]['folds'][fold]['test_indices']);keys=[rows[i]['key'] for i in indices]
            for arm in ORDER:
                u=units[f'{arm}/{task}/{fold}'];strategy=None
                if arm in ARMS:
                    saved=torch.load(u['best'],map_location='cpu',weights_only=False)
                    if (saved.get('schema')!=FT_SCHEMA or saved['config']!=u['config'] or saved['scaler']!=u['scaler']
                        or saved['best_epoch']!=u['best_epoch'] or saved['best_stage']!=u['best_stage']):raise ValueError('BEST freeze metadata mismatch')
                    strategy=saved['strategy_metadata']
                    if strategy['loss_schedule']!=u['config']['loss_schedule'] or strategy['lr_policy']!=u['config']['lr_policy'] or not math.isclose(strategy['align_coefficient'],m.align_coefficient(u['config']['loss_schedule'],u['best_epoch']),rel_tol=0,abs_tol=1e-12):raise ValueError('BEST strategy mismatch')
                    del saved
                u.update(test_keys=keys,target_contract=target_contract,pt_source=str(parent_path(arm)),head_source=str(head_path(task,fold)),
                    p=.5,alias=arm in ALIASES,strategy_metadata=strategy,
                    fusion_initialization=dict(source='src/modules/ua3m_paperfuse.py:PaperFuse.__init__',seed=m.native.SEED,policy='original fork_rng, identical construction order'),
                    split_source=str(SPLIT/f'{task}.json'),row_key_source=str(PARENT/'inputs/rows.json'),scaler_source=str(base_path(task,fold)/'scaler.json'),
                    formal_alias=str(unit_path(arm,task,fold,True)) if arm in ALIASES else None)
    if len(units)!=210:raise ValueError('140 new BEST + 70 historical alias estimators required')
    selection=dict(plan=PLAN,revision=REVISION,time=now(),FINAL=chosen,validation=validation,**rule)
    exclusive(OUT/'selection/validation_selection.json',selection)
    value=dict(plan=PLAN,revision=REVISION,frozen=True,time=now(),FINAL=chosen,FINAL_p=.5,validation=validation,units=units,selection=rule,
        registry=record,test_targets_read=False,shared_development_evaluation=True,
        BASE_KEEP=dict(source='BASE_ORIGINAL',reference=REFERENCE,root=str(REF),new_cost=0),new_estimators=140,alias_estimators=70)
    exclusive(OUT/'selection/final_frozen.json',value)
    return value
