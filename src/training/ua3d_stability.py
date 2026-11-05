"""Independent exact-BASE retraining adapter. Parent scientific functions and all source roots are read only."""
import copy
import json
import math
import shutil
import os
from pathlib import Path
import time
import traceback
import numpy as np
import torch
from src.modules import ua3d_stability as m
from src.dataset import ua3m_paperfuse as d
from src.training.ua3m_paperfuse import (read,write,exclusive,now,ignore_termination,seed_all,
    finite,metrics,process_identity,process_alive,scaler_validate,transform,summary,SOTA,
    require_tmux,save_checkpoint,save_predictions)
ROOT=Path(__file__).resolve().parents[2]
PLAN='UA3D-STABILITY-20261028-01';REVISION='r1';EXECUTION_REVISION='r1-strict-stability'
FT_SCHEMA='ua3d-stability-best-r1'
OUT=ROOT/'results/ua3d_stability_20261028/r1'
LOG=ROOT/'logs/ua3d_stability_20261028/r1'
REF=ROOT/'results/ua3d_base_opt_20261026/r6/repairs/concurrency_r1'
REF_ID='F_PD_N1_F0';REFERENCE=.804675462276949
PARENT=ROOT/'results/ua3m_paperfuse_20261025/r1/repairs/gpu0123_restart1'
DOWN=ROOT/'data/processed/mcl_ph_r4_20260930/downstream_direct_01.staging'
SPLIT=ROOT/'data/splits/mips_paper5_outer'
SCALERS=ROOT/'results/ua2d_p10k_nofp_align_20261024/r1/finetune/D_A020'
TASKS=('eea','egb','egc','ei','eps','nc','xc');ALLOWED_GPUS=(2,)
SPEC=read(ROOT/'configs/ua3d_stability_r1.json');LIMITS=SPEC['limits'];RUNS=list(SPEC['runs']);CANONICAL=SPEC['canonical'];DIAGNOSTICS=SPEC['diagnostics'];ORDER=['BASE_ORIGINAL']+RUNS

def resource_contract(kind):
    return dict(resource_revision=EXECUTION_REVISION,physical_gpus=[2],world_size=1,
        job_concurrency_limit=4 if kind in ('ft','test') else 1,diagnostic_max_concurrency=1,pt_max_concurrency=0,
        ft_max_concurrency=4,test_max_concurrency=4,phase_exclusive=True,ft_test_workers_per_job=0)
def head_path(task,fold):return PARENT/'initialization/heads'/task/f'fold{fold}.pt'
def base_path(task,fold):return SCALERS/task/f'fold{fold}'
def prepared_root():return PARENT
def parent_path(arm):return REF/'pretrain'/REF_ID/'deploy.pt'
def registry():return read(OUT/'registry/recipes.json')
def get_recipe(arm):
    if arm not in ORDER:raise ValueError('outside fixed matrix')
    return registry()['recipes'][arm]
def unit_path(run,task,fold,formal=False):
    if run=='BASE_ORIGINAL':return REF/('formal' if formal else 'finetune')/REF_ID/task/f'fold{fold}'
    if run not in RUNS:raise ValueError('unregistered run')
    return OUT/('formal' if formal else 'finetune')/'DET_BASE'/run/task/f'fold{fold}'
def config(run,task,fold):
    if task not in TASKS or type(fold) is not int or fold not in range(5):raise ValueError('task/fold outside fixed matrix')
    if run=='BASE_ORIGINAL':return read(unit_path(run,task,fold)/'config.json')
    recipe=get_recipe(run)
    return dict(plan=PLAN,revision=REVISION,recipe=recipe,task=task,fold=fold,seed=SPEC['runs'][run]+fold,seed_base=SPEC['runs'][run],run_id=run,numerical_policy='DET',
        parent=str(parent_path(run)),head=str(head_path(task,fold)),input_cache=str(PARENT/'inputs'),scaler_reference=str(base_path(task,fold)/'scaler.json'),
        model=m.CONFIG,world_size=1,batch=24,epochs=70,head_epochs=10,precision='FP32',workers=0,selection='strict original-unit validation RMSE',resources=resource_contract('ft'))
def update_plan(status,phase,detail):
    p=ROOT/'CURRENT_PLAN.md';text=p.read_text();a='<!-- UA3D_STABILITY_R1_STATUS_START -->';b='<!-- UA3D_STABILITY_R1_STATUS_END -->'
    left,rest=text.split(a,1);_,right=rest.split(b,1)
    p.write_text(left+a+'\n'+f'- 更新时间：{now()}；状态：{status}；阶段：{phase}。\n'
        +'- 用户2026-10-28明确授权STABILITY/r1完整固定周期；DSH执行；所有旧周期保持。\n'
        +'- '+detail+'\n- GPU2诊断≤1／FT≤4／TEST≤4，阶段互斥；PT/prepare/生成0；最终CPU仅最后1次。\n'
        +f'- t0=2026-10-28T21:42:21+08:00；硬截止=2026-10-29T21:42:21+08:00；预算 `{OUT.relative_to(ROOT)}/budget.json`。\n'
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
    return dict(**resource_contract(os.environ.get('UA3D_STABILITY_JOB_KIND','ft')),physical_gpu=2,logical_device='cuda:0',gpu_uuid=actual)
def verify_runtime(model,cfg,eval_mode=False):
    ft=cfg['recipe']['ft']
    if ft!=m.DEFAULT_FT:raise ValueError('non-head factor changed')
    m.check_det()
    if cfg['seed']!=SPEC['runs'][cfg['run_id']]+cfg['fold']:raise ValueError('training seed mismatch')
    if not isinstance(model.head[2],torch.nn.Dropout) or model.head[2].p!=ft['phead']:raise ValueError('head p mismatch')
    for name,expected in (('o8',0.),('encoder3d',0.),('fp_encoder',.2)):
        mods=[x for x in getattr(model,name).modules() if isinstance(x,torch.nn.Dropout)]
        if not mods or any(x.p!=expected for x in mods):raise ValueError('dropout mismatch '+name)
    if any(v.is_floating_point() and v.dtype!=torch.float32 for v in model.state_dict().values()):raise ValueError('non-FP32 model state')
    if eval_mode and any(x.training for x in model.modules()):raise ValueError('TEST must eval all modules')
def frozen_unit(frozen,arm,task,fold):
    if arm not in CANONICAL:raise ValueError('only canonical runs may TEST')
    if frozen.get('plan')!=PLAN or frozen.get('frozen') is not True:raise ValueError('early TEST')
    u=frozen['units'][f'{arm}/{task}/{fold}']
    if u['config']!=config(arm,task,fold) or u['best']!=str(unit_path(arm,task,fold)/'best.pt'):raise ValueError('frozen config mismatch')
    if u.get('inference_allowed') is not True:raise ValueError('frozen inference denied')
    return u

def unit(arm,task,fold,formal=False,physical_gpu=2,policy_setup=None):
    require_tmux()
    if arm=='BASE_ORIGINAL':raise ValueError('read-only alias cannot execute')
    if physical_gpu not in ALLOWED_GPUS or os.environ.get('CUDA_VISIBLE_DEVICES')!=str(physical_gpu) or os.environ.get('WORLD_SIZE','1')!='1':raise ValueError('allocated physical GPU/world1 mismatch')
    if policy_setup is None or policy_setup['policy']!='DET' or not policy_setup['configured_before_cuda']:raise ValueError('FT/TEST strict setup must precede CUDA')
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
            m.strict_module_state(model,saved['state_dict']);best_epoch=saved['best_epoch'];best_stage=saved['best_stage'];model.stage=best_stage;del saved
        else:
            seed_all(cfg['seed'])
            package=torch.load(parent_path(arm),map_location='cpu',weights_only=False)
            head=torch.load(head_path(task,fold),map_location='cpu',weights_only=False)
            model=m.initialize_ft(package,head,cfg['recipe']);del package,head
        verify_runtime(model,cfg)
        ds=TraceData(task,fold,formal,DOWN,SPLIT,prepared_root()/'inputs')
        if formal:
            if (selected['test_indices']!=ds.indices or selected['test_keys']!=ds.keys.tolist()):raise ValueError('frozen TEST row/key/target mismatch')
        else:scaler_validate(ds.raw_targets,ds.train,scaler)
        ds.targets=transform(ds.raw_targets,scaler);write(folder/'scaler.json',scaler)
        if formal:
            old_identity=read(unit_path('BASE_ORIGINAL',task,fold,True)/'identity_alignment.json')
            for key,value in (('indices',ds.indices),('sample_keys',ds.keys.tolist()),('raw_targets',ds.raw_targets.tolist()),('transformed_targets',ds.targets.tolist()),('scaler',scaler)):
                if old_identity[key]!=value:raise ValueError('paired raw target/scaler identity mismatch '+key)
        write(folder/'environment.json',dict(versions=m.versions(),setup=policy_setup,numerical=m.check_det(),physical=state['actual_device']))
        model.to('cuda:0');validation=d.loader(ds,ds.validation)
        parameter_counts=dict(total=sum(p.numel() for p in model.parameters()),backbone=sum(p.numel() for n,p in model.named_parameters() if n.startswith(tuple(name+'.' for name in m.NATIVE))))
        write(folder/'parameters.json',parameter_counts)
        def evaluate():
            ds.record_training=False
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
            train=d.loader(ds,ds.train,True,cfg['seed']);expected=70*len(train)
            seed_all(cfg['seed']);history=[];best_rmse=math.inf;train_tick=time.monotonic()
            for stage,epochs in (('head',10),('joint',60)):
                opt,groups=m.optimizer_ft(model,stage)
                scheduler=(torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=10,eta_min=3e-5) if stage=='head'
                    else torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt,T_0=10,T_mult=1,eta_min=1e-6)
                    if cfg['recipe']['ft']['scheduler']=='restart'
                    else torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=60,eta_min=1e-6))
                for stage_epoch in range(epochs):
                    tick=time.monotonic();state.update(phase=stage,epochs_started=state['epochs_started']+1,
                        epoch_open=True,epoch_expected_updates=len(train),epoch_expected_train=len(ds.train),epoch_expected_validation=len(ds.validation))
                    record();ds.training_rows=[];ds.record_training=True;batch_boundaries=[];clip_norms=[];model.train();losses=[];component_sum={};training_diag={};train_count=0;rates=[g['lr'] for g in opt.param_groups]
                    verify_runtime(model,cfg);runtime=model.runtime_contract()
                    for batch in train:
                        batch=batch.to('cuda:0');opt.zero_grad(set_to_none=True)
                        loss,diagnostics_train=model.training_loss(batch);output=batch.y.view(-1)
                        size=len(output);train_count+=size
                        for key,value in diagnostics_train.items():component_sum[key]=component_sum.get(key,0.)+float(value)*size
                        for key,value in model.last_diagnostics.items():training_diag[key]=training_diag.get(key,0)+value.detach().cpu().numpy()*size
                        if not torch.isfinite(loss):raise FloatingPointError('nonfinite loss')
                        loss.backward();clip_value=torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True);clip_norms.append(float(clip_value));batch_boundaries.append(len(ds.training_rows))
                        if state['attempted_updates']>=expected:raise RuntimeError('update allowance exhausted')
                        state['attempted_updates']+=1;state['train_graphs']+=len(output)
                        state['train_3d_graphs']=state.get('train_3d_graphs',0)+int(batch.availability[:,1].sum())
                        opt.step();state['completed_updates']+=1;losses.append(float(loss.detach()))
                    scheduler.step();state['validation_started']+=1
                    y,p,met,diag=evaluate();state['completed_epochs']+=1;epoch=state['completed_epochs']
                    epoch_folder=folder/'validation';epoch_folder.mkdir(exist_ok=True)
                    save_predictions(epoch_folder/f'epoch{epoch:03d}.npz',ds,y,p,epoch=epoch)
                    if met['rmse']<best_rmse:
                        best_rmse=met['rmse'];best_epoch=epoch;best_stage=stage;result=met;diagnostics=diag
                        save_checkpoint(folder/'best.pt',dict(schema=FT_SCHEMA,architecture=m.CONFIG,config=cfg,
                            state_dict={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                            best_epoch=best_epoch,best_stage=best_stage,scaler=scaler,outer_test='NOT_RUN'))
                        save_predictions(folder/'validation_predictions.npz',ds,y,p,best_epoch=best_epoch,best_stage=best_stage)
                    rec=dict(epoch=epoch,stage=stage,stage_epoch=stage_epoch+1,train_loss=float(np.mean(losses)),
                        validation=met,training_components={k:v/train_count for k,v in component_sum.items()},training_modality_diagnostics={k:np.asarray(v/train_count).tolist() for k,v in training_diag.items()},runtime_contract=runtime,training_rows=ds.training_rows.copy(),batch_boundaries=batch_boundaries,clip_norms=clip_norms,learning_rates=rates,optimizer_groups=groups,modality_diagnostics=diag,
                        seconds=time.monotonic()-tick,elapsed_seconds=time.monotonic()-train_tick,gpu_peak_bytes=torch.cuda.max_memory_allocated())
                    history.append(rec);write(folder/'history.json',history);print(json.dumps(rec),flush=True)
                    if arm in ('S42_A','S42_B') and task=='eps' and fold==0 and epoch in (10,11):
                        save_checkpoint(folder/f'boundary_epoch{epoch}.pt',dict(epoch=epoch,model=m.clone_tree(model.state_dict()),optimizer=m.clone_tree(opt.state_dict()),scheduler=m.clone_tree(scheduler.state_dict()),rng=dict(**m.rng_state(),data_loader=train.generator.get_state().clone())))
                    state['epoch_open']=False;state['ft_epochs']=state['epochs_started'];state['ft_updates']=state['attempted_updates'];record()
            if arm in ('S42_A','S42_B'):save_checkpoint(folder/'last.pt',dict(epoch=70,stage='joint',state_dict=m.clone_tree(model.state_dict())))
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


class TraceData(d.Data):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.record_training=False;self.training_rows=[]
    def __getitem__(self,i):
        value=super().__getitem__(i)
        if self.record_training:self.training_rows.append(self.indices[i])
        return value


def register_sources():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('registration must hide GPU')
    source=read(REF/'registry/recipes.json')['recipes'][REF_ID];pt=m.pt_config('P_CURRENT_SHARED',.1,0.)
    prior=read(REF/'final_review/results.json')
    if source['ft']!=m.DEFAULT_FT or source['pt']!=pt or source['pt_id']!=REF_ID or read(REF/'controller_exit.json')['exit']!=0:raise ValueError('original F source identity differs')
    if prior['formal'][REF_ID]['macro']!=REFERENCE or prior['validation'][REF_ID]['macro']!=SPEC['reference_validation'] or prior['FINAL']!='E_P_SYM_SHARED' or prior['performance_pass'] is not False:raise ValueError('protected original reference differs')
    cache=d.InputCache(PARENT/'inputs')
    try:rows=copy.deepcopy(cache.rows)
    finally:cache.close()
    d.validate_missing_inventory(read(PARENT/'inputs/geometry_missing.json'))
    original=read(PARENT/'inputs/original_splits.json');splits={};aliases={};totals=dict(ft_updates=0,test_graphs=0,train_graphs=0,validation_graphs=0)
    for task in TASKS:
        split=read(SPLIT/f'{task}.json');splits[task]=split
        if split!=original[task] or split.get('validation_is_test') is not False or len(split['folds'])!=5:raise ValueError('split identity mismatch')
        for fold,e in enumerate(split['folds']):
            sets=[set(e[k]) for k in ('train_indices','validation_indices','test_indices')]
            if any(sets[i]&sets[j] for i in range(3) for j in range(i)):raise ValueError('split overlap')
            totals['ft_updates']+=70*math.ceil(len(e['train_indices'])/24);totals['test_graphs']+=len(e['test_indices']);totals['train_graphs']+=70*len(e['train_indices']);totals['validation_graphs']+=70*len(e['validation_indices'])
            ft=unit_path('BASE_ORIGINAL',task,fold);formal=unit_path('BASE_ORIGINAL',task,fold,True);cfg=read(ft/'config.json');scaler=read(base_path(task,fold)/'scaler.json')
            required=dict(task=task,fold=fold,seed=42+fold,parent=str(parent_path('S42_A')),head=str(head_path(task,fold)),input_cache=str(PARENT/'inputs'),scaler_reference=str(base_path(task,fold)/'scaler.json'),model=m.CONFIG,world_size=1,batch=24,epochs=70,head_epochs=10,precision='FP32',workers=0)
            if any(cfg.get(k)!=v for k,v in required.items()) or cfg['recipe']!=source:raise ValueError('reference FT scientific config mismatch')
            for folder in (ft,formal):
                met=read(folder/'metrics.json');usage=read(folder/'consumption.json')
                if met['status']!='COMPLETE' or usage['status']!='COMPLETE' or met['config']!=cfg or met['scaler']!=scaler:raise ValueError('reference estimator incomplete')
            if not (ft/'best.pt').is_file() or not head_path(task,fold).is_file():raise FileNotFoundError('read-only BEST/head absent')
            aliases[f'BASE_ORIGINAL/{task}/{fold}']=dict(ft=str(ft),formal=str(formal),new_ft=0,new_inference=0)
    if totals!=dict(ft_updates=56000,test_graphs=5875,train_graphs=1315020,validation_graphs=329980):raise ValueError('per-run budget differs')
    if splits['eps']['folds'][0]['train_indices'][:24]!=SPEC['fixed_rows']:raise ValueError('fixed EPS training batch changed')
    lookup={x['row']:x for x in rows['eps']};batch_identity=[lookup[i] for i in SPEC['fixed_rows']]
    recipes={run:dict(id=run,pt_id=REF_ID,pt=pt,ft=copy.deepcopy(m.DEFAULT_FT),run_id=run,seed_base=seed,numerical_policy='DET',canonical=run in CANONICAL) for run,seed in SPEC['runs'].items()}
    exclusive(OUT/'registry/recipes.json',dict(plan=PLAN,revision=REVISION,recipes=recipes,aliases=aliases,splits=splits,rows=rows,per_recipe=totals,
        diagnostic_input=dict(task='eps',fold=0,indices=SPEC['fixed_rows'],identity=batch_identity,scaler=read(base_path('eps',0)/'scaler.json')),
        reference=REFERENCE,reference_root=str(REF),time=now(),model_constructed=False,checkpoint_loaded=False,test_targets_accessed=False))
    exclusive(OUT/'environment.json',dict(versions=m.versions(),controller_environment=m.numerical_state(),DET_required_env=m.DET_ENV,DEFAULT_policy='original entry and inherited environment; DET-specific env not injected'))
    return dict(status='COMPLETE',time=now(),model_constructed=False,checkpoint_loaded=False,no_forward=True,aliases=35)


def diagnostic(name,policy_setup):
    require_tmux()
    if name not in DIAGNOSTICS:raise ValueError('unregistered diagnostic')
    folder=OUT/'diagnostics'/name;folder.mkdir(parents=True,exist_ok=False)
    policy='DET' if name.startswith('DET_') else 'DEFAULT'
    if policy_setup['policy']!=policy:raise ValueError('diagnostic policy mismatch')
    cfg=dict(plan=PLAN,revision=REVISION,name=name,numerical_policy=policy,task='eps',fold=0,seed=42,fixed_rows=SPEC['fixed_rows'],ft=m.DEFAULT_FT,pt=str(parent_path('S42_A')),eval_best=str(unit_path('BASE_ORIGINAL','eps',0)/'best.pt'),head=str(head_path('eps',0)),policy_setup=policy_setup)
    write(folder/'config.json',cfg)
    state=dict(status='RUNNING',process=process_identity(os.getpid()),started=now(),phase='initialize',diagnostic_train_starts=0,
        diagnostic_eval_forwards=0,diagnostic_eval_graphs=0,diagnostic_train_forwards=0,diagnostic_backwards=0,diagnostic_updates=0,diagnostic_train_graphs=0)
    for k in ('diagnostic_eval_forwards','diagnostic_eval_graphs','diagnostic_train_forwards','diagnostic_backwards','diagnostic_updates','diagnostic_train_graphs'):state['completed_'+k]=0
    def record():write(folder/'consumption.json',state)
    record();ds=None;hooks=None;started=time.monotonic()
    try:
        state['actual_device']=gpu_contract();record()
        write(folder/'environment.json',dict(versions=m.versions(),setup=policy_setup,actual=m.numerical_state(),physical=state['actual_device']))
        ds=d.Data('eps',0,False,DOWN,SPLIT,PARENT/'inputs');scaler=read(base_path('eps',0)/'scaler.json');scaler_validate(ds.raw_targets,ds.train,scaler);ds.targets=transform(ds.raw_targets,scaler)
        positions={row:i for i,row in enumerate(ds.indices)};indices=[positions[i] for i in SPEC['fixed_rows']]
        if any(i not in ds.train for i in indices):raise ValueError('diagnostic record outside training split')
        identity=dict(indices=SPEC['fixed_rows'],keys=ds.keys[indices].tolist(),raw_targets=ds.raw_targets[indices].tolist(),scaler=scaler)
        write(folder/'input_identity.json',identity)
        batch=d.collate([ds[i] for i in indices]).to('cuda:0')
        seed_all(42)
        saved=torch.load(cfg['eval_best'],map_location='cpu',weights_only=False)
        if saved.get('schema')!='ua3d-base-opt-best-r6' or saved['config']!=config('BASE_ORIGINAL','eps',0) or saved['scaler']!=scaler:raise ValueError('diagnostic read-only BEST mismatch')
        model=m.BaseOptFuse(False,ft=saved['config']['recipe']['ft']);m.strict_module_state(model,saved['state_dict']);del saved
        model.to('cuda:0');model.eval();hooks=m.Hooks(model);eval_paths=[]
        for repeat in range(2):
            if policy=='DET':m.check_det()
            state['phase']='eval';state['diagnostic_eval_forwards']+=1;state['diagnostic_eval_graphs']+=24;record();hooks.begin();before=m.rng_state()
            with torch.inference_mode():p,weights,z=model.predict(batch)
            value=dict(hooks=hooks.take(),runtime=model.runtime_contract(),prediction=m.clone_tree(p),weights=m.clone_tree(weights),modalities=m.clone_tree(z),rng_before=before,rng_after=m.rng_state())
            path=folder/f'eval{repeat}.pt';save_checkpoint(path,value);eval_paths.append(str(path))
            state['completed_diagnostic_eval_forwards']+=1;state['completed_diagnostic_eval_graphs']+=24;record()
        hooks.close();hooks=None;del model
        seed_all(42);state['diagnostic_train_starts']=1;record()
        package=torch.load(parent_path('S42_A'),map_location='cpu',weights_only=False);head=torch.load(head_path('eps',0),map_location='cpu',weights_only=False)
        model=m.initialize_ft(package,head,get_recipe('S42_A'));del package,head
        save_checkpoint(folder/'initial.pt',dict(state_dict=m.clone_tree(model.state_dict()),rng=m.rng_state()))
        model.to('cuda:0');hooks=m.Hooks(model);seed_all(42);records=[]
        for stage in ('head','joint'):
            opt,groups=m.optimizer_ft(model,stage)
            scheduler=(torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=10,eta_min=3e-5) if stage=='head' else torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt,T_0=10,T_mult=1,eta_min=1e-6))
            for local_step in range(10):
                step=len(records)+1
                if policy=='DET':m.check_det()
                state['phase']=stage;model.train();runtime=model.runtime_contract();opt.zero_grad(set_to_none=True);hooks.begin();before=m.rng_state();rates=[g['lr'] for g in opt.param_groups]
                state['diagnostic_train_forwards']+=1;state['diagnostic_train_graphs']+=24;record()
                loss,parts=model.training_loss(batch)
                if not torch.isfinite(loss):raise FloatingPointError('nonfinite diagnostic loss')
                state['completed_diagnostic_train_forwards']+=1;state['completed_diagnostic_train_graphs']+=24;state['diagnostic_backwards']+=1;record();loss.backward();state['completed_diagnostic_backwards']+=1;record()
                if step in (1,11):save_checkpoint(folder/f'grad_pre_step{step}.pt',{n:m.clone_tree(p.grad) for n,p in model.named_parameters() if p.grad is not None})
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True)
                if step in (1,11):save_checkpoint(folder/f'grad_post_step{step}.pt',{n:m.clone_tree(p.grad) for n,p in model.named_parameters() if p.grad is not None})
                state['diagnostic_updates']+=1;record();opt.step();state['completed_diagnostic_updates']+=1;scheduler.step();record()
                item=dict(step=step,stage=stage,loss=float(loss.detach()),components={k:float(v) for k,v in parts.items()},clip_norm=float(norm),learning_rates=rates,optimizer_groups=groups,runtime=runtime,rng_before=before,rng_after=m.rng_state(),hooks=hooks.take())
                save_checkpoint(folder/f'train_step{step:02d}.pt',item);records.append(dict(step=step,stage=stage,loss=item['loss'],components=item['components'],clip_norm=item['clip_norm'],learning_rates=rates))
                write(folder/'history.json',records)
                if step in (1,11,20):save_checkpoint(folder/f'updated_step{step}.pt',dict(state_dict=m.clone_tree(model.state_dict()),optimizer=m.clone_tree(opt.state_dict()),scheduler=m.clone_tree(scheduler.state_dict()),rng=m.rng_state()))
        hooks.close();hooks=None
        if state['diagnostic_eval_forwards']!=2 or state['diagnostic_updates']!=20:raise ValueError('incomplete diagnostic budget')
        state.update(status='COMPLETE',ended=now(),elapsed_seconds=time.monotonic()-started);record()
        write(folder/'metrics.json',dict(status='COMPLETE',name=name,policy=policy,eval_paths=eval_paths,train_steps=20,identity=identity))
    except BaseException:
        ignore_termination();state.update(status='FAILED',error=traceback.format_exc(),ended=now(),elapsed_seconds=time.monotonic()-started);record();write(folder/'failure.json',dict(error=state['error'],retry=False));raise
    finally:
        if hooks is not None:hooks.close()
        if ds is not None:ds.close()


def compare_diagnostics():
    require_tmux()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('comparison must hide GPUs')
    if (OUT/'diagnostics/comparison.json').exists():raise FileExistsError('diagnostics compared only once')
    files=['initial.pt']+[f'eval{i}.pt' for i in range(2)]+[f'train_step{i:02d}.pt' for i in range(1,21)]+[f'grad_{stage}_step{i}.pt' for i in (1,11) for stage in ('pre','post')]+[f'updated_step{i}.pt' for i in (1,11,20)]
    result=dict(time=now(),checks={},numerical_diagnostic_pass=True,method='key/shape/dtype and torch.equal; exact scalar/RNG comparisons; no hashes, tolerance, or extra forward')
    for policy in ('DEFAULT','DET'):
        checks={}
        for repeat in (0,1):
            folder=OUT/'diagnostics'/f'{policy}_R{repeat}';a=torch.load(folder/'eval0.pt',map_location='cpu',weights_only=False);b=torch.load(folder/'eval1.pt',map_location='cpu',weights_only=False)
            checks[f'intra_eval_R{repeat}']=m.exact_compare(a,b);del a,b
        for name in files:
            a=torch.load(OUT/'diagnostics'/f'{policy}_R0'/name,map_location='cpu',weights_only=False);b=torch.load(OUT/'diagnostics'/f'{policy}_R1'/name,map_location='cpu',weights_only=False)
            checks[name]=m.exact_compare(a,b);del a,b
        first=next((dict(check=k,**v) for k,v in checks.items() if not v['equal']),None)
        result['checks'][policy]=dict(equal=first is None,first_divergence=first,max_abs_diff=max(v['max_abs_diff'] for v in checks.values()),details=checks)
        if policy=='DET' and first is not None:result['numerical_diagnostic_pass']=False
    exclusive(OUT/'diagnostics/comparison.json',result)
    if not result['numerical_diagnostic_pass']:raise RuntimeError('strict DET diagnostic exact reproduction failed; stop without FT or repair')
    return result

def scientific_config(cfg):
    return {k:cfg[k] for k in ('task','fold','seed','seed_base','numerical_policy','parent','head','input_cache','scaler_reference','model','world_size','batch','epochs','head_epochs','precision','workers','selection')}


def compare_pair(task,fold):
    require_tmux()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('same-seed comparison must hide GPU')
    path=OUT/'reproducibility'/task/f'fold{fold}.json'
    if path.exists():raise FileExistsError('each same-seed pair is compared once')
    dirs=[unit_path(run,task,fold) for run in ('S42_A','S42_B')];checks={}
    def add(name,a,b):checks[name]=m.exact_compare(a,b)
    usages=[read(d/'consumption.json') for d in dirs]
    if any(u['status']!='COMPLETE' for u in usages) or usages[0]['process']==usages[1]['process']:raise ValueError('independent complete FT processes required')
    add('scientific_config',*[scientific_config(read(d/'config.json')) for d in dirs])
    add('numerical_policy',*[read(d/'environment.json')['numerical'] for d in dirs])
    histories=[read(d/'history.json') for d in dirs]
    keys=('epoch','stage','stage_epoch','train_loss','validation','training_components','training_modality_diagnostics','runtime_contract','training_rows','batch_boundaries','clip_norms','learning_rates','optimizer_groups','modality_diagnostics')
    if any(len(h)!=70 for h in histories):raise ValueError('70 epochs required before exact comparison')
    for i in range(70):
        add(f'epoch{i+1:03d}/numerical_trajectory',*[{k:h[i][k] for k in keys} for h in histories])
        values=[]
        for d0 in dirs:
            with np.load(d0/'validation'/f'epoch{i+1:03d}.npz',allow_pickle=False) as p:values.append({k:p[k].copy() for k in p.files})
        add(f'epoch{i+1:03d}/validation_predictions',*values)
    met=[read(d/'metrics.json') for d in dirs]
    add('BEST_metadata',*[{k:v[k] for k in ('best_epoch','best_stage','scaler','r2','rmse','mae','diagnostics','parameters')} for v in met])
    for file in ('best.pt','last.pt'):
        saved=[torch.load(d/file,map_location='cpu',weights_only=False) for d in dirs]
        add(file+'/state_dict',*[v['state_dict'] for v in saved]);del saved
    if task=='eps' and fold==0:
        for epoch in (10,11):
            saved=[torch.load(d/f'boundary_epoch{epoch}.pt',map_location='cpu',weights_only=False) for d in dirs]
            add(f'boundary_epoch{epoch}',*saved);del saved
    first=next((dict(check=k,**v) for k,v in checks.items() if not v['equal']),None)
    result=dict(task=task,fold=fold,time=now(),equal=first is None,first_divergence=first,max_abs_diff=max(v['max_abs_diff'] for v in checks.values()),checks=checks,
        sources=[str(d) for d in dirs],independent_processes=[u['process'] for u in usages],comparison='torch.equal/key/shape/dtype and exact rows/predictions/scalars; no tolerance or file hash')
    exclusive(path,result)
    if first is not None:raise RuntimeError(f'same-seed exact reproduction failed at {task}/fold{fold}: {first}; no later FT/freeze/TEST or repair')
    return result


def freeze():
    check_prepared()
    if (OUT/'selection/final_frozen.json').exists():raise FileExistsError('one immutable freeze only')
    comparison=read(OUT/'diagnostics/comparison.json')
    if comparison['numerical_diagnostic_pass'] is not True:raise ValueError('diagnostic gate not passed')
    for task in TASKS:
        for fold in range(5):
            if read(OUT/'reproducibility'/task/f'fold{fold}.json')['equal'] is not True:raise ValueError('same-seed gate incomplete/failed')
    record=registry();validation={k:aggregate(k) for k in ORDER}
    units={key:u for value in validation.values() for key,u in value['units'].items()}
    contract=dict(source=str(DOWN),label='original property label',transform='fixed train-only scaler; inverse for metrics',truth_access='inside canonical formal TEST only',precision='FP32')
    for task in TASKS:
        rows={x['row']:x for x in record['rows'][task]}
        for fold in range(5):
            indices=sorted(record['splits'][task]['folds'][fold]['test_indices']);keys=[rows[i]['key'] for i in indices]
            for run in ORDER:
                u=units[f'{run}/{task}/{fold}'];environment=None
                if run in RUNS:
                    saved=torch.load(u['best'],map_location='cpu',weights_only=False)
                    if saved.get('schema')!=FT_SCHEMA or saved['config']!=u['config'] or saved['scaler']!=u['scaler'] or saved['best_epoch']!=u['best_epoch'] or saved['best_stage']!=u['best_stage']:raise ValueError('frozen BEST metadata mismatch')
                    environment=read(unit_path(run,task,fold)/'environment.json');del saved
                u.update(test_keys=keys,target_contract=contract,pt_source=str(parent_path(run)),head_source=str(head_path(task,fold)),
                    fusion_initialization=dict(source='src/modules/ua3m_paperfuse.py:PaperFuse.__init__',seed=m.native.SEED,policy='original fixed fork_rng'),
                    p=.5,inference_allowed=run in CANONICAL,role='canonical' if run in CANONICAL else 'same_seed_replica' if run=='S42_B' else 'read_only_reference',
                    numerical_policy='DET' if run in RUNS else 'historical',actual_environment=environment,
                    split_source=str(SPLIT/f'{task}.json'),row_key_source=str(PARENT/'inputs/rows.json'),scaler_source=str(base_path(task,fold)/'scaler.json'),
                    formal_alias=str(unit_path(run,task,fold,True)) if run=='BASE_ORIGINAL' else None)
    if len(units)!=175 or sum(u['inference_allowed'] for u in units.values())!=105:raise ValueError('frozen inventory must be 105 canonical +35 replica +35 reference')
    frozen=dict(plan=PLAN,revision=REVISION,frozen=True,time=now(),FINAL_recipe='DET_BASE',PRIMARY_RUN='S42_A',units=units,validation=validation,registry=record,
        canonical=CANONICAL,replica='S42_B',test_targets_read=False,numerical_diagnostic_pass=True,same_seed_repro_pass=True,selection='fixed recipe/primary run before training; no best-seed selection',shared_development_evaluation=True)
    exclusive(OUT/'selection/final_frozen.json',frozen)
    return frozen

def check_space():
    free=shutil.disk_usage(ROOT).free
    if free<SPEC['min_free_bytes']:raise RuntimeError('less than 60 GiB free; stop without deleting old evidence')
    return free

def seed_statistics(formal):
    macros=[formal[run]['macro'] for run in CANONICAL];tasks={}
    for task in TASKS:
        values=[formal[run]['tasks'][task]['mean'] for run in CANONICAL]
        tasks[task]=dict(seed_fold_means=dict(zip(CANONICAL,values)),mean=float(np.mean(values)),std=float(np.std(values,ddof=1)),range=max(values)-min(values),pass_range=max(values)-min(values)<=.020)
    result=dict(macros=dict(zip(CANONICAL,macros)),mean=float(np.mean(macros)),std=float(np.std(macros,ddof=1)),range=max(macros)-min(macros),macro_range_pass=max(macros)-min(macros)<=.005,tasks=tasks,
        aggregation='3 canonical macro values; task ranges use 3 corresponding five-fold means; S42_B excluded')
    return result,result['macro_range_pass'] and all(v['pass_range'] for v in tasks.values())


def primary_assessment(formal):
    primary=formal['S42_A'];reference=formal['BASE_ORIGINAL'];delta=primary['macro']-REFERENCE
    quality=dict(macro_pass=primary['macro']>=REFERENCE-.005,task_deltas={task:primary['tasks'][task]['mean']-reference['tasks'][task]['mean'] for task in TASKS},task_pass={task:primary['tasks'][task]['mean']>=reference['tasks'][task]['mean']-.010 for task in TASKS})
    quality['pass']=quality['macro_pass'] and all(quality['task_pass'].values())
    return quality,delta>=.010,delta
