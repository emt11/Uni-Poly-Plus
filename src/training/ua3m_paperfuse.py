"""Finite UA3M campaign: one cache build, two PTs, 70 FT and frozen TEST."""
import datetime
import json
import math
import os
from pathlib import Path
import random
import signal
import time
import traceback
import lmdb
import numpy as np
import torch
from src.modules import ua3m_paperfuse as m
from src.dataset import ua3m_paperfuse as d
from src.training.glt_dual_runtime import require_tmux,save_checkpoint,pretrain_source_fields
from src.training.ua2d_p10k_nofp_align import finite,metrics,process_identity,process_alive,SOTA,pt_learning_rate
from src.training.ua2d_a020_fp_fusion import scaler_validate,transform,load_prediction,align_predictions

ROOT=Path(__file__).resolve().parents[2];PLAN='UA3M-PAPERFUSE-20261025-01';REVISION='r1'
OUT=ROOT/'results/ua3m_paperfuse_20261025/r1';LOG=ROOT/'logs/ua3m_paperfuse_20261025/r1'
ARMS=m.ARMS;TASKS=m.TASKS
BASE=ROOT/'results/ua2d_p10k_nofp_align_20261024/r1'
DOWN=ROOT/'data/processed/mcl_ph_r4_20260930/downstream_direct_01.staging';SPLIT=ROOT/'data/splits/mips_paper5_outer'
GRAPH=ROOT/'results/ua2d_transfer_20261022/r1/inputs';FP=ROOT/'results/ua2d_fg10k_morgan_20261023/r1/inputs'
PT_CACHE=ROOT/'results/ua2d_pretrain_fp_20261019/fast_r4/prepared_cache';MAP_ROOT=ROOT/'results/ua2d_ptscale_20261022/r1/inputs'
INITIAL=ROOT/'results/ua2d_pretrain_fp_20261019/fast_r4/initialization/common_step0.pt'
BASE_MACRO=.785044436513689
LIMITS=dict(prepare_starts=1,cache_rows=5875,pt_starts=2,pt_updates=20000,pt_microbatches=120000,pt_presentations=20160000,
    ft_starts=70,epochs_started=4900,attempted_updates=112000,train_graphs=2630040,validation_started=4900,validation_graphs=659960,
    test_starts=70,test_graphs=11750,test_3d_graphs=11720,cpu_suites=1)
ZERO_COST=dict(fp_generation=0,geometry_generation=0,kernel_fit=0,kernel_load=0,kernel_query=0,fp_prediction=0,baseline_new_inference=0,output_fusion=0)



PARENT_OUT=OUT
REPAIR_OUT=PARENT_OUT/'repairs/prepare_repair1'
REPAIR_LOG=LOG/'prepare_repair1'
RESTART_OUT=PARENT_OUT/'repairs/gpu0123_restart1'
INPUT_OUT=None
ALLOWED_GPUS=(3,)
DEFAULT_LIMITS=dict(LIMITS)
EXECUTION_REVISION=None
DEADLINE=datetime.datetime.fromisoformat('2026-10-27T10:32:33.682164+08:00').timestamp()


def configure_execution(root=None):
    global OUT,LOG,LIMITS,EXECUTION_REVISION,INPUT_OUT,ALLOWED_GPUS
    target=Path(root).resolve() if root is not None else PARENT_OUT
    if target not in (PARENT_OUT,REPAIR_OUT,RESTART_OUT):raise ValueError('unauthorized execution root')
    OUT=target;INPUT_OUT=None
    LOG=REPAIR_LOG.parent/'gpu0123_restart1' if target==RESTART_OUT else REPAIR_LOG if target==REPAIR_OUT else REPAIR_LOG.parent
    EXECUTION_REVISION='gpu0123-r1' if target==RESTART_OUT else 'prepare-r1' if target==REPAIR_OUT else None
    ALLOWED_GPUS=(0,1,2,3) if target==RESTART_OUT else (3,)
    LIMITS=dict(DEFAULT_LIMITS)
    if EXECUTION_REVISION:LIMITS.update(prepare_starts=2,cache_rows=11750)
    if target==RESTART_OUT:LIMITS.update(prepare_starts=3,cache_rows=17625,pt_starts=3,pt_updates=30000,pt_microbatches=180000,pt_presentations=30240000)


def save_initialization(path,payload):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    # Exclusive creation prevents both an existing checkpoint and a competing writer.
    with path.open('xb') as stream:torch.save(payload,stream)


def now():return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
def read(p):return json.loads(Path(p).read_text())
def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);temp=p.with_suffix(p.suffix+'.tmp')
    temp.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n');os.replace(temp,p)
def exclusive(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False)
def ignore_termination():
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,signal.SIG_IGN)
def seed_all(seed):random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
def parent_path(arm):return OUT/'pretrain'/arm/'deploy.pt'
def unit_path(arm,task,fold,formal=False):return OUT/('formal' if formal else 'finetune')/arm/task/f'fold{fold}'
def prepared_root():return INPUT_OUT if INPUT_OUT is not None else OUT
def head_path(task,fold):return prepared_root()/'initialization/heads'/task/f'fold{fold}.pt'
def modal_path(arm):return prepared_root()/'initialization'/f'{arm}_modal.pt'
def base_path(task,fold,formal=False):return BASE/('formal' if formal else 'finetune')/'D_A020'/task/f'fold{fold}'
def config(arm,task,fold):
    if arm not in ARMS or task not in TASKS or type(fold) is not int or fold not in range(5):raise ValueError('outside UA3M matrix')
    return dict(plan=PLAN,revision=REVISION,arm=arm,task=task,fold=fold,seed=42+fold,parent=str(parent_path(arm)),
        head=str(head_path(task,fold)),input_cache=str(prepared_root()/'inputs'),scaler_reference=str(base_path(task,fold)/'scaler.json'),
        model=m.CONFIG,world_size=1,batch=24,epochs=70,head_epochs=10,precision='FP32',workers=0,
        loss='MSE' if arm=='UNI3M' else 'MSE+0.08*ALIGN_TMOL-0.08*gate_entropy')
def update_plan(status,phase,detail):
    p=ROOT/'CURRENT_PLAN.md';s=p.read_text();tag='UA3M_PAPERFUSE_GPU0123' if EXECUTION_REVISION=='gpu0123-r1' else 'UA3M_PAPERFUSE_REPAIR1' if EXECUTION_REVISION else 'UA3M_PAPERFUSE'
    start=f'\n<!-- {tag}_STATUS_START -->';end=f'\n<!-- {tag}_STATUS_END -->'
    block=start+f'\n- {now()}；状态：**{status}**；阶段：{phase}。\n- {detail}\n- 账本 `{OUT}/budget.json`；日志 `{LOG}`；独立复审PENDING，未归档。'+end
    if start in s:
        left,rest=s.split(start,1);_,right=rest.split(end,1);s=left+block+right
    else:s=s.rstrip()+'\n'+block+'\n'
    if EXECUTION_REVISION:
        lines=s.splitlines();lines[2]=f'> 最新索引（{now()}）：父UA3M-PAPERFUSE/r1保持**需返修**；返修UA3M-PAPERFUSE-REPAIR-20261025-01/{EXECUTION_REVISION}已获用户明确授权，状态**{status}**，阶段{phase}。独立返修状态块见文末；原失败与旧FINAL保留，独立复审PENDING。';s='\n'.join(lines)+'\n'
    else:
        for old in ('待授权','执行中','待审查','需返修','阻断'):s=s.replace('修订r1；状态：**'+old+'**','修订r1；状态：**'+status+'**',1)
    p.write_text(s);write(OUT/'status.json',dict(status=status,phase=phase,detail=detail,time=now(),independent_review='PENDING'))


def prepare():
    from src.dataset.mcl_ph_direct import DirectMCLSource
    from src.dataset.glt_dual import build_dual_sample
    from src.training.ua2d_p10k_nofp_align import InputCache
    require_tmux();torch.set_num_threads(2)
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('prepare is CPU only')
    folder=OUT/'inputs';folder.mkdir(exist_ok=False);started=time.monotonic();env=None
    state=dict(status='RUNNING',records=0,geometry_available=0,geometry_missing=0,started=now(),pid=os.getpid(),process=process_identity(os.getpid()))
    def record():write(folder/'coverage.json',dict(state,elapsed_seconds=time.monotonic()-started))
    record()
    try:
        pt=read(PT_CACHE/'metadata.json');mapping=read(MAP_ROOT/'metadata.json')
        if pt.get('schema')!='ua2d_prepared_templates_r4' or pt.get('status')!='COMPLETE' or pt.get('train_rows')!=911391 or mapping.get('records')!=910756 or mapping.get('status')!='COMPLETE' or mapping.get('eligibility_semantics')!='logical_slot_not_source_row':raise ValueError('PT immutable input metadata mismatch')
        for name,shape in (('eligible_logical.npy',(910756,)),('source_rows.npy',(910756,)),('keys.npy',(910756,))):
            if np.load(MAP_ROOT/name,mmap_mode='r',allow_pickle=False).shape!=shape:raise ValueError('logical mapping shape mismatch')
        common=torch.load(INITIAL,map_location='cpu',weights_only=False);heads=torch.load(INITIAL.parent/'task_heads/V1.pt',map_location='cpu',weights_only=False)
        if common.get('schema')!='ua2d_galformer_objectives_v3' or common.get('step')!=0 or common.get('seed')!=42 or heads.get('step')!=0:raise ValueError('step0 initialization identity mismatch')
        merged=dict(common['state_dict'],**heads['state_dict'])
        if any(k.startswith('fp_head.') for k in merged):raise ValueError('FP prediction in source initialization')
        (OUT/'initialization').mkdir(exist_ok=False)
        for arm in ARMS:save_initialization(modal_path(arm),m.modal_modules(arm).state_dict())
        for index,task in enumerate(TASKS):
            for fold in range(5):
                with torch.random.fork_rng(devices=[]):
                    torch.default_generator.manual_seed(20261025+100*index+fold);head=m.property_head(.1)
                save_initialization(head_path(task,fold),head.state_dict())
        write(OUT/'initialization/metadata.json',dict(common=str(INITIAL),heads=str(INITIAL.parent/'task_heads/V1.pt'),native_modules=list(m.NATIVE),mask_heads=list(m.MASK_HEADS),
            excluded=[k for k in merged if k.startswith(('proj2.','proj3.'))],new_modal_seed=20261025,head_seed='20261025+100*task_index+fold',model=m.CONFIG,fp_as_input=True,fp_prediction=False))
        del common,heads,merged
        graphs=InputCache(GRAPH);fps=d.MorganCache(FP,GRAPH,DOWN,SPLIT)
        env=lmdb.open(str(folder/'graphs.lmdb'),map_size=64*1024**3,max_readers=128);rows={};missing=[];splits={}
        for task in TASKS:
            split=read(SPLIT/f'{task}.json');splits[task]=split
            source=DirectMCLSource(DOWN,task=task,selected_indices=None,split_manifest=split,verify_key_digest=False)
            rows[task]=[]
            try:
                for i,entry in enumerate(source.entries):
                    row=int(entry['original_row']);key=source.samples[i][0].hex();sm=entry['normalized_smiles'];cache_key=f'{task}:{row}'
                    d.validate_join(graphs.mapping[task].get(row),fps.mapping[task].get(row),row,key,sm)
                    graph=graphs.get(task,row,key);fp=fps.get(task,row,key);available=entry['geometry_status']=='PASS';geometry=None
                    if available:
                        topology,trimer,text,static,_,identity=pretrain_source_fields(source,i,include_identity=True)
                        geometry=build_dual_sample(topology,trimer,text,static=static,geometry_mode='finite_incident',identity=identity)
                        if not bool(geometry.geometry_valid) or geometry.incidence_index.numel()==0:raise ValueError('original PASS geometry unavailable')
                        for name in ('mips_x','mips_backbone_mask','lga_spd','bond_path_features'):
                            if not torch.equal(getattr(graph,name),getattr(geometry,name)):raise ValueError('old2D/currentUA tensor semantics mismatch '+name)
                        state['geometry_available']+=1
                    else:
                        missing.append(dict(task=task,row=row,key=key,smiles=sm,reason=entry['geometry_status']));state['geometry_missing']+=1
                    payload=d.encode(dict(schema=d.CACHE_SCHEMA,task=task,row=row,sample_key=key,smiles=sm,clean=graph,geometry=geometry,fp=fp,available=available))
                    with env.begin(write=True) as txn:
                        if not txn.put(cache_key.encode(),payload,overwrite=False):raise ValueError('cache writer conflict')
                    rows[task].append(dict(row=row,key=key,smiles=sm,cache_key=cache_key,available=available));state['records']+=1
                    if state['records']%128==0:record();print(json.dumps(state),flush=True)
                graphs.memory.clear()
            finally:source.close()
        graphs.close()
        d.validate_missing_inventory(missing,state['records'],state['geometry_available'])
        write(folder/'geometry_missing.json',missing);write(folder/'rows.json',rows);write(folder/'original_splits.json',splits)
        write(folder/'references.json',dict(pt_cache=str(PT_CACHE),pt_metadata=pt,logical_map=str(MAP_ROOT),mapping_metadata=mapping,graph=str(GRAPH),fp=str(FP),fp_metadata=fps.metadata,source=str(DOWN),schema=d.CACHE_SCHEMA,model=m.CONFIG))
        env.sync();env.close();env=None;state.update(status='COMPLETE',ended=now());record()
        write(folder/'metadata.json',dict(plan=PLAN,revision=REVISION,schema=d.CACHE_SCHEMA,status='COMPLETE',records=5875,geometry_available=5860,geometry_missing=15,
            targets_cached=False,label_statistics_fitted=False,fp_generation=0,geometry_generation=0,model_forward=0,online_fallback=False))
    except BaseException:
        ignore_termination();state.update(status='FAILED',error=traceback.format_exc(),ended=now());record();raise
    finally:
        if env is not None:env.close()


def check_prepared():
    meta=read(prepared_root()/'inputs/metadata.json')
    if meta.get('status')!='COMPLETE' or meta.get('schema')!=d.CACHE_SCHEMA or (meta.get('records'),meta.get('geometry_available'),meta.get('geometry_missing'))!=(5875,5860,15):raise ValueError('prepare incomplete')
    coverage=read(prepared_root()/'inputs/coverage.json')
    if coverage.get('status')!='COMPLETE' or (coverage.get('records'),coverage.get('geometry_available'),coverage.get('geometry_missing'))!=(5875,5860,15):raise ValueError('physical coverage incomplete')
    d.validate_missing_inventory(read(prepared_root()/'inputs/geometry_missing.json'))
    for path in [*(modal_path(arm) for arm in ARMS),*(head_path(task,fold) for task in TASKS for fold in range(5)),prepared_root()/'initialization/metadata.json']:
        if not path.is_file() or path.stat().st_size==0:raise ValueError('initialization incomplete: '+str(path))
    cache=d.InputCache(prepared_root()/'inputs');cache.close()
    split=read(prepared_root()/'inputs/original_splits.json')
    if any(split[task]!=read(SPLIT/f'{task}.json') for task in TASKS):raise ValueError('original split changed')


def frozen_unit(frozen,arm,task,fold):
    if frozen.get('plan')!=PLAN or frozen.get('frozen') is not True:raise ValueError('TEST before valid freeze')
    chosen=frozen['units'][f'{arm}/{task}/{fold}']
    if chosen['config']!=config(arm,task,fold) or chosen['best']!=str(unit_path(arm,task,fold)/'best.pt'):raise ValueError('frozen unit identity mismatch')
    return chosen


def save_predictions(path,ds,y,p,**metadata):
    np.savez(path,y_true=finite(y),y_pred=finite(p),indices=np.array([ds.indices[i] for i in ds.validation]),sample_keys=ds.keys[ds.validation],
        availability=np.array([[True,ds.cache.mapping[ds.task][ds.indices[i]]['available'],True] for i in ds.validation],dtype=bool),**metadata)


def unit(arm,task,fold,formal=False,physical_gpu=3):
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
            model=m.PaperFuse(arm,False)
            if saved.get('schema')!=m.FT_SCHEMA or saved.get('config')!=cfg:raise ValueError('FT checkpoint schema/config mismatch')
            m.strict_module_state(model,saved['state_dict']);best_epoch=saved['best_epoch'];best_stage=saved['best_stage'];del saved
        else:
            seed_all(42+fold)
            package=torch.load(parent_path(arm),map_location='cpu',weights_only=False)
            head=torch.load(head_path(task,fold),map_location='cpu',weights_only=False)
            model=m.initialize_ft(package,head);del package,head
        ds=d.Data(task,fold,formal,DOWN,SPLIT,prepared_root()/'inputs')
        if formal:
            if selected['test_indices']!=ds.indices:raise ValueError('frozen TEST rows mismatch')
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
            seed_all(42+fold);history=[];best_r2=-math.inf;train_tick=time.monotonic()
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
                    if met['r2']>best_r2:
                        best_r2=met['r2'];best_epoch=epoch;best_stage=stage;result=met;diagnostics=diag
                        save_checkpoint(folder/'best.pt',dict(schema=m.FT_SCHEMA,architecture=m.CONFIG,config=cfg,
                            state_dict={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                            best_epoch=best_epoch,best_stage=best_stage,scaler=scaler,outer_test='NOT_RUN'))
                        save_predictions(folder/'validation_predictions.npz',ds,y,p,best_epoch=best_epoch,best_stage=best_stage)
                    rec=dict(epoch=epoch,stage=stage,stage_epoch=stage_epoch+1,train_loss=float(np.mean(losses)),
                        validation=met,training_components={k:float(v) for k,v in diagnostics_train.items()},learning_rates=rates,optimizer_groups=groups,modality_diagnostics=diag,
                        seconds=time.monotonic()-tick,elapsed_seconds=time.monotonic()-train_tick,gpu_peak_bytes=torch.cuda.max_memory_allocated())
                    history.append(rec);write(folder/'history.json',history);print(json.dumps(rec),flush=True)
                    state['epoch_open']=False;record()
            if state['completed_epochs']!=70 or state['attempted_updates']!=expected or state['completed_updates']!=expected:
                raise ValueError('incomplete FT trajectory')
        write(folder/'metrics.json',dict(status='COMPLETE',config=cfg,best_epoch=best_epoch,best_stage=best_stage,scaler=scaler,
            **result,diagnostics=diagnostics,parameters=parameter_counts))
        state.update(status='COMPLETE',ended=now(),elapsed_seconds=time.monotonic()-started,gpu_peak_bytes=torch.cuda.max_memory_allocated(),
            best_epoch=best_epoch,best_stage=best_stage);record()
    except BaseException:
        ignore_termination()
        state.update(status='FAILED',error=traceback.format_exc(),ended=now());record()
        write(folder/'failure.json',dict(error=state['error'],automatic_retry=False));raise
    finally:
        ignore_termination()
        if ds is not None:ds.close()



def summary(scores):
    if set(scores)!=set(TASKS) or any(len(scores[task])!=5 for task in TASKS):raise ValueError('seven tasks five folds required')
    tasks={task:dict(folds=list(map(float,scores[task])),mean=float(finite(scores[task]).mean()),std=float(finite(scores[task]).std(ddof=1))) for task in TASKS}
    return dict(tasks=tasks,macro=float(np.mean([v['mean'] for v in tasks.values()])))


def aggregate(arm,formal=False):
    scores={task:[] for task in TASKS};units={}
    for task in TASKS:
        for fold in range(5):
            folder=unit_path(arm,task,fold,formal);met=read(folder/'metrics.json');c=read(folder/'consumption.json')
            if c['status']!='COMPLETE' or met['status']!='COMPLETE' or met['config']!=config(arm,task,fold):raise ValueError('incomplete unit')
            entry=read(SPLIT/f'{task}.json')['folds'][fold]
            expected=dict(test_graphs=len(entry['test_indices']),test_started=1) if formal else dict(completed_epochs=70,
                epochs_started=70,attempted_updates=70*math.ceil(len(entry['train_indices'])/24),completed_updates=70*math.ceil(len(entry['train_indices'])/24),
                train_graphs=70*len(entry['train_indices']),validation_started=70,validation_graphs=70*len(entry['validation_indices']),epoch_open=False)
            if any(c.get(k)!=v for k,v in expected.items()):raise ValueError('unit scientific consumption mismatch')
            scores[task].append(met['r2'])
            ft=unit_path(arm,task,fold)
            units[f'{arm}/{task}/{fold}']=dict(config=met['config'],best=str(ft/'best.pt'),best_epoch=met['best_epoch'],best_stage=met['best_stage'],
                scaler=met['scaler'],scaler_path=str(ft/'scaler.json'),test_indices=sorted(entry['test_indices']),
                predictions=str(folder/('test_predictions.npz' if formal else 'validation_predictions.npz')))
    return dict(**summary(scores),units=units)



def nominate(validation):
    order=('B_A020','UNI3M','TMOL3M')
    if set(validation)!=set(order):raise ValueError('complete validation matrix required')
    finite([validation[a]['macro'] for a in order]);return max(order,key=lambda a:(validation[a]['macro'],-order.index(a)))


def freeze():
    check_prepared()
    if (OUT/'selection/final_frozen.json').exists():raise FileExistsError('freeze already exists')
    validation={a:aggregate(a) for a in ARMS};validation['B_A020']=read(BASE/'final_review/results.json')['validation']['D_A020']
    final=nominate(validation);units={k:v for a in ARMS for k,v in validation[a]['units'].items()}
    if len(units)!=70:raise ValueError('all70 BEST required')
    refs={}
    for task in TASKS:
        for fold in range(5):
            refs[f'{task}/{fold}']=dict(validation=str(base_path(task,fold)/'validation_predictions.npz'),test=str(base_path(task,fold,True)/'test_predictions.npz'),
                scaler=read(base_path(task,fold)/'scaler.json'),best=str(base_path(task,fold)/'best.pt'))
    exclusive(OUT/'selection/validation_selection.json',dict(plan=PLAN,validation=validation,FINAL=final,selection='global validation macro; tie baseline,UNI3M,TMOL3M'))
    frozen=dict(plan=PLAN,revision=REVISION,frozen=True,time=now(),FINAL=final,units=units,baseline_units=refs,
        inputs=read(prepared_root()/'inputs/metadata.json'),geometry_missing=read(prepared_root()/'inputs/geometry_missing.json'),
        splits={task:read(SPLIT/f'{task}.json') for task in TASKS})
    exclusive(OUT/'selection/final_frozen.json',frozen);return frozen


def comparisons(formal,final):
    specs=[('UNI3M_minus_B_A020','UNI3M','B_A020'),('TMOL3M_minus_B_A020','TMOL3M','B_A020'),('TMOL3M_minus_UNI3M','TMOL3M','UNI3M')]
    if final:specs.append(('FINAL_minus_B_A020',final,'B_A020'))
    result={}
    for name,a,b in specs:
        if a not in formal or b not in formal:result[name]=dict(status='未完成／不可比较',delta=None);continue
        paired={}
        for task in TASKS:
            x=finite(formal[a]['tasks'][task]['folds'])-finite(formal[b]['tasks'][task]['folds'])
            paired[task]=dict(folds=x.tolist(),mean=float(x.mean()),std=float(x.std(ddof=1)),improved_folds=int((x>0).sum()))
        result[name]=dict(status='COMPLETE',delta=float(formal[a]['macro']-formal[b]['macro']),tasks=paired,improved_folds=sum(v['improved_folds'] for v in paired.values()))
    return result


def window_objective(result,weights,denominators):
    if set(result['sums'])!=set(weights) or set(weights)!=set(denominators):raise ValueError('objective sum/count keys mismatch')
    return sum(weights[key]*value.float()/denominators[key].clamp_min(1) for key,value in result['sums'].items())
