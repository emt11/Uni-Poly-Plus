"""Fixed UA3D-SPATIAL r4 execution, FULL and frozen PT-3D evaluation."""
import datetime
import json
import math
import os
from pathlib import Path
import pickle
import time
import traceback
import numpy as np
import torch
from src.modules import ua3d_spatial as m
from src.dataset import ua3d_spatial as d
from src.training.ua3m_paperfuse import (read,write,exclusive,now,ignore_termination,seed_all,
    finite,metrics,process_identity,process_alive,scaler_validate,transform,summary,SOTA,
    require_tmux,save_checkpoint,pretrain_source_fields,pt_learning_rate,window_objective,save_predictions)

ROOT=Path(__file__).resolve().parents[2]
PLAN='UA3D-SPATIAL-20261025-01';REVISION='r4';EXECUTION_REVISION='r4'
OUT=ROOT/'results/ua3d_spatial_20261025/r4';LOG=ROOT/'logs/ua3d_spatial_20261025/r4'
PARENT=ROOT/'results/ua3m_paperfuse_20261025/r1/repairs/gpu0123_restart1'
BASE=ROOT/'results/ua2d_p10k_nofp_align_20261024/r1'
DOWN=ROOT/'data/processed/mcl_ph_r4_20260930/downstream_direct_01.staging'
SPLIT=ROOT/'data/splits/mips_paper5_outer'
PT_CACHE=ROOT/'results/ua2d_pretrain_fp_20261019/fast_r4/prepared_cache'
MAP_ROOT=ROOT/'results/ua2d_ptscale_20261022/r1/inputs'
INITIAL=ROOT/'results/ua2d_pretrain_fp_20261019/fast_r4/initialization/common_step0.pt'
ARMS=m.ARMS;TASKS=m.TASKS;ALLOWED_GPUS=(0,1,2,3)
LIMITS=dict(prepare_starts=1,cache_rows=5875,pt_starts=4,pt_updates=40000,pt_microbatches=240000,pt_presentations=40320000,
    ft_starts=140,ft_epochs=9800,ft_updates=224000,probe_starts=140,probe_epochs=4200,probe_updates=96000,
    feature_rows=94000,test_starts=280,test_graphs=47000,cpu_suites=1)
ZERO_COST=dict(fp_generation=0,geometry_generation=0,kernel_fit=0,fp_prediction=0,output_fusion=0,additional_seeds=0)


def configure_execution(root=None):
    if root is not None and Path(root).resolve()!=OUT:raise ValueError('unauthorized spatial root')


def parent_path(arm):return OUT/'pretrain'/arm/'deploy.pt'
def modal_path(arm):return PARENT/'initialization/TMOL3M_modal.pt'
def head_path(task,fold):return PARENT/'initialization/heads'/task/f'fold{fold}.pt'
def prepared_root():return PARENT
def base_path(task,fold,formal=False):return BASE/('formal' if formal else 'finetune')/'D_A020'/task/f'fold{fold}'
def unit_path(arm,task,fold,formal=False,mode='FULL'):return OUT/('formal' if formal else 'finetune')/mode/arm/task/f'fold{fold}'


def config(arm,task,fold,mode='FULL'):
    if arm not in ARMS or task not in TASKS or type(fold) is not int or fold not in range(5) or mode not in ('FULL','probe'):raise ValueError('outside r4 matrix')
    return dict(plan=PLAN,revision=REVISION,arm=arm,mode=mode,task=task,fold=fold,seed=42+fold,parent=str(parent_path(arm)),
        head=str(head_path(task,fold)),input_cache=str(PARENT/'inputs'),sidecar=str(OUT/'inputs'),
        scaler_reference=str(base_path(task,fold)/'scaler.json'),model=m.CONFIG,world_size=1,batch=24,
        epochs=70 if mode=='FULL' else 30,head_epochs=10 if mode=='FULL' else 30,precision='FP32',workers=0,
        loss='MSE+0.08*ALIGN_TMOL-0.08*gate_entropy' if mode=='FULL' else 'MSE',selection='strict original-unit validation RMSE')


def update_plan(status,phase,detail):
    p=ROOT/'CURRENT_PLAN.md';s=p.read_text();lines=s.splitlines();lines[2]=f'> 计划 ID：{PLAN}；修订：r4；状态：**{status}**。';s='\n'.join(lines)+'\n'
    tag='\n<!-- UA3D_SPATIAL_STATUS_START -->';end='\n<!-- UA3D_SPATIAL_STATUS_END -->'
    block=tag+f'\n- {now()}：状态**{status}**；阶段{phase}。用户“请你直接执行！”授权r4；Codex按明确要求直接执行。\n- {detail}\n- 账本：`{OUT}/budget.json`；日志：`{LOG}`；独立审查PENDING，未归档或替换部署。'+end
    if tag in s:
        left,rest=s.split(tag,1);_,right=rest.split(end,1);s=left+block+right
    else:s=s.rstrip()+block+'\n'
    p.write_text(s);write(OUT/'status.json',dict(status=status,phase=phase,detail=detail,time=now(),independent_review='PENDING'))


def prepare():
    from src.dataset.mcl_ph_direct import DirectMCLSource
    from src.dataset.ua3m_paperfuse import InputCache,validate_missing_inventory
    require_tmux();torch.set_num_threads(2)
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU prepare required')
    folder=OUT/'inputs';folder.mkdir(exist_ok=False);started=time.monotonic();cache=None
    state=dict(status='RUNNING',records=0,geometry_available=0,geometry_missing=0,directed_edges=0,
        valid_angle_atoms=0,atoms=0,seam_edges=0,no_edge_graphs=0,max_neighbors=0,pid=os.getpid(),process=process_identity(os.getpid()),started=now())
    def record():write(folder/'coverage.json',dict(state,elapsed_seconds=time.monotonic()-started))
    record()
    try:
        cache=InputCache(PARENT/'inputs');validate_missing_inventory(read(PARENT/'inputs/geometry_missing.json'))
        pt=read(PT_CACHE/'metadata.json');mapping=read(MAP_ROOT/'metadata.json')
        if pt.get('schema')!='ua2d_prepared_templates_r4' or pt.get('status')!='COMPLETE' or pt.get('train_rows')!=911391 or mapping.get('records')!=910756 or mapping.get('status')!='COMPLETE' or mapping.get('eligibility_semantics')!='logical_slot_not_source_row':raise ValueError('PT input metadata mismatch')
        sidecar={};splits={}
        for task in TASKS:
            split=read(SPLIT/f'{task}.json');splits[task]=split
            if split!=read(PARENT/'inputs/original_splits.json')[task]:raise ValueError('split changed')
            source=DirectMCLSource(DOWN,task=task,selected_indices=None,split_manifest=split,verify_key_digest=False)
            try:
                for i,entry in enumerate(source.entries):
                    row=int(entry['original_row']);key=source.samples[i][0].hex();sm=entry['normalized_smiles'];cache_key=f'{task}:{row}'
                    x=cache.get(task,row,key,sm);available=entry['geometry_status']=='PASS'
                    if available!=x['available']:raise ValueError('missing inventory changed')
                    fields=None
                    if available:
                        topology,trimer,text,static,_,identity=pretrain_source_fields(source,i,include_identity=True)
                        a=torch.as_tensor(np.array(static['token_pos_index_a'],copy=True)).long();b=torch.as_tensor(np.array(static['token_pos_index_b'],copy=True)).long()
                        g=x['three_d'];pos=trimer.trimer_pos.float()
                        if len(a)!=len(g.bond_distance) or not torch.equal(torch.as_tensor(static['token_z_a']).long(),g.bond_z_a.long()) or not torch.equal(torch.as_tensor(static['token_z_b']).long(),g.bond_z_b.long()):raise ValueError('sidecar original token identity mismatch')
                        distance=(pos[a]-pos[b]).norm(dim=-1)
                        if not torch.allclose(distance,g.bond_distance.float(),atol=1e-5,rtol=1e-5):raise ValueError('sidecar/cached geometry mismatch')
                        top=d.physical_topology(torch.stack((a,b)),g.bond_z_a,g.bond_z_b,len(pos),trimer.trimer_atomic_number,trimer.trimer_ru_offset)
                        fields=d.spatial_fields(pos,top);d.attach(g,fields);cov=d.coverage_of(g)
                        for k in ('atoms','directed_edges','valid_angle_atoms','seam_edges','no_edge_graphs'):state[k]+=cov[k]
                        state['max_neighbors']=max(state['max_neighbors'],cov['max_neighbors']);state['geometry_available']+=1
                    else:state['geometry_missing']+=1
                    if cache_key in sidecar:raise ValueError('duplicate sidecar row')
                    sidecar[cache_key]=dict(key=key,smiles=sm,available=available,fields=fields);state['records']+=1
                    if state['records']%128==0:record();print(json.dumps(state),flush=True)
                cache.memory.clear()
            finally:source.close()
        if (state['records'],state['geometry_available'],state['geometry_missing'])!=(5875,5860,15):raise ValueError('sidecar coverage mismatch')
        with (folder/'sidecar.pkl').open('xb') as f:pickle.dump(sidecar,f,protocol=pickle.HIGHEST_PROTOCOL)
        write(folder/'original_splits.json',splits)
        state.update(status='COMPLETE',ended=now());record()
        write(folder/'metadata.json',dict(schema=d.SCHEMA,plan=PLAN,revision=REVISION,status='COMPLETE',records=5875,
            geometry_available=5860,geometry_missing=15,parent=str(PARENT/'inputs'),source=str(DOWN),model=m.CONFIG,
            targets_cached=False,fp_generation=0,geometry_generation=0,model_forward=0))
    except BaseException:
        ignore_termination();state.update(status='FAILED',error=traceback.format_exc(),ended=now());record();raise
    finally:
        if cache is not None:cache.close()


def check_prepared():
    meta=read(OUT/'inputs/metadata.json');cov=read(OUT/'inputs/coverage.json')
    if meta.get('schema')!=d.SCHEMA or meta.get('status')!='COMPLETE' or cov.get('status')!='COMPLETE' or tuple(cov[k] for k in ('records','geometry_available','geometry_missing'))!=(5875,5860,15):raise ValueError('prepare incomplete')
    for path in (INITIAL,INITIAL.parent/'task_heads/V1.pt',modal_path('BASE'),OUT/'inputs/sidecar.pkl'):
        if not path.is_file():raise FileNotFoundError(path)
    splits=read(OUT/'inputs/original_splits.json')
    if any(splits[task]!=read(SPLIT/f'{task}.json') for task in TASKS):raise ValueError('frozen split mismatch')


def frozen_unit(frozen,arm,task,fold,mode='FULL'):
    if frozen.get('plan')!=PLAN or frozen.get('revision')!=REVISION or frozen.get('frozen') is not True:raise ValueError('TEST before valid global freeze')
    chosen=frozen['units'][f'{mode}/{arm}/{task}/{fold}']
    if chosen['config']!=config(arm,task,fold,mode) or chosen['best']!=str(unit_path(arm,task,fold,mode=mode)/'best.pt'):raise ValueError('frozen estimator mismatch')
    return chosen


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
            model=m.SpatialFuse(arm,False)
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



def extract_features(model,ds,indices,path,state,formal=False):
    if Path(path).exists():raise FileExistsError('feature cache exists; no extraction retry')
    z=[];seen=[];availability=[]
    model.eval()
    with torch.inference_mode():
        for data,rows,entries in d.geometry_loader(ds,indices):
            count=len(entries)
            output=model(data.to('cuda:0') if data is not None else None,rows.to('cuda:0'),count)
            if not torch.isfinite(output).all():raise FloatingPointError('nonfinite frozen PT features')
            z.append(output.cpu().numpy());seen.extend(entries);available=np.zeros(count,dtype=bool);available[rows.numpy()]=True;availability.extend(available.tolist())
            state['test_feature_rows' if formal else 'feature_rows']+=count
    if seen!=list(indices):raise ValueError('feature ordering mismatch')
    matrix=np.concatenate(z);available=np.asarray(availability,dtype=bool)
    if matrix.shape!=(len(indices),256) or np.any(matrix[~available]!=0):raise ValueError('missing 3D feature not exactly zero')
    # Cache contains only frozen features and identities, never labels.
    with Path(path).open('xb') as f:np.savez(f,z3=matrix,indices=np.array([ds.indices[i] for i in indices]),sample_keys=ds.keys[indices],availability=available,endpoint=str(parent_path(state['arm'])))
    return torch.from_numpy(matrix)


def probe_unit(arm,task,fold,formal=False,physical_gpu=3):
    require_tmux();check_prepared();torch.set_num_threads(2)
    if physical_gpu not in ALLOWED_GPUS or os.environ.get('CUDA_VISIBLE_DEVICES')!=str(physical_gpu) or os.environ.get('WORLD_SIZE','1')!='1':raise ValueError('probe allocated physical GPU/world1 mismatch')
    cfg=config(arm,task,fold,'probe');folder=unit_path(arm,task,fold,formal,'probe');folder.mkdir(parents=True,exist_ok=False)
    write(folder/'config.json',cfg);ds=None;started=time.monotonic()
    state=dict(status='RUNNING',arm=arm,pid=os.getpid(),process=process_identity(os.getpid()),physical_gpu=physical_gpu,started=now(),
        epochs_started=0,completed_epochs=0,attempted_updates=0,completed_updates=0,probe_epochs=0,probe_updates=0,
        feature_rows=0,test_feature_rows=0,train_graphs=0,validation_graphs=0,validation_started=0,test_graphs=0,test_started=0)
    def record():write(folder/'consumption.json',state)
    record()
    try:
        seed_all(42+fold);scaler=read(base_path(task,fold)/'scaler.json')
        ds=d.Data(task,fold,formal,DOWN,SPLIT,PARENT/'inputs')
        if formal:
            selected=frozen_unit(read(OUT/'selection/final_frozen.json'),arm,task,fold,'probe')
            if selected['test_indices']!=ds.indices:raise ValueError('frozen TEST feature indices mismatch')
            saved=torch.load(selected['best'],map_location='cpu',weights_only=False)
            if saved.get('schema')!=m.PROBE_SCHEMA or saved.get('config')!=cfg or saved.get('scaler')!=scaler or saved.get('best_epoch')!=selected['best_epoch']:raise ValueError('frozen probe checkpoint mismatch')
            head=m.property_head(.5);m.strict_module_state(head,saved['state_dict']);best_epoch=saved['best_epoch'];del saved
        else:
            scaler_validate(ds.raw_targets,ds.train,scaler)
            with torch.random.fork_rng(devices=[]):head=m.property_head(.5)
            m.strict_module_state(head,torch.load(head_path(task,fold),map_location='cpu',weights_only=False))
        package=torch.load(parent_path(arm),map_location='cpu',weights_only=False)
        if package.get('arm')!=arm:raise ValueError('requested probe arm and PT endpoint differ')
        encoder=m.Frozen3D(package).to('cuda:0');del package
        if any(p.requires_grad for p in encoder.parameters()) or encoder.training:raise ValueError('probe encoder not frozen')
        if formal:state['test_started']=1;record()
        features=extract_features(encoder,ds,list(range(len(ds))),folder/'features.npz',state,formal)
        del encoder;torch.cuda.empty_cache();record()
        ds.targets=transform(ds.raw_targets,scaler);labels=torch.tensor(ds.targets,dtype=torch.float32)
        write(folder/'scaler.json',scaler);write(folder/'parameters.json',dict(trainable_head=sum(p.numel() for p in head.parameters()),frozen_encoder=True,dimension=256))
        def evaluate():
            head.eval();pred=[]
            with torch.inference_mode():
                for start in range(0,len(ds.validation),24):
                    ids=ds.validation[start:start+24];pred.append(head(features[ids]).view(-1).numpy())
                    state['test_graphs' if formal else 'validation_graphs']+=len(ids)
            truth=ds.raw_targets[ds.validation];prediction=transform(np.concatenate(pred),scaler,True)
            return truth,prediction,metrics(truth,prediction)
        if formal:
            y,p,result=evaluate();save_predictions(folder/'test_predictions.npz',ds,y,p,best_epoch=best_epoch,best_stage='probe')
        else:
            seed_all(42+fold);generator=torch.Generator().manual_seed(42+fold)
            train=torch.utils.data.DataLoader(torch.tensor(ds.train),batch_size=24,shuffle=True,generator=generator,num_workers=0)
            opt=torch.optim.AdamW(head.parameters(),lr=.001,weight_decay=.02);best_rmse=math.inf;history=[]
            for epoch in range(1,31):
                tick=time.monotonic();head.train();losses=[];state['epochs_started']+=1;state['probe_epochs']+=1;record()
                for ids in train:
                    opt.zero_grad(set_to_none=True);loss=torch.nn.functional.mse_loss(head(features[ids]).view(-1),labels[ids])
                    if not torch.isfinite(loss):raise FloatingPointError('nonfinite probe loss')
                    loss.backward();torch.nn.utils.clip_grad_norm_(head.parameters(),5.,error_if_nonfinite=True)
                    state['attempted_updates']+=1;state['probe_updates']+=1;state['train_graphs']+=len(ids)
                    opt.step();state['completed_updates']+=1;losses.append(float(loss.detach()))
                state['validation_started']+=1;y,p,met=evaluate();state['completed_epochs']+=1
                if met['rmse']<best_rmse:
                    best_rmse=met['rmse'];best_epoch=epoch;result=met
                    save_checkpoint(folder/'best.pt',dict(schema=m.PROBE_SCHEMA,config=cfg,state_dict=head.state_dict(),best_epoch=epoch,best_stage='probe',scaler=scaler,outer_test='NOT_RUN'))
                    save_predictions(folder/'validation_predictions.npz',ds,y,p,best_epoch=epoch,best_stage='probe')
                rec=dict(epoch=epoch,train_loss=float(np.mean(losses)),validation=met,seconds=time.monotonic()-tick,learning_rate=.001)
                history.append(rec);write(folder/'history.json',history);record();print(json.dumps(rec),flush=True)
        write(folder/'metrics.json',dict(status='COMPLETE',config=cfg,best_epoch=best_epoch,best_stage='probe',scaler=scaler,**result))
        state.update(status='COMPLETE',ended=now(),elapsed_seconds=time.monotonic()-started);record()
    except BaseException:
        ignore_termination();state.update(status='FAILED',error=traceback.format_exc(),ended=now());record();write(folder/'failure.json',dict(error=state['error'],automatic_retry=False));raise
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
    if set(validation)!=set(order):raise ValueError('complete validation candidates required')
    finite([validation[a]['macro'] for a in order]);return max(order,key=lambda a:(validation[a]['macro'],-order.index(a)))


def freeze():
    check_prepared()
    if (OUT/'selection/final_frozen.json').exists():raise FileExistsError('global freeze already exists')
    validation={mode:{a:aggregate(a,mode=mode) for a in ARMS} for mode in ('FULL','probe')}
    final=nominate(validation['FULL']);g=nominate({a:validation['FULL'][a] for a in ARMS[2:]},ARMS[2:])
    units={k:v for mode in validation for a in ARMS for k,v in validation[mode][a]['units'].items()}
    if len(units)!=280:raise ValueError('all 280 estimators required before freeze')
    exclusive(OUT/'selection/validation_selection.json',dict(plan=PLAN,validation=validation,FINAL=final,G=g,selection='FULL global validation macro, declared exact tie order'))
    frozen=dict(plan=PLAN,revision=REVISION,frozen=True,time=now(),FINAL=final,G=g,units=units,inputs=read(OUT/'inputs/metadata.json'),splits={task:read(SPLIT/f'{task}.json') for task in TASKS})
    exclusive(OUT/'selection/final_frozen.json',frozen);return frozen


def comparisons(formal,g):
    result={}
    for mode,values in formal.items():
        specs=[('C_CONST_minus_BASE','C_CONST','BASE'),('distance','S_RAD','C_CONST'),('angle','S_RADANG','S_RAD')]
        if g:specs=[('G_minus_BASE',g,'BASE'),('G_minus_C_CONST',g,'C_CONST')]+specs
        result[mode]={}
        for name,a,b in specs:
            if a not in values or b not in values:result[mode][name]=dict(status='未完成／不可比较',delta=None,performance_pass=None);continue
            paired={}
            for task in TASKS:
                diff=finite(values[a]['tasks'][task]['folds'])-finite(values[b]['tasks'][task]['folds'])
                paired[task]=dict(folds=diff.tolist(),mean=float(diff.mean()),std=float(diff.std(ddof=1)),improved_folds=int((diff>0).sum()),degraded_folds=int((diff<0).sum()))
            delta=float(values[a]['macro']-values[b]['macro']);result[mode][name]=dict(status='COMPLETE',a=a,b=b,delta=delta,performance_pass=delta>=.010,tasks=paired)
    return result
