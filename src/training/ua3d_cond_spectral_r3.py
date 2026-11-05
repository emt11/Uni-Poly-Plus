"""r2 exact fixed-cycle accounting, fresh PT and joint FT; no historical weights."""
import copy,json,math,os,time,traceback
from pathlib import Path
import numpy as np
import torch
from src.modules import ua3d_cond_spectral_r3 as m
from src.dataset import ua3d_cond_spectral_r3 as d
from src.training.ua3m_paperfuse import (read,write,exclusive,now,ignore_termination,seed_all,finite,metrics,process_identity,scaler_validate,transform,summary,SOTA,require_tmux,save_checkpoint,pt_learning_rate,window_objective,save_predictions)
ROOT=Path(__file__).resolve().parents[2]
PLAN=m.PLAN;REVISION='r3';EXECUTION_REVISION='gpu2-pt1-fttest4-pss-boundedio-r3'
PARENT=ROOT/'results/ua3d_cond_spectral_20261104/r2'
OUT=PARENT/'repairs/memory_r3';LOG=ROOT/'logs/ua3d_cond_spectral_20261104/r2/repairs/memory_r3'
B=ROOT/'results/ua3d_base_opt_20261026/r6/repairs/concurrency_r1'
SURFACE=ROOT/'results/ua3d_hmr_20261103/r1/repairs/schema_surface_r2'
DOWN=ROOT/'data/processed/mcl_ph_r4_20260930/downstream_direct_01.staging'
SPLIT=ROOT/'data/splits/mips_paper5_outer'
PT_CACHE=ROOT/'results/ua2d_pretrain_fp_20261019/fast_r4/prepared_cache'
MAP_ROOT=ROOT/'results/ua2d_ptscale_20261022/r1/inputs'
INITIAL=OUT/'initialization/common_step0.pt'
TASKS=m.TASKS;ARMS=m.ARMS;ALLOWED_GPUS=(2,);REFERENCE=.804675462276949
UUID='GPU-31456dd5-a421-142b-644b-7a80f482dfb9'

def save_checkpoint(path,payload):
    # Runtime helper does not create nested directories; explicitly contain writes.
    path=Path(path)
    if not path.is_relative_to(OUT):raise ValueError('checkpoint outside new root')
    path.parent.mkdir(parents=True,exist_ok=True)
    from src.training.glt_dual_runtime import save_checkpoint as save
    save(path,payload)

def head_path(task,fold):return OUT/'initialization/heads'/task/f'fold{fold}.pt'
def base_path(task,fold):return B/'finetune/F_PD_N1_F0'/task/f'fold{fold}'
def parent_path(arm):return OUT/'pretrain/BASE_FRESH/deploy.pt'
def unit_path(arm,task,fold,formal=False):return OUT/('formal' if formal else 'finetune')/arm/task/f'fold{fold}'
def source_config(task,fold):return read(base_path(task,fold)/'config.json')
def config(arm,task,fold):
    if arm not in ARMS or task not in TASKS or fold not in range(5):raise ValueError('matrix identity')
    src=source_config(task,fold)
    return dict(plan=PLAN,revision=REVISION,science_recipe='r2',arm=arm,task=task,fold=fold,seed=42+fold,parent=str(parent_path(arm)),head=str(head_path(task,fold)),input_cache=src['input_cache'],scaler_reference=src['scaler_reference'],model=m.CONFIG,batch=24,epochs=70,workers=4,world_size=1)
def update_plan(status,phase,detail):
    path=ROOT/'CURRENT_PLAN.md';s=path.read_text();start='<!-- CONDSPEC_R3_STATUS_START -->';end='<!-- CONDSPEC_R3_STATUS_END -->'
    left,rest=s.split(start,1);_,right=rest.split(end,1)
    path.write_text(left+start+'\n'+f'{now()} 状态 {status}；阶段 {phase}。{detail}\nindependent_review=PENDING；完整正式TEST前未完成/不可比较。\n'+end+right)
    write(OUT/'status.json',dict(status=status,phase=phase,detail=detail,time=now(),independent_review='PENDING'))
def check_clock(prepare=False):
    c=read(OUT/'authorization/clock.json');limit=c['prepare_deadline_unix'] if prepare else c['deadline_unix']
    if time.time()>=limit:raise TimeoutError('EXPIRED: inherited stage/total deadline')
def check_prepared():
    identity=read(OUT/'identity.json')
    if identity.get('plan')!=PLAN or identity.get('revision')!=REVISION or not INITIAL.is_file():raise ValueError('fresh initialization identity')
    for task in TASKS:
        if read(SPLIT/f'{task}.json')!=identity['splits'][task]:raise ValueError('split changed')
    check_clock()
def gpu_contract():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='2' or os.environ.get('WORLD_SIZE')!='1' or torch.cuda.device_count()!=1:raise ValueError('GPU2/world1')
    actual='GPU-'+str(torch.cuda.get_device_properties(0).uuid).removeprefix('GPU-')
    if actual!=UUID:raise ValueError('GPU UUID')
    return dict(physical_gpu=2,logical_device='cuda:0',uuid=actual)
def frozen_unit(frozen,arm,task,fold):
    if frozen.get('plan')!=PLAN or frozen.get('frozen') is not True:raise ValueError('TEST before freeze')
    chosen=frozen['units'][f'{arm}/{task}/{fold}']
    if chosen['config']!=config(arm,task,fold):raise ValueError('frozen estimator')
    return chosen

def unit(arm,task,fold,formal=False,physical_gpu=2):
    require_tmux()
    if physical_gpu not in ALLOWED_GPUS or os.environ.get('CUDA_VISIBLE_DEVICES')!=str(physical_gpu) or os.environ.get('WORLD_SIZE','1')!='1':raise ValueError('allocated physical GPU/world1 mismatch')
    install_stop_handler();torch.set_num_threads(2);check_prepared();cfg=config(arm,task,fold)
    folder=unit_path(arm,task,fold,formal);folder.mkdir(parents=True,exist_ok=False);write(folder/'config.json',cfg)
    state=dict(status='RUNNING',pid=os.getpid(),process=process_identity(os.getpid()),started=now(),
        epochs_started=0,completed_epochs=0,attempted_updates=0,completed_updates=0,train_graphs=0,
        validation_started=0,validation_graphs=0,test_graphs=0,test_started=0,physical_gpu=physical_gpu,gpu_uuid=os.environ.get('UA3M_PHYSICAL_GPU_UUID'),allowed_gpus=list(ALLOWED_GPUS),resource_revision=EXECUTION_REVISION,phase='initialize')
    def record():write(folder/'consumption.json',state)
    record();ds=None;started=time.monotonic()
    try:
        state['actual_device']=gpu_contract();record()
        scaler=read(cfg['scaler_reference'])
        if formal:
            frozen=read(OUT/'selection/final_frozen.json');selected=frozen_unit(frozen,arm,task,fold)
            saved=torch.load(selected['best'],map_location='cpu',weights_only=False)
            if any(saved[k]!=selected[k] for k in ('best_epoch','best_stage','scaler')):raise ValueError('frozen checkpoint/scaler mismatch')
            if scaler!=saved['scaler']:raise ValueError('BEST scaler mismatch')
            model=m.Model(arm,fold)
            if saved.get('schema')!=m.FT_SCHEMA or saved.get('config')!=cfg:raise ValueError('FT checkpoint schema/config mismatch')
            m.strict_module_state(model,saved['state_dict']);best_epoch=saved['best_epoch'];best_stage=saved['best_stage'];del saved
        else:
            seed_all(42+fold)
            package=torch.load(parent_path(arm),map_location='cpu',weights_only=False)
            head=torch.load(head_path(task,fold),map_location='cpu',weights_only=False)
            if head.get('task')!=task or head.get('seed')!=2026110400+10*TASKS.index(task)+fold:raise ValueError('head task/seed identity')
            model=m.initialize_ft(package,head,arm,fold);del package,head
        ds=d.Data(task,fold,formal,DOWN,SPLIT,cfg['input_cache'],SURFACE,read(PARENT/'scalers'/task/f'fold{fold}.json'))
        if formal:
            if selected['test_indices']!=ds.indices or selected['test_keys']!=ds.keys.tolist():raise ValueError('frozen TEST row/key/target mismatch')
        else:scaler_validate(ds.raw_targets,ds.train,scaler)
        ds.targets=transform(ds.raw_targets,scaler);write(folder/'scaler.json',scaler)
        model.to('cuda:0');validation=d.loader(ds,ds.validation)
        parameter_counts=dict(total=sum(p.numel() for p in model.parameters()),backbone=sum(p.numel() for n,p in model.named_parameters() if n.startswith(tuple(name+'.' for name in m.NATIVE))))
        write(folder/'parameters.json',parameter_counts)
        def evaluate():
            model.eval();truth=[];pred=[];diag_sum={};gate_count=0
            with torch.inference_mode():
                for batch in validation:
                    batch=batch.to('cuda:0',non_blocking=True);p,weights,_=model.predict(batch)
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
                    )
                for stage_epoch in range(epochs):
                    check_clock();torch.cuda.synchronize();tick=time.monotonic();state.update(phase=stage,epochs_started=state['epochs_started']+1,
                        epoch_open=True,epoch_expected_updates=len(train),epoch_expected_train=len(ds.train),epoch_expected_validation=len(ds.validation))
                    record();model.train();losses=[];component_sum={};training_diag={};train_count=0;rates=[g['lr'] for g in opt.param_groups]
                    runtime=model.runtime_contract()
                    for batch in train:
                        batch=batch.to('cuda:0',non_blocking=True);opt.zero_grad(set_to_none=True)
                        state['attempted_updates']+=1
                        if state['attempted_updates']>expected:raise RuntimeError('update allowance exhausted')
                        loss,diagnostics_train=model.training_loss(batch);output=batch.y.view(-1)
                        size=len(output);train_count+=size
                        for key,value in diagnostics_train.items():component_sum[key]=component_sum.get(key,0.)+float(value)*size
                        for key,value in model.last_diagnostics.items():training_diag[key]=training_diag.get(key,0)+value.detach().cpu().numpy()*size
                        if not torch.isfinite(loss):raise FloatingPointError('nonfinite loss')
                        loss.backward();check_time_gradient(model,state)
                        if hasattr(model,'surface'):component_sum['time_gradient_norm']=component_sum.get('time_gradient_norm',0.)+float(model.surface.time_tensor.grad.detach().norm())*size
                        torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True)
                        state['train_graphs']+=len(output)
                        state['train_3d_graphs']=state.get('train_3d_graphs',0)+int(batch.availability[:,1].sum())
                        opt.step();state['completed_updates']+=1;losses.append(float(loss.detach()))
                    torch.cuda.synchronize();train_seconds=time.monotonic()-tick;scheduler.step();state['validation_started']+=1
                    y,p,met,diag=evaluate();state['completed_epochs']+=1;epoch=state['completed_epochs']
                    if met['rmse']<best_rmse:
                        best_rmse=met['rmse'];best_epoch=epoch;best_stage=stage;result=met;diagnostics=diag
                        save_checkpoint(folder/'best.pt',dict(schema=m.FT_SCHEMA,science_recipe='r2',architecture=m.CONFIG,config=cfg,
                            state_dict={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                            best_epoch=best_epoch,best_stage=best_stage,scaler=scaler,outer_test='NOT_RUN'))
                        save_predictions(folder/'validation_predictions.npz',ds,y,p,best_epoch=best_epoch,best_stage=best_stage)
                    rec=dict(epoch=epoch,stage=stage,stage_epoch=stage_epoch+1,train_loss=float(np.mean(losses)),
                        validation=met,training_components={k:v/train_count for k,v in component_sum.items()},training_modality_diagnostics={k:np.asarray(v/train_count).tolist() for k,v in training_diag.items()},runtime_contract=runtime,learning_rates=rates,optimizer_groups=groups,modality_diagnostics=diag,
                        training_seconds=train_seconds,batches=len(train),seconds=time.monotonic()-tick,elapsed_seconds=time.monotonic()-train_tick,gpu_peak_bytes=torch.cuda.max_memory_allocated())
                    history.append(rec);write(folder/'history.json',history);print(json.dumps(rec),flush=True)
                    check_clock();state['epoch_open']=False;state['ft_epochs']=state['epochs_started'];state['ft_updates']=state['attempted_updates'];record()
            state['ft_epochs']=state['epochs_started'];state['ft_updates']=state['attempted_updates']
            if state['completed_epochs']!=70 or state['attempted_updates']!=expected or state['completed_updates']!=expected:
                raise ValueError('incomplete FT trajectory')
        if formal:
            write(folder/'identity_alignment.json',dict(indices=ds.indices,sample_keys=ds.keys.tolist(),raw_targets=ds.raw_targets.tolist(),
                transformed_targets=ds.targets.tolist(),frozen_source='original split row/key; no TEST target read before freeze',scaler=scaler,matched=True))
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

def check_time_gradient(model,state):
    if not hasattr(model,'surface'):return
    t=model.surface.time_tensor
    if not t.requires_grad or t.grad_fn is None or t.grad is None:raise RuntimeError('diffusion time gradient disconnected')
    if not torch.isfinite(t.grad).all():raise FloatingPointError('nonfinite time gradient')
    model.last_diagnostics['time_gradient_norm']=t.grad.detach().norm()
    if state['phase']=='joint' and 'STATIC' not in model.recipe_arm:
        if not model.surface.condition[-1].weight.requires_grad:raise RuntimeError('condition gradient disconnected')
        if not any(p.grad is not None for n,p in model.named_parameters() if n.startswith(('o8.','encoder3d.'))):raise RuntimeError('BASE joint gradient disconnected')

def freeze():
    check_prepared();validation={a:aggregate(a) for a in ARMS};control=validation['C_FRESH']
    eligible=[a for a in ARMS[1:] if validation[a]['macro']>control['macro'] and all(validation[a]['tasks'][task]['mean']>=control['tasks'][task]['mean']-.010 for task in TASKS)]
    chosen=max(eligible,key=lambda a:(validation[a]['macro'],-ARMS.index(a))) if eligible else 'C_FRESH'
    units={k:v for a in ARMS for k,v in validation[a]['units'].items()}
    for a in ARMS:
        for task in TASKS:
            rows=read(Path(source_config(task,0)['input_cache'])/'rows.json')[task];keys={e['row']:e['key'] for e in rows}
            for fold in range(5):
                unit=units[f'{a}/{task}/{fold}'];unit['test_keys']=[keys[i] for i in unit['test_indices']]
    exclusive(OUT/'selection/final_frozen.json',dict(plan=PLAN,revision=REVISION,frozen=True,time=now(),frozen_unix=time.time(),FINAL=chosen,eligible=eligible,validation=validation,units=units,reference=REFERENCE,test_targets_read=False,rule='strict macro > C_FRESH; every task >= C_FRESH-.010; registration-order ties; otherwise C_FRESH'))

def report_results():
    frozen=read(OUT/'selection/final_frozen.json');formal={a:aggregate(a,True) for a in ARMS};final=formal[frozen['FINAL']];control=formal['C_FRESH']
    value=dict(plan=PLAN,revision=REVISION,FINAL=frozen['FINAL'],formal=formal,macro=final['macro'],delta_BASE=final['macro']-REFERENCE,delta_C_FRESH=final['macro']-control['macro'],target_met=final['macro']>=.810,performance_pass=final['macro']-REFERENCE>=.010,independent_review='PENDING',evaluation='shared development',sota_source='SOTA_RESULT.md',sota=SOTA)
    value['paired']={key:formal[a]['macro']-formal[b]['macro'] for key,a,b in [('conditional_gate','S_COND_GATE','S_STATIC_GATE'),('conditional_token','S_COND_TOKEN','S_STATIC_TOKEN'),('cross_readout','S_COND_GATE','S_COND_NOCROSS')]}
    value['paired']['interaction']=value['paired']['conditional_token']-value['paired']['conditional_gate']
    value['paper_comparison']={task:dict(project_mean=final['tasks'][task]['mean'],project_std=final['tasks'][task]['std'],paper_model=SOTA[task][1],paper_mean=SOTA[task][2],delta=final['tasks'][task]['mean']-SOTA[task][2],protocol='five outer folds seed42+fold; paper protocol equivalence unverified') for task in TASKS}
    write(OUT/'final_review/results.json',value)
    lines=['# Formal shared-development results',f"FINAL={frozen['FINAL']}; macro={final['macro']:.3f}; ΔBASE={value['delta_BASE']:+.3f}; ΔC_FRESH={value['delta_C_FRESH']:+.3f}.",'Paper values form a per-task envelope, not a single model; numerical comparison only.','','|Task|Project mean ± std|Paper model|Paper mean|Δ|','|---|---:|---|---:|---:|']
    for task,c in value['paper_comparison'].items():lines.append(f"|{task}|{c['project_mean']:.3f} ± {c['project_std']:.3f}|{c['paper_model']}|{c['paper_mean']:.3f}|{c['delta']:+.3f}|")
    (OUT/'final_review/comparison.md').write_text('\n'.join(lines)+'\n')
    return value

def pt_train():
    import multiprocessing as mp
    from src.training.glt_dual_runtime import rng_state
    require_tmux();install_stop_handler();torch.set_num_threads(2);gpu_contract();check_prepared();seed_all(42)
    folder=OUT/'pretrain/BASE_FRESH';folder.mkdir(parents=True,exist_ok=False)
    success=False;started=time.time();pt_deadline=read(OUT/'authorization/clock.json')['pt_deadline_unix'];write(folder/'clock.json',dict(new_pt_started_unix=started,deadline_unix=pt_deadline,parent_started_unix=read(OUT/'authorization/clock.json')['parent_pt_started_unix']))
    model=None;iterator=None;attempted=completed=consumed=0;durations=[]
    context=mp.get_context('spawn');prepared=context.Array('q',4)
    prefix=[]
    with (PARENT/'pretrain/BASE_FRESH/sampling.jsonl').open() as f:
        for line in f:
            try:prefix.append(json.loads(line))
            except json.JSONDecodeError:break
    if len(prefix)!=5148:raise ValueError('parent complete sampling prefix boundary mismatch')
    prefix_checked=0;worker_io={};wait_seconds=0.;last_window_started=time.monotonic()
    def record(status):
        with prepared.get_lock():count=int(sum(prepared[:]))
        write(folder/'consumption.json',dict(status=status,attempted_updates=attempted,completed_updates=completed,pt_microbatches=consumed//168,pt_presentations=consumed,prefetched_presentations=count,prefetched_not_consumed=count-consumed,stream=model.stream_state() if model is not None else None,worker_io=worker_io,prefix_checked=prefix_checked,prefix_boundary=len(prefix),data_wait_seconds=wait_seconds,elapsed_seconds=time.time()-started,historical_tensor_reads=0))
    def package(step):
        return dict(schema=m.PT_SCHEMA,plan=PLAN,revision=REVISION,step=step,science_recipe='r2',config=m.CONFIG,fp_prediction=False,historical_tensor_reads=0,stream=model.stream_state(),state_dict={k:v.detach().cpu().clone() for k,v in model.state_dict().items()})
    record('RUNNING')
    try:
        model=m.FreshPT();common=torch.load(INITIAL,map_location='cpu',weights_only=False);m.validate_package(common,0)
        m.strict_module_state(model,common['state_dict']);del common
        model.to('cuda:0');seed_all(42)
        opt=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=0.,betas=(.9,.999),eps=1e-8)
        ds=d.DecodedPT(PT_CACHE,MAP_ROOT,prepared)
        loader=torch.utils.data.DataLoader(ds,batch_size=None,num_workers=4,prefetch_factor=1,multiprocessing_context=context,worker_init_fn=d.worker_init,collate_fn=d.pt_collate,pin_memory=True,generator=torch.Generator().manual_seed(42))
        iterator=iter(loader)
        write(folder/'config.json',dict(plan=PLAN,revision=REVISION,model=m.CONFIG,pt=m.PT,microbatch=168,accumulation=6,updates=10000,workers=4,main_threads=2,precision='BF16 forward FP32 loss optimizer',initialization=str(INITIAL)))
        with (folder/'sampling.jsonl').open('x') as sampling,(folder/'history.jsonl').open('x') as history:
            for update in range(10000):
                check_clock()
                if time.time()>=pt_deadline:raise TimeoutError('PT 12h deadline')
                tick=time.monotonic();attempted+=1;model.completed_updates=update;record('RUNNING')
                weights=model.weights();totals={k:torch.zeros(()) for k in weights};batches=[]
                for micro in range(6):
                    wait_tick=time.monotonic();item=next(iterator);wait_seconds+=time.monotonic()-wait_tick;data,labels,trace=item.data,item.labels,item.trace
                    idx=trace['micro_index']
                    if idx<len(prefix):
                        if any(trace[k]!=prefix[idx][k] for k in ('micro_index','start_position','logical_slots','source_rows')):raise ValueError('normal parent sampling prefix identity mismatch')
                        prefix_checked+=1
                    worker_io[str(trace['io']['worker'])]=trace['io']
                    if trace['micro_index']!=update*6+micro or trace['start_position']!=(update*6+micro)*168:raise ValueError('PT ordered position mismatch')
                    consumed+=168;sampling.write(json.dumps(dict(step=update+1,**trace))+'\n')
                    batch=d.pt_batch(data,labels);counts=model.target_counts(batch,labels)
                    for k in totals:totals[k]+=counts[k]
                    batches.append((data,labels))
                sampling.flush();record('RUNNING');totals={k:v.to('cuda:0',non_blocking=True) for k,v in totals.items()}
                lr=pt_learning_rate(update)
                for group in opt.param_groups:group['lr']=lr
                opt.zero_grad(set_to_none=True);raw={k:torch.zeros((),device='cuda:0') for k in weights}
                for data,labels in batches:
                    data=data.to('cuda:0',non_blocking=True);labels={k:v.to('cuda:0',non_blocking=True) if torch.is_tensor(v) else v for k,v in labels.items()}
                    with torch.autocast('cuda',dtype=torch.bfloat16):result=model.pretrain_result(d.pt_batch(data,labels),labels)
                    with torch.autocast('cuda',enabled=False):loss=window_objective(result,weights,totals)
                    if not torch.isfinite(loss):raise FloatingPointError('nonfinite PT loss')
                    loss.backward()
                    for k in raw:raw[k]+=result['sums'][k].detach().float()
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step();completed+=1
                # Normal per-update loss logging already synchronizes; no diagnostic forward.
                rec=dict(step=completed,lr=lr,grad_norm=float(norm),loss={k:float(raw[k]/totals[k].clamp_min(1)) for k in raw},counts={k:float(v) for k,v in totals.items()},weights=weights,stream=model.stream_state(),gpu_peak_bytes=torch.cuda.max_memory_allocated())
                rec['seconds']=time.monotonic()-tick;rec['worker_io']=worker_io;rec['data_wait_seconds']=wait_seconds;rec['prefix_checked']=prefix_checked;durations.append(rec['seconds']);history.write(json.dumps(rec)+'\n');history.flush();write(folder/'timing.json',rec);record('RUNNING')
                if completed==20:last_window_started=time.monotonic()
                if completed==100 or completed%500==0:
                    stable=durations[20:] if completed<=500 else durations[-500:]
                    window_elapsed=time.monotonic()-last_window_started
                    per=max(float(np.mean(stable)),window_elapsed/(80 if completed==100 else 400 if completed==500 else 500));last_window_started=time.monotonic();remaining=(10000-completed)*per
                    prior=read(OUT/'gates/remaining_eta.json');downstream=prior['work_seconds']/prior['concurrency']+prior['formal_seconds']
                    total_eta=1.5*(remaining+downstream)+7200
                    memory=read(OUT/'memory_last_full.json');write(OUT/'gates/pt_eta.json',dict(worker_io=worker_io,data_wait_seconds=wait_seconds,memory_sample=memory,step=completed,seconds_per_update=per,remaining_pt=remaining,conservative_remaining=total_eta,deadline_unix=read(OUT/'authorization/clock.json')['deadline_unix']))
                    if time.time()+1.5*remaining>pt_deadline or time.time()+total_eta>=read(OUT/'authorization/clock.json')['deadline_unix']:raise TimeoutError('PT conservative ETA gate')
                if completed==858:
                    write(folder/'parent_boundary_snapshot.json',dict(completed=completed,prefix_checked=prefix_checked,parent_attempts=858,consumption=read(folder/'consumption.json'),memory=read(OUT/'memory_last_full.json')))
                if completed in (5000,10000):
                    pack=package(completed);save_checkpoint(folder/f'step{completed}.pt',pack)
                    if completed==10000:save_checkpoint(folder/'deploy.pt',pack)
                print(json.dumps(rec),flush=True)
        if next(iterator,None) is not None:raise ValueError('extra PT microbatch')
        m.validate_package(package(10000));success=True;record('COMPLETE')
    except BaseException:
        record('FAILED');write(folder/'failure.json',dict(error=traceback.format_exc(),automatic_retry=False));raise
    finally:
        ignore_termination()
        if iterator is not None:iterator._shutdown_workers()
        record('COMPLETE' if success else 'FAILED')

# Memory monitoring is CPU-only; these helpers never construct or run a model.
def parse_kb_fields(text):
    result={}
    for line in text.splitlines():
        if ':' not in line:continue
        name,value=line.split(':',1);parts=value.split()
        if parts and parts[0].isdigit():result[name]=int(parts[0])*(1024 if len(parts)>1 and parts[1]=='kB' else 1)
    return result

def pid_identity(pid):
    try:
        fields=(Path('/proc')/str(pid)/'stat').read_text().rsplit(')',1)[1].split()
        return dict(pid=int(pid),start_ticks=int(fields[19]),ppid=int(fields[1]),state=fields[0],rss_bytes=int(fields[21])*os.sysconf('SC_PAGE_SIZE'))
    except (FileNotFoundError,ProcessLookupError):return None

def memory_environment():
    mem=parse_kb_fields(Path('/proc/meminfo').read_text());cg=Path('/sys/fs/cgroup')
    result=dict(mem_available=mem['MemAvailable'],mem_total=mem['MemTotal'],cgroup_path=str(cg),cgroup_exclusive=False)
    for name in ('memory.current','memory.max','memory.swap.current'):
        value=(cg/name).read_text().strip();result[name]=int(value) if value!='max' else 'max'
    for name in ('memory.events','memory.stat'):
        result[name]={line.split()[0]:int(line.split()[1]) for line in (cg/name).read_text().splitlines()}
    return result

def memory_decision(pss_swap,env,complete=True):
    if not complete:return 'TELEMETRY_ERROR: living-process PSS coverage incomplete'
    if pss_swap is not None and pss_swap>160*1024**3:return 'MEMORY_STOP: PSS+SwapPss exceeds160GiB'
    if env['mem_available']<32*1024**3:return 'ENVIRONMENT_STOP: MemAvailable below32GiB'
    if env['memory.max']!='max' and env['memory.max']-env['memory.current']<8*1024**3:return 'ENVIRONMENT_STOP: shared cgroup headroom below8GiB'
    return None

def persist_memory_sample(out,sample):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    with (out/'memory.jsonl').open('a') as f:f.write(json.dumps(sample,allow_nan=False)+'\n');f.flush()
    write(out/'memory_latest.json',sample)
    if sample.get('full_pss') and sample.get('coverage_complete'):write(out/'memory_last_full.json',sample)
    if sample.get('decision'):
        name='telemetry_error.json' if sample['decision'].startswith('TELEMETRY_ERROR') else 'memory_breach.json'
        if not (out/name).exists():exclusive(out/name,sample)

def observe_pid_memory(entry,rollup_reader=None,identity_reader=None):
    rollup_reader=rollup_reader or (lambda pid:(Path('/proc')/str(pid)/'smaps_rollup').read_text())
    identity_reader=identity_reader or pid_identity
    try:
        fields=parse_kb_fields(rollup_reader(entry['pid']))
        required=('Pss','SwapPss','Pss_Anon','Pss_File','Pss_Shmem','Private_Clean','Private_Dirty','Rss')
        if any(k not in fields for k in required):raise ValueError('required PSS fields missing')
    except (FileNotFoundError,ProcessLookupError,PermissionError,ValueError) as exc:
        after=identity_reader(entry['pid'])
        if after is None or after['start_ticks']!=entry['start_ticks'] or after['state']=='Z':return dict(status='EXITED',identity=entry,reason=str(exc))
        raise RuntimeError(f"living PID {entry['pid']}:{entry['start_ticks']} PSS unreadable: {exc}") from exc
    after=identity_reader(entry['pid'])
    if after is None or after['start_ticks']!=entry['start_ticks'] or after['state']=='Z':return dict(status='EXITED',identity=entry,reason='identity exited/reused during read')
    return dict(status='MEASURED',fields=fields)

class MemoryMonitor:
    def __init__(self,out=OUT):
        self.out=Path(out);self.registry={};self.last_pss=0.;self.total_seconds=0.;self.samples=0
        for entry in read(self.out/'authorization/process_roots.json'):self.register(entry,entry['role'])
        self.register(pid_identity(os.getpid()),'controller')
    def register(self,entry,role):
        if entry is None:return
        key=f"{entry['pid']}:{entry['start_ticks']}"
        if key not in self.registry:self.registry[key]=dict(pid=entry['pid'],start_ticks=entry['start_ticks'],role=role,first_seen_unix=time.time())
    def discover(self):
        current={}
        for path in Path('/proc').iterdir():
            if not path.name.isdigit():continue
            try:entry=pid_identity(int(path.name))
            except PermissionError:continue
            if entry:current[entry['pid']]=entry
        for known in self.registry.values():
            if known['pid'] not in current:
                probe=pid_identity(known['pid'])
                if probe:current[probe['pid']]=probe
        active={entry['pid'] for entry in self.registry.values() if entry['pid'] in current and current[entry['pid']]['start_ticks']==entry['start_ticks']}
        while True:
            added={pid for pid,entry in current.items() if entry['ppid'] in active and pid not in active}
            if not added:break
            for pid in added:self.register(current[pid],'descendant')
            active|=added
        return {pid:current[pid] for pid in active}
    def sample(self,force=False):
        tick=time.monotonic();full=force or time.time()-self.last_pss>=15.;sample=dict(time_unix=time.time(),full_pss=full,metric='sum Pss+SwapPss',threshold_bytes=160*1024**3,pids=[],decision=None)
        try:
            env=memory_environment();sample['environment']=env;active=self.discover();measured={};exited=[];sample['coverage']=dict(discovered_live=sum(e['state']!='Z' for e in active.values()),measured=0)
            if full:
                for _ in range(4):
                    for pid,entry in active.items():
                        key=(pid,entry['start_ticks'])
                        if key in measured or entry['state']=='Z':continue
                        try:
                            result=observe_pid_memory(entry)
                            if result['status']=='EXITED':exited.append(result);continue
                            measured[key]=dict(**self.registry[f'{pid}:{entry["start_ticks"]}'],fields=result['fields'])
                            sample['pids']=list(measured.values());sample['coverage']['measured']=len(measured)
                        except Exception as exc:
                            sample['unmeasurable']=dict(pid=pid,start_ticks=entry['start_ticks'],error=str(exc))
                            raise
                    active=self.discover()
                    pending=[(pid,e) for pid,e in active.items() if e['state']!='Z' and (pid,e['start_ticks']) not in measured]
                    if not pending:break
                sample['coverage_complete']=not pending
                sample['pids']=list(measured.values());sample['exit_races']=exited
                sample['pss_swap_bytes']=sum(e['fields']['Pss']+e['fields']['SwapPss'] for e in measured.values())
                sample['totals']={field:sum(e['fields'][field] for e in measured.values()) for field in ('Pss','SwapPss','Pss_Anon','Pss_File','Pss_Shmem','Private_Clean','Private_Dirty','Rss')}
                live_count=sum(e['state']!='Z' for e in active.values())
                live_measured=sum(e['state']!='Z' and (pid,e['start_ticks']) in measured for pid,e in active.items())
                sample['coverage']=dict(discovered_live=live_count,measured_at_any_time=len(measured),measured_live=live_measured,rate=live_measured/live_count if live_count else 1.,missing=[dict(pid=pid,start_ticks=e['start_ticks']) for pid,e in pending])
                self.last_pss=time.time()
            else:
                sample['pids']=[dict(**self.registry[f'{pid}:{e["start_ticks"]}'],rss_bytes=e['rss_bytes'],state=e['state']) for pid,e in active.items()]
                sample['rss_sum_bytes']=sum(e['rss_bytes'] for e in active.values());sample['pss_swap_bytes']=None
                sample['coverage_complete']=True;sample['pss_note']='not sampled on light tick; next full sample within15s'
            sample['decision']=memory_decision(sample['pss_swap_bytes'],env,sample['coverage_complete'])
        except Exception as exc:
            sample.update(coverage_complete=False,decision='TELEMETRY_ERROR: '+str(exc),error=traceback.format_exc())
        sample['sampling_seconds']=time.monotonic()-tick;self.total_seconds+=sample['sampling_seconds'];self.samples+=1
        sample['monitor_total_seconds']=self.total_seconds;sample['monitor_samples']=self.samples
        write(self.out/'process_registry.json',list(self.registry.values()))
        # Decision sample is persisted and flushed before an exception can stop workers.
        persist_memory_sample(self.out,sample)
        if sample['decision']:raise RuntimeError(sample['decision'])
        return sample


def install_stop_handler():
    import signal
    def stop(signum,frame):
        ignore_termination()
        raise RuntimeError('controller termination signal '+str(signum))
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
