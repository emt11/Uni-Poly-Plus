"""FT-only r4 lifecycle execution, explicit r3 model aliases; no new PT."""
import copy,json,math,os,time,traceback
from pathlib import Path
import numpy as np
import torch
from src.modules import ua3d_cond_spectral_r4 as m
from src.dataset import ua3d_cond_spectral_r4 as d
from src.training.ua3m_paperfuse import (read,write,exclusive,now,ignore_termination,seed_all,finite,metrics,process_identity,scaler_validate,transform,summary,SOTA,require_tmux,save_checkpoint,save_predictions)
ROOT=Path(__file__).resolve().parents[2]
PLAN=m.PLAN;REVISION='r4';EXECUTION_REVISION='gpu2-fttest4-pss-lifecycle-r4'
PARENT=ROOT/'results/ua3d_cond_spectral_20261104/r2'
P3=PARENT/'repairs/memory_r3'
OUT=PARENT/'repairs/lifecycle_r4';LOG=ROOT/'logs/ua3d_cond_spectral_20261104/r2/repairs/lifecycle_r4'
B=ROOT/'results/ua3d_base_opt_20261026/r6/repairs/concurrency_r1'
SURFACE=ROOT/'results/ua3d_hmr_20261103/r1/repairs/schema_surface_r2'
DOWN=ROOT/'data/processed/mcl_ph_r4_20260930/downstream_direct_01.staging'
SPLIT=ROOT/'data/splits/mips_paper5_outer'
INPUT_PT=P3/'pretrain/BASE_FRESH/deploy.pt'
TASKS=m.TASKS;ARMS=m.ARMS;ALLOWED_GPUS=(2,);REFERENCE=.804675462276949
UUID='GPU-31456dd5-a421-142b-644b-7a80f482dfb9'

def save_checkpoint(path,payload):
    # Runtime helper does not create nested directories; explicitly contain writes.
    path=Path(path)
    if not path.is_relative_to(OUT):raise ValueError('checkpoint outside new root')
    path.parent.mkdir(parents=True,exist_ok=True)
    from src.training.glt_dual_runtime import save_checkpoint as save
    save(path,payload)

def head_path(task,fold):return P3/'initialization/heads'/task/f'fold{fold}.pt'
def base_path(task,fold):return B/'finetune/F_PD_N1_F0'/task/f'fold{fold}'
def parent_path(arm):return INPUT_PT
def unit_path(arm,task,fold,formal=False):return OUT/('formal' if formal else 'finetune')/arm/task/f'fold{fold}'
def source_config(task,fold):return read(base_path(task,fold)/'config.json')
def config(arm,task,fold):
    if arm not in ARMS or task not in TASKS or fold not in range(5):raise ValueError('matrix identity')
    src=source_config(task,fold)
    return dict(plan=PLAN,revision=REVISION,execution_revision='r4',input_pt_revision='r3',science_recipe='r2',arm=arm,task=task,fold=fold,seed=42+fold,parent=str(parent_path(arm)),head=str(head_path(task,fold)),input_cache=src['input_cache'],scaler_reference=src['scaler_reference'],model=m.CONFIG,batch=24,epochs=70,workers=4,world_size=1)
def update_plan(status,phase,detail):
    path=ROOT/'CURRENT_PLAN.md';s=path.read_text();lines=s.splitlines()
    for index,line in enumerate(lines):
        if line.startswith('> 状态：'):
            lines[index]=f'> 状态：**{status}**；阶段 {phase}；更新时间 {now()}。父r2/r3终态及原总截止独立保留。';break
    s='\n'.join(lines)+'\n';start='<!-- CONDSPEC_R4_STATUS_START -->';end='<!-- CONDSPEC_R4_STATUS_END -->'
    left,rest=s.split(start,1);_,right=rest.split(end,1)
    path.write_text(left+start+'\n'+f'{now()} 状态 {status}；阶段 {phase}。{detail}\nindependent_review=PENDING；完整正式TEST前未完成/不可比较。\n'+end+right)
    write(OUT/'status.json',dict(status=status,phase=phase,detail=detail,time=now(),independent_review='PENDING'))
def check_clock(prepare=False):
    c=read(OUT/'authorization/clock.json');limit=c['repair_prepare_deadline_unix'] if prepare else c['deadline_unix']
    if time.time()>=limit:raise TimeoutError('EXPIRED: inherited stage/total deadline')
def check_prepared():
    identity=read(OUT/'identity.json')
    if identity.get('plan')!=PLAN or identity.get('revision')!=REVISION or not INPUT_PT.is_file():raise ValueError('fresh initialization identity')
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
    state['model_reads']=dict(historical_BASE_tensor_reads=0,current_campaign_PT_alias=0,random_head_alias=0,current_r4_BEST=0)
    record();ds=None;started=time.monotonic()
    try:
        state['actual_device']=gpu_contract();record()
        scaler=read(cfg['scaler_reference'])
        if formal:
            frozen=read(OUT/'selection/final_frozen.json');selected=frozen_unit(frozen,arm,task,fold)
            state['model_reads']['current_r4_BEST']+=1;record()
            saved=torch.load(selected['best'],map_location='cpu',weights_only=False)
            if any(saved[k]!=selected[k] for k in ('best_epoch','best_stage','scaler')):raise ValueError('frozen checkpoint/scaler mismatch')
            if scaler!=saved['scaler']:raise ValueError('BEST scaler mismatch')
            model=m.Model(arm,fold)
            if saved.get('schema')!=m.FT_SCHEMA or saved.get('config')!=cfg:raise ValueError('FT checkpoint schema/config mismatch')
            m.strict_module_state(model,saved['state_dict']);best_epoch=saved['best_epoch'];best_stage=saved['best_stage'];del saved
        else:
            seed_all(42+fold)
            state['model_reads']['current_campaign_PT_alias']+=1;record()
            package=load_input_tensor(parent_path(arm),'pt',task,fold)
            state['model_reads']['random_head_alias']+=1;record()
            head=load_input_tensor(head_path(task,fold),'head',task,fold)
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
                unit=units[f'{a}/{task}/{fold}'];unit['test_keys']=[keys[i] for i in unit['test_indices']];unit['input_pt_revision']='r3';unit['input_pt']=str(INPUT_PT);unit['random_head']=str(head_path(task,fold))
    exclusive(OUT/'selection/final_frozen.json',dict(plan=PLAN,revision=REVISION,frozen=True,time=now(),frozen_unix=time.time(),input_pt_revision='r3',input_pt=str(INPUT_PT),source_aliases=read(OUT/'source_aliases.json'),FINAL=chosen,eligible=eligible,validation=validation,units=units,reference=REFERENCE,test_targets_read=False,rule='strict macro > C_FRESH; every task >= C_FRESH-.010; registration-order ties; otherwise C_FRESH'))

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

def allowed_input_path(path,kind,task,fold):
    expected=INPUT_PT if kind=='pt' else head_path(task,fold) if kind=='head' else None
    if expected is None or Path(path).resolve()!=expected.resolve():raise ValueError('model source outside explicit PT/random-head whitelist')
    return expected

def load_input_tensor(path,kind,task,fold):
    return torch.load(allowed_input_path(path,kind,task,fold),map_location='cpu',weights_only=False)

def install_stop_handler():
    import signal
    def stop(signum,frame):
        ignore_termination();raise RuntimeError('controller termination signal '+str(signum))
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)

# Lifecycle/memory routines are independent of all model/training state.
TERMINAL_STATES=frozenset(('Z','X','x'))
PSS_FIELDS=('Pss','SwapPss','Pss_Anon','Pss_File','Pss_Shmem','Private_Clean','Private_Dirty','Rss')

def parse_kb_fields(text):
    result={}
    for line in text.splitlines():
        if ':' not in line:continue
        name,value=line.split(':',1);parts=value.split()
        if name.replace('_','').isalpha() and parts and parts[0].isdigit():result[name]=int(parts[0])*(1024 if len(parts)>1 and parts[1]=='kB' else 1)
    return result

def pid_identity(pid):
    try:
        raw=(Path('/proc')/str(pid)/'stat').read_text();parts=raw.rsplit(')',1)[1].split()
        return dict(pid=int(pid),start_ticks=int(parts[19]),ppid=int(parts[1]),pgrp=int(parts[2]),state=parts[0],rss_bytes=int(parts[21])*os.sysconf('SC_PAGE_SIZE'),raw_stat=raw.strip())
    except (FileNotFoundError,ProcessLookupError):return None

def identity_key(entry):return f"{entry['pid']}:{entry['start_ticks']}"

def identity_status(entry,current,event=False):
    if current is None:return 'identity_disappeared'
    if current['start_ticks']!=entry['start_ticks']:return 'PID_reused'
    if current['state'] in TERMINAL_STATES:return 'terminal_state_'+current['state']
    if event:return 'pidfd_exit_event'
    return None

class PSSReadError(RuntimeError):
    def __init__(self,message,evidence):
        super().__init__(message);self.evidence=evidence

def observe_pid_memory(entry,rollup_reader=None,identity_reader=None,exit_event=None,clock=None,sleeper=None):
    """Only ESRCH/ENOENT gets <=3 reads within a single 250ms exit window."""
    import errno
    rollup_reader=rollup_reader or (lambda pid:(Path('/proc')/str(pid)/'smaps_rollup').read_text())
    identity_reader=identity_reader or pid_identity;exit_event=exit_event or (lambda:False)
    clock=clock or time.monotonic;sleeper=sleeper or time.sleep
    began=clock();attempts=[];evidence=dict(identity=entry,attempts=attempts,grace_seconds=.250,max_reads=3)
    def finish(reason,current,event):
        return dict(status='EXITED',reason=reason,identity=entry,after=current,exit_event=event,attempts=attempts,elapsed_seconds=clock()-began)
    for number in range(1,4):
        before=identity_reader(entry['pid']);event=exit_event();reason=identity_status(entry,before,event)
        if reason:return finish(reason,before,event)
        if attempts and clock()-began>=.250:
            evidence.update(before=before,after=before,exit_event=event,elapsed_seconds=clock()-began)
            raise PSSReadError('living PID PSS unresolved at250ms bound',evidence)
        observation=dict(attempt=number,before=before,started_offset_seconds=clock()-began);attempts.append(observation)
        try:
            fields=parse_kb_fields(rollup_reader(entry['pid']))
            missing=[name for name in PSS_FIELDS if name not in fields]
            if missing:raise ValueError('required PSS fields missing: '+','.join(missing))
        except (OSError,ValueError) as exc:
            after=identity_reader(entry['pid']);event=exit_event();reason=identity_status(entry,after,event)
            observation.update(after=after,exit_event=event,errno=getattr(exc,'errno',None),error=str(exc),exception_type=type(exc).__name__,finished_offset_seconds=clock()-began)
            if reason:return finish(reason,after,event)
            transient=isinstance(exc,OSError) and exc.errno in (errno.ESRCH,errno.ENOENT)
            remaining=.250-(clock()-began)
            if transient and number<3 and remaining>0:
                sleeper(min(.120,remaining));continue
            if transient and remaining>0:
                # No fourth PSS read: final identity/event observation inside grace.
                sleeper(min(.010,remaining));after=identity_reader(entry['pid']);event=exit_event();reason=identity_status(entry,after,event)
                observation.update(final_after=after,final_exit_event=event)
                if reason:return finish(reason,after,event)
            evidence.update(before=before,after=after,exit_event=event,errno=getattr(exc,'errno',None),elapsed_seconds=clock()-began)
            raise PSSReadError(f"living PID {entry['pid']}:{entry['start_ticks']} PSS unreadable after bounded reconciliation: {exc}",evidence) from exc
        after=identity_reader(entry['pid']);event=exit_event();reason=identity_status(entry,after,event)
        observation.update(after=after,exit_event=event,errno=None,fields=fields,finished_offset_seconds=clock()-began)
        if reason:return finish(reason,after,event)
        return dict(status='MEASURED',identity=entry,fields=fields,after=after,attempts=attempts,elapsed_seconds=clock()-began)
    raise AssertionError('unreachable bounded PSS loop')

def memory_environment():
    mem=parse_kb_fields(Path('/proc/meminfo').read_text());cg=Path('/sys/fs/cgroup')
    result=dict(mem_available=mem['MemAvailable'],mem_total=mem['MemTotal'],cgroup_path=str(cg),cgroup_exclusive=False)
    for name in ('memory.current','memory.max','memory.swap.current'):
        value=(cg/name).read_text().strip();result[name]=int(value) if value!='max' else 'max'
    for name in ('memory.events','memory.stat'):result[name]={line.split()[0]:int(line.split()[1]) for line in (cg/name).read_text().splitlines()}
    return result

def memory_decision(pss_swap,env,complete=True):
    if not complete:return 'TELEMETRY_ERROR: unresolved living identities/PSS coverage'
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

class MemoryMonitor:
    def __init__(self,out=OUT):
        self.out=Path(out);self.active={};self.retired=set();self.last_pss=0.;self.last_light=0.;self.total_seconds=0.;self.samples=0;self.last_identity_error=None
        for entry in read(self.out/'authorization/process_roots.json'):self.register(entry,entry['role'])
        self.register(pid_identity(os.getpid()),'controller')
    def event(self,kind,**value):
        row=dict(time_unix=time.time(),event=kind,**value)
        with (self.out/'lifecycle.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n');f.flush()
    def close_fd(self,entry):
        fd=entry.get('pidfd');entry['pidfd']=None
        if fd is not None:
            try:os.close(fd);self.event('pidfd_closed',key=identity_key(entry))
            except OSError as exc:self.event('pidfd_close_race',key=identity_key(entry),errno=exc.errno,error=str(exc))
    def register(self,entry,role):
        if entry is None:return None
        key=identity_key(entry)
        if key in self.retired:return None
        if key in self.active:return key
        row=dict(entry,role=role,first_seen_unix=time.time(),pidfd=None,pidfd_capability='unavailable')
        self.active[key]=row
        before=pid_identity(entry['pid']);reason=identity_status(entry,before)
        if reason:self.retire(key,reason,before=entry,after=before);return None
        try:
            fd=os.pidfd_open(entry['pid'],0);row['pidfd']=fd
            after=pid_identity(entry['pid']);reason=identity_status(entry,after)
            if reason:
                self.close_fd(row);self.retire(key,'pidfd_bind_'+reason,before=before,after=after);return None
            row['pidfd']=fd;row['pidfd_capability']='enabled';self.event('pidfd_bound',key=key,before=before,after=after)
        except (AttributeError,OSError) as exc:
            self.close_fd(row)
            row['pidfd_capability']='fallback_stat';row['pidfd_error']=dict(errno=getattr(exc,'errno',None),error=str(exc));self.event('pidfd_unavailable',key=key,detail=row['pidfd_error'])
        self.event('registered',key=key,identity={k:v for k,v in row.items() if k!='pidfd'})
        return key
    def exit_event(self,key):
        import select
        row=self.active.get(key)
        if row is None:return key in self.retired
        fd=row.get('pidfd')
        if fd is None:return False
        poll=select.poll();poll.register(fd,select.POLLIN|select.POLLHUP|select.POLLERR)
        events=poll.poll(0)
        if any(flags&select.POLLNVAL for _,flags in events):raise RuntimeError('registered pidfd unexpectedly invalid: '+key)
        if any(flags&select.POLLERR for _,flags in events):raise RuntimeError('pidfd poll error without confirmed terminal event: '+key)
        return any(flags&(select.POLLIN|select.POLLHUP) for _,flags in events)
    def retire(self,key,reason,**evidence):
        row=self.active.pop(key,None)
        if row is None:return
        self.close_fd(row);self.retired.add(key)
        self.event('terminal_confirmed',key=key,role=row['role'],reason=reason,real_exit=evidence.pop('real_exit',None),**evidence)
    def complete_direct(self,job,code):
        if job.get('start_ticks') is not None:self.retire(f"{job['pid']}:{job['start_ticks']}",'Popen.poll/wait',real_exit=code,after=pid_identity(job['pid']))
        self.event('direct_worker_wait',pid=job['pid'],start_ticks=job.get('start_ticks'),role=job['kind'],real_exit=code)
    def discover(self):
        # Probe only active identities. Terminal history is append-only, never rescanned.
        current={}
        for path in Path('/proc').iterdir():
            if not path.name.isdigit():continue
            try:entry=pid_identity(int(path.name))
            except PermissionError:continue
            if entry:current[entry['pid']]=entry
        for key,row in list(self.active.items()):
            before=current.get(row['pid'])
            if before is None:
                try:before=pid_identity(row['pid'])
                except OSError as exc:
                    if self.exit_event(key):self.retire(key,'confirmed_exit_during_identity_error',before=row,after=None,errno=exc.errno);continue
                    self.last_identity_error=dict(key=key,before=row,after=None,errno=exc.errno,error=str(exc));self.event('identity_refresh_error',**self.last_identity_error);raise
            reason=identity_status(row,before,self.exit_event(key))
            if reason:self.retire(key,reason,before=row,after=before)
            elif before is not None:row.update(before)
        parents={row['pid'] for row in self.active.values()}
        while True:
            candidates=[e for e in current.values() if e['ppid'] in parents and identity_key(e) not in self.active and identity_key(e) not in self.retired]
            if not candidates:break
            old=len(parents)
            for entry in candidates:
                key=self.register(entry,'descendant')
                if key:parents.add(entry['pid'])
            if len(parents)==old:break
        return {key:dict(row) for key,row in self.active.items()}
    def sample(self,force=False):
        now_tick=time.monotonic()
        if not force and now_tick-self.last_light<5:return None
        full=force or now_tick-self.last_pss>=15.;self.last_light=now_tick
        sample=dict(time_unix=time.time(),sample_started_unix=time.time(),full_pss=full,metric='sum Pss+SwapPss',threshold_bytes=160*1024**3,pids=[],exit_transitions=[],coverage_complete=False,pss_swap_bytes=None,decision=None)
        observations={}
        try:
            env=memory_environment();sample['environment']=env
            current=self.discover();sample['initial_identities']=[{k:v for k,v in e.items() if k!='pidfd'} for e in current.values()]
            if full:
                pending=[]
                for discovery_round in range(4):
                    for key,entry in current.items():
                        if key in observations:continue
                        try:result=observe_pid_memory(entry,exit_event=lambda key=key:self.exit_event(key))
                        except PSSReadError as exc:sample['failed_observation']=exc.evidence;raise
                        observations[key]=result
                        if result['status']=='EXITED':
                            sample['exit_transitions'].append(result);self.retire(key,result['reason'],observation=result)
                    current=self.discover()
                    pending=[key for key in current if key not in observations or observations[key]['status']!='MEASURED']
                    if not pending:break
                final_keys=set(current);measured={key:value for key,value in observations.items() if key in final_keys and value['status']=='MEASURED'}
                sample['pids']=[dict(key=key,**{k:v for k,v in current[key].items() if k!='pidfd'},**value) for key,value in measured.items()]
                sample['exited_measurements']=[dict(key=key,**value) for key,value in observations.items() if key not in final_keys]
                sample['coverage_complete']=not pending
                sample['coverage']=dict(discovered_live=len(final_keys),measured_live=len(measured),missing=pending,rate=len(measured)/len(final_keys) if final_keys else 1.)
                sample['pss_swap_bytes']=sum(value['fields']['Pss']+value['fields']['SwapPss'] for value in measured.values()) if not pending else None
                sample['totals']={name:sum(value['fields'][name] for value in measured.values()) for name in PSS_FIELDS} if not pending else None
                sample['discovery_rounds']=discovery_round+1;self.last_pss=time.monotonic()
            else:
                sample['pids']=[{k:v for k,v in row.items() if k!='pidfd'} for row in current.values()]
                sample['rss_sum_bytes']=sum(row['rss_bytes'] for row in current.values());sample['coverage_complete']=True
                sample['coverage']=dict(identity_complete=True,pss_rate=None,note='RSS/environment tick; no new PSS value claimed')
            sample['final_identities']=[{k:v for k,v in e.items() if k!='pidfd'} for e in current.values()]
            sample['decision']=memory_decision(sample['pss_swap_bytes'],env,sample['coverage_complete'])
        except Exception as exc:
            sample.update(coverage_complete=False,decision='TELEMETRY_ERROR: '+str(exc),error=traceback.format_exc())
            sample['all_observations_before_error']=observations
            sample['identity_refresh_error']=self.last_identity_error
            sample['final_identities']=[{k:v for k,v in e.items() if k!='pidfd'} for e in self.active.values()]
        sample['sample_finished_unix']=time.time();sample['sampling_seconds']=time.monotonic()-now_tick
        self.total_seconds+=sample['sampling_seconds'];self.samples+=1
        sample.update(monitor_total_seconds=self.total_seconds,monitor_samples=self.samples,active_identities=len(self.active),retired_identities=len(self.retired),open_pidfds=sum(e.get('pidfd') is not None for e in self.active.values()))
        write(self.out/'process_registry.json',dict(active=[{k:v for k,v in e.items() if k!='pidfd'} for e in self.active.values()],retired_count=len(self.retired),exit_history='lifecycle.jsonl'))
        persist_memory_sample(self.out,sample)
        if sample['decision']:raise RuntimeError(sample['decision'])
        return sample
    def signal_identity(self,key,signum):
        import signal
        row=self.active.get(key)
        if row is None:return
        before=pid_identity(row['pid']);reason=identity_status(row,before,self.exit_event(key))
        if reason:self.retire(key,'signal_skip_'+reason,before=row,after=before);return
        try:
            if row.get('pidfd') is not None and hasattr(signal,'pidfd_send_signal'):signal.pidfd_send_signal(row['pidfd'],signum)
            else:
                after=pid_identity(row['pid'])
                if identity_status(row,after):self.retire(key,'signal_identity_changed',before=before,after=after);return
                os.kill(row['pid'],signum)
            self.event('signal_sent',key=key,signum=signum,before=before)
        except ProcessLookupError as exc:
            self.event('signal_exit_race',key=key,signum=signum,before=before,after=pid_identity(row['pid']),errno=exc.errno)
    def close(self):
        for entry in self.active.values():self.close_fd(entry)
