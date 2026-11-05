"""FT-only r5 fixed-cohort execution, explicit r3 model aliases; no new PT."""
import copy,json,math,os,time,traceback
from pathlib import Path
import numpy as np
import torch
from src.modules import ua3d_cond_spectral_r5 as m
from src.dataset import ua3d_cond_spectral_r5 as d
from src.training.ua3m_paperfuse import (read,write,exclusive,now,ignore_termination,seed_all,finite,metrics,process_identity,scaler_validate,transform,summary,SOTA,require_tmux,save_checkpoint,save_predictions)
ROOT=Path(__file__).resolve().parents[2]
PLAN=m.PLAN;REVISION='r5';EXECUTION_REVISION='gpu2-fttest4-pss-cohort-r5'
PARENT=ROOT/'results/ua3d_cond_spectral_20261104/r2'
P3=PARENT/'repairs/memory_r3'
P4=PARENT/'repairs/lifecycle_r4'
OUT=PARENT/'repairs/coverage_r5';LOG=ROOT/'logs/ua3d_cond_spectral_20261104/r2/repairs/coverage_r5'
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
def is_parent_alias(arm,task,fold):return (arm,task,fold)==('C_FRESH','eps',0)
def unit_path(arm,task,fold,formal=False):
    root=P4 if not formal and is_parent_alias(arm,task,fold) else OUT
    return root/('formal' if formal else 'finetune')/arm/task/f'fold{fold}'
def source_config(task,fold):return read(base_path(task,fold)/'config.json')
def config(arm,task,fold):
    if arm not in ARMS or task not in TASKS or fold not in range(5):raise ValueError('matrix identity')
    src=source_config(task,fold)
    return dict(plan=PLAN,revision=REVISION,execution_revision='r5',input_pt_revision='r3',science_recipe='r2',arm=arm,task=task,fold=fold,seed=42+fold,parent=str(parent_path(arm)),head=str(head_path(task,fold)),input_cache=src['input_cache'],scaler_reference=src['scaler_reference'],model=m.CONFIG,batch=24,epochs=70,workers=4,world_size=1)
def update_plan(status,phase,detail):
    path=ROOT/'CURRENT_PLAN.md';s=path.read_text();lines=s.splitlines()
    for index,line in enumerate(lines):
        if line.startswith('> 状态：'):
            lines[index]=f'> 状态：**{status}**；阶段 {phase}；更新时间 {now()}。父r2/r3/r4终态及原总截止独立保留。';break
    s='\n'.join(lines)+'\n';start='<!-- CONDSPEC_R5_STATUS_START -->';end='<!-- CONDSPEC_R5_STATUS_END -->'
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
def canonical_entry(arm,task,fold):
    record=read(OUT/'registry/canonical.json');key=f'{arm}/{task}/{fold}'
    if key not in record['units']:raise ValueError('outside canonical whitelist')
    entry=record['units'][key]
    expected_kind='parent_complete_alias' if is_parent_alias(arm,task,fold) else 'new_ft'
    if entry['kind']!=expected_kind or entry['root']!=str(unit_path(arm,task,fold)) or entry['runner_config']!=config(arm,task,fold):raise ValueError('canonical source/runner mismatch')
    expected_schema='ua3d-cond-spectral-best-r4' if expected_kind=='parent_complete_alias' else m.FT_SCHEMA
    if entry['best_schema']!=expected_schema:raise ValueError('canonical source schema mismatch')
    return entry

def frozen_unit(frozen,arm,task,fold):
    if frozen.get('plan')!=PLAN or frozen.get('revision')!='r5' or frozen.get('frozen') is not True:raise ValueError('TEST before one valid global freeze')
    chosen=frozen['units'][f'{arm}/{task}/{fold}'];entry=canonical_entry(arm,task,fold)
    if chosen['runner_config']!=config(arm,task,fold) or chosen['estimator_config']!=entry['estimator_config'] or chosen['best_schema']!=entry['best_schema'] or chosen['best']!=str(Path(entry['root'])/'best.pt'):raise ValueError('frozen runner/estimator/source mismatch')
    return chosen

def unit(arm,task,fold,formal=False,physical_gpu=2):
    require_tmux()
    if physical_gpu not in ALLOWED_GPUS or os.environ.get('CUDA_VISIBLE_DEVICES')!=str(physical_gpu) or os.environ.get('WORLD_SIZE','1')!='1':raise ValueError('allocated physical GPU/world1 mismatch')
    install_stop_handler();torch.set_num_threads(2);check_prepared();cfg=config(arm,task,fold)
    canonical=canonical_entry(arm,task,fold)
    if not formal and canonical['kind']!='new_ft':raise ValueError('parent complete alias cannot be retrained or written')
    folder=unit_path(arm,task,fold,formal);folder.mkdir(parents=True,exist_ok=False);write(folder/'config.json',cfg)
    state=dict(status='RUNNING',pid=os.getpid(),process=process_identity(os.getpid()),started=now(),
        epochs_started=0,completed_epochs=0,attempted_updates=0,completed_updates=0,train_graphs=0,
        validation_started=0,validation_graphs=0,test_graphs=0,test_started=0,physical_gpu=physical_gpu,gpu_uuid=os.environ.get('UA3M_PHYSICAL_GPU_UUID'),allowed_gpus=list(ALLOWED_GPUS),resource_revision=EXECUTION_REVISION,phase='initialize')
    def record():write(folder/'consumption.json',state)
    state['model_reads']=dict(historical_BASE_tensor_reads=0,current_campaign_PT_alias=0,random_head_alias=0,parent_complete_BEST=0,current_r5_BEST=0)
    record();ds=None;started=time.monotonic()
    try:
        state['actual_device']=gpu_contract();record()
        scaler=read(cfg['scaler_reference'])
        if formal:
            frozen=read(OUT/'selection/final_frozen.json');selected=frozen_unit(frozen,arm,task,fold)
            state['model_reads']['parent_complete_BEST' if canonical['kind']=='parent_complete_alias' else 'current_r5_BEST']+=1;record()
            saved=torch.load(selected['best'],map_location='cpu',weights_only=False)
            if any(saved[k]!=selected[k] for k in ('best_epoch','best_stage','scaler')):raise ValueError('frozen checkpoint/scaler mismatch')
            if scaler!=saved['scaler']:raise ValueError('BEST scaler mismatch')
            model=m.Model(arm,fold)
            if saved.get('schema')!=selected['best_schema'] or saved.get('config')!=selected['estimator_config']:raise ValueError('FT checkpoint schema/config mismatch')
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
        write(folder/'metrics.json',dict(status='COMPLETE',config=cfg,estimator_config=canonical['estimator_config'],input_best_schema=canonical['best_schema'],best_epoch=best_epoch,best_stage=best_stage,scaler=scaler,
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
            canonical=canonical_entry(recipe_id,task,fold);folder=unit_path(recipe_id,task,fold,formal)
            met=read(folder/'metrics.json');c=read(folder/'consumption.json');entry=read(SPLIT/f'{task}.json')['folds'][fold]
            expected_config=canonical['runner_config'] if formal else canonical['estimator_config']
            if met.get('status')!='COMPLETE' or c.get('status')!='COMPLETE' or met['config']!=expected_config:raise ValueError('canonical estimator incomplete/config mismatch')
            updates=70*math.ceil(len(entry['train_indices'])/24)
            expected=dict(test_started=1,test_graphs=len(entry['test_indices'])) if formal else dict(epochs_started=70,completed_epochs=70,attempted_updates=updates,completed_updates=updates,train_graphs=70*len(entry['train_indices']),validation_started=70,validation_graphs=70*len(entry['validation_indices']),epoch_open=False)
            if any(c.get(k)!=v for k,v in expected.items()):raise ValueError('canonical scientific consumption mismatch')
            if canonical['kind']=='parent_complete_alias' and met['best_epoch']!=32:raise ValueError('parent fixed BEST epoch changed')
            scores[task].append(met['r2']);ft=Path(canonical['root'])
            units[f'{recipe_id}/{task}/{fold}']=dict(estimator_config=canonical['estimator_config'],runner_config=canonical['runner_config'],canonical_kind=canonical['kind'],best_schema=canonical['best_schema'],best=str(ft/'best.pt'),best_epoch=met['best_epoch'],best_stage=met['best_stage'],scaler=met['scaler'],scaler_path=str(ft/'scaler.json'),test_indices=sorted(entry['test_indices']),predictions=str(folder/('test_predictions.npz' if formal else 'validation_predictions.npz')))
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
    if sample.get('coverage_result')=='COMPLETE' and sample.get('fatal_stop') is None:write(out/'memory_last_full.json',sample)
    if sample.get('fatal_stop'):
        name='telemetry_error.json' if sample['fatal_stop'].startswith('TELEMETRY_ERROR') else 'memory_breach.json'
        if not (out/name).exists():exclusive(out/name,sample)

class PendingQueue:
    """Only never-served new identities get this immutable five-second deadline."""
    def __init__(self,clock=None,unix=None,emit=None):
        self.clock=clock or time.monotonic;self.unix=unix or time.time;self.emit=emit or (lambda kind,**data:None);self.entries={};self.late=[]
    def add(self,key,identity,first_seen=None):
        if key in self.entries:return self.entries[key]
        mono,wall=first_seen if first_seen is not None else (self.clock(),self.unix())
        value=dict(key=key,identity={k:v for k,v in identity.items() if k!='pidfd'},first_seen_monotonic=mono,first_seen_unix=wall,deadline_monotonic=mono+5.,deadline_unix=wall+5.,observer_calls=0,read_attempts=0,phase='UNATTEMPTED')
        self.entries[key]=value;self.emit('pending_enqueued',**value);return value
    def snapshot(self):return sorted(self.entries,key=lambda k:(self.entries[k]['deadline_monotonic'],k))
    def started(self,key):
        if key in self.entries:
            value=self.entries[key];value['observer_calls']+=1;value['phase']='IN_SERVICE';value['service_started_monotonic']=self.clock()
            self.emit('pending_service_started',**value,service_age_seconds=self.clock()-value['first_seen_monotonic'])
    def resolve(self,key,reason,attempts=0,evidence=None):
        value=self.entries.pop(key,None)
        if value is None:return
        now=self.clock();value.update(phase=reason,read_attempts=value['read_attempts']+attempts,completed_monotonic=now,age_seconds=now-value['first_seen_monotonic'],on_time=now<=value['deadline_monotonic'])
        if not value['on_time']:self.late.append(value)
        self.emit('pending_resolved',**value,evidence=evidence)
    def read_failed(self,key,evidence):
        if key not in self.entries:return
        value=self.entries[key];value.update(phase='READ_FAILED',read_attempts=len(evidence.get('attempts',[])),last_error=evidence)
        self.emit('pending_actual_read_failed',**value)
    def expired(self):
        now=self.clock()
        return self.late+[dict(value,timeout_observed_monotonic=now) for value in self.entries.values() if now>=value['deadline_monotonic']]

from src.training.ua3d_cond_spectral_r4 import MemoryMonitor as LifecycleMonitor

class MemoryMonitor(LifecycleMonitor):
    """One fixed cohort per pass; new tail identities are serviced, never ignored."""
    def __init__(self,out=OUT):
        self.pending_queue=PendingQueue(emit=lambda kind,**data:self.event(kind,**data));self.window=None;self.pass_number=0;self.next_pending_wake=None;self.refresh_needed=False;self.refresh_deadline=None
        super().__init__(out)
    @property
    def pending(self):return self.pending_queue.entries
    def register(self,entry,role):
        if entry is None:return None
        key=identity_key(entry);new=key not in self.active and key not in self.retired;seen=(time.monotonic(),time.time())
        result=super().register(entry,role)
        if result and new:
            self.pending_queue.add(result,self.active[result],seen)
            wake=seen[0]+.250;self.next_pending_wake=min(self.next_pending_wake,wake) if self.next_pending_wake is not None else wake
        return result
    def retire(self,key,reason,**evidence):
        self.pending_queue.resolve(key,'CONFIRMED_EXIT',len(evidence.get('observation',{}).get('attempts',[])),evidence=dict(reason=reason,**evidence));return super().retire(key,reason,**evidence)
    def has_pending(self):return bool(self.pending) or self.refresh_needed
    def service_due(self):
        return self.has_pending() and (self.next_pending_wake is None or time.monotonic()>=self.next_pending_wake)
    def service_timeout(self):
        now=time.monotonic();limits=[.250]
        if self.next_pending_wake is not None:limits.append(max(0.,self.next_pending_wake-now))
        if self.pending:limits.append(max(0.,min(x['deadline_monotonic'] for x in self.pending.values())-now))
        return min(limits)
    def pending_timeout(self):
        expired=self.pending_queue.expired()
        if expired:return dict(reason='TELEMETRY_ERROR_PENDING_DEADLINE',objects=expired)
        if self.refresh_deadline is not None and time.monotonic()>=self.refresh_deadline:return dict(reason='TELEMETRY_ERROR_PENDING_DEADLINE: fresh cohort refresh overdue',deadline_monotonic=self.refresh_deadline)
        return None
    def sample(self,force=False):
        began=time.monotonic();scheduled=self.next_pending_wake
        due=self.service_due()
        if not force and not due and began-self.last_light<5:return None
        self.last_light=began;self.pass_number+=1
        sample=dict(time_unix=time.time(),sample_started_unix=time.time(),pass_id=self.pass_number,full_pss=False,metric='sum Pss+SwapPss',threshold_bytes=160*1024**3,coverage_result=None,fatal_stop=None,decision=None,pss_total_all_active=None,pss_swap_bytes=None,cohort_complete=None,pending_service_requested=self.has_pending(),scheduled_service_monotonic=scheduled,actual_service_monotonic=began,service_wake_lag_seconds=max(0.,began-scheduled) if scheduled is not None else None)
        cohort={};observed={};exited=[]
        try:
            env=memory_environment();sample['environment']=env
            current=self.discover();sample['initial_identities']=[{k:v for k,v in e.items() if k!='pidfd'} for e in current.values()]
            timeout=self.pending_timeout()
            if timeout:sample['pending_deadline_evidence']=timeout;raise RuntimeError(timeout['reason'])
            full=force or self.has_pending() or began-self.last_pss>=15
            sample['full_pss']=full
            if full:
                fresh=self.window is None or time.monotonic()-self.window['started_monotonic']>=5. or self.refresh_needed or not self.pending
                if fresh:
                    self.window=dict(window_id=self.pass_number,started_monotonic=time.monotonic(),started_unix=time.time(),measurements={})
                    self.last_pss=time.monotonic();self.refresh_needed=False
                    ordered=sorted(current,key=lambda k:(self.pending.get(k,{}).get('deadline_monotonic',float('inf')),k))
                else:ordered=[key for key in self.pending_queue.snapshot() if key in current]
                # This finite target snapshot never grows in the tail discovery.
                cohort={key:dict(current[key]) for key in ordered};sample['cohort_targets']=[{k:v for k,v in e.items() if k!='pidfd'} for e in cohort.values()]
                for key,entry in cohort.items():
                    timeout=self.pending_timeout()
                    if timeout:sample['pending_deadline_evidence']=timeout;raise RuntimeError(timeout['reason'])
                    self.pending_queue.started(key)
                    try:result=observe_pid_memory(entry,exit_event=lambda key=key:self.exit_event(key))
                    except PSSReadError as exc:
                        sample['failed_observation']=exc.evidence;self.pending_queue.read_failed(key,exc.evidence);raise
                    result['read_finished_monotonic']=time.monotonic();result['read_finished_unix']=time.time();observed[key]=result
                    if result['status']=='EXITED':
                        exited.append(result);self.retire(key,result['reason'],observation=result)
                    else:
                        self.window['measurements'][key]=result
                        self.pending_queue.resolve(key,'MEASURED',len(result['attempts']),evidence=result)
                sample['cohort_complete']=all(key in observed for key in cohort)
                # Tail discovery may enqueue new unattempted identities, but cannot
                # retroactively change the cohort's denominator or cause a read retry.
                current=self.discover();now=time.monotonic()
                timeout=self.pending_timeout()
                if timeout:sample['pending_deadline_evidence']=timeout;raise RuntimeError(timeout['reason'])
                measured={key:value for key,value in self.window['measurements'].items() if key in current and now-value['read_finished_monotonic']<=5. and value['read_finished_monotonic']>=self.window['started_monotonic']}
                missing=[key for key in current if key not in measured]
                sample['pids']=[dict(value,key=key,identity={k:v for k,v in current[key].items() if k!='pidfd'}) for key,value in measured.items()]
                sample['observed_subset_sum_bytes']=sum(value['fields']['Pss']+value['fields']['SwapPss'] for value in measured.values())
                sample['active_coverage']=dict(measured=len(measured),active=len(current),rate=len(measured)/len(current) if current else 1.,missing=missing)
                sample['coverage']=dict(rate=sample['active_coverage']['rate'],discovered_live=len(current),measured_live=len(measured),missing=missing)
                sample['window_started_unix']=self.window['started_unix'];sample['window_started_monotonic']=self.window['started_monotonic'];sample['window_span_seconds']=now-self.window['started_monotonic']
                sample['cohort_coverage']=dict(processed=len(observed),target=len(cohort),rate=len(observed)/len(cohort) if cohort else 1.)
                sample['fatal_stop']=memory_decision(sample['observed_subset_sum_bytes'],env,True)
                if not missing and not self.pending and sample['window_span_seconds']<=5.:
                    sample['coverage_result']='COMPLETE';sample['pss_total_all_active']=sample['observed_subset_sum_bytes'];sample['pss_swap_bytes']=sample['pss_total_all_active'];sample['coverage_complete']=True
                    sample['totals']={field:sum(value['fields'][field] for value in measured.values()) for field in PSS_FIELDS}
                    self.refresh_deadline=None;self.refresh_needed=False
                else:
                    sample['coverage_result']='PENDING';sample['coverage_complete']=False
                    if not self.pending:
                        self.refresh_needed=True
                        if self.refresh_deadline is None:self.refresh_deadline=now+5.
            else:
                sample.update(pids=[{k:v for k,v in row.items() if k!='pidfd'} for row in current.values()],rss_sum_bytes=sum(row['rss_bytes'] for row in current.values()),coverage_complete=False,coverage=dict(identity_complete=True,pss_rate=None),light_only=True)
                sample['fatal_stop']=memory_decision(None,env,True)
                if self.pending:sample['coverage_result']='PENDING'
            sample['final_identities']=[{k:v for k,v in row.items() if k!='pidfd'} for row in current.values()]
        except Exception as exc:
            sample['fatal_stop']='TELEMETRY_ERROR_PENDING_DEADLINE: '+str(exc) if 'PENDING_DEADLINE' in str(exc) else 'TELEMETRY_ERROR: '+str(exc)
            sample.update(error=traceback.format_exc(),coverage_complete=False,cohort_complete=False)
            sample['final_identities']=[{k:v for k,v in row.items() if k!='pidfd'} for row in self.active.values()]
        if sample['fatal_stop']:sample['coverage_result']='STOP';sample['decision']=sample['fatal_stop']
        sample['observations_this_pass']=observed;sample['exit_transitions']=exited
        sample['pending']=[dict(value,age_seconds=time.monotonic()-value['first_seen_monotonic']) for value in self.pending.values()]
        sample['dispatch_allowed']=sample['coverage_result']=='COMPLETE' and not sample['fatal_stop'] and not self.pending
        sample['sample_finished_unix']=time.time();sample['sampling_seconds']=time.monotonic()-began
        self.total_seconds+=sample['sampling_seconds'];self.samples+=1
        sample.update(monitor_total_seconds=self.total_seconds,monitor_samples=self.samples,active_identities=len(self.active),retired_identities=len(self.retired),open_pidfds=sum(row.get('pidfd') is not None for row in self.active.values()))
        self.next_pending_wake=time.monotonic()+.250 if self.has_pending() else None
        write(self.out/'process_registry.json',dict(active=[{k:v for k,v in row.items() if k!='pidfd'} for row in self.active.values()],retired_count=len(self.retired),pending=sample['pending'],exit_history='lifecycle.jsonl'))
        persist_memory_sample(self.out,sample)
        if sample['fatal_stop']:raise RuntimeError(sample['fatal_stop'])
        return sample
