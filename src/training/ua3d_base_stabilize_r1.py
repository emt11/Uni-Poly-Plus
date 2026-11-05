"""Independent BASE stabilization lifecycle, artifacts and math; no old tensor loads."""
import copy,json,math,os,time,traceback,shutil
from pathlib import Path
import numpy as np
import torch
from src.modules import ua3d_base_stabilize_r1 as m
from src.dataset import ua3d_base_stabilize_r1 as d
from src.training.ua3m_paperfuse import read,write,exclusive,now,ignore_termination,seed_all,finite,metrics,process_identity,scaler_validate,transform,summary,SOTA,require_tmux,pt_learning_rate,window_objective,save_predictions
from src.training.ua3d_cond_spectral_r6 import MemoryMonitor as ParentMemoryMonitor,pid_identity,identity_status,identity_key
ROOT=Path(__file__).resolve().parents[2];PLAN=m.PLAN;REVISION='r1';EXECUTION_REVISION='gpu2-pt1-fttest4-base-stabilize-r1'
OUT=ROOT/'results/ua3d_base_stabilize_20261105/r1';LOG=ROOT/'logs/ua3d_base_stabilize_20261105/r1'
B=ROOT/'results/ua3d_base_opt_20261026/r6/repairs/concurrency_r1';P6=ROOT/'results/ua3d_cond_spectral_20261104/r2/repairs/mm_lifecycle_r6';D=ROOT/'results/ua3d_stability_20261028/r1'
DOWN=ROOT/'data/processed/mcl_ph_r4_20260930/downstream_direct_01.staging';SPLIT=ROOT/'data/splits/mips_paper5_outer';PT_CACHE=ROOT/'results/ua2d_pretrain_fp_20261019/fast_r4/prepared_cache';MAP_ROOT=ROOT/'results/ua2d_ptscale_20261022/r1/inputs'
INITIAL=OUT/'initialization/common_step0.pt';TASKS=m.TASKS;GROUPS=m.GROUPS;CANONICAL=m.CANONICAL;REFERENCE=.804675462276949;UUID='GPU-31456dd5-a421-142b-644b-7a80f482dfb9'

def seed_all(seed):
    import random
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)

def spec():return read(ROOT/'configs/ua3d_base_stabilize_r1_gpu2.json')
def base_path(task,fold):return B/'finetune/F_PD_N1_F0'/task/f'fold{fold}'
def head_path(task,fold):return OUT/'initialization/heads'/task/f'fold{fold}.pt'
def pt_path(run):
    if run not in ('PT_R0','PT_R1'):raise ValueError('PT whitelist')
    return OUT/'pretrain'/run/'deploy.pt'
def parent_path(group):return pt_path('PT_R1' if group=='S42_B' else 'PT_R0')
def unit_path(group,task,fold,estimator=None):return OUT/('formal' if estimator else 'finetune')/group/task/f'fold{fold}'/estimator if estimator else OUT/'finetune'/group/task/f'fold{fold}'
def config(group,task,fold):
    if group not in GROUPS or task not in TASKS or fold not in range(5):raise ValueError('matrix identity')
    source=read(base_path(task,fold)/'config.json')
    return dict(plan=PLAN,revision=REVISION,run=group,task=task,fold=fold,seed=m.SEEDS[group]+fold,seed_base=m.SEEDS[group],parent=str(parent_path(group)),head=str(head_path(task,fold)),input_cache=source['input_cache'],scaler_reference=source['scaler_reference'],model=m.CONFIG,batch=24,epochs=70,head_epochs=10,workers=0,precision='FP32',world_size=1,selection='strict original-unit validation RMSE',swa_epochs=[40,50,60,70],numerical_policy='DET')
def check_clock(prepare=False):
    c=read(OUT/'authorization/clock.json')
    if time.time()>=(c['prepare_deadline_unix'] if prepare else c['deadline_unix']):raise TimeoutError('EXPIRED: independent stabilization clock')
def check_space():
    free=shutil.disk_usage(ROOT).free
    if free<60*1024**3:raise RuntimeError('DISK_STOP: free space below60GiB')
    return free

def save_checkpoint(path,payload):
    path=Path(path)
    if not path.resolve().is_relative_to(OUT.resolve()):raise ValueError('write outside new root')
    path.parent.mkdir(parents=True,exist_ok=True)
    from src.training.glt_dual_runtime import save_checkpoint as save
    save(path,payload)
def load_tensor(path):
    path=Path(path).resolve()
    if not path.is_relative_to(OUT.resolve()) or path.suffix!='.pt':raise ValueError('historical model tensors forbidden')
    return torch.load(path,map_location='cpu',weights_only=False)
def gpu_contract():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='2' or os.environ.get('WORLD_SIZE')!='1' or torch.cuda.device_count()!=1:raise ValueError('GPU2/world1 contract')
    actual='GPU-'+str(torch.cuda.get_device_properties(0).uuid).removeprefix('GPU-')
    if actual!=UUID:raise ValueError('GPU2 UUID mismatch')
    return dict(physical_gpu=2,logical_device='cuda:0',uuid=actual)
def install_stop_handler():
    import signal
    def stop(sig,frame):ignore_termination();raise RuntimeError('controller termination signal '+str(sig))
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
def update_plan(status,phase,detail):
    p=ROOT/'CURRENT_PLAN.md';s=p.read_text();lines=s.splitlines()
    for i,line in enumerate(lines):
        if line.startswith('> 状态：'):lines[i]=f'> 状态：**{status}**；阶段 {phase}；更新时间 {now()}。';break
    s='\n'.join(lines)+'\n';start='<!-- BASE_STABILIZE_R1_STATUS_START -->';end='<!-- BASE_STABILIZE_R1_STATUS_END -->';left,rest=s.split(start,1);_,right=rest.split(end,1)
    p.write_text(left+start+'\n'+f'{now()} {status}／{phase}：{detail}\nindependent_review=PENDING；BASE不替换。\n'+end+right)
    write(OUT/'status.json',dict(status=status,phase=phase,detail=detail,time=now(),independent_review='PENDING'))
class MemoryMonitor(ParentMemoryMonitor):
    def __init__(self,out=OUT):super().__init__(out)
    def register(self,entry,role):
        key=super().register(entry,role)
        if key and role=='descendant':
            row=self.active[key];parent=next((p for p in self.active.values() if p['pid']==row.get('ppid') and p['role']=='pt'),None);actual=pid_identity(parent['pid']) if parent else None
            if parent and actual and actual['start_ticks']==parent['start_ticks'] and '(pt_data_worker)' in row.get('raw_stat',''):
                row.update(no_mm_eligible=True,mm_ancestry=[{k:parent[k] for k in ('pid','start_ticks','role')}],dataloader_role_evidence=dict(parent_identity=actual,registered_parent_role='pt',loader_workers=4,start_method='spawn',prefetch=1))
        return key

def verify_runtime(model,formal=False):
    m.check_det()
    if model.ft!=m.FT or model.encoder3d.mask_token.requires_grad:raise ValueError('BASE FT recipe/frozen mask mismatch')
    for name,p in [('o8',0.),('encoder3d',0.),('fp_encoder',.2),('head',.5)]:
        mods=[x for x in getattr(model,name).modules() if isinstance(x,torch.nn.Dropout)]
        if not mods or any(x.p!=p for x in mods):raise ValueError('dropout mismatch '+name)
    if formal and any(x.training for x in model.modules()):raise ValueError('all evaluation modules must be eval')
    if any(x.is_floating_point() and x.dtype!=torch.float32 for x in model.state_dict().values()):raise ValueError('FT state must be FP32')

def unit(group,task,fold,estimator=None):
    require_tmux();install_stop_handler();m.check_det();check_clock();check_space();cfg=config(group,task,fold)
    if estimator is not None and (group not in CANONICAL or estimator not in ('BEST','SWA')):raise ValueError('formal source whitelist')
    folder=unit_path(group,task,fold,estimator);folder.mkdir(parents=True,exist_ok=False);write(folder/'config.json',cfg)
    state=dict(status='RUNNING',pid=os.getpid(),process=process_identity(os.getpid()),started=now(),epochs_started=0,completed_epochs=0,attempted_updates=0,completed_updates=0,train_graphs=0,validation_started=0,validation_graphs=0,swa_validation_started=0,swa_validation_graphs=0,test_started=0,test_graphs=0,historical_tensor_reads=0,model_reads=dict(new_PT=0,new_head=0,new_estimator=0),phase='initialize')
    def record():write(folder/'consumption.json',state)
    record();ds=None;started=time.monotonic()
    try:
        state['actual_device']=gpu_contract();record();scaler=read(cfg['scaler_reference'])
        if estimator:
            seed_all(cfg['seed']);frozen=read(OUT/'selection/final_frozen.json');key=f'{group}/{task}/{fold}/{estimator}';entry=frozen['units'][key]
            if frozen['plan']!=PLAN or frozen['FINAL_ESTIMATOR']!='BASE_SWA_LAST4_CYCLES' or entry['config']!=cfg:raise ValueError('frozen estimator identity')
            expected=unit_path(group,task,fold)/('best.pt' if estimator=='BEST' else 'swa.pt')
            if entry['checkpoint']!=str(expected):raise ValueError('frozen source path')
            state['model_reads']['new_estimator']+=1;record();pack=load_tensor(expected)
            if pack.get('schema')!=m.FT_SCHEMA or pack['config']!=cfg or pack['estimator']!=estimator or pack['scaler']!=scaler:raise ValueError('new estimator schema/scaler')
            model=m.base.BaseOptFuse(False,ft=m.FT);m.strict_module_state(model,pack['state_dict']);model.stage='joint';del pack
        else:
            if not read(OUT/'comparisons/PT.json')['equal']:raise ValueError('FT before PT exact comparison')
            seed_all(cfg['seed']);state['model_reads']['new_PT']+=1;record();pack=load_tensor(parent_path(group));state['model_reads']['new_head']+=1;record();head=load_tensor(head_path(task,fold));model=m.initialize_ft(pack,head,task,fold);del pack,head
            if group in ('S42_A','S42_B'):save_checkpoint(folder/'initial_state.pt',dict(state_dict=m.clone_tree(model.state_dict()),seed=cfg['seed'],source='own compared new PT and common new random head'))
        verify_runtime(model);ds=d.Data(task,fold,bool(estimator),DOWN,SPLIT,cfg['input_cache']);ds.targets=transform(ds.raw_targets,scaler)
        if estimator:
            if entry['test_indices']!=ds.indices or entry['test_keys']!=ds.keys.tolist():raise ValueError('frozen row/key identity')
        else:scaler_validate(ds.raw_targets,ds.train,scaler)
        write(folder/'scaler.json',scaler);write(folder/'environment.json',dict(policy=m.check_det(),torch=torch.__version__,cuda=torch.version.cuda,actual_device=state['actual_device']))
        model.to('cuda:0');validation=d.loader(ds,ds.validation)
        def evaluate(kind):
            ds.record_training=False;model.eval();verify_runtime(model,True);truth=[];prediction=[]
            with torch.no_grad():
                for batch in validation:
                    batch=batch.to('cuda:0');p,_,_=model.predict(batch)
                    if not torch.isfinite(p).all():raise FloatingPointError('nonfinite evaluation')
                    truth.append(batch.y.view(-1).cpu().numpy());prediction.append(p.cpu().numpy());state[{'validation':'validation_graphs','swa':'swa_validation_graphs','test':'test_graphs'}[kind]]+=len(p)
            y=transform(np.concatenate(truth),scaler,True);p=transform(np.concatenate(prediction),scaler,True)
            return y,p,metrics(y,p)
        if estimator:
            state.update(phase='TEST',test_started=1);record();y,p,met=evaluate('test');save_predictions(folder/'test_predictions.npz',ds,y,p,estimator=estimator)
            write(folder/'identity_alignment.json',dict(indices=ds.indices,sample_keys=ds.keys.tolist(),raw_targets=ds.raw_targets.tolist(),scaler=scaler,matched=True));write(folder/'metrics.json',dict(status='COMPLETE',config=cfg,estimator=estimator,scaler=scaler,**met))
        else:
            train=d.loader(ds,ds.train,True,cfg['seed']);expected=70*len(train);seed_all(cfg['seed']);history=[];best_rmse=math.inf;average=m.SWA();best=None
            for stage,epochs in [('head',10),('joint',60)]:
                opt,groups=m.optimizer_ft(model,stage);scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=10,eta_min=3e-5) if stage=='head' else torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt,T_0=10,T_mult=1,eta_min=1e-6)
                for se in range(epochs):
                    check_clock();torch.cuda.synchronize();tick=time.monotonic();epoch=len(history)+1;state.update(phase=stage,epochs_started=epoch,epoch_open=True,epoch_expected_updates=len(train));record();ds.record_training=True;ds.training_rows=[];model.train();verify_runtime(model)
                    losses=[];components=[];clips=[];boundaries=[];rates=[g['lr'] for g in opt.param_groups]
                    for batch in train:
                        batch=batch.to('cuda:0');opt.zero_grad(set_to_none=True);state['attempted_updates']+=1;state['train_graphs']+=len(batch.y)
                        if state['attempted_updates']>expected:raise RuntimeError('FT budget exceeded')
                        loss,parts=model.training_loss(batch)
                        if not torch.isfinite(loss):raise FloatingPointError('nonfinite loss')
                        loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True);opt.step();state['completed_updates']+=1
                        losses.append(float(loss.detach()));components.append({k:float(v) for k,v in parts.items()});clips.append(float(norm));boundaries.append(len(ds.training_rows))
                    scheduler.step();torch.cuda.synchronize();training_seconds=time.monotonic()-tick;rows=ds.training_rows.copy();state['validation_started']+=1;y,p,met=evaluate('validation');state['completed_epochs']+=1
                    (folder/'validation').mkdir(exist_ok=True);save_predictions(folder/'validation'/f'epoch{epoch:03d}.npz',ds,y,p,epoch=epoch)
                    if met['rmse']<best_rmse:
                        best_rmse=met['rmse'];best=dict(best_epoch=epoch,best_stage=stage,**met)
                        save_checkpoint(folder/'best.pt',dict(schema=m.FT_SCHEMA,plan=PLAN,config=cfg,estimator='BEST',state_dict=m.clone_tree(model.state_dict()),scaler=scaler,**best));save_predictions(folder/'validation_predictions.npz',ds,y,p,best_epoch=epoch)
                    if epoch in (40,50,60,70):average.collect(model,epoch)
                    rec=dict(epoch=epoch,stage=stage,stage_epoch=se+1,losses=losses,components=components,clip_norms=clips,training_rows=rows,batch_boundaries=boundaries,validation=met,learning_rates=rates,optimizer_groups=groups,runtime=model.runtime_contract(),training_seconds=training_seconds,seconds=time.monotonic()-tick,gpu_peak_bytes=torch.cuda.max_memory_allocated())
                    history.append(rec);write(folder/'history.json',history);print(json.dumps(rec),flush=True);state.update(epoch_open=False);record()
            swa=average.state();save_checkpoint(folder/'swa.pt',dict(schema=m.FT_SCHEMA,plan=PLAN,config=cfg,estimator='SWA',state_dict=swa,scaler=scaler,swa_epochs=average.epochs,n_averaged=4));write(folder/'swa_sampling.json',dict(epochs=average.epochs,n_averaged=4,accumulation_dtype='float64',parameters={k:str(v) for k,v in average.dtypes.items()},buffer_names=list(average.buffers)))
            m.strict_module_state(model,swa);del swa,average;state.update(phase='SWA_validation',swa_validation_started=1);record();y,p,swa_met=evaluate('swa');save_predictions(folder/'swa_validation_predictions.npz',ds,y,p,estimator='SWA');write(folder/'metrics.json',dict(status='COMPLETE',config=cfg,scaler=scaler,BEST=best,SWA=swa_met))
            if state['completed_epochs']!=70 or state['completed_updates']!=expected:raise ValueError('incomplete FT')
        state.update(status='COMPLETE',ended=now(),elapsed_seconds=time.monotonic()-started,gpu_peak_bytes=torch.cuda.max_memory_allocated());record()
    except BaseException:
        ignore_termination();state.update(status='FAILED',error=traceback.format_exc(),ended=now());record();write(folder/'failure.json',dict(error=state['error'],automatic_retry=False));raise
    finally:
        ignore_termination()
        if ds is not None:ds.close()

def compared_record(path,left,right,label):
    result=m.exact_compare(left,right);write(path,dict(label=label,time=now(),**result))
    if not result['equal']:raise RuntimeError(label+': '+str(result['first_mismatch']))
    return result

def compare_pt():
    require_tmux();path=OUT/'comparisons/PT.json'
    if path.exists():raise FileExistsError('PT pair already compared')
    result=dict(equal=True,history_updates=0,sampling_microbatches=0,checkpoints={},historical_tensor_reads=0,extra_forwards=0)
    try:
        dirs=[pt_path(r).parent for r in ('PT_R0','PT_R1')]
        for folder in dirs:
            cons=read(folder/'consumption.json')
            if cons['status']!='COMPLETE' or cons['completed_updates']!=10000 or cons['pt_microbatches']!=60000 or cons['pt_presentations']!=10080000 or cons['stream']['counters']!=[60000]*3 or cons['initial']!=str(INITIAL):raise ValueError('PT pair incomplete/source differs')
        for filename,excluded,expected,counter in [('history.jsonl',{'seconds','gpu_peak_bytes'},10000,'history_updates'),('sampling.jsonl',{'io'},60000,'sampling_microbatches')]:
            with (dirs[0]/filename).open() as fa,(dirs[1]/filename).open() as fb:
                for index in range(expected):
                    sa,sb=fa.readline(),fb.readline()
                    if not sa or not sb:raise ValueError('PT comparison truncated '+filename)
                    aa,bb=json.loads(sa),json.loads(sb);aa={k:v for k,v in aa.items() if k not in excluded};bb={k:v for k,v in bb.items() if k not in excluded}
                    if aa!=bb:
                        diff=m.exact_compare(aa,bb);result['first_difference']=dict(file=filename,index=index,**diff);raise ValueError('PT numerical/sampling difference')
                    result[counter]+=1
                if fa.readline() or fb.readline():raise ValueError('extra PT comparison rows')
        for filename in ['step5000.pt','step10000.pt','deploy.pt']:
            values=[load_tensor(p/filename) for p in dirs]
            # Normal packages contain no timestamps/PIDs; every mathematical leaf is compared.
            diff=m.exact_compare(*values);result['checkpoints'][filename]=diff;del values
            if not diff['equal']:result['first_difference']=dict(file=filename,**diff);raise ValueError('PT tensor/optimizer/RNG difference')
        result.update(time=now());exclusive(path,result)
    except BaseException:
        result.update(equal=False,error=traceback.format_exc());exclusive(path,result);raise RuntimeError('PT_REPRODUCIBILITY_FAILED: '+str(result.get('first_difference',result['error'])))

def compare_pair(task,fold):
    require_tmux();path=OUT/'comparisons/FT'/task/f'fold{fold}.json'
    if path.exists():raise FileExistsError('FT pair already compared')
    result=dict(task=task,fold=fold,equal=True,checks={},extra_forwards=0)
    try:
        dirs=[unit_path(g,task,fold) for g in ('S42_A','S42_B')]
        cons=[read(p/'consumption.json') for p in dirs]
        if any(c['status']!='COMPLETE' or c['completed_epochs']!=70 or c['swa_validation_started']!=1 for c in cons) or cons[0]['process']==cons[1]['process']:raise ValueError('independent complete FT pair required')
        def check(name,a,b):
            v=m.exact_compare(a,b);result['checks'][name]=v
            if not v['equal']:result['first_difference']=dict(check=name,**v);raise ValueError('FT strict mismatch '+name)
        cfgs=[read(p/'config.json') for p in dirs];check('mathematical_config',*[{k:v for k,v in c.items() if k not in ('run','parent')} for c in cfgs])
        histories=[read(p/'history.json') for p in dirs]
        for epoch in range(70):
            check(f'epoch{epoch+1}',*[{k:v for k,v in h[epoch].items() if k not in ('seconds','training_seconds','gpu_peak_bytes')} for h in histories])
            vals=[]
            for p in dirs:
                with np.load(p/'validation'/f'epoch{epoch+1:03d}.npz',allow_pickle=False) as f:vals.append({k:f[k].copy() for k in f.files})
            check(f'validation{epoch+1}',*vals)
        for filename in ('initial_state.pt','best.pt','swa.pt'):
            values=[load_tensor(p/filename) for p in dirs];check(filename+'/parameters',*[x['state_dict'] for x in values])
            if filename!='initial_state.pt':check(filename+'/metadata',*[{k:v for k,v in x.items() if k not in ('state_dict','config')} for x in values])
            del values
        for filename in ('validation_predictions.npz','swa_validation_predictions.npz'):
            vals=[]
            for p in dirs:
                with np.load(p/filename,allow_pickle=False) as f:vals.append({k:f[k].copy() for k in f.files})
            check(filename,*vals)
        result.update(time=now());exclusive(path,result)
    except BaseException:
        result.update(equal=False,error=traceback.format_exc());exclusive(path,result);raise RuntimeError('REPRODUCIBILITY_FAILED: '+str(result.get('first_difference',result['error'])))

def aggregate(group,estimator,formal=False):
    scores={task:[] for task in TASKS};units={}
    for task in TASKS:
        for fold in range(5):
            root=unit_path(group,task,fold,estimator if formal else None);cons=read(root/'consumption.json');met=read(root/'metrics.json');cfg=config(group,task,fold);e=read(SPLIT/f'{task}.json')['folds'][fold]
            if cons['status']!='COMPLETE' or met['config']!=cfg:raise ValueError('incomplete estimator/config')
            if formal:
                if cons['test_graphs']!=len(e['test_indices']) or cons['test_started']!=1 or met['estimator']!=estimator:raise ValueError('formal identity/consumption')
                score=met['r2']
            else:
                expected=dict(completed_epochs=70,attempted_updates=70*math.ceil(len(e['train_indices'])/24),completed_updates=70*math.ceil(len(e['train_indices'])/24),train_graphs=70*len(e['train_indices']),validation_started=70,validation_graphs=70*len(e['validation_indices']),swa_validation_started=1,swa_validation_graphs=len(e['validation_indices']),epoch_open=False,test_started=0)
                if any(cons.get(k)!=v for k,v in expected.items()):raise ValueError('full FT/validation/SWA accounting')
                score=met[estimator]['r2']
            if not math.isfinite(score):raise FloatingPointError('nonfinite aggregate')
            scores[task].append(score);ft=unit_path(group,task,fold)
            units[f'{group}/{task}/{fold}/{estimator}']=dict(group=group,task=task,fold=fold,estimator=estimator,config=cfg,checkpoint=str(ft/('best.pt' if estimator=='BEST' else 'swa.pt')),scaler=met['scaler'],test_indices=sorted(e['test_indices']),BEST_epoch=read(ft/'metrics.json')['BEST']['best_epoch'],SWA_epochs=[40,50,60,70])
    return dict(**summary(scores),units=units)

def freeze():
    if not read(OUT/'comparisons/PT.json')['equal']:raise ValueError('PT reproduction not passed')
    for task in TASKS:
        for fold in range(5):
            if not read(OUT/'comparisons/FT'/task/f'fold{fold}.json')['equal']:raise ValueError('35 FT comparisons required')
    val={g:{e:aggregate(g,e) for e in ('BEST','SWA')} for g in CANONICAL}
    macro_delta=float(np.mean([val[g]['SWA']['macro']-val[g]['BEST']['macro'] for g in CANONICAL]));task_delta={task:float(np.mean([val[g]['SWA']['tasks'][task]['mean']-val[g]['BEST']['tasks'][task]['mean'] for g in CANONICAL])) for task in TASKS}
    passed=macro_delta>=-.005 and all(x>=-.020 for x in task_delta.values());exclusive(OUT/'gates/validation_quality.json',dict(passed=passed,macro_delta=macro_delta,task_delta=task_delta,validation=val))
    if not passed:raise RuntimeError('VALIDATION_QUALITY_GUARD_FAILED: no BEST fallback/formal TEST')
    units={k:v for g in CANONICAL for e in ('BEST','SWA') for k,v in val[g][e]['units'].items()}
    for unit in units.values():
        rows=read(Path(unit['config']['input_cache'])/'rows.json')[unit['task']];keys={r['row']:r['key'] for r in rows};unit['test_keys']=[keys[i] for i in unit['test_indices']]
    exclusive(OUT/'selection/final_frozen.json',dict(plan=PLAN,revision='r1',frozen=True,frozen_unix=time.time(),time=now(),FINAL_ESTIMATOR='BASE_SWA_LAST4_CYCLES',PRIMARY_RUN='S42_A',units=units,validation_guard=passed,reference=REFERENCE,test_targets_read=False))

def report_results(execution_complete=False):
    frozen=read(OUT/'selection/final_frozen.json');formal={g:{e:aggregate(g,e,True) for e in ('BEST','SWA')} for g in CANONICAL};reference=read(OUT/'recipe_audit.json')['reference_tasks'];stats={}
    for est in ('BEST','SWA'):
        macros=[formal[g][est]['macro'] for g in CANONICAL];task_ranges={task:float(np.ptp([formal[g][est]['tasks'][task]['mean'] for g in CANONICAL])) for task in TASKS};task_means={task:float(np.mean([formal[g][est]['tasks'][task]['mean'] for g in CANONICAL])) for task in TASKS}
        mean=float(np.mean(macros));std=float(np.std(macros,ddof=1));span=float(np.ptp(macros));stable=std<=.003 and span<=.006 and all(v<=.020 for v in task_ranges.values());retained=mean>=REFERENCE-.005 and macros[0]>=REFERENCE-.005 and min(macros)>=REFERENCE-.010 and all(task_means[task]>=reference['tasks'][task]['mean']-.020 for task in TASKS)
        stats[est]=dict(macros=macros,mean=mean,std=std,range=span,task_means=task_means,task_ranges=task_ranges,ft_seed_stability_pass=stable,quality_retained=retained)
    primary=formal['S42_A']['SWA'];pairs={g:{task:[a-b for a,b in zip(formal[g]['SWA']['tasks'][task]['folds'],formal[g]['BEST']['tasks'][task]['folds'])] for task in TASKS} for g in CANONICAL}
    v=dict(plan=PLAN,revision='r1',FINAL_ESTIMATOR=frozen['FINAL_ESTIMATOR'],PRIMARY_RUN='S42_A',formal=formal,seed_statistics=stats,macro=primary['macro'],delta_BASE=primary['macro']-REFERENCE,SWA_minus_BEST=pairs,primary_SWA_minus_BEST=primary['macro']-formal['S42_A']['BEST']['macro'],same_seed_PT_pass=True,same_seed_FT_pairs_pass=35,same_seed_repro_pass=True,execution_complete=execution_complete,ft_seed_stability_pass=stats['SWA']['ft_seed_stability_pass'],quality_retained=stats['SWA']['quality_retained'],stable_ready=(stats['SWA']['ft_seed_stability_pass'] and stats['SWA']['quality_retained']) if execution_complete else None,target_met=primary['macro']>=.810,performance_pass=primary['macro']-REFERENCE>=.010,independent_review='PENDING',evaluation='shared development; one PT seed42 and three FT seeds; not distinct PT-seed stability',paper_comparison={task:dict(project_mean=primary['tasks'][task]['mean'],project_std=primary['tasks'][task]['std'],paper_model=SOTA[task][1],paper_mean=SOTA[task][2],delta=primary['tasks'][task]['mean']-SOTA[task][2]) for task in TASKS})
    write(OUT/'final_review/results.json',v)
    lines=['# BASE stabilization formal shared-development evaluation',f"PRIMARY=S42_A; ESTIMATOR=BASE_SWA_LAST4_CYCLES; macro={v['macro']:.3f}; ΔBASE={v['delta_BASE']:+.3f}; stable_ready={v['stable_ready']}",'Paper per-task envelope; numerical comparison only; split/protocol equivalence unverified.','','|Task|Project mean ± std|Paper model|Paper mean|Delta|','|---|---:|---|---:|---:|']
    for task,p in v['paper_comparison'].items():lines.append(f"|{task}|{p['project_mean']:.3f} ± {p['project_std']:.3f}|{p['paper_model']}|{p['paper_mean']:.3f}|{p['delta']:+.3f}|")
    for est,s in stats.items():lines.append(f"{est}: seed macro mean={s['mean']:.3f}, std={s['std']:.3f}, range={s['range']:.3f}; task ranges="+'; '.join(f'{k}={x:.3f}' for k,x in s['task_ranges'].items()))
    (OUT/'final_review/report.md').write_text('\n\n'.join(lines)+'\n');return v
