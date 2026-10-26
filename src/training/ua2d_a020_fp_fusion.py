"""Finite A020 neural fusion cycle. Legacy artifacts are read-only references."""
import copy
import datetime
import json
import math
import os
from pathlib import Path
import random
import signal
import time
import traceback
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from src.modules import ua2d_a020_fp_fusion as m
from src.dataset import ua2d_a020_fp_fusion as d
from src.training.glt_dual_runtime import require_tmux,save_checkpoint
from src.training.ua2d_p10k_nofp_align import finite,metrics,process_identity,process_alive,SOTA,cache_metadata

ROOT=Path(__file__).resolve().parents[2]
PLAN='UA2D-A020-FP-FUSION-20261024-01';REVISION='r1'
OUT=ROOT/'results/ua2d_a020_fp_fusion_20261024/r1';LOG=ROOT/'logs/ua2d_a020_fp_fusion_20261024/r1'
BASE=ROOT/'results/ua2d_p10k_nofp_align_20261024/r1';PARENT=BASE/'pretrain/D_A020/deploy.pt'
DOWN=ROOT/'data/processed/mcl_ph_r4_20260930/downstream_direct_01.staging';SPLIT=ROOT/'data/splits/mips_paper5_outer'
GRAPH=ROOT/'results/ua2d_transfer_20261022/r1/inputs';FP=ROOT/'results/ua2d_fg10k_morgan_20261023/r1/inputs'
HEADS=ROOT/'results/glt_optimization_20261011/r4/U1/initialization/heads_2d'
COMMON=OUT/'initialization/common.pt'
ARMS=m.ARMS;TASKS=m.TASKS;POLICIES=(*ARMS,'NEURAL_LATE');ALL_POLICIES=('B_A020',*POLICIES)
BASE_MACRO=0.785044436513689
FINAL_ORDER=('B_A020','NEURAL_LATE','TMOL_FP','UNI_FP');WEIGHTS=(0.,.25,.5,.75,1.)
LIMITS=dict(prepare_starts=1,ft_starts=175,epochs_started=12250,attempted_updates=280000,
    train_graphs=6575100,validation_started=12250,validation_graphs=1649900,
    test_starts=175,test_graphs=29375,late_units=35,late_graphs=5875,cpu_suites=1)
ZERO_COST=dict(pt_starts=0,pt_updates=0,fp_generation=0,geometry_generation=0,kernel_fit=0,kernel_query=0,
               cache_rebuild=0,baseline_new_inference=0)
DATA_KW=dict(source_root=DOWN,split_root=SPLIT,graph_root=GRAPH,fp_root=FP)


def now():return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
def read(p):return json.loads(Path(p).read_text())
def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n');os.replace(tmp,p)
def exclusive_json(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False)
REPAIR_ROOT=None
ATTEMPT_MAP=None
REPAIR_PLAN='UA2D-A020-FP-FUSION-REPAIR-20261024-01'
REPAIR_REVISION='repair-r1'
RETRAIN_KEYS={'UNI_FP/egc/1','UNI_FP/egc/2'}


def execution_root():return REPAIR_ROOT if REPAIR_ROOT is not None else OUT
def execution_log():return LOG/'controller_repair1' if REPAIR_ROOT is not None else LOG

def validate_attempt_map(value,repair_root,parent_root):
    expected={f'{a}/{task}/{f}' for a in ARMS for task in TASKS for f in range(5)}
    units=value.get('units',{})
    if (value.get('plan')!=PLAN or value.get('revision')!=REVISION or value.get('execution_revision')!=REPAIR_REVISION
        or set(units)!=expected):raise ValueError('fixed175 attempt map identity mismatch')
    old=[];new=[];retrained=[]
    for key,entry in units.items():
        arm,task,fold=key.split('/');relative=Path('finetune')/arm/task/f'fold{fold}'
        if entry.get('source')=='parent_complete':
            if entry.get('attempt')!='r1' or Path(entry['ft_dir'])!=parent_root/relative or key in RETRAIN_KEYS:
                raise ValueError('invalid parent attempt reference '+key)
            old.append(key)
        elif entry.get('source')=='repair':
            if Path(entry['ft_dir'])!=repair_root/relative:raise ValueError('repair path mismatch '+key)
            expected_attempt='repair-r1-retrain' if key in RETRAIN_KEYS else 'repair-r1-first'
            if entry.get('attempt')!=expected_attempt:raise ValueError('unauthorized attempt '+key)
            if key in RETRAIN_KEYS:retrained.append(key)
            new.append(key)
        else:raise ValueError('unknown attempt source '+key)
    if len(old)!=46 or len(new)!=129 or set(retrained)!=RETRAIN_KEYS:raise ValueError('46+129 and exactly two retrain exceptions required')
    return value


def enable_repair(root):
    global REPAIR_ROOT,ATTEMPT_MAP
    root=Path(root)
    if root!=OUT/'repairs/controller_repair1':raise ValueError('only the authorized repair root is permitted')
    mapping=validate_attempt_map(read(root/'attempt_map.json'),root,OUT)
    if REPAIR_ROOT is not None and (REPAIR_ROOT!=root or ATTEMPT_MAP!=mapping):raise ValueError('attempt map cannot change')
    REPAIR_ROOT=root;ATTEMPT_MAP=mapping


def attempt_record(arm,task,fold):
    if ATTEMPT_MAP is None:return None
    return ATTEMPT_MAP['units'][f'{arm}/{task}/{fold}']


def unit_path(arm,task,fold,formal=False):
    if REPAIR_ROOT is None:return OUT/('formal' if formal else 'finetune')/arm/task/f'fold{fold}'
    if formal:return REPAIR_ROOT/'formal'/arm/task/f'fold{fold}'
    return Path(attempt_record(arm,task,fold)['ft_dir'])


def ignore_termination_signals():
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,signal.SIG_IGN)

def base_path(task,fold,formal=False):return BASE/('formal' if formal else 'finetune')/'D_A020'/task/f'fold{fold}'
def config(arm,task,fold):
    if arm not in ARMS or task not in TASKS or type(fold) is not int or fold not in range(5):raise ValueError('outside fixed matrix')
    return dict(plan=PLAN,revision=REVISION,arm=arm,task=task,fold=fold,seed=42+fold,parent=str(PARENT),
        common=str(COMMON),head_initial=str(HEADS/task/f'fold{fold}.pt'),scaler_reference=str(base_path(task,fold)/'scaler.json'),
        source=str(DOWN),split=str(SPLIT/f'{task}.json'),graph=str(GRAPH),fp=str(FP),physical_gpu=3,world_size=1,
        precision='FP32',batch=24,workers=0,epochs=70,head_epochs=10,joint_epochs=60,loss='property_MSE_only',
        architecture=m.ARCHITECTURE,fp_input='zero_after_validation' if arm.endswith('ZERO') else 'real_binary')
def update_plan(status,phase,detail):
    if REPAIR_ROOT is not None:
        path=ROOT/'CURRENT_PLAN.md';s=path.read_text()
        start='\n<!-- UA2D_A020_FP_FUSION_REPAIR_START -->';end='\n<!-- UA2D_A020_FP_FUSION_REPAIR_END -->'
        block=start+f'\n- {now()}；返修状态：**{status}**；阶段：{phase}。\n- {detail}\n- 累计账本：`{REPAIR_ROOT}/budget.json`；原账本/第19节快照只读；独立复审PENDING，未归档。'+end
        if start in s:
            left,rest=s.split(start,1);_,right=rest.split(end,1);s=left+block+right
        else:s=s.rstrip()+'\n'+block+'\n'
        for old in ('待授权','待执行','执行中','待审查','需返修','阻断'):
            s=s.replace('修订：repair-r1；状态：**'+old+'**','修订：repair-r1；状态：**'+status+'**',1)
        path.write_text(s);write(REPAIR_ROOT/'status.json',dict(status=status,phase=phase,detail=detail,time=now(),independent_review='PENDING'))
        return
    path=ROOT/'CURRENT_PLAN.md';s=path.read_text();start='\n<!-- UA2D_A020_FP_FUSION_STATUS_START -->';end='\n<!-- UA2D_A020_FP_FUSION_STATUS_END -->'
    block=start+f'\n- {now()}；状态：**{status}**；阶段：{phase}。\n- {detail}\n- 账本：`{OUT}/budget.json`；manifest：`{OUT}/execution_manifest.json`；独立复审PENDING，未归档。'+end
    if start in s:
        left,rest=s.split(start,1);_,right=rest.split(end,1);s=left+block+right
    else:s=s.rstrip()+'\n'+block+'\n'
    # Update only the current-plan index and current section16 state, never historical records.
    for old in ('待授权','待执行','执行中','待审查','需返修','阻断'):
        s=s.replace('状态**'+old+'**','状态**'+status+'**',1)
        s=s.replace('修订：r1；状态：**'+old+'**','修订：r1；状态：**'+status+'**',1)
    path.write_text(s);write(OUT/'status.json',dict(status=status,phase=phase,detail=detail,time=now(),independent_review='PENDING'))
def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)

def scaler_validate(raw,train,stored):
    if stored.get('fit')!='train' or stored.get('transform')!='standard':raise ValueError('requires inherited train-only standard scaler')
    fitted=StandardScaler().fit(finite(raw)[train].reshape(-1,1))
    expected=dict(mean=fitted.mean_.tolist(),scale=fitted.scale_.tolist(),fit='train',transform='standard')
    if expected!=stored:raise ValueError('inherited scaler does not match train-only labels')
    if not np.isfinite(stored['scale']).all() or min(stored['scale'])<=0:raise ValueError('invalid scaler')
    return stored

def transform(x,scaler,inverse=False):
    x=finite(x);scale=float(scaler['scale'][0]);mean=float(scaler['mean'][0])
    return x*scale+mean if inverse else (x-mean)/scale

def prepare():
    """Exactly one authorized preparation; no forward, generation or cache writing."""
    if REPAIR_ROOT is not None:raise RuntimeError('repair reuses prepare/common; no new preparation')
    require_tmux();start=time.monotonic()
    exclusive_json(OUT/'inputs/prepare_started.json',dict(time=now(),pid=os.getpid(),calls=1))
    parent=torch.load(PARENT,map_location='cpu',weights_only=False);m.validate_parent(parent)
    reference=read(BASE/'final_review/results.json')['formal']['D_A020']
    if reference['macro']!=BASE_MACRO:raise ValueError('formal D_A020 baseline mismatch')
    if read(BASE/'selection/final_frozen.json')['FINAL']!='D_A005':raise ValueError('old FINAL unexpectedly changed')
    evidence=dict(path=str(PARENT),config=parent['config'],attention=parent['attention'],stream_positions=parent['stream_positions'],
        arm=parent['arm'],step=parent['step'],schema=parent['schema'],parent_plan=parent['plan'],
        actual_objective='weighted_align_sum_v1; nominal lambda0.20, historical multiplier2; steady directional mean coefficient0.40',
        random_policy='legacy_offset_v1; 59998 cross-branch shared seeds; historical deviation retained',
        baseline_macro=BASE_MACRO,baseline_results=str(BASE/'final_review/results.json'),old_FINAL='D_A005')
    del parent
    graph=d.InputCache(GRAPH);fp=d.MorganCache(FP,GRAPH,DOWN,SPLIT)
    cache_metadata(GRAPH)
    if set(fp.mapping)!=set(TASKS):raise ValueError('FP tasks mismatch')
    units={};counts=dict(train_graphs=0,validation_graphs=0,test_graphs=0,attempted_updates=0)
    try:
        for task in TASKS:
            rows=sorted(graph.mapping[task]);keys,raw,smiles=d.source_rows(task,rows,DOWN,SPLIT)
            if set(rows)!=set(fp.mapping[task]):raise ValueError('FP/graph row coverage mismatch')
            lookup={row:i for i,row in enumerate(rows)};splits=read(SPLIT/f'{task}.json')
            if len(splits['folds'])!=5:raise ValueError('requires five folds')
            for row,key,sm in zip(rows,keys,smiles):
                d.validate_join(graph.mapping[task].get(row),fp.mapping[task].get(row),row,key,sm)
                graph.get(task,row,key);fp.get(task,row,key)
            graph.memory.clear()
            for fold,entry in enumerate(splits['folds']):
                sets=[set(entry[k]) for k in ('train_indices','validation_indices','test_indices')]
                if any(sets[i]&sets[j] for i in range(3) for j in range(i)) or set.union(*sets)!=set(rows):
                    raise ValueError('split leakage/coverage mismatch')
                scaler=read(base_path(task,fold)/'scaler.json')
                scaler_validate(raw,[lookup[r] for r in entry['train_indices']],scaler)
                bm=read(base_path(task,fold)/'metrics.json');bc=read(base_path(task,fold)/'config.json')
                if bm.get('status')!='COMPLETE' or bm.get('config')!=bc or bc.get('arm')!='D_A020' or bc.get('parent')!=str(PARENT):
                    raise ValueError('baseline BEST reference mismatch')
                paths=dict(best=str(base_path(task,fold)/'best.pt'),head=str(HEADS/task/f'fold{fold}.pt'),
                    scaler_path=str(base_path(task,fold)/'scaler.json'),validation=str(base_path(task,fold)/'validation_predictions.npz'),
                    test=str(base_path(task,fold,True)/'test_predictions.npz'))
                for path in paths.values():
                    if not Path(path).is_file():raise FileNotFoundError(path)
                units[f'{task}/{fold}']=dict(**paths,scaler=scaler,best_epoch=bm['best_epoch'],best_stage=bm['best_stage'],
                    rows={k:sorted(entry[k]) for k in ('train_indices','validation_indices','test_indices')})
                # Load only the original head initialization; no BEST model forward/load.
                head=torch.load(paths['head'],map_location='cpu',weights_only=False)
                if set(head)!={'0.weight','0.bias','3.weight','3.bias'} or head['0.weight'].shape!=(512,512):raise ValueError('original head schema mismatch')
                counts['train_graphs']+=70*len(sets[0]);counts['validation_graphs']+=70*len(sets[1])
                counts['test_graphs']+=len(sets[2]);counts['attempted_updates']+=70*math.ceil(len(sets[0])/24)
    finally:graph.close()
    for key,value in counts.items():
        if value*5!=LIMITS[key]:raise ValueError('registered data budget mismatch '+key)
    if len({e['smiles'] for rows in fp.mapping.values() for e in rows.values()})!=3654:raise ValueError('FP unique structure count mismatch')
    (OUT/'initialization').mkdir(exist_ok=False)
    save_checkpoint(COMMON,m.common_package())
    write(OUT/'initialization/metadata.json',dict(schema=m.COMMON_SCHEMA,branch_seed=m.BRANCH_SEED,architecture=m.ARCHITECTURE,
        path=str(COMMON),parent_reference=str(PARENT),head_references=[u['head'] for u in units.values()],paired_initialization='identical states; input-only ZERO difference',outer_rng_restored=True))
    write(OUT/'inputs/parent_evidence.json',dict(parent=evidence,baseline_units=units,graph_metadata=graph.metadata,
        fp_metadata=fp.metadata,source=str(DOWN),split_root=str(SPLIT),graph_root=str(GRAPH),fp_root=str(FP),
        rows_path=str(FP/'rows.json'),feature_spec=d.FP_SPEC,readonly=True,generated=False,model_forwards=0))
    write(OUT/'inputs/prepared.json',dict(plan=PLAN,revision=REVISION,status='COMPLETE',units=35,records=5875,
        per_arm_counts=counts,seconds=time.monotonic()-start,splits={task:read(SPLIT/f'{task}.json') for task in TASKS}))


def check_prepared():
    p=read(OUT/'inputs/prepared.json')
    if p.get('status')!='COMPLETE' or p.get('plan')!=PLAN or p.get('revision')!=REVISION:raise ValueError('prepare not complete')
    for task in TASKS:
        if read(SPLIT/f'{task}.json')!=p['splits'][task]:raise ValueError('original split changed')
    return p

def save_predictions(path,ds,y,p,**metadata):
    np.savez(path,y_true=finite(y).reshape(-1),y_pred=finite(p).reshape(-1),
             indices=np.array([ds.indices[i] for i in ds.validation]),sample_keys=ds.keys[ds.validation],**metadata)


def frozen_unit(frozen,arm,task,fold):
    if frozen.get('frozen') is not True or frozen.get('plan')!=PLAN or frozen.get('revision')!=REVISION:raise ValueError('TEST before valid freeze')
    selected=frozen['units'][f'{arm}/{task}/{fold}']
    if selected['config']!=config(arm,task,fold) or selected['best']!=str(unit_path(arm,task,fold)/'best.pt'):
        raise ValueError('frozen BEST identity mismatch')
    if REPAIR_ROOT is not None and selected.get('attempt_source')!=attempt_record(arm,task,fold):raise ValueError('frozen attempt changed')
    return selected


def unit(arm,task,fold,formal=False):
    require_tmux()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='3' or os.environ.get('WORLD_SIZE','1')!='1':raise ValueError('GPU3/world1 only')
    torch.set_num_threads(2);check_prepared();cfg=config(arm,task,fold)
    attempt=attempt_record(arm,task,fold)
    if not formal and attempt is not None and attempt['source']!='repair':raise ValueError('completed parent FT cannot be retrained')
    folder=unit_path(arm,task,fold,formal);folder.mkdir(parents=True,exist_ok=False);write(folder/'config.json',cfg)
    state=dict(status='RUNNING',pid=os.getpid(),process=process_identity(os.getpid()),started=now(),
        epochs_started=0,completed_epochs=0,attempted_updates=0,completed_updates=0,train_graphs=0,
        validation_started=0,validation_graphs=0,test_graphs=0,test_started=0,physical_gpu=3,phase='initialize')
    if attempt is not None:state.update(execution_plan=REPAIR_PLAN,execution_revision=REPAIR_REVISION,attempt_source=attempt)
    def record():write(folder/'consumption.json',state)
    record();ds=None;started=time.monotonic()
    try:
        common=torch.load(COMMON,map_location='cpu',weights_only=False)
        evidence=read(OUT/'inputs/parent_evidence.json')['baseline_units'][f'{task}/{fold}']
        scaler=read(evidence['scaler_path'])
        if scaler!=evidence['scaler']:raise ValueError('inherited scaler changed')
        if formal:
            frozen=read(execution_root()/'selection/final_frozen.json');selected=frozen_unit(frozen,arm,task,fold)
            saved=torch.load(selected['best'],map_location='cpu',weights_only=False)
            if any(saved[k]!=selected[k] for k in ('best_epoch','best_stage','scaler')):raise ValueError('frozen checkpoint/scaler mismatch')
            if scaler!=saved['scaler']:raise ValueError('BEST scaler mismatch')
            model=m.restore(saved,cfg,common);best_epoch=saved['best_epoch'];best_stage=saved['best_stage'];del saved
        else:
            seed_all(42+fold)
            parent=None if arm=='FP_MLP' else torch.load(PARENT,map_location='cpu',weights_only=False)
            head=None if arm=='FP_MLP' else torch.load(cfg['head_initial'],map_location='cpu',weights_only=False)
            model=m.initialize(arm,task,common,parent,head);del parent,head
        ds=d.Data(task,fold,formal,**DATA_KW,fp_only=arm=='FP_MLP')
        if formal:
            if selected['test_indices']!=ds.indices:raise ValueError('frozen TEST rows mismatch')
        else:scaler_validate(ds.raw_targets,ds.train,scaler)
        ds.targets=transform(ds.raw_targets,scaler);write(folder/'scaler.json',scaler)
        model.to('cuda:0');validation=d.loader(ds,ds.validation)
        parameter_counts=dict(total=sum(p.numel() for p in model.parameters()),backbone=sum(p.numel() for n,p in model.named_parameters() if n.startswith(('o8.','norm2.'))))
        write(folder/'parameters.json',parameter_counts)
        def evaluate():
            model.eval();truth=[];pred=[];gate_sum=np.zeros(2);gate_count=0
            with torch.inference_mode():
                for batch in validation:
                    batch=batch.to('cuda:0');p=model(batch)
                    if not torch.isfinite(p).all():raise FloatingPointError('nonfinite prediction')
                    truth.append(batch.y.view(-1).cpu().numpy());pred.append(p.cpu().numpy())
                    state['test_graphs' if formal else 'validation_graphs']+=len(p)
                    if hasattr(model,'last_modality_weights'):
                        weights=model.last_modality_weights.float().cpu().numpy();gate_sum+=weights.sum(axis=0);gate_count+=len(weights)
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
                opt,groups=m.optimizer(model,stage)
                scheduler=(torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=10,eta_min=3e-5) if stage=='head'
                    else torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt,T_0=10,T_mult=1,eta_min=1e-6))
                for stage_epoch in range(epochs):
                    tick=time.monotonic();state.update(phase=stage,epochs_started=state['epochs_started']+1,
                        epoch_open=True,epoch_expected_updates=len(train),epoch_expected_train=len(ds.train),epoch_expected_validation=len(ds.validation))
                    record();model.train();losses=[];rates=[g['lr'] for g in opt.param_groups]
                    for batch in train:
                        batch=batch.to('cuda:0');opt.zero_grad(set_to_none=True)
                        output=model(batch);loss=torch.nn.functional.mse_loss(output,batch.y.view(-1))
                        if not torch.isfinite(loss):raise FloatingPointError('nonfinite loss')
                        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True)
                        if state['attempted_updates']>=expected:raise RuntimeError('update allowance exhausted')
                        state['attempted_updates']+=1;state['train_graphs']+=len(output)
                        opt.step();state['completed_updates']+=1;losses.append(float(loss.detach()))
                    scheduler.step();state['validation_started']+=1
                    y,p,met,diag=evaluate();state['completed_epochs']+=1;epoch=state['completed_epochs']
                    if met['r2']>best_r2:
                        best_r2=met['r2'];best_epoch=epoch;best_stage=stage;result=met;diagnostics=diag
                        save_checkpoint(folder/'best.pt',dict(schema=m.SCHEMA,architecture=m.ARCHITECTURE,config=cfg,
                            state_dict={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                            best_epoch=best_epoch,best_stage=best_stage,scaler=scaler,outer_test='NOT_RUN'))
                        save_predictions(folder/'validation_predictions.npz',ds,y,p,best_epoch=best_epoch,best_stage=best_stage)
                    rec=dict(epoch=epoch,stage=stage,stage_epoch=stage_epoch+1,train_loss=float(np.mean(losses)),
                        validation=met,learning_rates=rates,optimizer_groups=groups,modality_diagnostics=diag,
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
        ignore_termination_signals()
        state.update(status='FAILED',error=traceback.format_exc(),ended=now());record()
        write(folder/'failure.json',dict(error=state['error'],automatic_retry=False));raise
    finally:
        ignore_termination_signals()
        if ds is not None:ds.close()


def load_prediction(path):
    with np.load(path,allow_pickle=False) as f:
        x={k:f[k].copy() for k in ('y_true','y_pred','indices','sample_keys')}
    n=len(x['indices'])
    if any(len(v)!=n for v in x.values()) or len({(int(r),str(k)) for r,k in zip(x['indices'],x['sample_keys'])})!=n:
        raise ValueError('prediction identity/shape mismatch')
    finite(x['y_true']);finite(x['y_pred']);return x


def align_predictions(left,right):
    lookup={(int(row),str(key)):i for i,(row,key) in enumerate(zip(right['indices'],right['sample_keys']))}
    identities=[(int(row),str(key)) for row,key in zip(left['indices'],left['sample_keys'])]
    if len(lookup)!=len(identities) or set(lookup)!=set(identities):raise ValueError('prediction row/key sets mismatch')
    order=[lookup[key] for key in identities];p=finite(right['y_pred'])[order];y=finite(right['y_true'])[order]
    if not np.array_equal(finite(left['y_true']),y):raise ValueError('paired original-unit target mismatch')
    return p


def blend(left,right,w):
    if w not in WEIGHTS:raise ValueError('unregistered global weight')
    other=align_predictions(left,right)
    return dict(y_true=left['y_true'],y_pred=w*finite(left['y_pred'])+(1-w)*other,indices=left['indices'],sample_keys=left['sample_keys'])

def summary(scores):
    tasks={task:dict(folds=[float(x) for x in scores[task]],mean=float(np.mean(finite(scores[task]))),
                    std=float(np.std(finite(scores[task]),ddof=1))) for task in TASKS}
    if any(len(scores[t])!=5 for t in TASKS):raise ValueError('five folds required')
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
    if REPAIR_ROOT is not None:
        for task in TASKS:
            for fold in range(5):units[f'{arm}/{task}/{fold}']['attempt_source']=attempt_record(arm,task,fold)
    return dict(**summary(scores),units=units)


def nominate(validation,late_scores):
    if set(validation)!=set((*ARMS,'B_A020')) or set(late_scores)!=set(WEIGHTS):raise ValueError('complete fixed validation matrix required')
    w=max(WEIGHTS,key=lambda w:(late_scores[w]['macro'],w))
    values={a:validation[a]['macro'] for a in FINAL_ORDER if a!='NEURAL_LATE'};values['NEURAL_LATE']=late_scores[w]['macro']
    finite(list(values.values()));final=max(FINAL_ORDER,key=lambda a:(values[a],-FINAL_ORDER.index(a)))
    return w,final,values


def freeze():
    check_prepared()
    if (execution_root()/'selection/final_frozen.json').exists():raise FileExistsError('one freeze only')
    validation={a:aggregate(a) for a in ARMS}
    old=read(BASE/'final_review/results.json')['validation']['D_A020'];validation['B_A020']=old
    late={w:{task:[] for task in TASKS} for w in WEIGHTS}
    refs=read(OUT/'inputs/parent_evidence.json')['baseline_units']
    for task in TASKS:
        for fold in range(5):
            a=load_prediction(refs[f'{task}/{fold}']['validation']);b=load_prediction(unit_path('FP_MLP',task,fold)/'validation_predictions.npz')
            expected=refs[f'{task}/{fold}']['rows']['validation_indices']
            if sorted(a['indices'].tolist())!=expected:raise ValueError('baseline validation split mismatch')
            for w in WEIGHTS:
                merged=blend(a,b,w);late[w][task].append(metrics(merged['y_true'],merged['y_pred'])['r2'])
    late={w:summary(scores) for w,scores in late.items()}
    w,final,values=nominate(validation,late)
    units={k:v for a in ARMS for k,v in validation[a]['units'].items()}
    if len(units)!=175 or len(refs)!=35:raise ValueError('complete matrix required before freeze')
    selection=dict(plan=PLAN,revision=REVISION,validation=validation,late_grid={str(w):x for w,x in late.items()},w=w,FINAL=final,
        candidate_scores=values,selection='validation_only; global_equal7tasks5folds; w tie larger; method tie B,LATE,TMOL,UNI')
    exclusive_json(execution_root()/'selection/validation_selection.json',selection)
    frozen=dict(plan=PLAN,revision=REVISION,time=now(),frozen=True,FINAL=final,w=w,units=units,baseline_units=refs,
        common=str(COMMON),input_references=str(OUT/'inputs/parent_evidence.json'),splits={task:read(SPLIT/f'{task}.json') for task in TASKS},
        validation_scores=values,selection=selection['selection'])
    if REPAIR_ROOT is not None:frozen.update(execution_plan=REPAIR_PLAN,execution_revision=REPAIR_REVISION,attempt_map=ATTEMPT_MAP)
    exclusive_json(execution_root()/'selection/final_frozen.json',frozen);return frozen


def late_unit(task,fold):
    frozen=read(execution_root()/'selection/final_frozen.json')
    if frozen.get('frozen') is not True:raise ValueError('late TEST before freeze')
    folder=unit_path('NEURAL_LATE',task,fold,True);folder.mkdir(parents=True,exist_ok=False)
    a=load_prediction(frozen['baseline_units'][f'{task}/{fold}']['test'])
    b=load_prediction(unit_path('FP_MLP',task,fold,True)/'test_predictions.npz')
    if sorted(a['indices'].tolist())!=frozen['baseline_units'][f'{task}/{fold}']['rows']['test_indices']:raise ValueError('baseline TEST split mismatch')
    result=blend(a,b,frozen['w']);np.savez(folder/'test_predictions.npz',**result)
    write(folder/'metrics.json',dict(status='COMPLETE',task=task,fold=fold,w=frozen['w'],**metrics(result['y_true'],result['y_pred']),
        baseline_reference=frozen['baseline_units'][f'{task}/{fold}']['test'],fp_predictions=str(unit_path('FP_MLP',task,fold,True)/'test_predictions.npz')))


def contrasts(formal,final):
    specs=[('UNI_FP_minus_B_A020','UNI_FP','B_A020'),('TMOL_FP_minus_B_A020','TMOL_FP','B_A020'),
        ('NEURAL_LATE_minus_B_A020','NEURAL_LATE','B_A020'),('UNI_FP_minus_UNI_ZERO','UNI_FP','UNI_ZERO'),
        ('TMOL_FP_minus_TMOL_ZERO','TMOL_FP','TMOL_ZERO'),('UNI_ZERO_minus_B_A020','UNI_ZERO','B_A020'),
        ('TMOL_ZERO_minus_B_A020','TMOL_ZERO','B_A020'),('UNI_FP_minus_TMOL_FP','UNI_FP','TMOL_FP')]
    if final:specs.append(('FINAL_minus_B_A020',final,'B_A020'))
    result={}
    for name,a,b in specs:
        if a not in formal or b not in formal:
            result[name]=dict(status='未完成／不可比较',delta=None);continue
        paired={};positives=0
        for task in TASKS:
            x=finite(formal[a]['tasks'][task]['folds'])-finite(formal[b]['tasks'][task]['folds']);positives+=int((x>0).sum())
            paired[task]=dict(folds=x.tolist(),mean=float(x.mean()),std=float(x.std(ddof=1)),improved_folds=int((x>0).sum()))
        result[name]=dict(status='COMPLETE',delta=float(formal[a]['macro']-formal[b]['macro']),tasks=paired,improved_folds=positives)
    return result
