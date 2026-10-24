"""Independent finite UA2D-TRANSFER cycle; no old controller or ledger imports."""
import copy
import datetime
import json
import math
import os
from pathlib import Path
import random
import time
import traceback
import lmdb
import numpy as np
import torch
from torch import nn
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
from src.dataset.glt_dual import glt_2d_collate as collate
from src.dataset.ua2d_fast_pretrain import decode_template
from src.modules.ua2d_transfer import ARMS,SCHEMA,ARCHITECTURE,BRANCH_SEED,initialize,restore,optimizer
from src.training.glt_dual_runtime import require_tmux,save_checkpoint
ROOT=Path(__file__).resolve().parents[2]
PLAN='UA2D-TRANSFER-20261022-01'
REVISION='r1'
OUT=ROOT/'results/ua2d_transfer_20261022/r1'
LOG=ROOT/'logs/ua2d_transfer_20261022/r1'
DOWN=ROOT/'data/processed/mcl_ph_r4_20260930/downstream_direct_01.staging'
SPLIT=ROOT/'data/splits/mips_paper5_outer'
PARENT=ROOT/'results/ua2d_pretrain_fp_20261019/fast_r4/pretrain/B_FP/deploy.pt'
HEADS=ROOT/'results/glt_optimization_20261011/r4/U1/initialization/heads_2d'
REFERENCE_BFP=PARENT.parents[2]/'final_review/results.json'
REFERENCE_OPT=ROOT/'results/ua2d_opt_20261021/r1/repair_ft1/results.json'
TASKS=('eea','egb','egc','ei','eps','nc','xc')
CACHE_SCHEMA='ua2d-transfer-input-2hop-r1'
LIMITS=dict(prepare_starts=1,initialization_starts=1,pt_starts=0,pt_updates=0,pt_presentations=0,
    ft_starts=140,epochs_started=9800,validation_started=9800,attempted_updates=224000,
    test_starts=140,test_graphs=23500,cpu_suites=1)
PROFILE=dict(batch=24,amp='fp32',workers=0,drop_last=False,head_epochs=10,joint_epochs=60,
    weight_decay=.02,clip=5.,encoder_dropout=0.,head_dropout=.1,head_lr=.0003,
    backbone_joint_lr=.0001,head_joint_lr=.001,
    head_scheduler=dict(name='CosineAnnealingLR',T_max=10,eta_min=.00003),
    joint_scheduler=dict(name='CosineAnnealingWarmRestarts',T_0=10,T_mult=1,eta_min=.000001))
SOTA={'eea':('Eea','Periodic-TDL',.934),'egb':('Egb','Periodic-TDL (+Chem, +RC)',.923),
    'egc':('Egc','Periodic-TDL (+Chem, +RC)',.911),'ei':('Ei','Periodic-TDL',.839),
    'eps':('EPS','Periodic-TDL',.787),'nc':('Nc','Periodic-TDL',.856),'xc':('Xc','Periodic-TDL',.423)}


def now():return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
def read(path):return json.loads(Path(path).read_text())
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');os.replace(tmp,path)
def finite(x):
    a=np.asarray(x,dtype=np.float64)
    if not np.isfinite(a).all():raise FloatingPointError('nonfinite result')
    return a
def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
def unit_path(arm,task,fold,formal=False):return OUT/('formal' if formal else 'finetune')/arm/task/f'fold{fold}'
def config(arm,task,fold):
    if arm not in ARMS or task not in TASKS or fold not in range(5):raise ValueError('outside transfer matrix')
    return dict(plan=PLAN,revision=REVISION,arm=arm,task=task,fold=fold,seed=42+fold,parent=str(PARENT),
        head_initial=str(HEADS/task/f'fold{fold}.pt'),residual_initial=str(OUT/'initialization/residual.pt'),
        source=str(DOWN),split=str(SPLIT/f'{task}.json'),cache=str(OUT/'inputs'),cache_schema=CACHE_SCHEMA,
        architecture=ARCHITECTURE,profile=PROFILE,readout='original_mean' if arm=='A_SP' else ('mean_residual' if arm=='R_MEAN' else 'std_residual'),
        anchor_enabled=arm in ('A_SP','R_STD_SP'),anchor_scope=['o8','norm2'],anchor_coefficient=.5e-3,
        anchor_origin=str(PARENT),input_semantics='original_2hop_2d')


def update_plan(status,phase,detail):
    p=ROOT/'CURRENT_PLAN.md';text=p.read_text();start='\n### UA2D-TRANSFER 最新执行状态\n';end='\n<!-- UA2D_TRANSFER_EXECUTION_END -->'
    entry=start+f'\n- {now()}；状态：{status}；阶段：{phase}。\n- {detail}\n- 交付：{OUT}；日志：{LOG}；DSH执行自检不构成Codex独立审查。\n'+end
    if start in text:
        before,rest=text.split(start,1);_,after=rest.split(end,1);text=before+entry+after
    else:text=text.rstrip()+'\n'+entry+'\n'
    p.write_text(text);write(OUT/'status.json',dict(plan=PLAN,revision=REVISION,status=status,phase=phase,time=now(),detail=detail,independent_review='PENDING'))


def cache_metadata(root):
    root=Path(root);m=read(root/'metadata.json');coverage=read(root/'coverage.json')
    if (m.get('plan')!=PLAN or m.get('schema')!=CACHE_SCHEMA or m.get('status')!='COMPLETE'
        or m.get('records')!=5875 or m.get('source')!=str(DOWN) or m.get('split')!=str(SPLIT)
        or m.get('max_hop')!=2 or m.get('online_fallback') is not False
        or coverage.get('status')!='COMPLETE' or coverage.get('records')!=5875):raise ValueError('transfer cache incomplete/identity mismatch')
    return m


def validate_rows(rows,expected_records,tasks):
    if set(rows)!=set(tasks) or sum(len(x) for x in rows.values())!=expected_records:raise ValueError('cache row mapping coverage mismatch')
    mapping={}
    for task,entries in rows.items():
        mapping[task]={}
        for e in entries:
            row=int(e['row'])
            if row in mapping[task] or e['cache_key']!=f'{task}:{row}' or not isinstance(e['key'],str) or not e['key']:
                raise ValueError('invalid/duplicate original row mapping')
            mapping[task][row]=e
    return mapping


class InputCache:
    def __init__(self,root):
        self.root=Path(root);self.metadata=cache_metadata(root);self.rows=read(self.root/'rows.json')
        if set(self.rows)!=set(TASKS) or sum(len(x) for x in self.rows.values())!=5875:raise ValueError('cache row mapping coverage mismatch')
        self.mapping=validate_rows(self.rows,5875,TASKS)
        self.env=lmdb.open(str(self.root/'graphs.lmdb'),readonly=True,lock=False,readahead=True,max_readers=128)
        if self.env.stat()['entries']!=5875:raise ValueError('physical graph cache coverage mismatch')
        self.memory={}
    def get(self,task,row,key):
        e=self.mapping[task].get(int(row))
        if not e or e['key']!=key or e['cache_key']!=f'{task}:{row}':raise ValueError('cache sample mapping mismatch')
        cache_key=e['cache_key']
        if cache_key not in self.memory:
            with self.env.begin() as txn:raw=txn.get(cache_key.encode())
            if raw is None:raise KeyError('missing cache record; fallback prohibited')
            record=decode_template(raw)
            if record.get('schema')!=CACHE_SCHEMA or record.get('sample_key')!=key or record.get('row')!=int(row) or record.get('task')!=task:raise ValueError('cache record identity mismatch')
            d=record['clean']
            if d.lga_path_index.shape[1]!=3 or d.bond_path_features.shape!=(len(d.lga_spd),2,14) or (d.lga_spd>2).any():raise ValueError('cache is not original 2-hop')
            self.memory[cache_key]=d
        return self.memory[cache_key].clone()
    def close(self):self.env.close()


class Data(torch.utils.data.Dataset):
    def __init__(self,task,fold,formal=False):
        from src.training.glt_dual_runtime import open_source
        from scripts.finetune_mcl_ph import compact_train_validation_indices
        manifest=read(SPLIT/f'{task}.json');entry=manifest['folds'][fold]
        if manifest.get('validation_is_test') is not False:raise ValueError('validation/TEST separation invalid')
        if formal:
            self.indices=sorted(entry['test_indices']);self.train=[];self.validation=list(range(len(self.indices)))
        else:self.indices,self.train,self.validation=compact_train_validation_indices(entry['train_indices'],entry['validation_indices'])
        source,frame=open_source(DOWN,DOWN,task=task,selected_indices=self.indices,split_manifest=manifest,verify_key_digest=False)
        try:
            if frame.original_row.astype(int).tolist()!=self.indices:raise ValueError('selected source row mismatch')
            self.raw_targets=finite(frame.label.to_numpy());self.targets=self.raw_targets.copy();self.keys=np.asarray([s[0].hex() for s in source.samples])
        finally:source.close()
        self.task=task;self.cache=InputCache(OUT/'inputs')
        for row,key in zip(self.indices,self.keys):
            mapped=self.cache.mapping[task].get(row)
            if not mapped or mapped['key']!=key:raise ValueError('selected cache/key/source mismatch')
    def __len__(self):return len(self.indices)
    def __getitem__(self,i):
        d=self.cache.get(self.task,self.indices[i],self.keys[i]);d.y=torch.tensor([self.targets[i]],dtype=torch.float32);return d
    def set_target_override(self,y):self.targets=np.asarray(y,dtype=np.float64)
    def close(self):self.cache.close()


def loader(ds,ids,shuffle=False,seed=42):
    return torch.utils.data.DataLoader(torch.utils.data.Subset(ds,ids),batch_size=24,shuffle=shuffle,num_workers=0,
        drop_last=False,collate_fn=collate,generator=torch.Generator().manual_seed(seed) if shuffle else None)
def metrics(y,p):
    y,p=finite(y).reshape(-1),finite(p).reshape(-1)
    return dict(r2=float(r2_score(y,p)),rmse=float(mean_squared_error(y,p)**.5),mae=float(mean_absolute_error(y,p)))


def frozen_best(selected,arm,task,fold):
    expected=unit_path(arm,task,fold)/'best.pt'
    if selected['config']!=config(arm,task,fold) or Path(selected['best'])!=expected:raise ValueError('frozen transfer best/config mismatch')
    return expected


def begin_epoch_reservation(state, updates):
    if state.get('epoch_reservation',{}).get('open'):raise ValueError('previous epoch reservation still open')
    state['epochs_started']+=1
    state['epoch_reservation']=dict(open=True,epoch=state['epochs_started'],updates=updates,validations=1,
        before_updates=state['attempted_updates'],before_validation=state['validation_started'])


def close_epoch_reservation(state):
    r=state['epoch_reservation']
    if (not r['open'] or state['attempted_updates']!=r['before_updates']+r['updates']
        or state['completed_updates']!=state['attempted_updates']
        or state['validation_started']!=r['before_validation']+1
        or state['completed_epochs']!=r['epoch']):raise ValueError('incomplete epoch cannot close reservation')
    r['open']=False


def conservative_ft_counts(state):
    actual={k:int(state.get(k,0)) for k in ('epochs_started','validation_started','attempted_updates','test_graphs')}
    charged=dict(actual);r=state.get('epoch_reservation',{})
    if r.get('open'):
        charged['attempted_updates']=max(actual['attempted_updates'],r['before_updates']+r['updates'])
        charged['validation_started']=max(actual['validation_started'],r['before_validation']+r['validations'])
    return dict(known_actual_lower_bound=actual,charged=charged,
        uncertain={k:charged[k]-actual[k] for k in charged},open_epoch=bool(r.get('open')))



def unit(arm, task, fold, formal=False):
    from src.utils import scale_targets, TargetScaler
    from sklearn.preprocessing import StandardScaler
    require_tmux()
    if os.environ.get('CUDA_VISIBLE_DEVICES') not in ('2', '3'):
        raise ValueError('physical GPU2/3 workers only')
    cache_metadata(OUT/'inputs')
    folder = unit_path(arm, task, fold, formal); folder.mkdir(parents=True, exist_ok=False)
    cfg = config(arm, task, fold); write(folder/'config.json', cfg)
    state = dict(status='RUNNING', pid=os.getpid(), started=now(), completed_epochs=0,
        epochs_started=0, validation_started=0, validation_completed=0, attempted_updates=0,
        completed_updates=0, test_started=0, test_graphs=0, phase='initialize')
    io_seconds=0.;io_calls=0
    def timed_write(path,value,checkpoint=False):
        nonlocal io_seconds,io_calls
        tick=time.monotonic()
        if checkpoint:save_checkpoint(path,value)
        else:write(path,value)
        io_seconds+=time.monotonic()-tick;io_calls+=1
    def record():
        state['engineering_revision']='ua2d-transfer-epoch-reservation-r1'
        timed_write(folder/'consumption.json',state)
    record()
    unit_started=time.monotonic()
    ds = None
    try:
        seed_all(42+fold)
        if formal:
            frozen = read(OUT/'selection/final_frozen.json')
            if frozen.get('frozen') is not True: raise ValueError('TEST before freeze')
            if frozen['splits'][task]!=read(SPLIT/f'{task}.json'):raise ValueError('frozen original split changed')
            selected = frozen['units'][f'{arm}/{task}/{fold}']
            saved = torch.load(frozen_best(selected,arm,task,fold), map_location='cpu', weights_only=False)
            if (selected['config'] != cfg or selected['best_epoch'] != saved['best_epoch']
                    or selected['best_stage'] != saved['best_stage'] or selected['scaler'] != saved['scaler']):
                raise ValueError('frozen complete best/scaler mismatch')
            model = restore(saved, cfg); scaler_json = saved['scaler']
            scaler = TargetScaler(task, StandardScaler(), transform_mode='standard')
            scaler.scaler.mean_ = np.array(scaler_json['mean']); scaler.scaler.scale_ = np.array(scaler_json['scale'])
            scaler.scaler.var_ = scaler.scaler.scale_**2; scaler.scaler.n_features_in_ = 1
            best_epoch, best_stage = saved['best_epoch'], saved['best_stage']
            del saved
        else:
            parent = torch.load(PARENT, map_location='cpu', weights_only=False)
            head = torch.load(cfg['head_initial'], map_location='cpu', weights_only=False)
            residual=torch.load(OUT/'initialization/residual.pt',map_location='cpu',weights_only=False)
            if residual.get('plan')!=PLAN or residual.get('seed')!=BRANCH_SEED or residual.get('schema')!=SCHEMA:raise ValueError('common residual initialization mismatch')
            model = initialize(parent, head, residual['state_dict'], arm); del parent,head,residual
        ds = Data(task, fold, formal)
        if formal:
            if selected['test_indices'] != ds.indices: raise ValueError('frozen TEST indices mismatch')
            ds.set_target_override(scaler.transform(ds.raw_targets).reshape(-1))
        else:
            scaler = scale_targets(ds, task, train_indices=ds.train, transform_mode='standard')
            scaler_json = dict(mean=scaler.scaler.mean_.tolist(), scale=scaler.scaler.scale_.tolist(),
                               fit='train', transform='standard')
        reference_scaler = read(PARENT.parents[2]/'finetune/B_FP'/task/f'fold{fold}/scaler.json')
        if scaler_json != reference_scaler: raise ValueError('B_FP train-only scaler mismatch')
        write(folder/'scaler.json', scaler_json)
        model.to('cuda:0')
        if not formal:model.cache_anchor()
        validation = loader(ds, ds.validation)
        def evaluate():
            model.eval(); targets, predictions = [], []
            with torch.inference_mode():
                for batch in validation:
                    batch = batch.to('cuda:0'); p = model(batch)
                    if not torch.isfinite(p).all(): raise FloatingPointError('nonfinite validation/TEST prediction')
                    targets.append(batch.y.cpu().numpy()); predictions.append(p.cpu().numpy())
                    if formal:
                        state['test_graphs'] += len(batch.y); record()
            y = scaler.inverse_transform(np.concatenate(targets)).reshape(-1)
            p = scaler.inverse_transform(np.concatenate(predictions)).reshape(-1)
            return y, p, metrics(y, p)
        if formal:
            state.update(phase='TEST', test_started=1); record()
            truth, prediction, result = evaluate()
        else:
            train = loader(ds, ds.train, True, 42+fold)
            expected = 70*len(train)
            if expected > 56000: raise RuntimeError('unit update reservation impossible')
            seed_all(42+fold); history = []; epoch_timings=[]; best_r2 = -math.inf
            for stage, epochs in (('head', 10), ('joint', 60)):
                opt, evidence = optimizer(model, stage)
                scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10, eta_min=3e-5)
                    if stage == 'head' else torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=10, T_mult=1, eta_min=1e-6))
                for stage_epoch in range(epochs):
                    epoch_tick=time.monotonic();io_before=io_seconds
                    state['phase']=stage;begin_epoch_reservation(state,len(train));record()
                    training_rates = [g['lr'] for g in opt.param_groups]
                    model.train(); losses = [];mse_losses=[];anchor_losses=[]
                    for batch in train:
                        batch = batch.to('cuda:0'); opt.zero_grad(set_to_none=True)
                        output = model(batch); mse = nn.functional.mse_loss(output, batch.y.view(-1));anchor=model.anchor_loss();loss=mse+anchor
                        if not torch.isfinite(loss): raise FloatingPointError('nonfinite loss')
                        loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
                        if state['attempted_updates'] >= expected: raise RuntimeError('unit update limit reached')
                        state['attempted_updates'] += 1
                        opt.step(); state['completed_updates'] += 1
                        losses.append(float(loss.detach()));mse_losses.append(float(mse.detach()));anchor_losses.append(float(anchor.detach()))
                    scheduler.step()
                    state['validation_started'] += 1
                    y, p, met = evaluate()
                    state['validation_completed'] += 1; state['completed_epochs'] += 1
                    rec = dict(epoch=state['completed_epochs'], stage=stage, stage_epoch=stage_epoch+1,
                        train_loss=float(np.mean(losses)),mse_loss=float(np.mean(mse_losses)),anchor_loss=float(np.mean(anchor_losses)),total_loss=float(np.mean(losses)), validation_r2=met['r2'], validation_rmse=met['rmse'],
                        learning_rates=training_rates, optimizer_groups=evidence,elapsed_seconds=time.monotonic()-unit_started,gpu_peak_bytes=torch.cuda.max_memory_allocated(),relations_last_batch=int(batch.lga_spd.numel()))

                    if met['r2'] > best_r2:
                        best_r2 = met['r2']; best_epoch = state['completed_epochs']; best_stage = stage
                        truth, prediction, result = y.copy(), p.copy(), met.copy()
                        timed_write(folder/'best.pt', dict(schema=SCHEMA, architecture=ARCHITECTURE,
                            config=cfg, state_dict={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                            best_epoch=best_epoch, best_stage=best_stage, scaler=scaler_json, outer_test='NOT_RUN'),checkpoint=True)
                    rec['epoch_seconds_before_history']=time.monotonic()-epoch_tick
                    rec['write_seconds_before_history']=io_seconds-io_before
                    history.append(rec);timed_write(folder/'history.json',history)
                    close_epoch_reservation(state);record()
                    epoch_timings.append(dict(epoch=state['completed_epochs'],epoch_seconds=time.monotonic()-epoch_tick,
                        write_seconds=io_seconds-io_before,write_seconds_cumulative=io_seconds,write_calls_cumulative=io_calls,
                        timing_scope='real epoch incl CPU/model/evaluation/write; no added CUDA synchronization'))
                    timed_write(folder/'epoch_timing.json',epoch_timings);print(json.dumps(rec),flush=True)
            if state['attempted_updates'] != expected or state['completed_updates'] != expected:
                raise RuntimeError('incomplete optimizer budget')
            timed_write(folder/'resume.pt',dict(schema=SCHEMA,architecture=ARCHITECTURE,config=cfg,
                model=model.state_dict(),optimizer=opt.state_dict(),scheduler=scheduler.state_dict(),
                stage=stage,stage_epoch=stage_epoch+1,epoch=state['completed_epochs'],batch_cursor='epoch_end',
                best_epoch=best_epoch,best_stage=best_stage,best_r2=best_r2,scaler=scaler_json,
                torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),
                numpy_rng=np.random.get_state(),python_rng=random.getstate(),loader_rng=train.generator.get_state(),
                consumption=copy.deepcopy(state),automatic_resume=False),checkpoint=True)
        np.savez(folder/('test_predictions.npz' if formal else 'validation_predictions.npz'),
            y_true=finite(truth), y_pred=finite(prediction), indices=np.array([ds.indices[i] for i in ds.validation]),
            sample_keys=ds.keys[ds.validation], best_epoch=best_epoch, best_stage=best_stage)
        write(folder/'metrics.json', dict(status='COMPLETE', config=cfg, best_epoch=best_epoch,
            best_stage=best_stage, scaler=scaler_json, **result))
        state.update(status='COMPLETE', ended=now(), best_epoch=best_epoch, best_stage=best_stage,elapsed_seconds=time.monotonic()-unit_started,gpu_peak_bytes=torch.cuda.max_memory_allocated(),write_seconds=io_seconds,write_calls=io_calls); record()
    except BaseException:
        state.update(status='FAILED', error=traceback.format_exc(), ended=now()); record()
        write(folder/'failure.json', dict(error=state['error'], automatic_retry=False, time=now()))
        raise
    finally:
        if ds is not None: ds.close()




def aggregate(arm,formal=False):
    tasks={};units={}
    for task in TASKS:
        folds=[]
        for fold in range(5):
            path=unit_path(arm,task,fold,formal);m=read(path/'metrics.json');c=read(path/'consumption.json')
            expected=70*math.ceil(len(read(SPLIT/f'{task}.json')['folds'][fold]['train_indices'])/24)
            if m['status']!='COMPLETE' or c['status']!='COMPLETE' or m['config']!=config(arm,task,fold):raise ValueError('incomplete or inconsistent transfer matrix')
            if not formal and (c['completed_epochs']!=70 or c['completed_updates']!=expected or c['attempted_updates']!=expected or c['epoch_reservation']['open']):raise ValueError('incomplete full FT budget')
            if formal and (c['test_started']!=1 or c['test_graphs']!=len(read(SPLIT/f'{task}.json')['folds'][fold]['test_indices'])):raise ValueError('incomplete formal TEST graphs')
            folds.append(m['r2']);units[f'{arm}/{task}/{fold}']=dict(config=m['config'],best=str(unit_path(arm,task,fold)/'best.pt'),
                best_epoch=m['best_epoch'],best_stage=m['best_stage'],scaler=m['scaler'],scaler_path=str(unit_path(arm,task,fold)/'scaler.json'),
                config_path=str(unit_path(arm,task,fold)/'config.json'),test_indices=sorted(read(SPLIT/f'{task}.json')['folds'][fold]['test_indices']),
                predictions=str(path/('test_predictions.npz' if formal else 'validation_predictions.npz')))
        tasks[task]=dict(folds=folds,mean=float(finite(folds).mean()),std=float(np.std(folds,ddof=1)))
    return dict(tasks=tasks,macro=float(np.mean([v['mean'] for v in tasks.values()])),units=units)


def nominate(validation):
    if set(validation)!=set(ARMS):raise ValueError('all four validation arms must be complete')
    scores={a:float(validation[a]['macro']) for a in ARMS};finite(list(scores.values()))
    return max(ARMS,key=lambda a:(scores[a],-ARMS.index(a)))


def freeze_final(validation):
    final=nominate(validation);units={k:v for a in ARMS for k,v in validation[a]['units'].items()}
    expected={f'{a}/{t}/{f}' for a in ARMS for t in TASKS for f in range(5)}
    if set(units)!=expected:raise ValueError('complete 140 logical units required before freeze')
    if (OUT/'selection/final_frozen.json').exists():raise FileExistsError('freeze is one-time only')
    frozen=dict(plan=PLAN,revision=REVISION,time=now(),frozen=True,FINAL=final,models=list(ARMS),units=units,
        validation_scores={a:validation[a]['macro'] for a in ARMS},splits={t:read(SPLIT/f'{t}.json') for t in TASKS},
        selection='validation_only; all_four_TEST_no_gate; no_TEST_reselection')
    path=OUT/'selection/final_frozen.json';path.parent.mkdir(exist_ok=True)
    with path.open('x') as f:json.dump(frozen,f,ensure_ascii=False,indent=2,allow_nan=False)
    return frozen


def outcome(delta):
    if delta is None:return dict(positive_test_delta=None,performance_pass=None,performance_status='未完成／不可比较')
    finite(delta)
    return dict(positive_test_delta=delta>0,performance_pass=delta>=.010,performance_status='通过' if delta>=.010 else '本次正增量但未达门槛' if delta>0 else '本次未观察到提升')


def registered_comparisons(formal,baseline,project,final):
    values=dict(formal,B_FP=baseline,A_MIX246=project)
    specs={**{a+'_minus_B_FP':{a:1,'B_FP':-1} for a in ARMS},
        'R_STD_minus_R_MEAN':{'R_STD':1,'R_MEAN':-1},
        'R_STD_SP_minus_R_STD':{'R_STD_SP':1,'R_STD':-1},
        'R_STD_SP_minus_A_SP':{'R_STD_SP':1,'A_SP':-1},
        'interaction':{'R_STD_SP':1,'R_STD':-1,'A_SP':-1,'B_FP':1},
        'FINAL_minus_A_MIX246':{final:1,'A_MIX246':-1} if final else {}}
    result={}
    for name,coef in specs.items():
        if not coef or any(k not in values for k in coef):result[name]=dict(delta=None,coefficients=coef,status='未完成／不可比较');continue
        paired={}
        for task in TASKS:
            x=sum(c*finite(values[a]['tasks'][task]['folds']) for a,c in coef.items())
            paired[task]=dict(folds=x.tolist(),mean=float(x.mean()),std=float(x.std(ddof=1)))
        delta=sum(c*values[a]['macro'] for a,c in coef.items())
        result[name]=dict(delta=float(delta),coefficients=coef,paired_tasks=paired,status='COMPLETE')
    return result
