"""Isolated UA2D P10K no-FP ALIGN/3D ablation campaign; all paths are explicit."""
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
from src.modules.glt_dual import Pure2DGLTModel,strict_module_state
from src.modules.ua2d_p10k_nofp_align import arm_config,package_attention_nofp,parse_model_stream_state,LEGACY_STREAM_POLICY
from src.training.glt_dual_runtime import require_tmux,save_checkpoint
ROOT=Path(__file__).resolve().parents[2]
PLAN='UA2D-P10K-NOFP-ALIGN-20261024-01'
REVISION='r1.1-GPU23'
OUT=ROOT/'results/ua2d_p10k_nofp_align_20261024/r1'
LOG=ROOT/'logs/ua2d_p10k_nofp_align_20261024/r1'
DOWN=ROOT/'data/processed/mcl_ph_r4_20260930/downstream_direct_01.staging'
SPLIT=ROOT/'data/splits/mips_paper5_outer'
PARENT=ROOT/'results/ua2d_pretrain_fp_20261019/fast_r4/pretrain/B_FP/deploy.pt'
HEADS=ROOT/'results/glt_optimization_20261011/r4/U1/initialization/heads_2d'
REFERENCE_BFP_SCALERS=ROOT/'results/ua2d_pretrain_fp_20261019/fast_r4/finetune/B_FP'
REFERENCE_P10K=ROOT/'results/ua2d_ptscale_20261022/r1/final_review/results.json'
CACHE_ROOT=ROOT/'results/ua2d_transfer_20261022/r1/inputs'
CACHE_PLAN='UA2D-TRANSFER-20261022-01'
PT_CACHE=ROOT/'results/ua2d_pretrain_fp_20261019/fast_r4/prepared_cache'
MAP_ROOT=ROOT/'results/ua2d_ptscale_20261022/r1/inputs'
INITIALIZATION=ROOT/'results/ua2d_pretrain_fp_20261019/fast_r4/initialization/common_step0.pt'
PT_SCHEMA='ua2d-p10k-nofp-align-pretrain-r1'
SCHEMA='ua2d-p10k-nofp-align-best-r1'
ARMS=('D_A000','D_A005','D_A010','D_A020','PURE2D')
NOMINATION_ORDER=('PURE2D','D_A000','D_A005','D_A010','D_A020')
STEPS={arm:10000 for arm in ARMS}
from src.training.ua2d_ptscale import ARCHITECTURE as REFERENCE_ARCHITECTURE,PROFILE as REFERENCE_PROFILE
ARCHITECTURE=copy.deepcopy(REFERENCE_ARCHITECTURE)
PROFILE=copy.deepcopy(REFERENCE_PROFILE)
def parent_path(arm):return OUT/'pretrain'/arm/'deploy.pt'

TASKS=('eea','egb','egc','ei','eps','nc','xc')
CACHE_SCHEMA='ua2d-transfer-input-2hop-r1'
LIMITS=dict(pt_starts=5,pt_updates=50000,pt_presentations=50400000,
    ft_starts=175,epochs_started=12250,validation_started=12250,validation_graphs=1649900,
    attempted_updates=280000,test_starts=175,test_graphs=29375,cpu_suites=1)
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
def identity_config(cfg):return {k:v for k,v in cfg.items() if k!='physical_gpu'}
def config(arm,task,fold,gpu):
    if arm not in ARMS or task not in TASKS or fold not in range(5) or gpu not in (2,3):raise ValueError('outside no-FP ALIGN matrix')
    return dict(plan=PLAN,revision=REVISION,arm=arm,task=task,fold=fold,seed=42+fold,parent=str(parent_path(arm)),
        parent_step=STEPS[arm],parent_plan=PLAN,parent_schema=PT_SCHEMA,parent_arm=arm,parent_world_size=1,
        head_initial=str(HEADS/task/f'fold{fold}.pt'),source=str(DOWN),split=str(SPLIT/f'{task}.json'),
        cache=str(CACHE_ROOT),cache_plan=CACHE_PLAN,cache_schema=CACHE_SCHEMA,
        architecture=ARCHITECTURE,profile=PROFILE,input_semantics='original_2hop_2d',physical_gpu=gpu)


def update_plan(status,phase,detail):
    p=ROOT/'CURRENT_PLAN.md';text=p.read_text();start='\n### UA2D-P10K-NOFP-ALIGN r1.1-GPU23 最新执行状态\n';end='\n<!-- UA2D_P10K_NOFP_ALIGN_EXECUTION_END -->'
    entry=start+f'\n- {now()}；状态：{status}；阶段：{phase}。\n- {detail}\n- 交付：{OUT}；日志：{LOG}；DSH执行自检不构成Codex独立审查。\n'+end
    if start in text:
        before,rest=text.split(start,1);_,after=rest.split(end,1);text=before+entry+after
    else:text=text.rstrip()+'\n'+entry+'\n'
    p.write_text(text);write(OUT/'status.json',dict(plan=PLAN,revision=REVISION,status=status,phase=phase,time=now(),detail=detail,independent_review='PENDING'))


def cache_metadata(root):
    root=Path(root);m=read(root/'metadata.json');coverage=read(root/'coverage.json')
    if (m.get('plan')!=CACHE_PLAN or m.get('schema')!=CACHE_SCHEMA or m.get('status')!='COMPLETE'
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
        self.task=task;self.cache=InputCache(CACHE_ROOT)
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
    if identity_config(selected['config'])!=identity_config(config(arm,task,fold,selected['config'].get('physical_gpu',2))) or Path(selected['best'])!=expected:raise ValueError('frozen transfer best/config mismatch')
    return expected


def begin_epoch_reservation(state, updates, validation_graphs):
    if state.get('epoch_reservation',{}).get('open'):raise ValueError('previous epoch reservation still open')
    state['epochs_started']+=1
    state['epoch_reservation']=dict(open=True,epoch=state['epochs_started'],updates=updates,validations=1,
        validation_graphs=validation_graphs,
        before_updates=state['attempted_updates'],before_validation=state['validation_started'],
        before_validation_graphs=state['validation_graphs'])


def close_epoch_reservation(state):
    r=state['epoch_reservation']
    if (not r['open'] or state['attempted_updates']!=r['before_updates']+r['updates']
        or state['completed_updates']!=state['attempted_updates']
        or state['validation_started']!=r['before_validation']+1
        or state['validation_graphs']!=r['before_validation_graphs']+r['validation_graphs']
        or state['completed_epochs']!=r['epoch']):raise ValueError('incomplete epoch cannot close reservation')
    r['open']=False


def conservative_ft_counts(state):
    actual={k:int(state.get(k,0)) for k in ('epochs_started','validation_started','validation_graphs','attempted_updates','test_graphs')}
    charged=dict(actual);r=state.get('epoch_reservation',{})
    if r.get('open'):
        charged['attempted_updates']=max(actual['attempted_updates'],r['before_updates']+r['updates'])
        charged['validation_started']=max(actual['validation_started'],r['before_validation']+r['validations'])
        charged['validation_graphs']=max(actual['validation_graphs'],r['before_validation_graphs']+r['validation_graphs'])
    return dict(known_actual_lower_bound=actual,charged=charged,
        uncertain={k:charged[k]-actual[k] for k in charged},open_epoch=bool(r.get('open')))



class MeanModel(Pure2DGLTModel):
    def __init__(self):
        super().__init__(dropout=0.,attention_scale_denominator=512);self.head[2].p=.1;self.stage='head'
    def train(self,mode=True):
        super().train(mode)
        if mode and self.stage=='head':self.o8.eval();self.norm2.eval()
        return self
    def forward(self,data):return super().forward(data).view(-1)


def validate_parent(parent,expected_arm):
    from dataclasses import asdict
    if expected_arm not in ARMS:raise ValueError('unregistered arm')
    try:parsed=arm_config(expected_arm)
    except ValueError as error:raise ValueError('unregistered arm: '+str(error))
    if (parent.get('schema')!=PT_SCHEMA or parent.get('step')!=STEPS[expected_arm]
        or parent.get('plan')!=PLAN or parent.get('revision')!=REVISION or parent.get('arm')!=expected_arm
        or parent.get('config')!=asdict(parsed) or 'fp_enabled' in parent.get('config',{})
        or parent.get('execution',{}).get('plan')!=PLAN
        or parent['execution'].get('world_size')!=1 or parent['execution'].get('microbatch')!=168
        or parent['execution'].get('accumulation')!=6 or parent['execution'].get('global_batch')!=1008
        or parent.get('eligible_rows')!=910756
        or package_attention_nofp(parent)['scale_denominator']!=512):raise ValueError('no-FP parent branch/schema/step mismatch')
    policy,_=parse_model_stream_state(parent.get('stream_positions',{}),parsed.branch3d)
    if policy!=LEGACY_STREAM_POLICY:raise ValueError('original r1 parent requires legacy stream policy')
    return parent


def initialize(parent,head,arm):
    validate_parent(parent,arm);model=MeanModel()
    for name in ('o8','norm2'):
        prefix=name+'.';strict_module_state(getattr(model,name),{k[len(prefix):]:v for k,v in parent['state_dict'].items() if k.startswith(prefix)})
    strict_module_state(model.head,head);return model


def restore(package,cfg):
    if (package.get('schema')!=SCHEMA or identity_config(package.get('config',{}))!=identity_config(cfg)
        or package.get('architecture')!=ARCHITECTURE
        or cfg.get('parent_step')!=STEPS[cfg['arm']] or cfg.get('parent')!=str(parent_path(cfg['arm']))):raise ValueError('FT checkpoint/source mismatch')
    model=MeanModel();strict_module_state(model,package['state_dict']);return model


def optimizer(model,stage):
    if stage not in ('head','joint'):raise ValueError('invalid stage')
    model.stage=stage
    for n,p in model.named_parameters():p.requires_grad_(stage=='joint' or n.startswith('head.'))
    groups=[]
    if stage=='joint':groups.append(dict(name='backbone',params=[p for n,p in model.named_parameters() if p.requires_grad and not n.startswith('head.')],lr=.0001))
    groups.append(dict(name='head',params=list(model.head.parameters()),lr=.0003 if stage=='head' else .001))
    ids=[id(p) for g in groups for p in g['params']]
    if len(ids)!=len(set(ids)) or set(ids)!={id(p) for p in model.parameters() if p.requires_grad}:raise ValueError('optimizer parameter coverage mismatch')
    evidence=[dict(name=g['name'],lr=g['lr'],tensors=len(g['params']),parameters=sum(p.numel() for p in g['params'])) for g in groups]
    return torch.optim.AdamW(groups,weight_decay=.02),evidence


def unit(arm, task, fold, formal=False):
    from src.utils import scale_targets, TargetScaler
    from sklearn.preprocessing import StandardScaler
    require_tmux()
    gpu=os.environ.get('CUDA_VISIBLE_DEVICES')
    if gpu not in ('2','3'):
        raise ValueError('only physical GPU2/GPU3 workers')
    gpu=int(gpu)
    cache_metadata(CACHE_ROOT)
    folder = unit_path(arm, task, fold, formal); folder.mkdir(parents=True, exist_ok=False)
    cfg = config(arm, task, fold, gpu); write(folder/'config.json', cfg)
    state = dict(status='RUNNING', pid=os.getpid(), process=process_identity(os.getpid()), started=now(), completed_epochs=0,
        epochs_started=0, validation_started=0, validation_completed=0, validation_graphs=0, attempted_updates=0,
        completed_updates=0, test_started=0, test_graphs=0, physical_gpu=gpu, phase='initialize')
    io_seconds=0.;io_calls=0
    def timed_write(path,value,checkpoint=False):
        nonlocal io_seconds,io_calls
        tick=time.monotonic()
        if checkpoint:save_checkpoint(path,value)
        else:write(path,value)
        io_seconds+=time.monotonic()-tick;io_calls+=1
    def record():
        state['engineering_revision']='ua2d-p10k-nofp-align-epoch-reservation-r1'
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
            if (identity_config(selected['config']) != identity_config(cfg) or selected['best_epoch'] != saved['best_epoch']
                    or selected['best_stage'] != saved['best_stage'] or selected['scaler'] != saved['scaler']):
                raise ValueError('frozen complete best/scaler mismatch')
            model = restore(saved, cfg); scaler_json = saved['scaler']
            scaler = TargetScaler(task, StandardScaler(), transform_mode='standard')
            scaler.scaler.mean_ = np.array(scaler_json['mean']); scaler.scaler.scale_ = np.array(scaler_json['scale'])
            scaler.scaler.var_ = scaler.scaler.scale_**2; scaler.scaler.n_features_in_ = 1
            best_epoch, best_stage = saved['best_epoch'], saved['best_stage']
            del saved
        else:
            parent = torch.load(parent_path(arm), map_location='cpu', weights_only=False)
            head = torch.load(cfg['head_initial'], map_location='cpu', weights_only=False)
            model = initialize(parent, head, arm); del parent,head
        ds = Data(task, fold, formal)
        if formal:
            if selected['test_indices'] != ds.indices: raise ValueError('frozen TEST indices mismatch')
            ds.set_target_override(scaler.transform(ds.raw_targets).reshape(-1))
        else:
            scaler = scale_targets(ds, task, train_indices=ds.train, transform_mode='standard')
            scaler_json = dict(mean=scaler.scaler.mean_.tolist(), scale=scaler.scaler.scale_.tolist(),
                               fit='train', transform='standard')
        reference_scaler = read(REFERENCE_BFP_SCALERS/task/f'fold{fold}/scaler.json')
        if scaler_json != reference_scaler: raise ValueError('B_FP train-only scaler mismatch')
        write(folder/'scaler.json', scaler_json)
        model.to('cuda:0')
        validation = loader(ds, ds.validation)
        def evaluate():
            model.eval(); targets, predictions = [], []
            with torch.inference_mode():
                for batch in validation:
                    batch = batch.to('cuda:0'); p = model(batch)
                    if not torch.isfinite(p).all(): raise FloatingPointError('nonfinite validation/TEST prediction')
                    targets.append(batch.y.cpu().numpy()); predictions.append(p.cpu().numpy())
                    if formal: state['test_graphs'] += len(batch.y)
                    else: state['validation_graphs'] += len(batch.y)
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
                    state['phase']=stage;begin_epoch_reservation(state,len(train),len(ds.validation));record()
                    training_rates = [g['lr'] for g in opt.param_groups]
                    model.train(); losses = []
                    for batch in train:
                        batch = batch.to('cuda:0'); opt.zero_grad(set_to_none=True)
                        output = model(batch); mse = nn.functional.mse_loss(output, batch.y.view(-1));loss=mse
                        if not torch.isfinite(loss): raise FloatingPointError('nonfinite loss')
                        loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
                        if state['attempted_updates'] >= expected: raise RuntimeError('unit update limit reached')
                        state['attempted_updates'] += 1
                        opt.step(); state['completed_updates'] += 1
                        losses.append(float(loss.detach()))
                    scheduler.step()
                    state['validation_started'] += 1
                    y, p, met = evaluate()
                    state['validation_completed'] += 1; state['completed_epochs'] += 1
                    rec = dict(epoch=state['completed_epochs'], stage=stage, stage_epoch=stage_epoch+1,
                        train_loss=float(np.mean(losses)), validation_r2=met['r2'], validation_rmse=met['rmse'],
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
            if m['status']!='COMPLETE' or c['status']!='COMPLETE' or identity_config(m['config'])!=identity_config(config(arm,task,fold,c['physical_gpu'])):raise ValueError('incomplete or inconsistent no-FP matrix')
            if not formal and (c['completed_epochs']!=70 or c['completed_updates']!=expected or c['attempted_updates']!=expected
                or c['validation_graphs']!=70*len(read(SPLIT/f'{task}.json')['folds'][fold]['validation_indices'])
                or c['epoch_reservation']['open']):raise ValueError('incomplete full FT budget')
            if formal and (c['test_started']!=1 or c['test_graphs']!=len(read(SPLIT/f'{task}.json')['folds'][fold]['test_indices'])):raise ValueError('incomplete formal TEST graphs')
            folds.append(m['r2']);units[f'{arm}/{task}/{fold}']=dict(config=m['config'],best=str(unit_path(arm,task,fold)/'best.pt'),
                best_epoch=m['best_epoch'],best_stage=m['best_stage'],scaler=m['scaler'],scaler_path=str(unit_path(arm,task,fold)/'scaler.json'),
                config_path=str(unit_path(arm,task,fold)/'config.json'),test_indices=sorted(read(SPLIT/f'{task}.json')['folds'][fold]['test_indices']),
                predictions=str(path/('test_predictions.npz' if formal else 'validation_predictions.npz')))
        tasks[task]=dict(folds=folds,mean=float(finite(folds).mean()),std=float(np.std(folds,ddof=1)))
    return dict(tasks=tasks,macro=float(np.mean([v['mean'] for v in tasks.values()])),units=units)


def nominate(validation):
    if set(validation)!=set(ARMS):raise ValueError('all 175 FT required')
    finite([validation[a]['macro'] for a in ARMS])
    return max(NOMINATION_ORDER,key=lambda a:(validation[a]['macro'],-NOMINATION_ORDER.index(a)))


def freeze_final(validation):
    final=nominate(validation);units={k:v for a in ARMS for k,v in validation[a]['units'].items()}
    expected={f'{a}/{t}/{f}' for a in ARMS for t in TASKS for f in range(5)}
    if set(units)!=expected:raise ValueError('complete 175 logical units required before freeze')
    if (OUT/'selection/final_frozen.json').exists():raise FileExistsError('freeze is one-time only')
    frozen=dict(plan=PLAN,revision=REVISION,time=now(),frozen=True,FINAL=final,models=list(ARMS),units=units,
        validation_scores={a:validation[a]['macro'] for a in ARMS},splits={t:read(SPLIT/f'{t}.json') for t in TASKS},
        selection='validation_only macro; tie PURE2D first then lambda ascending; all_five_TEST_no_gate; no_TEST_reselection')
    path=OUT/'selection/final_frozen.json';path.parent.mkdir(exist_ok=True)
    with path.open('x') as f:json.dump(frozen,f,ensure_ascii=False,indent=2,allow_nan=False)
    return frozen


def outcome(final,final_delta,d3_delta):
    if final_delta is None or d3_delta is None:return dict(positive_test_delta=None,performance_pass=None,performance_status='未完成／不可比较')
    finite([final_delta,d3_delta])
    align_gain=final_delta>=.010;three_d_gain=d3_delta>=.010
    note=('FINAL为PURE2D：该差值来自移除3D分支，不作为正ALIGN权重优化的收益' if final=='PURE2D'
          else 'ALIGN权重相对本轮无FP参考D_A010的收益')
    return dict(positive_test_delta=final_delta>0,align_gain_pass=align_gain,align_gain_note=note,
        three_d_gain_pass=three_d_gain,performance_status='ALIGN调权通过' if align_gain else ('3D包通过' if three_d_gain else '未达到0.010性能门槛'))


def registered_comparisons(formal,final,p10k):
    values=dict(formal,P10K_REF=p10k)
    specs={}
    for arm in ('D_A000','D_A005','D_A020'):specs[f'{arm}_minus_D_A010']={arm:1,'D_A010':-1}
    for arm in ('D_A005','D_A010','D_A020'):specs[f'{arm}_minus_D_A000']={arm:1,'D_A000':-1}
    specs['D_A010_minus_PURE2D']={'D_A010':1,'PURE2D':-1}
    specs['D_A000_minus_PURE2D']={'D_A000':1,'PURE2D':-1}
    if final:
        coef={final:1}
        coef['D_A010']=coef.get('D_A010',0)-1
        specs['FINAL_minus_D_A010']=coef
    for arm in ARMS:specs[f'{arm}_minus_P10K_REF']={arm:1,'P10K_REF':-1}
    result={}
    for name,coef in specs.items():
        if any(k not in values for k in coef):result[name]=dict(delta=None,coefficients=coef,status='未完成／不可比较');continue
        paired={}
        for task in TASKS:
            x=sum(c*finite(values[a]['tasks'][task]['folds']) for a,c in coef.items())
            paired[task]=dict(folds=x.tolist(),mean=float(x.mean()),std=float(x.std(ddof=1)))
        result[name]=dict(delta=float(sum(c*values[a]['macro'] for a,c in coef.items())),coefficients=coef,paired_tasks=paired,status='COMPLETE',
            interpretation='cross-resource/RNG numerical reference; not the single-factor effect of removing FP' if 'P10K_REF' in coef
            else 'same-cycle controlled contrast (identical data/initialization/budget)')
    return result


def pt_learning_rate(completed_updates):
    from scripts.pretrain_ua2d_galformer import learning_rate
    if type(completed_updates) is not int or not 0<=completed_updates<10000:raise ValueError('outside fixed10K PT')
    return learning_rate(completed_updates%5000)


def global_weight_warmup(completed_updates):
    return min(1.,completed_updates/1000.)


def window_objective(result,weights,denominators):
    """All six microbatch loss sums divide by this single optimization-window total.

    No DDP world multiplier and no second division by accumulation=6.
    """
    if set(weights)!=set(result['sums']) or set(denominators)!=set(weights):raise ValueError('loss target/count mismatch')
    return sum(weights[k]*v/denominators[k].clamp_min(1) for k,v in result['sums'].items())


def cache_references():
    reference=read(PT_CACHE/'metadata.json');mapping=read(MAP_ROOT/'metadata.json')
    p10k=read(REFERENCE_P10K)
    if (reference.get('schema')!='ua2d_prepared_templates_r4' or reference.get('status')!='COMPLETE'
        or reference.get('train_rows')!=911391 or reference.get('source')!=str(ROOT/'data/processed/mcl_ph_r4_20260930/pi1m_direct_01.staging')
        or reference.get('index')!=str(ROOT/'results/mcl_ph_next_20260929/p0/direct_indices_v1/summary.json')):raise ValueError('PT cache/reference identity mismatch')
    if (mapping.get('plan')!='UA2D-PTSCALE-20261022-01' or mapping.get('status')!='COMPLETE'
        or mapping.get('schema')!='ua2d-ptscale-logical-map-r1' or mapping.get('records')!=910756
        or mapping.get('excluded')!=635 or mapping.get('parent_cache')!=str(PT_CACHE)
        or mapping.get('eligibility_semantics')!='logical_slot_not_source_row'):raise ValueError('original PTSCALE eligible mapping mismatch')
    for name in ('eligible_logical.npy','keys.npy','source_rows.npy'):
        if not (MAP_ROOT/name).is_file():raise FileNotFoundError('missing immutable mapping '+name)
    if not p10k.get('complete_formal_TEST') or p10k.get('FINAL')!='P10K':raise ValueError('historical P10K formal reference not complete')
    p10k_formal=p10k['formal']['P10K']
    if set(p10k_formal['tasks'])!=set(TASKS) or any(len(p10k_formal['tasks'][t]['folds'])!=5 for t in TASKS):raise ValueError('incomplete P10K reference')
    finite([p10k_formal['macro'],*[x for t in TASKS for x in p10k_formal['tasks'][t]['folds']]])
    common=torch.load(INITIALIZATION,map_location='cpu',weights_only=False)
    if (common.get('schema')!='ua2d_galformer_objectives_v3' or common.get('step')!=0 or common.get('seed')!=42
        or common.get('plan')!='UA2D-PRETRAIN-FP-20261019-01'):raise ValueError('common step0 identity mismatch')
    return dict(pt_cache=dict(root=str(PT_CACHE),metadata=reference),
        pt_eligible=dict(root=str(MAP_ROOT),metadata=mapping),
        ft_cache=dict(root=str(CACHE_ROOT),metadata=cache_metadata(CACHE_ROOT)),
        p10k_reference=dict(source=str(REFERENCE_P10K),macro=p10k_formal['macro'],tasks=p10k_formal['tasks']),
        initialization=dict(path=str(INITIALIZATION),schema=common['schema'],step=common['step'],seed=common['seed'],
            plan=common['plan'],mask_head=str(INITIALIZATION.parent/'task_heads/V1.pt')),
        readonly=True,copied=False,rebuilt=False)


def summarize_timing(records):
    if not records:raise ValueError('no real timing records')
    result=dict(first_step=records[0]['step'],last_step=records[-1]['step'],steps=len(records))
    for key in ('step_seconds','loader_wait_seconds','decode_seconds','data_seconds'):
        v=finite([r[key] for r in records]);result[key]=dict(mean=float(v.mean()),median=float(np.median(v)),p90=float(np.percentile(v,90)),total=float(v.sum()))
    result['graphs_per_second']=len(records)*1008/result['step_seconds']['total']
    return result


def process_identity(pid):
    p=Path('/proc')/str(pid);stat=(p/'stat').read_text().split(') ',1)[1].split()
    return dict(pid=pid,start_ticks=int(stat[19]))


def process_alive(identity):
    try:
        stat=(Path('/proc')/str(identity['pid'])/'stat').read_text().split(') ',1)[1].split()
        return stat[0]!='Z' and int(stat[19])==identity['start_ticks']
    except FileNotFoundError:return False


def validate_resume_metadata(package,arm):
    stream=package.get('stream_positions',{})
    if arm not in ARMS:raise ValueError('unregistered arm')
    expected3d=60000 if arm_config(arm).branch3d else 0
    if (package.get('schema')!=PT_SCHEMA or package.get('plan')!=PLAN or package.get('revision')!=REVISION
        or package.get('arm')!=arm or package.get('step')!=10000 or package.get('completed_updates')!=10000
        or package.get('attempted_updates')!=10000 or package.get('schedule_position')!=10000
        or package.get('schedule_cycle_position')!=0 or package.get('global_weight_warmup_position')!=10000
        or package.get('next_global_position')!=10080000 or package.get('next_micro_index')!=60000
        or stream.get('two_d_position')!=60000 or stream.get('three_d_position')!=expected3d
        or package.get('execution',{}).get('world_size')!=1):raise ValueError('single-GPU resume step/branch/stream mismatch')
    policy,_=parse_model_stream_state(stream,arm_config(arm).branch3d)
    if policy!=LEGACY_STREAM_POLICY:raise ValueError('original r1 resume requires legacy stream policy')
    return 10000


def restore_pt_evidence(model,optimizer,package):
    """Strict reconstruction for evidence review; no controller retry/resume entry."""
    step=validate_resume_metadata(package,package.get('arm'))
    strict_module_state(model,package['model']);optimizer.load_state_dict(package['optimizer'])
    model.completed_updates=step
    model.restore_stream_state(package['stream_positions'])
    return dict(step=step,next_global_position=package['next_global_position'],
        next_micro_index=package['next_micro_index'],stream_positions=package['stream_positions'])
