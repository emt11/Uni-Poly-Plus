"""Isolated same-trajectory BEST/SWA4/EMA60; no inherited controller globals."""
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
from src.modules.ua2d_galformer_pretrain import package_attention,SCHEMA as PT_SCHEMA
from src.training.glt_dual_runtime import require_tmux,save_checkpoint
ROOT=Path(__file__).resolve().parents[2]
PLAN='UA2D-WAVG-20261022-01'
REVISION='r1'
OUT=ROOT/'results/ua2d_wavg_20261022/r1'
LOG=ROOT/'logs/ua2d_wavg_20261022/r1'
DOWN=ROOT/'data/processed/mcl_ph_r4_20260930/downstream_direct_01.staging'
SPLIT=ROOT/'data/splits/mips_paper5_outer'
PARENT=ROOT/'results/ua2d_pretrain_fp_20261019/fast_r4/pretrain/B_FP/deploy.pt'
HEADS=ROOT/'results/glt_optimization_20261011/r4/U1/initialization/heads_2d'
REFERENCE_BFP=PARENT.parents[2]/'final_review/results.json'
REFERENCE_OPT=ROOT/'results/ua2d_opt_20261021/r1/repair_ft1/results.json'
REFERENCE_TRANSFER=ROOT/'results/ua2d_transfer_20261022/r1/final_review/results.json'
CACHE_ROOT=ROOT/'results/ua2d_transfer_20261022/r1/inputs'
CACHE_PLAN='UA2D-TRANSFER-20261022-01'
KINDS=('BEST','SWA4','EMA60')
FINAL_ORDER=('SWA4','EMA60')
SCHEMA='ua2d-wavg-estimate-r1'
ARCHITECTURE=dict(schema=SCHEMA,max_hop=2,layers=6,dimension=512,attention_denominator=512,readout='mean_norm2',encoder_dropout=0.,head_dropout=.1)
TASKS=('eea','egb','egc','ei','eps','nc','xc')
CACHE_SCHEMA='ua2d-transfer-input-2hop-r1'
LIMITS=dict(prepare_starts=0,initialization_starts=0,pt_starts=0,pt_updates=0,pt_presentations=0,ft_starts=35,epochs_started=2450,attempted_updates=56000,validation_started=2520,test_starts=105,test_graphs=17625,cpu_suites=1)
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
def trajectory_path(task,fold):return OUT/'trajectories'/task/f'fold{fold}'
def unit_path(kind,task,fold,formal=False):return OUT/('formal' if formal else 'estimates')/kind/task/f'fold{fold}'

def config(task,fold):
    if task not in TASKS or fold not in range(5):raise ValueError('outside declared trajectory matrix')
    return dict(plan=PLAN,revision=REVISION,task=task,fold=fold,seed=42+fold,parent=str(PARENT),
        head_initial=str(HEADS/task/f'fold{fold}.pt'),source=str(DOWN),split=str(SPLIT/f'{task}.json'),
        cache_root=str(CACHE_ROOT),cache_schema=CACHE_SCHEMA,cache_plan=CACHE_PLAN,architecture=ARCHITECTURE,profile=PROFILE,
        trajectory=str(trajectory_path(task,fold)),SWA_epochs=[40,50,60,70],EMA_beta=.9,EMA_epochs=list(range(11,71)),
        loss='MSE',readout='mean_norm2',extra_training=False)


def update_plan(status,phase,detail):
    p=ROOT/'CURRENT_PLAN.md';text=p.read_text();start='\n### UA2D-WAVG 最新执行状态\n';end='\n<!-- UA2D_WAVG_EXECUTION_END -->'
    entry=start+f'\n- {now()}；状态：{status}；阶段：{phase}。\n- {detail}\n- 交付：{OUT}；日志：{LOG}；DSH执行自检不构成Codex独立审查。\n'+end
    if start in text:
        before,rest=text.split(start,1);_,after=rest.split(end,1);text=before+entry+after
    else:text=text.rstrip()+'\n'+entry+'\n'
    p.write_text(text);write(OUT/'status.json',dict(plan=PLAN,revision=REVISION,status=status,phase=phase,time=now(),detail=detail,independent_review='PENDING'))


def cache_metadata(root,expected_schema=CACHE_SCHEMA):
    root=Path(root);m=read(root/'metadata.json');coverage=read(root/'coverage.json')
    if (m.get('plan')!=CACHE_PLAN or m.get('schema')!=expected_schema or m.get('status')!='COMPLETE'
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
    def __init__(self,root,schema):
        self.root=Path(root);self.metadata=cache_metadata(root,schema);self.rows=read(self.root/'rows.json')
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
        self.task=task;self.cache=InputCache(CACHE_ROOT,CACHE_SCHEMA)
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




class MeanModel(Pure2DGLTModel):
    def __init__(self):
        super().__init__(dropout=0.,attention_scale_denominator=512);self.head[2].p=.1;self.stage='head'
    def train(self,mode=True):
        super().train(mode)
        if mode and self.stage=='head':self.o8.eval();self.norm2.eval()
        return self
    def forward(self,data):return super().forward(data).view(-1)


def initialize(parent,head):
    if (parent.get('schema')!=PT_SCHEMA or parent.get('step')!=5000
        or parent.get('config',{}).get('strategy')!='mask_align' or parent['config'].get('fp_enabled') is not True
        or parent.get('execution',{}).get('revision')!='r4' or parent.get('execution',{}).get('plan')!='UA2D-PRETRAIN-FP-20261019-01'
        or package_attention(parent)['scale_denominator']!=512):raise ValueError('B_FP parent identity mismatch')
    model=MeanModel()
    for name in ('o8','norm2'):
        prefix=name+'.';strict_module_state(getattr(model,name),{k[len(prefix):]:v for k,v in parent['state_dict'].items() if k.startswith(prefix)})
    strict_module_state(model.head,head);return model


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


class FixedAverages:
    """Detached device-local FP32 states; no forward, backward, RNG or student writes."""
    def __init__(self,model):
        state=model.state_dict();self.shapes={k:(v.shape,v.dtype,v.device) for k,v in state.items()}
        self.floating={k for k,v in state.items() if v.is_floating_point()}
        self.swa={k:torch.zeros_like(v,dtype=torch.float32) for k,v in state.items() if k in self.floating}
        self.ema={k:torch.zeros_like(v,dtype=torch.float32) for k,v in state.items() if k in self.floating}
        self.fixed={k:v.detach().clone() for k,v in state.items() if k not in self.floating}
        self.mismatch={k:torch.zeros((),dtype=torch.bool,device=v.device) for k,v in self.fixed.items()}
        self.last_epoch=0;self.swa_epochs=[];self.ema_epochs=[];self.ema_records=[];self.z=0.;self.beta=.9
    @torch.no_grad()
    def observe(self,model,epoch):
        if epoch!=self.last_epoch+1 or epoch not in range(1,71):raise ValueError('one ordered observation per global epoch required')
        state=model.state_dict()
        if set(state)!=set(self.shapes) or any((v.shape,v.dtype,v.device)!=self.shapes[k] for k,v in state.items()):raise ValueError('full averaging state/schema changed')
        for k,reference in self.fixed.items():self.mismatch[k].logical_or_(torch.any(state[k]!=reference))
        if epoch in (40,50,60,70):
            for k in self.swa:self.swa[k].add_(state[k].detach().float())
            self.swa_epochs.append(epoch)
        if epoch>=11:
            for k in self.ema:self.ema[k].mul_(.9).add_(state[k].detach().float(),alpha=.1)
            self.z=.9*self.z+.1;self.ema_epochs.append(epoch)
            self.ema_records.append(dict(global_epoch=epoch,joint_epoch=epoch-10,beta=.9,z=self.z,count=len(self.ema_epochs)))
        self.last_epoch=epoch
    def _complete(self):
        if self.last_epoch!=70 or self.swa_epochs!=[40,50,60,70] or self.ema_epochs!=list(range(11,71)) or len(self.ema_records)!=60:raise ValueError('fixed averaging sources incomplete')
    @torch.no_grad()
    def final_state(self,kind):
        self._complete()
        if kind not in ('SWA4','EMA60'):raise ValueError('not an averaged estimator')
        values=self.swa if kind=='SWA4' else self.ema;denominator=4 if kind=='SWA4' else self.z
        result={k:(v/denominator).detach().cpu().clone() for k,v in values.items()}
        result.update({k:v.detach().cpu().clone() for k,v in self.fixed.items()})
        # Integrity checks are materialized with the already-required checkpoint
        # transfer; observation never reads device scalars or synchronizes CUDA.
        if any(bool(x.detach().cpu()) for x in self.mismatch.values()):raise ValueError('nonfloating buffer changed across averaging sources')
        if any(not torch.isfinite(v).all() for v in result.values() if v.is_floating_point()):raise FloatingPointError('nonfinite averaged weights')
        return result
    def accumulator(self):
        self._complete()
        return dict(S={k:v.detach().cpu().clone() for k,v in self.ema.items()},z=self.z,beta=.9,count=60,
            source_epochs=list(self.ema_epochs),nonfloating={k:v.detach().cpu().clone() for k,v in self.fixed.items()})


def checkpoint_info(kind,cfg,best_epoch=None,best_stage=None):
    if kind=='BEST':
        if best_epoch not in range(1,71) or best_stage!=('head' if best_epoch<=10 else 'joint'):raise ValueError('invalid BEST epoch/stage')
        return dict(checkpoint_kind='BEST',source_epochs=[best_epoch],count=1,coefficients=[1.],
            best_epoch=best_epoch,best_stage=best_stage,selection='strict validation best across all70')
    if kind=='SWA4':
        return dict(checkpoint_kind='SWA4',source_epochs=[40,50,60,70],joint_source_epochs=[30,40,50,60],count=4,coefficients=[.25]*4,
            source_snapshots=[str(Path(cfg['trajectory'])/'swa_sources'/f'epoch{e}.pt') for e in (40,50,60,70)],selection='fixed epochs; never validation-selected')
    if kind=='EMA60':
        z=0.
        for _ in range(60):z=.9*z+.1
        return dict(checkpoint_kind='EMA60',source_epochs=list(range(11,71)),joint_source_epochs=list(range(1,61)),count=60,beta=.9,z=z,
            coefficients=[.1*.9**(70-e)/z for e in range(11,71)],initial_S=0,initial_z=0,
            accumulator=str(Path(cfg['trajectory'])/'ema_accumulator.pt'),selection='fixed epoch EMA, bias-corrected S/z')
    raise ValueError('unknown checkpoint estimator')


def restore(saved,cfg,kind):
    if (saved.get('schema')!=SCHEMA or saved.get('architecture')!=ARCHITECTURE or saved.get('config')!=cfg
        or saved.get('checkpoint_kind')!=kind):raise ValueError('WAVG complete checkpoint identity mismatch')
    if kind!='BEST' and ('best_epoch' in saved or 'best_stage' in saved):raise ValueError('averaged checkpoint must not impersonate a single BEST epoch')
    expected=checkpoint_info(kind,cfg,saved.get('best_epoch'),saved.get('best_stage'))
    if saved.get('checkpoint_metadata')!=expected:raise ValueError('checkpoint source/count/coefficient mismatch')
    with torch.random.fork_rng(devices=[]):model=MeanModel()
    strict_module_state(model,saved['state_dict']);return model


def evaluate(model,iterable,scaler,record_test=None):
    model.eval();truth=[];prediction=[]
    with torch.inference_mode():
        for batch in iterable:
            batch=batch.to('cuda:0');output=model(batch)
            if not torch.isfinite(output).all():raise FloatingPointError('nonfinite prediction')
            truth.append(batch.y.cpu().numpy());prediction.append(output.cpu().numpy())
            if record_test is not None:record_test(len(batch.y))
    y=scaler.inverse_transform(np.concatenate(truth)).reshape(-1);p=scaler.inverse_transform(np.concatenate(prediction)).reshape(-1)
    result=metrics(y,p);finite(list(result.values()));return y,p,result


def save_predictions(path,ds,y,p,info):
    np.savez(path,y_true=finite(y),y_pred=finite(p),indices=np.asarray([ds.indices[i] for i in ds.validation]),
        sample_keys=ds.keys[ds.validation],checkpoint_kind=info['checkpoint_kind'],source_epochs=np.asarray(info['source_epochs']))


def run_ft(task,fold):
    from src.utils import scale_targets
    require_tmux()
    if os.environ.get('CUDA_VISIBLE_DEVICES') not in ('2','3'):raise ValueError('GPU2/3 only')
    cache_metadata(CACHE_ROOT,CACHE_SCHEMA)
    folder=trajectory_path(task,fold);folder.mkdir(parents=True,exist_ok=False)
    cfg=config(task,fold);write(folder/'config.json',cfg)
    for kind in KINDS:
        d=unit_path(kind,task,fold);d.mkdir(parents=True,exist_ok=False);write(d/'config.json',cfg)
    state=dict(status='RUNNING',pid=os.getpid(),started=now(),epochs_started=0,completed_epochs=0,
        attempted_updates=0,completed_updates=0,validation_started=0,validation_completed=0,
        training_validation_started=0,training_validation_completed=0,endpoint_validation_started=0,
        endpoint_validation_completed=0,test_graphs=0,phase='initialize')
    started=time.monotonic();io_seconds=0.;io_calls=0;ds=None
    def persist(path,value,checkpoint=False):
        nonlocal io_seconds,io_calls
        tick=time.monotonic()
        if checkpoint:save_checkpoint(path,value)
        else:write(path,value)
        io_seconds+=time.monotonic()-tick;io_calls+=1
    def record():persist(folder/'consumption.json',state)
    record()
    try:
        seed_all(42+fold)
        parent=torch.load(PARENT,map_location='cpu',weights_only=False);head=torch.load(cfg['head_initial'],map_location='cpu',weights_only=False)
        model=initialize(parent,head);del parent,head
        ds=Data(task,fold);scaler=scale_targets(ds,task,train_indices=ds.train,transform_mode='standard')
        scaler_json=dict(mean=scaler.scaler.mean_.tolist(),scale=scaler.scaler.scale_.tolist(),fit='train',transform='standard')
        if scaler_json!=read(PARENT.parents[2]/'finetune/B_FP'/task/f'fold{fold}/scaler.json'):raise ValueError('original train-only scaler mismatch')
        write(folder/'scaler.json',scaler_json)
        for kind in KINDS:write(unit_path(kind,task,fold)/'scaler.json',scaler_json)
        model.to('cuda:0');averages=FixedAverages(model)
        validation=loader(ds,ds.validation);train=loader(ds,ds.train,True,42+fold);expected=70*len(train)
        seed_all(42+fold);history=[];timings=[];best_r2=-math.inf
        for stage,epochs in (('head',10),('joint',60)):
            opt,evidence=optimizer(model,stage)
            scheduler=(torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=10,eta_min=3e-5) if stage=='head'
                else torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt,T_0=10,T_mult=1,eta_min=1e-6))
            for stage_epoch in range(epochs):
                tick=time.monotonic();io_before=io_seconds
                state['phase']=stage;begin_epoch_reservation(state,len(train));record()
                rates=[g['lr'] for g in opt.param_groups];model.train();losses=[]
                for batch in train:
                    batch=batch.to('cuda:0');opt.zero_grad(set_to_none=True);output=model(batch)
                    loss=nn.functional.mse_loss(output,batch.y.view(-1))
                    if not torch.isfinite(loss):raise FloatingPointError('nonfinite MSE')
                    loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True)
                    if state['attempted_updates']>=expected:raise RuntimeError('FT updates exhausted')
                    state['attempted_updates']+=1;opt.step();state['completed_updates']+=1;losses.append(float(loss.detach()))
                scheduler.step();state['validation_started']+=1;state['training_validation_started']+=1
                y,p,met=evaluate(model,validation,scaler)
                state['validation_completed']+=1;state['training_validation_completed']+=1;state['completed_epochs']+=1
                epoch=state['completed_epochs'];averages.observe(model,epoch)
                if epoch in (40,50,60,70):
                    sources=folder/'swa_sources';sources.mkdir(exist_ok=True)
                    persist(sources/f'epoch{epoch}.pt',{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},True)
                if met['r2']>best_r2:
                    best_r2=met['r2'];best_epoch=epoch;best_stage=stage;best_y=y.copy();best_p=p.copy();best_metrics=dict(met)
                    best_info=checkpoint_info('BEST',cfg,best_epoch,best_stage)
                    persist(unit_path('BEST',task,fold)/'checkpoint.pt',dict(schema=SCHEMA,architecture=ARCHITECTURE,
                        checkpoint_kind='BEST',checkpoint_metadata=best_info,config=cfg,
                        state_dict={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                        best_epoch=best_epoch,best_stage=best_stage,scaler=scaler_json,outer_test='NOT_RUN'),True)
                rec=dict(epoch=epoch,stage=stage,stage_epoch=stage_epoch+1,train_loss=float(np.mean(losses)),
                    validation_r2=met['r2'],validation_rmse=met['rmse'],learning_rates=rates,optimizer_groups=evidence,
                    elapsed_seconds=time.monotonic()-started,gpu_peak_bytes=torch.cuda.max_memory_allocated(),
                    SWA_sources_so_far=list(averages.swa_epochs),EMA_count=len(averages.ema_epochs),EMA_z=averages.z)
                history.append(rec);persist(folder/'history.json',history)
                if epoch>=11:persist(folder/'ema_updates.json',averages.ema_records)
                close_epoch_reservation(state);record()
                timings.append(dict(epoch=epoch,epoch_seconds=time.monotonic()-tick,write_seconds=io_seconds-io_before,
                    write_calls_cumulative=io_calls,measurement='real training/validation/averaging/write; no added CUDA synchronization'))
                persist(folder/'epoch_timing.json',timings);print(json.dumps(rec),flush=True)
        if state['attempted_updates']!=expected or state['completed_updates']!=expected:raise ValueError('incomplete full FT trajectory')
        rng=dict(torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),numpy_rng=np.random.get_state(),
            python_rng=random.getstate(),loader_rng=train.generator.get_state())
        persist(folder/'ema_accumulator.pt',averages.accumulator(),True)
        best_info=checkpoint_info('BEST',cfg,best_epoch,best_stage)
        save_predictions(unit_path('BEST',task,fold)/'validation_predictions.npz',ds,best_y,best_p,best_info)
        write(unit_path('BEST',task,fold)/'metrics.json',dict(status='COMPLETE',config=cfg,checkpoint_metadata=best_info,scaler=scaler_json,**best_metrics))
        # Two endpoint validations only. Restore in a CPU RNG fork, and use a
        # private DataLoader generator so neither changes the student's stream.
        endpoint_loader=torch.utils.data.DataLoader(torch.utils.data.Subset(ds,ds.validation),batch_size=24,
            shuffle=False,num_workers=0,drop_last=False,collate_fn=collate,generator=torch.Generator().manual_seed(42+fold))
        sources={'BEST':best_info}
        for kind in ('SWA4','EMA60'):
            info=checkpoint_info(kind,cfg);sources[kind]=info
            package=dict(schema=SCHEMA,architecture=ARCHITECTURE,checkpoint_kind=kind,checkpoint_metadata=info,
                config=cfg,state_dict=averages.final_state(kind),scaler=scaler_json,outer_test='NOT_RUN')
            persist(unit_path(kind,task,fold)/'checkpoint.pt',package,True)
            averaged=restore(package,cfg,kind).to('cuda:0');del package
            state.update(phase=kind+'_endpoint_validation',validation_started=state['validation_started']+1,
                endpoint_validation_started=state['endpoint_validation_started']+1,endpoint_reservation=dict(open=True,kind=kind));record()
            tick=time.monotonic();ey,ep,em=evaluate(averaged,endpoint_loader,scaler)
            save_predictions(unit_path(kind,task,fold)/'validation_predictions.npz',ds,ey,ep,info)
            write(unit_path(kind,task,fold)/'metrics.json',dict(status='COMPLETE',config=cfg,checkpoint_metadata=info,scaler=scaler_json,**em))
            state['validation_completed']+=1;state['endpoint_validation_completed']+=1;state['endpoint_reservation']['open']=False;record()
            timings.append(dict(checkpoint_kind=kind,endpoint_validation_seconds=time.monotonic()-tick));del averaged
        write(folder/'estimator_sources.json',sources)
        if state['training_validation_started']!=70 or state['endpoint_validation_started']!=2 or state['validation_started']!=72:raise ValueError('shared training/endpoint count mismatch')
        persist(folder/'resume.pt',dict(schema='ua2d-wavg-trajectory-r1',config=cfg,model=model.state_dict(),optimizer=opt.state_dict(),
            scheduler=scheduler.state_dict(),stage='joint',stage_epoch=60,epoch=70,batch_cursor='epoch_end',
            best_epoch=best_epoch,best_stage=best_stage,best_r2=best_r2,scaler=scaler_json,**rng,consumption=copy.deepcopy(state),automatic_resume=False),True)
        persist(folder/'epoch_timing.json',timings)
        state.update(status='COMPLETE',ended=now(),elapsed_seconds=time.monotonic()-started,write_seconds=io_seconds,
            write_calls=io_calls,best_epoch=best_epoch,best_stage=best_stage,SWA_sources=list(averages.swa_epochs),EMA_updates=len(averages.ema_epochs));record()
    except BaseException:
        state.update(status='FAILED',error=traceback.format_exc(),ended=now());record();write(folder/'failure.json',dict(error=state['error'],automatic_retry=False));raise
    finally:
        if ds is not None:ds.close()


def run_test(kind,task,fold):
    from src.utils import TargetScaler
    from sklearn.preprocessing import StandardScaler
    require_tmux()
    if os.environ.get('CUDA_VISIBLE_DEVICES') not in ('2','3') or kind not in KINDS:raise ValueError('outside fixed formal TEST')
    folder=unit_path(kind,task,fold,True);folder.mkdir(parents=True,exist_ok=False);cfg=config(task,fold);write(folder/'config.json',cfg)
    state=dict(status='RUNNING',pid=os.getpid(),started=now(),test_started=0,test_graphs=0);ds=None
    def record():write(folder/'consumption.json',state)
    record()
    try:
        frozen=read(OUT/'selection/final_frozen.json');selected=frozen['units'][f'{kind}/{task}/{fold}']
        expected=unit_path(kind,task,fold)/'checkpoint.pt'
        if (not frozen['frozen'] or frozen['models']!=list(KINDS) or selected['checkpoint']!=str(expected)
            or selected['config']!=cfg or frozen['splits'][task]!=read(SPLIT/f'{task}.json')):raise ValueError('frozen source/config/split mismatch')
        saved=torch.load(expected,map_location='cpu',weights_only=False)
        if saved['checkpoint_metadata']!=selected['checkpoint_metadata'] or saved['scaler']!=selected['scaler']:raise ValueError('frozen estimator provenance/scaler changed')
        model=restore(saved,cfg,kind).to('cuda:0');scaler_json=saved['scaler'];del saved
        scaler=TargetScaler(task,StandardScaler(),transform_mode='standard');scaler.scaler.mean_=np.asarray(scaler_json['mean']);scaler.scaler.scale_=np.asarray(scaler_json['scale']);scaler.scaler.var_=scaler.scaler.scale_**2;scaler.scaler.n_features_in_=1
        ds=Data(task,fold,True)
        if ds.indices!=selected['test_indices']:raise ValueError('frozen TEST indices mismatch')
        ds.set_target_override(scaler.transform(ds.raw_targets).reshape(-1));write(folder/'scaler.json',scaler_json)
        state.update(test_started=1,phase='TEST');record()
        def counted(n):state['test_graphs']+=n;record()
        y,p,m=evaluate(model,loader(ds,ds.validation),scaler,counted)
        save_predictions(folder/'test_predictions.npz',ds,y,p,selected['checkpoint_metadata'])
        write(folder/'metrics.json',dict(status='COMPLETE',config=cfg,checkpoint_metadata=selected['checkpoint_metadata'],scaler=scaler_json,**m))
        state.update(status='COMPLETE',ended=now());record()
    except BaseException:
        state.update(status='FAILED',error=traceback.format_exc(),ended=now());record();write(folder/'failure.json',dict(error=state['error'],automatic_retry=False));raise
    finally:
        if ds is not None:ds.close()


def aggregate(kind,formal=False):
    tasks={};units={}
    for task in TASKS:
        values=[]
        for fold in range(5):
            path=unit_path(kind,task,fold,formal);m=read(path/'metrics.json');cfg=config(task,fold)
            trajectory=read(trajectory_path(task,fold)/'consumption.json')
            expected=70*math.ceil(len(read(SPLIT/f'{task}.json')['folds'][fold]['train_indices'])/24)
            if (trajectory['status']!='COMPLETE' or trajectory['completed_epochs']!=70 or trajectory['attempted_updates']!=expected
                or trajectory['completed_updates']!=expected or trajectory['training_validation_started']!=70
                or trajectory['training_validation_completed']!=70 or trajectory['endpoint_validation_started']!=2
                or trajectory['endpoint_validation_completed']!=2 or trajectory['validation_started']!=72
                or trajectory['validation_completed']!=72 or trajectory['epoch_reservation']['open']
                or trajectory.get('endpoint_reservation',{}).get('open')):raise ValueError('shared FT trajectory/endpoints incomplete')
            if m['status']!='COMPLETE' or m['config']!=cfg or m['checkpoint_metadata']['checkpoint_kind']!=kind:raise ValueError('estimator result identity mismatch')
            info=m['checkpoint_metadata'];expected_info=checkpoint_info(kind,cfg,info.get('best_epoch'),info.get('best_stage'))
            if info!=expected_info:raise ValueError('estimator provenance mismatch')
            if formal:
                c=read(path/'consumption.json')
                if c['status']!='COMPLETE' or c['test_started']!=1 or c['test_graphs']!=len(read(SPLIT/f'{task}.json')['folds'][fold]['test_indices']):raise ValueError('incomplete TEST predictions')
            values.append(m['r2']);units[f'{kind}/{task}/{fold}']=dict(config=cfg,checkpoint=str(unit_path(kind,task,fold)/'checkpoint.pt'),
                checkpoint_kind=kind,checkpoint_metadata=info,scaler=m['scaler'],scaler_path=str(unit_path(kind,task,fold)/'scaler.json'),
                trajectory=str(trajectory_path(task,fold)),test_indices=sorted(read(SPLIT/f'{task}.json')['folds'][fold]['test_indices']),
                predictions=str(path/('test_predictions.npz' if formal else 'validation_predictions.npz')))
        tasks[task]=dict(folds=values,mean=float(finite(values).mean()),std=float(np.std(values,ddof=1)))
    return dict(tasks=tasks,macro=float(np.mean([x['mean'] for x in tasks.values()])),units=units)


def nominate(validation):
    if set(validation)!=set(KINDS):raise ValueError('all 35 trajectories and three complete estimators required')
    finite([validation[k]['macro'] for k in KINDS])
    return max(FINAL_ORDER,key=lambda k:(validation[k]['macro'],-FINAL_ORDER.index(k)))


def freeze_final(validation):
    final=nominate(validation);units={u:v for kind in KINDS for u,v in validation[kind]['units'].items()}
    if set(units)!={f'{k}/{t}/{f}' for k in KINDS for t in TASKS for f in range(5)}:raise ValueError('full 105 estimator units required')
    for key,value in units.items():
        kind,task,fold=key.split('/');cfg=config(task,int(fold));info=value['checkpoint_metadata']
        if (value['config']!=cfg or value['checkpoint']!=str(unit_path(kind,task,int(fold))/'checkpoint.pt')
            or info!=checkpoint_info(kind,cfg,info.get('best_epoch'),info.get('best_stage'))):raise ValueError('frozen estimator source mismatch')
        required=[Path(value['checkpoint'])]
        if kind=='SWA4':required += [Path(p) for p in info['source_snapshots']]
        if kind=='EMA60':required += [Path(info['accumulator']),Path(cfg['trajectory'])/'ema_updates.json']
        if any(not p.is_file() for p in required):raise ValueError('frozen estimator source artifact missing')
    frozen=dict(plan=PLAN,revision=REVISION,frozen=True,time=now(),FINAL=final,models=list(KINDS),units=units,
        validation_scores={k:validation[k]['macro'] for k in KINDS},splits={t:read(SPLIT/f'{t}.json') for t in TASKS},
        selection='validation among SWA4/EMA60 only; BEST is direct same-trajectory control; all105 TEST without gate')
    p=OUT/'selection/final_frozen.json';p.parent.mkdir(exist_ok=True)
    with p.open('x') as f:json.dump(frozen,f,ensure_ascii=False,indent=2,allow_nan=False)
    return frozen


def registered_comparisons(formal,references,final):
    values=dict(formal,**references)
    specs={'SWA4_minus_BEST':('SWA4','BEST'),'EMA60_minus_BEST':('EMA60','BEST'),
        **{k+'_minus_B_FP':(k,'B_FP') for k in KINDS},
        'FINAL_minus_A_MIX246':(final,'A_MIX246'),'FINAL_minus_R_MEAN':(final,'R_MEAN')}
    result={}
    for label,(a,b) in specs.items():
        if a not in values or b not in values:result[label]=dict(delta=None,status='未完成／不可比较',project=a,reference=b);continue
        tasks={}
        for t in TASKS:
            x=finite(values[a]['tasks'][t]['folds'])-finite(values[b]['tasks'][t]['folds'])
            tasks[t]=dict(folds=x.tolist(),mean=float(x.mean()),std=float(x.std(ddof=1)))
        result[label]=dict(delta=values[a]['macro']-values[b]['macro'],status='COMPLETE',project=a,reference=b,paired_tasks=tasks)
    return result


def performance(direct,historical):
    if direct is None or historical is None:return dict(performance_pass=None,performance_status='未完成／不可比较')
    finite([direct,historical]);passed=direct>=.010 and historical>=.010
    return dict(performance_pass=passed,positive_direct_delta=direct>0,positive_historical_delta=historical>0,
        performance_status='通过' if passed else '本次正增量但未达双参考门槛' if direct>0 and historical>0 else '未达到双参考性能门槛')
