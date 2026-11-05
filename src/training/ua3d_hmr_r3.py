"""HMR science r2, execution r3; lifecycle and canonical-source repair, no BASE model import or forward path."""
import json
import math
import os
import random
import subprocess
import sys
import time
import traceback
from pathlib import Path
import numpy as np
import torch
from scripts.bootstrap_ua3d_hmr_r3 import ROOT,OUT,LOG,SOURCES,THREADS,P1,P2,PYTHON,ALIASES,read,write,exclusive,identity,parse_jobs,state_for,size,check_clock,fail,canonical_entry,unwrap_science,require_frozen,Lifecycle,cumulative
from src.dataset import ua3d_hmr_r2 as d
from src.modules import ua3d_hmr_r2 as m
BASE=ROOT/'results/ua3d_base_opt_20261026/r6/repairs/concurrency_r1'
PARENT=ROOT/'results/ua3d_anchor_opt_20261102/r2/feature_cache'
SIDE=ROOT/'results/ua3d_ph_20261028/r1/repairs/gpu2_r2/inputs'
INPUT=ROOT/'results/ua3m_paperfuse_20261025/r1/repairs/gpu0123_restart1/inputs'
DOWN=ROOT/'data/processed/mcl_ph_r4_20260930/downstream_direct_01.staging'
SPLIT=ROOT/'data/splits/mips_paper5_outer'
TASKS=('eea','egb','egc','ei','eps','nc','xc')
REFERENCE=.804675462276949
EXEC=read(ROOT/'configs/ua3d_hmr_r3_gpu2.json')
SPEC=EXEC['science_config']
SCHEMA='ua3d-hmr-residual-r2'
PYTHON=ROOT/'results/ua3d_hmr_20261103/r1/env/bin/python'

def cpu_contract():
    if not os.environ.get('TMUX') or os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU tmux/CVD contract')
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        if os.environ.get(key)!='1':raise ValueError('thread contract '+key)
def gpu_contract():
    if not os.environ.get('TMUX') or os.environ.get('CUDA_VISIBLE_DEVICES')!='2' or os.environ.get('WORLD_SIZE')!='1':raise ValueError('GPU2/tmux/world1')
    if torch.cuda.device_count()!=1:raise ValueError('one visible device')
    uuid='GPU-'+str(torch.cuda.get_device_properties(0).uuid).removeprefix('GPU-')
    if uuid!=SPEC['uuid']:raise ValueError('GPU UUID')
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    return dict(uuid=uuid,physical=2,logical='cuda:0',threads=1,TF32=False,AMP=False)
def seed(seed):random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed) if torch.cuda.is_available() else None

def metrics(y,p):
    y=np.asarray(y,dtype=np.float64).reshape(-1);p=np.asarray(p,dtype=np.float64).reshape(-1);d.finite(y,p)
    if len(y)!=len(p):raise ValueError('metric length')
    return dict(r2=float(1-((y-p)**2).sum()/((y-y.mean())**2).sum()),rmse=float(np.sqrt(((y-p)**2).mean())),mae=float(np.abs(y-p).mean()))
def inverse(x,scaler):return np.asarray(x,dtype=np.float64)*scaler['scale'][0]+scaler['mean'][0]
def base_folder(task,fold):return BASE/'finetune/F_PD_N1_F0'/task/f'fold{fold}'
def anchor(task,fold):return read(OUT/'registry/anchors.json')['anchors'][f'{task}/{fold}']
def cache_path(task,fold,formal):
    if formal:require_frozen(OUT)
    return (OUT if formal else P2)/'feature_cache'/('test' if formal else 'trainval')/task/f'fold{fold}.pt'
def load_cache(task,fold,formal):
    return torch.load(cache_path(task,fold,formal),map_location='cpu',weights_only=False)
def partition_positions(cache,part,task,fold):
    e=read(SPLIT/f'{task}.json')['folds'][fold];ids=sorted(e['test_indices']) if part=='test' else e[part+'_indices'];pos={row:i for i,row in enumerate(cache['indices'].tolist())};return [pos[row] for row in ids]

def pack_cache(task,fold,formal):
    import threading
    if not formal:raise ValueError('r3 cannot repack trainval')
    cpu_contract();check_clock(not formal)
    if formal and not read(OUT/'selection/final_frozen.json')['frozen']:raise ValueError('TEST access before unique freeze')
    source=PARENT/('test' if formal else 'trainval')/task/f'fold{fold}';meta=read(source/'metadata.json');prior=read(source/'consumption.json');a=anchor(task,fold);e=read(SPLIT/f'{task}.json')['folds'][fold]
    if meta['schema']!='ua3d-anchor-feature-cache-r2' or meta['status']!='COMPLETE' or prior['status']!='COMPLETE' or meta['task']!=task or meta['fold']!=fold or meta['formal']!=formal:raise ValueError('teacher cache schema/status')
    for k,v in {'source_best':'best','source_config':'source_config','source_schema':'schema','source_best_epoch':'best_epoch','source_best_stage':'best_stage','scaler':'scaler'}.items():
        if meta[k]!=a[v]:raise ValueError('teacher source '+k)
    if Path(a['best']).stat().st_mtime>prior['started_unix']:raise ValueError('BASE newer than teacher')
    mapping={r['row']:r for r in read(OUT/'registry/anchors.json')['rows'][task]};expected_parts=['test'] if formal else ['train','validation'];records={};arrays={};reused=0
    for part in expected_parts:
        ids=sorted(e['test_indices']) if formal else e[part+'_indices']
        if meta['partitions'][part]!=ids:raise ValueError('teacher split order')
        with np.load(source/f'{part}_predictions.npz',allow_pickle=False) as x:pred={k:x[k].copy() for k in x.files}
        if pred['indices'].tolist()!=ids or pred['sample_keys'].tolist()!=[mapping[r]['key'] for r in ids]:raise ValueError('teacher arrays identity')
        arrays[part]=pred
        for i,row in enumerate(ids):
            p=source/f'row{row}.pt';r=torch.load(p,map_location='cpu',weights_only=False);reused+=p.stat().st_size
            if any(r[k]!=v for k,v in dict(schema='ua3d-anchor-feature-cache-r2',task=task,fold=fold,row=row,key=mapping[row]['key'],partition=part,source_best=a['best']).items()):raise ValueError('teacher row identity')
            if r['availability'].tolist()!=[True,mapping[row]['available'],True] or any(x.shape!=(256,) for x in r['z']):raise ValueError('teacher availability/features')
            for x in [*r['z'],r['y'],r['y_base']]:
                if x.dtype!=torch.float32 or x.requires_grad or x.is_inference() or not torch.isfinite(x).all():raise ValueError('teacher finite ordinary FP32')
            if not mapping[row]['available'] and torch.count_nonzero(r['z'][1]):raise ValueError('missing z3 nonzero')
            if float(inverse(r['y'].numpy(),a['scaler']))!=pred['y_true'][i] or r['y_base_original']!=pred['y_pred'][i]:raise ValueError('teacher row/array disagree')
            # Target-bearing atoms/bonds are never copied into the new architecture.
            records[row]=dict(z=torch.cat(r['z']),availability=r['availability'].float(),y=r['y'],yb=r['y_base'],yb_original=float(r['y_base_original']),y_original=float(pred['y_true'][i]))
    if not formal:
        gate=read(source/'validation_gate.json')
        if not gate['identity_equal'] or not gate['prediction_allclose'] or abs(gate['r2_delta'])>1e-6 or gate['scaler']!=a['scaler']:raise ValueError('parent validation qualification')
        with np.load(a['validation_predictions'],allow_pickle=False) as x:
            original={k:x[k].copy() for k in x.files};val=arrays['validation']
        if any(not np.array_equal(val[k],original[k]) for k in ('indices','sample_keys','y_true')) or not np.allclose(val['y_pred'],original['y_pred'],atol=1e-6,rtol=1e-5) or abs(metrics(val['y_true'],val['y_pred'])['r2']-metrics(original['y_true'],original['y_pred'])['r2'])>1e-6:raise ValueError('BASE validation comparison')
    ids=sorted(records);payload={k:torch.stack([records[row][k] for row in ids]) for k in ('z','availability','y','yb')};payload.update(indices=torch.tensor(ids),keys=[mapping[row]['key'] for row in ids],yb_original=torch.tensor([records[row]['yb_original'] for row in ids],dtype=torch.float64),y_original=torch.tensor([records[row]['y_original'] for row in ids],dtype=torch.float64),scaler=a['scaler'],source=str(source),partitions={k:arrays[k]['indices'].tolist() for k in expected_parts})
    path=cache_path(task,fold,formal);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():raise ValueError('compact cache overwrite')
    torch.save(payload,path)
    exclusive(OUT/'registry/cache_qualification'/('test' if formal else 'trainval')/task/f'fold{fold}.json',dict(status='COMPLETE',source=str(source),rows=len(ids),bytes=path.stat().st_size,reused_bytes=reused,base=a,formal=formal,completed_unix=time.time(),encoder_forwards=0,thread=threading.get_ident()))
    if size(OUT/'feature_cache')>1024**3:raise ValueError('compact teacher >1GiB')

def batch_data(task,fold,formal,device):
    c=load_cache(task,fold,formal);s=d.load_surface(P2,task);rows=c['indices'];b={k:s[k][rows].to(device) for k in s};b.update({k:c[k].to(device) for k in ('z','availability','y','yb')});return c,b

def slices(b,ids):return {k:v[ids] for k,v in b.items()}
def save_model(path,model,arm,task,fold,epoch,met,stats):
    if path.exists():path.unlink()  # Only this unit's own evolving BEST/last, never old sources.
    payload=dict(schema=SCHEMA,arm=arm,task=task,fold=fold,seed=42+fold,state_dict=model.state_dict(),epoch=epoch,metrics=met,stats=stats,base=anchor(task,fold),config=SPEC)
    torch.save(dict(schema='ua3d-hmr-execution-r3',science_revision='r2',execution_revision='r3',source_kind='r3_new_ft',science_payload=payload),path)
def assert_finite_model(model):
    if any(not torch.isfinite(p).all() for p in model.parameters()):raise FloatingPointError('nonfinite model')
def better(candidate,best):return candidate['rmse']<best['rmse']

def optimizer_for(model):
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.0003,weight_decay=.0001,betas=(.9,.999),eps=1e-8)
    return opt,torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=70,eta_min=1e-6)

def fit_job(kind,arm,task,fold,repeat=''):
    if kind!='ft' or canonical_entry(OUT,arm,task,fold)['source_kind']!='r3_new_ft':raise ValueError('only 103 new FT authorized')
    gpu=gpu_contract();seed(42+fold);device='cuda:0';c,b=batch_data(task,fold,False,device);stats=read(P2/'scalers'/task/f'fold{fold}.json');model=m.Residual(arm,stats).to(device);train=partition_positions(c,'train',task,fold);val=partition_positions(c,'validation',task,fold);folder=job_folder(kind,arm,task,fold,repeat);folder.mkdir(parents=True,exist_ok=False)
    opt,sch=optimizer_for(model)
    write(folder/'device_start.json',dict(**gpu,**identity(),time=time.time()))
    usage=Usage(folder);usage.record();history=[];best_epoch=0;best=metrics(c['y_original'][val].numpy(),c['yb_original'][val].numpy());save_model(folder/'epoch0.pt',model,arm,task,fold,0,best,stats)
    initial={k:v.detach().clone() for k,v in model.state_dict().items()};finite_grad=False
    def update(ids):
        nonlocal finite_grad
        model.train();opt.zero_grad(set_to_none=True);mini=slices(b,ids);usage.add('train_forwards');usage.add('train_rows',len(ids));prediction=model(mini);loss=(prediction-mini['y']).square().mean()+.01*(prediction-mini['yb']).square().mean()
        if not torch.isfinite(loss):raise FloatingPointError('loss')
        usage.add('backwards');loss.backward()
        grads=[p.grad for p in model.parameters() if p.grad is not None]
        if not grads or any(not torch.isfinite(g).all() for g in grads):raise FloatingPointError('gradients')
        finite_grad=finite_grad or any(torch.count_nonzero(g) for g in grads);torch.nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True);usage.add('updates');opt.step();assert_finite_model(model);usage.record();return float(loss.detach())
    if kind=='diagnostic':
        n=64 if task=='eps' else 256;ids=torch.tensor(train[:n],device=device)
        if len(ids)!=n:raise ValueError('diagnostic size')
        for _ in range(2):
            model.eval();usage.add('eval_forwards');usage.add('eval_rows',len(ids));usage.record()
            with torch.no_grad():p=model(slices(b,ids))
            if not torch.equal(p,b['yb'][ids]):raise ValueError('zero residual anchor exact equality')
        for step in range(20):
            lr=opt.param_groups[0]['lr']
            if lr!=.0003:raise ValueError('diagnostic lr/scheduler')
            history.append(dict(step=step,loss=update(ids),lr=lr))
        model.eval()
        with torch.no_grad():
            # No extra model forward: mask and zero checks were covered in the two reserved evals.
            changed=[k for k,v in model.state_dict().items() if not torch.equal(v,initial[k])]
        if not finite_grad or not changed or not any(k.startswith('head.0') for k in changed) or (arm=='S_HMR' and 'filter_raw' not in changed):raise ValueError('diagnostic activation/gradient gate')
        torch.save(dict(state=model.state_dict(),initial=initial,history=history,changed=changed),folder/'trajectory.pt')
    else:
        usage.add('epoch0_aliases');usage.add('epoch0_alias_rows',len(val));save_model(folder/'best.pt',model,arm,task,fold,0,best,stats);usage.record()
        for epoch in range(1,71):
            check_clock();ids=torch.tensor(train,device=device)[torch.randperm(len(train),device=device)];losses=[];lr=opt.param_groups[0]['lr']
            expected=1e-6+(.0003-1e-6)*(1+math.cos(math.pi*(epoch-1)/70))/2
            if not math.isclose(lr,expected,rel_tol=1e-10,abs_tol=1e-12):raise ValueError('normal scheduler LR')
            for ids_batch in ids.split(256):losses.append(update(ids_batch))
            model.eval();pred=[];usage.add('validation');usage.add('validation_rows',len(val))
            with torch.no_grad():
                for ids_batch in torch.tensor(val,device=device).split(256):usage.add('validation_batches');pred.append(model(slices(b,ids_batch)).cpu())
            p=inverse(torch.cat(pred).numpy(),c['scaler']);met=metrics(c['y_original'][val].numpy(),p)
            if better(met,best):best=met;best_epoch=epoch;save_model(folder/'best.pt',model,arm,task,fold,epoch,best,stats)
            sch.step();usage.add('epochs');history.append(dict(epoch=epoch,lr=lr,loss=float(np.mean(losses)),validation=met,best_epoch=best_epoch));write(folder/'history.json',history);usage.record()
        save_model(folder/'last.pt',model,arm,task,fold,70,history[-1]['validation'],stats);write(folder/'metrics.json',dict(status='COMPLETE',best_epoch=best_epoch,validation=best,epochs=70,arm=arm,task=task,fold=fold))
    usage.finish();write(folder/'gpu.json',dict(**gpu,peak_memory_mib=torch.cuda.max_memory_allocated()/1024**2,finished_unix=time.time()))

class Usage:
    def __init__(self,folder):self.path=Path(folder)/'consumption.json';self.data=dict(status='RUNNING',attempted={},completed={},started_unix=time.time(),ended_unix=None)
    def add(self,key,n=1):self.data['attempted'][key]=self.data['attempted'].get(key,0)+n
    def record(self):write(self.path,self.data)
    def finish(self):self.data.update(status='COMPLETE',completed=dict(self.data['attempted']),ended_unix=time.time());self.record()
def job_folder(kind,arm='',task='',fold=0,repeat=''):
    if kind in ('ft','test'):return OUT/('finetune' if kind=='ft' else 'formal')/arm/task/f'fold{fold}'
    if kind=='diagnostic':return OUT/'diagnostics'/('S' if task=='eps' else 'L')/arm/repeat
    return OUT/kind

def formal_job(arm,task,fold):
    if not read(OUT/'selection/final_frozen.json')['frozen']:raise ValueError('formal before freeze')
    c=load_cache(task,fold,True);ids=partition_positions(c,'test',task,fold);folder=job_folder('test',arm,task,fold);folder.mkdir(parents=True,exist_ok=False);u=Usage(folder);u.record()
    if arm=='C_ANCHOR':
        cpu_contract();p=c['yb_original'][ids].numpy();u.add('anchor_batches',math.ceil(len(ids)/256))
    else:
        gpu=gpu_contract();seed(42+fold);c,b=batch_data(task,fold,True,'cuda:0');saved=read_science(arm,task,fold,'best');frozen=read(OUT/'selection/final_frozen.json')['estimators'][f'{arm}/{task}/{fold}']
        source=canonical_entry(OUT,arm,task,fold)
        if frozen['source_kind']!=source['source_kind'] or frozen['path']!=str(Path(source['folder'])/'best.pt'):raise ValueError('frozen canonical resolver provenance')
        if saved['schema']!=SCHEMA or saved['epoch']!=frozen['epoch'] or saved['base']!=anchor(task,fold) or any(saved[k]!=v for k,v in dict(arm=arm,task=task,fold=fold,config=SPEC).items()):raise ValueError('frozen BEST identity')
        model=m.Residual(arm,saved['stats']).to('cuda:0');model.load_state_dict(saved['state_dict'],strict=True);model.eval();pred=[]
        with torch.no_grad():
            for chunk in torch.tensor(ids,device='cuda:0').split(256):u.add('candidate_forwards');u.record();pred.append(model(slices(b,chunk)).cpu())
        p=inverse(torch.cat(pred).numpy(),c['scaler']);write(folder/'gpu.json',dict(**gpu,peak_memory_mib=torch.cuda.max_memory_allocated()/1024**2))
    y=c['y_original'][ids].numpy();indices=c['indices'][ids].numpy();keys=np.array([c['keys'][i] for i in ids]);met=metrics(y,p);u.add('test_rows',len(ids));u.add('test_batches',math.ceil(len(ids)/256));np.savez(folder/'test_predictions.npz',indices=indices,sample_keys=keys,y_true=y,y_pred=p)
    if arm=='C_ANCHOR':
        with np.load(BASE/'formal/F_PD_N1_F0'/task/f'fold{fold}'/'test_predictions.npz',allow_pickle=False) as orig:
            if any(not np.array_equal(v,orig[k]) for k,v in dict(indices=indices,sample_keys=keys,y_true=y).items()) or not np.allclose(p,orig['y_pred'],atol=1e-6,rtol=1e-5) or abs(met['r2']-metrics(orig['y_true'],orig['y_pred'])['r2'])>1e-6:raise ValueError('formal anchor identity/numerical gate')
    write(folder/'metrics.json',dict(status='COMPLETE',metrics=met,arm=arm,task=task,fold=fold,rows=len(ids)));u.finish()

def aggregate(formal=False):
    out={}
    for arm in m.ARMS:
        tasks={};complete=True
        for task in TASKS:
            values=[]
            for fold in range(5):
                if formal:
                    p=job_folder('test',arm,task,fold)/'metrics.json'
                    if not p.exists():complete=False;break
                    values.append(read(p)['metrics']['r2'])
                elif arm=='C_ANCHOR':
                    c=load_cache(task,fold,False);ids=partition_positions(c,'validation',task,fold);values.append(metrics(c['y_original'][ids],c['yb_original'][ids])['r2'])
                else:
                    p=Path(canonical_entry(OUT,arm,task,fold)['folder'])/'metrics.json'
                    if not p.exists():complete=False;break
                    values.append(read(p)['validation']['r2'])
            if len(values)==5:tasks[task]=dict(folds=values,mean=float(np.mean(values)),std=float(np.std(values,ddof=1)))
        if complete and len(tasks)==7:out[arm]=dict(tasks=tasks,macro=float(np.mean([v['mean'] for v in tasks.values()])))
    return out

def choose(validation):
    base=validation['C_ANCHOR'];eligible=[a for a in ('S_HMR','S_FIXED','S_STAT') if validation[a]['macro']>base['macro'] and all(validation[a]['tasks'][t]['mean']>=base['tasks'][t]['mean']-.010 for t in TASKS)]
    return max(eligible,key=lambda a:validation[a]['macro']) if eligible else 'BASE_KEEP'
def freeze():
    validation=aggregate(False)
    if len(validation)!=4:raise ValueError('freeze without 105 completed')
    selected=choose(validation);estimators={};exclusive(OUT/'selection/freeze_reservation.json',dict(time=time.time(),count=1))
    for arm in m.ARMS:
        for task in TASKS:
            for fold in range(5):
                key=f'{arm}/{task}/{fold}';a=anchor(task,fold)
                if arm=='C_ANCHOR':estimators[key]=dict(base=a,kind='read_only_teacher')
                else:
                    entry=canonical_entry(OUT,arm,task,fold);p=Path(entry['folder'])/'best.pt';s=read_science(arm,task,fold,'best')
                    estimators[key]=dict(source_kind=entry['source_kind'],path=str(p),epoch=s['epoch'],base=a,stats=s['stats'],scaler=a['scaler'],config=SPEC,source_cache=str(cache_path(task,fold,False)),surface_mapping=str(P2/'surface/mapping.json'),input_rows=read(OUT/'registry/anchors.json')['rows'][task])
    exclusive(OUT/'selection/final_frozen.json',dict(frozen=True,frozen_unix=time.time(),FINAL=selected,validation=validation,estimators=estimators,reference=REFERENCE,target_version='near809-met810',surface_config=SPEC['surface'],shared_development_test=True,science_revision='r2',execution_revision='r3'));return selected

def read_science(arm,task,fold,name='best'):
    entry=canonical_entry(OUT,arm,task,fold);saved=torch.load(Path(entry['folder'])/f'{name}.pt',map_location='cpu',weights_only=False);payload=unwrap_science(saved,entry,SPEC)
    shapes={'filter_raw':(4,3),'head.0.weight':(256,916),'head.0.bias':(256,),'head.3.weight':(64,256),'head.3.bias':(64,),'head.6.weight':(1,64),'head.6.bias':(1,)}
    for field,n in [('z',768),('stat',16),('spec',128)]:
        for stat in ('mean','scale'):shapes[f'{field}_{stat}']=(n,)
    if set(payload['state_dict'])!=set(shapes):raise ValueError('scientific checkpoint keys')
    for k,shape in shapes.items():
        x=payload['state_dict'][k]
        if tuple(x.shape)!=shape or x.dtype!=torch.float32 or not torch.isfinite(x).all():raise ValueError('scientific checkpoint shape/dtype/finite '+k)
    if payload['base']!=anchor(task,fold) or payload['stats']!=read(P2/'scalers'/task/f'fold{fold}.json'):raise ValueError('scientific checkpoint BASE/scaler provenance')
    return payload

def qualify_ft(arm,task,fold):
    e=canonical_entry(OUT,arm,task,fold);folder=Path(e['folder']);h=read(folder/'history.json');met=read(folder/'metrics.json');u=read(folder/'consumption.json');split=read(SPLIT/f'{task}.json')['folds'][fold]
    counts=dict(epochs=70,updates=70*math.ceil(len(split['train_indices'])/256),train_forwards=70*math.ceil(len(split['train_indices'])/256),train_rows=70*len(split['train_indices']),validation=70,validation_rows=70*len(split['validation_indices']),validation_batches=70*math.ceil(len(split['validation_indices'])/256),epoch0_aliases=1,epoch0_alias_rows=len(split['validation_indices']))
    if len(h)!=70 or [x['epoch'] for x in h]!=list(range(1,71)) or met['status']!='COMPLETE' or u['status']!='COMPLETE' or u['attempted']!=u['completed'] or any(u['completed'].get(k)!=v for k,v in counts.items()):raise ValueError('FT 70epoch/history/consumption')
    saved={name:read_science(arm,task,fold,name) for name in ('epoch0','best','last')}
    if saved['epoch0']['epoch']!=0 or saved['last']['epoch']!=70 or met['best_epoch']!=saved['best']['epoch'] or met['validation']!=saved['best']['metrics']:raise ValueError('FT BEST/epoch0/last')
    if torch.count_nonzero(saved['epoch0']['state_dict']['head.6.weight']) or torch.count_nonzero(saved['epoch0']['state_dict']['head.6.bias']):raise ValueError('epoch0 output is nonzero')
    best=saved['epoch0']['metrics'];epoch=0
    for row in h:
        if better(row['validation'],best):best=row['validation'];epoch=row['epoch']
        if row['best_epoch']!=epoch:raise ValueError('history strict BEST earliest tie')
    if epoch!=met['best_epoch'] or best!=met['validation']:raise ValueError('BEST history selection')
    if e['source_kind']=='parent_complete_alias' and epoch!=e['expected_best_epoch']:raise ValueError('parent alias BEST epoch')
    return dict(status='COMPLETE',source_kind=e['source_kind'],folder=str(folder),best_epoch=epoch,consumption=u,scientific_state_strict=True,model_forwards=0)

def qualify_parent():
    import concurrent.futures as cf
    cpu_contract();check_clock(True)
    if SPEC!=read(ROOT/'configs/ua3d_hmr_r2_gpu2.json'):raise ValueError('science config changed')
    for parent in (P1,P2):
        for name in ('controller','outer','bootstrap'):
            if read(parent/f'{name}_exit.json')['exit']!=1:raise ValueError('parent terminal evidence')
    for name in read(P2/'authorization/source_lock.json')['files']:
        if (ROOT/name).read_bytes()!=(P2/'authorization/source_locked'/name).read_bytes():raise ValueError('parent scientific source changed')
    pb=read(P2/'budget.json');ft=[j for j in pb['jobs'] if j['kind']=='ft'];good=[j for j in ft if j['exit']==0];bad=[j for j in ft if j['exit']!=0]
    if len(good)!=2 or len(bad)!=2 or {(j['arm'],j['task'],j['fold']) for j in good}!=set(ALIASES):raise ValueError('parent 2 complete/2 stopped FT')
    if {(j['arm'],j['task'],j['fold'],j['exit']) for j in bad}!={('S_STAT','eea',1,-15),('S_STAT','eea',2,-15)} or any(Path(j['consumption']).exists() for j in bad):raise ValueError('unknown parent consumption interval contract')
    if pb['update_accounting']['total_updates']!=940 or pb['cumulative']['geometry_reserved']!=7300 or pb['cumulative']['geometry_attempted']!=4231:raise ValueError('parent cumulative budget')
    for j in pb['jobs']:
        if j['kind'] in ('prepare','diagnostic') or j in good:
            if j['exit']!=0 or read(Path(j['log'])/'wait_exit.json')['exit']!=0 or read(Path(j['log'])/'end.json')['exit']!=0:raise ValueError('parent successful job wait evidence')
    coverage=read(P2/'surface/coverage.json');rotation=read(P2/'surface/rotation.json')
    if coverage['overall']<.99 or any(v['rate']<.95 for v in coverage['tasks'].values()) or len(rotation)!=4 or any(v['area']>.05 or v['volume']>.05 or v['descriptor_relative_L2']>.1 for v in rotation):raise ValueError('parent prepare gates')
    if read(P2/'surface/complete.json')['unique_attempts']!=3650 or read(P2/'gates/cost_pilot.json')['passed']!=True:raise ValueError('parent prepare/pilot qualification')
    exclusive(OUT/'registry/anchors.json',read(P2/'registry/anchors.json'))
    entries={}
    for arm in m.TRAIN_ARMS:
        for task in TASKS:
            for fold in range(5):
                alias=(arm,task,fold) in ALIASES;entries[f'{arm}/{task}/{fold}']=dict(arm=arm,task=task,fold=fold,seed=42+fold,source_kind='parent_complete_alias' if alias else 'r3_new_ft',folder=str((P2 if alias else OUT)/'finetune'/arm/task/f'fold{fold}'),expected_best_epoch=ALIASES.get((arm,task,fold)))
    exclusive(OUT/'registry/canonical.json',dict(entries=entries,parent_root=str(P2),science_revision='r2',execution_revision='r3'))
    def qualify_cache(pair):
        task,fold=pair;c=load_cache(task,fold,False);e=read(SPLIT/f'{task}.json')['folds'][fold];ids=sorted(set(e['train_indices'])|set(e['validation_indices']));rows={x['row']:x for x in anchor_rows[task]};qualification=read(P2/'registry/cache_qualification/trainval'/task/f'fold{fold}.json')
        if qualification['status']!='COMPLETE' or qualification['formal'] or qualification['base']!=anchor(task,fold) or c['indices'].tolist()!=ids or c['keys']!=[rows[i]['key'] for i in ids] or c['scaler']!=anchor(task,fold)['scaler']:raise ValueError('parent compact cache identity')
        for name in ('z','availability','y','yb','y_original','yb_original'):
            if not torch.isfinite(c[name]).all():raise ValueError('parent compact cache nonfinite')
        stats=read(P2/'scalers'/task/f'fold{fold}.json')
        if stats['train_rows']!=e['train_indices']:raise ValueError('input train-only scaler provenance')
        return dict(task=task,fold=fold,source=str(cache_path(task,fold,False)),input_scaler=str(P2/'scalers'/task/f'fold{fold}.json'),status='COMPLETE',new_features=0)
    anchor_rows=read(OUT/'registry/anchors.json')['rows'];resources=read(OUT/'authorization/resources.json')
    with cf.ThreadPoolExecutor(max_workers=min(32,resources['C_eff'])) as pool:qualified=list(pool.map(qualify_cache,[(t,f) for t in TASKS for f in range(5)]))
    surface_records=list((P2/'surface/structures').glob('*/status.json'))
    if len(surface_records)!=3650:raise ValueError('parent surface inventory')
    def check_surface(p):
        s=read(p)
        if s['available']:
            a=s['audit']
            if a['zero_modes']!=a['components'] or a['retained_modes']>64 or a['vertices']>16384 or a['residual']>1e-5 or a['mass_orthogonality']>1e-5 or not (p.parent/'audit.npz').exists():raise ValueError('parent component/spectral audit')
        elif s['reason'] not in d.REASONS:raise ValueError('parent unavailable reason')
    with cf.ThreadPoolExecutor(max_workers=min(32,resources['C_eff'])) as pool:list(pool.map(check_surface,surface_records))
    diag=[j for j in pb['jobs'] if j['kind']=='diagnostic']
    if len(diag)!=12 or any(j['exit']!=0 for j in diag):raise ValueError('parent 12 diagnostics')
    for arm in m.TRAIN_ARMS:
        for task in ('eps','egc'):
            name='S' if task=='eps' else 'L';x=torch.load(P2/'diagnostics'/name/arm/'R0/trajectory.pt',map_location='cpu',weights_only=False);y=torch.load(P2/'diagnostics'/name/arm/'R1/trajectory.pt',map_location='cpu',weights_only=False)
            if x['history']!=y['history'] or any(not torch.equal(v,y['state'][k]) for k,v in x['state'].items()):raise ValueError('parent same-seed trajectory')
    aliases={f'{a}/{t}/{f}':qualify_ft(a,t,f) for a,t,f in ALIASES}
    exclusive(OUT/'registry/parent_qualification.json',dict(status='COMPLETE',time=time.time(),cache=qualified,aliases=aliases,surface_source=str(P2/'surface'),input_scaler_source=str(P2/'scalers'),diagnostic_source=str(P2/'diagnostics'),pilot_source=str(P2/'gates/cost_pilot.json'),new_geometry=0,new_pack=0,new_diagnostics=0,new_pilot=0,old_unknown_updates=[0,140]))

def validate_job(j):
    log=Path(j['log']);process=read(log/'process.json');end=read(log/'end.json')
    if end['exit']!=0 or end['started_unix']!=process['started_unix'] or process['cwd']!=str(ROOT) or process['CVD']!=('2' if j['gpu'] else ''):raise ValueError('worker process/end contract')
    if j['kind']=='ft':qualify_ft(j['arm'],j['task'],j['fold'])
    else:
        u=read(Path(j['consumption']))
        if u['status']!='COMPLETE' or u['attempted']!=u['completed']:raise ValueError('worker consumption incomplete')
        if j['kind']=='test':
            folder=job_folder('test',j['arm'],j['task'],j['fold']);met=read(folder/'metrics.json')
            with np.load(folder/'test_predictions.npz',allow_pickle=False) as p:
                if metrics(p['y_true'],p['y_pred'])!=met['metrics'] or len(p['indices'])!=met['rows']:raise ValueError('formal output artifact metrics')
        elif j['kind']=='preflight':
            if any(u['attempted'].get(k,0) for k in ('model_constructs','forwards','backwards','optimizer_updates','real_labels','geometry','GPU_calls')):raise ValueError('pure CPU regression exceeded contract')
        elif j['kind']=='cpu':
            if read(OUT/'final_cpu/audit.json')['status']!='PASSED':raise ValueError('final CPU audit missing')

class Controller:
    def __init__(self):
        self.jobs=[];self.phase='INITIALIZING';self.engine=Lifecycle(validate_job,self.refresh);self.refresh()
    def refresh(self):
        parsed=parse_jobs(self.jobs);bykind={}
        for j in self.jobs:
            p=Path(j['consumption'])
            if p.exists():
                u=read(p);entry=bykind.setdefault(j['kind'],dict(attempted={},completed={}))
                for field in ('attempted','completed'):
                    for k,v in u[field].items():entry[field][k]=entry[field].get(k,0)+v
        updates=bykind.get('ft',{}).get('attempted',{}).get('updates',0);ledger=dict(jobs=self.jobs,**parsed,usage=bykind,cumulative=cumulative(self.jobs,updates),phase=self.phase,limits=EXEC['limits'],aliases=2,new_science=dict(geometry=0,diagnostics=0,trainval_pack=0,scaler_fit=0,BASE_forward=0,PT=0),lifecycle_events=self.engine.events if hasattr(self,'engine') else [])
        caps={'preflight':1,'ft':103,'test':140,'cpu':1}
        for kind,limit in caps.items():
            if parsed['counts'].get(kind,{}).get('reserved',0)>limit:raise ValueError('job budget '+kind)
        for k,limit in EXEC['limits'].items():
            kind='ft' if k.startswith('ft_') or k in ('validation','validation_batches','validation_rows','epoch0_aliases','epoch0_alias_rows') else 'test'
            field=k.removeprefix('ft_') if k.startswith('ft_') else k
            if kind=='ft' and field=='epochs':field='epochs'
            if k=='ft_train_rows':field='train_rows'
            if k=='ft_updates':field='updates'
            if bykind.get(kind,{}).get('attempted',{}).get(field,0)>limit:raise ValueError('usage cap '+k)
        ledger['new_bytes']=size(OUT)+size(LOG)
        if ledger['new_bytes']>8*1024**3:raise ValueError('r3 artifacts >8GiB')
        write(OUT/'budget.json',ledger);write(OUT/'cumulative_budget.json',ledger['cumulative']);write(OUT/'lifecycle/events.json',self.engine.events if hasattr(self,'engine') else [])
        write(OUT/'execution_manifest.json',dict(status='RUNNING',phase=self.phase,stages={k:state_for([j for j in self.jobs if j['kind']==k],v) for k,v in caps.items()},science_revision='r2',execution_revision='r3',independent_review='PENDING'))
        return ledger
    def dispatch(self,kind,arm='',task='',fold=0):
        self.engine.poll_all()
        if self.engine.pending():return False
        check_clock(kind=='preflight');key='/'.join(map(str,(kind,arm,task,fold)))
        if any(j['id']==key for j in self.jobs):raise ValueError('duplicate dispatch')
        if kind=='ft' and canonical_entry(OUT,arm,task,fold)['source_kind']!='r3_new_ft':raise ValueError('alias cannot be dispatched')
        gpu=kind=='ft' or kind=='test' and arm!='C_ANCHOR';log=LOG/'jobs'/key;log.mkdir(parents=True,exist_ok=False);folder=job_folder(kind,arm,task,fold);cmd=[str(PYTHON),str(ROOT/'scripts/run_ua3d_hmr_r3.py'),'worker','--kind',kind,'--arm',arm,'--task',task,'--fold',str(fold),'--log',str(log)]
        j=dict(id=key,kind=kind,arm=arm,task=task,fold=fold,gpu=gpu,reserved_unix=time.time(),started_unix=None,ended_unix=None,exit=None,qualified=False,argv=cmd,cwd=str(ROOT),log=str(log),consumption=str(folder/'consumption.json'),ownership_tokens=['run_ua3d_hmr_r3.py','worker','--kind '+kind])
        self.jobs.append(j);self.refresh();env=dict(os.environ,**THREADS);env['CUDA_VISIBLE_DEVICES']='2' if gpu else '';self.engine.spawn(j,env);write(log/'dispatch.json',dict(job=j,env={**THREADS,'CUDA_VISIBLE_DEVICES':env['CUDA_VISIBLE_DEVICES']}));return True
    def wait_all(self,prepare=False):
        while self.engine.active:
            check_clock(prepare);self.engine.poll_all()
            for p,j,s in self.engine.active.values():
                if j['kind'] in ('preflight','cpu') and time.time()-j['started_unix']>1800:raise TimeoutError('CPU suite timeout')
            time.sleep(.1)
    def one(self,kind,**kw):
        self.phase=kind
        while not self.dispatch(kind,**kw):check_clock(kind=='preflight');time.sleep(.1)
        self.wait_all(kind=='preflight')
    def admission(self):
        snapshots=self.engine.snapshots()
        if self.engine.pending():return False
        known=set()
        for x in snapshots:known.update([x['identity']['pid'],*x['nspid']])
        query=subprocess.run(['nvidia-smi','-i','2','--query-compute-apps=pid,used_memory,gpu_uuid','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True).stdout;unknown=[]
        for line in query.strip().splitlines():
            fields=[x.strip() for x in line.split(',')]
            if fields[-1]!=SPEC['uuid']:raise ValueError('GPU2 UUID changed')
            if int(fields[0]) not in known:unknown.append(line)
        write(OUT/'lifecycle/gpu_ownership.json',dict(time=time.time(),snapshots=snapshots,nvml=query,unknown=unknown,policy='namespace difference unresolved means wait, no guessed mapping or signal'))
        if unknown:return False
        free=float(subprocess.run(['nvidia-smi','-i','2','--query-gpu=memory.free','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True).stdout.strip());peaks=[]
        for j in self.jobs:
            p=job_folder(j['kind'],j['arm'],j['task'],j['fold'])/'gpu.json'
            if j['kind']==self.phase and p.exists():peaks.append(read(p)['peak_memory_mib'])
        return free>=max(4096,max(peaks,default=0)+1024)+3072
    def gate(self,label,prepare=False):
        complete=[j for j in self.jobs if j['kind']=='ft' and j['qualified']];test_done=sum(j['kind']=='test' and j['qualified'] for j in self.jobs);remaining=103-len(complete);parent_pilot=next(j for j in read(P2/'budget.json')['jobs'] if j['kind']=='ft' and j['arm']=='S_HMR');per=parent_pilot['ended_unix']-parent_pilot['started_unix'];routes=2
        if complete:
            per=max([per]+[j['ended_unix']-j['started_unix'] for j in complete]);union=parse_jobs(complete)['physical_gpu_union_seconds'];work=sum(j['ended_unix']-j['started_unix'] for j in complete);routes=min(8,max(1,work/max(union,1e-12)))
        eta=1.5*(per*(remaining+140-test_done)/routes+1800+2400);clock=read(OUT/'authorization/clock.json');passed=time.time()+eta<clock['deadline_unix'] and (not prepare or time.time()<clock['prepare_deadline_unix']);write(OUT/'gates'/f'{label}.json',dict(passed=passed,remaining_new_FT=remaining,remaining_outputs=140-test_done,effective_routes=routes,wrapper_bound_seconds=per,conservative_seconds=eta,deadline_unix=clock['deadline_unix'],observed_unix=time.time()))
        if not passed:raise TimeoutError('full remaining cost ETA gate')
    def parallel(self,kind,units):
        self.phase=kind;todo=list(units)
        while todo or self.engine.active:
            check_clock();self.engine.poll_all();done=sum(j['kind']==kind and j['qualified'] for j in self.jobs);cap=2 if done==0 else 4 if done<4 else 8
            while todo and len(self.engine.active)<cap:
                if not self.admission():break
                a,t,f=todo[0]
                if not self.dispatch(kind,arm=a,task=t,fold=f):break
                todo.pop(0)
            self.gate('progress');time.sleep(.1)
    def run(self):
        try:
            self.one('preflight');self.phase='qualification';qualify_parent();self.gate('initial',True)
            new=[(a,t,f) for a in m.TRAIN_ARMS for t in TASKS for f in range(5) if (a,t,f) not in ALIASES];priority=[('S_STAT','eea',1),('S_STAT','eea',2)];new=priority+[u for u in new if u not in priority];self.parallel('ft',new)
            for arm in m.TRAIN_ARMS:
                for task in TASKS:
                    for fold in range(5):qualify_ft(arm,task,fold)
            self.phase='freeze';freeze();self.refresh();self.phase='cache_test'
            for task in TASKS:
                for fold in range(5):pack_cache(task,fold,True)
            for task in TASKS:
                for fold in range(5):self.one('test',arm='C_ANCHOR',task=task,fold=fold)
            anchor_formal=aggregate(True)
            if abs(anchor_formal['C_ANCHOR']['macro']-REFERENCE)>1e-6:raise ValueError('BASE anchor macro gate')
            self.parallel('test',[(a,t,f) for a in m.TRAIN_ARMS for t in TASKS for f in range(5)]);self.refresh();write(OUT/'final_review/pre_cpu_results.json',scientific_results());self.one('cpu');self.refresh();write(OUT/'controller_return.json',dict(exit=0,time=time.time()));return 0
        except BaseException:
            first=traceback.format_exc();fail(self.phase,first);errors=self.engine.stop();write(OUT/'stop_evidence.json',dict(errors=errors,active=list(self.engine.active),time=time.time()));self.refresh();fail(self.phase,first);write(OUT/'controller_return.json',dict(exit=1,time=time.time(),phase=self.phase));return 1

def scientific_results():
    formal=aggregate(True);full=len(formal)==4;frozen=read(OUT/'selection/final_frozen.json') if (OUT/'selection/final_frozen.json').exists() else None;selected=frozen['FINAL'] if frozen else None;macro=REFERENCE if selected=='BASE_KEEP' else formal[selected]['macro'] if full and selected else None;near=bool(full and selected!='BASE_KEEP' and macro>=.809);target=bool(full and selected!='BASE_KEEP' and macro>=.810)
    return dict(reference=REFERENCE,FINAL=selected,formal=formal,validation=frozen['validation'] if frozen else None,formal_complete=full,selected_macro=macro,selected_delta=None if macro is None else macro-REFERENCE,near_goal_met=near,target_met=target,performance_pass=near,independent_review='PENDING',spectrum_increment={f'{a}-{b}':formal[a]['macro']-formal[b]['macro'] for a,b in [('S_HMR','S_FIXED'),('S_FIXED','S_STAT'),('S_HMR','S_STAT')]} if full else None)
