"""Isolated HMR r1 execution, no BASE model import or forward path."""
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
from scripts.bootstrap_ua3d_hmr_r1 import ROOT,OUT,LOG,SOURCES,THREADS,read,write,exclusive,identity,parse_jobs,state_for,size,check_clock,fail
from src.dataset import ua3d_hmr_r1 as d
from src.modules import ua3d_hmr_r1 as m
BASE=ROOT/'results/ua3d_base_opt_20261026/r6/repairs/concurrency_r1'
PARENT=ROOT/'results/ua3d_anchor_opt_20261102/r2/feature_cache'
SIDE=ROOT/'results/ua3d_ph_20261028/r1/repairs/gpu2_r2/inputs'
INPUT=ROOT/'results/ua3m_paperfuse_20261025/r1/repairs/gpu0123_restart1/inputs'
DOWN=ROOT/'data/processed/mcl_ph_r4_20260930/downstream_direct_01.staging'
SPLIT=ROOT/'data/splits/mips_paper5_outer'
TASKS=('eea','egb','egc','ei','eps','nc','xc')
REFERENCE=.804675462276949
SPEC=read(ROOT/'configs/ua3d_hmr_r1_gpu2.json')
SCHEMA='ua3d-hmr-residual-r1'
PYTHON=OUT/'env/bin/python'

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
    if formal and not (OUT/'selection/final_frozen.json').exists():raise ValueError('TEST reader before freeze')
    return OUT/'feature_cache'/('test' if formal else 'trainval')/task/f'fold{fold}.pt'
def load_cache(task,fold,formal):
    return torch.load(cache_path(task,fold,formal),map_location='cpu',weights_only=False)
def partition_positions(cache,part,task,fold):
    e=read(SPLIT/f'{task}.json')['folds'][fold];ids=sorted(e['test_indices']) if part=='test' else e[part+'_indices'];pos={row:i for i,row in enumerate(cache['indices'].tolist())};return [pos[row] for row in ids]

def register():
    cpu_contract();result=read(BASE/'final_review/results.json')
    if result['formal']['F_PD_N1_F0']['macro']!=REFERENCE or result['FINAL']!='E_P_SYM_SHARED':raise ValueError('original reference')
    counts=dict(train=0,validation=0,test=0,train_batches=0,validation_batches=0,test_batches=0);anchors={};rows=read(INPUT/'rows.json')
    recipe=read(BASE/'registry/recipes.json')['recipes']['F_PD_N1_F0']
    for task in TASKS:
        split=read(SPLIT/f'{task}.json')
        if len(split['folds'])!=5 or [x['row'] for x in rows[task]]!=list(range(len(rows[task]))):raise ValueError('row/fold identity')
        for fold,e in enumerate(split['folds']):
            sets=[set(e[k+'_indices']) for k in ('train','validation','test')]
            if any(sets[i]&sets[j] for i in range(3) for j in range(i)) or set.union(*sets)!=set(range(len(rows[task]))):raise ValueError('split partition')
            for role in ('train','validation','test'):
                n=len(e[role+'_indices']);counts[role]+=n;counts[role+'_batches']+=math.ceil(n/256)
            folder=base_folder(task,fold);cfg=read(folder/'config.json');met=read(folder/'metrics.json');scaler=read(folder/'scaler.json')
            if cfg['recipe']!=recipe or met['config']!=cfg or cfg['seed']!=42+fold or cfg['task']!=task or cfg['fold']!=fold or met['status']!='COMPLETE' or met['scaler']!=scaler:raise ValueError('BASE config/scaler identity')
            if scaler.get('fit')!='train' or scaler['scale'][0]<=0:raise ValueError('target scaler')
            saved=torch.load(folder/'best.pt',map_location='cpu',weights_only=False)
            meta=read(PARENT/'trainval'/task/f'fold{fold}'/'metadata.json')
            if any(saved[k]!=v for k,v in dict(schema=meta['source_schema'],config=cfg,best_epoch=met['best_epoch'],best_stage=met['best_stage'],scaler=scaler).items()):raise ValueError('BASE checkpoint metadata')
            if any(not torch.isfinite(x).all() for x in saved['state_dict'].values() if x.is_floating_point()):raise FloatingPointError('BASE state')
            del saved
            anchors[f'{task}/{fold}']=dict(best=str(folder/'best.pt'),source_config=cfg,schema=meta['source_schema'],best_epoch=met['best_epoch'],best_stage=met['best_stage'],scaler=scaler,validation_predictions=str(folder/'validation_predictions.npz'),test_path_exists=(PARENT/'test'/task/f'fold{fold}'/'metadata.json').exists())
    if counts!=dict(train=18786,validation=4714,test=5875,train_batches=85,validation_batches=45,test_batches=45):raise ValueError('split budget counts')
    exclusive(OUT/'registry/anchors.json',dict(anchors=anchors,counts=counts,rows={task:[{k:r[k] for k in ('row','key','available')} for r in rows[task]] for task in TASKS},reference=REFERENCE))
    exclusive(OUT/'registry/matrix.json',dict(estimators=[dict(arm=a,task=t,fold=f,seed=42+f) for a in m.ARMS for t in TASKS for f in range(5)],limits=SPEC['limits']))

def pack_cache(task,fold,formal):
    import threading
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
    c=load_cache(task,fold,formal);s=d.load_surface(OUT,task);rows=c['indices'];b={k:s[k][rows].to(device) for k in s};b.update({k:c[k].to(device) for k in ('z','availability','y','yb')});return c,b

def slices(b,ids):return {k:v[ids] for k,v in b.items()}
def save_model(path,model,arm,task,fold,epoch,met,stats):
    if path.exists():path.unlink()  # Only this unit's own evolving BEST/last, never old sources.
    torch.save(dict(schema=SCHEMA,arm=arm,task=task,fold=fold,seed=42+fold,state_dict=model.state_dict(),epoch=epoch,metrics=met,stats=stats,base=anchor(task,fold),config=SPEC),path)
def assert_finite_model(model):
    if any(not torch.isfinite(p).all() for p in model.parameters()):raise FloatingPointError('nonfinite model')
def fit_job(kind,arm,task,fold,repeat=''):
    gpu=gpu_contract();seed(42+fold);device='cuda:0';c,b=batch_data(task,fold,False,device);stats=read(OUT/'scalers'/task/f'fold{fold}.json');model=m.Residual(arm,stats).to(device);train=partition_positions(c,'train',task,fold);val=partition_positions(c,'validation',task,fold);folder=job_folder(kind,arm,task,fold,repeat);folder.mkdir(parents=True,exist_ok=False)
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.0003,weight_decay=.0001,betas=(.9,.999),eps=1e-8);sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=70,eta_min=1e-6)
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
            if met['rmse']<best['rmse']:best=met;best_epoch=epoch;save_model(folder/'best.pt',model,arm,task,fold,epoch,best,stats)
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
        gpu=gpu_contract();seed(42+fold);c,b=batch_data(task,fold,True,'cuda:0');saved=torch.load(job_folder('ft',arm,task,fold)/'best.pt',map_location='cpu',weights_only=False);frozen=read(OUT/'selection/final_frozen.json')['estimators'][f'{arm}/{task}/{fold}']
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
                    p=job_folder('ft',arm,task,fold)/'metrics.json'
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
                    p=job_folder('ft',arm,task,fold)/'best.pt';s=torch.load(p,map_location='cpu',weights_only=False)
                    estimators[key]=dict(path=str(p),epoch=s['epoch'],base=a,stats=s['stats'],scaler=a['scaler'],config=SPEC,source_cache=str(cache_path(task,fold,False)),surface_mapping=str(OUT/'surface/mapping.json'),input_rows=read(OUT/'registry/anchors.json')['rows'][task])
    exclusive(OUT/'selection/final_frozen.json',dict(frozen=True,frozen_unix=time.time(),FINAL=selected,validation=validation,estimators=estimators,reference=REFERENCE,target_version='near809-met810',surface_config=SPEC['surface'],shared_development_test=True));return selected

class Controller:
    def __init__(self,root=OUT,log=LOG,runner=subprocess.Popen):
        self.root=Path(root);self.log=Path(log);self.runner=runner;self.jobs=[];self.active={};self.phase='INITIALIZING';self.root.mkdir(parents=True,exist_ok=True);self.log.mkdir(parents=True,exist_ok=True)
        if (self.root/'budget.json').exists():raise ValueError('existing ledger: no resume')
        self.refresh()
    def refresh(self):
        parsed=parse_jobs(self.jobs);attempted={};completed={}
        for j in self.jobs:
            usage=Path(j['consumption'])
            if usage.exists():
                u=read(usage)
                for target,source in ((attempted,u['attempted']),(completed,u['completed'])):
                    for k,v in source.items():target[k]=target.get(k,0)+v
        budget=dict(jobs=self.jobs,**parsed,attempted=attempted,completed=completed,limits=SPEC['limits'],tN=read(self.root/'authorization/clock.json')['tN'] if (self.root/'authorization/clock.json').exists() else None,encoder_forwards=0,freeze_reserved=int((self.root/'selection/freeze_reservation.json').exists()),freeze_completed=int((self.root/'selection/final_frozen.json').exists()))
        mapping={'diagnostic_updates':sum(read(Path(j['consumption']))['attempted'].get('updates',0) for j in self.jobs if j['kind']=='diagnostic' and Path(j['consumption']).exists()),'ft_updates':sum(read(Path(j['consumption']))['attempted'].get('updates',0) for j in self.jobs if j['kind']=='ft' and Path(j['consumption']).exists())}
        budget['update_accounting']=dict(mapping,total_updates=sum(mapping.values()))
        if mapping['diagnostic_updates']>240 or mapping['ft_updates']>17850 or sum(mapping.values())>18090:raise ValueError('optimizer budget exceeded')
        for kind,limit in [('preflight',1),('prepare',1),('diagnostic',12),('ft',105),('test',140),('cpu',1)]:
            if parsed['counts'].get(kind,{}).get('reserved',0)>limit:raise ValueError('job cap '+kind)
        surface=self.root/'surface/structures'
        budget['surface_accounting']=dict(reserved=len(read(self.root/'surface/queue.json')) if (self.root/'surface/queue.json').exists() else 0,started=len(list(surface.glob('*/process.json'))),attempted=len(list(surface.glob('*/process.json'))),completed=sum(read(p)['exit']==0 for p in surface.glob('*/end.json')),source_rows=5875)
        budget['cache_accounting']={role:dict(completed_units=len(list((self.root/'registry/cache_qualification'/role).glob('*/*.json'))),bytes=size(self.root/'feature_cache'/role)) for role in ('trainval','test')}
        budget['artifact_bytes']=dict(surface=size(self.root/'surface'),teacher=size(self.root/'feature_cache'),total=size(self.root)+size(self.log))
        if budget['artifact_bytes']['surface']>32*1024**3 or budget['artifact_bytes']['teacher']>1024**3 or budget['artifact_bytes']['total']>48*1024**3:raise ValueError('artifact cap')
        write(self.root/'budget.json',budget);write(self.root/'execution_manifest.json',dict(status='RUNNING',phase=self.phase,stages={kind:state_for([j for j in self.jobs if j['kind']==kind],n) for kind,n in [('preflight',1),('prepare',1),('diagnostic',12),('ft',105),('test',140),('cpu',1)]},independent_review='PENDING',execution_complete=False))
        return budget
    def dispatch(self,kind,arm='',task='',fold=0,repeat='',gpu=False):
        key='/'.join(map(str,(kind,arm,task,fold,repeat)))
        if any(j['id']==key for j in self.jobs):raise ValueError('duplicate reservation')
        log=self.log/'jobs'/key;log.mkdir(parents=True,exist_ok=False);folder=job_folder(kind,arm,task,fold,repeat)
        cmd=[str(PYTHON),str(ROOT/'scripts/run_ua3d_hmr_r1.py'),'worker','--kind',kind,'--arm',arm,'--task',task,'--fold',str(fold),'--repeat',repeat,'--log',str(log)]
        j=dict(id=key,kind=kind,arm=arm,task=task,fold=fold,repeat=repeat,gpu=gpu,reserved_unix=time.time(),started_unix=None,ended_unix=None,exit=None,argv=cmd,cwd=str(ROOT),log=str(log),consumption=str(folder/'consumption.json'),process=None)
        self.jobs.append(j);self.refresh();env=dict(os.environ,**THREADS);env['CUDA_VISIBLE_DEVICES']='2' if gpu else '';stream=(log/'stdout.log').open('x');p=self.runner(cmd,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True);j['started_unix']=time.time();j['process']=identity(p.pid);self.active[key]=(p,j,stream);write(log/'dispatch.json',dict(**j,env={**THREADS,'CUDA_VISIBLE_DEVICES':env['CUDA_VISIBLE_DEVICES']}));self.refresh()
        return j
    def poll(self):
        for key,(p,j,f) in list(self.active.items()):
            rc=p.poll()
            if rc is None:
                limit=1800 if j['kind'] in ('preflight','cpu') or (j['kind']=='ft' and j['arm']=='S_HMR' and j['task']=='egc' and j['fold']==1) else None
                if limit and time.time()-j['started_unix']>limit:raise TimeoutError('job timeout '+key)
                continue
            rc=p.wait();f.close();j['exit']=rc;j['ended_unix']=time.time();exclusive(Path(j['log'])/'wait_exit.json',dict(exit=rc,started_unix=j['started_unix'],ended_unix=j['ended_unix']));del self.active[key];self.refresh()
            if rc:raise RuntimeError('scientific job failed '+key+' exit='+str(rc))
    def stop_active(self):
        import signal
        for p,j,f in list(self.active.values()):
            try:
                if identity(p.pid)==j['process']:os.killpg(p.pid,signal.SIGTERM)
            except ProcessLookupError:pass
        until=time.time()+10
        while self.active and time.time()<until:
            for key,(p,j,f) in list(self.active.items()):
                if p.poll() is not None:j['exit']=p.wait();j['ended_unix']=time.time();f.close();write(Path(j['log'])/'wait_exit.json',dict(exit=j['exit'],ended_unix=j['ended_unix'],stopped_by_controller=True));del self.active[key]
            if self.active:time.sleep(.2)
        for key,(p,j,f) in list(self.active.items()):
            if p.poll() is None:
                try:
                    if identity(p.pid)==j['process']:os.killpg(p.pid,signal.SIGKILL)
                except ProcessLookupError:pass
            j['exit']=p.wait();j['ended_unix']=time.time();f.close();write(Path(j['log'])/'wait_exit.json',dict(exit=j['exit'],ended_unix=j['ended_unix'],forced=True));del self.active[key]
        self.refresh()
    def wait_all(self,preparation=False):
        while self.active:check_clock(preparation);self.poll();time.sleep(.25)
    def one(self,kind,**kw):self.phase=kind;self.dispatch(kind,**kw);self.wait_all(kind in ('preflight','prepare','diagnostic'))
    def gpu_admission(self):
        query=subprocess.run(['nvidia-smi','-i','2','--query-compute-apps=pid,used_memory,gpu_uuid','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True).stdout
        known={j['process']['pid'] for p,j,f in self.active.values()};unknown=[]
        for line in query.strip().splitlines():
            fields=[x.strip() for x in line.split(',')]
            if not fields or not fields[0]:continue
            if int(fields[0]) not in known:unknown.append(line)
            elif fields[-1]!=SPEC['uuid']:raise ValueError('GPU PID UUID mismatch')
        if unknown:write(OUT/'gates/unknown_gpu_occupancy.json',dict(time=time.time(),lines=unknown));return False
        free=float(subprocess.run(['nvidia-smi','-i','2','--query-gpu=memory.free','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True).stdout.strip())
        peaks=[]
        for j in self.jobs:
            p=job_folder(j['kind'],j['arm'],j['task'],j['fold'],j['repeat'])/'gpu.json'
            if j['kind']==self.phase and p.exists():peaks.append(read(p)['peak_memory_mib'])
        need=max(4096,max(peaks,default=0)+1024)
        return free>=need+3072
    def parallel(self,kind,units):
        self.phase=kind;todo=list(units)
        while todo or self.active:
            check_clock();self.poll();done=sum(j['kind']==kind and j['exit']==0 for j in self.jobs);cap=8 if done else 4
            while todo and len(self.active)<cap:
                if not self.gpu_admission():break
                arm,task,fold=todo.pop(0);self.dispatch(kind,arm=arm,task=task,fold=fold,gpu=True)
            time.sleep(.3)
    def gate(self,label):
        clock=read(OUT/'authorization/clock.json');ft=[j for j in self.jobs if j['kind']=='ft' and j['exit']==0];diags=[j for j in self.jobs if j['kind']=='diagnostic' and j['exit']==0];remaining=105-len(ft)
        if ft:
            # Initially at most two routes. Real overlapping completed jobs can justify measured concurrency.
            intervals=[(j['started_unix'],j['ended_unix']) for j in ft];busy=parse_jobs(ft)['physical_gpu_union_seconds'];work=sum(b-a for a,b in intervals);routes=min(8,work/max(busy,1e-12)) if len(ft)>1 else 1;routes=min(routes,2) if len(ft)<4 else routes
            per=max(j['ended_unix']-j['started_unix'] for j in ft);gpu=per*remaining/max(routes,1)+per*140/max(routes,1)
        else:
            if len(diags)!=12:raise ValueError('initial cost gate requires all diagnostics')
            per=max(j['ended_unix']-j['started_unix'] for j in diags);routes=2;gpu=per/20*17850/routes+per*140/routes
        eta=1.5*(gpu+1800+2400);passed=time.time()+eta<clock['deadline_unix'];write(OUT/'gates'/f'cost_{label}.json',dict(passed=passed,remaining_ft=remaining,estimated_effective_routes=routes,conservative_remaining_seconds=eta,deadline_unix=clock['deadline_unix'],observed_unix=time.time(),estimator='diagnostic update bound then max observed normal FT wrapper; no eight-route assumption'))
        if not passed:raise TimeoutError('cost ETA gate')
    def run(self):
        try:
            register();self.one('preflight');self.one('prepare')
            for arm in m.TRAIN_ARMS:
                for task,fold in [('eps',0),('egc',1)]:
                    for repeat in ('R0','R1'):
                        self.phase='diagnostic'
                        while not self.gpu_admission():check_clock(True);time.sleep(1)
                        self.one('diagnostic',arm=arm,task=task,fold=fold,repeat=repeat,gpu=True)
                    x=torch.load(job_folder('diagnostic',arm,task,fold,'R0')/'trajectory.pt',weights_only=False,map_location='cpu');y=torch.load(job_folder('diagnostic',arm,task,fold,'R1')/'trajectory.pt',weights_only=False,map_location='cpu')
                    if x['history']!=y['history'] or any(not torch.equal(x['state'][k],y['state'][k]) for k in x['state']):raise ValueError('same-seed diagnostic trajectory mismatch')
            for task,fold in [('eps',0),('egc',1)]:
                common=None
                for arm in m.TRAIN_ARMS:
                    x=torch.load(job_folder('diagnostic',arm,task,fold,'R0')/'trajectory.pt',weights_only=False,map_location='cpu')['initial'];head={k:v for k,v in x.items() if k.startswith('head.')}
                    if common is not None and any(not torch.equal(v,common[k]) for k,v in head.items()):raise ValueError('shared head initialization mismatch')
                    common=head
            check_clock(True);self.gate('diagnostics');self.phase='ft'
            while not self.gpu_admission():check_clock();time.sleep(1)
            self.one('ft',arm='S_HMR',task='egc',fold=1,gpu=True);self.gate('pilot')
            units=[(a,t,f) for a in m.TRAIN_ARMS for t in TASKS for f in range(5) if (a,t,f)!=('S_HMR','egc',1)]
            self.parallel('ft',units);self.gate('all_ft');freeze();self.refresh();self.phase='cache_test'
            for task in TASKS:
                for fold in range(5):pack_cache(task,fold,True)
            for task in TASKS:
                for fold in range(5):self.one('test',arm='C_ANCHOR',task=task,fold=fold)
            formal=aggregate(True)
            if 'C_ANCHOR' not in formal or abs(formal['C_ANCHOR']['macro']-REFERENCE)>1e-6:raise ValueError('anchor macro gate')
            self.parallel('test',[(a,t,f) for a in m.TRAIN_ARMS for t in TASKS for f in range(5)]);self.refresh();write(OUT/'final_review/pre_cpu_results.json',scientific_results());self.one('cpu');self.refresh();write(OUT/'controller_exit.json',dict(exit=0,ended_unix=time.time()));return 0
        except BaseException:
            err=traceback.format_exc();fail(self.phase,err)
            try:self.stop_active()
            except BaseException:write(OUT/'cleanup_error.json',dict(error=traceback.format_exc()))
            # refresh writes RUNNING; preserve stopped state after consumption has been merged.
            fail(self.phase,err);write(OUT/'controller_exit.json',dict(exit=1,ended_unix=time.time(),phase=self.phase));return 1

def scientific_results():
    formal=aggregate(True);full=len(formal)==4;frozen=read(OUT/'selection/final_frozen.json') if (OUT/'selection/final_frozen.json').exists() else None;selected=frozen['FINAL'] if frozen else None
    macro=REFERENCE if selected=='BASE_KEEP' else formal[selected]['macro'] if full and selected else None;near=bool(full and selected!='BASE_KEEP' and macro>=.809);target=bool(full and selected!='BASE_KEEP' and macro>=.810)
    return dict(reference=REFERENCE,FINAL=selected,formal=formal,validation=frozen['validation'] if frozen else None,formal_complete=full,metrics_status='正式数值保留' if full else '未完成/不可比较',selected_macro=macro,selected_delta=None if macro is None else macro-REFERENCE,near_goal_met=near,target_met=target,performance_pass=near,independent_review='PENDING',spectrum_increment={f'{a}-{b}':formal[a]['macro']-formal[b]['macro'] for a,b in [('S_HMR','S_FIXED'),('S_FIXED','S_STAT'),('S_HMR','S_STAT')]} if full else None,limitations=['共享开发TEST，非独立盲测','single seed=42+fold','finite trimer van der Waals surface, not periodic bulk','Egc3380 vs paper4124; Xc432 vs paper430; numerical comparison only'])

def deliver(error):
    result=scientific_results();budget=read(OUT/'budget.json') if (OUT/'budget.json').exists() else {};jobs=budget.get('jobs',[]);cpu=state_for([j for j in jobs if j['kind']=='cpu'],1);pre=state_for([j for j in jobs if j['kind']=='preflight'],1);result.update(execution_complete=error is None and cpu['status']=='PASSED',final_cpu_pass=cpu['status']=='PASSED',cpu=cpu,preflight=pre,error=error,controller_exit=read(OUT/'controller_exit.json') if (OUT/'controller_exit.json').exists() else None,outer_exit=read(OUT/'outer_exit.json') if (OUT/'outer_exit.json').exists() else None,budget=str(OUT/'budget.json'))
    if result['formal_complete'] and error:result['metrics_status']='正式数值保留、执行失败'
    write(OUT/'final_review/results.json',result)
    sota={'eea':(.934,.013),'egb':(.923,.009),'egc':(.911,.008),'ei':(.839,.018),'eps':(.787,.033),'nc':(.856,.032),'xc':(.423,.102)}
    lines=['# UA3D-HMR-20261103-01/r1 交付','',f"状态：{'执行完成，待独立审查' if result['execution_complete'] else 'STOPPED'}；{result['metrics_status']}；independent_review=PENDING。",f"reference={REFERENCE!r}；FINAL={result['FINAL']}；macro={result['selected_macro']}；Δ={result['selected_delta']}。",f"near={result['near_goal_met']}；target={result['target_met']}；performance_pass={result['performance_pass']}；final_cpu={cpu['status']}。",'']
    if result['formal_complete']:
        for arm in m.ARMS:
            lines.extend([f'## {arm}', '', '|Task|项目 mean ± std|论文 mean ± std|Δ项目−论文|论文模型|','|---|---:|---:|---:|---|'])
            for task in TASKS:
                v=result['formal'][arm]['tasks'][task];mean,std=sota[task];model='Periodic-TDL (+Chem, +RC)' if task in ('egb','egc') else 'Periodic-TDL';lines.append(f"|{task}|{v['mean']:.3f} ± {v['std']:.3f}|{mean:.3f} ± {std:.3f}|{v['mean']-mean:+.3f}|{model}|")
            macro=result['formal'][arm]['macro'];lines.extend(['',f'macro={macro:.3f}；ΔBASE={macro-REFERENCE:+.3f}。七任务五折均值，std(ddof=1)，seed=42+fold。',''])
    else:lines.extend(['本周期未完成140正式输出；不以validation代替正式性能。',''])
    lines.extend(['论文参考见 SOTA_RESULT.md；论文macro为逐任务配置包络。仅数值对照，不宣称公平超过论文。','限制：'+'；'.join(result['limitations'])+'。','',f'错误：{error}' if error else '',f'证据根：{OUT}；原始预测 formal/；账本 budget.json；退出 controller_exit.json / outer_exit.json / bootstrap_exit.json。'])
    (OUT/'final_review/report.md').write_text('\n'.join(lines)+'\n');(OUT/'delivery_summary.md').write_text('\n'.join(lines)+'\n')
    changes=subprocess.run(['git','diff','--','CURRENT_PLAN.md','PROJECT_HISTORY.md'],capture_output=True,text=True,check=True).stdout
    for name in SOURCES:changes+=subprocess.run(['git','diff','--no-index','--','/dev/null',name],capture_output=True,text=True).stdout
    (OUT/'actual_changes.diff').write_text(changes)
    status=f"- 状态：{'EXECUTION_COMPLETE_PENDING_REVIEW' if result['execution_complete'] else 'STOPPED'}；{result['metrics_status']}。\n- 独立时钟见 `{OUT.relative_to(ROOT)}/authorization/clock.json`；旧周期不变。\n- 真实消费见 `{OUT.relative_to(ROOT)}/budget.json`；前置CPU={pre['status']}，最终CPU={cpu['status']}。\n- FINAL={result['FINAL']}；正式macro={result['selected_macro']}；reference={REFERENCE!r}；independent_review=PENDING。\n- 报告：`{OUT.relative_to(ROOT)}/delivery_summary.md`；首错：`{OUT.relative_to(ROOT)}/first_failure.json`（如有）。\n"
    plan=ROOT/'CURRENT_PLAN.md';text=plan.read_text();start='<!-- UA3D_HMR_R1_STATUS_START -->';end='<!-- UA3D_HMR_R1_STATUS_END -->';plan.write_text(text[:text.index(start)+len(start)]+'\n'+status+text[text.index(end):])
    history=ROOT/'PROJECT_HISTORY.md'
    with history.open('a') as f:f.write('\n\n## UA3D-HMR-20261103-01/r1（待审查）\n\n'+status+'\n科学问题：有限Trimer表面统计、固定谱和学习谱读出能否为原BASE增加预测信息。三残差臂对照C_ANCHOR；不替换BASE。完整状态、预算、首错和真实exit见本周期报告；不修改旧周期结论。\n')
    write(OUT/'execution_manifest.json',dict(status='COMPLETE_PENDING_REVIEW' if result['execution_complete'] else 'STOPPED',execution_complete=result['execution_complete'],independent_review='PENDING',final_cpu=cpu,formal_complete=result['formal_complete']))
