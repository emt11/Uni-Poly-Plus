"""Independent fixed-spectrum residual readout campaign; no BASE encoder path."""
import json,math,os,random,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np
import torch
from scripts.bootstrap_ua3d_hmr_readout_r1 import ROOT,OUT,LOG,P1,P2,P3,PYTHON,THREADS,SOURCES,TASKS,ARMS,TRAIN_ARMS,read,write,exclusive,identity,size,parse_jobs,state_for,Lifecycle,check_clock,fail,cumulative
from src.dataset import ua3d_hmr_readout_r1 as d
from src.modules import ua3d_hmr_readout_r1 as m
SPEC=read(ROOT/'configs/ua3d_hmr_readout_r1_gpu2.json')
SCHEMA='ua3d-hmr-readout-best-r1'
REFERENCE=.804675462276949
SPLIT=ROOT/'data/splits/mips_paper5_outer'
BASE=ROOT/'results/ua3d_base_opt_20261026/r6/repairs/concurrency_r1'
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

def partition_positions(cache,part,task,fold):
    e=read(SPLIT/f'{task}.json')['folds'][fold];ids=sorted(e['test_indices']) if part=='test' else e[part+'_indices'];pos={row:i for i,row in enumerate(cache['indices'].tolist())};return [pos[row] for row in ids]

def slices(b,ids):return {k:v[ids] for k,v in b.items()}

def assert_finite_model(model):
    if any(not torch.isfinite(p).all() for p in model.parameters()):raise FloatingPointError('nonfinite model')

def better(candidate,best):return candidate['rmse']<best['rmse']

def optimizer_for(model):
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.0003,weight_decay=.0001,betas=(.9,.999),eps=1e-8)
    return opt,torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=70,eta_min=1e-6)

class Usage:
    def __init__(self,folder):self.path=Path(folder)/'consumption.json';self.data=dict(status='RUNNING',attempted={},completed={},started_unix=time.time(),ended_unix=None)
    def add(self,key,n=1):self.data['attempted'][key]=self.data['attempted'].get(key,0)+n
    def record(self):write(self.path,self.data)
    def finish(self):self.data.update(status='COMPLETE',completed=dict(self.data['attempted']),ended_unix=time.time());self.record()

def load_cache(task,fold,formal=False):return d.cache(task,fold,formal)
def anchor(task,fold):return read(OUT/'registry/anchors.json')['anchors'][f'{task}/{fold}']
def job_folder(kind,arm='',task='',fold=0,repeat=''):
    if kind in ('ft','test'):return OUT/('finetune' if kind=='ft' else 'formal')/arm/task/f'fold{fold}'
    if kind=='diagnostic':return OUT/'diagnostics'/('S' if task=='eps' else 'L')/arm/repeat
    return OUT/kind

def batch_data(task,fold,formal,device):
    c=d.cache(task,fold,formal);s=d.surface(task);rows=c['indices'];b={k:s[k][rows].to(device) for k in s};b.update({k:c[k].to(device) for k in ('z','availability','y','yb')});return c,b

def save_model(path,model,arm,task,fold,epoch,met,stats):
    tmp=path.with_name(path.name+f'.tmp.{os.getpid()}');torch.save(dict(schema=SCHEMA,arm=arm,task=task,fold=fold,seed=42+fold,config=SPEC,stats=stats,base=anchor(task,fold),sources=dict(surface=str(P2/'surface'),trainval=str(d.cache_path(task,fold,False)),scaler=str(P2/'scalers'/task/f'fold{fold}.json')),state_dict=model.state_dict(),epoch=epoch,metrics=met),tmp);os.replace(tmp,path)

def fit_job(kind,arm,task,fold,repeat=''):
    gpu=gpu_contract();seed(42+fold);c,b=batch_data(task,fold,False,'cuda:0');stats=d.stats(task,fold);model=m.Readout(arm,stats,fold).to('cuda:0');train=partition_positions(c,'train',task,fold);val=partition_positions(c,'validation',task,fold);folder=job_folder(kind,arm,task,fold,repeat);folder.mkdir(parents=True,exist_ok=False);opt,sch=optimizer_for(model);u=Usage(folder);u.record();history=[];gradient_nonzero=set()
    if sum(p.numel() for p in model.parameters() if p.requires_grad)!=SPEC['parameter_counts'][arm] or model.filter_raw.requires_grad:raise ValueError('parameter count/fixed filter')
    initial={k:v.detach().clone() for k,v in model.state_dict().items()};best=metrics(c['y_original'][val],c['yb_original'][val]);best_epoch=0
    save_model(folder/'epoch0.pt',model,arm,task,fold,0,best,stats)
    def update(ids):
        model.train();opt.zero_grad(set_to_none=True);mini=slices(b,ids);u.add('train_forwards');u.add('train_rows',len(ids));pred=model(mini);loss=(pred-mini['y']).square().mean()+.01*(pred-mini['yb']).square().mean()
        if not torch.isfinite(loss):raise FloatingPointError('loss')
        u.add('backwards');loss.backward();grads=[p.grad for p in model.parameters() if p.grad is not None]
        if not grads or any(not torch.isfinite(g).all() for g in grads):raise FloatingPointError('gradient')
        if kind=='diagnostic':
            for name,param in model.named_parameters():
                if param.grad is not None and torch.count_nonzero(param.grad):gradient_nonzero.add(name)
        torch.nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True);u.add('updates');opt.step();assert_finite_model(model)
        if not torch.equal(model.filter_raw,initial['filter_raw']):raise ValueError('fixed spectrum changed')
        u.record();return float(loss.detach())
    if kind=='diagnostic':
        n=64 if task=='eps' else 256;ids=torch.tensor(train[:n],device='cuda:0')
        if len(ids)!=n:raise ValueError('diagnostic subset size')
        for _ in range(2):
            model.eval();u.add('eval_forwards');u.add('eval_rows',len(ids));u.record()
            with torch.no_grad():pred=model(slices(b,ids))
            if not torch.equal(pred,b['yb'][ids]):raise ValueError('zero output anchor')
        for step in range(20):
            if opt.param_groups[0]['lr']!=.0003:raise ValueError('diagnostic actual lr')
            stamp=time.perf_counter();loss=update(ids);history.append(dict(step=step,loss=loss,lr=opt.param_groups[0]['lr'],update_seconds=time.perf_counter()-stamp))
        changed=[k for k,v in model.state_dict().items() if not torch.equal(v,initial[k])]
        if not any(k.startswith('head.0') for k in changed) or not any(k.startswith('head.6') for k in changed):raise ValueError('common head did not activate')
        if arm!='N_BASE' and any(not any(k.startswith(prefix) for k in changed) for prefix in ('U.','V.','W.')):raise ValueError('extra branch did not activate')
        prefixes=['head.0','head.6']+([] if arm=='N_BASE' else ['U.','V.','W.'])
        if any(not any(name.startswith(prefix) for name in gradient_nonzero) for prefix in prefixes):raise ValueError('diagnostic nonzero gradients did not activate')
        torch.save(dict(initial=initial,state=model.state_dict(),history=history,changed=changed,gradient_nonzero=sorted(gradient_nonzero)),folder/'trajectory.pt')
    elif kind=='ft':
        u.add('epoch0_aliases');u.add('epoch0_alias_rows',len(val));u.record();save_model(folder/'best.pt',model,arm,task,fold,0,best,stats)
        for epoch in range(1,71):
            check_clock();ids=torch.tensor(train,device='cuda:0')[torch.randperm(len(train),device='cuda:0')];lr=opt.param_groups[0]['lr'];expected=1e-6+(.0003-1e-6)*(1+math.cos(math.pi*(epoch-1)/70))/2
            if not math.isclose(lr,expected,rel_tol=1e-10,abs_tol=1e-12):raise ValueError('scheduler actual LR')
            begin=time.perf_counter();losses=[update(chunk) for chunk in ids.split(256)];train_seconds=time.perf_counter()-begin;model.eval();pred=[];u.add('validation');u.add('validation_rows',len(val));begin=time.perf_counter()
            with torch.no_grad():
                for chunk in torch.tensor(val,device='cuda:0').split(256):u.add('validation_batches');pred.append(model(slices(b,chunk)).cpu())
            met=metrics(c['y_original'][val],inverse(torch.cat(pred).numpy(),c['scaler']));val_seconds=time.perf_counter()-begin
            if better(met,best):best=met;best_epoch=epoch;save_model(folder/'best.pt',model,arm,task,fold,epoch,best,stats)
            sch.step();u.add('epochs');history.append(dict(epoch=epoch,lr=lr,loss=float(np.mean(losses)),validation=met,best_epoch=best_epoch,train_seconds=train_seconds,validation_seconds=val_seconds));write(folder/'history.json',history);u.record()
        save_model(folder/'last.pt',model,arm,task,fold,70,history[-1]['validation'],stats);write(folder/'metrics.json',dict(status='COMPLETE',arm=arm,task=task,fold=fold,best_epoch=best_epoch,validation=best,epochs=70))
    else:raise ValueError('scientific kind')
    u.finish();write(folder/'gpu.json',dict(**gpu,peak_memory_mib=torch.cuda.max_memory_allocated()/1024**2,finished_unix=time.time()))

def expected_shapes(arm):
    shapes={'filter_raw':(4,3),'head.0.weight':(256,916),'head.0.bias':(256,),'head.3.weight':(64,256),'head.3.bias':(64,),'head.6.weight':(1,64),'head.6.bias':(1,)}
    for key,n in [('z',768),('stat',16),('spec',128)]:
        for kind in ('mean','scale'):shapes[f'{key}_{kind}']=(n,)
    if arm in ('N_ADD','N_BILIN'):shapes.update({'U.weight':(32,768),'V.weight':(32,144),'LN_z.weight':(32,),'LN_z.bias':(32,),'LN_s.weight':(32,),'LN_s.bias':(32,),'W.weight':(1,32)})
    return shapes

def validate_state(state,arm):
    shapes=expected_shapes(arm)
    if set(state)!=set(shapes):raise ValueError('state keys')
    for k,shape in shapes.items():
        if tuple(state[k].shape)!=shape or state[k].dtype!=torch.float32 or not torch.isfinite(state[k]).all():raise ValueError('state shape/dtype/finite '+k)
    def logit(x):return math.log(x/(1-x))
    fixed=torch.tensor([[0.,logit((4-.05)/(16-.05)),logit((t-.0625)/(64-.0625))] for t in (.25,1,4,16)])
    if not torch.equal(state['filter_raw'],fixed):raise ValueError('fixed filter state changed')

def read_model(arm,task,fold,name='best'):
    e=d.canonical(arm,task,fold)
    if arm=='C_FIXED':p=d.fixed_payload(task,fold,name);validate_state(p['state_dict'],'N_BASE');return p
    if arm not in TRAIN_ARMS or e['source_kind']!='readout_new':raise ValueError('new model resolver')
    p=torch.load(Path(e['folder'])/f'{name}.pt',map_location='cpu',weights_only=False)
    if p['schema']!=SCHEMA or p['config']!=SPEC or any(p[k]!=v for k,v in dict(arm=arm,task=task,fold=fold,seed=42+fold).items()) or p['stats']!=d.stats(task,fold) or p['base']!=anchor(task,fold):raise ValueError('new checkpoint identity/science')
    validate_state(p['state_dict'],arm);return p

def qualify_ft(arm,task,fold):
    e=d.canonical(arm,task,fold);folder=Path(e['folder']);h=read(folder/'history.json');met=read(folder/'metrics.json');u=read(folder/'consumption.json');split=read(SPLIT/f'{task}.json')['folds'][fold]
    counts=dict(epochs=70,updates=70*math.ceil(len(split['train_indices'])/256),train_rows=70*len(split['train_indices']),validation=70,validation_rows=70*len(split['validation_indices']),validation_batches=70*math.ceil(len(split['validation_indices'])/256),epoch0_aliases=1,epoch0_alias_rows=len(split['validation_indices']))
    if len(h)!=70 or [x['epoch'] for x in h]!=list(range(1,71)) or met['status']!='COMPLETE' or u['status']!='COMPLETE' or u['attempted']!=u['completed'] or any(u['completed'].get(k)!=v for k,v in counts.items()):raise ValueError('complete FT budget/history')
    saved={n:read_model(arm,task,fold,n) for n in ('epoch0','best','last')};zero=saved['epoch0']['state_dict']
    if saved['epoch0']['epoch']!=0 or saved['last']['epoch']!=70 or torch.count_nonzero(zero['head.6.weight']) or torch.count_nonzero(zero['head.6.bias']) or arm in ('N_ADD','N_BILIN') and torch.count_nonzero(zero['W.weight']):raise ValueError('zero epoch0 or last')
    best=saved['epoch0']['metrics'];epoch=0
    for row in h:
        if better(row['validation'],best):best=row['validation'];epoch=row['epoch']
        if row['best_epoch']!=epoch:raise ValueError('strict BEST history')
    if saved['best']['epoch']!=epoch or met['best_epoch']!=epoch or met['validation']!=best or saved['best']['metrics']!=best:raise ValueError('BEST selection')
    return dict(status='COMPLETE',source_kind=e['source_kind'],best_epoch=epoch,consumption=u,model_forwards=0)

def qualify_test_sources():
    import concurrent.futures as cf
    d.require_frozen();frozen=read(OUT/'selection/final_frozen.json')
    def one(pair):
        task,fold=pair;c=d.cache(task,fold,True);split=read(SPLIT/f'{task}.json')['folds'][fold];ids=sorted(split['test_indices']);rows={x['row']:x for x in read(OUT/'registry/anchors.json')['rows'][task]}
        if c['indices'].tolist()!=ids or c['keys']!=[rows[i]['key'] for i in ids] or c['scaler']!=anchor(task,fold)['scaler']:raise ValueError('TEST compact identity')
        for name in ('z','availability','y','yb','y_original','yb_original'):
            if not torch.isfinite(c[name]).all():raise FloatingPointError('TEST cached finite')
        for arm in ('C_ANCHOR','C_FIXED'):
            path=d.old_output(arm,task,fold)
            with np.load(path,allow_pickle=False) as p:
                if p['indices'].tolist()!=ids or p['sample_keys'].tolist()!=c['keys'] or not np.array_equal(p['y_true'],c['y_original'].numpy()):raise ValueError('old output row/key/label')
                calculated=metrics(p['y_true'],p['y_pred'])
                oldmet=read(path.parent/'metrics.json')['metrics']
                if any(abs(calculated[k]-oldmet[k])>1e-12 for k in calculated):raise ValueError('old output metrics')
                if arm=='C_ANCHOR' and not np.allclose(p['y_pred'],c['yb_original'].numpy(),atol=1e-6,rtol=1e-5):raise ValueError('old anchor prediction')
        value=dict(status='COMPLETE',task=task,fold=fold,source=str(d.cache_path(task,fold,True)),rows=len(ids),qualified_unix=time.time(),freeze_unix=frozen['frozen_unix'])
        exclusive(OUT/'registry/test_qualification'/task/f'fold{fold}.json',value);return value
    workers=min(35,read(OUT/'authorization/resources.json')['qualification_workers'])
    with cf.ThreadPoolExecutor(max_workers=workers) as pool:list(pool.map(one,[(t,f) for t in TASKS for f in range(5)]))

def formal_job(arm,task,fold):
    d.require_frozen();c=d.cache(task,fold,True);ids=partition_positions(c,'test',task,fold);folder=job_folder('test',arm,task,fold);folder.mkdir(parents=True,exist_ok=False);u=Usage(folder);u.record();source=d.canonical(arm,task,fold)
    if arm in ('C_ANCHOR','C_FIXED'):
        import shutil
        cpu_contract();path=d.old_output(arm,task,fold);shutil.copyfile(path,folder/'test_predictions.npz')
        with np.load(folder/'test_predictions.npz',allow_pickle=False) as saved:y=saved['y_true'].copy();p=saved['y_pred'].copy()
        u.add('alias_outputs');u.add('alias_rows',len(ids));u.add('alias_batches',math.ceil(len(ids)/256));provenance=dict(source_kind=source['source_kind'],source_predictions=str(path),new_model_forwards=0)
    else:
        gpu=gpu_contract();seed(42+fold);c,b=batch_data(task,fold,True,'cuda:0');payload=read_model(arm,task,fold);frozen=read(OUT/'selection/final_frozen.json')['estimators'][f'{arm}/{task}/{fold}']
        if frozen['epoch']!=payload['epoch'] or frozen['path']!=str(Path(source['folder'])/'best.pt'):raise ValueError('frozen BEST provenance')
        model=m.Readout(arm,payload['stats'],fold).to('cuda:0');model.load_state_dict(payload['state_dict'],strict=True);model.eval();pred=[]
        with torch.no_grad():
            for chunk in torch.tensor(ids,device='cuda:0').split(256):u.add('candidate_forwards');u.record();pred.append(model(slices(b,chunk)).cpu())
        y=c['y_original'][ids].numpy();p=inverse(torch.cat(pred).numpy(),c['scaler']);np.savez(folder/'test_predictions.npz',indices=c['indices'][ids].numpy(),sample_keys=np.array([c['keys'][i] for i in ids]),y_true=y,y_pred=p);u.add('new_outputs');u.add('new_rows',len(ids));provenance=dict(source_kind='readout_new',checkpoint=str(Path(source['folder'])/'best.pt'),BEST_epoch=payload['epoch']);write(folder/'gpu.json',dict(**gpu,peak_memory_mib=torch.cuda.max_memory_allocated()/1024**2))
    met=metrics(y,p);u.add('test_rows',len(ids));u.add('test_batches',math.ceil(len(ids)/256));write(folder/'metrics.json',dict(status='COMPLETE',metrics=met,arm=arm,task=task,fold=fold,rows=len(ids),provenance=provenance));u.finish()

def aggregate(formal=False):
    out={}
    for arm in ARMS:
        tasks={}
        for task in TASKS:
            values=[]
            for fold in range(5):
                if formal:
                    path=job_folder('test',arm,task,fold)/'metrics.json'
                    if not path.exists():break
                    value=read(path)['metrics']['r2']
                elif arm=='C_ANCHOR':
                    c=d.cache(task,fold);ids=partition_positions(c,'validation',task,fold);value=metrics(c['y_original'][ids],c['yb_original'][ids])['r2']
                else:
                    path=Path(d.canonical(arm,task,fold)['folder'])/'metrics.json'
                    if not path.exists():break
                    value=read(path)['validation']['r2']
                values.append(value)
            if len(values)==5:tasks[task]=dict(folds=values,mean=float(np.mean(values)),std=float(np.std(values,ddof=1)))
        if len(tasks)==7:out[arm]=dict(tasks=tasks,macro=float(np.mean([x['mean'] for x in tasks.values()])))
    return out

def choose(validation):
    order=SPEC['selection_tie_order'];base=validation['C_ANCHOR'];eligible=[a for a in order if validation[a]['macro']>base['macro'] and all(validation[a]['tasks'][t]['mean']>=base['tasks'][t]['mean']-.010 for t in TASKS)]
    return max(eligible,key=lambda a:validation[a]['macro']) if eligible else 'BASE_KEEP'
def freeze():
    validation=aggregate(False)
    if len(validation)!=5:raise ValueError('freeze without complete FT matrix')
    exclusive(OUT/'selection/freeze_reservation.json',dict(time=time.time(),count=1));entries={}
    for arm in ARMS:
        for task in TASKS:
            for fold in range(5):
                source=d.canonical(arm,task,fold);entry=dict(source,scaler=anchor(task,fold)['scaler'],input_scaler=str(P2/'scalers'/task/f'fold{fold}.json'),surface=str(P2/'surface'),validation=validation[arm]['tasks'][task]['folds'][fold])
                if arm!='C_ANCHOR':payload=read_model(arm,task,fold);entry.update(path=str(Path(source['folder'])/'best.pt'),epoch=payload['epoch'],schema=payload['schema'])
                else:entry.update(path=str(P3/'formal/C_ANCHOR'/task/f'fold{fold}'/'test_predictions.npz'),epoch=None,schema='base_output_alias')
                entries[f'{arm}/{task}/{fold}']=entry
    exclusive(OUT/'selection/final_frozen.json',dict(frozen=True,FINAL=choose(validation),frozen_unix=time.time(),estimators=entries,validation=validation,reference=REFERENCE,target_version='near809-target810-performance_is_target',config=SPEC,shared_development_test=True,statistics='7 task means of 5 fold R2; std ddof1'))

def scientific_results():
    formal=aggregate(True);full=len(formal)==5;frozen=read(OUT/'selection/final_frozen.json') if (OUT/'selection/final_frozen.json').exists() else None;selected=frozen['FINAL'] if frozen else None;macro=REFERENCE if selected=='BASE_KEEP' else formal[selected]['macro'] if full and selected else None;near=bool(full and macro>=.809);target=bool(full and macro>=.810);comparisons={}
    if full:
        for a,b in [('C_FIXED','N_BASE'),('N_ADD','C_FIXED'),('N_BILIN','N_ADD'),('N_BILIN','C_FIXED')]:
            comparisons[f'{a}-{b}']=dict(macro=formal[a]['macro']-formal[b]['macro'],tasks={t:dict(mean=formal[a]['tasks'][t]['mean']-formal[b]['tasks'][t]['mean'],fold_differences=[x-y for x,y in zip(formal[a]['tasks'][t]['folds'],formal[b]['tasks'][t]['folds'])]) for t in TASKS})
    return dict(FINAL=selected,reference=REFERENCE,selected_macro=macro,selected_delta=None if macro is None else macro-REFERENCE,delta_C_FIXED=None if not full else macro-formal['C_FIXED']['macro'],formal=formal,validation=frozen['validation'] if frozen else None,formal_complete=full,near_goal_met=near,target_met=target,performance_pass=target,comparisons=comparisons,BEST_epochs={a:{f'{t}/{f}':read_model(a,t,f)['epoch'] for t in TASKS for f in range(5)} for a in ARMS if a!='C_ANCHOR'} if full else {},independent_review='PENDING')

def validate_job(j):
    log=Path(j['log']);process=read(log/'process.json');end=read(log/'end.json')
    if end['exit']!=0 or end['started_unix']!=process['started_unix'] or process['cwd']!=str(ROOT) or process['CVD']!=('2' if j['gpu'] else ''):raise ValueError('process/end identity')
    u=read(j['consumption'])
    if u['status']!='COMPLETE' or u['attempted']!=u['completed']:raise ValueError('incomplete consumption')
    if j['kind']=='ft':qualify_ft(j['arm'],j['task'],j['fold'])
    elif j['kind']=='diagnostic':
        if u['completed'].get('updates')!=20 or u['completed'].get('eval_forwards')!=2 or not (job_folder('diagnostic',j['arm'],j['task'],j['fold'],j['repeat'])/'trajectory.pt').exists():raise ValueError('diagnostic artifacts')
    elif j['kind']=='test':
        folder=job_folder('test',j['arm'],j['task'],j['fold']);met=read(folder/'metrics.json')
        with np.load(folder/'test_predictions.npz',allow_pickle=False) as p:
            if met['metrics']!=metrics(p['y_true'],p['y_pred']) or met['rows']!=len(p['indices']):raise ValueError('formal artifact consistency')
    elif j['kind']=='qualify':
        if read(OUT/'registry/sources.json')['status']!='COMPLETE':raise ValueError('qualification missing')
    elif j['kind']=='cpu':
        if read(OUT/'final_cpu/audit.json')['status']!='PASSED':raise ValueError('final CPU audit missing')

class Controller:
    def __init__(self):self.jobs=[];self.phase='INITIALIZING';self.engine=Lifecycle(validate_job,self.refresh);self.refresh()
    def refresh(self):
        usage={};parsed=parse_jobs(self.jobs)
        for j in self.jobs:
            p=Path(j['consumption'])
            if p.exists():
                u=read(p);target=usage.setdefault(j['kind'],dict(attempted={},completed={}))
                for field in ('attempted','completed'):
                    for k,v in u[field].items():target[field][k]=target[field].get(k,0)+v
        limits=SPEC['limits'];caps={k:limits[k] for k in ('preflight','qualify','diagnostic','ft','test','cpu')}
        for kind,cap in caps.items():
            if parsed['counts'].get(kind,{}).get('reserved',0)>cap:raise ValueError('job budget '+kind)
        du=usage.get('diagnostic',{}).get('attempted',{}).get('updates',0);fu=usage.get('ft',{}).get('attempted',{}).get('updates',0)
        if du>240 or fu>17850 or du+fu>18090:raise ValueError('optimizer update budget')
        ft=usage.get('ft',{}).get('attempted',{})
        for key,limit in dict(epochs=7350,train_rows=3945060,validation=7350,validation_batches=9450,validation_rows=989940,epoch0_aliases=105,epoch0_alias_rows=14142).items():
            if ft.get(key,0)>limit:raise ValueError('FT budget '+key)
        output=usage.get('test',{}).get('attempted',{})
        for key,limit in dict(new_outputs=105,alias_outputs=70,new_rows=17625,alias_rows=11750,test_rows=29375,test_batches=225,candidate_forwards=135,alias_batches=90).items():
            if output.get(key,0)>limit:raise ValueError('output budget '+key)
        family=cumulative(self.jobs,usage);ledger=dict(jobs=self.jobs,**parsed,usage=usage,limits=limits,cumulative=family,phase=self.phase,lifecycle_events=self.engine.events,new_bytes=size(OUT)+size(LOG),forbidden_generation=dict(surface=0,spectrum=0,BASE_forward=0,stats_fit=0,trainval_pack=0))
        if ledger['new_bytes']>8*1024**3:raise ValueError('disk limit')
        write(OUT/'budget.json',ledger);write(OUT/'cumulative_budget.json',family);write(OUT/'lifecycle/events.json',self.engine.events);write(OUT/'execution_manifest.json',dict(status='RUNNING',phase=self.phase,stages={k:state_for([j for j in self.jobs if j['kind']==k],v) for k,v in caps.items()},independent_review='PENDING'));return ledger
    def dispatch(self,kind,arm='',task='',fold=0,repeat=''):
        self.engine.poll_all()
        if self.engine.pending():return False
        check_clock(kind in ('preflight','qualify'));key='/'.join(map(str,(kind,arm,task,fold,repeat)))
        if any(j['id']==key for j in self.jobs):raise ValueError('duplicate job')
        gpu=kind in ('diagnostic','ft') or kind=='test' and arm in TRAIN_ARMS;log=LOG/'jobs'/key;log.mkdir(parents=True,exist_ok=False);folder=job_folder(kind,arm,task,fold,repeat);cmd=[str(PYTHON),str(ROOT/'scripts/run_ua3d_hmr_readout_r1.py'),'worker','--kind',kind,'--arm',arm,'--task',task,'--fold',str(fold),'--repeat',repeat,'--log',str(log)]
        j=dict(id=key,kind=kind,arm=arm,task=task,fold=fold,repeat=repeat,gpu=gpu,reserved_unix=time.time(),started_unix=None,ended_unix=None,exit=None,qualified=False,argv=cmd,cwd=str(ROOT),log=str(log),consumption=str(folder/'consumption.json'),ownership_tokens=['run_ua3d_hmr_readout_r1.py','worker','--kind '+kind]);self.jobs.append(j);self.refresh();env=dict(os.environ,**THREADS);env['CUDA_VISIBLE_DEVICES']='2' if gpu else '';self.engine.spawn(j,env);write(log/'dispatch.json',dict(job=j,env={**THREADS,'CUDA_VISIBLE_DEVICES':env['CUDA_VISIBLE_DEVICES']}));return True
    def wait_all(self,prepare=False):
        while self.engine.active:
            check_clock(prepare);self.engine.poll_all()
            for p,j,s in self.engine.active.values():
                if j['kind'] in ('preflight','cpu') or j['kind']=='ft' and (j['arm'],j['task'],j['fold'])==('N_BILIN','egc',1):
                    if time.time()-j['started_unix']>1800:raise TimeoutError('CPU/pilot wrapper timeout')
            time.sleep(.1)
    def one(self,kind,**kw):
        self.phase=kind
        while not self.dispatch(kind,**kw):check_clock(kind in ('preflight','qualify'));time.sleep(.1)
        self.wait_all(kind in ('preflight','qualify'))
    def admission(self):
        available=int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:')))*1024
        if available<16*1024**3:return False
        snapshots=self.engine.snapshots()
        if self.engine.pending():return False
        known=set()
        for x in snapshots:known.update([x['identity']['pid'],*x['nspid']])
        query=subprocess.run(['nvidia-smi','-i','2','--query-compute-apps=pid,used_memory,gpu_uuid','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True).stdout;unknown=[]
        for line in query.strip().splitlines():
            fields=[x.strip() for x in line.split(',')]
            if fields[-1]!=SPEC['uuid']:raise ValueError('GPU UUID identity')
            if int(fields[0]) not in known:unknown.append(line)
        write(OUT/'lifecycle/gpu_ownership.json',dict(time=time.time(),snapshots=snapshots,nvml=query,unknown=unknown,policy='unknown or unresolved namespace PID waits, never guessed or killed'))
        if unknown:return False
        free=float(subprocess.run(['nvidia-smi','-i','2','--query-gpu=memory.free','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True).stdout.strip());peaks=[]
        for j in self.jobs:
            p=job_folder(j['kind'],j['arm'],j['task'],j['fold'],j['repeat'])/'gpu.json'
            if j['kind']==self.phase and p.exists():peaks.append(read(p)['peak_memory_mib'])
        return free>=max(4096,max(peaks,default=0)+1024)+3072
    def gate(self,label):
        ft=[j for j in self.jobs if j['kind']=='ft' and j['qualified']];pilot=next(j for j in ft if (j['arm'],j['task'],j['fold'])==('N_BILIN','egc',1));remaining=105-len(ft);test_done=sum(j['kind']=='test' and j['qualified'] for j in self.jobs);diags=[j for j in self.jobs if j['kind']=='diagnostic'];maxdiag=max(j['ended_unix']-j['started_unix'] for j in diags);bilin=max(j['ended_unix']-j['started_unix'] for j in diags if j['arm']=='N_BILIN');ratio=max(1,maxdiag/bilin);max_fold_batches=max(math.ceil(len(e['train_indices'])/256) for task in TASKS for e in read(SPLIT/f'{task}.json')['folds']);pilot_batches=math.ceil(len(read(SPLIT/'egc.json')['folds'][1]['train_indices'])/256);batch_ratio=max(1,max_fold_batches/pilot_batches);per=max(j['ended_unix']-j['started_unix'] for j in ft)*ratio*batch_ratio
        normal=[j for j in ft if j is not pilot];routes=2.
        if len(normal)>=2:routes=min(8,max(1,sum(j['ended_unix']-j['started_unix'] for j in normal)/max(parse_jobs(normal)['physical_gpu_union_seconds'],1e-12)))
        # Eight-worker waves with explicit contention penalty; first gate gives only two effective routes.
        waves=math.ceil(remaining/8);contention=8/routes;ft_est=waves*per*contention;h=read(job_folder('ft','N_BILIN','egc',1)/'history.json');val_batch=max(x['validation_seconds']/math.ceil(len(read(SPLIT/'egc.json')['folds'][1]['validation_indices'])/256) for x in h);candidate_remaining=max(0,135-sum(read(j['consumption'])['completed'].get('candidate_forwards',0) for j in self.jobs if j['kind']=='test' and j['qualified']));formal_est=val_batch*candidate_remaining*contention;eta=1.5*(ft_est+formal_est)+1800+1800+600;clock=read(OUT/'authorization/clock.json');passed=time.time()+eta<clock['deadline_unix'];write(OUT/'gates'/f'{label}.json',dict(passed=passed,remaining_FT=remaining,remaining_formal=175-test_done,FT_eight_worker_waves=waves,effective_routes=routes,contention_penalty=contention,diagnostic_slowest_ratio=ratio,max_fold_train_batches=max_fold_batches,pilot_train_batches=pilot_batches,fold_batch_ratio=batch_ratio,wrapper_bound=per,observed_validation_batch_bound=val_batch,FT_estimate=ft_est,candidate_estimate=formal_est,conservative_seconds=eta,deadline_unix=clock['deadline_unix'],observed_unix=time.time(),assumption='observed wrappers and validation batches; no extra probes; eight cap is not eightfold speed'))
        if not passed:raise TimeoutError('full cost ETA gate')
    def parallel(self,kind,units,gpu=True):
        self.phase=kind;todo=list(units)
        while todo or self.engine.active:
            check_clock();self.engine.poll_all();done=sum(j['kind']==kind and j['qualified'] for j in self.jobs);cap=(2 if done<2 else 4 if done<4 else 8) if gpu else min(70,read(OUT/'authorization/resources.json')['qualification_workers'])
            while todo and len(self.engine.active)<cap:
                if gpu and not self.admission():break
                if not gpu and int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:')))*1024<16*1024**3:break
                a,t,f=todo[0]
                if not self.dispatch(kind,arm=a,task=t,fold=f):break
                todo.pop(0)
            self.gate('progress');time.sleep(.1)
    def run(self):
        try:
            self.one('preflight');self.one('qualify')
            for arm in TRAIN_ARMS:
                for task,fold in [('eps',0),('egc',1)]:
                    for repeat in ('R0','R1'):
                        self.phase='diagnostic'
                        while not self.admission():check_clock();time.sleep(.2)
                        self.one('diagnostic',arm=arm,task=task,fold=fold,repeat=repeat)
                    x=torch.load(job_folder('diagnostic',arm,task,fold,'R0')/'trajectory.pt',map_location='cpu',weights_only=False);y=torch.load(job_folder('diagnostic',arm,task,fold,'R1')/'trajectory.pt',map_location='cpu',weights_only=False)
                    numeric=lambda history:[(r['step'],r['loss'],r['lr']) for r in history]
                    if numeric(x['history'])!=numeric(y['history']) or any(not torch.equal(v,y['state'][k]) for k,v in x['state'].items()):raise ValueError('diagnostic same-seed trajectory')
            for task,fold in [('eps',0),('egc',1)]:
                initial=[torch.load(job_folder('diagnostic',arm,task,fold,'R0')/'trajectory.pt',map_location='cpu',weights_only=False)['initial'] for arm in TRAIN_ARMS]
                for state in initial[1:]:
                    if any(not torch.equal(v,state[k]) for k,v in initial[0].items() if k.startswith('head.')):raise ValueError('common head initial state mismatch')
                if any(not torch.equal(initial[1][k],initial[2][k]) for k in initial[1] if k.startswith(('U.','V.','LN_','W.'))):raise ValueError('extra branch initial state mismatch')
            self.phase='ft'
            while not self.admission():check_clock();time.sleep(.2)
            self.one('ft',arm='N_BILIN',task='egc',fold=1);self.gate('pilot');self.parallel('ft',[(a,t,f) for a in TRAIN_ARMS for t in TASKS for f in range(5) if (a,t,f)!=('N_BILIN','egc',1)])
            for a in TRAIN_ARMS:
                for t in TASKS:
                    for f in range(5):qualify_ft(a,t,f)
            self.phase='freeze';freeze();self.refresh();self.phase='TEST_sources';qualify_test_sources();self.parallel('test',[(a,t,f) for a in ('C_ANCHOR','C_FIXED') for t in TASKS for f in range(5)],False)
            controls=aggregate(True)
            if abs(controls['C_ANCHOR']['macro']-REFERENCE)>1e-6 or abs(controls['C_FIXED']['macro']-read(P3/'final_review/results.json')['formal']['S_FIXED']['macro'])>1e-12:raise ValueError('control macro gate')
            self.parallel('test',[(a,t,f) for a in TRAIN_ARMS for t in TASKS for f in range(5)]);self.refresh();write(OUT/'final_review/pre_cpu_results.json',scientific_results());self.one('cpu');self.refresh();write(OUT/'controller_return.json',dict(exit=0,time=time.time()));return 0
        except BaseException:
            err=traceback.format_exc();fail(self.phase,err);errors=self.engine.stop();write(OUT/'stop_evidence.json',dict(errors=errors,active=list(self.engine.active),time=time.time()));self.refresh();fail(self.phase,err);write(OUT/'controller_return.json',dict(exit=1,time=time.time(),phase=self.phase));return 1
