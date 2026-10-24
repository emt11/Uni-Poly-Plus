"""Bounded UA2D-SOTA r1 development utilities. No TEST access in development."""
import datetime
import json
import os
import sqlite3
from pathlib import Path
import numpy as np
import torch
from torch import nn
from src.training.xc_kernel import FEATURE_SPEC, generator, fingerprint, restore_bits, metrics

ROOT = Path(__file__).resolve().parents[2]
from src.training.ua2d_sota_context import CAMPAIGN,OUT,LOG
UA = ROOT / 'results/glt_optimization_20261011/r4/U1'
T0 = ROOT / 'results/ua2d_fusion_20261017/p2_align20k_r7'
DOWN = ROOT / 'data/processed/mcl_ph_r4_20260930/downstream_direct_01.staging'
SPLIT = ROOT / 'data/splits/mips_paper5_outer'
TASKS = ('eea','egb','egc','ei','eps','nc','xc')
PARAMS = ('P_BLR','P_HLR','P_WD','P_COS')
ALPHAS = (.1,1.,10.)
STRATEGIES = ('GRAPH','ADD','JOINT')
# 2026-10-18 user authorization (m00494): guarantee normal completion. +4 FT starts cover the
# already-run P_HLR/Egc retries; ft_epochs 29600 = 29400 + 186 epochs of the four interrupted
# attempts (kept on disk as *.attempt1_interrupted) + 14 epochs accounting margin.
LIMITS = dict(pt_starts=5,pt_updates=25000,ft_starts=424,ft_epochs=29600,krr_fits=630,
              feature_units=70,feature_graphs=47000,neural_test=315,kernel_test=140,
              test_graphs=58750,fp_generated=5875,cpu_tests=1)

def now():
    return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()

def read(path):
    return json.loads(Path(path).read_text())

def write(path, value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    os.replace(tmp,path)

def finite(a):
    a=np.asarray(a,dtype=np.float64)
    if not np.isfinite(a).all(): raise FloatingPointError('nonfinite value')
    return a

def unit_path(arm,task,fold):
    return (UA if arm=='UA' else OUT)/'finetune'/('UA_2D' if arm=='UA' else arm)/task/f'fold{fold}'

def profile(arm):
    from scripts.run_glt_align_2d_unit import PROFILE
    p=dict(PROFILE)
    if arm=='P_BLR': p['backbone_lr']=3e-5
    if arm=='P_HLR': p['head_joint_lr']=3e-4
    if arm=='P_WD': p['weight_decay']=.005
    if arm=='P_COS': p['joint_scheduler']=dict(name='CosineAnnealingLR',T_max=60,eta_min=1e-6)
    return p

def config(arm):
    if arm=='I': return read(OUT/'development/I_config.json')
    parent=UA/'pretrain/P_ALIGN/deploy_05000.pt'
    if arm=='T0': parent=T0/'pretrain/P_ALIGN/deploy_05000.pt'
    elif arm.startswith(('T_','M_')): parent=OUT/'pretrain'/arm/'deploy_05000.pt'
    if arm not in ('UA',*PARAMS,'A','T0','T_RAWCE','T_AL05','M_ALL4','M_MS_CONST','M_MS_PH'):
        raise ValueError('unknown arm')
    return dict(plan='UA2D-SOTA-20261018-01',revision='r1',**({'campaign':CAMPAIGN} if CAMPAIGN!='r1' else {}),arm=arm,parent=str(parent),
                readout='residual_attention_r1' if arm=='A' else 'mean',profile=profile(arm),
                split=str(SPLIT),seed='42+fold',head_root=str(UA/'initialization/heads_2d'))

class ResidualReadout(nn.Module):
    def __init__(self):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.default_generator.manual_seed(20261018)
            self.scorer=nn.Sequential(nn.Linear(512,64),nn.GELU(),nn.Linear(64,1))
        self.s=nn.Parameter(torch.zeros(()))
    def forward(self,atoms,index,count):
        from src.modules.glt_dual import mean_pool
        mean=mean_pool(atoms,index,count)
        score=self.scorer(atoms).squeeze(-1)
        maxima=score.new_full((count,),-torch.inf).scatter_reduce_(0,index,score,reduce='amax',include_self=True)
        w=(score-maxima[index]).exp()
        denominator=w.new_zeros(count).index_add_(0,index,w)
        w=w/denominator[index]
        attention=atoms.new_zeros((count,512)).index_add_(0,index,atoms*w[:,None])
        return mean+torch.tanh(self.s)*(attention-mean)

class Attention2D(nn.Module):
    def __init__(self,base):
        super().__init__();self.o8=base.o8;self.norm2=base.norm2;self.head=base.head;self.readout=ResidualReadout()
    def features(self,data):
        atoms,_=self.o8(data)
        return self.norm2(self.readout(atoms,data.canonical_graph_index,data.graph_available.numel()))
    def forward(self,data): return self.head(self.features(data))

def load_model(arm,task,fold,best=True):
    from src.modules.glt_dual import restore_pure_2d,strict_module_state
    cfg=config(arm)
    if arm=='T0':
        from src.training.ua2d_sota_pt import bind_t0
        binding=bind_t0()
        write(unit_path(arm,task,fold)/'control_binding.json',binding)
    package=torch.load(cfg['parent'],map_location='cpu',weights_only=False)
    head=torch.load(Path(cfg['head_root'])/task/f'fold{fold}.pt',map_location='cpu',weights_only=False)
    if package.get('sota_pt_revision'):
        from src.training.ua2d_sota_pt import restore_new
        model=restore_new(package,head)
    else:
        model=restore_pure_2d(package,head)
    if cfg['readout']=='residual_attention_r1': model=Attention2D(model)
    if best:
        saved=torch.load(unit_path(arm,task,fold)/'best.pt',map_location='cpu',weights_only=False)
        if saved['task']!=task or saved['fold']!=fold or saved['seed']!=42+fold: raise ValueError('best identity mismatch')
        if saved['parent']!=cfg['parent']: raise ValueError('best parent mismatch')
        if arm!='UA' and saved['config']!=cfg: raise ValueError('best config mismatch')
        strict_module_state(model,saved['state_dict'])
    from scripts.finetune_mcl_ph import configure_periodic_tdl_dropout
    configure_periodic_tdl_dropout(model,task,profile='uniform_u1')
    return model

class Data(torch.utils.data.Dataset):
    def __init__(self,task,fold,role='development'):
        from src.training.glt_dual_runtime import open_source
        if task not in TASKS or fold not in range(5): raise ValueError('unit outside matrix')
        manifest=read(SPLIT/f'{task}.json'); self.entry=manifest['folds'][fold]
        if role=='development':
            from scripts.finetune_mcl_ph import compact_train_validation_indices
            self.indices,self.train,self.validation=compact_train_validation_indices(self.entry['train_indices'],self.entry['validation_indices'])
        elif role=='formal':
            if not read(OUT/'selection_before_test.json')['frozen']: raise ValueError('TEST before freeze')
            self.indices=sorted(self.entry['test_indices']);self.train=[];self.validation=list(range(len(self.indices)))
        else: raise ValueError('invalid role')
        self.source,frame=open_source(DOWN,DOWN,task=task,selected_indices=self.indices,split_manifest=manifest,verify_key_digest=False)
        if frame.original_row.astype(int).tolist()!=self.indices: raise ValueError('source row mismatch')
        self.y=finite(frame.label.to_numpy());self.targets=self.y.copy();self.cache={}
        self.keys=np.asarray([s[0].hex() for s in self.source.samples])
        self.smiles=frame.normalized_smiles.astype(str).tolist()
    def __len__(self): return len(self.indices)
    def __getitem__(self,i):
        from src.training.glt_dual_runtime import pretrain_source_fields
        from src.dataset.glt_dual import build_glt_2d_sample
        from scripts.finetune_mcl_ph import clone_clean_record
        if i not in self.cache:
            top,tri,text,static,_=pretrain_source_fields(self.source,i)
            self.cache[i]=build_glt_2d_sample(top,text,static=static)
        d=clone_clean_record(self.cache[i]);d.y=torch.tensor([self.targets[i]],dtype=torch.float32)
        return d
    def close(self): self.source.close()

def loader(ds,ids,shuffle=False,seed=42):
    from src.dataset.glt_dual import glt_2d_collate
    return torch.utils.data.DataLoader(torch.utils.data.Subset(ds,ids),batch_size=24,shuffle=shuffle,
        num_workers=0,collate_fn=glt_2d_collate,generator=torch.Generator().manual_seed(seed) if shuffle else None)

def normalize(h):
    h=finite(h);norm=np.linalg.norm(h,axis=1,keepdims=True)
    if (norm<=0).any(): raise ValueError('zero norm graph')
    return finite(h/norm)

def distances(a,b): return np.maximum(0,((a[:,None,:]-b[None,:,:])**2).sum(-1))

def graph_kernel(train,query):
    # Pairwise dot product avoids a multi-gigabyte N x N x 512 temporary.
    train=normalize(train);query=normalize(query)
    dt=np.maximum(0,np.sum(train**2,1)[:,None]+np.sum(train**2,1)[None,:]-2*train@train.T)
    mask=~np.eye(len(train),dtype=bool)
    positive=dt[mask & (dt>0)]
    if not len(positive): raise ValueError('no positive train distance')
    s=float(np.median(positive));np.fill_diagonal(dt,0)
    dq=np.maximum(0,np.sum(query**2,1)[:,None]+np.sum(train**2,1)[None,:]-2*query@train.T)
    return np.exp(-dt/s),np.exp(-dq/s),s,train

def combine(g,f,strategy):
    if strategy=='GRAPH': return g
    add=.5*g+.5*f
    if strategy=='ADD': return add
    if strategy=='JOINT': return .75*add+.25*g*f
    raise ValueError('unknown strategy')

class Fingerprints:
    def __init__(self):
        self.old=sqlite3.connect((ROOT/'results/fp_graph_fusion_20261017/f1/fingerprints.sqlite').as_uri()+'?mode=ro',uri=True)
        self.side=sqlite3.connect(str(OUT/'fingerprints.sqlite'))
        self.side.execute('CREATE TABLE IF NOT EXISTS bits (smiles TEXT PRIMARY KEY,value BLOB NOT NULL)');self.side.commit()
        self.gen=generator();self.generated=0
    def get(self,smiles):
        row=self.old.execute('SELECT bits FROM bits WHERE smiles=?',(smiles,)).fetchone()
        if row is None: row=self.side.execute('SELECT value FROM bits WHERE smiles=?',(smiles,)).fetchone()
        if row is None:
            if self.side.execute('SELECT COUNT(*) FROM bits').fetchone()[0]>=5875: raise RuntimeError('FP budget exhausted')
            fp,_=fingerprint(smiles,self.gen);bits=np.asarray(list(fp),dtype=np.uint8)
            self.side.execute('INSERT INTO bits VALUES (?,?)',(smiles,bits.tobytes()));self.side.commit();self.generated+=1
            return bits
        bits=np.frombuffer(row[0],dtype=np.uint8).copy()
        if bits.shape!=(2048,) or not np.isin(bits,[0,1]).all() or not bits.any(): raise ValueError('FP schema mismatch')
        return bits
    def close(self): self.old.close();self.side.close()

def fp_kernel(query,train):
    from rdkit import DataStructs
    a=restore_bits(query);b=restore_bits(train)
    return finite([DataStructs.BulkTanimotoSimilarity(fp,b,returnDistance=False) for fp in a])

@torch.inference_mode()
def extract(model,ds,ids):
    model.eval();rows=[]
    for batch in loader(ds,ids): rows.append(model.features(batch.to('cuda:0')).float().cpu().numpy())
    return finite(np.concatenate(rows))

def kernel_round(round_name,arm):
    from scipy.linalg import cho_factor,cho_solve
    from src.training.glt_dual_runtime import save_checkpoint
    if round_name not in ('F0','F0_recovery1','F1') or (round_name.startswith('F0') and arm!='UA'): raise ValueError('invalid kernel round')
    if CAMPAIGN=='r1_retest1' and round_name=='F0_recovery1': raise ValueError('retest forbids legacy recovery')
    if round_name=='F0_recovery1' and not read(OUT/'recovery_authorization.json').get('authorized'):
        raise ValueError('F0 recovery requires explicit authorization')
    initial=refresh_budget()
    if initial['feature_units']+35>LIMITS['feature_units'] or initial['feature_graphs']+23500>LIMITS['feature_graphs']:
        raise RuntimeError('insufficient remaining feature budget for a complete 35-fold round')
    root=OUT/'kernel'/round_name;root.mkdir(parents=True,exist_ok=False)
    fp=Fingerprints(); summary=[];fits=0;graphs=0
    write(root/'progress.json',dict(status='RUNNING',fits=0,feature_units=0,feature_graphs=0))
    try:
        for task in TASKS:
            for fold in range(5):
                ds=Data(task,fold)
                try:
                    model=load_model(arm,task,fold).to('cuda:0')
                    ids=ds.train+ds.validation
                    current=refresh_budget()
                    if current['feature_units']+1>LIMITS['feature_units'] or current['feature_graphs']+len(ids)>LIMITS['feature_graphs']:
                        raise RuntimeError('feature budget exhausted before extraction; no retry')
                    graphs+=len(ids)
                    write(root/'progress.json',dict(status='RUNNING',fits=fits,feature_units=len(summary)+1,feature_graphs=graphs,fp_generated=fp.generated))
                    h=extract(model,ds,ids)
                    encoder_folder=root/'encoder'/task/f'fold{fold}';encoder_folder.mkdir(parents=True)
                    save_checkpoint(encoder_folder/'deployment.pt',dict(config=config(arm),schema='pure_2d_encoder_readout_r1',state_dict={k:v.detach().cpu().clone() for k,v in model.state_dict().items() if not k.startswith('head.')}))
                    n=len(ds.train);ht,hv=h[:n],h[n:]
                    kg,vg,s,u=graph_kernel(ht,hv)
                    bits=np.stack([fp.get(ds.smiles[i]) for i in ids]);bt,bv=bits[:n],bits[n:]
                    kf=fp_kernel(bt,bt);vf=fp_kernel(bv,bt)
                    scaler=read(unit_path(arm,task,fold)/'scaler.json')
                    mu=float(scaler['mean'][0]);sd=float(scaler['scale'][0])
                    yt=ds.y[ds.train];yv=ds.y[ds.validation]
                    if not np.isclose(mu,yt.mean(),rtol=1e-12,atol=1e-12) or not np.isclose(sd,yt.std(),rtol=1e-12,atol=1e-12): raise ValueError('train-only scaler mismatch')
                    z=(yt-mu)/sd
                    for strategy in STRATEGIES:
                        folder=root/strategy/task/f'fold{fold}';folder.mkdir(parents=True,exist_ok=False)
                        k=combine(kg,kf,strategy);v=combine(vg,vf,strategy);trials=[];coefs={};preds={}
                        for alpha in ALPHAS:
                            if fits>=315: raise RuntimeError('round fit budget exhausted')
                            fits+=1
                            write(root/'progress.json',dict(status='RUNNING',fits=fits,feature_units=len(summary)+1,feature_graphs=graphs,fp_generated=fp.generated))
                            c=finite(cho_solve(cho_factor(k+alpha*np.eye(n),lower=True,check_finite=True),z))
                            p=finite(mu+sd*(v@c));m=metrics(yv,p)
                            coefs[alpha]=c;preds[alpha]=p;trials.append(dict(alpha=alpha,**m))
                        chosen=min(trials,key=lambda x:(x['rmse'],-x['alpha']));alpha=chosen['alpha']
                        np.savez(folder/'model.npz',train_u=u,train_bits=bt,train_indices=np.asarray([ds.indices[i] for i in ds.train]),
                            train_keys=ds.keys[ds.train],bandwidth=s,alpha=alpha,coef=coefs[alpha],mean=mu,scale=sd,
                            strategy=strategy,feature_spec=json.dumps(FEATURE_SPEC),encoder_readout=str(encoder_folder/'deployment.pt'),encoder_best=str(unit_path(arm,task,fold)/'best.pt'))
                        np.savez(folder/'validation_predictions.npz',indices=np.asarray([ds.indices[i] for i in ds.validation]),sample_keys=ds.keys[ds.validation],y_true=yv,y_pred=preds[alpha])
                        write(folder/'selection.json',dict(arm=arm,task=task,fold=fold,round=round_name,strategy=strategy,trials=trials,chosen=chosen,
                              scaler=scaler,support_count=n,package_bytes=(folder/'model.npz').stat().st_size,
                              encoder_best=str(unit_path(arm,task,fold)/'best.pt'),outer_test='NOT_RUN'))
                    feature_folder=root/'features'/task/f'fold{fold}';feature_folder.mkdir(parents=True)
                    np.savez(feature_folder/'train_validation.npz',features=h,indices=np.asarray([ds.indices[i] for i in ids]),sample_keys=ds.keys[ids],train_count=n)
                    summary.append(dict(task=task,fold=fold))
                    print(json.dumps(dict(round=round_name,task=task,fold=fold,fits=fits,graphs=graphs)),flush=True)
                    del model;torch.cuda.empty_cache()
                finally: ds.close()
        means={s:float(np.mean([read(root/s/t/f'fold{f}'/'selection.json')['chosen']['r2'] for t in TASKS for f in range(5)])) for s in STRATEGIES}
        chosen=max(('ADD','JOINT'),key=lambda x:means[x])
        neural=float(np.mean([read(unit_path(arm,t,f)/'metrics.json')['r2'] for t in TASKS for f in range(5)]))
        write(root/'progress.json',dict(status='PASS',fits=fits,feature_units=35,feature_graphs=graphs,fp_generated=fp.generated))
        write(root/'summary.json',dict(status='PASS',arm=arm,validation=means,neural_validation=neural,selected_fusion=chosen,
              fusion_eligible=means[chosen]-neural>=.010,fits=fits,feature_units=35,feature_graphs=graphs,fp_generated=fp.generated,outer_test='NOT_RUN',time=now()))
    finally: fp.close()

def run_ft(arm,task,fold):
    from src.utils import set_global_seed,scale_targets,train_epoch,evaluate
    from src.modules.glt_adaptation import AdaptationAdapter
    from src.training.glt_dual_runtime import save_checkpoint
    cfg=config(arm);p=cfg['profile'];folder=unit_path(arm,task,fold)
    folder.mkdir(parents=True,exist_ok=False);write(folder/'config.json',cfg)
    history=[];updates=dict(attempted=0,completed=0);ds=None
    write(folder/'updates.json',updates)
    try:
        set_global_seed(42+fold);model=load_model(arm,task,fold,best=False).to('cuda:0')
        ds=Data(task,fold)
        # Same StandardScaler implementation and original inner-train index order.
        ds.raw_targets=ds.y
        def override(v): ds.targets=np.asarray(v).reshape(-1)
        ds.set_target_override=override
        set_global_seed(42+fold)
        scaler=scale_targets(ds,task,train_indices=ds.train,transform_mode='standard')
        sc=dict(mean=scaler.scaler.mean_.tolist(),scale=scaler.scaler.scale_.tolist(),fit='train',transform='standard')
        if sc!=read(UA/'finetune/UA_2D'/task/f'fold{fold}'/'scaler.json'): raise ValueError('reference scaler mismatch')
        write(folder/'scaler.json',sc)
        train=loader(ds,ds.train,True,42+fold);val=loader(ds,ds.validation)
        exact=70*len(train)
        write(folder/'budget.json',dict(starts=1,epochs_limit=70,updates_limit=exact,train_rows=len(ds.train),validation_rows=len(ds.validation),outer_test='NOT_RUN'))
        class HeadAdapter(AdaptationAdapter):
            def train(self,mode=True):
                super().train(mode)
                if mode:
                    model.eval();model.head.train()
                    if hasattr(model,'readout'): model.readout.train()
                return self
        best_rmse=float('inf');best=None
        for stage,count in (('head',10),('joint',60)):
            head_ids={id(x) for x in model.head.parameters()}
            readout_ids={id(x) for x in model.readout.parameters()} if hasattr(model,'readout') else set()
            for x in model.parameters(): x.requires_grad_(stage=='joint' or id(x) in head_ids|readout_ids)
            groups=[]
            backbone=[x for x in model.parameters() if x.requires_grad and id(x) not in head_ids]
            if backbone: groups.append(dict(params=backbone,lr=p['backbone_lr'] if stage=='joint' else p['head_only_lr'],weight_decay=p['weight_decay'],name='backbone' if stage=='joint' else 'readout'))
            groups.append(dict(params=list(model.head.parameters()),lr=p['head_joint_lr'] if stage=='joint' else p['head_only_lr'],weight_decay=p['weight_decay'],name='head'))
            optimizer=torch.optim.AdamW(groups)
            def before(*_):
                if updates['attempted']>=exact: raise RuntimeError('FT update budget exhausted')
                updates['attempted']+=1;write(folder/'updates.json',updates)
            def after(*_): updates['completed']+=1;write(folder/'updates.json',updates)
            optimizer.register_step_pre_hook(before);optimizer.register_step_post_hook(after)
            if stage=='head': scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=10,eta_min=3e-5)
            elif p['joint_scheduler']['name']=='CosineAnnealingLR': scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=60,eta_min=1e-6)
            else: scheduler=torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=10,T_mult=1,eta_min=1e-6)
            wrapped=HeadAdapter(model) if stage=='head' else AdaptationAdapter(model)
            for ep in range(count):
                write(folder/'epochs_attempted.json',dict(attempted=len(history)+1))
                result=train_epoch(wrapped,train,nn.MSELoss(),optimizer,None,'cuda:0',epoch=len(history)+1,amp_dtype='fp32',max_grad_norm=5.,fail_nonfinite=True,return_timing=True)
                scheduler.step()
                loss,r2,y,pred=evaluate(wrapped,val,nn.MSELoss(),'cuda:0',scaler=scaler,amp_dtype='fp32')
                m=metrics(y,pred)
                rec=dict(epoch=len(history)+1,stage=stage,stage_epoch=ep+1,train_loss=float(result[0]),validation_loss=loss,validation_r2=m['r2'],validation_rmse=m['rmse'],training_steps=result[-1]['training_steps'],learning_rates=[g['lr'] for g in optimizer.param_groups])
                history.append(rec);write(folder/'progress.json',dict(history=history,updates=updates,outer_test='NOT_RUN'))
                print(json.dumps(rec),flush=True)
                if m['rmse']<best_rmse:
                    best_rmse=m['rmse'];best=dict(r2=m['r2'],rmse=m['rmse'],epoch=len(history),y=finite(y).reshape(-1),pred=finite(pred).reshape(-1))
                    save_checkpoint(folder/'best.pt',dict(state_dict={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                        config=cfg,task=task,fold=fold,seed=42+fold,parent=cfg['parent'],best_epoch=len(history),scaler_mean=sc['mean'],scaler_scale=sc['scale'],
                        head_initial=str(Path(cfg['head_root'])/task/f'fold{fold}.pt'),outer_test='NOT_RUN'))
        if updates['attempted']!=exact or updates['completed']!=exact: raise ValueError('FT final update mismatch')
        np.savez(folder/'validation_predictions.npz',indices=np.asarray([ds.indices[i] for i in ds.validation]),sample_keys=ds.keys[ds.validation],y_true=best['y'],y_pred=best['pred'],scaler_mean=sc['mean'],scaler_scale=sc['scale'],best_epoch=best['epoch'])
        write(folder/'metrics.json',dict(status='PASS',arm=arm,task=task,fold=fold,epochs=70,updates=updates,r2=best['r2'],rmse=best['rmse'],best_epoch=best['epoch'],outer_test='NOT_RUN'))
    except BaseException as e:
        write(folder/'failure.json',dict(error=repr(e),traceback=__import__('traceback').format_exc(),completed_epochs=len(history),updates=updates,retry=False,time=now()))
        raise
    finally:
        if ds is not None: ds.close()

def refresh_budget(publish=None):
    ft=list((OUT/'finetune').glob('*/*/fold*/config.json'))
    histories=[read(p) for p in (OUT/'finetune').glob('*/*/fold*/progress.json')]
    updates=[read(p) for p in (OUT/'finetune').glob('*/*/fold*/updates.json')]
    pt=[read(p) for p in (OUT/'pretrain').glob('*/consumption.json')]
    kernels=[read(p) for p in (OUT/'kernel').glob('*/progress.json')]
    # F0 first launch completed one unpersisted feature extraction before the SQLite schema error.
    failed=read(OUT/'failure_consumption.json') if CAMPAIGN=='r1' and (OUT/'failure_consumption.json').exists() else {}
    budget=dict(pt_starts=len(list(LOG.glob('pt_*/command.json'))),pt_updates=sum(p['attempted_updates'] for p in pt),
        ft_starts=max(len(ft),len(list(LOG.glob('ft_*/command.json')))),ft_epochs=sum(max(len(read(p.parent/'progress.json')['history']) if (p.parent/'progress.json').exists() else 0,read(p.parent/'epochs_attempted.json')['attempted'] if (p.parent/'epochs_attempted.json').exists() else 0) for p in ft),krr_fits=sum(p['fits'] for p in kernels),
        feature_units=sum(p['feature_units'] for p in kernels)+failed.get('feature_units',0),
        feature_graphs=sum(p['feature_graphs'] for p in kernels)+failed.get('feature_graphs',0),
        neural_test=sum(int(read(p)['new_neural_test']) for p in (OUT/'formal/neural').glob('*/*/fold*/intent.json')),
        kernel_test=len(list((OUT/'formal/kernel').glob('*/*/*/fold*/intent.json'))),
        test_graphs=sum(read(p)['graphs'] for p in (OUT/'formal/neural').glob('*/*/fold*/intent.json')),fp_generated=0,cpu_tests=len(list(LOG.glob('final_cpu/command.json'))))
    side=OUT/'fingerprints.sqlite'
    if side.exists():
        db=sqlite3.connect(side.as_uri()+'?mode=ro',uri=True)
        budget['fp_generated']=db.execute('SELECT COUNT(*) FROM bits').fetchone()[0];db.close()
    for key,val in budget.items():
        if val>LIMITS[key]: raise RuntimeError('budget exceeded: '+key)
    if publish is None: publish=os.environ.get('UA2D_SOTA_ROLE')!='worker'
    if publish: write(OUT/'budget.json',dict(campaign=CAMPAIGN,actual=budget,limits=LIMITS,ft_attempted_updates=sum(p['attempted'] for p in updates),
        ft_completed_updates=sum(p['completed'] for p in updates),pt_completed_updates=sum(p['completed_updates'] for p in pt),
        ft_completed_epochs=sum(len(p['history']) for p in histories),pt_attempted_presentations=budget['pt_updates']*1008,
        ph_generation=0,conformer_generation=0,extra_seed=0,time=now()))
    return budget

def validation_scores():
    arms=('UA',*PARAMS,'A','T0','T_RAWCE','T_AL05','M_ALL4','M_MS_CONST','M_MS_PH','I')
    values={};units={}
    for arm in arms:
        records=[]
        for task in TASKS:
            for fold in range(5):
                p=unit_path(arm,task,fold)/'metrics.json'
                if p.exists():
                    m=read(p)
                    if m['outer_test']!='NOT_RUN' or m['status']!='PASS': raise ValueError('invalid development record')
                    records.append(dict(task=task,fold=fold,r2=m['r2'],best_epoch=read_best_epoch(arm,task,fold)))
        units[arm]=records
        if len(records)==35: values[arm]=float(np.mean([r['r2'] for r in records]))
    return values,units

def read_best_epoch(arm,task,fold):
    m=read(unit_path(arm,task,fold)/'metrics.json')
    if 'best_epoch' in m: return m['best_epoch']
    p=np.load(unit_path(arm,task,fold)/'validation_predictions.npz',allow_pickle=False)
    try: return int(p['best_epoch'])
    finally: p.close()

def propose_integration():
    values,units=validation_scores()
    if 'UA' not in values: raise ValueError('UA development reference incomplete')
    ua=values['UA'];components=[];chosen_p=None;chosen_parent=None
    complete_p=[a for a in PARAMS if a in values]
    if len(complete_p)==4:
        chosen_p=max(PARAMS,key=lambda a:values[a])
        if values[chosen_p]-ua>=.010: components.append(chosen_p)
    if 'A' in values and values['A']-ua>=.010: components.append('A')
    pt=[a for a in ('T_RAWCE','T_AL05','M_ALL4','M_MS_CONST','M_MS_PH') if a in values]
    if pt and 'T0' in values:
        chosen_parent=max(pt,key=lambda a:values[a])
        if values[chosen_parent]-values['T0']>=.010 and values[chosen_parent]-ua>=.010: components.append(chosen_parent)
    cfg=config('UA');cfg['arm']='I'
    for arm in components:
        c=config(arm)
        if arm.startswith('P_'): cfg['profile']=c['profile']
        elif arm=='A': cfg['readout']=c['readout']
        else: cfg['parent']=c['parent']
    fields=('parent','profile','readout','head_root','split','seed')
    equivalent=next((a for a in values if all(config(a)[k]==cfg[k] for k in fields)),None)
    reference=max(components,key=lambda a:values[a]) if components else 'UA'
    proposal=dict(components=components,p_winner=chosen_p,pt_winner=chosen_parent,reference=reference,
        equivalent=equivalent,run=bool(components and equivalent is None),config=cfg,validation=values,time=now())
    write(OUT/'development/integration_proposal.json',proposal)
    write(OUT/'development/validation_summary.json',dict(scores=values,units=units,integration=proposal,outer_test='NOT_RUN',time=now()))
    if proposal['run']: write(OUT/'development/I_config.json',cfg)
    return proposal

def choose_core():
    values,units=validation_scores();ua=values['UA']
    candidates=['UA']+[a for a in (*PARAMS,'A') if a in values]
    for arm in ('T_RAWCE','T_AL05','M_ALL4','M_MS_CONST','M_MS_PH'):
        if arm in values and 'T0' in values and values[arm]-values['T0']>=.010 and values[arm]-ua>=.010: candidates.append(arm)
    if 'I' in values:
        reference=read(OUT/'development/integration_proposal.json')['reference']
        if values['I']>values[reference]: candidates.append('I')
    best=max(candidates,key=lambda a:values[a])
    if values[best]-ua<.010: best='UA'
    payload=dict(core=best,score=values[best],candidates=candidates,scores=values,time=now(),test_used=False)
    write(OUT/'development/core_selection.json',payload)
    return payload

def freeze_selection():
    values,units=validation_scores();core=read(OUT/'development/core_selection.json')['core']
    required=(*PARAMS,'A','T0','T_RAWCE','T_AL05','M_ALL4','M_MS_CONST','M_MS_PH')
    missing=[a for a in required if a not in values]
    if missing: raise ValueError('incomplete development blocks: '+str(missing))
    rounds=['F0']+(['F1'] if core!='UA' else [])
    kernels=[];systems=[dict(kind='neural',arm=core,validation=values[core])]
    for rnd in rounds:
        path=kernel_folder(rnd)/'summary.json'
        if not path.exists(): raise ValueError('incomplete kernel round '+rnd)
        summary=read(path)
        kernels.extend([dict(round=rnd,strategy=x,arm=summary['arm']) for x in ('GRAPH',summary['selected_fusion'])])
        if summary['fusion_eligible']:
            systems.append(dict(kind='kernel',round=rnd,strategy=summary['selected_fusion'],arm=summary['arm'],validation=summary['validation'][summary['selected_fusion']]))
    final=max(systems,key=lambda x:x['validation'])
    p_winner=max(PARAMS,key=lambda x:values[x])
    neural=[p_winner,'A','T0','T_RAWCE','T_AL05','M_ALL4','M_MS_CONST','M_MS_PH']
    if 'I' in values: neural.append('I')
    reference={p_winner:'UA','A':'UA','T0':'UA','T_RAWCE':'T0','T_AL05':'T0','M_ALL4':'T0','M_MS_CONST':'M_ALL4','M_MS_PH':'M_MS_CONST'}
    if 'I' in values: reference['I']=read(OUT/'development/integration_proposal.json')['reference']
    frozen=[]
    for arm in neural+['UA']:
        for task in TASKS:
            for fold in range(5):
                folder=unit_path(arm,task,fold)
                frozen.append(dict(arm=arm,task=task,fold=fold,best=str(folder/'best.pt'),config=config(arm),
                    scaler=read(folder/'scaler.json'),best_epoch=read_best_epoch(arm,task,fold),split=read(SPLIT/f'{task}.json')['folds'][fold]))
    ku=[]
    for kernel in kernels:
        for task in TASKS:
            for fold in range(5):
                folder=kernel_folder(kernel['round'])/kernel['strategy']/task/f'fold{fold}'
                choice=read(folder/'selection.json')
                ku.append(dict(**kernel,task=task,fold=fold,model=str(folder/'model.npz'),alpha=choice['chosen']['alpha']))
    if len(neural)*35>315 or len(ku)>140: raise RuntimeError('formal list exceeds budget')
    selection=dict(frozen=True,plan='UA2D-SOTA-20261018-01',revision='r1',campaign=CAMPAIGN,neural_core=core,final_system=final,
        eligible_systems=systems,neural_arms=neural,kernel_strategies=kernels,neural_units=frozen,kernel_units=ku,
        comparisons=reference,validation=values,time=now(),test_used_for_selection=False,final_test_policy='no reselection or supplementary experiments')
    path=OUT/'selection_before_test.json'
    if path.exists(): raise FileExistsError('selection already frozen')
    write(path,selection)
    return selection

def frozen_neural_unit(arm,task,fold):
    selection=read(OUT/'selection_before_test.json')
    if not selection.get('frozen'): raise ValueError('TEST before freeze')
    matches=[u for u in selection['neural_units'] if (u['arm'],u['task'],u['fold'])==(arm,task,fold)]
    if len(matches)!=1: raise ValueError('unit outside frozen list')
    u=matches[0]
    if u['config']!=config(arm) or u['scaler']!=read(unit_path(arm,task,fold)/'scaler.json') or u['best_epoch']!=read_best_epoch(arm,task,fold): raise ValueError('frozen unit changed')
    return selection,u

@torch.inference_mode()
def neural_formal(arm,task,fold):
    selection,u=frozen_neural_unit(arm,task,fold)
    if arm not in selection['neural_arms'] and arm!='UA': raise ValueError('unlisted formal neural arm')
    if arm=='UA' and not any(k['arm']=='UA' for k in selection['kernel_strategies']): raise ValueError('UA test forward not needed')
    folder=OUT/'formal/neural'/arm/task/f'fold{fold}';folder.mkdir(parents=True,exist_ok=False)
    entry=read(SPLIT/f'{task}.json')['folds'][fold]
    if entry!=u['split']: raise ValueError('frozen split changed')
    graphs=len(entry['test_indices'])
    actual=refresh_budget()
    if actual['test_graphs']+graphs>LIMITS['test_graphs']: raise RuntimeError('TEST forward budget exhausted')
    write(folder/'intent.json',dict(arm=arm,task=task,fold=fold,graphs=graphs,new_neural_test=arm!='UA',time=now()))
    ds=Data(task,fold,'formal')
    try:
        model=load_model(arm,task,fold).to('cuda:0').eval()
        h=extract(model,ds,ds.validation)
        # Reuse these exact encoder features for every frozen kernel strategy.
        standardized=finite(model.head(torch.tensor(h,dtype=torch.float32,device='cuda:0')).flatten().cpu().numpy())
        sc=u['scaler'];prediction=finite(sc['mean'][0]+sc['scale'][0]*standardized)
        truth=finite(((ds.y-sc['mean'][0])/sc['scale'][0]).astype(np.float32))*sc['scale'][0]+sc['mean'][0]
        m=metrics(truth,prediction)
        np.savez(folder/'test_predictions.npz',indices=np.asarray(ds.indices),sample_keys=ds.keys,y_true=truth,source_y_true=ds.y,y_pred=prediction,
            features=h,scaler_mean=sc['mean'],scaler_scale=sc['scale'],best_epoch=u['best_epoch'])
        write(folder/'metrics.json',dict(status='PASS',arm=arm,task=task,fold=fold,**m,outer_test='RUN',encoder_graphs=graphs,
            purpose='kernel shared features; existing UA predictions retained as reference' if arm=='UA' else 'frozen neural formal'))
    finally: ds.close()

def kernel_formal(round_name,strategy,task,fold):
    selection=read(OUT/'selection_before_test.json')
    if not selection.get('frozen'): raise ValueError('TEST before freeze')
    matches=[u for u in selection['kernel_units'] if (u['round'],u['strategy'],u['task'],u['fold'])==(round_name,strategy,task,fold)]
    if len(matches)!=1: raise ValueError('kernel outside frozen list')
    unit=matches[0];arm=unit['arm']
    folder=OUT/'formal/kernel'/round_name/strategy/task/f'fold{fold}';folder.mkdir(parents=True,exist_ok=False)
    write(folder/'intent.json',dict(**unit,time=now()))
    with np.load(unit['model'],allow_pickle=False) as p: model={k:p[k].copy() for k in p.files}
    if float(model['alpha'])!=unit['alpha'] or str(model['strategy'])!=strategy: raise ValueError('frozen kernel changed')
    entry=read(SPLIT/f'{task}.json')['folds'][fold]
    if model['train_indices'].tolist()!=entry['train_indices']: raise ValueError('frozen support rows mismatch')
    # Formal graph feature extraction already completed once by neural_formal.
    with np.load(OUT/'formal/neural'/arm/task/f'fold{fold}'/'test_predictions.npz',allow_pickle=False) as p:
        h=normalize(p['features']);ids=p['indices'].copy();keys=p['sample_keys'].copy();truth=p['source_y_true'].copy()
    if ids.tolist()!=sorted(entry['test_indices']): raise ValueError('formal query row order mismatch')
    ds=Data(task,fold,'formal');fp=Fingerprints()
    try:
        if not np.array_equal(ds.keys,keys) or not np.array_equal(ds.y,truth): raise ValueError('formal source identity mismatch')
        bits=np.stack([fp.get(x) for x in ds.smiles])
        train=model['train_u'];s=float(model['bandwidth'])
        d=np.maximum(0,np.sum(h*h,1)[:,None]+np.sum(train*train,1)[None,:]-2*h@train.T)
        kg=np.exp(-d/s);kf=fp_kernel(bits,model['train_bits'])
        prediction=finite(float(model['mean'])+float(model['scale'])*(combine(kg,kf,strategy)@model['coef']))
        np.savez(folder/'test_predictions.npz',indices=ids,sample_keys=keys,y_true=truth,y_pred=prediction)
        write(folder/'metrics.json',dict(status='PASS',**unit,**metrics(truth,prediction),outer_test='RUN',support_count=len(train)))
    finally: ds.close();fp.close()


def kernel_folder(round_name):
    if CAMPAIGN=='r1' and round_name=='F0' and (OUT/'kernel/F0_recovery1/summary.json').exists(): return OUT/'kernel/F0_recovery1'
    return OUT/'kernel'/round_name

def summarize_results():
    import csv
    root=OUT/'final_review';root.mkdir(parents=True,exist_ok=True)
    paper={}
    import re
    text=(ROOT/'SOTA_RESULT.md').read_text().split('## 各任务最佳 R²',1)[1]
    for line in text.splitlines():
        columns=[x.strip() for x in line.split('|')]
        if len(columns)>3 and columns[1].lower() in TASKS:
            match=re.search(r'\*\*([0-9.]+)\s*±\s*([0-9.]+)\*\*',columns[3])
            if match: paper[columns[1].lower()]=dict(model=columns[2],r2=float(match[1]),std=float(match[2]))
    if set(paper)!=set(TASKS): raise ValueError('paper seven-task references missing')
    selection=read(OUT/'selection_before_test.json') if (OUT/'selection_before_test.json').exists() else None
    systems={'UA':UA/'formal/UA_2D'}
    if selection:
        systems.update({a:OUT/'formal/neural'/a for a in selection['neural_arms']})
        systems.update({k['round']+'_'+k['strategy']:OUT/'formal/kernel'/k['round']/k['strategy'] for k in selection['kernel_strategies']})
    report={}
    for name,path in systems.items():
        tasks={}
        for task in TASKS:
            folds=[]
            for fold in range(5):
                p=path/task/f'fold{fold}'/'metrics.json'
                if p.exists():
                    m=read(p)
                    if m['status']=='PASS' and m['outer_test']=='RUN': folds.append(dict(fold=fold,r2=m['r2'],rmse=m['rmse'],metrics=str(p),predictions=str(p.with_name('test_predictions.npz'))))
            if len(folds)==5:
                r2=float(np.mean([f['r2'] for f in folds]));rmse=float(np.mean([f['rmse'] for f in folds]))
                tasks[task]=dict(status='已完成',r2=dict(mean=r2,std=float(np.std([f['r2'] for f in folds]))),
                    rmse=dict(mean=rmse,std=float(np.std([f['rmse'] for f in folds]))),paper=paper[task],minus_paper=r2-paper[task]['r2'],folds=folds)
            else: tasks[task]=dict(status='未完成／不可比较',reason=f'{len(folds)}/5 formal folds',folds=folds,paper=paper[task])
        complete=all(x['status']=='已完成' for x in tasks.values())
        report[name]=dict(tasks=tasks,complete=complete,macro_r2=float(np.mean([t['r2']['mean'] for t in tasks.values()])) if complete else None)
    r10_report=read(ROOT/'results/fce_tune_20261016/global_blend/final_review/seven_task_results.json')
    r10=r10_report['macro']['global']['r2']
    for name,system in report.items():
        for task,rec in system['tasks'].items():
            if rec['status']=='已完成':
                rec['minus_UA']=rec['r2']['mean']-report['UA']['tasks'][task]['r2']['mean']
                rec['r10_r2']=r10_report['tasks'][task]['stats']['global']['r2']['mean']
                rec['minus_r10']=rec['r2']['mean']-rec['r10_r2']
    refs=selection['comparisons'] if selection else {}
    if selection:
        for k in selection['kernel_strategies']:
            name=k['round']+'_'+k['strategy']
            refs[name]=k['arm'] if k['strategy']=='GRAPH' else k['round']+'_GRAPH'
    pairs=list(refs.items())
    if selection:
        for k in selection['kernel_strategies']:
            if k['strategy']!='GRAPH': pairs.append((k['round']+'_'+k['strategy'],k['arm']))
            if k['round']=='F1' and 'F0_'+k['strategy'] in report: pairs.append(('F1_'+k['strategy'],'F0_'+k['strategy']))
    comparisons=[];paired=[]
    for name,reference in dict.fromkeys(pairs):
        if name not in report or reference not in report or not report[name]['complete'] or not report[reference]['complete']:
            comparisons.append(dict(candidate=name,reference=reference,status='未完成／不可比较'));continue
        delta=report[name]['macro_r2']-report[reference]['macro_r2']
        comparisons.append(dict(candidate=name,reference=reference,status='已完成',delta=delta,effective=delta>=.010,
            cumulative_minus_UA=report[name]['macro_r2']-report['UA']['macro_r2'],cumulative_minus_r10=report[name]['macro_r2']-r10))
        for task in TASKS:
            a=report[name]['tasks'][task];b=report[reference]['tasks'][task]
            for x,y in zip(a['folds'],b['folds']):
                paired.append(dict(candidate=name,reference=reference,task=task,fold=x['fold'],r2_delta=x['r2']-y['r2'],rmse_delta=x['rmse']-y['rmse']))
    final_name=None
    if selection:
        final=selection['final_system'];final_name=final['arm'] if final['kind']=='neural' else final['round']+'_'+final['strategy']
    final_complete=bool(final_name in report and report[final_name]['complete'])
    final_result=dict(system=final_name,status='已完成' if final_complete else '未完成／不可比较')
    if final_complete:
        final_result.update(macro_r2=report[final_name]['macro_r2'],minus_UA=report[final_name]['macro_r2']-report['UA']['macro_r2'],
            minus_r10=report[final_name]['macro_r2']-r10,effective_vs_UA=report[final_name]['macro_r2']-report['UA']['macro_r2']>=.010,
            all_seven_numeric_targets=all(t['minus_paper']>=0 for t in report[final_name]['tasks'].values()))
    protocol='paper5_outer existing folds 0-4; inner train-only scaler; seed42+fold; equal 5 fold R2 then equal 7 tasks; std ddof=0. Historical TEST participated in route development. Paper data/split/PT corpus/seed statistics not proven identical; numerical comparison only.'
    write(root/'seven_task_results.json',dict(systems=report,comparisons=comparisons,final=final_result,paper=paper,protocol=protocol,independent_review='NOT_RUN',time=now()))
    with (root/'paired_deltas.csv').open('w') as h:
        writer=csv.DictWriter(h,fieldnames=['candidate','reference','task','fold','r2_delta','rmse_delta']);writer.writeheader();writer.writerows(paired)
    lines=['# UA2D-SOTA r1 七任务正式对照','',protocol,'',f'冻结最终系统：{final_name or "未冻结"}；状态：{final_result["status"]}。','',
        '| Task | 本计划最终系统 R² | 论文最佳 R² | 论文模型 | 项目 − 论文 |','|---|---:|---:|---|---:|']
    for task in TASKS:
        p=paper[task];t=report[final_name]['tasks'][task] if final_name else None
        value=f'{t["r2"]["mean"]:.3f}±{t["r2"]["std"]:.3f}' if t and t['status']=='已完成' else '未完成／不可比较'
        delta=f'{t["minus_paper"]:+.3f}' if t and t['status']=='已完成' else '—'
        lines.append(f'| {task} | {value} | {p["r2"]:.3f}±{p["std"]:.3f} | {p["model"]} | {delta} |')
    lines+=['','| Candidate | Reference | TEST macro ΔR² | ≥0.010 |','|---|---|---:|---|']
    for c in comparisons:
        lines.append(f'| {c["candidate"]} | {c["reference"]} | '+(f'{c["delta"]:+.3f} | {c["effective"]} |' if c['status']=='已完成' else '未完成／不可比较 | — |'))
    if final_complete: lines+=[f'\n最终系统 macro R² {final_result["macro_r2"]:.3f}；相对 UA {final_result["minus_UA"]:+.3f}；相对 r10 {final_result["minus_r10"]:+.3f}。']
    lines+=['','最终系统按 validation 冻结；不根据 TEST 切换系统或补实验。R² 越高越好，差值为项目减参考。RMSE 差值为项目减参考，越低越好。',
        '',f'完整精度结果：`{root}/seven_task_results.json`；成对差值：`{root}/paired_deltas.csv`。']
    (root/'seven_task_comparison.md').write_text('\n'.join(lines)+'\n')
    return final_complete

def capture_implementation():
    import subprocess,difflib
    changed=['src/modules/glt_dual_pretrain.py']
    diff=subprocess.check_output(['git','diff','--',*changed],text=True)
    files=['src/training/ua2d_sota.py','src/training/ua2d_sota_pt.py','scripts/run_ua2d_sota.py','scripts/ua2d_sota_dispatch.py',
        'scripts/ua2d_sota_continue.py','scripts/ua2d_sota_record.py','scripts/ua2d_sota_recover_f0.py','scripts/ua2d_sota_finish.py','tests/test_ua2d_sota_plan.py']
    for name in files:
        path=ROOT/name
        if path.exists(): diff+=''.join(difflib.unified_diff([],path.read_text().splitlines(True),fromfile='/dev/null',tofile='b/'+name))
    folder=OUT/'final_review';folder.mkdir(parents=True,exist_ok=True);(folder/'implementation.diff').write_text(diff)


def update_execution_plan(status,detail):
    path=ROOT/'CURRENT_PLAN.md';text=path.read_text()
    start=text.index('- 状态：');end=text.index('\n',start)
    text=text[:start]+f'- 状态：**{status}（{CAMPAIGN}）**。用户当前消息授权本批固定矩阵、正式评估及唯一 CPU suite；实际证据见 §12.3。旧 controller／r8 不恢复。'+text[end:]
    marker=f'### {CAMPAIGN} 执行快照'
    if marker in text:
        start=text.index(marker);end=text.index('\n## 13.',start);text=text[:start]+text[end:]
    at=text.index('\n## 13.');ledger=read(OUT/'budget.json');b=ledger['actual']
    block=marker+'\n\n'+f'- 时间：{now()}；状态：{status}。{detail}\n'
    block+='- 本地 dev／3503fbe780a5399410d6db6055305925765c3ff8 及已有未提交工作树；保护旧 r1/r2、用户修改与删除，不读 TODO、不操作远程；新批候选不复用旧候选结果。\n'
    block+=f'- 本批独立预算：PT {b["pt_starts"]} starts／{b["pt_updates"]} attempted updates；FT {b["ft_starts"]} starts／{b["ft_epochs"]} epochs；KRR {b["krr_fits"]} fits；train/validation 特征 {b["feature_units"]} units／{b["feature_graphs"]} graphs；neural TEST {b["neural_test"]}、kernel TEST {b["kernel_test"]}；新 FP {b["fp_generated"]}；CPU suite {b["cpu_tests"]} 次。原始更新计数见 `{OUT.relative_to(ROOT)}/budget.json`。失败计入，无自动重试。\n'
    block+=f'- 历史消耗仅另存快照 `{OUT.relative_to(ROOT)}/historical_consumption.json`，不清零旧账本、不混入本批预算。\n'
    block+=f'- 新结果 `{OUT.relative_to(ROOT)}/`；日志 `{LOG.relative_to(ROOT)}/`。单一 controller：`Uni-Poly:ua2d_sota_retest1_controller`；单元窗口 `retest1_<stage/arm/task/fold>`。命令、cwd、环境、PID、开始／结束与真实 exit 均按单元保留；worker/PT/预算/freeze/formal 共用明确 campaign 与根配置。\n'
    block+='- 实现：campaign context 与新独立 controller；共享运行模块仅扩展必要隔离与正式收尾，旧 r1/r2 controller 保持暂停。训练 workers=0，GPU2/3，PT 两 rank 独占、FT 每卡最多两路；不新增 hash、PH 或构象。\n'
    review=OUT/'final_review/seven_task_comparison.md'
    if review.exists(): block+='\n'+review.read_text().split('\n',2)[2]+'\n'
    else: block+='\n| Task | 本批正式结果 |\n|---|---|\n'+'\n'.join(f'| {t} | 未完成／不可比较：正式阶段未完成 |' for t in TASKS)+'\n'
    block+='\n- 未完成正式结果不由 smoke／validation 补齐；当前仅执行自检，不等于独立审查通过。PROJECT_HISTORY.md 不在本批审查通过前归档。\n'
    path.write_text(text[:at]+'\n'+block+text[at:])
