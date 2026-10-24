"""UA2D-PRETRAIN-FP: isolated training and explicitly authorized r3 recovery."""
import copy
import datetime
import json
import math
import os
from pathlib import Path
import sqlite3
import traceback
import numpy as np
import torch
from torch import nn
ROOT=Path(__file__).resolve().parents[2]
PLAN='UA2D-PRETRAIN-FP-20261019-01'
REVISION='r3'
OUT=ROOT/'results/ua2d_pretrain_fp_20261019/r1'
LOG=ROOT/'logs/ua2d_pretrain_fp_20261019/r1'
SOURCE=ROOT/'data/processed/mcl_ph_r4_20260930/pi1m_direct_01.staging'
INDEX=ROOT/'results/mcl_ph_next_20260929/p0/direct_indices_v1/summary.json'
DOWN=ROOT/'data/processed/mcl_ph_r4_20260930/downstream_direct_01.staging'
SPLIT=ROOT/'data/splits/mips_paper5_outer'
HEADS=ROOT/'results/glt_optimization_20261011/r4/U1/initialization/heads_2d'
TASKS=('eea','egb','egc','ei','eps','nc','xc')
ARMS=('V1','V2','B_FP')
INITIALIZATION_PLANS=(PLAN,)
LIMITS=dict(pt_starts=4,pt_updates=15001,pt_presentations=15120168,ft_starts=105,
    ft_epochs=7350,ft_updates=168000,validation_units=7350,test_starts=105,test_graphs=17625,
    fp_generated=911391,cpu_suites=2)
R2_LIMITS=dict(LIMITS,pt_starts=3,pt_updates=15000,pt_presentations=15120000,cpu_suites=1)
R2_USED=dict.fromkeys(LIMITS,0)
R2_USED.update(pt_starts=1,pt_updates=1,pt_presentations=168,cpu_suites=1)
# Explicit opt-in context for the user-requested engineering-only restart.
# Legacy invocations retain their original roots and budgets.
if os.environ.get('UA2D_PRETRAIN_FP_CAMPAIGN') == 'fast_r4':
    REVISION='r4'
    OUT=ROOT/'results/ua2d_pretrain_fp_20261019/fast_r4'
    LOG=ROOT/'logs/ua2d_pretrain_fp_20261019/fast_r4'
    LIMITS=dict(LIMITS,pt_starts=5,pt_updates=16392,pt_presentations=16522296,cpu_suites=3)
# Explicit opt-in context for the single-arm V1 attention-scale retrain.
# Every other invocation keeps the r1/r3 roots and the fast_r4 context above.
if os.environ.get('UA2D_PRETRAIN_FP_CAMPAIGN') == 'attnscale_r1':
    PLAN='UA2D-V1-ATTNSCALE-20261020-01'
    REVISION='r1'
    OUT=ROOT/'results/ua2d_v1_attnscale_20261020/r1'
    LOG=ROOT/'logs/ua2d_v1_attnscale_20261020/r1'
    ARMS=('V1_ATTN64',)
    INITIALIZATION_PLANS=('UA2D-PRETRAIN-FP-20261019-01',)
    LIMITS=dict(pt_starts=1,pt_updates=5000,pt_presentations=5040000,ft_starts=35,
        ft_epochs=2450,ft_updates=56000,validation_units=2450,test_starts=35,test_graphs=5875,
        fp_generated=0,cpu_suites=1)
FP_SPEC=dict(radius=2,fpSize=2048,includeChirality=True,useBondTypes=True,countSimulation=False,
    includeRingMembership=True,onlyNonzeroInvariants=False,includeRedundantEnvironments=False,
    countBounds=None,atomInvariantsGenerator=None,bondInvariantsGenerator=None,
    input='clean normalized_smiles with dummy atoms',bitorder='little',loss='per_molecule_balanced_BCE',
    weight='.1*min(1,completed_updates/1000)',head='512-256-GELU-2048; no dropout; seed20261019')


def now():return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
def read(path):return json.loads(Path(path).read_text())
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');os.replace(tmp,path)
def freeze(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)
def finite(a):
    a=np.asarray(a,dtype=np.float64)
    if not np.isfinite(a).all():raise FloatingPointError('nonfinite value')
    return a

def check_limits(counts):
    if set(counts)!=set(LIMITS) or any(type(counts[k]) is not int or not 0<=counts[k]<=v for k,v in LIMITS.items()):
        raise RuntimeError('campaign budget exceeded or invalid: '+repr(counts))


def inherited_recovery_jobs(budget):
    """Only adopt the known failed r2 handoff, without resetting any consumption."""
    if (budget.get('plan')!=PLAN or budget.get('revision')!='r2'
            or budget.get('limits')!=R2_LIMITS or budget.get('used')!=R2_USED):
        raise ValueError('recovery requires the exact recorded r2 failed-cycle budget')
    expected={
        'pretrain_fp_initialization':('initialize',0),
        'pretrain_fp_pt_V1':('pt',1),
        'pretrain_fp_final_cpu':('cpu',1),
    }
    jobs=budget.get('jobs',[])
    if (len(jobs)!=len(expected) or {j.get('name') for j in jobs}!=set(expected)
            or any((j.get('kind'),j.get('exit'))!=expected[j['name']] for j in jobs)):
        raise ValueError('recovery jobs do not match the completed r2 handoff')
    return copy.deepcopy(jobs)


def parent_path(arm):
    if arm not in ARMS:raise ValueError('unknown arm')
    path=OUT/'parent_bindings.json'
    if not path.exists():return OUT/'pretrain'/arm/'deploy.pt'
    binding=read(path)
    if binding.get('plan')!=PLAN or binding.get('revision')!=REVISION:
        raise ValueError('invalid parent binding revision')
    parent=Path(binding['parents'][arm])
    expected=OUT/'pretrain'/('V1_attempt2' if arm=='V1' else arm)/'deploy.pt'
    if parent!=expected:raise ValueError('unexpected recovery parent path')
    return parent

def objective_config(arm):
    from dataclasses import asdict
    from src.modules.ua2d_galformer_pretrain import PretrainConfig
    if arm not in ARMS:raise ValueError('unknown arm')
    base=read(OUT/'selection/baseline_frozen.json')['baseline'] if arm=='B_FP' else arm
    denominator=64 if arm=='V1_ATTN64' else 512
    if arm=='V1_ATTN64':base='V1'
    return asdict(PretrainConfig(strategy='mask_align' if base=='V1' else 'mask_align_geometry',
        fp_enabled=arm=='B_FP',attention_scale_denominator=denominator))

def initialize():
    from src.modules.ua2d_galformer_pretrain import PretrainConfig,UA2DGalformerPretrainer,SCHEMA
    from src.training.glt_dual_runtime import save_checkpoint
    folder=OUT/'initialization';folder.mkdir(exist_ok=False)
    (folder/'task_heads').mkdir()
    common=None
    for arm,strategy in [('V1','mask_align'),('V2','mask_align_geometry')]:
        torch.manual_seed(42);model=UA2DGalformerPretrainer(PretrainConfig(strategy=strategy))
        state=model.state_dict();heads={k:v for k,v in state.items() if k.startswith(('line_head.','coordinate_head.','length_head.','angle_head.'))}
        shared={k:v for k,v in state.items() if k not in heads}
        if common is None:
            common={k:v.clone() for k,v in shared.items()}
            save_checkpoint(folder/'common_step0.pt',dict(schema=SCHEMA,step=0,seed=42,state_dict=common,plan=PLAN,revision=REVISION))
        elif set(shared)!=set(common) or any(not torch.equal(v,common[k]) for k,v in shared.items()):
            raise ValueError('common initialization drift')
        save_checkpoint(folder/'task_heads'/f'{arm}.pt',dict(schema=SCHEMA,strategy=strategy,step=0,state_dict=heads))
        del model,state,shared,heads
    write(folder/'summary.json',dict(plan=PLAN,revision=REVISION,seed=42,common=str(folder/'common_step0.pt'),
        task_heads=['V1','V2'],fp_seed=20261019,trained_parent=False))


def load_initialization(model,path):
    from src.modules.ua2d_galformer_pretrain import SCHEMA
    from src.modules.glt_dual import strict_module_state
    common=torch.load(path,map_location='cpu',weights_only=False)
    if common.get('schema')!=SCHEMA or common.get('step')!=0 or common.get('seed')!=42 or common.get('plan') not in INITIALIZATION_PLANS:
        raise ValueError('invalid common step0')
    arm='V1' if model.config.reconstruct_3d_mask else 'V2'
    heads=torch.load(Path(path).parent/'task_heads'/f'{arm}.pt',map_location='cpu',weights_only=False)
    if heads.get('schema')!=SCHEMA or heads.get('step')!=0 or heads.get('strategy')!=model.config.strategy:
        raise ValueError('task head initialization mismatch')
    state=dict(common['state_dict'],**heads['state_dict'])
    # FP head is deliberately initialized in its independent RNG stream.
    state.update({k:v for k,v in model.state_dict().items() if k.startswith('fp_head.')})
    strict_module_state(model,state)


class FPReader:
    def __init__(self,root,source,index):
        root=Path(root);meta=read(root/'metadata.json')
        if meta.get('status')!='COMPLETE' or meta.get('source')!=str(source) or meta.get('index')!=str(index) or meta.get('spec')!=FP_SPEC or meta.get('rows')!=911391:
            raise ValueError('FP cache schema/source/count mismatch')
        self.db=sqlite3.connect((root/'labels.sqlite').resolve().as_uri()+'?mode=ro',uri=True)
    def get(self,row,source_smiles,topology):
        found=self.db.execute('SELECT rows.normalized,rows.source_smiles,fp.bits FROM rows JOIN fp ON rows.normalized=fp.normalized WHERE rows.source_row=?',(int(row),)).fetchone()
        if found is None or found[0]!=str(topology.normalized_canonical_smiles) or found[1]!=source_smiles:
            raise ValueError('FP cache row/canonical identity mismatch')
        bits=np.unpackbits(np.frombuffer(found[2],dtype=np.uint8),bitorder='little')
        if bits.shape!=(2048,) or not 0<int(bits.sum())<2048:raise ValueError('invalid packed FP')
        return bits
    def close(self):self.db.close()


def prepare_cache():
    from rdkit import rdBase
    from src.training.xc_kernel import generator,fingerprint
    from src.training.glt_dual_runtime import open_source,load_sample_index_artifact,pretrain_source_fields,require_tmux
    require_tmux()
    if not read(OUT/'selection/baseline_frozen.json')['frozen']:raise ValueError('FP cache before baseline freeze')
    folder=OUT/'fp_cache';folder.mkdir(exist_ok=False);source=None;db=None;generated=rows=0
    meta=dict(source=str(SOURCE),index=str(INDEX),spec=FP_SPEC,rdkit=rdBase.rdkitVersion,status='RUNNING',rows=0,generated=0)
    write(folder/'metadata.json',meta)
    try:
        indices=load_sample_index_artifact(INDEX,'train')['indices']
        if len(indices)!=911391:raise ValueError('FP train count changed')
        source,_=open_source(SOURCE,SOURCE,verify_key_digest=False)
        db=sqlite3.connect(folder/'labels.sqlite')
        db.execute('CREATE TABLE fp(normalized TEXT PRIMARY KEY,bits BLOB NOT NULL)')
        db.execute('CREATE TABLE rows(source_row INTEGER PRIMARY KEY,normalized TEXT NOT NULL,source_smiles TEXT NOT NULL)')
        gen=generator()
        for i in indices:
            i=int(i);top,_,text,_,_=pretrain_source_fields(source,i)
            normalized=str(top.normalized_canonical_smiles);entry=source.entries[i]
            if entry['pretrain_role']!='train' or entry['normalized_smiles']!=normalized or entry['source_smiles']!=text:
                raise ValueError('FP source/train/canonical row mismatch')
            if db.execute('SELECT 1 FROM fp WHERE normalized=?',(normalized,)).fetchone() is None:
                generated+=1
                if generated>len(indices):raise RuntimeError('FP budget exhausted')
                meta.update(generated=generated,rows=rows);write(folder/'metadata.json',meta)
                fp,_=fingerprint(normalized,gen);bits=np.asarray(list(fp),dtype=np.uint8)
                if not 0<int(bits.sum())<2048:raise ValueError('invalid FP target')
                db.execute('INSERT INTO fp VALUES (?,?)',(normalized,np.packbits(bits,bitorder='little').tobytes()))
            db.execute('INSERT INTO rows VALUES (?,?,?)',(i,normalized,text));rows+=1
            if rows%1000==0:
                db.commit();meta.update(rows=rows,generated=generated);write(folder/'metadata.json',meta)
                print(json.dumps(dict(rows=rows,generated=generated)),flush=True)
        db.commit();meta.update(status='COMPLETE',rows=rows,generated=generated);write(folder/'metadata.json',meta)
    except BaseException:
        meta.update(status='FAILED',rows=rows,generated=generated,error=traceback.format_exc());write(folder/'metadata.json',meta);raise
    finally:
        if source:source.close()
        if db:db.close()


class Data(torch.utils.data.Dataset):
    def __init__(self,task,fold,formal=False):
        from src.training.glt_dual_runtime import open_source
        from scripts.finetune_mcl_ph import compact_train_validation_indices
        if task not in TASKS or fold not in range(5):raise ValueError('unit outside matrix')
        manifest=read(SPLIT/f'{task}.json');entry=manifest['folds'][fold]
        if manifest.get('validation_is_test') is not False:raise ValueError('validation/test separation invalid')
        if formal:
            frozen=read(OUT/'selection/final_frozen.json')
            if not frozen['frozen']:raise ValueError('TEST before freeze')
            self.indices=sorted(entry['test_indices']);self.train=[];self.validation=list(range(len(self.indices)))
        else:self.indices,self.train,self.validation=compact_train_validation_indices(entry['train_indices'],entry['validation_indices'])
        self.source,frame=open_source(DOWN,DOWN,task=task,selected_indices=self.indices,split_manifest=manifest,verify_key_digest=False)
        if frame.original_row.astype(int).tolist()!=self.indices:raise ValueError('selected source row mismatch')
        self.raw_targets=finite(frame.label.to_numpy());self.targets=self.raw_targets.copy();self.cache={}
        self.keys=np.asarray([s[0].hex() for s in self.source.samples])
    def __len__(self):return len(self.indices)
    def __getitem__(self,i):
        from src.training.glt_dual_runtime import pretrain_source_fields
        from src.dataset.glt_dual import build_glt_2d_sample
        from scripts.finetune_mcl_ph import clone_clean_record
        if i not in self.cache:
            top,_,text,static,_=pretrain_source_fields(self.source,i)
            self.cache[i]=build_glt_2d_sample(top,text,static=static)
        d=clone_clean_record(self.cache[i]);d.y=torch.tensor([self.targets[i]],dtype=torch.float32);return d
    def set_target_override(self,y):self.targets=np.asarray(y,dtype=np.float64)
    def close(self):self.source.close()


def loader(ds,ids,shuffle=False,seed=42):
    from src.dataset.glt_dual import glt_2d_collate
    return torch.utils.data.DataLoader(torch.utils.data.Subset(ds,ids),batch_size=24,shuffle=shuffle,
        num_workers=0,drop_last=False,collate_fn=glt_2d_collate,generator=torch.Generator().manual_seed(seed) if shuffle else None)

def unit_path(arm,task,fold,formal=False):return OUT/('formal' if formal else 'finetune')/arm/task/f'fold{fold}'
def unit_config(arm,task,fold):
    from scripts.run_glt_align_2d_unit import PROFILE
    return dict(plan=PLAN,revision=REVISION,arm=arm,task=task,fold=fold,seed=42+fold,
        parent=str(parent_path(arm)),head_initial=str(HEADS/task/f'fold{fold}.pt'),
        profile=PROFILE,split=str(SPLIT/f'{task}.json'),objectives=objective_config(arm),prediction_path='pure_2d')


def finetune(arm,task,fold,formal=False):
    from src.training.glt_dual_runtime import require_tmux,save_checkpoint
    from src.modules.ua2d_galformer_pretrain import restore_downstream,package_attention,check_attention_record
    from src.modules.glt_dual import strict_module_state
    from src.utils import set_global_seed,scale_targets,evaluate,train_epoch,TargetScaler
    from sklearn.preprocessing import StandardScaler
    import scripts.finetune_mcl_ph as old
    from src.modules.glt_adaptation import AdaptationAdapter
    require_tmux()
    if os.environ.get('CUDA_VISIBLE_DEVICES') not in ('2','3'):raise ValueError('worker must use physical GPU2 or GPU3 only')
    folder=unit_path(arm,task,fold,formal);folder.mkdir(parents=True,exist_ok=False)
    cfg=unit_config(arm,task,fold);ds=None;history=[]
    progress=dict(status='RUNNING',attempted_updates=0,completed_updates=0,epochs_started=0,completed_epochs=0,
        validation_started=0,validation_completed=0,test_graphs=0)
    def record():write(folder/'consumption.json',progress)
    record()
    try:
        set_global_seed(42+fold)
        package=torch.load(cfg['parent'],map_location='cpu',weights_only=False)
        if package.get('step')!=5000 or package.get('config')!=cfg['objectives'] or package.get('execution',{}).get('revision')!=REVISION:
            raise ValueError('PT parent/config/revision mismatch')
        attention=package_attention(package)
        if attention['scale_denominator']!=cfg['objectives']['attention_scale_denominator']:
            raise ValueError('PT parent attention scale mismatch')
        head=torch.load(cfg['head_initial'],map_location='cpu',weights_only=False)
        model=restore_downstream(package,head)
        if model.attention_scale!=attention['scale'] or any(l.attention.scale!=attention['scale'] for l in model.o8.layers):
            raise ValueError('downstream attention scale mismatch')
        old.configure_periodic_tdl_dropout(model,task,profile='uniform_u1');model.to('cuda:0');del package
        if formal:
            selected=read(OUT/'selection/final_frozen.json')['units'][f'{arm}/{task}/{fold}']
            saved=torch.load(unit_path(arm,task,fold)/'best.pt',map_location='cpu',weights_only=False)
            scaler_json=read(unit_path(arm,task,fold)/'scaler.json')
            check_attention_record(saved.get('attention'),attention,'best.pt')
            check_attention_record(selected.get('attention'),attention,'final_frozen')
            if selected['config']!=cfg or saved['config']!=cfg or selected['best']!=str(unit_path(arm,task,fold)/'best.pt') or selected['best_epoch']!=saved['best_epoch'] or selected['scaler']!=scaler_json or saved['scaler']!=scaler_json:
                raise ValueError('frozen task/fold/head/parent/scaler mismatch')
            strict_module_state(model,saved['state_dict']);best_epoch=saved['best_epoch']
        ds=Data(task,fold,formal)
        if formal:
            if ds.indices!=selected['test_indices']:raise ValueError('frozen TEST indices mismatch')
            scaler=TargetScaler(task,StandardScaler(),transform_mode='standard')
            scaler.scaler.mean_=np.asarray(scaler_json['mean']);scaler.scaler.scale_=np.asarray(scaler_json['scale'])
            scaler.scaler.var_=scaler.scaler.scale_**2;scaler.scaler.n_features_in_=1
            ds.set_target_override(scaler.transform(ds.raw_targets).reshape(-1))
        else:
            scaler=scale_targets(ds,task,train_indices=ds.train,transform_mode='standard')
            scaler_json=dict(mean=scaler.scaler.mean_.tolist(),scale=scaler.scaler.scale_.tolist(),fit='train',transform='standard')
            write(folder/'scaler.json',scaler_json);save_checkpoint(folder/'head_initial.pt',head)
        validation=loader(ds,ds.validation)
        if formal:
            # Record actual graph presentations at each evaluation batch.
            def observed():
                for batch in validation:
                    progress['test_graphs']+=len(batch.y);record();yield batch
            loss,r2,truth,pred=evaluate(model,observed(),nn.MSELoss(),'cuda:0',scaler=scaler)
            if not math.isfinite(loss):raise FloatingPointError('nonfinite formal loss')
        else:
            train=loader(ds,ds.train,True,42+fold);expected=70*len(train)
            set_global_seed(42+fold);best_r2=-math.inf;best_state=None
            class HeadOnly(AdaptationAdapter):
                def train(self,mode=True):
                    super().train(mode)
                    if mode:model.eval();model.head.train()
                    return self
            class NoStep:
                def step(self):pass
            def observe(kind):
                progress[kind+'_updates']+=1
                if progress['attempted_updates']>expected:raise RuntimeError('FT unit update budget exhausted')
                record()
            for stage,epochs in [('head',10),('joint',60)]:
                old.periodic_tdl_trainability(model,head_only=stage=='head')
                wrapped=HeadOnly(model) if stage=='head' else AdaptationAdapter(model)
                optimizer,groups=old.periodic_tdl_optimizer(model,task,head_only=stage=='head',profile='uniform_u1')
                optimizer.register_step_pre_hook(lambda opt,args,kwargs:observe('attempted'))
                optimizer.register_step_post_hook(lambda opt,args,kwargs:observe('completed'))
                scheduler=(torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=10,eta_min=3e-5) if stage=='head' else
                    torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=10,T_mult=1,eta_min=1e-6))
                for stage_epoch in range(epochs):
                    progress['epochs_started']+=1;record()
                    result=train_epoch(wrapped,train,nn.MSELoss(),optimizer,NoStep(),'cuda:0',amp_dtype='fp32',max_grad_norm=5.,fail_nonfinite=True)
                    scheduler.step();progress['validation_started']+=1;record()
                    val_loss,r2,y,p=evaluate(wrapped,validation,nn.MSELoss(),'cuda:0',scaler=scaler)
                    finite([val_loss,r2]);finite(y);finite(p)
                    progress['validation_completed']+=1;progress['completed_epochs']+=1
                    rec=dict(epoch=progress['completed_epochs'],stage=stage,stage_epoch=stage_epoch+1,train_loss=result[0],
                        validation_loss=val_loss,validation_r2=r2,learning_rates=[g['lr'] for g in optimizer.param_groups],optimizer_groups=groups)
                    history.append(rec);write(folder/'history.json',history);record();print(json.dumps(rec),flush=True)
                    if r2>best_r2:
                        best_r2=float(r2);best_epoch=progress['completed_epochs']
                        best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                        truth=np.asarray(y).copy();pred=np.asarray(p).copy()
                        save_checkpoint(folder/'best.pt',dict(state_dict=best_state,config=cfg,best_epoch=best_epoch,scaler=scaler_json,
                            attention=attention,
                            task=task,fold=fold,seed=42+fold,parent=cfg['parent'],head_initial=cfg['head_initial'],outer_test='NOT_RUN'))
            if progress['completed_updates']!=expected or progress['attempted_updates']!=expected:raise ValueError('FT budget/count mismatch')
            r2=best_r2
        truth=finite(truth).reshape(-1);pred=finite(pred).reshape(-1)
        np.savez(folder/('test_predictions.npz' if formal else 'validation_predictions.npz'),y_true=truth,y_pred=pred,
            indices=np.asarray([ds.indices[i] for i in ds.validation]),sample_keys=ds.keys[ds.validation],best_epoch=best_epoch,
            scaler_mean=scaler_json['mean'],scaler_scale=scaler_json['scale'])
        from src.training.xc_kernel import metrics
        m=metrics(truth,pred);finite(list(m.values()))
        write(folder/'metrics.json',dict(status='COMPLETE',config=cfg,best_epoch=best_epoch,attention=attention,**m,outer_test='RUN' if formal else 'NOT_RUN'))
        progress['status']='COMPLETE';record()
    except BaseException as error:
        progress['status']='FAILED';record();write(folder/'failure.json',dict(error=repr(error),traceback=traceback.format_exc(),time=now(),retry=False));raise
    finally:
        if ds:ds.close()


def aggregate(arm,formal=False):
    rows={};units={}
    for task in TASKS:
        scores=[]
        for fold in range(5):
            folder=unit_path(arm,task,fold,formal);m=read(folder/'metrics.json');c=read(folder/'consumption.json')
            if m['status']!='COMPLETE' or c['status']!='COMPLETE' or m['config']!=unit_config(arm,task,fold):raise ValueError('incomplete unit')
            if not formal and (c['completed_epochs']!=70 or c['completed_updates']!=c['attempted_updates']):raise ValueError('incomplete FT')
            scores.append(float(m['r2']));units[f'{arm}/{task}/{fold}']=dict(metrics=str(folder/'metrics.json'),r2=m['r2'],
                best=str(unit_path(arm,task,fold)/'best.pt'),predictions=str(folder/('test_predictions.npz' if formal else 'validation_predictions.npz')),
                best_epoch=m['best_epoch'],config=m['config'],scaler=read(unit_path(arm,task,fold)/'scaler.json'))
        finite(scores);rows[task]=dict(folds=scores,mean=float(np.mean(scores)),std=float(np.std(scores,ddof=1)))
    return dict(tasks=rows,macro=float(np.mean([rows[t]['mean'] for t in TASKS])),units=units)


def freeze_baseline():
    values={a:aggregate(a) for a in ('V1','V2')}
    base='V2' if values['V2']['macro']>values['V1']['macro'] else 'V1'
    other='V1' if base=='V2' else 'V2';delta=values[base]['macro']-values[other]['macro']
    value=dict(plan=PLAN,revision=REVISION,frozen=True,time=now(),baseline=base,validation=values,
        delta=delta,validation_gate=delta>=.010,common=str(OUT/'initialization/common_step0.pt'),
        fp_definition=FP_SPEC,budget=read(OUT/'budget.json'),selection='full_precision_equal_task_equal_fold_validation; tie V1')
    freeze(OUT/'selection/baseline_frozen.json',value);return value


def freeze_final():
    base=read(OUT/'selection/baseline_frozen.json')['baseline'];values={a:aggregate(a) for a in ARMS}
    delta=values['B_FP']['macro']-values[base]['macro'];units={}
    for a in ARMS:
        for key,u in values[a]['units'].items():
            task=u['config']['task'];fold=u['config']['fold'];entry=read(SPLIT/f'{task}.json')['folds'][fold]
            units[key]=dict(u,test_indices=sorted(entry['test_indices']))
    frozen=dict(plan=PLAN,revision=REVISION,frozen=True,time=now(),baseline=base,validation=values,
        delta_val=delta,fp_validation_gate=delta>=.010,proposed='B_FP' if delta>=.010 else base,
        units=units,formal_arms=list(ARMS),rule='FP validation and TEST macro deltas each >=0.010, full precision; never reselect',
        budget=read(OUT/'budget.json'))
    freeze(OUT/'selection/final_frozen.json',frozen);return frozen


def reports(status,error=None):
    folder=OUT/'final_review';folder.mkdir(exist_ok=True)
    frozen=read(OUT/'selection/final_frozen.json') if (OUT/'selection/final_frozen.json').exists() else None
    results={};vals={}
    for a in ARMS:
        try:vals[a]=aggregate(a)
        except (FileNotFoundError,ValueError):pass
        try:results[a]=aggregate(a,True)
        except (FileNotFoundError,ValueError):pass
    implementation=ROOT/'docs/research/UA2D_PRETRAIN_FP_IMPLEMENTATION.md'
    if implementation.exists():
        (folder/'implementation_review.md').write_text(implementation.read_text()+'\n实际状态：'+status+'。独立审查尚未进行。\n')
    summary=dict(status=status,error=error,validation=vals,test=results,budget=read(OUT/'budget.json'),independent_review='待审查，执行自检不能替代独立审查')
    if frozen and len(results)==3:
        b=frozen['baseline'];other='V1' if b=='V2' else 'V2'
        dt=results['B_FP']['macro']-results[b]['macro'];db=results[b]['macro']-results[other]['macro']
        summary.update(delta_test=dt,fp_effective=frozen['fp_validation_gate'] and dt>=.010,
            baseline_effective=vals[b]['macro']-vals[other]['macro']>=.010 and db>=.010,
            proposed=frozen['proposed'],baseline=b)
    write(folder/'results.json',summary)
    for name,values in [('validation_report.md',vals),('test_report.md',results)]:
        lines=[f'# {PLAN} / {REVISION}',f'状态：{status}。'+(f'错误：{error}' if error else ''),
            '五折 seed=42+fold；先任务内五折均值，再七任务等权宏平均；标准差 ddof=1。',
            '| Arm | '+ ' | '.join(t.upper() for t in TASKS)+' | Macro R² |', '|---|'+'---|'*8]
        for a in ARMS:
            if a in values:
                v=values[a];lines.append('| '+a+' | '+' | '.join(f"{v['tasks'][t]['mean']:.3f}±{v['tasks'][t]['std']:.3f}" for t in TASKS)+f" | {v['macro']:.3f} |")
            else:lines.append('| '+a+' | '+' | '.join(['未完成／不可比较']*8)+' |')
        if 'delta_test' in summary:lines.append(f"FP Δvalidation={frozen['delta_val']:.3f}，ΔTEST={summary['delta_test']:.3f}；完整精度门槛通过={summary['fp_effective']}；冻结拟采用={summary['proposed']}。")
        lines.append('完整精度、折级指标、预测路径见 results.json；TEST 不参与选择。')
        (folder/name).write_text('\n'.join(lines)+'\n')
    references=dict(eea=(.934,'Periodic-TDL'),egb=(.923,'Periodic-TDL (+Chem,+RC)'),egc=(.911,'Periodic-TDL (+Chem,+RC)'),
        ei=(.839,'Periodic-TDL'),eps=(.787,'Periodic-TDL'),nc=(.856,'Periodic-TDL'),xc=(.423,'Periodic-TDL'))
    lines=['# 七任务 SOTA 数值对照','参考：SOTA_RESULT.md 的逐任务最佳模型包络，不是单个模型。',
        '| Task | Paper model | Reference | V1 R² / Δ | V2 R² / Δ | B_FP R² / Δ |','|---|---|---:|---|---|---|']
    for t,(ref,model) in references.items():
        values=[]
        for a in ARMS:
            if a in results:
                v=results[a]['tasks'][t];values.append(f"{v['mean']:.3f}±{v['std']:.3f} / {v['mean']-ref:+.3f}")
            else:values.append('未完成／不可比较：无完整正式结果')
        lines.append(f'| {t.upper()} | {model} | {ref:.3f} | '+' | '.join(values)+' |')
    lines+=['差值=项目−论文。项目为固定 paper5_outer 五折，seed42–46与fold绑定；论文原始 split／seed／预训练语料同一性未证明，只作数值对照，不宣称公平超过论文或独立盲测。',
        '正式原始预测：formal/{arm}/{task}/fold{fold}/test_predictions.npz；完整精度及路径见本目录 results.json。']
    (folder/'sota_comparison.md').write_text('\n'.join(lines)+'\n')
