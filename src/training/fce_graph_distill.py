"""r15 fixed graph recovery objective and explicit G05 branch continuation."""
import copy
import json
import math
import os
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Dataset
from src.dataset.glt_dual import build_dual_sample,dual_glt_collate
from src.dataset.glt_dual_pretrain import prepare_pretrain_sample,pretrain_collate
from src.training.glt_dual_runtime import (pretrain_source_fields,OrderedSampleStream,
    scheduled_lr,rng_state,restore_rng,open_source,load_sample_index_artifact)
from scripts.run_fce_tune_t3b_unit import pt_config_for as g05_config,validate_parent as validate_g05
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/fce_tune_20261016/graph_distill';LOG=ROOT/'logs/fce_tune_20261016/graph_distill'
G05=ROOT/'results/fce_tune_20261015/t3b/pretrain/T3B_G05'
RESUME=G05/'resume_05000.pt';TEACHER=G05/'deploy_05000.pt'
PI=ROOT/'data/processed/mcl_ph_r4_20260930/pi1m_direct_01.staging'
ARMS=('CONT_REF','GRAPH_DISTILL');REVISION='fce_graph_distill_r15_v1';VALIDATION_STEPS=(6000,7500,10000)
GRAPH_SPEC=dict(feature_dim=1024,normalization='per-graph FP32 L2; reject zero norm',
    reduction='feature squared sum then global graph mean',teacher=str(TEACHER),teacher_mode='fixed eval no_grad',warmup_new_updates=1000)
POLICY=dict(revision=REVISION,arms=list(ARMS),graph_spec=GRAPH_SPEC,weight=.75,
    ft_loss='MSE',adaptation_policy=None,recipe='R0',krr='r10 radius2 frozen alpha and arrays',task_switching=False)


def read(p):return json.loads(Path(p).read_text())
def write(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');os.replace(tmp,p)
def parent(arm):
    if arm not in ARMS:raise ValueError('undefined graph distill arm')
    return OUT/'pretrain'/arm/'deploy_10000.pt'
def pt_config_for(arm):
    if arm not in ARMS:raise ValueError('undefined graph distill arm')
    cfg=g05_config('T3B_G05')
    cfg.update(revision='r15',arm=arm,cycle_root=str(OUT),max_optimizer_steps=10000,
        graph_distill_revision=REVISION,parent_resume=str(RESUME),graph_spec=copy.deepcopy(GRAPH_SPEC),
        graph_weight_peak=0. if arm=='CONT_REF' else .1)
    return cfg
def validate_pt_config(cfg):
    if cfg!=pt_config_for(cfg.get('arm')):raise ValueError('r15 PT config mismatch')
    return cfg
def graph_weight(arm,new_update):
    if arm not in ARMS or not 1<=new_update<=5000:raise ValueError('invalid arm/new update')
    return (0. if arm=='CONT_REF' else .1)*min(new_update/1000.,1.)
def lr_at(step,cfg):
    return float(scheduled_lr(step,**{k:cfg[k] for k in ('lr','warmup_steps','schedule_total_steps','end_lr')}))


def actual_equal(a,b):
    if torch.is_tensor(a):return torch.is_tensor(b) and a.dtype==b.dtype and a.shape==b.shape and torch.equal(a.cpu(),b.cpu())
    if isinstance(a,np.ndarray):return isinstance(b,np.ndarray) and np.array_equal(a,b)
    if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(actual_equal(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)):return type(a)==type(b) and len(a)==len(b) and all(actual_equal(x,y) for x,y in zip(a,b))
    return a==b


def restore_branch(model,optimizer,state,cfg,*,rank,world,ordered_keys,source_identity):
    """Only explicit branch fields may differ. Map Adam by original parameter order."""
    validate_pt_config(cfg)
    if world!=2 or rank not in range(2):raise ValueError('two-rank resume required')
    identity=state['identity'];expected=g05_config('T3B_G05')
    if identity['config']!=expected or identity['world_size']!=world:raise ValueError('G05 original identity mismatch')
    for key,value in source_identity.items():
        if identity.get(key)!=value:raise ValueError('resume actual source mismatch '+key)
    if state['ordered_keys']!=ordered_keys or state.get('direct_binding') is not None:raise ValueError('resume ordered source/direct binding mismatch')
    if state['step']!=5000 or state['next_position']!=5040000 or state['scheduler']!=dict(step=5000,lr=lr_at(4999,cfg)):
        raise ValueError('resume step/cursor/LR trajectory mismatch')
    if len(state['rng'])!=world or any(set(r)!={'python','numpy','torch','cuda'} for r in state['rng']):raise ValueError('resume rank RNG mismatch')
    named=list(model.named_parameters());current=model.state_dict();saved=state['model']
    if list(current)!=list(saved) or list(saved)!=[n for n,p in named]:raise ValueError('PT state/parameter order mismatch')
    for n,v in current.items():
        w=saved[n]
        if v.shape!=w.shape or v.dtype!=w.dtype or not torch.isfinite(w).all():raise ValueError('resume tensor mismatch '+n)
    old=state['optimizer'];groups=old['param_groups'];indices=[i for g in groups for i in g['params']]
    if len(groups)!=1 or indices!=list(range(len(named))) or set(old['state'])!=set(indices):raise ValueError('Adam serialized order/state mismatch')
    g=groups[0]
    if g['lr']!=state['scheduler']['lr'] or tuple(g['betas'])!=(.9,.999) or g['eps']!=1e-8 or g['weight_decay']!=0:
        raise ValueError('Adam hyperparameters mismatch')
    for i,(name,p) in enumerate(named):
        moment=old['state'][i]
        if set(moment)!={'step','exp_avg','exp_avg_sq'} or float(moment['step'])!=5000:raise ValueError('Adam moment/step mismatch '+name)
        for key in ('exp_avg','exp_avg_sq'):
            if moment[key].shape!=p.shape or not torch.isfinite(moment[key]).all():raise ValueError('Adam moment shape/value mismatch '+name)
    model.load_state_dict(saved,strict=True);optimizer.load_state_dict(old)
    restored=optimizer.state_dict()
    if not actual_equal(restored,old):raise ValueError('restored Adam differs')
    return copy.deepcopy(state['rng'][rank]),dict(parent_resume=str(RESUME),parent_identity=identity,
        config=cfg,step=5000,next_position=5040000,scheduler=state['scheduler'],parameter_order=[n for n,p in named],rank=rank,world=world)


def bind_deployment(package,cfg):
    validate_pt_config(cfg)
    if package['step']!=10000:raise ValueError('fixed cumulative10000 required')
    return dict(package,training_config=cfg,graph_distill_revision=REVISION,arm=cfg['arm'],parent_resume=str(RESUME),teacher=str(TEACHER),new_updates=5000)
def validate_parent(package,arm):
    cfg=pt_config_for(arm)
    if package.get('step')!=10000 or package.get('training_config')!=cfg or package.get('graph_distill_revision')!=REVISION or package.get('arm')!=arm or package.get('parent_resume')!=str(RESUME) or package.get('teacher')!=str(TEACHER) or package.get('new_updates')!=5000:
        raise ValueError('r15 endpoint binding mismatch')
    for key in ('fusion_mode','fusion_revision','use_md200','atom_ce_input','align_enabled','local_enabled','third_task','opt_mode','geometry_revision','geometry_mode','angle_value_enabled','readout','torsion_modules','torsion_mode','geometry_target','element_mask','input_schema'):
        if package.get(key)!=cfg[key]:raise ValueError('r15 architecture mismatch '+key)
    return cfg


def paired_sample(source,index,*,position,cfg,seed=None):
    topology,trimer,smiles,static,target,identity=pretrain_source_fields(source,index,include_target=True,include_identity=True)
    # Build independently BEFORE masking, including for geometry-invalid graphs.
    clean=build_dual_sample(topology,trimer,smiles,static=static,geometry_mode=cfg['geometry_mode']).clone()
    clean.md200=torch.zeros((1,200));clean.md200_valid=torch.zeros(1,dtype=torch.bool)
    key=source.samples[index][0].hex()
    noisy,labels=prepare_pretrain_sample(topology,trimer,smiles,static=static,target=target,identity=identity,source_type='direct_pi1m',
        seed=cfg['seed'] if seed is None else seed,key=key,position=position,sigma=cfg['noise_sigma'],ratio=cfg['atom_mask_ratio'],
        opt_mode=cfg['opt_mode'],geometry_mode=cfg['geometry_mode'])
    if bool(clean.geometry_valid)!=bool(noisy.geometry_valid) or not bool(clean.graph_available) or not bool(noisy.graph_available):raise ValueError('paired availability mismatch')
    for name in ('mips_x','lga_edge_index','incidence_index','bond_canonical_a','bond_canonical_b'):
        if not torch.equal(getattr(clean,name),getattr(noisy,name)):raise ValueError('paired topology mismatch '+name)
    return noisy,labels,clean,dict(key=key,position=int(position),source_index=int(source.source_indices[index]) if hasattr(source,'source_indices') else int(index))

def paired_collate(rows):
    data,labels=pretrain_collate([(r[0],r[1]) for r in rows]);clean=dual_glt_collate([r[2] for r in rows])
    meta=[r[3] for r in rows];data.pair_keys=[r['key'] for r in meta];clean.pair_keys=list(data.pair_keys)
    data.pair_positions=[r['position'] for r in meta];clean.pair_positions=list(data.pair_positions)
    check_pair(data,clean)
    return data,labels,clean,meta

def check_pair(student,clean):
    if student.pair_keys!=clean.pair_keys or student.pair_positions!=clean.pair_positions or not torch.equal(student.geometry_valid,clean.geometry_valid) or not torch.equal(student.canonical_graph_index,clean.canonical_graph_index) or not torch.equal(student.graph_available,clean.graph_available):raise ValueError('paired graph identity/order/validity mismatch')
    if bool(clean.md200_valid.any()) or bool(student.md200_valid.any()):raise ValueError('MD must remain unavailable')

class PairedStream(Dataset):
    def __init__(self,source,cfg,rank):
        self.source=source;self.cfg=cfg;self.rank=rank;self.stream=OrderedSampleStream(len(source),cfg['seed'])
    def __len__(self):return 5000*6
    def positions(self,item):
        step=5000+item//6;micro=item%6
        return range(step*1008+micro*168+self.rank*84,step*1008+micro*168+self.rank*84+84)
    def __getitem__(self,item):
        if not 0<=item<len(self):raise IndexError(item)
        return paired_collate([paired_sample(self.source,self.stream.index_at(pos),position=pos,cfg=self.cfg) for pos in self.positions(item)])


def graph_errors(student,teacher):
    if student.ndim!=2 or student.shape!=teacher.shape or student.shape[1]!=1024:raise ValueError('1024D paired graph shape required')
    s,t=student.float(),teacher.detach().float()
    sn=s.norm(dim=1);tn=t.norm(dim=1)
    if not torch.isfinite(s).all() or not torch.isfinite(t).all() or not torch.isfinite(sn).all() or not torch.isfinite(tn).all() or bool((sn<=0).any()) or bool((tn<=0).any()):raise FloatingPointError('nonfinite/zero graph representation')
    errors=(s/sn[:,None]-t/tn[:,None]).square().sum(dim=1)
    if not torch.isfinite(errors).all():raise FloatingPointError('nonfinite graph distance')
    return errors,sn.detach(),tn.detach()
def graph_objective(errors,world,global_count):
    if world!=2 or global_count<=0:raise ValueError('invalid DDP graph denominator')
    return errors.sum()*world/global_count

def teacher_target(teacher,clean):
    if teacher.training or any(p.requires_grad for p in teacher.parameters()):raise ValueError('teacher must be fixed eval')
    with torch.no_grad():return teacher.fuse(teacher.encode(clean,atom_mask=None)).detach()


def validation_positions(start,rank,world):
    if world!=2 or rank not in range(world) or start not in range(0,1024,168):raise ValueError('invalid PT validation rank/position')
    count=min(84,(1024-start)//world)
    return range(start+rank*count,start+(rank+1)*count)


def evaluate_validation(model,teacher,cfg,step,device,rank,world):
    if step not in VALIDATION_STEPS or world!=2:raise ValueError('unauthorized PT validation')
    saved=rng_state();training=model.training;source=None;presented=0
    folder=OUT/'pt_validation'/cfg['arm']/f'step{step:05d}'
    try:
        if rank==0:folder.mkdir(parents=True,exist_ok=False)
        dist.barrier();artifact=load_sample_index_artifact(cfg['sample_index_artifact'],'fixed_validation')
        if not artifact['direct_contract'] or len(artifact['indices'])!=1024:raise ValueError('fixed validation schema mismatch')
        source,_=open_source(PI,PI,selected_indices=artifact['indices'].tolist(),verify_key_digest=False,readonly_readahead=True)
        if len(source)!=1024 or any(r['pretrain_role']!='validation' for r in source.entries):raise ValueError('validation row role mismatch')
        model.eval();sums=torch.zeros(5,dtype=torch.float64,device=device);counts=torch.zeros_like(sums)
        norms=torch.zeros(4,dtype=torch.float64,device=device);moments=torch.zeros((4,1024),dtype=torch.float64,device=device)
        with torch.no_grad():
            for start in range(0,1024,168):
                rows=[paired_sample(source,i,position=i,cfg=cfg,seed=42) for i in validation_positions(start,rank,world)]
                data,labels,clean,meta=paired_collate(rows);data=data.to(device);clean=clean.to(device)
                from src.training.glt_dual_runtime import move_labels
                labels=move_labels(labels,device)
                with torch.autocast(device.type,dtype=torch.bfloat16):
                    target=teacher_target(teacher,clean);result=model(data,labels,return_graph_feature=True)
                z=result['graph_feature'];errors,sn,tn=graph_errors(z,target)
                values=torch.cat((result['sums'][:2],result['raw_geometry_sums'],errors.sum().view(1)))
                den=torch.stack((result['counts'][0],result['counts'][1],result['counts'][1],result['angle_graphs'],errors.new_tensor(len(rows))))
                if not torch.isfinite(values).all():raise FloatingPointError('nonfinite PT validation')
                sums+=values.double();counts+=den.double();norms+=torch.stack((sn.sum(),tn.sum(),sn.square().sum(),tn.square().sum())).double()
                zs,zt=z.detach().double(),target.detach().double();moments+=torch.stack((zs.sum(0),zt.sum(0),zs.square().sum(0),zt.square().sum(0)))
                presented+=len(rows);write(folder/f'progress_rank{rank}.json',dict(student_presented=presented,teacher_presented=presented,updates=0))
        for v in (sums,counts,norms,moments):dist.all_reduce(v)
        if presented!=512 or int(counts[4])!=1024:raise ValueError('PT validation presentation mismatch')
        if rank==0:
            n=float(counts[4]);means=[float(sums[i]/counts[i]) if counts[i]>0 else None for i in range(5)]
            write(folder/'metrics.json',dict(status='PASS',arm=cfg['arm'],step=step,student_presented=1024,teacher_presented=1024,updates=0,
                components={name:dict(sum=float(sums[i]),count=float(counts[i]),mean=means[i]) for i,name in enumerate(('atom_ce','geometry','length_mse','angle_mse','graph_unweighted'))},
                norm_mean=dict(student=float(norms[0]/n),teacher=float(norms[1]/n)),
                feature_variance_mean=dict(student=float((moments[2]/n-(moments[0]/n).square()).mean()),teacher=float((moments[3]/n-(moments[1]/n).square()).mean())),
                checkpoint_selection=False,config=cfg))
    except BaseException as error:
        if folder.exists():write(folder/f'failure_rank{rank}.json',dict(error=repr(error),student_presented=presented,teacher_presented=presented))
        raise
    finally:
        if source is not None:source.close()
        model.train(training);restore_rng(saved)
