"""Fixed two-rank PT candidates and paired legacy MCL auxiliary view."""
from contextlib import nullcontext
import os
from types import SimpleNamespace
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
from src.training.ua2d_sota import ROOT,OUT,UA,T0,read,write,now,CAMPAIGN
from src.modules.glt_dual_pretrain import DualPretrainer,alignment_loss,alignment_local_anchor_count,global_objective,deployment_package
from src.modules.glt_dual import mean_pool,strict_module_state

ARMS=('T_RAWCE','T_AL05','M_ALL4','M_MS_CONST','M_MS_PH')
PI=ROOT/'data/processed/mcl_ph_r4_20260930/pi1m_direct_01.staging'
CACHE=ROOT/'data/processed/mcl_ph_r4_20261001/trajectory_train_5040000'
STATS=ROOT/'data/processed/mcl_ph_r4_20261001/statistics_train_full.npz'
REV='ua2d_sota_pt_r1'

def pt_config(arm):
    if arm not in ARMS: raise ValueError('PT arm outside matrix')
    cfg=read(UA/'initialization/P_ALIGN.json')
    cfg.update(plan='UA2D-SOTA-20261018-01',revision='r1',cycle='SOTA',arm=arm,**({'campaign':CAMPAIGN} if CAMPAIGN!='r1' else {}),cycle_root=str(OUT),expected_world_size=2,
        initial_state=str(UA/'initialization/P_ALIGN_step0.pt'),sota_pt_revision=REV,pt_validation=False,
        atom_ce_input='raw' if arm=='T_RAWCE' else 'fused',align_weight=.05 if arm=='T_AL05' else .1,
        mcl_enabled=arm.startswith('M_'),mcl_cutoffs=[4.,4.,4.] if arm=='M_ALL4' else [2.,3.,4.],
        mcl_descriptor='ph' if arm=='M_MS_PH' else 'zero',mcl_temperature=.1,mcl_weight=.1,mcl_warmup=1000,
        mcl_init_seed=20261018,trajectory_cache=str(CACHE),statistics=str(STATS),workers=0)
    return cfg

def validate_pt(cfg):
    if cfg!=pt_config(cfg.get('arm')): raise ValueError('strict r1 PT config mismatch')
    return cfg

class MCLView:
    """Only required heavy-atom fields; no legacy GLT carrier or target building."""
    def __init__(self,**kwargs): self.__dict__.update(kwargs)
    def to(self,device):
        return MCLView(**{k:v.to(device) if torch.is_tensor(v) else v for k,v in vars(self).items()})

def prepare_mcl(top,tri,text,static,*,seed,key,position,trajectory,statistics):
    from src.dataset.mcl_ph_view import (build_mcl_ph_view,centre_mapping,physical_bonds,synchronised_mask,
        cached_topology_trajectory,expert_relations,_bond_category)
    view=build_mcl_ph_view(top,tri,text,seed=seed,key=key,position=position,sigma=.03,ratio=.3)
    clean=view.base_view;noisy=view.field_view
    center,_,nonheavy=centre_mapping(top,tri,clean)
    valid=bool(getattr(tri,'trimer_geometry_valid',False)) and bool(static['geometry_valid'])
    trajectory=cached_topology_trajectory(trajectory)
    if not valid and trajectory.any(): raise ValueError('PH invalid geometry mismatch')
    elements,charges,aromatics,_=synchronised_mask(top,clean,view.mask)
    bonds,types,_=physical_bonds(static,clean)
    if valid:
        edges,scales,distance=expert_relations(noisy.positions)
        # A single 4 A relation list; all experts choose their own cutoff.
        mask=scales==2;edges=edges[:,mask];distance=distance[mask]
        edge_type=_bond_category(edges.t().contiguous(),bonds,types)
    else:
        edges=torch.zeros((2,0),dtype=torch.long);distance=torch.zeros(0);edge_type=torch.zeros(0,dtype=torch.long)
    descriptor=(trajectory-statistics['router_mean'])/statistics['router_std'] if valid else np.zeros((31,5),dtype=np.float32)
    if not np.isfinite(descriptor).all(): raise FloatingPointError('nonfinite standardized PH')
    return MCLView(element=elements,charge=charges,aromatic=aromatics,center=center,edges=edges,
        distance=distance.float(),edge_type=edge_type,descriptor=torch.from_numpy(descriptor.copy()),
        valid=valid and nonheavy==0,key=key,identity=str(top.normalized_canonical_smiles),position=position)

def pack_mcl(items):
    fields={k:[] for k in ('element','charge','aromatic','center','edges','distance','edge_type','atom_batch','center_batch')}
    offset=0
    for i,item in enumerate(items):
        n=item.element.numel()
        if item.center.numel() and ((item.center < -1).any() or (item.center>=n).any()): raise ValueError('MCL center outside heavy atom space')
        for k in ('element','charge','aromatic','distance','edge_type'): fields[k].append(getattr(item,k))
        fields['center'].append(torch.where(item.center>=0,item.center+offset,item.center))
        fields['edges'].append(item.edges+offset)
        fields['atom_batch'].append(torch.full((n,),i,dtype=torch.long))
        fields['center_batch'].append(torch.full((item.center.numel(),),i,dtype=torch.long));offset+=n
    packed={k:torch.cat(v,dim=1 if k=='edges' else 0) for k,v in fields.items()}
    return MCLView(**packed,descriptor=torch.stack([i.descriptor for i in items]),valid=torch.tensor([i.valid for i in items]),
                   keys=[i.key for i in items],identities=[i.identity for i in items],positions=[i.position for i in items],graphs=len(items))

class MCLAuxiliary(nn.Module):
    def __init__(self,arm):
        super().__init__()
        from src.modules.mcl_ph import SchNetExpert
        if arm not in ARMS[2:]: raise ValueError('invalid MCL arm')
        self.arm=arm;self.cutoffs=(4.,4.,4.) if arm=='M_ALL4' else (2.,3.,4.)
        with torch.random.fork_rng(devices=[]):
            torch.default_generator.manual_seed(20261018)
            self.experts=nn.ModuleList([SchNetExpert(c) for c in self.cutoffs])
            self.router=nn.Sequential(nn.Linear(155,128),nn.GELU(),nn.Linear(128,3))
            self.proj2=nn.Sequential(nn.Linear(512,256),nn.GELU(),nn.Linear(256,128))
            self.proj3=nn.Sequential(nn.Linear(128,256),nn.GELU(),nn.Linear(256,128))
    def forward(self,raw2,view,identities,valid):
        descriptor=view.descriptor.flatten(1)
        if self.arm!='M_MS_PH': descriptor=torch.zeros_like(descriptor)
        alpha=self.router(descriptor).softmax(-1);pooled=[]
        centers=view.center;keep=centers>=0
        for expert,cutoff in zip(self.experts,self.cutoffs):
            use=view.distance<=cutoff;edges=view.edges[:,use]
            states=expert(view.element,view.charge,view.aromatic,edges[0],edges[1],view.distance[use],view.edge_type[use],view.element.numel())
            pooled.append(mean_pool(states[centers[keep]],view.center_batch[keep],view.graphs))
        graph=(torch.stack(pooled,dim=1)*alpha[:,:,None]).sum(1)
        a=F.normalize(self.proj2(raw2),dim=-1);b=F.normalize(self.proj3(graph),dim=-1)
        loss,count=alignment_loss(a,b,identities,valid,temperature=.1)
        report=dict(route_mean=alpha.detach().float().mean(0).tolist(),route_entropy=float(-(alpha.detach().float().clamp_min(1e-12).log()*alpha.detach().float()).sum(-1).mean()))
        return loss,count,report

class Trainer(nn.Module):
    def __init__(self,cfg):
        super().__init__();self.cfg=validate_pt(cfg)
        self.base=DualPretrainer('token',geometry_head_norm=True,third_task='none',opt_mode='GLT_FUSION_R1',fusion_revision='glt_fusion_r1',
            atom_ce_input=cfg['atom_ce_input'],align_enabled=True,align_weight=cfg['align_weight'],align_warmup_steps=1000,
            align_temperature=.1,align_revision='ua2d_sota_r1',align_init_seed=314161,
            geometry_revision='glt_periodic_seam_r1',geometry_mode='finite_incident',angle_value_enabled=True,readout='incident')
        initial=torch.load(cfg['initial_state'],map_location='cpu',weights_only=False)
        if initial.get('step')!=0 or initial['config']!=read(UA/'initialization/P_ALIGN.json'): raise ValueError('PT common initial config mismatch')
        strict_module_state(self.base,initial['model'])
        self.aux=MCLAuxiliary(cfg['arm']) if cfg['mcl_enabled'] else None
    def forward(self,data,labels,mcl=None):
        result=self.base(data,labels,return_graph_feature=self.aux is not None)
        if self.aux is not None:
            if labels['align_identity']!=mcl.identities: raise ValueError('UA/MCL molecule pairing mismatch')
            valid=labels['align_valid'].bool() & mcl.valid.bool()
            loss,count,report=self.aux(result['raw_graph_2d'],mcl,labels['align_identity'],valid)
            result.update(mcl_sum=loss,mcl_count=count,mcl_report=report)
        return result

def export(trainer):
    pkg=deployment_package(trainer.base,5000)
    pkg.update(sota_pt_revision=REV,sota_config=trainer.cfg,plan='UA2D-SOTA-20261018-01',revision='r1',**({'campaign':CAMPAIGN} if CAMPAIGN!='r1' else {}),arm=trainer.cfg['arm'])
    return pkg

def restore_new(package,head):
    from src.modules.glt_dual import Pure2DGLTModel,_u1_encoder_shapes
    cfg=validate_pt(package['sota_config'])
    expected=dict(sota_pt_revision=REV,step=5000,plan='UA2D-SOTA-20261018-01',revision='r1',**({'campaign':CAMPAIGN} if CAMPAIGN!='r1' else {}),arm=cfg['arm'],
        atom_ce_input=cfg['atom_ce_input'],align_weight=cfg['align_weight'],align_temperature=.1,align_warmup_steps=1000,
        align_enabled=True,geometry_revision='glt_periodic_seam_r1',geometry_mode='finite_incident',angle_value_enabled=True,readout='incident',
        fusion_mode='token',fusion_revision='glt_fusion_r1',use_md200=False,fusion_gate='none',local_enabled=False,third_task='none',
        torsion_modules=False,opt_mode='GLT_FUSION_R1',pretrain_revision='ua2d_sota_r1',align_revision='ua2d_sota_r1',align_init_seed=314161,
        align_input='raw_graph_2d_graph_3d',endpoint_revision=None,fusion_attention_mode='global',incident_bias=0.0)
    if any(k not in package or type(package[k]) is not type(v) or package[k]!=v for k,v in expected.items()): raise ValueError('new r1 deployment metadata mismatch')
    model=Pure2DGLTModel();shapes=_u1_encoder_shapes(model);state=package['state_dict']
    if set(state)!=set(shapes): raise ValueError('new r1 encoder schema mismatch')
    for k,v in state.items():
        if tuple(v.shape)!=shapes[k] or v.dtype!=torch.float32 or not torch.isfinite(v).all(): raise ValueError('new r1 tensor mismatch '+k)
    for name in ('o8','norm2'):
        module=getattr(model,name);strict_module_state(module,{k:state[name+'.'+k] for k in module.state_dict()})
    strict_module_state(model.head,head)
    return model

def run_pt(arm):
    from src.training.glt_dual_runtime import (open_source,load_sample_index_artifact,IndexedFrozenDualSource,
        OrderedSampleStream,pretrain_source_fields,prepare_source_sample,move_labels,scheduled_lr,save_checkpoint)
    from src.dataset.glt_dual_pretrain import pretrain_collate
    from src.utils import set_global_seed
    cfg=pt_config(arm);rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE'])
    from src.training.ua2d_sota_scheduler import validate_pt_mapping
    physical=validate_pt_mapping(world,os.environ.get('CUDA_VISIBLE_DEVICES',''),CAMPAIGN,os.environ.get('UA2D_SOTA_RESOURCE_REVISION',''))
    torch.cuda.set_device(rank);device=torch.device('cuda',rank);dist.init_process_group('nccl')
    folder=OUT/'pretrain'/arm
    if rank==0: folder.mkdir(parents=True,exist_ok=False);write(folder/'config.json',cfg)
    dist.barrier();source=cache=None;attempted=completed=0
    try:
        source,_=open_source(PI,PI,verify_key_digest=False,readonly_readahead=True)
        index=load_sample_index_artifact(cfg['sample_index_artifact'],'train')
        if not index.get('direct_contract'): raise ValueError('PT requires direct row source')
        source=IndexedFrozenDualSource(source,index['indices'])
        stats=None
        if cfg['mcl_enabled']:
            from src.dataset.mcl_ph_view import load_geometric_statistics,validate_direct_statistics_source
            from src.dataset.mcl_ph_trajectory_cache import MCLPHTrajectoryCache,cache_identity,direct_source_identity,require_direct_source_rows
            expected=cache_identity(seed=42,noise_sigma=.03,mask_ratio=.3,global_batch=1008,sample_index_split='train',direct_source=direct_source_identity(source))
            cache=MCLPHTrajectoryCache(CACHE,expected=expected,verify_checksums=False)
            if not cache.complete or cache.total_positions!=5040000: raise ValueError('MCL cache incomplete')
            require_direct_source_rows(CACHE,source)
            stats=load_geometric_statistics(STATS);validate_direct_statistics_source(STATS,stats,source,seed=42,sigma=.03,ratio=.3)
        set_global_seed(42);trainer=Trainer(cfg).to(device)
        optimizer=torch.optim.AdamW(trainer.parameters(),lr=2e-4,weight_decay=0)
        module=torch.nn.parallel.DistributedDataParallel(trainer,device_ids=[rank],find_unused_parameters=True)
        set_global_seed(42+rank);stream=OrderedSampleStream(len(source),42)
        if rank==0: write(folder/'runtime.json',dict(status='RUNNING',config=cfg,start=now(),rank_count=2,accumulation=6,physical_gpus=physical,rendezvous=dict(endpoint=os.environ.get('MASTER_ADDR'),port=os.environ.get('MASTER_PORT'),run_id=os.environ.get('TORCHELASTIC_RUN_ID'))))
        for step in range(5000):
            attempted=step+1
            if rank==0: write(folder/'consumption.json',dict(starts=1,attempted_updates=attempted,completed_updates=completed,status='PREPARING_UPDATE'))
            prepared=[]
            for offset in range(6):
                rows=[];mcls=[]
                for local in range(84):
                    position=step*1008+offset*2*84+rank*84+local
                    i=stream.index_at(position);key=source.samples[i][0].hex()
                    rows.append(prepare_source_sample(source,i,seed=42,key=key,position=position,sigma=.03,ratio=.3,third_task='none',opt_mode='GLT_FUSION_R1',geometry_mode='finite_incident'))
                    if trainer.aux is not None:
                        top,tri,text,static,_=pretrain_source_fields(source,i)
                        mcls.append(prepare_mcl(top,tri,text,static,seed=42,key=key,position=position,trajectory=cache[position],statistics=stats))
                data,labels=pretrain_collate(rows);prepared.append((data,labels,pack_mcl(mcls) if mcls else None))
            counts=torch.zeros(3,device=device);extra=torch.zeros(2,device=device);mc=torch.zeros(1,device=device)
            for data,labels,mcl in prepared:
                masked=torch.bincount(data.canonical_graph_index[labels['atom_mask']],minlength=data.graph_available.numel())
                centers=torch.bincount(data.bond_batch[data.bond_length_weight>0],minlength=data.graph_available.numel())
                counts+=torch.tensor([(masked>0).sum(),((centers>0)&data.geometry_valid).sum(),0],device=device)
                extra[0]+=alignment_local_anchor_count(labels['align_identity'],labels['align_valid'],device=device)
                if mcl is not None: mc[0]+=alignment_local_anchor_count(labels['align_identity'],labels['align_valid'].bool()&mcl.valid,device=device)
            dist.all_reduce(counts);dist.all_reduce(extra);dist.all_reduce(mc)
            lr=scheduled_lr(step,lr=2e-4,warmup_steps=2000,schedule_total_steps=20000,end_lr=1e-9)
            for group in optimizer.param_groups: group['lr']=lr
            optimizer.zero_grad(set_to_none=True);attempted=step+1
            if rank==0: write(folder/'consumption.json',dict(starts=1,attempted_updates=attempted,completed_updates=completed,status='RUNNING'))
            totals=torch.zeros(3,device=device);ex=torch.zeros(2,device=device);mt=torch.zeros(1,device=device);reports=[]
            for offset,(data,labels,mcl) in enumerate(prepared):
                with module.no_sync() if offset<5 else nullcontext():
                    data=data.to(device);labels=move_labels(labels,device);mcl=mcl.to(device) if mcl is not None else None
                    with torch.autocast('cuda',dtype=torch.bfloat16):
                        result=module(data,labels,mcl)
                        loss=global_objective(result['sums'],counts,2,(1.,1.,0.))
                        weight=cfg['align_weight']*min((step+1)/1000,1.)
                        loss+=global_objective(result['extra_sums'],extra,2,(weight,0.))
                        if mcl is not None:
                            loss+=global_objective(result['mcl_sum'].reshape(1),mc,2,(.1*min((step+1)/1000,1.),))
                    if not torch.isfinite(loss): raise FloatingPointError('nonfinite PT objective')
                    loss.backward()
                totals+=result['sums'].detach();ex+=result['extra_sums'].detach()
                if mcl is not None: mt+=result['mcl_sum'].detach();reports.append(result['mcl_report'])
            grad=float(nn.utils.clip_grad_norm_(trainer.parameters(),1.,error_if_nonfinite=True));optimizer.step();completed=step+1
            dist.all_reduce(totals);dist.all_reduce(ex);dist.all_reduce(mt)
            rec=dict(step=completed,rank=rank,losses=(totals/counts.clamp_min(1)).tolist(),extra_losses=(ex/extra.clamp_min(1)).tolist(),mcl_loss=(mt/mc.clamp_min(1)).tolist(),route=reports,grad_norm=grad,lr=lr)
            print(__import__('json').dumps(rec),flush=True)
            with (folder/f'rank{rank}.jsonl').open('a') as h: h.write(__import__('json').dumps(rec)+'\n')
            if rank==0: write(folder/'consumption.json',dict(starts=1,attempted_updates=attempted,completed_updates=completed,status='RUNNING'))
        if rank==0:
            save_checkpoint(folder/'deploy_05000.pt',export(trainer))
            save_checkpoint(folder/'training_final.pt',dict(model=trainer.state_dict(),optimizer=optimizer.state_dict(),config=cfg,step=5000))
            write(folder/'runtime.json',dict(status='PASS',config=cfg,completed_steps=5000,end=now(),pt_validation_forwards=0))
            write(folder/'consumption.json',dict(starts=1,attempted_updates=5000,completed_updates=5000,status='COMPLETED'))
        dist.barrier()
    except BaseException as e:
        write(folder/f'failure_rank{rank}.json',dict(error=repr(e),traceback=__import__('traceback').format_exc(),attempted=attempted,completed=completed,retry=False,time=now()))
        raise
    finally:
        if cache is not None: cache.close()
        if source is not None: source.close()
        dist.destroy_process_group()

def bind_t0():
    cfg=read(T0/'initialization/P_ALIGN.json');original=read(UA/'initialization/P_ALIGN.json')
    fixed=('fusion_mode','use_md200','atom_ce_input','align_enabled','local_enabled','third_task','fusion_revision','opt_mode','geometry_head_norm','seed','microbatch','global_batch','save_every','lr','warmup_steps','schedule_total_steps','end_lr','weight_decay','amp_dtype','atom_mask_ratio','noise_sigma','loss_weights','sample_index_artifact','sample_index_split','geometry_revision','geometry_mode','angle_value_enabled','readout','align_revision','align_init_seed','align_temperature','align_weight','align_warmup_steps','fusion_gate')
    if any(cfg.get(k)!=original[k] for k in fixed) or cfg['expected_world_size']!=2 or cfg['max_optimizer_steps']!=20000:
        raise ValueError('T0 science/control config mismatch')
    a=torch.load(cfg['initial_state'],map_location='cpu',weights_only=False)
    b=torch.load(original['initial_state'],map_location='cpu',weights_only=False)
    if a['step']!=0 or b['step']!=0 or set(a['model'])!=set(b['model']): raise ValueError('T0 common step0 names mismatch')
    for k,v in a['model'].items():
        if v.shape!=b['model'][k].shape or v.dtype!=b['model'][k].dtype or not torch.equal(v,b['model'][k]): raise ValueError('T0 common step0 mismatch '+k)
    return dict(control='two_rank_step5000',config=cfg,common_initial_tensors_equal=True,initial=str(cfg['initial_state']),source=str(T0/'pretrain/P_ALIGN/deploy_05000.pt'),time=now())
