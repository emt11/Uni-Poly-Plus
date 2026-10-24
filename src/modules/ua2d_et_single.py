"""Local TorchMD-Net v2.6.1 ET adaptation; no external runtime dependency.

Copyright Universitat Pompeu Fabra 2020-2023. MIT license and original sources:
docs/third_party/torchmd_net_v2_6_1/. Project changes: explicit unrestricted
bond/radius/self union, one shared chemical gate on all message pathways,
102 corrupted element categories, fixed Gaussian basis, pure-2D deployment.
"""
import torch
from torch import nn
from torch.nn import functional as F
from src.modules.glt_dual import BondPathO8, Pure2DGLTModel, mean_pool, strict_module_state
from src.modules.glt_dual_pretrain import AlignmentProjection, alignment_loss, alignment_local_anchor_count
from src.modules.ua2d_galformer_pretrain import task_sum, PretrainConfig, SCHEMA as OLD_SCHEMA

PT_SCHEMA = 'ua2d-et-single-pretrain-r1'
FT_SCHEMA = 'ua2d-et-single-pure2d-ft-r1'
ARCHITECTURE = dict(schema=PT_SCHEMA, width=256, layers=6, heads=8, activation='silu',
    distance_influence='both', neighbor_embedding=True, dropout=0., rbf='fixed_gauss50_0_4',
    neighborhood='all_radius4_union_real_heavy_bonds_union_self_no_truncation',
    gate='chemical_and_self_1_else_cosine4_applied_once_each_path',
    neighbor_self=True, periodic_box=False, decoder=[562,256,25755], bridge=[256,512],
    reference='torchmd-net/v2.6.1', o8_attention_denominator=512, downstream='pure2d',activation_checkpoint=True)

class GaussianRBF(nn.Module):
    def __init__(self):
        super().__init__()
        offset = torch.linspace(0.,4.,50)
        self.register_buffer('offset',offset)
        self.register_buffer('coeff',-.5/(offset[1]-offset[0]).square())
    def forward(self,distance): return torch.exp(self.coeff*(distance.float()[:,None]-self.offset).square())

class ETLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.layernorm=nn.LayerNorm(256)
        self.q_proj=nn.Linear(256,256); self.k_proj=nn.Linear(256,256)
        self.v_proj=nn.Linear(256,768); self.o_proj=nn.Linear(256,768)
        self.vec_proj=nn.Linear(256,768,bias=False)
        self.dk_proj=nn.Linear(50,256); self.dv_proj=nn.Linear(50,768)
        for m in self.modules():
            if isinstance(m,nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)
    def forward(self,x,vec,edge,rbf,direction,gate):
        n=len(x); source,target=edge
        normalized=self.layernorm(x)
        q=self.q_proj(normalized).reshape(n,8,32)
        k=self.k_proj(normalized).reshape(n,8,32)
        v=self.v_proj(normalized).reshape(n,8,96)
        v1,v2,v3=self.vec_proj(vec).chunk(3,-1)
        # Vector contraction and all directional arithmetic remain FP32.
        dot=(v1.float()*v2.float()).sum(1)
        dk=F.silu(self.dk_proj(rbf)).reshape(-1,8,32)
        dv=F.silu(self.dv_proj(rbf)).reshape(-1,8,96)
        attention=F.silu((q[target]*k[source]*dk).sum(-1))
        # vector_cutoff=True style: gate the value path once, hence both
        # scalar and vector messages vanish at the nonbonded cutoff.
        value=v[source]*dv*gate[:,None,None]
        scalar,a,b=value.chunk(3,-1)
        scalar=scalar*attention[:,:,None]
        vector=vec[source].reshape(-1,3,8,32).float()*a[:,None].float()+direction[:,:,None,None]*b[:,None].float()
        scalar_sum=scalar.new_zeros(n,8,32).index_add(0,target,scalar).reshape(n,256)
        vector_sum=vector.new_zeros(n,3,8,32).index_add(0,target,vector).reshape(n,3,256)
        o1,o2,o3=self.o_proj(scalar_sum).chunk(3,-1)
        dx=dot*o2.float()+o3.float()
        dvec=v3.float()*o1[:,None].float()+vector_sum
        return dx,dvec

class AtomicET(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding=nn.Embedding(102,256)
        self.neighbor_embedding=nn.Embedding(102,256)
        self.distance_proj=nn.Linear(50,256);self.combine=nn.Linear(512,256)
        self.rbf=GaussianRBF();self.layers=nn.ModuleList([ETLayer() for _ in range(6)])
        self.out_norm=nn.LayerNorm(256)
        for m in (self.distance_proj,self.combine):
            nn.init.xavier_uniform_(m.weight);nn.init.zeros_(m.bias)
    def forward(self,z,positions,edge,chemical):
        source,target=edge
        with torch.autocast(device_type=positions.device.type,enabled=False):
            delta=positions[source].float()-positions[target].float()
            distance=delta.square().sum(-1).sqrt()
            direction=delta/distance.clamp_min(1e-8)[:,None]
            direction=torch.where((source!=target)[:,None],direction,torch.zeros_like(direction))
            gate=torch.where(chemical | (source==target),torch.ones_like(distance),
                             .5*(torch.cos(distance.clamp(max=4.)*torch.pi/4.)+1.))
            rbf=self.rbf(distance)
        x=self.embedding(z)
        message=self.neighbor_embedding(z)[source]*self.distance_proj(rbf)*gate[:,None]
        incoming=message.new_zeros(len(z),256).index_add(0,target,message)
        x=self.combine(torch.cat((x,incoming),-1))
        vec=x.new_zeros(len(z),3,256).float()
        for layer in self.layers:
            if self.training and torch.is_grad_enabled():
                from torch.utils.checkpoint import checkpoint
                dx,dvec=checkpoint(layer,x,vec,edge,rbf,direction,gate,use_reentrant=False,preserve_rng_state=False)
            else:
                dx,dvec=layer(x,vec,edge,rbf,direction,gate)
            x=x+dx;vec=vec+dvec
        return self.out_norm(x),vec

class ETPretrainer(nn.Module):
    def __init__(self):
        super().__init__();self.config=PretrainConfig();self.completed_updates=0
        self.o8=BondPathO8(.1,attention_scale_denominator=512)
        self.norm2=nn.LayerNorm(512);self.norm3=nn.LayerNorm(512)
        self.proj2=AlignmentProjection();self.proj3=AlignmentProjection()
        self.atom_head=nn.Sequential(nn.Linear(512,256),nn.GELU(),nn.Linear(256,101))
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20261020)
            self.et=AtomicET();self.bridge=nn.Linear(256,512)
            self.bond_head=nn.Sequential(nn.Linear(562,256),nn.SiLU(),nn.Linear(256,25755))
    @property
    def weights(self): return dict(mask_2d=1.,mask_3d=1.,contrast=.1*min(1.,self.completed_updates/1000))
    def target_counts(self,data,labels):
        graphs=len(data.geometry_valid)
        def count(indices): return torch.bincount(indices,minlength=graphs).gt(0).sum().float()
        return dict(mask_2d=count(data.canonical_graph_index[labels['atom_selected']]),
            mask_3d=count(data.bond_batch[labels['token_selected'] & (data.bond_length_weight>0)]),
            contrast=alignment_local_anchor_count(labels['identity'],data.geometry_valid & data.graph_available,device=data.mips_x.device))
    def decode(self,h,left,right,distance):
        return self.bond_head(torch.cat((h[left]+h[right],(h[left]-h[right]).abs(),self.et.rbf(distance)),-1))
    def forward(self,data,labels):
        graphs=len(data.geometry_valid);atoms,_=self.o8(data)
        scalar,vector=self.et(labels['et_z'],labels['et_pos'],labels['et_edge'],labels['et_chemical'])
        center=labels['et_center'];index=labels['et_graph']
        pooled=mean_pool(scalar[center],index[center],graphs)
        graph2=self.norm2(mean_pool(atoms,data.canonical_graph_index,graphs))
        graph3=self.norm3(self.bridge(pooled))
        sums={};counts={}
        chosen=labels['atom_selected']
        loss=F.cross_entropy(self.atom_head(atoms[chosen]).float(),labels['atom_label'][chosen],reduction='none')
        sums['mask_2d'],counts['mask_2d']=task_sum(loss,data.canonical_graph_index[chosen],graphs)
        selected=labels['token_selected'] & (data.bond_length_weight>0)
        left,right=labels['et_bond'][:,selected]
        with torch.autocast(device_type=atoms.device.type,enabled=False):
            distance=(labels['et_pos'][left].float()-labels['et_pos'][right].float()).norm(dim=-1)
        logits=self.decode(scalar,left,right,distance)
        loss=F.cross_entropy(logits.float(),labels['token_label'][selected],reduction='none')
        sums['mask_3d'],counts['mask_3d']=task_sum(loss,data.bond_batch[selected],graphs,data.bond_length_weight[selected])
        value,count=alignment_loss(self.proj2(graph2),self.proj3(graph3),labels['identity'],
            data.geometry_valid & data.graph_available,temperature=.1)
        sums['contrast']=2*value+vector.sum()*0;counts['contrast']=count
        return dict(sums=sums,counts=counts,diagnostics=dict(atomic_nodes=len(scalar),atomic_edges=labels['et_edge'].shape[1]))

def initialize_common(model,package):
    if package.get('schema')!=OLD_SCHEMA or package.get('step')!=0 or package.get('seed')!=42:
        raise ValueError('common_step0 identity/schema mismatch')
    expected=model.state_dict();copied=[]
    for prefix in ('o8.','norm2.','norm3.','proj2.','proj3.','atom_head.'):
        local={k[len(prefix):]:v for k,v in package['state_dict'].items() if k.startswith(prefix)}
        strict_module_state(getattr(model,prefix[:-1]),local);copied.extend(prefix+k for k in local)
    return copied

def deployment_package(model,step):
    return dict(schema=PT_SCHEMA,architecture=ARCHITECTURE,step=int(step),config=dict(arm='ET_SINGLE'),
        state_dict={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},prediction_paths=['pure_2d'])

def restore_downstream(package,head):
    if package.get('schema')!=PT_SCHEMA or package.get('step')!=5000 or package.get('architecture')!=ARCHITECTURE:
        raise ValueError('ET pretrain complete schema mismatch')
    model=Pure2DGLTModel(dropout=0.,attention_scale_denominator=512);model.head[2].p=.1
    for name in ('o8','norm2'):
        prefix=name+'.'
        strict_module_state(getattr(model,name),{k[len(prefix):]:v for k,v in package['state_dict'].items() if k.startswith(prefix)})
    strict_module_state(model.head,head)
    return model

def restore_ft(saved,cfg):
    if saved.get('schema')!=FT_SCHEMA or saved.get('config')!=cfg: raise ValueError('ET pure2D FT schema/config mismatch')
    model=Pure2DGLTModel(dropout=0.,attention_scale_denominator=512);model.head[2].p=.1
    strict_module_state(model,saved['state_dict']);return model

def ft_optimizer(model,stage):
    for name,p in model.named_parameters(): p.requires_grad_(stage=='joint' or name.startswith('head.'))
    groups=[]
    if stage=='joint': groups.append(dict(name='backbone',params=[p for m in (model.o8,model.norm2) for p in m.parameters()],lr=1e-4))
    groups.append(dict(name='head',params=list(model.head.parameters()),lr=3e-4 if stage=='head' else 1e-3))
    evidence=[dict(name=g['name'],lr=g['lr'],parameters=sum(p.numel() for p in g['params'])) for g in groups]
    return torch.optim.AdamW(groups,weight_decay=.02),evidence


def restore_pretrain(package):
    if (package.get('schema')!=PT_SCHEMA or package.get('architecture')!=ARCHITECTURE
        or type(package.get('step')) is not int or not 0<=package['step']<=5000):
        raise ValueError('complete ET pretrain schema mismatch')
    model=ETPretrainer();strict_module_state(model,package['state_dict'])
    model.completed_updates=package['step'];return model
