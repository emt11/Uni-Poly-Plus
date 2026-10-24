"""Isolated UA2D-OPT r1 readouts and exact trainability policies."""
import torch
from torch import nn
from src.modules.glt_dual import Pure2DGLTModel, mean_pool, strict_module_state
from src.modules.ua2d_galformer_pretrain import package_attention, SCHEMA as PT_SCHEMA
SCHEMA='ua2d-opt-complete-ft-r1'
ARCHITECTURE=dict(schema=SCHEMA,attention_denominator=512,layers=6,dimension=512,
    mix='p6+.25*tanh(a2)*(p2-p6)+.25*tanh(a4)*(p4-p6)',mix_initial=0.)

class OptModel(Pure2DGLTModel):
    def __init__(self,policy,*,schedule=None):
        if policy not in ('NONE','TOP2','MIX246'):raise ValueError('undeclared policy')
        super().__init__(dropout=0.,attention_scale_denominator=512)
        self.head[2].p=.1;self.policy=policy;self.stage='head'
        if schedule is not None:self.o8=Hop3O8(schedule,dropout=0.)
        if policy=='MIX246':
            self.a2=nn.Parameter(torch.zeros(()));self.a4=nn.Parameter(torch.zeros(()))
    def train(self,mode=True):
        super().train(mode)
        if mode:
            if self.stage=='head':self.o8.eval();self.norm2.eval()
            elif self.policy=='TOP2':
                for m in (self.o8.atom_embedding,self.o8.spd_embedding,self.o8.path_bias,self.o8.bond_bias,*self.o8.layers[:4]):m.eval()
        return self
    def features(self,data):
        if self.policy=='MIX246':
            features={}
            handles=[self.o8.layers[i-1].register_forward_hook(lambda m,x,y,k=i:features.__setitem__(k,y)) for i in (2,4,6)]
            try:atoms,_=self.o8(data)
            finally:
                for h in handles:h.remove()
            available=data.graph_available[data.canonical_graph_index].unsqueeze(-1)
            p={i:mean_pool(features[i]*available,data.canonical_graph_index,len(data.graph_available)) for i in (2,4,6)}
            pooled=p[6]+.25*self.a2.tanh()*(p[2]-p[6])+.25*self.a4.tanh()*(p[4]-p[6])
        else:
            atoms,_=self.o8(data)
            pooled=mean_pool(atoms,data.canonical_graph_index,len(data.graph_available))
        return self.norm2(pooled)
    def forward(self,data):return self.head(self.features(data)).view(-1)


def initialize(parent,head,arm):
    if arm not in ('A_TOP2','A_MIX246'):return initialize_new_parent(parent,head,arm)
    if (parent.get('schema')!=PT_SCHEMA or parent.get('step')!=5000
        or parent.get('config',{}).get('strategy')!='mask_align'
        or parent['config'].get('fp_enabled') is not True
        or parent.get('execution',{}).get('revision')!='r4'
        or parent.get('execution',{}).get('plan')!='UA2D-PRETRAIN-FP-20261019-01'
        or package_attention(parent)['scale_denominator']!=512):raise ValueError('B_FP parent schema mismatch')
    model=OptModel(arm[2:])
    for name in ('o8','norm2'):
        prefix=name+'.';strict_module_state(getattr(model,name),{k[len(prefix):]:v for k,v in parent['state_dict'].items() if k.startswith(prefix)})
    strict_module_state(model.head,head)
    return model


def restore(saved,expected_config):
    if (saved.get('schema')!=SCHEMA or saved.get('architecture')!=ARCHITECTURE
        or saved.get('config')!=expected_config or saved.get('best_stage') not in ('head','joint')):raise ValueError('complete FT schema mismatch')
    model=OptModel(expected_config['policy'],schedule=expected_config['schedule'])
    strict_module_state(model,saved['state_dict']);return model


def optimizer(model,stage):
    if stage not in ('head','joint'):raise ValueError('invalid phase')
    model.stage=stage
    for name,p in model.named_parameters():
        train=name.startswith('head.')
        if stage=='joint':
            train=(model.policy!='TOP2' or name.startswith(('o8.layers.4.','o8.layers.5.','norm2.','head.')))
        p.requires_grad_(train)
    groups=[]
    if stage=='joint':groups.append(dict(name='backbone',params=[p for n,p in model.named_parameters() if p.requires_grad and not n.startswith('head.')],lr=1e-4))
    groups.append(dict(name='head',params=list(model.head.parameters()),lr=3e-4 if stage=='head' else 1e-3))
    ids=[id(p) for g in groups for p in g['params']]
    if len(ids)!=len(set(ids)) or set(ids)!={id(p) for p in model.parameters() if p.requires_grad}:raise ValueError('optimizer coverage mismatch')
    evidence=[dict(name=g['name'],lr=g['lr'],tensors=len(g['params']),parameters=sum(p.numel() for p in g['params'])) for g in groups]
    return torch.optim.AdamW(groups,weight_decay=.02),evidence


class Hop3BondBias(nn.Module):
    def __init__(self):
        super().__init__()
        from src.modules.glt_dual import SharedBondPathBias
        old=SharedBondPathBias()
        self.type_emb=old.type_emb;self.conj_emb=old.conj_emb;self.ring_emb=old.ring_emb;self.stereo_emb=old.stereo_emb
        self.position=nn.Parameter(torch.empty(3,8,8));nn.init.normal_(self.position,std=.02)
    def forward(self,features,mask):
        if features.shape!=(*mask.shape,14) or mask.ndim!=2 or mask.size(1)!=3:raise ValueError('expected hop3 bond chemistry')
        if not torch.isfinite(features).all():raise ValueError('nonfinite hop3 chemistry')
        selected=features[mask]
        if selected.numel() and (not ((selected==0)|(selected==1)).all() or not (selected[:,:5].sum(-1)==1).all() or not (selected[:,7:].sum(-1)==1).all()):raise ValueError('invalid hop3 chemistry')
        z=self.type_emb(features[...,:5].argmax(-1))+self.conj_emb(features[...,5].long())+self.ring_emb(features[...,6].long())+self.stereo_emb(features[...,7:].argmax(-1))
        return (torch.einsum('rki,kij->rkj',z,self.position)*mask.unsqueeze(-1)).sum(1)/mask.sum(1,keepdim=True).clamp_min(1)


from src.modules.glt_dual import BondPathO8
class Hop3O8(BondPathO8):
    def __init__(self,schedule,dropout=.1):
        if len(schedule)!=6 or any(type(h)!=int or h not in (1,2,3) for h in schedule):raise ValueError('invalid six layer schedule')
        super().__init__(dropout=dropout,attention_scale_denominator=512)
        self.schedule=list(schedule)
        from src.modules.mips_local_graph import MIPSSinglePathNodeBias
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20261021)
            self.spd_embedding=nn.Embedding(4,8)
            self.path_bias=MIPSSinglePathNodeBias(512,8,3)
            self.bond_bias=Hop3BondBias()
    def forward(self,data,atom_mask=None,*,return_initial=False):
        if (data.lga_path_index.shape[1]!=4 or data.bond_path_features.shape[1]!=3
            or data.lga_spd.min()<0 or data.lga_spd.max()>3):raise ValueError('hop3 input schema mismatch')
        initial=self.atom_embedding(data,atom_mask=atom_mask)
        bias=self.spd_embedding(data.lga_spd.long())+self.path_bias(initial,data)+self.bond_bias(data.bond_path_features,data.bond_path_mask.bool())
        bias=bias.masked_fill(data.lga_relation_mask.bool().unsqueeze(-1),0)
        source,target=data.lga_edge_index.long();states=initial
        views={}
        for h in set(self.schedule):
            rows=getattr(data,f'opt_hop{h}_rows',None)
            if rows is None or rows.dtype!=torch.long or rows.ndim!=1:
                raise ValueError('missing cached hop candidate rows; online fallback forbidden')
            views[h]=(source[rows],target[rows],bias[rows])
        for h,layer in zip(self.schedule,self.layers):
            states=layer(states,*views[h])
        states=states*data.graph_available[data.canonical_graph_index].unsqueeze(-1)
        return (states,bias,initial) if return_initial else (states,bias)


from src.modules.ua2d_galformer_pretrain import UA2DGalformerPretrainer,PretrainConfig
class OptPretrainer(UA2DGalformerPretrainer):
    def __init__(self,schedule,target,vocab_size):
        super().__init__(PretrainConfig())
        self.o8=Hop3O8(schedule)
        self.target=target;self.vocab_size=vocab_size
        if target=='C':
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(20261021)
                self.atom_head=nn.Sequential(nn.Linear(512,256),nn.GELU(),nn.Linear(256,vocab_size))
        elif target!='E':raise ValueError('unknown CE objective')


def expanded_initialization(out,common_path,heads_path,vocab_size):
    from src.training.glt_dual_runtime import save_checkpoint
    from pathlib import Path
    common=torch.load(common_path,map_location='cpu',weights_only=False)
    heads=torch.load(heads_path,map_location='cpu',weights_only=False)
    if (common.get('schema')!=PT_SCHEMA or common.get('step')!=0 or common.get('seed')!=42
        or heads.get('schema')!=PT_SCHEMA or heads.get('step')!=0 or heads.get('strategy')!='mask_align'):raise ValueError('common step0 identity mismatch')
    old=dict(common['state_dict'],**heads['state_dict'])
    folder=Path(out)/'initialization';folder.mkdir(exist_ok=False)
    for target in ('E','C'):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(42);model=OptPretrainer([3]*6,target,vocab_size)
        state=model.state_dict();provenance={}
        for k,v in state.items():
            if target=='C' and k.startswith('atom_head.'):
                provenance[k]='independent_seed20261021_context_head';continue
            if k in old and old[k].shape==v.shape:
                state[k]=old[k].clone();provenance[k]='common_step0_or_V1_heads';continue
            if k=='o8.spd_embedding.weight':state[k][:3].copy_(old[k]);provenance[k]='first_three_rows_from_common_new_row_seed20261021'
            elif k=='o8.bond_bias.position':state[k][:2].copy_(old[k]);provenance[k]='first_two_slots_from_common_new_slot_seed20261021'
            elif k=='o8.path_bias.projections.3.weight':provenance[k]='new_projection_seed20261021'
            else:raise ValueError('unexpected step0 shape/key mismatch '+k)
        if set(old)-set(state):raise ValueError('dropped step0 tensors')
        strict_module_state(model,state)
        save_checkpoint(folder/f'{target}_step0.pt',dict(schema=SCHEMA,pt_schema=PT_SCHEMA,step=0,target=target,vocab_size=vocab_size,state_dict=state,provenance=provenance,common=str(common_path),heads=str(heads_path),schedule=[3]*6))
    return folder


def initialize_new_parent(parent,head,arm):
    from src.training.ua2d_opt import schedule,policy
    if (parent.get('schema')!=SCHEMA or parent.get('step')!=5000
        or parent.get('schedule')!=schedule(arm) or parent.get('plan')!='UA2D-OPT-20261021-01'
        or parent.get('config')!=__import__('dataclasses').asdict(PretrainConfig())):raise ValueError('new PT package schema mismatch')
    model=OptModel(policy(arm),schedule=schedule(arm))
    for name in ('o8','norm2'):
        prefix=name+'.';strict_module_state(getattr(model,name),{k[len(prefix):]:v for k,v in parent['state_dict'].items() if k.startswith(prefix)})
    strict_module_state(model.head,head);return model
