"""Isolated 2-hop B_FP transfer: zero residual statistics and fixed L2-SP."""
import math
import torch
from torch import nn
from src.modules.glt_dual import Pure2DGLTModel,mean_pool,strict_module_state
from src.modules.ua2d_galformer_pretrain import package_attention,SCHEMA as PT_SCHEMA

ARMS=('R_MEAN','R_STD','A_SP','R_STD_SP')
SCHEMA='ua2d-transfer-complete-ft-r1'
BRANCH_SEED=20261022
ARCHITECTURE=dict(schema=SCHEMA,layers=6,dimension=512,max_hop=2,attention_denominator=512,
    residual='LayerNorm512-Linear512x128-GELU-Linear128x512-zero-last',branch_seed=BRANCH_SEED,
    std='population;sqrt(v+1e-6)-sqrt(1e-6);FP32',anchor_coefficient=.5e-3,anchor_scope=['o8','norm2'])


def branch():
    # Forking CPU RNG prevents the extra module from consuming any common-head
    # initialization or training random stream. All three residual arms match.
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(BRANCH_SEED)
        module=nn.Sequential(nn.LayerNorm(512),nn.Linear(512,128),nn.GELU(),nn.Linear(128,512))
        nn.init.zeros_(module[-1].weight);nn.init.zeros_(module[-1].bias)
    return module


def atom_statistics(atoms,index,count):
    atoms=atoms.float();index=index.long()
    mean=mean_pool(atoms,index,count)
    variance=mean_pool((atoms-mean[index]).square(),index,count)
    std=torch.sqrt(variance+1e-6)-math.sqrt(1e-6)
    return mean,std


class TransferModel(Pure2DGLTModel):
    def __init__(self,arm):
        if arm not in ARMS:raise ValueError('undeclared transfer arm')
        super().__init__(dropout=0.,attention_scale_denominator=512)
        self.head[2].p=.1;self.arm=arm;self.stage='head';self._anchor=None
        if arm!='A_SP':self.residual=branch()

    def train(self,mode=True):
        super().train(mode)
        if mode and self.stage=='head':self.o8.eval();self.norm2.eval()
        return self

    def features(self,data):
        atoms,_=self.o8(data)
        count=data.graph_available.numel()
        if self.arm in ('R_STD','R_STD_SP'):mean,std=atom_statistics(atoms,data.canonical_graph_index,count)
        else:mean=mean_pool(atoms.float(),data.canonical_graph_index,count)
        if self.arm!='A_SP':
            value=std if self.arm in ('R_STD','R_STD_SP') else mean
            increment=self.residual(value)*data.graph_available.to(mean.dtype).unsqueeze(-1)
            mean=mean+increment
        return self.norm2(mean)

    def forward(self,data):return self.head(self.features(data)).view(-1)

    def cache_anchor(self):
        if self.arm not in ('A_SP','R_STD_SP'):return
        if self._anchor is not None:raise ValueError('theta0 may be cached only once')
        self._anchor={n:p.detach().float().clone() for n,p in self.named_parameters() if n.startswith(('o8.','norm2.'))}
        if not self._anchor or any(x.requires_grad for x in self._anchor.values()):raise ValueError('invalid fixed theta0')

    def anchor_loss(self):
        if self.stage!='joint' or self.arm not in ('A_SP','R_STD_SP'):
            return next(self.parameters()).new_zeros((),dtype=torch.float32)
        if self._anchor is None:raise ValueError('missing once-cached theta0')
        terms=[(p.float()-self._anchor[n]).square().sum() for n,p in self.named_parameters() if n in self._anchor]
        return .5e-3*torch.stack(terms).sum()


def initialize(parent,head,residual_state,arm):
    if (parent.get('schema')!=PT_SCHEMA or parent.get('step')!=5000
        or parent.get('config',{}).get('strategy')!='mask_align' or parent['config'].get('fp_enabled') is not True
        or parent.get('execution',{}).get('revision')!='r4'
        or parent.get('execution',{}).get('plan')!='UA2D-PRETRAIN-FP-20261019-01'
        or package_attention(parent)['scale_denominator']!=512):raise ValueError('B_FP PT parent identity mismatch')
    model=TransferModel(arm)
    for name in ('o8','norm2'):
        prefix=name+'.';strict_module_state(getattr(model,name),{k[len(prefix):]:v for k,v in parent['state_dict'].items() if k.startswith(prefix)})
    strict_module_state(model.head,head)
    if arm!='A_SP':strict_module_state(model.residual,residual_state)
    return model


def optimizer(model,stage):
    if stage not in ('head','joint'):raise ValueError('invalid FT stage')
    model.stage=stage
    for n,p in model.named_parameters():p.requires_grad_(stage=='joint' or n.startswith(('head.','residual.')))
    groups=[]
    if stage=='joint':groups.append(dict(name='backbone',params=[p for n,p in model.named_parameters() if p.requires_grad and n.startswith(('o8.','norm2.'))],lr=1e-4))
    groups.append(dict(name='head_and_residual',params=[p for n,p in model.named_parameters() if p.requires_grad and n.startswith(('head.','residual.'))],lr=3e-4 if stage=='head' else .001))
    ids=[id(p) for g in groups for p in g['params']]
    if len(ids)!=len(set(ids)) or set(ids)!={id(p) for p in model.parameters() if p.requires_grad}:raise ValueError('optimizer parameter coverage mismatch')
    evidence=[dict(name=g['name'],lr=g['lr'],tensors=len(g['params']),parameters=sum(p.numel() for p in g['params'])) for g in groups]
    return torch.optim.AdamW(groups,weight_decay=.02),evidence


def restore(saved,expected_config):
    if (saved.get('schema')!=SCHEMA or saved.get('architecture')!=ARCHITECTURE or saved.get('config')!=expected_config
        or saved.get('best_stage') not in ('head','joint')):raise ValueError('transfer FT package mismatch')
    model=TransferModel(expected_config['arm']);strict_module_state(model,saved['state_dict'])
    return model
