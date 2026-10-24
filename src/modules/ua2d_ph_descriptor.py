"""Isolated PTSCALE/P10K plus412 PH: zero-residual concat or channel gate."""
import torch
from torch import nn
from src.modules.glt_dual import Pure2DGLTModel,strict_module_state

ARMS=('CONCAT_PH','GATE_PH')
SCHEMA='ua2d-ph-descriptor-best-r1'
ARCHITECTURE=dict(o8_layers=6,dimension=512,hop=2,attention_denominator=512,readout='mean_norm2',
 ph='412-128-GELU-128-LayerNorm128;mask after encoder',head='UA512->512 plus zero PH128->512(no bias); GELU-dropout0.1-1',
 gate='640-64-GELU-128;2sigmoid;zero-last',branch_seed=20261023,ph_input='train-valid log1p+standardized 412 descriptors')


def new_branches():
    # The CPU fork isolates ALL added initialization from the existing training stream.
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(20261023)
        fp=nn.Sequential(nn.Linear(412,128),nn.GELU(),nn.Linear(128,128),nn.LayerNorm(128))
        projection=nn.Linear(128,512,bias=False);nn.init.zeros_(projection.weight)
        gate=nn.Sequential(nn.Linear(640,64),nn.GELU(),nn.Linear(64,128))
        nn.init.zeros_(gate[-1].weight);nn.init.zeros_(gate[-1].bias)
    return fp,projection,gate


class PHModel(Pure2DGLTModel):
    def __init__(self,arm):
        if arm not in ARMS:raise ValueError('unregistered PH fusion')
        super().__init__(dropout=0.,attention_scale_denominator=512)
        self.head[2].p=.1;self.stage='head';self.arm=arm
        self.ph_encoder,self.ph_projection,gate=new_branches()
        if arm=='GATE_PH':self.gate=gate

    def train(self,mode=True):
        super().train(mode)
        if mode and self.stage=='head':self.o8.eval();self.norm2.eval()
        return self

    def predict_features(self,ua,ph,valid):
        ph=ph.float()
        if ph.ndim!=2 or ph.shape[1]!=412:raise ValueError('PH feature shape mismatch')
        z=self.ph_encoder(ph)*valid.to(ph.dtype).reshape(-1,1)
        if self.arm=='GATE_PH':z=z*(2*torch.sigmoid(self.gate(torch.cat([ua,z],dim=-1))))
        value=self.head[0](ua)+self.ph_projection(z)
        return self.head[3](self.head[2](self.head[1](value))).view(-1)

    def forward(self,data):return self.predict_features(self.features(data),data.ph_features,data.ph_valid)


def validate_parent(parent):
    from src.modules.ua2d_galformer_pretrain import package_attention
    c=parent.get('config',{});e=parent.get('execution',{})
    if (parent.get('plan')!='UA2D-PTSCALE-20261022-01' or parent.get('revision')!='r1'
        or parent.get('schema')!='ua2d-ptscale-pretrain-r1' or parent.get('step')!=10000
        or c.get('strategy')!='mask_align' or c.get('fp_enabled') is not True or c.get('fusion')!='none'
        or c.get('contrast_weight')!=.1 or c.get('fp_weight')!=.1 or c.get('weight_warmup')!=1000
        or e.get('plan')!='UA2D-PTSCALE-20261022-01' or parent.get('eligible_rows')!=910756
        or package_attention(parent)['scale_denominator']!=512):raise ValueError('PTSCALE true-step10000 parent identity mismatch')
    return parent


def initialize(parent,head,common,arm):
    validate_parent(parent);model=PHModel(arm)
    for name in ('o8','norm2'):
        prefix=name+'.';strict_module_state(getattr(model,name),{k[len(prefix):]:v for k,v in parent['state_dict'].items() if k.startswith(prefix)})
    strict_module_state(model.head,head)
    strict_module_state(model.ph_encoder,common['ph_encoder']);strict_module_state(model.ph_projection,common['ph_projection'])
    if arm=='GATE_PH':strict_module_state(model.gate,common['gate'])
    return model


def optimizer(model,stage):
    if stage not in ('head','joint'):raise ValueError('unknown FT stage')
    model.stage=stage
    for name,p in model.named_parameters():p.requires_grad_(stage=='joint' or not name.startswith(('o8.','norm2.')))
    groups=[]
    if stage=='joint':groups.append(dict(name='backbone',params=[p for n,p in model.named_parameters() if n.startswith(('o8.','norm2.'))],lr=1e-4))
    groups.append(dict(name='head_ph_gate',params=[p for n,p in model.named_parameters() if not n.startswith(('o8.','norm2.'))],lr=3e-4 if stage=='head' else 1e-3))
    ids=[id(p) for group in groups for p in group['params']]
    if len(ids)!=len(set(ids)) or set(ids)!={id(p) for p in model.parameters() if p.requires_grad}:raise ValueError('optimizer coverage conflict')
    evidence=[dict(name=g['name'],lr=g['lr'],parameters=sum(p.numel() for p in g['params']),tensors=len(g['params'])) for g in groups]
    return torch.optim.AdamW(groups,weight_decay=.02),evidence


def restore(saved,expected):
    if saved.get('schema')!=SCHEMA or saved.get('config')!=expected or saved.get('architecture')!=ARCHITECTURE:
        raise ValueError('complete fusion checkpoint/config mismatch')
    model=PHModel(expected['arm']);strict_module_state(model,saved['state_dict']);return model
