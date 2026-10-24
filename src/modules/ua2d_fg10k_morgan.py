"""Isolated F_G10K plus true Morgan: shared zero-residual concat or channel gate."""
import torch
from torch import nn
from src.modules.glt_dual import Pure2DGLTModel,strict_module_state

ARMS=('CONCAT_FP','GATE_FP')
SCHEMA='ua2d-fg10k-morgan-best-r1'
ARCHITECTURE=dict(o8_layers=6,dimension=512,hop=2,attention_denominator=512,readout='mean_norm2',
 fp='2048-128-GELU-128-LayerNorm128',head='UA512->512 plus zero FP128->512(no bias); GELU-dropout0.1-1',
 gate='640-64-GELU-128;2sigmoid;zero-last',branch_seed=20261023,fp_input='real binary Morgan, no scaling')


def new_branches():
    # The CPU fork isolates ALL added initialization from the existing training stream.
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(20261023)
        fp=nn.Sequential(nn.Linear(2048,128),nn.GELU(),nn.Linear(128,128),nn.LayerNorm(128))
        projection=nn.Linear(128,512,bias=False);nn.init.zeros_(projection.weight)
        gate=nn.Sequential(nn.Linear(640,64),nn.GELU(),nn.Linear(64,128))
        nn.init.zeros_(gate[-1].weight);nn.init.zeros_(gate[-1].bias)
    return fp,projection,gate


class MorganModel(Pure2DGLTModel):
    def __init__(self,arm):
        if arm not in ARMS:raise ValueError('unregistered Morgan fusion')
        super().__init__(dropout=0.,attention_scale_denominator=512)
        self.head[2].p=.1;self.stage='head';self.arm=arm
        self.fp_encoder,self.fp_projection,gate=new_branches()
        if arm=='GATE_FP':self.gate=gate

    def train(self,mode=True):
        super().train(mode)
        if mode and self.stage=='head':self.o8.eval();self.norm2.eval()
        return self

    def predict_features(self,ua,fp):
        fp=fp.float()
        if fp.ndim!=2 or fp.shape[1]!=2048:raise ValueError('Morgan feature shape mismatch')
        z=self.fp_encoder(fp)
        if self.arm=='GATE_FP':z=z*(2*torch.sigmoid(self.gate(torch.cat([ua,z],dim=-1))))
        value=self.head[0](ua)+self.fp_projection(z)
        return self.head[3](self.head[2](self.head[1](value))).view(-1)

    def forward(self,data):return self.predict_features(self.features(data),data.morgan_fp)


def validate_parent(parent):
    e=parent.get('execution',{});c=parent.get('config',{})
    if (parent.get('plan')!='UA2D-3DSHIFT-20261022-01' or parent.get('revision')!='r1.5-ALL1-PCTRL-GPU2'
        or parent.get('schema')!='ua2d-3dshift-local-geometry-r1' or parent.get('variant')!='F_G10K'
        or parent.get('arm')!='F_G10K' or parent.get('step')!=10000 or parent.get('local_alignment') is not False
        or c.get('objective_policy')!='uniform1_align_mean_v1' or c.get('weight_warmup')!=0
        or any(c.get(k)!=1. for k in ['mask_2d_weight','mask_3d_weight','contrast_weight','fp_weight'])
        or parent.get('architecture',{}).get('image_geometry')!='finite_incident_trimer'
        or parent['architecture'].get('local_alignment') is not False or e.get('world_size')!=1
        or e.get('align_multiplier')!=1. or e.get('physical_gpus')!=[2]
        or c.get('attention_scale_denominator')!=512):raise ValueError('F_G10K read-only parent identity mismatch')
    return parent


def initialize(parent,head,common,arm):
    validate_parent(parent);model=MorganModel(arm)
    for name in ('o8','norm2'):
        prefix=name+'.';strict_module_state(getattr(model,name),{k[len(prefix):]:v for k,v in parent['state_dict'].items() if k.startswith(prefix)})
    strict_module_state(model.head,head)
    strict_module_state(model.fp_encoder,common['fp_encoder']);strict_module_state(model.fp_projection,common['fp_projection'])
    if arm=='GATE_FP':strict_module_state(model.gate,common['gate'])
    return model


def optimizer(model,stage):
    if stage not in ('head','joint'):raise ValueError('unknown FT stage')
    model.stage=stage
    for name,p in model.named_parameters():p.requires_grad_(stage=='joint' or not name.startswith(('o8.','norm2.')))
    groups=[]
    if stage=='joint':groups.append(dict(name='backbone',params=[p for n,p in model.named_parameters() if n.startswith(('o8.','norm2.'))],lr=1e-4))
    groups.append(dict(name='head_fp_gate',params=[p for n,p in model.named_parameters() if not n.startswith(('o8.','norm2.'))],lr=3e-4 if stage=='head' else 1e-3))
    ids=[id(p) for group in groups for p in group['params']]
    if len(ids)!=len(set(ids)) or set(ids)!={id(p) for p in model.parameters() if p.requires_grad}:raise ValueError('optimizer coverage conflict')
    evidence=[dict(name=g['name'],lr=g['lr'],parameters=sum(p.numel() for p in g['params']),tensors=len(g['params'])) for g in groups]
    return torch.optim.AdamW(groups,weight_decay=.02),evidence


def restore(saved,expected):
    if saved.get('schema')!=SCHEMA or saved.get('config')!=expected or saved.get('architecture')!=ARCHITECTURE:
        raise ValueError('complete fusion checkpoint/config mismatch')
    model=MorganModel(expected['arm']);strict_module_state(model,saved['state_dict']);return model
