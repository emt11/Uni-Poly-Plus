"""Matched residual heads with fixed spectra and rank-32 add/bilinear branches."""
import math
import torch
from torch import nn
from src.modules.ua3d_hmr_r2 import Residual as FixedMath
TRAIN_ARMS=('N_BASE','N_ADD','N_BILIN')
class Readout(nn.Module):
    spectral=FixedMath.spectral
    def __init__(self,arm,stats,fold):
        super().__init__()
        if arm not in TRAIN_ARMS:raise ValueError('new readout arm')
        self.arm=arm
        def logit(x):return math.log(x/(1-x))
        self.filter_raw=nn.Parameter(torch.tensor([[0.,logit((4-.05)/(16-.05)),logit((t-.0625)/(64-.0625))] for t in (.25,1,4,16)]),requires_grad=False)
        self.head=nn.Sequential(nn.Linear(916,256),nn.GELU(),nn.Dropout(.1),nn.Linear(256,64),nn.GELU(),nn.Dropout(.1),nn.Linear(64,1));nn.init.zeros_(self.head[-1].weight);nn.init.zeros_(self.head[-1].bias)
        for name in ('z_mean','z_scale','stat_mean','stat_scale','spec_mean','spec_scale'):self.register_buffer(name,torch.as_tensor(stats[name],dtype=torch.float32))
        if arm!='N_BASE':
            with torch.random.fork_rng(devices=[]):
                # CPU-only seeding: torch.manual_seed would also change CUDA RNG.
                torch.random.default_generator.manual_seed(2026110301+fold)
                self.U=nn.Linear(768,32,bias=False);self.V=nn.Linear(144,32,bias=False);self.LN_z=nn.LayerNorm(32,eps=1e-5);self.LN_s=nn.LayerNorm(32,eps=1e-5);self.W=nn.Linear(32,1,bias=False);nn.init.zeros_(self.W.weight)
    def branch(self,z,s):
        u=self.LN_z(self.U(z));v=self.LN_s(self.V(s));h=(u+v)/math.sqrt(2) if self.arm=='N_ADD' else u*v
        return self.W(h).flatten()
    def inputs(self,b):
        mask=b['mask'].bool();clean={}
        for name in ('lam','C','a','mode_mask','stat'):
            value=b[name];shape=(len(mask),)+(1,)*(value.ndim-1);clean[name]=torch.where(mask.reshape(shape),value,torch.zeros_like(value))
        for name in ('A','Ac'):clean[name]=torch.where(mask,b[name],torch.ones_like(b[name]))
        checks=[torch.isfinite(clean[k]).all() for k in clean]+[torch.isfinite(b[k]).all() for k in ('z','availability','yb')]
        if not torch.stack(checks).all():raise FloatingPointError('nonfinite active input')
        stat=(clean['stat']-self.stat_mean)/self.stat_scale;spec=(self.spectral(clean)-self.spec_mean)/self.spec_scale;stat=torch.where(mask[:,None],stat,0.);spec=torch.where(mask[:,None],spec,0.)
        if self.arm=='N_BASE':stat=torch.zeros_like(stat);spec=torch.zeros_like(spec)
        z=(b['z']-self.z_mean)/self.z_scale;s=torch.cat((stat,spec),1);x=torch.cat((z,b['availability'],b['yb'][:,None],s),1)
        return x,z,s
    def forward(self,b):
        x,z,s=self.inputs(b);delta=self.head(x).flatten()
        if self.arm!='N_BASE':delta=delta+self.branch(z,s)
        return torch.where(b['mask'].bool(),b['yb']+delta,b['yb'])
