"""Common zero-initialized residual head and sign-invariant spectral readout."""
import math
import torch
from torch import nn
ARMS=('C_ANCHOR','S_STAT','S_FIXED','S_HMR')
TRAIN_ARMS=ARMS[1:]

def logit(x):return math.log(x/(1-x))
class Residual(nn.Module):
    def __init__(self,arm,stats):
        super().__init__()
        if arm not in TRAIN_ARMS:raise ValueError('candidate identity')
        self.arm=arm
        self.filter_raw=nn.Parameter(torch.tensor([[0.,logit((4-.05)/(16-.05)),logit((t-.0625)/(64-.0625))] for t in (.25,1,4,16)]),requires_grad=arm=='S_HMR')
        self.head=nn.Sequential(nn.Linear(916,256),nn.GELU(),nn.Dropout(.1),nn.Linear(256,64),nn.GELU(),nn.Dropout(.1),nn.Linear(64,1))
        nn.init.zeros_(self.head[-1].weight);nn.init.zeros_(self.head[-1].bias)
        for key in ('z_mean','z_scale','stat_mean','stat_scale','spec_mean','spec_scale'):self.register_buffer(key,torch.as_tensor(stats[key],dtype=torch.float32))
    def spectral(self,b):
        mu=8*torch.tanh(self.filter_raw[:,0]);sig=.05+(16-.05)*torch.sigmoid(self.filter_raw[:,1]);t=.0625+(64-.0625)*torch.sigmoid(self.filter_raw[:,2])
        lam=b['lam'][:,None,:];g=torch.exp(-((lam-mu[None,:,None])/sig[None,:,None])**2-t[None,:,None]*lam)*b['mode_mask'][:,None,:]
        u=torch.einsum('bk,bjk,bkc->bjc',b['a'],g,b['C'])/b['Ac'][:,None,None];q=torch.einsum('bjk,bkc->bjc',g,b['C'].square())/b['A'][:,None,None]
        return torch.cat((u,q),dim=2).flatten(1)
    def forward(self,b):
        stat=(b['stat']-self.stat_mean)/self.stat_scale;spec=(self.spectral(b)-self.spec_mean)/self.spec_scale
        if self.arm=='S_STAT':spec=torch.zeros_like(spec)
        mask=b['mask'][:,None].bool();stat=torch.where(mask,stat,0.);spec=torch.where(mask,spec,0.)
        z=(b['z']-self.z_mean)/self.z_scale;x=torch.cat((z,b['availability'],b['yb'][:,None],stat,spec),dim=1)
        delta=self.head(x).flatten();return torch.where(b['mask'].bool(),b['yb']+delta,b['yb'])

def fit_stats(cache,surface,train):
    import numpy as np
    from src.dataset.ua3d_hmr_r1 import fixed_descriptor
    ids=torch.as_tensor(train,dtype=torch.long);rows=cache['indices'][ids];valid=surface['mask'][rows].bool()
    if not valid.any():raise ValueError('no valid training surfaces')
    def stats(x):
        x=x.double();mean=x.mean(0);std=x.std(0,unbiased=False);std=torch.where(std==0,torch.ones_like(std),std);return mean.tolist(),std.tolist()
    zm,zs=stats(cache['z'][ids]);sm,ss=stats(surface['stat'][rows][valid]);spec=[]
    for r in rows[valid].tolist():spec.append(fixed_descriptor({k:surface[k][r].double().numpy() for k in ('lam','C','a','A','Ac','mode_mask')}))
    pm,ps=stats(torch.from_numpy(np.array(spec)))
    return dict(z_mean=zm,z_scale=zs,stat_mean=sm,stat_scale=ss,spec_mean=pm,spec_scale=ps,fit='train valid surface only for stat/spec; all train for z',train_rows=rows.tolist())
