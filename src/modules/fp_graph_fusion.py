"""Opt-in graph-level three-candidate fusion; old token fusion stays untouched."""
import math
import torch
from torch import nn


class FPGraphFusion(nn.Module):
    names=('self','geometry','fp')
    dims=(512,512,2048)

    def __init__(self):
        super().__init__()
        # Isolated common initialization does not advance any FT RNG stream.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20261017)
            self.Wq=nn.Linear(512,128,bias=False)
            self.k_proj=nn.ModuleDict((n,nn.Linear(d,128,bias=False)) for n,d in zip(self.names,self.dims))
            self.v_proj=nn.ModuleDict((n,nn.Linear(d,512)) for n,d in zip(self.names,self.dims))
            for m in self.v_proj.values():nn.init.zeros_(m.weight);nn.init.zeros_(m.bias)

    def forward(self,h,fp,geometry_valid,*,constant=False,uniform=False):
        if h.ndim!=2 or h.shape[1]!=1024 or fp.shape!=(len(h),2048):raise ValueError('graph/FP shape mismatch')
        if geometry_valid.dtype!=torch.bool or geometry_valid.shape!=(len(h),):raise ValueError('geometry availability shape/type mismatch')
        if not torch.isfinite(h).all() or not torch.isfinite(fp).all() or not ((fp==0)|(fp==1)).all() or not (fp.sum(1)>0).all():raise ValueError('nonfinite graph or illegal/empty binary FP')
        x=torch.full_like(fp,.5) if constant else fp
        p=x*(math.sqrt(512)/torch.linalg.vector_norm(x,dim=1,keepdim=True))
        a=h[:,:512];b=torch.where(geometry_valid[:,None],h[:,512:],torch.zeros_like(a))
        candidates=(a,b,p);q=self.Wq(a)/math.sqrt(512)
        keys=torch.stack([self.k_proj[n](v) for n,v in zip(self.names,candidates)],1)
        vals=torch.stack([self.v_proj[n](v) for n,v in zip(self.names,candidates)],1)
        active=torch.stack((torch.ones_like(geometry_valid),geometry_valid,torch.ones_like(geometry_valid)),1)
        logits=torch.einsum('bd,bjd->bj',q,keys).masked_fill(~active,-torch.inf)
        attention=torch.softmax(logits,1)
        weights=active.to(h.dtype)/active.sum(1,keepdim=True) if uniform else attention
        vals=torch.where(active[:,:,None],vals,torch.zeros_like(vals))
        new=a+.5*(weights[:,:,None]*vals).sum(1)
        return torch.cat((new,b),1)


class FPGraphHead(nn.Module):
    def __init__(self,regressor,*,constant=False,uniform=False):
        super().__init__();self.fusion=FPGraphFusion();self.regressor=regressor
        self.constant=constant;self.uniform=uniform
    def forward(self,h,fp,valid):
        return self.regressor(self.fusion(h,fp,valid,constant=self.constant,uniform=self.uniform))
