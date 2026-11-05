"""Fresh BASE and conditional heat propagation; no historical tensor loaders."""
import copy
import torch
from torch import nn
from src.modules import ua3d_base_opt_r6 as base
native=base.native
NATIVE=base.NATIVE
TASKS=base.TASKS
PLAN='UA3D-COND-SPECTRAL-20261104-01'
PT_SCHEMA='ua3d-cond-spectral-pretrain-r2'
FT_SCHEMA='ua3d-cond-spectral-best-r2'
ARMS=('C_FRESH','S_STATIC_GATE','S_COND_GATE','S_STATIC_TOKEN','S_COND_TOKEN','S_COND_NOCROSS')
PT=base.pt_config('P_CURRENT_SHARED',.1,0.)
CONFIG=dict(native=native.CONFIG,revision='r2',times=[.0625,64.],channels=64,readout=164,stat=16,fp_prediction=False)
strict_module_state=base.strict_module_state

class FreshPT(base.BaseOptFuse):
    def __init__(self):
        super().__init__(True,pt=PT)
        # Parent pure constructor seeds modal definitions historically. Reinitialize
        # those modules in the current seed42 stream; no tensor file is read.
        for name in native.modal_modules('TMOL3M'):
            for layer in getattr(self,name).modules():
                if hasattr(layer,'reset_parameters'):layer.reset_parameters()

class Surface(nn.Module):
    def __init__(self,arm):
        super().__init__();self.arm=arm
        self.norms=nn.ModuleList([nn.LayerNorm(256) for _ in range(3)])
        self.condition=nn.Sequential(nn.Linear(771,128),nn.GELU(),nn.Dropout(.1),nn.Linear(128,64))
        nn.init.zeros_(self.condition[-1].weight);nn.init.zeros_(self.condition[-1].bias)
        times=torch.tensor([.25,1.,4.,16.]).repeat_interleave(16)
        self.beta=nn.Parameter(torch.logit((times-.0625)/(64.-.0625)))
        self.W=nn.Linear(16,64,bias=False);self.P=nn.Linear(64,8,bias=False)
        self.norm=nn.LayerNorm(164)
        self.head=nn.Sequential(nn.Linear(180,128),nn.GELU(),nn.Dropout(.1),nn.Linear(128,256))
        self.register_buffer('triangle',torch.triu_indices(8,8));self.last={}
    def forward(self,z,availability,s):
        mask=s['mask'].bool();m=mask[:,None]
        context=[n(v) for n,v in zip(self.norms,z)]
        context[1]=context[1]*availability[:,1:2]
        context=torch.cat((*context,availability.float()),-1)
        if 'STATIC' in self.arm:context=torch.zeros_like(context)
        t=.0625+(64.-.0625)*torch.sigmoid(self.beta+self.condition(context))
        if self.training and torch.is_grad_enabled():t.retain_grad();self.time_tensor=t
        lam=torch.where(m,s['lam'],torch.zeros_like(s['lam']))
        mode=s['mode_mask'].bool()&m
        C=torch.where(m[:,:,None],s['C'],torch.zeros_like(s['C']))
        B=torch.exp(-lam[:,:,None]*t[:,None,:])*self.W(C)*mode[:,:,None]
        positive=mode&(lam>0)
        A=torch.where(mask,s['A'],torch.ones_like(s['A'])).clamp_min(1e-12)
        Ac=torch.where(mask,s['Ac'],torch.ones_like(s['Ac'])).clamp_min(1e-12)
        a=torch.where(m,s['a'],torch.zeros_like(s['a']))
        u=(a[:,:,None]*B).sum(1)/Ac[:,None]
        q=(B.square()*positive[:,:,None]).sum(1)/A[:,None]
        BP=self.P(B)*positive[:,:,None]
        gram=BP.transpose(1,2)@BP/A[:,None,None]
        cross=gram[:,self.triangle[0],self.triangle[1]]
        if self.arm=='S_COND_NOCROSS':cross=torch.zeros_like(cross)
        features=torch.cat((u,q,cross),-1)
        stat=torch.where(m,s['stat'],torch.zeros_like(s['stat']))
        out=self.head(torch.cat((self.norm(features),stat),-1))
        out=torch.where(m,out,torch.zeros_like(out))
        with torch.no_grad():
            self.last=dict(time_mean=t.mean(),time_min=t.min(),time_max=t.max(),time_saturation=((t<=.07)|(t>=63.9)).float().mean(),nonzero_energy=q.mean(),cross_energy=gram.square().mean())
        return out

class Model(base.BaseOptFuse):
    def __init__(self,arm,fold=0):
        if arm not in ARMS:raise ValueError('unregistered arm')
        super().__init__(False,ft=copy.deepcopy(base.DEFAULT_FT));self.recipe_arm=arm
        if arm!='C_FRESH':
            old=self.fusion
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(2026110401+fold)
                self.surface=Surface(arm)
                self.fusion=Gate4()
                self.token=nn.TransformerEncoderLayer(256,4,512,.1,batch_first=True) if 'TOKEN' in arm else None
            with torch.no_grad():
                self.fusion.task_number.load_state_dict(old.task_number.state_dict());self.fusion.task_type.load_state_dict(old.task_type.state_dict())
                # gate input: three z vectors, surface z, task embedding.
                self.fusion.gate[0].weight[:,:768].copy_(old.gate[0].weight[:,:768])
                self.fusion.gate[0].weight[:,1024:].copy_(old.gate[0].weight[:,768:])
                self.fusion.gate[0].bias.copy_(old.gate[0].bias)
                self.fusion.gate[2].weight[:3].copy_(old.gate[2].weight)
                self.fusion.gate[2].bias[:3].copy_(old.gate[2].bias)
    def predict(self,batch):
        z,_,_=self.encode_modalities(batch);available=batch.availability
        if self.recipe_arm=='C_FRESH':f,weights=self.fuse(z,available)
        else:
            with torch.random.fork_rng(devices=[z[0].device]):
                zs=self.surface(z,available,batch.surface)
            available=torch.cat((available,batch.surface['mask'].bool()[:,None]),-1)
            tokens=torch.stack((*z,zs),1)
            if self.token is not None:
                with torch.random.fork_rng(devices=[z[0].device]):
                    tokens=self.token(tokens,src_key_padding_mask=~available)
                tokens=torch.where(available[:,:,None],tokens,torch.zeros_like(tokens))
            f,weights=self.fusion(list(tokens.unbind(1)),available)
            self.last_diagnostics=dict(self.surface.last,gate_mean=weights.mean(0))
        if not torch.isfinite(f).all():raise FloatingPointError('nonfinite fusion')
        return self.head(f).view(-1),weights,z
    def training_loss(self,batch):
        pred,weights,z=self.predict(batch)
        mse=nn.functional.mse_loss(pred,batch.y.view(-1))
        sums,counts=base.alignment_terms(self.contrastive_features(z),batch.availability,batch.identity)
        align=sum(v/counts[k].clamp_min(1) for k,v in sums.items())
        w=weights[:,:3];w=w/w.sum(-1,keepdim=True).clamp_min(1e-12)
        h=native.entropy(w);loss=mse+.08*align-.08*h
        return loss,dict(mse=mse.detach(),align=align.detach(),entropy=h.detach())

class Gate4(native.Tmol3MGating):
    def __init__(self):
        super().__init__();self.gate=nn.Sequential(nn.Linear(1032,512),nn.ReLU(),nn.Linear(512,4))

def validate_package(p,step=10000):
    if p.get('schema')!=PT_SCHEMA or p.get('plan')!=PLAN or p.get('revision')!='r2' or p.get('step')!=step or p.get('config')!=CONFIG or p.get('historical_tensor_reads')!=0:
        raise ValueError('fresh PT identity/schema mismatch')
    if step and p.get('stream',{}).get('counters')!=[step*6]*3:raise ValueError('PT RNG stream mismatch')

def initialize_ft(package,head,arm,fold):
    validate_package(package);model=Model(arm,fold)
    state={k:v for k,v in package['state_dict'].items() if not k.startswith(('atom_head.','line_head.'))}
    names=(*NATIVE,*native.modal_modules('TMOL3M').keys())
    expected={k for k in model.state_dict() if k.startswith(tuple(n+'.' for n in names))}
    if set(state)!=expected:raise ValueError('PT/FT exact key conversion mismatch')
    for name in names:
        prefix=name+'.';strict_module_state(getattr(model,name),{k[len(prefix):]:v for k,v in state.items() if k.startswith(prefix)})
    if head.get('schema')!='ua3d-cond-spectral-head-r2' or head['plan']!=PLAN or head['fold']!=fold:raise ValueError('new head identity')
    strict_module_state(model.head,head['state_dict']);return model

def optimizer_ft(model,stage):
    model.stage=stage;groups={}
    for name,p in model.named_parameters():
        is_native=name.startswith(tuple(n+'.' for n in NATIVE))
        p.requires_grad_(name!='encoder3d.mask_token' and (stage=='joint' or not is_native))
        if not p.requires_grad:continue
        key='native' if is_native else 'beta' if name=='surface.beta' else 'surface' if name.startswith('surface.') else 'modal_fusion_head'
        groups.setdefault(key,[]).append(p)
    rates=dict(native=1e-4,beta=3e-4 if stage=='head' else 1e-4,surface=3e-4,modal_fusion_head=3e-4 if stage=='head' else 1e-3)
    g=[dict(name=k,params=v,lr=rates[k]) for k,v in groups.items()]
    return torch.optim.AdamW(g,weight_decay=.02,betas=(.9,.999),eps=1e-8),[dict(name=x['name'],lr=x['lr'],parameters=sum(p.numel() for p in x['params'])) for x in g]
