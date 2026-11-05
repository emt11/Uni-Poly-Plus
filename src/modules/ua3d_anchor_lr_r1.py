"""Frozen downstream BASE with trainable adapters and same-batch difference anchoring."""
import torch
from torch import nn
from src.modules import ua3d_base_opt_r6 as base
from src.modules import ua3d_ph_opt_r2 as ph
from src.modules.ua3d_ph_r2 import SpatialRouter
from src.modules.glt_dual import mean_pool

RATES={'G_GEOM_1E4':1e-4,'G_GEOM_3E5':3e-5,'P_PE_1E4':1e-4,'P_PE_3E5':3e-5}
TRAIN_ARMS=tuple(RATES)
ARMS=('C_ANCHOR',)+TRAIN_ARMS
CANDIDATES=('P_PE_1E4','P_PE_3E5','G_GEOM_1E4','G_GEOM_3E5')
SCHEMA='ua3d-anchor-lr-adapter-best-r1'

def backend(candidate):
    if candidate=='C_ANCHOR':return candidate
    if candidate not in RATES:raise ValueError('fixed candidate')
    return 'G_GEOM' if candidate.startswith('G_GEOM_') else 'P_PE'


def freeze_base(model):
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    return model


class AnchorModel(nn.Module):
    def __init__(self,frozen,arm):
        super().__init__()
        if arm not in TRAIN_ARMS:
            raise ValueError('trainable arm required')
        self.base=freeze_base(frozen); self.candidate=arm; self.arm=backend(arm); arm=self.arm
        with torch.random.fork_rng(devices=[]):
            torch.default_generator.manual_seed(2026102801)
            self.adapter=SpatialRouter() if arm=='G_GEOM' else ph.GraphPHAdapter()
        if sum(p.numel() for p in self.adapter.parameters())!=(36643 if arm=='G_GEOM' else 42820):
            raise ValueError('adapter parameter count')
        self.cost=dict(frozen_H_batches=0,residual_U_batches=0,p_learn_reductions=0,full_encoder_batches=0)
        self.last={}
    def train(self,mode=True):
        super().train(mode); self.base.eval(); self.adapter.train(mode); return self
    def assert_frozen(self):
        if any(p.requires_grad or p.grad is not None for p in self.base.parameters()) or any(m.training for m in self.base.modules()):
            raise ValueError('original BASE must remain frozen and eval')
    def H(self,z,available):
        self.cost['frozen_H_batches']+=1
        f,w=self.base.fuse(z,available)
        return self.base.head(f).view(-1)
    def U(self,atoms,graph,count):
        self.cost['residual_U_batches']+=1
        return self.base.A2(self.base.norm2(mean_pool(atoms,graph,count)))
    def predict(self,batch):
        self.assert_frozen(); n=len(batch.y_base); z=list(batch.z)
        with torch.no_grad():
            h_original=self.H(batch.z,batch.availability)
            u_original=self.U(batch.atoms,batch.atom_graph,n) if self.arm in ('P_PE','P_LEARN') else None
        if self.arm=='G_GEOM':
            delta=torch.zeros_like(z[1]); info=[]
            for row,side in enumerate(batch.sides):
                if not bool(batch.availability[row,1]):
                    if side['available']:
                        raise ValueError('missing3D/sidecar disagreement')
                    continue
                value,details=self.adapter(states=batch.bonds[row],side=side,arm='M_GEOM')
                delta[row]=value; info.append(details)
            z[1]=(z[1]+delta)*batch.availability[:,1:2].to(z[1]); self.last=dict(spatial=info)
        else:
            before=self.adapter.reductions
            try:
                delta,info=self.adapter.forward_batch(batch.packed,self.arm)
            finally:
                self.cost['p_learn_reductions']+=self.adapter.reductions-before
            u_modified=self.U(batch.atoms+delta,batch.atom_graph,n)
            # Difference first is necessary for exact cancellation at zero residual.
            z[0]=batch.z[0]+(u_modified-u_original); self.last=dict(ph=info)
        if torch.count_nonzero(z[1][~batch.availability[:,1]]):
            raise ValueError('missing3D residual must be exactly zero')
        modified=self.H(z,batch.availability)
        prediction=batch.y_base+(modified-h_original)
        if not torch.isfinite(prediction).all():
            raise FloatingPointError('anchored prediction')
        self.last.update(prediction=prediction.detach(),y_base=batch.y_base.detach())
        return prediction
    def training_loss(self,batch):
        pred=self.predict(batch)
        mse=nn.functional.mse_loss(pred,batch.y)
        anchor=(pred-batch.y_base).square().mean()
        return mse+.01*anchor,dict(mse=mse.detach(),anchor_penalty=anchor.detach())


def optimizer(model):
    model.assert_frozen()
    params=list(model.adapter.parameters())
    ids={id(p) for p in params}
    if ids!={id(p) for p in model.parameters() if p.requires_grad}:
        raise ValueError('optimizer must cover only adapter exactly once')
    opt=torch.optim.AdamW(params,lr=RATES[model.candidate],weight_decay=.0001,betas=(.9,.999),eps=1e-8)
    if opt.param_groups[0]['lr']!=RATES[model.candidate]:raise ValueError('actual candidate LR')
    return opt,torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=70,eta_min=1e-6)


def verify_unchanged(model,source_state):
    model.assert_frozen()
    actual=model.base.state_dict()
    if set(actual)!=set(source_state) or any(not torch.equal(v.detach().cpu(),source_state[k]) for k,v in actual.items()):
        raise ValueError('frozen original BASE state changed')
