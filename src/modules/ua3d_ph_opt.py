"""Batched P-only execution; state keys, parameters and original ALIGN unchanged."""
import time
import torch
from src.modules import ua3d_ph_r2 as old
from src.dataset import ua3d_ph_opt as d
from src.modules.glt_dual import mean_pool

ARMS=old.ARMS; P_ARMS=old.P_ARMS; M_ARMS=old.M_ARMS; CANDIDATES=old.CANDIDATES
ADAPTER_SEED=old.ADAPTER_SEED; base=old.base
SCHEMA='ua3d-ph-opt-best-r1'


def reference_mean(flat,index,mask):
    values=[]
    for start in range(0,len(index),d.STRATEGY['event_chunk']):
        i=index[start:start+d.STRATEGY['event_chunk']]; m=mask[start:start+d.STRATEGY['event_chunk']]
        values.append((flat[i]*m.to(flat)).sum(1)/m.sum(1).clamp_min(1))
    return torch.cat(values) if values else flat.new_zeros(0)


def tuples_from_plan(f,plan):
    flat=f.reshape(-1)
    birth=reference_mean(flat,plan['birth_index'],plan['birth_mask'])
    finite_death=reference_mean(flat,plan['death_index'],plan['death_mask'])
    death=torch.where(plan['censored'],torch.ones_like(birth),finite_death)
    return torch.stack((birth,death,death-birth,plan['censored'].to(f)),-1)


def distribute(encoded,plan,nodes):
    if len(encoded)==0:
        # Empty graph-H1 is exactly zero, including the encoder's bias.
        return encoded.new_zeros((4*nodes,32))+encoded.sum()*0.
    values=[]; chunk=d.STRATEGY['node_chunk']; ids=plan['recipient_index']; weights=plan['recipient_weight']
    for start in range(0,len(ids),chunk):
        selected=encoded[ids[start:start+chunk]]
        values.append((selected*weights[start:start+chunk,:,None]).sum(1))
    return torch.cat(values) if values else encoded.new_zeros((0,32))


def filter_average(x,n):
    v=x.reshape(4,n,32)
    return ((v[0]/4+v[1]/4)+v[2]/4)+v[3]/4


class GraphPHAdapter(old.GraphPHAdapter):
    def __init__(self):
        super().__init__(); self.cpu_transfer_seconds=0.; self.pairing_seconds=0.
    def forward_batch(self,packed,arm):
        if arm not in P_ARMS:
            raise ValueError('P arm required')
        q=packed['rwpe']; n=len(q)
        p=q+self.pe(packed['pe_input'])
        learned=torch.sigmoid(self.filter(p)) if arm=='P_LEARN' else None
        f=learned if learned is not None else packed['fixed_filtration']
        plans=[]
        if arm=='P_LEARN':
            tick=time.monotonic(); detached=f.detach().cpu().numpy(); self.cpu_transfer_seconds+=time.monotonic()-tick
            tick=time.monotonic(); collections=[[],[]]
            for k in range(4):
                for g,edges in enumerate(packed['edges']):
                    a,b=packed['offsets'][g:g+2]; self.reductions+=1
                    es=d.old.graph_events(detached[a:b,k],edges)
                    for dim in (0,1):
                        collections[dim].append((k,a,es[dim]))
            plans=[d.event_plan(c,n) for c in collections]
            self.pairing_seconds+=time.monotonic()-tick
            tick=time.monotonic(); plans=[d.move(plan,f.device) for plan in plans]
            self.cpu_transfer_seconds+=time.monotonic()-tick
            self.reduction_seconds=self.pairing_seconds
        r=[]
        for dim,encoder in enumerate((self.h0,self.h1)):
            if arm=='P_PE':
                values=encoder(packed['local'+str(dim)])
            else:
                plan=plans[dim] if arm=='P_LEARN' else packed['channel'+str(dim)]
                tuples=tuples_from_plan(f,plan) if arm=='P_LEARN' else plan['tuples']
                values=distribute(encoder(tuples),plan,n)
            r.append(filter_average(values,n))
        delta=self.out(torch.cat((p,r[0],r[1]),-1))[packed['center']]
        delta=delta*packed['center_active'][:,None].to(delta)
        return delta,dict(pe=p.detach(),filtration=f.detach(),r0=r[0].detach(),r1=r[1].detach(),learnable=learned is not None and learned.requires_grad)


class OptimizedPHFuse(old.PHFuse):
    def modified_modalities(self,batch,original,atoms,bonds):
        delta,info=self.graph_ph.forward_batch(batch.packed,self.ph_arm)
        self.last_ph=dict(packed=info)
        z=list(original); h=atoms+delta
        if len(h)!=len(batch.two_d.canonical_graph_index):
            raise ValueError('packed center/canonical order mismatch')
        z[0]=self.A2(self.norm2(mean_pool(h,batch.two_d.canonical_graph_index,len(z[0]))))
        if torch.count_nonzero(z[1][~batch.availability[:,1]]):
            raise ValueError('missing geometry z3')
        return z


def attach_adapter(model,arm):
    if arm not in P_ARMS:
        return old.attach_adapter(model,arm)
    model.__class__=OptimizedPHFuse; model.ph_arm=arm; model.last_ph={}
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(ADAPTER_SEED); model.graph_ph=GraphPHAdapter()
    if sum(p.numel() for p in model.graph_ph.parameters())!=42820:
        raise ValueError('parameter count changed')
    return model


def PHFuse(arm,ft):
    if arm not in P_ARMS:
        return old.PHFuse(arm,ft)
    model=base.BaseOptFuse(False,ft=ft)
    return attach_adapter(model,arm)
