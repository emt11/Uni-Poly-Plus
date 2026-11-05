"""Weight-independent P input packing, with protected r2 paths for C0/M."""
import json
from pathlib import Path
import numpy as np
import torch
from src.dataset import ua3d_ph_r2 as old

SCHEMA='ua3d-ph-opt-packing-r1'
STRATEGY=dict(event_chunk=128,node_chunk=128,reference_padding='per-channel maximum membership',
              node_order='graph input order then local storage order',filter_order=[0,1,2,3],
              encoded_reduction='event encoding before weighted per-node padded sum; four filters added sequentially',
              learned_copy='one detached [all_nodes,4] CPU copy per forward; original r2 graph_events per graph/filter')


def padded(rows,dtype=torch.long):
    width=max((len(row) for row in rows),default=0)
    x=torch.zeros((len(rows),width),dtype=dtype); mask=torch.zeros((len(rows),width),dtype=torch.bool)
    for i,row in enumerate(rows):
        if row:
            x[i,:len(row)]=torch.as_tensor(row,dtype=dtype); mask[i,:len(row)]=True
    return x,mask


def event_plan(collections, total_nodes):
    """collections=(filter, node_offset, local events), stable event order."""
    births=[]; deaths=[]; censored=[]; recipients=[[] for _ in range(4*total_nodes)]; weights=[[] for _ in recipients]
    for k,offset,events in collections:
        for e in events:
            col=len(births)
            births.append([(i+offset)*4+k for i in e['birth']])
            deaths.append([(i+offset)*4+k for i in e['death']])
            censored.append(e['censored'])
            for i in e['nodes']:
                row=k*total_nodes+offset+i
                recipients[row].append(col); weights[row].append(e['count']/len(e['nodes']))
    bi,bm=padded(births); di,dm=padded(deaths); ri,rm=padded(recipients); rw,_=padded(weights,torch.float32)
    return dict(birth_index=bi,birth_mask=bm,death_index=di,death_mask=dm,censored=torch.tensor(censored,dtype=torch.bool),
                recipient_index=ri,recipient_mask=rm,recipient_weight=rw,events=len(births))


def pack_side(side):
    """No graph PH, RWPE or VR reduction: consume only already saved values/events."""
    q=side['rwpe'].float(); n=len(q); edges=np.asarray(side['edges'])
    neighbor=old.neighbor_mean(q,edges)
    fixed=q[:,[1,3,7,15]]
    local=[old.local_tuples(fixed[:,k],edges) for k in range(4)]
    tuples=[[old.event_tuples(fixed[:,k],side['events'][k][dim]) for k in range(4)] for dim in (0,1)]
    plans=[]
    for dim in (0,1):
        plan=event_plan([(k,0,side['events'][k][dim]) for k in range(4)],n)
        plan['tuples']=torch.cat(tuples[dim],0); plans.append(plan)
    return dict(schema=SCHEMA,rwpe=q,pe_input=torch.cat((q,neighbor),-1),fixed_filtration=fixed,
                local0=torch.cat([x[0] for x in local],0),local1=torch.cat([x[1] for x in local],0),
                fixed_tuples=tuples,fixed_plans=plans,events=side['events'],edges=edges,center=side['center'].long(),
                graph_available=side.get('graph_available',True),task=side.get('task'),row=side.get('row'),key=side.get('key'),nodes=n)


def batch_plan(sides,arm):
    sizes=[s['nodes'] for s in sides]; total=sum(sizes); offsets=np.cumsum([0]+sizes).tolist()
    centers=torch.cat([s['center']+o for s,o in zip(sides,offsets)],0)
    center_active=torch.cat([torch.full((len(s['center']),),s['graph_available'],dtype=torch.bool) for s in sides])
    packed=dict(rwpe=torch.cat([s['rwpe'] for s in sides]),pe_input=torch.cat([s['pe_input'] for s in sides]),
                fixed_filtration=torch.cat([s['fixed_filtration'] for s in sides]),center=centers,center_active=center_active,
                sizes=sizes,offsets=offsets,edges=[s['edges'] for s in sides],nodes=total)
    if arm=='P_PE':
        for dim in (0,1):
            packed['local'+str(dim)]=torch.cat([s['local'+str(dim)][k*s['nodes']:(k+1)*s['nodes']] for k in range(4) for s in sides])
    elif arm=='P_FIXED':
        for dim in (0,1):
            local=[s['fixed_plans'][dim] for s in sides]
            width=max((x['recipient_index'].shape[1] for x in local),default=0)
            ids=torch.zeros((4*total,width),dtype=torch.long); weights=torch.zeros((4*total,width)); event_offset=0
            for g,plan_local in enumerate(local):
                n=sizes[g]; w=plan_local['recipient_index'].shape[1]
                for k in range(4):
                    a=k*total+offsets[g]; b=a+n
                    ids[a:b,:w]=plan_local['recipient_index'][k*n:(k+1)*n]+event_offset
                    weights[a:b,:w]=plan_local['recipient_weight'][k*n:(k+1)*n]
                event_offset+=plan_local['events']
            plan=dict(recipient_index=ids,recipient_weight=weights,events=event_offset,
                      tuples=torch.cat([x['tuples'] for x in local],0))
            packed['channel'+str(dim)]=plan
    return packed


def move(value,device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value,dict):
        return {k:move(v,device) for k,v in value.items()}
    # Edges and event decisions deliberately remain on CPU; no per-graph tensor transfers.
    return value


class Batch:
    def __init__(self,base,packed):
        self.base=base; self.packed=packed
    def __getattr__(self,name):
        return getattr(self.base,name)
    def to(self,device,non_blocking=False):
        return Batch(self.base.to(device,non_blocking=non_blocking),move(self.packed,device))


def collate(samples):
    from src.dataset.ua3m_paperfuse import collate as original
    arms={x['ph_arm'] for x in samples}
    if len(arms)!=1:
        raise ValueError('mixed P arms in one batch')
    return Batch(original(samples),batch_plan([x['packed_side'] for x in samples],next(iter(arms))))


class PData(torch.utils.data.Dataset):
    def __init__(self,arm,task,fold,formal,source,split,cache,packing):
        from src.dataset.ua3m_paperfuse import Data as Original
        self.base=Original(task,fold,formal,source,split,cache); self.arm=arm; self.task=task; self.fold=fold
        self.packing=Path(packing); self.memo={}
    def __getattr__(self,name):
        return getattr(self.base,name)
    @property
    def targets(self):
        return self.base.targets
    @targets.setter
    def targets(self,x):
        self.base.targets=x
    def __len__(self):
        return len(self.base)
    def __getitem__(self,i):
        row=self.indices[i]
        if row not in self.memo:
            p=torch.load(self.packing/self.task/f'row{row}.pt',map_location='cpu',weights_only=False)
            if p['schema']!=SCHEMA or p['row']!=row or p['key']!=str(self.keys[i]) or p['task']!=self.task:
                raise ValueError('packed row/key mismatch')
            self.memo[row]=p
        x=self.base[i]; x.update(ph_arm=self.arm,packed_side=self.memo[row]); return x
    def close(self):
        self.base.close()


def make_data(arm,task,fold,formal,source,split,cache,packing,parent_sidecar):
    if arm.startswith('P_'):
        return PData(arm,task,fold,formal,source,split,cache,packing)
    return old.Data(task,fold,formal,source,split,cache,parent_sidecar)


def loader(ds,ids,shuffle=False,seed=42):
    if not isinstance(ds,PData):
        return old.loader(ds,ids,shuffle,seed)
    return torch.utils.data.DataLoader(torch.utils.data.Subset(ds,ids),batch_size=24,shuffle=shuffle,num_workers=0,
        drop_last=False,collate_fn=collate,generator=torch.Generator().manual_seed(seed) if shuffle else None)


def diagnostic_collate(ds,samples):
    return collate(samples) if isinstance(ds,PData) else old.collate(samples)


def workload(side):
    n=len(side['rwpe']); canonical=int(side['canonical_atoms']); bonds=int(side['bond_atoms'].shape[1]) if side['available'] else 0
    spatial=sum(int(e.shape[1]) for e in side.get('sp_edges',[]))
    memberships=sum(len(e['nodes']) for k in side['events'] for dim in k for e in dim)
    # Quadratic terms explicitly cover attention and worst-case dynamic event membership.
    return dict(rows=1,nodes=n,canonical_squared=canonical**2,bonds=bonds,bonds_squared=bonds**2,
                spatial_edges=spatial,fixed_memberships=memberships,learned_work=max(1,n)*max(1,len(side['edges'])),
                learned_membership_bound=4*n*n)


def sum_work(rows):
    return {k:sum(r[k] for r in rows) for k in rows[0]} if rows else {}
