"""Local GLT optimization mechanisms; no storage or experiment framework."""
import math
import torch
from torch import nn
from torch_scatter import scatter


def incident_pool(states, incidence, atom_graph, graphs):
    bond, atom = incidence.long()
    if bond.numel() and (bond.min()<0 or bond.max()>=states.size(0) or atom.min()<0 or atom.max()>=atom_graph.numel()):
        raise ValueError('incident endpoint index out of range')
    values=scatter(states[bond],atom,dim=0,dim_size=atom_graph.numel(),reduce='mean')
    used=torch.bincount(atom,minlength=atom_graph.numel())>0
    graph=scatter(values[used],atom_graph[used],dim=0,dim_size=graphs,reduce='mean')
    valid=torch.bincount(atom_graph[used],minlength=graphs)>0
    return graph,valid


class BondTokenFusion(nn.Module):
    def __init__(self, fusion_gate='none', *, attention_mode='global', incident_bias=0.0):
        super().__init__()
        if fusion_gate not in ('none','incident'):raise ValueError('invalid fusion gate')
        self.fusion_gate=fusion_gate
        self.set_attention_mode(attention_mode, incident_bias)
        self.gate=nn.Linear(1024,1) if fusion_gate=='incident' else None
        if self.gate is not None:
            nn.init.zeros_(self.gate.weight);nn.init.zeros_(self.gate.bias)
        self.q=nn.Linear(512,128);self.k=nn.Linear(512,128);self.v=nn.Linear(512,128)
        self.output=nn.Linear(128,512);self.scale=nn.Parameter(torch.tensor(.1))
        self.implementation='loop'
    def set_implementation(self, implementation):
        if implementation not in ('loop','bucket'):
            raise ValueError('unknown fusion implementation')
        self.implementation=implementation
    def set_attention_mode(self, mode, beta):
        if mode not in ('global','incident_only','incident_bias'):
            raise ValueError('unknown fusion attention mode')
        if isinstance(beta,bool) or not isinstance(beta,(int,float)) or not math.isfinite(beta):
            raise ValueError('invalid incident bias')
        if (mode != 'incident_bias' and beta != 0.0) or (mode == 'incident_bias' and beta not in (0.0,math.log(2.0))):
            raise ValueError('unsupported fixed incident bias')
        self.attention_mode=mode
        self.incident_bias=float(beta)

    def forward(self,atoms,bonds,atom_graph,bond_graph,valid,*,return_attention=False,incidence=None):
        if self.implementation=='bucket' and atoms.size(0):
            return self._forward_bucket(atoms,bonds,atom_graph,bond_graph,valid,
                return_attention=return_attention,incidence=incidence)
        return self._forward_loop(atoms,bonds,atom_graph,bond_graph,valid,
            return_attention=return_attention,incidence=incidence)

    def _forward_loop(self,atoms,bonds,atom_graph,bond_graph,valid,*,return_attention=False,incidence=None):
        result=atoms.clone()
        attention=[]
        gate=None
        local_mode=self.attention_mode != 'global'
        if local_mode:
            if incidence is None or incidence.ndim != 2 or incidence.shape[0] != 2 or incidence.dtype != torch.long:
                raise ValueError('local fusion requires int64 incidence [2,N]')
            if incidence.device != atoms.device:
                raise ValueError('incidence device mismatch')
            bond,atom=incidence
            if bond.numel() and (bond.min()<0 or bond.max()>=bonds.size(0) or atom.min()<0 or atom.max()>=atoms.size(0)):
                raise ValueError('fusion incidence out of range')
            if bond.numel() and not torch.equal(bond_graph[bond],atom_graph[atom]):
                raise ValueError('fusion incidence crosses graphs')
        if self.gate is not None:
            if incidence is None:raise ValueError('incident gate requires incidence')
            bond,atom=incidence.long()
            if bond.numel() and (bond.min()<0 or bond.max()>=bonds.size(0) or atom.min()<0 or atom.max()>=atoms.size(0)):raise ValueError('gate incidence out of range')
            local=scatter(bonds[bond],atom,dim=0,dim_size=atoms.size(0),reduce='mean')
            features=torch.cat([torch.nn.functional.layer_norm(atoms,(512,)),torch.nn.functional.layer_norm(local,(512,))],-1)
            gate=2*torch.sigmoid(self.gate(features))
        for graph in range(valid.numel()):
            if not bool(valid[graph]):continue
            ai=torch.where(atom_graph==graph)[0]; bi=torch.where(bond_graph==graph)[0]
            if not bi.numel():continue
            q=self.q(atoms[ai]).reshape(-1,4,32).transpose(0,1)
            k=self.k(bonds[bi]).reshape(-1,4,32).transpose(0,1)
            v=self.v(bonds[bi]).reshape(-1,4,32).transpose(0,1)
            # Keep the original global path and its arithmetic unchanged.
            if not local_mode:
                w=torch.softmax((q.float()@k.float().transpose(-1,-2))/32**.5,dim=-1).to(v.dtype)
            else:
                candidate=torch.zeros((ai.numel(),bi.numel()),dtype=torch.bool,device=atoms.device)
                selected=atom_graph[atom] == graph
                rows=torch.searchsorted(ai,atom[selected]);cols=torch.searchsorted(bi,bond[selected])
                candidate[rows,cols]=True  # Boolean presence: duplicate pairs do not add beta twice.
                if self.attention_mode == 'incident_only':
                    eligible=candidate.any(-1)
                    if not bool(eligible.any()):
                        if return_attention:
                            attention.append((graph,ai,bi,q.new_zeros((ai.numel(),bi.numel()),dtype=torch.float32)))
                        continue
                    score=(q[:,eligible].float()@k.float().transpose(-1,-2))/32**.5
                    score=score.masked_fill(~candidate[eligible].unsqueeze(0),float('-inf'))
                    sparse_w=torch.softmax(score,dim=-1).to(v.dtype)
                    # Expose zero weight on all non-candidates, including atoms with no incidence.
                    w=torch.zeros_like(q[:,:,0].unsqueeze(-1).expand(-1,-1,bi.numel()))
                    w[:,eligible]=sparse_w
                else:
                    score=(q.float()@k.float().transpose(-1,-2))/32**.5
                    w=torch.softmax(score+self.incident_bias*candidate.unsqueeze(0),dim=-1).to(v.dtype)
            if return_attention:
                attention.append((graph,ai,bi,w.float().mean(0)))
            update=(w@v).transpose(0,1).reshape(-1,128)
            delta=self.output(update)
            if gate is not None:delta=delta*gate[ai]
            if local_mode and self.attention_mode == 'incident_only':
                result[ai[eligible]]=atoms[ai[eligible]]+self.scale*delta[eligible]
            else:
                result[ai]=atoms[ai]+self.scale*delta
        return (result,attention) if return_attention else result

    def _forward_bucket(self,atoms,bonds,atom_graph,bond_graph,valid,*,return_attention=False,incidence=None):
        result=atoms.clone();attention=[];gate=None
        local_mode=self.attention_mode!='global'
        if local_mode:
            if incidence is None or incidence.ndim!=2 or incidence.shape[0]!=2 or incidence.dtype!=torch.long:
                raise ValueError('local fusion requires int64 incidence [2,N]')
            if incidence.device!=atoms.device:raise ValueError('incidence device mismatch')
            bond,atom=incidence
            if bond.numel() and (bond.min()<0 or bond.max()>=bonds.size(0) or atom.min()<0 or atom.max()>=atoms.size(0)):
                raise ValueError('fusion incidence out of range')
            if bond.numel() and not torch.equal(bond_graph[bond],atom_graph[atom]):
                raise ValueError('fusion incidence crosses graphs')
        if self.gate is not None:
            if incidence is None:raise ValueError('incident gate requires incidence')
            bond,atom=incidence.long()
            if bond.numel() and (bond.min()<0 or bond.max()>=bonds.size(0) or atom.min()<0 or atom.max()>=atoms.size(0)):
                raise ValueError('gate incidence out of range')
            local=scatter(bonds[bond],atom,dim=0,dim_size=atoms.size(0),reduce='mean')
            features=torch.cat([torch.nn.functional.layer_norm(atoms,(512,)),torch.nn.functional.layer_norm(local,(512,))],-1)
            gate=2*torch.sigmoid(self.gate(features))
        graphs=valid.numel()
        # Three batch-level transfers replace repeated per-graph validity and
        # index discovery. Stable orders preserve physical observation identity.
        ac=torch.bincount(atom_graph,minlength=graphs).cpu().tolist()
        bc=torch.bincount(bond_graph,minlength=graphs).cpu().tolist()
        vc=valid.detach().cpu().tolist()
        ao=torch.argsort(atom_graph,stable=True);bo=torch.argsort(bond_graph,stable=True)
        ap=[0];bp=[0]
        for a,b in zip(ac,bc):ap.append(ap[-1]+a);bp.append(bp[-1]+b)
        buckets={}
        for graph in range(graphs):
            if not vc[graph] or not bc[graph]:continue
            if not ac[graph]:
                if return_attention:attention.append((graph,ao[ap[graph]:ap[graph+1]],bo[bp[graph]:bp[graph+1]],atoms.new_zeros((0,bc[graph]),dtype=torch.float32)))
                continue
            key=(1<<(ac[graph]-1).bit_length(),1<<(bc[graph]-1).bit_length())
            buckets.setdefault(key,[]).append(graph)
        if local_mode:
            ar=torch.empty_like(atom_graph);br=torch.empty_like(bond_graph)
            ar[ao]=torch.arange(ao.numel(),device=atoms.device)-torch.repeat_interleave(torch.tensor(ap[:-1],device=atoms.device),torch.tensor(ac,device=atoms.device))
            br[bo]=torch.arange(bo.numel(),device=bonds.device)-torch.repeat_interleave(torch.tensor(bp[:-1],device=bonds.device),torch.tensor(bc,device=bonds.device))
        for (maxa,maxb),ids in buckets.items():
            ga=torch.tensor(ids,device=atoms.device,dtype=torch.long)
            ax=torch.arange(maxa,device=atoms.device);bx=torch.arange(maxb,device=bonds.device)
            am=ax.unsqueeze(0)<torch.tensor([ac[g] for g in ids],device=atoms.device).unsqueeze(1)
            bm=bx.unsqueeze(0)<torch.tensor([bc[g] for g in ids],device=bonds.device).unsqueeze(1)
            ai=ao[(torch.tensor([ap[g] for g in ids],device=atoms.device).unsqueeze(1)+ax).clamp_max(ao.numel()-1)]
            bi=bo[(torch.tensor([bp[g] for g in ids],device=bonds.device).unsqueeze(1)+bx).clamp_max(bo.numel()-1)]
            q=self.q(atoms[ai]).reshape(len(ids),maxa,4,32).transpose(1,2)
            k=self.k(bonds[bi]).reshape(len(ids),maxb,4,32).transpose(1,2)
            v=self.v(bonds[bi]).reshape(len(ids),maxb,4,32).transpose(1,2)
            score=(q.float()@k.float().transpose(-1,-2))/32**.5
            eligible=am
            if local_mode:
                lookup=torch.full((graphs,),-1,device=atoms.device,dtype=torch.long);lookup[ga]=torch.arange(len(ids),device=atoms.device)
                slots=lookup[atom_graph[atom]];take=slots>=0
                candidate=torch.zeros((len(ids),maxa,maxb),device=atoms.device,dtype=torch.bool)
                candidate[slots[take],ar[atom[take]],br[bond[take]]]=True
                if self.attention_mode=='incident_only':
                    eligible=am&candidate.any(-1)
                    if not bool(eligible.any()):
                        if return_attention:
                            for slot,graph in enumerate(ids):
                                attention.append((graph,ai[slot,:ac[graph]],bi[slot,:bc[graph]],atoms.new_zeros((ac[graph],bc[graph]),dtype=torch.float32)))
                        continue
                    score=score.masked_fill(~candidate.unsqueeze(1),float('-inf'))
                else:score=score+self.incident_bias*candidate.unsqueeze(1)
            score=score.masked_fill(~bm[:,None,None,:],float('-inf'))
            # Empty incidence rows have zero weights and zero projected update,
            # including output bias; they never feed an all-infinite softmax.
            score=torch.where(eligible[:,None,:,None],score,torch.zeros_like(score))
            w=torch.softmax(score,dim=-1).to(v.dtype)*eligible[:,None,:,None]
            update=(w@v).transpose(1,2).reshape(len(ids),maxa,128)
            delta=self.output(update)
            if gate is not None:delta=delta*gate[ai]
            selected=ai[eligible]
            result=result.index_copy(0,selected,atoms[selected]+self.scale*delta[eligible])
            if return_attention:
                average=w.float().mean(1)
                for slot,graph in enumerate(ids):
                    attention.append((graph,ai[slot,:ac[graph]],bi[slot,:bc[graph]],average[slot,:ac[graph],:bc[graph]]))
        if return_attention:attention.sort(key=lambda row:row[0])
        return (result,attention) if return_attention else result


def weighted_graph_loss(errors,weights,index,graphs):
    numerator=scatter(errors*weights,index,dim=0,dim_size=graphs,reduce='sum')
    denominator=scatter(weights,index,dim=0,dim_size=graphs,reduce='sum')
    return numerator/denominator.clamp_min(1e-12),denominator>0
