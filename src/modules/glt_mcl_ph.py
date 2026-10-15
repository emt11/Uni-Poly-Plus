"""PH-routed atom-to-bond residual, preserving the O8 + Galformer encoder."""
import torch
from torch import nn
from .glt_dual import DualGLTModel,mean_pool
from .mcl_ph import SchNetExpert,TopologyRouter,xavier_

class EndpointAdaptor(nn.Module):
    def __init__(self):
        super().__init__();self.norm=nn.LayerNorm(128)
        self.net=nn.Sequential(nn.Linear(384,512),nn.GELU(),nn.Linear(512,512));self.apply(xavier_)
    def forward(self,left,right):
        a,b=self.norm(left),self.norm(right)
        return self.net(torch.cat([a+b,(a-b).abs(),a*b],-1))

class GPEncoder(DualGLTModel):
    architecture_name='O8-Galformer-PH-EndpointResidual-v2'
    def __init__(self,*,route='GP',dropout=.1):
        if route not in ('GP','GU','GP_A'):raise ValueError('unknown GP route')
        super().__init__('concat',dropout=dropout,torsion=False)
        self.route=route;self.residual_scale=.1
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(42)
            self.experts=nn.ModuleList([SchNetExpert(c) for c in (2.,3.,4.)])
            self.router=TopologyRouter();self.adaptor=EndpointAdaptor()
        self.last_encoded=None
    def encode(self,data,*,atom_mask=None):
        if getattr(data,'gp_schema',None)!='gp-joint-view-v2':raise ValueError('GP joint schema required')
        out=super().encode(data,atom_mask=atom_mask)
        routing=self.router(data.mcl_trajectory.flatten(1))
        weights=(torch.full_like(routing['weights'],1/3) if self.route=='GU' else routing['weights'])
        states=[]
        for k,expert in enumerate(self.experts):
            selected=data.mcl_edge_scale==k;edges=data.mcl_edge_index[:,selected]
            states.append(expert(data.mcl_z,data.mcl_charge,data.mcl_aromatic,edges[0],edges[1],
                data.mcl_edge_distance[selected],data.mcl_edge_type[selected],data.mcl_z.numel()))
        mixed=(torch.stack(states,1)*weights[data.mcl_atom_batch].unsqueeze(-1)).sum(1)
        ends=data.mcl_bond_index.long()
        valid=data.mcl_readout_valid.bool()&out['geometry_valid']
        delta=self.adaptor(mixed[ends[:,0]],mixed[ends[:,1]])
        delta=torch.where(valid[data.bond_batch].unsqueeze(-1),delta,torch.zeros_like(delta))
        joint=out['bond_states']+self.residual_scale*delta
        out['bond_states']=joint;out['center_bond_states']=joint[data.bond_center]
        out['graph_3d']=mean_pool(joint[data.bond_center],data.bond_batch[data.bond_center],valid.numel())
        out['graph_3d']=torch.where(out['geometry_valid'].unsqueeze(-1),out['graph_3d'],torch.zeros_like(out['graph_3d']))
        out.update(router_dense=routing['dense'],router_weights=weights,mcl_valid=valid)
        self.last_encoded=out
        return out
