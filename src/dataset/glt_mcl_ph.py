"""Joint GP presentation: one draw for Galformer, SchNet edges, and PH."""
import torch
from .mcl_ph_view import (build_mcl_ph_view,prepare_mcl_ph_sample,mcl_ph_collate)
from .glt_dual import build_dual_sample
from .glt_dual_pretrain import pretrain_collate


def prepare_gp_sample(topology,trimer,smiles,*,seed,key,position,sigma=.03,ratio=.3,
                      static=None,target=None,identity=None,trajectory_override=None,view='noisy',**unused):
    shared=build_mcl_ph_view(topology,trimer,smiles,seed=seed,key=key,position=position,
                            sigma=sigma,ratio=ratio,identity=identity,view=view)
    data,mcl=prepare_mcl_ph_sample(topology,trimer,smiles,seed=seed,key=key,position=position,
        sigma=sigma,ratio=ratio,identity=shared.identity,static=static,view=view,
        trajectory_override=trajectory_override,shared_view=shared)
    clean=build_dual_sample(topology,trimer,smiles,identity=shared.identity,static=static)
    if data.bond_distance.numel()!=data.mcl_bond_index.size(0):
        raise ValueError('GP physical bond row count differs')
    if not torch.equal(data.bond_type,data.mcl_bond_type) or not torch.equal(data.bond_center,data.mcl_bond_center):
        raise ValueError('GP physical bond type/center order differs')
    if clean.geometry_valid:
        bonds=data.mcl_bond_index
        dist=(data.mcl_pos[bonds[:,0]]-data.mcl_pos[bonds[:,1]]).norm(dim=-1)
        if not torch.allclose(dist,data.bond_distance,rtol=1e-6,atol=1e-6):
            raise ValueError('GP shared noisy physical endpoints differ')
    path=clean.line_path
    valid=(clean.line_mask.sum(1)==1)&(~clean.line_is_self)&(path[:,0]<path[:,1])
    valid &= clean.bond_center[path[:,0].clamp_min(0)] & clean.bond_center[path[:,1].clamp_min(0)]
    selected=torch.where(valid)[0]
    labels=dict(atom_mask=(shared.mask if shared.mask is not None else torch.zeros(topology.mips_x.size(0),dtype=torch.bool)),
        atom_label=topology.mips_x[:,:101].argmax(-1),distance=clean.bond_distance[clean.bond_center].clone(),
        angle_pairs=path[selected,:2].reshape(-1,2),angle_cos=clean.line_angle[selected,0].cos(),
        distance_weight=torch.ones(int(clean.bond_center.sum())),fallback=shared.fallback,skip_reasons=[],
        fgr_pair_index=torch.empty((0,2),dtype=torch.long),fgr_target=torch.empty(0),fgr_spd=torch.empty(0,dtype=torch.long),fgr_valid=False,
        align_identity=(shared.identity['normalized_smiles'] if shared.identity else str(key)),
        align_valid=bool(clean.geometry_valid and clean.bond_center.any()),third_task='none')
    data.gp_schema='gp-joint-view-v2'
    return data,dict(glt=labels,mcl=mcl)


def gp_collate(records):
    data,labels=pretrain_collate([(d,t['glt']) for d,t in records])
    mcl,_=mcl_ph_collate([(d,t['mcl']) for d,t in records])
    for k in mcl.keys():
        if k.startswith('mcl_'):setattr(data,k,getattr(mcl,k))
    data.gp_schema='gp-joint-view-v2'
    return data,labels
