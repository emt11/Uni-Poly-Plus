"""Frozen local templates on a periodic bond covering graph (no coordinates made)."""
from collections import defaultdict
import torch

REVISION = 'glt_periodic_seam_r1'
MODES = ('finite_internal', 'finite_incident', 'periodic_template')


def geometry_options(revision=None, mode='finite_internal', angle_value_enabled=False, readout='internal'):
    if revision is None:
        if mode != 'finite_internal' or angle_value_enabled:
            raise ValueError('legacy geometry requires finite_internal and angle-off')
    elif revision != REVISION or mode not in MODES[1:] or not angle_value_enabled or readout != 'incident':
        raise ValueError('invalid periodic seam geometry configuration')
    return dict(geometry_revision=revision, geometry_mode=mode, angle_value_enabled=bool(angle_value_enabled), readout=readout)


def attach_periodic_template(data, qa, qb, angle_positions, offsets):
    """Keep physical observations, quotient states only; retain relative image paths.

    Physical bond k has canonical key (i,j,s) and anchor q of its first
    endpoint. Center observations define the local angle lookup. Two-hop
    paths are enumerated on the infinite covering adjacency, not closed
    open-Trimer ends. No absolute image enters a learned feature.
    """
    a, b = data.bond_canonical_a.tolist(), data.bond_canonical_b.tolist()
    keys, anchors = [], []
    for i,j,u,v in zip(a,b,qa.tolist(),qb.tolist()):
        forward, reverse = (i,j,v-u), (j,i,u-v)
        keys.append(min(forward,reverse)); anchors.append(u if forward <= reverse else v)
    unique = sorted(set(keys)); lookup = {key:k for k,key in enumerate(unique)}
    physical = torch.tensor([lookup[key] for key in keys], dtype=torch.long)
    data.periodic_bond_index = physical
    data.periodic_keys = torch.tensor(unique,dtype=torch.long).reshape(-1,3)
    data.periodic_observation_anchor = torch.tensor(anchors,dtype=torch.long)
    # Only center incident observations enter the template; left/right are
    # separately encoded by the network then weighted symmetrically.
    data.periodic_observation_weight = data.bond_length_weight.clone()
    count = len(unique)
    if count and bool((torch.bincount(physical, weights=data.periodic_observation_weight, minlength=count)<=0).any()):
        raise ValueError('periodic bond lacks a center physical observation')
    incidence = []
    for k,(i,j,s) in enumerate(unique):
        incidence.extend(((k,i),(k,j)))  # self-image has two incidences
    data.periodic_incidence_index = torch.tensor(incidence,dtype=torch.long).reshape(-1,2).t().contiguous()
    local_angles = defaultdict(list)
    for r in torch.where((data.line_mask.sum(-1)==1)&~data.line_is_self&data.line_center_angle)[0].tolist():
        x,y = data.line_path[r,:2].tolist()
        center = int(angle_positions[r,0,1]); q = int(offsets[center])
        nx,ny = (int(physical[x]),anchors[x]-q),(int(physical[y]),anchors[y]-q)
        local_angles[tuple(sorted((nx,ny)))].append(r)
    endpoint = defaultdict(list)
    for k,(i,j,s) in enumerate(unique):
        endpoint[i].append((k,0)); endpoint[j].append((k,s))
    def neighbors(node):
        k,q=node; i,j,s=unique[k]
        out=set()
        for atom,image in ((i,q),(j,q+s)):
            for other,offset in endpoint[atom]:
                candidate=(other,image-offset)
                if candidate != node: out.add(candidate)
        return sorted(out)
    def observations(left,right):
        k,q=left; l,v=right
        i,j,s=unique[k]; ii,jj,ss=unique[l]
        shared=set(((i,q),(j,q+s))) & set(((ii,v),(jj,v+ss)))
        if len(shared)!=1:raise ValueError('ambiguous periodic angle shared image')
        _,image=next(iter(shared))
        key=tuple(sorted(((k,q-image),(l,v-image))))
        if key not in local_angles:raise ValueError('periodic angle missing frozen center template: '+str(key))
        return local_angles[key]
    sources,targets,shifts,paths,groups,angles,masks,selfs,path_shifts=[],[],[],[],[],[],[],[],[]
    # Every image target is a separate attention relation; equal shortest
    # paths average within that image relation only. Multiple local angle
    # observations form equal-weight encoding observations per path.
    for k in range(count):
        origin=(k,0); first=neighbors(origin)
        candidates={origin:[(origin,)]}
        for node in first:candidates[node]=[(origin,node)]
        for node in first:
            for end in neighbors(node):
                if end==origin or end in first:continue
                candidates.setdefault(end,[]).append((origin,node,end))
        for end,walks in sorted(candidates.items()):
            group=len(sources);sources.append(k);targets.append(end[0]);shifts.append(end[1])
            for walk in walks:
                if len(walk)==1:
                    paths.append([k,k,-1]);path_shifts.append([0,0,0]);groups.append(group);angles.append([0.,0.]);masks.append([True,False]);selfs.append(True)
                    continue
                observations0=observations(walk[0],walk[1])
                observations1=observations(walk[1],walk[2]) if len(walk)==3 else [None]
                for r in observations0:
                    for t in observations1:
                        paths.append([node[0] for node in walk]+[-1]*(3-len(walk)))
                        path_shifts.append([node[1] for node in walk]+[0]*(3-len(walk)))
                        groups.append(group);angles.append([float(data.line_angle[r,0]),float(data.line_angle[t,0]) if t is not None else 0.])
                        masks.append([True,t is not None]);selfs.append(False)
    data.periodic_line_source=torch.tensor(sources,dtype=torch.long)
    data.periodic_line_target=torch.tensor(targets,dtype=torch.long)
    data.periodic_line_shift=torch.tensor(shifts,dtype=torch.long)
    data.periodic_line_path=torch.tensor(paths,dtype=torch.long).reshape(-1,3)
    data.periodic_line_path_shift=torch.tensor(path_shifts,dtype=torch.long).reshape(-1,3)
    data.periodic_line_group=torch.tensor(groups,dtype=torch.long)
    data.periodic_line_angle=torch.tensor(angles,dtype=torch.float32).reshape(-1,2)
    data.periodic_line_mask=torch.tensor(masks,dtype=torch.bool).reshape(-1,2)
    data.periodic_line_self=torch.tensor(selfs,dtype=torch.bool)
    return data
