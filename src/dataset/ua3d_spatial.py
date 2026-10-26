"""Physical-instance geometry sidecars and unchanged noisy PT stream."""
from pathlib import Path
import pickle
import numpy as np
import torch
from src.dataset import ua3m_paperfuse as parent
from src.dataset.ua2d_fast_pretrain import encode,decode_template
from src.dataset.ua2d_p10k_nofp_align import prepare_arm_sample,collate_arm_samples,micro_start,resolve_slot
from src.dataset.glt_dual import dual_glt_collate

SCHEMA='ua3d-spatial-sidecar-r4'
pt_batch=parent.pt_batch
worker_init=parent.worker_init


def physical_topology(endpoints,z_a,z_b,position_count,atomic=None,offsets=None):
    endpoints=torch.as_tensor(endpoints).long()
    if endpoints.ndim!=2 or endpoints.shape[0]!=2 or endpoints.shape[1]!=len(z_a) or len(z_a)!=len(z_b):raise ValueError('physical endpoint shape')
    if endpoints.numel()==0 or endpoints.min()<0 or endpoints.max()>=position_count:raise ValueError('physical endpoint index')
    z=torch.cat((z_a,z_b)).long();flat=endpoints.reshape(-1)
    if (z<=1).any():raise ValueError('native tokens must be physical heavy bonds')
    ids,inverse=torch.unique(flat,sorted=True,return_inverse=True);bonds=inverse.reshape(2,-1);n=len(ids)
    assigned={}
    for i,value in zip(flat.tolist(),z.tolist()):
        if i in assigned and assigned[i]!=value:raise ValueError('physical element identity conflict')
        assigned[i]=value
    if atomic is not None:
        atomic=torch.as_tensor(atomic).long()
        if set(torch.where(atomic>1)[0].tolist())!=set(ids.tolist()):raise ValueError('physical heavy atom coverage mismatch')
        if not torch.equal(atomic[flat],z):raise ValueError('physical endpoint element mismatch')
    adjacency=np.zeros((n,n),dtype=bool)
    a,b=bonds.numpy();adjacency[a,b]=True;adjacency[b,a]=True
    if np.diag(adjacency).any():raise ValueError('physical self chemical bond')
    excluded=adjacency.copy();np.fill_diagonal(excluded,True)
    for i in range(n):
        neighbors=np.flatnonzero(adjacency[i])
        if len(neighbors):excluded[i]|=adjacency[neighbors].any(axis=0)
    seam=None
    if offsets is not None:seam=torch.as_tensor(offsets).long()[ids]
    return dict(ids=ids,bonds=bonds,excluded=excluded,offsets=seam)


def spatial_fields(positions,topology):
    positions=torch.as_tensor(positions).float()
    if not torch.isfinite(positions).all():raise FloatingPointError('nonfinite physical coordinates')
    p=positions[topology['ids']].numpy();delta=p[:,None,:]-p[None,:,:]
    distance=np.sqrt(np.sum(delta*delta,axis=-1,dtype=np.float32));n=len(p)
    if not np.isfinite(distance).all():raise FloatingPointError('nonfinite spatial distances')
    offdiag=~np.eye(n,dtype=bool)
    if np.any((distance<=1e-8)&offdiag):raise ValueError('near-zero distinct physical spatial distance')
    mask=(distance<=4.5)&~topology['excluded'];target,source=np.nonzero(mask)
    dist=torch.from_numpy(distance[target,source].copy()).float()
    edges=torch.tensor(np.stack((source,target)),dtype=torch.long)
    degree=torch.bincount(edges[1],minlength=n)
    directions=torch.from_numpy((p[source]-p[target]).copy())/dist[:,None]
    first=torch.zeros((n,3));second=torch.zeros((n,3,3))
    first.index_add_(0,edges[1],directions)
    second.index_add_(0,edges[1],directions[:,:,None]*directions[:,None,:])
    valid=degree>=2;nf=degree.float();den=(nf*(nf-1)).clamp_min(1)
    c=(first.square().sum(-1)-nf)/den;c2=(second.square().sum((-1,-2))-nf)/den
    angles=torch.stack((c,(3*c2-1)/2),-1);angles[~valid]=0
    if not torch.isfinite(angles).all():raise FloatingPointError('nonfinite angular moments')
    offsets=topology['offsets'];seams=int((offsets[edges[0]]!=offsets[edges[1]]).sum()) if offsets is not None else -1
    return dict(sp_bond_atoms=topology['bonds'].clone(),sp_edges=edges,sp_distance=dist,sp_angle=angles,
        sp_degree=degree,sp_angle_valid=valid,sp_seams=torch.tensor([seams]),sp_graph=torch.zeros(n,dtype=torch.long))


def attach(data,fields):
    for k,v in fields.items():setattr(data,k,v)
    return data


def append_batch(batch,graphs):
    if not graphs:raise ValueError('empty spatial collate')
    atom_offset=0;endpoints=[];edges=[];graph_ids=[]
    for i,g in enumerate(graphs):
        endpoints.append(g.sp_bond_atoms+atom_offset);edges.append(g.sp_edges+atom_offset)
        graph_ids.append(torch.full((len(g.sp_degree),),i,dtype=torch.long));atom_offset+=len(g.sp_degree)
    batch.sp_bond_atoms=torch.cat(endpoints,1);batch.sp_edges=torch.cat(edges,1);batch.sp_graph=torch.cat(graph_ids)
    for name in ('sp_distance','sp_angle','sp_degree','sp_angle_valid','sp_seams'):setattr(batch,name,torch.cat([getattr(g,name) for g in graphs]))
    return batch


class PTData(parent.PTData):
    def __getitem__(self,micro_index):
        if self.environment is None:self._open()
        info=torch.utils.data.get_worker_info();worker=info.id if info else 0;start=micro_start(int(micro_index))
        records=[];rows=[];slots=[];fps=[]
        for offset in range(168):
            with self.prepared_counts.get_lock():self.prepared_counts[worker]+=1
            position=start+offset;selected=self.stream.index_at(position)
            index,key,source_row=resolve_slot(self.eligible,self.keys,self.rows,selected)
            if key!=self.selected_keys[selected].tobytes() or source_row!=int(self.selected_rows[selected]):raise ValueError('logical/key/source-row mismatch')
            if key in self.templates:template=self.templates.pop(key)
            else:
                with self.environment.begin() as txn:payload=txn.get(key)
                if payload is None:raise ValueError('missing PT template')
                template=decode_template(payload)
            if template['sample_key']!=key or not bool(template['clean'].geometry_valid):raise ValueError('invalid PT identity/geometry')
            if '_sp_topology' not in template:
                x=template['clean'];template['_sp_topology']=physical_topology(torch.stack((template['endpoint_a'],template['endpoint_b'])),x.bond_z_a,x.bond_z_b,len(template['positions']))
            self.templates[key]=template
            if len(self.templates)>self.cache_entries:self.templates.popitem(last=False)
            x,y=prepare_arm_sample(template,config=self.config,seed=42,position=position)
            attach(x,spatial_fields(y['physical_pos'],template['_sp_topology']))
            records.append((x,y));fps.append(parent.unpack_fp(template['fp_bits']));rows.append(source_row);slots.append(index)
        data,labels=collate_arm_samples(records);append_batch(data,[x for x,_ in records]);data.morgan_fp=torch.stack(fps)
        coverage=coverage_of(data)
        return encode((data,labels,dict(micro_index=int(micro_index),start_position=start,source_rows=rows,logical_slots=slots,spatial=coverage)))


def coverage_of(data):
    degrees=data.sp_degree
    noedge=0
    for i in range(int(data.sp_graph.max())+1 if data.sp_graph.numel() else 0):
        noedge+=int(not bool((degrees[data.sp_graph==i]>0).any()))
    return dict(atoms=len(degrees),directed_edges=data.sp_edges.shape[1],zero_neighbor_atoms=int((degrees==0).sum()),
        valid_angle_atoms=int(data.sp_angle_valid.sum()),max_neighbors=int(degrees.max()) if len(degrees) else 0,
        no_edge_graphs=noedge,seam_edges=int(data.sp_seams.sum()) if (data.sp_seams>=0).all() else None,
        seam_semantics='different original RU offset; PT immutable template has no RU offset, not inferred')


class InputCache(parent.InputCache):
    def __init__(self,root):
        super().__init__(root)
        from src.training.ua3d_spatial import OUT
        sidecar=OUT/'inputs'
        meta=parent.json.loads((sidecar/'metadata.json').read_text())
        if meta.get('schema')!=SCHEMA or meta.get('status')!='COMPLETE':raise ValueError('spatial sidecar incomplete')
        with (sidecar/'sidecar.pkl').open('rb') as f:self.sidecar=pickle.load(f)
    def get(self,task,row,key,smiles):
        x=super().get(task,row,key,smiles);entry=self.sidecar[f'{task}:{row}']
        if (entry['key'],entry['smiles'],entry['available'])!=(key,smiles,x['available']):raise ValueError('spatial sidecar join mismatch')
        if x['available']:attach(x['three_d'],entry['fields'])
        return x

    def get_geometry(self,task,row,key,smiles):
        entry=self.mapping[task].get(row)
        if not entry or (entry['key'],entry['smiles'])!=(key,smiles):raise ValueError('probe row/key/SMILES mismatch')
        side=self.sidecar[f'{task}:{row}']
        if (side['key'],side['smiles'],side['available'])!=(key,smiles,entry['available']):raise ValueError('probe sidecar identity mismatch')
        if not entry['available']:return None
        cache_key=entry['cache_key']
        if cache_key not in self.memory:
            with self.env.begin() as txn:payload=txn.get(cache_key.encode())
            if payload is None:raise ValueError('probe geometry cache missing')
            record=decode_template(payload)
            if record.get('task')!=task or record.get('row')!=row or record.get('sample_key')!=key or record.get('smiles')!=smiles or record.get('available') is not True:raise ValueError('probe native cache identity mismatch')
            # No 2D/FP tensors enter the probe model or its feature cache.
            self.memory[cache_key]={'geometry':parent.GraphData.from_dict(record['geometry'])}
        return attach(self.memory[cache_key]['geometry'].clone(),side['fields'])


class Data(parent.Data):
    def __init__(self,task,fold,formal,source,split,cache):
        super().__init__(task,fold,formal,source,split,cache);self.cache.close();self.cache=InputCache(cache)


def collate(samples):
    batch=parent.collate(samples);graphs=[x['three_d'] for x in samples if x['available']]
    if graphs:append_batch(batch.three_d,graphs)
    return batch


def loader(ds,indices,shuffle=False,seed=42):
    return torch.utils.data.DataLoader(torch.utils.data.Subset(ds,indices),batch_size=24,shuffle=shuffle,num_workers=0,
        drop_last=False,collate_fn=collate,generator=torch.Generator().manual_seed(seed) if shuffle else None)


def geometry_loader(ds,indices):
    # Probe extraction avoids constructing 2D/FP batches or reading any targets.
    def fetch(entries):
        xs=[ds.cache.get_geometry(ds.task,ds.indices[i],ds.keys[i],ds.smiles[i]) for i in entries]
        rows=[i for i,x in enumerate(xs) if x is not None];graphs=[xs[i] for i in rows]
        data=append_batch(dual_glt_collate(graphs),graphs) if graphs else None
        return data,torch.tensor(rows,dtype=torch.long),list(entries)
    return torch.utils.data.DataLoader(list(indices),batch_size=24,shuffle=False,num_workers=0,collate_fn=fetch)
