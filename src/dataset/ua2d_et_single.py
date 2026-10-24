"""Read-only old templates plus identity-bound atomic fields; no new coordinates."""
from collections import OrderedDict
from pathlib import Path
import json
import os
import numpy as np
import torch
from src.dataset.ua2d_fast_pretrain import (prepare_fast_sample,encode,decode_template,decode_batch,CACHE_SCHEMA)
from src.dataset.ua2d_galformer_pretrain import corruption,collate_samples
from src.modules.glt_dual import element_index

SCHEMA='ua2d-et-single-atomic-fields-r1'

def atomic_fields(topology,trimer,source_row,sample_key):
    required=('trimer_atomic_number','trimer_base_ru_atom_id','trimer_ru_offset',
              'mips_to_trimer_central_index','trimer_edge_index')
    if any(not hasattr(trimer,k) for k in required): raise ValueError('atomic identity fields missing')
    z=torch.as_tensor(trimer.trimer_atomic_number).long().flatten()
    base=torch.as_tensor(trimer.trimer_base_ru_atom_id).long().flatten()
    offsets=torch.as_tensor(trimer.trimer_ru_offset).long().flatten()
    mapping=torch.as_tensor(trimer.mips_to_trimer_central_index).long().flatten()
    canonical_base=torch.as_tensor(topology.canonical_to_trimer_base_atom_id).long().flatten()
    canonical_z=torch.as_tensor(topology.z).long().flatten()
    edges=torch.as_tensor(trimer.trimer_edge_index).long()
    if (len(z)!=len(base) or len(z)!=len(offsets) or mapping.shape!=canonical_base.shape
        or canonical_z.shape!=mapping.shape or edges.ndim!=2 or edges.shape[0]!=2
        or (mapping.numel() and (mapping.min()<0 or mapping.max()>=len(z)))
        or (edges.numel() and (edges.min()<0 or edges.max()>=len(z)))):
        raise ValueError('atomic identity shape/range mismatch')
    if not torch.equal(base[mapping],canonical_base) or not torch.equal(z[mapping],canonical_z) or (offsets[mapping]!=0).any():
        raise ValueError('central/canonical atomic identity conflict')
    inverse={int(b):i for i,b in enumerate(canonical_base.tolist())}
    if len(inverse)!=len(mapping): raise ValueError('duplicate canonical base identity')
    physical=(z>1).nonzero().flatten()
    identities=[(int(base[i]),int(offsets[i])) for i in physical]
    if len(set(identities))!=len(identities): raise ValueError('duplicate physical image identity')
    if any(int(base[i]) not in inverse for i in physical): raise ValueError('unmapped heavy physical atom')
    canonical=torch.tensor([inverse[int(base[i])] for i in physical],dtype=torch.long)
    if not torch.equal(z[physical],canonical_z[canonical]): raise ValueError('element/base image conflict')
    remap=torch.full((len(z),),-1,dtype=torch.long);remap[physical]=torch.arange(len(physical))
    keep=(z[edges[0]]>1)&(z[edges[1]]>1)
    chemical=remap[edges[:,keep]]
    if chemical.numel():
        chemical=torch.unique(torch.cat([chemical,chemical.flip(0)],1).t(),dim=0).t().contiguous()
        if (chemical[0]==chemical[1]).any(): raise ValueError('physical chemical self bond')
    return dict(schema=SCHEMA,source_row=int(source_row),sample_key=bytes(sample_key),
        physical_index=physical,z=z[physical],base=base[physical],offset=offsets[physical],
        center=(offsets[physical]==0),canonical=canonical,canonical_count=len(canonical_z),
        physical_count=len(z),chemical=chemical)


def endpoint_corruption(fields,bond,actions,generator):
    canonical=fields['canonical'];n=fields['canonical_count']
    priority=torch.zeros(n,dtype=torch.long)
    p=torch.where(actions==1,2,torch.where(actions==2,1,0))
    ids=canonical[bond].flatten()
    priority.scatter_reduce_(0,ids,p.repeat(2),reduce='amax',include_self=True)
    # A single random category per canonical atom, shared by all image copies.
    random=torch.randint(101,(n,),generator=generator)
    z=element_index(fields['z']).clone()
    z=torch.where(priority[canonical]==1,random[canonical],z)
    z=torch.where(priority[canonical]==2,torch.full_like(z,101),z)
    return z,priority


def neighborhood(positions,chemical):
    # Per-molecule finite geometry, all pairs, no neighbor truncation or box.
    n=len(positions)
    distance=torch.cdist(positions.float(),positions.float(),compute_mode='donot_use_mm_for_euclid_dist')
    adjacency=distance<4.
    if chemical.numel(): adjacency[chemical[0],chemical[1]]=True
    adjacency.fill_diagonal_(True)
    edge=adjacency.nonzero().t().contiguous()
    chemical_mask=torch.zeros(n,n,dtype=torch.bool)
    if chemical.numel(): chemical_mask[chemical[0],chemical[1]]=True
    return edge,chemical_mask[edge[0],edge[1]]


def prepare_et_sample(template,fields,*,config,position):
    if fields['schema']!=SCHEMA or fields['sample_key']!=template['sample_key']:
        raise ValueError('atomic/template schema or identity mismatch')
    if not bool(template['clean'].geometry_valid):
        raise ValueError('failed 3D sample is excluded from the entire ET training pool')
    if len(template['positions'])!=fields['physical_count']:
        raise ValueError('atomic/template physical count mismatch')
    data,labels=prepare_fast_sample(template,config=config,seed=42,position=position)
    m=len(data.bond_distance)
    if bool(data.geometry_valid):
        physical=fields['physical_index']
        remap=torch.full((fields['physical_count'],),-1,dtype=torch.long)
        remap[physical]=torch.arange(len(physical))
        bond=remap[labels['physical_edges']]
        if bond.numel() and (bond<0).any(): raise ValueError('supervised bond has non-heavy endpoint')
        expected=set(map(tuple,torch.sort(bond,dim=0).values.t().tolist()))
        actual=set(map(tuple,torch.sort(fields['chemical'],dim=0).values.t().tolist()))
        if expected!=actual: raise ValueError('atomic/template real heavy bond disagreement')
        generator=torch.Generator().manual_seed((20261020+1000033*int(position)+719)%(2**63-1))
        chosen,actions,_=corruption(len(template['group_first']),generator,.4)
        selected=chosen[template['group_index']];action=actions[template['group_index']]
        z,priority=endpoint_corruption(fields,bond,action,generator)
        pos=labels['physical_pos'][physical]
        if not torch.isfinite(pos).all(): raise ValueError('nonfinite active ET geometry')
        edge,chemical=neighborhood(pos,fields['chemical'])
        center=fields['center']
        if not center.any(): raise ValueError('valid geometry without central heavy atoms')
    else:
        z=torch.empty(0,dtype=torch.long);pos=torch.empty(0,3)
        edge=torch.empty(2,0,dtype=torch.long);chemical=torch.empty(0,dtype=torch.bool)
        bond=torch.zeros(2,m,dtype=torch.long);center=torch.empty(0,dtype=torch.bool)
        selected=torch.zeros(m,dtype=torch.bool);action=torch.zeros(m,dtype=torch.long)
    labels.update(token_selected=selected,token_action=action,et_z=z,et_pos=pos,et_edge=edge,
        et_chemical=chemical,et_bond=bond,et_center=center)
    return data,labels


def collate_et(samples):
    data,labels=collate_samples(samples)
    offset=0;edge=[];bond=[];graphs=[]
    for graph,(_,item) in enumerate(samples):
        edge.append(item['et_edge']+offset);bond.append(item['et_bond']+offset)
        graphs.append(torch.full((len(item['et_z']),),graph,dtype=torch.long));offset+=len(item['et_z'])
    for k in ('et_z','et_pos','et_chemical','et_center'):
        labels[k]=torch.cat([x[k] for _,x in samples])
    labels.update(et_edge=torch.cat(edge,1),et_bond=torch.cat(bond,1),et_graph=torch.cat(graphs))
    return data,labels


class PreparedETMicrobatches(torch.utils.data.Dataset):
    def __init__(self,cache,atomic,config,rank,prepared_counts):
        self.root=Path(cache);self.atomic=Path(atomic);self.config=config;self.rank=rank
        self.prepared_counts=prepared_counts;self.environment=self.atom_environment=None
        self.templates=OrderedDict()
    def __len__(self): return 5000*6
    def _open(self):
        import lmdb
        from src.training.glt_dual_runtime import OrderedSampleStream
        meta=json.loads((self.root/'metadata.json').read_text());atomic=json.loads((self.atomic/'metadata.json').read_text())
        if meta.get('schema')!=CACHE_SCHEMA or meta.get('status')!='COMPLETE' or meta.get('train_rows')!=911391:
            raise ValueError('old prepared cache mismatch')
        if atomic.get('schema')!=SCHEMA or atomic.get('status')!='COMPLETE' or atomic.get('completed')!=911391:
            raise ValueError('atomic fields incomplete/schema mismatch')
        if any(atomic[k]!=meta[k] for k in ('source','index')): raise ValueError('atomic/cache source binding mismatch')
        self.environment=lmdb.open(str(self.root/'templates.lmdb'),readonly=True,lock=False,readahead=True,max_readers=128)
        self.atom_environment=lmdb.open(str(self.atomic/'fields.lmdb'),readonly=True,lock=False,readahead=True,max_readers=128)
        self.keys=np.load(self.root/'keys.npy',mmap_mode='r',allow_pickle=False)
        self.rows=np.load(self.root/'source_rows.npy',mmap_mode='r',allow_pickle=False)
        if len(self.keys)!=911391 or self.rows.shape!=(911391,): raise ValueError('prepared row coverage mismatch')
        self.eligible=None
        if os.environ.get('UA2D_ET_SINGLE_REPAIR')=='repair1':
            sampling=self.atomic.parent/'sampling_repair1'
            summary=json.loads((sampling/'metadata.json').read_text())
            if summary.get('status')!='COMPLETE' or summary.get('policy')!='drop_failed_3d_entire_record' or summary.get('completed')!=911391:
                raise ValueError('3D eligibility filter incomplete/schema mismatch')
            self.eligible=np.load(sampling/'eligible_indices.npy',mmap_mode='r',allow_pickle=False)
            if len(self.eligible)!=summary['valid_rows'] or len(self.eligible)+summary['excluded_rows']!=911391:
                raise ValueError('3D filter count mismatch')
        self.stream=OrderedSampleStream(len(self.eligible) if self.eligible is not None else 911391,42)
    def __getitem__(self,micro_index):
        if self.environment is None: self._open()
        worker_info=torch.utils.data.get_worker_info();worker=worker_info.id if worker_info else 0
        start=(int(micro_index)*2+self.rank)*84;records=[];rows=[]
        for offset in range(84):
            with self.prepared_counts.get_lock(): self.prepared_counts[worker]+=1
            position=start+offset;index=self.stream.index_at(position)
            if self.eligible is not None:index=int(self.eligible[index])
            key=self.keys[index].tobytes()
            if key in self.templates: template,fields=self.templates.pop(key)
            else:
                with self.environment.begin() as txn: payload=txn.get(key)
                if payload is None: raise ValueError('missing old template')
                template=decode_template(payload)
                with self.atom_environment.begin() as txn: payload=txn.get(key)
                if payload is None: raise ValueError('missing atomic fields')
                # Atomic fields have no Data object; use the existing typed serializer.
                from src.dataset.ua2d_fast_pretrain import _tensors
                import pickle
                fields=_tensors(pickle.loads(payload))
                if fields['source_row']!=int(self.rows[index]) or fields['sample_key']!=key:
                    raise ValueError('source_row/sample_key binding mismatch')
            self.templates[key]=(template,fields)
            if len(self.templates)>2048: self.templates.popitem(last=False)
            records.append(prepare_et_sample(template,fields,config=self.config,position=position))
            rows.append(int(self.rows[index]))
        data,labels=collate_et(records)
        return encode((data,labels,dict(micro_index=int(micro_index),start_position=start,source_rows=rows)))
