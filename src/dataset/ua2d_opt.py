"""UA2D-OPT: read-only source mapping, lifted hop3 and clean context targets."""
from collections import Counter, OrderedDict
from pathlib import Path
import json
import numpy as np
import torch
from src.dataset.glt_dual import build_glt_2d_sample,glt_2d_collate
from src.dataset.ua2d_fast_pretrain import encode,decode_template,prepare_fast_sample
from src.dataset.canonical_periodic import build_canonical_periodic_topology
SCHEMA='ua2d-opt-hop3-r1'
CACHE_CONTRACT='ua2d-opt-numeric-context-hop-views-v1'
HOP_FIELDS=tuple(f'opt_hop{h}_rows' for h in (1,2,3))
FIELDS=('lga_edge_index','lga_spd','lga_path_index','lga_path_mask','lga_path_shift',
        'lga_source_image_shift','lga_relation_mask','bond_path_features','bond_path_mask')


def extend(topology,smiles,static=None):
    old=build_glt_2d_sample(topology,smiles,static=static)
    fresh=build_canonical_periodic_topology(str(topology.normalized_canonical_smiles),max_hops=3)
    if not torch.equal(fresh.mips_x,old.mips_x) or not torch.equal(fresh.mips_backbone_mask,old.mips_backbone_mask):raise ValueError('hop3 atom feature/order mismatch')
    keep=fresh.lga_spd<=2
    for field in ('lga_spd','lga_path_index','lga_path_mask','lga_path_shift','lga_source_image_shift'):
        new=getattr(fresh,field)[keep]
        if new.ndim==2:new=new[:,:3]
        if not torch.equal(new,getattr(old,field)):raise ValueError('hop2 path compatibility mismatch: '+field)
    if not torch.equal(fresh.lga_edge_index[:,keep],old.lga_edge_index):raise ValueError('hop2 edge order mismatch')
    # Every one-hop central observation defines a translation-invariant real
    # bond template. Longer walks translate these observed bond templates.
    chem={}
    for r in (old.lga_spd==1).nonzero().flatten().tolist():
        a,b=old.lga_path_index[r,:2].tolist();qa,qb=old.lga_path_shift[r,:2].tolist()
        key=(a,b,qb-qa);value=old.bond_path_features[r,0]
        if key in chem and not torch.equal(value,chem[key]):raise ValueError('ambiguous bond chemistry')
        chem[key]=value
    count=len(fresh.lga_spd);features=torch.zeros(count,3,14);mask=torch.zeros(count,3,dtype=torch.bool)
    for r,distance in enumerate(fresh.lga_spd.tolist()):
        for k in range(distance):
            a,b=fresh.lga_path_index[r,k:k+2].tolist();qa,qb=fresh.lga_path_shift[r,k:k+2].tolist()
            key=(a,b,qb-qa)
            if key not in chem:raise ValueError('missing real infinite bond template '+str(key))
            features[r,k]=chem[key];mask[r,k]=True
    if not torch.equal(features[keep,:2],old.bond_path_features) or not torch.equal(mask[keep,:2],old.bond_path_mask):raise ValueError('hop2 chemistry mismatch')
    data=old.clone()
    for field in FIELDS:
        if field=='bond_path_features':value=features
        elif field=='bond_path_mask':value=mask
        elif field=='lga_relation_mask':
            value=torch.zeros(count,dtype=torch.bool);value[keep]=old.lga_relation_mask
        else:value=getattr(fresh,field)
        setattr(data,field,value.clone())
    data.opt_schema=SCHEMA
    for h,name in zip((1,2,3),HOP_FIELDS):setattr(data,name,(data.lga_spd<=h).nonzero().flatten())
    return data


def attach_hop_rows(batch,samples):
    if not any(hasattr(x,HOP_FIELDS[0]) for x in samples):return batch
    rows={name:[] for name in HOP_FIELDS};offset=0
    for sample in samples:
        for name in HOP_FIELDS:
            value=getattr(sample,name,None)
            if value is None or value.dtype!=torch.long or value.ndim!=1:
                raise ValueError('missing prepared hop relation view')
            rows[name].append(value+offset)
        offset+=sample.lga_spd.numel()
    for name,value in rows.items():setattr(batch,name,torch.cat(value))
    return batch


def collate_opt(samples):
    return attach_hop_rows(glt_2d_collate(samples),samples)


def numeric_context_record(record,signatures,vocab):
    if len(signatures)!=len(record['clean'].mips_x):raise ValueError('context/atom count mismatch')
    for signature in signatures:
        if signature not in vocab:vocab[signature]=len(vocab)
    record['context_ids']=torch.tensor([vocab[s] for s in signatures],dtype=torch.long)
    record.pop('context',None)
    record['cache_contract']=CACHE_CONTRACT
    return record


def complete_cache(root):
    root=Path(root)
    manifest=json.loads((root/'cache_manifest.json').read_text())
    coverage=json.loads((root/'coverage.json').read_text())
    if (manifest.get('status')!='COMPLETE' or coverage.get('status')!='COMPLETE'
        or manifest.get('contract')!=CACHE_CONTRACT or manifest.get('schema')!=SCHEMA
        or manifest.get('prepared_pt_rows')!=910756 or manifest.get('prepared_downstream_records')!=5875
        or manifest.get('vocab_size',0)<=0):
        raise ValueError('numeric input cache incomplete or incompatible')
    return manifest


def context_signatures(data):
    elements=data.mips_x[:,:101].argmax(-1).tolist();counts=[Counter() for _ in elements];seen=set()
    for r in (data.lga_spd==1).nonzero().flatten().tolist():
        source,target=data.lga_edge_index[:,r].tolist();shift=int(data.lga_source_image_shift[r]);key=(source,shift,target)
        if key in seen:raise ValueError('duplicate physical one-hop observation')
        seen.add(key)
        counts[target][(elements[source],int(data.bond_path_features[r,0,:5].argmax()))]+=1
    return [str(e)+'|'+','.join(f'{z}:{b}:{n}' for (z,b),n in sorted(c.items())) for e,c in zip(elements,counts)]


def read_downstream(root,key,top):
    import lmdb
    env=lmdb.open(str(Path(root)/'downstream.lmdb'),readonly=True,lock=False,readahead=False,max_readers=128)
    try:
        with env.begin() as txn:payload=txn.get(bytes.fromhex(str(key)))
        if payload is None:raise ValueError('missing prepared downstream key')
        data=decode_template(payload)['clean']
        if data.opt_schema!=SCHEMA or not torch.equal(data.mips_x,top.mips_x):raise ValueError('prepared downstream identity/schema mismatch')
        return data
    finally:env.close()


class PreparedOptMicrobatches(torch.utils.data.Dataset):
    def __init__(self,root,cache,eligible,config,rank,prepared_counts,target,*,read_policy='original'):
        if read_policy not in ('original','b3c_io1_readahead'):raise ValueError('unsupported LMDB read policy')
        self.read_policy=read_policy
        self.root=Path(root);self.cache=Path(cache);self.eligible=Path(eligible)
        self.config=config;self.rank=rank;self.prepared_counts=prepared_counts;self.target=target
        self.old_env=self.new_env=None;self.templates=OrderedDict()
    def __len__(self):return 30000
    def setup(self):
        import lmdb
        from src.training.glt_dual_runtime import OrderedSampleStream
        self.indices=np.load(self.eligible,allow_pickle=False)
        if self.indices.shape!=(910756,):raise ValueError('eligible count mismatch')
        self.keys=np.load(self.cache/'keys.npy',allow_pickle=False);self.rows=np.load(self.cache/'source_rows.npy',allow_pickle=False)
        self.stream=OrderedSampleStream(910756,42)
        self.vocab_size=complete_cache(self.root)['vocab_size']
        self.old_env=lmdb.open(str(self.cache/'templates.lmdb'),readonly=True,lock=False,readahead=self.read_policy=='b3c_io1_readahead',max_readers=128)
        self.new_env=lmdb.open(str(self.root/'hop3/pretrain.lmdb'),readonly=True,lock=False,readahead=self.read_policy=='b3c_io1_readahead',max_readers=128)
    def __getitem__(self,index):
        from src.dataset.ua2d_galformer_pretrain import collate_samples
        if self.old_env is None:self.setup()
        samples=[];trace=dict(micro_index=index,positions=[],logical_indices=[],source_rows=[],sample_keys=[])
        worker=torch.utils.data.get_worker_info();slot=worker.id if worker else 0
        with self.old_env.begin() as old,self.new_env.begin() as new:
            for j in range(84):
                position=index*168+self.rank*84+j
                with self.prepared_counts.get_lock():self.prepared_counts[slot]+=1
                logical=int(self.indices[self.stream.index_at(position)]);key=bytes(self.keys[logical]);row=int(self.rows[logical])
                if key not in self.templates:
                    raw=old.get(key);extra=new.get(key)
                    if raw is None or extra is None:raise ValueError('missing prepared PT identity')
                    template=decode_template(raw);extended=decode_template(extra)
                    if (template.get('sample_key')!=key or not bool(template['clean'].geometry_valid)
                        or not torch.isfinite(template['positions']).all()):raise ValueError('invalid eligible geometry/identity')
                    if extended.get('sample_key')!=key or extended.get('cache_contract')!=CACHE_CONTRACT:
                        raise ValueError('prepared numeric template identity/schema mismatch')
                    context_ids=extended.get('context_ids')
                    if (not torch.is_tensor(context_ids) or context_ids.dtype!=torch.long
                        or context_ids.shape!=(len(template['clean'].mips_x),)
                        or (context_ids.numel() and (int(context_ids.min())<0 or int(context_ids.max())>=self.vocab_size))):
                        raise ValueError('missing or invalid persisted integer context labels')
                    for field in (*FIELDS,*HOP_FIELDS):setattr(template['clean'],field,getattr(extended['clean'],field))
                    template['context']=context_ids
                    self.templates[key]=template
                    if len(self.templates)>2048:self.templates.popitem(last=False)
                template=self.templates[key]
                # The old two RNG streams are consumed identically for E/C and all hops.
                data,labels=prepare_opt_sample(template,self.config,position,self.target)
                samples.append((data,labels))
                trace['positions'].append(position);trace['logical_indices'].append(logical)
                trace['source_rows'].append(row);trace['sample_keys'].append(key.hex())
        data,labels=collate_samples(samples)
        attach_hop_rows(data,[x for x,_ in samples])
        return encode((data.to_dict(),labels,trace))


def prepare_opt_sample(template,config,position,target):
    if target not in ('E','C') or not bool(template['clean'].geometry_valid) or not torch.isfinite(template['positions']).all():raise ValueError('eligible record geometry/target mismatch')
    data,labels=prepare_fast_sample(template,config=config,seed=42,position=position)
    if target=='C':labels['atom_label']=template['context']
    return data,labels
