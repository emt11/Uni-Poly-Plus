"""Read-only UA inputs, packed FP decoding and availability-aware subbatches."""
from pathlib import Path
import json
import lmdb
import numpy as np
import torch
from torch_geometric.data import Data as GraphData
from src.dataset.glt_dual import glt_2d_collate,dual_glt_collate
from src.dataset.ua2d_fast_pretrain import decode_template,encode
from src.dataset.ua2d_p10k_nofp_align import (PreparedNoFPMicrobatches,prepare_arm_sample,collate_arm_samples,micro_start,resolve_slot)
from src.modules.ua2d_p10k_nofp_align import arm_config
from src.dataset.ua2d_a020_fp_fusion import MorganCache,source_rows,validate_join
from scripts.finetune_mcl_ph import compact_train_validation_indices

CACHE_SCHEMA='ua3m-paperfuse-input-r1'


def unpack_fp(bits):
    packed=np.asarray(bits.cpu().numpy() if torch.is_tensor(bits) else bits)
    if packed.dtype!=np.uint8 or packed.shape!=(256,):raise ValueError('packed Morgan2048 schema mismatch')
    value=np.unpackbits(packed,bitorder='little')
    if value.shape!=(2048,) or not 0<int(value.sum())<2048:raise ValueError('invalid packed FP bits')
    return torch.tensor(value,dtype=torch.float32)


class ModalBatch:
    def __init__(self,two_d,three_d,rows,fp,availability,identity,y=None):
        self.two_d=two_d;self.three_d=three_d;self.three_d_rows=rows;self.morgan_fp=fp;self.availability=availability;self.identity=identity;self.y=y
    def to(self,device,non_blocking=False):
        return ModalBatch(self.two_d.to(device,non_blocking=non_blocking),self.three_d.to(device,non_blocking=non_blocking) if self.three_d is not None else None,
            self.three_d_rows.to(device,non_blocking=non_blocking),self.morgan_fp.to(device,non_blocking=non_blocking),self.availability.to(device,non_blocking=non_blocking),self.identity,
            self.y.to(device,non_blocking=non_blocking) if self.y is not None else None)


def pt_batch(data,labels):
    n=len(labels['identity']);return ModalBatch(data,data,torch.arange(n,device=data.morgan_fp.device),data.morgan_fp,torch.ones((n,3),dtype=torch.bool,device=data.morgan_fp.device),labels['identity'])


class PTData(PreparedNoFPMicrobatches):
    def __init__(self,root,mapping,prepared_counts):super().__init__(root,mapping,arm_config('D_A020'),prepared_counts=prepared_counts)
    def __getitem__(self,micro_index):
        if self.environment is None:self._open()
        info=torch.utils.data.get_worker_info();worker=info.id if info else 0;start=micro_start(int(micro_index))
        records=[];rows=[];slots=[];fps=[]
        for offset in range(168):
            with self.prepared_counts.get_lock():self.prepared_counts[worker]+=1
            position=start+offset;selected=self.stream.index_at(position)
            index,key,source_row=resolve_slot(self.eligible,self.keys,self.rows,selected)
            if key!=self.selected_keys[selected].tobytes() or source_row!=int(self.selected_rows[selected]):raise ValueError('PT logical/key/source-row mismatch')
            if key in self.templates:template=self.templates.pop(key)
            else:
                with self.environment.begin() as txn:payload=txn.get(key)
                if payload is None:raise ValueError('missing immutable PT template')
                template=decode_template(payload)
            if template['sample_key']!=key or not bool(template['clean'].geometry_valid):raise ValueError('PT invalid geometry/identity excluded from every objective')
            self.templates[key]=template
            if len(self.templates)>self.cache_entries:self.templates.popitem(last=False)
            records.append(prepare_arm_sample(template,config=self.config,seed=42,position=position));fps.append(unpack_fp(template['fp_bits']));rows.append(source_row);slots.append(index)
        data,labels=collate_arm_samples(records)
        if 'fp_target' in labels:raise ValueError('FP supervision is forbidden')
        data.morgan_fp=torch.stack(fps)
        return encode((data,labels,dict(micro_index=int(micro_index),start_position=start,source_rows=rows,logical_slots=slots)))


def worker_init(worker):
    torch.set_num_threads(1)


class InputCache:
    def __init__(self,root):
        self.root=Path(root);self.metadata=json.loads((self.root/'metadata.json').read_text())
        if self.metadata.get('schema')!=CACHE_SCHEMA or self.metadata.get('plan')!='UA3M-PAPERFUSE-20261025-01' or self.metadata.get('status')!='COMPLETE' or self.metadata.get('records')!=5875 or self.metadata.get('geometry_available')!=5860 or self.metadata.get('geometry_missing')!=15:
            raise ValueError('three-modal cache incomplete/schema mismatch')
        self.rows=json.loads((self.root/'rows.json').read_text());self.mapping={};self.memory={}
        for task,entries in self.rows.items():
            self.mapping[task]={}
            for entry in entries:
                row=entry['row']
                if row in self.mapping[task] or entry['cache_key']!=f'{task}:{row}':raise ValueError('cache duplicate row')
                self.mapping[task][row]=entry
        self.env=lmdb.open(str(self.root/'graphs.lmdb'),readonly=True,lock=False,readahead=True,max_readers=128)
        if self.env.stat()['entries']!=5875:raise ValueError('cache physical row count mismatch')
    def get(self,task,row,key,smiles):
        entry=self.mapping[task].get(row)
        if not entry or entry['key']!=key or entry['smiles']!=smiles:raise ValueError('three-modal row/key/SMILES join mismatch')
        cache_key=entry['cache_key']
        if cache_key not in self.memory:
            with self.env.begin() as txn:payload=txn.get(cache_key.encode())
            if payload is None:raise ValueError('cache miss; no online fallback')
            record=decode_template(payload)
            if record['schema']!=CACHE_SCHEMA or record['task']!=task or record['row']!=row or record['sample_key']!=key or record['smiles']!=smiles or record['available']!=entry['available']:
                raise ValueError('materialized row identity changed')
            if record['geometry'] is not None:record['geometry']=GraphData.from_dict(record['geometry'])
            if (record['geometry'] is not None)!=entry['available']:raise ValueError('geometry availability/physical payload mismatch')
            self.memory[cache_key]=record
        x=self.memory[cache_key]
        return dict(two_d=x['clean'].clone(),three_d=x['geometry'].clone() if x['geometry'] is not None else None,
                    fp=x['fp'].clone(),available=x['available'],identity=x['smiles'])
    def close(self):self.env.close()


class Data(torch.utils.data.Dataset):
    def __init__(self,task,fold,formal,source,split,cache):
        manifest=json.loads((Path(split)/f'{task}.json').read_text());entry=manifest['folds'][fold]
        if formal:self.indices=sorted(entry['test_indices']);self.train=[];self.validation=list(range(len(self.indices)))
        else:self.indices,self.train,self.validation=compact_train_validation_indices(entry['train_indices'],entry['validation_indices'])
        self.keys,self.raw_targets,self.smiles=source_rows(task,self.indices,source,split);self.targets=self.raw_targets.copy();self.task=task;self.cache=InputCache(cache)
        for row,key,sm in zip(self.indices,self.keys,self.smiles):
            e=self.cache.mapping[task].get(row)
            if not e or e['key']!=key or e['smiles']!=sm:raise ValueError('original split/source/cache row mismatch')
    def __len__(self):return len(self.indices)
    def __getitem__(self,i):
        x=self.cache.get(self.task,self.indices[i],self.keys[i],self.smiles[i]);x['y']=torch.tensor(float(self.targets[i]),dtype=torch.float32);return x
    def close(self):self.cache.close()


def collate(samples):
    ids=[i for i,x in enumerate(samples) if x['available']]
    two=glt_2d_collate([x['two_d'] for x in samples]);three=dual_glt_collate([samples[i]['three_d'] for i in ids]) if ids else None
    available=torch.tensor([[True,x['available'],True] for x in samples],dtype=torch.bool)
    return ModalBatch(two,three,torch.tensor(ids,dtype=torch.long),torch.stack([x['fp'] for x in samples]),available,[x['identity'] for x in samples],torch.stack([x['y'] for x in samples]))


def loader(ds,indices,shuffle=False,seed=42):
    return torch.utils.data.DataLoader(torch.utils.data.Subset(ds,indices),batch_size=24,shuffle=shuffle,num_workers=0,
        drop_last=False,collate_fn=collate,generator=torch.Generator().manual_seed(seed) if shuffle else None)


def validate_missing_inventory(missing,total=5875,available=5860):
    tasks=('eea','egb','egc','ei','eps','nc','xc');counts={task:sum(x['task']==task for x in missing) for task in tasks}
    if total!=5875 or available!=5860 or len(missing)!=15 or counts!={'eea':0,'egb':7,'egc':8,'ei':0,'eps':0,'nc':0,'xc':0} or len({x['smiles'] for x in missing})!=8:
        raise ValueError('fixed15 missing geometry inventory mismatch')
    if any(x['reason']=='PASS' for x in missing):raise ValueError('PASS cannot be added to missing whitelist')
    if len({(x['task'],x['row'],x['key']) for x in missing})!=15:raise ValueError('duplicate missing sample identity')
    return counts
