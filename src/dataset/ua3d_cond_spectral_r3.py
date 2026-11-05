"""Read-only raw graphs and fixed surface moments; pinned, bounded loaders."""
import json
from pathlib import Path
import numpy as np
import torch
from src.dataset import ua3m_paperfuse as raw
from src.dataset.ua2d_fast_pretrain import decode_batch
from src.dataset.ua3d_hmr_r2 import finite
PTData=raw.PTData
worker_init=raw.worker_init

class ModalBatch(raw.ModalBatch):
    def __init__(self,batch,surface):
        super().__init__(batch.two_d,batch.three_d,batch.three_d_rows,batch.morgan_fp,batch.availability,batch.identity,batch.y)
        self.surface=surface
    def pin_memory(self):
        self.two_d=self.two_d.pin_memory()
        if self.three_d is not None:self.three_d=self.three_d.pin_memory()
        for key in ('three_d_rows','morgan_fp','availability','y'):
            value=getattr(self,key)
            if value is not None:setattr(self,key,value.pin_memory())
        self.surface={k:v.pin_memory() for k,v in self.surface.items()};return self
    def to(self,device,non_blocking=True):
        return ModalBatch(super().to(device,non_blocking=non_blocking),{k:v.to(device,non_blocking=non_blocking) for k,v in self.surface.items()})

def pt_batch(data,labels):return raw.pt_batch(data,labels)

def tensor_storages(value,seen=None):
    """Unique live CPU storage estimates; excludes Python/mapping overhead."""
    result={} if seen is None else seen
    if torch.is_tensor(value):
        if value.device.type!='cpu':raise ValueError('raw template must be CPU')
        store=value.untyped_storage();result[(store.data_ptr(),store.nbytes())]=store.nbytes()
    elif hasattr(value,'to_dict'):tensor_storages(value.to_dict(),result)
    elif isinstance(value,dict):
        for item in value.values():tensor_storages(item,result)
    elif isinstance(value,(tuple,list)):
        for item in value:tensor_storages(item,result)
    elif isinstance(value,memoryview):raise ValueError('borrowed memoryview in raw template')
    return result

class TemplateLRU:
    def __init__(self,entries=128,max_bytes=256*1024**2):
        from collections import OrderedDict
        self.entries=entries;self.max_bytes=max_bytes;self.data=OrderedDict();self.refs={};self.bytes=0
    def get(self,key):
        if key not in self.data:return None
        item=self.data.pop(key);self.data[key]=item;return item[0]
    def _evict(self):
        _,(_,storages)=self.data.popitem(last=False)
        for key,size in storages.items():
            self.refs[key]-=1
            if self.refs[key]==0:self.bytes-=size;del self.refs[key]
    def put(self,key,value):
        if key in self.data:raise ValueError('duplicate template cache insert')
        storages=tensor_storages(value)
        if sum(storages.values())>self.max_bytes:return False
        self.data[key]=(value,storages)
        for item,size in storages.items():
            if item not in self.refs:self.refs[item]=0;self.bytes+=size
            self.refs[item]+=1
        while len(self.data)>self.entries or self.bytes>self.max_bytes:self._evict()
        return True

class DecodedPT(PTData):
    def __init__(self,root,mapping,prepared_counts):
        super().__init__(root,mapping,prepared_counts)
        self.lru=TemplateLRU();self.source_micros=0;self.reopens=0;self.reopen_seconds=0.;self.hits=0;self.misses=0
    def _open_environment(self):
        import lmdb
        self.environment=lmdb.open(str(self.root/'templates.lmdb'),readonly=True,lock=False,readahead=False,max_readers=128)
    def _open(self):
        from src.training.glt_dual_runtime import OrderedSampleStream
        meta=json.loads((self.root/'metadata.json').read_text());mapping=json.loads((self.mapping_root/'metadata.json').read_text())
        if meta.get('schema')!='ua2d_prepared_templates_r4' or meta.get('status')!='COMPLETE' or meta.get('train_rows')!=911391:raise ValueError('PT raw template metadata')
        if mapping.get('schema')!='ua2d-ptscale-logical-map-r1' or mapping.get('status')!='COMPLETE' or mapping.get('records')!=910756 or mapping.get('parent_cache')!=str(self.root):raise ValueError('logical mapping identity')
        self.keys=np.load(self.root/'keys.npy',mmap_mode='r',allow_pickle=False);self.rows=np.load(self.root/'source_rows.npy',mmap_mode='r',allow_pickle=False)
        self.eligible=np.load(self.mapping_root/'eligible_logical.npy',mmap_mode='r',allow_pickle=False)
        self.selected_keys=np.load(self.mapping_root/'keys.npy',mmap_mode='r',allow_pickle=False);self.selected_rows=np.load(self.mapping_root/'source_rows.npy',mmap_mode='r',allow_pickle=False)
        if self.keys.shape!=(911391,) or self.rows.shape!=(911391,) or self.eligible.shape!=(910756,) or self.selected_keys.shape!=(910756,) or self.selected_rows.shape!=(910756,):raise ValueError('PT mapping dimensions')
        self.stream=OrderedSampleStream(910756,42);self._open_environment()
    def __getitem__(self,index):
        import time
        started=time.monotonic()
        if self.environment is None:self._open()
        if self.source_micros and self.source_micros%64==0:
            tick=time.monotonic()
            # All read transactions exited; buffers=False returns owned bytes.
            # Decoded numpy/tensors never refer to the LMDB mmap.
            self.environment.close();self.environment=None;self._open_environment()
            self.reopens+=1;self.reopen_seconds+=time.monotonic()-tick
        info=torch.utils.data.get_worker_info();worker=info.id if info else 0;start=raw.micro_start(int(index))
        records=[];rows=[];slots=[];fps=[]
        for offset in range(168):
            with self.prepared_counts.get_lock():self.prepared_counts[worker]+=1
            position=start+offset;selected=self.stream.index_at(position)
            logical,key,row=raw.resolve_slot(self.eligible,self.keys,self.rows,selected)
            if key!=self.selected_keys[selected].tobytes() or row!=int(self.selected_rows[selected]):raise ValueError('PT logical/key/source mismatch')
            template=self.lru.get(key)
            if template is None:
                self.misses+=1
                with self.environment.begin(write=False,buffers=False) as txn:payload=txn.get(key)
                if payload is None or not isinstance(payload,bytes):raise ValueError('missing/non-owned LMDB payload')
                template=raw.decode_template(payload);tensor_storages(template);self.lru.put(key,template)
            else:self.hits+=1
            if template['sample_key']!=key or not bool(template['clean'].geometry_valid):raise ValueError('PT valid geometry/identity')
            records.append(raw.prepare_arm_sample(template,config=self.config,seed=42,position=position))
            fps.append(raw.unpack_fp(template['fp_bits']));rows.append(row);slots.append(logical)
        data,labels=raw.collate_arm_samples(records)
        if 'fp_target' in labels:raise ValueError('FP prediction forbidden')
        data.morgan_fp=torch.stack(fps);self.source_micros+=1
        return data,labels,dict(micro_index=int(index),start_position=start,source_rows=rows,logical_slots=slots,
            io=dict(worker=worker,source_microbatches=self.source_micros,lru_entries=len(self.lru.data),lru_storage_bytes=self.lru.bytes,hits=self.hits,misses=self.misses,reopens=self.reopens,reopen_seconds=self.reopen_seconds,source_seconds=time.monotonic()-started,buffers=False,readahead=False))

class PinnedPT:
    def __init__(self,data,labels,trace):self.data=data;self.labels=labels;self.trace=trace
    def pin_memory(self):
        self.data=self.data.pin_memory();self.labels={k:v.pin_memory() if torch.is_tensor(v) else v for k,v in self.labels.items()};return self

def pt_collate(value):return PinnedPT(*value)

class Data(raw.Data):
    def __init__(self,task,fold,formal,source,split,cache,surface,stats):
        super().__init__(task,fold,formal,source,split,cache)
        with np.load(Path(surface)/'surface/compact'/f'{task}.npz',allow_pickle=False) as f:self.surf={k:f[k] for k in ('lam','C','a','A','Ac','stat','mode_mask','mask')}
        status=json.loads((Path(surface)/'surface/status.json').read_text());entries={e['row']:e for e in status if e['task']==task}
        allrows=self.cache.rows[task]
        if len(allrows)!=len(self.surf['mask']):raise ValueError('surface row count')
        mapping={e['row']:i for i,e in enumerate(allrows)}
        self.surf_positions=[]
        for row,key in zip(self.indices,self.keys):
            e=entries[row]
            if e['key']!=key or bool(self.surf['mask'][mapping[row]])!=e['available']:raise ValueError('surface row/key/mask identity')
            self.surf_positions.append(mapping[row])
        self.stats=stats
    def __getitem__(self,i):
        x=super().__getitem__(i);j=self.surf_positions[i]
        s={k:torch.as_tensor(v[j].copy() if isinstance(v[j],np.ndarray) else v[j],dtype=torch.bool if k=='mode_mask' else torch.float32) for k,v in self.surf.items()}
        if bool(s['mask']):
            for value in s.values():
                if not torch.isfinite(value).all():raise FloatingPointError('valid surface nonfinite')
            s['stat']=(s['stat']-torch.tensor(self.stats['mean']))/torch.tensor(self.stats['std'])
        else:
            s={k:torch.zeros_like(v) for k,v in s.items()}
            s['A'].fill_(1);s['Ac'].fill_(1)
        x['surface']=s;return x

def collate(samples):return ModalBatch(raw.collate(samples),{k:torch.stack([x['surface'][k] for x in samples]) for k in samples[0]['surface']})

def loader(ds,indices,shuffle=False,seed=42):
    return torch.utils.data.DataLoader(torch.utils.data.Subset(ds,indices),batch_size=24,shuffle=shuffle,num_workers=4,
        multiprocessing_context='fork',worker_init_fn=worker_init,prefetch_factor=2,persistent_workers=False,pin_memory=True,
        drop_last=False,collate_fn=collate,generator=torch.Generator().manual_seed(seed) if shuffle else None)
