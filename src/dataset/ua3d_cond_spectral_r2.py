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

class DecodedPT(PTData):
    def __getitem__(self,index):return decode_batch(super().__getitem__(index))

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
