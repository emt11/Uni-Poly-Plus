"""Bounded CPU readers for fold-specific frozen BASE features; no encoder calls."""
import json
from collections import OrderedDict
from pathlib import Path
import numpy as np
import torch
from src.dataset import ua3d_ph_opt_r2 as packed
from src.dataset import ua3d_ph_r2 as spatial

SCHEMA='ua3d-anchor-feature-cache-r2'


class CachedBatch:
    def __init__(self,records,arm,packed_input=None,sides=None):
        self.arm=arm; self.records=records
        self.z=[torch.stack([r['z'][k] for r in records]) for k in range(3)]
        self.atoms=torch.cat([r['atoms'] for r in records],0)
        self.atom_graph=torch.cat([torch.full((len(r['atoms']),),i,dtype=torch.long) for i,r in enumerate(records)])
        self.availability=torch.stack([r['availability'] for r in records])
        self.y_base=torch.stack([r['y_base'] for r in records]).view(-1)
        self.y=torch.stack([r['y'] for r in records]).view(-1)
        self.keys=[r['key'] for r in records]; self.rows=[r['row'] for r in records]
        self.bonds=[r['bonds'] for r in records]
        self.packed=packed_input; self.sides=sides
    def to(self,device):
        # Construct from existing attributes, preserving detached ordinary FP32 inputs.
        result=object.__new__(CachedBatch)
        result.__dict__=dict(self.__dict__)
        for name in ('atoms','atom_graph','availability','y_base','y'):
            setattr(result,name,getattr(self,name).to(device))
        result.z=[x.to(device) for x in self.z]
        result.bonds=[x.to(device) if x is not None else None for x in self.bonds]
        result.packed=packed.move(self.packed,device) if self.packed is not None else None
        result.sides=[move_side(s,device) for s in self.sides] if self.sides is not None else None
        return result


def move_side(x,device):
    if torch.is_tensor(x):
        return x.to(device)
    if isinstance(x,dict):
        return {k:move_side(v,device) for k,v in x.items()}
    if isinstance(x,list):
        return [move_side(v,device) for v in x]
    return x


def collate(samples):
    arm=samples[0]['arm']
    if any(x['arm']!=arm for x in samples):
        raise ValueError('mixed adapter arms')
    records=[x['record'] for x in samples]
    if arm in ('P_PE','P_LEARN'):
        return CachedBatch(records,arm,packed.batch_plan([x['side'] for x in samples],arm))
    return CachedBatch(records,arm,sides=[x['side'] for x in samples] if arm=='G_GEOM' else None)


class CacheData(torch.utils.data.Dataset):
    def __init__(self,cache_root,arm,task,fold,formal,sidecar,packing,split):
        from src.modules.ua3d_anchor_lr_r1 import backend
        arm=backend(arm)
        registry=Path(cache_root)
        if formal and not json.loads((registry.parent.parent/'selection/final_frozen.json').read_text())['frozen']:
            raise ValueError('formal reader before new-cycle freeze')
        alias=json.loads((registry/('test' if formal else 'trainval')/task/f'fold{fold}.json').read_text())
        if alias['status']!='COMPLETE' or alias['formal']!=formal:raise ValueError('unqualified teacher cache alias')
        self.root=Path(alias['source'])
        self.meta=json.loads((self.root/'metadata.json').read_text())
        if self.meta.get('schema')!=SCHEMA or self.meta.get('status')!='COMPLETE' or self.meta['task']!=task or self.meta['fold']!=fold:
            raise ValueError('feature cache incomplete or wrong fold')
        self.arm=arm; self.task=task; self.fold=fold; self.formal=formal
        entry=json.loads((Path(split)/f'{task}.json').read_text())['folds'][fold]
        self.indices=sorted(entry['test_indices']) if formal else sorted(set(entry['train_indices'])|set(entry['validation_indices']))
        position={row:i for i,row in enumerate(self.indices)}
        self.train=[] if formal else [position[row] for row in entry['train_indices']]
        self.validation=list(range(len(self.indices))) if formal else [position[row] for row in entry['validation_indices']]
        self.rows={r['row']:r for r in self.meta['rows']}; self.keys=np.asarray([self.rows[row]['key'] for row in self.indices])
        self.scaler=self.meta['scaler']; self.sidecar=Path(sidecar); self.packing=Path(packing)
        self.lru=OrderedDict(); self.side_lru=OrderedDict(); self.limit=128
        self.router_stats=json.loads((self.sidecar/'scalers'/task/f'fold{fold}.json').read_text()) if arm=='G_GEOM' else None
    def __len__(self):
        return len(self.indices)
    def get_side(self,row,key):
        if row in self.side_lru:
            self.side_lru.move_to_end(row); return self.side_lru[row]
        root=self.packing if self.arm in ('P_PE','P_LEARN') else self.sidecar
        side=torch.load(root/self.task/f'row{row}.pt',map_location='cpu',weights_only=False)
        if side['task']!=self.task or side['row']!=row or side['key']!=key:
            raise ValueError('sidecar key/row mismatch')
        if self.arm=='G_GEOM':
            side['router']=torch.as_tensor((np.asarray(side['router_raw'],np.float64)-np.asarray(self.router_stats['mean']))/np.asarray(self.router_stats['scale']),dtype=torch.float32)
        self.side_lru[row]=side
        if len(self.side_lru)>self.limit:
            self.side_lru.popitem(last=False)
        return side
    def __getitem__(self,i):
        row=self.indices[i]; key=str(self.keys[i])
        if row not in self.lru:
            record=torch.load(self.root/f'row{row}.pt',map_location='cpu',weights_only=False)
            if record['schema']!=SCHEMA or record['task']!=self.task or record['fold']!=self.fold or record['row']!=row or record['key']!=key:
                raise ValueError('teacher record identity mismatch')
            for x in [*record['z'],record['atoms'],record['y_base'],record['y']]+([record['bonds']] if record['bonds'] is not None else []):
                if x.dtype!=torch.float32 or x.requires_grad or x.is_inference() or not torch.isfinite(x).all():
                    raise ValueError('teacher features must be finite ordinary detached FP32')
            self.lru[row]=record
        self.lru.move_to_end(row)
        record=self.lru[row]
        if len(self.lru)>self.limit:
            self.lru.popitem(last=False)
        return dict(arm=self.arm,record=record,side=self.get_side(row,key) if self.arm!='C_ANCHOR' else None)
    def close(self):
        self.lru.clear(); self.side_lru.clear()


def loader(ds,ids,shuffle=False,seed=42):
    return torch.utils.data.DataLoader(torch.utils.data.Subset(ds,ids),batch_size=24,shuffle=shuffle,num_workers=0,drop_last=False,
        collate_fn=collate,generator=torch.Generator().manual_seed(seed) if shuffle else None)
