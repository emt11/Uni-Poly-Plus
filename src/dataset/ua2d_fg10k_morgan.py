"""Explicit cached graph and real Morgan row identity; no online construction."""
import json
from pathlib import Path
import numpy as np
import torch
from src.training.ua2d_fpdecay_sg import InputCache,finite
from src.dataset.glt_dual import glt_2d_collate

INPUT_SCHEMA='ua2d-fg10k-morgan-input-r1'
FP_SPEC=dict(kind='MorganBinary',radius=2,fpSize=2048,includeChirality=True,useBondTypes=True,
 countSimulation=False,includeRingMembership=True,onlyNonzeroInvariants=False,includeRedundantEnvironments=False,
 countBounds=None,atomInvariantsGenerator=None,bondInvariantsGenerator=None,
 input='unchanged normalized_smiles with dummy connection atoms')


def validate_bits(blob):
    bits=np.frombuffer(blob,dtype=np.uint8)
    if bits.shape!=(2048,) or not np.isin(bits,[0,1]).all() or not 0<int(bits.sum())<2048:
        raise ValueError('FP must be2048 uint8 binary values, not256 packed bytes')
    return bits


class MorganCache:
    def __init__(self,root,graph_root):
        self.root=Path(root);m=json.loads((self.root/'metadata.json').read_text())
        if (m.get('plan')!='UA2D-FG10K-MORGAN-20261023-01' or m.get('schema')!=INPUT_SCHEMA or m.get('status')!='COMPLETE' or m.get('records')!=5875
            or m.get('unique_smiles')!=3654 or m.get('feature_spec')!=FP_SPEC
            or m.get('graph_cache')!=str(Path(graph_root)) or m.get('dtype')!='uint8'
            or m.get('payload')!='2048 uint8 binary entries, not packbits' or m.get('online_fallback') is not False):
            raise ValueError('Morgan cache identity/schema mismatch')
        self.metadata=m;self.rows=json.loads((self.root/'rows.json').read_text());self.mapping={}
        self.bits=np.load(self.root/'morgan.npy',mmap_mode='r',allow_pickle=False)
        if self.bits.shape!=(5875,2048) or self.bits.dtype!=np.uint8:raise ValueError('FP materialization shape mismatch')
        indices=[]
        for task,rows in self.rows.items():
            self.mapping[task]={}
            for e in rows:
                if e['row'] in self.mapping[task] or e['cache_key']!=f"{task}:{e['row']}":raise ValueError('duplicate FP row identity')
                self.mapping[task][e['row']]=e;indices.append(e['fp_index'])
        if sorted(indices)!=list(range(5875)):raise ValueError('FP index coverage conflict')

    def get(self,task,row,key):
        e=self.mapping[task].get(int(row))
        if not e or e['key']!=str(key):raise ValueError('FP sample key/row conflict')
        value=self.bits[e['fp_index']];validate_bits(value.tobytes())
        return torch.tensor(value.copy(),dtype=torch.float32)


class Data(torch.utils.data.Dataset):
    def __init__(self,task,fold,formal,*,source_root,split_root,graph_root,fp_root):
        from src.training.glt_dual_runtime import open_source
        from scripts.finetune_mcl_ph import compact_train_validation_indices
        self.task=task;manifest=json.loads((Path(split_root)/f'{task}.json').read_text());entry=manifest['folds'][fold]
        if manifest.get('validation_is_test') is not False:raise ValueError('validation/TEST not separate')
        if formal:self.indices=sorted(entry['test_indices']);self.train=[];self.validation=list(range(len(self.indices)))
        else:self.indices,self.train,self.validation=compact_train_validation_indices(entry['train_indices'],entry['validation_indices'])
        source,frame=open_source(source_root,source_root,task=task,selected_indices=self.indices,split_manifest=manifest,verify_key_digest=False)
        try:
            if frame.original_row.astype(int).tolist()!=self.indices:raise ValueError('original row order mismatch')
            self.raw_targets=finite(frame.label.to_numpy());self.targets=self.raw_targets.copy()
            self.keys=np.asarray([x[0].hex() for x in source.samples]);smiles=frame.normalized_smiles.astype(str).tolist()
        finally:source.close()
        self.graphs=InputCache(graph_root);self.fp=MorganCache(fp_root,graph_root)
        if self.fp.metadata.get('source')!=str(source_root) or self.fp.metadata.get('split')!=str(split_root):raise ValueError('FP source/split path identity mismatch')
        for row,key,sm in zip(self.indices,self.keys,smiles):
            ge=self.graphs.mapping[task].get(row);fe=self.fp.mapping[task].get(row)
            if not ge or not fe or ge['key']!=key or fe['key']!=key or fe['smiles']!=sm:raise ValueError('graph/FP/source identity mismatch')

    def __len__(self):return len(self.indices)
    def __getitem__(self,i):
        d=self.graphs.get(self.task,self.indices[i],self.keys[i]);d.y=torch.tensor([self.targets[i]],dtype=torch.float32)
        d.morgan_fp=self.fp.get(self.task,self.indices[i],self.keys[i]);return d
    def set_target_override(self,y):self.targets=np.asarray(y,dtype=np.float64)
    def close(self):self.graphs.close()


def collate(samples):
    batch=glt_2d_collate(samples);batch.morgan_fp=torch.stack([s.morgan_fp for s in samples]);return batch


def loader(ds,ids,shuffle=False,seed=42):
    return torch.utils.data.DataLoader(torch.utils.data.Subset(ds,ids),batch_size=24,shuffle=shuffle,num_workers=0,
        drop_last=False,collate_fn=collate,generator=torch.Generator().manual_seed(seed) if shuffle else None)
