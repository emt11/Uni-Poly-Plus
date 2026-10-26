"""Read-only graph/FP join by task, original row and existing sample key."""
from pathlib import Path
import json
import numpy as np
import torch
from src.training.ua2d_p10k_nofp_align import InputCache, finite
from src.dataset.glt_dual import glt_2d_collate
from src.training.glt_dual_runtime import open_source
from scripts.finetune_mcl_ph import compact_train_validation_indices

FP_SPEC=dict(kind='MorganBinary',radius=2,fpSize=2048,includeChirality=True,useBondTypes=True,
 countSimulation=False,includeRingMembership=True,onlyNonzeroInvariants=False,includeRedundantEnvironments=False,
 countBounds=None,atomInvariantsGenerator=None,bondInvariantsGenerator=None,
 input='unchanged normalized_smiles with dummy connection atoms')


def validate_bits(bits):
    if bits.shape!=(2048,) or bits.dtype!=np.uint8 or not np.isin(bits,[0,1]).all() or not 0<int(bits.sum())<2048:
        raise ValueError('FP must be nonempty/nonsaturated 2048 uint8 binary entries')


def validate_join(graph,fp,row,key,smiles):
    if not graph or not fp or graph.get('row')!=row or fp.get('row')!=row or graph.get('key')!=key or fp.get('key')!=key or fp.get('smiles')!=smiles:
        raise ValueError('task/row/key/normalized SMILES identity mismatch')


class MorganCache:
    def __init__(self,root,graph_root,source_root,split_root):
        self.root=Path(root);self.metadata=m=json.loads((self.root/'metadata.json').read_text())
        if (m.get('plan')!='UA2D-FG10K-MORGAN-20261023-01' or m.get('schema')!='ua2d-fg10k-morgan-input-r1'
            or m.get('status')!='COMPLETE' or m.get('records')!=5875 or m.get('unique_smiles')!=3654
            or m.get('feature_spec')!=FP_SPEC or m.get('dtype')!='uint8' or m.get('shape')!=[5875,2048]
            or m.get('payload')!='2048 uint8 binary entries, not packbits' or m.get('online_fallback') is not False
            or m.get('graph_cache')!=str(graph_root) or m.get('source')!=str(source_root) or m.get('split')!=str(split_root)):
            raise ValueError('Morgan cache metadata identity mismatch')
        self.bits=np.load(self.root/'morgan.npy',mmap_mode='r',allow_pickle=False)
        if self.bits.shape!=(5875,2048) or self.bits.dtype!=np.uint8:raise ValueError('FP tensor schema mismatch')
        rows=json.loads((self.root/'rows.json').read_text());self.mapping={};indices=[]
        for task,entries in rows.items():
            self.mapping[task]={}
            for e in entries:
                if e['row'] in self.mapping[task] or e['cache_key']!=f"{task}:{e['row']}":raise ValueError('duplicate FP identity')
                self.mapping[task][e['row']]=e;indices.append(e['fp_index'])
        if sorted(indices)!=list(range(5875)):raise ValueError('FP index coverage mismatch')
    def get(self,task,row,key):
        e=self.mapping[task].get(int(row))
        if not e or e['key']!=str(key):raise ValueError('FP row/key mismatch')
        bits=self.bits[e['fp_index']];validate_bits(bits)
        return torch.tensor(bits.copy(),dtype=torch.float32)


def source_rows(task,indices,source_root,split_root):
    manifest=json.loads((Path(split_root)/f'{task}.json').read_text())
    if manifest.get('validation_is_test') is not False:raise ValueError('validation/TEST separation invalid')
    source,frame=open_source(source_root,source_root,task=task,selected_indices=indices,split_manifest=manifest,verify_key_digest=False)
    try:
        if frame.original_row.astype(int).tolist()!=indices:raise ValueError('source original row mismatch')
        keys=np.asarray([x[0].hex() for x in source.samples]);targets=finite(frame.label.to_numpy())
        smiles=frame.normalized_smiles.astype(str).tolist()
    finally:source.close()
    return keys,targets,smiles


class Data(torch.utils.data.Dataset):
    def __init__(self,task,fold,formal,*,source_root,split_root,graph_root,fp_root,fp_only=False):
        entry=json.loads((Path(split_root)/f'{task}.json').read_text())['folds'][fold];self.task=task;self.fp_only=fp_only
        if formal:self.indices=sorted(entry['test_indices']);self.train=[];self.validation=list(range(len(self.indices)))
        else:self.indices,self.train,self.validation=compact_train_validation_indices(entry['train_indices'],entry['validation_indices'])
        self.keys,self.raw_targets,smiles=source_rows(task,self.indices,source_root,split_root)
        self.targets=self.raw_targets.copy();self.graphs=InputCache(graph_root)
        self.fp=MorganCache(fp_root,graph_root,source_root,split_root)
        for row,key,sm in zip(self.indices,self.keys,smiles):
            validate_join(self.graphs.mapping[task].get(row),self.fp.mapping[task].get(row),row,key,sm)
    def __len__(self):return len(self.indices)
    def __getitem__(self,i):
        fp=self.fp.get(self.task,self.indices[i],self.keys[i]);y=torch.tensor([self.targets[i]],dtype=torch.float32)
        if self.fp_only:return dict(morgan_fp=fp,y=y)
        d=self.graphs.get(self.task,self.indices[i],self.keys[i]);d.y=y;d.morgan_fp=fp;return d
    def close(self):self.graphs.close()


class FPBatch:
    def __init__(self,fp,y):self.morgan_fp=fp;self.y=y
    def to(self,device):return FPBatch(self.morgan_fp.to(device),self.y.to(device))


def collate(samples):
    if isinstance(samples[0],dict):return FPBatch(torch.stack([x['morgan_fp'] for x in samples]),torch.cat([x['y'] for x in samples]))
    batch=glt_2d_collate(samples);batch.morgan_fp=torch.stack([x.morgan_fp for x in samples]);return batch


def loader(ds,ids,shuffle=False,seed=42):
    return torch.utils.data.DataLoader(torch.utils.data.Subset(ds,ids),batch_size=24,shuffle=shuffle,num_workers=0,
        drop_last=False,collate_fn=collate,generator=torch.Generator().manual_seed(seed) if shuffle else None)
