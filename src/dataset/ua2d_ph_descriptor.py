"""412-dimensional offline finite-Trimer PH; no PH work in the training reader."""
import json
from pathlib import Path
import numpy as np
import torch
from src.training.ua2d_fpdecay_sg import InputCache,finite
from src.dataset.glt_dual import glt_2d_collate

INPUT_SCHEMA='ua2d-ph-descriptor-412-r1'
GROUPS=('ALL','C','CN','CO')
TAUS=np.array([1.5+.1*j for j in range(31)],dtype=np.float64)
STAT_NAMES=('mean','std_ddof0','min','max','sum','count')
COLUMNS=[]
for group in GROUPS:
    for dimension in (0,1):
        COLUMNS += [f'{group}.H{dimension}.betti@{x:.1f}' for x in TAUS]
        COLUMNS += [f'{group}.H{dimension}.{what}.{stat}' for what in ('birth','death','lifetime') for stat in STAT_NAMES]
        COLUMNS += [f'{group}.H{dimension}.censored_count',f'{group}.H{dimension}.finite_present']
    COLUMNS.append(group+'.nonempty')
SEMANTICS=dict(schema=INPUT_SCHEMA,columns=COLUMNS,groups=list(GROUPS),heavy='Z>1; ALL,C,CN,CO',
 coordinate_unit='angstrom',distance_dtype='float64',homology_coeff_field=2,max_edge_length=4.5,max_dimension=2,
 homology_dimensions=[0,1],persistence_dim_max=True,interval_filter='birth<death',betti='birth<=tau<death;divide max(N,1)',
 betti_thresholds=TAUS.tolist(),statistics_ddof=0,finite_statistics='finite deaths only; empty->zero',
 right_censored='death infinity stored as null + right_censored=true; retain all H0',empty_group='103 zeros',
 invalid_geometry='raw412 zeros, ph_valid false; preserve sample',implementation_version='ph-descriptor-412-r1')


def interval_features(intervals,n):
    intervals=np.asarray(intervals,dtype=np.float64).reshape(-1,2)
    if not np.isfinite(intervals[:,0]).all() or np.isnan(intervals[:,1]).any() or (intervals[:,0]<0).any():raise ValueError('invalid PH interval')
    intervals=intervals[intervals[:,0]<intervals[:,1]]
    finite_intervals=intervals[np.isfinite(intervals[:,1])]
    result=[float(np.count_nonzero((intervals[:,0]<=t)&(t<intervals[:,1])))/max(n,1) for t in TAUS]
    for values in (finite_intervals[:,0],finite_intervals[:,1],finite_intervals[:,1]-finite_intervals[:,0]):
        result+=([float(values.mean()),float(values.std(ddof=0)),float(values.min()),float(values.max()),float(values.sum()),float(len(values))] if len(values) else [0.]*6)
    result += [float(np.isinf(intervals[:,1]).sum()),float(len(finite_intervals)>0)]
    return result


def describe(z,positions,valid):
    from src.dataset.mcl_ph_view import persistence_pairs
    z=np.asarray(z,dtype=np.int64);positions=np.asarray(positions,dtype=np.float64)
    if not valid:return np.zeros(412,dtype=np.float32),dict(valid=False,groups=[])
    if z.ndim!=1 or positions.shape!=(len(z),3):raise ValueError('atom/coordinate shape mismatch')
    if not np.isfinite(positions).all():raise ValueError('valid geometry has nonfinite coordinates')
    selectors=(z>1,z==6,np.isin(z,[6,7]),np.isin(z,[6,8]));values=[];groups=[]
    for name,mask in zip(GROUPS,selectors):
        n=int(mask.sum());pairs=persistence_pairs(positions[mask],max_edge=4.5) if n else {0:[],1:[]}
        if n:
            for dim in (0,1):values+=interval_features(pairs[dim],n)
            values.append(1.)
        else:values += [0.]*103
        groups.append(dict(group=name,atoms=n,intervals={str(dim):[dict(birth=float(b),death=float(d) if np.isfinite(d) else None,right_censored=not bool(np.isfinite(d))) for b,d in pairs[dim]] for dim in (0,1)}))
    value=np.asarray(values,dtype=np.float32)
    if value.shape!=(412,) or not np.isfinite(value).all() or (value<0).any():raise ValueError('PH descriptor shape/value mismatch')
    return value,dict(valid=True,groups=groups)


def ph_job(index,z,positions,valid):
    value,barcode=describe(z,positions,valid);return index,value,barcode


def fit_feature_scaler(raw,valid,train_rows):
    raw=np.asarray(raw,dtype=np.float64);valid=np.asarray(valid,dtype=bool);train=np.asarray(train_rows,dtype=np.int64)
    chosen=train[valid[train]]
    if not len(chosen):raise ValueError('no valid training geometry for PH scaler')
    values=np.log1p(raw[chosen]);mean=values.mean(0);std=values.std(0,ddof=0)
    scale=np.where(np.ptp(values,axis=0)==0.,1.,np.maximum(std,1e-6))
    return dict(transform='log1p then train-valid standardization',mean=mean.tolist(),scale=scale.tolist(),
        training_rows=train.tolist(),valid_training_rows=chosen.tolist(),std_ddof=0,dimension=412,columns=COLUMNS)


def transform_features(raw,scaler):
    raw=np.asarray(raw,dtype=np.float64)
    if raw.shape[-1]!=412 or not np.isfinite(raw).all() or (raw<0).any() or scaler.get('columns')!=COLUMNS:raise ValueError('PH feature/scaler identity mismatch')
    mean=np.asarray(scaler['mean']);scale=np.asarray(scaler['scale'])
    if mean.shape!=(412,) or scale.shape!=(412,) or not np.isfinite(mean).all() or not np.isfinite(scale).all() or (scale<=0).any():raise ValueError('invalid PH scaler')
    return ((np.log1p(raw)-mean)/scale).astype(np.float32)


class PHCache:
    def __init__(self,root,graph_root):
        self.root=Path(root);m=json.loads((self.root/'metadata.json').read_text())
        if (m.get('plan')!='UA2D-PH-DESC-20261023-01' or m.get('schema')!=INPUT_SCHEMA or m.get('status')!='COMPLETE'
            or m.get('records')!=5875 or m.get('unique_structures')!=3654 or m.get('semantics')!=SEMANTICS
            or m.get('graph_cache')!=str(Path(graph_root)) or m.get('online_fallback') is not False):raise ValueError('PH cache identity/COMPLETE mismatch')
        self.metadata=m;self.rows=json.loads((self.root/'rows.json').read_text());self.mapping={}
        self.raw=np.load(self.root/'descriptors.npy',mmap_mode='r',allow_pickle=False);self.valid=np.load(self.root/'ph_valid.npy',mmap_mode='r',allow_pickle=False)
        if self.raw.shape!=(3654,412) or self.raw.dtype!=np.float32 or self.valid.shape!=(3654,) or self.valid.dtype!=np.bool_:raise ValueError('PH arrays malformed')
        for task,entries in self.rows.items():
            self.mapping[task]={}
            for e in entries:
                if e['row'] in self.mapping[task] or not 0<=e['ph_index']<3654 or e['cache_key']!=f"{task}:{e['row']}":raise ValueError('PH row mapping conflict')
                self.mapping[task][e['row']]=e
        if sum(len(v) for v in self.mapping.values())!=5875:raise ValueError('PH task coverage mismatch')

    def get(self,task,row,key,scaler):
        e=self.mapping[task].get(int(row))
        if not e or e['key']!=str(key):raise ValueError('PH sample identity mismatch')
        i=e['ph_index'];value=transform_features(self.raw[i],scaler)
        if not self.valid[i] and np.count_nonzero(self.raw[i]):raise ValueError('invalid geometry has nonzero raw PH')
        return torch.tensor(value),torch.tensor(bool(self.valid[i]))


class Data(torch.utils.data.Dataset):
    def __init__(self,task,fold,formal,*,source_root,split_root,graph_root,ph_root,feature_scaler):
        from src.training.glt_dual_runtime import open_source
        from scripts.finetune_mcl_ph import compact_train_validation_indices
        self.task=task;self.feature_scaler=feature_scaler;manifest=json.loads((Path(split_root)/f'{task}.json').read_text());entry=manifest['folds'][fold]
        if manifest.get('validation_is_test') is not False:raise ValueError('validation/TEST not separate')
        if formal:self.indices=sorted(entry['test_indices']);self.train=[];self.validation=list(range(len(self.indices)))
        else:self.indices,self.train,self.validation=compact_train_validation_indices(entry['train_indices'],entry['validation_indices'])
        source,frame=open_source(source_root,source_root,task=task,selected_indices=self.indices,split_manifest=manifest,verify_key_digest=False)
        try:
            if frame.original_row.astype(int).tolist()!=self.indices:raise ValueError('original row order mismatch')
            self.raw_targets=finite(frame.label.to_numpy());self.targets=self.raw_targets.copy();self.keys=np.asarray([x[0].hex() for x in source.samples]);smiles=frame.normalized_smiles.astype(str).tolist()
        finally:source.close()
        self.graphs=InputCache(graph_root);self.ph=PHCache(ph_root,graph_root)
        if self.ph.metadata.get('source')!=str(source_root) or self.ph.metadata.get('split')!=str(split_root):raise ValueError('PH source/split mismatch')
        for row,key,sm in zip(self.indices,self.keys,smiles):
            ge=self.graphs.mapping[task].get(row);pe=self.ph.mapping[task].get(row)
            if not ge or not pe or ge['key']!=key or pe['key']!=key or pe['smiles']!=sm:raise ValueError('graph/PH/source identity mismatch')
    def __len__(self):return len(self.indices)
    def __getitem__(self,i):
        d=self.graphs.get(self.task,self.indices[i],self.keys[i]);d.y=torch.tensor([self.targets[i]],dtype=torch.float32)
        d.ph_features,d.ph_valid=self.ph.get(self.task,self.indices[i],self.keys[i],self.feature_scaler);return d
    def set_target_override(self,y):self.targets=np.asarray(y,dtype=np.float64)
    def close(self):self.graphs.close()


def collate(samples):
    batch=glt_2d_collate(samples);batch.ph_features=torch.stack([s.ph_features for s in samples]);batch.ph_valid=torch.stack([s.ph_valid for s in samples]);return batch


def loader(ds,ids,shuffle=False,seed=42):
    return torch.utils.data.DataLoader(torch.utils.data.Subset(ds,ids),batch_size=24,shuffle=shuffle,num_workers=0,
        drop_last=False,collate_fn=collate,generator=torch.Generator().manual_seed(seed) if shuffle else None)
