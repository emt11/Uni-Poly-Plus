#!/usr/bin/env python3
"""Normal source qualification and train-only statistics; no preflight suite."""
import os,sys,json,math,concurrent.futures as cf
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.training import ua3d_cond_spectral_r2 as t
from src.modules import ua3d_cond_spectral_r2 as m
from src.dataset import ua3m_paperfuse as raw
import numpy as np
import torch

def qualify(task,fold):
    torch.set_num_threads(1);t.check_clock(True);cfg=t.source_config(task,fold)
    ds=raw.Data(task,fold,False,t.DOWN,t.SPLIT,cfg['input_cache'])
    try:
        scaler=t.read(cfg['scaler_reference']);t.scaler_validate(ds.raw_targets,ds.train,scaler)
        with np.load(t.SURFACE/'surface/compact'/f'{task}.npz',allow_pickle=False) as z:surf={k:z[k] for k in ('lam','C','a','A','Ac','stat','mode_mask','mask')}
        rows=ds.cache.rows[task];mapping={e['row']:i for i,e in enumerate(rows)}
        status={e['row']:e for e in t.read(t.SURFACE/'surface/status.json') if e['task']==task}
        if len(rows)!=len(surf['mask']):raise ValueError('surface length')
        positions=[]
        for i,(row,key) in enumerate(zip(ds.indices,ds.keys)):
            j=mapping[row];e=status[row]
            if e['key']!=key or e['available']!=bool(surf['mask'][j]):raise ValueError('surface identity mismatch')
            # Decode only train/validation graphs. No TEST labels or representations.
            x=ds[i]
            if x['available']!=ds.cache.mapping[task][row]['available']:raise ValueError('graph availability')
            if surf['mask'][j]:
                for k in surf:
                    if not np.isfinite(surf[k][j]).all():raise FloatingPointError('valid surface nonfinite')
            if i in ds.train and surf['mask'][j]:positions.append(j)
        if not positions:raise ValueError('empty valid training surfaces')
        values=surf['stat'][positions].astype(np.float64);mean=values.mean(0);std=values.std(0);std[std<1e-12]=1.
        t.exclusive(t.OUT/'scalers'/task/f'fold{fold}.json',dict(mean=mean.tolist(),std=std.tolist(),valid_train=len(positions),train_only=True))
        return dict(task=task,fold=fold,train=len(ds.train),validation=len(ds.validation),updates=70*math.ceil(len(ds.train)/24),input_cache=cfg['input_cache'],scaler=cfg['scaler_reference'],rows_and_keys_verified=True,test_targets_read=False)
    finally:ds.close()

def main():
    t.require_tmux()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU prepare GPU visibility')
    torch.set_num_threads(1);t.check_clock(True)
    pt=t.read(t.PT_CACHE/'metadata.json');mapping=t.read(t.MAP_ROOT/'metadata.json')
    if pt['schema']!='ua2d_prepared_templates_r4' or pt['train_rows']!=911391 or pt['status']!='COMPLETE' or mapping['records']!=910756 or mapping['excluded']!=635 or mapping['eligibility_semantics']!='logical_slot_not_source_row':raise ValueError('raw PT metadata')
    splits={task:t.read(t.SPLIT/f'{task}.json') for task in t.TASKS};counts=dict(train=0,validation=0,test=0,updates=0)
    for task,split in splits.items():
        if split['validation_is_test'] is not False:raise ValueError('TEST/validation split')
        for entry in split['folds']:
            sets=[set(entry[k+'_indices']) for k in ('train','validation','test')]
            if any(sets[i]&sets[j] for i in range(3) for j in range(i)):raise ValueError('split overlap')
            for name in ('train','validation','test'):counts[name]+=len(entry[name+'_indices'])
            counts['updates']+=70*math.ceil(len(entry['train_indices'])/24)
    if counts!=dict(train=18786,validation=4714,test=5875,updates=56000):raise ValueError('registered budget mismatch')
    workers=min(35,96,len(os.sched_getaffinity(0)))
    quota=Path('/sys/fs/cgroup/cpu.max').read_text().split()
    if quota[0]!='max':workers=min(workers,max(1,int(quota[0])//int(quota[1])))
    # 2GiB conservative admission per preparation process, under 160GiB cap.
    workers=min(workers,80)
    t.write(t.OUT/'authorization/prepare_resources.json',dict(workers=workers,threads=1,host_cap=160*1024**3,affinity=len(os.sched_getaffinity(0))))
    with cf.ProcessPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(qualify,task,fold) for task in t.TASKS for fold in range(5)]
        units=[]
        for f in cf.as_completed(futures):units.append(f.result());t.check_clock(True)
    t.seed_all(42);model=m.FreshPT()
    t.save_checkpoint(t.INITIAL,dict(schema=m.PT_SCHEMA,plan=t.PLAN,revision='r2',step=0,seed=42,config=m.CONFIG,historical_tensor_reads=0,state_dict=model.state_dict(),stream=model.stream_state(),fp_prediction=False))
    del model
    for ti,task in enumerate(t.TASKS):
        for fold in range(5):
            seed=2026110400+10*ti+fold;t.seed_all(seed);head=m.native.property_head(.5)
            t.save_checkpoint(t.head_path(task,fold),dict(schema='ua3d-cond-spectral-head-r2',plan=t.PLAN,task=task,fold=fold,seed=seed,state_dict=head.state_dict()))
    t.exclusive(t.OUT/'identity.json',dict(plan=t.PLAN,revision='r2',splits=splits,units=units,counts=counts,pt_metadata=pt,mapping_metadata=mapping,historical_tensor_reads=0,test_targets_read=False))
    t.exclusive(t.OUT/'sources.json',dict(raw_pt=str(t.PT_CACHE),raw_mapping=str(t.MAP_ROOT),surface=str(t.SURFACE),units=units,learned_features=False,historical_tensor_reads=0,metadata_initialization_paths_not_opened=True))
    print('PREPARE COMPLETE',flush=True)
if __name__=='__main__':main()
