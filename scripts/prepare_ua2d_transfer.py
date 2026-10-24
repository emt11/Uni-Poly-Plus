#!/usr/bin/env python3
"""The one CPU preparation: original 2-hop inputs, plus common residual init."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ.setdefault(k,'2')
from pathlib import Path
import sys
import time
import traceback
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def main():
    import lmdb
    import torch
    from src.dataset.mcl_ph_direct import DirectMCLSource
    from src.dataset.glt_dual import build_glt_2d_sample
    from src.dataset.ua2d_fast_pretrain import encode
    from src.modules.ua2d_transfer import branch,BRANCH_SEED,SCHEMA
    from src.training.glt_dual_runtime import pretrain_source_fields,require_tmux,save_checkpoint
    from src.training.ua2d_transfer import OUT,DOWN,SPLIT,TASKS,PLAN,CACHE_SCHEMA,now,read,write
    require_tmux();torch.set_num_threads(2)
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('preparation is CPU only')
    started=time.monotonic();folder=OUT/'inputs';folder.mkdir(exist_ok=False)
    state=dict(status='RUNNING',plan=PLAN,schema=CACHE_SCHEMA,records=0,initialization_starts=0,started=now())
    def record():
        state['elapsed_seconds']=time.monotonic()-started;write(folder/'coverage.json',state)
        if state['elapsed_seconds']>3600:raise TimeoutError('one preparation deadline exhausted')
    record();env=None
    try:
        init=OUT/'initialization';init.mkdir(exist_ok=False)
        state['initialization_starts']=1;record()
        phi=branch();save_checkpoint(init/'residual.pt',dict(plan=PLAN,schema=SCHEMA,seed=BRANCH_SEED,state_dict=phi.state_dict(),zero_last=True,training_rng_consumed=False))
        write(init/'config.json',dict(plan=PLAN,seed=BRANCH_SEED,shared_arms=['R_MEAN','R_STD','R_STD_SP'],architecture='LN512-Linear128-GELU-Linear512',last_weight_bias=0,initialization_starts=1))
        env=lmdb.open(str(folder/'graphs.lmdb'),map_size=64*1024**3,max_readers=128)
        rows={};split_records={}
        for task in TASKS:
            split=read(SPLIT/f'{task}.json');split_records[task]=split
            if split.get('validation_is_test') is not False:raise ValueError('invalid original split separation')
            source=DirectMCLSource(DOWN,task=task,selected_indices=None,split_manifest=split,verify_key_digest=False)
            rows[task]=[];seen=set()
            try:
                for i in range(len(source)):
                    topology,_,text,_,_=pretrain_source_fields(source,i)
                    # Build from the original max-hop2 function. Neither hop3
                    # caches nor a short static-cache inventory are inputs.
                    graph=build_glt_2d_sample(topology,text)
                    row=int(source.entries[i]['original_row']);key=source.samples[i][0].hex();cache_key=f'{task}:{row}'
                    if row in seen:raise ValueError('duplicate original task row')
                    if graph.lga_path_index.shape[1]!=3 or graph.bond_path_features.shape!=(graph.lga_spd.numel(),2,14) or (graph.lga_spd>2).any():raise ValueError('source graph not original 2-hop')
                    payload=encode(dict(schema=CACHE_SCHEMA,task=task,row=row,sample_key=key,clean=graph))
                    with env.begin(write=True) as txn:
                        if not txn.put(cache_key.encode(),payload,overwrite=False):raise ValueError('cache writer collision')
                    seen.add(row);rows[task].append(dict(row=row,key=key,cache_key=cache_key));state['records']+=1
                    if state['records']%128==0:record();print(state,flush=True)
                for fold in split['folds']:
                    if not all(set(fold[k]).issubset(seen) for k in ('train_indices','validation_indices','test_indices')):raise ValueError('cache does not cover original split')
                record()
            finally:source.close()
        if state['records']!=5875:raise ValueError('expected exactly 5875 downstream records')
        write(folder/'rows.json',rows);write(folder/'original_splits.json',split_records)
        env.sync();env.close();env=None
        state.update(status='COMPLETE',ended=now());record()
        write(folder/'metadata.json',dict(plan=PLAN,schema=CACHE_SCHEMA,status='COMPLETE',records=5875,
            records_per_task={k:len(v) for k,v in rows.items()},source=str(DOWN),split=str(SPLIT),max_hop=2,
            builder='src.dataset.glt_dual.build_glt_2d_sample(topology,smiles); no hop3/static-cache substitution',
            original_tensor_fields=True,targets_cached=False,target_fitting=False,hidden_states_cached=False,
            initialization_starts=1,preparation_seconds=state['elapsed_seconds'],readonly_training=True,online_fallback=False,
            hash_validation=False,geometry_generation_calls=0,fp_generation_calls=0))
    except BaseException:
        state.update(status='FAILED',error=traceback.format_exc(),ended=now());write(folder/'coverage.json',state);raise
    finally:
        if env is not None:env.close()

if __name__=='__main__':main()
