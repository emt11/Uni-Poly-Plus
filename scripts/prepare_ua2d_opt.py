#!/usr/bin/env python3
"""The single bounded preparation for UA2D-OPT r1; no geometry generation."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
import sys,time,traceback,concurrent.futures,multiprocessing as mp
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
_SOURCE=_OLD=None

def init_worker():
    import lmdb,torch
    from src.training.ua2d_opt import CACHE
    global _OLD
    torch.set_num_threads(1);_SOURCE._open_environment()
    _OLD=lmdb.open(str(CACHE/'templates.lmdb'),readonly=True,lock=False,readahead=False,max_readers=128)

def chunk(rows):
    from src.training.glt_dual_runtime import pretrain_source_fields
    from src.dataset.ua2d_opt import extend,context_signatures
    from src.dataset.ua2d_fast_pretrain import encode,decode_template
    import torch
    result=[]
    for logical,row,key in rows:
        top,tri,text,static,_,identity=pretrain_source_fields(_SOURCE,row,include_identity=True)
        if _SOURCE.samples[row][0]!=key or _SOURCE.entries[row]['pretrain_role']!='train':raise ValueError('logical/source row/key/train role mismatch')
        with _OLD.begin() as txn:raw=txn.get(key)
        if raw is None:raise ValueError('missing original 3D template')
        template=decode_template(raw)
        if (template['sample_key']!=key or not bool(template['clean'].geometry_valid)
            or not torch.isfinite(template['positions']).all()
            or not torch.equal(template['clean'].mips_x,top.mips_x)
            or template['identity']!=str(top.normalized_canonical_smiles)):raise ValueError('eligible template identity/geometry mismatch')
        data=extend(top,text,static)
        signatures=context_signatures(data)
        result.append((logical,row,key,encode(dict(clean=data,context=signatures,sample_key=key)),signatures,len(data.lga_spd)))
    return result

def main():
    import lmdb,numpy as np,torch
    from src.dataset.mcl_ph_direct import DirectMCLSource
    from src.training.ua2d_opt import OUT,CACHE,ELIGIBLE,SOURCE,DOWN,TASKS,SPLIT,read,write,now
    from src.training.glt_dual_runtime import require_tmux,pretrain_source_fields
    from src.dataset.ua2d_opt import extend,SCHEMA,CACHE_CONTRACT,numeric_context_record
    from src.dataset.ua2d_fast_pretrain import encode,decode_template
    require_tmux();torch.set_num_threads(1)
    global _SOURCE
    start=time.monotonic();folder=OUT/'inputs';folder.mkdir(exist_ok=False)
    (folder/'hop3').mkdir();(folder/'context').mkdir()
    state=dict(schema=SCHEMA,status='RUNNING',started=now(),prepared_pt_rows=0,prepared_downstream_records=0,elapsed_seconds=0.)
    def record():
        state['elapsed_seconds']=time.monotonic()-start;write(folder/'coverage.json',state)
        if state['elapsed_seconds']>14400:raise TimeoutError('prepare hard deadline')
    record();env=None;down_env=None
    try:
        meta=read(ELIGIBLE.parent/'metadata.json');cache=read(CACHE/'metadata.json')
        if meta.get('valid_rows')!=910756 or meta.get('excluded_rows')!=635 or meta.get('policy')!='drop_failed_3d_entire_record' or meta.get('status')!='COMPLETE':raise ValueError('eligible metadata mismatch')
        if cache.get('status')!='COMPLETE' or cache.get('train_rows')!=911391:raise ValueError('old cache metadata mismatch')
        eligible=np.load(ELIGIBLE,allow_pickle=False);rows=np.load(CACHE/'source_rows.npy',allow_pickle=False);keys=np.load(CACHE/'keys.npy',allow_pickle=False)
        if eligible.shape!=(910756,) or rows.shape!=(911391,) or keys.shape!=(911391,) or len(set(eligible.tolist()))!=910756 or np.any(eligible<0) or np.any(eligible>=911391):raise ValueError('logical mapping shape/range mismatch')
        write(folder/'sources.json',dict(source=str(SOURCE),eligible=str(ELIGIBLE),cache=str(CACHE),rows=str(CACHE/'source_rows.npy'),keys=str(CACHE/'keys.npy'),policy=meta['policy'],downstream=str(DOWN),split=str(SPLIT),workers=16,geometry_generated=0,fp_generated=0,hash_validation=False))
        _SOURCE=DirectMCLSource(SOURCE,task=None,selected_indices=None,split_manifest=None,verify_key_digest=False)
        _SOURCE.close()
        env=lmdb.open(str(folder/'hop3/pretrain.lmdb'),map_size=1024**4,max_readers=128)
        vocab={};unique=set();relations=0
        # map preserves deterministic logical order, independent of completion order.
        context=mp.get_context('fork')
        with concurrent.futures.ProcessPoolExecutor(max_workers=16,mp_context=context,initializer=init_worker) as pool:
            iterator=iter(eligible.tolist());pending=[]
            def submit():
                batch=[]
                for _ in range(32):
                    logical=next(iterator,None)
                    if logical is None:break
                    batch.append((logical,int(rows[logical]),bytes(keys[logical])))
                if batch:pending.append(pool.submit(chunk,batch))
            for _ in range(32):submit()
            while pending:
                future=pending.pop(0);values=future.result()
                with env.begin(write=True) as txn:
                    for logical,row,key,payload,signatures,nrel in values:
                        numeric_record=numeric_context_record(decode_template(payload),signatures,vocab)
                        if key not in unique:
                            if not txn.put(key,encode(numeric_record),overwrite=False):raise ValueError('PT cache writer collision')
                            unique.add(key)
                        state['prepared_pt_rows']+=1;relations+=nrel
                state['relations']=relations;state['unique_templates']=len(unique);record();submit()
                if state['prepared_pt_rows']%4096==0:print(state,flush=True)
        write(folder/'context/vocab.json',dict(schema=SCHEMA,mapping=vocab,size=len(vocab),source='910756_PT_train_only',order='eligible_logical_order_then_atom_order',signature='element|neighbor_element:bond_type:count sorted tuples; distinct physical images preserved',downstream_labels_read=False))
        np.save(folder/'logical_indices.npy',eligible,allow_pickle=False)
        down_env=lmdb.open(str(folder/'hop3/downstream.lmdb'),map_size=64*1024**3,max_readers=128)
        seen=set();down_rows={}
        for task in TASKS:
            manifest=read(SPLIT/f'{task}.json')
            if manifest.get('validation_is_test') is not False:raise ValueError('split overlap flag invalid')
            ds=DirectMCLSource(DOWN,task=task,selected_indices=None,split_manifest=manifest,verify_key_digest=False)
            try:
                down_rows[task]=[]
                for i in range(len(ds)):
                    top,_,text,static,_=pretrain_source_fields(ds,i)
                    key=ds.samples[i][0];data=extend(top,text,static)
                    if key not in seen:
                        with down_env.begin(write=True) as txn:
                            if not txn.put(key,encode(dict(clean=data,sample_key=key)),overwrite=False):raise ValueError('downstream writer collision')
                        seen.add(key)
                    down_rows[task].append(dict(row=int(ds.entries[i]['original_row']),key=key.hex()))
                    state['prepared_downstream_records']+=1
                record()
            finally:ds.close()
        if state['prepared_pt_rows']!=910756 or state['prepared_downstream_records']!=5875:raise ValueError('input preparation coverage mismatch')
        write(folder/'downstream_rows.json',down_rows)
        env.sync();down_env.sync();state.update(status='COMPLETE',vocab_size=len(vocab),ended=now());record()
        # Publish only after all deterministic fields and integer labels are durable.
        write(folder/'cache_manifest.json',dict(schema=SCHEMA,contract=CACHE_CONTRACT,status='COMPLETE',
            prepared_pt_rows=state['prepared_pt_rows'],prepared_downstream_records=state['prepared_downstream_records'],
            vocab_size=len(vocab),context_encoding='int64_per_atom',hop_relation_views=[1,2,3],
            source=str(SOURCE),eligible=str(ELIGIBLE),preprocessing_seconds=state['elapsed_seconds'],
            cache_bytes=sum(p.stat().st_size for p in (folder/'hop3').glob('*/data.mdb')),
            readonly_training=True,online_fallback=False))
    except BaseException:
        state.update(status='FAILED',error=traceback.format_exc(),ended=now());write(folder/'coverage.json',state);raise
    finally:
        if env is not None:env.close()
        if down_env is not None:down_env.close()
        if _SOURCE is not None:_SOURCE.close()

if __name__=='__main__':main()
