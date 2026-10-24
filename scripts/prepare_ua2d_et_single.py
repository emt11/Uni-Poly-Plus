#!/usr/bin/env python3
"""One bounded extraction of existing atomic identity fields; no geometry generation."""
import concurrent.futures
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time
import traceback
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
_SOURCE=None

def worker_init():
    import torch
    torch.set_num_threads(1);_SOURCE._open_environment()

def chunk(entries):
    from src.training.glt_dual_runtime import pretrain_source_fields
    from src.dataset.ua2d_et_single import atomic_fields
    from src.dataset.ua2d_fast_pretrain import encode
    output=[]
    for row,key in entries:
        top,tri,text,static,_=pretrain_source_fields(_SOURCE,row)
        if _SOURCE.samples[row][0]!=key or _SOURCE.entries[row]['pretrain_role']!='train':
            raise ValueError('atomic preparation source identity/role mismatch')
        fields=atomic_fields(top,tri,row,key)
        output.append((key,encode(fields),len(fields['z']),int(fields['center'].sum())))
    return output

def main():
    import argparse,json,lmdb,numpy as np,torch
    from src.dataset.mcl_ph_direct import DirectMCLSource
    from src.dataset.ua2d_et_single import SCHEMA
    from src.training.glt_dual_runtime import require_tmux,load_sample_index_artifact
    from src.training.ua2d_et_single import CACHE,SOURCE,INDEX,OUT,write
    require_tmux();torch.set_num_threads(1)
    output=OUT/'atomic_fields';output.mkdir(exist_ok=False)
    started=time.monotonic();environment=None
    meta=dict(schema=SCHEMA,status='RUNNING',source=str(SOURCE),index=str(INDEX),old_cache=str(CACHE),
        train_rows=911391,completed=0,attempted_rows=0,heavy_atoms=0,central_heavy_atoms=0,
        pid=os.getpid(),elapsed_seconds=0,deadline_seconds=7200)
    write(output/'metadata.json',meta)
    global _SOURCE
    try:
        old=json.loads((CACHE/'metadata.json').read_text())
        if old['status']!='COMPLETE' or old['train_rows']!=911391 or old['source']!=str(SOURCE) or old['index']!=str(INDEX):
            raise ValueError('fixed old cache source/index mismatch')
        rows=np.load(CACHE/'source_rows.npy',mmap_mode='r',allow_pickle=False)
        keys=np.load(CACHE/'keys.npy',mmap_mode='r',allow_pickle=False)
        expected=load_sample_index_artifact(INDEX,'train')['indices']
        if len(rows)!=911391 or len(keys)!=911391 or not np.array_equal(rows,expected):
            raise ValueError('fixed train row coverage/index mismatch')
        _SOURCE=DirectMCLSource(SOURCE,task=None,selected_indices=None,split_manifest=None,
            verify_key_digest=False,readahead=True)
        entries=[(int(r),k.tobytes()) for r,k in zip(rows,keys)]
        if len({key for _,key in entries})!=911391: raise ValueError('nonunique atomic row identity')
        _SOURCE.close()
        environment=lmdb.open(str(output/'fields.lmdb'),map_size=512*1024**3,max_readers=128)
        context=mp.get_context('fork')
        with concurrent.futures.ProcessPoolExecutor(max_workers=16,mp_context=context,initializer=worker_init) as pool:
            iterator=iter([entries[i:i+64] for i in range(0,len(entries),64)])
            pending=set()
            def submit():
                items=next(iterator,None)
                if items is not None:
                    meta['attempted_rows']+=len(items);pending.add(pool.submit(chunk,items))
            for _ in range(32): submit()
            while pending:
                if time.monotonic()-started>7200: raise TimeoutError('atomic preparation deadline')
                done,pending=concurrent.futures.wait(pending,return_when=concurrent.futures.FIRST_COMPLETED)
                for future in done:
                    values=future.result()
                    with environment.begin(write=True) as txn:
                        for key,payload,n,c in values:
                            if not txn.put(key,payload,overwrite=False): raise ValueError('atomic field writer conflict')
                            meta['heavy_atoms']+=n;meta['central_heavy_atoms']+=c
                    meta['completed']+=len(values);submit()
                meta['elapsed_seconds']=time.monotonic()-started;write(output/'metadata.json',meta)
                if meta['completed']%4096<64*len(done): print(json.dumps(meta),flush=True)
        if meta['completed']!=911391: raise ValueError('incomplete atomic coverage')
        environment.sync();meta.update(status='COMPLETE',elapsed_seconds=time.monotonic()-started)
        write(output/'metadata.json',meta)
    except BaseException:
        meta.update(status='FAILED',error=traceback.format_exc(),elapsed_seconds=time.monotonic()-started)
        write(output/'metadata.json',meta);raise
    finally:
        if environment is not None: environment.close()
        if _SOURCE is not None: _SOURCE.close()

if __name__=='__main__': main()
