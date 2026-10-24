#!/usr/bin/env python3
"""User-authorized one-pass exclusion of failed 3D cached records; no new samples."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import concurrent.futures
import json
import multiprocessing as mp
import os
import pickle
import time
import traceback
_ENV=_KEYS=None

def initializer(cache):
    import lmdb,numpy as np
    global _ENV,_KEYS
    _ENV=lmdb.open(str(Path(cache)/'templates.lmdb'),readonly=True,lock=False,readahead=True,max_readers=128)
    _KEYS=np.load(Path(cache)/'keys.npy',mmap_mode='r',allow_pickle=False)

def chunk(indices):
    output=[]
    with _ENV.begin() as txn:
        for i in indices:
            key=_KEYS[i].tobytes();payload=txn.get(key)
            if payload is None: raise ValueError('filter missing bound cache record')
            row=pickle.loads(payload)
            if row['sample_key']!=key: raise ValueError('filter cache identity mismatch')
            clean=row['clean'];valid=bool(clean['geometry_valid'])
            output.append((i,valid,str(clean['geometry_invalid_reason'])))
    return output

def main():
    import numpy as np
    from src.training.ua2d_et_single import OUT,CACHE,write
    from src.training.glt_dual_runtime import require_tmux
    require_tmux();folder=OUT/'sampling_repair1';folder.mkdir(exist_ok=False)
    started=time.monotonic();meta=dict(status='RUNNING',policy='drop_failed_3d_entire_record',
        parent_cache=str(CACHE),original_train_rows=911391,completed=0,valid_rows=0,excluded_rows=0,
        reasons={},pid=os.getpid(),elapsed_seconds=0.)
    write(folder/'metadata.json',meta)
    mask=np.zeros(911391,dtype=bool);seen=np.zeros(911391,dtype=bool)
    keys=np.load(CACHE/'keys.npy',mmap_mode='r',allow_pickle=False)
    rows=np.load(CACHE/'source_rows.npy',mmap_mode='r',allow_pickle=False)
    try:
        with (folder/'excluded.jsonl').open('x') as excluded:
            context=mp.get_context('spawn')
            with concurrent.futures.ProcessPoolExecutor(max_workers=16,mp_context=context,initializer=initializer,initargs=(str(CACHE),)) as pool:
                iterator=iter([list(range(i,min(i+256,911391))) for i in range(0,911391,256)])
                pending=set()
                def submit():
                    entry=next(iterator,None)
                    if entry is not None: pending.add(pool.submit(chunk,entry))
                for _ in range(32):submit()
                while pending:
                    done,pending=concurrent.futures.wait(pending,return_when=concurrent.futures.FIRST_COMPLETED)
                    for future in done:
                        for i,valid,reason in future.result():
                            if seen[i]:raise ValueError('duplicate eligibility writer')
                            seen[i]=True;mask[i]=valid
                            if not valid:
                                meta['excluded_rows']+=1;meta['reasons'][reason]=meta['reasons'].get(reason,0)+1
                                excluded.write(json.dumps(dict(logical_index=i,source_row=int(rows[i]),sample_key=keys[i].tobytes().hex(),reason=reason))+'\n')
                            else: meta['valid_rows']+=1
                            meta['completed']+=1
                        submit()
                    excluded.flush();meta['elapsed_seconds']=time.monotonic()-started;write(folder/'metadata.json',meta)
                    if meta['completed']%16384<256*len(done):print(json.dumps(meta),flush=True)
        if not seen.all() or not mask.any():raise ValueError('eligibility coverage incomplete/empty')
        np.save(folder/'eligible_indices.npy',np.flatnonzero(mask),allow_pickle=False)
        np.save(folder/'excluded_indices.npy',np.flatnonzero(~mask),allow_pickle=False)
        meta.update(status='COMPLETE',elapsed_seconds=time.monotonic()-started)
        write(folder/'metadata.json',meta)
    except BaseException:
        meta.update(status='FAILED',error=traceback.format_exc(),elapsed_seconds=time.monotonic()-started)
        write(folder/'metadata.json',meta);raise

if __name__=='__main__':main()
