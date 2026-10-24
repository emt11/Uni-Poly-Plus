#!/usr/bin/env python3
"""One CPU job: materialize eligible logical-slot references, never molecular data."""
import os
import sys
from pathlib import Path
import time
import traceback
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.training.ua2d_ptscale import *


def main():
    require_tmux();started=time.monotonic();folder=OUT/'inputs';folder.mkdir(exist_ok=False)
    try:
        reference=read(PT_REFERENCE);cache=read(PT_CACHE/'metadata.json');eligible_meta=read(ELIGIBLE.parent/'metadata.json')
        if (cache.get('schema')!='ua2d_prepared_templates_r4' or cache.get('status')!='COMPLETE'
            or cache.get('train_rows')!=911391 or cache.get('completed')!=911391
            or cache.get('source')!=reference['source'] or cache.get('index')!=reference['index']
            or cache.get('spec')!=reference['fp_spec']):raise ValueError('original PT cache/reference identity mismatch')
        if (eligible_meta.get('status')!='COMPLETE' or eligible_meta.get('valid_rows')!=910756
            or eligible_meta.get('excluded_rows')!=635 or eligible_meta.get('original_train_rows')!=911391
            or eligible_meta.get('parent_cache')!=str(PT_CACHE)
            or eligible_meta.get('policy')!='drop_failed_3d_entire_record'):raise ValueError('eligible identity/policy mismatch')
        eligible=np.load(ELIGIBLE,allow_pickle=False);keys=np.load(PT_CACHE/'keys.npy',mmap_mode='r',allow_pickle=False)
        rows=np.load(PT_CACHE/'source_rows.npy',mmap_mode='r',allow_pickle=False)
        if (eligible.shape!=(910756,) or eligible.dtype.kind not in 'iu' or eligible.min()<0 or eligible.max()>=911391
            or len(np.unique(eligible))!=910756 or keys.shape!=(911391,) or keys.dtype.itemsize!=32 or rows.shape!=(911391,)):
            raise ValueError('eligible logical-slot coverage mismatch')
        cache_metadata(CACHE_ROOT) # Read-only reference; never apply PT filtering to downstream samples.
        np.save(folder/'eligible_logical.npy',eligible,allow_pickle=False)
        np.save(folder/'keys.npy',keys[eligible],allow_pickle=False)
        np.save(folder/'source_rows.npy',rows[eligible],allow_pickle=False)
        write(folder/'pt_reference.json',reference)
        write(folder/'pt_config.json',reference['config'])
        write(folder/'source_metadata.json',dict(pt_cache=cache,eligible=eligible_meta,ft_cache=cache_metadata(CACHE_ROOT)))
        write(folder/'metadata.json',dict(plan=PLAN,revision=REVISION,schema='ua2d-ptscale-logical-map-r1',status='COMPLETE',records=910756,
            original_records=911391,excluded=635,parent_cache=str(PT_CACHE),eligible_source=str(ELIGIBLE),
            eligible_metadata=str(ELIGIBLE.parent/'metadata.json'),mapping='logical=eligible[stream.index_at(global_position)];key=keys[logical];source_row=source_rows[logical]',
            eligibility_semantics='logical_slot_not_source_row',cache_rebuilt=False,online_fallback=False,
            ft_cache=dict(root=str(CACHE_ROOT),plan=CACHE_PLAN,schema=CACHE_SCHEMA,records=5875,filtered=False),
            initialization=str(INITIALIZATION),mask_head=str(INITIALIZATION.parent/'task_heads/V1.pt'),fp_seed=20261019,
            prepare_seconds=time.monotonic()-started,time=now(),pid=os.getpid()))
        print('COMPLETE: 910756 logical slots, original LMDB/FT cache unchanged',flush=True)
    except BaseException:
        write(folder/'failure.json',dict(status='FAILED',error=traceback.format_exc(),time=now()));raise

if __name__=='__main__':main()
