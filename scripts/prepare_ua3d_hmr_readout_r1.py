#!/usr/bin/env python3
"""Read-only source qualification, never builds geometry or fits statistics."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True

def qualify():
    import concurrent.futures as cf
    import math,time
    import numpy as np
    import torch
    from src.training import ua3d_hmr_readout_r1 as t
    from src.dataset import ua3d_hmr_readout_r1 as d
    t.cpu_contract();t.check_clock(True);torch.set_num_threads(1);torch.set_num_interop_threads(1)
    b=t.read(t.P3/'budget.json');prior=t.read(t.P3/'final_review/results.json')
    if prior['status']!='COMPLETE_PENDING_REVIEW' or not prior['execution_complete'] or not prior['final_cpu_pass'] or prior['FINAL']!='S_FIXED':raise ValueError('P3 terminal result')
    for name in ('controller','outer','bootstrap'):
        if t.read(t.P3/f'{name}_exit.json')['exit']!=0:raise ValueError('P3 true exit')
    if t.read(t.P3/'cumulative_budget.json')['cumulative_update_interval']!=[18090,18230]:raise ValueError('historical update interval')
    for name in t.read(t.P3/'authorization/source_lock.json')['files']:
        if (t.ROOT/name).read_bytes()!=(t.P3/'authorization/source_locked'/name).read_bytes():raise ValueError('P3 source changed')
    for name in t.read(t.P2/'authorization/source_lock.json')['files']:
        if (t.ROOT/name).read_bytes()!=(t.P2/'authorization/source_locked'/name).read_bytes():raise ValueError('P2 source changed')
    anchors=t.read(t.P2/'registry/anchors.json');t.exclusive(t.OUT/'registry/anchors.json',anchors);old=t.read(t.P3/'registry/canonical.json')['entries'];entries={}
    for arm in t.ARMS:
        for task in t.TASKS:
            for fold in range(5):
                if arm=='C_FIXED':
                    source=old[f'S_FIXED/{task}/{fold}']
                    if source['source_kind']!='r3_new_ft' or Path(source['folder'])!=t.P3/'finetune/S_FIXED'/task/f'fold{fold}':raise ValueError('C_FIXED old resolver')
                    folder=source['folder'];kind='fixed_r3_science_payload'
                elif arm=='C_ANCHOR':folder=str(t.P3/'formal/C_ANCHOR'/task/f'fold{fold}');kind='base_output_alias'
                else:folder=str(t.OUT/'finetune'/arm/task/f'fold{fold}');kind='readout_new'
                entries[f'{arm}/{task}/{fold}']=dict(arm=arm,task=task,fold=fold,seed=42+fold,source_kind=kind,folder=folder)
    t.exclusive(t.OUT/'registry/canonical.json',dict(entries=entries,reference=t.REFERENCE))
    coverage=t.read(t.P2/'surface/coverage.json');rotation=t.read(t.P2/'surface/rotation.json');status=t.read(t.P2/'surface/status.json')
    if coverage['overall']<.99 or len(status)!=5875 or sum(not x['available'] for x in status)!=17 or len(rotation)!=4 or any(x['area']>.05 or x['volume']>.05 or x['descriptor_relative_L2']>.1 for x in rotation):raise ValueError('fixed geometry/mask/gates')
    totals=dict(train=0,validation=0,test=0,train_batches=0,validation_batches=0,test_batches=0)
    for task in t.TASKS:
        surf=d.surface(task);expected=[x['available'] for x in status if x['task']==task]
        if surf['mask'].bool().tolist()!=expected:raise ValueError('surface mask identity')
        for value in surf.values():
            if not torch.isfinite(value).all():raise FloatingPointError('surface input')
        for fold,split in enumerate(t.read(t.SPLIT/f'{task}.json')['folds']):
            for part in ('train','validation','test'):
                totals[part]+=len(split[part+'_indices']);totals[part+'_batches']+=math.ceil(len(split[part+'_indices'])/256)
    if totals!=dict(train=18786,validation=4714,test=5875,train_batches=85,validation_batches=45,test_batches=45):raise ValueError('split budget matrix')
    def one(pair):
        task,fold=pair;c=d.cache(task,fold);s=d.stats(task,fold);split=t.read(t.SPLIT/f'{task}.json')['folds'][fold];ids=sorted(set(split['train_indices'])|set(split['validation_indices']));mapping={x['row']:x for x in anchors['rows'][task]};q=t.read(t.P2/'registry/cache_qualification/trainval'/task/f'fold{fold}.json')
        if c['indices'].tolist()!=ids or c['keys']!=[mapping[i]['key'] for i in ids] or c['scaler']!=t.anchor(task,fold)['scaler'] or s['train_rows']!=split['train_indices'] or q['status']!='COMPLETE' or q['base']!=t.anchor(task,fold):raise ValueError('cache/scaler identity')
        for name,shape in [('z',(len(ids),768)),('availability',(len(ids),3)),('y',(len(ids),)),('yb',(len(ids),))]:
            if tuple(c[name].shape)!=shape or c[name].dtype!=torch.float32 or not torch.isfinite(c[name]).all():raise ValueError('cache shape/dtype/finite')
        if c['availability'][:,1].bool().tolist()!=[mapping[i]['available'] for i in ids]:raise ValueError('BASE availability')
        for name,n in [('z',768),('stat',16),('spec',128)]:
            for kind in ('mean','scale'):
                value=np.asarray(s[f'{name}_{kind}'])
                if value.shape!=(n,) or not np.isfinite(value).all() or kind=='scale' and np.any(value<=0):raise ValueError('input stats qualification')
        valid=t.partition_positions(c,'validation',task,fold)
        with np.load(t.anchor(task,fold)['validation_predictions'],allow_pickle=False) as original:
            if not np.array_equal(c['indices'][valid].numpy(),original['indices']) or not np.array_equal(np.array([c['keys'][i] for i in valid]),original['sample_keys']) or not np.array_equal(c['y_original'][valid].numpy(),original['y_true']) or not np.allclose(c['yb_original'][valid].numpy(),original['y_pred'],atol=1e-6,rtol=1e-5):raise ValueError('BASE validation provenance')
        qualified=t.qualify_ft('C_FIXED',task,fold)
        for name in ('epoch0','best','last'):
            p=d.fixed_payload(task,fold,name)
            if p['base']!=t.anchor(task,fold) or p['stats']!=s:raise ValueError('C_FIXED stats/scaler/source')
        j=next(j for j in b['jobs'] if j['kind']=='ft' and j['arm']=='S_FIXED' and j['task']==task and j['fold']==fold)
        if not j['qualified'] or j['exit']!=0 or t.read(Path(j['log'])/'wait_exit.json')['exit']!=0 or t.read(Path(j['log'])/'end.json')['exit']!=0:raise ValueError('C_FIXED completed wait')
        return dict(task=task,fold=fold,trainval=str(d.cache_path(task,fold,False)),scaler=str(t.P2/'scalers'/task/f'fold{fold}.json'),fixed=qualified,TEST_status='NOT_READ',TEST_cache_path=str(t.P3/'feature_cache/test'/task/f'fold{fold}.pt'),TEST_exists=(t.P3/'feature_cache/test'/task/f'fold{fold}.pt').exists())
    resources=t.read(t.OUT/'authorization/resources.json');workers=min(35,resources['qualification_workers'])
    with cf.ThreadPoolExecutor(max_workers=workers) as pool:qualified=list(pool.map(one,[(x,f) for x in t.TASKS for f in range(5)]))
    t.exclusive(t.OUT/'registry/sources.json',dict(status='COMPLETE',units=qualified,coverage=coverage,rotation=rotation,mask_unavailable=17,surface=str(t.P2/'surface'),P2=str(t.P2),P3=str(t.P3),historical_cumulative=str(t.P3/'cumulative_budget.json'),new_geometry=0,new_spectrum=0,new_BASE_forward=0,new_fit=0,new_pack=0,models_constructed=0,qualified_unix=time.time()))
    u=t.Usage(t.OUT/'qualify');u.add('qualification_units',35);u.finish()
if __name__=='__main__':qualify()
