#!/usr/bin/env python3
"""One read-only qualification identity. Never generates teacher features."""
import argparse
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def qualify(task,fold,formal):
    import os
    import numpy as np
    import torch
    from src.training import ua3d_anchor_lr_r2 as t
    from src.dataset.ua3d_anchor_lr_r2 import SCHEMA
    t.require_tmux();t.guard(preparation=not formal)
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('qualification is CPU-only')
    if formal and not t.read(t.OUT/'selection/final_frozen.json')['frozen']:raise ValueError('TEST cache access before new freeze')
    kind='cache_test' if formal else 'cache_trainval';folder=t.job_path(kind,task=task,fold=fold);folder.mkdir(parents=True,exist_ok=False)
    u=t.Consumption(folder,kind);u.add(kind+'_units');u.record()
    try:
        source=t.PARENT/'feature_cache'/('test' if formal else 'trainval')/task/f'fold{fold}'
        meta=t.read(source/'metadata.json');prior=t.read(source/'consumption.json');a=t.anchor(task,fold);entry=t.read(t.SPLIT/f'{task}.json')['folds'][fold]
        if meta['schema']!=SCHEMA or meta['status']!='COMPLETE' or prior['status']!='COMPLETE' or meta['task']!=task or meta['fold']!=fold or meta['formal']!=formal:raise ValueError('parent cache schema/status/identity')
        fields={'source_best':'best','source_schema':'schema','source_config':'source_config','source_best_epoch':'best_epoch','source_best_stage':'best_stage','scaler':'scaler'}
        if any(meta[k]!=a[v] for k,v in fields.items()):raise ValueError('teacher source/scaler identity')
        if Path(a['best']).stat().st_mtime>prior['started_unix']:raise ValueError('original BASE newer than parent teacher cache')
        rows=t.read(t.INPUT/'rows.json')[task];mapping={r['row']:r for r in rows}
        parts=['test'] if formal else ['train','validation'];seen=[];reused=0
        for part in parts:
            expected=sorted(entry['test_indices']) if formal else entry[part+'_indices']
            if meta['partitions'][part]!=expected:raise ValueError('parent cache split/order')
            with np.load(source/f'{part}_predictions.npz',allow_pickle=False) as npz:
                if npz['indices'].tolist()!=expected or npz['sample_keys'].tolist()!=[mapping[r]['key'] for r in expected]:raise ValueError('cache array row/key identity')
                if not np.isfinite(npz['y_true']).all() or not np.isfinite(npz['y_pred']).all():raise FloatingPointError('cache arrays')
                ys=npz['y_true'].copy();ps=npz['y_pred'].copy()
            for position,row in enumerate(expected):
                p=source/f'row{row}.pt';record=torch.load(p,map_location='cpu',weights_only=False);reused+=p.stat().st_size
                if any(record[k]!=v for k,v in dict(schema=SCHEMA,task=task,fold=fold,row=row,key=mapping[row]['key'],partition=part,source_best=a['best']).items()):raise ValueError('cached row identity')
                tensors=[*record['z'],record['atoms'],record['y'],record['y_base']]+([record['bonds']] if record['bonds'] is not None else [])
                if any(x.dtype!=torch.float32 or x.requires_grad or x.is_inference() or not torch.isfinite(x).all() for x in tensors):raise ValueError('cache ordinary detached finite FP32')
                if bool(record['availability'][1])!=bool(mapping[row]['available']) or record['availability'].tolist()!=[True,bool(mapping[row]['available']),True] or bool(record['availability'][1])!=(record['bonds'] is not None) or (not bool(record['availability'][1]) and torch.count_nonzero(record['z'][1])):raise ValueError('cached missing geometry mask')
                recorded=next(r for r in meta['rows'] if r['row']==row)
                if recorded['key']!=mapping[row]['key'] or recorded['available']!=bool(record['availability'][1]):raise ValueError('cache metadata missing mask/key')
                raw=float(t.transform(record['y'].numpy(),a['scaler'],True))
                if raw!=ys[position] or record['y_base_original']!=ps[position]:raise ValueError('cache prediction/target array and row disagree')
                u.add(kind+'_rows');u.add(kind+'_rows',completed=True);seen.append(row)
            u.record()
        if len(seen)!=len(set(seen)):raise ValueError('duplicate cached rows')
        if not formal:
            gate=t.read(source/'validation_gate.json')
            if not gate['identity_equal'] or not gate['prediction_allclose'] or abs(gate['r2_delta'])>1e-6 or gate['scaler']!=a['scaler']:raise ValueError('parent validation gate')
            with np.load(source/'validation_predictions.npz',allow_pickle=False) as cache,np.load(a['validation_predictions'],allow_pickle=False) as original:
                if any(not np.array_equal(cache[k],original[k]) for k in ['indices','sample_keys','y_true']) or not np.allclose(cache['y_pred'],original['y_pred'],atol=1e-6,rtol=1e-5):raise ValueError('original BASE validation consistency')
                met=t.metrics(cache['y_true'],cache['y_pred'])
                if abs(met['r2']-t.metrics(original['y_true'],original['y_pred'])['r2'])>1e-6:raise ValueError('BASE validation R2')
            t.write(folder/'validation_gate.json',gate)
        value=dict(status='COMPLETE',source=str(source),schema=SCHEMA,task=task,fold=fold,formal=formal,scaler=a['scaler'],source_best=a['best'],read_only=True,reused_bytes=reused,rows=len(seen),new_encoder=0,new_features=0,device='CPU',time=t.now())
        t.exclusive(t.OUT/'registry/cache_aliases'/('test' if formal else 'trainval')/task/f'fold{fold}.json',value)
        t.write(folder/'metadata.json',value);u.value['reused_bytes']=reused;u.add(kind+'_units',completed=True);u.finish()
    except BaseException:
        u.fail();raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--task',required=True);p.add_argument('--fold',type=int,required=True);p.add_argument('--formal',action='store_true');a=p.parse_args();qualify(a.task,a.fold,a.formal)
