#!/usr/bin/env python3
"""Alias qualified raw inputs/statistics; regenerate all model parameters fresh."""
import os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from src.training import ua3d_cond_spectral_r3 as t
from src.modules import ua3d_cond_spectral_r3 as m

def main():
    t.require_tmux();t.check_clock(True)
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU prepare must hide GPU')
    torch.set_num_threads(1)
    identity=t.read(t.PARENT/'identity.json');sources=t.read(t.PARENT/'sources.json');audit=t.read(t.PARENT/'shutdown_audit.json')
    if identity['plan']!=t.PLAN or identity['revision']!='r2' or identity['historical_tensor_reads']!=0 or identity['test_targets_read'] is not False or len(identity['units'])!=35:raise ValueError('parent qualified identity')
    if identity['counts']!=dict(train=18786,validation=4714,test=5875,updates=56000):raise ValueError('parent consumption contract')
    if sources['learned_features'] is not False or sources['historical_tensor_reads']!=0 or audit['status']!='STOPPED' or audit['pt_attempted_updates']!=858 or audit['pt_actual_completed_updates_interval']!=[857,858]:raise ValueError('parent provenance/unknown tail mismatch')
    if sources['raw_pt']!=str(t.PT_CACHE) or sources['raw_mapping']!=str(t.MAP_ROOT) or sources['surface']!=str(t.SURFACE):raise ValueError('parent raw paths')
    aliases=[]
    for task in t.TASKS:
        split=t.read(t.SPLIT/f'{task}.json')
        if split!=identity['splits'][task] or split['validation_is_test'] is not False:raise ValueError('original split identity')
        for fold in range(5):
            t.check_clock(True);unit=next(x for x in identity['units'] if x['task']==task and x['fold']==fold)
            source=next(x for x in sources['units'] if x['task']==task and x['fold']==fold)
            if unit!=source:raise ValueError('qualified source mismatch')
            cfg=t.source_config(task,fold);entry=split['folds'][fold]
            if cfg['input_cache']!=unit['input_cache'] or cfg['scaler_reference']!=unit['scaler'] or len(entry['train_indices'])!=unit['train'] or len(entry['validation_indices'])!=unit['validation'] or not unit['rows_and_keys_verified'] or unit['test_targets_read']:raise ValueError('qualified row/key scope')
            stats_path=t.PARENT/'scalers'/task/f'fold{fold}.json';stats=t.read(stats_path)
            if set(stats)!=set(('mean','std','valid_train','train_only')) or stats['train_only'] is not True or len(stats['mean'])!=16 or len(stats['std'])!=16 or not 0<stats['valid_train']<=unit['train']:raise ValueError('train-only stat schema')
            import math
            if any(not math.isfinite(x) for x in stats['mean']) or any(not math.isfinite(x) or x<=0 for x in stats['std']):raise ValueError('stat values')
            scaler=t.read(cfg['scaler_reference'])
            aliases.append(dict(task=task,fold=fold,qualification_source=str(t.PARENT/'identity.json'),stat_source=str(stats_path),stat_fit_train_only=True,valid_stat_rows=stats['valid_train'],target_scaler_source=cfg['scaler_reference'],target_scaler=scaler,input_cache=cfg['input_cache'],split_source=str(t.SPLIT/f'{task}.json'),new_stat_fit=0))
    t.exclusive(t.OUT/'source_aliases.json',dict(parent=str(t.PARENT),units=aliases,qualification_repeated=False,new_stat_fit=0,model_tensors_loaded=0))
    t.seed_all(42);model=m.FreshPT()
    t.save_checkpoint(t.INITIAL,dict(schema=m.PT_SCHEMA,plan=t.PLAN,revision='r3',science_recipe='r2',step=0,seed=42,config=m.CONFIG,historical_tensor_reads=0,state_dict=model.state_dict(),stream=model.stream_state(),fp_prediction=False))
    del model
    for ti,task in enumerate(t.TASKS):
        for fold in range(5):
            seed=2026110400+10*ti+fold;t.seed_all(seed);head=m.native.property_head(.5)
            t.save_checkpoint(t.head_path(task,fold),dict(schema='ua3d-cond-spectral-head-r3',plan=t.PLAN,revision='r3',science_recipe='r2',task=task,fold=fold,seed=seed,state_dict=head.state_dict()))
    derived=dict(identity,revision='r3',science_recipe='r2',qualification_source=str(t.PARENT/'identity.json'),qualification_repeated=False,new_stat_fit=0)
    t.exclusive(t.OUT/'identity.json',derived);t.exclusive(t.OUT/'sources.json',dict(sources,source_aliases=str(t.OUT/'source_aliases.json'),fresh_initialization=True,parent_model_tensors_loaded=0))
    t.check_clock(True);print('R3 ALIAS PREPARE AND FRESH INITIALIZATION COMPLETE',flush=True)
if __name__=='__main__':main()
