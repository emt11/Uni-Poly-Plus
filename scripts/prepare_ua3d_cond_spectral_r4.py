#!/usr/bin/env python3
"""Metadata-only adoption of r3 complete PT and original random heads; no models."""
import os,sys,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.training import ua3d_cond_spectral_r4 as t

def main():
    t.require_tmux();t.check_clock(True)
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU metadata preparation must hide GPU')
    # No tensor checkpoint load, model construction, graph replay or statistics fit.
    ident=t.read(t.P3/'identity.json');aliases=t.read(t.P3/'source_aliases.json');pt=t.read(t.P3/'pretrain/BASE_FRESH/consumption.json');jobs=t.read(t.P3/'jobs.json');cum=t.read(t.P3/'cumulative_budget.json');budget=t.read(t.P3/'budget.json')
    if ident.get('plan')!=t.PLAN or ident.get('revision')!='r3' or ident.get('science_recipe')!='r2' or ident.get('test_targets_read') is not False or len(ident['units'])!=35:raise ValueError('qualified r3 identity metadata')
    if ident['counts']!=dict(train=18786,validation=4714,test=5875,updates=56000):raise ValueError('fixed split consumption differs')
    if pt['status']!='COMPLETE' or pt['attempted_updates']!=10000 or pt['completed_updates']!=10000 or pt['pt_microbatches']!=60000 or pt['pt_presentations']!=10080000 or pt['stream']['counters']!=[60000]*3 or pt['historical_tensor_reads']!=0:raise ValueError('complete fresh PT source metadata')
    if len([j for j in jobs if j['kind']=='pt' and j['exit']==0])!=1 or any(j['exit'] is None for j in jobs):raise ValueError('parent actual exits not terminal/complete PT')
    if cum['total_attempts']!=11155 or cum['actual_completed_interval']!=[11154,11155] or budget['ft_updates']!=297 or budget['ft_complete']!=0:raise ValueError('parent retained cumulative contract')
    if not t.INPUT_PT.is_file():raise FileNotFoundError(t.INPUT_PT)
    records=[]
    for task in t.TASKS:
        split=t.read(t.SPLIT/f'{task}.json')
        if split!=ident['splits'][task] or split['validation_is_test'] is not False:raise ValueError('split changed')
        for fold in range(5):
            t.check_clock(True);unit=next(x for x in ident['units'] if x['task']==task and x['fold']==fold);a=next(x for x in aliases['units'] if x['task']==task and x['fold']==fold)
            cfg=t.source_config(task,fold);stats_path=t.PARENT/'scalers'/task/f'fold{fold}.json';stats=t.read(stats_path)
            if not a['stat_fit_train_only'] or a['new_stat_fit']!=0 or a['input_cache']!=cfg['input_cache'] or unit['scaler']!=cfg['scaler_reference'] or a['stat_source']!=str(stats_path):raise ValueError('alias provenance/fit scope')
            if stats.get('train_only') is not True or len(stats['mean'])!=16 or len(stats['std'])!=16 or not 0<stats['valid_train']<=unit['train'] or any(not math.isfinite(x) for x in stats['mean']) or any(not math.isfinite(x) or x<=0 for x in stats['std']):raise ValueError('train-only statistics schema')
            if unit['train']!=len(split['folds'][fold]['train_indices']) or unit['validation']!=len(split['folds'][fold]['validation_indices']) or unit['test_targets_read'] or not unit['rows_and_keys_verified']:raise ValueError('row/key qualification scope')
            head=t.head_path(task,fold)
            if not head.is_file():raise FileNotFoundError(head)
            records.append(dict(task=task,fold=fold,input_pt=str(t.INPUT_PT),input_pt_revision='r3',input_pt_schema='ua3d-cond-spectral-pretrain-r3',random_head=str(head),random_head_schema='ua3d-cond-spectral-head-r3',head_seed=2026110400+10*t.TASKS.index(task)+fold,qualification_source=str(t.P3/'identity.json'),stat_source=str(stats_path),target_scaler_source=cfg['scaler_reference'],input_cache=cfg['input_cache'],new_stat_fit=0,from_common_start=True,interrupted_best_allowed=False))
    t.exclusive(t.OUT/'source_aliases.json',dict(plan=t.PLAN,execution_revision='r4',science_recipe='r2',available_source_pt_completed=10000,input_pt=str(t.INPUT_PT),units=records,new_PT=0,new_initialization=0,new_stat_fit=0,model_tensors_loaded_during_prepare=0,prior_model_reads_must_be_counted_at_normal_initialization=True))
    t.exclusive(t.OUT/'identity.json',dict(ident,revision='r4',execution_revision='r4',input_pt_revision='r3',qualification_source=str(t.P3/'identity.json')))
    t.exclusive(t.OUT/'sources.json',dict(input_pt=str(t.INPUT_PT),random_heads_root=str(t.P3/'initialization/heads'),surface=str(t.SURFACE),aliases=str(t.OUT/'source_aliases.json'),forbidden='historical BASE tensors; interrupted FT BEST/optimizer/RNG; r2 partial PT',test_targets_read=False))
    t.check_clock(True);print('R4 METADATA ALIAS COMPLETE; NO MODEL/TENSOR INITIALIZATION',flush=True)
if __name__=='__main__':main()
