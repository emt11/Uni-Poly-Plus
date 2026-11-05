#!/usr/bin/env python3
"""Metadata-only adoption of r3 complete PT and original random heads; no models."""
import os,sys,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.training import ua3d_cond_spectral_r6 as t

def qualify_parent_aliases():
    """Read existing metadata only; no checkpoint or prediction array loads."""
    registry=t.read(t.P5/'registry/canonical.json')['units'];jobs5=t.read(t.P5/'jobs.json');jobs4=t.read(t.P4/'jobs.json')
    cumulative=t.read(t.P5/'cumulative_budget.json')
    if cumulative['total_attempts']!=339281 or cumulative['logged_completed']!=339280 or cumulative['actual_completed_interval']!=[339280,339281]:raise ValueError('retained r5 cumulative source changed')
    if any(j['exit'] is None for j in jobs5+jobs4):raise ValueError('parent jobs not terminal')
    qualified={}
    for arm in t.ARMS:
        for task in t.TASKS:
            split=t.read(t.SPLIT/f'{task}.json')
            for fold in range(5):
                if not t.is_parent_alias(arm,task,fold):continue
                key=f'{arm}/{task}/{fold}';root=t.unit_path(arm,task,fold);old=registry[key]
                if old['root']!=str(root):raise ValueError('fixed parent alias root mismatch')
                cfg=t.read(root/'config.json');met=t.read(root/'metrics.json');cons=t.read(root/'consumption.json');history=t.read(root/'history.json');scaler=t.read(root/'scaler.json')
                rev='r4' if root.is_relative_to(t.P4) else 'r5';schema='ua3d-cond-spectral-best-'+rev
                found=[j for j in (jobs4 if rev=='r4' else jobs5) if (j['kind'],j.get('arm'),j.get('task'),j.get('fold'))==('ft',arm,task,fold)]
                if len(found)!=1 or found[0]['exit']!=0:raise ValueError('parent completed true exit mismatch: '+key)
                entry=split['folds'][fold];updates=70*math.ceil(len(entry['train_indices'])/24)
                expected=dict(status='COMPLETE',epochs_started=70,completed_epochs=70,attempted_updates=updates,completed_updates=updates,train_graphs=70*len(entry['train_indices']),validation_graphs=70*len(entry['validation_indices']),validation_started=70,test_started=0,test_graphs=0,epoch_open=False)
                if any(cons.get(k)!=v for k,v in expected.items()):raise ValueError('parent complete consumption mismatch: '+key)
                if len(history)!=70 or [x['epoch'] for x in history]!=list(range(1,71)) or [x['stage'] for x in history]!=['head']*10+['joint']*60 or [x['stage_epoch'] for x in history]!=list(range(1,11))+list(range(1,61)):raise ValueError('parent full head/joint history mismatch')
                if any(not math.isfinite(x['validation'][k]) for x in history for k in ('rmse','r2','mae')):raise ValueError('nonfinite parent validation')
                winner=min(history,key=lambda x:x['validation']['rmse'])
                if met['status']!='COMPLETE' or met['best_epoch']!=winner['epoch'] or cons['best_epoch']!=winner['epoch'] or met['best_stage']!=winner['stage'] or any(met[k]!=winner['validation'][k] for k in ('rmse','r2','mae')):raise ValueError('original parent BEST selection mismatch')
                if cfg!=old['estimator_config'] or met['config']!=cfg or cfg['execution_revision']!=rev or cfg['revision']!=rev or cfg['input_pt_revision']!='r3' or cfg['science_recipe']!='r2' or met['scaler']!=scaler or scaler!=t.read(cfg['scaler_reference']):raise ValueError('parent estimator/scaler identity mismatch')
                runner=t.config(arm,task,fold)
                if any(cfg.get(k)!=v for k,v in runner.items() if k not in ('revision','execution_revision')):raise ValueError('parent/new scientific runner differs')
                if old['best_schema']!=schema:raise ValueError('parent BEST schema whitelist mismatch')
                for name in ('best.pt','validation_predictions.npz','metrics.json','history.json','config.json','scaler.json','consumption.json'):
                    if not (root/name).is_file():raise FileNotFoundError(root/name)
                qualified[key]=dict(kind='parent_complete_alias',arm=arm,task=task,fold=fold,root=str(root),estimator_config=cfg,runner_config=runner,best_schema=schema,input_pt=str(t.INPUT_PT),input_pt_revision='r3',random_head=str(t.head_path(task,fold)),seed=42+fold,scaler_source=runner['scaler_reference'],best_epoch_fixed=winner['epoch'],consumption_source=str(root/'consumption.json'),qualified_consumption=expected,source_job_exit=0,source_revision=rev)
    if len(qualified)!=198:raise ValueError('exact198 parent aliases required')
    return qualified

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
    canonical=qualify_parent_aliases()
    for arm,task,fold in t.NEW_UNITS:
        key=f'{arm}/{task}/{fold}';runner=t.config(arm,task,fold)
        canonical[key]=dict(kind='new_ft',arm=arm,task=task,fold=fold,root=str(t.unit_path(arm,task,fold)),estimator_config=runner,runner_config=runner,best_schema='ua3d-cond-spectral-best-r6',input_pt=str(t.INPUT_PT),input_pt_revision='r3',random_head=str(t.head_path(task,fold)),seed=42+fold,scaler_source=runner['scaler_reference'],best_epoch_fixed=None,consumption_source=str(t.unit_path(arm,task,fold)/'consumption.json'))
    if len(canonical)!=210 or sum(x['kind']=='parent_complete_alias' for x in canonical.values())!=198:raise ValueError('canonical exact matrix')
    t.exclusive(t.OUT/'registry/canonical.json',dict(plan=t.PLAN,execution_revision='r6',science_recipe='r2',units=canonical,parent_complete_aliases=198,new_ft_units=12,model_tensors_loaded_during_prepare=0))
    t.exclusive(t.OUT/'source_aliases.json',dict(plan=t.PLAN,execution_revision='r6',science_recipe='r2',available_source_pt_completed=10000,input_pt=str(t.INPUT_PT),units=records,new_PT=0,new_initialization=0,new_stat_fit=0,parent_complete_aliases=198,parent_registry=str(t.P5/'registry/canonical.json'),new_units=t.NEW_UNITS,model_tensors_loaded_during_prepare=0))
    t.exclusive(t.OUT/'identity.json',dict(ident,revision='r6',execution_revision='r6',input_pt_revision='r3',qualification_source=str(t.P3/'identity.json')))
    t.exclusive(t.OUT/'sources.json',dict(input_pt=str(t.INPUT_PT),random_heads_root=str(t.P3/'initialization/heads'),surface=str(t.SURFACE),canonical=str(t.OUT/'registry/canonical.json'),aliases=str(t.OUT/'source_aliases.json'),forbidden='historical BASE; partial PT; interrupted FT BEST/optimizer/RNG; parent complete BEST for initialization',test_targets_read=False))
    t.check_clock(True);print('R6 METADATA AND198 COMPLETE ALIASES QUALIFIED; NO MODELS/TENSORS',flush=True)
if __name__=='__main__':main()
