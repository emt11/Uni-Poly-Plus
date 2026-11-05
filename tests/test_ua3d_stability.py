"""Sole final CPU suite after 105 canonical TESTs. No actual model training or predictions."""
import math
import os
from pathlib import Path
import numpy as np
import pytest
import torch
from src.training import ua3d_stability as t
from scripts import run_ua3d_stability as c


def test_exact_comparison_requires_dtype_shape_and_torch_equal():
    a=dict(x=torch.tensor([1.,2.]),rng=np.array([1,2],dtype=np.uint32),scalar=.25)
    assert t.m.exact_compare(a,a)['equal'] is True
    b=dict(a,x=a['x'].clone());b['x'][0]=torch.nextafter(b['x'][0],torch.tensor(2.))
    r=t.m.exact_compare(a,b);assert r['equal'] is False and r['max_abs_diff']>0
    assert t.m.exact_compare(torch.ones(2),torch.ones(2,dtype=torch.float64))['equal'] is False
    assert t.m.exact_compare(torch.ones(2),torch.ones(1,2))['equal'] is False


def test_strict_policy_precedes_cuda_and_environment_consistent():
    assert os.environ['CUDA_VISIBLE_DEVICES']==''
    expected=dict(deterministic=True,warn_only=False,cudnn_benchmark=False,cudnn_deterministic=True,matmul_tf32=False,cudnn_tf32=False,matmul_precision='highest',fill_uninitialized_memory=True,cpu_threads=2,interop_threads=1)
    registration=t.read(t.OUT/'environment.json')
    b=t.read(t.OUT/'budget.json')
    for j in b['jobs']:
        if j['kind'] not in c.CAPS:continue
        path=Path(j['consumption']).parent;env=t.read(path/'environment.json');det=j['kind'] in ('ft','test') or j['arm'].startswith('DET_')
        assert env['versions']==registration['versions']
        assert env['physical']['physical_gpu']==2 and env['physical']['logical_device']=='cuda:0'
        if det:
            setup=env['setup'];assert setup['configured_before_cuda'] is True and setup['policy']=='DET'
            state=env.get('numerical',env.get('actual'))
            assert all(state[k]==v for k,v in expected.items())
            assert all(state['env'][k]==v and j['env'][k]==v for k,v in t.m.DET_ENV.items())
        else:
            assert not any(k in j['env'] for k in t.m.DET_ENV)
            assert env['setup']['policy']=='DEFAULT'


def test_diagnostic_counts_fixed_train_batch_and_exact_gate():
    b=t.read(t.OUT/'budget.json');js=[j for j in b['jobs'] if j['kind']=='diagnostic']
    assert [j['arm'] for j in js]==t.DIAGNOSTICS and all(j['exit']==0 for j in js)
    fixed=t.read(t.OUT/'registry/recipes.json')['diagnostic_input']
    assert fixed['indices']==t.SPEC['fixed_rows']==t.read(t.SPLIT/'eps.json')['folds'][0]['train_indices'][:24]
    for j in js:
        u=t.read(j['consumption'])
        for k,v in dict(diagnostic_train_starts=1,diagnostic_eval_forwards=2,diagnostic_eval_graphs=48,diagnostic_train_forwards=20,diagnostic_backwards=20,diagnostic_updates=20,diagnostic_train_graphs=480).items():assert u[k]==v
        h=t.read(Path(j['consumption']).parent/'history.json');assert len(h)==20 and [x['stage'] for x in h]==['head']*10+['joint']*10
    d=t.read(t.OUT/'diagnostics/comparison.json');assert d['numerical_diagnostic_pass'] is True
    assert d['checks']['DET']['equal'] is True and all(x['equal'] for x in d['checks']['DET']['details'].values())


def test_independent_full_runs_seed_roles_and_35_pairs():
    b=t.read(t.OUT/'budget.json');f=t.read(t.OUT/'selection/final_frozen.json')
    assert f['FINAL_recipe']=='DET_BASE' and f['PRIMARY_RUN']=='S42_A'
    assert f['canonical']==['S42_A','S142','S242'] and f['replica']=='S42_B'
    assert len(f['units'])==175 and sum(u['inference_allowed'] for u in f['units'].values())==105
    assert not (t.OUT/'formal/DET_BASE/S42_B').exists()
    assert len({(j['kind'],j['arm'],j['task'],j['fold']) for j in b['jobs']})==len(b['jobs'])
    assert sum(j['kind']=='ft' and j['exit']==0 for j in b['jobs'])==140
    assert sum(j['kind']=='test' and j['exit']==0 for j in b['jobs'])==105
    for run,seed in t.SPEC['runs'].items():
        assert sum(j['kind']=='ft' and j['arm']==run for j in b['jobs'])==35
        assert sum(j['kind']=='test' and j['arm']==run for j in b['jobs'])==(0 if run=='S42_B' else 35)
        for task in t.TASKS:
            for fold in range(5):
                path=t.unit_path(run,task,fold);cfg=t.read(path/'config.json')
                assert cfg['seed']==seed+fold and cfg['seed_base']==seed and cfg['numerical_policy']=='DET'
                assert cfg['recipe']['ft']==t.m.DEFAULT_FT and cfg['parent']==str(t.parent_path(run)) and cfg['head']==str(t.head_path(task,fold))
                assert cfg==f['units'][f'{run}/{task}/{fold}']['config']
    for task in t.TASKS:
        for fold in range(5):
            p=t.read(t.OUT/'reproducibility'/task/f'fold{fold}.json')
            assert p['equal'] is True and p['first_divergence'] is None and p['max_abs_diff']==0
            assert p['independent_processes'][0]!=p['independent_processes'][1]
            assert all(x['equal'] for x in p['checks'].values())


def test_70_epoch_order_predictions_and_best_preserved_across_stages():
    for run in t.RUNS:
        for task in t.TASKS:
            split=t.read(t.SPLIT/f'{task}.json')
            for fold in range(5):
                path=t.unit_path(run,task,fold);h=t.read(path/'history.json');met=t.read(path/'metrics.json')
                assert len(h)==70 and [r['stage'] for r in h]==['head']*10+['joint']*60
                assert met['best_epoch']==min(range(70),key=lambda i:h[i]['validation']['rmse'])+1
                assert met['best_stage']==('head' if met['best_epoch']<=10 else 'joint')
                for r in h:
                    rows=r['training_rows'];ends=r['batch_boundaries'];assert sorted(rows)==sorted(split['folds'][fold]['train_indices'])
                    assert ends[-1]==len(rows) and len(ends)==len(r['clip_norms'])==math.ceil(len(rows)/24)
                    assert all(ends[i]-(ends[i-1] if i else 0)==min(24,len(rows)-(ends[i-1] if i else 0)) for i in range(len(ends)))
                    assert r['runtime_contract']['dropout']['head.2']==dict(p=.5,training=True)
                    assert all(r['runtime_contract']['native_modes'][n]==(r['epoch']>10) for n in t.m.NATIVE)
                    assert all(math.isfinite(x) for x in r['clip_norms'])
                    with np.load(path/'validation'/f'epoch{r["epoch"]:03d}.npz',allow_pickle=False) as pred:
                        assert np.isfinite(pred['y_pred']).all()
                if run in ('S42_A','S42_B'):assert (path/'last.pt').is_file()
    for run in ('S42_A','S42_B'):
        for epoch in (10,11):
            s=torch.load(t.unit_path(run,'eps',0)/f'boundary_epoch{epoch}.pt',map_location='cpu',weights_only=False,mmap=True)
            assert s['epoch']==epoch and 'data_loader' in s['rng']


def test_formal_row_key_scaler_p_and_frozen_best_metadata():
    frozen=t.read(t.OUT/'selection/final_frozen.json');assert frozen['test_targets_read'] is False
    for task in t.TASKS:
        assert frozen['registry']['splits'][task]==t.read(t.SPLIT/f'{task}.json')
        for fold in range(5):
            ref=t.unit_path('BASE_ORIGINAL',task,fold,True)
            with np.load(ref/'test_predictions.npz',allow_pickle=False) as b:
                for run in t.CANONICAL:
                    path=t.unit_path(run,task,fold,True)
                    with np.load(path/'test_predictions.npz',allow_pickle=False) as a:
                        for key in ('indices','sample_keys','y_true','availability'):assert np.array_equal(a[key],b[key])
                        assert np.isfinite(a['y_pred']).all()
                    assert t.read(path/'scaler.json')==t.read(t.base_path(task,fold)/'scaler.json')
                    rt=t.read(path/'consumption.json')['evaluation_runtime'];assert rt['training'] is False and all(not x['training'] for x in rt['dropout'].values())
                    assert rt['dropout']['head.2']['p']==.5
                    u=frozen['units'][f'{run}/{task}/{fold}'];saved=torch.load(u['best'],map_location='cpu',weights_only=False,mmap=True)
                    assert saved['schema']==t.FT_SCHEMA and saved['config']==u['config'] and saved['best_epoch']==u['best_epoch'] and saved['scaler']==u['scaler']
                    del saved
    assert not (t.OUT/'finetune/BASE_ORIGINAL').exists() and not (t.OUT/'formal/BASE_ORIGINAL').exists()
    with pytest.raises(ValueError):t.frozen_unit(frozen,'S42_B','eps',0)


def test_thresholds_canonical_statistics_no_best_seed_selection():
    values={run:t.aggregate(run,True) for run in t.CANONICAL}
    macros=[values[run]['macro'] for run in t.CANONICAL]
    assert len(macros)==3 and 'S42_B' not in values
    stats,passed=t.seed_statistics(values)
    assert stats['mean']==float(np.mean(macros)) and stats['std']==float(np.std(macros,ddof=1))
    assert passed==((max(macros)-min(macros)<=.005) and all(max(values[r]['tasks'][task]['mean'] for r in t.CANONICAL)-min(values[r]['tasks'][task]['mean'] for r in t.CANONICAL)<=.020 for task in t.TASKS))
    synthetic={run:dict(macro=x,tasks={task:dict(mean=.5) for task in t.TASKS}) for run,x in zip(t.CANONICAL,[0.,.002,.005])}
    synthetic['S42_B']=dict(macro=99.,tasks={task:dict(mean=99.) for task in t.TASKS})
    assert t.seed_statistics(synthetic)[1] is True
    synthetic['S242']['macro']=.006;assert t.seed_statistics(synthetic)[1] is False
    synthetic['S242']['macro']=.005;synthetic['S242']['tasks']['eps']['mean']=.521
    assert t.seed_statistics(synthetic)[1] is False
    original=t.aggregate('BASE_ORIGINAL',True);primary=values['S42_A']
    q,p,d=t.primary_assessment(dict(BASE_ORIGINAL=original,S42_A=primary))
    assert d==primary['macro']-t.REFERENCE and p==(d>=.010)
    assert q['pass']==(primary['macro']>=t.REFERENCE-.005 and all(primary['tasks'][task]['mean']>=original['tasks'][task]['mean']-.010 for task in t.TASKS))
    f=t.read(t.OUT/'selection/final_frozen.json');assert f['PRIMARY_RUN']=='S42_A' and f['FINAL_recipe']=='DET_BASE'
    assert t.SPEC['PRIMARY_RUN']=='S42_A'


def test_order_resources_budget_clock_writer_and_true_exits():
    b=t.read(t.OUT/'budget.json');jobs=b['jobs'];f=t.read(t.OUT/'selection/final_frozen.json')
    assert b['budget_violations']==[] and b['charged_or_reserved']==t.LIMITS
    assert c.ledger(jobs)['budget_violations']==[]
    assert sum(j['kind']=='cpu' for j in jobs)==1
    for j in jobs:
        assert j['env']['WORLD_SIZE']=='1' and j['env']['CUDA_VISIBLE_DEVICES']==('2' if j['kind'] in c.CAPS else '')
        if j['kind']=='cpu':continue
        assert j['exit']==int((Path(j['log'])/'exit.txt').read_text())==t.read(Path(j['log'])/'end.json')['exit']==0
        if j['kind'] in c.CAPS:
            p=t.read(Path(j['log'])/'process.json');u=t.read(j['consumption'])
            assert u['process']==p['timeout_identity'] and u['actual_device']['gpu_uuid']==j['gpu_uuid']
    assert max(j['ended'] for j in jobs if j['kind']=='diagnostic')<min(j['reserved_at'] for j in jobs if j['kind']=='ft')
    assert max(j['ended'] for j in jobs if j['kind']=='ft')<f['time']<min(j['reserved_at'] for j in jobs if j['kind']=='test')
    assert max(j['ended'] for j in jobs if j['kind']=='test')<next(j['reserved_at'] for j in jobs if j['kind']=='cpu')
    counts=dict.fromkeys(c.CAPS,0);events=[]
    for j in jobs:
        if j['kind'] in c.CAPS:events.extend([(j['reserved_at'],1,j['kind']),(j['ended'],-1,j['kind'])])
    for _,change,kind in sorted(events):
        counts[kind]+=change;assert 0<=counts[kind]<=c.CAPS[kind] and sum(x>0 for x in counts.values())<=1
    with pytest.raises(ValueError):c.slot_available(['diagnostic'],'ft')
    assert not c.slot_available(['diagnostic'],'diagnostic') and not c.slot_available(['ft']*4,'ft')
    manifest=t.read(t.OUT/'execution_manifest.json');assert manifest['t0_unix']==t.SPEC['t0_unix'] and manifest['deadline_unix']==manifest['t0_unix']+86400
    for name in c.SOURCES:assert (t.OUT/'authorization/source_at_launch'/name).read_text()==(t.ROOT/name).read_text()
