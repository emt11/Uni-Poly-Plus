"""Sole final CPU suite, after every registered formal TEST; no training or predictions."""
import copy
import os
from pathlib import Path
import numpy as np
import pytest
import torch
from torch import nn
from src.modules import ua3d_base_opt_r6 as m
from src.training import ua3d_base_opt_r6 as t
if t.CONTINUATION:
    from scripts import run_ua3d_base_opt_r6_concurrency as c
else:
    from scripts import run_ua3d_base_opt_r6 as c

torch.set_num_threads(2)

def test_directional_losses_normalization_and_missing_positives():
    generator=torch.Generator().manual_seed(13)
    features=[torch.randn(6,7,generator=generator,requires_grad=True) for _ in range(3)]
    available=torch.ones(6,3,dtype=torch.bool);available[-2:,1]=False
    identity=['a','b','a','c','d','e']
    actual,counts=m.alignment_terms(features,available,identity)
    expected,reference_counts=m.native.alignment_terms(features,available,identity,'TMOL3M')
    for key in actual:
        assert torch.equal(counts[key],reference_counts[key])
        assert torch.allclose(actual[key],expected[key],rtol=1e-6,atol=1e-6)
    grad_a=torch.autograd.grad(sum(actual[k]/counts[k] for k in actual),features,retain_graph=True)
    grad_b=torch.autograd.grad(sum(expected[k]/counts[k] for k in expected),features)
    assert all(torch.allclose(a,b,rtol=1e-5,atol=1e-6) for a,b in zip(grad_a,grad_b))
    reverse,rn=m.alignment_terms(features,available,identity,m.PAPER)
    both,bn=m.alignment_terms(features,available,identity,m.CURRENT+m.PAPER)
    assert set(both)==set(actual)|set(reverse)
    for key in both:assert torch.equal(both[key],(actual|reverse)[key]) and torch.equal(bn[key],(counts|rn)[key])
    assert any(not torch.isclose(actual[f'align_{a}_{b}']/counts[f'align_{a}_{b}'],reverse[f'align_{b}_{a}']/rn[f'align_{b}_{a}']) for a,b in m.CURRENT)
    available[:,1]=False;available[0,1]=True
    sums,counts=m.alignment_terms(features,available,identity,m.CURRENT+m.PAPER)
    for key in ('align_0_1','align_1_0','align_1_2','align_2_1'):assert counts[key]==0 and sums[key]==0
    reduced=t.window_objective(dict(sums={'x':torch.tensor(12.),'y':torch.tensor(30.)}),
        {'x':.04,'y':.04},{'x':torch.tensor(3.),'y':torch.tensor(5.)})
    assert reduced==pytest.approx(.4)

def test_projection_copy_rng_state_conversion_and_native_dropout_positions():
    t.seed_all(42);shared=m.BaseOptFuse(True);rng=torch.random.get_rng_state().clone()
    t.seed_all(42);independent=m.BaseOptFuse(True,pt=m.pt_config('P_CURRENT_IND'))
    assert torch.equal(rng,torch.random.get_rng_state())
    for name,value in shared.state_dict().items():assert torch.equal(value,independent.state_dict()[name])
    assert all(torch.equal(a,b) and a.data_ptr()!=b.data_ptr() for a,b in zip(independent.Q2.parameters(),independent.QFP.parameters()))
    assert sum(p.numel() for p in independent.parameters())-sum(p.numel() for p in shared.parameters())==131584
    for arm in m.PT_ARMS:
        pt=m.pt_config(arm)
        model=m.BaseOptFuse.__new__(m.BaseOptFuse);nn.Module.__init__(model);model.pt=pt
        for update in (0,1,999,1000,9999):
            model.completed_updates=update
            assert sum(v for k,v in model.weights().items() if k.startswith('align_'))==pytest.approx(.24*min(1.,update/1000))
    downstream=m.BaseOptFuse(False,ft=dict(m.DEFAULT_FT,p2=.1,p3=.1,pfp=0.,phead=.1))
    m.ft_state(dict(state_dict=independent.state_dict()),downstream)
    assert not hasattr(downstream,'QFP') and not hasattr(downstream,'atom_head')
    for name,value in downstream.state_dict().items():
        if not name.startswith(('fusion.','head.')):assert torch.equal(value,independent.state_dict()[name])
    assert isinstance(downstream.encoder3d.layers[0].ffn[2],nn.Identity)
    assert all(v.p==.1 for v in downstream.o8.modules() if isinstance(v,nn.Dropout))
    assert all(v.p==0. for v in downstream.fp_encoder.modules() if isinstance(v,nn.Dropout))
    optimizer,groups=m.optimizer_ft(downstream,'head');downstream.train()
    assert all(not getattr(downstream,key).training for key in m.NATIVE)
    assert downstream.head.training and downstream.fp_encoder.training
    optimizer,groups=m.optimizer_ft(downstream,'joint');downstream.train()
    assert all(getattr(downstream,key).training for key in m.NATIVE)
    assert groups[0]['lr']==1e-4 and groups[1]['lr']==.001
    with pytest.raises(ValueError):m.ft_state(dict(state_dict={'unknown':torch.tensor(1.)}),downstream)
    with pytest.raises(ValueError):m.validate_package(dict(schema='r5'))

def test_fusion_initialization_and_absent_geometry_remains_zero():
    # Exercise only synthetic fusion representations, never encoder/property inference.
    t.seed_all(42);gate=m.BaseOptFuse(False);rng=torch.random.get_rng_state().clone()
    t.seed_all(42);residual=m.BaseOptFuse(False,ft=dict(m.DEFAULT_FT,fusion='residual'))
    assert torch.equal(rng,torch.random.get_rng_state())
    assert sum(p.numel() for p in residual.parameters())-sum(p.numel() for p in gate.parameters())==196864
    assert torch.count_nonzero(residual.residual.weight)==0 and torch.count_nonzero(residual.residual.bias)==0
    generator=torch.Generator().manual_seed(91);z=[torch.randn(4,256,generator=generator) for _ in range(3)]
    available=torch.ones(4,3,dtype=torch.bool);available[-2:,1]=False;z[1][-2:]=0
    expected,w=gate.fuse(z,available);actual,rw=residual.fuse(z,available)
    assert torch.equal(actual,expected) and torch.equal(w,rw)
    normalized=m.BaseOptFuse(False,ft=dict(m.DEFAULT_FT,fusion='modalnorm'))
    assert sum(p.numel() for p in normalized.parameters())-sum(p.numel() for p in gate.parameters())==1536
    with torch.no_grad():normalized.modalnorm[1].bias.fill_(3.)
    captured=[]
    hook=normalized.fusion.register_forward_pre_hook(lambda module,args:captured.append(args))
    normalized.fuse(z,available);hook.remove()
    assert torch.count_nonzero(captured[0][0][1][-2:])==0
    assert torch.count_nonzero(normalized.last_diagnostics['gate_mean'])>0
    assert torch.equal(w[-2:,1],torch.zeros(2))

def test_validation_guard_control_ties_and_full_precision():
    def item(macro,task_delta=0.):return dict(macro=macro,tasks={task:dict(mean=.8+task_delta) for task in t.TASKS})
    control=item(.8);tiny=item(.8+1e-12);bad=item(.9,-.006)
    chosen,rule=t.select({'control':control,'tiny':tiny,'bad':bad},['control','tiny','bad'],'control')
    assert chosen=='tiny' and rule['rejected']==['bad']
    assert t.select({'control':control,'tiny':control},['tiny','control'],'control')[0]=='control'
    with pytest.raises(ValueError):t.frozen_unit(dict(frozen=False),'x','eea',0)
    assert c.HARD_SECONDS==96*3600 and c.CPU_SECONDS==600 and c.CLOSEOUT_SECONDS==300

def test_final_actual_budget_alias_freeze_single_test_order_and_predictions():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    budget=t.read(t.OUT/'budget.json');assert not budget['budget_violations']
    jobs=budget['continuation_jobs'] if t.CONTINUATION else budget['jobs']
    assert len([j for j in jobs if j['kind']=='cpu'])==1
    assert jobs[-1]['kind']=='cpu' and all(j['exit']==0 for j in jobs[:-1])
    assert len([j for j in budget['jobs'] if j['kind']=='initialize'])==1
    if t.CONTINUATION:
        assert not any(j['kind']=='initialize' for j in jobs)
        assert [j['exit'] for j in budget['parent_jobs']]==[0,1]
        assert budget['extra_ft_used']==1 and budget['extra_ft_unit']==list(c.EXTRA_UNIT)
        assert jobs[0]['replaces_parent_job']=='base_opt_r6_ft_A_L_A8_B8_eea_0'
        assert t.INITIAL==t.SOURCE/'initialization/common_step0.pt'
        assert not (t.OUT/'initialization/common_step0.pt').exists()
        assert t.read(t.OUT/'initialization/source.json')['reinitialized'] is False
    assert [j['kind'] for j in jobs].index('test')>max(i for i,j in enumerate(jobs) if j['kind'] in ('ft','pt'))
    record=t.registry();assert len(record['recipes'])<=23 and len(record['pt'])==8
    unique=[t.signature(v) for v in record['recipes'].values()]
    assert all(unique[i]!=unique[j] for i in range(len(unique)) for j in range(i))
    for slot in record['slots']:
        assert slot['canonical'] in record['recipes']
        if slot['alias']:assert record['aliases'][slot['slot']]==slot['canonical']
    frozen=t.read(t.OUT/'selection/final_frozen.json');assert frozen['frozen'] and len(frozen['units'])==35*len(record['recipes'])
    assert frozen['test_targets_read'] is False
    values={key:t.aggregate(key) for key in record['recipes']}
    assert frozen['FINAL']==t.select(values,list(record['recipes']),t.BASELINE)[0]
    manifest=t.read(t.OUT/'execution_manifest.json')
    assert manifest['physical_gpus']==[2] and manifest['world_size']==1
    if t.CONTINUATION:
        assert manifest['resource_revision']=='gpu2-pt2-fttest4-c1'
        assert manifest['caps']==dict(pt=2,ft=4,test=4)
        assert manifest['t0_unix']==c.T0 and manifest['deadline_unix']==c.DEADLINE
        assert manifest['parent_controller_exit']==1 and manifest['parent_initialization_exit']==0
        assert c.verify_parent()['budget']==t.read(t.SOURCE/'budget.json')
        for line in (t.OUT/'scheduler/dispatch.jsonl').read_text().splitlines():
            slots=__import__('json').loads(line)['slots']
            assert slots['pt']<=2 and slots['ft']<=4 and slots['test']<=4
            assert not (slots['pt'] and (slots['ft'] or slots['test']))
            assert not (slots['ft'] and slots['test'])
        for prefix in ('E_','F_'):
            pt=[i for i,j in enumerate(jobs) if j['kind']=='pt' and j['arm'].startswith(prefix)]
            ft=[i for i,j in enumerate(jobs) if j['kind']=='ft' and j['arm'].startswith(prefix)]
            assert max(pt)<min(ft)
    else:assert manifest['gpu_concurrency']==1
    assert manifest['deadline_unix']-manifest['t0_unix']==96*3600
    assert manifest['controller_identity'] and manifest['independent_review']=='PENDING'
    for job in jobs:
        if job['kind'] in ('pt','ft','test'):
            assert job['physical_gpus']==[2] and job['physical_gpu']==2 and job['env']['CUDA_VISIBLE_DEVICES']=='2'
            assert job['env']['WORLD_SIZE']=='1' and job['gpu_uuid']==manifest['gpu_uuid']
        else:assert job['env']['CUDA_VISIBLE_DEVICES']==''
    for pt_id in record['pt']:
        usage=t.read(t.OUT/'pretrain'/pt_id/'consumption.json')
        assert usage['status']=='COMPLETE' and usage['completed_updates']==10000 and usage['pt_microbatches']==60000
        assert usage['pt_presentations']==10080000 and usage['stream']['counters']==[60000]*3
        assert usage['prefetched_not_consumed']==0 and len(usage['workers'])==4
    for recipe_id in record['recipes']:
        for task in t.TASKS:
            for fold in range(5):
                unit=t.frozen_unit(frozen,recipe_id,task,fold)
                assert 'test_targets' not in unit and Path(unit['best']).is_file()
                for formal in (False,True):
                    folder=t.unit_path(recipe_id,task,fold,formal)
                    with np.load(folder/('test_predictions.npz' if formal else 'validation_predictions.npz'),allow_pickle=False) as pred:
                        assert len(pred['indices'])==len(np.unique(pred['indices']))==len(pred['sample_keys'])==len(pred['y_true'])
                        actual=t.metrics(pred['y_true'],pred['y_pred']);saved=t.read(folder/'metrics.json')
                        for key in ('r2','mae','rmse'):assert actual[key]==pytest.approx(saved[key],rel=1e-10,abs=1e-10)
                        if formal:
                            assert pred['indices'].tolist()==unit['test_indices'] and pred['sample_keys'].tolist()==unit['test_keys']
                            alignment=t.read(folder/'identity_alignment.json')
                            assert alignment['matched'] and np.allclose(pred['y_true'],alignment['raw_targets'],rtol=1e-5,atol=1e-5)

def test_continuation_slot_limits_phase_exclusion_and_failed_parent_budget():
    if not t.CONTINUATION:return
    assert c.slot_available(['pt'],'pt') and not c.slot_available(['pt','pt'],'pt')
    assert c.slot_available(['ft']*3,'ft') and not c.slot_available(['ft']*4,'ft')
    assert c.slot_available(['test']*3,'test') and not c.slot_available(['test']*4,'test')
    with pytest.raises(ValueError):c.slot_available(['pt'],'ft')
    with pytest.raises(ValueError):c.slot_available(['ft'],'test')
    parent=t.read(t.SOURCE/'budget.json')
    value=c.cumulative_ledger(parent,[])
    assert value['charged_or_reserved']['initialize_starts']==1
    assert value['charged_or_reserved']['ft_starts']==1 and value['charged_or_reserved']['ft_updates']==700
    assert value['known_actual_lower_bound']['ft_updates']==684 and value['known_completed']['ft_epochs']==68
    assert t.LIMITS['ft_starts']==806 and t.LIMITS['ft_epochs']==56420 and t.LIMITS['ft_updates']==1288700
    assert value['parent_jobs'][1]['exit']==1 and value['limits']['cpu_suites']==1
    spec=dict(kind='ft',arm='A_L_A8_B8',task='eea',fold=0,reservation=dict(ft_starts=1,ft_epochs=70,ft_updates=700),
        log=str(t.OUT/'synthetic-no-process-record'),exit=1,consumption=None)
    used=c.cumulative_ledger(parent,[spec]);assert used['charged_or_reserved']['ft_updates']==1400 and used['extra_ft_used']==1
    assert 'replacement_allowance_reused' in c.cumulative_ledger(parent,[spec,spec])['budget_violations']
