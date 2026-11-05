"""The sole final CPU suite, after authorized execution; no model inference."""
import json
import os
from pathlib import Path
import numpy as np
import pytest
import torch
from src.modules import ua3d_base_opt as m
from src.training import ua3d_base_opt as t
from scripts import run_ua3d_base_opt as c

torch.set_num_threads(2)


def test_original_control_loss_and_gradients_match_native():
    generator=torch.Generator().manual_seed(17)
    z=[torch.randn(7,8,generator=generator,requires_grad=True) for _ in range(3)]
    available=torch.tensor([[True,True,True]]*5+[[True,False,True]]*2)
    identity=['a','b','a','c','d','e','f']
    actual,counts=m.alignment_terms(z,available,identity,'P_BASE')
    expected,native_counts=m.native.alignment_terms(z,available,identity,'TMOL3M')
    assert set(actual)==set(expected)
    for key in actual:
        assert torch.equal(counts[key],native_counts[key])
        assert torch.allclose(actual[key],expected[key],rtol=1e-6,atol=1e-6)
    a=sum(actual[k]/counts[k] for k in actual);b=sum(expected[k]/counts[k] for k in expected)
    ga=torch.autograd.grad(a,z,retain_graph=True);gb=torch.autograd.grad(b,z)
    assert all(torch.allclose(x,y,rtol=1e-5,atol=1e-6) for x,y in zip(ga,gb))


def test_symmetric_direction_loss_has_both_orientations_and_pair_counts():
    z=[torch.randn(5,9,requires_grad=True) for _ in range(3)]
    available=torch.ones(5,3,dtype=torch.bool);available[-2:,1]=False
    identity=['a','b','b','c','d']
    base,bn=m.alignment_terms(z,available,identity,'P_BASE')
    reverse,rn=m.alignment_terms(z,available,identity,'P_REVERSE')
    sym,sn=m.alignment_terms(z,available,identity,'P_SYM')
    assert set(sym)==set(base)|set(reverse)
    for key in sym:
        expected=(base|reverse)[key];assert torch.equal(sym[key],expected)
        assert torch.equal(sn[key],(bn|rn)[key])
    assert sn['align_0_2']==4 and sn['align_1_0']==2
    singleton=torch.tensor([[True,True,True]]+[[True,False,True]]*4)
    sums,counts=m.alignment_terms(z,singleton,identity,'P_SYM')
    for key in ('align_1_0','align_0_1','align_1_2','align_2_1'):
        assert sums[key]==0 and counts[key]==0 and torch.isfinite(sums[key])
    sum(sums.values()).backward();assert all(x.grad is not None and torch.isfinite(x.grad).all() for x in z)


def test_warmup_preserves_nominal_total_and_separate_normalization():
    for arm in m.ARMS:
        model=m.BaseOptFuse.__new__(m.BaseOptFuse);torch.nn.Module.__init__(model);model.pt_arm=arm
        for update in (0,1,999,1000,9999):
            model.completed_updates=update;weights=model.weights()
            assert sum(v for k,v in weights.items() if k.startswith('align_'))==pytest.approx(.24*min(1.,update/1000))
    result=dict(sums={'align_0_2':torch.tensor(12.),'align_1_0':torch.tensor(30.)})
    loss=t.window_objective(result,{'align_0_2':.04,'align_1_0':.04},{'align_0_2':torch.tensor(3.),'align_1_0':torch.tensor(5.)})
    assert loss==pytest.approx(.4)


def test_independent_projection_initialization_preserves_rng_and_shared_state():
    t.seed_all(42);control=m.BaseOptFuse('P_BASE',True);after=torch.random.get_rng_state().clone()
    t.seed_all(42);independent=m.BaseOptFuse('P_SYM_IND',True)
    assert torch.equal(after,torch.random.get_rng_state())
    assert all(torch.equal(value,independent.state_dict()[key]) for key,value in control.state_dict().items())
    assert sum(p.numel() for p in independent.parameters())-sum(p.numel() for p in control.parameters())==131584
    assert {id(p) for p in independent.QFP.parameters()}.isdisjoint({id(p) for p in independent.Q2.parameters()})
    assert any(not torch.equal(x,y) for x,y in zip(independent.QFP.parameters(),independent.Q2.parameters()))
    # FT always uses the native method and original three-direction objective.
    assert m.BaseOptFuse.training_loss is m.native.PaperFuse.training_loss


def test_serial_budget_failed_attempt_and_exact_tie_policy():
    assert t.ALLOWED_GPUS==(2,) and t.LIMITS['cpu_suites']==1
    assert t.nominate({arm:dict(macro=.8) for arm in m.ARMS})=='P_BASE'
    assert t.nominate({arm:dict(macro=.8+(1e-12 if arm=='P_SYM' else 0)) for arm in m.ARMS})=='P_SYM'
    job=dict(kind='pt',reservation=c.reservation('pt'),consumption=None,exit=1)
    value=c.ledger([job]);assert value['charged_or_reserved']['pt_updates']==10000
    assert value['known_actual_lower_bound']['pt_starts']==1 and value['known_actual_lower_bound']['pt_updates']==0
    assert not value['budget_violations']
    for kind in ('ft','test'):
        total={}
        for arm in t.ARMS:
            for task in t.TASKS:
                for fold in range(5):
                    for key,x in c.reservation(kind,task,fold).items():total[key]=total.get(key,0)+x
        assert all(t.LIMITS[k]==x for k,x in total.items())
    with pytest.raises(ValueError):t.frozen_unit(dict(frozen=False),'P_BASE','eea',0)
    with pytest.raises(ValueError):m.validate_package(dict(schema='old'))


def test_actual_evidence_budget_identity_freeze_and_prediction_metrics():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    budget=t.read(t.OUT/'budget.json');assert not budget['budget_violations']
    assert len([j for j in budget['jobs'] if j['kind']=='cpu'])==1
    for job in budget['jobs']:
        if job['kind'] in ('pt','ft','test'):
            assert job['physical_gpu']==2 and job['env']['CUDA_VISIBLE_DEVICES']=='2'
            assert job['env']['WORLD_SIZE']=='1' and job['gpu_uuid']
        if job['kind']!='cpu':assert job['exit']==0
    frozen=t.read(t.OUT/'selection/final_frozen.json');assert frozen['frozen'] is True and len(frozen['units'])==140
    selection=t.read(t.OUT/'selection/validation_selection.json')
    assert frozen['PT_DIRECTION_SELECTED']==t.nominate(selection['validation'])
    result=t.read(t.OUT/'results.json');assert len(result['formal'])==4 and result['independent_review']=='PENDING'
    for arm in t.ARMS:
        usage=t.read(t.OUT/'pretrain'/arm/'consumption.json')
        assert usage['status']=='COMPLETE' and usage['completed_updates']==10000 and usage['pt_microbatches']==60000
        assert usage['stream']['counters']==[60000]*3
        for task in t.TASKS:
            for fold in range(5):
                u=t.frozen_unit(frozen,arm,task,fold)
                for formal in (False,True):
                    folder=t.unit_path(arm,task,fold,formal)
                    p=np.load(folder/('test_predictions.npz' if formal else 'validation_predictions.npz'),allow_pickle=False)
                    assert len(p['indices'])==len(np.unique(p['indices']))
                    assert len(p['y_true'])==len(p['sample_keys'])==len(p['indices'])
                    if formal:
                        assert p['indices'].tolist()==u['test_indices'] and p['sample_keys'].tolist()==u['test_keys']
                        assert np.allclose(p['y_true'],u['test_targets'],rtol=1e-5,atol=1e-5)
                    actual=t.metrics(p['y_true'],p['y_pred']);saved=t.read(folder/'metrics.json')
                    for key in ('r2','rmse','mae'):assert actual[key]==pytest.approx(saved[key],rel=1e-10,abs=1e-10)
                assert Path(u['best']).is_file()
