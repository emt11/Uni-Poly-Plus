"""The single final CPU suite; never a preflight/smoke run."""
import copy
from pathlib import Path
import numpy as np
import pytest
import torch
from src.modules.ua2d_fg10k_morgan import (MorganModel,new_branches,optimizer,restore,SCHEMA,ARCHITECTURE,ARMS)
from src.dataset.ua2d_fg10k_morgan import validate_bits,MorganCache,INPUT_SCHEMA,FP_SPEC
from src.training import ua2d_fg10k_morgan as runtime
from scripts.run_ua2d_fg10k_morgan import can_dispatch


def test_fp_payload_is_real_binary_not_packed_or_empty():
    value=np.zeros(2048,dtype=np.uint8);value[[0,7,2047]]=1
    assert np.array_equal(validate_bits(value.tobytes()),value)
    for invalid in [bytes(256),bytes(2048),bytes([1])*2048,bytes([2])+bytes(2047)]:
        with pytest.raises(ValueError):validate_bits(invalid)


def test_cache_maps_original_identity_and_never_silently_substitutes(tmp_path):
    runtime.write(tmp_path/'metadata.json',dict(plan=runtime.PLAN,schema=INPUT_SCHEMA,status='COMPLETE',records=5875,unique_smiles=3654,
        graph_cache='/graph',dtype='uint8',feature_spec=FP_SPEC,payload='2048 uint8 binary entries, not packbits',online_fallback=False))
    rows={'eea':[dict(row=i,key=f'key{i}',cache_key=f'eea:{i}',fp_index=i,smiles='*CC*') for i in range(5875)]}
    runtime.write(tmp_path/'rows.json',rows);array=np.zeros((5875,2048),dtype=np.uint8);array[:,0]=1;array[1,1]=1;np.save(tmp_path/'morgan.npy',array)
    cache=MorganCache(tmp_path,'/graph');assert cache.get('eea',1,'key1').sum()==2
    with pytest.raises(ValueError):cache.get('eea',1,'key0')
    with pytest.raises(ValueError):cache.get('eea',99999,'missing')
    rows['eea'][1]['fp_index']=0;runtime.write(tmp_path/'rows.json',rows)
    with pytest.raises(ValueError):MorganCache(tmp_path,'/graph')


def test_added_initialization_does_not_touch_rng_and_is_shared():
    torch.manual_seed(1234);before=torch.get_rng_state().clone();a,b,g=new_branches();assert torch.equal(before,torch.get_rng_state())
    c,d,h=new_branches()
    for x,y in [(a,c),(b,d),(g,h)]:
        assert all(torch.equal(value,y.state_dict()[name]) for name,value in x.state_dict().items())
    assert torch.count_nonzero(b.weight)==0 and torch.count_nonzero(g[-1].weight)==0
    assert torch.count_nonzero(g[0].weight)>0


def test_zero_prediction_equivalence_gate_one_and_delayed_gradients():
    model=MorganModel('GATE_FP');model.eval();ua=torch.randn(3,512);bits=torch.randint(0,2,(3,2048)).float()
    expected=model.head(ua).view(-1);assert torch.equal(model.predict_features(ua,bits),expected)
    z=model.fp_encoder(bits);gate=2*torch.sigmoid(model.gate(torch.cat([ua,z],-1)));assert torch.equal(gate,torch.ones_like(gate))
    model.predict_features(ua,bits).sum().backward()
    assert model.fp_projection.weight.grad.abs().sum()>0
    assert all(p.grad is None or p.grad.abs().sum()==0 for p in model.fp_encoder.parameters())
    with torch.no_grad():model.fp_projection.weight.copy_(torch.linspace(-.01,.01,512*128).reshape(512,128))
    model.zero_grad(set_to_none=True);model.predict_features(ua,bits).square().sum().backward()
    assert model.fp_encoder[0].weight.grad.abs().sum()>0 and model.gate[-1].weight.grad.abs().sum()>0
    with torch.no_grad():model.gate[-1].weight.fill_(.01)
    model.zero_grad(set_to_none=True);model.predict_features(ua,bits).square().sum().backward()
    assert model.gate[0].weight.grad.abs().sum()>0
    del model


@pytest.mark.parametrize('arm',ARMS)
def test_optimizer_covers_every_trainable_parameter_and_complete_strict_restore(arm):
    model=MorganModel(arm)
    for stage in ('head','joint'):
        opt,evidence=optimizer(model,stage)
        actual=[id(p) for g in opt.param_groups for p in g['params']]
        assert len(actual)==len(set(actual)) and set(actual)=={id(p) for p in model.parameters() if p.requires_grad}
        model.train();assert model.o8.training==(stage=='joint')
    cfg=dict(arm=arm);saved=dict(schema=SCHEMA,architecture=ARCHITECTURE,config=cfg,state_dict=model.state_dict())
    rebuilt=restore(saved,cfg);assert all(torch.equal(v,rebuilt.state_dict()[k]) for k,v in model.state_dict().items())
    broken=dict(saved,state_dict=dict(saved['state_dict']));broken['state_dict'].pop('fp_projection.weight')
    with pytest.raises((ValueError,RuntimeError)):restore(broken,cfg)


def test_gpu3_total_four_and_two_complete_matrices():
    for arm in ARMS:
        active=[]
        for _ in range(4):assert can_dispatch(active,arm);active.append(dict(arm=arm))
        assert all(not can_dispatch(active,a) for a in ARMS)
    mixed=[dict(arm=ARMS[0]),dict(arm=ARMS[0]),dict(arm=ARMS[1]),dict(arm=ARMS[1])]
    assert all(not can_dispatch(mixed,a) for a in ARMS)
    assert len({(a,t,f) for a in ARMS for t in runtime.TASKS for f in range(5)})==70
    assert runtime.LIMITS['ft_starts']==runtime.LIMITS['test_starts']==70 and runtime.LIMITS['pt_starts']==0


def test_epoch_interruption_is_charged_conservatively():
    state=dict(epochs_started=0,validation_started=0,attempted_updates=0,completed_updates=0,completed_epochs=0)
    runtime.begin_epoch_reservation(state,17);state['attempted_updates']=4;state['completed_updates']=4
    charged=runtime.conservative_ft_counts(state)
    assert charged['charged']['attempted_updates']==17 and charged['known_actual_lower_bound']['attempted_updates']==4
    assert charged['charged']['validation_started']==1 and charged['open_epoch']


def test_single_freeze_requires_all70_and_preserves_complete_source(tmp_path,monkeypatch):
    monkeypatch.setattr(runtime,'OUT',tmp_path);monkeypatch.setattr(runtime,'INPUTS',tmp_path/'inputs');monkeypatch.setattr(runtime,'SPLIT',tmp_path/'splits')
    rows={task:[dict(row=f,key=f'k{f}',smiles='*CC*',fp_index=f,cache_key=f'{task}:{f}') for f in range(5)] for task in runtime.TASKS}
    runtime.write(tmp_path/'inputs/metadata.json',{'schema':INPUT_SCHEMA});runtime.write(tmp_path/'inputs/rows.json',rows);runtime.write(tmp_path/'authorization/FG_REF.json',{'reference_only':True})
    for task in runtime.TASKS:runtime.write(tmp_path/'splits'/f'{task}.json',dict(folds=[dict(test_indices=[f]) for f in range(5)]))
    validation={}
    for arm in ARMS:
        validation[arm]={'macro':-.5}
        for task in runtime.TASKS:
            for fold in range(5):
                cfg=runtime.unit_config(arm,task,fold);p=runtime.unit_path(arm,task,fold)
                runtime.write(p/'metrics.json',dict(status='COMPLETE',config=cfg,scaler={'fit':'train'},best_epoch=1,best_stage='head',attention=cfg['attention']))
                runtime.write(p/'consumption.json',dict(status='COMPLETE',completed_epochs=70,validation_started=70));runtime.write(p/'config.json',cfg);runtime.write(p/'scaler.json',{'fit':'train'});(p/'best.pt').touch()
    with pytest.raises(ValueError):runtime.freeze_final({ARMS[0]:validation[ARMS[0]]})
    # Failed freeze does not publish; all70 legitimate units then freeze exactly once.
    result=runtime.freeze_final(validation)
    assert len(result['units'])==70 and result['FINAL']=='GATE_FP'
    assert all(v['parent_step']==10000 and 'sample_mapping' in v for v in result['units'].values())
    with pytest.raises(FileExistsError):runtime.freeze_final(validation)
