"""One final CPU correctness suite; never run before the authorized experiment."""
import copy
from pathlib import Path
import numpy as np
import pytest
import torch
from src.dataset.ua2d_ph_descriptor import (describe,interval_features,fit_feature_scaler,transform_features,PHCache,INPUT_SCHEMA,SEMANTICS,COLUMNS)
from src.dataset.mcl_ph_view import persistence_pairs
from src.modules.ua2d_ph_descriptor import (PHModel,new_branches,optimizer,restore,validate_parent,ARMS,SCHEMA,ARCHITECTURE)
from src.training import ua2d_ph_descriptor as runtime
from scripts.run_ua2d_ph_descriptor import can_dispatch


def test_betti_half_open_boundary_finite_stats_and_all_censored_h0():
    values=interval_features([(0.,1.5),(1.5,2.),(0.,float('inf'))],3)
    assert len(values)==51 and values[0]==pytest.approx(2/3)
    assert values[31:37]==pytest.approx([.75,.75,0.,1.5,1.5,2.])
    assert values[49:]==[1.,1.]
    raw,barcode=describe([6,6],[[0,0,0],[10,0,0]],True)
    assert raw.shape==(412,) and np.all(raw[:31]==1.)
    assert raw[49]==2 and raw[50]==0 and len(barcode['groups'][0]['intervals']['0'])==2


def test_rips_triangles_fill_h1_and_cutoff_retains_right_censoring():
    triangle=np.array([[0.,0.,0.],[1.,0.,0.],[.5,np.sqrt(3)/2,0.]])
    assert persistence_pairs(triangle,max_edge=4.5)[1]==[]
    square=np.array([[0.,0.,0.],[4.,0.,0.],[4.,4.,0.],[0.,4.,0.]])
    pairs=persistence_pairs(square,max_edge=4.5)
    assert len(pairs[1])==1 and pairs[1][0][0]==pytest.approx(4.) and np.isinf(pairs[1][0][1])
    raw,barcode=describe([6]*4,square,True)
    assert raw[51+49]==1 and raw[51+50]==0
    unit_square=square/4
    finite=persistence_pairs(unit_square,max_edge=2.)[1]
    assert len(finite)==1 and finite[0]==pytest.approx((1.,np.sqrt(2.)))


def test_empty_groups_single_atom_and_invalid_geometry_are_not_dropped():
    raw,barcode=describe([7],[[0.,0.,0.]],True)
    assert raw[49]==1 and np.all(raw[:31]==1.)
    assert np.count_nonzero(raw[103:206])==0 and np.count_nonzero(raw[309:412])==0
    assert raw[102]==raw[308]==1.
    invalid,record=describe([6],[[float('nan'),0.,0.]],False)
    assert record['valid'] is False and np.count_nonzero(invalid)==0
    assert len(COLUMNS)==412 and SEMANTICS['homology_coeff_field']==2


def test_feature_scaler_uses_valid_train_only_and_constant_columns_are_one():
    raw=np.zeros((4,412),dtype=np.float32);raw[0,0]=1;raw[1,0]=3;raw[2,:]=1e7
    valid=np.array([True,True,True,False]);s=fit_feature_scaler(raw,valid,[0,1,3])
    assert s['valid_training_rows']==[0,1]
    assert s['mean'][0]==pytest.approx((np.log(2)+np.log(4))/2)
    assert all(x==1. for x in s['scale'][1:])
    changed=raw.copy();changed[2,:]=1e30
    assert fit_feature_scaler(changed,valid,[0,1,3])==s
    assert np.isfinite(transform_features(raw,s)).all()
    with pytest.raises(ValueError):fit_feature_scaler(raw,np.zeros(4,dtype=bool),[0,1])


def test_ph_cache_identity_and_no_invalid_bias_payload(tmp_path):
    runtime.write(tmp_path/'metadata.json',dict(plan=runtime.PLAN,schema=INPUT_SCHEMA,status='COMPLETE',records=5875,unique_structures=3654,semantics=SEMANTICS,graph_cache='/graph',online_fallback=False))
    raw=np.zeros((3654,412),dtype=np.float32);raw[0,0]=1;valid=np.zeros(3654,dtype=bool);valid[0]=True
    np.save(tmp_path/'descriptors.npy',raw);np.save(tmp_path/'ph_valid.npy',valid)
    rows={'eea':[dict(row=i,key=f'k{i}',cache_key=f'eea:{i}',ph_index=i%3654,smiles='*CC*') for i in range(5875)]};runtime.write(tmp_path/'rows.json',rows)
    c=PHCache(tmp_path,'/graph');scaler=fit_feature_scaler(raw,valid,[0]);value,mask=c.get('eea',0,'k0',scaler)
    assert value.shape==(412,) and bool(mask)
    with pytest.raises(ValueError):c.get('eea',0,'wrong-key',scaler)
    rows['eea'].append(rows['eea'][0]);runtime.write(tmp_path/'rows.json',rows)
    with pytest.raises(ValueError):PHCache(tmp_path,'/graph')


def test_isolated_common_initialization_and_ua_equivalence_and_real_gradients():
    torch.manual_seed(17);before=torch.get_rng_state().clone();a,b,g=new_branches();assert torch.equal(before,torch.get_rng_state())
    c,d,h=new_branches();assert all(torch.equal(v,c.state_dict()[k]) for k,v in a.state_dict().items())
    model=PHModel('GATE_PH').eval();ua=torch.randn(3,512);features=torch.randn(3,412);valid=torch.ones(3,dtype=torch.bool)
    assert torch.equal(model.predict_features(ua,features,valid),model.head(ua).view(-1))
    z=model.ph_encoder(features);assert torch.equal(2*torch.sigmoid(model.gate(torch.cat([ua,z],-1))),torch.ones_like(z))
    model.predict_features(ua,features,valid).sum().backward();assert model.ph_projection.weight.grad.abs().sum()>0
    with torch.no_grad():model.ph_projection.weight.copy_(torch.linspace(-.01,.01,512*128).reshape(512,128))
    model.zero_grad(set_to_none=True);model.predict_features(ua,features,valid).square().sum().backward()
    assert model.ph_encoder[0].weight.grad.abs().sum()>0 and model.gate[-1].weight.grad.abs().sum()>0
    # Mask AFTER every encoder layer: arbitrary biases cannot leak for invalid geometry.
    assert torch.equal(model.predict_features(ua,features,torch.zeros(3,dtype=torch.bool)),model.head(ua).view(-1))


@pytest.mark.parametrize('arm',ARMS)
def test_complete_restore_and_optimizer_coverage(arm):
    model=PHModel(arm)
    for stage in ('head','joint'):
        opt,evidence=optimizer(model,stage);ids=[id(p) for group in opt.param_groups for p in group['params']]
        assert len(ids)==len(set(ids)) and set(ids)=={id(p) for p in model.parameters() if p.requires_grad}
        model.train();assert model.o8.training==(stage=='joint')
    saved=dict(schema=SCHEMA,architecture=ARCHITECTURE,config={'arm':arm},state_dict=model.state_dict())
    restored=restore(saved,{'arm':arm});assert all(torch.equal(v,restored.state_dict()[k]) for k,v in model.state_dict().items())
    bad=dict(saved,state_dict=dict(saved['state_dict']));bad['state_dict'].pop('ph_projection.weight')
    with pytest.raises((ValueError,RuntimeError)):restore(bad,{'arm':arm})


def test_true_pts10k_parent_rejects_wrong_schema_or_step():
    parent=torch.load(runtime.PARENT,map_location='cpu',weights_only=False);validate_parent(parent)
    with pytest.raises(ValueError):validate_parent(dict(parent,step=5000))
    with pytest.raises(ValueError):validate_parent(dict(parent,schema='ua2d-3dshift-local-geometry-r1'))


def test_gpu2_only_global_four_same_card_pair_mapping_and_full_matrix():
    assert runtime.GPU_LIMITS=={2:4} and runtime.TOTAL_LIMIT==4
    for task in runtime.TASKS:
        for fold in range(5):
            pair=[runtime.unit_config(a,task,fold)['physical_gpu'] for a in ARMS]
            assert pair==[2,2]
    active=[dict(arm=ARMS[i%2],gpu=2) for i in range(4)]
    assert all(not can_dispatch(active,a,'eea',0) for a in ARMS)
    assert can_dispatch(active[:3],ARMS[1],'egc',4)
    assert len({(a,t,f) for a in ARMS for t in runtime.TASKS for f in range(5)})==70


def test_single_freeze_includes_70_complete_models_scalers_and_gpu_map(tmp_path,monkeypatch):
    monkeypatch.setattr(runtime,'OUT',tmp_path);monkeypatch.setattr(runtime,'INPUTS',tmp_path/'inputs');monkeypatch.setattr(runtime,'SPLIT',tmp_path/'split')
    rows={t:[dict(row=f,key=f'k{f}',smiles='*CC*',cache_key=f'{t}:{f}',ph_index=f) for f in range(5)] for t in runtime.TASKS}
    runtime.write(tmp_path/'inputs/metadata.json',dict(schema=INPUT_SCHEMA,semantics=SEMANTICS));runtime.write(tmp_path/'inputs/rows.json',rows);runtime.write(tmp_path/'authorization/P10K_REF.json',{'reference_only':True})
    gpu_map=dict(frozen=True,map={f'{t}/{f}':dict(gpu=2) for t in runtime.TASKS for f in range(5)})
    runtime.write(tmp_path/'selection/gpu_map.json',gpu_map)
    validation={}
    for task in runtime.TASKS:
        runtime.write(tmp_path/'split'/f'{task}.json',dict(folds=[dict(test_indices=[f]) for f in range(5)]))
        for f in range(5):runtime.write(runtime.feature_scaler_path(task,f),{'columns':COLUMNS,'fit':'valid train only'})
    for arm in ARMS:
        validation[arm]={'macro':-.5}
        for task in runtime.TASKS:
            for fold in range(5):
                cfg=runtime.unit_config(arm,task,fold);p=runtime.unit_path(arm,task,fold)
                runtime.write(p/'metrics.json',dict(status='COMPLETE',config=cfg,scaler={'fit':'train'},feature_scaler=runtime.read(runtime.feature_scaler_path(task,fold)),best_epoch=1,best_stage='head',attention=cfg['attention']))
                runtime.write(p/'consumption.json',dict(status='COMPLETE',completed_epochs=70,validation_started=70));runtime.write(p/'config.json',cfg);runtime.write(p/'scaler.json',{'fit':'train'});(p/'best.pt').touch()
    with pytest.raises(ValueError):runtime.freeze_final({ARMS[0]:validation[ARMS[0]]})
    frozen=runtime.freeze_final(validation)
    assert len(frozen['units'])==70 and frozen['FINAL']=='GATE_PH' and frozen['gpu_map']==gpu_map
    assert all('feature_scaler' in u and 'sample_mapping' in u for u in frozen['units'].values())
    with pytest.raises(FileExistsError):runtime.freeze_final(validation)
