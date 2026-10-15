"""Synthetic CPU checks; no source generation or real-batch GPU probe."""
import copy
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import torch
import pytest
from torch import nn
from tests.test_dual_glt import _toy_pair
from src.dataset.canonical_periodic import build_canonical_periodic_topology
from src.dataset.glt_dual_static import build_dual_static,build_pretrain_target
from src.dataset.glt_dual_pretrain import prepare_pretrain_sample
from src.dataset.glt_mcl_ph import prepare_gp_sample,gp_collate
from src.modules.glt_dual import DualGLTModel
from src.modules.glt_mcl_ph import EndpointAdaptor,GPEncoder
from src.modules.glt_mcl_ph_pretrain import GPPretrainer
COMMON=Path('results/gp_sota_20261004/initialization/common_step0_205_v2.pt')

def record(smiles='*COC*'):
    top=build_canonical_periodic_topology(smiles);_,trimer=_toy_pair(smiles)
    n=trimer.trimer_atomic_number.numel()
    trimer.trimer_formal_charge=torch.zeros(n,dtype=torch.long)
    trimer.trimer_is_aromatic=torch.zeros(n,dtype=torch.bool)
    trimer.trimer_chiral_tag=torch.zeros(n,dtype=torch.long)
    static=build_dual_static(top,trimer,smiles);target=build_pretrain_target(smiles)
    return top,trimer,smiles,static,target

def test_three_arms_share_all_common_and_new_initial_tensors():
    common={k:v for k,v in torch.load(COMMON,weights_only=False)['common_state_dict'].items() if not k.startswith('fp_head.')}
    prior=None
    for route in ('GP','GU','GP_A'):
        torch.manual_seed(42);m=GPPretrainer(route=route,common_path=COMMON)
        state=m.state_dict()
        for k,v in common.items():assert torch.equal(v,state[k]),k
        core={k:v.clone() for k,v in state.items() if not k.startswith(('align_proj2.','align_proj3.'))}
        if prior:
            assert prior.keys()==core.keys()
            for k,v in prior.items():assert torch.equal(v,core[k]),k
        prior=core
        assert m.third_task=='none' and not hasattr(m,'fp_head')
        assert bool(m.align_proj2 is not None)==(route=='GP_A')

def test_endpoint_symmetry_and_permutation():
    torch.manual_seed(42);m=EndpointAdaptor();a=torch.randn(5,128);b=torch.randn(5,128)
    assert torch.equal(m(a,b),m(b,a))
    order=torch.tensor([4,2,0,1,3]);torch.testing.assert_close(m(a[order],b[order]),m(a,b)[order])

def test_joint_view_parity_offsets_and_zero_residual():
    top,trimer,s,static,target=record()
    data,labels=prepare_gp_sample(top,trimer,s,seed=42,key='toy',position=0,static=static,target=target)
    old,truth=prepare_pretrain_sample(top,trimer,s,seed=42,key='toy',position=0,static=static,target=target)
    for k in old.keys():
        if torch.is_tensor(getattr(old,k)):assert torch.equal(getattr(old,k),getattr(data,k)),k
    assert 'fingerprint' not in labels['glt'] and 'fingerprint' not in truth
    for k in ('atom_mask','atom_label','distance','angle_pairs','angle_cos'):
        assert torch.equal(labels['glt'][k],truth[k]),k
    batch,y=gp_collate([(data,labels),(data.clone(),labels)])
    assert torch.equal(batch.mcl_atom_batch[batch.mcl_bond_index[:,0]],batch.bond_batch)
    assert torch.equal(batch.mcl_atom_batch[batch.mcl_bond_index[:,1]],batch.bond_batch)
    torch.manual_seed(42);gp=GPEncoder(dropout=0).eval();legacy=DualGLTModel(dropout=0).eval()
    state=legacy.state_dict();state.update({k:v for k,v in gp.state_dict().items() if k in state});legacy.load_state_dict(state,strict=True)
    gp.residual_scale=0
    with torch.no_grad():
        torch.testing.assert_close(gp(batch),legacy(batch),rtol=0,atol=0)
    gp.residual_scale=.1
    with torch.no_grad():
        a=gp(batch)
        one,_=gp_collate([(data,labels)])
        torch.testing.assert_close(a[0],gp(one)[0],rtol=1e-5,atol=1e-6)
    # Rotation/translation preserve all distance/angle/PH fields without coordinate generation.
    changed=copy.copy(trimer);changed.trimer_pos=trimer.trimer_pos[:,[1,2,0]]+torch.tensor([3.,2.,1.])
    d,t=prepare_gp_sample(top,changed,s,seed=42,key='toy',position=0,static=static,view='clean',ratio=None)
    original,orig=prepare_gp_sample(top,trimer,s,seed=42,key='toy',position=0,static=static,view='clean',ratio=None)
    for k in ('bond_distance','line_angle','mcl_edge_distance','mcl_trajectory'):
        torch.testing.assert_close(getattr(d,k),getattr(original,k),rtol=2e-5,atol=2e-5)
    # Validity gates must zero geometry after affine layers.
    fake=batch.clone();fake.geometry_valid[:]=False;fake.mcl_readout_valid[:]=False
    with torch.no_grad():assert not gp.fuse(gp.encode(fake))[:,512:].any()

def test_scope_freeze_and_actual_f1_groups(tmp_path):
    from src.training.gp_sota_contract import validate_ft_scope,ft_recipe,PLAN,REV
    from scripts.finetune_glt_mcl_ph import GPArm,optimizer_for
    key='F0/GP/xc/fold0/seed42'
    with pytest.raises(ValueError):validate_ft_scope('formal','xc',0,'GP','F0',dict(allowed_units=[key]))
    with pytest.raises(ValueError):validate_ft_scope('development','xc',2,'GP','F0',dict(plan=PLAN,revision=REV,allowed_units=[key]))
    with pytest.raises(ValueError):validate_ft_scope('smoke','ei',0,'GP','F0',{})
    assert validate_ft_scope('formal','xc',0,'GP','F0',dict(plan=PLAN,revision=REV,status='FROZEN',allowed_units=[key]))==key
    model=GPArm('GP');o,e=optimizer_for(model,'xc',head_only=False,recipe='F1')
    assert [g['lr'] for g in o.param_groups]==[6e-5,2e-4,1e-3]
    from scripts.finetune_mcl_ph import periodic_tdl_trainability
    periodic_tdl_trainability(model,head_only=True)
    assert all(not p.requires_grad for p in model.model.router.parameters())
    assert all(p.requires_grad for p in model.model.predictor.parameters())

def test_zero_update_promotion_contract_and_exact_weights(tmp_path,monkeypatch):
    import json
    from scripts.finetune_glt_mcl_ph import validate_promotion
    (tmp_path/'runtime.json').write_text(json.dumps(dict(status='PASS')))
    initial={'weight':torch.ones(1,1)};weights={'parameter':torch.tensor([3.])};contract={'actual_backbone_lr':2e-4}
    m=dict(outer_test='NOT_RUN',contract=contract,best_epoch=2,best_validation_rmse=.3,optimizer_updates=2,
        history=[dict(epoch=1,validation_rmse=.5,training_steps=1),dict(epoch=2,validation_rmse=.3,training_steps=1)])
    saved=dict(contract=contract,head_initial=initial,state_dict=weights,best_epoch=2,best_validation_rmse=.3,scaler_mean=[1.],scaler_scale=[2.],scaler_train_raw=[1.])
    frozen=tmp_path/'frozen_weights.pt';torch.save(dict(state_dict=weights,head_initial=initial,contract=contract),frozen)
    scope=dict(unit_contracts={'key':dict(contract=contract,development_unit=str(tmp_path),best_epoch=2,scaler_mean=[1.],scaler_scale=[2.],scaler_train_raw=[1.],frozen_weights=str(frozen))})
    monkeypatch.setattr(torch.optim.AdamW,'step',lambda *a,**kw:pytest.fail('promotion updated'))
    validate_promotion(tmp_path,m,saved,initial,contract,scope,'key')
    model=nn.Linear(1,1);state=model.state_dict();snapshot={k:v.clone() for k,v in state.items()};model.load_state_dict(snapshot,strict=True)
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in snapshot.items())
    wrong=dict(unit_contracts={'key':dict(contract={'actual_backbone_lr':1e-4},development_unit=str(tmp_path),best_epoch=2)})
    with pytest.raises(ValueError):validate_promotion(tmp_path,m,saved,initial,contract,wrong,'key')
    torch.save(dict(state_dict={'parameter':torch.tensor([4.])},head_initial=initial,contract=contract),frozen)
    with pytest.raises(ValueError):validate_promotion(tmp_path,m,saved,initial,contract,scope,'key')
    torch.save(dict(state_dict=weights,head_initial=initial,contract=contract),frozen)
    (tmp_path/'test_predictions.npz').touch()
    with pytest.raises(ValueError):validate_promotion(tmp_path,m,saved,initial,contract,scope,'key')

def test_synthetic_joint_property_gradient():
    from scripts.finetune_glt_mcl_ph import GPArm
    from scripts.finetune_mcl_ph import periodic_tdl_trainability
    top,trimer,s,static,target=record()
    d,t=prepare_gp_sample(top,trimer,s,seed=42,key='toy',position=0,static=static,view='clean',ratio=None)
    batch,_=gp_collate([(d,t)])
    torch.manual_seed(42);model=GPArm('GP');model.model.router.configure('top2')
    periodic_tdl_trainability(model,head_only=False)
    loss=(model(batch)-torch.tensor([[1.78]])).square().mean();loss.backward()
    for module in (model.model.router,model.model.adaptor,model.model.glt,model.model.experts):
        grads=[p.grad for p in module.parameters() if p.grad is not None]
        assert all(torch.isfinite(g).all() for g in grads) and sum(g.abs().sum() for g in grads)>0


@pytest.mark.parametrize('route', ['GP', 'GU', 'GP_A'])
def test_atom_geometry_objective_without_fp_keeps_optional_alignment(route):
    from src.modules.glt_dual_pretrain import global_objective
    rows=[]
    for position,smiles in enumerate(('*COC*','*CCN*')):
        top,trimer,s,static,target=record(smiles)
        rows.append(prepare_gp_sample(top,trimer,s,seed=42,key=s,position=position,static=static,target=target))
    batch,labels=gp_collate(rows)
    model=GPPretrainer(route=route).eval()
    result=model(batch,labels)
    assert 'fingerprint' not in labels
    assert result['counts'].tolist()==[2,2,0]
    assert result['sums'][2].item()==0 and result['targets'][-1].item()==0
    loss=global_objective(result['sums'],result['counts'],weights=[1.,1.,0.])
    loss=loss+.001*result['balance']
    if route=='GP_A':
        assert result['align_count'].item()==2
        loss=loss+.1*result['align_sum']/result['align_count']
    else:assert result['align_count'].item()==0
    assert torch.isfinite(loss)
    loss.backward()
    assert model.length_head[2].weight.grad.abs().sum()>0
    assert not any(name.startswith('fp_head.') for name in model.state_dict())
    if route=='GP_A':assert model.align_proj2[0].weight.grad.abs().sum()>0
