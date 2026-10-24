"""One final CPU suite for the authorized transfer implementation only."""
import copy
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from src.dataset.canonical_periodic import build_canonical_periodic_topology
from src.dataset.glt_dual import build_glt_2d_sample,glt_2d_collate
from src.modules.ua2d_transfer import (TransferModel,branch,atom_statistics,optimizer,restore,SCHEMA,ARCHITECTURE,ARMS)
from src.training import ua2d_transfer as t

torch.set_num_threads(2)


def batch():
    graph=build_glt_2d_sample(build_canonical_periodic_topology('*CC*',2),'*CC*')
    missing=graph.clone();missing.graph_available=False
    return glt_2d_collate([graph,missing])


def test_cache_mapping_identity_and_incomplete_publication(tmp_path):
    rows=dict(eea=[dict(row=7,key='ab',cache_key='eea:7'),dict(row=9,key='cd',cache_key='eea:9')])
    result=t.validate_rows(rows,2,('eea',));assert result['eea'][9]['key']=='cd'
    bad=copy.deepcopy(rows);bad['eea'][1]['row']=7
    with pytest.raises(ValueError):t.validate_rows(bad,2,('eea',))
    bad=copy.deepcopy(rows);bad['eea'][0]['cache_key']='eea:9'
    with pytest.raises(ValueError):t.validate_rows(bad,2,('eea',))
    with pytest.raises(ValueError):t.validate_rows(rows,5875,('eea',))
    t.write(tmp_path/'metadata.json',dict(status='RUNNING'))
    t.write(tmp_path/'coverage.json',dict(status='RUNNING'))
    with pytest.raises(ValueError):t.cache_metadata(tmp_path)


def test_cache_get_rejects_wrong_key_and_missing_record():
    cache=t.InputCache.__new__(t.InputCache)
    cache.mapping={'eea':{7:dict(row=7,key='ab',cache_key='eea:7')}};cache.memory={}
    with pytest.raises(ValueError):cache.get('eea',7,'wrong')
    class Transaction:
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def get(self,key):return None
    class Environment:
        def begin(self):return Transaction()
    cache.env=Environment()
    with pytest.raises(KeyError):cache.get('eea',7,'ab')


def test_shared_branch_rng_zero_last_and_parameter_count():
    torch.manual_seed(123);before=torch.get_rng_state().clone();a=branch()
    assert torch.equal(before,torch.get_rng_state())
    b=branch();assert all(torch.equal(x,b.state_dict()[n]) for n,x in a.state_dict().items())
    assert sum(p.numel() for p in a.parameters())==132736
    assert torch.count_nonzero(a[-1].weight)==0 and torch.count_nonzero(a[-1].bias)==0


@pytest.mark.parametrize('arm',['R_MEAN','R_STD','R_STD_SP'])
def test_zero_residual_matches_mean_and_gets_gradient(arm):
    data=batch();torch.manual_seed(42);reference=TransferModel('A_SP').eval()
    with torch.no_grad():expected=reference(data);original_feature=reference.features(data)
    del reference
    torch.manual_seed(42);model=TransferModel(arm).eval()
    assert torch.equal(model(data),expected)
    opt,_=optimizer(model,'head');assert all(not p.requires_grad for p in model.o8.parameters())
    model.eval();loss=model(data).square().sum();loss.backward()
    assert model.residual[-1].bias.grad is not None and model.residual[-1].bias.grad.abs().sum()>0
    assert model.residual[-1].weight.grad is not None and torch.isfinite(model.residual[-1].weight.grad).all()
    with torch.no_grad():model.residual[-1].bias.fill_(1.)
    assert torch.equal(model.features(data)[1],original_feature[1]) # unavailable graph residual stays zero


def test_population_std_permutation_repeated_set_and_single_atom():
    torch.manual_seed(3);atoms=torch.randn(6,512);ids=torch.tensor([0,0,0,1,1,1]);m,s=atom_statistics(atoms,ids,2)
    order=torch.tensor([5,2,0,4,1,3]);m2,s2=atom_statistics(atoms[order],ids[order],2)
    m3,s3=atom_statistics(atoms.repeat(2,1),ids.repeat(2),2)
    assert torch.allclose(m,m2,atol=1e-6) and torch.allclose(s,s2,atol=1e-6)
    assert torch.allclose(m,m3,atol=1e-6) and torch.allclose(s,s3,atol=1e-6)
    assert torch.allclose(s[0],torch.sqrt(atoms[:3].var(0,unbiased=False)+1e-6)-.001,atol=1e-6)
    one=torch.randn(1,512,requires_grad=True);mean,std=atom_statistics(one,torch.tensor([0]),1)
    assert torch.count_nonzero(std)==0
    (mean.sum()+std.sum()).backward();assert torch.isfinite(one.grad).all()


def test_fixed_anchor_scope_gradient_and_optimizer_coverage():
    model=TransferModel('R_STD_SP');model.cache_anchor();theta0=model._anchor['norm2.bias'].clone()
    with pytest.raises(ValueError):model.cache_anchor()
    for stage in ('head','joint'):
        opt,_=optimizer(model,stage)
        ids=[id(p) for g in opt.param_groups for p in g['params']]
        assert len(ids)==len(set(ids)) and set(ids)=={id(p) for p in model.parameters() if p.requires_grad}
        assert next(g for g in opt.param_groups if g['name']=='head_and_residual')['lr']==(.0003 if stage=='head' else .001)
        if stage=='head':assert model.anchor_loss().item()==0
    with torch.no_grad():model.norm2.bias.add_(.25)
    value=model.anchor_loss();assert value.item()==pytest.approx(.5e-3*512*.25**2,rel=1e-6)
    value.backward()
    assert torch.allclose(model.norm2.bias.grad,torch.full((512,),.001*.25))
    assert all(p.grad is None for p in model.head.parameters()) and all(p.grad is None for p in model.residual.parameters())
    assert all(not x.requires_grad for x in model._anchor.values()) and torch.equal(theta0,model._anchor['norm2.bias'])


@pytest.mark.parametrize('arm',ARMS)
def test_complete_checkpoint_restore_is_strict(arm,tmp_path):
    model=TransferModel(arm).eval();cfg=t.config(arm,'eea',0)
    payload=dict(schema=SCHEMA,architecture=ARCHITECTURE,config=cfg,state_dict=model.state_dict(),best_stage='head',best_epoch=4)
    path=tmp_path/'best.pt';torch.save(payload,path);saved=torch.load(path,map_location='cpu',weights_only=False)
    rebuilt=restore(saved,cfg).eval();data=batch()
    with torch.no_grad():assert torch.equal(model(data),rebuilt(data))
    with pytest.raises(ValueError):restore(dict(saved,schema='wrong'),cfg)


def test_validation_nomination_and_fixed_four_arm_freeze(tmp_path,monkeypatch):
    monkeypatch.setattr(t,'OUT',tmp_path);monkeypatch.setattr(t,'SPLIT',tmp_path/'splits');monkeypatch.setattr(t,'TASKS',('eea',))
    (tmp_path/'splits').mkdir();t.write(tmp_path/'splits/eea.json',dict(folds=[dict(test_indices=[i]) for i in range(5)]))
    values={a:dict(macro=.1,units={f'{a}/eea/{f}':dict(best=f'{a}-{f}') for f in range(5)}) for a in ARMS}
    assert t.nominate(values)=='R_MEAN' # no performance admission gate
    with pytest.raises(ValueError):t.nominate({a:values[a] for a in ARMS[1:]})
    frozen=t.freeze_final(values);assert frozen['models']==list(ARMS) and frozen['FINAL']=='R_MEAN'
    assert len(frozen['units'])==20 and 'eea' in frozen['splits']
    with pytest.raises(FileExistsError):t.freeze_final(values)


def test_open_epoch_is_conservatively_charged_without_double_count():
    state=dict(epochs_started=0,validation_started=0,attempted_updates=0,completed_updates=0,completed_epochs=0)
    t.begin_epoch_reservation(state,10);persisted=copy.deepcopy(state)
    charged=t.conservative_ft_counts(persisted)
    assert charged['charged']['attempted_updates']==10 and charged['known_actual_lower_bound']['attempted_updates']==0
    assert charged['charged']['validation_started']==1
    state.update(attempted_updates=4,completed_updates=4)
    assert t.conservative_ft_counts(state)['charged']['attempted_updates']==10
    with pytest.raises(ValueError):t.close_epoch_reservation(state)
    state.update(attempted_updates=10,completed_updates=10,validation_started=1,validation_completed=1,completed_epochs=1)
    t.close_epoch_reservation(state);final=t.conservative_ft_counts(state)
    assert final['charged']['attempted_updates']==10 and not final['open_epoch']
    assert all(v==0 for v in final['uncertain'].values())
