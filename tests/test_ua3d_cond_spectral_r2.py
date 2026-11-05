"""Sole final CPU suite; never run before complete formal TEST."""
import json
import numpy as np
import torch
from src.modules import ua3d_cond_spectral_r2 as m
from src.training import ua3d_cond_spectral_r2 as t

def test_formal_completion_and_budget():
    budget=t.read(t.OUT/'budget.json')
    assert budget['test_complete']==210 and budget['test_rows']==35250
    assert budget['pt_completed']==10000 and budget['ft_complete']==210 and budget['ft_completed_updates']==336000
    frozen=t.read(t.OUT/'selection/final_frozen.json')
    assert frozen['frozen'] and len(frozen['units'])==210
    for arm in m.ARMS:
        met=t.aggregate(arm,True)
        assert len(met['units'])==35
        for task in m.TASKS:
            for fold in range(5):
                path=t.unit_path(arm,task,fold,True)
                with np.load(path/'test_predictions.npz',allow_pickle=False) as p:
                    rows=p['indices'];keys=p['sample_keys']
                    assert len(rows)==len(set(rows.tolist()))==len(keys)
                    assert rows.tolist()==sorted(t.read(t.SPLIT/f'{task}.json')['folds'][fold]['test_indices'])
                    assert np.isfinite(p['y_true']).all() and np.isfinite(p['y_pred']).all()

def fixture_surface():
    # Orthonormal eigenbasis for a small discrete heat operator, one repeated root.
    torch.manual_seed(9);Q,_=torch.linalg.qr(torch.randn(6,6));lam=torch.tensor([0.,.2,.2,.7,1.2,2.])
    F=torch.randn(6,16);w=torch.rand(6);C=Q.T@F;a=Q.T@w
    return Q,F,w,dict(lam=lam[None],C=C[None],a=a[None],A=torch.tensor([6.]),Ac=w.sum()[None],stat=torch.randn(1,16),mode_mask=torch.ones(1,6,dtype=torch.bool),mask=torch.ones(1))

def test_compact_heat_and_invariance():
    Q,F,w,s=fixture_surface();model=m.Surface('S_COND_GATE').eval();z=[torch.randn(1,256) for _ in range(3)];avail=torch.ones(1,3,dtype=torch.bool)
    with torch.no_grad():
        context=torch.cat((*[n(v) for n,v in zip(model.norms,z)],avail.float()),-1)
        times=.0625+(64.-.0625)*torch.sigmoid(model.beta+model.condition(context))
        B=torch.exp(-s['lam'][:,:,None]*times[:,None,:])*model.W(s['C'])
        explicit=torch.stack([Q@torch.diag(torch.exp(-s['lam'][0]*times[0,c]))@Q.T@(F@model.W.weight[c]) for c in range(64)],-1)
        assert torch.allclose((Q@B[0]),explicit,atol=2e-5,rtol=2e-5)
        assert torch.allclose((s['a'][:,:,None]*B).sum(1)/s['Ac'][:,None],(explicit*w[:,None]).sum(0)[None]/w.sum(),atol=2e-5,rtol=2e-5)
        out=model(z,avail,s)
        rot=torch.eye(6);angle=.7;rot[1:3,1:3]=torch.tensor([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
        rot[4,4]=-1
        other=dict(s,C=(rot.T@s['C'][0])[None],a=(rot.T@s['a'][0])[None])
        assert torch.allclose(out,model(z,avail,other),atol=2e-5,rtol=2e-5)

def test_time_gradient_and_missing_surface():
    _,_,_,s=fixture_surface();model=m.Surface('S_COND_GATE').train();z=[torch.randn(1,256,requires_grad=True) for _ in range(3)];avail=torch.tensor([[True,False,True]])
    # Output layer starts zero by contract; nonzero weights here test reachable
    # conditioning -> BASE gradient after training has begun.
    with torch.no_grad():model.condition[-1].weight.normal_(0,.01)
    loss=model(z,avail,s).square().sum();loss.backward()
    assert model.time_tensor.grad is not None and torch.isfinite(model.time_tensor.grad).all()
    assert z[0].grad is not None and z[0].grad.abs().sum()>0
    assert z[1].grad.abs().sum()==0
    s['mask'].zero_();s['C'].fill_(float('nan'));s['a'].fill_(float('nan'));s['stat'].fill_(float('nan'))
    assert torch.equal(model(z,avail,s),torch.zeros(1,256))

def test_production_metrics():
    result=t.metrics(np.array([1.,2.,3.]),np.array([1.,2.,3.]))
    assert result['r2']==1 and result['rmse']==0
