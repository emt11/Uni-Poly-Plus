"""Sole final CPU suite: physical geometry, residuals, masking and frozen policy."""
from types import SimpleNamespace
import copy
import math
import numpy as np
import pytest
import torch
from torch_geometric.data import Data
from src.dataset import ua3d_spatial as d
from src.modules import ua3d_spatial as m
from src.training import ua3d_spatial as t
from scripts import run_ua3d_spatial as c

torch.set_num_threads(2)


def fixture():
    # Two disconnected three-atom heavy fragments in one finite physical chain view.
    p=torch.tensor([[0.,0.,0.],[1.,0.,0.],[2.,0.,0.],[0.,2.,0.],[1.,2.,.5],[2.,2.,0.]])
    bonds=torch.tensor([[0,1,3,4],[1,2,4,5]])
    top=d.physical_topology(bonds,torch.tensor([6,6,6,6]),torch.tensor([6,6,6,6]),6,torch.full((6,),6),torch.tensor([-1,-1,0,0,1,1]))
    return p,top


def graph(p,top):
    return d.attach(Data(geometry_valid=torch.tensor([True])),d.spatial_fields(p,top))


def test_all_spatial_edges_are_physical_directed_radius_and_hop_filtered():
    p,top=fixture();g=graph(p,top);edges={tuple(e) for e in g.sp_edges.T.tolist()}
    assert len(edges)==18  # 3x3 disconnected contacts, both directions
    assert (0,2) not in edges and (2,0) not in edges
    assert all((b,a) in edges and a!=b for a,b in edges)
    assert torch.all(g.sp_distance>1e-8) and torch.all(g.sp_distance<=4.5)
    for a,b in edges:assert not top['excluded'][a,b]
    assert g.sp_bond_atoms.tolist()==top['bonds'].tolist()
    bad=p.clone();bad[5]=bad[0]
    with pytest.raises(ValueError,match='near-zero'):graph(bad,top)
    bad=p.clone();bad[5,0]=float('nan')
    with pytest.raises(FloatingPointError):graph(bad,top)
    with pytest.raises(ValueError,match='element identity'):d.physical_topology(torch.tensor([[0,0],[1,2]]),torch.tensor([6,7]),torch.tensor([6,6]),3)


def test_angular_moments_match_explicit_three_atom_angles_and_validity():
    p,top=fixture();g=graph(p,top)
    for i in range(6):
        ids=g.sp_edges[0,g.sp_edges[1]==i];u=p[ids]-p[i];u=u/u.norm(dim=-1,keepdim=True)
        values=torch.stack([u[a]@u[b] for a in range(len(u)) for b in range(a+1,len(u))])
        expected=torch.stack((values.mean(),((3*values.square()-1)/2).mean()))
        assert torch.allclose(g.sp_angle[i],expected,atol=1e-6)
    far=p.clone();far[3:]+=100;empty=graph(far,top)
    assert empty.sp_edges.shape==(2,0) and not empty.sp_angle_valid.any() and torch.count_nonzero(empty.sp_angle)==0


def test_rotation_translation_and_physical_instance_permutation_equivariance():
    p,top=fixture();g=graph(p,top)
    q,_=torch.linalg.qr(torch.tensor([[1.,2.,3.],[2.,-1.,1.],[1.,1.,-2.]]));other=graph(p@q+torch.tensor([3.,-4.,7.]),top)
    assert torch.equal(g.sp_edges,other.sp_edges)
    assert torch.allclose(g.sp_distance,other.sp_distance,atol=2e-6)
    assert torch.allclose(g.sp_angle,other.sp_angle,atol=2e-6)
    permutation=torch.tensor([3,0,5,1,4,2]);inverse=torch.argsort(permutation)
    bonds=inverse[top['bonds']];pt=d.physical_topology(bonds,torch.full((4,),6),torch.full((4,),6),6)
    pg=graph(p[permutation],pt);mapped=permutation[pg.sp_edges]
    assert {tuple(x) for x in mapped.T.tolist()}=={tuple(x) for x in g.sp_edges.T.tolist()}
    assert torch.allclose(pg.sp_angle,g.sp_angle[permutation],atol=1e-6)
    module=m.SpatialResidual('S_RADANG');torch.nn.init.normal_(module.up.weight,std=.01)
    state=torch.randn(4,512)
    assert torch.allclose(module(state,g),module(state,pg),atol=2e-6)


def test_batch_offsets_do_not_join_different_physical_graphs():
    p,top=fixture();a=graph(p,top);b=graph(p+1.,top)
    batch=d.append_batch(Data(geometry_valid=torch.tensor([True,True])),[a,b])
    assert batch.sp_bond_atoms[:,:4].max()<6 and batch.sp_bond_atoms[:,4:].min()>=6
    assert torch.equal(batch.sp_graph[batch.sp_edges[0]],batch.sp_graph[batch.sp_edges[1]])
    module=m.SpatialResidual('S_RAD');torch.nn.init.normal_(module.up.weight,std=.02)
    x,y=torch.randn(4,512),torch.randn(4,512)
    assert torch.allclose(module(torch.cat((x,y)),batch),torch.cat((module(x,a),module(y,b))),atol=1e-6)


def test_identical_capacity_isolated_rng_zero_initial_and_no_neighbor_bias():
    before=torch.random.get_rng_state().clone();modules=[m.SpatialResidual(a) for a in m.ARMS[1:]]
    assert torch.equal(before,torch.random.get_rng_state())
    for module in modules:
        assert sum(v.numel() for v in module.parameters())==37632
        assert all(torch.equal(v,modules[0].state_dict()[k]) for k,v in module.state_dict().items())
    for module in modules:
        p,top=fixture();g=graph(p,top);x=torch.randn(4,512,requires_grad=True)
        output=module(x,g);assert torch.count_nonzero(output)==0
        output.sum().backward();assert module.up.weight.grad is not None
        p[3:]+=100;empty=graph(p,top);module.up.bias.data.fill_(5.)
        assert torch.count_nonzero(module(x,empty))==0
    # Mixed graph with an isolated atom verifies masking after biased Wup.
    g=graph(*fixture());g.sp_degree[0]=0;g.sp_edges=g.sp_edges[:,g.sp_edges[1]!=0]
    keep=g.sp_edges.shape[1];g.sp_distance=g.sp_distance[:keep]  # constant arm ignores numeric distances
    mod=m.SpatialResidual('C_CONST');mod.up.bias.data.fill_(12.)
    result=mod(torch.zeros(4,512),g)
    assert torch.allclose(result[0],torch.ones(512))  # endpoint0=0, endpoint1=12, divided by12


def test_distance_and_angle_channel_interventions():
    p,top=fixture();g=graph(p,top);changed=copy.deepcopy(g);changed.sp_distance=changed.sp_distance*.9;changed.sp_angle=changed.sp_angle+.2
    x=torch.randn(4,512)
    for arm in m.ARMS[1:]:
        module=m.SpatialResidual(arm);torch.nn.init.normal_(module.up.weight,std=.01)
        base=module(x,g);angles=copy.deepcopy(g);angles.sp_angle=changed.sp_angle
        distances=copy.deepcopy(g);distances.sp_distance=changed.sp_distance
        assert torch.equal(base,module(x,angles))==(arm!='S_RADANG')
        assert torch.equal(base,module(x,distances))==(arm=='C_CONST')


def test_pt_uses_already_noisy_physical_positions_and_preserves_tokens(monkeypatch):
    from contextlib import nullcontext
    from src.dataset.ua2d_fast_pretrain import decode_batch
    p,top=fixture();noisy=p.clone();noisy[3,2]+=.03
    class Counts(list):
        def get_lock(self):return nullcontext()
    ds=d.PTData('.', '.',Counts([0,0,0,0]));ds.environment=object()
    ds.stream=SimpleNamespace(index_at=lambda position:0)
    ds.eligible=ds.keys=ds.rows=np.array([0]);ds.selected_keys=np.array([b'key']);ds.selected_rows=np.array([0])
    clean=Data(geometry_valid=True,bond_z_a=torch.full((4,),6),bond_z_b=torch.full((4,),6))
    template=dict(sample_key=b'key',clean=clean,positions=p,endpoint_a=top['bonds'][0],endpoint_b=top['bonds'][1],fp_bits=np.concatenate((np.array([1],dtype=np.uint8),np.zeros(255,dtype=np.uint8))))
    ds.templates[b'key']=template;seen=[]
    monkeypatch.setattr(d,'resolve_slot',lambda *args:(0,b'key',0))
    def prepare(template,**kwargs):
        assert kwargs['seed']==42
        data=Data(bond_distance=torch.ones(4),corrupted=torch.full((4,512),-99.))
        return data,dict(physical_pos=noisy,identity='same')
    monkeypatch.setattr(d,'prepare_arm_sample',prepare)
    real=d.spatial_fields
    def observe(positions,topology):
        assert torch.equal(positions,noisy);seen.append(1)
        return real(positions,topology)
    monkeypatch.setattr(d,'spatial_fields',observe)
    def collate(records):
        assert all(torch.all(x.corrupted==-99.) for x,_ in records)
        return Data(geometry_valid=torch.ones(168,dtype=torch.bool)),dict(identity=['same']*168)
    monkeypatch.setattr(d,'collate_arm_samples',collate)
    batch,labels,trace=decode_batch(ds[0])
    expected=real(noisy,top)
    assert len(seen)==168 and trace['start_position']==0 and ds.prepared_counts[0]==168
    assert torch.equal(batch.sp_distance[:len(expected['sp_distance'])],expected['sp_distance'])
    assert not torch.equal(real(p,top)['sp_distance'],expected['sp_distance'])


def test_native_optimizer_freeze_and_shared_residual_identity():
    model=m.SpatialFuse('S_RADANG',False)
    assert len(model.encoder3d.layers)==6 and sum(p.numel() for p in model.encoder3d.spatial.parameters())==37632
    assert all(layer.attention.scale_denominator==512 for layer in model.encoder3d.layers)
    for stage in ('head','joint'):
        opt,groups=m.optimizer_ft(model,stage);model.train()
        assert model.encoder3d.training==(stage=='joint')
        assert all(p.requires_grad==(stage=='joint') for p in model.encoder3d.spatial.parameters())
        if stage=='joint':
            ids={id(p) for p in opt.param_groups[0]['params']}
            assert all(id(p) in ids for p in model.encoder3d.spatial.parameters()) and opt.param_groups[0]['lr']==1e-4
    with pytest.raises(ValueError):m.validate_package(dict(schema='old'))


def test_missing_projected_z3_exact_zero_and_frozen_encoder_gradients():
    # Exercise Frozen3D.forward without any 2D, FP, or gate module.
    model=m.Frozen3D.__new__(m.Frozen3D);torch.nn.Module.__init__(model)
    model.encoder3d=torch.nn.Linear(1,1);model.encoder3d.forward=lambda data:(None,data,torch.ones(len(data),dtype=torch.bool))
    model.norm3=torch.nn.Identity();model.A3=torch.nn.Linear(512,256);model.A3.bias.data.fill_(100.)
    model.requires_grad_(False);model.eval()
    z=model(torch.randn(2,512),torch.tensor([0,2]),3)
    assert torch.count_nonzero(z[1])==0 and torch.count_nonzero(model(None,torch.empty(0,dtype=torch.long),3))==0
    head=m.property_head(.5);loss=head(z).sum();loss.backward()
    assert all(p.grad is None for p in model.parameters()) and any(p.grad is not None for p in head.parameters())


def test_global_validation_ties_freeze_identity_and_paired_gate():
    values={a:t.summary({task:[.8]*5 for task in t.TASKS}) for a in t.ARMS}
    assert t.nominate(values)=='BASE'
    assert t.nominate({a:values[a] for a in t.ARMS[2:]},t.ARMS[2:])=='S_RAD'
    with pytest.raises(ValueError):t.frozen_unit(dict(frozen=False),'BASE','eea',0)
    comparisons=t.comparisons({'FULL':values,'probe':values},'S_RADANG')
    for mode in comparisons:
        for v in comparisons[mode].values():assert v['delta']==0. and not v['performance_pass']
    pt=c.reservation('pt');failed=dict(kind='pt',reservation=pt,consumption=None,exit=1)
    ledger=c.ledger([failed]);assert ledger['charged_or_reserved']['pt_updates']==10000 and ledger['known_actual_lower_bound']['pt_updates']==0
    assert ledger['known_actual_lower_bound']['pt_starts']==1 and not ledger['budget_violations']
    assert t.ALLOWED_GPUS==(0,1,2,3) and t.LIMITS['test_starts']==280


def test_native_initial_loading_rejects_missing_state_and_installs_only_after_load():
    def make():
        model=torch.nn.Module();model.arm='TMOL3M';model.spatial_arm='S_RAD'
        for name in (*m.NATIVE,*m.native.MASK_HEADS):setattr(model,name,torch.nn.Linear(1,1))
        for name,module in m.native.modal_modules('TMOL3M').items():setattr(model,name,module)
        model.install_spatial=lambda:model.encoder3d.add_module('spatial',m.SpatialResidual('S_RAD'))
        return model
    model=make();source={}
    for name in (*m.NATIVE,*m.native.MASK_HEADS):
        for key,value in getattr(model,name).state_dict().items():source[name+'.'+key]=torch.full_like(value,.25)
    common=dict(schema='ua2d_galformer_objectives_v3',step=0,seed=42,plan='UA2D-PRETRAIN-FP-20261019-01',state_dict=source)
    heads=dict(schema=common['schema'],step=0,strategy='mask_align',state_dict={})
    modal=m.native.modal_modules('TMOL3M').state_dict();evidence=m.load_initial(model,common,heads,modal)
    assert evidence['common_parameters_strict'] and evidence['added_parameters']==37632
    assert torch.equal(model.encoder3d.weight,source['encoder3d.weight'])
    bad=copy.deepcopy(common);bad['state_dict'].pop('o8.weight')
    with pytest.raises(ValueError):m.load_initial(make(),bad,heads,modal)
    with pytest.raises(ValueError,match='before spatial'):m.load_initial(model,common,heads,modal)
