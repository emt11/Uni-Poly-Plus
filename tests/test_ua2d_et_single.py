"""Only final ET_SINGLE suite; no campaign dispatch, training or formal predictions."""
import copy
import pytest
import torch
from src.modules.ua2d_et_single import (AtomicET,ETPretrainer,initialize_common,deployment_package,
    restore_downstream,restore_ft,restore_pretrain,FT_SCHEMA,ARCHITECTURE)
from src.dataset.ua2d_et_single import (atomic_fields,prepare_et_sample,collate_et,endpoint_corruption,neighborhood)
from src.training.ua2d_et_single import COMMON,HEADS,development_gate
from src.modules.ua2d_galformer_pretrain import PretrainConfig,task_sum,objective


def fixture(text='*COC*',key=b'fixture'):
    from src.dataset.canonical_periodic import build_canonical_periodic_topology
    from tests.test_dual_glt import _toy_pair
    from src.dataset.glt_dual_static import build_dual_static
    from src.dataset.ua2d_fast_pretrain import build_template
    top=build_canonical_periodic_topology(text);_,tri=_toy_pair(text)
    static=build_dual_static(top,tri,text)
    template=build_template(top,tri,text,static);template['sample_key']=key
    fields=atomic_fields(top,tri,17,key)
    return template,fields


def test_equivariance_invariance_and_no_cutoff_truncation():
    torch.manual_seed(7);model=AtomicET().eval()
    pos=torch.tensor([[0.,0.,0.],[1.2,0.,0.],[0.,1.1,.4],[5.,0.,0.]])
    chemical=torch.tensor([[0,1,1,0,0,3,3,0],[1,0,0,1,3,0,0,3]])
    edge,mask=neighborhood(pos,chemical)
    assert ((edge[0]==0)&(edge[1]==3)).any()  # true long chemical bond
    assert len(edge[0])==len(set(map(tuple,edge.t().tolist())))
    z=torch.tensor([5,7,101,8]);rot=torch.tensor([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])
    with torch.no_grad():
        s,v=model(z,pos,edge,mask);ss,vv=model(z,pos@rot+3.,edge,mask)
    torch.testing.assert_close(s,ss,atol=4e-5,rtol=4e-5)
    torch.testing.assert_close(torch.einsum('nac,ab->nbc',v,rot),vv,atol=4e-5,rtol=4e-5)
    # Radius graph does not inherit a 32-neighbor cap.
    dense=torch.zeros(40,3);e,_=neighborhood(dense,torch.empty(2,0,dtype=torch.long))
    assert e.shape[1]==1600


def test_endpoint_priority_and_periodic_random_synchronization():
    fields=dict(canonical=torch.tensor([0,1,0,1]),canonical_count=2,z=torch.tensor([6,8,6,8]))
    bond=torch.tensor([[0,2],[1,3]])
    z,p=endpoint_corruption(fields,bond,torch.tensor([2,1]),torch.Generator().manual_seed(2))
    assert p.tolist()==[2,2] and z.tolist()==[101]*4
    z,p=endpoint_corruption(fields,bond,torch.tensor([2,0]),torch.Generator().manual_seed(2))
    assert z[0]==z[2] and z[1]==z[3]


def test_old_2d_rng_and_atomic_batch_index_isolation():
    from src.dataset.ua2d_fast_pretrain import prepare_fast_sample
    config=PretrainConfig();a,fa=fixture();b,fb=fixture('*CCN*',b'fixture_b')
    x,y=prepare_et_sample(a,fa,config=config,position=12)
    old,labels=prepare_fast_sample(a,config=config,seed=42,position=12)
    assert torch.equal(x.mips_x,old.mips_x)
    assert torch.equal(y['atom_selected'],labels['atom_selected'])
    assert torch.equal(y['physical_pos'],labels['physical_pos'])
    second=prepare_et_sample(b,fb,config=config,position=13)
    data,target=collate_et([(x,y),second])
    source,destination=target['et_edge'];g=target['et_graph']
    assert torch.equal(g[source],g[destination])
    left,right=target['et_bond'];assert torch.equal(g[left],g[right])
    assert target['et_z'].max()<=101
    bad=copy.deepcopy(fa);bad['sample_key']=b'wrong'
    with pytest.raises(ValueError): prepare_et_sample(a,bad,config=config,position=12)


def test_common_initialization_symmetric_decoder_gradient_and_global_reduction():
    model=ETPretrainer()
    common=torch.load(COMMON,map_location='cpu',weights_only=False,mmap=True)
    copied=initialize_common(model,common)
    assert copied and all(torch.equal(model.state_dict()[k],common['state_dict'][k]) for k in copied)
    h=torch.randn(3,256,requires_grad=True);left=torch.tensor([0,1]);right=torch.tensor([1,2]);distance=torch.tensor([1.2,1.4])
    logits=model.decode(h,left,right,distance)
    torch.testing.assert_close(logits,model.decode(h,right,left,distance),rtol=0,atol=0)
    logits[:,:4].square().mean().backward()
    assert h.grad.abs().sum()>0 and model.bond_head[0].weight.grad.abs().sum()>0
    value,count=task_sum(torch.tensor([1.,3.,10.]),torch.tensor([0,0,1]),2,torch.tensor([1.,1.,.5]))
    assert value==12. and count==2.
    result=dict(sums={'a':torch.tensor(12.,requires_grad=True)},counts={'a':torch.tensor(2.)})
    assert objective(result,{'a':1.},{'a':torch.tensor(4.)})==3.
    # The scalar/vector encoder receives supervised gradients on a tiny graph.
    model.zero_grad(set_to_none=True)
    pos=torch.tensor([[0.,0.,0.],[1.,.2,0.],[2.,0.,.3]])
    edge,chem=neighborhood(pos,torch.tensor([[0,1,1,2],[1,0,2,1]]))
    s,v=model.et(torch.tensor([5,101,7]),pos,edge,chem)
    model.decode(s,left,right,distance)[:,:5].square().mean().backward()
    assert model.et.layers[0].q_proj.weight.grad.abs().sum()>0
    assert model.et.embedding.weight.grad.abs().sum()>0


def test_complete_pretrain_and_pure2d_restore(tmp_path):
    model=ETPretrainer();initialize_common(model,torch.load(COMMON,map_location='cpu',weights_only=False,mmap=True))
    package=deployment_package(model,5000)  # synthetic serialization fixture, not a trained result
    path=tmp_path/'fixture.pt';torch.save(package,path);loaded=torch.load(path,map_location='cpu',weights_only=False)
    recovered=restore_pretrain(loaded)
    with pytest.raises(ValueError):restore_pretrain(dict(loaded,schema='wrong'))
    assert all(torch.equal(v,recovered.state_dict()[k]) for k,v in model.state_dict().items())
    head=torch.load(HEADS/'eea/fold0.pt',map_location='cpu',weights_only=False)
    pure=restore_downstream(loaded,head).eval()
    assert not hasattr(pure,'et') and not hasattr(pure,'encoder3d')
    from src.dataset.canonical_periodic import build_canonical_periodic_topology
    top=build_canonical_periodic_topology('*COC*')
    with torch.no_grad(): expected=pure.forward_topology(top,'*COC*')
    cfg=dict(arm='ET_SINGLE');saved=dict(schema=FT_SCHEMA,config=cfg,state_dict=pure.state_dict())
    restored=restore_ft(saved,cfg).eval()
    with torch.no_grad(): torch.testing.assert_close(restored.forward_topology(top,'*COC*'),expected,rtol=0,atol=0)
    with pytest.raises(ValueError):restore_ft(dict(saved,schema='old'),cfg)


def test_gate_and_no_test_before_freeze(tmp_path,monkeypatch):
    assert not development_gate(.5099999,.5)['passed']
    assert development_gate(.5100001,.5)['passed']
    from scripts import run_ua2d_et_single as entry
    monkeypatch.setattr(entry,'OUT',tmp_path);(tmp_path/'selection').mkdir()
    decision=tmp_path/'selection/development_decision.json'
    decision.write_text('{"passed":false}')
    controller=entry.Controller.__new__(entry.Controller);controller.jobs=[];controller.refresh=lambda:dict(test_starts=0)
    with pytest.raises(RuntimeError,match='forbidden'):controller.launch('never','test',[])
    decision.write_text('{"passed":true}')
    with pytest.raises(RuntimeError,match='forbidden'):controller.launch('unfrozen','test',[])
    assert controller.jobs==[]
