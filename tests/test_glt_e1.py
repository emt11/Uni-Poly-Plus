"""E1's sole CPU verification: graph isolation, gradients and cache lifecycle."""
import copy,math,os
from pathlib import Path
from types import ModuleType
import pytest,torch
from torch import nn
from torch.utils.data import Dataset,DataLoader,Subset
from torch_geometric.data import Data,Batch
from src.modules.glt_dual_opt import BondTokenFusion
from src.modules.glt_adaptation import HeadFeatureCache

@pytest.fixture(scope='module',autouse=True)
def cpu():
    assert os.environ['CUDA_VISIBLE_DEVICES']==''
    torch.set_num_threads(16)

def inputs(device='cpu'):
    gen=torch.Generator().manual_seed(731)
    atoms=torch.randn(9,512,generator=gen).to(device)
    bonds=torch.randn(12,512,generator=gen).to(device)
    ag=torch.tensor([0,1,0,2,1,0,3,3,1],device=device)
    bg=torch.tensor([1,0,1,2,0,1,0,2,1,0,2,1],device=device)
    valid=torch.tensor([True,True,False,True],device=device)
    inc=torch.tensor([[1,4,6,9,0,2,5,8,11,1],[0,2,5,0,1,4,8,1,4,0]],device=device)
    return atoms,bonds,ag,bg,valid,inc

def legacy():
    path=Path('results/glt_optimization_20261011/r10/E1/before/src/modules/glt_dual_opt.py')
    module=ModuleType('legacy_e1');exec(compile(path.read_text(),str(path),'exec'),module.__dict__)
    return module.BondTokenFusion

@pytest.mark.parametrize('mode,beta,gate',[('global',0,'none'),('incident_only',0,'none'),('incident_bias',math.log(2),'none'),('global',0,'incident')])
def test_bucket_matches_original_outputs_attention_and_all_gradients(mode,beta,gate):
    torch.manual_seed(712)
    reference=legacy()(gate,attention_mode=mode,incident_bias=beta)
    optimized=BondTokenFusion(gate,attention_mode=mode,incident_bias=beta)
    optimized.load_state_dict(reference.state_dict(),strict=True);optimized.set_implementation('bucket')
    assert set(reference.state_dict())==set(optimized.state_dict())
    args=inputs();results=[];gradients=[]
    for module in (reference,optimized):
        a=args[0].clone().requires_grad_();b=args[1].clone().requires_grad_()
        output,attention=module(a,b,*args[2:5],incidence=args[5],return_attention=True)
        output.square().mean().backward()
        results.append((output.detach(),attention))
        gradients.append([a.grad,b.grad,*[p.grad for p in module.parameters()]])
    torch.testing.assert_close(results[0][0],results[1][0],atol=1e-5,rtol=1e-4)
    assert len(results[0][1])==len(results[1][1])
    for old,new in zip(results[0][1],results[1][1]):
        assert old[0]==new[0] and torch.equal(old[1],new[1]) and torch.equal(old[2],new[2])
        torch.testing.assert_close(old[3],new[3],atol=1e-5,rtol=1e-4)
    for old,new in zip(*gradients):
        assert (old is None)==(new is None)
        if old is not None:torch.testing.assert_close(old,new,atol=1e-5,rtol=1e-4)
    assert torch.equal(results[1][0][args[2]>=2],args[0][args[2]>=2])

def test_empty_incidence_and_output_bias_do_not_enable_updates():
    module=BondTokenFusion(attention_mode='incident_only');module.set_implementation('bucket')
    module.output.bias.data.fill_(4)
    args=inputs();empty=torch.empty((2,0),dtype=torch.long)
    result,attention=module(*args[:5],incidence=empty,return_attention=True)
    assert torch.equal(result,args[0]) and all(not row[3].any() for row in attention)
    # A malformed or cross-graph connection still fails explicitly.
    with pytest.raises(ValueError):module(*args[:5],incidence=torch.tensor([[0],[0]]))
    with pytest.raises(ValueError):module(*args[:5],incidence=args[5].int())
    with pytest.raises(ValueError):module.set_implementation('unknown')

def test_attention_can_be_omitted_and_input_states_are_not_mutated():
    module=BondTokenFusion();module.set_implementation('bucket');args=inputs()
    before=[v.clone() for v in args]
    plain=module(*args[:5],incidence=args[5]);observed,weights=module(*args[:5],incidence=args[5],return_attention=True)
    assert torch.equal(plain,observed) and len(weights)==2
    assert all(torch.equal(a,b) for a,b in zip(args,before))

def test_dual_readout_can_skip_attention_without_changing_features():
    from types import SimpleNamespace
    from src.modules.glt_dual import DualGLTModel
    args=inputs();fusion=BondTokenFusion();fusion.set_implementation('bucket')
    holder=SimpleNamespace(fusion_revision='glt_fusion_r1',fusion_mode='token',
        token_fusion=fusion,norm2=nn.LayerNorm(512),norm3=nn.LayerNorm(512))
    encoded=dict(atom_states=args[0],token_states=args[1],canonical_graph_index=args[2],
        bond_graph_index=args[3],geometry_valid=args[4],incidence_index=args[5],
        graph_3d=torch.randn(4,512))
    observed=dict(encoded);old=DualGLTModel.fuse(holder,observed)
    holder.collect_fusion_attention=False
    plain=dict(encoded);new=DualGLTModel.fuse(holder,plain)
    assert torch.equal(old,new) and 'local_attention' in observed and 'local_attention' not in plain

class Records(Dataset):
    def __init__(self):
        self.values=torch.randn(8,512,generator=torch.Generator().manual_seed(1));self.targets=torch.arange(8).float()
    def __len__(self):return 8
    def __getitem__(self,i):return Data(e1_input=self.values[i:i+1],y=self.targets[i:i+1])

class Model(nn.Module):
    def __init__(self,width):
        super().__init__();self.encoder=nn.Linear(512,width);self.head=nn.Sequential(nn.Dropout(.1),nn.Linear(width,1));self.calls=0
    def features(self,data):self.calls+=1;return self.encoder(data.e1_input)
    def forward(self,data):return self.head(self.features(data))

@pytest.mark.parametrize('width',[512,1024])
def test_cache_preserves_rng_sampler_head_gradients_and_reads_current_labels(width):
    model=Model(width).eval();model.encoder.requires_grad_(False)
    dataset=Records();context=dict(parent='saved-parent',configuration={'mode':'global'},input_view='clean',dtype='fp32',sample_indices=list(range(8)))
    model.feature_cache_context=context;cache=HeadFeatureCache(model,dataset,context)
    state=torch.get_rng_state().clone();cache.build('cpu',4,Batch.from_data_list)
    assert torch.equal(state,torch.get_rng_state()) and model.calls==2
    gen=torch.Generator().manual_seed(42)
    online=DataLoader(Subset(dataset,[0,2,4,6]),batch_size=2,shuffle=True,generator=gen,collate_fn=Batch.from_data_list)
    sampler_state=gen.get_state().clone();original=list(online);gen.set_state(sampler_state)
    cached=list(cache.loader(online));assert model.calls==2
    for old,new in zip(original,cached):
        assert torch.equal(old.y,new.y)
        torch.testing.assert_close(model.encoder(old.e1_input),new.cached_features,atol=1e-5,rtol=1e-4)
    model.head.train();state=torch.get_rng_state().clone()
    prediction=model.head(model.encoder(original[0].e1_input));prediction.square().mean().backward()
    gradients=[p.grad.clone() for p in model.head.parameters()];model.head.zero_grad(set_to_none=True)
    torch.set_rng_state(state);model.head(cached[0].cached_features).square().mean().backward()
    for a,b in zip(gradients,[p.grad for p in model.head.parameters()]):torch.testing.assert_close(a,b,atol=1e-5,rtol=1e-4)
    dataset.targets[0]=99
    validation=DataLoader(Subset(dataset,[0]),batch_size=1,collate_fn=Batch.from_data_list)
    assert next(iter(cache.loader(validation))).y.item()==99
    with pytest.raises(ValueError):cache.validate(dict(context,parent='another-parent'))
    model.encoder.requires_grad_(True)
    with pytest.raises(ValueError):cache.validate(context)
    model.encoder.requires_grad_(False);cache.invalidate()
    with pytest.raises(ValueError):next(iter(cache.loader(validation)))

def test_cache_rejects_incomplete_binding():
    with pytest.raises(ValueError):HeadFeatureCache(Model(512),Records(),{'parent':'only'})


@pytest.mark.parametrize('column_labels',[False,True])
def test_cache_preserves_actual_label_shape_mse_and_head_gradient(column_labels):
    model=Model(512).eval();model.encoder.requires_grad_(False);ds=Records()
    def collate(items):
        data=Batch.from_data_list(items)
        if column_labels:data.y=torch.stack([item.y for item in items])
        return data
    context=dict(parent='same',configuration={},input_view='clean',dtype='fp32',sample_indices=list(range(8)))
    model.feature_cache_context=context;cache=HeadFeatureCache(model,ds,context);cache.build('cpu',4,collate)
    loader=DataLoader(Subset(ds,[0,2,4]),batch_size=3,collate_fn=collate)
    online=next(iter(loader));cached=next(iter(cache.loader(loader)))
    assert online.y.shape==cached.y.shape and torch.equal(online.y,cached.y)
    state=torch.get_rng_state().clone();model.head.train()
    a=nn.MSELoss()(model.head(model.features(online)),online.y);a.backward()
    gradients=[p.grad.clone() for p in model.head.parameters()];model.head.zero_grad(set_to_none=True)
    torch.set_rng_state(state);b=nn.MSELoss()(model.head(cached.cached_features),cached.y);b.backward()
    torch.testing.assert_close(a,b,atol=1e-5,rtol=1e-4)
    for old,new in zip(gradients,[p.grad for p in model.head.parameters()]):torch.testing.assert_close(old,new,atol=1e-5,rtol=1e-4)
