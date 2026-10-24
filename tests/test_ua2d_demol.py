"""The single final CPU suite; no additional training or formal evaluation."""
import copy
from collections import Counter
import pytest
import torch
from torch_geometric.data import Data
from src.dataset.canonical_periodic import build_canonical_periodic_topology
from src.dataset.ua2d_demol import build, attach_bonds, collate
from src.modules.ua2d_demol import BondAdapter, BondModel, initialize, restore, optimizer, ARCHITECTURE, SCHEMA
from src.modules.glt_dual import mean_pool
from src.training.ua2d_demol import PARENT, HEADS, development_gate


def graph(text='*COC*'):
    return build(build_canonical_periodic_topology(text), text)


def test_lifted_neighbor_multiplicity_no_false_image_identification():
    # Two distinct translations of a 0--1 bond share one canonical pair.
    # Each physical bond has exactly two distinct neighbors and itself,
    # rather than being collapsed to a single canonical undirected edge.
    data = Data(lga_spd=torch.ones(2, dtype=torch.long),
        lga_path_index=torch.tensor([[0,1,-1],[0,1,-1]]),
        lga_path_shift=torch.tensor([[0,0,0],[0,1,0]]),
        lga_path_mask=torch.tensor([[1,1,0],[1,1,0]], dtype=torch.bool),
        bond_path_features=torch.ones(2,2,14), bond_path_mask=torch.ones(2,2,dtype=torch.bool))
    attach_bonds(data)
    assert data.demol_keys.tolist() == [[0,1,0],[0,1,1]]
    assert Counter(map(tuple,data.demol_line.t().tolist())) == Counter({(0,0):1,(1,0):2,(0,1):2,(1,1):1})
    assert data.demol_incidence.tolist() == [[0,0,1,1],[0,1,0,1]]
    # Same canonical atom, different images is a real periodic chain bond.
    chain = copy.deepcopy(data)
    chain.lga_spd = torch.ones(1,dtype=torch.long)
    chain.lga_path_index = torch.tensor([[0,0,-1]])
    chain.lga_path_shift = torch.tensor([[0,1,0]])
    chain.lga_path_mask = chain.lga_path_mask[:1]
    chain.bond_path_features = chain.bond_path_features[:1]
    chain.bond_path_mask = chain.bond_path_mask[:1]
    attach_bonds(chain)
    assert chain.demol_keys.tolist() == [[0,0,-1]]
    assert chain.demol_line.shape == (2,3)
    assert chain.demol_incidence[1].tolist() == [0,0]


def test_batch_isolation_and_endpoint_symmetry():
    a,b = graph(),graph('*CCN*')
    batch = collate([a,b]); offset = len(a.mips_x)
    left,right = batch.demol_keys[:,:2].unbind(1)
    assert torch.equal(batch.canonical_graph_index[left],batch.canonical_graph_index[right])
    s,t = batch.demol_line
    g = batch.canonical_graph_index[batch.demol_keys[:,0]]
    assert torch.equal(g[s],g[t])
    adapter = BondAdapter().eval()
    with torch.no_grad(): adapter.output.weight.normal_(0,.01)
    atoms = torch.randn(len(batch.mips_x),512)
    changed = batch.clone(); changed.demol_keys[:,:2] = changed.demol_keys[:,:2].flip(1)
    for arm in ('BOND_SELF','BOND_NEIGHBOR'):
        expected = adapter(atoms,batch,arm)
        torch.testing.assert_close(adapter(atoms,changed,arm),expected)
        separate = torch.cat([adapter(atoms[:offset],collate([a]),arm),adapter(atoms[offset:],collate([b]),arm)])
        torch.testing.assert_close(expected,separate)


@pytest.fixture(scope='module')
def model():
    parent = torch.load(PARENT,map_location='cpu',weights_only=False,mmap=True)
    head = torch.load(HEADS/'eea/fold0.pt',map_location='cpu',weights_only=False)
    m = initialize(parent,head,'BOND_NEIGHBOR').eval()
    for name in ('o8','norm2'):
        for k,v in getattr(m,name).state_dict().items():
            assert torch.equal(v,parent['state_dict'][name+'.'+k])
    return m


def test_zero_initialization_matches_ua_and_stage_gradient_paths(model):
    batch = collate([graph(),graph('*CCN*')])
    with torch.no_grad():
        atoms,_ = model.o8(batch)
        expected = model.head(model.norm2(mean_pool(atoms,batch.canonical_graph_index,2))).view(-1)
        assert model(batch).shape == (2,)
        torch.testing.assert_close(model(batch),expected,rtol=0,atol=0)
    optimizer(model,'head'); model.train()
    assert not model.o8.training and not model.adapter.training
    assert all(not p.requires_grad for p in model.adapter.parameters())
    optimizer(model,'joint'); model.eval()
    model.zero_grad(set_to_none=True)
    model(batch).square().sum().backward()
    assert model.adapter.output.weight.grad.abs().sum() > 0
    with torch.no_grad(): model.adapter.output.weight.normal_(0,.01)
    model.zero_grad(set_to_none=True)
    model(batch).square().sum().backward()
    assert model.adapter.initial[0].weight.grad.abs().sum() > 0
    assert model.adapter.update[0].weight.grad.abs().sum() > 0
    assert model.o8.layers[-1].attention.qkv.weight.grad.abs().sum() > 0
    assert model.head[-1].weight.grad.abs().sum() > 0


def test_complete_restore_and_same_adapter_initialization(model,tmp_path):
    # Nonzero learned adapter must be part of the complete restored prediction.
    batch = collate([graph()]); model.eval()
    cfg = dict(arm='BOND_NEIGHBOR')
    saved = dict(schema=SCHEMA,architecture=ARCHITECTURE,config=cfg,state_dict=model.state_dict(),best_stage='joint')
    path = tmp_path/'best.pt'; torch.save(saved,path)
    restored = restore(torch.load(path,map_location='cpu',weights_only=False),cfg).eval()
    with torch.no_grad(): torch.testing.assert_close(model(batch),restored(batch),rtol=0,atol=0)
    with pytest.raises(ValueError): restore(dict(saved,schema='wrong'),cfg)
    del restored
    torch.manual_seed(42); a = BondModel('BOND_SELF'); rng = torch.get_rng_state()
    torch.manual_seed(42); b = BondModel('BOND_NEIGHBOR')
    assert torch.equal(rng,torch.get_rng_state())
    assert all(torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items())
    assert sum(p.numel() for p in a.parameters()) == sum(p.numel() for p in b.parameters())


def test_full_precision_double_gate_and_test_dispatch(tmp_path,monkeypatch):
    assert not development_gate(.5099999,.49,.5)['passed']
    assert not development_gate(.5099999,.5,.49)['passed']
    assert development_gate(.5100001,.5,.5)['passed']
    from scripts import run_ua2d_demol as entry
    monkeypatch.setattr(entry,'OUT',tmp_path)
    (tmp_path/'selection').mkdir()
    decision = tmp_path/'selection/development_decision.json'
    decision.write_text('{"passed": false}')
    controller = entry.Controller.__new__(entry.Controller)
    controller.jobs = []; controller.refresh = lambda: dict(test_starts=0)
    with pytest.raises(RuntimeError,match='forbidden'): controller.launch('forbidden','test',[])
    decision.write_text('{"passed": true}')
    with pytest.raises(RuntimeError,match='forbidden'): controller.launch('unfrozen','test',[])
    assert controller.jobs == []
