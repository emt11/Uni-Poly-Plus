"""The one final r1 CPU suite: manual geometries, no conformer generation."""
import copy
from dataclasses import asdict
import numpy as np
import pytest
import torch
from torch_geometric.data import Data

from src.dataset.ua_hop_ph import exact_shells, periodic_coverage, prepare_sample, collate
from src.dataset.glt_dual_static import build_dual_static
from src.dataset.canonical_periodic import build_canonical_periodic_topology
from src.modules.ua_hop_ph import HopModel, HopAdapter, initialize, restore, optimizer, SCHEMA, ARCHITECTURE
from src.modules.ua2d_galformer_pretrain import PretrainConfig, SCHEMA as PT_SCHEMA
from src.modules.glt_dual import mean_pool
from tests.test_dual_glt import _toy_pair


@pytest.fixture(scope='module')
def samples():
    result = []
    for text in ('*COC*', '*CCN*'):
        top = build_canonical_periodic_topology(text)
        _, tri = _toy_pair(top.normalized_canonical_smiles)  # deterministic hand-written coordinates
        static = build_dual_static(top, tri, text)
        result.append(prepare_sample(top, tri, text, static)[0])
    return result


def test_true_line_hops_chain_rbf_and_no_self():
    pos = np.array([[i, 0., 0.] for i in range(6)])
    edge, hop, feat = exact_shells([[i, i+1] for i in range(5)], pos)
    pairs = {tuple(e): (int(k), f) for e, k, f in zip(edge.t().tolist(), hop, feat)}
    assert pairs[0, 2][0] == 2  # a legacy two-hop relation cannot become one hop
    assert pairs[0, 4][0] == 4
    assert all(s != t for s, t in pairs)
    expected = np.exp(-.5*((1.-np.arange(16)*4/15)/(4/15))**2)
    np.testing.assert_allclose(pairs[0, 4][1][:16], expected, rtol=2e-6, atol=1e-7)
    np.testing.assert_allclose(pairs[0, 4][1][16:], [-1., 1.], atol=1e-7)
    # Width is sigma in the denominator 2*sigma^2, and distant bonds are not clipped.
    _, _, long = exact_shells([[0, 1], [1, 2]], np.array([[0, 0, 0], [5, 0, 0], [10, 0, 0]]))
    assert float(long[0, 15]) < .01


def test_all_shortest_paths_are_averaged_before_nonlinearity():
    pos = np.array([[0., 0, 0], [1, 0, 0], [1, 2, 0], [0, 1, 0]])
    edge, hop, feat = exact_shells([[0, 1], [1, 2], [2, 3], [3, 0]], pos)
    pick = ((edge[0] == 0) & (edge[1] == 2)).nonzero().item()
    assert int(hop[pick]) == 2
    lengths = np.array([1., 2., np.sqrt(2), 1.])
    radial = np.exp(-.5*((lengths[:, None]-np.arange(16)*4/15)/(4/15))**2)
    expected = (2*radial[0]+radial[1]+2*radial[2]+radial[3])/6
    np.testing.assert_allclose(feat[pick, :16], expected, atol=1e-7)
    np.testing.assert_allclose(feat[pick, 16:], [0., -.5], atol=1e-7)
    adapter = HopAdapter()
    assert adapter.bias[0].in_features == 18 and adapter.bias[-1].out_features == 4


def test_physical_images_and_periodic_boundary_coverage():
    # Distinct physical atom rows remain distinct even when canonical IDs repeat.
    edge, hop, _ = exact_shells([[0, 1], [2, 3]], np.array([[0,0,0],[1,0,0],[2,0,0],[3,0,0]]))
    assert edge.shape == (2, 0) and hop.numel() == 0
    key = lambda q: ((0, q), (0, q+1))
    complete, detail = periodic_coverage([key(-1), key(0)], [0, 1], {(0, 0, 1)})
    assert complete == [0, 0, 0, 0]
    assert detail[0] == [False]*4
    complete, _ = periodic_coverage([key(q) for q in range(-4, 5)], [4], {(0, 0, 1)})
    assert complete == [1, 1, 1, 1]


def test_collation_preserves_bond_incidence_offsets_and_ph(samples):
    batch = collate(samples)
    n, m = samples[0].mips_x.shape[0], samples[0].bond_distance.numel()
    edges = samples[0].hop_edge_index.shape[1]
    assert torch.equal(batch.hop_edge_index[:, edges:], samples[1].hop_edge_index+m)
    inc = samples[0].incidence_index.shape[1]
    assert torch.equal(batch.incidence_index[:, inc:], samples[1].incidence_index+torch.tensor([[m], [n]]))
    assert batch.hop_ph.shape == (2, 31, 5)
    assert torch.isfinite(batch.hop_ph).all()
    for s in samples:
        assert s.hop_ph.shape == (31, 5)
    # Existing padding remains a negative sentinel after the shared collator.
    assert (batch.line_path[batch.line_path < 0] == -1).all()


def test_empty_shell_and_mask_after_nonzero_output_bias():
    a = HopAdapter()
    with torch.no_grad(): a.W_out.bias.fill_(3.)
    d = Data(hop_edge_index=torch.empty((2, 0), dtype=torch.long),
        hop_distance=torch.empty(0, dtype=torch.long), hop_features=torch.empty((0, 18)),
        bond_batch=torch.tensor([0, 1]), incidence_index=torch.tensor([[0, 1], [0, 2]]),
        geometry_valid=torch.tensor([True, False]), canonical_graph_index=torch.tensor([0, 0, 1]))
    value = a(torch.randn(2, 512), d, torch.full((2, 4), .25), 3)
    assert torch.equal(value[0], torch.full((512,), 3.))
    assert torch.count_nonzero(value[1:]) == 0  # no-incidence and no-geometry both mask learned bias
    d.incidence_index = torch.empty((2, 0), dtype=torch.long)
    assert torch.count_nonzero(a(torch.randn(2, 512), d, torch.full((2, 4), .25), 3)) == 0


def test_shared_initialization_capacity_and_optimizer_groups():
    torch.manual_seed(42); eq = HopModel('HOP_EQ')
    torch.manual_seed(42); ph = HopModel('HOP_PH')
    assert sum(p.numel() for p in ph.parameters())-sum(p.numel() for p in eq.parameters()) == 20484
    assert all(torch.equal(v, ph.state_dict()[k]) for k, v in eq.state_dict().items())
    assert len({id(b.norm) for b in ph.adapter.branches}) == 4
    assert len({id(b.qkv) for b in ph.adapter.branches}) == 4
    assert torch.equal(torch.softmax(ph.router(torch.randn(2, 155)), -1), torch.full((2, 4), .25))
    opt, evidence = optimizer(ph, 'joint')
    assert [x['lr'] for x in evidence] == [1e-4, 1e-3, 1e-3]
    params = [p for g in opt.param_groups for p in g['params']]
    assert len({id(p) for p in params}) == len(params)
    assert {id(p) for p in params} == {id(p) for p in ph.parameters()}
    opt, _ = optimizer(ph, 'head')
    assert {id(p) for g in opt.param_groups for p in g['params']} == {id(p) for p in ph.head.parameters()}
    assert not any(p.requires_grad for p in ph.encoder3d.parameters())


def test_zero_initial_residual_gradient_and_complete_restore(samples, tmp_path):
    batch = collate(samples)
    torch.manual_seed(42); model = HopModel('HOP_PH').eval()
    with torch.no_grad():
        atoms, _ = model.o8(batch)
        expected = model.head(model.norm2(mean_pool(atoms, batch.canonical_graph_index, 2))).view(-1)
        actual = model(batch)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    model(batch).sum().backward()
    assert model.adapter.W_out.weight.grad.abs().sum() > 0
    assert model.adapter.branches[0].qkv.weight.grad.abs().sum() == 0
    model.zero_grad(set_to_none=True)
    with torch.no_grad(): model.adapter.W_out.weight.normal_(0, .01)
    model(batch).square().sum().backward()
    assert model.adapter.branches[0].qkv.weight.grad.abs().sum() > 0
    assert model.encoder3d.layers[-1].attention.qkv.weight.grad.abs().sum() > 0
    assert model.router[-1].weight.grad.abs().sum() > 0
    cfg = dict(arm='HOP_PH', task='synthetic', fold=0)
    saved = dict(schema=SCHEMA, architecture=ARCHITECTURE, config=cfg,
        state_dict=model.state_dict(), best_epoch=10, best_stage='head')
    path = tmp_path/'complete.pt'; torch.save(saved, path)
    restored = restore(torch.load(path, map_location='cpu', weights_only=False), cfg).eval()
    with torch.no_grad(): torch.testing.assert_close(restored(batch), model(batch), rtol=0, atol=0)
    bad = copy.deepcopy(saved); bad['architecture']['input']['shells'] = 'cumulative'
    with pytest.raises(ValueError): restore(bad, cfg)


def test_declared_parent_loading_is_strict():
    torch.manual_seed(7); reference = HopModel('HOP_EQ')
    state = {k: v for k, v in reference.state_dict().items() if k.startswith(('o8.', 'encoder3d.', 'norm2.'))}
    parent = dict(schema=PT_SCHEMA, step=5000, config=asdict(PretrainConfig(fp_enabled=True)),
        execution=dict(plan='UA2D-PRETRAIN-FP-20261019-01', revision='r4'), state_dict=state)
    restored = initialize(parent, reference.head.state_dict(), 'HOP_EQ')
    assert all(torch.equal(v, restored.state_dict()[k]) for k, v in state.items())
    parent['step'] = 4999
    with pytest.raises(ValueError): initialize(parent, reference.head.state_dict(), 'HOP_EQ')
