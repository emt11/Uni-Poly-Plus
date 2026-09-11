"""Focused tests for the new route; no training and no coordinate generation.

Synthetic geometry is explicitly synthetic. Real ordinary/N=0 cache coverage
is provided separately by scripts/validate_dual_glt.py, not claimed here.
"""
import copy
import math

import pytest
import torch

from test_complete_trimer_glt import _toy_pair
from src.dataset.canonical_periodic import build_canonical_periodic_topology
from src.dataset.glt_dual import bond_paths, build_dual_sample, dual_glt_collate, two_hop_paths
from src.dataset.periodic_line_glt_complete import build_complete_trimer_glt_sample
from src.modules.glt_dual import (
    SharedBondPathBias, SourceAttention512, TransformerBlock, TypeGaussian,
    PathAngleBias, triplet_type, build_dual_glt_model,
)
from src.modules.mts_glt_distill import SourceQPreLNAttention


def sample(smiles='*COC*'):
    _, trimer = _toy_pair(smiles)
    topology = build_canonical_periodic_topology(smiles)
    return topology, trimer, build_dual_sample(topology, trimer, smiles)


def test_periodic_bond_paths_two_shifts_and_chemistry():
    topology, _, item = sample('*C*')
    assert topology.lga_source_image_shift.abs().max() == 2
    assert torch.equal(item.bond_path_mask.sum(-1), topology.lga_spd)
    # One canonical pair has multiple physical relations; none are collapsed.
    assert item.bond_path_features.size(0) == topology.lga_edge_index.size(1)
    assert item.bond_path_features.size(0) == 5
    assert not item.bond_path_mask[topology.lga_spd == 0].any()
    for smiles in ('*C=C*', '*c1ccc(*)cc1', '*C/C=C/C*'):
        top = build_canonical_periodic_topology(smiles)
        features, mask = bond_paths(top, smiles)
        assert (features[mask][:, :5].sum(-1) == 1).all()
        assert (features[mask][:, 7:].sum(-1) == 1).all()
        assert (features[~mask] == 0).all()


def test_bond_bias_reference_and_padding():
    _, _, item = sample()
    encoder = SharedBondPathBias()
    features, mask = item.bond_path_features, item.bond_path_mask
    actual = encoder(features, mask)
    expected = torch.zeros_like(actual)
    for row in range(mask.size(0)):
        for k in range(2):
            if mask[row, k]:
                f = features[row, k]
                z = (encoder.type_emb.weight[f[:5].argmax()]
                     + encoder.conj_emb.weight[int(f[5])]
                     + encoder.ring_emb.weight[int(f[6])]
                     + encoder.stereo_emb.weight[f[7:].argmax()])
                expected[row] += z @ encoder.position[k] / mask[row].sum()
    torch.testing.assert_close(actual, expected)
    g1 = torch.autograd.grad(actual.square().sum(), encoder.position, retain_graph=True)[0]
    g2 = torch.autograd.grad(expected.square().sum(), encoder.position)[0]
    torch.testing.assert_close(g1, g2)
    assert (actual[~mask.any(-1)] == 0).all()


def test_two_hop_physical_paths_and_reverse():
    top, tri, item = sample()
    row = build_complete_trimer_glt_sample(top, tri, '*COC*')
    assert item.bond_distance.numel() == 8  # 3*2+2
    assert item.bond_center.sum() == 2
    edges = {(int(s), int(t)): float(a) for s, t, a in zip(
        row['relations']['relation_source'], row['relations']['relation_target'], row['relations']['relation_angle'])}
    actual = set(zip(item.line_source.tolist(), item.line_target.tolist()))
    expected = set(edges) | {(i, i) for i in range(8)}
    expected |= {(s, t) for s, middle in edges for middle2, t in edges if middle == middle2}
    assert actual == expected
    for r, (s, t) in enumerate(zip(item.line_source.tolist(), item.line_target.tolist())):
        reverse = ((item.line_source == t) & (item.line_target == s)).nonzero().item()
        path = item.line_path[r][item.line_path[r] >= 0]
        reverse_path = item.line_path[reverse][item.line_path[reverse] >= 0]
        assert torch.equal(path.flip(0), reverse_path)
        if not item.line_is_self[r]:
            expected_angles = torch.tensor([edges[int(a), int(b)] for a, b in zip(path, path[1:])])
            torch.testing.assert_close(item.line_angle[r][item.line_mask[r]], expected_angles)
    # Physical lengths remain separate instead of averaging the two seams.
    tri2 = copy.copy(tri)
    tri2.trimer_pos = tri.trimer_pos.clone()
    tri2.trimer_pos[0, 1] += 0.7
    changed = build_dual_sample(top, tri2, '*COC*')
    assert not torch.equal(changed.bond_distance, item.bond_distance)
    assert torch.equal(changed.bond_distance[changed.bond_center], item.bond_distance[item.bond_center])


def test_invalid_angle_invalidates_whole_geometry_and_batch_offsets():
    top, tri, ordinary = sample()
    broken = copy.copy(tri)
    broken.trimer_pos = tri.trimer_pos.clone()
    broken.trimer_pos[0] = float('nan')
    invalid = build_dual_sample(top, broken, '*COC*')
    assert not invalid.geometry_valid
    assert invalid.bond_distance.numel() == 0
    batch = dual_glt_collate([ordinary, ordinary, invalid])
    r = ordinary.line_path.size(0)
    second = batch.line_path[r:2*r]
    assert torch.equal(second[second >= 0], ordinary.line_path[ordinary.line_path >= 0] + 8)
    assert (second[ordinary.line_path < 0] == -1).all()
    assert not any('md200' in key or key == 'mips_md' for key in batch.keys())
    row = build_complete_trimer_glt_sample(top, tri, '*COC*')
    row['relations']['relation_valid'][0] = False
    with pytest.raises(ValueError, match='required physical angle'):
        two_hop_paths(row)


def test_galformer_vocab_and_gaussian_reference():
    low, high, bond, expected = [], [], [], []
    counter = 0
    for a in range(101):
        for kind in range(5):
            for b in range(a, 101):
                low.append(a + 1 if a < 100 else 0)
                high.append(b + 1 if b < 100 else 0)
                bond.append(kind)
                expected.append(counter)
                counter += 1
    actual = triplet_type(torch.tensor(low), torch.tensor(high), torch.tensor(bond))
    assert torch.equal(actual, torch.tensor(expected))
    gaussian = TypeGaussian(256, 104)
    distance = torch.tensor([1.3, 1.8])
    pair = torch.tensor([[5, 7], [1, 6]])
    scale = gaussian.mul.weight[pair].sum(1)
    bias = gaussian.bias.weight[pair].sum(1)
    sigma = gaussian.stds.abs() + 0.01
    expected = torch.exp(-((scale * distance[:, None] + bias - gaussian.means) ** 2)
                         / (2 * sigma ** 2)) / (math.sqrt(2 * 3.14159) * sigma)
    torch.testing.assert_close(gaussian(distance, pair), expected)


def test_angle_post_mlp_mask_and_input_unchanged():
    encoder = PathAngleBias()
    types = torch.tensor([2, 3, 4])
    path = torch.tensor([[0, 1, -1], [0, 1, 2], [1, 1, -1]])
    before = path.clone()
    angle = torch.tensor([[1.1, 0.], [1.1, 1.7], [0., 0.]])
    mask = torch.tensor([[True, False], [True, True], [True, False]])
    with torch.no_grad():
        encoder.position[1][2].bias.fill_(7)
    output = encoder(types, path, angle, mask)
    expected = []
    for row in range(3):
        parts = []
        for k in range(2):
            if mask[row, k]:
                pairs = types[path[row, k:k+2]].unsqueeze(0)
                parts.append(encoder.position[k](encoder.gaussian(angle[row, k:k+1], pairs)))
        expected.append(encoder.heads(sum(parts) / len(parts)))
    torch.testing.assert_close(output, torch.cat(expected))
    assert torch.equal(path, before)


def test_attention_scaling_source_direction_and_residual_reference():
    block = TransformerBlock(dropout=0).eval()
    x = torch.randn(3, 512)
    source, target = torch.tensor([0, 1, 2, 1]), torch.tensor([1, 1, 2, 0])
    bias = torch.randn(4, 8)
    q, k, v = block.attention.qkv(block.norm1(x)).reshape(3, 3, 8, 64).unbind(1)
    scores = (q[source] * k[target]).sum(-1) / math.sqrt(512) + bias
    aggregate = torch.zeros_like(v)
    for t in range(3):
        rows = target == t
        aggregate[t] = (scores[rows].softmax(0).unsqueeze(-1) * v[source[rows]]).sum(0)
    residual = x + block.output(aggregate.reshape(3, 512))
    expected = residual + block.ffn(block.norm2(residual))
    torch.testing.assert_close(block(x, source, target, bias), expected)
    # Legacy class was not edited and still uses per-head scaling.
    assert SourceQPreLNAttention().head_dim == 64
    assert block.attention.scale == 512 ** -0.5


@pytest.mark.parametrize('mode', ['concat', 'kfuse'])
def test_model_sharing_gradients_masks_and_fusion(mode):
    torch.manual_seed(9)
    _, _, ordinary = sample()
    _, _, n0 = sample('*C*')
    batch = dual_glt_collate([ordinary, n0])
    model = build_dual_glt_model(mode, dropout=0)
    tensors = {'o8': [], 'glt': []}
    calls = {'o8': 0, 'glt': 0}
    def counter(name):
        def hook(*_):
            calls[name] += 1
        return hook
    handles = [model.o8.bond_bias.register_forward_hook(counter('o8')),
               model.glt.angle_bias.register_forward_hook(counter('glt'))]
    for name, branch in [('o8', model.o8), ('glt', model.glt)]:
        for layer in branch.layers:
            handles.append(layer.register_forward_pre_hook(
                lambda module, args, name=name: tensors[name].append(args[-1])))
    prediction = model(batch)
    assert prediction.shape == (2, 1) and torch.isfinite(prediction).all()
    assert calls == {'o8': 1, 'glt': 1}
    for values in tensors.values():
        assert len(values) == 6 and all(value is values[0] for value in values)
    for handle in handles:
        handle.remove()
    (prediction - torch.tensor([[0.7], [-0.2]])).square().mean().backward()
    for name in ['o8.bond_bias.position', 'o8.layers.0.attention.qkv.weight',
                 'glt.endpoint.weight', 'glt.distance_projection.weight',
                 'glt.angle_bias.heads.2.weight', 'glt.layers.0.attention.qkv.weight',
                 'predictor.0.weight']:
        grad = model.get_parameter(name).grad
        assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0, name
    if mode == 'kfuse':
        assert (model.kfuse.last_attention_weights == 1).all()
        assert model.kfuse.Wq.weight.grad.abs().sum() == 0
        assert model.kfuse.k_proj['glt3d'].weight.grad.abs().sum() == 0
        assert model.kfuse.v_proj['glt3d'].weight.grad.abs().sum() > 0
    encoded = model.encode(batch)
    assert encoded['geometry_valid'].tolist() == [True, False]
    assert (encoded['graph_3d'][1] == 0).all()
    if mode == 'kfuse':
        atoms = encoded['atom_states']
        raw = model.kfuse(atoms, {'glt3d': encoded['graph_3d']}, batch.canonical_graph_index)
        expected = atoms + 0.5 * model.kfuse.v_proj['glt3d'](encoded['graph_3d'])[batch.canonical_graph_index]
        torch.testing.assert_close(raw, expected)
        with torch.no_grad():
            model.kfuse.v_proj['glt3d'].bias.fill_(10)
        n0_batch = dual_glt_collate([n0])
        reference = model.encode(n0_batch)['graph_2d']
        torch.testing.assert_close(model(n0_batch), model.predictor(reference))
    # All 138 input columns are masked before the atom projection.
    masked = model.o8.atom_embedding(batch, torch.ones(batch.mips_x.size(0), dtype=torch.bool))
    torch.testing.assert_close(masked, model.o8.atom_embedding.projection.bias.expand_as(masked))
    assert not any('md200' in name or 'md_residual' in name for name, _ in model.named_modules())


def test_geometry_invariance_and_no_direct_stereo_token_input():
    top, tri, item = sample()
    moved = copy.copy(tri)
    rotation = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    moved.trimer_pos = tri.trimer_pos @ rotation + torch.tensor([3., -2., 1.])
    transformed = build_dual_sample(top, moved, '*COC*')
    model = build_dual_glt_model(dropout=0).eval()
    a = model.encode(dual_glt_collate([item]))
    b = model.encode(dual_glt_collate([transformed]))
    torch.testing.assert_close(a['graph_3d'], b['graph_3d'], atol=1e-5, rtol=1e-5)
    # Changing the 2D bond chemistry cannot directly change the 3D encoder.
    other = copy.deepcopy(item)
    other.bond_path_features[..., 5] = 1 - other.bond_path_features[..., 5]
    c = model.encode(dual_glt_collate([other]))
    torch.testing.assert_close(a['bond_states'], c['bond_states'])
    invalid = copy.deepcopy(item)
    invalid.geometry_valid = False
    assert not model.encode(dual_glt_collate([invalid]))['geometry_valid'].any()


@pytest.mark.parametrize('mode', ['concat', 'kfuse'])
def test_entire_batch_without_geometry(mode):
    top, tri, _ = sample()
    tri.trimer_geometry_valid = False
    item = build_dual_sample(top, tri, '*COC*')
    batch = dual_glt_collate([item])
    assert batch.bond_distance.numel() == batch.line_path.size(0) == 0
    model = build_dual_glt_model(mode, dropout=0)
    result = model(batch)
    assert result.shape == (1, 1) and torch.isfinite(result).all()
    result.sum().backward()
    assert model.predictor[0].weight.grad is not None
    assert not model.encode(batch)['geometry_valid'].any()
