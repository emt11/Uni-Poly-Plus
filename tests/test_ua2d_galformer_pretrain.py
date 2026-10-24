"""Final CPU correctness suite, synthetic molecules only, no training campaign."""
import copy
from dataclasses import replace
import pytest
import torch

from src.dataset.canonical_periodic import build_canonical_periodic_topology
from src.dataset.ua2d_galformer_pretrain import prepare_sample, collate_samples, corruption
from src.modules.ua2d_galformer_pretrain import (PretrainConfig, UA2DGalformerPretrainer,
    deployment_package, restore_downstream, objective)
from tests.test_dual_glt import _toy_pair


def sample(text, config, position=0):
    top = build_canonical_periodic_topology(text)
    _, tri = _toy_pair(text)
    return prepare_sample(top, tri, text, config=config, seed=42, position=position)


@pytest.fixture(scope='module')
def geometry_batch():
    config = PretrainConfig(strategy='mask_align_geometry')
    return collate_samples([sample('*COC*', config), sample('*CCN*', config, 1)])


def test_config_and_bert_corruption_are_explicit_and_replayable():
    for values in [dict(strategy='other'), dict(fusion='token'), dict(mask_ratio=0), dict(noise_sigma=float('nan'))]:
        with pytest.raises(ValueError):
            PretrainConfig(**values)
    for strategy in ('mask_align', 'mask_align_geometry'):
        with pytest.raises(ValueError, match='disable 3D-to-2D fusion'):
            PretrainConfig(strategy=strategy, fusion='3d_to_2d')
    a = corruption(1000, torch.Generator().manual_seed(42), .4)
    b = corruption(1000, torch.Generator().manual_seed(42), .4)
    assert all(torch.equal(x, y) for x, y in zip(a, b))
    selected, action, donor = a
    assert int(selected.sum()) == 400
    assert not (action[~selected] != 0).any()
    assert .7 < float((action[selected] == 1).float().mean()) < .9
    assert (donor[action == 2] != torch.arange(1000)[action == 2]).all()


def test_variants_share_views_and_common_initial_weights():
    base = PretrainConfig()
    geo = replace(base, strategy='mask_align_geometry')
    a, x = sample('*COC*', base, 9)
    b, y = sample('*COC*', geo, 9)
    for name in ('mips_x', 'bond_z_a', 'bond_z_b', 'bond_distance', 'line_angle'):
        assert torch.equal(getattr(a, name), getattr(b, name)), name
    for name in ('atom_selected', 'atom_label', 'physical_pos'):
        assert torch.equal(x[name], y[name]), name
    assert 'token_action' in x and not any(key.startswith('token_') for key in y)
    assert 'coordinate_target' not in x and 'coordinate_target' in y
    torch.manual_seed(7)
    first = UA2DGalformerPretrainer(base)
    torch.manual_seed(7)
    second = UA2DGalformerPretrainer(geo)
    assert not hasattr(first, 'coordinate_head') and not hasattr(first, 'length_head')
    assert hasattr(first, 'line_head') and not hasattr(second, 'line_head')
    for name, value in first.state_dict().items():
        if not name.startswith('line_head.'):
            assert torch.equal(value, second.state_dict()[name]), name


def test_atomic_mask_does_not_corrupt_independent_3d_identity_and_targets_stay_clean():
    config = PretrainConfig(mask_ratio=.99)
    top = build_canonical_periodic_topology('*COC*')
    _, tri = _toy_pair('*COC*')
    original = tri.trimer_pos.clone()
    data, labels = prepare_sample(top, tri, '*COC*', config=config, seed=42, position=3)
    assert torch.equal(tri.trimer_pos, original)
    masked = (data.mips_x == 0).all(-1)
    assert masked.any()
    from src.dataset.glt_dual import build_dual_sample
    clean = build_dual_sample(top, tri, '*COC*', geometry_mode='finite_incident')
    assert torch.equal(data.bond_z_a, clean.bond_z_a)
    assert torch.equal(data.bond_z_b, clean.bond_z_b)
    assert torch.equal(labels['atom_label'], top.mips_x[:, :101].argmax(-1))


def test_bond_and_physical_atom_offsets_are_distinct(geometry_batch):
    data, labels = geometry_batch
    a, b = labels['physical_edges']
    assert torch.equal(labels['physical_graph'][a], data.bond_batch)
    assert torch.equal(labels['physical_graph'][b], data.bond_batch)
    assert 'token_donor' not in labels
    left, right = labels['angle_pairs'].unbind(-1)
    assert torch.equal(data.bond_batch[left], labels['angle_graph'])
    assert torch.equal(data.bond_batch[right], labels['angle_graph'])
    assert torch.equal(data.bond_canonical_a.new_tensor([0, 1]), torch.unique(labels['angle_graph']))


def test_v2_has_exactly_five_objectives_and_all_heads_receive_gradients(geometry_batch):
    data, labels = geometry_batch
    model = UA2DGalformerPretrainer(PretrainConfig(strategy='mask_align_geometry')).eval()
    model.completed_updates = 1000
    result = model(data, labels)
    assert set(result['sums']) == {'mask_2d', 'contrast', 'coordinate', 'length', 'angle'}
    assert not hasattr(model, 'fusion') and not hasattr(model, 'line_head')
    expected = model.target_counts(data, labels)
    for key in expected:
        torch.testing.assert_close(expected[key], result['counts'][key])
    assert torch.autograd.grad(result['sums']['mask_2d'], model.encoder3d.endpoint.weight,
                               allow_unused=True, retain_graph=True)[0] is None
    geometry = sum(result['sums'][key] for key in ('coordinate', 'length', 'angle'))
    assert torch.autograd.grad(geometry, model.o8.atom_embedding.projection.weight,
                               allow_unused=True, retain_graph=True)[0] is None
    objective(result, model.weights).backward()
    for name in ['o8.atom_embedding.projection.weight', 'encoder3d.endpoint.weight',
                 'atom_head.2.weight', 'proj2.net.2.weight',
                 'proj3.net.2.weight', 'coordinate_head.2.weight',
                 'length_head.2.weight', 'angle_head.2.weight']:
        grad = model.get_parameter(name).grad
        assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0, name


def test_geometry_off_never_reads_reconstruction_targets():
    config = PretrainConfig()
    data, labels = collate_samples([sample('*COC*', config), sample('*CCN*', config, 1)])
    model = UA2DGalformerPretrainer(config).eval()
    result = model(data, labels)
    assert set(result['sums']) == {'mask_2d', 'mask_3d', 'contrast'}
    assert not hasattr(model, 'fusion')
    assert torch.equal(data.bond_batch[labels['token_donor']], data.bond_batch)
    assert all('target' not in key for key in labels)
    loss = objective(result, model.weights)
    assert torch.isfinite(loss)
    loss.backward()
    assert model.line_head[2].weight.grad.abs().sum() > 0


def test_v2_ignores_3d_mask_corruption_even_if_legacy_labels_are_injected(geometry_batch):
    data, labels = geometry_batch
    model = UA2DGalformerPretrainer(PretrainConfig(strategy='mask_align_geometry')).eval()
    poisoned = dict(labels, token_action=torch.ones(len(data.bond_distance), dtype=torch.long),
                    token_donor=torch.arange(len(data.bond_distance)),
                    token_label=torch.full((len(data.bond_distance),), -1))
    with torch.no_grad():
        reference = model(data, labels)
        actual = model(data, poisoned)
    for key in reference['sums']:
        torch.testing.assert_close(actual['sums'][key], reference['sums'][key])


def test_coordinate_decoder_is_rotation_equivariant_and_translation_invariant(geometry_batch):
    data, labels = geometry_batch
    model = UA2DGalformerPretrainer(PretrainConfig(strategy='mask_align_geometry')).eval()
    bonds = torch.randn(len(data.bond_distance), 512)
    prediction = model.coordinate_prediction(bonds, labels, 2)
    rotation = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    changed = dict(labels, physical_pos=labels['physical_pos'] @ rotation + 13.)
    actual = model.coordinate_prediction(bonds, changed, 2)
    torch.testing.assert_close(actual, prediction @ rotation, atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize('strategy', ['mask_align', 'mask_align_geometry'])
def test_package_restores_only_pure_2d_and_rejects_fusion(geometry_batch, strategy):
    data, _ = geometry_batch
    model = UA2DGalformerPretrainer(PretrainConfig(strategy=strategy)).eval()
    package = deployment_package(model, 0)
    assert package['prediction_paths'] == ['pure_2d']
    assert not any(name.startswith('fusion.') for name in package['state_dict'])
    from src.modules.glt_dual import Pure2DGLTModel
    head = Pure2DGLTModel().head.state_dict()
    pure = restore_downstream(package, head).eval()
    assert not hasattr(pure, 'encoder3d') and not hasattr(pure, 'fusion')
    assert all(torch.equal(value, pure.o8.state_dict()[name]) for name, value in model.o8.state_dict().items())
    with torch.no_grad():
        assert pure(data).shape == (2, 1)
    with pytest.raises(ValueError, match='disable 3D-to-2D fusion'):
        restore_downstream(package, head, fused=True)
    changed = copy.copy(package)
    changed['schema'] = 'ua2d_galformer_objectives_v1'
    with pytest.raises(ValueError, match='unsupported'):
        restore_downstream(changed, head)


def test_optimizer_window_denominators_are_used_without_extra_average():
    parameter = torch.tensor(2., requires_grad=True)
    result = dict(sums={'mask_2d': parameter * 3}, counts={'mask_2d': torch.tensor(1.)})
    loss = objective(result, {'mask_2d': 1.}, {'mask_2d': torch.tensor(6.)})
    torch.testing.assert_close(torch.autograd.grad(loss, parameter)[0], torch.tensor(.5))
