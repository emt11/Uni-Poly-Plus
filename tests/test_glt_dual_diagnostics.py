"""Minimal invariants for the optional pretraining diagnostics (Plan §4.2).

The diagnostics must be observation-only: identical loss, gradients,
state_dict, RNG consumption and data order whether they are enabled or not.
Synthetic coordinates are used here; the formal 16-record diagnostic set is
not replaced by these fixtures.
"""

import copy

import pytest
import torch

from test_complete_trimer_glt import _toy_pair
from src.dataset.canonical_periodic import build_canonical_periodic_topology
from src.dataset.glt_dual_pretrain import prepare_pretrain_sample, pretrain_collate
from src.modules.glt_dual_pretrain import DualPretrainer, global_objective
from scripts.pretrain_glt_dual import _module_grad_norms


def _rows(smiles_list=('*COC*', '*C*')):
    rows = []
    for position, smiles in enumerate(smiles_list):
        top = build_canonical_periodic_topology(smiles)
        _, tri = _toy_pair(smiles)
        rows.append(prepare_pretrain_sample(
            top, tri, smiles, seed=42, key=smiles, position=position
        ))
    return pretrain_collate(rows)


def _model(mode, diagnostics, seed=0):
    torch.manual_seed(seed)
    model = DualPretrainer(mode, collect_diagnostics=diagnostics)
    model.train()
    return model


@pytest.mark.parametrize('mode', ['concat', 'kfuse'])
def test_diagnostics_toggle_preserves_loss_gradients_state_and_rng(mode):
    data, labels = _rows()
    off = _model(mode, False)
    torch.manual_seed(1234)
    rng_before = torch.get_rng_state()
    result_off = off(data, labels)
    rng_after_off = torch.get_rng_state()
    loss_off = global_objective(result_off['sums'], result_off['counts'])
    loss_off.backward()
    grads_off = {name: parameter.grad.clone() for name, parameter in off.named_parameters()
                 if parameter.grad is not None}

    on = _model(mode, True)
    torch.manual_seed(1234)
    torch.set_rng_state(rng_before)
    result_on = on(data, labels)
    rng_after_on = torch.get_rng_state()
    loss_on = global_objective(result_on['sums'], result_on['counts'])
    loss_on.backward()

    torch.testing.assert_close(result_off['sums'], result_on['sums'])
    torch.testing.assert_close(result_off['counts'], result_on['counts'])
    torch.testing.assert_close(result_off['targets'], result_on['targets'])
    torch.testing.assert_close(loss_off, loss_on)
    # Diagnostics must not draw randomness.
    assert torch.equal(rng_after_off, rng_after_on)
    # Same architecture and parameters under the same seed.
    assert set(grads_off) == {name for name, parameter in on.named_parameters()
                              if parameter.grad is not None}
    for name, parameter in on.named_parameters():
        if name in grads_off:
            torch.testing.assert_close(grads_off[name], parameter.grad)
    for key, value in off.state_dict().items():
        torch.testing.assert_close(value, on.state_dict()[key])


def test_components_reconstruct_geometry_loss():
    for mode in ('concat', 'kfuse'):
        data, labels = _rows()
        model = _model(mode, True)
        result = model(data, labels)
        components = model.last_diagnostics['components']
        assert components['length_plus_angle_reconstructs_geometry'] is True
        torch.testing.assert_close(
            torch.tensor(components['length_sum'] + components['angle_sum']),
            torch.tensor(components['geometry_sum']),
            rtol=1e-5, atol=1e-6,
        )
        assert components['geometry_valid_count'] > 0
        assert components['graphs_without_angle'] == components['geometry_valid_count'] - components['angle_valid_count']


def test_component_tensors_carry_gradients():
    data, labels = _rows()
    model = _model('concat', True)
    result = model(data, labels)
    tensors = result['diagnostics']['component_tensors']
    for name in ('length', 'angle', 'fingerprint'):
        assert tensors[name].requires_grad, name
    for name, parameter in (('length', 'length_head.2.weight'),
                            ('angle', 'angle_head.2.weight'),
                            ('fingerprint', 'fp_head.3.weight')):
        grad = torch.autograd.grad(tensors[name], model.get_parameter(parameter),
                                   retain_graph=True)[0]
        assert grad is not None and torch.isfinite(grad).all(), name


def test_module_grad_norms_aligns_missing_gradients_to_zero():
    data, labels = _rows()
    model = _model('concat', True)
    result = model(data, labels)
    global_objective(result['sums'], result['counts']).backward()
    # Clear the angle head gradient on purpose: it must be reported as 0.0
    # instead of being dropped from the alignment.
    for parameter in model.angle_head.parameters():
        parameter.grad = None
    norms = _module_grad_norms(model)
    assert set(norms) >= {'o8_2d_encoder', 'glt_3d_encoder', 'length_head',
                          'angle_head', 'fp_head', 'atom_head', 'fusion'}
    assert norms['angle_head'] == 0.0
    assert norms['glt_3d_encoder'] > 0.0
    assert all(torch.isfinite(torch.tensor(value)) for value in norms.values())


def test_forward_diagnostic_override_skips_and_clears_detail():
    data, labels = _rows()
    model = _model('concat', True)
    model(data, labels, collect_diagnostics=True)
    assert model.last_diagnostics is not None
    result = model(data, labels, collect_diagnostics=False)
    assert model.last_diagnostics is None
    assert 'diagnostics' not in result
    model(data, labels, collect_diagnostics=True)
    assert model.last_diagnostics is not None


def test_gaussian_diagnostics_come_from_the_real_forward():
    data, labels = _rows()
    model = _model('concat', True)
    model(data, labels)
    diagnostics = model.last_diagnostics
    gaussian = diagnostics['gaussian']
    angle_sigma = model.encoder.glt.angle_bias.gaussian
    torch.testing.assert_close(
        torch.tensor(gaussian['angle_min_effective_sigma']),
        angle_sigma.stds.detach().abs().min() + 1e-2,
    )
    distance_sigma = model.encoder.glt.distance_basis
    torch.testing.assert_close(
        torch.tensor(gaussian['distance_min_effective_sigma']),
        distance_sigma.stds.detach().abs().min() + 1e-2,
    )
    total_rows = (gaussian['angle_rows_valid'] + gaussian['angle_rows_padding']
                  + gaussian['angle_rows_synthetic_self'])
    assert total_rows > 0
    assert gaussian['angle_basis_output_rms_valid_relations'] is not None


def test_angle_head_statistics_are_finite_and_descriptive():
    data, labels = _rows()
    model = _model('concat', True)
    model(data, labels)
    angle_head = model.last_diagnostics['angle_head']
    assert 0.0 <= angle_head['exact_plus_minus_one_fraction'] <= 1.0
    assert 0.0 <= angle_head['near_saturation_fraction_threshold_6'] <= 1.0
    for key in ('pre_tanh', 'tanh_derivative'):
        stats = angle_head[key]
        assert stats['count'] > 0
        for value in stats.values():
            assert value is None or torch.isfinite(torch.tensor(float(value)))
    derivative = angle_head['tanh_derivative']
    assert 0.0 <= derivative['min'] and derivative['max'] <= 1.0
