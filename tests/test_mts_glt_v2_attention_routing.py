from types import SimpleNamespace
import importlib.util
from pathlib import Path

import torch

from src.modules.periodic_line_glt_v2 import LocalPeriodicGraphLineTransformerV2
from src.modules.mts_glt_v2 import MTSGraphLineModelV2
from src.dataset.dataloader import mips_trimer_collate


def _encoder(mode, seed=17):
    torch.manual_seed(seed)
    return LocalPeriodicGraphLineTransformerV2(
        hidden_size=16, layers=2, heads=4, distance_basis=8,
        angle_basis=8, attention_conditioning_mode=mode,
    )


def _inputs():
    data = SimpleNamespace(
        glt_token_atom_a=torch.tensor([0, 1, 2, 0]),
        glt_token_atom_b=torch.tensor([1, 2, 3, 3]),
        glt_token_valid=torch.tensor([True, True, False, True]),
        glt_token_shift=torch.tensor([0, 1, 0, 1]),
    )
    states = torch.randn(4, 16, requires_grad=True)
    canonical = torch.randn(4, 16, requires_grad=True)
    return data, states, canonical


def test_sa_x2a_are_parameter_matched_zero_initialized_and_identity():
    sa, x2a = _encoder('self'), _encoder('x2a')
    for left, right in zip(
        sa.attention_conditioning_projection.parameters(),
        x2a.attention_conditioning_projection.parameters(),
    ):
        assert left.numel() == right.numel()
        assert torch.equal(left, right)
        assert torch.equal(left, torch.zeros_like(left))
    data, states, canonical = _inputs()
    for encoder in (sa, x2a):
        gamma = encoder._attention_modulation(data, states, canonical)
        assert torch.equal(gamma, torch.zeros_like(gamma))


def test_first_backward_reaches_projection_but_not_condition_source():
    source = torch.tensor([0, 1, 2, 3, 0])
    target = torch.tensor([0, 1, 2, 3, 1])
    bias = torch.randn(5, 4)
    for mode in ('self', 'x2a'):
        encoder = _encoder(mode)
        data, states, canonical = _inputs()
        gamma = encoder._attention_modulation(data, states, canonical)
        normalized = encoder.layers[0].norm1(states)
        output = encoder.layers[0].attention(
            normalized, source, target, bias,
            qk_states=(1.0 + gamma) * normalized,
        )
        output.square().mean().backward()
        gradient = encoder.attention_conditioning_projection.weight.grad
        assert gradient is not None and torch.isfinite(gradient).all()
        assert float(gradient.abs().sum()) > 0
        if mode == 'x2a':
            assert canonical.grad is None


def test_x2a_endpoint_swap_and_invalid_mask():
    encoder = _encoder('x2a')
    with torch.no_grad():
        encoder.attention_conditioning_projection.weight.normal_(0.0, 0.05)
    data, states, canonical = _inputs()
    first = encoder._attention_modulation(data, states, canonical)
    swapped = SimpleNamespace(**vars(data))
    swapped.glt_token_atom_a = data.glt_token_atom_b
    swapped.glt_token_atom_b = data.glt_token_atom_a
    second = encoder._attention_modulation(swapped, states, canonical)
    assert torch.equal(first, second)
    assert torch.equal(
        first[~data.glt_token_valid],
        torch.zeros_like(first[~data.glt_token_valid]),
    )


def test_routing_changes_only_qk_and_not_value_or_angle_bias():
    encoder = _encoder('x2a')
    with torch.no_grad():
        encoder.attention_conditioning_projection.weight.normal_(0.0, 0.05)
    data, states, canonical = _inputs()
    gamma = encoder._attention_modulation(data, states, canonical)
    normalized = encoder.layers[0].norm1(states)
    source = torch.tensor([0, 1, 2, 3, 0])
    target = torch.tensor([0, 1, 2, 3, 1])
    angle_bias = torch.randn(5, 4)
    saved_bias = angle_bias.clone()
    a0, v0 = encoder.layers[0].attention.attention_weights(
        normalized, source, target, angle_bias
    )
    a1, v1 = encoder.layers[0].attention.attention_weights(
        normalized, source, target, angle_bias,
        qk_states=(1.0 + gamma) * normalized,
    )
    assert torch.equal(v0, v1)
    assert torch.equal(angle_bias, saved_bias)
    assert not torch.equal(a0, a1)


def test_real_batch_step0_predictions_match_formal_baseline():
    fixture_path = Path(__file__).with_name('test_mts_periodic_line_glt.py')
    spec = importlib.util.spec_from_file_location('glt_fixture_attention', fixture_path)
    fixture = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(fixture)
    samples = [fixture._sample('*CCO*'), fixture._sample('*COC*')]
    records = [
        fixture.build_periodic_line_sample(
            fixture.sample_key_from_smiles(smiles), sample, sample
        )
        for smiles, sample in zip(('*CCO*', '*COC*'), samples)
    ]
    batch = mips_trimer_collate([
        fixture._attach(sample, record)
        for sample, record in zip(samples, records)
    ])
    torch.manual_seed(23)
    baseline = MTSGraphLineModelV2(glt_layers=1).eval()
    baseline_state = baseline.state_dict()
    with torch.no_grad():
        expected = baseline.forward_downstream(batch, 'o8_glt_atom')
    for attention_mode, downstream in (
        ('self', 'o8_glt_atom_attn_self'),
        ('x2a', 'o8_glt_atom_attn_x2a'),
    ):
        torch.manual_seed(31)
        candidate = MTSGraphLineModelV2(
            glt_layers=1, attention_conditioning_mode=attention_mode
        ).eval()
        merged = candidate.state_dict()
        merged.update({
            key: value for key, value in baseline_state.items() if key in merged
        })
        candidate.load_state_dict(merged, strict=True)
        with torch.no_grad():
            actual = candidate.forward_downstream(batch, downstream)
        assert torch.equal(expected, actual)
