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
        hidden_size=16, layers=1, heads=4, distance_basis=8,
        angle_basis=8, line_conditioning_mode=mode,
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


def test_sl_x2l_are_parameter_matched_and_zero_initialized():
    sl = _encoder('self')
    x2l = _encoder('x2l')
    sl_state = sl.line_conditioning_projection.state_dict()
    x2l_state = x2l.line_conditioning_projection.state_dict()
    assert sum(v.numel() for v in sl_state.values()) == sum(
        v.numel() for v in x2l_state.values()
    )
    for key in sl_state:
        assert torch.equal(sl_state[key], x2l_state[key])
        assert torch.equal(sl_state[key], torch.zeros_like(sl_state[key]))


def test_identity_initialization_has_nonzero_first_backward_gradient():
    for mode in ('self', 'x2l'):
        encoder = _encoder(mode)
        data, states, canonical = _inputs()
        conditioned, views = encoder._condition_line_inputs(
            data, states, canonical
        )
        assert torch.equal(conditioned, states)
        assert torch.equal(
            views['line_modulation'], torch.zeros_like(views['line_modulation'])
        )
        conditioned.square().mean().backward()
        gradient = encoder.line_conditioning_projection.weight.grad
        assert gradient is not None and torch.isfinite(gradient).all()
        assert float(gradient.abs().sum()) > 0
        assert states.grad is not None and float(states.grad.abs().sum()) > 0
        if mode == 'x2l':
            assert canonical.grad is None


def test_x2l_endpoint_swap_is_invariant_and_invalid_modulation_is_zero():
    encoder = _encoder('x2l')
    with torch.no_grad():
        encoder.line_conditioning_projection.weight.normal_(0.0, 0.05)
    data, states, canonical = _inputs()
    _, first = encoder._condition_line_inputs(data, states, canonical)
    swapped = SimpleNamespace(
        glt_token_atom_a=data.glt_token_atom_b,
        glt_token_atom_b=data.glt_token_atom_a,
        glt_token_valid=data.glt_token_valid,
        glt_token_shift=data.glt_token_shift,
    )
    _, second = encoder._condition_line_inputs(swapped, states, canonical)
    assert torch.equal(first['line_modulation'], second['line_modulation'])
    assert torch.equal(
        first['line_modulation'][~data.glt_token_valid],
        torch.zeros_like(first['line_modulation'][~data.glt_token_valid]),
    )


def test_real_batch_step0_predictions_match_formal_baseline():
    fixture_path = Path(__file__).with_name('test_mts_periodic_line_glt.py')
    spec = importlib.util.spec_from_file_location('glt_fixture', fixture_path)
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
    outputs = []
    for mode, downstream in (
        ('self', 'o8_glt_atom_line_self'),
        ('x2l', 'o8_glt_atom_line_x2l'),
    ):
        torch.manual_seed(31)
        candidate = MTSGraphLineModelV2(
            glt_layers=1, line_conditioning_mode=mode
        ).eval()
        merged = candidate.state_dict()
        merged.update({key: value for key, value in baseline_state.items() if key in merged})
        candidate.load_state_dict(merged, strict=True)
        with torch.no_grad():
            outputs.append(candidate.forward_downstream(batch, downstream))
    with torch.no_grad():
        expected = baseline.forward_downstream(batch, 'o8_glt_atom')
    assert torch.equal(expected, outputs[0])
    assert torch.equal(expected, outputs[1])
