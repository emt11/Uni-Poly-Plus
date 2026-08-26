#!/usr/bin/env python3
"""Deterministic identity/gradient sanity for SA and X2A Q/K routing."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.modules.periodic_line_glt_v2 import LocalPeriodicGraphLineTransformerV2


def run(mode):
    torch.manual_seed(42)
    encoder = LocalPeriodicGraphLineTransformerV2(
        hidden_size=512, layers=2, heads=8,
        attention_conditioning_mode=mode,
    ).eval()
    data = SimpleNamespace(
        glt_token_atom_a=torch.tensor([0, 1, 2, 0, 3]),
        glt_token_atom_b=torch.tensor([1, 2, 3, 3, 1]),
        glt_token_valid=torch.tensor([True, True, False, True, True]),
        glt_token_shift=torch.tensor([0, 1, 0, 1, 0]),
    )
    torch.manual_seed(99)
    states = torch.randn(5, 512, requires_grad=True)
    canonical = torch.randn(4, 512, requires_grad=True)
    gamma = encoder._attention_modulation(data, states, canonical)
    normalized = encoder.layers[0].norm1(states)
    source = torch.tensor([0, 1, 2, 3, 4, 0, 1])
    target = torch.tensor([0, 1, 2, 3, 4, 1, 0])
    angle_bias = torch.randn(7, 8)
    saved_bias = angle_bias.clone()
    a0, v0 = encoder.layers[0].attention.attention_weights(
        normalized, source, target, angle_bias
    )
    a1, v1 = encoder.layers[0].attention.attention_weights(
        normalized, source, target, angle_bias,
        qk_states=(1.0 + gamma) * normalized,
    )
    baseline = encoder.layers[0].attention(
        normalized, source, target, angle_bias
    )
    routed = encoder.layers[0].attention(
        normalized, source, target, angle_bias,
        qk_states=(1.0 + gamma) * normalized,
    )
    identity = bool(torch.equal(baseline, routed))
    routed.square().mean().backward()
    gradient = encoder.attention_conditioning_projection.weight.grad
    swapped = SimpleNamespace(**vars(data))
    swapped.glt_token_atom_a = data.glt_token_atom_b
    swapped.glt_token_atom_b = data.glt_token_atom_a
    swapped_gamma = encoder._attention_modulation(
        swapped, states.detach(), canonical.detach()
    )
    return {
        'mode': mode,
        'parameter_count': sum(
            p.numel() for p in encoder.attention_conditioning_projection.parameters()
        ),
        'step0_identity': identity,
        'wgamma_gradient_norm': float(torch.linalg.vector_norm(gradient)),
        'wgamma_gradient_finite': bool(torch.isfinite(gradient).all()),
        'condition_source_gradient_abs_sum': (
            0.0 if canonical.grad is None else float(canonical.grad.abs().sum())
        ),
        'content_gradient_norm': float(torch.linalg.vector_norm(states.grad)),
        'endpoint_swap_invariant': (
            bool(torch.equal(gamma, swapped_gamma)) if mode == 'x2a' else None
        ),
        'invalid_gamma_zero': bool(torch.equal(
            gamma[~data.glt_token_valid],
            torch.zeros_like(gamma[~data.glt_token_valid]),
        )),
        'value_path_unchanged': bool(torch.equal(v0, v1)),
        'angle_bias_unchanged': bool(torch.equal(angle_bias, saved_bias)),
        'attention_step0_equal': bool(torch.equal(a0, a1)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    sa, x2a = run('self'), run('x2a')
    result = {
        'schema': 'mts-glt-v2-attention-routing-gradient-sanity-v1',
        'sa': sa, 'x2a': x2a,
        'parameter_count_equal': sa['parameter_count'] == x2a['parameter_count'],
        'step0_equivalence_pass': sa['step0_identity'] and x2a['step0_identity'],
        'first_backward_gradient_pass': all(
            item['wgamma_gradient_finite'] and item['wgamma_gradient_norm'] > 0
            for item in (sa, x2a)
        ),
        'stop_gradient_pass': x2a['condition_source_gradient_abs_sum'] == 0.0,
        'value_path_unchanged_pass': sa['value_path_unchanged'] and x2a['value_path_unchanged'],
        'angle_bias_unchanged_pass': sa['angle_bias_unchanged'] and x2a['angle_bias_unchanged'],
        'endpoint_swap_pass': bool(x2a['endpoint_swap_invariant']),
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.tmp.{os.getpid()}')
    try:
        tmp.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
