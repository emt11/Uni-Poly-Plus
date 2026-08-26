#!/usr/bin/env python3
"""Small deterministic gradient/identity sanity for SL and X2L."""

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
        hidden_size=512, layers=1, heads=8,
        line_conditioning_mode=mode,
    )
    data = SimpleNamespace(
        glt_token_atom_a=torch.tensor([0, 1, 2, 0, 3]),
        glt_token_atom_b=torch.tensor([1, 2, 3, 3, 1]),
        glt_token_valid=torch.tensor([True, True, False, True, True]),
        glt_token_shift=torch.tensor([0, 1, 0, 1, 0]),
    )
    torch.manual_seed(99)
    states = torch.randn(5, 512, requires_grad=True)
    canonical = torch.randn(4, 512, requires_grad=True)
    conditioned, views = encoder._condition_line_inputs(data, states, canonical)
    identity = bool(torch.equal(conditioned, states))
    loss = conditioned.square().mean()
    loss.backward()
    gradient = encoder.line_conditioning_projection.weight.grad
    source_grad = canonical.grad
    swapped = SimpleNamespace(
        glt_token_atom_a=data.glt_token_atom_b,
        glt_token_atom_b=data.glt_token_atom_a,
        glt_token_valid=data.glt_token_valid,
        glt_token_shift=data.glt_token_shift,
    )
    _, swapped_views = encoder._condition_line_inputs(
        swapped, states.detach(), canonical.detach()
    )
    return {
        'mode': mode, 'parameter_count': sum(
            p.numel() for p in encoder.line_conditioning_projection.parameters()
        ),
        'step0_identity': identity,
        'wgamma_gradient_norm': float(torch.linalg.vector_norm(gradient)),
        'wgamma_gradient_finite': bool(torch.isfinite(gradient).all()),
        'condition_source_gradient_abs_sum': (
            0.0 if source_grad is None else float(source_grad.abs().sum())
        ),
        'content_gradient_norm': float(torch.linalg.vector_norm(states.grad)),
        'endpoint_swap_invariant': bool(torch.equal(
            views['line_modulation'], swapped_views['line_modulation']
        )) if mode == 'x2l' else None,
        'invalid_modulation_zero': bool(torch.equal(
            views['line_modulation'][~data.glt_token_valid],
            torch.zeros_like(views['line_modulation'][~data.glt_token_valid]),
        )),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    sl, x2l = run('self'), run('x2l')
    result = {
        'schema': 'mts-glt-v2-line-conditioning-gradient-sanity-v1',
        'sl': sl, 'x2l': x2l,
        'parameter_count_equal': sl['parameter_count'] == x2l['parameter_count'],
        'step0_equivalence_pass': sl['step0_identity'] and x2l['step0_identity'],
        'step0_wgamma_gradient_pass': all(
            item['wgamma_gradient_finite'] and item['wgamma_gradient_norm'] > 0
            for item in (sl, x2l)
        ),
        'stop_gradient_pass': x2l['condition_source_gradient_abs_sum'] == 0.0,
        'endpoint_swap_pass': bool(x2l['endpoint_swap_invariant']),
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
