#!/usr/bin/env python3
"""Geometry, parity and gradient sanity for reflection-invariant torsion bias."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset.dataloader import mips_trimer_collate
from src.dataset.periodic_line_torsion import (
    build_periodic_torsion_fields,
    canonical_torsion_quadruplet,
    reflection_invariant_dihedral_cosine,
)
from src.modules.mts_glt_v2 import MTSGraphLineModelV2
from src.modules.periodic_line_glt_v2 import TorsionRelationBiasV2


def _batch():
    path = ROOT / 'tests/test_mts_periodic_line_glt.py'
    spec = importlib.util.spec_from_file_location('torsion_sanity_fixture', path)
    fixture = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(fixture)
    samples = [fixture._sample('*CCO*'), fixture._sample('*CCCC*')]
    records = [
        fixture.build_periodic_line_sample(
            fixture.sample_key_from_smiles(smiles), sample, sample
        )
        for smiles, sample in zip(('*CCO*', '*CCCC*'), samples)
    ]
    return mips_trimer_collate([
        build_periodic_torsion_fields(fixture._attach(sample, record))
        for sample, record in zip(samples, records)
    ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    points = torch.tensor([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
        [1.2, 1.0, 0.0], [2.0, 1.3, 0.8],
    ])
    expected = reflection_invariant_dihedral_cosine(*points)
    rotation = torch.tensor([
        [0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]
    ])
    transformed = points @ rotation.T + torch.tensor([2.0, -3.0, 1.5])
    reflected = points.clone(); reflected[:, 0] *= -1
    geometry_invariance = all(abs(value - expected) < 1e-6 for value in (
        reflection_invariant_dihedral_cosine(*transformed),
        reflection_invariant_dihedral_cosine(*reflected),
        reflection_invariant_dihedral_cosine(*reversed(points)),
    ))
    atoms = ((1, -1), (2, 0), (3, 0), (4, 1))
    canonicalization = (
        canonical_torsion_quadruplet(atoms)
        == canonical_torsion_quadruplet(tuple(reversed(atoms)))
        == canonical_torsion_quadruplet(tuple((a, q + 1) for a, q in atoms))
    )
    batch = _batch()
    torch.manual_seed(23)
    baseline = MTSGraphLineModelV2(glt_layers=1).eval()
    baseline_state = baseline.state_dict()
    with torch.no_grad():
        expected_prediction = baseline.forward_downstream(batch, 'o8_glt_atom')
        _, _, baseline_bias, _ = baseline.glt._relations(batch, torch.float32)
    parity = True
    angle_unchanged = True
    parameter_counts = {}
    for mode, downstream in (
        ('count', 'o8_glt_atom_torsion_count'),
        ('full', 'o8_glt_atom_torsion'),
    ):
        candidate = MTSGraphLineModelV2(glt_layers=1, torsion_mode=mode).eval()
        merged = candidate.state_dict()
        merged.update({key: value for key, value in baseline_state.items() if key in merged})
        candidate.load_state_dict(merged, strict=True)
        parameter_counts[mode] = sum(p.numel() for p in candidate.glt.torsion_bias.parameters())
        with torch.no_grad():
            prediction = candidate.forward_downstream(batch, downstream)
            _, _, bias, views = candidate.glt._relations(
                batch, torch.float32, return_torsion=True
            )
        parity = parity and torch.equal(expected_prediction, prediction)
        angle_unchanged = angle_unchanged and torch.equal(baseline_bias, bias)
        angle_unchanged = angle_unchanged and torch.equal(
            baseline_bias[:views['angle_bias'].size(0)], views['angle_bias']
        )
    gradient_batch = _batch()
    gradient_batch.glt_torsion_observation_value = torch.cat([
        gradient_batch.glt_torsion_observation_value, torch.tensor([0.25])
    ])
    gradient_batch.glt_torsion_observation_relation = torch.cat([
        gradient_batch.glt_torsion_observation_relation, torch.tensor([0])
    ])
    gradient_batch.glt_relation_torsion_count[0] += 1
    module = TorsionRelationBiasV2(mode='full')
    bias, _ = module(gradient_batch, int(gradient_batch.glt_relation_source.numel()))
    (bias * torch.randn_like(bias)).sum().backward()
    gradients = {
        name: float(parameter.grad.abs().sum())
        for name, parameter in (
            ('mean_projection', module.mean_projection.weight),
            ('variance_projection', module.variance_projection.weight),
            ('count_embedding', module.count_embedding.weight),
        )
    }
    result = {
        'schema': 'mts-glt-v2-torsion-geometry-sanity-v1',
        'geometry_invariance_pass': bool(geometry_invariance),
        'canonicalization_pass': bool(canonicalization),
        'reflection_invariant_value': float(expected),
        'near_collinear_invalid_pass': reflection_invariant_dihedral_cosine(
            *torch.tensor([[0., 0., 0.], [1., 0., 0.], [2., 0., 0.], [3., 1., 0.]])
        ) is None,
        'step0_parity_pass': bool(parity),
        'angle_bias_unchanged_pass': bool(angle_unchanged),
        'first_gradient_pass': all(value > 0 for value in gradients.values()),
        'gradient_abs_sums': gradients,
        'new_parameter_count_tc': parameter_counts['count'],
        'new_parameter_count_tg': parameter_counts['full'],
        'parameter_count_equal': parameter_counts['count'] == parameter_counts['full'],
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.tmp.{os.getpid()}')
    try:
        tmp.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
        os.replace(tmp, path)
    finally:
        if tmp.exists(): tmp.unlink()
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
