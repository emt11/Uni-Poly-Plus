import importlib.util
from pathlib import Path

import torch

from src.dataset.dataloader import mips_trimer_collate
from src.dataset.periodic_line_torsion import (
    build_periodic_torsion_fields,
    canonical_torsion_quadruplet,
    reflection_invariant_dihedral_cosine,
)
from src.modules.mts_glt_v2 import MTSGraphLineModelV2
from src.modules.periodic_line_glt_v2 import TorsionRelationBiasV2


def _fixture():
    path = Path(__file__).with_name('test_mts_periodic_line_glt.py')
    spec = importlib.util.spec_from_file_location('glt_torsion_fixture', path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _batch():
    fixture = _fixture()
    samples = [fixture._sample('*CCO*'), fixture._sample('*CCCC*')]
    records = [
        fixture.build_periodic_line_sample(
            fixture.sample_key_from_smiles(smiles), sample, sample
        )
        for smiles, sample in zip(('*CCO*', '*CCCC*'), samples)
    ]
    attached = [
        build_periodic_torsion_fields(fixture._attach(sample, record))
        for sample, record in zip(samples, records)
    ]
    return mips_trimer_collate(attached)


def test_torsion_cosine_is_rigid_and_reflection_invariant():
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
    assert abs(reflection_invariant_dihedral_cosine(*transformed) - expected) < 1e-6
    assert abs(reflection_invariant_dihedral_cosine(*reflected) - expected) < 1e-6
    assert abs(reflection_invariant_dihedral_cosine(*reversed(points)) - expected) < 1e-6


def test_near_collinear_torsion_is_invalid_and_reversal_key_matches():
    points = torch.tensor([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0], [3.0, 1.0, 0.0],
    ])
    assert reflection_invariant_dihedral_cosine(*points) is None
    atoms = ((1, -1), (2, 0), (3, 0), (4, 1))
    shifted = tuple((atom, offset + 1) for atom, offset in atoms)
    assert canonical_torsion_quadruplet(atoms) == canonical_torsion_quadruplet(shifted)
    assert canonical_torsion_quadruplet(atoms) == canonical_torsion_quadruplet(tuple(reversed(atoms)))


def test_torsion_builder_and_collate_are_relation_aligned():
    batch = _batch()
    assert batch.glt_torsion_observation_value.numel() > 0
    assert int(batch.glt_torsion_observation_relation.max()) < int(
        batch.glt_relation_source.numel()
    )
    reconstructed = torch.bincount(
        batch.glt_torsion_observation_relation,
        minlength=batch.glt_relation_source.numel(),
    )
    assert torch.equal(reconstructed, batch.glt_relation_torsion_count)
    assert torch.isfinite(batch.glt_torsion_observation_value).all()


def test_tc_tg_are_parameter_matched_zero_initialized_and_tg_has_gradient():
    torch.manual_seed(7)
    tc = TorsionRelationBiasV2(basis_size=16, heads=4, mode='count')
    torch.manual_seed(7)
    tg = TorsionRelationBiasV2(basis_size=16, heads=4, mode='full')
    assert sum(p.numel() for p in tc.parameters()) == sum(p.numel() for p in tg.parameters())
    for module in (tc, tg):
        assert torch.equal(module.mean_projection.weight, torch.zeros_like(module.mean_projection.weight))
        assert torch.equal(module.variance_projection.weight, torch.zeros_like(module.variance_projection.weight))
        assert torch.equal(module.count_embedding.weight, torch.zeros_like(module.count_embedding.weight))
    batch = _batch()
    batch.glt_torsion_observation_value = torch.cat([
        batch.glt_torsion_observation_value, torch.tensor([0.25])
    ])
    batch.glt_torsion_observation_relation = torch.cat([
        batch.glt_torsion_observation_relation, torch.tensor([0])
    ])
    batch.glt_relation_torsion_count[0] += 1
    bias, _ = tg(batch, int(batch.glt_relation_source.numel()))
    assert torch.equal(bias, torch.zeros_like(bias))
    coefficients = torch.randn_like(bias)
    (bias * coefficients).sum().backward()
    for parameter in (
        tg.mean_projection.weight, tg.variance_projection.weight,
        tg.count_embedding.weight,
    ):
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert float(parameter.grad.abs().sum()) > 0


def test_real_batch_step0_b_tc_tg_parity_and_angle_bias_unchanged():
    batch = _batch()
    torch.manual_seed(23)
    baseline = MTSGraphLineModelV2(glt_layers=1).eval()
    baseline_state = baseline.state_dict()
    with torch.no_grad():
        expected = baseline.forward_downstream(batch, 'o8_glt_atom')
        _, _, base_bias, _ = baseline.glt._relations(batch, torch.float32)
    for torsion_mode, downstream in (
        ('count', 'o8_glt_atom_torsion_count'),
        ('full', 'o8_glt_atom_torsion'),
    ):
        candidate = MTSGraphLineModelV2(
            glt_layers=1, torsion_mode=torsion_mode
        ).eval()
        merged = candidate.state_dict()
        merged.update({key: value for key, value in baseline_state.items() if key in merged})
        candidate.load_state_dict(merged, strict=True)
        with torch.no_grad():
            actual = candidate.forward_downstream(batch, downstream)
            _, _, relation_bias, views = candidate.glt._relations(
                batch, torch.float32, return_torsion=True
            )
        assert torch.equal(expected, actual)
        assert torch.equal(base_bias, relation_bias)
        real_count = int(views['angle_bias'].size(0))
        assert torch.equal(base_bias[:real_count], views['angle_bias'])
