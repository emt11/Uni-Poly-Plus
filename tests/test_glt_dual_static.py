import copy

import numpy as np
import torch

from test_complete_trimer_glt import _toy_pair
from src.dataset.canonical_periodic import build_canonical_periodic_topology
from src.dataset.glt_dual import build_dual_sample
from src.dataset.glt_dual_pretrain import prepare_pretrain_sample
from src.dataset.glt_dual_static import (
    _projected_position_index, build_dual_static, build_pretrain_target,
    materialize_dual_geometry,
)


def test_heavy_indices_are_original_coordinate_indices():
    class Trimer:
        trimer_pos = torch.zeros(3, 3)
        trimer_base_ru_atom_id = torch.tensor([99, 0, 1])
        trimer_ru_offset = torch.tensor([-1, 0, 0])
        trimer_heavy_indices = torch.tensor([1, 2])

    assert _projected_position_index(Trimer()) == {(0, 0): 1, (1, 0): 2}


def test_static_materialization_matches_reference_and_has_no_clean_geometry():
    smiles = '*COC*'
    topology = build_canonical_periodic_topology(smiles)
    _, trimer = _toy_pair(smiles)
    static = build_dual_static(topology, trimer, smiles)
    assert 'token_distance' not in static and 'line_angle' not in static
    reference = build_dual_sample(topology, trimer, smiles)
    cached = build_dual_sample(topology, trimer, smiles, static=static)
    for name in (
        'bond_path_features', 'bond_path_mask', 'bond_z_a', 'bond_z_b',
        'bond_distance', 'bond_type', 'bond_center', 'line_source',
        'line_target', 'line_path', 'line_angle', 'line_mask',
        'line_path_group', 'line_is_self',
    ):
        torch.testing.assert_close(getattr(reference, name), getattr(cached, name))
    noisy = copy.copy(trimer)
    noisy.trimer_pos = trimer.trimer_pos + 0.03 * torch.ones_like(trimer.trimer_pos)
    dynamic = materialize_dual_geometry(static, noisy.trimer_pos)
    expected = build_dual_sample(topology, noisy, smiles)
    torch.testing.assert_close(dynamic['bond_distance'], expected.bond_distance)
    torch.testing.assert_close(dynamic['line_angle'], expected.line_angle)


def test_cached_pretrain_target_replays_fixed_noise():
    smiles = '*COC*'
    topology = build_canonical_periodic_topology(smiles)
    _, trimer = _toy_pair(smiles)
    static = build_dual_static(topology, trimer, smiles)
    target = build_pretrain_target(smiles)
    dynamic, labels = prepare_pretrain_sample(
        topology, trimer, smiles, seed=41, key='key', position=9,
        static=static, target=target,
    )
    reference, expected = prepare_pretrain_sample(
        topology, trimer, smiles, seed=41, key='key', position=9,
    )
    for name in dynamic.keys():
        left, right = getattr(dynamic, name), getattr(reference, name)
        if torch.is_tensor(left):
            torch.testing.assert_close(left, right)
        else:
            assert left == right
    for name in labels.keys():
        left, right = labels[name], expected[name]
        if torch.is_tensor(left):
            torch.testing.assert_close(left, right)
        else:
            assert left == right
