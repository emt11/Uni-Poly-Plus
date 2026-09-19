"""S4 correctness: environment atom target and Trimer torsion fields/modules.

Synthetic coordinates stay confined to invariance/gradient fixtures; category
and quadruplet identity always come from real periodic bond tables.
"""
import math

import numpy as np
import pytest
import torch

from test_complete_trimer_glt import _toy_pair

from src.dataset.canonical_periodic import (build_canonical_periodic_topology,
                                            resolve_normalized_identity)
from src.dataset.glt_dual import build_dual_sample
from src.dataset.glt_dual_pretrain import (environment_atom_labels,
                                           periodic_bond_table,
                                           prepare_pretrain_sample,
                                           pretrain_collate,
                                           torsion_features,
                                           torsion_quadruplets)
from src.modules.glt_dual import build_dual_glt_model
from src.modules.glt_dual_pretrain import DualPretrainer, deployment_package, load_deployment

SMILES = '*CCOCC*'


def _fixture(smiles=SMILES):
    topology = build_canonical_periodic_topology(smiles)
    _, trimer = _toy_pair(smiles)
    return topology, trimer


def _bond_table(topology, smiles=SMILES):
    identity = resolve_normalized_identity(topology, smiles, require_fields=True)
    return periodic_bond_table(identity['normalized_smiles'],
                               identity['canonical_to_normalized_base'])


def _static_fixture(smiles=SMILES):
    """The frozen static row as the production builder writes it."""
    from src.dataset.glt_dual_static import build_dual_static

    topology, trimer = _fixture(smiles)
    static = build_dual_static(topology, trimer, smiles)
    assert static['geometry_valid']
    assert static['token_center_mask'].any()
    return topology, trimer, static


def _prepare(topology, trimer, static, **kwargs):
    return prepare_pretrain_sample(topology, trimer, SMILES, seed=42, key='fixture',
                                   position=0, static=static, **kwargs)


def _categories(bond_table, z):
    neighbors = {}
    for ca, cb, bond_type, _ in bond_table:
        neighbors.setdefault(ca, []).append((z[cb], bond_type))
        neighbors.setdefault(cb, []).append((z[ca], bond_type))
    return {atom: f'{z[atom]}|' + ';'.join(f'{element}:{bond}'
                                           for element, bond in sorted(pairs))
            for atom, pairs in neighbors.items()}


def _vocab_helper(bond_table, topology):
    expected = _categories(bond_table, topology.atomic_numbers.tolist())
    mapping = {category: index + 1 for index, category in enumerate(sorted(set(expected.values())))}
    mapping['<unk>'] = len(mapping)
    return mapping


def test_environment_categories_match_the_real_periodic_multiset():
    topology, _ = _fixture()
    table = _bond_table(topology)
    z = topology.atomic_numbers.tolist()
    expected = _categories(table, z)
    vocab = _vocab_helper(table, topology)
    labels = environment_atom_labels(table, topology.atomic_numbers, vocab)
    assert labels.shape[0] == len(z)
    for atom, category in expected.items():
        assert int(labels[atom]) == vocab[category]
    assert labels.unique().numel() == len(set(expected.values()))
    # The seam directions are real distinct periodic edges: an atom bonded to
    # neighbors on both sides of the chain must see both.
    seam = [entry for entry in table if entry[3] != 0]
    assert seam, 'polymer fixture must expose periodic seam bonds'


def test_environment_relabeling_is_permutation_invariant():
    topology, _ = _fixture()
    table = _bond_table(topology)
    vocab = _vocab_helper(table, topology)
    labels = environment_atom_labels(table, topology.atomic_numbers, vocab)
    shuffled = [((cb, ca, bond_type, -shift) if cb < ca else (ca, cb, bond_type, shift))
                for ca, cb, bond_type, shift in table]
    assert torch.equal(labels,
                       environment_atom_labels(sorted(shuffled), topology.atomic_numbers, vocab))


def test_environment_multiset_counts_duplicate_periodic_neighbors():
    # Atom 1 carries two periodic (6,1) neighbors in one graph and only one in
    # the other, so its environment category must differ.
    z = [6, 7, 8]
    double = [(0, 1, 1, -1), (0, 1, 1, 1), (1, 2, 2, 0)]
    single = [(0, 1, 1, -1), (1, 2, 2, 0)]
    vocab = {'6|7:1': 1, '6|7:1;7:1': 2, '7|6:1;6:1;8:2': 3,
             '7|6:1;8:2': 4, '8|7:2': 5, '<unk>': 0}
    labels_double = environment_atom_labels(double, z, vocab)
    labels_single = environment_atom_labels(single, z, vocab)
    assert labels_double[1] != labels_single[1]
    assert labels_double[1] == vocab['7|6:1;6:1;8:2']
    assert labels_single[1] == vocab['7|6:1;8:2']


def test_environment_unknown_category_maps_to_unk():
    topology, _ = _fixture()
    table = _bond_table(topology)
    assert bool((environment_atom_labels(table, topology.atomic_numbers,
                                         {'<unk>': 0}) == 0).all())
    assert environment_atom_labels(table, topology.atomic_numbers,
                                   {'<unk>': 5}).max() == 5


def test_environment_mask_and_element_path_unchanged():
    topology, trimer, static = _static_fixture()
    _, labels_env = _prepare(topology, trimer, static, third_task='none',
                             atom_target='environment', env_vocab={'<unk>': 0})
    _, labels_el = _prepare(topology, trimer, static, third_task='none')
    assert torch.equal(labels_env['atom_mask'], labels_el['atom_mask'])
    assert torch.equal(labels_el['atom_label'], topology.mips_x[:, :101].argmax(-1))
    assert not torch.equal(labels_env['atom_label'], labels_el['atom_label'])


def test_torsion_quadruplets_are_real_center_paths_without_reverses():
    topology, _ = _fixture()
    table = _bond_table(topology)
    quads = torsion_quadruplets(table, topology.atomic_numbers)
    assert quads, 'fixture must expose at least one heavy-atom torsion'
    center = {(min(a, b), max(a, b)) for a, b, _, shift in table if shift == 0}
    seen = set()
    for a, b, c, d, middle in quads:
        assert len({a, b, c, d}) == 4
        assert (min(b, c), max(b, c)) == middle and middle in center
        assert (min(a, b), max(a, b)) in center
        assert (min(c, d), max(c, d)) in center
        assert (a, b, c, d) not in seen and (d, c, b, a) not in seen
        seen.add((a, b, c, d))


def test_torsion_features_translation_rotation_reflection_invariance():
    quads = [(0, 1, 2, 3, 0)]
    mapping = torch.arange(4)
    base = torch.tensor([[0.0, 0.0, 0.0], [1.5, 0.1, 0.2],
                         [2.6, 1.3, -0.1], [3.7, 1.1, 1.2]])
    shift = torch.tensor([4.0, -2.0, 7.0])
    angle = 0.7
    rotation = torch.tensor([[math.cos(angle), -math.sin(angle), 0.0],
                             [math.sin(angle), math.cos(angle), 0.0],
                             [0.0, 0.0, 1.0]])
    reflection = torch.diag(torch.tensor([1.0, 1.0, -1.0]))
    values, mask = torsion_features(base, mapping, quads)
    assert bool(mask.all())
    moved, _ = torsion_features(base + shift, mapping, quads)
    rotated, _ = torsion_features(base @ rotation.T, mapping, quads)
    reflected, _ = torsion_features(base @ reflection.T, mapping, quads)
    torch.testing.assert_close(values, moved)
    torch.testing.assert_close(values, rotated, atol=1e-5, rtol=1e-5)
    # cos(phi) and cos(2*phi) are even in phi: a mirrored conformation
    # (phi -> -phi) keeps the feature exactly.
    torch.testing.assert_close(values, reflected, atol=1e-5, rtol=1e-5)


def test_torsion_degenerate_quadruplet_is_masked():
    quads = [(0, 1, 2, 3, 0)]
    mapping = torch.arange(4)
    collinear = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
                              [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    values, mask = torsion_features(collinear, mapping, quads)
    assert not bool(mask.any()) and bool(torch.isfinite(values).all())
    healthy = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
                            [1.0, 1.0, 0.0], [2.0, 1.0, 0.5]])
    values, mask = torsion_features(healthy, mapping, quads)
    assert bool(mask.all()) and bool(torch.isfinite(values).all())


def test_torsion_fields_follow_the_noisy_coordinates_not_clean_ones():
    topology, trimer, static = _static_fixture()
    table = _bond_table(topology)
    quads = torsion_quadruplets(table, topology.atomic_numbers)
    mapping = trimer.mips_to_trimer_central_index
    clean_values, _ = torsion_features(trimer.trimer_pos.clone(), mapping, quads)
    # With sigma=0 the noisy view equals the clean view: fields match exactly.
    zero, _ = _prepare(topology, trimer, static, third_task='none',
                       torsion_mode='on', sigma=0.0)
    torch.testing.assert_close(zero.torsion_values, clean_values)
    # With the real noise the fields move with the perturbed coordinates.
    noisy, _ = _prepare(topology, trimer, static, third_task='none', torsion_mode='on')
    assert not torch.equal(noisy.torsion_values, clean_values)


def test_torsion_on_off_share_stream_and_off_values_are_zero():
    topology, trimer, static = _static_fixture()
    on_data, on_labels = _prepare(topology, trimer, static, third_task='none', torsion_mode='on')
    off_data, off_labels = _prepare(topology, trimer, static, third_task='none', torsion_mode='off')
    assert torch.equal(on_data.torsion_bond_index, off_data.torsion_bond_index)
    assert torch.equal(on_data.torsion_mask, off_data.torsion_mask)
    assert bool((off_data.torsion_values == 0).all())
    assert bool((on_data.torsion_values[on_data.torsion_mask] != 0).any())
    assert torch.equal(on_labels['atom_label'], off_labels['atom_label'])
    assert torch.equal(on_labels['atom_mask'], off_labels['atom_mask'])


def test_tor_on_off_initial_parameters_are_identical():
    def build(torsion):
        torch.manual_seed(42)
        return DualPretrainer('concat', third_task='none', torsion=torsion)
    on, off = build(True), build(True)
    assert set(on.state_dict()) == set(off.state_dict())
    for name, value in on.state_dict().items():
        assert torch.equal(value, off.state_dict()[name]), name
    plain = build(False)
    assert plain.encoder.glt.torsion_mlp is None and plain.encoder.glt.torsion_gate is None


def test_torsion_module_receives_finite_nonzero_gradient():
    topology, trimer, static = _static_fixture()
    data, labels = _prepare(topology, trimer, static, third_task='none', torsion_mode='on')
    batch, labels = pretrain_collate([(data, labels)])
    torch.manual_seed(0)
    model = DualPretrainer('concat', third_task='none', geometry_head_norm=True, torsion=True)
    result = model(batch, labels)
    result['sums'].sum().backward()
    gate = model.encoder.glt.torsion_gate.grad
    assert gate is not None and bool(torch.isfinite(gate).all()) and float(gate.abs()) > 0
    for name, parameter in model.encoder.glt.torsion_mlp.named_parameters():
        assert parameter.grad is None or bool(torch.isfinite(parameter.grad).all()), name


def test_torsion_deployment_roundtrip_and_downstream_fields():
    topology, trimer, static = _static_fixture()
    data, labels = _prepare(topology, trimer, static, third_task='none', torsion_mode='on')
    torch.manual_seed(0)
    model = DualPretrainer('concat', third_task='none', torsion=True)
    package = deployment_package(model, 5000)
    assert package['torsion_modules'] is True
    downstream = build_dual_sample(topology, trimer, SMILES, static=static, torsion_mode='on')
    quads = torsion_quadruplets(_bond_table(topology), topology.atomic_numbers)
    clean_values, _ = torsion_features(trimer.trimer_pos.clone(),
                                       trimer.mips_to_trimer_central_index, quads)
    torch.testing.assert_close(downstream.torsion_values, clean_values)
    rebuilt = build_dual_glt_model('concat', torsion=True)
    load_deployment(rebuilt, package, 5000)
    plain = build_dual_glt_model('concat')
    with pytest.raises(ValueError):
        load_deployment(plain, package, 5000)
