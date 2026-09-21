"""Structural contract tests for the MCL-PH frozen-Trimer view (sections 3, 5.3, 6, 7).

The synthetic fixture is a three-repeat-unit chain with two centre-internal
bonds per unit and real seam bonds, so the centre mapping, the cross-repeat-unit
synchronous mask and the three nested expert graphs can all be checked without
opening a frozen cache.  One additional test reads a single real frozen record
to confirm the declared provenance of the charge/aromaticity fields; it is
skipped (and reported as skipped, never as covered) when the cache is absent.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest
import torch
from torch_geometric.data import Data

from src.dataset import mcl_ph_view as view

BASE_COUNT = 3
REPEATS = 3
ATOM_COUNT = BASE_COUNT * REPEATS


def _positions():
    """Three repeat units on a gently bent chain, 1.5 A apart inside a unit."""
    offsets = [(0.0, 0.0, 0.0), (1.5, 0.0, 0.0), (3.0, 0.5, 0.0)]
    rows = []
    for repeat in range(REPEATS):
        for offset in offsets:
            rows.append((offset[0] + 4.5 * repeat, offset[1], offset[2]))
    return np.asarray(rows, dtype=np.float64)


def make_trimer():
    numbers, base, offset = [], [], []
    for repeat in range(REPEATS):
        for atom in range(BASE_COUNT):
            numbers.append(6 if atom != 1 else 8)
            base.append(atom)
            offset.append(repeat)
    edges, codes = [], []
    for left in range(ATOM_COUNT):
        for right in range(left + 1, ATOM_COUNT):
            same_unit = offset[left] == offset[right]
            neighbours = (right - left == 1) and same_unit
            seam = (right - left == 1) and (offset[right] - offset[left] == 1)
            if neighbours or seam:
                edges.extend([(left, right), (right, left)])
                codes.extend([1, 1])
    trimer = Data()
    trimer.trimer_pos = torch.tensor(_positions(), dtype=torch.float32)
    trimer.trimer_atomic_number = torch.tensor(numbers, dtype=torch.long)
    trimer.trimer_formal_charge = torch.tensor(
        [-1 if atom == 2 else 0 for atom in base], dtype=torch.long)
    trimer.trimer_is_aromatic = torch.tensor([atom == 0 for atom in base], dtype=torch.bool)
    trimer.trimer_base_ru_atom_id = torch.tensor(base, dtype=torch.long)
    trimer.trimer_ru_offset = torch.tensor(offset, dtype=torch.long)
    trimer.trimer_heavy_mask = torch.ones(ATOM_COUNT, dtype=torch.bool)
    trimer.trimer_heavy_indices = torch.arange(ATOM_COUNT, dtype=torch.long)
    trimer.trimer_central_ru_mask = torch.tensor([value == 0 for value in offset],
                                                 dtype=torch.bool)
    trimer.trimer_edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    trimer.trimer_bond_type = torch.tensor(codes, dtype=torch.long)
    trimer.trimer_chiral_tag = torch.zeros(ATOM_COUNT, dtype=torch.long)
    # canonical atom c is the centre repeat unit's atom c
    trimer.mips_to_trimer_central_index = torch.arange(BASE_COUNT, dtype=torch.long)
    trimer.trimer_geometry_valid = True
    return trimer


def make_topology():
    topology = Data()
    topology.z = torch.tensor([6, 8, 6], dtype=torch.long)
    topology.mips_x = torch.zeros((BASE_COUNT, 137))
    topology.mips_x[torch.arange(BASE_COUNT), torch.tensor([5, 7, 5])] = 1.0
    topology.canonical_to_trimer_base_atom_id = torch.arange(BASE_COUNT, dtype=torch.long)
    return topology


def make_static(trimer):
    """Minimal frozen static row: the centre bonds then both seam classes."""
    bonds = [(0, 1), (1, 2), (2, 3), (5, 6)]
    centers = [True, True, False, False]
    static = {
        'token_pos_index_a': np.asarray([a for a, _ in bonds], dtype=np.int32),
        'token_pos_index_b': np.asarray([b for _, b in bonds], dtype=np.int32),
        'token_bond_type': np.asarray([0, 1, 0, 0], dtype=np.uint8),
        'token_center_mask': np.asarray(centers, dtype=bool),
        'geometry_valid': True,
        'geometry_invalid_reason': '',
        'line_path': np.asarray([[0, 1, -1], [1, 2, -1], [2, 3, -1], [3, 1, -1],
                                 [0, 0, -1], [1, 1, -1], [2, 2, -1], [3, 3, -1]],
                                dtype=np.int32),
        'line_path_mask': np.asarray([[True, False]] * 8, dtype=bool),
        'line_is_self': np.asarray([False, False, False, False, True, True, True, True]),
        'angle_pos_triplet': _angle_triplets(),
        'angle_pairs': np.asarray([[0, 1]], dtype=np.int32),
    }
    assert static['line_path'].shape[0] == static['line_path_mask'].shape[0]
    assert static['angle_pos_triplet'].shape[0] == static['line_path'].shape[0]
    del trimer
    return static


def _angle_triplets():
    """Only the first line row carries a real one-hop angle, at atom 1."""
    triplets = np.full((8, 2, 3), -1, dtype=np.int32)
    triplets[0, 0] = (0, 1, 2)
    return triplets


def make_view(trimer):
    return view.build_trimer_view(trimer)


# ---------------------------------------------------------------------------
# Section 3 -- heavy view, mapping and physical bonds
# ---------------------------------------------------------------------------

def test_heavy_view_enumerates_every_real_heavy_atom_once():
    trimer = make_trimer()
    heavy = make_view(trimer)
    assert heavy.count == ATOM_COUNT
    assert heavy.z.tolist() == [6, 8, 6] * REPEATS
    assert int(heavy.charge.sum()) == -REPEATS
    assert int(heavy.aromatic.sum()) == REPEATS


def test_physical_bonds_are_undirected_and_counted_once():
    trimer = make_trimer()
    heavy = make_view(trimer)
    pairs = [tuple(pair) for pair in heavy.bond_ends.tolist()]
    assert len(pairs) == len(set(pairs)), 'a physical undirected bond is stored once'
    assert all(a < b for a, b in pairs), 'each row is stored min-first'
    # two bonds per repeat unit plus two seams
    assert len(pairs) == 2 * REPEATS + (REPEATS - 1)


def test_centre_mapping_is_a_complete_injection_onto_the_central_unit():
    trimer = make_trimer()
    topology = make_topology()
    heavy = make_view(trimer)
    central, image, unmapped = view.centre_mapping(topology, trimer, heavy)
    assert unmapped == 0
    assert central.tolist() == [0, 1, 2]
    assert image[:3].tolist() == [0, 1, 2]
    assert (image[3:] == -1).all(), 'side repeat units have no canonical counterpart'


def test_centre_mapping_rejects_a_missing_canonical_entry():
    trimer = make_trimer()
    topology = make_topology()
    topology.canonical_to_trimer_base_atom_id = torch.tensor([0, 1, 1], dtype=torch.long)
    with pytest.raises(ValueError):
        view.centre_mapping(topology, trimer, make_view(trimer))


def test_centre_mapping_rejects_an_element_disagreement():
    trimer = make_trimer()
    trimer.trimer_atomic_number[1] = 7
    topology = make_topology()
    with pytest.raises(ValueError):
        view.centre_mapping(topology, trimer, make_view(trimer))


# ---------------------------------------------------------------------------
# Section 3.3 -- nested expert relations
# ---------------------------------------------------------------------------

def test_expert_graphs_are_nested_and_stored_in_both_directions():
    trimer = make_trimer()
    heavy = make_view(trimer)
    index, scale, distance = view.expert_relations(heavy.positions)
    assert set(scale.tolist()) <= {0, 1, 2}
    previous = 0
    for slot, cutoff in enumerate(view.EXPERT_CUTOFFS):
        selected = scale == slot
        pairs = int(selected.sum()) // 2
        assert pairs >= previous
        previous = pairs
        assert bool((distance[selected] <= float(cutoff) + 1e-6).all())
        left = index[0, selected]
        right = index[1, selected]
        assert sorted(torch.cat([left, right]).tolist()) == \
            sorted(torch.cat([right, left]).tolist()), 'both directions are stored'
    assert int((distance > view.EXPERT_CUTOFFS[-1]).sum()) == 0


def test_expert_graph_never_connects_two_different_samples():
    trimer = make_trimer()
    heavy = make_view(trimer)
    index, scale, _ = view.expert_relations(heavy.positions)
    assert int(index.max()) < heavy.count
    assert int(index.min()) >= 0
    assert int((index[0] == index[1]).sum()) == 0


def test_cutoff_envelope_is_zero_at_the_boundary():
    from src.modules.mcl_ph import cutoff_envelope
    inside = cutoff_envelope(torch.tensor([0.5, 1.0]), 2.0)
    assert float(inside[0]) > 0
    assert float(cutoff_envelope(torch.tensor([2.0]), 2.0)[0]) == 0.0
    assert float(cutoff_envelope(torch.tensor([2.5]), 2.0)[0]) == 0.0


def test_bond_category_uses_the_frozen_chemical_code_and_marks_spatial_edges():
    trimer = make_trimer()
    heavy = make_view(trimer)
    static = make_static(trimer)
    index, bond_type, _ = view.physical_bonds(static, heavy)
    assert bond_type.tolist() == [0, 1, 0, 0]
    ends = torch.tensor([[0, 1], [0, 2], [3, 4]], dtype=torch.long)
    category = view._bond_category(ends, index, bond_type)
    assert category.tolist() == [0, view.NONBONDED_CATEGORY, view.NONBONDED_CATEGORY]


def test_rbf_basis_matches_the_declared_centres_and_width():
    distance = torch.tensor([0.0, 4.0])
    basis = view.rbf_features(distance)
    assert tuple(basis.shape) == (2, 64)
    assert float(basis[0, 0]) == pytest.approx(1.0)
    assert float(basis[1, -1]) == pytest.approx(1.0)
    assert float(basis[0, -1]) < 1e-6


# ---------------------------------------------------------------------------
# Section 2/6 -- synchronous cross-repeat-unit masking
# ---------------------------------------------------------------------------

def test_mask_is_synchronised_over_every_provable_copy():
    trimer = make_trimer()
    topology = make_topology()
    heavy = make_view(trimer)
    canonical_mask = torch.tensor([True, False, False])
    element, charge, aromatic, masked = view.synchronised_mask(topology, heavy, canonical_mask)
    # base identity 0 exists once per repeat unit
    assert masked.tolist() == [True, False, False] * REPEATS
    assert element[masked].tolist() == [view.ELEMENT_MASK] * REPEATS
    assert charge[masked].tolist() == [view.CHARGE_MASK] * REPEATS
    assert aromatic[masked].tolist() == [view.AROMATIC_MASK] * REPEATS
    untouched = ~masked
    assert element[untouched].tolist() == view.element_categories(heavy.z)[untouched].tolist()


def test_mask_leaves_the_geometry_view_untouched():
    trimer = make_trimer()
    topology = make_topology()
    heavy = make_view(trimer)
    before = heavy.positions.clone()
    view.synchronised_mask(topology, heavy, torch.tensor([True, False, False]))
    assert torch.equal(before, heavy.positions)


def test_mask_categories_stay_strictly_distinct_from_unknown():
    assert view.ELEMENT_MASK != view.ELEMENT_UNKNOWN
    assert view.CHARGE_MASK != view.CHARGE_OTHER
    unknown = view.element_categories(torch.tensor([0, 119, 200]))
    assert unknown.tolist() == [view.ELEMENT_UNKNOWN] * 3
    assert view.charge_categories(torch.tensor([9, -9])).tolist() == \
        [view.CHARGE_OTHER, view.CHARGE_OTHER]
    assert view.charge_categories(torch.tensor([-3, 0, 3])).tolist() == [0, 3, 6]


# ---------------------------------------------------------------------------
# Section 7 -- targets
# ---------------------------------------------------------------------------

def test_local_targets_drop_a_graph_without_centre_bonds():
    trimer = make_trimer()
    heavy = make_view(trimer)
    static = make_static(trimer)
    index, _, _ = view.physical_bonds(static, heavy)
    static_no_center = dict(static, token_center_mask=np.zeros(4, dtype=bool),
                            angle_pairs=np.empty((0, 2), np.int32))
    lengths, raw, angles, cosines = view.local_targets(static_no_center, heavy.positions, index)
    assert lengths.numel() == 0 and raw.numel() == 0
    assert angles.numel() == 0 and cosines.numel() == 0


def test_local_targets_keep_lengths_when_no_angle_exists():
    trimer = make_trimer()
    heavy = make_view(trimer)
    static = make_static(trimer)
    index, _, _ = view.physical_bonds(static, heavy)
    # A line table that genuinely has no one-hop relation, so the frozen
    # consistency check between the angle table and the line table still holds.
    static_no_angle = dict(
        static, angle_pairs=np.empty((0, 2), np.int32),
        line_path_mask=np.zeros((8, 2), dtype=bool),
        line_is_self=np.ones(8, dtype=bool),
        angle_pos_triplet=np.full((8, 2, 3), -1, dtype=np.int32))
    lengths, raw, angles, cosines = view.local_targets(static_no_angle, heavy.positions, index)
    assert lengths.tolist() == [[0, 1], [1, 2]]
    assert raw.numel() == 2
    assert angles.numel() == 0 and cosines.numel() == 0


def test_local_targets_measure_the_clean_coordinates():
    trimer = make_trimer()
    heavy = make_view(trimer)
    static = make_static(trimer)
    index, _, _ = view.physical_bonds(static, heavy)
    lengths, raw, angles, cosines = view.local_targets(static, heavy.positions, index)
    assert raw.tolist() == pytest.approx([1.5, np.sqrt(2.5)])
    assert lengths.tolist() == [[0, 1], [1, 2]]
    assert angles.tolist() == [[0, 1]]
    expected = -1.5 * 1.5 / (1.5 * float(np.sqrt(2.5)))
    assert float(cosines[0]) == pytest.approx(expected, abs=1e-6)


def test_nonbond_candidates_require_a_centre_endpoint_and_exclude_bonds():
    trimer = make_trimer()
    heavy = make_view(trimer)
    static = make_static(trimer)
    index, _, _ = view.physical_bonds(static, heavy)
    central = torch.arange(3, dtype=torch.long)
    pairs, distance = view.nonbond_candidates(heavy.positions, heavy, central, index)
    bonded = {tuple(sorted(pair)) for pair in index.tolist()}
    for left, right in pairs.tolist():
        assert left in central.tolist() or right in central.tolist()
        assert tuple(sorted((left, right))) not in bonded
    assert bool((distance > 0).all())
    assert bool((distance <= view.EXPERT_CUTOFFS[-1]).all())


def test_nonbond_sampling_is_bounded_and_binned():
    pairs = torch.tensor([[index, index + 1] for index in range(200)], dtype=torch.long)
    distance = torch.linspace(0.1, 3.9, 200)
    generator = torch.Generator().manual_seed(7)
    sampled, slots = view.sample_nonbond_pairs(pairs, distance, generator)
    counts = torch.bincount(slots, minlength=3)
    assert int(counts.max()) <= view.NONBOND_MAX_PAIRS
    assert int(sampled.size(0)) == int(counts.sum())
    lookup = {tuple(pair): float(value)
              for pair, value in zip(pairs.tolist(), distance.tolist())}
    for slot, (lower, upper) in enumerate(view.NONBOND_BINS):
        chosen = [lookup[tuple(pair)] for pair, value in zip(sampled.tolist(), slots.tolist())
                  if value == slot]
        if chosen:
            assert min(chosen) > float(lower)
            assert max(chosen) <= float(upper)


def test_nonbond_sampling_is_reproducible_for_one_sub_stream():
    pairs = torch.tensor([[index, index + 1] for index in range(60)], dtype=torch.long)
    distance = torch.linspace(0.1, 3.9, 60)
    first = view.sample_nonbond_pairs(pairs, distance, torch.Generator().manual_seed(11))
    second = view.sample_nonbond_pairs(pairs, distance, torch.Generator().manual_seed(11))
    assert torch.equal(first[0], second[0]) and torch.equal(first[1], second[1])


def test_geometric_statistics_use_log1p_and_a_positive_floor():
    record = view.geometric_statistics(torch.tensor([1.0, 1.0, 1.0]))
    assert record['sigma'] == pytest.approx(view.NORMALIZATION_STD_FLOOR)
    assert record['mu'] == pytest.approx(np.log1p(1.0))
    assert record['count'] == 3
    with pytest.raises(ValueError):
        view.geometric_statistics(torch.zeros(0))
    with pytest.raises(ValueError):
        view.normalise_geometric(torch.zeros(1), {'mu': 0.0, 'sigma': 0.0})


def test_normalisation_round_trips_through_the_shared_statistics():
    values = torch.tensor([1.0, 2.0, 3.0])
    record = view.geometric_statistics(values)
    normalized = view.normalise_geometric(values, record)
    restored = torch.expm1(normalized * record['sigma'] + record['mu'])
    assert torch.allclose(restored, values, atol=1e-6)


# ---------------------------------------------------------------------------
# Section 5.3 -- the reference view is isolated from the training stream
# ---------------------------------------------------------------------------

def test_reference_view_does_not_consume_the_training_stream():
    from src.dataset.glt_dual_pretrain import sample_generator

    trimer = make_trimer()
    other = make_trimer()
    key = 'ab' * 32
    before = sample_generator(42, key, 3).get_state()
    reference = view.reference_view(trimer, key, sigma=0.03)
    after = sample_generator(42, key, 3).get_state()
    assert torch.equal(before, after)
    assert not torch.equal(reference.trimer_pos, trimer.trimer_pos)
    repeat = view.reference_view(other, key, sigma=0.03)
    assert torch.equal(reference.trimer_pos, repeat.trimer_pos)


def test_reference_view_position_is_not_part_of_its_identity():
    trimer = make_trimer()
    key = 'cd' * 32
    first = view.reference_view(trimer, key, sigma=0.03)
    second = view.reference_view(trimer, key, sigma=0.03)
    assert torch.equal(first.trimer_pos, second.trimer_pos)


def test_labelled_sub_streams_are_distinct_and_stable():
    trimer = make_trimer()
    del trimer
    first = view.view_generator(42, 'ef' * 32, 1, view.PAIR_SUBSTREAM)
    second = view.view_generator(42, 'ef' * 32, 1, view.PAIR_SUBSTREAM)
    shared = view.view_generator(42, 'ef' * 32, 1)
    assert torch.equal(first.get_state(), second.get_state())
    assert not torch.equal(first.get_state(), shared.get_state())


# ---------------------------------------------------------------------------
# Provenance on one real frozen record
# ---------------------------------------------------------------------------

FROZEN = Path('data/processed/glt_dual_v2/pi1m/cohort_30f17b59bc5862a1')
CACHE = Path('data/processed/mips_trimer_scage')
STATIC_ROOT = Path('data/processed/glt_dual_v2/pi1m/dual_static_v1')
SPLIT = Path('results/glt_pred_20260918/s3b_prep/pretrain_split_v1.json')


@pytest.mark.skipif(not (FROZEN.is_dir() and CACHE.is_dir() and STATIC_ROOT.is_dir()
                         and SPLIT.is_file()),
                    reason='frozen P_train cache is not available in this checkout')
def test_one_real_record_keeps_the_frozen_identity_and_bond_provenance():
    import json as _json
    from src.training.glt_dual_runtime import (IndexedFrozenDualSource, load_sample_index_artifact,
                                               open_source)

    split = load_sample_index_artifact(str(SPLIT), 'train')
    source, _ = open_source(str(FROZEN), str(CACHE), dual_static_root=str(STATIC_ROOT))
    try:
        subset = IndexedFrozenDualSource(source, split['indices'][:8])
        checked = 0
        for index in range(len(subset)):
            topology, trimer, _ = subset[index]
            static = subset.static_for(index)
            if not bool(trimer.trimer_geometry_valid) or not bool(static['geometry_valid']):
                continue
            heavy = view.build_trimer_view(trimer)
            central, image, unmapped = view.centre_mapping(topology, trimer, heavy)
            assert unmapped == 0
            assert int(torch.unique(central).numel()) == int(central.numel())
            elements = torch.as_tensor(trimer.trimer_atomic_number).long()
            assert torch.equal(elements[torch.as_tensor(
                trimer.mips_to_trimer_central_index).long()], topology.z.long())
            index_pairs, bond_type, centre = view.physical_bonds(static, heavy)
            assert int(index_pairs.max()) < heavy.count
            assert int(bond_type.min()) >= 0 and int(bond_type.max()) <= 4
            assert bool(centre.any()), 'a real frozen record has centre-internal bonds'
            # charge and aromaticity are frozen record fields, never inferred.
            assert int(torch.as_tensor(trimer.trimer_formal_charge).numel()) == \
                int(torch.as_tensor(trimer.trimer_atomic_number).numel())
            assert bool(((heavy.charge >= -3) & (heavy.charge <= 3)).all())
            assert heavy.aromatic.dtype == torch.bool
            # every heavy copy of a canonical atom is the same base identity
            for value in torch.as_tensor(topology.canonical_to_trimer_base_atom_id).tolist():
                assert heavy.echo[int(value)] >= 1
            checked += 1
            if checked >= 2:
                break
        assert checked >= 1, 'no usable geometry in the sampled frozen records'
    finally:
        source.close()
