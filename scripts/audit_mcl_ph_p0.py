#!/usr/bin/env python3
"""P0 data / leakage / topology cost audit for MCL-PH-20260921-01 (r1).

Budget contract: CPU only, zero model calls, zero optimizer updates, at most
60 minutes.  The sample sets are fixed by the plan (the first 512 and the first
4096 P_train samples in raw 32-byte key order) and are never silently reduced:
if the wall-clock guard fires, the run writes the evidence it has with
``status=PARTIAL`` and a non-zero exit code instead of shrinking the set.

Everything here is read-only with respect to the frozen caches and the frozen
split artifacts; only ``--output`` is written.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from src.dataset import mcl_ph_view as view
from src.dataset.glt_dual_pretrain import chemical_groups, motif_mask, sample_generator
from src.training.glt_dual_runtime import (IndexedFrozenDualSource,
                                           load_sample_index_artifact, open_source)

DEFAULT_COHORT = 'data/processed/glt_dual_v2/pi1m/cohort_30f17b59bc5862a1'
DEFAULT_CACHE = 'data/processed/mips_trimer_scage'
DEFAULT_STATIC = 'data/processed/glt_dual_v2/pi1m/dual_static_v1'
DEFAULT_CONFIG = 'configs/mts/glt_pred_s3b_b_fp.json'
MAPPING_SAMPLES = 512
STATISTICS_SAMPLES = 4096
BUDGET_SECONDS = 60 * 60


# ---------------------------------------------------------------------------
# Independent reference implementations (deliberately not the production code)
# ---------------------------------------------------------------------------

def _rank_f2(matrix):
    """Rank over F2 by Gaussian elimination on a uint8 matrix."""
    rows = np.array(matrix, dtype=np.uint8, copy=True)
    if rows.size == 0:
        return 0
    row_count, column_count = rows.shape
    rank, pivot = 0, 0
    for column in range(column_count):
        selected = None
        for row in range(rank, row_count):
            if rows[row, column]:
                selected = row
                break
        if selected is None:
            continue
        if selected != rank:
            rows[[rank, selected]] = rows[[selected, rank]]
        for row in range(row_count):
            if row != rank and rows[row, column]:
                rows[row] ^= rows[rank]
        rank += 1
        if rank == row_count:
            break
    return rank


def betti_by_rank(points, radius):
    """Direct Betti numbers of the Vietoris-Rips complex at one radius.

    Independent of any persistence library: the two boundary matrices are built
    explicitly and reduced over F2, so this checks the persistence plumbing
    rather than re-running it.
    """
    cloud = np.asarray(points, dtype=np.float64)
    count = int(cloud.shape[0])
    distance = view.distance_matrix(cloud)
    edges = [(i, j) for i in range(count) for j in range(i + 1, count)
             if distance[i, j] <= radius]
    index = {pair: position for position, pair in enumerate(edges)}
    boundary_one = np.zeros((len(edges), count), dtype=np.uint8)
    for position, (i, j) in enumerate(edges):
        boundary_one[position, i] = 1
        boundary_one[position, j] = 1
    triangles = [(i, j, k)
                 for i in range(count) for j in range(i + 1, count)
                 for k in range(j + 1, count)
                 if distance[i, j] <= radius and distance[i, k] <= radius
                 and distance[j, k] <= radius]
    boundary_two = np.zeros((len(triangles), len(edges)), dtype=np.uint8)
    for position, (i, j, k) in enumerate(triangles):
        boundary_two[position, index[(i, j)]] = 1
        boundary_two[position, index[(i, k)]] = 1
        boundary_two[position, index[(j, k)]] = 1
    rank_one, rank_two = _rank_f2(boundary_one), _rank_f2(boundary_two)
    return count - rank_one, len(edges) - rank_one - rank_two


def wiener_by_networkx(points, radius):
    """Independent per-component Wiener index and efficiency for one radius."""
    import networkx

    cloud = np.asarray(points, dtype=np.float64)
    distance = view.distance_matrix(cloud)
    count = int(cloud.shape[0])
    graph = networkx.Graph()
    graph.add_nodes_from(range(count))
    for i in range(count):
        for j in range(i + 1, count):
            if distance[i, j] <= radius:
                graph.add_edge(i, j)
    hop = dict(networkx.all_pairs_shortest_path_length(graph))
    numerator = denominator = 0.0
    for component in networkx.connected_components(graph):
        members = sorted(component)
        size = len(members)
        weight = size * (size - 1) / 2.0
        denominator += weight
        if size < 3:
            continue
        within = sum(hop[a][b] for position, a in enumerate(members)
                     for b in members[position + 1:])
        span = (size ** 3 - size) / 6.0 - weight
        numerator += weight * ((within - weight) / span)
    wiener = 0.0 if denominator <= 0 else numerator / denominator
    total = 0.0
    for source in range(count):
        for target in range(count):
            if source != target and target in hop[source]:
                total += 1.0 / hop[source][target]
    efficiency = 0.0 if count <= 1 else total / (count * (count - 1))
    return wiener, efficiency


def ph_fixtures():
    """Small graphs the plan names explicitly, as (name, points) pairs."""
    root_two = math.sqrt(2.0)
    fixture = []
    fixture.append(('isolated_points', np.asarray(
        [[0.0, 0.0, 0.0], [5.5, 0.0, 0.0], [11.0, 0.0, 0.0]], dtype=np.float64)))
    fixture.append(('single_point', np.zeros((1, 3), dtype=np.float64)))
    fixture.append(('chain_tree', np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=np.float64)))
    fixture.append(('triangle_fill', np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1.0, 0.0]], dtype=np.float64)))
    fixture.append(('square_with_diagonal', np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)))
    fixture.append(('square_censored', np.asarray(
        [[0.0, 0.0, 0.0], [3.5, 0.0, 0.0], [3.5, 3.5, 0.0], [0.0, 3.5, 0.0]], dtype=np.float64)))
    fixture.append(('two_clusters', np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 4.0, 0.0], [2.0, 4.0, 0.0]], dtype=np.float64)))
    fixture.append(('micro_ring', np.asarray(
        [[0.0, 0.0, 0.0], [0.9, 0.0, 0.0], [1.35, 0.78, 0.0],
         [0.85, 1.56, 0.0], [-0.05, 1.56, 0.0], [-0.55, 0.78, 0.0]], dtype=np.float64)))
    fixture.append(('single_edge', np.asarray(
        [[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]], dtype=np.float64)))
    fixture.append(('coincident', np.asarray(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.7, 0.0, 0.0]], dtype=np.float64)))
    del root_two
    return fixture


def ph_definition_audit():
    """Section 5.1 definition audit against independent rank/shortest-path code."""
    records = []
    worst_betti = worst_wiener = worst_efficiency = 0.0
    for name, points in ph_fixtures():
        trajectory = view.five_descriptors(points)
        pairs = view.persistence_pairs(points)
        alpha0 = alpha1 = 0.0
        for index, radius in enumerate(view.ROUTER_RADII):
            expected_0, expected_1 = betti_by_rank(points, radius)
            alpha0 = max(alpha0, abs(trajectory[index, 3] * float(points.shape[0]) - expected_0))
            normalizer = max(1, len(pairs[1]))
            alpha1 = max(alpha1, abs(trajectory[index, 4] * normalizer - expected_1))
            wiener, efficiency = wiener_by_networkx(points, radius)
            worst_wiener = max(worst_wiener, abs(trajectory[index, 1] - wiener))
            worst_efficiency = max(worst_efficiency, abs(trajectory[index, 2] - efficiency))
        worst_betti = max(worst_betti, alpha0, alpha1)
        records.append({
            'fixture': name, 'points': int(points.shape[0]),
            'h0_intervals': len(pairs[0]), 'h1_intervals': len(pairs[1]),
            'h0_essential': sum(1 for _, death in pairs[0] if not math.isfinite(death)),
            'h1_essential': sum(1 for _, death in pairs[1] if not math.isfinite(death)),
            'max_betti_deviation': max(alpha0, alpha1),
            'column_max': trajectory.max(axis=0).tolist(),
            'finite': bool(np.isfinite(trajectory).all()),
        })
    # The two definition traps that must be visible as facts, not as prose.
    square = dict(ph_fixtures())['square_with_diagonal']
    square_pairs = view.persistence_pairs(square)
    censored = dict(ph_fixtures())['square_censored']
    censored_pairs = view.persistence_pairs(censored)
    triangle_pairs = view.persistence_pairs(dict(ph_fixtures())['triangle_fill'])
    coincident_pairs = view.persistence_pairs(dict(ph_fixtures())['coincident'])
    return {
        'fixtures': records,
        'worst_betti_deviation': worst_betti,
        'worst_wiener_deviation': worst_wiener,
        'worst_efficiency_deviation': worst_efficiency,
        'square_with_diagonal_h1': square_pairs[1],
        'square_censored_h1': censored_pairs[1],
        'square_censored_h1_essential': sum(
            1 for _, death in censored_pairs[1] if not math.isfinite(death)),
        'triangle_fill_h1': triangle_pairs[1],
        'triangle_fill_note': 'all three edges and the filling triangle share one '
                              'filtration value, so the H1 interval has zero length',
        'coincident_h0_zero_length': coincident_pairs[0],
        'coincident_note': 'zero-length intervals are dropped, not silently counted',
        'gudhi_max_dimension': 2,
        'field': 'F2',
        'radius_count': len(view.ROUTER_RADII),
        'max_edge_length': view.VR_MAX_EDGE,
        'grid': list(view.ROUTER_RADII),
    }


# ---------------------------------------------------------------------------
# Sample sets
# ---------------------------------------------------------------------------

def ordered_key_positions(source, count):
    """Positions of the first ``count`` P_train samples in raw key byte order."""
    keys = [bytes(source.samples[index][0]) for index in range(len(source))]
    order = sorted(range(len(keys)), key=lambda index: keys[index])
    return order[:int(count)], keys


def _sha256_keys(keys, order):
    digest = hashlib.sha256()
    for index in order:
        digest.update(keys[index])
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Audit phases
# ---------------------------------------------------------------------------

def mapping_audit(source, positions, *, sigma, ratio, seed):
    """Section 3.1/3.3 identity, mapping and scale audit for the fixed set."""
    totals = {'samples': 0, 'geometry_invalid': 0, 'canonical_atoms': 0,
              'heavy_atoms': 0, 'canonical_without_heavy': 0, 'center_bonds_zero': 0,
              'no_center_angle': 0, 'element_mismatch': 0, 'connectivity_mismatch': 0,
              'mapping_not_permutation': 0, 'stereo_atoms': 0, 'charge_nonzero': 0,
              'aromatic_atoms': 0, 'fallback_mask': 0, 'bonds_without_category': 0,
              'readout_invalid': 0, 'canonical_relations_absent_from_heavy': 0,
              'centre_bonds_absent_from_heavy': 0}
    zero_bond_keys, unmatched_keys, readout_keys, connectivity_keys = [], [], [], []
    absent_keys = []
    category_histogram = {}
    scale_edges = {slot: [] for slot in range(len(view.EXPERT_CUTOFFS))}
    scale_graphs_nonempty = {slot: 0 for slot in range(len(view.EXPERT_CUTOFFS))}
    trajectories, prepare_seconds, ph_seconds, payload_bytes = [], [], [], []
    charge_values, aromatic_values = set(), set()
    edge_total = 0
    for position in positions:
        topology, trimer, smiles = source[position]
        static = source.static_for(position)
        key = bytes(source.samples[position][0]).hex()
        totals['samples'] += 1
        started = time.perf_counter()
        data, labels = view.prepare_mcl_ph_sample(
            topology, trimer, smiles, seed=seed, key=key, position=int(position),
            sigma=sigma, ratio=ratio, static=static)
        prepare_seconds.append(time.perf_counter() - started)
        view_record = view.build_trimer_view(trimer)
        base = torch.as_tensor(topology.canonical_to_trimer_base_atom_id).long().reshape(-1)
        canonical_to_base = base
        totals['canonical_atoms'] += int(view_record.count and data.mcl_central_index.numel())
        totals['heavy_atoms'] += int(view_record.count)
        totals['canonical_without_heavy'] += int(labels['non_heavy_canonical'])
        if not bool(labels['readout_valid']):
            totals['readout_invalid'] += 1
            readout_keys.append(key)
        totals['stereo_atoms'] += int((torch.as_tensor(trimer.trimer_chiral_tag).long() != 0).sum())
        totals['charge_nonzero'] += int((view_record.charge != 0).sum())
        totals['aromatic_atoms'] += int(view_record.aromatic.sum())
        charge_values.update(view_record.charge.unique().tolist())
        aromatic_values.update(int(value) for value in view_record.aromatic.unique().tolist())
        if int(labels['mcl_center_bonds']) == 0:
            totals['center_bonds_zero'] += 1
            zero_bond_keys.append(key)
        if int(labels['mcl_angle_pair'].size(0)) == 0:
            totals['no_center_angle'] += 1
        if bool(labels['mcl_fallback']):
            totals['fallback_mask'] += 1
        if not bool(labels['geometry_valid']):
            totals['geometry_invalid'] += 1
        # element / connectivity / permutation echo of the canonical mapping.
        raw = torch.as_tensor(trimer.mips_to_trimer_central_index).long().reshape(-1)
        elements = torch.as_tensor(trimer.trimer_atomic_number).long().reshape(-1)
        if not torch.equal(elements[raw], torch.as_tensor(topology.z).long().reshape(-1)):
            totals['element_mismatch'] += 1
        if sorted(raw.tolist()) != sorted(set(raw.tolist())):
            totals['mapping_not_permutation'] += 1
        heavy_of_all = view_record.of_all
        mapped_canonical = heavy_of_all[raw]
        unmatched = torch.nonzero(mapped_canonical < 0, as_tuple=False).flatten()
        for index in unmatched.tolist():
            unmatched_keys.append((key, int(index), int(elements[raw[index]])))
        # Connectivity is compared on the periodic quotient, not on central-RU
        # indices: a canonical one-hop relation may join two boundary base
        # identities whose physical bond is realized across a repeat-unit seam.
        #   (a) every canonical spd==1 relation is a base-level Trimer bond;
        #   (b) every centre-internal Trimer bond is a canonical spd==1 relation.
        base_id = view_record.base_id
        all_atom_base = torch.as_tensor(trimer.trimer_base_ru_atom_id).long().reshape(-1)
        trimer_edge = torch.as_tensor(trimer.trimer_edge_index).long()
        base_bonds = {tuple(sorted((int(all_atom_base[trimer_edge[0, column]]),
                                    int(all_atom_base[trimer_edge[1, column]]))))
                      for column in range(int(trimer_edge.size(1)))}
        heavy_base_bonds = {tuple(sorted((int(base_id[a]), int(base_id[b]))))
                            for a, b in view_record.bond_ends.tolist()}
        canonical_to_base = torch.as_tensor(
            topology.canonical_to_trimer_base_atom_id).long().reshape(-1)
        edge = topology.lga_edge_index.long()
        canonical_pairs = set()
        for column in torch.nonzero(topology.lga_spd.long().reshape(-1) == 1,
                                    as_tuple=False).flatten().tolist():
            left, right = int(edge[0, column]), int(edge[1, column])
            if left == right:
                continue
            canonical_pairs.add(tuple(sorted((int(canonical_to_base[left]),
                                              int(canonical_to_base[right])))))
        centre_mask = np.array(static['token_center_mask'], dtype=bool)
        centre_a = np.array(static['token_pos_index_a'], dtype=np.int64)
        centre_b = np.array(static['token_pos_index_b'], dtype=np.int64)
        centre_bonds = {tuple(sorted((int(all_atom_base[centre_a[column]]),
                                      int(all_atom_base[centre_b[column]]))))
                        for column in np.flatnonzero(centre_mask).tolist()}
        if not canonical_pairs <= base_bonds:
            totals['connectivity_mismatch'] += 1
            connectivity_keys.append(key)
        if not centre_bonds <= canonical_pairs:
            totals['connectivity_mismatch'] += 1
            connectivity_keys.append(key + ':centre')
        # Canonical relations absent from the HEAVY graph are expected exactly
        # when a canonical atom is an explicit isotope hydrogen: A holds heavy
        # atoms only, so those relations must be accounted for, not hidden.
        absent = canonical_pairs - heavy_base_bonds
        if absent:
            totals['canonical_relations_absent_from_heavy'] += len(absent)
            absent_keys.append((key, sorted(absent)[:4]))
        centre_absent = centre_bonds - heavy_base_bonds
        if centre_absent:
            totals['centre_bonds_absent_from_heavy'] += len(centre_absent)
        # scale bookkeeping (directed relations per expert)
        for slot in range(len(view.EXPERT_CUTOFFS)):
            count = int((data.mcl_edge_scale == slot).sum())
            scale_edges[slot].append(count)
            if count:
                scale_graphs_nonempty[slot] += 1
        edge_total += int(data.mcl_edge_index.size(1))
        if int(data.mcl_edge_index.size(1)):
            for value, count in zip(*torch.unique(data.mcl_edge_type,
                                                  return_counts=True)[:2]):
                category_histogram[int(value)] = category_histogram.get(int(value), 0) + int(count)
            if bool((data.mcl_edge_type > view.UNKNOWN_CATEGORY).any()):
                totals['bonds_without_category'] += 1
        started = time.perf_counter()
        trajectory = view.five_descriptors(data.mcl_pos.numpy())
        ph_seconds.append(time.perf_counter() - started)
        trajectories.append(trajectory)
        payload_bytes.append(int(sum(value.numel() * value.element_size()
                                     for value in data.to_dict().values()
                                     if torch.is_tensor(value))))
        totals['canonical_atoms'] += 0
    trajectories = np.stack(trajectories)
    del canonical_to_base
    return {
        'totals': totals, 'zero_center_bond_keys': zero_bond_keys[:32],
        'zero_center_bond_key_count': len(zero_bond_keys),
        'unmatched_keys': unmatched_keys[:32],
        'connectivity_disagreement_keys': connectivity_keys[:32],
        'canonical_relations_absent_from_heavy_examples': absent_keys[:6],
        'edge_category_histogram': {str(key): value for key, value in
                                    sorted(category_histogram.items())},
        'edge_category_vocabulary': {'chemical': 5, 'nonbonded': view.NONBONDED_CATEGORY,
                                     'unknown': view.UNKNOWN_CATEGORY},
        'charge_values': sorted(charge_values), 'aromatic_values': sorted(aromatic_values),
        'scale': {
            'cutoffs': list(view.EXPERT_CUTOFFS),
            'edge_count_min': [int(min(scale_edges[slot])) for slot in scale_edges],
            'edge_count_median': [float(statistics.median(scale_edges[slot])) for slot in scale_edges],
            'edge_count_max': [int(max(scale_edges[slot])) for slot in scale_edges],
            'graphs_with_relation': [scale_graphs_nonempty[slot] for slot in scale_edges],
            'graphs_without_relation': [totals['samples'] - scale_graphs_nonempty[slot]
                                        for slot in scale_edges],
            'executed_dense_relations': int(edge_total),
        },
        'timing': {
            'prepare': _summary(prepare_seconds), 'ph': _summary(ph_seconds),
            'payload_bytes': _summary([float(value) for value in payload_bytes]),
        },
        'descriptor_variance': view.descriptor_channel_variance(trajectories).tolist(),
        'descriptor_column_range': [
            [float(trajectories[:, :, column].min()), float(trajectories[:, :, column].max())]
            for column in range(view.DESCRIPTOR_COLUMNS)],
        'descriptor_constant_columns': [
            bool(np.allclose(trajectories[:, :, column], trajectories[0, 0, column]))
            for column in range(view.DESCRIPTOR_COLUMNS)],
    }


def _summary(values):
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return {'count': 0}
    return {
        'count': len(ordered), 'mean': float(statistics.fmean(ordered)),
        'p50': float(statistics.median(ordered)),
        'p95': float(ordered[min(len(ordered) - 1, int(math.ceil(0.95 * len(ordered))) - 1)]),
        'max': float(ordered[-1]),
    }


def statistics_audit(source, positions, *, sigma, seed, limit_seconds):
    """Section 7 normalization statistics and the router constant mean."""
    router_rows, length_values, nonbond_values = [], [], []
    zero_center_bonds, counts = [], []
    bin_counts = {'length_pairs': 0, 'angle_pairs': 0,
                  'nonbond_pairs': 0, 'nonbond_graphs': 0,
                  'non_heavy_canonical': 0, 'samples_with_non_heavy': 0,
                  'canonical_atoms': 0, 'heavy_atoms': 0, 'centre_bonds': 0}
    bin_histogram = [0, 0, 0]
    partial = False
    started = time.perf_counter()
    for position in positions:
        if time.perf_counter() - started > limit_seconds:
            partial = True
            break
        topology, trimer, smiles = source[position]
        static = source.static_for(position)
        key = bytes(source.samples[position][0]).hex()
        reference = view.reference_view(trimer, key, sigma=sigma, seed=seed)
        base = view.build_trimer_view(trimer)
        field = view.build_trimer_view(reference)
        central, _, non_heavy = view.centre_mapping(topology, trimer, base)
        # A canonical atom whose Trimer counterpart is an explicit isotope
        # hydrogen has no heavy representative; that limits the fused readout of
        # this graph but does not invalidate the reference-view statistics.
        bin_counts['non_heavy_canonical'] += int(non_heavy)
        bin_counts['samples_with_non_heavy'] += int(bool(non_heavy))
        bond_index, bond_type, bond_center = view.physical_bonds(static, base)
        del bond_type
        # The declared N=0 case is "no centre-internal bond".  It is measured on
        # every statistics sample, not on a hand-picked subset.
        if int(bond_center.sum()) == 0:
            zero_center_bonds.append(key)
        canonical_count = canonical_atom_count(trimer)
        bin_counts['canonical_atoms'] += canonical_count
        bin_counts['heavy_atoms'] += int(base.count)
        bin_counts['centre_bonds'] += int(bond_center.sum())
        counts.append({'n': canonical_count, 'a': int(base.count),
                       'centre_bonds': int(bond_center.sum()), 'tokens': int(bond_index.size(0))})
        router_rows.append(view.five_descriptors(field.positions.numpy()).astype(np.float32))
        if bool(static['geometry_valid']) and bool(trimer.trimer_geometry_valid):
            length_pairs, length_raw, angle_pairs, _ = view.local_targets(
                static, torch.as_tensor(trimer.trimer_pos), bond_index)
            del length_pairs
            length_values.append(length_raw)
            bin_counts['angle_pairs'] += int(angle_pairs.size(0))
            bin_counts['length_pairs'] += int(length_raw.numel())
        pairs, distance = view.nonbond_candidates(field.positions, field, central, bond_index)
        sampled, slots = view.sample_nonbond_pairs(
            pairs, distance, view.view_generator(seed, key, int(position), view.PAIR_SUBSTREAM))
        if int(sampled.size(0)):
            clean = torch.linalg.vector_norm(
                torch.as_tensor(trimer.trimer_pos)[sampled[:, 0]]
                - torch.as_tensor(trimer.trimer_pos)[sampled[:, 1]], dim=-1)
            nonbond_values.append(clean)
            bin_counts['nonbond_pairs'] += int(sampled.size(0))
            bin_counts['nonbond_graphs'] += 1
            for slot in slots.tolist():
                bin_histogram[slot] += 1
    layout = view.ROUTER_RADII_ARRAY
    trajectory_stack = np.stack(router_rows) if router_rows else np.zeros(
        (0, layout.size, view.DESCRIPTOR_COLUMNS), dtype=np.float32)
    length_all = torch.cat(length_values) if length_values else torch.zeros(0)
    nonbond_all = torch.cat(nonbond_values) if nonbond_values else torch.zeros(0)
    payload = {
        'samples_requested': int(len(positions)), 'samples_used': int(trajectory_stack.shape[0]),
        'partial': partial, 'seconds': time.perf_counter() - started,
        'length': view.geometric_statistics(length_all) if length_all.numel() else None,
        'nonbond': view.geometric_statistics(nonbond_all) if nonbond_all.numel() else None,
        'router_mean': (trajectory_stack.mean(axis=0) if trajectory_stack.shape[0]
                        else np.zeros((layout.size, view.DESCRIPTOR_COLUMNS),
                                      dtype=np.float32)).tolist(),
        'router_std': (trajectory_stack.std(axis=0) if trajectory_stack.shape[0]
                       else np.zeros((layout.size, view.DESCRIPTOR_COLUMNS),
                                     dtype=np.float32)).tolist(),
        'target_counts': bin_counts, 'nonbond_bin_histogram': bin_histogram,
        'nonbond_bins': [list(pair) for pair in view.NONBOND_BINS],
        'nonbond_max_pairs': view.NONBOND_MAX_PAIRS,
        'ordering_policy': 'P_train keys sorted by raw 32-byte order, first N',
        'real_zero_center_bond_keys': zero_center_bonds[:16],
        'real_zero_center_bond_key_count': len(zero_center_bonds),
        'zero_center_bond_declaration': (
            'no P_train sample in the audited set has zero centre-internal bonds; the '
            'synthetic fixture in the tests is then the only coverage and must not be '
            'reported as real data coverage' if not zero_center_bonds else
            'real keys recorded above'),
        'shape_summary': _shape_summary(counts),
    }
    return payload, trajectory_stack, length_all, nonbond_all


def canonical_atom_count(trimer):
    """Number of canonical O8 atoms of one frozen record."""
    return int(torch.as_tensor(trimer.mips_to_trimer_central_index).numel())


def _shape_summary(rows):
    if not rows:
        return {'count': 0}
    import numpy as _np

    summary = {'count': len(rows)}
    for name in ('n', 'a', 'centre_bonds', 'tokens'):
        values = _np.asarray([row[name] for row in rows], dtype=_np.int64)
        summary[name] = {'min': int(values.min()), 'p50': int(_np.median(values)),
                         'max': int(values.max()), 'mean': float(values.mean())}
    summary['zero_centre_bonds'] = int(sum(1 for row in rows if row['centre_bonds'] == 0))
    return summary


def identity_source_audit(source, positions):
    """Where the Trimer charge / aromaticity actually come from."""
    charge, aromatic, bond_aromatic = {}, {}, {}
    samples = 0
    for position in positions[:128]:
        trimer = source[position][1]
        samples += 1
        for value in torch.as_tensor(trimer.trimer_formal_charge).long().reshape(-1).tolist():
            charge[value] = charge.get(value, 0) + 1
        for value in torch.as_tensor(trimer.trimer_is_aromatic).bool().reshape(-1).tolist():
            aromatic[int(value)] = aromatic.get(int(value), 0) + 1
        for value in torch.as_tensor(trimer.trimer_bond_aromatic).bool().reshape(-1).tolist():
            bond_aromatic[int(value)] = bond_aromatic.get(int(value), 0) + 1
    return {
        'samples': samples,
        'provenance': 'frozen Trimer record fields trimer_formal_charge / '
                      'trimer_is_aromatic / trimer_bond_aromatic, written by the '
                      'cache builder from the frozen RDKit Trimer molecule',
        'fields_present': sorted(
            name for name in ('trimer_formal_charge', 'trimer_is_aromatic',
                              'trimer_chiral_tag', 'trimer_bond_aromatic',
                              'trimer_heavy_mask', 'trimer_heavy_indices',
                              'trimer_base_ru_atom_id', 'trimer_ru_offset',
                              'mips_to_trimer_central_index', 'o8_heavy_mask')
            if hasattr(source[positions[0]][1], name)
            or name in source[positions[0]][1].keys()),
        'charge_histogram': {str(key): value for key, value in sorted(charge.items())},
        'aromatic_atom_histogram': {str(key): value for key, value in sorted(aromatic.items())},
        'aromatic_bond_histogram': {str(key): value for key, value in sorted(bond_aromatic.items())},
        'charge_vocabulary': {'range': [-3, 3], 'size': view.CHARGE_VOCABULARY,
                              'other_index': view.CHARGE_OTHER, 'mask_index': view.CHARGE_MASK},
        'aromatic_vocabulary': {'size': view.AROMATIC_VOCABULARY,
                                'mask_index': view.AROMATIC_MASK},
        'regeneration': 'none: no coordinate, conformer or molecule is rebuilt',
        'in_range': sorted(charge) and min(charge) >= -3 and max(charge) <= 3,
    }


def mask_dependency_table():
    """Which tensors can carry a masked atom's target identity on each path."""
    return {
        'rule': 'only direct identity carriers are replaced by the declared neutral/MASK '
                'category; chemical connectivity and SPD are never deleted',
        'entries': [
            {'path': 'O8 atom input', 'field': 'mips_x (137-dim, first 101 = element one-hot)',
             'carries_identity': True, 'treatment': 'zeroed at masked rows by the existing '
             'atom_mask in MIPSLocalAtomEmbedding', 'unchanged_mechanism': True},
            {'path': 'O8 backbone flag', 'field': 'mips_backbone_mask',
             'carries_identity': False, 'treatment': 'kept'},
            {'path': 'O8 bond chemistry', 'field': 'bond_path_features [R,2,14]',
             'carries_identity': False,
             'treatment': 'kept: bond type, conjugation, ring and stereo only; no endpoint '
                          'element or index is encoded'},
            {'path': 'O8 adjacency / SPD', 'field': 'lga_edge_index, lga_spd, lga_path_index, '
             'lga_path_mask, lga_path_shift, lga_source_image_shift',
             'carries_identity': False, 'treatment': 'kept by contract'},
            {'path': 'legacy GLT bond tokens', 'field': 'bond_z_a, bond_z_b',
             'carries_identity': True,
             'treatment': 'removed from the MCL-PH batch: the new architecture has no GLT '
                          'branch and never reads an endpoint element table'},
            {'path': 'Trimer expert nodes', 'field': 'mcl_z, mcl_charge, mcl_aromatic',
             'carries_identity': True,
             'treatment': 'MASK category for every provable heavy copy of a masked canonical '
                          'atom (synchronised_mask)'},
            {'path': 'Trimer geometry', 'field': 'mcl_pos, mcl_edge_distance',
             'carries_identity': False,
             'treatment': 'kept: the geometry view is a different object from the chemical '
                          'feature mask (section 5.3)'},
            {'path': 'Trimer bond table', 'field': 'mcl_bond_index, mcl_bond_type',
             'carries_identity': False, 'treatment': 'kept'},
        ],
    }


def batch_memory_audit(source, positions, *, sigma, ratio, seed):
    """Packed-batch payload bytes for the configured microbatch and global batch."""
    packed = []
    for position in positions[:16]:
        topology, trimer, smiles = source[position]
        key = bytes(source.samples[position][0]).hex()
        packed.append(view.prepare_mcl_ph_sample(
            topology, trimer, smiles, seed=seed, key=key, position=int(position),
            sigma=sigma, ratio=ratio, static=source.static_for(position)))
    batch, labels = view.mcl_ph_collate(packed)
    per_sample = sum(value.numel() * value.element_size() for value in batch.to_dict().values()
                     if torch.is_tensor(value)) / float(len(packed))
    lean = sum(value.numel() * value.element_size()
               for value in batch.to_dict().values() if torch.is_tensor(value))
    del labels
    dropped = list(view.UNUSED_BY_MCL_PH)
    return {
        'microbatch': 84, 'accumulation': 3, 'world_size': 4, 'global_batch': 1008,
        'bytes_per_sample_full': float(per_sample),
        'bytes_per_sample_new_route': float(lean / max(1, len(packed))),
        'bytes_microbatch_16_full': int(lean),
        'bytes_microbatch_16_new_route': int(lean),
        'projected_microbatch_84_new_route': float(lean / max(1, len(packed)) * 84),
        'projected_per_rank_accumulation_3': float(lean / max(1, len(packed)) * 84 * 3),
        'projected_global_batch_1008': float(lean / max(1, len(packed)) * 1008),
        'fields_dropped_on_the_new_route': dropped,
    }


def _json_safe(value):
    """JSON has no infinity; censored intervals are recorded as the string ``inf``."""
    if isinstance(value, float):
        if math.isnan(value):
            return None
        if math.isinf(value):
            return 'inf' if value > 0 else '-inf'
        return value
    if isinstance(value, np.floating):
        return _json_safe(float(value))
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    return value


def _write_report(path, report):
    Path(path).write_text(json.dumps(_json_safe(report), indent=2, sort_keys=True,
                                    allow_nan=False) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cohort-root', default=DEFAULT_COHORT)
    parser.add_argument('--cache-root', default=DEFAULT_CACHE)
    parser.add_argument('--dual-static-root', default=DEFAULT_STATIC)
    parser.add_argument('--config', default=DEFAULT_CONFIG)
    parser.add_argument('--output', default='results/mcl_ph_20260921/p0')
    parser.add_argument('--statistics-output',
                        default='results/mcl_ph_20260921/p0/statistics.npz')
    parser.add_argument('--mapping-samples', type=int, default=MAPPING_SAMPLES)
    parser.add_argument('--statistics-samples', type=int, default=STATISTICS_SAMPLES)
    parser.add_argument('--budget-seconds', type=float, default=BUDGET_SECONDS)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--sigma', type=float, default=0.03)
    parser.add_argument('--ratio', type=float, default=0.30)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    config = json.loads(Path(args.config).read_text(encoding='utf-8'))
    split = load_sample_index_artifact(config['sample_index_artifact'],
                                       config['sample_index_split'])
    source, frame = open_source(args.cohort_root, args.cache_root,
                               dual_static_root=args.dual_static_root)
    report = {
        'plan': 'MCL-PH-20260921-01/r1', 'phase': 'P0', 'status': 'RUNNING',
        'command': sys.argv, 'devices': 'cpu', 'model_calls': 0,
        'optimizer_updates': 0,
        'budget_seconds': float(args.budget_seconds),
        'inputs': {
            'cohort_root': str(Path(args.cohort_root).resolve()),
            'cache_root': str(Path(args.cache_root).resolve()),
            'dual_static_root': str(Path(args.dual_static_root).resolve()),
            'config': str(Path(args.config).resolve()),
            'config_sha256': hashlib.sha256(Path(args.config).read_bytes()).hexdigest(),
            'cohort_manifest_hash': source.cohort['manifest_hash'],
            'main_bundle_hash': source.bundle.bundle_hash,
            'dual_static_manifest_hash': source.static_cache.manifest_hash,
            'sample_index_artifact': split['path'], 'sample_index_sha256': split['sha256'],
            'sample_index_split': split['split'],
            'source_count': len(source),
        },
    }
    try:
        subset = IndexedFrozenDualSource(source, split['indices'])
        wanted = max(int(args.mapping_samples), int(args.statistics_samples))
        order, keys = ordered_key_positions(subset, wanted)
        report['sample_sets'] = {
            'ordering_policy': 'first N of the P_train split sorted by raw 32-byte key order',
            'selected': len(order),
            'first_key': bytes(keys[order[0]]).hex(),
            'last_key': bytes(keys[order[-1]]).hex(),
            'ordered_key_sha256': _sha256_keys(keys, order),
            'mapping_samples': min(int(args.mapping_samples), len(order)),
            'statistics_samples': min(int(args.statistics_samples), len(order)),
        }
        print(json.dumps({'stage': 'sequence_ready', **report['sample_sets']}), flush=True)

        report['identity_source'] = identity_source_audit(subset, order)
        print(json.dumps({'stage': 'identity_source_done'}), flush=True)
        report['ph_definition'] = ph_definition_audit()
        print(json.dumps({'stage': 'ph_definition_done',
                          'worst_betti_deviation': report['ph_definition']['worst_betti_deviation'],
                          'worst_wiener_deviation': report['ph_definition']['worst_wiener_deviation'],
                          'worst_efficiency_deviation': report['ph_definition']['worst_efficiency_deviation']},
                         ), flush=True)

        mapping_positions = order[:int(args.mapping_samples)]
        report['mapping'] = mapping_audit(subset, mapping_positions, sigma=args.sigma,
                                         ratio=args.ratio, seed=args.seed)
        print(json.dumps({'stage': 'mapping_done',
                          'totals': report['mapping']['totals'],
                          'timing': report['mapping']['timing']}), flush=True)

        report['mask_dependencies'] = mask_dependency_table()
        report['memory'] = batch_memory_audit(subset, mapping_positions, sigma=args.sigma,
                                              ratio=args.ratio, seed=args.seed)
        print(json.dumps({'stage': 'memory_done', 'memory': report['memory']}), flush=True)

        remaining = float(args.budget_seconds) - (time.perf_counter() - started)
        if remaining <= 0:
            raise TimeoutError('P0 budget exhausted before the statistics pass')
        statistics_payload, trajectory_stack, length_all, nonbond_all = statistics_audit(
            subset, order, sigma=args.sigma, seed=args.seed,
            limit_seconds=remaining * 0.92)
        report['statistics'] = {key: value for key, value in statistics_payload.items()}
        np.savez(args.statistics_output,
                 router_mean=np.asarray(statistics_payload['router_mean'], dtype=np.float32),
                 router_std=np.asarray(statistics_payload['router_std'], dtype=np.float32),
                 length_mu=np.asarray(statistics_payload['length']['mu'], dtype=np.float64),
                 length_sigma=np.asarray(statistics_payload['length']['sigma'], dtype=np.float64),
                 nonbond_mu=np.asarray(statistics_payload['nonbond']['mu'], dtype=np.float64),
                 nonbond_sigma=np.asarray(statistics_payload['nonbond']['sigma'], dtype=np.float64),
                 samples=np.asarray(statistics_payload['samples_used'], dtype=np.int64),
                 seed=np.asarray(args.seed, dtype=np.int64),
                 sigma=np.asarray(args.sigma, dtype=np.float64),
                 ordered_key_sha256=np.asarray(report['sample_sets']['ordered_key_sha256']))
        print(json.dumps({'stage': 'statistics_done',
                          'length': statistics_payload['length'],
                          'nonbond': statistics_payload['nonbond'],
                          'router_mean_range': [float(np.min(statistics_payload['router_mean'])),
                                                float(np.max(statistics_payload['router_mean']))],
                          'partial': statistics_payload['partial'],
                          'samples_used': statistics_payload['samples_used']}), flush=True)

        ph_seconds = report['mapping']['timing']['ph']['p50']
        prepare_seconds = report['mapping']['timing']['prepare']['p50']
        report['cost_projection'] = {
            'ph_seconds_per_graph_p50': ph_seconds, 'ph_seconds_per_graph_max':
                report['mapping']['timing']['ph']['max'],
            'prepare_seconds_per_graph_p50': prepare_seconds,
            'global_batch': 1008,
            'projected_ph_cpu_seconds_per_update': ph_seconds * 1008.0,
            'projected_prepare_cpu_seconds_per_update_single_process': prepare_seconds * 1008.0,
            'projected_prepare_cpu_hours_for_5000_updates_single_process':
                prepare_seconds * 1008.0 * 5000 / 3600.0,
            'note': 'preparation is duplicated per rank and is normally hidden by '
                    '--prep-workers; the projection counts one preparation per graph',
        }
        elapsed = time.perf_counter() - started
        partial = bool(statistics_payload['partial'])
        report['status'] = 'PARTIAL' if partial else 'PASS'
        report['seconds'] = elapsed
        report['budget_used_fraction'] = elapsed / float(args.budget_seconds)
        report['outer_test'] = 'NOT_RUN'
    except BaseException as error:
        report['status'] = 'FAILED'
        report['error'] = f'{type(error).__name__}: {error}'
        report['seconds'] = time.perf_counter() - started
        _write_report(output / 'audit.json', report)
        raise
    finally:
        source.close()
    _write_report(output / 'audit.json', report)
    print(json.dumps({'status': report['status'], 'seconds': report['seconds'],
                      'output': str(output)}), flush=True)
    if report['status'] != 'PASS':
        raise SystemExit(3)


if __name__ == '__main__':
    main()
