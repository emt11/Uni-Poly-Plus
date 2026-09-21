"""Objective-structure tests for the MCL-PH pre-training loss (section 7).

The declared structure is: one shared ``512 -> 101`` head applied to both the
2D and the fused state, a geometry-invalid graph contributing only its 2D term
at weight one, and independent effective denominators for the local-geometry and
per-scale non-bond terms.  A 4-rank gloo check then verifies the collective
semantics with partial and fully absent geometry.
"""
import json
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pytest
import torch

from test_mcl_ph_modules import build_batch, build_labels, make_pretrainer
from src.modules.mcl_ph_pretrain import (ATOM_CLASSES, BALANCE_WEIGHT, HUBER_BETA,
                                         MCLPHPretrainer)


def test_atom_head_is_shared_and_shaped_as_declared():
    model = make_pretrainer('gate')
    assert model.atom_head.in_features == 512
    assert model.atom_head.out_features == ATOM_CLASSES
    assert HUBER_BETA == 0.5
    assert BALANCE_WEIGHT == 1e-3


def test_objective_weights_each_task_by_its_own_effective_denominator():
    model = make_pretrainer('gate')
    batch, labels = build_batch(), build_labels(build_batch())
    report = model(batch, labels)
    plain = model.objective(report, world_size=1)
    scaled = model.objective(report, world_size=4)
    # The 2D and geometry terms follow the global count, so the factor of four
    # must be visible; the balance term is already a global function.
    expected = (plain - BALANCE_WEIGHT * report['balance']) * 4 \
        + BALANCE_WEIGHT * report['balance']
    assert float(scaled) == pytest.approx(float(expected), rel=1e-6)


def test_a_geometry_invalid_graph_contributes_only_its_two_dimensional_term():
    """Only the masked graph is scored, so the branch formula is observable."""
    model = make_pretrainer('gate', collect_diagnostics=True)
    labels = build_labels(build_batch())
    labels['atom_mask'] = torch.tensor([True, True, False], dtype=torch.bool)
    fused = model(build_batch(), labels)
    diagnostics = fused['diagnostics']
    assert int(fused['atom_count']) == 1
    expected = 0.5 * (diagnostics['atom_two_sum'] + diagnostics['atom_fused_sum'])
    assert float(fused['atom_sum']) == pytest.approx(expected, rel=1e-6)

    invalid_labels = build_labels(build_batch(), geometric=False)
    invalid_labels['atom_mask'] = torch.tensor([True, True, False], dtype=torch.bool)
    plain = model(build_batch(geometry_valid=(False, False), readout_valid=(False, False)),
                  invalid_labels)
    assert int(plain['atom_count']) == 1
    assert float(plain['atom_sum']) == pytest.approx(plain['diagnostics']['atom_two_sum'],
                                                     rel=1e-6)
    del model


def test_geometry_terms_never_enter_a_fabricated_denominator():
    model = make_pretrainer('gate')
    batch = build_batch()
    labels = build_labels(batch)
    labels['mcl_length_pair'] = torch.zeros((0, 2), dtype=torch.long)
    labels['mcl_length_target'] = torch.zeros(0)
    labels['mcl_length_raw'] = torch.zeros(0)
    labels['mcl_length_graph'] = torch.zeros(0, dtype=torch.long)
    labels['mcl_angle_pair'] = torch.zeros((0, 2), dtype=torch.long)
    labels['mcl_angle_target'] = torch.zeros(0)
    labels['mcl_angle_graph'] = torch.zeros(0, dtype=torch.long)
    labels['mcl_nonbond_pair'] = torch.zeros((0, 2), dtype=torch.long)
    labels['mcl_nonbond_slot'] = torch.zeros(0, dtype=torch.long)
    labels['mcl_nonbond_target'] = torch.zeros(0)
    labels['mcl_nonbond_raw'] = torch.zeros(0)
    labels['mcl_nonbond_graph'] = torch.zeros(0, dtype=torch.long)
    report = model(batch, labels)
    assert int(report['geo_count']) == 0
    assert int(report['local_count']) == 0
    assert int(report['nonbond_count']) == 0
    objective = model.objective(report)
    assert torch.isfinite(objective)
    objective.backward()
    gradient = model.local_decoder.bond[0].weight.grad
    assert gradient is None or float(gradient.abs().sum()) == 0.0, \
        'a decoder with no target must not receive a fabricated gradient'
    assert model.nonbond_decoder.net[0].weight.grad is None or \
        float(model.nonbond_decoder.net[0].weight.grad.abs().sum()) == 0.0


def test_an_empty_target_graph_produces_no_local_decoder_gradient():
    model = make_pretrainer('gate')
    batch, labels = build_batch(), build_labels(build_batch())
    report = model(batch, labels)
    report['geo_sum'] = report['geo_sum'] * 0.0
    report['geo_count'] = torch.zeros_like(report['geo_count'])
    objective = model.objective(report)
    objective.backward()
    gradient = model.local_decoder.length.weight.grad
    assert gradient is None or float(gradient.abs().sum()) == 0.0


def test_the_local_decoder_is_shared_across_the_three_experts():
    model = make_pretrainer('gate')
    pairs = torch.tensor([[0, 1]], dtype=torch.long)
    reference = model.local_decoder.bond_repr(model.encoder.branch.experts[0].element.weight[:4], pairs)
    other = model.local_decoder.bond_repr(model.encoder.branch.experts[1].element.weight[:4], pairs)
    assert reference.shape == other.shape
    assert len(list(model.local_decoder.parameters())) == len(
        list(MCLPHPretrainer('gate').local_decoder.parameters()))


def test_ddp_four_rank_partial_and_fully_absent_geometry():
    """0-update 4-rank gloo check: 2 distributed forwards/backwards, 8 rank calls."""
    script = Path(__file__).with_name('_mcl_ph_ddp_check.py')
    completed = subprocess.run(
        [sys.executable, '-m', 'torch.distributed.run', '--standalone',
         '--nproc_per_node=4', str(script)],
        capture_output=True, text=True, timeout=1200,
        env={**os.environ, 'OMP_NUM_THREADS': '1', 'MASTER_PORT': '29577'})
    assert completed.returncode == 0, completed.stdout + completed.stderr
    verdict = json.loads(completed.stdout.strip().splitlines()[-1])
    assert verdict['status'] == 'PASS'
    ranks = verdict['ranks']
    assert len(ranks) == 4
    partial = [rank['stages']['partial_zero'] for rank in ranks]
    # The atom mask term exists on every rank (a geometry-invalid graph keeps
    # its 2D sample); only the even ranks carry geometric targets.
    assert all(entry['local_atom_count'] == 2 for entry in partial)
    # A graph without a valid geometry produces no geometric target at all.
    assert sum(1 for entry in partial if entry['local_geometry_count'] > 0) == 2
    assert all(entry['objective_is_finite'] for entry in partial)
    assert all(entry['global_atom_count'] == 8 for entry in partial), \
        'the atom denominator is the global mask count over all four ranks'
    # The geometry denominator counts graphs, not ranks: the two even ranks carry
    # two geometry-valid graphs each, so the global denominator is 4 while the odd
    # ranks contribute 0.  It stays independent of the atom denominator (8).
    assert all(entry['local_geometry_count'] == (2.0 if rank['rank'] % 2 == 0 else 0.0)
               for rank, entry in zip(ranks, partial)), \
        'geometry validity is per graph, so an invalid graph contributes no target'
    assert all(entry['global_geometry_count'] == 4 for entry in partial), \
        'the geometry denominator only counts graphs with a real target'
    for entry in partial:
        if entry['local_atom_count'] > 0:
            assert entry['o8_gradient_norm'] > 0, 'the CE must still backpropagate'
    absent = [rank['stages']['all_zero'] for rank in ranks]
    assert all(entry['balance_empty'] for entry in absent)
    assert all(entry['objective_is_finite'] for entry in absent)
    assert all(entry['roster_gradient_norm'] in (None, 0.0) for entry in absent), \
        'no target exists, so no expert may receive a fabricated gradient'
    # Update 501 is the first Top-2 update: exactly one scale is rejected per
    # graph, and a rejected expert must still receive the geometry gradient from
    # the ranks that do carry a target.
    for entry in partial:
        assert entry['router_mode'] == 'top2'
        assert entry['selected_slots'] == 2
        for weights in entry['alpha']:
            assert abs(sum(weights) - 1.0) < 1e-5
            assert sum(1 for value in weights if value > 0) == 2, weights
        assert entry['router_gradient_norm'] > 0, 'the router keeps its task gradient'
    geometry_ranks = [entry for rank, entry in zip(ranks, partial) if rank['rank'] % 2 == 0]
    for slot in range(3):
        assert all(entry['expert_gradient_norms'][slot] > 0 for entry in geometry_ranks), \
            f'expert {slot}: a rejected expert must keep the geometry gradient'
    assert any(entry['expert_gradient_norms'][slot] > 0 for entry in geometry_ranks
               for slot in range(3))


COMMON_INIT = (Path(__file__).resolve().parents[1] / 'results' / 'glt_pred_20260918'
               / 's3b_prep' / 'common_init_v1.pt')


def test_only_the_o8_backbone_comes_from_the_shared_common_state():
    """Section 9: O8 loads by name; no old GLT key reaches the new modules.

    Instantiating the model and copying tensors is a 0-forward, 0-backward check,
    so it consumes none of the P1 CPU model budget.
    """
    if not COMMON_INIT.is_file():
        pytest.skip('the frozen common initialization artifact is not present')
    from scripts.pretrain_mcl_ph import SHARED_INIT_EXCLUDE_NAMES
    from src.training.glt_dual_runtime import apply_common_initialization

    artifact = torch.load(COMMON_INIT, map_location='cpu', weights_only=False)
    state = artifact['common_state_dict']
    for mode in ('cat', 'gate', 'xattn'):
        model = MCLPHPretrainer(mode, dropout=0.1, cutoffs=(2.0, 3.0, 4.0),
                               router_dense_updates=500)
        before = {name: value.clone() for name, value in model.state_dict().items()
                  if not name.startswith('encoder.o8.')}
        excluded = SHARED_INIT_EXCLUDE_NAMES(model, state)
        apply_common_initialization(model, COMMON_INIT, exclude=excluded)
        after = model.state_dict()
        for name, value in before.items():
            assert torch.equal(after[name], value), f'{mode}: {name} was overwritten'
        copied = [name for name in after if name.startswith('encoder.o8.')]
        assert copied, f'{mode}: the O8 backbone was not copied'
        for name in copied:
            assert torch.equal(after[name], state[name]), f'{mode}: {name} is not the shared state'
        assert 'encoder.glt.layers.0.linear.weight' not in after, \
            f'{mode}: a GLT key must not exist on the MCL-PH route'


def test_the_shared_new_state_covers_the_same_tensors_in_every_arm():
    """The shared new-parameter state is fusion independent by construction."""
    from scripts.pretrain_mcl_ph import shared_new_state

    sets = {mode: set(shared_new_state(MCLPHPretrainer(
        mode, dropout=0.1, cutoffs=(2.0, 3.0, 4.0), router_dense_updates=500)))
        for mode in ('cat', 'gate', 'xattn')}
    assert sets['cat'] == sets['gate'] == sets['xattn']
    assert not [name for name in sets['gate'] if 'fusion' in name or 'cross' in name]


def test_step_diagnostics_reports_finite_observation_fields():
    """The per-step diagnostics must survive a real report (1 forward, no backward).

    This path is observation only, but it writes every saved step record, so a
    defect in it loses the whole run: it was found by the GPU smoke, where the
    PyG batch is moved in place while the label dict stays on the host.
    """
    import math

    from scripts.pretrain_mcl_ph import _step_diagnostics

    model = make_pretrainer('gate', collect_diagnostics=True)
    batch = build_batch()
    labels = build_labels(batch)
    report = model(batch, labels)
    record = _step_diagnostics(model, report, batch, labels, 1)
    assert record['router_mode'] in ('dense', 'top2')
    assert record['world_size'] == 1
    assert record['trajectory']['nonzero_graphs'] == 2
    for value in record['losses'].values():
        assert math.isfinite(value)
    for value in record['trajectory']['per_column_mean'] + record['trajectory']['per_column_std']:
        assert math.isfinite(value)
    baseline = record['distance_copy_baseline']
    # The fixture's view distances are exactly its raw targets, so the diagnostic
    # gap between the noisy view and the clean target is zero on both tasks.
    assert baseline['nonbond_abs_log1p_error_max'] == 0.0
    assert baseline['nonbond_abs_log1p_error_mean'] == 0.0
    assert baseline['length_abs_log1p_error_mean'] == 0.0
