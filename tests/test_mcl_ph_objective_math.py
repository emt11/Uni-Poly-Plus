"""Distributed objective-arithmetic test for plan MCL-PH-20260921-01/r2.

``tests/_mcl_ph_objective_check.py`` is launched with ``torchrun`` on four gloo
ranks and its verdict is asserted here.  The check compares the *production*
objective, accumulation and balance code against hand-computed closed forms and
against a single-process global computation:

* the update-level denominator is the exact sum over ranks and microsteps of the
  effective graph counts, and a per-rank (pre-r2) denominator is demonstrably a
  different number;
* the balance term is computed per microstep and averaged over the accumulation;
* an empty denominator is a finite, differentiable zero;
* the rank-averaged gradient of a real pre-trainer forward equals the gradient of
  the same objective evaluated as one global computation.

Not model-free: one launch costs six model forwards and six model backwards (two
for the four-rank forward/backward and four for the rank-0 reference), which is
why the whole launch happens once per test session.
"""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CHECK = ROOT / 'tests' / '_mcl_ph_objective_check.py'
WORLD_SIZE = 4


@pytest.fixture(scope='module')
def verdict():
    environment = dict(os.environ, OMP_NUM_THREADS='1')
    environment.pop('MASTER_PORT', None)
    completed = subprocess.run(
        [sys.executable, '-m', 'torch.distributed.run', '--standalone',
         '--nproc_per_node', str(WORLD_SIZE), str(CHECK)],
        cwd=str(ROOT), capture_output=True, text=True, env=environment, timeout=420)
    payload = None
    for line in completed.stdout.splitlines():
        if line.startswith('{"status"'):
            payload = json.loads(line)
    assert payload is not None, completed.stdout[-4000:] + completed.stderr[-2000:]
    assert completed.returncode == 0, completed.stderr[-4000:]
    assert payload['status'] == 'PASS'
    return payload


def test_every_rank_reports_the_same_update_level_denominator(verdict):
    ranks = verdict['ranks']
    assert len(ranks) == WORLD_SIZE
    for rank in ranks:
        partial = rank['synthetic_partial']
        assert partial['denominators'] == partial['global_update_counts']
        assert partial['denominators'] == {'atom': 30.0, 'geometry': 17.0}, \
            'the denominator must sum every rank and every microstep'
        assert partial['max_deviation'] < 1e-5


def test_a_local_denominator_is_not_the_global_one(verdict):
    """The pre-r2 formula would not reproduce the hand-computed gradient."""
    expected = verdict['ranks'][0]['synthetic_partial']['expected']
    for rank in verdict['ranks']:
        legacy = rank['synthetic_partial']['legacy_local_denominator_gradient']
        assert abs(legacy - expected[0]) > 1e-3, (rank['rank'], legacy)


def test_an_empty_geometry_denominator_is_a_finite_differentiable_zero(verdict):
    for rank in verdict['ranks']:
        empty = rank['synthetic_empty_geometry']
        assert empty['denominators']['atom'] == 30.0
        assert empty['denominators']['geometry'] == 0.0
        assert empty['gradient'][1] == 0.0, 'no valid geometry must produce no gradient'
        assert empty['expected'][1] == 0.0
        statistics = empty['statistics'][0]['geometry']
        assert statistics['effective_graphs'] == 0.0
        assert statistics['numerator'] == 0.0 and statistics['math_loss'] == 0.0
        assert statistics['loss'] == 0.0


def test_the_maths_the_backward_scale_and_the_logged_loss_stay_separate(verdict):
    """``math_loss`` is the reported quantity; ``loss`` carries the DDP factor."""
    for rank in verdict['ranks']:
        world = float(rank['world_size'])
        for statistics in rank['synthetic_partial']['statistics']:
            for name in ('atom', 'geometry'):
                entry = statistics[name]
                if entry['effective_graphs'] == 0:
                    continue
                assert entry['backward_scale'] == world / entry['effective_graphs']
                assert abs(entry['math_loss']
                           - entry['numerator'] / entry['effective_graphs']) < 1e-12
                # ``loss`` is the float32 tensor the backward used, so the
                # comparison carries the single-precision epsilon.
                assert abs(entry['loss'] - entry['math_loss'] * world) \
                    < 1e-6 * max(1.0, abs(entry['loss']))


def test_the_balance_term_matches_its_global_closed_form(verdict):
    for rank in verdict['ranks']:
        balance = rank['balance']
        assert balance['accumulation'] == 3
        assert balance['max_deviation'] < 1e-6, balance['max_deviation']
        assert len(balance['term_values']) == 3


def test_the_rank_averaged_model_gradient_equals_one_global_computation(verdict):
    reference = verdict['ranks'][0]['model_partial']
    assert reference['update_denominators'] == reference['expected_update_counts']
    assert reference['update_denominators'] == {'atom': 3.0, 'geometry': 2.0}
    assert reference['max_deviation'] is not None
    assert reference['max_deviation'] < 1e-4 * max(1.0, reference['reference_norm'])
    assert reference['gradient_norm'] == pytest.approx(reference['reference_norm'],
                                                       rel=1e-5)
    for rank in verdict['ranks']:
        model = rank['model_partial']
        assert model['update_denominators'] == {'atom': 3.0, 'geometry': 2.0}
        assert model['local_counts'] != model['expected_update_counts']


def test_the_model_stage_runs_on_the_production_objective(verdict):
    """The fixture must actually exercise the real objective and accumulation."""
    counts = {'atom': [], 'geometry': []}
    for rank in verdict['ranks']:
        for step in rank['model_partial']['numerators']:
            counts['atom'].append(step['atom_count'])
            counts['geometry'].append(step['geo_count'])
    assert sum(counts['atom']) == 3.0, 'the per-rank atom counts must differ'
    assert counts['geometry'] == [2.0, 0.0, 0.0, 0.0], counts['geometry']
    for rank in verdict['ranks']:
        assert rank['model_partial']['accumulation'] == 1
