"""r4 zero-update verification of the router diagnostics switch.

Six forward passes over the tiny two-graph fixture, all on CPU, no backward and
no optimizer step.  They check the three properties the r4 order asks for: the
switch now reaches the branch, the collected fields are finite and mode-correct,
and turning monitoring on changes neither the model output nor the RNG stream.
"""
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tests'))

from test_mcl_ph_modules import build_batch, build_labels, make_pretrainer  # noqa: E402

from src.modules.mcl_ph import TopologyRouter  # noqa: E402

GRAPH_KEYS = ('router_logits_mean', 'router_logits_std', 'router_logits_min',
              'router_logits_max', 'router_probability_mean', 'router_probability_std')


def _forward(mode='gate', *, diagnostics, dense_updates=500, router_mode=None):
    model = make_pretrainer(mode, collect_diagnostics=diagnostics,
                            router_dense_updates=dense_updates)
    model.train()
    if router_mode is not None:
        model.set_router_mode(router_mode, 1)
    batch, labels = build_batch(), build_labels(build_batch())
    torch.manual_seed(1234)
    report = model(batch, labels)
    return model, report, torch.get_rng_state()


def test_the_switch_reaches_the_branch_and_the_fields_are_finite():
    model, report, _ = _forward(diagnostics=True)
    assert model.encoder.branch.collect_diagnostics is True
    router = report['diagnostics']['router']
    assert isinstance(router, dict) and router, 'router diagnostics are still empty'
    assert router['router_mode'] == TopologyRouter.DENSE
    for key in GRAPH_KEYS:
        values = router[key]
        assert len(values) == 3, key
        assert all(isinstance(value, float) and torch.isfinite(torch.tensor(value))
                   for value in values), key
    # Soft probabilities are a simplex: three non-negative weights per graph.
    assert all(0.0 <= value <= 1.0 for value in router['router_probability_mean'])
    assert sum(router['router_probability_mean']) == pytest.approx(1.0, abs=1e-6)
    for key in ('router_entropy_mean', 'router_entropy_std', 'router_entropy_min',
                'router_entropy_max'):
        assert torch.isfinite(torch.tensor(router[key])), key
    assert 0.0 <= router['router_entropy_min'] <= router['router_entropy_max']
    assert router['readout_valid_graphs'] == 2


def test_dense_mode_reports_no_hard_selection_instead_of_inventing_one():
    model, report, _ = _forward(diagnostics=True)
    router = report['diagnostics']['router']
    assert router['router_hard_selection'] is None
    assert router['router_hard_selection_note'].startswith('not_applicable')
    # The runner copies exactly this block into the per-step ``monitoring``
    # field, so the honest dense-mode marker is what a step record will show.
    assert model.last_diagnostics is report['diagnostics']
    assert model.last_diagnostics['router']['router_hard_selection'] is None


def test_top_two_mode_reports_the_hard_selection_counts():
    _, report, _ = _forward(diagnostics=True, dense_updates=0,
                            router_mode=TopologyRouter.TOP2)
    router = report['diagnostics']['router']
    assert router['router_mode'] == TopologyRouter.TOP2
    counts = router['router_hard_selection']
    assert isinstance(counts, list) and len(counts) == 3
    # Two graphs, two kept experts each: the counts are selections, not graphs.
    assert sum(counts) == 4 and all(isinstance(value, int) for value in counts)
    assert router['router_hard_selection_note'].startswith('counts')


@pytest.mark.parametrize('fusion_mode,dense_updates,router_mode', [
    ('gate', 500, TopologyRouter.DENSE),
    ('gate', 0, TopologyRouter.TOP2),
])
def test_monitoring_changes_neither_the_output_nor_the_random_stream(
        fusion_mode, dense_updates, router_mode):
    off_model, off_report, off_rng = _forward(fusion_mode, diagnostics=False,
                                              dense_updates=dense_updates,
                                              router_mode=router_mode)
    on_model, on_report, on_rng = _forward(fusion_mode, diagnostics=True,
                                           dense_updates=dense_updates,
                                           router_mode=router_mode)
    assert 'diagnostics' not in off_report and 'monitoring' not in off_report
    for key in ('fused', 'atom_states', 'mixed', 'alpha'):
        assert torch.equal(off_report[key], on_report[key]), key
    assert torch.equal(off_rng, on_rng), 'the observation consumed randomness'
    assert all(torch.equal(off_model.state_dict()[name], on_model.state_dict()[name])
               for name in off_model.state_dict())
