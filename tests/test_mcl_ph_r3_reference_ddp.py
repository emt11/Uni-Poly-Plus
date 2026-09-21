"""Real-DDP runtime and gradient-reference test for plan MCL-PH-20260921-01/r3.

``tests/_mcl_ph_r3_reference_ddp_check.py`` is launched here with ``torchrun`` on
two gloo ranks, and its three verdicts are asserted.  The launch performs the
real ``DistributedDataParallel`` path (``find_unused_parameters=True``, as in
production) in a configuration where one rank carries no geometry at all, and
compares the averaged gradient of every participating parameter block against one
single-process computation over both ranks' fixtures under the same update-level
denominators.

Cost: one launch of both configurations runs twelve model forwards and twelve
model backwards (two configurations x two ranks x (one model pass + two reference
passes) with backward on each).  Set ``MCL_PH_R3_CONFIGURATIONS=partial`` to
launch half of that when the budget of a round does not allow the full pair; the
``all-zero`` assertions are then skipped rather than silently passed.

The locally-unused case is documented rather than assumed: production leaves
``skip_all_reduce_unused_params`` at its default ``False``, so DDP reduces a
parameter that is unused on one rank, and that rank carries the rank-averaged
gradient instead of ``None``.  ``tests/_mcl_ph_r3_unused_probe.py`` is the
single-process counterpart that shows the same fixture leaving the decoder
parameters with ``grad is None`` when no reducer is involved.
"""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CHECK = ROOT / 'tests' / '_mcl_ph_r3_reference_ddp_check.py'
OUTPUT = ROOT / 'logs' / 'mcl_ph_20260921' / 'r3_reference_ddp_payload.json'
BLOCKS = ('encoder.o8', 'encoder.branch.experts', 'encoder.branch.router',
          'encoder.fusion', 'local_decoder', 'nonbond_decoder', 'atom_head')


@pytest.fixture(scope='module')
def verdict():
    environment = dict(os.environ, OMP_NUM_THREADS='1')
    environment.pop('MASTER_PORT', None)
    command = [sys.executable, '-m', 'torch.distributed.run', '--standalone',
               '--nproc_per_node', '2', str(CHECK)]
    selected = environment.get('MCL_PH_R3_CONFIGURATIONS')
    if selected:
        command += ['--configurations', *selected.split()]
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    command += ['--output', str(OUTPUT)]
    completed = subprocess.run(command, cwd=str(ROOT), capture_output=True,
                               text=True, env=environment, timeout=600)
    payload = None
    for line in completed.stdout.splitlines():
        if line.startswith('{"status"'):
            payload = json.loads(line)
    assert payload is not None, completed.stdout[-4000:] + completed.stderr[-2000:]
    assert completed.returncode == 0, completed.stderr[-4000:]
    return payload


def _runtime(verdict, name):
    configurations = verdict['verdicts']['real_ddp_runtime']['configurations']
    if name not in configurations:
        pytest.skip(f'the {name} configuration was not part of this launch')
    return configurations[name]


def test_the_three_verdicts_are_reported_separately(verdict):
    verdicts = verdict['verdicts']
    assert verdicts['analytic_formula_consistency']['status'] == 'CARRIED_OVER_FROM_R2'
    assert verdicts['analytic_formula_consistency']['re_run'] is False
    assert 'model_gradient_reference' in verdicts
    assert 'real_ddp_runtime' in verdicts


def test_real_ddp_runs_with_a_rank_that_has_no_geometry(verdict):
    assert verdict['status'] == 'PASS', verdict['verdicts']['model_gradient_reference']
    runtime = verdict['verdicts']['real_ddp_runtime']
    assert runtime['status'] == 'PASS'
    partial = _runtime(verdict, 'partial')
    # The reported counts belong to the rank that carries the geometry.
    assert partial['local_counts'] == {'atom': 2, 'geometry': 2}
    # The update-level denominators are the sum over both ranks: three atoms is
    # rank 0's two plus the geometry-free rank's one, and two geometric graphs is
    # rank 0's two plus that rank's none.
    assert partial['update_denominators'] == {'atom': 3.0, 'geometry': 2.0}
    assert partial['denominators_source'] == 'update_level'
    ranks = partial['per_rank']
    assert sorted(ranks) == ['0', '1']
    assert ranks['0']['local_counts'] == {'atom': 2, 'geometry': 2}
    assert ranks['1']['local_counts'] == {'atom': 1, 'geometry': 0}
    # Rank 1's own geometry numerator is exactly zero, so its decoders cannot
    # have been reached by its own forward.
    assert ranks['1']['numerator']['geometry'] == 0.0
    assert ranks['0']['numerator']['geometry'] > 0.0
    assert ranks['1']['numerator']['atom'] > 0.0
    for entry in ranks.values():
        assert entry['same_initial_state'] is True
        assert entry['update_denominators'] == {'atom': 3.0, 'geometry': 2.0}
    zero = _runtime(verdict, 'all_zero')
    assert zero['update_denominators'] == {'atom': 4.0, 'geometry': 0.0}
    assert zero['denominators_source'] == 'update_level'


def test_every_block_that_takes_part_in_the_compared_loss_is_compared(verdict):
    coverage = verdict['verdicts']['model_gradient_reference']['coverage']['partial']
    for block in BLOCKS:
        entry = coverage[block]
        assert entry['parameters'] > 0, block
        assert entry['max_deviation'] <= 1e-4, (block, entry['max_deviation'])
        assert entry['verified_on'], block


def test_the_geometry_decoders_are_verified_on_both_ranks(verdict):
    """Production reduces a parameter that only one rank uses.

    ``find_unused_parameters=True`` with ``skip_all_reduce_unused_params`` at its
    default ``False`` reduces the locally-used map across ranks, so the decoders
    unused by rank 1's own forward still carry a gradient there -- the reduced
    average, which is what the reference computes.  A ``None`` gradient on that
    rank would contradict that, so this test fails loudly if a launch ever leaves
    the decoders unverified on rank 1 (which is what the alternative
    ``MCL_PH_R3_CONFIGURATIONS``-gated shapes in
    ``tests/test_mcl_ph_r3_payload.py`` cover instead).
    """
    coverage = verdict['verdicts']['model_gradient_reference']['coverage']['partial']
    for block in ('local_decoder', 'nonbond_decoder'):
        entry = coverage[block]
        assert entry['verified_on'] == [0, 1], (block, entry)
        assert entry['unused_on_this_rank'] == 0, (block, entry)
        assert entry['compared_nonzero'] > 0, block


def test_a_zero_reference_block_is_not_claimed_as_verified(verdict):
    """With no geometry anywhere the decoders hold a zero loss: not verified."""
    coverage = verdict['verdicts']['model_gradient_reference']['coverage']['all_zero']
    for block in ('local_decoder', 'nonbond_decoder'):
        assert coverage[block]['compared_nonzero'] == 0, block
        assert coverage[block]['verified_on'] == [], block
    assert coverage['atom_head']['verified_on'] == [0, 1]


def test_unused_parameter_notes_carry_no_unsupported_skip_claim(verdict):
    reference = verdict['verdicts']['model_gradient_reference']
    assert reference['unused_parameter_count'] >= len(reference['unused_parameter_note'])
    assert len(reference['unused_parameter_note']) <= 8
    for note in reference['unused_parameter_note']:
        assert 'no gradient on rank' in note
        assert 'unexplained here' in note
