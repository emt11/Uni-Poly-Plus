"""Model-free tests for the r3 check's classification and payload structure.

Plan MCL-PH-20260921-01/r3.  ``tests/test_mcl_ph_r3_reference_ddp.py`` spends the
round's model budget on a real two-rank launch; these tests cover the logic that
decides what that launch means -- the per-parameter classification of ``None``,
zero and non-zero gradients, the aggregation across ranks, and the persisted
payload -- from synthetic records, so a defect there is caught without a launch.

Nothing here builds a model, runs a forward or touches a process group.
"""
import importlib
import importlib.util
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tests'))

SPEC = importlib.util.spec_from_file_location(
    '_mcl_ph_r3_reference_ddp_check',
    ROOT / 'tests' / '_mcl_ph_r3_reference_ddp_check.py')
CHECK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECK)

BLOCKS = CHECK.BLOCKS


def block_entry(compared=(), unused=(), absent=(), mismatch=(), zero_both=(),
                deviation=0.0):
    """One block's per-rank classification, exactly as ``_compare`` builds it."""
    entry = {'parameters': len(compared) + len(unused) + len(absent) +
                           len(mismatch) + len(zero_both),
             'absent': list(absent), 'zero_on_both': list(zero_both),
             'compared_nonzero': list(compared), 'unused_on_this_rank': list(unused),
             'presence_mismatch': list(mismatch), 'max_deviation': float(deviation)}
    entry['verified'] = bool(entry['compared_nonzero']) and \
        not entry['presence_mismatch'] and entry['max_deviation'] <= 1e-4
    return entry


def config_record(blocks, rank, name='partial', *, local_counts=None,
                  denominators=None, loss=1.0, same_initial_state=True):
    """One rank's record for one configuration, shaped like ``configuration()``."""
    counts = local_counts or ({'atom': 2, 'geometry': 2} if rank == 0 else
                              {'atom': 1, 'geometry': 0})
    statistics = {task: {'numerator': float(counts[task]), 'effective_graphs': 1.0}
                  for task in ('atom', 'geometry')}
    return {'rank': rank, 'configuration': name, 'world_size': 2,
            'local_counts': counts,
            'update_denominators': denominators or {'atom': 3.0, 'geometry': 2.0},
            'denominators_source': 'update_level', 'denominator_statistics': statistics,
            'loss': loss, 'same_initial_state': same_initial_state, 'blocks': blocks}


def rank_report(rank, records):
    """One rank's gathered report: the configuration records it holds."""
    return {'rank': rank, 'world_size': 2, **records}


def blocks_where_decoders(unused_on_this_rank, deviation=0.0):
    """The observed shape, parameterised by whether the decoders are used."""
    blocks = {}
    for block in BLOCKS:
        blocks[block] = block_entry(compared=['%s.weight' % block])
    for block in ('local_decoder', 'nonbond_decoder'):
        blocks[block] = block_entry(
            unused=(['%s.bond.0.weight' % block] if unused_on_this_rank else []),
            compared=[] if unused_on_this_rank else ['%s.bond.0.weight' % block])
    blocks['encoder.o8'] = block_entry(compared=['encoder.o8.proj.weight'],
                                      absent=['encoder.o8.path_bias.0.weight'])
    return blocks


def two_ranks(blocks_rank0, blocks_rank1, name='partial', **kwargs):
    return [rank_report(0, {name: config_record(blocks_rank0, 0, name, **kwargs)}),
            rank_report(1, {name: config_record(blocks_rank1, 1, name, **kwargs)})]


def test_a_locally_unused_parameter_is_counted_and_not_verified_there():
    gathered = two_ranks(blocks_where_decoders(False), blocks_where_decoders(True))
    verdict = CHECK._verdict(gathered)
    assert verdict['status'] == 'PASS'
    decoders = verdict['coverage']['partial']['local_decoder']
    assert decoders['verified_on'] == [0]
    assert decoders['unused_on_this_rank'] == 1
    assert verdict['unused_parameter_count'] == 2
    assert all('no gradient on rank' in note for note in verdict['unused_parameter_note'])
    assert all('unexplained here' in note for note in verdict['unused_parameter_note'])


def test_a_parameter_without_a_gradient_everywhere_is_absent_not_unused():
    gathered = two_ranks(blocks_where_decoders(False), blocks_where_decoders(False))
    verdict = CHECK._verdict(gathered)
    o8 = verdict['coverage']['partial']['encoder.o8']
    assert o8['absent'] == 1
    assert o8['unused_on_this_rank'] == 0
    decoders = verdict['coverage']['partial']['local_decoder']
    assert decoders['verified_on'] == [0, 1]
    assert decoders['unused_on_this_rank'] == 0
    assert verdict['unused_parameter_count'] == 0


def test_a_gradient_the_reference_does_not_have_is_a_problem():
    blocks = blocks_where_decoders(False)
    blocks['encoder.fusion'] = block_entry(mismatch=['encoder.fusion.gate'])
    verdict = CHECK._verdict(two_ranks(blocks, blocks_where_decoders(False)))
    assert verdict['status'] == 'FAIL'
    assert any('has a gradient the reference does not' in problem
               for problem in verdict['problems'])


def test_a_deviation_beyond_the_tolerance_is_a_problem():
    blocks = blocks_where_decoders(False)
    blocks['encoder.fusion'] = block_entry(compared=['encoder.fusion.gate'],
                                           deviation=1e-3)
    verdict = CHECK._verdict(two_ranks(blocks, blocks_where_decoders(False)))
    assert verdict['status'] == 'FAIL'
    assert any('deviation' in problem for problem in verdict['problems'])


def test_a_block_whose_reference_is_zero_is_not_verified():
    """Both ranks carry a zero reference on that block: nothing was compared."""
    blocks = blocks_where_decoders(False)
    blocks['local_decoder'] = block_entry(zero_both=['local_decoder.bond.0.weight'])
    verdict = CHECK._verdict(two_ranks(blocks, blocks))
    local = verdict['coverage']['partial']['local_decoder']
    assert local['compared_nonzero'] == 0
    assert local['verified_on'] == []
    assert local['zero_on_both'] == 1
    assert verdict['status'] == 'PASS'


def test_a_differing_initial_state_is_a_problem_on_any_rank():
    gathered = two_ranks(blocks_where_decoders(False), blocks_where_decoders(False))
    gathered[1]['partial']['same_initial_state'] = False
    verdict = CHECK._verdict(gathered)
    assert verdict['status'] == 'FAIL'
    assert any('differs from the tested one' in problem for problem in verdict['problems'])


def test_the_payload_keeps_the_per_rank_runtime_evidence():
    """The structure the real launch persists, from the same synthetic records."""
    gathered = two_ranks(blocks_where_decoders(False), blocks_where_decoders(True))
    payload = CHECK._payload(gathered, CHECK._verdict(gathered))
    assert payload['configurations'] == ['partial']
    partial = payload['verdicts']['real_ddp_runtime']['configurations']['partial']
    assert partial['local_counts'] == {'atom': 2, 'geometry': 2}
    assert partial['update_denominators'] == {'atom': 3.0, 'geometry': 2.0}
    assert partial['denominators_source'] == 'update_level'
    assert sorted(partial['per_rank']) == ['0', '1']
    assert partial['per_rank']['1']['local_counts'] == {'atom': 1, 'geometry': 0}
    assert partial['per_rank']['1']['numerator']['geometry'] == 0.0
    assert partial['per_rank']['0']['numerator']['geometry'] == 2.0
    assert payload['verdicts']['analytic_formula_consistency']['re_run'] is False
    assert payload['verdicts']['model_gradient_reference']['status'] == 'PASS'


def test_the_payload_reports_failure_when_a_problem_exists():
    blocks = blocks_where_decoders(False)
    blocks['atom_head'] = block_entry(compared=['atom_head.weight'], deviation=1.0)
    gathered = two_ranks(blocks, blocks_where_decoders(False))
    payload = CHECK._payload(gathered, CHECK._verdict(gathered))
    assert payload['status'] == 'FAIL'
    assert payload['verdicts']['real_ddp_runtime']['status'] == 'FAIL'


def test_block_lookup_covers_every_compared_parameter():
    for block in BLOCKS:
        assert CHECK._block_of(block + '.weight') == block
    assert CHECK._block_of('something_else.weight') is None


def test_the_launch_test_assertions_hold_on_the_observed_shape():
    """Replay the launch test's expectations without spending a second launch.

    The shape is the one the r3 run observed: every block compared and verified on
    both ranks in ``partial`` (rank 1's own forward never reaches the decoders, but
    the reducer hands it the averaged gradient), and the decoders' reference zero
    in ``all_zero``.  The alternative -- the decoders left without a gradient on
    rank 1 -- is the case the launch test deliberately does *not* accept under the
    production configuration, and it is covered separately above.
    """
    launch = importlib.import_module('test_mcl_ph_r3_reference_ddp')
    zero_blocks = blocks_where_decoders(False)
    for block in ('local_decoder', 'nonbond_decoder'):
        zero_blocks[block] = block_entry(zero_both=['%s.bond.0.weight' % block])
    gathered = []
    for rank in (0, 1):
        blocks = blocks_where_decoders(False)
        gathered.append(rank_report(rank, {
            'partial': config_record(blocks, rank, 'partial'),
            'all_zero': config_record(
                zero_blocks, rank, 'all_zero', local_counts={'atom': 2, 'geometry': 0},
                denominators={'atom': 4.0, 'geometry': 0.0})}))
    payload = CHECK._payload(gathered, CHECK._verdict(gathered))
    assert sorted(payload['configurations']) == ['all_zero', 'partial']
    for name in ('test_the_three_verdicts_are_reported_separately',
                 'test_real_ddp_runs_with_a_rank_that_has_no_geometry',
                 'test_every_block_that_takes_part_in_the_compared_loss_is_compared',
                 'test_the_geometry_decoders_are_verified_on_both_ranks',
                 'test_a_zero_reference_block_is_not_claimed_as_verified',
                 'test_unused_parameter_notes_carry_no_unsupported_skip_claim'):
        getattr(launch, name)(payload)


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q']))
