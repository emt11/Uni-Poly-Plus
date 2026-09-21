"""Model-free tests for the P0 revision report's judgement.

Plan MCL-PH-20260921-01/r3, item B.  ``scripts/audit_mcl_ph_p0_randic.py`` decides
whether a recomputed statistics record may be called PASS; the defects that made
the r2 revision's judgement unsafe (a missing sample identity counting as a match,
a non-finite or out-of-range column, an incomplete sample set) are pinned here
with synthetic records.

The frozen r1 artifacts and the old report are never read, recomputed or modified
by these tests: only the judgement and the column naming are exercised.
"""
import importlib.util
import math
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SPEC = importlib.util.spec_from_file_location(
    'audit_mcl_ph_p0_randic', ROOT / 'scripts' / 'audit_mcl_ph_p0_randic.py')
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)

SHA = 'c0402dca' + '0' * 56


def columns(**overrides):
    """A complete, in-range column summary, matching ``_column_summary``."""
    summary = {name: {'min': 0.0, 'max': 1.0, 'mean': 0.5, 'std': 0.1,
                      'within_declared_range': True} for name in AUDIT.COLUMNS}
    for name, values in overrides.items():
        summary[name].update(values)
    return summary


def payload(**overrides):
    record = {'sample_set': {'ordered_key_sha256': SHA},
              'statistics': {'samples_requested': 4096, 'samples_used': 4096,
                             'partial': False, 'columns': columns()}}
    for key, value in overrides.items():
        record[key] = value
    return record


def frozen(sha=SHA):
    return {'sample_sets': {'ordered_key_sha256': sha}} if sha else {}


def test_a_complete_matching_record_is_the_only_pass():
    status, problems = AUDIT.judge(payload(), frozen())
    assert status == 'PASS', problems
    assert problems == []


def test_a_missing_sample_identity_of_the_run_is_not_a_match():
    record = payload()
    record.pop('sample_set')
    status, problems = AUDIT.judge(record, frozen())
    assert status == 'FAILED'
    assert any('does not record the sample identity' in problem for problem in problems)


def test_a_missing_frozen_identity_is_a_problem_not_a_silent_match():
    status, problems = AUDIT.judge(payload(), frozen(sha=None))
    assert status == 'FAILED'
    assert any('frozen audit records no sample identity' in problem for problem in problems)
    assert AUDIT._frozen_sample_sha({}) is None


def test_a_differing_sample_identity_is_a_problem():
    status, problems = AUDIT.judge(payload(), frozen(sha='f' * 64))
    assert status == 'FAILED'
    assert any('differs from the frozen audit' in problem for problem in problems)


@pytest.mark.parametrize('column,value', [('randic', float('nan')),
                                          ('beta1_norm', float('inf')),
                                          ('wiener', None)])
def test_a_non_finite_statistic_is_a_problem(column, value):
    summary = columns()
    summary[column]['mean'] = value
    record = payload()
    record['statistics']['columns'] = summary
    status, problems = AUDIT.judge(record, frozen())
    assert status == 'FAILED'
    assert any(f'{column}.mean is not finite' in problem for problem in problems)


def test_a_column_beyond_the_declared_range_is_a_problem():
    summary = columns()
    summary['beta1_norm']['max'] = 1.001
    summary['beta1_norm']['within_declared_range'] = False
    record = payload()
    record['statistics']['columns'] = summary
    status, problems = AUDIT.judge(record, frozen())
    assert status == 'FAILED'
    assert any('beta1_norm column leaves its declared range' in problem
               for problem in problems)
    assert any('flagged out of its declared range' in problem for problem in problems)


def test_a_column_flagging_the_range_it_does_not_leave_is_a_problem():
    summary = columns()
    summary['randic']['within_declared_range'] = False
    record = payload()
    record['statistics']['columns'] = summary
    status, problems = AUDIT.judge(record, frozen())
    assert status == 'FAILED'
    assert any('randic column is flagged out of its declared range' in problem
               for problem in problems)


def test_a_rounding_inside_the_tolerance_still_passes():
    summary = columns()
    summary['randic']['max'] = 1.0 + AUDIT.RANGE_TOLERANCE / 2.0
    record = payload()
    record['statistics']['columns'] = summary
    status, problems = AUDIT.judge(record, frozen())
    assert status == 'PASS', problems


def test_an_incomplete_sample_set_is_a_problem():
    record = payload()
    record['statistics'].update(samples_used=4095, partial=True)
    status, problems = AUDIT.judge(record, frozen())
    assert status == 'FAILED'
    assert any('incomplete (4095 of 4096)' in problem for problem in problems)


def test_a_missing_column_is_a_problem():
    summary = columns()
    summary.pop('beta1_norm')
    record = payload()
    record['statistics']['columns'] = summary
    status, problems = AUDIT.judge(record, frozen())
    assert status == 'FAILED'
    assert any('the beta1_norm column is missing' in problem for problem in problems)


def test_the_h1_column_is_named_for_its_normalization():
    """Column 4 normalizes by the H1 interval count, not by the edge count."""
    assert AUDIT.COLUMNS[-1] == 'beta1_norm'
    assert 'betti1_per_edge' not in AUDIT.COLUMNS
    assert set(AUDIT.COLUMN_DEFINITIONS) == set(AUDIT.COLUMNS)
    definition = AUDIT.COLUMN_DEFINITIONS['beta1_norm']
    assert 'active H1 intervals' in definition
    assert 'max(1, B1)' in definition
    # The naming bridge to the r2 report, worded as a name change only.
    assert 'named betti1_per_edge in the r2 revision report' in definition


def test_the_column_summary_flags_a_value_outside_the_unit_range():
    import numpy as np

    stack = np.zeros((2, 3, len(AUDIT.COLUMNS)), dtype=np.float32)
    stack[0, 0, -1] = 1.5
    summary = AUDIT._column_summary(stack)
    assert summary['beta1_norm']['within_declared_range'] is False
    assert summary['beta1_norm']['max'] == pytest.approx(1.5)
    assert summary['randic']['within_declared_range'] is True
    assert math.isfinite(summary['randic']['std'])


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q']))
