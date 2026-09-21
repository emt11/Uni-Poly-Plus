"""r7C regression: the deployment checker's shared-init namespace mapping.

``tests/_mcl_ph_r3_deployment_check.py`` compares every deployment bundle against
the shared initialisation snapshot.  The snapshot records names relative to the
full pre-trainer (``encoder.branch.*``, ``atom_head.*``) while the bundle stores
``encoder.state_dict()`` (``branch.*``, ``o8.*``, ``fusion.*``), so the mapping
between the two namespaces is load-bearing: with the wrong one the comparison
shrinks to nothing and the checker reports "no update happened" on a run that
trained.

Every case below is built from synthetic tensors on the CPU: no model, no
dataset, no GPU, no training process, no checkpoint on disk.  The records are
assembled with the checker's own helpers, so the rules under test are the ones
the checker actually applies.
"""
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tests'))

import _mcl_ph_r3_deployment_check as checker  # noqa: E402

SHA = 'a' * 64


def _tensor(*values):
    return torch.tensor(list(values), dtype=torch.float32)


# The two namespaces of the same three tensors: the snapshot carries the
# pre-trainer prefix, the bundle does not, and the pre-training heads have no
# counterpart in the bundle at all.
SHARED = {'encoder.branch.a': _tensor(1.0, 2.0),
          'encoder.branch.b': _tensor(3.0),
          'encoder.fusion.expand.weight': _tensor(4.0),
          'atom_head.x': _tensor(9.0),
          'nonbond_decoder.y': _tensor(9.0)}
PACKAGE = {'branch.a': _tensor(1.0, 2.0),
           'branch.b': _tensor(3.0),
           'fusion.expand.weight': _tensor(4.0),
           'o8.y': _tensor(0.0)}


def _record(shared, package, sha=SHA, inference_mode='top2', route='mcl_ph'):
    """The per-arm record the checker's rules consume, built from raw states."""
    mapped = checker.shared_encoder_state(shared)
    deltas, missing = checker.shared_encoder_deltas(mapped, package)
    return {'inference_mode': inference_mode,
            'training_route': route,
            'shared_encoder_tensors': len(mapped),
            'changed_shared_encoder_tensors': sum(1 for value in deltas.values()
                                                  if value > 0.0),
            'max_abs_delta_from_shared_init': max(deltas.values()) if deltas else 0.0,
            'shared_encoder_missing_keys': missing,
            'shared_new_init_sha256': sha}


def _mapped(record):
    return record['shared_encoder_tensors']


# ------------------------------------------------------------------ case 1
def test_mapping_keeps_only_the_encoder_namespace():
    mapped = checker.shared_encoder_state(SHARED)
    assert set(mapped) == {'branch.a', 'branch.b', 'fusion.expand.weight'}
    assert 'atom_head.x' not in mapped
    assert 'nonbond_decoder.y' not in mapped
    deltas, missing = checker.shared_encoder_deltas(mapped, PACKAGE)
    assert missing == []
    assert sorted(deltas) == ['branch.a', 'branch.b', 'fusion.expand.weight']
    assert _mapped(_record(SHARED, PACKAGE)) == 3


# ------------------------------------------------------------------ case 2
def test_every_mapped_tensor_unchanged_is_reported_as_no_update():
    record = _record(SHARED, PACKAGE)
    assert record['changed_shared_encoder_tensors'] == 0
    problems = checker.arm_problems('cat', record, SHA)
    assert any('no update happened' in problem for problem in problems)
    assert any('no shared encoder tensor differs' in problem for problem in problems)


# ------------------------------------------------------------------ case 3
def test_a_changed_mapped_tensor_is_recognised_as_an_update():
    updated = dict(PACKAGE)
    updated['branch.b'] = _tensor(3.0001)
    record = _record(SHARED, updated)
    assert record['changed_shared_encoder_tensors'] == 1
    assert record['max_abs_delta_from_shared_init'] > 0.0
    assert checker.arm_problems('cat', record, SHA) == []


# ------------------------------------------------------------------ case 4
def test_a_missing_mapped_key_fails_instead_of_shrinking_the_comparison():
    incomplete = {name: value for name, value in PACKAGE.items() if name != 'branch.b'}
    complete = dict(incomplete)
    complete['branch.a'] = _tensor(1.0, 2.5)
    record = _record(SHARED, complete)
    assert record['shared_encoder_missing_keys'] == ['branch.b']
    problems = checker.arm_problems('cat', record, SHA)
    assert any('shared encoder keys are missing from the deployment package' in problem
               for problem in problems)
    assert any('branch.b' in problem for problem in problems)
    # One changed tensor must not mask the missing one: the missing-key rule is
    # not an intersection that quietly continues.
    assert record['changed_shared_encoder_tensors'] == 1


# ------------------------------------------------------------------ case 5
def test_fusion_parameters_are_not_shared_common_parameters():
    assert checker.FUSION_PREFIX == 'encoder.fusion.'
    names = {'encoder.fusion.expand.weight', 'encoder.branch.a', 'atom_head.x'}
    shared_names = checker.shared_initial_names(names)
    assert 'encoder.fusion.expand.weight' not in shared_names
    assert shared_names == ['atom_head.x', 'encoder.branch.a']
    # The old constant could never match a step_0000.json name, which is how a
    # fusion parameter would have been taken for a shared one.
    assert not 'encoder.fusion.expand.weight'.startswith('fusion.')


# ------------------------------------------------------- source contract
def test_the_checker_no_longer_reports_the_ambiguous_counters():
    source = Path(checker.__file__).read_text(encoding='utf-8')
    assert "'shared_tensors'" not in source
    assert "'changed_shared_tensors'" not in source
    assert "'shared_encoder_tensors'" in source
