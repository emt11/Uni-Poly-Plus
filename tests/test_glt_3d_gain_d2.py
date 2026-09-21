"""D2 arm tests for GLT-3D-GAIN-20260921-01 (r3).

Structural checks run on synthetic model instances; the geometry-independence
checks reuse the frozen xc/fold0 fixture already used by the D1 tests.  No
training update is executed here: the 4-epoch smoke budget belongs to the
launcher, not to this file.
"""
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.finetune_glt_3d_gain_d2 import (  # noqa: E402
    ARM_LRS, ARMS, COMMON_INIT_SEED, O8OnlyArm, build_arm, build_reference,
    effective_capacity, optimizer_for_arm,
)
from src.dataset.glt_dual import dual_glt_collate  # noqa: E402
from src.modules.glt_dual import build_dual_glt_model  # noqa: E402
from src.training.glt_dual_runtime import CleanLabeledDataset, open_source  # noqa: E402

COHORT = ROOT / 'data/processed/glt_dual_v2/downstream/cohort_1545eda5a8f6a1'
CACHE = ROOT / 'data/processed/mips_trimer_scage_downstream'
STATIC = ROOT / 'data/processed/glt_dual_v2/downstream/dual_static_v1'
SPLIT = Path('data/splits/mips_outer5_inner20')
CENTRE_ZERO_ROW = 0
GEOMETRY_FIELDS = (
    'bond_distance', 'bond_z_a', 'bond_z_b', 'bond_type', 'bond_center',
    'line_source', 'line_target', 'line_path', 'line_angle', 'line_mask',
    'line_path_group', 'line_is_self', 'geometry_valid', 'geometry_invalid_reason',
)


@pytest.fixture(scope='module')
def reference():
    """A common initialisation in the production shape, without the 150 MB
    deployment package: the arm relations under test are structural."""
    torch.manual_seed(COMMON_INIT_SEED)
    model = build_dual_glt_model('concat')
    model.eval()
    return model


@pytest.fixture(scope='module')
def arms(reference):
    return {arm: build_arm(arm, reference)[0] for arm in ARMS}


@pytest.fixture(scope='module')
def xc_unit():
    source, frame = open_source(str(COHORT), str(CACHE), task='xc',
                                dual_static_root=str(STATIC))
    dataset = CleanLabeledDataset(source, frame['label'].to_numpy(dtype=np.float64))
    manifest = json.loads((ROOT / SPLIT / 'xc.json').read_text(encoding='utf-8'))
    fold = manifest['folds'][0]
    train = [int(v) for v in fold['train_indices']]
    validation = [int(v) for v in fold['validation_indices']]
    samples = [dataset[index] for index in train[:4]]
    yield {'dataset': dataset, 'train': train, 'validation': validation,
           'samples': samples, 'manifest': manifest}
    source.close()


def test_arm_learning_rate_table_matches_the_plan():
    assert ARM_LRS['f2d'] == {'o8': 1e-5, 'glt': None, 'norm': 1e-4, 'head': 1e-4}
    assert ARM_LRS['fbase'] == {'o8': 1e-5, 'glt': 1e-5, 'norm': 1e-4, 'head': 1e-4}
    assert ARM_LRS['fnorm'] == {'o8': 1e-5, 'glt': 1e-5, 'norm': 1e-5, 'head': 1e-4}
    assert ARM_LRS['fstable'] == {'o8': 1e-5, 'glt': 3e-6, 'norm': 1e-5, 'head': 1e-4}


def test_arms_differ_only_in_the_declared_learning_rates():
    base, norm, stable = ARM_LRS['fbase'], ARM_LRS['fnorm'], ARM_LRS['fstable']
    assert {key for key in base if base[key] != norm[key]} == {'norm'}
    assert {key for key in norm if norm[key] != stable[key]} == {'glt'}
    assert 'glt' not in ARM_LRS['f2d'] or ARM_LRS['f2d']['glt'] is None


def test_f2d_does_not_construct_a_glt_branch(arms):
    model = arms['f2d']
    assert isinstance(model, O8OnlyArm)
    assert not hasattr(model, 'glt')
    assert not any(name.startswith('glt.') for name, _ in model.named_parameters())
    for arm in ('fbase', 'fnorm', 'fstable'):
        assert hasattr(arms[arm], 'glt')


def test_shared_initial_tensors_are_identical_across_arms(reference, arms):
    source = reference.state_dict()
    for arm, model in arms.items():
        target = model.state_dict()
        for name, value in target.items():
            assert name in source, f'{arm}: {name} has no reference source'
            assert torch.equal(value, source[name]), f'{arm}: {name} differs'
    # The 2D half, the normalisations and the head must agree between F2D and
    # FBASE by name, which is what makes the arms comparable.
    f2d, fbase = arms['f2d'].state_dict(), arms['fbase'].state_dict()
    for name in f2d:
        if name.startswith('o8.') or name.startswith('norm2.') or name.startswith('predictor.'):
            assert torch.equal(f2d[name], fbase[name]), name


def test_build_arm_reports_full_coverage(reference):
    for arm in ARMS:
        model, evidence = build_arm(arm, reference)
        assert evidence['unsourced'] == []
        assert set(evidence['copied']) == set(model.state_dict())
        if arm == 'f2d':
            assert any(name.startswith('glt.') for name in evidence['reference_only'])
        else:
            assert evidence['reference_only'] == []


def test_optimizer_groups_cover_every_trainable_tensor_exactly_once(arms):
    for arm, model in arms.items():
        optimizer, evidence = optimizer_for_arm(model, arm, weight_decay=0.02)
        grouped = [name for group in evidence for name in group['parameter_names']]
        trainable = sorted(name for name, p in model.named_parameters() if p.requires_grad)
        assert sorted(grouped) == trainable, arm
        assert len(grouped) == len(set(grouped)), f'{arm}: a parameter is in two groups'
        assert {group['name'] for group in evidence} == {'o8', 'norm', 'head'} | (
            {'glt'} if arm != 'f2d' else set())
        for group in evidence:
            assert group['weight_decay'] == 0.02
        lrs = {group['name']: group['lr'] for group in evidence}
        assert lrs['o8'] == ARM_LRS[arm]['o8'] and lrs['norm'] == ARM_LRS[arm]['norm']
        assert lrs['head'] == ARM_LRS[arm]['head']
        if arm != 'f2d':
            assert lrs['glt'] == ARM_LRS[arm]['glt']
        optimizer.zero_grad(set_to_none=True)


def test_optimizer_rejects_declared_glt_lr_without_a_glt_branch(reference):
    model, _ = build_arm('f2d', reference)
    with pytest.raises(ValueError, match='GLT lr without a GLT branch'):
        optimizer_for_arm(model, 'fbase', weight_decay=0.02)


def test_fbase_matches_the_reference_forward(reference, arms, xc_unit):
    batch = dual_glt_collate(xc_unit['samples'])
    reference.eval()
    fbase = arms['fbase'].eval()
    with torch.no_grad():
        expected = reference(batch)
        observed = fbase(batch)
    assert torch.equal(observed, expected)


def test_f2d_reports_dead_head_capacity(reference, arms):
    f2d = effective_capacity(arms['f2d'])
    assert f2d['dead_head_parameters'] == 512 * 512
    assert f2d['effective_parameters'] == f2d['total_parameters'] - 512 * 512
    base = effective_capacity(arms['fbase'])
    assert base['dead_head_parameters'] == 0
    assert base['effective_parameters'] == base['total_parameters']


def test_f2d_predicts_when_every_geometry_field_is_absent(arms, xc_unit):
    batch = dual_glt_collate(xc_unit['samples'])
    for name in GEOMETRY_FIELDS:
        if hasattr(batch, name):
            delattr(batch, name)
    model = arms['f2d'].eval()
    with torch.no_grad():
        output = model(batch)
    assert output.shape == (len(xc_unit['samples']), 1)
    assert torch.isfinite(output).all()


def test_fbase_requires_the_geometry_fields(arms, xc_unit):
    """The negative control for the F2D test above: the base arm does read 3D."""
    batch = dual_glt_collate(xc_unit['samples'])
    for name in GEOMETRY_FIELDS:
        if hasattr(batch, name):
            delattr(batch, name)
    with pytest.raises(AttributeError):
        with torch.no_grad():
            arms['fbase'](batch)


def test_centre_bond_zero_sample_keeps_a_finite_prediction(arms, xc_unit):
    model = arms['f2d'].eval()
    sample = xc_unit['samples'][CENTRE_ZERO_ROW]
    assert int(sample.bond_center.sum()) == 0
    assert bool(sample.geometry_valid) is True
    batch = dual_glt_collate([sample])
    with torch.no_grad():
        output = model(batch)
    assert output.shape == (1, 1)
    assert torch.isfinite(output).all()


def test_split_indices_are_disjoint_and_cover_the_fold(xc_unit):
    manifest = xc_unit['manifest']
    fold = manifest['folds'][0]
    train = set(int(v) for v in fold['train_indices'])
    validation = set(int(v) for v in fold['validation_indices'])
    test = set(int(v) for v in fold['test_indices'])
    assert not (train & validation) and not (train & test) and not (validation & test)
    assert len(train) + len(validation) + len(test) == int(manifest['sample_count'])


def test_best_checkpoint_round_trips_with_strict_loading(reference, arms):
    for arm in ARMS:
        payload = {'state_dict': {k: v.clone() for k, v in arms[arm].state_dict().items()},
                   'arm': arm}
        if arm == 'f2d':
            fresh = O8OnlyArm()
        else:
            fresh = build_dual_glt_model('concat')
        result = fresh.load_state_dict(payload['state_dict'], strict=True)
        assert result.missing_keys == [] and result.unexpected_keys == []

