"""D1 local tests for GLT-3D-GAIN-20260921-01.

Only the behaviours this round can break: the exposed z2/z3 split against the
model's own fuse(), batch/order invariance, eval-state preservation, train-only
scaler fitting, the centre-bond-zero row and the denominator guard.  These are
not a full-suite run and do not establish any model-quality claim.
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

from scripts.diagnose_glt_3d_gain import (  # noqa: E402
    _relative_stats, build_model, check_fuse_equivalence, collect_geometry_statistics,
    deterministic_derangement, fit_ridge, lookup_angle_means, replace_angle,
    replace_distance, split_fuse,
)
from src.dataset.glt_dual import dual_glt_collate  # noqa: E402
from src.training.glt_dual_runtime import CleanLabeledDataset, open_source  # noqa: E402

COHORT = ROOT / 'data/processed/glt_dual_v2/downstream/cohort_1545eda5a8f6a1'
CACHE = ROOT / 'data/processed/mips_trimer_scage_downstream'
STATIC = ROOT / 'data/processed/glt_dual_v2/downstream/dual_static_v1'
SPLIT = ROOT / 'data/splits/mips_outer5_inner20'
DEPLOY = ROOT / 'results/glt_pred_20260918/s3b_formal/b_fp/pretrain/deploy_05000.pt'
# The audited geometry-valid row with no centre bond in xc/fold0 train.
CENTRE_ZERO_ROW = 0


@pytest.fixture(scope='module')
def deployment():
    model, meta = build_model('concat', str(DEPLOY), device=torch.device('cpu'))
    assert meta['kind'] == 'deployment'
    return model, meta


@pytest.fixture(scope='module')
def xc_unit():
    source, frame = open_source(str(COHORT), str(CACHE), task='xc',
                                dual_static_root=str(STATIC))
    dataset = CleanLabeledDataset(source, frame['label'].to_numpy(dtype=np.float64))
    manifest = json.loads((SPLIT / 'xc.json').read_text(encoding='utf-8'))
    train = [int(value) for value in manifest['folds'][0]['train_indices']]
    indices = train[:4]
    assert indices[CENTRE_ZERO_ROW] == CENTRE_ZERO_ROW
    samples = [dataset[index] for index in indices]
    yield {'source': source, 'dataset': dataset, 'train': train,
           'indices': indices, 'samples': samples}
    source.close()


def _encode(model, samples):
    batch = dual_glt_collate(samples)
    with torch.no_grad():
        encoded = model.encode(batch)
        z2, z3 = split_fuse(model, encoded)
    return batch, encoded, z2, z3


def test_split_fuse_matches_model_fuse(deployment, xc_unit):
    model, _ = deployment
    batch, encoded, z2, z3 = _encode(model, xc_unit['samples'])
    with torch.no_grad():
        deviation = check_fuse_equivalence(model, encoded)
    assert deviation <= 1e-5
    assert torch.allclose(torch.cat([z2, z3], -1), model.fuse(encoded), rtol=1e-5, atol=1e-5)


def test_repeated_forward_is_deterministic(deployment, xc_unit):
    model, _ = deployment
    _, _, z2a, z3a = _encode(model, xc_unit['samples'])
    _, _, z2b, z3b = _encode(model, xc_unit['samples'])
    assert torch.equal(z2a, z2b)
    assert torch.equal(z3a, z3b)


def test_order_and_batch_composition_invariance(deployment, xc_unit):
    model, _ = deployment
    samples = xc_unit['samples']
    _, _, _, forward = _encode(model, samples)
    _, _, _, reverse = _encode(model, list(reversed(samples)))
    assert torch.allclose(forward, torch.flip(reverse, [0]), rtol=1e-5, atol=1e-5)
    _, _, _, subset = _encode(model, samples[:2])
    assert torch.allclose(subset, forward[:2], rtol=1e-5, atol=1e-5)


def test_eval_forward_preserves_state_and_rng(deployment, xc_unit):
    model, _ = deployment
    before = {name: value.clone() for name, value in model.state_dict().items()}
    torch_state = torch.random.get_rng_state().clone()
    numpy_state = np.random.get_state()
    _encode(model, xc_unit['samples'])
    for name, value in model.state_dict().items():
        assert torch.equal(value, before[name]), name
    assert torch.equal(torch.random.get_rng_state(), torch_state)
    after = np.random.get_state()
    assert after[0] == numpy_state[0] and np.array_equal(after[1], numpy_state[1])


def test_centre_bond_zero_row_yields_exact_zero_z3(deployment, xc_unit):
    model, _ = deployment
    sample = xc_unit['samples'][CENTRE_ZERO_ROW]
    assert int(sample.bond_center.sum()) == 0
    assert bool(sample.geometry_valid) is True
    _, encoded, z2, z3 = _encode(model, [sample])
    assert bool(encoded['geometry_valid'][0]) is False
    assert torch.isfinite(z3).all() and torch.isfinite(z2).all()
    assert float(z3.abs().sum()) == 0.0


def test_interventions_do_not_touch_topology_labels_or_the_source_batch(deployment, xc_unit):
    model, _ = deployment
    dataset, samples = xc_unit['dataset'], xc_unit['samples']
    statistics = collect_geometry_statistics(dataset, xc_unit['indices'], 4, 5)
    batch = dual_glt_collate(samples)
    original_distance = batch.bond_distance.clone()
    original_angle = batch.line_angle.clone()
    replaced_distance = replace_distance(batch, statistics)
    replaced_angle = replace_angle(batch, statistics)
    for clone in (replaced_distance, replaced_angle):
        for name in ('bond_z_a', 'bond_z_b', 'bond_type', 'line_path', 'line_mask',
                     'bond_center', 'line_path_group', 'bond_batch'):
            assert torch.equal(getattr(clone, name), getattr(batch, name)), name
        assert torch.equal(clone.y, batch.y)
    assert torch.equal(batch.bond_distance, original_distance)
    assert torch.equal(batch.line_angle, original_angle)
    assert not torch.equal(replaced_distance.bond_distance, original_distance)
    assert torch.equal(replaced_distance.line_angle, original_angle)
    assert torch.equal(replaced_angle.bond_distance, original_distance)
    assert torch.isfinite(replaced_distance.bond_distance).all()
    assert torch.isfinite(replaced_angle.line_angle).all()
    assert (replaced_distance.bond_distance > 0).all()


def test_relative_stats_denominator_guard():
    stats = _relative_stats(np.zeros((2, 4)), np.zeros((2, 4)))
    assert np.isfinite(stats['relative']['mean'])
    assert np.isfinite(stats['raw']['max'])
    changed = np.ones((1, 4))
    guarded = _relative_stats(changed, np.zeros((1, 4)))
    assert np.isfinite(guarded['relative']['max'])
    assert guarded['raw']['max'] == pytest.approx(2.0)
    assert guarded['relative']['max'] == pytest.approx(2.0 / 1e-8)


def test_ridge_scalers_fit_train_only():
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(0)
    x_train = rng.normal(size=(30, 5)) * 3 + 5
    y_train = rng.normal(size=30)
    x_validation = rng.normal(size=(12, 5)) * 50 - 20

    feature_scaler = StandardScaler().fit(x_train)
    label_scaler = StandardScaler().fit(y_train.reshape(-1, 1))
    reference = Ridge(alpha=1.0).fit(
        feature_scaler.transform(x_train),
        label_scaler.transform(y_train.reshape(-1, 1)).ravel())
    expected = label_scaler.inverse_transform(
        reference.predict(feature_scaler.transform(x_validation)).reshape(-1, 1)).reshape(-1)
    assert np.allclose(fit_ridge(x_train, y_train, x_validation, 1.0), expected)

    point = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
    alone = fit_ridge(x_train, y_train, point, 1.0)
    with_others = fit_ridge(x_train, y_train, np.vstack([point, x_validation]), 1.0)
    assert np.allclose(alone[0], with_others[0])


def test_derangement_is_fixed_and_fixed_point_free():
    first = deterministic_derangement(37)
    second = deterministic_derangement(37)
    assert np.array_equal(first, second)
    assert not np.any(first == np.arange(37))
    assert sorted(first.tolist()) == list(range(37))
    with pytest.raises(ValueError):
        deterministic_derangement(1)


def test_rare_angle_group_falls_back_to_global_mean():
    statistics = {'angle_keys_sorted': np.asarray([5, 9], dtype=np.int64),
                  'angle_means_sorted': np.asarray([1.5, 2.5], dtype=np.float64),
                  'angle_global': 7.0}
    result = lookup_angle_means(np.asarray([5, 9, 12345], dtype=np.int64), statistics)
    assert result[0] == pytest.approx(1.5)
    assert result[1] == pytest.approx(2.5)
    assert result[2] == pytest.approx(7.0)
    empty = {'angle_keys_sorted': np.zeros(0, dtype=np.int64),
             'angle_means_sorted': np.zeros(0, dtype=np.float64), 'angle_global': 3.0}
    assert np.all(lookup_angle_means(np.asarray([1, 2], dtype=np.int64), empty) == 3.0)


def test_geometry_statistics_use_only_the_given_pool(xc_unit):
    dataset = xc_unit['dataset']
    first = collect_geometry_statistics(dataset, xc_unit['indices'][:2], 4, 5)
    second = collect_geometry_statistics(dataset, xc_unit['indices'][:4], 4, 5)
    assert first['summary']['bond_observations'] < second['summary']['bond_observations']
    for value in first['distance_by_type']:
        assert np.isfinite(value) and value > 0
    assert np.isfinite(first['angle_global'])
