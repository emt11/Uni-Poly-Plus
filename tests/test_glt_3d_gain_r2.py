"""r2 numerical-robustness tests for the D1 CPU probes.

Synthetic only: these tests never read the saved features, never build a model
and never touch the 294-fit real-data budget.  They cover exactly the changes
this revision makes: float64 promotion, train-only scaler fitting, rank-deficient
input against an independent SVD reference, OOF inner-fold coverage, and the
pre-fit budget guard.
"""
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.diagnose_glt_3d_gain_r2 import (  # noqa: E402
    FitBudget, cross_fitted_oof, fit_ridge, r2_original, select,
)
from scripts.verify_glt_3d_gain_d1_features import assert_zero_z3_rows  # noqa: E402


def _manual_svd_ridge(x_train, y_train, x_validation, alpha):
    """Independent reference: train-only standardisation plus the explicit SVD
    ridge solution for an intercept model.  Uses no Ridge estimator."""
    from sklearn.preprocessing import StandardScaler

    feature_scaler = StandardScaler().fit(x_train)
    label_scaler = StandardScaler().fit(y_train.reshape(-1, 1))
    x = feature_scaler.transform(x_train)
    y = label_scaler.transform(y_train.reshape(-1, 1)).ravel()

    x_centered = x - x.mean(axis=0, keepdims=True)
    y_centered = y - y.mean()
    u, singular, vt = np.linalg.svd(x_centered, full_matrices=False)
    damping = np.zeros_like(singular)
    keep = singular > 1e-15
    damping[keep] = singular[keep] / (singular[keep] ** 2 + float(alpha))
    coefficient = vt.T @ (damping * (u.T @ y_centered))
    intercept = y.mean() - x.mean(axis=0) @ coefficient

    scaled = feature_scaler.transform(x_validation) @ coefficient + intercept
    return label_scaler.inverse_transform(scaled.reshape(-1, 1)).reshape(-1)


def test_float32_inputs_are_promoted_and_match_float64():
    rng = np.random.default_rng(0)
    x32 = rng.normal(size=(40, 6)).astype(np.float32)
    y32 = rng.normal(size=40).astype(np.float32)
    budget = FitBudget(10)
    from32 = fit_ridge(x32, y32, x32, 1.0, budget, 'test')
    assert from32.dtype == np.float64
    assert np.isfinite(from32).all()
    from64 = fit_ridge(x32.astype(np.float64), y32.astype(np.float64),
                       x32.astype(np.float64), 1.0, budget, 'test')
    assert np.allclose(from32, from64, rtol=0, atol=0)


def test_scalers_fit_train_only():
    rng = np.random.default_rng(1)
    x_train = rng.normal(size=(50, 4)) * 3 + 10
    y_train = rng.normal(size=50) * 5 - 2
    x_validation = rng.normal(size=(12, 4)) * 40 - 100
    budget = FitBudget(10)

    observed = fit_ridge(x_train, y_train, x_validation, 1.0, budget, 'test')
    expected = _manual_svd_ridge(x_train, y_train, x_validation, 1.0)
    assert np.allclose(observed, expected, rtol=1e-10, atol=1e-10)

    point = np.array([[1.0, 2.0, 3.0, 4.0]])
    alone = fit_ridge(x_train, y_train, point, 1.0, budget, 'test')
    with_others = fit_ridge(x_train, y_train, np.vstack([point, x_validation]), 1.0,
                            budget, 'test')
    assert np.allclose(alone[0], with_others[0], rtol=1e-12, atol=1e-12)


def test_rank_deficient_fixture_is_finite_and_matches_svd_reference():
    rng = np.random.default_rng(2)
    base = rng.normal(size=(60, 3))
    # Duplicate columns make the design rank deficient by construction.
    x_train = np.hstack([base, base, base[:, :1]])
    y_train = base[:, 0] * 2.0 - base[:, 1] + 0.1
    x_validation = np.hstack([rng.normal(size=(15, 3))] * 1)
    x_validation = np.hstack([x_validation, x_validation, x_validation[:, :1]])
    assert np.linalg.matrix_rank(x_train) < x_train.shape[1]

    budget = FitBudget(10)
    for alpha in (0.01, 1.0, 1000.0):
        observed = fit_ridge(x_train, y_train, x_validation, alpha, budget, 'test')
        assert np.isfinite(observed).all()
        expected = _manual_svd_ridge(x_train, y_train, x_validation, alpha)
        assert np.allclose(observed, expected, rtol=1e-9, atol=1e-9)
    assert budget.count == 3


def test_oof_inner_folds_fill_every_row_exactly_once():
    from sklearn.model_selection import KFold

    rng = np.random.default_rng(3)
    y = rng.normal(size=47)
    filled = np.zeros(len(y), dtype=np.int64)
    for _, inner_test in KFold(3, shuffle=True, random_state=42).split(np.zeros((len(y), 1))):
        filled[inner_test] += 1
    assert bool((filled == 1).all())
    assert int(filled.sum()) == len(y)


def test_budget_refuses_before_the_next_fit():
    rng = np.random.default_rng(4)
    x = rng.normal(size=(20, 3))
    y = rng.normal(size=20)
    budget = FitBudget(3)
    for _ in range(3):
        fit_ridge(x, y, x, 1.0, budget, 'test')
    assert budget.count == 3
    with pytest.raises(RuntimeError, match='budget exhausted'):
        fit_ridge(x, y, x, 1.0, budget, 'test')
    assert budget.count == 3
    assert budget.report() == {'executed': 3, 'allowed': 3, 'by_category': {'test': 3}}


def test_fit_ridge_rejects_shape_mismatch():
    rng = np.random.default_rng(5)
    budget = FitBudget(5)
    with pytest.raises(ValueError, match='width mismatch'):
        fit_ridge(rng.normal(size=(10, 3)), rng.normal(size=10),
                  rng.normal(size=(4, 2)), 1.0, budget, 'test')
    with pytest.raises(ValueError, match='row count mismatch'):
        fit_ridge(rng.normal(size=(10, 3)), rng.normal(size=9),
                  rng.normal(size=(4, 3)), 1.0, budget, 'test')
    assert budget.count == 0


def test_select_marks_grid_boundaries():
    lower = {'0.01': {'validation_r2': 0.9}, '100.0': {'validation_r2': 0.5}}
    assert select(lower)['boundary'] == 'lower_bound'
    upper = {'0.01': {'validation_r2': 0.1}, '100.0': {'validation_r2': 0.5}}
    assert select(upper)['boundary'] == 'upper_bound'
    interior = {'0.01': {'validation_r2': 0.1}, '1.0': {'validation_r2': 0.9},
                '100.0': {'validation_r2': 0.5}}
    chosen = select(interior)
    assert chosen['boundary'] == 'interior'
    assert chosen['selected_alpha'] == pytest.approx(1.0)


def test_r2_original_is_in_label_units():
    actual = np.array([1.0, 2.0, 3.0, 4.0])
    assert r2_original(actual, actual) == pytest.approx(1.0)
    assert r2_original(np.full(4, actual.mean()), actual) == pytest.approx(0.0)


def test_zero_z3_assertion_covers_both_degenerate_populations():
    """r3 repair: the check must cover zero-centre and geometry-invalid rows
    separately, not only their intersection."""
    n = 4
    z3 = np.zeros((n, 3))
    z3[1] = [1.0, 0.0, 0.0]        # a row with a centre bond may be non-zero
    counts = np.asarray([0, 5, 0, 5])            # rows 0,2 have no centre bond
    valid = np.asarray([True, True, False, False])  # rows 2,3 have invalid geometry
    assert assert_zero_z3_rows(z3, counts, valid) == {
        'zero_centre': 2, 'geometry_invalid': 2, 'union': 3}

    # A non-zero z3 on a geometry-invalid row whose centre count is positive:
    # this is exactly the case the previous intersection-only test missed.
    bad_invalid = z3.copy()
    bad_invalid[3] = [0.5, 0.0, 0.0]
    with pytest.raises(ValueError, match='geometry-invalid'):
        assert_zero_z3_rows(bad_invalid, counts, valid)

    bad_centre = z3.copy()
    bad_centre[0] = [0.0, 2.0, 0.0]
    with pytest.raises(ValueError, match='zero-centre'):
        assert_zero_z3_rows(bad_centre, counts, valid)

    with pytest.raises(ValueError, match='rows disagree'):
        assert_zero_z3_rows(z3, counts[:3], valid)


def test_cross_fitted_oof_fills_once_and_excludes_inner_test_labels():
    """r3 repair: exercise the production OOF fill directly, not just KFold."""
    from sklearn.model_selection import KFold

    rng = np.random.default_rng(7)
    n, d = 45, 6
    x = rng.normal(size=(n, d))
    y = rng.normal(size=n)
    budget = FitBudget(20)
    oof, filled = cross_fitted_oof(x, y, 1.0, budget, 'oof', folds=3, seed=99)
    assert bool((filled == 1).all())
    assert int(filled.sum()) == n
    assert budget.count == 3

    # Independent replay with the same splitter: identical fill positions, and
    # an explicit assertion that no inner test index enters its own fit.
    replay = np.zeros(n)
    for inner_train, inner_test in KFold(3, shuffle=True, random_state=99).split(x):
        assert not (set(inner_train.tolist()) & set(inner_test.tolist()))
        replay[inner_test] = fit_ridge(x[inner_train], y[inner_train], x[inner_test],
                                       1.0, budget, 'oof')
    assert np.allclose(oof, replay, rtol=0, atol=0)
    assert budget.count == 6


def test_cross_fitted_oof_cannot_predict_labels_independent_of_features():
    from sklearn.metrics import r2_score

    rng = np.random.default_rng(13)
    n = 60
    x = rng.normal(size=(n, 8))
    y = rng.normal(size=n)          # independent of x by construction
    budget = FitBudget(10)
    oof, _ = cross_fitted_oof(x, y, 1.0, budget, 'oof', folds=3, seed=42)
    # A model that never saw a row's own label cannot predict pure noise.
    assert r2_score(y, oof) < 0.2
