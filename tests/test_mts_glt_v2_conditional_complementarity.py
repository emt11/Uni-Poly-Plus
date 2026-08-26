import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "analyze_mts_glt_v2_conditional_complementarity.py"
)
SPEC = importlib.util.spec_from_file_location("conditional_complementarity", SCRIPT)
AUDIT = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(AUDIT)


def test_inner_split_is_deterministic_and_disjoint():
    indices = np.arange(50, dtype=np.int64)
    train_a, validation_a = AUDIT.inner_train_validation(indices, 2, 3, 42)
    train_b, validation_b = AUDIT.inner_train_validation(indices, 2, 3, 42)

    np.testing.assert_array_equal(train_a, train_b)
    np.testing.assert_array_equal(validation_a, validation_b)
    assert np.intersect1d(train_a, validation_a).size == 0
    np.testing.assert_array_equal(
        np.sort(np.concatenate((train_a, validation_a))), indices
    )


def test_ridge_map_recovers_linear_relationship():
    rng = np.random.default_rng(7)
    x = rng.normal(size=(120, 6)).astype(np.float32)
    weight = rng.normal(size=(6, 4)).astype(np.float32)
    y = x @ weight + 0.25

    model = AUDIT.fit_ridge(x[:80], y[:80], x[80:100], y[80:100])
    prediction = model.predict(x[100:])

    np.testing.assert_allclose(prediction, y[100:], rtol=2e-3, atol=2e-3)
    assert model.alpha in AUDIT.ALPHAS


def test_residual_metrics_separate_predictable_and_private_components():
    rng = np.random.default_rng(11)
    predictable = rng.normal(size=(64, 5))
    private = rng.normal(scale=0.1, size=(64, 5))
    target = predictable + private

    metrics, residual = AUDIT.residual_metrics(
        target, predictable, target.mean(axis=0, keepdims=True)
    )

    np.testing.assert_allclose(residual, private)
    assert 0.0 < metrics["residual_fraction"] < 1.0
    assert metrics["raw_effective_rank"] > 0.0
    assert metrics["predictable_effective_rank"] > 0.0
    assert metrics["residual_effective_rank"] > 0.0
