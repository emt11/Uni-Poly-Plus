import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "analyze_mts_glt_v2_conditional_property_probe.py"
)
SPEC = importlib.util.spec_from_file_location("conditional_property_probe", SCRIPT)
PROBE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(PROBE)


def test_scalar_ridge_selects_from_contract_and_recovers_linear_target():
    rng = np.random.default_rng(5)
    features = rng.normal(size=(100, 8))
    target = features @ rng.normal(size=8) + 0.7

    fitted = PROBE.fit_scalar_ridge(
        features[:60], target[:60], features[60:80], target[60:80]
    )

    np.testing.assert_allclose(
        fitted.predict(features[80:]), target[80:], rtol=3e-3, atol=3e-3
    )
    assert fitted.alpha in PROBE.PROPERTY_ALPHAS


def test_atom_residual_pool_is_graph_mean_and_ignores_invalid_atoms():
    residual = np.asarray([[1.0, 3.0], [3.0, 5.0], [100.0, 100.0], [8.0, 4.0]])
    graph_ids = np.asarray([0, 0, 0, 1])
    valid = np.asarray([True, True, False, True])

    pooled, counts = PROBE.pool_atom_residuals(residual, graph_ids, valid, 2)

    np.testing.assert_allclose(pooled, [[2.0, 4.0], [8.0, 4.0]])
    np.testing.assert_array_equal(counts, [2, 1])


def test_mechanical_signal_decision_uses_all_three_conditions():
    assert PROBE.decision(np.asarray([0.1] * 5 + [-0.01] * 3)) == "POSITIVE"
    assert PROBE.decision(np.asarray([0.1] * 4 + [-0.01] * 4)) == "NOT_ESTABLISHED"
