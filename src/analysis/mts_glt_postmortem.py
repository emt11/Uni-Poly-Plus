"""Leakage-safe linear probes and representation diagnostics for MTS-GLT-v1."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import explained_variance_score, r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler


ALPHAS = np.asarray(
    [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0, 10000.0],
    dtype=np.float64,
)
UNMATCHED_SHIFTS = (1, 7, 31, 127, 257)


@dataclass
class StandardizedRidge:
    scaler: StandardScaler
    model: Ridge
    alpha: float

    def predict(self, features):
        return self.model.predict(self.scaler.transform(features))


def _splits(size, seed, n_splits=5):
    if int(size) < 2:
        raise ValueError("at least two samples are required")
    count = min(int(n_splits), int(size))
    if count < 2:
        raise ValueError("at least two folds are required")
    return list(KFold(count, shuffle=True, random_state=int(seed)).split(
        np.arange(int(size))
    ))


def _score(y_true, y_pred, multioutput=False):
    if multioutput:
        return float(explained_variance_score(
            y_true, y_pred, multioutput="uniform_average"
        ))
    return float(r2_score(y_true, y_pred))


def _ridge_path_predictions(train_x, train_y, test_x, alphas):
    """Evaluate a ridge alpha path from one economy SVD.

    This is algebraically the ordinary centered L2 ridge solution.  Reusing
    the decomposition avoids refitting a 512-output model once per alpha.
    """
    train_x = np.asarray(train_x, dtype=np.float64)
    test_x = np.asarray(test_x, dtype=np.float64)
    train_y = np.asarray(train_y, dtype=np.float64)
    scalar = train_y.ndim == 1
    if scalar:
        train_y = train_y[:, None]
    y_mean = train_y.mean(axis=0, keepdims=True)
    centered_y = train_y - y_mean
    rows, columns = train_x.shape
    predictions = []
    if rows <= columns:
        # Dual ridge is substantially cheaper for the small downstream tasks.
        eigenvalues, vectors = np.linalg.eigh(train_x @ train_x.T)
        eigenvalues = np.maximum(eigenvalues, 0.0)
        projected_y = vectors.T @ centered_y
        test_basis = (test_x @ train_x.T) @ vectors
        for alpha in np.asarray(alphas, dtype=np.float64):
            predicted = (
                test_basis
                @ (projected_y / (eigenvalues[:, None] + float(alpha)))
                + y_mean
            )
            predictions.append(predicted[:, 0] if scalar else predicted)
    else:
        # Primal ridge bounds the decomposition by the representation width
        # for the large EGC task and concatenated 1024-dimensional probe.
        eigenvalues, vectors = np.linalg.eigh(train_x.T @ train_x)
        eigenvalues = np.maximum(eigenvalues, 0.0)
        projected_y = vectors.T @ train_x.T @ centered_y
        test_basis = test_x @ vectors
        for alpha in np.asarray(alphas, dtype=np.float64):
            predicted = (
                test_basis
                @ (projected_y / (eigenvalues[:, None] + float(alpha)))
                + y_mean
            )
            predictions.append(predicted[:, 0] if scalar else predicted)
    return predictions


def choose_alpha(features, targets, *, seed, multioutput=False, alphas=ALPHAS):
    features = np.asarray(features, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    splits = _splits(len(features), seed)
    alphas = np.asarray(alphas, dtype=np.float64)
    scores_by_alpha = [[] for _ in alphas]
    for train, heldout in splits:
        scaler = StandardScaler().fit(features[train])
        predictions = _ridge_path_predictions(
            scaler.transform(features[train]), targets[train],
            scaler.transform(features[heldout]), alphas,
        )
        for index, prediction in enumerate(predictions):
            score = _score(
                targets[heldout], prediction, multioutput=multioutput
            )
            if np.isfinite(score):
                scores_by_alpha[index].append(score)
    best_alpha, best_score = None, -float("inf")
    for alpha, scores in zip(alphas, scores_by_alpha):
        mean_score = float(np.mean(scores)) if scores else -float("inf")
        if mean_score > best_score:
            best_alpha, best_score = float(alpha), mean_score
    if best_alpha is None:
        raise RuntimeError("ridge alpha selection produced no finite score")
    return best_alpha


def fit_selected_ridge(features, targets, *, seed, multioutput=False):
    alpha = choose_alpha(
        features, targets, seed=seed, multioutput=multioutput
    )
    scaler = StandardScaler().fit(features)
    model = Ridge(alpha=alpha).fit(scaler.transform(features), targets)
    return StandardizedRidge(scaler=scaler, model=model, alpha=alpha)


def scalar_probe(train_x, train_y, test_x, test_y, *, seed):
    fitted = fit_selected_ridge(train_x, train_y, seed=seed)
    prediction = fitted.predict(test_x)
    return {
        "r2": float(r2_score(test_y, prediction)),
        "alpha": fitted.alpha,
        "prediction": np.asarray(prediction, dtype=np.float64),
    }


def residual_probe(train_o8, train_glt, train_y, test_o8, test_glt, test_y, *, seed):
    """Cross-fit O8 residuals before fitting the GLT correction."""
    train_o8 = np.asarray(train_o8, dtype=np.float64)
    train_glt = np.asarray(train_glt, dtype=np.float64)
    train_y = np.asarray(train_y, dtype=np.float64)
    oof_prediction = np.empty_like(train_y, dtype=np.float64)
    for inner_fold, (inner_train, inner_heldout) in enumerate(
        _splits(len(train_y), seed)
    ):
        # Alpha selection is nested inside inner_train; the current
        # inner-heldout rows never influence their own O8 prediction.
        fitted = fit_selected_ridge(
            train_o8[inner_train], train_y[inner_train],
            seed=int(seed) + 100 + inner_fold,
        )
        oof_prediction[inner_heldout] = fitted.predict(train_o8[inner_heldout])
    residual = train_y - oof_prediction
    residual_model = fit_selected_ridge(
        train_glt, residual, seed=int(seed) + 1000
    )
    o8_model = fit_selected_ridge(train_o8, train_y, seed=int(seed) + 2000)
    prediction = o8_model.predict(test_o8) + residual_model.predict(test_glt)
    return {
        "r2": float(r2_score(test_y, prediction)),
        "o8_alpha": o8_model.alpha,
        "residual_alpha": residual_model.alpha,
        "prediction": np.asarray(prediction, dtype=np.float64),
        "oof_residual_rms": float(np.sqrt(np.mean(residual * residual))),
    }


def linear_cka(first, second):
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    first = first - first.mean(axis=0, keepdims=True)
    second = second - second.mean(axis=0, keepdims=True)
    cross = first.T @ second
    numerator = float(np.sum(cross * cross))
    first_norm = float(np.linalg.norm(first.T @ first, ord="fro"))
    second_norm = float(np.linalg.norm(second.T @ second, ord="fro"))
    return numerator / max(first_norm * second_norm, 1e-12)


def cross_predictability(train_x, train_y, test_x, test_y, *, seed):
    """Standardize both representations from outer-train statistics."""
    x_scaler = StandardScaler().fit(train_x)
    y_scaler = StandardScaler().fit(train_y)
    scaled_train_x = x_scaler.transform(train_x)
    scaled_train_y = y_scaler.transform(train_y)
    scaled_test_x = x_scaler.transform(test_x)
    scaled_test_y = y_scaler.transform(test_y)
    alpha = choose_alpha(
        scaled_train_x, scaled_train_y, seed=seed, multioutput=True
    )
    model = Ridge(alpha=alpha).fit(scaled_train_x, scaled_train_y)
    prediction = model.predict(scaled_test_x)
    return {
        "explained_variance": float(explained_variance_score(
            scaled_test_y, prediction, multioutput="uniform_average"
        )),
        "alpha": float(alpha),
    }


def distribution(values):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if not len(values):
        return {
            "count": 0, "mean": None, "median": None, "std": None,
            "p10": None, "p90": None,
        }
    return {
        "count": int(len(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values, ddof=0)),
        "p10": float(np.quantile(values, 0.10)),
        "p90": float(np.quantile(values, 0.90)),
    }


def contrastive_cosines(o8_projection, glt_projection, shifts=UNMATCHED_SHIFTS):
    o8_projection = np.asarray(o8_projection, dtype=np.float64)
    glt_projection = np.asarray(glt_projection, dtype=np.float64)
    o8_projection = o8_projection / np.maximum(
        np.linalg.norm(o8_projection, axis=1, keepdims=True), 1e-12
    )
    glt_projection = glt_projection / np.maximum(
        np.linalg.norm(glt_projection, axis=1, keepdims=True), 1e-12
    )
    matched = np.sum(o8_projection * glt_projection, axis=1)
    size = len(matched)
    legal = sorted({int(value) % size for value in shifts if int(value) % size})
    unmatched = np.concatenate([
        np.sum(o8_projection * np.roll(glt_projection, shift, axis=0), axis=1)
        for shift in legal
    ]) if legal else np.empty(0, dtype=np.float64)
    matched_summary = distribution(matched)
    unmatched_summary = distribution(unmatched)
    separation = None
    if matched_summary["mean"] is not None and unmatched_summary["mean"] is not None:
        separation = matched_summary["mean"] - unmatched_summary["mean"]
    return {
        "legal_shifts": legal,
        "matched": matched_summary,
        "unmatched": unmatched_summary,
        "mean_separation": separation,
    }


def choose_direction(probe_summary):
    deltas = probe_summary["deltas_vs_p1"]

    def global_increment(name):
        row = deltas[name]
        return (
            row["macro_delta"] > 0
            and row["median_task_delta"] > 0
            and row["positive_tasks"] >= 5
        )

    p3_global, p4_global = global_increment("p3"), global_increment("p4")
    p3_bad = deltas["p3"]["macro_delta"] < -0.002 and deltas["p3"]["negative_tasks"] >= 5
    p4_bad = deltas["p4"]["macro_delta"] < -0.002 and deltas["p4"]["negative_tasks"] >= 5
    if (p3_global and not p4_bad) or (p4_global and not p3_bad):
        return "direction_1_fusion_problem"

    task_rows = probe_summary["tasks"]
    stable_tasks = 0
    for task in task_rows.values():
        for name in ("p3", "p4"):
            if task[name]["delta_vs_p1"] > 0 and task[name]["positive_folds"] >= 4:
                stable_tasks += 1
                break
    if stable_tasks >= 2:
        return "direction_4_task_dependent"
    if deltas["p2"]["macro_delta"] >= -0.005:
        return "direction_2_redundant_glt"
    return "direction_3_geometry_insufficient"
