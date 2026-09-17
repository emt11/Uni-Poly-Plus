"""Small synthetic integrity tests for the 8-task DND aggregator."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.aggregate_o8_dnd_8task import _shard


def _manifest():
    return {"folds": [{"fold": 0, "train_indices": [0], "validation_indices": [1],
                        "test_indices": [2, 3]}]}


def _write_shard(root, prediction_rows, metrics, task="egc", fold=0):
    folder = Path(root) / f"{task}_fold{fold}" / task / f"fold{fold}"
    folder.mkdir(parents=True)
    (folder / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    pd.DataFrame(prediction_rows).to_csv(folder / "predictions.csv", index=False)
    return folder


def _valid(root):
    targets = np.asarray([0.0, 1.0, 2.0, 4.0])
    predictions = np.asarray([2.2, 3.8])
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    metrics = {
        "task": "egc", "fold": 0, "protocol": "outer5_inner20", "outer_test": "RUN_ONCE",
        "test_r2": float(r2_score(targets[[2, 3]], predictions)),
        "test_mae": float(mean_absolute_error(targets[[2, 3]], predictions)),
        "test_rmse": float(np.sqrt(mean_squared_error(targets[[2, 3]], predictions))),
    }
    _write_shard(root, {"row_index": [2, 3], "target": targets[[2, 3]], "prediction": predictions}, metrics)
    return targets, metrics


def test_shard_accepts_exact_test_rows_and_recomputed_metrics(tmp_path):
    targets, _ = _valid(tmp_path)
    row, prediction = _shard(tmp_path, "egc", 0, _manifest(), targets)
    assert row["task"] == "egc" and row["fold"] == 0
    assert prediction["row_index"].tolist() == [2, 3]


def test_aggregator_rejects_wrong_test_indices(tmp_path):
    targets, metrics = _valid(tmp_path)
    folder = tmp_path / "egc_fold0" / "egc" / "fold0"
    pd.DataFrame({"row_index": [0, 3], "target": targets[[0, 3]], "prediction": [0.1, 3.8]}).to_csv(
        folder / "predictions.csv", index=False
    )
    with pytest.raises(ValueError, match="outer-test indices mismatch"):
        _shard(tmp_path, "egc", 0, _manifest(), targets)


def test_aggregator_rejects_raw_label_mismatch(tmp_path):
    targets, metrics = _valid(tmp_path)
    folder = tmp_path / "egc_fold0" / "egc" / "fold0"
    pd.DataFrame({"row_index": [2, 3], "target": [999.0, targets[3]], "prediction": [2.2, 3.8]}).to_csv(
        folder / "predictions.csv", index=False
    )
    with pytest.raises(ValueError, match="raw-label mismatch"):
        _shard(tmp_path, "egc", 0, _manifest(), targets)
