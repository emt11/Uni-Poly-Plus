"""Verified aggregation of GLT dual five-fold shards.

The aggregate may only be written when each fold reproduces from its own saved
predictions, matches the fixed split manifest, and (for legacy artifacts) can
be attributed to a formal-shard run.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.aggregate_glt_dual_finetune import aggregate  # noqa: E402
from scripts.create_mips_split_manifests import build_manifest  # noqa: E402

TASK = "eat"
SAMPLES = 20


def _fixture(tmp_path, *, protocol="outer5_inner20", write_run_json=True):
    raw_root, split_root = tmp_path / "raw", tmp_path / "splits"
    raw_root.mkdir(parents=True, exist_ok=True), split_root.mkdir(parents=True, exist_ok=True)
    labels = np.round(np.linspace(-5.0, 4.0, SAMPLES), 2)
    frame = pd.DataFrame({"smiles": [f"*C{'C' * (i % 4)}*" for i in range(SAMPLES)],
                          "Eat": labels})
    csv_path = raw_root / f"smi_{TASK}.csv"
    frame.to_csv(csv_path, index=False)
    manifest = build_manifest(TASK, csv_path, "outer5_inner20")
    (split_root / f"{TASK}.json").write_text(json.dumps(manifest), encoding="utf-8")

    shard_root = tmp_path / "shards"
    predictions_by_fold = {}
    for fold in range(5):
        folder = shard_root / f"{TASK}_fold{fold}" / TASK / f"fold{fold}"
        folder.mkdir(parents=True)
        test = list(map(int, manifest["folds"][fold]["test_indices"]))
        target = labels[test]
        prediction = target + 0.25 * np.cos(np.arange(len(test)))
        predictions_by_fold[fold] = (test, target, prediction)
        pd.DataFrame({"row_index": test, "target": target,
                      "prediction": prediction}).to_csv(folder / "predictions.csv", index=False)
        metrics = {
            "test_r2": float(r2_score(target, prediction)),
            "test_mae": float(mean_absolute_error(target, prediction)),
            "test_rmse": float(np.sqrt(mean_squared_error(target, prediction))),
            "task": TASK, "fold": fold, "best_validation_r2": 0.5, "best_epoch": 3,
        }
        if protocol is not None:
            metrics["protocol"] = protocol
        (folder / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
        if write_run_json:
            (shard_root / f"{TASK}_fold{fold}" / "run.json").write_text(json.dumps({
                "protocol": "outer5_inner20_formal_shard", "smoke": False,
                "formal_shard": True, "selected_tasks": [TASK], "selected_folds": [fold],
                "outer_test": "RUN",
                "command": ["scripts/finetune_glt_dual.py", "--task", TASK,
                            "--fold", str(fold), "--formal-shard"],
            }), encoding="utf-8")
    return shard_root, split_root, raw_root, predictions_by_fold


def _run(tmp_path, **kwargs):
    allow_legacy = bool(kwargs.pop("allow_legacy", False))
    shard_root, split_root, raw_root, folds = _fixture(tmp_path, **kwargs)
    output = tmp_path / "comparison_review_test"
    result = aggregate(shard_root, output, tasks=[TASK], split_root=split_root,
                       raw_root=raw_root,
                       allow_legacy_missing_protocol=allow_legacy)
    return result, output, folds


def test_correct_protocol_aggregates_and_recomputes(tmp_path):
    result, output, folds = _run(tmp_path)
    assert result["status"] == "PASS"
    assert result["legacy_compatibility_used"] is False
    task = result["tasks"][TASK]
    assert task["fold_count"] == 5
    recomputed = [task["folds"][f]["recomputed_test_r2"] for f in range(5)]
    stored = [task["folds"][f]["test_r2"] for f in range(5)]
    assert np.allclose(recomputed, stored, rtol=1e-6, atol=1e-8)
    # aggregated OOF covers every row exactly once, and both R2 conventions exist
    oof = pd.read_csv(output / TASK / "oof.csv")
    assert oof["row_index"].tolist() == list(range(SAMPLES))
    assert "pooled_oof" in task and "mean" in task["test_r2"]
    expected_pooled = r2_score(oof["target"], oof["prediction"])
    assert np.isclose(task["pooled_oof"]["r2"], expected_pooled, rtol=1e-9, atol=1e-12)


def test_missing_protocol_requires_the_compatibility_switch(tmp_path):
    with pytest.raises(ValueError, match="lack protocol"):
        _run(tmp_path, protocol=None)


def test_missing_protocol_accepted_with_switch_and_records_compatibility(tmp_path):
    result, _, _ = _run(tmp_path, protocol=None, allow_legacy=True)
    assert result["legacy_compatibility_used"] is True
    assert result["legacy_missing_protocol_accepted"] == [
        f"{TASK}/fold{fold}" for fold in range(5)]


def test_missing_protocol_without_formal_shard_run_record_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        _run(tmp_path, protocol=None, allow_legacy=True, write_run_json=False)


def test_wrong_protocol_is_rejected_even_with_switch(tmp_path):
    with pytest.raises(ValueError, match="protocol is not"):
        _run(tmp_path, protocol="outer5_inner20_smoke", allow_legacy=True)


def test_missing_sample_is_rejected(tmp_path):
    shard_root, split_root, raw_root, _ = _fixture(tmp_path)
    folder = shard_root / f"{TASK}_fold2" / TASK / "fold2"
    frame = pd.read_csv(folder / "predictions.csv").iloc[:-1]
    frame.to_csv(folder / "predictions.csv", index=False)
    with pytest.raises(ValueError, match="manifest test set"):
        aggregate(shard_root, tmp_path / "out", tasks=[TASK], split_root=split_root,
                  raw_root=raw_root)


def test_duplicate_row_index_is_rejected(tmp_path):
    shard_root, split_root, raw_root, _ = _fixture(tmp_path)
    folder = shard_root / f"{TASK}_fold1" / TASK / "fold1"
    frame = pd.read_csv(folder / "predictions.csv")
    frame.loc[frame.index[-1], "row_index"] = int(frame["row_index"].iloc[0])
    frame.to_csv(folder / "predictions.csv", index=False)
    with pytest.raises(ValueError, match="unique integers"):
        aggregate(shard_root, tmp_path / "out", tasks=[TASK], split_root=split_root,
                  raw_root=raw_root)


def test_wrong_fold_identity_is_rejected(tmp_path):
    shard_root, split_root, raw_root, folds = _fixture(tmp_path)
    folder = shard_root / f"{TASK}_fold3" / TASK / "fold3"
    other = shard_root / f"{TASK}_fold4" / TASK / "fold4/predictions.csv"
    pd.read_csv(other).to_csv(folder / "predictions.csv", index=False)
    with pytest.raises(ValueError, match="manifest test set"):
        aggregate(shard_root, tmp_path / "out", tasks=[TASK], split_root=split_root,
                  raw_root=raw_root)


def test_tampered_metric_and_tampered_label_are_rejected(tmp_path):
    shard_root, split_root, raw_root, _ = _fixture(tmp_path)
    folder = shard_root / f"{TASK}_fold0" / TASK / "fold0"
    metrics = json.loads((folder / "metrics.json").read_text())
    metrics["test_r2"] += 0.05
    (folder / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    with pytest.raises(ValueError, match="does not reproduce"):
        aggregate(shard_root, tmp_path / "out", tasks=[TASK], split_root=split_root,
                  raw_root=raw_root)

    nested = tmp_path / "labels"; nested.mkdir(exist_ok=True)
    shard_root, split_root, raw_root, _ = _fixture(nested)
    folder = shard_root / f"{TASK}_fold0" / TASK / "fold0"
    frame = pd.read_csv(folder / "predictions.csv")
    frame.loc[frame.index[0], "target"] = float(frame["target"].iloc[0]) + 1.0
    frame.to_csv(folder / "predictions.csv", index=False)
    with pytest.raises(ValueError, match="differ from raw labels"):
        aggregate(shard_root, tmp_path / "out2", tasks=[TASK], split_root=split_root,
                  raw_root=raw_root)
