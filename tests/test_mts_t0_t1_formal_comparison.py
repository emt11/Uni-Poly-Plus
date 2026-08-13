"""Direct contract tests for the formal T0/T1 paired comparison tools."""

from pathlib import Path

import pytest

from scripts.audit_mts_t0_t1_formal import (
    build_identity_audit,
    resolve_config,
)
from scripts.compare_mts_t0_t1_formal import (
    TASKS,
    FOLDS,
    compare,
    expected_finetune_hash,
    expected_profile_hash,
    expected_training_hash,
)


ROOT = Path(__file__).resolve().parents[1]
T0_CONFIG = ROOT / "configs/mts/experiments/T0_o8_matched_t1_formal_v1.json"
T1_CONFIG = ROOT / "configs/mts/experiments/T1_msta_formal_v1.json"
SOURCE = ROOT / "pretrained_models/mts/mts_joint_pretraining_pi1m_v2_seed42_canonical_angle20_v1.pth"


def test_formal_configs_only_change_declared_topology_identity():
    t0_raw = T0_CONFIG.read_text(encoding="utf-8")
    t1_raw = T1_CONFIG.read_text(encoding="utf-8")
    assert '"experiment_id": "T0_o8_matched_t1_formal_v1"' in t0_raw
    assert '"experiment_id": "T1_msta_formal_v1"' in t1_raw
    t0 = resolve_config(T0_CONFIG)
    t1 = resolve_config(T1_CONFIG)
    assert t0["topology_attention_variant"] == "o8"
    assert t1["topology_attention_variant"] == "msta_last2"
    for key in (
        "feature_config_hash",
        "geometry_model_config_hash",
        "source_geometry_model_config_hash",
        "topology_representation",
        "evaluation_protocol",
        "finetune_profile",
    ):
        assert t0[key] == t1[key]
    assert t0["graph_model_config_hash"] != t1["graph_model_config_hash"]


def test_identity_audit_passes_before_init_artifact_exists():
    if not SOURCE.is_file():
        pytest.skip("fixed T0 checkpoint is unavailable")
    audit = build_identity_audit(
        t0_config=T0_CONFIG,
        t1_config=T1_CONFIG,
        source_checkpoint=SOURCE,
        init_checkpoint=None,
    )
    assert audit["passed"]
    assert audit["source_checkpoint"]["sha256_matches_contract"]
    assert audit["raw_configs_match_except_declared"]


def test_protocol_hashes_are_deterministic_and_seed_sensitive():
    t0 = resolve_config(T0_CONFIG)
    finetune = expected_finetune_hash(t0)
    assert finetune == expected_finetune_hash(t0)
    assert expected_profile_hash() == expected_profile_hash()
    assert expected_training_hash(finetune, seed=42) != expected_training_hash(finetune, seed=43)


def test_compare_reports_paired_fold_and_macro_statistics():
    def arm(offset):
        records = {}
        for task_index, task in enumerate(TASKS):
            for fold in FOLDS:
                records[(task, fold)] = {
                    "metric": {
                        "r2": 0.1 * task_index + 0.01 * fold + offset,
                        "mae": 1.0 + 0.01 * fold + offset,
                        "rmse": 2.0 + 0.01 * fold + offset,
                        "wall_seconds": 3.0,
                    }
                }
        return {"records": records}

    result = compare(arm(0.0), arm(0.1))
    assert len(result["fold_rows"]) == 40
    assert len(result["task_rows"]) == 8
    assert result["macro"]["delta_r2_mean"] == pytest.approx(0.1)
    assert result["task_rows"][0]["paired_delta"]["r2_mean"] == pytest.approx(0.1)
    assert result["task_rows"][0]["delta_t"] == pytest.approx(0.1)
    assert result["task_delta_statistics"]["count"] == 8
    assert result["task_delta_statistics"]["mean"] == pytest.approx(0.1)
    assert result["task_delta_statistics"]["positive_count"] == 8
    assert len(result["leave_one_task_out"]) == 8
    assert result["delta_macro_without_xc"] == pytest.approx(0.1)
