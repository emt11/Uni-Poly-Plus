"""Direct contract tests for the matched G1/G2 formal cycle."""

import json
from pathlib import Path

from scripts.audit_mts_g1_g2_formal import (
    PAIR_ID,
    audit_checkpoint,
    audit_configs,
    audit_step0,
    resolve,
)

ROOT = Path(__file__).resolve().parents[1]
G1 = ROOT / "configs/mts/experiments/G1_t1_msta_angle_matched_formal_v1.json"
G2 = ROOT / "configs/mts/experiments/G2_t1_msta_cosine_distance_matched_formal_v1.json"
G1_STEP0 = ROOT / "pretrained_models/mts_multiscale_topology/g_family_step0_full_v2/g1_step0.pth"
G2_STEP0 = ROOT / "pretrained_models/mts_multiscale_topology/g_family_step0_full_v2/g2_step0.pth"
G1_CHECKPOINT = ROOT / "pretrained_models/mts_multiscale_topology/g_family_matched_v1/G1/mts_g1_pretrain_20k.pth"


def test_formal_configs_bind_only_declared_g1_g2_difference():
    report = audit_configs(G1, G2)
    assert report["status"] == "passed"
    assert report["protocol"]["shared_step0_id"] == PAIR_ID
    assert report["G1"]["g_family_arm"] == "g1"
    assert report["G2"]["g_family_arm"] == "g2"
    assert report["G1"]["pretraining_objective"] == "masked_atom_only"
    assert report["G2"]["pretraining_objective"] == "masked_atom_only"
    assert report["G1"]["angle_loss_weight"] == 0.0
    assert report["G2"]["angle_loss_weight"] == 0.0
    assert report["G1"]["geometry_model_config_hash"] != report["G2"]["geometry_model_config_hash"]
    assert report["G1"]["g_family_bundle_hash"] != report["G2"]["g_family_bundle_hash"]
    assert (
        report["G1"]["relation_geometry_bundle"]["PI1M_v2"]["artifact_hash"]
        == report["G2"]["relation_geometry_bundle"]["PI1M_v2"]["artifact_hash"]
    )
    assert (
        report["G1"]["relation_geometry_bundle"]["downstream_union"]["artifact_hash"]
        == report["G2"]["relation_geometry_bundle"]["downstream_union"]["artifact_hash"]
    )
    assert (
        report["G1"]["relation_geometry_bundle"]["PI1M_v2"]["root"]
        == report["G2"]["relation_geometry_bundle"]["PI1M_v2"]["root"]
    )
    assert (
        report["G1"]["relation_geometry_bundle"]["downstream_union"]["root"]
        == report["G2"]["relation_geometry_bundle"]["downstream_union"]["root"]
    )


def test_formal_configs_have_no_unresolved_placeholders():
    for path in (G1, G2):
        raw = json.loads(path.read_text(encoding="utf-8"))
        assert "<" not in json.dumps(raw, sort_keys=True)
        assert ">" not in json.dumps(raw, sort_keys=True)


def test_g1_g2_step0_common_state_matches():
    report = audit_step0(G1_STEP0, G2_STEP0)
    assert report["status"] == "passed"
    assert report["shared_step0_id"] == PAIR_ID
    assert report["common_state_hash"] == (
        "28f61aa90d547d2df8cf36cbe06dc39f28ca3ab4cef06393e10c5f6e08bb620d"
    )
    assert report["common_tensors_bit_identical"] is True
    assert report["common_state_tensor_count"] == 134


def test_g1_checkpoint_audit_passes_against_resolved_config():
    resolved = resolve(G1.resolve())
    report = audit_checkpoint(G1_CHECKPOINT, "g1", resolved)
    assert report["status"] == "passed"
    assert report["arm"] == "g1"
