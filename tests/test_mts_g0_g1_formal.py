"""Direct contract tests for the matched G0/G1 formal cycle."""

import json
from pathlib import Path

from scripts.audit_mts_g0_g1_formal import PAIR_ID, audit_configs


ROOT = Path(__file__).resolve().parents[1]
G0 = ROOT / "configs/mts/experiments/G0_t1_msta_matched_formal_v1.json"
G1 = ROOT / "configs/mts/experiments/G1_t1_msta_angle_matched_formal_v1.json"


def test_formal_configs_bind_only_declared_g0_g1_difference():
    report = audit_configs(G0, G1)
    assert report["status"] == "passed"
    assert report["protocol"]["shared_step0_id"] == PAIR_ID
    assert report["G0"]["g_family_arm"] == "g0"
    assert report["G1"]["g_family_arm"] == "g1"
    assert report["G0"]["pretraining_objective"] == "masked_atom_only"
    assert report["G1"]["pretraining_objective"] == "masked_atom_only"
    assert report["G0"]["angle_loss_weight"] == 0.0
    assert report["G1"]["angle_loss_weight"] == 0.0


def test_formal_configs_have_no_unresolved_placeholders():
    for path in (G0, G1):
        raw = json.loads(path.read_text(encoding="utf-8"))
        assert "<" not in json.dumps(raw, sort_keys=True)
        assert ">" not in json.dumps(raw, sort_keys=True)
