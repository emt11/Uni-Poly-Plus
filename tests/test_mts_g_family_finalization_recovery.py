"""Direct tests for the G-family milestone finalizer and the plots-decoupled
finalization regression.

These tests are read-only against the frozen G0 artifacts (milestone, final
checkpoint, completion marker, shared step-0) and write only into temporary
directories.  They prove:

- the finalization path produces a final checkpoint + completion marker when
  the ``plots/`` tree and its loss JSON/PNG do not exist (regression for the
  post-training plotting crash);
- the recovered state_dict is tensor-identical to the milestone and to the
  normally-written G0 final checkpoint;
- the rebuilt metadata matches the normally-written G0 final meta field by
  field, except for the explicitly documented unrecoverable runtime fields;
- the rebuilt target contract binds the current frozen cache identity exactly
  like the historical G0 final;
- the finalizer refuses SHA mismatches, wrong schemas and existing outputs.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

from scripts.finalize_mts_g_pretrain_milestone import (
    _load_and_validate_milestone,
    _resolve_config,
    _runtime_code_identity,
    main as finalize_main,
)

ROOT = Path(__file__).resolve().parents[1]
G0_DIR = ROOT / "pretrained_models/mts_multiscale_topology/g_family_matched_v1/G0"
G0_MILESTONE = G0_DIR / "mts_g0_pretrain_20k.step_20000.pth"
G0_FINAL = G0_DIR / "mts_g0_pretrain_20k.pth"
G0_CONFIG = ROOT / "configs/mts/experiments/G0_t1_msta_matched_formal_v1.json"
G0_STEP0 = ROOT / "pretrained_models/mts_multiscale_topology/g_family_step0_full_v2/g0_step0.pth"
G0_MILESTONE_SHA = "95e8c3c5535780f23660ed31e59dd3e656f6349ed1ab7d43b3a05a1f21050881"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finalize_g0(tmp_path: Path) -> Path:
    output = tmp_path / "mts_g0_pretrain_20k.pth"
    audit = tmp_path / "finalization_audit.json"
    finalize_main(
        [
            "--milestone", str(G0_MILESTONE),
            "--milestone-sha256", G0_MILESTONE_SHA,
            "--config", str(G0_CONFIG),
            "--arm", "g0",
            "--step0", str(G0_STEP0),
            "--output", str(output),
            "--audit-output", str(audit),
        ]
    )
    return output


def test_finalization_requires_no_plots_directory(tmp_path):
    assert not (ROOT / "plots").exists()
    output = _finalize_g0(tmp_path)
    assert output.is_file()
    assert Path(str(output) + ".complete.json").is_file()
    assert not (ROOT / "plots").exists()


def test_recovered_g0_state_dict_matches_milestone_and_final(tmp_path):
    output = _finalize_g0(tmp_path)
    milestone = torch.load(G0_MILESTONE, map_location="cpu", weights_only=False)
    recovered = torch.load(output, map_location="cpu", weights_only=False)
    reference = torch.load(G0_FINAL, map_location="cpu", weights_only=False)
    milestone_state = milestone["state_dict"]
    recovered_state = recovered["state_dict"]
    reference_state = reference["state_dict"]
    assert set(recovered_state) == set(milestone_state) == set(reference_state)
    assert len(recovered_state) == 134
    for key in recovered_state:
        value = recovered_state[key]
        assert tuple(value.shape) == tuple(milestone_state[key].shape)
        assert value.dtype == milestone_state[key].dtype
        assert torch.equal(value, milestone_state[key]), f"tensor differs from milestone: {key}"
        assert torch.equal(value, reference_state[key]), f"tensor differs from G0 final: {key}"


def test_recovered_g0_meta_matches_final_except_unrecoverable_fields(tmp_path):
    output = _finalize_g0(tmp_path)
    recovered = torch.load(output, map_location="cpu", weights_only=False)["meta"]
    reference = torch.load(G0_FINAL, map_location="cpu", weights_only=False)["meta"]
    allowed_differences = {
        "training_wall_seconds",
        "dirty_diff_sha",
        "recovered_from_milestone",
        "recovery_reason",
        "recovery_source_path",
        "recovery_source_sha256",
        "recovery_source_optimizer_step",
        "runtime_pretrain_code_identity",
        "finalization_code_identity",
        "original_failure_log_path",
        "original_failure_log_excerpt",
    }
    for key in sorted(set(reference) | set(recovered)):
        if key in allowed_differences:
            continue
        if key not in recovered:
            raise AssertionError(f"recovered meta is missing field: {key}")
        if recovered[key] != reference[key]:
            raise AssertionError(f"meta field mismatch: {key}: {recovered[key]!r} != {reference[key]!r}")
    assert recovered["training_wall_seconds"] is None
    assert recovered["dirty_diff_sha"] is None
    assert recovered["recovered_from_milestone"] is True
    assert recovered["optimizer_steps"] == 20000


def test_recovered_target_contract_binds_frozen_cache_like_g0_final(tmp_path):
    output = _finalize_g0(tmp_path)
    recovered = torch.load(output, map_location="cpu", weights_only=False)["meta"]
    reference = torch.load(G0_FINAL, map_location="cpu", weights_only=False)["meta"]
    assert recovered["target_contract"] == reference["target_contract"]
    assert recovered["source_contract_sha256"] == reference["source_contract_sha256"]
    assert recovered["final_cache_binding"] == reference["final_cache_binding"]
    assert recovered["cache_bundle_hash"] == reference["cache_bundle_hash"]


def test_runtime_code_identity_matches_launch_contract():
    milestone = torch.load(G0_MILESTONE, map_location="cpu", weights_only=False)
    rc = milestone["resume_contract"]
    identity = _runtime_code_identity(rc["pretrain_code_files"])
    assert identity["sha256"] == rc["pretrain_code_hash"]
    reference = torch.load(G0_FINAL, map_location="cpu", weights_only=False)["meta"]
    assert reference["pretrain_code_identity"] == identity


def test_finalizer_refuses_sha_mismatch(tmp_path):
    with pytest.raises(RuntimeError, match="SHA256 mismatch"):
        _load_and_validate_milestone(G0_MILESTONE, "0" * 64)


def test_finalizer_refuses_existing_output(tmp_path):
    output = _finalize_g0(tmp_path)
    audit = tmp_path / "second_audit.json"
    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        finalize_main(
            [
                "--milestone", str(G0_MILESTONE),
                "--milestone-sha256", G0_MILESTONE_SHA,
                "--config", str(G0_CONFIG),
                "--arm", "g0",
                "--step0", str(G0_STEP0),
                "--output", str(output),
                "--audit-output", str(audit),
            ]
        )


def test_finalizer_rejects_non_step20000_milestone(tmp_path):
    wrong = torch.load(G0_MILESTONE, map_location="cpu", weights_only=False)
    wrong = dict(wrong)
    wrong["optimizer_step"] = 18000
    forged = tmp_path / "forged_step18000.pth"
    torch.save(wrong, forged)
    with pytest.raises(RuntimeError, match="optimizer step"):
        _load_and_validate_milestone(forged, _sha256(forged))


def test_resolved_config_arm_matches(tmp_path):
    resolved = _resolve_config(G0_CONFIG)
    assert resolved["g_family_arm"] == "g0"
    assert resolved["topology_attention_variant"] == "msta_last2"
