#!/usr/bin/env python3
"""Strict G-family pretraining milestone finalizer.

Recovers the standard ``mts-model-v4`` final checkpoint and its
``.complete.json`` completion marker from an immutable optimizer-free
step-20000 milestone.  This is a read-only conversion: the milestone is never
modified, model weights are copied tensor-by-tensor without any forward or
backward pass, and the final metadata is rebuilt from the milestone launch
contract, the resolved experiment config, the frozen cache binding, the shared
step-0 artifact and the immutable pretraining profile.

The tool defaults to refusing to overwrite any existing output.  It is only
used for the formal G-family arms after a post-training finalization failure;
it must never be used to re-run or extend training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
PYTHON = os.environ.get("PYTHON", "/opt/conda/envs/MTS/bin/python")

from scripts.pretrain import (  # noqa: E402
    CACHE_BOND_ANGLE_SCHEMA,
    MIPS_CANONICAL_LGA_SCHEMA_VERSION,
    MIPS_EXPLICIT_FEATURE_SCHEMA,
    MIPS_EXPLICIT_LGA_SCHEMA_VERSION,
    MIPS_EXPLICIT_TOPOLOGY_SCHEMA,
    MIPS_TRIMER_ACCEPTANCE,
    MIPS_TRIMER_BUILDER_VERSION,
    MIPS_TRIMER_CACHE_BUNDLE_SCHEMA,
    MIPS_TRIMER_CACHE_LAYOUT_SCHEMA,
    MIPS_TRIMER_CONTENT_SCHEMA,
    MIPS_TRIMER_FEATURE_SCHEMA,
    MIPS_TRIMER_LMDB_SCHEMA,
    MIPS_TRIMER_MMFF_RELAX_STEPS,
    MIPS_TRIMER_PROTOCOL,
    MIPS_TRIMER_REQUIRE_MMFF_CONVERGENCE,
    MIPS_TRIMER_SELECTION,
    MIPS_TRIMER_TOPOLOGY_SCHEMA,
    MTS_ROUTE_NAME,
    MTS_ROUTE_SHORT_NAME,
    MTS_STAGE1_ID,
    PRETRAIN_CHECKPOINT_SCHEMA,
    PRETRAIN_PROFILE_ID,
    TOPOLOGY_EXPLICIT,
    _atomic_torch_save,
    _build_final_cache_binding,
    _canonical_json_hash,
    _file_sha256,
    _load_pretrain_profile,
    _PRETRAIN_CODE_FILES,
    _pretrain_code_identity,
    build_pretrain_target_contract,
    cache_bundle_binding_hash,
)

MILESTONE_SCHEMA = "mts-pretrain-milestone-v1"
COMPLETE_SCHEMA = "mts-pretrain-complete-v1"
SHARED_STEP0_ID = "mts_g_family_step0_v2_seed42"
FINAL_OPTIMIZER_STEPS = 20000
REFERENCE_COMMITS = {
    "mips": "26aafe52926a3f33bf2d3d382ae263360319812d",
    "scage": "82bcbb4647e31bf0d413a317e69a2526df75ce01",
}


def _sha256(path: Path) -> str:
    return _file_sha256(path)


def _resolve_config(config_path: Path) -> dict[str, Any]:
    text = subprocess.check_output(
        [PYTHON, "scripts/resolve_mips_trimer_scage.py", str(config_path)],
        cwd=ROOT,
        text=True,
    )
    return json.loads(text)


def _runtime_code_identity(files: dict[str, str]) -> dict[str, Any]:
    """Reconstruct the launch-time code identity from the resume contract.

    ``resume_contract.pretrain_code_files`` is the per-file digest map recorded
    at launch.  The aggregate digest is rebuilt with the same insertion-order
    scheme as ``_pretrain_code_identity`` so the stored ``pretrain_code_hash``
    can be verified against the file map itself.
    """
    missing = [
        relative for relative in _PRETRAIN_CODE_FILES if relative not in files
    ]
    if missing:
        raise RuntimeError(
            "milestone launch contract is missing code files: " + ", ".join(missing[:8])
        )
    digest = hashlib.sha256()
    for relative in files:
        value = files[relative]
        if not isinstance(value, str) or len(value) != 64:
            raise RuntimeError(f"invalid code file digest for {relative}")
        digest.update(relative.encode("utf-8"))
        digest.update(value.encode("ascii"))
    return {"files": dict(files), "sha256": digest.hexdigest()}


def _load_and_validate_milestone(
    milestone_path: Path, expected_sha256: str
) -> dict[str, Any]:
    milestone_path = milestone_path.resolve()
    if not milestone_path.is_file():
        raise RuntimeError(f"milestone is missing: {milestone_path}")
    if _sha256(milestone_path) != expected_sha256:
        raise RuntimeError(
            "milestone SHA256 mismatch: "
            f"expected {expected_sha256}, got {_sha256(milestone_path)}"
        )
    payload = torch.load(milestone_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise RuntimeError("milestone is not a dict payload")
    if payload.get("schema") != MILESTONE_SCHEMA:
        raise RuntimeError(
            f"milestone schema mismatch: {payload.get('schema')!r} != {MILESTONE_SCHEMA!r}"
        )
    if payload.get("checkpoint_schema") != PRETRAIN_CHECKPOINT_SCHEMA:
        raise RuntimeError(
            "milestone checkpoint schema mismatch: "
            f"{payload.get('checkpoint_schema')!r} != {PRETRAIN_CHECKPOINT_SCHEMA!r}"
        )
    if int(payload.get("optimizer_step", -1)) != FINAL_OPTIMIZER_STEPS:
        raise RuntimeError(
            f"milestone optimizer step is not {FINAL_OPTIMIZER_STEPS}: "
            f"{payload.get('optimizer_step')!r}"
        )
    state = payload.get("state_dict")
    if not isinstance(state, dict) or not state:
        raise RuntimeError("milestone has no non-empty state_dict")
    heads = payload.get("heads")
    if not isinstance(heads, dict) or not heads:
        raise RuntimeError("milestone has no non-empty pretraining heads")
    for key, value in list(state.items()) + list(heads.items()):
        if not torch.is_tensor(value):
            raise RuntimeError(f"milestone entry is not a tensor: {key}")
        if (value.is_floating_point() or value.is_complex()) and not torch.isfinite(value).all():
            raise RuntimeError(f"milestone tensor is non-finite: {key}")
    rc = payload.get("resume_contract")
    if not isinstance(rc, dict) or not rc:
        raise RuntimeError("milestone has no launch resume_contract")
    return {
        "path": milestone_path,
        "sha256": expected_sha256,
        "optimizer_step": int(payload["optimizer_step"]),
        "profile_id": str(payload.get("profile_id", "")),
        "profile_sha256": str(payload.get("profile_sha256", "")),
        "state_dict": state,
        "heads": heads,
        "resume_contract": rc,
    }


def _validate_step0(step0_path: Path, rc: dict[str, Any], arm: str) -> dict[str, Any]:
    """Rebuild the initialization identity exactly as the training launch did."""
    step0_path = step0_path.resolve()
    if not step0_path.is_file():
        raise RuntimeError(f"shared step-0 artifact is missing: {step0_path}")
    step0_sha = _sha256(step0_path)
    if rc.get("initialization_state") != step0_sha:
        raise RuntimeError(
            "step-0 SHA256 does not match the milestone launch contract: "
            f"{rc.get('initialization_state')!r} != {step0_sha}"
        )
    payload = torch.load(step0_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or not isinstance(payload.get("state_dict"), dict):
        raise RuntimeError("step-0 artifact must contain state_dict")
    meta = dict(payload.get("meta") or {})
    required = {
        "schema": "mts-pretrain-init-v1",
        "initialization": "fresh_paired",
        "model_identity": "T1",
        "optimizer_steps": 0,
        "paired_init_id": SHARED_STEP0_ID,
    }
    for key, expected in required.items():
        if meta.get(key) != expected:
            raise RuntimeError(
                f"step-0 metadata mismatch for {key}: "
                f"expected={expected!r}, observed={meta.get(key)!r}"
            )
    g_required = {
        "g_family_arm": arm,
        "geometry_mode": arm,
        "pretraining_objective": "masked_atom_only",
        "angle_loss_weight": 0.0,
        "shared_step0_id": SHARED_STEP0_ID,
        "g_family_bundle_hash": str(rc.get("g_family_bundle_hash") or ""),
    }
    for key, expected in g_required.items():
        observed = meta.get(key)
        if key == "angle_loss_weight":
            matches = False
            try:
                matches = float(observed) == float(expected)
            except (TypeError, ValueError):
                matches = False
        else:
            matches = observed == expected
        if not matches:
            raise RuntimeError(
                "step-0 G-family metadata mismatch for "
                f"{key}: expected={expected!r}, observed={observed!r}"
            )
    state = payload["state_dict"]
    for index in (4, 5):
        key = f"encoders.graph.encoder.layers.{index}.attention.local_output.weight"
        value = state.get(key)
        if value is None or not torch.isfinite(value).all() or torch.count_nonzero(value).item() != 0:
            raise RuntimeError(f"step-0 T1 local_output is not finite zero: {key}")
    return {
        "schema": str(meta["schema"]),
        "initialization": str(meta["initialization"]),
        "model_identity": "T1",
        "paired_init_id": str(meta["paired_init_id"]),
        "path": str(step0_path),
        "sha256": step0_sha,
        "target_config_hash": meta.get("target_config_hash"),
        "target_graph_model_config_hash": meta.get("target_graph_model_config_hash"),
        "optimizer_steps": int(meta["optimizer_steps"]),
        "g_family_arm": meta.get("g_family_arm"),
        "shared_step0_id": meta.get("shared_step0_id"),
        "pretraining_objective": meta.get("pretraining_objective"),
        "g_family_bundle_hash": meta.get("g_family_bundle_hash"),
        "_state_dict": state,
    }


def _validate_launch_contract(
    rc: dict[str, Any], resolved: dict[str, Any], arm: str, profile: dict[str, Any]
) -> dict[str, Any]:
    def require(field: str, expected: Any) -> None:
        observed = rc.get(field)
        if observed != expected:
            raise RuntimeError(
                f"launch contract field mismatch: {field}: "
                f"expected={expected!r}, observed={observed!r}"
            )

    require("config_schema", "mts-experiment-v3")
    require("stage", MTS_STAGE1_ID)
    require("topology_representation", "canonical_lifted")
    require("feature_config_hash", resolved["feature_config_hash"])
    require("o8_feature_config_hash", resolved["feature_config_hash"])
    require("graph_model_config_hash", resolved["graph_model_config_hash"])
    require("geometry_model_config_hash", resolved["geometry_model_config_hash"])
    require("source_geometry_model_config_hash", resolved["source_geometry_model_config_hash"])
    require("topology_attention_variant", "msta_last2")
    require("msta_layer_indices", list(resolved["msta_layer_indices"]))
    require("msta_local_spd", list(resolved["msta_local_spd"]))
    require("msta_context_spd", list(resolved["msta_context_spd"]))
    require("msta_share_relation_dropout", True)
    require("msta_local_output_bias", False)
    require("msta_local_output_init", "zero")
    require("g_family_arm", arm)
    require("g_family_bundle_hash", resolved["g_family_bundle_hash"])
    require("pretraining_objective", "masked_atom_only")
    require("angle_loss_weight", 0.0)
    require("shared_step0_id", SHARED_STEP0_ID)
    require("pretraining_dataset", "PI1M_v2")
    require("batch_size", 336)
    require("gradient_accumulation_steps", 1)
    require("lr", 0.0002)
    require("weight_decay", 0.0)
    require("adam_betas", [0.9, 0.98])
    require("adam_eps", 1e-8)
    require("warmup_steps", 2000)
    require("scheduler", "polynomial")
    require("scheduler_power", 1.0)
    require("end_lr", 1e-9)
    require("max_grad_norm", -1.0)
    require("amp_dtype", "bf16")
    require("angle_bins", 20)
    require("angle_focal_gamma", 2.0)
    require("angle_objective", "categorical")
    require("masked_atom_weight", 1.0)
    require("trimer_bond_angle_weight", 0.0)
    require("paired_init_id", SHARED_STEP0_ID)
    require("pretrain_profile", PRETRAIN_PROFILE_ID)
    require("pretrain_profile_hash", _canonical_json_hash(profile))
    require("source_cohort_hash", profile["cohort_hash"])
    if arm == "g1":
        bundle = resolved["relation_geometry_bundle"]["PI1M_v2"]
        require("relation_geometry_artifact_hash", bundle["artifact_hash"])
        require("relation_geometry_sidecar", bundle["root"])
        if rc.get("relation_geometry_artifact_hash") != bundle["artifact_hash"]:
            raise RuntimeError("G1 relation artifact does not bind the resolved bundle")
    else:
        require("relation_geometry_sidecar", None)
        require("relation_geometry_artifact_hash", None)
    require("g3_permutation_sidecar", None)
    require("g3_permutation_artifact_hash", None)
    return {"training_config_hash": rc["training_config_hash"]}


def _build_recovery_meta(
    *,
    rc: dict[str, Any],
    resolved: dict[str, Any],
    profile: dict[str, Any],
    init_info: dict[str, Any],
    arm: str,
    runtime_identity: dict[str, Any],
    finalization_identity: dict[str, Any],
    recovery: dict[str, Any],
) -> dict[str, Any]:
    init_state = {key: value for key, value in init_info.items() if key != "_state_dict"}
    feature_cohort_hash = rc["source_cohort_hash"]
    topology_artifact_hash = rc["topology_cache_artifact_hash"]
    trimer_artifact_hash = rc["trimer_cache_artifact_hash"]
    cache_bundle_hash = rc["cache_bundle_hash"]
    if (
        cache_bundle_binding_hash(
            cohort_hash=feature_cohort_hash,
            topology_artifact_hash=topology_artifact_hash,
            trimer_artifact_hash=trimer_artifact_hash,
        )
        != cache_bundle_hash
    ):
        raise RuntimeError("launch-contract cache bundle hash does not bind its artifacts")
    final_cache_binding = _build_final_cache_binding(profile)
    if final_cache_binding["done_artifact_id"]["topology"] != topology_artifact_hash:
        raise RuntimeError("frozen topology artifact does not bind the launch contract")
    if final_cache_binding["done_artifact_id"]["trimer"] != trimer_artifact_hash:
        raise RuntimeError("frozen Trimer artifact does not bind the launch contract")
    store_json_sha256 = final_cache_binding["store_sha256"]
    topology_frozen_sha = final_cache_binding["frozen_file_sha256"]["topology"]
    trimer_frozen_sha = final_cache_binding["frozen_file_sha256"]["trimer"]

    source_contract = {
        "schema": PRETRAIN_CHECKPOINT_SCHEMA,
        "profile_id": PRETRAIN_PROFILE_ID,
        "stage": MTS_STAGE1_ID,
        "baseline": MTS_ROUTE_NAME,
        "topology_representation": "canonical_lifted",
        "pretraining_dataset": "PI1M_v2",
        "source_cohort_hash": feature_cohort_hash,
        "topology_artifact_hash": topology_artifact_hash,
        "trimer_artifact_hash": trimer_artifact_hash,
        "angle_cache_schema": profile["angle_cache_schema"],
        "angle_cache_artifact_hash": profile["angle_cache_artifact"],
        "g_family_arm": arm,
        "g_family_bundle_hash": rc["g_family_bundle_hash"],
        "relation_geometry_sidecar": rc.get("relation_geometry_sidecar"),
        "relation_geometry_artifact_hash": rc.get("relation_geometry_artifact_hash"),
        "g3_permutation_sidecar": rc.get("g3_permutation_sidecar"),
        "g3_permutation_artifact_hash": rc.get("g3_permutation_artifact_hash"),
        "pretraining_objective": "masked_atom_only",
        "angle_loss_weight": 0.0,
        "shared_step0_id": SHARED_STEP0_ID,
        "cache_store_sha256": store_json_sha256,
        "topology_frozen_sha256": topology_frozen_sha,
        "trimer_frozen_sha256": trimer_frozen_sha,
        "feature_config_hash": rc["feature_config_hash"],
        "graph_model_config_hash": rc["graph_model_config_hash"],
        "geometry_model_config_hash": rc["geometry_model_config_hash"],
        "source_geometry_model_config_hash": rc["source_geometry_model_config_hash"],
        "pretrain_code_identity": runtime_identity,
        "training_hash": rc["training_config_hash"],
        "seed": 42,
        "world_size": 3,
        "batch_size": 336,
        "gradient_accumulation_steps": 1,
        "optimizer": "Adam",
        "optimizer_steps": FINAL_OPTIMIZER_STEPS,
        "random_initialization": True,
        "parent_checkpoint": None,
        "initialization": "fresh_paired",
        "initialization_state": init_state,
        "paired_init_id": SHARED_STEP0_ID,
    }
    source_contract_sha256 = _canonical_json_hash(source_contract)
    target_contract = build_pretrain_target_contract(
        profile_id=PRETRAIN_PROFILE_ID,
        source_cohort_hash=feature_cohort_hash,
        feature_config_hash=rc["feature_config_hash"],
        graph_model_config_hash=rc["graph_model_config_hash"],
        geometry_model_config_hash=rc["source_geometry_model_config_hash"],
        topology_cache_artifact_hash=topology_artifact_hash,
        trimer_cache_artifact_hash=trimer_artifact_hash,
        angle_cache_schema=profile["angle_cache_schema"],
        angle_cache_artifact_hash=profile["angle_cache_artifact"],
        cache_bundle_hash=cache_bundle_hash,
        store_json_sha256=store_json_sha256,
        topology_frozen_payload_sha256=topology_frozen_sha,
        trimer_frozen_payload_sha256=trimer_frozen_sha,
        optimizer_steps=FINAL_OPTIMIZER_STEPS,
        pretraining_objective="masked_atom_only",
        topology_representation="canonical_lifted",
    )
    model_config = {
        "core": "paper_corrected",
        "variant": "O8",
        "max_hops": 2,
        "spatial_mode": "trimer_scage",
        "graph_geometry_mode": resolved["graph_geometry_mode"],
        "descriptors": True,
        "modalities": ["graph"],
        "fusion_type": "none",
        "topology_representation": "canonical_lifted",
    }
    calculated_model_hash = hashlib.sha256(
        json.dumps(model_config, sort_keys=True).encode()
    ).hexdigest()
    source_csv = ROOT / "data" / "raw" / "PI1M_v2.csv"
    source_data_hash = _file_sha256(source_csv) if source_csv.is_file() else None
    try:
        git_sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except Exception:
        git_sha = "unknown"
    meta = {
        "schema": PRETRAIN_CHECKPOINT_SCHEMA,
        "baseline": MTS_ROUTE_NAME,
        "route_short_name": MTS_ROUTE_SHORT_NAME,
        "config_schema": "mts-experiment-v3",
        "stage": MTS_STAGE1_ID,
        "modalities": ["graph"],
        "graph_input": "star_linking",
        "geom_input": "repeat_unit",
        "fusion_type": "none",
        "pretraining_unique_smiles": True,
        "feature_schema": MIPS_TRIMER_FEATURE_SCHEMA,
        "topology_representation": "canonical_lifted",
        "cache_layout_schema": MIPS_TRIMER_CACHE_LAYOUT_SCHEMA,
        "cache_bundle_schema": MIPS_TRIMER_CACHE_BUNDLE_SCHEMA,
        "topology_lmdb_schema": MIPS_TRIMER_TOPOLOGY_SCHEMA,
        "cache_bundle_hash": cache_bundle_hash,
        "pretrain_profile": dict(profile),
        "pretrain_profile_sha256": _canonical_json_hash(profile),
        "pretrain_code_identity": runtime_identity,
        "finalization_code_identity": finalization_identity,
        "source_contract": source_contract,
        "source_contract_sha256": source_contract_sha256,
        "target_contract": target_contract,
        "final_cache_binding": final_cache_binding,
        "mips_local_lga_schema_version": MIPS_CANONICAL_LGA_SCHEMA_VERSION,
        "mips_core": "paper_corrected",
        "mips_max_hops": 2,
        "mips_use_descriptors": True,
        "mips_semantics": "paper_semantic",
        "mips_descriptor_components": "md200",
        "mips_descriptor_protocol": "source_star_sub",
        "mips_descriptor_fusion_mode": "graph_md_residual",
        "mips_descriptor_disturbance": 0.0,
        "mips_backbone_mode": "independent",
        "mips_input_norm": None,
        "mips_mask_mode": "zero",
        "mips_mask_policy": "canonical_exact",
        "mips_masked_loss_reduction": "atom_mean",
        "spatial_mode": "trimer_scage",
        "graph_geometry_mode": resolved["graph_geometry_mode"],
        "trimer_cache_hash": rc["trimer_cache_hash"],
        "trimer_cache_artifact_hash": trimer_artifact_hash,
        "topology_cache_hash": rc["topology_cache_hash"],
        "topology_cache_artifact_hash": topology_artifact_hash,
        "feature_cohort_hash": feature_cohort_hash,
        "source_cohort_hash": feature_cohort_hash,
        "feature_cache_item_timeout": 240,
        "trimer_conformer_protocol": MIPS_TRIMER_PROTOCOL,
        "trimer_content_schema": MIPS_TRIMER_CONTENT_SCHEMA,
        "trimer_lmdb_schema": MIPS_TRIMER_LMDB_SCHEMA,
        "trimer_builder_version": MIPS_TRIMER_BUILDER_VERSION,
        "trimer_mmff_relax_steps": MIPS_TRIMER_MMFF_RELAX_STEPS,
        "trimer_require_mmff_convergence": MIPS_TRIMER_REQUIRE_MMFF_CONVERGENCE,
        "trimer_acceptance": MIPS_TRIMER_ACCEPTANCE,
        "trimer_selection": MIPS_TRIMER_SELECTION,
        "trimer_num_candidates": 4,
        "trimer_max_heavy_atoms": 384,
        "mcl_distance_percentiles": [0.2, 0.5],
        "angle_cache_schema": profile["angle_cache_schema"],
        "angle_objective": "categorical",
        "angle_cache_artifact_hash": profile["angle_cache_artifact"],
        "angle_bins": 20,
        "angle_focal_gamma": 2.0,
        "pretraining_objective": "masked_atom_only",
        "angle_loss_weight": 0.0,
        "g_family_arm": arm,
        "g_family_bundle_hash": rc["g_family_bundle_hash"],
        "relation_geometry_sidecar": rc.get("relation_geometry_sidecar"),
        "relation_geometry_artifact_hash": rc.get("relation_geometry_artifact_hash"),
        "g3_permutation_sidecar": rc.get("g3_permutation_sidecar"),
        "g3_permutation_artifact_hash": rc.get("g3_permutation_artifact_hash"),
        "shared_step0_id": SHARED_STEP0_ID,
        "initialization": "fresh_paired",
        "initialization_state": init_state,
        "paired_init_id": SHARED_STEP0_ID,
        "reference_commits": dict(REFERENCE_COMMITS),
        "mips_variant": "O8",
        "model_identity": "T1",
        "topology_attention_variant": "msta_last2",
        "msta_layer_indices": list(resolved["msta_layer_indices"]),
        "msta_local_spd": list(resolved["msta_local_spd"]),
        "msta_context_spd": list(resolved["msta_context_spd"]),
        "msta_share_relation_dropout": True,
        "msta_local_output_bias": False,
        "msta_local_output_init": "zero",
        "experiment_id": resolved["experiment_id"],
        "feature_config_hash": rc["feature_config_hash"],
        "o8_feature_config_hash": rc["o8_feature_config_hash"],
        "model_config_hash": rc["graph_model_config_hash"],
        "graph_model_config_hash": rc["graph_model_config_hash"],
        "geometry_model_config_hash": rc["geometry_model_config_hash"],
        "source_geometry_model_config_hash": rc["geometry_model_config_hash"],
        "alignment_model_config_hash": None,
        "calculated_legacy_model_hash": calculated_model_hash,
        "source_data_hash": source_data_hash,
        "pretraining_dataset": "PI1M_v2",
        "tier": "1m",
        "random_seed": 42,
        "optimizer_steps": FINAL_OPTIMIZER_STEPS,
        "smoke_only": False,
        "training_wall_seconds": None,
        "git_sha": git_sha,
        "dirty_diff_sha": None,
        "training_config_hash": rc["training_config_hash"],
        "tokenizer_mlm_hash": None,
        "model": {
            "architecture": MTS_ROUTE_NAME,
            "layers": 6,
            "embedding_dim": 512,
            "heads": 8,
            "ffn_hidden_dim": 2048,
            "max_hops": 2,
            "path_nodes": 3,
            "readout": "atom_mean",
            "descriptor": "graph_level_MD200",
            "graph_token": False,
        },
        "dynamic_loss_config": None,
        "optimizer_schedule": {
            "optimizer": "Adam",
            "betas": [0.9, 0.98],
            "eps": 1e-8,
            "weight_decay": 0.0,
            "type": "linear_warmup_polynomial_decay",
            "batch_size_per_rank": 336,
            "world_size": 3,
            "effective_batch_size": 1008,
            "warmup_ratio": 0.1,
            "warmup_steps": 2000,
            "scheduler_power": 1.0,
            "end_lr": 1e-9,
            "total_optimizer_steps": FINAL_OPTIMIZER_STEPS,
        },
        "amp_dtype": "bf16",
        "m4p_priors": {
            "masked_atom": 1.0,
            "trimer_bond_angle": 0.0,
            "masked_spd": 0.0,
            "path_bond": 0.0,
        },
        "m4p_geometry_objective": None,
        "alignment_objective": None,
        "fusion_dropout": 0.0,
    }
    for key, value in recovery.items():
        if key in meta:
            raise RuntimeError(f"recovery field collides with final metadata: {key}")
        meta[key] = value
    return meta


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--milestone", type=Path, required=True)
    parser.add_argument("--milestone-sha256", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--arm", choices=("g0", "g1"), required=True)
    parser.add_argument("--step0", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--recovery-reason",
        default="post_training_plot_finalization_failure",
    )
    parser.add_argument("--failure-log", type=Path, default=None)
    parser.add_argument("--audit-output", type=Path, required=True)
    args = parser.parse_args(argv)

    milestone = _load_and_validate_milestone(args.milestone, args.milestone_sha256)
    rc = milestone["resume_contract"]
    resolved = _resolve_config(args.config)
    if resolved.get("g_family_arm") != args.arm:
        raise RuntimeError(
            f"resolved config arm mismatch: {resolved.get('g_family_arm')!r} != {args.arm!r}"
        )
    profile = _load_pretrain_profile(PRETRAIN_PROFILE_ID)
    if _canonical_json_hash(profile) != milestone["profile_sha256"]:
        raise RuntimeError(
            "active pretraining profile does not match the milestone profile_sha256"
        )
    runtime_identity = _runtime_code_identity(rc["pretrain_code_files"])
    if runtime_identity["sha256"] != rc["pretrain_code_hash"]:
        raise RuntimeError(
            "milestone launch-contract code digest mismatch: "
            f"{runtime_identity['sha256']} != {rc['pretrain_code_hash']}"
        )
    _validate_launch_contract(rc, resolved, args.arm, profile)
    init_info = _validate_step0(args.step0, rc, args.arm)

    milestone_state = milestone["state_dict"]
    step0_state = init_info.pop("_state_dict")
    missing = sorted(set(step0_state) - set(milestone_state))
    unexpected = sorted(set(milestone_state) - set(step0_state))
    if missing or unexpected:
        raise RuntimeError(
            "milestone/step-0 architecture mismatch: "
            f"missing={missing[:8]}, unexpected={unexpected[:8]}"
        )
    shape_dtype_mismatch = sorted(
        key
        for key in step0_state
        if tuple(milestone_state[key].shape) != tuple(step0_state[key].shape)
        or milestone_state[key].dtype != step0_state[key].dtype
    )
    if shape_dtype_mismatch:
        raise RuntimeError(
            "milestone/step-0 shape/dtype mismatch: "
            + ", ".join(shape_dtype_mismatch[:8])
        )

    output = args.output.resolve()
    if output.exists() or Path(str(output) + ".complete.json").exists():
        raise RuntimeError(f"refusing to overwrite existing output: {output}")
    finalization_identity = _pretrain_code_identity()
    failure_excerpt = None
    failure_log_path = None
    if args.failure_log is not None:
        failure_log_path = str(args.failure_log.resolve())
        if args.failure_log.is_file():
            lines = args.failure_log.read_text(encoding="utf-8", errors="replace").splitlines()
            failure_excerpt = "\n".join(lines[-25:])
    recovery = {
        "recovered_from_milestone": True,
        "recovery_reason": args.recovery_reason,
        "recovery_source_path": str(milestone["path"]),
        "recovery_source_sha256": milestone["sha256"],
        "recovery_source_optimizer_step": FINAL_OPTIMIZER_STEPS,
        "runtime_pretrain_code_identity": runtime_identity,
        "original_failure_log_path": failure_log_path,
        "original_failure_log_excerpt": failure_excerpt,
    }
    meta = _build_recovery_meta(
        rc=rc,
        resolved=resolved,
        profile=profile,
        init_info=init_info,
        arm=args.arm,
        runtime_identity=runtime_identity,
        finalization_identity=finalization_identity,
        recovery=recovery,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() or Path(str(output) + ".complete.json").exists():
        raise RuntimeError(f"refusing to overwrite existing output: {output}")
    _atomic_torch_save({"state_dict": milestone_state, "meta": meta}, output)
    checkpoint_sha256 = _sha256(output)
    marker_payload = {
        "schema": COMPLETE_SCHEMA,
        "checkpoint_schema": PRETRAIN_CHECKPOINT_SCHEMA,
        "checkpoint": str(output),
        "checkpoint_sha256": checkpoint_sha256,
        "optimizer_steps": FINAL_OPTIMIZER_STEPS,
        "profile_id": PRETRAIN_PROFILE_ID,
        "pretrain_code_sha256": runtime_identity["sha256"],
    }
    complete_path = Path(str(output) + ".complete.json")
    temporary = complete_path.with_name(complete_path.name + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(marker_payload, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, complete_path)

    audit = {
        "schema": "mts-g-family-finalization-recovery-audit-v1",
        "cycle_id": "mts_g1_step20k_finalization_recovery_v1",
        "arm": args.arm,
        "status": "passed",
        "milestone": {
            "path": str(milestone["path"]),
            "sha256": milestone["sha256"],
            "schema": MILESTONE_SCHEMA,
            "checkpoint_schema": PRETRAIN_CHECKPOINT_SCHEMA,
            "optimizer_step": milestone["optimizer_step"],
            "profile_sha256": milestone["profile_sha256"],
            "state_tensor_count": len(milestone_state),
            "head_tensor_count": len(milestone["heads"]),
        },
        "step0": {
            "path": init_info["path"],
            "sha256": init_info["sha256"],
            "paired_init_id": init_info["paired_init_id"],
        },
        "runtime_pretrain_code_identity": runtime_identity,
        "finalization_code_identity": finalization_identity,
        "recovery": recovery,
        "output": {
            "checkpoint": str(output),
            "checkpoint_sha256": checkpoint_sha256,
            "completion_marker": str(complete_path),
            "completion_marker_sha256": _sha256(complete_path),
            "optimizer_steps": FINAL_OPTIMIZER_STEPS,
            "pretrain_code_sha256": runtime_identity["sha256"],
        },
        "state_dict_key_count": len(milestone_state),
        "state_dict_keys_match_step0": True,
        "all_tensors_finite": True,
    }
    audit_output = args.audit_output.resolve()
    if audit_output.exists():
        raise RuntimeError(f"audit output already exists: {audit_output}")
    audit_output.parent.mkdir(parents=True, exist_ok=True)
    audit_tmp = audit_output.with_name(audit_output.name + f".tmp.{os.getpid()}")
    audit_tmp.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(audit_tmp, audit_output)
    print(
        json.dumps(
            {
                "arm": args.arm,
                "checkpoint": str(output),
                "checkpoint_sha256": checkpoint_sha256,
                "completion_marker": str(complete_path),
                "state_tensors": len(milestone_state),
                "audit": str(audit_output),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
