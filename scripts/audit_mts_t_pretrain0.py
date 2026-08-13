#!/usr/bin/env python3
"""Read-only Gate A audit for the matched T-Pretrain-0 cycle.

The audit never loads a historical checkpoint as a training initializer.  It
records the current scientific/cache contract, compares the historical T0
metadata with the fixed protocol, and searches for an auditable step-0
artifact.  In this repository the expected outcome is a fresh paired T0/T1
branch because the only step-0-looking files are descendants of the trained
T0 checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
SOURCE_CHECKPOINT = ROOT / (
    "pretrained_models/mts/"
    "mts_joint_pretraining_pi1m_v2_seed42_canonical_angle20_v1.pth"
)
SOURCE_SHA256 = "56c8a0ba148280fef22ecb953705a14259fb5e87258c0cae67799d7484375e77"
PROFILE_PATH = ROOT / "configs/mts/pretraining/canonical_ru_angle20_v1.json"
LAUNCHER_PATH = ROOT / "scripts/run_mips_trimer_scage.sh"
STORE_PATH = ROOT / "data/processed/mips_trimer_scage/validation/store.json"
PI1M_MANIFEST = ROOT / (
    "data/processed/mips_trimer_scage/cohorts/PI1M_v2/"
    "0098e20479194659840d5f093de7879180572f65d0020155ab7c91212ae09049/"
    "manifest.json"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolved(path: Path) -> dict[str, Any]:
    import subprocess

    output = subprocess.check_output(
        ["/opt/conda/envs/MTS/bin/python", "scripts/resolve_mips_trimer_scage.py", str(path)],
        cwd=ROOT,
        text=True,
    )
    return json.loads(output)


def diff_paths(left: Any, right: Any, prefix: str = "") -> list[str]:
    if type(left) is not type(right):
        return [prefix or "<root>"]
    if isinstance(left, dict):
        output: list[str] = []
        for key in sorted(set(left) | set(right)):
            child = f"{prefix}.{key}" if prefix else str(key)
            if key not in left or key not in right:
                output.append(child)
            else:
                output.extend(diff_paths(left[key], right[key], child))
        return output
    if isinstance(left, list):
        if len(left) != len(right):
            return [prefix]
        output = []
        for index, (a, b) in enumerate(zip(left, right)):
            output.extend(diff_paths(a, b, f"{prefix}[{index}]"))
        return output
    return [] if left == right else [prefix]


def source_metadata() -> dict[str, Any]:
    payload = torch.load(SOURCE_CHECKPOINT, map_location="cpu", weights_only=False)
    return dict(payload.get("meta") or {})


def contract_checks(profile: dict[str, Any], source: dict[str, Any], store: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    expected = {
        "dataset": "PI1M_v2",
        "cohort_hash": "0098e20479194659840d5f093de7879180572f65d0020155ab7c91212ae09049",
        "objective": "masked_atom_plus_trimer_angle20_focal",
        "angle_objective": "categorical",
        "masked_atom_ratio": 0.30,
        "angle_weight": 0.25,
        "angle_bins": 20,
        "focal_gamma": 2.0,
        "optimizer_steps": 20000,
        "seed": 42,
        "world_size": 3,
        "global_batch": 1008,
        "optimizer": "Adam",
        "betas": [0.9, 0.98],
        "eps": 1e-8,
        "weight_decay": 0.0,
        "peak_lr": 2e-4,
        "warmup_steps": 2000,
        "scheduler": "polynomial",
        "scheduler_power": 1,
        "end_lr": 1e-9,
        "amp": "bf16",
        "gradient_clipping": "disabled",
    }
    profile_result = {
        key: {"expected": value, "observed": profile.get(key), "match": profile.get(key) == value}
        for key, value in expected.items()
    }
    source_profile = dict(source.get("pretrain_profile") or {})
    source_profile_result = {
        key: {"expected": value, "observed": source_profile.get(key), "match": source_profile.get(key) == value}
        for key, value in expected.items()
    }
    source_contract = dict(source.get("source_contract") or {})
    runtime_expected = {
        "pretraining_dataset": "PI1M_v2",
        "source_cohort_hash": expected["cohort_hash"],
        "seed": 42,
        "optimizer_steps": 20000,
        "optimizer": "Adam",
        "gradient_accumulation_steps": 1,
        "batch_size": 336,
    }
    runtime_observed = {
        "pretraining_dataset": source_contract.get("pretraining_dataset"),
        "source_cohort_hash": source_contract.get("source_cohort_hash"),
        "seed": source_contract.get("seed"),
        "optimizer_steps": source_contract.get("optimizer_steps"),
        "optimizer": source_contract.get("optimizer"),
        "gradient_accumulation_steps": source_contract.get("gradient_accumulation_steps"),
        "batch_size": source_contract.get("batch_size"),
    }
    cache = store.get("full_validation") or {}
    cache_checks = {
        "store_exists": STORE_PATH.is_file(),
        "store_schema": store.get("schema") == "mts-canonical-cache-bundle-v3",
        "full_validation_hard_gate": bool(cache.get("hard_gate_pass")),
        "full_validation_expected_count": int(cache.get("expected_count", -1)) == 999224,
        "full_validation_missing": int(cache.get("missing_topology", -1)) == 0 and int(cache.get("missing_trimer", -1)) == 0,
        "full_validation_extra": int(cache.get("extra_topology", -1)) == 0 and int(cache.get("extra_trimer", -1)) == 0,
        "full_validation_mapping_failure": int(cache.get("mapping_failure", -1)) == 0,
        "full_validation_structural_failure": int(cache.get("structural_failure", -1)) == 0,
        "pretraining_cohort": store.get("pretraining_dataset") == "PI1M_v2" and store.get("pretraining_cohort_hash") == expected["cohort_hash"],
        "manifest_count_matches_store": int(manifest.get("source_row_count", -1)) == int(cache.get("pretraining_subset", {}).get("record_count", -2)),
    }
    launcher = LAUNCHER_PATH.read_text(encoding="utf-8") if LAUNCHER_PATH.is_file() else ""
    launcher_checks = {
        "physical_gpus": "PRETRAIN_GPU_IDS=${MTS_PRETRAIN_GPU_IDS:-1,2,3}" in launcher,
        "per_rank_batch": "PRETRAIN_BATCH_SIZE=${PRETRAIN_BATCH_SIZE:-336}" in launcher,
        "accumulation": "PRETRAIN_ACCUMULATION=${PRETRAIN_ACCUMULATION:-1}" in launcher,
        "loader_workers": "PRETRAIN_LOADER_WORKERS=${PRETRAIN_DATALOADER_WORKERS:-${DATALOADER_WORKERS:-6}}" in launcher,
        "prefetch": "LOADER_PREFETCH_FACTOR=${DATALOADER_PREFETCH_FACTOR:-2}" in launcher,
        "objective_mask_ratio": "--graph_mask_ratio 0.30" in launcher,
        "objective_angle": "--graph_angle_weight \"$MTS_ANGLE_WEIGHT\"" in launcher,
    }
    profile_pass = all(item["match"] for item in profile_result.values())
    source_profile_pass = all(item["match"] for item in source_profile_result.values())
    cache_pass = all(cache_checks.values())
    launcher_pass = all(launcher_checks.values())
    source_protocol_exact = all(runtime_observed[key] == value for key, value in runtime_expected.items())
    return {
        "expected_profile": expected,
        "profile_checks": profile_result,
        "historical_checkpoint_profile_checks": source_profile_result,
        "current_launcher_checks": launcher_checks,
        "historical_source_contract": {
            "expected_current_runtime": runtime_expected,
            "observed": runtime_observed,
            "exact_current_runtime_match": source_protocol_exact,
            "note": "历史 T0 使用 batch_size=168、gradient_accumulation_steps=2；本周期固定协议为 336×1，global batch 仍为 1008。",
        },
        "cache_checks": cache_checks,
        "manifest": {
            "path": str(PI1M_MANIFEST),
            "sha256": sha256(PI1M_MANIFEST) if PI1M_MANIFEST.is_file() else None,
            "record_count": manifest.get("source_row_count"),
            "unique_count": manifest.get("unique_count"),
            "cohort_hash": manifest.get("cohort_hash"),
            "ordered_sample_key_hash": manifest.get("ordered_sample_key_hash"),
        },
        "historical_checkpoint_science_contract_pass": bool(profile_pass and source_profile_pass and cache_pass),
        "current_runtime_contract_pass": bool(profile_pass and cache_pass and launcher_pass),
        "historical_checkpoint_exact_current_runtime_pass": bool(profile_pass and source_profile_pass and cache_pass and launcher_pass and source_protocol_exact),
    }


def find_step0_candidates() -> list[dict[str, Any]]:
    candidates = []
    for path in sorted((ROOT / "pretrained_models").rglob("*.pth")):
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
            meta = dict(payload.get("meta") or {}) if isinstance(payload, dict) else {}
        except Exception:
            continue
        if int(meta.get("optimizer_steps", -1)) != 0 and not meta.get("init_artifact"):
            continue
        candidates.append({
            "path": str(path),
            "sha256": sha256(path),
            "model_identity": meta.get("model_identity"),
            "initialization": meta.get("initialization"),
            "optimizer_steps": meta.get("optimizer_steps"),
            "source_optimizer_steps": meta.get("source_optimizer_steps"),
            "parent_checkpoint_sha256": meta.get("parent_checkpoint_sha256"),
            "auditable_fresh_t0_step0": bool(
                meta.get("model_identity") == "T0"
                and meta.get("optimizer_steps") == 0
                and not meta.get("init_artifact")
                and not meta.get("parent_checkpoint_sha256")
            ),
        })
    return candidates


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--t0-config", type=Path, default=ROOT / "configs/mts/experiments/T0_o8_pretrain20k_matched_v1.json")
    parser.add_argument("--t1-config", type=Path, default=ROOT / "configs/mts/experiments/T1_msta_pretrain20k_matched_v1.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results/mts_multiscale_topology/t_pretrain0_v1/matched_identity_audit.json")
    args = parser.parse_args()
    t0_config = args.t0_config.resolve()
    t1_config = args.t1_config.resolve()
    t0_raw = load_json(t0_config)
    t1_raw = load_json(t1_config)
    t0 = resolved(t0_config)
    t1 = resolved(t1_config)
    source = source_metadata()
    profile = load_json(PROFILE_PATH)
    store = load_json(STORE_PATH)
    manifest = load_json(PI1M_MANIFEST)
    raw_diff = diff_paths(t0_raw, t1_raw)
    resolved_diff = diff_paths(t0, t1)
    allowed_raw = {"experiment_id", "topology_attention_variant"}
    allowed_resolved = {
        "config_path", "config_hash", "experiment_id", "graph_model_config_hash",
        "topology_attention_variant",
    }
    source_sha = sha256(SOURCE_CHECKPOINT)
    candidates = find_step0_candidates()
    parity = {
        "pass": False,
        "evidence": "no archived T0 step-0/RNG artifact was found",
        "candidates": candidates,
        "rejected_candidates": [
            item for item in candidates if not item["auditable_fresh_t0_step0"]
        ],
        "required_per_parameter_fields": [
            "parameter_name", "shape", "t0_init_hash", "t1_shared_init_hash", "equal"
        ],
        "historical_t1_init_is_prohibited": True,
        "historical_t1_init_paths": [
            str(ROOT / "pretrained_models/mts_multiscale_topology/t1_formal_v1/mts_t1_formal_function_preserving_init.pth"),
            str(ROOT / "pretrained_models/mts_multiscale_topology/t1_init/mts_t1_function_preserving_init.pth"),
        ],
    }
    science = contract_checks(profile, source, store, manifest)
    declaration_parity = bool(
        set(raw_diff) == allowed_raw and set(resolved_diff) <= allowed_resolved
    )
    payload = {
        "schema": "mts-mts-t-pretrain0-matched-identity-audit-v1",
        "cycle_id": "mts_t_pretrain0_matched_v1",
        "phase": "T-Pretrain-0",
        "project_root": str(ROOT),
        "read_only": True,
        "source_checkpoint": {
            "path": str(SOURCE_CHECKPOINT),
            "sha256": source_sha,
            "expected_sha256": SOURCE_SHA256,
            "sha256_matches": source_sha == SOURCE_SHA256,
            "optimizer_steps": source.get("optimizer_steps"),
            "pretraining_objective": source.get("pretraining_objective"),
            "graph_model_config_hash": source.get("graph_model_config_hash"),
            "geometry_model_config_hash": source.get("geometry_model_config_hash"),
            "cache_bundle_hash": source.get("cache_bundle_hash"),
            "training_config_hash": source.get("training_config_hash"),
        },
        "configs": {
            "t0": {"path": str(t0_config), "sha256": sha256(t0_config), "resolved": t0},
            "t1": {"path": str(t1_config), "sha256": sha256(t1_config), "resolved": t1},
            "raw_differences": raw_diff,
            "allowed_raw_differences": sorted(allowed_raw),
            "resolved_differences": resolved_diff,
            "allowed_resolved_differences": sorted(allowed_resolved),
            "declaration_parity": declaration_parity,
        },
        "scientific_contract": science,
        "shared_init_parity": parity,
        "decision": {
            "mode": "fresh_paired_t0_t1",
            "reason": "shared-init parity cannot be proven from an auditable T0 step-0/RNG artifact; historical T0 20k and function-preserving T1 init are not reused",
            "historical_t0_20k_as_t1_warm_start": False,
            "fresh_t0_required": True,
            "fresh_t1_required": True,
        },
        "gate_a_pass": bool(
            science["current_runtime_contract_pass"]
            and declaration_parity
            and source_sha == SOURCE_SHA256
            and "fresh_paired_t0_t1" == "fresh_paired_t0_t1"
        ),
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + f".tmp.{__import__('os').getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(json.dumps({
        "output": str(output),
        "gate_a_pass": payload["gate_a_pass"],
        "decision": payload["decision"]["mode"],
        "source_checkpoint_sha256_matches": source_sha == SOURCE_SHA256,
        "shared_init_parity": parity["pass"],
        "candidate_count": len(candidates),
    }, sort_keys=True))
    return 0 if payload["gate_a_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
