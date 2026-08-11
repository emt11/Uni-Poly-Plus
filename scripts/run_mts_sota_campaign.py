#!/usr/bin/env python3
"""DAG-aware launcher and screener for the bounded MTS SOTA experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TASKS = ("eat", "eea", "egb", "egc", "ei", "eps", "nc", "xc")
STATE_PATH = ROOT / "results" / "mts_sota_v3" / "promotion_state.json"
STATE_SCHEMA = "mts-sota-promotion-v3"
BASE_PROFILE = "legacy_mts_huber_v1"
BASE_REFERENCE_MACRO_R2 = 0.8423414007

# The historical G0 identities remain read-only evidence.  They are deliberately
# not members of the active phase list below.
G0_LEGACY = "G0_current_mcl"
G0_HISTORICAL_CANONICAL = "G0_canonical_contract"
G0_CANONICAL = "G0_canonical_angle20_v1"
G0_CANONICAL_CHECKPOINT = (
    "pretrained_models/mts/"
    "mts_joint_pretraining_pi1m_v2_seed42_canonical_angle20_v1.pth"
)
G0_CANONICAL_CHECKPOINT_SHA256 = (
    "56c8a0ba148280fef22ecb953705a14259fb5e87258c0cae67799d7484375e77"
)
G0_CANONICAL_COMPLETE_SCHEMA = "mts-pretrain-complete-v1"
G0_CANONICAL_CHECKPOINT_SCHEMA = "mts-model-v4"
G0_CANONICAL_PROFILE = "canonical_ru_angle20_v1"
G0_CANONICAL_REPRESENTATION = "canonical_lifted"
G0_CANONICAL_DATASET = "PI1M_v2"
CANONICAL_REFERENCE_MACRO_R2 = 0.8407410728423622
CANONICAL_REFERENCE_TASKS = {
    "eat": 0.986754,
    "eea": 0.923731,
    "egb": 0.943120,
    "egc": 0.919504,
    "ei": 0.843157,
    "eps": 0.809600,
    "nc": 0.873512,
    "xc": 0.426550,
}
SMOKE_ROOT = ROOT / "results" / "mts_sota_v3" / "_smoke" / G0_CANONICAL

# G0 is an explicit parent baseline.  G1--G4 stay defined for a later
# activity cycle, but the old G0 identities are never active candidates.
PHASES = {
    "G": (
        G0_CANONICAL, "G1_mcl_disabled", "G2_coordinate_shuffled",
        "G3_mcl_rbf", "G4_mcl_rbf_shuffled",
    ),
    "V": ("V1_smiles", "V2_countfp", "V3_smiles_countfp"),
    "MT": ("MT0_single_task", "MT1_multitask_pcgrad"),
    "FINAL": (
        "FINAL_geometry_nested5",
        "FINAL_modality_nested5",
        "FINAL_multitask_nested5",
    ),
}


def template_config_path(name):
    return ROOT / "configs" / "mts" / "experiments" / f"{name}.json"


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _find_cache_layer(kind, artifact_id):
    """Return the frozen cache root matching a checkpoint artifact binding."""
    base = ROOT / "data" / "processed" / "mips_trimer_scage" / kind
    for root in sorted(base.glob("*")):
        done = root / ".done"
        frozen = root / ".frozen"
        if (
            done.is_file()
            and done.read_text(encoding="utf-8").strip() == str(artifact_id)
            and frozen.is_file()
        ):
            return root
    raise RuntimeError(
        f"checkpoint cache artifact {kind}={artifact_id!r} has no frozen root"
    )


def _validate_g0_checkpoint():
    """Validate the immutable Angle-20 checkpoint before any Stage-3 launch."""
    checkpoint = ROOT / G0_CANONICAL_CHECKPOINT
    complete_path = Path(str(checkpoint) + ".complete.json")
    if not checkpoint.is_file() or not complete_path.is_file():
        raise RuntimeError(
            f"missing canonical Angle-20 checkpoint or complete metadata: {checkpoint}"
        )
    actual_sha = _sha256(checkpoint)
    complete = json.loads(complete_path.read_text(encoding="utf-8"))
    checks = {
        "complete schema": complete.get("schema") == G0_CANONICAL_COMPLETE_SCHEMA,
        "complete checkpoint SHA": complete.get("checkpoint_sha256") == actual_sha == G0_CANONICAL_CHECKPOINT_SHA256,
        "complete checkpoint schema": complete.get("checkpoint_schema") == G0_CANONICAL_CHECKPOINT_SCHEMA,
        "complete optimizer steps": int(complete.get("optimizer_steps", -1)) == 20000,
        "complete profile": complete.get("profile_id") == G0_CANONICAL_PROFILE,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(
            "canonical Angle-20 complete metadata mismatch: " + ", ".join(failed)
        )

    # Import torch lazily so CLI parsing and static helpers remain lightweight.
    import torch

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or not isinstance(payload.get("meta"), dict):
        raise RuntimeError("canonical checkpoint lacks a metadata contract")
    meta = payload["meta"]
    source = meta.get("source_contract")
    target = meta.get("target_contract")
    profile = meta.get("pretrain_profile")
    if not isinstance(source, dict) or not isinstance(target, dict) or not isinstance(profile, dict):
        raise RuntimeError("canonical checkpoint source/target/profile contract is incomplete")
    exact_checks = {
        "checkpoint schema": meta.get("schema") == G0_CANONICAL_CHECKPOINT_SCHEMA,
        "checkpoint stage": meta.get("stage") == "mts_joint_pretraining",
        "checkpoint representation": meta.get("topology_representation") == G0_CANONICAL_REPRESENTATION,
        "checkpoint dataset": meta.get("pretraining_dataset") == G0_CANONICAL_DATASET,
        "profile id": profile.get("profile_id") == G0_CANONICAL_PROFILE,
        "profile representation": profile.get("representation") == G0_CANONICAL_REPRESENTATION,
        "profile dataset": profile.get("dataset") == G0_CANONICAL_DATASET,
        "profile steps": int(profile.get("optimizer_steps", -1)) == 20000,
        "source schema": source.get("schema") == G0_CANONICAL_CHECKPOINT_SCHEMA,
        "source representation": source.get("topology_representation") == G0_CANONICAL_REPRESENTATION,
        "source dataset": source.get("pretraining_dataset") == G0_CANONICAL_DATASET,
        "target schema": target.get("schema") == "mts-canonical-target-contract-v2",
        "target representation": target.get("topology_representation") == G0_CANONICAL_REPRESENTATION,
        "target checkpoint schema": target.get("checkpoint_schema") == G0_CANONICAL_CHECKPOINT_SCHEMA,
        "source contract digest": bool(meta.get("source_contract_sha256")),
        "target contract digest": bool(target.get("target_contract_sha256")),
    }
    failed = [name for name, passed in exact_checks.items() if not passed]
    if failed:
        raise RuntimeError(
            "canonical Angle-20 checkpoint contract mismatch: " + ", ".join(failed)
        )

    store = ROOT / "data" / "processed" / "mips_trimer_scage" / "validation" / "store.json"
    store_sha = _sha256(store) if store.is_file() else ""
    topology_artifact = str(target.get("topology_cache_artifact_hash", ""))
    trimer_artifact = str(target.get("trimer_cache_artifact_hash", ""))
    if not store_sha or store_sha != str(target.get("store_json_sha256", "")):
        raise RuntimeError("canonical checkpoint store.json binding mismatch")
    if store_sha != str(source.get("cache_store_sha256", "")):
        raise RuntimeError("canonical checkpoint source cache binding mismatch")
    topology_root = _find_cache_layer("topology", topology_artifact)
    trimer_root = _find_cache_layer("trimer", trimer_artifact)
    if _sha256(topology_root / ".frozen") != str(target.get("topology_frozen_payload_sha256", "")):
        raise RuntimeError("canonical checkpoint topology .frozen binding mismatch")
    if _sha256(trimer_root / ".frozen") != str(target.get("trimer_frozen_payload_sha256", "")):
        raise RuntimeError("canonical checkpoint Trimer .frozen binding mismatch")
    return {
        "checkpoint_sha256": actual_sha,
        "checkpoint_schema": G0_CANONICAL_CHECKPOINT_SCHEMA,
        "cache_store_sha256": store_sha,
        "topology_cache_artifact_hash": topology_artifact,
        "trimer_cache_artifact_hash": trimer_artifact,
        "feature_config_hash": str(target.get("feature_config_hash", "")),
        "source_cohort_hash": str(source.get("source_cohort_hash", "")),
    }


def _read_state():
    if not STATE_PATH.is_file():
        return {}
    payload = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    if payload.get("schema") != STATE_SCHEMA:
        # Never inherit an old F-based DAG or a partially written state.
        return {}
    return payload


def _atomic_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def materialize_config(name):
    """Freeze the promoted parent into a config rooted at the old MTS baseline."""
    payload = json.loads(template_config_path(name).read_text(encoding="utf-8"))
    state = _read_state()
    if name.startswith(("G", "V", "MT", "FINAL")):
        payload["finetune_profile"] = BASE_PROFILE
        payload["regression_loss"] = "huber"
        payload["huber_beta"] = 0.5
        payload["patience"] = 10
        payload["warmup_epochs"] = 5
        payload["weight_decay"] = 0.02
        payload["swa_start_epoch"] = -1
        payload["legacy_graph_trainable_from_epoch0"] = True
    if name.startswith(("V", "MT", "FINAL")):
        geometry_winner = state.get("G", {}).get("winner")
        if geometry_winner is None:
            raise RuntimeError("phase G must be promoted before V/MT/FINAL")
        payload["geometry_mode"] = (
            "mcl_rbf" if geometry_winner == "G3_mcl_rbf" else "current_mcl"
        )
    if name.startswith(("MT", "FINAL")):
        modality_winner = state.get("V", {}).get("winner")
        if modality_winner is None:
            raise RuntimeError("phase V must be promoted before MT/FINAL")
        modalities = {
            "V1_smiles": ["graph", "smiles"],
            "V2_countfp": ["graph", "fp"],
            "V3_smiles_countfp": ["graph", "smiles", "fp"],
            "G0_current_mcl": ["graph"],
            "G3_mcl_rbf": ["graph"],
        }[modality_winner]
        payload["modalities"] = modalities
        payload["fusion_mode"] = (
            "none" if modalities == ["graph"] else "zero_gated_residual"
        )
    if name.startswith("FINAL"):
        multitask_winner = state.get("MT", {}).get("winner")
        if multitask_winner is None:
            raise RuntimeError("phase MT must be promoted before FINAL")
        if name == "FINAL_geometry_nested5":
            payload["modalities"] = ["graph"]
            payload["fusion_mode"] = "none"
            payload["finetune_mode"] = "single_task"
        elif name == "FINAL_modality_nested5":
            payload["finetune_mode"] = "single_task"
        elif name == "FINAL_multitask_nested5":
            payload["finetune_mode"] = (
                "multitask_pcgrad"
                if multitask_winner == "MT1_multitask_pcgrad"
                else "single_task"
            )
        else:
            raise RuntimeError(f"unknown FINAL candidate: {name}")
    destination = (
        ROOT / "results" / "mts_sota_v3" / "configs" / f"{name}.json"
    )
    _atomic_json(destination, payload)
    return destination


def run_config(name, folds, seed, dry_run=False, tasks=TASKS, output_root=None):
    if name in PHASES["G"]:
        checkpoint_identity = _validate_g0_checkpoint()
    else:
        checkpoint_identity = None
    output = (
        Path(output_root)
        if output_root is not None
        else ROOT / "results" / "mts_sota_v3" / name
    )
    env = os.environ.copy()
    env.update({
        "CUDA_VISIBLE_DEVICES": "0,1,2",
        "EXPERIMENT_CONFIG": str(materialize_config(name)),
        "FINETUNE_ONLY": "1",
        # The immutable joint-pretraining artifact is seed 42.  Downstream
        # seed 43/44 runs must retain that checkpoint identity and include
        # seed 42 so the launcher's promotion/resume contract remains valid.
        "FINETUNE_SEEDS": (
            "42" if int(seed) == 42 else f"42 {int(seed)}"
        ),
        "RANDOM_SEED": "42",
        # Every active G candidate is tied to the immutable Angle-20
        # checkpoint.  Historical checkpoints are never a campaign input.
        "JOINT_CKPT": str(ROOT / G0_CANONICAL_CHECKPOINT),
        "TASKS": " ".join(tasks),
        "FOLD_IDS": " ".join(str(value) for value in folds),
        "RESULTS_DIR": str(output),
    })
    command = ["bash", "scripts/run_mts.sh"]
    print("[mts-sota]", name, " ".join(command), flush=True)
    if dry_run:
        # Plan §9.6: surface the exact checkpoint so a dry run proves the
        # campaign resolves the full dual-identity production file.
        print(f"[mts-sota] dry-run checkpoint: {env['JOINT_CKPT']}", flush=True)
        if checkpoint_identity is not None:
            print(
                "[mts-sota] checkpoint identity: "
                f"{checkpoint_identity['checkpoint_sha256']}",
                flush=True,
            )
        return 0
    return subprocess.call(command, cwd=ROOT, env=env)


def collect(name, seed=42, folds=(0, 1, 2, 3, 4), results_root=None, tasks=TASKS):
    base = (
        Path(results_root)
        if results_root is not None
        else ROOT / "results" / "mts_sota_v3" / name
    )
    root = base / "shards" / str(seed)
    values = {}
    controls = []
    for task in tasks:
        task_values = []
        for fold in folds:
            path = root / task / f"fold_{fold}.csv"
            if not path.is_file():
                raise RuntimeError(f"missing screening shard: {path}")
            frame = pd.read_csv(path)
            if len(frame) != 1:
                raise RuntimeError(f"invalid screening shard: {path}")
            task_values.append(float(frame.iloc[0]["avg_test_r2"]))
            payload = json.loads(frame.iloc[0]["per_fold_metrics"])
            item = payload[0]
            for key, value in item.items():
                if key.endswith("_test_r2") and (
                    "batch_shuffled" in key or "constant_zero" in key
                ):
                    controls.append((key, float(value), float(item["test_r2"])))
        values[task] = float(np.mean(task_values))
    return {
        "experiment_id": name,
        **{f"r2_{task}": values[task] for task in TASKS},
        "macro_r2": float(np.mean(list(values.values()))),
        "worst_control_margin": (
            min(real - control for _, control, real in controls)
            if controls else float("nan")
        ),
    }


def collect_inner_validation(name, seed=42, folds=(0, 1, 2, 3, 4)):
    """Collect only inner-validation scores for nested FINAL selection.

    Outer-test metrics are intentionally not read by this selector.  The
    shard still has to contain one complete row per task/fold, but the value
    used to choose the candidate is the validation score selected during that
    fold's training.
    """
    root = ROOT / "results" / "mts_sota_v3" / name / "shards" / str(seed)
    values = {}
    for task in TASKS:
        task_values = []
        for fold in folds:
            path = root / task / f"fold_{fold}.csv"
            if not path.is_file():
                raise RuntimeError(f"missing nested validation shard: {path}")
            frame = pd.read_csv(path)
            if len(frame) != 1 or "avg_best_val_r2" not in frame.columns:
                raise RuntimeError(f"invalid nested validation shard: {path}")
            value = float(frame.iloc[0]["avg_best_val_r2"])
            if not np.isfinite(value):
                raise RuntimeError(f"non-finite inner validation score: {path}")
            task_values.append(value)
        values[task] = float(np.mean(task_values))
    return {
        "experiment_id": name,
        **{f"inner_r2_{task}": values[task] for task in TASKS},
        "inner_macro_r2": float(np.mean(list(values.values()))),
    }


def _require_full_folds(folds):
    normalized = tuple(int(value) for value in folds)
    if normalized != (0, 1, 2, 3, 4):
        raise ValueError(
            "formal MTS campaign commands require exactly folds 0,1,2,3,4; "
            "use the separate smoke entry point for partial folds"
        )


def _check_g0_reproduction(candidate):
    """Verify the legacy MTS anchor before spending time on G1--G4.

    The F retirement must not accidentally change the old baseline.  This
    check uses the immutable historical five-fold summary and is deliberately
    performed before any geometry proposal is launched.
    """
    reference_path = ROOT / "results" / "mts" / "mts_summary.csv"
    if not reference_path.is_file():
        raise RuntimeError(f"missing legacy MTS reference summary: {reference_path}")
    reference = pd.read_csv(reference_path)
    required = {"task", "fold_test_r2"}
    if not required.issubset(reference.columns):
        raise RuntimeError(
            f"legacy MTS summary lacks required columns: {sorted(required - set(reference.columns))}"
        )
    reference_values = reference.groupby("task")["fold_test_r2"].mean()
    missing = sorted(set(TASKS) - set(reference_values.index))
    if missing:
        raise RuntimeError(f"legacy MTS reference lacks tasks: {missing}")
    deltas = {
        task: float(candidate[f"r2_{task}"] - reference_values[task])
        for task in TASKS
    }
    macro = float(np.mean([candidate[f"r2_{task}"] for task in TASKS]))
    reference_macro = float(reference_values.loc[list(TASKS)].mean())
    result = {
        "passed": bool(
            abs(macro - reference_macro) <= 0.001
            and min(deltas.values()) >= -0.005
        ),
        "candidate_macro_r2": macro,
        "reference_macro_r2": reference_macro,
        "macro_delta": macro - reference_macro,
        "task_deltas": deltas,
        "macro_tolerance": 0.001,
        "task_tolerance": -0.005,
    }
    return result


def _assert_canonical_g0_identity(seed=42, folds=(0, 1, 2, 3, 4)):
    """Strictly validate a completed new G0 before allowing reuse."""
    identity = _validate_g0_checkpoint()
    root = ROOT / "results" / "mts_sota_v3" / G0_CANONICAL
    shards = sorted((root / "shards" / str(seed)).glob("*/fold_*.csv"))
    if len(shards) != 8 * 5:
        raise RuntimeError(
            f"{G0_CANONICAL} reuse requires {8 * 5} shards; found {len(shards)}"
        )
    seen = set()
    identity_fields = {
        "checkpoint_sha256": identity["checkpoint_sha256"],
        "checkpoint_schema": identity["checkpoint_schema"],
        "cache_store_sha256": identity["cache_store_sha256"],
        "topology_cache_artifact_hash": identity["topology_cache_artifact_hash"],
        "trimer_cache_artifact_hash": identity["trimer_cache_artifact_hash"],
    }
    for shard in shards:
        frame = pd.read_csv(shard)
        if len(frame) != 1:
            raise RuntimeError(f"invalid {G0_CANONICAL} shard: {shard}")
        row = frame.iloc[0]
        if str(row.get("experiment_id", "")) != G0_CANONICAL:
            raise RuntimeError(f"{G0_CANONICAL} experiment identity mismatch: {shard.name}")
        for field, expected in identity_fields.items():
            if str(row.get(field, "")) != expected:
                raise RuntimeError(f"{G0_CANONICAL} {field} mismatch: {shard.name}")
        if int(row.get("seed", -1)) != int(seed):
            raise RuntimeError(f"{G0_CANONICAL} shard seed mismatch: {shard.name}")
        task = str(row.get("task", ""))
        if task not in TASKS:
            raise RuntimeError(f"{G0_CANONICAL} task mismatch: {shard.name}")
        prediction = Path(str(row.get("prediction_path", "")))
        if not prediction.is_absolute():
            prediction = ROOT / prediction
        expected_prediction_sha = str(row.get("prediction_sha256", ""))
        if not prediction.is_file() or not expected_prediction_sha:
            raise RuntimeError(f"{G0_CANONICAL} prediction missing: {shard.name}")
        if _sha256(prediction) != expected_prediction_sha:
            raise RuntimeError(f"{G0_CANONICAL} prediction hash mismatch: {shard.name}")
        split_path = ROOT / "data" / "splits" / "mips_shared5" / f"{task}.json"
        split_hash = _sha256_json(split_path)
        if str(row.get("split_manifest_hash", "")) != split_hash:
            raise RuntimeError(f"{G0_CANONICAL} split hash mismatch: {shard.name}")
        metrics = json.loads(str(row.get("per_fold_metrics", "[]")))
        if len(metrics) != 1:
            raise RuntimeError(f"{G0_CANONICAL} fold metrics mismatch: {shard.name}")
        fold = int(metrics[0].get("fold", -1))
        if fold not in folds:
            raise RuntimeError(f"{G0_CANONICAL} shard fold out of range: {shard.name}")
        unit = (task, fold)
        if unit in seen:
            raise RuntimeError(f"duplicate {G0_CANONICAL} unit: {unit}")
        seen.add(unit)
    expected = {(task, fold) for task in TASKS for fold in folds}
    if seen != expected:
        missing = sorted(expected - seen)
        raise RuntimeError(
            f"{G0_CANONICAL} reuse is missing units: {missing[:10]}"
        )
    print(
        f"[mts-sota] {G0_CANONICAL} reuse identity verified: "
        f"{len(seen)} units, checkpoint {identity['checkpoint_sha256'][:12]}..."
    )


def _sha256_json(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _write_historical_comparison(candidate, result_root):
    """Write diagnostic-only comparisons; none of these values is a gate."""
    best_path = ROOT / "results" / "best_result.csv"
    best_frame = pd.read_csv(best_path) if best_path.is_file() else pd.DataFrame()
    best_tasks = {
        str(row.task): float(row.best_r2)
        for row in best_frame.itertuples(index=False)
    } if not best_frame.empty else {}
    candidate_tasks = {
        task: float(candidate[f"r2_{task}"]) for task in TASKS
    }
    candidate_macro = float(candidate["macro_r2"])
    best_macro = (
        float(np.mean([best_tasks[task] for task in TASKS]))
        if set(TASKS).issubset(best_tasks) else None
    )
    comparison = {
        "experiment_id": G0_CANONICAL,
        "diagnostic_only": True,
        "candidate_macro_r2": candidate_macro,
        "candidate_task_r2": candidate_tasks,
        "comparisons": {
            "old_canonical": {
                "experiment_id": G0_HISTORICAL_CANONICAL,
                "reference_macro_r2": CANONICAL_REFERENCE_MACRO_R2,
                "macro_delta": candidate_macro - CANONICAL_REFERENCE_MACRO_R2,
                "reference_task_r2": dict(CANONICAL_REFERENCE_TASKS),
                "task_deltas": {
                    task: candidate_tasks[task] - CANONICAL_REFERENCE_TASKS[task]
                    for task in TASKS
                },
            },
            "old_mts": {
                "reference_experiment": "historical_mts_summary",
                "reference_macro_r2": BASE_REFERENCE_MACRO_R2,
                "macro_delta": candidate_macro - BASE_REFERENCE_MACRO_R2,
            },
            "best_result": {
                "path": str(best_path),
                "reference_macro_r2": best_macro,
                "macro_delta": candidate_macro - best_macro if best_macro is not None else None,
                "reference_task_r2": best_tasks,
                "task_deltas": (
                    {task: candidate_tasks[task] - best_tasks[task] for task in TASKS}
                    if best_macro is not None else {}
                ),
            },
        },
    }
    _atomic_json(Path(result_root) / "historical_comparison.json", comparison)
    return comparison


def _assert_smoke_identity(result_root):
    """Check the single smoke shard without treating it as formal evidence."""
    identity = _validate_g0_checkpoint()
    shard = Path(result_root) / "shards" / "42" / "eat" / "fold_0.csv"
    prediction = Path(result_root) / "predictions" / "42" / "eat" / "fold_0.npz"
    if not shard.is_file() or not prediction.is_file():
        raise RuntimeError(f"smoke did not produce atomic eat/fold0 outputs: {result_root}")
    frame = pd.read_csv(shard)
    if len(frame) != 1:
        raise RuntimeError("smoke shard is not exactly one row")
    row = frame.iloc[0]
    for field, expected in {
        "experiment_id": G0_CANONICAL,
        "checkpoint_sha256": identity["checkpoint_sha256"],
        "checkpoint_schema": identity["checkpoint_schema"],
        "cache_store_sha256": identity["cache_store_sha256"],
        "topology_cache_artifact_hash": identity["topology_cache_artifact_hash"],
        "trimer_cache_artifact_hash": identity["trimer_cache_artifact_hash"],
        "seed": 42,
        "task": "eat",
    }.items():
        if str(row.get(field, "")) != str(expected):
            raise RuntimeError(f"smoke identity mismatch ({field})")
    expected_sha = str(row.get("prediction_sha256", ""))
    if not expected_sha or _sha256(prediction) != expected_sha:
        raise RuntimeError("smoke prediction hash mismatch")
    if not np.isfinite(float(row.get("avg_test_r2", np.nan))):
        raise RuntimeError("smoke produced a non-finite metric")


def write_summary(names, seed, folds):
    rows = [collect(name, seed=seed, folds=folds) for name in names]
    frame = pd.DataFrame(rows)
    destination = ROOT / "results" / "mts_sota_v3" / "screening_summary.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    existing = pd.read_csv(destination) if destination.is_file() else pd.DataFrame()
    if not existing.empty:
        # F was a rejected loss/micro-finetuning experiment.  It must not be
        # reintroduced into an active screening summary by an old CSV.
        existing = existing[
            ~existing["experiment_id"].astype(str).str.startswith("F0_")
        ]
        existing = existing[~existing["experiment_id"].isin(frame["experiment_id"])]
        frame = pd.concat([existing, frame], ignore_index=True)
    temporary = destination.with_suffix(".csv.tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, destination)
    print(frame.to_string(index=False))


def _deltas(candidate, parent):
    values = [
        float(candidate[f"r2_{task}"]) - float(parent[f"r2_{task}"])
        for task in TASKS
    ]
    return {
        "macro_delta": float(candidate["macro_r2"] - parent["macro_r2"]),
        "nondecreasing_tasks": int(sum(value >= 0.0 for value in values)),
        "worst_task_delta": float(min(values)),
    }


def promote(phase, seed, folds):
    state = _read_state()
    if state.get("schema") not in {None, STATE_SCHEMA}:
        raise RuntimeError(
            f"unsupported promotion state schema: {state.get('schema')!r}"
        )
    # A stale v1 state may contain an F winner.  It is deliberately ignored;
    # the new DAG always starts from the immutable legacy MTS baseline.
    state = {
        "schema": STATE_SCHEMA,
        "base": {
            "name": BASE_PROFILE,
            "reference_macro_r2": BASE_REFERENCE_MACRO_R2,
        },
        **{key: value for key, value in state.items()
           if key in PHASES},
    }
    prerequisites = {
        "V": "G",
        "MT": "V",
        "FINAL": "MT",
    }
    prerequisite = prerequisites.get(phase)
    if prerequisite is not None and prerequisite not in state:
        raise RuntimeError(
            f"phase {prerequisite} must be promoted before phase {phase}"
        )
    rows = {name: collect(name, seed, folds) for name in PHASES[phase]}
    if phase == "G":
        # The canonical single-RU G0 is the promotion parent (Plan §4); the
        # legacy multi-RU G0 is history only and is not a G promotion base.
        rows[G0_CANONICAL] = collect(G0_CANONICAL, seed, folds)
    detail = {"seed": int(seed), "folds": list(folds), "candidates": rows}
    if phase == "G":
        parent, candidate = rows[G0_CANONICAL], rows["G3_mcl_rbf"]
        gate = _deltas(candidate, parent)
        gate["coordinate_margin"] = float(
            candidate["macro_r2"] - rows["G4_mcl_rbf_shuffled"]["macro_r2"]
        )
        gate["passed"] = bool(
            gate["macro_delta"] >= 0.003
            and gate["coordinate_margin"] >= 0.002
            and gate["nondecreasing_tasks"] >= 5
            and gate["worst_task_delta"] >= -0.010
        )
        detail["gate"] = gate
        winner = "G3_mcl_rbf" if gate["passed"] else G0_CANONICAL
    elif phase == "V":
        parent_name = state.get("G", {}).get("winner")
        if not parent_name:
            raise RuntimeError("phase G has not been promoted")
        parent = collect(parent_name, seed, folds)
        eligible = []
        detail["parent"] = parent_name
        detail["gates"] = {}
        for name, candidate in rows.items():
            gate = _deltas(candidate, parent)
            gate["control_margin"] = candidate["worst_control_margin"]
            gate["passed"] = bool(
                gate["macro_delta"] >= 0.003
                and gate["nondecreasing_tasks"] >= 5
                and gate["worst_task_delta"] >= -0.010
                and np.isfinite(gate["control_margin"])
                and gate["control_margin"] > 0.0
            )
            detail["gates"][name] = gate
            if gate["passed"]:
                eligible.append(name)
        winner = (
            max(eligible, key=lambda name: rows[name]["macro_r2"])
            if eligible else parent_name
        )
    elif phase == "MT":
        parent, candidate = rows["MT0_single_task"], rows["MT1_multitask_pcgrad"]
        gate = _deltas(candidate, parent)
        gate["passed"] = bool(
            gate["macro_delta"] >= 0.003
            and gate["nondecreasing_tasks"] >= 5
            and gate["worst_task_delta"] >= -0.010
        )
        detail["gate"] = gate
        winner = "MT1_multitask_pcgrad" if gate["passed"] else "MT0_single_task"
    else:
        # Nested outer-test values are report-only.  Select the FINAL model
        # from inner validation and use a deterministic simplicity tie-break.
        inner_rows = {
            name: collect_inner_validation(name, seed, folds)
            for name in PHASES[phase]
        }
        detail["inner_validation"] = inner_rows
        best_inner = max(item["inner_macro_r2"] for item in inner_rows.values())
        eligible = [
            name for name, item in inner_rows.items()
            if best_inner - item["inner_macro_r2"] < 0.002
        ]
        complexity = {
            "FINAL_geometry_nested5": 0,
            "FINAL_modality_nested5": 1,
            "FINAL_multitask_nested5": 2,
        }
        winner = min(
            eligible,
            key=lambda name: (complexity.get(name, 99), -inner_rows[name]["inner_macro_r2"]),
        )
    detail["winner"] = winner
    state[phase] = detail
    _atomic_json(STATE_PATH, state)
    print(json.dumps(detail, indent=2, sort_keys=True))
    return winner


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("validate", "screen", "summarize", "promote")
    )
    parser.add_argument("--phase", default="G")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--reuse-g0",
        action="store_true",
        help=(
            "strictly reuse the completed G0_canonical_angle20_v1 full-five-fold "
            "result (identity-verified) instead of rerunning it before G1-G4"
        ),
    )
    parser.add_argument(
        "--g0-only",
        action="store_true",
        help=(
            "run only the new canonical Angle-20 G0 parent and exit; "
            "never launches G1-G4"
        ),
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="run eat/fold0 in the isolated non-formal smoke result root",
    )
    args = parser.parse_args(argv)
    if args.g0_only and not (
        args.command == "screen" and args.phase == "G"
    ):
        parser.error("--g0-only is only valid with `screen --phase G`")
    if args.smoke and not (
        args.command == "screen" and args.phase == "G" and args.g0_only
    ):
        parser.error("--smoke is only valid with `screen --phase G --g0-only`")
    if args.smoke and (args.reuse_g0 or tuple(args.folds) != (0,)):
        parser.error("--smoke requires exactly fold 0 and cannot reuse formal G0")
    if args.phase == "F":
        parser.error(
            "F阶段已因性能退化移除；当前固定使用legacy_mts_huber_v1。"
        )
    if args.phase not in PHASES:
        parser.error(
            f"unsupported phase {args.phase!r}; choose one of {', '.join(PHASES)}"
        )
    if args.command in {"screen", "summarize", "promote"} and not args.smoke:
        _require_full_folds(args.folds)
    if args.command in {"screen", "summarize", "promote"} and args.seed != 42:
        parser.error("formal campaign screening and promotion use seed 42")
    names = PHASES[args.phase]
    if args.command == "validate":
        if args.phase == "G":
            _validate_g0_checkpoint()
        for name in names:
            resolved = materialize_config(name)
            environment = os.environ.copy()
            environment.update({
                "CUDA_VISIBLE_DEVICES": "0,1,2",
                "EXPERIMENT_CONFIG": str(resolved),
                "VALIDATE_ONLY": "1",
                "FINETUNE_ONLY": "1",
            })
            subprocess.run(
                ["bash", "scripts/run_mts.sh"],
                cwd=ROOT,
                env=environment,
                check=True,
                stdout=subprocess.DEVNULL,
            )
        print(f"validated {len(names)} phase-{args.phase} configs")
        return 0
    if args.command == "promote":
        promote(args.phase, args.seed, tuple(args.folds))
        return 0
    if args.command == "screen":
        if args.smoke:
            status = run_config(
                G0_CANONICAL,
                (0,),
                args.seed,
                args.dry_run,
                tasks=("eat",),
                output_root=SMOKE_ROOT,
            )
            if status:
                return status
            if not args.dry_run:
                _assert_smoke_identity(SMOKE_ROOT)
            print(f"[mts-sota] isolated smoke passed: {SMOKE_ROOT}")
            return 0

        # The new canonical G0 parent runs (or is strictly reused) before any
        # later geometry proposal.  Historical G0 identities are never read
        # as active inputs and do not gate this run.
        screen_names = names
        if args.phase == "G":
            if args.dry_run:
                screen_names = (G0_CANONICAL,)
            else:
                status = 0
                if args.reuse_g0:
                    print(
                        f"[mts-sota] reusing completed {G0_CANONICAL} "
                        "full-five-fold result",
                        flush=True,
                    )
                    _assert_canonical_g0_identity(
                        args.seed, tuple(args.folds)
                    )
                else:
                    status = run_config(
                        G0_CANONICAL, args.folds, args.seed, False
                    )
                if status:
                    return status
                baseline = collect(
                    G0_CANONICAL, args.seed, tuple(args.folds)
                )
                if args.g0_only:
                    _write_historical_comparison(
                        baseline, ROOT / "results" / "mts_sota_v3" / G0_CANONICAL
                    )
                    return 0
                # G1-G4 are intentionally not launched by this activity cycle;
                # this branch remains available only for a future handoff.
                screen_names = tuple(
                    name for name in names if name != G0_CANONICAL
                )
        for name in screen_names:
            status = run_config(name, args.folds, args.seed, args.dry_run)
            if status:
                return status
        if args.dry_run:
            return 0
    write_summary(names, args.seed, tuple(args.folds))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
