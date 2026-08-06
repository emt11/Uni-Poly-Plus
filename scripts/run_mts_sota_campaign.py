#!/usr/bin/env python3
"""DAG-aware launcher and screener for the bounded MTS SOTA experiments."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TASKS = ("eat", "eea", "egb", "egc", "ei", "eps", "nc", "xc")
PHASES = {
    "G": (
        "G0_current_mcl", "G1_mcl_disabled", "G2_coordinate_shuffled",
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
STATE_PATH = ROOT / "results" / "mts_sota_v3" / "promotion_state.json"
STATE_SCHEMA = "mts-sota-promotion-v3"
BASE_PROFILE = "legacy_mts_huber_v1"
BASE_REFERENCE_MACRO_R2 = 0.8423414007


def template_config_path(name):
    return ROOT / "configs" / "mts" / "experiments" / f"{name}.json"


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


def run_config(name, folds, seed, dry_run=False):
    output = ROOT / "results" / "mts_sota_v3" / name
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
        # The immutable joint checkpoint was metadata-migrated after the
        # geometry-identity hash correction.  Campaign candidates must use
        # that explicit artifact rather than falling back to the pre-fix path.
        "JOINT_CKPT": str(
            ROOT / "pretrained_models/mts/"
            "mts_joint_pretraining_pi1m_v2_seed42_migrated_v3.pth"
        ),
        "TASKS": " ".join(TASKS),
        "FOLD_IDS": " ".join(str(value) for value in folds),
        "RESULTS_DIR": str(output),
    })
    command = ["bash", "scripts/run_mts.sh"]
    print("[mts-sota]", name, " ".join(command), flush=True)
    if dry_run:
        return 0
    return subprocess.call(command, cwd=ROOT, env=env)


def collect(name, seed=42, folds=(0, 1, 2, 3, 4)):
    root = ROOT / "results" / "mts_sota_v3" / name / "shards" / str(seed)
    values = {}
    controls = []
    for task in TASKS:
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
    detail = {"seed": int(seed), "folds": list(folds), "candidates": rows}
    if phase == "G":
        parent, candidate = rows["G0_current_mcl"], rows["G3_mcl_rbf"]
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
        winner = "G3_mcl_rbf" if gate["passed"] else "G0_current_mcl"
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
        "--accept-g0-drift",
        action="store_true",
        help=(
            "continue G1-G4 after a small G0-vs-legacy reproduction drift; "
            "the drift report remains recorded and G promotion gates are unchanged"
        ),
    )
    parser.add_argument(
        "--reuse-g0",
        action="store_true",
        help=(
            "reuse the already completed G0 full-five-fold result instead of "
            "rerunning it before G1-G4"
        ),
    )
    args = parser.parse_args(argv)
    if args.phase == "F":
        parser.error(
            "F阶段已因性能退化移除；当前固定使用legacy_mts_huber_v1。"
        )
    if args.phase not in PHASES:
        parser.error(
            f"unsupported phase {args.phase!r}; choose one of {', '.join(PHASES)}"
        )
    if args.command in {"screen", "summarize", "promote"}:
        _require_full_folds(args.folds)
    if args.command in {"screen", "summarize", "promote"} and args.seed != 42:
        parser.error("formal campaign screening and promotion use seed 42")
    names = PHASES[args.phase]
    if args.command == "validate":
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
        # Reproduce the five-fold legacy anchor before launching any geometry
        # proposal.  A drifted G0 is a correctness failure, not a candidate
        # to be promoted or hidden by a later ablation.
        screen_names = names
        if args.phase == "G" and not args.dry_run:
            status = 0
            if not args.reuse_g0:
                status = run_config("G0_current_mcl", args.folds, args.seed, False)
            else:
                print(
                    "[mts-sota] reusing completed G0_current_mcl full-five-fold result",
                    flush=True,
                )
            if status:
                return status
            baseline = collect("G0_current_mcl", args.seed, tuple(args.folds))
            reproduction = _check_g0_reproduction(baseline)
            _atomic_json(
                ROOT / "results" / "mts_sota_v3" / "G0_current_mcl" /
                "baseline_reproduction.json",
                reproduction,
            )
            if not reproduction["passed"] and not args.accept_g0_drift:
                raise RuntimeError(
                    "G0 legacy baseline reproduction failed; "
                    "G1-G4 were not launched: " + json.dumps(reproduction, sort_keys=True)
                )
            if not reproduction["passed"] and args.accept_g0_drift:
                print(
                    "[mts-sota] accepting the recorded G0 reproduction drift "
                    "by explicit user override; continuing with G1-G4",
                    flush=True,
                )
            screen_names = tuple(name for name in names if name != "G0_current_mcl")
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
