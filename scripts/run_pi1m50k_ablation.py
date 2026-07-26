#!/usr/bin/env python3
"""Run resumable SCAGE-PolyGen pretext and downstream ablations."""

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import pandas as pd


ALL_TASKS = ("eat", "eea", "egb", "egc", "ei", "eps", "nc", "xc")
PRETEXT_CANDIDATES = {
    "p0_base": {
        "STAGE1_ECFP_WEIGHT": "0.0",
        "STAGE1_PERIODIC_CONTRAST_WEIGHT": "0.0",
    },
    "p1_perio": {
        "STAGE1_ECFP_WEIGHT": "0.0",
        "STAGE1_PERIODIC_CONTRAST_WEIGHT": "0.25",
        "PERIODIC_AUG_VIEWS": "1",
    },
    "p2_ecfp": {
        "STAGE1_ECFP_WEIGHT": "0.25",
        "STAGE1_ECFP_CAP": "0.15",
        "STAGE1_PERIODIC_CONTRAST_WEIGHT": "0.0",
    },
    "p3_perio_ecfp": {
        "STAGE1_ECFP_WEIGHT": "0.25",
        "STAGE1_ECFP_CAP": "0.15",
        "STAGE1_PERIODIC_CONTRAST_WEIGHT": "0.25",
        "PERIODIC_AUG_VIEWS": "1",
    },
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-hours", type=float, default=24.0)
    parser.add_argument("--state", default="results/scage_optimization_state.json")
    parser.add_argument(
        "--baseline", default="results/scage_polygen_ecfp_quality_pi1m50k.csv"
    )
    parser.add_argument("--best-target", default="results/best_result.csv")
    parser.add_argument("--pretrain-dataset", default="PI1M_50k")
    parser.add_argument("--pretrain-nproc", type=int, default=4)
    parser.add_argument("--cache-workers", type=int, default=16)
    parser.add_argument("--screen-tasks", nargs="+", default=["ei", "xc", "egb", "egc"])
    parser.add_argument("--screen-folds", nargs="+", default=["0", "1"])
    parser.add_argument(
        "--candidates", nargs="+", choices=tuple(PRETEXT_CANDIDATES),
        default=list(PRETEXT_CANDIDATES),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--reset-deadline", action="store_true",
        help="Start a fresh max-hours window while preserving completed experiments.",
    )
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--skip-pretraining", action="store_true")
    parser.add_argument("--include-pbc-ablations", action="store_true")
    return parser.parse_args()


def load_state(path, args):
    if path.exists():
        state = json.loads(path.read_text(encoding="utf-8"))
        if state.get("pretrain_dataset") != args.pretrain_dataset:
            raise ValueError("State file belongs to a different pretraining dataset")
        return state
    return {
        "version": 2,
        "pretrain_dataset": args.pretrain_dataset,
        "started_at": time.time(),
        "completed": {},
        "screen_scores": {},
    }


def save_state(path, state):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")


def run_stage(env, dry_run):
    command = ["bash", "scripts/run.sh"]
    printable = " ".join(f"{key}={value}" for key, value in sorted(env.items()))
    print(f"[optimization] {printable} {' '.join(command)}", flush=True)
    if not dry_run:
        subprocess.run(command, check=True, env={**os.environ, **env})


def pretrain_tag(dataset):
    mapping = {"PI1M_50k": "pi1m50k", "PI1M_20k": "pi1m20k", "PI1M_v2": "pi1m"}
    return mapping.get(dataset, dataset.lower().replace("_", ""))


def artifact_prefix(dataset, experiment):
    return f"scage_polygen_ecfp_quality_{pretrain_tag(dataset)}_{experiment}_v4"


def result_mean(path, required_tasks):
    frame = pd.read_csv(path)
    task_rows = frame.set_index("task")
    missing = sorted(set(required_tasks).difference(task_rows.index))
    if missing:
        raise ValueError(f"{path} is missing tasks: {missing}")
    return float(task_rows.loc[list(required_tasks), "avg_test_r2"].astype(float).mean())


def within_deadline(deadline):
    return time.time() < deadline


def common_env(args):
    return {
        "BASELINE": "scage_parallel",
        "PRETRAIN_DATASET": args.pretrain_dataset,
        "CONFORMER_PROFILE": "quality",
        "FP_MODE": "ecfp",
        "PRETRAIN_NPROC": str(args.pretrain_nproc),
        "CACHE_WORKERS": str(args.cache_workers),
        "DATALOADER_WORKERS": "0",
        "MODEL_VERSION": "v4",
    }


def run_pretraining(args, state, state_path, candidate, deadline):
    key = f"pretrain:{candidate}"
    prefix = artifact_prefix(args.pretrain_dataset, candidate)
    graph_checkpoint = Path(f"pretrained_models/{prefix}_graph_geom.pth")
    alignment_checkpoint = Path(f"pretrained_models/{prefix}_alignment.pth")
    if key in state["completed"]:
        return graph_checkpoint, alignment_checkpoint
    if args.skip_pretraining:
        if not graph_checkpoint.is_file() or not alignment_checkpoint.is_file():
            raise FileNotFoundError(f"Missing checkpoints for {candidate}: {prefix}")
        return graph_checkpoint, alignment_checkpoint
    if not within_deadline(deadline):
        return None, None
    run_stage({
        **common_env(args),
        **PRETEXT_CANDIDATES[candidate],
        "EXPERIMENT_TAG": candidate,
        "PRETRAIN_ONLY": "1",
        "REBUILD_FEATURE_CACHE": "1" if args.rebuild_cache else "0",
        "GRAPH_GEOM_EPOCHS": "10",
        "ALIGN_EPOCHS": "10",
    }, args.dry_run)
    if not args.dry_run:
        if not graph_checkpoint.is_file() or not alignment_checkpoint.is_file():
            raise FileNotFoundError(f"Pretraining did not create {prefix} checkpoints")
        state["completed"][key] = {
            "time": time.time(),
            "graph_checkpoint": str(graph_checkpoint),
            "alignment_checkpoint": str(alignment_checkpoint),
        }
        save_state(state_path, state)
    return graph_checkpoint, alignment_checkpoint


def run_downstream(args, state, state_path, candidate, graph_checkpoint,
                   alignment_checkpoint, tasks, folds, suffix, overrides, deadline):
    experiment = f"{candidate}_{suffix}"
    key = f"downstream:{experiment}"
    result_path = Path(f"results/{artifact_prefix(args.pretrain_dataset, experiment)}.csv")
    if key not in state["completed"] and within_deadline(deadline):
        run_stage({
            **common_env(args),
            **overrides,
            "EXPERIMENT_TAG": experiment,
            "STAGE3_ONLY": "1",
            "GRAPH_GEOM_CHECKPOINT": str(graph_checkpoint),
            "ALIGN_CHECKPOINT": str(alignment_checkpoint),
            "TASKS": " ".join(tasks),
            "FOLD_IDS": " ".join(folds),
            "TRAIN_EPOCHS": "100",
        }, args.dry_run)
        if not args.dry_run:
            if not result_path.is_file():
                raise FileNotFoundError(result_path)
            state["completed"][key] = {"time": time.time(), "result": str(result_path)}
            save_state(state_path, state)
    return result_path


def compare_full(args, candidate_path):
    report_path = candidate_path.with_suffix(".comparison.json")
    subprocess.run([
        "python", "scripts/compare_experiment_results.py",
        "--baseline", args.baseline,
        "--candidate", str(candidate_path),
        "--best-target", args.best_target,
        "--min-mean-r2-gain", "0.0",
        "--first-stage-mean-r2", "0.829",
        "--max-task-r2-drop", "0.01",
        "--output", str(report_path),
    ], check=True)
    return json.loads(report_path.read_text(encoding="utf-8"))


def main():
    args = parse_args()
    state_path = Path(args.state)
    state = load_state(state_path, args)
    if args.reset_deadline:
        state["started_at"] = time.time()
        save_state(state_path, state)
    deadline = float(state["started_at"]) + args.max_hours * 3600.0

    for candidate in args.candidates:
        graph_checkpoint, alignment_checkpoint = run_pretraining(
            args, state, state_path, candidate, deadline
        )
        if graph_checkpoint is None:
            break
        screen_result = run_downstream(
            args, state, state_path, candidate, graph_checkpoint, alignment_checkpoint,
            args.screen_tasks, args.screen_folds, "screen", {}, deadline,
        )
        if screen_result.is_file():
            state["screen_scores"][candidate] = result_mean(screen_result, args.screen_tasks)
            save_state(state_path, state)
        if not within_deadline(deadline):
            break

    if not state["screen_scores"]:
        print(json.dumps(state, indent=2))
        return

    winner = max(state["screen_scores"], key=state["screen_scores"].get)
    graph_checkpoint = Path(f"pretrained_models/{artifact_prefix(args.pretrain_dataset, winner)}_graph_geom.pth")
    alignment_checkpoint = Path(f"pretrained_models/{artifact_prefix(args.pretrain_dataset, winner)}_alignment.pth")
    full_result = run_downstream(
        args, state, state_path, winner, graph_checkpoint, alignment_checkpoint,
        ALL_TASKS, [str(index) for index in range(5)], "full", {}, deadline,
    )
    if full_result.is_file():
        report = compare_full(args, full_result)
        state["winner"] = winner
        state["winner_result"] = str(full_result)
        state["winner_mean_r2"] = report["candidate_mean_r2"]
        state["first_stage_met"] = report["first_stage_met"]
        state["stop_condition_met"] = report["stop_condition_met"]

    if args.include_pbc_ablations and within_deadline(deadline):
        pbc_modes = {
            "topology_only": {"SCAGE_FORCE_TOPOLOGY_ONLY": "1"},
            "no_periodic_images": {"SCAGE_USE_PBC_DISTANCE": "0"},
        }
        state.setdefault("pbc_scores", {})
        for name, overrides in pbc_modes.items():
            path = run_downstream(
                args, state, state_path, winner, graph_checkpoint, alignment_checkpoint,
                args.screen_tasks, args.screen_folds, name, overrides, deadline,
            )
            if path.is_file():
                state["pbc_scores"][name] = result_mean(path, args.screen_tasks)

    save_state(state_path, state)
    print(json.dumps(state, indent=2))


if __name__ == "__main__":
    main()
