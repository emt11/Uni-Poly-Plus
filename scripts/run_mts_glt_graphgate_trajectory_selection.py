#!/usr/bin/env python3
"""Run the gated 5k FusionWarm trajectory-selection experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/mts_glt_graphgate_v1/trajectory_selection_v1"
LOGS = ROOT / "logs/mts_glt_graphgate_v1/trajectory_selection_v1"
CHECKPOINT = ROOT / "results/mts_glt_graphgate_v1/pretrain/mts_glt_graphgate_probe_005k.pth"
SIDECAR = ROOT / "data/processed/mips_trimer_scage/periodic_line_glt_central_v1/downstream_union"
SCREEN_TASKS = ("xc", "ei", "egc")
OTHER_TASKS = ("eat", "eea", "egb", "eps", "nc")


def _scheduler(mode, tasks, folds, gpu_ids):
    result_root = OUTPUT / ("fusion_warm_5k" if mode == "o8_glt_graph" else "o8_only_5k")
    log_root = LOGS / ("fusion_warm_5k" if mode == "o8_glt_graph" else "o8_only_5k")
    command = [
        sys.executable, "scripts/run_mts_finetune_scheduler.py",
        "--python", sys.executable, "--gpu-ids", gpu_ids,
        "--results-dir", str(result_root), "--logs-dir", str(log_root),
        "--tasks", *tasks, "--folds", *map(str, folds), "--seeds", "42",
        "--checkpoint", str(CHECKPOINT), "--checkpoint-seed", "42",
        "--pretrain-dataset", "PI1M_v2", "--finetune-epochs", "100",
        "--finetune-patience", "10", "--batch-size", "32",
        "--eval-batch-size", "64", "--amp-dtype", "fp32",
        "--loader-workers", "2", "--evaluation-protocol", "historical_shared5",
        "--train-args", "--experiment_id", "mts_glt_graphgate_trajectory_selection_v1",
        "--config_schema", "mts-glt-graphgate-v1-downstream",
        "--graph_encoder_type", "mips_trimer_scage",
        "--topology_attention_variant", "o8", "--no-use_star_rbf",
        "--no-use_mcl", "--use_md200", "--mts_glt_version", "graphgate_v1",
        "--mts_glt_layers", "6", "--mts_glt_attention_variant", "mips",
        "--mts_glt_mode", mode, "--mts_glt_geometry_mode", "full",
        "--periodic_line_glt_sidecar", str(SIDECAR),
        "--graph_lr", "1e-5", "--fusion_lr", "1e-4", "--head_lr", "1e-4",
        "--warmup_epochs", "0", "--regression_loss", "huber",
        "--huber_beta", "0.5", "--head_dropout", "0.25",
    ]
    if mode == "o8_glt_graph":
        command.extend([
            "--mts_glt_fusion_strategy", "fusion_warm",
            "--mts_glt_fusion_warm_epochs", "5",
            "--mts_glt_initial_alpha", "0.05",
            "--mts_glt_fusion_warm_dir", str(result_root),
        ])
    return subprocess.call(command, cwd=ROOT)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("screen5k", "formal5k"), required=True)
    parser.add_argument("--gpu-ids", default="0,1,2,3")
    args = parser.parse_args(argv)
    if not CHECKPOINT.is_file() or not (SIDECAR / ".done").is_file():
        raise SystemExit("5k checkpoint or GraphGate downstream sidecar is missing")

    if args.phase == "screen5k":
        status = _scheduler("o8_glt_graph", SCREEN_TASKS, (0, 1, 2), args.gpu_ids)
    else:
        decision_path = OUTPUT / "screen5k_decision.json"
        if not decision_path.is_file():
            raise SystemExit("screen5k decision is missing")
        decision = json.loads(decision_path.read_text(encoding="utf-8"))
        if not bool(decision.get("run_formal5k", False)):
            raise SystemExit("screen5k GO conditions failed")
        calls = (
            ("o8_glt_graph", SCREEN_TASKS, (3, 4)),
            ("o8_glt_graph", OTHER_TASKS, range(5)),
            ("o8_only", SCREEN_TASKS, (3, 4)),
            ("o8_only", OTHER_TASKS, range(5)),
        )
        status = 0
        for mode, tasks, folds in calls:
            status = _scheduler(mode, tasks, folds, args.gpu_ids)
            if status:
                break
    if status:
        return status
    return subprocess.call([
        sys.executable,
        "scripts/report_mts_glt_graphgate_trajectory_selection.py",
        "--scope", args.phase,
    ], cwd=ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
