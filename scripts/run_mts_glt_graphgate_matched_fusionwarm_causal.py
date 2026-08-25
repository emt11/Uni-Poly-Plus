#!/usr/bin/env python3
"""Run the fixed 24-unit matched Full/Off FusionWarm causal screen."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/mts/glt_graphgate_v1/matched_fusionwarm_causal_v1.json"
OUTPUT = ROOT / "results/mts_glt_graphgate_v1/trimer_validation_v1/matched_5k/fusionwarm_causal_v1"
LOGS = ROOT / "logs/mts_glt_graphgate_v1/trimer_validation_v1/matched_5k/fusionwarm_causal_v1"
SIDECAR = ROOT / "data/processed/mips_trimer_scage/periodic_line_glt_central_v1/downstream_union"


def _checkpoint(path):
    resolved = (ROOT / path).resolve()
    payload = torch.load(resolved, map_location="cpu", weights_only=True)
    if payload.get("schema") != "mts-glt-graphgate-v1-probe-v1":
        raise RuntimeError(f"unexpected checkpoint schema: {resolved}")
    if int(payload.get("step", -1)) != 5000:
        raise RuntimeError(f"matched checkpoint is not step 5000: {resolved}")
    if len(payload.get("namespaces", {})) != 7:
        raise RuntimeError(f"matched checkpoint namespace mismatch: {resolved}")
    return resolved


def _scheduler(*, arm, mode, checkpoint, tasks, folds, gpu_ids, config):
    leaf = "fusionwarm" if mode == "o8_glt_graph" else "o8"
    result_root = OUTPUT / arm / leaf
    log_root = LOGS / arm / leaf
    command = [
        sys.executable, "scripts/run_mts_finetune_scheduler.py",
        "--python", sys.executable,
        "--gpu-ids", gpu_ids,
        "--results-dir", str(result_root),
        "--logs-dir", str(log_root),
        "--tasks", *tasks,
        "--folds", *[str(value) for value in folds],
        "--seeds", "42",
        "--checkpoint", str(checkpoint),
        "--checkpoint-seed", "42",
        "--pretrain-dataset", "PI1M_v2",
        "--finetune-epochs", str(config["epochs"]),
        "--finetune-patience", str(config["patience"]),
        "--batch-size", str(config["batch_size"]),
        "--eval-batch-size", str(config["eval_batch_size"]),
        "--amp-dtype", str(config["precision"]),
        "--loader-workers", str(config["loader_workers"]),
        "--evaluation-protocol", "historical_shared5",
        "--train-args",
        "--experiment_id", f"mts_glt_graphgate_matched_fusionwarm_causal_v1_{arm}",
        "--config_schema", "mts-glt-graphgate-v1-downstream",
        "--graph_encoder_type", "mips_trimer_scage",
        "--topology_attention_variant", "o8",
        "--no-use_star_rbf", "--no-use_mcl", "--use_md200",
        "--mts_glt_version", "graphgate_v1",
        "--mts_glt_layers", "6",
        "--mts_glt_attention_variant", "mips",
        "--mts_glt_mode", mode,
        "--mts_glt_geometry_mode", arm,
        "--periodic_line_glt_sidecar", str(SIDECAR),
        "--graph_lr", str(config["graph_lr"]),
        "--fusion_lr", str(config["fusion_lr"]),
        "--head_lr", str(config["head_lr"]),
        "--regression_loss", str(config["regression_loss"]),
        "--huber_beta", str(config["huber_beta"]),
        "--head_dropout", "0.25",
    ]
    if mode == "o8_glt_graph":
        command.extend([
            "--mts_glt_fusion_strategy", "fusion_warm",
            "--mts_glt_fusion_warm_epochs", str(config["fusion_warm_epochs"]),
            "--mts_glt_initial_alpha", str(config["initial_alpha"]),
            "--mts_glt_fusion_warm_dir", str(result_root),
            "--warmup_epochs", "0",
        ])
    return subprocess.call(command, cwd=ROOT)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-ids", default="0,1,2,3")
    args = parser.parse_args(argv)
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    full = _checkpoint(config["full_checkpoint"])
    off = _checkpoint(config["off_checkpoint"])
    forbidden = (ROOT / "results/mts_glt_graphgate_v1/pretrain/mts_glt_graphgate_probe_005k.pth").resolve()
    if full == forbidden or off == forbidden:
        raise RuntimeError("main-trajectory checkpoint is forbidden in matched causal screen")
    if not (SIDECAR / ".done").is_file():
        raise RuntimeError(f"missing downstream sidecar: {SIDECAR}")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    calls = (
        ("full", "o8_glt_graph", full, ("xc", "ei", "egc"), (0, 1, 2)),
        ("off", "o8_glt_graph", off, ("xc", "ei", "egc"), (0, 1, 2)),
        ("full", "o8_only", full, ("egc",), (0, 1, 2)),
        ("off", "o8_only", off, ("egc",), (0, 1, 2)),
    )
    for arm, mode, checkpoint, tasks, folds in calls:
        status = _scheduler(
            arm=arm, mode=mode, checkpoint=checkpoint, tasks=tasks,
            folds=folds, gpu_ids=args.gpu_ids, config=config,
        )
        if status:
            return status
    return subprocess.call([
        sys.executable,
        "scripts/report_mts_glt_graphgate_matched_fusionwarm_causal.py",
    ], cwd=ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
