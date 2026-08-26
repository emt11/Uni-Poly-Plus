#!/usr/bin/env python3
"""Run the schedule-matched GLT-v2 Warm5 arm and report against formal Warm0."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/mts/glt_v2_fusionwarm_vs_nowarm_formal5k_v1.json"


def _command(config, *, gpu_ids, smoke):
    checkpoint = (ROOT / config["checkpoint"]).resolve()
    sidecar = (ROOT / config["sidecar"]).resolve()
    output = (ROOT / config["output_root"]).resolve()
    logs = (ROOT / config["log_root"]).resolve()
    if not checkpoint.is_file():
        raise SystemExit(f"missing selected 5k checkpoint: {checkpoint}")
    if not (sidecar / ".done").is_file():
        raise SystemExit(f"missing frozen downstream sidecar: {sidecar}")

    if smoke:
        tasks, folds = ["eat"], ["0"]
        epochs, warm_epochs = 6, 5
        results_root = output / "smoke" / "Warm5"
        logs_root = logs / "smoke" / "Warm5"
    else:
        tasks = list(config["tasks"])
        folds = [str(value) for value in config["folds"]]
        epochs, warm_epochs = int(config["epochs"]), int(config["warm_epochs"])
        results_root = output / "Warm5"
        logs_root = logs / "Warm5"

    return [
        sys.executable, "scripts/run_mts_finetune_scheduler.py",
        "--python", sys.executable,
        "--gpu-ids", gpu_ids,
        "--results-dir", str(results_root),
        "--logs-dir", str(logs_root),
        "--tasks", *tasks,
        "--folds", *folds,
        "--seeds", str(config["seed"]),
        "--checkpoint", str(checkpoint),
        "--checkpoint-seed", str(config["seed"]),
        "--pretrain-dataset", "PI1M_v2",
        "--finetune-epochs", str(epochs),
        "--finetune-patience", str(config["patience"]),
        "--batch-size", str(config["train_batch_size"]),
        "--eval-batch-size", str(config["eval_batch_size"]),
        "--amp-dtype", config["precision"],
        "--loader-workers", str(config["workers"]),
        "--evaluation-protocol", config["evaluation_protocol"],
        "--train-args",
        "--experiment_id", config["experiment"] + ("_smoke" if smoke else ""),
        "--config_schema", "mts-glt-v2-downstream",
        "--graph_encoder_type", "mips_trimer_scage",
        "--topology_attention_variant", "o8",
        "--no-use_star_rbf", "--no-use_mcl", "--use_md200",
        "--mts_glt_version", "v2",
        "--mts_glt_layers", str(config["glt_layers"]),
        "--mts_glt_attention_variant", config["glt_attention_variant"],
        "--mts_glt_mode", config["glt_mode"],
        "--periodic_line_glt_sidecar", str(sidecar),
        "--mts_glt_fusion_strategy", "fusion_warm",
        "--mts_glt_fusion_warm_epochs", str(warm_epochs),
        "--mts_glt_initial_alpha", str(config["initial_alpha"]),
        "--mts_glt_fusion_stage2_trainability", "joint",
        "--mts_glt_fusion_warm_dir", str(results_root),
        "--graph_lr", str(config["graph_lr"]),
        "--fusion_lr", str(config["fusion_lr"]),
        "--head_lr", str(config["head_lr"]),
        "--warmup_epochs", str(config["lr_warmup_epochs"]),
        "--regression_loss", config["loss"],
        "--huber_beta", str(config["huber_beta"]),
        "--head_dropout", str(config["head_dropout"]),
        "--weight_decay", str(config["weight_decay"]),
    ]


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-ids", default="0,1,2,3")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args(argv)
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    if not args.report_only:
        status = subprocess.call(
            _command(config, gpu_ids=args.gpu_ids, smoke=args.smoke), cwd=ROOT
        )
        if status:
            return status
    if args.smoke:
        return 0
    return subprocess.call([
        sys.executable,
        "scripts/report_mts_glt_v2_fusionwarm_vs_nowarm_formal5k.py",
    ], cwd=ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
