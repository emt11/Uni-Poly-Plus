#!/usr/bin/env python3
"""Run the nine Warm0 units for the GraphGate schedule ablation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/mts/glt_graphgate_v1/fusionwarm_vs_nowarm_20k_v1.json"
OUTPUT = ROOT / "results/mts_glt_graphgate_v1/fusionwarm_vs_nowarm_20k"
LOGS = ROOT / "logs/mts_glt_graphgate_v1/fusionwarm_vs_nowarm_20k"


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-ids", default="0,1,2,3")
    args = parser.parse_args(argv)
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    checkpoint = (ROOT / config["checkpoint"]).resolve()
    sidecar = ROOT / "data/processed/mips_trimer_scage/periodic_line_glt_central_v1/downstream_union"
    if not checkpoint.is_file():
        raise SystemExit(f"missing 20k checkpoint: {checkpoint}")
    if not (sidecar / ".done").is_file():
        raise SystemExit(f"missing GraphGate downstream sidecar: {sidecar}")

    command = [
        sys.executable, "scripts/run_mts_finetune_scheduler.py",
        "--python", sys.executable,
        "--gpu-ids", args.gpu_ids,
        "--results-dir", str(OUTPUT / "Warm0"),
        "--logs-dir", str(LOGS / "Warm0"),
        "--tasks", *config["tasks"],
        "--folds", *(str(value) for value in config["folds"]),
        "--seeds", str(config["seed"]),
        "--checkpoint", str(checkpoint),
        "--checkpoint-seed", str(config["seed"]),
        "--pretrain-dataset", "PI1M_v2",
        "--finetune-epochs", str(config["epochs"]),
        "--finetune-patience", str(config["patience"]),
        "--batch-size", str(config["train_batch_size"]),
        "--eval-batch-size", str(config["eval_batch_size"]),
        "--amp-dtype", config["precision"],
        "--loader-workers", str(config["workers"]),
        "--evaluation-protocol", config["evaluation_protocol"],
        "--train-args",
        "--experiment_id", config["experiment"],
        "--config_schema", config["schema"],
        "--graph_encoder_type", "mips_trimer_scage",
        "--topology_attention_variant", "o8",
        "--no-use_star_rbf", "--no-use_mcl", "--use_md200",
        "--mts_glt_version", "graphgate_v1",
        "--mts_glt_layers", "6",
        "--mts_glt_attention_variant", "mips",
        "--mts_glt_mode", "o8_glt_graph",
        "--mts_glt_geometry_mode", "full",
        "--periodic_line_glt_sidecar", str(sidecar),
        "--mts_glt_fusion_strategy", "fusion_warm",
        "--mts_glt_fusion_warm_epochs", str(config["warm0_epochs"]),
        "--mts_glt_initial_alpha", str(config["initial_alpha"]),
        "--mts_glt_fusion_stage2_trainability", "joint",
        "--mts_glt_fusion_warm_dir", str(OUTPUT / "Warm0"),
        "--fusion_lr", str(config["fusion_head_lr"]),
        "--head_lr", str(config["fusion_head_lr"]),
        "--warmup_epochs", "0",
        "--regression_loss", config["loss"],
        "--huber_beta", str(config["huber_beta"]),
        "--head_dropout", "0.25",
    ]
    status = subprocess.call(command, cwd=ROOT)
    if status:
        return status
    return subprocess.call([
        sys.executable,
        "scripts/report_mts_glt_graphgate_fusionwarm_vs_nowarm_20k.py",
    ], cwd=ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
