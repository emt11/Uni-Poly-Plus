#!/usr/bin/env python3
"""Run the two new GraphGate readout arms for the causal screen."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/mts_glt_graphgate_v1/readout_causal_screen_v1"
LOGS = ROOT / "logs/mts_glt_graphgate_v1/readout_causal_screen_v1"
CHECKPOINT = ROOT / "pretrained_models/mts_glt_graphgate_v1/mts_glt_graphgate_v1.pth"
SIDECAR = ROOT / "data/processed/mips_trimer_scage/periodic_line_glt_central_v1/downstream_union"
TASKS = ("xc", "ei", "eea")
MODES = {
    "GM": "o8_glt_graph_mean",
    "AT": "o8_glt_atom_central",
}


def _command(arm, mode, gpu_ids):
    result_root = OUTPUT / arm
    log_root = LOGS / arm
    return [
        sys.executable, "scripts/run_mts_finetune_scheduler.py",
        "--python", sys.executable,
        "--gpu-ids", gpu_ids,
        "--results-dir", str(result_root),
        "--logs-dir", str(log_root),
        "--tasks", *TASKS,
        "--folds", "0", "1", "2",
        "--seeds", "42",
        "--checkpoint", str(CHECKPOINT),
        "--checkpoint-seed", "42",
        "--pretrain-dataset", "PI1M_v2",
        "--finetune-epochs", "100",
        "--finetune-patience", "10",
        "--batch-size", "32",
        "--eval-batch-size", "64",
        "--amp-dtype", "fp32",
        "--loader-workers", "2",
        "--evaluation-protocol", "historical_shared5",
        "--train-args",
        "--experiment_id", f"mts_glt_graphgate_readout_screen_{arm.lower()}_v1",
        "--config_schema", "mts-glt-graphgate-v1-downstream",
        "--graph_encoder_type", "mips_trimer_scage",
        "--topology_attention_variant", "o8",
        "--no-use_star_rbf", "--no-use_mcl", "--use_md200",
        "--mts_glt_version", "graphgate_v1",
        "--mts_glt_layers", "6",
        "--mts_glt_attention_variant", "mips",
        "--mts_glt_mode", mode,
        "--mts_glt_geometry_mode", "full",
        "--periodic_line_glt_sidecar", str(SIDECAR),
        "--mts_glt_fusion_strategy", "fusion_warm",
        "--mts_glt_fusion_warm_epochs", "5",
        "--mts_glt_initial_alpha", "0.05",
        "--mts_glt_fusion_warm_dir", str(result_root),
        "--graph_lr", "1e-5",
        "--fusion_lr", "1e-4",
        "--head_lr", "1e-4",
        "--warmup_epochs", "0",
        "--regression_loss", "huber",
        "--huber_beta", "0.5",
        "--head_dropout", "0.25",
    ]


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-ids", default="0,1,2,3")
    args = parser.parse_args(argv)
    if not CHECKPOINT.is_file():
        raise SystemExit(f"missing checkpoint: {CHECKPOINT}")
    if not (SIDECAR / ".done").is_file():
        raise SystemExit(f"missing downstream sidecar: {SIDECAR}")
    for arm, mode in MODES.items():
        status = subprocess.call(_command(arm, mode, args.gpu_ids), cwd=ROOT)
        if status:
            return status
    return subprocess.call([
        sys.executable,
        "scripts/report_mts_glt_graphgate_readout_causal_screen.py",
    ], cwd=ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
