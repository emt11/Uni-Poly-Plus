#!/usr/bin/env python3
"""Rerun matched O8+GLT folds with isolated fusion diagnostics enabled."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
TASKS = ("eat", "eea", "egb", "egc", "ei", "eps", "nc", "xc")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default="pretrained_models/mts_glt_v1/mts_glt_v1_seed42_final.pth",
    )
    parser.add_argument("--gpu-ids", default="0,1,2,3")
    parser.add_argument("--tasks", nargs="+", default=list(TASKS))
    parser.add_argument("--folds", nargs="+", default=["0", "1", "2", "3", "4"])
    args = parser.parse_args(argv)
    checkpoint = (ROOT / args.checkpoint).resolve()
    if not checkpoint.is_file():
        raise SystemExit(f"missing checkpoint: {checkpoint}")
    output = ROOT / "results/mts_glt_v1/postmortem"
    results = output / "fusion_rerun"
    logs = ROOT / "logs/mts_glt_v1/postmortem/fusion_rerun"
    sidecar = ROOT / "data/processed/mips_trimer_scage/periodic_line_glt_v1/downstream_union"
    command = [
        sys.executable, "scripts/run_mts_finetune_scheduler.py",
        "--python", sys.executable,
        "--gpu-ids", args.gpu_ids,
        "--results-dir", str(results),
        "--logs-dir", str(logs),
        "--tasks", *args.tasks,
        "--folds", *args.folds,
        "--seeds", "42",
        "--checkpoint", str(checkpoint),
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
        "--experiment_id", "mts_glt_v1_postmortem_fusion_audit",
        "--config_schema", "mts-glt-v1-downstream",
        "--graph_encoder_type", "mips_trimer_scage",
        "--topology_attention_variant", "o8",
        "--no-use_star_rbf", "--no-use_mcl", "--use_md200",
        "--mts_glt_mode", "o8_glt",
        "--periodic_line_glt_sidecar", str(sidecar),
        "--mts_glt_postmortem_dir", str(output),
    ]
    return subprocess.call(command, cwd=ROOT)


if __name__ == "__main__":
    raise SystemExit(main())

