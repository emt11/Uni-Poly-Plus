#!/usr/bin/env python3
"""Run matched O8-only or O8+GLT downstream evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--mode", choices=("o8_only", "o8_glt"), required=True)
    parser.add_argument("--gpu-ids", default="0,1,2,3")
    parser.add_argument(
        "--tasks", nargs="+",
        default=["eat", "eea", "egb", "egc", "ei", "eps", "nc", "xc"],
    )
    parser.add_argument("--folds", nargs="+", default=["0", "1", "2", "3", "4"])
    args = parser.parse_args(argv)
    checkpoint = Path(args.checkpoint).resolve()
    if not checkpoint.is_file():
        raise SystemExit(f"missing MTS-GLT-v1 checkpoint: {checkpoint}")
    sidecar = ROOT / "data/processed/mips_trimer_scage/periodic_line_glt_v1/downstream_union"
    if not (sidecar / ".done").is_file():
        raise SystemExit(f"missing downstream periodic line sidecar: {sidecar}")
    result_root = ROOT / "results/mts_glt_v1/downstream" / args.mode
    log_root = ROOT / "logs/mts_glt_v1/downstream" / args.mode
    command = [
        sys.executable, "scripts/run_mts_finetune_scheduler.py",
        "--python", sys.executable,
        "--gpu-ids", args.gpu_ids,
        "--results-dir", str(result_root),
        "--logs-dir", str(log_root),
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
        "--config_schema", "mts-glt-v1-downstream",
        "--graph_encoder_type", "mips_trimer_scage",
        "--topology_attention_variant", "o8",
        "--no-use_star_rbf", "--no-use_mcl", "--use_md200",
        "--mts_glt_mode", args.mode,
        "--periodic_line_glt_sidecar", str(sidecar),
    ]
    return subprocess.call(command, cwd=ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
