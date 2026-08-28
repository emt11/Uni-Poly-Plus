#!/usr/bin/env python3
"""Run matched MTS-GLT-v2 downstream probes or full evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--mode", choices=("o8_only", "o8_glt_atom"),
        required=True,
    )
    parser.add_argument("--layers", type=int, choices=(6,), required=True)
    parser.add_argument(
        "--attention-variant", choices=("mips",), required=True
    )
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--gpu-ids", default="0,1,2,3")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument(
        "--tasks", nargs="+",
        default=["eat", "eea", "egb", "egc", "ei", "eps", "nc", "xc"],
    )
    parser.add_argument("--folds", nargs="+", default=["0", "1", "2", "3", "4"])
    args = parser.parse_args(argv)
    checkpoint = Path(args.checkpoint).resolve()
    if not checkpoint.is_file():
        raise SystemExit(f"missing MTS-GLT-v2 checkpoint: {checkpoint}")
    sidecar = (
        ROOT / "data/processed/mips_trimer_scage/periodic_line_glt_v1/downstream_union"
    )
    if not (sidecar / ".done").is_file():
        raise SystemExit(f"missing downstream periodic line sidecar: {sidecar}")
    result_root = ROOT / "results/mts_glt_v2/downstream" / args.run_name / args.mode
    log_root = ROOT / "logs/mts_glt_v2/downstream" / args.run_name / args.mode
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
        "--checkpoint-tier", "1m",
        "--pretrain-dataset", "PI1M_v2",
        "--finetune-epochs", str(args.epochs),
        "--finetune-patience", str(args.patience),
        "--batch-size", "32",
        "--eval-batch-size", "64",
        "--amp-dtype", "fp32",
        "--loader-workers", "2",
        "--evaluation-protocol", "historical_shared5",
        "--train-args",
        "--config_schema", "mts-glt-v2-downstream",
        "--graph_encoder_type", "mips_trimer_scage",
        "--topology_attention_variant", "o8",
        "--no-use_star_rbf", "--no-use_mcl", "--use_md200",
        "--mts_glt_version", "v2",
        "--mts_glt_layers", str(args.layers),
        "--mts_glt_attention_variant", args.attention_variant,
        "--mts_glt_mode", args.mode,
        "--periodic_line_glt_sidecar", str(sidecar),
        "--target_transform", "recommended",
        "--regression_loss", "huber",
        "--huber_beta", "0.5",
        "--max_grad_norm", "1.0",
        "--weight_decay", "0.02",
        "--warmup_epochs", "5",
        "--head_dropout", "0.25",
        "--mts_o8_lr", "1e-5",
        "--mts_geometry_lr", "1e-5",
        "--mts_adapter_lr", "1e-5",
        "--graph_lr", "1e-5",
        "--fusion_lr", "1e-4",
        "--head_lr", "1e-4",
    ]
    return subprocess.call(command, cwd=ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
