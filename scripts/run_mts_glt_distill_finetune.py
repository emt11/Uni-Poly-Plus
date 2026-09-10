#!/usr/bin/env python3
"""Run one N+ student deployment bundle over the authorized 8x5 protocol."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", choices=("n_plus_2", "n_plus_1"), required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--gpu-ids", default="0,1,2,3")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--tasks", nargs="+", default=["eat", "eea", "egb", "egc", "ei", "eps", "nc", "xc"])
    parser.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--results-dir")
    parser.add_argument("--logs-dir")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    checkpoint = Path(args.checkpoint).resolve()
    if not checkpoint.is_file():
        raise SystemExit(f"missing N+ student checkpoint: {checkpoint}")
    result_root = Path(args.results_dir).resolve() if args.results_dir else ROOT / "results/mts_glt_v2_distill" / args.version / "downstream"
    log_root = Path(args.logs_dir).resolve() if args.logs_dir else ROOT / "logs/mts_glt_v2_distill" / args.version / "downstream"
    command = [
        sys.executable, "scripts/run_mts_finetune_scheduler.py",
        "--python", sys.executable, "--gpu-ids", args.gpu_ids,
        "--results-dir", str(result_root), "--logs-dir", str(log_root),
        "--tasks", *args.tasks,
        "--folds", *map(str, args.folds), "--seeds", "42",
        "--checkpoint", str(checkpoint), "--checkpoint-seed", "42",
        "--checkpoint-tier", "student-20k", "--pretrain-dataset", "PI1M_v2",
        "--finetune-epochs", str(args.epochs), "--finetune-patience", str(args.patience),
        "--batch-size", "32", "--eval-batch-size", "64", "--amp-dtype", "fp32",
        "--loader-workers", "2", "--evaluation-protocol", "historical_shared5",
        "--train-args", "--config_schema", "mts-glt-distill-downstream",
        "--graph_encoder_type", "mips_trimer_scage", "--graph_input", "star_linking",
        "--topology_attention_variant", "o8", "--no-use_star_rbf", "--no-use_mcl", "--use_md200",
        "--mts_glt_version", "distill", "--mts_glt_mode", "o8_only", "--mips_norm_mode", "pre",
        "--target_transform", "recommended", "--regression_loss", "huber", "--huber_beta", "0.5",
        "--max_grad_norm", "1.0", "--weight_decay", "0.02", "--warmup_epochs", "5",
        "--head_dropout", "0.1", "--mts_o8_lr", "1e-5", "--mts_adapter_lr", "1e-5",
        "--graph_lr", "1e-5", "--head_lr", "1e-4",
    ]
    if args.dry_run:
        command.insert(command.index("--train-args"), "--dry-run")
    return subprocess.call(command, cwd=ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
