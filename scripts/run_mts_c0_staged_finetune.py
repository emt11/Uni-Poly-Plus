#!/usr/bin/env python3
"""Run the fixed C0 head-10/full-90 outer5_inner20 campaign."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "results/mts_glt_distill_repair_control/c0/student/student_deploy_020k.pt"


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-ids", default="0,1,2,3")
    parser.add_argument("--tasks", nargs="+", default=["eat", "eea", "egb", "egc", "ei", "eps", "nc", "xc"])
    parser.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--stage1-epochs", type=int, default=10)
    parser.add_argument("--stage2-epochs", type=int, default=90)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--results-dir", default="results/mts_c0_transfer_optimization/staged_finetune")
    parser.add_argument("--logs-dir", default="logs/mts_c0_transfer_optimization/staged_finetune")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if not CHECKPOINT.is_file():
        raise SystemExit(f"missing C0 deployment: {CHECKPOINT}")
    command = [
        sys.executable, "scripts/run_mts_finetune_scheduler.py",
        "--python", sys.executable, "--gpu-ids", args.gpu_ids,
        "--results-dir", str((ROOT / args.results_dir).resolve()),
        "--logs-dir", str((ROOT / args.logs_dir).resolve()),
        "--tasks", *args.tasks, "--folds", *map(str, args.folds), "--seeds", "42",
        "--checkpoint", str(CHECKPOINT), "--checkpoint-seed", "42",
        "--checkpoint-tier", "student-20k", "--pretrain-dataset", "PI1M_v2",
        "--finetune-epochs", str(args.stage1_epochs + args.stage2_epochs),
        "--finetune-patience", str(args.patience),
        "--batch-size", "32", "--eval-batch-size", "64", "--amp-dtype", "fp32",
        "--loader-workers", "2", "--evaluation-protocol", "outer5_inner20",
        "--train-args", "--config_schema", "mts-glt-distill-repair-downstream",
        "--experiment_id", "mts_c0_stageft_outer5_inner20",
        "--split_manifest_dir", "data/splits/mips_outer5_inner20",
        "--graph_encoder_type", "mips_trimer_scage", "--graph_input", "star_linking",
        "--topology_attention_variant", "o8", "--no-use_star_rbf", "--no-use_mcl", "--use_md200",
        "--mts_glt_version", "distill_repair", "--distill_repair_version", "none",
        "--mts_glt_mode", "o8_only", "--mips_norm_mode", "pre",
        "--target_transform", "recommended", "--regression_loss", "huber", "--huber_beta", "0.5",
        "--max_grad_norm", "1.0", "--weight_decay", "0.02", "--warmup_epochs", "5",
        "--head_dropout", "0.1", "--mts_o8_lr", "1e-5", "--mts_adapter_lr", "1e-5",
        "--graph_lr", "1e-5", "--head_lr", "1e-4",
        "--finetune_strategy", "staged_head10",
        "--stage1_epochs", str(args.stage1_epochs), "--stage2_epochs", str(args.stage2_epochs),
    ]
    if args.dry_run:
        command.insert(command.index("--train-args"), "--dry-run")
    return subprocess.call(command, cwd=ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
