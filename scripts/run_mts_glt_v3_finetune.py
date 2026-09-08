#!/usr/bin/env python3
"""Launch authorized MTS-GLT-v3 Galformer or MIPS-concat downstream folds."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--mode", choices=("galformer", "mips_concat"), default="galformer")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--gpu-ids", default="0,1,2,3")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--tasks", nargs="+", default=["eat", "eea", "egb", "egc", "ei", "eps", "nc", "xc"])
    parser.add_argument("--folds", nargs="+", default=["0", "1", "2", "3", "4"])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    checkpoint = Path(args.checkpoint).resolve()
    if not checkpoint.is_file():
        raise SystemExit(f"missing MTS-GLT-v3 checkpoint: {checkpoint}")
    sidecar = ROOT / "data/processed/mips_trimer_scage/periodic_line_glt_image_v1/downstream_union"
    if args.mode == "mips_concat" and not (sidecar / ".done").is_file():
        raise SystemExit(f"missing downstream image sidecar: {sidecar}")
    result_root = ROOT / "results/mts_glt_v3_galformer_20k/downstream" / args.run_name / args.mode
    log_root = ROOT / "logs/mts_glt_v3_galformer_20k/downstream" / args.run_name / args.mode
    command = [
        sys.executable, "scripts/run_mts_finetune_scheduler.py",
        "--python", sys.executable, "--gpu-ids", args.gpu_ids,
        "--results-dir", str(result_root), "--logs-dir", str(log_root),
        "--tasks", *args.tasks, "--folds", *args.folds, "--seeds", "42",
        "--checkpoint", str(checkpoint), "--checkpoint-seed", "42",
        "--checkpoint-tier", "v3-20k", "--pretrain-dataset", "PI1M_v2",
        "--finetune-epochs", str(args.epochs), "--finetune-patience", str(args.patience),
        "--batch-size", "32", "--eval-batch-size", "64", "--amp-dtype", "fp32",
        "--loader-workers", "2", "--evaluation-protocol", "historical_shared5",
        *( ["--dry-run"] if args.dry_run else [] ),
        "--train-args", "--config_schema", "mts-glt-v3-downstream",
        "--graph_encoder_type", "mips_trimer_scage", "--graph_input", "star_linking",
        "--topology_attention_variant", "o8", "--no-use_star_rbf", "--no-use_mcl", "--use_md200",
        "--mts_glt_version", "v3", "--glt_readout_mode", args.mode,
        "--periodic_line_glt_sidecar", str(sidecar) if args.mode == "mips_concat" else "",
        "--target_transform", "recommended", "--regression_loss", "huber", "--huber_beta", "0.5",
        "--max_grad_norm", "1.0", "--weight_decay", "0.02", "--warmup_epochs", "5",
        "--head_dropout", "0.25", "--mts_o8_lr", "1e-5", "--mts_geometry_lr", "1e-5",
        "--mts_adapter_lr", "1e-5", "--graph_lr", "1e-5", "--fusion_lr", "1e-4", "--head_lr", "1e-4",
    ]
    return subprocess.call(command, cwd=ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
