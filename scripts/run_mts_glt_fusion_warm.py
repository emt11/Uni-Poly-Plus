#!/usr/bin/env python3
"""Run the matched MTS-GLT-v1 FusionWarm screening or full expansion."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
ALL_TASKS = ("eat", "eea", "egb", "egc", "ei", "eps", "nc", "xc")
SCREENING_TASKS = ("ei", "nc", "eps")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default="pretrained_models/mts_glt_v1/mts_glt_v1_seed42_final.pth",
    )
    parser.add_argument("--phase", choices=("screening", "full"), default="screening")
    parser.add_argument("--gpu-ids", default="0,1,2,3")
    parser.add_argument("--warm-epochs", type=int, default=5)
    parser.add_argument("--initial-alpha", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args(argv)

    checkpoint = (ROOT / args.checkpoint).resolve()
    if not checkpoint.is_file():
        raise SystemExit(f"missing MTS-GLT-v1 checkpoint: {checkpoint}")
    if not 0 < int(args.warm_epochs) < int(args.epochs):
        raise SystemExit("warm epochs must be positive and smaller than epochs")
    sidecar = (
        ROOT
        / "data/processed/mips_trimer_scage/periodic_line_glt_v1/downstream_union"
    )
    if not (sidecar / ".done").is_file():
        raise SystemExit(f"missing downstream periodic line sidecar: {sidecar}")

    output = ROOT / "results/mts_glt_v1/fusion_warm"
    results = output / "folds"
    logs = ROOT / "logs/mts_glt_v1/fusion_warm"
    tasks = ALL_TASKS if args.phase == "full" else SCREENING_TASKS
    command = [
        sys.executable, "scripts/run_mts_finetune_scheduler.py",
        "--python", sys.executable,
        "--gpu-ids", args.gpu_ids,
        "--results-dir", str(results),
        "--logs-dir", str(logs),
        "--tasks", *tasks,
        "--folds", "0", "1", "2", "3", "4",
        "--seeds", "42",
        "--checkpoint", str(checkpoint),
        "--checkpoint-seed", "42",
        "--pretrain-dataset", "PI1M_v2",
        "--finetune-epochs", str(args.epochs),
        "--finetune-patience", "10",
        "--batch-size", "32",
        "--eval-batch-size", "64",
        "--amp-dtype", "fp32",
        "--loader-workers", "2",
        "--evaluation-protocol", "historical_shared5",
        "--train-args",
        "--experiment_id", "mts_glt_v1_fusion_warm",
        "--config_schema", "mts-glt-v1-downstream",
        "--graph_encoder_type", "mips_trimer_scage",
        "--topology_attention_variant", "o8",
        "--no-use_star_rbf", "--no-use_mcl", "--use_md200",
        "--mts_glt_mode", "o8_glt",
        "--periodic_line_glt_sidecar", str(sidecar),
        "--mts_glt_fusion_strategy", "fusion_warm",
        "--mts_glt_fusion_warm_epochs", str(args.warm_epochs),
        "--mts_glt_initial_alpha", str(args.initial_alpha),
        "--mts_glt_fusion_warm_dir", str(output),
        "--warmup_epochs", "0",
    ]
    status = subprocess.call(command, cwd=ROOT)
    if status:
        return status
    return subprocess.call([
        sys.executable, "scripts/report_mts_glt_fusion_warm.py",
        "--scope", args.phase,
    ], cwd=ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
