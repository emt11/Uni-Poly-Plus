#!/usr/bin/env python3
"""Run the B0-v2 formal 8x5 fine-tuning on four single-GPU slots.

Requires the published final.pth.  Reports one shard CSV + prediction npz per
task/fold under results/mts_b0_periodic_coordinate_denoising_v2/finetune/.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOWNSTREAM_SIDECAR = (
    "data/processed/mips_trimer_scage/star_rbf_v2/"
    "fb6a23c6bc3dc9f85193d7dad6cc30aca57119626778beca10689ffe89e7909c/"
    "downstream_union/ede5f8bc4ba277708c57787249ec85733b1a30a9c149ed9703ec55c80a843bb2"
)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--gpu-ids", default="0,1,2,3")
    parser.add_argument("--tasks", nargs="+",
                        default=["eat", "eea", "egb", "egc", "ei", "eps", "nc", "xc"])
    args = parser.parse_args(argv)
    checkpoint = Path(args.checkpoint).resolve()
    if not checkpoint.is_file():
        raise SystemExit(f"missing final checkpoint: {checkpoint}")
    if not Path(str(checkpoint) + ".complete.json").is_file():
        raise SystemExit(f"missing completion marker: {checkpoint}.complete.json")
    results = ROOT / "results/mts_b0_periodic_coordinate_denoising_v2/finetune"
    logs = ROOT / "logs/mts_b0_v2_finetune"
    command = [
        sys.executable, "scripts/run_mts_finetune_scheduler.py",
        "--python", sys.executable,
        "--gpu-ids", args.gpu_ids,
        "--results-dir", str(results),
        "--logs-dir", str(logs),
        "--tasks", *args.tasks,
        "--folds", "0", "1", "2", "3", "4",
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
        "--config_schema", "mts-config-v3",
        "--graph_encoder_type", "mips_trimer_scage",
        "--topology_attention_variant", "o8",
        "--use_star_rbf", "--no-use_mcl", "--use_md200",
        "--star_rbf_upper", "3.75",
        "--star_rbf_v2_sidecar", str(ROOT / DOWNSTREAM_SIDECAR),
    ]
    return subprocess.call(command, cwd=ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
