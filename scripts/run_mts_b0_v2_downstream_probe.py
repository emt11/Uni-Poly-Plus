#!/usr/bin/env python3
"""Run the B0-v2 downstream probe for one probe checkpoint.

For the given probe checkpoint, fine-tunes xc/ei/eat fold 0 with the normal
MTS protocol (epochs=100, patience=10) and the fixed B0-v2 downstream
identity (O8 / Star ON / MCL OFF / MD200 ON / upper=3.75).  Results land
under results/mts_b0_periodic_coordinate_denoising_v2/downstream_probe/<tag>.
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
    parser.add_argument("--probe", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--gpu-ids", default="0,1,2,3")
    args = parser.parse_args(argv)
    probe = Path(args.probe).resolve()
    if not probe.is_file():
        raise SystemExit(f"missing probe checkpoint: {probe}")
    if probe.stat().st_size == 0:
        raise SystemExit(f"empty probe checkpoint: {probe}")
    results = ROOT / "results/mts_b0_periodic_coordinate_denoising_v2/downstream_probe" / args.tag
    logs = ROOT / "logs/mts_b0_v2_downstream_probe" / args.tag
    command = [
        sys.executable, "scripts/run_mts_finetune_scheduler.py",
        "--python", sys.executable,
        "--gpu-ids", args.gpu_ids,
        "--results-dir", str(results),
        "--logs-dir", str(logs),
        "--tasks", "xc", "ei", "eat",
        "--folds", "0", "--seeds", "42",
        "--checkpoint", str(probe),
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
