#!/usr/bin/env python3
"""Run GraphGate FusionWarm screening and its conditional 8x5 expansion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/mts_glt_graphgate_v1/fusion_realization_v1"
LOGS = ROOT / "logs/mts_glt_graphgate_v1/fusion_realization_v1"
SCREENING_TASKS = ("xc", "ei", "eps")
EXPANSION_TASKS = ("eat", "eea", "egb", "egc", "nc")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("screening", "full"), default="screening")
    parser.add_argument("--gpu-ids", default="0,1,2,3")
    parser.add_argument(
        "--checkpoint",
        default="pretrained_models/mts_glt_graphgate_v1/mts_glt_graphgate_v1.pth",
    )
    args = parser.parse_args(argv)

    checkpoint = (ROOT / args.checkpoint).resolve()
    if not checkpoint.is_file():
        raise SystemExit(f"missing GraphGate checkpoint: {checkpoint}")
    sidecar = (
        ROOT
        / "data/processed/mips_trimer_scage/periodic_line_glt_central_v1/downstream_union"
    )
    if not (sidecar / ".done").is_file():
        raise SystemExit(f"missing GraphGate downstream sidecar: {sidecar}")

    if args.phase == "full":
        decision_path = OUTPUT / "fusion_realization_screening.json"
        if not decision_path.is_file():
            raise SystemExit("screening report is missing; full expansion is not authorized")
        decision = json.loads(decision_path.read_text(encoding="utf-8"))
        if not bool(decision.get("expand_8x5", False)):
            raise SystemExit("screening GO conditions failed; full expansion is stopped")
        tasks = EXPANSION_TASKS
    else:
        tasks = SCREENING_TASKS

    command = [
        sys.executable, "scripts/run_mts_finetune_scheduler.py",
        "--python", sys.executable,
        "--gpu-ids", args.gpu_ids,
        "--results-dir", str(OUTPUT / "folds"),
        "--logs-dir", str(LOGS),
        "--tasks", *tasks,
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
        "--experiment_id", "mts_glt_graphgate_fusion_realization_v1",
        "--config_schema", "mts-glt-graphgate-v1-downstream",
        "--graph_encoder_type", "mips_trimer_scage",
        "--topology_attention_variant", "o8",
        "--no-use_star_rbf", "--no-use_mcl", "--use_md200",
        "--mts_glt_version", "graphgate_v1",
        "--mts_glt_layers", "6",
        "--mts_glt_attention_variant", "mips",
        "--mts_glt_mode", "o8_glt_graph",
        "--mts_glt_geometry_mode", "full",
        "--periodic_line_glt_sidecar", str(sidecar),
        "--mts_glt_fusion_strategy", "fusion_warm",
        "--mts_glt_fusion_warm_epochs", "5",
        "--mts_glt_initial_alpha", "0.05",
        "--mts_glt_fusion_warm_dir", str(OUTPUT),
        "--graph_lr", "1e-5",
        "--fusion_lr", "1e-4",
        "--head_lr", "1e-4",
        "--warmup_epochs", "0",
        "--regression_loss", "huber",
        "--huber_beta", "0.5",
        "--head_dropout", "0.25",
    ]
    status = subprocess.call(command, cwd=ROOT)
    if status:
        return status
    return subprocess.call([
        sys.executable,
        "scripts/report_mts_glt_graphgate_fusion_realization.py",
        "--scope", args.phase,
    ], cwd=ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
