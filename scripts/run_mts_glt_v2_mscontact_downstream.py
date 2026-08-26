#!/usr/bin/env python3
"""Run the matched S4/MS45 downstream screening without extra modes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/mts/mscontact_v1/downstream.json"


def command(arm, config, gpu_ids, results_root, logs_root):
    checkpoint = (ROOT / config["checkpoint"][arm]).resolve()
    line_sidecar = (ROOT / config["line_sidecar"]).resolve()
    spatial_sidecar = (ROOT / config["spatial_sidecar"]).resolve()
    for path in (checkpoint, line_sidecar / ".done", spatial_sidecar / ".done"):
        if not path.is_file():
            raise SystemExit(f"missing required downstream artifact: {path}")
    results = Path(results_root).resolve() / arm
    logs = Path(logs_root).resolve() / arm
    return [
        sys.executable, "scripts/run_mts_finetune_scheduler.py",
        "--python", sys.executable, "--gpu-ids", gpu_ids,
        "--results-dir", str(results), "--logs-dir", str(logs),
        "--tasks", *config["tasks"],
        "--folds", *(str(value) for value in config["folds"]),
        "--seeds", str(config["seed"]),
        "--checkpoint", str(checkpoint),
        "--checkpoint-seed", str(config["seed"]),
        "--checkpoint-tier", "1m", "--pretrain-dataset", "PI1M_v2",
        "--finetune-epochs", str(config["epochs"]),
        "--finetune-patience", str(config["patience"]),
        "--batch-size", str(config["train_batch"]),
        "--eval-batch-size", str(config["eval_batch"]),
        "--amp-dtype", config["precision"],
        "--loader-workers", str(config["workers"]),
        "--evaluation-protocol", config["evaluation_protocol"],
        "--train-args",
        "--experiment_id", f"mts_glt_v2_mscontact_v1_{arm}",
        "--config_schema", "mts-glt-v2-downstream",
        "--graph_encoder_type", "mips_trimer_scage",
        "--topology_attention_variant", "o8",
        "--no-use_star_rbf", "--no-use_mcl", "--use_md200",
        "--mts_glt_version", "v2", "--mts_glt_layers", "6",
        "--mts_glt_attention_variant", "mips",
        "--mts_glt_mode", "o8_glt_atom_spatial",
        "--periodic_line_glt_sidecar", str(line_sidecar),
        "--periodic_spatial_contact_sidecar", str(spatial_sidecar),
        "--mts_spatial_shell_mode", arm,
        "--mts_glt_fusion_strategy", "fusion_warm",
        "--mts_glt_fusion_warm_epochs", str(config["encoder_freeze_warm_epochs"]),
        "--mts_glt_initial_alpha", "0.05",
        "--mts_glt_fusion_stage2_trainability", "joint",
        "--mts_glt_fusion_warm_dir", str(results),
        "--warmup_epochs", str(config["lr_warmup_epochs"]),
        "--regression_loss", "huber", "--huber_beta", str(config["huber_beta"]),
        "--max_grad_norm", str(config["grad_clip"]),
        "--head_dropout", str(config["head_dropout"]),
        "--weight_decay", "0.02",
        "--mts_o8_lr", str(config["encoder_lr"]),
        "--mts_geometry_lr", str(config["encoder_lr"]),
        "--mts_adapter_lr", str(config["encoder_lr"]),
        "--graph_lr", str(config["encoder_lr"]),
        "--fusion_lr", str(config["fusion_lr"]),
        "--head_lr", str(config["head_lr"]),
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--arm", choices=("s4", "ms45", "c5_mixed", "all"), default="all"
    )
    parser.add_argument("--gpu-ids", default="0,1,2,3")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument(
        "--results-root",
        default=str(ROOT / "results/mts_glt_v2/mscontact_v1/downstream"),
    )
    parser.add_argument(
        "--logs-root",
        default=str(ROOT / "logs/mts_glt_v2/mscontact_v1/downstream"),
    )
    args = parser.parse_args()
    config = json.loads(Path(args.config).resolve().read_text())
    arms = ("s4", "ms45") if args.arm == "all" else (args.arm,)
    for arm in arms:
        status = subprocess.call(
            command(arm, config, args.gpu_ids, args.results_root, args.logs_root),
            cwd=ROOT,
        )
        if status:
            return status
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
