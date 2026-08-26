#!/usr/bin/env python3
"""Run the two new GLT-v2 interaction arms, then build the paired report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/mts/glt_v2_x23_conditioning_screen_v1.json"


def _command(config, arm, gpu_ids, smoke=False):
    mode = {
        "s3": "o8_glt_atom_self3d",
        "x23": "o8_glt_atom_x23",
    }[arm]
    output = ROOT / config["output_root"]
    log_root = ROOT / config["log_root"]
    suffix = f"smoke/{arm}" if smoke else arm
    tasks = ["eat"] if smoke else config["tasks"]
    folds = ["0"] if smoke else [str(v) for v in config["folds"]]
    epochs = 2 if smoke else config["epochs"]
    result_root = output / suffix
    return [
        sys.executable, "scripts/run_mts_finetune_scheduler.py",
        "--python", sys.executable, "--gpu-ids", gpu_ids,
        "--results-dir", str(result_root),
        "--logs-dir", str(log_root / suffix),
        "--tasks", *tasks, "--folds", *folds,
        "--seeds", str(config["seed"]),
        "--checkpoint", str((ROOT / config["checkpoint"]).resolve()),
        "--checkpoint-seed", str(config["seed"]),
        "--pretrain-dataset", "PI1M_v2",
        "--finetune-epochs", str(epochs),
        "--finetune-patience", str(config["patience"]),
        "--batch-size", str(config["train_batch_size"]),
        "--eval-batch-size", str(config["eval_batch_size"]),
        "--amp-dtype", config["precision"],
        "--loader-workers", str(config["workers"]),
        "--evaluation-protocol", config["evaluation_protocol"],
        "--train-args",
        "--experiment_id", config["experiment"] + "_" + suffix,
        "--config_schema", "mts-glt-v2-downstream",
        "--graph_encoder_type", "mips_trimer_scage",
        "--topology_attention_variant", "o8",
        "--no-use_star_rbf", "--no-use_mcl", "--use_md200",
        "--mts_glt_version", "v2", "--mts_glt_layers", str(config["glt_layers"]),
        "--mts_glt_attention_variant", config["glt_attention_variant"],
        "--mts_glt_mode", mode,
        "--periodic_line_glt_sidecar", str((ROOT / config["sidecar"]).resolve()),
        "--mts_glt_fusion_strategy", "legacy_zero",
        "--save_best_checkpoint",
        "--best_checkpoint_dir", str(output / "checkpoints" / suffix),
        "--mts_glt_interaction_diagnostics_dir", str(output / "interaction_units" / suffix),
        "--graph_lr", str(config["graph_lr"]),
        "--fusion_lr", str(config["fusion_lr"]),
        "--head_lr", str(config["head_lr"]),
        "--warmup_epochs", str(config["warmup_epochs"]),
        "--regression_loss", config["loss"], "--huber_beta", str(config["huber_beta"]),
        "--head_dropout", str(config["head_dropout"]),
        "--weight_decay", str(config["weight_decay"]),
    ]


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-ids", default="0,1,2,3")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args(argv)
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    if not args.report_only:
        for arm in ("s3", "x23"):
            status = subprocess.call(
                _command(config, arm, args.gpu_ids, args.smoke), cwd=ROOT
            )
            if status:
                return status
    if args.smoke:
        return 0
    return subprocess.call([
        sys.executable, "scripts/report_mts_glt_v2_x23_conditioning_screen.py"
    ], cwd=ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
