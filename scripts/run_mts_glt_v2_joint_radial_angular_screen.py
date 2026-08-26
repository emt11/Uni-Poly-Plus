#!/usr/bin/env python3
"""Run AC/RA downstream mechanism screening; never rerun baseline B."""
from __future__ import annotations
import argparse, json, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/mts/glt_v2_joint_radial_angular_screen_v1.json"

def command(config, arm, gpu_ids, smoke=False):
    mode = {
        "ac": "o8_glt_atom_sbf_angle_control",
        "ra": "o8_glt_atom_sbf_radial_angle",
    }[arm]
    output, logs = ROOT/config["output_root"], ROOT/config["log_root"]
    suffix = f"smoke_v2/{arm}" if smoke else arm
    return [
        sys.executable, "scripts/run_mts_finetune_scheduler.py",
        "--python", sys.executable, "--gpu-ids", gpu_ids,
        "--results-dir", str(output/suffix), "--logs-dir", str(logs/suffix),
        "--tasks", *( ["eat"] if smoke else config["tasks"]),
        "--folds", *( ["0"] if smoke else [str(x) for x in config["folds"]]),
        "--seeds", str(config["seed"]),
        "--checkpoint", str((ROOT/config["checkpoint"]).resolve()),
        "--checkpoint-seed", str(config["seed"]), "--pretrain-dataset", "PI1M_v2",
        "--finetune-epochs", str(2 if smoke else config["epochs"]),
        "--finetune-patience", str(config["patience"]),
        "--batch-size", str(config["train_batch_size"]),
        "--eval-batch-size", str(config["eval_batch_size"]),
        "--amp-dtype", config["precision"], "--loader-workers", str(config["workers"]),
        "--evaluation-protocol", config["evaluation_protocol"],
        "--graph-lr", str(config["graph_lr"]),
        "--fusion-lr", str(config["fusion_lr"]),
        "--head-lr", str(config["head_lr"]),
        "--train-args",
        "--experiment_id", config["experiment"]+"_"+suffix.replace("/", "_"),
        "--config_schema", "mts-glt-v2-downstream", "--graph_encoder_type", "mips_trimer_scage",
        "--topology_attention_variant", "o8", "--no-use_star_rbf", "--no-use_mcl", "--use_md200",
        "--mts_glt_version", "v2", "--mts_glt_layers", str(config["glt_layers"]),
        "--mts_glt_attention_variant", config["glt_attention_variant"],
        "--mts_glt_mode", mode,
        "--periodic_line_glt_sidecar", str((ROOT/config["sidecar"]).resolve()),
        "--mts_glt_fusion_strategy", "legacy_zero", "--save_best_checkpoint",
        "--best_checkpoint_dir", str(output/("smoke_v2/checkpoints" if smoke else "checkpoints")/arm),
        "--mts_glt_joint_basis_diagnostics_dir", str(output/("smoke_v2/joint_basis_units" if smoke else "joint_basis_units")/arm),
        "--graph_lr", str(config["graph_lr"]), "--fusion_lr", str(config["fusion_lr"]),
        "--head_lr", str(config["head_lr"]), "--warmup_epochs", str(config["warmup_epochs"]),
        "--regression_loss", config["loss"], "--huber_beta", str(config["huber_beta"]),
        "--max_grad_norm", str(config["max_grad_norm"]), "--head_dropout", str(config["head_dropout"]),
        "--weight_decay", str(config["weight_decay"]),
    ]

def main(argv=None):
    p=argparse.ArgumentParser(); p.add_argument("--gpu-ids", default="0,1,2,3")
    p.add_argument("--smoke", action="store_true"); p.add_argument("--report-only", action="store_true")
    a=p.parse_args(argv); c=json.loads(CONFIG.read_text()); out=ROOT/c["output_root"]
    if not a.report_only:
        prerequisites = (
            ("analyze_mts_glt_v2_joint_ra_basis.py", out/"basis_sanity.json"),
            ("check_mts_glt_v2_joint_ra_sanity.py", out/"gradient_sanity.json"),
        )
        for script, artifact in prerequisites:
            if not artifact.exists():
                subprocess.run([sys.executable, "scripts/"+script], cwd=ROOT, check=True)
        for arm in ("ac", "ra"):
            code=subprocess.call(command(c, arm, a.gpu_ids, a.smoke), cwd=ROOT)
            if code: return code
    if a.smoke: return 0
    return subprocess.call([sys.executable,"scripts/report_mts_glt_v2_joint_radial_angular_screen.py"],cwd=ROOT)
if __name__ == "__main__": raise SystemExit(main())
