#!/usr/bin/env python3
"""Run the bounded sigma/lambda screening for the B0-v2 route.

Each arm is a real three-GPU DDP trajectory.  The script only aggregates the
last stable window; it never promotes a checkpoint or changes production
defaults.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = ROOT / "configs/mts/b0_v2_probe.json"
TORCHRUN_BIN = os.environ.get("TORCHRUN_BIN", "/opt/conda/envs/MTS/bin/torchrun")


def _read_metrics(path: Path):
    rows = []
    if not path.is_file():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--run-id", default="sigma_lambda_v2")
    parser.add_argument("--steps", type=int, default=500,
                        help="stop_after optimizer steps; scheduler horizon stays 20000")
    parser.add_argument("--window", type=int, default=100)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=336)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument(
        "--arms", default="",
        help="comma-separated sigma:lambda arms; default runs all four",
    )
    args = parser.parse_args(argv)
    if args.steps < 300 or args.steps > 2500:
        raise ValueError("screening stop steps must be in the 300-2500 range")
    if args.window <= 0 or args.window > args.steps:
        raise ValueError("screening window must be positive and <= steps")
    arm_list = []
    if args.arms:
        for arm in args.arms.split(","):
            sigma, coordinate_weight = arm.split(":")
            arm_list.append((float(sigma), float(coordinate_weight)))
    else:
        arm_list = [(0.03, 0.1), (0.03, 0.3), (0.05, 0.1), (0.05, 0.3)]

    base = json.loads(BASE_CONFIG.read_text(encoding="utf-8"))
    base.update({
        # The scheduler horizon and warmup are the formal 20k/2000 contract.
        # --resume-smoke-stop-steps only exits the loop early.
        "max_optimizer_steps": 20000,
        "warmup_steps": 2000,
        "batch_size": int(args.batch_size),
        "gradient_accumulation_steps": int(args.gradient_accumulation_steps),
        "global_batch_size": int(args.batch_size) * 3 * int(args.gradient_accumulation_steps),
        "loader_workers": int(args.workers),
        "checkpoint_interval_steps": 2000,
    })
    root = ROOT / "results/mts_b0_periodic_coordinate_denoising_v2/screening" / args.run_id
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    for sigma, coordinate_weight in arm_list:
        tag = f"sigma{sigma:.2f}_lambda{coordinate_weight:.1f}"
        result_root = root / tag
        payload = dict(base)
        payload.update({
            "experiment_id": f"mts_b0_v2_screen_{tag}",
            "noise_sigma": sigma,
            "coordinate_loss_weight": coordinate_weight,
            "result_root": str(result_root),
            "output_path": str(result_root / "unused.pth"),
        })
        config_path = result_root / "config.json"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        command = [
            TORCHRUN_BIN, "--standalone", "--nproc_per_node=3",
            "scripts/pretrain.py", "--b0_config", str(config_path),
            "--resume_smoke_stop_steps", str(args.steps),
        ]
        env = dict(os.environ)
        env["PYTHONPATH"] = str(ROOT)
        env["CUDA_VISIBLE_DEVICES"] = "1,2,3"
        started = time.monotonic()
        completed = subprocess.run(command, cwd=ROOT, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
        elapsed = time.monotonic() - started
        (result_root / "run.log").write_text(completed.stdout, encoding="utf-8")
        metrics = _read_metrics(result_root / "b0_training_metrics.jsonl")
        stable = metrics[-int(args.window):]
        finite = bool(stable) and all(
            all(abs(float(row.get(name, 0.0))) < float("inf") for name in ("loss", "masked_atom_loss", "coordinate_loss", "r_denoise", "displacement_rms"))
            for row in stable
        )
        summary = {
            "tag": tag,
            "sigma": sigma,
            "coordinate_loss_weight": coordinate_weight,
            "steps": args.steps,
            "stable_window": len(stable),
            "returncode": completed.returncode,
            "wall_seconds": elapsed,
            "finite": finite,
            "mean_loss": sum(float(x["loss"]) for x in stable) / len(stable) if stable else None,
            "mean_r_denoise": sum(float(x["r_denoise"]) for x in stable) / len(stable) if stable else None,
            "mean_displacement_rms": sum(float(x["displacement_rms"]) for x in stable) / len(stable) if stable else None,
            "min_r_denoise": min((float(x["r_denoise"]) for x in stable), default=None),
            "max_r_denoise": max((float(x["r_denoise"]) for x in stable), default=None),
            "metrics_path": str(result_root / "b0_training_metrics.jsonl"),
            "log_path": str(result_root / "run.log"),
            "command": command,
        }
        rows.append(summary)
        print(json.dumps(summary, sort_keys=True), flush=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"schema": "mts-b0-v2-screening-v1", "rows": rows}, indent=2) + "\n", encoding="utf-8")
    return 0 if all(row["returncode"] == 0 and row["finite"] for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
