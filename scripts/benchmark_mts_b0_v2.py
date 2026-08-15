#!/usr/bin/env python3
"""Run the bounded real B0-v2 global-batch/worker benchmark."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = ROOT / "configs/mts/b0_v2_probe.json"
PYTHON_BIN = os.environ.get("PYTHON_BIN", "/opt/conda/envs/MTS/bin/python")
TORCHRUN_BIN = os.environ.get("TORCHRUN_BIN", "/opt/conda/envs/MTS/bin/torchrun")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--workers", nargs="+", type=int, default=[0, 2, 4, 6])
    parser.add_argument("--run-id", default="default")
    args = parser.parse_args(argv)
    base = json.loads(BASE_CONFIG.read_text(encoding="utf-8"))
    base.update({
        "result_root": f"results/mts_b0_periodic_coordinate_denoising_v2/benchmark/{args.run_id}",
        "output_path": f"results/mts_b0_periodic_coordinate_denoising_v2/benchmark/{args.run_id}/unused.pth",
        "max_optimizer_steps": int(args.steps),
        "global_batch_size": 1008,
        "lr": 0.0002,
        "weight_decay": 0.0,
    })
    rows = []
    for batch_size, accumulation in ((336, 1), (168, 2), (84, 4), (42, 8)):
        for workers in args.workers:
            tag = f"batch{batch_size}_acc{accumulation}_w{workers}"
            payload = dict(base)
            payload["experiment_id"] = f"mts_b0_v2_benchmark_{tag}"
            payload["batch_size"] = int(batch_size)
            payload["gradient_accumulation_steps"] = int(accumulation)
            payload["loader_workers"] = int(workers)
            payload["result_root"] = str(
                ROOT / "results/mts_b0_periodic_coordinate_denoising_v2/benchmark" / args.run_id / tag
            )
            payload["output_path"] = str(Path(payload["result_root"]) / "unused.pth")
            config_path = Path(payload["result_root"]) / "config.json"
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config_path.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            started = time.monotonic()
            env = dict(os.environ)
            env["PYTHONPATH"] = str(ROOT)
            env["CUDA_VISIBLE_DEVICES"] = "1,2,3"
            command = [
                TORCHRUN_BIN, "--standalone", "--nproc_per_node=3",
                "scripts/pretrain.py", "--b0_config", str(config_path)
            ]
            completed = subprocess.run(
                command, cwd=ROOT, env=env, text=True,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
            )
            elapsed = time.monotonic() - started
            log_path = Path(payload["result_root"]) / "run.log"
            log_path.write_text(completed.stdout, encoding="utf-8")
            metrics = Path(payload["result_root"]) / "b0_training_metrics.jsonl"
            rows.append({
                "tag": tag,
                "batch_size_per_rank": batch_size,
                "gradient_accumulation_steps": accumulation,
                "global_batch_size": batch_size * accumulation * 3,
                "workers": workers,
                "returncode": completed.returncode,
                "wall_seconds": elapsed,
                "metrics_path": str(metrics),
                "log_path": str(log_path),
                "command": command,
            })
            print(json.dumps(rows[-1], sort_keys=True), flush=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"schema": "mts-b0-v2-benchmark-v1", "rows": rows}, indent=2) + "\n", encoding="utf-8")
    return 0 if all(row["returncode"] == 0 for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
