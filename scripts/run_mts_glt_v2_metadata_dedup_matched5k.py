#!/usr/bin/env python3
"""Prepare (or explicitly execute) the matched FULL -> DEDUP 5k pair.

The default is a dry-run.  Execution requires ``--execute`` and the two
configs are always run in this order with the stop value fixed at exactly
5,000 optimizer steps.  This guard prevents an accidental long trajectory
from being started by this helper.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = (
    ROOT / "configs/mts/metadata_dedup_matched5k_v1/full_005k.json",
    ROOT / "configs/mts/metadata_dedup_matched5k_v1/dedup_005k.json",
)
LOG_ROOT = ROOT / "logs/mts_glt_v2/metadata_dedup_matched5k_v1"
PYTHON_BIN = os.environ.get("PYTHON_BIN", "/opt/conda/envs/MTS/bin/python")


def _plan():
    commands = []
    for config in CONFIGS:
        payload = json.loads(config.read_text(encoding="utf-8"))
        if int(payload["stop_after_steps"]) != 5000:
            raise ValueError(f"metadata-dedup config must stop at 5000: {config}")
        commands.append({
            "experiment_config": str(config),
            "metadata_mode": payload["glt_metadata_mode"],
            "stop_after_steps": int(payload["stop_after_steps"]),
            "command": ["scripts/run_mips_trimer_scage.sh"],
        })
    return {
        "schema": "mts-glt-v2-metadata-dedup-matched5k-launcher-v1",
        "preflight_required": True,
        "preflight": [
            PYTHON_BIN,
            "scripts/analyze_mts_glt_v2_metadata_dedup_sanity.py",
            "--output", "results/mts_glt_v2/metadata_dedup_matched5k_v1",
        ],
        "order": [item["metadata_mode"] for item in commands],
        "stop_after_steps": 5000,
        "cuda_visible_devices": "1,2,3",
        "commands": commands,
        "execute_requires_explicit_flag": True,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--execute", action="store_true",
        help="run both 5k jobs; omitted means dry-run only",
    )
    args = parser.parse_args()
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    plan = _plan()
    plan_path = LOG_ROOT / "launcher_plan.json"
    plan_path.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(plan, indent=2))
    if not args.execute:
        print(f"dry-run only; plan written to {plan_path}")
        return
    subprocess.run(plan["preflight"], cwd=ROOT, check=True)
    for item in plan["commands"]:
        env = os.environ.copy()
        env["EXPERIMENT_CONFIG"] = item["experiment_config"]
        env["CUDA_VISIBLE_DEVICES"] = "1,2,3"
        subprocess.run(
            ["bash", str(ROOT / "scripts/run_mips_trimer_scage.sh")],
            cwd=ROOT, env=env, check=True,
        )


if __name__ == "__main__":
    main()
