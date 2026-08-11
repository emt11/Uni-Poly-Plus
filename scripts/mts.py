#!/usr/bin/env python3
"""Production dispatcher for MIPS-Trimer-SCAGE (MTS)."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = os.environ.get("PYTHON_BIN", sys.executable)


def _run(command, env=None):
    merged = os.environ.copy()
    if env:
        merged.update(env)
    return subprocess.call(command, cwd=ROOT, env=merged)


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="mts.py",
        description="MIPS-Trimer-SCAGE (MTS) production dispatcher",
    )
    parser.add_argument(
        "command",
        choices=(
            "doctor",
            "benchmark-pretrain",
            "preflight-angle",
            "build-angle-v2",
            "build-angle-cache",
            "build-mcl-thresholds",
            "prepare-downstream",
            "validate-cache",
            "finalize-cache",
            "pretrain",
            "finetune",
            "summarize",
            "sota-campaign",
            "geometry-ablation",
        ),
    )
    parser.add_argument("args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.command == "doctor":
        return _run([PYTHON, "scripts/doctor_mips_trimer_scage.py", *args.args])
    if args.command == "benchmark-pretrain":
        return _run([PYTHON, "scripts/benchmark_mts_pretrain.py", *args.args])
    if args.command == "preflight-angle":
        return _run([PYTHON, "scripts/preflight_mts_angle_cache.py", *args.args])
    if args.command == "build-angle-v2":
        return _run([PYTHON, "scripts/build_mts_angle_v2.py", *args.args])
    if args.command == "prepare-downstream":
        if args.args:
            raise SystemExit("prepare-downstream takes no positional arguments")
        return _run([PYTHON, "scripts/prepare_mips_trimer_downstream.py"])
    if args.command == "validate-cache":
        return _run([PYTHON, "scripts/validate_mts_cache.py", *args.args])
    if args.command == "build-angle-cache":
        return _run([PYTHON, "scripts/prepare_mts_angle_cache.py", *args.args])
    if args.command == "build-mcl-thresholds":
        return _run([PYTHON, "scripts/build_mts_mcl_thresholds.py", *args.args])
    if args.command == "finalize-cache":
        return _run([PYTHON, "scripts/finalize_mips_trimer_cache.py", *args.args])
    if args.command == "summarize":
        summary_args = list(args.args)
        if not summary_args:
            summary_args = [
                "--results-root", "results/mts_finetune_v2",
                "--output-csv", "results/mts_finetune_v2/mts_finetune_summary.csv",
                "--output-md", "results/mts_finetune_v2/mts_finetune_summary.md",
            ]
        return _run([PYTHON, "scripts/summarize_mips_trimer_scage.py", *summary_args])
    if args.command == "sota-campaign":
        return _run([PYTHON, "scripts/run_mts_sota_campaign.py", *args.args])
    if args.command == "geometry-ablation":
        return _run([
            PYTHON, "scripts/run_mts_geometry_injection_ablation.py", *args.args
        ])
    if args.command == "pretrain":
        profile = "configs/mts/pretraining/canonical_ru_angle20_v1.json"
        benchmark = ""
        output = "pretrained_models/mts/mts_joint_pretraining_pi1m_v2_seed42_canonical_angle20_v1.pth"
        resume = False
        pretrain_parser = argparse.ArgumentParser(add_help=False)
        pretrain_parser.add_argument("--profile", default=profile)
        pretrain_parser.add_argument("--benchmark", default=benchmark)
        pretrain_parser.add_argument("--output", default=output)
        pretrain_parser.add_argument("--resume", action="store_true")
        parsed = pretrain_parser.parse_args(args.args)
        profile = str(parsed.profile)
        env = {
            "PRETRAIN_ONLY": "1",
            "PRETRAIN_PROFILE": profile,
            "JOINT_CKPT": str(parsed.output),
        }
        if parsed.benchmark:
            env["PRETRAIN_BENCHMARK_JSON"] = str(parsed.benchmark)
        if parsed.resume:
            env["RESUME"] = "1"
        return _run(["bash", "scripts/run_mts.sh"], env)
    if args.args:
        raise SystemExit(
            "finetune uses environment variables; remove positional arguments"
        )
    return _run(["bash", "scripts/run_mts.sh"], {"FINETUNE_ONLY": "1"})


if __name__ == "__main__":
    raise SystemExit(main())
