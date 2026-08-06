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
            "prepare-downstream",
            "finalize-cache",
            "pretrain",
            "finetune",
            "summarize",
            "sota-campaign",
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
    if args.args:
        raise SystemExit(
            "pretrain/finetune use environment variables; remove positional "
            "arguments"
        )
    if args.command == "pretrain":
        return _run(["bash", "scripts/run_mts.sh"], {"PRETRAIN_ONLY": "1"})
    return _run(["bash", "scripts/run_mts.sh"], {"FINETUNE_ONLY": "1"})


if __name__ == "__main__":
    raise SystemExit(main())
