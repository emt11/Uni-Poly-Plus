#!/usr/bin/env python3
"""Explicit prepare/validate/train/finetune/report entry for N+ distillation."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
CONFIG = {
    "n_plus_2": "configs/mts/glt_distill_n_plus_2.json",
    "n_plus_1": "configs/mts/glt_distill_n_plus_1.json",
}


def run(command):
    print("+", subprocess.list2cmdline([str(v) for v in command]), flush=True)
    subprocess.run([str(v) for v in command], cwd=ROOT, check=True)


def prepare():
    done = [ROOT / "data/processed/mips_trimer_scage/periodic_line_glt_distill_v1" / version / ".done" for version in CONFIG]
    if not all(path.is_file() for path in done):
        run([sys.executable, "scripts/build_mts_glt_distill_sidecars.py"])


def validate():
    run([sys.executable, "-m", "pytest", "-q", "tests/test_mts_glt_distill.py", "tests/test_mts_glt_v3.py"])


def _complete_checkpoint(path, schema, version, step):
    if not path.is_file():
        return False
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return (
        payload.get("schema") == schema
        and payload.get("version") == version
        and int(payload.get("step", -1)) == step
    )


def pretrain(version, stage):
    root = ROOT / "results/mts_glt_v2_distill" / version / stage
    step = 5000 if stage == "teacher" else 20000
    expected = root / (
        "teacher_005k.pt" if stage == "teacher" else "student_deploy_020k.pt"
    )
    schema = (
        "mts-glt-distill-teacher-state-v1" if stage == "teacher"
        else "mts-glt-distill-student-deploy-v1"
    )
    if _complete_checkpoint(expected, schema, version, step):
        print(f"reuse complete {version} {stage}: {expected}", flush=True)
        return
    if root.exists():
        raise RuntimeError(f"incomplete or conflicting pretraining directory: {root}")
    run(["torchrun", "--standalone", "--nproc_per_node=3", "scripts/pretrain_mts_glt_distill.py", "--config", CONFIG[version], "--stage", stage])


def finetune(version):
    checkpoint = ROOT / "results/mts_glt_v2_distill" / version / "student/student_deploy_020k.pt"
    run([sys.executable, "scripts/run_mts_glt_distill_finetune.py", "--version", version, "--checkpoint", checkpoint])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "validate", "teacher", "student", "finetune", "report", "all"))
    parser.add_argument("--version", choices=tuple(CONFIG))
    args = parser.parse_args()
    if args.stage in {"teacher", "student", "finetune"} and not args.version:
        parser.error("--version is required for this stage")
    if args.stage == "prepare": prepare()
    elif args.stage == "validate": validate()
    elif args.stage == "teacher": pretrain(args.version, "teacher")
    elif args.stage == "student": pretrain(args.version, "student")
    elif args.stage == "finetune": finetune(args.version)
    elif args.stage == "report": run([sys.executable, "scripts/report_mts_glt_distill.py"])
    else:
        prepare(); validate()
        for version in ("n_plus_2", "n_plus_1"):
            pretrain(version, "teacher"); pretrain(version, "student")
        for version in ("n_plus_2", "n_plus_1"):
            finetune(version)
        run([sys.executable, "scripts/report_mts_glt_distill.py"])


if __name__ == "__main__":
    main()
