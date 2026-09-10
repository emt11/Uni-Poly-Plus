#!/usr/bin/env python3
"""Stage-explicit entry for the GLT-V2 revision-2 New-C0 baseline.

New-C0 is the no-distillation baseline of the revised GLT-V2 route: 138-D
atom input, full-row canonical atom masking, six Pre-LN O8 layers with GELU,
a trainable AtomicConditionedMD200, pre/post-MD masked atom CE, canonical
atom mean pooling, an Identity graph wrapper and a 512->512->1 predictor.

It is an independent experiment: its config declares the archived
``results/mts_glt_distill_repair_control`` and ``results/mts_glt_v2_distill``
roots as protected, so it can never write into them.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
CONFIG = "configs/mts/glt_v2_r2_c0_gelu138_mipshead.json"
EXPERIMENT_ID = "glt_v2_r2_c0_gelu138_mipshead"
RESULT_ROOT = ROOT / "results/glt_v2_r2_c0_gelu138_mipshead"
LOG_ROOT = ROOT / "logs/glt_v2_r2_c0_gelu138_mipshead"
VALIDATE_TESTS = (
    "tests/test_mts_glt_distill.py",
    "tests/test_mts_student_architecture.py",
    "tests/test_mips_atom_embedding.py",
    "tests/test_mts_new_c0_contract.py",
)


def run(command):
    print("+", subprocess.list2cmdline([str(v) for v in command]), flush=True)
    subprocess.run([str(v) for v in command], cwd=ROOT, check=True)


def prepare():
    if not (ROOT / "data/splits/mips_outer5_inner20").is_dir():
        run([sys.executable, "scripts/create_mips_split_manifests.py", "--protocol", "outer5_inner20"])
    done = [
        ROOT / "data/processed/mips_trimer_scage/periodic_line_glt_distill_v2" / version / ".done"
        for version in ("n_plus_1", "n_plus_2")
    ]
    if not all(path.is_file() for path in done):
        run([sys.executable, "scripts/build_mts_glt_distill_repair_sidecars_parallel.py", "--processes", "8"])


def validate():
    run([sys.executable, "-m", "pytest", "-q", *VALIDATE_TESTS])


def student(resume=None):
    stage_root = RESULT_ROOT / "student"
    if resume is None and stage_root.exists():
        raise SystemExit(f"refusing to write into an existing stage directory: {stage_root}")
    command = [
        "torchrun", "--standalone", "--nproc_per_node=3",
        "scripts/pretrain_mts_glt_distill.py", "--config", CONFIG, "--stage", "student",
    ]
    if resume is not None:
        command.extend(("--resume", str(resume)))
    run(command)


def finetune():
    checkpoint = RESULT_ROOT / "student/student_deploy_020k.pt"
    if not checkpoint.is_file():
        raise SystemExit(f"missing New-C0 deployment: {checkpoint}")
    run([
        sys.executable, "scripts/run_mts_glt_distill_repair_finetune.py",
        "--group", "c0", "--checkpoint", str(checkpoint),
        "--experiment-id", f"{EXPERIMENT_ID}_outer5_inner20",
        "--results-dir", str(RESULT_ROOT / "downstream/outer5_inner20"),
        "--logs-dir", str(LOG_ROOT / "downstream/outer5_inner20"),
    ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "validate", "student", "finetune", "all"))
    parser.add_argument("--resume")
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare()
    elif args.stage == "validate":
        validate()
    elif args.stage == "student":
        student(args.resume)
    elif args.stage == "finetune":
        finetune()
    else:
        prepare()
        validate()
        student()
        finetune()


if __name__ == "__main__":
    main()
