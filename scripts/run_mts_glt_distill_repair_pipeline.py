#!/usr/bin/env python3
"""Stage-explicit orchestration for the C0/C1/C2 repair experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
CONFIG = {
    "c0": "configs/mts/glt_distill_repair_c0.json",
    "c1": "configs/mts/glt_distill_repair_c1.json",
    "c2": "configs/mts/glt_distill_repair_c2.json",
}
VERSIONS = {"c0": "none", "c1": "n_plus_1", "c2": "n_plus_2"}


def run(command):
    print("+", subprocess.list2cmdline([str(v) for v in command]), flush=True)
    subprocess.run([str(v) for v in command], cwd=ROOT, check=True)


def prepare():
    run([sys.executable, "scripts/create_mips_split_manifests.py", "--protocol", "outer5_inner20"])
    done = [ROOT / "data/processed/mips_trimer_scage/periodic_line_glt_distill_v2" / v / ".done" for v in ("n_plus_1", "n_plus_2")]
    if not all(path.is_file() for path in done):
        run([sys.executable, "scripts/build_mts_glt_distill_repair_sidecars_parallel.py", "--processes", "8"])


def validate():
    run([sys.executable, "-m", "pytest", "-q", "tests/test_mts_glt_distill_repair.py", "tests/test_mts_glt_distill.py", "tests/test_mts_student_architecture.py"])


def _complete(path, schema, version, step):
    if not path.is_file():
        return False
    payload = torch.load(path, map_location="cpu", weights_only=False)
    expected_revision = None if version == "none" else 2
    return (
        payload.get("schema") == schema and payload.get("version") == version
        and int(payload.get("step", -1)) == step
        and payload.get("geometry_revision") == expected_revision
    )


def _latest_resume(root, stage, version, final_step):
    revision = None if version == "none" else 2
    candidates = []
    paths = list(root.glob(f"{stage}_[0-9][0-9][0-9]k.pt"))
    rolling = root / f"{stage}_resume_latest.pt"
    if rolling.is_file():
        paths.append(rolling)
    for path in paths:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        step = int(payload.get("step", -1))
        if (
            payload.get("schema") == f"mts-glt-distill-repair-{stage}-state-v1"
            and payload.get("version") == version
            and payload.get("geometry_revision") == revision
            and 0 < step < final_step
        ):
            candidates.append((step, path))
    return max(candidates)[1] if candidates else None


def pretrain(group, stage):
    if group == "c0" and stage == "teacher":
        raise ValueError("C0 has no teacher")
    version = VERSIONS[group]
    root = ROOT / "results/mts_glt_distill_repair_control" / group / stage
    step = 5000 if stage == "teacher" else 20000
    path = root / ("teacher_005k.pt" if stage == "teacher" else "student_deploy_020k.pt")
    schema = f"mts-glt-distill-repair-{stage}-state-v1" if stage == "teacher" else "mts-glt-distill-repair-student-deploy-v1"
    if _complete(path, schema, version, step):
        print(f"reuse complete {group} {stage}: {path}")
        return
    resume = None
    if root.exists():
        resume = _latest_resume(root, stage, version, step)
        if resume is None:
            raise RuntimeError(f"incomplete/conflicting formal stage without valid resume: {root}")
    command = [
        "torchrun", "--standalone", "--nproc_per_node=3",
        "scripts/pretrain_mts_glt_distill.py", "--config", CONFIG[group],
        "--stage", stage,
    ]
    if stage == "student" and group != "c0":
        command.extend(("--teacher-checkpoint", str(
            ROOT / "results/mts_glt_distill_repair_control" / group / "teacher/teacher_005k.pt"
        )))
    if resume is not None:
        command.extend(("--resume", str(resume)))
    run(command)


def audit_common_initialization():
    payloads = [
        torch.load(
            ROOT / "results/mts_glt_distill_repair_control" / group / "student/student_step0_common.pt",
            map_location="cpu", weights_only=False,
        ) for group in ("c0", "c1", "c2")
    ]
    hashes = [payload["sha256"] for payload in payloads]
    if len(set(hashes)) != 1:
        raise RuntimeError(f"C0/C1/C2 common student initialization differs: {hashes}")
    output = ROOT / "results/mts_glt_distill_repair_control/comparison/initialization_audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"schema": "repair-common-init-audit-v1", "passed": True, "sha256": hashes[0], "groups": ["c0", "c1", "c2"]}, indent=2) + "\n")


def finetune(group):
    checkpoint = ROOT / "results/mts_glt_distill_repair_control" / group / "student/student_deploy_020k.pt"
    run([sys.executable, "scripts/run_mts_glt_distill_repair_finetune.py", "--group", group, "--checkpoint", checkpoint])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "validate", "teacher", "student", "audit-init", "finetune", "report", "all"))
    parser.add_argument("--group", choices=tuple(CONFIG))
    args = parser.parse_args()
    if args.stage in {"teacher", "student", "finetune"} and not args.group:
        parser.error("--group is required")
    if args.stage == "prepare": prepare()
    elif args.stage == "validate": validate()
    elif args.stage in {"teacher", "student"}: pretrain(args.group, args.stage)
    elif args.stage == "audit-init": audit_common_initialization()
    elif args.stage == "finetune": finetune(args.group)
    elif args.stage == "report": run([sys.executable, "scripts/report_mts_glt_distill_repair.py"])
    else:
        prepare(); validate()
        for group in ("c2", "c1"):
            pretrain(group, "teacher")
        for group in ("c0", "c1", "c2"):
            pretrain(group, "student")
        audit_common_initialization()
        for group in ("c0", "c1", "c2"):
            finetune(group)
        run([sys.executable, "scripts/report_mts_glt_distill_repair.py"])


if __name__ == "__main__":
    main()
