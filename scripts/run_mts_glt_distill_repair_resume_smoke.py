#!/usr/bin/env python3
"""Run a fresh interrupted-versus-continuous repair pretraining smoke."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess

import torch


ROOT = Path(__file__).resolve().parents[1]
MODEL_ATOL = 2e-6
OPTIMIZER_ATOL = 2e-5


def _run(command: list[str], gpu_ids: str) -> None:
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = gpu_ids
    print("+", subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, cwd=ROOT, env=environment, check=True)


def _assert_nested_close(left, right, path: str = "checkpoint") -> None:
    if isinstance(left, torch.Tensor):
        if not isinstance(right, torch.Tensor):
            raise AssertionError(f"{path}: tensor/type mismatch")
        if left.is_floating_point() or left.is_complex():
            # The two branches are separate BF16/NCCL launches, so their first
            # two nominally identical steps already differ at sub-micro scale.
            # Use an absolute bound for near-zero Adam moments; keep RNG,
            # counters, scheduler and all non-floating state exact below.
            atol = OPTIMIZER_ATOL if path.startswith("checkpoint.optimizer") else MODEL_ATOL
            torch.testing.assert_close(left, right, rtol=0, atol=atol, msg=path)
        else:
            torch.testing.assert_close(left, right, rtol=0, atol=0, msg=path)
        return
    if isinstance(left, dict):
        if not isinstance(right, dict) or left.keys() != right.keys():
            raise AssertionError(f"{path}: mapping keys mismatch")
        for key in left:
            _assert_nested_close(left[key], right[key], f"{path}.{key}")
        return
    if isinstance(left, (list, tuple)):
        if not isinstance(right, type(left)) or len(left) != len(right):
            raise AssertionError(f"{path}: sequence mismatch")
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            _assert_nested_close(left_item, right_item, f"{path}[{index}]")
        return
    if left != right:
        raise AssertionError(f"{path}: {left!r} != {right!r}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-ids", default="0,1,2")
    parser.add_argument(
        "--result-root",
        default="results/mts_glt_distill_repair_control/smoke/resume_current_v2",
    )
    args = parser.parse_args()

    interrupted = (ROOT / args.result_root).resolve()
    continuous = interrupted.with_name(interrupted.name + "_continuous")
    if interrupted.exists() or continuous.exists():
        raise RuntimeError("resume smoke output already exists; choose a fresh --result-root")

    common = [
        "torchrun", "--standalone", "--nproc_per_node=3",
        "scripts/pretrain_mts_glt_distill.py",
        "--config", "configs/mts/glt_distill_repair_c0.json",
        "--stage", "student",
    ]
    _run(common + ["--stop-after", "2", "--result-root", str(interrupted)], args.gpu_ids)
    checkpoint_two = interrupted / "student/student_step_000002.pt"
    metrics_path = interrupted / "student/training_metrics.jsonl"
    with metrics_path.open("a") as handle:
        handle.write(json.dumps({"step": 3, "synthetic_abandoned_tail": True}) + "\n")
        handle.write(json.dumps({"step": 4, "synthetic_abandoned_tail": True}) + "\n")
    _run(
        common + [
            "--stop-after", "4", "--result-root", str(interrupted),
            "--resume", str(checkpoint_two),
        ],
        args.gpu_ids,
    )
    _run(common + ["--stop-after", "4", "--result-root", str(continuous)], args.gpu_ids)

    resumed = torch.load(
        interrupted / "student/student_step_000004.pt",
        map_location="cpu", weights_only=False,
    )
    uninterrupted = torch.load(
        continuous / "student/student_step_000004.pt",
        map_location="cpu", weights_only=False,
    )
    _assert_nested_close(resumed, uninterrupted)
    abandoned = interrupted / "student/training_metrics.abandoned_after_000002.jsonl"
    abandoned_rows = [json.loads(line) for line in abandoned.read_text().splitlines()]
    if [row.get("step") for row in abandoned_rows] != [3, 4] or not all(
        row.get("synthetic_abandoned_tail") is True for row in abandoned_rows
    ):
        raise AssertionError("resume did not preserve the abandoned log tail")
    retained_rows = [json.loads(line) for line in metrics_path.read_text().splitlines()]
    if [row.get("step") for row in retained_rows] != [1, 2, 3, 4] or any(
        row.get("synthetic_abandoned_tail") for row in retained_rows
    ):
        raise AssertionError("resume did not replace the active log tail")
    print(
        "RESUME_CURRENT_NUMERICAL_MATCH "
        f"step={resumed['step']} model_atol={MODEL_ATOL:g} "
        f"optimizer_atol={OPTIMIZER_ATOL:g} rng_exact=true abandoned_tail_preserved=true",
        flush=True,
    )


if __name__ == "__main__":
    main()
