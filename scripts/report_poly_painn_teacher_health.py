#!/usr/bin/env python3
"""Summarise PolyPaiNN formal-run health from immutable logs/checkpoints."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from src.modules.poly_painn_teacher import PolyPaiNNTeacher, load_teacher_deployment
from src.training.glt_dual_runtime import write_json


def _finite_tensors(value):
    if torch.is_tensor(value):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(_finite_tensors(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite_tensors(item) for item in value)
    return True


def _records(log_path):
    records = []
    with Path(log_path).open(encoding="utf-8") as stream:
        for line in stream:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("rank") == 0 and "step" in row:
                records.append(row)
    records.sort(key=lambda row: int(row["step"]))
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--log", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-steps", type=int, default=5000)
    args = parser.parse_args()
    run = Path(args.run).resolve()
    log = Path(args.log).resolve()
    output = Path(args.output).resolve()
    records = _records(log)
    by_step = {}
    duplicate = 0
    for row in records:
        step = int(row["step"])
        if step in by_step:
            duplicate += 1
        by_step[step] = row
    ordered = [by_step[step] for step in sorted(by_step)]
    expected = list(range(1, int(args.expected_steps) + 1))
    missing = [step for step in expected if step not in by_step]
    finite = all(np.isfinite([
        float(row["noise_mse"]), float(row["grad_norm_preclip"]),
        float(row["scalar_rms"]), float(row["vector_rms"]),
    ]).all() for row in ordered)
    last = ordered[-500:] if len(ordered) >= 500 else []
    last_noise = np.asarray([float(row["noise_mse"]) for row in last], dtype=np.float64)
    scalar = np.asarray([float(row["scalar_rms"]) for row in ordered], dtype=np.float64)
    vector = np.asarray([float(row["vector_rms"]) for row in ordered], dtype=np.float64)
    checkpoints = []
    checkpoint_finite = True
    for step in range(1000, int(args.expected_steps) + 1, 1000):
        path = run / f"resume_{step:05d}.pt"
        deployment = run / f"teacher_deploy_{step:05d}.pt"
        if not path.is_file() or not deployment.is_file():
            checkpoint_finite = False
            checkpoints.append({"step": step, "resume": str(path), "deployment": str(deployment), "status": "MISSING"})
            continue
        payload = torch.load(path, map_location="cpu", weights_only=False)
        package = torch.load(deployment, map_location="cpu", weights_only=False)
        ok = _finite_tensors(payload) and _finite_tensors(package)
        no_head = not any(str(name).startswith("noise_head.") for name in package.get("encoder", {}))
        ok = bool(ok and no_head and int(payload.get("step", -1)) == step
                  and int(package.get("step", -1)) == step)
        checkpoint_finite = checkpoint_finite and ok
        checkpoints.append({"step": step, "resume": str(path), "deployment": str(deployment),
                            "finite": bool(_finite_tensors(payload) and _finite_tensors(package)),
                            "deployment_excludes_noise_head": bool(no_head),
                            "status": "PASS" if ok else "FAIL"})
    status = "PASS" if (
        len(ordered) == int(args.expected_steps) and not missing and duplicate == 0
        and finite and checkpoint_finite and (float(scalar.max()) if len(scalar) else 0.0) > 1e-8
        and (float(vector.max()) if len(vector) else 0.0) > 1e-8 and len(last) == 500
        and np.isfinite(last_noise).all()
    ) else "INCOMPLETE_OR_FAIL"
    report = {
        "schema": "poly-painn-teacher-health-v1", "run": str(run), "log": str(log),
        "expected_steps": int(args.expected_steps), "observed_steps": len(ordered),
        "missing_steps": missing, "duplicate_step_records": int(duplicate),
        "all_logged_values_finite": bool(finite),
        "last500_noise_mse_mean": float(last_noise.mean()) if len(last) == 500 else None,
        "last500_noise_mse_std": float(last_noise.std(ddof=1)) if len(last) == 500 else None,
        "last500_stable_window": bool(len(last) == 500 and float(last_noise.std(ddof=1)) < max(1e-12, abs(float(last_noise.mean())) * 2.0)),
        "scalar_rms_min": float(scalar.min()) if len(scalar) else None,
        "scalar_rms_max": float(scalar.max()) if len(scalar) else None,
        "vector_rms_min": float(vector.min()) if len(vector) else None,
        "vector_rms_max": float(vector.max()) if len(vector) else None,
        "no_collapse_to_zero": bool(len(scalar) and scalar.max() > 1e-8 and vector.max() > 1e-8),
        "checkpoints": checkpoints, "checkpoint_values_finite": bool(checkpoint_finite),
        "status": status,
    }
    write_json(output, report)
    print(json.dumps(report, ensure_ascii=False), flush=True)
    if status != "PASS":
        raise RuntimeError("PolyPaiNN teacher health gate failed or is incomplete")


if __name__ == "__main__":
    main()
