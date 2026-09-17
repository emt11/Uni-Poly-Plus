#!/usr/bin/env python3
"""Bounded three-GPU scheduler for one O8 attribution arm."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.finetune_glt_o8_control import TASKS, fixed_manifest


def validate_gpus(gpus):
    values = [str(value) for value in gpus]
    if len(values) != 3:
        raise ValueError("O8 control grid requires exactly three GPU slots")
    if len(set(values)) != len(values):
        raise ValueError("O8 control grid requires distinct GPU slots")
    return values


def build_command(args, task, fold, output):
    command = [
        sys.executable, "scripts/finetune_glt_o8_control.py",
        "--config", args.config, "--checkpoint", args.checkpoint,
        "--raw-root", args.raw_root, "--cohort-root", args.cohort_root,
        "--cache-root", args.cache_root, "--dual-static-root", args.dual_static_root,
        "--split-root", args.split_root, "--output", str(output), "--arm", args.arm,
        "--task", task, "--fold", str(fold), "--formal-shard",
        "--clean-cache-gib", format(float(args.clean_cache_gib), "g"),
    ]
    return command


def run_grid(args):
    gpus = validate_gpus(args.gpus)
    output = Path(args.output).resolve()
    logs = Path(args.log_root).resolve()
    if args.resume:
        if not output.is_dir():
            raise ValueError("--resume requires an existing output root")
    else:
        output.mkdir(parents=True, exist_ok=False)
    logs.mkdir(parents=True, exist_ok=True)
    tasks = list(args.tasks) if args.tasks else list(TASKS)
    if sorted(set(tasks)) != sorted(tasks) or any(task not in TASKS for task in tasks):
        raise ValueError("unknown or duplicate task")
    if args.clean_cache_gib < 0:
        raise ValueError("--clean-cache-gib must be non-negative")
    for task in tasks:
        fixed_manifest(task, Path(args.raw_root) / f"smi_{task}.csv",
                       Path(args.split_root) / f"{task}.json")
    jobs = [(task, fold) for task in tasks for fold in range(5)]
    if args.resume:
        pending = []
        for task, fold in jobs:
            unit = output / f"{task}_fold{fold}"
            if not unit.exists():
                pending.append((task, fold))
                continue
            summary = unit / "summary.json"
            if not summary.is_file():
                raise RuntimeError(f"partial O8 grid unit exists: {unit}")
            try:
                if not json.loads(summary.read_text()).get("tasks"):
                    raise ValueError
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                raise RuntimeError(f"partial O8 grid unit exists: {unit}") from exc
        jobs = pending
    free = list(gpus)
    active = {}
    failures = []
    completed = []
    while jobs or active:
        while jobs and free and not failures:
            task, fold = jobs.pop(0)
            gpu = free.pop(0)
            unit = output / f"{task}_fold{fold}"
            log_path = logs / f"{args.arm}_{task}_fold{fold}.log"
            command = build_command(args, task, fold, unit)
            handle = log_path.open("w", encoding="utf-8")
            handle.write("COMMAND=" + " ".join(command) + "\n")
            handle.write("CUDA_VISIBLE_DEVICES=" + gpu + "\n")
            handle.flush()
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
            process = subprocess.Popen(command, cwd=Path(__file__).resolve().parents[1],
                                       env=env, stdout=handle, stderr=subprocess.STDOUT)
            active[gpu] = (task, fold, process, handle, log_path)
        if not active:
            break
        changed = False
        for gpu, (task, fold, process, handle, log_path) in list(active.items()):
            code = process.poll()
            if code is None:
                continue
            handle.write(f"EXIT_CODE={code}\n")
            handle.close()
            del active[gpu]
            free.append(gpu)
            changed = True
            row = {"task": task, "fold": fold, "gpu": gpu,
                   "log": str(log_path), "exit_code": code}
            completed.append(row)
            if code != 0:
                failures.append(row)
        if failures:
            # Fail-stop: wait for already owned shards but never dispatch a new one.
            for gpu, (task, fold, process, handle, log_path) in list(active.items()):
                code = process.wait()
                handle.write(f"EXIT_CODE={code}\n")
                handle.close()
                completed.append({"task": task, "fold": fold, "gpu": gpu,
                                  "log": str(log_path), "exit_code": code})
                del active[gpu]
            raise RuntimeError("O8 grid shard failed: " + json.dumps(failures, sort_keys=True))
        if not changed:
            time.sleep(0.05)
    if jobs:
        raise RuntimeError("O8 grid stopped with pending jobs")
    print(json.dumps({"status": "PASS", "arm": args.arm,
                      "completed": completed, "count": len(completed)}, sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "checkpoint", "raw-root", "cohort-root", "cache-root", "dual-static-root", "output", "log-root"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--split-root", default="data/splits/mips_outer5_inner20")
    parser.add_argument("--arm", choices=("A", "B"), required=True)
    parser.add_argument("--task", action="append", dest="tasks")
    parser.add_argument("--gpu", action="append", dest="gpus", required=True)
    parser.add_argument("--clean-cache-gib", type=float, default=0.0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    from src.training.glt_dual_runtime import require_tmux
    require_tmux()
    run_grid(args)


if __name__ == "__main__":
    main()
