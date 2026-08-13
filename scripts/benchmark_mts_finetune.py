#!/usr/bin/env python3
"""Finite, resumable MTS fine-tuning speed benchmark.

The benchmark is deliberately bounded to four tasks, fold 0 and two epochs.
It measures training throughput from machine-readable ``perf_counter`` fields
written to each shard, while the evaluation-batch sweep is performed by one
training process so every batch size sees the same frozen fold model state.
This script never launches the formal 8-task x 5-fold campaign and never
changes cache/checkpoint artifacts or production defaults by itself.
"""

from __future__ import annotations

import argparse
import ast
import csv
import datetime as dt
import hashlib
import json
import math
import os
import re
import subprocess
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TASKS = ("egc", "egb", "eat", "xc")
GPU_IDS = ("0", "1", "2", "3")
FOLD_ID = 0
EPOCHS = 2
SCHEMA = "mts-finetune-benchmark-v2"


def _source_identity():
    files = (
        "scripts/benchmark_mts_finetune.py",
        "scripts/run_mips_trimer_scage.sh",
        "scripts/train.py",
        "src/utils.py",
        "src/dataset/dataloader.py",
        "src/dataset/dataset.py",
        "src/modules/uni_encoder.py",
        "src/modules/mips_local_graph.py",
    )
    hashes = {
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        for name in files
    }
    digest = hashlib.sha256()
    for name, value in hashes.items():
        digest.update(name.encode())
        digest.update(value.encode())
    return {"sha256": digest.hexdigest(), "files": hashes}


def _duration_seconds(value):
    parts = [int(item) for item in value.split(":")]
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    raise ValueError(value)


def _json_number(value, default=0.0):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return float(default)
    return value if math.isfinite(value) else float(default)


def _parse_shard_metrics(result_root, task):
    """Read timing from the one-fold machine-readable shard, if present."""

    path = result_root / "shards" / "42" / task / "fold_0.csv"
    if not path.is_file() or path.stat().st_size == 0:
        return None
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            row = next(csv.DictReader(handle))
        fold_metrics = json.loads(str(row.get("per_fold_metrics", "[]")))
        if not isinstance(fold_metrics, list):
            return None
        values = [item for item in fold_metrics if isinstance(item, dict)]
        steps = sum(int(item.get("training_steps", 0)) for item in values)
        seconds = sum(
            _json_number(item.get("training_seconds"), 0.0)
            for item in values
        )
        if steps <= 0 or seconds <= 0:
            return None
        return {
            "training_steps": int(steps),
            "training_seconds": float(seconds),
            "optimizer_steps_per_second": float(steps / seconds),
            "source": "shard_per_fold_metrics_perf_counter",
            "shard": str(path.relative_to(ROOT)),
        }
    except (OSError, StopIteration, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def _steady_training_metrics(log_root, tasks, result_root=None):
    """Return aggregate timing and BF16 parity, with a legacy log fallback.

    The fallback keeps the small unit test and old diagnostic logs readable;
    official benchmark runs always pass ``result_root`` and use shard timing.
    """

    total_steps = 0
    total_seconds = 0.0
    parity = {}
    per_task = {}
    for task in str(tasks).split():
        machine = (
            _parse_shard_metrics(Path(result_root), task)
            if result_root is not None else None
        )
        path = Path(log_root) / f"finetune_seed42_{task}_fold0.log"
        text = path.read_text(encoding="utf-8", errors="replace") if path.is_file() else ""
        if machine is None:
            steps = 0
            seconds = 0.0
            for left, right, duration in re.findall(
                r"Training:\s+100%.*?(\d+)/(\d+)\s+\[([0-9:]+)<", text
            ):
                if int(left) == int(right):
                    steps += int(right)
                    seconds += _duration_seconds(duration)
            if steps > 0 and seconds > 0:
                machine = {
                    "training_steps": int(steps),
                    "training_seconds": float(seconds),
                    "optimizer_steps_per_second": float(steps / seconds),
                    "source": "legacy_tqdm_log_fallback",
                }
        if machine is not None:
            per_task[task] = machine
            total_steps += int(machine["training_steps"])
            total_seconds += float(machine["training_seconds"])
        matches = re.findall(
            r"Fine-tune BF16 parity gate: (\{.*?\}), pass=(True|False)",
            text,
        )
        if matches:
            values, passed = matches[-1]
            try:
                parsed = ast.literal_eval(values)
            except (SyntaxError, ValueError):
                parsed = {"raw": values}
            parity[task] = {
                "metrics": parsed,
                "passed": passed == "True",
            }
    return {
        "steady_training_steps": int(total_steps),
        "steady_training_seconds": float(total_seconds),
        "steady_optimizer_steps_per_second": (
            float(total_steps / total_seconds) if total_seconds > 0 else None
        ),
        "per_task": per_task,
        "bf16_parity_gates": parity,
        "finite": bool(
            total_steps > 0
            and total_seconds > 0
            and math.isfinite(total_seconds)
            and all(
                math.isfinite(float(item["optimizer_steps_per_second"]))
                for item in per_task.values()
            )
        ),
    }


def _shm_available_bytes():
    try:
        stat = os.statvfs("/dev/shm")
        return int(stat.f_bavail * stat.f_frsize)
    except OSError:
        return None


def _cpu_snapshot():
    try:
        line = next(
            line for line in Path("/proc/stat").read_text().splitlines()
            if line.startswith("cpu ")
        )
        values = [int(value) for value in line.split()[1:]]
        idle = values[3] + (values[4] if len(values) > 4 else 0)
        return sum(values), idle
    except (OSError, StopIteration, ValueError, IndexError):
        return None


def _gpu_memory_bytes():
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used",
                "--format=csv,noheader,nounits",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=False,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return {}
    values = {}
    for line in completed.stdout.splitlines():
        try:
            gpu, memory = [item.strip() for item in line.split(",", 1)]
            values[str(gpu)] = int(float(memory)) * 1024 * 1024
        except (ValueError, IndexError):
            continue
    return values


def _monitor_snapshot():
    return {
        "timestamp": time.perf_counter(),
        "shm_available_bytes": _shm_available_bytes(),
        "cpu": _cpu_snapshot(),
        "gpu_memory_bytes": _gpu_memory_bytes(),
    }


def _update_monitor(monitor, previous, current):
    available = current.get("shm_available_bytes")
    if available is not None:
        monitor["shared_memory_minimum_available_bytes"] = min(
            monitor.get("shared_memory_minimum_available_bytes", available),
            available,
        )
    gpu_values = list(current.get("gpu_memory_bytes", {}).values())
    if gpu_values:
        monitor["peak_gpu_memory_bytes"] = max(
            monitor.get("peak_gpu_memory_bytes", 0), max(gpu_values)
        )
    before = previous.get("cpu") if previous else None
    after = current.get("cpu")
    if before and after:
        total_delta = after[0] - before[0]
        idle_delta = after[1] - before[1]
        if total_delta > 0:
            util = 100.0 * (1.0 - idle_delta / total_delta)
            monitor["cpu_utilization_peak_percent"] = max(
                monitor.get("cpu_utilization_peak_percent", 0.0), util
            )


def _mode_result_path(run_root, name):
    return Path(run_root) / name / "mode.json"


def _load_mode_result(run_root, name):
    path = _mode_result_path(run_root, name)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return None


def _run_mode(mode, run_root, epochs=EPOCHS, resume=False):
    name = mode["name"]
    prior = _load_mode_result(run_root, name) if resume else None
    if prior and int(prior.get("returncode", 1)) == 0:
        prior["reused"] = True
        return prior
    mode_root = Path(run_root) / name
    result_root = mode_root / "results"
    log_root = mode_root / "logs"
    result_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)
    eval_batches = tuple(int(value) for value in mode.get("eval_batches", ()))
    env = os.environ.copy()
    env.update({
        "PYTHONPATH": str(ROOT),
        "PYTHON_BIN": env.get("PYTHON_BIN", "/opt/conda/envs/MTS/bin/python"),
        "EXPERIMENT_CONFIG": "configs/mts/geometry_injection_ablation/A3_star_mcl_real.json",
        "FINETUNE_ONLY": "1",
        "PRETRAIN_ONLY": "0",
        "TASKS": " ".join(TASKS),
        "FOLD_IDS": str(FOLD_ID),
        "FINETUNE_SEEDS": "42",
        "MTS_FINETUNE_EPOCHS": str(int(epochs)),
        "MTS_FINETUNE_PATIENCE": str(int(epochs)),
        "MTS_ABLATION_SMOKE": "1",
        "MTS_FINETUNE_BATCH_SIZE": "32",
        "MTS_FINETUNE_EVAL_BATCH_SIZE": str(int(mode["eval_batch_size"])),
        "MTS_FINETUNE_AMP_DTYPE": str(mode["amp_dtype"]),
        "MTS_FINETUNE_GPU_IDS": ",".join(GPU_IDS),
        "MTS_FINETUNE_SCHEDULE": "lpt_v1",
        "DATALOADER_WORKERS": str(int(mode["workers"])),
        "FINETUNE_DATALOADER_WORKERS": str(int(mode["workers"])),
        "DATALOADER_PREFETCH_FACTOR": "2",
        "MTS_BENCHMARK_LEGACY_SYNC": "0",
        "RESULTS_DIR": str(result_root),
        "LOG_DIR": str(log_root),
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
    })
    if eval_batches:
        eval_root = mode_root / "eval_predictions"
        env["MTS_BENCHMARK_EVAL_BATCHES"] = ",".join(
            str(value) for value in eval_batches
        )
        env["MTS_BENCHMARK_EVAL_OUTPUT_DIR"] = str(eval_root)
    else:
        env.pop("MTS_BENCHMARK_EVAL_BATCHES", None)
        env.pop("MTS_BENCHMARK_EVAL_OUTPUT_DIR", None)
    log_path = log_root / "launcher.log"
    started = time.perf_counter()
    monitor = {
        "shared_memory_available_before_bytes": _shm_available_bytes(),
        "shared_memory_minimum_available_bytes": _shm_available_bytes(),
        "shared_memory_available_after_bytes": None,
        "peak_gpu_memory_bytes": 0,
        "cpu_utilization_peak_percent": 0.0,
        "poll_interval_seconds": 1.0,
    }
    previous = _monitor_snapshot()
    with log_path.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(
            ["bash", "scripts/run_mts.sh"],
            cwd=ROOT,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
        while process.poll() is None:
            time.sleep(1.0)
            current = _monitor_snapshot()
            _update_monitor(monitor, previous, current)
            previous = current
        returncode = int(process.wait())
    final_snapshot = _monitor_snapshot()
    _update_monitor(monitor, previous, final_snapshot)
    monitor["shared_memory_available_after_bytes"] = final_snapshot.get(
        "shm_available_bytes"
    )
    metrics = _steady_training_metrics(log_root, " ".join(TASKS), result_root)
    result = {
        **mode,
        "returncode": returncode,
        "wall_seconds": float(time.perf_counter() - started),
        "results_root": str(result_root.relative_to(ROOT)),
        "log": str(log_path.relative_to(ROOT)),
        "metrics": metrics,
        "monitor": monitor,
        "completed_shards": len(metrics["per_task"]),
    }
    mode_path = _mode_result_path(run_root, name)
    mode_path.parent.mkdir(parents=True, exist_ok=True)
    mode_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def _mode(name, workers, eval_batch_size, amp_dtype="fp32", eval_batches=()):
    return {
        "name": str(name),
        "amp_dtype": str(amp_dtype),
        "workers": int(workers),
        "train_batch_size": 32,
        "eval_batch_size": int(eval_batch_size),
        "prefetch_factor": 2,
        "physical_gpu_ids": ",".join(GPU_IDS),
        "tasks": list(TASKS),
        "fold_ids": [FOLD_ID],
        "epochs": EPOCHS,
        "eval_batches": list(eval_batches),
    }


def _baseline_mode():
    return _mode("fp32_workers0_eval64", 0, 64)


def _worker_mode(workers):
    return _mode(f"fp32_workers{int(workers)}_eval64", workers, 64)


def _eval_mode(workers):
    return _mode(
        f"eval_sweep_workers{int(workers)}",
        workers,
        64,
        eval_batches=(64, 128, 256),
    )


def _bf16_mode(workers, eval_batch_size):
    return _mode(
        f"bf16_workers{int(workers)}_eval{int(eval_batch_size)}",
        workers,
        eval_batch_size,
        amp_dtype="bf16",
    )


def _valid_training_result(result):
    if not result or int(result.get("returncode", 1)) != 0:
        return False
    metrics = result.get("metrics", {})
    rate = metrics.get("steady_optimizer_steps_per_second")
    return bool(metrics.get("finite") and rate is not None and float(rate) > 0)


def _select_worker(results):
    candidates = [
        result for result in results.values()
        if _valid_training_result(result)
        and result.get("amp_dtype") == "fp32"
        and int(result.get("eval_batch_size", -1)) == 64
        and int(result.get("workers", -1)) in (0, 2, 4, 6)
    ]
    if not candidates:
        return None, {"reason": "no_completed_fp32_worker_candidate"}
    maximum = max(
        float(item["metrics"]["steady_optimizer_steps_per_second"])
        for item in candidates
    )
    within_two_percent = [
        item for item in candidates
        if (maximum - float(item["metrics"]["steady_optimizer_steps_per_second"]))
        / maximum <= 0.02
    ]
    selected = min(
        within_two_percent,
        key=lambda item: (
            int(item["workers"]),
            float(item.get("wall_seconds", float("inf"))),
        ),
    )
    return int(selected["workers"]), {
        "maximum_rate": maximum,
        "within_two_percent_workers": sorted(
            int(item["workers"]) for item in within_two_percent
        ),
        "selected_workers": int(selected["workers"]),
        "selected_rate": float(
            selected["metrics"]["steady_optimizer_steps_per_second"]
        ),
        "rule": "highest aggregate optimizer steps/s; within 2% fewer workers",
    }


def _read_prediction(path):
    with np.load(path, allow_pickle=False) as payload:
        metadata = json.loads(str(np.asarray(payload["metadata"]).item()))
        return (
            np.asarray(payload["y_true"]),
            np.asarray(payload["y_pred"]),
            metadata,
        )


def _select_eval_batch(details):
    """Select the fastest safe eval batch from an already measured sweep.

    ``passed`` is set only when the candidate is available, uses the same
    fold-best model state, and has exact targets/allclose predictions.  Every
    task must also carry a finite ``eval_seconds`` measurement; otherwise the
    candidate cannot be promoted on speed evidence.
    """

    candidates = []
    for batch_name, payload in details.items():
        if not bool(payload.get("passed")):
            continue
        task_times = []
        valid_timing = True
        for task in TASKS:
            value = payload.get("tasks", {}).get(task, {}).get("eval_seconds")
            try:
                value = float(value)
            except (TypeError, ValueError):
                valid_timing = False
                break
            if not math.isfinite(value) or value < 0:
                valid_timing = False
                break
            task_times.append(value)
        if not valid_timing or len(task_times) != len(TASKS):
            continue
        candidates.append({
            "batch_size": int(batch_name),
            "total_eval_seconds": float(sum(task_times)),
        })
    if not candidates:
        return 64, {
            "candidate_totals": {},
            "rule": (
                "safe means available, same-state prediction compatible and "
                "finite timing; no safe timed candidate, fallback to 64"
            ),
            "tie_policy": "none",
        }
    selected = min(
        candidates,
        key=lambda item: (item["total_eval_seconds"], item["batch_size"]),
    )
    return int(selected["batch_size"]), {
        "candidate_totals": {
            str(item["batch_size"]): item["total_eval_seconds"]
            for item in sorted(candidates, key=lambda item: item["batch_size"])
        },
        "selected_total_eval_seconds": selected["total_eval_seconds"],
        "rule": (
            "among safe candidates choose minimum four-task eval_seconds "
            "sum; no tie band"
        ),
        "tie_policy": "none; exact minimum, then lower batch size",
    }


def _evaluate_batch_sweep(run_root, mode_result, workers):
    mode_root = Path(run_root) / mode_result["name"]
    eval_root = mode_root / "eval_predictions"
    details = {}
    passed_batches = []
    reference = None
    for batch_size in (64, 128, 256):
        per_task = {}
        batch_passed = True
        for task in TASKS:
            path = eval_root / f"batch_{batch_size}" / task / "fold_0.npz"
            if not path.is_file():
                batch_passed = False
                per_task[task] = {"available": False}
                continue
            try:
                y_true, y_pred, metadata = _read_prediction(path)
                state_ok = metadata.get("model_state_scope") == (
                    "same_train_and_evaluate_fold_best_model_state"
                )
                if reference is None:
                    reference = {}
                if task not in reference:
                    reference[task] = (y_true, y_pred)
                    target_equal = True
                    prediction_equal = True
                    maximum_delta = 0.0
                else:
                    ref_true, ref_pred = reference[task]
                    target_equal = bool(np.array_equal(ref_true, y_true))
                    prediction_equal = bool(
                        np.allclose(ref_pred, y_pred, rtol=0, atol=1e-5)
                    )
                    maximum_delta = float(np.max(np.abs(ref_pred - y_pred)))
                current_passed = bool(
                    state_ok and target_equal and prediction_equal
                )
                batch_passed = batch_passed and current_passed
                per_task[task] = {
                    "available": True,
                    "model_state_scope_valid": state_ok,
                    "target_equal": target_equal,
                    "prediction_allclose_rtol0_atol1e-5": prediction_equal,
                    "maximum_prediction_delta": maximum_delta,
                    "eval_seconds": metadata.get("eval_seconds"),
                    "path": str(path.relative_to(ROOT)),
                }
            except (OSError, ValueError, KeyError, TypeError):
                batch_passed = False
                per_task[task] = {"available": False, "malformed": True}
        details[str(batch_size)] = {
            "passed": bool(batch_passed),
            "tasks": per_task,
        }
        if batch_passed:
            passed_batches.append(batch_size)
    selected, selection = _select_eval_batch(details)
    return {
        "candidates": [64, 128, 256],
        "passed_batches": passed_batches,
        "selected": selected,
        "same_frozen_model_state": bool(passed_batches),
        "details": details,
        "selection": selection,
        "rule": "safe candidates require exact targets/allclose predictions; choose minimum four-task eval_seconds sum",
        "workers": int(workers),
    }


def _shard_gpu_rows(run_root, mode_name):
    rows = {}
    root = Path(run_root) / mode_name / "results" / "shards" / "42"
    for task in TASKS:
        path = root / task / "fold_0.csv"
        if not path.is_file():
            continue
        try:
            with path.open(newline="", encoding="utf-8") as handle:
                row = next(csv.DictReader(handle))
            rows[task] = {
                "physical_gpu_id": str(row.get("physical_gpu_id", "")),
                "amp_dtype": str(row.get("amp_dtype", "")),
                "eval_batch_size": int(row.get("eval_batch_size", -1)),
            }
        except (OSError, StopIteration, ValueError):
            continue
    return rows


def _new_state(run_root, args):
    return {
        "schema": SCHEMA,
        "status": "running",
        "scientific_scope": "two_epoch_speed_smoke_only",
        "run_id": args.run_id,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source_identity": _source_identity(),
        "fixed_config": {
            "tasks": list(TASKS),
            "fold_ids": [FOLD_ID],
            "epochs": EPOCHS,
            "train_batch_size": 32,
            "prefetch_factor": 2,
            "physical_gpu_ids": ",".join(GPU_IDS),
        },
        "results": {},
        "stages": {},
        "selection": {},
    }


def _save_state(run_root, state):
    state["updated_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    path = Path(run_root) / "benchmark.json"
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _load_or_create_state(run_root, args):
    path = Path(run_root) / "benchmark.json"
    if path.is_file():
        if not args.resume:
            raise SystemExit(
                f"refusing to reuse benchmark run root without --resume: {run_root}"
            )
        state = json.loads(path.read_text(encoding="utf-8"))
        if state.get("schema") != SCHEMA:
            raise SystemExit(f"benchmark schema mismatch in {path}")
        current_identity = _source_identity()
        if state.get("source_identity") != current_identity:
            state.setdefault("source_identity_history", []).append({
                "recorded_source_identity": state.get("source_identity"),
                "updated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                "reason": "BF16 dtype remediation rerun",
            })
            state["source_identity"] = current_identity
        return state
    if Path(run_root).exists() and any(Path(run_root).iterdir()) and not args.resume:
        raise SystemExit(
            f"refusing to reuse non-empty benchmark run root without --resume: {run_root}"
        )
    Path(run_root).mkdir(parents=True, exist_ok=True)
    return _new_state(run_root, args)


def _execute_mode(state, run_root, mode, args):
    result = _run_mode(
        mode,
        run_root,
        epochs=args.epochs,
        resume=args.resume,
    )
    state["results"][mode["name"]] = result
    _save_state(run_root, state)
    return result


def _ensure_baseline(state, run_root, args):
    result = state["results"].get(_baseline_mode()["name"])
    if not result or not _valid_training_result(result):
        result = _execute_mode(state, run_root, _baseline_mode(), args)
    return result


def _execute_baseline_and_workers(state, run_root, args):
    baseline = _ensure_baseline(state, run_root, args)
    for workers in (2, 4, 6):
        mode = _worker_mode(workers)
        result = state["results"].get(mode["name"])
        if not result or int(result.get("returncode", 1)) != 0:
            _execute_mode(state, run_root, mode, args)
    worker_results = {
        name: result for name, result in state["results"].items()
        if name.startswith("fp32_workers") and name.endswith("_eval64")
    }
    # The stage-A baseline is the workers=0 candidate; no duplicate writer is
    # started for it.
    worker_results[_baseline_mode()["name"]] = baseline
    selected, selection = _select_worker(worker_results)
    state["selection"]["workers"] = selection
    if selected is not None:
        state["selection"]["selected_workers"] = int(selected)
    state["stages"]["workers"] = {
        "status": "completed",
        "candidates": sorted(worker_results),
        "selected_workers": selected,
    }
    _save_state(run_root, state)
    return selected


def _require_workers(state):
    selected = state.get("selection", {}).get("selected_workers")
    if selected is None:
        raise SystemExit(
            "worker selection is missing; run --stage workers (or --stage all) first"
        )
    return int(selected)


def _execute_eval(state, run_root, args):
    workers = _require_workers(state)
    mode = _eval_mode(workers)
    result = state["results"].get(mode["name"])
    if not result or int(result.get("returncode", 1)) != 0:
        result = _execute_mode(state, run_root, mode, args)
    if int(result.get("returncode", 1)) == 0:
        gate = _evaluate_batch_sweep(run_root, result, workers)
    else:
        gate = {
            "candidates": [64, 128, 256],
            "passed_batches": [],
            "selected": 64,
            "same_frozen_model_state": False,
            "reason": "eval_sweep_mode_failed",
            "workers": workers,
        }
    state["selection"]["eval_batch"] = gate
    state["stages"]["eval"] = {
        "status": "completed" if result.get("returncode") == 0 else "completed_with_failure",
        "mode": mode["name"],
        "gate": gate,
    }
    _save_state(run_root, state)
    return int(gate["selected"])


def _execute_bf16(state, run_root, args):
    workers = _require_workers(state)
    eval_batch = int(state.get("selection", {}).get("eval_batch", {}).get("selected", 64))
    mode = _bf16_mode(workers, eval_batch)
    result = state["results"].get(mode["name"])
    if not result or int(result.get("returncode", 1)) != 0:
        # A later eval-only selection correction must not implicitly launch a
        # second BF16 training just because the selected eval batch changed.
        # Reuse the already completed BF16 smoke for this worker setting; a
        # fresh mode is created only when no completed BF16 result exists.
        completed_bf16 = [
            candidate for candidate in state["results"].values()
            if (
                candidate.get("amp_dtype") == "bf16"
                and int(candidate.get("workers", -1)) == workers
                and int(candidate.get("returncode", 1)) == 0
            )
        ]
        if completed_bf16:
            result = max(
                completed_bf16,
                key=lambda candidate: float(candidate.get("wall_seconds", 0.0)),
            )
        else:
            result = _execute_mode(state, run_root, mode, args)
    metrics = result.get("metrics", {})
    fp32 = None
    for candidate in state["results"].values():
        if (
            int(candidate.get("workers", -1)) == workers
            and candidate.get("amp_dtype") == "fp32"
            and int(candidate.get("eval_batch_size", -1)) == 64
        ):
            fp32 = candidate
            break
    fp32_rate = (
        fp32.get("metrics", {}).get("steady_optimizer_steps_per_second")
        if fp32 else None
    )
    bf16_rate = metrics.get("steady_optimizer_steps_per_second")
    speedup = (
        float(bf16_rate) / float(fp32_rate)
        if fp32_rate and bf16_rate else None
    )
    parity = metrics.get("bf16_parity_gates", {})
    egc_gate = parity.get("egc", {})
    egc_metrics = egc_gate.get("metrics", {}) if egc_gate else {}
    parity_passed = bool(
        egc_gate.get("passed")
        and egc_metrics.get("finite") is True
        and _json_number(egc_metrics.get("relative_loss_delta"), float("inf")) <= 0.02
    )
    gate = {
        "representative_task": "egc/fold0",
        "relative_loss_delta": egc_metrics.get("relative_loss_delta"),
        "finite": egc_metrics.get("finite"),
        "parity_passed": parity_passed,
        "minimum_speedup": 1.10,
        "measured_speedup": speedup,
        "speed_passed": bool(speedup is not None and speedup >= 1.10),
        "no_nan_or_oom": bool(
            int(result.get("returncode", 1)) == 0 and metrics.get("finite")
        ),
        "passed": bool(
            int(result.get("returncode", 1)) == 0
            and parity_passed
            and speedup is not None
            and speedup >= 1.10
            and metrics.get("finite")
        ),
        "parity_gates": parity,
    }
    state["selection"]["bf16"] = gate
    state["stages"]["bf16"] = {
        "status": "completed" if result.get("returncode") == 0 else "completed_with_failure",
        "mode": result.get("name", mode["name"]),
        "gate": gate,
    }
    _save_state(run_root, state)
    return gate


def _execute_smoke(state, run_root):
    workers = _require_workers(state)
    selected_name = f"fp32_workers{workers}_eval64"
    result = state["results"].get(selected_name)
    rows = _shard_gpu_rows(run_root, selected_name) if result else {}
    used = sorted({row["physical_gpu_id"] for row in rows.values()})
    smoke = {
        "mode": selected_name,
        "workers": workers,
        "tasks": rows,
        "used_physical_gpu_ids": used,
        "passed": bool(
            _valid_training_result(result)
            and set(used) == set(GPU_IDS)
            and len(rows) == len(TASKS)
        ),
        "note": "four-task fold0 two-epoch FP32 smoke is the selected worker candidate",
    }
    state["four_slot_smoke"] = smoke
    state["stages"]["smoke"] = {"status": "completed", **smoke}
    _save_state(run_root, state)
    return smoke


def _finalize(state, run_root):
    workers = state.get("selection", {}).get("selected_workers")
    eval_gate = state.get("selection", {}).get("eval_batch", {})
    bf16_gate = state.get("selection", {}).get("bf16", {})
    smoke = state.get("four_slot_smoke", {})
    selected_fp32_ok = workers is not None and _valid_training_result(
        state["results"].get(f"fp32_workers{workers}_eval64")
    )
    positive_worker_ok = any(
        _valid_training_result(result)
        and int(result.get("workers", 0)) > 0
        for result in state["results"].values()
        if result.get("amp_dtype") == "fp32"
    )
    core_complete = bool(
        _valid_training_result(state["results"].get(_baseline_mode()["name"]))
        and positive_worker_ok
        and selected_fp32_ok
        and smoke.get("passed")
    )
    amp = "bf16" if bf16_gate.get("passed") else "fp32"
    state["selection"]["production_defaults"] = {
        "FINETUNE_LOADER_WORKERS": int(workers) if workers is not None else 0,
        "MTS_FINETUNE_EVAL_BATCH_SIZE": int(eval_gate.get("selected", 64)),
        "MTS_FINETUNE_AMP_DTYPE": amp,
        "promotion_rule": "only completed finite candidates and BF16 requires parity plus >=10% speedup",
    }
    state["core_gates"] = {
        "baseline_completed": _valid_training_result(
            state["results"].get(_baseline_mode()["name"])
        ),
        "positive_worker_completed": positive_worker_ok,
        "selected_fp32_completed": selected_fp32_ok,
        "four_slot_smoke_passed": bool(smoke.get("passed")),
        "formal_training_not_run": True,
    }
    state["status"] = "complete" if core_complete else "completed_with_failures"
    state["completed_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    state["uncompleted"] = [] if core_complete else [
        key for key, value in state["core_gates"].items()
        if key != "formal_training_not_run" and not value
    ]
    _save_state(run_root, state)
    return core_complete


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument(
        "--run-id",
        default="worker_eval_amp_20260811",
        help="Isolated result directory name under results/mts_speed_optimization/finetune",
    )
    parser.add_argument(
        "--stage",
        choices=("all", "baseline", "workers", "eval", "bf16", "smoke"),
        default="all",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.epochs != EPOCHS:
        raise SystemExit("the speed-readiness benchmark is fixed to two epochs")
    if "/" in args.run_id or args.run_id in {"", ".", ".."}:
        raise SystemExit("--run-id must be a simple directory name")
    run_root = ROOT / "results" / "mts_speed_optimization" / "finetune" / args.run_id
    if args.dry_run:
        if run_root.exists() and not args.resume:
            raise SystemExit(f"refusing to reuse benchmark run root: {run_root}")
        state = _load_or_create_state(run_root, args)
        state["dry_run"] = True
        state["stage_plan"] = {
            "baseline": _baseline_mode(),
            "workers": [_worker_mode(value) for value in (0, 2, 4, 6)],
            "eval": "same-model-state eval batches 64,128,256",
            "bf16": "selected workers/eval batch, four tasks, fold0, two epochs",
            "smoke": "selected FP32 workers candidate",
        }
        _save_state(run_root, state)
        print(json.dumps(state, indent=2))
        return 0
    state = _load_or_create_state(run_root, args)
    if args.stage in {"all", "baseline"}:
        _ensure_baseline(state, run_root, args)
        state["stages"]["baseline"] = {
            "status": "completed",
            "mode": _baseline_mode()["name"],
        }
        _save_state(run_root, state)
        if args.stage == "baseline":
            return 0
    if args.stage in {"all", "workers"}:
        _execute_baseline_and_workers(state, run_root, args)
        if args.stage == "workers":
            return 0
    if args.stage in {"all", "eval"}:
        _execute_eval(state, run_root, args)
        if args.stage == "eval":
            return 0
    if args.stage in {"all", "bf16"}:
        _execute_bf16(state, run_root, args)
        if args.stage == "bf16":
            return 0
    if args.stage in {"all", "smoke"}:
        _execute_smoke(state, run_root)
        if args.stage == "smoke":
            return 0
    complete = _finalize(state, run_root)
    print(json.dumps(state, indent=2))
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
