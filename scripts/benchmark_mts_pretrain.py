#!/usr/bin/env python3
"""Select a reproducible three-GPU MTS pretraining batch configuration.

The production objective is a global batch of 1008. This cycle confirms only
the two authorized finalists: (168, 2) and (336, 1)
samples/rank and gradient
accumulation steps.  This wrapper runs the existing finite DDP benchmark for
each candidate, records the raw result, and writes the selected configuration only
after all candidates pass the memory/finite checks.  It deliberately does not
start a 20k-step training job.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import datetime as dt
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CANDIDATES = ((168, 2), (336, 1))
WORKER_SWEEP_CANDIDATES = (4, 6, 8)


def _last_json(text: str):
    result = None
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("{"):
            continue
        try:
            value = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and "samples_per_second" in value:
            result = value
    return result


def _parse_worker_sweep(value: str):
    """Parse a comma-separated, non-negative, duplicate-free worker list."""
    if not isinstance(value, str) or not value.strip():
        raise SystemExit("--worker-sweep requires comma-separated integers")
    workers = []
    for token in value.split(","):
        token = token.strip()
        if not token:
            raise SystemExit(
                "--worker-sweep must contain only non-negative integers"
            )
        try:
            worker = int(token)
        except ValueError as exc:
            raise SystemExit(
                "--worker-sweep must contain only non-negative integers"
            ) from exc
        if worker < 0:
            raise SystemExit(
                "--worker-sweep must contain only non-negative integers"
            )
        if worker in workers:
            raise SystemExit("--worker-sweep values must be unique")
        workers.append(worker)
    return tuple(workers)


def _run_candidate(
    batch_size: int, accumulation: int, batches: int, workers: int,
    prefetch_factor: int, gpu_ids: str, log_prefix: str = "",
):
    log_dir = ROOT / "logs" / "mts_speed_optimization" / "pretrain"
    log_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    stem = f"b{batch_size}_a{accumulation}_w{workers}_p{prefetch_factor}"
    if log_prefix:
        stem = f"{log_prefix}_{stem}"
    env.update({
        "CUDA_VISIBLE_DEVICES": str(gpu_ids),
        "PRETRAIN_DATASET": "PI1M_v2",
        "PRETRAIN_BENCHMARK_ONLY": "1",
        "PRETRAIN_BENCHMARK_STAGE": "joint",
        "PRETRAIN_BENCHMARK_BATCHES": str(int(batches)),
        "EXPERIMENT_CONFIG": os.environ.get("EXPERIMENT_CONFIG", ""),
        "PRETRAIN_BATCH_SIZE": str(int(batch_size)),
        "PRETRAIN_ACCUMULATION": str(int(accumulation)),
        "DATALOADER_WORKERS": str(int(workers)),
        "DATALOADER_PREFETCH_FACTOR": str(int(prefetch_factor)),
        "MTS_PRETRAIN_GPU_IDS": str(gpu_ids),
        "LOG_DIR": str(log_dir / stem),
        "PRETRAIN_ONLY": "1",
        "STAGE3_ONLY": "0",
    })
    command = ["bash", "scripts/run_mts.sh"]
    log_path = log_dir / f"{stem}.log"
    shm_before = os.statvfs("/dev/shm").f_bavail * os.statvfs("/dev/shm").f_frsize
    shm_minimum = shm_before
    started = time.monotonic()
    started_at = dt.datetime.now(dt.timezone.utc).isoformat()
    with log_path.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=env,
            text=True,
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
        while process.poll() is None:
            available = (
                os.statvfs("/dev/shm").f_bavail
                * os.statvfs("/dev/shm").f_frsize
            )
            shm_minimum = min(shm_minimum, available)
            time.sleep(1.0)
        returncode = int(process.returncode)
    stdout = log_path.read_text(encoding="utf-8", errors="replace")
    result = _last_json(stdout)
    if result is None:
        result = {}
    result.update({
        "batch_size_per_rank": int(batch_size),
        "gradient_accumulation_steps": int(accumulation),
        "global_batch_size": int(batch_size) * 3 * int(accumulation),
        "loader_workers_requested": int(workers),
        "loader_prefetch_factor_requested": int(prefetch_factor),
        "physical_gpu_ids": str(gpu_ids),
        "started_at_utc": started_at,
        "finished_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "shared_memory_available_before_bytes": int(shm_before),
        "shared_memory_minimum_available_bytes": int(shm_minimum),
        "shared_memory_available_after_bytes": int(
            os.statvfs("/dev/shm").f_bavail * os.statvfs("/dev/shm").f_frsize
        ),
        "returncode": returncode,
        "wall_seconds": float(time.monotonic() - started),
    })
    return result


def _reusable_worker_result(
    benchmark_path: Path, config: str, batches: int, gpu_ids: str,
):
    """Return the compatible workers=4 result from the existing benchmark."""
    if not benchmark_path.is_file():
        return None
    try:
        payload = json.loads(benchmark_path.read_text(encoding="utf-8"))
        current_meta = _config_metadata(config)
    except (OSError, json.JSONDecodeError):
        return None
    if (
        payload.get("schema") != "mts-pretrain-benchmark-v2"
        or payload.get("dataset") != "PI1M_v2"
        or payload.get("stage") != "mts_joint_pretraining"
        or payload.get("physical_gpu_ids") != gpu_ids
        or int(payload.get("world_size", -1)) != 3
        or int(payload.get("target_global_batch", -1)) != 1008
        or int(payload.get("batches", -1)) != int(batches)
        or int(payload.get("warmup_batches", -1)) != 50
        or payload.get("config") != config
    ):
        return None
    prior_meta = payload.get("config_metadata", {})
    if (
        prior_meta.get("config_sha256") != current_meta.get("config_sha256")
        or prior_meta.get("gpu_visibility") != current_meta.get("gpu_visibility")
    ):
        return None
    # The benchmark wrapper itself changes in this cycle.  Reuse is still
    # valid when every runtime file used by the training subprocess is
    # unchanged; exclude only the wrapper from this compatibility comparison.
    prior_files = prior_meta.get("pretrain_code_files", {})
    current_files = current_meta.get("pretrain_code_files", {})
    for relative, digest in prior_files.items():
        if relative == "scripts/benchmark_mts_pretrain.py":
            continue
        if current_files.get(relative) != digest:
            return None
    for item in payload.get("results", []):
        if (
            int(item.get("batch_size_per_rank", -1)) == 336
            and int(item.get("gradient_accumulation_steps", -1)) == 1
            and int(item.get("loader_workers_requested", -1)) == 4
            and int(item.get("loader_prefetch_factor_requested", -1)) == 2
            and _passes_gate(item)
        ):
            result = dict(item)
            result["worker_sweep_source"] = "reused_existing_benchmark"
            result["worker_sweep_source_path"] = str(benchmark_path)
            return result
    return None


def _run_worker_sweep(args, output: Path, shm_bytes: int, workers):
    """Run only the fixed batch=336/accumulation=1 worker comparison."""
    if tuple(sorted(workers)) != WORKER_SWEEP_CANDIDATES:
        raise SystemExit(
            "--worker-sweep must contain exactly 4,6,8 for this benchmark"
        )
    if args.matrix or args.resume_json:
        raise SystemExit(
            "--worker-sweep cannot be combined with --matrix or --resume-json"
        )
    if args.batches != 500 or args.prefetch_factor != 2 or args.loader_workers != 0:
        raise SystemExit(
            "--worker-sweep is fixed to 50 warmup, 500 measured batches, "
            "workers=4,6,8 and prefetch_factor=2"
        )
    benchmark_path = ROOT / "results/mts_speed_optimization/pretrain/benchmark.json"
    results = []
    reused = _reusable_worker_result(
        benchmark_path, args.config, args.batches, args.gpu_ids
    )
    for worker in workers:
        if worker == 4 and reused is not None:
            print(
                "[mts-benchmark] reuse workers=4 from "
                f"{benchmark_path}",
                flush=True,
            )
            results.append(reused)
            continue
        print(f"[mts-benchmark] worker sweep candidate workers={worker}", flush=True)
        results.append(_run_candidate(
            336, 1, args.batches, worker, args.prefetch_factor, args.gpu_ids,
            log_prefix="worker_sweep_20260811",
        ))
        results[-1]["worker_sweep_source"] = "measured"

    valid = [item for item in results if _passes_gate(item)]
    selected = _select_candidate(valid) if valid else None
    baseline = next((item for item in results if int(
        item.get("loader_workers_requested", -1)
    ) == 4), None)
    speedups = {
        str(int(item["loader_workers_requested"])): (
            float(item.get("samples_per_second", 0.0))
            / max(float(baseline.get("samples_per_second", 0.0)), 1e-9)
            if baseline else None
        )
        for item in results
    }
    payload = {
        "schema": "mts-pretrain-worker-sweep-v1",
        "dataset": "PI1M_v2",
        "stage": "mts_joint_pretraining",
        "world_size": 3,
        "target_global_batch": 1008,
        "physical_gpu_ids": args.gpu_ids,
        "batch_size_per_rank": 336,
        "gradient_accumulation_steps": 1,
        "global_batch_size": 1008,
        "prefetch_factor": 2,
        "workers": list(workers),
        "shared_memory_bytes": int(shm_bytes),
        "config": str(args.config),
        "config_metadata": _config_metadata(args.config),
        "warmup_batches": 50,
        "measurement_batches": int(args.batches),
        "results": results,
        "speedup_vs_workers4": speedups,
        "all_candidates_passed": len(valid) == len(results),
        "selected": ({
            "workers": int(selected["loader_workers_requested"]),
            "prefetch_factor": int(selected["loader_prefetch_factor_requested"]),
            "samples_per_second": float(selected["samples_per_second"]),
            "peak_memory_fraction": float(selected.get("peak_memory_fraction", 0.0)),
        } if selected else None),
    }
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if not valid:
        raise SystemExit("No worker-sweep candidate passed the finite/memory gate")
    return 0


def _passes_gate(item):
    return bool(
        item.get("returncode") == 0
        and item.get("loss_finite", False)
        and item.get("gradient_finite", False)
        and item.get("parameters_finite", False)
        and float(item.get("peak_memory_fraction", 1.0)) < 0.80
        and float(item.get("samples_per_second", 0.0)) > 0.0
        and float(item.get("rank_wait_fraction", 1.0)) < 0.15
        and int(item.get("shared_memory_minimum_available_bytes", 0)) > 0
    )


def _select_candidate(items):
    """Select by throughput, then resource use within the 2% tie band."""
    best_speed = max(float(item["samples_per_second"]) for item in items)
    near = [
        item for item in items
        if float(item["samples_per_second"]) >= best_speed * 0.98
    ]
    return min(
        near,
        key=lambda item: (
            float(item.get("peak_memory_fraction", 1.0)),
            int(item.get("shared_memory_available_before_bytes", 0))
            - int(item.get("shared_memory_minimum_available_bytes", 0)),
            int(item.get("loader_workers_requested", 0)),
            int(item.get("loader_prefetch_factor_requested", 2)),
            -float(item["samples_per_second"]),
        ),
    )


def _config_metadata(path):
    config_path = ROOT / path
    config = json.loads(config_path.read_text(encoding="utf-8"))
    digest = __import__("hashlib").sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    code_files = [
        "scripts/resolve_mips_trimer_scage.py",
        "scripts/benchmark_mts_pretrain.py", "scripts/pretrain.py",
        "scripts/run_mips_trimer_scage.sh", "src/dataset/dataloader.py",
        "src/dataset/dataset.py", "src/dataset/lmdb_cache.py",
        "src/dataset/mips_trimer_contract.py", "src/dataset/trimer_mcl.py",
        "src/modules/mips_local_graph.py", "src/modules/uni_encoder.py",
        "src/utils.py",
    ]
    file_hashes = {}
    code_digest = __import__("hashlib").sha256()
    for relative in code_files:
        candidate = ROOT / relative
        if not candidate.is_file():
            continue
        value = __import__("hashlib").sha256(candidate.read_bytes()).hexdigest()
        file_hashes[relative] = value
        code_digest.update(relative.encode())
        code_digest.update(value.encode())
    try:
        import torch
        torch_version = torch.__version__
        cuda_version = torch.version.cuda
        gpu_names = [
            torch.cuda.get_device_name(index)
            for index in range(torch.cuda.device_count())
        ]
    except Exception:
        torch_version = None
        cuda_version = None
        gpu_names = []
    return {
        "config_sha256": digest,
        "config": config,
        "pretrain_code_sha256": code_digest.hexdigest(),
        "pretrain_code_files": file_hashes,
        "torch_version": torch_version,
        "cuda_version": cuda_version,
        "gpu_names": gpu_names,
        "cpu_count": os.cpu_count(),
        "host_memory_bytes": int(
            os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        ),
        "gpu_visibility": "1,2,3",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batches", type=int, default=500)
    parser.add_argument(
        "--config",
        default="",
    )
    parser.add_argument("--loader-workers", type=int, default=0)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--gpu-ids", default="1,2,3")
    parser.add_argument(
        "--matrix", action="store_true",
        help="Run the staged workers/prefetch/batch selection matrix.",
    )
    parser.add_argument(
        "--resume-json", default="",
        help="Reuse completed candidates from an earlier v2 benchmark JSON.",
    )
    parser.add_argument(
        "--worker-sweep", default="",
        help="Run the fixed batch=336 worker comparison for 4,6,8 workers.",
    )
    parser.add_argument(
        "--output",
        default="results/mts_speed_optimization/pretrain/benchmark.json",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if not args.config:
        raise SystemExit(
            "No active MTS configuration; pass --config after defining the next schema."
        )
    config_path = ROOT / args.config
    if not config_path.is_file():
        raise SystemExit(f"MTS configuration does not exist: {config_path}")
    os.environ["EXPERIMENT_CONFIG"] = str(config_path)
    if args.batches <= 0:
        raise SystemExit("--batches must be positive")
    gpu_ids = [value.strip() for value in args.gpu_ids.split(",") if value.strip()]
    if gpu_ids != ["1", "2", "3"]:
        raise SystemExit("MTS pretraining benchmark must use physical GPUs 1,2,3")
    if args.loader_workers < 0 or args.prefetch_factor <= 0:
        raise SystemExit("loader workers must be non-negative and prefetch positive")
    worker_sweep = _parse_worker_sweep(args.worker_sweep) if args.worker_sweep else None
    shm_bytes = os.statvfs("/dev/shm").f_frsize * os.statvfs("/dev/shm").f_blocks
    if args.loader_workers > 0 and shm_bytes < 8 * 1024**3:
        raise SystemExit(
            "loader_workers>0 requires /dev/shm >= 8 GiB; use workers=0 on "
            f"the current {shm_bytes / 1024**2:.0f} MiB shared-memory mount"
        )
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        if worker_sweep is not None:
            if tuple(sorted(worker_sweep)) != WORKER_SWEEP_CANDIDATES:
                raise SystemExit(
                    "--worker-sweep must contain exactly 4,6,8 for this benchmark"
                )
            payload = {
                "schema": "mts-pretrain-worker-sweep-v1",
                "world_size": 3,
                "target_global_batch": 1008,
                "physical_gpu_ids": args.gpu_ids,
                "batch_size_per_rank": 336,
                "gradient_accumulation_steps": 1,
                "global_batch_size": 1008,
                "warmup_batches": 50,
                "measurement_batches": int(args.batches),
                "prefetch_factor": 2,
                "workers": list(worker_sweep),
                "selected": None,
                "config": str(args.config),
            }
            output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
            print(json.dumps(payload, indent=2))
            return 0
        payload = {
            "schema": "mts-pretrain-benchmark-v2",
            "world_size": 3,
            "target_global_batch": 1008,
            "physical_gpu_ids": args.gpu_ids,
            "warmup_batches": 50,
            "measurement_batches": int(args.batches),
            "candidates": [
                {"batch_size_per_rank": b, "gradient_accumulation_steps": a,
                 "global_batch_size": b * 3 * a}
                for b, a in CANDIDATES
            ],
            "selected": None,
            "config": str(args.config),
        }
        output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(payload, indent=2))
        return 0

    if worker_sweep is not None:
        return _run_worker_sweep(args, output, shm_bytes, worker_sweep)

    results = []
    if args.resume_json:
        prior = json.loads(Path(args.resume_json).read_text(encoding="utf-8"))
        if (
            prior.get("schema") != "mts-pretrain-benchmark-v2"
            or prior.get("physical_gpu_ids") != args.gpu_ids
            or int(prior.get("batches", -1)) != int(args.batches)
        ):
            raise SystemExit("resume benchmark JSON does not match this run")
        results.extend(prior.get("results", []))

    def run(batch_size, accumulation, workers, prefetch_factor):
        if any(
            int(item.get("batch_size_per_rank", -1)) == int(batch_size)
            and int(item.get("gradient_accumulation_steps", -1)) == int(accumulation)
            and int(item.get("loader_workers_requested", -1)) == int(workers)
            and int(item.get("loader_prefetch_factor_requested", -1))
            == int(prefetch_factor)
            and _passes_gate(item)
            for item in results
        ):
            print(
                "[mts-benchmark] reuse candidate "
                f"batch={batch_size} accumulation={accumulation} "
                f"workers={workers} prefetch={prefetch_factor}",
                flush=True,
            )
            return
        print(
            "[mts-benchmark] candidate "
            f"batch={batch_size} accumulation={accumulation} "
            f"workers={workers} prefetch={prefetch_factor}",
            flush=True,
        )
        results.append(_run_candidate(
            batch_size, accumulation, args.batches, workers,
            prefetch_factor, args.gpu_ids,
        ))

    if args.matrix:
        for workers in (0, 1, 2, 4):
            run(168, 2, workers, 2)
        worker_valid = [
            item for item in results
            if _passes_gate(item)
            and int(item.get("batch_size_per_rank", -1)) == 168
            and int(item.get("loader_prefetch_factor_requested", -1)) == 2
        ]
        if not worker_valid:
            raise SystemExit("No loader-worker candidate completed")
        best_worker = _select_candidate(worker_valid)["loader_workers_requested"]
        if int(best_worker) > 0:
            run(168, 2, int(best_worker), 4)
        data_valid = [
            item for item in results
            if _passes_gate(item)
            and int(item.get("loader_workers_requested", -1)) == int(best_worker)
        ]
        best_data = _select_candidate(data_valid)
        best_prefetch = int(best_data["loader_prefetch_factor_requested"])
        run(336, 1, int(best_worker), best_prefetch)
    else:
        for batch_size, accumulation in CANDIDATES:
            run(
                batch_size, accumulation, args.loader_workers,
                args.prefetch_factor,
            )

    valid = [
        item for item in results
        if _passes_gate(item)
    ]
    if not valid:
        raise SystemExit(
            "No MTS batch candidate passed the finite/memory benchmark; "
            "do not start the 20k pretraining run."
        )
    selected = _select_candidate(valid)
    baseline = next((
        item for item in results
        if int(item.get("batch_size_per_rank", -1)) == 168
        and int(item.get("gradient_accumulation_steps", -1)) == 2
        and int(item.get("loader_workers_requested", -1)) == 0
        and int(item.get("loader_prefetch_factor_requested", -1)) == 2
    ), None)
    speedup = (
        float(selected["samples_per_second"])
        / max(float(baseline["samples_per_second"]), 1e-9)
        if baseline else None
    )
    promoted = bool(speedup is not None and speedup >= 1.15)
    payload = {
        "schema": "mts-pretrain-benchmark-v2",
        "dataset": "PI1M_v2",
        "stage": "mts_joint_pretraining",
        "world_size": 3,
        "target_global_batch": 1008,
        "physical_gpu_ids": args.gpu_ids,
        "shared_memory_bytes": int(shm_bytes),
        "config": str(args.config),
        "config_metadata": _config_metadata(args.config),
        "batches": int(args.batches),
        "warmup_batches": 50,
        "results": results,
        "baseline": baseline,
        "promotion_gate": {
            "minimum_speedup": 1.15,
            "measured_speedup": speedup,
            "passed": promoted,
        },
        "selected": ({
            "batch_size_per_rank": int(selected["batch_size_per_rank"]),
            "gradient_accumulation_steps": int(
                selected["gradient_accumulation_steps"]
            ),
            "global_batch_size": 1008,
            "loader_workers": int(selected["loader_workers_requested"]),
            "loader_prefetch_factor": int(
                selected["loader_prefetch_factor_requested"]
            ),
            "samples_per_second": float(selected["samples_per_second"]),
            "peak_memory_fraction": float(selected.get("peak_memory_fraction", 0.0)),
            "training_mode": "uninterrupted",
        } if promoted else None),
        "best_measured": {
            "batch_size_per_rank": int(selected["batch_size_per_rank"]),
            "gradient_accumulation_steps": int(selected["gradient_accumulation_steps"]),
            "loader_workers": int(selected["loader_workers_requested"]),
            "loader_prefetch_factor": int(selected["loader_prefetch_factor_requested"]),
            "samples_per_second": float(selected["samples_per_second"]),
        },
    }
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
