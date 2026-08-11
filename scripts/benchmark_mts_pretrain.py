#!/usr/bin/env python3
"""Select the reproducible three-GPU MTS pretraining batch profile.

The production objective is a global batch of 1008. This cycle confirms only
the two authorized finalists: (168, 2) and (336, 1)
samples/rank and gradient
accumulation steps.  This wrapper runs the existing finite DDP benchmark for
each candidate, records the raw result, and writes the selected profile only
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
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CANDIDATES = ((168, 2), (336, 1))


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


def _run_candidate(batch_size: int, accumulation: int, batches: int, workers: int):
    log_dir = ROOT / "logs" / "mts" / "benchmark"
    log_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update({
        "CUDA_VISIBLE_DEVICES": "0,1,2",
        "PRETRAIN_DATASET": "PI1M_v2",
        "PRETRAIN_BENCHMARK_ONLY": "1",
        "PRETRAIN_BENCHMARK_STAGE": "joint",
        "PRETRAIN_BENCHMARK_BATCHES": str(int(batches)),
        "PRETRAIN_PROFILE": "canonical_ru_angle20_v1",
        "PRETRAIN_BATCH_SIZE": str(int(batch_size)),
        "PRETRAIN_ACCUMULATION": str(int(accumulation)),
        "DATALOADER_WORKERS": str(int(workers)),
        "LOG_DIR": str(log_dir / f"b{batch_size}_a{accumulation}"),
        "PRETRAIN_ONLY": "1",
        "STAGE3_ONLY": "0",
    })
    command = ["bash", "scripts/run_mts.sh"]
    started = time.monotonic()
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    result = _last_json(completed.stdout)
    if result is None:
        result = {}
    result.update({
        "batch_size_per_rank": int(batch_size),
        "gradient_accumulation_steps": int(accumulation),
        "global_batch_size": int(batch_size) * 3 * int(accumulation),
        "loader_workers_requested": int(workers),
        "returncode": int(completed.returncode),
        "wall_seconds": float(time.monotonic() - started),
    })
    (log_dir / f"b{batch_size}_a{accumulation}.log").write_text(
        completed.stdout, encoding="utf-8"
    )
    return result


def _profile_metadata(path):
    profile_path = ROOT / path
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    digest = __import__("hashlib").sha256(
        json.dumps(profile, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    code_files = [
        "configs/mts/default.json", "configs/mts/pretraining/canonical_ru_angle20_v1.json",
        "scripts/pretrain.py", "scripts/run_mips_trimer_scage.sh", "src/dataset/dataloader.py",
        "src/dataset/dataset.py", "src/dataset/mips_trimer_contract.py", "src/dataset/trimer_mcl.py",
        "src/modules/mips_local_graph.py", "src/modules/uni_encoder.py",
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
    except Exception:
        torch_version = None
        cuda_version = None
    return {
        "profile_id": profile.get("profile_id"),
        "profile_sha256": digest,
        "profile": profile,
        "pretrain_code_sha256": code_digest.hexdigest(),
        "pretrain_code_files": file_hashes,
        "torch_version": torch_version,
        "cuda_version": cuda_version,
        "gpu_visibility": "0,1,2",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batches", type=int, default=500)
    parser.add_argument(
        "--profile",
        default="configs/mts/pretraining/canonical_ru_angle20_v1.json",
    )
    parser.add_argument("--loader-workers", type=int, default=0)
    parser.add_argument(
        "--output",
        default="results/mips_trimer_scage/pretrain_benchmark.json",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.batches <= 0:
        raise SystemExit("--batches must be positive")
    if os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2") != "0,1,2":
        raise SystemExit("MTS benchmark must expose exactly CUDA devices 0,1,2")
    shm_bytes = os.statvfs("/dev/shm").f_frsize * os.statvfs("/dev/shm").f_blocks
    if args.loader_workers > 0 and shm_bytes < 8 * 1024**3:
        raise SystemExit(
            "loader_workers>0 requires /dev/shm >= 8 GiB; use workers=0 on "
            f"the current {shm_bytes / 1024**2:.0f} MiB shared-memory mount"
        )
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        payload = {
            "schema": "mts-pretrain-benchmark-v1",
            "candidates": [
                {"batch_size_per_rank": b, "gradient_accumulation_steps": a,
                 "global_batch_size": b * 3 * a}
                for b, a in CANDIDATES
            ],
            "selected": None,
            "profile": str(args.profile),
        }
        output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(payload, indent=2))
        return 0

    results = []
    for batch_size, accumulation in CANDIDATES:
        print(
            f"[mts-benchmark] candidate batch={batch_size} accumulation={accumulation}",
            flush=True,
        )
        results.append(_run_candidate(
            batch_size, accumulation, args.batches, args.loader_workers
        ))

    valid = [
        item for item in results
        if item.get("returncode") == 0
        and float(item.get("peak_memory_fraction", 1.0)) < 0.90
        and float(item.get("samples_per_second", 0.0)) > 0.0
        and float(item.get("rank_wait_fraction", 1.0)) < 0.15
    ]
    if not valid:
        raise SystemExit(
            "No MTS batch candidate passed the finite/memory benchmark; "
            "do not start the 20k pretraining run."
        )
    # Throughput is primary.  Within 2%, prefer the lower reserved memory.
    best_speed = max(float(item["samples_per_second"]) for item in valid)
    near = [
        item for item in valid
        if float(item["samples_per_second"]) >= best_speed * 0.98
    ]
    selected = min(
        near,
        key=lambda item: (
            float(item.get("peak_memory_fraction", 1.0)),
            -float(item["samples_per_second"]),
        ),
    )
    payload = {
        "schema": "mts-pretrain-benchmark-v1",
        "dataset": "PI1M_v2",
        "stage": "mts_joint_pretraining",
        "world_size": 3,
        "target_global_batch": 1008,
        "profile": str(args.profile),
        "profile_metadata": _profile_metadata(args.profile),
        "batches": int(args.batches),
        "results": results,
        "selected": {
            "batch_size_per_rank": int(selected["batch_size_per_rank"]),
            "gradient_accumulation_steps": int(
                selected["gradient_accumulation_steps"]
            ),
            "global_batch_size": 1008,
            "samples_per_second": float(selected["samples_per_second"]),
            "peak_memory_fraction": float(selected.get("peak_memory_fraction", 0.0)),
            "training_mode": "uninterrupted",
        },
    }
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
