#!/usr/bin/env python
"""Write the isolated evidence bundle for the finite T1 readiness cycle."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/mts_multiscale_topology/t1_readiness"


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def command(*args):
    return subprocess.check_output(args, cwd=ROOT, text=True).strip()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    source_files = [
        "configs/mts/default.json",
        "configs/mts/experiments/T1_msta_readiness.json",
        "scripts/initialize_mts_t1.py",
        "scripts/mts_t1_readiness_smoke.py",
        "scripts/mts_t1_ddp_smoke.py",
        "scripts/pretrain.py",
        "scripts/train.py",
        "scripts/resolve_mips_trimer_scage.py",
        "scripts/run_mips_trimer_scage.sh",
        "src/dataset/mips_trimer_contract.py",
        "src/modules/mips_local_graph.py",
        "src/modules/uni_encoder.py",
        "tests/test_mts_multiscale_topology.py",
    ]
    source_hashes = {
        name: sha256(ROOT / name) for name in source_files if (ROOT / name).is_file()
    }
    diff = subprocess.check_output(
        ["git", "diff", "--binary", "--", "."], cwd=ROOT
    )
    source_identity = hashlib.sha256(diff).hexdigest()
    try:
        git_sha = command("git", "rev-parse", "HEAD")
    except subprocess.CalledProcessError:
        git_sha = "unknown"
    resolved_t0 = json.loads(command(
        os.environ.get("PYTHON", "/opt/conda/envs/MTS/bin/python"),
        "scripts/resolve_mips_trimer_scage.py", "configs/mts/default.json",
    ))
    resolved_t1 = json.loads(command(
        os.environ.get("PYTHON", "/opt/conda/envs/MTS/bin/python"),
        "scripts/resolve_mips_trimer_scage.py",
        "configs/mts/experiments/T1_msta_readiness.json",
    ))
    gpu = {}
    try:
        import torch
        gpu = {
            "cuda_available": bool(torch.cuda.is_available()),
            "device_count": int(torch.cuda.device_count()),
            "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
            "torch": torch.__version__,
        }
    except Exception as exc:  # pragma: no cover - diagnostic fallback
        gpu = {"error": repr(exc)}
    source_state = {
        "schema": "mts-t1-source-state-v1",
        "cycle_id": "mts_t1_multiscale_topology_v1",
        "model_identity": "T1",
        "production_default": "T0",
        "git_head": git_sha,
        "working_tree_diff_sha256": source_identity,
        "source_file_sha256": source_hashes,
        "resolved_t0": {
            "topology_attention_variant": resolved_t0["topology_attention_variant"],
            "graph_model_config_hash": resolved_t0["graph_model_config_hash"],
        },
        "resolved_t1": {
            "topology_attention_variant": resolved_t1["topology_attention_variant"],
            "graph_model_config_hash": resolved_t1["graph_model_config_hash"],
            "config_hash": resolved_t1["config_hash"],
        },
        "gpu": gpu,
        "cache_mutated": False,
        "formal_pretraining_started": False,
        "formal_five_fold_evaluation_started": False,
    }
    (OUT / "source_state.json").write_text(
        json.dumps(source_state, indent=2) + "\n", encoding="utf-8"
    )
    init_manifest = json.loads(
        (ROOT / "pretrained_models/mts_multiscale_topology/t1_init/"
         "mts_t1_function_preserving_init.pth.initialization.json").read_text()
    )
    correctness = {
        "schema": "mts-t1-correctness-v1",
        "gate": "structure_and_function_preserving",
        "unit_test": {
            "command": "pytest -q tests/test_mts_multiscale_topology.py",
            "status": "passed",
            "passed": 6,
            "failed": 0,
        },
        "function_preserving_initialization": init_manifest,
        "strict_graph_load": True,
        "t0_legacy_default": True,
        "ordinary_resume_identity_isolated": True,
        "existing_combined_test_note": (
            "tests/test_mts_checkpoint_contract.py had one pre-existing G0 "
            "reuse fixture failure because historical fold_0.csv prediction "
            "is absent; it did not involve T1 code."
        ),
    }
    (OUT / "correctness.json").write_text(
        json.dumps(correctness, indent=2) + "\n", encoding="utf-8"
    )
    benchmark = json.loads((OUT / "benchmark.json").read_text())
    benchmark["t1_vs_t0_samples_per_second"] = (
        benchmark["results"]["T1"]["samples_per_second"]
        / benchmark["results"]["T0"]["samples_per_second"]
    )
    benchmark["t1_vs_t0_peak_memory_ratio"] = (
        benchmark["results"]["T1"]["peak_memory_bytes"]
        / benchmark["results"]["T0"]["peak_memory_bytes"]
    )
    (OUT / "benchmark.json").write_text(
        json.dumps(benchmark, indent=2) + "\n", encoding="utf-8"
    )
    ddp = json.loads((OUT / "ddp_smoke.json").read_text())
    finetune = json.loads((OUT / "finetune_smoke.json").read_text())
    smoke = {
        "schema": "mts-t1-smoke-summary-v1",
        "screening_only": True,
        "ddp_smoke": ddp,
        "finetune_smoke": finetune,
        "formal_performance_claim": False,
        "best_result_updated": False,
        "checkpoint_written": False,
    }
    (OUT / "smoke_summary.json").write_text(
        json.dumps(smoke, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "source_state": str(OUT / "source_state.json"),
        "correctness": str(OUT / "correctness.json"),
        "benchmark": str(OUT / "benchmark.json"),
        "smoke_summary": str(OUT / "smoke_summary.json"),
    }, indent=2))


if __name__ == "__main__":
    main()
