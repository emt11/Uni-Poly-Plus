#!/usr/bin/env python3
"""Executor and contract gate for the MTS A0--A4 geometry ablation cycle.

The command deliberately has no pretraining path.  ``validate`` is read-only;
``prepare-random-mask`` writes only a new sidecar root; ``smoke`` writes the
isolated ``_smoke`` result root.  Formal 8-task x 5-fold runs are intentionally
not started by this cycle.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
PYTHON = os.environ.get("PYTHON_BIN", sys.executable)
CONFIG_ROOT = ROOT / "configs/mts/geometry_injection_ablation"
RESULT_ROOT = ROOT / "results/mts_geometry_injection_ablation_v1"
SMOKE_ROOT = RESULT_ROOT / "_smoke"
CHECKPOINT = ROOT / (
    "pretrained_models/mts/"
    "mts_joint_pretraining_pi1m_v2_seed42_canonical_angle20_v1.pth"
)
CHECKPOINT_SHA256 = (
    "56c8a0ba148280fef22ecb953705a14259fb5e87258c0cae67799d7484375e77"
)
EXPERIMENTS = (
    "A0_no3d_forward", "A1_star_only", "A2_mcl_real",
    "A3_star_mcl_real", "A4_star_mcl_random_mask",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve(config: Path) -> dict:
    command = [PYTHON, "scripts/resolve_mips_trimer_scage.py", str(config)]
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError(
            f"resolver rejected {config.name}: {result.stderr.strip() or result.stdout.strip()}"
        )
    return json.loads(result.stdout)


def _configs() -> list[Path]:
    paths = sorted(CONFIG_ROOT.glob("*.json"))
    expected = {f"{name}.json" for name in EXPERIMENTS}
    if {path.name for path in paths} != expected:
        raise RuntimeError(
            "active geometry-injection config directory must contain exactly "
            f"{sorted(expected)}"
        )
    stale = ROOT / "configs/mts/geometry_causal_ablation"
    if any(stale.glob("*.json")):
        raise RuntimeError("duplicate geometry_causal_ablation configs are active")
    return paths


def _cache_identity() -> dict:
    from scripts.audit_mips_trimer_cache import _specs

    specs = _specs(ROOT)
    result = {}
    for name in ("topology", "trimer"):
        root = Path(specs[name]["root"])
        result[name] = {
            "root": str(root),
            "done": (root / ".done").read_text().strip()
            if (root / ".done").is_file() else None,
            "frozen": (root / ".frozen").is_file(),
        }
    store = Path(specs["topology"]["root"]).parents[1] / "validation/store.json"
    result["store_sha256"] = _sha256(store) if store.is_file() else None
    return result


def _validate(verbose=True) -> dict:
    if not CHECKPOINT.is_file() or _sha256(CHECKPOINT) != CHECKPOINT_SHA256:
        raise RuntimeError("fixed A0-A4 checkpoint is missing or has the wrong SHA256")
    complete = json.loads(Path(str(CHECKPOINT) + ".complete.json").read_text())
    if complete.get("checkpoint_sha256") != CHECKPOINT_SHA256 or int(
        complete.get("optimizer_steps", -1)
    ) != 20000:
        raise RuntimeError("fixed checkpoint completion metadata is stale")
    cache = _cache_identity()
    if any(not cache[name]["frozen"] for name in ("topology", "trimer")):
        raise RuntimeError("A0-A4 requires frozen topology and Trimer artifacts")
    resolved = [_resolve(path) for path in _configs()]
    geometries = {item["geometry_model_config_hash"] for item in resolved}
    sources = {item["source_geometry_model_config_hash"] for item in resolved}
    if len(geometries) != 5 or len(sources) != 1:
        raise RuntimeError("A0-A4 geometry/source hash identity gate failed")
    if any(item.get("shared_checkpoint_sha256") != CHECKPOINT_SHA256 for item in resolved):
        raise RuntimeError("A0-A4 shared checkpoint identity gate failed")
    a4 = next(item for item in resolved if item["experiment_id"] == "A4_star_mcl_random_mask")
    sidecar = ROOT / a4["random_mask_sidecar"]
    if not sidecar.is_dir():
        raise RuntimeError(
            f"A4 sidecar is missing; run prepare-random-mask first: {sidecar}"
        )
    from src.dataset.mts_ablation_random_mask import AblationRandomMaskSidecar

    loader = AblationRandomMaskSidecar(sidecar)
    if int(loader.meta.get("record_count", -1)) != 3655:
        raise RuntimeError("A4 sidecar record_count must be 3655")
    payload = {
        "checkpoint": str(CHECKPOINT),
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "cache": cache,
        "experiments": [
            {
                "id": item["experiment_id"],
                "config_hash": item["config_hash"],
                "geometry_hash": item["geometry_model_config_hash"],
                "source_geometry_hash": item["source_geometry_model_config_hash"],
                "use_star_rbf": item["use_star_rbf"],
                "use_mcl": item["use_mcl"],
                "mcl_mask_mode": item["mcl_mask_mode"],
                "shared_checkpoint": item["shared_checkpoint"],
                "random_mask_sidecar": item["random_mask_sidecar"],
            }
            for item in resolved
        ],
        "a4_sidecar": str(sidecar),
    }
    if verbose:
        print(json.dumps(payload, indent=2, sort_keys=True))
    return payload


def _cohort_and_specs():
    from scripts.audit_mips_trimer_cache import _specs
    from src.dataset.lmdb_cache import load_cohort

    cohort_parent = ROOT / "data/processed/mips_trimer_scage/cohorts/downstream_union"
    current = json.loads((cohort_parent / "current.json").read_text())
    cohort_dir = cohort_parent / current["cohort_hash"]
    cohort = load_cohort(cohort_dir, load_text=False, verify_integrity=True)
    specs = _specs(ROOT)
    threshold_path = cohort_dir / "mcl_thresholds.npy"
    return cohort, specs, threshold_path


def _prepare_random_mask():
    from scripts.resolve_mips_trimer_scage import _random_mask_sidecar_path
    from src.dataset.mts_ablation_random_mask import build_random_mask_sidecar

    cohort, specs, threshold_path = _cohort_and_specs()
    target = _random_mask_sidecar_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_dir():
        try:
            from src.dataset.mts_ablation_random_mask import AblationRandomMaskSidecar
            AblationRandomMaskSidecar(target)
            print(f"[a4-mask] validated existing sidecar: {target}")
            return target
        except Exception as exc:
            print(f"[a4-mask] replacing incomplete sidecar {target}: {exc}")
            shutil.move(str(target), str(target) + ".invalid")
    output = build_random_mask_sidecar(
        cohort,
        Path(specs["trimer"]["root"]),
        threshold_path,
        target,
    )
    print(f"[a4-mask] built {output}")
    return output


def _smoke(experiments, task, fold, epochs):
    if not (1 <= int(epochs) <= 2):
        raise ValueError("smoke epochs must be 1 or 2")
    _validate(verbose=False)
    SMOKE_ROOT.mkdir(parents=True, exist_ok=True)
    for experiment in experiments:
        config = CONFIG_ROOT / f"{experiment}.json"
        result_root = SMOKE_ROOT / experiment
        log_root = ROOT / "logs/mts_geometry_injection_ablation_v1" / experiment
        result_root.mkdir(parents=True, exist_ok=True)
        log_root.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env.update({
            "EXPERIMENT_CONFIG": str(config),
            "FINETUNE_ONLY": "1",
            "TASKS": str(task),
            "FOLD_IDS": str(fold),
            "MTS_FINETUNE_EPOCHS": str(int(epochs)),
            "MTS_FINETUNE_PATIENCE": str(int(epochs)),
            "MTS_FINETUNE_BATCH_SIZE": "32",
            "MTS_ABLATION_SMOKE": "1",
            "RESULTS_DIR": str(result_root),
            "LOG_DIR": str(log_root),
            "CUDA_VISIBLE_DEVICES": "0,1,2",
            "DATALOADER_WORKERS": "0",
            "CACHE_WORKERS": "0",
            "MTS_RUN_MULTI_SEED": "0",
        })
        command = ["bash", "scripts/run_mips_trimer_scage.sh"]
        print(f"[smoke] starting {experiment}: {' '.join(command)}", flush=True)
        with (log_root / "launcher.log").open("a", encoding="utf-8") as log:
            result = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f"{experiment} smoke failed with status {result.returncode}")
        if list(result_root.glob("shards/**/*.csv")) == []:
            raise RuntimeError(f"{experiment} smoke produced no shard")
        print(f"[smoke] completed {experiment}", flush=True)
    print(f"[smoke] all experiments completed under {SMOKE_ROOT}")


def _status():
    payload = {
        "checkpoint_sha256": _sha256(CHECKPOINT) if CHECKPOINT.is_file() else None,
        "formal_results_exist": (RESULT_ROOT / "shards").is_dir(),
        "smoke_results": sorted(path.name for path in SMOKE_ROOT.glob("*") if path.is_dir()),
    }
    try:
        payload["validation"] = _validate(verbose=False)
    except Exception as exc:
        payload["validation_error"] = str(exc)
    print(json.dumps(payload, indent=2, sort_keys=True))


def main(argv=None):
    parser = argparse.ArgumentParser(prog="run_mts_geometry_injection_ablation.py")
    parser.add_argument("command", choices=(
        "validate", "prepare-random-mask", "smoke", "finetune",
        "summarize", "run-all", "status",
    ))
    parser.add_argument("--experiments", default=",".join(EXPERIMENTS))
    parser.add_argument("--task", default="eat")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=2)
    args = parser.parse_args(argv)
    if args.command == "validate":
        _validate()
        return 0
    if args.command == "prepare-random-mask":
        _prepare_random_mask()
        return 0
    if args.command == "status":
        _status()
        return 0
    if args.command == "smoke":
        requested = [value for value in args.experiments.split(",") if value]
        unknown = set(requested) - set(EXPERIMENTS)
        if unknown:
            raise SystemExit(f"unknown A0-A4 experiment(s): {sorted(unknown)}")
        _smoke(requested, args.task, args.fold, args.epochs)
        return 0
    raise SystemExit(
        f"{args.command} is reserved for a later Codex-approved cycle; "
        "this readiness cycle never starts formal ablation runs."
    )


if __name__ == "__main__":
    raise SystemExit(main())
