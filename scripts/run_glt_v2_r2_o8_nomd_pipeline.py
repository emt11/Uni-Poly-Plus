#!/usr/bin/env python3
"""Stage-explicit runner for GLT-V2 revision-2 pure O8/MIPS-CE baseline.

The route is intentionally independent from the historical O8+MD packages.
No teacher, GLT line sidecar, MD200 sidecar, or geometry cache build is
started by this entry point.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
CONFIG = "configs/mts/glt_v2_r2_o8_nomd_mipsloss_005k.json"
EXPERIMENT_ID = "glt_v2_r2_o8_nomd_mipsloss_005k"
RESULT_ROOT = ROOT / "results" / EXPERIMENT_ID
LOG_ROOT = ROOT / "logs" / EXPERIMENT_ID
CHECKPOINT = RESULT_ROOT / "student" / "student_deploy_005k.pt"


def run(command, *, log_path: Path | None = None):
    print("+", " ".join(str(value) for value in command), flush=True)
    if log_path is None:
        return subprocess.run([str(value) for value in command], cwd=ROOT, check=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        return subprocess.run(
            [str(value) for value in command], cwd=ROOT,
            stdout=handle, stderr=subprocess.STDOUT, check=True,
        )


def preflight():
    import json
    payload = json.loads((ROOT / CONFIG).read_text(encoding="utf-8"))
    if payload.get("schema") != "mts-glt-v2-r2-o8-nomd-mipsloss-v1":
        raise RuntimeError("no-MD config schema mismatch")
    if payload.get("use_md200") is not False or payload.get("version") != "none":
        raise RuntimeError("no-MD config unexpectedly enables MD/teacher")
    if CHECKPOINT.exists():
        import torch
        checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
        if (
            checkpoint.get("schema") != "mts-glt-v2-r2-o8-nomd-student-deploy-v1"
            or checkpoint.get("version") != "none"
            or int(checkpoint.get("step", -1)) != 5000
            or checkpoint.get("use_md200") is not False
        ):
            raise RuntimeError(f"existing no-MD checkpoint identity mismatch: {CHECKPOINT}")
        print(json.dumps({"config": str(ROOT / CONFIG), "result_root": str(RESULT_ROOT),
                          "checkpoint": str(CHECKPOINT), "status": "completed_valid"}, sort_keys=True))
        return
    print(json.dumps({"config": str(ROOT / CONFIG), "result_root": str(RESULT_ROOT),
                      "checkpoint": str(CHECKPOINT), "status": "preflight_ok"}, sort_keys=True))


def validate():
    tests = [
        "tests/test_mts_glt_distill.py",
        "tests/test_mts_student_architecture.py",
        "tests/test_mts_new_c0_contract.py",
        "tests/test_mts_nomd_route.py",
    ]
    run([sys.executable, "-m", "pytest", "-q", *tests])


def smoke():
    target = RESULT_ROOT / "smoke"
    if target.exists():
        raise FileExistsError(f"refusing to overwrite smoke output: {target}")
    run([
        "torchrun", "--standalone", "--nproc_per_node=3",
        "scripts/pretrain_mts_glt_distill.py", "--config", CONFIG,
        "--stage", "student", "--stop-after", "2",
        "--result-root", str(target),
    ], log_path=LOG_ROOT / "smoke.log")


def pretrain():
    preflight()
    run([
        "torchrun", "--standalone", "--nproc_per_node=3",
        "scripts/pretrain_mts_glt_distill.py", "--config", CONFIG,
        "--stage", "student",
    ], log_path=LOG_ROOT / "pretrain_005k.log")


def finetune(*, tasks=None, folds=None, gpu_ids="0,1,2,3", results_dir=None, logs_dir=None):
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(f"missing no-MD deployment: {CHECKPOINT}")
    result_dir = Path(results_dir) if results_dir else RESULT_ROOT / "downstream" / "outer5_inner20"
    log_dir = Path(logs_dir) if logs_dir else LOG_ROOT / "downstream" / "outer5_inner20"
    if result_dir.exists() and any(result_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite downstream output: {result_dir}")
    tasks = list(tasks or ["eat", "eea", "egb", "egc", "ei", "eps", "nc", "xc"])
    folds = [str(v) for v in (folds or [0, 1, 2, 3, 4])]
    run([
        sys.executable, "scripts/run_mts_finetune_scheduler.py",
        "--python", sys.executable, "--gpu-ids", gpu_ids,
        "--results-dir", str(result_dir), "--logs-dir", str(log_dir),
        "--tasks", *tasks, "--folds", *folds, "--seeds", "42",
        "--checkpoint", str(CHECKPOINT), "--checkpoint-seed", "42",
        "--checkpoint-tier", "student-5k", "--pretrain-dataset", "PI1M_v2",
        "--finetune-epochs", "100", "--finetune-patience", "10",
        "--batch-size", "32", "--eval-batch-size", "64", "--amp-dtype", "fp32",
        "--loader-workers", "2", "--evaluation-protocol", "outer5_inner20",
        "--train-args", "--config_schema", "mts-glt-v2-r2-nomd-downstream",
        "--experiment_id", f"{EXPERIMENT_ID}_outer5_inner20",
        "--split_manifest_dir", "data/splits/mips_outer5_inner20",
        "--graph_encoder_type", "mips_trimer_scage", "--graph_input", "star_linking",
        "--topology_attention_variant", "o8", "--no-use_star_rbf", "--no-use_mcl",
        "--no-use_md200", "--mts_glt_version", "distill_nomd",
        "--distill_repair_version", "none", "--mts_glt_mode", "o8_only",
        "--mips_norm_mode", "pre", "--target_transform", "standard",
        "--regression_loss", "mse", "--max_grad_norm", "1.0",
        "--weight_decay", "0.02", "--warmup_epochs", "5", "--head_dropout", "0.1",
        "--mts_o8_lr", "1e-5", "--graph_lr", "1e-5", "--head_lr", "1e-4",
        "--cache_layers", "ru_base,topology",
    ], log_path=log_dir / "scheduler.log")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("preflight", "validate", "smoke", "pretrain", "finetune"))
    parser.add_argument("--gpu-ids", default="0,1,2,3")
    parser.add_argument("--tasks", nargs="+")
    parser.add_argument("--folds", nargs="+", type=int)
    args = parser.parse_args(argv)
    if args.stage == "preflight":
        preflight()
    elif args.stage == "validate":
        validate()
    elif args.stage == "smoke":
        smoke()
    elif args.stage == "pretrain":
        pretrain()
    elif args.stage == "finetune":
        finetune(tasks=args.tasks, folds=args.folds, gpu_ids=args.gpu_ids)


if __name__ == "__main__":
    main()
