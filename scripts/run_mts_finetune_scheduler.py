#!/usr/bin/env python3
"""CLI bridge from the MTS launcher to the real Python fold scheduler."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training.finetune.scheduler import (  # noqa: E402
    ScheduledUnit,
    dispatch_units,
    run_subprocess_scheduler,
)
from src.training.finetune.config import parse_arguments as parse_finetune_arguments  # noqa: E402


SCIENTIFIC_FLAGS = {
    "--amp_dtype", "--batch_size", "--eval_batch_size", "--epochs",
    "--evaluation_protocol", "--graph_lr", "--head_dropout", "--head_lr",
    "--huber_beta", "--loader_workers", "--max_grad_norm",
    "--mts_adapter_lr", "--mts_geometry_lr", "--mts_glt_fusion_warm_epochs",
    "--mts_glt_fusion_strategy", "--mts_glt_initial_alpha", "--mts_o8_lr",
    "--patience", "--regression_loss", "--target_transform",
    "--warmup_epochs", "--weight_decay", "--fusion_lr",
}

RESOLVED_FIELDS = {
    "model_mode": "mts_glt_mode",
    "checkpoint": "pretrained_model_path",
    "checkpoint_tier": "checkpoint_tier",
    "target_transform": "target_transform",
    "loss": "regression_loss",
    "beta": "huber_beta",
    "grad_clip": "max_grad_norm",
    "weight_decay": "weight_decay",
    "configured_encoder_freeze_warm_epochs": "mts_glt_fusion_warm_epochs",
    "fusion_strategy": "mts_glt_fusion_strategy",
    "initial_alpha": "mts_glt_initial_alpha",
    "lr_warmup_epochs": "warmup_epochs",
    "head_dropout": "head_dropout",
    "o8_lr": "mts_o8_lr",
    "glt_lr": "mts_geometry_lr",
    "fusion_lr": "fusion_lr",
    "md200_lr": "mts_adapter_lr",
    "graph_adapter_lr": "mts_adapter_lr",
    "head_lr": "head_lr",
    "epochs": "epochs",
    "patience": "patience",
    "train_batch": "batch_size",
    "eval_batch": "eval_batch_size",
    "precision": "amp_dtype",
    "workers": "loader_workers",
}


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def _atomic_json(path: Path, payload: dict) -> None:
    _atomic_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_provenance() -> dict:
    def run(*command):
        return subprocess.run(
            command, cwd=ROOT, check=True, text=True, capture_output=True
        ).stdout.rstrip("\n")
    status = run("git", "status", "--porcelain")
    return {
        "git_commit": run("git", "rev-parse", "HEAD"),
        "working_tree_dirty": bool(status),
        "git_status_porcelain": status,
        "git_diff_stat": run("git", "diff", "--stat"),
    }


def _flag_values(arguments: list[str]) -> dict[str, list[str | None]]:
    values: dict[str, list[str | None]] = {}
    index = 0
    while index < len(arguments):
        token = arguments[index]
        if token.startswith("--"):
            flag, separator, inline = token.partition("=")
            value = inline if separator else None
            if not separator and index + 1 < len(arguments) and not arguments[index + 1].startswith("--"):
                value = arguments[index + 1]
            values.setdefault(flag, []).append(value)
        index += 1
    return values


def assert_no_duplicate_scientific_flags(arguments: list[str]) -> None:
    duplicated = {
        flag: values for flag, values in _flag_values(arguments).items()
        if flag in SCIENTIFIC_FLAGS and len(values) > 1
    }
    if duplicated:
        raise ValueError(f"duplicate scientific CLI flags: {duplicated}")


def _append_forwarded(command: list[str], flag: str, value) -> None:
    present = _flag_values(command).get(flag, [])
    if present:
        if len(present) != 1 or str(present[0]) != str(value):
            raise ValueError(
                f"conflicting forwarded scientific argument {flag}: "
                f"train_args={present}, scheduler={value}"
            )
        return
    command.extend((flag, str(value)))


def resolved_config_from_command(command: list[str], provenance: dict, checkpoint_sha256: str) -> dict:
    assert_no_duplicate_scientific_flags(command)
    parsed = parse_finetune_arguments(command[2:])
    payload = {
        output: getattr(parsed, source)
        for output, source in RESOLVED_FIELDS.items()
    }
    payload.update({
        "task": parsed.tasks[0],
        "fold": int(parsed.fold_ids[0]),
        "seed": int(parsed.seed),
        "checkpoint_sha256": checkpoint_sha256,
        **provenance,
    })
    payload["encoder_freeze_warm_epochs"] = (
        int(parsed.mts_glt_fusion_warm_epochs)
        if parsed.mts_glt_fusion_strategy == "fusion_warm" else 0
    )
    return payload


def _valid_shard(results_root: Path, unit: ScheduledUnit) -> bool:
    shard = results_root / "shards" / str(unit.seed) / unit.task / f"fold_{unit.fold}.csv"
    prediction = results_root / "predictions" / str(unit.seed) / unit.task / f"fold_{unit.fold}.npz"
    if not shard.is_file() or not shard.stat().st_size or not prediction.is_file():
        return False
    try:
        import json as _json
        import numpy as np
        import pandas as pd

        frame = pd.read_csv(shard)
        if len(frame) != 1:
            return False
        row = frame.iloc[0]
        if str(row.get("task", "")) != unit.task or int(row.get("seed", -1)) != unit.seed:
            return False
        metrics = row.get("per_fold_metrics")
        if metrics is not None and str(metrics) != "nan":
            parsed = _json.loads(str(metrics))
            if len(parsed) != 1 or int(parsed[0].get("fold", -1)) != unit.fold:
                return False
        with np.load(prediction, allow_pickle=False) as payload:
            metadata = _json.loads(str(np.asarray(payload["metadata"]).item()))
            y_true = np.asarray(payload["y_true"])
            y_pred = np.asarray(payload["y_pred"])
            indices = np.asarray(payload["sample_indices"])
            if y_true.shape != y_pred.shape or y_true.shape != indices.shape:
                return False
            if not np.isfinite(y_true).all() or not np.isfinite(y_pred).all():
                return False
        return (
            metadata.get("task") == unit.task
            and int(metadata.get("seed", -1)) == unit.seed
            and int(metadata.get("fold", -1)) == unit.fold
        )
    except Exception:
        return False


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", required=True)
    parser.add_argument("--gpu-ids", required=True)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--logs-dir", required=True)
    parser.add_argument("--tasks", nargs="+", required=True)
    parser.add_argument("--folds", nargs="+", type=int, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--schedule", default="")
    parser.add_argument("--task-order", default="")
    parser.add_argument("--fold-order", default="")
    parser.add_argument("--train-script", default="scripts/train.py")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-seed", required=True)
    parser.add_argument("--pretrain-dataset", default="PI1M_v2")
    parser.add_argument("--checkpoint-tier")
    parser.add_argument("--finetune-epochs", required=True)
    parser.add_argument("--finetune-patience", required=True)
    parser.add_argument("--batch-size", required=True)
    parser.add_argument("--eval-batch-size", required=True)
    parser.add_argument("--amp-dtype", required=True)
    parser.add_argument("--loader-workers", required=True)
    parser.add_argument("--evaluation-protocol", required=True)
    parser.add_argument("--graph-lr")
    parser.add_argument("--fusion-lr")
    parser.add_argument("--head-lr")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--train-args", nargs=argparse.REMAINDER, default=[])
    args = parser.parse_args(argv)

    gpu_ids = [value for value in args.gpu_ids.split(",") if value]
    if not gpu_ids or len(set(gpu_ids)) != len(gpu_ids):
        raise SystemExit("MTS finetune scheduler requires one or more unique GPU ids")
    task_order = (
        args.task_order.split(",")
        if args.schedule == "lpt_v1" and args.task_order else None
    )
    fold_order = (
        [int(v) for v in args.fold_order.split(",") if v]
        if args.schedule == "lpt_v1" and args.fold_order else None
    )
    if args.schedule == "lpt_v1" and (task_order is None or fold_order is None):
        raise SystemExit("lpt_v1 requires explicit task and fold order")
    ordered = dispatch_units(args.tasks, args.folds, task_order=task_order, fold_order=fold_order)
    units = [
        ScheduledUnit(seed=seed, task=task, fold=fold)
        for seed in args.seeds
        for task, fold in ordered
    ]
    results_root = Path(args.results_dir)
    logs_root = Path(args.logs_dir)
    results_root.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(args.checkpoint).resolve()
    if not checkpoint.is_file():
        raise SystemExit(f"missing finetune checkpoint: {checkpoint}")
    provenance = _git_provenance()
    checkpoint_sha256 = _sha256_file(checkpoint)

    fixed = list(args.train_args)
    # argparse.REMAINDER may receive the separator itself from a caller.
    if fixed and fixed[0] == "--":
        fixed = fixed[1:]

    def command_factory(unit: ScheduledUnit, gpu: str):
        if os.environ.get("MTS_FAKE_TRAIN", "0") == "1":
            fake = os.environ.get("MTS_FAKE_TRAIN_CMD")
            if not fake:
                raise RuntimeError("MTS_FAKE_TRAIN=1 requires MTS_FAKE_TRAIN_CMD")
            return [
                args.python, fake,
                "--task", unit.task, "--fold", str(unit.fold),
                "--shard", str(results_root / "shards" / str(unit.seed) / unit.task / f"fold_{unit.fold}.csv"),
                "--prediction", str(results_root / "predictions" / str(unit.seed) / unit.task / f"fold_{unit.fold}.npz"),
            ]
        command = [
            args.python, args.train_script,
            *fixed,
            "--tasks", unit.task,
            "--fold_ids", str(unit.fold),
            "--pretrained_model_path", args.checkpoint,
            "--checkpoint_seed", str(args.checkpoint_seed),
            "--seed", str(unit.seed),
            "--predictions_dir", str(results_root / "predictions" / str(unit.seed)),
            "--checkpoint_pretraining_dataset", args.pretrain_dataset,
            "--epochs", str(args.finetune_epochs),
            "--patience", str(args.finetune_patience),
            "--batch_size", str(args.batch_size),
            "--eval_batch_size", str(args.eval_batch_size),
            "--amp_dtype", args.amp_dtype,
            "--loader_workers", str(args.loader_workers),
            "--evaluation_protocol", args.evaluation_protocol,
            "--results_dir", str(results_root / "shards" / str(unit.seed) / unit.task / f"fold_{unit.fold}.csv"),
        ]
        if args.checkpoint_tier is not None:
            command.extend(("--checkpoint_tier", str(args.checkpoint_tier)))
        for flag, value in (
            ("--graph_lr", args.graph_lr),
            ("--fusion_lr", args.fusion_lr),
            ("--head_lr", args.head_lr),
        ):
            if value is not None:
                _append_forwarded(command, flag, value)
        unit_root = results_root / "resolved" / str(unit.seed) / unit.task / f"fold_{unit.fold}"
        config_path = unit_root / "resolved_config.json"
        command_path = unit_root / "resolved_command.txt"
        command.extend((
            "--resolved_config_path", str(config_path),
            "--resolved_command_path", str(command_path),
        ))
        assert_no_duplicate_scientific_flags(command)
        resolved = resolved_config_from_command(command, provenance, checkpoint_sha256)
        resolved["resolved_command"] = command
        _atomic_json(config_path, resolved)
        _atomic_text(command_path, shlex.join(command) + "\n")
        return command

    if args.dry_run:
        commands = []
        for index, unit in enumerate(units):
            commands.append(command_factory(unit, gpu_ids[index % len(gpu_ids)]))
        report = {
            "schema": "mts-finetune-scheduler-dry-run-v1",
            "dry_run": True,
            "units": len(units),
            "commands": commands,
            "checkpoint_sha256": checkpoint_sha256,
            **provenance,
        }
        logs_root.mkdir(parents=True, exist_ok=True)
        _atomic_json(logs_root / "scheduler_dry_run.json", report)
        print(json.dumps(report, sort_keys=True))
        return report

    report = run_subprocess_scheduler(
        units,
        gpu_ids=gpu_ids,
        command_factory=command_factory,
        should_skip=lambda unit: _valid_shard(results_root, unit),
        log_dir=logs_root,
        cwd=ROOT,
    )
    report.update({"schema": "mts-finetune-scheduler-v1", "units": len(units)})
    logs_root.mkdir(parents=True, exist_ok=True)
    (logs_root / "scheduler_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
