#!/usr/bin/env python3
"""CLI bridge from the MTS launcher to the real Python fold scheduler."""

from __future__ import annotations

import argparse
import json
import os
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
    parser.add_argument("--finetune-epochs", required=True)
    parser.add_argument("--finetune-patience", required=True)
    parser.add_argument("--batch-size", required=True)
    parser.add_argument("--eval-batch-size", required=True)
    parser.add_argument("--amp-dtype", required=True)
    parser.add_argument("--loader-workers", required=True)
    parser.add_argument("--evaluation-protocol", required=True)
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
        return [
            args.python, args.train_script,
            *fixed,
            "--tasks", unit.task,
            "--fold_ids", str(unit.fold),
            "--pretrained_model_path", args.checkpoint,
            "--checkpoint_seed", str(args.checkpoint_seed),
            "--seed", str(unit.seed),
            "--predictions_dir", str(results_root / "predictions" / str(unit.seed)),
            "--checkpoint_pretraining_dataset", args.pretrain_dataset,
            "--checkpoint_tier", "1m",
            "--epochs", str(args.finetune_epochs),
            "--patience", str(args.finetune_patience),
            "--batch_size", str(args.batch_size),
            "--eval_batch_size", str(args.eval_batch_size),
            "--amp_dtype", args.amp_dtype,
            "--loader_workers", str(args.loader_workers),
            "--evaluation_protocol", args.evaluation_protocol,
            "--target_transform", "recommended",
            "--regression_loss", "huber", "--huber_beta", "0.5",
            "--max_grad_norm", "1.0", "--graph_lr", "1e-5",
            "--fusion_lr", "1e-4", "--head_lr", "1e-4",
            "--weight_decay", "0.02", "--warmup_epochs", "5",
            "--head_dropout", "0.25",
            "--results_dir", str(results_root / "shards" / str(unit.seed) / unit.task / f"fold_{unit.fold}.csv"),
        ]

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
