#!/usr/bin/env python3
"""O8-only downstream attribution run on fixed outer5_inner20 splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from scripts.create_mips_split_manifests import build_manifest
from src.dataset.glt_o8_control import O8CleanLabeledDataset, O8OnlySource, o8_collate
from src.modules.glt_o8_control import (
    O8ControlModel, apply_matched_predictor, load_fixed_concat_o8, load_o8_deployment,
)
from src.training.glt_dual_runtime import require_tmux, save_checkpoint, write_json
from src.utils import evaluate, scale_targets, set_global_seed, test_model, train_epoch, _cosine_scheduler


TASKS = ("eat", "eea", "egb", "ei", "eps", "nc", "xc")


def fixed_manifest(task, csv_path, path):
    expected = build_manifest(task, csv_path, "outer5_inner20")
    if path.exists():
        actual = json.loads(path.read_text(encoding="utf-8"))
        for field in ("protocol", "sample_count", "sample_order_hash"):
            if actual.get(field) != expected[field]:
                raise ValueError(f"fixed manifest mismatch: {field}")
        if actual.get("validation_is_test") is not False or len(actual["folds"]) != 5:
            raise ValueError("requires five separated folds")
        for old, new in zip(actual["folds"], expected["folds"]):
            for field in ("fold", "train_indices", "validation_indices", "test_indices"):
                if old[field] != new[field]:
                    raise ValueError(f"existing fixed split differs: {field}")
        return actual
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, expected)
    return expected


def optimizer_for(model, config):
    groups = [
        {"params": list(model.o8.parameters()), "lr": float(config["encoder_lr"]),
         "weight_decay": float(config["finetune_weight_decay"]), "name": "o8"},
        {"params": list(model.norm2.parameters()) + list(model.predictor.parameters()),
         "lr": float(config["fusion_head_lr"]),
         "weight_decay": float(config["finetune_weight_decay"]), "name": "norm2_predictor"},
    ]
    return torch.optim.AdamW(groups)


def _fit(model, scaler, train_loader, val_loader, device, optimizer, scheduler, config,
         *, task, fold_id, validation_only=False):
    criterion = nn.MSELoss()
    best, best_r2, best_epoch, stalled = None, -float("inf"), -1, 0
    for epoch in range(int(config["epochs"])):
        train_epoch(model, train_loader, criterion, optimizer, scheduler, device,
                    epoch=epoch + 1, amp_dtype="fp32", max_grad_norm=1.,
                    fail_nonfinite=True)
        val_loss, val_r2, _, _ = evaluate(model, val_loader, criterion, device,
                                           scaler=scaler, amp_dtype="fp32")
        if not np.isfinite(val_loss) or not np.isfinite(val_r2):
            raise FloatingPointError("nonfinite O8 validation metric")
        print(json.dumps({"task": task, "fold": int(fold_id), "epoch": epoch + 1,
                          "validation_r2": float(val_r2)}), flush=True)
        if val_r2 > best_r2:
            best = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            best_r2, best_epoch, stalled = float(val_r2), epoch + 1, 0
        else:
            stalled += 1
        if stalled >= int(config["patience"]):
            break
    if best is None:
        raise RuntimeError("no finite validation-selected O8 checkpoint")
    model.load_state_dict(best, strict=True)
    if validation_only:
        return best, best_r2, best_epoch, None
    result = test_model(model, val_loader, scaler, device, return_predictions=True)
    return best, best_r2, best_epoch, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "checkpoint", "raw-root", "cohort-root", "cache-root", "dual-static-root", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--split-root", default="data/splits/mips_outer5_inner20")
    parser.add_argument("--task", action="append", dest="tasks")
    parser.add_argument("--fold", action="append", type=int, dest="folds")
    parser.add_argument("--arm", choices=("A", "B"), required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--formal-shard", action="store_true")
    parser.add_argument("--clean-cache-gib", type=float, default=0.0)
    args = parser.parse_args()
    require_tmux()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    if args.clean_cache_gib < 0 or not np.isfinite(args.clean_cache_gib):
        raise ValueError("--clean-cache-gib must be finite and non-negative")
    tasks = list(args.tasks) if args.tasks else list(TASKS)
    folds = [int(value) for value in args.folds] if args.folds else list(range(5))
    if sorted(set(tasks)) != sorted(tasks) or any(task not in TASKS for task in tasks):
        raise ValueError("unknown or duplicate task selection")
    if any(fold < 0 or fold >= 5 for fold in folds):
        raise ValueError("fold must lie in 0..4")
    if args.smoke or args.formal_shard:
        if len(tasks) != 1 or len(folds) != 1:
            raise ValueError("smoke/formal-shard requires one task and one fold")
    elif tasks != list(TASKS) or folds != list(range(5)):
        raise ValueError("partial selection requires --smoke or --formal-shard")
    if args.arm == "A" and not args.checkpoint:
        raise ValueError("Arm A requires Fixed Concat deploy")
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    package = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_config = dict(config)
    if args.smoke:
        run_config["epochs"] = min(2, int(run_config.get("epochs", 2)))
    write_json(output / "run.json", {
        "schema": "glt-sci-o8ctrl-downstream-run-v1", "command": sys.argv,
        "arm": args.arm, "tasks": tasks, "folds": folds,
        "protocol": "outer5_inner20_smoke" if args.smoke else "outer5_inner20_formal_shard" if args.formal_shard else "outer5_inner20",
        "outer_test": "NOT_RUN" if args.smoke else "RUN_ONCE",
        "clean_cache_gib": float(args.clean_cache_gib), "uses_trimer": False,
        "uses_geometry": False,
    })
    all_folds, summaries = [], {}
    for task in tasks:
        csv_path = Path(args.raw_root) / f"smi_{task}.csv"
        manifest = fixed_manifest(task, csv_path, Path(args.split_root) / f"{task}.json")
        source = O8OnlySource(
            args.cohort_root, args.cache_root, static_root=args.dual_static_root,
            task=task,
        )
        try:
            if len(source) != int(manifest["sample_count"]):
                raise ValueError("O8 downstream cohort count differs from split")
            frame = pd.DataFrame(source.entries)
            if frame["original_row"].astype(int).tolist() != list(range(len(frame))):
                raise ValueError("O8 downstream row order differs from property CSV")
            targets = frame["label"].to_numpy(dtype=np.float64)
            if not np.isfinite(targets).all():
                raise ValueError("nonfinite downstream labels")
            dataset = O8CleanLabeledDataset(
                source, targets, cache_capacity_bytes=int(args.clean_cache_gib * (1024 ** 3))
            )
            task_results = []
            for fold in manifest["folds"]:
                if int(fold["fold"]) not in folds:
                    continue
                fold_id = int(fold["fold"])
                folder = output / task / f"fold{fold_id}"
                folder.mkdir(parents=True, exist_ok=False)
                set_global_seed(int(config["seed"]) + fold_id)
                train, validation, test = [fold[f"{name}_indices"] for name in ("train", "validation", "test")]
                scaler = scale_targets(dataset, task, train_indices=train, transform_mode="standard")
                def loader(indices, training=False):
                    return DataLoader(
                        Subset(dataset, indices),
                        batch_size=int(config["finetune_batch"] if training else config["eval_batch"]),
                        shuffle=bool(training), num_workers=0, collate_fn=o8_collate,
                        generator=torch.Generator().manual_seed(int(config["seed"]) + fold_id),
                    )
                train_loader, val_loader = loader(train, True), loader(validation)
                model = O8ControlModel().to(device)
                if args.arm == "A":
                    load_fixed_concat_o8(model, package, int(config["downstream_step"]))
                else:
                    load_o8_deployment(model, package, int(config["downstream_step"]))
                predictor_digest = apply_matched_predictor(model, int(config["seed"]), fold_id)
                optimizer = optimizer_for(model, config)
                scheduler = _cosine_scheduler(
                    optimizer, int(config["epochs"]) * max(1, len(train_loader)),
                    int(config["finetune_warmup"]) * max(1, len(train_loader)),
                )
                if args.smoke:
                    best, best_r2, best_epoch, _ = _fit(
                        model, scaler, train_loader, val_loader, device, optimizer, scheduler,
                        run_config, task=task, fold_id=fold_id, validation_only=True,
                    )
                    result = {
                        "task": task, "fold": fold_id, "protocol": "outer5_inner20_smoke",
                        "smoke": True, "best_validation_r2": float(best_r2),
                        "best_epoch": int(best_epoch), "outer_test": "NOT_RUN",
                        "scaler_fit_split": "train", "predictor_init_digest": predictor_digest,
                        "input_width": 1024, "zero_pad_second_half": True,
                    }
                    save_checkpoint(folder / "best.pt", {
                        "state_dict": best, "arm": args.arm, "task": task, "fold": fold_id,
                        "protocol": "outer5_inner20_smoke", "split": fold, "config": run_config,
                        "predictor_init_digest": predictor_digest,
                        "scaler_mean": scaler.scaler.mean_.tolist(),
                        "scaler_scale": scaler.scaler.scale_.tolist(),
                    })
                else:
                    # The validation loader is deliberately not used as the
                    # formal test loader.  Build a separate loader only after
                    # the best validation state has been restored.
                    best, best_r2, best_epoch, _ = _fit(
                        model, scaler, train_loader, val_loader, device, optimizer, scheduler,
                        run_config, task=task, fold_id=fold_id, validation_only=True,
                    )
                    test_loss, test_r2, y_true, y_pred = evaluate(
                        model, loader(test), nn.MSELoss(), device, scaler=scaler, amp_dtype="fp32"
                    )
                    del test_loss
                    if not np.isfinite(y_pred).all() or not np.isfinite(test_r2):
                        raise FloatingPointError("nonfinite O8 test prediction/metric")
                    result = {
                        "task": task, "fold": fold_id, "protocol": "outer5_inner20",
                        "test_r2": float(test_r2),
                        "test_mae": float(mean_absolute_error(y_true, y_pred)),
                        "test_rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
                        "best_validation_r2": float(best_r2), "best_epoch": int(best_epoch),
                        "outer_test": "RUN_ONCE", "predictor_init_digest": predictor_digest,
                    }
                    pd.DataFrame({"row_index": test, "target": y_true.reshape(-1),
                                  "prediction": y_pred.reshape(-1)}).to_csv(folder / "predictions.csv", index=False)
                    save_checkpoint(folder / "best.pt", {
                        "state_dict": best, "arm": args.arm, "task": task, "fold": fold_id,
                        "protocol": "outer5_inner20", "split": fold, "config": run_config,
                        "predictor_init_digest": predictor_digest,
                        "scaler_mean": scaler.scaler.mean_.tolist(),
                        "scaler_scale": scaler.scaler.scale_.tolist(),
                    })
                write_json(folder / "metrics.json", result)
                all_folds.append(result)
                task_results.append(result)
            summaries[task] = {"folds": task_results, "outer_test": "NOT_RUN" if args.smoke else "RUN_ONCE"}
        finally:
            source.close()
    pd.DataFrame(all_folds).to_csv(output / "all_fold_metrics.csv", index=False)
    summary = {
        "schema": "glt-sci-o8ctrl-downstream-summary-v1",
        "arm": args.arm, "tasks": summaries,
        "protocol": "outer5_inner20_smoke" if args.smoke else "outer5_inner20_formal_shard" if args.formal_shard else "outer5_inner20",
        "outer_test": "NOT_RUN" if args.smoke else "RUN_ONCE",
        "interpretation": "O8-only attribution arm; no Macro8",
    }
    write_json(output / "summary.json", summary)


if __name__ == "__main__":
    main()
