#!/usr/bin/env python3
"""Strict G0/G1 formal shard audit and task-level comparison.

This intentionally reuses the established MTS shard/checkpoint verifier and
adds the G-family arm, shared-step-0 and relation-artifact bindings that are
not part of the older T0/T1 comparison protocol.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from scripts.audit_mts_g0_g1_formal import audit_checkpoint
from scripts.compare_mts_t0_t1_formal import (
    FOLDS,
    TASKS,
    _metric_summary,
    _read_arm,
    _write_json,
    digest,
    compare as _compare_arms,
    expected_finetune_hash,
    expected_profile_hash,
    expected_training_hash,
    resolve,
    sha256,
)

ROOT = Path(__file__).resolve().parents[1]
PAIR_ID = "mts_g_family_step0_v2_seed42"


def _matched_code_limitation_text() -> str:
    """Read the honest launch-time code-difference classification.

    The G0/G1 launch code identity differs only in
    ``src/modules/mips_local_graph.py``.  The evidence says the delta is the
    G1 relation-geometry implementation/vectorization and the shared O8
    attention math is unchanged, but the intermediate G0-era file content has
    no recoverable snapshot, so strict per-code causal matching cannot be
    byte-proven.  This limitation is reported but does not block the
    comparison.
    """
    classification = ROOT / "results/mts_multiscale_topology/g_family_matched_v1/code_difference_classification.json"
    if classification.is_file():
        payload = json.loads(classification.read_text(encoding="utf-8"))
        return str(payload.get("limitation_text") or payload.get("conclusion"))
    return (
        "G0/G1 launch-time code identity differs in src/modules/mips_local_graph.py; "
        "strict per-code causal matching cannot be byte-proven."
    )


def _checkpoint_meta(path: Path) -> tuple[dict[str, Any], str]:
    path = path.resolve()
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return dict(payload.get("meta") or {}), sha256(path)


def _validate_g_identity(
    *, name: str, arm: str, info: dict[str, Any], config: dict[str, Any], checkpoint: Path
) -> dict[str, Any]:
    meta, checkpoint_sha = _checkpoint_meta(checkpoint)
    # Reuse the formal checkpoint audit so the comparison applies the same
    # strict completion/finite/shared-init contract.  The shared v2 step-0
    # artifact was initialized from the readiness-named config; the audit
    # accepts that hash only when its resolved scientific payload is identical
    # to the formal config (experiment name/path are the sole difference).
    checkpoint_audit = audit_checkpoint(checkpoint, arm, config)
    expected_bundle = config["g_family_bundle_hash"]
    expected = {
        "g_family_arm": arm,
        "model_identity": "T1",
        "initialization": "fresh_paired",
        "optimizer_steps": 20000,
        "smoke_only": False,
        "paired_init_id": PAIR_ID,
        "shared_step0_id": PAIR_ID,
        "pretraining_objective": "masked_atom_only",
        "g_family_bundle_hash": expected_bundle,
        "experiment_id": config["experiment_id"],
        "topology_attention_variant": "msta_last2",
    }
    mismatch = {
        key: (meta.get(key), value)
        for key, value in expected.items()
        if meta.get(key) != value
    }
    if mismatch:
        raise RuntimeError(f"{name}: checkpoint identity mismatch: {mismatch}")
    if arm == "g1":
        expected_relation = config["relation_geometry_bundle"]["PI1M_v2"]["artifact_hash"]
        if meta.get("relation_geometry_artifact_hash") != expected_relation:
            raise RuntimeError(f"{name}: PI1M_v2 relation artifact mismatch")
    for key, value in info["records"].items():
        row = value["row"]
        row_expected = {
            "g_family_arm": arm,
            "g_family_bundle_hash": expected_bundle,
            "checkpoint_g_family_bundle_hash": expected_bundle,
            "shared_step0_id": PAIR_ID,
        }
        if arm == "g1":
            row_expected["relation_geometry_artifact_hash"] = config["relation_geometry_bundle"]["downstream_union"]["artifact_hash"]
        for field, expected_value in row_expected.items():
            if str(row.get(field)) != str(expected_value):
                raise RuntimeError(f"{name}: {field} mismatch in {key}")
        prediction = Path(value["prediction"]["path"])
        with np.load(prediction, allow_pickle=False) as arrays:
            metadata = json.loads(str(np.asarray(arrays["metadata"]).item()))
        # The production prediction metadata contract records
        # task/fold/seed/finetune hashes/checkpoint_sha256 (enforced by
        # ``_read_arm``) but not the G-family arm/bundle/step-0 identity.
        # That identity is enforced here at the checkpoint meta and CSV shard
        # row levels above, and every prediction is bound to the verified
        # checkpoint via its checkpoint_sha256 field.
    return {
        "path": str(checkpoint.resolve()),
        "sha256": checkpoint_sha,
        "optimizer_steps": int(meta["optimizer_steps"]),
        "training_wall_seconds": meta.get("training_wall_seconds"),
        "g_family_arm": arm,
        "shared_step0_id": PAIR_ID,
        "g_family_bundle_hash": expected_bundle,
        "relation_geometry_artifact_hash": meta.get("relation_geometry_artifact_hash"),
        "config_binding": checkpoint_audit["config_binding"],
    }


def _task_rows(base: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for row in base["task_rows"]:
        rows.append({
            "task": row["task"],
            "g0": row["t0"],
            "g1": row["t1"],
            "g1_minus_g0": {
                "r2_mean": row["paired_delta"]["r2_mean"],
                "r2_sample_std": row["paired_delta"]["r2_std"],
                "mae_mean": row["paired_delta"]["mae_mean"],
                "rmse_mean": row["paired_delta"]["rmse_mean"],
            },
            "fold_delta_r2": row["paired_delta_values"]["r2"],
        })
    return rows


def _write_csv(path: Path, base: dict[str, Any]) -> None:
    rows: list[dict[str, Any]] = []
    for row in base["fold_rows"]:
        rows.append({
            "level": "fold",
            "task": row["task"],
            "fold": row["fold"],
            "g0_r2": row["t0_r2"],
            "g1_r2": row["t1_r2"],
            "g1_minus_g0_r2": row["delta_r2"],
            "g0_mae": row["t0_mae"],
            "g1_mae": row["t1_mae"],
            "g1_minus_g0_mae": row["delta_mae"],
            "g0_rmse": row["t0_rmse"],
            "g1_rmse": row["t1_rmse"],
            "g1_minus_g0_rmse": row["delta_rmse"],
            "g0_wall_seconds": row["t0_wall_seconds"],
            "g1_wall_seconds": row["t1_wall_seconds"],
        })
    for row in _task_rows(base):
        rows.append({
            "level": "task",
            "task": row["task"],
            "fold": "",
            "g0_r2_mean": row["g0"]["r2_mean"],
            "g0_r2_std": row["g0"]["r2_std"],
            "g1_r2_mean": row["g1"]["r2_mean"],
            "g1_r2_std": row["g1"]["r2_std"],
            "g1_minus_g0_r2_mean": row["g1_minus_g0"]["r2_mean"],
            "g1_minus_g0_r2_sample_std": row["g1_minus_g0"]["r2_sample_std"],
            "g1_minus_g0_mae_mean": row["g1_minus_g0"]["mae_mean"],
            "g1_minus_g0_rmse_mean": row["g1_minus_g0"]["rmse_mean"],
        })
    rows.append({
        "level": "task_delta_summary",
        "task": "all_tasks",
        "fold": "",
        **{f"g1_minus_g0_{key}": value for key, value in base["task_delta_statistics"].items()},
    })
    for row in base["leave_one_task_out"]:
        rows.append({
            "level": "leave_one_task_out",
            "task": row["excluded_task"],
            "fold": "",
            "g1_minus_g0_delta_macro_mean": row["delta_r2_mean"],
            "g1_minus_g0_delta_macro_median": row["delta_r2_median"],
            "g1_minus_g0_delta_macro_sample_std": row["delta_r2_sample_std"],
        })
    rows.append({"level": "macro", "task": "macro", "fold": "", **base["macro"]})
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False, float_format="%.17g")


def _wall_text(value):
    if value is None or (isinstance(value, float) and value != value):
        return "unavailable"
    return f"{float(value):.1f}"


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    comparison = payload["comparison"]
    lines = [
        "# Matched G1 − G0 formal comparison",
        "",
        "本报告来自两臂各 20k masked-atom-only 预训练及 seed-42、8 任务 × 5 folds 正式微调。",
        "所有 shard、prediction、checkpoint、bundle、shared step-0 和 cohort artifact 身份已先审计。",
        "",
        "## Task-level R²（五折 mean ± sample std）",
        "",
        "| task | G0 | G1 | G1 − G0 | improved folds |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in _task_rows(comparison):
        delta = row["fold_delta_r2"]
        lines.append(
            f"| {row['task']} | {row['g0']['r2_mean']:.6f} ± {row['g0']['r2_std']:.6f} | "
            f"{row['g1']['r2_mean']:.6f} ± {row['g1']['r2_std']:.6f} | "
            f"{row['g1_minus_g0']['r2_mean']:.6f} | {sum(value > 0 for value in delta)}/5 |"
        )
    stats = comparison["task_delta_statistics"]
    macro = comparison["macro"]
    lines += [
        "",
        "## Summary",
        "",
        f"- task-level delta mean/median/sample std: `{stats['mean']:.6f}` / `{stats['median']:.6f}` / `{stats['sample_std']:.6f}`",
        f"- positive/negative/zero task count: `{stats['positive_count']}/{stats['negative_count']}/{stats['zero_count']}`",
        f"- macro fold-mean delta: `{macro['delta_r2_mean']:.6f} ± {macro['delta_r2_std']:.6f}`",
        f"- G0/G1 pretraining wall time (s): `{_wall_text(payload['timing']['pretraining_wall_seconds']['G0'])}` / `{_wall_text(payload['timing']['pretraining_wall_seconds']['G1'])}`",
        f"- G0/G1 downstream wall time (s): `{payload['timing']['downstream_wall_seconds']['G0']:.1f}` / `{payload['timing']['downstream_wall_seconds']['G1']:.1f}`",
        "",
        "## Leave-one-task-out",
        "",
        "| excluded task | delta macro mean | delta median | delta sample std |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in comparison["leave_one_task_out"]:
        lines.append(
            f"| {row['excluded_task']} | {row['delta_r2_mean']:.6f} | "
            f"{row['delta_r2_median']:.6f} | {row['delta_r2_sample_std']:.6f} |"
        )
    lines += [
        "",
        "## Limitations",
        "",
        "- `historical_shared5` 使用项目共享 validation/test fold，不是独立盲测。",
        "- 本报告只比较 G1−G0；不启动 G2/G3、多 seed、超参数搜索或正式重评。",
        "- 是否进入后续周期由 Codex 审查决定；本报告不自动宣布晋级。",
        f"- `matched_code_limitation`：{_matched_code_limitation_text()}",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--g0-root", type=Path, required=True)
    parser.add_argument("--g1-root", type=Path, required=True)
    parser.add_argument("--g0-config", type=Path, default=ROOT / "configs/mts/experiments/G0_t1_msta_matched_formal_v1.json")
    parser.add_argument("--g1-config", type=Path, default=ROOT / "configs/mts/experiments/G1_t1_msta_angle_matched_formal_v1.json")
    parser.add_argument("--g0-checkpoint", type=Path, required=True)
    parser.add_argument("--g1-checkpoint", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args(argv)
    g0_config = resolve(args.g0_config.resolve())
    g1_config = resolve(args.g1_config.resolve())
    g0 = _read_arm(
        name="G0", root=args.g0_root.resolve(), config_path=args.g0_config.resolve(),
        checkpoint=args.g0_checkpoint, expected_model_identity="T1", allow_init=False,
        workers=args.workers, fresh_paired=True, paired_init_id=PAIR_ID,
        expected_t0_sha256=None,
    )
    g1 = _read_arm(
        name="G1", root=args.g1_root.resolve(), config_path=args.g1_config.resolve(),
        checkpoint=args.g1_checkpoint, expected_model_identity="T1", allow_init=False,
        workers=args.workers, fresh_paired=True, paired_init_id=PAIR_ID,
        expected_t0_sha256=None,
    )
    g0_checkpoint = _validate_g_identity(
        name="G0", arm="g0", info=g0, config=g0_config, checkpoint=args.g0_checkpoint
    )
    g1_checkpoint = _validate_g_identity(
        name="G1", arm="g1", info=g1, config=g1_config, checkpoint=args.g1_checkpoint
    )
    if g0_config["feature_config_hash"] != g1_config["feature_config_hash"]:
        raise RuntimeError("G0/G1 feature/cache identity differs")
    if g0_checkpoint.get("shared_step0_id") != g1_checkpoint.get("shared_step0_id"):
        raise RuntimeError("G0/G1 shared step-0 identity differs")
    comparison = _compare_arms(g0, g1)
    payload = {
        "schema": "mts-g0-g1-formal-comparison-v1",
        "cycle_id": "mts_g0_g1_matched_formal_v1",
        "status": "formal_comparison_verified",
        "arms": {
            "G0": {"config": g0_config, "checkpoint": g0_checkpoint, "verified_counts": g0["counts"]},
            "G1": {"config": g1_config, "checkpoint": g1_checkpoint, "verified_counts": g1["counts"]},
        },
        "protocol": {
            "tasks": list(TASKS), "folds": list(FOLDS), "seed": 42,
            "evaluation_protocol": "historical_shared5", "independent_blind_test": False,
            "train_batch_size": 32, "eval_batch_size": 64, "amp_dtype": "fp32",
            "dataloader_workers": args.workers, "schedule": "lpt_v1",
        },
        "comparison": comparison,
        "timing": {
            "pretraining_wall_seconds": {
                "G0": g0_checkpoint["training_wall_seconds"],
                "G1": g1_checkpoint["training_wall_seconds"],
            },
            "downstream_wall_seconds": {
                "G0": float(sum(row["metric"]["wall_seconds"] for row in g0["records"].values())),
                "G1": float(sum(row["metric"]["wall_seconds"] for row in g1["records"].values())),
            },
        },
        "limitations": [
            "historical_shared5 reuses the project's shared validation/test fold and is not an independent blind test.",
            "No G2/G3, multi-seed, hyperparameter search, or automatic promotion was run.",
            _matched_code_limitation_text(),
        ],
    }
    _write_json(args.output_json.resolve(), payload)
    _write_csv(args.output_csv.resolve(), comparison)
    _write_markdown(args.output_md.resolve(), payload)
    print(json.dumps({"status": payload["status"], "verified": {"G0": g0["counts"], "G1": g1["counts"]}, "output": str(args.output_json.resolve())}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
