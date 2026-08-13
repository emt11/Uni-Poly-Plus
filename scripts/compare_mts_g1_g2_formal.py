#!/usr/bin/env python3
"""Strict G1/G2 formal shard audit and task-level comparison.

This reuses the established MTS shard/checkpoint verifier and the verified
G1 arm (40 verified shards + 40 verified predictions, read-only), and adds the
G2 candidate arm that activates the endpoint-distance branch of the same
frozen relation-geometry sidecar.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from scripts.audit_mts_g1_g2_formal import audit_checkpoint
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


def _checkpoint_meta(path: Path) -> tuple[dict[str, Any], str]:
    path = path.resolve()
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return dict(payload.get("meta") or {}), sha256(path)


def _matched_code_limitation_text(
    g1_code_identity: dict[str, Any] | None, g2_code_identity: dict[str, Any] | None
) -> str:
    """Describe the launch-time code difference between G1 and G2.

    Both arms record a full ``pretrain_code_identity`` (per-file sha256 of the
    science/entry files at launch).  Comparing the two file maps is therefore
    byte-auditable, unlike the G0-era snapshot gap: the only differing files
    are the G-family formal whitelist entry points modified this cycle.
    """
    g1_files = (g1_code_identity or {}).get("files") or {}
    g2_files = (g2_code_identity or {}).get("files") or {}
    if not g1_files or not g2_files:
        return (
            "G1/G2 launch-time code identity maps are unavailable; "
            "strict per-code causal matching cannot be byte-proven."
        )
    g1_sha = str((g1_code_identity or {}).get("sha256") or "unavailable")
    g2_sha = str((g2_code_identity or {}).get("sha256") or "unavailable")
    differing = sorted(
        key for key in set(g1_files) | set(g2_files) if g1_files.get(key) != g2_files.get(key)
    )
    if differing == ["scripts/pretrain.py", "scripts/run_mips_trimer_scage.sh"]:
        return (
            "G1/G2 launch-time code identity differs only in the G-family "
            "formal pretraining whitelist entry points "
            "(scripts/pretrain.py and scripts/run_mips_trimer_scage.sh, "
            "g0/g1 -> g0/g1/g2); all 14 other recorded science/config files "
            "are byte-identical (G1={}, G2={}). The whitelist diff is "
            "byte-auditable against git commit 0e4fde4, the G1-era snapshot."
        ).format(g1_sha[:16], g2_sha[:16])
    return (
        "G1/G2 launch-time code identity differs in files: "
        f"{differing} (G1={g1_sha[:16]}..., G2={g2_sha[:16]}...); "
        "strict per-code causal matching cannot be byte-proven."
    )


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
            "relation_geometry_artifact_hash": config["relation_geometry_bundle"]["downstream_union"]["artifact_hash"],
        }
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
        "pretrain_code_sha256": meta.get("pretrain_code_sha256")
        or (meta.get("pretrain_code_identity") or {}).get("sha256"),
        "pretrain_code_identity": meta.get("pretrain_code_identity"),
        "config_binding": checkpoint_audit["config_binding"],
    }


def _task_rows(base: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for row in base["task_rows"]:
        rows.append({
            "task": row["task"],
            "g1": row["t0"],
            "g2": row["t1"],
            "g2_minus_g1": {
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
            "g1_r2": row["t0_r2"],
            "g2_r2": row["t1_r2"],
            "g2_minus_g1_r2": row["delta_r2"],
            "g1_mae": row["t0_mae"],
            "g2_mae": row["t1_mae"],
            "g2_minus_g1_mae": row["delta_mae"],
            "g1_rmse": row["t0_rmse"],
            "g2_rmse": row["t1_rmse"],
            "g2_minus_g1_rmse": row["delta_rmse"],
            "g1_wall_seconds": row["t0_wall_seconds"],
            "g2_wall_seconds": row["t1_wall_seconds"],
        })
    for row in _task_rows(base):
        rows.append({
            "level": "task",
            "task": row["task"],
            "fold": "",
            "g1_r2_mean": row["g1"]["r2_mean"],
            "g1_r2_std": row["g1"]["r2_std"],
            "g2_r2_mean": row["g2"]["r2_mean"],
            "g2_r2_std": row["g2"]["r2_std"],
            "g2_minus_g1_r2_mean": row["g2_minus_g1"]["r2_mean"],
            "g2_minus_g1_r2_sample_std": row["g2_minus_g1"]["r2_sample_std"],
            "g2_minus_g1_mae_mean": row["g2_minus_g1"]["mae_mean"],
            "g2_minus_g1_rmse_mean": row["g2_minus_g1"]["rmse_mean"],
        })
    rows.append({
        "level": "task_delta_summary",
        "task": "all_tasks",
        "fold": "",
        **{f"g2_minus_g1_{key}": value for key, value in base["task_delta_statistics"].items()},
    })
    for row in base["leave_one_task_out"]:
        rows.append({
            "level": "leave_one_task_out",
            "task": row["excluded_task"],
            "fold": "",
            "g2_minus_g1_delta_macro_mean": row["delta_r2_mean"],
            "g2_minus_g1_delta_macro_median": row["delta_r2_median"],
            "g2_minus_g1_delta_macro_sample_std": row["delta_r2_sample_std"],
        })
    rows.append({"level": "macro", "task": "macro", "fold": "", **base["macro"]})
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False, float_format="%.17g")


def _wall_text(value):
    if value is None or (isinstance(value, float) and value != value):
        return "unavailable"
    return f"{float(value):.1f}"


def _promotion_verdict(stats: dict[str, Any]) -> str:
    mean = float(stats["mean"])
    positive = int(stats["positive_count"])
    if mean > 0 and positive >= 5:
        return (
            "晋级候选：G2−G1 task-level R² mean > 0 且至少 5/8 任务为正；"
            "建议下一周期执行 G3 置乱负对照（本周期不自动启动 G3）。"
        )
    return (
        "保留 G1 为 G-family incumbent：G2−G1 未满足 mean > 0 且 ≥5/8 任务为正；"
        "不把 G2 设为生产默认，报告负向/混合结果后停止。"
    )


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    comparison = payload["comparison"]
    stats = comparison["task_delta_statistics"]
    lines = [
        "# Matched G2 − G1 formal comparison",
        "",
        "本报告来自两臂各 20k masked-atom-only 预训练及 seed-42、8 任务 × 5 folds 正式微调。",
        "G1 为已验收 incumbent（只读复用），G2 为在共享冻结 relation-geometry sidecar 上",
        "增加 endpoint distance 的候选。所有 shard、prediction、checkpoint、bundle、",
        "shared step-0 和 cohort artifact 身份已先审计。",
        "",
        "## Task-level R²（五折 mean ± sample std）",
        "",
        "| task | G1 | G2 | G2 − G1 | improved folds |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in _task_rows(comparison):
        delta = row["fold_delta_r2"]
        lines.append(
            f"| {row['task']} | {row['g1']['r2_mean']:.6f} ± {row['g1']['r2_std']:.6f} | "
            f"{row['g2']['r2_mean']:.6f} ± {row['g2']['r2_std']:.6f} | "
            f"{row['g2_minus_g1']['r2_mean']:.6f} | {sum(value > 0 for value in delta)}/5 |"
        )
    macro = comparison["macro"]
    lines += [
        "",
        "## Summary",
        "",
        f"- task-level delta mean/median/sample std: `{stats['mean']:.6f}` / `{stats['median']:.6f}` / `{stats['sample_std']:.6f}`",
        f"- positive/negative/zero task count: `{stats['positive_count']}/{stats['negative_count']}/{stats['zero_count']}`",
        f"- macro fold-mean delta: `{macro['delta_r2_mean']:.6f} ± {macro['delta_r2_std']:.6f}`",
        f"- G1/G2 pretraining wall time (s): `{_wall_text(payload['timing']['pretraining_wall_seconds']['G1'])}` / `{_wall_text(payload['timing']['pretraining_wall_seconds']['G2'])}`",
        f"- G1/G2 downstream wall time (s): `{payload['timing']['downstream_wall_seconds']['G1']:.1f}` / `{payload['timing']['downstream_wall_seconds']['G2']:.1f}`",
        "",
        "## Promotion verdict",
        "",
        f"- {_promotion_verdict(stats)}",
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
        "- 本报告只比较 G2−G1；不启动 G3、多 seed、超参数搜索或正式重评。",
        "- 是否进入后续周期由 Codex 审查决定；本报告不自动宣布晋级。",
        f"- `matched_code_limitation`：{payload['limitations'][-1]}",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--g1-root", type=Path, required=True)
    parser.add_argument("--g2-root", type=Path, required=True)
    parser.add_argument("--g1-config", type=Path, default=ROOT / "configs/mts/experiments/G1_t1_msta_angle_matched_formal_v1.json")
    parser.add_argument("--g2-config", type=Path, default=ROOT / "configs/mts/experiments/G2_t1_msta_cosine_distance_matched_formal_v1.json")
    parser.add_argument("--g1-checkpoint", type=Path, required=True)
    parser.add_argument("--g2-checkpoint", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args(argv)
    g1_config = resolve(args.g1_config.resolve())
    g2_config = resolve(args.g2_config.resolve())
    g1 = _read_arm(
        name="G1", root=args.g1_root.resolve(), config_path=args.g1_config.resolve(),
        checkpoint=args.g1_checkpoint, expected_model_identity="T1", allow_init=False,
        workers=args.workers, fresh_paired=True, paired_init_id=PAIR_ID,
        expected_t0_sha256=None,
    )
    g2 = _read_arm(
        name="G2", root=args.g2_root.resolve(), config_path=args.g2_config.resolve(),
        checkpoint=args.g2_checkpoint, expected_model_identity="T1", allow_init=False,
        workers=args.workers, fresh_paired=True, paired_init_id=PAIR_ID,
        expected_t0_sha256=None,
    )
    g1_checkpoint = _validate_g_identity(
        name="G1", arm="g1", info=g1, config=g1_config, checkpoint=args.g1_checkpoint
    )
    g2_checkpoint = _validate_g_identity(
        name="G2", arm="g2", info=g2, config=g2_config, checkpoint=args.g2_checkpoint
    )
    if g1_config["feature_config_hash"] != g2_config["feature_config_hash"]:
        raise RuntimeError("G1/G2 feature/cache identity differs")
    if g1_checkpoint.get("shared_step0_id") != g2_checkpoint.get("shared_step0_id"):
        raise RuntimeError("G1/G2 shared step-0 identity differs")
    if g1_config["relation_geometry_bundle"] != g2_config["relation_geometry_bundle"]:
        raise RuntimeError("G1/G2 relation-geometry sidecar differs")
    comparison = _compare_arms(g1, g2)
    code_limitation = _matched_code_limitation_text(
        g1_checkpoint.get("pretrain_code_identity"),
        g2_checkpoint.get("pretrain_code_identity"),
    )
    payload = {
        "schema": "mts-g1-g2-formal-comparison-v1",
        "cycle_id": "mts_g1_g2_matched_formal_v1",
        "status": "formal_comparison_verified",
        "arms": {
            "G1": {"config": g1_config, "checkpoint": g1_checkpoint, "verified_counts": g1["counts"]},
            "G2": {"config": g2_config, "checkpoint": g2_checkpoint, "verified_counts": g2["counts"]},
        },
        "protocol": {
            "tasks": list(TASKS), "folds": list(FOLDS), "seed": 42,
            "evaluation_protocol": "historical_shared5", "independent_blind_test": False,
            "train_batch_size": 32, "eval_batch_size": 64, "amp_dtype": "fp32",
            "dataloader_workers": args.workers, "schedule": "lpt_v1",
        },
        "comparison": comparison,
        "promotion_verdict": _promotion_verdict(comparison["task_delta_statistics"]),
        "timing": {
            "pretraining_wall_seconds": {
                "G1": g1_checkpoint["training_wall_seconds"],
                "G2": g2_checkpoint["training_wall_seconds"],
            },
            "downstream_wall_seconds": {
                "G1": float(sum(row["metric"]["wall_seconds"] for row in g1["records"].values())),
                "G2": float(sum(row["metric"]["wall_seconds"] for row in g2["records"].values())),
            },
        },
        "limitations": [
            "historical_shared5 reuses the project's shared validation/test fold and is not an independent blind test.",
            "No G3, multi-seed, hyperparameter search, or automatic promotion was run.",
            code_limitation,
        ],
    }
    _write_json(args.output_json.resolve(), payload)
    _write_csv(args.output_csv.resolve(), comparison)
    _write_markdown(args.output_md.resolve(), payload)
    print(json.dumps({"status": payload["status"], "verified": {"G1": g1["counts"], "G2": g2["counts"]}, "output": str(args.output_json.resolve())}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
