#!/usr/bin/env python3
"""Matched Full-3D versus Geometry-Off GraphGate 5k analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_mts_trimer_sensitivity import TASKS, _dataset, _folds  # noqa: E402
from scripts.report_mts_glt_v2_downstream import summarize  # noqa: E402
from src.analysis.mts_glt_postmortem import scalar_probe  # noqa: E402
from src.dataset.lmdb_cache import sample_key_from_smiles  # noqa: E402
from src.modules import MTSGraphGateModel  # noqa: E402
from src.utils import get_data_loader  # noqa: E402


OUTPUT = ROOT / "results/mts_glt_graphgate_v1/trimer_validation_v1"
MATCHED = OUTPUT / "matched_5k"
CHECKPOINTS = {
    "full": ROOT / "pretrained_models/mts_glt_graphgate_v1/trimer_validation_v1/full_5k.pth",
    "off": ROOT / "pretrained_models/mts_glt_graphgate_v1/trimer_validation_v1/off_5k.pth",
}
DOWNSTREAM_BASE = MATCHED / "neural"


def _atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _arm_model(checkpoint, device, geometry_mode):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if payload.get("schema") != "mts-glt-graphgate-v1-probe-v1" or int(payload.get("step", -1)) != 5000:
        raise RuntimeError("matched arm requires a GraphGate-v1 5k probe checkpoint")
    namespaces = payload["namespaces"]
    model = MTSGraphGateModel(layers=6)
    o8_state = model.o8_encoder.state_dict()
    expected_o8 = {
        key for key in o8_state
        if not key.startswith("md_residual.") and not key.startswith("star_distance_bias.")
    }
    if set(namespaces["o8_encoder"]) != expected_o8:
        raise RuntimeError("matched GraphGate O8 namespace mismatch")
    o8_state.update(namespaces["o8_encoder"])
    model.o8_encoder.load_state_dict(o8_state, strict=True)
    glt_state = dict(namespaces["glt_line_encoder"])
    glt_state.update({
        "query_pool." + key: value
        for key, value in namespaces["query_pool"].items()
    })
    model.glt_line_encoder.load_state_dict(glt_state, strict=True)
    model = model.to(device).eval()
    model.glt_geometry_mode = geometry_mode
    return model


def extract(arm, device_name):
    device = torch.device(device_name)
    model = _arm_model(CHECKPOINTS[arm], device, arm)
    feature_root = MATCHED / arm / "frozen_features"
    feature_root.mkdir(parents=True, exist_ok=True)
    for task in TASKS:
        dataset = _dataset(task)
        frame = pd.read_csv(ROOT / "data/raw" / f"smi_{task}.csv")
        loader = get_data_loader(
            dataset, batch_size=64, shuffle=False, drop_last=False,
            num_workers=2, pin_memory=device.type == "cuda", persistent_workers=True,
        )
        o8, glt, valid = [], [], []
        with torch.inference_mode():
            for batch in loader:
                batch = batch.to(device, non_blocking=True)
                z_o8, _ = model.o8_encoder._forward_impl(
                    batch, use_star=False, use_md=False
                )
                output = model.glt_line_encoder(
                    batch, geometry_mode=arm
                )["graph_geometry"]
                o8.append(z_o8.float().cpu())
                glt.append(output.float().cpu())
                valid.append(batch.glt_query_valid.bool().cpu())
        arrays = {
            "row_index": np.arange(len(dataset), dtype=np.int64),
            "sample_key": np.asarray([
                sample_key_from_smiles(str(value)).hex()
                for value in frame["smiles"].astype(str)
            ]),
            "target": np.asarray(dataset.raw_targets, dtype=np.float64),
            "valid_3d": torch.cat(valid).numpy().astype(bool),
            "z_o8": torch.cat(o8).numpy(),
            "z_glt": torch.cat(glt).numpy(),
        }
        for name, value in arrays.items():
            if name != "sample_key" and not np.isfinite(value).all():
                raise RuntimeError(f"non-finite matched feature: {arm}/{task}/{name}")
        with (feature_root / f"{task}.npz").open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        print(f"matched extraction {arm}/{task}: {len(dataset)}", flush=True)


def analyze_frozen():
    rows = []
    task_summary = {}
    for task_index, task in enumerate(TASKS):
        arm_data = {}
        for arm in ("full", "off"):
            with np.load(MATCHED / arm / "frozen_features" / f"{task}.npz", allow_pickle=False) as payload:
                arm_data[arm] = {key: np.asarray(payload[key]) for key in payload.files}
        if not np.array_equal(arm_data["full"]["sample_key"], arm_data["off"]["sample_key"]):
            raise RuntimeError(f"matched feature key mismatch: {task}")
        if not np.array_equal(arm_data["full"]["valid_3d"], arm_data["off"]["valid_3d"]):
            raise RuntimeError(f"matched feature valid-mask mismatch: {task}")
        valid = arm_data["full"]["valid_3d"].astype(bool)
        task_rows = []
        for fold, split in enumerate(_folds(task)):
            train = np.asarray(split["train_indices"], dtype=np.int64)
            test = np.asarray(split["test_indices"], dtype=np.int64)
            train, test = train[valid[train]], test[valid[test]]
            row = {"task": task, "fold": fold, "n_train": len(train), "n_test": len(test)}
            seed = 88000 + task_index * 100 + fold
            for arm in ("full", "off"):
                data = arm_data[arm]
                for name, features in (
                    ("o8", data["z_o8"]),
                    ("glt", data["z_glt"]),
                    ("concat", np.concatenate((data["z_o8"], data["z_glt"]), axis=1)),
                ):
                    row[f"{arm}_{name}"] = scalar_probe(
                        features[train], data["target"][train],
                        features[test], data["target"][test], seed=seed,
                    )["r2"]
            rows.append(row); task_rows.append(row)
            print(f"matched frozen {task}/fold{fold}", flush=True)
        task_summary[task] = {
            key: float(np.mean([row[key] for row in task_rows]))
            for key in ("full_o8", "full_glt", "full_concat", "off_o8", "off_glt", "off_concat")
        }
    frame = pd.DataFrame(rows)
    frame.to_csv(MATCHED / "frozen_ridge_fold_results.csv", index=False)
    interaction = np.asarray([
        (task_summary[task]["full_concat"] - task_summary[task]["full_o8"])
        - (task_summary[task]["off_concat"] - task_summary[task]["off_o8"])
        for task in TASKS
    ])
    payload = {
        "schema": "mts-trimer-matched-5k-frozen-v1",
        "tasks": task_summary,
        "macro": {
            key: float(np.mean([row[key] for row in task_summary.values()]))
            for key in next(iter(task_summary.values()))
        },
        "interaction": {
            "macro8": float(interaction.mean()),
            "median_task": float(np.median(interaction)),
            "positive_tasks": int((interaction > 0).sum()),
            "task": {task: float(value) for task, value in zip(TASKS, interaction)},
        },
    }
    _atomic_json(MATCHED / "frozen_ridge_summary.json", payload)


def analyze_neural():
    tasks, folds = ["xc", "ei", "eps"], [0, 1, 2]
    summaries = {}
    for arm in ("full", "off"):
        run_name = f"trimer_validation_v1_matched_5k_{arm}"
        summaries[arm] = summarize(
            run_name, tasks, folds, fused_mode="o8_glt_graph",
            result_base=DOWNSTREAM_BASE,
        )
        _atomic_json(MATCHED / arm / "neural_paired_summary.json", summaries[arm])
    full, off = summaries["full"], summaries["off"]
    task_rows = {}
    for task in tasks:
        full_row, off_row = full["tasks"][task], off["tasks"][task]
        fused_difference = full_row["fused_mean_r2"] - off_row["fused_mean_r2"]
        interaction = full_row["mean_delta"] - off_row["mean_delta"]
        task_rows[task] = {
            "full_fused_minus_off_fused": float(fused_difference),
            "interaction": float(interaction),
            "full": full_row, "off": off_row,
        }
    fused = np.asarray([task_rows[t]["full_fused_minus_off_fused"] for t in tasks])
    interaction = np.asarray([task_rows[t]["interaction"] for t in tasks])
    causal_positive = (
        fused.mean() > 0 and int((fused > 0).sum()) >= 2
        and interaction.mean() > 0 and np.median(interaction) > 0
        and int((interaction > 0).sum()) >= 2
    )
    payload = {
        "schema": "mts-trimer-matched-5k-neural-v1",
        "tasks": task_rows,
        "full_fused_minus_off_fused": {
            "macro3": float(fused.mean()),
            "positive_tasks": int((fused > 0).sum()),
        },
        "interaction": {
            "macro3": float(interaction.mean()),
            "median_task": float(np.median(interaction)),
            "positive_tasks": int((interaction > 0).sum()),
        },
        "trimer_3d_positive_at_5k": bool(causal_positive),
    }
    _atomic_json(MATCHED / "neural_matched_summary.json", payload)


def write_final_report():
    geometry = json.loads((OUTPUT / "geometry_distribution.json").read_text())
    sensitivity = json.loads((OUTPUT / "sensitivity_summary.json").read_text())
    association = json.loads((OUTPUT / "prediction_error_association.json").read_text())
    matched = json.loads((MATCHED / "neural_matched_summary.json").read_text())
    frozen = json.loads((MATCHED / "frozen_ridge_summary.json").read_text())
    full = sensitivity["comparisons"]["concat_full_minus_atom_line_only"]
    direction = sensitivity["comparisons"]["left_minus_right"]
    full_abs = float(np.mean(np.abs(list(full["task_delta"].values()))))
    q4 = association["q4_worse_task_counts"]
    end_sensitive = (
        float(direction["mean_absolute_task_delta"]) >= full_abs
        and max(q4["boundary_distance_relative_asymmetry"], q4["span1_angle_abs_deg_mean"]) >= 5
    )
    if matched["trimer_3d_positive_at_5k"]:
        conclusion = "local_geometry_end_sensitive" if end_sensitive else "local_geometry_supported"
    else:
        conclusion = "geometry_encoded_but_not_realized"
    payload = {
        "schema": "mts-trimer-validation-final-v1",
        "conclusion": conclusion,
        "stage_b": sensitivity["stage_c_decision"],
        "matched_5k_frozen": frozen,
        "matched_5k": matched,
        "end_sensitive": bool(end_sensitive),
    }
    _atomic_json(OUTPUT / "trimer_validation_report.json", payload)
    pi1m = geometry["sidecar_distributions"]["PI1M_v2"]
    sample = geometry["sample_distributions"]["PI1M_v2_sample"]
    comparisons = sensitivity["comparisons"]
    associations = association["q4_worse_task_counts"]
    lines = [
        "# MTS Trimer 3D Validation v1", "",
        f"**结论：`{conclusion}`**", "",
        "## 开放 Trimer 一致性", "",
        f"- 跨 RU 键 relative asymmetry p50/p90/p99：`{pi1m['boundary_distance_relative_asymmetry']['p50']:.6f}` / `{pi1m['boundary_distance_relative_asymmetry']['p90']:.6f}` / `{pi1m['boundary_distance_relative_asymmetry']['p99']:.6f}`",
        f"- span-1 角度 asymmetry (deg) p50/p90/p99：`{pi1m['span1_angle_absolute_asymmetry_deg']['p50']:.3f}` / `{pi1m['span1_angle_absolute_asymmetry_deg']['p90']:.3f}` / `{pi1m['span1_angle_absolute_asymmetry_deg']['p99']:.3f}`",
        f"- 内部键 outer-central relative difference p50/p90/p99：`{sample['internal_bond_outer_relative_mean']['p50']:.6f}` / `{sample['internal_bond_outer_relative_mean']['p90']:.6f}` / `{sample['internal_bond_outer_relative_mean']['p99']:.6f}`",
        f"- normalized Kabsch RMSD p50/p90/p99：`{sample['kabsch_rmsd_over_rg_mean']['p50']:.3f}` / `{sample['kabsch_rmsd_over_rg_mean']['p90']:.3f}` / `{sample['kabsch_rmsd_over_rg_mean']['p99']:.3f}`",
        "", "## 阶段 B：现有 20k 表示敏感性", "",
        f"- Full concat - atom-line concat macro8：`{full['macro_delta']:+.6f}`",
        f"- Median task delta：`{full['median_task_delta']:+.6f}`",
        f"- Positive tasks：`{full['positive_tasks']}/8`", "",
        f"- Distance-only 增量：`{comparisons['concat_distance_only_minus_atom_line_only']['macro_delta']:+.6f}`",
        f"- Angle-only 增量：`{comparisons['concat_angle_only_minus_atom_line_only']['macro_delta']:+.6f}`",
        f"- Left/Right mean absolute task difference：`{direction['mean_absolute_task_delta']:.6f}`",
        f"- 高不对称 Q4 更差：distance `{associations['boundary_distance_relative_asymmetry']}/8`，angle `{associations['span1_angle_abs_deg_mean']}/8`；未达到 end-sensitive 条件。",
        "", "## 阶段 C：matched 5k frozen Ridge", "",
        f"- Full O8 / GLT / concat macro8：`{frozen['macro']['full_o8']:.6f}` / `{frozen['macro']['full_glt']:.6f}` / `{frozen['macro']['full_concat']:.6f}`",
        f"- Off O8 / GLT / concat macro8：`{frozen['macro']['off_o8']:.6f}` / `{frozen['macro']['off_glt']:.6f}` / `{frozen['macro']['off_concat']:.6f}`",
        f"- Frozen interaction macro8 / median / positive tasks：`{frozen['interaction']['macro8']:+.6f}` / `{frozen['interaction']['median_task']:+.6f}` / `{frozen['interaction']['positive_tasks']}/8`",
        "", "## 阶段 C：matched 5k", "",
        f"- Full fused - Off fused macro3：`{matched['full_fused_minus_off_fused']['macro3']:+.6f}` ({matched['full_fused_minus_off_fused']['positive_tasks']}/3 positive)",
        f"- Interaction macro3：`{matched['interaction']['macro3']:+.6f}`",
        f"- Interaction median task：`{matched['interaction']['median_task']:+.6f}` ({matched['interaction']['positive_tasks']}/3 positive)",
        "", "| Task | Full O8 | Full fused | Off O8 | Off fused | Interaction |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for task in ("xc", "ei", "eps"):
        row = matched["tasks"][task]
        lines.append(
            f"| {task} | {row['full']['o8_mean_r2']:.6f} | {row['full']['fused_mean_r2']:.6f} "
            f"| {row['off']['o8_mean_r2']:.6f} | {row['off']['fused_mean_r2']:.6f} "
            f"| {row['interaction']:+.6f} |"
        )
    lines += [
        "", "Trimer 仅为开放局部 3D 代理。阶段 B 是训练后敏感性诊断，不单独构成因果证据；阶段 C 才是共享 step-0 的 matched 短预训练对照。", "",
        "虽然 Geometry-Off 的 Masked Line loss 明显更高，且 frozen probe 显示几何信息被编码，但神经微调的 Full fused 并未在至少 2/3 任务超过 Off fused，因此判定为几何已编码、尚未稳定兑现为下游收益。", "",
        "本周期没有增加构象、重建 cache、运行正式 20k，且未修改既有 GraphGate、GLT-v2 或 B0 正式结果。", "",
    ]
    path = OUTPUT / "trimer_validation_report.md"
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text("\n".join(lines))
    temporary.replace(path)
    print(path)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    extract_parser = sub.add_parser("extract")
    extract_parser.add_argument("--arm", choices=("full", "off"), required=True)
    extract_parser.add_argument("--device", default="cuda:0")
    sub.add_parser("analyze-frozen")
    sub.add_parser("analyze-neural")
    sub.add_parser("report")
    args = parser.parse_args()
    if args.command == "extract":
        extract(args.arm, args.device)
    elif args.command == "analyze-frozen":
        analyze_frozen()
    elif args.command == "analyze-neural":
        analyze_neural()
    else:
        write_final_report()


if __name__ == "__main__":
    main()
