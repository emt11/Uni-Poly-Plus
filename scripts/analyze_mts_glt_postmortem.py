#!/usr/bin/env python3
"""Extract and analyse MTS-GLT-v1 postmortem representations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.mts_glt_postmortem import (  # noqa: E402
    choose_direction,
    contrastive_cosines,
    cross_predictability,
    linear_cka,
    residual_probe,
    scalar_probe,
)
from src.dataset import UniDataset  # noqa: E402
from src.dataset.lmdb_cache import sample_key_from_smiles  # noqa: E402
from src.modules import MTSGraphLineModel  # noqa: E402
from src.utils import get_data_loader  # noqa: E402


TASKS = ("eat", "eea", "egb", "egc", "ei", "eps", "nc", "xc")
OUTPUT = ROOT / "results/mts_glt_v1/postmortem"
PROBE = ROOT / "results/mts_glt_v1/pretrain/mts_glt_v1_probe_020k.pth"
FINAL = ROOT / "pretrained_models/mts_glt_v1/mts_glt_v1_seed42_final.pth"
SIDECAR = ROOT / "data/processed/mips_trimer_scage/periodic_line_glt_v1/downstream_union"
SPLITS = ROOT / "data/splits/mips_shared5"
FORMAL = ROOT / "results/mts_glt_v1/final_report.json"


def _atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _checkpoint_components():
    probe = torch.load(PROBE, map_location="cpu", weights_only=True)
    final = torch.load(FINAL, map_location="cpu", weights_only=True)
    probe_state = probe["state_dict"]
    final_state = final["state_dict"]
    probe_model = {
        key: value for key, value in probe_state.items() if key.startswith("model.")
    }
    if set(probe_model) != set(final_state):
        raise RuntimeError("20k probe and final encoder key sets differ")
    unequal = [
        key for key in probe_model
        if not torch.equal(probe_model[key], final_state[key])
    ]
    if unequal:
        raise RuntimeError(
            "20k probe and final encoder tensors differ: " + ", ".join(unequal[:10])
        )
    model = MTSGraphLineModel()
    model.load_state_dict({
        key[len("model."):]: value for key, value in probe_model.items()
    }, strict=True)
    o8_projection = nn.Sequential(nn.LayerNorm(512), nn.Linear(512, 256))
    glt_projection = nn.Sequential(nn.LayerNorm(512), nn.Linear(512, 256))
    for name, module in (
        ("o8_projection", o8_projection), ("glt_projection", glt_projection)
    ):
        prefix = name + "."
        state = {
            key[len(prefix):]: value
            for key, value in probe_state.items() if key.startswith(prefix)
        }
        module.load_state_dict(state, strict=True)
    return model, o8_projection, glt_projection


def _dataset(task):
    return UniDataset(
        root=str(ROOT / "data"),
        dataset=f"smi_{task}",
        smiles_model_name=str(
            ROOT / "pretrained_models/encoders/PubChem10M_SMILES_BPE_450k"
        ),
        graph_encoder_type="mips_trimer_scage",
        graph_input="star_linking",
        use_feature_cache=True,
        feature_source_dataset="smi_all",
        fp_mode="ecfp",
        cache_layers="ru_base,topology,trimer,md200",
        mips_core="paper_corrected",
        mips_max_hops=2,
        mips_use_descriptors=True,
        mips_descriptor_protocol="source_star_sub",
        spatial_mode="trimer_scage",
        graph_geometry_mode="trimer_scage_mcl",
        topology_representation="canonical_lifted",
        trimer_num_candidates=4,
        trimer_max_heavy_atoms=384,
        mips_variant="O8",
        experiment_id="mts_glt_v1_postmortem",
        modalities=("graph",),
        periodic_line_glt_sidecar=str(SIDECAR),
    )


def extract_features(device_name, tasks):
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA feature extraction requested but unavailable")
    model, o8_projection, glt_projection = _checkpoint_components()
    model, o8_projection, glt_projection = (
        model.to(device).eval(), o8_projection.to(device).eval(),
        glt_projection.to(device).eval(),
    )
    feature_root = OUTPUT / "frozen_features"
    feature_root.mkdir(parents=True, exist_ok=True)
    for task in tasks:
        dataset = _dataset(task)
        frame = pd.read_csv(ROOT / "data/raw" / f"smi_{task}.csv")
        if len(frame) != len(dataset):
            raise RuntimeError(f"{task} CSV/Dataset length mismatch")
        loader = get_data_loader(
            dataset, batch_size=64, shuffle=False, drop_last=False,
            num_workers=2, pin_memory=device.type == "cuda", persistent_workers=True,
        )
        o8_values, glt_values, po8_values, pglt_values, valid_values = [], [], [], [], []
        with torch.inference_mode():
            for batch in loader:
                batch = batch.to(device, non_blocking=True)
                views = model.encode_views(batch)
                o8_values.append(views["z_o8"].float().cpu())
                glt_values.append(views["z_glt"].float().cpu())
                po8_values.append(o8_projection(views["z_o8"]).float().cpu())
                pglt_values.append(glt_projection(views["z_glt"]).float().cpu())
                valid_values.append(views["valid_3d"].cpu())
        arrays = {
            "row_index": np.arange(len(dataset), dtype=np.int64),
            "sample_key": np.asarray([
                sample_key_from_smiles(str(value)).hex()
                for value in frame.iloc[:, 0].astype(str)
            ]),
            "target": np.asarray(dataset.raw_targets, dtype=np.float64),
            "valid_3d": torch.cat(valid_values).numpy().astype(bool),
            "z_o8": torch.cat(o8_values).numpy(),
            "z_glt": torch.cat(glt_values).numpy(),
            "p_o8": torch.cat(po8_values).numpy(),
            "p_glt": torch.cat(pglt_values).numpy(),
        }
        for name, value in arrays.items():
            if name != "sample_key" and not np.isfinite(value).all():
                raise RuntimeError(f"non-finite frozen feature: {task}/{name}")
        path = feature_root / f"{task}.npz"
        with path.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        print(f"extracted {task}: rows={len(dataset)} valid={int(arrays['valid_3d'].sum())}")


def _read_manifest(task):
    payload = json.loads((SPLITS / f"{task}.json").read_text(encoding="utf-8"))
    if payload.get("schema") != "mips-shared-validation-test-fold-v1":
        raise RuntimeError(f"unexpected split schema for {task}")
    return payload["folds"]


def _task_probe_summary(rows):
    task_summary = {}
    for task in TASKS:
        selected = [row for row in rows if row["task"] == task]
        task_row = {}
        p1 = np.asarray([row["p1_r2"] for row in selected])
        for name in ("p1", "p2", "p3", "p4"):
            values = np.asarray([row[f"{name}_r2"] for row in selected])
            deltas = values - p1
            task_row[name] = {
                "mean_r2": float(values.mean()),
                "sample_std_r2": float(values.std(ddof=1)),
                "fold_r2": values.tolist(),
                "delta_vs_p1": float(deltas.mean()),
                "positive_folds": int((deltas > 0).sum()),
            }
        task_summary[task] = task_row
    macro_p1 = float(np.mean([value["p1"]["mean_r2"] for value in task_summary.values()]))
    deltas = {}
    for name in ("p2", "p3", "p4"):
        task_deltas = np.asarray([
            value[name]["delta_vs_p1"] for value in task_summary.values()
        ])
        deltas[name] = {
            "macro_delta": float(task_deltas.mean()),
            "median_task_delta": float(np.median(task_deltas)),
            "positive_tasks": int((task_deltas > 0).sum()),
            "negative_tasks": int((task_deltas < 0).sum()),
        }
    return {
        "schema": "mts-glt-v1-frozen-probe-v1",
        "macro8": {
            name: float(np.mean([value[name]["mean_r2"] for value in task_summary.values()]))
            for name in ("p1", "p2", "p3", "p4")
        },
        "p1_macro": macro_p1,
        "deltas_vs_p1": deltas,
        "tasks": task_summary,
    }


def analyze_features():
    probe_rows, redundancy_rows = [], []
    for task_index, task in enumerate(TASKS):
        with np.load(OUTPUT / "frozen_features" / f"{task}.npz", allow_pickle=False) as data:
            arrays = {key: np.asarray(data[key]) for key in data.files}
        valid = arrays["valid_3d"].astype(bool)
        for fold, split in enumerate(_read_manifest(task)):
            train = np.asarray(split["train_indices"], dtype=np.int64)
            test = np.asarray(split["test_indices"], dtype=np.int64)
            train, test = train[valid[train]], test[valid[test]]
            if np.intersect1d(train, test).size:
                raise RuntimeError(f"split overlap: {task}/fold{fold}")
            y_train, y_test = arrays["target"][train], arrays["target"][test]
            o8_train, o8_test = arrays["z_o8"][train], arrays["z_o8"][test]
            glt_train, glt_test = arrays["z_glt"][train], arrays["z_glt"][test]
            seed = 42000 + task_index * 100 + fold
            p1 = scalar_probe(o8_train, y_train, o8_test, y_test, seed=seed)
            p2 = scalar_probe(glt_train, y_train, glt_test, y_test, seed=seed + 1)
            p3 = scalar_probe(
                np.concatenate([o8_train, glt_train], axis=1), y_train,
                np.concatenate([o8_test, glt_test], axis=1), y_test,
                seed=seed + 2,
            )
            p4 = residual_probe(
                o8_train, glt_train, y_train, o8_test, glt_test, y_test,
                seed=seed + 3,
            )
            probe_rows.append({
                "task": task, "fold": fold,
                "n_train": int(len(train)), "n_test": int(len(test)),
                "p1_r2": p1["r2"], "p2_r2": p2["r2"],
                "p3_r2": p3["r2"], "p4_r2": p4["r2"],
                "p1_alpha": p1["alpha"], "p2_alpha": p2["alpha"],
                "p3_alpha": p3["alpha"],
                "p4_o8_alpha": p4["o8_alpha"],
                "p4_residual_alpha": p4["residual_alpha"],
                "p4_oof_residual_rms": p4["oof_residual_rms"],
            })
            o8_to_glt = cross_predictability(
                o8_train, glt_train, o8_test, glt_test, seed=seed + 10
            )
            glt_to_o8 = cross_predictability(
                glt_train, o8_train, glt_test, o8_test, seed=seed + 11
            )
            cosine = contrastive_cosines(
                arrays["p_o8"][test], arrays["p_glt"][test]
            )
            redundancy_rows.append({
                "task": task, "fold": fold,
                "n_test": int(len(test)),
                "heldout_linear_cka": float(linear_cka(o8_test, glt_test)),
                "o8_to_glt_explained_variance": o8_to_glt["explained_variance"],
                "glt_to_o8_explained_variance": glt_to_o8["explained_variance"],
                "o8_to_glt_alpha": o8_to_glt["alpha"],
                "glt_to_o8_alpha": glt_to_o8["alpha"],
                "matched_cosine_mean": cosine["matched"]["mean"],
                "matched_cosine_median": cosine["matched"]["median"],
                "matched_cosine_std": cosine["matched"]["std"],
                "matched_cosine_p10": cosine["matched"]["p10"],
                "matched_cosine_p90": cosine["matched"]["p90"],
                "unmatched_cosine_mean": cosine["unmatched"]["mean"],
                "unmatched_cosine_median": cosine["unmatched"]["median"],
                "unmatched_cosine_std": cosine["unmatched"]["std"],
                "unmatched_cosine_p10": cosine["unmatched"]["p10"],
                "unmatched_cosine_p90": cosine["unmatched"]["p90"],
                "cosine_mean_separation": cosine["mean_separation"],
                "unmatched_shifts": ";".join(map(str, cosine["legal_shifts"])),
            })
            print(f"analysed {task}/fold{fold}")
    pd.DataFrame(probe_rows).to_csv(
        OUTPUT / "frozen_probe_fold_results.csv", index=False
    )
    probe_summary = _task_probe_summary(probe_rows)
    probe_summary["direction"] = choose_direction(probe_summary)
    _atomic_json(OUTPUT / "frozen_probe_summary.json", probe_summary)
    pd.DataFrame(redundancy_rows).to_csv(
        OUTPUT / "redundancy_fold_results.csv", index=False
    )
    redundancy = {
        "schema": "mts-glt-v1-heldout-redundancy-v1",
        "folds": redundancy_rows,
        "means": {
            key: float(np.mean([row[key] for row in redundancy_rows]))
            for key in (
                "heldout_linear_cka", "o8_to_glt_explained_variance",
                "glt_to_o8_explained_variance", "matched_cosine_mean",
                "unmatched_cosine_mean", "cosine_mean_separation",
            )
        },
    }
    _atomic_json(OUTPUT / "redundancy_analysis.json", redundancy)
    write_report(probe_summary, redundancy)


def aggregate_fusion():
    formal = json.loads(FORMAL.read_text(encoding="utf-8"))
    rows, trajectory = [], []
    for task in TASKS:
        for fold in range(5):
            path = OUTPUT / "fusion_audit_units" / task / f"fold_{fold}.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            for epoch in payload["gate_trajectory"]:
                trajectory.append({"task": task, "fold": fold, **epoch})
            test = payload["test"]
            historical_o8 = formal["tasks"][task]["o8_only"]["fold_r2"][fold]
            rows.append({
                "task": task, "fold": fold,
                "final_gate": payload["final_gate"],
                "tanh_gate": payload["final_tanh_gate"],
                "mean_o8_norm": test["o8_norm"]["mean"],
                "mean_projected_glt_norm": test["projected_glt_norm"]["mean"],
                "mean_delta_3d_norm": test["delta_3d_norm"]["mean"],
                "rho_mean": test["rho"]["mean"],
                "rho_median": test["rho"]["median"],
                "rho_p10": test["rho"]["p10"],
                "rho_p90": test["rho"]["p90"],
                "cosine_mean": test["cosine_o8_delta"]["mean"],
                "cosine_median": test["cosine_o8_delta"]["median"],
                "valid_count": test["valid_count"],
                "projection_relative_change": payload["projection_relative_change"],
                "best_epoch": payload["best_epoch"],
                "historical_o8_only_r2": historical_o8,
                "rerun_fused_r2": payload["rerun_fused_r2"],
                "fused_minus_o8_only": payload["rerun_fused_r2"] - historical_o8,
            })
    pd.DataFrame(trajectory).to_csv(
        OUTPUT / "fusion_gate_trajectory.csv", index=False
    )
    pd.DataFrame(rows).to_csv(OUTPUT / "fusion_audit.csv", index=False)
    trajectory_frame = pd.DataFrame(trajectory)
    abs_gate = trajectory_frame["tanh_gate"].abs().to_numpy()
    first_open_epochs = []
    for (_, _), group in trajectory_frame.groupby(["task", "fold"]):
        opened = group.loc[group["tanh_gate"].abs() >= 0.001, "epoch"]
        if len(opened):
            first_open_epochs.append(int(opened.iloc[0]))
    trajectory_summary = {
        "epochs_recorded": int(len(trajectory_frame)),
        "fraction_abs_tanh_below_0_001": float(np.mean(abs_gate < 0.001)),
        "fraction_abs_tanh_below_0_01": float(np.mean(abs_gate < 0.01)),
        "folds_reaching_abs_tanh_0_001": int(len(first_open_epochs)),
        "median_first_abs_tanh_0_001_epoch": (
            float(np.median(first_open_epochs)) if first_open_epochs else None
        ),
    }
    _atomic_json(OUTPUT / "fusion_audit.json", {
        "schema": "mts-glt-v1-fusion-audit-v1", "folds": rows,
        "trajectory_summary": trajectory_summary,
    })


def write_report(probe_summary, redundancy):
    formal = json.loads(FORMAL.read_text(encoding="utf-8"))
    audit_path = OUTPUT / "fusion_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8")) if audit_path.is_file() else None
    lines = [
        "# MTS-GLT-v1 Postmortem", "",
        "> 口径：现有seed-42 20k checkpoint；historical_shared5共享验证/测试fold，非独立盲测。", "",
        "## 固定正式结果", "",
        f"- O8-only macro8：`{formal['macro8']['o8_only']:.6f}`",
        f"- O8+GLT macro8：`{formal['macro8']['o8_glt']:.6f}`",
        f"- 正式delta：`{formal['macro8_fused_minus_o8_only']:+.6f}`",
        "- 本周期未重新预训练，也未修改上述正式结果。", "",
    ]
    if audit is not None:
        folds = audit["folds"]
        trajectory = audit["trajectory_summary"]
        lines.extend([
            "## Fusion audit", "",
            f"- mean tanh(gate)：`{np.mean([x['tanh_gate'] for x in folds]):.6f}`",
            f"- mean |tanh(gate)|：`{np.mean([abs(x['tanh_gate']) for x in folds]):.6f}`",
            f"- median rho：`{np.median([x['rho_median'] for x in folds]):.6f}`",
            f"- mean projection relative change：`{np.mean([x['projection_relative_change'] for x in folds]):.6f}`",
            f"- trajectory中`|tanh(gate)|<0.01`的epoch比例：`{trajectory['fraction_abs_tanh_below_0_01']:.3%}`",
            f"- 首次达到`|tanh(gate)|>=0.001`的fold数：`{trajectory['folds_reaching_abs_tanh_0_001']}/40`",
            "- 40-fold重跑逐fold精确复现原fused结果；完整trajectory和rho分布见对应CSV/JSON。", "",
        ])
    lines.extend([
        "## Frozen probes", "",
        "| Probe | macro8 | delta vs P1 | median task delta | positive tasks |",
        "|---|---:|---:|---:|---:|",
    ])
    for name in ("p1", "p2", "p3", "p4"):
        if name == "p1":
            lines.append(f"| {name.upper()} | {probe_summary['macro8'][name]:.6f} | 0.000000 | 0.000000 | - |")
        else:
            row = probe_summary["deltas_vs_p1"][name]
            lines.append(
                f"| {name.upper()} | {probe_summary['macro8'][name]:.6f} | "
                f"{row['macro_delta']:+.6f} | {row['median_task_delta']:+.6f} | "
                f"{row['positive_tasks']}/8 |"
            )
    lines.extend([
        "", "### Task-level paired deltas", "",
        "| Task | P2−P1 | P3−P1 | P3 positive folds | P4−P1 | P4 positive folds |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for task, row in probe_summary["tasks"].items():
        lines.append(
            f"| {task} | {row['p2']['delta_vs_p1']:+.6f} | "
            f"{row['p3']['delta_vs_p1']:+.6f} | {row['p3']['positive_folds']}/5 | "
            f"{row['p4']['delta_vs_p1']:+.6f} | {row['p4']['positive_folds']}/5 |"
        )
    means = redundancy["means"]
    lines.extend([
        "", "## Held-out redundancy", "",
        f"- mean linear CKA：`{means['heldout_linear_cka']:.6f}`",
        f"- O8→GLT explained variance：`{means['o8_to_glt_explained_variance']:.6f}`",
        f"- GLT→O8 explained variance：`{means['glt_to_o8_explained_variance']:.6f}`",
        f"- matched/unmatched cosine mean：`{means['matched_cosine_mean']:.6f}` / `{means['unmatched_cosine_mean']:.6f}`",
        "- 这些指标只解释probe结果，不单独决定路线。", "",
        "## 单一结论", "",
        f"```text\n{probe_summary['direction']}\n```", "",
        "P3满足macro delta>0、median task delta>0及5/8任务为正；P4没有触发预设的强负向否决条件。结合当前fusion典型rho仅约0.24%，本周期将问题归入fusion利用不足，而不是据此宣称GLT-v1已经获得正式性能提升。", "",
        "本周期到此停止；未自动启动FusionWarm、InfoNCE筛选、torsion、contact或新20k训练。", "",
    ])
    (OUTPUT / "postmortem_report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main(argv=None):
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    extract = sub.add_parser("extract")
    extract.add_argument("--device", default="cuda:0")
    extract.add_argument("--tasks", nargs="+", default=list(TASKS))
    sub.add_parser("aggregate-fusion")
    sub.add_parser("analyze")
    args = parser.parse_args(argv)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if args.command == "extract":
        extract_features(args.device, args.tasks)
    elif args.command == "aggregate-fusion":
        aggregate_fusion()
    else:
        analyze_features()


if __name__ == "__main__":
    main()
