#!/usr/bin/env python3
"""Frozen GraphGate geometry sensitivity and downstream-error association."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.mts_glt_postmortem import scalar_probe  # noqa: E402
from src.analysis.mts_trimer_validation import (  # noqa: E402
    VARIANTS, distribution, graphgate_variant_view,
)
from src.dataset import UniDataset  # noqa: E402
from src.dataset.lmdb_cache import sample_key_from_smiles  # noqa: E402
from src.modules import MTSGraphGateModel  # noqa: E402
from src.utils import get_data_loader  # noqa: E402


TASKS = ("eat", "eea", "egb", "egc", "ei", "eps", "nc", "xc")
OUTPUT = ROOT / "results/mts_glt_graphgate_v1/trimer_validation_v1"
CHECKPOINT = ROOT / "pretrained_models/mts_glt_graphgate_v1/mts_glt_graphgate_v1.pth"
SIDECAR = ROOT / "data/processed/mips_trimer_scage/periodic_line_glt_central_v1/downstream_union"
SPLITS = ROOT / "data/splits/mips_shared5"
FORMAL = ROOT / "results/mts_glt_graphgate_v1/downstream/formal_20k"


def _atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _load_model(device):
    payload = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    if payload.get("schema") != "mts-glt-graphgate-v1-probe-v1" or int(payload.get("step", -1)) != 20000:
        raise RuntimeError("expected the completed GraphGate-v1 20k probe checkpoint")
    namespaces = payload["namespaces"]
    model = MTSGraphGateModel(layers=6)
    o8_state = model.o8_encoder.state_dict()
    expected_o8 = {
        key for key in o8_state
        if not key.startswith("md_residual.") and not key.startswith("star_distance_bias.")
    }
    if set(namespaces["o8_encoder"]) != expected_o8:
        raise RuntimeError("GraphGate O8 namespace mismatch")
    o8_state.update(namespaces["o8_encoder"])
    model.o8_encoder.load_state_dict(o8_state, strict=True)
    glt_state = dict(namespaces["glt_line_encoder"])
    glt_state.update({"query_pool." + key: value for key, value in namespaces["query_pool"].items()})
    model.glt_line_encoder.load_state_dict(glt_state, strict=True)
    return model.to(device).eval()


def _dataset(task):
    return UniDataset(
        root=str(ROOT / "data"), dataset=f"smi_{task}",
        smiles_model_name=str(ROOT / "pretrained_models/encoders/PubChem10M_SMILES_BPE_450k"),
        graph_encoder_type="mips_trimer_scage", graph_input="star_linking",
        use_feature_cache=True, feature_source_dataset="smi_all",
        fp_mode="disabled", cache_layers="ru_base,topology,trimer,md200",
        mips_core="paper_corrected", mips_max_hops=2,
        mips_use_descriptors=True, mips_descriptor_protocol="source_star_sub",
        spatial_mode="trimer_scage", graph_geometry_mode="trimer_scage_mcl",
        topology_representation="canonical_lifted", trimer_num_candidates=4,
        trimer_max_heavy_atoms=384, mips_variant="O8",
        experiment_id="mts_trimer_validation_v1", modalities=("graph",),
        periodic_line_glt_sidecar=str(SIDECAR),
    )


def _cosine_and_relative(first, second):
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    denominator = np.maximum(
        np.linalg.norm(first, axis=1) * np.linalg.norm(second, axis=1), 1e-12
    )
    cosine = np.sum(first * second, axis=1) / denominator
    relative = np.linalg.norm(first - second, axis=1) / np.maximum(
        np.linalg.norm(first, axis=1), 1e-12
    )
    return cosine, relative


def extract(device_name, tasks):
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA sensitivity extraction requested but unavailable")
    model = _load_model(device)
    feature_root = OUTPUT / "sensitivity_features"
    feature_root.mkdir(parents=True, exist_ok=True)
    representation_rows = []
    for task in tasks:
        dataset = _dataset(task)
        frame = pd.read_csv(ROOT / "data/raw" / f"smi_{task}.csv")
        if len(dataset) != len(frame):
            raise RuntimeError(f"dataset/CSV mismatch for {task}")
        loader = get_data_loader(
            dataset, batch_size=64, shuffle=False, drop_last=False,
            num_workers=2, pin_memory=device.type == "cuda", persistent_workers=True,
        )
        o8_values = []
        variant_values = {name: [] for name in VARIANTS}
        valid_values = []
        with torch.inference_mode():
            for batch in loader:
                batch = batch.to(device, non_blocking=True)
                z_o8, _ = model.o8_encoder._forward_impl(
                    batch, use_star=False, use_md=False
                )
                o8_values.append(z_o8.float().cpu())
                valid_values.append(batch.glt_query_valid.bool().cpu())
                for name in VARIANTS:
                    view = graphgate_variant_view(batch, name)
                    output = model.glt_line_encoder(view)["graph_geometry"]
                    variant_values[name].append(output.float().cpu())
        arrays = {
            "row_index": np.arange(len(dataset), dtype=np.int64),
            "sample_key": np.asarray([
                sample_key_from_smiles(str(value)).hex()
                for value in frame["smiles"].astype(str)
            ]),
            "target": np.asarray(dataset.raw_targets, dtype=np.float64),
            "valid_3d": torch.cat(valid_values).numpy().astype(bool),
            "z_o8": torch.cat(o8_values).numpy(),
        }
        for name in VARIANTS:
            arrays[f"z_glt_{name}"] = torch.cat(variant_values[name]).numpy()
        for name, values in arrays.items():
            if name != "sample_key" and not np.isfinite(values).all():
                raise RuntimeError(f"non-finite sensitivity feature: {task}/{name}")
        valid = arrays["valid_3d"]
        for pair_name, first_name, second_name in (
            ("full_vs_atom", "full", "atom_line_only"),
            ("left_vs_right", "left_only", "right_only"),
        ):
            cosine, relative = _cosine_and_relative(
                arrays[f"z_glt_{first_name}"][valid], arrays[f"z_glt_{second_name}"][valid]
            )
            representation_rows.append({
                "task": task, "comparison": pair_name,
                "cosine": distribution(cosine),
                "relative_l2": distribution(relative),
            })
        with (feature_root / f"{task}.npz").open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        print(f"extracted {task}: rows={len(dataset)} valid={int(valid.sum())}", flush=True)
    _atomic_json(OUTPUT / "representation_sensitivity.json", {
        "schema": "mts-trimer-representation-sensitivity-v1",
        "comparisons": representation_rows,
    })


def _folds(task):
    payload = json.loads((SPLITS / f"{task}.json").read_text(encoding="utf-8"))
    if payload.get("schema") != "mips-shared-validation-test-fold-v1":
        raise RuntimeError(f"unexpected split schema for {task}")
    return payload["folds"]


def _probe_summary(rows):
    frame = pd.DataFrame(rows)
    tasks = {}
    for task in TASKS:
        selected = frame[frame.task == task]
        task_result = {}
        for representation in ("p0_o8",) + tuple(
            f"{kind}_{variant}" for variant in VARIANTS for kind in ("glt", "concat")
        ):
            values = selected[representation].to_numpy(dtype=np.float64)
            task_result[representation] = {
                "mean_r2": float(values.mean()),
                "sample_std_r2": float(values.std(ddof=1)),
                "fold_r2": values.tolist(),
            }
        tasks[task] = task_result

    macro = {
        representation: float(np.mean([value[representation]["mean_r2"] for value in tasks.values()]))
        for representation in next(iter(tasks.values()))
    }
    comparisons = {}
    atom = np.asarray([tasks[t]["concat_atom_line_only"]["mean_r2"] for t in TASKS])
    for variant in ("full", "distance_only", "angle_only", "left_only", "right_only"):
        candidate = np.asarray([tasks[t][f"concat_{variant}"]["mean_r2"] for t in TASKS])
        delta = candidate - atom
        comparisons[f"concat_{variant}_minus_atom_line_only"] = {
            "macro_delta": float(delta.mean()),
            "median_task_delta": float(np.median(delta)),
            "positive_tasks": int((delta > 0).sum()),
            "task_delta": {task: float(value) for task, value in zip(TASKS, delta)},
        }
    left = np.asarray([tasks[t]["concat_left_only"]["mean_r2"] for t in TASKS])
    right = np.asarray([tasks[t]["concat_right_only"]["mean_r2"] for t in TASKS])
    comparisons["left_minus_right"] = {
        "macro_delta": float((left - right).mean()),
        "mean_absolute_task_delta": float(np.abs(left - right).mean()),
        "task_delta": {task: float(value) for task, value in zip(TASKS, left - right)},
    }
    full = comparisons["concat_full_minus_atom_line_only"]
    stable_tasks = sum(
        full["task_delta"][task] > 0
        and sum(
            row[f"concat_full"] > row[f"concat_atom_line_only"]
            for row in rows if row["task"] == task
        ) >= 4
        for task in TASKS
    )
    global_evidence = (
        full["macro_delta"] > 0
        and full["median_task_delta"] > 0
        and full["positive_tasks"] >= 5
    )
    return {
        "schema": "mts-trimer-sensitivity-probe-v1",
        "macro8": macro, "tasks": tasks, "comparisons": comparisons,
        "stage_c_decision": {
            "global_evidence": bool(global_evidence),
            "stable_positive_tasks": int(stable_tasks),
            "run_matched_5k": bool(global_evidence or stable_tasks >= 2),
        },
    }


def analyze_probes():
    rows = []
    for task_index, task in enumerate(TASKS):
        with np.load(OUTPUT / "sensitivity_features" / f"{task}.npz", allow_pickle=False) as payload:
            data = {key: np.asarray(payload[key]) for key in payload.files}
        valid = data["valid_3d"].astype(bool)
        for fold, split in enumerate(_folds(task)):
            train = np.asarray(split["train_indices"], dtype=np.int64)
            test = np.asarray(split["test_indices"], dtype=np.int64)
            train, test = train[valid[train]], test[valid[test]]
            seed = 74000 + task_index * 100 + fold
            row = {"task": task, "fold": fold, "n_train": len(train), "n_test": len(test)}
            p0 = scalar_probe(data["z_o8"][train], data["target"][train], data["z_o8"][test], data["target"][test], seed=seed)
            row["p0_o8"] = p0["r2"]
            for variant in VARIANTS:
                glt = data[f"z_glt_{variant}"]
                probe = scalar_probe(glt[train], data["target"][train], glt[test], data["target"][test], seed=seed)
                row[f"glt_{variant}"] = probe["r2"]
                concat = np.concatenate((data["z_o8"], glt), axis=1)
                probe = scalar_probe(concat[train], data["target"][train], concat[test], data["target"][test], seed=seed)
                row[f"concat_{variant}"] = probe["r2"]
            rows.append(row)
            print(f"probe {task}/fold{fold}", flush=True)
    pd.DataFrame(rows).to_csv(OUTPUT / "sensitivity_fold_results.csv", index=False)
    summary = _probe_summary(rows)
    _atomic_json(OUTPUT / "sensitivity_summary.json", summary)
    print(json.dumps(summary["stage_c_decision"], sort_keys=True), flush=True)


def analyze_prediction_association():
    metrics = pd.read_csv(OUTPUT / "downstream_geometry_metrics.csv")
    metric_by_key = metrics.set_index("sample_key")
    metric_names = (
        "boundary_distance_relative_asymmetry",
        "span1_angle_abs_deg_mean",
        "kabsch_rmsd_over_rg_mean",
        "mmff_energy_per_heavy_atom",
        "ru_atoms",
    )
    rows = []
    q4_worse_tasks = {name: 0 for name in metric_names}
    for task in TASKS:
        frame = pd.read_csv(ROOT / "data/raw" / f"smi_{task}.csv")
        sample_keys = np.asarray([sample_key_from_smiles(value).hex() for value in frame["smiles"].astype(str)])
        delta_se = np.full(len(frame), np.nan, dtype=np.float64)
        covered = np.zeros(len(frame), dtype=bool)
        for fold in range(5):
            paths = [FORMAL / mode / f"predictions/42/{task}/fold_{fold}.npz" for mode in ("o8_only", "o8_glt_graph")]
            with np.load(paths[0], allow_pickle=False) as first, np.load(paths[1], allow_pickle=False) as second:
                indices = np.asarray(first["sample_indices"], dtype=np.int64)
                if not np.array_equal(indices, second["sample_indices"]) or not np.allclose(first["y_true"], second["y_true"]):
                    raise RuntimeError(f"prediction pairing mismatch: {task}/fold{fold}")
                if covered[indices].any():
                    raise RuntimeError(f"duplicate held-out prediction index: {task}/fold{fold}")
                y = np.asarray(first["y_true"], dtype=np.float64)
                delta_se[indices] = np.square(y - first["y_pred"]) - np.square(y - second["y_pred"])
                covered[indices] = True
        if not covered.all():
            raise RuntimeError(f"held-out predictions do not cover {task}")
        for metric_name in metric_names:
            values = np.asarray([
                metric_by_key.loc[key, metric_name] if key in metric_by_key.index else np.nan
                for key in sample_keys
            ], dtype=np.float64)
            finite = np.isfinite(values) & np.isfinite(delta_se)
            correlation = spearmanr(values[finite], delta_se[finite]) if finite.sum() >= 3 else None
            quartiles = pd.qcut(values[finite], 4, labels=False, duplicates="drop")
            low = delta_se[finite][np.asarray(quartiles) == 0]
            high_index = int(np.max(quartiles)) if len(quartiles) else 0
            high = delta_se[finite][np.asarray(quartiles) == high_index]
            q1_mean = float(low.mean()) if low.size else None
            q4_mean = float(high.mean()) if high.size else None
            q4_worse = bool(q1_mean is not None and q4_mean is not None and q4_mean < q1_mean)
            q4_worse_tasks[metric_name] += int(q4_worse)
            rows.append({
                "task": task, "metric": metric_name, "count": int(finite.sum()),
                "spearman": float(correlation.statistic) if correlation is not None else np.nan,
                "spearman_pvalue": float(correlation.pvalue) if correlation is not None else np.nan,
                "q1_delta_se_mean": q1_mean, "q1_delta_se_median": float(np.median(low)) if low.size else np.nan,
                "q4_delta_se_mean": q4_mean, "q4_delta_se_median": float(np.median(high)) if high.size else np.nan,
                "q4_minus_q1_mean": (q4_mean - q1_mean) if q1_mean is not None and q4_mean is not None else np.nan,
                "q4_worse_than_q1": q4_worse,
            })
    pd.DataFrame(rows).to_csv(OUTPUT / "prediction_error_association.csv", index=False)
    _atomic_json(OUTPUT / "prediction_error_association.json", {
        "schema": "mts-trimer-prediction-error-association-v1",
        "delta_se": "squared_error_o8_minus_squared_error_fused; positive is improvement",
        "q4_worse_task_counts": q4_worse_tasks,
        "rows": rows,
    })


def write_report():
    sensitivity = json.loads((OUTPUT / "sensitivity_summary.json").read_text(encoding="utf-8"))
    association = json.loads((OUTPUT / "prediction_error_association.json").read_text(encoding="utf-8"))
    full = sensitivity["comparisons"]["concat_full_minus_atom_line_only"]
    lines = [
        "# MTS Trimer 3D Sensitivity", "",
        "> Existing seed-42 GraphGate-v1 20k checkpoint; frozen encoders; historical_shared5 held-out folds.", "",
        f"- Full concat minus atom-line concat macro delta: `{full['macro_delta']:+.6f}`",
        f"- Median task delta: `{full['median_task_delta']:+.6f}`",
        f"- Positive tasks: `{full['positive_tasks']}/8`",
        f"- Matched 5k decision: `{str(sensitivity['stage_c_decision']['run_matched_5k']).lower()}`", "",
        "## High-asymmetry Q4 worse task counts", "",
    ]
    for name, count in association["q4_worse_task_counts"].items():
        lines.append(f"- {name}: `{count}/8`")
    lines += [
        "", "阶段B是训练后敏感性诊断，不单独构成3D因果证据。",
        "只有条件触发的matched 5k能够比较Full-3D与Geometry-Off训练效应。", "",
    ]
    path = OUTPUT / "sensitivity_report.md"
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text("\n".join(lines), encoding="utf-8")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    extract_parser = sub.add_parser("extract")
    extract_parser.add_argument("--device", default="cuda:0")
    extract_parser.add_argument("--tasks", nargs="+", default=list(TASKS))
    sub.add_parser("analyze")
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if args.command == "extract":
        extract(args.device, args.tasks)
    else:
        analyze_probes()
        analyze_prediction_association()
        write_report()


if __name__ == "__main__":
    main()
