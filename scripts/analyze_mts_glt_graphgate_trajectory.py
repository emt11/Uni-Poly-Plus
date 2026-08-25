#!/usr/bin/env python3
"""Compare frozen GraphGate representations at 5k and 20k."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from torch_geometric.utils import scatter, softmax


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.mts_glt_postmortem import (  # noqa: E402
    cross_predictability, distribution, linear_cka, residual_probe, scalar_probe,
)
from src.dataset import UniDataset  # noqa: E402
from src.dataset.lmdb_cache import sample_key_from_smiles  # noqa: E402
from src.modules import MTSGraphGateModel  # noqa: E402
from src.utils import get_data_loader  # noqa: E402


TASKS = ("eat", "eea", "egb", "egc", "ei", "eps", "nc", "xc")
OUTPUT = ROOT / "results/mts_glt_graphgate_v1/trajectory_selection_v1"
FEATURES = OUTPUT / "representation_features"
SIDECAR = ROOT / "data/processed/mips_trimer_scage/periodic_line_glt_central_v1/downstream_union"
SPLITS = ROOT / "data/splits/mips_shared5"
CHECKPOINTS = {
    "005k": ROOT / "results/mts_glt_graphgate_v1/pretrain/mts_glt_graphgate_probe_005k.pth",
    "020k": ROOT / "results/mts_glt_graphgate_v1/pretrain/mts_glt_graphgate_probe_020k.pth",
}


def _atomic_json(path, payload):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _load_model(step, device):
    payload = torch.load(CHECKPOINTS[step], map_location="cpu", weights_only=True)
    expected_step = 5000 if step == "005k" else 20000
    if payload.get("schema") != "mts-glt-graphgate-v1-probe-v1" or int(payload.get("step", -1)) != expected_step:
        raise RuntimeError(f"unexpected GraphGate checkpoint: {CHECKPOINTS[step]}")
    model = MTSGraphGateModel(layers=6)
    namespaces = payload["namespaces"]
    o8_state = model.o8_encoder.state_dict()
    expected_o8 = {key for key in o8_state if not key.startswith("md_residual.") and not key.startswith("star_distance_bias.")}
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
        use_feature_cache=True, feature_source_dataset="smi_all", fp_mode="disabled",
        cache_layers="ru_base,topology,trimer,md200", mips_core="paper_corrected",
        mips_max_hops=2, mips_use_descriptors=True,
        mips_descriptor_protocol="source_star_sub", spatial_mode="trimer_scage",
        graph_geometry_mode="trimer_scage_mcl", topology_representation="canonical_lifted",
        trimer_num_candidates=4, trimer_max_heavy_atoms=384, mips_variant="O8",
        experiment_id="mts_glt_graphgate_trajectory_selection_v1", modalities=("graph",),
        periodic_line_glt_sidecar=str(SIDECAR),
    )


def _query_statistics(module, line_states, token_batch, token_valid, query_valid, expected):
    pool = module.query_pool
    graph_count = int(query_valid.numel())
    normalized = pool.line_norm(line_states)
    query = pool.query_norm(pool.query)
    q = pool.q(query).view(1, pool.heads, pool.head_dim).expand(graph_count, -1, -1)
    k = pool.k(normalized).view(-1, pool.heads, pool.head_dim)
    v = pool.v(normalized).view(-1, pool.heads, pool.head_dim)
    eligible = token_valid.bool() & query_valid.bool()[token_batch.long()]
    message = torch.zeros((graph_count, pool.heads, pool.head_dim), device=line_states.device, dtype=line_states.dtype)
    entropy = torch.zeros((graph_count, pool.heads), device=line_states.device, dtype=torch.float32)
    top1 = torch.zeros_like(entropy); top5 = torch.zeros_like(entropy)
    if bool(eligible.any()):
        indices = torch.nonzero(eligible, as_tuple=False).flatten()
        groups = token_batch.long()[indices]
        logits = (q[groups] * k[indices]).sum(-1) * (pool.head_dim ** -0.5)
        weights = softmax(logits, groups, num_nodes=graph_count).to(v.dtype)
        message = scatter(weights.unsqueeze(-1) * v[indices], groups, dim=0, dim_size=graph_count, reduce="sum")
        for graph in torch.nonzero(query_valid, as_tuple=False).flatten().tolist():
            selected = weights[groups == graph].float()
            entropy[graph] = -(selected * torch.log(selected.clamp_min(1e-12))).sum(0)
            ordered = torch.sort(selected, dim=0, descending=True).values
            top1[graph] = ordered[0]
            top5[graph] = ordered[: min(5, ordered.size(0))].sum(0)
    recomputed = pool.final_norm(pool.query.unsqueeze(0) + pool.out(message.reshape(graph_count, -1)))
    recomputed = recomputed * query_valid.to(recomputed.dtype).unsqueeze(-1)
    torch.testing.assert_close(recomputed, expected, rtol=1e-5, atol=1e-6)
    return {
        "query_entropy": entropy.mean(1), "query_top1": top1.mean(1),
        "query_top5": top5.mean(1), "query_effective_tokens": torch.exp(entropy).mean(1),
    }


def extract(step, device_name, tasks):
    device = torch.device(device_name)
    model = _load_model(step, device)
    destination = FEATURES / step; destination.mkdir(parents=True, exist_ok=True)
    for task in tasks:
        dataset = _dataset(task); frame = pd.read_csv(ROOT / "data/raw" / f"smi_{task}.csv")
        if len(dataset) != len(frame):
            raise RuntimeError(f"dataset/CSV mismatch: {task}")
        loader = get_data_loader(dataset, batch_size=64, shuffle=False, drop_last=False,
                                 num_workers=2, pin_memory=device.type == "cuda", persistent_workers=True)
        values = {name: [] for name in ("z_o8", "z_glt", "valid_3d", "query_entropy", "query_top1", "query_top5", "query_effective_tokens")}
        with torch.inference_mode():
            for batch in loader:
                batch = batch.to(device, non_blocking=True)
                z_o8, _ = model.o8_encoder._forward_impl(batch, use_star=False, use_md=False)
                glt = model.glt_line_encoder(batch)
                stats = _query_statistics(
                    model.glt_line_encoder, glt["line_states"], batch.glt_token_batch,
                    batch.glt_token_valid, glt["query_valid"], glt["graph_geometry"],
                )
                values["z_o8"].append(z_o8.float().cpu())
                values["z_glt"].append(glt["graph_geometry"].float().cpu())
                values["valid_3d"].append(glt["query_valid"].cpu())
                for name, tensor in stats.items(): values[name].append(tensor.cpu())
        arrays = {
            "row_index": np.arange(len(dataset), dtype=np.int64),
            "sample_key": np.asarray([sample_key_from_smiles(str(value)).hex() for value in frame["smiles"]]),
            "target": np.asarray(dataset.raw_targets, dtype=np.float64),
        }
        for name, parts in values.items():
            arrays[name] = torch.cat(parts).numpy().astype(bool if name == "valid_3d" else np.float32)
        for name, array in arrays.items():
            if name != "sample_key" and not np.isfinite(array).all():
                raise RuntimeError(f"non-finite feature: {step}/{task}/{name}")
        with (destination / f"{task}.npz").open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        print(f"extracted {step}/{task}: rows={len(dataset)} valid={int(arrays['valid_3d'].sum())}", flush=True)


def _folds(task):
    payload = json.loads((SPLITS / f"{task}.json").read_text(encoding="utf-8"))
    return payload["folds"]


def _spectral(values):
    values = np.asarray(values, dtype=np.float64)
    centered = values - values.mean(0, keepdims=True)
    std = values.std(0, ddof=0)
    singular = np.linalg.svd(centered, compute_uv=False)
    energy = np.square(singular); probabilities = energy / max(float(energy.sum()), 1e-12)
    entropy = -np.sum(probabilities[probabilities > 0] * np.log(probabilities[probabilities > 0]))
    return {
        "channel_std_mean": float(std.mean()), "channel_std_median": float(np.median(std)),
        "effective_rank": float(np.exp(entropy)),
        "sv_top1_mass": float(probabilities[:1].sum()),
        "sv_top10_mass": float(probabilities[:10].sum()),
        "sv_top50_mass": float(probabilities[:50].sum()),
    }


def _pairwise_cosine(values):
    values = np.asarray(values, dtype=np.float64)
    normalized = values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)
    matrix = normalized @ normalized.T
    return distribution(matrix[np.triu_indices(len(matrix), k=1)])


def analyze():
    probe_rows, drift_rows = [], []
    task_cache = {}
    for task_index, task in enumerate(TASKS):
        datasets = {}
        for step in ("005k", "020k"):
            with np.load(FEATURES / step / f"{task}.npz", allow_pickle=False) as payload:
                datasets[step] = {key: np.asarray(payload[key]) for key in payload.files}
        first, second = datasets["005k"], datasets["020k"]
        if not np.array_equal(first["sample_key"], second["sample_key"]) or not np.array_equal(first["valid_3d"], second["valid_3d"]):
            raise RuntimeError(f"5k/20k key or valid-mask mismatch: {task}")
        valid = first["valid_3d"].astype(bool); task_cache[task] = datasets
        for fold, split in enumerate(_folds(task)):
            train = np.asarray(split["train_indices"], dtype=np.int64); test = np.asarray(split["test_indices"], dtype=np.int64)
            train, test = train[valid[train]], test[valid[test]]
            step_results = {}
            for step_index, step in enumerate(("005k", "020k")):
                data = datasets[step]; y_train, y_test = data["target"][train], data["target"][test]
                o8_train, o8_test = data["z_o8"][train], data["z_o8"][test]
                glt_train, glt_test = data["z_glt"][train], data["z_glt"][test]
                seed = 91000 + task_index * 100 + fold * 10 + step_index
                p1 = scalar_probe(o8_train, y_train, o8_test, y_test, seed=seed)
                p2 = scalar_probe(glt_train, y_train, glt_test, y_test, seed=seed + 1)
                p3 = scalar_probe(np.concatenate((o8_train, glt_train), 1), y_train,
                                  np.concatenate((o8_test, glt_test), 1), y_test, seed=seed + 2)
                p4 = residual_probe(o8_train, glt_train, y_train, o8_test, glt_test, y_test, seed=seed + 3)
                row = {"task": task, "fold": fold, "step": step, "n_train": len(train), "n_test": len(test),
                       "p1": p1["r2"], "p2": p2["r2"], "p3": p3["r2"], "p4": p4["r2"],
                       "c3": p3["r2"] - p1["r2"], "c4": p4["r2"] - p1["r2"]}
                probe_rows.append(row); step_results[step] = row
            drift = {"task": task, "fold": fold,
                     "c3_005k": step_results["005k"]["c3"], "c3_020k": step_results["020k"]["c3"],
                     "c4_005k": step_results["005k"]["c4"], "c4_020k": step_results["020k"]["c4"]}
            for step_index, step in enumerate(("005k", "020k")):
                data = datasets[step]; o8_train, o8_test = data["z_o8"][train], data["z_o8"][test]
                glt_train, glt_test = data["z_glt"][train], data["z_glt"][test]
                seed = 93000 + task_index * 100 + fold * 10 + step_index
                drift[f"cka_{step}"] = float(linear_cka(o8_test, glt_test))
                drift[f"o8_to_glt_ev_{step}"] = cross_predictability(o8_train, glt_train, o8_test, glt_test, seed=seed)["explained_variance"]
                drift[f"glt_to_o8_ev_{step}"] = cross_predictability(glt_train, o8_train, glt_test, o8_test, seed=seed + 1)["explained_variance"]
                for prefix, values in (("o8", o8_test), ("glt", glt_test)):
                    for name, value in _spectral(values).items(): drift[f"{prefix}_{name}_{step}"] = value
                    for name, value in _pairwise_cosine(values).items(): drift[f"{prefix}_pair_cosine_{name}_{step}"] = value
                for name in ("query_entropy", "query_top1", "query_top5", "query_effective_tokens"):
                    drift[f"{name}_{step}"] = float(np.mean(data[name][test]))
            drift_rows.append(drift)
            print(f"analysed {task}/fold{fold}", flush=True)
    pd.DataFrame(probe_rows).to_csv(OUTPUT / "representation_probe_folds.csv", index=False)
    pd.DataFrame(drift_rows).to_csv(OUTPUT / "representation_drift_folds.csv", index=False)
    probe = pd.DataFrame(probe_rows); drift = pd.DataFrame(drift_rows)
    summary = {"schema": "mts-glt-graphgate-representation-drift-v1", "steps": {}, "drift_means": {}}
    for step in ("005k", "020k"):
        selected = probe[probe.step == step]
        task = selected.groupby("task")[["p1", "p2", "p3", "p4", "c3", "c4"]].mean()
        summary["steps"][step] = {name: float(task[name].mean()) for name in task.columns}
    for column in drift.columns:
        if column not in {"task", "fold"}:
            summary["drift_means"][column] = float(pd.to_numeric(drift[column]).mean())
    _atomic_json(OUTPUT / "representation_drift_summary.json", summary)
    lines = ["# GraphGate 5k to 20k Representation Drift", "",
             "| Step | P1 | P2 | P3 | P4 | C3 | C4 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for step in ("005k", "020k"):
        row = summary["steps"][step]
        lines.append(f"| {step} | {row['p1']:.6f} | {row['p2']:.6f} | {row['p3']:.6f} | {row['p4']:.6f} | {row['c3']:+.6f} | {row['c4']:+.6f} |")
    lines.extend(["", "P3/P4决定互补性；CKA、cross-predictability、Query与rank指标仅用于解释。", ""])
    (OUTPUT / "representation_drift_report.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(); sub = parser.add_subparsers(dest="command", required=True)
    extract_parser = sub.add_parser("extract"); extract_parser.add_argument("--step", choices=("005k", "020k"), required=True)
    extract_parser.add_argument("--device", default="cuda:0"); extract_parser.add_argument("--tasks", nargs="+", default=list(TASKS))
    sub.add_parser("analyze"); args = parser.parse_args(argv)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if args.command == "extract": extract(args.step, args.device, args.tasks)
    else: analyze()


if __name__ == "__main__":
    main()
