#!/usr/bin/env python3
"""Frozen Ridge admission probe for the MTS-GLT-v2 Compact19 descriptor."""

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

from src.analysis.mts_glt_postmortem import scalar_probe  # noqa: E402
from src.dataset import UniDataset  # noqa: E402
from src.modules import MTSGraphLineModelV2, compact_trimer_descriptors  # noqa: E402
from src.utils import get_data_loader  # noqa: E402

TASKS = ("eat", "eea", "egb", "egc", "ei", "eps", "nc", "xc")
SIDECAR = ROOT / "data/processed/mips_trimer_scage/periodic_line_glt_v1/downstream_union"
SPLITS = ROOT / "data/splits/mips_shared5"


def _model(checkpoint, layers, attention_variant, device):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = payload.get("state_dict", payload)
    model_state = {
        key[len("model."):]: value
        for key, value in state.items() if str(key).startswith("model.")
    }
    model = MTSGraphLineModelV2(
        glt_layers=layers, glt_attention_variant=attention_variant,
        use_compact19=False,
    )
    model.load_state_dict(model_state, strict=True)
    return model.to(device).eval()


def _dataset(task):
    return UniDataset(
        root=str(ROOT / "data"), dataset=f"smi_{task}",
        smiles_model_name=str(
            ROOT / "pretrained_models/encoders/PubChem10M_SMILES_BPE_450k"
        ),
        graph_encoder_type="mips_trimer_scage", graph_input="star_linking",
        use_feature_cache=True, feature_source_dataset="smi_all", fp_mode="ecfp",
        cache_layers="ru_base,topology,trimer,md200", mips_core="paper_corrected",
        mips_max_hops=2, mips_use_descriptors=True,
        mips_descriptor_protocol="source_star_sub", spatial_mode="trimer_scage",
        graph_geometry_mode="trimer_scage_mcl",
        topology_representation="canonical_lifted", trimer_num_candidates=4,
        trimer_max_heavy_atoms=384, mips_variant="O8",
        experiment_id="mts_glt_v2_compact19_probe", modalities=("graph",),
        periodic_line_glt_sidecar=str(SIDECAR),
    )


def _extract(model, task, device, output_root):
    dataset = _dataset(task)
    loader = get_data_loader(
        dataset, batch_size=64, shuffle=False, drop_last=False,
        num_workers=2, pin_memory=device.type == "cuda", persistent_workers=True,
    )
    o8, glt, compact, valid = [], [], [], []
    with torch.inference_mode():
        for batch in loader:
            batch = batch.to(device, non_blocking=True)
            views = model.encode_views(batch)
            descriptor, descriptor_valid = compact_trimer_descriptors(batch)
            o8.append(views["z_o8"].float().cpu())
            glt.append(views["z_glt"].float().cpu())
            compact.append(descriptor.float().cpu())
            valid.append((views["valid_3d"] & descriptor_valid).cpu())
    arrays = {
        "target": np.asarray(dataset.raw_targets, dtype=np.float64),
        "z_o8": torch.cat(o8).numpy(),
        "z_glt": torch.cat(glt).numpy(),
        "compact19": torch.cat(compact).numpy(),
        "valid": torch.cat(valid).numpy().astype(bool),
    }
    if any(not np.isfinite(value).all() for value in arrays.values()):
        raise RuntimeError(f"non-finite Compact19 frozen feature for {task}")
    feature_root = output_root / "frozen_features"
    feature_root.mkdir(parents=True, exist_ok=True)
    with (feature_root / f"{task}.npz").open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    return arrays


def _splits(task):
    payload = json.loads((SPLITS / f"{task}.json").read_text(encoding="utf-8"))
    return payload["folds"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--layers", required=True, type=int, choices=(6, 12))
    parser.add_argument(
        "--attention-variant", required=True, choices=("mips", "paper")
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-root", default="results/mts_glt_v2/compact19_probe")
    args = parser.parse_args()
    device = torch.device(args.device)
    output_root = (ROOT / args.output_root).resolve()
    model = _model(
        args.checkpoint.resolve(), args.layers, args.attention_variant, device
    )
    rows = []
    for task_index, task in enumerate(TASKS):
        path = output_root / "frozen_features" / f"{task}.npz"
        if path.is_file():
            with np.load(path, allow_pickle=False) as loaded:
                arrays = {key: np.asarray(loaded[key]) for key in loaded.files}
        else:
            arrays = _extract(model, task, device, output_root)
        base = np.concatenate((arrays["z_o8"], arrays["z_glt"]), axis=1)
        augmented = np.concatenate((base, arrays["compact19"]), axis=1)
        valid = arrays["valid"].astype(bool)
        for fold, split in enumerate(_splits(task)):
            train = np.asarray(split["train_indices"], dtype=np.int64)
            test = np.asarray(split["test_indices"], dtype=np.int64)
            train, test = train[valid[train]], test[valid[test]]
            seed = 52000 + task_index * 100 + fold
            o8_only = scalar_probe(
                arrays["z_o8"][train], arrays["target"][train],
                arrays["z_o8"][test], arrays["target"][test], seed=seed,
            )
            baseline = scalar_probe(
                base[train], arrays["target"][train],
                base[test], arrays["target"][test], seed=seed,
            )
            descriptor = scalar_probe(
                augmented[train], arrays["target"][train],
                augmented[test], arrays["target"][test], seed=seed,
            )
            rows.append({
                "task": task, "fold": fold,
                "o8_r2": float(o8_only["r2"]),
                "base_r2": float(baseline["r2"]),
                "concat_delta": float(baseline["r2"] - o8_only["r2"]),
                "compact19_r2": float(descriptor["r2"]),
                "delta": float(descriptor["r2"] - baseline["r2"]),
                "train_count": int(train.size), "test_count": int(test.size),
            })
    frame = pd.DataFrame(rows)
    summary = {}
    for task in TASKS:
        selected = frame[frame.task == task]
        summary[task] = {
            "o8_mean_r2": float(selected.o8_r2.mean()),
            "base_mean_r2": float(selected.base_r2.mean()),
            "concat_mean_delta": float(selected.concat_delta.mean()),
            "compact19_mean_r2": float(selected.compact19_r2.mean()),
            "mean_delta": float(selected.delta.mean()),
            "positive_folds": int((selected.delta > 0).sum()),
        }
    deltas = np.asarray([summary[task]["mean_delta"] for task in TASKS])
    concat_deltas = np.asarray([
        summary[task]["concat_mean_delta"] for task in TASKS
    ])
    report = {
        "schema": "mts-glt-v2-compact19-ridge-probe-v1",
        "tasks": summary,
        "macro_delta": float(deltas.mean()),
        "median_task_delta": float(np.median(deltas)),
        "positive_tasks": int((deltas > 0).sum()),
        "frozen_concat_macro_delta": float(concat_deltas.mean()),
        "frozen_concat_median_task_delta": float(np.median(concat_deltas)),
        "frozen_concat_positive_tasks": int((concat_deltas > 0).sum()),
        "admitted": bool(
            deltas.mean() > 0 and np.median(deltas) > 0 and (deltas > 0).sum() >= 5
        ),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_root / "fold_results.csv", index=False)
    temporary = output_root / "report.json.tmp"
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(output_root / "report.json")
    print(json.dumps({
        key: report[key] for key in (
            "macro_delta", "median_task_delta", "positive_tasks", "admitted",
            "frozen_concat_macro_delta", "frozen_concat_positive_tasks",
        )
    }, sort_keys=True))


if __name__ == "__main__":
    main()
