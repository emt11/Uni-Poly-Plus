#!/usr/bin/env python3
"""C7 spike-batch attribution and C8 healthy-checkpoint bad-batch test.

Recovers the deterministic global batches of steps 2659/2660/2676 from the
original sampling stream (sample keys only; no optimizer step, no training),
attributes the length/angle error to individual graphs and bonds, and re-runs
the step-2676 batch under the 2600/2660/2700 model states.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.dataset.glt_dual_pretrain import prepare_pretrain_sample, pretrain_collate
from src.modules.glt_dual_pretrain import DualPretrainer, per_graph
from src.training.glt_dual_runtime import write_json, open_source, OrderedSampleStream

WORLD, MICRO, ACCUM = 4, 84, 3
STEPS = (2659, 2660, 2676)
MODELS = {"2600": "resume_02600.pt", "2660": "resume_02660.pt", "2700": "resume_02700.pt"}


def batch_positions(step, batch_size):
    return [step * batch_size + offset * WORLD * MICRO + rank * MICRO + local
            for offset in range(ACCUM) for rank in range(WORLD) for local in range(MICRO)]


def collate_chunk(chunk, device):
    data, labels = pretrain_collate([(row["noisy"], row["labels"]) for row in chunk])
    data = data.to(device)
    labels = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in labels.items()}
    return data, labels


def forward_with(model, data, labels):
    captured, handles = {}, []
    for name, module in (("length_out", model.length_head[2]),
                         ("angle_pre", model.angle_head[2])):
        handles.append(module.register_forward_hook(
            lambda m, i, o, name=name: captured.__setitem__(name, o.detach())))
    try:
        with torch.autocast("cuda", enabled=False):
            out = model(data, labels)
    finally:
        for handle in handles:
            handle.remove()
    return out, captured


def evaluate(model, chunks, device, want_bonds=False, keys=None):
    length_errors, angle_errors = [], []
    bonds, per_graph_rows = [], []
    graph_offset = 0
    for chunk_index, chunk in enumerate(chunks):
        data, labels = collate_chunk(chunk, device)
        out, captured = forward_with(model, data, labels)
        center = data.bond_center.bool()
        pred = captured["length_out"].float().flatten()
        if int(center.sum()):
            err = (pred - labels["distance"].float()).square()
            means, valid = per_graph(err, data.bond_batch[center], len(chunk))
            for graph in range(len(chunk)):
                if bool(valid[graph]):
                    per_graph_rows.append({
                        "graph": graph_offset + graph,
                        "key": keys[graph_offset + graph] if keys else None,
                        "valid_length_targets": int((data.bond_batch[center] == graph).sum()),
                        "length_mse": float(means[graph]),
                    })
            if want_bonds:
                batch_index = data.bond_batch[center]
                for local in torch.argsort(err, descending=True)[:100].tolist():
                    graph = int(batch_index[local])
                    bonds.append({
                        "key": keys[graph_offset + graph] if keys else None,
                        "graph": graph_offset + graph,
                        "token_index": int(torch.nonzero(center).flatten()[local]),
                        "z_a": int(data.bond_z_a[center][local]),
                        "z_b": int(data.bond_z_b[center][local]),
                        "bond_type": int(data.bond_type[center][local]),
                        "clean_target_length": float(labels["distance"][local]),
                        "noisy_input_length": float(data.bond_distance[center][local]),
                        "predicted_length": float(pred[local]),
                        "squared_error": float(err[local]),
                        "finite": bool(torch.isfinite(pred[local])
                                       and torch.isfinite(labels["distance"][local])),
                    })
            length_errors.append(err)
        pre = captured["angle_pre"].float().flatten()
        if pre.numel():
            err = (torch.tanh(pre) - labels["angle_cos"].float()).square()
            angle_errors.append(err)
        graph_offset += len(chunk)
    length_errors = torch.cat(length_errors) if length_errors else torch.empty(0)
    angle_errors = torch.cat(angle_errors) if angle_errors else torch.empty(0)
    result = {
        "length_mse_mean": float(length_errors.mean()) if length_errors.numel() else None,
        "length_mse_max": float(length_errors.max()) if length_errors.numel() else None,
        "angle_mse_mean": float(angle_errors.mean()) if angle_errors.numel() else None,
        "angle_mse_max": float(angle_errors.max()) if angle_errors.numel() else None,
        "per_graph": per_graph_rows,
    }
    if want_bonds:
        result["top_bonds"] = sorted(bonds, key=lambda r: -r["squared_error"])[:100]
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--cohort-root", required=True)
    parser.add_argument("--dual-static-root", required=True)
    parser.add_argument("--pretrain-target-root", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--config", default="configs/mts/glt_dual_three_task_concat.json")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    out = Path(args.out_dir)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    batch_size = int(config["global_batch"])
    source, _ = open_source(args.cohort_root, args.cache_root,
                            dual_static_root=args.dual_static_root,
                            pretrain_target_root=args.pretrain_target_root)
    report = {"steps": {}, "world": WORLD, "microbatch": MICRO, "accumulation": ACCUM,
              "batch_size": batch_size, "scope": "full global batch (1008 samples) per step"}
    try:
        stream = OrderedSampleStream(len(source), config["seed"])
        prepared = {}
        attribution = {}
        for step in STEPS:
            positions = batch_positions(step, batch_size)
            rows = []
            keys = []
            for position in positions:
                index = stream.index_at(position)
                key = source.samples[index][0].hex()
                keys.append(key)
                noisy, labels = prepare_pretrain_sample(
                    *source[index], seed=config["seed"], key=key, position=position,
                    sigma=config["noise_sigma"], ratio=config["atom_mask_ratio"],
                    static=source.static_for(index), target=source.target_for(index))
                rows.append({"position": position, "key": key, "noisy": noisy, "labels": labels})
            prepared[step] = (rows, keys)
            model = DualPretrainer("concat", collect_diagnostics=False).to(device)
            state = torch.load(Path(args.checkpoint_dir) / MODELS["2660"],
                               map_location="cpu", weights_only=False)
            model.load_state_dict(state["model"], strict=True)
            model.eval()
            chunks = [rows[i:i + MICRO] for i in range(0, len(rows), MICRO)]
            attribution[step] = evaluate(model, chunks, device,
                                         want_bonds=(step == 2676), keys=keys)
            del model
            torch.cuda.empty_cache()
            print(json.dumps({"step": step, "prepared": len(rows),
                              "length_mse_mean": attribution[step]["length_mse_mean"]}), flush=True)
        # concentration
        for step, value in attribution.items():
            errors = sorted((row["length_mse"] for row in value["per_graph"]), reverse=True)
            total = sum(errors) or 1e-12
            value["concentration"] = {
                "graphs": len(errors),
                "top1_fraction": errors[0] / total if errors else None,
                "top5_fraction": sum(errors[:5]) / total if errors else None,
                "top10_fraction": sum(errors[:10]) / total if errors else None,
                "top20_fraction": sum(errors[:20]) / total if errors else None,
                "classification": ("SINGLE_SAMPLE_DOMINANT" if errors and errors[0] / total > 0.5
                                   else "FEW_SAMPLES_DOMINANT" if sum(errors[:10]) / total > 0.8
                                   else "BROAD_BATCH_FAILURE"),
            }
            value["top_graphs"] = sorted(value["per_graph"],
                                         key=lambda r: -r["length_mse"])[:20]
            del value["per_graph"]
        write_json(out / "spike_batch_attribution.json", attribution)

        # C8: the step-2676 batch under three model states
        rows, keys = prepared[2676]
        chunks = [rows[i:i + MICRO] for i in range(0, len(rows), MICRO)]
        bad_batch = {}
        for label, filename in MODELS.items():
            model = DualPretrainer("concat").to(device)
            state = torch.load(Path(args.checkpoint_dir) / filename,
                               map_location="cpu", weights_only=False)
            model.load_state_dict(state["model"], strict=True)
            model.eval()
            result = evaluate(model, chunks, device, keys=keys)
            bad_batch[label] = {k: v for k, v in result.items() if k != "per_graph"}
            del model
            torch.cuda.empty_cache()
        write_json(out / "bad_batch_checkpoint_test.json", bad_batch)

        # C7.3 sample history for the worst graphs
        worst = [row["key"] for row in attribution[2676]["top_graphs"][:5] if row["key"]]
        history = {}
        position_by_key = {row["key"]: row["position"] for row in rows}
        for key in dict.fromkeys(worst):
            index = next(i for i in range(len(source)) if source.samples[i][0].hex() == key)
            position = position_by_key[key]
            single = []
            for label, filename in MODELS.items():
                model = DualPretrainer("concat").to(device)
                state = torch.load(Path(args.checkpoint_dir) / filename,
                                   map_location="cpu", weights_only=False)
                model.load_state_dict(state["model"], strict=True)
                model.eval()
                history_row = prepared[2676][0][next(i for i, r in enumerate(rows) if r["key"] == key)]
                result = evaluate(model, [[history_row]], device, keys=[key])
                single.append({"model": label, "length_mse_mean": result["length_mse_mean"],
                               "angle_mse_mean": result["angle_mse_mean"]})
                del model
                torch.cuda.empty_cache()
            history[key] = {"position": position, "by_model": single}
        write_json(out / "sample_history.json", history)
        report["steps"] = {str(step): {"samples": len(prepared[step][0])} for step in STEPS}
        write_json(out / "spike_summary.json", report)
        print(json.dumps({"status": "C7-C8 done"}, indent=2))
    finally:
        source.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
