#!/usr/bin/env python3
"""Fixed-batch diagnostics for the frozen Concat/KFuse pretraining checkpoints.

The six ``resume_*.pt`` checkpoints (Concat and KFuse at 2k/3k/5k) each run on
the SAME 16 real frozen records, selected from the original sampling stream
immediately after ``--base-step`` with the original world size, seed and
microbatch.  Batches of 8 reuse one prepared CPU sample list (motif mask,
perturbed coordinates, clean targets) across every mode, checkpoint and
precision, so all differences are checkpoint differences rather than data
differences.

Per checkpoint and precision (eval mode, no optimizer update) the report holds:
component losses with valid counts, target/prediction distributions, angle-head
pre-tanh and derivative statistics, representation scales, real Gaussian basis
statistics (read through read-only hooks on that same forward), the
"perturbed geometry used directly as the prediction" reference and - for the
first batch in FP32 - the component gradient norms (unweighted and
weight-scaled) and their pairwise cosines with respect to the full 3D encoder
parameter list.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.dataset.glt_dual import build_dual_sample
from src.dataset.glt_dual_pretrain import prepare_pretrain_sample, pretrain_collate
from src.modules.glt_dual_pretrain import DualPretrainer, per_graph
from src.training.glt_dual_runtime import write_json, open_source, OrderedSampleStream

MODES = ("concat", "kfuse")
STEPS = (2000, 3000, 5000)
RECORDS, BATCH = 16, 8
COMPONENTS = ("chem", "length", "angle", "fingerprint")


def _device(spec: str) -> torch.device:
    device = torch.device(spec)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA diagnostics requested without CUDA")
        torch.cuda.set_device(device)
    return device


def fixed_samples(source, config, base_step, records=RECORDS):
    """One deterministic CPU sample list from the original sampling stream."""

    world = 4
    micro, batch_size = int(config["microbatch"]), int(config["global_batch"])
    stream = OrderedSampleStream(len(source), config["seed"])
    base = base_step * batch_size
    entries = []
    for local in range(records):
        position = base + 0 * world * micro + 0 * micro + local
        index = stream.index_at(position)
        key = source.samples[index][0].hex()
        static = source.static_for(index)
        target = source.target_for(index)
        noisy, labels = prepare_pretrain_sample(
            *source[index], seed=config["seed"], key=key, position=position,
            sigma=config["noise_sigma"], ratio=config["atom_mask_ratio"],
            static=static, target=target,
        )
        clean = build_dual_sample(*source[index], static=static)
        entries.append({"position": position, "index": index, "key": key,
                        "clean": clean, "noisy": noisy, "labels": labels})
    return entries


def collate_chunk(chunk, device):
    data, labels = pretrain_collate([(row["noisy"], row["labels"]) for row in chunk])
    data = data.to(device)
    labels = {name: (value.to(device) if torch.is_tensor(value) else value)
              for name, value in labels.items()}
    return data, labels


def _angle_rows(sample):
    """Physical one-hop center angle rows, matching the training target rule."""

    path, mask, is_self = sample.line_path, sample.line_mask, sample.line_is_self
    rows = []
    for row in range(path.size(0)):
        if int(mask[row].sum()) != 1 or bool(is_self[row]):
            continue
        a, b = int(path[row, 0]), int(path[row, 1])
        if a < b and bool(sample.bond_center[a]) and bool(sample.bond_center[b]):
            rows.append((row, a, b))
    return rows


def perturbed_reference(chunk, labels):
    """Error of using the perturbed geometry directly, with training reductions.

    Length uses the perturbed center-bond distances against the clean distance
    targets; angles look up the perturbed cosine of exactly the (a, b) bond
    pairs that produced the clean angle targets.  Per-graph means and the
    valid-graph denominator follow the training loss.
    """

    graphs = labels["angle_graph"].new_tensor(len(chunk)).numel()
    graphs = len(chunk)
    center_sq, angle_sq, angle_graph_index = [], [], []
    length_graphs, angle_graphs = [], []
    without_center_angles = 0
    for graph, row in enumerate(chunk):
        noisy, clean, sample_labels = row["noisy"], row["clean"], row["labels"]
        center = clean.bond_center.bool()
        if int(center.sum()):
            error = (noisy.bond_distance[center].float()
                     - sample_labels["distance"].float()).square()
            length_graphs.append((graph, error))
        lookup = {}
        for angle_row, a, b in _angle_rows(clean):
            lookup[(a, b)] = float(noisy.line_angle[angle_row, 0].cos())
        pairs = sample_labels["angle_pairs"].tolist()
        if pairs:
            values = []
            for a, b in pairs:
                key = (int(a), int(b))
                if key not in lookup:
                    raise ValueError("angle pair is not a valid center one-hop relation")
                values.append(lookup[key])
            error = (torch.tensor(values, dtype=torch.float32)
                     - sample_labels["angle_cos"].float()).square()
            angle_graphs.append((graph, error))
        else:
            without_center_angles += 1
    def reduce(pairs_, count_graphs):
        if not pairs_:
            return None
        index = torch.cat([torch.full_like(error, graph, dtype=torch.long)
                           for graph, error in pairs_])
        values = torch.cat([error for _, error in pairs_])
        means, valid = per_graph(values, index, count_graphs)
        return float(means[valid].mean()) if bool(valid.any()) else None

    return {
        "length_from_perturbed_distance": reduce(length_graphs, graphs),
        "angle_from_perturbed_cosine": reduce(angle_graphs, graphs),
        "graphs": graphs,
        "samples_without_center_angles": without_center_angles,
        "reduction": "per-graph mean over valid graphs (same as the training term)",
    }


def component_gradients(model, out, weights, world, counts, params):
    """Component gradients over the FULL 3D encoder parameter list."""

    tensors = out["diagnostics"]["component_tensors"]
    # chem -> weight[0]; length AND angle -> the geometry weight[1]; fp -> weight[2].
    component_weight = {"chem": weights[0], "length": weights[1],
                        "angle": weights[1], "fingerprint": weights[2]}
    gradients, norms = {}, {}
    for index, name in enumerate(COMPONENTS):
        grad = torch.autograd.grad(tensors[name], params,
                                   retain_graph=True, allow_unused=True)
        aligned = [g if g is not None else torch.zeros_like(p)
                   for g, p in zip(grad, params)]
        flat = torch.cat([g.reshape(-1) for g in aligned]) if aligned else torch.zeros(1)
        gradients[name] = flat
        scale = component_weight[name] * world / max(1, int(counts))
        norms[name] = {
            "unweighted_norm": float(flat.norm()),
            "weight_scaled_norm": float(flat.norm()) * scale,
            "params_total": len(params),
            "params_with_grad": int(sum(g is not None for g in grad)),
        }
    cosines = {}
    for i, a in enumerate(COMPONENTS):
        for b in COMPONENTS[i + 1:]:
            x, y = gradients[a], gradients[b]
            denominator = float(x.norm() * y.norm())
            cosines[f"{a}__{b}"] = (float(torch.dot(x, y) / denominator)
                                    if denominator > 0 else None)
    return norms, cosines, ("cosines are null where a component gradient is zero" 
                            if any(v["unweighted_norm"] == 0 for v in norms.values()) else None)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--cohort-root", required=True)
    parser.add_argument("--dual-static-root", required=True)
    parser.add_argument("--pretrain-target-root", required=True)
    parser.add_argument("--concat-checkpoint-root", required=True)
    parser.add_argument("--kfuse-checkpoint-root", required=True)
    parser.add_argument("--report-json", required=True)
    parser.add_argument("--config-root", default="configs/mts")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--base-step", type=int, default=2000)
    args = parser.parse_args()
    device = _device(args.device)
    config = {
        mode: json.loads((Path(args.config_root) /
                          f"glt_dual_three_task_{mode}.json").read_text(encoding="utf-8"))
        for mode in MODES
    }
    roots = {"concat": args.concat_checkpoint_root, "kfuse": args.kfuse_checkpoint_root}
    source, _frame = open_source(args.cohort_root, args.cache_root,
                                 dual_static_root=args.dual_static_root,
                                 pretrain_target_root=args.pretrain_target_root)
    report = {
        "schema": "glt-dual-pretrain-fixed-batch-diagnostics-v1",
        "records": RECORDS, "batch": BATCH, "base_step": args.base_step,
        "device": str(device), "world": 4,
        "identity": {
            "cohort_manifest_hash": source.cohort["manifest_hash"],
            "cohort_root": str(Path(args.cohort_root).resolve()),
            "cache_root": str(Path(args.cache_root).resolve()),
            "main_bundle_hash": source.bundle.bundle_hash,
            "dual_static_manifest_hash": (source.static_cache.manifest_hash
                                          if source.static_cache is not None else None),
            "pretrain_target_manifest_hash": (source.target_cache.manifest_hash
                                              if source.target_cache is not None else None),
        },
        "checkpoints": {},
    }
    failures = []
    try:
        samples = fixed_samples(source, config["concat"], args.base_step)
        report["sample_key_positions"] = [
            {"position": row["position"], "key": row["key"]} for row in samples]
        report["checkpoints"] = {mode: {"steps": {}} for mode in MODES}
        for mode in MODES:
            run_config = config[mode]
            weights = list(run_config["loss_weights"])
            for step in STEPS:
                path = Path(roots[mode]) / f"resume_{step:05d}.pt"
                if not path.is_file():
                    failures.append(f"missing checkpoint: {path}")
                    report["checkpoints"][mode]["steps"][str(step)] = {
                        "status": "MISSING", "path": str(path)}
                    continue
                state = torch.load(path, map_location="cpu", weights_only=False)
                state_identity = dict(state.get("identity", {}))
                if int(state.get("step", -1)) != step:
                    failures.append(f"checkpoint step mismatch: {path}")
                if state_identity.get("config", {}).get("fusion_mode") != mode:
                    failures.append(f"checkpoint fusion mode mismatch: {path}")
                for name, expected in (
                    ("cohort_hash", report["identity"]["cohort_manifest_hash"]),
                    ("main_bundle_hash", report["identity"]["main_bundle_hash"]),
                    ("dual_static_manifest_hash", report["identity"]["dual_static_manifest_hash"]),
                    ("pretrain_target_manifest_hash", report["identity"]["pretrain_target_manifest_hash"]),
                ):
                    if expected is not None and state_identity.get(name) != expected:
                        failures.append(f"checkpoint {name} mismatch: {path}")
                model = DualPretrainer(mode, collect_diagnostics=True).to(device)
                model.load_state_dict(state["model"], strict=True)
                model.eval()
                entry = {"status": "OK", "path": str(path), "step": step,
                         "architecture": model.encoder.architecture_name,
                         "world_size_recorded": state_identity.get("world_size"),
                         "precisions": {}, "batches": []}
                for start in range(0, RECORDS, BATCH):
                    chunk = samples[start:start + BATCH]
                    data, labels = collate_chunk(chunk, device)
                    batch_id = f"batch{start // BATCH}"
                    entry["batches"].append({
                        "batch": batch_id,
                        "sample_keys": [row["key"] for row in chunk],
                        "perturbed_reference": perturbed_reference(chunk, labels),
                    })
                    for precision in ("fp32", "bf16"):
                        with torch.autocast(device.type, dtype=torch.bfloat16,
                                            enabled=precision == "bf16"):
                            out = model(data, labels)
                        diagnostics = out["diagnostics"]
                        entry["precisions"].setdefault(precision, {})[batch_id] = {
                            "components": diagnostics["components"],
                            "targets": diagnostics["targets"],
                            "predictions": diagnostics["predictions"],
                            "angle_head": diagnostics["angle_head"],
                            "representations": diagnostics["representations"],
                            "gaussian": diagnostics["gaussian"],
                        }
                        if batch_id == "batch0" and precision == "fp32":
                            params = [p for p in model.encoder.glt.parameters()
                                      if p.requires_grad]
                            norms, cosines, note = component_gradients(
                                model, out, weights, 4, int(out["counts"][1]),
                                params)
                            entry["precisions"][precision][batch_id][
                                "gradient_norms_3d_encoder"] = norms
                            entry["precisions"][precision][batch_id][
                                "gradient_cosines_3d_encoder"] = cosines
                            if note:
                                entry["precisions"][precision][batch_id][
                                    "gradient_note"] = note
                report["checkpoints"][mode]["steps"][str(step)] = entry
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()
    finally:
        source.close()
    report["failures"] = failures
    report["status"] = "PASS" if not failures else "FAIL"
    write_json(Path(args.report_json), report)
    print(json.dumps({"report": args.report_json, "status": report["status"],
                      "failures": failures}, indent=2))
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
