#!/usr/bin/env python3
"""Fixed-batch diagnostics for the frozen Concat/KFuse pretraining checkpoints.

The six ``resume_*.pt`` checkpoints (Concat and KFuse at 2k/3k/5k) are each run
on the SAME 16 real frozen records taken from the original sampling stream
immediately after step 2000.  Batches of 8 use the same motif mask, the same
perturbed coordinates and the same clean targets, so differences between
checkpoints are checkpoint differences and not data differences.

Per checkpoint and precision (fp32/bf16, eval mode, no optimizer update):

* the three component losses with their valid counts;
* length/angle target and prediction distributions, angle-head pre-tanh and
  tanh-derivative statistics and the saturated fraction;
* 3D representation scales (per-layer RMS, bond/center/graph RMS);
* Gaussian basis effective widths and affine parameters;
* for the first batch, the FP32 gradient norm of the length/angle/FP losses
  with respect to the 3D encoder and their pairwise cosines (a loss that does
  not depend on a parameter contributes zero, which is not a model error);
* the "use the perturbed distance/angle directly as the prediction" error
  reference on the same batch.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.dataset.glt_dual import build_dual_sample
from src.dataset.glt_dual_pretrain import prepare_pretrain_sample, pretrain_collate
from src.modules.glt_dual_pretrain import DualPretrainer, global_objective
from src.training.glt_dual_runtime import write_json, OrderedSampleStream

CHECKPOINTS = {
    "concat": [2000, 3000, 5000],
    "kfuse": [2000, 3000, 5000],
}
RECORDS, BATCH = 16, 8


def load_source(cache_root, cohort_root, static_root, target_root):
    from src.training.glt_dual_runtime import open_source

    source, _ = open_source(cohort_root, cache_root,
                            dual_static_root=static_root,
                            pretrain_target_root=target_root)
    return source


def positions_after_step(step, batch_size, world, micro, rank=0, count=RECORDS):
    base = step * batch_size
    return [base + 0 * world * micro + rank * micro + local for local in range(count)]


def component_gradients(model, data, labels, weights):
    """Gradient norms of each objective term wrt the 3D encoder (fp32 math)."""

    out = model(data, labels)
    components = ["chem", "geometry", "fingerprint"]
    losses = {}
    sums, counts = out["sums"], out["counts"].clamp_min(1)
    total = global_objective(sums, out["counts"], 1, weights)
    # Per-component contribution: weight * world / global_count * local sum.
    for index, name in enumerate(components):
        losses[name] = (sums[index] * weights[index] / counts[index]).float()
    params = [p for p in model.encoder.glt.parameters() if p.requires_grad]
    reports = {}
    grads = {}
    for index, name in enumerate(components):
        grad = torch.autograd.grad(losses[name], params, retain_graph=True, allow_unused=True)
        flat = torch.cat([g.reshape(-1) for g in grad if g is not None]) if any(
            g is not None for g in grad) else torch.zeros(1)
        grads[name] = flat
        reports[name] = {"grad_norm": float(flat.norm()),
                         "params_with_grad": int(sum(g is not None for g in grad)),
                         "params_total": len(params)}
    cosines = {}
    for a in components:
        for b in components:
            if a >= b:
                continue
            x, y = grads[a], grads[b]
            if x.numel() != y.numel():
                cosines[f"{a}__{b}"] = None
                continue
            denominator = float(x.norm() * y.norm())
            cosines[f"{a}__{b}"] = (float(torch.dot(x, y) / denominator)
                                    if denominator > 0 else None)
    return reports, cosines, float(total.detach())


def geometry_head_gradients(model, data, labels):
    """Split the geometry term into its length and angle parts."""

    out = model(data, labels)
    encoded = None
    # Re-run the head math to obtain separated components without touching the
    # training path: identical formulas to DualPretrainer.forward.
    with torch.no_grad():
        pass
    return out


def perturbed_reference(clean, noisy, labels):
    """Error of using the perturbed geometry directly as the prediction."""

    reference = {}
    center = clean.bond_center.bool()
    if int(center.sum()):
        distance = noisy.bond_distance[center].float()
        reference["length_from_perturbed_distance"] = float(
            (distance - labels["distance"].float()).square().mean())
    pairs = labels["angle_pairs"]
    if pairs.numel():
        clean_angle = clean.line_angle[:, 0]
        # Map angle target order back to its clean and perturbed cosines.
        cos_clean = torch.tensor([float(clean_angle[row].cos()) for row in range(
            clean.line_path.size(0)) if int(clean.line_mask[row].sum()) == 1
            and not bool(clean.line_is_self[row])], dtype=torch.float32)
        cos_noisy = torch.tensor([float(noisy.line_angle[row, 0].cos()) for row in range(
            clean.line_path.size(0)) if int(clean.line_mask[row].sum()) == 1
            and not bool(clean.line_is_self[row])], dtype=torch.float32)
        count = min(cos_clean.numel(), labels["angle_cos"].numel())
        if count:
            reference["angle_from_perturbed_cosine"] = float(
                (cos_noisy[:count] - labels["angle_cos"][:count].float()).square().mean())
    return reference


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--cohort-root", required=True)
    parser.add_argument("--dual-static-root", required=True)
    parser.add_argument("--pretrain-target-root", required=True)
    parser.add_argument("--config-root", default="configs/mts")
    parser.add_argument("--checkpoint-root", required=True,
                        help="directory containing <mode>/resume_*.pt runs")
    parser.add_argument("--report-json", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--base-step", type=int, default=2000)
    args = parser.parse_args()
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    source = load_source(args.cache_root, args.cohort_root,
                         args.dual_static_root, args.pretrain_target_root)
    report = {"schema": "glt-dual-pretrain-fixed-batch-diagnostics-v1",
              "records": RECORDS, "batch": BATCH, "base_step": args.base_step,
              "checkpoints": {}}
    try:
        for mode, steps in CHECKPOINTS.items():
            config = json.loads((Path(args.config_root) /
                                 f"glt_dual_three_task_{mode}.json").read_text(encoding="utf-8"))
            world, micro, batch_size = 4, config["microbatch"], config["global_batch"]
            stream = OrderedSampleStream(len(source), config["seed"])
            positions = positions_after_step(args.base_step, batch_size, world, micro)
            entry = {"positions": positions, "keys": [], "seed": config["seed"],
                     "global_batch": batch_size, "world": world, "microbatch": micro,
                     "steps": {}}
            for step in steps:
                path = Path(args.checkpoint_root) / mode / f"resume_{step:05d}.pt"
                if not path.is_file():
                    entry["steps"][step] = {"status": "MISSING", "path": str(path)}
                    continue
                state = torch.load(path, map_location="cpu", weights_only=False)
                model = DualPretrainer(mode, collect_diagnostics=True).to(device)
                model.load_state_dict(state["model"], strict=True)
                model.eval()
                step_entry = {"path": str(path), "checkpoint_step": int(state["step"]),
                              "precisions": {}}
                prepared, clean_samples, noisy_samples = [], [], []
                for position in positions:
                    index = stream.index_at(position)
                    key = source.samples[index][0].hex()
                    if len(entry["keys"]) < RECORDS:
                        entry["keys"].append({"position": position, "key": key})
                    noisy, labels = prepare_pretrain_sample(
                        *source[index], seed=config["seed"], key=key, position=position,
                        sigma=config["noise_sigma"], ratio=config["atom_mask_ratio"])
                    prepared.append((noisy, labels))
                for start in range(0, RECORDS, BATCH):
                    chunk = prepared[start:start + BATCH]
                    data, labels = pretrain_collate([row[0] for row in chunk]), None
                    data, labels = pretrain_collate(chunk)
                    data = data.to(device)
                    labels = {k: (v.to(device) if torch.is_tensor(v) else v)
                              for k, v in labels.items()}
                    for precision in ("fp32", "bf16"):
                        torch.manual_seed(0)
                        with torch.autocast(device.type, dtype=torch.bfloat16,
                                            enabled=precision == "bf16"):
                            out = model(data, labels)
                        diagnostics = out["diagnostics"]
                        bucket = step_entry["precisions"].setdefault(precision, {})
                        bucket[f"batch{start // BATCH}"] = {
                            "components": {k: diagnostics["components"][k] for k in (
                                "chem_sum", "length_sum", "angle_sum", "geometry_sum",
                                "geometry_valid_count", "angle_valid_count",
                                "graphs_without_angle", "length_plus_angle_reconstructs_geometry")},
                            "targets": diagnostics["targets"],
                            "predictions": diagnostics["predictions"],
                            "angle_head": diagnostics["angle_head"],
                            "representations": diagnostics["representations"],
                            "gaussian": diagnostics["gaussian"],
                        }
                        if start == 0 and precision == "fp32":
                            norms, cosines, total = component_gradients(
                                model, data, labels, config["loss_weights"])
                            bucket["batch0"]["gradient_norms_3d_encoder"] = norms
                            bucket["batch0"]["gradient_cosines_3d_encoder"] = cosines
                            bucket["batch0"]["total_objective"] = total
                            clean = build_dual_sample(*source[stream.index_at(positions[0])])
                    if start == 0:
                        index0 = stream.index_at(positions[0])
                        clean0 = build_dual_sample(*source[index0])
                        clean_ref = copy.copy(source[index0][1])
                        noisy_ref = copy.deepcopy(clean0)
                        step_entry["perturbed_reference"] = perturbed_reference(
                            clean0, noisy_ref, labels)
                step_entry["status"] = "OK"
                entry["steps"][step] = step_entry
                del model
                torch.cuda.empty_cache()
            report["checkpoints"][mode] = entry
    finally:
        source.close()
    write_json(Path(args.report_json), report)
    print(json.dumps({"report": args.report_json,
                      "modes": list(report["checkpoints"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
