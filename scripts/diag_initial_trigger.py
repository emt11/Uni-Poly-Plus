#!/usr/bin/env python3
"""Offline initial-trigger diagnostics for the Concat geometry instability.

C1 component swap, C2 parameter deltas, C3 optimizer state, C4 layer scale,
C5 length-head trace, C6 angle-head trace, C7 spike-batch attribution and
C8 healthy-checkpoint bad-batch test.  Read-only: eval-mode forwards only, no
optimizer step, no training, no cache write.
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

STEPS = (2000, 2600, 2660, 2700, 2800)
SWAP_PREFIX = ("2600", "2660", "2700", "2800")
PANEL, CHUNK = 16, 8
HEADS = ("atom_head", "length_head", "angle_head", "fp_head")


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def stats(values):
    flat = values.detach().float().reshape(-1)
    if flat.numel() == 0:
        return {"count": 0}
    q = torch.quantile(flat, torch.tensor([0.5, 0.9, 0.95, 0.99], device=flat.device))
    return {"count": int(flat.numel()), "mean": float(flat.mean()),
            "std": float(flat.std(unbiased=False)), "min": float(flat.min()),
            "max": float(flat.max()), "p50": float(q[0]), "p90": float(q[1]),
            "p95": float(q[2]), "p99": float(q[3]),
            "abs_p90": float(flat.abs().quantile(0.9)),
            "abs_p99": float(flat.abs().quantile(0.99)),
            "abs_max": float(flat.abs().max())}


def l2_stats(tensor):
    rows = tensor.detach().float().reshape(-1, tensor.shape[-1]) if tensor.dim() > 1 else tensor.detach().float()
    norms = rows.norm(dim=-1) if rows.dim() > 1 else rows.abs()
    return {"rms": float(tensor.detach().float().pow(2).mean().sqrt()), **stats(norms)}


def load_state(path):
    state = torch.load(path, map_location="cpu", weights_only=False)
    return state


def build_model(mode, device, state_dict=None, diagnostics=True):
    model = DualPretrainer(mode, collect_diagnostics=diagnostics).to(device)
    if state_dict is not None:
        model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def merged_state(base_sd, sd_by_step, encoder_step, length_step, angle_step):
    """E from encoder_step, L from length_step, A from angle_step, other heads from 2600."""

    merged = dict(base_sd)
    for name in list(merged):
        if name.startswith("encoder."):
            merged[name] = sd_by_step[encoder_step][name]
        elif name.startswith("length_head."):
            merged[name] = sd_by_step[length_step][name]
        elif name.startswith("angle_head."):
            merged[name] = sd_by_step[angle_step][name]
        elif name.startswith(("atom_head.", "fp_head.")):
            merged[name] = sd_by_step["2600"][name]
    return merged


def capture_forward(model, data, labels, targets):
    """Forward with read-only hooks; returns wrapper output and captured tensors."""

    captured, handles = {}, []

    def hook(name):
        def fn(module, inputs, output):
            value = output[0] if isinstance(output, tuple) else output
            if torch.is_tensor(value):
                captured[name] = value.detach()
        return fn

    for name, module in targets:
        handles.append(module.register_forward_hook(hook(name)))
    try:
        with torch.autocast("cuda", enabled=False):
            out = model(data, labels)
    finally:
        for handle in handles:
            handle.remove()
    return out, captured


def panel_metrics(model, panels, device):
    """Aggregate panel metrics for one model state (eval, fp32)."""

    length_err, angle_err, pre_tanh_all, length_pred = [], [], [], []
    graph_len, graph_ang, graph_counts = [], [], []
    repr_rms, sat = [], []
    for chunk in panels:
        data, labels = collate_chunk(chunk, device)
        out, captured = capture_forward(model, data, labels, (
            ("length_out", model.length_head[2]),
            ("angle_pre", model.angle_head[2]),
            ("length_h0", model.length_head[0]),
            ("glt_triplet", model.encoder.glt.triplet),
            ("glt_layer0", model.encoder.glt.layers[0]),
            ("glt_layer5", model.encoder.glt.layers[5]),
        ))
        diag = out["diagnostics"]
        center = data.bond_center.bool()
        pred = captured["length_out"].float().flatten()
        if int(center.sum()):
            err = (pred - labels["distance"].float()).square()
            graph_len.append(per_graph(err, data.bond_batch[center], len(chunk)))
            length_err.append(err)
            length_pred.append(pred)
            graph_counts.append(int(center.sum()))
        pre = captured["angle_pre"].float().flatten()
        if pre.numel():
            tanh = torch.tanh(pre)
            err = (tanh - labels["angle_cos"].float()).square()
            graph_ang.append(per_graph(err, labels["angle_graph"], len(chunk)))
            angle_err.append(err)
            pre_tanh_all.append(pre)
            sat.append(float((tanh.abs() > 0.9999).float().mean()))
        repr_rms.append((diag["representations"]["center_bond_states_rms"],
                         diag["representations"]["graph_3d_rms"]))
        layer_rms = {"triplet": float(captured["glt_triplet"].pow(2).mean().sqrt()),
                     "layer0": float(captured["glt_layer0"].pow(2).mean().sqrt()),
                     "layer5": float(captured["glt_layer5"].pow(2).mean().sqrt())}
    def graph_mean(pairs):
        if not pairs:
            return None
        means, valid = [], []
        for m, v in pairs:
            means.append(m); valid.append(v)
        means = torch.cat(means); valid = torch.cat(valid)
        return float(means[valid].mean()) if bool(valid.any()) else None
    length_err = torch.cat(length_err) if length_err else torch.empty(0)
    angle_err = torch.cat(angle_err) if angle_err else torch.empty(0)
    pre = torch.cat(pre_tanh_all) if pre_tanh_all else torch.empty(0)
    tanh_all = torch.tanh(pre) if pre.numel() else torch.empty(0)
    deriv = (1.0 - tanh_all.square()) if pre.numel() else torch.empty(0)
    return {
        "length_loss": graph_mean(graph_len),
        "angle_loss": graph_mean(graph_ang),
        "total_geometry_loss": (graph_mean(graph_len) or 0.0) + (graph_mean(graph_ang) or 0.0),
        "length_prediction": stats(torch.cat(length_pred)) if length_pred else {"count": 0},
        "length_abs_error": {k: v for k, v in stats(length_err).items() if k.startswith("abs") or k in ("p90", "p99", "max", "count")},
        "angle_pre_tanh": stats(pre),
        "angle_exact_pm1_fraction": float((tanh_all.abs() == 1.0).float().mean()) if pre.numel() else None,
        "angle_abs_gt_0_95": float((tanh_all.abs() > 0.95).float().mean()) if pre.numel() else None,
        "angle_abs_gt_0_99": float((tanh_all.abs() > 0.99).float().mean()) if pre.numel() else None,
        "tanh_derivative": ({"mean": float(deriv.mean()), "p50": float(deriv.quantile(0.5)),
                             "p10": float(deriv.quantile(0.1)),
                             "fraction_lt_1e-3": float((deriv < 1e-3).float().mean()),
                             "fraction_lt_1e-6": float((deriv < 1e-6).float().mean())}
                            if deriv.numel() else {"count": 0}),
        "center_bond_states_rms": repr_rms[0][0] if repr_rms else None,
        "graph_3d_rms": repr_rms[0][1] if repr_rms else None,
        "layer_rms_sample": layer_rms,
        "saturation_fraction_mean": (sum(sat) / len(sat)) if sat else None,
    }


def collate_chunk(chunk, device):
    data, labels = pretrain_collate([(row["noisy"], row["labels"]) for row in chunk])
    data = data.to(device)
    labels = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in labels.items()}
    return data, labels


# --------------------------------------------------------------------------- #
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--cohort-root", required=True)
    parser.add_argument("--dual-static-root", required=True)
    parser.add_argument("--pretrain-target-root", required=True)
    parser.add_argument("--checkpoint-dir", required=True,
                        help="B.3 replay dir holding resume_02600..02800.pt")
    parser.add_argument("--pretrain-2000", required=True)
    parser.add_argument("--config", default="configs/mts/glt_dual_three_task_concat.json")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    ckpt_path = {"2000": Path(args.pretrain_2000)}
    for step in SWAP_PREFIX:
        ckpt_path[step] = Path(args.checkpoint_dir) / f"resume_0{step}.pt"
    states = {step: load_state(path) for step, path in ckpt_path.items()}
    source, _ = open_source(args.cohort_root, args.cache_root,
                            dual_static_root=args.dual_static_root,
                            pretrain_target_root=args.pretrain_target_root)
    verdicts = {}
    try:
        identity = states["2000"]["identity"]
        for step, state in states.items():
            checks = {
                "step": int(state["step"]) == int(step),
                "scheduler_step": int(state["scheduler"]["step"]) == int(step),
                "next_position": int(state["next_position"]) == int(step) * int(config["global_batch"]),
                "fusion_concat": state["identity"]["config"]["fusion_mode"] == "concat",
                "identity_match": state["identity"] == identity,
            }
            verdicts[step] = checks
            if not all(checks.values()):
                write_json(out / "identity_stop.json", {"step": step, "checks": checks})
                raise SystemExit(f"identity mismatch at {step}: {checks}")
        sd = {step: state["model"] for step, state in states.items()}

        # fixed CPU panel (same construction as B.2)
        stream = OrderedSampleStream(len(source), config["seed"])
        base = 2000 * config["global_batch"]
        panel = []
        for local in range(PANEL):
            position = base + local
            index = stream.index_at(position)
            key = source.samples[index][0].hex()
            static, target = source.static_for(index), source.target_for(index)
            noisy, labels = prepare_pretrain_sample(
                *source[index], seed=config["seed"], key=key, position=position,
                sigma=config["noise_sigma"], ratio=config["atom_mask_ratio"],
                static=static, target=target)
            panel.append({"position": position, "key": key, "noisy": noisy, "labels": labels})
        panels = [panel[i:i + CHUNK] for i in range(0, PANEL, CHUNK)]

        # ---- C1 component swap -------------------------------------------- #
        swap = {}
        combos = [("HEALTHY_2600", "2600", "2600", "2600"),
                  ("ONSET_2660", "2660", "2660", "2660"),
                  ("CROSS_A_E2600_L2660", "2600", "2660", "2600"),
                  ("CROSS_B_E2660_L2600", "2660", "2600", "2600"),
                  ("CROSS_C_E2600_L2600_A2660", "2600", "2600", "2660"),
                  ("CROSS_D_E2660_L2600_A2660", "2660", "2600", "2660"),
                  ("LATE_E2700_L2600_A2600", "2700", "2600", "2600"),
                  ("LATE_E2800_L2600_A2600", "2800", "2600", "2600"),
                  ("E2600_L2700_A2600", "2600", "2700", "2600"),
                  ("E2600_L2800_A2600", "2600", "2800", "2600"),
                  ("E2600_L2600_A2700", "2600", "2600", "2700"),
                  ("E2600_L2600_A2800", "2600", "2600", "2800"),
                  ("ALL_2800", "2800", "2800", "2800")]
        for name, e, l, a in combos:
            model = build_model("concat", device, merged_state(sd["2600"], sd, e, l, a))
            swap[name] = {"encoder": e, "length_head": l, "angle_head": a,
                          **panel_metrics(model, panels, device)}
            del model
            torch.cuda.empty_cache()
        for name, value in swap.items():
            swap[name]["verdict"] = {
                "length_bad": (value["length_loss"] or 0) > 10 * (swap["HEALTHY_2600"]["length_loss"] or 1e-9),
                "angle_bad": (value["angle_loss"] or 0) > 10 * (swap["HEALTHY_2600"]["angle_loss"] or 1e-9),
                "saturated": (value["angle_exact_pm1_fraction"] or 0) > 0.05,
            }
        write_json(out / "component_swap.json", {"identity_checks": verdicts, "swap": swap})

        # ---- C2 parameter deltas ------------------------------------------ #
        def group_of(name):
            if name.startswith("encoder.glt.layers."):
                return "glt_layer" + name.split(".")[3]
            for prefix, label in (("encoder.o8.", "o8_encoder"), ("encoder.glt.", "glt_3d_encoder"),
                                  ("encoder.norm", "fusion"), ("encoder.kfuse", "fusion"),
                                  ("length_head.", "length_head"), ("angle_head.", "angle_head"),
                                  ("fp_head.", "fp_head"), ("atom_head.", "atom_head")):
                if name.startswith(prefix):
                    return label
            return "other"
        deltas = {}
        for lo, hi in (("2600", "2660"), ("2660", "2700"), ("2700", "2800")):
            per_group = {}
            for name in sd["2600"]:
                group = group_of(name)
                before, after = sd[lo][name].float(), sd[hi][name].float()
                row = per_group.setdefault(group, {"delta_l2_sq": 0.0, "before_sq": 0.0,
                                                   "delta_max_abs": 0.0, "tensors": 0})
                diff = after - before
                row["delta_l2_sq"] += float(diff.pow(2).sum())
                row["before_sq"] += float(before.pow(2).sum())
                row["delta_max_abs"] = max(row["delta_max_abs"], float(diff.abs().max()))
                row["tensors"] += 1
            per_group = {g: {"param_norm_before": r["before_sq"] ** 0.5,
                             "delta_l2": r["delta_l2_sq"] ** 0.5,
                             "relative_delta": (r["delta_l2_sq"] ** 0.5) / ((r["before_sq"] ** 0.5) + 1e-12),
                             "delta_max_abs": r["delta_max_abs"], "tensors": r["tensors"]}
                         for g, r in per_group.items()}
            head_detail = {}
            for head in ("length_head", "angle_head"):
                for name in sd["2600"]:
                    if name.startswith(head + "."):
                        before, after = sd[lo][name].float(), sd[hi][name].float()
                        diff = after - before
                        head_detail[name] = {
                            "shape": list(before.shape),
                            "weight_l2" if "weight" in name else "bias_l2": float(before.norm()),
                            "rms_before": float(before.pow(2).mean().sqrt()),
                            "rms_after": float(after.pow(2).mean().sqrt()),
                            "maxabs_before": float(before.abs().max()),
                            "maxabs_after": float(after.abs().max()),
                            "delta_l2": float(diff.norm()),
                            "relative_delta": float(diff.norm()) / (float(before.norm()) + 1e-12),
                        }
            deltas[f"{lo}->{hi}"] = {"groups": per_group, "geometry_head_tensor_detail": head_detail}
        write_json(out / "parameter_delta.json", deltas)

        # ---- C3 optimizer state ------------------------------------------- #
        names = [name for name, _ in build_model("concat", torch.device("cpu")).named_parameters()]
        opt = {}
        for step, state in states.items():
            ostate = state["optimizer"]
            groups = ostate["param_groups"]
            entry = {"param_group_lrs": [g["lr"] for g in groups],
                     "betas": [g.get("betas") for g in groups],
                     "eps": [g.get("eps") for g in groups],
                     "weight_decay": [g.get("weight_decay") for g in groups],
                     "modules": {}}
            for index, pstate in ostate["state"].items():
                name = names[int(index)]
                group = group_of(name)
                exp_avg = pstate.get("exp_avg")
                exp_avg_sq = pstate.get("exp_avg_sq")
                if exp_avg is None:
                    continue
                proxy = (exp_avg.abs() / (exp_avg_sq.sqrt() + 1e-8))
                row = entry["modules"].setdefault(group, {"exp_avg_rms_sq": 0.0, "exp_avg_sq_rms_sq": 0.0,
                                                          "exp_avg_abs_max": 0.0, "exp_avg_sq_abs_max": 0.0,
                                                          "proxy_sum": 0.0, "proxy_count": 0,
                                                          "proxy_max": 0.0, "tensors": 0})
                row["exp_avg_rms_sq"] += float(exp_avg.pow(2).sum())
                row["exp_avg_sq_rms_sq"] += float(exp_avg_sq.pow(2).sum())
                row["exp_avg_abs_max"] = max(row["exp_avg_abs_max"], float(exp_avg.abs().max()))
                row["exp_avg_sq_abs_max"] = max(row["exp_avg_sq_abs_max"], float(exp_avg_sq.abs().max()))
                row["proxy_sum"] += float(proxy.sum()); row["proxy_count"] += proxy.numel()
                row["proxy_max"] = max(row["proxy_max"], float(proxy.max()))
                row["tensors"] += 1
            entry["modules"] = {g: {"exp_avg_rms": r["exp_avg_rms_sq"] ** 0.5,
                                    "exp_avg_sq_rms": r["exp_avg_sq_rms_sq"] ** 0.5,
                                    "exp_avg_abs_max": r["exp_avg_abs_max"],
                                    "exp_avg_sq_abs_max": r["exp_avg_sq_abs_max"],
                                    "update_proxy_mean": r["proxy_sum"] / max(1, r["proxy_count"]),
                                    "update_proxy_max": r["proxy_max"], "tensors": r["tensors"]}
                                for g, r in entry["modules"].items()}
            opt[step] = entry
        write_json(out / "optimizer_state_delta.json", opt)

        # ---- C4/C5/C6 layer + head traces --------------------------------- #
        layer_scale, length_trace, angle_trace = {}, {}, {}
        for step in SWAP_PREFIX:
            model = build_model("concat", device, sd[step])
            entry, lentry, aentry = {}, {}, {}
            for chunk in panels:
                data, labels = collate_chunk(chunk, device)
                targets = [("triplet", model.encoder.glt.triplet),
                           ("distance_projection", model.encoder.glt.distance_projection)]
                targets += [(f"layer{index}", layer) for index, layer in enumerate(model.encoder.glt.layers)]
                targets += [("length_in", model.length_head[0]), ("length_hidden", model.length_head[1]),
                            ("length_out", model.length_head[2]),
                            ("angle_in", model.angle_head[0]), ("angle_hidden", model.angle_head[1]),
                            ("angle_pre", model.angle_head[2])]
                fout, captured = capture_forward(model, data, labels, targets)
                for name in ("triplet", "distance_projection"):
                    entry.setdefault(name, []).append(l2_stats(captured[name]))
                for index in range(6):
                    entry.setdefault(f"layer{index}", []).append(l2_stats(captured[f"layer{index}"]))
                for name, bucket in (("length_in", lentry), ("length_hidden", lentry), ("length_out", lentry)):
                    bucket.setdefault(name, []).append(l2_stats(captured[name]))
                for name in ("angle_in", "angle_hidden", "angle_pre"):
                    aentry.setdefault(name, []).append(l2_stats(captured[name]))
                graph = fout["diagnostics"]["representations"]
                entry.setdefault("graph_3d", []).append({"rms": graph["graph_3d_rms"]})
            def first(rows):
                return {name: values[0] for name, values in rows.items()}
            layer_scale[step] = first(entry)
            length_trace[step] = first(lentry)
            angle_trace[step] = first(aentry)
            del model
            torch.cuda.empty_cache()
        write_json(out / "layer_scale.json", layer_scale)
        write_json(out / "length_head_trace.json", length_trace)
        write_json(out / "angle_head_trace.json", angle_trace)

        print(json.dumps({"status": "C1-C6 done", "out": str(out)}, indent=2))
    finally:
        source.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
