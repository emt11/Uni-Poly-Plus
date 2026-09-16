#!/usr/bin/env python3
"""Phase 7 pretrain health review for the fixed (geometry-head-norm) Concat 5k.

Read-only: parses the existing per-step rank logs and the diagnostics JSONL of
the fixed run, plus the two earlier formal 5k runs (broken Concat, KFuse) and
the bounded broken-Concat replay.  Nothing is retrained, replayed or rewritten.

The verdict ``FORMAL_FIXED_5K_HEALTHY`` is computed from explicit blocking
checks whose thresholds are recorded in the report.  Thresholds are derived
from measured reference magnitudes, not tuned to the observed result:

* per-graph length loss: healthy ~2.5e-4 .. 3e-4, broken peak 64.78 (step 2676
  of the bounded replay) -> blocking threshold 5e-2
* raw 3D representation RMS: pre-failure healthy 1.10 .. 1.22, broken band
  5.77 .. 9.17 -> blocking threshold 3.0
* angle tanh saturation: healthy 0.0, broken 1.0 (fully saturated) -> 1e-3
* minimum tanh derivative: healthy >4e-3, broken 0.0 -> 1e-3
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.training.glt_dual_runtime import write_json


# Blocking thresholds.  See module docstring for the measured references.
LENGTH_PER_GRAPH_MAX = 5e-2
GRAPH3D_RMS_MAX = 3.0
SATURATION_FRACTION_MAX = 1e-3
TANH_DERIVATIVE_MIN = 1e-3
ANGLE_GRAD_LAST500_MIN = 1e-5
RANK_SPREAD_RTOL = 1e-6
INITIAL_TRANSIENT_STEPS = 98
WINDOWS = (100, 500)
SNAPSHOTS = (1000, 2000, 3000, 4000, 5000)


def _load_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _load_step_log(path):
    """Per-step losses from a rank-interleaved stdout log.

    The log concatenates one JSON object per rank with no guaranteed newline,
    so records are located with a raw decoder instead of line splitting.
    """
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    decoder = json.JSONDecoder()
    by_step = {}
    cursor = text.find('{"step"')
    while cursor != -1:
        try:
            record, _ = decoder.raw_decode(text, cursor)
        except ValueError:
            cursor = text.find('{"step"', cursor + 1)
            continue
        step = int(record["step"])
        by_step.setdefault(step, []).append(record)
        cursor = text.find('{"step"', cursor + 7)
    return by_step


def _loss_matrix(by_step):
    """step -> per-loss mean/min/max over the ranks that printed that step."""
    rows = {}
    for step, records in by_step.items():
        per_rank = [record["losses"] for record in records]
        width = len(per_rank[0])
        mean = [statistics.fmean(row[i] for row in per_rank) for i in range(width)]
        spread = [
            (max(row[i] for row in per_rank) - min(row[i] for row in per_rank))
            / max(abs(mean[i]), 1e-12)
            for i in range(width)
        ]
        rows[step] = {"mean": mean, "relative_rank_spread": spread,
                      "ranks": len(per_rank)}
    return rows


def _window(matrix, steps, size):
    values = [matrix[step]["mean"] for step in steps[-size:]]
    return {
        "window": size,
        "step_range": [steps[max(0, len(steps) - size)], steps[-1]],
        "loss_mean": [statistics.fmean(v[i] for v in values) for i in range(len(values[0]))],
        "loss_max": [max(v[i] for v in values) for i in range(len(values[0]))],
        "loss_min": [min(v[i] for v in values) for i in range(len(values[0]))],
    }


def _per_graph(record, key):
    count = int(record["components"]["geometry_valid_count"])
    return float(record["components"][key]) / count if count else None


def _dense(rows, key):
    """(step, value) pairs for a nested diagnostic field, e.g. ``a.b``."""
    parts = key.split(".")

    def resolve(row):
        value = row
        for part in parts:
            if not isinstance(value, dict) or part not in value:
                return None
            value = value[part]
        return value

    pairs = []
    for row in rows:
        value = resolve(row)
        if value is not None:
            pairs.append((int(row["step"]), value))
    return pairs


def _describe(fixed_rows, key):
    """Steady-state and overall view of one dense diagnostic series."""
    values = _dense(fixed_rows, key)
    steady = [item for item in values if item[0] > INITIAL_TRANSIENT_STEPS]
    numbers = [value for _, value in values]
    steady_numbers = [value for _, value in steady]
    return {
        "sample_count": len(values),
        "max": max(values, key=lambda item: item[1]) if values else None,
        "final": values[-1] if values else None,
        "median": statistics.median(numbers) if numbers else None,
        "steady_max": max(steady, key=lambda item: item[1]) if steady else None,
        "steady_median": statistics.median(steady_numbers) if steady_numbers else None,
    }


def _snapshot(matrix, step):
    entry = matrix.get(step)
    if entry is None:
        return None
    return {
        "step": step,
        "chem": entry["mean"][0], "geometry": entry["mean"][1], "fingerprint": entry["mean"][2],
        "max_relative_rank_spread": max(entry["relative_rank_spread"]),
        "ranks": entry["ranks"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixed-run", required=True)
    parser.add_argument("--fixed-log", required=True)
    parser.add_argument("--old-concat-run", required=True)
    parser.add_argument("--old-concat-log", required=True)
    parser.add_argument("--kfuse-run", required=True)
    parser.add_argument("--kfuse-log", required=True)
    parser.add_argument("--broken-replay", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    run_logs = {
        name: _loss_matrix(_load_step_log(path)) for name, path in (
            ("fixed_concat_geonorm", args.fixed_log),
            ("broken_concat", args.old_concat_log),
            ("kfuse", args.kfuse_log),
        )
    }
    steps = {name: sorted(matrix) for name, matrix in run_logs.items()}
    for name, ordered in steps.items():
        if not ordered:
            raise SystemExit(f"no training records found for {name}")
    if steps["fixed_concat_geonorm"][-1] != 5000:
        raise SystemExit("fixed run did not reach step 5000")

    fixed_rows = _load_jsonl(Path(args.fixed_run) / "diagnostics_steps.jsonl")
    diagnosed = {int(row["step"]): row for row in fixed_rows}
    if set(diagnosed) != set(steps["fixed_concat_geonorm"]):
        raise SystemExit("fixed diagnostics and log step sets differ")

    # Every objective value the run logged, plus all grad norms, must be finite.
    nonfinite = sorted(
        step for step, entry in run_logs["fixed_concat_geonorm"].items()
        if not all(math.isfinite(value) for value in entry["mean"])
    )
    nonfinite_grad = sorted(
        int(row["step"]) for row in fixed_rows
        if not math.isfinite(float(row["grad_total_preclip"]))
    )

    length_series = [(step, value) for step, value in
                     ((int(row["step"]), _per_graph(row, "length_sum")) for row in fixed_rows)
                     if value is not None]
    angle_series = [(step, value) for step, value in
                    ((int(row["step"]), _per_graph(row, "angle_sum")) for row in fixed_rows)
                    if value is not None]
    length_steady = [item for item in length_series if item[0] > INITIAL_TRANSIENT_STEPS]
    saturation_values = [
        (int(row["step"]), float(row["angle_head"]["exact_plus_minus_one_fraction"]))
        for row in fixed_rows if "angle_head" in row
    ]
    pre_tanh_abs = [
        (int(row["step"]),
         max(abs(float(row["angle_head"]["pre_tanh"]["max"])),
             abs(float(row["angle_head"]["pre_tanh"]["min"]))))
        for row in fixed_rows if "angle_head" in row
    ]
    tanh_derivative = [
        (int(row["step"]), float(row["angle_head"]["tanh_derivative"]["min"]))
        for row in fixed_rows if "angle_head" in row
    ]
    graph3d = _describe(fixed_rows, "representations.graph_3d_rms")
    bond_rms = _describe(fixed_rows, "representations.bond_states_rms")
    grad_total = [(int(row["step"]), float(row["grad_total_preclip"])) for row in fixed_rows]
    grad_angle = [(int(row["step"]), float(row["grad_norms"]["angle_head"]))
                  for row in fixed_rows if "grad_norms" in row]
    grad_length = [(int(row["step"]), float(row["grad_norms"]["length_head"]))
                   for row in fixed_rows if "grad_norms" in row]
    replay_rows = _load_jsonl(Path(args.broken_replay) / "diagnostics_steps.jsonl")
    replay_grad = [(int(row["step"]), float(row["grad_total_preclip"])) for row in replay_rows]

    def tail(series, size):
        return [value for _, value in series[-size:]]

    grad_total_steady = [item for item in grad_total if item[0] > INITIAL_TRANSIENT_STEPS]
    graph3d_steady = [item for item in _dense(fixed_rows, "representations.graph_3d_rms")
                      if item[0] > INITIAL_TRANSIENT_STEPS]
    rank_spread = [
        max(run_logs[name][step]["relative_rank_spread"])
        for name in run_logs for step in steps[name]
    ]

    checks = {
        "S1_no_length_explosion": {
            "rule": f"steady-state (step > {INITIAL_TRANSIENT_STEPS}) per-graph length loss <= {LENGTH_PER_GRAPH_MAX}",
            "observed_steady_max": max(length_steady, key=lambda item: item[1]) if length_steady else None,
            "observed_transient_max": max(length_series, key=lambda item: item[1]) if length_series else None,
        },
        "S2_angle_head_not_saturated": {
            "rule": (f"max exact +-1 tanh fraction <= {SATURATION_FRACTION_MAX}, "
                     f"min tanh derivative >= {TANH_DERIVATIVE_MIN}, "
                     f"min angle_head grad over last 500 >= {ANGLE_GRAD_LAST500_MIN}"),
            "observed_max_saturation": max(saturation_values, key=lambda item: item[1]),
            "observed_min_tanh_derivative": min(tanh_derivative, key=lambda item: item[1]),
            "observed_min_angle_grad_last500": min(tail(grad_angle, 500)),
            "observed_min_length_grad_last500": min(tail(grad_length, 500)),
        },
        "S3_representation_bounded": {
            "rule": (f"steady-state max raw graph_3d RMS <= {GRAPH3D_RMS_MAX} and final <= steady-state max"),
            "observed_steady_max_graph_3d_rms": max(graph3d_steady, key=lambda item: item[1]),
            "observed_overall_max_graph_3d_rms": graph3d["max"],
            "observed_final_graph_3d_rms": graph3d["final"],
        },
        "S4_all_values_finite": {
            "rule": "no nonfinite logged objective or pre-clip gradient norm; run reached step 5000",
            "nonfinite_objective_steps": nonfinite,
            "nonfinite_gradient_steps": nonfinite_grad,
            "completed_steps": steps["fixed_concat_geonorm"][-1],
        },
        "S5_ddp_rank_consistency": {
            "rule": f"per-step relative spread across ranks <= {RANK_SPREAD_RTOL}",
            "observed_max_relative_rank_spread": max(rank_spread),
        },
        "S6_objectives_converged": {
            "rule": "last-100 chem/geometry/fingerprint all below their step-1000 values",
            "step_1000": _snapshot(run_logs["fixed_concat_geonorm"], 1000),
            "last_100": _window(run_logs["fixed_concat_geonorm"],
                                steps["fixed_concat_geonorm"], 100),
        },
        "S7_gradient_scale_settled": {
            "rule": ("max pre-clip gradient norm over the last 500 steps < 1.0 (the clip threshold), "
                     "i.e. late training no longer needs clipping; earlier clipping is recorded as a "
                     "count rather than treated as a failure"),
            "observed_steady_max_preclip": max(grad_total_steady, key=lambda item: item[1]),
            "observed_overall_max_preclip": max(grad_total, key=lambda item: item[1]),
            "observed_max_last500": max(grad_total[-500:], key=lambda item: item[1]),
            "steps_at_or_above_clip": sum(1 for _, value in grad_total if value >= 1.0),
            "total_steps": len(grad_total),
            "last_step_at_or_above_clip": max(
                (step for step, value in grad_total if value >= 1.0), default=None),
            "clip_steps_per_500": {
                f"{low}-{low + 499}": sum(1 for step, value in grad_total
                                          if low <= step <= low + 499 and value >= 1.0)
                for low in range(1, 5001, 500)
            },
            "broken_route_contrast": {
                "overall_max": max(replay_grad, key=lambda item: item[1]),
                "max_last500": max(replay_grad[-500:], key=lambda item: item[1]),
                "fraction_at_or_above_clip": sum(1 for _, value in replay_grad if value >= 1.0)
                                             / len(replay_grad),
            },
        },
    }
    verdict = {
        "S1_no_length_explosion": checks["S1_no_length_explosion"]["observed_steady_max"][1] <= LENGTH_PER_GRAPH_MAX,
        "S2_angle_head_not_saturated": (
            max(saturation_values, key=lambda item: item[1])[1] <= SATURATION_FRACTION_MAX
            and min(tanh_derivative, key=lambda item: item[1])[1] >= TANH_DERIVATIVE_MIN
            and min(tail(grad_angle, 500)) >= ANGLE_GRAD_LAST500_MIN),
        "S3_representation_bounded": (
            max(graph3d_steady, key=lambda item: item[1])[1] <= GRAPH3D_RMS_MAX
            and graph3d["final"][1] <= max(graph3d_steady, key=lambda item: item[1])[1]),
        "S4_all_values_finite": not nonfinite and not nonfinite_grad
                                and steps["fixed_concat_geonorm"][-1] == 5000,
        "S5_ddp_rank_consistency": max(rank_spread) <= RANK_SPREAD_RTOL,
        "S6_objectives_converged": all(
            checks["S6_objectives_converged"]["last_100"]["loss_mean"][i]
            < checks["S6_objectives_converged"]["step_1000"][name]
            for i, name in enumerate(("chem", "geometry", "fingerprint"))),
        "S7_gradient_scale_settled": max(grad_total[-500:], key=lambda item: item[1])[1] < 1.0,
    }
    blocking = sorted(verdict)
    healthy = all(verdict.values())

    replay_length = [(_per_graph(row, "length_sum"), int(row["step"])) for row in replay_rows]
    replay_graph3d = _dense(replay_rows, "representations.graph_3d_rms")
    replay_saturation = [(int(row["step"]),
                          float(row["angle_head"]["exact_plus_minus_one_fraction"]))
                         for row in replay_rows if "angle_head" in row]
    replay_tanh = [(int(row["step"]), float(row["angle_head"]["tanh_derivative"]["min"]))
                   for row in replay_rows if "angle_head" in row]

    identities = {}
    for name, root in (("fixed_concat_geonorm", args.fixed_run),
                       ("broken_concat", args.old_concat_run),
                       ("kfuse", args.kfuse_run)):
        run = json.loads((Path(root) / "run.json").read_text(encoding="utf-8"))
        config = run["identity"]["config"]
        identities[name] = {
            "world_size": run["identity"].get("world_size"),
            "accumulation": run.get("accumulation"),
            "microbatch": config.get("microbatch"),
            "global_batch": config.get("global_batch"),
            "max_optimizer_steps": config.get("max_optimizer_steps"),
            "lr": config.get("lr"), "warmup_steps": config.get("warmup_steps"),
            "schedule_total_steps": config.get("schedule_total_steps"),
            "amp_dtype": config.get("amp_dtype"),
            "loss_weights": config.get("loss_weights"),
            "noise_sigma": config.get("noise_sigma"),
            "atom_mask_ratio": config.get("atom_mask_ratio"),
            "geometry_head_norm": config.get("geometry_head_norm"),
            "cohort_hash": run["identity"].get("cohort_hash"),
            "main_bundle_hash": run["identity"].get("main_bundle_hash"),
            "dual_static_manifest_hash": run["identity"].get("dual_static_manifest_hash"),
            "run_json": str(Path(root) / "run.json"),
        }

    report = {
        "scope": "Phase 7 static health review of existing logs and diagnostics; no retraining",
        "steps_available": {name: [ordered[0], ordered[-1], len(ordered)]
                            for name, ordered in steps.items()},
        "identities": identities,
        "controlled_change": {
            "matched": [key for key in ("cohort_hash", "main_bundle_hash",
                                        "dual_static_manifest_hash", "global_batch",
                                        "max_optimizer_steps", "lr", "warmup_steps",
                                        "schedule_total_steps", "amp_dtype",
                                        "loss_weights", "noise_sigma", "atom_mask_ratio")
                        if identities["fixed_concat_geonorm"][key] == identities["broken_concat"][key]
                        == identities["kfuse"][key]],
            "changed_fixed_vs_broken": {
                "geometry_head_norm": [identities["broken_concat"]["geometry_head_norm"],
                                       identities["fixed_concat_geonorm"]["geometry_head_norm"]],
                "world_size": [identities["broken_concat"]["world_size"],
                               identities["fixed_concat_geonorm"]["world_size"]],
                "accumulation": [identities["broken_concat"]["accumulation"],
                                 identities["fixed_concat_geonorm"]["accumulation"]],
            },
            "caveat": ("microbatch 84 and global batch 1008 are matched, but world_size differs "
                       "(4 -> 3), so the per-step sample composition is not identical; the "
                       "sample-matched single-change test is the P2 replay, not this comparison"),
        },
        "snapshots": {
            name: {str(step): _snapshot(matrix, step) for step in SNAPSHOTS}
            for name, matrix in run_logs.items()
        },
        "windows": {
            name: {str(size): _window(matrix, steps[name], size) for size in WINDOWS}
            for name, matrix in run_logs.items()
        },
        "fixed_only": {
            "geometry_head_norm": identities["fixed_concat_geonorm"]["geometry_head_norm"],
            "length_loss_per_graph": {
                "steady_max": max(length_steady, key=lambda item: item[1]) if length_steady else None,
                "steady_median": statistics.median([value for _, value in length_steady]) if length_steady else None,
                "transient_max": max(length_series, key=lambda item: item[1]) if length_series else None,
                "transient_steps_above_1": [step for step, value in length_series if value > 1.0],
                "note": ("transient = step <= %d; the same statistic in the broken replay peaked at "
                         "64.78 per graph at step 2676" % INITIAL_TRANSIENT_STEPS),
            },
            "raw_graph_3d_rms": graph3d,
            "raw_bond_states_rms": bond_rms,
            "angle_pre_tanh_abs_max": {
                "overall": max(pre_tanh_abs, key=lambda item: item[1]),
                "steady": max((item for item in pre_tanh_abs
                               if item[0] > INITIAL_TRANSIENT_STEPS), key=lambda item: item[1]),
            },
            "angle_exact_saturation": {
                "max": max(saturation_values, key=lambda item: item[1]),
                "median": statistics.median([value for _, value in saturation_values]),
            },
            "angle_tanh_derivative_min": {
                "overall": min(tanh_derivative, key=lambda item: item[1]),
                "typical_median": statistics.median([value for _, value in tanh_derivative]),
            },
            "angle_head_grad": {
                "min_last500": min(tail(grad_angle, 500)),
                "median_last500": statistics.median(tail(grad_angle, 500)),
                "max_last500": max(tail(grad_angle, 500)),
            },
            "length_head_grad": {
                "min_last500": min(tail(grad_length, 500)),
                "median_last500": statistics.median(tail(grad_length, 500)),
            },
            "preclip_grad_total": {
                "steady_max": max(grad_total_steady, key=lambda item: item[1]),
                "max_last500": max(grad_total[-500:], key=lambda item: item[1]),
            },
            "residual_concern": {
                "raw_bond_states_rms_final": bond_rms["final"],
                "pre_failure_healthy_band": [1.07, 1.22],
                "broken_band": [5.77, 9.17],
                "statement": ("the normalized geometry path is stable and the geometry objective "
                              "converged, but the raw 3D representation still sits above the "
                              "pre-failure healthy band; not a blocking failure of this round, "
                              "reported for the next planning step"),
            },
        },
        "broken_route_evidence": {
            "source": str(Path(args.broken_replay) / "diagnostics_steps.jsonl"),
            "step_range": [int(replay_rows[0]["step"]), int(replay_rows[-1]["step"])],
            "peak_per_graph_length_loss": [max(replay_length, key=lambda item: item[0])[0],
                                           max(replay_length, key=lambda item: item[0])[1]],
            "graph_3d_rms_first_last_max": [replay_graph3d[0], replay_graph3d[-1],
                                            max(replay_graph3d, key=lambda item: item[1])],
            "max_exact_saturation": max(replay_saturation, key=lambda item: item[1]),
            "min_tanh_derivative": min(replay_tanh, key=lambda item: item[1]),
            "caveat": ("this replay resumed from the already-drifting broken checkpoint at step 2000; "
                       "it documents the failure state, it is not a matched control for the fixed run"),
        },
        "thresholds": {
            "length_per_graph_max": LENGTH_PER_GRAPH_MAX,
            "graph_3d_rms_max": GRAPH3D_RMS_MAX,
            "saturation_fraction_max": SATURATION_FRACTION_MAX,
            "tanh_derivative_min": TANH_DERIVATIVE_MIN,
            "angle_grad_last500_min": ANGLE_GRAD_LAST500_MIN,
            "rank_spread_rtol": RANK_SPREAD_RTOL,
            "initial_transient_steps": INITIAL_TRANSIENT_STEPS,
        },
        "checks": checks,
        "blocking_checks": blocking,
        "check_results": verdict,
        "FORMAL_FIXED_5K_HEALTHY": "YES" if healthy else "NO",
    }
    write_json(args.output, report)
    print(json.dumps({key: report[key] for key in
                      ("FORMAL_FIXED_5K_HEALTHY", "check_results", "steps_available")},
                     indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
