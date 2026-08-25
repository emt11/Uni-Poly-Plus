#!/usr/bin/env python3
"""Assemble GraphGate paired metrics, coverage and channel-gate diagnostics."""

import argparse
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]


def _mean(values):
    return float(sum(values) / len(values)) if values else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--paired-summary", required=True)
    parser.add_argument("--audit-root", required=True)
    parser.add_argument("--stage", choices=("screen_5k", "formal_20k"), required=True)
    parser.add_argument("--output-root", default="results/mts_glt_graphgate_v1")
    args = parser.parse_args()
    paired = json.loads((ROOT / args.paired_summary).read_text(encoding="utf-8"))
    coverage = json.loads((ROOT / "results/mts_glt_graphgate_v1/sidecar_qc/coverage_report.json").read_text(encoding="utf-8"))
    units = []
    for path in sorted((ROOT / args.audit_root).glob("*/fold_*.json")):
        units.append(json.loads(path.read_text(encoding="utf-8")))
    test_rho = [unit["test"]["rho"] for unit in units if unit.get("test")]
    alpha = [unit["test"]["alpha"] for unit in units if unit.get("test")]
    fold_diagnostics = {
        f"{unit['task']}/fold_{int(unit['fold'])}": unit["test"]
        for unit in units if unit.get("test")
    }
    diagnostics = {
        "fold_count": len(units),
        "rho_mean": _mean([row["mean"] for row in test_rho if row["mean"] is not None]),
        "rho_median_fold": statistics.median([row["median"] for row in test_rho if row["median"] is not None]) if test_rho else None,
        "rho_p10_mean": _mean([row["p10"] for row in test_rho if row["p10"] is not None]),
        "rho_p90_mean": _mean([row["p90"] for row in test_rho if row["p90"] is not None]),
        "alpha_mean_abs": _mean([row["mean_abs"] for row in alpha]),
        "alpha_median_abs": _mean([row["median_abs"] for row in alpha]),
        "alpha_p10": _mean([row["p10"] for row in alpha]),
        "alpha_p90": _mean([row["p90"] for row in alpha]),
        "fraction_abs_lt_0_01": _mean([row["fraction_abs_lt_0_01"] for row in alpha]),
        "fraction_positive": _mean([row["fraction_positive"] for row in alpha]),
        "fraction_negative": _mean([row["fraction_negative"] for row in alpha]),
    }
    if args.stage == "screen_5k":
        passed = paired["macro_delta"] > 0 and paired["median_task_delta"] > 0 and paired["positive_tasks"] >= 2
    else:
        passed = paired["macro_delta"] > 0 and paired["median_task_delta"] > 0 and paired["positive_tasks"] >= 5
    payload = {
        "schema": "mts-glt-graphgate-v1-report-v1", "stage": args.stage,
        "paired": paired, "coverage": coverage, "fusion_diagnostics": diagnostics,
        "fusion_diagnostics_by_fold": fold_diagnostics,
        "passed": bool(passed),
        "comparison": "same GraphGate pretrain checkpoint: O8-only vs O8+GLT GraphGate",
        "candidate_benchmark_only": 0.843636,
    }
    output_root = ROOT / args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    json_path = output_root / f"{args.stage}_report.json"
    md_path = output_root / f"{args.stage}_report.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [f"# MTS-GLT-GraphGate-v1 {args.stage}", "", f"- O8-only macro: `{paired['macro_o8']:.6f}`",
             f"- O8+GLT macro: `{paired['macro_fused']:.6f}`", f"- Delta: `{paired['macro_delta']:+.6f}`",
             f"- Median task delta: `{paired['median_task_delta']:+.6f}`", f"- Positive tasks: `{paired['positive_tasks']}/{len(paired['tasks'])}`",
             f"- Stage passed: `{str(passed).lower()}`", "", "## Task results", "", "| Task | O8-only mean +/- sample std | O8+GLT mean +/- sample std | Delta mean +/- sample std | Positive folds |", "|---|---:|---:|---:|---:|"]
    for task, row in paired["tasks"].items():
        lines.append(
            f"| {task} | {row['o8_mean_r2']:.6f} +/- {row['o8_sample_std_r2']:.6f} "
            f"| {row['fused_mean_r2']:.6f} +/- {row['fused_sample_std_r2']:.6f} "
            f"| {row['mean_delta']:+.6f} +/- {row['sample_std_delta']:.6f} "
            f"| {row['positive_folds']} |"
        )
    lines += ["", "## Fusion diagnostics", "", f"- rho mean: `{diagnostics['rho_mean']}`", f"- mean |alpha|: `{diagnostics['alpha_mean_abs']}`",
              f"- fraction |alpha| < 0.01: `{diagnostics['fraction_abs_lt_0_01']}`", "", "`0.843636` is a candidate-level benchmark only; this report does not attribute gains to a single component."]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json_path)


if __name__ == "__main__":
    main()
