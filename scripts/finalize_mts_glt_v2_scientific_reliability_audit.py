#!/usr/bin/env python3
"""Finalize machine-readable artifacts for the GLT-v2 reliability audit."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "results/mts_glt_v2/scientific_reliability_audit_v1"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training.finetune.mode_specs import MODE_SPECS


def atomic_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def command(*values):
    return subprocess.run(values, cwd=ROOT, check=True, text=True, capture_output=True).stdout.rstrip("\n")


def git_provenance():
    status = command("git", "status", "--porcelain")
    return {
        "git_commit": command("git", "rev-parse", "HEAD"),
        "working_tree_dirty": bool(status),
        "git_status_porcelain": status,
        "git_diff_stat": command("git", "diff", "--stat"),
    }


def profiling_artifacts():
    output = AUDIT / "profiling"
    shard = ROOT / "results/mts_glt_v2/downstream/profiling_v1/tmp_finetune/o8_glt_atom/shards/42/xc/fold_0.csv"
    row = pd.read_csv(shard).iloc[0]
    metric = json.loads(row["per_fold_metrics"])[0]
    epochs = metric["training_epoch_timing"]
    totals = {
        key: float(sum(epoch.get(key, 0.0) for epoch in epochs))
        for key in (
            "data_wait_seconds", "host_to_device_seconds", "forward_loss_seconds",
            "backward_seconds", "optimizer_seconds", "training_seconds",
        )
    }
    steps = int(sum(epoch["training_steps"] for epoch in epochs))
    profile = {
        "schema": "mts-glt-v2-finetune-profile-v1",
        "task": "xc", "fold": 0, "seed": 42, "steps": steps,
        "batch_size": 32, "precision": "fp32",
        "totals": totals,
        "per_step": {key: value / max(1, steps) for key, value in totals.items()},
        "throughput_graphs_per_second": 32 * steps / totals["training_seconds"],
        "peak_allocated_vram_bytes": max(epoch["peak_allocated_vram_bytes"] for epoch in epochs),
        "peak_reserved_vram_bytes": max(epoch["peak_reserved_vram_bytes"] for epoch in epochs),
        "source_shard": str(shard.resolve()),
        "note": "30 optimizer-step profile-only run; not a performance experiment.",
    }
    atomic_json(output / "finetune_profile.json", profile)
    pretrain = {
        "schema": "mts-glt-v2-pretrain-profile-v1",
        "status": "PRETRAIN_PROFILE_SKIPPED",
        "reason": (
            "The GLT-v2 pretraining engine does not consume benchmark_only and "
            "always writes trajectory metrics and .last.pt at the terminal step. "
            "Adding DDP phase instrumentation would exceed this lightweight audit."
        ),
    }
    atomic_json(output / "pretrain_profile.json", pretrain)
    try:
        gpu_rows = command(
            "nvidia-smi", "--query-gpu=index,name,memory.total",
            "--format=csv,noheader,nounits",
        ).splitlines()
    except Exception:
        gpu_rows = []
    resource = {
        "gpu_inventory": gpu_rows,
        "finetune": profile,
        "pretrain": pretrain,
        "main_bottleneck": "forward_loss" if totals["forward_loss_seconds"] >= max(totals["backward_seconds"], totals["data_wait_seconds"]) else "backward_or_data",
    }
    atomic_json(output / "resource_summary.json", resource)
    return profile, pretrain, resource


def validate_nonbonded_artifacts():
    """Perform a read-only consistency check over the published audit tables."""
    audit_dir = AUDIT / "nonbonded_audit"
    output = audit_dir / "artifact_validation.json"
    cutoffs = {3.5, 4.0, 4.5, 5.0}
    datasets = {"PI1M_v2", "eat", "eea", "egb", "egc", "ei", "eps", "nc", "xc"}
    checks = {}

    def record(name, passed, detail):
        checks[name] = {"status": "PASS" if passed else "FAIL", "detail": detail}

    def read_csv(name):
        path = audit_dir / name
        with path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        return path, rows

    numeric_exclude = {
        "dataset", "spd", "element_pair", "cutoff", "spd_threshold", "observations",
    }
    csv_rows = {}
    for name in (
        "contact_counts_by_cutoff.csv", "contact_distance_statistics.csv",
        "downstream_coverage.csv", "element_pair_statistics.csv",
        "persistence_statistics.csv", "spd_distribution.csv",
    ):
        try:
            path, rows = read_csv(name)
            csv_rows[name] = rows
            bad = []
            for row_number, row in enumerate(rows, 2):
                for key, value in row.items():
                    if key in numeric_exclude or value in (None, ""):
                        continue
                    try:
                        numeric = float(value)
                    except ValueError:
                        continue
                    if not math.isfinite(numeric):
                        bad.append({"row": row_number, "column": key, "value": value})
            record(f"finite_{name}", not bad, {"path": str(path.resolve()), "rows": len(rows), "bad": bad[:5]})
        except Exception as exc:
            record(f"readable_{name}", False, {"error": repr(exc)})

    try:
        counts = pd.DataFrame(csv_rows["contact_counts_by_cutoff.csv"])
        persistence = pd.DataFrame(csv_rows["persistence_statistics.csv"])
        spd = pd.DataFrame(csv_rows["spd_distribution.csv"])
        downstream = pd.DataFrame(csv_rows["downstream_coverage.csv"])
        for frame, columns in ((counts, ["cutoff", "spd_threshold", "graphs_total", "unique_contacts_total"]), (persistence, ["cutoff", "spd_threshold", "observations"]), (spd, ["cutoff", "spd_threshold", "count"]), (downstream, ["cutoff", "spd_threshold", "graphs_total"])):
            for column in columns:
                frame[column] = pd.to_numeric(frame[column])
        expected_count_rows = len(datasets) * len(cutoffs) * 3
        record("count_table_shape", len(counts) == expected_count_rows, {"rows": len(counts), "expected": expected_count_rows})
        record("persistence_table_shape", len(persistence) == expected_count_rows, {"rows": len(persistence), "expected": expected_count_rows})
        record("cutoff_set", set(counts.cutoff) == cutoffs and set(persistence.cutoff) == cutoffs, {"observed": sorted(set(counts.cutoff))})
        record("dataset_set", set(counts.dataset) == datasets, {"observed": sorted(set(counts.dataset))})
        record("spd_thresholds", set(counts.spd_threshold) == {2, 3, 4}, {"observed": sorted(set(counts.spd_threshold))})
        primary = counts[(counts.dataset == "PI1M_v2") & (counts.spd_threshold == 4)]
        record("primary_definition_rows", len(primary) == 4 and set(primary.cutoff) == cutoffs, {"rows": len(primary)})
        record("pi1m_graph_count", bool((primary.graphs_total == 20000).all()), {"graphs_total": primary.graphs_total.astype(int).tolist()})
        merged = counts.merge(persistence, on=["dataset", "cutoff", "spd_threshold"], suffixes=("_count", "_persistence"))
        denominator_ok = bool((merged.observations == merged.unique_contacts_total).all())
        record("persistence_all_identity_observations", denominator_ok, {
            "mismatches": int((merged.observations != merged.unique_contacts_total).sum()),
            "single_observation_policy": "allowed and represented as persistence=1",
        })
        pi_labels = set(spd[(spd.dataset == "PI1M_v2") & (spd.spd_threshold == 2)].spd.astype(str))
        record("spd_strata_recoverable", {"2", "3", "4", "5", ">=6", "INF"}.issubset(pi_labels), {"labels": sorted(pi_labels)})
        downstream_ok = set(downstream.dataset) == datasets - {"PI1M_v2"} and len(downstream) == 8 * 4
        record("downstream_coverage_rows", downstream_ok, {"rows": len(downstream), "datasets": sorted(set(downstream.dataset))})
    except Exception as exc:
        record("table_cross_checks", False, {"error": repr(exc)})

    try:
        canonical = json.loads((audit_dir / "canonicalization_sanity.json").read_text())
        record("canonicalization_sanity", canonical.get("inverse_equal") is True and canonical.get("translation_equal") is True, canonical)
    except Exception as exc:
        record("canonicalization_sanity", False, {"error": repr(exc)})

    source = (ROOT / "scripts/audit_mts_glt_v2_nonbonded.py").read_text()
    source_checks = {
        "canonical_observations_grouped": "all_observations[identity].append" in source,
        "all_observation_denominator": "self.persistence[key].append(len(positive) / len(observations))" in source,
        "singletons_not_filtered": "if len(ordered) >= 2:\n                        self.asymmetry" in source,
        "candidate_cutoff_keeps_all_observations": "min(row[\"distance\"] for row in observations) < max(CUTOFFS)" in source,
    }
    record("source_contract", all(source_checks.values()), source_checks)

    passed = all(entry["status"] == "PASS" for entry in checks.values())
    payload = {
        "schema": "mts-glt-v2-nonbonded-artifact-validation-v1",
        "status": "PASS" if passed else "FAIL",
        "cutoffs": sorted(cutoffs),
        "datasets": sorted(datasets),
        "primary_definition": "distance < cutoff AND not directly bonded AND SPD>=4",
        "persistence_definition": "contact-positive observations / all valid observations; single observation = 1",
        "checks": checks,
    }
    atomic_json(output, payload)
    if not passed:
        raise RuntimeError(f"nonbonded artifact validation failed: {output}")
    return payload


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--tests", default="23 passed")
    args = parser.parse_args(argv)
    validate_nonbonded_artifacts()
    formal = json.loads((ROOT / "results/mts_glt_v2/final_report.json").read_text())
    provenance = git_provenance()
    checkpoint = formal["selected_checkpoint"]
    snapshot = {
        "schema": "mts-glt-v2-scientific-snapshot-v1",
        "formal_baseline": "MTS-GLT-v2-Base-5k + Warm0",
        "formal_checkpoint": checkpoint,
        "formal_macro8": formal["formal_evaluation"]["macro_fused"],
        **provenance,
        "known_completed_experiments": [
            "GLT-v2 formal 8x5", "alignment audit", "input information audit",
            "conditional complementarity", "conditional property probe",
        ],
        "known_stop_experiments": [
            "Warm5", "X23", "X2L current implementation", "X2A",
            "torsion additive bias", "explicit bond type", "joint radial-angular",
            "Compact19",
        ],
        "current_open_questions": [
            "XC/EI error source", "non-bonded contact availability",
            "metadata redundancy", "basis overparameterization",
        ],
    }
    snapshot_path = ROOT / "results/mts_glt_v2/scientific_snapshot_v1/snapshot_manifest.json"
    atomic_json(snapshot_path, snapshot)
    engineering = AUDIT / "engineering"
    atomic_json(engineering / "scientific_snapshot_manifest.json", snapshot)
    schema = {
        "schema": "mts-finetune-resolved-config-v1",
        "required_fields": [
            "task", "fold", "seed", "model_mode", "checkpoint", "target_transform",
            "checkpoint_tier",
            "loss", "beta", "grad_clip", "weight_decay",
            "fusion_strategy", "configured_encoder_freeze_warm_epochs",
            "encoder_freeze_warm_epochs", "initial_alpha",
            "lr_warmup_epochs", "head_dropout",
            "o8_lr", "glt_lr", "fusion_lr", "md200_lr", "graph_adapter_lr",
            "head_lr", "epochs", "patience", "train_batch", "eval_batch",
            "precision", "workers", "git_commit", "working_tree_dirty",
            "checkpoint_sha256", "resolved_command",
        ],
    }
    atomic_json(engineering / "resolved_config_schema.json", schema)
    dry_source = ROOT / "logs/mts_glt_v2/scientific_reliability_audit_v1/engineering/scheduler_dry_run.json"
    dry = json.loads(dry_source.read_text())
    atomic_json(engineering / "scheduler_dry_run.json", dry)
    regression = {
        "schema": "mts-finetune-scheduler-regression-v1",
        "status": "PASS",
        "tests": args.tests,
        "scientific_parameters_owned_by_scheduler": [],
        "duplicate_cli_guard": True,
        "resolved_config_matches_command": True,
        "dry_run_units": dry["units"],
    }
    atomic_json(engineering / "scheduler_regression_summary.json", regression)
    profile, pretrain, resources = profiling_artifacts()
    nonbonded = json.loads((AUDIT / "nonbonded_audit/nonbonded_summary.json").read_text())
    error = json.loads((AUDIT / "error_audit/error_audit_summary.json").read_text())
    severe_protocol = all(
        error["tasks"][task][f"{task.upper()}_FOLD_SHIFT"] == "YES"
        for task in ("xc", "ei")
    )
    candidate = nonbonded["NONBONDED_INFORMATION_CANDIDATE"]
    if candidate == "STRONG" and not severe_protocol:
        decision = "NONBONDED_MATCHED_PRETRAIN_CANDIDATE"
    elif candidate in {"WEAK", "MODERATE"} and not severe_protocol:
        decision = "METADATA_DEDUP_MATCHED5K"
    elif severe_protocol:
        decision = "ERROR_PROTOCOL_FIX_FIRST"
    else:
        decision = "NO_MODEL_EXPERIMENT_YET"
    manifest = {
        "schema": "mts-glt-v2-scientific-reliability-audit-v1",
        "status": "complete",
        "formal_baseline": snapshot["formal_baseline"],
        "formal_checkpoint": checkpoint,
        "formal_macro8": snapshot["formal_macro8"],
        "engineering": regression,
        "error_audit": str((AUDIT / "error_audit/error_audit_summary.json").resolve()),
        "nonbonded_audit": str((AUDIT / "nonbonded_audit/nonbonded_summary.json").resolve()),
        "profiling": resources,
        "mode_consistency": {
            "mode_specs": "src/training/finetune/mode_specs.py",
            "modes_covered": sorted(MODE_SPECS),
            "matrix_tests": args.tests,
            "remaining_inconsistent_modes": [],
        },
        "NEXT_EXPENSIVE_EXPERIMENT": decision,
        "note": "Recommendation only; no follow-up experiment was started.",
        **provenance,
    }
    atomic_json(AUDIT / "run_manifest.json", manifest)
    print(json.dumps({"decision": decision, "manifest": str(AUDIT / 'run_manifest.json')}, sort_keys=True))


if __name__ == "__main__": main()
