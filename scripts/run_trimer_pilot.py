#!/usr/bin/env python3
"""Formal 300-record Trimer pilot under the first-valid geometry protocol.

Protocol: 4 candidates per round, retry 4 (max 2 rounds, max 8 attempts),
60 s total wall-clock budget per polymer, first-valid selection, no energy
ranking, ordinary geometry failure -> exclusion (rejection ledger, no
placeholder record).  Contract/isotope/mapping corruption is a hard stop.

The 300 sample identities are reused verbatim from the previous pilot run
(pilot_samples.jsonl) so old-protocol and new-protocol results stay
comparable.  Output goes to a fresh build-spec-addressed directory:

    data/processed/trimer_pilot/<build_spec_hash>/
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, rdBase

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.validate_dual_glt import audit_frozen_stereo  # noqa: E402
from src.dataset.cache_spec import ROUTE_BUILD_SPEC_HASHES  # noqa: E402
from src.dataset.dataset import _compute_ru_base_layer, _compute_topology_layer  # noqa: E402
from src.dataset.lmdb_cache import sample_key_from_smiles  # noqa: E402
from src.dataset.trimer_mcl import (  # noqa: E402
    TRIMER_CANDIDATES_PER_ROUND,
    TRIMER_ETKDG_MAX_ITERATIONS,
    TRIMER_ETKDG_TIMEOUT_SECONDS,
    TRIMER_MAX_ROUNDS,
    TRIMER_MMFF_RELAX_MAX_ITERATIONS,
    TRIMER_SAMPLE_TIMEOUT_SECONDS,
    TrimerContractError,
    TrimerGeometryRejection,
    attach_finite_trimer_mcl,
)

SOURCE_PILOT = ROOT / "data/processed/trimer_v10_pilot_r2_20260913/pilot_samples.jsonl"


def write_json(path: Path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n",
                    encoding="utf-8")


def write_jsonl(path: Path, rows):
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")


def percentile(values, q):
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=float), q))


def fold_intersection_report(accepted_keys: set[str]) -> list[dict]:
    """Intersect the accepted downstream identities with the EXISTING fold
    splits (never regenerated): original vs accepted per task x fold x split."""

    splits_root = ROOT / "data/splits/mips_outer5_inner20"
    report = []
    for split_path in sorted(splits_root.glob("*.json")):
        task = split_path.stem
        manifest = json.loads(split_path.read_text(encoding="utf-8"))
        csv_path = ROOT / "data/raw" / f"smi_{task}.csv"
        smiles_rows = pd.read_csv(csv_path).iloc[:, 0].astype(str).str.strip().tolist()
        row_keys = [sample_key_from_smiles(smiles).hex() for smiles in smiles_rows]
        for fold in manifest["folds"]:
            fold_id = int(fold["fold"])
            for split_name in ("train", "validation", "test"):
                indices = fold[f"{split_name}_indices"]
                keys = [row_keys[int(i)] for i in indices]
                accepted = sum(1 for key in keys if key in accepted_keys)
                report.append({
                    "task": task, "fold": fold_id, "split": split_name,
                    "original_count": len(keys), "accepted_count": accepted,
                    "rejected_count": len(keys) - accepted,
                })
    return report


def analyze(summaries, rejections) -> dict:
    accepted_rows = [row for row in summaries if row["geometry_valid"]]
    rejected_rows = [row for row in summaries if not row["geometry_valid"]]
    source = len(summaries)

    def attempts_of(row):
        return row["candidate_attempts"]

    round0_accepted = sum(1 for row in accepted_rows if row["selected_round"] == 0)
    round1_accepted = sum(1 for row in accepted_rows if row["selected_round"] == 1)
    round1_invoked = sum(1 for row in summaries if row["round1_invoked"])
    candidate_distribution = Counter(
        row["selected_candidate_id"] for row in accepted_rows
    )
    attempt_values = [attempts_of(row) for row in summaries]
    times = [row["elapsed_seconds"] for row in summaries]
    rejection_breakdown = Counter(row["failure_code"] for row in rejected_rows)
    selected_converged = sum(1 for row in accepted_rows if row["mmff_converged"])
    selected_nonconverged = sum(1 for row in accepted_rows if not row["mmff_converged"])
    basic = {
        "source": source,
        "accepted": len(accepted_rows),
        "rejected": len(rejected_rows),
        "acceptance_rate": (len(accepted_rows) / source) if source else None,
        "rejection_rate": (len(rejected_rows) / source) if source else None,
        "contract_errors": 0,
        "accounting_consistent": source == len(accepted_rows) + len(rejected_rows),
    }
    report = {
        "basic_success": basic,
        "rejection_breakdown": dict(sorted(rejection_breakdown.items())),
        "rounds": {
            "accepted_in_round0": round0_accepted,
            "round1_invoked": round1_invoked,
            "round1_recovered": round1_accepted,
            "round1_failed": max(0, round1_invoked - round1_accepted),
        },
        "candidate_distribution": {
            str(k): v for k, v in sorted(candidate_distribution.items())
        },
        "attempts": {
            "mean": statistics.fmean(attempt_values) if attempt_values else None,
            "median": statistics.median(attempt_values) if attempt_values else None,
            "p90": percentile(attempt_values, 90),
            "max": max(attempt_values) if attempt_values else None,
        },
        "mmff_selected": {
            "converged": selected_converged,
            "nonconverged": selected_nonconverged,
            "nonconverged_rate": (
                selected_nonconverged / len(accepted_rows)
                if accepted_rows else None
            ),
        },
        "runtime_seconds": {
            "median": statistics.median(times) if times else None,
            "mean": statistics.fmean(times) if times else None,
            "p90": percentile(times, 90),
            "p95": percentile(times, 95),
            "max": max(times) if times else None,
            "budget": TRIMER_SAMPLE_TIMEOUT_SECONDS,
        },
    }
    return report


def report_markdown(report: dict) -> str:
    basic = report["basic_success"]
    lines = [
        "# Trimer first-valid 300-record pilot report",
        "",
        f"- source = {basic['source']}  accepted = {basic['accepted']}  "
        f"rejected = {basic['rejected']}",
        f"- acceptance_rate = {basic['acceptance_rate']:.4f}  "
        f"rejection_rate = {basic['rejection_rate']:.4f}",
        f"- contract_errors = {basic['contract_errors']}  "
        f"accounting_consistent = {basic['accounting_consistent']}",
        "",
        "## Rejection breakdown",
    ]
    for code, count in report["rejection_breakdown"].items():
        lines.append(f"- {code}: {count}")
    lines += [
        "",
        "## Rounds",
        f"- accepted_in_round0 = {report['rounds']['accepted_in_round0']}",
        f"- round1_invoked = {report['rounds']['round1_invoked']}",
        f"- round1_recovered = {report['rounds']['round1_recovered']}",
        f"- round1_failed = {report['rounds']['round1_failed']}",
        "",
        "## Candidate-id distribution (selected)",
    ]
    for key, count in report["candidate_distribution"].items():
        lines.append(f"- candidate {key}: {count}")
    attempts = report["attempts"]
    lines += [
        "",
        "## Candidate attempts per sample",
        f"- mean={attempts['mean']:.3f} median={attempts['median']:.1f} "
        f"p90={attempts['p90']:.1f} max={attempts['max']}",
        "",
        "## MMFF (diagnostic only, not selection)",
        f"- converged={report['mmff_selected']['converged']} "
        f"nonconverged={report['mmff_selected']['nonconverged']} "
        f"nonconverged_rate={report['mmff_selected']['nonconverged_rate']}",
        "",
        "## Runtime (seconds, whole sample)",
        f"- median={report['runtime_seconds']['median']:.3f} "
        f"mean={report['runtime_seconds']['mean']:.3f} "
        f"p90={report['runtime_seconds']['p90']:.3f} "
        f"p95={report['runtime_seconds']['p95']:.3f} "
        f"max={report['runtime_seconds']['max']:.3f} "
        f"budget={report['runtime_seconds']['budget']}",
    ]
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=Path, default=SOURCE_PILOT)
    args = parser.parse_args(argv)

    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"refusing non-empty output directory: {output}")
    output.mkdir(parents=True)

    samples = [
        json.loads(line)
        for line in args.samples.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(samples) != 300:
        raise RuntimeError(f"expected the fixed 300-sample list, got {len(samples)}")

    build_spec_hash = ROUTE_BUILD_SPEC_HASHES["trimer"]
    config = {
        "sample_source": str(args.samples),
        "sample_count": len(samples),
        "selection_rule": "identical identities to the previous 300-record pilot",
        "build_spec_hash": build_spec_hash,
        "geometry": {
            "num_candidates_per_round": TRIMER_CANDIDATES_PER_ROUND,
            "max_rounds": TRIMER_MAX_ROUNDS,
            "max_total_candidates": TRIMER_CANDIDATES_PER_ROUND * TRIMER_MAX_ROUNDS,
            "total_timeout_seconds": TRIMER_SAMPLE_TIMEOUT_SECONDS,
            "per_embed_timeout_seconds": TRIMER_ETKDG_TIMEOUT_SECONDS,
            "embed_max_iterations": TRIMER_ETKDG_MAX_ITERATIONS,
            "mmff_variant": "MMFF94",
            "mmff_relax_max_iterations": TRIMER_MMFF_RELAX_MAX_ITERATIONS,
            "selection": "first_valid",
            "energy_ranking": False,
            "failure_policy": "exclude_geometry_failure",
        },
        "seed_policy": "geometry_seed_spec + sample identity + round_id + candidate_id (sha256)",
        "rdkit_version": rdBase.rdkitVersion,
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json(output / "pilot_config.json", config)

    summaries, rejections = [], []
    started_all = time.monotonic()
    try:
        for position, sample in enumerate(samples, 1):
            started = time.monotonic()
            canonical = sample["canonical_smiles"]
            sample_id = sample["sample_id"]
            ru = _compute_ru_base_layer(canonical)
            topology = _compute_topology_layer(canonical, ru, max_hops=2)
            topology.smiles = canonical
            if not bool(getattr(topology, "graph_available", False)):
                raise TrimerContractError("pilot_topology_unavailable")
            try:
                record = attach_finite_trimer_mcl(
                    topology, canonical,
                    num_candidates=TRIMER_CANDIDATES_PER_ROUND,
                    max_rounds=TRIMER_MAX_ROUNDS,
                    timeout_seconds=TRIMER_SAMPLE_TIMEOUT_SECONDS,
                    sample_key=sample_id,
                )
            except TrimerGeometryRejection as rejection:
                entry = rejection.ledger_entry(sample_id, canonical)
                entry["pilot_index"] = sample["pilot_index"]
                summaries.append({
                    "sample_id": sample_id,
                    "canonical_smiles": canonical,
                    "geometry_valid": False,
                    "failure_code": rejection.code,
                    "round_reached": rejection.round_reached,
                    "candidate_attempts": rejection.candidate_attempts,
                    "round1_invoked": rejection.round_reached >= 1,
                    "selected_round": None,
                    "selected_candidate_id": None,
                    "mmff_converged": None,
                    "elapsed_seconds": time.monotonic() - started,
                })
                rejections.append(entry)
                write_jsonl(output / "rejections.jsonl", rejections)
                print(json.dumps({
                    "completed": position, "total": len(samples),
                    "sample_id": sample_id, "valid": False,
                    "code": rejection.code,
                    "attempts": rejection.candidate_attempts,
                    "seconds": summaries[-1]["elapsed_seconds"],
                }, sort_keys=True), flush=True)
                continue

            diagnostics = record.generation_diagnostics
            attempts = diagnostics["num_candidates_requested"] - (
                TRIMER_CANDIDATES_PER_ROUND
                - len(diagnostics["rounds"][-1]["candidates"])
            ) if diagnostics["rounds"] else 0
            # attempts: exact count = requested so far minus unattempted tail
            attempted_rows = [
                row for round_rows in diagnostics["rounds"]
                for row in round_rows["candidates"]
            ]
            attempts = len(attempted_rows)
            summary = {
                "sample_id": sample_id,
                "canonical_smiles": canonical,
                "geometry_valid": True,
                "failure_code": None,
                "round_reached": int(record.trimer_conformer_round_id),
                "candidate_attempts": attempts,
                "round1_invoked": int(diagnostics["num_rounds"]) > 1,
                "selected_round": int(record.trimer_conformer_round_id),
                "selected_candidate_id": int(record.trimer_conformer_candidate_id),
                "selected_energy": float(record.trimer_conformer_energy),
                "mmff_converged": bool(record.selected_converged),
                "elapsed_seconds": time.monotonic() - started,
                "attempts_rows": [
                    {k: row.get(k) for k in (
                        "round_id", "candidate_id", "final_valid", "rejection",
                        "mmff_converged", "mmff_energy",
                    )}
                    for row in attempted_rows
                ],
            }
            summaries.append(summary)
            independent = audit_frozen_stereo(
                record, canonical, topology=topology, return_details=True
            )
            if independent["double_bonds"] != diagnostics["declared_double_bond_stereo_count"] \
                    or independent["tetrahedral_centers"] != diagnostics["declared_tetrahedral_stereo_count"]:
                raise TrimerContractError("independent_final_stereo_count_mismatch")
            print(json.dumps({
                "completed": position, "total": len(samples),
                "sample_id": sample_id, "valid": True,
                "round": summary["selected_round"],
                "candidate": summary["selected_candidate_id"],
                "attempts": attempts,
                "seconds": summary["elapsed_seconds"],
            }, sort_keys=True), flush=True)
    except TrimerContractError as exc:
        write_json(output / "HARD_STOP.json", {
            "error": str(exc),
            "processed": len(summaries),
            "at": datetime.now(timezone.utc).isoformat(),
        })
        print(json.dumps({"hard_stop": str(exc), "processed": len(summaries)}))
        return 1

    report = analyze(summaries, rejections)
    write_jsonl(output / "rejections.jsonl", rejections)
    write_jsonl(output / "pilot_sample_summary.jsonl", summaries)
    pd.DataFrame(summaries).to_csv(
        output / "pilot_sample_summary.csv", index=False
    )
    accepted_keys = np.frombuffer(
        b"".join(bytes.fromhex(row["sample_id"]) for row in summaries if row["geometry_valid"]),
        dtype=np.uint8,
    ).reshape(-1, 32)
    np.save(output / "accepted_sample_keys.npy", accepted_keys)
    fold_report = fold_intersection_report({
        row["sample_id"] for row in summaries if row["geometry_valid"]
    })
    write_json(output / "fold_intersection.json", fold_report)
    report["fold_intersection"] = fold_report
    report["wall_time_seconds"] = time.monotonic() - started_all
    write_json(output / "pilot_report.json", report)
    (output / "pilot_report.md").write_text(report_markdown(report), encoding="utf-8")
    (output / "PILOT_COMPLETE").write_text(
        datetime.now(timezone.utc).isoformat() + "\n", encoding="utf-8"
    )
    print(json.dumps({"pilot_complete": True, **report["basic_success"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
