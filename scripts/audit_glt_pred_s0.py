#!/usr/bin/env python3
"""Read-only S0 audit for GLT-PRED-20260918.

The audit deliberately does not build a cache, generate coordinates, run a
model, or read any outer-test predictions.  It verifies the immutable cohort
joins and creates the deterministic P_train/P_val identity index used by the
later, separately-authorised stages.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

from src.training.glt_dual_runtime import open_source


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
            yield row


def _identity_split(identities, seed=42):
    """Return deterministic 95/5 sets without task labels.

    Sorting the identity strings before applying the seeded permutation keeps
    the split independent of JSONL ordering.  The split is by identity, not
    by repeated property rows or by sample key.
    """

    values = sorted(set(str(value) for value in identities))
    generator = np.random.default_rng(int(seed))
    order = generator.permutation(len(values))
    n_train = int(round(0.95 * len(values)))
    train = {values[int(index)] for index in order[:n_train]}
    validation = {values[int(index)] for index in order[n_train:]}
    if train & validation or len(train) + len(validation) != len(values):
        raise AssertionError("identity split is not disjoint/exhaustive")
    return train, validation


def _xc_diagnostics(path: Path):
    result = {"path": str(path.resolve()), "exists": path.is_file()}
    if not path.is_file():
        return result
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    result["rows"] = len(rows)
    result["columns"] = list(rows[0]) if rows else []
    smiles_column = next((name for name in ("smiles", "SMILES", "source_smiles")
                          if rows and name in rows[0]), None)
    label_columns = [name for name in (rows[0] if rows else {})
                     if name not in {smiles_column, "sample_key"}]
    result["smiles_column"] = smiles_column
    result["label_columns"] = label_columns
    if smiles_column:
        values = [str(row.get(smiles_column, "")) for row in rows]
        result["duplicate_smiles_rows"] = len(values) - len(set(values))
    for name in label_columns:
        numbers = []
        for row in rows:
            try:
                number = float(row[name])
            except (TypeError, ValueError):
                continue
            if math.isfinite(number):
                numbers.append(number)
        if numbers:
            result.setdefault("finite_label_ranges", {})[name] = {
                "count": len(numbers), "min": min(numbers), "max": max(numbers),
            }
    return result


def _candidate_counts(topology):
    edge_index = torch.as_tensor(topology.lga_edge_index).long()
    spd = torch.as_tensor(topology.lga_spd).long()
    if edge_index.ndim != 2 or edge_index.shape[0] != 2 or spd.numel() != edge_index.shape[1]:
        raise ValueError("invalid canonical shortest-path table")
    counts = {"spd2": 0, "spd3": 0}
    seen = set()
    for column, value in enumerate(spd.tolist()):
        if value not in (2, 3):
            continue
        left, right = (int(edge_index[0, column]), int(edge_index[1, column]))
        if left == right:
            # Periodic lifted tables may contain a self image at a nonzero
            # shift.  FGR requires two distinct centre-RU atoms, so this is
            # not a valid pair even though its graph SPD is nonzero.
            continue
        pair = (min(left, right), max(left, right))
        if pair in seen:
            continue
        seen.add(pair)
        counts[f"spd{value}"] += 1
    counts["total"] = counts["spd2"] + counts["spd3"]
    return counts


def _geometry_sample(topology, trimer):
    counts = _candidate_counts(topology)
    geometry_valid = bool(getattr(trimer, "trimer_geometry_valid", False))
    coordinates = torch.as_tensor(getattr(trimer, "trimer_pos", torch.empty(0, 3))).float()
    mapping = torch.as_tensor(getattr(trimer, "mips_to_trimer_central_index", torch.empty(0))).long()
    distances_finite = False
    if geometry_valid and counts["total"] and mapping.numel() == int(topology.mips_x.size(0)):
        edge_index = torch.as_tensor(topology.lga_edge_index).long()
        spd = torch.as_tensor(topology.lga_spd).long()
        pairs = []
        for column, value in enumerate(spd.tolist()):
            if value not in (2, 3):
                continue
            left, right = (int(edge_index[0, column]), int(edge_index[1, column]))
            if left == right:
                continue
            pair = (min(left, right), max(left, right))
            if pair not in pairs:
                pairs.append(pair)
        if pairs:
            indices = torch.as_tensor(pairs, dtype=torch.long)
            positions = coordinates[mapping[indices]]
            distances = torch.linalg.vector_norm(positions[:, 0] - positions[:, 1], dim=-1)
            distances_finite = bool(torch.isfinite(distances).all() and (distances > 0).all())
    return {
        "geometry_valid": geometry_valid,
        "candidate_counts": counts,
        "candidate_distance_finite_positive": distances_finite,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pi1m-cohort", required=True)
    parser.add_argument("--downstream-cohort", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--split-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sample-limit", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.sample_limit < 0:
        raise ValueError("--sample-limit must be non-negative")

    pi1m_root = Path(args.pi1m_cohort)
    downstream_root = Path(args.downstream_cohort)
    pi1m_manifest = json.loads((pi1m_root / "manifest.json").read_text(encoding="utf-8"))
    downstream_manifest = json.loads((downstream_root / "manifest.json").read_text(encoding="utf-8"))

    pi_rows = list(_read_jsonl(pi1m_root / "records.jsonl"))
    down_rows = list(_read_jsonl(downstream_root / "records.jsonl"))
    required_pi = {"sample_key", "source_smiles", "normalized_smiles"}
    if any(required_pi - set(row) for row in pi_rows):
        raise ValueError("PI1M cohort record lacks required identity fields")
    required_down = required_pi | {"task", "label", "original_row"}
    if any(required_down - set(row) for row in down_rows):
        raise ValueError("downstream cohort record lacks required identity fields")

    pi_ids = [str(row["normalized_smiles"]) for row in pi_rows]
    down_ids = [str(row["normalized_smiles"]) for row in down_rows]
    pi_keys = [str(row["sample_key"]) for row in pi_rows]
    duplicate_pi_keys = len(pi_keys) - len(set(pi_keys))
    duplicate_pi_ids = len(pi_ids) - len(set(pi_ids))
    down_identity_counts = Counter(down_ids)
    overlap = set(pi_ids) & set(down_ids)
    eligible_ids = sorted(set(pi_ids) - overlap)
    train_ids, validation_ids = _identity_split(eligible_ids, seed=args.seed)
    selected = [
        (index, row) for index, row in enumerate(pi_rows)
        if str(row["normalized_smiles"]) in train_ids
    ][:args.sample_limit]

    split_root = Path(args.split_root)
    split_manifests = {}
    for path in sorted(split_root.glob("*.json")):
        try:
            split_manifests[path.stem] = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            split_manifests[path.stem] = {"status": "INVALID_JSON"}

    target_summary = {
        "sample_limit": int(args.sample_limit), "selected_count": len(selected),
        "geometry_valid_count": 0, "candidate_nonempty_count": 0,
        "candidate_distance_valid_count": 0, "spd2_total": 0, "spd3_total": 0,
        "ru_sizes": Counter(), "geometry_invalid_reasons": Counter(),
    }
    source = None
    if selected:
        source, _ = open_source(str(pi1m_root), args.cache_root)
        try:
            for index, _ in selected:
                topology, trimer, _ = source[index]
                sample = _geometry_sample(topology, trimer)
                if sample["geometry_valid"]:
                    target_summary["geometry_valid_count"] += 1
                else:
                    reason = str(getattr(trimer, "trimer_geometry_invalid_reason", "unknown"))
                    target_summary["geometry_invalid_reasons"][reason] += 1
                counts = sample["candidate_counts"]
                target_summary["spd2_total"] += counts["spd2"]
                target_summary["spd3_total"] += counts["spd3"]
                if counts["total"]:
                    target_summary["candidate_nonempty_count"] += 1
                if sample["candidate_distance_finite_positive"]:
                    target_summary["candidate_distance_valid_count"] += 1
                target_summary["ru_sizes"][int(topology.mips_x.size(0))] += 1
        finally:
            source.close()
    target_summary["geometry_invalid_reasons"] = dict(target_summary["geometry_invalid_reasons"])
    target_summary["ru_sizes"] = {str(k): int(v) for k, v in target_summary["ru_sizes"].items()}
    if target_summary["selected_count"]:
        target_summary["fgr_coverage_fraction"] = (
            target_summary["candidate_distance_valid_count"] /
            target_summary["selected_count"]
        )
    else:
        target_summary["fgr_coverage_fraction"] = None

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "PASS",
        "audit": "GLT-PRED-20260918-01/S0",
        "seed": int(args.seed),
        "identity_contract": "cohort normalized_smiles (canonical P-SMILES)",
        "pi1m": {
            "root": str(pi1m_root.resolve()), "manifest": pi1m_manifest,
            "record_count": len(pi_rows), "unique_identity_count": len(set(pi_ids)),
            "duplicate_identity_rows": duplicate_pi_ids,
            "duplicate_sample_key_rows": duplicate_pi_keys,
            "manifest_sample_count": int(pi1m_manifest.get("sample_count", -1)),
        },
        "downstream": {
            "root": str(downstream_root.resolve()), "manifest": downstream_manifest,
            "record_count": len(down_rows), "unique_identity_count": len(set(down_ids)),
            "task_counts": dict(Counter(str(row["task"]) for row in down_rows)),
            "identity_reuse_rows": int(sum(v - 1 for v in down_identity_counts.values() if v > 1)),
        },
        "identity_overlap": {
            "pi1m_downstream_unique_identity_count": len(overlap),
            "eligible_pi1m_unique_identity_count": len(eligible_ids),
            "excluded_identity_rule": "all downstream normalized_smiles",
        },
        "p_split": {
            "train_identity_count": len(train_ids),
            "validation_identity_count": len(validation_ids),
            "train_fraction": len(train_ids) / max(1, len(eligible_ids)),
            "validation_fraction": len(validation_ids) / max(1, len(eligible_ids)),
            "seed": int(args.seed),
            "identity_sha256": hashlib.sha256(
                "\n".join(sorted(eligible_ids)).encode("utf-8")).hexdigest(),
        },
        "target_coverage": target_summary,
        "xc_source": _xc_diagnostics(Path(args.raw_root) / "smi_xc.csv"),
        "split_manifests": {
            name: {
                "protocol": value.get("protocol"), "sample_count": value.get("sample_count"),
                "fold_count": len(value.get("folds", [])),
            } for name, value in split_manifests.items()
        },
        "outer_test_accessed": False,
        "model_training_executed": False,
        "cache_modified": False,
        "sampled_records": [
            {"index": int(index), "sample_key": str(row["sample_key"]),
             "normalized_smiles": str(row["normalized_smiles"])}
            for index, row in selected
        ],
        "inputs_sha256": {
            "pi1m_manifest": _sha256_file(pi1m_root / "manifest.json"),
            "downstream_manifest": _sha256_file(downstream_root / "manifest.json"),
            "pi1m_records": _sha256_file(pi1m_root / "records.jsonl"),
            "downstream_records": _sha256_file(downstream_root / "records.jsonl"),
        },
    }
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"status": "PASS", "output": str(output),
                      "pi1m": len(pi_rows), "eligible": len(eligible_ids),
                      "sampled": len(selected), "fgr_coverage": target_summary["fgr_coverage_fraction"]},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
