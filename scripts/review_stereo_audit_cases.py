#!/usr/bin/env python3
"""Phase A bounded review of the frozen Trimer chemistry audit cases.

Read-only.  Selects a deterministic, documented set of at most 48 accepted real
records from the existing per-sample stereo audit and, for each one, recomputes
the H-count and valence expectations **per repeat-unit copy** so the audit's
aggregate flags can be attributed to a specific RU and atom.

The audit under review (scripts/audit_live_trimer_accepted.py) computes

    expected_h(base) = source total_h(base)
                     + connection_code  (only at the two outer boundary atoms)

and then derives both ``center_ru_chemistry_ok`` and ``terminal_cap_ok`` from
the same whole-molecule ``h_ok``/``valence_ok``.  This script reports, per case,
the observed H/valence, the audit expectation, and the expectation implied by
the declared terminal-capping rule
(``missing_seam_bond_order_hydrogen_equivalents``), without deciding the verdict
itself: the verdict fields are filled in only when the evidence is unambiguous.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import importlib.util
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from src.dataset.cache_lifecycle import ReadonlyArtifact, snapshot_tree

AUDIT_PATH = Path(__file__).resolve().parent / "audit_live_trimer_accepted.py"


def _load_audit_module():
    spec = importlib.util.spec_from_file_location("live_trimer_audit", AUDIT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_active_bindings(cache_root):
    store = json.loads((Path(cache_root) / "store.json").read_text(encoding="utf-8"))
    return store["artifacts"]


def _classify(record):
    """Deterministic selection class from the published audit record."""

    failures = set(str(code).split(":")[0] for code in record.get("failures") or [])
    unresolved = set(str(code).split(":")[0] for code in record.get("unresolved") or [])
    flags = record.get("flags") or {}
    stereo_status = (record.get("diagnostics") or {}).get("stereo", {}).get("status")
    declared = int((record.get("diagnostics") or {}).get("stereo", {}).get("declared_bonds") or 0)
    if "FAIL_HYBRIDIZATION_CHEMISTRY" in failures:
        return "FAIL_HYBRIDIZATION_CHEMISTRY"
    if "FAIL_STEREO" in failures:
        return "FAIL_STEREO"
    if "FAIL_VALENCE" in failures:
        return "FAIL_UNEXPECTED_H_WITH_VALENCE"
    if "FAIL_UNEXPECTED_H" in failures:
        return "FAIL_UNEXPECTED_H_ONLY"
    if failures:
        return "FAIL_OTHER:" + ",".join(sorted(failures))
    if record.get("status") == "AUDIT_UNRESOLVED" or unresolved:
        return "UNRESOLVED_STEREO_TERMINAL_REFERENCE"
    if record.get("status") == "AUDIT_PASS" and declared > 0:
        return "PASS_WITH_DECLARED_STEREO"
    if record.get("status") == "AUDIT_PASS" and stereo_status in (None, "N/A"):
        return "PASS_NO_DECLARED_STEREO"
    return "PASS_OTHER"


QUOTAS = (
    ("FAIL_UNEXPECTED_H_ONLY", 8),
    ("FAIL_UNEXPECTED_H_WITH_VALENCE", 8),
    ("FAIL_HYBRIDIZATION_CHEMISTRY", 4),
    ("FAIL_STEREO", 4),
    ("UNRESOLVED_STEREO_TERMINAL_REFERENCE", 8),
    ("PASS_WITH_DECLARED_STEREO", 6),
    ("PASS_NO_DECLARED_STEREO", 4),
)
TOTAL_CAP = 48
SPREAD_LIMIT = 400  # deterministic candidate window per class


def _select(records_path):
    """Two-pass deterministic selection; returns the chosen records."""

    classes = defaultdict(list)
    skipped_before = 0
    with Path(records_path).open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if not line.strip():
                continue
            record = json.loads(line)
            name = _classify(record)
            if len(classes[name]) < SPREAD_LIMIT:
                classes[name].append({
                    "index": index,
                    "sample_key": record["sample_key"],
                    "position": record.get("position"),
                    "source_row": record.get("source_row"),
                })
            else:
                skipped_before += 1
    selected, selection_report = [], {}
    for name, quota in QUOTAS:
        candidates = classes.get(name, [])
        if not candidates:
            selection_report[name] = {"available": 0, "chosen": 0}
            continue
        count = min(quota, len(candidates))
        step = max(1, len(candidates) // count)
        chosen = candidates[::step][:count]
        selection_report[name] = {
            "available_first_%d" % SPREAD_LIMIT: len(candidates), "chosen": len(chosen),
            "rule": "ascending file order, every step-th candidate (step=%d)" % step,
            "sample_keys": [item["sample_key"] for item in chosen],
        }
        selected.extend((name, item) for item in chosen)
    if len(selected) > TOTAL_CAP:
        selected = selected[:TOTAL_CAP]
    return selected, selection_report, skipped_before


def _per_ru_analysis(audit, source_info, arrays, base_global, dummy_base, connection_code):
    """Observed vs expected H count and valence, attributed to RU and base atom."""

    base_count = len(source_info["atoms"])
    h_counts = Counter()
    for index in np.flatnonzero(arrays["atomic"] == 1):
        parent = int(arrays["parents"][index])
        if parent >= 0:
            h_counts[parent] += 1
    source_dummy_order = Counter()
    for item in source_info["dummy"]:
        source_dummy_order[item["neighbor_base"]] += audit._bond_order(item["code"], item["aromatic"])
    expected_valence = {}
    for ru in (-1, 0, 1):
        expected_valence[ru] = {}
        for base_id, meta in enumerate(source_info["atoms"]):
            value = float(meta["total_valence"]) - float(source_dummy_order[base_id])
            value += float(connection_code * sum(
                item["neighbor_base"] == base_id for item in source_info["dummy"]))
            expected_valence[ru][base_id] = value

    h_mismatch, valence_mismatch = [], []
    for ru in (-1, 0, 1):
        for base_id, meta in enumerate(source_info["atoms"]):
            global_index = base_global[ru][base_id]
            expected_h_audit = int(meta["total_h"])
            if ru == -1 and base_id == dummy_base[0]:
                expected_h_audit += int(connection_code)
            if ru == 1 and base_id == dummy_base[1]:
                expected_h_audit += int(connection_code)
            owned = int(source_dummy_order[base_id])
            is_cap_atom = (ru == -1 and base_id == dummy_base[0]) or (ru == 1 and base_id == dummy_base[1])
            expected_h_declared = int(meta["total_h"]) + owned if is_cap_atom else int(meta["total_h"])
            observed_h = int(h_counts[global_index])
            if observed_h != expected_h_audit:
                h_mismatch.append({
                    "ru": int(ru), "base_id": int(base_id), "z": int(meta["z"]),
                    "aromatic": bool(meta["aromatic"]), "charge": int(meta["formal_charge"]),
                    "is_cap_atom": bool(is_cap_atom),
                    "expected_audit": expected_h_audit,
                    "expected_declared_cap_rule": expected_h_declared,
                    "observed": observed_h,
                    "delta_audit": observed_h - expected_h_audit,
                    "matches_declared_cap_rule": bool(observed_h == expected_h_declared),
                })
            expected_val = expected_valence[ru][base_id]
            valence_mismatch.append((expected_val, global_index, ru, base_id))
    # Observed valence is recomputed from the frozen directed bond table.  Each
    # physical bond is stored once per direction, so pairs are counted once.
    observed_valence = {}
    edge = arrays["edge"]
    counted_pairs = set()
    for column in range(edge.shape[1]):
        left, right = int(edge[0, column]), int(edge[1, column])
        pair = (min(left, right), max(left, right))
        if pair in counted_pairs:
            continue
        counted_pairs.add(pair)
        order = audit._bond_order(int(arrays["bond"][column]), bool(arrays["aromatic_bond"][column]))
        observed_valence[left] = observed_valence.get(left, 0.0) + order
        observed_valence[right] = observed_valence.get(right, 0.0) + order
    valence_rows = []
    for expected_val, global_index, ru, base_id in valence_mismatch:
        observed_val = float(observed_valence.get(global_index, 0.0))
        if abs(observed_val - expected_val) > 0.51:
            meta = source_info["atoms"][base_id]
            valence_rows.append({
                "ru": int(ru), "base_id": int(base_id), "z": int(meta["z"]),
                "aromatic": bool(meta["aromatic"]), "charge": int(meta["formal_charge"]),
                "is_cap_atom": bool((ru == -1 and base_id == dummy_base[0])
                                    or (ru == 1 and base_id == dummy_base[1])),
                "expected_audit": round(float(expected_val), 4),
                "observed_from_bond_table": round(observed_val, 4),
                "delta": round(observed_val - float(expected_val), 4),
            })
    return h_counts, h_mismatch, valence_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-jsonl", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--cohort-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-cases", type=int, default=TOTAL_CAP)
    parser.add_argument("--keys-jsonl", default=None,
                        help="optional {sample_key,bucket} list; review exactly these keys")
    args = parser.parse_args()

    torch.set_num_threads(1)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    audit = _load_audit_module()

    cache_root = Path(args.cache_root).resolve()
    bindings = _load_active_bindings(cache_root)
    topology_root = (cache_root / bindings["topology"]["path"]).resolve()
    trimer_root = (cache_root / bindings["trimer"]["path"]).resolve()
    before = [snapshot_tree(topology_root), snapshot_tree(trimer_root)]

    smiles_by_key = {}
    records_path = Path(args.cohort_root) / "records.jsonl"
    with records_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            smiles_by_key[row["sample_key"]] = row.get("normalized_smiles") or row.get("source_smiles")

    if args.keys_jsonl:
        selected, selection_report = [], {"explicit_key_list": {"path": args.keys_jsonl}}
        with Path(args.keys_jsonl).open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                selected.append((row.get("bucket", "EXPLICIT_KEY_LIST"),
                                 {"sample_key": row["sample_key"],
                                  "position": row.get("position"),
                                  "source_row": row.get("source_row")}))
        selection_report["explicit_key_list"]["count"] = len(selected)
    else:
        selected, selection_report, _ = _select(args.audit_jsonl)
    if args.max_cases < len(selected):
        selected = selected[:args.max_cases]

    topology = ReadonlyArtifact(cache_root, bindings["topology"])
    trimer = ReadonlyArtifact(cache_root, bindings["trimer"])
    cases = []
    try:
        wanted = {item["sample_key"]: name for name, item in selected}
        audit_records = {}
        with Path(args.audit_jsonl).open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                key = record.get("sample_key")
                if key in wanted:
                    audit_records[key] = record
                    if len(audit_records) == len(wanted):
                        break
        for name, item in selected:
            key = item["sample_key"]
            record = audit_records.get(key)
            row = {"selection_class": name, "sample_key": key,
                   "position": item["position"], "source_row": item["source_row"]}
            if record is None:
                row["analysis"] = "AUDIT_RECORD_NOT_FOUND"
                cases.append(row)
                continue
            row.update({
                "status": record.get("status"), "failures": record.get("failures"),
                "unresolved": record.get("unresolved"),
                "high_risk_groups": (record.get("diagnostics") or {}).get("high_risk_groups"),
                "flags": record.get("flags"),
                "stereo_status": (record.get("diagnostics") or {}).get("stereo", {}).get("status"),
                "stereo_declared_bonds": (record.get("diagnostics") or {}).get("stereo", {}).get("declared_bonds"),
            })
            smiles = smiles_by_key.get(key)
            payload_key = bytes.fromhex(key)
            try:
                top = topology[payload_key]
                tri = trimer[payload_key]
                source_info = audit._source_info(smiles)
                audit._topology_check(top, source_info)
                arrays = audit._record_arrays(tri)
                base_count = len(source_info["atoms"])
                base_global = {}
                source_region = arrays["is_source"]
                for ru in (-1, 0, 1):
                    base_global[ru] = {}
                    for base_id in range(base_count):
                        found = np.flatnonzero(source_region & (arrays["offset"] == ru)
                                               & (arrays["base"] == base_id))
                        base_global[ru][base_id] = int(found[0]) if found.size == 1 else None
                dummy_base = [d["neighbor_base"] for d in source_info["dummy"]]
                dummy_codes = [d["code"] for d in source_info["dummy"]]
                connection_code = 1 if dummy_codes[0] != dummy_codes[1] or 4 in dummy_codes else dummy_codes[0]
                _, h_mismatch, valence_rows = _per_ru_analysis(
                    audit, source_info, arrays, base_global, dummy_base, connection_code)
                cap_atoms = {(int(ru), int(base_id)) for ru, base_id in
                             ((-1, dummy_base[0]), (1, dummy_base[1]))}
                mismatch_keys = {(row_["ru"], row_["base_id"]) for row_ in h_mismatch}
                row["analysis"] = {
                    "normalized_smiles": source_info["canonical"],
                    "base_atom_count": int(base_count),
                    "dummy_base_atoms": [int(v) for v in dummy_base],
                    "dummy_bond_codes": [int(v) for v in dummy_codes],
                    "connection_code": int(connection_code),
                    "terminal_cap_atoms": sorted([list(v) for v in cap_atoms]),
                    "h_mismatch_count": len(h_mismatch),
                    "h_mismatches": h_mismatch[:12],
                    "h_mismatch_all_at_cap_atoms": bool(mismatch_keys and mismatch_keys <= cap_atoms),
                    "h_mismatch_all_match_declared_rule": bool(
                        h_mismatch and all(item["matches_declared_cap_rule"] for item in h_mismatch)),
                    "h_mismatch_rus": sorted({item["ru"] for item in h_mismatch}),
                    "valence_mismatch_count": len(valence_rows),
                    "valence_mismatches": valence_rows[:12],
                    "valence_mismatch_all_at_cap_atoms": bool(
                        valence_rows and {(r["ru"], r["base_id"]) for r in valence_rows} <= cap_atoms),
                    "bond_type_checks": {
                        name: bool((record.get("flags") or {}).get(name))
                        for name in ("aromatic_equivalence_ok", "formal_charge_ok",
                                     "heavy_connectivity_ok", "bond_table_ok",
                                     "o8_mapping_ok", "source_identity_ok")
                    },
                }
            except Exception as error:  # recorded per case, never fatal
                row["analysis"] = f"REVIEW_ERROR:{type(error).__name__}:{error}"
            cases.append(row)
    finally:
        topology.close()
        trimer.close()
    after = [snapshot_tree(topology_root), snapshot_tree(trimer_root)]

    summary = Counter(case.get("selection_class") for case in cases)
    outcome = Counter()
    for case in cases:
        analysis = case.get("analysis")
        if not isinstance(analysis, dict):
            outcome["NO_ANALYSIS"] += 1
            continue
        if analysis["h_mismatch_count"] == 0 and analysis["valence_mismatch_count"] == 0:
            outcome["NO_RECOMPUTED_MISMATCH"] += 1
        elif analysis["h_mismatch_all_at_cap_atoms"] or analysis["valence_mismatch_all_at_cap_atoms"]:
            outcome["MISMATCH_AT_TERMINAL_CAP_ATOMS"] += 1
        else:
            outcome["MISMATCH_OUTSIDE_CAP_ATOMS"] += 1
    report = {
        "scope": "Phase A bounded read-only review; no conformer generation, no build",
        "selected_class_counts": dict(summary),
        "recomputed_outcomes": dict(outcome),
        "selection": selection_report,
        "cache_zero_write": before == after,
        "cache_paths": {"topology": str(topology_root), "trimer": str(trimer_root)},
    }
    (output_dir / "selection.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")
    with (output_dir / "audit_cases.jsonl").open("w", encoding="utf-8") as handle:
        for case in cases:
            handle.write(json.dumps(case, sort_keys=True, ensure_ascii=False) + "\n")
    print(json.dumps({"selected": len(cases), "by_class": dict(summary),
                      "outcomes": dict(outcome), "cache_zero_write": before == after},
                     indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
