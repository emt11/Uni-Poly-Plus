#!/usr/bin/env python3
"""Phase A: quantify which source-level features explain the audit FAIL statuses.

Read-only.  Joins the published per-sample audit records with the cohort's
normalized source SMILES and computes, for every AUDIT_FAIL record and a
comparable control sample of AUDIT_PASS records, the source features that the
audit's own derivation does not represent:

* explicit hydrogen atoms written in the source P-SMILES (``[2H]``/``[H]``);
  RDKit's ``GetTotalNumHs`` excludes them while the audit's ``h_counts`` counts
  every atom with atomic number 1, so each explicit H shifts the expected count;
* an asymmetric / aromatic attachment pair, where the declared terminal-capping
  policy "mismatch_single" makes the finite seam order differ from a dummy's own
  order, which the audit only models at the two outer cap atoms.

This is a bounded explanation study over existing records: it does not estimate
an error rate and does not re-label any sample as chemically valid.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rdkit import Chem

FAIL_CODES = ("FAIL_UNEXPECTED_H", "FAIL_VALENCE", "FAIL_CENTER_RU_CHEMISTRY",
              "FAIL_TERMINAL_CAP", "FAIL_HYBRIDIZATION_CHEMISTRY", "FAIL_STEREO")


def _source_features(smiles):
    molecule = Chem.MolFromSmiles(str(smiles))
    if molecule is None:
        return None
    explicit_h = 0
    isotope_h = 0
    dummies = []
    aromatic_atoms = 0
    charged_aromatic = 0
    for atom in molecule.GetAtoms():
        number = int(atom.GetAtomicNum())
        if number == 1:
            explicit_h += 1
            if int(atom.GetIsotope()) != 0:
                isotope_h += 1
            continue
        if number == 0:
            neighbours = list(atom.GetNeighbors())
            if len(neighbours) == 1:
                bond = molecule.GetBondBetweenAtoms(int(atom.GetIdx()),
                                                   int(neighbours[0].GetIdx()))
                code = (4 if bond.GetIsAromatic() else
                        {Chem.BondType.SINGLE: 1, Chem.BondType.DOUBLE: 2,
                         Chem.BondType.TRIPLE: 3}.get(bond.GetBondType(), 5))
                dummies.append(code)
            continue
        if atom.GetIsAromatic():
            aromatic_atoms += 1
            if int(atom.GetFormalCharge()) != 0:
                charged_aromatic += 1
    asymmetric = bool(len(dummies) == 2 and (dummies[0] != dummies[1] or 4 in dummies))
    return {
        "explicit_h": explicit_h, "isotope_h": isotope_h,
        "dummy_codes": dummies, "asymmetric_attachment": asymmetric,
        "aromatic_atoms": aromatic_atoms, "charged_aromatic": charged_aromatic,
        "heavy_atoms": int(molecule.GetNumHeavyAtoms()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-jsonl", required=True)
    parser.add_argument("--cohort-root", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--pass-sample", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--explained-keys-jsonl", default=None,
                        help="optional path writing the decomposed H-mismatch keys")
    args = parser.parse_args()

    fail_records, pass_keys = {}, []
    status_counts = Counter()
    failure_counts = Counter()
    with Path(args.audit_jsonl).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            status = record.get("status")
            status_counts[status] += 1
            for code in record.get("failures") or []:
                failure_counts[str(code).split(":")[0]] += 1
            if status == "AUDIT_FAIL":
                fail_records[record["sample_key"]] = {
                    "source_row": record.get("source_row"),
                    "failures": sorted(str(c).split(":")[0] for c in record.get("failures") or []),
                    "unresolved": sorted(str(c).split(":")[0] for c in record.get("unresolved") or []),
                    "flags": record.get("flags") or {},
                }
            elif status == "AUDIT_PASS":
                pass_keys.append(record["sample_key"])

    rng = random.Random(args.seed)
    rng.shuffle(pass_keys)
    control_keys = set(pass_keys[:args.pass_sample])
    wanted = set(fail_records) | control_keys

    smiles_by_key = {}
    with (Path(args.cohort_root) / "records.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            key = row["sample_key"]
            if key in wanted:
                smiles_by_key[key] = row.get("normalized_smiles") or row.get("source_smiles")

    def bucket(features):
        if features is None:
            return "UNPARSEABLE"
        if features["explicit_h"] > 0 and features["asymmetric_attachment"]:
            return "EXPLICIT_H_AND_ASYMMETRIC_ATTACHMENT"
        if features["explicit_h"] > 0:
            return "EXPLICIT_H_SOURCE"
        if features["asymmetric_attachment"]:
            return "ASYMMETRIC_ATTACHMENT"
        return "NEITHER"

    report = {"fail": defaultdict(Counter), "pass": defaultdict(Counter),
              "fail_failure_combo": Counter(), "fail_by_code": Counter(),
              "fail_feature_rows": [], "counts": {}}
    for key, info in sorted(fail_records.items()):
        features = _source_features(smiles_by_key.get(key))
        name = bucket(features)
        report["fail"][name]["total"] += 1
        if features is not None:
            report["fail"][name]["explicit_h_atoms"] += features["explicit_h"]
            report["fail"][name]["isotope_h_atoms"] += features["isotope_h"]
        for code in info["failures"]:
            if code in FAIL_CODES:
                report["fail_by_code"][code] += 1
        report["fail_failure_combo"]["+".join(info["failures"])] += 1
        if features is not None and name != "NEITHER":
            report["fail_feature_rows"].append({
                "sample_key": key, "bucket": name,
                "explicit_h": features["explicit_h"], "isotope_h": features["isotope_h"],
                "dummy_codes": features["dummy_codes"], "failures": info["failures"],
            })
    for key in sorted(control_keys):
        features = _source_features(smiles_by_key.get(key))
        report["pass"][bucket(features)]["total"] += 1

    report["counts"] = {
        "audit_status": dict(status_counts),
        "pass_control_sampled": len(control_keys),
        "fail_records_with_smiles": len(smiles_by_key) - len(
            control_keys & set(smiles_by_key)),
    }
    total_fail = sum(c["total"] for c in report["fail"].values())
    neither = report["fail"]["NEITHER"]["total"] + report["fail"]["UNPARSEABLE"]["total"]
    report["interpretation"] = {
        "fail_total": total_fail,
        "fail_explained_by_source_features": total_fail - neither,
        "fail_neither": neither,
        "note": ("feature buckets describe source-level properties that the audit's expected-H "
                 "derivation does not represent; they are not an error-rate estimate and do not "
                 "re-label the samples as chemically valid"),
    }
    feature_rows = report["fail_feature_rows"]
    report = {
        "counts": report["counts"],
        "fail_buckets": {k: dict(v) for k, v in report["fail"].items()},
        "pass_control_buckets": {k: dict(v) for k, v in report["pass"].items()},
        "fail_by_audit_code": dict(report["fail_by_code"]),
        "top_failure_combos": report["fail_failure_combo"].most_common(10),
        "interpretation": report["interpretation"],
        "feature_rows_written": len(feature_rows),
    }
    Path(args.output_json).write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")
    if args.explained_keys_jsonl:
        with Path(args.explained_keys_jsonl).open("w", encoding="utf-8") as handle:
            for row in feature_rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
