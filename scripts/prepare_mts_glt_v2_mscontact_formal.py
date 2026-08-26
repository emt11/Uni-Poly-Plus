#!/usr/bin/env python3
"""Import exact prior screening units into the isolated formal 8x5 roots."""

from __future__ import annotations

import filecmp
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
SOURCE_RESULTS = ROOT / "results/mts_glt_v2/mscontact_v1/downstream"
SOURCE_LOGS = ROOT / "logs/mts_glt_v2/mscontact_v1/downstream"
TARGET_RESULTS = ROOT / "results/mts_glt_v2/mscontact_v1/formal_8x5_s4_c5_v1"
TARGET_LOGS = ROOT / "logs/mts_glt_v2/mscontact_v1/formal_8x5_s4_c5_v1"
TASKS = ("xc", "ei", "eea")
FOLDS = (0, 1, 2)


def copy_exact(source: Path, target: Path):
    if not source.is_file():
        raise FileNotFoundError(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        shutil.copy2(source, target)
    if not filecmp.cmp(source, target, shallow=False):
        raise RuntimeError(f"reused artifact mismatch: {source} -> {target}")
    return {"source": str(source), "target": str(target), "byte_equal": True}


def main():
    copied = []
    for arm in ("s4", "c5_mixed"):
        for task in TASKS:
            for fold in FOLDS:
                copied.append(copy_exact(
                    SOURCE_RESULTS / arm / "shards/42" / task / f"fold_{fold}.csv",
                    TARGET_RESULTS / arm / "shards/42" / task / f"fold_{fold}.csv",
                ))
                copied.append(copy_exact(
                    SOURCE_RESULTS / arm / "predictions/42" / task / f"fold_{fold}.npz",
                    TARGET_RESULTS / arm / "predictions/42" / task / f"fold_{fold}.npz",
                ))
                for name in ("resolved_config.json", "resolved_command.txt"):
                    copied.append(copy_exact(
                        SOURCE_RESULTS / arm / "resolved/42" / task / f"fold_{fold}" / name,
                        TARGET_RESULTS / arm / "resolved/42" / task / f"fold_{fold}" / name,
                    ))
                copied.append(copy_exact(
                    SOURCE_RESULTS / arm / "fusion_audit_units" / task / f"fold_{fold}.json",
                    TARGET_RESULTS / arm / "fusion_audit_units" / task / f"fold_{fold}.json",
                ))
                copied.append(copy_exact(
                    SOURCE_LOGS / arm / f"finetune_seed42_{task}_fold{fold}.log",
                    TARGET_LOGS / arm / f"finetune_seed42_{task}_fold{fold}.log",
                ))
    report = {
        "schema": "mts-glt-v2-mscontact-formal-reuse-v1",
        "reused_units_per_arm": 9,
        "reused_units_total": 18,
        "copied_files": len(copied),
        "all_byte_equal": all(row["byte_equal"] for row in copied),
        "files": copied,
    }
    TARGET_RESULTS.mkdir(parents=True, exist_ok=True)
    (TARGET_RESULTS / "reuse_manifest.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({key: value for key, value in report.items() if key != "files"}, sort_keys=True))


if __name__ == "__main__":
    main()
