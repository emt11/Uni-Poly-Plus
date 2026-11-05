#!/usr/bin/env python3
"""One CPU structure prepare. Labels are never fitted or cached."""
import json
import os
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from src.dataset.ua3d_ph import SCHEMA, lifted_graph, random_walk_pe
OUT = ROOT / "results/ua3d_ph_20261028/r1/inputs"
SOURCE = ROOT / "results/ua3m_paperfuse_20261025/r1/repairs/gpu0123_restart1/inputs"

def prepare():
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("prepare must hide GPUs")
    if OUT.exists():
        raise FileExistsError("PH prepare exists; no rerun")
    OUT.mkdir(parents=True)
    rows = json.loads((SOURCE / "rows.json").read_text())
    count = sum(len(v) for v in rows.values())
    if count != 5875:
        raise RuntimeError("unexpected source rows")
    metadata = {
        "schema": SCHEMA,
        "rows": count,
        "status": "NOT_MATERIALIZED",
        "labels_used": False,
        "model_forward": 0,
        "reason": "full VR/graph materialization is implemented in src/dataset/ua3d_ph.py and is launched only by the PH controller after the STABILITY gate",
    }
    (OUT / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (OUT / "coverage.json").write_text(json.dumps({"rows": count, "materialized": 0}, indent=2) + "\n")
    return metadata

if __name__ == "__main__":
    print(json.dumps(prepare()))
