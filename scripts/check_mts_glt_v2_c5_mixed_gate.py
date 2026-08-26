#!/usr/bin/env python3
"""Pretraining gate for the matched C5-Mixed versus MS45 comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.check_mts_glt_v2_mscontact_init import make  # noqa: E402


def relation_gate(sidecar_root: Path):
    shell = np.load(sidecar_root / "pair_shell_id.npy", mmap_mode="r")
    valid = np.load(sidecar_root / "pair_valid.npy", mmap_mode="r")
    atom_a = np.load(sidecar_root / "pair_atom_a.npy", mmap_mode="r")
    atom_b = np.load(sidecar_root / "pair_atom_b.npy", mmap_mode="r")
    shift = np.load(sidecar_root / "pair_shift.npy", mmap_mode="r")
    if not (len(shell) == len(valid) == len(atom_a) == len(atom_b) == len(shift)):
        raise RuntimeError(f"spatial sidecar relation arrays disagree: {sidecar_root}")
    mismatch = 0
    compared = 0
    chunk = 1_000_000
    for start in range(0, len(valid), chunk):
        end = min(len(valid), start + chunk)
        valid_chunk = np.asarray(valid[start:end], dtype=bool)
        shell_chunk = np.asarray(shell[start:end])
        ms45_active = valid_chunk & ((shell_chunk == 0) | (shell_chunk == 1))
        c5_active = valid_chunk
        mismatch += int(np.count_nonzero(ms45_active ^ c5_active))
        compared += int(np.count_nonzero(ms45_active & c5_active))
        # Both modes consume the same canonical key rows.  Touch all key
        # arrays in bounded chunks so malformed lengths cannot hide here.
        _ = (
            np.asarray(atom_a[start:end])[ms45_active],
            np.asarray(atom_b[start:end])[ms45_active],
            np.asarray(shift[start:end])[ms45_active],
        )
    return {
        "sidecar": str(sidecar_root.resolve()),
        "relation_rows": int(len(valid)),
        "compared_relation_keys": compared,
        "relation_key_mismatch": mismatch,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ms45 = make("ms45", args.seed)
    c5 = make("c5_mixed", args.seed)
    ms45_state, c5_state = ms45.state_dict(), c5.state_dict()
    shared = sorted(set(ms45_state) & set(c5_state))
    unequal = [name for name in shared if not torch.equal(ms45_state[name], c5_state[name])]
    relation_reports = {
        name: relation_gate(ROOT / path)
        for name, path in {
            "pretrain": "data/processed/mips_trimer_scage/spatial_contact_v1/PI1M_v2",
            "downstream": "data/processed/mips_trimer_scage/spatial_contact_v1/downstream_union",
        }.items()
    }
    report = {
        "schema": "mts-glt-v2-c5-mixed-gate-v1",
        "seed": int(args.seed),
        "ms45_parameter_count": int(sum(p.numel() for p in ms45.parameters())),
        "c5_mixed_parameter_count": int(sum(p.numel() for p in c5.parameters())),
        "parameter_count_delta": int(
            sum(p.numel() for p in c5.parameters())
            - sum(p.numel() for p in ms45.parameters())
        ),
        "shared_state_tensors": len(shared),
        "fresh_state_mismatches": unequal,
        "matched_initialization_basis": "same seed and fresh full-state tensor equality",
        "relation_universe": relation_reports,
    }
    report["pass"] = bool(
        report["parameter_count_delta"] == 0
        and not unequal
        and all(value["relation_key_mismatch"] == 0 for value in relation_reports.values())
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))
    if not report["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
