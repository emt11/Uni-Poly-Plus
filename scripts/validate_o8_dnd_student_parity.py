#!/usr/bin/env python3
"""Compare the DND student's O8/mask/fingerprint inputs with Arm-B inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.dataset.glt_o8_control import build_o8_pretrain_sample
from src.dataset.o8_dnd_student import build_student_pretrain_sample
from src.training.glt_dual_runtime import OrderedSampleStream, open_source, require_tmux, write_json


def _equal(left, right):
    if torch.is_tensor(left) or torch.is_tensor(right):
        return torch.equal(torch.as_tensor(left), torch.as_tensor(right))
    return left == right


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("cohort-root", "cache-root", "dual-static-root", "pretrain-target-root", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--count", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    require_tmux()
    if args.count <= 0:
        raise ValueError("--count must be positive")
    source, _ = open_source(
        args.cohort_root, args.cache_root,
        dual_static_root=args.dual_static_root,
        pretrain_target_root=args.pretrain_target_root,
    )
    stream = OrderedSampleStream(len(source), args.seed)
    checks = []
    try:
        for position in range(int(args.count)):
            index = stream.index_at(position)
            topology, trimer, smiles = source[index]
            key = source.samples[index][0].hex()
            static, target = source.static_for(index), source.target_for(index)
            arm_data, arm_labels = build_o8_pretrain_sample(
                topology, static, target, seed=args.seed, key=key,
                position=position, ratio=0.30,
            )
            dnd_data, dnd_labels, _, _ = build_student_pretrain_sample(
                topology, trimer, smiles, seed=args.seed, key=key,
                position=position, ratio=0.30, static=static, target=target,
            )
            exact_fields = all(_equal(getattr(arm_data, name), getattr(dnd_data, name))
                               for name in ("mips_x", "mips_backbone_mask", "lga_edge_index",
                                            "lga_spd", "lga_path_index", "lga_path_mask",
                                            "bond_path_features", "bond_path_mask"))
            exact_labels = all(_equal(arm_labels[name], dnd_labels[name])
                               for name in ("atom_mask", "atom_label", "fingerprint", "fallback"))
            checks.append({"position": position, "index": int(index), "key": key,
                           "graph_tensors_exact": bool(exact_fields),
                           "mask_label_fingerprint_exact": bool(exact_labels)})
            if not exact_fields or not exact_labels:
                raise AssertionError(f"O8 parity mismatch at position {position}")
    finally:
        source.close()
    report = {
        "schema": "eq3d-dnd-o8-student-parity-v1", "count": len(checks),
        "seed": int(args.seed), "positions": checks,
        "status": "PASS" if len(checks) == int(args.count) else "FAIL",
    }
    write_json(args.output, report)
    print(json.dumps({"status": report["status"], "count": len(checks),
                      "output": str(Path(args.output).resolve())}, ensure_ascii=False), flush=True)
    if report["status"] != "PASS":
        raise RuntimeError("DND student O8 parity failed")


if __name__ == "__main__":
    main()
