#!/usr/bin/env python3
"""Verify runtime static reuse and the clean/noisy contract on real PI1M samples.

Two properties must hold exactly:

1. sharing the pure-function static parts (identity mapping, bond path
   features) between the clean and noisy views produces bit-identical tensors
   to two independent builds;
2. clean coordinates produce the clean length/angle targets while the noisy
   clone is what the encoder sees -- cached clean distances/angles must never
   reach the noisy encoder.

Read-only: the frozen bundle is opened with lock=False and nothing is written.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.dataset.canonical_periodic import resolve_normalized_identity
from src.dataset.glt_dual import bond_paths, build_dual_sample
from src.dataset.glt_dual_pretrain import (
    chemical_targets,
    motif_mask,
    prepare_pretrain_sample,
    sample_generator,
)
from src.dataset.cache_lifecycle import zero_write_snapshot
from src.training.glt_dual_runtime import open_source, write_json

STATIC_FIELDS = (
    "mips_x", "mips_backbone_mask", "bond_path_features", "bond_path_mask",
    "bond_z_a", "bond_z_b", "bond_type", "bond_center", "line_source",
    "line_target", "line_path", "line_path_group", "line_mask", "line_is_self",
)
GEOMETRY_FIELDS = ("bond_distance", "line_angle")
SIGMA, RATIO, SEED = 0.03, 0.3, 20260915


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--cohort-root", required=True)
    parser.add_argument("--samples", type=int, default=40)
    parser.add_argument("--report-json", required=True)
    args = parser.parse_args()
    cache_root = Path(args.cache_root).resolve()
    before = zero_write_snapshot(cache_root)
    source, _ = open_source(args.cohort_root, cache_root)
    failures, checked, geometry_pairs = [], 0, 0
    try:
        total = len(source)
        indices = [i * total // args.samples for i in range(args.samples)]
        for order, index in enumerate(indices):
            topology, trimer, smiles = source[index]
            key = source.samples[index][0].hex()
            identity = resolve_normalized_identity(
                topology, smiles, require_fields=True
            )
            static = bond_paths(topology, smiles, identity=identity)
            shared = build_dual_sample(
                topology, trimer, smiles, identity=identity,
                bond_path_features=static,
            )
            independent = build_dual_sample(topology, trimer, smiles)
            for name in STATIC_FIELDS + GEOMETRY_FIELDS:
                left = torch.as_tensor(getattr(shared, name))
                right = torch.as_tensor(getattr(independent, name))
                if left.shape != right.shape or not torch.equal(left, right):
                    failures.append(f"{key}:reuse:{name}")
            if shared.bond_path_features.data_ptr() == static[0].data_ptr():
                failures.append(f"{key}:reuse:aliased_static_tensor")

            noisy, targets = prepare_pretrain_sample(
                topology, trimer, smiles, seed=SEED, key=key,
                position=order, sigma=SIGMA, ratio=RATIO,
            )
            clean = build_dual_sample(topology, trimer, smiles)
            if not bool(clean.geometry_valid):
                failures.append(f"{key}:unexpected_invalid_geometry")
                continue
            generator = sample_generator(SEED, key, order)
            # Replicate the documented draw order: the motif mask consumes the
            # sample generator before the coordinate noise is drawn.
            motif_mask(
                topology,
                chemical_targets(identity["normalized_smiles"])[0],
                generator,
                RATIO,
            )
            changed = copy.copy(trimer)
            changed.trimer_pos = trimer.trimer_pos.float().clone()
            changed.trimer_pos += SIGMA * torch.randn(
                changed.trimer_pos.shape, generator=generator
            )
            expected = build_dual_sample(topology, changed, smiles)
            for name in STATIC_FIELDS + GEOMETRY_FIELDS:
                left = torch.as_tensor(getattr(noisy, name))
                right = torch.as_tensor(getattr(expected, name))
                if left.shape != right.shape or not torch.equal(left, right):
                    failures.append(f"{key}:noise_stream:{name}")
            for name in STATIC_FIELDS:
                if not torch.equal(
                    torch.as_tensor(getattr(noisy, name)),
                    torch.as_tensor(getattr(clean, name)),
                ):
                    failures.append(f"{key}:static_view_differs:{name}")
            if torch.equal(noisy.bond_distance, clean.bond_distance):
                failures.append(f"{key}:clean_geometry_leaked_into_encoder")
            geometry_pairs += 1
            expected_targets = clean.bond_distance[clean.bond_center].clone()
            if not torch.equal(targets["distance"], expected_targets):
                failures.append(f"{key}:targets_not_from_clean_coordinates")
            cosines = []
            for row in range(clean.line_path.size(0)):
                if int(clean.line_mask[row].sum()) != 1 or bool(
                    clean.line_is_self[row]
                ):
                    continue
                a, b = clean.line_path[row, :2].tolist()
                if a < b and clean.bond_center[a] and clean.bond_center[b]:
                    cosines.append(clean.line_angle[row, 0].cos())
            if cosines and not torch.equal(
                targets["angle_cos"], torch.stack(cosines)
            ):
                failures.append(f"{key}:angle_targets_not_from_clean_coordinates")
            checked += 1
    finally:
        source.close()
    zero_write = zero_write_snapshot(cache_root) == before
    if not zero_write:
        failures.append("frozen cache was modified")
    report = {
        "status": "PASS" if not failures else "FAIL",
        "scope": "real PI1M samples: static reuse equivalence + clean/noisy contract",
        "main_bundle_hash": None,
        "cohort_root": str(Path(args.cohort_root).resolve()),
        "samples_checked": checked,
        "clean_noisy_geometry_pairs": geometry_pairs,
        "sigma": SIGMA, "mask_ratio": RATIO, "seed": SEED,
        "cache_zero_write": zero_write,
        "failures": failures[:10],
    }
    write_json(args.report_json, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
