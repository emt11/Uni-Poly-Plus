#!/usr/bin/env python
"""Create an explicitly audited metadata-migrated MTS joint checkpoint.

The completed PI1M_v2 joint checkpoint predates the geometry-identity hash
correction.  Its tensors and cache bindings remain valid; only the source
geometry identity needs to be rewritten.  This script never mutates the
input artifact and never uses non-strict model loading.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable
EXPECTED_STAGE = "mts_joint_pretraining"
EXPECTED_DATASET = "PI1M_v2"
EXPECTED_ROUTE = "MIPS-Trimer-SCAGE"
EXPECTED_SCHEMA = "mts-model-v2"
EXPECTED_SOURCE_HASH_KEY = "source_geometry_model_config_hash"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def current_source_geometry_hash(config: Path) -> str:
    raw = subprocess.check_output(
        [PYTHON, str(ROOT / "scripts/resolve_mips_trimer_scage.py"), str(config)],
        cwd=ROOT,
        text=True,
    )
    payload = json.loads(raw)
    value = payload.get("source_geometry_model_config_hash")
    if not isinstance(value, str) or len(value) != 64:
        raise RuntimeError("resolver did not return a valid source geometry hash")
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/mts/experiments/G0_current_mcl.json",
    )
    args = parser.parse_args()
    source = args.input if args.input.is_absolute() else ROOT / args.input
    target = args.output if args.output.is_absolute() else ROOT / args.output
    config = args.config if args.config.is_absolute() else ROOT / args.config
    if not source.is_file():
        raise FileNotFoundError(source)
    if target.exists():
        raise FileExistsError(
            f"refusing to overwrite existing checkpoint: {target}; choose a new path"
        )

    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("meta"), dict):
        raise RuntimeError("input is not an MTS checkpoint with metadata")
    meta = checkpoint["meta"]
    required = {
        "schema": EXPECTED_SCHEMA,
        "stage": EXPECTED_STAGE,
        "baseline": EXPECTED_ROUTE,
        "pretraining_dataset": EXPECTED_DATASET,
        "source_cohort_hash": None,
        "feature_config_hash": None,
        "graph_model_config_hash": None,
        "trimer_cache_hash": None,
        "trimer_cache_artifact_hash": None,
        "topology_cache_hash": None,
        "topology_cache_artifact_hash": None,
        "angle_cache_schema": "mts-trimer-bond-angle-cache-v1",
        "angle_cache_artifact_hash": None,
        "pretraining_objective": "masked_atom_plus_trimer_bond_angle",
        "optimizer_steps": 20000,
        "random_seed": 42,
    }
    for key, expected in required.items():
        actual = meta.get(key)
        if expected is None:
            if not isinstance(actual, str) or not actual:
                raise RuntimeError(f"input metadata missing {key}")
        elif actual != expected:
            raise RuntimeError(
                f"input metadata {key} mismatch: expected {expected!r}, got {actual!r}"
            )
    commits = meta.get("reference_commits", {})
    if commits.get("mips") != "26aafe52926a3f33bf2d3d382ae263360319812d":
        raise RuntimeError("input MIPS reference commit is not the approved one")
    if commits.get("scage") != "82bcbb4647e31bf0d413a317e69a2526df75ce01":
        raise RuntimeError("input SCAGE reference commit is not the approved one")

    new_source_hash = current_source_geometry_hash(config)
    old_source_hash = meta.get(EXPECTED_SOURCE_HASH_KEY, meta.get("geometry_model_config_hash"))
    if not isinstance(old_source_hash, str) or len(old_source_hash) != 64:
        raise RuntimeError("input geometry identity is absent or malformed")
    if old_source_hash == new_source_hash:
        raise RuntimeError("input already has the current source geometry identity")

    migrated = dict(checkpoint)
    migrated_meta = dict(meta)
    migrated_meta[EXPECTED_SOURCE_HASH_KEY] = new_source_hash
    # The joint checkpoint is the immutable source geometry artifact.  The
    # resolved downstream experiment hash is intentionally not copied here.
    migrated_meta["geometry_model_config_hash"] = new_source_hash
    migrated_meta["checkpoint_migration"] = {
        "schema": "mts-joint-checkpoint-metadata-migration-v1",
        "from_geometry_model_config_hash": old_source_hash,
        "to_source_geometry_model_config_hash": new_source_hash,
        "reason": "corrected source geometry identity; tensors/cache unchanged",
        "input_sha256": sha256_file(source),
    }
    migrated["meta"] = migrated_meta

    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    try:
        torch.save(migrated, tmp)
        os.replace(tmp, target)
    finally:
        if tmp.exists():
            tmp.unlink()
    print(json.dumps({
        "input": str(source),
        "output": str(target),
        "input_sha256": sha256_file(source),
        "output_sha256": sha256_file(target),
        "from_geometry_model_config_hash": old_source_hash,
        "source_geometry_model_config_hash": new_source_hash,
        "state_dict_unchanged": True,
    }, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
