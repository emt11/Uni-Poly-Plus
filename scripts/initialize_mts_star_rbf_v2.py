#!/usr/bin/env python3
"""Strictly convert the frozen G1 seed-42 step-0 into the R2 identity."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.initialize_mts_t_pretrain0 import build_model


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""): h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--parent", type=Path, default=ROOT / "pretrained_models/mts_multiscale_topology/g_family_step0_full_v2/g1_step0.pth")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    resolved = json.loads(subprocess.check_output([
        "/opt/conda/envs/MTS/bin/python", "scripts/resolve_mips_trimer_scage.py", str(args.config)
    ], cwd=ROOT, text=True))
    if resolved["star_rbf_definition"] != "trimer_periodic_relation_rbf_v2":
        raise RuntimeError("target config is not Star-RBF v2")
    parent = torch.load(args.parent, map_location="cpu", weights_only=False)
    meta = parent.get("meta", {})
    if meta.get("g_family_arm") != "g1" or meta.get("shared_step0_id") != "mts_g_family_step0_v2_seed42":
        raise RuntimeError("parent is not the frozen G1 seed-42 step-0")
    model = build_model(
        "msta_last2", graph_geometry_mode="g1", g_family_arm="g1",
        use_star_rbf=True, use_mcl=False,
        star_rbf_definition="trimer_periodic_relation_rbf_v2",
        star_rbf_upper=float(resolved["star_rbf_v2_upper"]),
    )
    model.load_state_dict(parent["state_dict"], strict=True)
    # The legacy checkpoint contains the non-parameter RBF center buffer for
    # [0, 3].  Strict loading proves the complete architecture/state layout;
    # then replace only that definition-bound buffer with the frozen v2 range.
    star = model.encoders["graph"].encoder.star_distance_bias
    star.centers.copy_(torch.linspace(0.0, float(resolved["star_rbf_v2_upper"]), star.centers.numel()))
    projection = model.state_dict()["encoders.graph.encoder.star_distance_bias.projection.weight"]
    if torch.count_nonzero(projection).item() != 0:
        raise RuntimeError("R2 step-0 Star-RBF projection must remain zero")
    output_meta = dict(meta)
    output_meta.update({
        "paired_init_id": "mts_star_rbf_v2_r2_step0_seed42",
        "shared_step0_id": "mts_star_rbf_v2_r2_step0_seed42",
        "g_family_bundle_hash": resolved["g_family_bundle_hash"],
        "target_config_path": str(args.config.resolve()),
        "target_config_hash": resolved["config_hash"],
        "target_graph_model_config_hash": resolved["graph_model_config_hash"],
        "target_geometry_model_config_hash": resolved["geometry_model_config_hash"],
        "star_rbf_definition": resolved["star_rbf_definition"],
        "star_rbf_v2_model_semantic_hash": resolved["star_rbf_v2_model_semantic_hash"],
        "star_rbf_v2_upper": resolved["star_rbf_v2_upper"],
        "star_rbf_v2_bundle": resolved["star_rbf_v2_bundle"],
        "backbone_definition": "legacy_g1_frozen",
        "parent_checkpoint": str(args.parent.resolve()),
        "parent_checkpoint_sha256": sha(args.parent),
        "optimizer_steps": 0,
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists(): raise FileExistsError(args.output)
    tmp = Path(tempfile.mkstemp(prefix=args.output.name + ".tmp-", dir=args.output.parent)[1])
    try:
        torch.save({"schema": "mts-pretrain-init-v1", "state_dict": model.state_dict(), "meta": output_meta}, tmp)
        os.replace(tmp, args.output)
    finally: tmp.unlink(missing_ok=True)
    print(json.dumps({"path": str(args.output), "sha256": sha(args.output), "meta": output_meta}, indent=2))


if __name__ == "__main__": main()
