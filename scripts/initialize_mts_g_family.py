#!/usr/bin/env python3
"""Create an isolated, shared step-0 identity for the T1 G-family.

This command only instantiates models and writes new readiness artifacts.  It
does not read or modify a training state, optimizer, frozen cache, or existing
checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from src.modules.mips_local_graph import MIPSLocalGraphEncoder, topology_attention_identity


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "pretrained_models/mts_multiscale_topology/g_family_step0_v1"


def _state_hash(state):
    digest = hashlib.sha256()
    for key in sorted(state):
        digest.update(key.encode("utf-8"))
        value = state[key].detach().cpu().contiguous()
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"refusing to overwrite existing G-family step0 directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(int(args.seed))
    reference = MIPSLocalGraphEncoder(
        topology_attention_variant="msta_last2",
        graph_geometry_mode="g0",
        g_family_arm="g0",
        use_star_rbf=False,
        use_mcl=False,
    )
    common_state = {key: value.detach().cpu().clone() for key, value in reference.state_dict().items()}
    common_hash = _state_hash(common_state)
    arms = []
    for arm in ("g0", "g1", "g2", "g3"):
        # Strict loading proves all four arms have one identical parameter
        # layout; geometry inputs differ only at runtime, not in state shape.
        model = MIPSLocalGraphEncoder(
            topology_attention_variant="msta_last2",
            graph_geometry_mode=arm,
            g_family_arm=arm,
            use_star_rbf=False,
            use_mcl=False,
        )
        model.load_state_dict(common_state, strict=True)
        state_path = output / f"{arm}_step0.pth"
        torch.save(
            {
                "state_dict": common_state,
                "metadata": {
                    "schema": "mts-g-family-step0-v1",
                    "model_identity": "T1",
                    "topology_attention": topology_attention_identity("msta_last2"),
                    "g_family_arm": arm,
                    "geometry_mode": arm,
                    "use_star_rbf": False,
                    "use_mcl": False,
                    "pretraining_objective": "masked_atom_only",
                    "angle_loss_weight": 0.0,
                    "shared_step0_id": "mts_g_family_step0_v1_seed42",
                    "seed": int(args.seed),
                    "common_state_hash": common_hash,
                },
            },
            state_path,
        )
        arms.append({"arm": arm, "path": str(state_path), "sha256": hashlib.sha256(state_path.read_bytes()).hexdigest()})
    metadata = {
        "schema": "mts-g-family-step0-v1",
        "shared_step0_id": "mts_g_family_step0_v1_seed42",
        "seed": int(args.seed),
        "topology": topology_attention_identity("msta_last2"),
        "arms": arms,
        "common_state_hash": common_hash,
        "optimizer_inherited": False,
        "scheduler_inherited": False,
        "sampler_inherited": False,
        "rng_inherited": False,
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
