#!/usr/bin/env python3
"""Create full-model, shared step-0 states for the T1 G-family.

The earlier graph-only readiness artifact is retained as an audit artifact.
This initializer writes a new, consumable ``mts-pretrain-init-v1`` payload for
each arm, with the complete ``UniEncoderAttention`` state expected by
``pretrain.py``.  No optimizer, scheduler, sampler, RNG, trained checkpoint,
or frozen cache is read or inherited.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.initialize_mts_t_pretrain0 import build_model  # noqa: E402
from src.modules.mips_local_graph import topology_attention_identity  # noqa: E402
from src.utils import set_global_seed  # noqa: E402


PAIR_ID = "mts_g_family_step0_v2_seed42"
ARMS = ("g0", "g1", "g2", "g3")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _state_hash(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        value = state[key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _resolve_config(path: Path) -> dict[str, Any]:
    resolver = ROOT / "scripts/resolve_mips_trimer_scage.py"
    python = os.environ.get("MTS_PYTHON", "/opt/conda/envs/MTS/bin/python")
    output = subprocess.check_output(
        [python, str(resolver), str(path)], cwd=ROOT, text=True
    )
    return json.loads(output)


def _save(path: Path, state: dict[str, torch.Tensor], meta: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite step-0 artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkstemp(prefix=path.name + ".tmp-", dir=path.parent)[1])
    try:
        torch.save({"schema": "mts-pretrain-init-v1", "state_dict": state, "meta": meta}, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-root",
        type=Path,
        default=ROOT / "configs/mts/experiments",
        help="Directory containing the four G-family readiness configs.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "pretrained_models/mts_multiscale_topology/g_family_step0_full_v2",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    output_root = args.output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty output root: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    configs = {
        arm: (args.config_root / f"{arm.upper()}_t1_msta_sidecar_readiness_v1.json").resolve()
        for arm in ARMS
    }
    resolved = {arm: _resolve_config(path) for arm, path in configs.items()}
    for arm, payload in resolved.items():
        if payload.get("topology_attention_variant") != "msta_last2":
            raise RuntimeError(f"{arm}: G-family step0 requires T1/msta_last2")
        if payload.get("g_family_arm") != arm:
            raise RuntimeError(f"{arm}: config g_family_arm mismatch")
        if payload.get("pretraining_objective") != "masked_atom_only":
            raise RuntimeError(f"{arm}: G-family objective must be masked_atom_only")
        if float(payload.get("angle_loss_weight", 1.0)) != 0.0:
            raise RuntimeError(f"{arm}: G-family angle_loss_weight must be zero")

    set_global_seed(int(args.seed))
    reference = build_model(
        "msta_last2",
        graph_geometry_mode="g0",
        g_family_arm="g0",
        use_star_rbf=False,
        use_mcl=False,
    )
    common_state = {
        key: value.detach().cpu().clone() for key, value in reference.state_dict().items()
    }
    common_hash = _state_hash(common_state)
    records = []
    for arm in ARMS:
        model = build_model(
            "msta_last2",
            graph_geometry_mode=arm,
            g_family_arm=arm,
            use_star_rbf=False,
            use_mcl=False,
        )
        model.load_state_dict(common_state, strict=True)
        state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if _state_hash(state) != common_hash:
            raise RuntimeError(f"{arm}: shared step-0 state hash mismatch")
        meta = {
            "schema": "mts-pretrain-init-v1",
            "initialization": "fresh_paired",
            "paired_init_id": PAIR_ID,
            "model_identity": "T1",
            "topology_attention_variant": "msta_last2",
            "topology_attention": topology_attention_identity("msta_last2"),
            "g_family_arm": arm,
            "geometry_mode": arm,
            "pretraining_objective": "masked_atom_only",
            "angle_loss_weight": 0.0,
            "shared_step0_id": PAIR_ID,
            "g_family_bundle_hash": resolved[arm]["g_family_bundle_hash"],
            "optimizer_steps": 0,
            "optimizer_state_inherited": False,
            "scheduler_state_inherited": False,
            "sampler_state_inherited": False,
            "rng_inherited": False,
            "random_seed": int(args.seed),
            "common_state_hash": common_hash,
            "target_config_path": str(configs[arm]),
            "target_config_hash": resolved[arm]["config_hash"],
            "target_graph_model_config_hash": resolved[arm]["graph_model_config_hash"],
            "relation_geometry_bundle": resolved[arm].get("relation_geometry_bundle"),
            "g3_permutation_bundle": resolved[arm].get("g3_permutation_bundle"),
            "parent_checkpoint": None,
        }
        path = output_root / f"{arm}_step0.pth"
        _save(path, state, meta)
        records.append({
            "arm": arm,
            "path": str(path),
            "sha256": _sha256(path),
            "state_hash": common_hash,
            "config_hash": resolved[arm]["config_hash"],
        })

    manifest = {
        "schema": "mts-g-family-full-step0-v2",
        "shared_step0_id": PAIR_ID,
        "seed": int(args.seed),
        "topology_attention": topology_attention_identity("msta_last2"),
        "common_state_hash": common_hash,
        "g_family_bundle_hashes": {
            arm: resolved[arm]["g_family_bundle_hash"] for arm in ARMS
        },
        "arms": records,
        "optimizer_inherited": False,
        "scheduler_inherited": False,
        "sampler_inherited": False,
        "rng_inherited": False,
        "source_checkpoint": None,
    }
    (output_root / "metadata.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
