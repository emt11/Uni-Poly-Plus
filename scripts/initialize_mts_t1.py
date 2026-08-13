#!/usr/bin/env python
"""Explicit, function-preserving T0 -> T1 MSTA initialization.

This is intentionally separate from ordinary pretraining resume.  It copies
the T0 state, adds zero local projections to the final two attention layers,
strictly loads the converted graph state into a T1 encoder, and writes a new
``*_init.pth`` artifact without inheriting optimizer/sampler state.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path

import torch

from src.modules.mips_local_graph import (
    MIPSLocalGraphEncoder,
    add_function_preserving_t1_parameters,
    checkpoint_topology_attention_variant,
    topology_attention_identity,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolved_target_identity(config: Path) -> dict:
    resolver = PROJECT_ROOT / "scripts/resolve_mips_trimer_scage.py"
    payload = subprocess.check_output(
        [os.environ.get("PYTHON", os.sys.executable), str(resolver), str(config)],
        cwd=PROJECT_ROOT,
        text=True,
    )
    resolved = json.loads(payload)
    if resolved.get("topology_attention_variant") != "msta_last2":
        raise RuntimeError("T1 initializer requires a msta_last2 target config")
    return resolved


def initialize(source: Path, output: Path, target_config: Path) -> dict:
    if output.exists():
        raise FileExistsError(
            f"refusing to overwrite an existing T1 initialization artifact: {output}"
        )
    payload = torch.load(source, map_location="cpu")
    if not isinstance(payload, dict) or not isinstance(payload.get("state_dict"), dict):
        raise RuntimeError("source checkpoint must contain a state_dict mapping")
    source_meta = dict(payload.get("meta") or {})
    source_variant = checkpoint_topology_attention_variant(source_meta)
    if source_variant != "o8":
        raise RuntimeError(
            "explicit T0 -> T1 initialization accepts only a T0/O8 source; "
            f"received {source_variant!r}"
        )
    resolved = _resolved_target_identity(target_config)
    identity = topology_attention_identity("msta_last2")
    state = add_function_preserving_t1_parameters(
        payload["state_dict"], prefix="encoders.graph.encoder."
    )
    graph_state = {
        key[len("encoders.graph.encoder."):]: value
        for key, value in state.items()
        if key.startswith("encoders.graph.encoder.")
    }
    target_model = MIPSLocalGraphEncoder(
        topology_attention_variant="msta_last2",
    )
    expected_graph_state = target_model.state_dict()
    if set(graph_state) != set(expected_graph_state):
        missing = sorted(set(expected_graph_state) - set(graph_state))
        unexpected = sorted(set(graph_state) - set(expected_graph_state))
        raise RuntimeError(
            "T0 -> T1 strict conversion has an architecture mismatch: "
            f"missing={missing[:8]}, unexpected={unexpected[:8]}"
        )
    shape_mismatch = sorted(
        key for key in graph_state
        if tuple(graph_state[key].shape) != tuple(expected_graph_state[key].shape)
    )
    if shape_mismatch:
        raise RuntimeError(
            "T0 -> T1 strict conversion has shape mismatches: "
            + ", ".join(shape_mismatch[:8])
        )
    target_model.load_state_dict(graph_state, strict=True)

    meta = dict(source_meta)
    meta.update(identity)
    meta.update({
        "model_identity": "T1",
        "source_model_identity": "T0",
        "topology_attention_variant": "msta_last2",
        "graph_model_config_hash": resolved["graph_model_config_hash"],
        "source_graph_model_config_hash": source_meta.get(
            "graph_model_config_hash"
        ),
        "parent_checkpoint": str(source.resolve()),
        "parent_checkpoint_sha256": _sha256(source),
        "initialization": "function_preserving",
        "init_artifact": True,
        "optimizer_state_inherited": False,
        "scheduler_state_inherited": False,
        "sampler_state_inherited": False,
        "source_optimizer_steps": int(source_meta.get("optimizer_steps", 0)),
        "optimizer_steps": 0,
        "pretraining_status": "untrained_initialization",
        "target_config_path": str(target_config.resolve()),
        "target_config_hash": resolved["config_hash"],
    })
    converted = {"state_dict": state, "meta": meta}
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=output.parent, prefix=output.name + ".tmp-", delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(converted, temporary)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    manifest = {
        "schema": "mts-t1-initialization-v1",
        "source_checkpoint": str(source.resolve()),
        "source_checkpoint_sha256": meta["parent_checkpoint_sha256"],
        "output_checkpoint": str(output.resolve()),
        "output_checkpoint_sha256": _sha256(output),
        "initialization": "function_preserving",
        "source_model_identity": "T0",
        "model_identity": "T1",
        "target_config": str(target_config.resolve()),
        "target_graph_model_config_hash": resolved["graph_model_config_hash"],
        "optimizer_state_inherited": False,
        "strict_graph_load": True,
    }
    manifest_path = output.with_suffix(output.suffix + ".initialization.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-checkpoint", required=True, type=Path)
    parser.add_argument("--output-checkpoint", required=True, type=Path)
    parser.add_argument(
        "--target-config",
        type=Path,
        default=PROJECT_ROOT / "configs/mts/experiments/T1_msta_readiness.json",
    )
    args = parser.parse_args()
    result = initialize(
        args.source_checkpoint.resolve(),
        args.output_checkpoint.resolve(),
        args.target_config.resolve(),
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
