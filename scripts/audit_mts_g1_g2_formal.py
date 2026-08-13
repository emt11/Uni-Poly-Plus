#!/usr/bin/env python3
"""Read-only audit for the matched G1/G2 formal cycle.

The audit is intentionally separate from training.  It validates the paired
configuration contract (G1 is the accepted incumbent, G2 is the candidate that
adds endpoint distance to the shared frozen relation-geometry sidecar), then
(after the G2 pretraining arm) validates checkpoint identity and finite
tensors.  The final shard audit is delegated to the strict G1/G2 comparison
script so one identity implementation is used for both the audit and report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
PYTHON = "/opt/conda/envs/MTS/bin/python"
PAIR_ID = "mts_g_family_step0_v2_seed42"
ARMS = ("g1", "g2")
G_FAMILY_COHORTS = ("PI1M_v2", "downstream_union")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _config_contract_without_name(payload: dict[str, Any]) -> dict[str, Any]:
    """Return the scientific portion of a resolved config.

    The shared G-family step-0 artifact predates the formal experiment names.
    Its target hash therefore refers to the corresponding ``*_sidecar_readiness``
    config, while the formal run records the new experiment name.  The resolver
    includes both ``experiment_id`` and ``config_path`` in its serialized hash,
    even though neither changes the model or data contract.  Keep the identity
    check strict by accepting only an on-disk config whose resolved scientific
    payload is byte-for-byte equivalent after removing those two naming fields.
    """

    return {
        key: value
        for key, value in payload.items()
        if key not in {"config_path", "config_hash", "experiment_id"}
    }


def _find_compatible_config_hash(
    target_hash: str, resolved: dict[str, Any]
) -> dict[str, str] | None:
    """Find a local naming-only alias for a checkpoint target config hash."""

    config_root = ROOT / "configs/mts/experiments"
    arm = str(resolved.get("g_family_arm", "")).upper()
    candidates = sorted(config_root.glob(f"{arm}_*_sidecar_readiness*.json"))
    for path in candidates:
        try:
            candidate = resolve(path.resolve())
        except Exception:
            continue
        if candidate.get("config_hash") != target_hash:
            continue
        if _config_contract_without_name(candidate) != _config_contract_without_name(resolved):
            continue
        return {
            "path": str(path.resolve()),
            "experiment_id": str(candidate.get("experiment_id")),
            "config_hash": str(target_hash),
        }
    return None


def resolve(config: Path) -> dict[str, Any]:
    output = subprocess.check_output(
        [PYTHON, "scripts/resolve_mips_trimer_scage.py", str(config)],
        cwd=ROOT,
        text=True,
    )
    return json.loads(output)


def audit_configs(g1_config: Path, g2_config: Path) -> dict[str, Any]:
    g1 = resolve(g1_config.resolve())
    g2 = resolve(g2_config.resolve())
    if g1["g_family_arm"] != "g1" or g2["g_family_arm"] != "g2":
        raise RuntimeError("formal configs are not G1/G2 arms")
    for name, value in (("G1", g1), ("G2", g2)):
        if value["topology_attention_variant"] != "msta_last2":
            raise RuntimeError(f"{name}: formal config is not T1/msta_last2")
        if value["pretraining_objective"] != "masked_atom_only":
            raise RuntimeError(f"{name}: objective is not masked_atom_only")
        if float(value["angle_loss_weight"]) != 0.0:
            raise RuntimeError(f"{name}: angle loss must be zero")
        if value["shared_step0_id"] != PAIR_ID:
            raise RuntimeError(f"{name}: shared step-0 identity mismatch")
        if any(token in json.dumps(value, sort_keys=True) for token in ("<", ">")):
            raise RuntimeError(f"{name}: unresolved placeholder in resolved identity")
    common_fields = (
        "feature_config_hash", "graph_model_config_hash", "source_geometry_model_config_hash",
        "topology_representation", "topology_attention_variant", "msta_layer_indices",
        "msta_local_spd", "msta_context_spd", "modalities", "fusion_mode",
        "finetune_mode", "evaluation_protocol", "regression_loss", "huber_beta",
        "finetune_profile", "pretraining_objective", "angle_loss_weight", "shared_step0_id",
    )
    mismatches = {
        field: (g1.get(field), g2.get(field))
        for field in common_fields if g1.get(field) != g2.get(field)
    }
    if mismatches:
        raise RuntimeError(f"undeclared G1/G2 config differences: {mismatches}")
    # G1 and G2 read the same frozen relation-geometry sidecar; G2 only
    # activates the existing endpoint-distance branch.  Both cohorts must be
    # byte-identical across arms.
    for cohort in G_FAMILY_COHORTS:
        g1_bundle = g1["relation_geometry_bundle"][cohort]
        g2_bundle = g2["relation_geometry_bundle"][cohort]
        if g1_bundle.get("root") != g2_bundle.get("root"):
            raise RuntimeError(f"G1/G2 sidecar root differs for {cohort}")
        if g1_bundle.get("artifact_hash") != g2_bundle.get("artifact_hash"):
            raise RuntimeError(f"G1/G2 sidecar artifact hash differs for {cohort}")
    if g1["geometry_model_config_hash"] == g2["geometry_model_config_hash"]:
        raise RuntimeError("G1/G2 geometry model hashes must differ")
    if g1["g_family_bundle_hash"] == g2["g_family_bundle_hash"]:
        raise RuntimeError("G1/G2 bundle hashes must differ")
    return {
        "schema": "mts-g1-g2-formal-config-audit-v1",
        "cycle_id": "mts_g1_g2_matched_formal_v1",
        "status": "passed",
        "allowed_scientific_difference": [
            "experiment_id", "g_family_arm", "geometry_mode",
            "g_family_bundle_hash", "geometry_model_config_hash",
        ],
        "frozen_shared_contract": [
            "relation_geometry_bundle.PI1M_v2.root",
            "relation_geometry_bundle.PI1M_v2.artifact_hash",
            "relation_geometry_bundle.downstream_union.root",
            "relation_geometry_bundle.downstream_union.artifact_hash",
            "shared_step0_id",
        ],
        "G1": g1,
        "G2": g2,
        "protocol": {
            "optimizer_steps": 20000,
            "global_batch": 1008,
            "world_size": 3,
            "physical_gpus": [1, 2, 3],
            "seed": 42,
            "shared_step0_id": PAIR_ID,
        },
    }


def audit_step0(g1_step0: Path, g2_step0: Path) -> dict[str, Any]:
    """Verify G1/G2 share the same common model state at step 0."""

    g1_step0 = g1_step0.resolve()
    g2_step0 = g2_step0.resolve()
    g1 = torch.load(g1_step0, map_location="cpu", weights_only=False)
    g2 = torch.load(g2_step0, map_location="cpu", weights_only=False)
    g1_meta = dict(g1.get("meta") or {})
    g2_meta = dict(g2.get("meta") or {})
    if g1_meta.get("shared_step0_id") != PAIR_ID or g2_meta.get("shared_step0_id") != PAIR_ID:
        raise RuntimeError("step-0 shared_step0_id mismatch")
    if g1_meta.get("common_state_hash") != g2_meta.get("common_state_hash"):
        raise RuntimeError("G1/G2 step-0 common_state_hash differs")
    g1_state = g1.get("state_dict") or {}
    g2_state = g2.get("state_dict") or {}
    if set(g1_state) != set(g2_state):
        raise RuntimeError("G1/G2 step-0 state key sets differ")
    mismatches = []
    for key in sorted(g1_state):
        left, right = g1_state[key], g2_state[key]
        if not (torch.is_tensor(left) and torch.is_tensor(right)):
            if left != right:
                mismatches.append(key)
            continue
        if left.dtype != right.dtype or tuple(left.shape) != tuple(right.shape):
            mismatches.append(key)
        elif not torch.equal(left, right):
            mismatches.append(key)
    if mismatches:
        raise RuntimeError(f"G1/G2 step-0 common model state differs in {len(mismatches)} keys: {mismatches[:5]}")
    common_state_hash = str(g1_meta.get("common_state_hash"))
    return {
        "status": "passed",
        "g1_step0": {"path": str(g1_step0), "sha256": _sha256(g1_step0)},
        "g2_step0": {"path": str(g2_step0), "sha256": _sha256(g2_step0)},
        "common_state_hash": common_state_hash,
        "shared_step0_id": PAIR_ID,
        "common_state_tensor_count": len(g1_state),
        "common_tensors_bit_identical": True,
    }


def audit_checkpoint(path: Path, arm: str, resolved: dict[str, Any]) -> dict[str, Any]:
    path = path.resolve()
    complete = Path(str(path) + ".complete.json")
    if not path.is_file() or not complete.is_file():
        raise RuntimeError(f"{arm}: checkpoint or completion metadata is missing")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    meta = dict(payload.get("meta") or {})
    expected = {
        "g_family_arm": arm,
        "model_identity": "T1",
        "initialization": "fresh_paired",
        "paired_init_id": PAIR_ID,
        "shared_step0_id": PAIR_ID,
        "pretraining_objective": "masked_atom_only",
        "g_family_bundle_hash": resolved["g_family_bundle_hash"],
        "experiment_id": resolved["experiment_id"],
        "optimizer_steps": 20000,
        "smoke_only": False,
    }
    mismatches = {
        key: (meta.get(key), value) for key, value in expected.items()
        if meta.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"{arm}: checkpoint identity mismatch: {mismatches}")
    initialization = dict(meta.get("initialization_state") or {})
    checkpoint_config_hash = meta.get("config_hash") or initialization.get("target_config_hash")
    if checkpoint_config_hash != resolved["config_hash"]:
        alias = _find_compatible_config_hash(str(checkpoint_config_hash), resolved)
        if alias is None:
            raise RuntimeError(
                f"{arm}: checkpoint config hash mismatch: "
                f"{checkpoint_config_hash!r} != {resolved['config_hash']!r}"
            )
        config_binding = {
            "resolved_config_hash": resolved["config_hash"],
            "checkpoint_target_config_hash": checkpoint_config_hash,
            "compatible_name_only_alias": alias,
        }
    else:
        config_binding = {
            "resolved_config_hash": resolved["config_hash"],
            "checkpoint_target_config_hash": checkpoint_config_hash,
            "compatible_name_only_alias": None,
        }
    if meta.get("relation_geometry_artifact_hash") != resolved["relation_geometry_bundle"]["PI1M_v2"]["artifact_hash"]:
        raise RuntimeError(f"{arm}: PI1M_v2 relation artifact mismatch")
    if not payload.get("state_dict") or any(
        not torch.isfinite(value).all().item()
        for value in payload["state_dict"].values()
        if torch.is_tensor(value) and (value.is_floating_point() or value.is_complex())
    ):
        raise RuntimeError(f"{arm}: checkpoint contains non-finite tensor")
    complete_meta = json.loads(complete.read_text(encoding="utf-8"))
    if complete_meta.get("optimizer_steps") != 20000 or complete_meta.get("checkpoint_sha256") != _sha256(path):
        raise RuntimeError(f"{arm}: completion metadata mismatch")
    return {
        "arm": arm,
        "path": str(path),
        "sha256": _sha256(path),
        "meta": meta,
        "completion": complete_meta,
        "config_binding": config_binding,
        "state_tensor_count": len(payload["state_dict"]),
        "status": "passed",
    }


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("config", "step0", "checkpoint"), required=True)
    parser.add_argument("--g1-config", type=Path, default=ROOT / "configs/mts/experiments/G1_t1_msta_angle_matched_formal_v1.json")
    parser.add_argument("--g2-config", type=Path, default=ROOT / "configs/mts/experiments/G2_t1_msta_cosine_distance_matched_formal_v1.json")
    parser.add_argument("--g1-step0", type=Path, default=ROOT / "pretrained_models/mts_multiscale_topology/g_family_step0_full_v2/g1_step0.pth")
    parser.add_argument("--g2-step0", type=Path, default=ROOT / "pretrained_models/mts_multiscale_topology/g_family_step0_full_v2/g2_step0.pth")
    parser.add_argument("--g1-checkpoint", type=Path)
    parser.add_argument("--g2-checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = audit_configs(args.g1_config, args.g2_config)
    if args.phase == "step0":
        report["step0"] = audit_step0(args.g1_step0, args.g2_step0)
    if args.phase == "checkpoint":
        if args.g1_checkpoint is None or args.g2_checkpoint is None:
            parser.error("checkpoint phase requires --g1-checkpoint and --g2-checkpoint")
        report["step0"] = audit_step0(args.g1_step0, args.g2_step0)
        report["checkpoints"] = {
            "G1": audit_checkpoint(args.g1_checkpoint, "g1", report["G1"]),
            "G2": audit_checkpoint(args.g2_checkpoint, "g2", report["G2"]),
        }
    _write(args.output.resolve(), report)
    print(json.dumps({"status": report["status"], "output": str(args.output.resolve())}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
