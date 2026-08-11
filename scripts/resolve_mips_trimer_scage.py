#!/usr/bin/env python
"""Validate the one immutable MIPS-Trimer-SCAGE (MTS) production config."""

import argparse
import hashlib
import json
import shlex
from pathlib import Path

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset.mips_trimer_contract import (
    ABLATION_IDS,
    ABLATION_RANDOM_MASK_PAYLOAD_VERSION,
    ABLATION_RANDOM_MASK_SCHEMA,
    ABLATION_RANDOM_MASK_SEED,
    CACHE_BUNDLE_SCHEMA,
    CACHE_LAYOUT_SCHEMA,
    CANONICAL_LGA_SCHEMA_VERSION,
    CHECKPOINT_SCHEMA,
    CONFIG_SCHEMA,
    EXPERIMENT_CONFIG_SCHEMA,
    FEATURE_SCHEMA,
    EXPLICIT_FEATURE_SCHEMA,
    EXPLICIT_LGA_SCHEMA_VERSION,
    EXPLICIT_TOPOLOGY_LMDB_SCHEMA,
    TOPOLOGY_CANONICAL,
    TOPOLOGY_EXPLICIT,
    TOPOLOGY_REPRESENTATIONS,
    TOPOLOGY_LMDB_SCHEMA,
    TRIMER_ACCEPTANCE,
    TRIMER_BUILDER_VERSION,
    TRIMER_CONTENT_SCHEMA,
    TRIMER_LMDB_SCHEMA,
    TRIMER_MMFF_RELAX_MAX_ITERATIONS,
    TRIMER_PROTOCOL,
    TRIMER_REQUIRE_MMFF_CONVERGENCE,
    TRIMER_SELECTION,
    ROUTE_NAME,
    ROUTE_SHORT_NAME,
    MTS_SHARED_CHECKPOINT,
    MTS_SHARED_CHECKPOINT_SHA256,
)


DEFAULT = (
    Path(__file__).resolve().parents[1]
    / "configs/mts/default.json"
)


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _sha256_file(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _random_mask_sidecar_path():
    """Return the deterministic v2 sidecar root bound to current artifacts."""
    cohort_parent = PROJECT_ROOT / "data/processed/mips_trimer_scage/cohorts/downstream_union"
    cohort_dirs = sorted(
        path for path in cohort_parent.iterdir()
        if path.is_dir() and (path / "manifest.json").is_file()
    ) if cohort_parent.is_dir() else []
    if len(cohort_dirs) != 1:
        raise ValueError("expected exactly one downstream_union cohort manifest")
    cohort_dir = cohort_dirs[0]
    manifest = json.loads((cohort_dir / "manifest.json").read_text(encoding="utf-8"))
    cohort_hash = str(manifest["cohort_hash"])
    threshold_path = cohort_dir / "mcl_thresholds.npy"
    threshold_meta_path = cohort_dir / "mcl_thresholds_metadata.json"
    threshold_meta = json.loads(threshold_meta_path.read_text(encoding="utf-8"))
    trimer_artifact_hash = str(threshold_meta["trimer_artifact_hash"])
    trimer_parent = PROJECT_ROOT / "data/processed/mips_trimer_scage/trimer"
    trimer_done = [
        path for path in trimer_parent.glob("*/.done")
        if path.read_text(encoding="utf-8").strip() == trimer_artifact_hash
    ]
    if len(trimer_done) != 1:
        raise ValueError("cannot resolve the unique frozen Trimer artifact")
    identity = {
        "schema": ABLATION_RANDOM_MASK_SCHEMA,
        "payload_version": ABLATION_RANDOM_MASK_PAYLOAD_VERSION,
        "seed": ABLATION_RANDOM_MASK_SEED,
        "cohort_hash": cohort_hash,
        "trimer_artifact_hash": trimer_artifact_hash,
        "threshold_file_sha256": _sha256_file(threshold_path),
    }
    root_hash = digest(identity)
    return (
        PROJECT_ROOT / "data/processed/mips_trimer_scage/ablation_random_mask"
        / "downstream_union" / root_hash
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", nargs="?", default=str(DEFAULT))
    parser.add_argument("--shell", action="store_true")
    args = parser.parse_args()
    path = Path(args.config).resolve()
    config = json.loads(path.read_text(encoding="utf-8"))
    reference = json.loads(DEFAULT.read_text(encoding="utf-8"))
    experiment = config.get("schema_version") == EXPERIMENT_CONFIG_SCHEMA
    experiment_values = None
    if experiment:
        allowed = {
            "schema_version", "experiment_id", "parent_config",
            "geometry_mode", "modalities", "fusion_mode", "finetune_mode",
            "evaluation_protocol", "regression_loss", "huber_beta",
            "finetune_profile",
            "patience", "warmup_epochs", "weight_decay", "swa_start_epoch",
            "legacy_graph_trainable_from_epoch0",
            "modality_control", "controlled_modality",
            "topology_representation",
            "ablation",
        }
        unknown = sorted(set(config) - allowed)
        missing = sorted({"schema_version", "experiment_id", "parent_config"} - set(config))
        if unknown or missing:
            raise ValueError(
                f"invalid {EXPERIMENT_CONFIG_SCHEMA}: unknown={unknown}, missing={missing}"
            )
        if Path(config["parent_config"]).name != "default.json":
            raise ValueError("MTS experiments must inherit configs/mts/default.json")
        experiment_values = {
            "experiment_id": str(config["experiment_id"]),
            "geometry_mode": str(config.get("geometry_mode", "current_mcl")),
            "modalities": list(config.get("modalities", ["graph"])),
            "fusion_mode": str(config.get("fusion_mode", "none")),
            "finetune_mode": str(config.get("finetune_mode", "single_task")),
            "evaluation_protocol": str(config.get("evaluation_protocol", "historical_shared5")),
            "regression_loss": str(config.get("regression_loss", "huber")),
            "huber_beta": float(config.get("huber_beta", 0.5)),
            "finetune_profile": str(config.get(
                "finetune_profile", "legacy_mts_huber_v1"
            )),
            "patience": int(config.get("patience", 10)),
            "warmup_epochs": int(config.get("warmup_epochs", 5)),
            "weight_decay": float(config.get("weight_decay", 0.02)),
            "swa_start_epoch": int(config.get("swa_start_epoch", -1)),
            "legacy_graph_trainable_from_epoch0": bool(config.get(
                "legacy_graph_trainable_from_epoch0", True
            )),
            "modality_control": str(config.get("modality_control", "real")),
            "controlled_modality": config.get("controlled_modality"),
            "topology_representation": str(config.get(
                "topology_representation", TOPOLOGY_CANONICAL
            )),
        }
        ablation = config.get("ablation")
        # Historical non-ablation experiment configs remain valid as the
        # full-production (A3-equivalent) route.  The stricter object contract
        # applies whenever an active A0--A4 config declares ``ablation``.
        if ablation is None:
            experiment_values["ablation_id"] = None
            experiment_values["use_star_rbf"] = True
            experiment_values["use_mcl"] = True
            experiment_values["mcl_random_mask"] = False
            experiment_values["shared_checkpoint"] = None
        else:
            if not isinstance(ablation, dict):
                raise ValueError("MTS ablation must be an object")
            ablation_unknown = sorted(set(ablation) - {
            "id", "star", "mcl", "mcl_random_mask", "shared_checkpoint",
            })
            ablation_missing = sorted({
            "id", "star", "mcl", "mcl_random_mask", "shared_checkpoint",
            } - set(ablation))
            if ablation_unknown or ablation_missing:
                raise ValueError(
                    "invalid MTS ablation contract: "
                    f"unknown={ablation_unknown}, missing={ablation_missing}"
                )
            ablation_id = str(ablation["id"])
            if ablation_id not in ABLATION_IDS:
                raise ValueError(f"unsupported MTS ablation id: {ablation_id!r}")
            switches = {key: ablation[key] for key in (
            "star", "mcl", "mcl_random_mask"
            )}
            if any(not isinstance(value, bool) for value in switches.values()):
                raise ValueError("MTS ablation switches must be JSON booleans")
            expected_switches = {
            "A0_no3d_forward": (False, False, False),
            "A1_star_only": (True, False, False),
            "A2_mcl_real": (False, True, False),
            "A3_star_mcl_real": (True, True, False),
            "A4_star_mcl_random_mask": (True, True, True),
            }[ablation_id]
            observed_switches = (
            bool(ablation["star"]), bool(ablation["mcl"]),
            bool(ablation["mcl_random_mask"]),
            )
            if observed_switches != expected_switches:
                raise ValueError(
                f"ablation {ablation_id} has switches {observed_switches}; "
                f"expected {expected_switches}"
                )
            checkpoint_path = Path(str(ablation["shared_checkpoint"]))
            if not checkpoint_path.is_absolute():
                checkpoint_path = PROJECT_ROOT / checkpoint_path
            checkpoint_path = checkpoint_path.resolve()
            expected_checkpoint = (PROJECT_ROOT / MTS_SHARED_CHECKPOINT).resolve()
            if checkpoint_path != expected_checkpoint:
                raise ValueError(
                "all A0-A4 experiments must use the fixed shared checkpoint: "
                f"{expected_checkpoint}"
                )
            if not checkpoint_path.is_file():
                raise ValueError(f"shared checkpoint is missing: {checkpoint_path}")
            if _sha256_file(checkpoint_path) != MTS_SHARED_CHECKPOINT_SHA256:
                raise ValueError("shared checkpoint SHA256 does not match the contract")
            complete_path = Path(str(checkpoint_path) + ".complete.json")
            if not complete_path.is_file():
                raise ValueError("shared checkpoint completion metadata is missing")
            complete = json.loads(complete_path.read_text(encoding="utf-8"))
            if (
            complete.get("checkpoint_sha256") != MTS_SHARED_CHECKPOINT_SHA256
            or int(complete.get("optimizer_steps", -1)) != 20000
            ):
                raise ValueError("shared checkpoint completion metadata is stale")
            experiment_values["ablation_id"] = ablation_id
            experiment_values["use_star_rbf"] = bool(ablation["star"])
            experiment_values["use_mcl"] = bool(ablation["mcl"])
            experiment_values["mcl_random_mask"] = bool(ablation["mcl_random_mask"])
            experiment_values["shared_checkpoint"] = str(checkpoint_path)
        if experiment_values["geometry_mode"] not in {
            "current_mcl", "mcl_rbf", "disabled", "coordinate_shuffled",
            "mcl_rbf_coordinate_shuffled",
        }:
            raise ValueError("unsupported geometry_mode")
        if experiment_values["modalities"] not in (
            ["graph"], ["graph", "smiles"], ["graph", "fp"],
            ["graph", "smiles", "fp"],
        ):
            raise ValueError("modalities must keep graph as the anchor")
        if experiment_values["fusion_mode"] not in {"none", "zero_gated_residual"}:
            raise ValueError("unsupported fusion_mode")
        if experiment_values["regression_loss"] != "huber":
            raise ValueError(
                "MTS production experiments use the fixed Huber loss; "
                "the retired F/MSE logic is not supported"
            )
        if experiment_values["huber_beta"] != 0.5:
            raise ValueError("legacy_mts_huber_v1 requires huber_beta=0.5")
        if experiment_values["finetune_profile"] != "legacy_mts_huber_v1":
            raise ValueError(
                "MTS experiments must use finetune_profile=legacy_mts_huber_v1"
            )
        if (
            experiment_values["patience"],
            experiment_values["warmup_epochs"],
            experiment_values["weight_decay"],
            experiment_values["swa_start_epoch"],
            experiment_values["legacy_graph_trainable_from_epoch0"],
        ) != (10, 5, 0.02, -1, True):
            raise ValueError(
                "legacy_mts_huber_v1 requires patience=10, warmup_epochs=5, "
                "weight_decay=0.02, swa_start_epoch=-1 and full graph training"
            )
        if experiment_values["modality_control"] not in {"real", "batch_shuffled", "constant_zero"}:
            raise ValueError("unsupported modality_control")
        if experiment_values["controlled_modality"] not in {None, "smiles", "fp"}:
            raise ValueError("controlled_modality must be smiles or fp")
        if experiment_values["topology_representation"] not in TOPOLOGY_REPRESENTATIONS:
            raise ValueError("unsupported topology_representation")
        config_for_contract = reference
    else:
        config_for_contract = config
    if not experiment and config != reference:
        unknown = sorted(set(config) - set(reference))
        missing = sorted(set(reference) - set(config))
        raise ValueError(
            f"{ROUTE_NAME} ({ROUTE_SHORT_NAME}) has one fixed production config; "
            f"unknown={unknown}, missing={missing}, or a fixed value changed"
        )
    contract_checks = {
        "schema_version": CONFIG_SCHEMA,
        "feature_schema": FEATURE_SCHEMA,
        "topology_lmdb_schema": TOPOLOGY_LMDB_SCHEMA,
        "lga_schema_version": CANONICAL_LGA_SCHEMA_VERSION,
        "cache_bundle_schema": CACHE_BUNDLE_SCHEMA,
        "trimer_cache_schema": TRIMER_CONTENT_SCHEMA,
        "trimer_lmdb_schema": TRIMER_LMDB_SCHEMA,
        "cache_layout_schema": CACHE_LAYOUT_SCHEMA,
        "checkpoint_schema": CHECKPOINT_SCHEMA,
    }
    for key, expected in contract_checks.items():
        if config_for_contract.get(key) != expected:
            raise ValueError(
                f"MTS contract mismatch for {key}: "
                f"expected {expected!r}, got {config.get(key)!r}"
            )
    trimer = config_for_contract.get("trimer", {})
    protocol_values = {
        "protocol": TRIMER_PROTOCOL,
        "etkdg_max_iterations": 42,
        "etkdg_timeout_seconds": 60,
        "mmff_max_iterations": TRIMER_MMFF_RELAX_MAX_ITERATIONS,
        "mmff_variant": "MMFF94",
        "require_mmff_convergence": TRIMER_REQUIRE_MMFF_CONVERGENCE,
        "acceptance": TRIMER_ACCEPTANCE,
        "selection": TRIMER_SELECTION,
    }
    for key, expected in protocol_values.items():
        if trimer.get(key) != expected:
            raise ValueError(
                f"MTS Trimer contract mismatch for {key}: "
                f"expected {expected!r}, got {trimer.get(key)!r}"
            )
    if int(config_for_contract.get("multimer_builder_version", -1)) != 2:
        raise ValueError("unexpected multimer builder version")
    if config_for_contract.get("boundary_distance_algorithm") != "expanded_graph_shortest_path":
        raise ValueError("unexpected boundary distance algorithm")
    topology_representation = (
        experiment_values["topology_representation"]
        if experiment else str(config_for_contract.get(
            "topology_representation", TOPOLOGY_CANONICAL
        ))
    )
    if topology_representation not in TOPOLOGY_REPRESENTATIONS:
        raise ValueError("unsupported topology_representation")
    serialized_hash = digest(config)
    graph_contract = {
        key: config_for_contract[key] for key in (
            "architecture", "mips_core", "mips_variant",
            "atom_feature_mode", "layers", "hidden_dim", "heads",
            "ffn_dim", "max_hops", "boundary_threshold",
            "attachment_site_policy", "mismatched_bond_policy",
            "boundary_distance_algorithm", "multimer_builder_version",
            "attention_direction", "attention_scale", "norm_mode",
            "activation", "spd_bias_mode", "path_bias_mode",
            "star_distance_rbf", "trimer", "descriptor",
        )
    }
    graph_contract["topology_representation"] = topology_representation
    graph_hash = digest(graph_contract)
    resolved_feature_schema = (
        EXPLICIT_FEATURE_SCHEMA
        if topology_representation == TOPOLOGY_EXPLICIT else FEATURE_SCHEMA
    )
    resolved_topology_schema = (
        EXPLICIT_TOPOLOGY_LMDB_SCHEMA
        if topology_representation == TOPOLOGY_EXPLICIT else TOPOLOGY_LMDB_SCHEMA
    )
    resolved_lga_schema = (
        EXPLICIT_LGA_SCHEMA_VERSION
        if topology_representation == TOPOLOGY_EXPLICIT
        else CANONICAL_LGA_SCHEMA_VERSION
    )
    feature_hash = digest({
        "feature_schema": resolved_feature_schema,
        "topology_representation": topology_representation,
        "trimer_cache_schema": config_for_contract["trimer_cache_schema"],
        "max_hops": config_for_contract["max_hops"],
        "boundary_threshold": config_for_contract["boundary_threshold"],
        "attachment_site_policy": config_for_contract["attachment_site_policy"],
        "mismatched_bond_policy": config_for_contract["mismatched_bond_policy"],
        "boundary_distance_algorithm": config_for_contract["boundary_distance_algorithm"],
        "multimer_builder_version": config_for_contract["multimer_builder_version"],
        "trimer": config_for_contract["trimer"],
        "descriptor": config_for_contract["descriptor"],
    })
    resolved_geometry_mode = (
        experiment_values["geometry_mode"] if experiment
        else "trimer_scage_mcl"
    )
    geometry_hash = digest({
        "feature_schema": config_for_contract["feature_schema"],
        "trimer_cache_schema": config_for_contract["trimer_cache_schema"],
        "star_distance_rbf": config_for_contract["star_distance_rbf"],
        "trimer": config_for_contract["trimer"],
        # Geometry proposals change the instantiated MCL module even when
        # their zero-initialized parameters can be migrated from the joint
        # checkpoint.  Keep that experiment identity separate from the
        # immutable O8/topology hash.
        "geometry_mode": resolved_geometry_mode,
        "mcl_use_distance_bias": "mcl_rbf" in resolved_geometry_mode,
        "mcl_coordinate_shuffle": "coordinate_shuffled" in resolved_geometry_mode,
        "mcl_num_rbf": 64,
        "mcl_rbf_range": [0.0, 8.0],
        "mcl_distance_percentiles": [0.20, 0.50],
        "mcl_layers": 2,
        "use_star_rbf": (
            experiment_values["use_star_rbf"] if experiment else True
        ),
        "use_mcl": experiment_values["use_mcl"] if experiment else True,
        "mcl_mask_mode": (
            "count_matched_random"
            if experiment and experiment_values["mcl_random_mask"]
            else "real"
        ),
        "random_mask_schema": ABLATION_RANDOM_MASK_SCHEMA if experiment else None,
        "random_mask_seed": ABLATION_RANDOM_MASK_SEED if experiment else None,
        "random_mask_payload_version": (
            ABLATION_RANDOM_MASK_PAYLOAD_VERSION if experiment else None
        ),
    })
    payload = {
        "config_path": str(path),
        "config_hash": serialized_hash,
        "graph_model_config_hash": graph_hash,
        "geometry_model_config_hash": geometry_hash,
        "source_geometry_model_config_hash": digest({
            "feature_schema": config_for_contract["feature_schema"],
            "trimer_cache_schema": config_for_contract["trimer_cache_schema"],
            "star_distance_rbf": config_for_contract["star_distance_rbf"],
            "trimer": config_for_contract["trimer"],
            "geometry_mode": "trimer_scage_mcl",
            "mcl_use_distance_bias": False,
            "mcl_coordinate_shuffle": False,
            "mcl_num_rbf": 64,
            "mcl_rbf_range": [0.0, 8.0],
            "mcl_distance_percentiles": [0.20, 0.50],
            "mcl_layers": 2,
        }),
        "feature_config_hash": feature_hash,
        "topology_representation": topology_representation,
        "resolved_config_schema": (
            EXPERIMENT_CONFIG_SCHEMA if experiment else CONFIG_SCHEMA
        ),
        "experiment_id": (
            experiment_values["experiment_id"] if experiment else ROUTE_NAME
        ),
        "graph_geometry_mode": (
            resolved_geometry_mode
        ),
        "modalities": (
            experiment_values["modalities"] if experiment else ["graph"]
        ),
        "fusion_mode": (
            experiment_values["fusion_mode"] if experiment else "none"
        ),
        "finetune_mode": (
            experiment_values["finetune_mode"] if experiment else "single_task"
        ),
        "evaluation_protocol": (
            experiment_values["evaluation_protocol"] if experiment else "historical_shared5"
        ),
        "regression_loss": (
            experiment_values["regression_loss"] if experiment else "huber"
        ),
        "huber_beta": (
            experiment_values["huber_beta"] if experiment else 0.5
        ),
        "finetune_profile": (
            experiment_values["finetune_profile"]
            if experiment else "legacy_mts_huber_v1"
        ),
        "modality_control": (
            experiment_values["modality_control"] if experiment else "real"
        ),
        "controlled_modality": (
            experiment_values["controlled_modality"] if experiment else None
        ),
        "ablation_id": (
            experiment_values["ablation_id"] if experiment else None
        ),
        "use_star_rbf": (
            experiment_values["use_star_rbf"] if experiment else True
        ),
        "use_mcl": (
            experiment_values["use_mcl"] if experiment else True
        ),
        "mcl_random_mask": (
            experiment_values["mcl_random_mask"] if experiment else False
        ),
        "shared_checkpoint": (
            experiment_values["shared_checkpoint"] if experiment else None
        ),
        "shared_checkpoint_sha256": (
            MTS_SHARED_CHECKPOINT_SHA256 if experiment else None
        ),
        "mcl_mask_mode": (
            "count_matched_random"
            if experiment and experiment_values["mcl_random_mask"]
            else "real"
        ),
        "random_mask_schema": ABLATION_RANDOM_MASK_SCHEMA if experiment else None,
        "random_mask_seed": ABLATION_RANDOM_MASK_SEED if experiment else None,
        "random_mask_payload_version": (
            ABLATION_RANDOM_MASK_PAYLOAD_VERSION if experiment else None
        ),
        "random_mask_sidecar": (
            str(_random_mask_sidecar_path().relative_to(PROJECT_ROOT))
            if experiment and experiment_values["ablation_id"]
            == "A4_star_mcl_random_mask"
            else None
        ),
        "runtime_contract": {
            "feature_schema": resolved_feature_schema,
            "topology_lmdb_schema": resolved_topology_schema,
            "lga_schema_version": resolved_lga_schema,
            "cache_bundle_schema": CACHE_BUNDLE_SCHEMA,
            "cache_layout_schema": CACHE_LAYOUT_SCHEMA,
            "trimer_lmdb_schema": TRIMER_LMDB_SCHEMA,
            "trimer_builder_version": TRIMER_BUILDER_VERSION,
            "trimer_protocol": TRIMER_PROTOCOL,
        },
    }
    if args.shell:
        for key, value in payload.items():
            # Scalar values are consumed by ``eval`` in the production shell
            # wrapper.  JSON-encoding a scalar string first leaves literal
            # quote characters in CONFIG_HASH/FEATURE_CONFIG_HASH and makes
            # checkpoint metadata differ from the real artifact hashes.
            rendered = (
                json.dumps(value, sort_keys=True, separators=(",", ":"))
                if isinstance(value, (dict, list, tuple, bool, int, float))
                else ("" if value is None else str(value))
            )
            print(f"{key.upper()}={shlex.quote(rendered)}")
    else:
        print(json.dumps(payload, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
