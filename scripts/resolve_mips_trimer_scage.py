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
    CACHE_BUNDLE_SCHEMA,
    CACHE_LAYOUT_SCHEMA,
    CANONICAL_LGA_SCHEMA_VERSION,
    CHECKPOINT_SCHEMA,
    CONFIG_SCHEMA,
    EXPERIMENT_CONFIG_SCHEMA,
    FEATURE_SCHEMA,
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
)


DEFAULT = (
    Path(__file__).resolve().parents[1]
    / "configs/mts/default.json"
)


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


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
        }
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
    serialized_hash = digest(config)
    graph_hash = digest({
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
    })
    feature_hash = digest({
        "feature_schema": config_for_contract["feature_schema"],
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
        "runtime_contract": {
            "feature_schema": FEATURE_SCHEMA,
            "topology_lmdb_schema": TOPOLOGY_LMDB_SCHEMA,
            "lga_schema_version": CANONICAL_LGA_SCHEMA_VERSION,
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
