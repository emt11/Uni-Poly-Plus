#!/usr/bin/env python
"""Validate the one immutable MIPS-Trimer-SCAGE (MTS) production config."""

import argparse
import hashlib
import json
import shlex
from pathlib import Path

import sys
import numpy as np

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
    STAR_RBF_V2_SIDECAR_SCHEMA,
)


DEFAULT = (
    Path(__file__).resolve().parents[1]
    / "configs/mts/default.json"
)

# Isolated, pretraining-only experiment descriptors.  They inherit the
# immutable production route contract but carry a distinct experiment/config
# identity for the matched T0/T1 pretraining cycle.  The resolved runtime
# schema remains ``mts-config-v3`` so existing checkpoint consumers continue
# to enforce the production contract.
PRETRAIN_EXPERIMENT_CONFIG_SCHEMA = "mts-pretrain-experiment-v1"
G_FAMILY_COHORTS = ("PI1M_v2", "downstream_union")


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


def _resolve_g_family_artifact_bundle(bundle, *, kind, relation_bundle=None):
    """Validate an immutable two-cohort bundle without reading payload arrays."""
    if not isinstance(bundle, dict) or set(bundle) != set(G_FAMILY_COHORTS):
        raise ValueError(
            f"{kind} bundle must contain exactly {list(G_FAMILY_COHORTS)}"
        )
    resolved = {}
    expected_schema = {
        "relation_geometry": "mts-relation-geometry-sidecar-v1",
        "g3_permutation": "mts-relation-geometry-permutation-v1",
    }[kind]
    for cohort in G_FAMILY_COHORTS:
        entry = bundle[cohort]
        if not isinstance(entry, dict) or set(entry) != {"root", "artifact_hash"}:
            raise ValueError(f"{kind}.{cohort} requires only root and artifact_hash")
        expected_hash = str(entry["artifact_hash"])
        if len(expected_hash) != 64:
            raise ValueError(f"{kind}.{cohort} artifact_hash must be SHA256")
        root = Path(str(entry["root"]))
        if not root.is_absolute():
            root = PROJECT_ROOT / root
        root = root.resolve()
        metadata_path = root / "metadata.json"
        done_path = root / ".done"
        frozen_path = root / ".frozen"
        if not (metadata_path.is_file() and done_path.is_file() and frozen_path.is_file()):
            raise ValueError(f"{kind}.{cohort} artifact is incomplete: {root}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
        artifact_hash = str(metadata.get("artifact_hash", ""))
        metadata_hash = digest({
            key: value for key, value in metadata.items() if key != "artifact_hash"
        })
        if (
            metadata.get("schema") != expected_schema
            or frozen.get("schema") != expected_schema
            or metadata.get("cohort") != cohort
            or artifact_hash != expected_hash
            or done_path.read_text(encoding="utf-8").strip() != expected_hash
            or frozen.get("artifact_hash") != expected_hash
            or metadata_hash != expected_hash
        ):
            raise ValueError(f"{kind}.{cohort} immutable identity mismatch")
        if kind == "relation_geometry":
            source = metadata.get("source_identity")
            if not isinstance(source, dict) or set(source) != {"topology", "trimer"}:
                raise ValueError(f"{kind}.{cohort} source binding is incomplete")
            for name in ("topology", "trimer"):
                if len(str(source[name].get("done_artifact_hash", ""))) != 64:
                    raise ValueError(f"{kind}.{cohort} source {name} is unbound")
        else:
            relation_entry = relation_bundle[cohort]
            if (
                metadata.get("source_sidecar_artifact")
                != relation_entry["artifact_hash"]
                or int(metadata.get("seed", -1)) != 42
            ):
                raise ValueError(f"{kind}.{cohort} source sidecar binding mismatch")
        resolved[cohort] = {
            "root": str(root.relative_to(PROJECT_ROOT)),
            "artifact_hash": artifact_hash,
            "cohort_hash": str(metadata.get("cohort_hash", "")),
        }
    return resolved


def _resolve_star_rbf_v2_bundle(bundle):
    if not isinstance(bundle, dict) or set(bundle) != set(G_FAMILY_COHORTS):
        raise ValueError("Star-RBF v2 bundle must contain PI1M_v2 and downstream_union")
    resolved = {}
    semantic_hash = upper = None
    for cohort in G_FAMILY_COHORTS:
        entry = bundle[cohort]
        if not isinstance(entry, dict) or set(entry) != {"root", "artifact_hash"}:
            raise ValueError(f"star_rbf_v2_bundle.{cohort} requires root and artifact_hash")
        root = Path(str(entry["root"]))
        if not root.is_absolute(): root = PROJECT_ROOT / root
        metadata_path = root / "metadata.json"
        if not all((root / name).is_file() for name in ("metadata.json", ".done", ".frozen")):
            raise ValueError(f"Star-RBF v2 artifact is incomplete: {root}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        artifact = str(entry["artifact_hash"])
        frozen = json.loads((root / ".frozen").read_text(encoding="utf-8"))
        cohort_pointer = (
            PROJECT_ROOT / "data/processed/mips_trimer_scage/cohorts"
            / cohort / "current.json"
        )
        current = json.loads(cohort_pointer.read_text(encoding="utf-8"))
        cohort_root = cohort_pointer.parent / str(current["cohort_hash"])
        cohort_manifest = json.loads(
            (cohort_root / "manifest.json").read_text(encoding="utf-8")
        )
        cohort_count = int(
            np.load(cohort_root / "sample_keys.npy", mmap_mode="r", allow_pickle=False).shape[0]
        )
        if (
            metadata.get("schema") != STAR_RBF_V2_SIDECAR_SCHEMA
            or metadata.get("cohort") != cohort
            or metadata.get("selection") != "full_cohort"
            or int(metadata.get("record_count", -1)) != cohort_count
            or metadata.get("cohort_hash") != current["cohort_hash"]
            or metadata.get("ordered_sample_key_hash")
            != cohort_manifest["ordered_sample_key_hash"]
            or metadata.get("artifact_hash") != artifact
            or (root / ".done").read_text(encoding="utf-8").strip() != artifact
            or frozen.get("artifact_hash") != artifact
            or digest({k: v for k, v in metadata.items() if k != "artifact_hash"}) != artifact
        ):
            raise ValueError(f"Star-RBF v2 immutable identity mismatch: {cohort}")
        observed_semantic = str(metadata.get("model_semantic_hash", ""))
        observed_upper = float(metadata.get("rbf", {}).get("upper", -1))
        if semantic_hash is None:
            semantic_hash, upper = observed_semantic, observed_upper
        elif observed_semantic != semantic_hash or observed_upper != upper:
            raise ValueError("Star-RBF v2 cohort model semantics mismatch")
        resolved[cohort] = {"root": str(root.relative_to(PROJECT_ROOT)), "artifact_hash": artifact}
    return resolved, semantic_hash, upper


def g_family_bundle_identity_hash(arm, geometry_mode, relation_bundle, permutation_bundle):
    return digest({
        "g_family_arm": arm,
        "geometry_mode": geometry_mode,
        "relation_geometry_artifacts": (
            {cohort: relation_bundle[cohort]["artifact_hash"] for cohort in G_FAMILY_COHORTS}
            if relation_bundle is not None else None
        ),
        "g3_permutation_artifacts": (
            {cohort: permutation_bundle[cohort]["artifact_hash"] for cohort in G_FAMILY_COHORTS}
            if permutation_bundle is not None else None
        ),
    })


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
    config_schema = config.get("schema_version")
    pretrain_experiment = config_schema == PRETRAIN_EXPERIMENT_CONFIG_SCHEMA
    experiment = config_schema in {
        EXPERIMENT_CONFIG_SCHEMA,
        PRETRAIN_EXPERIMENT_CONFIG_SCHEMA,
    }
    experiment_values = None
    if config_schema == EXPERIMENT_CONFIG_SCHEMA:
        allowed = {
            "schema_version", "experiment_id", "parent_config",
            "geometry_mode", "modalities", "fusion_mode", "finetune_mode",
            "evaluation_protocol", "regression_loss", "huber_beta",
            "finetune_profile",
            "patience", "warmup_epochs", "weight_decay", "swa_start_epoch",
            "legacy_graph_trainable_from_epoch0",
            "modality_control", "controlled_modality",
            "topology_representation",
            "topology_attention_variant", "msta_layer_indices",
            "msta_local_spd", "msta_context_spd",
            "msta_share_relation_dropout", "msta_local_output_bias",
            "msta_local_output_init",
            "g_family_arm", "relation_geometry_bundle",
            "g3_permutation_bundle", "pretraining_objective",
            "angle_loss_weight", "shared_step0_id",
            "star_rbf", "star_rbf_v2_bundle", "backbone_definition",
            "attention_scale",
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
            "topology_attention_variant": str(config.get(
                "topology_attention_variant", "msta_last2"
            )),
            "msta_layer_indices": [
                int(value) for value in config.get("msta_layer_indices", [4, 5])
            ],
            "msta_local_spd": [
                int(value) for value in config.get("msta_local_spd", [0, 1])
            ],
            "msta_context_spd": [
                int(value) for value in config.get(
                    "msta_context_spd", [0, 1, 2]
                )
            ],
            "msta_share_relation_dropout": bool(config.get(
                "msta_share_relation_dropout", True
            )),
            "msta_local_output_bias": bool(config.get(
                "msta_local_output_bias", False
            )),
            "msta_local_output_init": str(config.get(
                "msta_local_output_init", "zero"
            )),
            "g_family_arm": config.get("g_family_arm"),
            "relation_geometry_bundle": config.get("relation_geometry_bundle"),
            "g3_permutation_bundle": config.get("g3_permutation_bundle"),
            "pretraining_objective": str(config.get("pretraining_objective", "masked_atom_only")),
            "angle_loss_weight": float(config.get("angle_loss_weight", 0.0)),
            "shared_step0_id": config.get("shared_step0_id"),
            "star_rbf": config.get("star_rbf"),
            "star_rbf_v2_bundle": config.get("star_rbf_v2_bundle"),
            "backbone_definition": config.get("backbone_definition"),
            "attention_scale": str(config.get("attention_scale", "head_dim")),
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
            "g0", "g1", "g2", "g3",
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
        if experiment_values["topology_attention_variant"] not in {
            "o8", "msta_last2"
        }:
            raise ValueError("unsupported topology_attention_variant")
        if experiment_values["msta_layer_indices"] != [4, 5]:
            raise ValueError("MSTA layer indices must be [4, 5]")
        if experiment_values["msta_local_spd"] != [0, 1]:
            raise ValueError("MSTA local SPD support must be [0, 1]")
        if experiment_values["msta_context_spd"] != [0, 1, 2]:
            raise ValueError("MSTA context SPD support must be [0, 1, 2]")
        if not experiment_values["msta_share_relation_dropout"]:
            raise ValueError("MSTA requires shared relation dropout")
        if experiment_values["msta_local_output_bias"]:
            raise ValueError("MSTA local_output must be bias-free")
        if experiment_values["msta_local_output_init"] != "zero":
            raise ValueError("MSTA local_output must use zero initialization")
        if experiment_values["g_family_arm"] is not None:
            if experiment_values["g_family_arm"] not in {"g0", "g1", "g2", "g3"}:
                raise ValueError("unsupported g_family_arm")
            if experiment_values["geometry_mode"] != experiment_values["g_family_arm"]:
                raise ValueError("G-family arm and geometry_mode must match")
            if experiment_values["angle_loss_weight"] != 0.0 or experiment_values["pretraining_objective"] != "masked_atom_only":
                raise ValueError("G-family readiness requires masked_atom_only and angle_loss_weight=0")
            if config.get("ablation") is not None:
                raise ValueError("G-family configs cannot use the historical A0-A4 ablation object")
            experiment_values["use_star_rbf"] = False
            experiment_values["use_mcl"] = False
            if experiment_values["shared_step0_id"] is None:
                raise ValueError("G-family config requires shared_step0_id")
            arm = experiment_values["g_family_arm"]
            relation_bundle = experiment_values["relation_geometry_bundle"]
            permutation_bundle = experiment_values["g3_permutation_bundle"]
            if arm == "g0":
                if relation_bundle is not None or permutation_bundle is not None:
                    raise ValueError("G0 must not bind relation geometry artifacts")
                resolved_relation_bundle = None
                resolved_permutation_bundle = None
            else:
                resolved_relation_bundle = _resolve_g_family_artifact_bundle(
                    relation_bundle, kind="relation_geometry"
                )
                if arm == "g3":
                    resolved_permutation_bundle = _resolve_g_family_artifact_bundle(
                        permutation_bundle,
                        kind="g3_permutation",
                        relation_bundle=resolved_relation_bundle,
                    )
                else:
                    if permutation_bundle is not None:
                        raise ValueError(f"{arm.upper()} must not bind a G3 permutation")
                    resolved_permutation_bundle = None
            experiment_values["relation_geometry_bundle"] = resolved_relation_bundle
            experiment_values["g3_permutation_bundle"] = resolved_permutation_bundle
            experiment_values["g_family_bundle_hash"] = g_family_bundle_identity_hash(
                arm, experiment_values["geometry_mode"],
                resolved_relation_bundle, resolved_permutation_bundle,
            )
        star_contract = experiment_values.get("star_rbf")
        if star_contract is not None:
            if experiment_values.get("attention_scale") != "head_dim":
                raise ValueError("R2 requires attention_scale=head_dim")
            expected_star = {
                "enabled": True,
                "definition": "trimer_periodic_relation_rbf_v2",
                "max_spd": 2,
                "max_abs_shift": 2,
                "shift0": "central_direct",
                "shift1": "dual_observation_rbf_mean",
                "shift2": "outer_trimer_direct",
                "inversion": "canonical_pair_shared",
                "trivial_self": "zero_bias",
                "asymmetry": {"stored": True, "model_input": False, "hard_filter": False},
            }
            if star_contract != expected_star:
                raise ValueError("invalid Star-RBF v2 scientific contract")
            bundle, semantic_hash, upper = _resolve_star_rbf_v2_bundle(
                experiment_values.get("star_rbf_v2_bundle")
            )
            if experiment_values.get("backbone_definition") != "legacy_g1_frozen":
                raise ValueError("R2 requires backbone_definition=legacy_g1_frozen")
            experiment_values["star_rbf_v2_bundle"] = bundle
            experiment_values["star_rbf_v2_model_semantic_hash"] = semantic_hash
            experiment_values["star_rbf_v2_upper"] = upper
            experiment_values["use_star_rbf"] = True
            experiment_values["use_mcl"] = False
        elif experiment_values.get("star_rbf_v2_bundle") is not None:
            raise ValueError("Star-RBF v2 bundle requires star_rbf contract")
        config_for_contract = reference
    elif config_schema == PRETRAIN_EXPERIMENT_CONFIG_SCHEMA:
        allowed = {
            "schema_version", "experiment_id", "parent_config",
            "topology_representation", "topology_attention_variant",
            "msta_layer_indices", "msta_local_spd", "msta_context_spd",
            "msta_share_relation_dropout", "msta_local_output_bias",
            "msta_local_output_init", "pretrain_profile", "paired_init_id",
        }
        unknown = sorted(set(config) - allowed)
        missing = sorted({"schema_version", "experiment_id", "parent_config"} - set(config))
        if unknown or missing:
            raise ValueError(
                f"invalid {PRETRAIN_EXPERIMENT_CONFIG_SCHEMA}: "
                f"unknown={unknown}, missing={missing}"
            )
        if Path(config["parent_config"]).name != "default.json":
            raise ValueError(
                "MTS pretraining experiments must inherit configs/mts/default.json"
            )
        if str(config.get("pretrain_profile", "canonical_ru_angle20_v1")) != "canonical_ru_angle20_v1":
            raise ValueError(
                "matched MTS pretraining experiments must use canonical_ru_angle20_v1"
            )
        if str(config.get("paired_init_id", "")) != "mts_t_pretrain0_matched_v1":
            raise ValueError(
                "matched MTS pretraining experiments must use the paired-init identity"
            )
        experiment_values = {
            "experiment_id": str(config["experiment_id"]),
            "geometry_mode": "trimer_scage_mcl",
            "modalities": ["graph"],
            "fusion_mode": "none",
            "finetune_mode": "single_task",
            "evaluation_protocol": "historical_shared5",
            "regression_loss": "huber",
            "huber_beta": 0.5,
            "finetune_profile": "legacy_mts_huber_v1",
            "patience": 10,
            "warmup_epochs": 5,
            "weight_decay": 0.02,
            "swa_start_epoch": -1,
            "legacy_graph_trainable_from_epoch0": True,
            "modality_control": "real",
            "controlled_modality": None,
            "topology_representation": str(config.get(
                "topology_representation", TOPOLOGY_CANONICAL
            )),
            "topology_attention_variant": str(config.get(
                "topology_attention_variant", "msta_last2"
            )),
            "msta_layer_indices": [int(value) for value in config.get(
                "msta_layer_indices", [4, 5]
            )],
            "msta_local_spd": [int(value) for value in config.get(
                "msta_local_spd", [0, 1]
            )],
            "msta_context_spd": [int(value) for value in config.get(
                "msta_context_spd", [0, 1, 2]
            )],
            "msta_share_relation_dropout": bool(config.get(
                "msta_share_relation_dropout", True
            )),
            "msta_local_output_bias": bool(config.get(
                "msta_local_output_bias", False
            )),
            "msta_local_output_init": str(config.get(
                "msta_local_output_init", "zero"
            )),
            "ablation_id": None,
            "use_star_rbf": True,
            "use_mcl": True,
            "mcl_random_mask": False,
            "shared_checkpoint": None,
        }
        if experiment_values["topology_representation"] not in TOPOLOGY_REPRESENTATIONS:
            raise ValueError("unsupported topology_representation")
        if experiment_values["topology_attention_variant"] not in {"o8", "msta_last2"}:
            raise ValueError("unsupported topology_attention_variant")
        if experiment_values["msta_layer_indices"] != [4, 5]:
            raise ValueError("MSTA layer indices must be [4, 5]")
        if experiment_values["msta_local_spd"] != [0, 1]:
            raise ValueError("MSTA local SPD support must be [0, 1]")
        if experiment_values["msta_context_spd"] != [0, 1, 2]:
            raise ValueError("MSTA context SPD support must be [0, 1, 2]")
        if not experiment_values["msta_share_relation_dropout"]:
            raise ValueError("MSTA requires shared relation dropout")
        if experiment_values["msta_local_output_bias"]:
            raise ValueError("MSTA local_output must be bias-free")
        if experiment_values["msta_local_output_init"] != "zero":
            raise ValueError("MSTA local_output must use zero initialization")
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
    topology_attention_variant = (
        experiment_values["topology_attention_variant"]
        if experiment else str(config_for_contract.get(
            "topology_attention_variant", "msta_last2"
        ))
    )
    msta_layer_indices = (
        experiment_values["msta_layer_indices"]
        if experiment else [int(value) for value in config_for_contract.get(
            "msta_layer_indices", [4, 5]
        )]
    )
    msta_local_spd = (
        experiment_values["msta_local_spd"]
        if experiment else [int(value) for value in config_for_contract.get(
            "msta_local_spd", [0, 1]
        )]
    )
    msta_context_spd = (
        experiment_values["msta_context_spd"]
        if experiment else [int(value) for value in config_for_contract.get(
            "msta_context_spd", [0, 1, 2]
        )]
    )
    msta_share_relation_dropout = (
        experiment_values["msta_share_relation_dropout"]
        if experiment else bool(config_for_contract.get(
            "msta_share_relation_dropout", True
        ))
    )
    msta_local_output_bias = (
        experiment_values["msta_local_output_bias"]
        if experiment else bool(config_for_contract.get(
            "msta_local_output_bias", False
        ))
    )
    msta_local_output_init = (
        experiment_values["msta_local_output_init"]
        if experiment else str(config_for_contract.get(
            "msta_local_output_init", "zero"
        ))
    )
    if topology_attention_variant not in {"o8", "msta_last2"}:
        raise ValueError("unsupported topology_attention_variant")
    if msta_layer_indices != [4, 5] or msta_local_spd != [0, 1] \
            or msta_context_spd != [0, 1, 2]:
        raise ValueError("invalid MSTA topology attention support")
    if not msta_share_relation_dropout or msta_local_output_bias \
            or msta_local_output_init != "zero":
        raise ValueError("invalid MSTA local-output contract")
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
    # Keep the historical T0 graph hash stable so existing O8 checkpoints
    # remain strictly loadable.  The T1 identity is a new graph contract and
    # therefore carries the explicit MSTA fields in its hash.
    if topology_attention_variant != "o8":
        graph_contract.update({
            "topology_attention_variant": topology_attention_variant,
            "msta_layer_indices": msta_layer_indices,
            "msta_local_spd": msta_local_spd,
            "msta_context_spd": msta_context_spd,
            "msta_share_relation_dropout": msta_share_relation_dropout,
            "msta_local_output_bias": msta_local_output_bias,
            "msta_local_output_init": msta_local_output_init,
        })
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
        "star_rbf_definition": (
            "trimer_periodic_relation_rbf_v2"
            if experiment and experiment_values.get("star_rbf") else
            "legacy_sample_direct_link_v1"
        ),
        "star_rbf_v2_model_semantic_hash": (
            experiment_values.get("star_rbf_v2_model_semantic_hash")
            if experiment else None
        ),
        "star_rbf_v2_upper": (
            experiment_values.get("star_rbf_v2_upper") if experiment else None
        ),
        "backbone_definition": (
            experiment_values.get("backbone_definition") if experiment else None
        ),
        "attention_scale": (
            experiment_values.get("attention_scale", "head_dim")
            if experiment else "head_dim"
        ),
        "use_mcl": experiment_values["use_mcl"] if experiment else True,
        "mcl_mask_mode": (
            "count_matched_random"
            if experiment and experiment_values["mcl_random_mask"]
            else "real"
        ),
        "random_mask_schema": (
            ABLATION_RANDOM_MASK_SCHEMA
            if config_schema == EXPERIMENT_CONFIG_SCHEMA else None
        ),
        "random_mask_seed": (
            ABLATION_RANDOM_MASK_SEED
            if config_schema == EXPERIMENT_CONFIG_SCHEMA else None
        ),
        "random_mask_payload_version": (
            ABLATION_RANDOM_MASK_PAYLOAD_VERSION
            if config_schema == EXPERIMENT_CONFIG_SCHEMA else None
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
            CONFIG_SCHEMA
            if config_schema == PRETRAIN_EXPERIMENT_CONFIG_SCHEMA
            else (EXPERIMENT_CONFIG_SCHEMA if experiment else CONFIG_SCHEMA)
        ),
        "config_source_schema": config_schema,
        "experiment_id": (
            experiment_values["experiment_id"] if experiment else ROUTE_NAME
        ),
        "graph_geometry_mode": (
            resolved_geometry_mode
        ),
        "topology_attention_variant": topology_attention_variant,
        "msta_layer_indices": msta_layer_indices,
        "msta_local_spd": msta_local_spd,
        "msta_context_spd": msta_context_spd,
        "msta_share_relation_dropout": msta_share_relation_dropout,
        "msta_local_output_bias": msta_local_output_bias,
        "msta_local_output_init": msta_local_output_init,
        "g_family_arm": (
            experiment_values.get("g_family_arm") if experiment else None
        ),
        "relation_geometry_bundle": (
            experiment_values.get("relation_geometry_bundle") if experiment else None
        ),
        "star_rbf_definition": (
            "trimer_periodic_relation_rbf_v2"
            if experiment and experiment_values.get("star_rbf") else
            "legacy_sample_direct_link_v1"
        ),
        "star_rbf_v2_bundle": (
            experiment_values.get("star_rbf_v2_bundle") if experiment else None
        ),
        "star_rbf_v2_model_semantic_hash": (
            experiment_values.get("star_rbf_v2_model_semantic_hash") if experiment else None
        ),
        "star_rbf_v2_upper": (
            experiment_values.get("star_rbf_v2_upper", 3.0) if experiment else 3.0
        ),
        "star_rbf_v2_sidecar_pi1m_v2": (
            (experiment_values.get("star_rbf_v2_bundle") or {}).get("PI1M_v2", {}).get("root") if experiment else None
        ),
        "star_rbf_v2_artifact_pi1m_v2": (
            (experiment_values.get("star_rbf_v2_bundle") or {}).get("PI1M_v2", {}).get("artifact_hash") if experiment else None
        ),
        "star_rbf_v2_sidecar_downstream_union": (
            (experiment_values.get("star_rbf_v2_bundle") or {}).get("downstream_union", {}).get("root") if experiment else None
        ),
        "star_rbf_v2_artifact_downstream_union": (
            (experiment_values.get("star_rbf_v2_bundle") or {}).get("downstream_union", {}).get("artifact_hash") if experiment else None
        ),
        "backbone_definition": (
            experiment_values.get("backbone_definition") if experiment else None
        ),
        "attention_scale": (
            experiment_values.get("attention_scale", "head_dim")
            if experiment else "head_dim"
        ),
        "g3_permutation_bundle": (
            experiment_values.get("g3_permutation_bundle") if experiment else None
        ),
        "g_family_bundle_hash": (
            experiment_values.get("g_family_bundle_hash") if experiment else None
        ),
        "relation_geometry_sidecar_pi1m_v2": (
            (experiment_values.get("relation_geometry_bundle") or {}).get("PI1M_v2", {}).get("root")
            if experiment else None
        ),
        "relation_geometry_artifact_pi1m_v2": (
            (experiment_values.get("relation_geometry_bundle") or {}).get("PI1M_v2", {}).get("artifact_hash")
            if experiment else None
        ),
        "relation_geometry_sidecar_downstream_union": (
            (experiment_values.get("relation_geometry_bundle") or {}).get("downstream_union", {}).get("root")
            if experiment else None
        ),
        "relation_geometry_artifact_downstream_union": (
            (experiment_values.get("relation_geometry_bundle") or {}).get("downstream_union", {}).get("artifact_hash")
            if experiment else None
        ),
        "g3_permutation_sidecar_pi1m_v2": (
            (experiment_values.get("g3_permutation_bundle") or {}).get("PI1M_v2", {}).get("root")
            if experiment else None
        ),
        "g3_permutation_artifact_pi1m_v2": (
            (experiment_values.get("g3_permutation_bundle") or {}).get("PI1M_v2", {}).get("artifact_hash")
            if experiment else None
        ),
        "g3_permutation_sidecar_downstream_union": (
            (experiment_values.get("g3_permutation_bundle") or {}).get("downstream_union", {}).get("root")
            if experiment else None
        ),
        "g3_permutation_artifact_downstream_union": (
            (experiment_values.get("g3_permutation_bundle") or {}).get("downstream_union", {}).get("artifact_hash")
            if experiment else None
        ),
        "pretraining_objective": (
            experiment_values.get("pretraining_objective", "joint")
            if experiment else "joint"
        ),
        "angle_loss_weight": (
            experiment_values.get("angle_loss_weight", 0.25)
            if experiment else 0.25
        ),
        "shared_step0_id": (
            experiment_values.get("shared_step0_id") if experiment else None
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
            MTS_SHARED_CHECKPOINT_SHA256
            if config_schema == EXPERIMENT_CONFIG_SCHEMA else None
        ),
        "mcl_mask_mode": (
            "count_matched_random"
            if experiment and experiment_values["mcl_random_mask"]
            else "real"
        ),
            "random_mask_schema": (
                ABLATION_RANDOM_MASK_SCHEMA
                if config_schema == EXPERIMENT_CONFIG_SCHEMA else None
            ),
            "random_mask_seed": (
                ABLATION_RANDOM_MASK_SEED
                if config_schema == EXPERIMENT_CONFIG_SCHEMA else None
            ),
            "random_mask_payload_version": (
            ABLATION_RANDOM_MASK_PAYLOAD_VERSION
            if config_schema == EXPERIMENT_CONFIG_SCHEMA else None
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
