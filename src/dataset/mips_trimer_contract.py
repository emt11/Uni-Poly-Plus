"""Single source of truth for the active MIPS-Trimer-SCAGE (MTS) contract.

The cache schemas intentionally retain their historical ``mips-trimer-scage``
prefix: they identify immutable feature content and changing that prefix would
force a needless million-record cache rebuild.  Route and stage display names
are kept separately below.
"""

import hashlib
import json

CONFIG_SCHEMA = "mts-config-v3"
EXPERIMENT_CONFIG_SCHEMA = "mts-experiment-v3"
FEATURE_SCHEMA = "mts-canonical-periodic-feature-v1"
LEGACY_FEATURE_SCHEMA = "mips-trimer-scage-feature-v4"
TRIMER_CONTENT_SCHEMA = "mips-trimer-scage-trimer-v5"
TRIMER_LMDB_SCHEMA = "mips-trimer-scage-trimer-lmdb-v3"
CACHE_LAYOUT_SCHEMA = "mips-trimer-scage-lmdb-layout-v2"
CHECKPOINT_SCHEMA = "mts-model-v3"
CACHE_BOND_ANGLE_SCHEMA = "mts-trimer-bond-angle-cache-v1"
CACHE_CONTINUOUS_ANGLE_SCHEMA = "mts-angle-continuous-cache-v1"
CACHE_BUNDLE_SCHEMA = "mips-trimer-scage-cache-bundle-v2"
TOPOLOGY_LMDB_SCHEMA = "mts-canonical-periodic-topology-lmdb-v1"
CANONICAL_LGA_SCHEMA_VERSION = 2
# Descriptive aliases used by topology-only tools/tests.
CANONICAL_FEATURE_SCHEMA = FEATURE_SCHEMA
CANONICAL_TOPOLOGY_SCHEMA = TOPOLOGY_LMDB_SCHEMA
CANONICAL_CHECKPOINT_SCHEMA = CHECKPOINT_SCHEMA
CANONICAL_PERIODIC_TOPOLOGY_SCHEMA = TOPOLOGY_LMDB_SCHEMA
MTS_CANONICAL_PERIODIC_FEATURE_SCHEMA = FEATURE_SCHEMA
MTS_CANONICAL_PERIODIC_TOPOLOGY_LMDB_SCHEMA = TOPOLOGY_LMDB_SCHEMA
TRIMER_SCHEMA_VERSION = 5
TRIMER_BUILDER_VERSION = 10
TRIMER_PROTOCOL = "etkdgv3x4-mmff94-relax200-lowest-finite-v1"
TRIMER_MMFF_VARIANT = "MMFF94"
TRIMER_MMFF_RELAX_MAX_ITERATIONS = 200
TRIMER_REQUIRE_MMFF_CONVERGENCE = False
TRIMER_ACCEPTANCE = "finite_3d_coordinates_and_finite_mmff_energy"
TRIMER_SELECTION = "lowest_finite_post_relaxation_energy"

# Public route/stage identity.  The underscore form remains the stable Python
# backend selector for compatibility with Dataset/model internals; user-facing
# CLI, logs and checkpoint metadata use the MTS names.
ROUTE_NAME = "MIPS-Trimer-SCAGE"
ROUTE_SHORT_NAME = "MTS"
ROUTE_INTERNAL = "mips_trimer_scage"
STAGE1_ID = "mts_joint_pretraining"
STAGE1_NAME = "MTS Joint Pretraining"
STAGE2_ID = "mts_property_finetune"
STAGE2_NAME = "MTS Property Fine-tuning"
# Kept as a source-level sentinel for callers that used to refer to a
# geometry-adaptation stage.  It is deliberately not a valid production
# stage and is rejected by normalize_stage below.
STAGE3_ID = "mts_property_finetune"
STAGE3_NAME = STAGE2_NAME

_ROUTE_ALIASES = {
    ROUTE_NAME.lower(): ROUTE_INTERNAL,
    ROUTE_SHORT_NAME.lower(): ROUTE_INTERNAL,
    ROUTE_INTERNAL: ROUTE_INTERNAL,
    "MIPS_Trimer_SCAGE".lower(): ROUTE_INTERNAL,
}

_STAGE_ALIASES = {
    STAGE1_ID: STAGE1_ID,
    # Read-only compatibility aliases for historical metadata/tests.  The
    # production pretrain dispatcher rejects these spellings before they can
    # select a separate training stage.
    "mts_topology": STAGE1_ID,
    "topology_pretrain": STAGE1_ID,
    "joint_pretrain": STAGE1_ID,
    "joint": STAGE1_ID,
    STAGE2_ID: STAGE2_ID,
    "stage2_geometry_adapt": STAGE2_ID,
    "finetune": STAGE2_ID,
}


def normalize_route(value):
    """Return the stable backend selector for a route spelling."""

    key = str(value or "").strip().lower()
    if key not in _ROUTE_ALIASES:
        raise ValueError(
            f"Unsupported route {value!r}; only {ROUTE_NAME} (MTS) is active."
        )
    return _ROUTE_ALIASES[key]


def normalize_stage(value):
    """Return a canonical public MTS stage id."""

    key = str(value or "").strip().lower()
    if key not in _STAGE_ALIASES:
        raise ValueError(
            f"Unsupported MTS stage {value!r}; expected "
            f"{STAGE1_ID} or {STAGE2_ID}."
        )
    return _STAGE_ALIASES[key]


def stage_display_name(value):
    stage = normalize_stage(value)
    return {
        STAGE1_ID: STAGE1_NAME,
        STAGE2_ID: STAGE2_NAME,
    }[stage]


def cache_bundle_binding_hash(
    *,
    cohort_hash,
    topology_artifact_hash,
    trimer_artifact_hash=None,
):
    """Return the immutable cache binding used by checkpoints.

    The hash deliberately contains only cache content identity.  Optimizer
    settings, loss weights and random seeds belong to a training hash and must
    not make an otherwise identical frozen feature bundle look different.
    ``None`` is retained for the topology-only Stage 1 export.
    """
    payload = {
        "schema": CACHE_BUNDLE_SCHEMA,
        "cohort_hash": str(cohort_hash or ""),
        "topology_artifact_hash": str(topology_artifact_hash or ""),
        "trimer_artifact_hash": str(trimer_artifact_hash or ""),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()


def validate_runtime_args(args) -> None:
    """Reject CLI overrides that would create a second production contract."""

    if getattr(args, "graph_encoder_type", None) != "mips_trimer_scage":
        return
    experiment = getattr(args, "config_schema", None) == EXPERIMENT_CONFIG_SCHEMA
    fixed = {
        "config_schema": CONFIG_SCHEMA,
        "mips_core": "paper_corrected",
        "mips_variant": "O8",
        "mips_max_hops": 2,
        "mips_atom_feature_mode": "mips137",
        "mips_attention_scale": "head_dim",
        "mips_norm_mode": "post",
        "mips_activation": "relu",
        "mips_spd_bias_mode": "per_head",
        "mips_path_bias_mode": "per_head_single_path_node",
        "mips_descriptor_components": "md200",
        "mips_descriptor_protocol": "source_star_sub",
        "mips_descriptor_fusion_mode": "graph_md_residual",
        "spatial_mode": "trimer_scage",
        "mips_fusion_mode": "none",
        "projection_mode": "plain",
        "trimer_num_candidates": 4,
        "trimer_max_heavy_atoms": 384,
    }
    if experiment:
        fixed["config_schema"] = EXPERIMENT_CONFIG_SCHEMA
    else:
        fixed["graph_geometry_mode"] = "trimer_scage_mcl"
    for name, expected in fixed.items():
        observed = getattr(args, name, None)
        if observed != expected:
            raise ValueError(
                f"{ROUTE_NAME} fixed contract mismatch for {name}: "
                f"expected {expected!r}, got {observed!r}"
            )
    if not bool(getattr(args, "mips_use_descriptors", False)):
        raise ValueError(f"{ROUTE_NAME} requires MD200")
    if list(getattr(args, "modalities", [])) != ["graph"]:
        if not experiment:
            raise ValueError(f"{ROUTE_NAME} production config is graph-only")
    if getattr(args, "fusion_type", None) != "none":
        if not experiment:
            raise ValueError(f"{ROUTE_NAME} production config requires fusion_type=none")
    geometry_modes = {
        "trimer_scage_mcl", "current_mcl", "mcl_rbf", "disabled",
        "coordinate_shuffled", "mcl_rbf_coordinate_shuffled",
    }
    if getattr(args, "graph_geometry_mode", None) not in geometry_modes:
        raise ValueError("unsupported mts-experiment-v3 geometry_mode")
    percentiles = tuple(float(value) for value in getattr(
        args, "mcl_distance_percentiles", (0.20, 0.50)
    ))
    if percentiles != (0.20, 0.50):
        raise ValueError(f"{ROUTE_NAME} requires MCL percentiles 0.20/0.50")
    if bool(getattr(args, "scage_use_pbc_distance", False)):
        raise ValueError(f"PBC distance is not part of {ROUTE_NAME}")
    if bool(getattr(args, "scage_use_descriptors", False)):
        raise ValueError(f"legacy SCAGE descriptors are not part of {ROUTE_NAME}")
