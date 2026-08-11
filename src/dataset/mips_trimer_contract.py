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
FEATURE_SCHEMA = "mts-canonical-periodic-feature-v3"
LEGACY_FEATURE_SCHEMA = "mips-trimer-scage-feature-v4"
EXPLICIT_FEATURE_SCHEMA = "mts-explicit-kru-feature-v1"
EXPLICIT_TOPOLOGY_LMDB_SCHEMA = "mts-explicit-kru-topology-lmdb-v1"
# v7/v5 bind the independent normalized-Trimer atom identity table and the
# canonical periodic topology contract.  Keep these as single-source
# constants: changing a builder or config independently would make a cache
# appear reusable while its mapping semantics had changed.
TRIMER_CONTENT_SCHEMA = "mips-trimer-scage-trimer-v8"
TRIMER_LMDB_SCHEMA = "mips-trimer-scage-trimer-lmdb-v6"
CACHE_LAYOUT_SCHEMA = "mips-trimer-scage-lmdb-layout-v2"
CHECKPOINT_SCHEMA = "mts-model-v3"
# Historical cache-migration checkpoints keep the v3 identity above.  Fresh
# canonical pretraining uses a separate schema so a newly trained artifact
# cannot be mistaken for the metadata-migrated v3 control checkpoint.
PRETRAIN_CHECKPOINT_SCHEMA = "mts-model-v4"
PRETRAIN_TRAIN_STATE_SCHEMA = "mts-train-state-v3"
CACHE_BOND_ANGLE_SCHEMA = "mts-trimer-bond-angle-cache-v2"
CACHE_CONTINUOUS_ANGLE_SCHEMA = "mts-angle-continuous-cache-v1"
CACHE_MCL_THRESHOLD_SCHEMA = "mts-mcl-threshold-array-v2"
CACHE_BUNDLE_SCHEMA = "mts-canonical-cache-bundle-v3"
CACHE_TOPOLOGY_COST_SCHEMA = "mts-topology-cost-v1"
# Geometry-injection ablation (A0--A4).  These identifiers are part of the
# experiment contract, not a free-form naming convention.
ABLATION_IDS = (
    "A0_no3d_forward",
    "A1_star_only",
    "A2_mcl_real",
    "A3_star_mcl_real",
    "A4_star_mcl_random_mask",
)
# v2 adds the ordered 32-byte sample-key payload and a manifest-bound .done
# marker.  The frozen Trimer/threshold artifacts remain unchanged.
ABLATION_RANDOM_MASK_SCHEMA = "mts-mcl-count-matched-random-mask-v2"
ABLATION_RANDOM_MASK_SEED = 42
ABLATION_RANDOM_MASK_PAYLOAD_VERSION = 2
MTS_SHARED_CHECKPOINT = (
    "pretrained_models/mts/"
    "mts_joint_pretraining_pi1m_v2_seed42_canonical_angle20_v1.pth"
)
MTS_SHARED_CHECKPOINT_SHA256 = (
    "56c8a0ba148280fef22ecb953705a14259fb5e87258c0cae67799d7484375e77"
)
TOPOLOGY_LMDB_SCHEMA = "mts-canonical-periodic-topology-lmdb-v3"
MIGRATION_SCHEMA = "mts-canonical-cache-migration-v3"
TARGET_CONTRACT_SCHEMA = "mts-canonical-target-contract-v1"
PRETRAIN_TARGET_CONTRACT_SCHEMA = "mts-canonical-target-contract-v2"
PRETRAIN_PROFILE_SCHEMA = "mts-pretrain-profile-v1"
PRETRAIN_PROFILE_ID = "canonical_ru_angle20_v1"
BUILDER_VERSION = 12
CANONICAL_LGA_SCHEMA_VERSION = 2
EXPLICIT_LGA_SCHEMA_VERSION = 3
TOPOLOGY_CANONICAL = "canonical_lifted"
TOPOLOGY_EXPLICIT = "explicit_k_ru"
TOPOLOGY_REPRESENTATIONS = (TOPOLOGY_CANONICAL, TOPOLOGY_EXPLICIT)
# Descriptive aliases used by topology-only tools/tests.
CANONICAL_FEATURE_SCHEMA = FEATURE_SCHEMA
CANONICAL_TOPOLOGY_SCHEMA = TOPOLOGY_LMDB_SCHEMA
CANONICAL_CHECKPOINT_SCHEMA = CHECKPOINT_SCHEMA
CANONICAL_PERIODIC_TOPOLOGY_SCHEMA = TOPOLOGY_LMDB_SCHEMA
MTS_CANONICAL_PERIODIC_FEATURE_SCHEMA = FEATURE_SCHEMA
MTS_CANONICAL_PERIODIC_TOPOLOGY_LMDB_SCHEMA = TOPOLOGY_LMDB_SCHEMA
TRIMER_SCHEMA_VERSION = 8
TRIMER_BUILDER_VERSION = BUILDER_VERSION
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


def _canonical_json_hash(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()


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


def build_target_contract(
    *,
    source_cohort_hash,
    feature_config_hash,
    graph_model_config_hash,
    geometry_model_config_hash,
    topology_cache_artifact_hash,
    trimer_cache_artifact_hash,
    angle_cache_artifact_hash,
    cache_bundle_hash,
    store_json_sha256,
    topology_frozen_payload_sha256,
    trimer_frozen_payload_sha256,
    optimizer_steps,
    pretraining_objective,
):
    """Compute the complete target contract for a migrated canonical checkpoint.

    This is the single source of truth for the frozen production identity of a
    canonical single-RU checkpoint.  The schema constants come exclusively from
    this contract module; the hashes and artifact bindings come from the frozen
    bundle/store.  Callers must never hand-write a second set of schema
    constants, so the checkpoint cannot silently drift from the active
    contract.  ``source_contract`` (the pre-migration identity) is kept
    separately by the migration tool and is never computed here.
    """
    contract = {
        "schema": TARGET_CONTRACT_SCHEMA,
        "config_schema": CONFIG_SCHEMA,
        "experiment_config_schema": EXPERIMENT_CONFIG_SCHEMA,
        "feature_schema": FEATURE_SCHEMA,
        "topology_representation": TOPOLOGY_CANONICAL,
        "cache_layout_schema": CACHE_LAYOUT_SCHEMA,
        "cache_bundle_schema": CACHE_BUNDLE_SCHEMA,
        "topology_lmdb_schema": TOPOLOGY_LMDB_SCHEMA,
        "trimer_content_schema": TRIMER_CONTENT_SCHEMA,
        "trimer_lmdb_schema": TRIMER_LMDB_SCHEMA,
        "trimer_builder_version": TRIMER_BUILDER_VERSION,
        "canonical_lga_schema_version": CANONICAL_LGA_SCHEMA_VERSION,
        "trimer_protocol": TRIMER_PROTOCOL,
        "checkpoint_schema": CHECKPOINT_SCHEMA,
        "source_cohort_hash": str(source_cohort_hash),
        "feature_config_hash": str(feature_config_hash),
        "graph_model_config_hash": str(graph_model_config_hash),
        "geometry_model_config_hash": str(geometry_model_config_hash),
        "topology_cache_artifact_hash": str(topology_cache_artifact_hash),
        "trimer_cache_artifact_hash": str(trimer_cache_artifact_hash),
        # The angle cache schema is fixed to the continuous sidecar contract
        # (Plan contract-g0-baseline §3.2); the expected value comes from this
        # module's constant, never from the checkpoint itself.
        "angle_cache_schema": CACHE_CONTINUOUS_ANGLE_SCHEMA,
        "angle_cache_artifact_hash": str(angle_cache_artifact_hash),
        "cache_bundle_hash": str(cache_bundle_hash),
        "store_json_sha256": str(store_json_sha256),
        "topology_frozen_payload_sha256": str(topology_frozen_payload_sha256),
        "trimer_frozen_payload_sha256": str(trimer_frozen_payload_sha256),
        "optimizer_steps": int(optimizer_steps),
        "pretraining_objective": str(pretraining_objective),
    }
    # Self-excluding digest: target_contract_sha256 is computed over every
    # other field, so it cannot be part of its own definition.  The loader and
    # the doctor recompute it the same way.
    contract["target_contract_sha256"] = _canonical_json_hash(contract)
    return contract


def build_pretrain_target_contract(
    *,
    profile_id,
    source_cohort_hash,
    feature_config_hash,
    graph_model_config_hash,
    geometry_model_config_hash,
    topology_cache_artifact_hash,
    trimer_cache_artifact_hash,
    angle_cache_schema,
    angle_cache_artifact_hash,
    cache_bundle_hash,
    store_json_sha256,
    topology_frozen_payload_sha256,
    trimer_frozen_payload_sha256,
    optimizer_steps,
    pretraining_objective,
    topology_representation=TOPOLOGY_CANONICAL,
):
    """Build the v2 target contract used by fresh Angle-20 pretraining.

    ``build_target_contract`` remains the v1 migration contract for historical
    ``mts-model-v3`` artifacts.  Fresh training must carry its actual
    categorical sidecar identity and the v4 checkpoint schema directly, so it
    uses this separate constructor rather than mutating a migrated contract.
    """
    contract = {
        "schema": PRETRAIN_TARGET_CONTRACT_SCHEMA,
        "profile_id": str(profile_id),
        "topology_representation": str(topology_representation),
        "config_schema": CONFIG_SCHEMA,
        "experiment_config_schema": EXPERIMENT_CONFIG_SCHEMA,
        "feature_schema": FEATURE_SCHEMA,
        "cache_layout_schema": CACHE_LAYOUT_SCHEMA,
        "cache_bundle_schema": CACHE_BUNDLE_SCHEMA,
        "topology_lmdb_schema": TOPOLOGY_LMDB_SCHEMA,
        "trimer_content_schema": TRIMER_CONTENT_SCHEMA,
        "trimer_lmdb_schema": TRIMER_LMDB_SCHEMA,
        "trimer_builder_version": TRIMER_BUILDER_VERSION,
        "canonical_lga_schema_version": CANONICAL_LGA_SCHEMA_VERSION,
        "trimer_protocol": TRIMER_PROTOCOL,
        "checkpoint_schema": PRETRAIN_CHECKPOINT_SCHEMA,
        "source_cohort_hash": str(source_cohort_hash),
        "feature_config_hash": str(feature_config_hash),
        "graph_model_config_hash": str(graph_model_config_hash),
        "geometry_model_config_hash": str(geometry_model_config_hash),
        "topology_cache_artifact_hash": str(topology_cache_artifact_hash),
        "trimer_cache_artifact_hash": str(trimer_cache_artifact_hash),
        "angle_cache_schema": str(angle_cache_schema),
        "angle_cache_artifact_hash": str(angle_cache_artifact_hash),
        "cache_bundle_hash": str(cache_bundle_hash),
        "store_json_sha256": str(store_json_sha256),
        "topology_frozen_payload_sha256": str(topology_frozen_payload_sha256),
        "trimer_frozen_payload_sha256": str(trimer_frozen_payload_sha256),
        "optimizer_steps": int(optimizer_steps),
        "pretraining_objective": str(pretraining_objective),
    }
    contract["target_contract_sha256"] = _canonical_json_hash(contract)
    return contract


def validate_runtime_args(args) -> None:
    """Reject CLI overrides that would create a second production contract."""

    if getattr(args, "graph_encoder_type", None) != "mips_trimer_scage":
        return
    experiment = getattr(args, "config_schema", None) == EXPERIMENT_CONFIG_SCHEMA
    fixed = {
        "config_schema": CONFIG_SCHEMA,
        "topology_representation": TOPOLOGY_CANONICAL,
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
        # Explicit k-RU is an isolated comparison representation, not a
        # mutation of the canonical feature contract.
        observed_representation = getattr(
            args, "topology_representation", TOPOLOGY_CANONICAL
        )
        if observed_representation not in TOPOLOGY_REPRESENTATIONS:
            raise ValueError("unsupported MTS topology representation")
        fixed.pop("topology_representation", None)
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
