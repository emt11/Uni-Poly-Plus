"""Single source of truth for the active MIPS-Trimer-SCAGE (MTS) contract.

The cache schemas intentionally retain their historical ``mips-trimer-scage``
prefix: they identify immutable feature content and changing that prefix would
force a needless million-record cache rebuild.  Route and stage display names
are kept separately below.
"""

CONFIG_SCHEMA = "mts-config-v3"
FEATURE_SCHEMA = "mts-canonical-periodic-feature-v3"
LEGACY_FEATURE_SCHEMA = "mips-trimer-scage-feature-v4"
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
TOPOLOGY_LMDB_SCHEMA = "mts-canonical-periodic-topology-lmdb-v3"
MIGRATION_SCHEMA = "mts-canonical-cache-migration-v3"
TARGET_CONTRACT_SCHEMA = "mts-canonical-target-contract-v1"
PRETRAIN_TARGET_CONTRACT_SCHEMA = "mts-canonical-target-contract-v2"
BUILDER_VERSION = 12
CANONICAL_LGA_SCHEMA_VERSION = 2
TOPOLOGY_CANONICAL = "canonical_lifted"
TOPOLOGY_REPRESENTATIONS = (TOPOLOGY_CANONICAL,)
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


def validate_runtime_args(args) -> None:
    """Validate the fixed MTS-GLT-v2 baseline runtime."""

    if getattr(args, "graph_encoder_type", None) != ROUTE_INTERNAL:
        return
    schema = getattr(args, "config_schema", None)
    if schema not in {CONFIG_SCHEMA, "mts-glt-v2", "mts-glt-v2-downstream", "manual"}:
        raise ValueError(f"unsupported {ROUTE_NAME} config schema: {schema!r}")
    fixed = {
        "topology_representation": TOPOLOGY_CANONICAL,
        "topology_attention_variant": "o8",
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
        "graph_geometry_mode": "trimer_scage_mcl",
        "mips_fusion_mode": "none",
        "projection_mode": "plain",
        "trimer_num_candidates": 4,
        "trimer_max_heavy_atoms": 384,
    }
    for name, expected in fixed.items():
        observed = getattr(args, name, None)
        if observed != expected:
            raise ValueError(
                f"{ROUTE_NAME} runtime mismatch for {name}: "
                f"expected {expected!r}, got {observed!r}"
            )
    switches = {
        "use_star_rbf": False,
        "use_mcl": False,
        "use_md200": True,
        "mips_use_descriptors": True,
    }
    for name, expected in switches.items():
        if bool(getattr(args, name, not expected)) is not expected:
            raise ValueError(
                f"{ROUTE_NAME} runtime requires {name}={expected}"
            )
    if getattr(args, "mts_glt_version", None) not in {None, "v2"}:
        raise ValueError("MTS-GLT-v2 baseline requires mts_glt_version=v2")
    if getattr(args, "mts_glt_mode", None) not in {
        None, "none", "o8_only", "o8_glt_atom"
    }:
        raise ValueError("MTS-GLT-v2 baseline downstream mode is invalid")
    if list(getattr(args, "modalities", [])) != ["graph"]:
        raise ValueError(f"{ROUTE_NAME} baseline is graph-only")
    if getattr(args, "fusion_type", None) != "none":
        raise ValueError(f"{ROUTE_NAME} baseline requires fusion_type=none")
    if bool(getattr(args, "scage_use_pbc_distance", False)):
        raise ValueError(f"PBC distance is not part of {ROUTE_NAME}")
    if bool(getattr(args, "scage_use_descriptors", False)):
        raise ValueError(f"legacy SCAGE descriptors are not part of {ROUTE_NAME}")
