"""Build-spec based cache identity for the MTS frozen cache store.

This module replaces the retired schema/builder-version compatibility system.
Cache compatibility now answers exactly three questions:

1. which sample is this?            -> canonical sample key (sha256 of the
                                       RDKit-canonical P-SMILES; computed by
                                       ``src.dataset.lmdb_cache`` because it
                                       needs RDKit, this module stays pure)
2. how was this artifact built?     -> ``build_spec`` / ``build_spec_hash``
3. is it safe to read?              -> ``.frozen`` marker + ``store.json``

There are deliberately no schema versions, builder versions, migrations or
format levels here.  ``build_spec`` values may contain historical policy
names; a policy change changes the value, which changes the hash, which is
the entire compatibility mechanism.

This module is intentionally dependency-free (stdlib only): the formal
training reader imports it without RDKit or LMDB side effects.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


STORE_FILENAME = "store.json"
DEFAULT_CACHE_ROOT_NAME = "mts_cache"

ARTIFACT_TYPES = ("ru_base", "topology", "trimer", "md200")

# Fold/split identity must never enter a build spec: the whole-dataset cache
# is built once and every fold reuses it through index selection only.
FORBIDDEN_SPEC_KEYS = frozenset({
    "fold", "fold_id", "split", "split_id", "split_type", "scaffold_id",
    "train_idx", "val_idx", "valid_idx", "test_idx", "use_idxs",
    "task", "dataset_name", "target", "targets",
})

# Metadata keys that carry no build-spec meaning: manual version counters and
# parent references.  Parent references are re-bound through ``parents`` below
# using the registered parent artifact's build_spec_hash instead.  The trimer
# "protocol" string is a versioned display name whose real parameters are all
# present in the spec; the md200 "protocol" ("source_star_sub") is a genuine
# descriptor parameter and is kept.
_VERSION_METADATA_KEYS = frozenset({
    "builder_version",
    "multimer_builder_version",
    "feature_content_schema",
    "trimer_content_schema",
    "trimer_schema_version",
    "random_seed",
})
_PER_TYPE_METADATA_KEYS = {
    "trimer": frozenset({"protocol"}),
}
_PARENT_METADATA_KEYS = {
    "ru_base_feature_config_hash": "ru_base",
    "topology_feature_config_hash": "topology",
}


def _apply_semantic_aliases(artifact_type: str, parameters: dict) -> dict:
    """Translate known historical label values into semantic parameters.

    Each alias states that the historical label denotes exactly the semantics
    the current builder code implements (verified against
    ``normalize_polymer_smiles`` / ``_attach_mips_descriptors`` /
    ``build_canonical_periodic_topology``).  Anything unknown is kept
    verbatim; a future builder change edits the route spec instead.
    """

    output = {}
    for key, value in parameters.items():
        if (
            artifact_type == "ru_base"
            and key == "normalization"
            and value == "rdkit_canonical_psmiles_v1"
        ):
            output["canonicalization"] = dict(
                RU_BASE_BUILD_SPEC["parameters"]["canonicalization"]
            )
            continue
        if (
            artifact_type == "ru_base"
            and key == "molecule_binary"
            and value == "rdkit_mol_binary"
        ):
            output["molecule_serialization"] = "rdkit_mol_binary"
            continue
        if (
            artifact_type == "topology"
            and key == "lga_schema"
            and int(value) == 2
        ):
            output["lga_relation_encoding"] = "canonical_lifted_image_shift"
            continue
        if artifact_type == "md200" and key in {
            "components", "descriptor_schema", "protocol", "embedding",
            "optimizer",
        }:
            # The retired labels (components=[RDKit2DNormalized200],
            # descriptor_schema=5, protocol=source_star_sub) all denote the
            # single descriptor semantics documented in MD200_BUILD_SPEC.
            continue
        output[key] = value
    if artifact_type == "md200":
        output["descriptor"] = dict(MD200_BUILD_SPEC["parameters"]["descriptor"])
    return output


class StoreError(RuntimeError):
    """store.json is missing or structurally invalid."""


class SpecError(ValueError):
    """A build spec is not a valid canonical build identity."""


def canonical_json(value) -> str:
    """Deterministic serialization used for every hash in this module."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def build_spec_hash(build_spec) -> str:
    return hashlib.sha256(canonical_json(build_spec).encode("utf-8")).hexdigest()


def validate_build_spec(build_spec) -> dict:
    """Reject spec inputs that would tie artifact identity to a data split."""

    if not isinstance(build_spec, dict):
        raise SpecError("build_spec must be a mapping")
    if not isinstance(build_spec.get("artifact_type"), str):
        raise SpecError("build_spec requires a string artifact_type")

    def _walk(node, path):
        if isinstance(node, dict):
            for key, value in node.items():
                key = str(key)
                if key.lower() in FORBIDDEN_SPEC_KEYS:
                    raise SpecError(
                        f"build_spec must not depend on split identity: "
                        f"{path}/{key}"
                    )
                _walk(value, f"{path}/{key}")
        elif isinstance(node, (list, tuple)):
            for index, value in enumerate(node):
                _walk(value, f"{path}[{index}]")

    _walk(build_spec, "")
    return build_spec


def build_spec_from_metadata(artifact_type: str, metadata: dict) -> dict:
    """Return the explicit build spec written by the current generator.

    The rebuilt path intentionally has no inference rules for historical
    ``build_config`` payloads.  Metadata without the exact build spec is not a
    current artifact and cannot be registered.
    """

    observed = metadata.get("build_spec")
    if not isinstance(observed, dict):
        raise StoreError(f"{artifact_type} metadata has no explicit build_spec")
    observed = validate_build_spec(observed)
    if observed.get("artifact_type") != str(artifact_type):
        raise StoreError(f"{artifact_type} metadata build_spec type mismatch")
    expected_hash = build_spec_hash(observed)
    recorded_hash = metadata.get("build_spec_hash")
    if recorded_hash is not None and str(recorded_hash) != expected_hash:
        raise StoreError(f"{artifact_type} metadata build_spec_hash mismatch")
    return observed


# ---------------------------------------------------------------------------
# Route-expected build specs.  These describe the CURRENT scientific protocol
# (MIPS-Trimer-SCAGE, ETKDGv3 + MMFF94 open trimer, 3 RU, explicit H).  They
# are derived from the same constants the builders use; no version counters
# appear here — every entry describes what is actually computed.  A protocol
# change edits these parameters directly.
# ---------------------------------------------------------------------------

RU_BASE_BUILD_SPEC = validate_build_spec({
    "artifact_type": "ru_base",
    "parameters": {
        "attachment_site_policy": "two_sites_shared_boundary_allowed",
        "boundary_distance_algorithm": "expanded_graph_shortest_path",
        # normalize_polymer_smiles: RDKit parse -> canonical isomeric SMILES,
        # whitespace stripped, dummy atoms preserved, unparsable rows keep a
        # deterministic "INVALID::{source}" identity.
        "canonicalization": {
            "toolkit": "rdkit",
            "parse": "MolFromSmiles",
            "canonical_output_smiles": True,
            "isomeric": True,
            "strip_whitespace": True,
            "dummy_atom_policy": "preserve",
            "invalid_identity_policy": "INVALID::{source}",
        },
        "mismatched_bond_policy": "single",
        "molecule_serialization": "rdkit_mol_binary",
    },
    "parents": {},
})

TOPOLOGY_BUILD_SPEC = validate_build_spec({
    "artifact_type": "topology",
    "parameters": {
        "atom_features": "mips137_topology_only_trimer_central_ru",
        "boundary_distance_algorithm": "diagnostic_only_canonical_ru",
        "boundary_threshold": 5,
        # LGA relation rows carry explicit source image shifts and star-edge
        # masks (the behaviour behind the retired "lga_schema=2" label).
        "lga_relation_encoding": "canonical_lifted_image_shift",
        "max_hops": 2,
        "max_model_atoms": 384,
        "max_repeat_units": 1,
        "mismatched_bond_policy": "single",
        "topology_representation": "single_canonical_ru_lifted_relations",
    },
    "parents": {"ru_base": build_spec_hash(RU_BASE_BUILD_SPEC)},
})

# The seed derivation policy is part of the geometry algorithm: the whole
# seed spec lives inside the build_spec so seeds are recoverable from the
# artifact identity alone.  It is deliberately not a version number.
TRIMER_SEED_POLICY = {
    "hash": "sha256",
    "material": ["geometry_seed_spec", "trimer_canonical_identity", "round_id"],
    "candidate_level_material": [
        "geometry_seed_spec", "trimer_canonical_identity", "round_id",
        "candidate_id",
    ],
    "material_separator": ":",
    "output_bytes": 4,
    "byte_order": "little",
    "mask": "0x7FFFFFFF",
    # Only parameters that affect the random search trajectory of ONE
    # candidate enter the seed material.  Timeout, selection policy, failure
    # policy, rounds and candidate counts are execution policy, not seeds.
    "geometry_seed_spec": {
        "embedder": "ETKDGv3",
        "use_random_coords": True,
        "enforce_chirality": True,
        "embed_max_iterations": 200,
        "prune_rms_thresh": -1.0,
        "force_field": "MMFF94",
        "mmff_relax_max_iterations": 200,
        "representation": "open_trimer",
        "repeat_units": 3,
        "close_periodic": False,
        "explicit_h": True,
        "terminal_capping": "missing_seam_bond_order_hydrogen_equivalents",
    },
}

TRIMER_BUILD_SPEC = validate_build_spec({
    "artifact_type": "trimer",
    "parameters": {
        "allow_2d_for_mcl": False,
        "attachment_site_policy": "two_sites_shared_boundary_allowed",
        "coordinate_payload": "single-conformer-explicit-all-atom",
        "embedder_timeout_seconds": 60,
        "etkdg_enforce_chirality": True,
        "etkdg_max_iterations": 200,
        "etkdg_rmsd_pruning": False,
        "etkdg_use_random_coords": True,
        "failure_policy": "exclude_geometry_failure",
        "hard_timeout_seconds": 60,
        "max_rounds": 2,
        "max_total_candidates": 8,
        "mmff_relax_max_iterations": 200,
        "mmff_variant": "MMFF94",
        "num_candidates_per_round": 4,
        "seed_policy": TRIMER_SEED_POLICY,
        "selection": "first_valid",
        "energy_ranking": False,
    },
    "parents": {
        "ru_base": build_spec_hash(RU_BASE_BUILD_SPEC),
    },
})

MD200_BUILD_SPEC = validate_build_spec({
    "artifact_type": "md200",
    "parameters": {
        # _attach_mips_descriptors: ru_base molecule binary -> star_sub
        # (both wildcards replaced by the opposite neighbour's element) ->
        # canonical SMILES -> RDKit2DNormalized.process -> drop return[0] ->
        # 200 float values; any non-finite value marks the row invalid.
        "descriptor": {
            "family": "rdkit_2d_descriptors_normalized",
            "dimension": 200,
            "drop_first_return_value": True,
            "source_transform": "star_sub",
            "source_molecule": "ru_base_mol_binary",
            "source_smiles": "rdkit_canonical",
            "nonfinite_policy": "mark_invalid",
            "invalid_payload": "zeros_200_valid_false",
            "embedding": "none",
            "optimizer": "none",
        },
    },
    "parents": {},
})

ROUTE_BUILD_SPECS = {
    "ru_base": RU_BASE_BUILD_SPEC,
    "topology": TOPOLOGY_BUILD_SPEC,
    "trimer": TRIMER_BUILD_SPEC,
    "md200": MD200_BUILD_SPEC,
}

ROUTE_BUILD_SPEC_HASHES = {
    name: build_spec_hash(spec) for name, spec in ROUTE_BUILD_SPECS.items()
}

# Fields the formal reader and the training route genuinely consume.  The
# reader checks presence (and the identity constraints in frozen_store);
# it never asks which schema produced the record.
REQUIRED_FIELDS = {
    "ru_base": (
        "normalized_polymer_smiles",
        "ru_chemistry_valid",
        "ru_base_valid",
        "ru_base_failure_code",
        "ru_atomic_number",
        "ru_edge_index",
        "ru_bond_type",
    ),
    "topology": (
        "x",
        "z",
        "edge_index",
        "canonical_ru_atom_index",
        "graph_available",
    ),
    "trimer": (
        "trimer_geometry_valid",
        "trimer_geometry_is_3d",
        "trimer_2d_fallback",
        "trimer_pos",
        "trimer_atomic_number",
        "trimer_edge_index",
        "trimer_bond_type",
        "trimer_central_ru_mask",
        "mips_to_trimer_central_index",
        "o8_heavy_mask",
        "o8_heavy_indices",
        "trimer_heavy_mask",
        "trimer_heavy_indices",
        "star_3d_distance",
        "star_3d_valid",
    ),
    "md200": (
        "mips_md",
        "mips_md_valid",
    ),
}

# Physical fields written by the new offline-only artifact builder.  These are
# deliberately not a compatibility schema: the names are exactly the fields
# consumed by current training/sidecar code or needed to prove atom identity.
# Runtime batches may create convenient aliases, but no alias is serialized.
RECORD_FIELDS = {
    "ru_base": (
        "normalized_polymer_smiles", "ru_chemistry_valid", "ru_base_valid",
        "ru_base_failure_code", "ru_mol_binary", "ru_normalized_mol_binary",
        "source_to_normalized_atom_id",
        "source_to_normalized_canonical_atom_id",
        "source_to_normalized_attachment_map", "ru_atomic_number",
        "ru_edge_index", "ru_bond_type", "ru_left_boundary",
        "ru_right_boundary", "ru_backbone", "ru_canonical_atom_index",
        "ru_attachment_bond_type", "ru_attachment_bond_type_left",
        "ru_attachment_bond_type_right", "ru_attachment_bond_mismatch",
        "ru_connection_bond_policy", "ru_shared_boundary",
    ),
    "topology": (
        "smiles", "normalized_canonical_smiles",
        "source_to_normalized_atom_id",
        "source_to_normalized_canonical_atom_id",
        "source_to_normalized_attachment_map", "mips_x",
        "mips_backbone_mask", "z", "edge_index", "edge_attr",
        "lga_edge_index", "lga_spd", "lga_path_index", "lga_path_shift",
        "lga_path_mask", "lga_path_bond_hist", "lga_source_image_shift",
        "lga_star_edge_mask", "canonical_ru_atom_index",
        "canonical_to_trimer_base_atom_id", "graph_available",
        "mips_condition_valid", "mips_alias_free", "mts_canonical_periodic",
        "topology_failure_code", "mips_boundary_distance",
        "mips_distance_threshold",
    ),
    "trimer": (
        "trimer_pos", "trimer_atomic_number", "trimer_isotope",
        "trimer_source_atom_count", "is_source_atom",
        "is_source_explicit_h", "is_added_h", "h_parent_heavy_index",
        "trimer_atom_id", "trimer_edge_index", "trimer_bond_type",
        "trimer_bond_aromatic", "trimer_formal_charge",
        "trimer_is_aromatic", "trimer_chiral_tag",
        "trimer_attachment_role", "trimer_internal_degree",
        "trimer_base_ru_atom_id", "trimer_ru_offset",
        "trimer_central_ru_mask", "trimer_central_atom_index",
        "mips_to_trimer_central_index", "o8_heavy_mask",
        "o8_heavy_indices", "trimer_heavy_mask", "trimer_heavy_indices",
        "trimer_geometry_valid", "trimer_geometry_is_3d",
        "trimer_2d_fallback", "trimer_geometry_source",
        "star_3d_distance", "star_3d_asymmetry", "star_3d_valid",
    ),
}

_BINDING_KEYS = ("path", "build_spec", "build_spec_hash", "artifact_hash",
                 "record_count")


def binding_for(artifact_type, *, path, build_spec, artifact_hash,
                record_count, **extra) -> dict:
    binding = {
        "path": str(path),
        "build_spec": validate_build_spec(build_spec),
        "build_spec_hash": build_spec_hash(build_spec),
        "artifact_hash": str(artifact_hash),
        "record_count": int(record_count),
    }
    binding.update(extra)
    return binding


def validate_store(store) -> dict:
    """Structural validation of a parsed store.json.  No versions involved."""

    if not isinstance(store, dict):
        raise StoreError("store.json must contain a JSON object")
    artifacts = store.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        raise StoreError("store.json is missing an 'artifacts' mapping")
    for name, binding in artifacts.items():
        if name not in ARTIFACT_TYPES:
            raise StoreError(f"store.json has an unknown artifact type: {name}")
        if not isinstance(binding, dict):
            raise StoreError(f"store.json binding is not an object: {name}")
        for key in _BINDING_KEYS:
            if key not in binding:
                raise StoreError(
                    f"store.json binding {name} is missing key: {key}"
                )
        expected = build_spec_hash(binding["build_spec"])
        if binding["build_spec_hash"] != expected:
            raise StoreError(
                f"store.json binding {name} build_spec_hash does not match "
                "its build_spec"
            )
    return store


def load_store(cache_root) -> dict:
    path = Path(cache_root) / STORE_FILENAME
    if not path.is_file():
        raise StoreError(
            f"cache store is missing: {path}. Run "
            "scripts/create_cache_store.py to register frozen artifacts."
        )
    with open(path, encoding="utf-8") as handle:
        return validate_store(json.load(handle))


def save_store(cache_root, store) -> Path:
    store = validate_store(store)
    path = Path(cache_root) / STORE_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(store, handle, sort_keys=True, indent=2)
        handle.write("\n")
    os.replace(temporary, path)
    return path


def route_mismatches(store) -> dict:
    """Return {artifact_type: (expected_hash, observed_hash)} mismatches."""

    mismatches = {}
    artifacts = store.get("artifacts", {})
    for name, expected_hash in ROUTE_BUILD_SPEC_HASHES.items():
        binding = artifacts.get(name)
        if binding is None:
            continue
        if str(binding.get("build_spec_hash")) != expected_hash:
            mismatches[name] = (expected_hash, str(binding.get("build_spec_hash")))
    return mismatches


def describe_route_mismatch(artifact_type, store) -> str:
    """Human-readable parameter diff between route spec and stored spec."""

    artifacts = store.get("artifacts", {})
    binding = artifacts.get(artifact_type) or {}
    route = ROUTE_BUILD_SPECS[artifact_type]
    observed = binding.get("build_spec", {})
    route_params = route.get("parameters", {})
    observed_params = observed.get("parameters", {})
    lines = []
    for key in sorted(set(route_params) | set(observed_params)):
        expected_value = route_params.get(key, "<absent>")
        observed_value = observed_params.get(key, "<absent>")
        if expected_value != observed_value:
            lines.append(f"  {key}: route={expected_value!r} artifact={observed_value!r}")
    route_parents = route.get("parents", {})
    observed_parents = observed.get("parents", {})
    for key in sorted(set(route_parents) | set(observed_parents)):
        if route_parents.get(key) != observed_parents.get(key):
            lines.append(
                f"  parent {key}: route={route_parents.get(key, '<absent>')[:12]}… "
                f"artifact={observed_parents.get(key, '<absent>')[:12]}…"
            )
    header = (
        f"artifact {artifact_type} build_spec does not match the current "
        f"route (artifact={binding.get('build_spec_hash', '?')[:12]}…, "
        f"route={ROUTE_BUILD_SPEC_HASHES[artifact_type][:12]}…)"
    )
    if lines:
        return header + "\n" + "\n".join(lines)
    return header
