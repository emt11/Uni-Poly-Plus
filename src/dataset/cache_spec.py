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
    """Derive the factual build spec of an existing artifact.

    ``metadata`` is the artifact's ``metadata.json``.  Manual version counters
    are dropped (they describe nothing the parameters do not already
    determine), and legacy parent config hashes are re-bound to registered
    parent artifact types.  Everything else in ``build_config`` is kept
    verbatim: those are the parameters that actually produced the content.
    """

    build_config = metadata.get("build_config")
    if not isinstance(build_config, dict):
        raise StoreError(
            f"{artifact_type} metadata has no build_config mapping"
        )
    parameters = {}
    parents = {}
    dropped = set(_VERSION_METADATA_KEYS) | set(
        _PER_TYPE_METADATA_KEYS.get(str(artifact_type), ())
    )
    for key, value in build_config.items():
        if key in dropped:
            continue
        if key in _PARENT_METADATA_KEYS:
            parents[_PARENT_METADATA_KEYS[key]] = str(value)
            continue
        parameters[key] = value
    return validate_build_spec({
        "artifact_type": str(artifact_type),
        "parameters": parameters,
        "parents": parents,
    })


# ---------------------------------------------------------------------------
# Route-expected build specs.  These describe the CURRENT scientific protocol
# (MIPS-Trimer-SCAGE, ETKDGv3 + MMFF94 open trimer, 3 RU, explicit H).  They
# are derived from the same constants the builders use; no version counters
# appear here.  A protocol change edits these parameters directly.
# ---------------------------------------------------------------------------

RU_BASE_BUILD_SPEC = validate_build_spec({
    "artifact_type": "ru_base",
    "parameters": {
        "attachment_site_policy": "two_sites_shared_boundary_allowed",
        "boundary_distance_algorithm": "expanded_graph_shortest_path",
        "mismatched_bond_policy": "single",
        "molecule_binary": "rdkit_mol_binary",
        "normalization": "rdkit_canonical_psmiles_v1",
    },
    "parents": {},
})

TOPOLOGY_BUILD_SPEC = validate_build_spec({
    "artifact_type": "topology",
    "parameters": {
        "atom_features": "mips137_topology_only_trimer_central_ru",
        "boundary_distance_algorithm": "diagnostic_only_canonical_ru",
        "boundary_threshold": 5,
        "lga_schema": 2,
        "max_hops": 2,
        "max_model_atoms": 384,
        "max_repeat_units": 1,
        "mismatched_bond_policy": "single",
        "topology_representation": "single_canonical_ru_lifted_relations",
    },
    "parents": {"ru_base": build_spec_hash(RU_BASE_BUILD_SPEC)},
})

TRIMER_BUILD_SPEC = validate_build_spec({
    "artifact_type": "trimer",
    "parameters": {
        "allow_2d_for_mcl": False,
        "attachment_site_policy": "two_sites_shared_boundary_allowed",
        "conformer_selection": "converged-first-lowest-finite-mmff-energy",
        "coordinate_payload": "single-conformer-explicit-all-atom",
        "etkdg_enforce_chirality": True,
        "etkdg_max_iterations": 200,
        "etkdg_rmsd_pruning": False,
        "etkdg_use_random_coords": True,
        "max_heavy_atoms": 384,
        "max_rounds": 2,
        "mmff_relax_max_iterations": 200,
        "mmff_variant": "MMFF94",
        "num_candidates_per_round": 8,
        "worker_hard_timeout_seconds": 240,
    },
    "parents": {
        "ru_base": build_spec_hash(RU_BASE_BUILD_SPEC),
        "topology": build_spec_hash(TOPOLOGY_BUILD_SPEC),
    },
})

MD200_BUILD_SPEC = validate_build_spec({
    "artifact_type": "md200",
    "parameters": {
        "components": ["RDKit2DNormalized200"],
        "descriptor_schema": 5,
        "embedding": "none",
        "optimizer": "none",
        "protocol": "source_star_sub",
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
