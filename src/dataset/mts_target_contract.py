"""Shared canonical MTS target-contract factory.

Migration, Dataset resolution, validation and finalization must derive the
same metadata and content-addressed hashes.  A physical target root is the
only value allowed to differ between a preflight and production contract.
"""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy

from .cache_spec import (
    TOPOLOGY_BUILD_SPEC,
    TRIMER_BUILD_SPEC,
    build_spec_hash,
)

from .mips_trimer_contract import (
    BUILDER_VERSION,
    CACHE_LAYOUT_SCHEMA,
    FEATURE_SCHEMA,
    MIGRATION_SCHEMA,
    TARGET_CONTRACT_SCHEMA,
    TOPOLOGY_LMDB_SCHEMA,
    TRIMER_CONTENT_SCHEMA,
    TRIMER_LMDB_SCHEMA,
    TRIMER_PROTOCOL,
    TRIMER_SCHEMA_VERSION,
)


def contract_hash(value: dict) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _base_metadata(*, schema: str, ru_base_hash: str, rdkit_version: str,
                   random_seed: int) -> dict:
    return {
        "cache_layout_schema": CACHE_LAYOUT_SCHEMA,
        "contract_schema": TARGET_CONTRACT_SCHEMA,
        "migration_schema": MIGRATION_SCHEMA,
        "schema": schema,
        "feature_schema": FEATURE_SCHEMA,
        "feature_content_schema": FEATURE_SCHEMA,
        "ru_base_content_hash": str(ru_base_hash),
        "rdkit_version": str(rdkit_version),
        "random_seed": int(random_seed),
        "builder_version": BUILDER_VERSION,
        "mapping_protocol": (
            "formal-charge-aromatic-chiral-attachment-role-degree-v1"
        ),
    }


def make_topology_metadata(*, ru_base_hash: str, rdkit_version: str,
                           random_seed: int = 42) -> dict:
    meta = _base_metadata(
        schema=TOPOLOGY_LMDB_SCHEMA,
        ru_base_hash=ru_base_hash,
        rdkit_version=rdkit_version,
        random_seed=random_seed,
    )
    meta["build_spec"] = deepcopy(TOPOLOGY_BUILD_SPEC)
    meta["build_spec_hash"] = build_spec_hash(TOPOLOGY_BUILD_SPEC)
    meta["build_config"] = deepcopy(TOPOLOGY_BUILD_SPEC["parameters"])
    meta["feature_config_hash"] = contract_hash(meta)
    return meta


def make_trimer_metadata(*, ru_base_hash: str, topology_hash: str,
                         rdkit_version: str, random_seed: int = 42) -> dict:
    del topology_hash  # Trimer and Topology are sibling artifacts over RU/source.
    meta = _base_metadata(
        schema=TRIMER_LMDB_SCHEMA,
        ru_base_hash=ru_base_hash,
        rdkit_version=rdkit_version,
        random_seed=random_seed,
    )
    meta["trimer_content_schema"] = TRIMER_CONTENT_SCHEMA
    meta["trimer_schema_version"] = TRIMER_SCHEMA_VERSION
    meta["build_spec"] = deepcopy(TRIMER_BUILD_SPEC)
    meta["build_spec_hash"] = build_spec_hash(TRIMER_BUILD_SPEC)
    meta["build_config"] = deepcopy(TRIMER_BUILD_SPEC["parameters"])
    meta["feature_config_hash"] = contract_hash(meta)
    return meta


def make_target_contract(*, ru_base_hash: str, rdkit_version: str,
                         random_seed: int = 42) -> dict:
    topology = make_topology_metadata(
        ru_base_hash=ru_base_hash,
        rdkit_version=rdkit_version,
        random_seed=random_seed,
    )
    trimer = make_trimer_metadata(
        ru_base_hash=ru_base_hash,
        topology_hash=topology["feature_config_hash"],
        rdkit_version=rdkit_version,
        random_seed=random_seed,
    )
    return {
        "schema": TARGET_CONTRACT_SCHEMA,
        "feature_schema": FEATURE_SCHEMA,
        "migration_schema": MIGRATION_SCHEMA,
        "builder_version": BUILDER_VERSION,
        "ru_base_content_hash": str(ru_base_hash),
        "topology": deepcopy(topology),
        "trimer": deepcopy(trimer),
    }


__all__ = [
    "contract_hash",
    "make_target_contract",
    "make_topology_metadata",
    "make_trimer_metadata",
]
