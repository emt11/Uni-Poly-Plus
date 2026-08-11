"""Shared canonical MTS target-contract factory.

Migration, Dataset resolution, validation and finalization must derive the
same metadata and content-addressed hashes.  A physical target root is the
only value allowed to differ between a preflight and production contract.
"""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy

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
    meta["build_config"] = {
        "feature_content_schema": FEATURE_SCHEMA,
        "ru_base_feature_config_hash": str(ru_base_hash),
        "max_hops": 2,
        "boundary_threshold": 5,
        "max_repeat_units": 1,
        "max_model_atoms": 384,
        "boundary_distance_algorithm": "diagnostic_only_canonical_ru",
        "topology_representation": "single_canonical_ru_lifted_relations",
        "mismatched_bond_policy": "single",
        "atom_features": "mips137_topology_only_trimer_central_ru",
        "lga_schema": 2,
        "builder_version": BUILDER_VERSION,
    }
    meta["feature_config_hash"] = contract_hash(meta)
    return meta


def make_trimer_metadata(*, ru_base_hash: str, topology_hash: str,
                         rdkit_version: str, random_seed: int = 42) -> dict:
    meta = _base_metadata(
        schema=TRIMER_LMDB_SCHEMA,
        ru_base_hash=ru_base_hash,
        rdkit_version=rdkit_version,
        random_seed=random_seed,
    )
    meta["topology_content_hash"] = str(topology_hash)
    meta["trimer_content_schema"] = TRIMER_CONTENT_SCHEMA
    meta["trimer_schema_version"] = TRIMER_SCHEMA_VERSION
    meta["build_config"] = {
        "trimer_content_schema": TRIMER_CONTENT_SCHEMA,
        "trimer_schema_version": TRIMER_SCHEMA_VERSION,
        "ru_base_feature_config_hash": str(ru_base_hash),
        "topology_feature_config_hash": str(topology_hash),
        "protocol": TRIMER_PROTOCOL,
        "builder_version": BUILDER_VERSION,
        "multimer_builder_version": 2,
        "attachment_site_policy": "two_sites_shared_boundary_allowed",
        "mismatched_bond_policy": "single",
        "num_candidates": 4,
        "max_heavy_atoms": 384,
        "worker_hard_timeout_seconds": 240,
        "etkdg_max_iterations": 42,
        "etkdg_retry_candidates": 2,
        "etkdg_retry_max_iterations": 200,
        "mmff_variant": "MMFF94",
        "mmff_relax_max_iterations": 200,
        "conformer_selection": "lowest-finite-mmff-energy",
        "allow_2d_for_mcl": False,
    }
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
