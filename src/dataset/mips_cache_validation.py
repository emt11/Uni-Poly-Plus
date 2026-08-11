"""Shared validation rules for the MTS feature cache."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .mips_trimer_contract import (
    CACHE_BUNDLE_SCHEMA,
    CACHE_MCL_THRESHOLD_SCHEMA,
    FEATURE_SCHEMA,
    MIGRATION_SCHEMA,
    TARGET_CONTRACT_SCHEMA,
    BUILDER_VERSION,
)


def _bool_value(value, default=False):
    if value is None:
        return bool(default)
    return bool(torch.as_tensor(value).bool().reshape(-1)[0].item())


def trimer_can_enter_mcl(data, graph_index=0):
    """Return the single runtime eligibility decision for Trimer-MCL.

    This function is deliberately shared by cache validation, pretraining and
    the encoder fallback path.  A valid flag by itself is not sufficient: a
    record must contain genuine 3-D coordinates and a complete O8-to-central
    RU mapping.  The function never raises for an ordinary bad sample; it
    returns ``False`` so the caller can use the exact O8 topology fallback.
    """

    try:
        graph_index = int(graph_index)
        graph_available = getattr(data, "graph_available", True)
        if graph_available is not True and torch.is_tensor(graph_available):
            if graph_index >= int(graph_available.numel()):
                return False
            if not bool(graph_available.reshape(-1)[graph_index].item()):
                return False
        elif not bool(graph_available):
            return False

        def sample_flag(name, default=False):
            value = getattr(data, name, default)
            if torch.is_tensor(value):
                flat = value.reshape(-1)
                if graph_index >= int(flat.numel()):
                    return bool(default)
                return bool(flat[graph_index].item())
            if isinstance(value, (list, tuple)):
                if graph_index >= len(value):
                    return bool(default)
                return bool(value[graph_index])
            return bool(value)

        if not sample_flag("trimer_geometry_valid"):
            return False
        if not sample_flag("trimer_geometry_is_3d"):
            return False
        if sample_flag("trimer_2d_fallback"):
            return False

        trimer_batch = getattr(data, "trimer_batch", None)
        positions = getattr(data, "trimer_pos", None)
        mapping = getattr(data, "mips_to_trimer_central_index", None)
        central_mask = getattr(data, "trimer_central_ru_mask", None)
        node_batch = getattr(data, "batch", None)
        if any(value is None for value in (positions, mapping, central_mask)):
            return False
        positions = torch.as_tensor(positions)
        if positions.ndim != 2 or positions.size(-1) != 3:
            return False
        if not bool(torch.isfinite(positions).all()):
            return False
        # LMDB records are unbatched and therefore do not carry the PyG
        # ``trimer_batch``/``batch`` vectors.  Treat their complete Trimer and
        # O8 node set as graph zero.  The collator adds the explicit vectors
        # before the same predicate is used for a multi-graph batch.
        if trimer_batch is None:
            trimer_batch = torch.zeros(positions.size(0), dtype=torch.long)
        else:
            trimer_batch = torch.as_tensor(trimer_batch).long().reshape(-1)
        if node_batch is None:
            node_batch = torch.zeros(
                int(torch.as_tensor(mapping).numel()), dtype=torch.long
            )
        atoms = torch.nonzero(trimer_batch == graph_index, as_tuple=False).flatten()
        nodes = torch.nonzero(
            torch.as_tensor(node_batch).long().reshape(-1) == graph_index,
            as_tuple=False,
        ).flatten()
        mapping = torch.as_tensor(mapping).long().reshape(-1)
        central_mask = torch.as_tensor(central_mask).bool().reshape(-1)
        if not atoms.numel() or mapping.numel() < int(nodes.numel()):
            return False
        graph_mapping = mapping[nodes]
        if graph_mapping.numel() != nodes.numel():
            return False
        if bool((graph_mapping < 0).any()) or bool(
            (graph_mapping >= positions.size(0)).any()
        ):
            return False
        if central_mask.numel() != positions.size(0):
            return False
        if not bool(torch.isin(graph_mapping, atoms).all()):
            return False
        return bool(central_mask[graph_mapping].all())
    except (AttributeError, IndexError, RuntimeError, TypeError, ValueError):
        return False


def validate_mcl_record(topology, trimer):
    """Return one canonical quality result for a topology/Trimer pair.

    The quality gate only answers whether a sample is safe for MCL.  It does
    not judge force-field convergence, energy, or Star-distance quality.
    """

    graph_available = _bool_value(
        getattr(topology, "graph_available", False)
    )
    geometry_valid = _bool_value(
        getattr(trimer, "trimer_geometry_valid", False)
    )
    is_3d = _bool_value(getattr(trimer, "trimer_geometry_is_3d", False))
    is_2d_fallback = _bool_value(
        getattr(trimer, "trimer_2d_fallback", False)
    )
    # This counter is specifically "2-D records entering MCL", not all
    # invalid geometry records.  A graph-unavailable sample is never sent to
    # MCL and therefore must not fail this invariant.
    two_d_mcl = graph_available and geometry_valid and (
        not is_3d or is_2d_fallback
    )

    positions = getattr(trimer, "trimer_pos", None)
    positions_tensor = (
        torch.as_tensor(positions) if positions is not None else None
    )
    finite_failure = bool(
        geometry_valid
        and (
            positions_tensor is None
            or positions_tensor.numel() == 0
            or positions_tensor.ndim != 2
            or positions_tensor.size(-1) != 3
            or not bool(torch.isfinite(positions_tensor).all())
        )
    )

    mapping = getattr(trimer, "mips_to_trimer_central_index", None)
    mapping_tensor = (
        torch.as_tensor(mapping).long() if mapping is not None else None
    )
    raw_node_count = getattr(topology, "num_nodes", 0)
    node_count = int(raw_node_count or 0)
    trimer_count = int(
        positions_tensor.size(0)
        if positions_tensor is not None and positions_tensor.ndim > 0 else 0
    )
    mapping_failure = False
    central_mapping_failure = False
    if geometry_valid and graph_available:
        mapping_failure = bool(
            mapping_tensor is None
            or int(mapping_tensor.numel()) != node_count
            or bool((mapping_tensor < 0).any())
            or bool((mapping_tensor >= trimer_count).any())
        )
        if not mapping_failure:
            central_values = getattr(trimer, "trimer_central_ru_mask", None)
            if central_values is None:
                central_mapping_failure = True
            else:
                central_mask = torch.as_tensor(central_values).bool().reshape(-1)
                if int(central_mask.numel()) != trimer_count:
                    central_mapping_failure = True
                else:
                    central_mapping_failure = bool(
                        (~central_mask[mapping_tensor]).any()
                    )

    # Keep the batch-independent implementation above for the detailed
    # diagnostics, while using the shared eligibility predicate for the final
    # decision.  This prevents Dataset, audit and pretraining from silently
    # drifting apart.
    mcl_valid = bool(
        graph_available
        and geometry_valid
        and is_3d
        and not is_2d_fallback
        and not finite_failure
        and not mapping_failure
        and not central_mapping_failure
    )
    return {
        "graph_available": graph_available,
        "geometry_valid": geometry_valid,
        "mcl_valid": mcl_valid,
        "two_d_mcl": int(two_d_mcl),
        "mapping_failure": int(mapping_failure or central_mapping_failure),
        "finite_coordinate_failure": int(finite_failure),
        "star_3d_valid": int(_bool_value(
            getattr(trimer, "star_3d_valid", False)
        )),
    }


def _json_digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_md200_artifacts(md_spec, downstream, cache_root):
    """Verify the production MD200 LMDB and its downstream mmap pair."""
    md_root = Path(md_spec["root"])
    manifest_path = md_root / "manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError("MD200 manifest is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    count = int(manifest.get("count", 0))
    expected_count = int(downstream.get("record_count", 0))
    if count < expected_count:
        raise RuntimeError(
            f"MD200 LMDB coverage is incomplete: {count} < {expected_count}"
        )
    cohort_hash = downstream.get("cohort_hash")
    cohort_dir = Path(cache_root) / "cohorts" / "downstream_union" / str(cohort_hash)
    if not cohort_dir.is_dir():
        raise RuntimeError("downstream cohort directory is missing")
    candidates = sorted(cohort_dir.glob("md200_*/md200_metadata.json"))
    if not candidates:
        raise RuntimeError("production MD200 mmap metadata is missing")
    observed = None
    for metadata_path in candidates:
        value = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (
            value.get("cohort_hash") == cohort_hash
            and value.get("ordered_sample_key_hash")
            == downstream.get("ordered_sample_key_hash")
            and value.get("shape") == [expected_count, 200]
            and value.get("dtype") == "float32"
        ):
            observed = (metadata_path, value)
            break
    if observed is None:
        raise RuntimeError("no MD200 mmap matches the downstream cohort")
    metadata_path, metadata = observed
    values_path = metadata_path.parent / "md200.npy"
    valid_path = metadata_path.parent / "md200_valid.npy"
    if not values_path.is_file() or not valid_path.is_file():
        raise RuntimeError("MD200 mmap data files are missing")
    values = np.load(values_path, mmap_mode="r")
    valid = np.load(valid_path, mmap_mode="r")
    if values.shape != (expected_count, 200) or values.dtype != np.float32:
        raise RuntimeError("MD200 values mmap has invalid shape/dtype")
    if valid.shape != (expected_count,) or valid.dtype != np.bool_:
        raise RuntimeError("MD200 validity mmap has invalid shape/dtype")
    if metadata.get("values_sha256") != _sha256_file(values_path):
        raise RuntimeError("MD200 values mmap hash mismatch")
    if metadata.get("valid_sha256") != _sha256_file(valid_path):
        raise RuntimeError("MD200 validity mmap hash mismatch")
    if metadata.get("shape") != [expected_count, 200]:
        raise RuntimeError("MD200 metadata shape mismatch")
    return {
        "metadata": metadata,
        "metadata_path": str(metadata_path),
        "values_sha256": metadata["values_sha256"],
        "valid_sha256": metadata["valid_sha256"],
    }


def verify_frozen_cache_bundle(specs, *, store_path, required_layers=None,
                               pretraining_cohort_hash=None):
    """Verify the immutable cache bundle before any training Dataset opens it."""
    required_layers = tuple(required_layers or specs.keys())
    store_path = Path(store_path)
    if not store_path.is_file():
        raise RuntimeError(f"cache store validation is missing: {store_path}")
    store = json.loads(store_path.read_text(encoding="utf-8"))
    if store.get("schema") != CACHE_BUNDLE_SCHEMA:
        raise RuntimeError("unsupported or stale cache store schema")
    if (
        store.get("migration_schema") != MIGRATION_SCHEMA
        or store.get("contract_schema") != TARGET_CONTRACT_SCHEMA
        or int(store.get("builder_version", -1)) != BUILDER_VERSION
        or not store.get("transaction_id")
    ):
        raise RuntimeError("cache bundle contract binding is missing or stale")
    if (
        pretraining_cohort_hash is not None
        and store.get("pretraining_cohort_hash") != pretraining_cohort_hash
    ):
        raise RuntimeError("cache store pretraining cohort hash mismatch")
    artifact_hashes = store.get("done_artifact_id", {})
    if not artifact_hashes:
        raise RuntimeError("cache bundle is missing done_artifact_id bindings")
    full_validation = store.get("full_validation", {})
    if full_validation.get("record_count") != 999224:
        raise RuntimeError("cache store does not contain exact-union validation")
    for key in ("mapping_failure", "two_d_mcl"):
        if int(full_validation.get(key, -1)) != 0:
            raise RuntimeError(f"cache full validation failed: {key}")
    full_rate = full_validation.get(
        "geometry_rate_given_graph",
        full_validation.get("mcl_rate_given_graph"),
    )
    if full_rate is None or float(full_rate) < 0.90:
        raise RuntimeError("cache full validation is below the 90% MCL gate")
    audit = store.get("audit", {})
    if int(audit.get("mapping_failure", -1)) != 0 or int(
        audit.get("two_d_mcl", -1)
    ) != 0:
        raise RuntimeError("cache 10k audit failed hard invariants")
    audit_rate = audit.get("geometry_rate_given_graph")
    if audit_rate is None or float(audit_rate) < 0.90:
        raise RuntimeError("cache 10k audit is below the 90% MCL gate")
    for name in required_layers:
        if name not in specs:
            raise RuntimeError(f"cache bundle has no layer specification: {name}")
        root = Path(specs[name]["root"])
        done = root / ".done"
        marker = root / ".frozen"
        manifest = root / "manifest.json"
        if not done.is_file() or not marker.is_file() or not manifest.is_file():
            raise RuntimeError(f"cache layer is not frozen and complete: {name}")
        done_artifact_id = done.read_text(encoding="utf-8").strip()
        done_file_sha256 = _sha256_file(done)
        if len(done_artifact_id) != 64 or artifact_hashes.get(name) != done_artifact_id:
            raise RuntimeError(f"cache layer artifact hash mismatch: {name}")
        payload = json.loads(marker.read_text(encoding="utf-8"))
        metadata_hash = _json_digest(specs[name]["meta"])
        manifest_hash = _json_digest(
            json.loads(manifest.read_text(encoding="utf-8"))
        )
        if payload.get("done_artifact_id") != done_artifact_id:
            raise RuntimeError(f"frozen marker done artifact mismatch: {name}")
        if payload.get("done_file_sha256") != done_file_sha256:
            raise RuntimeError(f"frozen marker done file mismatch: {name}")
        if (
            payload.get("layer") != name
            or payload.get("schema") != "mts-canonical-cache-freeze-v2"
            or payload.get("cache_layout_schema") != specs[name]["meta"].get(
                "cache_layout_schema"
            )
            or payload.get("metadata_hash") != metadata_hash
            or payload.get("manifest_hash") != manifest_hash
            or payload.get("contract_schema") != TARGET_CONTRACT_SCHEMA
            or int(payload.get("builder_version", -1)) != BUILDER_VERSION
            or not payload.get("transaction_id")
        ):
            raise RuntimeError(f"invalid frozen marker payload: {name}")
        if payload.get("transaction_id") != store.get("transaction_id"):
            raise RuntimeError(f"frozen marker transaction mismatch: {name}")
        meta = specs[name].get("meta", {})
        if name in {"topology", "trimer"} and (
            meta.get("migration_schema") != MIGRATION_SCHEMA
            or meta.get("contract_schema") != TARGET_CONTRACT_SCHEMA
            or int(meta.get("builder_version", -1)) != BUILDER_VERSION
            or meta.get("feature_schema") != FEATURE_SCHEMA
        ):
            raise RuntimeError(f"cache layer metadata contract mismatch: {name}")
        manifest_value = json.loads(manifest.read_text(encoding="utf-8"))
        if int(payload.get("record_count", -1)) != int(
            manifest_value.get("count", -2)
        ):
            raise RuntimeError(f"frozen marker count mismatch: {name}")
    downstream = store.get("downstream_union", {})
    downstream_hash = downstream.get("cohort_hash")
    if not downstream_hash:
        raise RuntimeError("cache store has no downstream union binding")
    # The persisted hard-mask thresholds are optional during graph-only
    # pretraining, but a frozen production bundle must contain both cohort
    # arrays for Stage 3's mmap fast path.  Bind them to the Trimer artifact
    # and ordered-key hash; a stale array is never silently reused.
    trimer_root = Path(specs["trimer"]["root"])
    cohort_root = trimer_root.parents[1] / "cohorts"
    # The exact-union report is the production key-set gate, while the
    # pretraining threshold mmap is row-ordered to the PI1M_v2 subset.  The
    # finalizer embeds that subset diagnostic under ``pretraining_subset`` so
    # threshold verification does not accidentally demand a 999,224-row PI1M
    # array.
    pretraining_threshold_report = full_validation.get(
        "pretraining_subset", full_validation
    )
    for report_name, report in (
        ("pretraining", pretraining_threshold_report),
        ("downstream", downstream),
    ):
        cohort_name = report.get(
            "cohort_name",
            "PI1M_v2" if report_name == "pretraining" else "downstream_union",
        )
        cohort_hash = report.get("cohort_hash")
        threshold_dir = cohort_root / str(cohort_name) / str(cohort_hash)
        threshold_path = threshold_dir / "mcl_thresholds.npy"
        threshold_meta_path = threshold_dir / "mcl_thresholds_metadata.json"
        if not threshold_path.is_file() or not threshold_meta_path.is_file():
            raise RuntimeError(
                f"{report_name} cohort is missing persisted MCL thresholds"
            )
        threshold_meta = json.loads(
            threshold_meta_path.read_text(encoding="utf-8")
        )
        expected_count = int(report.get("record_count", -1))
        if (
            threshold_meta.get("schema") != CACHE_MCL_THRESHOLD_SCHEMA
            or threshold_meta.get("cohort_hash") != cohort_hash
            or threshold_meta.get("ordered_sample_key_hash")
            != report.get("ordered_sample_key_hash")
            or threshold_meta.get("trimer_artifact_hash")
            != artifact_hashes.get("trimer")
            or threshold_meta.get("trimer_done_artifact_id")
            != artifact_hashes.get("trimer")
            or threshold_meta.get("trimer_done_file_sha256")
            != _sha256_file(trimer_root / ".done")
            or threshold_meta.get("trimer_contract_hash")
            != specs["trimer"]["meta"].get("feature_config_hash")
            or threshold_meta.get("shape") != [expected_count, 2]
        ):
            raise RuntimeError(
                f"{report_name} MCL threshold metadata is stale or inconsistent"
            )
        thresholds = np.load(threshold_path, mmap_mode="r")
        if tuple(thresholds.shape) != (expected_count, 2) \
                or thresholds.dtype != np.float32:
            raise RuntimeError(f"{report_name} MCL threshold shape mismatch")
    if "md200" in required_layers:
        md_spec = specs["md200"]
        md_manifest = json.loads(
            (Path(md_spec["root"]) / "manifest.json").read_text(
                encoding="utf-8"
            )
        )
        if int(md_manifest.get("count", 0)) < int(
            downstream.get("record_count", 0)
        ):
            raise RuntimeError("MD200 cache does not cover downstream union")
        cohort_dir = Path(md_spec["root"]).parents[1] / "cohorts" / "downstream_union"
        pointer = cohort_dir / "current.json"
        if not pointer.is_file():
            raise RuntimeError("downstream cohort pointer is missing")
        pointer_value = json.loads(pointer.read_text(encoding="utf-8"))
        if pointer_value.get("cohort_hash") != downstream_hash:
            raise RuntimeError("downstream cohort binding mismatch")
        _verify_md200_artifacts(
            md_spec,
            downstream,
            trimer_root.parents[1],
        )
    return store
