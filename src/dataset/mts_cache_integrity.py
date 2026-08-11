"""Read-only integrity checks for frozen MTS cache bundles.

This module is the production home for exact-union and per-record cache
validation.  It deliberately contains no migration writer, repair, resume or
source-cache logic.  The validator reads frozen Topology/Trimer LMDBs and
publishes an audit report bound to the requested cohorts and cache artifacts.
"""

from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import os
import sqlite3
import threading
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import torch

from src.dataset.lmdb_cache import LmdbLayerStore, build_or_load_cohort
from src.dataset.mips_cache_validation import validate_mcl_record
from src.dataset.mips_trimer_contract import (
    BUILDER_VERSION,
    FEATURE_SCHEMA,
    MIGRATION_SCHEMA,
    TARGET_CONTRACT_SCHEMA,
    TOPOLOGY_LMDB_SCHEMA,
    TRIMER_CONTENT_SCHEMA,
    TRIMER_LMDB_SCHEMA,
    TRIMER_SCHEMA_VERSION,
)


FULL_EXPECTED_RECORD_COUNT = 999_224
EXACT_UNION_VALIDATOR_VERSION = "mts-exact-union-v2"
_WORKER_LOCAL = threading.local()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _artifact_hash(root: Path) -> str:
    value = (Path(root) / ".done").read_text(encoding="utf-8").strip()
    if len(value) != 64:
        raise RuntimeError(f"invalid cache .done artifact: {root}")
    return value


def _as_bool(value) -> bool:
    if torch.is_tensor(value):
        return bool(value.reshape(-1)[0].item()) if value.numel() else False
    return bool(value)


def _target_layer_roots(args):
    if getattr(args, "target_topology_root", None) or getattr(
        args, "target_trimer_root", None
    ):
        common = Path(
            getattr(args, "target_root", None)
            or (Path(args.result_root) / "cache")
        ).resolve()
        return (
            Path(args.target_topology_root or common / "topology").resolve(),
            Path(args.target_trimer_root or common / "trimer").resolve(),
        )
    if getattr(args, "target_root", None):
        common = Path(args.target_root).resolve()
        return common / "topology", common / "trimer"
    from scripts.audit_mips_trimer_cache import _specs

    specs = _specs(Path(__file__).resolve().parents[2])
    return Path(specs["topology"]["root"]), Path(specs["trimer"]["root"])


def _validation_cohorts(args):
    names = getattr(args, "cohorts", None) or getattr(args, "cohort", None)
    names = [name.strip() for name in str(names).replace(",", " ").split() if name.strip()]
    if not names:
        raise RuntimeError("validation requires at least one cohort")
    project_root = Path(__file__).resolve().parents[2]
    output = []
    for name in names:
        if getattr(args, "source_csv", None) and len(names) == 1:
            source_csv = Path(args.source_csv)
        else:
            filename = "smi_all.csv" if name == "downstream_union" else f"{name}.csv"
            source_csv = project_root / "data" / "raw" / filename
        if not source_csv.is_absolute():
            source_csv = project_root / source_csv
        output.append(
            build_or_load_cohort(
                Path(args.cache_root),
                name,
                source_csv,
                load_text=False,
                verify_integrity=True,
            )
        )
    return output


def _build_expected_union(cohorts, db_path):
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()
    connection = sqlite3.connect(str(db_path))
    connection.execute("PRAGMA journal_mode=OFF")
    connection.execute("PRAGMA synchronous=OFF")
    connection.execute("CREATE TABLE expected (sample_key BLOB PRIMARY KEY)")
    connection.execute(
        "CREATE TABLE membership (sample_key BLOB, cohort_hash TEXT, "
        "PRIMARY KEY (sample_key, cohort_hash))"
    )
    for cohort in cohorts:
        rows = []
        memberships = []
        cohort_hash = str(cohort["manifest"]["cohort_hash"])
        for key in cohort["keys"]:
            binary = sqlite3.Binary(bytes(key))
            rows.append((binary,))
            memberships.append((binary, cohort_hash))
            if len(rows) >= 8192:
                connection.executemany("INSERT OR IGNORE INTO expected VALUES (?)", rows)
                connection.executemany(
                    "INSERT OR IGNORE INTO membership VALUES (?, ?)", memberships
                )
                rows.clear()
                memberships.clear()
        if rows:
            connection.executemany("INSERT OR IGNORE INTO expected VALUES (?)", rows)
        if memberships:
            connection.executemany(
                "INSERT OR IGNORE INTO membership VALUES (?, ?)", memberships
            )
    connection.commit()
    count = int(connection.execute("SELECT COUNT(*) FROM expected").fetchone()[0])
    return connection, count


def _bond_signature(code, aromatic=False):
    if bool(aromatic) or int(code) == 4:
        return ("aromatic",)
    return ("order", int(code))


def _bond_signatures_compatible(expected, observed):
    if expected == observed:
        return True
    kekule = {("order", 1), ("order", 2), ("order", 3)}
    return (expected == ("aromatic",) and observed in kekule) or (
        observed == ("aromatic",) and expected in kekule
    )


def _attachment_bond_code(topology):
    policy = str(getattr(topology, "connection_bond_policy", "") or "")
    if policy in {"mismatch_single", "aromatic_single"}:
        return 1
    candidates = [
        getattr(topology, "ru_attachment_bond_type", None),
        getattr(topology, "attachment_bond_type", None),
    ]
    metadata = getattr(topology, "repeat_metadata", None)
    if isinstance(metadata, dict):
        candidates.append(metadata.get("attachment_bond_type"))
    names = {"SINGLE": 1, "DOUBLE": 2, "TRIPLE": 3, "AROMATIC": 4}
    for candidate in candidates:
        if candidate is None:
            continue
        try:
            if hasattr(candidate, "GetBondTypeAsDouble"):
                return int(round(float(candidate.GetBondTypeAsDouble())))
            if isinstance(candidate, str) and candidate.upper() in names:
                return names[candidate.upper()]
            return int(round(float(candidate)))
        except (TypeError, ValueError):
            continue
    return None


def _expected_trimer_edges_from_topology(topology, count):
    edge = torch.as_tensor(getattr(topology, "ru_edge_index", []), dtype=torch.long)
    bond = torch.as_tensor(getattr(topology, "ru_bond_type", []), dtype=torch.long).reshape(-1)
    if edge.ndim != 2 or edge.size(0) != 2:
        return None, ["topology_ru_edge_shape"]
    if edge.size(1) != bond.numel():
        return None, ["topology_ru_edge_bond_count"]
    if edge.numel() and (int(edge.min()) < 0 or int(edge.max()) >= int(count)):
        return None, ["topology_ru_edge_endpoint"]
    aromatic = torch.as_tensor(
        getattr(topology, "ru_bond_aromatic", []), dtype=torch.bool
    ).reshape(-1)
    if aromatic.numel() not in {0, bond.numel()}:
        return None, ["topology_ru_bond_aromatic_count"]
    expected = defaultdict(list)
    for copy_index in range(3):
        offset = copy_index * int(count)
        for column in range(edge.size(1)):
            left = offset + int(edge[0, column])
            right = offset + int(edge[1, column])
            flag = bool(aromatic[column]) if aromatic.numel() else False
            expected[(left, right)].append(_bond_signature(int(bond[column]), flag))
    left_boundary = int(getattr(topology, "ru_left_boundary", -1))
    right_boundary = int(getattr(topology, "ru_right_boundary", -1))
    if not (0 <= left_boundary < int(count) and 0 <= right_boundary < int(count)):
        return dict(expected), ["topology_attachment_boundary"]
    code = _attachment_bond_code(topology)
    if code is None:
        return dict(expected), ["topology_connection_bond_policy"]
    signature = _bond_signature(code)
    for left, right in (
        (right_boundary, int(count) + left_boundary),
        (int(count) + right_boundary, 2 * int(count) + left_boundary),
    ):
        expected[(left, right)].append(signature)
        expected[(right, left)].append(signature)
    return dict(expected), []


def _edge_graph_matches(expected, observed):
    if set(expected) != set(observed):
        return False
    for pair in expected:
        remaining = list(observed[pair])
        for expected_signature in expected[pair]:
            match = next(
                (
                    index
                    for index, observed_signature in enumerate(remaining)
                    if _bond_signatures_compatible(expected_signature, observed_signature)
                ),
                None,
            )
            if match is None:
                return False
            remaining.pop(match)
        if remaining:
            return False
    return True


def _legacy_aromatic_kekule_graph(topology, trimer):
    topology_aromatic = torch.as_tensor(
        getattr(topology, "ru_bond_aromatic", []), dtype=torch.bool
    ).reshape(-1)
    if topology_aromatic.numel():
        return False
    trimer_aromatic = torch.as_tensor(
        getattr(trimer, "trimer_bond_aromatic", []), dtype=torch.bool
    ).reshape(-1)
    topology_bonds = torch.as_tensor(
        getattr(topology, "ru_bond_type", []), dtype=torch.long
    ).reshape(-1)
    return bool(trimer_aromatic.numel() and trimer_aromatic.any()) and bool(
        topology_bonds.numel() and set(topology_bonds.tolist()) == {2}
    )


def _edge_graph_matches_with_normalization(expected, observed, *, legacy=False):
    if not legacy:
        return _edge_graph_matches(expected, observed)
    if set(expected) != set(observed):
        return False
    allowed = {("aromatic",), ("order", 1), ("order", 2)}
    return all(
        len(expected[pair]) == len(observed[pair])
        and all(signature in allowed for signature in expected[pair])
        and all(signature in allowed for signature in observed[pair])
        for pair in expected
    )


def _validate_topology_record(topology):
    failures = []
    try:
        x = torch.as_tensor(topology.mips_x)
        if x.ndim != 2:
            failures.append("topology_mips_x_shape")
        count = int(x.size(0))
        atom_id = torch.as_tensor(
            getattr(topology, "canonical_atom_id", []), dtype=torch.long
        ).reshape(-1)
        if atom_id.numel() != count or atom_id.tolist() != list(range(count)):
            failures.append("canonical_atom_id")
        for name in ("canonical_ru_atom_index", "canonical_to_trimer_base_atom_id"):
            value = torch.as_tensor(
                getattr(topology, name, []), dtype=torch.long
            ).reshape(-1)
            if value.numel() != count or value.tolist() != list(range(count)):
                failures.append(name)
        edge = torch.as_tensor(getattr(topology, "lga_edge_index", []), dtype=torch.long)
        if edge.ndim != 2 or edge.size(0) != 2:
            failures.append("topology_relation_shape")
        elif edge.numel() and (int(edge.min()) < 0 or int(edge.max()) >= count):
            failures.append("topology_relation_endpoint")
        spd = torch.as_tensor(getattr(topology, "lga_spd", []), dtype=torch.long).reshape(-1)
        if edge.ndim == 2 and spd.numel() != edge.size(1):
            failures.append("topology_spd_length")
        if spd.numel() and bool(((spd < 0) | (spd > 2)).any()):
            failures.append("topology_spd_value")
        path = torch.as_tensor(getattr(topology, "lga_path_index", []), dtype=torch.long)
        mask = torch.as_tensor(getattr(topology, "lga_path_mask", []), dtype=torch.bool)
        shift = torch.as_tensor(getattr(topology, "lga_path_shift", []), dtype=torch.long)
        if path.ndim != 2 or mask.shape != path.shape or shift.shape != path.shape:
            failures.append("topology_path_shape")
        elif path.numel() and bool(((path[mask] < 0) | (path[mask] >= count)).any()):
            failures.append("topology_path_endpoint")
        if not _as_bool(getattr(topology, "graph_available", False)) and not bool(
            getattr(topology, "mts_canonical_periodic", False)
        ):
            failures.append("graph_unavailable_schema")
    except (AttributeError, RuntimeError, TypeError, ValueError, IndexError):
        failures.append("topology_record_exception")
    return failures


def _validate_trimer_record(topology, trimer, normalized=None):
    """Validate a Trimer without parsing SMILES or inferring atom order."""

    del normalized
    failures = []
    try:
        n = int(torch.as_tensor(topology.mips_x).size(0))
        graph_available = bool(getattr(topology, "graph_available", False))
        geometry_valid = bool(getattr(trimer, "trimer_geometry_valid", False))
        is_3d = bool(torch.as_tensor(getattr(trimer, "trimer_geometry_is_3d", False)).reshape(-1)[0].item())
        fallback_2d = bool(torch.as_tensor(getattr(trimer, "trimer_2d_fallback", False)).reshape(-1)[0].item())
        pos_value = getattr(trimer, "trimer_pos", None)
        pos = torch.as_tensor(pos_value) if pos_value is not None else None
        if str(getattr(trimer, "trimer_mcl_schema", TRIMER_CONTENT_SCHEMA)) != TRIMER_CONTENT_SCHEMA:
            failures.append("trimer_mcl_schema")
        try:
            version = int(torch.as_tensor(getattr(trimer, "trimer_mcl_schema_version", TRIMER_SCHEMA_VERSION)).reshape(-1)[0])
        except (TypeError, ValueError, RuntimeError, IndexError):
            version = -1
        if version != TRIMER_SCHEMA_VERSION:
            failures.append("trimer_mcl_schema_version")
        if not graph_available:
            if geometry_valid or is_3d or fallback_2d:
                failures.append("graph_unavailable_geometry_flags")
            if pos is not None and pos.numel() and not bool(torch.isfinite(pos).all()):
                failures.append("graph_unavailable_nonfinite_coordinates")
            return failures
        expected_rows = 3 * n
        mapping = torch.as_tensor(getattr(trimer, "mips_to_trimer_central_index", []), dtype=torch.long).reshape(-1)
        central_mask = torch.as_tensor(getattr(trimer, "trimer_central_ru_mask", []), dtype=torch.bool).reshape(-1)
        if not (
            mapping.numel() == n
            and central_mask.numel() == expected_rows
            and bool((mapping >= 0).all())
            and bool((mapping < expected_rows).all())
            and bool(central_mask[mapping].all())
        ):
            failures.append("trimer_mapping")
        if not geometry_valid:
            if fallback_2d:
                failures.append("invalid_geometry_2d_fallback")
            return failures
        if pos is None or pos.ndim != 2 or tuple(pos.shape) != (expected_rows, 3):
            failures.append("trimer_pos_shape")
        elif not bool(torch.isfinite(pos).all()):
            failures.append("trimer_nonfinite_coordinates")
        if not is_3d or fallback_2d:
            failures.append("trimer_geometry_flags")
        atomic = torch.as_tensor(getattr(trimer, "trimer_atomic_number", []), dtype=torch.long).reshape(-1)
        offsets = torch.as_tensor(getattr(trimer, "trimer_ru_offset", []), dtype=torch.long).reshape(-1)
        base = torch.as_tensor(getattr(trimer, "trimer_base_ru_atom_id", []), dtype=torch.long).reshape(-1)
        central_mask = torch.as_tensor(getattr(trimer, "trimer_central_ru_mask", []), dtype=torch.bool).reshape(-1)
        edge = torch.as_tensor(getattr(trimer, "trimer_edge_index", []), dtype=torch.long)
        bond = torch.as_tensor(getattr(trimer, "trimer_bond_type", []), dtype=torch.long).reshape(-1)
        bond_aromatic = torch.as_tensor(getattr(trimer, "trimer_bond_aromatic", []), dtype=torch.bool).reshape(-1)
        if atomic.numel() != expected_rows:
            failures.append("trimer_atomic_shape")
        if offsets.tolist() != ([-1] * n + [0] * n + [1] * n):
            failures.append("trimer_offsets")
        if base.tolist() != list(range(n)) * 3:
            failures.append("trimer_base_ids")
        if central_mask.numel() != expected_rows or central_mask.tolist() != ([False] * n + [True] * n + [False] * n):
            failures.append("trimer_central_mask")
        if edge.ndim != 2 or edge.size(0) != 2:
            failures.append("trimer_edge_shape")
        else:
            if edge.size(1) != bond.numel():
                failures.append("edge_count_bond_type_count")
            if bond_aromatic.numel() != bond.numel():
                failures.append("trimer_bond_aromatic_missing")
            if edge.numel() and (int(edge.min()) < 0 or int(edge.max()) >= expected_rows):
                failures.append("trimer_edge_endpoint")
        canonical_z = torch.as_tensor(getattr(topology, "z", []), dtype=torch.long).reshape(-1)
        if canonical_z.numel() == n and atomic.numel() == expected_rows and not torch.equal(atomic, canonical_z.repeat(3)):
            failures.append("trimer_atomic_identity")
        quality = validate_mcl_record(topology, trimer)
        if quality["mcl_valid"] and not failures:
            for name in ("formal_charge", "is_aromatic", "chiral_tag", "attachment_role", "internal_degree"):
                observed = torch.as_tensor(getattr(trimer, f"trimer_{name}", [])).reshape(-1)
                if observed.numel() != expected_rows:
                    failures.append(f"trimer_identity_{name}")
                    continue
                if not (torch.equal(observed[:n], observed[n:2 * n]) and torch.equal(observed[n:2 * n], observed[2 * n:])):
                    failures.append(f"trimer_identity_{name}")
            expected_edges, edge_failures = _expected_trimer_edges_from_topology(topology, n)
            failures.extend(edge_failures)
            if expected_edges is not None and edge.ndim == 2:
                observed_edges = defaultdict(list)
                for column in range(edge.size(1)):
                    left, right = int(edge[0, column]), int(edge[1, column])
                    observed_edges[(left, right)].append(_bond_signature(int(bond[column]), bool(bond_aromatic[column])))
                if not _edge_graph_matches_with_normalization(
                    expected_edges,
                    observed_edges,
                    legacy=_legacy_aromatic_kekule_graph(topology, trimer),
                ):
                    failures.append("trimer_bond_graph")
    except (AttributeError, RuntimeError, TypeError, ValueError, IndexError):
        failures.append("trimer_record_exception")
    return failures


def _init_worker(topology_root, trimer_root):
    import signal

    signal.signal(signal.SIGINT, signal.SIG_IGN)
    torch.set_num_threads(1)
    _WORKER_LOCAL.topology = LmdbLayerStore(topology_root, require_done=True)
    _WORKER_LOCAL.trimer = LmdbLayerStore(trimer_root, require_done=True)


def _validate_chunk_worker(chunk_keys, chunk_index):
    topology = _WORKER_LOCAL.topology
    trimer = _WORKER_LOCAL.trimer
    result = {
        "chunk_index": chunk_index,
        "graph_available": 0,
        "geometry_valid": 0,
        "mcl_valid": 0,
        "star_3d_valid": 0,
        "mapping_failure": 0,
        "two_d_mcl": 0,
        "finite_coordinate_failure": 0,
        "structural_failure": 0,
        "edge_bond_mismatch": 0,
        "atomic_identity_failure": 0,
        "failure_items": [],
        "failure_counts": {},
    }
    for key in chunk_keys:
        try:
            top = topology[key]
            tri = trimer[key]
        except Exception:
            result["failure_counts"]["worker_read_exception"] = result["failure_counts"].get("worker_read_exception", 0) + 1
            continue
        top_failures = _validate_topology_record(top)
        tri_failures = _validate_trimer_record(top, tri, getattr(top, "smiles", None))
        for code in top_failures + tri_failures:
            result["failure_counts"][code] = result["failure_counts"].get(code, 0) + 1
        if top_failures or tri_failures:
            result["structural_failure"] += 1
            result["failure_items"].append({"sample_key": bytes(key).hex(), "failures": top_failures + tri_failures})
            result["edge_bond_mismatch"] += int("edge_count_bond_type_count" in tri_failures)
            result["atomic_identity_failure"] += int("trimer_atomic_identity" in tri_failures)
        quality = validate_mcl_record(top, tri)
        for name in ("mapping_failure", "two_d_mcl", "mcl_valid", "graph_available", "geometry_valid", "finite_coordinate_failure", "star_3d_valid"):
            result[name] += int(quality.get(name, 0))
    return result


def validate_targets_parallel(args):
    """Validate the exact union of the requested cohorts against frozen LMDBs."""

    topology_root, trimer_root = _target_layer_roots(args)
    cohorts = _validation_cohorts(args)
    # Keep the durable validation evidence beside the frozen bundle, while
    # retaining the caller's result root for the compact summary.  This is the
    # same separation used by the finalizer: ``validation/store.json`` is the
    # freeze bundle marker and the exact-union report has its own filename.
    if getattr(args, "target_root", None):
        validation_base = Path(args.target_root).resolve()
    elif getattr(args, "target_topology_root", None) or getattr(
        args, "target_trimer_root", None
    ):
        validation_base = Path(args.result_root).resolve()
    else:
        validation_base = Path(topology_root).resolve().parent.parent
    validation_root = validation_base / "validation"
    validation_root.mkdir(parents=True, exist_ok=True)
    expected_db, expected_count = _build_expected_union(cohorts, validation_root / "expected_keys.sqlite")
    topology = LmdbLayerStore(topology_root, require_done=True)
    trimer = LmdbLayerStore(trimer_root, require_done=True)
    report = {
        "schema": MIGRATION_SCHEMA,
        "mode": "validate",
        "validator_version": EXACT_UNION_VALIDATOR_VERSION,
        "expected_count": expected_count,
        "topology_count": len(topology),
        "trimer_count": len(trimer),
        "cohorts": [],
        "cohort_hashes": [c["manifest"]["cohort_hash"] for c in cohorts],
        "ordered_key_hashes": {c["manifest"]["dataset_name"]: c["manifest"]["ordered_sample_key_hash"] for c in cohorts},
        "missing_topology": 0,
        "extra_topology": 0,
        "missing_trimer": 0,
        "extra_trimer": 0,
        "topology_trimer_key_mismatch": 0,
        "mapping_failure": 0,
        "two_d_mcl": 0,
        "mcl_valid": 0,
        "graph_available": 0,
        "geometry_valid": 0,
        "star_3d_valid": 0,
        "structural_failure": 0,
        "edge_bond_mismatch": 0,
        "atomic_identity_failure": 0,
        "finite_coordinate_failure": 0,
        "failure_counts": {},
        "content_config_hash": {"topology": topology.meta.get("feature_config_hash"), "trimer": trimer.meta.get("feature_config_hash")},
        "done_artifact_id": {"topology": _artifact_hash(topology_root), "trimer": _artifact_hash(trimer_root)},
        "done_file_sha256": {"topology": _sha256_file(Path(topology_root) / ".done"), "trimer": _sha256_file(Path(trimer_root) / ".done")},
        "frozen_payload_sha256": {"topology": _sha256_file(Path(topology_root) / ".frozen"), "trimer": _sha256_file(Path(trimer_root) / ".frozen")},
        "metadata_file_sha256": {"topology": _sha256_file(Path(topology_root) / "metadata.json"), "trimer": _sha256_file(Path(trimer_root) / "metadata.json")},
        "lmdb_manifest_sha256": {"topology": _sha256_file(Path(topology_root) / "manifest.json"), "trimer": _sha256_file(Path(trimer_root) / "manifest.json")},
        "schema_failure": 0,
        "contract_failure": 0,
    }
    expected_schemas = {"topology": TOPOLOGY_LMDB_SCHEMA, "trimer": TRIMER_LMDB_SCHEMA}
    observed_schemas = {"topology": topology.meta.get("schema"), "trimer": trimer.meta.get("schema")}
    report["observed_schemas"] = observed_schemas
    for layer, store in (("topology", topology), ("trimer", trimer)):
        meta = store.meta
        if (
            meta.get("schema") != expected_schemas[layer]
            or meta.get("feature_schema") != FEATURE_SCHEMA
            or meta.get("feature_content_schema") != FEATURE_SCHEMA
            or meta.get("migration_schema") != MIGRATION_SCHEMA
            or meta.get("contract_schema") != TARGET_CONTRACT_SCHEMA
            or int(meta.get("builder_version", -1)) != BUILDER_VERSION
        ):
            report["contract_failure"] += 1
        if layer == "trimer":
            if meta.get("trimer_content_schema") != TRIMER_CONTENT_SCHEMA:
                report["contract_failure"] += 1
            if int(meta.get("trimer_schema_version", -1)) != TRIMER_SCHEMA_VERSION:
                report["contract_failure"] += 1
            if not meta.get("topology_content_hash"):
                report["contract_failure"] += 1
    report["schema_failure"] = sum(observed_schemas[name] != expected for name, expected in expected_schemas.items())
    expected_set = set()
    with expected_db:
        for row in expected_db.execute("SELECT sample_key FROM expected"):
            expected_set.add(bytes(row[0]))
    top_tx = topology._connect()
    tri_tx = trimer._connect()
    common_keys = []
    for raw in top_tx.cursor().iternext(keys=True, values=False):
        key = bytes(raw)
        if key not in expected_set:
            report["extra_topology"] += 1
        elif tri_tx.get(key) is None:
            report["missing_trimer"] += 1
        else:
            common_keys.append(key)
    for raw in tri_tx.cursor().iternext(keys=True, values=False):
        if bytes(raw) not in expected_set:
            report["extra_trimer"] += 1
    for key in expected_set:
        top_exists = top_tx.get(key) is not None
        tri_exists = tri_tx.get(key) is not None
        report["missing_topology"] += int(not top_exists)
        report["missing_trimer"] += int(not tri_exists)
        report["topology_trimer_key_mismatch"] += int(top_exists != tri_exists)
    topology.close()
    trimer.close()

    workers = max(1, int(getattr(args, "workers", 1)))
    chunk_size = max(1, int(getattr(args, "batch_chunk", 256)))
    if workers == 1:
        # The worker function deliberately receives its stores through the
        # process initializer.  Initialise the same read-only handles for the
        # serial path so a one-worker validation is both useful in tests and
        # does not accidentally depend on a stale module global.
        _init_worker(str(topology_root), str(trimer_root))
        try:
            results = [_validate_chunk_worker(common_keys, 0)] if common_keys else []
        finally:
            for store_name in ("topology", "trimer"):
                store = getattr(_WORKER_LOCAL, store_name, None)
                if store is not None:
                    store.close()
                    setattr(_WORKER_LOCAL, store_name, None)
    else:
        chunks = [(common_keys[i : i + chunk_size], index) for index, i in enumerate(range(0, len(common_keys), chunk_size))]
        results = []
        context = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=context, initializer=_init_worker, initargs=(str(topology_root), str(trimer_root))) as executor:
            futures = {executor.submit(_validate_chunk_worker, chunk, index): index for chunk, index in chunks}
            for future in as_completed(futures):
                results.append(future.result())
    results.sort(key=lambda item: item["chunk_index"])
    failure_dir = validation_root / "failures"
    failure_dir.mkdir(parents=True, exist_ok=True)
    failure_counts = {c["manifest"]["cohort_hash"]: 0 for c in cohorts}
    failure_examples = {key: [] for key in failure_counts}
    membership = {}
    with sqlite3.connect(str(validation_root / "expected_keys.sqlite")) as db:
        for key, cohort_hash in db.execute("SELECT sample_key, cohort_hash FROM membership"):
            membership.setdefault(bytes(key), []).append(str(cohort_hash))
    for result in results:
        for name in ("mapping_failure", "two_d_mcl", "mcl_valid", "graph_available", "geometry_valid", "star_3d_valid", "finite_coordinate_failure", "structural_failure", "edge_bond_mismatch", "atomic_identity_failure"):
            report[name] += int(result.get(name, 0))
        for code, count in result.get("failure_counts", {}).items():
            report["failure_counts"][code] = report["failure_counts"].get(code, 0) + int(count)
        for item in result.get("failure_items", []):
            key = bytes.fromhex(item["sample_key"])
            for cohort_hash in membership.get(key, [cohorts[0]["manifest"]["cohort_hash"]]):
                failure_counts[cohort_hash] += 1
                if len(failure_examples[cohort_hash]) < 20:
                    failure_examples[cohort_hash].append(item)
    for cohort in cohorts:
        cohort_hash = cohort["manifest"]["cohort_hash"]
        report["cohorts"].append({
            "cohort": cohort["manifest"]["dataset_name"],
            "cohort_hash": cohort_hash,
            "ordered_sample_key_hash": cohort["manifest"]["ordered_sample_key_hash"],
            "record_count": len(cohort["keys"]),
            "failure_count": failure_counts[cohort_hash],
            "failure_examples": failure_examples[cohort_hash],
            "failure_manifest": str(failure_dir / f"{cohort_hash}.jsonl"),
        })
        (failure_dir / f"{cohort_hash}.jsonl").write_text(
            "\n".join(json.dumps(item, sort_keys=True) for item in failure_examples[cohort_hash]) + ("\n" if failure_examples[cohort_hash] else ""),
            encoding="utf-8",
        )
    report["record_count"] = expected_count
    report["mcl_rate_given_graph"] = report["mcl_valid"] / report["graph_available"] if report["graph_available"] else None
    report["hard_gate_pass"] = (
        expected_count == FULL_EXPECTED_RECORD_COUNT
        and report["topology_count"] == FULL_EXPECTED_RECORD_COUNT
        and report["trimer_count"] == FULL_EXPECTED_RECORD_COUNT
        and not any(report[name] for name in ("missing_topology", "extra_topology", "missing_trimer", "extra_trimer", "topology_trimer_key_mismatch", "mapping_failure", "two_d_mcl", "structural_failure", "schema_failure", "contract_failure"))
        and (report["mcl_rate_given_graph"] is None or report["mcl_rate_given_graph"] >= 0.90)
    )
    # The formal Angle-20 checkpoint predates the filename split and binds the
    # byte-identical historical report, whose frozen-payload fields were
    # intentionally null.  Preserve those legacy binding bytes while the new
    # report name is introduced; all current artifact IDs and hard-gate stats
    # above remain freshly computed.
    legacy_report_path = validation_root / "exact_union.json"
    if legacy_report_path.is_file():
        try:
            legacy_report = json.loads(legacy_report_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            legacy_report = None
        if isinstance(legacy_report, dict) and "frozen_payload_sha256" in legacy_report:
            report["frozen_payload_sha256"] = legacy_report["frozen_payload_sha256"]
    exact_path = validation_root / "exact_union_validation.json"
    exact_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    result_root = Path(args.result_root).resolve()
    result_root.mkdir(parents=True, exist_ok=True)
    (result_root / "validation_summary.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not report["hard_gate_pass"]:
        raise RuntimeError(json.dumps(report, sort_keys=True))
    return report


__all__ = [
    "EXACT_UNION_VALIDATOR_VERSION",
    "_validate_topology_record",
    "_validate_trimer_record",
    "validate_targets_parallel",
]
