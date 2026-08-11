#!/usr/bin/env python
"""Finalize and freeze the PI1M_v2 MTS feature cache.

This command is intentionally separate from training.  Downstream feature
preparation is performed by ``prepare_mips_trimer_downstream.py`` so this
process never starts a writer while holding the finalizer lock.  This command
only validates and freezes already-complete layers.
"""

from __future__ import annotations

import argparse
import atexit
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path

import numpy as np
import torch

# Cache validation is a sequential LMDB scan.  PyTorch's default intra-op
# pool (often one thread per host core) makes the tiny per-record quantile and
# mask operations much slower and inflates RSS.  Keep this utility bounded;
# training processes set their own thread policy separately.
torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.audit_mips_trimer_cache import _specs  # noqa: E402
from src.dataset.lmdb_cache import (  # noqa: E402
    LmdbLayerStore,
    build_or_load_cohort,
    mcl_thresholds_cached,
)
from src.dataset.mips_cache_validation import (  # noqa: E402
    validate_mcl_record,
    verify_frozen_cache_bundle,
)
from src.dataset.mips_trimer_contract import (  # noqa: E402
    BUILDER_VERSION,
    CACHE_BUNDLE_SCHEMA,
    CACHE_BOND_ANGLE_SCHEMA,
    CACHE_MCL_THRESHOLD_SCHEMA,
    MIGRATION_SCHEMA,
    TARGET_CONTRACT_SCHEMA,
)
from src.dataset.mts_target_contract import make_target_contract  # noqa: E402
from src.dataset.trimer_angle_cache import validate_angle_cache  # noqa: E402
from src.dataset.trimer_angle_continuous_cache import continuous_angle_root  # noqa: E402
from src.dataset.mts_cache_integrity import (  # noqa: E402
    EXACT_UNION_VALIDATOR_VERSION,
    validate_targets_parallel as validate_exact_union_parallel,
)

EXPECTED_COUNT = 999_224
PI1M_DATASET = "PI1M_v2"
OLD_ROOTS = (
    PROJECT_ROOT
    / "data/processed/mips_trimer_scage/ru_base/5616a1a66a1937acade1da2a2eced0f90d5e123d4f0c847411b6b20265857ea2",
    PROJECT_ROOT
    / "data/processed/mips_trimer_scage/topology/25671dde5f9c08b9676ff09d19face3681b828fe0040b905537bc77680647970",
)


def _json_hash(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _artifact_hash(root: Path) -> str:
    path = root / ".done"
    if not path.is_file():
        raise RuntimeError(f"missing cache .done: {root}")
    value = path.read_text(encoding="utf-8").strip()
    if len(value) != 64:
        raise RuntimeError(f"invalid cache .done hash: {root}")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _active_writer_pids():
    """Find cache writers without relying on a stale hard-coded PID."""

    current = os.getpid()
    output = subprocess.run(
        ["ps", "-eo", "pid=,args="],
        check=False,
        capture_output=True,
        text=True,
    ).stdout
    found = []
    for line in output.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            pid_text, command = line.split(None, 1)
            pid = int(pid_text)
        except (ValueError, TypeError):
            continue
        if pid == current:
            continue
        if (
            ("scripts/pretrain.py" in command or "scripts/train.py" in command)
            and "--cache_only" in command
        ):
            found.append((pid, command))
    return found


def _assert_no_writer_locks(specs):
    for name, spec in specs.items():
        lock_path = Path(spec["root"]) / ".writer.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("a+", encoding="utf-8") as handle:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError(
                    f"cache writer lock is active for {name}: {lock_path}"
                ) from exc
            finally:
                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                except OSError:
                    pass


def _prepare_downstream():
    """Verify the immutable downstream layers without invoking retired jobs.

    The former implementation delegated to ``run_mips_trimer_scage.sh``.
    That script now intentionally rejects the retired topology/geometry split,
    while the canonical migration already owns the immutable Topology/Trimer
    layers.  Preparation therefore means a read-only coverage check here;
    missing records are a hard error and are never silently rebuilt or cleared.
    """

    specs = _specs(PROJECT_ROOT)
    _assert_no_writer_locks(specs)
    report = _validate_downstream_union(specs)
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


def _load_cohort(source_csv: Path, name: str):
    cohort = build_or_load_cohort(
        PROJECT_ROOT / "data/processed/mips_trimer_scage",
        name,
        source_csv,
        load_text=False,
        verify_integrity=True,
    )
    if name == PI1M_DATASET and int(cohort["manifest"]["unique_count"]) != 995_799:
        raise RuntimeError(
            f"PI1M_v2 cohort count mismatch: "
            f"{cohort['manifest'].get('unique_count')} != 995799"
        )
    return cohort


def _validate_full_pi1m(cohort, specs):
    # Retired: the serial 999k PI1M rescan is forbidden by Plan.MD §7.  Full
    # validation always runs through the 32-worker parallel exact-union
    # validator (_validate_exact_union_store) or reuses its report (§9.2).
    raise SystemExit(
        "_validate_full_pi1m is retired; run the parallel exact-union validator "
        "or reuse its existing exact_union_validation.json report"
    )


def _assert_layer_integrity(specs):
    """Validate every layer's metadata/done/manifest before freezing."""

    stores = {}
    try:
        for name, spec in specs.items():
            stores[name] = LmdbLayerStore(
                spec["root"], expected_meta=spec["meta"]
            )
            # MD200 is a downstream-union layer and may intentionally contain
            # only the rows present in smi_all.csv.  The three pretraining
            # layers, unlike MD200, must cover the complete PI1M_v2 cohort.
            minimum = EXPECTED_COUNT if name != "md200" else 1
            if len(stores[name]) < minimum:
                raise RuntimeError(
                    f"cache layer is incomplete: {name} has {len(stores[name])}, "
                    f"expected at least {minimum}"
                )
    finally:
        for store in stores.values():
            store.close()


def _validate_downstream_union(specs):
    """Ensure every downstream-union key has all frozen training layers."""
    source_csv = PROJECT_ROOT / "data/raw/smi_all.csv"
    if not source_csv.is_file():
        raise RuntimeError(f"downstream union source is missing: {source_csv}")
    cohort = _load_cohort(source_csv, "downstream_union")
    stores = {}
    try:
        for name, spec in specs.items():
            stores[name] = LmdbLayerStore(
                spec["root"], expected_meta=spec["meta"]
            )
        missing = {name: 0 for name in stores}
        for index, key in enumerate(cohort["keys"]):
            for name, store in stores.items():
                if key not in store:
                    missing[name] += 1
        if any(missing.values()):
            raise RuntimeError(
                "downstream union cache coverage is incomplete: "
                + json.dumps(missing, sort_keys=True)
            )
        return {
            "cohort_name": cohort["manifest"].get("dataset_name", "downstream_union"),
            "cohort_hash": cohort["manifest"]["cohort_hash"],
            "ordered_sample_key_hash": cohort["manifest"][
                "ordered_sample_key_hash"
            ],
            "record_count": len(cohort["keys"]),
            "done_artifact_id": {
                name: _artifact_hash(Path(spec["root"]))
                for name, spec in specs.items()
            },
        }
    finally:
        for store in stores.values():
            store.close()


def _validate_mcl_sidecars(specs, cohorts):
    """Validate both threshold arrays and their current-Trimer bindings."""

    trimer_root = Path(specs["trimer"]["root"])
    reports = []
    for cohort in cohorts:
        shape = (len(cohort["keys"]), 2)
        cached = mcl_thresholds_cached(cohort, trimer_root, shape)
        if cached is None:
            raise RuntimeError(
                "MCL threshold sidecar is missing or stale for "
                f"{cohort['manifest']['dataset_name']}"
            )
        path, metadata = cached
        values = np.load(path, mmap_mode="r")
        if tuple(values.shape) != shape or values.dtype != np.float32:
            raise RuntimeError(f"invalid MCL threshold shape/dtype: {path}")
        if _sha256_file(Path(path)) != metadata.get("values_sha256"):
            raise RuntimeError(f"MCL threshold file hash mismatch: {path}")
        finite = np.isfinite(values)
        if np.any(finite[:, 0] != finite[:, 1]):
            raise RuntimeError(
                f"MCL threshold rows must be both finite or both NaN: {path}"
            )
        reports.append({
            "cohort": cohort["manifest"]["dataset_name"],
            "cohort_hash": cohort["manifest"]["cohort_hash"],
            "record_count": shape[0],
            "path": str(path),
            "trimer_artifact_hash": metadata.get("trimer_artifact_hash"),
            "trimer_done_file_sha256": metadata.get("trimer_done_file_sha256"),
            "finite_rows": int(finite[:, 0].sum()),
            "invalid_rows": int((~finite[:, 0]).sum()),
        })
    return reports


def _validate_exact_union_store():
    """Run the 32-worker parallel exact-union validator (Plan.MD §7/§9)."""

    args = argparse.Namespace(
        cohorts="PI1M_v2 downstream_union",
        cohort="PI1M_v2",
        source_csv=None,
        cache_root=str(PROJECT_ROOT / "data/processed/mips_trimer_scage"),
        result_root=str(PROJECT_ROOT / "results/mts_canonical_migration"),
        target_root=None,
        target_topology_root=None,
        target_trimer_root=None,
        full=True,
        workers=32,
        batch_chunk=256,
    )
    return validate_exact_union_parallel(args)


def _try_reuse_exact_union(exact_union_path, pi1m_cohort, specs):
    """Reuse a prior exact-union validation only if it still binds every
    production artifact and cohort identity (Plan.MD §9.2).

    Returns True when ``exact_union_validation.json`` is a valid exact-union report
    whose artifact hashes and cohort hashes all match the current cache; the
    caller then skips the 32-worker parallel validation and the retired serial
    PI1M scan.  Every binding is compared byte-for-byte against the current
    layers, so a stale report is never mistaken for current evidence.
    """
    if not exact_union_path.is_file():
        return False
    try:
        report = json.loads(exact_union_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    if report.get("schema") != MIGRATION_SCHEMA:
        return False
    if report.get("mode") != "validate":
        return False
    if report.get("validator_version") != EXACT_UNION_VALIDATOR_VERSION:
        return False
    if report.get("hard_gate_pass") is not True:
        return False
    if int(report.get("expected_count", 0)) != EXPECTED_COUNT:
        return False
    if int(report.get("record_count", 0)) != EXPECTED_COUNT:
        return False
    # Artifact bindings must match the current layers byte-for-byte.
    for name in ("topology", "trimer"):
        root = Path(specs[name]["root"])
        expected = {
            "done_artifact_id": _artifact_hash(root),
            "done_file_sha256": _sha256_file(root / ".done"),
            "metadata_file_sha256": _sha256_file(root / "metadata.json"),
            "lmdb_manifest_sha256": _sha256_file(root / "manifest.json"),
            "content_config_hash": specs[name]["meta"].get(
                "feature_config_hash"
            ),
        }
        observed = {
            "done_artifact_id": report.get("done_artifact_id", {}).get(name),
            "done_file_sha256": report.get("done_file_sha256", {}).get(name),
            "metadata_file_sha256": report.get(
                "metadata_file_sha256", {}
            ).get(name),
            "lmdb_manifest_sha256": report.get(
                "lmdb_manifest_sha256", {}
            ).get(name),
            "content_config_hash": report.get(
                "content_config_hash", {}
            ).get(name),
        }
        if observed != expected:
            return False
    # Cohort identity must match the production key sets exactly.
    downstream = _load_cohort(
        PROJECT_ROOT / "data/raw/smi_all.csv", "downstream_union"
    )
    pi1m = pi1m_cohort["manifest"]
    down = downstream["manifest"]
    if set(report.get("cohort_hashes", [])) != {
        pi1m["cohort_hash"],
        down["cohort_hash"],
    }:
        return False
    observed_keys = report.get("ordered_key_hashes", {})
    if (
        observed_keys.get("PI1M_v2") != pi1m["ordered_sample_key_hash"]
        or observed_keys.get("downstream_union")
        != down["ordered_sample_key_hash"]
    ):
        return False
    return True


def _assert_exact_union_hard_gate(report):
    required_counts = {
        "expected_count": EXPECTED_COUNT,
        "record_count": EXPECTED_COUNT,
        "topology_count": EXPECTED_COUNT,
        "trimer_count": EXPECTED_COUNT,
    }
    for field, expected in required_counts.items():
        if int(report.get(field, -1)) != expected:
            raise RuntimeError(
                f"exact-union hard gate failed: {field}="
                f"{report.get(field)!r}, expected {expected}"
            )
    for field in (
        "missing_topology", "extra_topology", "missing_trimer", "extra_trimer",
        "topology_trimer_key_mismatch", "structural_failure",
        "atomic_identity_failure", "mapping_failure", "two_d_mcl",
    ):
        if int(report.get(field, -1)) != 0:
            raise RuntimeError(
                f"exact-union hard gate failed: {field}={report.get(field)!r}"
            )
    if float(report.get("mcl_rate_given_graph", 0.0) or 0.0) < 0.90:
        raise RuntimeError(
            "exact-union hard gate failed: MCL valid/Graph available < 90%"
        )


def _validate_angle_sidecars(specs, cohorts):
    """Verify categorical and continuous angle sidecars against new Trimer."""

    trimer_root = Path(specs["trimer"]["root"])
    trimer_done = _artifact_hash(trimer_root)
    trimer_done_file_sha256 = _sha256_file(trimer_root / ".done")
    trimer_contract_hash = specs["trimer"]["meta"].get("feature_config_hash")
    reports = []
    for cohort in cohorts:
        manifest = cohort["manifest"]
        angle_root = trimer_root / "derived" / "bond_angle" / str(
            manifest["cohort_hash"]
        )
        metadata = validate_angle_cache(
            angle_root,
            cohort_hash=manifest["cohort_hash"],
            ordered_key_hash=manifest["ordered_sample_key_hash"],
            trimer_artifact_hash=trimer_done,
            record_count=len(cohort["keys"]),
            trimer_done_file_sha256=trimer_done_file_sha256,
            trimer_contract_hash=trimer_contract_hash,
        )
        continuous_root = continuous_angle_root(
            trimer_root, manifest["cohort_hash"]
        )
        required = (
            "angle_offsets.npy",
            "angle_indices.npy",
            "angle_cos.npy",
            "validation_mask.npy",
            "metadata.json",
            ".done",
            ".frozen",
        )
        if not all((continuous_root / name).is_file() for name in required):
            raise RuntimeError(
                f"continuous angle sidecar is incomplete: {continuous_root}"
            )
        continuous_meta = json.loads(
            (continuous_root / "metadata.json").read_text(encoding="utf-8")
        )
        if (
            continuous_meta.get("cohort_hash") != manifest["cohort_hash"]
            or continuous_meta.get("ordered_sample_key_hash")
            != manifest["ordered_sample_key_hash"]
            or continuous_meta.get("trimer_artifact_hash") != trimer_done
            or continuous_meta.get("trimer_done_artifact_id") != trimer_done
            or continuous_meta.get("trimer_done_file_sha256") != trimer_done_file_sha256
            or continuous_meta.get("trimer_contract_hash") != trimer_contract_hash
            or int(continuous_meta.get("record_count", -1)) != len(cohort["keys"])
        ):
            raise RuntimeError(
                f"continuous angle sidecar metadata is stale: {continuous_root}"
            )
        continuous_artifact = hashlib.sha256(
            json.dumps(
                continuous_meta, sort_keys=True, separators=(",", ":")
            ).encode()
        ).hexdigest()
        if (
            (continuous_root / ".done").read_text(encoding="utf-8").strip()
            != continuous_artifact
        ):
            raise RuntimeError(
                f"continuous angle sidecar .done mismatch: {continuous_root}"
            )
        frozen = json.loads(
            (continuous_root / ".frozen").read_text(encoding="utf-8")
        )
        if (
            frozen.get("artifact_hash") != continuous_artifact
            or frozen.get("schema") != continuous_meta.get("schema")
        ):
            raise RuntimeError(
                f"continuous angle sidecar frozen marker is stale: {continuous_root}"
            )
        for filename, digest in continuous_meta.get("files", {}).items():
            if _sha256_file(continuous_root / filename) != digest:
                raise RuntimeError(
                    f"continuous angle sidecar file hash mismatch: {continuous_root / filename}"
                )
        cosine = np.load(continuous_root / "angle_cos.npy", mmap_mode="r")
        if cosine.dtype != np.float32 or not bool(np.isfinite(cosine).all()):
            raise RuntimeError(f"continuous angle sidecar is non-finite: {continuous_root}")
        reports.append(
            {
                "cohort": manifest["dataset_name"],
                "cohort_hash": manifest["cohort_hash"],
                "record_count": len(cohort["keys"]),
                "angle_count": int(metadata.get("angle_count", -1)),
                "continuous_angle_count": int(continuous_meta.get("angle_count", -1)),
            }
        )
    return reports


def _read_audit(path: Path, full_report, specs, audit_cohort):
    if not path.is_file():
        raise RuntimeError(f"10k audit is missing: {path}")
    audit = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema": "mts-canonical-cache-audit-v2",
        "migration_schema": MIGRATION_SCHEMA,
        "cohort_hash": audit_cohort["manifest"]["cohort_hash"],
        "ordered_sample_key_hash": audit_cohort["manifest"][
            "ordered_sample_key_hash"
        ],
        "record_count": len(audit_cohort["keys"]),
        "mapping_failure": 0,
        "two_d_mcl": 0,
    }
    for key, expected in required.items():
        if audit.get(key) != expected:
            raise RuntimeError(
                f"10k audit mismatch for {key}: "
                f"expected {expected!r}, got {audit.get(key)!r}"
            )
    if audit.get("geometry_rate_given_graph") is None or float(
        audit["geometry_rate_given_graph"]
    ) < 0.90:
        raise RuntimeError("10k audit is below the 90% MCL quality gate")
    for name in ("topology", "trimer"):
        expected_hash = _artifact_hash(Path(specs[name]["root"]))
        if audit.get("done_artifact_id", {}).get(name) != expected_hash:
            raise RuntimeError(f"10k audit {name} artifact hash mismatch")
    if audit.get("trimer_feature_config_hash") != specs["trimer"]["meta"][
        "feature_config_hash"
    ]:
        raise RuntimeError("10k audit Trimer feature hash mismatch")
    return audit


def _run_audit(output: Path, source_csv: Path, cohort_name: str):
    """Run the read-only audit as part of finalize, never just read stale JSON."""
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts/audit_mips_trimer_cache.py"),
        "--source-csv", os.path.relpath(source_csv, PROJECT_ROOT),
        "--dataset-name", str(cohort_name),
        "--output", os.path.relpath(output, PROJECT_ROOT),
    ]
    completed = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            f"10k audit failed with exit code {completed.returncode}"
        )


def _freeze_roots(specs, *, cohort_hashes=()):
    """Atomically publish matching v3 frozen markers for every layer."""

    transaction_id = uuid.uuid4().hex
    payloads = {}
    for name, spec in specs.items():
        root = Path(spec["root"])
        done_artifact_id = _artifact_hash(root)
        done_file_sha256 = _sha256_file(root / ".done")
        metadata_hash = _json_hash(spec["meta"])
        manifest_hash = _json_hash(
            json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        )
        existing = root / ".frozen"
        if existing.is_file():
            try:
                old = json.loads(existing.read_text(encoding="utf-8"))
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                raise RuntimeError(f"invalid existing frozen marker: {existing}") from exc
            # MD200 was frozen by the downstream preparer under the legacy
            # v1 marker.  Its LMDB payload is immutable and already covered by
            # the current manifest; the finalizer upgrades only this marker to
            # the common v2 bundle transaction below.  Topology/Trimer never
            # accept a legacy marker because that could hide a stale canonical
            # artifact.
            if (
                name in {"ru_base", "md200"}
                and old.get("schema") == "mips-trimer-scage-cache-freeze-v1"
            ):
                # These two immutable downstream/input layers were published
                # by the legacy preparer.  Upgrade only when every content
                # binding in that marker still matches the current layer;
                # never bless a stale marker merely because its schema is old.
                if (
                    old.get("layer") != name
                    or old.get("done_hash") != done_artifact_id
                    or old.get("manifest_hash") != manifest_hash
                    or old.get("metadata_hash") != metadata_hash
                    or old.get("feature_config_hash")
                    != spec["meta"].get("feature_config_hash")
                ):
                    raise RuntimeError(
                        f"legacy frozen marker does not match current contract: "
                        f"{existing}"
                    )
                old = None
            if old is None:
                pass
            elif (
                old.get("schema") != "mts-canonical-cache-freeze-v2"
                or old.get("layer") != name
                or old.get("done_artifact_id") != done_artifact_id
                or old.get("done_file_sha256") != done_file_sha256
                or old.get("metadata_hash") != metadata_hash
                or old.get("manifest_hash") != manifest_hash
                or old.get("contract_schema") != TARGET_CONTRACT_SCHEMA
                or int(old.get("builder_version", -1)) != BUILDER_VERSION
                or not old.get("transaction_id")
            ):
                raise RuntimeError(
                    f"existing frozen marker does not match current contract: {existing}"
                )
            else:
                transaction_id = str(old["transaction_id"])
        payloads[name] = (
            root,
            {
                "schema": "mts-canonical-cache-freeze-v2",
                "layer": name,
                "cache_layout_schema": spec["meta"]["cache_layout_schema"],
                "contract_schema": TARGET_CONTRACT_SCHEMA,
                "builder_version": BUILDER_VERSION,
                "feature_config_hash": spec["meta"]["feature_config_hash"],
                "metadata_hash": metadata_hash,
                "manifest_hash": manifest_hash,
                "done_artifact_id": done_artifact_id,
                "done_file_sha256": done_file_sha256,
                "cohort_hashes": sorted(str(value) for value in cohort_hashes),
                "transaction_id": transaction_id,
                "record_count": int(
                    json.loads((root / "manifest.json").read_text(encoding="utf-8"))["count"]
                ),
                "frozen_at": time.time(),
            },
        )
    temporary_paths = []
    committed = []
    try:
        for _name, (root, payload) in payloads.items():
            temporary = root / ".frozen.tmp"
            with temporary.open("w", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, sort_keys=True, indent=2) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            temporary_paths.append(temporary)
        for temporary in temporary_paths:
            marker = temporary.with_name(".frozen")
            os.replace(temporary, marker)
            committed.append(marker)
    except Exception:
        for temporary in temporary_paths:
            temporary.unlink(missing_ok=True)
        for marker in committed:
            try:
                payload = json.loads(marker.read_text(encoding="utf-8"))
            except Exception:
                payload = {}
            if payload.get("transaction_id") == transaction_id:
                marker.unlink(missing_ok=True)
        raise
    return transaction_id


def _rollback_freeze_transaction(specs, transaction_id):
    for spec in specs.values():
        marker = Path(spec["root"]) / ".frozen"
        if not marker.is_file():
            continue
        try:
            payload = json.loads(marker.read_text(encoding="utf-8"))
        except Exception:
            payload = {}
        if payload.get("transaction_id") == transaction_id:
            marker.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prepare-downstream",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--validate-full", action="store_true")
    parser.add_argument("--audit-10k", action="store_true")
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--audit", default="results/mts/cache_audit_10k.json")
    parser.add_argument("--cohort-source-csv", default="data/raw/PI1M_v2.csv")
    parser.add_argument("--cohort-name", default=PI1M_DATASET)
    parser.add_argument("--audit-source-csv", default="data/raw/PI1M_preflight10k.csv")
    parser.add_argument("--audit-cohort-name", default="PI1M_preflight10k")
    # Historical roots are immutable inputs to migration and are never
    # deleted by the production finalizer.
    parser.add_argument("--delete-old", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.delete_old:
        raise SystemExit("historical MTS cache roots are immutable; deletion is disabled")
    # No flags means the safe complete operation; explicit flags select
    # individual stages for diagnostics, never bypassing prerequisites.
    if args.prepare_downstream:
        raise SystemExit(
            "--prepare-downstream is retired; run "
            "scripts/mips_trimer_scage.py prepare-downstream first."
        )
    run_full = args.validate_full or not any(
        (args.validate_full, args.audit_10k, args.freeze)
    )
    run_audit = args.audit_10k or not any(
        (args.validate_full, args.audit_10k, args.freeze)
    )
    run_freeze = args.freeze
    # Freeze is never a standalone shortcut: it always performs the exact
    # union validation and real 10k audit in the same lifecycle transaction.
    if run_freeze and not run_full:
        run_full = True
        run_audit = True
    specs = _specs(PROJECT_ROOT)
    # Lock the cache lifecycle before preparing any downstream records.  This
    # prevents two finalizers from both observing an idle writer and racing on
    # the same LMDB roots.
    lifecycle_path = PROJECT_ROOT / "data/processed/mips_trimer_scage/.lifecycle.lock"
    lifecycle_path.parent.mkdir(parents=True, exist_ok=True)
    lifecycle_handle = lifecycle_path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(lifecycle_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        lifecycle_handle.close()
        raise SystemExit("another cache finalizer is active") from exc

    def _release_lifecycle_lock():
        try:
            fcntl.flock(lifecycle_handle.fileno(), fcntl.LOCK_UN)
            lifecycle_handle.close()
        except OSError:
            pass

    atexit.register(_release_lifecycle_lock)
    active = _active_writer_pids()
    if active:
        raise SystemExit(
            "cache writer still active; finalize will not race it: "
            + "; ".join(f"pid={pid}" for pid, _ in active)
        )
    _assert_no_writer_locks(specs)
    cohort = _load_cohort(
        (PROJECT_ROOT / args.cohort_source_csv).resolve(), args.cohort_name
    )
    validation_dir = Path(specs["topology"]["root"]).parents[1] / "validation"
    validation_dir.mkdir(parents=True, exist_ok=True)
    store_path = validation_dir / "store.json"
    exact_union_path = validation_dir / "exact_union_validation.json"
    legacy_exact_union_path = validation_dir / "exact_union.json"
    # §9.1: report-path separation.  store.json is reserved for the frozen
    # bundle marker only; the exact-union validation evidence lives in
    # exact_union_validation.json.  A leftover migration-schema store.json from before the
    # separation is migrated (or superseded) before any new validation runs.
    if store_path.is_file():
        try:
            legacy = json.loads(store_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            legacy = {}
        if legacy.get("schema") == MIGRATION_SCHEMA:
            if exact_union_path.is_file():
                print(f"superseding migration-schema store.json: {store_path}")
                store_path.unlink(missing_ok=True)
            else:
                os.replace(store_path, exact_union_path)
                print(
                    f"migrated migration-schema store.json to exact_union_validation.json: "
                    f"{exact_union_path}"
                )
    if not exact_union_path.is_file() and legacy_exact_union_path.is_file():
        # Preserve the report bytes (and therefore existing checkpoint hash
        # bindings) while moving the evidence to its dedicated name.
        os.replace(legacy_exact_union_path, exact_union_path)
        print(
            f"migrated legacy exact-union report to {exact_union_path}"
        )
    if run_freeze and store_path.is_file():
        # A valid bundle is an idempotent completed transaction.  An invalid
        # marker is preserved under a timestamped name before this run can
        # perform any new validation or freeze work.
        try:
            verify_frozen_cache_bundle(
                specs,
                store_path=store_path,
                required_layers=specs.keys(),
                pretraining_cohort_hash=cohort["manifest"]["cohort_hash"],
            )
        except Exception:
            invalid = store_path.with_name(
                f"store.json.invalid.{time.time_ns()}"
            )
            os.replace(store_path, invalid)
        else:
            print(f"MIPS-Trimer-SCAGE cache already frozen: {store_path}")
            return 0
    _assert_no_writer_locks(specs)
    if run_full or run_freeze:
        _assert_layer_integrity(specs)
    full_report = None
    if run_full:
        # §9.2: reuse the prior exact-union evidence only when every artifact
        # and cohort binding still matches the current cache byte-for-byte.
        # Otherwise run the 32-worker parallel exact-union validator (the
        # retired serial PI1M scan and serial validation are never used for a
        # fresh gate).  The exact-union report is the sole production gate and
        # store payload, so finalization cannot silently weaken key-set or
        # Trimer structural checks.
        if _try_reuse_exact_union(exact_union_path, cohort, specs):
            print(f"reusing existing exact-union validation: {exact_union_path}")
            full_report = json.loads(
                exact_union_path.read_text(encoding="utf-8")
            )
        else:
            full_report = _validate_exact_union_store()
    else:
        path = exact_union_path
        if not path.is_file():
            raise SystemExit("full PI1M_v2 validation is required before this stage")
        full_report = json.loads(path.read_text(encoding="utf-8"))
    _assert_exact_union_hard_gate(full_report)
    audit = None
    if run_audit:
        audit_cohort = _load_cohort(
            (PROJECT_ROOT / args.audit_source_csv).resolve(),
            args.audit_cohort_name,
        )
        audit_output = (PROJECT_ROOT / args.audit).resolve()
        _run_audit(
            audit_output,
            (PROJECT_ROOT / args.audit_source_csv).resolve(),
            args.audit_cohort_name,
        )
        audit = _read_audit(
            audit_output, full_report, specs, audit_cohort
        )
    if run_freeze:
        if audit is None:
            audit = _read_audit(
                (PROJECT_ROOT / args.audit).resolve(),
                full_report,
                specs,
                _load_cohort(
                    (PROJECT_ROOT / args.audit_source_csv).resolve(),
                    args.audit_cohort_name,
                ),
            )
        downstream_report = _validate_downstream_union(specs)
        sidecar_cohorts = [cohort, _load_cohort(
            (PROJECT_ROOT / "data/raw/smi_all.csv").resolve(),
            "downstream_union",
        )]
        angle_report = _validate_angle_sidecars(specs, sidecar_cohorts)
        mcl_report = _validate_mcl_sidecars(specs, sidecar_cohorts)
        # The exact-union report contains 999,224 keys (PI1M plus the
        # downstream-only tail), while the pretraining threshold mmap is
        # row-ordered to the 995,799-key PI1M_v2 cohort.  Preserve that
        # distinction inside the frozen bundle so its verifier checks the
        # correct sidecar shape instead of demanding an impossible union array.
        full_validation_for_store = dict(full_report)
        full_validation_for_store["pretraining_subset"] = {
            "cohort_name": cohort["manifest"]["dataset_name"],
            "cohort_hash": cohort["manifest"]["cohort_hash"],
            "ordered_sample_key_hash": cohort["manifest"][
                "ordered_sample_key_hash"
            ],
            "record_count": len(cohort["keys"]),
        }
        # Freeze all layers first.  The store marker is the final commit point;
        # a failed freeze must never leave a valid-looking store.json.
        transaction_id = _freeze_roots(
            specs,
            cohort_hashes=(
                cohort["manifest"]["cohort_hash"],
                downstream_report["cohort_hash"],
            ),
        )
        # Write the store marker only after full validation, audit and all
        # immutable layer markers have passed.  It binds all artifacts.
        store = {
            "schema": CACHE_BUNDLE_SCHEMA,
            "migration_schema": MIGRATION_SCHEMA,
            "contract_schema": TARGET_CONTRACT_SCHEMA,
            "builder_version": BUILDER_VERSION,
            "transaction_id": transaction_id,
            "pretraining_dataset": PI1M_DATASET,
            "pretraining_cohort_hash": cohort["manifest"]["cohort_hash"],
            "full_validation": full_validation_for_store,
            "audit": audit,
            "downstream_union": downstream_report,
            "angle_sidecars": angle_report,
            "mcl_sidecars": mcl_report,
            "done_artifact_id": {
                name: _artifact_hash(Path(specs[name]["root"]))
                for name in specs
            },
            "done_file_sha256": {
                name: _sha256_file(Path(specs[name]["root"]) / ".done")
                for name in specs
            },
            "metadata_file_sha256": {
                name: _sha256_file(Path(specs[name]["root"]) / "metadata.json")
                for name in specs
            },
            "lmdb_manifest_sha256": {
                name: _sha256_file(Path(specs[name]["root"]) / "manifest.json")
                for name in specs
            },
            "frozen_payload_sha256": {
                name: _sha256_file(Path(specs[name]["root"]) / ".frozen")
                for name in specs
            },
            "written_at": time.time(),
        }
        temporary = validation_dir / "store.json.tmp"
        try:
            with temporary.open("w", encoding="utf-8") as handle:
                handle.write(json.dumps(store, sort_keys=True, indent=2) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, store_path)
            # The store marker is the final commit point, but verify it once
            # through the exact reader used by training before releasing the
            # lifecycle lock.  A malformed marker must never authorize a
            # later Dataset construction.
            verify_frozen_cache_bundle(
                specs,
                store_path=store_path,
                required_layers=specs.keys(),
                pretraining_cohort_hash=cohort["manifest"]["cohort_hash"],
            )
        except Exception:
            temporary.unlink(missing_ok=True)
            store_path.unlink(missing_ok=True)
            _rollback_freeze_transaction(specs, transaction_id)
            raise
        acceptance = {
            "schema": "mts-canonical-final-acceptance-v1",
            "migration_schema": MIGRATION_SCHEMA,
            "contract_schema": TARGET_CONTRACT_SCHEMA,
            "transaction_id": transaction_id,
            "hard_gates": {
                "exact_union_count": int(full_report.get("expected_count", 0)) == EXPECTED_COUNT,
                "topology_count": int(full_report.get("topology_count", 0)) == EXPECTED_COUNT,
                "trimer_count": int(full_report.get("trimer_count", 0)) == EXPECTED_COUNT,
                "missing_extra_zero": not any(full_report.get(name, 0) for name in (
                    "missing_topology", "extra_topology", "missing_trimer", "extra_trimer",
                    "topology_trimer_key_mismatch",
                )),
                "structural_zero": int(full_report.get("structural_failure", 0)) == 0,
                "atomic_identity_zero": int(full_report.get("atomic_identity_failure", 0)) == 0,
                "mapping_zero": int(full_report.get("mapping_failure", 0)) == 0,
                "two_d_mcl_zero": int(full_report.get("two_d_mcl", 0)) == 0,
                "mcl_rate_ge_90": float(full_report.get("mcl_rate_given_graph", 0.0) or 0.0) >= 0.90,
                "audit_pass": int(audit.get("mapping_failure", 1)) == 0 and int(audit.get("two_d_mcl", 1)) == 0,
                "angle_sidecars_complete": len(angle_report) == 2,
                "mcl_sidecars_complete": len(mcl_report) == 2,
                "freeze_store_verified": True,
                "no_writer": not bool(_active_writer_pids()),
                "training_not_started": True,
            },
            "evidence": {
                "store": str(store_path),
                "validation": str(exact_union_path),
                "audit": str((PROJECT_ROOT / args.audit).resolve()),
                "angle_sidecars": angle_report,
                "mcl_sidecars": mcl_report,
            },
            "created_at": time.time(),
        }
        acceptance_path = PROJECT_ROOT / "results/mts_canonical_migration" / "final_acceptance.json"
        acceptance_path.parent.mkdir(parents=True, exist_ok=True)
        acceptance_tmp = acceptance_path.with_suffix(".json.tmp")
        acceptance_tmp.write_text(json.dumps(acceptance, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        os.replace(acceptance_tmp, acceptance_path)
        print("MIPS-Trimer-SCAGE (MTS) cache frozen for PI1M_v2")
    print(
        json.dumps(
            {
                "pretraining_dataset": PI1M_DATASET,
                "cohort_hash": cohort["manifest"]["cohort_hash"],
                "full_validation": full_report,
                "audit": audit,
                "frozen": run_freeze,
            },
            sort_keys=True,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
