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
    compute_mcl_thresholds,
    materialize_mcl_threshold_array,
)
from src.dataset.mips_cache_validation import (  # noqa: E402
    validate_mcl_record,
    verify_frozen_cache_bundle,
)

EXPECTED_COUNT = 995_799
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
    env = os.environ.copy()
    env.update({
        "CUDA_VISIBLE_DEVICES": "",
        "PRETRAIN_DATASET": PI1M_DATASET,
        "STAGE2_DATASET": PI1M_DATASET,
        "PRETRAIN_CACHE_ONLY": "1",
        "PRETRAIN_ONLY": "0",
        "STAGE3_ONLY": "0",
        "VALIDATE_ONLY": "0",
        "REBUILD_FEATURE_CACHE": "0",
        # Full validation is a single finalizer operation.  Cache preparation
        # must not deserialize 995,799 LMDB records just to prove that the
        # already-complete PI1M layers are still present.
        "CACHE_VALIDATE": "sample",
    })
    command = ["bash", "scripts/run_mips_trimer_scage.sh"]
    completed = subprocess.run(command, cwd=PROJECT_ROOT, env=env)
    if completed.returncode != 0:
        raise RuntimeError(
            f"downstream cache preparation failed with exit code "
            f"{completed.returncode}"
        )


def _load_cohort(source_csv: Path, name: str):
    cohort = build_or_load_cohort(
        PROJECT_ROOT / "data/processed/mips_trimer_scage",
        name,
        source_csv,
        load_text=False,
        verify_integrity=True,
    )
    if name == PI1M_DATASET and int(cohort["manifest"]["unique_count"]) != EXPECTED_COUNT:
        raise RuntimeError(
            f"PI1M_v2 cohort count mismatch: "
            f"{cohort['manifest'].get('unique_count')} != {EXPECTED_COUNT}"
        )
    return cohort


def _validate_full_pi1m(cohort, specs):
    stores = {}
    failure_path = (
        Path(specs["trimer"]["root"])
        / "failures"
        / f"{cohort['manifest']['cohort_hash']}.json"
    )
    failure_path.parent.mkdir(parents=True, exist_ok=True)
    failure_tmp = failure_path.with_suffix(".json.tmp")
    try:
        for name in ("ru_base", "topology", "trimer"):
            store = LmdbLayerStore(
                specs[name]["root"], expected_meta=specs[name]["meta"]
            )
            stores[name] = store
            if len(store) < EXPECTED_COUNT:
                raise RuntimeError(
                    f"{name} cache has {len(store)} records; expected at least "
                    f"{EXPECTED_COUNT}"
                )

        mapping_failure = 0
        two_d_mcl = 0
        mcl_valid = 0
        graph_available = 0
        geometry_valid = 0
        failures = {}
        keys = cohort["keys"]
        mcl_thresholds = np.full((len(keys), 2), np.nan, dtype=np.float32)
        first_failure = True
        with failure_tmp.open("w", encoding="utf-8") as failure_handle:
            failure_handle.write("[\n")
            for index, key in enumerate(keys):
                # Validate the dependency layer as well; a matching LMDB
                # count alone must not allow a missing PI1M key to be frozen.
                _ = stores["ru_base"][key]
                topology = stores["topology"][key]
                trimer = stores["trimer"][key]
                mcl_thresholds[index] = compute_mcl_thresholds(trimer)
                quality = validate_mcl_record(topology, trimer)
                graph_available += int(quality["graph_available"])
                geometry_valid += int(quality["geometry_valid"])
                mcl_valid += int(quality["mcl_valid"])
                mapping_failure += int(quality["mapping_failure"])
                two_d_mcl += int(quality["two_d_mcl"])
                if not quality["geometry_valid"]:
                    code = str(getattr(trimer, "trimer_failure_code", "unknown"))
                    failures[code] = failures.get(code, 0) + 1
                    item = {
                        "sample_key": bytes(key).hex(),
                        "error": code[:240],
                    }
                    if not first_failure:
                        failure_handle.write(",\n")
                    json.dump(item, failure_handle, sort_keys=True)
                    first_failure = False
            failure_handle.write("\n]\n")
        geometry_rate = mcl_valid / graph_available if graph_available else None
        if mapping_failure != 0:
            failure_tmp.unlink(missing_ok=True)
            raise RuntimeError(f"full validation mapping_failure={mapping_failure}")
        if two_d_mcl != 0:
            failure_tmp.unlink(missing_ok=True)
            raise RuntimeError(f"full validation two_d_mcl={two_d_mcl}")
        if geometry_rate is None or geometry_rate < 0.90:
            failure_tmp.unlink(missing_ok=True)
            raise RuntimeError(
                f"full validation geometry_rate_given_graph={geometry_rate}"
            )
        materialize_mcl_threshold_array(
            cohort,
            mcl_thresholds,
            specs["trimer"]["root"],
        )
        os.replace(failure_tmp, failure_path)
        return {
            "schema": "mips-trimer-scage-pi1m-validation-v2",
            "cohort_name": cohort["manifest"].get("dataset_name", PI1M_DATASET),
            "cohort_hash": cohort["manifest"]["cohort_hash"],
            "ordered_sample_key_hash": cohort["manifest"][
                "ordered_sample_key_hash"
            ],
            "record_count": len(keys),
            "graph_available": graph_available,
            "geometry_valid": geometry_valid,
            "mcl_valid": mcl_valid,
            "geometry_rate_given_graph": geometry_rate,
            "mapping_failure": mapping_failure,
            "two_d_mcl": two_d_mcl,
            "failure_counts": failures,
            "artifact_hashes": {
                name: _artifact_hash(Path(specs[name]["root"]))
                for name in ("ru_base", "topology", "trimer")
            },
            "metadata_hashes": {
                name: _json_hash(stores[name].meta)
                for name in ("ru_base", "topology", "trimer")
            },
            "validated_at": time.time(),
        }
    finally:
        for store in stores.values():
            store.close()


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
    cohort = _load_cohort(source_csv, "smi_all")
    stores = {}
    try:
        for name, spec in specs.items():
            stores[name] = LmdbLayerStore(
                spec["root"], expected_meta=spec["meta"]
            )
        missing = {name: 0 for name in stores}
        mcl_thresholds = np.full(
            (len(cohort["keys"]), 2), np.nan, dtype=np.float32
        )
        for index, key in enumerate(cohort["keys"]):
            for name, store in stores.items():
                if key not in store:
                    missing[name] += 1
            if not missing["trimer"]:
                mcl_thresholds[index] = compute_mcl_thresholds(
                    stores["trimer"][key]
                )
        if any(missing.values()):
            raise RuntimeError(
                "downstream union cache coverage is incomplete: "
                + json.dumps(missing, sort_keys=True)
            )
        materialize_mcl_threshold_array(
            cohort,
            mcl_thresholds,
            specs["trimer"]["root"],
        )
        return {
            "cohort_name": cohort["manifest"].get("dataset_name", "smi_all"),
            "cohort_hash": cohort["manifest"]["cohort_hash"],
            "ordered_sample_key_hash": cohort["manifest"][
                "ordered_sample_key_hash"
            ],
            "record_count": len(cohort["keys"]),
            "artifact_hashes": {
                name: _artifact_hash(Path(spec["root"]))
                for name, spec in specs.items()
            },
        }
    finally:
        for store in stores.values():
            store.close()


def _read_audit(path: Path, full_report, specs, audit_cohort):
    if not path.is_file():
        raise RuntimeError(f"10k audit is missing: {path}")
    audit = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema": "mips-trimer-scage-cache-audit-v1",
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
        if audit.get(f"{name}_artifact_hash") != expected_hash:
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


def _freeze_roots(specs):
    payloads = {}
    for name, spec in specs.items():
        root = Path(spec["root"])
        payload = {
            "schema": "mips-trimer-scage-cache-freeze-v1",
            "layer": name,
            "cache_layout_schema": spec["meta"]["cache_layout_schema"],
            "feature_config_hash": spec["meta"]["feature_config_hash"],
            "metadata_hash": _json_hash(spec["meta"]),
            "manifest_hash": _json_hash(
                json.loads((root / "manifest.json").read_text(encoding="utf-8"))
            ),
            "done_hash": _artifact_hash(root),
            "record_count": int(
                json.loads((root / "manifest.json").read_text(encoding="utf-8"))["count"]
            ),
            "frozen_at": time.time(),
        }
        payloads[name] = (root, payload)
    temporary_paths = []
    committed = []
    try:
        for _name, (root, payload) in payloads.items():
            temporary = root / ".frozen.tmp"
            temporary.write_text(
                json.dumps(payload, sort_keys=True, indent=2) + "\n",
                encoding="utf-8",
            )
            temporary_paths.append(temporary)
        for temporary in temporary_paths:
            marker = temporary.with_name(".frozen")
            os.replace(temporary, marker)
            committed.append(marker)
    except Exception:
        for temporary in temporary_paths:
            temporary.unlink(missing_ok=True)
        for marker in committed:
            marker.unlink(missing_ok=True)
        raise


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
    parser.add_argument("--delete-old", action="store_true")
    args = parser.parse_args()
    if args.delete_old and not args.freeze:
        raise SystemExit("--delete-old requires --freeze and a successful finalization")
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
    # A previous failed finalize must never remain a valid-looking gate.  Do
    # this only after the exclusive lifecycle lock is held, so two finalizers
    # cannot invalidate each other's store marker.
    stale_store = Path(specs["trimer"]["root"]) / "validation" / "store.json"
    if stale_store.is_file():
        stale_store.unlink()
    _assert_no_writer_locks(specs)
    if run_full or run_freeze:
        _assert_layer_integrity(specs)
    full_report = None
    if run_full:
        full_report = _validate_full_pi1m(cohort, specs)
        root = Path(specs["trimer"]["root"])
        validation_dir = root / "validation" / "cohorts"
        validation_dir.mkdir(parents=True, exist_ok=True)
        path = validation_dir / f"{cohort['manifest']['cohort_hash']}.json"
        path_tmp = path.with_suffix(path.suffix + ".tmp")
        path_tmp.write_text(
            json.dumps(full_report, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(path_tmp, path)
        index_path = Path(specs["trimer"]["root"]) / "validation.json"
        index_tmp = index_path.with_suffix(index_path.suffix + ".tmp")
        index_tmp.write_text(
            json.dumps(
                {
                    "schema": "mips-trimer-scage-validation-index-v1",
                    "latest_cohort_hash": cohort["manifest"]["cohort_hash"],
                    "cohort_validation": str(
                        Path("validation")
                        / "cohorts"
                        / f"{cohort['manifest']['cohort_hash']}.json"
                    ),
                },
                sort_keys=True,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(index_tmp, index_path)
    else:
        path = Path(specs["trimer"]["root"]) / "validation" / "cohorts" / (
            f"{cohort['manifest']['cohort_hash']}.json"
        )
        if not path.is_file():
            raise SystemExit("full PI1M_v2 validation is required before this stage")
        full_report = json.loads(path.read_text(encoding="utf-8"))
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
        # Freeze all layers first.  The store marker is the final commit point;
        # a failed freeze must never leave a valid-looking store.json.
        _freeze_roots(specs)
        # Write the store marker only after full validation, audit and all
        # immutable layer markers have passed.  It binds all artifacts.
        store = {
            "schema": "mips-trimer-scage-store-validation-v2",
            "pretraining_dataset": PI1M_DATASET,
            "pretraining_cohort_hash": cohort["manifest"]["cohort_hash"],
            "full_validation": full_report,
            "audit": audit,
            "downstream_union": downstream_report,
            "artifact_hashes": {
                name: _artifact_hash(Path(specs[name]["root"]))
                for name in specs
            },
            "written_at": time.time(),
        }
        root = Path(specs["trimer"]["root"])
        validation_dir = root / "validation"
        validation_dir.mkdir(parents=True, exist_ok=True)
        temporary = validation_dir / "store.json.tmp"
        temporary.write_text(
            json.dumps(store, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        store_path = validation_dir / "store.json"
        os.replace(temporary, store_path)
        try:
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
            store_path.unlink(missing_ok=True)
            raise
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
    if args.delete_old:
        for root in OLD_ROOTS:
            if root.exists():
                shutil.rmtree(root)
                print(f"deleted obsolete cache root: {root}")


if __name__ == "__main__":
    main()
