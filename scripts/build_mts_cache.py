#!/usr/bin/env python3
"""Build, audit, freeze and atomically publish one new-protocol MTS bundle.

This command never scans or reuses historical cache artifacts.  Re-running the
same command resumes only its content-addressed ``.staging`` bundle.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import pickle
import shutil
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

import lmdb
import numpy as np
import torch
from rdkit import Chem, rdBase
from torch_geometric.data import Data

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset.cache_lifecycle import (  # noqa: E402
    CacheAuditError,
    CacheLifecycleError,
    IntentionalBuildInterrupt,
    PublishedCacheDataset,
    StagingWriter,
    append_jsonl,
    artifact_identity,
    atomic_json,
    bundle_identity,
    deserialize_record,
    failure_distribution,
    json_hash,
    load_source_rows,
    ordered_key_hash,
    read_rejections,
    select_record_fields,
    serialize_record,
    sha256_file,
    snapshot_tree,
)
from src.dataset.cache_spec import (  # noqa: E402
    RECORD_FIELDS,
    REQUIRED_FIELDS,
    ROUTE_BUILD_SPECS,
    build_spec_hash,
)
from src.dataset.dataset import (  # noqa: E402
    _compute_ru_base_layer,
    _compute_topology_layer_impl,
)
from src.dataset.graph_data import build_periodic_multimer_mol  # noqa: E402
from src.dataset.trimer_mcl import (  # noqa: E402
    TrimerContractError,
    TrimerGeometryRejection,
    attach_finite_trimer_mcl,
)


def _metadata(layer, artifact_hash, source_manifest_hash, parents):
    spec = ROUTE_BUILD_SPECS[layer]
    return {
        "artifact_type": layer,
        "artifact_hash": artifact_hash,
        "build_spec": spec,
        "build_spec_hash": build_spec_hash(spec),
        "source_manifest_hash": source_manifest_hash,
        "parents": dict(sorted(parents.items())),
        "record_fields": list(RECORD_FIELDS[layer]),
        "toolchain": {"rdkit": rdBase.rdkitVersion, "torch": torch.__version__},
    }


def _write_source(staging, rows, manifest):
    root = staging / "source"
    root.mkdir(parents=True, exist_ok=True)
    records_path = root / "records.jsonl"
    expected = "".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
        for row in rows
    )
    if records_path.exists():
        if records_path.read_text(encoding="utf-8") != expected:
            raise CacheLifecycleError("staging source records mismatch")
    else:
        records_path.write_text(expected, encoding="utf-8")
    observed = {**manifest, "records_file_sha256": sha256_file(records_path)}
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text(encoding="utf-8")) != observed:
            raise CacheLifecycleError("staging source manifest mismatch")
    else:
        atomic_json(manifest_path, observed)


# RU failure codes with a dedicated diagnostic identity.  Everything else a
# sample raises is recorded honestly as UNCLASSIFIED_SAMPLE_FAILURE with its
# exception type and message; classification can be refined after the build.
KNOWN_RU_FAILURE_CODES = ("RU_BUILD_UNSUPPORTED", "RU_DEGENERATE_DUMMY_ONLY")

UNCLASSIFIED = "UNCLASSIFIED_SAMPLE_FAILURE"


def _split_ru_failure_code(raw):
    """Map a recorded RU failure code onto (failure_code, exception_type,
    exception_message).  Known builder-capability codes keep their identity;
    anything else becomes an explicit UNCLASSIFIED_SAMPLE_FAILURE."""
    text = str(raw or "").strip()
    head, separator, tail = text.partition(":")
    if head in KNOWN_RU_FAILURE_CODES:
        return head, head, (tail if separator else text)
    if not separator:
        return UNCLASSIFIED, text, text
    return UNCLASSIFIED, head, tail


def _failure_entry(payload, layer, *, failure_code, exception_type,
                   exception_message, elapsed_seconds, candidate_attempts=0,
                   round_reached=-1):
    """One per-sample failure ledger row (see the relaxed failure policy)."""
    return {
        "sample_key": str(payload["sample_key"]),
        "source_row": int(payload.get("source_row", -1)),
        "canonical_smiles": str(payload.get("normalized_smiles", ""))[:300],
        "layer": str(layer),
        "failure_code": str(failure_code),
        "exception_type": str(exception_type)[:80],
        "exception_message": str(exception_message)[:240],
        "worker_pid": int(os.getpid()),
        "elapsed_seconds": round(float(elapsed_seconds), 3),
        "candidate_attempts": int(candidate_attempts),
        "round_reached": int(round_reached),
    }


class _FailureLedger:
    """Parent-side writer for one per-layer failure ledger (JSONL).

    Full tracebacks are deduplicated by content hash into a sidecar file, so
    a recurring failure never repeats its long traceback per sample.
    """

    def __init__(self, layer_dir: Path):
        self.path = layer_dir / "rejections.jsonl"
        self.traceback_path = layer_dir / "failure_tracebacks.jsonl"
        self.path.touch(exist_ok=True)
        self.rows = read_rejections(self.path)
        self._seen_tracebacks = set()
        if self.traceback_path.is_file():
            for line in self.traceback_path.read_text(
                encoding="utf-8"
            ).splitlines():
                if line.strip():
                    self._seen_tracebacks.add(
                        json.loads(line)["traceback_hash"]
                    )

    def append(self, entry: dict, traceback_text: str | None = None) -> None:
        key = bytes.fromhex(entry["sample_key"])
        if key in self.rows:
            raise CacheLifecycleError(
                f"duplicate failure ledger key: {entry['sample_key']}"
            )
        row = dict(entry)
        if traceback_text:
            digest = hashlib.sha256(
                traceback_text.encode("utf-8")
            ).hexdigest()[:16]
            row["traceback_hash"] = digest
            if digest not in self._seen_tracebacks:
                self._seen_tracebacks.add(digest)
                append_jsonl(self.traceback_path, {
                    "traceback_hash": digest,
                    "traceback": traceback_text[-4000:],
                })
        append_jsonl(self.path, row)
        self.rows[key] = row


def _progress_line(phase, processed, total, counts, failure_counts,
                   started_wall, fs_path):
    elapsed = max(time.monotonic() - started_wall, 1e-9)
    remaining = max(0, total - processed)
    rate = processed / elapsed
    rss_bytes = 0
    try:
        for line in Path("/proc/self/status").read_text(
            encoding="utf-8"
        ).splitlines():
            if line.startswith("VmRSS:"):
                rss_bytes = int(line.split()[1]) * 1024
                break
    except OSError:
        pass
    try:
        disk_free_bytes = shutil.disk_usage(fs_path).free
    except OSError:
        disk_free_bytes = None
    print(json.dumps({
        "phase": phase, "processed": processed, "total": total,
        "unresolved": remaining,
        "samples_per_second": round(rate, 3),
        "eta_seconds": round(remaining / max(rate, 1e-9), 1)
        if processed else None,
        "rss_bytes": rss_bytes,
        "disk_free_bytes": disk_free_bytes,
        **counts,
        "failure_counts": dict(sorted(failure_counts.items())),
    }), flush=True)


def _run_struct(staging, rows, metadata, workers):
    """Compute RU + Topology records through the shared worker pool.

    Scheduling-only parallelisation of the pilot-verified serial phase: the
    scientific generators are unchanged and the parent remains the single
    writer for both layers.  Relaxed per-sample failure policy: a sample-local
    exception is a terminal per-sample ledger entry and the build continues;
    RU failure skips Topology/Trimer for that sample.  Resume skips samples
    whose RU and Topology states are both terminal (written, failed, or
    skipped).  Only parent/writer/infrastructure failures stop the build.
    """

    ru_writer = StagingWriter(staging / "ru_base", metadata["ru_base"])
    topology_writer = StagingWriter(staging / "topology", metadata["topology"])
    ru_ledger = _FailureLedger(staging / "ru_base")
    topology_ledger = _FailureLedger(staging / "topology")
    processed_new = 0
    writer_seconds = 0.0
    total = len(rows)
    started_wall = time.monotonic()
    last_report = 0.0

    def failure_counts():
        counter = Counter()
        for row in ru_ledger.rows.values():
            counter[f"ru_base:{row.get('failure_code', '?')}"] += 1
        for row in topology_ledger.rows.values():
            counter[f"topology:{row.get('failure_code', '?')}"] += 1
        return counter

    def unclassified_count():
        return sum(
            1 for row in list(ru_ledger.rows.values())
            + list(topology_ledger.rows.values())
            if row.get("failure_code") == UNCLASSIFIED
        )

    def progress(force=False):
        nonlocal last_report
        now = time.monotonic()
        if not force and now - last_report < 30.0:
            return
        last_report = now
        ru_failed = len(ru_ledger.rows)
        _progress_line(
            "struct", ru_writer.count() + ru_failed, total,
            {
                "ru_success": ru_writer.count(),
                "ru_failed": ru_failed,
                "topology_success": topology_writer.count(),
                "topology_failed": len(topology_ledger.rows),
                "skipped_parent_failed": ru_failed,
                "unclassified_sample_failures": unclassified_count(),
            },
            failure_counts(), started_wall, staging,
        )

    def commit_result(result):
        nonlocal processed_new, writer_seconds
        key = bytes.fromhex(result["sample_key"])
        status = result["status"]
        if status == "ru_failed":
            if key in ru_writer:
                raise CacheLifecycleError(
                    "ledger/data contradiction: RU record and RU failure "
                    f"for {result['sample_key']}"
                )
            ru_ledger.append(result["entry"], result.get("traceback_text"))
            processed_new += 1
            progress()
            return
        if status != "ok":
            raise CacheLifecycleError(
                f"unexpected struct worker status: {status}"
            )
        has_topo_payload = "topology_payload" in result
        has_topo_entry = "topology_entry" in result
        if has_topo_payload == has_topo_entry:
            raise CacheLifecycleError(
                "struct worker topology terminal state missing: "
                f"{result['sample_key']}"
            )
        if key not in ru_writer:
            ru_data = deserialize_record(result.pop("ru_payload"), key)
            inserted_ru, _, elapsed_ru = ru_writer.put(key, ru_data)
            if not inserted_ru:
                raise CacheLifecycleError("duplicate struct LMDB write")
            writer_seconds += elapsed_ru
        else:
            # resume re-run after a crash between the two layer writes
            result.pop("ru_payload")
        if has_topo_payload:
            topo_data = deserialize_record(result.pop("topology_payload"), key)
            if key not in topology_writer:
                inserted_topo, _, elapsed_topo = topology_writer.put(
                    key, topo_data
                )
                if not inserted_topo:
                    raise CacheLifecycleError("duplicate struct LMDB write")
                writer_seconds += elapsed_topo
        if has_topo_entry:
            entry = result.pop("topology_entry")
            traceback_text = result.pop("topology_traceback_text", None)
            if key in topology_writer:
                raise CacheLifecycleError(
                    "ledger/data contradiction: Topology record and Topology "
                    f"failure for {result['sample_key']}"
                )
            if key not in topology_ledger.rows:
                topology_ledger.append(entry, traceback_text)
        processed_new += 1
        progress()

    jobs = []
    for row in rows:
        key = bytes.fromhex(row["sample_key"])
        if key in ru_ledger.rows:
            continue  # RU failed: terminal, downstream skipped
        ru_written = key in ru_writer
        topo_terminal = key in topology_writer or key in topology_ledger.rows
        if not ru_written and topo_terminal:
            raise CacheLifecycleError(
                "staging contradiction: Topology terminal without RU record"
            )
        if ru_written and topo_terminal:
            continue
        jobs.append({**row, "phase": "struct"})
    struct_timeout = 600.0

    if int(workers) <= 1 or len(jobs) <= 1:
        try:
            for job in jobs:
                commit_result(_build_struct_one(job))
            progress(force=True)
            ru_writer.sync()
            topology_writer.sync()
        finally:
            ru_writer.close()
            topology_writer.close()
        return {"struct_new": processed_new,
                "struct_writer_seconds": writer_seconds}

    context = mp.get_context("spawn")

    def start_worker():
        parent, child = context.Pipe(duplex=True)
        process = context.Process(target=_worker_loop, args=(child,))
        process.start()
        child.close()
        return {"process": process, "connection": parent, "job": None,
                "job_id": None, "started": 0.0}

    states = [start_worker() for _ in range(min(int(workers), len(jobs)))]
    iterator = iter(jobs)
    next_job_id = 0
    retried = {}

    def worker_crash_result(job, reason):
        exitcode = None
        for state in states:
            if state["job"] is not None and \
                    state["job"]["sample_key"] == job["sample_key"]:
                exitcode = state["process"].exitcode
                break
        return {
            "status": "ru_failed", "sample_key": job["sample_key"],
            "entry": _failure_entry(
                job, "ru_base",
                failure_code="WORKER_CRASH", exception_type="WorkerDied",
                exception_message=f"{reason} (exitcode={exitcode})",
                elapsed_seconds=0.0,
            ),
        }

    def assign(index):
        nonlocal next_job_id
        state = states[index]
        try:
            job = next(iterator)
        except StopIteration:
            state["job"] = None
            return
        next_job_id += 1
        state.update(job=job, job_id=next_job_id, started=time.monotonic())
        try:
            state["connection"].send((next_job_id, job))
        except (EOFError, BrokenPipeError, OSError):
            # The job was never delivered: sample-local worker failure, the
            # build continues with a replacement worker (relaxed policy).
            _stop_worker(state)
            states[index] = start_worker()
            commit_result(worker_crash_result(job, "IPC send to worker failed"))
            assign(index)

    def mark_worker_crash(index, reason):
        state = states[index]
        failed_job = state["job"]
        started_at = state["started"]
        exitcode = state["process"].exitcode
        _stop_worker(state)
        states[index] = start_worker()
        commit_result({
            "status": "ru_failed", "sample_key": failed_job["sample_key"],
            "entry": _failure_entry(
                failed_job, "ru_base",
                failure_code="WORKER_CRASH", exception_type="WorkerDied",
                exception_message=f"{reason} (exitcode={exitcode})",
                elapsed_seconds=time.monotonic() - started_at,
            ),
        })
        assign(index)

    for index in range(len(states)):
        assign(index)
    try:
        while any(state["job"] is not None for state in states):
            progressed = False
            for index in range(len(states)):
                state = states[index]
                if state["job"] is None:
                    continue
                if state["connection"].poll():
                    try:
                        returned, result = pickle.loads(
                            state["connection"].recv_bytes()
                        )
                    except (EOFError, OSError):
                        mark_worker_crash(
                            index, "worker connection closed unexpectedly"
                        )
                        progressed = True
                        continue
                    if returned != state["job_id"]:
                        raise CacheLifecycleError("worker job identity mismatch")
                    commit_result(result)
                    assign(index)
                    progressed = True
                    continue
                if not state["process"].is_alive():
                    mark_worker_crash(index, "worker process died unexpectedly")
                    progressed = True
                    continue
                if time.monotonic() - state["started"] >= struct_timeout:
                    timed_out = state["job"]
                    _stop_worker(state)
                    states[index] = start_worker()
                    if retried.get(timed_out["sample_key"]):
                        # The deterministic retry also hung: this sample is a
                        # terminal per-sample failure, the build continues.
                        commit_result({
                            "status": "ru_failed",
                            "sample_key": timed_out["sample_key"],
                            "entry": _failure_entry(
                                timed_out, "ru_base",
                                failure_code="TIMEOUT",
                                exception_type="StructJobTimeout",
                                exception_message=(
                                    "struct job exceeded its timeout twice"
                                ),
                                elapsed_seconds=struct_timeout,
                            ),
                        })
                    else:
                        retried[timed_out["sample_key"]] = True
                        assign(index)
                    progressed = True
            if not progressed:
                time.sleep(0.01)
        progress(force=True)
    finally:
        try:
            ru_writer.sync()
            topology_writer.sync()
        finally:
            ru_writer.close()
            topology_writer.close()
            for state in states:
                try:
                    if state["process"].is_alive():
                        state["connection"].send(None)
                except Exception:
                    pass
            for state in states:
                _stop_worker(state)
    return {"struct_new": processed_new,
            "struct_writer_seconds": writer_seconds}


def _trimer_identity_carrier(normalized_smiles: str) -> Data:
    molecule = Chem.MolFromSmiles(str(normalized_smiles))
    if molecule is None:
        raise TrimerContractError("source_identity_corruption")
    ru, metadata = build_periodic_multimer_mol(
        molecule, num_repeat_units=1, close_periodic=False
    )
    count = int(metadata["base_atom_count"])
    carrier = Data()
    carrier.num_nodes = count
    carrier.canonical_ru_atom_index = torch.arange(count, dtype=torch.long)
    carrier.canonical_to_trimer_base_atom_id = torch.arange(count, dtype=torch.long)
    carrier.z = torch.tensor(
        [atom.GetAtomicNum() for atom in ru.GetAtoms()], dtype=torch.long
    )
    carrier.graph_available = True
    return carrier


def _build_struct_one(payload):
    """Compute the RU and Topology records for one sample inside a worker.

    Scheduling only: the scientific generators are the same functions the
    serial pilot path used, and the parent remains the single LMDB writer.
    Under the relaxed failure policy every sample-local Exception is a
    terminal per-sample failure result (RU or Topology layer); workers never
    die from Python-level sample exceptions.  KeyboardInterrupt and SystemExit
    are deliberately not caught.
    """

    started = time.monotonic()
    key = payload["sample_key"]
    # Test-only fault injection (integration tests prove the per-sample
    # failure path).  Inert unless the environment variable is set.
    fault_key = os.environ.get("MTS_CACHE_FAULT_UNCLASSIFIED_KEY")

    def ru_failure(exc):
        return {
            "status": "ru_failed", "sample_key": key,
            "entry": _failure_entry(
                payload, "ru_base",
                failure_code=UNCLASSIFIED,
                exception_type=type(exc).__name__,
                exception_message=str(exc),
                elapsed_seconds=time.monotonic() - started,
            ),
            "traceback_text": traceback.format_exc(),
        }

    try:
        if fault_key and key == fault_key:
            raise ValueError("unexpected sample-local condition")
        ru = _compute_ru_base_layer(payload["source_smiles"])
    except Exception as exc:
        return ru_failure(exc)
    if not bool(getattr(ru, "ru_base_valid", False)):
        failure_code, exception_type, message = _split_ru_failure_code(
            getattr(ru, "ru_base_failure_code", "")
        )
        return {
            "status": "ru_failed", "sample_key": key,
            "entry": _failure_entry(
                payload, "ru_base",
                failure_code=failure_code,
                exception_type=exception_type,
                exception_message=message,
                elapsed_seconds=time.monotonic() - started,
            ),
        }
    try:
        ru_payload = serialize_record(
            bytes.fromhex(key), select_record_fields("ru_base", ru)
        )
    except Exception as exc:
        return ru_failure(exc)
    topology_entry = None
    topology_traceback = None
    topology_payload = None
    try:
        topology = _compute_topology_layer_impl(
            payload["source_smiles"], ru, max_hops=int(
                ROUTE_BUILD_SPECS["topology"]["parameters"]["max_hops"]
            )
        )
        topology_payload = serialize_record(
            bytes.fromhex(key), select_record_fields("topology", topology)
        )
    except Exception as exc:
        # A Topology failure never blocks the Trimer layer: both depend only
        # on the RU record.
        topology_entry = _failure_entry(
            payload, "topology",
            failure_code=UNCLASSIFIED,
            exception_type=type(exc).__name__,
            exception_message=str(exc),
            elapsed_seconds=time.monotonic() - started,
        )
        topology_traceback = traceback.format_exc()
    result = {
        "status": "ok", "sample_key": key, "ru_payload": ru_payload,
    }
    if topology_payload is not None:
        result["topology_payload"] = topology_payload
    if topology_entry is not None:
        result["topology_entry"] = topology_entry
        result["topology_traceback_text"] = topology_traceback
    return result


def _build_trimer_one(payload):
    started = time.monotonic()
    key = payload["sample_key"]
    # Test-only fault injection (integration tests prove the per-sample
    # failure path).  Inert unless the environment variable is set.
    fault_key = os.environ.get("MTS_CACHE_FAULT_SAMPLE_KEY")
    try:
        if fault_key and key == fault_key:
            raise ValueError("unexpected sample-local condition")
        carrier = _trimer_identity_carrier(payload["normalized_smiles"])
        attach_finite_trimer_mcl(
            carrier,
            payload["normalized_smiles"],
            build_spec=ROUTE_BUILD_SPECS["trimer"],
            sample_key=bytes.fromhex(payload["sample_key"]),
        )
        diagnostics = dict(getattr(carrier, "generation_diagnostics", {}))
        data = select_record_fields("trimer", carrier)
        return {
            "status": "accepted", "sample_key": key,
            "payload": serialize_record(bytes.fromhex(key), data),
            "elapsed_seconds": time.monotonic() - started,
            "candidate_attempts": sum(
                len(row.get("candidates", []))
                for row in diagnostics.get("rounds", [])
            ),
            "round_reached": int(getattr(carrier, "trimer_conformer_round_id", -1)),
        }
    except TrimerGeometryRejection as exc:
        return {
            "status": "trimer_failed", "sample_key": key,
            "entry": _failure_entry(
                payload, "trimer",
                failure_code=str(exc.code),
                exception_type="TrimerGeometryRejection",
                exception_message=str(exc.last_embed_failure or exc.code),
                elapsed_seconds=float(exc.elapsed_seconds),
                candidate_attempts=int(exc.candidate_attempts),
                round_reached=int(
                    exc.round_reached if exc.round_reached is not None else -1
                ),
            ),
        }
    except Exception as exc:
        # Any other sample-local exception (including former contract errors
        # such as stereo/mapping contradictions) is a terminal per-sample
        # Trimer failure; the build continues.
        return {
            "status": "trimer_failed", "sample_key": key,
            "entry": _failure_entry(
                payload, "trimer",
                failure_code=UNCLASSIFIED,
                exception_type=type(exc).__name__,
                exception_message=str(exc),
                elapsed_seconds=time.monotonic() - started,
            ),
            "traceback_text": traceback.format_exc(),
        }


def _worker_loop(connection):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    torch.set_num_threads(1)
    try:
        while True:
            message = connection.recv()
            if message is None:
                return
            job_id, payload = message
            if payload.get("phase") == "struct":
                result = _build_struct_one(payload)
            else:
                result = _build_trimer_one(payload)
            connection.send_bytes(pickle.dumps((job_id, result), protocol=5))
    except (EOFError, BrokenPipeError, KeyboardInterrupt):
        return
    finally:
        connection.close()


def _stop_worker(state):
    try:
        state["connection"].close()
    except Exception:
        pass
    process = state["process"]
    if process.is_alive():
        process.terminate()
        process.join(timeout=2)
    if process.is_alive():
        process.kill()
    process.join(timeout=2)


def _run_trimer(staging, rows, metadata, workers, interrupt_after):
    writer = StagingWriter(staging / "trimer", metadata["trimer"])
    runtime_path = staging / "trimer" / "runtime.jsonl"
    runtime_path.touch(exist_ok=True)
    ledger = _FailureLedger(staging / "trimer")
    rejections = ledger.rows
    runtime = {}
    if runtime_path.is_file():
        for line in runtime_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                item = json.loads(line)
                key = bytes.fromhex(item["sample_key"])
                if key in runtime:
                    raise CacheLifecycleError("duplicate runtime ledger key")
                runtime[key] = item
    accepted_existing = {
        bytes.fromhex(row["sample_key"]) for row in rows
        if bytes.fromhex(row["sample_key"]) in writer
    }
    if accepted_existing & set(rejections):
        raise CacheLifecycleError("accepted/rejected staging overlap")
    ru_rejections = read_rejections(staging / "ru_base" / "rejections.jsonl")
    topology_failures = read_rejections(
        staging / "topology" / "rejections.jsonl"
    )
    jobs = []
    ru_environment = _raw_artifact(staging / "ru_base")
    try:
        with ru_environment.begin(buffers=False) as transaction:
            for row in rows:
                key = bytes.fromhex(row["sample_key"])
                if key in accepted_existing or key in rejections:
                    continue
                if key in ru_rejections:
                    continue  # RU capability boundary: never reaches Trimer
                payload = transaction.get(key)
                if payload is None:
                    raise CacheLifecycleError("Trimer parent RU record is missing")
                ru = deserialize_record(bytes(payload), key)
                normalized = str(ru.normalized_polymer_smiles)
                if normalized != row["normalized_smiles"]:
                    raise TrimerContractError("source_to_RU_identity_corruption")
                jobs.append({**row, "normalized_smiles": normalized})
    finally:
        ru_environment.close()
    resume = bool(accepted_existing or rejections)
    hard_timeout = float(
        ROUTE_BUILD_SPECS["trimer"]["parameters"]["hard_timeout_seconds"]
    )
    processed_new = 0
    writer_seconds = 0.0
    started_wall = time.monotonic()
    last_progress = 0.0

    def commit_result(result):
        nonlocal processed_new, writer_seconds, last_progress
        key = bytes.fromhex(result["sample_key"])
        status = result["status"]
        if status == "accepted":
            data = deserialize_record(result.pop("payload"), key)
            inserted, payload_bytes, elapsed = writer.put(key, data)
            if not inserted:
                raise CacheLifecycleError("duplicate accepted Trimer write")
            writer_seconds += elapsed
            runtime_row = {
                "sample_key": key.hex(), "status": "accepted",
                "elapsed_seconds": float(result["elapsed_seconds"]),
                "candidate_attempts": int(result["candidate_attempts"]),
                "round_reached": int(result["round_reached"]),
                "record_bytes": int(payload_bytes),
                "writer_seconds": float(elapsed),
            }
        elif status == "trimer_failed":
            entry = result["entry"]
            ledger.append(entry, result.get("traceback_text"))
            runtime_row = {
                "sample_key": key.hex(), "status": "rejected",
                "elapsed_seconds": float(entry["elapsed_seconds"]),
                "candidate_attempts": int(entry["candidate_attempts"]),
                "round_reached": int(entry["round_reached"]),
                "record_bytes": 0, "writer_seconds": 0.0,
            }
        else:
            raise CacheLifecycleError(
                f"unexpected trimer worker status: {status}"
            )
        if key not in runtime:
            append_jsonl(runtime_path, runtime_row)
            runtime[key] = runtime_row
        processed_new += 1
        if (processed_new % 500 == 0 or time.monotonic() - last_progress >= 30.0):
            last_progress = time.monotonic()
            accepted_now = sum(
                1 for row in runtime.values() if row["status"] == "accepted"
            )
            rejected_now = len(rejections)
            ru_failed_now = len(ru_rejections)
            combined = Counter(
                f"ru_base:{row.get('failure_code', '?')}"
                for row in ru_rejections.values()
            )
            combined.update(
                f"topology:{row.get('failure_code', '?')}"
                for row in topology_failures.values()
            )
            combined.update(
                f"trimer:{row.get('failure_code', '?')}"
                for row in rejections.values()
            )
            _progress_line(
                "trimer", len(runtime), len(rows),
                {
                    "ru_success": len(rows) - ru_failed_now,
                    "ru_failed": ru_failed_now,
                    "topology_success": (
                        len(rows) - ru_failed_now - len(topology_failures)
                    ),
                    "topology_failed": len(topology_failures),
                    "skipped_parent_failed": ru_failed_now,
                    "trimer_success": accepted_now,
                    "trimer_failed": rejected_now,
                    "unclassified_sample_failures": sum(
                        1 for row in list(ru_rejections.values())
                        + list(topology_failures.values())
                        + list(rejections.values())
                        if row.get("failure_code") == UNCLASSIFIED
                    ),
                },
                combined, started_wall, staging,
            )
        if interrupt_after and processed_new >= int(interrupt_after):
            writer.sync()
            raise IntentionalBuildInterrupt(
                f"intentional interrupt after {processed_new} new terminal keys"
            )

    if int(workers) <= 1:
        try:
            for job in jobs:
                commit_result(_build_trimer_one(job))
        finally:
            writer.close()
        return resume, processed_new, writer_seconds

    context = mp.get_context("spawn")

    def start_worker():
        parent, child = context.Pipe(duplex=True)
        process = context.Process(target=_worker_loop, args=(child,))
        process.start()
        child.close()
        return {"process": process, "connection": parent, "job": None,
                "job_id": None, "started": 0.0}

    states = [start_worker() for _ in range(min(int(workers), len(jobs)))]
    iterator = iter(jobs)
    next_job_id = 0

    def assign(index):
        nonlocal next_job_id
        state = states[index]
        try:
            job = next(iterator)
        except StopIteration:
            state["job"] = None
            return
        next_job_id += 1
        state.update(job=job, job_id=next_job_id, started=time.monotonic())
        try:
            state["connection"].send((next_job_id, job))
        except (EOFError, BrokenPipeError, OSError):
            # Job never delivered: per-sample worker failure, build continues.
            _stop_worker(state)
            states[index] = start_worker()
            commit_result({
                "status": "trimer_failed", "sample_key": job["sample_key"],
                "entry": _failure_entry(
                    job, "trimer",
                    failure_code="WORKER_CRASH", exception_type="WorkerDied",
                    exception_message="IPC send to worker failed",
                    elapsed_seconds=0.0,
                ),
            })
            assign(index)

    def mark_worker_crash(index, reason):
        state = states[index]
        failed_job = state["job"]
        started_at = state["started"]
        exitcode = state["process"].exitcode
        _stop_worker(state)
        states[index] = start_worker()
        commit_result({
            "status": "trimer_failed", "sample_key": failed_job["sample_key"],
            "entry": _failure_entry(
                failed_job, "trimer",
                failure_code="WORKER_CRASH", exception_type="WorkerDied",
                exception_message=f"{reason} (exitcode={exitcode})",
                elapsed_seconds=time.monotonic() - started_at,
            ),
        })
        assign(index)

    for index in range(len(states)):
        assign(index)
    try:
        while any(state["job"] is not None for state in states):
            progressed = False
            for index in range(len(states)):
                state = states[index]
                if state["job"] is None:
                    continue
                if state["connection"].poll():
                    try:
                        returned, result = pickle.loads(
                            state["connection"].recv_bytes()
                        )
                    except (EOFError, OSError):
                        mark_worker_crash(
                            index, "worker connection closed unexpectedly"
                        )
                        progressed = True
                        continue
                    if returned != state["job_id"]:
                        raise CacheLifecycleError("worker job identity mismatch")
                    commit_result(result)
                    assign(index)
                    progressed = True
                    continue
                if not state["process"].is_alive():
                    mark_worker_crash(index, "worker process died unexpectedly")
                    progressed = True
                    continue
                if time.monotonic() - state["started"] >= hard_timeout:
                    timed_out = state["job"]
                    _stop_worker(state)
                    states[index] = start_worker()
                    commit_result({
                        "status": "trimer_failed",
                        "sample_key": timed_out["sample_key"],
                        "entry": _failure_entry(
                            timed_out, "trimer",
                            failure_code="TIMEOUT",
                            exception_type="SampleTimeout",
                            exception_message=(
                                "trimer exceeded the shared hard deadline"
                            ),
                            elapsed_seconds=hard_timeout,
                        ),
                    })
                    assign(index)
                    progressed = True
            if not progressed:
                time.sleep(0.01)
    finally:
        try:
            writer.close()
        finally:
            for state in states:
                try:
                    if state["process"].is_alive():
                        state["connection"].send(None)
                except Exception:
                    pass
            for state in states:
                _stop_worker(state)
    return resume, processed_new, writer_seconds


def _raw_artifact(root):
    environment = lmdb.open(
        str(root / "data.lmdb"), subdir=True, readonly=True, lock=False,
        readahead=False, meminit=False,
    )
    return environment


def _check_fields(layer, data, key):
    keys = set(data.keys())
    allowed = set(RECORD_FIELDS[layer])
    if layer == "topology":
        # select_record_fields writes the derived node count explicitly; it
        # is a scalar projection of mips_x, not an unrecorded alias.
        allowed.add("num_nodes")
    if not keys <= allowed:
        raise CacheAuditError(f"{layer} record has aliases/extra fields: {sorted(keys-allowed)}")
    required = set(REQUIRED_FIELDS[layer])
    if layer == "topology":
        required.discard("x")
        required.add("mips_x")
    missing = required - keys
    if missing:
        raise CacheAuditError(f"{layer} record missing required fields: {sorted(missing)}")
    if layer == "ru_base":
        z = torch.as_tensor(data.ru_atomic_number)
        edge = torch.as_tensor(data.ru_edge_index)
        if z.ndim != 1 or edge.ndim != 2 or edge.size(0) != 2:
            raise CacheAuditError("invalid RU record shape")
    elif layer == "topology":
        x, z = torch.as_tensor(data.mips_x), torch.as_tensor(data.z)
        relation = torch.as_tensor(data.lga_edge_index)
        path = torch.as_tensor(data.lga_path_index)
        if x.ndim != 2 or x.size(1) != 137 or z.shape != (x.size(0),):
            raise CacheAuditError("invalid Topology atom shape")
        if relation.ndim != 2 or relation.size(0) != 2 or path.size(0) != relation.size(1):
            raise CacheAuditError("invalid Topology relation shape")
    else:
        pos = torch.as_tensor(data.trimer_pos)
        z = torch.as_tensor(data.trimer_atomic_number)
        edge = torch.as_tensor(data.trimer_edge_index)
        if pos.ndim != 2 or pos.size(1) != 3 or z.shape != (pos.size(0),):
            raise CacheAuditError("invalid Trimer atom shape")
        if not bool(torch.isfinite(pos).all()):
            raise CacheAuditError("non-finite Trimer coordinates")
        if edge.ndim != 2 or edge.size(0) != 2:
            raise CacheAuditError("invalid Trimer edge shape")
        if not bool(data.trimer_geometry_valid) or not bool(data.trimer_geometry_is_3d):
            raise CacheAuditError("accepted Trimer has invalid geometry flags")


def _audit_layer(root, layer, source_keys, expected_keys, source_manifest_hash,
                 artifact_hash, build_spec, parents, rejected_keys=()):
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    expected_metadata = _metadata(layer, artifact_hash, source_manifest_hash, parents)
    if metadata != expected_metadata:
        raise CacheAuditError(f"{layer} metadata differs from generator build_spec")
    environment = _raw_artifact(root)
    digest = __import__("hashlib").sha256()
    observed = []
    payload_bytes = 0
    try:
        with environment.begin(buffers=False) as txn:
            cursor = txn.cursor()
            for key, payload in cursor:
                key, payload = bytes(key), bytes(payload)
                observed.append(key)
                digest.update(key)
                digest.update(len(payload).to_bytes(8, "little"))
                digest.update(payload)
                payload_bytes += len(payload)
                data = deserialize_record(payload, key)
                _check_fields(layer, data, key)
            lmdb_count = int(txn.stat()["entries"])
    finally:
        environment.close()
    if len(observed) != len(set(observed)) or lmdb_count != len(observed):
        raise CacheAuditError(f"{layer} LMDB key uniqueness/count failure")
    if set(observed) != set(expected_keys):
        raise CacheAuditError(f"{layer} LMDB key coverage failure")
    rejected_keys = list(rejected_keys)
    manifest = {
        "artifact_type": layer,
        "artifact_hash": artifact_hash,
        "build_spec_hash": build_spec_hash(build_spec),
        "source_manifest_hash": source_manifest_hash,
        "parents": dict(sorted(parents.items())),
        "source_count": len(source_keys),
        "record_count": len(observed),
        "accepted_count": len(expected_keys),
        "rejected_count": len(rejected_keys),
        "ordered_accepted_key_hash": ordered_key_hash(expected_keys),
        "ordered_rejected_key_hash": ordered_key_hash(rejected_keys),
        "data_digest": digest.hexdigest(),
        "payload_bytes": payload_bytes,
        "data_file_sha256": sha256_file(root / "data.lmdb" / "data.mdb"),
    }
    return manifest


def _audit_freeze_publish(cache_root, staging, final, rows, source_manifest,
                          artifact_hashes, metadata):
    source_keys = [bytes.fromhex(row["sample_key"]) for row in rows]
    source_set = set(source_keys)

    # RU layer: per-sample failures exclude a sample from RU, Topology and
    # Trimer entirely (downstream layers are SKIPPED_PARENT_FAILED).
    ru_rejection_path = staging / "ru_base" / "rejections.jsonl"
    ru_rejection_path.touch(exist_ok=True)
    ru_rejections = read_rejections(ru_rejection_path)
    ru_rejected_set = set(ru_rejections)
    if ru_rejected_set - source_set:
        raise CacheAuditError("RU rejection ledger contains foreign keys")
    ru_accepted = [key for key in source_keys if key not in ru_rejected_set]
    ru_rejected = [key for key in source_keys if key in ru_rejected_set]
    ru_accepted_set = set(ru_accepted)

    # Topology layer: a per-sample failure is terminal for Topology only and
    # never blocks the Trimer layer (both depend only on the RU record).
    topology_rejection_path = staging / "topology" / "rejections.jsonl"
    topology_rejection_path.touch(exist_ok=True)
    topology_rejections = read_rejections(topology_rejection_path)
    topo_failed_set = set(topology_rejections)
    if topo_failed_set - ru_accepted_set:
        raise CacheAuditError("Topology failure ledger contains foreign keys")
    topo_env = _raw_artifact(staging / "topology")
    try:
        with topo_env.begin(buffers=False) as txn:
            topo_accepted_set = {bytes(key) for key, _ in txn.cursor()}
    finally:
        topo_env.close()
    if topo_accepted_set & topo_failed_set:
        raise CacheAuditError("Topology accepted/failed are not disjoint")
    if topo_accepted_set | topo_failed_set != ru_accepted_set:
        raise CacheAuditError(
            "Topology accepted/failed do not cover the RU-accepted cohort"
        )
    topo_accepted = [key for key in ru_accepted if key in topo_accepted_set]
    topo_failed = [key for key in ru_accepted if key in topo_failed_set]

    rejection_path = staging / "trimer" / "rejections.jsonl"
    rejections = read_rejections(rejection_path)
    trimer_env = _raw_artifact(staging / "trimer")
    try:
        with trimer_env.begin(buffers=False) as txn:
            accepted_set = {bytes(key) for key, _ in txn.cursor()}
    finally:
        trimer_env.close()
    rejected_set = set(rejections)
    if accepted_set & rejected_set:
        raise CacheAuditError("accepted/rejected are not disjoint")
    if accepted_set | rejected_set != ru_accepted_set:
        raise CacheAuditError(
            "Trimer accepted/rejected do not cover the RU-accepted cohort"
        )
    accepted = [key for key in ru_accepted if key in accepted_set]
    rejected = [key for key in ru_accepted if key in rejected_set]
    if len(ru_accepted) != len(accepted) + len(rejected):
        raise CacheAuditError("terminal accounting mismatch")
    runtime_rows = []
    runtime_keys = set()
    runtime_path = staging / "trimer" / "runtime.jsonl"
    for line in runtime_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        key = bytes.fromhex(item["sample_key"])
        if key in runtime_keys:
            raise CacheAuditError("runtime ledger contains duplicate keys")
        runtime_keys.add(key)
        runtime_rows.append(item)
    if runtime_keys != ru_accepted_set:
        raise CacheAuditError(
            "runtime ledger does not cover the RU-accepted cohort"
        )

    manifests = {}
    parent_map = {
        "ru_base": {},
        "topology": {"ru_base": artifact_hashes["ru_base"]},
        "trimer": {"ru_base": artifact_hashes["ru_base"]},
    }
    layer_expected = {
        "ru_base": (source_keys, ru_accepted, ru_rejected, source_keys),
        "topology": (ru_accepted, topo_accepted, topo_failed, ru_accepted),
        "trimer": (ru_accepted, accepted, rejected, ru_accepted),
    }
    for layer in ("ru_base", "topology", "trimer"):
        cohort, expected, rejected_for_layer, terminal = layer_expected[layer]
        manifest = _audit_layer(
            staging / layer, layer, cohort, expected,
            source_manifest["source_manifest_hash"], artifact_hashes[layer],
            ROUTE_BUILD_SPECS[layer], parent_map[layer], rejected_for_layer,
        )
        # Every rejecting layer publishes its ordered accepted/rejected key
        # cohorts; topology is a pure pass-through of the RU cohort.
        accepted_for_layer = [key for key in cohort if key in set(expected)]
        accepted_array = np.frombuffer(
            b"".join(accepted_for_layer), dtype=np.uint8
        ).reshape(-1, 32)
        rejected_array = np.frombuffer(
            b"".join(rejected_for_layer), dtype=np.uint8
        ).reshape(-1, 32)
        for name, values in (("accepted_keys.npy", accepted_array),
                             ("rejected_keys.npy", rejected_array)):
            temporary = staging / layer / f"{name}.tmp.{os.getpid()}"
            with temporary.open("wb") as handle:
                np.save(handle, values, allow_pickle=False)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, staging / layer / name)
        manifest["accepted_keys_file_sha256"] = sha256_file(
            staging / layer / "accepted_keys.npy"
        )
        manifest["rejected_keys_file_sha256"] = sha256_file(
            staging / layer / "rejected_keys.npy"
        )
        manifest["rejections_file_sha256"] = sha256_file(
            staging / layer / "rejections.jsonl"
        )
        manifest["unclassified_failure_count"] = sum(
            1 for row in read_rejections(
                staging / layer / "rejections.jsonl"
            ).values()
            if row.get("failure_code") == UNCLASSIFIED
        )
        if layer == "trimer":
            manifest["runtime_file_sha256"] = sha256_file(
                staging / layer / "runtime.jsonl"
            )
        manifests[layer] = manifest

    # Join audit is independent of generation: exact published Topology and
    # Trimer records must agree on central atom identity for every key that
    # has both records.  A Topology failure removes only the Topology record;
    # its Trimer record stays valid without a join check.
    topo_env = _raw_artifact(staging / "topology")
    tri_env = _raw_artifact(staging / "trimer")
    try:
        with topo_env.begin() as topo_txn, tri_env.begin() as tri_txn:
            for key in accepted:
                if key not in topo_accepted_set:
                    continue
                topology = deserialize_record(bytes(topo_txn.get(key)), key)
                trimer = deserialize_record(bytes(tri_txn.get(key)), key)
                mapping = torch.as_tensor(trimer.mips_to_trimer_central_index).long()
                positions = torch.as_tensor(trimer.trimer_pos)
                central = torch.as_tensor(trimer.trimer_central_ru_mask).bool()
                if mapping.numel() != torch.as_tensor(topology.mips_x).size(0):
                    raise CacheAuditError("O8/Trimer mapping length mismatch")
                if mapping.numel() and (
                    int(mapping.min()) < 0 or int(mapping.max()) >= positions.size(0)
                    or not bool(central[mapping].all())
                    or not torch.equal(
                        torch.as_tensor(topology.z).long(),
                        torch.as_tensor(trimer.trimer_atomic_number).long()[mapping],
                    )
                ):
                    raise CacheAuditError("O8/Trimer atom identity corruption")
    finally:
        topo_env.close()
        tri_env.close()

    for layer, manifest in manifests.items():
        atomic_json(staging / layer / "manifest.json", manifest)
    # Only after every layer and the cross-layer join have passed do markers
    # appear.  Each marker binds exactly one final manifest hash.
    for layer, manifest in manifests.items():
        lock_path = staging / layer / ".writer.lock"
        if lock_path.exists():
            lock_path.unlink()
        atomic_json(
            staging / layer / ".frozen",
            {"manifest_hash": json_hash(manifest)},
        )
    bundle_manifest = {
        "bundle_hash": final.name,
        "source_manifest_hash": source_manifest["source_manifest_hash"],
        "artifacts": {
            layer: {
                "artifact_hash": artifact_hashes[layer],
                "manifest_hash": json_hash(manifests[layer]),
            }
            for layer in manifests
        },
    }
    atomic_json(staging / "bundle_manifest.json", bundle_manifest)
    if final.exists():
        raise CacheLifecycleError(f"published bundle already exists: {final}")
    os.replace(staging, final)

    artifacts = {}
    for layer, manifest in manifests.items():
        artifacts[layer] = {
            "path": str((final / layer).relative_to(cache_root)),
            "artifact_hash": artifact_hashes[layer],
            "manifest_hash": json_hash(manifest),
            "build_spec": ROUTE_BUILD_SPECS[layer],
            "build_spec_hash": build_spec_hash(ROUTE_BUILD_SPECS[layer]),
            "source_manifest_hash": source_manifest["source_manifest_hash"],
            "ordered_accepted_key_hash": manifest["ordered_accepted_key_hash"],
            "record_count": manifest["record_count"],
            "parents": manifest["parents"],
        }
    store = {
        "bundle_hash": final.name,
        "source": {
            "path": str((final / "source").relative_to(cache_root)),
            "source_manifest_hash": source_manifest["source_manifest_hash"],
            "source_count": len(source_keys),
        },
        "artifacts": artifacts,
    }
    atomic_json(cache_root / "store.json", store)
    return store, manifests, rejections


def _existing_bundle(cache_root, final):
    store_path = cache_root / "store.json"
    if not final.is_dir() or not store_path.is_file():
        return False
    store = json.loads(store_path.read_text(encoding="utf-8"))
    if "bundle_hash" not in store:
        # Legacy pre-lifecycle active-store binding: production publish will
        # supersede it atomically; historical artifact directories stay.
        print("legacy-format store.json present; publish will supersede it")
        return False
    if store.get("bundle_hash") != final.name:
        raise CacheLifecycleError("published bundle exists but is not active")
    before = snapshot_tree(cache_root)
    dataset = PublishedCacheDataset(cache_root)
    try:
        if len(dataset):
            _ = dataset[0]
    finally:
        dataset.close()
    if snapshot_tree(cache_root) != before:
        raise CacheLifecycleError("readonly duplicate execution modified cache")
    print(f"existing published bundle verified read-only: {final}")
    return True


def _make_report(cache_root, store, manifests, rejections, staging_existed,
                 phase_stats, build_started, source_rows):
    trimer_root = cache_root / store["artifacts"]["trimer"]["path"]
    runtime = [
        json.loads(line) for line in (trimer_root / "runtime.jsonl").read_text(
            encoding="utf-8"
        ).splitlines() if line.strip()
    ]
    elapsed = [float(row["elapsed_seconds"]) for row in runtime]
    accepted_runtime = [row for row in runtime if row["status"] == "accepted"]
    source_count = len(source_rows)
    accepted_count = manifests["trimer"]["accepted_count"]
    rejected_count = manifests["trimer"]["rejected_count"]
    total_wall = time.monotonic() - build_started
    full_rows = 1_000_000
    ru_manifest = manifests["ru_base"]
    topo_manifest = manifests["topology"]
    ru_accepted_count = ru_manifest["accepted_count"]
    ru_rejected_count = ru_manifest["rejected_count"]
    topo_accepted_count = topo_manifest["accepted_count"]
    topo_failed_count = topo_manifest["rejected_count"]
    unclassified_failure_count = sum(
        manifest.get("unclassified_failure_count", 0)
        for manifest in manifests.values()
    )
    bundle_root = trimer_root.parent
    acceptance = accepted_count / source_count
    mean_cpu = float(np.mean(elapsed)) if elapsed else 0.0
    workers = int(phase_stats["workers"])
    projected_cpu_hours = mean_cpu * full_rows / 3600.0
    efficiency = 0.85
    projected_wall_hours = projected_cpu_hours / max(1, workers) / efficiency
    data_bytes = sum(
        (cache_root / binding["path"] / "data.lmdb" / "data.mdb").stat().st_size
        for binding in store["artifacts"].values()
    )
    before = snapshot_tree(cache_root)
    dataset = PublishedCacheDataset(cache_root)
    try:
        for index in range(min(8, len(dataset))):
            _ = dataset[index]
    finally:
        dataset.close()
    zero_write = snapshot_tree(cache_root) == before
    return {
        "scope": "cache lifecycle build report (per-source-layer terminal accounting)",
        "source_count": source_count,
        "ru_success_count": ru_accepted_count,
        "ru_failed_count": ru_rejected_count,
        "ru_accepted_count": ru_accepted_count,
        "ru_rejected_count": ru_rejected_count,
        "ru_rejection_distribution": dict(sorted(Counter(
            row["failure_code"] for row in read_rejections(
                bundle_root / "ru_base" / "rejections.jsonl"
            ).values()).items())),
        "topology_success_count": topo_accepted_count,
        "topology_failed_count": topo_failed_count,
        "topology_skipped_parent_failed": ru_rejected_count,
        "topology_failure_distribution": failure_distribution(read_rejections(
            bundle_root / "topology" / "rejections.jsonl")),
        "trimer_success_count": accepted_count,
        "trimer_failed_count": rejected_count,
        "trimer_skipped_parent_failed": ru_rejected_count,
        "unclassified_failure_count": unclassified_failure_count,
        "p_ru_accepted": ru_accepted_count / source_count,
        "p_topology_accepted_given_ru": (
            topo_accepted_count / ru_accepted_count if ru_accepted_count else None
        ),
        "p_trimer_accepted_given_ru": (
            accepted_count / ru_accepted_count if ru_accepted_count else None
        ),
        "p_geometry": accepted_count / source_count,
        "accepted_count": accepted_count,
        "rejected_count": rejected_count,
        "failure_distribution": failure_distribution(rejections),
        "accounting_ok": (
            source_count == ru_accepted_count + ru_rejected_count
            and ru_accepted_count == topo_accepted_count + topo_failed_count
            and ru_accepted_count == accepted_count + rejected_count
        ),
        "resume_detected": bool(staging_existed),
        "resume_skipped_terminal": int(phase_stats["resume_skipped_terminal"]),
        "duplicate_keys": 0,
        "runtime_seconds": {
            "median": float(np.median(elapsed)) if elapsed else None,
            "p90": float(np.percentile(elapsed, 90)) if elapsed else None,
            "p95": float(np.percentile(elapsed, 95)) if elapsed else None,
        },
        "bytes": {
            "accepted_record_mean": (
                float(np.mean([row["record_bytes"] for row in accepted_runtime]))
                if accepted_runtime else None
            ),
            "lmdb_bytes_per_source": data_bytes / source_count,
            "total_lmdb_bytes": data_bytes,
            "manifest_sidecar_overhead": sum(
                path.stat().st_size for path in cache_root.rglob("*")
                if path.is_file() and "data.lmdb" not in path.parts
            ),
        },
        "throughput_samples_per_second": {
            "generation_cpu_equivalent": source_count / max(sum(elapsed), 1e-9),
            "end_to_end_wall": source_count / max(total_wall, 1e-9),
            "writer": accepted_count / max(
                sum(float(row["writer_seconds"]) for row in accepted_runtime), 1e-9
            ),
        },
        "estimates_for_1m_raw_rows": {
            "accepted": int(round(full_rows * acceptance)),
            "rejected": int(round(full_rows * (1.0 - acceptance))),
            "disk_bytes": int(round(data_bytes / source_count * full_rows)),
            "cpu_hours": projected_cpu_hours,
            "wall_hours_at_pilot_workers_and_85pct_efficiency": projected_wall_hours,
            "assumption": "linear extrapolation from this fixed pilot; canonical dedup not applied to 1m estimate",
        },
        "recommended_workers": workers,
        "recommended_layout": "one LMDB per RU/Topology/Trimer layer; no extra sharding at projected size",
        "zero_write_ok": bool(zero_write),
        "full_rebuild_ready": bool(
            zero_write
            and source_count == ru_accepted_count + ru_rejected_count
            and ru_accepted_count == topo_accepted_count + topo_failed_count
            and ru_accepted_count == accepted_count + rejected_count
            and staging_existed and phase_stats["resume_skipped_terminal"] > 0
        ),
        "store": store,
        "manifests": manifests,
    }


def _raise_interrupt(signum, frame):
    raise KeyboardInterrupt(f"signal {signum}")


def main(argv=None):
    import signal
    signal.signal(signal.SIGTERM, _raise_interrupt)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--source-csv", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--interrupt-after", type=int, default=0)
    parser.add_argument("--report-json", type=Path)
    args = parser.parse_args(argv)
    if args.limit != 0 and not 100 <= args.limit <= 5000:
        parser.error("--limit must be 0 (full production source) or in [100, 5000]")
    cache_root = args.cache_root.resolve()
    cache_root.mkdir(parents=True, exist_ok=True)
    rows, source_manifest = load_source_rows(args.source_csv, args.limit)
    source_hash = source_manifest["source_manifest_hash"]
    artifact_hashes = {}
    artifact_hashes["ru_base"] = artifact_identity("ru_base", source_hash, {})
    artifact_hashes["topology"] = artifact_identity(
        "topology", source_hash, {"ru_base": artifact_hashes["ru_base"]}
    )
    artifact_hashes["trimer"] = artifact_identity(
        "trimer", source_hash, {"ru_base": artifact_hashes["ru_base"]}
    )
    bundle_hash = bundle_identity(source_hash, artifact_hashes)
    builds_root = cache_root / "builds"
    builds_root.mkdir(exist_ok=True)
    staging = builds_root / f"{bundle_hash}.staging"
    final = builds_root / bundle_hash
    if _existing_bundle(cache_root, final):
        return 0
    staging_existed = staging.exists()
    staging.mkdir(exist_ok=True)
    _write_source(staging, rows, source_manifest)
    parents = {
        "ru_base": {},
        "topology": {"ru_base": artifact_hashes["ru_base"]},
        "trimer": {"ru_base": artifact_hashes["ru_base"]},
    }
    metadata = {
        layer: _metadata(layer, artifact_hashes[layer], source_hash, parents[layer])
        for layer in ("ru_base", "topology", "trimer")
    }
    build_started = time.monotonic()
    try:
        phase_stats = _run_struct(staging, rows, metadata, args.workers)
        trimer_root = staging / "trimer"
        pre_terminal = 0
        if trimer_root.exists():
            counter_writer = StagingWriter(trimer_root, metadata["trimer"])
            try:
                pre_terminal = counter_writer.count()
            finally:
                counter_writer.close()
            pre_terminal += len(read_rejections(trimer_root / "rejections.jsonl"))
        resume, processed_new, writer_seconds = _run_trimer(
            staging, rows, metadata, args.workers, args.interrupt_after
        )
        phase_stats.update({
            "workers": int(args.workers),
            "trimer_new_terminal": int(processed_new),
            "trimer_writer_seconds": float(writer_seconds),
            "resume_skipped_terminal": int(pre_terminal if resume else 0),
        })
        store, manifests, rejections = _audit_freeze_publish(
            cache_root, staging, final, rows, source_manifest,
            artifact_hashes, metadata,
        )
        report = _make_report(
            cache_root, store, manifests, rejections,
            staging_existed or resume, phase_stats, build_started, rows,
        )
        if args.report_json:
            args.report_json.parent.mkdir(parents=True, exist_ok=True)
            atomic_json(args.report_json.resolve(), report)
        print(json.dumps(report, sort_keys=True, indent=2))
        # Exit code reflects build success and accounting integrity.  The
        # resume-evidence flag stays in the report: a first full run cannot
        # have resumed anything by definition.
        return 0 if report["accounting_ok"] and report["zero_write_ok"] else 2
    except (IntentionalBuildInterrupt, KeyboardInterrupt) as exc:
        print(f"interrupted; staging preserved, nothing published: {exc}",
              file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
