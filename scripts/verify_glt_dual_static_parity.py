#!/usr/bin/env python3
"""Bounded parity check for an independently rebuilt static/target artifact.

The command deliberately uses the fixed real sample keys from the first parity
report (20 PI1M and 12 downstream).  It writes a new temporary artifact and
compares it with both the published artifact and the online runtime path.  No
conformer generation, model execution, or write to a frozen cache is allowed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from src.dataset.cache_lifecycle import CacheLifecycleError, atomic_json, json_hash, zero_write_snapshot
from src.dataset.glt_dual import build_dual_sample
from src.dataset.glt_dual_cache import DualFrozenBundle, load_active_dual_store, load_dual_cohort, ordered_key_hash
from src.dataset.glt_dual_pretrain import prepare_pretrain_sample, _unpack_fingerprint
from src.dataset.glt_dual_static import (
    DualStaticCache, PretrainTargetsCache, build_pretrain_target, load_chunk_payload,
    write_chunk,
)
from scripts.build_glt_dual_static_cache import _build_one, _worker_init


CLASS_QUOTAS = (("ordinary", 12), ("no_center_angles", 8))
DOWNSTREAM_QUOTAS = (("geometry_fallback", 9), ("ordinary", 3))
COMPARE_ATOL = 1e-6
COMPARE_RTOL = 1e-6
DEFAULT_EXPECTED_KEY_JSON = (
    Path(__file__).resolve().parents[1] / "tests" / "fixtures" /
    "glt_dual_parity_expected_keys.json"
)


class MissingFixture(CacheLifecycleError):
    """The requested real category is absent from the selected artifact."""


class BudgetExceeded(CacheLifecycleError):
    """The bounded temporary output exceeded its explicit budget."""


class ExpectedKeyMismatch(CacheLifecycleError):
    """The fixed historical parity key list cannot be reproduced exactly."""


class ParityMismatch(CacheLifecycleError):
    def __init__(self, message, details):
        super().__init__(message)
        self.details = details


def _classify_reference(cache, index):
    record = cache.get(index)
    if not record.get("geometry_valid"):
        return "geometry_fallback"
    if len(record["angle_pairs"]) == 0:
        return "no_center_angles"
    return "ordinary"


def _key_list_digest(keys):
    return hashlib.sha256("\n".join(keys).encode("ascii")).hexdigest()


def _load_expected_keys(path):
    path = Path(path).resolve()
    if not path.is_file():
        raise ExpectedKeyMismatch(f"fixed parity key fixture is missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ExpectedKeyMismatch(f"fixed parity key fixture is unreadable: {path}") from exc
    expected = {}
    for route in ("pi1m", "downstream"):
        entry = payload.get(route)
        if not isinstance(entry, dict) or not isinstance(entry.get("keys"), list):
            raise ExpectedKeyMismatch(f"fixed parity key fixture lacks {route} keys: {path}")
        keys = [str(value) for value in entry["keys"]]
        if not keys or len(keys) != len(set(keys)):
            raise ExpectedKeyMismatch(f"fixed parity {route} key list is empty or duplicated")
        if any(len(value) != 64 or any(char not in "0123456789abcdef" for char in value)
               for value in keys):
            raise ExpectedKeyMismatch(f"fixed parity {route} key list contains invalid sample key")
        digest = entry.get("sha256")
        if digest != _key_list_digest(keys):
            raise ExpectedKeyMismatch(f"fixed parity {route} key digest does not match fixture")
        expected[route] = keys
    return expected


def _select_fixed(reference, expected_keys, quotas, limit):
    """Resolve exactly the historical keys; never replace them by quota scan."""

    if int(limit) != len(expected_keys):
        raise ExpectedKeyMismatch(
            f"requested limit {limit} differs from fixed {len(expected_keys)} keys"
        )
    quota_map = dict(quotas)
    counts = {name: 0 for name, _ in quotas}
    chosen = []
    for key_hex in expected_keys:
        try:
            index = reference.index_for_key(bytes.fromhex(key_hex))
        except (ValueError, CacheLifecycleError) as exc:
            raise ExpectedKeyMismatch(
                f"fixed parity key is absent from reference artifact: {key_hex}"
            ) from exc
        class_name = _classify_reference(reference, index)
        if class_name not in quota_map:
            raise ExpectedKeyMismatch(
                f"fixed parity key has unexpected class {class_name}: {key_hex}"
            )
        counts[class_name] += 1
        if counts[class_name] > quota_map[class_name]:
            raise ExpectedKeyMismatch(
                f"fixed parity {class_name} quota exceeded by key: {key_hex}"
            )
        chosen.append((class_name, index))
    missing = {name: quota - counts.get(name, 0)
               for name, quota in quotas if counts.get(name, 0) != quota}
    if missing:
        raise ExpectedKeyMismatch(f"fixed parity class counts differ from contract: {missing}")
    return chosen, counts


def _temporary_bytes(root):
    root = Path(root)
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def _check_budget(root, budget):
    used = _temporary_bytes(root)
    if used > int(budget):
        raise BudgetExceeded(f"temporary parity output exceeds budget: {used} > {budget}")
    return used


def _build_subset(rows, cache_root, bundle_hash, artifact_root, *, targets, chunk_size,
                  workers, size_budget):
    """Build and freeze one independent bounded artifact exactly once."""

    import multiprocessing as mp
    import os

    artifact_root = Path(artifact_root).resolve()
    staging = artifact_root.with_name(artifact_root.name + ".staging")
    if artifact_root.exists() or staging.exists():
        raise CacheLifecycleError(f"temporary candidate path already exists: {artifact_root}")
    staging.mkdir(parents=True)
    keys = [bytes.fromhex(str(row["sample_key"])) for row in rows]
    key_array = np.frombuffer(b"".join(keys), dtype=np.uint8).reshape(-1, 32)
    np.save(staging / "sample_keys.npy", key_array)
    items = []
    context = mp.get_context("fork")
    with context.Pool(processes=max(1, int(workers)), initializer=_worker_init,
                      initargs=(str(cache_root), bundle_hash)) as pool:
        for offset in range(0, len(rows), max(1, int(chunk_size))):
            _check_budget(staging.parent, size_budget)
            block = rows[offset:offset + max(1, int(chunk_size))]
            payloads = [
                (offset + i, str(row["sample_key"]), str(row["source_smiles"]),
                 str(row["normalized_smiles"]), bool(targets))
                for i, row in enumerate(block)
            ]
            built = list(pool.imap(_build_one, payloads, chunksize=1))
            built.sort(key=lambda value: value[0])
            expected_indices = list(range(offset, offset + len(block)))
            if [value[0] for value in built] != expected_indices:
                raise CacheLifecycleError("temporary parity worker result order mismatch")
            values = [value[3] if targets else value[2] for value in built]
            manifest = write_chunk(
                staging, offset, values, targets=targets,
                quarantine_root=staging / ".interrupted",
            )
            items.append({"start": offset, "count": len(block),
                          "target": bool(targets),
                          "path": f"chunks/chunk_{offset:08d}",
                          "arrays": manifest["arrays"]})
            _check_budget(staging.parent, size_budget)

    manifest = {
        "format": "glt-dual-pretrain-targets-v1" if targets else "glt-dual-static-v1",
        "sample_count": len(rows),
        "chunks": items,
        "ordered_sample_key_hash": ordered_key_hash(key_array),
        "parent_bundle_hash": bundle_hash,
        "cohort_manifest_hash": "temporary-parity-derived",
        "build_parameters": {"chunk_size": int(chunk_size), "targets": bool(targets)},
    }
    for item in items:
        load_chunk_payload(staging / item["path"], item, targets=targets)
    atomic_json(staging / "manifest.json", manifest)
    atomic_json(staging / ".frozen", {"manifest_hash": json_hash(manifest)})
    artifact_root.parent.mkdir(parents=True, exist_ok=True)
    os.replace(staging, artifact_root)
    return artifact_root, manifest, keys


def _as_array(value):
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _compare_values(left, right, path, differences, float_differences):
    """Compare nested model/cache values with exact integer semantics."""

    if (torch.is_tensor(left) or torch.is_tensor(right)
            or isinstance(left, np.ndarray) or isinstance(right, np.ndarray)):
        try:
            left_array, right_array = _as_array(left), _as_array(right)
        except Exception:
            differences.append({"path": path, "reason": "array conversion failed"})
            return
        if left_array.shape != right_array.shape:
            differences.append({"path": path, "reason": "shape mismatch",
                                "left": list(left_array.shape), "right": list(right_array.shape)})
            return
        left_float = np.issubdtype(left_array.dtype, np.floating)
        right_float = np.issubdtype(right_array.dtype, np.floating)
        if left_float or right_float:
            try:
                left_float_array = left_array.astype(np.float64, copy=False)
                right_float_array = right_array.astype(np.float64, copy=False)
            except (TypeError, ValueError):
                differences.append({"path": path, "reason": "numeric dtype mismatch"})
                return
            if (not np.isfinite(left_float_array).all()
                    or not np.isfinite(right_float_array).all()):
                differences.append({"path": path, "reason": "nonfinite"})
                return
            if not np.allclose(left_float_array, right_float_array,
                               atol=COMPARE_ATOL, rtol=COMPARE_RTOL):
                float_differences.append({
                    "path": path,
                    "max_abs": float(np.max(np.abs(left_float_array - right_float_array))),
                })
        elif not np.array_equal(left_array, right_array):
            differences.append({"path": path, "reason": "value mismatch"})
        return

    if isinstance(left, dict) or isinstance(right, dict):
        if not isinstance(left, dict) or not isinstance(right, dict):
            differences.append({"path": path, "reason": "mapping mismatch"})
            return
        for name in sorted(set(left) | set(right)):
            if name not in left or name not in right:
                differences.append({"path": f"{path}.{name}", "reason": "missing"})
            else:
                _compare_values(left[name], right[name], f"{path}.{name}",
                                differences, float_differences)
        return
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        if not isinstance(left, (list, tuple)) or not isinstance(right, (list, tuple)):
            differences.append({"path": path, "reason": "sequence mismatch"})
            return
        if len(left) != len(right):
            differences.append({"path": path, "reason": "length mismatch",
                                "left": len(left), "right": len(right)})
            return
        for index, (a, b) in enumerate(zip(left, right)):
            _compare_values(a, b, f"{path}[{index}]", differences, float_differences)
        return
    if isinstance(left, (str, bytes, bool, int)) or isinstance(right, (str, bytes, bool, int)):
        if left != right:
            differences.append({"path": path, "reason": "scalar mismatch",
                                "left": str(left), "right": str(right)})
        return
    if left != right:
        differences.append({"path": path, "reason": "scalar mismatch",
                            "left": str(left), "right": str(right)})


def _compare(left, right, label):
    differences, float_differences = [], []
    _compare_values(left, right, label, differences, float_differences)
    return differences, float_differences


def _record_fields(item):
    if hasattr(item, "keys"):
        return {name: getattr(item, name) for name in item.keys()}
    return dict(item)


def _check_static_and_runtime(reference, candidate, topology, trimer, row, key_hex,
                              *, seed, sigma, ratio, position, target_reference,
                              target_candidate, class_name):
    details = {"sample_key": key_hex, "class": class_name, "differences": [],
               "float_differences": [], "target_differences": [],
               "runtime_differences": [], "pretrain_differences": [],
               "center_angle_empty": None}
    diffs, floats = _compare(reference, candidate, "static")
    details["differences"].extend(diffs)
    details["float_differences"].extend(floats)
    if target_reference is not None:
        expected = build_pretrain_target(row["normalized_smiles"])
        target_diffs, target_floats = _compare(target_reference, target_candidate, "target")
        details["target_differences"].extend(target_diffs)
        details["target_differences"].extend(target_floats)
        expected_diffs, expected_floats = _compare(target_candidate, expected, "target_expected")
        details["target_differences"].extend(expected_diffs)
        details["target_differences"].extend(expected_floats)
        unpacked = _unpack_fingerprint(target_candidate["fingerprint_packed"])
        expected_bits = torch.from_numpy(
            np.unpackbits(expected["fingerprint_packed"], bitorder="little")[:2048].copy()
        ).float()
        bit_diffs, bit_floats = _compare(unpacked, expected_bits, "fingerprint_unpacked")
        details["target_differences"].extend(bit_diffs)
        details["target_differences"].extend(bit_floats)

    online = build_dual_sample(topology, trimer, row["source_smiles"])
    cached = build_dual_sample(topology, trimer, row["source_smiles"], static=candidate)
    runtime_diffs, runtime_floats = _compare(
        _record_fields(online), _record_fields(cached), "runtime_clean"
    )
    details["runtime_differences"].extend(runtime_diffs)
    details["runtime_differences"].extend(runtime_floats)
    if target_candidate is not None:
        old_input, old_labels = prepare_pretrain_sample(
            topology, trimer, row["source_smiles"], seed=seed, key=key_hex,
            position=position, sigma=sigma, ratio=ratio,
        )
        new_input, new_labels = prepare_pretrain_sample(
            topology, trimer, row["source_smiles"], seed=seed, key=key_hex,
            position=position, sigma=sigma, ratio=ratio, static=candidate,
            target=target_candidate,
        )
        pre_diffs, pre_floats = _compare(
            _record_fields(old_input), _record_fields(new_input), "pretrain_input"
        )
        label_diffs, label_floats = _compare(old_labels, new_labels, "pretrain_labels")
        details["pretrain_differences"].extend(pre_diffs)
        details["pretrain_differences"].extend(pre_floats)
        details["pretrain_differences"].extend(label_diffs)
        details["pretrain_differences"].extend(label_floats)
        if class_name == "no_center_angles":
            details["center_angle_empty"] = bool(
                old_labels["angle_pairs"].numel() == 0
                and old_labels["angle_cos"].numel() == 0
                and new_labels["angle_pairs"].numel() == 0
                and new_labels["angle_cos"].numel() == 0
            )
    return details


def _route(reference_root, reference_targets, cohort_root, cache_root, *, quotas, limit,
           targets, chunk_size, workers, temp_root, size_budget, expected_keys,
           seed=20260915, sigma=0.03, ratio=0.3):
    store = load_active_dual_store(cache_root)
    cohort = load_dual_cohort(cohort_root, cache_root)
    records = {str(row["sample_key"]): row for row in cohort["records"]}
    reference = DualStaticCache(
        reference_root,
        parent_bundle_hash=cohort["manifest"]["main_bundle_hash"],
        cohort_manifest_hash=cohort["manifest_hash"],
    )
    reference_targets_cache = None
    bundle = None
    candidate = candidate_target = None
    try:
        if targets:
            if not reference_targets:
                raise MissingFixture("PI1M reference target artifact is required")
            reference_targets_cache = PretrainTargetsCache(
                reference_targets,
                parent_bundle_hash=cohort["manifest"]["main_bundle_hash"],
                cohort_manifest_hash=cohort["manifest_hash"],
            )
        chosen, counts = _select_fixed(reference, expected_keys, quotas, limit)
        rows = []
        for _, index in chosen:
            key_hex = bytes(reference.sample_keys[index]).hex()
            if key_hex not in records:
                raise CacheLifecycleError(f"selected key is absent from cohort: {key_hex}")
            rows.append(records[key_hex])
        route_name = "pi1m" if targets else "downstream"
        candidate_root, _, _ = _build_subset(
            rows, cache_root, store["bundle_hash"], Path(temp_root) / f"{route_name}_static",
            targets=False, chunk_size=chunk_size, workers=workers, size_budget=size_budget,
        )
        candidate = DualStaticCache(candidate_root)
        target_candidate_root = None
        if targets:
            target_candidate_root, _, _ = _build_subset(
                rows, cache_root, store["bundle_hash"], Path(temp_root) / f"{route_name}_targets",
                targets=True, chunk_size=chunk_size, workers=workers, size_budget=size_budget,
            )
            candidate_target = PretrainTargetsCache(target_candidate_root)
        bundle = DualFrozenBundle(
            cache_root, expected_bundle_hash=cohort["manifest"]["main_bundle_hash"]
        )
        details, mismatches = [], []
        for class_name, index in chosen:
            key = bytes(reference.sample_keys[index])
            key_hex = key.hex()
            row = records[key_hex]
            reference_row = reference.get(index)
            candidate_row = candidate.get_by_key(key)
            target_reference_row = reference_targets_cache.get_by_key(key) if reference_targets_cache else None
            target_candidate_row = candidate_target.get_by_key(key) if candidate_target else None
            topology, trimer = bundle.topology[key], bundle.trimer[key]
            item = _check_static_and_runtime(
                reference_row, candidate_row, topology, trimer, row, key_hex,
                seed=seed, sigma=sigma, ratio=ratio, position=index,
                target_reference=target_reference_row, target_candidate=target_candidate_row,
                class_name=class_name,
            )
            details.append(item)
            if any(item[name] for name in ("differences", "target_differences",
                                            "runtime_differences", "pretrain_differences")):
                mismatches.append(item)
        if mismatches:
            raise ParityMismatch(f"{route_name} parity mismatch in {len(mismatches)} samples", details)
        return {
            "sample_count": len(chosen), "class_counts": counts,
            "keys": [bytes(reference.sample_keys[index]).hex() for _, index in chosen],
            "sample_details": details,
            "real_n_zero_proven": False,
            "n_zero_status": "not_proven_by_selected_real_keys",
            "reference_root": str(reference_root),
            "temporary_static_root": str(candidate_root),
            "temporary_target_root": str(target_candidate_root) if target_candidate_root else None,
        }
    finally:
        if candidate is not None:
            candidate.close()
        if candidate_target is not None:
            candidate_target.close()
        if bundle is not None:
            bundle.close()
        if reference_targets_cache is not None:
            reference_targets_cache.close()
        reference.close()


def _new_temp_root(value):
    if value:
        path = Path(value).resolve()
        if path.exists():
            raise CacheLifecycleError(f"--temp-root must name a new path: {path}")
        path.mkdir(parents=True)
        return path
    return Path(tempfile.mkdtemp(prefix="glt_dual_static_parity_"))


def _zero_write_ok(value):
    if value is True:
        return True
    return isinstance(value, dict) and bool(value) and all(item is True for item in value.values())


def _apply_success_gates(report):
    """Turn a nominal comparison PASS into failure when safety gates fail."""

    if report.get("status") != "PASS":
        return
    if not _zero_write_ok(report.get("frozen_cache_zero_write")):
        report.update({
            "status": "ACTIVE_CACHE_MODIFIED",
            "error_type": "ActiveCacheModified",
            "error": "active frozen cache changed during parity",
        })
        return
    if report.get("within_size_budget") is not True:
        report.update({
            "status": "BUDGET_EXCEEDED",
            "error_type": "BudgetExceeded",
            "error": "temporary parity output is missing or exceeds its budget",
        })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pi1m-cache-root", required=True)
    parser.add_argument("--pi1m-cohort-root", required=True)
    parser.add_argument("--pi1m-reference-static", required=True)
    parser.add_argument("--pi1m-reference-targets", required=True)
    parser.add_argument("--downstream-cache-root", required=True)
    parser.add_argument("--downstream-cohort-root", required=True)
    parser.add_argument("--downstream-reference-static", required=True)
    parser.add_argument("--report-json", required=True)
    parser.add_argument("--temp-root")
    parser.add_argument("--expected-key-json", default=str(DEFAULT_EXPECTED_KEY_JSON))
    parser.add_argument("--limit-pi1m", type=int, default=20)
    parser.add_argument("--limit-downstream", type=int, default=12)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--size-budget-bytes", type=int, default=1024 ** 3)
    parser.add_argument("--seed", type=int, default=20260915)
    parser.add_argument("--sigma", type=float, default=0.03)
    parser.add_argument("--ratio", type=float, default=0.3)
    args = parser.parse_args()
    report_path = Path(args.report_json).resolve()
    if report_path.exists():
        raise SystemExit(f"refusing to overwrite existing report: {report_path}")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "NOT_RUN", "scope": "fixed 20 PI1M + 12 downstream real keys",
        "report_json": str(report_path), "temporary_root": None,
        "size_budget_bytes": int(args.size_budget_bytes),
        "model_status": "NOT_RUN", "conformer_generation": "NOT_RUN",
        "seed": int(args.seed), "sigma": float(args.sigma), "mask_ratio": float(args.ratio),
        "expected_key_json": str(Path(args.expected_key_json).resolve()),
    }
    before = {}
    temp_root = None
    try:
        expected_keys = _load_expected_keys(args.expected_key_json)
        report["expected_key_digests"] = {
            route: _key_list_digest(keys) for route, keys in expected_keys.items()
        }
        temp_root = _new_temp_root(args.temp_root)
        report["temporary_root"] = str(temp_root)
        before = {
            "pi1m": zero_write_snapshot(Path(args.pi1m_cache_root)),
            "downstream": zero_write_snapshot(Path(args.downstream_cache_root)),
        }
        pi1m = _route(
            args.pi1m_reference_static, args.pi1m_reference_targets, args.pi1m_cohort_root,
            args.pi1m_cache_root, quotas=CLASS_QUOTAS, limit=args.limit_pi1m, targets=True,
            chunk_size=4096, workers=args.workers, temp_root=temp_root,
            expected_keys=expected_keys["pi1m"],
            size_budget=args.size_budget_bytes, seed=args.seed, sigma=args.sigma,
            ratio=args.ratio,
        )
        report["pi1m"] = pi1m
        if pi1m.get("keys") != expected_keys["pi1m"]:
            raise ExpectedKeyMismatch("selected PI1M parity keys differ from fixed fixture")
        downstream = _route(
            args.downstream_reference_static, None, args.downstream_cohort_root,
            args.downstream_cache_root, quotas=DOWNSTREAM_QUOTAS,
            limit=args.limit_downstream, targets=False, chunk_size=512,
            workers=args.workers, temp_root=temp_root,
            expected_keys=expected_keys["downstream"],
            size_budget=args.size_budget_bytes, seed=args.seed, sigma=args.sigma,
            ratio=args.ratio,
        )
        report["downstream"] = downstream
        if downstream.get("keys") != expected_keys["downstream"]:
            raise ExpectedKeyMismatch("selected downstream parity keys differ from fixed fixture")
        report["key_list_sha256"] = {
            "pi1m": hashlib.sha256(
                "\n".join(pi1m["keys"]).encode("ascii")
            ).hexdigest(),
            "downstream": hashlib.sha256(
                "\n".join(downstream["keys"]).encode("ascii")
            ).hexdigest(),
        }
        report["status"] = "PASS"
    except MissingFixture as exc:
        report.update({"status": "MISSING_FIXTURE", "error_type": type(exc).__name__, "error": str(exc)})
    except BudgetExceeded as exc:
        report.update({"status": "BUDGET_EXCEEDED", "error_type": type(exc).__name__, "error": str(exc)})
    except ExpectedKeyMismatch as exc:
        report.update({"status": "UNRESOLVED_PROVENANCE", "error_type": type(exc).__name__, "error": str(exc)})
    except ParityMismatch as exc:
        report.update({"status": "DATA_MISMATCH", "error_type": type(exc).__name__, "error": str(exc),
                       "parity_details": exc.details})
    except Exception as exc:
        report.update({"status": "SCRIPT_ERROR", "error_type": type(exc).__name__, "error": str(exc)})
    finally:
        if before:
            try:
                after = {
                    "pi1m": zero_write_snapshot(Path(args.pi1m_cache_root)),
                    "downstream": zero_write_snapshot(Path(args.downstream_cache_root)),
                }
                report["frozen_cache_zero_write"] = {
                    name: before[name] == after[name] for name in before
                }
            except Exception as exc:
                report["frozen_cache_zero_write"] = False
                report["zero_write_error"] = f"{type(exc).__name__}: {exc}"
        else:
            report["frozen_cache_zero_write"] = None
        if temp_root is not None:
            try:
                report["temporary_bytes"] = _temporary_bytes(temp_root)
                report["within_size_budget"] = report["temporary_bytes"] <= int(args.size_budget_bytes)
            except Exception as exc:
                report["temporary_bytes_error"] = f"{type(exc).__name__}: {exc}"
        _apply_success_gates(report)
        atomic_json(report_path, report)
    print(json.dumps({
        key: report.get(key) for key in (
            "status", "temporary_root", "temporary_bytes", "within_size_budget",
            "frozen_cache_zero_write", "error_type", "error",
        )
    }, indent=2, sort_keys=True))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
