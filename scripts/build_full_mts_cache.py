#!/usr/bin/env python3
"""Production entry point for the full PI1M-scale MTS cache rebuild.

Thin wrapper around the pilot-validated builder (scripts/build_mts_cache.py):
production defaults, a preflight gate, and nothing else.  The scientific
build specs, staging/audit/freeze/publish logic and store.json handling are
the already-verified pilot code paths.

    # preflight only (no build):
    python scripts/build_full_mts_cache.py --preflight \
        --cache-root data/processed/mips_trimer_scage \
        --source-csv data/raw/PI1M_v2.csv --workers 32

    # full rebuild (explicitly launched by the user):
    python scripts/build_full_mts_cache.py \
        --cache-root data/processed/mips_trimer_scage \
        --source-csv data/raw/PI1M_v2.csv --limit 0 --workers 32
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

EXPECTED_DISK_GIB = 150.0  # >= 2x the ~74 GiB full-scale estimate


def _source_csv_sha256(path: Path) -> str:
    import hashlib
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def preflight(args) -> tuple[bool, dict]:
    checks = {}
    ok = True

    source_csv = Path(args.source_csv).resolve()
    checks["source_csv_exists"] = source_csv.is_file()
    ok &= checks["source_csv_exists"]
    if checks["source_csv_exists"]:
        checks["source_csv_sha256"] = _source_csv_sha256(source_csv)
        checks["source_csv_bytes"] = source_csv.stat().st_size

    # 2) build_spec hashes known and identical to the validated pilot bundle
    try:
        from src.dataset.cache_spec import ROUTE_BUILD_SPECS, build_spec_hash
        hashes = {
            layer: build_spec_hash(ROUTE_BUILD_SPECS[layer])
            for layer in ("ru_base", "topology", "trimer")
        }
        checks["build_spec_hashes"] = hashes
        pilot_root = (
            ROOT / "data/processed/mts_cache_pilot_20260913/builds/"
            "7339faaf6401fe58176dbfe5a9e3692eced0a5aec2751fc1d979450d2f34074e"
        )
        if pilot_root.is_dir():
            pilot_match = all(
                json.loads((pilot_root / layer / "metadata.json")
                           .read_text(encoding="utf-8"))["build_spec_hash"]
                == hashes[layer]
                for layer in hashes
            )
            checks["build_specs_match_validated_pilot"] = pilot_match
            ok &= pilot_match
    except Exception as exc:  # noqa: BLE001
        checks["build_spec_hashes"] = f"error: {exc}"
        ok = False

    # 3) output filesystem writable
    cache_root = Path(args.cache_root).resolve()
    try:
        cache_root.mkdir(parents=True, exist_ok=True)
        probe = cache_root / ".preflight_write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        checks["output_writable"] = True
    except Exception as exc:  # noqa: BLE001
        checks["output_writable"] = f"error: {exc}"
        ok = False

    # 4) disk space (>= 2x expected artifact size)
    usage = shutil.disk_usage(cache_root)
    free_gib = usage.free / 2 ** 30
    checks["disk_free_gib"] = round(free_gib, 1)
    checks["disk_required_gib"] = EXPECTED_DISK_GIB
    checks["disk_sufficient"] = free_gib >= EXPECTED_DISK_GIB
    ok &= checks["disk_sufficient"]

    # 5) source identity for the target bundle (bundle hash + staging state)
    try:
        from src.dataset.cache_lifecycle import (
            artifact_identity, bundle_identity, load_source_rows,
        )
        rows, manifest = load_source_rows(source_csv, int(args.limit))
        source_hash = manifest["source_manifest_hash"]
        checks["source_count"] = manifest["source_count"]
        checks["selection"] = manifest["selection"]
        artifact_hashes = {
            "ru_base": artifact_identity("ru_base", source_hash, {}),
        }
        artifact_hashes["topology"] = artifact_identity(
            "topology", source_hash, {"ru_base": artifact_hashes["ru_base"]}
        )
        artifact_hashes["trimer"] = artifact_identity(
            "trimer", source_hash, {"ru_base": artifact_hashes["ru_base"]}
        )
        bundle_hash = bundle_identity(source_hash, artifact_hashes)
        checks["bundle_hash"] = bundle_hash
        checks["ordered_source_key_hash"] = manifest["ordered_source_key_hash"]

        builds_root = cache_root / "builds"
        staging = builds_root / f"{bundle_hash}.staging"
        final = builds_root / bundle_hash
        checks["staging_exists"] = staging.is_dir()
        checks["final_exists"] = final.is_dir()
        lock_conflict = False
        if staging.is_dir():
            import fcntl
            for layer in ("ru_base", "topology", "trimer"):
                lock_path = staging / layer / ".writer.lock"
                if not lock_path.is_file():
                    continue
                try:
                    handle = lock_path.open("a+")
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                    handle.close()
                except OSError:
                    lock_conflict = True
        checks["staging_writer_lock_conflict"] = lock_conflict
        ok &= not lock_conflict
        if final.is_dir():
            store_path = cache_root / "store.json"
            active = (
                store_path.is_file()
                and json.loads(store_path.read_text(encoding="utf-8"))
                .get("bundle_hash") == bundle_hash
            )
            checks["already_published_and_active"] = active
        # §23: unrelated historical staging dirs are listed and ignored
        ignored = []
        if builds_root.is_dir():
            for entry in builds_root.iterdir():
                if entry.name not in {f"{bundle_hash}.staging", bundle_hash}:
                    ignored.append(entry.name)
        checks["ignored_historical_build_dirs"] = sorted(ignored)

        # legacy store.json note
        store_path = cache_root / "store.json"
        if store_path.is_file():
            legacy = json.loads(store_path.read_text(encoding="utf-8"))
            checks["existing_store_json_format"] = (
                "lifecycle" if "bundle_hash" in legacy else "legacy"
            )
            checks["publish_will_supersede_existing_store"] = True
    except Exception as exc:  # noqa: BLE001
        checks["source_manifest"] = f"error: {exc}"
        ok = False

    # 8) worker count sanity
    cpu_count = os.cpu_count() or 1
    checks["cpu_count"] = cpu_count
    checks["workers_valid"] = 1 <= int(args.workers) <= max(cpu_count, 1) + 8
    ok &= checks["workers_valid"]

    # 9) required imports
    try:
        import lmdb  # noqa: F401
        import numpy  # noqa: F401
        import torch  # noqa: F401
        from rdkit import Chem  # noqa: F401
        checks["imports"] = "ok"
    except Exception as exc:  # noqa: BLE001
        checks["imports"] = f"error: {exc}"
        ok = False

    return bool(ok), checks


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path,
                        default=Path("data/processed/mips_trimer_scage"))
    parser.add_argument("--source-csv", type=Path,
                        default=Path("data/raw/PI1M_v2.csv"))
    parser.add_argument("--limit", type=int, default=0,
                        help="0 = full production source manifest")
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--interrupt-after", type=int, default=0)
    parser.add_argument("--report-json", type=Path)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args(argv)

    if args.preflight:
        passed, checks = preflight(args)
        print(json.dumps({"preflight_pass": passed, "checks": checks},
                         indent=2, sort_keys=True, default=str))
        return 0 if passed else 2

    from scripts.build_mts_cache import main as build_main
    return build_main([
        "--cache-root", str(args.cache_root),
        "--source-csv", str(args.source_csv),
        "--limit", str(args.limit),
        "--workers", str(args.workers),
        "--interrupt-after", str(args.interrupt_after),
        "--report-json", str(args.report_json or (
            args.cache_root / "production_build_report.json"
        )),
    ])


if __name__ == "__main__":
    raise SystemExit(main())
