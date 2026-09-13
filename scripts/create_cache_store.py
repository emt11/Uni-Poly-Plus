#!/usr/bin/env python3
"""Register frozen MTS cache artifacts into store.json (offline, read-only).

This is the single active-artifact entry point for formal training.  The
script never writes to an artifact directory: it reads each frozen artifact's
``metadata.json``/``manifest.json``/``.done``/``.frozen``, derives its factual
``build_spec`` (manual version counters are dropped, parent references are
re-bound to the registered parent's build_spec_hash), computes the
``build_spec_hash`` and records the binding in ``store.json``.

Selection rule: exactly one frozen artifact per layer may be registered
unless ``--select layer=hash-prefix`` disambiguates.  The report states
whether each binding matches the current route build_spec; a mismatch blocks
formal training through the frozen reader until the artifact is rebuilt or
the route genuinely changes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset.cache_spec import (  # noqa: E402
    ARTIFACT_TYPES,
    ROUTE_BUILD_SPEC_HASHES,
    StoreError,
    binding_for,
    build_spec_from_metadata,
    build_spec_hash,
    load_store,
    save_store,
    validate_store,
)


def _read_json(path: Path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def scan_artifacts(cache_root: Path, layer: str) -> list[dict]:
    found = []
    layer_root = cache_root / layer
    if not layer_root.is_dir():
        return found
    for artifact_dir in sorted(layer_root.iterdir()):
        if not artifact_dir.is_dir():
            continue
        metadata_path = artifact_dir / "metadata.json"
        manifest_path = artifact_dir / "manifest.json"
        done_path = artifact_dir / ".done"
        frozen_path = artifact_dir / ".frozen"
        missing = [
            item.name for item in
            (metadata_path, manifest_path, done_path, frozen_path)
            if not item.is_file()
        ]
        if missing:
            continue  # incomplete or legacy layout: not registrable
        metadata = _read_json(metadata_path)
        manifest = _read_json(manifest_path)
        try:
            frozen_payload = _read_json(frozen_path)
        except Exception:
            frozen_payload = {}
        found.append({
            "dir": artifact_dir,
            "metadata": metadata,
            "manifest": manifest,
            "done_artifact_id": done_path.read_text(encoding="utf-8").strip(),
            "frozen_payload": frozen_payload,
        })
    return found


def scan_materialized_md200(cache_root: Path, cohorts_root: Path) -> dict:
    """Map cohort name -> materialized MD200 array dir (relative path)."""

    materialized = {}
    if not cohorts_root.is_dir():
        return materialized
    for cohort_dir in sorted(path for path in cohorts_root.iterdir() if path.is_dir()):
        pointer = cohort_dir / "current.json"
        if not pointer.is_file():
            continue
        pointer_value = _read_json(pointer)
        cohort_hash = str(pointer_value.get("cohort_hash", ""))
        cohort_payload = cohort_dir / cohort_hash
        manifest_path = cohort_payload / "manifest.json"
        if not manifest_path.is_file():
            continue
        manifest = _read_json(manifest_path)
        for metadata_path in sorted(cohort_payload.glob("md200_*/md200_metadata.json")):
            metadata = _read_json(metadata_path)
            if (
                metadata.get("cohort_hash") == cohort_hash
                and metadata.get("ordered_sample_key_hash")
                == manifest.get("ordered_sample_key_hash")
                and metadata.get("dtype") == "float32"
                and (metadata_path.parent / "md200.npy").is_file()
                and (metadata_path.parent / "md200_valid.npy").is_file()
            ):
                materialized[str(cohort_dir.name)] = str(
                    metadata_path.parent.relative_to(cache_root)
                )
                break
    return materialized


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", default="data/processed/mips_trimer_scage")
    parser.add_argument(
        "--select", action="append", default=[],
        help="layer=hash-prefix when several frozen artifacts exist",
    )
    parser.add_argument(
        "--check", action="store_true",
        help="verify an existing store.json without writing",
    )
    args = parser.parse_args(argv)

    cache_root = Path(args.cache_root).resolve()
    selections = {}
    for item in args.select:
        if "=" not in item:
            parser.error(f"--select expects layer=hash-prefix, got {item!r}")
        name, _, prefix = item.partition("=")
        selections[name] = prefix

    if args.check:
        store = load_store(cache_root)
    else:
        registered = {}
        report_lines = ["# cache store registration report", ""]
        for layer in ARTIFACT_TYPES:
            candidates = scan_artifacts(cache_root, layer)
            if not candidates:
                report_lines.append(f"## {layer}: NO frozen artifact found")
                continue
            if len(candidates) > 1 and layer not in selections:
                raise StoreError(
                    f"{layer}: {len(candidates)} frozen artifacts; pass "
                    f"--select {layer}=<hash-prefix>"
                )
            if layer in selections:
                prefix = selections[layer]
                candidates = [
                    item for item in candidates
                    if build_spec_hash(
                        build_spec_from_metadata(layer, item["metadata"])
                    ).startswith(prefix)
                ]
                if len(candidates) != 1:
                    raise StoreError(
                        f"{layer}: --select {prefix!r} matched "
                        f"{len(candidates)} artifacts"
                    )
            chosen = candidates[0]
            build_spec = build_spec_from_metadata(layer, chosen["metadata"])
            # Re-bind parent references to the registered parent artifacts'
            # build_spec_hash so the identity chain is self-consistent.
            for parent_name in list(build_spec.get("parents", {})):
                parent_binding = registered.get(parent_name)
                if parent_binding is None:
                    raise StoreError(
                        f"{layer} references parent {parent_name} which is "
                        "not registered yet"
                    )
                build_spec["parents"][parent_name] = parent_binding[
                    "build_spec_hash"
                ]
            spec_hash = build_spec_hash(build_spec)
            manifest = chosen["manifest"]
            binding = binding_for(
                layer,
                path=str(chosen["dir"].relative_to(cache_root)),
                build_spec=build_spec,
                artifact_hash=chosen["done_artifact_id"],
                record_count=int(manifest.get("count", -1)),
            )
            binding["rdkit_version"] = str(
                chosen["metadata"].get("rdkit_version", "")
            )
            registered[layer] = binding
            route_hash = ROUTE_BUILD_SPEC_HASHES.get(layer)
            status = (
                "MATCHES ROUTE" if spec_hash == route_hash
                else "ROUTE MISMATCH (blocks formal training for this layer)"
            )
            report_lines.append(
                f"## {layer}: {spec_hash[:16]}…  {status}\n"
                f"- path: {binding['path']}\n"
                f"- records: {binding['record_count']}\n"
                f"- artifact_hash: {binding['artifact_hash'][:16]}…"
            )

        if "md200" in registered:
            materialized = scan_materialized_md200(
                cache_root, cache_root / "cohorts"
            )
            if materialized:
                registered["md200"]["materialized"] = materialized
                report_lines.append(
                    f"- materialized arrays: {sorted(materialized)}"
                )

        store = {
            "artifacts": registered,
            "created_at": _now(),
        }
        report = "\n".join(report_lines)

    if not args.check:
        print(report)
        save_store(cache_root, store)
        print(f"store written: {cache_root / 'store.json'}")
    else:
        mismatches = []
        for layer, expected in ROUTE_BUILD_SPEC_HASHES.items():
            binding = store["artifacts"].get(layer)
            if binding and binding["build_spec_hash"] != expected:
                mismatches.append(layer)
        print(f"store OK: {sorted(store['artifacts'])}")
        print(
            "route mismatches: "
            + (", ".join(mismatches) if mismatches else "none")
        )
    return 0


def _now():
    import time
    return time.time()


if __name__ == "__main__":
    raise SystemExit(main())
