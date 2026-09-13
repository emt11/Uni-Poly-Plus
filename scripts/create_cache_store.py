#!/usr/bin/env python3
"""Register route-compatible frozen MTS cache artifacts into store.json.

store.json is the single active-artifact entry point for formal training and
contains ONLY artifacts whose build_spec matches the current route.  An
artifact whose build_spec differs (e.g. a retired geometry protocol) is
reported and left in place on disk — never registered as active.

Registration reads each frozen artifact's ``metadata.json`` /
``manifest.json`` / ``.frozen`` (``.done`` is accepted as historical evidence
but is NOT required and NOT a reader condition), derives the factual
``build_spec`` (manual version counters are dropped; known historical label
values are translated to semantic parameters), computes the
``build_spec_hash`` and records the binding.

Lifecycle: building -> .frozen.  Nothing else.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
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
)


def _read_json(path: Path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _manifest_artifact_hash(manifest: dict) -> str:
    """sha256 of the canonical manifest JSON — identical to the historical
    `.done` payload, so old and new artifacts share one rule."""

    return hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


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
        frozen_path = artifact_dir / ".frozen"
        missing = [
            item.name for item in
            (metadata_path, manifest_path, frozen_path)
            if not item.is_file()
        ]
        if missing:
            continue  # incomplete layout: not registrable
        metadata = _read_json(metadata_path)
        manifest = _read_json(manifest_path)
        done_path = artifact_dir / ".done"
        done_note = None
        if done_path.is_file():
            observed = done_path.read_text(encoding="utf-8").strip()
            expected = _manifest_artifact_hash(manifest)
            if observed != expected:
                done_note = "historical .done disagrees with manifest digest"
        try:
            frozen_payload = _read_json(frozen_path)
        except Exception:
            frozen_payload = {}
        found.append({
            "dir": artifact_dir,
            "metadata": metadata,
            "manifest": manifest,
            "artifact_hash": _manifest_artifact_hash(manifest),
            "done_note": done_note,
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


def collect_active_bindings(cache_root: Path, selections: dict) -> tuple[dict, list[str]]:
    """Return ({layer: active binding}, report lines).

    Only artifacts whose derived build_spec_hash equals the current route
    hash become active.  Everything else is reported and skipped.
    """

    registered: dict[str, dict] = {}
    report: list[str] = ["# cache store registration report", ""]
    for layer in ARTIFACT_TYPES:
        candidates = scan_artifacts(cache_root, layer)
        if not candidates:
            report.append(f"## {layer}: no frozen artifact (active: absent)")
            continue
        derived = []
        for item in candidates:
            spec = build_spec_from_metadata(layer, item["metadata"])
            for parent_name in list(spec.get("parents", {})):
                parent_binding = registered.get(parent_name)
                if parent_binding is not None:
                    spec["parents"][parent_name] = parent_binding[
                        "build_spec_hash"
                    ]
                else:
                    # Parent is not active: the child cannot be either.
                    spec = None
                    break
            if spec is not None:
                derived.append((item, spec, build_spec_hash(spec)))
        route_hash = ROUTE_BUILD_SPEC_HASHES[layer]
        matching = [item for item in derived if item[2] == route_hash]
        if len(matching) > 1 and layer not in selections:
            raise StoreError(
                f"{layer}: {len(matching)} route-compatible artifacts; pass "
                f"--select {layer}=<hash-prefix>"
            )
        if layer in selections:
            prefix = selections[layer]
            matching = [item for item in matching if item[2].startswith(prefix)]
            if len(matching) != 1:
                raise StoreError(
                    f"{layer}: --select {prefix!r} matched {len(matching)} "
                    "route-compatible artifacts"
                )
        skipped = len(candidates) - len(matching)
        if not matching:
            report.append(
                f"## {layer}: ACTIVE ARTIFACT ABSENT — "
                f"{len(candidates)} frozen artifact(s) found, none matches "
                "the current route build_spec (left in place, not registered)"
            )
            continue
        chosen_item, chosen_spec, spec_hash = matching[0]
        binding = binding_for(
            layer,
            path=str(chosen_item["dir"].relative_to(cache_root)),
            build_spec=chosen_spec,
            artifact_hash=chosen_item["artifact_hash"],
            record_count=int(chosen_item["manifest"].get("count", -1)),
        )
        binding["rdkit_version"] = str(
            chosen_item["metadata"].get("rdkit_version", "")
        )
        registered[layer] = binding
        report.append(
            f"## {layer}: {spec_hash[:16]}…  ACTIVE (matches route)\n"
            f"- path: {binding['path']}\n"
            f"- records: {binding['record_count']}\n"
            f"- artifact_hash: {binding['artifact_hash'][:16]}…"
            + (f"\n- note: {chosen_item['done_note']}"
               if chosen_item["done_note"] else "")
            + (f"\n- skipped {skipped} non-route-compatible artifact(s)"
               if skipped else "")
        )
    if "md200" in registered:
        materialized = scan_materialized_md200(cache_root, cache_root / "cohorts")
        if materialized:
            registered["md200"]["materialized"] = materialized
            report.append(f"- materialized arrays: {sorted(materialized)}")
        else:
            report.append(
                "- materialized arrays: none (run "
                "scripts/materialize_cache_derived.py; the training reader "
                "will refuse md200 until then)"
            )
    return registered, report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", default="data/processed/mips_trimer_scage")
    parser.add_argument(
        "--select", action="append", default=[],
        help="layer=hash-prefix when several route-compatible artifacts exist",
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
        print(f"store OK: {sorted(store['artifacts'])}")
        for layer in ARTIFACT_TYPES:
            if layer in store["artifacts"] and store["artifacts"][layer][
                    "build_spec_hash"] != ROUTE_BUILD_SPEC_HASHES[layer]:
                print(f"WARNING: {layer} binding deviates from the route")
        return 0

    registered, report = collect_active_bindings(cache_root, selections)
    store = {"artifacts": registered, "created_at": time.time()}
    print("\n".join(report))
    save_store(cache_root, store)
    print(f"store written: {cache_root / 'store.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
