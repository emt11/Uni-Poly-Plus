#!/usr/bin/env python3
"""Validate and freeze the corrected explicit k-RU topology artifact.

This finalizer never writes canonical RU/Topology/Trimer/MD200 roots.  It
accepts the explicit comparison only after both PI1M_v2 and downstream
cohorts have been fully validated against the frozen canonical Trimer layer.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
# Keep this finalizer runnable as a direct script without relying on the
# caller to export PYTHONPATH.  All production wrappers already execute from
# ROOT, but the freeze contract must also be reproducible from an isolated
# CLI invocation.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
EXPECTED_UNION_COUNT = 999_224
EXPECTED_COHORTS = {
    "0098e20479194659840d5f093de7879180572f65d0020155ab7c91212ae09049": 995_799,
    "ede5f8bc4ba277708c57787249ec85733b1a30a9c149ed9703ec55c80a843bb2": 3_655,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _explicit_specs():
    from src.dataset.dataset import UniDataset
    from src.dataset.mips_trimer_contract import TOPOLOGY_EXPLICIT

    dataset = UniDataset.__new__(UniDataset)
    dataset.root = str(ROOT / "data")
    dataset.cache_layers = ("ru_base", "topology", "trimer", "md200")
    dataset.topology_representation = TOPOLOGY_EXPLICIT
    return dataset._lmdb_cache_specs({})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="results/mts_explicit_k_ru/final_acceptance.json",
    )
    args = parser.parse_args(argv)
    specs = _explicit_specs()
    topology_root = Path(specs["topology"]["root"])
    trimer_root = Path(specs["trimer"]["root"])
    lifecycle = topology_root.parents[1] / ".lifecycle.lock"
    lifecycle.parent.mkdir(parents=True, exist_ok=True)
    lock = lifecycle.open("a+", encoding="utf-8")
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except (BlockingIOError, OSError) as exc:
        lock.close()
        raise SystemExit("an LMDB writer is active; explicit freeze refused") from exc

    try:
        required = [
            topology_root / "metadata.json",
            topology_root / "manifest.json",
            topology_root / ".done",
            trimer_root / ".done",
            trimer_root / ".frozen",
        ]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise RuntimeError("missing required artifact: " + ", ".join(missing))
        metadata = json.loads((topology_root / "metadata.json").read_text())
        manifest = json.loads((topology_root / "manifest.json").read_text())
        if metadata != specs["topology"]["meta"]:
            raise RuntimeError("explicit topology metadata/config hash mismatch")
        if int(manifest.get("count", -1)) != EXPECTED_UNION_COUNT:
            raise RuntimeError(
                f"explicit exact-union count mismatch: {manifest.get('count')}"
            )
        observed_cohorts = set(manifest.get("cohort_hashes", []))
        if not set(EXPECTED_COHORTS) <= observed_cohorts:
            raise RuntimeError("explicit manifest is missing a required cohort")

        validations = {}
        for cohort_hash, expected_count in EXPECTED_COHORTS.items():
            path = topology_root / "validation" / "cohorts" / f"{cohort_hash}.json"
            if not path.is_file():
                raise RuntimeError(f"missing full validation report: {path}")
            report = json.loads(path.read_text(encoding="utf-8"))
            if int(report.get("record_count", -1)) != expected_count:
                raise RuntimeError(f"validation count mismatch for {cohort_hash}")
            if int(report.get("mapping_failure", 1)) != 0:
                raise RuntimeError(f"mapping failure for {cohort_hash}")
            if int(report.get("two_d_mcl", 1)) != 0:
                raise RuntimeError(f"2D geometry entered MCL for {cohort_hash}")
            rate = report.get("geometry_rate_given_graph")
            if rate is None or float(rate) < 0.90:
                raise RuntimeError(f"MCL coverage below 90% for {cohort_hash}")
            validations[cohort_hash] = {
                "path": str(path.relative_to(ROOT)),
                "sha256": _sha256(path),
                "record_count": expected_count,
                "geometry_rate_given_graph": float(rate),
            }

        payload = {
            "schema": "mts-explicit-kru-topology-frozen-v1",
            "topology_representation": "explicit_k_ru",
            "record_count": EXPECTED_UNION_COUNT,
            "topology_feature_config_hash": metadata["feature_config_hash"],
            "topology_done_artifact_id": (
                topology_root / ".done"
            ).read_text(encoding="utf-8").strip(),
            "topology_done_sha256": _sha256(topology_root / ".done"),
            "topology_metadata_sha256": _sha256(topology_root / "metadata.json"),
            "topology_manifest_sha256": _sha256(topology_root / "manifest.json"),
            "canonical_trimer_done_artifact_id": (
                trimer_root / ".done"
            ).read_text(encoding="utf-8").strip(),
            "canonical_trimer_done_sha256": _sha256(trimer_root / ".done"),
            "canonical_trimer_frozen_sha256": _sha256(trimer_root / ".frozen"),
            "cohort_validations": validations,
        }
        payload["frozen_payload_sha256"] = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        frozen = topology_root / ".frozen"
        if frozen.exists():
            observed = json.loads(frozen.read_text(encoding="utf-8"))
            if observed != payload:
                raise RuntimeError("existing explicit .frozen payload differs")
        else:
            _atomic_json(frozen, payload)
        output = ROOT / args.output
        output.parent.mkdir(parents=True, exist_ok=True)
        _atomic_json(output, {**payload, "passed": True})
        print(json.dumps({**payload, "passed": True}, sort_keys=True, indent=2))
    finally:
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        lock.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
