#!/usr/bin/env python
"""Read-only audit for the frozen MTS LMDB feature cache.

The audit never opens an LMDB writer and never computes a feature.  It reads
the ordered keys from a cohort manifest and deserializes each topology and
Trimer record once, producing a compact JSON report suitable for the 10k gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import torch

# The audit performs one sequential record read at a time.  Limit PyTorch's
# thread pools so 10k/PI1M scans do not create hundreds of idle worker
# threads or retain large temporary allocator arenas.
torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset.dataset import UniDataset  # noqa: E402
from src.dataset.lmdb_cache import (  # noqa: E402
    LmdbLayerStore,
    build_or_load_cohort,
)
from src.dataset.mips_cache_validation import validate_mcl_record  # noqa: E402
from src.dataset.mips_trimer_contract import MIGRATION_SCHEMA  # noqa: E402


def _specs(project_root: Path):
    dataset = object.__new__(UniDataset)
    dataset.root = str(project_root / "data")
    dataset.mips_max_hops = 2
    dataset.trimer_num_candidates = 4
    dataset.trimer_max_heavy_atoms = 384
    dataset.feature_cache_item_timeout = 240
    # Resolve all immutable layers so finalize can bind/freeze the downstream
    # MD200 artifact as well.  The 10k audit itself still reads only topology
    # and Trimer records.
    dataset.cache_layers = ("ru_base", "topology", "trimer", "md200")
    return dataset._lmdb_cache_specs({})


def _size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-csv", default="data/raw/PI1M_preflight10k.csv")
    parser.add_argument("--dataset-name", default="PI1M_preflight10k")
    parser.add_argument(
        "--output",
        default="results/mts/cache_audit_10k.json",
    )
    args = parser.parse_args()

    source_csv = (PROJECT_ROOT / args.source_csv).resolve()
    cohort = build_or_load_cohort(
        PROJECT_ROOT / "data/processed/mips_trimer_scage",
        args.dataset_name,
        source_csv,
        load_text=False,
        verify_integrity=True,
    )
    specs = _specs(PROJECT_ROOT)
    stores = {}
    try:
        for name in ("topology", "trimer"):
            stores[name] = LmdbLayerStore(
                specs[name]["root"], expected_meta=specs[name]["meta"]
            )

        started = time.monotonic()
        graph_valid = graph_unavailable = 0
        trimer_valid = trimer_unavailable = 0
        mcl_valid = 0
        star_valid = 0
        mapping_failure = 0
        finite_failure = 0
        two_d_mcl = 0
        failures = Counter()
        for key in cohort["keys"]:
            topology = stores["topology"][key]
            trimer = stores["trimer"][key]
            quality = validate_mcl_record(topology, trimer)
            available = quality["graph_available"]
            graph_valid += int(available)
            graph_unavailable += int(not available)
            if not available:
                failures[
                    f"graph:{getattr(topology, 'topology_failure_code', 'unknown')}"
                ] += 1
            valid = quality["geometry_valid"]
            trimer_valid += int(valid)
            trimer_unavailable += int(not valid)
            if not valid:
                failures[
                    f"trimer:{getattr(trimer, 'trimer_failure_code', 'unknown')}"
                ] += 1
            two_d_mcl += int(quality["two_d_mcl"])
            mapping_failure += int(quality["mapping_failure"])
            finite_failure += int(quality["finite_coordinate_failure"])
            mcl_valid += int(quality["mcl_valid"])
            star_valid += int(quality["star_3d_valid"])

        elapsed = max(time.monotonic() - started, 1e-9)
        manifest = cohort["manifest"]
        topology_done = Path(specs["topology"]["root"]) / ".done"
        trimer_done = Path(specs["trimer"]["root"]) / ".done"
        report = {
            "schema": "mts-canonical-cache-audit-v2",
            "migration_schema": MIGRATION_SCHEMA,
            "cache_layout_schema": stores["trimer"].meta.get(
                "cache_layout_schema"
            ),
            "trimer_schema": stores["trimer"].meta.get("schema"),
            "trimer_feature_config_hash": stores["trimer"].meta.get(
                "feature_config_hash"
            ),
            "done_artifact_id": {
                "topology": topology_done.read_text(encoding="utf-8").strip(),
                "trimer": trimer_done.read_text(encoding="utf-8").strip(),
            },
            "done_file_sha256": {
                "topology": hashlib.sha256(topology_done.read_bytes()).hexdigest(),
                "trimer": hashlib.sha256(trimer_done.read_bytes()).hexdigest(),
            },
            "metadata_file_sha256": {
                "topology": hashlib.sha256((Path(specs["topology"]["root"]) / "metadata.json").read_bytes()).hexdigest(),
                "trimer": hashlib.sha256((Path(specs["trimer"]["root"]) / "metadata.json").read_bytes()).hexdigest(),
            },
            "lmdb_manifest_sha256": {
                "topology": hashlib.sha256((Path(specs["topology"]["root"]) / "manifest.json").read_bytes()).hexdigest(),
                "trimer": hashlib.sha256((Path(specs["trimer"]["root"]) / "manifest.json").read_bytes()).hexdigest(),
            },
            "cohort_hash": manifest["cohort_hash"],
            "ordered_sample_key_hash": manifest["ordered_sample_key_hash"],
            "record_count": len(cohort["keys"]),
            "graph_valid": graph_valid,
            "graph_unavailable": graph_unavailable,
            "trimer_geometry_valid": trimer_valid,
            "trimer_geometry_unavailable": trimer_unavailable,
            "mcl_valid": mcl_valid,
            "geometry_rate_given_graph": (
                mcl_valid / graph_valid if graph_valid else None
            ),
            "star_3d_valid": star_valid,
            "mapping_failure": mapping_failure,
            "finite_coordinate_failure": finite_failure,
            "two_d_mcl": two_d_mcl,
            "failure_counts": dict(failures),
            "elapsed_seconds": elapsed,
            "read_samples_per_second": len(cohort["keys"]) / elapsed,
            "topology_bytes": _size_bytes(Path(specs["topology"]["root"])),
            "trimer_bytes": _size_bytes(Path(specs["trimer"]["root"])),
            "topology_bytes_per_sample": _size_bytes(
                Path(specs["topology"]["root"])
            ) / max(len(stores["topology"]), 1),
            "trimer_bytes_per_sample": _size_bytes(
                Path(specs["trimer"]["root"])
            ) / max(len(stores["trimer"]), 1),
            "validation_timestamp": time.time(),
        }
        # Finite-coordinate failures are diagnostics contributing to the MCL
        # coverage rate; they are not a separate cache gate.  Mapping errors
        # and accidental 2-D input remain hard invariants.
        if mapping_failure or two_d_mcl:
            raise RuntimeError(
                f"audit invariant failure: mapping={mapping_failure}, "
                f"two_d_mcl={two_d_mcl}"
            )
        output = (PROJECT_ROOT / args.output).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, output)
        print(json.dumps(report, indent=2, sort_keys=True))
    finally:
        for store in stores.values():
            store.close()


if __name__ == "__main__":
    main()
