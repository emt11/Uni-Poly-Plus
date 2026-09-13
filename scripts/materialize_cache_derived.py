#!/usr/bin/env python3
"""Offline materialization of derived cache arrays (explicit, read-only input).

Flow: frozen MD200 LMDB -> explicit offline command -> md200_<build_spec_hash>/
mmap arrays -> re-run scripts/create_cache_store.py to register them -> the
formal training reader mmap-reads them and never writes.

The training path refuses to run this logic: a missing derived array raises
CacheMissingDerivedArtifact instead of being generated on the fly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset.cache_spec import load_store  # noqa: E402
from src.dataset.lmdb_cache import build_or_load_cohort  # noqa: E402
from src.dataset.frozen_store import (  # noqa: E402
    ArtifactIdentityError,
    FrozenArtifact,
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def materialize_md200_for_cohort(cohort, md_artifact: FrozenArtifact,
                                 overwrite: bool = False) -> Path:
    """Write cohort_dir/md200_<build_spec_hash>/{md200.npy,md200_valid.npy}."""

    cohort_dir = Path(cohort["root"])
    count = len(cohort["keys_array"])
    output_dir = cohort_dir / f"md200_{md_artifact.build_spec_hash}"
    values_path = output_dir / "md200.npy"
    valid_path = output_dir / "md200_valid.npy"
    metadata_path = output_dir / "md200_metadata.json"
    if metadata_path.is_file() and values_path.is_file() \
            and valid_path.is_file() and not overwrite:
        print(f"[materialize] already present: {output_dir}")
        return output_dir

    output_dir.mkdir(parents=True, exist_ok=True)
    values_tmp = output_dir / "md200.npy.tmp"
    valid_tmp = output_dir / "md200_valid.npy.tmp"
    values = np.lib.format.open_memmap(
        values_tmp, mode="w+", dtype=np.float32, shape=(count, 200)
    )
    validity = np.lib.format.open_memmap(
        valid_tmp, mode="w+", dtype=np.bool_, shape=(count,)
    )
    for index, key in enumerate(cohort["keys_array"]):
        data = md_artifact.get(key)
        vector = getattr(data, "mips_md", None)
        if not torch.is_tensor(vector) or vector.numel() != 200:
            raise ArtifactIdentityError(
                f"invalid MD200 cache row for {bytes(key).hex()[:16]}…"
            )
        values[index] = vector.detach().cpu().float().numpy()
        validity[index] = bool(getattr(data, "mips_md_valid", False))
        if index % 10000 == 0 and index:
            print(f"[materialize] {index}/{count}", flush=True)
    values.flush()
    validity.flush()
    del values, validity
    values_path.unlink(missing_ok=True)
    valid_path.unlink(missing_ok=True)
    values_tmp.replace(values_path)
    valid_tmp.replace(valid_path)

    metadata = {
        "cohort_hash": str(cohort["manifest"]["cohort_hash"]),
        "ordered_sample_key_hash": cohort["manifest"]["ordered_sample_key_hash"],
        "build_spec_hash": md_artifact.build_spec_hash,
        "shape": [count, 200],
        "dtype": "float32",
        "values_sha256": _sha256_file(values_path),
        "valid_sha256": _sha256_file(valid_path),
        "created_at": time.time(),
    }
    metadata_tmp = metadata_path.with_suffix(".json.tmp")
    with open(metadata_tmp, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, sort_keys=True, indent=2)
        handle.write("\n")
    metadata_tmp.replace(metadata_path)
    print(f"[materialize] wrote {output_dir} ({count} rows)")
    return output_dir


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", default="data/processed/mips_trimer_scage")
    parser.add_argument(
        "--cohorts", default="",
        help="space/comma separated cohort names; default: every cohort with "
             "a current.json pointer",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    cache_root = Path(args.cache_root).resolve()
    store = load_store(cache_root)
    binding = store["artifacts"].get("md200")
    if binding is None:
        raise RuntimeError(
            "store.json has no active md200 artifact; register it first"
        )
    md_artifact = FrozenArtifact(cache_root, "md200", binding)

    cohorts_root = cache_root / "cohorts"
    if args.cohorts.strip():
        names = [
            item.strip() for item in args.cohorts.replace(",", " ").split()
            if item.strip()
        ]
    else:
        names = sorted(
            path.name for path in cohorts_root.iterdir()
            if (path / "current.json").is_file()
        ) if cohorts_root.is_dir() else []

    for name in names:
        cohort_dir = cohorts_root / name
        source_csv = ROOT / "data" / "raw" / (
            "smi_all.csv" if name == "downstream_union" else f"{name}.csv"
        )
        cohort = build_or_load_cohort(
            cache_root, name, str(source_csv),
            load_text=False, verify_integrity=False, allow_build=False,
        )
        materialize_md200_for_cohort(
            cohort, md_artifact, overwrite=args.overwrite
        )
    md_artifact.close()
    print(
        "done. Re-run scripts/create_cache_store.py to bind the new arrays."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
