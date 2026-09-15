#!/usr/bin/env python3
"""Freeze the 8-task row provenance and normalized unique structure union."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from scripts.create_mips_split_manifests import TASKS, build_manifest
from src.dataset.cache_lifecycle import CacheLifecycleError, atomic_json, json_hash
from src.dataset.glt_dual_cache import ordered_key_hash, sha256_file
from src.dataset.lmdb_cache import normalize_polymer_smiles, sample_key_from_normalized


def _git_identity(root: Path) -> dict:
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    diff = subprocess.check_output(["git", "diff", "--binary", "HEAD"], cwd=root)
    return {
        "code_commit": commit, "code_dirty": bool(diff),
        "code_diff_sha256": hashlib.sha256(diff).hexdigest(),
    }


def _write_jsonl(path: Path, rows) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(
                row, sort_keys=True, separators=(",", ":"), ensure_ascii=True
            ) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _roles(manifest: dict, count: int) -> list[list[dict]]:
    memberships = [[] for _ in range(count)]
    for fold in manifest["folds"]:
        fold_id = int(fold["fold"])
        seen = set()
        for role in ("train", "validation", "test"):
            for index in fold[f"{role}_indices"]:
                index = int(index)
                if index in seen or index < 0 or index >= count:
                    raise CacheLifecycleError("fixed split contains invalid overlap/index")
                seen.add(index)
                memberships[index].append({"fold": fold_id, "role": role})
        if len(seen) != count:
            raise CacheLifecycleError("fixed fold does not cover every task row")
    return memberships


def build(raw_root: Path, split_root: Path, output: Path) -> dict:
    raw_root, split_root, output = raw_root.resolve(), split_root.resolve(), output.resolve()
    if output.exists() or output.with_name(output.name + ".staging").exists():
        raise FileExistsError(f"downstream union output already exists: {output}")
    staging = output.with_name(output.name + ".staging")
    staging.mkdir(parents=True)
    rows, unique, raw_values = [], {}, []
    task_counts, split_hashes, source_hashes, fold_counts = {}, {}, {}, {}
    key_tasks = defaultdict(set)
    for task in TASKS:
        csv_path = raw_root / f"smi_{task}.csv"
        split_path = split_root / f"{task}.json"
        if not csv_path.is_file() or not split_path.is_file():
            raise FileNotFoundError(f"missing downstream input for task={task}")
        actual_split = json.loads(split_path.read_text(encoding="utf-8"))
        expected_split = build_manifest(task, csv_path, "outer5_inner20")
        # The checked-in manifests record a repository-relative source_csv;
        # rebuilding with a resolved raw root records the same file as an
        # absolute path.  Compare that field by resolved identity and every
        # other field exactly, without rewriting the fixed manifest.
        observed_source = (Path(__file__).resolve().parents[1]
                           / str(actual_split.get("source_csv", ""))).resolve()
        if observed_source != csv_path:
            raise CacheLifecycleError(f"fixed split source differs for {task}")
        comparable_actual = {**actual_split, "source_csv": str(csv_path)}
        if comparable_actual != expected_split:
            raise CacheLifecycleError(f"fixed outer5_inner20 manifest differs for {task}")
        with csv_path.open(newline="", encoding="utf-8") as handle:
            table = list(csv.reader(handle))
        if not table or len(table[0]) < 2:
            raise CacheLifecycleError(f"invalid property CSV: {csv_path}")
        task_rows = table[1:]
        memberships = _roles(actual_split, len(task_rows))
        task_counts[task] = len(task_rows)
        split_hashes[task] = sha256_file(split_path)
        source_hashes[task] = sha256_file(csv_path)
        fold_counts[task] = [
            {
                "fold": int(fold["fold"]),
                "train": len(fold["train_indices"]),
                "validation": len(fold["validation_indices"]),
                "test": len(fold["test_indices"]),
            }
            for fold in actual_split["folds"]
        ]
        for original_row, values in enumerate(task_rows):
            if len(values) < 2:
                raise CacheLifecycleError(f"truncated property row: {task}/{original_row}")
            source_smiles = str(values[0]).strip()
            try:
                label = float(values[1])
            except ValueError as exc:
                raise CacheLifecycleError(
                    f"invalid label: {task}/{original_row}"
                ) from exc
            if not math.isfinite(label):
                raise CacheLifecycleError(f"nonfinite label: {task}/{original_row}")
            normalized, valid = normalize_polymer_smiles(source_smiles)
            if not valid:
                raise CacheLifecycleError(
                    f"DOWNSTREAM_CACHE_BLOCKER invalid structure: {task}/{original_row}"
                )
            key = sample_key_from_normalized(normalized)
            row = {
                "task": task, "original_row": int(original_row), "label": label,
                "source_smiles": source_smiles, "normalized_smiles": normalized,
                "sample_key": key.hex(), "fold_membership": memberships[original_row],
            }
            rows.append(row)
            raw_values.append(source_smiles)
            key_tasks[key].add(task)
            unique.setdefault(key, {
                "sample_key": key.hex(), "source_smiles": source_smiles,
                "normalized_smiles": normalized,
                "source_row": len(unique),
            })
    structures = list(unique.values())
    keys = np.frombuffer(
        b"".join(bytes.fromhex(row["sample_key"]) for row in rows),
        dtype=np.uint8,
    ).reshape(-1, 32)
    np.save(staging / "row_keys.npy", keys)
    _write_jsonl(staging / "rows.jsonl", rows)
    _write_jsonl(staging / "structures.jsonl", structures)
    with (staging / "structures.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["smiles", "normalized_smiles"])
        for row in structures:
            writer.writerow([row["source_smiles"], row["normalized_smiles"]])
        handle.flush()
        os.fsync(handle.fileno())
    manifest = {
        "artifact_type": "glt_dual_downstream_union",
        "task_order": list(TASKS),
        "row_count": len(rows),
        "task_counts": task_counts,
        "unique_raw_structure_count": len(set(raw_values)),
        "unique_normalized_structure_count": len(structures),
        "cross_task_duplicated_structure_count": sum(
            len(tasks) > 1 for tasks in key_tasks.values()
        ),
        "row_duplicate_count": len(rows) - len(structures),
        "ordered_row_key_hash": ordered_key_hash(keys),
        "source_csv_sha256": source_hashes,
        "split_manifest_sha256": split_hashes,
        "per_fold_counts": fold_counts,
        "rows_file_sha256": sha256_file(staging / "rows.jsonl"),
        "row_keys_file_sha256": sha256_file(staging / "row_keys.npy"),
        "structures_file_sha256": sha256_file(staging / "structures.jsonl"),
        "structures_csv_sha256": sha256_file(staging / "structures.csv"),
        "split_policy": "outer5_inner20_byte_preserved",
        "dropped_downstream_rows": 0,
        **_git_identity(Path(__file__).resolve().parents[1]),
    }
    atomic_json(staging / "manifest.json", manifest)
    atomic_json(staging / ".frozen", {"manifest_hash": json_hash(manifest)})
    os.replace(staging, output)
    return {**manifest, "manifest_hash": json_hash(manifest), "output": str(output)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", default="data/raw")
    parser.add_argument("--split-root", default="data/splits/mips_outer5_inner20")
    parser.add_argument("--output", required=True)
    parser.add_argument("--report-json")
    args = parser.parse_args()
    result = build(Path(args.raw_root), Path(args.split_root), Path(args.output))
    if args.report_json:
        path = Path(args.report_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(path, result)
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
