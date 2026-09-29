#!/usr/bin/env python3
"""Create the single deterministic ``paper5_outer`` MIPS split contract."""

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import KFold, train_test_split


TASKS = ("eat", "eea", "egb", "egc", "ei", "eps", "nc", "xc")
PROTOCOL = "paper5_outer"
OUTER_SEED = 100
INNER_SEED = 42


def digest(values):
    return hashlib.sha256("\n".join(map(str, values)).encode("utf-8")).hexdigest()


def build_manifest(task, path, protocol=PROTOCOL):
    """Build one reproducible five-fold manifest.

    The outer split matches the Periodic-TDL implementation (KFold seed 100).
    The released implementation sorts each outer-train complement before the
    fixed-seed inner split; validation uses ``random_state=42``.  This ordering
    reproduces the five checked-in official fold files exactly.
    """
    if str(protocol) != PROTOCOL:
        raise ValueError(f"only {PROTOCOL!r} is active; got {protocol!r}")
    task = str(task).lower()
    if task not in TASKS:
        raise ValueError(f"unknown task {task!r}; expected one of {TASKS}")
    path = Path(path)
    frame = pd.read_csv(path)
    smiles = frame.iloc[:, 0].astype(str).str.strip().tolist()
    if len(smiles) < 5:
        raise ValueError(f"{task}: at least five rows are required")

    folds = []
    splitter = KFold(n_splits=5, shuffle=True, random_state=OUTER_SEED)
    for fold, (outer_train, held_out) in enumerate(splitter.split(smiles)):
        outer_train = sorted(int(index) for index in outer_train)
        train, validation = train_test_split(
            outer_train, test_size=0.20, shuffle=True, random_state=INNER_SEED,
        )
        train = [int(index) for index in train]
        validation = [int(index) for index in validation]
        test = [int(index) for index in held_out]
        sets = {"train": set(train), "validation": set(validation), "test": set(test)}
        if any(sets[left] & sets[right]
               for left, right in (("train", "validation"),
                                   ("train", "test"),
                                   ("validation", "test"))):
            raise AssertionError(f"{task}/fold{fold}: split overlap")
        if set.union(*sets.values()) != set(range(len(smiles))):
            raise AssertionError(f"{task}/fold{fold}: split does not cover the cohort")
        folds.append({
            "fold": fold,
            "train_indices": train,
            "validation_indices": validation,
            "test_indices": test,
            "inner_split_random_state": INNER_SEED,
            "sets_disjoint": True,
            "union_equals_full_cohort": True,
        })

    sample_order_digest = digest(smiles)
    return {
        "schema": "mips-paper5-outer-fold-v1",
        "protocol": PROTOCOL,
        "task": task,
        "source_csv": str(path),
        "sample_count": len(smiles),
        "sample_order_hash": sample_order_digest,
        "sample_order_sha256": sample_order_digest,
        "splitter": {
            "name": "KFold",
            "n_splits": 5,
            "shuffle": True,
            "random_state": OUTER_SEED,
        },
        "validation_is_test": False,
        "inner_splitter": {
            "name": "train_test_split",
            "test_size": 0.20,
            "shuffle": True,
            "random_state": INNER_SEED,
        },
        "folds": folds,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", default="data/raw")
    parser.add_argument("--output", default="data/splits/mips_paper5_outer")
    parser.add_argument("--protocol", choices=(PROTOCOL,), default=PROTOCOL)
    parser.add_argument("--tasks", nargs="+", default=list(TASKS), choices=TASKS)
    args = parser.parse_args()

    destination = Path(args.output)
    destination.mkdir(parents=True, exist_ok=True)
    for task in args.tasks:
        path = Path(args.raw_root) / f"smi_{task}.csv"
        payload = build_manifest(task, path, args.protocol)
        output = destination / f"{task}.json"
        temporary = output.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
        temporary.replace(output)
        print(output)


if __name__ == "__main__":
    main()
