#!/usr/bin/env python3
"""Create deterministic MIPS five-fold manifests."""

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import KFold, train_test_split


TASKS = ("eat", "eea", "egb", "egc", "ei", "eps", "nc", "xc")


def digest(values):
    return hashlib.sha256(
        "\n".join(map(str, values)).encode("utf-8")
    ).hexdigest()


def build_manifest(task, path, protocol):
    frame = pd.read_csv(path)
    smiles = frame.iloc[:, 0].astype(str).str.strip().tolist()
    folds = []
    splitter = KFold(n_splits=5, shuffle=True, random_state=1)
    for fold, (outer_train, held_out) in enumerate(splitter.split(smiles)):
        if protocol == "outer5_inner20":
            train, validation = train_test_split(
                outer_train, test_size=0.20, shuffle=True,
                random_state=42 + fold,
            )
            train = sorted(map(int, train))
            validation = sorted(map(int, validation))
        else:
            train = list(map(int, outer_train))
            validation = list(map(int, held_out))
        folds.append({
            "fold": fold, "train_indices": train,
            "validation_indices": validation,
            "test_indices": list(map(int, held_out)),
            "inner_split_random_state": 42 + fold if protocol == "outer5_inner20" else None,
        })
    separated = protocol == "outer5_inner20"
    return {
        "schema": "mips-outer5-inner20-fold-v1" if separated else "mips-shared-validation-test-fold-v1",
        "protocol": protocol, "task": task, "source_csv": str(path),
        "sample_count": len(smiles), "sample_order_hash": digest(smiles),
        "sample_order_sha256": digest(smiles),
        "splitter": {"name": "KFold", "n_splits": 5, "shuffle": True, "random_state": 1},
        "validation_is_test": not separated,
        "inner_splitter": ({"name": "train_test_split", "test_size": 0.20, "shuffle": True, "random_state": "42 + fold_id"} if separated else None),
        "folds": folds,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", default="data/raw")
    parser.add_argument("--output")
    parser.add_argument(
        "--protocol",
        choices=("historical_shared5", "outer5_inner20"),
        default="historical_shared5",
    )
    parser.add_argument("--tasks", nargs="+", default=list(TASKS))
    args = parser.parse_args()
    destination = Path(args.output or (
        "data/splits/mips_outer5_inner20"
        if args.protocol == "outer5_inner20" else "data/splits/mips_shared5"
    ))
    destination.mkdir(parents=True, exist_ok=True)
    for task in args.tasks:
        path = Path(args.raw_root) / f"smi_{task}.csv"
        payload = build_manifest(task, path, args.protocol)
        output = destination / f"{task}.json"
        temporary = output.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(payload, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(output)
        print(output)


if __name__ == "__main__":
    main()
