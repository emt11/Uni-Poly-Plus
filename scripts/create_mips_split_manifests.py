#!/usr/bin/env python3
"""Create immutable shared validation/test 5-fold manifests."""

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import KFold


TASKS = ("eat", "eea", "egb", "egc", "ei", "eps", "nc", "xc")


def digest(values):
    return hashlib.sha256(
        "\n".join(map(str, values)).encode("utf-8")
    ).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", default="data/raw")
    parser.add_argument("--output", default="data/splits/mips_shared5")
    parser.add_argument("--tasks", nargs="+", default=list(TASKS))
    args = parser.parse_args()
    destination = Path(args.output)
    destination.mkdir(parents=True, exist_ok=True)
    for task in args.tasks:
        path = Path(args.raw_root) / f"smi_{task}.csv"
        frame = pd.read_csv(path)
        smiles = frame.iloc[:, 0].astype(str).str.strip().tolist()
        folds = []
        splitter = KFold(n_splits=5, shuffle=True, random_state=1)
        for fold, (train, held_out) in enumerate(splitter.split(smiles)):
            folds.append({
                "fold": fold,
                "train_indices": train.tolist(),
                "validation_indices": held_out.tolist(),
                "test_indices": held_out.tolist(),
            })
        payload = {
            "schema": "mips-shared-validation-test-fold-v1",
            "protocol": "shared_validation_test_fold",
            "task": task,
            "source_csv": str(path),
            "sample_count": len(smiles),
            "sample_order_hash": digest(smiles),
            "sample_order_sha256": digest(smiles),
            "splitter": {
                "name": "KFold", "n_splits": 5,
                "shuffle": True, "random_state": 1,
            },
            "validation_is_test": True,
            "folds": folds,
        }
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
