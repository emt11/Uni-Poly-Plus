#!/usr/bin/env python3
"""Test-only fake finetune job that satisfies the scheduler shard contract."""
import argparse
import json

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--shard", required=True)
    parser.add_argument("--prediction", required=True)
    args = parser.parse_args()
    metrics = [{"fold": int(args.fold), "test_r2": 0.5}]
    row = {
        "task": args.task,
        "seed": 42,
        "per_fold_metrics": json.dumps(metrics, sort_keys=True),
        "avg_test_r2": 0.5,
        "std_test_r2": 0.0,
        "avg_test_mae": 1.0,
        "std_test_mae": 0.0,
        "avg_test_rmse": 1.0,
        "std_test_rmse": 0.0,
    }
    from pathlib import Path

    Path(args.shard).parent.mkdir(parents=True, exist_ok=True)
    Path(args.prediction).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(args.shard, index=False)
    values = np.zeros(5, dtype=np.float32)
    indices = np.arange(5, dtype=np.int64)
    metadata = json.dumps({"task": args.task, "seed": 42, "fold": int(args.fold)})
    np.savez(
        args.prediction,
        y_true=values, y_pred=values, sample_indices=indices,
        metadata=np.asarray(metadata),
    )
    print(f"fake train done: {args.task} fold {args.fold}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
