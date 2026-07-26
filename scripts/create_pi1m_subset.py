#!/usr/bin/env python3
"""Create deterministic nested PI1M subsets without using property labels."""

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="data/raw/PI1M_v2.csv")
    parser.add_argument("--base", default="data/raw/PI1M_50k.csv")
    parser.add_argument("--output", required=True)
    parser.add_argument("--sample-size", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def smiles_column(frame):
    for name in ("smiles", "SMILES", "p_smiles", "PSMILES"):
        if name in frame.columns:
            return name
    return frame.columns[0]


def stable_key(smiles, seed):
    payload = f"{seed}\0{smiles}".encode("utf-8")
    return hashlib.blake2b(payload, digest_size=16).digest()


def main():
    args = parse_args()
    source = pd.read_csv(args.source)
    column = smiles_column(source)
    source[column] = source[column].astype(str).str.strip()
    source = source[source[column].ne("")].drop_duplicates(column, keep="first")
    source = source.set_index(column, drop=False)

    base_path = Path(args.base)
    base_smiles = []
    if base_path.is_file():
        base = pd.read_csv(base_path)
        base_column = smiles_column(base)
        base_smiles = list(dict.fromkeys(base[base_column].astype(str).str.strip()))
    missing = [smiles for smiles in base_smiles if smiles not in source.index]
    if missing:
        raise ValueError(f"Base subset contains {len(missing)} SMILES absent from source")
    if args.sample_size < len(base_smiles):
        raise ValueError("sample-size cannot be smaller than the base subset")
    if args.sample_size > len(source):
        raise ValueError("sample-size exceeds unique source SMILES")

    base_set = set(base_smiles)
    candidates = [smiles for smiles in source.index if smiles not in base_set]
    candidates.sort(key=lambda smiles: stable_key(smiles, args.seed))
    selected = base_smiles + candidates[:args.sample_size - len(base_smiles)]
    output = source.loc[selected].reset_index(drop=True)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    metadata = {
        "source": args.source,
        "base": args.base if base_path.is_file() else None,
        "sample_size": len(output),
        "base_size": len(base_smiles),
        "seed": args.seed,
        "nested": True,
    }
    output_path.with_suffix(".subset.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
