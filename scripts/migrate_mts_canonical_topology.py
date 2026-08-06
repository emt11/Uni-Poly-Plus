#!/usr/bin/env python3
"""Migrate one verified explicit MTS topology record.

This utility is deliberately record-scoped.  It never scans or rewrites an
existing cache; callers can use it to audit a sample before constructing a
new canonical-periodic layer.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.dataset.canonical_periodic import migrate_explicit_topology_to_canonical


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("old_topology", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--ru-base", type=Path, default=None)
    parser.add_argument("--max-hops", type=int, default=2)
    args = parser.parse_args()
    old = torch.load(args.old_topology, map_location="cpu", weights_only=False)
    ru = (
        torch.load(args.ru_base, map_location="cpu", weights_only=False)
        if args.ru_base is not None else None
    )
    migrated = migrate_explicit_topology_to_canonical(
        old, ru, max_hops=args.max_hops, verify_features=True
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(migrated, args.output)
    print(
        f"migrated one record: nodes={migrated.mips_x.size(0)} "
        f"relations={migrated.lga_edge_index.size(1)} "
        f"schema={migrated.canonical_periodic_topology_schema}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
