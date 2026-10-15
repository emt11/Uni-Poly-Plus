#!/usr/bin/env python3
"""Recheck one bounded PI1M geometry case without worker contention."""

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.pilot_mcl_ph_geometry_store import SOURCE, worker_init
from src.dataset.canonical_periodic import build_canonical_periodic_topology
from src.dataset.trimer_mcl import attach_finite_trimer_mcl, TrimerGeometryRejection


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-index', type=int, required=True)
    args = parser.parse_args()
    if not 0 <= args.source_index < 959588:
        raise ValueError('PI1M source index out of range')
    with SOURCE.open(encoding='utf-8') as stream:
        for index, line in enumerate(stream):
            if index == args.source_index:
                row = json.loads(line)
                break
        else:
            raise ValueError('PI1M source index unavailable')
    worker_init()
    started = time.perf_counter()
    topology = build_canonical_periodic_topology(row['source_smiles'])
    try:
        trimer = attach_finite_trimer_mcl(
            topology, row['source_smiles'], sample_key=row['sample_key'])
        result = {'status': 'PASS', 'atoms': int(trimer.trimer_pos.size(0)),
                  'selected_converged': bool(trimer.selected_converged),
                  'selected_candidate_id': int(trimer.trimer_conformer_candidate_id)}
    except TrimerGeometryRejection as error:
        result = {'status': 'GEOMETRY_REJECTION', 'reason': str(error)}
    result.update(source_index=args.source_index,
                  wall_seconds=time.perf_counter() - started)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
