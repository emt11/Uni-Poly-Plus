#!/usr/bin/env python3
"""Small stage timing probe for the unchanged Trimer generator."""

from concurrent.futures import ProcessPoolExecutor
import argparse
import json
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.pilot_mcl_ph_geometry_store import selected_rows, worker_init


def profile(row):
    from src.dataset.canonical_periodic import build_canonical_periodic_topology
    from src.dataset.trimer_mcl import attach_finite_trimer_mcl
    index, smiles, key = row
    started = time.perf_counter()
    topology = build_canonical_periodic_topology(smiles)
    topo_seconds = time.perf_counter() - started
    trimer = attach_finite_trimer_mcl(topology, smiles, sample_key=key)
    diag = trimer.generation_diagnostics
    return {'source_index': index, 'topology_seconds': topo_seconds,
            'trimer_seconds': time.perf_counter() - started - topo_seconds,
            'embed_seconds': diag['embed_time'], 'mmff_seconds': diag['mmff_time'],
            'stereo_seconds': diag['stereo_time'],
            'candidate_attempts': diag['num_candidates_requested'],
            'mmff_attempts': diag['num_mmff_attempted'],
            'selected_candidate_id': int(trimer.trimer_conformer_candidate_id),
            'selected_energy': float(trimer.trimer_conformer_energy),
            'selected_converged': bool(trimer.selected_converged),
            'star_3d_valid': bool(trimer.star_3d_valid)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples', type=int, default=64)
    parser.add_argument('--workers', type=int, default=16)
    args = parser.parse_args()
    if not 1 <= args.samples <= 256 or not 1 <= args.workers <= 32:
        raise ValueError('stage profile is limited to 256 samples and 32 workers')
    rows = selected_rows(args.samples)
    with ProcessPoolExecutor(max_workers=args.workers, initializer=worker_init) as pool:
        results = list(pool.map(profile, rows))
    for row in results:
        print(json.dumps(row))
    print(json.dumps({'summary': {'samples': len(results),
          'mean_seconds': {k: statistics.mean(r[k] for r in results)
                           for k in ('topology_seconds', 'trimer_seconds',
                                     'embed_seconds', 'mmff_seconds', 'stereo_seconds')},
          'max_trimer_seconds': max(r['trimer_seconds'] for r in results),
          'nonconverged_selected': sum(not r['selected_converged'] for r in results),
          'selected_first_candidate': sum(r['selected_candidate_id'] == 0 for r in results),
          'star_3d_invalid': sum(not r['star_3d_valid'] for r in results)}}))
