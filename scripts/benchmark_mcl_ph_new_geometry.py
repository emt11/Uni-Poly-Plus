#!/usr/bin/env python3
"""Bounded cost probe for rebuilding missing Trimer geometry from raw SMILES."""

import json
import resource
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset.canonical_periodic import build_canonical_periodic_topology
from src.dataset.trimer_mcl import (TrimerContractError, TrimerGeometryRejection,
                                    attach_finite_trimer_mcl)


PROBES = {
    'downstream': (0, 500, 2500, 6000),
    'pi1m': (0, 100000, 500000, 900000),
}
EXTENDED_PROBES = {
    'downstream': tuple(int((i + 0.5) * 6265 / 4) for i in range(4)),
    'pi1m': tuple(int((i + 0.5) * 959588 / 16) for i in range(16)),
}
SOURCES = {
    'downstream': ROOT / 'data/processed/glt_dual_v2/downstream/cohort_1545eda5a8f6a1_paper5_outer/records.jsonl',
    'pi1m': ROOT / 'data/processed/glt_dual_v2/pi1m/cohort_30f17b59bc5862a1/records.jsonl',
}


def selected_rows(path, indices):
    wanted = set(indices)
    selected = {}
    with path.open(encoding='utf-8') as stream:
        for index, line in enumerate(stream):
            if index in wanted:
                selected[index] = json.loads(line)
            if len(selected) == len(wanted):
                break
    if set(selected) != wanted:
        raise ValueError('probe source is missing a requested row')
    return selected


def main(extended=False):
    for source_name, indices in (EXTENDED_PROBES if extended else PROBES).items():
        selected = selected_rows(SOURCES[source_name], indices)
        for index in indices:
            row = selected[index]
            started = time.perf_counter()
            result = {'source': source_name, 'source_index': index,
                      'source_smiles': row['source_smiles']}
            try:
                data = build_canonical_periodic_topology(row['source_smiles'])
                data = attach_finite_trimer_mcl(
                    data, row['source_smiles'], sample_key=row['sample_key'])
                result.update(status='PASS',
                              canonical_atoms=int(data.mips_x.size(0)),
                              trimer_atoms=int(data.trimer_atomic_number.numel()),
                              heavy_atoms=int((data.trimer_atomic_number > 1).sum()))
            except (TrimerGeometryRejection, TrimerContractError) as error:
                result.update(status=type(error).__name__, reason=str(error))
            result['wall_seconds'] = time.perf_counter() - started
            result['max_rss_mb'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
            print(json.dumps(result), flush=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--extended', action='store_true')
    args = parser.parse_args()
    main(extended=args.extended)
