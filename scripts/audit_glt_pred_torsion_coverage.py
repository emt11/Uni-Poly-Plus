#!/usr/bin/env python3
"""Audit real Trimer torsion coverage on deterministic P_train positions.

Positions 0..--samples-1 of the frozen P_train stream (seed 42) are scanned;
per sample the heavy-atom quadruplet count and the count of non-degenerate
(clean-coordinate) torsions are recorded.  If a majority of samples carry no
usable torsion, the S4 TOR trajectories must stop instead of widening the
path definition.
"""
import argparse
import json
import time
from pathlib import Path

import torch

from src.dataset.canonical_periodic import resolve_normalized_identity
from src.dataset.glt_dual_pretrain import (_topology_z, periodic_bond_table,
                                           torsion_features,
                                           torsion_quadruplets)
from src.training.glt_dual_runtime import (IndexedFrozenDualSource,
                                           load_sample_index_artifact,
                                           open_source, OrderedSampleStream)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pi1m-cohort-root', required=True)
    parser.add_argument('--cache-root', required=True)
    parser.add_argument('--dual-static-root', required=True)
    parser.add_argument('--split-artifact', required=True)
    parser.add_argument('--samples', type=int, default=1024)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    artifact = load_sample_index_artifact(args.split_artifact, 'train')
    base, _ = open_source(args.pi1m_cohort_root, args.cache_root,
                          dual_static_root=args.dual_static_root)
    subset = IndexedFrozenDualSource(base, artifact['indices'])
    stream = OrderedSampleStream(len(subset), 42)
    per_sample = []
    started = time.perf_counter()
    for position in range(int(args.samples)):
        index = stream.index_at(position)
        topology, trimer, smiles = subset[index]
        identity = resolve_normalized_identity(topology, smiles, require_fields=True)
        table = periodic_bond_table(identity['normalized_smiles'],
                                    identity['canonical_to_normalized_base'])
        quads = torsion_quadruplets(table, _topology_z(topology))
        valid = 0
        if quads and bool(getattr(trimer, 'trimer_geometry_valid', False)):
            mapping = torch.as_tensor(trimer.mips_to_trimer_central_index,
                                      dtype=torch.long).reshape(-1)
            _, mask = torsion_features(trimer.trimer_pos, mapping, quads)
            valid = int(mask.sum())
        per_sample.append({'position': position, 'quadruplets': len(quads),
                           'valid_torsions': valid})
    covered = sum(1 for row in per_sample if row['valid_torsions'] > 0)
    payload = {
        'schema_version': 'glt-pred-torsion-coverage-v1',
        'seed': 42,
        'samples': len(per_sample),
        'samples_with_valid_torsion': covered,
        'coverage_fraction': covered / max(1, len(per_sample)),
        'quadruplet_mean': sum(row['quadruplets'] for row in per_sample) / max(1, len(per_sample)),
        'valid_mean': sum(row['valid_torsions'] for row in per_sample) / max(1, len(per_sample)),
        'zero_valid_samples': sum(1 for row in per_sample if row['valid_torsions'] == 0),
        'elapsed_seconds': time.perf_counter() - started,
        'per_sample': per_sample,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=1) + '\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in payload.items() if k != 'per_sample'}, indent=1))


if __name__ == '__main__':
    main()
