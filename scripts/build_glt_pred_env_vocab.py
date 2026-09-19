#!/usr/bin/env python3
"""Build the S4 environment-atom vocabulary over P_train only.

Categories are ``{central element}|{sorted (neighbor element, bond type)
multiset}`` over the canonical periodic chemical bonds of each sample (built
from the frozen normalized SMILES: center-internal bonds plus both periodic
seam directions).  Categories seen on fewer than --min-count atoms collapse
to ``<unk>``.  Workers report per-category atom counts, so the P_train UNK
fraction is derived from the same pass; only the 1,024 fixed-P_val samples
are re-scanned for their own fraction.
"""
import argparse
import json
import multiprocessing as mp
import time
from collections import Counter
from pathlib import Path

import torch

from src.training.glt_dual_runtime import load_sample_index_artifact, open_source


def _worker_init(cohort_root, cache_root, dual_static_root):
    global _WORKER_SOURCE
    _WORKER_SOURCE, _ = open_source(cohort_root, cache_root,
                                    dual_static_root=dual_static_root)


def _sample_category_counts(source_index):
    """{category: atom count} for one frozen sample."""
    from src.dataset.glt_dual_pretrain import _topology_z, periodic_bond_table

    topology, _, _ = _WORKER_SOURCE[int(source_index)]
    normalized = getattr(topology, 'normalized_canonical_smiles', None)
    if normalized is None:
        raise ValueError('frozen topology is missing normalized_canonical_smiles')
    bond_table = periodic_bond_table(
        str(normalized), topology.canonical_to_trimer_base_atom_id)
    z = _topology_z(topology).tolist()
    neighbors = {}
    for ca, cb, bond_type, _ in bond_table:
        neighbors.setdefault(ca, []).append((z[cb], bond_type))
        neighbors.setdefault(cb, []).append((z[ca], bond_type))
    counts = Counter()
    for atom, pairs in neighbors.items():
        category = f'{z[atom]}|' + ';'.join(f'{element}:{bond}'
                                            for element, bond in sorted(pairs))
        counts[category] += 1
    return counts


def _sample_chunk(indices):
    counts = Counter()
    for source_index in indices:
        counts.update(_sample_category_counts(source_index))
    return counts


def _scan(indices, workers, cohort_root, cache_root, dual_static_root):
    """Per-category atom counts; returns (counts, None) when running inline."""
    counts = Counter()
    chunks = [indices[start:start + 2048].tolist()
              for start in range(0, len(indices), 2048)]
    if int(workers) <= 1:
        for values in chunks:
            counts.update(_sample_chunk(values))
        return counts, None
    with mp.Pool(processes=int(workers), initializer=_worker_init,
                 initargs=(cohort_root, cache_root, dual_static_root)) as pool:
        for part in pool.imap(_sample_chunk, chunks):
            counts.update(part)
    return counts, None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pi1m-cohort-root', required=True)
    parser.add_argument('--cache-root', required=True)
    parser.add_argument('--dual-static-root', required=True)
    parser.add_argument('--split-artifact', required=True)
    parser.add_argument('--min-count', type=int, default=20)
    parser.add_argument('--workers', type=int, default=12)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.min_count < 0:
        raise ValueError('--min-count must be non-negative')

    train = load_sample_index_artifact(args.split_artifact, 'train')
    fixed = load_sample_index_artifact(args.split_artifact, 'fixed_validation')
    source, _ = open_source(args.pi1m_cohort_root, args.cache_root,
                            dual_static_root=args.dual_static_root)
    source.close()
    started = time.perf_counter()
    train_counts, _ = _scan(train['indices'], args.workers, args.pi1m_cohort_root,
                            args.cache_root, args.dual_static_root)
    total_atoms = sum(train_counts.values())
    vocabulary = sorted(category for category, seen in train_counts.items()
                        if seen >= int(args.min_count))
    mapping = {category: index for index, category in enumerate(vocabulary)}
    unk = len(vocabulary)
    mapping['<unk>'] = unk
    vocab_atoms = sum(train_counts[category] for category in vocabulary)
    fixed_counts, _ = _scan(fixed['indices'], args.workers, args.pi1m_cohort_root,
                            args.cache_root, args.dual_static_root)
    fixed_atoms = sum(fixed_counts.values())
    fixed_unk = sum(count for category, count in fixed_counts.items()
                    if mapping.get(category, unk) == unk)
    payload = {
        'schema_version': 'glt-pred-env-vocab-v1',
        'seed': 42,
        'min_count': int(args.min_count),
        'p_train_count': int(len(train['indices'])),
        'vocab_size': unk + 1,
        'categories': mapping,
        'p_train_unk_fraction': (total_atoms - vocab_atoms) / max(1, total_atoms),
        'p_train_atoms': total_atoms,
        'fixed_pval_unk_fraction': fixed_unk / max(1, fixed_atoms),
        'fixed_pval_atoms': fixed_atoms,
        'scan_seconds': time.perf_counter() - started,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=1,
                                 sort_keys=True) + '\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in payload.items() if k != 'categories'},
                     indent=1, sort_keys=True))


if __name__ == '__main__':
    main()
