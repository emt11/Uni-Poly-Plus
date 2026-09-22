#!/usr/bin/env python3
"""One-global-batch equivalence gate for the trajectory cache (r10R2, section 23).

Builds a real cache for exactly the first global batch of the declared schedule
(positions ``0 .. 1007``) and then reads that batch twice through the *training*
DataLoader path -- once with the online trajectory computation and once with the
cache -- comparing every column the model consumes.  This is the hard gate that
precedes the full 5.04M build: a trajectory-only comparison would miss a cache
that quietly changed the RNG stream, and an inline comparison would miss a cache
that only breaks under worker processes.
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from scripts.build_mcl_ph_trajectory_cache import BUILDER_CLASS, build_cache
from src.dataset.mcl_ph_trajectory_cache import (MCLPHTrajectoryCache, cache_identity,
                                                 sha256_file)
from src.dataset.mcl_ph_view import (MCLPHMicrobatchStream, load_geometric_statistics,
                                     mcl_ph_collate, prepare_mcl_ph_sample)
from src.training.glt_dual_runtime import (IndexedFrozenDualSource, OrderedSampleStream,
                                           load_sample_index_artifact, open_source)

BATCH_FIELDS = ('mcl_z', 'mcl_charge', 'mcl_aromatic', 'mcl_masked', 'mcl_pos',
                'mcl_bond_type', 'mcl_edge_scale', 'mcl_edge_distance', 'mcl_edge_type',
                'mcl_central_index', 'mcl_image_to_canonical', 'mcl_bond_index',
                'mcl_edge_index', 'mcl_atom_batch', 'mcl_trajectory', 'mcl_geometry_valid',
                'mcl_readout_valid')
LABEL_FIELDS = ('atom_mask', 'atom_label', 'mcl_length_target', 'mcl_length_raw',
                'mcl_angle_target', 'mcl_nonbond_target', 'mcl_nonbond_raw',
                'mcl_nonbond_slot', 'mcl_length_pair', 'mcl_angle_pair', 'mcl_nonbond_pair',
                'mcl_length_graph', 'mcl_angle_graph', 'mcl_nonbond_graph',
                'mcl_local_valid', 'mcl_nonbond_valid', 'mcl_center_bonds',
                'mcl_geometry_valid', 'mcl_readout_valid')


def parse():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/mts/mcl_ph_cat.json')
    parser.add_argument('--cohort-root',
                        default='data/processed/glt_dual_v2/pi1m/cohort_30f17b59bc5862a1')
    parser.add_argument('--cache-root', default='data/processed/mips_trimer_scage')
    parser.add_argument('--dual-static-root',
                        default='data/processed/glt_dual_v2/pi1m/dual_static_v1')
    parser.add_argument('--sample-index-artifact',
                        default='results/glt_pred_20260918/s3b_prep/pretrain_split_v1.json')
    parser.add_argument('--statistics', default='results/mcl_ph_20260921/p0/statistics.npz')
    parser.add_argument('--output', default='results/mcl_ph_20260921/p2r2_gate/cache_1008')
    parser.add_argument('--report', default='results/mcl_ph_20260921/p2r2_gate/batch_gate.json')
    parser.add_argument('--workers', type=int, default=12)
    parser.add_argument('--shard-size', type=int, default=100_000)
    parser.add_argument('--chunk-size', type=int, default=252)
    return parser.parse_args()


def load_batch(source, statistics, *, positions, workers, trajectory_cache):
    """The full global batch of the first step, through the real stream."""
    dataset = MCLPHMicrobatchStream(
        source, seed=42, world=1, rank=0, microbatch=84, accumulation=positions // 84,
        start_step=0, max_steps=1, sigma=0.03, ratio=0.3, statistics=statistics,
        trajectory_cache=trajectory_cache)
    if workers == 0:
        return [dataset[item] for item in range(len(dataset))]
    loader = torch.utils.data.DataLoader(dataset, batch_size=None, num_workers=workers,
                                         prefetch_factor=2, persistent_workers=False)
    return list(loader)


def main():
    args = parse()
    config = json.loads((ROOT / args.config).read_text(encoding='utf-8'))
    positions = 1008
    identity = None
    started = time.perf_counter()
    probe = BUILDER_CLASS(cohort_root=str(ROOT / args.cohort_root),
                          cache_root=str(ROOT / args.cache_root),
                          dual_static_root=str(ROOT / args.dual_static_root),
                          sample_index_artifact=str(ROOT / args.sample_index_artifact),
                          sample_index_split='train', output=str(ROOT / args.output),
                          shard_size=args.shard_size, total_positions=positions, seed=42,
                          sigma=float(config['noise_sigma']),
                          ratio=float(config['atom_mask_ratio']), identity={})
    try:
        identity = cache_identity(
            seed=42, noise_sigma=float(config['noise_sigma']),
            mask_ratio=float(config['atom_mask_ratio']), global_batch=1008,
            sample_index_split='train', sample_index_artifact_sha256=probe.split_sha256,
            cohort_manifest_hash=probe.source.cohort['manifest_hash'],
            dual_static_manifest_hash=probe.source.static_cache.manifest_hash)
    finally:
        probe.close()
    print(f'=== identity {json.dumps(identity, sort_keys=True)}', flush=True)
    payload = dict(cohort_root=str(ROOT / args.cohort_root),
                   cache_root=str(ROOT / args.cache_root),
                   dual_static_root=str(ROOT / args.dual_static_root),
                   sample_index_artifact=str(ROOT / args.sample_index_artifact),
                   sample_index_split='train', output=str(ROOT / args.output),
                   shard_size=args.shard_size, total_positions=positions, seed=42,
                   sigma=float(config['noise_sigma']),
                   ratio=float(config['atom_mask_ratio']), identity=identity)
    manifest = build_cache(builder_payload=payload, root=ROOT / args.output,
                           shard_size=args.shard_size, total_positions=positions,
                           chunk_size=args.chunk_size, workers=args.workers,
                           requested_total_positions=positions, max_updates=1)
    build_seconds = time.perf_counter() - started
    print(f'=== cache built in {build_seconds:.1f}s sha256={manifest["shards"][0]["sha256"]}',
          flush=True)

    statistics = load_geometric_statistics(ROOT / args.statistics)
    split = load_sample_index_artifact(str(ROOT / args.sample_index_artifact), 'train')
    base, _ = open_source(str(ROOT / args.cohort_root), str(ROOT / args.cache_root),
                          dual_static_root=str(ROOT / args.dual_static_root))
    source = IndexedFrozenDualSource(base, split['indices'])
    cache = MCLPHTrajectoryCache(ROOT / args.output, expected=identity,
                                 required_positions=positions)
    report = {'plan': 'MCL-PH-20260921-01/r10R2', 'stage': 'batch_gate',
              'positions': positions, 'workers': int(args.workers),
              'cache_root': str(ROOT / args.output), 'identity': identity,
              'manifest_sha256': manifest['shards'][0]['sha256'],
              'build_seconds': round(build_seconds, 2), 'items': [], 'problems': []}
    try:
        started = time.perf_counter()
        online = load_batch(source, statistics, positions=positions, workers=args.workers,
                            trajectory_cache=None)
        online_seconds = time.perf_counter() - started
        print(f'=== online batch in {online_seconds:.1f}s', flush=True)
        started = time.perf_counter()
        cached = load_batch(source, statistics, positions=positions, workers=args.workers,
                            trajectory_cache=cache)
        cached_seconds = time.perf_counter() - started
        print(f'=== cached batch in {cached_seconds:.1f}s', flush=True)
        report['online_seconds'] = round(online_seconds, 2)
        report['cached_seconds'] = round(cached_seconds, 2)
        assert len(online) == len(cached) == positions // 84
        for item, (first, second) in enumerate(zip(online, cached)):
            batch_a, labels_a = first
            batch_b, labels_b = second
            row = {'item': item, 'problems': []}
            for name in BATCH_FIELDS:
                if not torch.equal(getattr(batch_a, name), getattr(batch_b, name)):
                    row['problems'].append(f'batch.{name}')
            for name in LABEL_FIELDS:
                if not torch.equal(labels_a[name], labels_b[name]):
                    row['problems'].append(f'labels.{name}')
            if list(labels_a['sample_key']) != list(labels_b['sample_key']):
                row['problems'].append('labels.sample_key')
            for name in ('mcl_fallback_count', 'mcl_statistics_applied'):
                if labels_a[name] != labels_b[name]:
                    row['problems'].append(f'labels.{name}')
            row['samples'] = len(labels_a['sample_key'])
            row['trajectory_nonzero_graphs'] = int(
                (batch_a.mcl_trajectory.abs().sum(-1).sum(-1) > 0).sum())
            report['items'].append(row)
            if row['problems']:
                report['problems'].append({'item': item, 'fields': row['problems']})
        # The same batch inline, to prove the reader does not depend on the
        # worker processes either.
        inline = load_batch(source, statistics, positions=positions, workers=0,
                            trajectory_cache=cache)
        for item, (first, second) in enumerate(zip(cached, inline)):
            for name in BATCH_FIELDS:
                if not torch.equal(getattr(first[0], name), getattr(second[0], name)):
                    report['problems'].append({'item': item, 'fields': [f'inline.{name}']})
            for name in LABEL_FIELDS:
                if not torch.equal(first[1][name], second[1][name]):
                    report['problems'].append({'item': item, 'fields': [f'inline.{name}']})
    finally:
        base.close()
    report['status'] = 'PASS' if not report['problems'] else 'FAIL'
    report['samples_compared'] = sum(row['samples'] for row in report['items'])
    destination = Path(args.report)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    print(json.dumps({key: report[key] for key in ('status', 'samples_compared', 'problems',
                                                   'online_seconds', 'cached_seconds')},
                     indent=2, sort_keys=True))
    print(f'=== BATCH GATE {report["status"]}')
    raise SystemExit(0 if report['status'] == 'PASS' else 3)


if __name__ == '__main__':
    main()
