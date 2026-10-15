#!/usr/bin/env python3
"""Compare an exact edge-collapse prototype to the stored PH presentations."""
import argparse
import json
from pathlib import Path
import sys
import time

import gudhi
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.dataset import mcl_ph_view as view
from src.dataset.mcl_ph_direct import DirectMCLSource
from src.dataset.mcl_ph_trajectory_cache import MCLPHTrajectoryCache
from src.training.glt_dual_runtime import OrderedSampleStream, load_sample_index_artifact


def collapsed_pairs(points, *, max_edge=view.VR_MAX_EDGE):
    cloud = view._as_point_cloud(points)
    pairs = {0: [], 1: []}
    if not len(cloud):
        return pairs
    tree = gudhi.RipsComplex(points=cloud.tolist(), max_edge_length=float(max_edge)).create_simplex_tree(max_dimension=1)
    tree.collapse_edges()
    tree.expansion(2)
    for dimension, (birth, death) in tree.persistence(homology_coeff_field=2, persistence_dim_max=True):
        if dimension in pairs and birth < death:
            pairs[dimension].append((float(birth), float(death)))
    return pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--sample-index', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--positions', type=int, default=32)
    parser.add_argument('--repeats', type=int, default=2)
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError(args.report)
    torch.set_num_threads(1)
    train = load_sample_index_artifact(args.sample_index, 'train')['indices']
    stream = OrderedSampleStream(len(train), 42)
    cache = MCLPHTrajectoryCache(args.cache, verify_checksums=False)
    positions = np.linspace(0, len(cache) - 1, args.positions, dtype=np.int64)
    rows = [int(train[stream.index_at(int(p))]) for p in positions]
    selected = sorted(set(rows))
    local = {row: i for i, row in enumerate(selected)}
    source = DirectMCLSource(args.source, task=None, selected_indices=selected, split_manifest=None)
    original = view.persistence_pairs
    clouds, expected, identities = [], [], []
    try:
        for position, row in zip(positions, rows):
            i = local[row]
            record = source.record(i)
            if not bool(record.static['geometry_valid']):
                continue
            shared = view.build_mcl_ph_view(record.topology, record.trimer,
                source.entries[i]['source_smiles'], seed=42, key=source.samples[i][0].hex(),
                position=int(position), sigma=0.03, ratio=0.30)
            clouds.append(shared.field_view.positions.numpy())
            expected.append(np.asarray(cache[int(position)]).copy())
            identities.append({'position': int(position), 'source_index': row, 'atoms': len(clouds[-1])})
        for cloud, cached in zip(clouds, expected):
            ref = original(cloud)
            candidate = collapsed_pairs(cloud)
            for dimension in (0, 1):
                if sorted(ref[dimension]) != sorted(candidate[dimension]):
                    raise ValueError('edge collapse changed persistence intervals')
            for implementation in (original, collapsed_pairs):
                view.persistence_pairs = implementation
                np.testing.assert_array_equal(view.five_descriptors(cloud).astype(np.float32), cached)
        timings = {'reference': [], 'edge_collapse': []}
        for repeat in range(args.repeats):
            order = [('reference', original), ('edge_collapse', collapsed_pairs)]
            if repeat % 2:
                order.reverse()
            for name, implementation in order:
                view.persistence_pairs = implementation
                start = time.perf_counter()
                for cloud in clouds:
                    view.five_descriptors(cloud)
                timings[name].append(time.perf_counter() - start)
        means = {name: float(np.mean(values)) for name, values in timings.items()}
        report = {'conclusion': 'PASS', 'scope': '32 existing presentations; CPU PH only; no training',
            'gudhi': gudhi.__version__, 'positions': identities, 'repeats': args.repeats,
            'seconds': timings, 'mean_seconds': means,
            'speedup': means['reference'] / means['edge_collapse'],
            'intervals': 'exact H0/H1 multiset', 'trajectories': 'exact float32 match to frozen cache',
            'coordinate_generation': False, 'optimizer_updates': 0}
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report), flush=True)
    finally:
        view.persistence_pairs = original
        source.close()
        cache.close()


if __name__ == '__main__':
    main()
