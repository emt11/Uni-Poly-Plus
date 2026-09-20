#!/usr/bin/env python3
"""Build the small downstream PH sidecar for the retention round.

Only the structures this round's folds (default 0/1) actually train or validate
on are computed, deduplicated across tasks by the full 32-byte sample key, using
the frozen downstream Trimer coordinates and the existing ``glt-ph-betti-v2``
algorithm.  The layout matches the pretraining sidecar so the same key-addressed
reader can be reused.  The F_CONST profile is the mean of the *valid* profiles in
the existing P_train sidecar -- never a downstream validation/test statistic.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

from src.dataset.glt_ph import PH_BINS, PH_CHANNELS, PH_SCHEMA, ph_profile
from src.training.glt_dual_runtime import open_source

CONST_CHUNK = 50000


def p_train_mean_profile(root):
    """Mean of the valid P_train profiles, computed in chunks over the mmap."""
    root = Path(root)
    profiles = np.load(root / 'ph_profile.npy', mmap_mode='r')
    valid = np.asarray(np.load(root / 'ph_valid.npy', mmap_mode='r'))
    total = np.zeros((PH_CHANNELS, PH_BINS), dtype=np.float64)
    count = 0
    for start in range(0, len(profiles), CONST_CHUNK):
        stop = min(start + CONST_CHUNK, len(profiles))
        block_valid = valid[start:stop]
        if not block_valid.any():
            continue
        block = np.asarray(profiles[start:stop], dtype=np.float64)[block_valid]
        if not np.isfinite(block).all():
            raise ValueError('P_train sidecar contains non-finite profiles')
        total += block.sum(axis=0)
        count += int(block_valid.sum())
    if count == 0:
        raise ValueError('P_train sidecar has no valid profile')
    return (total / count).astype(np.float32), count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('cohort-root', 'cache-root', 'dual-static-root', 'output-root'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--split-root', default='data/splits/mips_outer5_inner20')
    parser.add_argument('--tasks', nargs='*', default=['xc', 'eps', 'eat'])
    parser.add_argument('--folds', type=int, nargs='*', default=[0, 1])
    parser.add_argument('--p-train-sidecar', required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    output = Path(args.output_root)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'refusing to overwrite an existing sidecar: {output}')
    output.mkdir(parents=True, exist_ok=True)

    keys, profiles, valids = [], [], []
    row_of = {}
    per_task, failures = {}, []
    for task in args.tasks:
        manifest_path = Path(args.split_root) / f'{task}.json'
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        needed = set()
        for fold in manifest['folds']:
            if fold['fold'] in args.folds:
                needed |= set(fold['train_indices']) | set(fold['validation_indices'])
        source, _ = open_source(args.cohort_root, args.cache_root, task=task,
                                dual_static_root=args.dual_static_root)
        try:
            rows_before = len(keys)
            reused, fresh_invalid = 0, 0
            for index in sorted(needed):
                key = source.samples[int(index)][0]
                if len(key) != 32:
                    raise ValueError(f'sample key must be 32 bytes, got {len(key)}')
                hex_key = key.hex()
                if hex_key in row_of:
                    reused += 1
                    continue
                _, trimer, _ = source[int(index)]
                if trimer is None:
                    raise ValueError(f'no frozen Trimer geometry for key {hex_key}')
                # A degenerate but legitimately present geometry stays on the
                # invalid contract (zeros + valid=False); only a mapping or
                # identity failure raises.
                profile, ok = ph_profile(
                    np.asarray(trimer.trimer_pos, dtype=np.float64),
                    np.asarray(trimer.trimer_atomic_number, dtype=np.int64))
                if not np.isfinite(profile).all():
                    raise ValueError(f'non-finite PH profile for key {hex_key}')
                if not ok:
                    fresh_invalid += 1
                    failures.append({'task': task, 'key': hex_key, 'reason': 'degenerate'})
                row_of[hex_key] = len(keys)
                keys.append(np.frombuffer(key, dtype=np.uint8)[:32])
                profiles.append(profile)
                valids.append(bool(ok))
            per_task[task] = {
                'manifest_sha256': hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                'needed_structures': len(needed),
                'deduplicated_reuse': reused,
                'new_rows': len(keys) - rows_before,
                'new_invalid': fresh_invalid,
                'cohort_manifest_hash': source.cohort['manifest_hash'],
            }
        finally:
            source.close()

    const_profile, const_count = p_train_mean_profile(args.p_train_sidecar)
    np.save(output / 'sample_keys.npy', np.stack(keys).astype(np.uint8))
    np.save(output / 'ph_profile.npy', np.stack(profiles).astype(np.float32))
    np.save(output / 'ph_valid.npy', np.asarray(valids, dtype=bool))
    np.save(output / 'p_train_mean_profile.npy', const_profile)
    total_valid = int(sum(valids))
    metadata = {
        'schema_version': PH_SCHEMA,
        'records': len(keys), 'valid_records': total_valid,
        'invalid_records': len(keys) - total_valid,
        'channels': PH_CHANNELS, 'bins': PH_BINS, 'radius': [0.8, 6.0],
        'tasks': per_task, 'folds': list(args.folds),
        'dedup': 'full 32-byte sample key across tasks',
        'const_profile_source': {
            'sidecar': str(args.p_train_sidecar),
            'valid_rows': const_count,
            'means': const_profile.tolist(),
        },
        'failures': failures,
        'build_seconds': time.perf_counter() - started,
        'note': ('structures used by the selected folds only; a structure shared with '
                 'another fold\'s legal training set is cached once and does not change '
                 'any fold boundary'),
    }
    (output / 'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    (output / '.done').write_text('ok\n', encoding='utf-8')
    print(json.dumps({'records': len(keys), 'valid': total_valid,
                      'invalid': len(keys) - total_valid,
                      'reused': len(keys) and (sum(row['deduplicated_reuse']
                                                   for row in per_task.values())),
                      'const_valid_rows': const_count,
                      'seconds': metadata['build_seconds']}, indent=2))


if __name__ == '__main__':
    main()
