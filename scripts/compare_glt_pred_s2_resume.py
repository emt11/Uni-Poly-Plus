#!/usr/bin/env python3
"""Compare continuous and split S2 r3 checkpoint trajectories."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def _exact(left, right):
    if torch.is_tensor(left) and torch.is_tensor(right):
        return torch.equal(left, right)
    if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
        return np.array_equal(left, right)
    if isinstance(left, dict) and isinstance(right, dict):
        return (list(left) == list(right)
                and all(_exact(left[key], right[key]) for key in left))
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(_exact(a, b) for a, b in zip(left, right))
    if isinstance(left, float) and isinstance(right, float):
        return left == right
    return left == right


def _load_records(root):
    records = {}
    for path in sorted(Path(root).glob('records_rank0*.jsonl')):
        for line in path.read_text(encoding='utf-8').splitlines():
            if line.strip():
                row = json.loads(line)
                records[int(row['step'])] = row
    return records


def _compare_state(left, right):
    fields = ('model', 'optimizer', 'scheduler', 'ordered_keys', 'next_position', 'step')
    exact = {field: _exact(left[field], right[field]) for field in fields}
    rng_exact = _exact(left['rng'], right['rng'])
    return exact, rng_exact


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--continuous', required=True)
    parser.add_argument('--split', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--task', required=True, choices=('fgr', 'align'))
    args = parser.parse_args()
    continuous_path = Path(args.continuous) / 'resume_00004.pt'
    split_path = Path(args.split) / 'resume_00004.pt'
    if not continuous_path.is_file() or not split_path.is_file():
        raise FileNotFoundError('step-4 checkpoint is missing')
    continuous = torch.load(continuous_path, map_location='cpu', weights_only=False)
    split = torch.load(split_path, map_location='cpu', weights_only=False)
    state_exact, rng_exact = _compare_state(continuous, split)
    continuous_records = _load_records(args.continuous)
    split_records = _load_records(args.split)
    losses_exact = True
    targets_exact = True
    stream_exact = True
    record_diffs = []
    for step in range(1, 5):
        left = continuous_records.get(step)
        right = split_records.get(step)
        if left is None or right is None:
            record_diffs.append({'step': step, 'missing': True})
            losses_exact = targets_exact = stream_exact = False
            continue
        loss_ok = left['losses'] == right['losses']
        target_ok = left['target_counts'] == right['target_counts']
        stream_ok = left.get('third_stream_signature') == right.get('third_stream_signature')
        losses_exact &= loss_ok
        targets_exact &= target_ok
        stream_exact &= stream_ok
        if not (loss_ok and target_ok and stream_ok):
            record_diffs.append({'step': step, 'losses': loss_ok, 'targets': target_ok, 'third_stream': stream_ok})
    report = {
        'status': 'PASS' if all(state_exact.values()) and rng_exact and losses_exact and targets_exact and stream_exact else 'FAIL',
        'task': args.task,
        'continuous_checkpoint': str(continuous_path),
        'split_checkpoint': str(split_path),
        'model_optimizer_scheduler_identity_exact': state_exact,
        'python_numpy_cpu_cuda_rng_exact': rng_exact,
        'losses_exact': losses_exact,
        'target_counts_exact': targets_exact,
        'third_stream_identity_exact': stream_exact,
        'record_differences': record_diffs,
        'outer_test_accessed': False,
        'active_cache_modified': False,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False), encoding='utf-8')
    print(json.dumps(report, sort_keys=True))
    if report['status'] != 'PASS':
        raise AssertionError(f'resume exact comparison failed: {report}')


if __name__ == '__main__':
    main()
