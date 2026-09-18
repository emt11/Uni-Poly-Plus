#!/usr/bin/env python3
"""Bounded read-only S2 smoke on real PI1M records (never opens outer-test data)."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch

from src.dataset.glt_dual_pretrain import prepare_pretrain_sample, pretrain_collate
from src.modules.glt_dual_pretrain import DualPretrainer, global_objective
from src.training.glt_dual_runtime import open_source


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cohort-root', required=True)
    parser.add_argument('--cache-root', required=True)
    parser.add_argument('--dual-static-root', required=True)
    parser.add_argument('--pretrain-target-root')
    parser.add_argument('--samples', type=int, default=4)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if not 1 <= args.samples <= 32:
        raise ValueError('--samples must be in [1,32]')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    result = {'status': 'PASS', 'outer_test_accessed': False, 'sample_limit': args.samples,
              'device': str(device), 'tasks': {}}
    for task in ('fp', 'none', 'fgr', 'align'):
        source, _ = open_source(
            args.cohort_root, args.cache_root,
            dual_static_root=args.dual_static_root,
            pretrain_target_root=(args.pretrain_target_root if task == 'fp' else None),
        )
        try:
            rows = []
            for index in range(min(args.samples, len(source))):
                key, _ = source.samples[index]
                rows.append(prepare_pretrain_sample(
                    *source[index], seed=42, key=key.hex(), position=index,
                    sigma=0.03, ratio=0.3, static=source.static_for(index),
                    target=(source.target_for(index) if task == 'fp' else None),
                    third_task=task, fgr_mu=0.0, fgr_sigma=1.0,
                    fgr_max_pairs=32,
                ))
            data, labels = pretrain_collate(rows)
        finally:
            source.close()
        model = DualPretrainer('concat', third_task=task).to(device)
        model.train()
        data, labels = data.to(device), {
            key: value.to(device) if torch.is_tensor(value) else value
            for key, value in labels.items()
        }
        output = model(data, labels)
        counts = output['counts'].detach().clone().clamp_min(1)
        loss = global_objective(output['sums'], counts)
        if not torch.isfinite(loss):
            raise FloatingPointError(f'{task} loss is non-finite')
        loss.backward()
        finite_gradients = all(
            parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
            for parameter in model.parameters()
        )
        if not finite_gradients:
            raise FloatingPointError(f'{task} gradient is non-finite')
        result['tasks'][task] = {
            'sample_count': len(rows),
            'valid_counts': output['counts'].detach().cpu().tolist(),
            'target_counts': output['targets'].detach().cpu().tolist(),
            'loss': float(loss.detach().cpu()),
            'finite_gradients': finite_gradients,
        }
        del model, data, labels, output, loss
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding='utf-8')
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
