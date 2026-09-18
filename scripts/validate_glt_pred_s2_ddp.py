#!/usr/bin/env python3
"""Three-rank S2 objective-boundary smoke for FGR and ALIGN only."""

import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from src.dataset.glt_dual_pretrain import prepare_pretrain_sample, pretrain_collate
from src.modules.glt_dual_pretrain import DualPretrainer, global_objective
from src.training.glt_dual_runtime import move_labels, open_source, require_tmux, write_json


def _case_payload(task, case, rank):
    if task == 'fgr':
        valid = (rank == 0) if case == 'partial_zero' else False
        return valid, None
    if task == 'align':
        if case == 'partial_repeat':
            return rank != 2, f'align-{rank}'
        if case == 'all_zero':
            return False, f'align-zero-{rank}'
        if case == 'all_same_identity':
            return True, 'same-identity'
    raise ValueError(f'unsupported S2 DDP case {task}/{case}')


def _run_case(source, task, case, rank, world, device):
    selected = None
    # The selection is deterministic and capped; no random real sample search
    # or benchmark cohort is introduced by this validation.
    for index in range(min(32, len(source))):
        key, _ = source.samples[index]
        row = prepare_pretrain_sample(
            *source[index], seed=42, key=key.hex(), position=index,
            sigma=0.03, ratio=0.3, static=source.static_for(index), target=None,
            third_task=task, fgr_mu=0.0, fgr_sigma=1.0, fgr_max_pairs=32,
        )
        if task != 'fgr' or row[1]['fgr_target'].numel():
            selected = (index, key.hex(), row)
            break
    if selected is None:
        raise RuntimeError(f'no usable {task} row in first 32 records')
    index, key, prepared = selected
    data, labels = pretrain_collate([prepared])
    valid, identity = _case_payload(task, case, rank)
    labels['align_valid'] = torch.tensor([valid], dtype=torch.bool)
    if identity is not None:
        labels['align_identity'] = [identity]
    if task == 'fgr' and not valid:
        data.geometry_valid = torch.zeros_like(data.geometry_valid)
    data, labels = data.to(device), move_labels(labels, device)
    model = DualPretrainer('concat', third_task=task).to(device)
    module = DistributedDataParallel(model, device_ids=[device.index], find_unused_parameters=True)
    result = module(data, labels)
    local_counts = result['counts'].detach().clone()
    global_counts = local_counts.clone()
    dist.all_reduce(global_counts)
    loss = global_objective(result['sums'], global_counts, world)
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError(f'{task}/{case} non-finite loss')
    loss.backward()
    gradients = [p.grad for p in module.module.parameters() if p.grad is not None]
    finite = all(bool(torch.isfinite(g).all()) for g in gradients)
    if not finite:
        raise FloatingPointError(f'{task}/{case} non-finite gradient')
    projection = [p.grad for name, p in module.module.named_parameters()
                  if task == 'align' and name.startswith(('align_proj2', 'align_proj3'))]
    rows = [None] * world
    dist.all_gather_object(rows, {
        'rank': rank, 'sample_index': index, 'sample_key': key,
        'local_counts': local_counts.cpu().tolist(),
        'global_counts': global_counts.cpu().tolist(),
        'loss': float(loss.detach().cpu()), 'finite_gradients': finite,
        'projection_gradients_present': bool(projection),
        'projection_gradients_finite': all(bool(torch.isfinite(g).all()) for g in projection) if projection else True,
    })
    dist.barrier()
    return {
        'rows': rows,
        'global_counts': global_counts.cpu().tolist(),
        'finite_loss': all(torch.isfinite(torch.tensor(row['loss'])) for row in rows),
        'backward_completed': True,
        'finite_gradients': all(row['finite_gradients'] for row in rows),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cohort-root', required=True)
    parser.add_argument('--cache-root', required=True)
    parser.add_argument('--dual-static-root', required=True)
    parser.add_argument('--report-json', required=True)
    args = parser.parse_args()
    require_tmux()
    rank, world = int(os.environ['RANK']), int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])
    if world != 3 or not torch.cuda.is_available():
        raise RuntimeError('S2 DDP smoke requires exactly three CUDA ranks')
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)
    dist.init_process_group('nccl')
    source = None
    try:
        source, _ = open_source(args.cohort_root, args.cache_root,
                                dual_static_root=args.dual_static_root,
                                pretrain_target_root=None)
        cases = {
            'fgr_partial_zero': _run_case(source, 'fgr', 'partial_zero', rank, world, device),
            'fgr_all_zero': _run_case(source, 'fgr', 'all_zero', rank, world, device),
            'align_partial_repeat': _run_case(source, 'align', 'partial_repeat', rank, world, device),
            'align_all_zero': _run_case(source, 'align', 'all_zero', rank, world, device),
            'align_all_same_identity': _run_case(source, 'align', 'all_same_identity', rank, world, device),
        }
        if rank == 0:
            report = {
                'status': 'PASS', 'world_size': world, 'backend': dist.get_backend(),
                'outer_test_accessed': False, 'optimizer_updates': 0, 'cases': cases,
            }
            write_json(args.report_json, report)
            print(json.dumps(report, sort_keys=True), flush=True)
        dist.barrier()
    finally:
        if source is not None:
            source.close()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
