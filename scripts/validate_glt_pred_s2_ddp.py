#!/usr/bin/env python3
"""Three-rank S2 r3 boundary checks for true-Trimer FGR and ALIGN."""

import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel

from src.dataset.glt_dual_pretrain import prepare_pretrain_sample, pretrain_collate
from src.modules.glt_dual_pretrain import (
    DualPretrainer, alignment_loss, global_objective,
)
from src.training.glt_dual_runtime import move_labels, open_source, require_tmux, write_json


def _fgr_case_payload(case, rank):
    if case == 'partial_invalid':
        return rank == 0
    if case == 'all_zero':
        return False
    raise ValueError(f'unsupported FGR case {case}')


def _first_fgr_row(source):
    for index in range(min(32, len(source))):
        key, _ = source.samples[index]
        row = prepare_pretrain_sample(
            *source[index], seed=42, key=key.hex(), position=index,
            sigma=0.03, ratio=0.3, static=source.static_for(index), target=None,
            third_task='fgr', fgr_mu=0.0, fgr_sigma=1.0, fgr_max_pairs=32,
        )
        if row[1]['fgr_target'].numel():
            return index, key.hex(), row
    raise RuntimeError('no usable true-Trimer FGR row in first 32 records')


def _run_fgr_case(source, case, rank, world, device):
    index, key, prepared = _first_fgr_row(source)
    data, labels = pretrain_collate([prepared])
    if not _fgr_case_payload(case, rank):
        data.geometry_valid = torch.zeros_like(data.geometry_valid)
    data, labels = data.to(device), move_labels(labels, device)
    model = DualPretrainer('concat', third_task='fgr').to(device)
    module = DistributedDataParallel(model, device_ids=[device.index], find_unused_parameters=True)
    result = module(data, labels)
    local_counts = result['counts'].detach().clone()
    global_counts = local_counts.clone()
    dist.all_reduce(global_counts)
    loss = global_objective(result['sums'], global_counts, world)
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError(f'fgr/{case} non-finite loss')
    loss.backward()
    gradients = [parameter.grad for parameter in module.module.parameters()
                 if parameter.grad is not None]
    finite = all(bool(torch.isfinite(gradient).all()) for gradient in gradients)
    if not finite:
        raise FloatingPointError(f'fgr/{case} non-finite gradient')
    rows = [None] * world
    dist.all_gather_object(rows, {
        'rank': rank, 'sample_index': index, 'sample_key': key,
        'local_counts': local_counts.cpu().tolist(),
        'global_counts': global_counts.cpu().tolist(),
        'loss': float(loss.detach().cpu()), 'finite_gradients': finite,
    })
    dist.barrier()
    return {
        'rows': rows,
        'global_counts': global_counts.cpu().tolist(),
        'finite_loss': all(torch.isfinite(torch.tensor(row['loss'])) for row in rows),
        'backward_completed': True,
        'finite_gradients': all(row['finite_gradients'] for row in rows),
    }


class _AlignmentPair(nn.Module):
    def __init__(self):
        super().__init__()
        self.p2 = nn.Linear(128, 128, bias=False)
        self.p3 = nn.Linear(128, 128, bias=False)
        with torch.no_grad():
            identity = torch.eye(128)
            self.p2.weight.copy_(identity)
            self.p3.weight.copy_(identity)

    def forward(self, raw2, raw3):
        return F.normalize(self.p2(raw2), dim=-1), F.normalize(self.p3(raw3), dim=-1)


def _alignment_case_payload(case, rank):
    if case == 'partial_invalid':
        return f'ABC'[rank], rank != 2
    if case == 'multi_positive':
        return ['A', 'A', 'B'][rank], True
    if case == 'all_zero':
        return f'zero-{rank}', False
    if case == 'all_same_identity':
        return ['same', 'same', 'same'][rank], True
    raise ValueError(f'unsupported ALIGN case {case}')


def _alignment_reference(z2, z3, identities, valid, temperature=0.1):
    valid = valid.bool()
    distinct = {identity for identity, keep in zip(identities, valid.tolist()) if keep}
    if len(distinct) < 2 or not bool(valid.any()):
        return z2.sum() * 0 + z3.sum() * 0, torch.tensor(0., device=z2.device)
    valid_columns = valid.unsqueeze(0)
    logits23 = z2 @ z3.t() / temperature
    logits32 = z3 @ z2.t() / temperature
    positive = torch.tensor(
        [[left == right and keep for right, keep in zip(identities, valid.tolist())]
         for left in identities], device=z2.device, dtype=torch.bool
    )
    anchor = valid & torch.tensor(
        [identity in distinct for identity in identities], device=z2.device
    )

    def directional(logits):
        denominator = torch.logsumexp(logits.masked_fill(~valid_columns, -torch.inf), dim=1)
        count = positive.sum(1).clamp_min(1).float()
        return -(logits.masked_fill(~positive, 0).sum(1) / count - denominator)

    return (directional(logits23)[anchor] + directional(logits32)[anchor]).sum() / 2, anchor.sum().float()


def _run_align_case(case, rank, world, device):
    torch.manual_seed(20260918 + rank)
    raw2 = torch.zeros(1, 128, device=device)
    raw3 = torch.zeros(1, 128, device=device)
    raw2[0, rank] = 1.0
    raw3[0, (rank + 7) % 128] = 1.0
    identity, valid = _alignment_case_payload(case, rank)
    module = DistributedDataParallel(_AlignmentPair().to(device), device_ids=[device.index])
    z2, z3 = module(raw2, raw3)
    z2.retain_grad()
    z3.retain_grad()
    local_sum, local_count = alignment_loss(
        z2, z3, [identity], torch.tensor([valid], device=device), temperature=0.1
    )
    if not bool(torch.isfinite(local_sum)):
        raise FloatingPointError(f'align/{case} non-finite loss')
    local_z2 = z2.detach().cpu()
    local_z3 = z3.detach().cpu()
    gathered_z2, gathered_z3, gathered_ids, gathered_valid = [None] * world, [None] * world, [None] * world, [None] * world
    dist.all_gather_object(gathered_z2, local_z2)
    dist.all_gather_object(gathered_z3, local_z3)
    dist.all_gather_object(gathered_ids, identity)
    dist.all_gather_object(gathered_valid, bool(valid))
    global_sum = local_sum.detach().clone()
    global_count = local_count.detach().clone()
    dist.all_reduce(global_sum)
    dist.all_reduce(global_count)
    local_sum.backward()
    if z2.grad is None or z3.grad is None:
        raise RuntimeError(f'align/{case} missing representation gradients')
    if not bool(torch.isfinite(z2.grad).all() and torch.isfinite(z3.grad).all()):
        raise FloatingPointError(f'align/{case} non-finite representation gradient')
    parameter_gradients = [parameter.grad for parameter in module.module.parameters()]
    if not all(gradient is not None and bool(torch.isfinite(gradient).all())
               for gradient in parameter_gradients):
        raise FloatingPointError(f'align/{case} non-finite projection gradient')
    local_record = {
        'rank': rank, 'identity': identity, 'valid': bool(valid),
        'local_count': float(local_count), 'z2_grad': z2.grad.detach().cpu(),
        'z3_grad': z3.grad.detach().cpu(),
        'p2_grad': module.module.p2.weight.grad.detach().cpu(),
        'p3_grad': module.module.p3.weight.grad.detach().cpu(),
    }
    records = [None] * world
    dist.all_gather_object(records, local_record)
    reference = None
    if rank == 0:
        # Keep the independent reference on CPU.  This avoids queuing a second
        # CUDA autograd graph while the distributed all-gather backward from
        # the observed path is still completing.
        ref_z2 = torch.cat(gathered_z2, dim=0).requires_grad_()
        ref_z3 = torch.cat(gathered_z3, dim=0).requires_grad_()
        flat_ids = list(gathered_ids)
        flat_valid = list(gathered_valid)
        ref_sum, ref_count = _alignment_reference(
            ref_z2, ref_z3, flat_ids, torch.tensor(flat_valid)
        )
        ref_sum.backward()
        observed_z2_grad = torch.cat([record['z2_grad'] for record in records], dim=0)
        observed_z3_grad = torch.cat([record['z3_grad'] for record in records], dim=0)
        expected_z2_grad = ref_z2.grad.detach().cpu()
        expected_z3_grad = ref_z3.grad.detach().cpu()
        z2_match = bool(torch.allclose(observed_z2_grad, expected_z2_grad, atol=1e-5, rtol=1e-5))
        z3_match = bool(torch.allclose(observed_z3_grad, expected_z3_grad, atol=1e-5, rtol=1e-5))
        ref_module = _AlignmentPair()
        ref_raw2 = torch.cat([
            torch.nn.functional.one_hot(torch.tensor([record['rank']]), 128).float()
            for record in records
        ], dim=0)
        ref_raw3 = torch.cat([
            torch.nn.functional.one_hot(torch.tensor([(record['rank'] + 7) % 128]), 128).float()
            for record in records
        ], dim=0)
        ref_proj2, ref_proj3 = ref_module(ref_raw2, ref_raw3)
        ref_proj_sum, _ = _alignment_reference(
            ref_proj2, ref_proj3, flat_ids, torch.tensor(flat_valid)
        )
        ref_proj_sum.backward()
        expected_p2 = ref_module.p2.weight.grad.detach().cpu() / world
        expected_p3 = ref_module.p3.weight.grad.detach().cpu() / world
        observed_p2 = records[0]['p2_grad']
        observed_p3 = records[0]['p3_grad']
        p2_match = bool(torch.allclose(observed_p2, expected_p2, atol=1e-5, rtol=1e-5))
        p3_match = bool(torch.allclose(observed_p3, expected_p3, atol=1e-5, rtol=1e-5))
        reference = {
            'reference_sum': float(ref_sum.detach().cpu()),
            'reference_count': float(ref_count.detach().cpu()),
            'loss_abs_error': abs(float(global_sum.cpu()) - float(ref_sum.detach().cpu())),
            'count_abs_error': abs(float(global_count.cpu()) - float(ref_count.detach().cpu())),
            'z2_gradient_match': z2_match, 'z3_gradient_match': z3_match,
            'p2_gradient_match': p2_match, 'p3_gradient_match': p3_match,
            'positive_count_A': (2 if case == 'multi_positive' else None),
        }
        if not (z2_match and z3_match and p2_match and p3_match
                and reference['loss_abs_error'] <= 1e-5
                and reference['count_abs_error'] <= 1e-5):
            raise AssertionError(f'ALIGN distributed reference mismatch: {reference}')
    dist.barrier()
    return {
        'global_loss': float(global_sum.cpu()),
        'global_anchor_count': float(global_count.cpu()),
        'finite_loss': True, 'backward_completed': True,
        'projection_gradients_finite': True,
        'reference': reference,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cohort-root', required=True)
    parser.add_argument('--cache-root', required=True)
    parser.add_argument('--dual-static-root', required=True)
    parser.add_argument('--report-json', required=True)
    parser.add_argument('--only', choices=('all', 'fgr', 'align'), default='all')
    args = parser.parse_args()
    require_tmux()
    rank, world = int(os.environ['RANK']), int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])
    if world != 3 or not torch.cuda.is_available():
        raise RuntimeError('S2 r3 DDP smoke requires exactly three CUDA ranks')
    torch.set_num_threads(1)
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)
    dist.init_process_group('nccl')
    source = None
    try:
        if args.only in ('all', 'fgr'):
            source, _ = open_source(
                args.cohort_root, args.cache_root,
                dual_static_root=args.dual_static_root, pretrain_target_root=None,
            )
        fgr_cases = {}
        align_cases = {}
        if args.only in ('all', 'fgr'):
            if rank == 0:
                print('R3_DDP_START=fgr', flush=True)
            fgr_cases = {
                'fgr_partial_invalid': _run_fgr_case(source, 'partial_invalid', rank, world, device),
                'fgr_all_zero': _run_fgr_case(source, 'all_zero', rank, world, device),
            }
        if args.only in ('all', 'align'):
            if rank == 0:
                print('R3_DDP_START=align', flush=True)
            align_cases = {
                case: _run_align_case(case, rank, world, device)
                for case in ('partial_invalid', 'multi_positive', 'all_zero', 'all_same_identity')
            }
        if rank == 0:
            report = {
                'status': 'PASS', 'world_size': world, 'backend': dist.get_backend(),
                'outer_test_accessed': False, 'optimizer_updates': 0,
                'selected': args.only, 'fgr_cases': fgr_cases, 'align_cases': align_cases,
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
