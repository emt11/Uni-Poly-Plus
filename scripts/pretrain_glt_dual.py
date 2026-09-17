#!/usr/bin/env python3
"""Explicit three-task training entry; never builds frozen layers."""
import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from src.dataset.glt_dual_pretrain import prepare_pretrain_sample, pretrain_collate
from src.modules.glt_dual_pretrain import DualPretrainer, global_objective, deployment_package
from src.training.glt_dual_runtime import (require_tmux, open_source, save_checkpoint, write_json,
                                         rng_state, restore_rng, scheduled_lr, move_labels,
                                         OrderedSampleStream, RankMicrobatchStream)
from src.utils import set_global_seed


def _module_grad_norms(model):
    """Gradient norms per module group, measured before clipping."""

    groups = {
        'o8_2d_encoder': model.encoder.o8,
        'glt_3d_encoder': model.encoder.glt,
        'length_head': model.length_head,
        'angle_head': model.angle_head,
        'fp_head': model.fp_head,
        'atom_head': model.atom_head,
    }
    seen, report = set(), {}
    for name, module in groups.items():
        params = [p for p in module.parameters() if p.requires_grad]
        seen.update(id(p) for p in params)
        flat = [p.grad.detach().reshape(-1) for p in params if p.grad is not None]
        report[name] = float(torch.cat(flat).norm()) if flat else 0.0
    fusion = [p.grad.detach().reshape(-1) for p in model.parameters()
              if id(p) not in seen and p.grad is not None]
    report['fusion'] = float(torch.cat(fusion).norm()) if fusion else 0.0
    return report


def _reference_steps(path):
    """Rank-0 step records from an earlier run log, keyed by step."""

    records = {}
    for line in Path(path).read_text(encoding='utf-8', errors='replace').splitlines():
        line = line.strip()
        if not line.startswith('{"step"'):
            continue
        try:
            row = json.loads(line)
        except Exception:
            continue
        if row.get('rank') == 0:
            records[int(row['step'])] = row
    return records


def _check_reference(reference, step, lr, target_counts, losses, rtol=1e-3):
    """Abort when the resumed trajectory no longer matches the original run."""

    expected = reference.get(step)
    if expected is None:
        return
    if abs(float(expected['lr']) - lr) > 1e-12:
        raise RuntimeError(f'resume LR differs from the original run at step {step}')
    if [float(x) for x in expected['target_counts']] != [float(x) for x in target_counts]:
        raise RuntimeError(f'resume target counts differ from the original run at step {step}')
    for name, value, original in zip(('chem', 'geometry', 'fingerprint'), losses,
                                     expected['losses']):
        if abs(value - float(original)) > rtol * max(1e-6, abs(float(original))):
            print(json.dumps({'reference_mismatch': name, 'step': step,
                              'observed': value, 'original': float(original)}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'cohort-root', 'cache-root', 'output'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--dual-static-root',
                        help='training-ready dual_static_v1 artifact')
    parser.add_argument('--pretrain-target-root',
                        help='pretrain_targets_v1 artifact')
    parser.add_argument('--resume')
    parser.add_argument('--prep-workers', type=int, default=0,
                        help='DataLoader workers per rank for input prefetch '
                             '(0 = inline preparation, the exact original path)')
    parser.add_argument('--diagnostics', action='store_true',
                        help='observation only: per-step component/gradient logging; '
                             'never writes deployment packages')
    parser.add_argument('--stop-after-step', type=int, default=0,
                        help='diagnostic stop position (requires --diagnostics)')
    parser.add_argument('--diagnostic-save-steps', type=int, nargs='*', default=[],
                        help='steps inside this run at which to save a resume state')
    parser.add_argument('--no-deploy', action='store_true',
                        help='diagnostic runs: never write a deployment package')
    parser.add_argument('--geometry-head-norm', action='store_true',
                        help='P2 single change: non-affine LayerNorm of the 3D bond state '
                             'feeding the geometry heads only')
    parser.add_argument('--reference-log',
                        help='original run log for the initial-10-step cross-check')
    parser.add_argument('--timing', action='store_true',
                        help='optional bounded phase timing; adds synchronization only for the timing run')
    args = parser.parse_args()
    require_tmux()
    config = json.loads(Path(args.config).read_text(encoding='utf-8'))
    if config.get('use_md200') is not False or config['amp_dtype'] not in ('fp32', 'bf16'):
        raise ValueError('dual route requires no MD200 and fp32/bf16 precision')
    if (min(config['microbatch'], config['global_batch'], config['max_optimizer_steps'], config['save_every']) <= 0
            or config['warmup_steps'] < 0
            or config['schedule_total_steps'] < max(config['max_optimizer_steps'], config['warmup_steps'])
            or len(config['loss_weights']) != 3):
        raise ValueError('invalid training budget, schedule or objective weights')
    rank, world = int(os.environ.get('RANK', 0)), int(os.environ.get('WORLD_SIZE', 1))
    device = torch.device('cuda', int(os.environ.get('LOCAL_RANK', 0))) if torch.cuda.is_available() else torch.device('cpu')
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    if world > 1:
        dist.init_process_group('nccl' if device.type == 'cuda' else 'gloo')
    output = Path(args.output)
    preparation_error = [None]
    if rank == 0:
        try:
            if output.exists() and not args.resume:
                raise FileExistsError('new training requires a new output directory')
            output.mkdir(parents=True, exist_ok=bool(args.resume))
        except OSError as exc:
            preparation_error[0] = str(exc)
    if world > 1:
        dist.broadcast_object_list(preparation_error, src=0)
    if preparation_error[0] is not None:
        raise FileExistsError(preparation_error[0])
    if args.pretrain_target_root and not args.dual_static_root:
        raise ValueError('--pretrain-target-root requires --dual-static-root')
    source, frame = open_source(
        args.cohort_root, args.cache_root,
        dual_static_root=args.dual_static_root,
        pretrain_target_root=args.pretrain_target_root,
    )
    try:
        micro, batch_size = config['microbatch'], config['global_batch']
        if batch_size % (micro * world):
            raise ValueError('microbatch * accumulation * world must equal global_batch')
        accumulation = batch_size // (micro * world)
        set_global_seed(config['seed'])
        base = DualPretrainer(config['fusion_mode'],
                              collect_diagnostics=bool(args.diagnostics),
                              geometry_head_norm=bool(args.geometry_head_norm or config.get('geometry_head_norm', False))).to(device)
        optimizer = torch.optim.AdamW(base.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
        module = DistributedDataParallel(base, device_ids=[device.index] if device.type == 'cuda' else None,
                                         find_unused_parameters=True) if world > 1 else base
        start = 0
        identity = dict(config=config, world_size=world, sample_count=len(source),
                        cohort_root=str(Path(args.cohort_root).resolve()),
                        cohort_hash=source.cohort['manifest_hash'],
                        cache_root=str(Path(args.cache_root).resolve()),
                        main_bundle_hash=source.bundle.bundle_hash,
                        dual_static_manifest_hash=(source.static_cache.manifest_hash
                                                   if source.static_cache is not None else None),
                        pretrain_target_manifest_hash=(source.target_cache.manifest_hash
                                                       if source.target_cache is not None else None))
        # Exact ordered identities, without adding a separate cache schema.
        ordered_keys = [key.hex() for key, _ in source.samples]
        resume_rng = None
        if args.resume:
            state = torch.load(args.resume, map_location='cpu', weights_only=False)
            if state['identity'] != identity or state['ordered_keys'] != ordered_keys:
                raise ValueError('resume data/config/world size differs')
            base.load_state_dict(state['model'], strict=True)
            optimizer.load_state_dict(state['optimizer'])
            start = state['step']
            if state['next_position'] != start * batch_size or state['scheduler']['step'] != start:
                raise ValueError('resume scheduler/data position mismatch')
            # Restore after constructing the optional DataLoader iterator below.
            # Worker-base-seed creation consumes the process Torch RNG; doing it
            # before restore keeps that setup draw outside the resumed stream.
            resume_rng = state['rng'][rank]
        else:
            # Identical initialization; independent dropout streams after DDP sync.
            set_global_seed(config['seed'] + rank)
        if not 0 <= start < config['max_optimizer_steps']:
            raise ValueError('resume step must precede configured training end')
        if any(int(p.stem.rsplit('_', 1)[-1]) > start for p in output.glob('deploy_*.pt')):
            raise FileExistsError('resume would overwrite later existing deployment checkpoints')
        if any(int(p.stem.rsplit('_', 1)[-1]) > start for p in output.glob('resume_*.pt')):
            raise FileExistsError('resume would overwrite later existing training checkpoints')
        stop = config['max_optimizer_steps']
        if args.stop_after_step:
            if not args.diagnostics:
                raise ValueError('--stop-after-step requires --diagnostics')
            if not start < int(args.stop_after_step) <= config['max_optimizer_steps']:
                raise ValueError('--stop-after-step must satisfy start < stop <= config.max_optimizer_steps')
            stop = int(args.stop_after_step)
        save_steps = sorted({int(value) for value in args.diagnostic_save_steps})
        for value in save_steps:
            if not start < value <= stop:
                raise ValueError('--diagnostic-save-steps must lie inside this run')
        if rank == 0:
            write_json(output / 'run.json', dict(identity=identity, command=sys.argv,
                accumulation=accumulation, diagnostics=bool(args.diagnostics),
                geometry_head_norm=bool(args.geometry_head_norm or config.get('geometry_head_norm', False)),
                stop_after_step=stop, diagnostic_save_steps=save_steps))
        stream = OrderedSampleStream(len(source), config['seed'])
        # Optional CPU prefetch.  The prepared items are identical to the
        # inline path (pure function of sample and absolute position); only
        # where they are built changes.  Workers are forked, so the read-only
        # frozen source (LMDB opened with lock=False) is inherited instead of
        # re-loaded, and workers never touch CUDA.
        prefetch = None
        if int(args.prep_workers) > 0:
            dataset = RankMicrobatchStream(
                source, seed=config['seed'], world=world, rank=rank,
                microbatch=micro, accumulation=accumulation,
                start_step=start, max_steps=stop,
                sigma=config['noise_sigma'], ratio=config['atom_mask_ratio'],
            )
            prefetch = iter(torch.utils.data.DataLoader(
                dataset, batch_size=None, num_workers=int(args.prep_workers),
                prefetch_factor=4, persistent_workers=False, pin_memory=False,
            ))
        if resume_rng is not None:
            restore_rng(resume_rng)
        reference = _reference_steps(args.reference_log) if args.reference_log else {}
        diagnostics_path = output / 'diagnostics_steps.jsonl'
        timing_path = output / 'timing_steps.jsonl'
        dense_lo = min(save_steps) - 60 if save_steps else None
        for step in range(start, stop):
            timing = {}
            step_started = time.perf_counter()
            prep_started = time.perf_counter()
            if prefetch is not None:
                prepared = [next(prefetch) for _ in range(accumulation)]
            else:
                prepared = []
                for offset in range(accumulation):
                    rows = []
                    for local in range(micro):
                        position = step * batch_size + offset * world * micro + rank * micro + local
                        index = stream.index_at(position)
                        rows.append(prepare_pretrain_sample(
                            *source[index], seed=config['seed'],
                            key=source.samples[index][0].hex(), position=position,
                            sigma=config['noise_sigma'], ratio=config['atom_mask_ratio'],
                            static=source.static_for(index),
                            target=source.target_for(index)))
                    prepared.append(pretrain_collate(rows))
            timing['preparation_seconds'] = time.perf_counter() - prep_started
            counts = torch.zeros(3, device=device)
            for data, labels in prepared:
                masked = torch.bincount(data.canonical_graph_index[labels['atom_mask']], minlength=data.graph_available.numel())
                centers = torch.bincount(data.bond_batch[data.bond_center], minlength=data.graph_available.numel())
                counts += torch.tensor([(masked > 0).sum(), ((centers > 0) & data.geometry_valid).sum(),
                                        data.graph_available.sum()], device=device)
            if world > 1:
                dist.all_reduce(counts)
            lr = scheduled_lr(step, **{key: config[key] for key in ('lr', 'warmup_steps', 'schedule_total_steps', 'end_lr')})
            for group in optimizer.param_groups:
                group['lr'] = lr
            optimizer.zero_grad(set_to_none=True)
            totals = torch.zeros(3, device=device)
            target_counts = torch.zeros(4, device=device)
            fallbacks, reasons = 0, {}
            h2d_seconds = forward_backward_seconds = optimizer_seconds = 0.0
            for offset, (data, labels) in enumerate(prepared):
                fallbacks += labels['fallback_count']
                for reason in labels['skip_reasons']:
                    if reason:
                        reasons[reason] = reasons.get(reason, 0) + 1
                sync = module.no_sync() if world > 1 and offset + 1 < accumulation else nullcontext()
                with sync:
                    if args.timing and device.type == 'cuda':
                        torch.cuda.synchronize(device)
                    h2d_started = time.perf_counter()
                    moved_data = data.to(device, non_blocking=True)
                    moved_labels = move_labels(labels, device)
                    if args.timing and device.type == 'cuda':
                        torch.cuda.synchronize(device)
                    h2d_seconds += time.perf_counter() - h2d_started
                    if args.timing and device.type == 'cuda':
                        torch.cuda.synchronize(device)
                    forward_backward_started = time.perf_counter()
                    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=config['amp_dtype'] == 'bf16'):
                        result = module(moved_data, moved_labels)
                        loss = global_objective(result['sums'], counts, world, config['loss_weights'])
                    if not torch.isfinite(loss):
                        raise FloatingPointError('nonfinite pretraining loss')
                    loss.backward()
                    if args.timing and device.type == 'cuda':
                        torch.cuda.synchronize(device)
                    forward_backward_seconds += time.perf_counter() - forward_backward_started
                totals += result['sums'].detach()
                target_counts += result['targets'].detach()
            if args.timing and device.type == 'cuda':
                torch.cuda.synchronize(device)
            optimizer_started = time.perf_counter()
            grad_total_preclip = float(torch.nn.utils.clip_grad_norm_(
                base.parameters(), 1., error_if_nonfinite=True))
            optimizer.step()
            if args.timing and device.type == 'cuda':
                torch.cuda.synchronize(device)
            optimizer_seconds = time.perf_counter() - optimizer_started
            if world > 1:
                dist.all_reduce(totals)
                dist.all_reduce(target_counts)
            record = dict(step=step + 1, rank=rank, lr=float(lr),
                losses=(totals / counts.clamp_min(1)).tolist(), valid_graphs=counts.tolist(),
                target_counts=target_counts.tolist(), local_fallbacks=fallbacks, local_skip_reasons=reasons)
            if args.timing:
                timing.update(h2d_seconds=h2d_seconds,
                              forward_backward_seconds=forward_backward_seconds,
                              optimizer_seconds=optimizer_seconds,
                              step_seconds=time.perf_counter() - step_started)
                record['timing'] = timing
            print(json.dumps(record), flush=True)
            if args.timing and rank == 0:
                with timing_path.open('a', encoding='utf-8') as handle:
                    handle.write(json.dumps(record, sort_keys=True) + '\n')
            if args.diagnostics and rank == 0 and base.last_diagnostics is not None:
                components = base.last_diagnostics['components']
                record = dict(step=step + 1, lr=float(lr), grad_total_preclip=grad_total_preclip,
                              grad_norms=_module_grad_norms(base),
                              components={k: components[k] for k in (
                                  'chem_sum', 'length_sum', 'angle_sum', 'geometry_sum',
                                  'geometry_valid_count', 'angle_valid_count', 'graphs_without_angle')},
                              global_losses=(totals / counts.clamp_min(1)).tolist(),
                              target_counts=target_counts.tolist(),
                              rank_scoped_statistics='representation/Gaussian/angle-head values below are rank0-local')
                dense = (step + 1) in save_steps or (dense_lo is not None and dense_lo <= step + 1 <= stop)
                if dense or (step + 1) % 20 == 0:
                    record['angle_head'] = base.last_diagnostics['angle_head']
                    record['representations'] = base.last_diagnostics['representations']
                    record['gaussian'] = base.last_diagnostics['gaussian']
                    record['predictions'] = base.last_diagnostics['predictions']
                with diagnostics_path.open('a', encoding='utf-8') as handle:
                    handle.write(json.dumps(record, sort_keys=True) + '\n')
                if step + 1 == start + 10 and reference:
                    _check_reference(reference, step + 1, float(lr),
                                     target_counts.tolist(),
                                     (totals / counts.clamp_min(1)).tolist())
            if args.diagnostics:
                if (step + 1) in save_steps:
                    states = [None] * world
                    if world > 1:
                        dist.all_gather_object(states, rng_state())
                    else:
                        states[0] = rng_state()
                    if rank == 0:
                        save_checkpoint(output / f'resume_{step + 1:05d}.pt', dict(identity=identity,
                            ordered_keys=ordered_keys, step=step + 1,
                            next_position=(step + 1) * batch_size, model=base.state_dict(),
                            optimizer=optimizer.state_dict(), rng=states,
                            scheduler=dict(step=step + 1, lr=float(lr))))
                        if not args.no_deploy:
                            save_checkpoint(output / f'deploy_{step + 1:05d}.pt',
                                            deployment_package(base, step + 1))
                if world > 1:
                    dist.barrier()
                continue
            if (step + 1) % config['save_every'] == 0 or step + 1 == config['max_optimizer_steps']:
                states = [None] * world
                if world > 1:
                    dist.all_gather_object(states, rng_state())
                else:
                    states[0] = rng_state()
                if rank == 0:
                    save_checkpoint(output / f'resume_{step + 1:05d}.pt', dict(identity=identity,
                        ordered_keys=ordered_keys, step=step + 1, next_position=(step + 1) * batch_size,
                        model=base.state_dict(), optimizer=optimizer.state_dict(), rng=states,
                        scheduler=dict(step=step + 1, lr=float(lr))))
                    save_checkpoint(output / f'deploy_{step + 1:05d}.pt', deployment_package(base, step + 1))
                if world > 1:
                    dist.barrier()
    finally:
        source.close()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
