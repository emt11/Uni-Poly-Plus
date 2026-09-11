#!/usr/bin/env python3
"""Explicit three-task training entry; never builds frozen layers."""
import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from src.dataset.glt_dual_pretrain import prepare_pretrain_sample, pretrain_collate
from src.modules.glt_dual_pretrain import DualPretrainer, global_objective, deployment_package
from src.training.glt_dual_runtime import (require_tmux, open_source, save_checkpoint, write_json,
                                         rng_state, restore_rng, scheduled_lr, move_labels, OrderedSampleStream)
from src.utils import set_global_seed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'samples-csv', 'topology-root', 'trimer-root', 'output'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--resume')
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
    source, frame = open_source(args.samples_csv, args.topology_root, args.trimer_root)
    try:
        micro, batch_size = config['microbatch'], config['global_batch']
        if batch_size % (micro * world):
            raise ValueError('microbatch * accumulation * world must equal global_batch')
        accumulation = batch_size // (micro * world)
        set_global_seed(config['seed'])
        base = DualPretrainer(config['fusion_mode']).to(device)
        optimizer = torch.optim.AdamW(base.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
        module = DistributedDataParallel(base, device_ids=[device.index] if device.type == 'cuda' else None,
                                         find_unused_parameters=True) if world > 1 else base
        start = 0
        identity = dict(config=config, world_size=world, sample_count=len(source),
                        samples_csv=str(Path(args.samples_csv).resolve()),
                        topology_root=str(Path(args.topology_root).resolve()),
                        trimer_root=str(Path(args.trimer_root).resolve()))
        # Exact ordered identities, without adding a separate cache schema.
        ordered_keys = [key.hex() for key, _ in source.samples]
        if args.resume:
            state = torch.load(args.resume, map_location='cpu', weights_only=False)
            if state['identity'] != identity or state['ordered_keys'] != ordered_keys:
                raise ValueError('resume data/config/world size differs')
            base.load_state_dict(state['model'], strict=True)
            optimizer.load_state_dict(state['optimizer'])
            start = state['step']
            if state['next_position'] != start * batch_size or state['scheduler']['step'] != start:
                raise ValueError('resume scheduler/data position mismatch')
            restore_rng(state['rng'][rank])
        else:
            # Identical initialization; independent dropout streams after DDP sync.
            set_global_seed(config['seed'] + rank)
        if not 0 <= start < config['max_optimizer_steps']:
            raise ValueError('resume step must precede configured training end')
        if any(int(p.stem.rsplit('_', 1)[-1]) > start for p in output.glob('deploy_*.pt')):
            raise FileExistsError('resume would overwrite later existing deployment checkpoints')
        if any(int(p.stem.rsplit('_', 1)[-1]) > start for p in output.glob('resume_*.pt')):
            raise FileExistsError('resume would overwrite later existing training checkpoints')
        if rank == 0:
            write_json(output / 'run.json', dict(identity=identity, command=sys.argv, accumulation=accumulation))
        stream = OrderedSampleStream(len(source), config['seed'])
        for step in range(start, config['max_optimizer_steps']):
            prepared = []
            for offset in range(accumulation):
                rows = []
                for local in range(micro):
                    position = step * batch_size + offset * world * micro + rank * micro + local
                    index = stream.index_at(position)
                    rows.append(prepare_pretrain_sample(*source[index], seed=config['seed'],
                        key=source.samples[index][0].hex(), position=position,
                        sigma=config['noise_sigma'], ratio=config['atom_mask_ratio']))
                prepared.append(pretrain_collate(rows))
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
            for offset, (data, labels) in enumerate(prepared):
                fallbacks += labels['fallback_count']
                for reason in labels['skip_reasons']:
                    if reason:
                        reasons[reason] = reasons.get(reason, 0) + 1
                sync = module.no_sync() if world > 1 and offset + 1 < accumulation else nullcontext()
                with sync:
                    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=config['amp_dtype'] == 'bf16'):
                        result = module(data.to(device), move_labels(labels, device))
                        loss = global_objective(result['sums'], counts, world, config['loss_weights'])
                    if not torch.isfinite(loss):
                        raise FloatingPointError('nonfinite pretraining loss')
                    loss.backward()
                totals += result['sums'].detach()
                target_counts += result['targets'].detach()
            torch.nn.utils.clip_grad_norm_(base.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            if world > 1:
                dist.all_reduce(totals)
                dist.all_reduce(target_counts)
            print(json.dumps(dict(step=step + 1, rank=rank, lr=float(lr),
                losses=(totals / counts.clamp_min(1)).tolist(), valid_graphs=counts.tolist(),
                target_counts=target_counts.tolist(), local_fallbacks=fallbacks, local_skip_reasons=reasons)), flush=True)
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
