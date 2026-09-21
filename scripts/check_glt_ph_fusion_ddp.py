#!/usr/bin/env python3
"""DDP empty-geometry collective/backward check (0 optimizer updates).

``--mode partial`` leaves only some ranks without usable geometry, ``--mode
all-empty`` leaves every rank without it, and ``--mode control`` runs the same
path with real geometry on every rank as the reference.

The P_train audit found no geometry-invalid structure among its 256 sampled
rows, so the empty case applies the declared geometry-invalid branch to whole
samples *before* collation: the 3D bond/line tables, the masked-3D targets and
the spatial relations become empty while the O8/2D fields and the atom table
stay exactly as they are -- which is what the frozen complete-Trimer contract
yields when ``trimer_geometry_valid`` is false.  Every rank still runs the full
forward/backward and the 2D task keeps contributing; no optimizer step is taken
and the process group is always torn down.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from scripts.pretrain_glt_ph_fusion import build_model, fusion_optimizer, window_counts
from src.dataset.glt_galformer_ph import PHBettiReader
from src.dataset.glt_ph_fusion_inputs import (fusion_collate, load_ph_stats,
                                              prepare_fusion_sample)
from src.modules.glt_dual_pretrain import global_objective
from src.modules.glt_ph_fusion_candidates import FusionPretrainer
from src.training.glt_dual_runtime import (IndexedFrozenDualSource, load_sample_index_artifact,
                                           open_source, require_tmux, write_json)
from src.utils import set_global_seed


def log_stage(rank, stage):
    """Progress marker: a hang must be attributable to one stage."""
    print(f'[rank {rank}] stage={stage}', flush=True)


class _Stream:
    """Deterministic first-N stream; the check never depends on a shuffle."""

    def __init__(self, size, seed):
        self.size, self.seed = int(size), int(seed)

    def index_at(self, position):
        return int(position) % self.size


def empty_sample(data, labels):
    """Apply the declared geometry-invalid branch to one prepared sample."""
    data.bond_z_a = data.bond_z_a.new_zeros(0)
    data.bond_z_b = data.bond_z_b.new_zeros(0)
    data.bond_type = data.bond_type.new_zeros(0)
    data.bond_distance = data.bond_distance.new_zeros(0)
    data.bond_center = data.bond_center.new_zeros(0, dtype=torch.bool)
    data.bond_atom_index = data.bond_atom_index.new_zeros((2, 0))
    data.spatial_edge_index = data.spatial_edge_index.new_zeros((2, 0))
    data.spatial_distance = data.spatial_distance.new_zeros(0)
    data.spatial_bonded = data.spatial_bonded.new_zeros(0, dtype=torch.bool)
    data.spatial_scale = data.spatial_scale.new_zeros(0)
    for name in ('line_source', 'line_target', 'line_path_group'):
        setattr(data, name, getattr(data, name).new_zeros(0))
    data.line_path = data.line_path.new_zeros((0, 3))
    data.line_angle = data.line_angle.new_zeros((0, 2))
    data.line_mask = data.line_mask.new_zeros((0, 2), dtype=torch.bool)
    data.line_is_self = data.line_is_self.new_zeros(0, dtype=torch.bool)
    for name in ('mask3d_rows', 'mask3d_policy', 'mask3d_donor_atoms'):
        setattr(data, name, getattr(data, name).new_zeros(0))
    labels['label_3d'] = labels['label_3d'].new_zeros(0)
    data.geometry_valid = False
    data.bond_count = torch.tensor(0)
    return data, labels


def build_batch(source, stream, reader, *, seed, arm, const_profile, samples, rank,
                world, empties):
    rows = []
    for local in range(samples):
        position = local * world + rank
        index = stream.index_at(position)
        key, _ = source.samples[index]
        data, labels = prepare_fusion_sample(
            *source[index], static=source.static_for(index), seed=seed, key=key.hex(),
            position=position, ph_reader=(reader if arm == 'PH' else None), arm=arm,
            const_profile=const_profile)
        if empties:
            data, labels = empty_sample(data, labels)
        rows.append((data, labels))
    return fusion_collate(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--arm', default='PH')
    parser.add_argument('--mode', required=True, choices=('control', 'partial', 'all-empty'))
    parser.add_argument('--cohort-root', required=True)
    parser.add_argument('--cache-root', required=True)
    parser.add_argument('--dual-static-root', required=True)
    parser.add_argument('--sample-index-artifact', required=True)
    parser.add_argument('--ph-sidecar-root', required=True)
    parser.add_argument('--profile-stats', required=True)
    parser.add_argument('--samples', type=int, default=4)
    parser.add_argument('--output', required=True)
    parser.add_argument('--no-ddp', action='store_true', help='diagnostic only')
    args = parser.parse_args()
    require_tmux()
    rank = int(os.environ.get('RANK', 0))
    world = int(os.environ.get('WORLD_SIZE', 1))
    device = torch.device('cuda', int(os.environ.get('LOCAL_RANK', 0)))
    torch.cuda.set_device(device)
    if not args.no_ddp and world > 1:
        dist.init_process_group('nccl')
    config = json.loads(Path(args.config).read_text(encoding='utf-8'))
    source = None
    try:
        set_global_seed(int(config['seed']) + rank)
        namespace = argparse.Namespace(profile_stats=args.profile_stats,
                                       allow_smoke_stats=True,
                                       ph_sidecar_root=args.ph_sidecar_root)
        model, _ = build_model(args.arm, namespace, config)
        model = model.to(device)
        trainer = FusionPretrainer(model, objective='R0').to(device)
        optimizer, _, _ = fusion_optimizer(model, config)
        module = (trainer if args.no_ddp or world == 1 else
                  DistributedDataParallel(trainer, device_ids=[device.index],
                                          find_unused_parameters=True))
        source, _ = open_source(args.cohort_root, args.cache_root,
                                dual_static_root=args.dual_static_root)
        reader = PHBettiReader(args.ph_sidecar_root)
        index = load_sample_index_artifact(args.sample_index_artifact, 'train')
        source = IndexedFrozenDualSource(source, index['indices'])
        log_stage(rank, 'source-open')
        const_profile = torch.as_tensor(load_ph_stats(args.profile_stats)['mean'])
        # partial: rank 0 alone has no usable geometry; all-empty: every rank
        empties = (args.mode == 'all-empty'
                   or (args.mode == 'partial' and rank == 0))
        data, labels = build_batch(
            source, _Stream(len(source), int(config['seed'])), reader,
            seed=int(config['seed']), arm=args.arm, const_profile=const_profile,
            samples=args.samples, rank=rank, world=world, empties=empties)
        log_stage(rank, 'batch-built')
        counts = window_counts([(data, labels)], device, 'R0')
        log_stage(rank, 'window-counts')
        moved = data.to(device)
        moved_labels = {key: value.to(device) if torch.is_tensor(value) else value
                        for key, value in labels.items()}
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.bfloat16,
                            enabled=config['amp_dtype'] == 'bf16'):
            result = module(moved, moved_labels)
            loss = global_objective(result['sums'], counts, world,
                                    [float(value) for value in config['loss_weights']])
        log_stage(rank, 'forward')
        finite_loss = bool(torch.isfinite(loss))
        loss.backward()
        log_stage(rank, 'backward')
        gradients = [parameter.grad.detach() for parameter in trainer.parameters()
                     if parameter.grad is not None]
        finite_grad = all(bool(torch.isfinite(value).all()) for value in gradients)
        nonzero_grad = any(float(value.abs().sum()) > 0 for value in gradients)
        optimizer.zero_grad(set_to_none=True)
        record = {'rank': rank, 'mode': args.mode, 'emptied_rank': bool(empties),
                  'finite_loss': finite_loss, 'finite_grad': finite_grad,
                  'nonzero_grad': nonzero_grad,
                  'two_d_targets': float(result['counts'][0]),
                  'three_d_targets': float(result['counts'][1]),
                  'graph_count': int(moved.graph_available.numel()),
                  'spatial_edge_count': int(moved.spatial_edge_index.size(1))}
        if dist.is_initialized():
            gathered = [None] * world
            dist.all_gather_object(gathered, record)
            log_stage(rank, 'gathered')
            if rank == 0:
                write_json(args.output, {
                    'check': 'ddp geometry-empty backward', 'mode': args.mode,
                    'arm': args.arm, 'world_size': world, 'optimizer_updates': 0,
                    'loss_finite_all_ranks': all(item['finite_loss'] for item in gathered),
                    'grad_finite_all_ranks': all(item['finite_grad'] for item in gathered),
                    'grad_nonzero_all_ranks': all(item['nonzero_grad'] for item in gathered),
                    'two_d_continued': all(item['two_d_targets'] > 0 for item in gathered),
                    'ranks': gathered,
                    'note': ('empty geometry is produced through the declared invalid '
                             'branch before collation; the 2D task still contributes')})
                print(json.dumps(gathered, indent=1), flush=True)
        elif rank == 0:
            record['optimizer_updates'] = 0
            write_json(args.output, {'check': 'ddp geometry-empty backward',
                                     'mode': args.mode, 'arm': args.arm,
                                     'world_size': world, 'ranks': [record]})
            print(json.dumps(record, indent=1), flush=True)
        log_stage(rank, 'done')
    except Exception as error:
        print(f'[rank {rank}] FAILED {type(error).__name__}: {error}', flush=True)
        traceback.print_exc()
        raise
    finally:
        if source is not None:
            try:
                source.close()
            except Exception:
                pass
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
