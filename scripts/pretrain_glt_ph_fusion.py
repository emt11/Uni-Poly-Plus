#!/usr/bin/env python3
"""GLT-PH end-to-end pretraining entry (family B plus its two references).

``--arm`` selects one of R0 / R2D / CONST / STAT / PH.  R0 and R2D are the
references (new-recipe original architecture, and the 2D-only trunk); CONST /
STAT / PH are the three arms of family B, which differ *only* in the declared
conditional input that drives the spatial scale router.

Everything else is the existing, already-validated recipe: the same O8/GLT
mathematics, the same 40% 80/10/10 masking, the same window-normalised
objective with DDP-corrected counts, the same resume/save discipline, and the
r6 failure discipline (results first, FAIL ``runtime.json``, non-zero exit).
"""
import argparse
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from src.dataset.glt_galformer_ph import PHBettiReader
from src.dataset.glt_ph_fusion_inputs import (ARMS, SPATIAL_RADII, fusion_collate,
                                              load_ph_stats, prepare_fusion_sample)
from src.modules.glt_dual_pretrain import alignment_local_anchor_count, global_objective
from src.modules.glt_galformer_ph import GLTGalPH
from src.modules.glt_ph_fusion_candidates import (FusionPretrainer, GLTFusionB,
                                                  R2DModel, fusion_deployment_package)
from src.training.glt_dual_runtime import (IndexedFrozenDualSource, OrderedSampleStream,
                                           apply_common_initialization,
                                           load_sample_index_artifact, move_labels,
                                           open_source, ordered_text_hash, require_tmux,
                                           restore_rng, rng_state, save_checkpoint,
                                           scheduled_lr, sha256_file, write_json)
from src.utils import set_global_seed

FAMILY_ARMS = {'reference': ('R0', 'R2D'), 'B': ('CONST', 'STAT', 'PH')}
ARM_OBJECTIVE = {'R0': 'R0', 'R2D': 'R2D', 'CONST': 'R0', 'STAT': 'R0', 'PH': 'R0'}
NO_DECAY_PREFIXES = ('conditional.', 'spatial.')
SINGLETON_PREFIXES = ('mask_2d_embedding', 'mask_3d_embedding', 'cls_2d', 'cls_3d',
                      'virtual_to_real_bias', 'real_to_virtual_bias', 'virtual_self_bias')
PACKED_FIELDS = ('atom_batch', 'bond_atom_index', 'spatial_edge_index', 'spatial_distance',
                 'spatial_bonded', 'spatial_scale', 'profile_input', 'profile_valid')
MASK_GROUPS = ('mask2d_rows', 'mask2d_policy', 'mask2d_donor_atoms',
               'mask3d_rows', 'mask3d_policy', 'mask3d_donor_atoms')
RUN_CONTEXT = {}


def arm_family(arm):
    for family, members in FAMILY_ARMS.items():
        if arm in members:
            return family
    raise ValueError(f'arm must be one of {tuple(ARM_OBJECTIVE)}')


def decay_split(model):
    """Matrix parameters decay; biases, normalisations, PH and routers do not."""
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        matrix = parameter.dim() >= 2 and not name.startswith(NO_DECAY_PREFIXES)
        (decay if matrix else no_decay).append((name, parameter))
    return decay, no_decay


def fusion_optimizer(model, config):
    decay, no_decay = decay_split(model)
    groups = []
    if decay:
        groups.append(dict(params=[item[1] for item in decay],
                           weight_decay=float(config['weight_decay']), name='matrix_decay'))
    if no_decay:
        groups.append(dict(params=[item[1] for item in no_decay], weight_decay=0.0,
                           name='bias_norm_ph_router'))
    if not groups:
        raise RuntimeError('no trainable parameter for the pretraining optimizer')
    return torch.optim.AdamW(groups, lr=float(config['lr'])), decay, no_decay


def optimizer_group_record(optimizer, decay, no_decay, **extra):
    return dict(optimizer=type(optimizer).__name__,
                groups=[dict(name=group.get('name'), weight_decay=float(group['weight_decay']),
                             learning_rate=float(group['lr']),
                             parameter_count=len(group['params'])) for group in
                        optimizer.param_groups],
                decay_parameters=[name for name, _ in decay],
                no_decay_parameters=[name for name, _ in no_decay], **extra)


def build_model(arm, args, config):
    """One model instance per arm; new modules under the family's isolated seed."""
    if arm in ARMS:
        if not args.profile_stats:
            raise ValueError('the family-B arms require --profile-stats')
        stats = load_ph_stats(args.profile_stats)
        if (stats['scope'] and stats['scope'].startswith('SMOKE_ONLY')
                and not args.allow_smoke_stats):
            raise ValueError('SMOKE_ONLY statistics require an explicit --allow-smoke-stats')
        model = GLTFusionB(arm=arm, profile_stats=stats,
                           dropout=float(config.get('dropout', 0.1)))
        stats_record = {'path': str(Path(args.profile_stats).resolve()),
                        'sha256': sha256_file(args.profile_stats),
                        'scope': stats['scope'], 'source': stats['source']}
        return model, stats_record
    if arm == 'R2D':
        return R2DModel(dropout=float(config.get('dropout', 0.1))), None
    return GLTGalPH(summary_mode='cls', ph_mode=None,
                    dropout=float(config.get('dropout', 0.1))), None


def common_init_exclude(model, state):
    """Exclude every artifact tensor the model does not own (R2D drops the 3D side)."""
    current = model.state_dict()
    return tuple(name for name in state
                 if name not in current or tuple(current[name].shape) != tuple(state[name].shape))


def prepare_window(source, stream, reader, *, step, micro, accumulation, world, rank,
                   seed, arm, const_profile, radii):
    """Build one optimizer window.

    Family-B arms carry the declared spatial and conditional fields; the R0/R2D
    references use the original data path untouched, so a reference batch is
    exactly the historical ``galformer_collate`` output.
    """
    from src.dataset.glt_galformer_ph import galformer_collate, prepare_galformer_sample

    fused = arm in ARMS
    prepared = []
    positions = []
    batch_size = micro * accumulation * world
    for offset in range(accumulation):
        rows = []
        for local in range(micro):
            position = step * batch_size + offset * world * micro + rank * micro + local
            index = stream.index_at(position)
            key, _ = source.samples[index]
            if fused:
                rows.append(prepare_fusion_sample(
                    *source[index], static=source.static_for(index), seed=seed,
                    key=key.hex(), position=position,
                    ph_reader=(reader if arm == 'PH' else None), arm=arm,
                    const_profile=const_profile, radii=radii))
            else:
                rows.append(prepare_galformer_sample(
                    *source[index], static=source.static_for(index), seed=seed,
                    key=key.hex(), position=position, ph_reader=None))
            positions.append(position)
        prepared.append(fusion_collate(rows) if fused else galformer_collate(rows))
    return prepared, positions


def window_counts(prepared, device, objective):
    counts = torch.zeros(4, device=device)
    for data, labels in prepared:
        count_2d = torch.tensor(float(labels['label_2d'].numel()), device=device)
        if objective == 'R0':
            count_3d = torch.tensor(float(labels['label_3d'].numel()), device=device)
            valid = (data.graph_available.bool() & data.geometry_valid.bool())
            count_cl = alignment_local_anchor_count(labels['identity'], valid, device=device)
        else:
            count_3d = torch.zeros((), device=device)
            count_cl = torch.zeros((), dtype=torch.long, device=device)
        counts += torch.stack([count_2d, count_3d, count_cl.float(),
                               torch.zeros((), device=device)])
    if dist.is_initialized():
        dist.all_reduce(counts)
    return counts


def stream_digest(prepared):
    digest = hashlib.sha256()
    for data, labels in prepared:
        for name in MASK_GROUPS:
            value = getattr(data, name).long().detach().cpu().contiguous()
            digest.update(name.encode('ascii') + str(tuple(value.shape)).encode('ascii'))
            digest.update(value.numpy().tobytes())
        for name in ('label_2d', 'label_3d'):
            value = labels[name].long().detach().cpu().contiguous()
            digest.update(name.encode('ascii') + value.numpy().tobytes())
        for name in PACKED_FIELDS:
            value = getattr(data, name, None)
            if torch.is_tensor(value):
                value = value.detach().cpu().contiguous()
                digest.update(name.encode('ascii'))
                digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def parameter_deltas(model, before):
    return {name: float((parameter.detach().float() - before[name]).norm())
            for name, parameter in model.named_parameters() if name in before}


def monitor_row(model, trainer, data, labels, step, lr, grad_norm, losses, before=None):
    """Routing, message norms, conditional separation and update sizes."""
    row = {'step': int(step), 'lr': float(lr), 'grad_norm_preclip': float(grad_norm),
           'losses': [float(value) for value in losses], 'all_finite': True}
    if isinstance(model, GLTFusionB):
        with torch.no_grad():
            conditional = model.conditional_input(data)
            row['conditional'] = {
                'norm_mean': float(conditional.norm(dim=-1).mean()),
                'spread': float((conditional.max(0).values - conditional.min(0).values).max()),
                'input_spread': float((data.profile_input.max(0).values
                                       - data.profile_input.min(0).values).max()),
                'source': sorted(set(data.profile_source)),
                'valid_fraction': float(data.profile_valid.float().mean())}
        out = model(data)
        row['spatial_metrics'] = out['spatial_metrics']
        row['summary_norm'] = float(out['g3'].detach().norm(dim=-1).mean())
    if before is not None:
        row['parameter_update'] = parameter_deltas(model, before)
    return row


def write_failure(context, error):
    """FAIL runtime.json: what ran, which stage died, what was kept."""
    output = context.get('output')
    if output is None or int(os.environ.get('RANK', 0)) != 0:
        return
    try:
        write_json(Path(output) / 'runtime.json', dict(
            status='FAIL', stage=context.get('stage', 'unknown'),
            error=f'{type(error).__name__}: {error}', command=sys.argv,
            arm=context.get('arm'), step=context.get('step'),
            raw_seconds=context.get('raw_seconds'),
            checkpoints_written=sorted(path.name for path in Path(output).glob('*.pt')),
            records_written=sorted(path.name for path in Path(output).glob('records_rank*.jsonl')),
            metrics_written=bool(list(Path(output).glob('records_rank*.jsonl'))),
            checkpoint_kept=bool(list(Path(output).glob('*.pt')))))
    except Exception as nested:      # never mask the original failure
        print(f'RETENTION_WRITE_FAILURE_FAILED {nested}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--arm', required=True, choices=tuple(ARM_OBJECTIVE))
    parser.add_argument('--mode', default='smoke', choices=('smoke', 'pretrain'))
    parser.add_argument('--cohort-root', required=True)
    parser.add_argument('--cache-root', required=True)
    parser.add_argument('--dual-static-root', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--sample-index-artifact')
    parser.add_argument('--sample-index-split', default='train')
    parser.add_argument('--common-init-artifact')
    parser.add_argument('--profile-stats', help='P_train conditional-input statistics [3,32]')
    parser.add_argument('--allow-smoke-stats', action='store_true',
                        help='explicitly accept SMOKE_ONLY standardization statistics')
    parser.add_argument('--ph-sidecar-root', help='required by the PH arm')
    parser.add_argument('--radii', type=float, nargs='*', default=list(SPATIAL_RADII))
    parser.add_argument('--updates', type=int, default=0,
                        help='bounded run: this many optimizer updates from the start')
    parser.add_argument('--save-steps', type=int, nargs='*', default=[])
    parser.add_argument('--deploy-steps', type=int, nargs='*', default=[])
    parser.add_argument('--resume')
    parser.add_argument('--prep-workers', type=int, default=0)
    parser.add_argument('--monitor', help='JSONL monitor path')
    parser.add_argument('--monitor-steps', type=int, nargs='*', default=[1, 2, 4])
    parser.add_argument('--log-every', type=int, default=1)
    args = parser.parse_args()
    require_tmux()
    config = json.loads(Path(args.config).read_text(encoding='utf-8'))
    family = arm_family(args.arm)
    if str(config.get('family', family)) != family:
        raise ValueError(f'config family {config.get("family")} does not hold arm {args.arm}')
    objective = ARM_OBJECTIVE[args.arm]
    if args.arm == 'PH' and not args.ph_sidecar_root:
        raise ValueError('the PH arm requires --ph-sidecar-root')
    if int(config['global_batch']) % (int(config['microbatch'])
                                      * max(1, int(os.environ.get('WORLD_SIZE', 1)))):
        raise ValueError('microbatch * accumulation * world must equal global_batch')
    rank = int(os.environ.get('RANK', 0))
    world = int(os.environ.get('WORLD_SIZE', 1))
    device = (torch.device('cuda', int(os.environ.get('LOCAL_RANK', 0)))
              if torch.cuda.is_available() else torch.device('cpu'))
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    if world > 1:
        dist.init_process_group('nccl' if device.type == 'cuda' else 'gloo')
    output = Path(args.output)
    RUN_CONTEXT.update(output=output, arm=args.arm, stage='setup')
    if rank == 0:
        if output.exists() and not args.resume:
            raise FileExistsError('a new training run requires a new output directory')
        output.mkdir(parents=True, exist_ok=bool(args.resume))
    set_global_seed(int(config['seed']))
    model, stats_record = build_model(args.arm, args, config)
    trainer = FusionPretrainer(model, objective=objective,
                               temperature=float(config.get('contrastive_temperature', 0.1))
                               ).to(device)
    common_path = args.common_init_artifact or config.get('common_init_artifact')
    common_record = None
    if common_path:
        artifact = torch.load(common_path, map_location='cpu', weights_only=False)
        exclude = common_init_exclude(model, artifact['common_state_dict'])
        common_record = apply_common_initialization(model, common_path, exclude=exclude)
        common_record['excluded'] = sorted(exclude)
    optimizer, decay, no_decay = fusion_optimizer(model, config)
    records_path = output / f'records_rank{rank}.jsonl'
    monitor_path = Path(args.monitor) if args.monitor else None
    monitor_steps = {int(value) for value in args.monitor_steps} if monitor_path else set()
    reader = PHBettiReader(args.ph_sidecar_root) if args.arm == 'PH' else None
    const_profile = (torch.as_tensor(load_ph_stats(args.profile_stats)['mean'])
                     if args.arm in ARMS else None)
    source, _ = open_source(args.cohort_root, args.cache_root,
                            dual_static_root=args.dual_static_root)
    try:
        sample_index = load_sample_index_artifact(args.sample_index_artifact,
                                                  args.sample_index_split)
        if sample_index is not None:
            if len(source) != int(sample_index['payload'].get('pi1m_record_count', len(source))):
                raise ValueError('sample index base cohort count mismatch')
            source = IndexedFrozenDualSource(source, sample_index['indices'])
        if args.arm == 'PH' and not len(reader) == len(source):
            raise ValueError('the PH sidecar must cover the whole pretraining manifest')
        micro = int(config['microbatch'])
        batch_size = int(config['global_batch'])
        accumulation = batch_size // (micro * world)
        start = 0
        identity = dict(
            plan='GLT-PH-END2END-20260920-01', arm=args.arm, family=family,
            mode=args.mode, objective=objective, config=config, world_size=world,
            sample_count=len(source),
            cohort_root=str(Path(args.cohort_root).resolve()),
            cohort_hash=source.cohort['manifest_hash'],
            cache_root=str(Path(args.cache_root).resolve()),
            main_bundle_hash=source.bundle.bundle_hash,
            dual_static_manifest_hash=(source.static_cache.manifest_hash
                                       if source.static_cache is not None else None),
            sample_index_artifact_sha256=(sample_index['sha256'] if sample_index else None),
            sample_index_split=(sample_index['split'] if sample_index else None),
            common_init_record=common_record, profile_stats=stats_record,
            spatial_radii=[float(value) for value in args.radii],
            optimizer='adamw', weight_decay=float(config['weight_decay']),
            decay_parameters=sorted(name for name, _ in decay),
            no_decay_parameters=sorted(name for name, _ in no_decay),
            grad_clip=float(config.get('grad_clip', 1.0)),
            mask_candidate_rate=0.40, mask_policy='80_10_10',
            ph_sidecar_root=(str(Path(args.ph_sidecar_root).resolve())
                             if args.ph_sidecar_root else None))
        ordered_keys_hash = ordered_text_hash(key.hex() for key, _ in source.samples)
        resume_rng = None
        if args.resume:
            state = torch.load(args.resume, map_location='cpu', weights_only=False)
            if state['identity'] != identity or state['ordered_keys_hash'] != ordered_keys_hash:
                raise ValueError('resume data/config/identity differs')
            trainer.load_state_dict(state['model'], strict=True)
            optimizer.load_state_dict(state['optimizer'])
            start = int(state['step'])
            if state['next_position'] != start * batch_size or state['scheduler']['step'] != start:
                raise ValueError('resume scheduler/data position mismatch')
            resume_rng = state['rng'][rank]
        else:
            set_global_seed(int(config['seed']) + rank)
        stop = (start + int(args.updates) if args.updates
                else int(config['max_optimizer_steps']))
        if stop > int(config['max_optimizer_steps']):
            raise ValueError('--updates exceeds the configured training end')
        if not 0 <= start < stop:
            raise ValueError('start position must precede the training end')
        deployment_step = int(config.get('deployment_step', config['max_optimizer_steps']))
        save_steps = sorted({int(value) for value in args.save_steps})
        deploy_steps = sorted({int(value) for value in args.deploy_steps})
        for value in save_steps + deploy_steps:
            if not 0 < value <= stop:
                raise ValueError('--save-steps/--deploy-steps must lie inside this run')
        if any(int(path.stem.rsplit('_', 1)[-1]) > start for path in output.glob('resume_*.pt')):
            raise FileExistsError('resume would overwrite a later training checkpoint')
        if any(int(path.stem.rsplit('_', 1)[-1]) > start for path in output.glob('deploy_*.pt')):
            raise FileExistsError('resume would overwrite a later deployment')
        if rank == 0:
            write_json(output / 'run.json', dict(
                identity=identity, command=sys.argv, accumulation=accumulation,
                start_step=start, stop_step=stop, save_steps=save_steps,
                deploy_steps=deploy_steps, deployment_step=deployment_step,
                arm=args.arm, family=family, mode=args.mode,
                parameter_counts={'total': int(sum(p.numel() for p in model.parameters())),
                                  'trainable': int(sum(p.numel() for p in model.parameters()
                                                       if p.requires_grad))}))
            write_json(output / 'optimizer_groups.json', optimizer_group_record(
                optimizer, decay, no_decay, arm=args.arm, family=family,
                stats_scope=(stats_record or {}).get('scope')))
        module = (DistributedDataParallel(
            trainer, device_ids=[device.index] if device.type == 'cuda' else None,
            find_unused_parameters=True) if world > 1 else trainer)
        if rank == 0:
            write_json(output / 'runtime.json', dict(
                status='RUNNING', command=sys.argv, identity=identity, arm=args.arm,
                world_size=world, device=str(device), start_step=start, stop_step=stop,
                started_at_monotonic=time.perf_counter()))
        stream = OrderedSampleStream(len(source), int(config['seed']))
        if resume_rng is not None:
            restore_rng(resume_rng)
        weights = [float(value) for value in config['loss_weights']]
        RUN_CONTEXT['stage'] = 'train'
        for step in range(start, stop):
            step_started = time.perf_counter()
            prepared, positions = prepare_window(
                source, stream, reader, step=step, micro=micro, accumulation=accumulation,
                world=world, rank=rank, seed=int(config['seed']), arm=args.arm,
                const_profile=const_profile, radii=tuple(args.radii))
            counts = window_counts(prepared, device, objective)
            lr = scheduled_lr(step, **{key: config[key] for key in
                                       ('lr', 'warmup_steps', 'schedule_total_steps', 'end_lr')})
            for group in optimizer.param_groups:
                group['lr'] = lr
            optimizer.zero_grad(set_to_none=True)
            totals = torch.zeros(4, device=device)
            for offset, (data, labels) in enumerate(prepared):
                sync = (module.no_sync() if world > 1 and offset + 1 < accumulation
                        else nullcontext())
                with sync:
                    moved_data = data.to(device, non_blocking=True)
                    moved_labels = move_labels(labels, device)
                    with torch.autocast(device.type, dtype=torch.bfloat16,
                                        enabled=config['amp_dtype'] == 'bf16'):
                        result = module(moved_data, moved_labels)
                        loss = global_objective(result['sums'], counts, world, weights)
                    if not torch.isfinite(loss):
                        raise FloatingPointError('nonfinite pretraining loss')
                    loss.backward()
                totals += result['sums'].detach()
            grad_norm = float(torch.nn.utils.clip_grad_norm_(
                trainer.parameters(), float(config.get('grad_clip', 1.0)),
                error_if_nonfinite=True))
            before = ({name: parameter.detach().float().clone()
                       for name, parameter in model.named_parameters()
                       if name.startswith(('conditional.', 'spatial.'))}
                      if (step + 1) in monitor_steps and rank == 0 else None)
            optimizer.step()
            if world > 1:
                dist.all_reduce(totals)
            record = dict(step=step + 1, rank=rank, lr=float(lr), arm=args.arm,
                          global_counts=counts.tolist(),
                          losses=(totals / counts.clamp_min(1)).tolist(),
                          grad_norm_preclip=grad_norm,
                          position_first=int(positions[0]), position_last=int(positions[-1]),
                          stream_digest=stream_digest(prepared),
                          step_seconds=time.perf_counter() - step_started)
            if rank == 0 and device.type == 'cuda':
                record['peak_gpu_memory_gib'] = torch.cuda.max_memory_allocated(device) / 2 ** 30
            if (step + 1) % max(1, int(args.log_every)) == 0 or step + 1 == stop:
                print(json.dumps(record), flush=True)
            with records_path.open('a', encoding='utf-8') as handle:
                handle.write(json.dumps(record, sort_keys=True) + '\n')
            if rank == 0 and (step + 1) in monitor_steps and isinstance(model, GLTFusionB):
                row = monitor_row(model, trainer, moved_data, moved_labels, step + 1,
                                  lr, grad_norm, record['losses'], before=before)
                row['arm'] = args.arm
                with monitor_path.open('a', encoding='utf-8') as handle:
                    handle.write(json.dumps(row) + '\n')
            if (step + 1) in save_steps or (step + 1) % int(config['save_every']) == 0:
                states = [None] * world
                if world > 1:
                    dist.all_gather_object(states, rng_state())
                else:
                    states[0] = rng_state()
                if rank == 0:
                    save_checkpoint(output / f'resume_{step + 1:05d}.pt', dict(
                        identity=identity, ordered_keys_hash=ordered_keys_hash,
                        step=step + 1, next_position=(step + 1) * batch_size,
                        model=trainer.state_dict(), optimizer=optimizer.state_dict(),
                        rng=states, scheduler=dict(step=step + 1, lr=float(lr))))
            if (step + 1) in deploy_steps or step + 1 == deployment_step:
                if rank == 0:
                    save_checkpoint(output / f'deploy_{step + 1:05d}.pt',
                                    fusion_deployment_package(trainer, step + 1,
                                                              arm=args.arm,
                                                              extra={'profile_stats': stats_record}))
                if world > 1:
                    dist.barrier()
        RUN_CONTEXT['stage'] = 'finalise'
        if rank == 0:
            runtime = json.loads((output / 'runtime.json').read_text(encoding='utf-8'))
            runtime.update(status='PASS', stage='complete', completed_steps=int(stop - start),
                           finished_at_monotonic=time.perf_counter(),
                           process_wall_seconds=float(time.perf_counter()
                                                      - RUN_CONTEXT.get('process_started',
                                                                        time.perf_counter())))
            write_json(output / 'runtime.json', runtime)
    finally:
        source.close()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    RUN_CONTEXT['process_started'] = time.perf_counter()
    try:
        main()
    except Exception as error:                      # results first, then FAIL
        write_failure(RUN_CONTEXT, error)
        print(f'PH_FUSION_RUN_FAILED arm={RUN_CONTEXT.get("arm")} '
              f'stage={RUN_CONTEXT.get("stage")} {type(error).__name__}: {error}', flush=True)
        raise
