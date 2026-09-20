#!/usr/bin/env python3
"""GLT-GALPH r2 pretraining entry: N0/N1/C0/C1 from one implementation.

Arm = (summary_mode, ph_mode): N0 mean/none, N1 mean/global, C0 cls/none,
C1 cls/global.  Objectives are 2D masked native-token CE, 3D masked line-node
CE, dual-view multi-positive InfoNCE and (N1/C1) masked-patch PH Huber.

Denominators are global over the whole optimizer window and all ranks: the
window counts are reduced before the first backward and every microbatch is
scaled with ``global_objective`` (DDP averages gradients, so each local sum is
multiplied by ``world_size`` and divided by the global count).  A per-rank
``local_sum / local_count`` would not be a strict global token mean.
"""
import argparse
import hashlib
from contextlib import nullcontext
import json
import math
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from src.dataset.glt_galformer_ph import (PHBettiReader, galformer_collate,
                                          prepare_galformer_sample)
from src.modules.glt_dual_pretrain import (alignment_local_anchor_count,
                                           global_objective)
from src.modules.glt_galformer_ph_pretrain import (GalformerPretrainer,
                                                   galformer_deployment_package)
from src.modules.glt_galformer_ph import PH_ENCODER_VERSION, PH_PATCHES
from src.training.glt_dual_runtime import (IndexedFrozenDualSource,
                                           OrderedSampleStream,
                                           apply_common_initialization,
                                           load_sample_index_artifact,
                                           move_labels, open_source,
                                           ordered_text_hash, require_tmux,
                                           restore_rng, rng_state,
                                           save_checkpoint, scheduled_lr,
                                           write_json)
from src.utils import set_global_seed

ARMS = {'N0': ('mean', None), 'N1': ('mean', 'global'),
        'C0': ('cls', None), 'C1': ('cls', 'global')}
MASK_GROUPS = ('mask2d_rows', 'mask2d_policy', 'mask2d_donor_atoms',
               'mask3d_rows', 'mask3d_policy', 'mask3d_donor_atoms')
# The PH path the residual gate multiplies: encoder and projection, not ph_head.
PH_PATH_PREFIXES = ('ph_encoder.', 'ph_to_summary.', 'alpha_ph')


def is_ph_path_parameter(name):
    return name.startswith(PH_PATH_PREFIXES)


def ph_parameter_state(model, weight_decay):
    """Per-tensor norms, task gradients and the coupled decay term of the PH path."""
    per_tensor, gradient_square, decay_square = {}, 0.0, 0.0
    for name, parameter in model.named_parameters():
        if not is_ph_path_parameter(name):
            continue
        parameter_norm = float(parameter.detach().float().norm())
        gradient_norm = (float(parameter.grad.detach().float().norm())
                         if parameter.grad is not None else 0.0)
        decay_norm = float(weight_decay) * parameter_norm
        per_tensor[name] = {'parameter_norm': parameter_norm,
                            'gradient_norm': gradient_norm,
                            'decay_term_norm': decay_norm}
        gradient_square += gradient_norm ** 2
        decay_square += decay_norm ** 2
    return {'per_tensor': per_tensor, 'gradient_norm_total': gradient_square ** 0.5,
            'decay_term_norm_total': decay_square ** 0.5}


def ph_monitor_row(model, data, probe, const, step, *, weight_decay, loss_terms,
                   pre_clip=None, post_clip=None, before=None):
    """One monitoring row: gate, norms, gradients, decay, update size, sensitivity.

    The probe forward runs under ``eval`` + ``no_grad``: it consumes no dropout RNG,
    leaves no gradient, and cannot move the sample position or the scheduler.
    """
    row = {'step': int(step), 'weight_decay': float(weight_decay),
           'alpha_ph': float(model.alpha_ph.detach()),
           'tanh_alpha': float(torch.tanh(model.alpha_ph.detach())),
           'losses': loss_terms,
           'ph_parameter_norms': {name: float(parameter.detach().float().norm())
                                  for name, parameter in model.named_parameters()
                                  if is_ph_path_parameter(name)}}
    if pre_clip is not None:
        row['pre_clip'] = pre_clip
    if post_clip is not None:
        row['post_clip'] = post_clip
    if before is not None:
        row['parameter_update'] = {
            name: float((parameter.detach().float() - before[name]).norm())
            for name, parameter in model.named_parameters()
            if is_ph_path_parameter(name) and name in before}
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            probe_mask = torch.zeros((probe.size(0), PH_PATCHES), dtype=torch.bool,
                                     device=probe.device)
            summary = model.ph_encoder.summarize(model.ph_encoder(probe, probe_mask)).float()
            average = model.ph_encoder.summarize(model.ph_encoder(
                const.unsqueeze(0).expand_as(probe).contiguous(), probe_mask)).float()
            out = model(data)
            residual = torch.tanh(model.alpha_ph.float()) * model.ph_to_summary(
                out['ph_summary'].float())
            valid = data.ph_valid.bool().unsqueeze(-1)
            masked = torch.where(valid, residual, torch.zeros_like(residual))
            reference = torch.where(valid, out['g3'].float() - residual,
                                    out['g3'].float())
            reference_norm = float(reference.norm())
            row['probe_sensitivity'] = {
                'probe_count': int(probe.size(0)),
                'profile_spread': float((probe.max(0).values - probe.min(0).values).max()),
                'summary_norm_mean': float(summary.norm(dim=-1).mean()),
                'summary_spread': float((summary.max(0).values - summary.min(0).values).max()),
                'real_vs_const_max_abs': float((summary - average).abs().max())}
            row['residual_relative_norm'] = (float(masked.norm()) / reference_norm
                                             if reference_norm else 0.0)
            row['graph_count'] = int(data.graph_available.numel())
            row['all_finite'] = bool(torch.isfinite(masked).all()
                                     and torch.isfinite(summary).all())
    finally:
        if was_training:
            model.train()
    return row


class _MicrobatchStream(torch.utils.data.Dataset):
    """Optional multi-worker prefetch of the exact training microbatch stream.

    One item is one collated microbatch, produced in global position order; every
    per-sample random stream (2D/3D/PH masking) is a pure function of
    ``(seed, sample_key, position)``, so preparing in workers cannot change the
    sample order or the mask streams.
    """

    def __init__(self, source, reader, *, seed, world, rank, microbatch,
                 accumulation, start_step, max_steps):
        self.source, self.reader = source, reader
        self.stream = OrderedSampleStream(len(source), int(seed))
        self.seed, self.world, self.rank = int(seed), int(world), int(rank)
        self.microbatch, self.accumulation = int(microbatch), int(accumulation)
        self.batch_size = self.microbatch * self.world * self.accumulation
        self.start_step, self.steps = int(start_step), max(0, int(max_steps) - int(start_step))

    def __len__(self):
        return self.steps * self.accumulation

    def __getitem__(self, item):
        step = self.start_step + item // self.accumulation
        offset = item % self.accumulation
        rows = []
        for local in range(self.microbatch):
            rows.append(self._one(step, offset, local))
        return galformer_collate(rows)

    def _one(self, step, offset, local):
        position = (step * self.batch_size + offset * self.world * self.microbatch
                    + self.rank * self.microbatch + local)
        index = self.stream.index_at(position)
        key, _ = self.source.samples[index]
        return prepare_galformer_sample(
            *self.source[index], seed=self.seed, key=key.hex(), position=position,
            static=self.source.static_for(index), ph_reader=self.reader)


def _tensor_bytes(value):
    value = value.detach().cpu().contiguous()
    return str(tuple(value.shape)).encode('ascii'), value.numpy().tobytes()


def _stream_digest(prepared, ph_mode):
    """Digest of the positions, 2D/3D/PH masks and targets of one window."""
    digest = hashlib.sha256()
    for data, labels in prepared:
        for name in MASK_GROUPS:
            digest.update(name.encode('ascii'))
            for part in _tensor_bytes(getattr(data, name).long()):
                digest.update(part)
        for name in ('label_2d', 'label_3d'):
            digest.update(name.encode('ascii'))
            for part in _tensor_bytes(labels[name].long()):
                digest.update(part)
        if ph_mode:
            for name, value in (('ph_mask', data.ph_mask.to(torch.uint8)),
                                ('ph_valid', data.ph_valid.to(torch.uint8)),
                                ('label_ph', labels['label_ph'].float())):
                digest.update(name.encode('ascii'))
                for part in _tensor_bytes(value):
                    digest.update(part)
    return digest.hexdigest()


def _prepare_window(source, stream, reader, *, step, micro, accumulation, world, rank,
                    seed):
    prepared = []
    positions = []
    batch_size = micro * accumulation * world
    for offset in range(accumulation):
        rows = []
        for local in range(micro):
            position = step * batch_size + offset * world * micro + rank * micro + local
            index = stream.index_at(position)
            key, _ = source.samples[index]
            rows.append(prepare_galformer_sample(
                *source[index], seed=seed, key=key.hex(), position=position,
                static=source.static_for(index), ph_reader=reader))
            positions.append(position)
        prepared.append(galformer_collate(rows))
    return prepared, positions


def _window_counts(prepared, device, ph_mode):
    """Local window denominators, matching each loss term exactly."""
    counts = torch.zeros(4, device=device)
    for data, labels in prepared:
        count_2d = torch.tensor(float(labels['label_2d'].numel()), device=device)
        count_3d = torch.tensor(float(labels['label_3d'].numel()), device=device)
        valid = (data.graph_available.bool() & data.geometry_valid.bool())
        count_cl = alignment_local_anchor_count(
            labels['identity'], valid, device=device)
        count_ph = ((data.ph_valid.bool() & data.graph_available.bool()).sum().to(device)
                    if ph_mode else torch.zeros((), device=device))
        counts += torch.stack([count_2d, count_3d, count_cl, count_ph.float()])
    if dist.is_initialized():
        dist.all_reduce(counts)
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    for name in ('cohort-root', 'cache-root', 'output'):
        parser.add_argument('--' + name)
    parser.add_argument('--dual-static-root',
                        help='training-ready dual_static_v1 artifact')
    parser.add_argument('--arm', choices=tuple(ARMS), help='N0/N1/C0/C1 (default: config)')
    parser.add_argument('--sample-index-artifact')
    parser.add_argument('--sample-index-split', choices=('train', 'validation', 'fixed_validation'))
    parser.add_argument('--common-init-artifact')
    parser.add_argument('--ph-sidecar-root')
    parser.add_argument('--resume')
    parser.add_argument('--updates', type=int, default=0,
                        help='bounded run: this many optimizer updates from the start '
                             'position (0 = the configured training end)')
    parser.add_argument('--save-steps', type=int, nargs='*', default=[],
                        help='absolute steps at which to write a resume checkpoint')
    parser.add_argument('--deploy-steps', type=int, nargs='*', default=[],
                        help='absolute steps at which to force a deployment export')
    parser.add_argument('--prep-workers', type=int, default=0,
                        help='DataLoader workers per rank for input prefetch')
    parser.add_argument('--ph-no-weight-decay', action='store_true',
                        help='trainability pre-check: PH path in its own Adam group with '
                             'weight_decay=0; every other parameter keeps the configured '
                             'settings')
    parser.add_argument('--alpha-ph-tanh-init', type=float,
                        help='trainability pre-check: start tanh(alpha_ph) at this value')
    parser.add_argument('--ph-monitor', help='JSONL path for the PH monitor rows')
    parser.add_argument('--ph-monitor-steps', type=int, nargs='*',
                        default=[0, 1, 2, 16, 64, 128, 256])
    parser.add_argument('--ph-probe', help='fixed probe profiles [N,3,32] for monitoring')
    parser.add_argument('--ph-const-profile', help='P_train mean profile for monitoring')
    parser.add_argument('--log-every', type=int, default=10)
    parser.add_argument('--write-common-init',
                        help='evaluation-only: write the seeded step-0 shared state and exit')
    args = parser.parse_args()
    require_tmux()
    config = json.loads(Path(args.config).read_text(encoding='utf-8'))
    arm = str(args.arm or config.get('arm', '')).upper()
    if arm not in ARMS:
        raise ValueError('config/--arm must select one of N0/N1/C0/C1')
    summary_mode, ph_mode = ARMS[arm]
    if config.get('summary_mode', summary_mode) != summary_mode \
            or config.get('ph_mode', ph_mode) != ph_mode:
        raise ValueError('config summary/ph mode does not match the selected arm')
    configured_ph_version = config.get('ph_encoder_version')
    expected_ph_version = PH_ENCODER_VERSION if ph_mode == 'global' else None
    if configured_ph_version != expected_ph_version:
        raise ValueError('config PH encoder version does not match implementation')
    for key in ('microbatch', 'global_batch', 'max_optimizer_steps', 'save_every'):
        if int(config[key]) <= 0:
            raise ValueError(f'config {key} must be positive')
    if len(config['loss_weights']) != 4:
        raise ValueError('the Galformer objective has four weighted terms')
    if config['amp_dtype'] not in ('fp32', 'bf16'):
        raise ValueError('precision must be fp32 or bf16')
    if args.write_common_init:
        # The shared step-0 state of the four arms, frozen as one artifact.
        set_global_seed(config['seed'])
        model = GalformerPretrainer('mean', None).model
        state = {name: value.detach().cpu().clone()
                 for name, value in model.state_dict().items()}
        save_checkpoint(args.write_common_init, dict(
            schema_version='glt-galformer-common-init-v1', seed=int(config['seed']),
            architecture=model.architecture_name, arm='N0',
            common_state_sha256=hashlib.sha256(
                json.dumps(sorted(state), separators=(',', ':')).encode()).hexdigest(),
            common_state_dict=state))
        print(json.dumps({'written': args.write_common_init,
                          'tensors': len(state),
                          'common_state_sha256': hashlib.sha256(
                              json.dumps(sorted(state), separators=(',', ':')).encode()).hexdigest()}),
              flush=True)
        return
    for name in ('cohort_root', 'cache_root', 'output', 'dual_static_root'):
        if not getattr(args, name):
            raise ValueError(f'--{name.replace("_", "-")} is required for training')
    rank, world = int(os.environ.get('RANK', 0)), int(os.environ.get('WORLD_SIZE', 1))
    expected_world = config.get('expected_world_size')
    if expected_world is not None and int(world) != int(expected_world):
        raise ValueError(f'config requires world size {int(expected_world)}, got {world}')
    device = (torch.device('cuda', int(os.environ.get('LOCAL_RANK', 0)))
              if torch.cuda.is_available() else torch.device('cpu'))
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

    reader = None
    sidecar_root = args.ph_sidecar_root or config.get('ph_sidecar_root')
    source, _ = open_source(args.cohort_root, args.cache_root,
                            dual_static_root=args.dual_static_root)
    try:
        sample_index_path = args.sample_index_artifact or config.get('sample_index_artifact')
        sample_index_split = args.sample_index_split or config.get('sample_index_split')
        sample_index = load_sample_index_artifact(sample_index_path, sample_index_split)
        if sample_index is not None:
            if len(source) != int(sample_index['payload'].get('pi1m_record_count', len(source))):
                raise ValueError('sample index base cohort count mismatch')
            expected_manifest = sample_index['payload'].get('pi1m_manifest_hash')
            if expected_manifest and expected_manifest != source.cohort['manifest_hash']:
                raise ValueError('sample index/base cohort manifest mismatch')
            source = IndexedFrozenDualSource(source, sample_index['indices'])
        if ph_mode == 'global':
            if not sidecar_root:
                raise ValueError('the PH arms require --ph-sidecar-root')
            reader = PHBettiReader(sidecar_root)
            if not len(reader) == len(source):
                raise ValueError('PH sidecar must cover the whole pretraining manifest')
        micro, batch_size = config['microbatch'], config['global_batch']
        if batch_size % (micro * world):
            raise ValueError('microbatch * accumulation * world must equal global_batch')
        accumulation = batch_size // (micro * world)
        set_global_seed(config['seed'])
        trainer = GalformerPretrainer(summary_mode, ph_mode,
                                      temperature=float(config.get('contrastive_temperature', 0.1))
                                      ).to(device)
        common_init = None
        common_init_path = args.common_init_artifact or config.get('common_init_artifact')
        if common_init_path:
            # The artifact holds the shared step-0 state in the encoder key space.
            common_init = apply_common_initialization(trainer.model, common_init_path)
        if args.alpha_ph_tanh_init is not None:
            if ph_mode != 'global':
                raise ValueError('--alpha-ph-tanh-init requires a PH arm')
            if not -1.0 < float(args.alpha_ph_tanh_init) < 1.0:
                raise ValueError('--alpha-ph-tanh-init must lie strictly inside (-1, 1)')
            with torch.no_grad():
                # Zero-initializing the gate starves every parameter behind it;
                # this pre-check opens it by a fixed, recorded amount instead.
                trainer.model.alpha_ph.fill_(math.atanh(float(args.alpha_ph_tanh_init)))
        weight_decay = float(config.get('weight_decay', 0.0))
        if args.ph_no_weight_decay:
            if ph_mode != 'global':
                raise ValueError('--ph-no-weight-decay requires a PH arm')
            decayed, ph_params = [], []
            for name, parameter in trainer.model.named_parameters():
                (ph_params if is_ph_path_parameter(name) else decayed).append(parameter)
            if not ph_params:
                raise ValueError('the PH path has no parameters to separate')
            optimizer = torch.optim.Adam(
                [dict(params=decayed, weight_decay=weight_decay),
                 dict(params=ph_params, weight_decay=0.0)], lr=float(config['lr']))
        else:
            optimizer = torch.optim.Adam(trainer.parameters(), lr=float(config['lr']),
                                         weight_decay=weight_decay)
        # Decay actually applied to the PH path, for the monitor's decay term.
        effective_wd = 0.0 if args.ph_no_weight_decay else weight_decay
        module = (DistributedDataParallel(trainer,
                                          device_ids=[device.index] if device.type == 'cuda' else None,
                                          find_unused_parameters=True)
                  if world > 1 else trainer)
        start = 0
        identity = dict(
            config=config, arm=arm, summary_mode=summary_mode, ph_mode=ph_mode,
            world_size=world, sample_count=len(source),
            device_type=device.type,
            cohort_root=str(Path(args.cohort_root).resolve()),
            cohort_hash=source.cohort['manifest_hash'],
            cache_root=str(Path(args.cache_root).resolve()),
            main_bundle_hash=source.bundle.bundle_hash,
            dual_static_manifest_hash=(source.static_cache.manifest_hash
                                       if source.static_cache is not None else None),
            ph_sidecar_root=(str(Path(sidecar_root).resolve()) if sidecar_root else None),
            sample_index_artifact_sha256=(sample_index['sha256'] if sample_index else None),
            sample_index_split=(sample_index['split'] if sample_index else None),
            common_init_artifact_sha256=(common_init['sha256'] if common_init else None),
            objective='galformer_mask2d_mask3d_cl' + ('_ph' if ph_mode else ''),
            ph_encoder_version=(PH_ENCODER_VERSION if ph_mode else None),
            mask_candidate_rate=0.40, mask_policy='80_10_10',
            contrastive_temperature=float(config.get('contrastive_temperature', 0.1)),
            optimizer='adam', grad_clip=float(config.get('grad_clip', 1.0)),
            ph_path_weight_decay=('zero' if args.ph_no_weight_decay else 'coupled'),
            alpha_ph_tanh_init=(float(args.alpha_ph_tanh_init)
                                if args.alpha_ph_tanh_init is not None else None))
        ordered_keys_hash = ordered_text_hash(
            key.hex() for key, _ in source.samples)
        resume_rng = None
        if args.resume:
            state = torch.load(args.resume, map_location='cpu', weights_only=False)
            if state['identity'] != identity or state['ordered_keys_hash'] != ordered_keys_hash:
                raise ValueError('resume data/config/world size differs')
            trainer.load_state_dict(state['model'], strict=True)
            optimizer.load_state_dict(state['optimizer'])
            start = int(state['step'])
            if state['next_position'] != start * batch_size or state['scheduler']['step'] != start:
                raise ValueError('resume scheduler/data position mismatch')
            resume_rng = state['rng'][rank]
        else:
            set_global_seed(config['seed'] + rank)   # independent dropout streams
        stop = config['max_optimizer_steps']
        if args.updates:
            stop = start + int(args.updates)
            if stop > config['max_optimizer_steps']:
                raise ValueError('--updates exceeds the configured training end')
        if not 0 <= start < stop:
            raise ValueError('start position must precede the training end')
        save_steps = sorted({int(value) for value in args.save_steps})
        deploy_steps = sorted({int(value) for value in args.deploy_steps})
        deployment_step = int(config.get('deployment_step', config['max_optimizer_steps']))
        for value in save_steps + deploy_steps:
            if not 0 < value <= stop:
                raise ValueError('--save-steps/--deploy-steps must lie inside this run')
        if any(int(p.stem.rsplit('_', 1)[-1]) > start for p in output.glob('deploy_*.pt')):
            raise FileExistsError('resume would overwrite later deployment checkpoints')
        if any(int(p.stem.rsplit('_', 1)[-1]) > start for p in output.glob('resume_*.pt')):
            raise FileExistsError('resume would overwrite later training checkpoints')
        if rank == 0:
            write_json(output / 'run.json', dict(
                identity=identity, command=sys.argv, accumulation=accumulation,
                start_step=start, stop_step=stop, save_steps=save_steps,
                deploy_steps=deploy_steps, deployment_step=deployment_step,
                prep_workers=int(args.prep_workers), arm=arm))
            write_json(output / 'runtime.json', dict(
                status='RUNNING', command=sys.argv, identity=identity, rank=rank,
                world_size=world, device=str(device), arm=arm,
                started_at_monotonic=time.perf_counter()))
        stream = OrderedSampleStream(len(source), config['seed'])
        prefetch = None
        if int(args.prep_workers) > 0:
            dataset = _MicrobatchStream(source, reader, seed=config['seed'], world=world,
                                        rank=rank, microbatch=micro,
                                        accumulation=accumulation, start_step=start,
                                        max_steps=stop)
            prefetch = iter(torch.utils.data.DataLoader(
                dataset, batch_size=None, num_workers=int(args.prep_workers),
                prefetch_factor=4, persistent_workers=False, pin_memory=False))
        if resume_rng is not None:
            restore_rng(resume_rng)
        weights = [float(value) for value in config['loss_weights']]
        records_path = output / f'records_rank{rank}.jsonl'
        monitor_steps = {int(value) for value in args.ph_monitor_steps} if args.ph_monitor else set()
        probe = const = None
        if monitor_steps:
            if not args.ph_probe or not args.ph_const_profile:
                raise ValueError('--ph-monitor requires --ph-probe and --ph-const-profile')
            import numpy as np
            probe = torch.as_tensor(np.load(args.ph_probe), dtype=torch.float32).to(device)
            const = torch.as_tensor(np.load(args.ph_const_profile),
                                    dtype=torch.float32).to(device)
            monitor_path = Path(args.ph_monitor)
        for step in range(start, stop):
            step_started = time.perf_counter()
            prepared_started = time.perf_counter()
            if prefetch is not None:
                prepared = [next(prefetch) for _ in range(accumulation)]
                positions = [(step * batch_size + offset * world * micro
                              + rank * micro + local)
                             for offset in range(accumulation) for local in range(micro)]
            else:
                prepared, positions = _prepare_window(
                    source, stream, reader, step=step, micro=micro,
                    accumulation=accumulation, world=world, rank=rank,
                    seed=config['seed'])
            preparation_seconds = time.perf_counter() - prepared_started
            counts = _window_counts(prepared, device, ph_mode)
            lr = scheduled_lr(step, **{key: config[key] for key in
                                       ('lr', 'warmup_steps', 'schedule_total_steps', 'end_lr')})
            for group in optimizer.param_groups:
                group['lr'] = lr
            # A row labelled k is the state after exactly k optimizer updates,
            # so the requested steps are matched by the completed-update counter.
            monitor_now = bool(monitor_steps) and (step + 1) in monitor_steps and rank == 0
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
            pre_clip = (ph_parameter_state(trainer.model, effective_wd)
                        if monitor_now else None)
            grad_norm = float(torch.nn.utils.clip_grad_norm_(
                trainer.parameters(), float(config.get('grad_clip', 1.0)),
                error_if_nonfinite=True))
            post_clip = (ph_parameter_state(trainer.model, effective_wd)
                         if monitor_now else None)
            ph_grad_norms = None
            if ph_mode == 'global':
                def _grad_norm(module):
                    values = [parameter.grad.detach().float().reshape(-1)
                              for parameter in module.parameters()
                              if parameter.grad is not None]
                    return float(torch.cat(values).norm()) if values else 0.0
                ph_grad_norms = {
                    'alpha_ph': (float(trainer.model.alpha_ph.grad.detach().float().abs().max())
                                 if trainer.model.alpha_ph.grad is not None else 0.0),
                    'scale_fuse': _grad_norm(trainer.model.ph_encoder.scale_fuse),
                    'ph_encoder': _grad_norm(trainer.model.ph_encoder),
                }
            before = ({name: parameter.detach().float().clone()
                       for name, parameter in trainer.model.named_parameters()
                       if is_ph_path_parameter(name)} if monitor_now else None)
            if rank == 0 and 0 in monitor_steps and step == 0:
                # Step 0 is the initial state: recorded before any optimizer update.
                initial = ph_monitor_row(
                    trainer.model, moved_data, probe, const, 0,
                    weight_decay=effective_wd,
                    loss_terms=(totals / counts.clamp_min(1)).tolist(),
                    pre_clip=pre_clip, post_clip=post_clip)
                initial.update(arm=arm, lr=float(lr), grad_norm_preclip=grad_norm,
                               losses_scope='rank0_local_window_before_all_reduce',
                               note='initial state, before the first optimizer update')
                with monitor_path.open('a', encoding='utf-8') as handle:
                    handle.write(json.dumps(initial) + '\n')
            optimizer.step()
            if world > 1:
                dist.all_reduce(totals)
            record = dict(
                step=step + 1, rank=rank, lr=float(lr), arm=arm,
                global_counts=counts.tolist(),
                losses=(totals / counts.clamp_min(1)).tolist(),
                sums=totals.tolist(),
                grad_norm_preclip=grad_norm,
                position_first=int(positions[0]), position_last=int(positions[-1]),
                stream_digest=_stream_digest(prepared, bool(ph_mode)),
                preparation_seconds=preparation_seconds,
                step_seconds=time.perf_counter() - step_started)
            if ph_grad_norms is not None:
                record['ph_grad_norms'] = ph_grad_norms
            if rank == 0 and device.type == 'cuda':
                record['peak_gpu_memory_gib'] = torch.cuda.max_memory_allocated(device) / 2 ** 30
            if (step + 1) % max(1, int(args.log_every)) == 0 or step + 1 == stop:
                print(json.dumps(record), flush=True)
            with records_path.open('a', encoding='utf-8') as handle:
                handle.write(json.dumps(record, sort_keys=True) + '\n')
            if monitor_now:
                row = ph_monitor_row(
                    trainer.model, moved_data, probe, const, step + 1,
                    weight_decay=effective_wd, loss_terms=record['losses'],
                    pre_clip=pre_clip, post_clip=post_clip, before=before)
                row['arm'] = arm
                row['lr'] = float(lr)
                row['grad_norm_preclip'] = grad_norm
                with monitor_path.open('a', encoding='utf-8') as handle:
                    handle.write(json.dumps(row) + '\n')
            if (step + 1) in save_steps or (step + 1) % config['save_every'] == 0 \
                    or step + 1 == config['max_optimizer_steps']:
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
                                    galformer_deployment_package(trainer, step + 1))
                if world > 1:
                    dist.barrier()
        if rank == 0:
            runtime = json.loads((output / 'runtime.json').read_text(encoding='utf-8'))
            runtime.update(status='PASS', completed_steps=int(stop - start),
                           finished_at_monotonic=time.perf_counter())
            write_json(output / 'runtime.json', runtime)
    finally:
        source.close()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
