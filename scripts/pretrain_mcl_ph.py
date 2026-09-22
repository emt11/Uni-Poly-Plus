#!/usr/bin/env python3
"""MCL-PH pre-training entry (MCL-PH-20260921-01/r1).

One trajectory per fusion arm.  The shared parameter initial state is fixed
before any arm runs and is recorded by content hash, so ``M-CAT``, ``M-GATE``
and ``M-XATTN`` differ only in their fusion module and their declared
downstream head, never in the O8 backbone, the three experts, the router or the
geometry decoders.  ``GLT_REF`` is not this script: it is the pre-existing dual
route, run unchanged.
"""
import argparse
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import sys
import faulthandler
import os
import resource
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from src.dataset.mcl_ph_view import (MCLPHMicrobatchStream, load_geometric_statistics,
                                     mcl_ph_collate, prepare_mcl_ph_sample)
from src.modules.mcl_ph import global_sum
from src.modules.mcl_ph_pretrain import (BALANCE_WEIGHT, MCLPHPretrainer,
                                         effective_graph_counts)
from src.training.glt_dual_runtime import (IndexedFrozenDualSource, OrderedSampleStream,
                                           apply_common_initialization, load_sample_index_artifact,
                                           move_labels, open_source, require_tmux,
                                           restore_rng, rng_state, save_checkpoint, sha256_file,
                                           write_json)
from src.utils import set_global_seed

ARMS = ('cat', 'gate', 'xattn')
SHARED_INIT_SEED = 20260921
SAVED_STEPS = (1, 2, 500, 501, 1000, 2000, 3000, 4000, 5000)
FORMAL_WORLD_SIZE = 4


def _gradient_groups(model):
    """Gradient norm per declared module group, measured before clipping."""

    groups = {
        'o8_2d_encoder': model.encoder.o8,
        'expert_2a': None, 'expert_3a': None, 'expert_4a': None,
        'router': model.encoder.branch.router,
        'fusion': model.encoder.fusion,
        'atom_head': model.atom_head,
        'local_geometry_decoder': model.local_decoder,
        'nonbond_decoder': model.nonbond_decoder,
    }
    for slot, expert in enumerate(model.encoder.branch.experts):
        groups[f'expert_{int(expert.cutoff)}a'] = expert
    seen, report = set(), {}
    for name, module in groups.items():
        if module is None:
            report[name] = 0.0
            continue
        params = [p for p in module.parameters() if p.requires_grad]
        seen.update(id(p) for p in params)
        flat = [p.grad.detach().reshape(-1) for p in params if p.grad is not None]
        report[name] = float(torch.cat(flat).norm()) if flat else 0.0
    return report


def _parameter_snapshot(model):
    digest = hashlib.sha256()
    summary = {}
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().float().cpu().reshape(-1)
        digest.update(name.encode('utf-8'))
        digest.update(tensor.numpy().tobytes())
        if tensor.numel():
            summary[name] = {'shape': list(value.shape),
                             'rms': float(tensor.pow(2).mean().sqrt()),
                             'mean': float(tensor.mean())}
    return digest.hexdigest(), summary


class StageLogger:
    """Per-rank stage log plus a stall watchdog for the export epilogue.

    A run that hangs after its last step leaves nothing behind: no record, no
    stack, and a ``runtime.json`` that still says RUNNING.  Every mark here is
    written and flushed immediately so an externally stopped process still shows
    its last completed stage, and the ``faulthandler`` timer is re-armed at every
    mark so that a stage which stops making progress dumps the stacks of *this*
    rank to its own file.

    Neither the marks nor the watchdog add synchronization, collectives or RNG
    use: the log is a file write on the marking process, and the watchdog is a
    timer thread that only reads stacks.
    """

    DEFAULT_STALL_SECONDS = 120.0

    def __init__(self, directory, rank, stall_seconds=None):
        self.rank = int(rank)
        self.pid = os.getpid()
        self.stall_seconds = (self.DEFAULT_STALL_SECONDS if stall_seconds is None
                              else float(stall_seconds))
        directory = Path(directory)
        self.handle = (directory / f'stages_rank{self.rank}.log').open('a', encoding='utf-8')
        self.stack_handle = (directory / f'stall_stack_rank{self.rank}.txt').open(
            'a', encoding='utf-8')

    def mark(self, stage, event, **extra):
        self.handle.write(json.dumps({'monotonic': time.perf_counter(), 'rank': self.rank,
                                      'pid': self.pid, 'stage': str(stage),
                                      'event': str(event), **extra}, sort_keys=True) + '\n')
        self.handle.flush()
        self._arm()

    def tensor_progress(self, event, name):
        self.mark(f'tensor:{name}', event)

    def _arm(self):
        if self.stall_seconds <= 0:
            return
        faulthandler.cancel_dump_traceback_later()
        faulthandler.dump_traceback_later(self.stall_seconds, repeat=True,
                                         file=self.stack_handle)

    def close(self):
        faulthandler.cancel_dump_traceback_later()
        try:
            self.handle.close()
            self.stack_handle.close()
        except OSError:
            pass


class _NullStages:
    """Stand-in for the stage log before the output directory exists.

    Failures can happen before there is anywhere to log them; every mark call
    has to stay valid so that the original error is what the caller sees.
    """

    def mark(self, *args, **kwargs):
        return None

    def close(self):
        return None


def memory_record():
    """Per-rank memory observation, one labelled record per measurement window.

    The two numbers do not cover the same interval, so they must not be read as
    two fields of one window: the CUDA peaks start at the previous
    ``reset_peak_memory_stats`` call (a step window), while the CPU figure covers
    the whole life of this rank process.  Each side therefore carries its own
    ``window`` label, and the CPU side states that the dataloader workers are
    separate processes that its peak RSS does not include.
    """
    record = {'units': 'bytes',
              'cuda': {'window': 'step_since_last_reset'},
              'cpu': {'window': 'rank_process_lifetime'}}
    if torch.cuda.is_available():
        record['cuda'].update(peak_allocated_bytes=int(torch.cuda.max_memory_allocated()),
                              peak_reserved_bytes=int(torch.cuda.max_memory_reserved()),
                              statistics_available=True)
    else:
        record['cuda'].update(peak_allocated_bytes='NOT_MEASURED',
                              peak_reserved_bytes='NOT_MEASURED',
                              statistics_available=False)
    try:
        record['cpu'].update(
            peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024,
            scope='rank_process_peak_rss_dataloader_workers_excluded')
    except (ValueError, OSError):  # pragma: no cover - platform dependent
        record['cpu'].update(peak_rss_bytes='NOT_MEASURED', scope='NOT_MEASURED')
    return record


def write_failure_record(error, rank, phase):
    """Write a failure where it can be read, before any cleanup can block.

    The record is a per-rank append-only log *and* the per-rank JSON the earlier
    rounds already used; rank 0 owns ``runtime.json``.  ``phase`` distinguishes
    the record written on the way out of the training body from the one written
    by the outermost handler, so a duplicate is recognisable instead of looking
    like two independent failures.  Nothing here is collective.
    """
    code = _failure_exit_code(error)
    record = dict(status='FAILED', rank=int(rank), exit_code=code, command=sys.argv,
                  phase=str(phase), error=f'{type(error).__name__}: {error}')
    output = _OUTPUT[0]
    if output is not None and Path(output).is_dir():
        try:
            with (Path(output) / f'failure_rank{rank}.log').open('a', encoding='utf-8') as handle:
                handle.write(json.dumps({'monotonic': time.perf_counter(), **record},
                                        sort_keys=True) + '\n')
            write_json(Path(output) / f'runtime_failure_rank{rank}.json', record)
            if int(rank) == 0:
                write_json(Path(output) / 'runtime.json', record)
        except OSError as write_error:
            record['record_error'] = f'{type(write_error).__name__}: {write_error}'
    print(json.dumps(record, sort_keys=True), flush=True)
    return record


def shared_new_state(model):
    """Tensors that must be identical in every arm, keyed by parameter name."""
    prefixes = ('encoder.branch.', 'atom_head.', 'local_decoder.', 'nonbond_decoder.')
    return {name: value.detach().cpu().clone()
            for name, value in model.state_dict().items()
            if name.startswith(prefixes)}


SHARED_INIT_SCHEMA = 'mcl-ph-shared-new-init-v2'


def build_shared_init(path, *, dropout, cutoffs, dense_updates):
    """Create the one shared initial state for the new route, once."""
    previous = torch.get_rng_state()
    set_global_seed(SHARED_INIT_SEED)
    reference = MCLPHPretrainer('gate', dropout=dropout, cutoffs=cutoffs,
                                router_dense_updates=dense_updates)
    state = shared_new_state(reference)
    # v2: the pre-r2 artifact was built while a recursive ``apply`` still
    # overwrote the router and gate initialisation, so it must never be
    # reused as the initial state of the fixed route.
    payload = {'schema': SHARED_INIT_SCHEMA, 'seed': SHARED_INIT_SEED,
               'source': 'MCLPHPretrainer(fusion=gate)', 'state_dict': state}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(payload, temporary)
    os.replace(temporary, path)
    torch.set_rng_state(previous)
    return payload


def SHARED_INIT_EXCLUDE_NAMES(model, artifact_state):
    """Names excluded from the shared GLT dual state on the MCL-PH route.

    Only the O8 backbone is shared between the two routes: the GLT encoder and
    the GLT heads exist in the artifact but not here, and this route's own
    modules must keep their independent initialization.
    """
    return tuple(sorted(
        {name for name in model.state_dict() if not name.startswith('encoder.o8.')}
        | {name for name in artifact_state if not name.startswith('encoder.o8.')}))


def apply_shared_init(model, path):
    payload = torch.load(Path(path), map_location='cpu', weights_only=False)
    if payload.get('schema') != SHARED_INIT_SCHEMA:
        raise ValueError('shared initialization artifact has an unsupported schema: '
                         f"{payload.get('schema')!r}; the v1 artifact predates the "
                         'r2 initialization fix and may not be reused')
    state = payload['state_dict']
    current = model.state_dict()
    expected = set(shared_new_state(model))
    if set(state) != expected:
        missing = sorted(expected - set(state))
        extra = sorted(set(state) - expected)
        raise ValueError(f'shared initialization tensor set mismatch: '
                         f'missing={missing[:4]} extra={extra[:4]}')
    for name, value in state.items():
        if tuple(current[name].shape) != tuple(value.shape):
            raise ValueError(f'shared initialization shape mismatch: {name}')
    with torch.no_grad():
        for name, value in state.items():
            current[name].copy_(value)
    return {'path': str(Path(path).resolve()), 'sha256': sha256_file(path),
            'schema': payload['schema'], 'seed': payload['seed'],
            'tensor_count': len(state)}


def reduce_update_denominators(update_counts, global_sum, device):
    """The exact denominator of one optimizer update, one entry per task.

    ``update_counts`` are this rank's own effective graph counts for the update
    and ``global_sum`` is the collective the objective already uses, so both the
    value and the collective behaviour are unchanged by this call.  It is a named
    helper because the total belongs to the update rather than to the loop that
    computes it: a local name that only ever holds this reduction cannot be
    mistaken for the initialization identity the export metadata reads.
    """
    denominators = {}
    for name, value in update_counts.items():
        denominator_total = global_sum(torch.tensor([float(value)], device=device))
        denominators[name] = float(denominator_total.detach().reshape(-1)[0])
    return denominators


def _step_diagnostics(model, report, batch, labels, world):
    """Observation only: read from tensors the objective already computed."""
    atom_states = report['atom_states']
    fused = report['fused']
    # One device for the whole report: the diagnostics are observation only and
    # must never fail because a caller mixed a host batch with moved labels.
    device = atom_states.device
    increment = (fused - atom_states).detach().float()
    reference = atom_states.detach().float().norm()
    trajectory = batch.mcl_trajectory.detach().float().to(device)
    pairs = labels['mcl_nonbond_pair'].long().to(device)
    position = batch.mcl_pos.float().to(device)
    view_distance = (torch.linalg.vector_norm(position[pairs[:, 0]] - position[pairs[:, 1]],
                                              dim=-1) if int(pairs.size(0))
                     else torch.zeros(0, device=device))
    raw = labels['mcl_nonbond_raw'].float().to(device)
    copy_error = (torch.log1p(view_distance) - torch.log1p(raw)).abs() if int(raw.numel()) \
        else torch.zeros(0, device=device)
    length_pairs = labels['mcl_length_pair'].long().to(device)
    length_view = (torch.linalg.vector_norm(position[length_pairs[:, 0]]
                                            - position[length_pairs[:, 1]], dim=-1)
                   if int(length_pairs.size(0)) else torch.zeros(0, device=device))
    length_raw = labels['mcl_length_raw'].float().to(device)
    length_copy = (torch.log1p(length_view) - torch.log1p(length_raw)).abs() \
        if int(length_raw.numel()) else torch.zeros(0, device=device)
    return {
        'losses': {name: float(value.detach()) for name, value in
                   (('atom', report['atom_sum'] / report['atom_count'].clamp_min(1)),
                    ('geometry', report['geo_sum'] / report['geo_count'].clamp_min(1)),
                    ('balance', report['balance']))},
        'valid_graphs': {'atom': int(report['atom_count']), 'geometry': int(report['geo_count']),
                         'local': int(report['local_count']), 'nonbond': int(report['nonbond_count'])},
        'target_counts': [float(value) for value in report['targets'].detach()],
        'router': dict(report.get('router_diagnostics') or {}),
        'router_mode': model.encoder.branch.router.mode,
        'expert_update_norm': {f'{int(expert.cutoff)}a': float(states.detach().float().norm())
                               for expert, states in zip(model.encoder.branch.experts,
                                                         report['expert_states'])},
        'fusion': {
            'increment_norm': float(increment.norm()),
            'activation_norm': float(reference),
            'relative_increment': float(increment.norm() / reference.clamp_min(1e-12)),
            'gate': report.get('gate_diagnostics'),
        },
        'trajectory': {
            # Same reporting rule as the router diagnostics: the population std
            # stays finite for a single graph, and an empty batch reports that
            # nothing was measured instead of NaN.
            'per_column_mean': ([float(value) for value in trajectory.mean((0, 1))]
                                if int(trajectory.size(0)) else None),
            'per_column_std': ([float(value) for value in
                                trajectory.std((0, 1), correction=0)]
                               if int(trajectory.size(0)) else None),
            'statistics': ('NOT_APPLICABLE_EMPTY_BATCH' if not int(trajectory.size(0))
                           else 'population_std_correction_0'),
            'nonzero_graphs': int((trajectory.abs().sum(-1).sum(-1) > 0).sum()),
        },
        'distance_copy_baseline': {
            'nonbond_abs_log1p_error_mean': float(copy_error.mean()) if int(copy_error.numel()) else None,
            'nonbond_abs_log1p_error_max': float(copy_error.max()) if int(copy_error.numel()) else None,
            'length_abs_log1p_error_mean': float(length_copy.mean()) if int(length_copy.numel()) else None,
            'note': 'increment between the view distance and the clean target; the model '
                    'must beat this on the same target set to be useful',
        },
        'world_size': int(world),
    }


_OUTPUT = [None]


def _trajectory_cache_identity(config, *, batch_size, sample_index, source):
    """The identity an offline topology cache must match field for field."""
    from src.dataset.mcl_ph_trajectory_cache import cache_identity

    return cache_identity(seed=int(config['seed']),
                          noise_sigma=float(config['noise_sigma']),
                          mask_ratio=float(config['atom_mask_ratio']),
                          global_batch=int(batch_size),
                          sample_index_split=sample_index['split'],
                          sample_index_artifact_sha256=sample_index['sha256'],
                          cohort_manifest_hash=source.cohort['manifest_hash'],
                          dual_static_manifest_hash=source.static_cache.manifest_hash)


def open_trajectory_cache(path, *, config, batch_size, sample_index, source):
    """Open the offline topology cache and return ``(cache, identity block)``.

    A cache that does not describe this run's presentations is refused here,
    before the model is built and before any position is read.  ``None`` keeps
    the declared online behaviour of every earlier MCL-PH run.
    """
    from src.dataset.mcl_ph_trajectory_cache import MCLPHTrajectoryCache, online_provenance

    if not path:
        return None, online_provenance()
    expected = _trajectory_cache_identity(config, batch_size=batch_size,
                                          sample_index=sample_index, source=source)
    cache = MCLPHTrajectoryCache(path, expected=expected)
    block = {'mode': 'cached', 'path': str(Path(path).resolve()),
             'schema': cache.manifest['schema'], 'dtype': cache.manifest['dtype'],
             'total_positions': cache.total_positions, 'shard_size': cache.shard_size,
             'manifest_identity': expected}
    return cache, block


def _failure_exit_code(error):
    if isinstance(error, SystemExit):
        code = error.code
        return 0 if code is None else (code if isinstance(code, int) else 1)
    return 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'cohort-root', 'cache-root', 'output', 'statistics'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--dual-static-root', required=True)
    parser.add_argument('--sample-index-artifact')
    parser.add_argument('--sample-index-split', default='train')
    parser.add_argument('--common-init-artifact')
    parser.add_argument('--shared-new-init', required=True,
                        help='fixed shared initial state for the new parameters; created once '
                             'if it does not exist yet')
    parser.add_argument('--prep-workers', type=int, default=0)
    parser.add_argument('--trajectory-cache', help='offline noisy-topology cache root; '
                                                   'omitted means the declared online path')
    parser.add_argument('--diagnostics', action='store_true')
    parser.add_argument('--stop-after-step', type=int, default=0)
    parser.add_argument('--no-deploy', action='store_true')
    parser.add_argument('--resume')
    args = parser.parse_args()
    require_tmux()
    config = json.loads(Path(args.config).read_text(encoding='utf-8'))
    fusion_mode = str(config['fusion_mode']).lower()
    if fusion_mode not in ARMS:
        raise ValueError(f'MCL-PH pretraining arms are {ARMS}, got {fusion_mode!r}')
    if config.get('use_md200') is not False or config.get('third_task', 'none') != 'none':
        raise ValueError('the MCL-PH route has no MD200 and no fingerprint objective')
    if config['amp_dtype'] not in ('fp32', 'bf16'):
        raise ValueError('precision must be fp32 or bf16')
    cutoffs = tuple(float(value) for value in config.get('cutoffs', (2.0, 3.0, 4.0)))
    if cutoffs != (2.0, 3.0, 4.0):
        raise ValueError('the contract fixes the three expert cutoffs at 2/3/4 A')
    dense_updates = int(config.get('router_dense_updates', 500))
    top_k = int(config.get('router_top_k_afterwards', 2))
    if top_k != 2:
        raise ValueError('Top-1 routing is forbidden and the contract fixes Top-2')
    rank, world = int(os.environ.get('RANK', 0)), int(os.environ.get('WORLD_SIZE', 1))
    expected_world = config.get('expected_world_size')
    if expected_world is not None and int(world) != int(expected_world):
        raise ValueError(f'config requires world size {int(expected_world)}, got {int(world)}')
    device = (torch.device('cuda', int(os.environ.get('LOCAL_RANK', 0)))
              if torch.cuda.is_available() else torch.device('cpu'))
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    if world > 1:
        dist.init_process_group('nccl' if device.type == 'cuda' else 'gloo')
    output = Path(args.output)
    _OUTPUT[0] = output
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
    if rank == 0 and not Path(args.shared_new_init).is_file():
        build_shared_init(args.shared_new_init, dropout=float(config.get('dropout', 0.1)),
                          cutoffs=cutoffs, dense_updates=dense_updates)
    if world > 1:
        dist.barrier()
    statistics = load_geometric_statistics(args.statistics)
    source, frame = open_source(args.cohort_root, args.cache_root,
                               dual_static_root=args.dual_static_root)
    stages = _NullStages()
    # Set only after the training body and the export epilogue both finished, so
    # that the cleanup below never promotes a record belonging to a failed run.
    training_complete = False
    try:
        sample_index = load_sample_index_artifact(
            args.sample_index_artifact or config.get('sample_index_artifact'),
            args.sample_index_split or config.get('sample_index_split'))
        if sample_index is None:
            raise ValueError('the MCL-PH route trains on the frozen P_train subset')
        if len(source) != int(sample_index['payload'].get('pi1m_record_count', len(source))):
            raise ValueError('sample index base cohort count mismatch')
        expected_manifest = sample_index['payload'].get('pi1m_manifest_hash')
        if expected_manifest and expected_manifest != source.cohort['manifest_hash']:
            raise ValueError('sample index/base cohort manifest mismatch')
        source = IndexedFrozenDualSource(source, sample_index['indices'])
        micro, batch_size = int(config['microbatch']), int(config['global_batch'])
        if batch_size % (micro * world):
            raise ValueError('microbatch * accumulation * world must equal global_batch')
        accumulation = batch_size // (micro * world)
        trajectory_cache, trajectory_block = open_trajectory_cache(
            args.trajectory_cache, config=config, batch_size=batch_size,
            sample_index=sample_index, source=source)
        set_global_seed(int(config['seed']))
        base = MCLPHPretrainer(fusion_mode, dropout=float(config.get('dropout', 0.1)),
                               cutoffs=cutoffs, router_dense_updates=dense_updates,
                               collect_diagnostics=bool(args.diagnostics)).to(device)
        common = None
        common_path = args.common_init_artifact or config.get('common_init_artifact')
        if common_path:
            # Only the O8 tensors of the shared GLT dual state belong to the
            # MCL-PH initial state: the GLT encoder and the GLT heads are this
            # task's own modules, and the contract forbids strict-loading old
            # GLT keys into the new experts and decoders.  Both sides are
            # excluded by name so the copy is exactly the shared backbone.
            artifact_state = (torch.load(Path(common_path), map_location='cpu',
                                         weights_only=False).get('common_state_dict') or {})
            excluded = SHARED_INIT_EXCLUDE_NAMES(base, artifact_state)
            common = apply_common_initialization(base, common_path, exclude=excluded)
            covered = [name for name in base.state_dict() if name.startswith('encoder.o8.')]
            if not covered:
                raise ValueError('the O8 backbone was not covered by the common initialization')
            common['o8_tensor_count'] = len(covered)
            common['artifact_tensors_not_used'] = len(excluded) - (
                len(base.state_dict()) - len(covered))
        shared_init = apply_shared_init(base, args.shared_new_init)
        optimizer = torch.optim.AdamW(base.parameters(), lr=float(config['lr']),
                                      weight_decay=float(config['weight_decay']))
        module = (DistributedDataParallel(base, device_ids=[device.index] if device.type == 'cuda'
                                         else None, find_unused_parameters=True)
                  if world > 1 else base)
        start = 0
        identity = dict(
            config=config, world_size=world, sample_count=len(source),
            cohort_root=str(Path(args.cohort_root).resolve()),
            cohort_hash=source.cohort['manifest_hash'],
            cache_root=str(Path(args.cache_root).resolve()),
            main_bundle_hash=source.bundle.bundle_hash,
            dual_static_manifest_hash=source.static_cache.manifest_hash,
            sample_index_artifact_sha256=sample_index['sha256'],
            sample_index_split=sample_index['split'],
            statistics_sha256=sha256_file(args.statistics),
            statistics_samples=statistics['samples'],
            common_init_artifact_sha256=(common['sha256'] if common else None),
            shared_new_init_sha256=shared_init['sha256'],
            fusion_mode=fusion_mode, cutoffs=list(cutoffs),
            router_dense_updates=dense_updates, router_top_k=top_k,
            route='mcl_ph', third_task='none', use_md200=False,
            trajectory_cache=trajectory_block)
        ordered_keys = [key.hex() for key, _ in source.samples]
        resume_rng = None
        if args.resume:
            state = torch.load(args.resume, map_location='cpu', weights_only=False)
            if state['identity'] != identity or state['ordered_keys'] != ordered_keys:
                raise ValueError('resume data/config/world size differs')
            base.load_state_dict(state['model'], strict=True)
            optimizer.load_state_dict(state['optimizer'])
            start = int(state['step'])
            if state['next_position'] != start * batch_size or state['scheduler']['step'] != start:
                raise ValueError('resume scheduler/data position mismatch')
            resume_rng = state['rng'][rank]
        else:
            set_global_seed(int(config['seed']) + rank)
        if not 0 <= start < int(config['max_optimizer_steps']):
            raise ValueError('resume step must precede the configured training end')
        stop = int(config['max_optimizer_steps'])
        if args.stop_after_step:
            if not args.diagnostics:
                raise ValueError('--stop-after-step requires --diagnostics')
            if not start < int(args.stop_after_step) <= stop:
                raise ValueError('--stop-after-step must satisfy start < stop <= max_optimizer_steps')
            stop = int(args.stop_after_step)
        if trajectory_cache is not None:
            # Coverage is a property of the schedule, not of the data: the run may
            # only read positions that a *completed* shard actually holds.
            trajectory_cache.require_positions(stop * batch_size)
            trajectory_block = trajectory_cache.provenance(
                required_positions=stop * batch_size)
        if rank == 0:
            write_json(output / 'run.json', dict(identity=identity, command=sys.argv,
                                                 accumulation=accumulation, stop_after_step=stop,
                                                 trajectory_cache=trajectory_block))
            write_json(output / 'runtime.json', dict(
                status='RUNNING', architecture=base.encoder.architecture_name,
                command=sys.argv, config=config, identity=identity, rank=rank,
                world_size=world, device=str(device), prep_workers=int(args.prep_workers),
                diagnostics=bool(args.diagnostics), fusion_mode=fusion_mode,
                trajectory_cache=trajectory_block,
                parameter_count=int(sum(p.numel() for p in base.parameters())),
                started_at_monotonic=time.perf_counter()))
            digest, summary = _parameter_snapshot(base)
            write_json(output / 'step_0000.json', dict(
                step=0, note='initial values only; no optimizer update has run',
                parameter_digest=digest, parameters=summary,
                router_mode=base.encoder.branch.router.mode,
                common_initialization=common, shared_new_initialization=shared_init))
        stream = OrderedSampleStream(len(source), int(config['seed']))
        # Per-rank stage log and stall watchdog: the export epilogue is the last
        # thing a run does, so it is exactly the part that has to be observable
        # from outside when the process is killed.
        stages = StageLogger(output, rank, stall_seconds=float(
            os.environ.get('MCL_PH_STALL_SECONDS', StageLogger.DEFAULT_STALL_SECONDS)))
        stages.mark('loop', 'enter', steps=stop - start, start=start, stop=stop)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        prefetch = None
        if int(args.prep_workers) > 0:
            dataset = MCLPHMicrobatchStream(
                source, seed=int(config['seed']), world=world, rank=rank,
                microbatch=micro, accumulation=accumulation, start_step=start, max_steps=stop,
                sigma=float(config['noise_sigma']), ratio=float(config['atom_mask_ratio']),
                statistics=statistics, trajectory_cache=trajectory_cache)
            prefetch = iter(torch.utils.data.DataLoader(
                dataset, batch_size=None, num_workers=int(args.prep_workers),
                prefetch_factor=4, persistent_workers=False, pin_memory=False))
        if resume_rng is not None:
            restore_rng(resume_rng)
        for step in range(start, stop):
            step_number = step + 1
            step_started = time.perf_counter()
            stages.mark('step', 'enter', step=step_number)
            base.encoder.branch.router.configure(
                base.encoder.branch.routing_mode_for_step(step_number), step_number)
            prep_started = time.perf_counter()
            if prefetch is not None:
                prepared = [next(prefetch) for _ in range(accumulation)]
            else:
                prepared = []
                for offset in range(accumulation):
                    rows = []
                    for local in range(micro):
                        position = (step * batch_size + offset * world * micro
                                    + rank * micro + local)
                        index = stream.index_at(position)
                        rows.append(prepare_mcl_ph_sample(
                            *source[index], seed=int(config['seed']),
                            key=source.samples[index][0].hex(), position=position,
                            sigma=float(config['noise_sigma']),
                            ratio=float(config['atom_mask_ratio']),
                            static=source.static_for(index), statistics=statistics,
                            trajectory_override=(None if trajectory_cache is None
                                                 else trajectory_cache[position])))
                    prepared.append(mcl_ph_collate(rows))
            preparation_seconds = time.perf_counter() - prep_started
            lr = _scheduled_lr(step, config)
            for group in optimizer.param_groups:
                group['lr'] = lr
            optimizer.zero_grad(set_to_none=True)
            # The exact denominator of one optimizer update: the effective graph
            # count of each main task, summed over every microstep of this update
            # and every rank.  It is known before the backward passes, so the
            # gradient of the update is the gradient of
            # ``sum numerator / sum count`` over the whole update.
            update_counts = {'atom': 0, 'geometry': 0}
            for host_batch, host_labels in prepared:
                local_counts = effective_graph_counts(
                    host_batch.canonical_graph_index, host_labels,
                    int(host_batch.graph_available.numel()))['counts']
                update_counts['atom'] += local_counts['atom']
                update_counts['geometry'] += local_counts['geometry']
            denominators = reduce_update_denominators(update_counts, global_sum, device)
            totals = torch.zeros(3, device=device)
            counts = torch.zeros(2, device=device)
            target_counts = torch.zeros(5, device=device)
            diagnostics = None
            objective_statistics = None
            forward_started = time.perf_counter()
            for offset, (batch, labels) in enumerate(prepared):
                sync = module.no_sync() if world > 1 and offset + 1 < accumulation else nullcontext()
                with sync:
                    moved_data = batch.to(device, non_blocking=True)
                    moved_labels = move_labels(labels, device)
                    with torch.autocast(device.type, dtype=torch.bfloat16,
                                        enabled=config['amp_dtype'] == 'bf16'):
                        report = module(moved_data, moved_labels)
                        loss = base.objective(report, weights=(1.0, 1.0, BALANCE_WEIGHT),
                                              world_size=world, denominators=denominators,
                                              accumulation=len(prepared))
                    if not torch.isfinite(loss):
                        raise FloatingPointError('nonfinite MCL-PH pretraining loss')
                    loss.backward()
                if objective_statistics is None and offset == len(prepared) - 1:
                    objective_statistics = report.get('objective_statistics')
                if diagnostics is None and offset == len(prepared) - 1:
                    # The diagnostics describe the tensors the model consumed.
                    # ``batch.to(device)`` mutates the PyG store in place while
                    # the label dict stays on the host, so the two must never be
                    # mixed inside one expression.
                    diagnostics = _step_diagnostics(base, report, moved_data, moved_labels, world)
                totals += torch.stack([report['atom_sum'], report['geo_sum'],
                                       report['balance']]).detach()
                counts += torch.stack([report['atom_count'], report['geo_count']]).detach().float()
                target_counts += report['targets'].detach()
            forward_backward_seconds = time.perf_counter() - forward_started
            if world > 1:
                dist.all_reduce(totals)
                dist.all_reduce(counts)
                dist.all_reduce(target_counts)
            grad_norms = _gradient_groups(base) if args.diagnostics else None
            grad_total = float(torch.nn.utils.clip_grad_norm_(base.parameters(), 1.0,
                                                              error_if_nonfinite=True))
            optimizer.step()
            record = dict(step=step_number, rank=rank, lr=float(lr),
                          losses={'atom': float(totals[0] / counts[0].clamp_min(1)),
                                  'geometry': float(totals[1] / counts[1].clamp_min(1)),
                                  # The balance is defined per distributed
                                  # microstep and averaged over the accumulation;
                                  # the rank sum is divided out here because its
                                  # value is identical on every rank.
                                  'balance': float(totals[2] / (len(prepared)
                                                                * max(1, world)))},
                          objective_statistics=objective_statistics,
                          update_denominators=denominators,
                          valid_graphs={'atom': float(counts[0]), 'geometry': float(counts[1])},
                          target_counts=[float(value) for value in target_counts],
                          grad_total_preclip=grad_total,
                          grad_norms=grad_norms,
                          preparation_seconds=preparation_seconds,
                          forward_backward_seconds=forward_backward_seconds,
                          step_seconds=time.perf_counter() - step_started,
                          router_mode=base.encoder.branch.router.mode)
            if diagnostics is not None:
                record['diagnostics'] = diagnostics
            if base.last_diagnostics is not None:
                record['monitoring'] = base.last_diagnostics
            record['memory'] = memory_record()
            if torch.cuda.is_available():
                # The window above ends here; the next step's window starts now.
                torch.cuda.reset_peak_memory_stats()
            print(json.dumps(record, default=str), flush=True)
            if rank == 0 and (step_number in SAVED_STEPS or step_number in (start + 1, stop)):
                with (output / 'steps.jsonl').open('a', encoding='utf-8') as handle:
                    handle.write(json.dumps(record, sort_keys=True, default=str) + '\n')
            if step_number == stop:
                # The RNG gather is a collective: every rank must reach it, so it
                # stays outside the rank-0 guard that owns the file writes.
                stages.mark('rng_gather', 'enter', step=step_number)
                states = [None] * world
                if world > 1:
                    dist.all_gather_object(states, rng_state())
                else:
                    states[0] = rng_state()
                stages.mark('rng_gather', 'complete', step=step_number)
                if rank == 0:
                    stages.mark('resume_save', 'enter', step=step_number)
                    save_checkpoint(output / f'resume_{step_number:05d}.pt', dict(
                        identity=identity, ordered_keys=ordered_keys, step=step_number,
                        next_position=step_number * batch_size, model=base.state_dict(),
                        optimizer=optimizer.state_dict(), rng=states,
                        scheduler=dict(step=step_number, lr=float(lr))))
                    stages.mark('resume_save', 'complete', step=step_number)
                    if not args.no_deploy:
                        from src.modules.mcl_ph import deployment_package
                        stages.mark('deployment_package', 'enter', step=step_number)
                        package = deployment_package(
                            base.encoder, step_number, cutoffs=cutoffs,
                            router_dense_updates=dense_updates, router_top_k=top_k,
                            source={'cohort_hash': source.cohort['manifest_hash'],
                                    'statistics_sha256': sha256_file(args.statistics),
                                    'shared_new_init_sha256': shared_init['sha256']},
                            progress=stages.tensor_progress)
                        stages.mark('deployment_package', 'complete', step=step_number)
                        stages.mark('deploy_save', 'enter', step=step_number)
                        save_checkpoint(output / f'deploy_{step_number:05d}.pt', package)
                        stages.mark('deploy_save', 'complete', step=step_number)
            stages.mark('barrier', 'enter', step=step_number)
            if world > 1:
                dist.barrier()
            stages.mark('step', 'complete', step=step_number)
        if rank == 0:
            # Training and the export epilogue are done; the cleanup below and
            # this process's own exit are not.  The record says exactly that
            # much: a stall in the cleanup must leave a record that never
            # reached PASS instead of one that claims the run finished.
            runtime = json.loads((output / 'runtime.json').read_text(encoding='utf-8'))
            runtime.update(status='TRAINING_COMPLETE', completed_steps=int(stop - start),
                           finished_at_monotonic=time.perf_counter(),
                           export_complete=bool(not args.no_deploy),
                           cleanup='pending', main_returned=False, memory=memory_record())
            write_json(output / 'runtime.json', runtime)
        stages.mark('loop', 'complete', steps=stop - start)
        training_complete = True
    except BaseException as error:
        # Write the failure before the cleanup below: ``source.close()`` and
        # ``destroy_process_group()`` can both wait on other ranks, and a record
        # that only lands after a successful teardown is no record at all.
        stages.mark('failure', 'enter', error=type(error).__name__)
        write_failure_record(error, rank, phase='before_cleanup')
        raise
    finally:
        stages.mark('cleanup', 'enter')
        source.close()
        stages.mark('cleanup', 'source_closed')
        if dist.is_initialized():
            dist.destroy_process_group()
        stages.mark('cleanup', 'process_group_destroyed')
        stages.mark('cleanup', 'complete')
        stages.close()
        if training_complete and rank == 0:
            # Cleanup ran to its end.  Still not PASS: nothing inside this
            # process can witness its own exit code, so the record stops one
            # level short and waits for the launcher's evidence.
            runtime = json.loads((output / 'runtime.json').read_text(encoding='utf-8'))
            runtime.update(cleanup='complete', cleanup_finished_at_monotonic=time.perf_counter())
            write_json(output / 'runtime.json', runtime)


def finalize_runtime_record():
    """Write the PASS record once ``main()`` returned and its cleanup completed.

    A process cannot observe its own exit code, so this record claims only what
    is true from the inside: the training body finished, the export finished and
    the cleanup finished.  The exit code itself is the launcher's evidence, and
    the launcher writes the completion marker on that basis.
    """
    output = _OUTPUT[0]
    if os.environ.get('RANK', '0') != '0' or output is None or not Path(output).is_dir():
        return
    path = Path(output) / 'runtime.json'
    try:
        runtime = json.loads(path.read_text(encoding='utf-8'))
        if runtime.get('cleanup') != 'complete':
            # Training ended without a completed cleanup: leave the record at
            # ``TRAINING_COMPLETE`` rather than promoting it to PASS.
            return
        runtime.update(status='PASS', main_returned=True, process_exit='observed_by_launcher')
        write_json(path, runtime)
    except (OSError, ValueError) as error:
        print(json.dumps({'status': 'RUNTIME_RECORD_FAILED',
                          'error': f'{type(error).__name__}: {error}'}), flush=True)


def _scheduled_lr(step, config):
    from src.training.glt_dual_runtime import scheduled_lr
    return scheduled_lr(step, lr=float(config['lr']), warmup_steps=int(config['warmup_steps']),
                        schedule_total_steps=int(config['schedule_total_steps']),
                        end_lr=float(config['end_lr']))


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        # A failure that never entered the training body (a bad argument, a
        # missing file) has no stage log yet, so this handler still owns it; a
        # failure inside the body was already recorded before the cleanup, and
        # the ``phase`` field says which of the two a record came from.
        write_failure_record(error, int(os.environ.get('RANK', 0)),
                             phase='at_exit')
        raise
    else:
        # Reached only when ``main()`` returned, and therefore only after the
        # cleanup in its ``finally`` block.  What is still missing at this point
        # is the process's own exit code, which the launcher observes.
        finalize_runtime_record()
