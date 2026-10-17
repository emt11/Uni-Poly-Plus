#!/usr/bin/env python3
"""Atom/geometry pretraining with optional FGR/ALIGN; never builds FP heads."""
import argparse
import math
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from src.dataset.glt_dual_pretrain import prepare_pretrain_sample, pretrain_collate
from src.modules.glt_dual_pretrain import (DualPretrainer, global_objective,
                                            deployment_package,
                                            alignment_local_anchor_count)
from src.training.glt_dual_runtime import (require_tmux, open_source, save_checkpoint, write_json,
                                         rng_state, restore_rng, scheduled_lr, move_labels,
                                         OrderedSampleStream, RankMicrobatchStream,
                                         IndexedFrozenDualSource, load_sample_index_artifact,
                                         apply_common_initialization, pretrain_source_fields, prepare_source_sample,
                                         pretrain_loader)
from src.utils import set_global_seed


def write_consumption(path, *, attempted, completed, status):
    """Publish one rank-0 ledger atomically for the concurrently reading controller."""
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    write_json(temporary, dict(starts=1, attempted_updates=attempted,
                               completed_updates=completed, status=status))
    os.replace(temporary, path)


def _module_grad_norms(model):
    """Gradient norms per module group, measured before clipping."""

    groups = {
        'o8_2d_encoder': model.encoder.o8,
        'glt_3d_encoder': model.encoder.glt,
        'length_head': model.length_head,
        'angle_head': model.angle_head,
        'fgr_head': model.fgr_head,
        'align_proj2': model.align_proj2,
        'align_proj3': model.align_proj3,
        'atom_head': model.atom_head,
    }
    seen, report = set(), {}
    for name, module in groups.items():
        if module is None:
            report[name] = 0.0
            continue
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
    for name, value, original in zip(('chem', 'geometry', 'third'), losses,
                                     expected['losses']):
        if abs(value - float(original)) > rtol * max(1e-6, abs(float(original))):
            print(json.dumps({'reference_mismatch': name, 'step': step,
                              'observed': value, 'original': float(original)}), flush=True)


def _diagnostic_update(step, start, every, save_steps):
    """Return whether detailed observations are needed for this update."""

    return (step == start + 1 or step % every == 0 or step in save_steps)


def _time_summary(values):
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return {"count": 0, "mean_seconds": None, "median_seconds": None, "p95_seconds": None}
    return {
        "count": len(ordered),
        "mean_seconds": float(statistics.fmean(ordered)),
        "median_seconds": float(statistics.median(ordered)),
        "p95_seconds": float(ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))]),
    }


def _third_stream_signature(prepared, third_task):
    """Hash task-specific identities without changing the public RNG stream."""

    digest = hashlib.sha256()
    for _, labels in prepared:
        if third_task == 'fgr':
            for name in ('fgr_pair_index', 'fgr_spd', 'fgr_graph'):
                value = labels[name].detach().cpu().contiguous()
                digest.update(name.encode('utf-8'))
                digest.update(str(tuple(value.shape)).encode('ascii'))
                digest.update(value.numpy().tobytes())
        elif third_task == 'align':
            payload = list(zip(
                [str(value) for value in labels['align_identity']],
                [bool(value) for value in labels['align_valid'].tolist()],
            ))
            digest.update(json.dumps(payload, separators=(',', ':'), sort_keys=False).encode('utf-8'))
    return digest.hexdigest()


def alignment_weight_at(config,step):
    if not config.get('align_enabled'):return 0.
    weight=float(config.get('align_weight',.1));warmup=int(config.get('align_warmup_steps',1000))
    if not math.isfinite(weight) or weight<=0 or warmup<=0:raise ValueError('invalid ALIGN weight/warmup')
    return weight*min(float(step)/warmup,1.)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'cohort-root', 'cache-root', 'output'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--dual-static-root',
                        help='training-ready dual_static_v1 artifact')
    parser.add_argument('--pretrain-target-root',
                        help='optional BRICS-only pretrain_targets_v1 artifact')
    parser.add_argument('--sample-index-artifact',
                        help='read-only pretrain source-index split artifact')
    parser.add_argument('--sample-index-split', choices=('train', 'validation', 'fixed_validation'),
                        help='logical split exposed by --sample-index-artifact')
    parser.add_argument('--common-init-artifact',
                        help='frozen common encoder/head initialization artifact')
    parser.add_argument('--resume')
    parser.add_argument('--prep-workers', type=int, default=0,
                        help='DataLoader workers per rank for input prefetch '
                             '(0 = inline preparation, the exact original path)')
    parser.add_argument('--lmdb-readahead', action='store_true',
                        help='normal kernel prefetch for contiguous direct LMDB values')
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
    parser.add_argument('--third-task', choices=('none', 'fgr', 'align'),
                        help='optional objective; defaults to config third_task or none')
    parser.add_argument('--fgr-mu', type=float,
                        help='P_train FGR log-distance mean (required for a formal FGR run)')
    parser.add_argument('--fgr-sigma', type=float,
                        help='P_train FGR log-distance scale (required positive)')
    parser.add_argument('--fgr-max-pairs', type=int,
                        help='maximum deterministic FGR pairs per graph (default 32)')
    parser.add_argument('--align-temperature', type=float,
                        help='fixed graph ALIGN temperature (default 0.1)')
    parser.add_argument('--reference-log',
                        help='original run log for the initial-10-step cross-check')
    parser.add_argument('--timing', action='store_true',
                        help='optional bounded phase timing; adds synchronization only for the timing run')
    parser.add_argument('--diagnostics-every', type=int, default=1,
                        help='collect detailed diagnostics on the first update, this interval, and save steps')
    parser.add_argument('--benchmark-steps', type=int, default=0,
                        help='bounded steady-state benchmark updates; no formal checkpoints/deploy are written')
    parser.add_argument('--benchmark-warmup', type=int, default=0,
                        help='benchmark updates to exclude before the measured window')
    args = parser.parse_args()
    require_tmux()
    config = json.loads(Path(args.config).read_text(encoding='utf-8'))
    if config.get('ua2d_pt_revision'):
        from src.training.ua2d_pt20k import validate_config as validate_ua2d_pt
        validate_ua2d_pt(config)
        if os.environ.get('CUDA_VISIBLE_DEVICES') != '2,3':
            raise ValueError('P2 requires physical GPU2,3')
        if (args.resume or args.diagnostics or args.benchmark_steps or args.stop_after_step
                or args.no_deploy or args.diagnostic_save_steps or args.timing
                or args.common_init_artifact or args.reference_log or args.third_task
                or args.pretrain_target_root or args.sample_index_artifact or args.sample_index_split
                or args.geometry_head_norm or args.benchmark_warmup or args.diagnostics_every != 1
                or args.fgr_mu is not None or args.fgr_sigma is not None
                or args.fgr_max_pairs is not None or args.align_temperature is not None
                or args.dual_static_root or args.prep_workers != 8 or not args.lmdb_readahead):
            raise ValueError('P2 runtime/scientific override forbidden')
        from src.training.ua2d_pt20k import PI as ua2d_pi
        if Path(args.cohort_root).resolve() != ua2d_pi or Path(args.cache_root).resolve() != ua2d_pi:
            raise ValueError('P2 must reuse the original direct training cohort')
    if config.get('t3_revision'):
        from scripts.run_fce_tune_t3_unit import validate_pt_config,PI
        validate_pt_config(config)
        if os.environ.get('CUDA_VISIBLE_DEVICES')!='2,3':raise ValueError('T3 PT requires physical GPU2,3')
        if (args.prep_workers!=8 or not args.lmdb_readahead or args.diagnostics or args.no_deploy
                or args.stop_after_step or args.diagnostic_save_steps or args.benchmark_steps or args.timing
                or args.common_init_artifact or args.reference_log or args.third_task or args.fgr_mu is not None
                or args.fgr_sigma is not None or args.fgr_max_pairs is not None or args.align_temperature is not None
                or args.pretrain_target_root or args.sample_index_artifact or args.sample_index_split
                or Path(args.cohort_root).resolve()!=PI or Path(args.cache_root).resolve()!=PI
                or (args.dual_static_root and Path(args.dual_static_root).resolve()!=PI)):
            raise ValueError('T3 PT runtime/scientific override forbidden')
    if config.get('t3b_revision'):
        from scripts.run_fce_tune_t3b_unit import validate_pt_config,PI
        validate_pt_config(config)
        if os.environ.get('CUDA_VISIBLE_DEVICES')!='2,3':raise ValueError('T3b PT requires physical GPU2,3')
        if (args.prep_workers!=8 or not args.lmdb_readahead or args.diagnostics or args.no_deploy
                or args.resume or args.benchmark_warmup or args.stop_after_step or args.diagnostic_save_steps or args.benchmark_steps or args.timing
                or args.common_init_artifact or args.reference_log or args.third_task or args.fgr_mu is not None
                or args.fgr_sigma is not None or args.fgr_max_pairs is not None or args.align_temperature is not None
                or args.pretrain_target_root or args.sample_index_artifact or args.sample_index_split
                or Path(args.cohort_root).resolve()!=PI or Path(args.cache_root).resolve()!=PI
                or (args.dual_static_root and Path(args.dual_static_root).resolve()!=PI)):
            raise ValueError('T3b PT runtime/scientific override forbidden')
    r10=bool(config.get('r10_revision'))
    if r10:
        from src.modules.glt_r10 import validate_pt_config
        validate_pt_config(config)
        if os.environ.get('CUDA_VISIBLE_DEVICES')!='2,3':raise ValueError('r10 PT requires physical GPU2,3')
    third_task = str(args.third_task or config.get('third_task', 'none')).lower()
    fgr_mu = float(args.fgr_mu if args.fgr_mu is not None else config.get('fgr_mu', 0.0))
    fgr_sigma = float(args.fgr_sigma if args.fgr_sigma is not None else config.get('fgr_sigma', 1.0))
    fgr_max_pairs = int(args.fgr_max_pairs if args.fgr_max_pairs is not None
                        else config.get('fgr_max_pairs', 32))
    align_temperature = float(args.align_temperature if args.align_temperature is not None
                              else config.get('align_temperature', 0.1))
    if third_task not in {'none', 'fgr', 'align'}:
        raise ValueError('unsupported third task')
    if not torch.isfinite(torch.tensor(fgr_mu)) or not torch.isfinite(torch.tensor(fgr_sigma)) or fgr_sigma <= 0:
        raise ValueError('FGR normalization must be finite with positive sigma')
    if fgr_max_pairs <= 0:
        raise ValueError('FGR max pairs must be positive')
    if not torch.isfinite(torch.tensor(align_temperature)) or align_temperature <= 0:
        raise ValueError('ALIGN temperature must be finite and positive')
    atom_target = str(config.get('atom_target', 'element')).lower()
    if atom_target not in {'element', 'environment'}:
        raise ValueError('unsupported atom pretraining target')
    env_vocab = None
    env_vocab_path = config.get('env_vocab')
    if atom_target == 'environment':
        if not env_vocab_path:
            raise ValueError('environment atom target requires config env_vocab')
        vocab_payload = json.loads(Path(env_vocab_path).read_text(encoding='utf-8'))
        if vocab_payload.get('categories', {}).get('<unk>') is None:
            raise ValueError('environment vocabulary is missing categories/<unk>')
        env_vocab = vocab_payload['categories']
    torsion_mode = config.get('torsion_mode')
    if torsion_mode is not None and str(torsion_mode).lower() not in {'on', 'off'}:
        raise ValueError('torsion_mode must be on or off when present')
    torsion_mode = str(torsion_mode).lower() if torsion_mode else None
    from src.modules.glt_dual import torsion_options
    torsion_options(config)
    if config.get('torsion_revision') and not r10:
        from src.modules.glt_dual import s4_resource_options
        s4_resource_options(config)
        if os.environ.get('CUDA_VISIBLE_DEVICES')!='2,3':raise ValueError('S4 PT requires physical GPU2,3')
    if config.get('reference_revision'):
        from scripts.run_glt_pt_matched_reference_unit import validate_config
        validate_config(config)
        if os.environ.get('CUDA_VISIBLE_DEVICES')!='2,3':raise ValueError('S5 PT requires physical GPU2,3')
    new_cycle = config.get('fusion_revision') == 'glt_fusion_r1'
    if (not new_cycle and config.get('use_md200') is not False) or config['amp_dtype'] not in ('fp32', 'bf16'):
        raise ValueError('dual route requires no MD200 and fp32/bf16 precision')
    if new_cycle and config.get('use_md200') and not config.get('md200_root'):
        raise ValueError('MD-on arm requires the explicit MD200 sidecar')
    if args.diagnostics_every <= 0:
        raise ValueError('--diagnostics-every must be a positive integer')
    if not args.diagnostics and args.diagnostics_every != 1:
        raise ValueError('--diagnostics-every requires --diagnostics')
    if args.benchmark_steps < 0 or args.benchmark_warmup < 0:
        raise ValueError('--benchmark-steps and --benchmark-warmup must be non-negative')
    if args.benchmark_steps and args.timing:
        raise ValueError('--benchmark-steps cannot be combined with synchronized --timing')
    if (min(config['microbatch'], config['global_batch'], config['max_optimizer_steps'], config['save_every']) <= 0
            or config['warmup_steps'] < 0
            or config['schedule_total_steps'] < max(config['max_optimizer_steps'], config['warmup_steps'])
            or len(config['loss_weights']) != 3):
        raise ValueError('invalid training budget, schedule or objective weights')
    rank, world = int(os.environ.get('RANK', 0)), int(os.environ.get('WORLD_SIZE', 1))
    expected_world = config.get('expected_world_size')
    if expected_world is not None and int(world) != int(expected_world):
        raise ValueError(f'config requires world size {int(expected_world)}, got {world}')
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
        # The current objectives do not require the legacy Morgan target cache.
        pretrain_target_root=None,
        readonly_readahead=args.lmdb_readahead,
        verify_key_digest=not bool(config.get('opt_mode')),
        md200_root=config.get('md200_root'),
    )
    try:
        from src.training.glt_dual_runtime import apply_geometry_quality
        source=apply_geometry_quality(source,config.get('quality_root'))
        if r10 and config.get('qc_weight_root'):
            from src.dataset.glt_r10 import attach_qc_source
            source=attach_qc_source(source,config['qc_weight_root'])
        sample_index_path = args.sample_index_artifact or config.get('sample_index_artifact')
        sample_index_split = args.sample_index_split or config.get('sample_index_split')
        sample_index = load_sample_index_artifact(sample_index_path, sample_index_split)
        is_direct = bool(sample_index and sample_index.get('direct_contract'))
        if sample_index is not None:
            if len(source) != int(sample_index['payload'].get('pi1m_record_count', len(source))):
                raise ValueError('sample index base cohort count mismatch')
            expected_manifest = sample_index['payload'].get('pi1m_manifest_hash')
            if expected_manifest and expected_manifest != source.cohort['manifest_hash']:
                raise ValueError('sample index/base cohort manifest mismatch')
            source = IndexedFrozenDualSource(source, sample_index['indices'])
        micro, batch_size = config['microbatch'], config['global_batch']
        if batch_size % (micro * world):
            raise ValueError('microbatch * accumulation * world must equal global_batch')
        accumulation = batch_size // (micro * world)
        if config.get('torsion_revision') and config['accumulation']!=accumulation:raise ValueError('S4 accumulation/runtime mismatch')
        if config.get('reference_revision') and config['accumulation']!=accumulation:raise ValueError('S5 accumulation/runtime mismatch')
        if config.get('t3_revision'):
            from scripts.run_fce_tune_t3_unit import validate_pt_runtime
            validate_pt_runtime(config,world,accumulation)
        if config.get('t3b_revision'):
            from scripts.run_fce_tune_t3b_unit import validate_pt_runtime
            validate_pt_runtime(config,world,accumulation)
        set_global_seed(config['seed'])
        from src.modules.glt_dual import locality_options
        base = DualPretrainer(config['fusion_mode'], **locality_options(config),
                              collect_diagnostics=bool(args.diagnostics),
                              geometry_head_norm=bool(args.geometry_head_norm or config.get('geometry_head_norm', False)),
                              third_task=third_task, fgr_mu=fgr_mu,
                              fgr_sigma=fgr_sigma,
                              align_temperature=align_temperature,
                              atom_target=atom_target,
                              env_vocab_size=(len(env_vocab) if env_vocab is not None else None),
                              torsion=bool(config.get('torsion_modules',torsion_mode is not None)), torsion_revision=config.get('torsion_revision'), opt_mode=config.get('opt_mode'),
                              fusion_revision=config.get('fusion_revision'), use_md200=config.get('use_md200', False),
                              atom_ce_input=config.get('atom_ce_input', 'raw'),
                              align_enabled=config.get('align_enabled', False), local_enabled=config.get('local_enabled', False),
                              geometry_scales=config.get('geometry_scales'),
                              fusion_gate=config.get("fusion_gate","none"), quality_root=config.get("quality_root"),
                              align_weight=config.get("align_weight",.1), align_warmup_steps=config.get("align_warmup_steps",1000),
                              align_revision=config.get("align_revision"),align_init_seed=config.get("align_init_seed"),align_direction=config.get('align_direction','symmetric'),
                              **({'nonbonded_revision':config['nonbonded_revision']} if r10 else {}),
                              **{key:config[key] for key in ('geometry_revision','geometry_mode','angle_value_enabled','readout') if key in config}).to(device)
        if r10:
            from src.modules.glt_r10 import configure_runtime
            configure_runtime(base.encoder)
        if config.get('initial_state'):
            initial = torch.load(config['initial_state'], map_location='cpu', weights_only=False)
            if initial['opt_mode'] != config.get('opt_mode') or initial.get('step')!=0:raise ValueError('step0 mode/step mismatch')
            if config.get('geometry_revision') and initial.get('config')!=config:raise ValueError('step0 config mismatch')
            for name,value in initial['model'].items():
                target=base.state_dict().get(name)
                if target is None or value.shape!=target.shape or value.dtype!=target.dtype or not torch.isfinite(value).all():raise ValueError('step0 tensor mismatch '+name)
            base.load_state_dict(initial['model'], strict=True)
        common_init = None
        common_init_path = args.common_init_artifact or config.get('common_init_artifact')
        if common_init_path:
            # The environment output head has a different output space; every
            # other shared parameter must come from the same step-0 state.
            exclude = ('atom_head.head.weight', 'atom_head.head.bias') if atom_target == 'environment' else ()
            common_init = apply_common_initialization(base, common_init_path, exclude=exclude, direct=is_direct)
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
                                                       if source.target_cache is not None else None),
                        third_task=third_task, fgr_mu=fgr_mu,
                        fgr_sigma=fgr_sigma, fgr_max_pairs=fgr_max_pairs,
                        align_temperature=align_temperature,
                        atom_target=atom_target,
                        atom_target_vocab_size=(len(env_vocab) if env_vocab is not None else None),
                        torsion_mode=torsion_mode,
                        env_vocab_path=(str(Path(env_vocab_path).resolve()) if env_vocab_path else None),
                        sample_index_artifact_sha256=(sample_index['sha256'] if sample_index else None),
                        sample_index_split=(sample_index['split'] if sample_index else None),
                        common_init_artifact_sha256=(common_init['sha256'] if common_init else None))
        # Exact ordered identities, without adding a separate cache schema.
        bindings = None
        if is_direct:
            from src.training.mcl_ph_direct_contract import assert_actual_equal
        if is_direct and not config.get('opt_mode'):
            from src.training.mcl_ph_direct_contract import direct_binding, direct_identity
            bindings = direct_binding(source, common_init=common_init_path, seed=config['seed'],
                                      total_positions=int(config['max_optimizer_steps']) * batch_size)
            identity = direct_identity(identity)
        ordered_keys = [key.hex() for key, _ in source.samples]
        resume_rng = None
        if args.resume:
            state = torch.load(args.resume, map_location='cpu', weights_only=False)
            if state['identity'] != identity or state['ordered_keys'] != ordered_keys:
                raise ValueError('resume data/config/world size differs')
            if is_direct:
                assert_actual_equal(state.get('direct_binding'), bindings)
            base.load_state_dict(state['model'], strict=True)
            optimizer.load_state_dict(state['optimizer'])
            if is_direct:
                from src.training.mcl_ph_direct_contract import verify_restored_checkpoint
                verify_restored_checkpoint(base, optimizer, state, batch_size=batch_size)
            start = state['step']
            if state['next_position'] != start * batch_size or state['scheduler']['step'] != start:
                raise ValueError('resume scheduler/data position mismatch')
            # Restore after constructing the optional DataLoader iterator below.
            # Its dedicated generator keeps setup draws outside model RNG;
            # restoring here also protects historical checkpoints.
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
            if args.benchmark_steps:
                raise ValueError('--stop-after-step cannot be combined with --benchmark-steps')
            if not start < int(args.stop_after_step) <= config['max_optimizer_steps']:
                raise ValueError('--stop-after-step must satisfy start < stop <= config.max_optimizer_steps')
            stop = int(args.stop_after_step)
        if args.benchmark_steps:
            remaining = config['max_optimizer_steps'] - start
            if args.benchmark_steps > remaining:
                raise ValueError('--benchmark-steps exceeds the configured remaining updates')
            if args.benchmark_warmup >= args.benchmark_steps:
                raise ValueError('--benchmark-warmup must satisfy 0 <= warmup < benchmark-steps')
            stop = start + args.benchmark_steps
        save_steps = sorted({int(value) for value in args.diagnostic_save_steps})
        for value in save_steps:
            if not start < value <= stop:
                raise ValueError('--diagnostic-save-steps must lie inside this run')
        if rank == 0:
            write_json(output / 'run.json', dict(identity=identity, command=sys.argv,
                accumulation=accumulation, diagnostics=bool(args.diagnostics),
                geometry_head_norm=bool(args.geometry_head_norm or config.get('geometry_head_norm', False)),
                stop_after_step=stop, diagnostic_save_steps=save_steps,
                diagnostics_every=int(args.diagnostics_every),
                benchmark_steps=int(args.benchmark_steps),
                benchmark_warmup=int(args.benchmark_warmup),
                benchmark_mode=bool(args.benchmark_steps), third_task=third_task,
                fgr_mu=fgr_mu, fgr_sigma=fgr_sigma, fgr_max_pairs=fgr_max_pairs,
                align_temperature=align_temperature))
            write_json(output / 'runtime.json', dict(
                status='RUNNING', command=sys.argv, config=config, identity=identity,
                rank=rank, world_size=world, device=str(device), prep_workers=int(args.prep_workers),
                diagnostics=bool(args.diagnostics), diagnostics_every=int(args.diagnostics_every),
                benchmark_steps=int(args.benchmark_steps), benchmark_warmup=int(args.benchmark_warmup),
                third_task=third_task, fgr_mu=fgr_mu, fgr_sigma=fgr_sigma,
                fgr_max_pairs=fgr_max_pairs, align_temperature=align_temperature,
                atom_target=atom_target, torsion_mode=torsion_mode,
                started_at_monotonic=time.perf_counter()))
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
                third_task=third_task, fgr_mu=fgr_mu, fgr_sigma=fgr_sigma,
                fgr_max_pairs=fgr_max_pairs,
                atom_target=atom_target, env_vocab=env_vocab,
                torsion_mode=(None if config.get('geometry_revision') and not config.get('torsion_modules') else torsion_mode), torsion_revision=config.get('torsion_revision'), opt_mode=config.get('opt_mode'), geometry_mode=config.get('geometry_mode','finite_internal'),nonbonded_revision=config.get('nonbonded_revision'),
            )
            prefetch = iter(pretrain_loader(
                dataset, workers=args.prep_workers, seed=config['seed'], rank=rank))
        if resume_rng is not None:
            restore_rng(resume_rng)
            if is_direct:
                assert_actual_equal(rng_state(), resume_rng, 'restored.rng')
        reference = _reference_steps(args.reference_log) if args.reference_log else {}
        diagnostics_path = output / 'diagnostics_steps.jsonl'
        timing_path = output / 'timing_steps.jsonl'
        dense_lo = min(save_steps) - 60 if save_steps else None
        benchmark_mode = bool(args.benchmark_steps)
        benchmark_times, benchmark_graphs = [], []
        benchmark_window_started = None
        if benchmark_mode:
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            if world > 1:
                dist.barrier()
        rank_record_path = (output / f'records_rank{rank}.jsonl'
                            if args.diagnostics or args.timing or benchmark_mode or config.get('opt_mode') else None)
        for step in range(start, stop):
            step_number = step + 1
            if new_cycle and rank == 0:
                write_consumption(output / 'consumption.json', attempted=step_number,
                                  completed=step, status='RUNNING')
            step_started = time.perf_counter()
            # All ranks finish warmup before the first timed update.  The
            # barrier is deliberately at the transition, rather than relying
            # on wall-clock sleep, so the benchmark window is deterministic.
            if benchmark_mode and step_number == start + args.benchmark_warmup + 1:
                if device.type == 'cuda':
                    torch.cuda.synchronize(device)
                if world > 1:
                    dist.barrier()
                benchmark_window_started = time.perf_counter()
                step_started = time.perf_counter()
            timing = {}
            diag_this_step = bool(
                args.diagnostics and _diagnostic_update(
                    step_number, start, args.diagnostics_every, save_steps))
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
                        rows.append(prepare_source_sample(
                            source, index, seed=config['seed'],
                            key=source.samples[index][0].hex(), position=position,
                            sigma=config['noise_sigma'], ratio=config['atom_mask_ratio'],
                            third_task=third_task, fgr_mu=fgr_mu,
                            fgr_sigma=fgr_sigma, fgr_max_pairs=fgr_max_pairs,
                            atom_target=atom_target, env_vocab=env_vocab,
                            torsion_mode=(None if config.get('geometry_revision') and not config.get('torsion_modules') else torsion_mode), torsion_revision=config.get('torsion_revision'), opt_mode=config.get('opt_mode'), geometry_mode=config.get('geometry_mode','finite_internal'),nonbonded_revision=config.get('nonbonded_revision')))
                    prepared.append(pretrain_collate(rows))
            if config.get('torsion_revision'):
                torsion_counts=sum((data.torsion_counts.sum(0) for data,_ in prepared),torch.zeros(6,dtype=torch.long)).tolist()
            timing['preparation_seconds'] = time.perf_counter() - prep_started
            counts = torch.zeros(3, device=device)
            extra_counts = torch.zeros(2, device=device)
            for data, labels in prepared:
                masked = torch.bincount(data.canonical_graph_index[labels['atom_mask']], minlength=data.graph_available.numel())
                centers = torch.bincount(data.bond_batch[(data.bond_length_weight>0) if config.get('geometry_revision') else data.bond_center], minlength=data.graph_available.numel())
                if third_task == 'fgr':
                    third_count = (labels['fgr_graph_valid'].to(device)
                                   & data.geometry_valid.to(device).bool()).sum()
                elif third_task == 'align':
                    third_count = alignment_local_anchor_count(
                        labels['align_identity'], labels['align_valid'], device=device)
                else:
                    third_count = torch.tensor(0., device=device)
                geo_valid=(centers > 0) & data.geometry_valid
                geo_count=(labels['geometry_quality_weight'][geo_valid].sum() if config.get('qc_weight_root') else geo_valid.sum())
                counts += torch.stack([(masked > 0).sum().to(device),
                                       geo_count.to(device),
                                       third_count.to(device)])
                if new_cycle:
                    if config.get('align_enabled'):
                        extra_counts[0] += alignment_local_anchor_count(
                            labels['align_identity'], labels['align_valid'], device=device)
                    if config.get('local_enabled'):
                        incident_graphs = data.canonical_graph_index[data.incidence_index[1]].unique()
                        extra_counts[1] += data.geometry_valid[incident_graphs].sum().to(device)
            if world > 1:
                dist.all_reduce(counts)
                if new_cycle:
                    dist.all_reduce(extra_counts)
            lr = scheduled_lr(step, **{key: config[key] for key in ('lr', 'warmup_steps', 'schedule_total_steps', 'end_lr')})
            for group in optimizer.param_groups:
                group['lr'] = lr
            optimizer.zero_grad(set_to_none=True)
            totals = torch.zeros(3, device=device)
            extra_totals = torch.zeros(2, device=device)
            raw_geometry_totals = torch.zeros(2, device=device)
            target_counts = torch.zeros(4, device=device)
            fallbacks, reasons = 0, {}
            h2d_seconds = forward_backward_seconds = optimizer_seconds = 0.0
            objective_weights = list(config['loss_weights'])
            if third_task == 'align':
                objective_weights[2] = 0.1 * min(float(step_number) / 1000., 1.)
            if new_cycle and third_task == 'fgr':
                objective_weights[2] = 0.1 * min(float(step_number) / 1000., 1.)
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
                        result = module(moved_data, moved_labels,
                                        collect_diagnostics=diag_this_step)
                        loss = global_objective(result['sums'], counts, world, objective_weights,fractional_counts=bool(config.get('qc_weight_root')))
                        if new_cycle:
                            extra_weights = [alignment_weight_at(config,step_number),
                                             0.05 if config.get('local_enabled') else 0.]
                            extra_weights = [extra_weights[0],extra_weights[1]*min(float(step_number)/1000.,1.)]
                            loss = loss + global_objective(result['extra_sums'], extra_counts, world, extra_weights)
                    if not torch.isfinite(loss):
                        raise FloatingPointError('nonfinite pretraining loss')
                    loss.backward()
                    if args.timing and device.type == 'cuda':
                        torch.cuda.synchronize(device)
                    forward_backward_seconds += time.perf_counter() - forward_backward_started
                totals += result['sums'].detach()
                if new_cycle:
                    extra_totals += result['extra_sums'].detach()
                    raw_geometry_totals += result['raw_geometry_sums'].detach()
                target_counts += result['targets'].detach()
            if args.timing and device.type == 'cuda':
                torch.cuda.synchronize(device)
            optimizer_started = time.perf_counter()
            grad_total_preclip = float(torch.nn.utils.clip_grad_norm_(
                base.parameters(), 1., error_if_nonfinite=True))
            optimizer.step()
            if new_cycle and rank == 0:
                write_consumption(output / 'consumption.json', attempted=step_number,
                                  completed=step_number, status='RUNNING')
            if args.timing and device.type == 'cuda':
                torch.cuda.synchronize(device)
            optimizer_seconds = time.perf_counter() - optimizer_started
            if world > 1:
                dist.all_reduce(totals)
                dist.all_reduce(target_counts)
                if new_cycle:
                    dist.all_reduce(extra_totals)
                    dist.all_reduce(raw_geometry_totals)
            record = dict(step=step + 1, rank=rank, lr=float(lr),
                losses=(totals / counts.clamp_min(1)).tolist(), valid_graphs=counts.tolist(),
                target_counts=target_counts.tolist(), local_fallbacks=fallbacks, local_skip_reasons=reasons,
                third_stream_signature=(None if is_direct else _third_stream_signature(prepared, third_task)),
                third_task=third_task)
            if config.get('torsion_revision'):
                record['local_torsion_counts']=torsion_counts
                record['torsion_count_scope']='rank presentations: quadruplets,valid,graphs_with,valid_graphs_without,degenerate,seam_paths'
            if new_cycle:
                record.update(extra_losses=(extra_totals/extra_counts.clamp_min(1)).tolist(),
                              extra_valid_graphs=extra_counts.tolist(),
                              raw_geometry_losses=(raw_geometry_totals/counts[1].clamp_min(1)).tolist(),
                              extra_weights=extra_weights,extra_weighted_contributions=((extra_totals/extra_counts.clamp_min(1))*torch.tensor(extra_weights,device=device)).tolist())
            if config.get('geometry_revision') and rank == 0:
                record['grad_norms'] = _module_grad_norms(base)
                record['grad_total_preclip'] = grad_total_preclip
            record['cpu_observed_timing'] = dict(
                preparation_seconds=timing['preparation_seconds'],
                step_seconds=time.perf_counter() - step_started,
                scope='host wall time without additional CUDA synchronization')
            if args.timing:
                timing.update(h2d_seconds=h2d_seconds,
                              forward_backward_seconds=forward_backward_seconds,
                              optimizer_seconds=optimizer_seconds,
                              step_seconds=time.perf_counter() - step_started)
                record['timing'] = timing
            print(json.dumps(record), flush=True)
            if rank_record_path is not None:
                with rank_record_path.open('a', encoding='utf-8') as handle:
                    handle.write(json.dumps(record, sort_keys=True) + '\n')
            if args.timing and rank == 0:
                with timing_path.open('a', encoding='utf-8') as handle:
                    handle.write(json.dumps(record, sort_keys=True) + '\n')
            if diag_this_step and rank == 0 and base.last_diagnostics is not None:
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
            if config.get('geometry_revision') and step_number in (1000,2500,5000):
                if config.get('endpoint_revision'):
                    from scripts.run_glt_pt_locality_unit import evaluate_process_validation
                else:
                    from scripts.evaluate_glt_pt_endpoint import evaluate_process_validation
                evaluate_process_validation(base, config, step_number, device, rank, world)
            if benchmark_mode:
                if step_number > start + args.benchmark_warmup:
                    # This is intentionally a CPU-observed diagnostic only;
                    # the primary benchmark uses the complete window boundary
                    # below and performs no per-update CUDA synchronization.
                    benchmark_times.append(time.perf_counter() - step_started)
                    benchmark_graphs.append(float(counts[2].detach().cpu()))
                continue
            if args.diagnostics:
                if (step + 1) in save_steps:
                    states = [None] * world
                    if world > 1:
                        dist.all_gather_object(states, rng_state())
                    else:
                        states[0] = rng_state()
                    if rank == 0:
                        save_checkpoint(output / f'resume_{step + 1:05d}.pt', dict(identity=identity,
                            ordered_keys=ordered_keys, direct_binding=bindings, step=step + 1,
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
                        ordered_keys=ordered_keys, direct_binding=bindings, step=step + 1, next_position=(step + 1) * batch_size,
                        model=base.state_dict(), optimizer=optimizer.state_dict(), rng=states,
                        scheduler=dict(step=step + 1, lr=float(lr))))
                    package = deployment_package(base, step + 1)
                    if config.get('ua2d_pt_revision'):
                        from src.training.ua2d_pt20k import bind_package as bind_ua2d_package
                        package = bind_ua2d_package(package, config)
                    if r10:
                        from src.modules.glt_r10 import bind_package
                        package=bind_package(package,config)
                    if config.get('reference_revision'):
                        from scripts.run_glt_pt_matched_reference_unit import bind_deployment
                        package = bind_deployment(package, config)
                    if config.get('t3_revision'):
                        from scripts.run_fce_tune_t3_unit import bind_deployment
                        package=bind_deployment(package,config)
                    if config.get('t3b_revision'):
                        from scripts.run_fce_tune_t3b_unit import bind_deployment
                        package=bind_deployment(package,config)
                    save_checkpoint(output / f'deploy_{step + 1:05d}.pt', package)
                if world > 1:
                    dist.barrier()
        if benchmark_mode:
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            local_window_seconds = time.perf_counter() - benchmark_window_started
            payload = dict(rank=rank, step_seconds_cpu_observed=benchmark_times,
                           graph_counts=benchmark_graphs,
                           window_seconds=float(local_window_seconds))
            gathered = [None] * world
            if world > 1:
                dist.all_gather_object(gathered, payload)
            else:
                gathered[0] = payload
            if rank == 0:
                slowest = [max(item['step_seconds_cpu_observed'][index] for item in gathered)
                           for index in range(len(benchmark_times))]
                measured_updates = len(slowest)
                window_seconds = float(max(item['window_seconds'] for item in gathered))
                write_json(output / 'benchmark.json', dict(
                    status='PASS', benchmark_steps=int(args.benchmark_steps),
                    benchmark_warmup=int(args.benchmark_warmup),
                    measured_updates=measured_updates, world_size=world,
                    global_batch=int(batch_size),
                    samples=int(measured_updates * batch_size),
                    window_seconds=window_seconds,
                    samples_per_second=float(measured_updates * batch_size / max(window_seconds, 1e-12)),
                    slowest_rank_window_seconds=window_seconds,
                    per_rank_window_seconds={str(item['rank']): float(item['window_seconds'])
                                     for item in gathered},
                    step_timing_scope='cpu_observed_without_per_update_cuda_sync',
                    slowest_rank_step_seconds_cpu_observed=[
                        max(item['step_seconds_cpu_observed'][index] for item in gathered)
                        for index in range(measured_updates)],
                    per_rank_step_timing={str(item['rank']): _time_summary(item['step_seconds_cpu_observed'])
                                     for item in gathered},
                    graph_count_sum=float(sum(max(item['graph_counts'][index] for item in gathered)
                                              for index in range(measured_updates))),
                    records_path=[str(output / f"records_rank{item['rank']}.jsonl") for item in gathered],
                ))
        if rank == 0:
            runtime = json.loads((output / 'runtime.json').read_text(encoding='utf-8'))
            runtime.update(status='PASS', completed_steps=int(stop - start),
                           benchmark_mode=benchmark_mode, finished_at_monotonic=time.perf_counter())
            write_json(output / 'runtime.json', runtime)
            if new_cycle:
                write_consumption(output / 'consumption.json', attempted=stop,
                                  completed=stop, status='COMPLETED')
    finally:
        source.close()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
