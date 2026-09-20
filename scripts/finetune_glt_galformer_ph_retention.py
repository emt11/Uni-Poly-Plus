#!/usr/bin/env python3
"""PH retention downstream runner: F_OFF / F_CONST / F_REAL on one C1 trunk.

All three groups load the same frozen C1 ``deploy_05000.pt`` (verified by
sha256 against the previous cycle's version record), build the identical module
tree and therefore the identical initial tensors; only the PH input differs.
Exactly one of ``--updates``/``--smoke``/``--development`` must be selected:
the first runs a bounded trainability check, the other two are validation-only
over xc/eps/eat folds 0/1 and no outer-test loader exists in this file.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from scripts.create_mips_split_manifests import build_manifest
from scripts.finetune_glt_galformer_ph import (fixed_manifest, fit_select_and_test,
                                               optimizer_for)
from src.dataset.glt_ph_downstream import (GROUPS, PHRetentionDataset, fold_coverage,
                                           key_row_map, load_const_profile,
                                           open_sidecar, retention_collate)
from src.modules.glt_galformer_ph import GLTGalPH
from src.modules.glt_galformer_ph_downstream import TRAINING_ONLY_HEADS
from src.modules.glt_galformer_ph_pretrain import load_galformer_deployment
from src.modules.glt_galformer_ph_retention import GalformerPHDownstream
from src.modules.glt_galph_checkpoint_identity import (DEFAULT_IDENTITY, load_identity,
                                                       verify_checkpoint)
from src.training.glt_dual_runtime import (TASKS, open_source, require_tmux,
                                           save_checkpoint, write_json)
from src.utils import set_global_seed, scale_targets, train_epoch, _cosine_scheduler

GROUPS_TO_MODE = {'F_OFF': 'off', 'F_CONST': 'const', 'F_REAL': 'real'}
DIAGNOSTIC_BATCHES = 8
ALLOWED_TASKS = ('xc', 'eps', 'eat')
ALLOWED_FOLDS = (0, 1)
SMOKE_EPOCH_CAP = 2
DEVELOPMENT_EPOCH_CAP = 30
PROTOCOLS = ('ph_retention_bounded_updates', 'ph_retention_smoke',
             'ph_retention_development')


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class EpochHistory(dict):
    """``validation_summary`` that also keeps every epoch's record.

    ``fit_select_and_test`` updates its summary in place once per epoch, so
    counting the updates is how the caller learns the epochs actually run and
    the per-epoch validation R2 without touching the shared helper.
    """

    def __init__(self):
        super().__init__()
        self.history = []

    def update(self, *args, **kwargs):
        super().update(*args, **kwargs)
        self.history.append(dict(self))


def resolve_protocol(updates, smoke, development):
    """Exactly one of updates/smoke/development must be requested explicitly."""
    if updates is not None and int(updates) <= 0:
        raise ValueError('--updates must be a positive integer')
    chosen = [name for name, active in (('updates', updates is not None),
                                        ('smoke', bool(smoke)),
                                        ('development', bool(development))) if active]
    if len(chosen) != 1:
        raise ValueError('select exactly one of --updates, --smoke and --development')
    return {'updates': PROTOCOLS[0], 'smoke': PROTOCOLS[1],
            'development': PROTOCOLS[2]}[chosen[0]]


def epoch_budget(protocol, configured):
    """Smoke/bounded runs cap at two epochs, development at thirty."""
    cap = DEVELOPMENT_EPOCH_CAP if protocol == PROTOCOLS[2] else SMOKE_EPOCH_CAP
    return min(int(configured), cap)


def runtime_status(units):
    """A unit that ran fewer optimizer steps than requested can never be a PASS."""
    incomplete = sorted(f'{unit["group"]}/{unit["task"]}/fold{unit["fold"]}'
                        for unit in units if unit.get('updates_incomplete'))
    return ('FAIL' if incomplete else 'PASS'), incomplete


def resolve_selection(tasks, folds, *, smoke=False, development=False, updates=0):
    """Only the retention scope (xc/eps/eat, folds 0/1) exists; no outer-test mode."""
    selected_tasks = list(tasks) if tasks else list(ALLOWED_TASKS)
    selected_folds = [int(value) for value in folds] if folds else list(ALLOWED_FOLDS)
    if sorted(set(selected_tasks) - set(TASKS)) or any(v < 0 or v >= 5 for v in selected_folds):
        raise ValueError('task/fold selection is outside TASKS or outer5_inner20')
    if smoke or updates:
        if len(selected_tasks) != 1 or len(selected_folds) != 1:
            raise ValueError('a bounded run requires exactly one --task and one --fold')
    if not set(selected_tasks).issubset(set(ALLOWED_TASKS)):
        raise ValueError('this round only runs xc, eps and eat')
    if any(value not in ALLOWED_FOLDS for value in selected_folds):
        raise ValueError('this round only runs folds 0 and 1')
    return selected_tasks, selected_folds


def bounded_updates(model, loader, optimizer, scheduler, device, updates, summary):
    """Run exactly ``updates`` optimizer steps with the development recipe."""
    criterion = nn.MSELoss()
    model.train()
    seen, losses, gate_grads = 0, [], []
    for batch in loader:
        batch = batch.to(device, non_blocking=True)
        target = batch.y.view(-1, 1)
        optimizer.zero_grad(set_to_none=True)
        output, aux = model(batch)
        loss = criterion(output, target)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError('nonfinite training loss')
        loss.backward()
        gamma_grad = (float(model.gamma.grad.detach().abs().max())
                      if model.gamma.grad is not None else 0.0)
        proj_grad = float(model.ph_proj.weight.grad.detach().norm()) \
            if model.ph_proj.weight.grad is not None else 0.0
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        seen += 1
        losses.append(float(loss.detach()))
        gate_grads.append({'step': seen,
                           'batch_profile_sum': float(batch.ph_profile.sum()),
                           'batch_profile_first_row_sum': float(batch.ph_profile[0].sum()),
                           'gamma_grad_abs_max': gamma_grad,
                           'ph_proj_grad_norm': proj_grad,
                           'fusion_gate_mean': aux.get('gate_mean'),
                           'retention_gate_tanh': model.last_ph_stats.get('gate_tanh'),
                           'residual_relative_norm': model.last_ph_stats.get(
                               'residual_relative_norm'),
                           'ph_valid_fraction': model.last_ph_stats.get('ph_valid_fraction')})
        if seen >= updates:
            break
    summary.update(optimizer_updates=seen, updates_requested=int(updates),
                   training_losses=losses, gate_gradients=gate_grads)
    if seen < int(updates):
        # A loader that ends early must never be reported as a completed check.
        summary['updates_incomplete'] = int(updates) - seen
    return summary


def ph_diagnostics(model, loader, device, const_profile, limit=DIAGNOSTIC_BATCHES):
    """Gate value, residual scale, PH validity and the actual PH input seen.

    The profile fingerprints answer the r5 smoke gate directly: they show whether
    this arm fed the encoder the sample's own profile or one fixed profile, and
    how far the frozen encoder's *summary* moves between the two inputs.
    """
    model.eval()
    gates, ratios, valids, norms, retention_gates = [], [], [], [], []
    profile_norms, profile_spreads, summary_deltas, const_deltas = [], [], [], []
    with torch.inference_mode():
        for index, batch in enumerate(loader):
            if index >= limit:
                break
            batch = batch.to(device, non_blocking=True)
            _, aux = model(batch)
            stats = model.last_ph_stats
            gates.append(float(aux.get('gate_mean', float('nan'))))
            ratios.append(float(stats.get('residual_relative_norm', 0.0)))
            valids.append(float(stats.get('ph_valid_fraction', 0.0)))
            norms.append(float(stats.get('residual_norm', 0.0)))
            retention_gates.append(float(stats.get('gate_tanh', 0.0)))
            profiles = batch.ph_profile.float()
            profile_norms.append(float(profiles.norm(dim=(1, 2)).mean()))
            profile_spreads.append(float((profiles.max(0).values
                                          - profiles.min(0).values).max()))
            const = const_profile.to(device=profiles.device, dtype=torch.float32)
            const = const.unsqueeze(0).expand_as(profiles).contiguous()
            const_deltas.append(float((profiles - const).abs().max()))
            mask = batch.ph_mask.bool()
            own = model.encoder.ph_encoder.summarize(
                model.encoder.ph_encoder(profiles, mask)).float()
            fixed = model.encoder.ph_encoder.summarize(
                model.encoder.ph_encoder(const, mask)).float()
            summary_deltas.append(float((own - fixed).abs().max()))
    mean = lambda values: (float(np.mean(values)) if values else None)
    return {'gate_mean': mean(gates), 'retention_gate_tanh': mean(retention_gates),
            'residual_relative_norm': mean(ratios), 'residual_norm': mean(norms),
            'ph_valid_fraction': mean(valids), 'batches': len(gates),
            'profile_norm_mean': mean(profile_norms),
            'profile_cross_sample_spread': mean(profile_spreads),
            'profile_vs_const_max_abs': mean(const_deltas),
            'encoder_summary_own_vs_const_max_abs': mean(summary_deltas),
            'profile_source': ('zero_placeholder' if model.ph_residual == 'off' else
                               'p_train_mean_fixed' if model.ph_residual == 'const' else
                               'sample_own_frozen')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'checkpoint', 'ph-sidecar', 'const-profile', 'raw-root',
                 'cohort-root', 'cache-root', 'output'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--group', choices=GROUPS, required=True)
    parser.add_argument('--dual-static-root',
                        help='downstream training-ready dual_static_v1 artifact')
    parser.add_argument('--split-root', default='data/splits/mips_outer5_inner20')
    parser.add_argument('--readout', choices=('DUAL', '2D_ONLY'), default='DUAL')
    parser.add_argument('--task', action='append', dest='tasks')
    parser.add_argument('--fold', action='append', type=int, dest='folds')
    parser.add_argument('--smoke', action='store_true',
                        help='single task/fold, at most two epochs, validation-only')
    parser.add_argument('--development', action='store_true',
                        help='xc/eps/eat folds 0/1, validation-only')
    parser.add_argument('--updates', type=int, default=None,
                        help='bounded trainability check: exactly this many optimizer steps')
    parser.add_argument('--checkpoint-identity', default=DEFAULT_IDENTITY,
                        help='the committed identity record this round runs against')
    parser.add_argument('--clean-cache-gib', type=float, default=0.0)
    args = parser.parse_args()
    process_started = time.perf_counter()
    require_tmux()
    config = json.loads(Path(args.config).read_text(encoding='utf-8'))
    protocol = resolve_protocol(args.updates, args.smoke, args.development)
    requested_updates = int(args.updates) if args.updates is not None else 0
    selected_tasks, selected_folds = resolve_selection(
        args.tasks, args.folds, smoke=args.smoke, development=args.development,
        updates=requested_updates)
    epoch_cap = (DEVELOPMENT_EPOCH_CAP if protocol == PROTOCOLS[2] else SMOKE_EPOCH_CAP)
    run_config = dict(config)
    run_config['epochs'] = epoch_budget(protocol, config.get('epochs', DEVELOPMENT_EPOCH_CAP))
    run_config['readout'] = args.readout
    run_config['adaptation'] = 'full'
    if args.development and args.readout != 'DUAL':
        raise ValueError('development runs the DUAL readout')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    checkpoint_sha = sha256_file(args.checkpoint)
    identity = load_identity(args.checkpoint_identity)
    package = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    verified = verify_checkpoint(identity, args.checkpoint, package, sha256=checkpoint_sha)
    if int(package.get('step', -1)) != int(run_config['downstream_step']):
        raise ValueError('deployment step does not match downstream_step')
    if package.get('ph_mode') != 'global' or package.get('summary_mode') != 'cls':
        raise ValueError('retention requires the C1 deployment')

    reader = open_sidecar(args.ph_sidecar)
    key_rows = key_row_map(reader)
    const_profile = load_const_profile(args.const_profile)
    sidecar_meta = json.loads((Path(args.ph_sidecar) / 'metadata.json')
                              .read_text(encoding='utf-8'))
    write_json(output / 'run.json', dict(
        config=run_config, command=sys.argv, protocol=protocol, group=args.group,
        ph_residual=GROUPS_TO_MODE[args.group], readout=args.readout,
        adaptation='full', selected_tasks=selected_tasks,
        selected_folds=selected_folds, updates=requested_updates,
        epochs_cap=int(epoch_cap),
        checkpoint=str(args.checkpoint), checkpoint_sha256=checkpoint_sha,
        checkpoint_identity=verified,
        ph_sidecar=str(args.ph_sidecar), ph_sidecar_records=len(reader),
        const_profile=str(args.const_profile),
        const_profile_source=sidecar_meta.get('const_profile_source'),
        ph_encoder_version=package.get('ph_encoder_version'),
        ph_downstream='retention_residual_into_r3', outer_test='NOT_RUN'))
    all_folds = []
    for task in selected_tasks:
        csv_path = Path(args.raw_root) / f'smi_{task}.csv'
        manifest_path = Path(args.split_root) / f'{task}.json'
        if not manifest_path.is_file():
            raise FileNotFoundError(f'development requires an existing manifest: {manifest_path}')
        manifest = fixed_manifest(task, csv_path, manifest_path)
        source, frame = open_source(args.cohort_root, args.cache_root, task=task,
                                    dual_static_root=args.dual_static_root)
        dataset = None
        try:
            if len(frame) != int(manifest['sample_count']):
                raise ValueError('downstream cohort task row count differs from fixed split')
            if frame['original_row'].astype(int).tolist() != list(range(len(frame))):
                raise ValueError('downstream cohort task row order differs from property CSV')
            targets = frame['label'].to_numpy(dtype=np.float64)
            if not np.isfinite(targets).all():
                raise ValueError('nonfinite labels')
            dataset = PHRetentionDataset(
                source, targets, group=args.group, reader=reader,
                const_profile=(const_profile if args.group == 'F_CONST' else None),
                key_rows=key_rows,
                cache_capacity_bytes=int(args.clean_cache_gib * (1024 ** 3)))
            for fold in [f for f in manifest['folds'] if f['fold'] in selected_folds]:
                fold_id = fold['fold']
                fold_started = time.perf_counter()
                folder = output / task / f'fold{fold_id}'
                folder.mkdir(parents=True, exist_ok=False)
                set_global_seed(int(run_config['seed']) + fold_id)
                train, validation = (fold['train_indices'], fold['validation_indices'])

                def loader(indices, training=False):
                    return DataLoader(
                        Subset(dataset, indices),
                        batch_size=run_config['finetune_batch'] if training else run_config['eval_batch'],
                        shuffle=training, num_workers=0, collate_fn=retention_collate,
                        generator=torch.Generator().manual_seed(int(run_config['seed']) + fold_id))

                train_loader, val_loader = loader(train, True), loader(validation)
                scaler = scale_targets(dataset, task, train_indices=train,
                                       transform_mode='standard')
                encoder = GLTGalPH(str(package['summary_mode']), package.get('ph_mode'),
                                   dropout=float(run_config.get('dropout', 0.1))).to(device)
                load_galformer_deployment(encoder, package, int(run_config['downstream_step']))
                model = GalformerPHDownstream(
                    encoder, readout=args.readout, ph_residual=GROUPS_TO_MODE[args.group],
                    dropout=float(run_config.get('dropout', 0.1))).to(device)
                frozen_heads = model.freeze_training_only_heads()
                optimizer = optimizer_for(model, run_config)
                scheduler = _cosine_scheduler(
                    optimizer, run_config['epochs'] * len(train_loader),
                    run_config['finetune_warmup'] * len(train_loader))
                unit = dict(
                    protocol=protocol, group=args.group,
                    ph_residual=GROUPS_TO_MODE[args.group], task=task, fold=fold_id,
                    readout=args.readout, adaptation='full',
                    pretrain_summary_mode=str(package['summary_mode']),
                    pretrain_ph_mode=str(package.get('ph_mode')),
                    ph_encoder_version=package.get('ph_encoder_version'),
                    checkpoint_sha256=checkpoint_sha,
                    checkpoint_identity=verified,
                    frozen_training_only_heads=frozen_heads,
                    trainable_parameter_count=int(sum(p.numel() for p in model.parameters()
                                                      if p.requires_grad)),
                    total_parameter_count=int(sum(p.numel() for p in model.parameters())),
                    epochs_configured=int(run_config['epochs']),
                    epochs_cap=int(epoch_cap),
                    train_sample_count=int(len(train)),
                    validation_sample_count=int(len(validation)),
                    outer_test='NOT_RUN', validation_only=True,
                    scaler_fit_split='train',
                    coverage=fold_coverage(reader, [dataset.source.samples[int(i)][0].hex()
                                                    for i in list(train) + list(validation)],
                                           key_rows))
                if protocol == PROTOCOLS[0]:
                    unit = bounded_updates(model, train_loader, optimizer, scheduler,
                                           device, requested_updates, unit)
                    unit['best_validation_r2'] = None
                    unit['best_epoch'] = None
                else:
                    validation_summary = EpochHistory()
                    best, best_r2, best_epoch, _ = fit_select_and_test(
                        model, scaler, train_loader, val_loader, None, device, optimizer,
                        scheduler, run_config, task=task, fold_id=fold_id,
                        validation_only=True, validation_summary=validation_summary)
                    unit.update(validation_summary)
                    unit['epochs_run'] = len(validation_summary.history)
                    unit['validation_r2_history'] = [
                        float(record['validation_r2']) for record in validation_summary.history]
                    unit['best_validation_r2'] = float(best_r2)
                    unit['best_epoch'] = int(best_epoch)
                    unit['diagnostics'] = ph_diagnostics(model, val_loader, device,
                                                         const_profile)
                    save_checkpoint(folder / 'best.pt', dict(
                        state_dict=best, architecture=model.architecture_name,
                        group=args.group, task=task, fold=fold_id, protocol=protocol,
                        split=fold, config=run_config, best_validation_r2=best_r2,
                        best_epoch=best_epoch,
                        epochs_run=int(unit.get('epochs_run', -1)),
                        checkpoint_sha256=checkpoint_sha,
                        checkpoint_identity=verified,
                        scaler_mean=scaler.scaler.mean_.tolist(),
                        scaler_scale=scaler.scaler.scale_.tolist()))
                unit['wall_seconds'] = float(time.perf_counter() - fold_started)
                write_json(folder / 'metrics.json', unit)
                all_folds.append(unit)
                del model, encoder, optimizer, scheduler
        finally:
            if dataset is not None:
                dataset.cache_stats()
            source.close()
    pd.DataFrame(all_folds).to_csv(output / 'all_fold_metrics.csv', index=False)
    status, incomplete = runtime_status(all_folds)
    write_json(output / 'summary.json', dict(
        protocol=protocol, group=args.group, ph_residual=GROUPS_TO_MODE[args.group],
        units=all_folds, outer_test='NOT_RUN', validation_only=True,
        updates_incomplete=incomplete,
        interpretation=('bounded trainability check' if protocol == PROTOCOLS[0] else
                        'development-fold validation-only evaluation; not OOF and not '
                        'independent blind evidence')))
    write_json(output / 'runtime.json', dict(
        status=status, pid=os.getpid(), command=sys.argv,
        protocol=protocol, group=args.group, selected_tasks=selected_tasks,
        selected_folds=selected_folds, updates=requested_updates,
        updates_incomplete=incomplete, outer_test='NOT_RUN',
        process_wall_seconds=float(time.perf_counter() - process_started)))
    if incomplete:
        raise RuntimeError('fewer optimizer updates ran than requested: '
                           + ', '.join(incomplete))


if __name__ == '__main__':
    main()
