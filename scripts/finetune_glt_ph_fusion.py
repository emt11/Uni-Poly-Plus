#!/usr/bin/env python3
"""GLT-PH end-to-end downstream runner (plan section 7.1).

One arm per invocation: R0 / R2D / CURRENT / CONST / STAT / PH.  The exploration
protocol is fixed: ``outer5_inner20`` folds read from the real split manifests,
train-only target scaler, Huber(delta 0.5) on the standardized labels, AdamW with
the trunk at 1e-5 and the new readout/fusion/head at 1e-4, best-by-validation-R2
selection, validation only -- there is no outer-test loader in this file.

Failure discipline follows r6: ``best.pt`` and the core ``metrics.json`` are
written before any optional diagnostic, a failing diagnostic keeps both and
records ``complete=false`` with its stage, and the process exits non-zero after
writing a FAIL ``runtime.json``.
"""
import argparse
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
from scripts.finetune_glt_galformer_ph import fixed_manifest
from src.dataset.glt_dual import dual_glt_collate
from src.dataset.glt_ph_downstream import (fold_coverage, key_row_map, load_const_profile,
                                           open_sidecar)
from src.dataset.glt_ph_fusion_inputs import (ARMS, FusionDataset, fusion_downstream_collate,
                                              load_ph_stats)
from src.modules.glt_galformer_ph import GLTGalPH
from src.modules.glt_galformer_ph_downstream import TRAINING_ONLY_HEADS
from src.modules.glt_galph_checkpoint_identity import (DEFAULT_IDENTITY, load_identity,
                                                       verify_checkpoint)
from src.modules.glt_ph_fusion_candidates import (GLTFusionB, R2DModel, FusionDownstream,
                                                  load_fusion_deployment)
from src.training.glt_dual_runtime import (CleanLabeledDataset, open_source, require_tmux,
                                           save_checkpoint, write_json)
from src.utils import _cosine_scheduler, evaluate, scale_targets, set_global_seed, train_epoch

PREFIX_ARMS = ('CONST', 'STAT', 'PH')
ALL_ARMS = ('R0', 'R2D', 'CURRENT') + PREFIX_ARMS
REFERENCE_FROZEN = ('ph_encoder', 'ph_to_summary', 'alpha_ph')
ENCODER_PREFIXES = ('o8.', 'glt.', 'conditional.', 'spatial.')
HUBER_DELTA = 0.5
EPOCH_CAPS = {'smoke': 1, 'development': 30, 'formal': 100}
RUN_CONTEXT = {}


def arm_family(arm):
    return 'reference' if arm == 'R0' else ('R2D' if arm == 'R2D' else
                                            ('CURRENT' if arm == 'CURRENT' else 'B'))


def downstream_optimizer(model, config):
    """Trunk at encoder_lr, new readout/fusion/head at fusion_head_lr (section 7.1)."""
    groups = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        trunk = name.startswith(ENCODER_PREFIXES)
        decay = parameter.dim() >= 2 and not name.startswith(('conditional.', 'spatial.'))
        key = ('encoder' if trunk else 'head', 'decay' if decay else 'no_decay')
        groups.setdefault(key, []).append((name, parameter))
    built = []
    for (role, decay_kind), items in sorted(groups.items()):
        built.append(dict(params=[item[1] for item in items],
                          lr=float(config['encoder_lr'] if role == 'encoder'
                                   else config['fusion_head_lr']),
                          weight_decay=float(config['finetune_weight_decay']
                                             if decay_kind == 'decay' else 0.0),
                          name=f'{role}_{decay_kind}'))
    if not built:
        raise RuntimeError('full adaptation found no trainable parameter')
    return torch.optim.AdamW(built), {f'{role}_{decay}': sorted(name for name, _ in items)
                                      for (role, decay), items in groups.items()}


def build_encoder(arm, args, config, device):
    """Instantiate one arm's encoder and load its deployment strictly."""
    record = {'arm': arm, 'family': arm_family(arm)}
    if arm == 'R2D':
        encoder = R2DModel(dropout=float(config.get('dropout', 0.1)))
        package = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        load_fusion_deployment(encoder, package, expected_step=int(config['downstream_step']))
        record['checkpoint'] = str(Path(args.checkpoint).resolve())
        record['checkpoint_step'] = int(package['step'])
        record['checkpoint_architecture'] = package['architecture']
        return encoder.to(device), record
    if arm in PREFIX_ARMS:
        stats = load_ph_stats(args.profile_stats)
        if stats['scope'] and stats['scope'].startswith('SMOKE_ONLY') and not args.allow_smoke_stats:
            raise ValueError('SMOKE_ONLY statistics require --allow-smoke-stats')
        encoder = GLTFusionB(arm=arm, profile_stats=stats,
                             dropout=float(config.get('dropout', 0.1)))
        package = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        load_fusion_deployment(encoder, package, expected_step=int(config['downstream_step']))
        record.update(checkpoint=str(Path(args.checkpoint).resolve()),
                      checkpoint_step=int(package['step']),
                      checkpoint_architecture=package['architecture'],
                      profile_stats={'path': str(Path(args.profile_stats).resolve()),
                                     'scope': stats['scope'], 'source': stats['source']})
        return encoder.to(device), record
    if arm == 'CURRENT':
        identity = load_identity(args.identity)
        package = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        verify_checkpoint(identity, args.checkpoint, package)
        encoder = GLTGalPH('cls', 'global', dropout=float(config.get('dropout', 0.1)))
        from src.modules.glt_galformer_ph_pretrain import load_galformer_deployment
        load_galformer_deployment(encoder, package, int(config['downstream_step']))
        record.update(checkpoint=str(Path(args.checkpoint).resolve()),
                      checkpoint_step=int(package['step']),
                      checkpoint_architecture=package['architecture'],
                      checkpoint_sha256=identity['sha256'],
                      identity_record=str(Path(args.identity).resolve()))
        return encoder.to(device), record
    encoder = GLTGalPH('cls', None, dropout=float(config.get('dropout', 0.1)))
    package = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    load_fusion_deployment(encoder, package, expected_step=int(config['downstream_step']))
    record.update(checkpoint=str(Path(args.checkpoint).resolve()),
                  checkpoint_step=int(package['step']),
                  checkpoint_architecture=package['architecture'])
    return encoder.to(device), record


def build_dataset(arm, args, source, targets):
    if arm in PREFIX_ARMS:
        reader = (open_sidecar(args.ph_sidecar_root) if arm == 'PH' else None)
        key_rows = key_row_map(reader) if reader is not None else {}
        dataset = FusionDataset(source, targets, arm=arm, ph_reader=reader,
                                const_profile=np.load(args.const_profile),
                                key_rows=key_rows)
        return dataset, (fusion_downstream_collate, reader, key_rows)
    return CleanLabeledDataset(source, targets), (dual_glt_collate, None, None)


def select_and_select_best(model, scaler, loaders, device, optimizer, scheduler, config,
                           *, task, fold_id):
    criterion = nn.HuberLoss(delta=HUBER_DELTA)
    train_loader, val_loader = loaders
    best, best_r2, best_epoch, stalled, history = None, -float('inf'), -1, 0, []
    for epoch in range(int(config['epochs'])):
        train_epoch(model, train_loader, criterion, optimizer, scheduler, device,
                    epoch=epoch + 1, amp_dtype='fp32', max_grad_norm=1.0,
                    fail_nonfinite=True)
        loss, r2, targets, predictions = evaluate(model, val_loader, criterion, device,
                                                  scaler=scaler, amp_dtype='fp32')
        if not np.isfinite(r2):
            raise FloatingPointError('nonfinite validation R2')
        history.append({'epoch': epoch + 1, 'validation_loss': float(loss),
                        'validation_r2': float(r2)})
        print(json.dumps({'task': task, 'fold': fold_id, 'epoch': epoch + 1,
                          'validation_r2': float(r2)}), flush=True)
        if r2 > best_r2:
            best = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            best_r2, best_epoch, stalled = float(r2), epoch + 1, 0
        else:
            stalled += 1
        if stalled >= int(config['patience']):
            break
    if best is None:
        raise RuntimeError('no finite validation-selected checkpoint')
    model.load_state_dict(best, strict=True)
    return best, best_r2, best_epoch, history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--arm', required=True, choices=ALL_ARMS)
    parser.add_argument('--mode', default='smoke', choices=tuple(EPOCH_CAPS))
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--identity', default=DEFAULT_IDENTITY)
    parser.add_argument('--profile-stats')
    parser.add_argument('--allow-smoke-stats', action='store_true')
    parser.add_argument('--ph-sidecar-root',
                        default='results/glt_galph_ph_retention_20260920/p0/ph_sidecar_downstream')
    parser.add_argument('--const-profile',
                        default=('results/glt_galph_ph_retention_20260920/p0/'
                                 'ph_sidecar_downstream/p_train_mean_profile.npy'))
    parser.add_argument('--raw-root', default='data/raw')
    parser.add_argument('--split-root', default='data/splits/mips_outer5_inner20')
    parser.add_argument('--cohort-root', required=True)
    parser.add_argument('--cache-root', required=True)
    parser.add_argument('--dual-static-root', required=True)
    parser.add_argument('--tasks', nargs='*', default=['xc'])
    parser.add_argument('--folds', type=int, nargs='*', default=[0])
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    require_tmux()
    config = json.loads(Path(args.config).read_text(encoding='utf-8'))
    run_config = dict(config)
    run_config['epochs'] = int(args.epochs or EPOCH_CAPS[args.mode])
    if run_config['epochs'] > EPOCH_CAPS[args.mode]:
        raise ValueError(f'{args.mode} allows at most {EPOCH_CAPS[args.mode]} epochs')
    readout = '2D_ONLY' if args.arm == 'R2D' else 'DUAL'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    RUN_CONTEXT.update(output=output, arm=args.arm, stage='setup')
    encoder, checkpoint_record = build_encoder(args.arm, args, run_config, device)
    model = FusionDownstream(encoder, readout=readout,
                             dropout=float(run_config.get('dropout', 0.1))).to(device)
    frozen = model.freeze_training_only_heads()
    if args.arm == 'CURRENT':
        for name in REFERENCE_FROZEN:
            module = getattr(model.encoder, name, None)
            if module is not None:
                module.requires_grad_(False)
                frozen.append(name)
    trainable = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    total = int(sum(p.numel() for p in model.parameters()))
    optimizer, group_record = downstream_optimizer(model, run_config)
    if os.environ.get('RANK', '0') == '0':
        write_json(output / 'run.json', dict(
            plan='GLT-PH-END2END-20260920-01', arm=args.arm, mode=args.mode,
            readout=readout, config=run_config, command=sys.argv,
            checkpoint=checkpoint_record, frozen_modules=frozen,
            parameter_counts={'total': total, 'trainable': trainable},
            optimizer_groups=group_record, outer_test='NOT_RUN', validation_only=True))
    all_folds = []
    source = None
    RUN_CONTEXT['stage'] = 'data'
    for task in args.tasks:
        manifest = fixed_manifest(task, Path(args.raw_root) / f'smi_{task}.csv',
                                  Path(args.split_root) / f'{task}.json')
        # One task-scoped view: manifest indices are task-local, exactly as the
        # retained zero-update check and the r6 units read them.
        source, frame = open_source(args.cohort_root, args.cache_root, task=task,
                                    dual_static_root=args.dual_static_root)
        try:
            if len(frame) != int(manifest['sample_count']):
                raise ValueError('downstream cohort task rows differ from the split')
            targets = frame['label'].to_numpy(dtype=np.float64)
            for fold in [item for item in manifest['folds'] if int(item['fold']) in args.folds]:
                fold_id = int(fold['fold'])
                fold_started = time.perf_counter()
                folder = output / task / f'fold{fold_id}'
                folder.mkdir(parents=True, exist_ok=False)
                set_global_seed(int(run_config['seed']) + fold_id)
                train = list(fold['train_indices'])
                validation = list(fold['validation_indices'])
                dataset, (collate, reader, key_rows) = build_dataset(
                    args.arm, args, source, targets)
                scaler = scale_targets(dataset, task, train_indices=train,
                                       transform_mode='standard')

                def loader(indices, training=False):
                    return DataLoader(Subset(dataset, indices),
                                      batch_size=int(run_config['finetune_batch'] if training
                                                     else run_config['eval_batch']),
                                      shuffle=training, num_workers=0, collate_fn=collate,
                                      generator=torch.Generator().manual_seed(
                                          int(run_config['seed']) + fold_id))

                loaders = (loader(train, True), loader(validation))
                scheduler = _cosine_scheduler(
                    optimizer, run_config['epochs'] * len(loaders[0]),
                    int(run_config['warmup']) * len(loaders[0]))
                unit = dict(
                    plan='GLT-PH-END2END-20260920-01', arm=args.arm, mode=args.mode,
                    family=arm_family(args.arm), task=task, fold=fold_id, readout=readout,
                    adaptation='full', outer_test='NOT_RUN', validation_only=True,
                    scaler_fit_split='train', train_sample_count=len(train),
                    validation_sample_count=len(validation),
                    checkpoint=checkpoint_record, config=run_config,
                    epochs_configured=int(run_config['epochs']),
                    frozen_modules=frozen,
                    parameter_counts={'total': total, 'trainable': trainable},
                    optimizer_groups=group_record,
                    coverage=(fold_coverage(reader, [source.samples[int(i)][0].hex()
                                                     for i in list(train) + list(validation)],
                                            key_rows) if reader is not None else None))
                RUN_CONTEXT['stage'] = f'train:{task}:fold{fold_id}'
                best, best_r2, best_epoch, history = select_and_select_best(
                    model, scaler, loaders, device, optimizer, scheduler, run_config,
                    task=task, fold_id=fold_id)
                unit['validation_r2_history'] = [row['validation_r2'] for row in history]
                unit['epochs_run'] = len(history)
                unit['best_validation_r2'] = float(best_r2)
                unit['best_epoch'] = int(best_epoch)
                unit['stage_completed'] = 'selection'
                unit['complete'] = False
                save_checkpoint(folder / 'best.pt', dict(
                    state_dict=best, architecture=model.architecture_name, arm=args.arm,
                    readout=readout, task=task, fold=fold_id, best_validation_r2=best_r2,
                    best_epoch=best_epoch, epochs_run=len(history),
                    scaler_mean=scaler.scaler.mean_.tolist(),
                    scaler_scale=scaler.scaler.scale_.tolist(),
                    checkpoint=checkpoint_record, split=fold))
                write_json(folder / 'metrics.json', unit)
                RUN_CONTEXT['stage'] = f'diagnostics:{task}:fold{fold_id}'
                try:
                    unit['diagnostics'] = {
                        'validation_r2_final_epoch': float(history[-1]['validation_r2']),
                        'epochs_run': len(history)}
                    unit['wall_seconds'] = float(time.perf_counter() - fold_started)
                except Exception as error:
                    unit['failure'] = dict(stage='diagnostics', checkpoint_kept=True,
                                           error=f'{type(error).__name__}: {error}')
                    write_json(folder / 'metrics.json', unit)
                    raise
                unit['stage_completed'] = 'diagnostics'
                unit['complete'] = True
                write_json(folder / 'metrics.json', unit)
                all_folds.append(unit)
                del best
        finally:
            source.close()
    RUN_CONTEXT['stage'] = 'finalise'
    pd.DataFrame(all_folds).to_csv(output / 'all_fold_metrics.csv', index=False)
    write_json(output / 'summary.json', dict(
        plan='GLT-PH-END2END-20260920-01', arm=args.arm, mode=args.mode,
        readout=readout, units=all_folds, outer_test='NOT_RUN', validation_only=True,
        interpretation=('implementation/interfaces smoke over development folds; '
                        'not OOF, not an independent blind test, and not a '
                        'performance claim')))
    write_json(output / 'runtime.json', dict(
        status='PASS', stage='complete', command=sys.argv, arm=args.arm,
        mode=args.mode, tasks=args.tasks, folds=args.folds, outer_test='NOT_RUN',
        process_wall_seconds=float(time.perf_counter()
                                   - RUN_CONTEXT.get('process_started', time.perf_counter()))))


def write_failure(context, error):
    output = context.get('output')
    if output is None:
        return
    try:
        write_json(Path(output) / 'runtime.json', dict(
            status='FAIL', stage=context.get('stage', 'unknown'), arm=context.get('arm'),
            error=f'{type(error).__name__}: {error}', command=sys.argv,
            checkpoints_written=sorted(path.name for path in Path(output).glob('**/*.pt')),
            metrics_written=sorted(path.name for path in Path(output).glob('**/metrics.json')),
            checkpoint_kept=bool(list(Path(output).glob('**/*.pt')))))
    except Exception as nested:
        print(f'PH_FUSION_WRITE_FAILURE_FAILED {nested}', flush=True)


if __name__ == '__main__':
    RUN_CONTEXT['process_started'] = time.perf_counter()
    try:
        main()
    except Exception as error:
        write_failure(RUN_CONTEXT, error)
        print(f'PH_FUSION_FINETUNE_FAILED arm={RUN_CONTEXT.get("arm")} '
              f'stage={RUN_CONTEXT.get("stage")} {type(error).__name__}: {error}', flush=True)
        raise
