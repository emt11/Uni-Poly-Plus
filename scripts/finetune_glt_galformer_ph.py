#!/usr/bin/env python3
"""GALPH downstream fine-tuning on the fixed outer5_inner20 manifests.

Round 1 runs the DUAL readout only (``--development``: xc/eps/eat, folds 0/1,
validation-only, never reads outer-test).  The protocol mirrors
``finetune_glt_dual.py``: FULL adaptation, train-only target scaler, 30 epochs
with warmup 5 and patience 10, best-by-validation-R2 checkpoint selection.
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
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from scripts.create_mips_split_manifests import build_manifest
from src.dataset.glt_dual import dual_glt_collate
from src.modules.glt_galformer_ph import GLTGalPH
from src.modules.glt_galformer_ph_downstream import GalformerDownstream
from src.modules.glt_galformer_ph_pretrain import load_galformer_deployment
from src.training.glt_dual_runtime import (TASKS, CleanLabeledDataset, open_source,
                                           require_tmux, save_checkpoint, write_json)
from src.utils import (set_global_seed, scale_targets, train_epoch, evaluate,
                       test_model, _cosine_scheduler)


def fixed_manifest(task, csv_path, path):
    expected = build_manifest(task, csv_path, 'outer5_inner20')
    if path.exists():
        actual = json.loads(path.read_text(encoding='utf-8'))
        for field in ('protocol', 'sample_count', 'sample_order_hash'):
            if actual.get(field) != expected[field]:
                raise ValueError(f'fixed manifest mismatch: {field}')
        if actual.get('validation_is_test') is not False or len(actual['folds']) != 5:
            raise ValueError('requires five separated folds')
        for old, new in zip(actual['folds'], expected['folds']):
            for field in ('fold', 'train_indices', 'validation_indices', 'test_indices'):
                if old[field] != new[field]:
                    raise ValueError(f'existing fixed split differs: {field}; not overwritten')
        return actual
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, expected)
    return expected


def optimizer_for(model, config):
    """Encoder trunk at encoder_lr, readout/gate/fusion/property head at head lr."""
    groups, seen = [], set()
    for name, module in (('o8', model.encoder.o8), ('glt', model.encoder.glt)):
        params = [p for p in module.parameters() if p.requires_grad]
        seen.update(id(p) for p in params)
        if params:
            groups.append(dict(params=params, lr=config['encoder_lr'], name=name))
    params = [p for p in model.parameters() if p.requires_grad and id(p) not in seen]
    if params:
        groups.append(dict(params=params, lr=config['fusion_head_lr'], name='fusion_head'))
    if not groups:
        raise RuntimeError('full adaptation found no trainable parameters')
    return torch.optim.AdamW(groups, weight_decay=config['finetune_weight_decay'])


def fit_select_and_test(model, scaler, train_loader, val_loader, test_loader, device,
                        optimizer, scheduler, config, *, task, fold_id,
                        validation_only=False, validation_summary=None):
    criterion = nn.MSELoss()
    best, best_r2, best_epoch, stalled = None, -float('inf'), -1, 0
    for epoch in range(config['epochs']):
        train_epoch(model, train_loader, criterion, optimizer, scheduler, device,
                    epoch=epoch + 1, amp_dtype='fp32', max_grad_norm=1.,
                    fail_nonfinite=True)
        val_loss, val_r2, val_targets, val_predictions = evaluate(
            model, val_loader, criterion, device, scaler=scaler, amp_dtype='fp32')
        if validation_summary is not None:
            validation_summary.update(
                validation_loss=float(val_loss), validation_r2=float(val_r2),
                validation_mae=float(mean_absolute_error(val_targets, val_predictions)),
                validation_rmse=float(np.sqrt(mean_squared_error(val_targets, val_predictions))))
        print(json.dumps(dict(task=task, fold=fold_id, epoch=epoch + 1,
                              validation_r2=val_r2)), flush=True)
        if not np.isfinite(val_r2):
            raise FloatingPointError('nonfinite validation R2')
        if val_r2 > best_r2:
            best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_r2, best_epoch, stalled = val_r2, epoch + 1, 0
        else:
            stalled += 1
        if stalled >= config['patience']:
            break
    if best is None:
        raise RuntimeError('no finite validation-selected checkpoint')
    model.load_state_dict(best, strict=True)
    if validation_only:
        return best, best_r2, best_epoch, None
    result = test_model(model, test_loader, scaler, device, return_predictions=True)
    return best, best_r2, best_epoch, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'checkpoint', 'raw-root', 'cohort-root', 'cache-root', 'output'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--dual-static-root',
                        help='downstream training-ready dual_static_v1 artifact')
    parser.add_argument('--split-root', default='data/splits/mips_outer5_inner20')
    parser.add_argument('--task', action='append', dest='tasks')
    parser.add_argument('--fold', action='append', type=int, dest='folds')
    parser.add_argument('--readout', choices=('DUAL', '2D_ONLY'),
                        help='downstream fusion readout (round 1 runs DUAL only)')
    parser.add_argument('--smoke', action='store_true',
                        help='bounded train/validation-only task/fold smoke; never reads outer-test')
    parser.add_argument('--development', action='store_true',
                        help='validation-only development mode; never reads outer-test')
    parser.add_argument('--formal-shard', action='store_true',
                        help='run exactly one task/fold including outer-test for a grid launcher')
    parser.add_argument('--clean-cache-gib', type=float, default=0.0)
    args = parser.parse_args()
    process_started = time.perf_counter()
    require_tmux()
    config = json.loads(Path(args.config).read_text(encoding='utf-8'))
    if not np.isfinite(args.clean_cache_gib) or args.clean_cache_gib < 0:
        raise ValueError('--clean-cache-gib must be finite and non-negative')
    if args.smoke and args.development:
        raise ValueError('--smoke and --development are mutually exclusive')
    if args.formal_shard:
        raise ValueError('round 1 is development-only: this runner has no outer-test path')
    selected_tasks = list(args.tasks) if args.tasks else list(TASKS)
    selected_folds = [int(value) for value in args.folds] if args.folds else list(range(5))
    if args.development and not args.tasks:
        selected_tasks = ['xc', 'eps', 'eat']
    if args.development and not args.folds:
        selected_folds = [0, 1]
    if sorted(set(selected_tasks) - set(TASKS)) or any(v < 0 or v >= 5 for v in selected_folds):
        raise ValueError('task/fold selection is outside TASKS or outer5_inner20')
    if args.smoke:
        if len(selected_tasks) != 1 or len(selected_folds) != 1:
            raise ValueError('--smoke requires exactly one --task and one --fold')
    elif args.development:
        if not set(selected_tasks).issubset({'xc', 'eps', 'eat'}):
            raise ValueError('--development only permits xc, eps and eat')
        if any(value not in (0, 1) for value in selected_folds):
            raise ValueError('--development only permits folds 0 and 1')
    elif args.formal_shard:
        if len(selected_tasks) != 1 or len(selected_folds) != 1:
            raise ValueError('--formal-shard requires exactly one --task and --fold')
    elif selected_tasks != list(TASKS) or selected_folds != list(range(5)):
        raise ValueError('partial task/fold selection requires --smoke, --development or --formal-shard')
    run_config = dict(config)
    readout = str(args.readout or run_config.get('readout', 'DUAL')).upper()
    if args.development and readout != 'DUAL':
        raise ValueError('round 1 development runs the DUAL readout only')
    run_config['readout'] = readout
    run_config['adaptation'] = 'full'
    if args.smoke:
        run_config['epochs'] = min(int(run_config.get('epochs', 2)), 2)
    elif args.development:
        run_config['epochs'] = min(int(run_config.get('epochs', 30)), 30)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    package = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    validation_only = bool(args.smoke or args.development)
    protocol = ('outer5_inner20_smoke' if args.smoke else
                'outer5_inner20_development' if args.development else
                'outer5_inner20_formal_shard' if args.formal_shard else
                'outer5_inner20')
    write_json(output / 'run.json', dict(
        config=run_config, command=sys.argv, protocol=protocol,
        smoke=bool(args.smoke), development=bool(args.development),
        formal_shard=bool(args.formal_shard), readout=readout,
        adaptation='full', selected_tasks=selected_tasks,
        selected_folds=selected_folds,
        deployment_step=int(run_config['downstream_step']),
        pretrain_arm=package.get('summary_mode', '?') + '/' + str(package.get('ph_mode')),
        ph_downstream=('absent_zero_residual' if package.get('ph_mode') == 'global' else 'none'),
        outer_test='NOT_RUN' if validation_only else 'RUN'))
    all_folds = []
    for task in selected_tasks:
        csv_path = Path(args.raw_root) / f'smi_{task}.csv'
        manifest_path = Path(args.split_root) / f'{task}.json'
        if validation_only and not manifest_path.is_file():
            raise FileNotFoundError(
                f'--smoke/--development requires an existing manifest: {manifest_path}')
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
            dataset = CleanLabeledDataset(
                source, targets,
                cache_capacity_bytes=int(args.clean_cache_gib * (1024 ** 3)))
            predictions = np.full(len(dataset), np.nan)
            visits = np.zeros(len(dataset), dtype=np.int64)
            results = []
            for fold in [f for f in manifest['folds'] if f['fold'] in selected_folds]:
                fold_id = fold['fold']
                fold_started = time.perf_counter()
                folder = output / task / f'fold{fold_id}'
                folder.mkdir(parents=True, exist_ok=False)
                set_global_seed(run_config['seed'] + fold_id)
                train, validation, test = [fold[f'{split}_indices']
                                           for split in ('train', 'validation', 'test')]
                scaler = scale_targets(dataset, task, train_indices=train,
                                       transform_mode='standard')

                def loader(indices, training=False):
                    return DataLoader(
                        Subset(dataset, indices),
                        batch_size=run_config['finetune_batch'] if training else run_config['eval_batch'],
                        shuffle=training, num_workers=0, collate_fn=dual_glt_collate,
                        generator=torch.Generator().manual_seed(run_config['seed'] + fold_id))

                train_loader, val_loader = loader(train, True), loader(validation)
                encoder = GLTGalPH(str(package['summary_mode']), package.get('ph_mode'),
                                   dropout=float(run_config.get('dropout', 0.1))).to(device)
                load_galformer_deployment(encoder, package,
                                          int(run_config['downstream_step']))
                model = GalformerDownstream(
                    encoder, readout=readout,
                    dropout=float(run_config.get('dropout', 0.1))).to(device)
                frozen_heads = model.freeze_training_only_heads()
                total_parameter_count = int(sum(p.numel() for p in model.parameters()))
                trainable_parameter_count = int(sum(p.numel() for p in model.parameters()
                                                    if p.requires_grad))
                optimizer = optimizer_for(model, run_config)
                scheduler = _cosine_scheduler(
                    optimizer, run_config['epochs'] * len(train_loader),
                    run_config['finetune_warmup'] * len(train_loader))
                validation_summary = {}
                best, best_r2, best_epoch, _ = fit_select_and_test(
                    model, scaler, train_loader, val_loader, None, device, optimizer,
                    scheduler, run_config, task=task, fold_id=fold_id,
                    validation_only=True, validation_summary=validation_summary)
                result = dict(
                    task=task, fold=fold_id, smoke=bool(args.smoke),
                    development=bool(args.development), protocol=protocol,
                    readout=readout, adaptation='full',
                    best_validation_r2=float(best_r2), best_epoch=int(best_epoch),
                    outer_test='NOT_RUN', validation_only=True,
                    scaler_fit_split='train',
                    deployment_step=int(run_config['downstream_step']),
                    pretrain_summary_mode=str(package['summary_mode']),
                    pretrain_ph_mode=str(package.get('ph_mode')),
                    pretrain_step=int(package.get('step', -1)),
                    ph_downstream=('absent_zero_residual'
                                   if package.get('ph_mode') == 'global' else 'none'),
                    frozen_training_only_heads=frozen_heads,
                    train_sample_count=int(len(train)),
                    validation_sample_count=int(len(validation)),
                    trainable_parameter_count=trainable_parameter_count,
                    total_parameter_count=total_parameter_count,
                    wall_seconds=float(time.perf_counter() - fold_started))
                result.update(validation_summary)
                save_checkpoint(folder / 'best.pt', dict(
                    state_dict=best, architecture=model.architecture_name,
                    summary_mode=encoder.summary_mode, ph_mode=encoder.ph_mode,
                    readout=readout, task=task, fold=fold_id, protocol=protocol,
                    split=fold, config=run_config, best_validation_r2=best_r2,
                    best_epoch=best_epoch, scaler_mean=scaler.scaler.mean_.tolist(),
                    scaler_scale=scaler.scaler.scale_.tolist()))
                write_json(folder / 'metrics.json', result)
                results.append(result)
                all_folds.append(result)
                del model, optimizer, scheduler
        finally:
            if dataset is not None:
                dataset.cache_stats()
            source.close()
    pd.DataFrame(all_folds).to_csv(output / 'all_fold_metrics.csv', index=False)
    summary = dict(
        protocol=protocol, readout=readout, adaptation='full',
        tasks={row['task']: row for row in all_folds},
        outer_test='NOT_RUN', validation_only=True,
        ph_downstream=('absent_zero_residual' if package.get('ph_mode') == 'global' else 'none'),
        interpretation=('development-fold validation-only evaluation; not OOF and not '
                        'independent blind evidence; PH is not provided at downstream '
                        'time in this round'))
    write_json(output / 'summary.json', summary)
    write_json(output / 'runtime.json', dict(
        status='PASS', pid=os.getpid(), command=sys.argv, protocol=protocol,
        development=bool(args.development), smoke=bool(args.smoke),
        selected_tasks=selected_tasks, selected_folds=selected_folds,
        readout=readout, outer_test='NOT_RUN',
        process_wall_seconds=float(time.perf_counter() - process_started)))


if __name__ == '__main__':
    main()
