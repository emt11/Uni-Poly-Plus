#!/usr/bin/env python3
"""Clean dual-encoder fine-tuning using fixed outer5_inner20 manifests."""
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
from src.modules.glt_dual import build_dual_glt_model
from src.modules.glt_dual_pretrain import load_deployment
from src.modules.glt_adaptation import (AdaptationAdapter, configure_adaptation,
                                         trainable_parameters)
from src.training.glt_dual_runtime import (TASKS, require_tmux, open_source, CleanLabeledDataset,
                                         save_checkpoint, write_json)
from src.utils import set_global_seed, scale_targets, train_epoch, evaluate, test_model, _cosine_scheduler


class EvaluationAdapter(AdaptationAdapter):
    """Existing evaluation utilities expect ``(prediction, auxiliary)``."""


def _rss_bytes():
    """Best-effort parent RSS for the timing provenance record."""

    try:
        for line in Path('/proc/self/status').read_text(encoding='utf-8').splitlines():
            if line.startswith('VmRSS:'):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError):
        return None
    return None


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


def optimizer_for(model, config, adaptation='full'):
    """Build an optimizer containing every and only trainable parameter.

    The old/full grouping is preserved.  HEAD and LoRA use a single explicit
    fusion/head group so frozen encoder tensors cannot silently enter AdamW.
    """
    adaptation = str(adaptation).lower()
    if adaptation in {'head', 'lora'}:
        params = [parameter for _, parameter in trainable_parameters(model)]
        if not params:
            raise RuntimeError(f'{adaptation} adaptation has no trainable parameters')
        return torch.optim.AdamW(
            [dict(params=params, lr=config['fusion_head_lr'], name=adaptation)],
            weight_decay=config['finetune_weight_decay'])
    groups, seen = [], set()
    for name, module, lr in [('o8', model.o8, config['encoder_lr']),
                             ('glt', model.glt, config['encoder_lr'])]:
        params = [p for p in module.parameters() if p.requires_grad]
        seen.update(id(p) for p in params)
        if params:
            groups.append(dict(params=params, lr=lr, name=name))
    params = [p for p in model.parameters() if p.requires_grad and id(p) not in seen]
    if params:
        groups.append(dict(params=params, lr=config['fusion_head_lr'], name='fusion_head'))
    if not groups:
        raise RuntimeError('full adaptation found no trainable parameters')
    return torch.optim.AdamW(groups, weight_decay=config['finetune_weight_decay'])


def _collect_features(encoder, dataset, indices, device, batch_size):
    """Collect frozen fused features without constructing a test loader."""
    loader = DataLoader(Subset(dataset, [int(i) for i in indices]),
                        batch_size=int(batch_size), shuffle=False, num_workers=0,
                        collate_fn=dual_glt_collate)
    rows, labels = [], []
    encoder.eval()
    with torch.inference_mode():
        for batch in loader:
            batch = batch.to(device, non_blocking=True)
            encoded = encoder.encode(batch)
            rows.append(encoder.fuse(encoded).detach().float().cpu().numpy())
            labels.append(batch.y.detach().float().cpu().numpy().reshape(-1))
    if not rows:
        raise RuntimeError('ridge feature collection is empty')
    return np.concatenate(rows, axis=0), np.concatenate(labels, axis=0)


def fit_ridge_validation(encoder, dataset, scaler, train_indices, validation_indices,
                         device, config, *, alpha=1.0):
    """Fit one train-only standardized Ridge readout and evaluate validation."""
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    x_train, y_train = _collect_features(
        encoder, dataset, train_indices, device, config['finetune_batch'])
    x_val, y_val = _collect_features(
        encoder, dataset, validation_indices, device, config['eval_batch'])
    feature_scaler = StandardScaler().fit(x_train)
    model = Ridge(alpha=float(alpha), fit_intercept=True)
    model.fit(feature_scaler.transform(x_train), y_train)
    predicted = model.predict(feature_scaler.transform(x_val)).reshape(-1, 1)
    actual = y_val.reshape(-1, 1)
    # Dataset labels are target-scaled; report the same original-unit metrics
    # as the neural validation path without fitting anything on validation.
    actual_original = scaler.inverse_transform(actual)
    predicted_original = scaler.inverse_transform(predicted)
    return {
        'validation_loss': float(np.mean((actual - predicted) ** 2)),
        'validation_r2': float(r2_score(actual_original, predicted_original)),
        'validation_mae': float(mean_absolute_error(actual_original, predicted_original)),
        'validation_rmse': float(np.sqrt(mean_squared_error(actual_original, predicted_original))),
        'ridge_alpha': float(alpha), 'feature_dim': int(x_train.shape[1]),
        'train_sample_count': int(len(train_indices)),
        'validation_sample_count': int(len(validation_indices)),
    }


def fit_select_and_test(model, scaler, train_loader, val_loader, test_loader, device,
                        optimizer, scheduler, config, *, task, fold_id,
                        validation_only=False, timing=False, timing_sink=None,
                        validation_summary=None):
    encoder, criterion = model.encoder, nn.MSELoss()
    best, best_r2, best_epoch, stalled = None, -float('inf'), -1, 0
    for epoch in range(config['epochs']):
        epoch_started = time.perf_counter()
        if timing:
            train_result = train_epoch(
                model, train_loader, criterion, optimizer, scheduler, device,
                epoch=epoch + 1, amp_dtype='fp32', max_grad_norm=1., fail_nonfinite=True,
                return_timing=True)
            train_timing = train_result[-1]
        else:
            train_epoch(model, train_loader, criterion, optimizer, scheduler, device,
                        epoch=epoch + 1, amp_dtype='fp32', max_grad_norm=1., fail_nonfinite=True)
            train_timing = None
        validation_started = time.perf_counter()
        val_loss, val_r2, val_targets, val_predictions = evaluate(
            model, val_loader, criterion, device, scaler=scaler, amp_dtype='fp32')
        if validation_summary is not None:
            val_mae = float(mean_absolute_error(val_targets, val_predictions))
            val_rmse = float(np.sqrt(mean_squared_error(val_targets, val_predictions)))
        else:
            # Keep the established helper contract for callers/tests that use
            # a lightweight evaluate stub returning no prediction arrays.
            val_mae = val_rmse = None
        validation_seconds = time.perf_counter() - validation_started
        finite_values = [val_loss, val_r2]
        if validation_summary is not None:
            finite_values.extend((val_mae, val_rmse))
        if not all(np.isfinite(value) for value in finite_values):
            raise FloatingPointError('nonfinite validation metrics')
        print(json.dumps(dict(task=task, fold=fold_id, epoch=epoch + 1,
                              validation_r2=val_r2)), flush=True)
        if val_r2 > best_r2:
            copy_started = time.perf_counter()
            best = {k: v.detach().cpu().clone() for k, v in encoder.state_dict().items()}
            best_copy_seconds = time.perf_counter() - copy_started
            best_r2, best_epoch, stalled = val_r2, epoch + 1, 0
            if validation_summary is not None:
                validation_summary.update(
                    validation_loss=float(val_loss),
                    validation_r2=float(val_r2),
                    validation_mae=val_mae,
                    validation_rmse=val_rmse,
                )
        else:
            best_copy_seconds = 0.0
            stalled += 1
        if timing and timing_sink is not None:
            timing_sink.append(dict(task=task, fold=int(fold_id), epoch=int(epoch + 1),
                                    train=train_timing,
                                    validation_seconds=float(validation_seconds),
                                    best_copy_seconds=float(best_copy_seconds),
                                    epoch_seconds=float(time.perf_counter() - epoch_started)))
        if stalled >= config['patience']:
            break
    if best is None:
        raise RuntimeError('no finite validation-selected checkpoint')
    encoder.load_state_dict(best, strict=True)
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
    parser.add_argument('--task', action='append', dest='tasks',
                        help='select task(s); required as a single task with --smoke')
    parser.add_argument('--fold', action='append', type=int, dest='folds',
                        help='select fold(s); required as a single fold with --smoke')
    parser.add_argument('--smoke', action='store_true',
                        help='bounded train/validation-only task/fold smoke; never reads outer-test')
    parser.add_argument('--development', action='store_true',
                        help='validation-only development mode; never reads outer-test')
    parser.add_argument('--formal-shard', action='store_true',
                        help='run exactly one task/fold including outer-test for a grid launcher')
    parser.add_argument('--adaptation', choices=('full', 'head', 'lora', 'ridge'),
                        help='downstream adaptation policy (default: config adaptation or full)')
    parser.add_argument('--lora-rank', type=int, default=8)
    parser.add_argument('--lora-alpha', type=float, default=8.0)
    parser.add_argument('--lora-dropout', type=float, default=0.0)
    parser.add_argument('--ridge-alpha', type=float, default=1.0)
    parser.add_argument('--clean-cache-gib', type=float, default=0.0,
                        help='bounded process-local clean Data cache capacity in GiB (0 disables it)')
    parser.add_argument('--timing', action='store_true',
                        help='write per-epoch preparation/training/validation timing')
    args = parser.parse_args()
    process_started = time.perf_counter()
    require_tmux()
    config = json.loads(Path(args.config).read_text(encoding='utf-8'))
    if config.get('use_md200') is not False:
        raise ValueError('dual route does not support MD200')
    if not np.isfinite(args.clean_cache_gib) or args.clean_cache_gib < 0:
        raise ValueError('--clean-cache-gib must be finite and non-negative')
    if args.smoke and args.development:
        raise ValueError('--smoke and --development are mutually exclusive')
    selected_tasks = list(args.tasks) if args.tasks else list(TASKS)
    selected_folds = [int(value) for value in args.folds] if args.folds else list(range(5))
    if args.development and not args.tasks:
        selected_tasks = ['xc', 'eps', 'eat']
    if args.development and not args.folds:
        selected_folds = [0, 1]
    unknown_tasks = sorted(set(selected_tasks) - set(TASKS))
    if unknown_tasks or any(value < 0 or value >= 5 for value in selected_folds):
        raise ValueError('task/fold selection is outside TASKS or outer5_inner20')
    if args.smoke:
        if len(selected_tasks) != 1 or len(selected_folds) != 1:
            raise ValueError('--smoke requires exactly one --task and one --fold')
    elif args.development:
        if not selected_tasks or not set(selected_tasks).issubset({'xc', 'eps', 'eat'}):
            raise ValueError('--development only permits xc, eps and eat')
        if not selected_folds or any(value not in (0, 1) for value in selected_folds):
            raise ValueError('--development only permits folds 0 and 1')
    elif args.formal_shard:
        if len(selected_tasks) != 1 or len(selected_folds) != 1:
            raise ValueError('--formal-shard requires exactly one --task and --fold')
    elif selected_tasks != list(TASKS) or selected_folds != list(range(5)):
        raise ValueError('partial task/fold selection requires --smoke or --formal-shard')
    run_config = dict(config)
    adaptation = str(args.adaptation or run_config.get('adaptation', 'full')).lower()
    run_config['adaptation'] = adaptation
    if args.smoke:
        run_config['epochs'] = min(int(run_config.get('epochs', 2)), 2)
    elif args.development:
        run_config['epochs'] = min(int(run_config.get('epochs', 30)), 30)
    if adaptation == 'lora' and (args.lora_rank <= 0 or args.lora_alpha <= 0 or args.lora_dropout < 0):
        raise ValueError('invalid LoRA rank/alpha/dropout')
    if adaptation == 'ridge' and (not np.isfinite(args.ridge_alpha) or args.ridge_alpha <= 0):
        raise ValueError('--ridge-alpha must be finite and positive')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    package = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    validation_only = bool(args.smoke or args.development)
    if adaptation == 'ridge' and not validation_only:
        raise ValueError('ridge adaptation is validation-only in GLT-PRED S1')
    write_json(output / 'run.json', dict(config=run_config, command=sys.argv,
        protocol=('outer5_inner20_smoke' if args.smoke else
                  'outer5_inner20_development' if args.development else
                  'outer5_inner20_formal_shard' if args.formal_shard else
                  'outer5_inner20'),
        smoke=bool(args.smoke), development=bool(args.development),
        formal_shard=bool(args.formal_shard), adaptation=adaptation,
        lora_rank=int(args.lora_rank), lora_alpha=float(args.lora_alpha),
        lora_dropout=float(args.lora_dropout), ridge_alpha=float(args.ridge_alpha),
        selected_tasks=selected_tasks, selected_folds=selected_folds,
        outer_test='NOT_RUN' if validation_only else 'RUN'))
    all_folds, task_summary, timing_records = [], {}, []
    source_open_records, cache_records, save_records = [], [], []
    for task in selected_tasks:
        csv_path = Path(args.raw_root) / f'smi_{task}.csv'
        manifest_path = Path(args.split_root) / f'{task}.json'
        if validation_only and not manifest_path.is_file():
            raise FileNotFoundError(
                f'--smoke requires an existing outer5_inner20 manifest: {manifest_path}'
            )
        manifest = fixed_manifest(task, csv_path, manifest_path)
        source_started = time.perf_counter()
        source, frame = open_source(
            args.cohort_root, args.cache_root, task=task,
            dual_static_root=args.dual_static_root,
        )
        source_open_records.append(dict(task=task,
                                        seconds=float(time.perf_counter() - source_started),
                                        rows=int(len(frame))))
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
                cache_capacity_bytes=int(args.clean_cache_gib * (1024 ** 3)),
            )
            predictions = np.full(len(dataset), np.nan)
            visits = np.zeros(len(dataset), dtype=np.int64)
            results = []
            folds = [fold for fold in manifest['folds'] if fold['fold'] in selected_folds]
            for fold in folds:
                fold_id = fold['fold']
                fold_started = time.perf_counter()
                folder = output / task / f'fold{fold_id}'
                folder.mkdir(parents=True, exist_ok=False)
                set_global_seed(run_config['seed'] + fold_id)
                train, validation, test = [fold[f'{split}_indices'] for split in ('train', 'validation', 'test')]
                scaler = scale_targets(dataset, task, train_indices=train, transform_mode='standard')
                def loader(indices, training=False):
                    return DataLoader(Subset(dataset, indices),
                        batch_size=run_config['finetune_batch'] if training else run_config['eval_batch'],
                        shuffle=training, num_workers=0, collate_fn=dual_glt_collate,
                        generator=torch.Generator().manual_seed(run_config['seed'] + fold_id))
                train_loader, val_loader = loader(train, True), loader(validation)
                encoder = build_dual_glt_model(run_config['fusion_mode']).to(device)
                load_deployment(encoder, package, run_config['downstream_step'])
                adaptation_meta = configure_adaptation(
                    encoder, adaptation, rank=args.lora_rank,
                    alpha=args.lora_alpha, dropout=args.lora_dropout)
                total_parameter_count = int(sum(parameter.numel()
                    for parameter in encoder.parameters()))
                trainable_parameter_count = int(sum(parameter.numel()
                    for parameter in encoder.parameters() if parameter.requires_grad))
                if adaptation == 'ridge':
                    ridge_result = fit_ridge_validation(
                        encoder, dataset, scaler, train, validation, device,
                        run_config, alpha=args.ridge_alpha)
                    ridge_result.update(
                        task=task, fold=int(fold_id), adaptation='ridge',
                        protocol=('outer5_inner20_development'
                                  if args.development else 'outer5_inner20_smoke'),
                        outer_test='NOT_RUN', validation_only=True,
                        scaler_fit_split='train',
                        deployment_step=int(run_config['downstream_step']),
                        trainable_parameter_count=trainable_parameter_count,
                        total_parameter_count=total_parameter_count,
                        wall_seconds=float(time.perf_counter() - fold_started))
                    write_json(folder / 'metrics.json', ridge_result)
                    results.append(ridge_result)
                    all_folds.append(ridge_result)
                    del encoder
                    continue
                frozen_modules = [] if adaptation == 'full' else [encoder.o8, encoder.glt]
                if hasattr(encoder, 'norm2'):
                    frozen_modules.extend([encoder.norm2, encoder.norm3])
                if hasattr(encoder, 'kfuse'):
                    frozen_modules.append(encoder.kfuse)
                model = EvaluationAdapter(encoder).register_frozen(*frozen_modules)
                optimizer = optimizer_for(encoder, run_config, adaptation=adaptation)
                scheduler = _cosine_scheduler(optimizer, run_config['epochs'] * len(train_loader),
                                               run_config['finetune_warmup'] * len(train_loader))
                if validation_only:
                    validation_summary = {}
                    best, best_r2, best_epoch, _ = fit_select_and_test(
                        model, scaler, train_loader, val_loader, None, device,
                        optimizer, scheduler, run_config, task=task, fold_id=fold_id,
                        validation_only=True, timing=args.timing, timing_sink=timing_records,
                        validation_summary=validation_summary)
                    smoke_result = dict(
                        task=task, fold=fold_id, smoke=bool(args.smoke),
                        development=bool(args.development), adaptation=adaptation,
                        protocol=('outer5_inner20_development'
                                  if args.development else 'outer5_inner20_smoke'),
                        best_validation_r2=float(best_r2), best_epoch=int(best_epoch),
                        outer_test='NOT_RUN', validation_only=True,
                        scaler_fit_split='train', deployment_step=int(run_config['downstream_step']),
                        train_sample_count=int(len(train)),
                        validation_sample_count=int(len(validation)),
                        trainable_parameter_count=trainable_parameter_count,
                        total_parameter_count=total_parameter_count,
                        wall_seconds=float(time.perf_counter() - fold_started),
                    )
                    smoke_result.update(validation_summary)
                    save_started = time.perf_counter()
                    save_checkpoint(folder / 'best.pt', dict(
                        state_dict=best, fusion_mode=run_config['fusion_mode'],
                        architecture=encoder.architecture_name, task=task, fold=fold_id,
                        protocol=('outer5_inner20_development'
                                  if args.development else 'outer5_inner20_smoke'),
                        split=fold, config=run_config,
                        best_validation_r2=best_r2, best_epoch=best_epoch,
                        scaler_mean=scaler.scaler.mean_.tolist(),
                        scaler_scale=scaler.scaler.scale_.tolist(),
                    ))
                    write_json(folder / 'metrics.json', smoke_result)
                    save_records.append(dict(task=task, fold=int(fold_id),
                                             kind='validation_only_best_and_metrics',
                                             seconds=float(time.perf_counter() - save_started)))
                    results.append(smoke_result)
                    all_folds.append(smoke_result)
                    del model, encoder, optimizer, scheduler
                    continue
                best, best_r2, best_epoch, result = fit_select_and_test(
                    model, scaler, train_loader, val_loader, loader(test), device,
                    optimizer, scheduler, run_config, task=task, fold_id=fold_id,
                    timing=args.timing, timing_sink=timing_records)
                y_true, y_pred = result.pop('_y_true'), result.pop('_y_pred')
                if not np.isfinite(y_pred).all() or not all(np.isfinite(v) for v in result.values()):
                    raise FloatingPointError('nonfinite test predictions/metrics')
                if visits[test].any():
                    raise ValueError('outer test indices predicted more than once')
                predictions[test], visits[test] = y_pred, visits[test] + 1
                result.update(task=task, fold=fold_id, protocol='outer5_inner20',
                              formal_shard=bool(args.formal_shard),
                              best_validation_r2=best_r2, best_epoch=best_epoch)
                save_started = time.perf_counter()
                pd.DataFrame(dict(row_index=test, target=y_true, prediction=y_pred)).to_csv(folder / 'predictions.csv', index=False)
                save_checkpoint(folder / 'best.pt', dict(state_dict=best, fusion_mode=run_config['fusion_mode'],
                    architecture=encoder.architecture_name, task=task, fold=fold_id, protocol='outer5_inner20',
                    split=fold, config=run_config, best_validation_r2=best_r2, best_epoch=best_epoch,
                    scaler_mean=scaler.scaler.mean_.tolist(), scaler_scale=scaler.scaler.scale_.tolist()))
                write_json(folder / 'metrics.json', result)
                save_records.append(dict(task=task, fold=int(fold_id),
                                         kind='formal_predictions_best_and_metrics',
                                         seconds=float(time.perf_counter() - save_started)))
                results.append(result)
                all_folds.append(result)
                del model, encoder, optimizer, scheduler
            if args.formal_shard:
                task_summary[task] = dict(
                    formal_shard=True, task=task, fold=int(selected_folds[0]),
                    outer_test='RUN_ONCE', test_metrics=results[0] if results else None,
                )
                continue
            if not (visits == 1).all():
                if validation_only:
                    task_summary[task] = dict(
                        smoke=bool(args.smoke), development=bool(args.development),
                        adaptation=adaptation, validation_only=True,
                        best_validation_r2=float(results[0].get('best_validation_r2',
                                                                results[0].get('validation_r2', float('nan')))),
                        best_epoch=int(results[0].get('best_epoch', 0)),
                        outer_test='NOT_RUN',
                    )
                    continue
                raise ValueError('incomplete OOF coverage')
            task_summary[task] = {metric: dict(mean=float(np.mean([r[metric] for r in results])),
                std=float(np.std([r[metric] for r in results], ddof=1))) for metric in ('test_r2', 'test_mae', 'test_rmse')}
            task_summary[task]['pooled_oof'] = dict(r2=float(r2_score(targets, predictions)),
                mae=float(mean_absolute_error(targets, predictions)), rmse=float(np.sqrt(mean_squared_error(targets, predictions))))
            pd.DataFrame(dict(row_index=np.arange(len(targets)), target=targets, prediction=predictions)).to_csv(output / task / 'oof.csv', index=False)
        finally:
            if dataset is not None:
                cache_records.append(dict(task=task, stats=dataset.cache_stats()))
            source.close()
    final_save_started = time.perf_counter()
    pd.DataFrame(all_folds).to_csv(output / 'all_fold_metrics.csv', index=False)
    if args.formal_shard:
        summary = dict(protocol='outer5_inner20_formal_shard', formal_shard=True,
            tasks=task_summary, outer_test='RUN_ONCE',
            interpretation='one isolated task/fold shard; aggregate only after all 40 shards')
    elif validation_only:
        summary = dict(
            protocol=('outer5_inner20_development'
                      if args.development else 'outer5_inner20_smoke'),
            smoke=bool(args.smoke), development=bool(args.development),
            adaptation=adaptation,
            tasks=task_summary, outer_test='NOT_RUN',
            interpretation=('development-fold validation-only evaluation; not OOF or independent blind evidence'
                            if args.development else
                            'single selected task/fold validation-only smoke; not OOF or macro8'))
    else:
        summary = dict(protocol='outer5_inner20', tasks=task_summary,
            macro8_r2=float(np.mean([r['test_r2']['mean'] for r in task_summary.values()])),
            std_definition='sample std across 5 folds (ddof=1)',
            interpretation='development-fold evaluation; not an independent blind test')
    write_json(output / 'summary.json', summary)
    if args.timing:
        write_json(output / 'timing.json', dict(
            protocol=summary['protocol'], records=timing_records,
            source_open=source_open_records, cache=cache_records,
            checkpoint_and_metric_saves=save_records,
        ))
    finalization_seconds = time.perf_counter() - final_save_started
    write_json(output / 'runtime.json', dict(
        status='PASS', pid=os.getpid(), command=sys.argv,
        protocol=summary['protocol'], smoke=bool(args.smoke), development=bool(args.development),
        formal_shard=bool(args.formal_shard), selected_tasks=selected_tasks,
        selected_folds=selected_folds, adaptation=adaptation,
        outer_test='NOT_RUN' if validation_only else 'RUN',
        clean_cache_gib=float(args.clean_cache_gib), timing=bool(args.timing),
        source_open=source_open_records, cache=cache_records,
        checkpoint_and_metric_saves=save_records,
        finalization_seconds=float(finalization_seconds),
        process_wall_seconds=float(time.perf_counter() - process_started),
        rss_bytes_at_end=_rss_bytes(),
    ))


if __name__ == '__main__':
    main()
