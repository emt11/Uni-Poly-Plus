#!/usr/bin/env python3
"""Clean dual-encoder fine-tuning using fixed outer5_inner20 manifests."""
import argparse
import json
from pathlib import Path
import sys

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
from src.training.glt_dual_runtime import (TASKS, require_tmux, open_source, CleanLabeledDataset,
                                         save_checkpoint, write_json)
from src.utils import set_global_seed, scale_targets, train_epoch, evaluate, test_model, _cosine_scheduler


class EvaluationAdapter(nn.Module):
    """Existing evaluation utilities expect (prediction, auxiliary)."""
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, data):
        return self.encoder(data), None


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
    groups, seen = [], set()
    for name, module, lr in [('o8', model.o8, config['encoder_lr']),
                             ('glt', model.glt, config['encoder_lr'])]:
        params = list(module.parameters())
        seen.update(id(p) for p in params)
        groups.append(dict(params=params, lr=lr, name=name))
    groups.append(dict(params=[p for p in model.parameters() if id(p) not in seen],
                       lr=config['fusion_head_lr'], name='fusion_head'))
    return torch.optim.AdamW(groups, weight_decay=config['finetune_weight_decay'])


def fit_select_and_test(model, scaler, train_loader, val_loader, test_loader, device,
                        optimizer, scheduler, config, *, task, fold_id,
                        validation_only=False):
    encoder, criterion = model.encoder, nn.MSELoss()
    best, best_r2, best_epoch, stalled = None, -float('inf'), -1, 0
    for epoch in range(config['epochs']):
        train_epoch(model, train_loader, criterion, optimizer, scheduler, device,
                    epoch=epoch + 1, amp_dtype='fp32', max_grad_norm=1., fail_nonfinite=True)
        val_loss, val_r2, _, _ = evaluate(model, val_loader, criterion, device,
                                         scaler=scaler, amp_dtype='fp32')
        if not np.isfinite(val_r2) or not np.isfinite(val_loss):
            raise FloatingPointError('nonfinite validation metrics')
        print(json.dumps(dict(task=task, fold=fold_id, epoch=epoch + 1,
                              validation_r2=val_r2)), flush=True)
        if val_r2 > best_r2:
            best = {k: v.detach().cpu().clone() for k, v in encoder.state_dict().items()}
            best_r2, best_epoch, stalled = val_r2, epoch + 1, 0
        else:
            stalled += 1
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
    for name in ('config', 'checkpoint', 'raw-root', 'topology-root', 'trimer-root', 'output'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--split-root', default='data/splits/mips_outer5_inner20')
    parser.add_argument('--task', action='append', dest='tasks',
                        help='select task(s); required as a single task with --smoke')
    parser.add_argument('--fold', action='append', type=int, dest='folds',
                        help='select fold(s); required as a single fold with --smoke')
    parser.add_argument('--smoke', action='store_true',
                        help='bounded train/validation-only task/fold smoke; never reads outer-test')
    args = parser.parse_args()
    require_tmux()
    config = json.loads(Path(args.config).read_text(encoding='utf-8'))
    if config.get('use_md200') is not False:
        raise ValueError('dual route does not support MD200')
    selected_tasks = list(args.tasks) if args.tasks else list(TASKS)
    selected_folds = [int(value) for value in args.folds] if args.folds else list(range(5))
    unknown_tasks = sorted(set(selected_tasks) - set(TASKS))
    if unknown_tasks or any(value < 0 or value >= 5 for value in selected_folds):
        raise ValueError('task/fold selection is outside TASKS or outer5_inner20')
    if args.smoke:
        if len(selected_tasks) != 1 or len(selected_folds) != 1:
            raise ValueError('--smoke requires exactly one --task and one --fold')
    elif selected_tasks != list(TASKS) or selected_folds != list(range(5)):
        raise ValueError('partial task/fold selection requires --smoke')
    run_config = dict(config)
    if args.smoke:
        run_config['epochs'] = min(int(run_config.get('epochs', 2)), 2)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    package = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    write_json(output / 'run.json', dict(config=run_config, command=sys.argv,
        protocol='outer5_inner20_smoke' if args.smoke else 'outer5_inner20',
        smoke=bool(args.smoke), selected_tasks=selected_tasks,
        selected_folds=selected_folds, outer_test='NOT_RUN' if args.smoke else 'RUN'))
    all_folds, task_summary = [], {}
    for task in selected_tasks:
        csv_path = Path(args.raw_root) / f'smi_{task}.csv'
        manifest_path = Path(args.split_root) / f'{task}.json'
        if args.smoke and not manifest_path.is_file():
            raise FileNotFoundError(
                f'--smoke requires an existing outer5_inner20 manifest: {manifest_path}'
            )
        manifest = fixed_manifest(task, csv_path, manifest_path)
        source, frame = open_source(csv_path, args.topology_root, args.trimer_root)
        try:
            targets = frame.iloc[:, 1].to_numpy(dtype=np.float64)
            if not np.isfinite(targets).all():
                raise ValueError('nonfinite labels')
            dataset = CleanLabeledDataset(source, targets)
            predictions = np.full(len(dataset), np.nan)
            visits = np.zeros(len(dataset), dtype=np.int64)
            results = []
            folds = [fold for fold in manifest['folds'] if fold['fold'] in selected_folds]
            for fold in folds:
                fold_id = fold['fold']
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
                model = EvaluationAdapter(encoder)
                optimizer = optimizer_for(encoder, run_config)
                scheduler = _cosine_scheduler(optimizer, run_config['epochs'] * len(train_loader),
                                               run_config['finetune_warmup'] * len(train_loader))
                if args.smoke:
                    best, best_r2, best_epoch, _ = fit_select_and_test(
                        model, scaler, train_loader, val_loader, None, device,
                        optimizer, scheduler, run_config, task=task, fold_id=fold_id,
                        validation_only=True)
                    smoke_result = dict(
                        task=task, fold=fold_id, smoke=True,
                        protocol='outer5_inner20_smoke',
                        best_validation_r2=float(best_r2), best_epoch=int(best_epoch),
                        outer_test='NOT_RUN', validation_only=True,
                        scaler_fit_split='train', deployment_step=int(run_config['downstream_step']),
                    )
                    save_checkpoint(folder / 'best.pt', dict(
                        state_dict=best, fusion_mode=run_config['fusion_mode'],
                        architecture=encoder.architecture_name, task=task, fold=fold_id,
                        protocol='outer5_inner20_smoke', split=fold, config=run_config,
                        best_validation_r2=best_r2, best_epoch=best_epoch,
                        scaler_mean=scaler.scaler.mean_.tolist(),
                        scaler_scale=scaler.scaler.scale_.tolist(),
                    ))
                    write_json(folder / 'metrics.json', smoke_result)
                    results.append(smoke_result)
                    all_folds.append(smoke_result)
                    del model, encoder, optimizer, scheduler
                    continue
                best, best_r2, best_epoch, result = fit_select_and_test(
                    model, scaler, train_loader, val_loader, loader(test), device,
                    optimizer, scheduler, run_config, task=task, fold_id=fold_id)
                y_true, y_pred = result.pop('_y_true'), result.pop('_y_pred')
                if not np.isfinite(y_pred).all() or not all(np.isfinite(v) for v in result.values()):
                    raise FloatingPointError('nonfinite test predictions/metrics')
                if visits[test].any():
                    raise ValueError('outer test indices predicted more than once')
                predictions[test], visits[test] = y_pred, visits[test] + 1
                result.update(task=task, fold=fold_id, best_validation_r2=best_r2, best_epoch=best_epoch)
                pd.DataFrame(dict(row_index=test, target=y_true, prediction=y_pred)).to_csv(folder / 'predictions.csv', index=False)
                save_checkpoint(folder / 'best.pt', dict(state_dict=best, fusion_mode=run_config['fusion_mode'],
                    architecture=encoder.architecture_name, task=task, fold=fold_id, protocol='outer5_inner20',
                    split=fold, config=run_config, best_validation_r2=best_r2, best_epoch=best_epoch,
                    scaler_mean=scaler.scaler.mean_.tolist(), scaler_scale=scaler.scaler.scale_.tolist()))
                write_json(folder / 'metrics.json', result)
                results.append(result)
                all_folds.append(result)
                del model, encoder, optimizer, scheduler
            if not (visits == 1).all():
                if args.smoke:
                    task_summary[task] = dict(
                        smoke=True, validation_only=True,
                        best_validation_r2=float(results[0]['best_validation_r2']),
                        best_epoch=int(results[0]['best_epoch']),
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
            source.close()
    pd.DataFrame(all_folds).to_csv(output / 'all_fold_metrics.csv', index=False)
    if args.smoke:
        summary = dict(protocol='outer5_inner20_smoke', smoke=True,
            tasks=task_summary, outer_test='NOT_RUN',
            interpretation='single selected task/fold validation-only smoke; not OOF or macro8')
    else:
        summary = dict(protocol='outer5_inner20', tasks=task_summary,
            macro8_r2=float(np.mean([r['test_r2']['mean'] for r in task_summary.values()])),
            std_definition='sample std across 5 folds (ddof=1)',
            interpretation='development-fold evaluation; not an independent blind test')
    write_json(output / 'summary.json', summary)


if __name__ == '__main__':
    main()
