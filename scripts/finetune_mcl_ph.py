#!/usr/bin/env python3
"""MCL-PH downstream arms (MCL-PH-20260921-01/r1).

Five arms that share one protocol and differ only in the encoder family that
was pre-trained:

| arm | pre-trained route | 3D branch | downstream head |
| --- | --- | --- | --- |
| ``glt_ref``  | the pre-existing dual GLT route | GLT    | the original 1024-wide concat head |
| ``o8_only``  | the ``glt_ref`` package's O8       | none   | the new 512-wide head |
| ``m_cat``    | this contract, ``F_CAT``           | MCL-PH | the new 512-wide head |
| ``m_gate``   | this contract, ``F_GATE``          | MCL-PH | the new 512-wide head |
| ``m_xattn``  | this contract, ``F_XATTN``         | MCL-PH | the new 512-wide head |

Only train and validation records and label values of the requested task/fold
are decoded/materialised; the outer-test indices are read for split coverage.
The current frozen JSONL still needs a raw-byte scan for file integrity and row
location, so it does not provide byte-level isolation from unselected labels.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset

from src.dataset.glt_dual import dual_glt_collate
from src.dataset.mcl_ph_view import (load_geometric_statistics, mcl_ph_collate,
                                     prepare_mcl_ph_sample)
from src.modules.glt_dual import BondPathO8, build_dual_glt_model, mean_pool
from src.modules.glt_dual_pretrain import load_deployment as load_dual_deployment
from src.modules.mcl_ph import build_fusion
from src.modules.mcl_ph import load_deployment as load_mcl_deployment
from src.modules.glt_adaptation import AdaptationAdapter
from src.training.glt_dual_runtime import (open_source, require_tmux, save_checkpoint,
                                           sha256_file, write_json)
from src.utils import _cosine_scheduler, evaluate, scale_targets, set_global_seed, train_epoch
from scripts.finetune_glt_3d_gain_d2 import resolve_fold

ARMS = ('glt_ref', 'o8_only', 'm_cat', 'm_gate', 'm_xattn')
MCL_FUSION = {'m_cat': 'cat', 'm_gate': 'gate', 'm_xattn': 'xattn'}
BACKBONE_LR = 1e-5
HEAD_LR = 1e-4
WEIGHT_DECAY = 0.02
SCHEDULE_TOTAL_EPOCHS = 30
SCHEDULE_WARMUP_EPOCHS = 5
HEAD_INIT_SEED = 20260921
TASKS = ('xc', 'eps', 'eat')
FOLDS = (0, 1)
SPLIT_PROTOCOL = 'outer5_inner20'


class DownstreamHead(nn.Module):
    """The declared head: mean pool, one LayerNorm, two dense layers."""

    def __init__(self, hidden=512, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(hidden)
        self.net = nn.Sequential(nn.Linear(hidden, 256), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(256, 1))

    def forward(self, pooled):
        return self.net(self.norm(pooled))


def build_head(seed=HEAD_INIT_SEED, dropout=0.1):
    """One shared initial head tensor for every 512-wide arm."""
    previous = torch.get_rng_state()
    torch.manual_seed(int(seed))
    head = DownstreamHead(512, dropout)
    torch.set_rng_state(previous)
    return head


class MCLPHArm(nn.Module):
    """One pre-trained MCL-PH encoder plus the freshly created downstream head."""

    def __init__(self, fusion_mode, dropout=0.1):
        from src.modules.mcl_ph import MCLPHEncoder

        super().__init__()
        self.fusion_mode = str(fusion_mode)
        self.encoder = MCLPHEncoder(fusion_mode, dropout=dropout)
        self.head = build_head(dropout=dropout)

    def forward(self, batch):
        encoded = self.encoder.encode(batch, atom_mask=None)
        fused = self.encoder.fuse(encoded)
        pooled = mean_pool(fused, batch.canonical_graph_index, batch.graph_available.numel())
        return self.head(pooled)


class O8OnlyArm(nn.Module):
    """The O8 branch alone behind the new 512-wide head; no 3D instance exists."""

    def __init__(self, dropout=0.1):
        super().__init__()
        self.o8 = BondPathO8(dropout)
        self.head = build_head(dropout=dropout)

    def forward(self, batch):
        atoms, _ = self.o8(batch, atom_mask=None)
        pooled = mean_pool(atoms, batch.canonical_graph_index, batch.graph_available.numel())
        return self.head(pooled)


class GLTReferenceArm(nn.Module):
    """The pre-existing dual route with its original 1024-wide concat head."""

    def __init__(self, dropout=0.1, torsion=False):
        super().__init__()
        self.model = build_dual_glt_model('concat', dropout=dropout, torsion=torsion)

    def forward(self, batch):
        return self.model(batch)


class MCLPHCleanDataset(Dataset):
    """Clean downstream samples: no mask, frozen coordinates as the 3D view."""

    def __init__(self, source, targets, statistics):
        self.source = source
        self.raw_targets = np.asarray(targets, dtype=np.float64)
        self.targets = self.raw_targets.copy()
        self.statistics = statistics
        if self.raw_targets.reshape(-1).size != len(source):
            raise ValueError('downstream target count differs from the frozen source')

    def set_target_override(self, values):
        values = np.asarray(values).reshape(-1)
        if values.size != len(self.source):
            raise ValueError('target override count differs from the frozen source')
        self.targets = values

    def __len__(self):
        return len(self.source)

    def __getitem__(self, index):
        index = int(index)
        key = bytes(self.source.samples[index][0]).hex()
        data, labels = prepare_mcl_ph_sample(
            *self.source[index], seed=42, key=key, position=index, sigma=0.0, ratio=None,
            static=self.source.static_for(index), statistics=self.statistics, view='clean')
        data.y = torch.tensor([float(self.targets[index])], dtype=torch.float32)
        return data, labels


class CleanDualDataset(Dataset):
    """Clean dual-route samples for the two reference arms."""

    def __init__(self, source, targets):
        from src.dataset.glt_dual import build_dual_sample

        self.source = source
        self.raw_targets = np.asarray(targets, dtype=np.float64)
        self.targets = self.raw_targets.copy()
        self._build = build_dual_sample

    def set_target_override(self, values):
        values = np.asarray(values).reshape(-1)
        if values.size != len(self.source):
            raise ValueError('target override count differs from the frozen source')
        self.targets = values

    def __len__(self):
        return len(self.source)

    def __getitem__(self, index):
        index = int(index)
        data = self._build(*self.source[index], static=self.source.static_for(index))
        data.y = torch.tensor([float(self.targets[index])], dtype=torch.float32)
        return data


def build_mcl_arm(fusion_mode, package, *, dropout=0.1, expected_step):
    """Assemble encoder + head and load the pre-trained tensors strictly."""
    arm = MCLPHArm(fusion_mode, dropout)
    load_mcl_deployment(arm.encoder, package, expected_step, expected_fusion=fusion_mode)
    return arm


def build_o8_only(package, *, dropout=0.1, expected_step, torsion=False):
    """The reference O8 alone behind the new head."""
    reference = build_dual_glt_model('concat', dropout=dropout, torsion=torsion)
    load_dual_deployment(reference, package, expected_step)
    arm = O8OnlyArm(dropout)
    source_state = reference.state_dict()
    target_state = arm.state_dict()
    copied = []
    for name, value in target_state.items():
        if not name.startswith('o8.'):
            continue
        if name not in source_state:
            raise ValueError(f'O8 tensor {name} is absent from the reference package')
        if tuple(source_state[name].shape) != tuple(value.shape):
            raise ValueError(f'O8 tensor {name} has a different shape')
        with torch.no_grad():
            value.copy_(source_state[name])
        copied.append(name)
    expected = [name for name in source_state if name.startswith('o8.')]
    if sorted(copied) != sorted(expected):
        raise ValueError('the O8 branch was not fully copied from the reference package')
    return arm, copied


def parameter_groups(model, arm):
    """Two learning rates with weight decay disabled for bias, LN and router."""
    head = model.head if hasattr(model, 'head') else model.model.predictor
    groups = {'backbone': [], 'backbone_no_decay': [], 'head': [], 'head_no_decay': []}

    def decay_free(name):
        return (name.endswith('.bias') or 'norm' in name.lower() or 'router' in name.lower()
                or name.endswith('.gate.weight'))

    head_ids = {id(parameter) for parameter in head.parameters()}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if id(parameter) in head_ids:
            groups['head_no_decay' if decay_free(name) else 'head'].append(parameter)
        else:
            groups['backbone_no_decay' if decay_free(name) else 'backbone'].append(parameter)
    entries, evidence = [], []
    for key, lr in (('backbone', BACKBONE_LR), ('backbone_no_decay', BACKBONE_LR),
                    ('head', HEAD_LR), ('head_no_decay', HEAD_LR)):
        if not groups[key]:
            continue
        decay = 0.0 if key.endswith('no_decay') else WEIGHT_DECAY
        entries.append({'params': groups[key], 'lr': lr, 'weight_decay': decay, 'name': key})
        evidence.append({'name': key, 'lr': lr, 'weight_decay': decay,
                         'num_tensors': len(groups[key]),
                         'num_parameters': int(sum(p.numel() for p in groups[key]))})
    optimizer = torch.optim.AdamW(entries)
    grouped = {id(p) for group in entries for p in group['params']}
    missing = [name for name, parameter in model.named_parameters()
               if parameter.requires_grad and id(parameter) not in grouped]
    if missing:
        raise ValueError('trainable tensors outside every group: ' + ','.join(missing[:5]))
    return optimizer, evidence


def run_selection_epochs(model, train_loader, validation_loader, criterion, optimizer,
                         scheduler, device, *, epochs, patience, scaler, progress=None):
    """Train at most ``epochs`` epochs and select on validation R2 only.

    The validation predictions of the selected epoch are kept exactly as they
    were computed for that epoch; no refit and no outer-test read exists here.
    """
    wrapped = AdaptationAdapter(model)
    best_state, best_r2, best_epoch, stalled = None, -float('inf'), -1, 0
    best_predictions = None
    history, updates = [], 0
    for epoch in range(int(epochs)):
        train_result = train_epoch(wrapped, train_loader, criterion, optimizer, scheduler,
                                   device, epoch=epoch + 1, amp_dtype='fp32',
                                   max_grad_norm=1.0, fail_nonfinite=True,
                                   return_timing=True)
        updates += int(train_result[-1]['training_steps'])
        validation_loss, validation_r2, targets, predictions = evaluate(
            wrapped, validation_loader, criterion, device, scaler=scaler, amp_dtype='fp32')
        if not all(np.isfinite([validation_loss, validation_r2])):
            raise FloatingPointError('nonfinite validation metrics')
        if not np.isfinite(predictions).all():
            raise FloatingPointError('nonfinite validation predictions')
        record = {'epoch': epoch + 1, 'train_loss': float(train_result[0]),
                  'validation_loss': float(validation_loss),
                  'validation_r2': float(validation_r2),
                  'learning_rates': [group['lr'] for group in optimizer.param_groups],
                  'training_steps': int(train_result[-1]['training_steps'])}
        history.append(record)
        if progress is not None:
            progress(record)
        if validation_r2 > best_r2:
            best_state = {key: value.detach().cpu().clone()
                          for key, value in model.state_dict().items()}
            best_predictions = (np.asarray(targets, dtype=np.float64).reshape(-1),
                                np.asarray(predictions, dtype=np.float64).reshape(-1))
            best_r2, best_epoch, stalled = validation_r2, epoch + 1, 0
        else:
            stalled += 1
        if stalled >= int(patience):
            break
    if best_state is None:
        raise RuntimeError('no finite validation-selected checkpoint')
    return {'history': history, 'state': best_state, 'best_r2': float(best_r2),
            'best_epoch': int(best_epoch), 'stalled_epochs': int(stalled),
            'optimizer_updates': int(updates), 'predictions': best_predictions}


def unit_directory(output, arm, task, fold):
    if arm not in ARMS:
        raise ValueError(f'unknown arm: {arm}')
    if task not in TASKS:
        raise ValueError(f'unknown task: {task}')
    if int(fold) not in FOLDS:
        raise ValueError(f'unknown fold: {fold}')
    return Path(output) / str(arm) / str(task) / f'fold{int(fold)}'


def compact_train_validation_indices(train_indices, validation_indices):
    """Return only train/validation task rows and their compact dataset indices."""
    train_indices = [int(index) for index in train_indices]
    validation_indices = [int(index) for index in validation_indices]
    if set(train_indices) & set(validation_indices):
        raise ValueError('train and validation indices overlap')
    selected = sorted(set(train_indices) | set(validation_indices))
    if not selected:
        raise ValueError('the selected train/validation rows are empty')
    compact = {original: index for index, original in enumerate(selected)}
    return (selected,
            [compact[index] for index in train_indices],
            [compact[index] for index in validation_indices])


def mcl_ph_downstream_collate(records):
    """Pack one downstream batch: a single ``Data``, no supervision targets.

    ``mcl_ph_collate`` returns ``(batch, labels)`` because the pre-training loss
    needs the geometry targets.  The downstream loop feeds one object to the
    model and reads ``batch.y``, so the labels are dropped here while the
    per-graph validity flags stay on the batch.
    """
    batch, _labels = mcl_ph_collate(list(records))
    return batch


def build_summary(common, *, config, device, command):
    """The successful unit summary, built from the run's own ``common`` record.

    ``common`` already carries ``optimizer_groups``, so it must arrive here by
    expansion and never as an explicit keyword: passing it both ways raises
    ``TypeError: dict() got multiple values for keyword argument`` at the end of
    an otherwise successful unit.
    """
    return dict(status='PASS', command=command, config=config, device=str(device), **{
        key: value for key, value in common.items()
        if key not in ('history', 'load_state_dict_result')})


def run_unit(args, folder, started, statistics, config, manifest):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    package = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    expected = int(args.expected_pretrain_step)
    if args.arm == 'glt_ref':
        torsion = bool(package.get('torsion_modules', False))
        model = GLTReferenceArm(torsion=torsion)
        load_dual_deployment(model.model, package, expected)
        collate = dual_glt_collate
        dataset_class = CleanDualDataset
        copied = None
    elif args.arm == 'o8_only':
        torsion = bool(package.get('torsion_modules', False))
        model, copied = build_o8_only(package, expected_step=expected, torsion=torsion)
        collate = dual_glt_collate
        dataset_class = CleanDualDataset
    else:
        model = build_mcl_arm(MCL_FUSION[args.arm], package, expected_step=expected)
        collate = mcl_ph_downstream_collate
        dataset_class = MCLPHCleanDataset
        copied = None
    model.to(device)
    optimizer, group_evidence = parameter_groups(model, args.arm)

    train_indices, validation_indices, split_evidence = resolve_fold(
        manifest, args.task, args.fold,
        cohort_rows=int(manifest.get('sample_count', -1)))
    selected_indices, train_local_indices, validation_local_indices = \
        compact_train_validation_indices(train_indices, validation_indices)
    split_path = Path(args.split_root) / f'{args.task}.json'
    source, frame = open_source(
        args.cohort_root, args.cache_root, task=args.task,
        dual_static_root=args.dual_static_root, selected_indices=selected_indices,
        expected_task_rows=int(manifest['sample_count']),
        expected_split_sha256=sha256_file(split_path))
    try:
        if frame['original_row'].astype(int).tolist() != selected_indices:
            raise ValueError('selected downstream rows differ from train/validation indices')
        dataset = (dataset_class(source, frame['label'].to_numpy(dtype=np.float64), statistics)
                   if args.arm in MCL_FUSION else
                   dataset_class(source, frame['label'].to_numpy(dtype=np.float64)))
        set_global_seed(int(config['seed']) + int(args.fold))
        scaler = scale_targets(dataset, args.task, train_indices=train_local_indices,
                               transform_mode='standard')
        train_loader = DataLoader(
            Subset(dataset, train_local_indices), batch_size=int(config['finetune_batch']),
            shuffle=True, num_workers=0, collate_fn=collate,
            generator=torch.Generator().manual_seed(int(config['seed']) + int(args.fold)))
        validation_loader = DataLoader(
            Subset(dataset, validation_local_indices), batch_size=int(config['eval_batch']),
            shuffle=False, num_workers=0, collate_fn=collate)
        schedule_total = SCHEDULE_TOTAL_EPOCHS * len(train_loader)
        schedule_warmup = SCHEDULE_WARMUP_EPOCHS * len(train_loader)
        scheduler = _cosine_scheduler(optimizer, schedule_total, schedule_warmup)
        criterion = nn.MSELoss()
        result = run_selection_epochs(
            model, train_loader, validation_loader, criterion, optimizer, scheduler, device,
            epochs=int(args.epochs), patience=int(config['patience']), scaler=scaler,
            progress=lambda record: print(json.dumps({'arm': args.arm, **record}), flush=True))
        best = result['state']
        model.load_state_dict(best, strict=True)
        reload_check = model.load_state_dict(best, strict=True)
        executed_epochs = len(result['history'])
        validation_keys = [source.samples[index][0].hex()
                           for index in validation_local_indices]
        targets, predictions = result['predictions']
        np.savez(folder / 'validation_predictions.npz',
                 sample_keys=np.asarray(validation_keys),
                 validation_indices=np.asarray(validation_indices, dtype=np.int64),
                 y_true=targets, y_pred=predictions,
                 best_epoch=np.asarray(int(result['best_epoch']), dtype=np.int64),
                 split_protocol=np.asarray(SPLIT_PROTOCOL),
                 outer_test=np.asarray('NOT_RUN'))
        common = dict(
            arm=args.arm, task=args.task, fold=int(args.fold), stage=args.stage,
            protocol=f'mcl_ph_{args.stage}', outer_test='NOT_RUN',
            pretrained_route=('dual_glt' if args.arm in ('glt_ref', 'o8_only') else 'mcl_ph'),
            requested_epochs=int(args.epochs), executed_epochs=int(executed_epochs),
            optimizer_updates=int(result['optimizer_updates']),
            schedule_total_epochs=SCHEDULE_TOTAL_EPOCHS,
            schedule_warmup_epochs=SCHEDULE_WARMUP_EPOCHS,
            best_validation_r2=float(result['best_r2']), best_epoch=int(result['best_epoch']),
            stalled_epochs=int(result['stalled_epochs']), history=result['history'],
            train_sample_count=len(train_indices),
            validation_sample_count=len(validation_indices), scaler_fit_split='train',
            split=split_evidence, learning_rate_table={'backbone': BACKBONE_LR, 'head': HEAD_LR},
            weight_decay=WEIGHT_DECAY, adaptation='full',
            pretrain_step=expected, pretrain_package_sha256=sha256_file(args.checkpoint),
            optimizer_groups=group_evidence,
            copied_o8_tensors=(len(copied) if copied else None),
            load_state_dict_result={'missing_keys': list(reload_check.missing_keys),
                                    'unexpected_keys': list(reload_check.unexpected_keys)},
            wall_seconds=float(time.perf_counter() - started))
        write_json(folder / 'metrics.json', common)
        save_checkpoint(folder / 'best.pt', dict(
            state_dict=best, arm=args.arm, task=args.task, fold=int(args.fold),
            architecture=model.__class__.__name__,
            protocol=f'mcl_ph_{args.stage}', config=config, stage=args.stage,
            best_validation_r2=float(result['best_r2']), best_epoch=int(result['best_epoch']),
            requested_epochs=int(args.epochs), executed_epochs=int(executed_epochs),
            optimizer_updates=int(result['optimizer_updates']),
            scaler_mean=scaler.scaler.mean_.tolist(), scaler_scale=scaler.scaler.scale_.tolist(),
            optimizer_groups=group_evidence, split=split_evidence))
    finally:
        source.close()
    summary = build_summary(common, config=config, device=device, command=sys.argv)
    write_json(folder / 'run.json', dict(summary, history=common['history']))
    write_json(folder / 'runtime.json', dict(
        status='PASS', arm=args.arm, task=args.task, fold=int(args.fold), stage=args.stage,
        pid=os.getpid(), command=sys.argv, exit_code=0,
        process_wall_seconds=float(time.perf_counter() - started)))
    return summary


def failure_record(error, args, started):
    """The non-PASS record of a failed unit, carrying the real exit code."""
    if isinstance(error, SystemExit):
        code = error.code
        code = 0 if code is None else (code if isinstance(code, int) else 1)
    else:
        code = 1
    return dict(status='FAILED', arm=args.arm, task=args.task, fold=int(args.fold),
                stage=args.stage, requested_epochs=int(args.epochs), pid=os.getpid(),
                command=sys.argv, error=f'{type(error).__name__}: {error}', exit_code=code,
                process_wall_seconds=float(time.perf_counter() - started))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', required=True, choices=ARMS)
    parser.add_argument('--stage', required=True, choices=('smoke', 'development'))
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--expected-pretrain-step', type=int, required=True)
    parser.add_argument('--cohort-root', required=True)
    parser.add_argument('--cache-root', required=True)
    parser.add_argument('--dual-static-root', required=True)
    parser.add_argument('--split-root', required=True)
    parser.add_argument('--statistics')
    parser.add_argument('--task', default='xc', choices=TASKS)
    parser.add_argument('--fold', type=int, default=0, choices=FOLDS)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.stage == 'smoke' and not 1 <= args.epochs <= 1:
        raise ValueError('smoke units run exactly one epoch')
    if args.stage == 'development' and not 1 <= args.epochs <= SCHEDULE_TOTAL_EPOCHS:
        raise ValueError(f'development units allow 1..{SCHEDULE_TOTAL_EPOCHS} epochs')
    if args.arm in MCL_FUSION and not args.statistics:
        raise ValueError('the MCL-PH downstream arms require the shared statistics artifact')
    require_tmux()
    started = time.perf_counter()
    config = json.loads(Path(args.config).read_text(encoding='utf-8'))
    manifest = json.loads((Path(args.split_root) / f'{args.task}.json').read_text(encoding='utf-8'))
    if str(manifest.get('protocol')) != SPLIT_PROTOCOL:
        raise ValueError('the fixed outer5_inner20 split is required')
    statistics = (load_geometric_statistics(args.statistics)
                  if args.arm in MCL_FUSION else None)
    folder = unit_directory(args.output, args.arm, args.task, args.fold)
    folder.mkdir(parents=True, exist_ok=False)
    try:
        summary = run_unit(args, folder, started, statistics, config, manifest)
    except BaseException as error:
        write_json(folder / 'runtime.json', failure_record(error, args, started))
        raise
    print(json.dumps({'status': 'PASS', 'arm': args.arm, 'output': str(folder),
                      'executed_epochs': summary['executed_epochs'],
                      'optimizer_updates': summary['optimizer_updates']}))


if __name__ == '__main__':
    main()
