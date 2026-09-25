#!/usr/bin/env python3
"""MCL-PH official Periodic-TDL five-fold downstream arms.

Five arms that share one protocol and differ only in the encoder family that
was pre-trained:

| arm | pre-trained route | 3D branch | downstream head |
| --- | --- | --- | --- |
| ``glt_ref``  | the pre-existing dual GLT route | GLT    | the original 1024-wide concat head |
| ``o8_only``  | the ``glt_ref`` package's O8       | none   | the new 512-wide head |
| ``m_cat``    | this contract, ``F_CAT``           | MCL-PH | the new 512-wide head |
| ``m_gate``   | this contract, ``F_GATE``          | MCL-PH | the new 512-wide head |
| ``m_xattn``  | this contract, ``F_XATTN``         | MCL-PH | the new 512-wide head |

Training decodes only train and validation records. The paper5_outer stage
decodes test records after the validation-selected checkpoint is restored.
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
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts
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
from src.utils import evaluate, scale_targets, set_global_seed, train_epoch
from scripts.finetune_glt_3d_gain_d2 import resolve_fold, split_indices

ARMS = ('glt_ref', 'o8_only', 'm_cat', 'm_gate', 'm_xattn')
MCL_FUSION = {'m_cat': 'cat', 'm_gate': 'gate', 'm_xattn': 'xattn'}
SCHEDULE_TOTAL_EPOCHS = 30
HEAD_INIT_SEED = 20260921
TASKS = ('eat', 'eea', 'egb', 'egc', 'ei', 'eps', 'nc', 'xc')
PAPER_TASKS = ('eea', 'egb', 'ei', 'eps', 'nc')
FOLDS = (0, 1, 2, 3, 4)
SPLIT_PROTOCOL = 'outer5_inner20'
PAPER_SPLIT_PROTOCOL = 'periodic_tdl_official5'
PERIODIC_TDL_EPOCHS = {'egc': 50}
# EAT is absent from the paper's nine tasks. Its entries below are an explicit
# project adaptation using the Eea settings, not a published Periodic-TDL value.
PERIODIC_TDL_WEIGHT_DECAY = {'egc': 0.0, 'eea': 0.02, 'egb': 0.02,
                             'ei': 0.02, 'eps': 0.02, 'nc': 0.02,
                             'xc': 1e-4, 'eat': 0.02}
PERIODIC_TDL_HEAD_DROPOUT = {'egc': 0.0, 'eea': 0.1, 'egb': 0.1,
                             'ei': 0.1, 'eps': 0.1, 'nc': 0.1,
                             'xc': 0.3, 'eat': 0.1}


def periodic_tdl_head(model):
    return model.head if hasattr(model, 'head') else model.model.predictor


def periodic_tdl_trainability(model, *, head_only):
    head_ids = {id(parameter) for parameter in periodic_tdl_head(model).parameters()}
    for parameter in model.parameters():
        parameter.requires_grad_(not head_only or id(parameter) in head_ids)


def periodic_tdl_optimizer(model, task, *, head_only):
    """Paper learning rates and per-task weight decay on all trainable tensors."""
    decay = PERIODIC_TDL_WEIGHT_DECAY[task]
    head_ids = {id(parameter) for parameter in periodic_tdl_head(model).parameters()}
    groups = []
    if not head_only:
        backbone = [parameter for parameter in model.parameters()
                    if id(parameter) not in head_ids]
        groups.append({'params': backbone, 'lr': 2e-4 if task == 'xc' else 1e-4,
                       'weight_decay': decay, 'name': 'backbone'})
    head = [parameter for parameter in model.parameters() if id(parameter) in head_ids]
    groups.append({'params': head, 'lr': 3e-4 if head_only else 1e-3,
                   'weight_decay': decay, 'name': 'head'})
    evidence = [{'name': group['name'], 'lr': group['lr'],
                 'weight_decay': decay, 'num_tensors': len(group['params']),
                 'num_parameters': sum(p.numel() for p in group['params'])}
                for group in groups]
    return torch.optim.AdamW(groups), evidence


def configure_periodic_tdl_dropout(model, task):
    """Use paper task-specific head dropout; disable encoder module dropout."""
    head = periodic_tdl_head(model)
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = 0.0
    for module in head.modules():
        if isinstance(module, nn.Dropout):
            module.p = PERIODIC_TDL_HEAD_DROPOUT[task]


def run_periodic_tdl_epochs(model, train_loader, validation_loader, criterion,
                            device, *, task, scaler, progress=None):
    """Ten frozen-head epochs, then 50/60 joint epochs; select minimum raw RMSE."""
    class HeadOnlyAdapter(AdaptationAdapter):
        def train(self, mode=True):
            super().train(mode)
            if mode:
                model.eval()
                periodic_tdl_head(model).train()
            return self

    class EpochSchedulerProxy:
        def step(self):
            pass

    best_state, best_epoch, best_rmse, best_r2 = None, -1, float('inf'), None
    best_predictions = None
    history, updates, group_evidence = [], 0, {}
    for stage, stage_epochs in (('head', 10),
                                ('joint', PERIODIC_TDL_EPOCHS.get(task, 60))):
        wrapped = HeadOnlyAdapter(model) if stage == 'head' else AdaptationAdapter(model)
        periodic_tdl_trainability(model, head_only=(stage == 'head'))
        optimizer, group_evidence[stage] = periodic_tdl_optimizer(
            model, task, head_only=(stage == 'head'))
        if stage == 'head':
            scheduler = CosineAnnealingLR(optimizer, T_max=10,
                                          eta_min=3e-5)
        else:
            scheduler = CosineAnnealingWarmRestarts(
                optimizer, T_0=10, T_mult=1, eta_min=1e-6)
        for stage_epoch in range(1, stage_epochs + 1):
            result = train_epoch(wrapped, train_loader, criterion, optimizer,
                                 EpochSchedulerProxy(),
                                 device, epoch=len(history) + 1, amp_dtype='fp32',
                                 max_grad_norm=5.0, fail_nonfinite=True,
                                 return_timing=True)
            scheduler.step()
            updates += int(result[-1]['training_steps'])
            validation_loss, r2, targets, predictions = evaluate(
                wrapped, validation_loader, criterion, device, scaler=scaler,
                amp_dtype='fp32')
            targets = np.asarray(targets, dtype=np.float64).reshape(-1)
            predictions = np.asarray(predictions, dtype=np.float64).reshape(-1)
            rmse = float(np.sqrt(np.mean((targets - predictions) ** 2)))
            if not all(np.isfinite([validation_loss, r2, rmse])):
                raise FloatingPointError('nonfinite validation metrics')
            record = {'epoch': len(history) + 1, 'stage': stage,
                      'stage_epoch': stage_epoch, 'train_loss': float(result[0]),
                      'validation_loss': float(validation_loss),
                      'validation_r2': float(r2), 'validation_rmse': rmse,
                      'learning_rates': [group['lr'] for group in optimizer.param_groups],
                      'training_steps': int(result[-1]['training_steps'])}
            history.append(record)
            if progress is not None:
                progress(record)
            if rmse < best_rmse:
                best_state = {key: value.detach().cpu().clone()
                              for key, value in model.state_dict().items()}
                best_predictions = (targets.copy(), predictions.copy())
                best_rmse, best_r2, best_epoch = rmse, float(r2), len(history)
    if best_state is None:
        raise RuntimeError('no finite validation-selected checkpoint')
    return {'history': history, 'state': best_state, 'best_r2': best_r2,
            'best_rmse': best_rmse, 'best_epoch': best_epoch,
            'optimizer_updates': updates, 'predictions': best_predictions,
            'optimizer_groups_by_stage': group_evidence}


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


def unit_directory(output, arm, task, fold):
    if arm not in ARMS:
        raise ValueError(f'unknown arm: {arm}')
    if task not in TASKS:
        raise ValueError(f'unknown task: {task}')
    if int(fold) not in FOLDS:
        raise ValueError(f'unknown fold: {fold}')
    return Path(output) / str(arm) / str(task) / f'fold{int(fold)}'


def validate_stage_scope(stage, task, fold, epochs, cohort_index, strategy='periodic_tdl'):
    if stage != 'paper5_outer' or strategy != 'periodic_tdl':
        raise ValueError('only official paper5_outer Periodic-TDL fine-tuning is active')
    if task not in PAPER_TASKS or fold not in FOLDS:
        raise ValueError('official Periodic-TDL folds are verified only for Eea/Egb/Ei/EPS/Nc')
    if epochs != 70:
        raise ValueError('official Periodic-TDL tasks require 70 epochs')
    if not cohort_index:
        raise ValueError('a trusted cohort index is required')


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
    split_protocol = PAPER_SPLIT_PROTOCOL
    if not args.cohort_split_root:
        raise ValueError('official-fold evaluation requires the frozen cohort split root')
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
    configure_periodic_tdl_dropout(model, args.task)

    train_indices, validation_indices, split_evidence = resolve_fold(
        manifest, args.task, args.fold,
        cohort_rows=int(manifest.get('sample_count', -1)),
        expected_protocol=split_protocol)
    selected_indices, train_local_indices, validation_local_indices = \
        compact_train_validation_indices(train_indices, validation_indices)
    split_path = Path(args.split_root) / f'{args.task}.json'
    cohort_split_path = Path(args.cohort_split_root) / f'{args.task}.json'
    cohort_split_sha256 = sha256_file(cohort_split_path)
    original_split = json.loads(cohort_split_path.read_text(encoding='utf-8'))
    if (original_split.get('sample_order_sha256') != manifest.get('sample_order_sha256')
            or int(original_split.get('sample_count', -1)) != int(manifest['sample_count'])):
        raise ValueError('official folds do not match the frozen cohort row identity')
    source, frame = open_source(
        args.cohort_root, args.cache_root, task=args.task,
        dual_static_root=args.dual_static_root, selected_indices=selected_indices,
        expected_task_rows=int(manifest['sample_count']),
        expected_split_sha256=cohort_split_sha256,
        record_index_path=args.cohort_index)
    try:
        if frame['original_row'].astype(int).tolist() != selected_indices:
            raise ValueError('selected downstream rows differ from train/validation indices')
        dataset = (dataset_class(source, frame['label'].to_numpy(dtype=np.float64), statistics)
                   if args.arm in MCL_FUSION else
                   dataset_class(source, frame['label'].to_numpy(dtype=np.float64)))
        set_global_seed(int(config['seed']) + int(args.fold))
        scaler = scale_targets(dataset, args.task, train_indices=train_local_indices,
                               transform_mode='standard')
        batch_size = 24
        train_loader = DataLoader(
            Subset(dataset, train_local_indices), batch_size=batch_size,
            shuffle=True, num_workers=0, collate_fn=collate,
            generator=torch.Generator().manual_seed(int(config['seed']) + int(args.fold)))
        validation_loader = DataLoader(
            Subset(dataset, validation_local_indices), batch_size=24,
            shuffle=False, num_workers=0, collate_fn=collate)
        criterion = nn.MSELoss()
        result = run_periodic_tdl_epochs(
            model, train_loader, validation_loader, criterion, device,
            task=args.task, scaler=scaler,
            progress=lambda record: print(json.dumps({'arm': args.arm, **record}), flush=True))
        group_evidence = result['optimizer_groups_by_stage']['joint']
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
            split_protocol=np.asarray(split_protocol),
            finetune_strategy=np.asarray(args.finetune_strategy),
                 outer_test=np.asarray('NOT_RUN'))
        test_result = {}
        if args.stage == 'paper5_outer':
            # The test source is opened only after the validation-selected state
            # has been restored. LMDB allows one open handle per environment in
            # this process, so release the training source before opening test.
            # No subsequent optimizer or selection step occurs.
            source.close()
            source = None
            fold_entry = next(item for item in manifest['folds']
                              if int(item['fold']) == int(args.fold))
            test_indices = sorted(split_indices(fold_entry, 'test',
                                                int(manifest['sample_count'])))
            test_source, test_frame = open_source(
                args.cohort_root, args.cache_root, task=args.task,
                dual_static_root=args.dual_static_root, selected_indices=test_indices,
                expected_task_rows=int(manifest['sample_count']),
                expected_split_sha256=cohort_split_sha256,
                record_index_path=args.cohort_index)
            try:
                if test_frame['original_row'].astype(int).tolist() != test_indices:
                    raise ValueError('selected downstream rows differ from outer-test indices')
                raw_targets = test_frame['label'].to_numpy(dtype=np.float64)
                test_dataset = (dataset_class(test_source, raw_targets, statistics)
                                if args.arm in MCL_FUSION else dataset_class(test_source, raw_targets))
                test_dataset.set_target_override(scaler.transform(raw_targets).reshape(-1))
                test_loader = DataLoader(test_dataset, batch_size=24, shuffle=False,
                                         num_workers=0, collate_fn=collate)
                _, test_r2, test_truth, test_prediction = evaluate(
                    AdaptationAdapter(model), test_loader, criterion, device, scaler=scaler)
                test_truth = np.asarray(test_truth, dtype=np.float64).reshape(-1)
                test_prediction = np.asarray(test_prediction, dtype=np.float64).reshape(-1)
                if not (np.isfinite(test_truth).all() and np.isfinite(test_prediction).all()
                        and np.isfinite(test_r2)):
                    raise ValueError('outer-test predictions or R2 are non-finite')
                test_result = dict(test_r2=float(test_r2),
                                   test_rmse=float(np.sqrt(np.mean((test_prediction-test_truth)**2))),
                                   test_mae=float(np.mean(np.abs(test_prediction-test_truth))),
                                   test_sample_count=len(test_indices))
                np.savez(folder / 'test_predictions.npz',
                         sample_keys=np.asarray([sample[0].hex() for sample in test_source.samples]),
                         test_indices=np.asarray(test_indices, dtype=np.int64),
                         y_true=test_truth, y_pred=test_prediction,
                         best_epoch=np.asarray(int(result['best_epoch']), dtype=np.int64),
                         split_protocol=np.asarray(split_protocol),
                         outer_test=np.asarray('RUN'))
            finally:
                test_source.close()
        common = dict(
            arm=args.arm, task=args.task, fold=int(args.fold), stage=args.stage,
            finetune_strategy=args.finetune_strategy,
            selection_metric='validation_rmse',
            protocol=f'mcl_ph_{args.stage}',
            outer_test='RUN',
            pretrained_route=('dual_glt' if args.arm in ('glt_ref', 'o8_only') else 'mcl_ph'),
            requested_epochs=int(args.epochs), executed_epochs=int(executed_epochs),
            optimizer_updates=int(result['optimizer_updates']),
            schedule_total_epochs=int(args.epochs), schedule_warmup_epochs=10,
            best_validation_r2=float(result['best_r2']), best_epoch=int(result['best_epoch']),
            best_validation_rmse=float(result['best_rmse']), stalled_epochs=None,
            history=result['history'],
            train_sample_count=len(train_indices),
            validation_sample_count=len(validation_indices),
            scaler_fit_split='train',
            split=dict(split_evidence, outer_test='RUN'),
            evaluation_split_sha256=sha256_file(split_path),
            evaluation_split_path=str(split_path.resolve()),
            cohort_split_sha256=cohort_split_sha256,
            official_source=manifest.get('official_source'),
            **test_result,
            learning_rate_table={'head_stage1': 3e-4, 'backbone_stage2': 1e-4,
                                 'head_stage2': 1e-3},
            weight_decay=PERIODIC_TDL_WEIGHT_DECAY[args.task],
            adaptation='head10_then_joint',
            pretrain_step=expected, pretrain_package_sha256=sha256_file(args.checkpoint),
            optimizer_groups=group_evidence,
            optimizer_groups_by_stage=result['optimizer_groups_by_stage'],
            copied_o8_tensors=(len(copied) if copied else None),
            load_state_dict_result={'missing_keys': list(reload_check.missing_keys),
                                    'unexpected_keys': list(reload_check.unexpected_keys)},
            wall_seconds=float(time.perf_counter() - started))
        write_json(folder / 'metrics.json', common)
        save_checkpoint(folder / 'best.pt', dict(
            state_dict=best, arm=args.arm, task=args.task, fold=int(args.fold),
            architecture=model.__class__.__name__,
            protocol=f'mcl_ph_{args.stage}', config=config, stage=args.stage,
            best_validation_r2=float(result['best_r2']),
            best_validation_rmse=float(result['best_rmse']),
            finetune_strategy=args.finetune_strategy,
            best_epoch=int(result['best_epoch']),
            requested_epochs=int(args.epochs), executed_epochs=int(executed_epochs),
            optimizer_updates=int(result['optimizer_updates']),
            scaler_mean=scaler.scaler.mean_.tolist(), scaler_scale=scaler.scaler.scale_.tolist(),
            optimizer_groups=group_evidence, split=common['split'],
            outer_test=common['outer_test']))
    finally:
        if source is not None:
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
    parser.add_argument('--stage', required=True, choices=('paper5_outer',))
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--expected-pretrain-step', type=int, required=True)
    parser.add_argument('--cohort-root', required=True)
    parser.add_argument('--cache-root', required=True)
    parser.add_argument('--dual-static-root', required=True)
    parser.add_argument('--split-root', required=True)
    parser.add_argument('--cohort-split-root', help='frozen cohort split binding for paper5_outer')
    parser.add_argument('--cohort-index', help='trusted byte-offset index; required for label-isolated stages')
    parser.add_argument('--statistics')
    parser.add_argument('--task', required=True, choices=PAPER_TASKS)
    parser.add_argument('--fold', type=int, default=0, choices=FOLDS)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--finetune-strategy', choices=('periodic_tdl',),
                        default='periodic_tdl')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    validate_stage_scope(args.stage, args.task, args.fold, args.epochs,
                         args.cohort_index, args.finetune_strategy)
    if args.arm in MCL_FUSION and not args.statistics:
        raise ValueError('the MCL-PH downstream arms require the shared statistics artifact')
    require_tmux()
    started = time.perf_counter()
    config = json.loads(Path(args.config).read_text(encoding='utf-8'))
    manifest = json.loads((Path(args.split_root) / f'{args.task}.json').read_text(encoding='utf-8'))
    if str(manifest.get('protocol')) != PAPER_SPLIT_PROTOCOL:
        raise ValueError(f'the fixed {PAPER_SPLIT_PROTOCOL} split is required')
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
