#!/usr/bin/env python3
"""MCL-PH paper5_outer eight-task downstream arms.

Five arms that share the paper5_outer protocol and differ only in the encoder family that
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
import hashlib
import shutil

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
                                           sha256_file, write_json, pretrain_source_fields)
from src.utils import evaluate, scale_targets, set_global_seed, train_epoch
from src.training.split_protocol import resolve_fold, split_indices

ARMS = ('glt_ref', 'o8_only', 'm_cat', 'm_gate', 'm_xattn')
DEVELOPMENT_ARMS = ('m_o8_only', 'm_xattn_scale')
MCL_FUSION = {'m_cat': 'cat', 'm_gate': 'gate', 'm_xattn': 'xattn'}
MCL_FUSION['m_xattn_scale'] = 'xattn'
MODEL_VARIANT = {'glt_ref': 'glt_ref', 'o8_only': 'o8_only', **MCL_FUSION,
                 'm_o8_only': 'm_o8_only'}
SCHEDULE_TOTAL_EPOCHS = 30
HEAD_INIT_SEED = 20260921
R5_PLAN = 'MCL-PH-NEXT-20261004-01'
R6_PLAN = 'MCL-PH-NEXT-20261004-02'
OPT_PLAN = 'MCL-PH-OPT-20261006-01'
TASKS = ('eat', 'eea', 'egb', 'egc', 'ei', 'eps', 'nc', 'xc')
PAPER_TASKS = TASKS
FOLDS = (0, 1, 2, 3, 4)
SPLIT_PROTOCOL = 'paper5_outer'
PAPER_SPLIT_PROTOCOL = 'paper5_outer'
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


def head_initial_sha256(model):
    digest = hashlib.sha256()
    for name, tensor in sorted(periodic_tdl_head(model).state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode('utf-8'))
        digest.update(str(tuple(value.shape)).encode('ascii'))
        digest.update(str(value.dtype).encode('ascii'))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def periodic_tdl_trainability(model, *, head_only, ft_candidate=None):
    if ft_candidate not in (None, 'FT_PARTIAL', 'FT_LOW_LR', 'FT_REG'):
        raise ValueError('unknown FT generalization candidate')
    head_ids = {id(parameter) for parameter in periodic_tdl_head(model).parameters()}
    allowed = head_ids
    if not head_only and ft_candidate == 'FT_PARTIAL':
        encoder = model.model
        modules = [*encoder.o8.layers[-2:], *encoder.glt.layers[-2:],
                   encoder.norm2, encoder.norm3]
        allowed = head_ids | {id(p) for module in modules for p in module.parameters()}
    for parameter in model.parameters():
        parameter.requires_grad_((not head_only and ft_candidate != 'FT_PARTIAL')
                                 or id(parameter) in allowed)


def periodic_tdl_optimizer(model, task, *, head_only, backbone_lr=None, ft_candidate=None, profile=None):
    """Paper learning rates and per-task weight decay on all trainable tensors."""
    if profile not in (None,'uniform_u1'):raise ValueError('unknown FT profile')
    if profile and (ft_candidate is not None or backbone_lr not in (None,1e-4)):raise ValueError('U1 profile override forbidden')
    decay = 0.02 if profile else PERIODIC_TDL_WEIGHT_DECAY[task]
    head_ids = {id(parameter) for parameter in periodic_tdl_head(model).parameters()}
    groups = []
    if not head_only:
        backbone = [parameter for parameter in model.parameters()
                    if parameter.requires_grad and id(parameter) not in head_ids]
        lr = 1e-4 if profile else (backbone_lr if backbone_lr is not None else (2e-4 if task == 'xc' else 1e-4))
        if ft_candidate == 'FT_LOW_LR':
            lr *= 0.3
        groups.append({'params': backbone, 'lr': lr,
                       'weight_decay': decay + (0.01 if ft_candidate == 'FT_REG' else 0),
                       'name': 'backbone'})
    head = [parameter for parameter in model.parameters()
            if parameter.requires_grad and id(parameter) in head_ids]
    groups.append({'params': head, 'lr': 3e-4 if head_only else 1e-3,
                   'weight_decay': decay, 'name': 'head'})
    evidence = [{'name': group['name'], 'lr': group['lr'],
                 'weight_decay': group['weight_decay'], 'num_tensors': len(group['params']),
                 'num_parameters': sum(p.numel() for p in group['params'])}
                for group in groups]
    return torch.optim.AdamW(groups), evidence


def configure_periodic_tdl_dropout(model, task, *, profile=None):
    """Use paper task-specific head dropout; disable encoder module dropout."""
    if profile not in (None,'uniform_u1'):raise ValueError('unknown dropout profile')
    head = periodic_tdl_head(model)
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = 0.0
    for module in head.modules():
        if isinstance(module, nn.Dropout):
            module.p = 0.10 if profile else PERIODIC_TDL_HEAD_DROPOUT[task]


def run_periodic_tdl_epochs(model, train_loader, validation_loader, criterion,
                            device, *, task, scaler, progress=None, smoke=False, backbone_lr=None,
                            ft_candidate=None, profile=None, step_observer=None):
    """Ten frozen-head epochs, then 50/60 joint epochs; select minimum raw RMSE."""
    if profile not in (None,'uniform_u1'):raise ValueError('unknown FT profile')
    if profile and (smoke or ft_candidate is not None or backbone_lr not in (None,1e-4)):raise ValueError('U1 profile override forbidden')
    class HeadOnlyAdapter(AdaptationAdapter):
        def forward(self,data):
            if hasattr(data,'cached_features'):
                return periodic_tdl_head(model)(data.cached_features),None
            return super().forward(data)
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
    stages = (('head', 1), ('joint', 1)) if smoke else (
        ('head', 10), ('joint', 60 if profile else PERIODIC_TDL_EPOCHS.get(task, 60)))
    online_train,online_validation=train_loader,validation_loader
    feature_cache=None
    for stage, stage_epochs in stages:
        if stage=='joint' and feature_cache is not None:
            feature_cache.invalidate()
            train_loader,validation_loader=online_train,online_validation
        wrapped = HeadOnlyAdapter(model) if stage == 'head' else AdaptationAdapter(model)
        periodic_tdl_trainability(model, head_only=(stage == 'head'), ft_candidate=ft_candidate)
        context=getattr(model,'feature_cache_context',None)
        if stage=='head' and context is not None:
            from src.modules.glt_adaptation import HeadFeatureCache
            from torch.utils.data import Subset
            model.eval()
            dataset=online_train.dataset.dataset if isinstance(online_train.dataset,Subset) else online_train.dataset
            feature_cache=HeadFeatureCache(model,dataset,context)
            feature_cache.build(device,online_train.batch_size,online_train.collate_fn)
            train_loader=feature_cache.loader(online_train)
            validation_loader=feature_cache.loader(online_validation)
        optimizer, group_evidence[stage] = periodic_tdl_optimizer(
            model, task, head_only=(stage == 'head'), backbone_lr=backbone_lr,
            ft_candidate=ft_candidate, profile=profile)
        if step_observer is not None:
            optimizer.register_step_pre_hook(lambda opt,args,kwargs:step_observer('attempted'))
            optimizer.register_step_post_hook(lambda opt,args,kwargs:step_observer('completed'))
        if ft_candidate:
            named = dict(model.named_parameters())
            for group in group_evidence[stage]:
                members = {id(p) for item in optimizer.param_groups
                           if item['name'] == group['name'] for p in item['params']}
                group['parameter_names'] = [name for name, p in named.items() if id(p) in members]
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
            if profile:
                record.update(ft_profile=profile,optimizer_groups=group_evidence[stage],encoder_dropout=0.0,head_dropout=0.1,grad_clip=5.0,scheduler=dict(name=type(scheduler).__name__,T_max=10,eta_min=3e-5) if stage=='head' else dict(name=type(scheduler).__name__,T_0=10,T_mult=1,eta_min=1e-6))
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

    def __init__(self, dropout=0.1, torsion=False, fusion_mode='concat', readout='internal',
                 fusion_revision=None, use_md200=False, geometry_revision=None,
                 geometry_mode='finite_internal', angle_value_enabled=False):
        super().__init__()
        self.model = build_dual_glt_model(fusion_mode, dropout=dropout, torsion=torsion, readout=readout,
                                         fusion_revision=fusion_revision, use_md200=use_md200, geometry_revision=geometry_revision,
                                         geometry_mode=geometry_mode, angle_value_enabled=angle_value_enabled)

    def forward(self, batch):
        return self.model(batch)


class MCLPHCleanDataset(Dataset):
    """Clean downstream samples: no mask, frozen coordinates as the 3D view."""

    def __init__(self, source, targets, statistics):
        self.source = source
        self.raw_targets = np.asarray(targets, dtype=np.float64)
        self.targets = self.raw_targets.copy()
        self.statistics = statistics
        self._clean_cache = {}
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
        if index not in self._clean_cache:
            topology, trimer, smiles, static, _ = pretrain_source_fields(self.source, index)
            key = bytes(self.source.samples[index][0]).hex()
            self._clean_cache[index] = prepare_mcl_ph_sample(
                topology, trimer, smiles, seed=42, key=key, position=index,
                sigma=0.0, ratio=None, static=static,
                statistics=self.statistics, view='clean')
        # PyG .to() mutates Data, and collate may change supervision tensors.
        # Keep the cached clean CPU record independent of every presentation.
        data, labels = clone_clean_record(self._clean_cache[index])
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
        self._clean_cache = {}

    def set_target_override(self, values):
        values = np.asarray(values).reshape(-1)
        if values.size != len(self.source):
            raise ValueError('target override count differs from the frozen source')
        self.targets = values

    def __len__(self):
        return len(self.source)

    def __getitem__(self, index):
        index = int(index)
        if index not in self._clean_cache:
            topology, trimer, smiles, static, _ = pretrain_source_fields(self.source, index)
            self._clean_cache[index] = self._build(topology, trimer, smiles, static=static)
        data = clone_clean_record(self._clean_cache[index])
        data.y = torch.tensor([float(self.targets[index])], dtype=torch.float32)
        return data


def clone_clean_record(value):
    """Clone mutable clean features; labels are assigned after each lookup."""
    if isinstance(value, (torch.Tensor,)) or hasattr(value, 'clone'):
        return value.clone()
    if isinstance(value, dict):
        return {key: clone_clean_record(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(clone_clean_record(item) for item in value)
    if isinstance(value, list):
        return [clone_clean_record(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.copy()
    return value


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


def build_mcl_o8_only(package, *, dropout=0.1, expected_step):
    """Extract all O8 tensors from the same strictly loaded MCL-XATTN package."""
    reference = build_mcl_arm('xattn', package, dropout=dropout,
                              expected_step=expected_step)
    arm = O8OnlyArm(dropout)
    arm.o8.load_state_dict(reference.encoder.o8.state_dict(), strict=True)
    for name, value in arm.o8.state_dict().items():
        if not torch.equal(value, reference.encoder.o8.state_dict()[name]):
            raise ValueError('MCL O8 extraction differs: ' + name)
    return arm, ['o8.' + name for name in arm.o8.state_dict()]


def unit_directory(output, arm, task, fold):
    if arm not in (*ARMS, *DEVELOPMENT_ARMS):
        raise ValueError(f'unknown arm: {arm}')
    if task not in TASKS:
        raise ValueError(f'unknown task: {task}')
    if int(fold) not in FOLDS:
        raise ValueError(f'unknown fold: {fold}')
    return Path(output) / str(arm) / str(task) / f'fold{int(fold)}'


def training_contract(args):
    """Bind the explicit r6 recipe by values, including actual data/config identity."""
    if getattr(args, 'experiment_plan', None) != R6_PLAN:
        if getattr(args, 'candidate_config', None):
            raise ValueError('candidate config requires r6 plan')
        return None
    recipe = json.loads(Path(args.candidate_config).read_text()) if getattr(args, 'candidate_config', None) else None
    expected = dict(plan=R6_PLAN, revision='r1', task='xc', arm='glt_ref',
        target_transform='standard', backbone_joint_lr=1e-4, head_initial_lr=3e-4,
        head_joint_lr=1e-3, weight_decay=1e-4, head_dropout=.3, encoder_dropout=0.,
        precision='fp32', batch_size=24, head_epochs=10, joint_epochs=60,
        head_scheduler=dict(T_max=10, eta_min=3e-5),
        joint_scheduler=dict(T_0=10, T_mult=1, eta_min=1e-6))
    if recipe != expected:
        raise ValueError('r6 explicit recipe differs')
    return dict(recipe=recipe, seed=42+args.fold, fold=args.fold,
        config=json.loads(Path(args.config).read_text()),
        split=json.loads((Path(args.split_root)/'xc.json').read_text()),
        source_root=str(Path(args.cohort_root).resolve()),
        source_metadata=json.loads((Path(args.cohort_root)/'direct_source.json').read_text()),
        checkpoint=str(Path(args.checkpoint).resolve()), pretrain_step=5000)


def resolve_target_transform(args):
    mode = getattr(args, 'target_transform', None)
    default = 'identity' if args.task == 'xc' else 'standard'
    if mode not in (None, 'identity', 'standard'):
        raise ValueError('unsupported explicit target transform')
    if mode is not None and mode != default and getattr(args, 'experiment_plan', None) not in (R5_PLAN, R6_PLAN):
        raise ValueError('target transform change requires the r5 experiment contract')
    return mode or default


def validate_stage_scope(stage, task, fold, epochs, cohort_index, strategy='periodic_tdl',
                         *, experiment_plan=None, arm=None, target_transform=None):
    if stage not in ('paper5_outer', 'development', 'smoke') or strategy != 'periodic_tdl':
        raise ValueError('unsupported Periodic-TDL fine-tuning stage')
    if task not in TASKS or fold not in FOLDS:
        raise ValueError('paper5_outer is defined for all eight active tasks')
    r6 = experiment_plan == R6_PLAN
    optimization = experiment_plan == OPT_PLAN
    if optimization:
        if task not in ('eea', 'eps') or arm != 'm_xattn' or target_transform != 'standard' or stage != 'development':
            raise ValueError('unit outside MCL optimization development scope')
    if r6 and (task != "xc" or arm != "glt_ref" or target_transform != "standard"):
        raise ValueError("unit outside r6 Xc contract")
    r5 = experiment_plan in (R5_PLAN, R6_PLAN)
    if experiment_plan is not None and not (r5 or optimization):
        raise ValueError('unknown experiment plan')
    if r5:
        if target_transform != 'standard' or (arm, task) not in (('glt_ref', 'xc'), ('m_xattn', 'nc')):
            raise ValueError('unit differs from the r5 candidate contract')
        allowed = ((task == 'xc' and fold in (0, 1)) or (task == 'nc' and fold in (2, 3, 4)))
        if stage == 'development' and not allowed:
            raise ValueError('unit outside r5 development scope')
    if not (r5 or optimization) and stage == 'development' and (task not in ('ei', 'eps', 'nc') or fold not in (0, 1)):
        raise ValueError('development is restricted to ei/eps/nc folds 0/1')
    expected_epochs = 10 + PERIODIC_TDL_EPOCHS.get(task, 60)
    if stage == 'smoke':
        if (task != ('xc' if r5 else 'ei') or fold != 0):
            raise ValueError('the authorized smoke is restricted to ei/fold0')
        expected_epochs = 2
    if epochs != expected_epochs:
        raise ValueError(f'paper5_outer {task} requires {expected_epochs} epochs')
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
    direct_source = (Path(args.cohort_root) / 'direct_source.json').is_file()
    if not direct_source and not args.cohort_split_root:
        raise ValueError('paper5_outer evaluation requires the frozen cohort split root')
    if not direct_source and not args.cohort_union_manifest:
        raise ValueError('paper5_outer evaluation requires the frozen cohort union manifest')
    initialization_seed = set_global_seed(int(config['seed']) + int(args.fold))
    target_transform = resolve_target_transform(args)
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
    elif args.arm == 'm_o8_only':
        model, copied = build_mcl_o8_only(package, expected_step=expected)
        collate = dual_glt_collate
        dataset_class = CleanDualDataset
    else:
        model = build_mcl_arm(MCL_FUSION[args.arm], package, expected_step=expected)
        if args.arm == 'm_xattn_scale':
            model.encoder.fusion.enable_learnable_scale()
        collate = mcl_ph_downstream_collate
        dataset_class = MCLPHCleanDataset
        copied = None
    initialization_hash = None if direct_source else head_initial_sha256(model)
    if direct_source:
        save_checkpoint(folder / 'head_initial.pt', {
            name: value.detach().cpu().clone()
            for name, value in periodic_tdl_head(model).state_dict().items()})
    model.to(device)
    configure_periodic_tdl_dropout(model, args.task)

    if getattr(args, 'reuse_development_unit', None):
        return promote_development_unit(args, folder, started, statistics,
                                        config, manifest, model, collate, dataset_class)

    train_indices, validation_indices, split_evidence = resolve_fold(
        manifest, args.task, args.fold,
        cohort_rows=int(manifest.get('sample_count', -1)),
        expected_protocol=split_protocol)
    selected_indices, train_local_indices, validation_local_indices = \
        compact_train_validation_indices(train_indices, validation_indices)
    split_path = Path(args.split_root) / f'{args.task}.json'
    cohort_split_path = (Path(args.cohort_split_root) / f'{args.task}.json'
                         if not direct_source else split_path)
    cohort_split_sha256 = None if direct_source else sha256_file(cohort_split_path)
    original_split = json.loads(cohort_split_path.read_text(encoding='utf-8'))
    if not direct_source and (original_split.get('sample_order_sha256') != manifest.get('sample_order_sha256')
            or int(original_split.get('sample_count', -1)) != int(manifest['sample_count'])):
        raise ValueError('paper5_outer folds do not match the frozen cohort row identity')
    source, frame = open_source(
        args.cohort_root, args.cache_root, task=args.task,
        dual_static_root=args.dual_static_root, selected_indices=selected_indices,
        expected_task_rows=int(manifest['sample_count']),
        expected_split_sha256=cohort_split_sha256,
        record_index_path=args.cohort_index,
        union_manifest_path=args.cohort_union_manifest, split_manifest=manifest)
    try:
        if frame['original_row'].astype(int).tolist() != selected_indices:
            raise ValueError('selected downstream rows differ from train/validation indices')
        dataset = (dataset_class(source, frame['label'].to_numpy(dtype=np.float64), statistics)
                   if args.arm in MCL_FUSION else
                   dataset_class(source, frame['label'].to_numpy(dtype=np.float64)))
        set_global_seed(initialization_seed)
        scaler = scale_targets(dataset, args.task, train_indices=train_local_indices,
                               transform_mode=target_transform)
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
            smoke=(args.stage == 'smoke'),
            backbone_lr=(1e-4 if getattr(args, 'experiment_plan', None) == R6_PLAN else None),
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
                record_index_path=args.cohort_index,
                union_manifest_path=args.cohort_union_manifest, split_manifest=manifest)
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
            training_contract=training_contract(args),
            arm=args.arm, task=args.task, fold=int(args.fold), stage=args.stage,
            model_variant=MODEL_VARIANT[args.arm],
            config_path=str(Path(args.config).resolve()),
            config_sha256=None if direct_source else sha256_file(args.config),
            finetune_strategy=args.finetune_strategy,
            selection_metric='validation_rmse',
            protocol=f'mcl_ph_{args.stage}',
            outer_test='RUN' if args.stage == 'paper5_outer' else 'NOT_RUN',
            pretrained_route=('dual_glt' if args.arm in ('glt_ref', 'o8_only') else 'mcl_ph'),
            requested_epochs=int(args.epochs), executed_epochs=int(executed_epochs),
            optimizer_updates=int(result['optimizer_updates']),
            schedule_total_epochs=int(args.epochs), schedule_warmup_epochs=10,
            best_validation_r2=float(result['best_r2']), best_epoch=int(result['best_epoch']),
            best_validation_rmse=float(result['best_rmse']), stalled_epochs=None,
            history=result['history'],
            train_sample_count=len(train_indices),
            validation_sample_count=len(validation_indices),
            scaler_fit_split='none' if target_transform == 'identity' else 'train',
            experiment_plan=getattr(args, 'experiment_plan', None),
            target_transform=scaler.transform_mode,
            initialization_seed=initialization_seed,
            head_initial_sha256=initialization_hash,
            head_initial_state_path=str((folder / 'head_initial.pt').resolve()) if direct_source else None,
            source_identity_contract='direct' if direct_source else 'legacy_digest',
            source_root=str(Path(args.cohort_root).resolve()),
            head_init_seed=(HEAD_INIT_SEED if args.arm != 'glt_ref' else initialization_seed),
            split=dict(split_evidence, outer_test=(
                'RUN' if args.stage == 'paper5_outer' else 'NOT_RUN')),
            evaluation_split_sha256=None if direct_source else sha256_file(split_path),
            evaluation_split_path=str(split_path.resolve()),
            cohort_split_sha256=cohort_split_sha256,
            cohort_union_manifest_path=(str(Path(args.cohort_union_manifest).resolve())
                                        if args.cohort_union_manifest else None),
            split_generator=manifest.get('splitter'),
            **test_result,
            learning_rate_table={
                'head_stage1': next(group['lr'] for group in result['optimizer_groups_by_stage']['head']
                                    if group['name'] == 'head'),
                'backbone_stage2': next(group['lr'] for group in group_evidence
                                        if group['name'] == 'backbone'),
                'head_stage2': next(group['lr'] for group in group_evidence
                                    if group['name'] == 'head')},
            weight_decay=PERIODIC_TDL_WEIGHT_DECAY[args.task],
            adaptation='head10_then_joint',
            pretrain_step=expected, pretrain_package_sha256=(
                None if direct_source else sha256_file(args.checkpoint)),
            pretrain_package_path=str(Path(args.checkpoint).resolve()),
            optimizer_groups=group_evidence,
            optimizer_groups_by_stage=result['optimizer_groups_by_stage'],
            copied_o8_tensors=(len(copied) if copied else None),
            load_state_dict_result={'missing_keys': list(reload_check.missing_keys),
                                    'unexpected_keys': list(reload_check.unexpected_keys)},
            wall_seconds=float(time.perf_counter() - started))
        write_json(folder / 'metrics.json', common)
        save_checkpoint(folder / 'best.pt', dict(
            training_contract=common['training_contract'],
            state_dict=best, arm=args.arm, task=args.task, fold=int(args.fold),
            model_variant=MODEL_VARIANT[args.arm],
            config_path=str(Path(args.config).resolve()),
            config_sha256=common['config_sha256'],
            architecture=model.__class__.__name__,
            protocol=f'mcl_ph_{args.stage}', config=config, stage=args.stage,
            best_validation_r2=float(result['best_r2']),
            best_validation_rmse=float(result['best_rmse']),
            finetune_strategy=args.finetune_strategy,
            best_epoch=int(result['best_epoch']),
            requested_epochs=int(args.epochs), executed_epochs=int(executed_epochs),
            optimizer_updates=int(result['optimizer_updates']),
            scaler_mean=scaler.scaler.mean_.tolist(), scaler_scale=scaler.scaler.scale_.tolist(),
            target_transform=scaler.transform_mode,
            initialization_seed=initialization_seed,
            head_initial_sha256=initialization_hash,
            optimizer_groups=group_evidence, split=common['split'],
            experiment_plan=getattr(args, 'experiment_plan', None),
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


def promote_development_unit(args, folder, started, statistics, config, manifest,
                             model, collate, dataset_class):
    """Evaluate a frozen development checkpoint once, without retraining it."""
    from src.training.mcl_ph_direct_contract import assert_actual_equal
    old = Path(args.reuse_development_unit).resolve()
    metrics = json.loads((old / 'metrics.json').read_text())
    runtime = json.loads((old / 'runtime.json').read_text())
    state = torch.load(old / 'best.pt', map_location='cpu', weights_only=False)
    target_transform = resolve_target_transform(args)
    if metrics.get('target_transform') != target_transform or state.get('target_transform') != target_transform:
        raise ValueError('promotion target transform differs')
    if requested_contract := training_contract(args):
        if metrics.get('training_contract') != requested_contract or state.get('training_contract') != requested_contract:
            raise ValueError('promotion training contract differs')
    previous_plan = metrics.get('experiment_plan')
    requested_plan = getattr(args, 'experiment_plan', None)
    if previous_plan != requested_plan and not (
            requested_plan == R5_PLAN and previous_plan is None and args.arm == 'm_xattn'
            and args.task == 'nc' and args.fold in (0, 1)):
        raise ValueError('promotion experiment identity differs')
    if args.stage != 'paper5_outer' or not getattr(args, 'frozen_selection', None):
        raise ValueError('promotion requires a frozen paper5_outer selection')
    for key, expected in dict(arm=args.arm, task=args.task, fold=args.fold,
                              stage='development', outer_test='NOT_RUN',
                              executed_epochs=args.epochs, pretrain_step=5000,
                              source_identity_contract='direct').items():
        if metrics.get(key) != expected:
            raise ValueError('development checkpoint contract differs: ' + key)
    if runtime.get('status') != 'PASS' or runtime.get('exit_code') != 0:
        raise ValueError('cannot promote a failed development unit')
    if (old / 'test_predictions.npz').exists():
        raise ValueError('development checkpoint already has test predictions')
    if Path(metrics['pretrain_package_path']).resolve() != Path(args.checkpoint).resolve():
        raise ValueError('promotion pretraining package differs')
    if Path(metrics['source_root']).resolve() != Path(args.cohort_root).resolve():
        raise ValueError('promotion source root differs')
    assert_actual_equal(state['config'], config, 'promotion.config')
    assert_actual_equal(torch.load(old / 'head_initial.pt', weights_only=False),
                        torch.load(folder / 'head_initial.pt', weights_only=False),
                        'promotion.head_initial')
    for name, tensor in state['state_dict'].items():
        if not torch.isfinite(tensor).all():
            raise ValueError('non-finite development tensor: ' + name)
    train, validation, _ = resolve_fold(manifest, args.task, args.fold,
        cohort_rows=int(manifest['sample_count']), expected_protocol=PAPER_SPLIT_PROTOCOL)
    selected, train_local, val_local = compact_train_validation_indices(train, validation)
    source, frame = open_source(args.cohort_root, args.cache_root, task=args.task,
        dual_static_root=args.dual_static_root, selected_indices=selected,
        expected_task_rows=int(manifest['sample_count']), record_index_path=args.cohort_index,
        split_manifest=manifest)
    try:
        raw = frame['label'].to_numpy(dtype=np.float64)
        dataset = (dataset_class(source, raw, statistics) if args.arm in MCL_FUSION
                   else dataset_class(source, raw))
        scaler = scale_targets(dataset, args.task, train_indices=train_local,
                              transform_mode=target_transform)
        assert_actual_equal(np.asarray(state['scaler_mean']), scaler.scaler.mean_, 'scaler.mean')
        assert_actual_equal(np.asarray(state['scaler_scale']), scaler.scaler.scale_, 'scaler.scale')
        with np.load(old / 'validation_predictions.npz', allow_pickle=False) as prediction:
            if (prediction['validation_indices'].tolist() != validation
                    or prediction['sample_keys'].tolist() !=
                    [source.samples[i][0].hex() for i in val_local]):
                raise ValueError('promotion validation identities differ')
            truth = scaler.inverse_transform(scaler.transform(raw[val_local]).astype(np.float32))
            np.testing.assert_allclose(prediction['y_true'].reshape(-1), truth.reshape(-1),
                                       rtol=0, atol=1e-6)
            rmse = float(np.sqrt(np.mean((prediction['y_true'].reshape(-1) -
                                         prediction['y_pred'].reshape(-1)) ** 2)))
            if not np.isfinite(rmse) or not np.isclose(rmse, metrics['best_validation_rmse'],
                                                      rtol=1e-6, atol=1e-7):
                raise ValueError('promotion validation RMSE differs')
    finally:
        source.close()
    model.load_state_dict(state['state_dict'], strict=True)
    device = next(model.parameters()).device
    fold = next(item for item in manifest['folds'] if item['fold'] == args.fold)
    indices = sorted(split_indices(fold, 'test', int(manifest['sample_count'])))
    source, frame = open_source(args.cohort_root, args.cache_root, task=args.task,
        dual_static_root=args.dual_static_root, selected_indices=indices,
        expected_task_rows=int(manifest['sample_count']), record_index_path=args.cohort_index,
        split_manifest=manifest)
    try:
        if frame['original_row'].astype(int).tolist() != indices:
            raise ValueError('promotion outer-test source rows differ')
        raw = frame['label'].to_numpy(dtype=np.float64)
        dataset = (dataset_class(source, raw, statistics) if args.arm in MCL_FUSION
                   else dataset_class(source, raw))
        dataset.set_target_override(scaler.transform(raw).reshape(-1))
        loader = DataLoader(dataset, batch_size=24, shuffle=False, num_workers=0, collate_fn=collate)
        _, r2, truth, prediction = evaluate(AdaptationAdapter(model), loader, nn.MSELoss(),
                                          device, scaler=scaler)
        truth, prediction = np.asarray(truth).reshape(-1), np.asarray(prediction).reshape(-1)
        if not np.isfinite(truth).all() or not np.isfinite(prediction).all() or not np.isfinite(r2):
            raise ValueError('promotion outer-test predictions are non-finite')
        np.savez(folder / 'test_predictions.npz', sample_keys=np.asarray(
            [sample[0].hex() for sample in source.samples]),
            test_indices=np.asarray(indices, dtype=np.int64), y_true=truth, y_pred=prediction,
            best_epoch=np.asarray(state['best_epoch']), split_protocol=np.asarray(PAPER_SPLIT_PROTOCOL),
            outer_test=np.asarray('RUN'))
    finally:
        source.close()
    shutil.copyfile(old / 'validation_predictions.npz', folder / 'validation_predictions.npz')
    common = dict(metrics, stage='paper5_outer', protocol='mcl_ph_paper5_outer', outer_test='RUN',
        experiment_plan=requested_plan, reused_experiment_plan=previous_plan,
        split=dict(metrics['split'], outer_test='RUN'), reused_development_unit=str(old),
        promotion_optimizer_updates=0, test_r2=float(r2),
        test_rmse=float(np.sqrt(np.mean((prediction-truth)**2))),
        test_mae=float(np.mean(np.abs(prediction-truth))), test_sample_count=len(indices),
        head_initial_state_path=str((folder / 'head_initial.pt').resolve()),
        wall_seconds=float(time.perf_counter()-started))
    write_json(folder / 'metrics.json', common)
    save_checkpoint(folder / 'best.pt', dict(state, stage='paper5_outer',
        experiment_plan=requested_plan, reused_experiment_plan=previous_plan,
        protocol='mcl_ph_paper5_outer', outer_test='RUN', split=common['split'],
        reused_development_unit=str(old), promotion_optimizer_updates=0))
    summary = build_summary(common, config=config, device=device, command=sys.argv)
    write_json(folder / 'run.json', dict(summary, history=common['history']))
    write_json(folder / 'runtime.json', dict(status='PASS', exit_code=0, pid=os.getpid(),
        command=sys.argv, arm=args.arm, task=args.task, fold=args.fold, stage='paper5_outer',
        reused_development_unit=str(old), promotion_optimizer_updates=0,
        process_wall_seconds=float(time.perf_counter()-started)))
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
    parser.add_argument('--arm', required=True, choices=(*ARMS, *DEVELOPMENT_ARMS))
    parser.add_argument('--stage', required=True,
                        choices=('paper5_outer', 'development', 'smoke'))
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--expected-pretrain-step', type=int, required=True)
    parser.add_argument('--cohort-root', required=True)
    parser.add_argument('--cache-root', required=True)
    parser.add_argument('--dual-static-root', required=True)
    parser.add_argument('--split-root', required=True)
    parser.add_argument('--cohort-split-root', help='frozen cohort split binding for paper5_outer')
    parser.add_argument('--cohort-union-manifest',
                        help='trusted frozen cohort union manifest, stored outside active artifacts')
    parser.add_argument('--cohort-index', help='trusted byte-offset index; required for label-isolated stages')
    parser.add_argument('--statistics')
    parser.add_argument('--task', required=True, choices=TASKS)
    parser.add_argument('--fold', type=int, default=0, choices=FOLDS)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--finetune-strategy', choices=('periodic_tdl',),
                        default='periodic_tdl')
    parser.add_argument('--output', required=True)
    parser.add_argument('--reuse-development-unit',
                        help='validation-selected development checkpoint; zero training updates')
    parser.add_argument('--frozen-selection', help='global route selection frozen before any test')
    parser.add_argument('--experiment-plan', choices=(R5_PLAN, R6_PLAN, OPT_PLAN))
    parser.add_argument('--optimization-scope', help='authorized optimization unit list; no hashes')
    parser.add_argument('--target-transform', choices=('standard', 'identity'))
    parser.add_argument('--candidate-config')
    args = parser.parse_args()
    training_contract(args)
    validate_stage_scope(args.stage, args.task, args.fold, args.epochs,
                         args.cohort_index, args.finetune_strategy,
                         experiment_plan=args.experiment_plan, arm=args.arm,
                         target_transform=args.target_transform)
    resolve_target_transform(args)
    if args.experiment_plan and args.stage == 'paper5_outer' and not args.frozen_selection:
        raise ValueError('r5 test evaluation requires a frozen selection')
    if (args.experiment_plan and args.stage == 'paper5_outer'
            and (args.task == 'nc' or args.fold in (0, 1)) and not args.reuse_development_unit):
        raise ValueError('r5 formal unit must reuse its completed development checkpoint')
    if args.reuse_development_unit and (args.stage != 'paper5_outer' or not args.frozen_selection):
        raise ValueError('development reuse requires paper5_outer and frozen selection')
    if args.frozen_selection:
        selection = json.loads(Path(args.frozen_selection).read_text())
        key = f'{args.arm}/{args.task}/fold{args.fold}'
        if selection.get('status') != 'FROZEN' or key not in selection.get('allowed_units', []):
            raise ValueError('unit is absent from the frozen global selection')
        if args.experiment_plan:
            contract = selection.get('unit_contracts', {}).get(key)
            if not isinstance(contract, dict):
                raise ValueError('r5 unit lacks a frozen experiment contract')
            actual = dict(experiment_plan=args.experiment_plan,
                          target_transform=resolve_target_transform(args),
                          checkpoint=str(Path(args.checkpoint).resolve()),
                          source_root=str(Path(args.cohort_root).resolve()),
                          config=str(Path(args.config).resolve()),
                          split_root=str(Path(args.split_root).resolve()))
            if args.experiment_plan == R6_PLAN:
                actual['training_contract'] = training_contract(args)
            if contract != actual:
                raise ValueError('r5 frozen experiment contract differs')
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
    if args.experiment_plan == OPT_PLAN:
        if not args.optimization_scope:
            raise ValueError('optimization requires the explicit unit scope')
        scope = json.loads(Path(args.optimization_scope).read_text())
        expected = scope.get('units', {}).get(str(folder.resolve()))
        actual = dict(task=args.task, fold=args.fold, checkpoint=str(Path(args.checkpoint).resolve()),
                      config=str(Path(args.config).resolve()), epochs=args.epochs,
                      source=str(Path(args.cohort_root).resolve()), split=str(Path(args.split_root).resolve()))
        if scope.get('plan') != OPT_PLAN or scope.get('status') != 'AUTHORIZED' or expected != actual:
            raise ValueError('optimization unit differs from the authorized scope')
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
