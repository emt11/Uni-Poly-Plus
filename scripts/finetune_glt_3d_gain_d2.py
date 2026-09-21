#!/usr/bin/env python3
"""D2 minimal fine-tuning arms for GLT-3D-GAIN-20260921-01 (r3 smoke).

Four arms, all initialised from the **same** S5 ``B_FP`` step5000 deployment
package with one shared head draw:

| arm | o8 | glt | norm2/norm3 | predictor |
| --- | --- | --- | --- | --- |
| ``f2d``     | 1e-5 | —      | 1e-4 | 1e-4 |
| ``fbase``   | 1e-5 | 1e-5   | 1e-4 | 1e-4 |
| ``fnorm``   | 1e-5 | 1e-5   | **1e-5** | 1e-4 |
| ``fstable`` | 1e-5 | **3e-6** | **1e-5** | 1e-4 |

``f2d`` runs the O8 branch only: its 1024-wide head keeps the base arm's exact
parameter shapes, and the geometry half of its input is structurally zero.  The
GLT branch is neither constructed nor executed, so no coordinate, geometry
validity flag or 3D static field is read on that path.

This script only ever trains and selects on the requested task/fold's train and
validation rows; it never touches the outer-test split.
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
from torch.utils.data import DataLoader, Subset

from src.dataset.glt_dual import dual_glt_collate
from src.modules.glt_adaptation import AdaptationAdapter
from src.modules.glt_dual import BondPathO8, build_dual_glt_model, mean_pool
from src.modules.glt_dual_pretrain import load_deployment
from src.training.glt_dual_runtime import (CleanLabeledDataset, open_source, require_tmux,
                                          save_checkpoint, write_json)
from src.utils import _cosine_scheduler, evaluate, scale_targets, set_global_seed, train_epoch

ARMS = ('f2d', 'fbase', 'fnorm', 'fstable')
ARM_LRS = {
    'f2d':     {'o8': 1e-5, 'glt': None,  'norm': 1e-4, 'head': 1e-4},
    'fbase':   {'o8': 1e-5, 'glt': 1e-5,  'norm': 1e-4, 'head': 1e-4},
    'fnorm':   {'o8': 1e-5, 'glt': 1e-5,  'norm': 1e-5, 'head': 1e-4},
    'fstable': {'o8': 1e-5, 'glt': 3e-6,  'norm': 1e-5, 'head': 1e-4},
}
COMMON_INIT_SEED = 42
SCHEDULE_TOTAL_EPOCHS = 30      # the staged development schedule length
SCHEDULE_WARMUP_EPOCHS = 5


class O8OnlyArm(nn.Module):
    """F2D: the O8 branch alone behind the unchanged 1024-wide head.

    ``norm2`` is applied exactly once, to the 2D graph vector; the second half
    of the head input is a structural zero and carries no geometry signal.  The
    GLT module is absent, so it is neither instantiated nor executed.
    """

    architecture_name = 'O8-BondPath-GalformerTrimer-Hop2'
    fusion_mode = 'concat'

    def __init__(self, dropout=0.1):
        super().__init__()
        self.o8 = BondPathO8(dropout)
        self.norm2 = nn.LayerNorm(512)
        self.predictor = nn.Sequential(nn.Linear(1024, 512), nn.GELU(),
                                       nn.Dropout(dropout), nn.Linear(512, 1))

    def encode(self, data, *, atom_mask=None):
        atoms, _ = self.o8(data, atom_mask=atom_mask)
        return {'graph_2d': mean_pool(atoms, data.canonical_graph_index,
                                      data.graph_available.numel())}

    def fuse(self, encoded):
        z2 = self.norm2(encoded['graph_2d'])
        return torch.cat([z2, z2.new_zeros(z2.shape)], -1)

    def forward(self, data, *, atom_mask=None):
        return self.predictor(self.fuse(self.encode(data, atom_mask=atom_mask)))


def build_reference(package, *, dropout=0.1, torsion=False, seed=COMMON_INIT_SEED):
    """One common initialisation: deployment tensors plus a fixed head draw."""
    set_global_seed(seed)
    model = build_dual_glt_model('concat', dropout=dropout, torsion=torsion)
    load_deployment(model, package, 5000)
    return model


def build_arm(arm, reference, *, dropout=0.1, torsion=False):
    """Construct one arm and copy every identically-named tensor from the
    reference, so the shared initial state is bit-identical across arms."""
    if arm == 'f2d':
        model = O8OnlyArm(dropout)
    elif arm in ('fbase', 'fnorm', 'fstable'):
        model = build_dual_glt_model('concat', dropout=dropout, torsion=torsion)
    else:
        raise ValueError(f'unknown arm: {arm}')
    target_state = model.state_dict()
    source_state = reference.state_dict()
    copied, unsourced = [], []
    for name, value in target_state.items():
        if name not in source_state:
            unsourced.append(name)
            continue
        if tuple(source_state[name].shape) != tuple(value.shape):
            raise ValueError(f'shared tensor shape mismatch: {name}')
        with torch.no_grad():
            value.copy_(source_state[name])
        copied.append(name)
    if unsourced:
        raise ValueError('arm tensors without a reference source: ' + ','.join(unsourced[:5]))
    return model, {'copied': sorted(copied), 'unsourced': unsourced,
                   'reference_only': sorted(set(source_state) - set(target_state))}


def optimizer_for_arm(model, arm, *, weight_decay):
    """Every and only trainable tensor, in exactly one group."""
    lrs = ARM_LRS[arm]
    named = dict(model.named_parameters())
    groups, seen, names_by_group = [], set(), {}

    def add(name, params, lr):
        params = [parameter for parameter in params if parameter.requires_grad]
        if not params:
            return
        identities = {id(parameter) for parameter in params}
        for parameter in params:
            if id(parameter) in seen:
                raise RuntimeError(f'parameter appears in two groups: {name}')
            seen.add(id(parameter))
        groups.append(dict(params=params, lr=float(lr), name=name))
        names_by_group[name] = sorted(n for n, p in named.items() if id(p) in identities)

    add('o8', model.o8.parameters(), lrs['o8'])
    if lrs['glt'] is not None:
        if not hasattr(model, 'glt'):
            raise ValueError(f'arm {arm} declares a GLT lr without a GLT branch')
        add('glt', model.glt.parameters(), lrs['glt'])
    elif hasattr(model, 'glt'):
        raise ValueError(f'arm {arm} must not carry a GLT branch')
    norm_modules = [model.norm2] + ([model.norm3] if hasattr(model, 'norm3') else [])
    add('norm', [p for module in norm_modules for p in module.parameters()], lrs['norm'])
    add('head', model.predictor.parameters(), lrs['head'])

    ungrouped = [name for name, parameter in model.named_parameters()
                 if parameter.requires_grad and id(parameter) not in seen]
    if ungrouped:
        raise RuntimeError('trainable tensors outside every group: ' + ','.join(ungrouped[:5]))
    optimizer = torch.optim.AdamW(groups, weight_decay=float(weight_decay))
    return optimizer, [{'name': group['name'], 'lr': group['lr'],
                        'weight_decay': float(weight_decay),
                        'num_tensors': len(group['params']),
                        'num_parameters': int(sum(p.numel() for p in group['params'])),
                        'parameter_names': names_by_group[group['name']]}
                       for group in groups]


def effective_capacity(model):
    """Report how many head parameters are dead because their input is zero."""
    first = model.predictor[0]
    zero_columns = 512 if getattr(model, 'glt', None) is None else 0
    dead = int(zero_columns * first.weight.shape[0])
    total = int(sum(p.numel() for p in model.parameters()))
    return {'total_parameters': total, 'dead_head_parameters': dead,
            'effective_parameters': total - dead,
            'note': ('the second 512 columns of predictor.0 receive a structural '
                     'zero input and never contribute' if dead else 'no dead columns')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', required=True, choices=ARMS)
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--cohort-root', required=True)
    parser.add_argument('--cache-root', required=True)
    parser.add_argument('--dual-static-root', required=True)
    parser.add_argument('--split-root', required=True)
    parser.add_argument('--task', default='xc')
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    require_tmux()
    if args.epochs < 1:
        raise ValueError('epochs must be at least one')

    started = time.perf_counter()
    config = json.loads(Path(args.config).read_text(encoding='utf-8'))
    package = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    torsion = bool(package.get('torsion_modules', False))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output = Path(args.output)
    folder = output / args.arm / args.task / f'fold{args.fold}'
    folder.mkdir(parents=True, exist_ok=False)

    reference = build_reference(package, torsion=torsion)
    model, init_evidence = build_arm(args.arm, reference, torsion=torsion)
    model.to(device)
    optimizer, group_evidence = optimizer_for_arm(
        model, args.arm, weight_decay=config['finetune_weight_decay'])

    manifest = json.loads((Path(args.split_root) / f'{args.task}.json').read_text(encoding='utf-8'))
    fold_entry = next(item for item in manifest['folds'] if int(item['fold']) == args.fold)
    train_indices = [int(v) for v in fold_entry['train_indices']]
    validation_indices = [int(v) for v in fold_entry['validation_indices']]

    source, frame = open_source(args.cohort_root, args.cache_root, task=args.task,
                               dual_static_root=args.dual_static_root)
    try:
        if len(frame) != int(manifest['sample_count']):
            raise ValueError('downstream cohort task row count differs from the fixed split')
        if frame['original_row'].astype(int).tolist() != list(range(len(frame))):
            raise ValueError('downstream cohort task row order differs from the property CSV')
        dataset = CleanLabeledDataset(source, frame['label'].to_numpy(dtype=np.float64),
                                      torsion_mode=('on' if torsion else None))
        set_global_seed(config['seed'] + args.fold)
        scaler = scale_targets(dataset, args.task, train_indices=train_indices,
                               transform_mode='standard')
        train_loader = DataLoader(Subset(dataset, train_indices),
                                  batch_size=config['finetune_batch'], shuffle=True,
                                  num_workers=0, collate_fn=dual_glt_collate,
                                  generator=torch.Generator().manual_seed(
                                      config['seed'] + args.fold))
        validation_loader = DataLoader(Subset(dataset, validation_indices),
                                       batch_size=config['eval_batch'], shuffle=False,
                                       num_workers=0, collate_fn=dual_glt_collate)
        schedule_total = SCHEDULE_TOTAL_EPOCHS * len(train_loader)
        schedule_warmup = SCHEDULE_WARMUP_EPOCHS * len(train_loader)
        scheduler = _cosine_scheduler(optimizer, schedule_total, schedule_warmup)
        wrapped = AdaptationAdapter(model)
        criterion = nn.MSELoss()

        best, best_r2, best_epoch, stalled = None, -float('inf'), -1, 0
        history = []
        for epoch in range(args.epochs):
            train_result = train_epoch(wrapped, train_loader, criterion, optimizer, scheduler,
                                       device, epoch=epoch + 1, amp_dtype='fp32',
                                       max_grad_norm=1.0, fail_nonfinite=True,
                                       return_timing=True)
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
            print(json.dumps({'arm': args.arm, **record}), flush=True)
            if validation_r2 > best_r2:
                best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                best_r2, best_epoch, stalled = validation_r2, epoch + 1, 0
            else:
                stalled += 1
            if stalled >= config['patience']:
                break
        if best is None:
            raise RuntimeError('no finite validation-selected checkpoint')
        model.load_state_dict(best, strict=True)
        reload_check = model.load_state_dict(best, strict=True)

        save_checkpoint(folder / 'best.pt', dict(
            state_dict=best, arm=args.arm, architecture=model.architecture_name,
            fusion_mode='concat', task=args.task, fold=int(args.fold),
            protocol='glt_3d_gain_d2_smoke', config=config,
            best_validation_r2=float(best_r2), best_epoch=int(best_epoch),
            executed_epochs=int(args.epochs),
            scaler_mean=scaler.scaler.mean_.tolist(),
            scaler_scale=scaler.scaler.scale_.tolist(),
            init_evidence=init_evidence, optimizer_groups=group_evidence,
            capacity=effective_capacity(model)))
        write_json(folder / 'metrics.json', dict(
            arm=args.arm, task=args.task, fold=int(args.fold),
            protocol='glt_3d_gain_d2_smoke', outer_test='NOT_RUN',
            executed_epochs=int(args.epochs),
            schedule_total_epochs=SCHEDULE_TOTAL_EPOCHS,
            schedule_warmup_epochs=SCHEDULE_WARMUP_EPOCHS,
            best_validation_r2=float(best_r2), best_epoch=int(best_epoch),
            history=history,
            train_sample_count=len(train_indices),
            validation_sample_count=len(validation_indices),
            scaler_fit_split='train',
            load_state_dict_result={'missing_keys': list(reload_check.missing_keys),
                                    'unexpected_keys': list(reload_check.unexpected_keys)},
            wall_seconds=float(time.perf_counter() - started)))
    finally:
        source.close()

    write_json(output / f'run_{args.arm}.json', dict(
        arm=args.arm, command=sys.argv, config=config,
        learning_rate_table=ARM_LRS[args.arm],
        optimizer_groups=group_evidence, init_evidence=init_evidence,
        capacity=effective_capacity(model), device=str(device)))
    write_json(output / f'runtime_{args.arm}.json', dict(
        status='PASS', arm=args.arm, pid=os.getpid(), command=sys.argv,
        process_wall_seconds=float(time.perf_counter() - started)))
    print(json.dumps({'status': 'PASS', 'arm': args.arm, 'output': str(folder)}))


if __name__ == '__main__':
    main()
