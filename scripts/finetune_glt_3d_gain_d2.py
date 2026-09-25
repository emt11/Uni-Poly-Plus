#!/usr/bin/env python3
"""D2 minimal fine-tuning arms for GLT-3D-GAIN-20260921-01 (r4).

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

Random streams (r4).  All four arms must draw their O8 and head dropout from the
*same* stream positions, and the three dual arms must draw their 3D dropout from
the same stream as each other.  ``f2d`` has no GLT branch, so without isolation
the 3D dropout of the other arms would shift the shared stream from the second
batch onwards.  ``DualArm`` therefore runs the 3D forward inside
``IsolatedRandomStream``: an independent, continuously advancing generator state
whose consumption is invisible to the shared O8/head stream.  Dropout stays on
with its configured probability; only which generator serves it changes.  The
``f2d`` path is untouched, so the shared stream is exactly the one ``f2d`` sees.

This script only ever trains and selects on the requested task/fold's train and
validation rows; it never touches the outer-test split.  Every unit writes into
its own ``<output>/<arm>/<task>/fold<k>/`` directory and refuses to reuse one.
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
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from src.dataset.glt_dual import dual_glt_collate
from src.modules.glt_adaptation import AdaptationAdapter
from src.modules.glt_dual import BondPathO8, DualGLTModel, build_dual_glt_model, mean_pool
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
GLT_STREAM_SEED = COMMON_INIT_SEED  # the private 3D stream's single seed
SCHEDULE_TOTAL_EPOCHS = 30      # the staged development schedule length
SCHEDULE_WARMUP_EPOCHS = 5
STAGE_EPOCH_LIMIT = {'smoke': 1, 'development': SCHEDULE_TOTAL_EPOCHS}
TASKS = ('xc', 'eps', 'eat')
FOLDS = (0, 1)
SPLIT_PROTOCOL = 'outer5_inner20'
SPLIT_NAMES = ('train', 'validation', 'test')


class IsolatedRandomStream:
    """A private, continuously advancing RNG stream for one block of compute.

    ``torch`` dropout draws from the ambient default generators, so a block run
    under this context leaves the caller's stream exactly where it found it: the
    ambient state is saved on entry, the private state is installed, and on exit
    the advanced private state is kept for the next entry while the ambient state
    is restored.  Restoration also happens when the block raises.  The private
    stream is seeded once and never re-seeded per call, so repeated blocks keep
    drawing fresh dropout instead of replaying one batch's masks.
    """

    def __init__(self, seed, device=None):
        self.seed = int(seed)
        self.device = torch.device(device) if device is not None else None
        self.cpu_state = None
        self.cuda_state = None
        self.entries = 0
        self._ambient_cpu = None
        self._ambient_cuda = None

    @property
    def uses_cuda(self):
        return self.device is not None and self.device.type == 'cuda'

    def _ensure_states(self):
        if self.cpu_state is None:
            generator = torch.Generator()
            generator.manual_seed(self.seed)
            self.cpu_state = generator.get_state()
        if self.uses_cuda and self.cuda_state is None:
            generator = torch.Generator(device=self.device)
            generator.manual_seed(self.seed)
            self.cuda_state = generator.get_state()

    def digest(self):
        """Short fingerprint of the private stream's current position."""
        self._ensure_states()
        parts = [bytes(self.cpu_state.numpy())]
        if self.cuda_state is not None:
            parts.append(bytes(self.cuda_state.numpy()))
        return hashlib.sha256(b''.join(parts)).hexdigest()[:16]

    def __enter__(self):
        self._ensure_states()
        self._ambient_cpu = torch.get_rng_state()
        self._ambient_cuda = torch.cuda.get_rng_state(self.device) if self.uses_cuda else None
        torch.set_rng_state(self.cpu_state)
        if self.cuda_state is not None:
            torch.cuda.set_rng_state(self.cuda_state, self.device)
        self.entries += 1
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.cpu_state = torch.get_rng_state()
        if self.uses_cuda:
            self.cuda_state = torch.cuda.get_rng_state(self.device)
        torch.set_rng_state(self._ambient_cpu)
        if self._ambient_cuda is not None:
            torch.cuda.set_rng_state(self._ambient_cuda, self.device)
        return False


def ambient_rng_digest(device=None):
    """Fingerprint of the shared (O8/head) stream position, without advancing it."""
    parts = [bytes(torch.get_rng_state().numpy())]
    if device is not None and torch.device(device).type == 'cuda':
        parts.append(bytes(torch.cuda.get_rng_state(torch.device(device)).numpy()))
    return hashlib.sha256(b''.join(parts)).hexdigest()[:16]


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


class DualArm(DualGLTModel):
    """FBASE/FNORM/FSTABLE: the dual route with the 3D branch on its own stream.

    ``encode`` mirrors ``DualGLTModel.encode`` exactly; the only difference is
    that the 3D forward runs inside ``self.glt_stream``.  How many numbers the
    GLT dropout consumes therefore cannot shift the O8/head stream that all four
    arms share, and the three dual arms walk the same private 3D stream.
    """

    def __init__(self, dropout=0.1, *, torsion=False, device=None,
                 stream_seed=GLT_STREAM_SEED):
        super().__init__('concat', dropout=dropout, torsion=torsion)
        self.glt_stream = IsolatedRandomStream(stream_seed, device=device)

    def encode(self, data, *, atom_mask=None):
        atoms, bias = self.o8(data, atom_mask)
        with self.glt_stream:
            result = self.glt(data)
        result.update(atom_states=atoms, bond_path_attention_bias=bias,
                      canonical_graph_index=data.canonical_graph_index,
                      graph_2d=mean_pool(atoms, data.canonical_graph_index,
                                         data.graph_available.numel()))
        return result


def build_reference(package, *, dropout=0.1, torsion=False, seed=COMMON_INIT_SEED):
    """One common initialisation: deployment tensors plus a fixed head draw."""
    set_global_seed(seed)
    model = build_dual_glt_model('concat', dropout=dropout, torsion=torsion)
    load_deployment(model, package, 5000)
    return model


def build_arm(arm, reference, *, dropout=0.1, torsion=False, device=None):
    """Construct one arm and copy every identically-named tensor from the
    reference, so the shared initial state is bit-identical across arms."""
    if arm == 'f2d':
        model = O8OnlyArm(dropout)
    elif arm in ('fbase', 'fnorm', 'fstable'):
        model = DualArm(dropout, torsion=torsion, device=device)
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


def unit_directory(output, arm, task, fold):
    """The arm/task/fold unit folder; no two units ever share a directory."""
    if arm not in ARMS:
        raise ValueError(f'unknown arm: {arm}')
    if task not in TASKS:
        raise ValueError(f'unknown task: {task}')
    if int(fold) not in FOLDS:
        raise ValueError(f'unknown fold: {fold}')
    return Path(output) / str(arm) / str(task) / f'fold{int(fold)}'


def prepare_unit_directory(path):
    """Create the unit folder, refusing to reuse or overwrite an existing one."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    return path


def split_indices(entry, name, sample_count):
    """One fold's index array: integral, in range and free of duplicates."""
    values = entry.get(f'{name}_indices')
    if not isinstance(values, (list, tuple)):
        raise ValueError(f'the split manifest has no usable {name}_indices array')
    indices = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f'{name}_indices holds a non-integer entry: {value!r}')
        if not 0 <= value < sample_count:
            raise ValueError(f'{name}_indices holds an out-of-range index: {value}')
        indices.append(value)
    if len(set(indices)) != len(indices):
        raise ValueError(f'{name}_indices repeats an index')
    return indices


def resolve_fold(manifest, task, fold, *, cohort_rows, expected_protocol=SPLIT_PROTOCOL):
    """Train/validation rows of one fold, after checking the whole split contract.

    The manifest is only read: the fixed ``outer5_inner20`` split is validated
    here, never rebuilt or rewritten.  The three index arrays must be integral,
    in range, duplicate-free, pairwise disjoint and jointly cover the cohort
    exactly.  The outer-test rows appear only in that coverage check; the runner
    never materialises them, so no outer-test label or prediction is ever read.
    """
    if str(manifest.get('protocol')) != expected_protocol:
        raise ValueError(f'split manifest protocol {manifest.get("protocol")!r} '
                         f'!= {expected_protocol!r}')
    if manifest.get('validation_is_test') is not False:
        raise ValueError('split manifest does not declare validation_is_test=false')
    if str(manifest.get('task')) != str(task):
        raise ValueError(f'split manifest task {manifest.get("task")!r} != {task!r}')
    entry = next((item for item in manifest['folds'] if int(item['fold']) == int(fold)), None)
    if entry is None:
        raise ValueError(f'split manifest has no fold {fold}')
    if int(manifest['sample_count']) != int(cohort_rows):
        raise ValueError('downstream cohort task row count differs from the fixed split')
    count = int(manifest['sample_count'])
    splits = {name: split_indices(entry, name, count) for name in SPLIT_NAMES}
    sets = {name: set(values) for name, values in splits.items()}
    for left, right in (('train', 'validation'), ('train', 'test'), ('validation', 'test')):
        shared = sets[left] & sets[right]
        if shared:
            raise ValueError(f'{left} and {right} overlap on {len(shared)} rows')
    union = sets['train'] | sets['validation'] | sets['test']
    if union != set(range(count)):
        raise ValueError(f'the fold does not cover the cohort exactly: '
                         f'{count - len(union)} of {count} rows are missing')
    evidence = {'protocol': manifest.get('protocol'), 'fold': int(fold), 'task': str(task),
                'validation_is_test': manifest.get('validation_is_test'),
                'sample_count': count,
                'train_rows': len(splits['train']), 'validation_rows': len(splits['validation']),
                'test_rows': len(splits['test']), 'sets_disjoint': True,
                'indices': 'integral, in range, duplicate-free',
                'union_equals_full_cohort': True, 'outer_test': 'NOT_RUN'}
    return splits['train'], splits['validation'], evidence


def run_epochs(model, train_loader, validation_loader, criterion, optimizer, scheduler,
               device, *, epochs, patience, scaler, progress=None):
    """Train at most ``epochs`` epochs, selecting on validation only.

    Returns the epochs that actually ran (so ``executed_epochs`` is a fact, not
    the requested budget) together with the validation-selected state and the
    number of optimizer updates performed.
    """
    wrapped = AdaptationAdapter(model)
    best, best_r2, best_epoch, stalled = None, -float('inf'), -1, 0
    history, updates = [], 0
    for epoch in range(int(epochs)):
        train_result = train_epoch(wrapped, train_loader, criterion, optimizer, scheduler,
                                   device, epoch=epoch + 1, amp_dtype='fp32',
                                   max_grad_norm=1.0, fail_nonfinite=True, return_timing=True)
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
            best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_r2, best_epoch, stalled = validation_r2, epoch + 1, 0
        else:
            stalled += 1
        if stalled >= int(patience):
            break
    if best is None:
        raise RuntimeError('no finite validation-selected checkpoint')
    return {'history': history, 'state': best, 'best_r2': float(best_r2),
            'best_epoch': int(best_epoch), 'stalled_epochs': int(stalled),
            'optimizer_updates': int(updates)}


def run_unit(args, folder, started):
    """Everything one unit does; returns the summary written as its run record."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = json.loads(Path(args.config).read_text(encoding='utf-8'))
    package = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    torsion = bool(package.get('torsion_modules', False))

    reference = build_reference(package, torsion=torsion)
    model, init_evidence = build_arm(args.arm, reference, torsion=torsion, device=device)
    model.to(device)
    optimizer, group_evidence = optimizer_for_arm(
        model, args.arm, weight_decay=config['finetune_weight_decay'])

    manifest = json.loads((Path(args.split_root) / f'{args.task}.json').read_text(encoding='utf-8'))
    source, frame = open_source(args.cohort_root, args.cache_root, task=args.task,
                               dual_static_root=args.dual_static_root)
    try:
        train_indices, validation_indices, split_evidence = resolve_fold(
            manifest, args.task, args.fold, cohort_rows=len(frame))
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
        criterion = nn.MSELoss()

        ambient_begin = ambient_rng_digest(device)
        result = run_epochs(model, train_loader, validation_loader, criterion, optimizer,
                            scheduler, device, epochs=args.epochs,
                            patience=config['patience'], scaler=scaler,
                            progress=lambda record: print(json.dumps({'arm': args.arm, **record}),
                                                          flush=True))
        ambient_end = ambient_rng_digest(device)
        best, best_r2, best_epoch = result['state'], result['best_r2'], result['best_epoch']
        model.load_state_dict(best, strict=True)
        reload_check = model.load_state_dict(best, strict=True)

        stream = getattr(model, 'glt_stream', None)
        rng_evidence = {
            'common_stream': ('the ambient O8/head stream; identical draws in all four arms'),
            'glt_stream_seed': None if stream is None else stream.seed,
            'glt_stream_entries': 0 if stream is None else int(stream.entries),
            'glt_stream_digest': None if stream is None else stream.digest(),
            'ambient_digest_begin': ambient_begin, 'ambient_digest_end': ambient_end,
        }
        executed_epochs = len(result['history'])
        write_json(folder / 'metrics.json', dict(
            arm=args.arm, task=args.task, fold=int(args.fold), stage=args.stage,
            protocol=f'glt_3d_gain_d2_{args.stage}', outer_test='NOT_RUN',
            requested_epochs=int(args.epochs), executed_epochs=int(executed_epochs),
            optimizer_updates=int(result['optimizer_updates']),
            schedule_total_epochs=SCHEDULE_TOTAL_EPOCHS,
            schedule_warmup_epochs=SCHEDULE_WARMUP_EPOCHS,
            best_validation_r2=float(best_r2), best_epoch=int(best_epoch),
            stalled_epochs=int(result['stalled_epochs']),
            history=result['history'],
            train_sample_count=len(train_indices),
            validation_sample_count=len(validation_indices),
            scaler_fit_split='train',
            split=split_evidence, rng=rng_evidence,
            load_state_dict_result={'missing_keys': list(reload_check.missing_keys),
                                    'unexpected_keys': list(reload_check.unexpected_keys)},
            wall_seconds=float(time.perf_counter() - started)))
        save_checkpoint(folder / 'best.pt', dict(
            state_dict=best, arm=args.arm, architecture=model.architecture_name,
            fusion_mode='concat', task=args.task, fold=int(args.fold),
            protocol=f'glt_3d_gain_d2_{args.stage}', config=config, stage=args.stage,
            best_validation_r2=float(best_r2), best_epoch=int(best_epoch),
            requested_epochs=int(args.epochs), executed_epochs=int(executed_epochs),
            optimizer_updates=int(result['optimizer_updates']),
            scaler_mean=scaler.scaler.mean_.tolist(),
            scaler_scale=scaler.scaler.scale_.tolist(),
            init_evidence=init_evidence, optimizer_groups=group_evidence,
            rng=rng_evidence, split=split_evidence,
            capacity=effective_capacity(model)))
    finally:
        source.close()

    summary = dict(
        status='PASS', arm=args.arm, task=args.task, fold=int(args.fold), stage=args.stage,
        command=sys.argv, config=config, learning_rate_table=ARM_LRS[args.arm],
        optimizer_groups=group_evidence, init_evidence=init_evidence,
        capacity=effective_capacity(model), device=str(device),
        requested_epochs=int(args.epochs), executed_epochs=int(executed_epochs),
        optimizer_updates=int(result['optimizer_updates']),
        best_validation_r2=float(best_r2), best_epoch=int(best_epoch),
        split=split_evidence, rng=rng_evidence, history=result['history'])
    write_json(folder / 'run.json', summary)
    write_json(folder / 'runtime.json', dict(
        status='PASS', arm=args.arm, task=args.task, fold=int(args.fold), stage=args.stage,
        pid=os.getpid(), command=sys.argv, exit_code=0,
        process_wall_seconds=float(time.perf_counter() - started)))
    return summary


def failure_record(error, args, started):
    """The unit's runtime record when it did not finish; never a PASS."""
    return dict(status='FAILED', arm=args.arm, task=args.task, fold=int(args.fold),
                stage=args.stage, requested_epochs=int(args.epochs), pid=os.getpid(),
                command=sys.argv, error=f'{type(error).__name__}: {error}', exit_code=1,
                process_wall_seconds=float(time.perf_counter() - started))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', required=True, choices=ARMS)
    parser.add_argument('--stage', required=True, choices=tuple(STAGE_EPOCH_LIMIT))
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--cohort-root', required=True)
    parser.add_argument('--cache-root', required=True)
    parser.add_argument('--dual-static-root', required=True)
    parser.add_argument('--split-root', required=True)
    parser.add_argument('--task', default='xc', choices=TASKS)
    parser.add_argument('--fold', type=int, default=0, choices=FOLDS)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    limit = STAGE_EPOCH_LIMIT[args.stage]
    if not 1 <= args.epochs <= limit:
        raise ValueError(f'{args.stage} runs allow 1..{limit} epochs, requested {args.epochs}')
    require_tmux()

    started = time.perf_counter()
    folder = prepare_unit_directory(unit_directory(args.output, args.arm, args.task, args.fold))
    try:
        summary = run_unit(args, folder, started)
    except BaseException as error:  # a failed unit records its failure and exits non-zero
        write_json(folder / 'runtime.json', failure_record(error, args, started))
        raise
    print(json.dumps({'status': 'PASS', 'arm': args.arm, 'output': str(folder),
                      'executed_epochs': summary['executed_epochs'],
                      'optimizer_updates': summary['optimizer_updates']}))


if __name__ == '__main__':
    main()
