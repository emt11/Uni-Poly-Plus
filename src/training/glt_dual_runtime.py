"""Isolated dual-route I/O and training utilities; importing starts no work."""
import json
import os
import random
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.dataset.glt_dual import FrozenDualLayerSource, build_dual_sample


TASKS = ('eat', 'eea', 'egb', 'egc', 'ei', 'eps', 'nc', 'xc')


class OrderedSampleStream:
    """Recover sampling solely from absolute position; no hidden iterator RNG."""
    def __init__(self, size, seed):
        self.size, self.seed = size, seed
        self.epoch, self.permutation = -1, None

    def index_at(self, position):
        epoch, within = divmod(position, self.size)
        if epoch != self.epoch:
            self.epoch = epoch
            self.permutation = torch.randperm(self.size, generator=torch.Generator().manual_seed(self.seed + epoch))
        return int(self.permutation[within])


def require_tmux():
    if not os.environ.get('TMUX'):
        raise RuntimeError('training must run in logged tmux session Uni-Poly')
    session = subprocess.check_output(['tmux', 'display-message', '-p', '#S'], text=True).strip()
    if session != 'Uni-Poly':
        raise RuntimeError('training requires tmux session Uni-Poly')


def open_source(csv_path, topology_root, trimer_root):
    from src.dataset.lmdb_cache import sample_key_from_smiles
    frame = pd.read_csv(csv_path)
    smiles = frame.iloc[:, 0].astype(str).str.strip().tolist()
    source = FrozenDualLayerSource(topology_root, trimer_root,
                                  [(sample_key_from_smiles(s), s) for s in smiles])
    if not len(source):
        source.close()
        raise ValueError('empty frozen source')
    return source, frame


def save_checkpoint(path, payload):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(payload, temporary)
    os.replace(temporary, path)


def write_json(path, payload):
    with Path(path).open('w', encoding='utf-8') as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if state['cuda'] is not None:
        torch.cuda.set_rng_state_all(state['cuda'])


def scheduled_lr(step, *, lr, warmup_steps, schedule_total_steps, end_lr):
    if step < warmup_steps:
        return lr * (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, schedule_total_steps - warmup_steps)
    return end_lr + (lr - end_lr) * .5 * (1 + np.cos(np.pi * min(1., progress)))


def move_labels(labels, device):
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in labels.items()}


class CleanLabeledDataset(Dataset):
    def __init__(self, source, targets):
        self.source = source
        self.raw_targets = np.asarray(targets, dtype=np.float64)
        self.targets = self.raw_targets.copy()

    def set_target_override(self, targets):
        self.targets = np.asarray(targets).reshape(-1)

    def __len__(self):
        return len(self.source)

    def __getitem__(self, index):
        data = build_dual_sample(*self.source[index])
        data.y = torch.tensor([float(self.targets[index])], dtype=torch.float32)
        return data
