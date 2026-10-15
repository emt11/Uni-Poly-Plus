"""Actual row, normalization and initialization values for direct run recovery."""
from pathlib import Path
import numpy as np
import torch


def assert_actual_equal(actual, expected, path='binding'):
    if isinstance(expected, torch.Tensor):
        if not isinstance(actual, torch.Tensor) or actual.dtype != expected.dtype or actual.shape != expected.shape or not torch.equal(actual.cpu(), expected.cpu()):
            raise ValueError(f'{path}: tensor values/dtype/shape differ')
    elif isinstance(expected, np.ndarray):
        if not isinstance(actual, np.ndarray) or actual.dtype != expected.dtype or actual.shape != expected.shape or not np.array_equal(actual, expected):
            raise ValueError(f'{path}: array values/dtype/shape differ')
    elif isinstance(expected, dict):
        if not isinstance(actual, dict) or actual.keys() != expected.keys():
            raise ValueError(f'{path}: fields differ')
        for key in expected:
            assert_actual_equal(actual[key], expected[key], f'{path}.{key}')
    elif isinstance(expected, (list, tuple)):
        if type(actual) != type(expected) or len(actual) != len(expected):
            raise ValueError(f'{path}: sequence differs')
        for index, value in enumerate(expected):
            assert_actual_equal(actual[index], value, f'{path}[{index}]')
    elif actual != expected:
        raise ValueError(f'{path}: value differs')


def direct_binding(source, *, common_init, shared_init=None, statistics=None,
                   seed=42, total_positions=None):
    base = source.base
    if base.kind != 'pi1m':
        raise ValueError('direct pretraining requires PI1M')
    indices = np.asarray([source.source_index_at(i) for i in range(len(source))], dtype=np.int64)
    rows = [base.entries[int(index)] for index in indices]
    if any(row['pretrain_role'] != 'train' for row in rows):
        raise ValueError('direct pretraining selected non-train rows')
    if not common_init:
        raise ValueError('direct pretraining requires a shared common initialization')
    payload = torch.load(common_init, map_location='cpu', weights_only=False)
    binding = dict(schema='mcl-ph-direct-run-binding-v1', source_root=str(base.root.resolve()),
                   source_indices=indices,
                   source_keys=np.frombuffer(b''.join(bytes.fromhex(row['sample_key']) for row in rows), dtype=np.uint8).reshape(-1,32),
                   normalized_smiles=[row['normalized_smiles'] for row in rows],
                   source_smiles=[row['source_smiles'] for row in rows],
                   common_initial_state=payload['common_state_dict'])
    from src.training.glt_dual_runtime import OrderedSampleStream
    total = len(source) if total_positions is None else int(total_positions)
    if total <= 0:
        raise ValueError('sampling coverage must be positive')
    stream = OrderedSampleStream(len(source), int(seed))
    order = np.empty(total, dtype=np.int64)
    for first in range(0, total, len(source)):
        stream.index_at(first)
        count = min(len(source), total - first)
        order[first:first + count] = indices[stream.permutation[:count].numpy()]
    binding.update(sampling_seed=int(seed), sampling_source_indices=order)
    if shared_init:
        binding['shared_initial_state'] = torch.load(shared_init, map_location='cpu', weights_only=False)['state_dict']
    if statistics:
        with np.load(statistics, allow_pickle=False) as archive:
            binding['statistics'] = {key: archive[key].copy() for key in archive.files}
    return binding


def direct_identity(identity):
    """Legacy digest/count identifiers have no authority in a direct run."""
    return {**{name: value for name, value in identity.items()
               if 'sha256' not in name and 'hash' not in name},
            'source_identity_contract': 'actual-rows-arrays-tensors-v1'}


def verify_restored_checkpoint(model, optimizer, state, *, batch_size):
    """Check a load without taking an optimizer update or advancing data."""
    if (int(state['next_position']) != int(state['step']) * int(batch_size)
            or int(state['scheduler']['step']) != int(state['step'])):
        raise ValueError('checkpoint scheduler/data position differs')
    if any(not torch.isfinite(value).all() for value in state['model'].values()):
        raise ValueError('checkpoint model contains NaN/Inf')
    assert_actual_equal(model.state_dict(), state['model'], 'restored.model')
    assert_actual_equal(optimizer.state_dict(), state['optimizer'], 'restored.optimizer')
