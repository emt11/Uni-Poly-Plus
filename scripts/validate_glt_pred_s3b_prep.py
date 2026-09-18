#!/usr/bin/env python3
"""Run the bounded, no-update S3b subset and fixed-P_val reader smoke.

Reads the four frozen formal configs, one at a time, opens the P_train subset
reader they declare, prepares a small number of real positions and compares the
public inputs, mask/noise streams and sample order across the four variants.
No optimizer step, no forward/backward, no outer-test access.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from src.dataset.glt_dual_pretrain import prepare_pretrain_sample
from src.training.glt_dual_runtime import (IndexedFrozenDualSource,
                                            load_sample_index_artifact,
                                            open_source, OrderedSampleStream)


def _digest(value):
    digest = hashlib.sha256()
    if torch.is_tensor(value):
        value = value.detach().cpu().contiguous()
        digest.update(str(tuple(value.shape)).encode('ascii'))
        digest.update(str(value.dtype).encode('ascii'))
        digest.update(value.numpy().tobytes())
    elif isinstance(value, dict):
        for key in sorted(value):
            digest.update(str(key).encode('utf-8'))
            digest.update(_digest(value[key]).encode('ascii'))
    elif isinstance(value, (list, tuple)):
        for item in value:
            digest.update(_digest(item).encode('ascii'))
    elif value is None:
        digest.update(b'None')
    else:
        digest.update(str(value).encode('utf-8'))
    return digest.hexdigest()


def _finite_data(data):
    for value in data.to_dict().values():
        if torch.is_tensor(value) and not bool(torch.isfinite(value.float()).all()):
            return False
    return True


def _load_fgr_stats(path):
    payload = json.loads(Path(path).read_text(encoding='utf-8'))
    if payload.get('full_p_train_scan') is not True:
        raise ValueError('FGR stats are not a full P_train scan')
    for name in ('mu', 'sigma'):
        if name not in payload or not math.isfinite(float(payload[name])):
            raise ValueError(f'FGR stats {name} is missing or nonfinite')
    if float(payload['sigma']) <= 0:
        raise ValueError('FGR stats sigma is not positive')
    return payload


def _open_variant(args, task, indices=None):
    """Open one task variant exactly as the runner does, including the subset view."""

    source, _ = open_source(
        args.pi1m_cohort_root, args.cache_root,
        dual_static_root=args.dual_static_root,
        pretrain_target_root=(args.pretrain_target_root if task == 'fp' else None),
    )
    if indices is not None:
        source = IndexedFrozenDualSource(source, indices)
    return source


def build(args):
    configs = []
    for path in args.config:
        payload = json.loads(Path(path).read_text(encoding='utf-8'))
        payload['_path'] = str(Path(path).resolve())
        configs.append(payload)
    if len(configs) != 4:
        raise ValueError('the smoke requires exactly the four formal configs')
    stats = _load_fgr_stats(args.fgr_stats)
    split = load_sample_index_artifact(args.split_artifact, 'train')
    p_train = set(int(value) for value in split['indices'].tolist())
    fixed_payload = json.loads(Path(args.fixed_pval).read_text(encoding='utf-8'))
    fixed_indices = np.asarray(fixed_payload['source_indices'], dtype=np.int64)
    p_val = set(int(value) for value in
                load_sample_index_artifact(args.split_artifact, 'validation')['indices'].tolist())
    downstream_ids = set()
    with Path(args.downstream_records).open(encoding='utf-8') as handle:
        for line in handle:
            if line.strip():
                downstream_ids.add(str(json.loads(line)['normalized_smiles']))

    common_paths = {config.get('common_init_artifact') for config in configs}
    if len(common_paths) != 1 or not Path(next(iter(common_paths))).is_file():
        raise AssertionError('the four configs must point at one existing common-init artifact')
    if {config.get('sample_index_artifact') for config in configs} != {args.split_artifact}:
        raise AssertionError('the four configs must share the frozen P_train split artifact')
    for config in configs:
        if config.get('sample_index_split') != 'train':
            raise AssertionError('formal configs must use sample_index_split=train')
        if int(config.get('expected_world_size', -1)) != 4 or int(config['global_batch']) != 1008 \
                or int(config['microbatch']) != 84:
            raise AssertionError('formal configs must keep the 4-GPU microbatch/global-batch contract')

    subset = {'positions': int(args.positions), 'tasks': {}}
    reference = None
    for config in configs:
        task = str(config.get('third_task', 'fp'))
        source = _open_variant(args, task, split['indices'])
        try:
            if task == 'fp' and source.target_cache is None:
                raise AssertionError('FP target cache was not opened for the fp config')
            if task != 'fp' and source.target_cache is not None:
                raise AssertionError(f'{task} unexpectedly opened the FP target cache')
            if len(source) != len(p_train):
                raise AssertionError('subset source length does not match the P_train artifact')
            if task == 'fgr':
                if float(config['fgr_mu']) != float(stats['mu']) or \
                        float(config['fgr_sigma']) != float(stats['sigma']):
                    raise AssertionError('T_FGR config does not carry the fitted P_train statistics')
            if task == 'align' and float(config.get('align_temperature', -1)) != 0.1:
                raise AssertionError('T_ALIGN config must keep temperature 0.1')
            logical_stream = OrderedSampleStream(len(split['indices']), 42)
            rows = []
            for position in range(int(args.positions)):
                logical = logical_stream.index_at(position)
                source_index = int(split['indices'][logical])
                if source_index in p_val:
                    raise AssertionError('P_train stream selected a P_val index')
                row = source.cohort['records'][source_index]
                if str(row['normalized_smiles']) in downstream_ids:
                    raise AssertionError('P_train stream selected a downstream identity')
                key, _ = source.samples[logical]
                static = source.static_for(logical)
                target = source.target_for(logical) if task == 'fp' else None
                data, labels = prepare_pretrain_sample(
                    *source[logical], seed=42, key=key.hex(), position=position,
                    sigma=0.03, ratio=0.30, static=static, target=target,
                    third_task=task, fgr_mu=float(config.get('fgr_mu', 0.0)),
                    fgr_sigma=float(config.get('fgr_sigma', 1.0)),
                    fgr_max_pairs=int(config.get('fgr_max_pairs', 32)))
                if not _finite_data(data):
                    raise ValueError(f'{task} produced nonfinite prepared tensor')
                if task == 'fgr' and not bool(torch.isfinite(labels['fgr_target']).all()):
                    raise ValueError('FGR target is nonfinite')
                if task == 'align' and not isinstance(labels['align_identity'], str):
                    raise ValueError('ALIGN identity metadata is missing')
                if task == 'none' and labels.get('fp_target') is not None:
                    raise ValueError('B_NONE prepared an FP target')
                rows.append({'position': position, 'source_index': source_index,
                             'sample_key': key.hex(),
                             'data_digest': _digest(data.to_dict()),
                             'mask_digest': _digest(labels['atom_mask'])})
            subset['tasks'][task] = {
                'config': config['_path'], 'positions': rows,
                'target_cache_opened': bool(source.target_cache is not None),
                'finite_prepared_tensors': True,
            }
            if reference is None:
                reference = rows
            else:
                for expected, observed in zip(reference, rows):
                    if (expected['source_index'], expected['sample_key'], expected['data_digest'],
                            expected['mask_digest']) != (observed['source_index'], observed['sample_key'],
                                                         observed['data_digest'], observed['mask_digest']):
                        raise AssertionError('common stream/public input parity mismatch')
        finally:
            source.close()

    fixed = {'count': int(len(fixed_indices)), 'tasks': {}}
    for config in configs:
        task = str(config.get('third_task', 'fp'))
        if not np.array_equal(fixed_indices, np.asarray(fixed_payload['source_indices'])):
            raise AssertionError('fixed P_val artifact order changed')
        source = _open_variant(args, task, fixed_indices)
        try:
            records = source.cohort['records']
            result = {'valid_count': 0, 'fgr_count': 0, 'fgr_spd2': 0, 'fgr_spd3': 0,
                      'fp_target_count': 0, 'align_valid_count': 0, 'source_indices': []}
            for position in range(len(fixed_indices)):
                source_index = int(fixed_indices[position])
                row = records[source_index]
                if row['normalized_smiles'] in downstream_ids:
                    raise AssertionError('fixed P_val contains a downstream identity')
                key, _ = source.samples[position]
                static = source.static_for(position)
                target = source.target_for(position) if task == 'fp' else None
                _, labels = prepare_pretrain_sample(
                    *source[position], seed=42, key=key.hex(), position=position,
                    sigma=0.03, ratio=0.30, static=static, target=target,
                    third_task=task, fgr_mu=float(config.get('fgr_mu', 0.0)),
                    fgr_sigma=float(config.get('fgr_sigma', 1.0)),
                    fgr_max_pairs=int(config.get('fgr_max_pairs', 32)))
                result['source_indices'].append(int(source_index))
                result['valid_count'] += int(bool(static['geometry_valid']))
                if task == 'fp':
                    result['fp_target_count'] += int(target is not None)
                elif task == 'fgr':
                    result['fgr_count'] += int(labels['fgr_target'].numel() > 0)
                    result['fgr_spd2'] += int((labels['fgr_spd'] == 2).sum())
                    result['fgr_spd3'] += int((labels['fgr_spd'] == 3).sum())
                elif task == 'align':
                    result['align_valid_count'] += int(bool(labels['align_valid']))
            result['source_order_exact'] = result['source_indices'] == fixed_indices.tolist()
            result.pop('source_indices')
            fixed['tasks'][task] = result
        finally:
            source.close()
    payload = {'split_artifact_sha256': split['sha256'],
               'p_train_count': int(len(p_train)),
               'p_val_count': int(len(p_val)),
               'p_train_p_val_intersection': int(len(p_train & p_val)),
               'subset_source_length': int(len(p_train)),
               'common_init_artifact': next(iter(common_paths)),
               'configs': [config['_path'] for config in configs],
               'subset': subset, 'fixed_validation': fixed,
               'optimizer_updates': 0, 'formal_training_started': False,
               'outer_test_accessed': False, 'active_cache_modified': False,
               'new_conformer_generated': False}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2,
                                 sort_keys=True, allow_nan=False) + '\n', encoding='utf-8')
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pi1m-cohort-root', required=True)
    parser.add_argument('--downstream-records', required=True)
    parser.add_argument('--cache-root', required=True)
    parser.add_argument('--dual-static-root', required=True)
    parser.add_argument('--pretrain-target-root', required=True)
    parser.add_argument('--split-artifact', required=True)
    parser.add_argument('--fixed-pval', required=True)
    parser.add_argument('--fgr-stats', required=True)
    parser.add_argument('--config', action='append', required=True,
                        help='one of the four frozen formal configs; repeat for all four')
    parser.add_argument('--positions', type=int, default=32)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.positions <= 0:
        raise ValueError('--positions must be positive')
    print(json.dumps(build(args), ensure_ascii=False, indent=2, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
