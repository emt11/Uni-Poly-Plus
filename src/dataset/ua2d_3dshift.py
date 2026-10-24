"""Ordered single-GPU reader for the UA2D-3DSHIFT sidecar cache.

The reader keeps the legacy prepared-template cache and the PTSCALE logical mapping
read-only.  One ordered item is one CPU microbatch of 168 logical slots; every
deterministic image/angle/incidence mapping comes from the sidecar LMDB, never from
online graph construction or a missing-record fallback.
"""
from collections import OrderedDict
from pathlib import Path
import json
import pickle

import numpy as np
import torch

from src.dataset.ua2d_fast_pretrain import (CACHE_SCHEMA, decode_batch, decode_template, encode, worker_init,
                                            prepare_fast_sample)
from src.dataset.ua2d_galformer_pretrain import collate_samples
from src.dataset.ua2d_ptscale import resolve_slot
from src.modules.ua2d_3dshift import PRETRAIN_UPDATES

SIDECAR_SCHEMA = 'ua2d-3dshift-input-r1'
SIDECAR_STATUS = 'COMPLETE'
RECORDS = 910756
EXCLUDED = 635
ROOT = Path(__file__).resolve().parents[2]
PARENT_CACHE = ROOT / 'results/ua2d_pretrain_fp_20261019/fast_r4/prepared_cache'
MAPPING = ROOT / 'results/ua2d_ptscale_20261022/r1/inputs'
TASKS = ('eea', 'egb', 'egc', 'ei', 'eps', 'nc', 'xc')


def decode_sidecar(payload):
    record = pickle.loads(payload)
    if record.get('schema') != SIDECAR_SCHEMA:
        raise ValueError('sidecar schema mismatch')
    return record


def sidecar_metadata(root):
    root = Path(root)
    meta = json.loads((root / 'metadata.json').read_text())
    if (meta.get('plan') != 'UA2D-3DSHIFT-20261022-01'
            or meta.get('schema') != SIDECAR_SCHEMA or meta.get('status') != SIDECAR_STATUS
            or meta.get('records') != RECORDS or meta.get('excluded_rows') != EXCLUDED
            or Path(meta.get('parent_cache','')).resolve() != PARENT_CACHE.resolve() or Path(meta.get('mapping','')).resolve() != MAPPING.resolve()):
        raise ValueError('3dshift sidecar identity/COMPLETE mismatch')
    return meta


def _tensor(value):
    return torch.from_numpy(np.ascontiguousarray(value))


def attach(data, record):
    """Attach the cached deterministic mapping to one prepared sample."""
    data.ds_token_count = int(record['token_count'])
    data.ds_group_count = int(record['group_count'])
    data.ds_atom_count = int(record['atom_count'])
    data.ds_rel1_count = int(record['rel1_count'])
    data.ds_rel2_count = int(record['rel2_count'])
    data.ds_obs_count = int(record['obs_count'])
    data.ds_path_count = int(record['path_count'])
    data.ds_token_group = _tensor(record['token_group']).long()
    data.ds_group_first = _tensor(record['group_first']).long()
    data.ds_group_atom_a = _tensor(record['group_atom_a']).long()
    data.ds_group_atom_b = _tensor(record['group_atom_b']).long()
    data.ds_group_shift = _tensor(record['group_shift']).long()
    data.ds_token_weight = _tensor(record['token_weight']).float()
    data.ds_group_weight = _tensor(record['group_weight']).float()
    data.ds_group_self = _tensor(record['group_self']).bool()
    data.ds_rel1_source = _tensor(record['rel1_source']).long()
    data.ds_rel1_target = _tensor(record['rel1_target']).long()
    data.ds_rel1_shift = _tensor(record['rel1_shift']).long()
    data.ds_rel1_self = _tensor(record['rel1_self']).bool()
    data.ds_rel1_obs_count = _tensor(record['rel1_obs_count']).long()
    data.ds_obs_source = _tensor(record['obs_source']).long()
    data.ds_obs_target = _tensor(record['obs_target']).long()
    data.ds_obs_pos = _tensor(record['obs_pos']).long()
    data.ds_rel2_source = _tensor(record['rel2_source']).long()
    data.ds_rel2_target = _tensor(record['rel2_target']).long()
    data.ds_rel2_shift = _tensor(record['rel2_shift']).long()
    data.ds_rel2_path_count = _tensor(record['rel2_path_count']).long()
    data.ds_rel2_path = _tensor(record['rel2_path']).long()
    data.ds_inc_bond = _tensor(record['inc_bond']).long()
    data.ds_inc_atom = _tensor(record['inc_atom']).long()
    rel1 = data.ds_rel1_count
    if data.ds_rel1_obs_count.sum().item() != data.ds_obs_count:
        raise ValueError('cached observation grouping mismatch')
    data.ds_obs_rel = torch.repeat_interleave(torch.arange(rel1), data.ds_rel1_obs_count)
    return data


def collate_shift(records):
    """Concatenate prepared samples and their per-graph cached mappings."""
    data, labels = collate_samples(records)
    fields = dict(token_group=[], group_first=[], group_atom_a=[], group_atom_b=[], group_shift=[],
                  token_weight=[], group_weight=[], group_self=[], rel1_source=[], rel1_target=[], rel1_shift=[],
                  rel1_self=[], rel1_obs_count=[], obs_source=[], obs_target=[], obs_pos=[],
                  obs_rel=[], rel2_source=[], rel2_target=[], rel2_shift=[], rel2_path_count=[],
                  rel2_path=[], inc_bond=[], inc_atom=[], group_masked=[], group_action=[])
    offsets = dict(token=0, group=0, atom=0, rel1=0, rel2=0, path=0, obs=0, position=0)
    graphs = dict(group=[], atom=[])
    for index, (sample, sample_labels) in enumerate(records):
        add = dict(token_group=offsets['group'], group_first=offsets['token'],
                   group_atom_a=offsets['atom'], group_atom_b=offsets['atom'],
                   rel1_source=offsets['group'], rel1_target=offsets['group'],
                   obs_source=offsets['token'], obs_target=offsets['token'],
                   obs_pos=offsets['position'], obs_rel=offsets['rel1'],
                   rel2_source=offsets['group'], rel2_target=offsets['group'],
                   rel2_path=offsets['rel1'], inc_bond=offsets['group'], inc_atom=offsets['atom'])
        for name, shift in add.items():
            value = getattr(sample, 'ds_' + name)
            fields[name].append(value + shift if shift else value)
        fields['group_shift'].append(sample.ds_group_shift)
        fields['token_weight'].append(sample.ds_token_weight)
        fields['group_action'].append(sample.ds_group_action)
        fields['group_weight'].append(sample.ds_group_weight)
        fields['group_self'].append(sample.ds_group_self)
        fields['rel1_shift'].append(sample.ds_rel1_shift)
        fields['rel1_self'].append(sample.ds_rel1_self)
        fields['rel1_obs_count'].append(sample.ds_rel1_obs_count)
        fields['rel2_shift'].append(sample.ds_rel2_shift)
        fields['rel2_path_count'].append(sample.ds_rel2_path_count)
        fields['group_masked'].append(sample.ds_group_masked)
        graphs['group'].append(torch.full((sample.ds_group_count,), index, dtype=torch.long))
        graphs['atom'].append(torch.full((sample.ds_atom_count,), index, dtype=torch.long))
        offsets['token'] += sample.ds_token_count
        offsets['group'] += sample.ds_group_count
        offsets['atom'] += sample.ds_atom_count
        offsets['rel1'] += sample.ds_rel1_count
        offsets['rel2'] += sample.ds_rel2_count
        offsets['path'] += sample.ds_path_count
        offsets['obs'] += sample.ds_obs_count
        offsets['position'] += int(sample_labels['physical_pos'].size(0))
    for name, chunks in fields.items():
        setattr(data, 'ds_' + name, torch.cat(chunks))
    data.ds_group_graph = torch.cat(graphs['group'])
    data.ds_atom_graph = torch.cat(graphs['atom'])
    data.ds_token_count = offsets['token']
    data.ds_group_count = offsets['group']
    data.ds_atom_count = offsets['atom']
    data.ds_rel1_count = offsets['rel1']
    data.ds_rel2_count = offsets['rel2']
    data.ds_obs_count = offsets['obs']
    data.ds_path_count = offsets['path']
    return data, labels


class PreparedShiftedMicrobatches(torch.utils.data.Dataset):
    """One ordered item is one 168-template CPU microbatch of the fixed 10000x6 cycle."""

    def __init__(self, root, sidecar_root, mapping_root, config, *, variant, prepared_counts,
                 cache_entries=256):
        if not config.reconstruct_3d_mask or not config.fp_enabled:
            raise ValueError('3dshift keeps the mask_align + FP recipe')
        if variant not in ('F_G10K', 'P_G', 'F_L', 'P_L'):
            raise ValueError('unknown 3dshift variant')
        self.root = Path(root)
        self.sidecar_root = Path(sidecar_root)
        self.mapping_root = Path(mapping_root)
        self.config = config
        self.variant = variant
        self.prepared_counts = prepared_counts
        self.cache_entries = cache_entries
        self.environment = self.sidecar = None
        self.eligible = self.keys = self.rows = self.stream = None
        self.templates = OrderedDict()
        self.records = OrderedDict()

    def __len__(self):
        return PRETRAIN_UPDATES * 6

    def _open(self):
        import lmdb
        from src.training.glt_dual_runtime import OrderedSampleStream

        meta = json.loads((self.root / 'metadata.json').read_text())
        if meta.get('schema') != CACHE_SCHEMA or meta.get('status') != 'COMPLETE' or meta.get('train_rows') != 911391:
            raise ValueError('prepared template cache incomplete or incompatible')
        sidecar_metadata(self.sidecar_root)
        mapping = json.loads((self.mapping_root / 'metadata.json').read_text())
        if (mapping.get('plan') != 'UA2D-PTSCALE-20261022-01' or mapping.get('status') != 'COMPLETE'
                or mapping.get('schema') != 'ua2d-ptscale-logical-map-r1'
                or mapping.get('records') != RECORDS or mapping.get('excluded') != EXCLUDED
                or mapping.get('parent_cache') != str(self.root)):
            raise ValueError('logical mapping identity/COMPLETE mismatch')
        self.environment = lmdb.open(str(self.root / 'templates.lmdb'), readonly=True, lock=False,
                                     readahead=True, max_readers=128)
        self.sidecar = lmdb.open(str(self.sidecar_root / 'sidecar.lmdb'), readonly=True, lock=False,
                                 readahead=True, max_readers=128)
        self.keys = np.load(self.root / 'keys.npy', mmap_mode='r', allow_pickle=False)
        self.rows = np.load(self.root / 'source_rows.npy', mmap_mode='r', allow_pickle=False)
        self.eligible = np.load(self.mapping_root / 'eligible_logical.npy', mmap_mode='r', allow_pickle=False)
        self.selected_keys = np.load(self.mapping_root / 'keys.npy', mmap_mode='r', allow_pickle=False)
        self.selected_rows = np.load(self.mapping_root / 'source_rows.npy', mmap_mode='r', allow_pickle=False)
        if self.keys.shape != (911391,) or self.rows.shape != (911391,):
            raise ValueError('cache row mapping changed')
        if (self.eligible.shape != (RECORDS,) or self.selected_keys.shape != (RECORDS,)
                or self.selected_rows.shape != (RECORDS,)):
            raise ValueError('logical mapping shape mismatch')
        self.stream = OrderedSampleStream(RECORDS, 42)

    def _record(self, key):
        if key in self.records:
            record=self.records.pop(key);self.records[key]=record;return record
        with self.sidecar.begin() as txn:
            payload = txn.get(key)
        if payload is None:
            raise ValueError('3dshift sidecar record missing; no fallback')
        record = decode_sidecar(payload)
        self.records[key] = record
        if len(self.records) > self.cache_entries:
            self.records.popitem(last=False)
        return record

    def __getitem__(self, micro_index):
        if self.environment is None:
            self._open()
        info = torch.utils.data.get_worker_info()
        worker = info.id if info else 0
        micro_index = int(micro_index)
        start = micro_index * 168
        records, rows, logical_slots = [], [], []
        for offset in range(168):
            with self.prepared_counts.get_lock():
                self.prepared_counts[worker] += 1
            position = start + offset
            selected = self.stream.index_at(position)
            index, key, source_row = resolve_slot(self.eligible, self.keys,
                                                  self.rows, selected)
            if key != self.selected_keys[selected].tobytes() or source_row != int(self.selected_rows[selected]):
                raise ValueError('persisted logical/source mapping mismatch')
            record = self._record(key)
            if (record['logical_slot'] != index or record['source_row'] != source_row
                    or record['sample_key'] != key):
                raise ValueError('sidecar logical/source identity mismatch')
            if key in self.templates:
                template = self.templates.pop(key)
            else:
                with self.environment.begin() as txn:
                    payload = txn.get(key)
                if payload is None:
                    raise ValueError('prepared template missing')
                template = decode_template(payload)
                if template['sample_key'] != key:
                    raise ValueError('prepared identity mismatch')
            self.templates[key] = template
            if len(self.templates) > self.cache_entries:
                self.templates.popitem(last=False)
            if not bool(template['clean'].geometry_valid):
                raise ValueError('excluded invalid geometry entered PT; no fallback')
            if record['identity'] != template['identity']:
                raise ValueError('sidecar/template identity mismatch')
            if (int(record['token_count']) != len(template['endpoint_a'])
                    or int(record['token_count']) != len(template['group_index'])
                    or int(record['atom_count']) != int(template['atom_label'].numel())):
                raise ValueError('sidecar shape does not match the prepared template')
            fp = np.unpackbits(template['fp_bits'].numpy(), bitorder='little') if self.config.fp_enabled else None
            data, labels = prepare_fast_sample(template, config=self.config, seed=42, position=position,
                                               fp_target=fp)
            attach(data, record)
            if not torch.equal(data.ds_token_group, torch.as_tensor(template['group_index']).long()):
                raise ValueError('cached canonical partition differs from corruption groups')
            if not torch.equal(data.ds_group_first, torch.as_tensor(template['group_first']).long()):
                raise ValueError('cached canonical representative differs from corruption donors')
            action = labels['token_action'].long()
            if not torch.equal(action, action[data.ds_group_first][data.ds_token_group]):
                raise ValueError('mask action differs inside one canonical bond group')
            data.ds_group_action = action[data.ds_group_first].clone()
            data.ds_group_masked = data.ds_group_action != 0
            if int(labels['atom_selected'].numel()) != data.ds_atom_count:
                raise ValueError('canonical atom count mismatch')
            records.append((data, labels))
            rows.append(source_row)
            logical_slots.append(index)
        data, labels = collate_shift(records)
        return encode((data, labels, dict(micro_index=micro_index, start_position=start,
                                          source_rows=rows, logical_slots=logical_slots,
                                          variant=self.variant)))
