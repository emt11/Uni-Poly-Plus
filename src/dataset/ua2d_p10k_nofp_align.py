"""Read-only original cache with preregistered split 2D/3D sample streams.

Values in the eligible mapping are LOGICAL slots, never source rows, exactly as
in the PTSCALE mapping this cycle reuses.  The 2D atom corruption draws from a
per-position mask stream that does not depend on the 3D stream, so PURE2D sees
byte-identical 2D masks at the same global position as every D arm.  This cycle
never unpacks the cached FP bits.
"""
from collections import OrderedDict
from pathlib import Path
import numpy as np
import torch
from src.dataset.ua2d_fast_pretrain import (CACHE_SCHEMA, decode_template, encode,
                                            noisy_geometry, worker_init)
from src.dataset.glt_dual import dual_glt_collate
from src.dataset.ua2d_galformer_pretrain import corruption
from src.modules.ua2d_p10k_nofp_align import (GEOMETRY_STREAM_SEED_SALT,
                                              MASK_STREAM_SEED_SALT,
                                              STREAM_MODULUS, NoFPAlignConfig)

MICROBATCHES = 10000 * 6


def resolve_slot(eligible, keys, source_rows, selected):
    logical = int(eligible[selected])
    if logical < 0 or logical >= len(keys) or len(keys) != len(source_rows):
        raise ValueError('logical slot out of bounds')
    return logical, keys[logical].tobytes(), int(source_rows[logical])


def micro_start(micro_index):
    if not 0 <= int(micro_index) < MICROBATCHES:
        raise ValueError('single-GPU micro index outside the fixed 10K trajectory')
    return int(micro_index) * 168


def split_sample_generators(seed, position):
    """Independent per-position 2D-mask and 3D-geometry sample streams."""
    base = (int(seed) + 1000003 * int(position)) % STREAM_MODULUS
    mask = torch.Generator().manual_seed((base + MASK_STREAM_SEED_SALT) % STREAM_MODULUS)
    geometry = torch.Generator().manual_seed((base + GEOMETRY_STREAM_SEED_SALT) % STREAM_MODULUS)
    return mask, geometry


def apply_atom_corruption(data, generator, ratio):
    """Historical 40% BERT corruption of the 2D atom features, mask stream only."""
    selected, action, donor = corruption(len(data.mips_x), generator, ratio)
    original = data.mips_x.clone()
    backbone = data.mips_backbone_mask.clone()
    data.mips_x[action == 1] = 0
    data.mips_backbone_mask[action == 1] = False
    random = action == 2
    data.mips_x[random] = original[donor[random]]
    data.mips_backbone_mask[random] = backbone[donor[random]]
    return selected


def prepare_arm_sample(template, *, config, seed, position):
    if not isinstance(config, NoFPAlignConfig):
        raise ValueError('arm sample requires the no-FP config')
    mask_generator, geometry_generator = split_sample_generators(seed, position)
    clean = template['clean']
    data = clean.clone()
    positions = template['positions'].clone()
    labels = dict(atom_label=template['atom_label'], identity=template['identity'],
                  strategy=config.strategy)
    selected = apply_atom_corruption(data, mask_generator, config.mask_ratio)
    labels['atom_selected'] = selected
    if config.branch3d:
        valid = bool(clean.geometry_valid)
        if valid:
            positions += config.noise_sigma * torch.randn(positions.shape, generator=geometry_generator)
            data.bond_distance, data.line_angle = noisy_geometry(template, positions)
            chosen, actions, donors = corruption(len(template['group_first']), geometry_generator, config.mask_ratio)
            index = template['group_index']
            token_selected = chosen[index]
            token_action = actions[index]
            token_donor = template['group_first'][donors[index]]
            physical = positions
        else:
            m = len(data.bond_distance)
            token_selected = torch.zeros(m, dtype=torch.bool)
            token_action = torch.zeros(m, dtype=torch.long)
            token_donor = torch.arange(m)
            physical = torch.empty((0, 3), dtype=torch.float32)
        labels.update(token_selected=token_selected, token_label=template['token_label'],
                      token_action=token_action, token_donor=token_donor,
                      physical_pos=physical,
                      physical_edges=torch.stack([template['endpoint_a'], template['endpoint_b']]),
                      physical_center=template['physical_center'])
    return data, labels


def collate_arm_samples(samples):
    if not samples or len({y['strategy'] for _, y in samples}) != 1:
        raise ValueError('empty or mixed-strategy pretraining batch')
    branch3d = 'token_selected' in samples[0][1]
    if len({'token_selected' in labels for _, labels in samples}) != 1:
        raise ValueError('mixed 3D-branch availability in batch')
    data = dual_glt_collate([x for x, _ in samples])
    names = ['atom_selected', 'atom_label']
    if branch3d:
        names += ['token_selected', 'token_label', 'token_action', 'physical_pos', 'physical_center']
    labels = {name: torch.cat([y[name] for _, y in samples]) for name in names}
    if branch3d:
        donors, edges, graphs = [], [], []
        bond_offset = position_offset = 0
        for graph, (x, y) in enumerate(samples):
            donors.append(y['token_donor'] + bond_offset)
            edges.append(y['physical_edges'] + position_offset)
            graphs.append(torch.full((len(y['physical_pos']),), graph, dtype=torch.long))
            bond_offset += len(x.bond_distance)
            position_offset += len(y['physical_pos'])
        labels.update(token_donor=torch.cat(donors), physical_edges=torch.cat(edges, dim=1),
                      physical_graph=torch.cat(graphs))
    labels['identity'] = [y['identity'] for _, y in samples]
    labels['strategy'] = samples[0][1]['strategy']
    return data, labels


class PreparedNoFPMicrobatches(torch.utils.data.Dataset):
    """One ordered item is one CPU microbatch, independent of worker scheduling."""

    def __init__(self, root, mapping_root, config, *, prepared_counts, cache_entries=2048):
        self.root = Path(root)
        self.mapping_root = Path(mapping_root)
        self.config = config
        self.prepared_counts = prepared_counts
        self.cache_entries = cache_entries
        self.environment = self.keys = self.rows = self.stream = None
        self.templates = OrderedDict()

    def __len__(self):
        return MICROBATCHES

    def _open(self):
        import json
        import lmdb
        from src.training.glt_dual_runtime import OrderedSampleStream
        meta = json.loads((self.root / 'metadata.json').read_text())
        if meta.get('schema') != CACHE_SCHEMA or meta.get('status') != 'COMPLETE' or meta.get('train_rows') != 911391:
            raise ValueError('prepared cache incomplete or incompatible')
        self.environment = lmdb.open(str(self.root / 'templates.lmdb'), readonly=True,
                                     lock=False, readahead=True, max_readers=128)
        self.keys = np.load(self.root / 'keys.npy', mmap_mode='r', allow_pickle=False)
        self.rows = np.load(self.root / 'source_rows.npy', mmap_mode='r', allow_pickle=False)
        if len(self.keys) != 911391 or self.rows.shape != (911391,):
            raise ValueError('cache row mapping changed')
        mapping = json.loads((self.mapping_root / 'metadata.json').read_text())
        if (mapping.get('plan') != 'UA2D-PTSCALE-20261022-01' or mapping.get('status') != 'COMPLETE'
                or mapping.get('schema') != 'ua2d-ptscale-logical-map-r1' or mapping.get('records') != 910756
                or mapping.get('parent_cache') != str(self.root)):
            raise ValueError('mapping identity/COMPLETE mismatch')
        self.eligible = np.load(self.mapping_root / 'eligible_logical.npy', mmap_mode='r', allow_pickle=False)
        self.selected_keys = np.load(self.mapping_root / 'keys.npy', mmap_mode='r', allow_pickle=False)
        self.selected_rows = np.load(self.mapping_root / 'source_rows.npy', mmap_mode='r', allow_pickle=False)
        if (self.eligible.shape != (910756,) or self.selected_keys.shape != (910756,)
                or self.selected_rows.shape != (910756,)):
            raise ValueError('logical mapping shape mismatch')
        self.stream = OrderedSampleStream(910756, 42)

    def __getitem__(self, micro_index):
        if self.environment is None:
            self._open()
        info = torch.utils.data.get_worker_info()
        worker = info.id if info else 0
        start = micro_start(int(micro_index))
        records, rows, logical_slots = [], [], []
        for offset in range(168):
            with self.prepared_counts.get_lock():
                self.prepared_counts[worker] += 1
            position = start + offset
            selected = self.stream.index_at(position)
            index, key, source_row = resolve_slot(self.eligible, self.keys, self.rows, selected)
            if key != self.selected_keys[selected].tobytes() or source_row != int(self.selected_rows[selected]):
                raise ValueError('persisted logical/source mapping mismatch')
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
            if not bool(template['clean'].geometry_valid):
                raise ValueError('excluded invalid geometry entered PT; no fallback')
            self.templates[key] = template
            if len(self.templates) > self.cache_entries:
                self.templates.popitem(last=False)
            records.append(prepare_arm_sample(template, config=self.config, seed=42, position=position))
            rows.append(source_row)
            logical_slots.append(index)
        data, labels = collate_arm_samples(records)
        return encode((data, labels, dict(micro_index=int(micro_index), start_position=start,
                                          source_rows=rows, logical_slots=logical_slots)))
