"""Prepared immutable templates and ordered CPU prefetch for unchanged PT tasks."""
from collections import OrderedDict
import pickle
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data

from .glt_dual import build_dual_sample
from .ua2d_galformer_pretrain import corruption, collate_samples

CACHE_SCHEMA = 'ua2d_prepared_templates_r4'


def _arrays(value):
    if isinstance(value, Data):
        value = value.to_dict()
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    if isinstance(value, dict):
        return {k: _arrays(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(_arrays(v) for v in value)
    if isinstance(value, list):
        return [_arrays(v) for v in value]
    return value


def _tensors(value):
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value)
    if isinstance(value, dict):
        return {k: _tensors(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(_tensors(v) for v in value)
    if isinstance(value, list):
        return [_tensors(v) for v in value]
    return value


def encode(value):
    # Bytes cross the multiprocessing pipe; tensors do not consume /dev/shm.
    return pickle.dumps(_arrays(value), protocol=pickle.HIGHEST_PROTOCOL)


def decode_template(payload):
    value = _tensors(pickle.loads(payload))
    value['clean'] = Data.from_dict(value['clean'])
    return value


def decode_batch(payload):
    fields, labels, trace = _tensors(pickle.loads(payload))
    return Data.from_dict(fields), labels, trace


def build_template(topology, trimer, smiles, static, identity=None):
    """Compute clean geometry, canonical joins, labels and periodic groups once."""
    from src.modules.glt_dual import triplet_type
    clean = build_dual_sample(topology, trimer, smiles, static=static,
                              geometry_mode='finite_incident', identity=identity)
    m = len(clean.bond_distance)
    valid = bool(clean.geometry_valid)
    positions = trimer.trimer_pos.clone().float()
    if valid:
        a = torch.as_tensor(np.array(static['token_pos_index_a'], copy=True)).long()
        b = torch.as_tensor(np.array(static['token_pos_index_b'], copy=True)).long()
        triplets = torch.as_tensor(np.array(static['angle_pos_triplet'], copy=True)).long()
        center = torch.zeros(len(positions), dtype=torch.bool)
        center[torch.as_tensor(trimer.mips_to_trimer_central_index).long()] = True
        offsets = torch.as_tensor(trimer.trimer_ru_offset)
        groups = {}
        group_index = torch.empty(m, dtype=torch.long)
        first = []
        for i, (ca, cb) in enumerate(zip(clean.bond_canonical_a.tolist(), clean.bond_canonical_b.tolist())):
            shift = int(offsets[b[i]] - offsets[a[i]])
            key = min((ca, cb, shift), (cb, ca, -shift))
            if key not in groups:
                groups[key] = len(groups)
                first.append(i)
            group_index[i] = groups[key]
        real = (clean.line_mask.sum(-1) == 1) & ~clean.line_is_self & clean.line_center_angle
        pairs = torch.sort(clean.line_path[real, :2], dim=-1).values
        unique = {}
        for pair, value in zip(pairs.tolist(), clean.line_angle[real, 0].cos()):
            unique.setdefault(tuple(pair), value)
        angle_pairs = torch.tensor(list(unique), dtype=torch.long).reshape(-1, 2)
        angle_target = torch.stack(list(unique.values())) if unique else torch.empty(0)
    else:
        a = b = torch.empty(0, dtype=torch.long)
        triplets = torch.empty((0, 2, 3), dtype=torch.long)
        center = torch.empty(0, dtype=torch.bool)
        group_index = torch.empty(0, dtype=torch.long)
        first = []
        angle_pairs = torch.empty((0, 2), dtype=torch.long)
        angle_target = torch.empty(0)
    return dict(clean=clean, positions=positions, endpoint_a=a, endpoint_b=b,
                triplets=triplets, physical_center=center, group_index=group_index,
                group_first=torch.tensor(first, dtype=torch.long),
                atom_label=clean.mips_x[:, :101].argmax(-1),
                token_label=triplet_type(clean.bond_z_a, clean.bond_z_b, clean.bond_type),
                angle_pairs=angle_pairs, angle_target=angle_target,
                identity=str(topology.normalized_canonical_smiles))


def noisy_geometry(template, positions):
    """Same geometry arithmetic as materialize_dual_geometry, with cached indices."""
    clean = template['clean']
    if not bool(clean.geometry_valid):
        return clean.bond_distance.clone(), clean.line_angle.clone()
    a, b = template['endpoint_a'], template['endpoint_b']
    distances = torch.linalg.vector_norm(positions[a] - positions[b], dim=-1)
    triplets = template['triplets']
    mask = clean.line_mask.bool()
    angles = torch.zeros(mask.shape, dtype=positions.dtype)
    real = mask & (triplets[..., 0] >= 0)
    if bool(real.any()):
        rows = triplets[real]
        first = positions[rows[:, 0]] - positions[rows[:, 1]]
        second = positions[rows[:, 2]] - positions[rows[:, 1]]
        denom = torch.linalg.vector_norm(first, dim=-1) * torch.linalg.vector_norm(second, dim=-1)
        if not torch.isfinite(denom).all() or (denom <= 0).any():
            raise ValueError('nonfinite/degenerate noisy angle')
        angles[real] = torch.acos(torch.clamp((first * second).sum(-1) / denom, -1., 1.))
    if not torch.isfinite(distances).all() or (distances <= 0).any() or not torch.isfinite(angles).all():
        raise ValueError('invalid noisy geometry')
    return distances, angles


def prepare_fast_sample(template, *, config, seed, position, fp_target=None):
    generator = torch.Generator().manual_seed((int(seed) + 1000003 * int(position)) % (2**63 - 1))
    clean = template['clean']
    data = clean.clone()
    positions = template['positions'].clone()
    if bool(clean.geometry_valid):
        positions += config.noise_sigma * torch.randn(positions.shape, generator=generator)
    data.bond_distance, data.line_angle = noisy_geometry(template, positions)
    selected, action, donor = corruption(len(data.mips_x), generator, config.mask_ratio)
    original = data.mips_x.clone()
    backbone = data.mips_backbone_mask.clone()
    data.mips_x[action == 1] = 0
    data.mips_backbone_mask[action == 1] = False
    random = action == 2
    data.mips_x[random] = original[donor[random]]
    data.mips_backbone_mask[random] = backbone[donor[random]]
    valid = bool(clean.geometry_valid)
    physical = positions if valid else torch.empty((0, 3), dtype=torch.float32)
    labels = dict(atom_selected=selected, atom_label=template['atom_label'],
                  identity=template['identity'], strategy=config.strategy,
                  physical_pos=physical,
                  physical_edges=torch.stack([template['endpoint_a'], template['endpoint_b']]),
                  physical_center=template['physical_center'])
    if config.reconstruct_3d_mask:
        m = len(data.bond_distance)
        if valid:
            chosen, actions, donors = corruption(len(template['group_first']), generator, config.mask_ratio)
            index = template['group_index']
            token_selected = chosen[index]
            token_action = actions[index]
            token_donor = template['group_first'][donors[index]]
        else:
            token_selected = torch.zeros(m, dtype=torch.bool)
            token_action = torch.zeros(m, dtype=torch.long)
            token_donor = torch.arange(m)
        labels.update(token_selected=token_selected, token_action=token_action,
                      token_donor=token_donor, token_label=template['token_label'])
    if config.reconstruct_geometry:
        if valid:
            original_pos = template['positions']
            coordinate = (original_pos - original_pos.mean(0)) - (positions - positions.mean(0))
        else:
            coordinate = physical.clone()
        labels.update(coordinate_target=coordinate, length_target=clean.bond_distance,
                      angle_pairs=template['angle_pairs'], angle_target=template['angle_target'])
    if config.fp_enabled:
        if fp_target is None:raise ValueError('FP target missing')
        labels['fp_target'] = torch.as_tensor(fp_target, dtype=torch.float32).clone()
    elif fp_target is not None:
        raise ValueError('FP-off must not receive targets')
    return data, labels


def worker_init(_):
    torch.set_num_threads(1)


class PreparedMicrobatches(torch.utils.data.Dataset):
    """One ordered item is one CPU microbatch, independent of worker scheduling."""
    def __init__(self, root, config, *, rank, prepared_counts, cache_entries=2048):
        self.root = Path(root)
        self.config = config
        self.rank = int(rank)
        self.prepared_counts = prepared_counts
        self.cache_entries = cache_entries
        self.environment = self.keys = self.rows = self.stream = None
        self.templates = OrderedDict()

    def __len__(self):return 5000 * 6

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
        if len(self.keys) != 911391 or self.rows.shape != (911391,):raise ValueError('cache row mapping changed')
        self.stream = OrderedSampleStream(911391, 42)

    def __getitem__(self, micro_index):
        if self.environment is None:self._open()
        info = torch.utils.data.get_worker_info()
        worker = info.id if info else 0
        start = (int(micro_index) * 2 + self.rank) * 84
        records, rows = [], []
        for offset in range(84):
            with self.prepared_counts.get_lock():self.prepared_counts[worker] += 1
            position = start + offset
            index = self.stream.index_at(position)
            key = self.keys[index].tobytes()
            if key in self.templates:
                template = self.templates.pop(key)
            else:
                with self.environment.begin() as txn:payload = txn.get(key)
                if payload is None:raise ValueError('prepared template missing')
                template = decode_template(payload)
                if template['sample_key'] != key:raise ValueError('prepared identity mismatch')
            self.templates[key] = template
            if len(self.templates) > self.cache_entries:self.templates.popitem(last=False)
            fp = np.unpackbits(template['fp_bits'].numpy(), bitorder='little') if self.config.fp_enabled else None
            records.append(prepare_fast_sample(template, config=self.config, seed=42, position=position, fp_target=fp))
            rows.append(int(self.rows[index]))
        data, labels = collate_samples(records)
        return encode((data, labels, dict(micro_index=int(micro_index), start_position=start, source_rows=rows)))
