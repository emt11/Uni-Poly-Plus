"""Independent O8/Galformer views; no changes to running legacy campaigns."""
import copy
import math
import numpy as np
import torch

from .glt_dual import build_dual_sample, dual_glt_collate
from .glt_dual_static import build_dual_static


def corruption(size, generator, ratio):
    """Selected positions and BERT actions: 0 unchanged, 1 MASK, 2 random."""
    selected = torch.zeros(size, dtype=torch.bool)
    action = torch.zeros(size, dtype=torch.long)
    donor = torch.arange(size)
    count = min(size, max(1, math.ceil(size * ratio))) if size else 0
    indices = torch.randperm(size, generator=generator)[:count]
    selected[indices] = True
    draw = torch.rand(count, generator=generator)
    action[indices[draw < .8]] = 1
    random = indices[(draw >= .8) & (draw < .9)]
    if size > 1:
        other = torch.randint(size - 1, (len(random),), generator=generator)
        donor[random] = other + (other >= random).long()
        action[random] = 2
    else:
        action[random] = 1
    return selected, action, donor


def prepare_sample(topology, trimer, smiles, *, config, seed, position, static=None,
                   identity=None, fp_target=None):
    """Share noisy coordinates/2D masks; apply independent 3D masks only in V1."""
    generator = torch.Generator().manual_seed((int(seed) + 1000003 * int(position)) % (2**63 - 1))
    if static is None:
        static = build_dual_static(topology, trimer, smiles)
    clean = build_dual_sample(topology, trimer, smiles, static=static,
                              geometry_mode='finite_incident', identity=identity)
    changed = copy.copy(trimer)
    positions = trimer.trimer_pos.clone().float()
    if clean.geometry_valid:
        positions += config.noise_sigma * torch.randn(positions.shape, generator=generator)
    changed.trimer_pos = positions
    data = build_dual_sample(topology, changed, smiles, static=static,
                             geometry_mode='finite_incident', identity=identity)
    selected, action, donor = corruption(len(data.mips_x), generator, config.mask_ratio)
    original = data.mips_x.clone()
    original_backbone = data.mips_backbone_mask.clone()
    data.mips_x[action == 1] = 0
    data.mips_backbone_mask[action == 1] = False
    random = action == 2
    data.mips_x[random] = original[donor[random]]
    data.mips_backbone_mask[random] = original_backbone[donor[random]]
    labels = dict(atom_selected=selected, atom_label=original[:, :101].argmax(-1),
                  identity=str(topology.normalized_canonical_smiles), strategy=config.strategy)
    m = len(data.bond_distance)
    if clean.geometry_valid:
        a = torch.as_tensor(np.array(static['token_pos_index_a'], copy=True)).long()
        b = torch.as_tensor(np.array(static['token_pos_index_b'], copy=True)).long()
        center = torch.zeros(len(positions), dtype=torch.bool)
        center[torch.as_tensor(trimer.mips_to_trimer_central_index).long()] = True
        # Center over physical atoms: coordinate denoising must not learn an
        # arbitrary lab-frame translation. Relative-vector decoding is equivariant.
        clean_centered = trimer.trimer_pos.float() - trimer.trimer_pos.float().mean(0)
        noisy_centered = positions - positions.mean(0)
        coordinate_target = clean_centered - noisy_centered
    else:
        a = b = torch.empty(0, dtype=torch.long)
        positions = torch.empty((0, 3), dtype=torch.float32)
        center = torch.empty(0, dtype=torch.bool)
        coordinate_target = positions.clone()
    labels.update(physical_pos=positions, physical_edges=torch.stack([a, b]), physical_center=center)
    if config.reconstruct_3d_mask:
        from src.modules.glt_dual import triplet_type
        token_selected = torch.zeros(m, dtype=torch.bool)
        token_action = torch.zeros(m, dtype=torch.long)
        token_donor = torch.arange(m)
        if clean.geometry_valid:
            offsets = torch.as_tensor(trimer.trimer_ru_offset)
            # All physical copies of a periodic line token share its corruption.
            groups = {}
            for i, (ca, cb) in enumerate(zip(data.bond_canonical_a.tolist(), data.bond_canonical_b.tolist())):
                shift = int(offsets[b[i]] - offsets[a[i]])
                key = min((ca, cb, shift), (cb, ca, -shift))
                groups.setdefault(key, []).append(i)
            members = list(groups.values())
            chosen, actions, donors = corruption(len(members), generator, config.mask_ratio)
            for g, rows in enumerate(members):
                token_selected[rows] = chosen[g]
                token_action[rows] = actions[g]
                token_donor[rows] = members[int(donors[g])][0]
        labels.update(token_selected=token_selected,
                      token_label=triplet_type(clean.bond_z_a, clean.bond_z_b, clean.bond_type),
                      token_action=token_action, token_donor=token_donor)
    if config.reconstruct_geometry:
        valid_angle = (clean.line_mask.sum(-1) == 1) & ~clean.line_is_self & clean.line_center_angle
        pairs = torch.sort(clean.line_path[valid_angle, :2], dim=-1).values
        # Unique undirected angles; target indices address bond rows, not atoms.
        unique = {}
        for pair, value in zip(pairs.tolist(), clean.line_angle[valid_angle, 0].cos()):
            unique.setdefault(tuple(pair), value)
        labels.update(coordinate_target=coordinate_target,
                      length_target=clean.bond_distance.clone(),
                      angle_pairs=torch.tensor(list(unique), dtype=torch.long).reshape(-1, 2),
                      angle_target=torch.stack(list(unique.values())) if unique else torch.empty(0))
    if config.fp_enabled:
        if fp_target is None:
            raise ValueError('FP enabled without clean cached target')
        labels['fp_target'] = torch.as_tensor(fp_target, dtype=torch.float32).clone()
    elif fp_target is not None:
        raise ValueError('FP target supplied to FP-off model')
    return data, labels


def collate_samples(samples):
    if not samples or len({y['strategy'] for _, y in samples}) != 1:
        raise ValueError('empty or mixed-strategy pretraining batch')
    if len({'fp_target' in labels for _, labels in samples}) != 1:
        raise ValueError('mixed FP target availability in batch')
    data = dual_glt_collate([x for x, _ in samples])
    mask_3d = samples[0][1]['strategy'] == 'mask_align'
    names = ['atom_selected', 'atom_label', 'physical_pos', 'physical_center']
    if mask_3d:
        names += ['token_selected', 'token_label', 'token_action']
    labels = {name: torch.cat([y[name] for _, y in samples]) for name in names}
    if 'fp_target' in samples[0][1]:
        labels['fp_target'] = torch.stack([y['fp_target'] for _, y in samples])
    labels['identity'] = [y['identity'] for _, y in samples]
    labels['strategy'] = samples[0][1]['strategy']
    donors, edges, graphs, pairs, angle_graph = [], [], [], [], []
    bond_offset = position_offset = 0
    reconstruct = 'coordinate_target' in samples[0][1]
    for graph, (x, y) in enumerate(samples):
        if mask_3d:
            donors.append(y['token_donor'] + bond_offset)
        edges.append(y['physical_edges'] + position_offset)
        graphs.append(torch.full((len(y['physical_pos']),), graph, dtype=torch.long))
        if reconstruct:
            pairs.append(y['angle_pairs'] + bond_offset)
            angle_graph.append(torch.full((len(y['angle_pairs']),), graph, dtype=torch.long))
        bond_offset += len(x.bond_distance)
        position_offset += len(y['physical_pos'])
    labels.update(physical_edges=torch.cat(edges, dim=1), physical_graph=torch.cat(graphs))
    if mask_3d:
        labels['token_donor'] = torch.cat(donors)
    if reconstruct:
        for name in ('coordinate_target', 'length_target', 'angle_target'):
            labels[name] = torch.cat([y[name] for _, y in samples])
        labels.update(angle_pairs=torch.cat(pairs), angle_graph=torch.cat(angle_graph))
    return data, labels
