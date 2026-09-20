"""GLT-GALPH r2 data path: native-token masking, PH sidecar and patch masking.

Masking follows the plan exactly: 40% candidate rate per graph with an
80/10/10 MASK/REPLACE/KEEP split, drawn from independent deterministic streams
``(seed, sample_key, absolute_position, 'mask2d'|'mask3d'|'maskph')`` so the 2D,
3D and PH masks never interact.  REPLACE donors come from the same graph (the
plan's mini-batch pool restricted to a deterministic, connectivity-preserving
choice).  PH profiles come from the mmap'd Betti sidecar and are key-checked
against the manifest position.
"""
import numpy as np
import torch
from torch_geometric.data import Data

from .glt_dual import build_dual_sample, dual_glt_collate
from .glt_dual_pretrain import sample_generator

CANDIDATE_RATE = 0.40
KEEP_RATIO = 0.10
REPLACE_RATIO = 0.10
PH_PATCHES = 8
PH_PATCHES_MASKED = 2


class PHBettiReader:
    """Read-only mmap access to the PH Betti sidecar, key-validated."""

    def __init__(self, root, verify_keys=True):
        from pathlib import Path
        root = Path(root)
        if not (root / '.done').is_file():
            raise FileNotFoundError(f'PH sidecar is incomplete: {root}')
        self.profiles = np.load(root / 'ph_profile.npy', mmap_mode='r')
        self.valid = np.load(root / 'ph_valid.npy', mmap_mode='r')
        self.keys = np.load(root / 'sample_keys.npy', mmap_mode='r')
        self.verify_keys = bool(verify_keys)

    def __len__(self):
        return int(self.profiles.shape[0])

    def get(self, position, key):
        position = int(position)
        if position >= len(self):
            raise IndexError('PH sidecar is shorter than the manifest')
        if self.verify_keys:
            expected = np.frombuffer(bytes.fromhex(key), dtype=np.uint8)[:32]
            if not np.array_equal(np.asarray(self.keys[position]), expected):
                raise ValueError('PH sidecar sample key does not match the manifest order')
        return (np.asarray(self.profiles[position], dtype=np.float32),
                bool(self.valid[position]))


def _choose(candidates, generator, rate=CANDIDATE_RATE):
    if candidates <= 0:
        return torch.zeros(0, dtype=torch.long)
    count = max(1, int(np.floor(rate * candidates)))
    count = min(count, candidates)
    return torch.randperm(candidates, generator=generator)[:count]


def _policies(rows, generator, keep_ratio=KEEP_RATIO, replace_ratio=REPLACE_RATIO):
    """0 KEEP / 1 MASK / 2 REPLACE per selected row (80/10/10 by default)."""
    total = int(rows.numel())
    if total == 0:
        return torch.zeros(0, dtype=torch.long), torch.zeros(0, dtype=torch.long)
    draw = torch.rand(total, generator=generator)
    mask_ratio = 1.0 - keep_ratio - replace_ratio          # 0.80 by default
    replace = draw < replace_ratio
    mask = (draw >= replace_ratio) & (draw < replace_ratio + mask_ratio)
    policy = torch.zeros(total, dtype=torch.long)
    policy[mask] = 1
    policy[replace] = 2
    replace_indices = torch.where(policy == 2)[0]
    return policy, replace_indices


def _donors(selected, replace_positions, pool_size, generator):
    """One same-graph donor token per REPLACE row, excluding the selected set."""
    if replace_positions.numel() == 0:
        return torch.zeros(0, dtype=torch.long)
    selected_set = set(int(value) for value in selected.tolist())
    pool = torch.tensor([index for index in range(pool_size) if index not in selected_set],
                        dtype=torch.long)
    if pool.numel() == 0:
        pool = selected.clone()
    picks = torch.randint(pool.numel(), (int(replace_positions.numel()),), generator=generator)
    return pool[picks]


def mask_2d(data, generator):
    """Single-graph masking (one sample per call; collate adds node offsets)."""
    atoms = int(data.mips_x.size(0))
    selected = _choose(atoms, generator)
    policy, replace_positions = _policies(selected, generator)
    donor = _donors(selected, replace_positions, atoms, generator)
    data.mask2d_rows = selected
    data.mask2d_policy = policy
    data.mask2d_donor_atoms = donor
    return data


def mask_3d(data, generator):
    """Mask the full physical bond set of one sample (not center-only)."""
    bonds = int(data.bond_distance.numel())
    selected = _choose(bonds, generator)
    policy, replace_positions = _policies(selected, generator)
    donor = _donors(selected, replace_positions, bonds, generator)
    data.mask3d_rows = selected
    data.mask3d_policy = policy
    data.mask3d_donor_atoms = donor
    return data


def ph_fields(profile, valid, generator):
    """Per-graph PH profile plus the 2/8 masked patch choice."""
    profile = torch.as_tensor(profile, dtype=torch.float32)
    masked = torch.randperm(PH_PATCHES, generator=generator)[:PH_PATCHES_MASKED]
    mask = torch.zeros(PH_PATCHES, dtype=torch.bool)
    mask[masked] = True
    return profile, mask, bool(valid)


def prepare_galformer_sample(topology, trimer, smiles, *, static, seed, key, position,
                             ph_reader=None):
    """Clean-geometry Galformer sample: O8/bond fields plus mask and PH fields."""
    data = build_dual_sample(topology, trimer, smiles, static=static)
    generator_2d = sample_generator(seed, f'{key}:mask2d', position)
    generator_3d = sample_generator(seed, f'{key}:mask3d', position)
    data = mask_2d(data, generator_2d)
    data = mask_3d(data, generator_3d)

    label_2d = topology.mips_x[:, :101].argmax(-1).long()[data.mask2d_rows]
    z_a, z_b = data.bond_z_a.long(), data.bond_z_b.long()
    from ..modules.glt_dual import triplet_type
    line_class = triplet_type(z_a, z_b, data.bond_type.long())[data.mask3d_rows]

    labels = {'label_2d': label_2d, 'label_3d': line_class,
              'identity': str(key), 'position': int(position)}
    if ph_reader is not None:
        generator_ph = sample_generator(seed, f'{key}:maskph', position)
        profile, valid = ph_reader.get(position, key)
        profile_t, patch_mask, ph_valid = ph_fields(profile, valid, generator_ph)
        data.ph_profile = profile_t            # [3,32]
        data.ph_mask = patch_mask              # [8]
        data.ph_valid = torch.tensor(ph_valid)  # scalar
        # Targets: only the masked patches, flattened to 24 values.
        patches = profile_t.reshape(3, PH_PATCHES, 4)
        targets = patches[:, patch_mask, :].reshape(-1)
        labels['label_ph'] = targets
        labels['ph_patch_mask'] = patch_mask
    return data, labels


def galformer_collate(records):
    """Collate Galformer samples, reusing the existing O8/bond collate."""
    samples = [item for item, _ in records]
    batch = dual_glt_collate(samples)
    atom_offsets, bond_offsets = [], []
    atom_offset = bond_offset = 0
    for item in samples:
        atom_offsets.append(atom_offset)
        bond_offsets.append(bond_offset)
        atom_offset += int(item.mips_x.size(0))
        bond_offset += int(item.bond_distance.numel())
    rows2, policy2, donors2 = [], [], []
    rows3, policy3, donors3 = [], [], []
    labels_2d, labels_3d, identities = [], [], []
    ph_profiles, ph_masks, ph_valids, labels_ph = [], [], [], []
    for item, (_, label) in zip(samples, records):
        offset = atom_offsets[len(labels_2d)]
        rows2.append(item.mask2d_rows + offset)
        policy2.append(item.mask2d_policy)
        donors2.append(item.mask2d_donor_atoms + offset)
        bond_offset = bond_offsets[len(labels_3d)]
        rows3.append(item.mask3d_rows + bond_offset)
        policy3.append(item.mask3d_policy)
        donors3.append(item.mask3d_donor_atoms + bond_offset)
        labels_2d.append(label['label_2d'])
        labels_3d.append(label['label_3d'])
        identities.append(label['identity'])
        if 'label_ph' in label:
            ph_profiles.append(item.ph_profile)
            ph_masks.append(item.ph_mask)
            ph_valids.append(item.ph_valid)
            labels_ph.append(label['label_ph'])
    batch.mask2d_rows = torch.cat(rows2) if rows2 else torch.zeros(0, dtype=torch.long)
    batch.mask2d_policy = torch.cat(policy2) if policy2 else torch.zeros(0, dtype=torch.long)
    batch.mask2d_donor_atoms = torch.cat(donors2) if donors2 else torch.zeros(0, dtype=torch.long)
    batch.mask3d_rows = torch.cat(rows3) if rows3 else torch.zeros(0, dtype=torch.long)
    batch.mask3d_policy = torch.cat(policy3) if policy3 else torch.zeros(0, dtype=torch.long)
    batch.mask3d_donor_atoms = torch.cat(donors3) if donors3 else torch.zeros(0, dtype=torch.long)
    if ph_profiles:
        batch.ph_profile = torch.stack(ph_profiles, 0)   # [B,3,32]
        batch.ph_mask = torch.stack(ph_masks, 0)         # [B,8]
        batch.ph_valid = torch.stack(ph_valids, 0)       # [B]
    labels = {'label_2d': torch.cat(labels_2d), 'label_3d': torch.cat(labels_3d),
              'identity': identities}
    if labels_ph:
        labels['label_ph'] = torch.cat(labels_ph)
        labels['ph_patch_mask'] = torch.stack(ph_masks, 0)
    return batch, labels
