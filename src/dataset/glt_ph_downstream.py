"""PH retention data layer: small downstream sidecar, group dataset and collate.

The sidecar keeps the ``glt-ph-betti-v2`` schema of the pretraining sidecar
(``sample_keys.npy`` / ``ph_profile.npy`` / ``ph_valid.npy`` / ``metadata.json``
/ ``.done``) so the already-validated key-addressed reader can be reused, but it
only covers the structures this round's folds actually use and is deduplicated
by the full 32-byte sample key across tasks.

A sample's PH input is decided by the retention group:

* ``F_OFF``   -- the profile is unused and the contribution is explicitly zero
* ``F_CONST`` -- every sample sees the fixed P_train mean profile, but keeps its
                 own validity flag, so the mask matches ``F_REAL`` exactly
* ``F_REAL``  -- the sample's own profile

All three groups use the full eight patches: downstream never masks PH.
"""
import numpy as np
import torch

from .glt_dual import dual_glt_collate
from .glt_galformer_ph import PHBettiReader, PH_PATCHES
from ..training.glt_dual_runtime import CleanLabeledDataset

GROUPS = ('F_OFF', 'F_CONST', 'F_REAL')
PH_CHANNELS = 3
PH_BINS = 32


def load_const_profile(path):
    """The P_train mean profile used by F_CONST (never a downstream statistic)."""
    profile = np.load(path)
    if profile.shape != (PH_CHANNELS, PH_BINS):
        raise ValueError('constant PH profile must be [3,32]')
    if not np.isfinite(profile).all():
        raise ValueError('constant PH profile contains non-finite values')
    return torch.as_tensor(profile, dtype=torch.float32)


class PHRetentionDataset(CleanLabeledDataset):
    """Downstream samples carrying one retention group's PH input."""

    def __init__(self, source, targets, *, group, reader=None, const_profile=None,
                 key_rows=None, cache_capacity_bytes=0):
        super().__init__(source, targets, cache_capacity_bytes=cache_capacity_bytes)
        if group not in GROUPS:
            raise ValueError(f'group must be one of {GROUPS}')
        if group != 'F_OFF' and reader is None:
            raise ValueError(f'{group} requires the downstream PH sidecar')
        if group == 'F_CONST' and const_profile is None:
            raise ValueError('F_CONST requires the P_train mean profile')
        self.group = group
        self.reader = reader
        self.key_rows = dict(key_rows or {})
        self.const_profile = (None if const_profile is None
                              else torch.as_tensor(const_profile, dtype=torch.float32))
        if self.const_profile is not None and self.const_profile.shape != (PH_CHANNELS, PH_BINS):
            raise ValueError('constant PH profile must be [3,32]')

    def _profile(self, key):
        if self.group == 'F_OFF':
            return torch.zeros(PH_CHANNELS, PH_BINS, dtype=torch.float32), False
        if self.group == 'F_CONST':
            _, valid = self.reader.get(self.key_rows.get(key.hex(), -1), key.hex())
            return self.const_profile.clone(), bool(valid)
        row = self.key_rows.get(key.hex(), -1)
        profile, valid = self.reader.get(row, key.hex())
        tensor = torch.as_tensor(np.asarray(profile, dtype=np.float32).copy())
        # A non-finite profile must never be silently multiplied by zero: the
        # validity flag is the only sanctioned way to drop a sample's PH input.
        if not torch.isfinite(tensor).all():
            raise ValueError(f'non-finite PH profile for sample key {key.hex()}')
        return tensor, bool(valid)

    def __getitem__(self, index):
        data = super().__getitem__(index)
        key = self.source.samples[int(index)][0]
        profile, valid = self._profile(key)
        data.ph_profile = profile
        data.ph_valid = torch.tensor(bool(valid))
        data.ph_mask = torch.zeros(PH_PATCHES, dtype=torch.bool)
        data.sample_key = key.hex()
        return data


def retention_collate(records):
    """``dual_glt_collate`` plus the stacked PH fields (one row per graph)."""
    batch = dual_glt_collate(records)
    batch.ph_profile = torch.stack([data.ph_profile for data in records], 0)
    batch.ph_mask = torch.stack([data.ph_mask for data in records], 0)
    batch.ph_valid = torch.stack([data.ph_valid for data in records], 0)
    batch.sample_keys = [data.sample_key for data in records]
    return batch


def open_sidecar(root):
    """Reuse the key-addressed reader of the pretraining sidecar."""
    return PHBettiReader(root)


def key_row_map(reader):
    """Full-key -> sidecar row, so reads take the reader's verified fast path."""
    keys = np.asarray(reader.keys)
    return {bytes(row).hex(): index for index, row in enumerate(keys)}


def fold_coverage(reader, keys, key_rows):
    """Deterministic coverage of one fold's structures (no dataset counters)."""
    valid = invalid = missing = 0
    for key in keys:
        row = key_rows.get(key)
        if row is None:
            missing += 1
            continue
        _, ok = reader.get(row, key)
        valid += int(bool(ok))
        invalid += int(not bool(ok))
    return {'samples': len(keys), 'valid': valid, 'invalid': invalid, 'missing': missing}
