"""Sample-key aligned Original-MIPS MD200 sidecar used by MTS-GLT-v3."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch


SCHEMA = "original-mips-md200-pi1m-v1"


class MD200Sidecar:
    def __init__(self, root):
        self.root = Path(root)
        self.metadata = json.loads((self.root / "metadata.json").read_text())
        if self.metadata.get("schema") != SCHEMA:
            raise ValueError("MD200 sidecar schema mismatch")
        self.keys = np.load(self.root / "sample_keys.npy", mmap_mode="r")
        self.values = np.load(self.root / "values.f32.npy", mmap_mode="r")
        self.valid = np.load(self.root / "valid.bool.npy", mmap_mode="r")
        if self.values.shape != (len(self.keys), 200) or self.valid.shape != (len(self.keys),):
            raise ValueError("MD200 sidecar shape mismatch")
        self._index = {bytes(value): i for i, value in enumerate(self.keys)}

    def get(self, key):
        index = self._index[bytes(key)]
        return torch.from_numpy(np.array(self.values[index], copy=True)), bool(self.valid[index])


class DatasetWithMD200(torch.utils.data.Dataset):
    def __init__(self, dataset, sidecar):
        self.dataset = dataset
        self.is_mts_route = bool(getattr(dataset, "is_mts_route", False))
        self.sidecar = sidecar if isinstance(sidecar, MD200Sidecar) else MD200Sidecar(sidecar)
        if len(dataset) != len(self.sidecar.keys):
            raise ValueError("dataset and MD200 sidecar counts differ")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data = self.dataset[index]
        if getattr(self.dataset, "_cohort_row_mode", False):
            key = bytes(self.dataset._cohort["row_keys_array"][int(index)])
        else:
            from .lmdb_cache import sample_key_from_smiles
            key = sample_key_from_smiles(str(data.smiles))
        data.mips_md, data.mips_md_valid = self.sidecar.get(key)
        return data


def write_md200_sidecar(root, sample_keys, values, valid, metadata=None):
    root = Path(root)
    if root.exists():
        raise FileExistsError(f"refusing to overwrite MD200 sidecar: {root}")
    root.mkdir(parents=True)
    np.save(root / "sample_keys.npy", np.asarray([np.frombuffer(bytes(key), dtype=np.uint8) for key in sample_keys], dtype=np.uint8))
    np.save(root / "values.f32.npy", np.asarray(values, dtype=np.float32))
    np.save(root / "valid.bool.npy", np.asarray(valid, dtype=bool))
    payload = {"schema": SCHEMA, "sample_count": len(sample_keys), "dimension": 200, **(metadata or {})}
    (root / "metadata.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    (root / ".done").write_text("complete\n")


__all__ = ["SCHEMA", "MD200Sidecar", "DatasetWithMD200", "write_md200_sidecar"]
