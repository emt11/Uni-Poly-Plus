"""Dataset/collate bridge for the Original-MIPS + Atomic-PC route."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import torch

from .dataloader import mips_trimer_collate
from .lmdb_cache import sample_key_from_smiles
from src.modules.atomic_point_encoder import PackedAtomicPointCloud


class OriginalMIPSAtomicPCCollator:
    """Attach MD200 and the frozen MTS Trimer already present on each item.

    This is also the geometry path used by joint pretraining.  In particular,
    the collator must not replace these fields with the independent all-atom
    Random60 LMDB.  Both MD200 and geometry are joined by sample identity, not
    by an assumed row order.
    """

    def __init__(self, md_by_key: Mapping[bytes | str, Any]):
        self.md_by_key = {
            (bytes.fromhex(key) if isinstance(key, str) else bytes(key)): np.asarray(value)
            for key, value in md_by_key.items()
        }

    @staticmethod
    def _key(item: Any) -> bytes:
        key = getattr(item, "sample_key", None)
        if isinstance(key, bytes) and len(key) == 32:
            return key
        smiles = getattr(item, "smiles", None)
        if smiles is None:
            raise ValueError("A0 collator requires explicit sample_key or smiles")
        return sample_key_from_smiles(str(smiles))

    @staticmethod
    def _geometry_valid(item: Any) -> bool:
        return (
            bool(getattr(item, "trimer_geometry_valid", False))
            and bool(getattr(item, "trimer_geometry_is_3d", False))
            and not bool(getattr(item, "trimer_2d_fallback", False))
            and bool(getattr(item, "graph_available", True))
        )

    def __call__(self, data_list):
        if not data_list:
            raise ValueError("cannot collate an empty A0 batch")
        keys = [self._key(item) for item in data_list]
        missing_md = [key.hex() for key in keys if key not in self.md_by_key]
        if missing_md:
            raise KeyError(f"restored MD200 missing sample keys: {missing_md[:3]}")
        invalid_geometry = [
            key.hex() for key, item in zip(keys, data_list)
            if not self._geometry_valid(item)
        ]
        if invalid_geometry:
            raise ValueError(
                "Atomic-PC received invalid frozen MTS Trimer geometry; "
                f"pre-filter failures before DataLoader ({invalid_geometry[:3]})"
            )
        batch = mips_trimer_collate(data_list)
        values = []
        for key in keys:
            value = np.asarray(self.md_by_key[key])
            if value.shape != (200,) or not np.isfinite(value).all():
                raise ValueError(f"restored MD200 row is invalid for {key.hex()}")
            values.append(torch.as_tensor(value, dtype=torch.float32))
        batch.mips_md = torch.stack(values, dim=0)
        # The official path encodes None/NaN as zeros rather than dropping a
        # row.  Every explicit row therefore remains available to KFuse.
        batch.mips_md_valid = torch.ones(len(keys), dtype=torch.bool)
        batch.atomic_point_cloud = PackedAtomicPointCloud(
            coords=batch.trimer_pos.float(),
            atomic_number=batch.trimer_atomic_number.long(),
            ru_offset=batch.trimer_ru_offset.long(),
            batch=batch.trimer_batch.long(),
            ptr=batch.trimer_ptr.long(),
            sample_keys=tuple(key.hex() for key in keys),
            source_smiles=tuple(str(value) for value in batch.smiles),
        )
        batch.original_mips_sample_keys = tuple(key.hex() for key in keys)
        batch.original_mips_knowledge = {"md": batch.mips_md}
        return batch


__all__ = ["OriginalMIPSAtomicPCCollator"]
