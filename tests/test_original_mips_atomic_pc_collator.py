from types import SimpleNamespace

import numpy as np
import pytest
import torch

import src.dataset.original_mips_atomic_pc as collator_module
from src.dataset.lmdb_cache import sample_key_from_smiles
from src.dataset.original_mips_atomic_pc import OriginalMIPSAtomicPCCollator
from src.dataset.original_mips_atomic_pc_joint import OriginalMIPSAtomicPCJointCollator


def _item(smiles, coords, atomic_number, ru_offset):
    return SimpleNamespace(
        smiles=smiles,
        sample_key=sample_key_from_smiles(smiles),
        trimer_pos=torch.tensor(coords, dtype=torch.float32),
        trimer_atomic_number=torch.tensor(atomic_number, dtype=torch.long),
        trimer_ru_offset=torch.tensor(ru_offset, dtype=torch.long),
        trimer_geometry_valid=True,
        trimer_geometry_is_3d=True,
        trimer_2d_fallback=False,
        graph_available=True,
    )


def _fake_mts_collate(items):
    counts = [int(item.trimer_pos.size(0)) for item in items]
    ptr = [0]
    for count in counts:
        ptr.append(ptr[-1] + count)
    return SimpleNamespace(
        trimer_pos=torch.cat([item.trimer_pos for item in items]),
        trimer_atomic_number=torch.cat([item.trimer_atomic_number for item in items]),
        trimer_ru_offset=torch.cat([item.trimer_ru_offset for item in items]),
        trimer_batch=torch.cat([
            torch.full((count,), index, dtype=torch.long)
            for index, count in enumerate(counts)
        ]),
        trimer_ptr=torch.tensor(ptr, dtype=torch.long),
        smiles=[item.smiles for item in items],
    )


@pytest.mark.parametrize(
    "collator_type", [OriginalMIPSAtomicPCCollator, OriginalMIPSAtomicPCJointCollator]
)
def test_finetune_and_pretrain_collators_pack_the_same_attached_mts_trimer(
    monkeypatch, collator_type
):
    monkeypatch.setattr(collator_module, "mips_trimer_collate", _fake_mts_collate)
    items = [
        _item("*CC*", [[0, 0, 0], [1, 0, 0]], [6, 6], [-1, 0]),
        _item("*CO*", [[2, 0, 0], [3, 0, 0], [4, 0, 0]], [6, 8, 6], [0, 1, 1]),
    ]
    md = {
        item.sample_key: np.full((200,), index, dtype=np.float32)
        for index, item in enumerate(items)
    }

    batch = collator_type(md)(items)
    packed = batch.atomic_point_cloud

    assert torch.equal(packed.coords, batch.trimer_pos)
    assert torch.equal(packed.atomic_number, batch.trimer_atomic_number)
    assert torch.equal(packed.ru_offset, batch.trimer_ru_offset)
    assert torch.equal(packed.batch, batch.trimer_batch)
    assert torch.equal(packed.ptr, batch.trimer_ptr)
    assert packed.sample_keys == tuple(item.sample_key.hex() for item in items)
    assert batch.original_mips_sample_keys == packed.sample_keys
    assert torch.equal(batch.mips_md[0], torch.zeros(200))
    assert torch.equal(batch.mips_md[1], torch.ones(200))


def test_collator_rejects_invalid_mts_geometry_before_batching(monkeypatch):
    item = _item("*CC*", [[0, 0, 0]], [6], [0])
    item.trimer_geometry_is_3d = False
    monkeypatch.setattr(
        collator_module,
        "mips_trimer_collate",
        lambda _: pytest.fail("invalid geometry must fail before batching"),
    )
    with pytest.raises(ValueError, match="invalid frozen MTS Trimer geometry"):
        OriginalMIPSAtomicPCCollator({item.sample_key: np.zeros(200)})([item])
