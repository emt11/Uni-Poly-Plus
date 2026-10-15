"""Clean feature reuse must preserve inputs and fresh labels after mutation."""
import numpy as np
import pytest
import torch
from torch_geometric.data import Data
from scripts import finetune_mcl_ph as runner


@pytest.mark.parametrize('mcl', [False, True])
def test_clean_cache_survives_collate_mutation_and_target_override(monkeypatch, mcl):
    calls = []
    class Source:
        samples = [(b'x' * 32, '*CC*')]
        def __len__(self): return 1
        def pretrain_fields_for(self, index, *, include_target):
            calls.append(index)
            return None, None, '*CC*', {}, None
    def build(*args, **kwargs):
        data = Data(x=torch.tensor([[1., 2.]]), edge_index=torch.empty(2, 0, dtype=torch.long))
        return (data, {'distance': torch.tensor([3.])}) if mcl else data
    if mcl:
        monkeypatch.setattr(runner, 'prepare_mcl_ph_sample', build)
        dataset = runner.MCLPHCleanDataset(Source(), [7.], None)
    else:
        dataset = runner.CleanDualDataset(Source(), [7.])
        dataset._build = build
    first = dataset[0]
    data = first[0] if mcl else first
    assert data.y.item() == 7.
    data.x.fill_(999.)
    data.y.fill_(999.)
    if mcl: first[1]['distance'].fill_(999.)
    dataset.set_target_override([11.])
    second = dataset[0]
    other = second[0] if mcl else second
    assert torch.equal(other.x, torch.tensor([[1., 2.]]))
    assert other.y.item() == 11.
    if mcl: assert second[1]['distance'].item() == 3.
    assert calls == [0]
    assert dataset.raw_targets.tolist() == [7.]
    with pytest.raises(ValueError): dataset.set_target_override([1., 2.])
