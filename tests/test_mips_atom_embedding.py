"""局部行为约束：MIPS 137 维属性加 backbone 标记的 138 维输入。"""

from types import SimpleNamespace

import pytest
import torch

from src.dataset.graph_data import MIPS_ATOM_FEATURE_DIM
from src.modules.mips_local_graph import (
    MIPS_ATOM_INPUT_DIM,
    MIPSLocalAtomEmbedding,
    MIPSLocalGraphEncoder,
)


def _data(rows=3):
    return SimpleNamespace(
        mips_x=torch.arange(rows * MIPS_ATOM_FEATURE_DIM, dtype=torch.float32).reshape(
            rows, MIPS_ATOM_FEATURE_DIM
        ),
        mips_backbone_mask=torch.tensor([0, 1, 0][:rows], dtype=torch.long),
    )


def test_input_is_138d_with_backbone_as_last_column():
    data = _data()
    embedding = MIPSLocalAtomEmbedding(hidden_dim=4)

    assert MIPS_ATOM_FEATURE_DIM == 137
    assert MIPS_ATOM_INPUT_DIM == 138
    assert embedding.projection.in_features == 138
    assert not hasattr(embedding, "backbone")

    with torch.no_grad():
        embedding.projection.weight.zero_()
        embedding.projection.weight[:, -1] = 1.0
        embedding.projection.bias.zero_()
    output = embedding(data)
    torch.testing.assert_close(output[:, 0], data.mips_backbone_mask.float())


def test_mask_zeros_all_138_columns_and_returns_projection_bias():
    data = _data()
    embedding = MIPSLocalAtomEmbedding(hidden_dim=4)
    atom_mask = torch.tensor([True, True, False])

    with torch.no_grad():
        embedding.projection.weight.fill_(1.0)
        embedding.projection.bias.copy_(torch.tensor([1.0, 2.0, 3.0, 4.0]))
    output = embedding(data, atom_mask=atom_mask)

    expected_bias = embedding.projection.bias.detach().expand(2, -1)
    torch.testing.assert_close(output[atom_mask], expected_bias)
    assert torch.equal(data.mips_x, _data().mips_x)
    assert torch.equal(data.mips_backbone_mask, _data().mips_backbone_mask)


def test_unmasked_rows_keep_chemical_and_backbone_inputs():
    data = _data()
    original_x = data.mips_x.clone()
    original_backbone = data.mips_backbone_mask.clone()
    embedding = MIPSLocalAtomEmbedding(hidden_dim=4)
    atom_mask = torch.tensor([True, False, True])

    with torch.no_grad():
        embedding.projection.weight.copy_(
            torch.arange(4 * 138, dtype=torch.float32).reshape(4, 138)
        )
        embedding.projection.bias.zero_()
    output = embedding(data, atom_mask=atom_mask)
    concatenated = torch.cat(
        [original_x, original_backbone.float().unsqueeze(-1)], dim=-1
    )
    expected = embedding.projection(concatenated)
    torch.testing.assert_close(output[~atom_mask], expected[~atom_mask])
    assert torch.equal(data.mips_x, original_x)
    assert torch.equal(data.mips_backbone_mask, original_backbone)


def test_backbone_column_receives_gradient_and_atom_target_stays_101():
    data = _data()
    embedding = MIPSLocalAtomEmbedding(hidden_dim=4)
    output = embedding(data).sum()
    output.backward()

    assert embedding.projection.weight.grad is not None
    assert torch.isfinite(embedding.projection.weight.grad[:, -1]).all()
    assert bool((embedding.projection.weight.grad[:, -1].abs() > 0).any())

    encoder = MIPSLocalGraphEncoder()
    assert encoder.masked_atom_classes == 101


def test_new_projection_state_round_trips_strictly(tmp_path):
    source = MIPSLocalAtomEmbedding(hidden_dim=4)
    path = tmp_path / "mips_atom_embedding_138.pt"
    torch.save(source.state_dict(), path)

    restored = MIPSLocalAtomEmbedding(hidden_dim=4)
    restored.load_state_dict(torch.load(path, weights_only=True), strict=True)
    for key, value in source.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[key], value)


def test_old_137d_projection_cannot_load_into_new_model_strictly():
    embedding = MIPSLocalAtomEmbedding(hidden_dim=4)
    old_state = {
        "projection.weight": torch.zeros(4, MIPS_ATOM_FEATURE_DIM),
        "projection.bias": torch.zeros(4),
    }
    with pytest.raises(RuntimeError):
        embedding.load_state_dict(old_state, strict=True)
