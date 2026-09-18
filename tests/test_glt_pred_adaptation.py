import torch
from torch import nn

from src.modules.glt_adaptation import LoRAQVMergedLinear, configure_adaptation


class _Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(512, 1536)


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = _Attention()


class _ToyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.o8 = nn.ModuleList([_Block()])
        self.glt = nn.ModuleList([_Block()])
        self.norm2 = nn.LayerNorm(512)
        self.norm3 = nn.LayerNorm(512)
        self.predictor = nn.Linear(512, 1)


def test_lora_qv_is_identity_and_leaves_k_unchanged():
    torch.manual_seed(3)
    base = nn.Linear(512, 1536)
    layer = LoRAQVMergedLinear(base)
    values = torch.randn(4, 512)
    torch.testing.assert_close(layer(values), base(values))
    assert not layer.base.weight.requires_grad
    assert not layer.base.bias.requires_grad
    assert layer.q_a.requires_grad and layer.q_b.requires_grad
    assert layer.v_a.requires_grad and layer.v_b.requires_grad
    with torch.no_grad():
        layer.q_b.normal_()
        layer.v_b.normal_()
    before = layer(values)
    layer.q_b.grad = layer.v_b.grad = None
    before[:, 512:1024].sum().backward()
    assert torch.count_nonzero(layer.q_b.grad) == 0
    assert torch.count_nonzero(layer.v_b.grad) == 0


def test_head_freezes_encoder_and_lora_only_qv_plus_predictor_train():
    head = _ToyEncoder()
    meta = configure_adaptation(head, 'head')
    assert meta['adaptation'] == 'head'
    assert all(not p.requires_grad for name, p in head.named_parameters()
               if not name.startswith('predictor.'))
    assert all(p.requires_grad for p in head.predictor.parameters())

    lora = _ToyEncoder()
    meta = configure_adaptation(lora, 'lora')
    assert len(meta['lora_modules']) == 2
    for name, parameter in lora.named_parameters():
        if any(token in name for token in ('q_a', 'q_b', 'v_a', 'v_b', 'predictor')):
            assert parameter.requires_grad, name
        else:
            assert not parameter.requires_grad, name
