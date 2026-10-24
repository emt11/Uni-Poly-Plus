"""Targeted CPU checks for the V1 attention-scale change.

Covers the three questions that the single-factor experiment depends on:
1. An independent numeric reference proves the real attention applies
   64**-0.5 to the QK dot product and adds the bias after scaling.
2. The scale travels through PT export -> downstream restore -> frozen unit
   records without being defaulted away.
3. A legacy package without the new field still restores the old 512 scale,
   while a package that claims the new scale but omits the record is rejected.
"""
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.modules.glt_dual import BondPathO8, GalformerTrimer3D, Pure2DGLTModel, SourceAttention512
from src.modules.ua2d_galformer_pretrain import (
    PretrainConfig, UA2DGalformerPretrainer, attention_record, check_attention_record,
    deployment_package, package_attention, restore_downstream)


def reference_attention(packed, source, target, bias, scale):
    """Independent dense reference: softmax(dot(q,k)*scale + bias) @ v."""
    q, k, v = packed.reshape(-1, 3, 8, 64).unbind(1)
    q, k, v, bias = q.double(), k.double(), v.double(), bias.double()
    scores = (q[source] * k[target]).sum(-1) * scale + bias
    out = torch.zeros(q.size(0), 8, 64, dtype=torch.float64)
    for node in target.unique():
        mask = target == node
        weight = torch.softmax(scores[mask], dim=0)
        message = (weight.unsqueeze(-1) * v[source[mask]]).sum(0, keepdim=True)
        out.index_add_(0, node.view(1), message)
    return out.reshape(-1, 512)


def synthetic_inputs(nodes=7, edges=12, seed=0):
    generator = torch.Generator().manual_seed(seed)
    states = torch.randn(nodes, 512, generator=generator)
    source = torch.tensor([0, 1, 2, 3, 4, 5, 0, 2, 4, 6, 1, 3])
    target = torch.tensor([1, 2, 3, 4, 5, 6, 2, 4, 6, 0, 3, 5])
    bias = torch.randn(edges, 8, generator=generator)
    return states, source, target, bias


@pytest.mark.parametrize('denominator', [64, 512])
def test_numeric_scale_and_bias_position(denominator):
    states, source, target, bias = synthetic_inputs()
    module = SourceAttention512(dropout=0.0, scale_denominator=denominator).eval()
    with torch.no_grad():
        packed = module.qkv(states)
        got = module(states, source, target, bias)
    scale = denominator ** -0.5
    want = reference_attention(packed, source, target, bias, scale)
    assert torch.allclose(got.double(), want, atol=2e-5)
    # (dot + bias) * scale is a different function; the module must not use it.
    wrong = reference_attention(packed, source, target, bias * scale, scale)
    assert not torch.allclose(got.double(), wrong, atol=1e-3)


def test_scale_changes_the_scores_only():
    states, source, target, bias = synthetic_inputs(seed=3)
    narrow = SourceAttention512(dropout=0.0, scale_denominator=64).eval()
    wide = SourceAttention512(dropout=0.0, scale_denominator=512).eval()
    wide.load_state_dict(narrow.state_dict())
    with torch.no_grad():
        packed = narrow.qkv(states)
        out64 = narrow(states, source, target, bias)
        out512 = wide(states, source, target, bias)
    assert narrow.scale == 0.125 and wide.scale == 512 ** -0.5
    assert torch.allclose(out64.double(), reference_attention(packed, source, target, bias, 0.125), atol=2e-5)
    assert torch.allclose(out512.double(), reference_attention(packed, source, target, bias, 512 ** -0.5), atol=2e-5)
    assert not torch.allclose(out64, out512, atol=1e-5)


def test_invalid_denominator_is_rejected():
    with pytest.raises(ValueError):
        SourceAttention512(scale_denominator=0)
    with pytest.raises(ValueError):
        SourceAttention512(scale_denominator='64')
    with pytest.raises(ValueError):
        PretrainConfig(attention_scale_denominator=128)
    with pytest.raises(ValueError):
        PretrainConfig(attention_scale_denominator=True)
    assert PretrainConfig(attention_scale_denominator=64).attention_scale == 0.125
    assert PretrainConfig().attention_scale == 512 ** -0.5


def test_every_block_uses_the_configured_scale():
    torch.manual_seed(0)
    o8 = BondPathO8(0.1, attention_scale_denominator=64)
    assert [layer.attention.scale for layer in o8.layers] == [0.125] * 6
    encoder = GalformerTrimer3D(0.1, attention_scale_denominator=64)
    assert [layer.attention.scale for layer in encoder.layers] == [0.125] * 6
    pure = Pure2DGLTModel(attention_scale_denominator=64)
    assert pure.attention_scale == 0.125 and pure.attention_scale_denominator == 64
    assert [layer.attention.scale for layer in pure.o8.layers] == [0.125] * 6
    legacy = Pure2DGLTModel()
    assert legacy.attention_scale == 512 ** -0.5
    assert [layer.attention.scale for layer in legacy.o8.layers] == [512 ** -0.5] * 6


def build_package(denominator):
    config = PretrainConfig(attention_scale_denominator=denominator)
    torch.manual_seed(7)
    model = UA2DGalformerPretrainer(config).eval()
    package = deployment_package(model, 5000)
    head = Pure2DGLTModel()
    head_state = {key[len('head.'):]: value for key, value in head.state_dict().items() if key.startswith('head.')}
    return package, head_state


def test_export_and_restore_keep_the_new_scale():
    package, head_state = build_package(64)
    assert package['attention'] == attention_record(PretrainConfig(attention_scale_denominator=64))
    assert package['config']['attention_scale_denominator'] == 64
    assert package_attention(package)['scale'] == 0.125
    restored = restore_downstream(package, head_state)
    assert restored.attention_scale == 0.125
    assert restored.attention_scale_denominator == 64
    assert [layer.attention.scale for layer in restored.o8.layers] == [0.125] * 6
    for key, value in restored.o8.state_dict().items():
        assert torch.equal(value, package['state_dict']['o8.' + key])


def test_legacy_package_keeps_512_and_conflicts_raise():
    package, head_state = build_package(512)
    legacy = dict(package, config=dict(package['config']))
    legacy.pop('attention')
    legacy['config'].pop('attention_scale_denominator')
    assert package_attention(legacy)['scale'] == 512 ** -0.5
    restored = restore_downstream(legacy, head_state)
    assert restored.attention_scale == 512 ** -0.5
    assert package_attention(package)['scale'] == 512 ** -0.5
    new_package, _ = build_package(64)
    missing = dict(new_package)
    missing.pop('attention')
    with pytest.raises(ValueError):
        package_attention(missing)
    conflicted = dict(new_package, attention=dict(new_package['attention'], scale=0.25))
    with pytest.raises(ValueError):
        package_attention(conflicted)


def test_record_checks_cover_pt_freeze_and_formal_restore():
    expected = attention_record(PretrainConfig(attention_scale_denominator=64))
    check_attention_record(expected, expected, 'matching')
    with pytest.raises(ValueError):
        check_attention_record(None, expected, 'missing')
    with pytest.raises(ValueError):
        check_attention_record(dict(expected, scale_denominator=512, scale=512 ** -0.5), expected, 'conflict')
    legacy = attention_record(PretrainConfig())
    check_attention_record(None, legacy, 'legacy')
