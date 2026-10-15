"""Numerical and isolation checks for the graph-local XATTN optimization."""
import copy

import pytest
import torch

from src.modules.mcl_ph import FusionCrossAttention, FUSION_SCALE


def reference(module, hidden, tokens, graph):
    n = len(hidden)
    q = module.query(module.norm2(hidden)).reshape(n, module.heads, module.dim)
    k = module.key(module.norm3(tokens)).reshape(n, module.heads, module.dim)
    v = module.value(module.norm3(tokens)).reshape(n, module.heads, module.dim)
    score = torch.einsum('ihd,jhd->hij', q.float(), k.float()) * module.scale
    same = graph[:, None] == graph[None, :]
    score = score.masked_fill(~same[None], float('-inf'))
    attended = torch.einsum('hij,jhd->ihd', score.softmax(-1), v.float())
    return hidden + FUSION_SCALE * module.out(attended.to(hidden.dtype).reshape(n, -1))


@pytest.mark.parametrize('sizes', [(1, 3, 9), (17, 1, 2)])
def test_forward_and_all_gradients_match_reference(sizes):
    torch.manual_seed(41)
    left = FusionCrossAttention(hidden=16, token=8, heads=2, dim=4, chunk=3)
    right = copy.deepcopy(left)
    graph = torch.cat([torch.full((size,), i * 7 + 2) for i, size in enumerate(sizes)])
    permutation = torch.randperm(len(graph))
    graph = graph[permutation]
    x = torch.randn(len(graph), 16, requires_grad=True)
    t = torch.randn(len(graph), 8, requires_grad=True)
    xx, tt = x.detach().clone().requires_grad_(), t.detach().clone().requires_grad_()
    actual, expected = left(x, t, graph), reference(right, xx, tt, graph)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
    probe = torch.randn_like(actual)
    (actual * probe).sum().backward()
    (expected * probe).sum().backward()
    for a, b in [(x.grad, xx.grad), (t.grad, tt.grad)]:
        torch.testing.assert_close(a, b, rtol=3e-5, atol=2e-6)
    for (name, a), (_, b) in zip(left.named_parameters(), right.named_parameters()):
        torch.testing.assert_close(a.grad, b.grad, rtol=3e-5, atol=2e-6, msg=name)


def test_graph_isolation_empty_and_learnable_residual():
    torch.manual_seed(42)
    module = FusionCrossAttention(hidden=16, token=8, heads=2, dim=4)
    x, t = torch.randn(7, 16), torch.randn(7, 8)
    graph = torch.tensor([2, 5, 2, 5, 2, 5, 5])
    original = module(x, t, graph)
    changed = t.clone()
    changed[graph == 5] += torch.randn_like(changed[graph == 5]) * 10
    torch.testing.assert_close(module(x, changed, graph)[graph == 2], original[graph == 2])
    assert module(x[:0], t[:0], graph[:0]).shape == (0, 16)
    assert 'residual_scale' not in module.state_dict()
    module.enable_learnable_scale()
    torch.testing.assert_close(module(x, t, graph), original)
    module(x, t, graph).square().sum().backward()
    assert module.residual_scale.grad is not None
    assert torch.isfinite(module.residual_scale.grad)
    with pytest.raises(ValueError):
        module.enable_learnable_scale()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_cuda_forward_and_gradients_match_reference(dtype):
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip('CUDA BF16 unavailable')
    torch.manual_seed(43)
    left = FusionCrossAttention().cuda()
    right = copy.deepcopy(left)
    graph = torch.cat([torch.full((n,), i) for i, n in enumerate((1, 3, 17))]).cuda()
    graph = graph[torch.randperm(len(graph), device='cuda')]
    x = torch.randn(len(graph), 512, device='cuda', requires_grad=True)
    t = torch.randn(len(graph), 128, device='cuda', requires_grad=True)
    xx, tt = x.detach().clone().requires_grad_(), t.detach().clone().requires_grad_()
    with torch.autocast('cuda', dtype=dtype, enabled=dtype != torch.float32):
        actual, expected = left(x, t, graph), reference(right, xx, tt, graph)
    def compare(a, b, name):
        if dtype == torch.float32:
            torch.testing.assert_close(a, b, rtol=3e-5, atol=3e-6, msg=name)
            return
        # Padding changes BF16 GEMM reduction order. Bound both aggregate and
        # peak error by two BF16 ulps at the reference tensor scale; an
        # elementwise relative check alone is undefined near gradient zeros.
        epsilon = torch.finfo(torch.bfloat16).eps
        delta = (a.float() - b.float()).abs()
        norm_error = (delta.norm() / b.float().norm().clamp_min(1e-12)).item()
        peak_error = (delta.max() / b.float().abs().max().clamp_min(1e-12)).item()
        print(f'BF16 {name}: relative_l2={norm_error:.8g} relative_peak={peak_error:.8g}')
        assert torch.isfinite(a).all(), name
        assert norm_error <= 2 * epsilon, name
        assert peak_error <= 2 * epsilon, name
    compare(actual, expected, 'forward')
    probe = torch.randn_like(actual)
    (actual * probe).sum().backward()
    (expected * probe).sum().backward()
    for name, a, b in [('hidden', x.grad, xx.grad), ('tokens', t.grad, tt.grad)]:
        compare(a, b, name)
    for (name, a), (_, b) in zip(left.named_parameters(), right.named_parameters()):
        compare(a.grad, b.grad, name)
