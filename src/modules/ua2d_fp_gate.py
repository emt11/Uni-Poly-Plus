"""Small learnable scalar gate for the original UA 2D graph representation."""
import math
import torch
from torch import nn


class FPGate(nn.Module):
    def __init__(self):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.default_generator.manual_seed(20261017)
            self.u = nn.Linear(2048, 32)
            self.v = nn.Linear(32, 512)
            self.gate = nn.Linear(544, 1)
            nn.init.zeros_(self.v.weight)
            nn.init.zeros_(self.v.bias)
            nn.init.zeros_(self.gate.weight)
            nn.init.constant_(self.gate.bias, math.log(.1 / .9))

    def components(self, a, x):
        if a.ndim != 2 or a.shape[1] != 512 or x.shape != (len(a), 2048):
            raise ValueError('FP/gate input shape mismatch')
        if not torch.isfinite(a).all() or not torch.isfinite(x).all():
            raise FloatingPointError('nonfinite FP/gate input')
        norm = torch.linalg.vector_norm(x, dim=1, keepdim=True)
        if (norm <= 0).any():
            raise ValueError('zero FP input')
        p = math.sqrt(512) * x / norm
        q = torch.nn.functional.gelu(self.u(p))
        delta = self.v(q)
        g = torch.sigmoid(self.gate(torch.cat((a, q), dim=1)))
        result = a + g * delta
        if not torch.isfinite(result).all() or not torch.isfinite(g).all():
            raise FloatingPointError('nonfinite gated features')
        return result, g, delta

    def forward(self, a, x):
        return self.components(a, x)[0]


class GatedHead(nn.Module):
    def __init__(self, regressor, *, constant=False):
        super().__init__()
        self.fusion = FPGate()
        self.regressor = regressor
        self.constant = constant

    def features(self, a, bits):
        x = torch.full_like(bits, .5) if self.constant else bits
        return self.fusion(a, x)

    def forward(self, a, bits):
        return self.regressor(self.features(a, bits))
