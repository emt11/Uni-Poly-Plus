"""Small, explicit downstream adaptation policies for GLT-PRED S1.

The helpers in this module deliberately leave the frozen deployment schema
untouched.  A deployment package is loaded first; LoRA modules are then
inserted for an adaptation run.  The default/full path therefore remains the
historical implementation.
"""

from __future__ import annotations

from typing import Iterable

import torch
from torch import nn


class LoRAQVMergedLinear(nn.Module):
    """A merged QKV projection with low-rank Q/V updates only.

    ``base`` is the original merged QKV layer and stays frozen.  B matrices
    are zero-initialized, so replacing a projection immediately after loading
    a deployment package is an exact identity.  The K slice has no update.
    """

    def __init__(self, base: nn.Linear, rank: int = 8, alpha: float = 8.0,
                 dropout: float = 0.0):
        super().__init__()
        if base.in_features != 512 or base.out_features != 1536:
            raise ValueError("GLT QKV projection must be 512 -> 1536")
        if int(rank) <= 0 or not torch.isfinite(torch.tensor(float(alpha))) or alpha <= 0:
            raise ValueError("invalid LoRA rank/alpha")
        self.base = base
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.dropout = nn.Dropout(float(dropout))
        self.q_a = nn.Parameter(torch.empty(self.rank, base.in_features))
        self.q_b = nn.Parameter(torch.zeros(512, self.rank))
        self.v_a = nn.Parameter(torch.empty(self.rank, base.in_features))
        self.v_b = nn.Parameter(torch.zeros(512, self.rank))
        nn.init.normal_(self.q_a, std=0.02)
        nn.init.normal_(self.v_a, std=0.02)
        for parameter in self.base.parameters():
            parameter.requires_grad = False

    def forward(self, inputs):
        output = self.base(inputs)
        dropped = self.dropout(inputs)
        q = (dropped @ self.q_a.t()) @ self.q_b.t() * self.scaling
        v = (dropped @ self.v_a.t()) @ self.v_b.t() * self.scaling
        delta = torch.cat((q, torch.zeros_like(q), v), dim=-1)
        return output + delta


def _replace_qkv(root: nn.Module, *, rank: int, alpha: float, dropout: float):
    replaced = []
    for module_name, module in list(root.named_modules()):
        if not module_name or not module_name.endswith("attention.qkv"):
            continue
        parent_name, _, child_name = module_name.rpartition(".")
        parent = root.get_submodule(parent_name)
        old = getattr(parent, child_name)
        if isinstance(old, LoRAQVMergedLinear):
            replaced.append(module_name)
            continue
        if not isinstance(old, nn.Linear):
            raise TypeError(f"{module_name} is not a merged Linear")
        replacement = LoRAQVMergedLinear(old, rank, alpha, dropout)
        # The deployment package is loaded after the encoder is moved to its
        # device.  New adapter parameters must follow the original projection
        # rather than defaulting to CPU.
        replacement = replacement.to(device=old.weight.device, dtype=old.weight.dtype)
        setattr(parent, child_name, replacement)
        replaced.append(module_name)
    if not replaced:
        raise ValueError("no GLT/O8 merged QKV projections found for LoRA")
    return replaced


def _set_requires_grad(module: nn.Module, enabled: bool):
    for parameter in module.parameters():
        parameter.requires_grad = bool(enabled)


def configure_adaptation(encoder: nn.Module, adaptation: str, *, rank: int = 8,
                         alpha: float = 8.0, dropout: float = 0.0):
    """Configure ``full``, ``head`` or ``lora`` in-place.

    Returns metadata used in the run record.  ``ridge`` is feature-only and
    intentionally does not mutate the encoder.
    """

    adaptation = str(adaptation).lower()
    if adaptation not in {"full", "head", "lora", "ridge"}:
        raise ValueError("adaptation must be full, head, lora or ridge")
    if adaptation == "full" or adaptation == "ridge":
        _set_requires_grad(encoder, adaptation == "full")
        return {"adaptation": adaptation, "lora_modules": []}

    if not hasattr(encoder, "predictor"):
        raise ValueError("dual encoder has no downstream predictor")
    _set_requires_grad(encoder, False)
    encoder.eval()
    _set_requires_grad(encoder.predictor, True)
    if adaptation == "head":
        encoder.predictor.train()
        return {"adaptation": adaptation, "lora_modules": []}

    # Load the historical deployment first, then replace QKV.  The original
    # base remains a child module and is frozen; only Q/V A/B and predictor
    # parameters are trainable.
    replaced = _replace_qkv(encoder, rank=rank, alpha=alpha, dropout=dropout)
    for name, module in encoder.named_modules():
        if isinstance(module, LoRAQVMergedLinear):
            module.eval()
            module.dropout.train()
            for parameter in (module.q_a, module.q_b, module.v_a, module.v_b):
                parameter.requires_grad = True
    encoder.predictor.train()
    return {"adaptation": adaptation, "lora_modules": replaced,
            "rank": int(rank), "alpha": float(alpha), "dropout": float(dropout)}


def trainable_parameters(module: nn.Module) -> list[tuple[str, nn.Parameter]]:
    return [(name, parameter) for name, parameter in module.named_parameters()
            if parameter.requires_grad]


class AdaptationAdapter(nn.Module):
    """Evaluation wrapper that preserves frozen-module eval mode on ``train``."""

    def __init__(self, encoder: nn.Module):
        super().__init__()
        self.encoder = encoder
        self._frozen = []

    def register_frozen(self, *modules: nn.Module):
        self._frozen = [module for module in modules if module is not None]
        return self

    def train(self, mode: bool = True):
        super().train(mode)
        for module in self._frozen:
            module.eval()
        return self

    def forward(self, data):
        return self.encoder(data), None
