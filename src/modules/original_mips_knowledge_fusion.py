"""Isolated, parity-preserving Original-MIPS attentive KFuse."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import nn


class OriginalMIPSAttentiveFusion(nn.Module):
    """The official ``AtttentiveFusion`` equations without a DGL dependency.

    ``batch`` is the node-to-graph index that replaces DGL's
    ``broadcast_nodes`` operation.  Parameter names intentionally match the
    official module (``k_proj``, ``v_proj`` and ``Wq``) so parity scripts can
    copy state tensors by name.
    """

    architecture_name = "Original-MIPS-AtttentiveFusion"

    def __init__(
        self,
        d_model: int = 512,
        knodes: Sequence[str] = ("md", "atomic_pc"),
        knowledge_dims: Mapping[str, int] | None = None,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.knodes = tuple(str(name) for name in knodes)
        if self.d_model != 512:
            raise ValueError("Original MIPS KFuse requires d_model=512")
        if not self.knodes:
            raise ValueError("Original MIPS KFuse requires at least one modality")
        dims = {"md": 200, "atomic_pc": 512}
        if knowledge_dims is not None:
            dims.update({str(key): int(value) for key, value in knowledge_dims.items()})
        missing = [name for name in self.knodes if name not in dims]
        if missing:
            raise ValueError(f"knowledge dimensions missing for {missing}")
        self.knowledge_dims = {name: dims[name] for name in self.knodes}
        self.d_attn = self.d_model // 4
        self.k_proj = nn.ModuleDict([
            (name, nn.Linear(self.knowledge_dims[name], self.d_attn, bias=False))
            for name in self.knodes
        ])
        self.v_proj = nn.ModuleDict([
            (name, nn.Linear(self.knowledge_dims[name], self.d_model))
            for name in self.knodes
        ])
        self.Wq = nn.Linear(self.d_model, self.d_attn, bias=False)
        self.fusion_call_count = 0
        self.last_attention_logits: torch.Tensor | None = None
        self.last_attention_weights: torch.Tensor | None = None

    @staticmethod
    def _broadcast(feature: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        if feature.ndim != 2:
            raise ValueError("knowledge features must be [B,D]")
        if batch.ndim != 1 or batch.dtype != torch.long:
            raise ValueError("batch must be int64 [N]")
        if batch.numel() and (int(batch.min()) < 0 or int(batch.max()) >= feature.size(0)):
            raise ValueError("node batch index is outside knowledge batch")
        return feature[batch]

    def _vectors(
        self,
        knowledge: Mapping[str, torch.Tensor],
        batch: torch.Tensor,
        projection: nn.ModuleDict,
    ) -> torch.Tensor:
        if tuple(knowledge.keys()) != self.knodes:
            raise ValueError(
                f"knowledge modality order must be {self.knodes}, got {tuple(knowledge.keys())}"
            )
        vectors = []
        for name in self.knodes:
            value = knowledge[name]
            if value.ndim != 2 or value.size(1) != self.knowledge_dims[name]:
                raise ValueError(
                    f"knowledge[{name!r}] must be [B,{self.knowledge_dims[name]}], got {tuple(value.shape)}"
                )
            vectors.append(self._broadcast(projection[name](value), batch))
        return torch.stack(vectors, dim=1)

    def forward(
        self,
        node_feature: torch.Tensor,
        knowledge: Mapping[str, torch.Tensor],
        batch: torch.Tensor,
    ) -> torch.Tensor:
        if node_feature.ndim != 2 or node_feature.size(1) != self.d_model:
            raise ValueError(f"node_feature must be [N,{self.d_model}]")
        batch = torch.as_tensor(batch, device=node_feature.device, dtype=torch.long).reshape(-1)
        if batch.numel() != node_feature.size(0):
            raise ValueError("batch length does not match node features")
        if len(knowledge) == 0:
            self.fusion_call_count += 1
            self.last_attention_logits = None
            self.last_attention_weights = None
            return node_feature
        # The official implementation receives graph-level knowledge and
        # broadcasts each projected modality to graph nodes before stacking.
        normalized = {
            name: torch.as_tensor(knowledge[name], device=node_feature.device)
            for name in self.knodes
        }
        k_vectors = self._vectors(normalized, batch, self.k_proj)
        v_vectors = self._vectors(normalized, batch, self.v_proj)
        q = self.Wq(node_feature) / (self.d_model ** 0.5)
        logits = torch.bmm(q.unsqueeze(1), k_vectors.transpose(1, 2))
        weights = torch.softmax(logits, dim=-1)
        out = torch.bmm(weights, v_vectors).squeeze(1)
        self.fusion_call_count += 1
        self.last_attention_logits = logits.detach()
        self.last_attention_weights = weights.detach()
        return node_feature + out * 0.5

    def reset_trace(self) -> None:
        self.fusion_call_count = 0
        self.last_attention_logits = None
        self.last_attention_weights = None


__all__ = ["OriginalMIPSAttentiveFusion"]
