import torch
import torch.nn as nn
import torch.nn.functional as F

from src.kg import load_kg_embedding_store


class KGEncoder(nn.Module):
    def __init__(
        self,
        joint_embedding_dim: int = 256,
        kg_root: str = '.',
        freeze_kg_embeddings: bool = False,
    ):
        super().__init__()
        kg_store = load_kg_embedding_store(root=kg_root)
        self.embedding_dim = kg_store.embedding_dim
        self.padding_idx = kg_store.padding_idx
        self.hidden_channels = joint_embedding_dim

        self.embedding = nn.Embedding.from_pretrained(
            kg_store.embedding_weight,
            freeze=freeze_kg_embeddings,
            padding_idx=self.padding_idx,
        )
        self.attention = nn.Linear(self.embedding_dim, 1)
        self.projection = nn.Sequential(
            nn.Linear(self.embedding_dim, joint_embedding_dim),
            nn.LayerNorm(joint_embedding_dim),
            nn.ReLU(),
        )

    def forward(self, kg_entity_ids: torch.Tensor, kg_mask: torch.Tensor):
        embeddings = self.embedding(kg_entity_ids)
        scores = self.attention(embeddings).squeeze(-1)
        mask = kg_mask.to(dtype=torch.bool, device=scores.device)
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        weights = F.softmax(scores, dim=1)
        weights = torch.where(mask, weights, torch.zeros_like(weights))

        normalizer = weights.sum(dim=1, keepdim=True).clamp(min=1e-8)
        weights = weights / normalizer
        pooled = torch.sum(embeddings * weights.unsqueeze(-1), dim=1)
        return self.projection(pooled)
