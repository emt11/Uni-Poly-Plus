"""PyTorch implementation of the released MIPS polymer model backbone."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MIPSFeedForward(nn.Module):
    def __init__(self, dim, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.network = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return x + self.dropout(self.network(self.norm(x)))


class MIPSLocalizedAttention(nn.Module):
    def __init__(self, dim, num_heads, dropout, attention_dropout):
        super().__init__()
        if dim % num_heads:
            raise ValueError("MIPS embedding dimension must be divisible by num_heads")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = dim ** -0.5  # Matches the public MIPS implementation.
        self.norm = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3)
        self.output = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.attention_dropout = nn.Dropout(attention_dropout)

    def forward(self, x, attention_bias, visible, node_valid):
        residual = x
        batch_size, num_nodes, _ = x.shape
        qkv = self.qkv(self.norm(x)).reshape(
            batch_size, num_nodes, 3, self.num_heads, self.head_dim
        ).permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(0)
        # DGL source implementation assigns Q/V to source nodes, K to
        # destinations, and normalizes edge_softmax over incoming edges. Dense
        # rows therefore represent destinations and columns represent sources.
        scores = torch.matmul(query * self.scale, key.transpose(-1, -2)).transpose(-1, -2)
        scores = scores + attention_bias.transpose(1, 2)[:, None]
        visible = visible.transpose(1, 2)
        scores = scores.masked_fill(~visible[:, None], torch.finfo(scores.dtype).min)
        weights = self.attention_dropout(torch.softmax(scores.float(), dim=-1).to(scores.dtype))
        output = torch.matmul(weights, value).transpose(1, 2).reshape(batch_size, num_nodes, self.dim)
        output = residual + self.dropout(self.output(output))
        return output * node_valid.unsqueeze(-1).to(output.dtype)


class MIPSLayer(nn.Module):
    def __init__(self, dim, num_heads, dropout, attention_dropout):
        super().__init__()
        self.attention = MIPSLocalizedAttention(
            dim, num_heads, dropout, attention_dropout
        )
        self.feed_forward = MIPSFeedForward(dim, dropout)

    def forward(self, x, attention_bias, visible, node_valid):
        x = self.attention(x, attention_bias, visible, node_valid)
        x = self.feed_forward(x)
        return x * node_valid.unsqueeze(-1).to(x.dtype)


class MIPSDescriptorFusion(nn.Module):
    DIMS = {"md": 200, "atom_pair_3d": 512}

    def __init__(self, dim):
        super().__init__()
        attention_dim = dim // 4
        self.dim = dim
        self.query = nn.Linear(dim, attention_dim, bias=False)
        self.keys = nn.ModuleDict({
            name: nn.Linear(width, attention_dim, bias=False)
            for name, width in self.DIMS.items()
        })
        self.values = nn.ModuleDict({
            name: nn.Linear(width, dim) for name, width in self.DIMS.items()
        })

    def forward(self, nodes, md, atom_pair_3d):
        descriptors = {"md": md, "atom_pair_3d": atom_pair_3d}
        keys = torch.stack([self.keys[name](descriptors[name]) for name in self.DIMS], dim=1)
        values = torch.stack([self.values[name](descriptors[name]) for name in self.DIMS], dim=1)
        query = self.query(nodes) / math.sqrt(self.dim)
        attention = torch.softmax(torch.einsum("bnd,bkd->bnk", query, keys), dim=-1)
        context = torch.einsum("bnk,bkd->bnd", attention, values)
        return nodes + 0.5 * context


class MIPSGraphEncoder(nn.Module):
    """Star-linking localized Graph Transformer with MIPS descriptor fusion."""

    expects_data = True
    uses_geometry = False

    def __init__(
        self,
        num_layer=6,
        emb_dim=512,
        num_heads=8,
        dropout=0.1,
        attention_dropout=0.1,
        max_length=3,
    ):
        super().__init__()
        if max_length != 3:
            raise ValueError("The released MIPS model uses max_length=3 (0/1/2-hop attention)")
        self.emb_dim = int(emb_dim)
        self.max_length = int(max_length)
        self.input_projection = nn.Linear(137, self.emb_dim)
        # The paper adds a shared learnable vector B to backbone atoms. Keep it
        # separate from chemical atom features; index zero remains exactly zero.
        self.backbone_embedding = nn.Embedding(2, self.emb_dim, padding_idx=0)
        self.distance_embedding = nn.Embedding(self.max_length, 1)
        nn.init.zeros_(self.distance_embedding.weight)
        self.path_projections = nn.ModuleList([
            nn.Linear(self.emb_dim, 1, bias=False) for _ in range(self.max_length)
        ])
        self.layers = nn.ModuleList([
            MIPSLayer(self.emb_dim, num_heads, dropout, attention_dropout)
            for _ in range(num_layer)
        ])
        self.descriptor_fusion = MIPSDescriptorFusion(self.emb_dim)
        self.descriptor_disturbance = 0.0

    @staticmethod
    def _to_padded(flat, batch_index):
        batch_size = int(batch_index.max().item()) + 1
        counts = torch.bincount(batch_index, minlength=batch_size)
        max_nodes = int(counts.max().item())
        padded = flat.new_zeros((batch_size, max_nodes, flat.size(-1)))
        valid = torch.zeros(batch_size, max_nodes, dtype=torch.bool, device=flat.device)
        for graph_idx in range(batch_size):
            selected = flat[batch_index == graph_idx]
            padded[graph_idx, :selected.size(0)] = selected
            valid[graph_idx, :selected.size(0)] = True
        return padded, valid

    def _path_bias(self, path_nodes, initial_nodes):
        valid = path_nodes >= 0
        safe = path_nodes.clamp_min(0)
        values = []
        for hop, projection in enumerate(self.path_projections):
            # The projection is linear and bias-free, so projecting the N node
            # vectors before gathering is exactly equivalent to gathering an
            # [B,N,N,D] tensor and projecting afterwards.  The latter scales as
            # O(B*N^2*D) memory and can request tens of GiB for one long repeat
            # unit; this form retains only the required O(B*N^2) scalar bias.
            projected = projection(initial_nodes).squeeze(-1)
            selected = torch.gather(
                projected[:, None, :].expand(-1, initial_nodes.size(1), -1),
                2,
                safe[..., hop],
            )
            values.append(selected * valid[..., hop])
        denominator = valid.sum(dim=-1).clamp_min(1)
        return torch.stack(values, dim=-1).sum(dim=-1) / denominator

    def _disturb_descriptors(self, md, atom_pair):
        rate = float(self.descriptor_disturbance)
        if not self.training or rate <= 0:
            return md, atom_pair
        md_mask = torch.rand_like(md) < rate
        md = torch.where(md_mask, torch.rand_like(md), md)
        fp_mask = torch.rand_like(atom_pair) < rate
        atom_pair = torch.where(fp_mask, 1.0 - atom_pair, atom_pair)
        return md, atom_pair

    def forward_with_x(self, data, x_override):
        masked = x_override.abs().sum(dim=-1) == 0
        mips_x = data.mips_x.clone()
        mips_x[masked] = 0
        return self.forward(data, mips_x_override=mips_x)

    def forward(self, data, mips_x_override=None):
        mips_x = data.mips_x if mips_x_override is None else mips_x_override
        initial_flat = self.input_projection(mips_x.float())
        initial_flat = initial_flat + self.backbone_embedding(
            data.mips_backbone_mask.long()
        )
        initial, node_valid = self._to_padded(initial_flat, data.batch.long())
        num_nodes = initial.size(1)
        spd = data.scage_spd[:, :num_nodes, :num_nodes].long()
        paths = data.mips_path_nodes[:, :num_nodes, :num_nodes]
        visible = (spd < self.max_length) & node_valid[:, :, None] & node_valid[:, None, :]
        distance_bias = self.distance_embedding(spd.clamp(0, self.max_length - 1)).squeeze(-1)
        attention_bias = distance_bias + self._path_bias(paths, initial)

        nodes = initial
        for layer in self.layers:
            nodes = layer(nodes, attention_bias, visible, node_valid)

        md, atom_pair = self._disturb_descriptors(
            data.mips_md.float(), data.mips_atom_pair_3d.float()
        )
        nodes = self.descriptor_fusion(nodes, md, atom_pair)
        nodes = nodes * node_valid.unsqueeze(-1).to(nodes.dtype)
        graph = nodes.sum(dim=1) / node_valid.sum(dim=1, keepdim=True).clamp_min(1)
        flat_nodes = torch.cat([
            nodes[idx, :int(node_valid[idx].sum().item())]
            for idx in range(nodes.size(0))
        ], dim=0)
        return graph, flat_nodes
