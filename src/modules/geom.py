import torch
import os
import math
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, radius_graph
try:
    from torch_geometric.utils import scatter
except ImportError:  # pragma: no cover - compatibility with older PyG stacks
    from torch_scatter import scatter
from typing import Optional, Callable
from torch import nn, Tensor


GEOM_EPS = 1e-8
PAINN_UPDATE_SCALE = 0.05
PAINN_MAX_VECTOR_NORM = 20.0


def _unpack_geometry_input(data_or_z, pos: Optional[Tensor] = None, batch: Optional[Tensor] = None):
    """Support both encoder(data) and encoder(z, pos, batch) without changing callers."""
    if pos is not None:
        z = data_or_z
        batch = torch.zeros_like(z) if batch is None else batch
        return z.long(), pos.float(), batch.long()

    data = data_or_z
    if hasattr(data, 'x3d') and hasattr(data, 'pos3d'):
        z = data.x3d
        pos = data.pos3d
        batch = getattr(data, 'batch3d', None)
    elif hasattr(data, 'z') and hasattr(data, 'pos'):
        z = data.z
        pos = data.pos
        batch = getattr(data, 'batch', None)
    else:
        raise AttributeError("Geometry encoder expects x3d/pos3d or z/pos fields.")

    batch = torch.zeros_like(z) if batch is None else batch
    return z.long(), pos.float(), batch.long()


class GaussianRBF(nn.Module):
    def __init__(self, num_rbf: int, cutoff: float):
        super().__init__()
        centers = torch.linspace(0.0, cutoff, num_rbf)
        self.register_buffer('centers', centers)
        self.gamma = float(num_rbf) / cutoff

    def forward(self, distances: Tensor) -> Tensor:
        return torch.exp(-self.gamma * (distances.unsqueeze(-1) - self.centers) ** 2)


class CosineCutoff(nn.Module):
    def __init__(self, cutoff: float):
        super().__init__()
        self.cutoff = cutoff

    def forward(self, distances: Tensor) -> Tensor:
        cutoffs = 0.5 * (torch.cos(distances * math.pi / self.cutoff) + 1.0)
        return cutoffs * (distances < self.cutoff).to(distances.dtype)


class PaiNNInteraction(nn.Module):
    def __init__(self, hidden_channels: int, num_rbf: int, update_scale: float = PAINN_UPDATE_SCALE):
        super().__init__()
        self.filter_net = nn.Sequential(
            nn.Linear(num_rbf, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels * 3)
        )
        self.scalar_net = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels * 3)
        )
        self.update_scale = update_scale

    def forward(
        self,
        s: Tensor,
        v: Tensor,
        edge_index: Tensor,
        edge_rbf: Tensor,
        edge_unit: Tensor
    ):
        row, col = edge_index
        filters = self.filter_net(edge_rbf)
        scalar_messages = self.scalar_net(s[col]) * filters
        msg_ss, msg_sv, msg_vv = scalar_messages.chunk(3, dim=-1)

        v_j = v[col]
        v_dot = (v_j * edge_unit.unsqueeze(-1)).sum(dim=1)
        ds = msg_ss + msg_sv * v_dot
        dv = (
            msg_vv.unsqueeze(1) * v_j
            + msg_sv.unsqueeze(1) * edge_unit.unsqueeze(-1)
        )

        s = s + self.update_scale * scatter(ds, row, dim=0, dim_size=s.size(0), reduce='mean')
        v = v + self.update_scale * scatter(dv, row, dim=0, dim_size=v.size(0), reduce='mean')
        return s, v


class PaiNNMixing(nn.Module):
    def __init__(self, hidden_channels: int, eps: float = 1e-8, update_scale: float = PAINN_UPDATE_SCALE):
        super().__init__()
        self.eps = eps
        self.vector_u = nn.Linear(hidden_channels, hidden_channels, bias=False)
        self.vector_v = nn.Linear(hidden_channels, hidden_channels, bias=False)
        self.scalar_mlp = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels * 3)
        )
        self.update_scale = update_scale

    def forward(self, s: Tensor, v: Tensor):
        u = self.vector_u(v)
        v_proj = self.vector_v(v)
        v_norm = torch.sqrt(torch.sum(v_proj ** 2, dim=1) + self.eps)

        ds, dv, dsv = self.scalar_mlp(torch.cat([s, v_norm], dim=-1)).chunk(3, dim=-1)
        uv_dot = torch.sum(u * v_proj, dim=1)

        s = s + self.update_scale * (ds + dsv * uv_dot)
        v = v + self.update_scale * dv.unsqueeze(1) * u
        return s, v


class PaiNNEncoder(nn.Module):
    """
    PaiNN geometry encoder with scalar and vector features.

    The public output is graph-level embeddings with shape
    [num_graphs, hidden_channels].
    """
    def __init__(
        self,
        hidden_channels: int = 128,
        num_layers: int = 6,
        num_rbf: int = 50,
        cutoff: float = 10.0,
        max_num_neighbors: int = 32,
        readout: str = 'mean',
        max_z: int = 100,
        eps: float = GEOM_EPS,
        load_from_pretrain: Optional[str] = None,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.num_rbf = num_rbf
        self.cutoff = cutoff
        self.max_num_neighbors = max_num_neighbors
        self.readout_name = readout
        self.eps = eps

        self.embedding = nn.Embedding(max_z, hidden_channels)
        self.rbf = GaussianRBF(num_rbf=num_rbf, cutoff=cutoff)
        self.cutoff_fn = CosineCutoff(cutoff=cutoff)
        self.interactions = nn.ModuleList([
            PaiNNInteraction(hidden_channels, num_rbf) for _ in range(num_layers)
        ])
        self.mixing = nn.ModuleList([
            PaiNNMixing(hidden_channels) for _ in range(num_layers)
        ])
        self.scalar_norms = nn.ModuleList([
            nn.LayerNorm(hidden_channels) for _ in range(num_layers)
        ])

        if readout == 'mean':
            self.readout = global_mean_pool
        elif readout in {'add', 'sum'}:
            self.readout = global_add_pool
        elif readout == 'max':
            self.readout = global_max_pool
        else:
            raise ValueError(f"Unsupported PaiNN readout: {readout}")

        if load_from_pretrain is not None:
            self.load_pretrained_weights(load_from_pretrain)

    def load_pretrained_weights(self, pretrain_path: str):
        if not os.path.exists(pretrain_path):
            raise FileNotFoundError(f"Pretrained model file not found: {pretrain_path}")

        state_dict = torch.load(pretrain_path, map_location='cpu')
        current_state = self.state_dict()
        compatible_state = {
            key: value
            for key, value in state_dict.items()
            if key in current_state and current_state[key].shape == value.shape
        }
        skipped = len(state_dict) - len(compatible_state)
        self.load_state_dict(compatible_state, strict=False)
        print(
            f"Loaded {len(compatible_state)} compatible PaiNN weights from {pretrain_path}; "
            f"skipped {skipped} incompatible weights."
        )

    def _edge_features(self, pos: Tensor, batch: Tensor, cell: Optional[Tensor] = None, pbc: Optional[Tensor] = None):
        if not torch.isfinite(pos).all():
            raise ValueError("PaiNNEncoder received non-finite coordinates.")

        if cell is not None and pbc is not None:
            return self._periodic_edge_features(pos, batch, cell, pbc)

        edge_index = radius_graph(
            pos,
            r=self.cutoff,
            batch=batch,
            loop=False,
            max_num_neighbors=self.max_num_neighbors
        )
        if edge_index.numel() == 0:
            return edge_index, pos.new_empty((0, self.num_rbf)), pos.new_empty((0, 3))

        row, col = edge_index
        edge_vec = pos[row] - pos[col]
        return self._finalize_edge_features(edge_index, edge_vec, pos)

    def _periodic_edge_features(self, pos: Tensor, batch: Tensor, cell: Tensor, pbc: Tensor):
        if cell.dim() == 2:
            cell = cell.unsqueeze(0)
        if pbc.dim() == 1:
            pbc = pbc.unsqueeze(0)

        edge_chunks = []
        edge_vec_chunks = []
        unique_batches = torch.unique(batch, sorted=True)
        for graph_id in unique_batches.tolist():
            graph_mask = batch == graph_id
            node_idx = torch.nonzero(graph_mask, as_tuple=False).view(-1)
            if node_idx.numel() == 0:
                continue

            graph_pos = pos[node_idx]
            graph_cell = cell[int(graph_id)]
            graph_pbc = pbc[int(graph_id)]
            active_axes = torch.nonzero(graph_pbc.bool(), as_tuple=False).flatten()
            if active_axes.numel() != 1:
                edge_index = radius_graph(
                    graph_pos,
                    r=self.cutoff,
                    batch=graph_pos.new_zeros(graph_pos.size(0), dtype=torch.long),
                    loop=False,
                    max_num_neighbors=self.max_num_neighbors,
                )
                if edge_index.numel() > 0:
                    row, col = edge_index
                    edge_chunks.append(torch.stack([node_idx[row], node_idx[col]], dim=0))
                    edge_vec_chunks.append(graph_pos[row] - graph_pos[col])
                continue

            graph_edges, graph_vecs = self._periodic_graph_edges(
                node_idx, graph_pos, graph_cell[int(active_axes[0].item())]
            )
            if graph_edges.numel() > 0:
                edge_chunks.append(graph_edges)
                edge_vec_chunks.append(graph_vecs)

        if not edge_chunks:
            return pos.new_empty((2, 0), dtype=torch.long), pos.new_empty((0, self.num_rbf)), pos.new_empty((0, 3))

        edge_index = torch.cat(edge_chunks, dim=1)
        edge_vec = torch.cat(edge_vec_chunks, dim=0)
        return self._finalize_edge_features(edge_index, edge_vec, pos)

    def _periodic_graph_edges(self, node_idx: Tensor, graph_pos: Tensor, vector_t: Tensor):
        rows = []
        cols = []
        vecs = []
        n_nodes = graph_pos.size(0)
        shifts = (-1.0, 0.0, 1.0)
        for local_row in range(n_nodes):
            candidate_cols = []
            candidate_vecs = []
            candidate_dists = []
            target_pos = graph_pos[local_row]
            for shift in shifts:
                shifted_pos = graph_pos + vector_t.view(1, 3) * shift
                edge_vec = target_pos.view(1, 3) - shifted_pos
                edge_dist = torch.linalg.vector_norm(edge_vec, dim=-1)
                valid = edge_dist < self.cutoff
                if shift == 0.0:
                    valid[local_row] = False
                valid = valid & (edge_dist > self.eps)
                if valid.any():
                    local_cols = torch.nonzero(valid, as_tuple=False).view(-1)
                    candidate_cols.append(local_cols)
                    candidate_vecs.append(edge_vec[local_cols])
                    candidate_dists.append(edge_dist[local_cols])
            if not candidate_cols:
                continue
            local_cols = torch.cat(candidate_cols, dim=0)
            local_vecs = torch.cat(candidate_vecs, dim=0)
            local_dists = torch.cat(candidate_dists, dim=0)
            if local_cols.numel() > self.max_num_neighbors:
                keep = torch.argsort(local_dists)[:self.max_num_neighbors]
                local_cols = local_cols[keep]
                local_vecs = local_vecs[keep]
            rows.append(node_idx[local_row].repeat(local_cols.numel()))
            cols.append(node_idx[local_cols])
            vecs.append(local_vecs)

        if not rows:
            return graph_pos.new_empty((2, 0), dtype=torch.long), graph_pos.new_empty((0, 3))
        return torch.stack([torch.cat(rows), torch.cat(cols)], dim=0), torch.cat(vecs, dim=0)

    def _finalize_edge_features(self, edge_index: Tensor, edge_vec: Tensor, pos: Tensor):
        if edge_index.numel() == 0:
            return edge_index, pos.new_empty((0, self.num_rbf)), pos.new_empty((0, 3))
        edge_dist = torch.norm(edge_vec, dim=-1)
        valid_edge_mask = edge_dist > self.eps
        edge_index = edge_index[:, valid_edge_mask]
        edge_vec = edge_vec[valid_edge_mask]
        edge_dist = edge_dist[valid_edge_mask].clamp_min(self.eps)
        if edge_index.numel() == 0:
            return edge_index, pos.new_empty((0, self.num_rbf)), pos.new_empty((0, 3))

        edge_unit = edge_vec / edge_dist.unsqueeze(-1)
        edge_rbf = self.rbf(edge_dist).clamp_(0.0, 1.0) * self.cutoff_fn(edge_dist).unsqueeze(-1)
        return edge_index, edge_rbf, edge_unit

    def _stabilize_features(self, s: Tensor, v: Tensor, norm: nn.LayerNorm):
        s = norm(s)
        vector_norm = torch.linalg.vector_norm(v, dim=1, keepdim=True)
        vector_scale = torch.clamp(PAINN_MAX_VECTOR_NORM / (vector_norm + self.eps), max=1.0)
        v = v * vector_scale
        return s, v

    def encode_nodes(self, z: Tensor, pos: Optional[Tensor] = None, batch: Optional[Tensor] = None):
        data = z if pos is None and not torch.is_tensor(z) else None
        z, pos, batch = _unpack_geometry_input(z, pos, batch)
        cell = getattr(data, 'cell', None) if data is not None else None
        pbc = getattr(data, 'pbc', None) if data is not None else None

        s = self.embedding(z)
        v = torch.zeros(s.size(0), 3, self.hidden_channels, device=s.device, dtype=s.dtype)

        edge_index, edge_rbf, edge_unit = self._edge_features(pos, batch, cell=cell, pbc=pbc)
        if edge_index.numel() > 0:
            for interaction, mixing, norm in zip(self.interactions, self.mixing, self.scalar_norms):
                s, v = interaction(s, v, edge_index, edge_rbf, edge_unit)
                s, v = mixing(s, v)
                s, v = self._stabilize_features(s, v, norm)
        else:
            for mixing, norm in zip(self.mixing, self.scalar_norms):
                s, v = mixing(s, v)
                s, v = self._stabilize_features(s, v, norm)
        return s, batch

    def forward(self, z: Tensor, pos: Optional[Tensor] = None, batch: Optional[Tensor] = None) -> Tensor:
        data = z if pos is None and not torch.is_tensor(z) else None
        s, batch = self.encode_nodes(z, pos, batch)
        if not torch.isfinite(s).all():
            raise ValueError("PaiNNEncoder produced non-finite node features.")
        pool_mask = getattr(data, 'geom_pool_mask', None) if data is not None else None
        if pool_mask is not None:
            pool_mask = pool_mask.bool()
            s = s[pool_mask]
            batch = batch[pool_mask]
        graph_embedding = self.readout(s, batch)
        return graph_embedding

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}('
                f'hidden_channels={self.hidden_channels}, '
                f'num_layers={self.num_layers}, '
                f'num_rbf={self.num_rbf}, '
                f'cutoff={self.cutoff}, '
                f'readout={self.readout_name})')
