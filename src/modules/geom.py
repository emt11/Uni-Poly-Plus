import torch
import os
import math
from torch_geometric.nn.models import SchNet
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, radius_graph
try:
    from torch_geometric.utils import scatter
except ImportError:  # pragma: no cover - compatibility with older PyG stacks
    from torch_scatter import scatter
from typing import Optional, Callable
from torch import nn, Tensor


GEOM_EPS = 1e-8


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


class SchNetEncoder(SchNet):
    """
    Encoder version of SchNet that outputs high-dimensional graph representations instead of scalar values.
    """
    def __init__(
        self,
        hidden_channels: int = 128,
        num_filters: int = 128,
        num_interactions: int = 6,
        num_gaussians: int = 50,
        cutoff: float = 10.0,
        interaction_graph: Optional[Callable] = None,
        max_num_neighbors: int = 32,
        readout: str = 'mean',
        dipole: bool = False,
        mean: Optional[float] = None,
        std: Optional[float] = None,
        atomref: Optional[Tensor] = None,
        load_from_pretrain: Optional[str] = '../pretrained_models/encoders/schnet_qm9_model.pt',
    ):
        super().__init__(
            hidden_channels=hidden_channels,
            num_filters=num_filters,
            num_interactions=num_interactions,
            num_gaussians=num_gaussians,
            cutoff=cutoff,
            interaction_graph=interaction_graph,
            max_num_neighbors=max_num_neighbors,
            readout=readout,
            dipole=dipole,
            mean=mean,
            std=std,
            atomref=atomref,
        )
        
        # Remove regression output layers
        # self.lin1 = Linear(hidden_channels, hidden_channels // 2)
        # self.act = ShiftedSoftplus()
        # self.lin2 = Linear(hidden_channels // 2, 1)
        
        if load_from_pretrain is not None:
            self.load_pretrained_weights(load_from_pretrain)
            
    def load_pretrained_weights(self, pretrain_path: str):
        """
        Load weights from pretrained model path.

        Args:
            pretrain_path (str): File path to pretrained model.
        """
        if not os.path.exists(pretrain_path):
            raise FileNotFoundError(f"Pretrained model file not found: {pretrain_path}")
        
        # Load pretrained weights
        state_dict = torch.load(pretrain_path, map_location='cpu')
        self.load_state_dict(state_dict, strict=False)


    def forward(self, z: Tensor, pos: Optional[Tensor] = None, batch: Optional[Tensor] = None) -> Tensor:
        """
        Forward pass, returns graph-level high-dimensional representations.
        
        Args:
            z (torch.Tensor): Atomic numbers for each atom, shape [num_atoms].
            pos (torch.Tensor): Coordinates for each atom, shape [num_atoms, 3].
            batch (torch.Tensor, optional): Batch indices, shape [num_atoms]. Defaults to None.
        
        Returns:
            torch.Tensor: Graph-level embeddings, shape [num_graphs, hidden_channels].
        """
        z, pos, batch = _unpack_geometry_input(z, pos, batch)

        h = self.embedding(z)
        edge_index, edge_weight = self.interaction_graph(pos, batch)
        edge_attr = self.distance_expansion(edge_weight)

        for interaction in self.interactions:
            h = h + interaction(h, edge_index, edge_weight, edge_attr)


        # h = self.lin1(h)
        # h = self.act(h)
        # h = self.lin2(h)

        if self.dipole:
            # Calculate center of mass
            mass = self.atomic_mass[z].view(-1, 1)
            M = self.sum_aggr(mass, batch, dim=0)
            c = self.sum_aggr(mass * pos, batch, dim=0) / M
            h = h * (pos - c.index_select(0, batch))

        if not self.dipole and self.mean is not None and self.std is not None:
            h = h * self.std + self.mean

        if not self.dipole and self.atomref is not None:
            h = h + self.atomref(z)

        # Use readout to generate graph-level embeddings
        graph_embedding = self.readout(h, batch, dim=0)
        # if self.dipole:
        #     out = torch.norm(out, dim=-1, keepdim=True)

        # if self.scale is not None:
        #     out = self.scale * out

        return graph_embedding

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}('
                f'hidden_channels={self.hidden_channels}, '
                f'num_filters={self.num_filters}, '
                f'num_interactions={self.num_interactions}, '
                f'num_gaussians={self.num_gaussians}, '
                f'cutoff={self.cutoff})')


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
    def __init__(self, hidden_channels: int, num_rbf: int):
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

        s = s + scatter(ds, row, dim=0, dim_size=s.size(0), reduce='mean')
        v = v + scatter(dv, row, dim=0, dim_size=v.size(0), reduce='mean')
        return s, v


class PaiNNMixing(nn.Module):
    def __init__(self, hidden_channels: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.vector_u = nn.Linear(hidden_channels, hidden_channels, bias=False)
        self.vector_v = nn.Linear(hidden_channels, hidden_channels, bias=False)
        self.scalar_mlp = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels * 3)
        )

    def forward(self, s: Tensor, v: Tensor):
        u = self.vector_u(v)
        v_proj = self.vector_v(v)
        v_norm = torch.sqrt(torch.sum(v_proj ** 2, dim=1) + self.eps)

        ds, dv, dsv = self.scalar_mlp(torch.cat([s, v_norm], dim=-1)).chunk(3, dim=-1)
        uv_dot = torch.sum(u * v_proj, dim=1)

        s = s + ds + dsv * uv_dot
        v = v + dv.unsqueeze(1) * u
        return s, v


class PaiNNEncoder(nn.Module):
    """
    PaiNN geometry encoder with scalar and vector features.

    The public output matches SchNetEncoder: graph-level embeddings with shape
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

    def _edge_features(self, pos: Tensor, batch: Tensor):
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
        edge_dist = torch.norm(edge_vec, dim=-1)
        valid_edge_mask = edge_dist > self.eps
        edge_index = edge_index[:, valid_edge_mask]
        edge_vec = edge_vec[valid_edge_mask]
        edge_dist = edge_dist[valid_edge_mask].clamp_min(self.eps)
        if edge_index.numel() == 0:
            return edge_index, pos.new_empty((0, self.num_rbf)), pos.new_empty((0, 3))

        edge_unit = edge_vec / edge_dist.unsqueeze(-1)
        edge_rbf = self.rbf(edge_dist) * self.cutoff_fn(edge_dist).unsqueeze(-1)
        return edge_index, edge_rbf, edge_unit

    def forward(self, z: Tensor, pos: Optional[Tensor] = None, batch: Optional[Tensor] = None) -> Tensor:
        z, pos, batch = _unpack_geometry_input(z, pos, batch)

        s = self.embedding(z)
        v = torch.zeros(s.size(0), 3, self.hidden_channels, device=s.device, dtype=s.dtype)

        edge_index, edge_rbf, edge_unit = self._edge_features(pos, batch)
        if edge_index.numel() > 0:
            for interaction, mixing in zip(self.interactions, self.mixing):
                s, v = interaction(s, v, edge_index, edge_rbf, edge_unit)
                s, v = mixing(s, v)
        else:
            for mixing in self.mixing:
                s, v = mixing(s, v)

        graph_embedding = self.readout(s, batch)
        return graph_embedding

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}('
                f'hidden_channels={self.hidden_channels}, '
                f'num_layers={self.num_layers}, '
                f'num_rbf={self.num_rbf}, '
                f'cutoff={self.cutoff}, '
                f'readout={self.readout_name})')
