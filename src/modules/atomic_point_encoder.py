"""Minimal runtime Atomic Point Cloud encoder for the MIPS extension.

The module deliberately has no dependency on the MIPS graph/science
implementation.  One point is one atom from the audited open-Trimer
geometry.  Geometry enters only through pairwise distances and a dynamic
per-forward kNN graph; absolute coordinates are never concatenated to a
learned feature.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence
import math
import time

import torch
from torch import nn
from torch_geometric.utils import softmax as segment_softmax
from torch_scatter import scatter

try:  # The project environment already provides this CUDA/CPU extension.
    from torch_cluster import knn_graph as _torch_cluster_knn_graph
except Exception:  # pragma: no cover - exercised only in a missing dependency env
    _torch_cluster_knn_graph = None


ATOMIC_PC_SCHEMA = "atomic-point-cloud-v1"
ATOMIC_PC_KNN_BACKENDS = ("torch_cluster", "torch_cdist_reference")


@dataclass(frozen=True)
class PackedAtomicPointCloud:
    """Packed variable-length atom points.

    ``batch`` is sorted and gives one graph id per point.  ``ptr`` is optional
    at the model boundary but retained by the adapter for auditable joins.
    ``sample_keys`` and ``source_smiles`` are metadata only and never enter the
    neural computation.
    """

    coords: torch.Tensor
    atomic_number: torch.Tensor
    ru_offset: torch.Tensor
    batch: torch.Tensor
    ptr: torch.Tensor | None = None
    sample_keys: tuple[str, ...] | None = None
    source_smiles: tuple[str, ...] | None = None

    @property
    def batch_index(self) -> torch.Tensor:
        return self.batch

    @property
    def central_flag(self) -> torch.Tensor:
        """Central-role indicator derived solely from ``ru_offset``."""
        return self.ru_offset.eq(0)

    def to(self, device: torch.device | str) -> "PackedAtomicPointCloud":
        """Return a device-moved packed view without changing metadata."""
        target = torch.device(device)
        return PackedAtomicPointCloud(
            self.coords.to(target), self.atomic_number.to(target),
            self.ru_offset.to(target), self.batch.to(target),
            None if self.ptr is None else self.ptr.to(target),
            self.sample_keys, self.source_smiles,
        )


def pack_point_clouds(records: Sequence[Any], *, device: torch.device | str | None = None) -> PackedAtomicPointCloud:
    """Pack record-like objects without padding or row-order inference."""

    if not records:
        raise ValueError("cannot pack an empty point-cloud sequence")
    coords_parts, z_parts, offset_parts, batch_parts = [], [], [], []
    ptr = [0]
    sample_keys: list[str] = []
    smiles: list[str] = []
    for graph_id, record in enumerate(records):
        coords = _field(record, "trimer_coords", "coords")
        z = _field(record, "trimer_atomic_number", "atomic_number", "z")
        offset = _field(record, "trimer_ru_offset", "ru_offset", "offset")
        _validate_point_tensors(coords, z, offset)
        n = int(coords.size(0))
        coords_parts.append(coords)
        z_parts.append(z)
        offset_parts.append(offset)
        batch_parts.append(torch.full((n,), graph_id, dtype=torch.long, device=coords.device))
        ptr.append(ptr[-1] + n)
        key = _field_optional(record, "sample_key")
        sample_keys.append(_key_text(key) if key is not None else "")
        source = _field_optional(record, "source_smiles")
        smiles.append(str(source) if source is not None else "")
    target_device = torch.device(device) if device is not None else coords_parts[0].device
    coords_out = torch.cat([value.to(target_device) for value in coords_parts], dim=0)
    z_out = torch.cat([value.to(target_device) for value in z_parts], dim=0)
    offset_out = torch.cat([value.to(target_device) for value in offset_parts], dim=0)
    batch_out = torch.cat([value.to(target_device) for value in batch_parts], dim=0)
    return PackedAtomicPointCloud(
        coords=coords_out,
        atomic_number=z_out,
        ru_offset=offset_out,
        batch=batch_out,
        ptr=torch.tensor(ptr, dtype=torch.long, device=target_device),
        sample_keys=tuple(sample_keys),
        source_smiles=tuple(smiles),
    )


def _field(value: Any, *names: str) -> Any:
    for name in names:
        if isinstance(value, Mapping) and name in value:
            return value[name]
        if hasattr(value, name):
            result = getattr(value, name)
            if result is not None:
                return result
    raise ValueError("point-cloud record is missing " + "/".join(names))


def _field_optional(value: Any, *names: str) -> Any:
    for name in names:
        if isinstance(value, Mapping) and name in value:
            return value[name]
        if hasattr(value, name):
            result = getattr(value, name)
            if result is not None:
                return result
    return None


def _key_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.hex()
    return str(value)


def _validate_point_tensors(coords: torch.Tensor, z: torch.Tensor, ru_offset: torch.Tensor) -> None:
    if not torch.is_tensor(coords) or not torch.is_tensor(z) or not torch.is_tensor(ru_offset):
        raise TypeError("Atomic-PC fields must be torch tensors")
    if coords.dtype != torch.float32:
        raise TypeError(f"trimer_coords must be float32, got {coords.dtype}")
    if z.dtype != torch.int64:
        raise TypeError(f"trimer_atomic_number must be int64, got {z.dtype}")
    if ru_offset.dtype != torch.int64:
        raise TypeError(f"trimer_ru_offset must be int64, got {ru_offset.dtype}")
    if coords.ndim != 2 or tuple(coords.shape[1:]) != (3,):
        raise ValueError(f"trimer_coords must have shape [N,3], got {tuple(coords.shape)}")
    if z.ndim != 1 or ru_offset.ndim != 1 or z.numel() != coords.size(0) or ru_offset.numel() != coords.size(0):
        raise ValueError("Atomic-PC fields must share the point count")
    if coords.numel() == 0:
        raise ValueError("Atomic-PC records may not be empty")
    if not bool(torch.isfinite(coords).all()):
        raise ValueError("trimer_coords contains NaN or Inf")
    if z.numel() and (int(z.min()) < 1 or int(z.max()) > 118):
        raise ValueError("atomic numbers must lie in 1..118")
    if ru_offset.numel() and not bool(torch.isin(ru_offset, torch.tensor([-1, 0, 1], device=ru_offset.device)).all()):
        raise ValueError("trimer_ru_offset values must lie in {-1,0,+1}")


def _unpack_cloud(cloud: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    coords = _field(cloud, "coords", "trimer_coords")
    z = _field(cloud, "atomic_number", "trimer_atomic_number", "z")
    offset = _field(cloud, "ru_offset", "trimer_ru_offset", "offset")
    batch = _field(cloud, "batch", "batch_index", "graph_index")
    ptr = _field_optional(cloud, "ptr")
    _validate_point_tensors(coords, z, offset)
    if not torch.is_tensor(batch) or batch.dtype != torch.int64 or batch.ndim != 1 or batch.numel() != coords.size(0):
        raise ValueError("packed point batch must be int64 [sum_N]")
    if batch.numel() and int(batch.min()) < 0:
        raise ValueError("packed point batch indices must be non-negative")
    if batch.numel() > 1 and bool((batch[1:] < batch[:-1]).any()):
        raise ValueError("packed point batch indices must be sorted")
    if ptr is not None:
        if not torch.is_tensor(ptr) or ptr.dtype != torch.int64 or ptr.ndim != 1 or ptr.numel() < 2:
            raise ValueError("ptr must be int64 [B+1]")
        if int(ptr[0]) != 0 or int(ptr[-1]) != int(coords.size(0)) or bool((ptr[1:] < ptr[:-1]).any()):
            raise ValueError("ptr does not describe the packed point count")
    return coords, z, offset, batch, ptr


def _reference_knn_edges(coords: torch.Tensor, batch: torch.Tensor, k: int) -> torch.Tensor:
    """Reference per-sample cdist+topk graph, used only when requested."""

    edges: list[torch.Tensor] = []
    graph_count = int(batch.max().item()) + 1 if batch.numel() else 0
    for graph_id in range(graph_count):
        indices = torch.nonzero(batch == graph_id, as_tuple=False).flatten()
        n = int(indices.numel())
        if n <= 1:
            continue
        local = coords[indices]
        distances = torch.cdist(local, local, p=2)
        distances.fill_diagonal_(float("inf"))
        count = min(int(k), n - 1)
        nearest = torch.topk(distances, count, dim=1, largest=False, sorted=True).indices
        dst = torch.arange(n, device=coords.device).unsqueeze(1).expand(n, count).reshape(-1)
        src = nearest.reshape(-1)
        edges.append(torch.stack((indices[src], indices[dst]), dim=0))
    if not edges:
        return torch.empty((2, 0), dtype=torch.long, device=coords.device)
    return torch.cat(edges, dim=1)


def _index_control_edges(batch: torch.Tensor, k: int) -> torch.Tensor:
    """Coordinate-free deterministic neighborhood for the future A-NG path."""

    edges: list[torch.Tensor] = []
    graph_count = int(batch.max().item()) + 1 if batch.numel() else 0
    for graph_id in range(graph_count):
        indices = torch.nonzero(batch == graph_id, as_tuple=False).flatten()
        n = int(indices.numel())
        if n <= 1:
            continue
        count = min(int(k), n - 1)
        dst_local = torch.arange(n, device=batch.device).repeat_interleave(count)
        offsets = torch.arange(1, count + 1, device=batch.device).repeat(n)
        src_local = (dst_local + offsets) % n
        edges.append(torch.stack((indices[src_local], indices[dst_local]), dim=0))
    if not edges:
        return torch.empty((2, 0), dtype=torch.long, device=batch.device)
    return torch.cat(edges, dim=1)


def count_knn_boundary_ties(coords: torch.Tensor, batch: torch.Tensor, k: int, *, atol: float = 1e-7) -> int:
    """Count samples/destinations whose kNN boundary has an exact tie."""

    ties = 0
    graph_count = int(batch.max().item()) + 1 if batch.numel() else 0
    for graph_id in range(graph_count):
        indices = torch.nonzero(batch == graph_id, as_tuple=False).flatten()
        local = coords[indices]
        n = int(local.size(0))
        if n <= 1 or n - 1 <= int(k):
            continue
        distances = torch.cdist(local, local, p=2)
        distances.fill_diagonal_(float("inf"))
        ordered = torch.sort(distances, dim=1).values
        boundary = ordered[:, int(k) - 1]
        ties += int((torch.isclose(ordered[:, int(k)], boundary, atol=atol, rtol=0.0)).sum())
    return ties


class AtomicPointAttentionLayer(nn.Module):
    """Transparent distance-conditioned local multi-head attention."""

    def __init__(self, hidden_dim: int = 256, distance_dim: int = 32, num_heads: int = 8, dropout: float = 0.1) -> None:
        super().__init__()
        hidden_dim, num_heads = int(hidden_dim), int(num_heads)
        if hidden_dim % num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.distance_bias = nn.Sequential(
            nn.Linear(distance_dim, distance_dim), nn.SiLU(),
            # The Atomic-PC-v1 contract uses one scalar distance bias per
            # directed edge; the scalar is broadcast to all attention heads.
            nn.Linear(distance_dim, 1),
        )
        self.distance_value = nn.Sequential(
            nn.Linear(distance_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.output = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(float(dropout))
        self.norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim), nn.SiLU(),
            nn.Dropout(float(dropout)), nn.Linear(4 * hidden_dim, hidden_dim),
        )
        self.ffn_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, distance_embedding: torch.Tensor) -> torch.Tensor:
        if edge_index.numel() == 0:
            return self.ffn_norm(x + self.ffn(self.norm(x)))
        source, destination = edge_index
        q = self.q_proj(x).view(-1, self.num_heads, self.head_dim)
        key = self.k_proj(x).view(-1, self.num_heads, self.head_dim)
        value = self.v_proj(x).view(-1, self.num_heads, self.head_dim)
        logits = (q[destination] * key[source]).sum(dim=-1) / math.sqrt(self.head_dim)
        logits = logits + self.distance_bias(distance_embedding)
        alpha = segment_softmax(logits.float(), destination, num_nodes=x.size(0)).to(value.dtype)
        conditioned = value[source] + self.distance_value(distance_embedding).view(-1, self.num_heads, self.head_dim)
        messages = alpha.unsqueeze(-1) * conditioned
        aggregated = scatter(
            messages.reshape(-1, self.hidden_dim), destination,
            dim=0, dim_size=x.size(0), reduce="sum",
        )
        x = self.norm(x + self.dropout(self.output(aggregated)))
        x = self.ffn_norm(x + self.dropout(self.ffn(x)))
        return x


class AtomicPointEncoder(nn.Module):
    """Atomic-PC-v1 encoder producing one 512-channel vector per graph."""

    expects_packed_point_cloud = True

    def __init__(
        self,
        *,
        hidden_dim: int = 256,
        output_dim: int = 512,
        num_layers: int = 4,
        k_neighbors: int = 24,
        element_embedding_dim: int = 64,
        role_embedding_dim: int = 8,
        distance_dim: int = 32,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_geometry: bool = True,
        knn_backend: str = "torch_cluster",
        pool_scope: str = "all",
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.output_dim = int(output_dim)
        self.num_layers = int(num_layers)
        self.k_neighbors = int(k_neighbors)
        self.element_embedding_dim = int(element_embedding_dim)
        self.role_embedding_dim = int(role_embedding_dim)
        self.distance_dim = int(distance_dim)
        self.use_geometry = bool(use_geometry)
        self.knn_backend = str(knn_backend)
        self.pool_scope = str(pool_scope)
        if self.pool_scope not in ("all", "center"):
            raise ValueError(f"pool_scope must be 'all' or 'center', got {self.pool_scope!r}")
        if self.hidden_dim != 256 or self.output_dim != 512:
            raise ValueError("Atomic-PC-v1 fixes hidden_dim=256 and output_dim=512")
        if self.num_layers != 4 or self.k_neighbors != 24:
            raise ValueError("Atomic-PC-v1 fixes num_layers=4 and k_neighbors=24")
        if self.knn_backend not in ATOMIC_PC_KNN_BACKENDS:
            raise ValueError(f"unknown kNN backend: {self.knn_backend}")
        if self.knn_backend == "torch_cluster" and _torch_cluster_knn_graph is None:
            raise RuntimeError("torch_cluster backend requested but torch_cluster is unavailable")
        if self.element_embedding_dim != 64 or self.role_embedding_dim != 8 or self.distance_dim != 32:
            raise ValueError("Atomic-PC-v1 fixes element=64, role=8, distance=32")
        self.element_embedding = nn.Embedding(119, self.element_embedding_dim)
        self.role_embedding = nn.Embedding(2, self.role_embedding_dim)
        self.input_projection = nn.Linear(self.element_embedding_dim + self.role_embedding_dim, self.hidden_dim)
        self.distance_encoder = nn.Sequential(
            nn.Linear(1, self.distance_dim), nn.SiLU(),
            nn.Linear(self.distance_dim, self.distance_dim), nn.SiLU(),
        )
        self.layers = nn.ModuleList(
            AtomicPointAttentionLayer(self.hidden_dim, self.distance_dim, num_heads, dropout)
            for _ in range(self.num_layers)
        )
        self.pool_projection = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.pool_score = nn.Linear(self.hidden_dim, 1, bias=False)
        self.output_projection = nn.Linear(self.hidden_dim, self.output_dim)
        self._last_stats: dict[str, Any] = {}

    @property
    def last_stats(self) -> dict[str, Any]:
        return dict(self._last_stats)

    def _edges(self, coords: torch.Tensor, batch: torch.Tensor) -> tuple[torch.Tensor, float]:
        if not self.use_geometry:
            return _index_control_edges(batch, self.k_neighbors), 0.0
        if self.knn_backend == "torch_cluster":
            start = time.perf_counter()
            if coords.is_cuda:
                torch.cuda.synchronize(coords.device)
            edge_index = _torch_cluster_knn_graph(
                coords, k=self.k_neighbors, batch=batch, loop=False,
                flow="source_to_target",
            )
            if coords.is_cuda:
                torch.cuda.synchronize(coords.device)
            return edge_index.long(), time.perf_counter() - start
        start = time.perf_counter()
        edge_index = _reference_knn_edges(coords, batch, self.k_neighbors)
        if coords.is_cuda:
            torch.cuda.synchronize(coords.device)
        return edge_index, time.perf_counter() - start

    def _pool_hidden(
        self,
        x: torch.Tensor,
        batch: torch.Tensor,
        ru_offset: torch.Tensor,
        graph_count: int,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Attention-pool final point states using the configured scope.

        ``pool_scope='center'`` masks non-central points *before* the segment
        softmax.  This is intentionally kept as the only difference from the
        historical all-point readout; message-passing has already consumed the
        full packed trimer before this method is called.
        """

        if x.ndim != 2 or batch.ndim != 1 or ru_offset.ndim != 1:
            raise ValueError("pool inputs must be x=[N,H], batch=[N], ru_offset=[N]")
        if x.size(0) != batch.numel() or x.size(0) != ru_offset.numel():
            raise ValueError("pool inputs must share the point count")
        central_mask = ru_offset.eq(0)
        central_counts = scatter(
            central_mask.to(dtype=torch.long), batch,
            dim=0, dim_size=int(graph_count), reduce="sum",
        )
        if self.pool_scope == "center" and bool((central_counts <= 0).any()):
            missing = torch.nonzero(central_counts <= 0, as_tuple=False).flatten().tolist()
            raise ValueError(f"center pooling requires central points for every graph; missing={missing}")

        scores = self.pool_score(torch.tanh(self.pool_projection(x))).squeeze(-1)
        if self.pool_scope == "center":
            # The -inf mask is applied before segment_softmax.  Zeroing after
            # an all-point softmax would leave the central weights summing to
            # less than one and is therefore not equivalent.
            masked_scores = scores.masked_fill(~central_mask, float("-inf"))
            weights = segment_softmax(masked_scores.float(), batch, num_nodes=graph_count).to(x.dtype)
            weights = weights.masked_fill(~central_mask, 0)
            pooling_count = int(central_mask.sum().item())
        else:
            weights = segment_softmax(scores.float(), batch, num_nodes=graph_count).to(x.dtype)
            pooling_count = int(x.size(0))
        pooled = scatter(
            weights.unsqueeze(-1) * x, batch,
            dim=0, dim_size=int(graph_count), reduce="sum",
        )
        weight_sums = scatter(
            weights.float(), batch,
            dim=0, dim_size=int(graph_count), reduce="sum",
        )
        weight_error = (weight_sums - 1.0).abs()
        stats = {
            "pool_scope": self.pool_scope,
            "point_count_used_for_pooling": pooling_count,
            "central_point_count": int(central_mask.sum().item()),
            "central_point_count_per_graph": [int(value) for value in central_counts.detach().cpu().tolist()],
            "central_full_ratio_per_graph": [
                float(central_counts[index].item() / max(1, int((batch == index).sum().item())))
                for index in range(int(graph_count))
            ],
            "pool_weight_sum_min": float(weight_sums.min().detach().cpu()),
            "pool_weight_sum_max": float(weight_sums.max().detach().cpu()),
            "pool_weight_sum_max_error": float(weight_error.max().detach().cpu()),
        }
        return pooled, weights, stats

    def forward(
        self,
        cloud: Any,
        *,
        return_aux: bool = False,
        return_point_states: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
        """Encode a packed cloud, optionally exposing CCDD intermediates.

        The default return contract is unchanged.  ``return_point_states`` is
        a route-local opt-in used by the joint pretraining objective: it
        exposes the final message-passing states and the *same* dynamic kNN
        edge index used by this forward pass.  Keeping this behind an explicit
        flag avoids retaining those tensors in the downstream path.
        """
        coords, z, ru_offset, batch, ptr = _unpack_cloud(cloud)
        parameter = next(self.parameters())
        device = parameter.device
        coords = coords.to(device=device)
        z = z.to(device=device)
        ru_offset = ru_offset.to(device=device)
        batch = batch.to(device=device)
        if ptr is not None:
            ptr = ptr.to(device=device)
            graph_count = int(ptr.numel()) - 1
        else:
            graph_count = int(batch.max().item()) + 1 if batch.numel() else 0
        if graph_count <= 0:
            raise ValueError("packed point cloud must contain at least one graph")
        central_flag = (ru_offset == 0).long()
        initial = self.input_projection(torch.cat((self.element_embedding(z), self.role_embedding(central_flag)), dim=-1))
        edge_index, knn_seconds = self._edges(coords, batch)
        source, destination = edge_index
        if edge_index.numel():
            distances = torch.linalg.vector_norm(coords[destination] - coords[source], dim=-1)
            if self.use_geometry:
                distance_embedding = self.distance_encoder(torch.log1p(distances).unsqueeze(-1))
            else:  # Kept explicit for the A-NG matched-control path.
                distance_embedding = torch.zeros(
                    (edge_index.size(1), self.distance_dim), device=device, dtype=initial.dtype
                )
        else:
            distance_embedding = initial.new_zeros((0, self.distance_dim))
        x = initial
        for layer in self.layers:
            x = layer(x, edge_index, distance_embedding)
        pooled, _weights, pool_stats = self._pool_hidden(x, batch, ru_offset, graph_count)
        output = self.output_projection(pooled)
        finite = bool(torch.isfinite(output).all())
        if edge_index.numel():
            source, destination = edge_index
            cross_to_central = ru_offset[source].ne(0) & ru_offset[destination].eq(0)
            cross_edge_count = int(cross_to_central.sum().item())
            central_receivers = int(torch.unique(destination[cross_to_central]).numel())
        else:
            cross_edge_count = 0
            central_receivers = 0
        self._last_stats = {
            "knn_seconds": float(knn_seconds),
            "edge_count": int(edge_index.size(1)),
            "point_count": int(coords.size(0)),
            "point_count_used_for_message_passing": int(coords.size(0)),
            "graph_count": graph_count,
            "cross_ru_neighbor_edge_count": cross_edge_count,
            "central_nodes_receiving_neighbor_edges": central_receivers,
            "finite": finite,
            "output_norm": output.detach().norm(dim=-1).cpu(),
            "max_abs_output": output.detach().abs().amax(dim=-1).cpu(),
            **pool_stats,
        }
        if not finite:
            raise FloatingPointError("Atomic-PC output contains NaN or Inf")
        if return_aux or return_point_states:
            auxiliary = self.last_stats
            if return_point_states:
                auxiliary = {
                    **auxiliary,
                    # These tensors intentionally remain attached to the
                    # autograd graph.  CCDD must train the message layers,
                    # Center-RU pool and output projection from this exact
                    # noisy-kNN pass.
                    "point_states": x,
                    "edge_index": edge_index,
                    "pool_weights": _weights,
                    "coords": coords,
                    "ru_offset": ru_offset,
                    "batch": batch,
                }
            return output, auxiliary
        return output


__all__ = [
    "ATOMIC_PC_SCHEMA", "ATOMIC_PC_KNN_BACKENDS", "PackedAtomicPointCloud",
    "pack_point_clouds", "count_knn_boundary_ties", "AtomicPointAttentionLayer",
    "AtomicPointEncoder",
]
