"""A small PaiNN-style atom-centred E(3)-equivariant teacher.

This module is deliberately independent of the GLT/BondPath implementation.  It
consumes only atomic numbers, coordinates and a mapping of the canonical centre
RU atoms.  Coordinates are never written by this module and the denoising head
is intentionally excluded from deployment packages.
"""

from __future__ import annotations

from collections import OrderedDict

import torch
from torch import nn
from torch_cluster import radius_graph


def _segment_sum(value: torch.Tensor, index: torch.Tensor, size: int) -> torch.Tensor:
    """Deterministic sum by index (no CUDA atomic ``index_add_``).

    Edges are sorted by ``(target, source)`` before this helper is called.  A
    fixed-order prefix/segment reduction therefore gives identical floating
    point accumulation across a continuous run and a checkpoint resume.
    """
    result = value.new_zeros((size,) + value.shape[1:])
    if value.numel() == 0:
        return result
    unique, counts = torch.unique_consecutive(index, return_counts=True)
    reduced = torch.segment_reduce(value, reduce="sum", lengths=counts)
    return result.index_copy(0, unique, reduced)


class _InvariantMLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value)


class _PaiNNLayer(nn.Module):
    def __init__(self, hidden_channels: int, rbf_dim: int, cutoff: float,
                 edge_chunk_size: int = 32768):
        super().__init__()
        self.hidden_channels = int(hidden_channels)
        self.cutoff = float(cutoff)
        self.edge_chunk_size = int(edge_chunk_size)
        self.filter = nn.Sequential(
            nn.Linear(rbf_dim, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, 3 * hidden_channels),
        )
        context_dim = 4 * hidden_channels
        self.scalar_update = _InvariantMLP(context_dim, hidden_channels, hidden_channels)
        # A scalar gate multiplies vectors channel-wise.  No coordinate-dependent
        # bias is used, so this operation commutes with every SO(3) rotation.
        self.vector_gate = nn.Sequential(
            nn.LayerNorm(context_dim),
            nn.Linear(context_dim, hidden_channels),
            nn.Sigmoid(),
        )
        centers = torch.linspace(0.0, float(cutoff), int(rbf_dim))
        self.register_buffer("rbf_centers", centers)
        self.rbf_gamma = 10.0 / max(float(cutoff) ** 2, 1e-6)

    def _rbf(self, distance: torch.Tensor) -> torch.Tensor:
        value = torch.exp(-self.rbf_gamma * (distance[:, None] - self.rbf_centers[None, :]) ** 2)
        # A smooth polynomial envelope makes the message exactly zero at cutoff.
        scaled = (distance / self.cutoff).clamp_min(0.0)
        envelope = (1.0 - scaled).clamp_min(0.0).square()
        return value * envelope[:, None]

    def forward(self, scalar: torch.Tensor, vector: torch.Tensor,
                pos: torch.Tensor, batch: torch.Tensor, *, cutoff: float,
                max_num_neighbors: int):
        node_count = int(pos.size(0))
        edge_index = radius_graph(
            pos.float(), r=float(cutoff), batch=batch, loop=False,
            max_num_neighbors=int(max_num_neighbors), flow="source_to_target",
        )
        if edge_index.numel():
            source, target = edge_index
            relative = pos[source].float() - pos[target].float()
            distance = relative.norm(dim=-1).clamp_min(1e-8)
            direction = relative / distance[:, None]
            # Canonical edge order removes dependence on the radius-kernel's
            # internal pair enumeration.  Target is primary (for segments),
            # source secondary (for a stable order within each segment).
            order = torch.argsort(target * node_count + source, stable=True)
            source, target = source[order], target[order]
            distance, direction = distance[order], direction[order]
        else:
            source = target = torch.empty(0, dtype=torch.long, device=pos.device)
            distance = direction = pos.new_empty((0,)) if not edge_index.numel() else None

        scalar_message = scalar.new_zeros((node_count, self.hidden_channels))
        vector_message = vector.new_zeros((node_count, self.hidden_channels, 3))
        if edge_index.numel():
            for begin in range(0, int(source.numel()), self.edge_chunk_size):
                end = min(begin + self.edge_chunk_size, int(source.numel()))
                src = source[begin:end]
                dst = target[begin:end]
                unit = direction[begin:end]
                filt = self.filter(self._rbf(distance[begin:end]))
                f_scalar, f_vector, f_mix = filt.chunk(3, dim=-1)
                source_scalar = scalar[src]
                source_vector = vector[src]
                projected = (source_vector * unit[:, None, :]).sum(dim=-1)
                sm = f_scalar * source_scalar + f_vector * projected
                vm = (f_vector * source_scalar)[:, :, None] * unit[:, None, :]
                vm = vm + f_mix[:, :, None] * source_vector
                scalar_message = scalar_message + _segment_sum(sm, dst, node_count)
                vector_message = vector_message + _segment_sum(vm, dst, node_count)

        scalar_norm = vector.norm(dim=-1)
        message_norm = vector_message.norm(dim=-1)
        context = torch.cat(
            [scalar, scalar_message, scalar_norm, message_norm], dim=-1
        )
        scalar = scalar + self.scalar_update(context)
        # Recompute the invariant gate after the scalar residual so the final
        # layer's scalar parameters participate in the denoising objective too.
        gate_context = torch.cat(
            [scalar, scalar_message, vector.norm(dim=-1), message_norm], dim=-1
        )
        vector = vector + vector_message * self.vector_gate(gate_context)[:, :, None]
        return scalar, vector, edge_index


class PolyPaiNNTeacher(nn.Module):
    """Six-layer scalar/vector atom-cloud encoder with a denoising head."""

    architecture_name = "poly_painn_teacher_v1"

    def __init__(self, *, hidden_channels: int = 256, num_layers: int = 6,
                 cutoff: float = 5.0, max_num_neighbors: int = 64,
                 rbf_dim: int = 32, max_atomic_number: int = 118):
        super().__init__()
        if hidden_channels <= 0 or num_layers <= 0 or cutoff <= 0:
            raise ValueError("invalid PolyPaiNN architecture")
        if max_num_neighbors <= 0 or rbf_dim <= 0:
            raise ValueError("invalid radius/RBF configuration")
        self.hidden_channels = int(hidden_channels)
        self.num_layers = int(num_layers)
        self.cutoff = float(cutoff)
        self.max_num_neighbors = int(max_num_neighbors)
        self.rbf_dim = int(rbf_dim)
        self.max_atomic_number = int(max_atomic_number)
        self.element_embedding = nn.Embedding(self.max_atomic_number + 2, hidden_channels)
        self.layers = nn.ModuleList([
            _PaiNNLayer(hidden_channels, rbf_dim, cutoff)
            for _ in range(num_layers)
        ])
        # One scalar coefficient per vector channel produces a physical 3-vector.
        self.noise_head = nn.Linear(hidden_channels, 1, bias=False)

    def _validate_input(self, data):
        for name in ("z", "pos", "batch", "central_index", "central_batch"):
            if not hasattr(data, name):
                raise ValueError(f"teacher input is missing {name}")
        z = data.z.long().reshape(-1)
        pos = data.pos
        batch = data.batch.long().reshape(-1)
        central = data.central_index.long().reshape(-1)
        central_batch = data.central_batch.long().reshape(-1)
        if pos.ndim != 2 or pos.size(-1) != 3 or pos.size(0) != z.numel():
            raise ValueError("teacher positions must be [N,3] and match z")
        if batch.numel() != z.numel() or (batch.numel() and int(batch.min()) < 0):
            raise ValueError("teacher batch is invalid")
        if central.numel() == 0 or (central.numel() and (
                int(central.min()) < 0 or int(central.max()) >= z.numel())):
            raise ValueError("teacher central mapping is invalid")
        if central_batch.numel() != central.numel():
            raise ValueError("teacher central_batch is invalid")
        if z.numel() and (int(z.min()) < 1 or int(z.max()) > self.max_atomic_number):
            raise ValueError("teacher atomic number is outside supported vocabulary")
        return z, pos, batch, central, central_batch

    def forward(self, data):
        z, pos, batch, central, central_batch = self._validate_input(data)
        # All geometry is evaluated in FP32 even when the surrounding model is
        # under BF16 autocast.  This keeps distances and directions well-defined.
        pos_fp32 = pos.float()
        scalar = self.element_embedding(z)
        vector = scalar.new_zeros((z.numel(), self.hidden_channels, 3))
        edge_index = None
        for layer in self.layers:
            scalar, vector, edge_index = layer(
                scalar, vector, pos_fp32, batch,
                cutoff=self.cutoff, max_num_neighbors=self.max_num_neighbors,
            )
        central_scalar = scalar[central]
        central_vector = vector[central]
        graph_count = int(batch.max().item()) + 1 if batch.numel() else 0
        graph_scalar = scalar.new_zeros((graph_count, self.hidden_channels))
        graph_scalar.index_add_(0, central_batch, central_scalar)
        counts = scalar.new_zeros((graph_count,))
        counts.index_add_(0, central_batch, scalar.new_ones((central_batch.numel(),)))
        graph_scalar = graph_scalar / counts.clamp_min(1).unsqueeze(-1)
        predicted_noise = (central_vector * self.noise_head.weight[:, :, None]).sum(dim=1)

        if edge_index is None:
            edge_index = torch.empty((2, 0), dtype=torch.long, device=z.device)
        degree = torch.bincount(edge_index[1], minlength=z.numel())
        histogram = torch.bincount(
            degree.clamp_max(self.max_num_neighbors),
            minlength=self.max_num_neighbors + 1,
        )
        return {
            "central_scalar_states": central_scalar,
            "central_vector_states": central_vector,
            "graph_scalar": graph_scalar,
            "predicted_noise": predicted_noise,
            "neighbor_hist": histogram,
            "neighbor_count": degree,
            "edge_count": torch.tensor(edge_index.size(1), device=z.device),
        }


def teacher_global_objective(local_graph_sums: torch.Tensor,
                             global_graph_count: torch.Tensor | float,
                             world_size: int = 1) -> torch.Tensor:
    """Scale local graph-mean sums for DDP's gradient averaging.

    ``local_graph_sums`` contains one mean-vector-MSE value per local graph.
    The caller all-reduces ``global_graph_count`` across ranks before invoking
    this helper.  Multiplication by ``world_size`` compensates for DDP's
    gradient average and yields the global graph mean.
    """
    if local_graph_sums.ndim != 1 or local_graph_sums.numel() == 0:
        raise ValueError("teacher objective requires non-empty graph sums")
    count = torch.as_tensor(global_graph_count, dtype=local_graph_sums.dtype,
                            device=local_graph_sums.device)
    if count.numel() != 1 or float(count.item()) <= 0 or int(world_size) <= 0:
        raise ValueError("teacher objective has invalid global graph count/world")
    return local_graph_sums.sum() * int(world_size) / count


def teacher_deployment_package(model: PolyPaiNNTeacher, step: int) -> dict:
    state = OrderedDict(
        (name, value.detach().cpu().clone())
        for name, value in model.state_dict().items()
        if not name.startswith("noise_head.")
    )
    return {
        "schema": "poly-painn-teacher-deployment-v1",
        "architecture": model.architecture_name,
        "step": int(step),
        "hyperparameters": {
            "hidden_channels": model.hidden_channels,
            "num_layers": model.num_layers,
            "cutoff": model.cutoff,
            "max_num_neighbors": model.max_num_neighbors,
            "rbf_dim": model.rbf_dim,
            "max_atomic_number": model.max_atomic_number,
        },
        "encoder": state,
    }


def load_teacher_deployment(model: PolyPaiNNTeacher, package: dict, *, expected_step: int | None = None):
    if package.get("schema") != "poly-painn-teacher-deployment-v1":
        raise ValueError("unexpected PolyPaiNN deployment schema")
    if package.get("architecture") != model.architecture_name:
        raise ValueError("PolyPaiNN deployment architecture mismatch")
    if expected_step is not None and int(package.get("step", -1)) != int(expected_step):
        raise ValueError("PolyPaiNN deployment step mismatch")
    if any(str(name).startswith("noise_head.") for name in package.get("encoder", {})):
        raise ValueError("denoising head is not allowed in deployment encoder")
    encoder = package.get("encoder", {})
    expected = {name for name in model.state_dict() if not name.startswith("noise_head.")}
    if set(encoder) != expected:
        raise ValueError("PolyPaiNN deployment encoder keys are incomplete")
    result = model.load_state_dict(encoder, strict=False)
    if set(result.unexpected_keys) or set(result.missing_keys) != {
            name for name in model.state_dict() if name.startswith("noise_head.")
    }:
        raise ValueError("PolyPaiNN deployment contains unexpected backbone keys")
    return model


__all__ = [
    "PolyPaiNNTeacher", "teacher_global_objective", "teacher_deployment_package",
    "load_teacher_deployment",
]
