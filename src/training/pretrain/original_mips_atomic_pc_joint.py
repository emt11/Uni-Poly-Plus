"""Joint O8 + Original-MIPS MD200 + Center-RU Atomic-PC pretraining.

This module owns the route-specific pretraining heads and objective.  The
downstream model remains :class:`OriginalMIPSAtomicPCModel`; the state names
for its three reusable trainable components (``graph_encoder``,
``atomic_point_encoder`` and ``fusion``) are deliberately kept identical so a
joint checkpoint can be loaded without transplanting or silently reinitialising
any of those components.

The MD disturbance below is the historical Original-MIPS continuous-k-vector
operation: selected descriptor positions are replaced with independent
``U[0,1]`` values.  It is intentionally not a new learned corruption or a
fixed-vector ``disturb_fp`` path.
"""

from __future__ import annotations

from typing import Any as TypingAny

import torch
import torch.nn.functional as F
from torch import nn

from src.modules.atomic_point_encoder import (
    AtomicPointEncoder,
    PackedAtomicPointCloud,
)
from src.modules.mips_local_graph import MIPSLocalGraphEncoder
from src.modules.original_mips_knowledge_fusion import OriginalMIPSAttentiveFusion
from src.modules.original_mips_atomic_pc import OriginalMIPSAtomicPCModel
from src.training.pretrain.engine import _joint_canonical_mask


JOINT_PRETRAIN_SCHEMA = "original-mips-atomic-pc-center-ru-joint-pretrain-v1"
JOINT_CHECKPOINT_SCHEMA = "original-mips-atomic-pc-center-ru-joint-checkpoint-v1"
ATOM_TARGET_DIM = 101
ATOM_MASK_RATE = 0.30
MD_KVEC_MASK_RATE = 0.30
COORD_NOISE_SIGMA = 0.20


def disturb_original_mips_md200(
    md: torch.Tensor,
    *,
    rate: float = MD_KVEC_MASK_RATE,
    training: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the unchanged Original-MIPS continuous descriptor disturbance.

    Historical MIPS code uses ``torch.rand_like(md) < rate`` and replaces
    selected MD values with a second independent ``torch.rand_like`` draw.
    Returning the mask makes the audit able to prove the requested rate while
    leaving the numerical operation itself untouched.
    """

    if md.ndim != 2 or int(md.size(1)) != 200:
        raise ValueError(f"MD200 must be [B,200], got {tuple(md.shape)}")
    rate = float(rate)
    if not 0.0 <= rate <= 1.0:
        raise ValueError("MD k-vector mask rate must lie in [0,1]")
    if not training or rate <= 0.0:
        return md, torch.zeros_like(md, dtype=torch.bool)
    mask = torch.rand_like(md) < rate
    disturbed = torch.where(mask, torch.rand_like(md), md)
    return disturbed, mask


class CCDDDistanceHead(nn.Module):
    """Temporary clean-coordinate distance decoder for CCDD."""

    def __init__(self, input_dim: int = 1024) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(input_dim), 256),
            nn.SiLU(),
            nn.Linear(256, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


def _as_bool_vector(value: TypingAny, *, device: torch.device, count: int) -> torch.Tensor:
    result = torch.as_tensor(value, device=device, dtype=torch.bool).reshape(-1)
    if result.numel() != int(count):
        raise ValueError(f"validity vector length {result.numel()} != {count}")
    return result


class OriginalMIPSAtomicPCJointPretrainer(nn.Module):
    """Fixed joint pretraining graph with MAE and CCDD objectives."""

    architecture_name = (
        "Current O8 + Original-MIPS MD200 + Center-RU Atomic-PC-v1 + Original-MIPS KFuse"
    )
    pretrain_schema = JOINT_PRETRAIN_SCHEMA
    knowledge_names = ("md", "atomic_pc")

    def __init__(
        self,
        *,
        graph_encoder: MIPSLocalGraphEncoder | None = None,
        atomic_point_encoder: AtomicPointEncoder | None = None,
        fusion: OriginalMIPSAttentiveFusion | None = None,
        atom_head: nn.Module | None = None,
        ccdd_head: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.graph_encoder = graph_encoder or MIPSLocalGraphEncoder(
            core="paper_corrected",
            num_layer=6,
            emb_dim=512,
            num_heads=8,
            dropout=0.10,
            max_hops=2,
            use_descriptors=True,
            graph_geometry_mode="trimer_scage_mcl",
            use_star_rbf=False,
            use_mcl=False,
            topology_attention_variant="o8",
        )
        self.atomic_point_encoder = atomic_point_encoder or AtomicPointEncoder(
            hidden_dim=256,
            output_dim=512,
            num_layers=4,
            k_neighbors=24,
            use_geometry=True,
            knn_backend="torch_cluster",
            pool_scope="center",
        )
        if str(self.atomic_point_encoder.pool_scope) != "center":
            raise ValueError("joint pretraining requires Atomic-PC pool_scope='center'")
        self.fusion = fusion or OriginalMIPSAttentiveFusion(
            d_model=512,
            knodes=("md", "atomic_pc"),
            knowledge_dims={"md": 200, "atomic_pc": 512},
        )
        if tuple(self.fusion.knodes) != self.knowledge_names:
            raise ValueError("joint pretraining requires the MD200, AtomicPC modality order")
        self.atom_head = atom_head or nn.Linear(512, ATOM_TARGET_DIM)
        self.ccdd_head = ccdd_head or CCDDDistanceHead(1024)

        # These branches exist in the current O8 object for compatibility but
        # are explicitly not part of this route: KFuse is the only MD path,
        # and Star-RBF is off.  Freezing them also keeps the optimizer contract
        # and gradient audit unambiguous.
        for parameter in self.graph_encoder.md_residual.parameters():
            parameter.requires_grad = False
        star_bias = getattr(self.graph_encoder, "star_distance_bias", None)
        if star_bias is not None:
            for parameter in star_bias.parameters():
                parameter.requires_grad = False

    @staticmethod
    def _point_cloud(data: TypingAny) -> PackedAtomicPointCloud:
        return OriginalMIPSAtomicPCModel._point_cloud(data)

    @staticmethod
    def _knowledge_md(data: TypingAny, graph_count: int, device: torch.device) -> torch.Tensor:
        md = getattr(data, "mips_md", None)
        if md is None:
            raise ValueError("joint pretraining batch is missing restored MD200")
        md = torch.as_tensor(md, device=device, dtype=torch.float32)
        if md.ndim != 2 or tuple(md.shape) != (int(graph_count), 200):
            raise ValueError(f"MD200 must be [{graph_count},200], got {tuple(md.shape)}")
        valid_value = getattr(data, "mips_md_valid", None)
        if valid_value is not None:
            valid = _as_bool_vector(valid_value, device=device, count=graph_count)
            md = md * valid.to(md.dtype).unsqueeze(-1)
        if not bool(torch.isfinite(md).all()):
            raise FloatingPointError("MD200 batch contains NaN or Inf")
        return md

    @staticmethod
    def _masked_atom_loss(
        data: TypingAny,
        fused_nodes: torch.Tensor,
        atom_head: nn.Module,
        canonical_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, int, int, torch.Tensor]:
        first = torch.as_tensor(
            data.canonical_first_node_index,
            device=fused_nodes.device,
            dtype=torch.long,
        ).reshape(-1)
        selected = torch.as_tensor(
            canonical_mask,
            device=fused_nodes.device,
            dtype=torch.bool,
        ).reshape(-1)
        if selected.numel() != first.numel():
            raise ValueError("joint MAE mask must be canonical-atom aligned")
        target_indices = first[selected]
        if target_indices.numel() == 0:
            zero = fused_nodes.sum() * 0.0
            return zero, 0, 0, fused_nodes.new_empty((0, ATOM_TARGET_DIM))
        labels = torch.as_tensor(data.mips_x, device=fused_nodes.device)[
            target_indices, :ATOM_TARGET_DIM
        ].argmax(dim=-1).long()
        logits = atom_head(fused_nodes[target_indices])
        if logits.ndim != 2 or int(logits.size(1)) != ATOM_TARGET_DIM:
            raise ValueError("joint MAE atom head must output 101 classes")
        loss = F.cross_entropy(logits.float(), labels, reduction="mean")
        correct = int((logits.detach().argmax(dim=-1) == labels).sum().item())
        return loss, int(labels.numel()), correct, logits

    def forward(
        self,
        data: TypingAny,
        *,
        atom_mask: torch.Tensor | None = None,
        seed: int = 42,
        stream_step: int = 0,
        atom_mask_rate: float = ATOM_MASK_RATE,
        md_kvec_mask_rate: float = MD_KVEC_MASK_RATE,
        coord_noise_sigma: float = COORD_NOISE_SIGMA,
    ) -> dict[str, TypingAny]:
        parameter = next(self.parameters())
        device = parameter.device
        if atom_mask is None:
            atom_mask = _joint_canonical_mask(
                data, int(seed), int(stream_step), float(atom_mask_rate)
            )
        canonical_mask = torch.as_tensor(
            atom_mask, device=device, dtype=torch.bool
        ).reshape(-1)

        # O8 receives the canonical/lifted node mask.  The production
        # canonical-periodic batches have one canonical node per lifted atom;
        # the index form below also handles a canonical mask explicitly.
        first = torch.as_tensor(
            data.canonical_first_node_index, device=device, dtype=torch.long
        ).reshape(-1)
        if canonical_mask.numel() != first.numel():
            raise ValueError("atom_mask does not match canonical_first_node_index")
        node_mask = torch.zeros(
            int(data.mips_x.size(0)), dtype=torch.bool, device=device
        )
        node_mask[first[canonical_mask]] = True
        _o8_graph, o8_node_states = self.graph_encoder._forward_impl(
            data, atom_mask=node_mask, use_star=False, use_md=False
        )
        if o8_node_states.ndim != 2 or int(o8_node_states.size(1)) != 512:
            raise ValueError("O8 node states must be [N,512]")

        # The only coordinate corruption is additive Gaussian noise.  kNN is
        # rebuilt by AtomicPointEncoder from this noisy tensor below; no clean
        # edge index is constructed or passed to message passing.
        clean_cloud = self._point_cloud(data)
        clean_coords = clean_cloud.coords.to(device=device, dtype=torch.float32)
        sigma = float(coord_noise_sigma)
        if sigma < 0.0:
            raise ValueError("coordinate noise sigma must be non-negative")
        noisy_coords = clean_coords + torch.randn_like(clean_coords) * sigma
        noisy_cloud = PackedAtomicPointCloud(
            coords=noisy_coords,
            atomic_number=clean_cloud.atomic_number.to(device=device),
            ru_offset=clean_cloud.ru_offset.to(device=device),
            batch=clean_cloud.batch.to(device=device),
            ptr=None if clean_cloud.ptr is None else clean_cloud.ptr.to(device=device),
            sample_keys=clean_cloud.sample_keys,
            source_smiles=clean_cloud.source_smiles,
        )
        atomic_pc, atomic_aux = self.atomic_point_encoder(
            noisy_cloud, return_point_states=True
        )
        if tuple(atomic_pc.shape) != (int(_o8_graph.size(0)), 512):
            raise ValueError("AtomicPC512 graph shape does not match O8 graph count")
        edge_index = atomic_aux["edge_index"].long()
        point_states = atomic_aux["point_states"]
        if point_states.ndim != 2 or int(point_states.size(1)) != 256:
            raise ValueError("Atomic-PC point states must be [N,256]")
        batch_index = torch.as_tensor(
            data.batch, device=device, dtype=torch.long
        ).reshape(-1)
        if batch_index.numel() != o8_node_states.size(0):
            raise ValueError("O8 node/batch index length mismatch")

        clean_md = self._knowledge_md(data, int(_o8_graph.size(0)), device)
        disturbed_md, md_mask = disturb_original_mips_md200(
            clean_md,
            rate=float(md_kvec_mask_rate),
            training=self.training,
        )
        self.fusion.reset_trace()
        fused_nodes = self.fusion(
            o8_node_states,
            {"md": disturbed_md, "atomic_pc": atomic_pc},
            batch_index,
        )
        mae_loss, atom_count, atom_correct, atom_logits = self._masked_atom_loss(
            data, fused_nodes, self.atom_head, canonical_mask
        )

        source, destination = edge_index
        ru_offset = clean_cloud.ru_offset.to(device=device, dtype=torch.long)
        central_receiver = ru_offset[destination].eq(0) if edge_index.numel() else torch.zeros(
            (0,), dtype=torch.bool, device=device
        )
        if bool(central_receiver.any()):
            selected_source = source[central_receiver]
            selected_destination = destination[central_receiver]
            edge_features = torch.cat(
                (
                    point_states[selected_source] + point_states[selected_destination],
                    (point_states[selected_source] - point_states[selected_destination]).abs(),
                    atomic_pc[clean_cloud.batch.to(device=device)[selected_destination]],
                ),
                dim=-1,
            )
            ccdd_pred = self.ccdd_head(edge_features)
            ccdd_target = torch.linalg.vector_norm(
                clean_coords[selected_destination] - clean_coords[selected_source], dim=-1
            )
            ccdd_loss = F.l1_loss(ccdd_pred.float(), ccdd_target.float(), reduction="mean")
        else:
            ccdd_pred = point_states.new_empty((0,))
            ccdd_target = point_states.new_empty((0,))
            ccdd_loss = point_states.sum() * 0.0
        total_loss = mae_loss + ccdd_loss
        if not bool(torch.isfinite(total_loss).all()):
            raise FloatingPointError("joint MAE+CCDD loss is non-finite")
        attention = self.fusion.last_attention_weights
        return {
            "loss": total_loss,
            "total_loss": total_loss,
            "mae_loss": mae_loss,
            "ccdd_loss": ccdd_loss,
            "atom_count": int(atom_count),
            "atom_correct": int(atom_correct),
            "atom_logits": atom_logits,
            "canonical_mask": canonical_mask,
            "node_mask": node_mask,
            "md_mask": md_mask,
            "clean_md": clean_md,
            "disturbed_md": disturbed_md,
            "md_mask_fraction": float(md_mask.float().mean().detach().cpu())
            if md_mask.numel() else 0.0,
            "ccdd_pred": ccdd_pred,
            "ccdd_target": ccdd_target,
            "ccdd_edge_count": int(central_receiver.sum().item()),
            "edge_index": edge_index,
            "point_states": point_states,
            "atomic_pc": atomic_pc,
            "o8_graph": _o8_graph,
            "o8_node_states": o8_node_states,
            "fused_node_states": fused_nodes,
            "fusion_attention": attention,
            "fusion_call_count": int(self.fusion.fusion_call_count),
            "noisy_coords": noisy_coords,
            "clean_coords": clean_coords,
            "atomic_aux": atomic_aux,
        }


def trainable_group_parameters(
    model: OriginalMIPSAtomicPCJointPretrainer,
) -> dict[str, list[nn.Parameter]]:
    """Return the exact gradient-audit groups used by the route."""

    return {
        "o8": [p for p in model.graph_encoder.parameters() if p.requires_grad],
        "atomic_pc": [p for p in model.atomic_point_encoder.parameters() if p.requires_grad],
        "kfuse": [p for p in model.fusion.parameters() if p.requires_grad],
        "atom_head": [p for p in model.atom_head.parameters() if p.requires_grad],
        "ccdd_head": [p for p in model.ccdd_head.parameters() if p.requires_grad],
    }


def gradient_summary(parameters: list[nn.Parameter]) -> dict[str, TypingAny]:
    gradients = [p.grad for p in parameters]
    finite = bool(gradients) and all(
        g is not None and bool(torch.isfinite(g).all()) for g in gradients
    )
    nonzero = bool(gradients) and any(
        g is not None and float(g.detach().abs().sum().cpu()) > 0.0 for g in gradients
    )
    norm = 0.0
    finite_gradients = [gradient for gradient in gradients if gradient is not None]
    if finite_gradients:
        norm = float(
            torch.sqrt(
                sum(
                    (gradient.detach().float().pow(2).sum() for gradient in finite_gradients),
                    torch.zeros((), device=finite_gradients[0].device),
                )
            ).cpu()
        )
    return {
        "parameter_count": int(sum(p.numel() for p in parameters)),
        "parameter_tensors": int(len(parameters)),
        "all_finite": bool(finite),
        "any_nonzero": bool(nonzero),
        "l2_norm": float(norm),
    }


__all__ = [
    "JOINT_PRETRAIN_SCHEMA",
    "JOINT_CHECKPOINT_SCHEMA",
    "ATOM_TARGET_DIM",
    "ATOM_MASK_RATE",
    "MD_KVEC_MASK_RATE",
    "COORD_NOISE_SIGMA",
    "disturb_original_mips_md200",
    "CCDDDistanceHead",
    "OriginalMIPSAtomicPCJointPretrainer",
    "trainable_group_parameters",
    "gradient_summary",
]
