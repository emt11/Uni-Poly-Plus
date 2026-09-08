"""Graph-only wrapper used by the retained MTS-GLT-v2 baseline."""

from __future__ import annotations

import os
from typing import Optional, Sequence

import torch
from torch import nn

from src.dataset.mips_trimer_contract import ROUTE_INTERNAL, ROUTE_NAME
from .mips_local_graph import MIPSLocalGraphEncoder


SUPPORTED_MODALITIES = ("graph",)
SUPPORTED_FUSION_TYPES = ("none",)


def _model_load_log(message: str) -> None:
    mode = os.environ.get("UNIPOLY_MODEL_LOAD_LOG", "concise").strip().lower()
    if mode not in {"quiet", "concise", "verbose"}:
        raise ValueError("UNIPOLY_MODEL_LOAD_LOG must be quiet, concise, or verbose")
    if mode != "quiet":
        print(message, flush=True)


class UniEncoderAttention(nn.Module):
    """Single graph encoder plus the baseline regression head.

    The historical class name is retained because launcher/checkpoint callers
    import it, but its supported contract is intentionally one graph modality
    and no cross-modal fusion.
    """

    def __init__(
        self,
        joint_embedding_dim: int,
        smiles_model_name: Optional[str],
        gnn_model_name: Optional[str],
        modality_list: Sequence[str],
        output_attention_weights: bool = True,
        freeze_encoder: bool = False,
        graph_num_layers: int = 6,
        graph_emb_dim: int = 512,
        graph_dropout: float = 0.1,
        graph_encoder_type: str = ROUTE_INTERNAL,
        mips_core: str = "paper_corrected",
        mips_max_hops: int = 2,
        mips_use_descriptors: bool = True,
        spatial_mode: str = "trimer_scage",
        graph_geometry_mode: str = "trimer_scage_mcl",
        trimer_num_candidates: int = 4,
        trimer_max_heavy_atoms: int = 384,
        mips_variant: str = "O8",
        mips_atom_feature_mode: str = "mips137",
        mips_attention_scale: str = "head_dim",
        mips_norm_mode: str = "post",
        mips_activation: str = "relu",
        mips_spd_bias_mode: str = "per_head",
        mips_path_bias_mode: str = "per_head_single_path_node",
        mips_multi_scale_hop_gate: bool = False,
        mips_semantics: str = "paper_semantic",
        mips_descriptor_fusion_mode: str = "graph_md_residual",
        mips_descriptor_components: str = "md200",
        mips_backbone_mode: str = "independent",
        mips_mask_mode: str = "zero",
        mips_mask_policy: str = "canonical_exact",
        mips_masked_loss_reduction: str = "atom_mean",
        use_star_rbf: bool = False,
        star_rbf_upper: float = 3.0,
        use_mcl: bool = False,
        topology_attention_variant: str = "o8",
        fusion_type: str = "none",
        head_dropout: float = 0.25,
        **kwargs,
    ):
        super().__init__()
        if tuple(modality_list) != ("graph",):
            raise ValueError(f"{ROUTE_NAME} supports graph modality only")
        if str(fusion_type) != "none":
            raise ValueError(f"{ROUTE_NAME} does not enable cross-modal fusion")
        if str(graph_encoder_type).lower() == "mts":
            graph_encoder_type = ROUTE_INTERNAL
        if str(graph_encoder_type).lower() != ROUTE_INTERNAL:
            raise ValueError(f"graph_encoder_type must be {ROUTE_INTERNAL!r}")
        if kwargs:
            # Reject silently supplied legacy switches instead of allowing an
            # old route to appear active through an ignored argument.
            unknown = ", ".join(sorted(kwargs))
            raise TypeError(f"unsupported baseline encoder options: {unknown}")
        if smiles_model_name not in (None, "") or gnn_model_name not in (None, ""):
            raise ValueError("MTS-GLT-v2 baseline does not load text or alternate GNN encoders")

        self.modality_list = ["graph"]
        self.joint_embedding_dim = int(joint_embedding_dim)
        self.output_attention_weights = bool(output_attention_weights)
        self.freeze_encoder = bool(freeze_encoder)
        self.graph_encoder_type = ROUTE_INTERNAL
        self.fusion_type = "none"
        self.mips_fusion_mode = "none"
        self.projection_mode = "plain"
        self.modality_control = "real"
        self.encoders = nn.ModuleDict({
            "graph": EncoderModule(
                joint_embedding_dim=self.joint_embedding_dim,
                freeze_encoder=self.freeze_encoder,
                graph_num_layers=graph_num_layers,
                graph_emb_dim=graph_emb_dim,
                graph_dropout=graph_dropout,
                mips_core=mips_core,
                mips_max_hops=mips_max_hops,
                mips_use_descriptors=mips_use_descriptors,
                spatial_mode=spatial_mode,
                graph_geometry_mode=graph_geometry_mode,
                trimer_num_candidates=trimer_num_candidates,
                trimer_max_heavy_atoms=trimer_max_heavy_atoms,
                mips_variant=mips_variant,
                mips_atom_feature_mode=mips_atom_feature_mode,
                mips_attention_scale=mips_attention_scale,
                mips_norm_mode=mips_norm_mode,
                mips_activation=mips_activation,
                mips_spd_bias_mode=mips_spd_bias_mode,
                mips_path_bias_mode=mips_path_bias_mode,
                mips_multi_scale_hop_gate=mips_multi_scale_hop_gate,
                mips_semantics=mips_semantics,
                mips_descriptor_fusion_mode=mips_descriptor_fusion_mode,
                mips_descriptor_components=mips_descriptor_components,
                mips_backbone_mode=mips_backbone_mode,
                mips_mask_mode=mips_mask_mode,
                mips_mask_policy=mips_mask_policy,
                mips_masked_loss_reduction=mips_masked_loss_reduction,
                use_star_rbf=use_star_rbf,
                star_rbf_upper=star_rbf_upper,
                use_mcl=use_mcl,
                topology_attention_variant=topology_attention_variant,
            )
        })
        self.fusion_module = None
        self.residual_modality_gates = nn.ParameterDict()
        self.shared_private_projections = nn.ModuleDict()
        self.alignment_projections = nn.ModuleDict()
        self.mlp = nn.Sequential(
            nn.Linear(self.joint_embedding_dim, 128),
            nn.GELU(),
            nn.Dropout(float(head_dropout)),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(float(head_dropout)),
            nn.Linear(64, 1),
        )
        self.modality_heads = nn.ModuleDict()

    def encode_modalities(self, data):
        value = self.encoders["graph"](data)
        self.shared_modality_embeddings = value.unsqueeze(1)
        self.private_modality_embeddings = torch.zeros_like(self.shared_modality_embeddings)
        return self.shared_modality_embeddings

    def fuse_embeddings(self, embeddings, availability_mask=None):
        if embeddings.ndim != 3 or embeddings.size(1) != 1:
            raise ValueError("MTS-GLT-v2 baseline expects one graph embedding")
        return embeddings[:, 0], embeddings.new_ones((embeddings.size(0), 1))

    def encode_global_and_tokens(self, data):
        graph_features, node_features = self.encoders["graph"].encode_global_and_tokens(data)
        graph_features = self.encoders["graph"].projection(
            self.encoders["graph"].norm(graph_features)
        )
        node_features = self.encoders["graph"].projection(
            self.encoders["graph"].norm(node_features)
        )
        return graph_features, node_features, data.batch

    def forward(self, data, availability_mask=None):
        embeddings = self.encode_modalities(data)
        fused, weights = self.fuse_embeddings(embeddings, availability_mask)
        self.unimodal_embedding = fused
        self.attention_visual_weights = weights
        self.attention_visual_labels = ["graph"]
        self.fusion_visual_weights = {"graph": (weights, ["graph"])}
        return self.mlp(fused), embeddings


class EncoderModule(nn.Module):
    def __init__(
        self,
        joint_embedding_dim: int,
        freeze_encoder: bool,
        graph_num_layers: int = 6,
        graph_emb_dim: int = 512,
        graph_dropout: float = 0.1,
        mips_core: str = "paper_corrected",
        mips_max_hops: int = 2,
        mips_use_descriptors: bool = True,
        spatial_mode: str = "trimer_scage",
        graph_geometry_mode: str = "trimer_scage_mcl",
        trimer_num_candidates: int = 4,
        trimer_max_heavy_atoms: int = 384,
        mips_variant: str = "O8",
        mips_atom_feature_mode: str = "mips137",
        mips_attention_scale: str = "head_dim",
        mips_norm_mode: str = "post",
        mips_activation: str = "relu",
        mips_spd_bias_mode: str = "per_head",
        mips_path_bias_mode: str = "per_head_single_path_node",
        mips_multi_scale_hop_gate: bool = False,
        mips_semantics: str = "paper_semantic",
        mips_descriptor_fusion_mode: str = "graph_md_residual",
        mips_descriptor_components: str = "md200",
        mips_backbone_mode: str = "independent",
        mips_mask_mode: str = "zero",
        mips_mask_policy: str = "canonical_exact",
        mips_masked_loss_reduction: str = "atom_mean",
        use_star_rbf: bool = False,
        star_rbf_upper: float = 3.0,
        use_mcl: bool = False,
        topology_attention_variant: str = "o8",
    ):
        super().__init__()
        self.modality = "graph"
        self.encoder = MIPSLocalGraphEncoder(
            core=mips_core,
            num_layer=graph_num_layers,
            emb_dim=graph_emb_dim,
            num_heads=8,
            dropout=graph_dropout,
            max_hops=mips_max_hops,
            use_descriptors=mips_use_descriptors,
            spatial_mode=spatial_mode,
            graph_geometry_mode=graph_geometry_mode,
            trimer_num_candidates=trimer_num_candidates,
            trimer_max_heavy_atoms=trimer_max_heavy_atoms,
            variant=mips_variant,
            atom_feature_mode=mips_atom_feature_mode,
            attention_scale=mips_attention_scale,
            norm_mode=mips_norm_mode,
            activation=mips_activation,
            spd_bias_mode=mips_spd_bias_mode,
            path_bias_mode=mips_path_bias_mode,
            multi_scale_hop_gate=mips_multi_scale_hop_gate,
            semantics=mips_semantics,
            descriptor_fusion_mode=mips_descriptor_fusion_mode,
            descriptor_components=mips_descriptor_components,
            backbone_mode=mips_backbone_mode,
            mask_mode=mips_mask_mode,
            mask_policy=mips_mask_policy,
            masked_loss_reduction=mips_masked_loss_reduction,
            use_star_rbf=use_star_rbf,
            star_rbf_upper=star_rbf_upper,
            use_mcl=use_mcl,
            topology_attention_variant=topology_attention_variant,
        )
        self.norm = nn.LayerNorm(int(graph_emb_dim))
        self.projection = nn.Sequential(
            nn.Linear(int(graph_emb_dim), int(joint_embedding_dim)),
            nn.LayerNorm(int(joint_embedding_dim)),
            nn.ReLU(),
        )
        if freeze_encoder:
            for parameter in self.encoder.parameters():
                parameter.requires_grad = False

    def encode_global_and_tokens(self, data):
        graph_features, node_features = self.encoder(data)
        return graph_features, node_features, data.batch

    def forward(self, data):
        encoded = self.encoder(data)
        graph_features = encoded[0] if isinstance(encoded, tuple) else encoded
        if graph_features.ndim != 2:
            raise ValueError("graph encoder must return [batch, hidden] features")
        return self.projection(self.norm(graph_features))


__all__ = ["SUPPORTED_MODALITIES", "SUPPORTED_FUSION_TYPES", "UniEncoderAttention", "EncoderModule"]
