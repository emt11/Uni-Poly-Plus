import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
from transformers import RobertaModel
from transformers.utils import logging as transformers_logging
from src.dataset.mips_trimer_contract import ROUTE_INTERNAL, ROUTE_NAME

from .mips_local_graph import MIPSLocalGraphEncoder


SUPPORTED_MODALITIES = ('smiles', 'graph', 'fp')
SUPPORTED_FUSION_TYPES = ('none', 'zero_gated_residual')


def _model_load_log_mode():
    mode = os.environ.get("UNIPOLY_MODEL_LOAD_LOG", "concise").strip().lower()
    if mode not in {"quiet", "concise", "verbose"}:
        raise ValueError(
            "UNIPOLY_MODEL_LOAD_LOG must be quiet, concise, or verbose"
        )
    return mode


def _model_load_log(message):
    if _model_load_log_mode() != "quiet":
        print(message, flush=True)


def _load_roberta_encoder(model_name):
    """Load the encoder without repeating expected MLM-head diagnostics."""

    if _model_load_log_mode() == "verbose":
        return RobertaModel.from_pretrained(model_name)
    previous_verbosity = transformers_logging.get_verbosity()
    progress_was_enabled = transformers_logging.is_progress_bar_enabled()
    try:
        transformers_logging.set_verbosity_error()
        transformers_logging.disable_progress_bar()
        return RobertaModel.from_pretrained(model_name)
    finally:
        transformers_logging.set_verbosity(previous_verbosity)
        if progress_was_enabled:
            transformers_logging.enable_progress_bar()


class SharedPrivateProjection(nn.Module):
    """Split a modality into aligned shared and modality-private halves."""

    def __init__(self, dim):
        super().__init__()
        if int(dim) % 2:
            raise ValueError("shared/private projection requires an even dimension")
        self.half_dim = int(dim) // 2
        self.shared = nn.Sequential(
            nn.LayerNorm(int(dim)),
            nn.Linear(int(dim), self.half_dim),
            nn.GELU(),
        )
        self.private = nn.Sequential(
            nn.LayerNorm(int(dim)),
            nn.Linear(int(dim), self.half_dim),
            nn.GELU(),
        )
        self.output_norm = nn.LayerNorm(int(dim))

    def forward(self, embedding):
        shared = self.shared(embedding)
        private = self.private(embedding)
        combined = self.output_norm(torch.cat([shared, private], dim=-1))
        return combined, shared, private


class LoRALinear(nn.Module):
    """Minimal frozen-base LoRA adapter used by MTS SMILES experiments."""

    def __init__(self, base, rank=8, alpha=16.0, dropout=0.05):
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError("LoRALinear requires nn.Linear")
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad = False
        self.lora_a = nn.Parameter(torch.empty(int(rank), base.in_features))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, int(rank)))
        nn.init.kaiming_uniform_(self.lora_a, a=5 ** 0.5)
        self.scale = float(alpha) / float(rank)
        self.dropout = nn.Dropout(float(dropout))

    def forward(self, values):
        update = F.linear(F.linear(self.dropout(values), self.lora_a), self.lora_b)
        return self.base(values) + self.scale * update


def inject_roberta_lora(encoder, last_layers=4):
    """Inject rank-8 LoRA into Q/K/V/output projections of final layers."""
    layers = list(encoder.encoder.layer)
    for layer in layers[-int(last_layers):]:
        attention = layer.attention
        for parent, name in (
            (attention.self, "query"), (attention.self, "key"),
            (attention.self, "value"), (attention.output, "dense"),
        ):
            setattr(parent, name, LoRALinear(getattr(parent, name)))


class UniEncoderAttention(nn.Module):
    def __init__(
        self,
        joint_embedding_dim: int,
        smiles_model_name: Optional[str],
        gnn_model_name: Optional[str],
        modality_list: List[str],
        output_attention_weights: bool = True,
        freeze_encoder: bool = False,
        num_heads: int = 8,
        ff_dim: Optional[int] = None,
        dropout: float = 0.1,
        graph_num_layers: int = 6,
        graph_emb_dim: int = 256,
        graph_dropout: float = 0.1,
        graph_encoder_type: str = ROUTE_INTERNAL,
        scage_dist_bar=None,
        scage_num_heads: int = 16,
        scage_ffn_hidden_dim: int = 256,
        scage_num_kernels: int = 128,
        scage_attention_dropout: float = 0.1,
        scage_use_pbc_distance: bool = True,
        scage_use_descriptors: bool = False,
        scage_distance_mode: str = 'mips_dual',
        scage_distance_rbf: int = 32,
        scage_distance_cutoff: float = 12.0,
        scage_distance_scales=(4.0, 8.0, 12.0),
        scage_distance_taus=(0.5, 1.0, 1.5),
        scage_topology_bias: bool = True,
        scage_topology_max_distance: int = 20,
        scage_topology_locality_mode: str = 'soft',
        scage_topology_locality_threshold: int = 5,
        scage_topology_locality_tau: float = 1.0,
        scage_periodic_image_mode: str = 'explicit_images',
        scage_periodic_image_cap: int = 1,
        scage_periodic_image_temperature: float = 0.5,
        scage_force_topology_only: bool = False,
        mips_core: str = "topology_plus",
        mips_max_hops: Optional[int] = None,
        mips_use_descriptors: bool = False,
        spatial_mode: str = "trimer_scage",
        graph_geometry_mode: str = "trimer_scage_mcl",
        mcl_distance_percentiles=(0.20, 0.50),
        trimer_num_candidates: int = 4,
        trimer_max_heavy_atoms: int = 384,
        mips_variant: str = None,
        mips_fusion_mode: str = "none",
        projection_mode: str = "shared_private",
        modality_control: str = "real",
        controlled_modality: Optional[str] = None,
        mips_atom_feature_mode: str = None,
        mips_attention_scale: str = None,
        mips_norm_mode: str = None,
        mips_activation: str = None,
        mips_spd_bias_mode: str = None,
        mips_path_bias_mode: str = None,
        mips_multi_scale_hop_gate: Optional[bool] = None,
        mips_semantics: str = None,
        mips_descriptor_fusion_mode: str = "graph_md_residual",
        mips_descriptor_components: str = "md200",
        mips_descriptor_disturbance: float = 0.0,
        mips_backbone_mode: str = None,
        mips_input_norm: bool = None,
        mips_mask_mode: str = None,
        mips_mask_policy: str = None,
        mips_masked_loss_reduction: str = None,
        use_star_rbf: bool = True,
        use_mcl: bool = True,
        mcl_mask_mode: str = "real",
        fusion_type: str = 'none',
        fp_mode: str = 'ecfp',
        fusion_dropout: Optional[float] = None,
        head_dropout: float = 0.25,
        fp_bit_dropout: float = 0.0,
        modality_dropout=None,
        alignment_projection_dim: int = 256,
        unimodal_auxiliary: bool = False,
        cross_task_auxiliary_tasks=None,
    ):
        super().__init__()
        unsupported = [modality for modality in modality_list if modality not in SUPPORTED_MODALITIES]
        if unsupported:
            raise ValueError(
                f"Unsupported modality: {unsupported[0]}. Current supported modalities are: "
                f"{', '.join(SUPPORTED_MODALITIES)}."
            )
        if not modality_list:
            raise ValueError("At least one modality must be enabled.")
        self.modality_list = modality_list
        self.joint_embedding_dim = joint_embedding_dim
        self.output_attention_weights = output_attention_weights
        self.freeze_encoder = freeze_encoder
        if fusion_type not in SUPPORTED_FUSION_TYPES:
            raise ValueError(
                f"Unsupported fusion_type: {fusion_type}. Supported values are: "
                f"{', '.join(SUPPORTED_FUSION_TYPES)}."
            )
        self.fusion_type = fusion_type
        self.mips_fusion_mode = str(mips_fusion_mode)
        self.projection_mode = str(projection_mode)
        self.modality_control = str(modality_control)
        self.controlled_modality = controlled_modality
        if self.mips_fusion_mode != "none":
            raise ValueError(
                "MIPS-Trimer-SCAGE uses a single graph path; parallel fusion "
                "is not an active model component."
            )
        if self.projection_mode not in {"plain", "shared_private"}:
            raise ValueError("projection_mode must be plain or shared_private")
        if self.modality_control not in {
            "real", "batch_shuffled", "constant_zero"
        }:
            raise ValueError("unsupported modality control")
        self.graph_encoder_type = str(graph_encoder_type).lower()
        if self.graph_encoder_type == "mts":
            self.graph_encoder_type = ROUTE_INTERNAL
        self.modality_dropout = dict(modality_dropout or {})
        if self.mips_fusion_mode != "none":
            raise ValueError(
                "The retired parallel/self-attention fusion path is not available; "
                "use mips_fusion_mode='none' with none or zero_gated_residual."
            )

        self.encoders = nn.ModuleDict({
            modality: EncoderModule(
                modality=modality,
                joint_embedding_dim=joint_embedding_dim,
                freeze_encoder=freeze_encoder,
                smiles_model_name=smiles_model_name,
                gnn_model_name=gnn_model_name,
                graph_num_layers=graph_num_layers,
                graph_emb_dim=graph_emb_dim,
                graph_dropout=graph_dropout,
                graph_encoder_type=graph_encoder_type,
                scage_dist_bar=scage_dist_bar,
                scage_num_heads=scage_num_heads,
                scage_ffn_hidden_dim=scage_ffn_hidden_dim,
                scage_num_kernels=scage_num_kernels,
                scage_attention_dropout=scage_attention_dropout,
                scage_use_pbc_distance=scage_use_pbc_distance,
                scage_use_descriptors=scage_use_descriptors,
                scage_distance_mode=scage_distance_mode,
                scage_distance_rbf=scage_distance_rbf,
                scage_distance_cutoff=scage_distance_cutoff,
                scage_distance_scales=scage_distance_scales,
                scage_distance_taus=scage_distance_taus,
                scage_topology_bias=scage_topology_bias,
                scage_topology_max_distance=scage_topology_max_distance,
                scage_topology_locality_mode=scage_topology_locality_mode,
                scage_topology_locality_threshold=scage_topology_locality_threshold,
                scage_topology_locality_tau=scage_topology_locality_tau,
                scage_periodic_image_mode=scage_periodic_image_mode,
                scage_periodic_image_cap=scage_periodic_image_cap,
                scage_periodic_image_temperature=scage_periodic_image_temperature,
                scage_force_topology_only=scage_force_topology_only,
                mips_core=mips_core,
                mips_max_hops=mips_max_hops,
                mips_use_descriptors=mips_use_descriptors,
                spatial_mode=spatial_mode,
                graph_geometry_mode=graph_geometry_mode,
                mcl_distance_percentiles=mcl_distance_percentiles,
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
                mips_descriptor_disturbance=mips_descriptor_disturbance,
                mips_backbone_mode=mips_backbone_mode,
                mips_input_norm=mips_input_norm,
                mips_mask_mode=mips_mask_mode,
                mips_mask_policy=mips_mask_policy,
                mips_masked_loss_reduction=mips_masked_loss_reduction,
                use_star_rbf=use_star_rbf,
                use_mcl=use_mcl,
                mcl_mask_mode=mcl_mask_mode,
                fp_mode=fp_mode,
                fp_bit_dropout=fp_bit_dropout,
                low_capacity_adapter=(
                    self.fusion_type == 'zero_gated_residual'
                    and modality in {'smiles', 'fp'}
                ),
            )
            for modality in modality_list
        })
        if self.fusion_type == 'zero_gated_residual' and 'smiles' in self.encoders:
            inject_roberta_lora(self.encoders['smiles'].encoder, last_layers=4)

        ff_dim = ff_dim or (joint_embedding_dim * 2)
        if self.fusion_type == "none":
            if list(modality_list) != ["graph"]:
                raise ValueError(
                    "fusion_type='none' is reserved for the graph-only "
                    f"{ROUTE_NAME} route"
                )
            self.fusion_module = None
        elif self.fusion_type == "zero_gated_residual":
            if 'graph' not in modality_list:
                raise ValueError("zero_gated_residual requires a graph anchor")
            self.fusion_module = None
        self.residual_modality_gates = nn.ParameterDict({
            name: nn.Parameter(torch.zeros(()))
            for name in modality_list if name != 'graph'
        }) if self.fusion_type == 'zero_gated_residual' else nn.ParameterDict()

        self.shared_private_projections = nn.ModuleDict()
        if self.projection_mode == "shared_private":
            self.shared_private_projections.update({
                name: SharedPrivateProjection(joint_embedding_dim)
                for name in modality_list
            })

        projection_dim = int(alignment_projection_dim)
        self.alignment_projections = nn.ModuleDict()
        if self.fusion_type not in {"none", "zero_gated_residual"}:
            for name in tuple(modality_list) + ('fusion',):
                input_dim = (
                    joint_embedding_dim
                    if name == "fusion" or self.projection_mode == "plain"
                    else joint_embedding_dim // 2
                )
                self.alignment_projections[name] = nn.Sequential(
                    nn.Linear(input_dim, joint_embedding_dim),
                    nn.GELU(),
                    nn.Linear(joint_embedding_dim, projection_dim),
                )

        self.mlp = nn.Sequential(
            nn.Linear(joint_embedding_dim, 128),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(64, 1),
        )
        # Optional deep supervision for Stage 3. These heads regularize each
        # modality representation; the fused head remains the inference path.
        self.modality_heads = nn.ModuleDict()
        if unimodal_auxiliary:
            self.modality_heads.update({
                name: nn.Sequential(
                    nn.LayerNorm(joint_embedding_dim),
                    nn.Linear(joint_embedding_dim, 64),
                    nn.GELU(),
                    nn.Dropout(head_dropout),
                    nn.Linear(64, 1),
                )
                for name in modality_list
            })
        self.cross_task_auxiliary_tasks = tuple(cross_task_auxiliary_tasks or ())
        self.cross_task_aux_heads = nn.ModuleDict({
            name: nn.Sequential(
                nn.LayerNorm(joint_embedding_dim),
                nn.Linear(joint_embedding_dim, 64),
                nn.GELU(),
                nn.Dropout(head_dropout),
                nn.Linear(64, 1),
            )
            for name in self.cross_task_auxiliary_tasks
        })

    def encode_modalities(self, data):
        encoded = []
        shared_parts = []
        private_parts = []
        for name in self.modality_list:
            value = self.encoders[name](data)
            control_applies = (
                self.modality_control != "real"
                and (
                    name == self.controlled_modality
                    if self.controlled_modality is not None
                    else name != "graph"
                )
            )
            if control_applies:
                if self.modality_control == "constant_zero":
                    value = torch.zeros_like(value)
                elif value.size(0) > 1:
                    value = value.roll(shifts=1, dims=0)
            if self.projection_mode == "shared_private":
                value, shared, private = self.shared_private_projections[name](value)
            else:
                shared = value
                private = torch.zeros_like(value)
            encoded.append(value)
            shared_parts.append(shared)
            private_parts.append(private)
        self.shared_modality_embeddings = torch.stack(shared_parts, dim=1)
        self.private_modality_embeddings = torch.stack(private_parts, dim=1)
        return torch.stack(encoded, dim=1)

    def split_shared_private(self, name, embedding):
        if self.projection_mode == "plain":
            return embedding, embedding, torch.zeros_like(embedding)
        if name not in self.shared_private_projections:
            raise KeyError(f"No shared/private projection exists for {name!r}")
        return self.shared_private_projections[name](embedding)

    def intrinsic_availability_mask(self, data, device=None):
        """Return modalities that physically exist for each sample.

        SMILES and fingerprints are always available in the current dataset.
        The finite non-PBC graph builder may reject an over-long repeat unit;
        those graph placeholders must never participate in fusion or alignment.
        """
        device = device or next(self.parameters()).device
        batch_size = len(data.smiles)
        mask = torch.ones(
            batch_size, len(self.modality_list), dtype=torch.bool, device=device
        )
        if 'graph' in self.modality_list:
            available = getattr(data, 'graph_available', None)
            if available is not None:
                available = torch.as_tensor(available, device=device).view(-1).bool()
                if available.numel() != batch_size:
                    raise ValueError(
                        "graph_available must contain one value per graph: "
                        f"{available.numel()} vs {batch_size}"
                    )
                mask[:, self.modality_list.index('graph')] = available
        for name, field in (
            ('smiles', 'smiles_available'), ('fp', 'fp_available')
        ):
            if name in self.modality_list and hasattr(data, field):
                available = torch.as_tensor(
                    getattr(data, field), device=device
                ).view(-1).bool()
                if available.numel() != batch_size:
                    raise ValueError(
                        f"{field} must contain one value per graph"
                    )
                mask[:, self.modality_list.index(name)] = available
        return mask

    def sample_availability_mask(
        self, batch_size, drop_probabilities=None, min_available=2, device=None,
        intrinsic_mask=None,
    ):
        probabilities = drop_probabilities if drop_probabilities is not None else self.modality_dropout
        device = device or next(self.parameters()).device
        if intrinsic_mask is None:
            intrinsic_mask = torch.ones(
                batch_size, len(self.modality_list), dtype=torch.bool, device=device
            )
        else:
            intrinsic_mask = intrinsic_mask.to(device=device, dtype=torch.bool)
        mask = intrinsic_mask.clone()
        if not probabilities or len(self.modality_list) < min_available:
            return mask
        for idx, name in enumerate(self.modality_list):
            probability = (
                0.0 if self.fusion_type == 'zero_gated_residual' and name == 'graph'
                else float(probabilities.get(name, 0.0))
            )
            if probability > 0:
                mask[:, idx] &= torch.rand(batch_size, device=device) >= probability
        for row in range(batch_size):
            required = min(min_available, int(intrinsic_mask[row].sum().item()))
            missing = required - int(mask[row].sum().item())
            if missing > 0:
                candidates = torch.nonzero(
                    intrinsic_mask[row] & ~mask[row], as_tuple=False
                ).flatten()
                restore = candidates[
                    torch.randperm(candidates.numel(), device=device)[:missing]
                ]
                mask[row, restore] = True
        return mask

    def fuse_embeddings(self, embeddings, availability_mask=None):
        if self.fusion_type == 'none':
            if embeddings.ndim != 3 or embeddings.size(1) != 1:
                raise ValueError("fusion_type='none' requires one graph modality")
            weights = embeddings.new_ones((embeddings.size(0), 1))
            return embeddings[:, 0], weights
        if self.fusion_type == 'zero_gated_residual':
            graph_index = self.modality_list.index('graph')
            fused = embeddings[:, graph_index]
            weights = embeddings.new_zeros(
                (embeddings.size(0), len(self.modality_list))
            )
            weights[:, graph_index] = 1.0
            for index, name in enumerate(self.modality_list):
                if name == 'graph':
                    continue
                gate = torch.tanh(self.residual_modality_gates[name])
                available = (
                    availability_mask[:, index].to(embeddings.dtype)
                    if availability_mask is not None
                    else embeddings.new_ones(embeddings.size(0))
                )
                coefficient = gate * available
                fused = fused + coefficient.unsqueeze(-1) * embeddings[:, index]
                weights[:, index] = coefficient
            return fused, weights
        raise RuntimeError(
            "Unsupported fusion path; MTS only supports none and "
            "zero_gated_residual."
        )

    def project_alignment(self, name, embedding):
        if name not in self.alignment_projections:
            raise KeyError(f"No alignment projection exists for {name!r}")
        return self.alignment_projections[name](embedding)

    def predict_modalities(self, embeddings):
        """Return per-modality predictions in the same order as modality_list."""
        if len(self.modality_heads) != len(self.modality_list):
            raise RuntimeError("Unimodal auxiliary heads were not enabled for this model")
        if embeddings.ndim != 3 or embeddings.size(1) != len(self.modality_list):
            raise ValueError(
                "Expected modality embeddings with shape [batch, num_modalities, dim]"
            )
        return torch.stack([
            self.modality_heads[name](embeddings[:, idx])
            for idx, name in enumerate(self.modality_list)
        ], dim=1)

    def predict_cross_tasks(self, fused_embedding):
        if not self.cross_task_auxiliary_tasks:
            return fused_embedding.new_empty((fused_embedding.size(0), 0))
        return torch.cat([
            self.cross_task_aux_heads[name](fused_embedding)
            for name in self.cross_task_auxiliary_tasks
        ], dim=1)

    def forward(self, data, availability_mask=None):
        embeddings = self.encode_modalities(data)
        if self.fusion_type == 'zero_gated_residual':
            intrinsic_mask = self.intrinsic_availability_mask(
                data, device=embeddings.device
            )
            if availability_mask is None and self.training:
                availability_mask = self.sample_availability_mask(
                    embeddings.size(0), device=embeddings.device,
                    intrinsic_mask=intrinsic_mask,
                    min_available=(
                        1 if self.fusion_type == 'zero_gated_residual' else 2
                    ),
                )
            elif availability_mask is None:
                availability_mask = intrinsic_mask
            else:
                availability_mask = availability_mask.to(
                    device=embeddings.device, dtype=torch.bool
                ) & intrinsic_mask

        fused_output, modality_attention = self.fuse_embeddings(embeddings, availability_mask)
        visual_name = 'zero_gated_residual' if self.fusion_type == 'zero_gated_residual' else 'graph'
        self.unimodal_embedding = fused_output
        self.attention_visual_weights = modality_attention
        self.attention_visual_labels = list(self.modality_list)
        self.fusion_visual_weights = {
            visual_name: (modality_attention, list(self.modality_list)),
        }

        output = self.mlp(fused_output)
        return output, embeddings


class EncoderModule(nn.Module):
    def __init__(
        self,
        modality: str,
        joint_embedding_dim: int,
        freeze_encoder: bool,
        smiles_model_name: Optional[str] = None,
        gnn_model_name: Optional[str] = None,
        graph_num_layers: int = 6,
        graph_emb_dim: int = 256,
        graph_dropout: float = 0.1,
        graph_encoder_type: str = ROUTE_INTERNAL,
        scage_dist_bar=None,
        scage_num_heads: int = 16,
        scage_ffn_hidden_dim: int = 256,
        scage_num_kernels: int = 128,
        scage_attention_dropout: float = 0.1,
        scage_use_pbc_distance: bool = True,
        scage_use_descriptors: bool = False,
        scage_distance_mode: str = 'mips_dual',
        scage_distance_rbf: int = 32,
        scage_distance_cutoff: float = 12.0,
        scage_distance_scales=(4.0, 8.0, 12.0),
        scage_distance_taus=(0.5, 1.0, 1.5),
        scage_topology_bias: bool = True,
        scage_topology_max_distance: int = 20,
        scage_topology_locality_mode: str = 'soft',
        scage_topology_locality_threshold: int = 5,
        scage_topology_locality_tau: float = 1.0,
        scage_periodic_image_mode: str = 'explicit_images',
        scage_periodic_image_cap: int = 1,
        scage_periodic_image_temperature: float = 0.5,
        scage_force_topology_only: bool = False,
        mips_core: str = "topology_plus",
        mips_max_hops: Optional[int] = None,
        mips_use_descriptors: bool = False,
        spatial_mode: str = "trimer_scage",
        graph_geometry_mode: str = "trimer_scage_mcl",
        mcl_distance_percentiles=(0.20, 0.50),
        trimer_num_candidates: int = 4,
        trimer_max_heavy_atoms: int = 384,
        mips_variant: str = None,
        mips_atom_feature_mode: str = None,
        mips_attention_scale: str = None,
        mips_norm_mode: str = None,
        mips_activation: str = None,
        mips_spd_bias_mode: str = None,
        mips_path_bias_mode: str = None,
        mips_multi_scale_hop_gate: Optional[bool] = None,
        mips_semantics: str = None,
        mips_descriptor_fusion_mode: str = "graph_md_residual",
        mips_descriptor_components: str = "md200",
        mips_descriptor_disturbance: float = 0.0,
        mips_backbone_mode: str = None,
        mips_input_norm: bool = None,
        mips_mask_mode: str = None,
        mips_mask_policy: str = None,
        mips_masked_loss_reduction: str = None,
        use_star_rbf: bool = True,
        use_mcl: bool = True,
        mcl_mask_mode: str = "real",
        fp_mode: str = 'ecfp',
        fp_bit_dropout: float = 0.0,
        low_capacity_adapter: bool = False,
    ):
        super().__init__()
        self.modality = modality
        self.low_capacity_adapter = bool(low_capacity_adapter)

        encoder, input_dim = self._initialize_encoder(
            modality=modality,
            joint_embedding_dim=joint_embedding_dim,
            smiles_model_name=smiles_model_name,
            gnn_model_name=gnn_model_name,
            graph_num_layers=graph_num_layers,
            graph_emb_dim=graph_emb_dim,
            graph_dropout=graph_dropout,
            graph_encoder_type=graph_encoder_type,
            scage_dist_bar=scage_dist_bar,
            scage_num_heads=scage_num_heads,
            scage_ffn_hidden_dim=scage_ffn_hidden_dim,
            scage_num_kernels=scage_num_kernels,
            scage_attention_dropout=scage_attention_dropout,
            scage_use_pbc_distance=scage_use_pbc_distance,
            scage_use_descriptors=scage_use_descriptors,
            scage_distance_mode=scage_distance_mode,
            scage_distance_rbf=scage_distance_rbf,
            scage_distance_cutoff=scage_distance_cutoff,
            scage_distance_scales=scage_distance_scales,
            scage_distance_taus=scage_distance_taus,
            scage_topology_bias=scage_topology_bias,
            scage_topology_max_distance=scage_topology_max_distance,
            scage_topology_locality_mode=scage_topology_locality_mode,
            scage_topology_locality_threshold=scage_topology_locality_threshold,
            scage_topology_locality_tau=scage_topology_locality_tau,
            scage_periodic_image_mode=scage_periodic_image_mode,
            scage_periodic_image_cap=scage_periodic_image_cap,
            scage_periodic_image_temperature=scage_periodic_image_temperature,
            scage_force_topology_only=scage_force_topology_only,
            mips_core=mips_core,
            mips_max_hops=mips_max_hops,
            mips_use_descriptors=mips_use_descriptors,
            spatial_mode=spatial_mode,
            graph_geometry_mode=graph_geometry_mode,
            mcl_distance_percentiles=mcl_distance_percentiles,
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
            mips_descriptor_disturbance=mips_descriptor_disturbance,
            mips_backbone_mode=mips_backbone_mode,
            mips_input_norm=mips_input_norm,
            mips_mask_mode=mips_mask_mode,
            mips_mask_policy=mips_mask_policy,
            mips_masked_loss_reduction=mips_masked_loss_reduction,
            use_star_rbf=use_star_rbf,
            use_mcl=use_mcl,
            mcl_mask_mode=mcl_mask_mode,
            fp_mode=fp_mode,
            fp_bit_dropout=fp_bit_dropout,
        )
        self.encoder = encoder
        self.norm = nn.LayerNorm(input_dim) if encoder else None
        self.projection = self._create_projection(
            input_dim,
            joint_embedding_dim,
            hidden_dim=(
                128
                if self.low_capacity_adapter and modality == 'smiles'
                else None
            ),
        )
        if encoder and freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

    def _initialize_encoder(
        self,
        modality: str,
        joint_embedding_dim: int,
        smiles_model_name: Optional[str],
        gnn_model_name: Optional[str],
        graph_num_layers: int,
        graph_emb_dim: int,
        graph_dropout: float,
        graph_encoder_type: str,
        scage_dist_bar,
        scage_num_heads: int,
        scage_ffn_hidden_dim: int,
        scage_num_kernels: int,
        scage_attention_dropout: float,
        scage_use_pbc_distance: bool,
        scage_use_descriptors: bool,
        scage_distance_mode: str,
        scage_distance_rbf: int,
        scage_distance_cutoff: float,
        scage_distance_scales,
        scage_distance_taus,
        scage_topology_bias: bool,
        scage_topology_max_distance: int,
        scage_topology_locality_mode: str,
        scage_topology_locality_threshold: int,
        scage_topology_locality_tau: float,
        scage_periodic_image_mode: str,
        scage_periodic_image_cap: int,
        scage_periodic_image_temperature: float,
        scage_force_topology_only: bool,
        mips_core: str,
        mips_max_hops: Optional[int],
        mips_use_descriptors: bool,
        spatial_mode: str,
        graph_geometry_mode: str,
        mcl_distance_percentiles,
        trimer_num_candidates: int,
        trimer_max_heavy_atoms: int,
        mips_variant: str,
        mips_atom_feature_mode: str,
        mips_attention_scale: str,
        mips_norm_mode: str,
        mips_activation: str,
        mips_spd_bias_mode: str,
        mips_path_bias_mode: str,
        mips_multi_scale_hop_gate: Optional[bool],
        mips_semantics: str,
        mips_descriptor_fusion_mode: str,
        mips_descriptor_components: str,
        mips_descriptor_disturbance: float,
        mips_backbone_mode: str,
        mips_input_norm: bool,
        mips_mask_mode: str,
        mips_mask_policy: str,
        mips_masked_loss_reduction: str,
        use_star_rbf: bool,
        use_mcl: bool,
        mcl_mask_mode: str,
        fp_mode: str,
        fp_bit_dropout: float,
    ):
        if modality == 'smiles':
            encoder = _load_roberta_encoder(smiles_model_name)
            input_dim = encoder.config.hidden_size
            _model_load_log(
                f"Loaded SMILES encoder: {smiles_model_name} "
                "(MLM head intentionally omitted)."
            )
        elif modality == 'graph':
            graph_encoder_type = str(graph_encoder_type).lower()
            if graph_encoder_type == 'mips_trimer_scage':
                encoder = MIPSLocalGraphEncoder(
                    core=mips_core,
                    num_layer=graph_num_layers,
                    emb_dim=graph_emb_dim,
                    num_heads=scage_num_heads,
                    dropout=graph_dropout,
                    max_hops=mips_max_hops,
                    use_descriptors=mips_use_descriptors,
                    spatial_mode=spatial_mode,
                    graph_geometry_mode=graph_geometry_mode,
                    mcl_distance_percentiles=mcl_distance_percentiles,
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
                    descriptor_disturbance=mips_descriptor_disturbance,
                    backbone_mode=mips_backbone_mode,
                    input_norm=mips_input_norm,
                    mask_mode=mips_mask_mode,
                    mask_policy=mips_mask_policy,
                    masked_loss_reduction=mips_masked_loss_reduction,
                    use_star_rbf=use_star_rbf,
                    use_mcl=use_mcl,
                    mcl_mask_mode=mcl_mask_mode,
                )
                _model_load_log(
                    "Using sparse non-PBC MIPS PyG graph encoder "
                    f"(layers={graph_num_layers}, emb_dim={graph_emb_dim}, "
                    f"heads={scage_num_heads}, core={mips_core}, "
                    f"variant={encoder.variant}, spatial={encoder.spatial_mode}, "
                    f"geometry={encoder.graph_geometry_mode}, "
                    f"max_hops={encoder.max_hops}, "
                    f"feature_mode={encoder.feature_mode}, readout=mean)."
                )
                input_dim = encoder.emb_dim
            else:
                raise ValueError(
                    "graph_encoder_type must be 'mips_trimer_scage'; "
                    "no other graph encoder is available"
                )
        elif modality == 'fp':
            input_dims = {
                "ecfp": 1024,
                "mixfp": 1048,
                "attachment_count": 2570,
            }
            if fp_mode not in input_dims:
                raise ValueError(f"Unsupported fp_mode: {fp_mode}")
            input_dim = input_dims[fp_mode]
            encoder = FingerprintEncoder(
                input_dim=input_dim,
                joint_embedding_dim=joint_embedding_dim,
                bit_dropout=fp_bit_dropout,
                hidden_dim=(128 if self.low_capacity_adapter else None),
            )
        else:
            raise ValueError(
                f"Unsupported modality: {modality}. Current supported modalities are: "
                f"{', '.join(SUPPORTED_MODALITIES)}."
            )

        return encoder, input_dim

    @staticmethod
    def _get_pretrained_path(model_path: Optional[str], label: str):
        if model_path is None:
            return None

        model_path = str(model_path).strip()
        if not model_path or model_path.lower() in {'none', 'null'}:
            return None

        if not os.path.exists(model_path):
            print(f"{label} pretrained file not found: {model_path}")
            return None

        if os.path.getsize(model_path) == 0:
            print(f"{label} pretrained file is empty: {model_path}")
            return None

        return model_path

    @staticmethod
    def _create_projection(
        input_dim: int,
        joint_embedding_dim: int,
        hidden_dim: Optional[int] = None,
    ):
        if hidden_dim is not None:
            return nn.Sequential(
                nn.Linear(input_dim, int(hidden_dim)),
                nn.LayerNorm(int(hidden_dim)),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(int(hidden_dim), joint_embedding_dim),
            )
        return nn.Sequential(
            nn.Linear(input_dim, joint_embedding_dim),
            nn.LayerNorm(joint_embedding_dim),
            nn.ReLU(),
        )

    def _encode_smiles(self, data):
        input_ids = data.input_ids_smiles.to(self.encoder.device)
        attention_mask = data.attention_mask_smiles.to(self.encoder.device)
        view_count = 1
        batch_size = input_ids.size(0)
        if input_ids.ndim == 3:
            view_count = input_ids.size(1)
            input_ids = input_ids.flatten(0, 1)
            attention_mask = attention_mask.flatten(0, 1)
        features = self.encoder(input_ids, attention_mask=attention_mask).last_hidden_state
        pooled = self.projection(self.norm(features[:, 0, :]))
        if view_count > 1:
            pooled = pooled.reshape(batch_size, view_count, -1).mean(dim=1)
        return pooled

    def encode_global_and_tokens(self, data):
        if self.modality == 'graph':
            if getattr(self.encoder, 'expects_data', False):
                graph_features, node_features = self.encoder(data)
            else:
                graph_features = self.encoder(data)
                node_features = graph_features.unsqueeze(1)
            graph_features = self.projection(self.norm(graph_features))
            node_features = self.projection(self.norm(node_features))
            return graph_features, node_features, data.batch

        global_features = self(data)
        return global_features, global_features.unsqueeze(1), None

    def forward(self, data):
        if self.modality == 'smiles':
            return self._encode_smiles(data)
        if self.modality == 'graph':
            if getattr(self.encoder, 'expects_data', False):
                features, _ = self.encoder(data)
            else:
                features = self.encoder(data)
            return self.projection(self.norm(features))
        if self.modality == 'fp':
            return self.encoder(data.fp)
        raise ValueError(
            f"Unsupported modality: {self.modality}. Current supported modalities are: "
            f"{', '.join(SUPPORTED_MODALITIES)}."
        )

class FingerprintEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        joint_embedding_dim: int,
        dropout: float = 0.1,
        bit_dropout: float = 0.0,
        hidden_dim: Optional[int] = None,
    ):
        super().__init__()
        self.bit_dropout = float(bit_dropout)
        hidden_dim = (
            int(hidden_dim)
            if hidden_dim is not None
            else max(joint_embedding_dim * 2, 256)
        )
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, joint_embedding_dim),
            nn.LayerNorm(joint_embedding_dim),
            nn.ReLU(),
        )

    def forward(self, fp):
        fp = fp.float()
        if self.training and self.bit_dropout > 0:
            fp = fp * (torch.rand_like(fp) >= self.bit_dropout).to(fp.dtype)
        return self.network(fp)
