import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
from transformers import RobertaModel

from .geom import PaiNNEncoder
from .graph import GNN_graphpred
from .scage_graph import SCAGEGraphEncoder
from .mips_periodic_graph import MIPSPeriodicGraphEncoder


SUPPORTED_MODALITIES = ('smiles', 'graph', 'fp', 'geom')
SUPPORTED_FUSION_TYPES = ('self_attention_pooling', 'parallel_attention')


class UniEncoderAttention(nn.Module):
    def __init__(
        self,
        joint_embedding_dim: int,
        smiles_model_name: Optional[str],
        gnn_model_name: Optional[str],
        geom_model_name: Optional[str],
        modality_list: List[str],
        output_attention_weights: bool = True,
        freeze_encoder: bool = False,
        num_heads: int = 8,
        ff_dim: Optional[int] = None,
        dropout: float = 0.1,
        geometry_encoder: str = 'painn',
        graph_num_layers: int = 6,
        graph_emb_dim: int = 256,
        graph_dropout: float = 0.1,
        graph_pooling: str = 'attention',
        graph_jk: str = 'sum',
        graph_norm: str = 'graph',
        graph_residual: bool = True,
        graph_encoder_type: str = 'gin',
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
        fusion_type: str = 'self_attention_pooling',
        fp_mode: str = 'ecfp',
        parallel_attention_layers: int = 1,
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
        self.graph_encoder_type = str(graph_encoder_type).lower()
        self.modality_dropout = dict(modality_dropout or {})
        if self.fusion_type == 'parallel_attention':
            valid_modalities = set(modality_list) in ({'graph'}, {'smiles', 'graph', 'fp'})
            if self.graph_encoder_type != 'scage' or not valid_modalities:
                raise ValueError(
                    "fusion_type='parallel_attention' requires graph_encoder_type='scage' "
                    "and either Stage 1 modalities=graph or Stage 2/3 modalities=smiles graph fp."
                )

        self.encoders = nn.ModuleDict({
            modality: EncoderModule(
                modality=modality,
                joint_embedding_dim=joint_embedding_dim,
                freeze_encoder=freeze_encoder,
                smiles_model_name=smiles_model_name,
                gnn_model_name=gnn_model_name,
                geom_model_name=geom_model_name,
                geometry_encoder=geometry_encoder,
                graph_num_layers=graph_num_layers,
                graph_emb_dim=graph_emb_dim,
                graph_dropout=graph_dropout,
                graph_pooling=graph_pooling,
                graph_jk=graph_jk,
                graph_norm=graph_norm,
                graph_residual=graph_residual,
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
                fp_mode=fp_mode,
                fp_bit_dropout=fp_bit_dropout,
            )
            for modality in modality_list
        })

        ff_dim = ff_dim or (joint_embedding_dim * 2)
        self.fusion_module = FusionModule(
            joint_embedding_dim=joint_embedding_dim,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout=dropout,
        )
        fusion_dropout = dropout if fusion_dropout is None else float(fusion_dropout)
        self.parallel_attention_fusion = ParallelAttentionFusion(
            joint_embedding_dim=joint_embedding_dim,
            num_heads=num_heads,
            ff_dim=ff_dim,
            modality_names=modality_list,
            num_layers=parallel_attention_layers,
            dropout=fusion_dropout,
        )

        projection_dim = int(alignment_projection_dim)
        self.alignment_projections = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(joint_embedding_dim, joint_embedding_dim),
                nn.GELU(),
                nn.Linear(joint_embedding_dim, projection_dim),
            )
            for name in tuple(modality_list) + ('fusion',)
        })

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
        return torch.stack([self.encoders[name](data) for name in self.modality_list], dim=1)

    def intrinsic_availability_mask(self, data, device=None):
        """Return modalities that physically exist for each sample.

        SMILES and fingerprints are always available in the current dataset.
        The periodic graph builder may explicitly reject an over-long or
        aliasing repeat unit; those graph placeholders must never participate
        in fusion or alignment.
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
            probability = float(probabilities.get(name, 0.0))
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
        if self.fusion_type == 'parallel_attention':
            return self.parallel_attention_fusion(embeddings, availability_mask=availability_mask)
        if availability_mask is not None and not bool(availability_mask.all()):
            raise ValueError("Availability masks are supported only by parallel_attention fusion")
        return self.fusion_module(embeddings)

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
        if self.fusion_type == 'parallel_attention':
            intrinsic_mask = self.intrinsic_availability_mask(
                data, device=embeddings.device
            )
            if availability_mask is None and self.training:
                availability_mask = self.sample_availability_mask(
                    embeddings.size(0), device=embeddings.device,
                    intrinsic_mask=intrinsic_mask,
                )
            elif availability_mask is None:
                availability_mask = intrinsic_mask
            else:
                availability_mask = availability_mask.to(
                    device=embeddings.device, dtype=torch.bool
                ) & intrinsic_mask

        fused_output, modality_attention = self.fuse_embeddings(embeddings, availability_mask)
        visual_name = 'parallel' if self.fusion_type == 'parallel_attention' else 'flat'
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
        geom_model_name: Optional[str] = None,
        geometry_encoder: str = 'painn',
        graph_num_layers: int = 6,
        graph_emb_dim: int = 256,
        graph_dropout: float = 0.1,
        graph_pooling: str = 'attention',
        graph_jk: str = 'sum',
        graph_norm: str = 'graph',
        graph_residual: bool = True,
        graph_encoder_type: str = 'gin',
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
        fp_mode: str = 'ecfp',
        fp_bit_dropout: float = 0.0,
    ):
        super().__init__()
        self.modality = modality

        encoder, input_dim = self._initialize_encoder(
            modality=modality,
            joint_embedding_dim=joint_embedding_dim,
            smiles_model_name=smiles_model_name,
            gnn_model_name=gnn_model_name,
            geom_model_name=geom_model_name,
            geometry_encoder=geometry_encoder,
            graph_num_layers=graph_num_layers,
            graph_emb_dim=graph_emb_dim,
            graph_dropout=graph_dropout,
            graph_pooling=graph_pooling,
            graph_jk=graph_jk,
            graph_norm=graph_norm,
            graph_residual=graph_residual,
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
            fp_mode=fp_mode,
            fp_bit_dropout=fp_bit_dropout,
        )
        self.encoder = encoder
        self.norm = nn.LayerNorm(input_dim) if encoder else None
        self.projection = self._create_projection(input_dim, joint_embedding_dim)
        self.geom_context_embedding = (
            # 0=PBC/screw, 1=fallback, 2=repeat-unit, 3=s-mer center context.
            nn.Embedding(4, joint_embedding_dim) if modality == 'geom' else None
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
        geom_model_name: Optional[str],
        geometry_encoder: str,
        graph_num_layers: int,
        graph_emb_dim: int,
        graph_dropout: float,
        graph_pooling: str,
        graph_jk: str,
        graph_norm: str,
        graph_residual: bool,
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
        fp_mode: str,
        fp_bit_dropout: float,
    ):
        if modality == 'smiles':
            encoder = RobertaModel.from_pretrained(smiles_model_name)
            input_dim = encoder.config.hidden_size
            print(f"Loaded smiles pretrained weights from {smiles_model_name}")
        elif modality == 'geom':
            encoder = self._initialize_geometry_encoder(geometry_encoder, geom_model_name)
            input_dim = encoder.hidden_channels
        elif modality == 'graph':
            graph_encoder_type = str(graph_encoder_type).lower()
            if graph_encoder_type == 'gin':
                encoder = GNN_graphpred(
                    num_layer=graph_num_layers,
                    emb_dim=graph_emb_dim,
                    num_tasks=1,
                    JK=graph_jk,
                    drop_ratio=graph_dropout,
                    gnn_type='gin',
                    graph_pooling=graph_pooling,
                    norm_type=graph_norm,
                    residual=graph_residual,
                )
                gnn_model_path = self._get_pretrained_path(gnn_model_name, "GNN")
                if gnn_model_path:
                    encoder.from_pretrained(model_file=gnn_model_path)
                    print(f"Loaded GNN pretrained weights from {gnn_model_path}")
                else:
                    print("No GNN pretrained weights provided; using random initialization.")
                input_dim = encoder.emb_dim
            elif graph_encoder_type == 'scage':
                encoder = MIPSPeriodicGraphEncoder(
                    num_layer=graph_num_layers,
                    emb_dim=graph_emb_dim,
                    num_heads=scage_num_heads,
                    dropout=graph_dropout,
                    num_kernels=scage_num_kernels,
                    max_hops=5,
                    num_rbf=scage_distance_rbf,
                    max_distance=scage_distance_cutoff,
                )
                print(
                    "Using sparse MIPS-PBC PyG graph encoder "
                    f"(layers={graph_num_layers}, emb_dim={graph_emb_dim}, "
                    f"heads={scage_num_heads}, max_hops=5, "
                    f"distance_rbf={scage_distance_rbf}, "
                    f"distance_cutoff={scage_distance_cutoff}, readout=mean)."
                )
                input_dim = encoder.emb_dim
            else:
                raise ValueError("graph_encoder_type must be 'gin' or 'scage'")
        elif modality == 'fp':
            input_dim = 1024 if fp_mode == 'ecfp' else 1048
            encoder = FingerprintEncoder(
                input_dim=input_dim,
                joint_embedding_dim=joint_embedding_dim,
                bit_dropout=fp_bit_dropout,
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
    def _initialize_geometry_encoder(geometry_encoder: str, geom_model_name: Optional[str]):
        geometry_encoder = geometry_encoder.lower()
        geom_model_path = EncoderModule._get_pretrained_path(geom_model_name, f"{geometry_encoder} geometry")
        loaded_geom_path = geom_model_path
        if geometry_encoder == 'painn':
            load_path = geom_model_path
            loaded_geom_path = load_path
            encoder = PaiNNEncoder(
                load_from_pretrain=load_path,
                hidden_channels=128,
                num_layers=6,
                num_rbf=50,
                cutoff=10,
                max_num_neighbors=32,
                readout='mean',
            )
        else:
            raise ValueError("geometry_encoder must be 'painn'")

        if loaded_geom_path:
            print(f"Loaded {geometry_encoder} geometry encoder; pretrained path: {loaded_geom_path}")
        else:
            print(f"No {geometry_encoder} geometry pretrained weights provided; using random initialization.")
        return encoder

    @staticmethod
    def _create_projection(input_dim: int, joint_embedding_dim: int):
        return nn.Sequential(
            nn.Linear(input_dim, joint_embedding_dim),
            nn.LayerNorm(joint_embedding_dim),
            nn.ReLU(),
        )

    def _encode_smiles(self, data):
        input_ids = data.input_ids_smiles.to(self.encoder.device)
        attention_mask = data.attention_mask_smiles.to(self.encoder.device)
        features = self.encoder(input_ids, attention_mask=attention_mask).last_hidden_state
        return self.projection(self.norm(features[:, 0, :]))

    def encode_global_and_tokens(self, data):
        if self.modality == 'graph':
            if getattr(self.encoder, 'expects_data', False):
                graph_features, node_features = self.encoder(data)
            elif getattr(self.encoder, 'uses_geometry', False):
                graph_features, node_features = self.encoder(data)
            else:
                graph_features, node_features = self.encoder(data.x, data.edge_index, data.edge_attr, data.batch)
            graph_features = self.projection(self.norm(graph_features))
            node_features = self.projection(self.norm(node_features))
            return graph_features, node_features, data.batch

        if self.modality == 'geom':
            graph_features = self.encoder(data)
            node_features, node_batch = self.encoder.encode_nodes(data)
            pool_mask = getattr(data, 'geom_pool_mask', None)
            if pool_mask is not None:
                pool_mask = pool_mask.bool()
                node_features = node_features[pool_mask]
                node_batch = node_batch[pool_mask]
            graph_features = self._project_geom(graph_features, data)
            node_features = self.projection(self.norm(node_features))
            return graph_features, node_features, node_batch

        global_features = self(data)
        return global_features, global_features.unsqueeze(1), None

    def forward(self, data):
        if self.modality == 'smiles':
            return self._encode_smiles(data)
        if self.modality == 'graph':
            if getattr(self.encoder, 'expects_data', False):
                features, _ = self.encoder(data)
            elif getattr(self.encoder, 'uses_geometry', False):
                features, _ = self.encoder(data)
            else:
                features, _ = self.encoder(data.x, data.edge_index, data.edge_attr, data.batch)
            return self.projection(self.norm(features))
        if self.modality == 'geom':
            features = self.encoder(data)
            return self._project_geom(features, data)
        if self.modality == 'fp':
            return self.encoder(data.fp)
        raise ValueError(
            f"Unsupported modality: {self.modality}. Current supported modalities are: "
            f"{', '.join(SUPPORTED_MODALITIES)}."
        )

    def _project_geom(self, features, data):
        projected = self.projection(self.norm(features))
        if self.geom_context_embedding is None:
            return projected
        context = getattr(data, 'geom_context_id', None)
        if context is None:
            context = projected.new_full((projected.size(0),), 2, dtype=torch.long)
        context = context.long().clamp_(0, 3).to(projected.device)
        return projected + self.geom_context_embedding(context)


class FingerprintEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        joint_embedding_dim: int,
        dropout: float = 0.1,
        bit_dropout: float = 0.0,
    ):
        super().__init__()
        self.bit_dropout = float(bit_dropout)
        hidden_dim = max(joint_embedding_dim * 2, 256)
        self.network = nn.Sequential(
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


class AttentionPooling(nn.Module):
    def __init__(self, joint_embedding_dim: int):
        super().__init__()
        self.attention = nn.Linear(joint_embedding_dim, 1)

    def forward(self, embeddings: torch.Tensor, availability_mask=None):
        scores = self.attention(embeddings).squeeze(-1)
        if availability_mask is not None:
            if availability_mask.shape != scores.shape:
                raise ValueError("availability_mask must have shape [batch, modalities]")
            if not bool(availability_mask.any(dim=1).all()):
                raise ValueError("Every sample must have at least one available modality")
            scores = scores.masked_fill(~availability_mask, -1e12)
        weights = F.softmax(scores, dim=1)
        output = torch.sum(embeddings * weights.unsqueeze(-1), dim=1)
        return output, weights


class FusionModule(nn.Module):
    def __init__(
        self,
        joint_embedding_dim: int,
        num_heads: int,
        ff_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=joint_embedding_dim,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.layer_norm1 = nn.LayerNorm(joint_embedding_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(joint_embedding_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, joint_embedding_dim),
        )
        self.layer_norm2 = nn.LayerNorm(joint_embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.attention_pooling = AttentionPooling(joint_embedding_dim)

    def forward(self, embeddings: torch.Tensor):
        embeddings = embeddings.permute(1, 0, 2)

        attn_output, _ = self.multihead_attn(embeddings, embeddings, embeddings)
        attn_output = attn_output.permute(1, 0, 2)

        fused_embedding = embeddings.permute(1, 0, 2) + attn_output
        fused_embedding = self.layer_norm1(fused_embedding)

        ff_output = self.feed_forward(fused_embedding)
        ff_output = self.dropout(ff_output)

        fused_embedding = fused_embedding + ff_output
        fused_embedding = self.layer_norm2(fused_embedding)

        output, attention_weights = self.attention_pooling(fused_embedding)
        return output, attention_weights



class ParallelAttentionBlock(nn.Module):
    def __init__(
        self,
        joint_embedding_dim: int,
        num_heads: int,
        ff_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.attn_norm = nn.LayerNorm(joint_embedding_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=joint_embedding_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ffn_norm = nn.LayerNorm(joint_embedding_dim)
        self.ffn = nn.Sequential(
            nn.Linear(joint_embedding_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, joint_embedding_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens, availability_mask=None):
        normalized = self.attn_norm(tokens)
        key_padding_mask = None if availability_mask is None else ~availability_mask
        attended, attention = self.attention(
            normalized, normalized, normalized,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=False,
        )
        tokens = tokens + self.dropout(attended)
        tokens = tokens + self.dropout(self.ffn(self.ffn_norm(tokens)))
        return tokens, attention


class ParallelAttentionFusion(nn.Module):
    def __init__(
        self,
        joint_embedding_dim: int,
        num_heads: int,
        ff_dim: int,
        modality_names,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("parallel_attention_layers must be at least 1")
        self.modality_names = tuple(modality_names)
        self.input_norms = nn.ModuleDict({
            name: nn.LayerNorm(joint_embedding_dim) for name in self.modality_names
        })
        self.modality_embeddings = nn.ParameterDict({
            name: nn.Parameter(torch.zeros(joint_embedding_dim)) for name in self.modality_names
        })
        self.missing_tokens = nn.ParameterDict({
            name: nn.Parameter(torch.zeros(joint_embedding_dim)) for name in self.modality_names
        })
        self.blocks = nn.ModuleList([
            ParallelAttentionBlock(
                joint_embedding_dim=joint_embedding_dim,
                num_heads=num_heads,
                ff_dim=ff_dim,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])
        self.pooling = AttentionPooling(joint_embedding_dim)
        self.output_norm = nn.LayerNorm(joint_embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        for embedding in self.modality_embeddings.values():
            nn.init.normal_(embedding, mean=0.0, std=0.02)
        for token in self.missing_tokens.values():
            nn.init.normal_(token, mean=0.0, std=0.02)

    def forward(self, embeddings, availability_mask=None):
        if embeddings.size(1) != len(self.modality_names):
            raise ValueError(
                f"Expected {len(self.modality_names)} modality tokens, got {embeddings.size(1)}"
            )
        if availability_mask is None:
            availability_mask = torch.ones(
                embeddings.shape[:2], dtype=torch.bool, device=embeddings.device
            )
        if availability_mask.shape != embeddings.shape[:2]:
            raise ValueError("availability_mask must match [batch, modalities]")
        if not bool(availability_mask.any(dim=1).all()):
            raise ValueError("Every sample must retain at least one modality")
        base_tokens = torch.stack([
            self.input_norms[name](embeddings[:, idx])
            for idx, name in enumerate(self.modality_names)
        ], dim=1)
        missing = torch.stack([
            self.missing_tokens[name] for name in self.modality_names
        ], dim=0).unsqueeze(0)
        base_tokens = torch.where(availability_mask.unsqueeze(-1), base_tokens, missing)
        type_embeddings = torch.stack([
            self.modality_embeddings[name] for name in self.modality_names
        ], dim=0).unsqueeze(0)
        tokens = base_tokens + type_embeddings
        for block in self.blocks:
            tokens, _ = block(tokens, availability_mask=availability_mask)

        attended, pooling_weights = self.pooling(tokens, availability_mask=availability_mask)
        valid = availability_mask.unsqueeze(-1).to(base_tokens.dtype)
        valid_count = valid.sum(dim=1).clamp_min(1.0)
        mean_residual = (base_tokens * valid).sum(dim=1) / valid_count.sqrt()
        output = self.output_norm(mean_residual + attended)
        return self.dropout(output), pooling_weights
