import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
from transformers import RobertaModel

from .geom import PaiNNEncoder, SchNetEncoder
from .graph import GNN_graphpred


SUPPORTED_MODALITIES = ('smiles', 'graph', 'fp', 'geom')


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

        self.encoders = nn.ModuleDict({
            modality: EncoderModule(
                modality=modality,
                joint_embedding_dim=joint_embedding_dim,
                freeze_encoder=freeze_encoder,
                smiles_model_name=smiles_model_name,
                gnn_model_name=gnn_model_name,
                geom_model_name=geom_model_name,
                geometry_encoder=geometry_encoder,
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

        self.mlp = nn.Sequential(
            nn.Linear(joint_embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, data):
        embeddings = [self.encoders[modality](data) for modality in self.modality_list]
        embeddings = torch.stack(embeddings, dim=1)

        fused_output, modality_attention = self.fusion_module(embeddings)
        self.unimodal_embedding = fused_output
        self.attention_visual_weights = modality_attention

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
        )
        self.encoder = encoder
        self.norm = nn.LayerNorm(input_dim) if encoder else None
        self.projection = self._create_projection(input_dim, joint_embedding_dim)

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
    ):
        if modality == 'smiles':
            encoder = RobertaModel.from_pretrained(smiles_model_name)
            input_dim = encoder.config.hidden_size
            print(f"Loaded smiles pretrained weights from {smiles_model_name}")
        elif modality == 'geom':
            encoder = self._initialize_geometry_encoder(geometry_encoder, geom_model_name)
            input_dim = encoder.hidden_channels
        elif modality == 'graph':
            encoder = GNN_graphpred(
                num_layer=5,
                emb_dim=300,
                num_tasks=1,
                JK='last',
                drop_ratio=0,
                gnn_type='gin',
                graph_pooling='mean',
            )
            gnn_model_path = self._get_pretrained_path(gnn_model_name, "GNN")
            if gnn_model_path:
                encoder.from_pretrained(model_file=gnn_model_path)
                print(f"Loaded GNN pretrained weights from {gnn_model_path}")
            else:
                print("No GNN pretrained weights provided; using random initialization.")
            input_dim = encoder.emb_dim
        elif modality == 'fp':
            encoder = None
            input_dim = 1024
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
        if geometry_encoder == 'schnet':
            encoder = SchNetEncoder(
                load_from_pretrain=geom_model_path,
                cutoff=10,
                max_num_neighbors=32,
                readout='mean',
            )
        elif geometry_encoder == 'painn':
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
            raise ValueError("geometry_encoder must be 'painn' or 'schnet'")

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

    def forward(self, data):
        if self.modality == 'smiles':
            return self._encode_smiles(data)
        if self.modality == 'graph':
            features, _ = self.encoder(data.x, data.edge_index, data.edge_attr, data.batch)
            return self.projection(self.norm(features))
        if self.modality == 'geom':
            features = self.encoder(data)
            return self.projection(self.norm(features))
        if self.modality == 'fp':
            return self.projection(data.fp)
        raise ValueError(
            f"Unsupported modality: {self.modality}. Current supported modalities are: "
            f"{', '.join(SUPPORTED_MODALITIES)}."
        )


class AttentionPooling(nn.Module):
    def __init__(self, joint_embedding_dim: int):
        super().__init__()
        self.attention = nn.Linear(joint_embedding_dim, 1)

    def forward(self, embeddings: torch.Tensor):
        scores = self.attention(embeddings).squeeze(-1)
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
