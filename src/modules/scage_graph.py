"""SCAGE atom input with polymer-aware continuous distance attention.

AtomEmbedding, feed-forward blocks, and Graph Token follow KazeDog/scage. The
polymer extension replaces its percentile hard visibility masks with a
per-head Gaussian-RBF bias over valid PBC minimum-image/Euclidean distances.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.dataset.graph_data import (
    SCAGE_ATOM_VOCABS,
    SCAGE_CATEGORICAL_FEATURES,
)


class GaussianKernel(nn.Module):
    def __init__(self, num_kernels=128, start=0.0, stop=9.0, std_width=1.0):
        super().__init__()
        means = torch.linspace(float(start), float(stop), int(num_kernels))
        spacing = means[1] - means[0] if means.numel() > 1 else means.new_tensor(1.0)
        self.register_buffer("means", means)
        self.register_buffer("std", spacing * float(std_width))
        self.mul = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, values):
        values = self.mul * values.unsqueeze(-1) + self.bias
        std = self.std.clamp_min(1e-8)
        return torch.exp(-0.5 * ((values - self.means) / std) ** 2) / (
            math.sqrt(2.0 * math.pi) * std
        )


class SCAGEAtomEmbedding(nn.Module):
    def __init__(self, emb_dim=512, num_kernels=128):
        super().__init__()
        self.emb_dim = int(emb_dim)
        self.categorical_embeddings = nn.ModuleDict({
            name: nn.Embedding(len(SCAGE_ATOM_VOCABS[name]) + 5, self.emb_dim, padding_idx=0)
            for name in SCAGE_CATEGORICAL_FEATURES
        })
        self.continuous_embeddings = nn.ModuleDict({
            name: nn.Sequential(
                GaussianKernel(num_kernels=num_kernels, start=0.0, stop=9.0),
                nn.Linear(int(num_kernels), self.emb_dim),
            )
            for name in ("mass", "van_der_waals_radius", "partial_charge")
        })
        self.graph_embedding = nn.Embedding(1, self.emb_dim)
        self.backbone_embedding = nn.Embedding(3, self.emb_dim)
        self.ru_index_embedding = nn.Embedding(8, self.emb_dim)
        self.mask_token = nn.Parameter(torch.zeros(self.emb_dim))
        self.final_layer_norm = nn.LayerNorm(self.emb_dim)
        nn.init.normal_(self.mask_token, std=0.02)

    def forward(
        self, categorical, continuous, padding_mask, atom_mask=None,
        backbone_role=None, ru_index=None,
    ):
        atom_embedding = None
        for name in SCAGE_CATEGORICAL_FEATURES:
            value = self.categorical_embeddings[name](categorical[name])
            atom_embedding = value if atom_embedding is None else atom_embedding + value
        for name, embedding in self.continuous_embeddings.items():
            atom_embedding = atom_embedding + embedding(continuous[name])
        if atom_mask is not None:
            atom_embedding = torch.where(
                atom_mask.unsqueeze(-1), self.mask_token.view(1, 1, -1), atom_embedding
            )
        # MIPS masks complete chemical atom features but retains the polymer
        # backbone role as structural context.
        if backbone_role is not None:
            atom_embedding = atom_embedding + self.backbone_embedding(backbone_role.clamp(0, 2))
        if ru_index is not None:
            atom_embedding = atom_embedding + self.ru_index_embedding(ru_index.clamp(0, 7))
        atom_embedding = atom_embedding.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        graph_token = self.graph_embedding.weight.view(1, 1, -1).expand(atom_embedding.size(0), -1, -1)
        return self.final_layer_norm(torch.cat([graph_token, atom_embedding], dim=1))


class PositionWiseFeedForward(nn.Module):
    def __init__(self, emb_dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(emb_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, emb_dim)
        self.activation_dropout = nn.Dropout(dropout)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(emb_dim, eps=1e-6)

    def forward(self, x):
        residual = x
        x = self.fc2(self.activation_dropout(F.gelu(self.fc1(x))))
        return self.layer_norm(residual + self.dropout(x))


class GraphTopologyBias(nn.Module):
    """MIPS shortest-path and multi-hop path-bond attention encoding."""

    def __init__(self, num_heads, max_distance=20, max_path_length=5):
        super().__init__()
        self.num_heads = int(num_heads)
        self.max_distance = int(max_distance)
        if self.max_distance < 1:
            raise ValueError("SCAGE topology max distance must be positive")
        # The final index represents disconnected or longer-than-cap pairs.
        self.spatial_embedding = nn.Embedding(self.max_distance + 2, self.num_heads)
        self.max_path_length = int(max_path_length)
        self.path_embeddings = nn.ModuleList([
            nn.Embedding(size, self.num_heads, padding_idx=0)
            for size in (5, 7, 3, 3, 3)
        ])
        nn.init.normal_(self.spatial_embedding.weight, mean=0.0, std=0.02)
        for embedding in self.path_embeddings:
            nn.init.normal_(embedding.weight, mean=0.0, std=0.02)
            with torch.no_grad():
                embedding.weight[0].zero_()

    def forward(self, shortest_path, path_bond_fields):
        distance_index = shortest_path.long().clamp(0, self.max_distance + 1)
        atom_bias = self.spatial_embedding(distance_index)
        path_bias = atom_bias.new_zeros(atom_bias.shape)
        for field_idx, embedding in enumerate(self.path_embeddings):
            path_bias = path_bias + embedding(path_bond_fields[..., field_idx].long()).sum(dim=-2)
        path_length = (path_bond_fields[..., 0] > 0).sum(dim=-1).clamp_min(1)
        atom_bias = atom_bias + path_bias / path_length.unsqueeze(-1)
        batch_size, nodes, _, _ = atom_bias.shape
        token_bias = atom_bias.new_zeros(batch_size, nodes + 1, nodes + 1, self.num_heads)
        token_bias[:, 1:, 1:, :] = atom_bias
        return token_bias.permute(0, 3, 1, 2)


class MultiScaleAttention(nn.Module):
    def __init__(
        self,
        emb_dim,
        num_heads,
        num_scales,
        dropout=0.1,
        attention_dropout=0.1,
        distance_mode="multiscale_bias",
        distance_rbf=32,
        distance_cutoff=12.0,
        distance_scales=(4.0, 8.0, 12.0),
        distance_taus=(0.5, 1.0, 1.5),
        locality_mode="hard",
        locality_threshold=5,
        locality_tau=1.0,
        periodic_image_cap=1,
    ):
        super().__init__()
        if emb_dim % num_heads:
            raise ValueError("SCAGE embedding dimension must be divisible by the number of heads")
        self.emb_dim = int(emb_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.emb_dim // self.num_heads
        self.q_proj = nn.Linear(self.emb_dim, self.emb_dim)
        self.k_proj = nn.Linear(self.emb_dim, self.emb_dim)
        self.v_proj = nn.Linear(self.emb_dim, self.emb_dim)
        self.distance_mode = str(distance_mode).lower()
        if self.distance_mode not in {"bias", "mask", "multiscale_bias", "mips_dual"}:
            raise ValueError("unsupported SCAGE distance mode")
        self.locality_mode = str(locality_mode).lower()
        if self.locality_mode not in {"hard", "soft", "none"}:
            raise ValueError("SCAGE locality_mode must be hard, soft, or none")
        self.locality_threshold = int(locality_threshold)
        self.locality_tau = float(locality_tau)
        self.distance_cutoff = float(distance_cutoff)
        self.distance_rbf = int(distance_rbf)
        self.distance_scales = tuple(float(value) for value in distance_scales)
        self.distance_taus = tuple(float(value) for value in distance_taus)
        self.periodic_image_cap = int(periodic_image_cap)
        self.periodic_shift_embedding = nn.Embedding(
            2 * self.periodic_image_cap + 1, self.num_heads
        )
        if len(self.distance_scales) != len(self.distance_taus) or not self.distance_scales:
            raise ValueError("SCAGE distance scales and taus must have the same non-zero length")
        if any(scale <= 0 for scale in self.distance_scales) or any(tau <= 0 for tau in self.distance_taus):
            raise ValueError("SCAGE distance scales and taus must be positive")
        self.distance_kernel = None
        self.distance_projection = None
        self.output_projection = None
        self.multiscale_kernels = nn.ModuleList()
        self.multiscale_distance_projections = nn.ModuleList()
        self.multiscale_output_projections = nn.ModuleList()
        self.multiscale_fusion = None
        if self.distance_mode in {"bias", "mips_dual"}:
            self.distance_kernel = GaussianKernel(
                num_kernels=self.distance_rbf,
                start=0.0,
                stop=self.distance_cutoff,
            )
            self.distance_projection = nn.Linear(self.distance_rbf, self.num_heads, bias=False)
            self.output_projection = nn.Linear(self.emb_dim, self.emb_dim)
            if self.distance_mode == "mips_dual":
                self.topology_output_projection = nn.Linear(self.emb_dim, self.emb_dim)
                self.geometry_output_projection = nn.Linear(self.emb_dim, self.emb_dim)
                self.dual_fusion = nn.Linear(self.emb_dim * 2, self.emb_dim)
        elif self.distance_mode == "multiscale_bias":
            for scale in self.distance_scales:
                self.multiscale_kernels.append(GaussianKernel(
                    num_kernels=self.distance_rbf, start=0.0, stop=scale
                ))
                self.multiscale_distance_projections.append(
                    nn.Linear(self.distance_rbf, self.num_heads, bias=False)
                )
                self.multiscale_output_projections.append(nn.Linear(self.emb_dim, self.emb_dim))
            self.multiscale_fusion = nn.Sequential(
                nn.Linear(self.emb_dim * len(self.distance_scales), self.emb_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(self.emb_dim, self.emb_dim),
            )
        if self.distance_mode == "mask":
            self.scale_projections = nn.ModuleList([
                nn.Linear(self.emb_dim, self.emb_dim) for _ in range(int(num_scales))
            ])
            self.scale_linear = nn.Sequential(
                nn.Linear(self.emb_dim * int(num_scales), self.emb_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(self.emb_dim, self.emb_dim),
            )
        else:
            # In continuous-bias mode hard visibility scales are intentionally
            # absent: distance affects scores, not whether a pair can attend.
            self.scale_projections = nn.ModuleList()
            self.scale_linear = None
        self.attention_dropout = nn.Dropout(attention_dropout)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(self.emb_dim, eps=1e-6)

    def _distance_bias(self, distance, geometry_valid, image_shift=None, pair_mask=None):
        # distance includes the graph token at index zero. Its row/column stays
        # unbiased so the graph token remains globally visible.
        clamped = distance.clamp(min=0.0, max=self.distance_cutoff)
        rbf = self.distance_kernel(clamped)
        bias = self.distance_projection(rbf).permute(0, 3, 1, 2)
        if image_shift is not None:
            shift = (image_shift + self.periodic_image_cap).clamp(
                0, 2 * self.periodic_image_cap
            )
            lower = shift.floor().long()
            upper = shift.ceil().long()
            fraction = (shift - lower).unsqueeze(-1)
            shift_bias = (
                (1.0 - fraction) * self.periodic_shift_embedding(lower)
                + fraction * self.periodic_shift_embedding(upper)
            ).permute(0, 3, 1, 2)
            bias = bias + shift_bias
        bias[:, :, 0, :] = 0.0
        bias[:, :, :, 0] = 0.0
        valid = geometry_valid.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        bias = torch.where(valid, bias, torch.zeros_like(bias))
        if pair_mask is not None:
            bias = bias.masked_fill(pair_mask.unsqueeze(1), 0.0)
        return bias

    def _explicit_image_bias(self, image_distances, image_valid, geometry_valid, pair_mask=None):
        """Aggregate per-image distance biases without collapsing image identity."""
        if image_distances is None or image_valid is None:
            raise ValueError("explicit_images requires image distances and validity flags")
        if image_distances.dim() != 4:
            raise ValueError("image_distances must have shape [batch, images, nodes, nodes]")
        image_count = image_distances.size(1)
        expected_count = 2 * self.periodic_image_cap + 1
        if image_count != expected_count:
            raise ValueError(
                f"expected {expected_count} periodic image slots, received {image_count}"
            )
        aggregate = None
        any_valid = torch.zeros_like(image_distances[:, 0], dtype=torch.bool)
        for image_idx in range(image_count):
            distance = image_distances[:, image_idx].float()
            shift_index = image_idx
            finite = torch.isfinite(distance)
            valid = finite & image_valid[:, image_idx].view(-1, 1, 1)
            valid = valid & geometry_valid.view(-1, 1, 1)
            finite_distance = distance.nan_to_num(
                nan=self.distance_cutoff,
                posinf=self.distance_cutoff,
                neginf=0.0,
            ).clamp_min(0.0)
            clamped = finite_distance.clamp(max=self.distance_cutoff)
            rbf = self.distance_kernel(clamped)
            bias = self.distance_projection(rbf).permute(0, 3, 1, 2).float()
            shell_ids = torch.full(
                (distance.size(0),), shift_index, dtype=torch.long, device=distance.device
            )
            shell_bias = self.periodic_shift_embedding(shell_ids).view(
                distance.size(0), self.num_heads, 1, 1
            ).float()
            # A pair outside the RBF cutoff must remain disfavoured. Previously
            # those pairs were discarded and later received zero bias, which
            # silently restored unrestricted QK attention. The log-sigmoid tail
            # is continuous at the cutoff and avoids a hard visibility mask.
            soft_visibility = torch.sigmoid(
                (self.distance_cutoff - finite_distance) / 1.0
            ).clamp_min(1e-8)
            bias = bias + shell_bias + soft_visibility.log().unsqueeze(1)
            bias = bias.masked_fill(~valid.unsqueeze(1), -torch.inf)
            aggregate = bias if aggregate is None else torch.logaddexp(aggregate, bias)
            any_valid |= valid

        aggregate = torch.where(any_valid.unsqueeze(1), aggregate, torch.zeros_like(aggregate))
        # Graph Token is global and must not acquire a shell-count prior.  Use a
        # multiplicative mask instead of in-place writes so autograd can retain
        # the logaddexp graph across all image shells.
        token_mask = torch.ones_like(aggregate)
        token_mask[:, :, 0, :] = 0.0
        token_mask[:, :, :, 0] = 0.0
        aggregate = aggregate * token_mask
        if pair_mask is not None:
            aggregate = aggregate.masked_fill(pair_mask.unsqueeze(1), 0.0)
        return aggregate

    def _multiscale_distance_bias(self, distance, geometry_valid, scale_idx):
        scale = self.distance_scales[scale_idx]
        tau = self.distance_taus[scale_idx]
        clamped = distance.clamp(min=0.0, max=scale)
        rbf = self.multiscale_kernels[scale_idx](clamped)
        bias = self.multiscale_distance_projections[scale_idx](rbf).permute(0, 3, 1, 2)
        soft_visibility = torch.sigmoid((scale - distance) / tau).clamp_min(1e-8)
        bias = bias + soft_visibility.log().unsqueeze(1)

        # The graph token is global. It receives neither a distance penalty nor
        # a geometry bias at any scale.
        bias[:, :, 0, :] = 0.0
        bias[:, :, :, 0] = 0.0
        valid = geometry_valid.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        return torch.where(valid, bias, torch.zeros_like(bias))

    def forward(
        self, x, distance, thresholds, padding_mask, geometry_valid=None,
        topology_bias=None, topology_distance=None,
        image_shift=None, geometry_pair_mask=None,
        image_distances=None, image_valid=None, explicit_images=False,
    ):
        residual = x
        batch_size, length, _ = x.shape
        q = self.q_proj(x).view(batch_size, length, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, length, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, length, self.num_heads, self.head_dim).transpose(1, 2)
        # Keep attention logits and softmax in FP32 under autocast.
        scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) / math.sqrt(self.head_dim)

        key_padding = padding_mask.unsqueeze(1).unsqueeze(2)
        if self.distance_mode == "mips_dual":
            if topology_bias is None or topology_distance is None:
                raise ValueError("mips_dual requires cached topology bias and distance")
            if geometry_valid is None:
                geometry_valid = torch.ones(batch_size, dtype=torch.bool, device=x.device)
            topology_scores = scores + topology_bias.float()
            if self.locality_mode == "hard":
                visible = topology_distance < self.locality_threshold
                visible[:, 0, :] = ~padding_mask
                visible[:, :, 0] = ~padding_mask
                diagonal = torch.arange(length, device=x.device)
                visible[:, diagonal, diagonal] = ~padding_mask
                topology_scores = topology_scores.masked_fill(~visible.unsqueeze(1), -1e12)
            elif self.locality_mode == "soft":
                soft = torch.sigmoid(
                    (float(self.locality_threshold) - topology_distance.float())
                    / max(self.locality_tau, 1e-6)
                ).clamp_min(1e-8).log()
                soft[:, 0, :] = 0.0
                soft[:, :, 0] = 0.0
                topology_scores = topology_scores + soft.unsqueeze(1)
            topology_scores = topology_scores.masked_fill(key_padding, -1e12)
            topology_attention = self.attention_dropout(F.softmax(topology_scores, dim=-1))
            topology_output = torch.matmul(topology_attention, v.float())
            topology_output = topology_output.transpose(1, 2).reshape(batch_size, length, self.emb_dim)
            topology_output = self.topology_output_projection(topology_output.to(x.dtype))

            with torch.autocast(device_type=x.device.type, enabled=False):
                if explicit_images:
                    geometry_bias = self._explicit_image_bias(
                        image_distances.float(), image_valid, geometry_valid,
                        pair_mask=geometry_pair_mask,
                    ).float()
                else:
                    geometry_bias = self._distance_bias(
                        distance.float(), geometry_valid, image_shift=image_shift,
                        pair_mask=geometry_pair_mask,
                    ).float()
            geometry_scores = scores + geometry_bias
            geometry_scores = geometry_scores.masked_fill(key_padding, -1e12)
            geometry_attention = self.attention_dropout(F.softmax(geometry_scores, dim=-1))
            geometry_output = torch.matmul(geometry_attention, v.float())
            geometry_output = geometry_output.transpose(1, 2).reshape(batch_size, length, self.emb_dim)
            geometry_output = self.geometry_output_projection(geometry_output.to(x.dtype))
            geometry_output = geometry_output * geometry_valid.view(-1, 1, 1).to(x.dtype)
            fused_output = self.dual_fusion(
                torch.cat([topology_output, geometry_output], dim=-1)
            )
            # Invalid geometry must be a true topology-only fallback. Applying
            # dual_fusion([topology, 0]) still transforms the topology branch
            # and introduces the fusion bias, so select the unmodified branch
            # output for those samples explicitly.
            x = torch.where(
                geometry_valid.view(-1, 1, 1), fused_output, topology_output
            )
            return self.layer_norm(residual + self.dropout(x))

        if topology_bias is not None:
            scores = scores + topology_bias.float()
        if self.distance_mode == "bias":
            if geometry_valid is None:
                geometry_valid = torch.ones(batch_size, dtype=torch.bool, device=x.device)
            scores = scores + self._distance_bias(
                distance, geometry_valid, image_shift=image_shift,
                pair_mask=geometry_pair_mask,
            )
            scores = scores.masked_fill(key_padding, -1e12)
            attention = self.attention_dropout(F.softmax(scores, dim=-1))
            output = torch.matmul(attention, v.float()).transpose(1, 2).reshape(batch_size, length, self.emb_dim)
            output = output.to(x.dtype)
            x = self.output_projection(output)
            return self.layer_norm(residual + self.dropout(x))

        if self.distance_mode == "multiscale_bias":
            if geometry_valid is None:
                geometry_valid = torch.ones(batch_size, dtype=torch.bool, device=x.device)
            outputs = []
            for scale_idx, projection in enumerate(self.multiscale_output_projections):
                scale_scores = scores + self._multiscale_distance_bias(
                    distance, geometry_valid, scale_idx
                )
                scale_scores = scale_scores.masked_fill(key_padding, -1e12)
                attention = self.attention_dropout(F.softmax(scale_scores, dim=-1))
                output = torch.matmul(attention, v).transpose(1, 2).reshape(
                    batch_size, length, self.emb_dim
                )
                outputs.append(projection(output))
            x = self.multiscale_fusion(torch.cat(outputs, dim=-1))
            return self.layer_norm(residual + self.dropout(x))

        valid_pairs = (~padding_mask).unsqueeze(1) & (~padding_mask).unsqueeze(2)
        diagonal = torch.arange(length, device=x.device)
        outputs = []
        for scale_idx, projection in enumerate(self.scale_projections):
            visible = valid_pairs & (distance < thresholds[:, scale_idx].view(-1, 1, 1))
            visible[:, 0, :] = ~padding_mask
            visible[:, :, 0] = ~padding_mask
            visible[:, diagonal, diagonal] = ~padding_mask
            scale_scores = scores.masked_fill(~visible.unsqueeze(1), -1e12)
            attention = self.attention_dropout(F.softmax(scale_scores, dim=-1))
            output = torch.matmul(attention, v).transpose(1, 2).reshape(batch_size, length, self.emb_dim)
            outputs.append(projection(output))

        x = self.scale_linear(torch.cat(outputs, dim=-1))
        x = self.layer_norm(residual + self.dropout(x))
        return x


class EncoderAtomLayer(nn.Module):
    def __init__(
        self,
        emb_dim,
        ffn_hidden_dim,
        num_heads,
        num_scales,
        dropout,
        attention_dropout,
        distance_mode,
        distance_rbf,
        distance_cutoff,
        distance_scales,
        distance_taus,
        locality_mode,
        locality_threshold,
        locality_tau,
        periodic_image_cap,
    ):
        super().__init__()
        self.self_attention = MultiScaleAttention(
            emb_dim=emb_dim,
            num_heads=num_heads,
            num_scales=num_scales,
            dropout=dropout,
            attention_dropout=attention_dropout,
            distance_mode=distance_mode,
            distance_rbf=distance_rbf,
            distance_cutoff=distance_cutoff,
            distance_scales=distance_scales,
            distance_taus=distance_taus,
            locality_mode=locality_mode,
            locality_threshold=locality_threshold,
            locality_tau=locality_tau,
            periodic_image_cap=periodic_image_cap,
        )
        self.feed_forward = PositionWiseFeedForward(emb_dim, ffn_hidden_dim, dropout=dropout)

    def forward(
        self, x, distance, thresholds, padding_mask, geometry_valid=None,
        topology_bias=None, topology_distance=None,
        image_shift=None, geometry_pair_mask=None,
        image_distances=None, image_valid=None, explicit_images=False,
    ):
        x = self.self_attention(
            x,
            distance,
            thresholds,
            padding_mask,
            geometry_valid=geometry_valid,
            topology_bias=topology_bias,
            topology_distance=topology_distance,
            image_shift=image_shift,
            geometry_pair_mask=geometry_pair_mask,
            image_distances=image_distances,
            image_valid=image_valid,
            explicit_images=explicit_images,
        )
        return self.feed_forward(x)


class SCAGEGraphEncoder(nn.Module):
    """Original SCAGE backbone using star-linked/PBC polymer inputs."""

    uses_geometry = True
    architecture_name = "scage_distance_bias_transformer"

    def __init__(
        self,
        num_layer=6,
        emb_dim=512,
        num_heads=16,
        dist_bars=(20.0, 50.0),
        dropout=0.1,
        attention_dropout=0.1,
        ffn_hidden_dim=256,
        num_kernels=128,
        use_pbc_distance=True,
        pooling=None,
        distance_mode="mips_dual",
        distance_rbf=32,
        distance_cutoff=12.0,
        distance_scales=(4.0, 8.0, 12.0),
        distance_taus=(0.5, 1.0, 1.5),
        topology_bias=True,
        topology_max_distance=20,
        topology_locality_mode="soft",
        topology_locality_threshold=5,
        topology_locality_tau=1.0,
        periodic_image_mode="explicit_images",
        periodic_image_cap=1,
        periodic_image_temperature=0.5,
        force_topology_only=False,
        use_descriptors=False,
    ):
        super().__init__()
        if not dist_bars:
            raise ValueError("SCAGE requires at least one distance percentile")
        self.num_layer = int(num_layer)
        self.emb_dim = int(emb_dim)
        self.num_heads = int(num_heads)
        self.ffn_hidden_dim = int(ffn_hidden_dim)
        self.num_kernels = int(num_kernels)
        self.dist_bars = tuple(float(value) for value in dist_bars)
        self.use_pbc_distance = bool(use_pbc_distance)
        self.distance_mode = str(distance_mode).lower()
        if self.distance_mode not in {"bias", "mask", "multiscale_bias", "mips_dual"}:
            raise ValueError("unsupported SCAGE distance_mode")
        self.distance_rbf = int(distance_rbf)
        self.distance_cutoff = float(distance_cutoff)
        self.distance_scales = tuple(float(value) for value in distance_scales)
        self.distance_taus = tuple(float(value) for value in distance_taus)
        self.use_topology_bias = bool(topology_bias)
        self.topology_max_distance = int(topology_max_distance)
        self.topology_locality_mode = str(topology_locality_mode)
        self.topology_locality_threshold = int(topology_locality_threshold)
        self.topology_locality_tau = float(topology_locality_tau)
        self.periodic_image_mode = str(periodic_image_mode).lower()
        if self.periodic_image_mode not in {"fixed_min", "dynamic_nearest", "dynamic_soft", "explicit_images"}:
            raise ValueError(
                "periodic_image_mode must be fixed_min, dynamic_nearest, dynamic_soft, or explicit_images"
            )
        self.periodic_image_cap = int(periodic_image_cap)
        self.periodic_image_temperature = float(periodic_image_temperature)
        self.force_topology_only = bool(force_topology_only)
        self.use_descriptors = bool(use_descriptors)
        self.architecture_name = (
            "scage_mips_dual_topology_geometry_transformer"
            if self.distance_mode == "mips_dual"
            else ("scage_multiscale_distance_bias_transformer"
            if self.distance_mode == "multiscale_bias"
            else ("scage_distance_bias_transformer" if self.distance_mode == "bias"
                  else "original_multiscale_transformer"))
        )
        self.atom_embedding = SCAGEAtomEmbedding(self.emb_dim, self.num_kernels)
        self.topology_bias = (
            GraphTopologyBias(
                num_heads=self.num_heads,
                max_distance=self.topology_max_distance,
            )
            if self.use_topology_bias else None
        )
        self.layers = nn.ModuleList([
            EncoderAtomLayer(
                emb_dim=self.emb_dim,
                ffn_hidden_dim=self.ffn_hidden_dim,
                num_heads=self.num_heads,
                num_scales=len(self.dist_bars),
                dropout=dropout,
                attention_dropout=attention_dropout,
                distance_mode=self.distance_mode,
                distance_rbf=self.distance_rbf,
                distance_cutoff=self.distance_cutoff,
                distance_scales=self.distance_scales,
                distance_taus=self.distance_taus,
                locality_mode=self.topology_locality_mode,
                locality_threshold=self.topology_locality_threshold,
                locality_tau=self.topology_locality_tau,
                periodic_image_cap=self.periodic_image_cap,
            )
            for _ in range(self.num_layer)
        ])
        descriptor_dims = (11, 60, 80, 210, 224, 114)
        self.descriptor_projections = nn.ModuleList([
            nn.Sequential(nn.Linear(dim, self.emb_dim), nn.GELU(), nn.LayerNorm(self.emb_dim))
            for dim in descriptor_dims
        ])
        self.descriptor_type_embedding = nn.Embedding(len(descriptor_dims), self.emb_dim)
        self.descriptor_cross_attention = nn.MultiheadAttention(
            self.emb_dim, self.num_heads, dropout=attention_dropout, batch_first=True
        )
        self.descriptor_layer_norm = nn.LayerNorm(self.emb_dim)
        self.graph_readout_attention = nn.MultiheadAttention(
            self.emb_dim, self.num_heads, dropout=attention_dropout, batch_first=True
        )
        self.graph_readout_layer_norm = nn.LayerNorm(self.emb_dim)
        self.last_distance_sources = []

    def forward(self, data_or_x, edge_index=None, edge_attr=None, batch=None):
        if not hasattr(data_or_x, "x"):
            raise ValueError("Original-input SCAGE requires a PyG Data/Batch object")
        return self._forward_impl(
            data_or_x,
            atom_mask=None,
            force_topology=self.force_topology_only,
            disable_descriptors=not self.use_descriptors,
        )

    def forward_with_x(self, data, x):
        if x.shape != data.x.shape:
            raise ValueError("Masked SCAGE x override must have the same shape as data.x")
        atom_mask = (x.abs().sum(dim=-1) == 0) & (data.x.abs().sum(dim=-1) != 0)
        return self._forward_impl(
            data,
            atom_mask=atom_mask,
            force_topology=self.force_topology_only,
            disable_descriptors=not self.use_descriptors,
        )

    def forward_topology_only(self, data):
        return self._forward_impl(
            data, atom_mask=None, force_topology=True, disable_descriptors=True
        )

    def forward_geometry_pretext(self, data):
        """Encode geometry-masked pretext inputs without descriptor leakage."""
        return self._forward_impl(
            data, atom_mask=None, force_topology=False, disable_descriptors=True
        )

    def _forward_impl(self, data, atom_mask, force_topology, disable_descriptors):
        self._validate_scage_fields(data)
        batch = getattr(data, "batch", data.x.new_zeros(data.x.size(0), dtype=torch.long))
        order, padding_mask = self._dense_layout(batch)
        categorical = {
            name: self._dense_values(getattr(data, name), order, padding_mask, torch.long)
            for name in SCAGE_CATEGORICAL_FEATURES
        }
        continuous = {
            name: self._dense_values(getattr(data, name), order, padding_mask, torch.float)
            for name in ("mass", "van_der_waals_radius", "partial_charge")
        }
        backbone_role = self._dense_values(
            data.scage_backbone_role, order, padding_mask, torch.long
        )
        ru_index = self._dense_values(
            getattr(data, "scage_ru_index", torch.zeros_like(data.scage_backbone_role)),
            order, padding_mask, torch.long,
        )
        dense_atom_mask = None
        if atom_mask is not None:
            dense_atom_mask = self._dense_values(atom_mask, order, padding_mask, torch.bool)

        h = self.atom_embedding(
            categorical,
            continuous,
            padding_mask,
            atom_mask=dense_atom_mask,
            backbone_role=backbone_role,
            ru_index=ru_index,
        )
        atom_distance, atom_image_shift, atom_image_distances, image_valid, sources = self._distance_matrix(
            data, data.edge_index, batch, order, padding_mask.size(1), h.device, h.dtype, force_topology
        )
        self.last_distance_sources = sources
        data.scage_distance_sources = sources
        thresholds = (
            self._distance_thresholds(atom_distance, padding_mask)
            if self.distance_mode == "mask"
            else atom_distance.new_zeros((atom_distance.size(0), 0))
        )
        distance = self._with_graph_token_distance(atom_distance)
        image_shift = self._with_graph_token_distance(atom_image_shift)
        image_distances = self._with_graph_token_image_distances(atom_image_distances)
        geometry_pair_mask = getattr(data, "scage_geometry_pair_mask", None)
        if geometry_pair_mask is not None:
            geometry_pair_mask = geometry_pair_mask.to(device=h.device, dtype=torch.bool)
            token_pair_mask = torch.zeros_like(distance, dtype=torch.bool)
            token_pair_mask[:, 1:, 1:] = geometry_pair_mask[:, :padding_mask.size(1), :padding_mask.size(1)]
            geometry_pair_mask = token_pair_mask
        topology_bias = None
        topology_distance = None
        if self.topology_bias is not None:
            atom_topology_distance, path_bond_fields = self._cached_topology_matrices(
                data, padding_mask.size(1), h.device
            )
            topology_bias = self.topology_bias(atom_topology_distance, path_bond_fields)
            topology_distance = self._with_graph_token_distance(atom_topology_distance.float())
        geometry_valid = torch.tensor(
            [source in {"screw", "pbc", "smer_center", "euclidean"} for source in sources],
            dtype=torch.bool,
            device=h.device,
        )
        token_padding = torch.zeros((padding_mask.size(0), 1), dtype=torch.bool, device=h.device)
        full_padding = torch.cat([token_padding, padding_mask], dim=1)
        for layer in self.layers:
            h = layer(
                h,
                distance,
                thresholds,
                full_padding,
                geometry_valid=geometry_valid,
                topology_bias=topology_bias,
                topology_distance=topology_distance,
                image_shift=image_shift,
                geometry_pair_mask=geometry_pair_mask,
                image_distances=image_distances,
                image_valid=image_valid,
                explicit_images=self.periodic_image_mode == "explicit_images",
            )

        atoms = h[:, 1:, :]
        if disable_descriptors:
            atoms = self.descriptor_layer_norm(atoms)
        else:
            descriptor_tokens, descriptor_valid = self._descriptor_tokens(data, h.device, h.dtype)
            with torch.autocast(device_type=h.device.type, enabled=False):
                descriptor_update, _ = self.descriptor_cross_attention(
                    atoms.float(), descriptor_tokens.float(), descriptor_tokens.float(), need_weights=False
                )
            descriptor_update = descriptor_update.to(h.dtype)
            descriptor_update = descriptor_update * descriptor_valid.view(-1, 1, 1).to(h.dtype)
            atoms = self.descriptor_layer_norm(atoms + descriptor_update)
        atoms = atoms.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        graph_token = h[:, :1, :]
        with torch.autocast(device_type=h.device.type, enabled=False):
            graph_update, _ = self.graph_readout_attention(
                graph_token.float(), atoms.float(), atoms.float(),
                key_padding_mask=padding_mask, need_weights=False,
            )
        graph_update = graph_update.to(h.dtype)
        graph_embedding = self.graph_readout_layer_norm(graph_token + graph_update).squeeze(1)
        flat_atoms = atoms[~padding_mask]
        return graph_embedding, flat_atoms

    @staticmethod
    def _validate_scage_fields(data):
        required = list(SCAGE_CATEGORICAL_FEATURES) + [
            "mass", "van_der_waals_radius", "partial_charge", "scage_backbone_role",
            "scage_spd", "scage_path_bond_fields",
        ]
        missing = [name for name in required if not hasattr(data, name)]
        if missing:
            raise ValueError(
                "SCAGE original atom input is missing fields: " + ", ".join(missing)
                + ". Rebuild the feature cache with scage-input-v1."
            )

    def _descriptor_tokens(self, data, device, dtype):
        names = ("shape", "usrcat", "autocorr3d", "rdf", "morse", "whim")
        tokens = []
        for type_idx, (name, projection) in enumerate(zip(names, self.descriptor_projections)):
            value = getattr(data, f"scage_descriptor_{name}", None)
            if value is None:
                batch_size = int(data.scage_spd.size(0))
                value = projection[0].weight.new_zeros(batch_size, projection[0].in_features)
            token = projection(value.to(device=device, dtype=dtype))
            type_ids = torch.full((token.size(0),), type_idx, device=device, dtype=torch.long)
            tokens.append(token + self.descriptor_type_embedding(type_ids))
        valid = getattr(
            data, "scage_descriptor_valid",
            torch.zeros(tokens[0].size(0), dtype=torch.bool, device=device),
        ).to(device=device, dtype=torch.bool)
        return torch.stack(tokens, dim=1), valid

    def _cached_topology_matrices(self, data, max_nodes, device):
        if not hasattr(data, "scage_spd") or not hasattr(data, "scage_path_bond_fields"):
            raise ValueError("SCAGE MIPS topology fields are missing; rebuild scage-mips-v1 cache")
        distance = data.scage_spd[:, :max_nodes, :max_nodes].to(device=device, dtype=torch.long)
        path = data.scage_path_bond_fields[:, :max_nodes, :max_nodes].to(
            device=device, dtype=torch.long
        )
        return distance, path

    @staticmethod
    def _dense_layout(batch):
        batch_size = int(batch.max().item()) + 1 if batch.numel() else 1
        order = [torch.nonzero(batch == idx, as_tuple=False).view(-1) for idx in range(batch_size)]
        max_nodes = max((int(idx.numel()) for idx in order), default=0)
        padding = torch.ones(batch_size, max_nodes, dtype=torch.bool, device=batch.device)
        for graph_idx, idx in enumerate(order):
            padding[graph_idx, :idx.numel()] = False
        return order, padding

    @staticmethod
    def _dense_values(values, order, padding_mask, dtype):
        dense = torch.zeros(padding_mask.shape, device=values.device, dtype=dtype)
        for graph_idx, idx in enumerate(order):
            if idx.numel():
                dense[graph_idx, :idx.numel()] = values[idx].to(dtype=dtype)
        return dense

    def _distance_thresholds(self, distance, padding_mask):
        thresholds = distance.new_empty((distance.size(0), len(self.dist_bars)))
        for graph_idx in range(distance.size(0)):
            count = int((~padding_mask[graph_idx]).sum().item())
            values = distance[graph_idx, :count, :count].reshape(-1)
            values = values[torch.isfinite(values)]
            if values.numel() == 0:
                thresholds[graph_idx] = float("inf")
                continue
            quantiles = values.new_tensor(self.dist_bars).clamp(0, 100) / 100.0
            thresholds[graph_idx] = torch.quantile(values, quantiles)
        return thresholds

    def _distance_matrix(self, data, edge_index, batch, order, max_nodes, device, dtype, force_topology):
        distance = torch.full((len(order), max_nodes, max_nodes), float("inf"), device=device, dtype=dtype)
        image_shift = torch.zeros((len(order), max_nodes, max_nodes), device=device, dtype=dtype)
        image_count = 2 * self.periodic_image_cap + 1
        image_distances = torch.full(
            (len(order), image_count, max_nodes, max_nodes),
            float("inf"), device=device, dtype=dtype,
        )
        image_valid = torch.zeros((len(order), image_count), device=device, dtype=torch.bool)
        sources = []
        for graph_idx, node_idx in enumerate(order):
            count = node_idx.numel()
            coords = None if force_topology else self._coords_for_graph(data, graph_idx, node_idx, device, dtype)
            if coords is None:
                distance[graph_idx, :count, :count] = self._topology_distance(
                    edge_index, batch, graph_idx, node_idx, count, device, dtype
                )
                sources.append("topology_fallback")
            else:
                coordinate_distance, coordinate_shift, candidate_distances, candidate_valid = (
                    self._coordinate_distance(data, coords, graph_idx)
                )
                distance[graph_idx, :count, :count] = coordinate_distance
                image_shift[graph_idx, :count, :count] = coordinate_shift
                image_distances[graph_idx, :, :count, :count] = candidate_distances
                image_valid[graph_idx] = candidate_valid
                if self._has_screw(data, graph_idx):
                    sources.append("screw")
                elif self._has_smer(data, graph_idx):
                    sources.append("smer_center")
                else:
                    sources.append("pbc" if self._has_pbc(data, graph_idx) else "euclidean")
        return distance, image_shift, image_distances, image_valid, sources

    @staticmethod
    def _has_pbc(data, graph_idx):
        return bool(
            hasattr(data, "pbc") and bool(data.pbc[graph_idx].bool().any().item())
        )

    @staticmethod
    def _has_screw(data, graph_idx):
        return bool(hasattr(data, "screw_valid") and bool(data.screw_valid.flatten()[graph_idx].item()))

    @staticmethod
    def _has_smer(data, graph_idx):
        return bool(hasattr(data, "smer_valid") and bool(data.smer_valid.flatten()[graph_idx].item()))

    def _coords_for_graph(self, data, graph_idx, node_idx, device, dtype):
        coordinate_ok = getattr(data, "geom_coordinate_ok", getattr(data, "geom_build_ok", True))
        if coordinate_ok is not None:
            if torch.is_tensor(coordinate_ok):
                if not bool(coordinate_ok.flatten()[graph_idx].item()):
                    return None
            elif not bool(coordinate_ok):
                return None
        if not hasattr(data, "pos3d") or not hasattr(data, "graph_to_geom_index"):
            return None
        mapping = data.graph_to_geom_index[node_idx].to(device=device)
        if mapping.numel() != node_idx.numel() or (mapping < 0).any() or (mapping >= data.pos3d.size(0)).any():
            return None
        coords = data.pos3d[mapping].to(device=device, dtype=dtype)
        if coords.size(0) != node_idx.numel() or not torch.isfinite(coords).all():
            return None
        geom_atomic_numbers = getattr(data, "z", getattr(data, "x3d", None))
        if geom_atomic_numbers is None:
            return None
        expected = data.atomic_num[node_idx].to(device=device)
        # atomic_num stores the original SCAGE vocabulary index: H is 0, C is 5, etc.
        expected_atomic_numbers = expected + 1
        misc_index = len(SCAGE_ATOM_VOCABS["atomic_num"]) - 1
        mapped_z = geom_atomic_numbers[mapping].to(device=device).long()
        known = expected != misc_index
        if known.any() and not torch.equal(expected_atomic_numbers[known].long(), mapped_z[known]):
            return None
        return coords

    def _coordinate_distance(self, data, coords, graph_idx):
        diff = coords.unsqueeze(1) - coords.unsqueeze(0)
        image_count = 2 * self.periodic_image_cap + 1
        candidate_slots = coords.new_full(
            (image_count, coords.size(0), coords.size(0)), float("inf")
        )
        candidate_valid = torch.zeros(image_count, dtype=torch.bool, device=coords.device)
        center_slot = self.periodic_image_cap
        if self._has_screw(data, graph_idx):
            rotation = data.screw_rotation[graph_idx].to(device=coords.device, dtype=coords.dtype)
            translation = data.screw_translation[graph_idx].to(device=coords.device, dtype=coords.dtype)
            shifts, images = self._screw_images(coords, rotation, translation)
            candidate_distances = torch.stack([
                torch.cdist(coords, image) for image in images
            ], dim=0)
            for local_idx, shift in enumerate(shifts):
                slot = center_slot + int(shift)
                candidate_slots[slot] = candidate_distances[local_idx]
                candidate_valid[slot] = True
            shift_values = coords.new_tensor(shifts).view(-1, 1, 1)
            if self.periodic_image_mode in {"fixed_min", "dynamic_nearest"}:
                minimum, indices = candidate_distances.min(dim=0)
                selected_shift = shift_values.expand_as(candidate_distances).gather(
                    0, indices.unsqueeze(0)
                ).squeeze(0)
                return minimum, selected_shift, candidate_slots, candidate_valid
            if self.periodic_image_mode == "explicit_images":
                minimum, indices = candidate_distances.min(dim=0)
                selected_shift = shift_values.expand_as(candidate_distances).gather(
                    0, indices.unsqueeze(0)
                ).squeeze(0)
                return minimum, selected_shift, candidate_slots, candidate_valid
            weights = F.softmax(
                -candidate_distances / max(self.periodic_image_temperature, 1e-6), dim=0
            )
            soft_distance = (weights * candidate_distances).sum(dim=0)
            soft_shift = (weights * shift_values).sum(dim=0)
            diagonal = torch.arange(coords.size(0), device=coords.device)
            soft_distance[diagonal, diagonal] = 0.0
            soft_shift[diagonal, diagonal] = 0.0
            return soft_distance, soft_shift, candidate_slots, candidate_valid
        if self._has_smer(data, graph_idx) and hasattr(data, "smer_image_pos3d"):
            node_idx = torch.nonzero(data.batch == graph_idx, as_tuple=False).flatten()
            finite_images = data.smer_image_pos3d[node_idx].to(
                device=coords.device, dtype=coords.dtype
            ).permute(1, 0, 2)
            if finite_images.shape == (3, coords.size(0), 3) and torch.isfinite(finite_images).all():
                stacked = torch.stack(
                    [torch.cdist(coords, finite_images[cell]) for cell in range(3)], dim=0
                )
                candidate_slots[center_slot - 1:center_slot + 2] = stacked
                candidate_valid[center_slot - 1:center_slot + 2] = True
                minimum, indices = stacked.min(0)
                return minimum, indices.ne(1).to(coords.dtype), candidate_slots, candidate_valid
        if self.use_pbc_distance and self._has_pbc(data, graph_idx):
            active_axes = torch.nonzero(data.pbc[graph_idx].bool(), as_tuple=False).flatten()
            if active_axes.numel() != 1:
                raise ValueError("SCAGE currently supports exactly one periodic axis")
            vector_t = data.cell[graph_idx, int(active_axes[0])].to(
                device=coords.device, dtype=coords.dtype
            )
            vector_norm = vector_t.norm().clamp_min(0.5)
            if self.periodic_image_mode == "explicit_images":
                nmax = self.periodic_image_cap
            elif self.periodic_image_mode == "fixed_min":
                nmax = 1
            else:
                nmax = int(torch.ceil(self.distance_cutoff / vector_norm).clamp(
                    1, self.periodic_image_cap
                ).item())
            shifts = list(range(-nmax, nmax + 1))
            stacked = torch.stack([
                torch.linalg.vector_norm(
                    diff - float(shift) * vector_t.view(1, 1, 3), dim=-1
                )
                for shift in shifts
            ])
            for local_idx, shift in enumerate(shifts):
                candidate_slots[center_slot + shift] = stacked[local_idx]
                candidate_valid[center_slot + shift] = True
            minimum, indices = stacked.min(0)
            shift_values = coords.new_tensor(shifts).view(-1, 1, 1).expand_as(stacked)
            selected_shift = shift_values.gather(0, indices.unsqueeze(0)).squeeze(0)
            return minimum, selected_shift, candidate_slots, candidate_valid
        euclidean = torch.linalg.vector_norm(diff, dim=-1)
        candidate_slots[center_slot] = euclidean
        candidate_valid[center_slot] = True
        return (
            euclidean,
            torch.zeros(diff.shape[:2], device=coords.device, dtype=coords.dtype),
            candidate_slots,
            candidate_valid,
        )

    def _screw_images(self, coords, rotation, translation):
        if self.periodic_image_mode == "fixed_min":
            nmax = 1
        elif self.periodic_image_mode == "explicit_images":
            nmax = self.periodic_image_cap
        else:
            skew_axis = torch.stack([
                rotation[2, 1] - rotation[1, 2],
                rotation[0, 2] - rotation[2, 0],
                rotation[1, 0] - rotation[0, 1],
            ])
            axis = skew_axis / skew_axis.norm().clamp_min(1e-8)
            if skew_axis.norm() < 1e-6:
                axis = translation / translation.norm().clamp_min(1e-8)
            centered = coords - coords.mean(dim=0, keepdim=True)
            axial = centered @ axis
            radial = centered - axial.unsqueeze(-1) * axis
            radial_radius = radial.norm(dim=-1).max()
            rise = torch.dot(axis, translation).abs().clamp_min(0.5)
            estimate = torch.ceil(
                (self.distance_cutoff + 2.0 * radial_radius) / rise
            ).long()
            nmax = int(estimate.clamp(1, self.periodic_image_cap).item())
        image_by_shift = {0: coords}
        for shift in range(1, nmax + 1):
            image_by_shift[shift] = image_by_shift[shift - 1] @ rotation.transpose(0, 1) + translation
            image_by_shift[-shift] = (image_by_shift[-shift + 1] - translation) @ rotation
        shifts = list(range(-nmax, nmax + 1))
        return shifts, [image_by_shift[shift] for shift in shifts]

    @staticmethod
    def _with_graph_token_distance(distance):
        batch_size, nodes, _ = distance.shape
        result = distance.new_zeros(batch_size, nodes + 1, nodes + 1)
        result[:, 1:, 1:] = distance
        return result

    @staticmethod
    def _with_graph_token_image_distances(distance):
        batch_size, images, nodes, _ = distance.shape
        result = distance.new_full(
            (batch_size, images, nodes + 1, nodes + 1), float("inf")
        )
        result[:, :, 1:, 1:] = distance
        return result

    @staticmethod
    def _topology_distance(edge_index, batch, graph_idx, node_idx, count, device, dtype):
        distance = torch.full((count, count), float(count + 1), device=device, dtype=dtype)
        diagonal = torch.arange(count, device=device)
        distance[diagonal, diagonal] = 0.0
        if edge_index is not None and edge_index.numel():
            local = torch.full((batch.numel(),), -1, device=device, dtype=torch.long)
            local[node_idx] = torch.arange(count, device=device)
            src, dst = edge_index.to(device)
            keep = (batch[src] == graph_idx) & (batch[dst] == graph_idx)
            source, target = local[src[keep]], local[dst[keep]]
            valid = (source >= 0) & (target >= 0)
            distance[source[valid], target[valid]] = 1.0
        for pivot in range(count):
            distance = torch.minimum(
                distance,
                distance[:, pivot:pivot + 1] + distance[pivot:pivot + 1, :],
            )
        return distance
