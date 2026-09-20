"""GLT-GALPH r2 model: one implementation for the four pretraining arms.

``summary_mode`` picks the graph representation route:
  * 'mean' (Route-N): g = graph mean of real token states; no virtual token
  * 'cls'  (Route-C): one CLS token per graph joins all six layers through its
    own virtual-edge biases; g = the CLS state

``ph_mode`` adds the PH modality ('global'): 8-patch Betti-profile encoder,
masked-patch reconstruction and a zero-initialized residual into the graph
summary (Route-N: into g3, Route-C: into CLS_3D).

N0/N1/C0/C1 are exactly (mean|cls) x (none|global).  Old B_FP paths stay
untouched: the existing O8/Galformer3D components are reused so shared
parameters keep their names.  Special relation types follow the existing
vocabulary (BONDED 0..25754, UNBONDED 25755..25855, VIRTUAL 25856, MASK 25857,
PADDING 25858).
"""
import torch
from torch import nn
from torch.nn import functional as F

from .glt_dual import (BondPathO8, GalformerTrimer3D, element_index, mean_pool,
                       triplet_type)

PH_BINS = 32
PH_CHANNELS = 3
PH_PATCHES = 8
PH_RADII_PER_PATCH = PH_BINS // PH_PATCHES      # 4
PH_PATCH_DIM = PH_CHANNELS * PH_RADII_PER_PATCH  # 12
PH_PROFILE_DIM = PH_CHANNELS * PH_BINS           # 96
MASK_TYPE = 25857
LINE_CLASSES = 25755
KEEP, MASK, REPLACE = 0, 1, 2


def _head(in_dim, hidden, out_dim):
    return nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, out_dim))


class PHProfileEncoder(nn.Module):
    """[B,3,32] Betti profile -> [B,8,512] patch tokens."""

    def __init__(self, hidden=512):
        super().__init__()
        self.patch = _head(PH_PATCH_DIM, 128, hidden)
        self.scale_embedding = nn.Parameter(torch.zeros(PH_PATCHES, hidden))
        self.mask_token = nn.Parameter(torch.zeros(hidden))
        self.norm = nn.LayerNorm(hidden)

    def patchify(self, profile):
        """[3,32] or [B,3,32] -> [B,8,12]: 8 radius patches, 3 channels x 4 radii."""
        batch = profile.reshape(-1, PH_CHANNELS, PH_BINS)
        shaped = batch.reshape(batch.shape[0], PH_CHANNELS, PH_PATCHES,
                               PH_RADII_PER_PATCH)
        return shaped.permute(0, 2, 1, 3).reshape(batch.shape[0], PH_PATCHES,
                                                  PH_PATCH_DIM)

    def forward(self, profile, patch_mask):
        if patch_mask.dim() == 1:
            patch_mask = patch_mask.unsqueeze(0)
        patches = self.patchify(profile)
        tokens = self.patch(patches) + self.scale_embedding.unsqueeze(0)
        # A masked patch contributes no raw value: only the learned mask token
        # and its filtration-scale embedding.
        masked_token = (self.mask_token + self.scale_embedding).unsqueeze(0)
        return torch.where(patch_mask.unsqueeze(-1), masked_token, tokens)

    def summarize(self, tokens):
        return self.norm(tokens.mean(1))


class GLTGalPH(nn.Module):
    architecture_name = 'O8-GalformerTrimer-GalPretrain'

    def __init__(self, summary_mode='mean', ph_mode=None, dropout=0.1):
        super().__init__()
        if summary_mode not in ('mean', 'cls'):
            raise ValueError('summary_mode must be mean or cls')
        if ph_mode not in (None, 'global'):
            raise ValueError('ph_mode must be none or global')
        self.summary_mode = summary_mode
        self.ph_mode = ph_mode
        if ph_mode == 'global':
            self.architecture_name = 'O8-GalformerTrimer-GalPretrain-PHGlobal'
        self.o8 = BondPathO8(dropout)
        self.glt = GalformerTrimer3D(dropout)
        self.mask_2d_embedding = nn.Parameter(torch.zeros(512))
        self.mask_3d_embedding = nn.Parameter(torch.zeros(512))
        self.head_2d = _head(512, 512, 101)
        self.head_3d = _head(512, 512, LINE_CLASSES)
        self.cl_proj2 = _head(512, 256, 128)
        self.cl_proj3 = _head(512, 256, 128)
        nn.init.normal_(self.mask_2d_embedding, std=0.02)
        nn.init.normal_(self.mask_3d_embedding, std=0.02)
        if summary_mode == 'cls':
            self.cls_2d = nn.Parameter(torch.zeros(512))
            self.cls_3d = nn.Parameter(torch.zeros(512))
            self.virtual_to_real_bias = nn.Parameter(torch.zeros(8))
            self.real_to_virtual_bias = nn.Parameter(torch.zeros(8))
            self.virtual_self_bias = nn.Parameter(torch.zeros(8))
            nn.init.normal_(self.cls_2d, std=0.02)
            nn.init.normal_(self.cls_3d, std=0.02)
        if ph_mode == 'global':
            self.ph_encoder = PHProfileEncoder()
            self.ph_to_summary = nn.Linear(512, 512, bias=False)
            self.alpha_ph = nn.Parameter(torch.zeros(1))
            self.ph_head = _head(512, 256, PH_PROFILE_DIM)

    # ------------------------------------------------------------------ 2D
    def _o8_states(self, data):
        """O8 atom states with the 2D mask/replace applied before any consumer."""
        initial = self.o8.atom_embedding(data)
        rows = getattr(data, 'mask2d_rows', None)
        if rows is not None and rows.numel():
            policy = data.mask2d_policy.long()
            initial = initial.clone()
            mask_rows = rows[policy == MASK]
            replace_rows = rows[policy == REPLACE]
            if mask_rows.numel():
                initial[mask_rows] = self.mask_2d_embedding
            if replace_rows.numel():
                initial[replace_rows] = self._donor_embeddings(data)
        # PathNode and bond-path biases read the masked/replaced states, so no
        # original atom feature can leak through the attention bias.
        bias = (self.o8.spd_embedding(data.lga_spd.long())
                + self.o8.path_bias(initial, data)
                + self.o8.bond_bias(data.bond_path_features, data.bond_path_mask.bool()))
        relation_mask = getattr(data, 'lga_relation_mask', None)
        if relation_mask is not None:
            bias = bias.masked_fill(relation_mask.bool().unsqueeze(-1), 0)
        source, target = data.lga_edge_index.long()
        return initial, bias, source, target

    def _donor_embeddings(self, data):
        """Embedding of the donor atom features for the REPLACE rows."""
        donor = data.mask2d_donor_atoms.long()
        features = data.mips_x.float()[donor]
        backbone = data.mips_backbone_mask.to(torch.float32)[donor].unsqueeze(-1)
        return self.o8.atom_embedding.projection(torch.cat([features, backbone], -1))

    # ---------------------------------------------------------------- CLS
    def _append_cls(self, states, graph_index, graphs, bias, source, target, cls):
        n_real = states.size(0)
        combined = torch.cat([states, cls.expand(graphs, states.size(-1))], 0)
        nodes = torch.arange(n_real, device=states.device)
        cls_of_node = n_real + graph_index
        self_cls = n_real + torch.arange(graphs, device=states.device)
        extra_source = torch.cat([nodes, cls_of_node, self_cls])
        extra_target = torch.cat([cls_of_node, nodes, self_cls])
        extra_bias = torch.cat([self.real_to_virtual_bias.expand(n_real, 8),
                                self.virtual_to_real_bias.expand(n_real, 8),
                                self.virtual_self_bias.expand(graphs, 8)])
        return (combined, torch.cat([bias, extra_bias]),
                torch.cat([source, extra_source]), torch.cat([target, extra_target]))

    # ------------------------------------------------------------------ 3D
    def _glt_states(self, data, graphs):
        za = element_index(data.bond_z_a)
        zb = element_index(data.bond_z_b)
        z_a, z_b = data.bond_z_a.long(), data.bond_z_b.long()
        bond_type = data.bond_type.long()
        distance = data.bond_distance.float()
        rows = getattr(data, 'mask3d_rows', None)
        if rows is not None and rows.numel():
            policy = data.mask3d_policy.long()
            replace_rows = rows[policy == REPLACE]
            if replace_rows.numel():
                # Donor line-node *inputs* only: element pair, bond type and
                # distance.  Line topology and angles stay with the real graph.
                donor = data.mask3d_donor_atoms.long()      # one row per REPLACE row
                za, zb, z_a, z_b = za.clone(), zb.clone(), z_a.clone(), z_b.clone()
                bond_type, distance = bond_type.clone(), distance.clone()
                za[replace_rows] = element_index(data.bond_z_a.long()[donor])
                zb[replace_rows] = element_index(data.bond_z_b.long()[donor])
                z_a[replace_rows] = data.bond_z_a.long()[donor]
                z_b[replace_rows] = data.bond_z_b.long()[donor]
                bond_type[replace_rows] = data.bond_type.long()[donor]
                distance[replace_rows] = data.bond_distance.float()[donor]
        endpoints = self.glt.endpoint(F.one_hot(za, 101).float()) \
            + self.glt.endpoint(F.one_hot(zb, 101).float())
        radial = self.glt.distance_projection(
            self.glt.distance_basis(distance, torch.stack([za, zb], -1)))
        states = self.glt.triplet(torch.cat([endpoints, radial], -1))
        types = triplet_type(z_a, z_b, bond_type)
        if rows is not None and rows.numel():
            mask_rows = rows[data.mask3d_policy.long() == MASK]
            if mask_rows.numel():
                states = states.clone()
                states[mask_rows] = self.mask_3d_embedding
                types = types.clone()
                types[mask_rows] = MASK_TYPE
        path_bias = self.glt.angle_bias(types, data.line_path, data.line_angle,
                                        data.line_mask)
        bias = mean_pool(path_bias, data.line_path_group, data.line_source.numel())
        counts = torch.bincount(data.bond_batch.long(), minlength=graphs)
        valid = data.geometry_valid.bool() & (counts > 0)
        return states, bias, data.line_source.long(), data.line_target.long(), valid

    # ------------------------------------------------------------- forward
    def forward(self, data):
        graphs = data.graph_available.numel()
        atom_graph = data.canonical_graph_index.long()
        atoms, bias2, source2, target2 = self._o8_states(data)
        if self.summary_mode == 'cls':
            atoms, bias2, source2, target2 = self._append_cls(
                atoms, atom_graph, graphs, bias2, source2, target2, self.cls_2d)
        for layer in self.o8.layers:
            atoms = layer(atoms, source2, target2, bias2)
        n_atom = data.mips_x.size(0)
        atom_hidden = atoms[:n_atom] * data.graph_available.bool()[atom_graph].unsqueeze(-1)
        cls2 = atoms[n_atom:] if self.summary_mode == 'cls' else None

        bonds, bias3, source3, target3, valid3 = self._glt_states(data, graphs)
        bond_graph = data.bond_batch.long()
        if self.summary_mode == 'cls':
            bonds, bias3, source3, target3 = self._append_cls(
                bonds, bond_graph, graphs, bias3, source3, target3, self.cls_3d)
        for layer in self.glt.layers:
            bonds = layer(bonds, source3, target3, bias3)
        n_bond = data.bond_distance.numel()
        bond_hidden = bonds[:n_bond]
        cls3 = bonds[n_bond:] if self.summary_mode == 'cls' else None

        if self.summary_mode == 'mean':
            g2 = mean_pool(atom_hidden, atom_graph, graphs)
            summary3 = mean_pool(bond_hidden, bond_graph, graphs)
        else:
            g2, summary3 = cls2, cls3
        summary3 = torch.where(valid3.unsqueeze(-1), summary3,
                               torch.zeros_like(summary3))

        ph_summary = None
        if self.ph_mode == 'global':
            tokens = self.ph_encoder(data.ph_profile.float(), data.ph_mask.bool())
            ph_summary = self.ph_encoder.summarize(tokens)
            residual = torch.tanh(self.alpha_ph) * self.ph_to_summary(ph_summary)
            residual = torch.where(data.ph_valid.bool().unsqueeze(-1), residual,
                                   torch.zeros_like(residual))
            summary3 = summary3 + residual
        return {'g2': g2, 'g3': summary3, 'cls2': cls2, 'cls3': cls3,
                'atom_states': atom_hidden, 'bond_states': bond_hidden,
                'line3d_valid': valid3, 'ph_summary': ph_summary}
