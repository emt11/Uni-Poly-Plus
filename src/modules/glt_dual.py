"""O8 bond-path + Galformer physical-Trimer encoder, without MD200.

References (MIT): robbenplus/MolGT src/models/modeling.py (bond path);
peizhenbai/Galformer model/{module_utils,model_3d}.py (Gaussian/Residual).
Project adaptations: 512/6/8, physical line-hop <=2, center-only readout,
explicit post-MLP padding masks, and no virtual CLS node.
"""

import math

import torch
from torch import nn
from torch_geometric.utils import softmax
from torch_scatter import scatter

from .mips_local_graph import MIPSLocalAtomEmbedding, MIPSSinglePathNodeBias
from .original_mips_knowledge_fusion import OriginalMIPSAttentiveFusion


def mean_pool(states, index, count):
    return scatter(states, index.long(), dim=0, dim_size=count, reduce='mean')


def locality_options(metadata):
    """Strict S3 architecture fields; old deployment packages retain their defaults."""
    names=('endpoint_revision','endpoint_module','endpoint_input_mode','endpoint_init_seed','fusion_attention_mode','incident_bias')
    if not metadata.get('endpoint_revision'):
        if any(metadata.get(k) is not None for k in names[1:4]):raise ValueError('incomplete endpoint metadata')
        return {}
    if any(k not in metadata for k in names):raise ValueError('missing S3 architecture metadata')
    if (metadata['endpoint_revision']!='glt_pt_locality_r3' or metadata['endpoint_module']!='mlp512_nobias_zero_last'
            or metadata['endpoint_input_mode'] not in ('zero','masked_mean') or type(metadata['endpoint_init_seed']) is not int
            or metadata['endpoint_init_seed']!=314159 or metadata['fusion_attention_mode'] not in ('global','incident_only')
            or type(metadata['incident_bias']) not in (int,float) or metadata['incident_bias']!=0.0
            or (metadata['fusion_attention_mode']=='global' and metadata['endpoint_input_mode']!='zero')):raise ValueError('invalid S3 architecture metadata')
    return {k:metadata[k] for k in names}


PHYSICAL_TORSION = dict(torsion_revision='glt_physical_torsion_r1',
    torsion_modules=True, torsion_mode='on', torsion_input_domain='physical_finite_trimer_all_heavy_bonds',
    torsion_feature='cos_phi_cos2phi', torsion_phi='projected_b0_backward_v1',
    torsion_init_seed=314167, torsion_aggregation='valid_mean', torsion_gate_initial=0.0,
    torsion_central_epsilon=1e-8, torsion_projection_epsilon=1e-3)

S4_RESOURCE = dict(resource_revision=2, expected_world_size=2, microbatch=84,
                   accumulation=6, global_batch=1008, reference_protocol='history_only_four_rank')

def s4_resource_options(metadata):
    if not metadata.get('torsion_revision'):return {}
    torsion_options(metadata)
    if any(k not in metadata or type(metadata[k]) is not type(v) or metadata[k]!=v
           for k,v in S4_RESOURCE.items()):raise ValueError('invalid S4 resource metadata')
    from pathlib import Path
    common=str(Path(__file__).resolve().parents[2]/'results/glt_periodic_seam_20261009/r1/initialization/F_CE_step0.pt')
    if metadata.get('common_initialization')!=common:raise ValueError('invalid S4 public initialization source')
    return dict(S4_RESOURCE,common_initialization=common)

def torsion_options(metadata):
    """Validate the complete new contract, while preserving legacy behavior."""
    revision=metadata.get('torsion_revision')
    if revision is None:
        if any(k in metadata for k in PHYSICAL_TORSION if k not in ('torsion_revision','torsion_modules','torsion_mode')):
            raise ValueError('incomplete physical torsion metadata')
        return {}
    if any(k not in metadata or type(metadata[k]) is not type(v) or metadata[k]!=v
           for k,v in PHYSICAL_TORSION.items()):
        raise ValueError('invalid physical torsion metadata')
    return dict(PHYSICAL_TORSION)


class SharedBondPathBias(nn.Module):
    def __init__(self):
        super().__init__()
        self.type_emb = nn.Embedding(5, 8)
        self.conj_emb = nn.Embedding(2, 8)
        self.ring_emb = nn.Embedding(2, 8)
        self.stereo_emb = nn.Embedding(7, 8)
        self.position = nn.Parameter(torch.empty(2, 8, 8))
        for parameter in self.parameters():
            nn.init.normal_(parameter, std=0.02)

    def forward(self, features, mask):
        if features.shape != (*mask.shape, 14) or mask.ndim != 2 or mask.size(1) != 2:
            raise ValueError('bond paths must be [R,2,14] with [R,2] mask')
        if not torch.isfinite(features).all():
            raise ValueError('nonfinite bond features')
        selected = features[mask]
        if selected.numel() and (not ((selected == 0) | (selected == 1)).all()
                                or not (selected[:, :5].sum(-1) == 1).all()
                                or not (selected[:, 7:].sum(-1) == 1).all()):
            raise ValueError('invalid fourteen-dimensional bond chemistry')
        z = (self.type_emb(features[..., :5].argmax(-1))
             + self.conj_emb(features[..., 5].long())
             + self.ring_emb(features[..., 6].long())
             + self.stereo_emb(features[..., 7:].argmax(-1)))
        transformed = torch.einsum('rki,kij->rkj', z, self.position)
        return (transformed * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)


class SourceAttention512(nn.Module):
    """Source-Q, target-K; source values and incoming softmax. Isolated from legacy.

    scale_denominator only changes the scalar applied to the QK dot product;
    the structural/angle bias is added after that scaling and is never scaled.
    The default 512 preserves every existing checkpoint and entry point.
    """
    def __init__(self, dropout=0.1, *, scale_denominator=512):
        super().__init__()
        if type(scale_denominator) is not int or scale_denominator <= 0:
            raise ValueError('invalid attention scale denominator')
        self.qkv = nn.Linear(512, 1536)
        self.dropout = nn.Dropout(dropout)
        self.scale_denominator = scale_denominator
        self.scale = scale_denominator ** -0.5

    def forward(self, states, source, target, bias):
        q, k, v = self.qkv(states).reshape(-1, 3, 8, 64).unbind(1)
        score = (q[source].float() * k[target].float()).sum(-1) * self.scale
        weight = softmax(score + bias.float(), target, num_nodes=states.size(0)).to(v.dtype)
        messages = self.dropout(weight).unsqueeze(-1) * v[source]
        return torch.zeros_like(v).index_add(0, target, messages).reshape(-1, 512)


class TransformerBlock(nn.Module):
    def __init__(self, dropout=0.1, *, o8=False, attention_scale_denominator=512):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(512), nn.LayerNorm(512)
        self.attention = SourceAttention512(dropout, scale_denominator=attention_scale_denominator)
        self.output = nn.Linear(512, 512)
        self.dropout = nn.Dropout(dropout)
        # O8 retains its original intermediate dropout; Galformer has only
        # the output feature dropout in the two-dense-layer FFN.
        self.ffn = nn.Sequential(nn.Linear(512, 2048), nn.GELU(),
                                 nn.Dropout(dropout) if o8 else nn.Identity(),
                                 nn.Linear(2048, 512))

    def forward(self, states, source, target, bias):
        states = states + self.dropout(self.output(self.attention(self.norm1(states), source, target, bias)))
        return states + self.dropout(self.ffn(self.norm2(states)))


class BondPathO8(nn.Module):
    def __init__(self, dropout=0.1, *, attention_scale_denominator=512):
        super().__init__()
        self.atom_embedding = MIPSLocalAtomEmbedding(512)
        self.spd_embedding = nn.Embedding(3, 8)
        nn.init.zeros_(self.spd_embedding.weight)
        self.path_bias = MIPSSinglePathNodeBias(512, 8, 2)
        self.bond_bias = SharedBondPathBias()
        self.layers = nn.ModuleList([TransformerBlock(dropout, o8=True, attention_scale_denominator=attention_scale_denominator) for _ in range(6)])

    def forward(self, data, atom_mask=None, *, return_initial=False):
        initial = self.atom_embedding(data, atom_mask=atom_mask)
        bias = (self.spd_embedding(data.lga_spd.long()) + self.path_bias(initial, data)
                + self.bond_bias(data.bond_path_features, data.bond_path_mask.bool()))
        bias = bias.masked_fill(data.lga_relation_mask.bool().unsqueeze(-1), 0)
        states = initial
        source, target = data.lga_edge_index.long()
        for layer in self.layers:
            states = layer(states, source, target, bias)
        states = states * data.graph_available[data.canonical_graph_index].unsqueeze(-1)
        return (states,bias,initial) if return_initial else (states,bias)


def element_index(z):
    return torch.where((z >= 1) & (z <= 100), z.long() - 1, torch.full_like(z.long(), 100))


def triplet_type(z_a, z_b, bond_type):
    """Exact bonded portion of Galformer Vocab's low/type/high loop order."""
    low = torch.minimum(element_index(z_a), element_index(z_b))
    high = torch.maximum(element_index(z_a), element_index(z_b))
    return 5 * (101 * low - low * (low - 1) // 2) + bond_type.long() * (101 - low) + high - low


class TypeGaussian(nn.Module):
    def __init__(self, size, types):
        super().__init__()
        self.means = nn.Parameter(torch.empty(size))
        self.stds = nn.Parameter(torch.empty(size))
        self.mul = nn.Embedding(types, 1)
        self.bias = nn.Embedding(types, 1)
        nn.init.uniform_(self.means, 0, 3)
        nn.init.uniform_(self.stds, 0, 3)
        nn.init.ones_(self.mul.weight)
        nn.init.zeros_(self.bias.weight)

    def forward(self, value, pair_type):
        mul = self.mul(pair_type).sum(-2)
        bias = self.bias(pair_type).sum(-2)
        x = mul * value.float().unsqueeze(-1) + bias
        std = self.stds.float().abs() + 1e-2
        return torch.exp(-0.5 * ((x - self.means.float()) / std).square()) / (math.sqrt(2 * 3.14159) * std)


class PathAngleBias(nn.Module):
    def __init__(self):
        super().__init__()
        # 25755 bonded + 101 unbonded + virtual/mask/padding categories.
        self.gaussian = TypeGaussian(128, 25859)
        self.position = nn.ModuleList([
            nn.Sequential(nn.Linear(128, 128), nn.GELU(), nn.Linear(128, 128)) for _ in range(2)])
        self.heads = nn.Sequential(nn.Linear(128, 128), nn.GELU(), nn.Linear(128, 8))

    def forward(self, types, path, angles, mask):
        if path.shape != (angles.size(0), 3) or angles.shape != mask.shape or angles.size(1) != 2:
            raise ValueError('line paths require 3 tokens and 2 angle positions')
        if not torch.isfinite(angles[mask]).all() or not mask.any(-1).all():
            raise ValueError('invalid or empty required angle path')
        extended = torch.cat([types, types.new_tensor([25858])])
        indices = torch.where(path >= 0, path, path.new_full(path.shape, types.numel()))
        path_types = extended[indices]
        features = []
        for k in range(2):
            encoded = self.gaussian(angles[:, k], path_types[:, k:k + 2])
            # Mask after the MLP: its learned bias must not contaminate padding.
            features.append(self.position[k](encoded) * mask[:, k:k + 1])
        pooled = sum(features) / mask.sum(-1, keepdim=True).clamp_min(1)
        return self.heads(pooled)


class GalformerTrimer3D(nn.Module):
    def __init__(self, dropout=0.1, *, torsion=False, geometry_mode='finite_internal', angle_value_enabled=False, torsion_revision=None, nonbonded_revision=None, attention_scale_denominator=512):
        super().__init__()
        self.geometry_mode = geometry_mode
        self.torsion_revision = torsion_revision
        if torsion_revision not in (None, 'glt_physical_torsion_r1') or (torsion_revision and (not torsion or geometry_mode != 'finite_incident')):
            raise ValueError('invalid physical torsion model mode')
        self.angle_value = nn.Linear(128,512) if angle_value_enabled else None
        self.endpoint = nn.Linear(101, 512)
        self.distance_basis = TypeGaussian(256, 104)
        self.distance_projection = nn.Linear(256, 512)
        self.triplet = nn.Sequential(nn.Linear(1024, 512), nn.GELU(), nn.Linear(512, 512))
        self.angle_bias = PathAngleBias()
        self.layers = nn.ModuleList([TransformerBlock(dropout, attention_scale_denominator=attention_scale_denominator) for _ in range(6)])
        # S4 torsion head: real Trimer heavy-atom quadruplets feed
        # [cos(phi), cos(2*phi)] through a small MLP, pooled per physical
        # center bond and added to the initial 3D bond token through one
        # scalar gate initialized at zero (strictly zero initial increment).
        self.torsion_mlp = (nn.Sequential(nn.Linear(2, 64), nn.GELU(), nn.Linear(64, 512))
                            if torsion and torsion_revision is None else None)
        self.torsion_gate = nn.Parameter(torch.zeros(1)) if torsion and torsion_revision is None else None
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        if torsion_revision:
            # Construct after the common initialization loop, including Linear
            # constructors, so all public layers retain their RNG stream.
            with torch.random.fork_rng(devices=[]):
                torch.default_generator.manual_seed(314167)
                self.torsion_mlp = nn.Sequential(nn.Linear(2,64),nn.GELU(),nn.Linear(64,512))
                self.torsion_gate = nn.Parameter(torch.zeros(1))
                for layer in self.torsion_mlp:
                    if isinstance(layer,nn.Linear):
                        nn.init.normal_(layer.weight,std=.02);nn.init.zeros_(layer.bias)
        self.nonbonded_revision=nonbonded_revision
        self.nonbonded_input=None
        if nonbonded_revision is not None:
            if nonbonded_revision!='glt_nonbonded_r10_g1' or torsion or geometry_mode!='finite_incident':
                raise ValueError('invalid nonbonded input contract')
            from .glt_r10 import NonbondedInput
            self.nonbonded_input=NonbondedInput()

    def _torsion_increment(self, data, bond_count):
        if self.torsion_revision and getattr(data,'torsion_revision',None) != self.torsion_revision:
            raise ValueError('model/input physical torsion revision mismatch')
        index = data.torsion_bond_index.long()
        valid = data.torsion_mask.bool()
        features = self.torsion_mlp(data.torsion_values.float())
        pooled = features.new_zeros((bond_count, features.size(-1)))
        counts = features.new_zeros((bond_count,))
        pooled.index_add_(0, index[valid], features[valid])
        counts.index_add_(0, index[valid], torch.ones_like(index[valid], dtype=features.dtype))
        return pooled / counts.clamp_min(1.0).unsqueeze(-1)

    def forward(self, data, *, endpoint_increment=None):
        za, zb = element_index(data.bond_z_a), element_index(data.bond_z_b)
        distance = data.bond_distance.float()
        if not torch.isfinite(distance).all() or (distance <= 0).any():
            raise ValueError('invalid physical bond distance')
        a = torch.nn.functional.one_hot(za, 101).float()
        b = torch.nn.functional.one_hot(zb, 101).float()
        endpoints = self.endpoint(a) + self.endpoint(b)
        radial = self.distance_projection(self.distance_basis(distance, torch.stack([za, zb], -1)))
        states = self.triplet(torch.cat([endpoints, radial], -1))
        if self.nonbonded_input is not None:
            states = states + self.nonbonded_input(data,states.size(0))
        if self.torsion_mlp is not None:
            states = states + self.torsion_gate * self._torsion_increment(data, states.size(0))
        types = triplet_type(data.bond_z_a, data.bond_z_b, data.bond_type)
        periodic = self.geometry_mode == 'periodic_template'
        if getattr(data, 'geometry_mode', 'finite_internal') != self.geometry_mode:
            raise ValueError('model/input geometry mode mismatch')
        if periodic:
            index = data.periodic_bond_index.long()
            count = data.periodic_keys.size(0)
            weight = data.periodic_observation_weight
            denominator = states.new_zeros(count).index_add(0,index,weight.to(states.dtype))
            states = states.new_zeros((count,512)).index_add(0,index,states*weight.to(states.dtype).unsqueeze(-1)) / denominator.clamp_min(1e-12).unsqueeze(-1)
            # Masked endpoint types are identical across physical copies.
            chosen = torch.full((count,), -1, dtype=torch.long, device=index.device)
            chosen.scatter_(0,index,torch.arange(index.numel(),device=index.device))
            types = types[chosen]
            path, angles, mask = data.periodic_line_path, data.periodic_line_angle, data.periodic_line_mask
            groups, source, target = data.periodic_line_group, data.periodic_line_source, data.periodic_line_target
            is_self = data.periodic_line_self
        else:
            path, angles, mask = data.line_path, data.line_angle, data.line_mask
            groups, source, target = data.line_path_group, data.line_source, data.line_target
            is_self = data.line_is_self
        path_bias = self.angle_bias(types, path, angles, mask)
        bias = mean_pool(path_bias, groups, source.numel())
        if self.angle_value is not None:
            real = (mask.sum(-1)==1) & ~is_self
            pair_types = types[path[real,:2]]
            features = self.angle_bias.gaussian(angles[real,0],pair_types)
            increment = self.angle_value(features)
            # Real directed one-hop angles increment the receiving token even
            # when attention has a single neighbor and its softmax is constant.
            states = states + mean_pool(increment,target[groups[real]],states.size(0))
        if endpoint_increment is not None:
            if endpoint_increment.shape!=states.shape:raise ValueError("endpoint increment shape mismatch")
            states=states+endpoint_increment
        for layer in self.layers:
            states = layer(states, source, target, bias)
        token_states = states
        if periodic:
            states = states[data.periodic_bond_index]
        center = data.bond_center.bool()
        batch = data.bond_batch[center]
        count = data.geometry_valid.numel()
        numbers = torch.bincount(batch, minlength=count)
        valid = data.geometry_valid.bool() & (numbers > 0)
        graph = mean_pool(states[center], batch, count)
        graph = torch.where(valid.unsqueeze(-1), graph, torch.zeros_like(graph))
        return dict(bond_states=states, token_states=token_states, center_bond_states=states[center],
                    graph_3d=graph, geometry_valid=valid, angle_bias=bias)


class DualGLTModel(nn.Module):
    architecture_name = 'O8-BondPath-GalformerTrimer-Hop2'

    def __init__(self, fusion_mode='concat', dropout=0.1, *, torsion=False, readout='internal',
                 fusion_revision=None, use_md200=False, geometry_revision=None,
                 geometry_mode="finite_internal", angle_value_enabled=False, fusion_gate="none",
                 endpoint_revision=None,endpoint_module=None,endpoint_input_mode=None,endpoint_init_seed=None,
                 fusion_attention_mode="global",incident_bias=0.0,torsion_revision=None,nonbonded_revision=None):
        super().__init__()
        if fusion_mode not in {'concat', 'kfuse', 'token', 'tk'}:
            raise ValueError('fusion_mode must be concat or kfuse')
        from src.dataset.glt_periodic_geometry import geometry_options
        options=geometry_options(geometry_revision,geometry_mode,angle_value_enabled,readout)
        for key,value in options.items():setattr(self,key,value)
        self.fusion_revision = fusion_revision
        self.use_md200 = bool(use_md200)
        if fusion_revision not in (None, 'glt_fusion_r1'):
            raise ValueError('unknown fusion revision')
        if fusion_revision is None and (use_md200 or fusion_mode == 'tk'):
            raise ValueError('MD/TK requires glt_fusion_r1')
        if fusion_revision:
            self.architecture_name = 'O8-BondPath-GalformerTrimer-Hop2-MD200-r1'
        self.fusion_mode = fusion_mode
        self.fusion_gate = fusion_gate
        if fusion_gate != "none" and (fusion_mode != "token" or geometry_mode != "finite_incident"):raise ValueError("gate requires finite token model")
        if readout not in {'internal', 'incident'}:raise ValueError('invalid readout')
        self.readout = readout
        self.o8 = BondPathO8(dropout)
        self.torsion_revision = torsion_revision
        self.glt = GalformerTrimer3D(dropout, torsion=torsion, geometry_mode=geometry_mode, angle_value_enabled=angle_value_enabled, torsion_revision=torsion_revision,nonbonded_revision=nonbonded_revision)
        if fusion_mode in {'concat', 'token'} or fusion_revision:
            self.norm2, self.norm3 = nn.LayerNorm(512), nn.LayerNorm(512)
        if fusion_mode in {'kfuse', 'tk'}:
            self.kfuse = OriginalMIPSAttentiveFusion(
                knodes=('md', 'glt3d') if fusion_revision else ('glt3d',),
                knowledge_dims={'md': 200, 'glt3d': 512})
        if fusion_mode in {'token', 'tk'}:
            from .glt_dual_opt import BondTokenFusion
            self.token_fusion = BondTokenFusion(fusion_gate)
        self.predictor = nn.Sequential(nn.Linear(1024 if fusion_revision or fusion_mode in {'concat', 'token'} else 512, 512),
                                       nn.GELU(), nn.Dropout(dropout), nn.Linear(512, 1))

        self.endpoint_revision=endpoint_revision
        self.endpoint_module=endpoint_module
        self.endpoint_input_mode=endpoint_input_mode
        self.endpoint_init_seed=endpoint_init_seed
        self.fusion_attention_mode=fusion_attention_mode
        self.incident_bias=incident_bias
        if endpoint_revision:
            locality_options({k:getattr(self,k) for k in ('endpoint_revision','endpoint_module','endpoint_input_mode','endpoint_init_seed','fusion_attention_mode','incident_bias')})
            if fusion_mode!='token' or fusion_gate!='none' or use_md200 or geometry_mode!='finite_incident':
                raise ValueError('S3 requires finite token model without gate/MD')
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(endpoint_init_seed)
                self.endpoint_mlp=nn.Sequential(nn.Linear(512,512,bias=False),nn.GELU(),nn.Linear(512,512,bias=False))
                nn.init.zeros_(self.endpoint_mlp[-1].weight)
            self.token_fusion.set_attention_mode(fusion_attention_mode,incident_bias)
        elif any(v is not None for v in (endpoint_module,endpoint_input_mode,endpoint_init_seed)) or fusion_attention_mode!='global' or incident_bias!=0.0:
            raise ValueError('new architecture requires explicit endpoint revision')

    def endpoint_increment(self, data, initial):
        a,b=data.bond_canonical_a,data.bond_canonical_b
        if a.dtype!=torch.long or b.dtype!=torch.long or a.shape!=data.bond_distance.shape or b.shape!=a.shape:
            raise ValueError('endpoint physical mapping schema mismatch')
        if a.numel() and (a.min()<0 or b.min()<0 or a.max()>=initial.size(0) or b.max()>=initial.size(0)):
            raise ValueError('endpoint physical mapping out of range')
        if a.numel() and (not torch.equal(data.canonical_graph_index[a],data.bond_batch) or not torch.equal(data.canonical_graph_index[b],data.bond_batch)):
            raise ValueError('endpoint physical mapping crosses graphs')
        inputs=(initial[a]+initial[b])*.5 if self.endpoint_input_mode=='masked_mean' else initial.new_zeros((a.numel(),512))
        return self.endpoint_mlp(inputs)

    def encode(self, data, *, atom_mask=None):
        if self.endpoint_revision:
            atoms,bias,initial=self.o8(data,atom_mask,return_initial=True)
            result=self.glt(data,endpoint_increment=self.endpoint_increment(data,initial))
        else:
            atoms,bias=self.o8(data,atom_mask)
            result=self.glt(data)
        if self.readout == 'incident':
            from .glt_dual_opt import incident_pool
            pooled, available = incident_pool(result['bond_states'], data.incidence_index,
                data.canonical_graph_index, data.geometry_valid.numel())
            result['geometry_valid'] = data.geometry_valid.bool() & available
            result['graph_3d'] = torch.where(result['geometry_valid'].unsqueeze(-1), pooled, torch.zeros_like(pooled))
        result['bond_graph_index'] = data.periodic_bond_batch if self.geometry_mode == 'periodic_template' else data.bond_batch
        if self.fusion_revision:
            result['incidence_index'] = data.incidence_index
            if not hasattr(data, 'md200') or not hasattr(data, 'md200_valid'):
                raise ValueError('new architecture requires explicit MD values and validity')
            result['md200'], result['md200_valid'] = data.md200, data.md200_valid
        result.update(atom_states=atoms, bond_path_attention_bias=bias,
                      canonical_graph_index=data.canonical_graph_index,
                      graph_2d=mean_pool(atoms, data.canonical_graph_index, data.graph_available.numel()))
        return result

    def fuse(self, encoded):
        valid = encoded['geometry_valid']
        if self.fusion_revision:
            atoms = encoded['atom_states']
            index = encoded['canonical_graph_index']
            if 'fused_atom_states' not in encoded:
                fused = atoms
                if self.fusion_mode in {'token', 'tk'}:
                    collect=getattr(self,'collect_fusion_attention',True)
                    value = self.token_fusion(
                        atoms, encoded['token_states'], index,
                        encoded['bond_graph_index'], valid, return_attention=collect, incidence=encoded.get('incidence_index'))
                    if collect:
                        fused,attention=value
                        encoded['local_attention']=attention
                    else:fused=value
                if self.fusion_mode in {'kfuse', 'tk'}:
                    md_valid = encoded['md200_valid'].bool() & self.use_md200
                    fused = self.kfuse(fused, {'md': encoded['md200'], 'glt3d': encoded['graph_3d']},
                                      index, availability={'md': md_valid, 'glt3d': valid})
                encoded['fused_atom_states'] = fused
            pooled = mean_pool(encoded['fused_atom_states'], index, valid.numel())
            geometry = self.norm3(encoded['graph_3d'])
            geometry = torch.where(valid.unsqueeze(-1), geometry, torch.zeros_like(geometry))
            return torch.cat([self.norm2(pooled), geometry], -1)
        if self.fusion_mode in {'concat', 'token'}:
            if self.fusion_mode == 'token':
                if 'fused_atom_states' not in encoded:
                    encoded['fused_atom_states'] = self.token_fusion(encoded['atom_states'], encoded['token_states'], encoded['canonical_graph_index'], encoded['bond_graph_index'], valid)
                graph2 = mean_pool(encoded['fused_atom_states'], encoded['canonical_graph_index'], valid.numel())
            else:
                graph2 = encoded['graph_2d']
            geometry = self.norm3(encoded['graph_3d'])
            geometry = torch.where(valid.unsqueeze(-1), geometry, torch.zeros_like(geometry))
            features = torch.cat([self.norm2(graph2), geometry], -1)
        else:
            atoms = encoded['atom_states']
            index = encoded['canonical_graph_index']
            fused = self.kfuse(atoms, {'glt3d': encoded['graph_3d']}, index)
            # Mask the entire projected update, including v_proj.bias.
            fused = torch.where(valid[index].unsqueeze(-1), fused, atoms)
            features = mean_pool(fused, index, encoded['graph_2d'].size(0))
        return features

    def forward(self, data, *, atom_mask=None):
        return self.predictor(self.fuse(self.encode(data, atom_mask=atom_mask)))


def build_dual_glt_model(fusion_mode='concat', *, dropout=0.1, torsion=False, readout='internal',
                         fusion_revision=None, use_md200=False, geometry_revision=None,
                 geometry_mode="finite_internal", angle_value_enabled=False, fusion_gate="none", **locality):
    return DualGLTModel(fusion_mode=fusion_mode, dropout=dropout, torsion=torsion, readout=readout,
                        fusion_revision=fusion_revision, use_md200=use_md200, geometry_revision=geometry_revision,
                        geometry_mode=geometry_mode, angle_value_enabled=angle_value_enabled, fusion_gate=fusion_gate, **locality)


class Pure2DGLTModel(nn.Module):
    """Original UA_2D prediction path, constructed without a 3D encoder.

    Returns standardized predictions; the caller retains the original train-only
    scaler and inverse transform, just as with the U1 Pure2D wrapper.
    """
    def __init__(self, dropout=0.1, *, attention_scale_denominator=512):
        super().__init__()
        self.o8 = BondPathO8(dropout, attention_scale_denominator=attention_scale_denominator)
        self.attention_scale_denominator = attention_scale_denominator
        self.attention_scale = attention_scale_denominator ** -0.5
        self.norm2 = nn.LayerNorm(512)
        self.head = nn.Sequential(nn.Linear(512,512), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(512,1))

    def features(self, data):
        atoms, _ = self.o8(data)
        return self.norm2(mean_pool(atoms, data.canonical_graph_index, data.graph_available.numel()))

    def forward(self, data):
        return self.head(self.features(data))

    def forward_topology(self, topology, smiles):
        """One canonical 2D Topology in; no geometry carrier required."""
        from src.dataset.glt_dual_pretrain import validate_canonical_topology
        from src.dataset.glt_dual import build_glt_2d_sample, glt_2d_collate
        validate_canonical_topology(topology)
        data = glt_2d_collate([build_glt_2d_sample(topology, smiles)])
        return self(data.to(next(self.parameters()).device))


def strict_module_state(module, state):
    current = module.state_dict()
    if not isinstance(state, dict) or set(state) != set(current):
        raise ValueError('module tensor names mismatch')
    for name, value in state.items():
        if (not torch.is_tensor(value) or value.shape != current[name].shape
                or value.dtype != current[name].dtype or not torch.isfinite(value).all()):
            raise ValueError('module tensor shape/dtype/value mismatch: ' + name)
    module.load_state_dict(state, strict=True)


def _u1_encoder_shapes(model):
    """Explicit full F_CE/U1 token encoder schema; no Galformer construction."""
    shapes = {k: tuple(v.shape) for k,v in model.state_dict().items() if not k.startswith('head.')}
    def linear(name, inputs, outputs):
        shapes[name+'.weight'] = (outputs, inputs)
        shapes[name+'.bias'] = (outputs,)
    def norm(name):
        shapes[name+'.weight'] = shapes[name+'.bias'] = (512,)
    def gaussian(name, size, types):
        shapes[name+'.means'] = shapes[name+'.stds'] = (size,)
        shapes[name+'.mul.weight'] = shapes[name+'.bias.weight'] = (types,1)
    norm('norm3')
    linear('glt.angle_value',128,512)
    linear('glt.endpoint',101,512)
    gaussian('glt.distance_basis',256,104)
    linear('glt.distance_projection',256,512)
    linear('glt.triplet.0',1024,512);linear('glt.triplet.2',512,512)
    gaussian('glt.angle_bias.gaussian',128,25859)
    for i in range(2):
        linear(f'glt.angle_bias.position.{i}.0',128,128)
        linear(f'glt.angle_bias.position.{i}.2',128,128)
    linear('glt.angle_bias.heads.0',128,128);linear('glt.angle_bias.heads.2',128,8)
    for i in range(6):
        prefix=f'glt.layers.{i}'
        norm(prefix+'.norm1');norm(prefix+'.norm2')
        linear(prefix+'.attention.qkv',512,1536)
        linear(prefix+'.output',512,512)
        linear(prefix+'.ffn.0',512,2048);linear(prefix+'.ffn.3',2048,512)
    for name in ('q','k','v'):linear('token_fusion.'+name,512,128)
    linear('token_fusion.output',128,512)
    shapes['token_fusion.scale'] = ()
    return shapes


def restore_pure_2d(package, head_state, expected_step=5000, *, dropout=0.1):
    """Extract only the explicitly supported F_CE/U1 full encoder schema.

    Other architectures require their own explicit extraction schema; they are
    rejected, including otherwise valid S3/S4 branches, rather than filtered.
    """
    from .glt_dual_pretrain import deployment_model_options
    options = deployment_model_options(package, expected_step)
    expected = dict(fusion_mode='token', fusion_revision='glt_fusion_r1',
        geometry_revision='glt_periodic_seam_r1',geometry_mode='finite_incident',
        angle_value_enabled=True,readout='incident',use_md200=False,
        fusion_gate='none',torsion=False,torsion_revision=None,
        fusion_attention_mode='global',incident_bias=0.0)
    if options != expected or package.get('endpoint_revision') is not None:
        raise ValueError('unsupported full package for pure 2D extraction')
    model = Pure2DGLTModel(dropout)
    shapes = _u1_encoder_shapes(model)
    state = package['state_dict']
    if not isinstance(state, dict) or set(state) != set(shapes):
        raise ValueError('full U1 encoder tensor schema mismatch')
    for name,value in state.items():
        if (not torch.is_tensor(value) or tuple(value.shape) != shapes[name]
                or value.dtype != torch.float32 or not torch.isfinite(value).all()):
            raise ValueError('full U1 encoder tensor shape/dtype/value mismatch: ' + name)
    # Exact submodule names come from the independently constructed 2D modules.
    strict_module_state(model.o8, {name: state['o8.'+name] for name in model.o8.state_dict()})
    strict_module_state(model.norm2, {name: state['norm2.'+name] for name in model.norm2.state_dict()})
    strict_module_state(model.head, head_state)
    return model
