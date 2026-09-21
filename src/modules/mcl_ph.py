"""Three independent multiscale distance experts with a PH-driven scale router.

Plan ``MCL-PH-20260921-01/r1``, contract sections 4, 5.2 and 8.

The module replaces the GLT 3D channel by:

* ``SchNetExpert`` -- one independent SchNet-inspired encoder per cutoff
  ``2/3/4 A``; three interaction blocks, hidden 128, no parameter sharing, no
  dropout, no coordinate update and no vector state;
* ``TopologyRouter`` -- one linear map from the ``[31, 5]`` topology trajectory
  to three scale weights, dense softmax for the warm-up updates and a Top-2
  softmax afterwards;
* ``FusionGate`` / ``FusionCrossAttention`` / ``FusionConcat`` -- the three
  declared ways of combining the 2D representation with the already
  scale-selected 3D readout.  A fusion module never sees the individual expert
  outputs, the raw PH curve or ``alpha``, so it cannot select a scale again.
"""

from __future__ import annotations

import math

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from .glt_dual import BondPathO8, mean_pool
from ..dataset.mcl_ph_view import (AROMATIC_VOCABULARY, BOND_CATEGORY_COUNT,
                                   CHARGE_VOCABULARY, ELEMENT_VOCABULARY,
                                   RBF_CENTERS, RBF_WIDTH, DESCRIPTOR_COLUMNS,
                                   ROUTER_RADII)

EXPERT_HIDDEN = 128
EXPERT_BLOCKS = 3
EXPERT_EDGE_FEATURES = 64 + 8
FUSION_SCALE = 0.1
ROUTER_DESCRIPTOR_DIM = len(ROUTER_RADII) * DESCRIPTOR_COLUMNS
XATTN_HEADS = 4
XATTN_DIM = 32
XATTN_QUERY_CHUNK = 512


def xavier_(module):
    """Declared initialization for every Linear in the new branch."""
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def _normal_embeddings_(module, std=0.02):
    if isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=std)


class GlobalSum(torch.autograd.Function):
    """Autograd-aware global sum, running one collective on every rank.

    The forward really is ``sum over ranks``.  Its derivative with respect to
    each rank's own contribution is one, so the balance term keeps a correct
    gradient instead of a detached global mean.
    """

    @staticmethod
    def forward(ctx, value):
        if not dist.is_available() or not dist.is_initialized() or dist.get_world_size() == 1:
            return value.clone()
        total = value.clone()
        dist.all_reduce(total)
        return total

    @staticmethod
    def backward(ctx, gradient):
        return gradient


def global_sum(value):
    return GlobalSum.apply(value)


def rbf_kernel(distance):
    """Fixed 64-bin Gaussian basis, centres ``4k/63`` and width ``4/63``."""
    value = torch.as_tensor(distance).float().reshape(-1, 1)
    centres = torch.as_tensor(RBF_CENTERS, dtype=torch.float32,
                              device=value.device).reshape(1, -1)
    return torch.exp(-0.5 * ((value - centres) / float(RBF_WIDTH)) ** 2)


def cutoff_envelope(distance, cutoff):
    """``0.5 * (1 + cos(pi d / c))`` inside the cutoff and exactly zero outside."""
    value = torch.as_tensor(distance).float()
    inside = value <= float(cutoff)
    envelope = 0.5 * (1.0 + torch.cos(math.pi * value / float(cutoff)))
    return torch.where(inside, envelope, torch.zeros_like(envelope))


class SchNetInteraction(nn.Module):
    """One declared interaction block: ``V``, ``F`` and a bias-free ``U``."""

    def __init__(self, hidden=EXPERT_HIDDEN, edge_features=EXPERT_EDGE_FEATURES):
        super().__init__()
        self.value = nn.Linear(hidden, hidden, bias=False)
        self.filter = nn.Sequential(nn.Linear(edge_features, hidden), nn.SiLU(),
                                    nn.Linear(hidden, hidden))
        self.update = nn.Sequential(nn.Linear(hidden, hidden, bias=False), nn.SiLU(),
                                    nn.Linear(hidden, hidden, bias=False))
        self.apply(xavier_)
        # ``F`` keeps its (zero) biases; a bias-free ``U`` is what makes an
        # empty neighbourhood produce a strictly zero increment.
        with torch.no_grad():
            for layer in self.filter:
                if isinstance(layer, nn.Linear):
                    layer.bias.zero_()

    def forward(self, states, source, target, edge_features, envelope, count):
        message = self.value(states[source]) * self.filter(edge_features) * envelope.unsqueeze(-1)
        aggregated = states.new_zeros((count, states.size(-1)), dtype=torch.float32)
        aggregated.index_add_(0, target, message.float())
        return states + self.update(aggregated.to(states.dtype))


class SchNetExpert(nn.Module):
    """One independent multiscale distance expert (section 4)."""

    def __init__(self, cutoff, hidden=EXPERT_HIDDEN, blocks=EXPERT_BLOCKS,
                 edge_features=EXPERT_EDGE_FEATURES):
        super().__init__()
        self.cutoff = float(cutoff)
        self.element = nn.Embedding(ELEMENT_VOCABULARY, hidden)
        self.charge = nn.Embedding(CHARGE_VOCABULARY, hidden)
        self.aromatic = nn.Embedding(AROMATIC_VOCABULARY, hidden)
        self.bond_type = nn.Embedding(BOND_CATEGORY_COUNT, edge_features - 64)
        self.interactions = nn.ModuleList(
            [SchNetInteraction(hidden, edge_features) for _ in range(int(blocks))])
        self.apply(_normal_embeddings_)

    def initial_state(self, element, charge, aromatic, count):
        return (self.element(element) + self.charge(charge) + self.aromatic(aromatic)).to(
            dtype=self.element.weight.dtype)

    def edge_features(self, distance, edge_type):
        return torch.cat([rbf_kernel(distance), self.bond_type(edge_type).float()], dim=-1)

    def forward(self, element, charge, aromatic, source, target, distance, edge_type, count):
        states = self.initial_state(element, charge, aromatic, count)
        if int(source.numel()) == 0:
            # Every ``U`` is bias-free, so ``U(0) = 0`` exactly and an isolated
            # atom keeps its own representation with a zero increment.
            return states
        features = self.edge_features(distance, edge_type)
        envelope = cutoff_envelope(distance, self.cutoff)
        for interaction in self.interactions:
            states = interaction(states, source, target, features, envelope, count)
        return states


class TopologyRouter(nn.Module):
    """``Flatten155 -> Linear128 -> GELU -> Linear3`` with dense/Top-2 routing."""

    DENSE = 'dense'
    TOP2 = 'top2'

    def __init__(self, dense_updates=500, top_k=2):
        super().__init__()
        self.dense_updates = int(dense_updates)
        self.top_k = int(top_k)
        self.net = nn.Sequential(nn.Linear(ROUTER_DESCRIPTOR_DIM, 128), nn.GELU(),
                                 nn.Linear(128, 3))
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, std=0.02)
                nn.init.zeros_(layer.bias)
        self.mode = self.DENSE
        self.step = 0

    def configure(self, mode, step=None):
        if mode not in (self.DENSE, self.TOP2):
            raise ValueError('router mode must be dense or top2')
        self.mode = str(mode)
        if step is not None:
            self.step = int(step)

    def routing_mode_for_step(self, update):
        """Dense for updates ``1..dense_updates``, Top-2 from the next update on."""
        return self.DENSE if int(update) <= self.dense_updates else self.TOP2

    def forward(self, trajectory):
        logits = self.net(trajectory)
        dense = torch.softmax(logits, dim=-1)
        if self.mode == self.DENSE:
            weights, hard = dense, None
            order = None
        else:
            # Stable descending order keeps tied logits in ascending cutoff
            # order, which is the declared tie rule.
            order = torch.argsort(logits, dim=-1, descending=True, stable=True)
            keep = order[:, :self.top_k]
            masked = torch.full_like(logits, float('-inf'))
            masked.scatter_(1, keep, logits.gather(1, keep))
            weights = torch.softmax(masked, dim=-1)
            hard = keep
        return {'logits': logits, 'dense': dense, 'weights': weights, 'top_k': hard,
                'order': order}


def balance_term(dense_probabilities, valid):
    """``3 * sum_k (mean_b over V p_bk)^2 - 1`` on the globally valid graph set.

    ``dense_probabilities`` are the pre-Top-2 soft probabilities of this rank's
    graphs.  With no valid graph anywhere the term is a finite, differentiable
    zero instead of an empty-mean NaN.

    This returns the *mathematical* term of section 5.2 only.  The distributed
    backward scale is applied by the objective, which keeps the mathematics, the
    scaling and the logged statistics separate.
    """
    if dense_probabilities.ndim != 2 or dense_probabilities.size(1) != 3:
        raise ValueError('router probabilities must be [B,3]')
    flag = torch.as_tensor(valid).bool().reshape(-1)
    if int(flag.numel()) != int(dense_probabilities.size(0)):
        raise ValueError('router validity flags must match the graph count')
    local_sum = (dense_probabilities * flag.unsqueeze(-1).to(dense_probabilities.dtype)).sum(0)
    total = global_sum(local_sum)
    count = global_sum(flag.to(dense_probabilities.dtype).sum().reshape(1))
    if float(count.detach()) <= 0:
        return dense_probabilities.sum() * 0.0, True
    mean = total / count.clamp_min(1.0)
    return 3.0 * (mean ** 2).sum() - 1.0, False


class MCLPHBranch(nn.Module):
    """Three experts, one topology router and the declared scale mixture."""

    def __init__(self, cutoffs=(2.0, 3.0, 4.0), router_dense_updates=500,
                 router_top_k=2):
        super().__init__()
        self.cutoffs = tuple(float(value) for value in cutoffs)
        self.experts = nn.ModuleList([SchNetExpert(cutoff) for cutoff in self.cutoffs])
        self.router = TopologyRouter(router_dense_updates, router_top_k)
        self.collect_diagnostics = False
        self.last_diagnostics = None

    def routing_mode_for_step(self, update):
        return self.router.routing_mode_for_step(update)

    def forward(self, data):
        count = int(data.mcl_z.numel())
        element, charge, aromatic = data.mcl_z.long(), data.mcl_charge.long(), data.mcl_aromatic.long()
        edge_index = data.mcl_edge_index.long()
        scale = data.mcl_edge_scale.long()
        distance = data.mcl_edge_distance.float()
        edge_type = data.mcl_edge_type.long()
        states = []
        for slot, expert in enumerate(self.experts):
            selected = torch.nonzero(scale == slot, as_tuple=False).flatten()
            source = edge_index[0, selected]
            target = edge_index[1, selected]
            states.append(expert(element, charge, aromatic, source, target,
                                 distance[selected], edge_type[selected], count))
        trajectory = data.mcl_trajectory.float()
        if trajectory.ndim != 3 or tuple(trajectory.shape[1:]) != (
                len(ROUTER_RADII), DESCRIPTOR_COLUMNS):
            raise ValueError(f'topology trajectory must be [G,{len(ROUTER_RADII)},'
                             f'{DESCRIPTOR_COLUMNS}]')
        routed = self.router(trajectory.reshape(trajectory.size(0), -1))
        alpha = routed['weights']
        atom_graph = data.mcl_atom_batch.long()
        if int(atom_graph.numel()) != int(count):
            raise ValueError('one graph index per Trimer heavy atom is required')
        if int(alpha.size(0)) != int(data.graph_available.numel()):
            raise ValueError('one routing weight per graph is required')
        # Every graph shares one alpha; the mixture is dense over experts.
        mixed = torch.zeros((count, states[0].size(-1)), dtype=states[0].dtype,
                            device=states[0].device)
        for slot, value in enumerate(states):
            mixed = mixed + alpha[:, slot][atom_graph].unsqueeze(-1).to(mixed.dtype) * value
        valid = data.mcl_readout_valid.bool()
        balance, balance_empty = balance_term(routed['dense'], valid)
        result = {'expert_states': states, 'mixed': mixed, 'alpha': alpha,
                  'router_logits': routed['logits'], 'router_dense': routed['dense'],
                  'router_top_k': routed['top_k'], 'balance': balance,
                  'balance_empty_valid': balance_empty,
                  'readout_valid': valid}
        if self.collect_diagnostics:
            result['diagnostics'] = self._diagnostics(routed, valid, count)
            self.last_diagnostics = result['diagnostics']
        return result

    def _diagnostics(self, routed, valid, count):
        logits, dense, hard = routed['logits'], routed['dense'], routed['top_k']
        entropy = -(dense.clamp_min(1e-12).log() * dense).sum(-1)
        flat = logits.detach()
        probabilities = dense.detach()
        entropy_flat = entropy.detach()
        tie = ((flat[:, 0] == flat[:, 1]) | (flat[:, 0] == flat[:, 2])
               | (flat[:, 1] == flat[:, 2]))
        return {
            # Logits are reported as per-expert statistics: the full per-graph
            # matrix is the same information at a size that grows with the batch.
            'router_logits_mean': [float(value) for value in flat.mean(0)],
            'router_logits_std': [float(value) for value in flat.std(0)],
            'router_logits_min': [float(value) for value in flat.min(0).values],
            'router_logits_max': [float(value) for value in flat.max(0).values],
            'router_probability_mean': [float(value) for value in probabilities.mean(0)],
            'router_probability_std': [float(value) for value in probabilities.std(0)],
            'router_entropy_mean': float(entropy_flat.mean()) if entropy.numel() else 0.0,
            'router_entropy_std': float(entropy_flat.std()) if entropy.numel() else 0.0,
            'router_entropy_min': float(entropy_flat.min()) if entropy.numel() else 0.0,
            'router_entropy_max': float(entropy_flat.max()) if entropy.numel() else 0.0,
            'router_hard_selection': ([int(value) for value in torch.bincount(
                hard.detach().reshape(-1), minlength=3)] if hard is not None else None),
            # Dense routing has no hard choice to report: every graph uses the
            # soft mixture, so the field must stay null instead of inventing an
            # argmax selection that the forward never made.
            'router_hard_selection_note': ('counts of the Top-2 experts per graph'
                                           if hard is not None else
                                           'not_applicable: dense routing uses the soft '
                                           'mixture for every graph'),
            'router_tie_count': int(tie.sum().item()),
            'router_tie_rate': float(tie.float().mean().item()) if tie.numel() else 0.0,
            'router_mode': self.router.mode,
            'readout_valid_graphs': int(valid.sum().item()),
            'graphs': int(count and valid.numel()),
        }


class FusionGate(nn.Module):
    """``h + 0.1 * Wo(g * v)`` with the declared GMU-style channel gate."""

    def __init__(self, hidden=512, token=128):
        super().__init__()
        self.norm2 = nn.LayerNorm(hidden)
        self.norm3 = nn.LayerNorm(token)
        self.collect_diagnostics = False
        self.last_diagnostics = None
        self.to_reduced = nn.Linear(hidden, token, bias=False)
        self.to_value = nn.Linear(token, token, bias=False)
        self.gate = nn.Linear(4 * token, token)
        self.out = nn.Linear(token, hidden, bias=False)
        nn.init.xavier_uniform_(self.to_reduced.weight)
        nn.init.xavier_uniform_(self.to_value.weight)
        nn.init.xavier_uniform_(self.out.weight)
        nn.init.normal_(self.gate.weight, std=0.001)
        nn.init.zeros_(self.gate.bias)

    def forward(self, hidden, tokens):
        u = self.to_reduced(self.norm2(hidden))
        v = self.to_value(self.norm3(tokens))
        gate = torch.sigmoid(self.gate(torch.cat([u, v, u * v, (u - v).abs()], dim=-1)))
        if self.collect_diagnostics:
            self.last_diagnostics = {
                'gate_mean': [float(value) for value in gate.detach().mean(0)],
                'gate_std': [float(value) for value in gate.detach().std(0)],
                'gate_min': float(gate.detach().min()), 'gate_max': float(gate.detach().max()),
            }
        return hidden + FUSION_SCALE * self.out(gate * v)


class FusionCrossAttention(nn.Module):
    """One 4x32 cross-attention layer over the graph's own centre atoms."""

    def __init__(self, hidden=512, token=128, heads=XATTN_HEADS, dim=XATTN_DIM,
                 chunk=XATTN_QUERY_CHUNK):
        super().__init__()
        self.heads, self.dim, self.chunk = int(heads), int(dim), int(chunk)
        inner = self.heads * self.dim
        self.norm2 = nn.LayerNorm(hidden)
        self.norm3 = nn.LayerNorm(token)
        self.query = nn.Linear(hidden, inner, bias=False)
        self.key = nn.Linear(token, inner, bias=False)
        self.value = nn.Linear(token, inner, bias=False)
        self.out = nn.Linear(inner, hidden, bias=False)
        for layer in (self.query, self.key, self.value, self.out):
            nn.init.xavier_uniform_(layer.weight)
        self.scale = float(self.dim) ** -0.5

    def forward(self, hidden, tokens, graph):
        count = int(hidden.size(0))
        query = self.query(self.norm2(hidden)).reshape(count, self.heads, self.dim)
        key = self.key(self.norm3(tokens)).reshape(count, self.heads, self.dim)
        value = self.value(self.norm3(tokens)).reshape(count, self.heads, self.dim)
        # Padding/graph mask only: a key is visible to a query of its own graph.
        same = graph.reshape(-1, 1) == graph.reshape(1, -1)
        output = hidden.new_zeros((count, self.heads, self.dim), dtype=torch.float32)
        for start in range(0, count, self.chunk):
            stop = min(start + self.chunk, count)
            score = torch.einsum('ihd,jhd->hij', query[start:stop].float(), key.float())
            score = score * self.scale
            # The mask is applied to the whole key set, so each chunk performs
            # an exact softmax over every key; nothing is renormalised locally.
            score = score.masked_fill(~same[start:stop].unsqueeze(0), float('-inf'))
            weight = torch.softmax(score, dim=-1)
            output[start:stop] = torch.einsum('hij,jhd->ihd', weight, value.float())
        joined = output.to(hidden.dtype).reshape(count, -1)
        return hidden + FUSION_SCALE * self.out(joined)


class FusionConcat(nn.Module):
    """Atom-level concatenation control (never a graph-vector broadcast)."""

    def __init__(self, hidden=512, token=128):
        super().__init__()
        self.norm2 = nn.LayerNorm(hidden)
        self.norm3 = nn.LayerNorm(token)
        self.expand = nn.Linear(token, hidden)
        self.project = nn.Linear(2 * hidden, hidden)
        for layer in (self.expand, self.project):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, hidden, tokens):
        return self.project(torch.cat([self.norm2(hidden), self.expand(self.norm3(tokens))], -1))


FUSION_MODES = ('none', 'cat', 'gate', 'xattn')


def build_fusion(mode):
    mode = str(mode).lower()
    if mode in ('none', 'o8_only'):
        return None
    if mode == 'gate':
        return FusionGate()
    if mode == 'xattn':
        return FusionCrossAttention()
    if mode == 'cat':
        return FusionConcat()
    raise ValueError('unsupported MCL-PH fusion mode: ' + str(mode))


class MCLPHEncoder(nn.Module):
    """2D O8 backbone plus the multiscale 3D branch and one declared fusion."""

    architecture_name = 'MCL-PH-O8-MultiscaleDistanceRouter'

    def __init__(self, fusion_mode='gate', dropout=0.1, cutoffs=(2.0, 3.0, 4.0),
                 router_dense_updates=500):
        super().__init__()
        if str(fusion_mode).lower() not in FUSION_MODES:
            raise ValueError('unsupported MCL-PH fusion mode')
        self.fusion_mode = str(fusion_mode).lower()
        self.o8 = BondPathO8(dropout)
        self.branch = MCLPHBranch(cutoffs, router_dense_updates)
        self.fusion = build_fusion(self.fusion_mode)
        self.dropout = float(dropout)
        self.collect_diagnostics = False

    def _set_fusion_diagnostics(self, enabled):
        if self.fusion is not None and hasattr(self.fusion, 'collect_diagnostics'):
            self.fusion.collect_diagnostics = bool(enabled)
        # The branch owns the router observations (logits, soft probabilities,
        # entropy and routing mode).  Without this the switch reached the fusion
        # only, so ``diagnostics.router`` stayed empty in every recorded run.
        if hasattr(self.branch, 'collect_diagnostics'):
            self.branch.collect_diagnostics = bool(enabled)

    def encode(self, data, *, atom_mask=None):
        self._set_fusion_diagnostics(self.collect_diagnostics)
        collect = self.collect_diagnostics
        if self.fusion is not None:
            self.fusion.collect_diagnostics = collect
        atoms, bias = self.o8(data, atom_mask=atom_mask)
        branch = self.branch(data)
        graph = data.canonical_graph_index.long()
        valid = data.mcl_readout_valid.bool()
        index = data.mcl_central_index.long()
        tokens = torch.zeros((atoms.size(0), branch['mixed'].size(-1)),
                             dtype=branch['mixed'].dtype, device=atoms.device)
        present = index >= 0
        if bool(present.any()):
            tokens[present] = branch['mixed'][index[present]]
        result = dict(branch)
        if self.fusion is not None and getattr(self.fusion, 'last_diagnostics', None) is not None:
            result['gate_diagnostics'] = self.fusion.last_diagnostics
        result.update(atom_states=atoms, bond_path_attention_bias=bias,
                      canonical_graph_index=graph, tokens=tokens,
                      graph_2d=mean_pool(atoms, graph, data.graph_available.numel()),
                      readout_valid=valid)
        return result

    def fuse(self, encoded):
        """Return the fused atom states; invalid graphs return ``H2`` unchanged."""
        atoms = encoded['atom_states']
        if self.fusion is None:
            return atoms
        node_valid = encoded['readout_valid'][encoded['canonical_graph_index']]
        if not bool(node_valid.any()):
            return atoms
        if not bool(node_valid.all()):
            output = atoms.clone()
            selected = torch.nonzero(node_valid, as_tuple=False).flatten()
            gathered_graph = torch.unique(encoded['canonical_graph_index'][selected],
                                          sorted=True)
            remap = torch.full((int(encoded['readout_valid'].numel()),), -1,
                               dtype=torch.long, device=atoms.device)
            remap[gathered_graph] = torch.arange(gathered_graph.numel(), device=atoms.device)
            output[selected] = self._fuse_selected(
                atoms[selected], encoded['tokens'][selected],
                remap[encoded['canonical_graph_index'][selected]])
            return output
        return self._fuse_selected(atoms, encoded['tokens'],
                                   encoded['canonical_graph_index'])

    def _fuse_selected(self, atoms, tokens, graph):
        if self.fusion_mode == 'xattn':
            return self.fusion(atoms, tokens, graph)
        return self.fusion(atoms, tokens)

    def forward(self, data, *, atom_mask=None):
        return self.fuse(self.encode(data, atom_mask=atom_mask))


def deployment_package(encoder, step, *, cutoffs=(2.0, 3.0, 4.0),
                       router_dense_updates=500, router_top_k=2, source=None,
                       progress=None):
    """Section 9.3 inference bundle: O8, experts, router, fusion and their LNs.

    ``progress`` is an optional ``callable(event, name)`` observation hook for
    the host copies: the export is the last thing a run does, so a run that
    stalls here leaves no record of which tensor it was inside.  Passing it
    changes no value, no device placement and no ordering.
    """
    state = {}
    for name, value in encoder.state_dict().items():
        if progress is not None:
            progress('tensor_start', name)
        state[name] = value.detach().cpu().clone()
        if progress is not None:
            progress('tensor_complete', name)
    return {
        'architecture': encoder.architecture_name,
        'fusion_mode': encoder.fusion_mode,
        'step': int(step),
        'cutoffs': [float(value) for value in cutoffs],
        'node_vocabulary': {'element': ELEMENT_VOCABULARY, 'charge': CHARGE_VOCABULARY,
                            'aromatic': AROMATIC_VOCABULARY},
        'edge_vocabulary': {'bond_categories': BOND_CATEGORY_COUNT,
                            'rbf_bins': 64, 'rbf_max': 4.0},
        'ph_definition': {'radii': [float(value) for value in ROUTER_RADII],
                          'columns': DESCRIPTOR_COLUMNS, 'max_edge_length': 4.5,
                          'field': 'F2', 'max_dimension': 1},
        'router': {'dense_updates': int(router_dense_updates), 'top_k': int(router_top_k),
                   'inference_mode': 'top2'},
        'training_route': 'mcl_ph',
        'source': dict(source or {}),
        'state_dict': state,
    }


def load_deployment(encoder, package, expected_step=None, *, expected_fusion=None):
    """Strict load: reject any legacy GLT/GALPH bundle or the wrong fusion mode."""
    if package.get('architecture') != encoder.architecture_name:
        raise ValueError('deployment is not an MCL-PH bundle')
    if package.get('training_route') != 'mcl_ph':
        raise ValueError('deployment was not produced by the MCL-PH route')
    if package.get('fusion_mode') != encoder.fusion_mode:
        raise ValueError('deployment fusion mode differs from the model')
    if expected_fusion is not None and package.get('fusion_mode') != str(expected_fusion):
        raise ValueError('deployment fusion mode differs from the requested arm')
    if expected_step is not None and int(package.get('step', -1)) != int(expected_step):
        raise ValueError('deployment step mismatch')
    if package.get('router', {}).get('inference_mode') != 'top2':
        raise ValueError('deployment must fix the inference routing to Top-2')
    current = encoder.state_dict()
    if set(package['state_dict']) != set(current):
        missing = sorted(set(current) - set(package['state_dict']))
        extra = sorted(set(package['state_dict']) - set(current))
        raise ValueError(f'deployment tensor set mismatch: missing={missing[:4]} extra={extra[:4]}')
    for name, value in package['state_dict'].items():
        if tuple(value.shape) != tuple(current[name].shape):
            raise ValueError(f'deployment tensor shape mismatch: {name}')
    current.update(package['state_dict'])
    encoder.load_state_dict(current, strict=True)
    encoder.branch.router.configure(TopologyRouter.TOP2,
                                    package.get('router', {}).get('dense_updates', 500))
    return encoder
