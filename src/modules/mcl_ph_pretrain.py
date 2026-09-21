"""Two atom cross-entropies, multiscale geometry supervision and the balance term.

Plan ``MCL-PH-20260921-01/r1``, contract section 7.

``L = L_atom + L_geo + 1e-3 * L_bal``

* ``L_atom``  -- one shared ``Linear512 -> 101`` head applied to the 2D state
  and to the fused state; a geometry-invalid graph contributes its 2D term only.
* ``L_geo``   -- three local-geometry terms (one per scale, one shared decoder)
  and the per-scale non-bond distance term, each with its own effective
  denominator.
* ``L_bal``   -- the router's load-balance term, computed on the globally valid
  graph set through an autograd-aware collective.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .mcl_ph import MCLPHEncoder, xavier_

ATOM_CLASSES = 101
BALANCE_WEIGHT = 1e-3
HUBER_BETA = 0.5
EXPERT_HIDDEN = 128


def per_graph(loss, index, graphs):
    """Per-graph mean and the graphs that actually contributed a target."""
    counts = torch.bincount(index, minlength=int(graphs))
    means = loss.new_zeros(int(graphs)).index_add(0, index, loss) / counts.clamp_min(1)
    return means, counts > 0


class LocalGeometryDecoder(nn.Module):
    """One decoder shared by the three experts (section 7.2)."""

    def __init__(self, hidden=EXPERT_HIDDEN):
        super().__init__()
        self.bond = nn.Sequential(nn.Linear(3 * hidden, 256), nn.GELU(),
                                  nn.Linear(256, hidden))
        self.length = nn.Linear(hidden, 1)
        self.angle = nn.Sequential(nn.Linear(2 * hidden, 128), nn.GELU(),
                                   nn.Linear(128, 1), nn.Tanh())
        self.apply(xavier_)

    def bond_repr(self, states, bonds):
        left, right = states[bonds[:, 0]], states[bonds[:, 1]]
        return self.bond(torch.cat([left + right, (left - right).abs(), left * right], dim=-1))

    def lengths(self, states, bonds):
        return self.length(self.bond_repr(states, bonds)).float().flatten()

    def angles(self, states, angle_pairs, bonds):
        """``angle_pairs`` indexes two centre bonds; the angle lives on their atoms."""
        first = self.bond_repr(states, bonds[angle_pairs[:, 0]])
        second = self.bond_repr(states, bonds[angle_pairs[:, 1]])
        return self.angle(torch.cat([first + second, (first - second).abs()], dim=-1)).float().flatten()


class NonBondDecoder(nn.Module):
    """One decoder shared by the three declared distance bins (section 7.3)."""

    def __init__(self, hidden=EXPERT_HIDDEN):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(3 * hidden, 256), nn.GELU(), nn.Linear(256, 1))
        self.apply(xavier_)

    def forward(self, states, pairs):
        left, right = states[pairs[:, 0]], states[pairs[:, 1]]
        features = torch.cat([left + right, (left - right).abs(), left * right], dim=-1)
        return self.net(features).float().flatten()


class MCLPHPretrainer(nn.Module):
    """The new pretraining objective; no fingerprint, no MD200, no third task."""

    def __init__(self, fusion_mode='gate', dropout=0.1, cutoffs=(2.0, 3.0, 4.0),
                 router_dense_updates=500, collect_diagnostics=False):
        super().__init__()
        self.encoder = MCLPHEncoder(fusion_mode, dropout, cutoffs, router_dense_updates)
        self.atom_head = nn.Linear(512, ATOM_CLASSES)
        self.local_decoder = LocalGeometryDecoder()
        self.nonbond_decoder = NonBondDecoder()
        self.collect_diagnostics = bool(collect_diagnostics)
        self.last_diagnostics = None
        self.apply(_initialise_new_heads)

    @property
    def fusion_mode(self):
        return self.encoder.fusion_mode

    def set_router_mode(self, mode, step=None):
        self.encoder.branch.router.configure(mode, step)

    def forward(self, data, labels):
        self.last_diagnostics = None
        self.encoder.collect_diagnostics = self.collect_diagnostics
        encoded = self.encoder.encode(data, atom_mask=labels['atom_mask'])
        fused = self.encoder.fuse(encoded)
        graphs = int(data.graph_available.numel())
        mask = labels['atom_mask'].bool()
        atom_target = labels['atom_label'].long()
        flat_two = self.atom_head(encoded['atom_states']).float()
        flat_fused = self.atom_head(fused).float()
        loss_two, _ = per_graph(
            F.cross_entropy(flat_two[mask], atom_target[mask], reduction='none'),
            data.canonical_graph_index[mask], graphs)
        loss_fused, _ = per_graph(
            F.cross_entropy(flat_fused[mask], atom_target[mask], reduction='none'),
            data.canonical_graph_index[mask], graphs)
        geometric = data.mcl_geometry_valid.bool()
        per_graph_atom = torch.where(geometric, 0.5 * loss_two + 0.5 * loss_fused, loss_two)
        masked_graph = torch.zeros(graphs, dtype=torch.bool, device=per_graph_atom.device)
        if bool(mask.any()):
            masked_graph[data.canonical_graph_index[mask]] = True
        atom_valid = masked_graph & (torch.bincount(data.canonical_graph_index[mask],
                                                    minlength=graphs) > 0)
        local, nonbond, geo_valid = self._geometry_terms(data, labels, encoded, graphs)
        geometry = torch.where(local['valid'] & nonbond['valid'],
                               0.5 * local['value'] + 0.5 * nonbond['value'],
                               torch.where(local['valid'], local['value'],
                                           torch.where(nonbond['valid'], nonbond['value'],
                                                       local['value'] * 0.0)))
        report = {
            'atom_sum': per_graph_atom[atom_valid].sum(),
            'atom_count': atom_valid.sum(),
            'geo_sum': geometry[geo_valid].sum(),
            'geo_count': geo_valid.sum(),
            'local_count': local['valid'].sum(),
            'nonbond_count': nonbond['valid'].sum(),
            'balance': encoded['balance'],
            'balance_empty_valid': encoded['balance_empty_valid'],
            'atom_states': encoded['atom_states'],
            'mixed': encoded['mixed'],
            'expert_states': encoded['expert_states'],
            'fused': fused,
            'alpha': encoded['alpha'],
            'router_top_k': encoded['router_top_k'],
            'router_diagnostics': encoded.get('diagnostics'),
            'gate_diagnostics': encoded.get('gate_diagnostics'),
            'targets': torch.stack([
                mask.sum().to(per_graph_atom.dtype),
                labels['mcl_length_pair'].new_tensor(labels['mcl_length_pair'].size(0)),
                labels['mcl_angle_pair'].new_tensor(labels['mcl_angle_pair'].size(0)),
                labels['mcl_nonbond_pair'].new_tensor(labels['mcl_nonbond_pair'].size(0)),
                geo_valid.sum().to(per_graph_atom.dtype)]),
        }
        if self.collect_diagnostics:
            report['diagnostics'] = {
                'atom_two_sum': float(loss_two[atom_valid].sum().detach()),
                'atom_fused_sum': float(loss_fused[atom_valid].sum().detach()),
                'atom_valid_count': int(atom_valid.sum()),
                'geometry_valid_count': int(geo_valid.sum()),
                'local_valid_count': int(local['valid'].sum()),
                'nonbond_valid_count': int(nonbond['valid'].sum()),
                'balance_value': float(encoded['balance'].detach()),
                'router': encoded.get('diagnostics'),
            }
            self.last_diagnostics = report['diagnostics']
        return report

    def _geometry_terms(self, data, labels, encoded, graphs):
        """Per-graph local-geometry and non-bond values with independent denominators."""
        device = encoded['atom_states'].device
        length_pairs = labels['mcl_length_pair'].long()
        angle_pairs = labels['mcl_angle_pair'].long()
        length_target = labels['mcl_length_target'].float()
        angle_target = labels['mcl_angle_target'].float()
        length_graph = labels['mcl_length_graph'].long()
        angle_graph = labels['mcl_angle_graph'].long()
        bonds = data.mcl_bond_index.long()
        per_expert = []
        for states in encoded['expert_states']:
            terms, weights = [], []
            if int(length_pairs.size(0)):
                prediction = self.local_decoder.lengths(states, length_pairs)
                loss, valid = per_graph(
                    F.huber_loss(prediction, length_target, reduction='none', delta=HUBER_BETA),
                    length_graph, graphs)
                terms.append(loss)
                weights.append(valid.float())
            if int(angle_pairs.size(0)):
                prediction = self.local_decoder.angles(states, angle_pairs, bonds)
                loss, valid = per_graph(
                    F.huber_loss(prediction, angle_target, reduction='none', delta=HUBER_BETA),
                    angle_graph, graphs)
                terms.append(loss)
                weights.append(valid.float())
            if not terms:
                per_expert.append(torch.zeros(graphs, device=device))
                continue
            # Inside one graph the present terms are equally weighted; the
            # three experts are then weighted equally with each other.
            present = torch.stack(weights).sum(0).clamp_min(1.0)
            per_expert.append(torch.stack(terms).sum(0) / present)
        local_value = torch.stack(per_expert).mean(0)
        covered = torch.zeros(graphs, dtype=torch.bool, device=device)
        if int(length_pairs.size(0)):
            covered[length_graph] = True
        if int(angle_pairs.size(0)):
            covered[angle_graph] = True
        nonbond_value, nonbond_valid = self._nonbond_terms(labels, encoded, graphs)
        return ({'value': local_value, 'valid': covered},
                {'value': nonbond_value, 'valid': nonbond_valid},
                covered | nonbond_valid)

    def _nonbond_terms(self, labels, encoded, graphs):
        device = encoded['atom_states'].device
        value = torch.zeros(graphs, device=device)
        valid = torch.zeros(graphs, dtype=torch.bool, device=device)
        pairs = labels['mcl_nonbond_pair'].long()
        if not int(pairs.size(0)):
            return value, valid
        slot = labels['mcl_nonbond_slot'].long()
        target = labels['mcl_nonbond_target'].float()
        graph = labels['mcl_nonbond_graph'].long()
        bins = []
        for index, states in enumerate(encoded['expert_states']):
            selected = torch.nonzero(slot == index, as_tuple=False).flatten()
            if not int(selected.numel()):
                bins.append(None)
                continue
            prediction = self.nonbond_decoder(states, pairs[selected])
            loss, present = per_graph(
                F.huber_loss(prediction, target[selected], reduction='none', delta=HUBER_BETA),
                graph[selected], graphs)
            bins.append((loss, present))
        available = [item for item in bins if item is not None]
        if not available:
            return value, valid
        stacked = torch.stack([item[0] for item in available])
        present = torch.stack([item[1].float() for item in available])
        non_empty = present.sum(0)
        value = (stacked * present).sum(0) / non_empty.clamp_min(1.0)
        valid = non_empty > 0
        return value, valid

    def objective(self, report, weights=(1.0, 1.0, BALANCE_WEIGHT), world_size=1):
        """DDP-correct global objective: each task keeps its own denominator."""
        world = float(max(1, int(world_size)))
        weight = torch.as_tensor(weights, dtype=torch.float32,
                                 device=report['atom_sum'].device)
        atom = report['atom_sum'] * world / report['atom_count'].to(torch.float32).clamp_min(1.0)
        geometry = (report['geo_sum'] * world
                    / report['geo_count'].to(torch.float32).clamp_min(1.0))
        return atom * weight[0] + geometry * weight[1] + report['balance'] * weight[2]


def _initialise_new_heads(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
