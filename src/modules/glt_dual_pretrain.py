"""Three-task objective for the physical Trimer dual encoder (no teacher)."""
import torch
from torch import nn
from torch.nn import functional as F
from .glt_dual import build_dual_glt_model, mean_pool


class AtomGraphDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(1024, 512) for _ in range(2)])
        self.head = nn.Linear(512, 101)

    def forward(self, states, edge_index):
        source, target = edge_index
        for layer in self.layers:
            neighbors = mean_pool(states[source], target, states.size(0))
            states = F.gelu(layer(torch.cat([states, neighbors], -1)))
        return self.head(states)


def per_graph(loss, index, graphs):
    counts = torch.bincount(index, minlength=graphs)
    means = loss.new_zeros(graphs).index_add(0, index, loss) / counts.clamp_min(1)
    return means, counts > 0


class DualPretrainer(nn.Module):
    def __init__(self, fusion_mode='concat'):
        super().__init__()
        self.encoder = build_dual_glt_model(fusion_mode)
        self.encoder.predictor = nn.Identity()
        self.atom_head = AtomGraphDecoder()
        self.length_head = nn.Sequential(nn.Linear(512, 256), nn.GELU(), nn.Linear(256, 1))
        self.angle_head = nn.Sequential(nn.Linear(1024, 256), nn.GELU(), nn.Linear(256, 1), nn.Tanh())
        self.fp_head = nn.Sequential(nn.Linear(1024 if fusion_mode == 'concat' else 512, 512),
                                     nn.GELU(), nn.Dropout(0.1), nn.Linear(512, 2048))

    def forward(self, data, labels):
        encoded = self.encoder.encode(data, atom_mask=labels['atom_mask'])
        graphs = data.graph_available.numel()
        mask = labels['atom_mask']
        atom_logits = self.atom_head(encoded['atom_states'], data.lga_edge_index)
        chem, chem_valid = per_graph(F.cross_entropy(atom_logits[mask].float(),
            labels['atom_label'][mask], reduction='none'), data.canonical_graph_index[mask], graphs)
        lengths = self.length_head(encoded['center_bond_states']).float().flatten()
        length_loss, length_valid = per_graph((lengths - labels['distance'].float()).square(),
            data.bond_batch[data.bond_center], graphs)
        a, b = labels['angle_pairs'].unbind(1)
        left, right = encoded['bond_states'][a], encoded['bond_states'][b]
        angles = self.angle_head(torch.cat([left + right, (left - right).abs()], -1)).float().flatten()
        angle_loss, angle_valid = per_graph((angles - labels['angle_cos'].float()).square(),
                                            labels['angle_graph'], graphs)
        geo_valid = length_valid & encoded['geometry_valid']
        geometry = length_loss + angle_loss
        fp_logits = self.fp_head(self.encoder.fuse(encoded)).float()
        fingerprint = F.binary_cross_entropy_with_logits(fp_logits, labels['fingerprint'].float(),
                                                         reduction='none').mean(-1)
        fp_valid = data.graph_available.bool()
        return dict(sums=torch.stack([chem[chem_valid].sum(), geometry[geo_valid].sum(),
                                      fingerprint[fp_valid].sum()]),
            counts=torch.stack([chem_valid.sum(), geo_valid.sum(), fp_valid.sum()]),
            targets=torch.stack([mask.sum(), data.bond_center.sum(),
                                  labels['angle_graph'].new_tensor(a.numel()), fp_valid.sum() * 2048]),
            angle_graphs=angle_valid.sum())


def global_objective(sums, global_counts, world_size=1, weights=(1., 1., .1)):
    """DDP averages gradients: multiply local graph sums by world/global count."""
    return (sums * sums.new_tensor(weights) * world_size / global_counts.clamp_min(1)).sum()


def deployment_package(pretrainer, step):
    return dict(architecture=pretrainer.encoder.architecture_name,
        fusion_mode=pretrainer.encoder.fusion_mode, step=int(step), use_md200=False,
        state_dict={k: v.detach().cpu().clone() for k, v in pretrainer.encoder.state_dict().items()
                    if not k.startswith('predictor.')})


def load_deployment(model, package, expected_step=5000):
    if (package.get('architecture') != model.architecture_name
            or package.get('fusion_mode') != model.fusion_mode
            or package.get('use_md200') is not False or package.get('step') != expected_step):
        raise ValueError('dual checkpoint architecture/fusion/step mismatch')
    current = model.state_dict()
    expected = {name for name in current if not name.startswith('predictor.')}
    if set(package['state_dict']) != expected:
        raise ValueError('deployment must contain exactly encoder and fusion tensors')
    current.update(package['state_dict'])
    model.load_state_dict(current, strict=True)
