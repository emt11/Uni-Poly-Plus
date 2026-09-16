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


def _stats(values):
    """Detached descriptive statistics for one diagnostic tensor."""

    flat = values.detach().float().reshape(-1)
    if flat.numel() == 0:
        return {"count": 0}
    quantiles = torch.quantile(flat, torch.tensor([0.05, 0.5, 0.95], device=flat.device))
    return {
        "count": int(flat.numel()),
        "mean": float(flat.mean()), "std": float(flat.std(unbiased=False)),
        "q05": float(quantiles[0]), "q50": float(quantiles[1]), "q95": float(quantiles[2]),
        "min": float(flat.min()), "max": float(flat.max()),
    }


class DualPretrainer(nn.Module):
    def __init__(self, fusion_mode='concat', *, collect_diagnostics=False):
        super().__init__()
        self.encoder = build_dual_glt_model(fusion_mode)
        self.encoder.predictor = nn.Identity()
        self.atom_head = AtomGraphDecoder()
        self.length_head = nn.Sequential(nn.Linear(512, 256), nn.GELU(), nn.Linear(256, 1))
        self.angle_head = nn.Sequential(nn.Linear(1024, 256), nn.GELU(), nn.Linear(256, 1), nn.Tanh())
        self.fp_head = nn.Sequential(nn.Linear(1024 if fusion_mode == 'concat' else 512, 512),
                                     nn.GELU(), nn.Dropout(0.1), nn.Linear(512, 2048))
        # Diagnostics are observation only: no parameter, buffer, RNG draw or
        # forward value depends on this flag.
        self.collect_diagnostics = bool(collect_diagnostics)
        self.last_diagnostics = None

    def forward(self, data, labels):
        captured, handles = {}, []
        if self.collect_diagnostics:
            # Read-only hooks: they copy the pre-tanh angle output and the 3D
            # layer states; the computations themselves are unchanged.
            handles.append(self.angle_head[2].register_forward_hook(
                lambda module, inputs, output: captured.__setitem__('pre_tanh', output.detach())))
            def _layer_hook(layer_index):
                def hook(module, inputs, output):
                    captured.setdefault('layer_rms', {})[layer_index] = float(
                        output.detach().float().pow(2).mean().sqrt())
                return hook

            for index, layer in enumerate(self.encoder.glt.layers):
                handles.append(layer.register_forward_hook(_layer_hook(index)))
            handles.append(self.encoder.glt.distance_basis.register_forward_hook(
                lambda module, inputs, output: captured.__setitem__('distance_basis', output.detach())))
            handles.append(self.encoder.glt.angle_bias.gaussian.register_forward_hook(
                lambda module, inputs, output: captured.__setitem__('angle_basis', output.detach())))
        try:
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
            result = dict(sums=torch.stack([chem[chem_valid].sum(), geometry[geo_valid].sum(),
                                            fingerprint[fp_valid].sum()]),
                counts=torch.stack([chem_valid.sum(), geo_valid.sum(), fp_valid.sum()]),
                targets=torch.stack([mask.sum(), data.bond_center.sum(),
                                     labels['angle_graph'].new_tensor(a.numel()), fp_valid.sum() * 2048]),
                angle_graphs=angle_valid.sum())
            if self.collect_diagnostics:
                pre_tanh = captured.get('pre_tanh')
                # One valid path step per row marks a real one-hop angle; rows
                # that are padding or the synthetic self relation are counted
                # separately and excluded from the angle statistics.
                line_mask = data.line_mask.detach().bool()
                rows_with_one_step = line_mask.sum(-1) == 1
                line_self = getattr(data, 'line_is_self',
                                    torch.zeros(rows_with_one_step.shape, dtype=torch.bool,
                                                device=line_mask.device)).detach().bool()
                angle_rows = rows_with_one_step & ~line_self
                distance_sigma = self.encoder.glt.distance_basis
                angle_sigma = self.encoder.glt.angle_bias.gaussian
                diagnostics = {
                    # Gradient-carrying scalars, same formulas and valid masks as
                    # the training objective.  Diagnostics only: never registered
                    # as parameters or buffers, never returned to the loss path.
                    "component_tensors": {
                        "chem": chem[chem_valid].sum(),
                        "length": length_loss[geo_valid].sum(),
                        "angle": angle_loss[geo_valid].sum(),
                        "fingerprint": fingerprint[fp_valid].sum(),
                    },
                    "components": {
                        "chem_sum": float(chem[chem_valid].sum().detach()), "chem_count": int(chem_valid.sum()),
                        "length_sum": float(length_loss[geo_valid].sum().detach()),
                        "angle_sum": float(angle_loss[geo_valid].sum().detach()),
                        "geometry_sum": float(geometry[geo_valid].sum().detach()),
                        "geometry_valid_count": int(geo_valid.sum()),
                        "angle_valid_count": int(angle_valid.sum()),
                        "graphs_without_angle": int((geo_valid & ~angle_valid).sum()),
                        "fp_sum": float(fingerprint[fp_valid].sum().detach()), "fp_count": int(fp_valid.sum()),
                        "length_plus_angle_reconstructs_geometry": bool(torch.isclose(
                            length_loss[geo_valid].sum() + angle_loss[geo_valid].sum(),
                            geometry[geo_valid].sum(), rtol=1e-5, atol=1e-6)),
                    },
                    "targets": {"length": _stats(labels['distance']),
                                "angle_cos": _stats(labels['angle_cos'])},
                    "predictions": {"length": _stats(lengths), "angle_cos": _stats(angles)},
                    "angle_head": ({"pre_tanh": _stats(torch.empty(0)), "tanh_derivative": _stats(torch.empty(0)),
                                    "exact_plus_minus_one_fraction": None,
                                    "near_saturation_fraction_threshold_6": None,
                                    "note": "no angle rows in this batch"}
                                   if pre_tanh is None or pre_tanh.numel() == 0 else {
                        "pre_tanh": _stats(pre_tanh.float()),
                        "tanh_derivative": _stats(1.0 - torch.tanh(pre_tanh.float()).square()),
                        "exact_plus_minus_one_fraction": float((torch.tanh(pre_tanh.float()).abs() == 1.0)
                                                               .float().mean()),
                        "near_saturation_fraction_threshold_6": float((pre_tanh.float().abs() > 6.0)
                                                                      .float().mean()),
                        "note": "thresholds are descriptive statistics only, not a validity gate",
                    }),
                    "representations": {
                        "bond_states_rms": float(encoded['bond_states'].detach().float().pow(2).mean().sqrt()),
                        "center_bond_states_rms": float(encoded['center_bond_states'].detach().float().pow(2).mean().sqrt()),
                        "graph_3d_rms": float(encoded['graph_3d'].detach().float().pow(2).mean().sqrt()),
                        "layer_rms": captured.get('layer_rms', {}),
                    },
                    "gaussian": {
                        "distance_min_effective_sigma": float(distance_sigma.stds.detach().abs().min() + 1e-2),
                        "distance_mean_effective_sigma": float((distance_sigma.stds.detach().abs() + 1e-2).mean()),
                        "distance_max_effective_sigma": float(distance_sigma.stds.detach().abs().max() + 1e-2),
                        "angle_min_effective_sigma": float(angle_sigma.stds.detach().abs().min() + 1e-2),
                        "angle_mean_effective_sigma": float((angle_sigma.stds.detach().abs() + 1e-2).mean()),
                        "angle_max_effective_sigma": float(angle_sigma.stds.detach().abs().max() + 1e-2),
                        "distance_affine_mul_mean": float(distance_sigma.mul.weight.detach().mean()),
                        "distance_affine_mul_std": float(distance_sigma.mul.weight.detach().std()),
                        "distance_affine_bias_mean": float(distance_sigma.bias.weight.detach().mean()),
                        # Real forward outputs, read through read-only hooks.
                        "distance_basis_output_rms": (
                            float(captured['distance_basis'].float().pow(2).mean().sqrt())
                            if captured.get('distance_basis') is not None
                            and captured['distance_basis'].numel() else None),
                        "angle_basis_output_rms_valid_relations": (
                            float(captured['angle_basis'][angle_rows].float().pow(2).mean().sqrt())
                            if captured.get('angle_basis') is not None and int(angle_rows.sum()) else None),
                        "angle_rows_valid": int(angle_rows.sum()),
                        "angle_rows_padding": int((line_mask.sum(-1) == 0).sum()),
                        "angle_rows_synthetic_self": int((rows_with_one_step & line_self).sum()),
                    },
                }
                result['diagnostics'] = diagnostics
                self.last_diagnostics = diagnostics
            return result
        finally:
            for handle in handles:
                handle.remove()


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
