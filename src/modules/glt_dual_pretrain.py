"""Three-task objective for the physical Trimer dual encoder (no teacher)."""
import torch
import torch.distributed as dist
import torch.distributed.nn.functional as dist_nn
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


class FGRDecoder(nn.Module):
    """Symmetric endpoint decoder for the fusion-conditioned geometry task."""

    def __init__(self, context_dim=1024):
        super().__init__()
        self.sum_norm = nn.LayerNorm(512, elementwise_affine=False)
        self.diff_norm = nn.LayerNorm(512, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(512 + 512 + int(context_dim), 256), nn.GELU(), nn.Linear(256, 1)
        )

    def forward(self, left, right, context):
        if left.shape != right.shape or left.ndim != 2 or left.size(-1) != 512:
            raise ValueError('FGR endpoint states must be [N,512] and paired')
        return self.mlp(torch.cat([self.sum_norm(left + right),
                                    self.diff_norm((left - right).abs()), context], dim=-1))


class AlignmentProjection(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(512, 256), nn.GELU(), nn.Linear(256, 128))

    def forward(self, value):
        return F.normalize(self.net(value).float(), dim=-1, eps=1e-8)


def _gather_padded(value, lengths):
    """Gather a local [N,D] tensor after deterministic zero-padding."""

    world = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
    if world == 1:
        return value, lengths
    max_length = max(int(length) for length in lengths)
    padded = value.new_zeros((max_length, value.size(-1)))
    padded[:value.size(0)] = value
    # PyTorch's distributed autograd gather uses reduce-scatter/all-to-all in
    # backward, which keeps remote-key gradients in the same collective order
    # as DDP.  A hand-written all_reduce here can deadlock against DDP buckets.
    return torch.cat(dist_nn.all_gather(padded), dim=0), lengths


def _gather_objects(value):
    world = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
    if world == 1:
        return [value]
    gathered = [None] * world
    dist.all_gather_object(gathered, value)
    return gathered


def alignment_local_anchor_count(identities, valid, *, device=None):
    """Count valid local anchors when the distributed pool has negatives.

    This helper has no autograd state and is intentionally usable before the
    forward pass to establish the optimizer-window denominator.
    """
    local_ids = [str(value) for value, keep in zip(identities, valid) if bool(keep)]
    gathered = _gather_objects(local_ids)
    unique = {item for group in gathered for item in group}
    count = len(local_ids) if len(unique) >= 2 else 0
    if device is None:
        return count
    return torch.tensor(float(count), device=device)


def alignment_loss(graph_2d, graph_3d, identities, valid, *, temperature=0.1):
    """Return (local sum, local valid-anchor count) for two-way multi-positive InfoNCE."""
    if not math_is_finite_positive(temperature):
        raise ValueError('ALIGN temperature must be finite and positive')
    if graph_2d.shape != graph_3d.shape or graph_2d.ndim != 2 or graph_2d.size(-1) != 128:
        raise ValueError('ALIGN projected representations must be [B,128] and paired')
    valid = valid.bool()
    world = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
    if world == 1:
        lengths = [graph_2d.size(0)]
    else:
        local_length = torch.tensor([graph_2d.size(0)], device=graph_2d.device, dtype=torch.long)
        gathered_lengths = [torch.zeros_like(local_length) for _ in range(world)]
        dist.all_gather(gathered_lengths, local_length)
        lengths = [int(value.item()) for value in gathered_lengths]
    z2, _ = _gather_padded(graph_2d.float(), lengths)
    z3, _ = _gather_padded(graph_3d.float(), lengths)
    max_length = max(lengths)
    local_ids = [str(value) for value in identities]
    local_mask = [bool(value) for value in valid.tolist()]
    ids = _gather_objects(local_ids + [''] * (max_length - len(local_ids)))
    masks = _gather_objects(local_mask + [False] * (max_length - len(local_mask)))
    all_ids = [item for group in ids for item in group]
    all_valid = torch.tensor([item for group in masks for item in group],
                             device=graph_2d.device, dtype=torch.bool)
    local_valid = valid
    distinct = {item for item, keep in zip(all_ids, all_valid.tolist()) if keep}
    if len(distinct) < 2 or not bool(local_valid.any()):
        # Keep the differentiable gather outputs in the graph even for a rank
        # with no local anchors; otherwise other ranks enter reduce-scatter
        # during backward while this rank skips it.
        return (z2.sum() * 0 + z3.sum() * 0), torch.tensor(0., device=graph_2d.device)
    # Padding slots are never valid and therefore never enter either the
    # denominator or positive mask.
    logits23 = graph_2d.float() @ z3.t() / float(temperature)
    logits32 = graph_3d.float() @ z2.t() / float(temperature)
    all_ids_tensor = all_ids
    local_ids = [str(value) for value in identities]
    positive23 = torch.tensor(
        [[a == b and keep for b, keep in zip(all_ids_tensor, all_valid.tolist())]
         for a in local_ids], device=graph_2d.device, dtype=torch.bool)
    positive32 = torch.tensor(
        [[a == b and keep for b, keep in zip(all_ids_tensor, all_valid.tolist())]
         for a in local_ids], device=graph_2d.device, dtype=torch.bool)
    # The two masks are identical for paired rows; separate variables keep the
    # directional formulas explicit and prevent accidental transpose changes.
    anchor = local_valid & (torch.tensor([item in distinct for item in local_ids],
                                         device=graph_2d.device, dtype=torch.bool))
    if not bool(anchor.any()):
        return (z2.sum() * 0 + z3.sum() * 0), torch.tensor(0., device=graph_2d.device)
    def directional(logits, positives):
        log_den = torch.logsumexp(logits, dim=1)
        positive_count = positives.sum(1).clamp_min(1).float()
        return -(logits.masked_fill(~positives, 0).sum(1) / positive_count - log_den)
    local_loss = .5 * (directional(logits23, positive23) + directional(logits32, positive32))
    return local_loss[anchor].sum(), anchor.sum().float()


def math_is_finite_positive(value):
    return bool(torch.isfinite(torch.tensor(float(value))) and float(value) > 0)


class DualPretrainer(nn.Module):
    def __init__(self, fusion_mode='concat', *, collect_diagnostics=False,
                 geometry_head_norm=False, third_task='fp', fgr_mu=0.0,
                 fgr_sigma=1.0, align_temperature=0.1):
        super().__init__()
        third_task = str(third_task).lower()
        if third_task not in {'fp', 'none', 'fgr', 'align'}:
            raise ValueError('unsupported third pretraining task')
        if not math_is_finite_positive(align_temperature):
            raise ValueError('ALIGN temperature must be finite and positive')
        if not math_is_finite_positive(fgr_sigma):
            raise ValueError('FGR sigma must be finite and positive')
        self.encoder = build_dual_glt_model(fusion_mode)
        self.encoder.predictor = nn.Identity()
        self.atom_head = AtomGraphDecoder()
        self.length_head = nn.Sequential(nn.Linear(512, 256), nn.GELU(), nn.Linear(256, 1))
        self.angle_head = nn.Sequential(nn.Linear(1024, 256), nn.GELU(), nn.Linear(256, 1), nn.Tanh())
        self.third_task = third_task
        self.fgr_mu = float(fgr_mu)
        self.fgr_sigma = float(fgr_sigma)
        self.align_temperature = float(align_temperature)
        self.fp_head = (nn.Sequential(nn.Linear(1024 if fusion_mode == 'concat' else 512, 512),
                                      nn.GELU(), nn.Dropout(0.1), nn.Linear(512, 2048))
                        if third_task == 'fp' else None)
        if third_task == 'fgr':
            self.fgr_context = (nn.Identity() if fusion_mode == 'concat'
                                else nn.Linear(512, 1024))
            self.fgr_head = FGRDecoder(1024)
        else:
            self.fgr_context = None
            self.fgr_head = None
        if third_task == 'align':
            self.align_proj2 = AlignmentProjection()
            self.align_proj3 = AlignmentProjection()
        else:
            self.align_proj2 = self.align_proj3 = None
        # Diagnostics are observation only: no parameter, buffer, RNG draw or
        # forward value depends on this flag.
        self.collect_diagnostics = bool(collect_diagnostics)
        self.last_diagnostics = None
        # P2 single-change experiment: ONE geometry-specific, non-affine
        # normalization of the 3D bond representation shared by the length and
        # angle heads.  Non-affine => no parameters, no buffers, no optimizer
        # state, unchanged checkpoint schema.  The raw 3D state that feeds
        # fusion/pooling/fingerprint/chemistry is left untouched.
        self.geometry_head_norm = bool(geometry_head_norm)
        self.geometry_norm = (nn.LayerNorm(512, elementwise_affine=False)
                              if self.geometry_head_norm else None)

    def forward(self, data, labels, *, collect_diagnostics=None):
        diagnostics_enabled = (self.collect_diagnostics
                               if collect_diagnostics is None else bool(collect_diagnostics))
        # Do not let a sampled update's detached observations leak into an
        # unsampled update when diagnostics are decimated by the caller.
        self.last_diagnostics = None
        captured, handles = {}, []
        if diagnostics_enabled:
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
            bond_states = encoded['bond_states']
            if self.geometry_norm is not None:
                bond_states = self.geometry_norm(bond_states)
            center_rows = data.bond_center.bool()
            lengths = self.length_head(bond_states[center_rows]).float().flatten()
            length_loss, length_valid = per_graph((lengths - labels['distance'].float()).square(),
                data.bond_batch[data.bond_center], graphs)
            a, b = labels['angle_pairs'].unbind(1)
            left, right = bond_states[a], bond_states[b]
            angles = self.angle_head(torch.cat([left + right, (left - right).abs()], -1)).float().flatten()
            angle_loss, angle_valid = per_graph((angles - labels['angle_cos'].float()).square(),
                                                labels['angle_graph'], graphs)
            geo_valid = length_valid & encoded['geometry_valid']
            geometry = length_loss + angle_loss
            fused = self.encoder.fuse(encoded)
            if self.third_task == 'fp':
                fp_logits = self.fp_head(fused).float()
                third_per_graph = F.binary_cross_entropy_with_logits(
                    fp_logits, labels['fingerprint'].float(), reduction='none').mean(-1)
                third_valid = data.graph_available.bool()
                third_target_count = third_valid.sum() * 2048
            elif self.third_task == 'fgr':
                pair_index = labels['fgr_pair_index'].long()
                if pair_index.numel():
                    pair_states = self.fgr_head(
                        encoded['atom_states'][pair_index[:, 0]],
                        encoded['atom_states'][pair_index[:, 1]],
                        self.fgr_context(fused)[labels['fgr_graph'].long()])
                    pair_loss = F.huber_loss(pair_states.flatten(), labels['fgr_target'].float(),
                                             reduction='none', delta=.5)
                    fgr_per_graph, pair_valid = per_graph(
                        pair_loss, labels['fgr_graph'].long(), graphs)
                else:
                    fgr_per_graph = fused.sum(-1) * 0
                    pair_valid = torch.zeros(graphs, dtype=torch.bool, device=fused.device)
                third_per_graph = fgr_per_graph
                third_valid = pair_valid & data.geometry_valid.bool()
                third_target_count = torch.tensor(labels['fgr_target'].numel(),
                                                  device=fused.device, dtype=torch.long)
            elif self.third_task == 'align':
                z2 = self.align_proj2(encoded['graph_2d'])
                z3 = self.align_proj3(encoded['graph_3d'])
                align_valid = labels['align_valid'].bool() & encoded['geometry_valid'].bool()
                third_sum, third_count = alignment_loss(
                    z2, z3, labels['align_identity'], align_valid,
                    temperature=self.align_temperature)
                # Use graph-connected zeros for per-graph diagnostic paths;
                # the objective consumes the explicitly pooled ALIGN sum below.
                third_per_graph = fused.sum(-1) * 0
                third_valid = torch.zeros(graphs, dtype=torch.bool, device=fused.device)
                third_target_count = third_count.detach().to(dtype=torch.long)
            else:
                third_sum = fused.sum() * 0
                third_per_graph = fused.sum(-1) * 0
                third_valid = torch.zeros(graphs, dtype=torch.bool, device=fused.device)
                third_target_count = torch.zeros((), dtype=torch.long, device=fused.device)
            if self.third_task != 'align':
                third_sum = third_per_graph[third_valid].sum()
                third_count = third_valid.sum()
            else:
                third_count = third_count.to(dtype=torch.long)
            result = dict(sums=torch.stack([chem[chem_valid].sum(), geometry[geo_valid].sum(),
                                            third_sum]),
                counts=torch.stack([chem_valid.sum(), geo_valid.sum(), third_count]),
                targets=torch.stack([mask.sum(), data.bond_center.sum(),
                                     labels['angle_graph'].new_tensor(a.numel()),
                                     third_target_count.to(dtype=torch.long)]),
                angle_graphs=angle_valid.sum())
            if diagnostics_enabled:
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
                        "fingerprint" if self.third_task == 'fp' else "third": third_sum,
                    },
                    "components": {
                        "chem_sum": float(chem[chem_valid].sum().detach()), "chem_count": int(chem_valid.sum()),
                        "length_sum": float(length_loss[geo_valid].sum().detach()),
                        "angle_sum": float(angle_loss[geo_valid].sum().detach()),
                        "geometry_sum": float(geometry[geo_valid].sum().detach()),
                        "geometry_valid_count": int(geo_valid.sum()),
                        "angle_valid_count": int(angle_valid.sum()),
                        "graphs_without_angle": int((geo_valid & ~angle_valid).sum()),
                        "third_task": self.third_task,
                        "third_sum": float(third_sum.detach()), "third_count": int(third_count),
                        "fp_sum": float(third_sum.detach()) if self.third_task == 'fp' else None,
                        "fp_count": int(third_count) if self.third_task == 'fp' else 0,
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
                    "geometry_head_norm_state": {
                        "enabled": bool(self.geometry_head_norm),
                        "state_rms": float(bond_states[center_rows].detach().float().pow(2).mean().sqrt())
                        if bool(center_rows.any()) else None,
                        "state_abs_p99": float(bond_states[center_rows].detach().float().abs().quantile(0.99))
                        if bool(center_rows.any()) else None,
                    },
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
