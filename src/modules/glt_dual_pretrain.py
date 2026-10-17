"""Atom/geometry objectives with optional FGR or ALIGN (no FP, no teacher)."""
import torch
import torch.distributed as dist
import torch.distributed.nn.functional as dist_nn
from torch import nn
from torch.nn import functional as F
from .glt_dual import build_dual_glt_model, mean_pool


class AtomGraphDecoder(nn.Module):
    def __init__(self, num_classes=101):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(1024, 512) for _ in range(2)])
        self.head = nn.Linear(512, int(num_classes))

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


def alignment_loss(graph_2d, graph_3d, identities, valid, *, temperature=0.1, direction='symmetric'):
    """Return (local sum, local valid-anchor count) for two-way multi-positive InfoNCE."""
    if not math_is_finite_positive(temperature):
        raise ValueError('ALIGN temperature must be finite and positive')
    if direction not in ('symmetric','2d_anchor'):
        raise ValueError('invalid ALIGN anchor direction')
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
        valid_columns = all_valid.to(device=logits.device).unsqueeze(0)
        masked_logits = logits.masked_fill(~valid_columns, -torch.inf)
        log_den = torch.logsumexp(masked_logits, dim=1)
        positive_count = positives.sum(1).clamp_min(1).float()
        return -(logits.masked_fill(~positives, 0).sum(1) / positive_count - log_den)
    local_loss = (.5 * (directional(logits23, positive23) + directional(logits32, positive32))
                  if direction=='symmetric' else directional(logits23, positive23) + z2.sum()*0)
    return local_loss[anchor].sum(), anchor.sum().float()


def math_is_finite_positive(value):
    return bool(torch.isfinite(torch.tensor(float(value))) and float(value) > 0)


class DualPretrainer(nn.Module):
    ENV_HEAD_INIT_SEED = 20260918

    def __init__(self, fusion_mode='concat', *, collect_diagnostics=False,
                 geometry_head_norm=False, third_task='none', fgr_mu=0.0,
                 fgr_sigma=1.0, align_temperature=0.1, atom_target='element',
                 env_vocab_size=None, torsion=False, torsion_revision=None, opt_mode=None,
                 fusion_revision=None, use_md200=False, atom_ce_input='raw',
                 align_enabled=False, local_enabled=False, geometry_scales=None,
                 geometry_revision=None, geometry_mode="finite_internal", angle_value_enabled=False, readout="internal", fusion_gate="none", align_weight=0.1, align_warmup_steps=1000, quality_root=None, align_revision=None, align_init_seed=None, align_direction='symmetric', **locality):
        super().__init__()
        third_task = str(third_task).lower()
        if third_task not in {'none', 'fgr', 'align'}:
            raise ValueError('unsupported third pretraining task')
        if not math_is_finite_positive(align_temperature):
            raise ValueError('ALIGN temperature must be finite and positive')
        if not math_is_finite_positive(fgr_sigma):
            raise ValueError('FGR sigma must be finite and positive')
        atom_target = str(atom_target).lower()
        if atom_target not in {'element', 'environment'}:
            raise ValueError('unsupported atom pretraining target')
        if (atom_target == 'environment') != (env_vocab_size is not None):
            raise ValueError('environment atom target requires the vocabulary size')
        self.atom_target = atom_target
        self.opt_mode = opt_mode
        if atom_ce_input not in ('raw', 'fused'):
            raise ValueError('atom CE input must be raw or fused')
        if align_enabled and local_enabled:
            raise ValueError('A+L is outside this cycle')
        self.align_weight=float(align_weight)
        self.align_warmup_steps=int(align_warmup_steps)
        self.quality_root=quality_root
        if not math_is_finite_positive(self.align_weight) or self.align_warmup_steps<=0:raise ValueError("invalid ALIGN schedule")
        self.atom_ce_input = atom_ce_input
        self.align_enabled = bool(align_enabled)
        self.local_enabled = bool(local_enabled)
        if local_enabled and fusion_mode != 'tk':
            raise ValueError('local constraint requires TK fusion')
        self.geometry_scales = geometry_scales
        if geometry_scales is not None and (set(geometry_scales) != {'length', 'angle'}
                or not all(math_is_finite_positive(v) for v in geometry_scales.values())):
            raise ValueError('geometry standard deviations must be positive and finite')
        self.encoder = build_dual_glt_model(fusion_mode, torsion=bool(torsion), torsion_revision=torsion_revision,
            fusion_revision=fusion_revision, use_md200=use_md200,
            geometry_revision=geometry_revision, geometry_mode=geometry_mode,
            angle_value_enabled=angle_value_enabled, readout=readout, fusion_gate=fusion_gate, **locality)
        self.encoder.predictor = nn.Identity()
        self.atom_head = AtomGraphDecoder(101 if atom_target == 'element' else int(env_vocab_size))
        if atom_target == 'environment':
            # Environment-specific output parameters get their own fixed seed;
            # every shared parameter is overwritten by the common initialization.
            generator = torch.Generator(device='cpu').manual_seed(self.ENV_HEAD_INIT_SEED)
            nn.init.normal_(self.atom_head.head.weight, std=0.02, generator=generator)
            nn.init.zeros_(self.atom_head.head.bias)
        self.length_head = nn.Sequential(nn.Linear(512, 256), nn.GELU(), nn.Linear(256, 1))
        self.angle_head = nn.Sequential(nn.Linear(1024, 256), nn.GELU(), nn.Linear(256, 1), nn.Tanh())
        self.third_task = third_task
        self.fgr_mu = float(fgr_mu)
        self.fgr_sigma = float(fgr_sigma)
        self.align_temperature = float(align_temperature)
        if third_task == 'fgr':
            self.fgr_context = (nn.Identity() if fusion_mode == 'concat' or fusion_revision
                                else nn.Linear(512, 1024))
            self.fgr_head = FGRDecoder(1024)
        else:
            self.fgr_context = None
            self.fgr_head = None
        self.align_revision=align_revision
        self.align_init_seed=align_init_seed
        self.align_direction=align_direction
        if align_direction not in ('symmetric','2d_anchor') or (align_direction!='symmetric' and align_revision!='glt_align_r10_a2'):
            raise ValueError('ALIGN direction requires the A2 revision')
        if align_revision is not None:
            allowed_ce=('raw','fused') if align_revision=='glt_align_r10_a1' else ('fused',)
            if align_revision not in ('glt_align_2d_u1','glt_align_r10_a1','glt_align_r10_a2') or type(align_init_seed) is not int or align_init_seed!=314161 or not align_enabled or third_task!='none' or align_temperature!=0.10 or self.align_weight!=0.10 or self.align_warmup_steps!=1000 or self.encoder.endpoint_revision or fusion_mode!='token' or fusion_gate!='none' or use_md200 or geometry_mode!='finite_incident' or geometry_revision!='glt_periodic_seam_r1' or readout!='incident' or not angle_value_enabled or atom_ce_input not in allowed_ce or not geometry_head_norm or opt_mode!='GLT_FUSION_R1' or local_enabled or torsion or quality_root is not None:raise ValueError('invalid revision-specific ALIGN configuration')
            if align_revision=='glt_align_r10_a2' and align_direction!='2d_anchor':raise ValueError('A2 requires 2D anchors')
        elif align_init_seed is not None:raise ValueError('ALIGN seed requires revision')
        if third_task == 'align' or align_enabled:
            if align_revision:
                with torch.random.fork_rng(devices=[]):
                    torch.default_generator.manual_seed(align_init_seed)
                    self.align_proj2 = AlignmentProjection()
                    self.align_proj3 = AlignmentProjection()
            else:
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
        # fusion/pooling/chemistry is left untouched.
        self.geometry_head_norm = bool(geometry_head_norm)
        self.geometry_norm = (nn.LayerNorm(512, elementwise_affine=False)
                              if self.geometry_head_norm else None)

    def forward(self, data, labels, *, collect_diagnostics=None, return_graph_feature=False):
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
            fused = self.encoder.fuse(encoded)
            atom_input = encoded['fused_atom_states'] if self.atom_ce_input == 'fused' or self.opt_mode == 'M_TOKEN_CE' else encoded['atom_states']
            atom_logits = self.atom_head(atom_input, data.lga_edge_index)
            chem, chem_valid = per_graph(F.cross_entropy(atom_logits[mask].float(),
                labels['atom_label'][mask], reduction='none'), data.canonical_graph_index[mask], graphs)
            bond_states = encoded['bond_states']
            if self.geometry_norm is not None:
                bond_states = self.geometry_norm(bond_states)
            center_rows = (data.bond_length_weight > 0) if self.encoder.geometry_revision or self.opt_mode == 'G_SEAM_GEO' else data.bond_center.bool()
            lengths = self.length_head(bond_states[center_rows]).float().flatten()
            if self.encoder.geometry_revision or self.opt_mode == 'G_SEAM_GEO':
                from .glt_dual_opt import weighted_graph_loss
                length_loss, length_valid = weighted_graph_loss((lengths-labels['distance'].float()).square(), labels['distance_weight'], data.bond_batch[center_rows], graphs)
            else:
                length_loss, length_valid = per_graph((lengths-labels['distance'].float()).square(), data.bond_batch[center_rows], graphs)
            a, b = labels['angle_pairs'].unbind(1)
            left, right = bond_states[a], bond_states[b]
            angles = self.angle_head(torch.cat([left + right, (left - right).abs()], -1)).float().flatten()
            angle_loss, angle_valid = per_graph((angles - labels['angle_cos'].float()).square(),
                                                labels['angle_graph'], graphs)
            geo_valid = length_valid & encoded['geometry_valid']
            raw_length, raw_angle = length_loss, angle_loss
            if self.geometry_scales is not None:
                length_loss = length_loss / float(self.geometry_scales['length'])**2
                angle_loss = angle_loss / float(self.geometry_scales['angle'])**2
            geometry = length_loss + angle_loss
            quality=labels.get('geometry_quality_weight')
            if quality is not None:
                if quality.shape!=(graphs,) or not torch.isfinite(quality).all() or not ((quality==0)|(quality==.5)|(quality==1)).all():
                    raise ValueError('invalid geometry quality weight')
                geometry=geometry*quality
            if self.third_task == 'fgr':
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
                counts=torch.stack([chem_valid.sum(), geo_valid.sum() if quality is None else quality[geo_valid].sum(), third_count]),
                targets=torch.stack([mask.sum(), center_rows.sum(),
                                     labels['angle_graph'].new_tensor(a.numel()),
                                     third_target_count.to(dtype=torch.long)]),
                angle_graphs=angle_valid.sum() if quality is None else quality[angle_valid].sum())
            if self.encoder.fusion_revision:
                zero = fused.sum() * 0
                align_sum, align_count = zero, zero.detach()
                if self.align_enabled:
                    align_sum, align_count = alignment_loss(
                        self.align_proj2(encoded['graph_2d']),
                        self.align_proj3(encoded['graph_3d']), labels['align_identity'],
                        labels['align_valid'].bool() & encoded['geometry_valid'],
                        temperature=self.align_temperature,direction=self.align_direction)
                local_sum, local_count = zero, zero.detach()
                if self.local_enabled:
                    local_sum, local_count = local_attention_loss(
                        encoded['local_attention'], data.incidence_index, zero)
                result['extra_sums'] = torch.stack([align_sum, local_sum])
                result['extra_counts'] = torch.stack([align_count, local_count])
                result['raw_geometry_sums'] = torch.stack([
                    raw_length[geo_valid].sum() if quality is None else (raw_length*quality)[geo_valid].sum(),
                    raw_angle[geo_valid].sum() if quality is None else (raw_angle*quality)[geo_valid].sum()])
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
                        "third": third_sum,
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
            if return_graph_feature:
                result['graph_feature'] = fused
            return result
        finally:
            for handle in handles:
                handle.remove()


def global_objective(sums, global_counts, world_size=1, weights=(1., 1., .1), *, fractional_counts=False):
    """DDP averages gradients: multiply local graph sums by world/global count."""
    denominators=(torch.where(global_counts>0,global_counts,torch.ones_like(global_counts))
                  if fractional_counts else global_counts.clamp_min(1))
    return (sums * sums.new_tensor(weights) * world_size / denominators).sum()


def local_attention_loss(attention, incidence, zero):
    """KL(P||A), uniform incidence targets, atom then graph averaging."""
    losses = []
    bond, atom = incidence.long()
    for _, ai, bi, weights in attention:
        atom_losses = []
        for row, index in enumerate(ai):
            incident = bond[atom == index].unique()
            if not incident.numel():
                continue
            columns = torch.searchsorted(bi, incident)
            if bool((columns >= bi.numel()).any()) or not torch.equal(bi[columns], incident):
                raise ValueError('center incidence leaves graph physical bond tokens')
            probabilities = weights[row, columns]
            if not bool(torch.isfinite(probabilities).all()) or bool((probabilities <= 0).any()):
                raise FloatingPointError('local attention probabilities invalid')
            atom_losses.append(-probabilities.log().mean() - probabilities.new_tensor(float(incident.numel())).log())
        if atom_losses:
            losses.append(torch.stack(atom_losses).mean())
    return (torch.stack(losses).sum(), zero.new_tensor(float(len(losses)))) if losses else (zero, zero.detach())


def deployment_package(pretrainer, step):
    package=dict(architecture=pretrainer.encoder.architecture_name,
        fusion_gate=pretrainer.encoder.fusion_gate, quality_root=pretrainer.quality_root,
        align_weight=pretrainer.align_weight, align_warmup_steps=pretrainer.align_warmup_steps, align_temperature=pretrainer.align_temperature,
        fusion_mode=pretrainer.encoder.fusion_mode, step=int(step), use_md200=pretrainer.encoder.use_md200,
        fusion_revision=pretrainer.encoder.fusion_revision,
        atom_ce_input=pretrainer.atom_ce_input, align_enabled=pretrainer.align_enabled,
        local_enabled=pretrainer.local_enabled, geometry_scales=pretrainer.geometry_scales,
        torsion_modules=bool(pretrainer.encoder.glt.torsion_mlp is not None),
        readout=pretrainer.encoder.readout, opt_mode=pretrainer.opt_mode,
        geometry_revision=pretrainer.encoder.geometry_revision, geometry_mode=pretrainer.encoder.geometry_mode,
        angle_value_enabled=pretrainer.encoder.angle_value_enabled,
        torsion_mode='off' if pretrainer.encoder.glt.torsion_mlp is None else 'on',
        input_schema='periodic_local_observations_v1' if pretrainer.encoder.geometry_revision else 'legacy',
        geometry_target='seam' if pretrainer.encoder.geometry_revision or pretrainer.opt_mode == 'G_SEAM_GEO' else 'internal',
        third_task=pretrainer.third_task, element_mask='all_copies_unknown' if pretrainer.encoder.fusion_revision or pretrainer.opt_mode in ('F_TOKEN','M_TOKEN_CE') else 'legacy',
        state_dict={k: v.detach().cpu().clone() for k, v in pretrainer.encoder.state_dict().items()
                    if not k.startswith('predictor.')})

    if pretrainer.encoder.endpoint_revision:
        from .glt_dual import locality_options
        package.update(locality_options({k:getattr(pretrainer.encoder,k) for k in ('endpoint_revision','endpoint_module','endpoint_input_mode','endpoint_init_seed','fusion_attention_mode','incident_bias')}))
    if pretrainer.align_revision:
        package.update(pretrain_revision=pretrainer.align_revision,align_revision=pretrainer.align_revision,align_init_seed=pretrainer.align_init_seed,align_input='raw_graph_2d_graph_3d',fusion_attention_mode='global',incident_bias=0.0,endpoint_revision=None)
    if pretrainer.encoder.torsion_revision:
        from .glt_dual import PHYSICAL_TORSION,S4_RESOURCE
        from pathlib import Path
        package.update(PHYSICAL_TORSION)
        package.update(S4_RESOURCE,common_initialization=str(Path(__file__).resolve().parents[2]/'results/glt_periodic_seam_20261009/r1/initialization/F_CE_step0.pt'))
        package.update(plan='GLT-OPTIMIZATION-20261011-01',revision='r6',cycle='S4',arm='T_FCE',common_initial_state='results/glt_periodic_seam_20261009/r1/initialization/F_CE_step0.pt')
    return package


def u1_alignment_options(package):
    if package.get('r10_revision'):
        from .glt_r10 import validate_package
        cfg=validate_package(package)
        return {k:package[k] for k in ('align_revision','align_init_seed','align_direction')} if cfg['align_enabled'] else {}
    names=('pretrain_revision','align_revision','align_init_seed','align_input','align_enabled','align_temperature','align_weight','align_warmup_steps')
    if not package.get('pretrain_revision') and not package.get('align_revision'):
        if package.get('align_init_seed') is not None or package.get('align_input') is not None:raise ValueError('incomplete U1 ALIGN metadata')
        return {}
    expected=dict(pretrain_revision='glt_align_2d_u1',align_revision='glt_align_2d_u1',align_init_seed=314161,align_input='raw_graph_2d_graph_3d',align_enabled=True,align_temperature=.1,align_weight=.1,align_warmup_steps=1000,fusion_attention_mode='global',incident_bias=0.0,endpoint_revision=None,fusion_gate='none',local_enabled=False,use_md200=False,torsion_modules=False,atom_ce_input='fused',geometry_revision='glt_periodic_seam_r1',geometry_mode='finite_incident',readout='incident',angle_value_enabled=True,third_task='none',opt_mode='GLT_FUSION_R1')
    if any(k not in package or type(package[k]) is not type(v) or package[k]!=v for k,v in expected.items()):raise ValueError('invalid U1 ALIGN metadata')
    return {k:package[k] for k in names}


def load_deployment(model, package, expected_step=5000):
    from .glt_dual import locality_options
    from .glt_dual import torsion_options
    torsion=torsion_options(package)
    r10=bool(package.get('r10_revision'))
    if torsion and not r10:
        from .glt_dual import s4_resource_options
        s4_resource_options(package)
    if package.get('torsion_revision')!=model.torsion_revision:
        raise ValueError('deployment physical torsion revision mismatch')
    if torsion and not r10 and any(package.get(k)!=v for k,v in dict(plan='GLT-OPTIMIZATION-20261011-01',revision='r6',cycle='S4',arm='T_FCE').items()):
        raise ValueError('deployment physical torsion cycle mismatch')
    alignment=u1_alignment_options(package)
    if package.get('nonbonded_revision')!=model.glt.nonbonded_revision:raise ValueError('deployment nonbonded revision mismatch')
    locality=locality_options(package)
    if bool(locality)!=bool(model.endpoint_revision) or any(getattr(model,k)!=v for k,v in locality.items()):
        raise ValueError('deployment locality/endpoint mismatch')
    if (package.get('architecture') != model.architecture_name
            or package.get('fusion_mode') != model.fusion_mode
            or package.get('use_md200') != model.use_md200 or package.get('step') != expected_step
            or package.get('fusion_revision') != model.fusion_revision):
        raise ValueError('dual checkpoint architecture/fusion/step mismatch')
    if package.get('readout', 'internal') != model.readout:
        raise ValueError('deployment readout mismatch')
    from src.dataset.glt_periodic_geometry import geometry_options
    options=geometry_options(package.get('geometry_revision'), package.get('geometry_mode','finite_internal'),
        package.get('angle_value_enabled',False),package.get('readout','internal'))
    for key,value in options.items():
        if getattr(model,key)!=value:raise ValueError('deployment geometry mismatch: '+key)
    if package.get('torsion_modules',False) != (model.glt.torsion_mlp is not None):
        raise ValueError('deployment torsion modules mismatch')
    if model.geometry_revision:
        if (package.get('input_schema')!='periodic_local_observations_v1'
                or package.get('geometry_target')!='seam'
                or package.get('element_mask')!='all_copies_unknown'):
            raise ValueError('deployment geometry schema/target/mask mismatch')
    if package.get('fusion_gate','none') != model.fusion_gate:raise ValueError('deployment gate mismatch')
    current = model.state_dict()
    expected = {name for name in current if not name.startswith('predictor.')}
    if set(package['state_dict']) != expected:
        raise ValueError('deployment must contain exactly encoder and fusion tensors')
    for name,value in package['state_dict'].items():
        if value.shape!=current[name].shape or value.dtype!=current[name].dtype:
            raise ValueError('deployment tensor shape/dtype mismatch: '+name)
        if (locality or alignment or torsion) and not torch.isfinite(value).all():raise ValueError('nonfinite S3 deployment tensor: '+name)
    current.update(package['state_dict'])
    model.load_state_dict(current, strict=True)


def deployment_model_options(package, expected_step=5000):
    """Resolve only the published GLT encoder contracts before construction."""
    from .glt_dual import locality_options, torsion_options, s4_resource_options
    from src.dataset.glt_periodic_geometry import geometry_options
    required = ('architecture', 'fusion_mode', 'fusion_revision', 'use_md200', 'step', 'state_dict')
    if not isinstance(package, dict) or any(k not in package for k in required):
        raise ValueError('incomplete GLT deployment package')
    if type(package['step']) is not int or package['step'] != expected_step:
        raise ValueError('deployment step mismatch')
    if package['fusion_revision'] not in (None, 'glt_fusion_r1'):
        raise ValueError('unknown fusion revision')
    legacy = package['fusion_revision'] is None and package.get('input_schema') == 'legacy'
    # The saved original F_CE predates gate/connection/ALIGN-revision fields.
    # Defaults apply only to its exact observed producer schema and semantics.
    fce_fields = {'architecture','fusion_mode','step','use_md200','fusion_revision',
        'atom_ce_input','align_enabled','local_enabled','geometry_scales',
        'torsion_modules','readout','opt_mode','geometry_revision','geometry_mode',
        'angle_value_enabled','torsion_mode','input_schema','geometry_target',
        'third_task','element_mask','state_dict'}
    old_fce = set(package) == fce_fields and all(package.get(k) == v for k,v in dict(
        fusion_mode='token',fusion_revision='glt_fusion_r1',use_md200=False,
        atom_ce_input='fused',align_enabled=False,local_enabled=False,torsion_modules=False,
        readout='incident',opt_mode='GLT_FUSION_R1',geometry_revision='glt_periodic_seam_r1',
        geometry_mode='finite_incident',angle_value_enabled=True,torsion_mode='off',
        input_schema='periodic_local_observations_v1',geometry_target='seam',
        third_task='none',element_mask='all_copies_unknown',geometry_scales=None).items())
    fields = ('geometry_revision', 'geometry_mode', 'angle_value_enabled', 'readout',
              'fusion_gate', 'torsion_modules', 'torsion_mode', 'align_enabled',
              'atom_ce_input', 'local_enabled', 'third_task', 'opt_mode',
              'input_schema', 'geometry_target', 'element_mask',
              'align_temperature','align_weight','align_warmup_steps')
    if not legacy and any(k not in package for k in fields if not (old_fce and k in ('fusion_gate','align_temperature','align_weight','align_warmup_steps'))):
        raise ValueError('new deployment is missing architecture metadata')
    if legacy and (package.get('geometry_revision') is not None
                   or package.get('pretrain_revision') is not None
                   or package.get('endpoint_revision') is not None
                   or package.get('torsion_revision') is not None):
        raise ValueError('legacy defaults do not apply to revised packages')
    for name in ('use_md200', 'torsion_modules', 'align_enabled', 'local_enabled', 'angle_value_enabled'):
        if name in package and type(package[name]) is not bool:
            raise ValueError('invalid boolean metadata: ' + name)
    geometry = geometry_options(package.get('geometry_revision'),
        package.get('geometry_mode', 'finite_internal'),
        package.get('angle_value_enabled', False), package.get('readout', 'internal'))
    locality = locality_options(package)
    connection = dict(fusion_attention_mode=package.get('fusion_attention_mode','global'),
                      incident_bias=package.get('incident_bias',0.0))
    if not locality and (connection['fusion_attention_mode'] != 'global'
                         or type(connection['incident_bias']) not in (int,float)
                         or connection['incident_bias'] != 0.0):
        raise ValueError('connection requires an explicit endpoint contract')
    torsion = torsion_options(package)
    u1_alignment_options(package)
    if package.get('align_enabled') and not package.get('pretrain_revision'):
        # Earlier fusion packages used ALIGN without the later U1 projections.
        if package['fusion_revision'] != 'glt_fusion_r1' or package.get('opt_mode') != 'GLT_FUSION_R1':
            raise ValueError('unsupported legacy ALIGN contract')
    if torsion and not package.get('r10_revision'):
        s4_resource_options(package)
        if any(package.get(k) != v for k,v in dict(plan='GLT-OPTIMIZATION-20261011-01',
                revision='r6', cycle='S4', arm='T_FCE').items()):
            raise ValueError('deployment physical torsion cycle mismatch')
    if package.get('reference_revision') is not None:
        if package['reference_revision'] != 'glt_matched_reference_s5':
            raise ValueError('unknown reference revision')
        # Reuse S5's exact config/source binding rather than weaken its contract.
        from scripts.run_glt_pt_matched_reference_unit import reference_metadata
        cfg = package.get('training_config')
        if not isinstance(cfg, dict) or any(package.get(k) != v for k,v in reference_metadata(cfg).items()):
            raise ValueError('S5 parent metadata/source mismatch')
        if any(package.get(k) for k in ('torsion_modules', 'torsion_revision', 'endpoint_revision', 'align_enabled')):
            raise ValueError('S5 deployment contains forbidden branch')
    if package.get('torsion_mode', 'off') != ('on' if package.get('torsion_modules', False) else 'off'):
        raise ValueError('torsion mode/modules mismatch')
    if geometry['geometry_revision'] and any(package.get(k) != v for k,v in dict(
            input_schema='periodic_local_observations_v1',geometry_target='seam',
            element_mask='all_copies_unknown').items()):
        raise ValueError('deployment geometry schema/target/mask mismatch')
    architecture = ('O8-BondPath-GalformerTrimer-Hop2-MD200-r1'
                    if package['fusion_revision'] else 'O8-BondPath-GalformerTrimer-Hop2')
    if package['architecture'] != architecture:
        raise ValueError('unknown GLT architecture')
    result=dict(fusion_mode=package['fusion_mode'], fusion_revision=package['fusion_revision'],
        use_md200=package['use_md200'], fusion_gate=package.get('fusion_gate', 'none'),
        torsion=package.get('torsion_modules', False), torsion_revision=package.get('torsion_revision'),
        **geometry, **{**connection, **locality})
    if package.get('nonbonded_revision'):
        if not package.get('r10_revision'):raise ValueError('nonbonded input requires r10 contract')
        result['nonbonded_revision']=package['nonbonded_revision']
    return result


def restore_deployment(package, expected_step=5000, *, dropout=0.1):
    """Construct the declared encoder, then use the existing strict loader."""
    from .glt_dual import build_dual_glt_model
    options = deployment_model_options(package, expected_step)
    if not isinstance(package['state_dict'], dict) or any(
            not torch.is_tensor(value) or not torch.isfinite(value).all()
            for value in package['state_dict'].values()):
        raise ValueError('invalid/nonfinite deployment tensors')
    model = build_dual_glt_model(dropout=dropout, **options)
    load_deployment(model, package, expected_step)
    return model
