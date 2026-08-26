import os
import sys
import json
import warnings
import hashlib
import time
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset as TorchDataset, WeightedRandomSampler

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.dataset.mips_trimer_contract import (
    CACHE_BUNDLE_SCHEMA as MIPS_TRIMER_CACHE_BUNDLE_SCHEMA,
    TOPOLOGY_LMDB_SCHEMA as MIPS_TRIMER_TOPOLOGY_SCHEMA,
    CHECKPOINT_SCHEMA as MIPS_TRIMER_CHECKPOINT_SCHEMA,
    ROUTE_INTERNAL as MTS_ROUTE_INTERNAL,
    ROUTE_NAME as MTS_ROUTE_NAME,
    ROUTE_SHORT_NAME as MTS_ROUTE_SHORT_NAME,
    STAGE1_ID as MTS_STAGE1_ID,
    validate_runtime_args as validate_mips_trimer_runtime,
)
from src.dataset.lmdb_cache import sample_key_from_smiles
from src.training.finetune.config import parse_arguments
from src.training.finetune.mode_specs import missing_batch_fields


class SingleFoldContext:
    """Validated task/seed/fold coordinates for one finite job."""

    __slots__ = ("task", "fold", "seed", "experiment_id")

    def __init__(self, task: str, fold: int, seed: int, experiment_id: str):
        self.task = str(task)
        self.fold = int(fold)
        self.seed = int(seed)
        self.experiment_id = str(experiment_id)

    def validate(self) -> None:
        if not self.task:
            raise ValueError("single-fold task is required")
        if self.fold not in range(5):
            raise ValueError("single-fold fold must be in [0, 4]")


def run_single_fold(train_and_evaluate, *args, context: SingleFoldContext, **kwargs):
    """Compatibility boundary for callers that already own the fold loop."""
    context.validate()
    return train_and_evaluate(*args, **kwargs)


def _capture_initial_model_state(model, enabled):
    """Snapshot the fold initialization only when full-train refit needs it."""
    if not enabled:
        return None
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def _atomic_save_best_checkpoint(model, path, metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.tmp.{os.getpid()}')
    payload = {
        'state_dict': {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        },
        **dict(metadata),
    }
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


@torch.no_grad()
def _collect_interaction_diagnostics(model, loader, device):
    base = model.module if hasattr(model, 'module') else model
    encoder = base.encoders['graph'].encoder
    if getattr(encoder, 'interaction_update', None) is None:
        return None
    was_training = bool(model.training)
    model.eval()
    ratios = []
    cosines = []
    valid_count = 0
    for batch in loader:
        batch = batch.to(device, non_blocking=True)
        views = encoder.interaction_views(batch)
        valid = views['valid'].bool()
        finite = (
            torch.isfinite(views['h3']).all(dim=-1)
            & torch.isfinite(views['h3_new']).all(dim=-1)
            & torch.isfinite(views['delta3']).all(dim=-1)
        )
        keep = valid & finite
        if not bool(keep.any()):
            continue
        h3 = views['h3'][keep]
        h3_new = views['h3_new'][keep]
        delta = views['delta3'][keep]
        ratios.append(
            torch.linalg.vector_norm(delta, dim=-1)
            / (torch.linalg.vector_norm(h3, dim=-1) + 1e-8)
        )
        cosines.append(torch.nn.functional.cosine_similarity(h3, h3_new, dim=-1))
        valid_count += int(keep.sum())
    if was_training:
        model.train()
    if not ratios:
        raise RuntimeError('interaction diagnostics found no finite valid atoms')
    ratio = torch.cat(ratios).float().cpu()
    cosine = torch.cat(cosines).float().cpu()
    return {
        'mean_abs_tanh_g23': float(
            torch.tanh(encoder.interaction_update.gate.detach()).abs().mean()
        ),
        'mean_update_norm_ratio': float(ratio.mean()),
        'median_update_norm_ratio': float(ratio.median()),
        'p25_update_norm_ratio': float(torch.quantile(ratio, 0.25)),
        'p75_update_norm_ratio': float(torch.quantile(ratio, 0.75)),
        'mean_h3_h3_new_cosine': float(cosine.mean()),
        'valid_atom_count': int(valid_count),
    }


@torch.no_grad()
def _collect_line_conditioning_diagnostics(model, loader, device):
    base = model.module if hasattr(model, 'module') else model
    encoder = base.encoders['graph'].encoder
    projection = getattr(
        getattr(encoder, 'glt', None), 'line_conditioning_projection', None
    )
    if projection is None:
        return None
    was_training = bool(model.training)
    model.eval()
    absolute_modulation = []
    relative_change = []
    cosine = []
    internal = []
    cross_ru = []
    valid_count = 0
    for batch in loader:
        batch = batch.to(device, non_blocking=True)
        views = encoder.line_conditioning_views(batch)
        valid = views['line_valid'].bool()
        h0 = views['line_input']
        hnew = views['conditioned_line_input']
        modulation = views['line_modulation']
        finite = (
            torch.isfinite(h0).all(dim=-1)
            & torch.isfinite(hnew).all(dim=-1)
            & torch.isfinite(modulation).all(dim=-1)
        )
        keep = valid & finite
        if not bool(keep.any()):
            continue
        per_line_abs = modulation[keep].abs().mean(dim=-1)
        absolute_modulation.append(per_line_abs)
        relative_change.append(
            torch.linalg.vector_norm(hnew[keep] - h0[keep], dim=-1)
            / (torch.linalg.vector_norm(h0[keep], dim=-1) + 1e-8)
        )
        cosine.append(torch.nn.functional.cosine_similarity(h0[keep], hnew[keep], dim=-1))
        shift = views['line_shift'][keep].abs()
        if bool((shift == 0).any()):
            internal.append(per_line_abs[shift == 0])
        if bool((shift != 0).any()):
            cross_ru.append(per_line_abs[shift != 0])
        valid_count += int(keep.sum())
    if was_training:
        model.train()
    if not absolute_modulation:
        raise RuntimeError('line diagnostics found no finite valid tokens')
    values = torch.cat(absolute_modulation).float().cpu()
    ratio = torch.cat(relative_change).float().cpu()
    cos = torch.cat(cosine).float().cpu()
    return {
        'mean_abs_modulation': float(values.mean()),
        'median_abs_modulation': float(values.median()),
        'p25_abs_modulation': float(torch.quantile(values, 0.25)),
        'p75_abs_modulation': float(torch.quantile(values, 0.75)),
        'p95_abs_modulation': float(torch.quantile(values, 0.95)),
        'mean_relative_token_change': float(ratio.mean()),
        'median_relative_token_change': float(ratio.median()),
        'mean_h0_hnew_cosine': float(cos.mean()),
        'valid_line_count': int(valid_count),
        'internal_line_mean_abs_modulation': (
            float(torch.cat(internal).mean()) if internal else float('nan')
        ),
        'cross_ru_line_mean_abs_modulation': (
            float(torch.cat(cross_ru).mean()) if cross_ru else float('nan')
        ),
    }


def _finite_distribution(values):
    values = torch.cat(values).detach().float().cpu()
    if values.numel() == 0 or not bool(torch.isfinite(values).all()):
        raise RuntimeError('bond-bias diagnostics require finite observations')
    return {
        'count': int(values.numel()),
        'mean': float(values.mean()),
        'median': float(values.median()),
        'p25': float(torch.quantile(values, 0.25)),
        'p75': float(torch.quantile(values, 0.75)),
        'p95': float(torch.quantile(values, 0.95)),
    }


@torch.no_grad()
def _collect_o8_bond_bias_diagnostics(model, loader, device):
    """Audit the exact fold-best BC/BT bias on canonical test bonds."""
    base = model.module if hasattr(model, 'module') else model
    encoder = base.encoders['graph'].encoder
    module = getattr(getattr(encoder, 'o8', None), 'direct_bond_bias', None)
    if module is None:
        return None
    was_training = bool(model.training)
    model.eval()
    all_abs, internal_abs, cross_abs = [], [], []
    category_abs = {index: [] for index in range(module.num_categories)}
    category_count = {index: 0 for index in range(module.num_categories)}
    for batch in loader:
        batch = batch.to(device, non_blocking=True)
        categories = batch.glt_token_bond_type.long().reshape(-1)
        shifts = batch.glt_token_shift.long().reshape(-1).abs()
        if categories.numel() != shifts.numel():
            raise ValueError('bond diagnostic token field length mismatch')
        bias = module.category_bias(categories)
        if not bool(torch.isfinite(bias).all()):
            raise RuntimeError('bond bias contains NaN/Inf')
        per_token = bias.abs().mean(dim=-1)
        all_abs.append(per_token)
        if bool((shifts == 0).any()):
            internal_abs.append(per_token[shifts == 0])
        if bool((shifts != 0).any()):
            cross_abs.append(per_token[shifts != 0])
        for category in range(module.num_categories):
            selected = categories == category
            if bool(selected.any()):
                category_abs[category].append(per_token[selected])
                category_count[category] += int(selected.sum())
    if was_training:
        model.train()
    if not all_abs:
        raise RuntimeError('bond-bias diagnostics found no test bonds')
    category_vectors = module.category_bias(
        torch.arange(module.num_categories, device=device)
    ).detach().float().cpu()
    return {
        'mode': str(module.mode),
        'num_categories': int(module.num_categories),
        'new_parameter_count': int(sum(p.numel() for p in module.parameters())),
        'absolute_bias': _finite_distribution(all_abs),
        'internal_absolute_bias': _finite_distribution(internal_abs),
        'cross_ru_absolute_bias': _finite_distribution(cross_abs),
        'category_count': {str(k): int(v) for k, v in category_count.items()},
        'category_mean_absolute_bias': {
            str(k): (
                float(torch.cat(v).mean().cpu()) if v else None
            ) for k, v in category_abs.items()
        },
        'category_per_head_bias': {
            str(k): [float(x) for x in category_vectors[k].tolist()]
            for k in range(module.num_categories)
        },
    }


@torch.no_grad()
def _collect_attention_routing_diagnostics(model, loader, device):
    from torch_scatter import scatter

    base = model.module if hasattr(model, 'module') else model
    encoder = base.encoders['graph'].encoder
    projection = getattr(
        getattr(encoder, 'glt', None), 'attention_conditioning_projection', None
    )
    if projection is None:
        return None
    was_training = bool(model.training)
    model.eval()
    gammas, entropy0, entropy1, changes, kls = [], [], [], [], []
    internal_gamma, cross_gamma = [], []
    internal_change, cross_change = [], []
    valid_count = 0
    for batch in loader:
        batch = batch.to(device, non_blocking=True)
        views = encoder.attention_conditioning_views(batch)
        gamma = views['gamma']
        valid = views['token_valid'].bool() & torch.isfinite(gamma).all(dim=-1)
        if bool(valid.any()):
            per_line = gamma[valid].abs().mean(dim=-1)
            gammas.append(per_line)
            shifts = views['token_shift'][valid].abs()
            if bool((shifts == 0).any()):
                internal_gamma.append(per_line[shifts == 0])
            if bool((shifts != 0).any()):
                cross_gamma.append(per_line[shifts != 0])
            valid_count += int(valid.sum())
        w0 = views['baseline_attention'].float().clamp_min(1e-12)
        w1 = views['conditioned_attention'].float().clamp_min(1e-12)
        target = views['target'].long()
        token_count = int(gamma.size(0))
        h0 = scatter(-(w0 * w0.log()), target, dim=0, dim_size=token_count, reduce='sum')
        h1 = scatter(-(w1 * w1.log()), target, dim=0, dim_size=token_count, reduce='sum')
        kl = scatter(w1 * (w1.log() - w0.log()), target, dim=0, dim_size=token_count, reduce='sum')
        entropy0.append(h0[valid].reshape(-1))
        entropy1.append(h1[valid].reshape(-1))
        kls.append(kl[valid].reshape(-1))
        per_relation_change = (w1 - w0).abs().mean(dim=-1)
        changes.append(per_relation_change)
        source_shift = views['token_shift'][views['source'].long()].abs()
        target_shift = views['token_shift'][target].abs()
        cross_relation = (source_shift != 0) | (target_shift != 0)
        if bool((~cross_relation).any()):
            internal_change.append(per_relation_change[~cross_relation])
        if bool(cross_relation.any()):
            cross_change.append(per_relation_change[cross_relation])
        if not torch.equal(views['baseline_value'], views['conditioned_value']):
            raise RuntimeError('attention routing changed the V tensor')
    if was_training:
        model.train()
    if not gammas:
        raise RuntimeError('attention diagnostics found no finite valid lines')
    gamma = torch.cat(gammas).cpu()
    base_entropy = torch.cat(entropy0).cpu()
    conditioned_entropy = torch.cat(entropy1).cpu()
    return {
        'mean_abs_gamma': float(gamma.mean()),
        'median_abs_gamma': float(gamma.median()),
        'p25_abs_gamma': float(torch.quantile(gamma, 0.25)),
        'p75_abs_gamma': float(torch.quantile(gamma, 0.75)),
        'p95_abs_gamma': float(torch.quantile(gamma, 0.95)),
        'baseline_attention_entropy_mean': float(base_entropy.mean()),
        'conditioned_attention_entropy_mean': float(conditioned_entropy.mean()),
        'conditioned_attention_entropy_median': float(conditioned_entropy.median()),
        'attention_entropy_delta': float((conditioned_entropy - base_entropy).mean()),
        'mean_attention_probability_change': float(torch.cat(changes).mean()),
        'mean_attention_kl': float(torch.cat(kls).mean()),
        'internal_line_mean_abs_gamma': (
            float(torch.cat(internal_gamma).mean())
            if internal_gamma else float('nan')
        ),
        'cross_ru_line_mean_abs_gamma': (
            float(torch.cat(cross_gamma).mean())
            if cross_gamma else float('nan')
        ),
        'internal_relation_attention_change': (
            float(torch.cat(internal_change).mean())
            if internal_change else float('nan')
        ),
        'cross_ru_relation_attention_change': (
            float(torch.cat(cross_change).mean())
            if cross_change else float('nan')
        ),
        'valid_line_count': int(valid_count),
        'value_path_unchanged': True,
    }


@torch.no_grad()
def _collect_torsion_diagnostics(model, loader, device):
    base = model.module if hasattr(model, 'module') else model
    encoder = base.encoders['graph'].encoder
    if getattr(getattr(encoder, 'glt', None), 'torsion_bias', None) is None:
        return None
    was_training = bool(model.training)
    model.eval()
    torsion_values, angle_values = [], []
    internal_values, cross_values = [], []
    counts = []
    relation_total = relation_covered = graph_total = graph_covered = 0
    internal_total = internal_covered = cross_total = cross_covered = 0
    for batch in loader:
        batch = batch.to(device, non_blocking=True)
        views = encoder.torsion_views(batch)
        real = views['real_relation_mask'].bool()
        covered = views['torsion_covered'][real]
        relation_total += int(real.sum())
        relation_covered += int(covered.sum())
        graph_total += int(batch.glt_graph_torsion_covered.numel())
        graph_covered += int(batch.glt_graph_torsion_covered.sum())
        relation_counts = views['torsion_count'][real][covered]
        if relation_counts.numel():
            counts.append(relation_counts.float())
        torsion = views['torsion_bias_real'].float()
        angle = views['angle_bias'].float()
        if covered.any():
            torsion_values.append(torsion[covered].abs().reshape(-1))
            angle_values.append(angle[covered].abs().reshape(-1))
        source_cross = views['torsion_source_cross_ru'][real]
        internal = ~source_cross
        internal_total += int(internal.sum())
        internal_covered += int((internal & covered).sum())
        cross_total += int(source_cross.sum())
        cross_covered += int((source_cross & covered).sum())
        if (internal & covered).any():
            internal_values.append(torsion[internal & covered].abs().reshape(-1))
        if (source_cross & covered).any():
            cross_values.append(torsion[source_cross & covered].abs().reshape(-1))
    if was_training:
        model.train()
    if relation_covered == 0 or not counts:
        raise RuntimeError('torsion diagnostics found no covered relations')
    count_values = torch.cat(counts).cpu()
    torsion_abs = torch.cat(torsion_values).cpu()
    angle_abs = torch.cat(angle_values).cpu()
    mean_torsion = float(torsion_abs.mean())
    mean_angle = float(angle_abs.mean())
    return {
        'relation_total': relation_total,
        'relation_covered': relation_covered,
        'torsion_relation_coverage': relation_covered / max(relation_total, 1),
        'graph_total': graph_total,
        'graph_covered': graph_covered,
        'torsion_graph_coverage': graph_covered / max(graph_total, 1),
        'mean_torsion_observations': float(count_values.mean()),
        'median_torsion_observations': float(count_values.median()),
        'p25_torsion_observations': float(torch.quantile(count_values, 0.25)),
        'p75_torsion_observations': float(torch.quantile(count_values, 0.75)),
        'max_torsion_observations': float(count_values.max()),
        'internal_torsion_coverage': internal_covered / max(internal_total, 1),
        'cross_ru_torsion_coverage': cross_covered / max(cross_total, 1),
        'mean_abs_torsion_bias': mean_torsion,
        'median_abs_torsion_bias': float(torsion_abs.median()),
        'p25_abs_torsion_bias': float(torch.quantile(torsion_abs, 0.25)),
        'p75_abs_torsion_bias': float(torch.quantile(torsion_abs, 0.75)),
        'p95_abs_torsion_bias': float(torch.quantile(torsion_abs, 0.95)),
        'mean_abs_angle_bias': mean_angle,
        'torsion_angle_bias_ratio': mean_torsion / (mean_angle + 1e-8),
        'internal_mean_abs_torsion_bias': (
            float(torch.cat(internal_values).mean())
            if internal_values else float('nan')
        ),
        'cross_ru_mean_abs_torsion_bias': (
            float(torch.cat(cross_values).mean())
            if cross_values else float('nan')
        ),
    }


@torch.no_grad()
def _collect_joint_basis_diagnostics(model, loader, device):
    """Measure the restored fold-best AC/RA relation-bias utilization."""
    base = model.module if hasattr(model, 'module') else model
    encoder = base.encoders['graph'].encoder
    branch = getattr(getattr(encoder, 'glt', None), 'joint_basis_bias', None)
    if branch is None:
        return None
    was_training = bool(model.training)
    model.eval()
    joint_abs, angle_abs, distances, angles, observation_bias = [], [], [], [], []
    internal_abs, cross_abs = [], []
    for batch in loader:
        batch = batch.to(device, non_blocking=True)
        views = encoder.joint_basis_views(batch)
        active = views['active_relation'].bool()
        joint = views['joint_bias'].float()
        real = views['real_relation_mask'].bool()
        angle = views['angle_bias'].float()
        if active.any():
            values = joint[active].abs().reshape(-1)
            joint_abs.append(values.cpu())
            cross = views['source_cross_ru'].bool()[active]
            per_relation = joint[active].norm(dim=-1)
            if (~cross).any():
                internal_abs.append(per_relation[~cross].cpu())
            if cross.any():
                cross_abs.append(per_relation[cross].cpu())
        real_active = active[real]
        if real_active.any():
            angle_abs.append(angle[real_active].abs().reshape(-1).cpu())
        relation_index = views['observation_relation'].long()
        if relation_index.numel():
            distances.append(views['observation_distance'].float().cpu())
            angles.append(views['observation_angle'].float().cpu())
            observation_bias.append(joint[relation_index].norm(dim=-1).cpu())
    if was_training:
        model.train()
    if not joint_abs or not distances:
        raise RuntimeError('joint-basis diagnostics found no active observations')

    def distribution(values):
        values = torch.cat(values).float()
        return {
            'mean': float(values.mean()), 'median': float(values.median()),
            'p25': float(torch.quantile(values, 0.25)),
            'p75': float(torch.quantile(values, 0.75)),
            'p95': float(torch.quantile(values, 0.95)),
            'count': int(values.numel()),
        }

    distance = torch.cat(distances).numpy()
    theta = torch.cat(angles).numpy()
    response = torch.cat(observation_bias).numpy()
    from scipy.stats import spearmanr
    distance_correlation = float(spearmanr(distance, response).statistic)
    angle_correlation = float(spearmanr(theta, response).statistic)
    quartile_edges = torch.quantile(torch.from_numpy(distance), torch.linspace(0, 1, 5))
    quartile_response = []
    distance_tensor = torch.from_numpy(distance)
    response_tensor = torch.from_numpy(response)
    for index in range(4):
        selected = ((distance_tensor >= quartile_edges[index])
                    & ((distance_tensor <= quartile_edges[index + 1]) if index == 3
                       else (distance_tensor < quartile_edges[index + 1])))
        quartile_response.append({
            'quartile': index + 1,
            'lower': float(quartile_edges[index]),
            'upper': float(quartile_edges[index + 1]),
            'count': int(selected.sum()),
            'mean_bias_norm': float(response_tensor[selected].mean()),
        })
    joint_distribution = distribution(joint_abs)
    angle_distribution = distribution(angle_abs)
    return {
        'mode': str(branch.mode),
        'new_parameter_count': int(sum(p.numel() for p in branch.parameters() if p.requires_grad)),
        'joint_bias_abs': joint_distribution,
        'existing_angle_bias_abs': angle_distribution,
        'mean_joint_to_angle_ratio': (
            joint_distribution['mean'] / max(angle_distribution['mean'], 1e-12)
        ),
        'internal_relation_bias_norm': distribution(internal_abs) if internal_abs else None,
        'cross_ru_relation_bias_norm': distribution(cross_abs) if cross_abs else None,
        'source_distance_quartiles': quartile_response,
        'spearman_source_distance_bias_norm': distance_correlation,
        'spearman_angle_bias_norm': angle_correlation,
    }


def _cohort_hash_from_current_manifest(root, dataset_name):
    """Read an existing immutable cohort pointer without rebuilding it."""

    pointer = os.path.join(
        root, "processed", "mips_trimer_scage", "cohorts",
        str(dataset_name), "current.json",
    )
    try:
        with open(pointer, encoding="utf-8") as handle:
            value = json.load(handle)
        cohort_hash = str(value.get("cohort_hash", ""))
        return cohort_hash or None
    except (OSError, ValueError, TypeError):
        return None


SUPPORTED_MODALITIES = ('graph', 'smiles', 'fp')
SUPPORTED_FUSION_TYPES = ('none', 'zero_gated_residual')
CROSS_TASK_AUXILIARY_MAP = {
    task: tuple(other for other in ('eat', 'eea', 'egb', 'egc', 'ei', 'eps', 'nc', 'xc') if other != task)
    for task in ('eat', 'eea', 'egb', 'egc', 'ei', 'eps', 'nc', 'xc')
}


class _MTSMultiTaskFoldDataset(TorchDataset):
    """Leakage-filtered, task-balanced view over the eight downstream sets."""

    is_mts_route = True

    def __init__(self, entries):
        self.entries = list(entries)

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        dataset, row, target, task_index = self.entries[int(index)]
        data = dataset[int(row)]
        data.y = torch.tensor([float(target)], dtype=torch.float)
        data.mts_task_index = torch.tensor(int(task_index), dtype=torch.long)
        return data

def _scage_checkpoint_key_compatibility(
    model_keys,
    checkpoint_keys,
    expected_stage,
    unimodal_aux_weight=0.0,
    cross_task_aux_weight=0.0,
):
    """Classify checkpoint keys under the Stage 1/Stage 2 transfer contract."""
    model_keys = set(model_keys)
    checkpoint_keys = set(checkpoint_keys)
    missing = sorted(model_keys - checkpoint_keys)
    unexpected = sorted(checkpoint_keys - model_keys)
    if expected_stage == 'mips_pretrain':
        # Graph-only downstream deliberately skips Stage 2. Only the graph
        # encoder is transferred; projection/fusion/head parameters start from
        # their deterministic Stage 3 initialization.
        allowed_missing = {
            key for key in missing
            if not key.startswith('encoders.graph.')
        }
        allowed_unexpected = set()
    else:
        # Multimodal downstream must strictly inherit Stage 2
        # encoder/projection/fusion parameters. Only downstream auxiliary heads
        # may be newly initialized.
        allowed_unexpected = {
            'alignment_mask_head.weight',
            'alignment_mask_head.bias',
        }
        allowed_missing = {
            key for key in missing
            if key.startswith('mlp.') or (
                float(unimodal_aux_weight) > 0.0
                and key.startswith('modality_heads.')
            ) or (
                float(cross_task_aux_weight) > 0.0
                and key.startswith('cross_task_aux_heads.')
            )
        }
    incompatible = [
        key for key in missing if key not in allowed_missing
    ] + [
        key for key in unexpected if key not in allowed_unexpected
    ]
    retained_unexpected = [
        key for key in unexpected if key not in allowed_unexpected
    ]
    return missing, retained_unexpected, incompatible


def select_mts_checkpoint_transfer_keys(
    model_keys, checkpoint_keys, *, allowed_missing=()
):
    """Return the exact learned MTS topology keys transferred to a fold.

    Stage-3 fine-tuning deliberately reinitializes the MD200 residual,
    graph projection/norm and regression head.  Every other common
    ``encoders.graph.encoder`` tensor is part of the learned B0-v2 topology
    state. Keeping the selection
    in one small, testable function prevents a descriptive log line from
    becoming a weaker loading contract.
    """

    model_keys = set(model_keys)
    checkpoint_keys = set(checkpoint_keys)
    prefix = "encoders.graph.encoder."
    candidates = tuple(sorted(
        key for key in model_keys
        if key.startswith(prefix) and ".md_residual." not in key
    ))
    allowed_missing = set(allowed_missing)
    invalid_allowed = allowed_missing - set(candidates)
    if invalid_allowed:
        raise RuntimeError(
            "MTS checkpoint allowed-missing set is not a topology tensor: "
            + ", ".join(sorted(invalid_allowed)[:10])
        )
    missing = tuple(
        key for key in candidates
        if key not in checkpoint_keys and key not in allowed_missing
    )
    if missing:
        raise RuntimeError(
            "MTS checkpoint missing transferable topology tensors: "
            + ", ".join(missing[:10])
        )
    return tuple(key for key in candidates if key in checkpoint_keys)


def build_mts_downstream_model(args, auxiliary_tasks=()):
    """Construct the downstream MTS UniEncoder with the exact fixed switches.

    The publish step and every fold job share this one construction so the
    published final checkpoint always matches the training-time architecture.
    """
    from src.modules import UniEncoderAttention

    model = UniEncoderAttention(
        joint_embedding_dim=args.joint_embedding_dim,
        smiles_model_name=args.smiles_model_name,
        gnn_model_name="",
        modality_list=args.modalities,
        freeze_encoder=args.freeze_encoder,
        graph_num_layers=args.graph_num_layers,
        graph_emb_dim=args.graph_emb_dim,
        graph_dropout=args.graph_dropout,
        graph_encoder_type=args.graph_encoder_type,
        scage_dist_bar=args.scage_dist_bar,
        scage_num_heads=args.scage_num_heads,
        scage_ffn_hidden_dim=args.scage_ffn_hidden_dim,
        scage_num_kernels=args.scage_num_kernels,
        scage_attention_dropout=args.scage_attention_dropout,
        scage_use_pbc_distance=args.scage_use_pbc_distance,
        scage_use_descriptors=args.scage_use_descriptors,
        scage_distance_mode=args.scage_distance_mode,
        scage_distance_rbf=args.scage_distance_rbf,
        scage_distance_cutoff=args.scage_distance_cutoff,
        scage_distance_scales=args.scage_distance_scales,
        scage_distance_taus=args.scage_distance_taus,
        scage_topology_bias=args.scage_topology_bias,
        scage_topology_max_distance=args.scage_topology_max_distance,
        scage_topology_locality_mode=args.scage_topology_locality_mode,
        scage_topology_locality_threshold=args.scage_topology_locality_threshold,
        scage_topology_locality_tau=args.scage_topology_locality_tau,
        scage_periodic_image_mode=args.scage_periodic_image_mode,
        scage_periodic_image_cap=args.scage_periodic_image_cap,
        scage_periodic_image_temperature=args.scage_periodic_image_temperature,
        scage_force_topology_only=args.scage_force_topology_only,
        mips_core=args.mips_core,
        mips_max_hops=args.mips_max_hops,
        mips_use_descriptors=args.mips_use_descriptors,
        spatial_mode=args.spatial_mode,
        graph_geometry_mode=args.graph_geometry_mode,
        trimer_num_candidates=args.trimer_num_candidates,
        trimer_max_heavy_atoms=args.trimer_max_heavy_atoms,
        mips_variant=args.mips_variant,
        mips_fusion_mode=args.mips_fusion_mode,
        projection_mode=args.projection_mode,
        modality_control=args.modality_control,
        controlled_modality=args.controlled_modality,
        mips_atom_feature_mode=args.mips_atom_feature_mode,
        mips_attention_scale=args.mips_attention_scale,
        mips_norm_mode=args.mips_norm_mode,
        mips_activation=args.mips_activation,
        mips_spd_bias_mode=args.mips_spd_bias_mode,
        mips_path_bias_mode=args.mips_path_bias_mode,
        mips_multi_scale_hop_gate=args.mips_multi_scale_hop_gate,
        mips_semantics=args.mips_semantics,
        mips_descriptor_fusion_mode=args.mips_descriptor_fusion_mode,
        mips_descriptor_components=args.mips_descriptor_components,
        mips_descriptor_disturbance=args.mips_descriptor_disturbance,
        mips_backbone_mode=args.mips_backbone_mode,
        mips_input_norm=args.mips_input_norm,
        mips_mask_mode=args.mips_mask_mode,
        mips_mask_policy=args.mips_mask_policy,
        mips_masked_loss_reduction=args.mips_masked_loss_reduction,
        topology_attention_variant=args.topology_attention_variant,
        star_rbf_upper=args.star_rbf_upper,
        use_star_rbf=bool(getattr(args, 'use_star_rbf', True)),
        use_mcl=bool(getattr(args, 'use_mcl', False)),
        fusion_type=args.fusion_type,
        fp_mode=args.fp_mode,
        fusion_dropout=args.fusion_dropout,
        head_dropout=args.head_dropout,
        unimodal_auxiliary=args.unimodal_aux_weight > 0,
        cross_task_auxiliary_tasks=auxiliary_tasks,
        fp_bit_dropout=args.fp_bit_dropout,
        modality_dropout={
            'fp': args.fp_modality_dropout,
            'smiles': args.smiles_modality_dropout,
            'graph': args.graph_modality_dropout,
        },
    )
    glt_mode = str(getattr(args, "mts_glt_mode", "none"))
    if glt_mode != "none":
        if tuple(args.modalities) != ("graph",):
            raise ValueError("MTS-GLT downstream supports graph-only mode")
        if bool(getattr(args, "use_star_rbf", False)):
            raise ValueError("MTS-GLT downstream requires Star-RBF off")
        glt_version = str(getattr(args, "mts_glt_version", "v1"))
        if glt_version == "graphgate_v1":
            if glt_mode not in {
                "o8_only", "o8_glt_graph", "o8_glt_graph_mean",
                "o8_glt_atom_central",
            }:
                raise ValueError("unsupported GraphGate downstream readout")
            from src.modules import MTSGraphGateModel
            encoder = MTSGraphGateModel(layers=6)
            encoder.glt_geometry_mode = str(args.mts_glt_geometry_mode)
        elif glt_version == "v2":
            from src.modules import MTSGraphLineModelV2
            interaction_mode = {
                "o8_glt_atom_self3d": "self3d",
                "o8_glt_atom_x23": "x23",
            }.get(glt_mode, "none")
            line_conditioning_mode = {
                "o8_glt_atom_line_self": "self",
                "o8_glt_atom_line_x2l": "x2l",
            }.get(glt_mode, "none")
            attention_conditioning_mode = {
                "o8_glt_atom_attn_self": "self",
                "o8_glt_atom_attn_x2a": "x2a",
            }.get(glt_mode, "none")
            torsion_mode = {
                "o8_glt_atom_torsion_count": "count",
                "o8_glt_atom_torsion": "full",
            }.get(glt_mode, "none")
            joint_basis_mode = {
                "o8_glt_atom_sbf_angle_control": "control",
                "o8_glt_atom_sbf_radial_angle": "radial",
            }.get(glt_mode, "none")
            encoder = MTSGraphLineModelV2(
                glt_layers=int(args.mts_glt_layers),
                glt_attention_variant=str(args.mts_glt_attention_variant),
                use_compact19=bool(args.mts_glt_use_compact19),
                interaction_mode=interaction_mode,
                line_conditioning_mode=line_conditioning_mode,
                attention_conditioning_mode=attention_conditioning_mode,
                torsion_mode=torsion_mode,
                joint_basis_mode=joint_basis_mode,
                glt_metadata_mode=str(
                    getattr(args, "mts_glt_metadata_mode", "full")
                ),
                o8_bond_bias_mode=str(args.mts_o8_bond_bias_mode),
                use_spatial_contact=(glt_mode == "o8_glt_atom_spatial"),
                spatial_shell_mode=str(args.mts_spatial_shell_mode),
            )
        else:
            from src.modules import MTSGraphLineModel
            encoder = MTSGraphLineModel()
        encoder.downstream_mode = glt_mode
        model.encoders["graph"].encoder = encoder
    return model


def select_mts_glt_graph_state(model_state, checkpoint_state, *, graphgate=False):
    """Map a GLT pretrain container into the exact downstream graph module."""
    graph_prefix = 'encoders.graph.encoder.'
    mapped = {
        graph_prefix + str(key)[len('model.'):]: value
        for key, value in checkpoint_state.items()
        if str(key).startswith('model.')
    }
    expected = {key for key in model_state if key.startswith(graph_prefix)}
    optionally_pretrained_downstream = {
        key for key in expected
        if (
            key.startswith(graph_prefix + 'interaction_update.')
            or key == graph_prefix + 'spatial_channel_gate'
            or key == graph_prefix + 'atom_channel_gate'
            or key.startswith(graph_prefix + 'atom_fusion_norm.')
            or key.startswith(graph_prefix + 'atom_fusion_projection.')
            or key.startswith(graph_prefix + 'compact19_residual.')
            or key.startswith(graph_prefix + 'o8.md_residual.')
            or key.startswith(graph_prefix + 'o8.direct_bond_bias.')
            or key.startswith(graph_prefix + 'glt.line_conditioning_projection.')
            or key.startswith(graph_prefix + 'glt.attention_conditioning_projection.')
            or key.startswith(graph_prefix + 'glt.torsion_bias.')
            or key.startswith(graph_prefix + 'glt.joint_basis_bias.')
        )
    }
    # Historical probes contain several frozen downstream tensors. Preserve
    # their existing load behavior, while allowing new probes to omit them.
    expected -= {
        key for key in optionally_pretrained_downstream if key not in mapped
    }
    if graphgate:
        expected = {
            key for key in expected
            if (
                key.startswith(graph_prefix + "o8_encoder.")
                and not key.startswith(graph_prefix + "o8_encoder.md_residual.")
                and not key.startswith(graph_prefix + "o8_encoder.star_distance_bias.")
            )
            or key.startswith(graph_prefix + "glt_line_encoder.")
        }
    if set(mapped) != expected:
        missing = sorted(expected - set(mapped))
        unexpected = sorted(set(mapped) - expected)
        raise RuntimeError(
            'MTS-GLT graph checkpoint mismatch; missing='
            + ','.join(missing[:10]) + ' unexpected='
            + ','.join(unexpected[:10])
        )
    return mapped


def run_finetune_job(config=None, task=None, seed=None, fold=None):
    """Run exactly one task/seed/fold through the real MTS training path."""
    args = parse_arguments() if config is None else config
    if task is not None:
        args.tasks = [str(task)]
    if seed is not None:
        args.seed = int(seed)
    if fold is not None:
        args.fold_ids = [int(fold)]
    validate_mips_trimer_runtime(args)
    from src.dataset import UniDataset
    from src.modules import UniEncoderAttention
    from src.utils import (
        TargetScaler, fit_fixed_epochs, get_data_loader, scale_targets,
        set_global_seed, test_model, train_and_evaluate,
    )
    # Frozen cache readers perform only their local schema/shape/index checks.
    # The old full-bundle hash audit was a separate offline concern and is no
    # longer part of a Finetune startup.
    # Ignore warnings
    warnings.filterwarnings("ignore")

    pre_trained_model_dict = {
        'smiles_model_name': args.smiles_model_name,
        'gnn_model_name': "",
    }

    result_output_dir = args.results_dir
    model_output_dir = args.models_dir
    model_modality_list = args.modalities

    task_list = args.tasks
    dataset_task_list = list(task_list)
    selected_cross_task_aux = set(args.cross_task_aux_tasks)
    if float(args.cross_task_aux_weight) > 0:
        for task in task_list:
            if selected_cross_task_aux and task not in selected_cross_task_aux:
                continue
            for auxiliary_task in CROSS_TASK_AUXILIARY_MAP.get(task, ()):
                if auxiliary_task not in dataset_task_list:
                    dataset_task_list.append(auxiliary_task)
    dataset_name_list = ['smi_' + task for task in dataset_task_list]
    dataset_list = [
        UniDataset(
            root=args.root,
            dataset=dataset_name,
            smiles_model_name=pre_trained_model_dict['smiles_model_name'],
            graph_encoder_type=args.graph_encoder_type,
            graph_input=args.graph_input,
            use_feature_cache=not args.disable_feature_cache,
            feature_source_dataset=args.feature_source_dataset,
            rebuild_feature_cache=args.rebuild_feature_cache,
            max_smiles_length=args.max_smiles_length,
            max_smiles_length_cap=args.max_smiles_length_cap,
            fp_mode=args.fp_mode,
            feature_cache_workers=args.feature_cache_workers,
            feature_cache_chunksize=args.feature_cache_chunksize,
            feature_cache_partial_every=args.feature_cache_partial_every,
            feature_cache_item_timeout=args.feature_cache_item_timeout,
            cache_layers=args.cache_layers,
            cache_validate=args.cache_validate,
            cache_commit_size=args.cache_commit_size,
            embed_tries_multiplier=args.embed_tries_multiplier,
            conformer_3d_count=args.conformer_3d_count,
            conformer_keep_count=args.conformer_keep_count,
            conformer_profile=args.conformer_profile,
            scage_distance_mode=args.scage_distance_mode,
            scage_distance_rbf=args.scage_distance_rbf,
            scage_distance_cutoff=args.scage_distance_cutoff,
            mips_core=args.mips_core,
            mips_max_hops=args.mips_max_hops,
            mips_use_descriptors=args.mips_use_descriptors,
            mips_descriptor_protocol=args.mips_descriptor_protocol,
            spatial_mode=args.spatial_mode,
            graph_geometry_mode=args.graph_geometry_mode,
            topology_representation=args.topology_representation,
            trimer_num_candidates=args.trimer_num_candidates,
            trimer_max_heavy_atoms=args.trimer_max_heavy_atoms,
            mips_variant=args.mips_variant,
            finite_variant=args.finite_variant,
            conformer_mode=args.conformer_mode,
            field_layout=args.field_layout,
            field_channels=args.field_channels,
            experiment_id=args.experiment_id,
            feature_config_hash="manual",
            modalities=args.modalities,
            star_rbf_v2_sidecar=args.star_rbf_v2_sidecar,
            periodic_line_glt_sidecar=args.periodic_line_glt_sidecar,
            periodic_spatial_contact_sidecar=(
                args.periodic_spatial_contact_sidecar
                if str(args.mts_glt_mode) == "o8_glt_atom_spatial" else None
            ),
            periodic_line_torsion=str(args.mts_glt_mode) in {
                'o8_glt_atom_torsion_count', 'o8_glt_atom_torsion',
            },
            periodic_line_joint_ra=str(args.mts_glt_mode) in {
                'o8_glt_atom_sbf_angle_control',
                'o8_glt_atom_sbf_radial_angle',
            },
        )
        for dataset_name in dataset_name_list
    ]
    dataset_by_task = dict(zip(dataset_task_list, dataset_list))
    raw_targets_by_task = {
        task: np.asarray(dataset.raw_targets, dtype=np.float64)
        for task, dataset in dataset_by_task.items()
    }
    if args.graph_encoder_type == "mips_trimer_scage" and not args.cache_only:
        cache_dataset = dataset_list[0]
        original_layers = cache_dataset.cache_layers
        cache_dataset.cache_layers = ("ru_base", "topology", "trimer", "md200")
        specs = cache_dataset._lmdb_cache_specs({})
        cache_dataset.cache_layers = original_layers
        unfrozen = [
            name for name, spec in specs.items()
            if not os.path.isfile(os.path.join(spec["root"], ".frozen"))
        ]
        if unfrozen:
            raise RuntimeError(
                f"{MTS_ROUTE_NAME} training requires frozen cache artifacts: "
                + ", ".join(unfrozen)
            )
    if args.cache_only:
        print(
            "Downstream feature-cache prebuild complete: "
            + ", ".join(
                f"{task}={len(dataset_by_task[task])}"
                for task in dataset_task_list
            )
        )
        return
    raw_target_maps = {}
    for task, dataset in dataset_by_task.items():
        target_map = {}
        for data, value in zip(dataset, raw_targets_by_task[task]):
            smiles = str(data.smiles)
            if smiles in target_map and not np.isclose(target_map[smiles], value):
                raise ValueError(f'Conflicting duplicate labels for {task}: {smiles}')
            target_map[smiles] = float(value)
        raw_target_maps[task] = target_map

    def configure_cross_task_targets(task, dataset, training_indices, scope):
        auxiliary_tasks = (
            CROSS_TASK_AUXILIARY_MAP.get(task, ())
            if (
                float(args.cross_task_aux_weight) > 0
                and (
                    not selected_cross_task_aux
                    or task in selected_cross_task_aux
                )
            ) else ()
        )
        auxiliary_tasks = tuple(
            name for name in auxiliary_tasks if name in raw_target_maps
        )
        aux_values = np.zeros(
            (len(dataset), len(auxiliary_tasks)), dtype=np.float32
        )
        aux_masks = np.zeros_like(aux_values, dtype=bool)
        for aux_idx, auxiliary_task in enumerate(auxiliary_tasks):
            auxiliary_map = raw_target_maps[auxiliary_task]
            matched_train = [
                int(index) for index in training_indices
                if str(dataset[int(index)].smiles) in auxiliary_map
            ]
            if not matched_train:
                continue
            auxiliary_values = np.array([
                auxiliary_map[str(dataset[index].smiles)]
                for index in matched_train
            ], dtype=np.float64)
            auxiliary_scaler = TargetScaler(
                auxiliary_task,
                StandardScaler(),
                transform_mode=args.target_transform,
            )
            auxiliary_scaler.scaler.fit(
                auxiliary_scaler._pre_transform(
                    auxiliary_values.reshape(-1, 1)
                )
            )
            scaled_values = auxiliary_scaler.transform(
                auxiliary_values.reshape(-1, 1)
            ).reshape(-1)
            for index, value in zip(matched_train, scaled_values):
                aux_values[index, aux_idx] = float(value)
                aux_masks[index, aux_idx] = True
            print(
                f'Cross-task auxiliary {task} <- {auxiliary_task}: '
                f'{len(matched_train)}/{len(training_indices)} {scope} labels; '
                'all samples outside that training partition excluded'
            )
        dataset.set_auxiliary_target_overrides(aux_values, aux_masks)
        return auxiliary_tasks

    def build_pcgrad_fold(task, fold_train_indices, test_indices, ordered_smiles, fold_seed):
        """Build an eight-task balanced training view without target-fold leakage."""
        task_order = (task,) + tuple(
            name for name in dataset_task_list if name != task
        )
        held_out_keys = {
            sample_key_from_smiles(ordered_smiles[int(index)])
            for index in test_indices
        }
        entries = []
        counts = []
        for task_index, name in enumerate(task_order):
            source = dataset_by_task[name]
            source_smiles = list(getattr(source, "_row_smiles", ()))
            if len(source_smiles) != len(source):
                raise RuntimeError(f"missing immutable source-row SMILES for {name}")
            if name == task:
                allowed = [int(index) for index in fold_train_indices]
            else:
                allowed = [
                    index for index, smiles in enumerate(source_smiles)
                    if sample_key_from_smiles(smiles) not in held_out_keys
                ]
            if not allowed:
                raise RuntimeError(f"no leakage-safe multitask rows remain for {name}")
            values = raw_targets_by_task[name][allowed]
            task_scaler = TargetScaler(
                name, StandardScaler(), transform_mode=args.target_transform
            )
            task_scaler.scaler.fit(task_scaler._pre_transform(values.reshape(-1, 1)))
            scaled = task_scaler.transform(values.reshape(-1, 1)).reshape(-1)
            entries.extend(
                (source, row, value, task_index)
                for row, value in zip(allowed, scaled)
            )
            counts.append(len(allowed))
        multitask_dataset = _MTSMultiTaskFoldDataset(entries)
        weights = []
        cursor = 0
        for count in counts:
            weights.extend([1.0 / float(count)] * count)
            cursor += count
        generator = torch.Generator().manual_seed(int(fold_seed))
        sampler = WeightedRandomSampler(
            torch.as_tensor(weights, dtype=torch.double),
            num_samples=max(counts) * len(counts), replacement=True,
            generator=generator,
        )
        return multitask_dataset, sampler, task_order[1:]

    # Feature-cache workers must be created before this process initializes
    # CUDA. Cache workers are CPU-only and PolyGen's CPU optimizer must not
    # inherit the downstream training CUDA context.
    set_global_seed(args.seed)
    if args.graph_encoder_type == "mips_trimer_scage" and not torch.cuda.is_available():
        raise RuntimeError(f"{MTS_ROUTE_NAME} Stage 3 requires CUDA")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    freeze_encoder = args.freeze_encoder
    pretrained_model_path = args.pretrained_model_path
    epochs = args.epochs
    patience = args.patience
    t1_init_artifact = False
    checkpoint_meta = {}

    for task in task_list:
        print(f"\nStarting task: {task}")
        dataset = dataset_by_task[task]
        raw_targets = raw_targets_by_task[task]
        # ``--results_dir`` historically had two meanings.  Treat an
        # existing directory (or a path without a CSV suffix) as a result
        # root and give every task its own file; this prevents concurrent
        # Stage-3 workers from writing the same CSV.
        result_root = Path(result_output_dir)
        if result_root.exists() and result_root.is_dir():
            task_result_output = result_root / f"{task}.csv"
        elif len(task_list) > 1 and result_root.suffix.lower() != ".csv":
            task_result_output = result_root / f"{task}.csv"
        else:
            task_result_output = result_root
        task_result_output = str(task_result_output)
        task_result_file_initialized = False

        print("Start 5-fold Cross Validation")
        if args.graph_encoder_type == 'mips_trimer_scage':
            manifest_path = Path(args.split_manifest_dir) / f"{task}.json"
            if not manifest_path.is_file():
                raise RuntimeError(
                    f"Missing fixed split manifest: {manifest_path}. Run "
                    "scripts/create_mips_split_manifests.py first."
                )
            manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
            # The manifest indexes are defined over the task CSV rows.  Do
            # not hash ``data.smiles`` from the feature store here: content
            # keyed caches may return a representative P-SMILES for a
            # chemically equivalent row (the source task row can therefore
            # differ while resolving to the same feature key).  Such a
            # representative string is not the split order and made valid
            # manifests fail spuriously.  The Dataset preserves the CSV row
            # order, so bind the fixed folds to that immutable source order.
            task_csv = Path(args.root) / "raw" / f"smi_{task}.csv"
            if not task_csv.is_file():
                raise RuntimeError(
                    f"Task CSV required for fixed split validation is missing: "
                    f"{task_csv}"
                )
            ordered_smiles = (
                pd.read_csv(task_csv, usecols=[0])
                .iloc[:, 0]
                .astype(str)
                .str.strip()
                .tolist()
            )
            if (
                manifest.get("schema")
                != "mips-shared-validation-test-fold-v1"
                or int(manifest.get("sample_count", -1)) != len(dataset)
                or not bool(manifest.get("validation_is_test", False))
            ):
                raise RuntimeError(
                    f"Fixed split manifest does not match task cohort: {manifest_path}"
                )
            splits = [
                (
                    np.asarray(item["train_indices"], dtype=np.int64),
                    np.asarray(item["test_indices"], dtype=np.int64),
                )
                for item in manifest["folds"]
            ]
        else:
            splits = list(
                KFold(n_splits=5, shuffle=True, random_state=1).split(
                    np.arange(len(dataset))
                )
            )
        fold_metrics = []
        best_fold_val_r2 = -float('inf')
        best_model_state = None

        selected_folds = set(args.fold_ids)
        if not selected_folds or any(fold < 0 or fold >= 5 for fold in selected_folds):
            raise ValueError("--fold_ids must contain one or more values from 0 to 4")
        validated_mode_batch = False
        for fold, (train_indices, test_indices) in enumerate(splits):
            if fold not in selected_folds:
                continue
            fold_started = time.monotonic()
            print(f"\nFold {fold + 1}")
            task_offset = sum((idx + 1) * ord(char) for idx, char in enumerate(task))
            fold_seed = int(args.seed) + 1009 * task_offset + fold
            set_global_seed(fold_seed)
            print(f"Fold seed: {fold_seed}")
            if args.evaluation_protocol == 'nested5':
                ranked = sorted(
                    (hashlib.sha256(sample_key_from_smiles(ordered_smiles[int(index)])).digest(), int(index))
                    for index in train_indices
                )
                inner_count = max(1, int(round(0.10 * len(ranked))))
                val_set = {index for _, index in ranked[:inner_count]}
                val_indices = np.asarray(sorted(val_set), dtype=np.int64)
                fold_train_indices = np.asarray(
                    [int(index) for index in train_indices if int(index) not in val_set],
                    dtype=np.int64,
                )
                print("Nested 5-fold protocol: outer test is excluded from model selection")
            else:
                fold_train_indices = train_indices
                val_indices = test_indices
                print("Shared 5-fold protocol: validation and test use the same held-out fold")
            print(
                f"Fold partitions: train={len(fold_train_indices)}, "
                f"validation={len(val_indices)}, test={len(test_indices)}"
            )
            scaler = scale_targets(
                dataset,
                task,
                train_indices=fold_train_indices,
                raw_targets=raw_targets,
                transform_mode=args.target_transform,
            )
            if args.finetune_mode == 'multitask_pcgrad':
                dataset.clear_auxiliary_target_overrides()
                multitask_dataset, multitask_sampler, auxiliary_tasks = build_pcgrad_fold(
                    task, fold_train_indices, test_indices, ordered_smiles, fold_seed
                )
                train_loader = get_data_loader(
                    multitask_dataset,
                    batch_size=args.batch_size,
                    shuffle=False,
                    sampler=multitask_sampler,
                    drop_last=False,
                    num_workers=args.loader_workers,
                    pin_memory=True,
                    persistent_workers=args.loader_workers > 0,
                )
            else:
                auxiliary_tasks = configure_cross_task_targets(
                    task, dataset, fold_train_indices, 'fold-train'
                )
                train_loader = get_data_loader(
                    dataset,
                    indices=fold_train_indices,
                    batch_size=args.batch_size,
                    shuffle=True,
                    drop_last=False,
                    num_workers=args.loader_workers,
                    pin_memory=True,
                    persistent_workers=args.loader_workers > 0,
                )
            test_loader = get_data_loader(
                dataset,
                indices=test_indices,
                batch_size=args.eval_batch_size,
                shuffle=False,
                drop_last=False,
                num_workers=args.loader_workers,
                pin_memory=True,
                persistent_workers=args.loader_workers > 0,
            )
            val_loader = get_data_loader(
                dataset,
                indices=val_indices,
                batch_size=args.eval_batch_size,
                shuffle=False,
                drop_last=False,
                num_workers=args.loader_workers,
                pin_memory=True,
                persistent_workers=args.loader_workers > 0,
            )
            if not validated_mode_batch and str(args.mts_glt_mode) != "none":
                validation_batch = next(iter(val_loader))
                missing = missing_batch_fields(args.mts_glt_mode, validation_batch)
                if missing:
                    raise RuntimeError(
                        f"MTS mode {args.mts_glt_mode} missing required batch fields: "
                        + ", ".join(missing)
                    )
                validated_mode_batch = True

            model = build_mts_downstream_model(
                args, auxiliary_tasks=auxiliary_tasks,
            )
            if (
                args.graph_encoder_type == MTS_ROUTE_INTERNAL
                and bool(getattr(args, "use_star_rbf", False))
                and getattr(args, "star_rbf_upper", None) is not None
            ):
                star_bias = model.encoders["graph"].encoder.star_distance_bias
                sidecar = getattr(dataset, "_star_rbf_v2_sidecar", None)
                if sidecar is not None and abs(
                    float(sidecar.rbf_upper) - float(args.star_rbf_upper)
                ) > 1e-9:
                    raise RuntimeError(
                        "MTS RBF upper mismatch: sidecar="
                        f"{float(sidecar.rbf_upper)}, "
                        f"config={float(args.star_rbf_upper)}"
                    )
                centers = star_bias.centers
                if (
                    int(centers.numel()) != 32
                    or abs(float(centers[0])) > 1e-9
                    or abs(float(centers[-1]) - float(args.star_rbf_upper)) > 1e-6
                ):
                    raise RuntimeError(
                        "MTS RBF definition mismatch: expected 32 centers in "
                        f"[0.0, {args.star_rbf_upper}], got num={centers.numel()} "
                        f"first={float(centers[0])} last={float(centers[-1])}"
                    )
                rbf_spacing = float(centers[1] - centers[0])
                if abs(
                    float(star_bias.gamma)
                    - 0.5 / max(rbf_spacing * rbf_spacing, 1e-12)
                ) > 1e-4:
                    raise RuntimeError(
                        "MTS RBF gamma must be 0.5/spacing^2 from the actual "
                        f"centers, got gamma={float(star_bias.gamma)} "
                        f"spacing={rbf_spacing}"
                    )
            if pretrained_model_path:
                checkpoint = torch.load(pretrained_model_path, map_location='cpu')
                graphgate_checkpoint = (
                    str(getattr(args, 'mts_glt_version', 'v1')) == 'graphgate_v1'
                )
                if args.graph_encoder_type == 'mips_trimer_scage':
                    # Metadata is historical provenance only.  The active
                    # compatibility contract is the real state-dict key/shape
                    # selection below followed by strict=True loading.  This
                    # applies equally to metadata-rich checkpoints and the
                    # minimal final ``{"state_dict": ...}`` payload.
                    if not isinstance(checkpoint, dict):
                        raise RuntimeError("MTS checkpoint must be a mapping")
                    if graphgate_checkpoint:
                        if checkpoint.get("schema") != "mts-glt-graphgate-v1-probe-v1":
                            raise RuntimeError("GraphGate downstream requires a GraphGate probe checkpoint")
                        namespaces = checkpoint.get("namespaces")
                        if not isinstance(namespaces, dict):
                            raise RuntimeError("GraphGate checkpoint namespaces are missing")
                        checkpoint_state = {}
                        for key, value in namespaces.get("o8_encoder", {}).items():
                            checkpoint_state["model.o8_encoder." + str(key)] = value
                        for key, value in namespaces.get("glt_line_encoder", {}).items():
                            checkpoint_state["model.glt_line_encoder." + str(key)] = value
                        for key, value in namespaces.get("query_pool", {}).items():
                            checkpoint_state["model.glt_line_encoder.query_pool." + str(key)] = value
                    else:
                        checkpoint_state = checkpoint.get("state_dict", checkpoint)
                    if not isinstance(checkpoint_state, dict):
                        raise RuntimeError("MTS checkpoint does not contain a state_dict mapping")
                    checkpoint_meta = (
                        checkpoint.get("meta", {})
                        if isinstance(checkpoint.get("meta"), dict) else {}
                    )
                    expected_stage = MTS_STAGE1_ID
                    if not graphgate_checkpoint:
                        checkpoint_state = checkpoint['state_dict']
                    # B0's DDP-visible pretraining container stores the
                    # UniEncoder under ``model.`` and keeps its two pretext
                    # heads alongside it.  Downstream consumes only the
                    # strict UniEncoder state; discard heads and normalize
                    # the prefix before the existing compatibility audit.
                    if (
                        str(getattr(args, 'mts_glt_mode', 'none')) == 'none'
                        and checkpoint_state and any(
                            str(key).startswith("model.")
                            for key in checkpoint_state
                        )
                    ):
                        checkpoint_state = {
                            str(key)[len("model."):]: value
                            for key, value in checkpoint_state.items()
                            if str(key).startswith("model.")
                        }
                else:
                    checkpoint_state = checkpoint.get('state_dict', checkpoint) if isinstance(checkpoint, dict) else checkpoint
                model_keys = set(model.state_dict())
                checkpoint_keys = set(checkpoint_state)
                missing = sorted(model_keys - checkpoint_keys)
                unexpected = sorted(checkpoint_keys - model_keys)
                allowed_transfer_missing = set()
                if (
                    args.graph_encoder_type == 'mips_trimer_scage'
                    and str(getattr(args, 'mts_glt_mode', 'none')) == 'none'
                ):
                    missing, unexpected, incompatible = (
                        _scage_checkpoint_key_compatibility(
                            model_keys,
                            checkpoint_keys,
                            expected_stage,
                            unimodal_aux_weight=args.unimodal_aux_weight,
                            cross_task_aux_weight=args.cross_task_aux_weight,
                        )
                    )
                    if incompatible:
                        raise RuntimeError(
                            f"{args.graph_encoder_type.upper()} {expected_stage} checkpoint mismatch. "
                            "Re-run MTS Joint Pretraining "
                            "with the same run.sh model configuration. Mismatched keys: "
                            + ", ".join(incompatible[:10])
                        )
                merged_state = model.state_dict()
                if (
                    args.graph_encoder_type == 'mips_trimer_scage'
                    and str(getattr(args, 'mts_glt_mode', 'none')) != 'none'
                ):
                    glt_state = select_mts_glt_graph_state(
                        merged_state, checkpoint_state,
                        graphgate=graphgate_checkpoint,
                    )
                    merged_state.update(glt_state)
                    missing = sorted(set(merged_state) - set(glt_state))
                    unexpected = []
                    print(
                        f'MTS-GLT checkpoint load: strictly loaded '
                        f'{len(glt_state)} graph tensors for {args.mts_glt_mode}.'
                    )
                elif args.graph_encoder_type == 'mips_trimer_scage':
                    # Joint pretraining intentionally exports a complete model
                    # container, but Stage 3 migrates only learned structural
                    # modules.  MD200, graph norm/projection and regression
                    # head remain at their fold-seeded initialization.
                    transfer_keys = select_mts_checkpoint_transfer_keys(
                        merged_state,
                        checkpoint_state,
                        allowed_missing=allowed_transfer_missing,
                    )
                    transferred = {
                        key: checkpoint_state[key] for key in transfer_keys
                    }
                    merged_state.update(transferred)
                    print(
                        'B0-v2 checkpoint load: loaded O8/Star topology encoder '
                        f'({len(transferred)} tensors); '
                        'MD200/projection/head reinitialized by fold seed.'
                    )
                else:
                    merged_state.update({
                        key: value for key, value in checkpoint_state.items()
                        if key in merged_state
                    })
                model.load_state_dict(merged_state, strict=True)
                print(f"Loaded pretrained model from {pretrained_model_path}")
                print(f"Checkpoint load: {len(missing)} missing keys, {len(unexpected)} unexpected keys")
                if str(args.mts_glt_fusion_strategy) == 'fusion_warm':
                    from src.utils import initialize_mts_glt_fusion_warm
                    observed_alpha = initialize_mts_glt_fusion_warm(
                        model, args.mts_glt_initial_alpha
                    )
                    print(
                        'Initialized MTS-GLT FusionWarm after strict checkpoint '
                        f'load: alpha={observed_alpha:.9f}'
                    )
            initial_model_state = _capture_initial_model_state(
                model, bool(args.refit_full_train)
            )

            model.to(device)
            print("Using GPU for model training." if torch.cuda.is_available() else "Using CPU for model training.")

            fold_context = SingleFoldContext(
                task=task,
                fold=int(fold),
                seed=int(fold_seed),
                experiment_id=str(args.experiment_id),
            )
            metrics = run_single_fold(
                train_and_evaluate,
                model, scaler, train_loader, val_loader, test_loader,
                device, num_epochs=epochs, patience=patience, max_grad_norm=args.max_grad_norm,
                smiles_lr=args.smiles_lr, graph_lr=args.graph_lr,
                geom_lr=args.geom_lr, fp_lr=args.fp_lr, fusion_lr=args.fusion_lr,
                head_lr=args.head_lr,
                weight_decay=args.weight_decay, warmup_epochs=args.warmup_epochs,
                freeze_smiles_epochs=args.freeze_smiles_epochs,
                deep_unfreeze_epoch=args.deep_unfreeze_epoch,
                fp_unfreeze_epoch=args.fp_unfreeze_epoch,
                regression_loss=args.regression_loss,
                huber_beta=args.huber_beta,
                unimodal_aux_weight=args.unimodal_aux_weight,
                fusion_prior_kl_weight=args.fusion_prior_kl_weight,
                fusion_prior=args.fusion_prior,
                cross_task_aux_weight=args.cross_task_aux_weight,
                swa_start_epoch=args.swa_start_epoch,
                evaluate_test=not args.refit_full_train,
                return_predictions=not args.refit_full_train,
                mts_o8_lr=args.mts_o8_lr,
                mts_geometry_lr=args.mts_geometry_lr,
                mts_adapter_lr=args.mts_adapter_lr,
                pcgrad=args.finetune_mode == 'multitask_pcgrad',
                amp_dtype=args.amp_dtype,
                mts_glt_postmortem=bool(
                    args.mts_glt_postmortem_dir
                    or args.mts_glt_fusion_warm_dir
                ),
                mts_glt_fusion_strategy=args.mts_glt_fusion_strategy,
                mts_glt_fusion_warm_epochs=args.mts_glt_fusion_warm_epochs,
                mts_glt_initial_alpha=args.mts_glt_initial_alpha,
                mts_glt_fusion_stage2_trainability=(
                    args.mts_glt_fusion_stage2_trainability
                ),
                context=fold_context,
            )
            interaction_diagnostics = _collect_interaction_diagnostics(
                model, test_loader, device
            )
            line_diagnostics = _collect_line_conditioning_diagnostics(
                model, test_loader, device
            )
            attention_diagnostics = _collect_attention_routing_diagnostics(
                model, test_loader, device
            )
            torsion_diagnostics = _collect_torsion_diagnostics(
                model, test_loader, device
            )
            joint_basis_diagnostics = _collect_joint_basis_diagnostics(
                model, test_loader, device
            )
            bond_bias_diagnostics = _collect_o8_bond_bias_diagnostics(
                model, test_loader, device
            )
            if bool(args.save_best_checkpoint):
                if not str(args.best_checkpoint_dir):
                    raise ValueError(
                        '--best_checkpoint_dir is required when '
                        '--save_best_checkpoint is enabled'
                    )
                best_path = (
                    Path(args.best_checkpoint_dir) / str(task)
                    / f'fold_{int(fold)}.pth'
                )
                _atomic_save_best_checkpoint(model, best_path, {
                    'schema': 'mts-glt-v2-downstream-fold-best-v1',
                    'task': str(task), 'fold': int(fold),
                    'seed': int(args.seed), 'fold_seed': int(fold_seed),
                    'mts_glt_mode': str(args.mts_glt_mode),
                    'mts_o8_bond_bias_mode': str(args.mts_o8_bond_bias_mode),
                    'pretrained_checkpoint': str(pretrained_model_path),
                    'best_epoch': int(metrics.get('best_epoch', -1)),
                    'test_r2': float(metrics.get('test_r2', float('nan'))),
                })
            if bond_bias_diagnostics is not None:
                if not str(args.mts_o8_bond_diagnostics_dir):
                    raise ValueError(
                        'O8 bond-bias modes require '
                        '--mts_o8_bond_diagnostics_dir'
                    )
                bond_bias_diagnostics.update({
                    'task': str(task), 'fold': int(fold),
                    'seed': int(args.seed), 'fold_seed': int(fold_seed),
                    'mts_glt_mode': str(args.mts_glt_mode),
                    'mts_o8_bond_bias_mode': str(args.mts_o8_bond_bias_mode),
                    'best_epoch': int(metrics.get('best_epoch', -1)),
                    'test_r2': float(metrics.get('test_r2', float('nan'))),
                })
                unit_path = (
                    Path(args.mts_o8_bond_diagnostics_dir)
                    / str(task) / f'fold_{int(fold)}.json'
                )
                unit_path.parent.mkdir(parents=True, exist_ok=True)
                unit_tmp = unit_path.with_name(
                    unit_path.name + f'.tmp.{os.getpid()}'
                )
                try:
                    unit_tmp.write_text(
                        json.dumps(
                            bond_bias_diagnostics, indent=2, sort_keys=True
                        ) + '\n', encoding='utf-8'
                    )
                    os.replace(unit_tmp, unit_path)
                finally:
                    if unit_tmp.exists():
                        unit_tmp.unlink()
            if interaction_diagnostics is not None:
                if not str(args.mts_glt_interaction_diagnostics_dir):
                    raise ValueError(
                        'interaction modes require '
                        '--mts_glt_interaction_diagnostics_dir'
                    )
                interaction_diagnostics.update({
                    'task': str(task), 'fold': int(fold),
                    'seed': int(args.seed), 'fold_seed': int(fold_seed),
                    'mts_glt_mode': str(args.mts_glt_mode),
                    'best_epoch': int(metrics.get('best_epoch', -1)),
                    'test_r2': float(metrics.get('test_r2', float('nan'))),
                })
                unit_path = (
                    Path(args.mts_glt_interaction_diagnostics_dir)
                    / str(task) / f'fold_{int(fold)}.json'
                )
                unit_path.parent.mkdir(parents=True, exist_ok=True)
                unit_tmp = unit_path.with_name(
                    unit_path.name + f'.tmp.{os.getpid()}'
                )
                try:
                    unit_tmp.write_text(
                        json.dumps(interaction_diagnostics, indent=2, sort_keys=True)
                        + '\n', encoding='utf-8'
                    )
                    os.replace(unit_tmp, unit_path)
                finally:
                    if unit_tmp.exists():
                        unit_tmp.unlink()
            if line_diagnostics is not None:
                if not str(args.mts_glt_interaction_diagnostics_dir):
                    raise ValueError(
                        'line conditioning modes require '
                        '--mts_glt_interaction_diagnostics_dir'
                    )
                line_diagnostics.update({
                    'task': str(task), 'fold': int(fold),
                    'seed': int(args.seed), 'fold_seed': int(fold_seed),
                    'mts_glt_mode': str(args.mts_glt_mode),
                    'best_epoch': int(metrics.get('best_epoch', -1)),
                    'test_r2': float(metrics.get('test_r2', float('nan'))),
                })
                unit_path = (
                    Path(args.mts_glt_interaction_diagnostics_dir)
                    / str(task) / f'fold_{int(fold)}.json'
                )
                unit_path.parent.mkdir(parents=True, exist_ok=True)
                unit_tmp = unit_path.with_name(
                    unit_path.name + f'.tmp.{os.getpid()}'
                )
                try:
                    unit_tmp.write_text(
                        json.dumps(line_diagnostics, indent=2, sort_keys=True)
                        + '\n', encoding='utf-8'
                    )
                    os.replace(unit_tmp, unit_path)
                finally:
                    if unit_tmp.exists():
                        unit_tmp.unlink()
            if attention_diagnostics is not None:
                if not str(args.mts_glt_interaction_diagnostics_dir):
                    raise ValueError(
                        'attention routing modes require '
                        '--mts_glt_interaction_diagnostics_dir'
                    )
                attention_diagnostics.update({
                    'task': str(task), 'fold': int(fold),
                    'seed': int(args.seed), 'fold_seed': int(fold_seed),
                    'mts_glt_mode': str(args.mts_glt_mode),
                    'best_epoch': int(metrics.get('best_epoch', -1)),
                    'test_r2': float(metrics.get('test_r2', float('nan'))),
                })
                unit_path = (
                    Path(args.mts_glt_interaction_diagnostics_dir)
                    / str(task) / f'fold_{int(fold)}.json'
                )
                unit_path.parent.mkdir(parents=True, exist_ok=True)
                unit_tmp = unit_path.with_name(unit_path.name + f'.tmp.{os.getpid()}')
                try:
                    unit_tmp.write_text(
                        json.dumps(attention_diagnostics, indent=2, sort_keys=True)
                        + '\n', encoding='utf-8'
                    )
                    os.replace(unit_tmp, unit_path)
                finally:
                    if unit_tmp.exists():
                        unit_tmp.unlink()
            if torsion_diagnostics is not None:
                if not str(args.mts_glt_interaction_diagnostics_dir):
                    raise ValueError(
                        'torsion modes require '
                        '--mts_glt_interaction_diagnostics_dir'
                    )
                torsion_diagnostics.update({
                    'task': str(task), 'fold': int(fold),
                    'seed': int(args.seed), 'fold_seed': int(fold_seed),
                    'mts_glt_mode': str(args.mts_glt_mode),
                    'best_epoch': int(metrics.get('best_epoch', -1)),
                    'test_r2': float(metrics.get('test_r2', float('nan'))),
                })
                unit_path = (
                    Path(args.mts_glt_interaction_diagnostics_dir)
                    / str(task) / f'fold_{int(fold)}.json'
                )
                unit_path.parent.mkdir(parents=True, exist_ok=True)
                unit_tmp = unit_path.with_name(unit_path.name + f'.tmp.{os.getpid()}')
                try:
                    unit_tmp.write_text(
                        json.dumps(torsion_diagnostics, indent=2, sort_keys=True)
                        + '\n', encoding='utf-8'
                    )
                    os.replace(unit_tmp, unit_path)
                finally:
                    if unit_tmp.exists():
                        unit_tmp.unlink()
            if joint_basis_diagnostics is not None:
                if not str(args.mts_glt_joint_basis_diagnostics_dir):
                    raise ValueError(
                        'joint SBF modes require '
                        '--mts_glt_joint_basis_diagnostics_dir'
                    )
                joint_basis_diagnostics.update({
                    'task': str(task), 'fold': int(fold),
                    'seed': int(args.seed), 'fold_seed': int(fold_seed),
                    'mts_glt_mode': str(args.mts_glt_mode),
                    'best_epoch': int(metrics.get('best_epoch', -1)),
                    'test_r2': float(metrics.get('test_r2', float('nan'))),
                })
                unit_path = (
                    Path(args.mts_glt_joint_basis_diagnostics_dir)
                    / str(task) / f'fold_{int(fold)}.json'
                )
                unit_path.parent.mkdir(parents=True, exist_ok=True)
                unit_tmp = unit_path.with_name(unit_path.name + f'.tmp.{os.getpid()}')
                try:
                    unit_tmp.write_text(
                        json.dumps(joint_basis_diagnostics, indent=2, sort_keys=True)
                        + '\n', encoding='utf-8'
                    )
                    os.replace(unit_tmp, unit_path)
                finally:
                    if unit_tmp.exists():
                        unit_tmp.unlink()
            postmortem = metrics.pop('_mts_glt_postmortem', None)
            graphgate_audit = metrics.pop('_mts_glt_graphgate_audit', None)
            if graphgate_audit is not None:
                graphgate_audit.update({
                    'task': str(task), 'fold': int(fold), 'seed': int(args.seed),
                    'fold_seed': int(fold_seed),
                    'best_epoch': int(metrics.get('best_epoch', -1)),
                    'test_r2': float(metrics.get('test_r2', float('nan'))),
                })
                # Scheduler units pass a concrete ``.../shards/.../fold_N.csv``
                # path.  Keep diagnostics beside the shard tree instead of
                # accidentally turning that CSV path into a directory.
                audit_root = Path(task_result_output)
                while audit_root.name != 'shards' and audit_root != audit_root.parent:
                    audit_root = audit_root.parent
                if audit_root.name == 'shards':
                    audit_root = audit_root.parent
                else:
                    audit_root = Path(task_result_output).parent
                audit_path = audit_root / 'fusion_audit_units' / str(task) / f'fold_{int(fold)}.json'
                audit_path.parent.mkdir(parents=True, exist_ok=True)
                audit_tmp = audit_path.with_name(audit_path.name + f'.tmp.{os.getpid()}')
                try:
                    audit_tmp.write_text(json.dumps(graphgate_audit, indent=2, sort_keys=True) + '\n', encoding='utf-8')
                    os.replace(audit_tmp, audit_path)
                finally:
                    if audit_tmp.exists():
                        audit_tmp.unlink()
            if postmortem is not None:
                postmortem.update({
                    'task': str(task),
                    'fold': int(fold),
                    'seed': int(args.seed),
                    'fold_seed': int(fold_seed),
                    'best_epoch': int(metrics.get('best_epoch', -1)),
                    'rerun_fused_r2': float(metrics.get('test_r2', float('nan'))),
                })
                audit_root = Path(
                    args.mts_glt_fusion_warm_dir
                    or args.mts_glt_postmortem_dir
                )
                postmortem.update({
                    'fusion_strategy': str(args.mts_glt_fusion_strategy),
                    'fusion_warm_epochs': int(args.mts_glt_fusion_warm_epochs),
                    'initial_alpha': float(args.mts_glt_initial_alpha),
                    'stage2_trainability': str(
                        args.mts_glt_fusion_stage2_trainability
                    ),
                })
                unit_path = (
                    audit_root / 'fusion_audit_units'
                    / str(task) / f'fold_{int(fold)}.json'
                )
                unit_path.parent.mkdir(parents=True, exist_ok=True)
                unit_tmp = unit_path.with_name(
                    unit_path.name + f'.tmp.{os.getpid()}'
                )
                try:
                    unit_tmp.write_text(
                        json.dumps(postmortem, indent=2, sort_keys=True) + '\n',
                        encoding='utf-8',
                    )
                    os.replace(unit_tmp, unit_path)
                finally:
                    if unit_tmp.exists():
                        unit_tmp.unlink()
            # The speed benchmark can request several evaluation batch sizes
            # after training has selected the fold's best model.  All of these
            # evaluations therefore use the exact same in-memory model state,
            # validation split, scaler, and sample order; they are not
            # independent training runs.  The feature is opt-in and has no
            # effect on production launches.
            benchmark_eval_batches = os.environ.get(
                'MTS_BENCHMARK_EVAL_BATCHES', ''
            ).strip()
            if benchmark_eval_batches and args.predictions_dir:
                try:
                    requested_eval_batches = sorted({
                        int(value.strip())
                        for value in benchmark_eval_batches.split(',')
                        if value.strip()
                    })
                except ValueError as exc:
                    raise ValueError(
                        'MTS_BENCHMARK_EVAL_BATCHES must be comma-separated '
                        'positive integers'
                    ) from exc
                if not requested_eval_batches or any(
                    value <= 0 for value in requested_eval_batches
                ):
                    raise ValueError(
                        'MTS_BENCHMARK_EVAL_BATCHES must contain positive '
                        'integers'
                    )
                benchmark_eval_root = Path(os.environ.get(
                    'MTS_BENCHMARK_EVAL_OUTPUT_DIR',
                    str(Path(args.predictions_dir).parent / 'benchmark_eval_predictions'),
                ))
                benchmark_eval_records = []
                for eval_batch_size in requested_eval_batches:
                    eval_started = time.perf_counter()
                    benchmark_loader = get_data_loader(
                        dataset,
                        indices=test_indices,
                        batch_size=eval_batch_size,
                        shuffle=False,
                        drop_last=False,
                        num_workers=args.loader_workers,
                        pin_memory=True,
                        persistent_workers=args.loader_workers > 0,
                    )
                    benchmark_metrics = test_model(
                        model,
                        benchmark_loader,
                        scaler,
                        device,
                        return_predictions=True,
                        amp_dtype=args.amp_dtype,
                    )
                    benchmark_path = (
                        benchmark_eval_root / f'batch_{eval_batch_size}'
                        / task / f'fold_{fold}.npz'
                    )
                    benchmark_path.parent.mkdir(parents=True, exist_ok=True)
                    benchmark_metadata = {
                        'task': task,
                        'fold': int(fold),
                        'seed': int(args.seed),
                        'fold_seed': int(fold_seed),
                        'eval_batch_size': int(eval_batch_size),
                        'amp_dtype': args.amp_dtype,
                        'loader_workers': int(args.loader_workers),
                        'sample_order': 'test_indices_in_source_order',
                        'model_state_scope': (
                            'same_train_and_evaluate_fold_best_model_state'
                        ),
                        'source_primary_eval_batch_size': int(args.eval_batch_size),
                        'eval_seconds': float(time.perf_counter() - eval_started),
                    }
                    benchmark_tmp = benchmark_path.with_name(
                        benchmark_path.name + f'.tmp.{os.getpid()}'
                    )
                    try:
                        with benchmark_tmp.open('wb') as handle:
                            np.savez(
                                handle,
                                y_true=np.asarray(
                                    benchmark_metrics['_y_true'],
                                    dtype=np.float64,
                                ),
                                y_pred=np.asarray(
                                    benchmark_metrics['_y_pred'],
                                    dtype=np.float64,
                                ),
                                sample_indices=np.asarray(
                                    test_indices, dtype=np.int64
                                ),
                                metadata=np.asarray(json.dumps(
                                    benchmark_metadata, sort_keys=True
                                )),
                            )
                        os.replace(benchmark_tmp, benchmark_path)
                    finally:
                        if benchmark_tmp.exists():
                            benchmark_tmp.unlink()
                    benchmark_eval_records.append({
                        **benchmark_metadata,
                        'path': str(benchmark_path),
                        'test_r2': float(benchmark_metrics['test_r2']),
                        'test_mae': float(benchmark_metrics['test_mae']),
                        'test_rmse': float(benchmark_metrics['test_rmse']),
                    })
                metrics['benchmark_eval_batch_records'] = benchmark_eval_records
            if args.refit_full_train:
                if initial_model_state is None:
                    raise RuntimeError(
                        'Full-train refit requires the captured fold initialization.'
                    )
                refit_epochs = int(metrics.get('best_epoch', -1))
                if refit_epochs <= 0:
                    raise RuntimeError(
                        'Full-train refit requires a raw validation-selected '
                        'best_epoch; disable SWA or refit_full_train.'
                    )
                print(
                    f'Refitting fold {fold + 1} from the initial checkpoint on '
                    f'all {len(train_indices)} outer-train samples for '
                    f'{refit_epochs} selected epoch(s).'
                )
                refit_scaler = scale_targets(
                    dataset,
                    task,
                    train_indices=train_indices,
                    raw_targets=raw_targets,
                    transform_mode=args.target_transform,
                )
                auxiliary_tasks = configure_cross_task_targets(
                    task, dataset, train_indices, 'outer-train refit'
                )
                refit_train_loader = get_data_loader(
                    dataset,
                    indices=train_indices,
                    batch_size=args.batch_size,
                    shuffle=True,
                    drop_last=False,
                    num_workers=args.loader_workers,
                    pin_memory=True,
                    persistent_workers=args.loader_workers > 0,
                )
                test_loader = get_data_loader(
                    dataset,
                    indices=test_indices,
                    batch_size=args.eval_batch_size,
                    shuffle=False,
                    drop_last=False,
                    num_workers=args.loader_workers,
                    pin_memory=True,
                    persistent_workers=args.loader_workers > 0,
                )
                set_global_seed(fold_seed)
                model.load_state_dict(initial_model_state)
                model.to(device)
                refit_details = fit_fixed_epochs(
                    model,
                    refit_train_loader,
                    device,
                    num_epochs=refit_epochs,
                    max_grad_norm=args.max_grad_norm,
                    smiles_lr=args.smiles_lr,
                    graph_lr=args.graph_lr,
                    geom_lr=args.geom_lr,
                    fp_lr=args.fp_lr,
                    fusion_lr=args.fusion_lr,
                    head_lr=args.head_lr,
                    weight_decay=args.weight_decay,
                    warmup_epochs=args.warmup_epochs,
                    freeze_smiles_epochs=args.freeze_smiles_epochs,
                    deep_unfreeze_epoch=args.deep_unfreeze_epoch,
                    fp_unfreeze_epoch=args.fp_unfreeze_epoch,
                    regression_loss=args.regression_loss,
                    huber_beta=args.huber_beta,
                    unimodal_aux_weight=args.unimodal_aux_weight,
                    fusion_prior_kl_weight=args.fusion_prior_kl_weight,
                    fusion_prior=args.fusion_prior,
                    cross_task_aux_weight=args.cross_task_aux_weight,
                    mts_o8_lr=args.mts_o8_lr,
                    mts_geometry_lr=args.mts_geometry_lr,
                    mts_adapter_lr=args.mts_adapter_lr,
                    amp_dtype=args.amp_dtype,
                )
                refit_test_metrics = test_model(
                    model, test_loader, refit_scaler, device,
                    return_predictions=True,
                    amp_dtype=args.amp_dtype,
                )
                metrics.update(refit_test_metrics)
                metrics.update(refit_details)
                metrics['refit_full_train'] = True
            else:
                metrics['refit_full_train'] = False
                metrics['refit_epochs'] = 0
            if (
                args.graph_encoder_type == 'mips_trimer_scage'
                and len(args.modalities) > 1
                and hasattr(model, 'modality_control')
            ):
                original_control = model.modality_control
                original_modality = model.controlled_modality
                for modality in model.modality_list:
                    model.controlled_modality = modality
                    for control in ('batch_shuffled', 'constant_zero'):
                        model.modality_control = control
                        control_scaler = (
                            refit_scaler if args.refit_full_train else scaler
                        )
                        controlled = test_model(
                            model, test_loader, control_scaler, device,
                            amp_dtype=args.amp_dtype,
                        )
                        for key, value in controlled.items():
                            metrics[
                                f'{modality}_{control}_{key}'
                            ] = float(value)
                model.modality_control = original_control
                model.controlled_modality = original_modality
            prediction_true = metrics.pop('_y_true', None)
            prediction_values = metrics.pop('_y_pred', None)
            prediction_path = None
            if args.graph_encoder_type == 'mips_trimer_scage':
                if prediction_true is None or prediction_values is None:
                    raise RuntimeError('MTS fold did not produce raw-space predictions')
                if not args.predictions_dir:
                    raise RuntimeError('MTS Stage 3 requires --predictions_dir')
                prediction_path = (
                    Path(args.predictions_dir) / task / f'fold_{fold}.npz'
                )
                prediction_path.parent.mkdir(parents=True, exist_ok=True)
                prediction_tmp = prediction_path.with_name(
                    prediction_path.name + f'.tmp.{os.getpid()}'
                )
                metadata = {
                    'task': task,
                    'fold': int(fold),
                    'seed': int(args.seed),
                    'fold_seed': int(fold_seed),
                    'fold_validation_protocol': (
                        'nested_outer5_inner_hash10'
                        if args.evaluation_protocol == 'nested5'
                        else 'shared_validation_test_fold'
                    ),
                    'independent_blind_test': args.evaluation_protocol == 'nested5',
                    'amp_dtype': args.amp_dtype,
                    'train_batch_size': int(args.batch_size),
                    'eval_batch_size': int(args.eval_batch_size),
                    'physical_gpu_id': os.environ.get('CUDA_VISIBLE_DEVICES', ''),
                    'resolved_config_path': args.resolved_config_path,
                    'resolved_command_path': args.resolved_command_path,
                }
                try:
                    with prediction_tmp.open('wb') as handle:
                        np.savez(
                            handle,
                            y_true=np.asarray(prediction_true, dtype=np.float64),
                            y_pred=np.asarray(prediction_values, dtype=np.float64),
                            sample_indices=np.asarray(test_indices, dtype=np.int64),
                            metadata=np.asarray(
                                json.dumps(metadata, sort_keys=True)
                            ),
                        )
                    os.replace(prediction_tmp, prediction_path)
                finally:
                    if prediction_tmp.exists():
                        prediction_tmp.unlink()
            metrics["fold_wall_seconds"] = float(
                time.monotonic() - fold_started
            )
            metrics["fold"] = int(fold)

            fold_metrics.append(metrics)
            print(
                f"Fold {fold + 1} Test R2: {metrics['test_r2']:.3f}, "
                f"MAE: {metrics['test_mae']:.3f}, RMSE: {metrics['test_rmse']:.3f}"
            )
            if best_model_state is None or metrics['best_val_r2'] > best_fold_val_r2:
                best_fold_val_r2 = metrics['best_val_r2']
                best_model_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

            torch.cuda.empty_cache()

        avg_test_r2 = np.mean([metric['test_r2'] for metric in fold_metrics])
        std_test_r2 = np.std([metric['test_r2'] for metric in fold_metrics])
        avg_test_mae = np.mean([metric['test_mae'] for metric in fold_metrics])
        std_test_mae = np.std([metric['test_mae'] for metric in fold_metrics])
        avg_test_rmse = np.mean([metric['test_rmse'] for metric in fold_metrics])
        std_test_rmse = np.std([metric['test_rmse'] for metric in fold_metrics])
        avg_val_r2 = np.mean([metric['best_val_r2'] for metric in fold_metrics])
        std_val_r2 = np.std([metric['best_val_r2'] for metric in fold_metrics])

        print("\nAverage of metrics over all folds")
        print(f"Test R2 = {avg_test_r2:.3f}")
        print(f"Test MAE = {avg_test_mae:.3f}")
        print(f"Test RMSE = {avg_test_rmse:.3f}")
        print(f"Standard Deviation of Test R2 = {std_test_r2:.3f}")
        print(f"Standard Deviation of Test MAE = {std_test_mae:.3f}")
        print(f"Standard Deviation of Test RMSE = {std_test_rmse:.3f}")
        print(f"Best Validation R2 = {avg_val_r2:.3f} +/- {std_val_r2:.3f}")

        # Downstream model persistence is intentionally disabled. The best
        # fold state remains in memory for evaluation, but saved_models is not
        # populated after training.
        # os.makedirs(os.path.join(model_output_dir, task), exist_ok=True)
        # torch.save(best_model_state, os.path.join(model_output_dir, f'{task}/{args.model_name}_best.pth'))
        # print(f"Best fold model saved by validation R2: {best_fold_val_r2:.3f}")

        # Save results
        result = {
            'task': task,
            'model_name': args.model_name,
            # Keep every result shard self-describing.  This is intentionally
            # redundant with the checkpoint metadata: a shard must be safe to
            # resume/merge without consulting a mutable command line or the
            # current cache directory.
            'experiment_id': args.experiment_id,
            'amp_dtype': args.amp_dtype,
            'train_batch_size': int(args.batch_size),
            'eval_batch_size': int(args.eval_batch_size),
            'physical_gpu_id': os.environ.get('CUDA_VISIBLE_DEVICES', ''),
            'checkpoint_path': str(pretrained_model_path) if pretrained_model_path else None,
            'resolved_config_path': args.resolved_config_path,
            'resolved_command_path': args.resolved_command_path,
            'resolved_config': (
                Path(args.resolved_config_path).read_text(encoding='utf-8')
                if args.resolved_config_path and Path(args.resolved_config_path).is_file()
                else None
            ),
            'model_modality_list': model_modality_list,
            'fusion_type': args.fusion_type,
            'fp_mode': args.fp_mode,
            'fp_dim': {
                'ecfp': 1024,
                'mixfp': 1048,
                'attachment_count': 2570,
                'disabled': 0,
            }[args.fp_mode],
            'mips_core': args.mips_core if args.graph_encoder_type == 'mips_trimer_scage' else None,
            'mips_max_hops': args.mips_max_hops if args.graph_encoder_type == 'mips_trimer_scage' else None,
            'mips_use_descriptors': (
                bool(args.mips_use_descriptors)
                if args.graph_encoder_type == 'mips_trimer_scage' else None
            ),
            'spatial_mode': (
                args.spatial_mode if args.graph_encoder_type == 'mips_trimer_scage' else None
            ),
            'graph_geometry_mode': (
                args.graph_geometry_mode
                if args.graph_encoder_type == 'mips_trimer_scage' else None
            ),
            'topology_attention_variant': args.topology_attention_variant,
            'mts_glt_mode': args.mts_glt_mode,
            'mts_o8_bond_bias_mode': args.mts_o8_bond_bias_mode,
            'mts_glt_geometry_mode': args.mts_glt_geometry_mode,
            'mts_glt_metadata_mode': getattr(args, 'mts_glt_metadata_mode', 'full'),
            'mts_glt_fusion_strategy': args.mts_glt_fusion_strategy,
            'mts_glt_fusion_warm_epochs': int(args.mts_glt_fusion_warm_epochs),
            'mts_glt_initial_alpha': float(args.mts_glt_initial_alpha),
            'mts_glt_fusion_stage2_trainability': (
                args.mts_glt_fusion_stage2_trainability
            ),
            'star_rbf_upper': getattr(args, 'star_rbf_upper', None),
            'star_rbf_v2_sidecar': getattr(args, 'star_rbf_v2_sidecar', None),
            'feature_cohort': getattr(dataset, 'feature_cohort_name', None),
            'feature_cache_item_timeout': int(
                args.feature_cache_item_timeout
            ),
            'evaluation_protocol': args.evaluation_protocol,
            'mips_fusion_mode': args.mips_fusion_mode,
            'projection_mode': args.projection_mode,
            'modality_control': args.modality_control,
            'mips_variant': (
                args.mips_variant if args.graph_encoder_type == 'mips_trimer_scage' else None
            ),
            'fusion_dropout': args.fusion_dropout,
            'head_dropout': args.head_dropout,
            'fp_bit_dropout': args.fp_bit_dropout,
            'modality_dropout': (
                f"smiles={args.smiles_modality_dropout};graph={args.graph_modality_dropout};"
                f"fp={args.fp_modality_dropout}"
            ),
            'regression_loss': args.regression_loss,
            'finetune_mode': args.finetune_mode,
            'huber_beta': args.huber_beta,
            'unimodal_aux_weight': args.unimodal_aux_weight,
            'cross_task_aux_weight': args.cross_task_aux_weight,
            'cross_task_auxiliary_tasks': (
                ';'.join(CROSS_TASK_AUXILIARY_MAP.get(task, ()))
                if (
                    args.cross_task_aux_weight > 0
                    and (
                        not selected_cross_task_aux
                        or task in selected_cross_task_aux
                    )
                ) else ''
            ),
            'cross_task_aux_target_tasks': ';'.join(
                args.cross_task_aux_tasks
            ),
            'fusion_prior_kl_weight': args.fusion_prior_kl_weight,
            'fusion_prior': args.fusion_prior,
            'swa_start_epoch': args.swa_start_epoch,
            'swa_selected_folds': sum(
                int(metric.get('swa_selected', False)) for metric in fold_metrics
            ),
            'swa_mean_snapshots': np.mean([
                metric.get('swa_snapshots', 0) for metric in fold_metrics
            ]),
            'seed': args.seed,
            'prediction_path': str(prediction_path) if prediction_path else None,
            'target_transform': args.target_transform,
            'fold_validation_protocol': (
                'nested_outer5_inner_hash10'
                if args.evaluation_protocol == 'nested5'
                else 'shared_validation_test_fold'
            ),
            'independent_blind_test': args.evaluation_protocol == 'nested5',
            'refit_full_train': bool(args.refit_full_train),
            'avg_refit_epochs': np.mean([
                metric.get('refit_epochs', 0) for metric in fold_metrics
            ]),
            'avg_best_val_r2': float(avg_val_r2),
            'std_best_val_r2': float(std_val_r2),
            'optimizer_lrs': (
                f"mts_o8={args.mts_o8_lr};mts_geometry={args.mts_geometry_lr};"
                f"mts_adapter={args.mts_adapter_lr};head={args.head_lr}"
                if args.graph_encoder_type == 'mips_trimer_scage' else
                f"smiles={args.smiles_lr};graph={args.graph_lr};"
                f"fp={'frozen' if args.fp_unfreeze_epoch < 0 else args.fp_lr};"
                f"fusion={args.fusion_lr};head={args.head_lr}"
            ),
            'fp_unfreeze_epoch': args.fp_unfreeze_epoch,
            'batch_size': args.batch_size,
            'total_fold_wall_seconds': float(sum(
                metric.get("fold_wall_seconds", 0.0)
                for metric in fold_metrics
            )),
            'estimated_gpu_hours': float(sum(
                metric.get("fold_wall_seconds", 0.0)
                for metric in fold_metrics
            ) / 3600.0),
            'graph_input': args.graph_input,
            'geom_input': args.geom_input,
            'graph_encoder_type': args.graph_encoder_type,
            'topology_representation': args.topology_representation,
            'baseline': (
                MTS_ROUTE_NAME
                if args.graph_encoder_type == 'mips_trimer_scage' else 'retired_route'
            ),
            'route_short_name': (
                MTS_ROUTE_SHORT_NAME
                if args.graph_encoder_type == 'mips_trimer_scage' else None
            ),
            'scage_backbone': (
                'sparse_non_pbc_mips_starlink_spd_single_path_node'
                if args.graph_encoder_type == 'mips_trimer_scage' else None
            ),
            'scage_input': (
                'mips137_independent_backbone_embedding'
                if args.graph_encoder_type == 'mips_trimer_scage' else None
            ),
            'scage_checkpoint_schema': (
                MIPS_TRIMER_CHECKPOINT_SCHEMA
                if args.graph_encoder_type == 'mips_trimer_scage' else None
            ),
            'cache_bundle_schema': (
                MIPS_TRIMER_CACHE_BUNDLE_SCHEMA
                if args.graph_encoder_type == 'mips_trimer_scage' else None
            ),
            'topology_lmdb_schema': (
                MIPS_TRIMER_TOPOLOGY_SCHEMA
                if args.graph_encoder_type == 'mips_trimer_scage' else None
            ),
            'pretraining_dataset': (
                checkpoint_meta.get('pretraining_dataset')
                if args.graph_encoder_type == 'mips_trimer_scage' and pretrained_model_path
                else None
            ),
            'avg_test_r2': float(avg_test_r2),
            'std_test_r2': float(std_test_r2),
            'avg_test_mae': float(avg_test_mae),
            'std_test_mae': float(std_test_mae),
            'avg_test_rmse': float(avg_test_rmse),
            'std_test_rmse': float(std_test_rmse),
            'per_fold_metrics': json.dumps(fold_metrics, sort_keys=True),
        }

        # Save to CSV.  Stage-3 campaign units write exactly one fold per
        # shard.  Use an atomic replacement for those paths so an interrupted
        # process can never leave a partially written CSV that looks complete
        # to the resume logic.  Legacy aggregate outputs retain append mode.
        os.makedirs(os.path.dirname(task_result_output) or ".", exist_ok=True)
        results_df = pd.DataFrame([result])
        write_header = not task_result_file_initialized
        shard_path = str(task_result_output).replace("\\", "/")
        atomic_shard = "/shards/" in shard_path and write_header
        if atomic_shard:
            tmp_path = f"{task_result_output}.tmp.{os.getpid()}"
            try:
                results_df.to_csv(
                    tmp_path,
                    mode='w',
                    header=True,
                    index=False,
                    float_format="%.17g",
                )
                os.replace(tmp_path, task_result_output)
            finally:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
        else:
            results_df.to_csv(
                task_result_output,
                mode='w' if write_header else 'a',
                header=write_header,
                index=False,
                float_format="%.17g",
            )
        print(f"Results have been appended to '{task_result_output}'.")


def main():
    return run_finetune_job(parse_arguments())


if __name__ == "__main__":
    main()
