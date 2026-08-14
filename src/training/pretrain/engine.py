import os
import sys
import argparse
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
import numpy as np
import json
import hashlib
import math
import subprocess
import time
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.dataset.mips_trimer_contract import (
    TOPOLOGY_CANONICAL,
    TOPOLOGY_EXPLICIT,
    ROUTE_NAME as MTS_ROUTE_NAME,
    ROUTE_INTERNAL as MTS_ROUTE_INTERNAL,
    STAGE1_ID as MTS_STAGE1_ID,
    STAGE2_ID as MTS_STAGE2_ID,
    PRETRAIN_CHECKPOINT_SCHEMA,
    PRETRAIN_TRAIN_STATE_SCHEMA,
    stage_display_name,
    normalize_stage,
    validate_runtime_args as validate_mips_trimer_runtime,
)
from src.dataset.mips_cache_validation import trimer_can_enter_mcl
from src.training.pretrain.config import (
    PretrainRuntimeConfig,
    dataset_kwargs_from_args,
    parse_arguments,
)
from src.training.pretrain.objectives import compose_joint_payload_loss
from src.training.common.rng import (
    capture_rng_state as _capture_rng_state,
    restore_rng_state as _restore_rng_state,
)


def _gather_rng_states(local_state, distributed, rank, world_size):
    """Collect rank-local stochastic state before a distributed checkpoint."""
    if not distributed:
        return [local_state]
    gathered = [None for _ in range(int(world_size))]
    # All ranks call this function at the same optimizer boundary.  Object
    # gather is intentional: Python/NumPy RNG tuples are not tensors.
    dist.all_gather_object(gathered, local_state)
    return gathered if rank == 0 else None


def _gather_rank_states(local_state, distributed, rank, world_size):
    """Gather a rank-local DataLoader/sampler state at a checkpoint barrier."""
    if not distributed:
        return [local_state]
    gathered = [None for _ in range(int(world_size))]
    dist.all_gather_object(gathered, local_state)
    return gathered if rank == 0 else None


def _atomic_torch_save(payload, path):
    from src.training.common.checkpoint import atomic_torch_save

    atomic_torch_save(payload, path)

def _base_model(model):
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model = model.module
    if isinstance(model, MIPSPretrainContainer):
        return model.model
    return model


def _distributed_enabled():
    return dist.is_available() and dist.is_initialized()


def _distributed_sum_count_mean(local_sum, local_count, device):
    """Convert a local sum into a global sample-weighted mean.

    DDP averages gradients across ranks, so multiplying the local sum by
    ``world_size / global_count`` yields the same gradient as a single global
    mean even when ranks contain different numbers of valid targets.
    """
    count = torch.tensor(float(local_count), device=device, dtype=torch.float32)
    global_count = count.clone()
    world_size = 1
    if _distributed_enabled():
        world_size = dist.get_world_size()
        dist.all_reduce(global_count, op=dist.ReduceOp.SUM)
    if float(global_count.item()) <= 0.0:
        return local_sum * 0.0, 0
    return local_sum * (float(world_size) / global_count), int(global_count.item())


def _distributed_sum_int(local_value, device):
    """All-reduce a small integer statistic without synchronizing gradients."""
    value = torch.tensor(int(local_value), device=device, dtype=torch.long)
    if _distributed_enabled():
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
    return int(value.item())


def _distributed_joint_counts(counts, device):
    """Reduce all joint-pretraining counters in one collective."""
    names = (
        "masked_atoms", "angle_graphs", "masked_correct", "angle_correct",
        "angle_targets", "mcl_valid_graphs", "graphs",
    )
    packed = torch.tensor(
        [int(counts[name]) for name in names], device=device, dtype=torch.long
    )
    if _distributed_enabled():
        dist.all_reduce(packed, op=dist.ReduceOp.SUM)
    values = packed.cpu().tolist()
    return dict(zip(names, (int(value) for value in values)))


def _global_mean_from_count(local_sum, global_count):
    """DDP-correct global mean after counts have already been reduced."""
    if int(global_count) <= 0:
        return local_sum * 0.0
    world_size = dist.get_world_size() if _distributed_enabled() else 1
    return local_sum * (float(world_size) / float(global_count))


def _all_reduce_gradients(modules):
    """Legacy synchronization used only by non-MIPS training routes.

    The non-PBC MIPS Stage 1/2 route is wrapped in real
    ``DistributedDataParallel`` and must never call this helper.
    """
    if not _distributed_enabled():
        return
    world_size = dist.get_world_size()
    for module in modules:
        for parameter in module.parameters():
            if parameter.grad is None:
                parameter.grad = torch.zeros_like(parameter)
            dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
            parameter.grad.div_(world_size)


@torch.no_grad()
def _assert_ddp_parameters_synced(module, step, samples_per_tensor=16):
    """Cheap every-update divergence detector for the three-rank campaign."""
    if not dist.is_available() or not dist.is_initialized():
        return
    checksum = torch.zeros(3, device=next(module.parameters()).device, dtype=torch.float64)
    for parameter in module.parameters():
        flat = parameter.detach().reshape(-1)
        if not flat.numel():
            continue
        if flat.numel() <= int(samples_per_tensor):
            sample = flat.double()
        else:
            sample_count = int(samples_per_tensor)
            indices = (
                torch.arange(sample_count, device=flat.device, dtype=torch.long)
                * (flat.numel() - 1)
                // (sample_count - 1)
            )
            sample = flat[indices].double()
        checksum[0] += sample.sum()
        checksum[1] += sample.square().sum()
        checksum[2] += sample.abs().max()
    minimum, maximum = checksum.clone(), checksum.clone()
    dist.all_reduce(minimum, op=dist.ReduceOp.MIN)
    dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
    tolerance = 1e-10 * maximum.abs().clamp_min(1.0)
    if bool(((maximum - minimum).abs() > tolerance).any()):
        raise RuntimeError(
            f"DDP parameter divergence detected after optimizer step {step}: "
            f"min={minimum.tolist()}, max={maximum.tolist()}"
        )


def _distributed_sum_mapping(values, device):
    """Sum a scalar mapping across ranks, including keys absent on some ranks."""
    if not _distributed_enabled():
        return dict(values)
    gathered_keys = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered_keys, sorted(values))
    keys = sorted({key for rank_keys in gathered_keys for key in rank_keys})
    if not keys:
        return {}
    tensor = torch.tensor(
        [float(values.get(key, 0.0)) for key in keys],
        dtype=torch.float64,
        device=device,
    )
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return {key: float(tensor[idx].item()) for idx, key in enumerate(keys)}


def _gradient_vector(modules):
    values = [
        parameter.grad.detach().float().flatten()
        for module in modules for parameter in module.parameters()
        if parameter.grad is not None
    ]
    return torch.cat(values) if values else torch.zeros(1)


def _bf16_parity_gate(base_model, data, atom_head, args, geometry_adapt=False):
    modules = (base_model, atom_head)
    for module in modules:
        module.zero_grad(set_to_none=True)
    fp32_loss, _ = _mips_masked_atom_loss(
        base_model, data, atom_head, args.graph_mask_ratio, args.seed, 0,
        geometry_adapt=geometry_adapt,
    )
    fp32_loss.backward()
    fp32_grad = _gradient_vector(modules)
    for module in modules:
        module.zero_grad(set_to_none=True)
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        bf16_loss, _ = _mips_masked_atom_loss(
            base_model, data, atom_head, args.graph_mask_ratio, args.seed, 0,
            geometry_adapt=geometry_adapt,
        )
    bf16_loss.backward()
    bf16_grad = _gradient_vector(modules)
    relative_delta = float(
        (bf16_loss.float() - fp32_loss.float()).abs()
        / fp32_loss.detach().float().abs().clamp_min(1e-8)
    )
    cosine = float(F.cosine_similarity(fp32_grad, bf16_grad, dim=0).item())
    finite = bool(torch.isfinite(bf16_loss) and torch.isfinite(bf16_grad).all())
    for module in modules:
        module.zero_grad(set_to_none=True)
    passed = finite and relative_delta <= 0.02 and cosine >= 0.98
    return passed, {"relative_loss_delta": relative_delta, "gradient_cosine": cosine, "finite": finite}


def _unique_smiles_indices(dataset):
    seen = set()
    indices = []
    raw_items = getattr(dataset, "data_list", None)
    for index in range(len(dataset)):
        # Lazy v2 caches keep ``(smiles, target)`` rows and only deserialize a
        # PyG object in __getitem__. Reading the tuple avoids one million
        # random shard reads merely to discover that PI1M_v2 is already unique.
        raw_item = raw_items[index] if raw_items is not None else dataset[index]
        smiles = str(raw_item[0] if isinstance(raw_item, tuple) else raw_item.smiles)
        if smiles in seen:
            continue
        seen.add(smiles)
        indices.append(index)
    return np.asarray(indices, dtype=np.int64)


class _ShardAwareDistributedSampler(torch.utils.data.Sampler):
    """DDP sampler that preserves lazy 2048-item cache locality.

    Each rank owns one fixed contiguous source range, so it reads only about
    one third of the immutable cache shards instead of all ranks repeatedly
    deserializing every shard. Shard order and rows inside each owned shard are
    independently shuffled every epoch. The equal contiguous ranges preserve
    identical global coverage (apart from the normal distributed tail drop).
    """

    def __init__(
        self, dataset_size, num_replicas, rank, seed, shard_size=2048
    ):
        self.dataset_size = int(dataset_size)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.seed = int(seed)
        self.shard_size = int(shard_size)
        self.epoch = 0
        self.num_samples = self.dataset_size // self.num_replicas
        self.total_size = self.num_samples * self.num_replicas

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return self.num_samples


class _RankSliceSampler(torch.utils.data.Sampler):
    """Non-padding deterministic sampler for distributed evaluation."""

    def __init__(self, length, rank, world_size):
        self.indices = tuple(range(int(rank), int(length), int(world_size)))

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        rank_start = self.rank * self.num_samples
        rank_end = rank_start + self.num_samples
        blocks = []
        cursor = rank_start
        while cursor < rank_end:
            shard_end = ((cursor // self.shard_size) + 1) * self.shard_size
            block_end = min(rank_end, shard_end)
            blocks.append(list(range(cursor, block_end)))
            cursor = block_end
        block_order = torch.randperm(
            len(blocks), generator=generator
        ).tolist()
        permutation = []
        for block_id in block_order:
            block = blocks[block_id]
            order = torch.randperm(
                len(block), generator=generator
            ).tolist()
            permutation.extend(block[offset] for offset in order)
        if len(permutation) != self.num_samples:
            raise RuntimeError(
                "rank-contiguous shard sampler produced an invalid length"
            )
        return iter(permutation)


class _CostBalancedDistributedSampler(torch.utils.data.Sampler):
    """Deterministic fixed-size batches balanced by graph cost."""

    def __init__(
        self, costs, batch_size, num_replicas, rank, seed, drop_last=False,
    ):
        super().__init__()
        values = np.asarray(costs, dtype=np.float64)
        if values.ndim == 2:
            values = values[:, 0] + 0.25 * values[:, 1]
        if values.ndim != 1:
            raise ValueError("topology costs must be [N] or [N,2]")
        self.costs = values
        self.dataset_size = int(values.size)
        self.batch_size = int(batch_size)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        global_batch = self.batch_size * self.num_replicas
        self.num_batches = (
            self.dataset_size // global_batch
            if self.drop_last else int(math.ceil(self.dataset_size / global_batch))
        )
        self.epoch = 0
        # Number of rank-local samples already consumed in the current epoch.
        # This cursor is intentionally not included in ``__len__``: training
        # uses absolute batch indices for accumulation and end-of-epoch logic.
        # A resumed iterator can therefore start directly at the checkpointed
        # batch without deserializing every preceding LMDB record.
        self.start_sample_index = 0

    def __len__(self):
        return self.num_batches * self.batch_size

    def set_epoch(self, epoch):
        self.epoch = int(epoch)
        self.start_sample_index = 0

    def set_start_batch(self, batch_index):
        sample_index = int(batch_index) * self.batch_size
        if sample_index < 0 or sample_index > len(self):
            raise ValueError(
                f"invalid cost-sampler resume batch {batch_index}: "
                f"sample offset {sample_index}, length {len(self)}"
            )
        self.start_sample_index = sample_index

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        order = np.arange(self.dataset_size, dtype=np.int64)
        rng.shuffle(order)
        global_batch = self.batch_size * self.num_replicas
        required = self.num_batches * global_batch
        if required > order.size:
            if self.drop_last:
                order = order[:required]
            else:
                order = np.concatenate([order, np.resize(order, required - order.size)])
        else:
            order = order[:required]
        output = []
        for start in range(0, required, global_batch):
            window = order[start:start + global_batch]
            rank_indices = [[] for _ in range(self.num_replicas)]
            rank_costs = [0.0 for _ in range(self.num_replicas)]
            for index in window[np.argsort(-self.costs[window], kind="stable")]:
                candidates = [r for r in range(self.num_replicas)
                              if len(rank_indices[r]) < self.batch_size]
                target = min(candidates, key=lambda r: (rank_costs[r], r))
                rank_indices[target].append(int(index))
                rank_costs[target] += float(self.costs[index])
            output.extend(rank_indices[self.rank])
        return iter(output[self.start_sample_index:])


def _polymer_ecfp_pos_weight(dataset, indices, device):
    positives = torch.zeros(2048, dtype=torch.float64)
    valid_count = 0
    for index in indices:
        sample = dataset[int(index)]
        valid = getattr(sample, 'polymer_ecfp_valid', False)
        valid = bool(valid.flatten()[0].item()) if torch.is_tensor(valid) else bool(valid)
        if not valid:
            continue
        target = getattr(sample, 'polymer_ecfp_target', None)
        if target is None or target.numel() != 2048:
            continue
        positives += target.detach().cpu().double().view(-1)
        valid_count += 1
    if valid_count == 0:
        raise ValueError("No valid 2048-bit Polymer ECFP targets were found")
    negatives = float(valid_count) - positives
    pos_weight = (negatives / positives.clamp_min(1.0)).clamp_(1.0, 20.0).float().to(device)
    return pos_weight, valid_count


def _merge_periodic_aug_stats(total, update):
    for key in (
        'attempted', 'success', 'skipped', 'same_smiles', 'valid_contrastive',
        'graph_cache_hits', 'graph_cache_misses',
    ):
        total[key] = int(total.get(key, 0)) + int(update.get(key, 0))
    return total


def _finalize_periodic_aug_stats(epoch, stats):
    attempted = int(stats.get('attempted', 0))
    success = int(stats.get('success', 0))
    return {
        'epoch': int(epoch),
        'attempted': attempted,
        'success': success,
        'skipped': int(stats.get('skipped', 0)),
        'same_smiles': int(stats.get('same_smiles', 0)),
        'valid_contrastive': int(stats.get('valid_contrastive', 0)),
        'graph_cache_hits': int(stats.get('graph_cache_hits', 0)),
        'graph_cache_misses': int(stats.get('graph_cache_misses', 0)),
        'success_rate': float(success / attempted) if attempted else None,
        'same_smiles_rate': float(stats.get('same_smiles', 0) / attempted) if attempted else None,
    }


def _stage_loss_weights(args):
    if args.pretrain_stage == 'scage_m4p':
        return {'contrastive': 0.0, 'graph': 1.0, 'geom': 0.0}
    if args.pretrain_stage == 'graph_geom':
        return {
            'contrastive': 0.0,
            'graph': float(args.graph_pretrain_weight),
            'geom': float(args.geom_denoise_weight),
        }
    if args.pretrain_stage == 'alignment':
        if args.graph_encoder_type == 'mips_trimer_scage':
            return {'contrastive': 1.0, 'graph': 0.0, 'geom': 0.0}
        return {
            'contrastive': 1.0,
            'graph': float(args.graph_pretrain_weight),
            'geom': float(args.geom_denoise_weight),
        }
    return {
        'contrastive': 1.0,
        'graph': float(args.graph_pretrain_weight),
        'geom': float(args.geom_denoise_weight),
    }


class DynamicPretrainLossWeighter:
    """SCAGE-style dynamic weighting with priors applied after normalization."""

    def __init__(
        self,
        task_names,
        init_window=200,
        recent_window=20,
        temperature=1.0,
        task_priors=None,
        effective_caps=None,
        device=None,
    ):
        self.task_names = list(task_names)
        self.init_window = int(init_window)
        self.recent_window = int(recent_window)
        self.temperature = float(temperature)
        if self.temperature <= 0:
            raise ValueError("dynamic loss temperature must be positive")
        self.device = device
        self.task_priors = {
            name: float((task_priors or {}).get(name, 1.0)) for name in self.task_names
        }
        self.effective_caps = {
            name: float(value) for name, value in (effective_caps or {}).items()
        }
        self.step = 0
        self.loss_init = {name: [] for name in self.task_names}
        self.loss_recent = {name: [] for name in self.task_names}
        self.loss_prev_recent = {name: [] for name in self.task_names}
        self.last_weights = {name: 1.0 / max(1, len(self.task_names)) for name in self.task_names}
        self.last_normalized = {name: 0.0 for name in self.task_names}
        self.last_effective = {name: 0.0 for name in self.task_names}
        self.last_baselines = {name: None for name in self.task_names}

    def _synchronized_statistics(self, loss_terms, reference=None):
        """Return detached per-task means shared by every distributed rank."""
        if not _distributed_enabled():
            return {
                name: loss_terms[name].detach()
                for name in self.task_names if name in loss_terms
            }

        reference = (
            next(iter(loss_terms.values()))
            if loss_terms else reference
        )
        if reference is None:
            raise ValueError("A differentiable reference is required for an empty local task set")
        values = reference.new_zeros(len(self.task_names))
        counts = reference.new_zeros(len(self.task_names))
        for idx, name in enumerate(self.task_names):
            if name in loss_terms:
                values[idx] = loss_terms[name].detach()
                counts[idx] = 1.0
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
        dist.all_reduce(counts, op=dist.ReduceOp.SUM)
        return {
            name: values[idx] / counts[idx].clamp_min(1.0)
            for idx, name in enumerate(self.task_names)
            if float(counts[idx].item()) > 0.0
        }

    def _mean_or_current(self, values, current_value):
        if not values:
            return current_value.detach().clamp_min(1e-8)
        return torch.stack(values).mean().to(current_value.device).clamp_min(1e-8)

    def _push(self, values, value, max_len):
        values.append(value.detach())
        if len(values) > max_len:
            values.pop(0)

    def _apply_effective_caps(self, effective, active_names):
        """Cap final coefficients and redistribute excess among uncapped tasks."""
        effective = effective / effective.sum().clamp_min(1e-8)
        for _ in range(len(active_names)):
            capped = []
            excess = effective.new_tensor(0.0)
            for idx, name in enumerate(active_names):
                cap = self.effective_caps.get(name)
                if cap is not None and float(effective[idx].item()) > cap:
                    excess = excess + effective[idx] - cap
                    effective[idx] = cap
                    capped.append(idx)
            if float(excess.item()) <= 0:
                break
            recipients = [idx for idx in range(len(active_names)) if idx not in capped]
            if not recipients:
                break
            recipient_values = effective[recipients]
            if float(recipient_values.sum().item()) <= 0:
                effective[recipients] += excess / len(recipients)
            else:
                effective[recipients] += excess * recipient_values / recipient_values.sum()
        return effective / effective.sum().clamp_min(1e-8)

    def __call__(self, loss_terms, reference=None):
        statistics = self._synchronized_statistics(loss_terms, reference=reference)
        active_names = [name for name in self.task_names if name in statistics]
        if not active_names:
            if reference is None:
                raise ValueError("dynamic pretrain loss received no active terms")
            self.last_weights = {}
            self.last_normalized = {}
            self.last_effective = {}
            return reference * 0.0
        reference = next(iter(loss_terms.values())) if loss_terms else reference

        def local_loss(name):
            # A task can be valid on other ranks but absent from this rank's
            # batch. Its local gradient contribution is correctly zero while
            # every rank still updates identical dynamic-weight state.
            return loss_terms.get(name, reference * 0.0)

        if len(active_names) == 1:
            only_name = active_names[0]
            self.last_weights = {only_name: 1.0}
            self.last_normalized = {
                only_name: float(statistics[only_name].detach().cpu().item())
            }
            self.last_effective = {only_name: 1.0}
            self._record(statistics)
            return local_loss(only_name)

        if self.step < self.init_window:
            priors = reference.new_tensor(
                [self.task_priors[name] for name in active_names]
            )
            prior_sum = priors.sum().clamp_min(1e-8)
            effective = self._apply_effective_caps(priors / prior_sum, active_names)
            total = torch.sum(torch.stack([local_loss(name) for name in active_names]) * effective)
            self._record(statistics)
            self.last_weights = {name: 1.0 / len(active_names) for name in active_names}
            self.last_normalized = {
                name: float(statistics[name].detach().cpu().item()) for name in active_names
            }
            self.last_effective = {
                name: float(effective[idx].detach().cpu().item())
                for idx, name in enumerate(active_names)
            }
            return total

        normalized = []
        normalized_statistics = []
        trend_ratios = []
        for name in active_names:
            statistic = statistics[name]
            init_mean = self._mean_or_current(self.loss_init[name], statistic)
            recent_mean = self._mean_or_current(self.loss_recent[name], statistic)
            prev_recent_mean = self._mean_or_current(self.loss_prev_recent[name], recent_mean)
            normalized.append(local_loss(name) / init_mean)
            normalized_statistics.append(statistic / init_mean)
            trend_ratios.append(recent_mean / prev_recent_mean)

        trend_tensor = torch.stack(trend_ratios)
        weights = F.softmax(trend_tensor / self.temperature, dim=0).detach()
        normalized_tensor = torch.stack(normalized)
        priors = normalized_tensor.new_tensor([self.task_priors[name] for name in active_names])
        effective = priors * weights
        effective = self._apply_effective_caps(effective, active_names)
        total = torch.sum(normalized_tensor * effective)
        self.last_weights = {name: float(weights[idx].detach().cpu().item()) for idx, name in enumerate(active_names)}
        self.last_normalized = {
            name: float(normalized_statistics[idx].detach().cpu().item())
            for idx, name in enumerate(active_names)
        }
        self.last_effective = {
            name: float(effective[idx].detach().cpu().item())
            for idx, name in enumerate(active_names)
        }
        self.last_baselines = {
            name: float(self._mean_or_current(self.loss_init[name], statistics[name]).cpu().item())
            for name in active_names
        }
        self._record(statistics)
        return total

    def _record(self, loss_terms):
        for name in self.task_names:
            if name not in loss_terms:
                continue
            value = loss_terms[name]
            # Baselines are collected only during warm-up and then frozen.
            if self.step < self.init_window:
                self._push(self.loss_init[name], value, self.init_window)
            if self.step % self.recent_window == 0 and self.loss_recent[name]:
                self.loss_prev_recent[name] = list(self.loss_recent[name])
                self.loss_recent[name] = []
            self._push(self.loss_recent[name], value, self.recent_window)
        self.step += 1


def _graph_mask_atom_loss(base_model, data, graph_atom_head, mask_ratio):
    if 'graph' not in base_model.encoders:
        return data.x.new_tensor(0.0)

    graph_module = base_model.encoders['graph']
    graph_encoder = graph_module.encoder
    from src.dataset.graph_data import allowable_features
    num_atom_symbols = len(allowable_features['possible_atom_symbols'])
    atom_symbol_targets = data.x[:, :num_atom_symbols].argmax(dim=1).long()
    mask = torch.rand(data.x.size(0), device=data.x.device) < float(mask_ratio)
    if not mask.any():
        mask[torch.randint(data.x.size(0), (1,), device=data.x.device)] = True

    masked_x = data.x.clone()
    masked_x[mask] = 0.0
    _, node_rep = _graph_encode_nodes(base_model, data, x_override=masked_x)
    return F.cross_entropy(graph_atom_head(node_rep[mask]), atom_symbol_targets[mask])


def _empty_periodic_aug_stats():
    return {
        'attempted': 0,
        'success': 0,
        'skipped': 0,
        'same_smiles': 0,
        'valid_contrastive': 0,
        'graph_cache_hits': 0,
        'graph_cache_misses': 0,
    }


def _graph_periodic_aug_loss(base_model, data, graph_input, temperature, max_mrus, retry):
    stats = _empty_periodic_aug_stats()
    if 'graph' not in base_model.encoders or not hasattr(data, 'smiles'):
        return data.x.new_tensor(0.0), stats

    from torch_geometric.data import Batch
    from src.dataset.graph_data import build_mips_graph_for_input, repeat_cut_augment_smiles

    graph_encoder = base_model.encoders['graph'].encoder
    aug_graphs = []
    valid_indices = []
    for sample_idx, smiles in enumerate(data.smiles):
        stats['attempted'] += 1
        aug_graph = None
        aug_smiles = None
        attempts = max(1, int(retry))
        for _ in range(attempts):
            try:
                aug_smiles, _ = repeat_cut_augment_smiles(
                    smiles,
                    max_mrus=max_mrus,
                    return_n=True,
                )
                aug_graph = build_mips_graph_for_input(aug_smiles, graph_input=graph_input)
                break
            except Exception:
                aug_graph = None
        if aug_graph is None:
            stats['skipped'] += 1
            continue
        if str(aug_smiles) == str(smiles):
            stats['same_smiles'] += 1
            stats['skipped'] += 1
            continue
        stats['success'] += 1
        aug_graphs.append(aug_graph)
        valid_indices.append(sample_idx)

    stats['valid_contrastive'] = len(valid_indices)
    if len(valid_indices) < 2:
        return data.x.new_tensor(0.0), stats

    # PerioGT augmentation changes the repeat-unit graph but does not provide a
    # synchronized PBC conformer.  SCAGE therefore uses topology distance for
    # both views of this auxiliary task, avoiding asymmetric geometry inputs.
    if getattr(graph_encoder, 'uses_geometry', False):
        orig_graph, _ = graph_encoder.forward_topology_only(data)
    else:
        orig_graph, _ = _graph_encode_nodes(base_model, data)
    valid_index = torch.tensor(valid_indices, dtype=torch.long, device=orig_graph.device)
    z_orig = orig_graph.index_select(0, valid_index)

    aug_batch = Batch.from_data_list(aug_graphs).to(data.x.device)
    if getattr(graph_encoder, 'uses_geometry', False):
        z_aug, _ = graph_encoder.forward_topology_only(aug_batch)
    else:
        z_aug, _ = graph_encoder(aug_batch.x, aug_batch.edge_index, aug_batch.edge_attr, aug_batch.batch)

    z_orig = F.normalize(z_orig, dim=-1)
    z_aug = F.normalize(z_aug, dim=-1)
    logits = torch.matmul(z_orig, z_aug.t()) / float(temperature)
    labels = torch.arange(logits.size(0), device=logits.device)
    loss = 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))
    return loss, stats


def _is_scage_graph_encoder(base_model):
    return (
        'graph' in base_model.encoders
        and getattr(
            base_model.encoders['graph'].encoder, 'architecture_name', ''
        ) == 'mips_localized_non_pbc'
    )


def _graph_encode_nodes(
    base_model, data, x_override=None, geometry_adapt=False
):
    graph_encoder = base_model.encoders['graph'].encoder
    if x_override is not None and hasattr(graph_encoder, 'forward_with_x'):
        if geometry_adapt:
            return graph_encoder.forward_geometry_with_x(data, x_override)
        return graph_encoder.forward_with_x(data, x_override)
    if getattr(graph_encoder, 'expects_data', False):
        return graph_encoder(data)
    if getattr(graph_encoder, 'uses_geometry', False):
        return graph_encoder(data)
    x = data.x if x_override is None else x_override
    return graph_encoder(x, data.edge_index, data.edge_attr, data.batch)


def _graph_encode_geometry_pretext(base_model, data):
    graph_encoder = base_model.encoders['graph'].encoder
    if not hasattr(graph_encoder, 'forward_geometry_pretext'):
        raise ValueError("geometry pretext encoding requires a SCAGE graph encoder")
    return graph_encoder.forward_geometry_pretext(data)


def _mean_loss_by_graph(per_target_loss, graph_ids):
    """Average targets within graph, then average graphs with valid targets."""
    graph_losses = []
    for graph_id in torch.unique(graph_ids, sorted=True):
        selected = graph_ids == graph_id
        if bool(selected.any()):
            graph_losses.append(per_target_loss[selected].mean())
    if not graph_losses:
        return per_target_loss.sum() * 0.0
    return torch.stack(graph_losses).mean()


def _mips_masked_atom_loss(
    base_model, data, prediction_head, mask_ratio, seed, epoch,
    geometry_adapt=False,
):
    """MIPS fused masked-atom prediction with deterministic per-polymer masks."""
    graph_encoder = base_model.encoders['graph'].encoder
    mask_policy = getattr(graph_encoder, "mask_policy", "canonical_exact")
    masks = []
    geometry_valid = _mcl_valid_graph_mask(data) if geometry_adapt else None
    for graph_idx, smiles in enumerate(data.smiles):
        if hasattr(data, "graph_available") and not bool(
            data.graph_available.flatten()[graph_idx].item()
        ):
            continue
        if geometry_adapt and not bool(geometry_valid[graph_idx]):
            continue
        node_indices = torch.nonzero(data.batch == graph_idx, as_tuple=False).flatten()
        if not node_indices.numel():
            continue
        digest = hashlib.sha256(f"{int(seed)}:{int(epoch)}:{smiles}".encode("utf-8")).digest()
        generator = torch.Generator(device='cpu')
        generator.manual_seed(int.from_bytes(digest[:8], 'little') % (2 ** 63 - 1))
        if mask_policy == "expanded_bernoulli":
            selected = (
                torch.rand(node_indices.numel(), generator=generator)
                < float(mask_ratio)
            ).to(node_indices.device)
            selected_nodes = node_indices[selected]
        elif mask_policy == "canonical_exact":
            canonical = getattr(data, "canonical_ru_atom_index", None)
            if canonical is None:
                canonical = torch.arange(
                    node_indices.numel(), device=node_indices.device
                )
            else:
                canonical = canonical[node_indices]
            groups = torch.unique(canonical, sorted=True)
            count = max(1, int(round(float(mask_ratio) * groups.numel())))
            selected = torch.randperm(groups.numel(), generator=generator)[:count]
            selected_groups = groups[selected.to(groups.device)]
            selected_nodes = node_indices[
                torch.isin(canonical, selected_groups)
            ]
        else:
            raise ValueError(f"unsupported MIPS mask policy: {mask_policy}")
        if not selected_nodes.numel():
            continue
        masks.append(selected_nodes)
    if not masks:
        return data.x.new_tensor(0.0), 0
    mask_indices = torch.cat(masks)
    x_override = data.x.clone()
    x_override[mask_indices] = 0.0
    _, node_rep = _graph_encode_nodes(
        base_model, data, x_override=x_override,
        geometry_adapt=geometry_adapt,
    )
    # Released MIPS derives labels from the first 101 entries of its 137-wide
    # atom feature vector (100 elements plus the unknown category).
    canonical = getattr(data, "canonical_ru_atom_index", None)
    if canonical is not None and mask_policy == "canonical_exact":
        # Count one representative per canonical atom even though every
        # equivalent copy was hidden from the encoder.
        representatives = []
        identities = torch.stack(
            [data.batch[mask_indices], canonical[mask_indices]], dim=1
        )
        for identity in torch.unique(identities, dim=0, sorted=True):
            matching = (
                (data.batch[mask_indices] == identity[0])
                & (canonical[mask_indices] == identity[1])
            )
            representatives.append(
                mask_indices[
                    torch.nonzero(matching, as_tuple=False)[0, 0]
                ]
            )
        target_indices = torch.stack(representatives)
    else:
        target_indices = mask_indices
    if getattr(graph_encoder, "masked_atom_target", "scage119") == "mips101":
        targets = data.mips_x[target_indices, :101].argmax(dim=-1).long()
    elif getattr(graph_encoder, "architecture_name", "") == "mips_localized_non_pbc":
        targets = data.atomic_num[target_indices].long()
    else:
        targets = data.mips_x[target_indices, :101].argmax(dim=-1).long()
    logits = prediction_head(node_rep[target_indices])
    per_target = F.cross_entropy(
        logits.float(), targets, reduction="none"
    )
    if getattr(graph_encoder, "masked_loss_reduction", "graph_mean") == "atom_mean":
        loss = per_target.mean()
    else:
        loss = _mean_loss_by_graph(per_target, data.batch[target_indices])
    return loss, int(target_indices.numel())


def _undirected_lga_relation_ids(data):
    """Return graph-local undirected identities for every LGA edge.

    Cached topology records created before this fix contain directed
    ``canonical_pair_index`` values.  Deriving the identity from the cached
    node mapping at runtime fixes reverse-edge leakage without invalidating
    the immutable topology LMDB.
    """
    source, target = data.lga_edge_index.long()
    canonical = data.canonical_ru_atom_index.long()
    graph = data.batch[target].long()
    left = torch.minimum(canonical[source], canonical[target])
    right = torch.maximum(canonical[source], canonical[target])
    keys = torch.stack((graph, left, right), dim=1)
    _, inverse = torch.unique(keys, dim=0, sorted=True, return_inverse=True)
    return inverse


def _mcl_valid_graph_mask(data):
    """Return the exact per-graph geometry mask used by Trimer-MCL."""
    cached = getattr(data, "mcl_valid", None)
    if cached is not None:
        return torch.as_tensor(
            cached, dtype=torch.bool, device=data.graph_available.device
        ).flatten()
    graph_available = data.graph_available.bool().flatten()
    return torch.tensor(
        [trimer_can_enter_mcl(data, graph_idx)
         for graph_idx in range(int(graph_available.numel()))],
        dtype=torch.bool,
        device=graph_available.device,
    )


def _select_mips_relation_targets(data, max_pairs):
    """Select one shared, undirected relation set for SPD and path losses."""
    valid = data.graph_available[data.batch[data.lga_edge_index[1]]].bool()
    edge_indices = torch.nonzero(valid, as_tuple=False).flatten()
    if not edge_indices.numel():
        return None
    selected_edges = []
    edge_graphs = data.batch[data.lga_edge_index[1, edge_indices]]
    for graph_id in torch.unique(edge_graphs, sorted=True):
        graph_edges = edge_indices[edge_graphs == graph_id]
        selection = _stratified_pair_selection(
            data.lga_spd[graph_edges], max_pairs=max_pairs
        )
        selected_edges.append(graph_edges[selection])
    if not selected_edges:
        return None
    edge_indices = torch.cat(selected_edges)
    relation_ids = _undirected_lga_relation_ids(data)
    selected_pairs = torch.unique(relation_ids[edge_indices])
    relation_mask = torch.isin(relation_ids, selected_pairs)
    # Pick the first selected edge for each undirected relation without a
    # Python loop.  The relation id is already graph-local, so this remains
    # safe when the same canonical pair occurs in different graphs.
    selected_ids = relation_ids[edge_indices]
    sorted_order = torch.argsort(selected_ids, stable=True)
    sorted_ids = selected_ids[sorted_order]
    first = torch.ones_like(sorted_ids, dtype=torch.bool)
    if sorted_ids.numel() > 1:
        first[1:] = sorted_ids[1:] != sorted_ids[:-1]
    representatives = edge_indices[sorted_order[first]]
    return relation_mask, representatives


def _mips_lga_relation_losses(base_model, data, spd_head, path_head, max_pairs):
    """Compute SPD and path/bond losses from one relation-corrupted forward."""
    selected = _select_mips_relation_targets(data, max_pairs)
    if selected is None:
        zero = data.x.new_tensor(0.0)
        return zero, 0, zero, 0
    relation_mask, edge_indices = selected
    data.lga_relation_mask = relation_mask
    try:
        _, node_rep = base_model.encoders["graph"].encoder.forward_relation_pretext(data)
    finally:
        delattr(data, "lga_relation_mask")
    source, target = data.lga_edge_index[:, edge_indices]
    representation = torch.cat(
        [node_rep[source], node_rep[target],
         torch.abs(node_rep[source] - node_rep[target])], dim=-1
    )
    spd_targets = data.lga_spd[edge_indices].long()
    if spd_head is not None:
        spd_logits = spd_head(representation)
        spd_per_target = F.cross_entropy(
            spd_logits.float(), spd_targets, reduction="none"
        )
        spd_loss = _mean_loss_by_graph(
            spd_per_target, data.batch[target]
        )
    else:
        spd_loss = representation.sum() * 0.0

    hist = data.lga_path_bond_hist[edge_indices]
    position_mask = hist.sum(dim=-1) > 0
    if path_head is not None and bool(position_mask.any()):
        path_logits = path_head(
            representation.unsqueeze(1).expand(
                -1, hist.size(1), -1
            ).reshape(-1, representation.size(-1))
        ).reshape(hist.size(0), hist.size(1), -1)
        path_targets = hist.argmax(dim=-1).long()
        path_per_target = F.cross_entropy(
            path_logits[position_mask].float(),
            path_targets[position_mask], reduction="none",
        )
        path_graph_ids = data.batch[target].unsqueeze(1).expand_as(position_mask)
        path_loss = _mean_loss_by_graph(
            path_per_target, path_graph_ids[position_mask]
        )
        path_count = int(position_mask.sum().item())
    else:
        path_loss = representation.sum() * 0.0
        path_count = 0
    return spd_loss, int(spd_targets.numel()), path_loss, path_count


def _mips_lga_spd_loss(base_model, data, prediction_head, max_pairs):
    """Compatibility wrapper for callers that request SPD only."""
    spd_loss, spd_count, _, _ = _mips_lga_relation_losses(
        base_model, data, prediction_head,
        None,
        max_pairs,
    )
    return spd_loss, spd_count


def _mips_path_bond_loss(base_model, data, prediction_head, max_pairs):
    """Compatibility wrapper for callers that request path/bond only."""
    # The standalone diagnostic entry point remains available; the production
    # stage calls _mips_lga_relation_losses directly.
    _, _, path_loss, path_count = _mips_lga_relation_losses(
        base_model, data, None,
        prediction_head, max_pairs,
    )
    return path_loss, path_count


def _mips_trimer_distance_loss(base_model, data, prediction_head, max_pairs):
    """Regress selected central-RU pair distances with those edges hidden.

    Pair identity is local to each graph.  The selected undirected pair is
    removed in both directions from every conformer's radius graph during the
    same forward pass, so the target distance cannot be read as an input edge.
    """
    required = (
        "trimer_positions", "trimer_conformer_mask", "trimer_geometry_valid",
        "trimer_central_atom_index", "trimer_central_atom_mask",
        "canonical_ru_atom_local_index",
    )
    if any(not hasattr(data, name) for name in required):
        return data.x.new_tensor(0.0), 0
    graph_count = int(data.graph_available.numel())
    atom_capacity = int(data.trimer_positions.size(2))
    hidden_pair_mask = torch.zeros(
        graph_count, atom_capacity, atom_capacity,
        dtype=torch.bool, device=data.x.device,
    )
    selected = []
    for graph_id in range(graph_count):
        if (
            not bool(data.graph_available[graph_id])
            or not bool(data.trimer_geometry_valid[graph_id])
            or not bool(data.trimer_conformer_mask[graph_id, 0])
        ):
            continue
        central = data.trimer_central_atom_index[
            graph_id, data.trimer_central_atom_mask[graph_id]
        ].long()
        if central.numel() < 2:
            continue
        positions = data.trimer_positions[graph_id, 0, central]
        distances = torch.cdist(positions, positions)
        source, target = torch.triu_indices(
            central.numel(), central.numel(), offset=1, device=data.x.device
        )
        keep = distances[source, target] < 6.0
        source, target = source[keep], target[keep]
        if not source.numel():
            continue
        order = torch.argsort(distances[source, target], stable=True)
        order = order[:int(max_pairs)]
        source, target = source[order], target[order]
        for local_source, local_target in zip(source.tolist(), target.tolist()):
            atom_source = int(central[local_source])
            atom_target = int(central[local_target])
            hidden_pair_mask[graph_id, atom_source, atom_target] = True
            hidden_pair_mask[graph_id, atom_target, atom_source] = True
            selected.append((
                graph_id, local_source, local_target,
                distances[local_source, local_target],
            ))
    if not selected:
        return data.x.new_tensor(0.0), 0
    data.trimer_distance_pair_mask = hidden_pair_mask
    try:
        _, node_rep = _graph_encode_nodes(base_model, data)
    finally:
        delattr(data, "trimer_distance_pair_mask")
    pair_representations = []
    targets = []
    target_graphs = []
    local_canonical = data.canonical_ru_atom_local_index.long()
    for graph_id, source_id, target_id, distance in selected:
        graph_nodes = torch.nonzero(
            data.batch == graph_id, as_tuple=False
        ).flatten()
        source_nodes = graph_nodes[local_canonical[graph_nodes] == source_id]
        target_nodes = graph_nodes[local_canonical[graph_nodes] == target_id]
        if not source_nodes.numel() or not target_nodes.numel():
            continue
        source_rep = node_rep[source_nodes[0]]
        target_rep = node_rep[target_nodes[0]]
        pair_representations.append(torch.cat((
            source_rep, target_rep, torch.abs(source_rep - target_rep)
        )))
        targets.append(distance)
        target_graphs.append(graph_id)
    if not pair_representations:
        return data.x.new_tensor(0.0), 0
    prediction = prediction_head(
        torch.stack(pair_representations)
    ).squeeze(-1)
    target = torch.stack(targets).float()
    per_target = F.smooth_l1_loss(
        prediction.float(), target, beta=0.5, reduction="none"
    )
    return (
        _mean_loss_by_graph(
            per_target,
            torch.tensor(target_graphs, device=data.x.device, dtype=torch.long),
        ),
        int(target.numel()),
    )


def _multiclass_focal_loss(logits, targets, gamma=2.0):
    if logits.numel() == 0:
        return logits.new_tensor(0.0)
    ce = F.cross_entropy(logits, targets.long(), reduction='none')
    pt = torch.exp(-ce)
    return (((1.0 - pt) ** float(gamma)) * ce).mean()


def _scage_ecfp_loss(graph_rep, data, ecfp_head, pos_weight=None):
    valid = getattr(data, 'polymer_ecfp_valid', None)
    target = getattr(data, 'polymer_ecfp_target', None)
    if valid is None or target is None:
        return graph_rep.new_tensor(0.0), 0
    valid = valid.flatten().bool()
    if not valid.any():
        return graph_rep.new_tensor(0.0), 0
    logits = ecfp_head(graph_rep[valid])
    return F.binary_cross_entropy_with_logits(
        logits, target[valid].float(), pos_weight=pos_weight
    ), int(valid.sum().item())


def _stratified_pair_selection(distances, max_pairs):
    if distances.numel() <= int(max_pairs):
        return torch.arange(distances.numel(), device=distances.device)
    buckets = [
        distances == 1,
        distances == 2,
        (distances >= 3) & (distances <= 5),
        (distances >= 6) & (distances <= 10),
        distances > 10,
    ]
    quota = max(1, int(max_pairs) // len(buckets))
    selected = []
    for mask in buckets:
        indices = torch.nonzero(mask, as_tuple=False).flatten()
        if indices.numel() > quota:
            indices = indices[torch.randperm(indices.numel(), device=indices.device)[:quota]]
        selected.append(indices)
    selected = torch.cat(selected) if selected else distances.new_empty(0, dtype=torch.long)
    remaining = int(max_pairs) - int(selected.numel())
    if remaining > 0:
        all_indices = torch.arange(distances.numel(), device=distances.device)
        available = all_indices[~torch.isin(all_indices, selected)]
        if available.numel() > remaining:
            available = available[torch.randperm(available.numel(), device=available.device)[:remaining]]
        selected = torch.cat([selected, available])
    return selected


def _scage_periodic_sp_loss(data, node_rep, sp_head, max_distance, max_pairs, gamma):
    reps, targets = [], []
    metadata_valid = getattr(data, 'star_link_metadata_valid', None)
    for graph_idx in torch.unique(data.batch, sorted=True).tolist():
        if metadata_valid is not None and not bool(metadata_valid.flatten()[graph_idx].item()):
            continue
        node_idx = torch.nonzero(data.batch == graph_idx, as_tuple=False).flatten()
        n = int(node_idx.numel())
        if n < 2:
            continue
        local = torch.full((data.batch.numel(),), -1, device=data.x.device, dtype=torch.long)
        local[node_idx] = torch.arange(n, device=data.x.device)
        distance = torch.full((n, n), float(max_distance), device=data.x.device)
        diagonal = torch.arange(n, device=data.x.device)
        distance[diagonal, diagonal] = 0
        src, dst = data.edge_index
        keep = (data.batch[src] == graph_idx) & (data.batch[dst] == graph_idx)
        ls, ld = local[src[keep]], local[dst[keep]]
        distance[ls, ld] = 1
        for k in range(n):
            distance = torch.minimum(distance, distance[:, k:k + 1] + distance[k:k + 1, :])
        row, col = torch.triu_indices(n, n, offset=1, device=data.x.device)
        pair_targets = distance[row, col].clamp(max=float(max_distance)).long()
        selection = _stratified_pair_selection(pair_targets, max_pairs=max_pairs)
        row, col, pair_targets = row[selection], col[selection], pair_targets[selection]
        left, right = node_rep[node_idx[row]], node_rep[node_idx[col]]
        reps.append(torch.cat([left, right, torch.abs(left - right)], dim=-1))
        targets.append(pair_targets)
    if not reps:
        return node_rep.new_tensor(0.0), 0
    reps = torch.cat(reps, dim=0)
    targets = torch.cat(targets, dim=0)
    return _multiclass_focal_loss(sp_head(reps), targets, gamma=gamma), int(targets.numel())


def _angle_value(p0, center, p2):
    v1, v2 = p0 - center, p2 - center
    denom = torch.linalg.vector_norm(v1) * torch.linalg.vector_norm(v2)
    if float(denom.detach().cpu().item()) <= 1e-8:
        return None
    return torch.acos(torch.clamp(torch.dot(v1, v2) / denom, -1.0, 1.0))


def _unsigned_dihedral(p0, p1, p2, p3):
    b0, b1, b2 = p1 - p0, p2 - p1, p3 - p2
    b1_norm = torch.linalg.vector_norm(b1)
    if float(b1_norm.detach().cpu().item()) <= 1e-8:
        return None
    b1u = b1 / b1_norm
    v = b0 - torch.dot(b0, b1u) * b1u
    w = b2 - torch.dot(b2, b1u) * b1u
    denom = torch.linalg.vector_norm(v) * torch.linalg.vector_norm(w)
    if float(denom.detach().cpu().item()) <= 1e-8:
        return None
    return torch.acos(torch.clamp(torch.dot(v, w) / denom, -1.0, 1.0))


def _signed_dihedral(p0, p1, p2, p3):
    b0, b1, b2 = p1 - p0, p2 - p1, p3 - p2
    b1_norm = torch.linalg.vector_norm(b1)
    if float(b1_norm.detach().cpu().item()) <= 1e-8:
        return None
    b1u = b1 / b1_norm
    v = b0 - torch.dot(b0, b1u) * b1u
    w = b2 - torch.dot(b2, b1u) * b1u
    if float(torch.linalg.vector_norm(v) * torch.linalg.vector_norm(w)) <= 1e-8:
        return None
    return torch.atan2(torch.dot(torch.cross(b1u, v, dim=0), w), torch.dot(v, w))


def _angle_bin(value, bins):
    return torch.clamp((value / torch.pi * int(bins)).long(), max=int(bins) - 1)


def _circular_regression_loss(prediction, target_angles):
    """Regress a periodic angle through its unit-circle representation."""
    prediction = prediction.float()
    target_angles = target_angles.to(device=prediction.device, dtype=prediction.dtype)
    target = torch.stack([target_angles.sin(), target_angles.cos()], dim=-1)
    prediction_norm = torch.linalg.vector_norm(prediction, dim=-1, keepdim=True).clamp_min(1e-6)
    direction = prediction / prediction_norm
    direction_loss = 1.0 - (direction * target).sum(dim=-1)
    magnitude_loss = (prediction_norm.squeeze(-1) - 1.0).pow(2)
    return direction_loss.mean() + 0.05 * magnitude_loss.mean()


def _balanced_shift_cross_entropy(logits, targets, balance_power):
    """Balance signed periodic-image classes without unstable full inverse weighting."""
    balance_power = float(balance_power)
    if balance_power <= 0:
        return F.cross_entropy(logits, targets)
    class_count = int(logits.size(-1))
    counts = torch.bincount(targets, minlength=class_count).to(logits)
    present = counts > 0
    if int(present.sum().item()) <= 1:
        return F.cross_entropy(logits, targets)
    weights = torch.zeros_like(counts)
    present_counts = counts[present]
    weights[present] = (present_counts.mean() / present_counts).pow(balance_power)
    weights[present] = weights[present].clamp(0.25, 4.0)
    weights[present] /= weights[present].mean().clamp_min(1e-8)
    return F.cross_entropy(logits, targets, weight=weights)


def _scage_screw_geometry_loss(
    data,
    graph_rep,
    node_rep,
    angle_head,
    torsion_head,
    distance_head,
    shift_head,
    screw_head,
    angle_bins,
    torsion_bins,
    boundary_angle_weight,
    boundary_torsion_weight,
    torsion_objective,
    distance_component_weight,
    angle_component_weight,
    screw_component_weight,
    shift_balance_power,
    gamma,
    max_pairs=128,
    image_cap=1,
):
    internal_reps, internal_targets = [], []
    boundary_reps, boundary_targets = [], []
    torsion_reps, torsion_targets = [], []
    distance_reps, distance_targets, shift_targets = [], [], []
    screw_reps, screw_targets = [], []
    valid_samples = 0
    for graph_idx in torch.unique(data.batch, sorted=True).tolist():
        has_screw = bool(data.screw_valid.flatten()[graph_idx].item())
        has_smer = bool(
            hasattr(data, 'smer_valid') and data.smer_valid.flatten()[graph_idx].item()
        )
        has_polygen = bool(
            hasattr(data, 'polygen_periodic_valid')
            and data.polygen_periodic_valid.flatten()[graph_idx].item()
        )
        if not (has_screw or has_smer or has_polygen):
            continue
        if not bool(data.star_link_metadata_valid.flatten()[graph_idx].item()):
            continue
        node_idx = torch.nonzero(data.batch == graph_idx, as_tuple=False).flatten()
        mapping = data.graph_to_geom_index[node_idx].long()
        if mapping.numel() != node_idx.numel() or (mapping < 0).any():
            continue
        pos = data.pos3d[mapping]
        if not torch.isfinite(pos).all():
            continue
        global_to_local = torch.full(
            (data.batch.numel(),), -1, device=data.x.device, dtype=torch.long
        )
        global_to_local[node_idx] = torch.arange(node_idx.numel(), device=data.x.device)
        pair_global = data.attachment_pair[graph_idx].long()
        pair_local = global_to_local[pair_global]
        if (pair_local < 0).any():
            continue
        left_boundary, right_boundary = int(pair_local[0]), int(pair_local[1])
        ptr0 = int(data.ordered_backbone_ptr[graph_idx].item())
        ptr1 = int(data.ordered_backbone_ptr[graph_idx + 1].item())
        path_global = data.ordered_backbone_path[ptr0:ptr1]
        path_local = global_to_local[path_global]
        if path_local.numel() < 2 or (path_local < 0).any():
            continue

        rotation = data.screw_rotation[graph_idx]
        translation = data.screw_translation[graph_idx]
        row, col = torch.triu_indices(pos.size(0), pos.size(0), offset=1, device=pos.device)
        pair_mask = getattr(data, 'scage_geometry_pair_mask', None)
        if pair_mask is not None:
            selected_mask = pair_mask[graph_idx, :pos.size(0), :pos.size(0)][row, col]
            row, col = row[selected_mask], col[selected_mask]
        same_distance = torch.linalg.vector_norm(pos[row] - pos[col], dim=-1)
        images = []
        shifts = []
        finite_images = None
        if has_screw:
            plus, minus = pos, pos
            for shift in range(1, int(image_cap) + 1):
                plus = plus @ rotation.transpose(0, 1) + translation
                minus = (minus - translation) @ rotation
                images.extend([plus, minus])
                shifts.extend([shift, -shift])
        elif has_polygen:
            active_axes = torch.nonzero(data.pbc[graph_idx].bool(), as_tuple=False).flatten()
            if active_axes.numel() == 1:
                vector_t = data.cell[graph_idx, int(active_axes[0])]
                for shift in range(1, int(image_cap) + 1):
                    images.extend([pos + shift * vector_t, pos - shift * vector_t])
                    shifts.extend([shift, -shift])
        elif hasattr(data, 'smer_image_pos3d'):
            finite_images = data.smer_image_pos3d[node_idx].permute(1, 0, 2)
            if finite_images.shape == (3, pos.size(0), 3) and torch.isfinite(finite_images).all():
                images = [finite_images[0], finite_images[2]]
                shifts = [-1, 1]
        if images and row.numel():
            cross = torch.stack([
                torch.linalg.vector_norm(pos[row] - image[col], dim=-1) for image in images
            ])
            nearest_cross = cross.min(dim=0).values
            all_distances = torch.cat([same_distance.unsqueeze(0), cross], dim=0)
            all_shifts = pos.new_tensor([0, *shifts], dtype=torch.long)
            nearest_all_idx = all_distances.min(dim=0).indices
            nearest_shift = all_shifts[nearest_all_idx]
            buckets = torch.clamp((nearest_cross / 2.0).floor().long() + 1, max=20)
            selection = _stratified_pair_selection(buckets, max_pairs=max_pairs)
            left_rep, right_rep = node_rep[node_idx[row[selection]]], node_rep[node_idx[col[selection]]]
            distance_reps.append(torch.cat([
                left_rep, right_rep, torch.abs(left_rep - right_rep), left_rep * right_rep
            ], dim=-1))
            distance_targets.append(torch.stack([
                same_distance[selection], nearest_cross[selection]
            ], dim=-1))
            shift_targets.append(nearest_shift[selection] + int(image_cap))

        if has_screw:
            trace = torch.trace(rotation)
            screw_angle = torch.acos(torch.clamp((trace - 1.0) / 2.0, -1.0, 1.0))
            axis = torch.stack([
                rotation[2, 1] - rotation[1, 2],
                rotation[0, 2] - rotation[2, 0],
                rotation[1, 0] - rotation[0, 1],
            ])
            axis_norm = axis.norm()
            if axis_norm < 1e-6:
                axis = translation / translation.norm().clamp_min(1e-8)
            else:
                axis = axis / axis_norm
            rise = torch.dot(axis, translation).abs()
            screw_reps.append(graph_rep[graph_idx])
            screw_targets.append(torch.stack([rise, screw_angle.sin(), screw_angle.cos()]))

        star_pair = {tuple(sorted((left_boundary, right_boundary)))}
        neighbors = [set() for _ in range(int(node_idx.numel()))]
        src, dst = data.edge_index
        keep = (data.batch[src] == graph_idx) & (data.batch[dst] == graph_idx)
        for source, target in zip(global_to_local[src[keep]].tolist(), global_to_local[dst[keep]].tolist()):
            if tuple(sorted((source, target))) in star_pair:
                continue
            neighbors[source].add(target)
        local_geometry_mask = None
        if pair_mask is not None:
            local_geometry_mask = pair_mask[
                graph_idx, :pos.size(0), :pos.size(0)
            ]

        def target_geometry_is_masked(atom_indices):
            if local_geometry_mask is None:
                return True
            atom_indices = list(dict.fromkeys(int(idx) for idx in atom_indices))
            for first_idx in range(len(atom_indices)):
                for second_idx in range(first_idx + 1, len(atom_indices)):
                    if not bool(local_geometry_mask[
                        atom_indices[first_idx], atom_indices[second_idx]
                    ].item()):
                        return False
            return True

        for center, adjacent in enumerate(neighbors):
            adjacent = sorted(adjacent)
            for first_idx in range(len(adjacent)):
                for second_idx in range(first_idx + 1, len(adjacent)):
                    left, right = adjacent[first_idx], adjacent[second_idx]
                    if not target_geometry_is_masked((left, center, right)):
                        continue
                    angle = _angle_value(pos[left], pos[center], pos[right])
                    if angle is None:
                        continue
                    internal_reps.append(torch.cat([
                        node_rep[node_idx[left]], node_rep[node_idx[center]], node_rep[node_idx[right]]
                    ]))
                    internal_targets.append(_angle_bin(angle, angle_bins))

        if path_local.numel() >= 3:
            left_neighbor = int(path_local[1].item())
            right_neighbor = int(path_local[-2].item())
            if has_screw:
                previous_right = (pos[right_boundary] - translation) @ rotation
                next_left = pos[left_boundary] @ rotation.transpose(0, 1) + translation
                next_left_neighbor = pos[left_neighbor] @ rotation.transpose(0, 1) + translation
            elif has_polygen:
                active_axes = torch.nonzero(data.pbc[graph_idx].bool(), as_tuple=False).flatten()
                if active_axes.numel() != 1:
                    continue
                vector_t = data.cell[graph_idx, int(active_axes[0])]
                previous_right = pos[right_boundary] - vector_t
                next_left = pos[left_boundary] + vector_t
                next_left_neighbor = pos[left_neighbor] + vector_t
            elif finite_images is not None:
                previous_right = finite_images[0, right_boundary]
                next_left = finite_images[2, left_boundary]
                next_left_neighbor = finite_images[2, left_neighbor]
            else:
                continue
            left_angle = _angle_value(previous_right, pos[left_boundary], pos[left_neighbor])
            right_angle = _angle_value(pos[right_neighbor], pos[right_boundary], next_left)
            if left_angle is not None and target_geometry_is_masked(
                (right_boundary, left_boundary, left_neighbor)
            ):
                boundary_reps.append(torch.cat([
                    node_rep[node_idx[right_boundary]], node_rep[node_idx[left_boundary]],
                    node_rep[node_idx[left_neighbor]]
                ]))
                boundary_targets.append(_angle_bin(left_angle, angle_bins))
            if right_angle is not None and target_geometry_is_masked(
                (right_neighbor, right_boundary, left_boundary)
            ):
                boundary_reps.append(torch.cat([
                    node_rep[node_idx[right_neighbor]], node_rep[node_idx[right_boundary]],
                    node_rep[node_idx[left_boundary]]
                ]))
                boundary_targets.append(_angle_bin(right_angle, angle_bins))
            torsion = _signed_dihedral(
                pos[right_neighbor], pos[right_boundary], next_left, next_left_neighbor
            )
            if torsion is not None and target_geometry_is_masked(
                (right_neighbor, right_boundary, left_boundary, left_neighbor)
            ):
                torsion_reps.append(torch.cat([
                    node_rep[node_idx[right_neighbor]], node_rep[node_idx[right_boundary]],
                    node_rep[node_idx[left_boundary]], node_rep[node_idx[left_neighbor]]
                ]))
                torsion_targets.append(torsion)
        valid_samples += 1

    geometry_components = {}
    if internal_targets:
        logits = angle_head(torch.stack(internal_reps))
        geometry_components['internal_angle'] = _multiclass_focal_loss(
            logits, torch.stack(internal_targets), gamma
        )
    if boundary_targets:
        logits = angle_head(torch.stack(boundary_reps))
        boundary_loss = _multiclass_focal_loss(logits, torch.stack(boundary_targets), gamma)
        if 'internal_angle' in geometry_components:
            geometry_components['angle'] = (
                geometry_components.pop('internal_angle') + float(boundary_angle_weight) * boundary_loss
            ) / (1.0 + float(boundary_angle_weight))
        else:
            geometry_components['angle'] = boundary_loss
    elif 'internal_angle' in geometry_components:
        geometry_components['angle'] = geometry_components.pop('internal_angle')
    if torsion_targets:
        logits = torsion_head(torch.stack(torsion_reps))
        torsion_values = torch.stack(torsion_targets)
        if torsion_objective == 'circular':
            geometry_components['torsion'] = _circular_regression_loss(
                logits, torsion_values
            )
        else:
            categorical_targets = torch.clamp(
                (((torsion_values + torch.pi) / (2.0 * torch.pi)) * int(torsion_bins)).long(),
                min=0, max=int(torsion_bins) - 1,
            )
            geometry_components['torsion'] = _multiclass_focal_loss(
                logits, categorical_targets, gamma
            )
    if distance_targets:
        pair_rep = torch.cat(distance_reps)
        distance_target = torch.cat(distance_targets)
        distance_loss = F.smooth_l1_loss(distance_head(pair_rep), distance_target, beta=0.5)
        shift_loss = _balanced_shift_cross_entropy(
            shift_head(pair_rep), torch.cat(shift_targets), shift_balance_power
        )
        geometry_components['distance'] = distance_loss + 0.2 * shift_loss
    if screw_targets:
        geometry_components['screw'] = F.smooth_l1_loss(
            screw_head(torch.stack(screw_reps)), torch.stack(screw_targets), beta=0.25
        )
    if not geometry_components:
        return node_rep.new_tensor(0.0), {
            'samples': 0, 'internal': 0, 'boundary': 0, 'torsion': 0,
            'distance_pairs': 0, 'shift_negative': 0, 'shift_center': 0,
            'shift_positive': 0, 'screw_ops': 0,
        }, {}
    component_priors = {
        'distance': float(distance_component_weight),
        'angle': float(angle_component_weight),
        'torsion': float(boundary_torsion_weight),
        'screw': float(screw_component_weight),
    }
    active_weight = sum(component_priors[name] for name in geometry_components)
    total = sum(
        component_priors[name] * value for name, value in geometry_components.items()
    ) / max(active_weight, 1e-8)
    signed_shift_targets = (
        torch.cat(shift_targets) - int(image_cap)
        if shift_targets else node_rep.new_empty(0, dtype=torch.long)
    )
    component_logs = {
        name: value.detach() for name, value in geometry_components.items()
    }
    if distance_targets:
        component_logs['distance_regression'] = distance_loss.detach()
        component_logs['image_shift'] = shift_loss.detach()
    return total, {
        'samples': valid_samples,
        'internal': len(internal_targets),
        'boundary': len(boundary_targets),
        'torsion': len(torsion_targets),
        'distance_pairs': sum(item.size(0) for item in distance_targets),
        'shift_negative': int((signed_shift_targets < 0).sum().item()),
        'shift_center': int((signed_shift_targets == 0).sum().item()),
        'shift_positive': int((signed_shift_targets > 0).sum().item()),
        'screw_ops': len(screw_targets),
    }, component_logs


def _make_scage_geometry_pair_mask(data, ratio):
    batch_size, max_nodes, _ = data.scage_spd.shape
    mask = torch.zeros((batch_size, max_nodes, max_nodes), dtype=torch.bool, device=data.x.device)
    for graph_idx in range(batch_size):
        has_screw = bool(data.screw_valid.flatten()[graph_idx].item())
        has_smer = bool(
            hasattr(data, 'smer_valid') and data.smer_valid.flatten()[graph_idx].item()
        )
        has_polygen = bool(
            hasattr(data, 'polygen_periodic_valid')
            and data.polygen_periodic_valid.flatten()[graph_idx].item()
        )
        if not (has_screw or has_smer or has_polygen) or float(ratio) <= 0:
            continue
        count = int((data.batch == graph_idx).sum().item())
        row, col = torch.triu_indices(count, count, offset=1, device=data.x.device)
        target_count = min(
            int(max(1, round(float(ratio) * row.numel()))),
            int(row.numel()),
        )
        if target_count <= 0:
            continue
        choice = torch.randperm(row.numel(), device=data.x.device)[:target_count]
        selected_row, selected_col = row[choice], col[choice]
        mask[graph_idx, selected_row, selected_col] = True
        mask[graph_idx, selected_col, selected_row] = True

        def mask_clique(atom_indices):
            atom_indices = list(dict.fromkeys(int(idx) for idx in atom_indices))
            for first_idx in range(len(atom_indices)):
                for second_idx in range(first_idx + 1, len(atom_indices)):
                    first = atom_indices[first_idx]
                    second = atom_indices[second_idx]
                    mask[graph_idx, first, second] = True
                    mask[graph_idx, second, first] = True

        node_idx = torch.nonzero(data.batch == graph_idx, as_tuple=False).flatten()
        global_to_local = torch.full(
            (data.batch.numel(),), -1, device=data.x.device, dtype=torch.long
        )
        global_to_local[node_idx] = torch.arange(count, device=data.x.device)
        pair_local = global_to_local[data.attachment_pair[graph_idx].long()]
        if (pair_local < 0).any():
            continue
        left_boundary, right_boundary = pair_local.tolist()
        star_pair = {tuple(sorted((left_boundary, right_boundary)))}
        neighbors = [set() for _ in range(count)]
        src, dst = data.edge_index
        keep = (data.batch[src] == graph_idx) & (data.batch[dst] == graph_idx)
        for source, target in zip(
            global_to_local[src[keep]].tolist(), global_to_local[dst[keep]].tolist()
        ):
            if tuple(sorted((source, target))) not in star_pair:
                neighbors[source].add(target)
        angle_candidates = []
        for center, adjacent in enumerate(neighbors):
            adjacent = sorted(adjacent)
            for first_idx in range(len(adjacent)):
                for second_idx in range(first_idx + 1, len(adjacent)):
                    angle_candidates.append((adjacent[first_idx], center, adjacent[second_idx]))
        if angle_candidates:
            angle_count = min(
                max(1, int(round(float(ratio) * len(angle_candidates)))),
                len(angle_candidates),
            )
            selected = torch.randperm(len(angle_candidates), device=data.x.device)[:angle_count]
            for candidate_idx in selected.tolist():
                mask_clique(angle_candidates[candidate_idx])

        ptr0 = int(data.ordered_backbone_ptr[graph_idx].item())
        ptr1 = int(data.ordered_backbone_ptr[graph_idx + 1].item())
        path_local = global_to_local[data.ordered_backbone_path[ptr0:ptr1]]
        if path_local.numel() >= 3 and not (path_local < 0).any():
            mask_clique((
                int(path_local[-2]), right_boundary,
                left_boundary, int(path_local[1]),
            ))
    return mask


def _multi_positive_supcon_loss(embeddings, identities, temperature):
    embeddings = F.normalize(embeddings, dim=-1)
    logits = embeddings @ embeddings.t() / float(temperature)
    identity_mask = identities[:, None] == identities[None, :]
    self_mask = torch.eye(logits.size(0), dtype=torch.bool, device=logits.device)
    positive_mask = identity_mask & ~self_mask
    logits = logits.masked_fill(self_mask, float('-inf'))
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    positive_count = positive_mask.sum(dim=1)
    valid = positive_count > 0
    if not valid.any():
        return embeddings.new_tensor(0.0)
    per_anchor = -(log_prob.masked_fill(~positive_mask, 0.0).sum(dim=1) / positive_count.clamp_min(1))
    return per_anchor[valid].mean()


def _repeat_cut_identity_key(value):
    """Normalize cached and online cut identities to stable hashable keys."""
    if value is None:
        return None
    if torch.is_tensor(value):
        value = value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return tuple(_repeat_cut_identity_key(item) for item in value)
    if isinstance(value, dict):
        return tuple(
            sorted(
                (str(key), _repeat_cut_identity_key(item))
                for key, item in value.items()
            )
        )
    try:
        hash(value)
    except TypeError:
        return repr(value)
    return value


def _scage_repeat_cut_consistency_loss(
    base_model, data, projection_head, graph_input, temperature, max_mrus, retry, views_per_polymer
):
    from src.dataset.graph_data import repeat_cut_augment_smiles
    from src.dataset.dataloader import custom_collate

    stats = {
        'identities_attempted': len(data.smiles), 'views_requested': len(data.smiles) * int(views_per_polymer),
        'views_valid': 0, 'duplicate_views': 0, 'failed_views': 0,
        'distinct_cut_views': 0, 'valid_identities': 0, 'positive_pairs': 0,
        'positive_count_sum': 0, 'graph_cache_hits': 0, 'graph_cache_misses': 0,
    }
    graph_encoder = base_model.encoders['graph'].encoder
    original_rep, _ = graph_encoder(data)
    augmented_graphs, augmented_ids, valid_original_ids = [], [], []
    for sample_idx, smiles in enumerate(data.smiles):
        unique_views = set()
        unique_cuts = set()
        cached_views = (
            list(data.repeat_cut_smiles[sample_idx])
            if hasattr(data, 'repeat_cut_smiles') else []
        )
        cached_cuts = (
            list(data.repeat_cut_identities[sample_idx])
            if hasattr(data, 'repeat_cut_identities') else []
        )
        for view_idx in range(int(views_per_polymer)):
            accepted = None
            # Feature-cache views are generated with max_mrus=3. Reusing them
            # under a different runtime cap silently violates the requested
            # augmentation policy and can reintroduce oversized 3-MRU graphs.
            if int(max_mrus) == 3 and view_idx < len(cached_views):
                try:
                    cached_cut = _repeat_cut_identity_key(
                        cached_cuts[view_idx] if view_idx < len(cached_cuts) else None
                    )
                    cached_graph, cache_hit = _cached_repeat_cut_graph(
                        cached_views[view_idx], graph_input,
                        mips_core=graph_encoder.core,
                        mips_max_hops=graph_encoder.max_hops,
                    )
                    stats['graph_cache_hits' if cache_hit else 'graph_cache_misses'] += 1
                    accepted = (str(cached_views[view_idx]), cached_graph, cached_cut)
                except Exception:
                    accepted = None
            for _attempt in range(max(1, int(retry))):
                if accepted is not None:
                    break
                try:
                    aug_smiles, _, metadata = repeat_cut_augment_smiles(
                        smiles, max_mrus=max_mrus, return_n=True, return_metadata=True
                    )
                    cut_identity = _repeat_cut_identity_key(
                        metadata.get('cut_identity')
                    )
                    if (
                        str(aug_smiles) == str(smiles)
                        or str(aug_smiles) in unique_views
                        or (cut_identity is not None and cut_identity in unique_cuts)
                    ):
                        stats['duplicate_views'] += 1
                        continue
                    augmented_graph, cache_hit = _cached_repeat_cut_graph(
                        aug_smiles, graph_input,
                        mips_core=graph_encoder.core,
                        mips_max_hops=graph_encoder.max_hops,
                    )
                    stats['graph_cache_hits' if cache_hit else 'graph_cache_misses'] += 1
                    accepted = (str(aug_smiles), augmented_graph, cut_identity)
                    break
                except Exception:
                    accepted = None
            if accepted is None:
                stats['failed_views'] += 1
                continue
            unique_views.add(accepted[0])
            if accepted[2] is not None:
                unique_cuts.add(accepted[2])
                stats['distinct_cut_views'] += 1
            augmented = accepted[1]
            augmented.smiles = accepted[0]
            augmented.input_ids_smiles = data.input_ids_smiles[sample_idx:sample_idx + 1].detach().cpu()
            augmented.attention_mask_smiles = data.attention_mask_smiles[sample_idx:sample_idx + 1].detach().cpu()
            augmented.fp = data.fp[sample_idx:sample_idx + 1].detach().cpu()
            augmented.y = data.y[sample_idx].detach().cpu().reshape(-1)
            augmented.z = torch.ones(1, dtype=torch.long)
            augmented.pos = torch.zeros(1, 3)
            augmented.pos_confs = torch.zeros(1, 1, 3)
            augmented.graph_to_geom_index = torch.full(
                (augmented.x.size(0),), -1, dtype=torch.long
            )
            augmented.geom_build_ok = False
            augmented.geom_coordinate_ok = False
            augmented.geom_context_id = 1
            augmented.screw_valid = False
            augmented.smer_valid = False
            augmented.polymer_ecfp_target = torch.zeros(2048)
            augmented.polymer_ecfp_valid = False
            augmented.polymer_ecfp_source = 'repeat_cut_view'
            descriptor_valid = bool(data.scage_descriptor_valid[sample_idx].item())
            augmented.scage_descriptor_valid_confs = torch.tensor([descriptor_valid])
            for descriptor_name in ('shape', 'usrcat', 'autocorr3d', 'rdf', 'morse', 'whim'):
                selected = getattr(data, f'scage_descriptor_{descriptor_name}')[sample_idx]
                setattr(
                    augmented,
                    f'scage_descriptor_{descriptor_name}_confs',
                    selected.detach().cpu().unsqueeze(0),
                )
            augmented_graphs.append(augmented)
            augmented_ids.append(sample_idx)
            stats['views_valid'] += 1
        if unique_views:
            valid_original_ids.append(sample_idx)
    if len(valid_original_ids) < 2 or not augmented_graphs:
        return original_rep.new_tensor(0.0), stats
    aug_batch = custom_collate(augmented_graphs, random_conformer=False).to(data.x.device)
    augmented_rep, _ = graph_encoder(aug_batch)
    representations, identities = [], []
    for sample_idx in valid_original_ids:
        representations.append(original_rep[sample_idx])
        identities.append(sample_idx)
    representations.extend(list(augmented_rep))
    identities.extend(augmented_ids)
    identities = torch.tensor(identities, dtype=torch.long, device=data.x.device)
    projected = projection_head(torch.stack(representations))
    stats['valid_identities'] = len(valid_original_ids)
    counts = torch.bincount(identities, minlength=len(data.smiles))
    stats['positive_pairs'] = int((counts * (counts - 1)).sum().item())
    stats['positive_count_sum'] = int((counts[counts > 0] - 1).sum().item())
    return _multi_positive_supcon_loss(projected, identities, temperature), stats


def _symmetric_infonce(left, right, temperature, valid_mask=None):
    if valid_mask is not None:
        valid_mask = valid_mask.to(device=left.device, dtype=torch.bool).view(-1)
        left = left[valid_mask]
        right = right[valid_mask]
    left = F.normalize(left, dim=-1)
    right = F.normalize(right, dim=-1)
    if _distributed_enabled():
        from torch.distributed.nn.functional import all_gather

        local_count = torch.tensor([left.size(0)], device=left.device, dtype=torch.long)
        gathered_counts = [torch.zeros_like(local_count) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_counts, local_count)
        counts = [int(value.item()) for value in gathered_counts]
        global_count = sum(counts)
        if global_count < 2:
            return (left.sum() + right.sum()) * 0.0
        max_count = max(counts)
        if left.size(0) < max_count:
            padding = left.new_zeros((max_count - left.size(0), left.size(1)))
            left_padded = torch.cat([left, padding], dim=0)
            right_padded = torch.cat([right, padding], dim=0)
        else:
            left_padded, right_padded = left, right
        gathered_left_parts = list(all_gather(left_padded))
        gathered_right_parts = list(all_gather(right_padded))
        gathered_left = torch.cat([
            part[:count] for part, count in zip(gathered_left_parts, counts)
        ], dim=0)
        gathered_right = torch.cat([
            part[:count] for part, count in zip(gathered_right_parts, counts)
        ], dim=0)
        if left.size(0) == 0:
            return (gathered_left.sum() + gathered_right.sum()) * 0.0
        offset = sum(counts[:dist.get_rank()])
        labels = torch.arange(left.size(0), device=left.device) + offset
        left_logits = left @ gathered_right.t() / float(temperature)
        right_logits = right @ gathered_left.t() / float(temperature)
        local_loss = 0.5 * (
            F.cross_entropy(left_logits.float(), labels)
            + F.cross_entropy(right_logits.float(), labels)
        )
        # DDP averages gradients over ranks. Rescale the local mean so the
        # resulting gradient equals a global per-sample mean for unequal counts.
        return local_loss * (dist.get_world_size() * left.size(0) / global_count)
    if left.size(0) < 2:
        return (left.sum() + right.sum()) * 0.0
    logits = left @ right.t() / float(temperature)
    labels = torch.arange(logits.size(0), device=logits.device)
    return 0.5 * (
        F.cross_entropy(logits.float(), labels) + F.cross_entropy(logits.t().float(), labels)
    )


def _scage_semantic_alignment_losses(base_model, data, args):
    if 'graph' not in base_model.encoders or len(base_model.encoders) < 2:
        raise ValueError("Parallel semantic alignment requires Graph plus another modality")
    embeddings = base_model.encode_modalities(data)
    shared_embeddings = base_model.shared_modality_embeddings
    private_embeddings = base_model.private_modality_embeddings
    modality_index = {name: idx for idx, name in enumerate(base_model.modality_list)}
    graph = shared_embeddings[:, modality_index['graph']]
    projected_graph = base_model.project_alignment('graph', graph)
    projected_smiles = (
        base_model.project_alignment(
            'smiles', shared_embeddings[:, modality_index['smiles']]
        ) if 'smiles' in modality_index else None
    )
    projected_fp = (
        base_model.project_alignment(
            'fp', shared_embeddings[:, modality_index['fp']]
        ) if 'fp' in modality_index else None
    )

    drop_probabilities = {
        'fp': args.alignment_fp_drop,
        'smiles': args.alignment_smiles_drop,
        'graph': args.alignment_graph_drop,
    }
    intrinsic_mask = base_model.intrinsic_availability_mask(
        data, device=embeddings.device
    )
    masks = [
        base_model.sample_availability_mask(
            embeddings.size(0), drop_probabilities,
            min_available=min(2, len(base_model.modality_list)),
            device=embeddings.device, intrinsic_mask=intrinsic_mask,
        )
        for _ in range(2)
    ]
    fused_views = []
    for mask in masks:
        fused, _ = base_model.fuse_embeddings(embeddings, availability_mask=mask)
        fused_views.append(base_model.project_alignment('fusion', fused))
    fused_view_loss = _symmetric_infonce(fused_views[0], fused_views[1], args.temperature)

    full_mask = intrinsic_mask
    full_fused, full_weights = base_model.fuse_embeddings(embeddings, availability_mask=full_mask)
    teacher = F.normalize(base_model.project_alignment('fusion', full_fused), dim=-1).detach()
    lomo_losses = []
    for missing_name in base_model.modality_list:
        present = full_mask[:, modality_index[missing_name]]
        if not bool(present.any()):
            continue
        lomo_mask = full_mask.clone()
        lomo_mask[:, modality_index[missing_name]] = False
        student, _ = base_model.fuse_embeddings(embeddings, availability_mask=lomo_mask)
        student = F.normalize(base_model.project_alignment('fusion', student), dim=-1)
        lomo_losses.append(1.0 - (student[present] * teacher[present]).sum(dim=-1).mean())
    lomo_loss = (
        torch.stack(lomo_losses).mean()
        if lomo_losses else embeddings.sum() * 0.0
    )

    prior_by_name = {'smiles': 0.30, 'graph': 0.40, 'fp': 0.30}
    prior = full_weights.new_tensor([prior_by_name[name] for name in base_model.modality_list])
    sample_prior = prior.unsqueeze(0) * full_mask.to(dtype=full_weights.dtype)
    sample_prior = sample_prior / sample_prior.sum(dim=1, keepdim=True).clamp_min(1e-8)
    safe_weights = full_weights.clamp_min(1e-8)
    pooling_kl = (
        safe_weights
        * (safe_weights.log() - sample_prior.clamp_min(1e-8).log())
        * full_mask
    ).sum(dim=1).mean()
    mean_weights = full_weights.mean(dim=0)
    graph_valid = full_mask[:, modality_index['graph']]
    shared_private_orthogonal = torch.stack([
        F.cosine_similarity(
            shared_embeddings[:, idx],
            private_embeddings[:, idx],
            dim=-1,
        ).square().mean()
        for idx in range(shared_embeddings.size(1))
    ]).mean()
    losses = {
        'fused_view': fused_view_loss,
        'graph_smiles': (
            _symmetric_infonce(
                projected_graph, projected_smiles, args.temperature,
                valid_mask=graph_valid,
            ) if projected_smiles is not None else embeddings.sum() * 0.0
        ),
        'graph_fp': (
            _symmetric_infonce(
                projected_graph, projected_fp, args.temperature,
                valid_mask=graph_valid,
            ) if projected_fp is not None else embeddings.sum() * 0.0
        ),
        'lomo': lomo_loss,
        'pooling_kl': pooling_kl,
        'shared_private': shared_private_orthogonal,
    }
    missing_rates = {
        name: float(torch.stack([~mask[:, idx] for mask in masks]).float().mean().item())
        for idx, name in enumerate(base_model.modality_list)
    }
    stats = {
        'missing_rates': missing_rates,
        'pooling_entropy': float(
            (-(full_weights.clamp_min(1e-8) * full_weights.clamp_min(1e-8).log()).sum(dim=1).mean()).item()
        ),
        'pooling_weights': {
            name: float(mean_weights[idx].item())
            for idx, name in enumerate(base_model.modality_list)
        },
    }
    return losses, stats


def _joint_canonical_mask(data, seed, stream_step, mask_ratio):
    """Vectorized stateless mask over canonical atoms.

    Keys depend only on sample identity, local canonical atom id, optimizer
    stream step and seed, so rank assignment and batch composition cannot
    change the selected atoms.
    """
    device = data.mips_x.device
    canonical_periodic = bool(
        getattr(data, "mts_canonical_periodic", False)
        or getattr(data, "mips_local_lga_schema_version", 0) == 2
    )
    if canonical_periodic:
        # One mask decision per canonical node.  The Trimer collator lifts
        # these states to all three RU copies downstream.
        canonical = torch.arange(
            int(data.mips_x.size(0)), device=data.mips_x.device,
            dtype=torch.long,
        )
    else:
        canonical = data.canonical_ru_atom_index.long()
    graph_available = torch.as_tensor(
        getattr(data, "graph_available", torch.ones(len(data.smiles))),
        device=device,
    ).bool().flatten()
    required = (
        "canonical_graph_index", "canonical_local_index",
        "canonical_first_node_index", "mts_sample_hash64",
    )
    if not all(hasattr(data, name) for name in required):
        raise ValueError(
            "MTS joint pretraining requires vectorized canonical metadata "
            "from mips_trimer_collate"
        )
    graph_ids = data.canonical_graph_index.long()
    local_ids = data.canonical_local_index.long()
    sample_hash = data.mts_sample_hash64.long()[graph_ids]
    values = sample_hash.clone()
    values ^= local_ids * 6364136223846793005
    values ^= (int(stream_step) * 1442695040888963407) & ((1 << 63) - 1)
    values ^= (int(seed) * 2862933555777941757) & ((1 << 63) - 1)
    values ^= values >> 30
    values *= 3935559000370003845
    values ^= values >> 27
    values *= 2691343689449507681
    values ^= values >> 31
    scores = (values & ((1 << 53) - 1)).to(torch.float64) / float(1 << 53)
    canonical_selected = (scores < float(mask_ratio)) & graph_available[graph_ids]

    selected_counts = torch.zeros(
        graph_available.numel(), dtype=torch.long, device=device
    )
    selected_counts.index_add_(0, graph_ids, canonical_selected.long())
    minimum = torch.full(
        (graph_available.numel(),), float("inf"), dtype=scores.dtype, device=device
    )
    minimum.scatter_reduce_(0, graph_ids, scores, reduce="amin", include_self=True)
    empty = graph_available & (selected_counts == 0)
    canonical_selected |= empty[graph_ids] & (scores == minimum[graph_ids])
    return canonical_selected[canonical]


def _joint_masked_atom_terms(data, node_rep, prediction_head, atom_mask):
    if not hasattr(data, "canonical_first_node_index"):
        raise ValueError("Missing canonical_first_node_index in MTS batch")
    representatives = data.canonical_first_node_index.long()
    # Canonical topology has one node per atom, so representatives are the
    # nodes themselves.  Explicit test-reference batches still use the first
    # copy mapping supplied by the legacy collator.
    if atom_mask.numel() == node_rep.size(0) and representatives.numel() == node_rep.size(0):
        target_indices = torch.nonzero(atom_mask, as_tuple=False).flatten()
    else:
        target_indices = representatives[atom_mask[representatives]]
    if target_indices.numel() == 0:
        zero = node_rep.sum() * 0.0
        return zero, zero.detach(), 0, 0
    targets = data.mips_x[target_indices, :101].argmax(dim=-1).long()
    logits = prediction_head(node_rep[target_indices])
    per_target = F.cross_entropy(logits.float(), targets, reduction="none")
    correct = int((logits.detach().argmax(dim=-1) == targets).sum().item())
    return (
        per_target.sum(), per_target.detach().sum(), int(per_target.numel()), correct
    )


def _multiclass_focal_terms(logits, targets, alpha, gamma=2.0):
    log_prob = F.log_softmax(logits.float(), dim=-1)
    selected_log_prob = log_prob.gather(1, targets.long().view(-1, 1)).squeeze(1)
    probability = selected_log_prob.exp()
    weights = alpha.to(logits.device, dtype=logits.dtype)[targets.long()]
    return -weights * (1.0 - probability).pow(float(gamma)) * selected_log_prob


class TrimerAngleHead(nn.Module):
    """SCAGE-style 20-bin angle head with DDP-safe LayerNorm."""

    def __init__(self, dim=512, hidden=256, bins=20, alpha=None, dropout=0.10,
                 objective='categorical'):
        super().__init__()
        self.objective = str(objective)
        output_dim = int(bins) if self.objective == 'categorical' else 1
        self.net = nn.Sequential(
            nn.Linear(int(dim), int(hidden)),
            nn.LayerNorm(int(hidden)),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden), output_dim),
        )
        if alpha is None:
            alpha = torch.ones(int(bins), dtype=torch.float)
        self.register_buffer("alpha", torch.as_tensor(alpha, dtype=torch.float))
        self.bins = int(bins)

    def forward(self, values):
        return self.net(values)


def _angle_alpha_from_dataset(dataset, bins=20):
    """Build the fixed SCAGE-style inverse-frequency focal weights."""
    counts = getattr(dataset, "angle_class_counts", None)
    if counts is None:
        store = getattr(dataset, "_lazy_feature_store", None)
        counts = getattr(store, "angle_class_counts", None)
    if counts is None:
        raise RuntimeError(
            "MTS joint pretraining requires the frozen bond-angle cache "
            "with angle_class_counts.npy"
        )
    counts = np.asarray(counts, dtype=np.int64).reshape(-1)
    if counts.shape != (int(bins),):
        raise RuntimeError(
            f"angle class histogram must have shape [{int(bins)}], got {counts.shape}"
        )
    raw = np.log(200000.0 / (counts.astype(np.float64) + 1.0) + 1.0)
    raw = raw / max(float(raw.mean()), 1e-12)
    return torch.as_tensor(raw, dtype=torch.float32)


def _joint_angle_terms(data, trimer_states, angle_head, gamma=2.0):
    if not hasattr(data, "trimer_angle_index"):
        zero = trimer_states.sum() * 0.0
        return zero, zero.detach(), 0, 0, 0, zero.detach()
    indices = data.trimer_angle_index.long()
    objective = getattr(angle_head, 'objective', 'categorical')
    bins = getattr(data, 'trimer_angle_bins', None)
    cosine_targets = getattr(data, 'trimer_angle_cos', None)
    ptr = getattr(data, "trimer_angle_ptr", None)
    valid_graph = getattr(data, "mcl_valid", None)
    if ptr is None or indices.numel() == 0:
        zero = trimer_states.sum() * 0.0
        return zero, zero.detach(), 0, 0, 0, zero.detach()
    graph_total = int(ptr.numel()) - 1
    counts = (ptr[1:] - ptr[:-1]).long()
    graph_ids = torch.repeat_interleave(
        torch.arange(graph_total, device=indices.device), counts
    )
    selected = torch.ones_like(graph_ids, dtype=torch.bool)
    if valid_graph is not None:
        selected &= torch.as_tensor(
            valid_graph, device=indices.device, dtype=torch.bool
        )[graph_ids]
    triplets = indices[selected]
    if objective == 'categorical':
        if bins is None or bins.numel() != indices.size(0):
            raise RuntimeError('categorical angle objective requires angle bins')
        targets = bins.long()[selected]
    else:
        if cosine_targets is None or cosine_targets.numel() != indices.size(0):
            raise RuntimeError('cosine angle objective requires Angle-v2 targets')
        targets = cosine_targets.float()[selected]
    selected_graphs = graph_ids[selected]
    if triplets.numel() == 0:
        zero = trimer_states.sum() * 0.0
        return zero, zero.detach(), 0, 0, 0, zero.detach()
    if bool((triplets < 0).any()) or bool(
        (triplets >= trimer_states.size(0)).any()
    ):
        raise ValueError("Trimer angle index is outside the collated Trimer range")
    representation = (
        trimer_states[triplets[:, 0]]
        + trimer_states[triplets[:, 1]]
        + trimer_states[triplets[:, 2]]
    )
    logits = angle_head(representation)
    if objective == 'categorical':
        per_angle = _multiclass_focal_terms(
            logits, targets, angle_head.alpha, gamma=gamma
        )
        correct_count = int(
            (logits.detach().argmax(dim=-1) == targets).sum().item()
        )
        per_angle_mae = per_angle.detach()
    else:
        predictions = logits.float().reshape(-1).clamp(-1.0, 1.0)
        per_angle = F.smooth_l1_loss(
            predictions, targets.float(), reduction='none'
        )
        # Retain the existing packed-count/logging interface: for continuous
        # targets, "accuracy" means |cos prediction error| <= 0.1.
        correct_count = int(
            ((predictions.detach() - targets).abs() <= 0.1).sum().item()
        )
        per_angle_mae = (predictions.detach() - targets).abs()
    graph_loss_sum = per_angle.new_zeros(graph_total)
    graph_loss_count = per_angle.new_zeros(graph_total)
    graph_loss_sum.index_add_(0, selected_graphs, per_angle)
    graph_loss_count.index_add_(
        0, selected_graphs, torch.ones_like(per_angle)
    )
    active = graph_loss_count > 0
    graph_means = graph_loss_sum[active] / graph_loss_count[active]
    graph_mae_sum = per_angle_mae.new_zeros(graph_total)
    graph_mae_sum.index_add_(0, selected_graphs, per_angle_mae)
    graph_mae_means = graph_mae_sum[active] / graph_loss_count[active]
    return (
        graph_means.sum(), graph_means.detach().sum(),
        int(active.sum().item()), correct_count, int(targets.numel()),
        graph_mae_means.sum(),
    )


def _bf16_joint_parity_gate(base_model, data, atom_head, angle_head, args):
    """Compare the complete one-forward joint objective in FP32/BF16."""
    mask = _joint_canonical_mask(data, args.seed, 0, args.graph_mask_ratio)

    def evaluate(dtype=None):
        base_model.zero_grad(set_to_none=True)
        atom_head.zero_grad(set_to_none=True)
        angle_head.zero_grad(set_to_none=True)
        context = (
            torch.autocast(device_type="cuda", dtype=dtype)
            if dtype is not None else nullcontext()
        )
        with context:
            _, nodes, aux = base_model.encoders["graph"].encoder.forward_joint_pretrain(
                data, mask
            )
            atom_sum, _, atom_count, _ = _joint_masked_atom_terms(
                data, nodes, atom_head, mask
            )
            if str(getattr(args, "pretraining_objective", "joint")) == "masked_atom_only":
                angle_sum = atom_sum.new_zeros(())
                angle_count = 0
            else:
                angle_sum, _, angle_count, _, _, _ = _joint_angle_terms(
                    data, aux["final_trimer_states"], angle_head,
                    gamma=float(args.scage_focal_gamma),
                )
            atom_loss = atom_sum / max(1, atom_count)
            angle_loss = angle_sum / max(1, angle_count)
            loss = atom_loss + float(args.graph_angle_weight) * angle_loss
        loss.backward()
        grad = _gradient_vector((base_model, atom_head, angle_head))
        return loss.detach(), grad.detach()

    fp32_loss, fp32_grad = evaluate(None)
    bf16_loss, bf16_grad = evaluate(torch.bfloat16)
    delta = float(
        (bf16_loss.float() - fp32_loss.float()).abs()
        / fp32_loss.float().abs().clamp_min(1e-8)
    )
    cosine = float(F.cosine_similarity(fp32_grad, bf16_grad, dim=0).item())
    finite = bool(torch.isfinite(bf16_loss) and torch.isfinite(bf16_grad).all())
    base_model.zero_grad(set_to_none=True)
    atom_head.zero_grad(set_to_none=True)
    angle_head.zero_grad(set_to_none=True)
    return finite and delta <= 0.02 and cosine >= 0.98, {
        "relative_loss_delta": delta, "gradient_cosine": cosine, "finite": finite,
    }


@torch.no_grad()
def _evaluate_angle_v2_validation(train_module, loader, args, device):
    """Evaluate the deterministic 1% holdout on every DDP rank."""
    was_training = train_module.training
    train_module.eval()
    totals = torch.zeros(4, device=device, dtype=torch.float64)
    for data in loader:
        data = data.to(device)
        with torch.autocast(
            device_type='cuda', dtype=torch.bfloat16,
            enabled=args.amp_dtype == 'bf16' and device.type == 'cuda',
        ):
            payload = train_module(MTS_STAGE1_ID, data, args, 0)
        totals[0] += payload['detached_terms']['masked_atom_sum'].double()
        totals[1] += float(payload['counts']['masked_atoms'])
        totals[2] += payload['detached_terms']['angle_mae_sum'].double()
        totals[3] += float(payload['counts']['angle_graphs'])
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
    if was_training:
        train_module.train()
    atom_ce = float((totals[0] / totals[1].clamp_min(1.0)).item())
    angle_mae = float((totals[2] / totals[3].clamp_min(1.0)).item())
    if not np.isfinite(atom_ce) or not np.isfinite(angle_mae):
        raise RuntimeError('non-finite Angle-v2 validation metric')
    return atom_ce, angle_mae


def _fusion_conditioned_masked_atom_loss(base_model, data, prediction_head, mask_ratio):
    mask_indices = []
    graph_available = getattr(data, 'graph_available', None)
    if graph_available is None:
        graph_available = torch.ones(
            len(data.smiles), dtype=torch.bool, device=data.x.device
        )
    else:
        graph_available = torch.as_tensor(
            graph_available, device=data.x.device
        ).view(-1).bool()
    for graph_idx in range(len(data.smiles)):
        if not bool(graph_available[graph_idx]):
            continue
        nodes = torch.nonzero(data.batch == graph_idx, as_tuple=False).flatten()
        if not nodes.numel():
            continue
        count = max(1, int(round(float(mask_ratio) * nodes.numel())))
        mask_indices.append(nodes[torch.randperm(nodes.numel(), device=nodes.device)[:count]])
    if not mask_indices:
        return data.x.new_tensor(0.0), 0
    mask_indices = torch.cat(mask_indices)
    masked_x = data.x.clone()
    masked_x[mask_indices] = 0.0
    graph_module = base_model.encoders['graph']
    raw_graph, raw_nodes = graph_module.encoder.forward_with_x(data, masked_x)
    masked_graph = graph_module.projection(graph_module.norm(raw_graph))
    modality_embeddings = []
    for name in base_model.modality_list:
        value = masked_graph if name == 'graph' else base_model.encoders[name](data)
        if base_model.projection_mode == "shared_private":
            value, _, _ = base_model.shared_private_projections[name](value)
        modality_embeddings.append(value)
    embeddings = torch.stack(modality_embeddings, dim=1)
    intrinsic_mask = base_model.intrinsic_availability_mask(
        data, device=embeddings.device
    )
    fused, _ = base_model.fuse_embeddings(
        embeddings, availability_mask=intrinsic_mask
    )
    context = fused[data.batch[mask_indices]]
    logits = prediction_head(torch.cat([raw_nodes[mask_indices], context], dim=-1))
    targets = data.atomic_num[mask_indices].long()
    return F.cross_entropy(logits.float(), targets), int(mask_indices.numel())


class MIPSPretrainContainer(nn.Module):
    """One DDP-visible module containing the encoder and every Stage 1 head.

    Calling encoder submodules behind DDP bypasses reducer preparation.  This
    container keeps the existing loss helpers while ensuring every trainable
    tensor is reached from one real DDP forward.
    """

    def __init__(self, model, heads=None):
        super().__init__()
        self.model = model
        self.heads = nn.ModuleDict(dict(heads or {}))

    def forward(self, stage, data, args, epoch=0):
        if stage == "mts_joint_pretraining":
            return self._mts_joint_pretraining(data, args, epoch)
        if stage in {"mips_stage1", "stage2_geometry_adapt"}:
            return self._mips_stage1(
                data, args, epoch,
                geometry_adapt=stage == "stage2_geometry_adapt"
            )
        if stage == "alignment":
            losses, stats = _scage_semantic_alignment_losses(
                self.model, data, args
            )
            fused_mask, fused_count = _fusion_conditioned_masked_atom_loss(
                self.model, data, self.model.alignment_mask_head,
                args.graph_mask_ratio,
            )
            losses["fused_mask"] = fused_mask
            return {
                "losses": losses,
                "stats": stats,
                "fused_mask_count": fused_count,
            }
        raise ValueError(f"unsupported MIPS DDP stage: {stage}")

    def _mts_joint_pretraining(self, data, args, stream_step):
        """One shared O8+Trimer-MCL forward for both MTS pretext tasks."""
        mask = _joint_canonical_mask(
            data, args.seed, stream_step, args.graph_mask_ratio
        )
        graph, node_states, aux = self.model.encoders["graph"].encoder.forward_joint_pretrain(
            data, canonical_atom_mask=mask
        )
        atom_sum, atom_detached, atom_count, atom_correct = _joint_masked_atom_terms(
            data, node_states, self.heads["mips_atom"], mask
        )
        if str(getattr(args, "pretraining_objective", "joint")) == "masked_atom_only":
            # G-family input geometry is deliberately not reused as an
            # Angle-20 target.  Keep the angle head in the strict checkpoint
            # layout and zero-anchor it for DDP, but never inspect angle
            # labels or invoke the angle loss helper in this objective.
            angle_sum = atom_sum.new_zeros(())
            angle_detached = angle_sum.detach()
            angle_graph_count = 0
            angle_correct = 0
            angle_targets = 0
            angle_mae_sum = angle_detached
        else:
            angle_sum, angle_detached, angle_graph_count, angle_correct, angle_targets, angle_mae_sum = _joint_angle_terms(
                data,
                aux["final_trimer_states"],
                self.heads["angle"],
                gamma=float(args.scage_focal_gamma),
            )
        # The zero anchors are only for the rare rank-local empty target case;
        # they do not alter any numerical loss value or gradient of active
        # parameters.
        zero_reference = atom_sum * 0.0 + angle_sum * 0.0
        for head_name in ("mips_atom", "angle"):
            for parameter in self.heads[head_name].parameters():
                zero_reference = zero_reference + parameter.reshape(-1).sum() * 0.0
        return {
            "graph": graph,
            "loss_terms": {
                "masked_atom_sum": atom_sum,
                "angle_sum": angle_sum,
            },
            "detached_terms": {
                "masked_atom_sum": atom_detached,
                "angle_sum": angle_detached,
                "angle_mae_sum": angle_mae_sum.detach(),
            },
            "counts": {
            "masked_atoms": int(atom_count),
            "angle_graphs": int(angle_graph_count),
            "masked_correct": int(atom_correct),
            "angle_correct": int(angle_correct),
            "angle_targets": int(angle_targets),
            "mcl_valid_graphs": int(aux["mcl_valid_graph_mask"].sum().item()),
            "graphs": int(data.graph_available.numel()),
            },
            "mcl_valid": aux["mcl_valid_graph_mask"],
            "angle_valid": aux["angle_valid_graph_mask"],
            "zero_reference": zero_reference,
        }

    def _mips_stage1(self, data, args, epoch, geometry_adapt=False):
        zero = data.x.new_tensor(0.0)
        mask_loss, mask_count = _mips_masked_atom_loss(
            self.model, data, self.heads["mips_atom"],
            mask_ratio=args.graph_mask_ratio, seed=args.seed, epoch=epoch,
            geometry_adapt=geometry_adapt,
        )
        loss_terms = {}
        counts = {}
        if mask_count and float(args.scage_mips_mask_weight) > 0:
            loss_terms["mips_mask"] = mask_loss
            counts["masked_atoms"] = mask_count

        sp_loss, sp_count = zero, 0
        path_loss, path_count = zero, 0
        if not geometry_adapt and (
            float(args.mips_spd_weight) > 0
            or float(args.mips_path_bond_weight) > 0
        ):
            # SPD and path/bond share one relation-corrupted O8 forward.  The
            # atom task remains a separate masked-atom view so relation labels
            # cannot alter its semantics.
            sp_loss, sp_count, path_loss, path_count = _mips_lga_relation_losses(
                self.model, data,
                self.heads["masked_spd"] if float(args.mips_spd_weight) > 0 else None,
                self.heads["path_bond"] if float(args.mips_path_bond_weight) > 0 else None,
                max_pairs=args.scage_sp_max_pairs,
            )
        if sp_count:
            loss_terms["masked_spd"] = sp_loss
            counts["sp_pairs"] = sp_count
        if path_count:
            loss_terms["path_bond"] = path_loss
            counts["path_bond_pairs"] = path_count

        # Stage 2 intentionally does not train the SPD/path heads.  Keep only
        # the small head collection in the autograd graph with an exact zero
        # so a fixed ``find_unused_parameters=False`` reducer is safe.  The
        # complete UniEncoderAttention object is frozen/configured by the
        # caller; anchoring every unrelated Trimer/MD/fusion parameter here
        # was a substantial per-microbatch overhead.
        zero_reference = mask_loss * 0.0
        for parameter in self.heads.parameters():
            if parameter.requires_grad:
                zero_reference = zero_reference + parameter.reshape(-1).sum() * 0.0
        if geometry_adapt:
            for name in ("masked_spd", "path_bond"):
                for parameter in self.heads[name].parameters():
                    zero_reference = zero_reference + parameter.reshape(-1).sum() * 0.0

        return {
            "loss_terms": loss_terms,
            "counts": counts,
            "mask_loss": mask_loss,
            "spd_loss": sp_loss,
            "path_loss": path_loss,
            # Keeps the graph representation in the DDP output tree even when
            # a batch has no valid auxiliary target.
            "zero_reference": zero_reference,
        }


def _alignment_total(losses, args):
    return (
        args.alignment_fused_weight * losses['fused_view']
        + args.alignment_graph_smiles_weight * losses['graph_smiles']
        + args.alignment_graph_fp_weight * losses['graph_fp']
        + args.alignment_lomo_weight * losses['lomo']
        + args.alignment_pooling_kl_weight * losses['pooling_kl']
        + args.alignment_fused_mask_weight * losses.get('fused_mask', losses['fused_view'].new_tensor(0.0))
        + args.alignment_shared_private_weight * losses['shared_private']
    )


def _bf16_alignment_parity_gate(base_model, data, args):
    base_model.zero_grad(set_to_none=True)
    torch.manual_seed(args.seed)
    fp32_loss = _alignment_total(_scage_semantic_alignment_losses(base_model, data, args)[0], args)
    fp32_loss.backward()
    fp32_grad = _gradient_vector((base_model,))
    base_model.zero_grad(set_to_none=True)
    torch.manual_seed(args.seed)
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        bf16_loss = _alignment_total(
            _scage_semantic_alignment_losses(base_model, data, args)[0], args
        )
    bf16_loss.backward()
    bf16_grad = _gradient_vector((base_model,))
    delta = float(
        (bf16_loss.float() - fp32_loss.float()).abs()
        / fp32_loss.detach().float().abs().clamp_min(1e-8)
    )
    cosine = float(F.cosine_similarity(fp32_grad, bf16_grad, dim=0).item())
    finite = bool(torch.isfinite(bf16_loss) and torch.isfinite(bf16_grad).all())
    base_model.zero_grad(set_to_none=True)
    return finite and delta <= 0.02 and cosine >= 0.98, {
        'relative_loss_delta': delta, 'gradient_cosine': cosine, 'finite': finite,
    }


def _scage_alignment_optimizer(base_model, args):
    groups = []
    learning_rates = {
        "graph": args.alignment_graph_lr,
        "smiles": args.alignment_smiles_lr,
        "fp": args.alignment_fp_lr,
    }
    for name in base_model.modality_list:
        groups.append({
            'params': list(base_model.encoders[name].parameters()),
            'lr': float(learning_rates[name]),
            'name': name,
        })
    groups.extend([
        {
            'params': (
                list(base_model.alignment_projections.parameters())
                + list(base_model.shared_private_projections.parameters())
            ),
            'lr': float(args.alignment_projection_lr),
            'name': 'alignment_projection',
        },
    ])
    parallel_fusion = getattr(base_model, "parallel_attention_fusion", None)
    if parallel_fusion is not None:
        groups.append({
            'params': list(parallel_fusion.parameters()),
            'lr': float(args.alignment_fusion_lr),
            'name': 'fusion',
        })
    if hasattr(base_model, 'alignment_mask_head'):
        groups.append({
            'params': list(base_model.alignment_mask_head.parameters()),
            'lr': float(args.alignment_fusion_lr),
            'name': 'alignment_mask_head',
        })
    return optim.AdamW(groups, weight_decay=float(args.weight_decay))


def _module_gradient_norm(module):
    if module is None:
        return 0.0
    squares = []
    for parameter in module.parameters():
        if parameter.grad is not None:
            squares.append(parameter.grad.detach().float().pow(2).sum())
    if not squares:
        return 0.0
    return float(torch.sqrt(torch.stack(squares).sum()).cpu().item())


def _module_gradient_diagnostic(module):
    if module is None:
        return {"finite": None, "nonzero": None, "norm": None}
    if isinstance(module, torch.Tensor):
        parameters = (module,)
    else:
        parameters = module.parameters()
    gradients = [
        parameter.grad.detach().float()
        for parameter in parameters
        if parameter.grad is not None
    ]
    if not gradients:
        return {"finite": False, "nonzero": False, "norm": 0.0}
    finite = all(bool(torch.isfinite(value).all()) for value in gradients)
    norm = float(torch.sqrt(torch.stack([value.pow(2).sum() for value in gradients]).sum()).cpu().item())
    return {"finite": bool(finite), "nonzero": bool(norm > 0.0), "norm": norm}


def _batched_shortest_path_targets(data, max_distance):
    targets = []
    pair_indices = []
    max_distance = int(max_distance)
    for graph_idx in torch.unique(data.batch, sorted=True).tolist():
        node_idx = torch.nonzero(data.batch == graph_idx, as_tuple=False).view(-1)
        n = int(node_idx.numel())
        if n < 2:
            continue
        dist = torch.full((n, n), max_distance, device=data.x.device, dtype=torch.long)
        diag = torch.arange(n, device=data.x.device)
        dist[diag, diag] = 0
        local = torch.full((int(data.batch.numel()),), -1, device=data.x.device, dtype=torch.long)
        local[node_idx] = torch.arange(n, device=data.x.device)
        src, dst = data.edge_index
        keep = (data.batch[src] == graph_idx) & (data.batch[dst] == graph_idx)
        if keep.any():
            ls = local[src[keep]]
            ld = local[dst[keep]]
            valid = (ls >= 0) & (ld >= 0)
            dist[ls[valid], ld[valid]] = 1
        dist_float = dist.float()
        inf = float(max_distance)
        for k in range(n):
            dist_float = torch.minimum(dist_float, dist_float[:, k:k + 1] + dist_float[k:k + 1, :])
        dist = dist_float.clamp(max=inf).long()
        row, col = torch.triu_indices(n, n, offset=1, device=data.x.device)
        if row.numel() == 0:
            continue
        pair_indices.append(torch.stack([node_idx[row], node_idx[col]], dim=1))
        targets.append(dist[row, col])
    if not targets:
        empty_pairs = data.edge_index.new_empty((0, 2))
        empty_targets = data.x.new_empty((0,), dtype=torch.long)
        return empty_pairs, empty_targets
    return torch.cat(pair_indices, dim=0), torch.cat(targets, dim=0)


def _graph_shortest_path_loss(base_model, data, sp_head, max_distance):
    if not _is_scage_graph_encoder(base_model):
        return data.x.new_tensor(0.0)
    _, node_rep = _graph_encode_nodes(base_model, data)
    pairs, targets = _batched_shortest_path_targets(data, max_distance=max_distance)
    if pairs.numel() == 0:
        return data.x.new_tensor(0.0)
    pair_rep = torch.cat([node_rep[pairs[:, 0]], node_rep[pairs[:, 1]], torch.abs(node_rep[pairs[:, 0]] - node_rep[pairs[:, 1]])], dim=-1)
    return F.cross_entropy(sp_head(pair_rep), targets)


def _graph_angle_loss(base_model, data, angle_head, angle_bins):
    if not _is_scage_graph_encoder(base_model) or not hasattr(data, 'pos3d') or not hasattr(data, 'batch3d'):
        return None
    _, node_rep = _graph_encode_nodes(base_model, data)
    reps = []
    targets = []
    angle_bins = int(angle_bins)
    for graph_idx in torch.unique(data.batch, sorted=True).tolist():
        node_idx = torch.nonzero(data.batch == graph_idx, as_tuple=False).view(-1)
        n = int(node_idx.numel())
        if n < 3:
            continue
        coordinate_ok = getattr(data, 'geom_coordinate_ok', getattr(data, 'geom_build_ok', None))
        if coordinate_ok is not None and not bool(coordinate_ok.flatten()[graph_idx].item()):
            continue
        if not hasattr(data, 'graph_to_geom_index'):
            continue
        mapping = data.graph_to_geom_index[node_idx].long()
        if mapping.numel() != n or (mapping < 0).any() or (mapping >= data.pos3d.size(0)).any():
            continue
        pos = data.pos3d[mapping]
        if not torch.isfinite(pos).all():
            continue
        local = torch.full((int(data.batch.numel()),), -1, device=data.x.device, dtype=torch.long)
        local[node_idx] = torch.arange(n, device=data.x.device)
        src, dst = data.edge_index
        keep = (data.batch[src] == graph_idx) & (data.batch[dst] == graph_idx)
        neigh = [[] for _ in range(n)]
        for s_idx, d_idx in zip(src[keep].tolist(), dst[keep].tolist()):
            center = int(local[s_idx].item())
            nb = int(local[d_idx].item())
            if center >= 0 and nb >= 0 and nb not in neigh[center]:
                neigh[center].append(nb)
        for center, ns in enumerate(neigh):
            if len(ns) < 2:
                continue
            for a_i in range(len(ns)):
                for b_i in range(a_i + 1, len(ns)):
                    left, right = ns[a_i], ns[b_i]
                    v1 = pos[left] - pos[center]
                    v2 = pos[right] - pos[center]
                    denom = torch.linalg.vector_norm(v1) * torch.linalg.vector_norm(v2)
                    if float(denom.detach().cpu().item()) <= 1e-8:
                        continue
                    cos = torch.clamp(torch.dot(v1, v2) / denom, -1.0, 1.0)
                    angle = torch.acos(cos)
                    target = torch.clamp((angle / torch.pi * angle_bins).long(), max=angle_bins - 1)
                    global_center = node_idx[center]
                    global_left = node_idx[left]
                    global_right = node_idx[right]
                    reps.append(torch.cat([node_rep[global_left], node_rep[global_center], node_rep[global_right]], dim=-1))
                    targets.append(target)
    if not reps:
        return None
    return F.cross_entropy(angle_head(torch.stack(reps, dim=0)), torch.stack(targets, dim=0).long())

def _graph_pretrain_loss(
    base_model,
    data,
    graph_atom_head,
    mask_ratio,
    mask_weight,
    periodic_aug_weight,
    graph_input,
    repeat_cut_max_mrus,
    repeat_cut_retry,
    repeat_cut_temperature,
):
    mask_loss = _graph_mask_atom_loss(base_model, data, graph_atom_head, mask_ratio)
    periodic_aug_loss = data.x.new_tensor(0.0)
    periodic_aug_stats = _empty_periodic_aug_stats()
    if float(periodic_aug_weight) > 0:
        periodic_aug_loss, periodic_aug_stats = _graph_periodic_aug_loss(
            base_model,
            data,
            graph_input=graph_input,
            temperature=repeat_cut_temperature,
            max_mrus=repeat_cut_max_mrus,
            retry=repeat_cut_retry,
        )
    total = float(mask_weight) * mask_loss + float(periodic_aug_weight) * periodic_aug_loss
    return total, mask_loss, periodic_aug_loss, periodic_aug_stats

def _geom_denoise_loss(
    base_model,
    data,
    geom_noise_head,
    noise_std,
    noise_std_min=None,
    noise_std_max=None,
):
    if 'geom' not in base_model.encoders:
        return data.pos3d.new_tensor(0.0)

    geom_module = base_model.encoders['geom']
    clean_pos = data.pos3d
    target_pos = clean_pos.detach().clone()
    if noise_std_min is not None or noise_std_max is not None:
        if noise_std_min is None or noise_std_max is None:
            raise ValueError("--geom_noise_std_min and --geom_noise_std_max must be set together")
        min_std = float(noise_std_min)
        max_std = float(noise_std_max)
        if min_std < 0.0 or max_std <= 0.0 or min_std > max_std:
            raise ValueError("Require 0 <= --geom_noise_std_min <= --geom_noise_std_max")
        std = torch.empty((), device=target_pos.device, dtype=target_pos.dtype).uniform_(min_std, max_std)
    else:
        std = torch.as_tensor(float(noise_std), device=target_pos.device, dtype=target_pos.dtype)
    noise = torch.randn_like(target_pos) * std
    try:
        data.pos3d = target_pos + noise
        node_rep, _ = geom_module.encoder.encode_nodes(data)
        pred_noise = geom_noise_head(node_rep)
        per_node = F.smooth_l1_loss(pred_noise, noise, reduction='none').mean(dim=-1)
        context = getattr(data, 'geom_context_id', None)
        node_batch = getattr(data, 'batch3d', None)
        if context is None or node_batch is None:
            return per_node.mean()
        sample_weights = torch.where(context.to(per_node.device).long() == 1, 0.3, 1.0)
        weights = sample_weights[node_batch.to(per_node.device)]
        return (per_node * weights).sum() / weights.sum().clamp_min(1e-8)
    finally:
        data.pos3d = clean_pos

def run_pretrain(args=None):
    pretrain_started = time.monotonic()
    if args is None:
        args = parse_arguments()
    if (
        args.graph_encoder_type == MTS_ROUTE_INTERNAL
        and not args.cache_only
        and str(getattr(args, "config_schema", "manual")) != "manual"
    ):
        raise RuntimeError(
            "No active MTS configuration schema; define the next configuration "
            "before launching production MTS."
        )
    if not args.cache_only and not args.benchmark_only and not args.resume_smoke:
        if not args.cache_only and not args.benchmark_only and not args.resume_smoke:
            output_path = Path(args.save_path).resolve()
            historical = "mts_joint_pretraining_pi1m_v2_seed42_canonical_20260808"
            if historical in output_path.name:
                raise RuntimeError(
                    "formal canonical pretraining cannot target the historical "
                    "20260808 checkpoint"
                )
            complete_path = Path(str(output_path) + ".complete.json")
            last_path = Path(str(output_path) + ".last.pt")
            if args.resume_state:
                if output_path.exists() or complete_path.exists() or not last_path.exists():
                    raise RuntimeError(
                        "resume requires an existing matching .last.pt and no final/complete output"
                    )
            elif os.environ.get("MTS_RESTART_SAME_PATH") != "1" and (
                output_path.exists()
                or complete_path.exists()
                or last_path.exists()
            ):
                raise RuntimeError(
                    "fresh canonical pretraining refuses to overwrite an existing output; "
                    "use --resume with its matching .last.pt or choose a new path"
                )
    resume_smoke_stop_steps = int(getattr(args, "resume_smoke_stop_steps", 0) or 0)
    if resume_smoke_stop_steps:
        if not args.resume_smoke:
            raise ValueError("--resume_smoke_stop_steps requires --resume_smoke")
        if resume_smoke_stop_steps <= 0 or resume_smoke_stop_steps > int(args.max_optimizer_steps):
            raise ValueError("--resume_smoke_stop_steps must be between 1 and --max_optimizer_steps")
    loop_max_optimizer_steps = (
        resume_smoke_stop_steps if resume_smoke_stop_steps else int(args.max_optimizer_steps)
    )
    validate_mips_trimer_runtime(args)
    if args.graph_encoder_type == MTS_ROUTE_INTERNAL:
        requested_stage = normalize_stage(args.pretrain_stage)
    else:
        requested_stage = args.pretrain_stage
    args.geometry_adapt = False
    if (
        args.graph_encoder_type == MTS_ROUTE_INTERNAL
        and requested_stage != MTS_STAGE1_ID
    ):
        raise ValueError(
            f"{MTS_ROUTE_NAME} pretraining only supports {MTS_STAGE1_ID}; "
            f"property fine-tuning belongs to train.py as {MTS_STAGE2_ID}"
        )
    if args.graph_encoder_type == MTS_ROUTE_INTERNAL:
        args.pretrain_stage = MTS_STAGE1_ID
    args.checkpoint_stage = requested_stage
    if (
        args.graph_encoder_type == "mips_trimer_scage"
        and args.dataset_name != "PI1M_v2"
    ):
        raise ValueError(
            f"{MTS_ROUTE_NAME} pretraining is fixed to the full PI1M_v2 "
            "cohort; PI1M_50k/PI1M_200k are not active pretraining datasets."
        )
    if (
        args.graph_encoder_type == "mips_trimer_scage"
        and not args.cache_only
        and not args.resume_smoke
        and int(args.max_optimizer_steps) != 20000
    ):
        raise ValueError(
            f"{MTS_STAGE1_ID} is fixed to exactly 20000 optimizer steps"
        )
    stage1_weights = (
        args.scage_mips_mask_weight,
        args.graph_angle_weight,
        args.mips_spd_weight,
        args.mips_path_bond_weight,
        args.mips_repeat_consistency_weight,
        args.mips_distance_weight,
        args.mips_conformer_weight,
        args.scage_screw_geometry_weight,
    )
    if any(weight < 0 for weight in stage1_weights):
        raise ValueError("SCAGE Stage 1 task weights must be non-negative")
    if args.graph_encoder_type == MTS_ROUTE_INTERNAL:
        selected_weights = (
            float(args.scage_mips_mask_weight),
            float(args.graph_angle_weight),
            float(args.mips_spd_weight),
            float(args.mips_path_bond_weight),
        )
        allowed_angle_weights = (
            {0.10, 0.25, 0.50}
            if args.angle_objective == 'cosine' else {0.25}
        )
        if (
            selected_weights[0] != 1.0
            or selected_weights[1] not in allowed_angle_weights
            or selected_weights[2:] != (0.0, 0.0)
        ):
            raise ValueError(
                "MTS joint weights require atom=1.0 and SPD/path=0; "
                "categorical angle uses 0.25 while Angle-v2 calibration "
                "allows 0.10/0.25/0.50"
            )
        retired_weights = (
            args.scage_ecfp_weight,
            args.mips_repeat_consistency_weight,
            args.mips_distance_weight,
            args.mips_conformer_weight,
            args.scage_screw_geometry_weight,
            # The Trimer bond-angle task is the only enabled auxiliary target.
        )
        if any(float(weight) != 0.0 for weight in retired_weights):
            raise ValueError(
                "selected O8 disables ECFP, repeat/cut, distance, conformer, "
                "screw, repeat/cut, distance and other retired objectives"
            )
        if args.dynamic_pretrain_loss:
            raise ValueError("MTS joint pretraining uses fixed loss weights")
    if args.scage_geometry_max_pairs < 1:
        raise ValueError("--scage_geometry_max_pairs must be positive")
    if not 0.0 <= args.scage_shift_balance_power <= 1.0:
        raise ValueError("--scage_shift_balance_power must be in [0, 1]")
    if args.cache_only:
        # Must happen before any CUDA query and before forking RDKit/torch
        # workers. CPU Adam otherwise performs a CUDA graph health check in a
        # bad fork and rejects valid periodic candidates.
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    requested_distributed = int(os.environ.get('WORLD_SIZE', '1')) > 1
    if args.cache_only and requested_distributed:
        raise ValueError("--cache_only must run as one CPU process, not through torchrun")
    distributed = requested_distributed
    if distributed:
        local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend='nccl',
            timeout=timedelta(hours=24),
            device_id=torch.device('cuda', local_rank),
        )
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        local_rank, rank, world_size = 0, 0, 1
    if args.geom_input in {"polygen_periodic", "screw_periodic", "smer_context"} and args.graph_encoder_type != "mips_trimer_scage":
        raise ValueError("polygen_periodic, screw_periodic and smer_context require graph_encoder_type=scage")
    if args.graph_encoder_type == 'mips_trimer_scage':
        if distributed and world_size != 3:
            raise ValueError(
                "The non-PBC MIPS campaign requires exactly three DDP ranks "
                f"(physical GPUs 1,2,3); received world_size={world_size}."
            )
        if args.mips_max_hops is None:
            args.mips_max_hops = (
                2 if args.mips_core == "paper_corrected" else 5
            )
        fixed = {
            'graph_num_layers': (args.graph_num_layers, 6),
            'graph_emb_dim': (args.graph_emb_dim, 512),
            'scage_num_heads': (args.scage_num_heads, 8),
            'scage_ffn_hidden_dim': (args.scage_ffn_hidden_dim, 2048),
            'scage_num_kernels': (args.scage_num_kernels, 128),
        }
        mismatched = [
            f"{name}={actual} (required {expected})"
            for name, (actual, expected) in fixed.items()
            if actual != expected
        ]
        if (
            args.graph_input != 'star_linking'
            or args.geom_input != 'repeat_unit'
            or args.scage_use_pbc_distance
            or mismatched
        ):
            raise ValueError(
                "The second route is fixed to non-PBC sparse MIPS with "
                "star_linking and geom_input=repeat_unit; "
                + ", ".join(mismatched)
            )
    from src.dataset import UniDataset
    if (
        args.graph_encoder_type == MTS_ROUTE_INTERNAL
        and args.angle_objective == 'cosine'
        and not args.angle_cache_root_override
    ):
        from scripts.audit_mips_trimer_cache import _specs as _cache_specs
        from src.dataset.lmdb_cache import build_or_load_cohort
        from src.dataset.trimer_angle_continuous_cache import continuous_angle_root
        cache_specs = _cache_specs(Path(PROJECT_ROOT))
        cohort = build_or_load_cohort(
            Path(args.root) / 'processed' / 'mips_trimer_scage',
            'PI1M_v2', Path(args.root) / 'raw' / 'PI1M_v2.csv',
            load_text=False, verify_integrity=False,
        )
        args.angle_cache_root_override = str(continuous_angle_root(
            cache_specs['trimer']['root'],
            cohort['manifest']['cohort_hash'],
        ))
    dataset_kwargs = dataset_kwargs_from_args(args)
    # Cache readers perform direct schema/shape/index checks.  The historical
    # frozen-bundle SHA audit remains an explicit offline tool, not a startup
    # dependency for Pretrain.
    if args.cache_only:
        warnings.filterwarnings("ignore")
        dataset = UniDataset(**dataset_kwargs)
        print(
            f"[feature_cache] cache-only complete: path={dataset.feature_cache_path}, "
            f"usable_rows={len(dataset)}"
        )
        return

    from src.modules import UniEncoderAttention
    from src.utils import compute_contrastive_loss, get_data_loader, set_global_seed
    # All ranks must construct exactly the same parameters.  Rank-specific
    # randomness is enabled only after DDP has broadcast the model state.
    set_global_seed(args.seed)

    # Get all available GPUs
    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
        if rank == 0:
            print(f"Found {n_gpus} GPUs available; distributed world_size={world_size}")
        device = torch.device("cuda", local_rank)
    else:
        if args.graph_encoder_type == "mips_trimer_scage":
            raise RuntimeError(f"{MTS_ROUTE_NAME} pretraining requires CUDA")
        print("No GPU available, using CPU")
        device = torch.device("cpu")

    if device.type == "cuda":
        # These affect only float32 matmuls outside the BF16 autocast region
        # and are deterministic on the fixed CUDA hardware used by the route.
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Ignore warnings
    warnings.filterwarnings("ignore")

    # Build dataset and DataLoader (using the same dataset for unsupervised training, only using input features)
    if distributed:
        # Cache construction is a single-writer CPU operation.  Rank zero may
        # build a missing/rebuilt cache; all other ranks open it only after the
        # atomic final save has completed.
        if rank == 0:
            dataset = UniDataset(**dataset_kwargs)
        dist.barrier()
        if rank != 0:
            dataset_kwargs['rebuild_feature_cache'] = False
            dataset = UniDataset(**dataset_kwargs)
    else:
        dataset = UniDataset(**dataset_kwargs)
    if args.graph_encoder_type == "mips_trimer_scage":
        cohort = getattr(dataset, "_cohort", None)
        if (
            cohort is None
            or int(cohort["manifest"].get("unique_count", -1)) != 995799
        ):
            raise RuntimeError(
                f"{MTS_ROUTE_NAME} topology/geometry stages require the complete "
                "PI1M_v2 cohort (995799 records)."
            )
        # Resolve all production layers for the lifecycle gate, even though
        # Stage 1 itself only requests ru_base/topology and Stage 2 adds
        # Trimer.  The finalized downstream union is frozen before training.
        original_layers = dataset.cache_layers
        dataset.cache_layers = ("ru_base", "topology", "trimer", "md200")
        specs = dataset._lmdb_cache_specs({})
        dataset.cache_layers = original_layers
        unfrozen = [
            name for name, spec in specs.items()
            if not os.path.isfile(os.path.join(spec["root"], ".frozen"))
        ]
        if unfrozen:
            raise RuntimeError(
                f"{MTS_ROUTE_NAME} cache artifacts are not frozen: "
                + ", ".join(unfrozen)
            )
        if getattr(getattr(dataset, "_lazy_feature_store", None), "angle_offsets", None) is None:
            raise RuntimeError(
                "MTS joint pretraining requires the frozen Trimer bond-angle "
                "cache; run scripts/prepare_mts_angle_cache.py first"
            )
        if args.angle_objective == 'categorical':
            angle_counts = np.asarray(
                getattr(dataset, "angle_class_counts", np.zeros(20, dtype=np.int64)),
                dtype=np.int64,
            ).reshape(-1)
            if angle_counts.shape != (20,) or int(angle_counts.sum()) <= 0:
                raise RuntimeError("MTS angle cache has no valid class histogram")
            args.angle_majority_class = int(angle_counts.argmax())
            args.angle_majority_baseline = float(
                angle_counts.max() / angle_counts.sum()
            )
        else:
            args.angle_majority_class = -1
            args.angle_majority_baseline = float('nan')
    indices = np.arange(len(dataset))
    if args.pretrain_unique_smiles and args.pretrain_stage in {'mts_joint_pretraining', 'scage_m4p', 'alignment'}:
        # PI1M_v2 is a content-addressed, already unique cohort.  Do not
        # rebuild a million-entry Python ``set`` (or deserialize graph rows)
        # just to rediscover that invariant on every DDP rank.  Other
        # datasets retain the historical canonical-SMILES deduplication.
        manifest = getattr(dataset, "_cohort", {}).get("manifest", {})
        cohort_is_unique = (
            args.graph_encoder_type == "mips_trimer_scage"
            and int(manifest.get("unique_count", -1)) == len(dataset)
            and int(manifest.get("record_count", len(dataset))) == len(dataset)
        )
        if cohort_is_unique:
            indices = np.arange(len(dataset), dtype=np.int64)
            if rank == 0:
                print(
                    "Pretraining identity deduplication skipped: "
                    "PI1M_v2 manifest guarantees unique sample keys"
                )
        else:
            indices = _unique_smiles_indices(dataset)
            print(
                f"Pretraining identity deduplication: {len(dataset)} rows -> "
                f"{len(indices)} unique SMILES"
            )
    angle_validation_indices = np.empty((0,), dtype=np.int64)
    if args.angle_objective == 'cosine':
        validation_mask = getattr(dataset, 'angle_validation_mask', None)
        if validation_mask is None:
            raise RuntimeError(
                'Angle-v2 requires its deterministic 1% validation mask'
            )
        validation_mask = np.asarray(validation_mask, dtype=np.bool_)
        if validation_mask.shape != (len(dataset),):
            raise RuntimeError('Angle-v2 validation mask shape mismatch')
        angle_validation_indices = indices[validation_mask[indices]]
        indices = indices[~validation_mask[indices]]
        if not len(angle_validation_indices) or not len(indices):
            raise RuntimeError('Angle-v2 train/validation split is empty')
        if rank == 0:
            print(
                'Angle-v2 deterministic split: '
                f'train={len(indices)}, validation={len(angle_validation_indices)}'
            )
    sampler = None
    loader_generator = torch.Generator()
    loader_generator.manual_seed(int(args.seed) + 9176 * int(rank))
    if distributed:
        from torch.utils.data import Subset
        subset = Subset(dataset, [int(index) for index in indices])
        if args.graph_encoder_type == "mips_trimer_scage":
            cost_path = None
            if getattr(cohort, "get", None) is not None:
                cost_path = Path(cohort["root"]) / "topology_cost.npy"
            if args.batch_balance == "cost":
                if cost_path is None or not cost_path.is_file():
                    raise RuntimeError(
                        "cost-balanced MIPS sampling requires the frozen "
                        "topology_cost.npy derived artifact"
                    )
                costs = np.load(cost_path, mmap_mode="r")
                if (
                    costs.ndim != 2
                    or tuple(costs.shape) != (len(dataset), 2)
                    or costs.dtype != np.uint32
                ):
                    raise RuntimeError(
                        "topology_cost.npy is incomplete or has an invalid shape/dtype"
                    )
                sampler = _CostBalancedDistributedSampler(
                    costs[np.asarray(indices, dtype=np.int64)],
                    batch_size=args.batch_size,
                    num_replicas=world_size,
                    rank=rank,
                    seed=args.seed,
                    drop_last=False,
                )
            else:
                # LMDB provides single-record random access. DistributedSampler
                # pads deterministically and covers every source row.
                sampler = torch.utils.data.DistributedSampler(
                    subset,
                    num_replicas=world_size,
                    rank=rank,
                    shuffle=True,
                    seed=args.seed,
                    drop_last=False,
                )
        else:
            sampler = _ShardAwareDistributedSampler(
                len(subset),
                num_replicas=world_size,
                rank=rank,
                seed=args.seed,
            )
    dataloader = get_data_loader(
        dataset, indices=indices, batch_size=args.batch_size, shuffle=True,
        drop_last=(
            distributed and args.graph_encoder_type != "mips_trimer_scage"
        ),
        random_conformer=True,
        num_workers=args.loader_workers, pin_memory=True,
        prefetch_factor=args.loader_prefetch_factor,
        persistent_workers=args.loader_workers > 0, sampler=sampler,
        generator=loader_generator,
    )
    angle_validation_loader = None
    if args.angle_objective == 'cosine':
        validation_sampler = None
        if distributed:
            from torch.utils.data import Subset
            validation_subset = Subset(
                dataset, [int(index) for index in angle_validation_indices]
            )
            validation_sampler = _RankSliceSampler(
                len(validation_subset), rank, world_size
            )
        angle_validation_loader = get_data_loader(
            dataset, indices=angle_validation_indices,
            batch_size=args.batch_size, shuffle=False, drop_last=False,
            random_conformer=False, num_workers=args.loader_workers,
            pin_memory=True, prefetch_factor=args.loader_prefetch_factor,
            persistent_workers=args.loader_workers > 0,
            sampler=validation_sampler,
        )

    # Initialize model
    model = UniEncoderAttention(
        joint_embedding_dim=args.joint_embedding_dim,
        smiles_model_name=args.smiles_model_name,
        gnn_model_name=args.gnn_model_name,
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
        mcl_distance_percentiles=args.mcl_distance_percentiles,
        trimer_num_candidates=args.trimer_num_candidates,
        trimer_max_heavy_atoms=args.trimer_max_heavy_atoms,
        mips_variant=args.mips_variant,
        mips_fusion_mode=args.mips_fusion_mode,
        projection_mode=args.projection_mode,
        modality_control=args.modality_control,
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
        msta_layer_indices=args.msta_layer_indices,
        msta_local_spd=args.msta_local_spd,
        msta_context_spd=args.msta_context_spd,
        msta_share_relation_dropout=args.msta_share_relation_dropout,
        msta_local_output_bias=args.msta_local_output_bias,
        msta_local_output_init=args.msta_local_output_init,
        star_rbf_definition=args.star_rbf_definition,
        star_rbf_upper=args.star_rbf_upper,
        fusion_type=args.fusion_type,
        fp_mode=args.fp_mode,
        fusion_dropout=args.fusion_dropout,
        alignment_projection_dim=args.alignment_projection_dim,
    )

    if args.pretrained_model_path:
        checkpoint = torch.load(args.pretrained_model_path, map_location='cpu')
        if not isinstance(checkpoint, dict):
            raise RuntimeError("pretraining checkpoint must be a mapping")
        checkpoint_state = checkpoint.get("state_dict", checkpoint)
        if not isinstance(checkpoint_state, dict):
            raise RuntimeError("pretraining checkpoint does not contain a state_dict mapping")
        checkpoint_meta = (
            checkpoint.get("meta", {})
            if isinstance(checkpoint.get("meta"), dict) else {}
        )
        model_state = model.state_dict()
        missing = sorted(set(model_state) - set(checkpoint_state))
        unexpected = sorted(set(checkpoint_state) - set(model_state))
        if args.graph_encoder_type == 'mips_trimer_scage':
            graph_mismatch = [
                key for key in list(missing) + list(unexpected)
                if 'encoders.graph.encoder' in key
                and ".trimer_mcl." not in key
            ]
            if graph_mismatch:
                raise RuntimeError(
                    "SCAGE checkpoint architecture mismatch; incompatible "
                    "graph tensor keys: " + ", ".join(graph_mismatch[:10])
                )
        for key, value in checkpoint_state.items():
            if key in model_state and tuple(value.shape) != tuple(model_state[key].shape):
                raise RuntimeError(f"pretraining checkpoint tensor shape mismatch: {key}")
            if torch.is_tensor(value) and value.is_floating_point() and not torch.isfinite(value).all():
                raise RuntimeError(f"pretraining checkpoint tensor is non-finite: {key}")
        model_state.update({key: value for key, value in checkpoint_state.items() if key in model_state})
        model.load_state_dict(model_state, strict=True)
        print(f"Loaded pretraining checkpoint from {args.pretrained_model_path}")
        print(f"Checkpoint load: {len(missing)} missing keys, {len(unexpected)} unexpected keys")

    model = model.to(device)
    base_model = _base_model(model)
    if args.geometry_adapt:
        raise RuntimeError("MTS Geometry Adaptation is retired; use joint pretraining")
    loss_weights = _stage_loss_weights(args)
    dynamic_loss_weighter = None
    display_stage = (
        stage_display_name(args.checkpoint_stage)
        if args.graph_encoder_type == MTS_ROUTE_INTERNAL
        else args.pretrain_stage
    )
    print(
        "Pretraining stage: "
        f"{display_stage} "
        f"(contrastive={loss_weights['contrastive']}, graph={loss_weights['graph']}, geom={loss_weights['geom']}, "
        f"dynamic_loss={args.dynamic_pretrain_loss})"
    )

    aux_modules = nn.ModuleList()
    graph_atom_head = None
    geom_noise_head = None
    graph_sp_head = None
    graph_angle_head = None
    m4p_ecfp_head = None
    m4p_torsion_head = None
    m4p_distance_head = None
    m4p_shift_head = None
    m4p_screw_head = None
    m4p_projection_head = None
    mips_atom_head = None
    mips_path_bond_head = None
    m4p_ecfp_pos_weight = None
    m4p_ecfp_valid_count = 0
    if args.pretrain_stage in {'scage_m4p', MTS_STAGE1_ID}:
        graph_encoder = base_model.encoders['graph'].encoder
        graph_dim = graph_encoder.emb_dim
        atom_classes = int(
            getattr(graph_encoder, "masked_atom_classes", 119)
        )
        if getattr(graph_encoder, "semantics", "") == "public_code_diagnostic":
            mips_atom_head = nn.Sequential(
                nn.Linear(graph_dim, graph_dim),
                nn.GELU(),
                nn.Dropout(args.graph_dropout),
                nn.Linear(graph_dim, atom_classes),
            ).to(device)
        else:
            mips_atom_head = nn.Linear(graph_dim, atom_classes).to(device)
        if args.pretrain_stage == MTS_STAGE1_ID:
            graph_angle_head = TrimerAngleHead(
                dim=graph_dim,
                hidden=256,
                bins=int(args.scage_angle_bins),
                alpha=(
                    _angle_alpha_from_dataset(dataset, args.scage_angle_bins)
                    if args.angle_objective == 'categorical' else None
                ),
                dropout=0.10,
                objective=args.angle_objective,
            ).to(device)
        else:
            # Retained only for non-production historical diagnostics.
            graph_sp_head = nn.Linear(graph_dim * 3, 3).to(device)
            mips_path_bond_head = nn.Linear(graph_dim * 3, 6).to(device)
            aux_modules.extend([
                mips_atom_head, graph_sp_head, mips_path_bond_head,
            ])
    elif not (args.pretrain_stage == 'alignment' and args.graph_encoder_type == 'mips_trimer_scage') \
            and 'graph' in args.modalities and loss_weights['graph'] > 0:
        graph_dim = base_model.encoders['graph'].encoder.emb_dim
        from src.dataset.graph_data import allowable_features
        graph_atom_head = nn.Linear(graph_dim, len(allowable_features['possible_atom_symbols'])).to(device)
        aux_modules.append(graph_atom_head)
        if args.graph_encoder_type == 'mips_trimer_scage':
            graph_sp_head = nn.Linear(graph_dim * 3, int(args.scage_sp_max_distance) + 1).to(device)
            graph_angle_head = nn.Linear(graph_dim * 3, int(args.scage_angle_bins)).to(device)
            aux_modules.extend([graph_sp_head, graph_angle_head])
    if args.pretrain_stage == 'alignment' and args.graph_encoder_type == 'mips_trimer_scage':
        graph_dim = base_model.encoders['graph'].encoder.emb_dim
        atom_classes = int(
            base_model.encoders['graph'].encoder.masked_atom_classes
        )
        base_model.alignment_mask_head = nn.Linear(
            graph_dim + base_model.joint_embedding_dim,
            atom_classes,
        ).to(device)
    if args.pretrain_stage != 'scage_m4p' and 'geom' in args.modalities and loss_weights['geom'] > 0:
        geom_dim = base_model.encoders['geom'].encoder.hidden_channels
        geom_noise_head = nn.Linear(geom_dim, 3).to(device)
        aux_modules.append(geom_noise_head)

    if args.amp_dtype == 'bf16':
        if device.type != 'cuda' or not torch.cuda.is_bf16_supported():
            args.amp_dtype = 'fp32'
            if rank == 0:
                print("BF16 disabled: CUDA BF16 support is unavailable")
        elif args.pretrain_stage == MTS_STAGE1_ID:
            parity_batch = None
            for candidate in dataloader:
                candidate = candidate.to(device)
                if not args.geometry_adapt or bool(
                    _mcl_valid_graph_mask(candidate).any()
                ):
                    parity_batch = candidate
                    break
            if parity_batch is None:
                raise RuntimeError(
                    "BF16 parity gate could not find a valid geometry batch"
                )
            passed, parity = _bf16_joint_parity_gate(
                base_model, parity_batch, mips_atom_head, graph_angle_head, args
            )
            if distributed:
                passed_tensor = torch.tensor(int(passed), device=device)
                dist.all_reduce(passed_tensor, op=dist.ReduceOp.MIN)
                passed = bool(passed_tensor.item())
            if rank == 0:
                print(f"BF16 parity gate: {parity}, pass={passed}")
            if not passed:
                args.amp_dtype = 'fp32'
        elif args.pretrain_stage == 'alignment' and args.graph_encoder_type == 'mips_trimer_scage':
            parity_batch = next(iter(dataloader)).to(device)
            passed, parity = _bf16_alignment_parity_gate(base_model, parity_batch, args)
            if distributed:
                passed_tensor = torch.tensor(int(passed), device=device)
                dist.all_reduce(passed_tensor, op=dist.ReduceOp.MIN)
                passed = bool(passed_tensor.item())
            if rank == 0:
                print(f"BF16 alignment parity gate: {parity}, pass={passed}")
            if not passed:
                args.amp_dtype = 'fp32'

    mips_ddp_route = (
        args.graph_encoder_type == "mips_trimer_scage"
        and args.pretrain_stage == MTS_STAGE1_ID
    )
    if mips_ddp_route:
        # Restrict the trainable/DDP parameter set to the active stage.  The
        # model object still owns the complete production encoder for strict
        # checkpoint compatibility, but frozen branches must not receive
        # zero-gradient anchors or optimizer state during pretraining.
        for parameter in model.parameters():
            parameter.requires_grad = False
        graph_encoder = base_model.encoders["graph"].encoder
        active_modules = (
            graph_encoder.atom_embedding,
            graph_encoder.spd_embedding,
            graph_encoder.path_bias,
            graph_encoder.layers,
        )
        active_modules = active_modules + (
            graph_encoder.star_distance_bias,
            graph_encoder.trimer_mcl,
        )
        for module in active_modules:
            for parameter in module.parameters():
                parameter.requires_grad = True
    mips_heads = {}
    if args.pretrain_stage == MTS_STAGE1_ID:
        mips_heads = {
            "mips_atom": mips_atom_head,
            "angle": graph_angle_head,
        }
    train_container = (
        MIPSPretrainContainer(model, mips_heads).to(device)
        if mips_ddp_route else None
    )
    train_module = train_container if train_container is not None else model
    if distributed and mips_ddp_route:
        from torch.nn.parallel import DistributedDataParallel
        train_module = DistributedDataParallel(
            train_container,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=True,
            find_unused_parameters=False,
        )
        _assert_ddp_parameters_synced(train_module, step=0)
        # Parameter initialization is synchronized by DDP.  Only stochastic
        # data transforms and dropout now receive rank-specific streams.
        set_global_seed(args.seed + 100003 * rank)

    if args.pretrain_stage == 'alignment' and args.graph_encoder_type == 'mips_trimer_scage':
        optimizer = _scage_alignment_optimizer(base_model, args)
    elif args.pretrain_stage == MTS_STAGE1_ID and args.graph_encoder_type == 'mips_trimer_scage':
        trainable_parameters = [
            parameter for parameter in train_container.parameters()
            if parameter.requires_grad
        ]
        adam_kwargs = dict(
            lr=args.lr, betas=(0.9, 0.98), eps=1e-8, weight_decay=0.0,
        )
        # Match the public MIPS implementation: ordinary torch.optim.Adam,
        # without forcing a backend solely for bitwise interruption parity.
        optimizer = optim.Adam(trainable_parameters, **adam_kwargs)
    else:
        optimizer = optim.AdamW(
            list(model.parameters()) + list(aux_modules.parameters()),
            lr=args.lr,
            weight_decay=float(args.weight_decay),
        )

    accumulation_steps = max(1, int(args.gradient_accumulation_steps))
    available_batches = int(args.epochs) * len(dataloader)
    scheduled_batches = (
        min(available_batches, int(args.max_steps))
        if int(args.max_steps) > 0 else available_batches
    )
    total_optimizer_steps = max(1, math.ceil(scheduled_batches / accumulation_steps))
    if int(args.max_optimizer_steps) > 0:
        total_optimizer_steps = min(
            total_optimizer_steps, int(args.max_optimizer_steps)
        )
    if args.pretrain_stage == MTS_STAGE1_ID:
        warmup_steps = min(
            max(0, total_optimizer_steps - 1),
            max(0, int(args.warmup_steps)),
        )
    else:
        warmup_steps = min(
            total_optimizer_steps - 1,
            max(0, int(round(total_optimizer_steps * float(args.warmup_ratio)))),
        )

    def lr_scale(update_step):
        if warmup_steps > 0 and update_step < warmup_steps:
            return float(update_step + 1) / float(warmup_steps)
        decay_steps = max(1, total_optimizer_steps - warmup_steps)
        progress = min(1.0, max(0.0, (update_step - warmup_steps) / decay_steps))
        if args.pretrain_stage == MTS_STAGE1_ID or args.mips_scheduler == 'polynomial':
            base_lr = max(float(args.lr), 1e-12)
            floor = min(1.0, max(0.0, float(args.end_lr) / base_lr))
            return floor + (1.0 - floor) * (1.0 - progress) ** float(args.scheduler_power)
        if args.mips_scheduler == 'linear':
            return 1.0 - progress
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_scale)
    if rank == 0:
        print(
            f"Optimizer schedule: updates={total_optimizer_steps}, "
            f"warmup={warmup_steps}, {args.mips_scheduler} decay"
        )

    # Runtime-only resume settings.  These fields describe the numerical and
    # sampler state needed to continue a trajectory; no config, source,
    # artifact, checkpoint or code digest is persisted in .last.pt.
    runtime_config = PretrainRuntimeConfig.from_args(args, world_size=world_size)
    resume_contract = {
        "layout_version": 1,
        "batch_size": runtime_config.batch_size,
        "gradient_accumulation_steps": runtime_config.gradient_accumulation_steps,
        "lr": runtime_config.learning_rate,
        "weight_decay": (
            0.0 if args.graph_encoder_type == MTS_ROUTE_INTERNAL
            else float(args.weight_decay)
        ),
        "adam_betas": [0.9, 0.98],
        "adam_eps": 1e-8,
        "warmup_steps": runtime_config.warmup_steps,
        "scheduler": runtime_config.scheduler,
        "scheduler_power": float(args.scheduler_power),
        "end_lr": float(args.end_lr),
        "max_grad_norm": float(args.max_grad_norm),
        "amp_dtype": runtime_config.amp_dtype,
        "target_optimizer_steps": runtime_config.max_optimizer_steps,
        "epoch_cap": int(args.epochs),
        "loader_workers": int(args.loader_workers),
        "loader_prefetch_factor": int(args.loader_prefetch_factor),
        "sampler_type": type(sampler).__name__ if sampler is not None else "random",
        "batch_balance": str(args.batch_balance),
        "optimizer_impl": "adam",
        "matmul_precision": "high_tf32" if device.type == "cuda" else "default",
        "seed": runtime_config.seed,
        "world_size": runtime_config.world_size,
    }
    resume_epoch = 0
    resume_step_idx = 0
    global_step = 0
    optimizer_steps_completed = 0
    # Iterating over already-consumed batches to reconstruct the sampler
    # position can itself consume Python/NumPy/Torch/CUDA RNG state (for
    # example through dataset-side stochastic feature handling).  Keep the
    # checkpointed per-rank state and restore it again immediately before the
    # first batch that is actually optimized after a resume.
    resume_rng_state = None
    resume_loader_state = None
    if args.resume_state:
        resume_path = os.path.abspath(args.resume_state)
        if not os.path.isfile(resume_path):
            raise RuntimeError(f"resume state does not exist: {resume_path}")
        # This is an internally generated, identity-bound train-state file.
        # PyTorch 2.6+ defaults to ``weights_only=True``; that mode cannot
        # deserialize the NumPy/RNG state captured for exact trajectory
        # resumption.  The resume contract is validated immediately below,
        # and the file is only accepted from the explicit user-provided path.
        resume_payload = torch.load(
            resume_path, map_location="cpu", weights_only=False
        )
        if resume_payload.get("schema") != PRETRAIN_TRAIN_STATE_SCHEMA:
            raise RuntimeError(
                "resume state uses an obsolete checkpoint schema; rerun from "
                "the latest stage checkpoint"
            )
        resume_meta = resume_payload.get("meta", {})
        runtime_fields = {
            "layout_version": 1,
            "batch_size": int(args.batch_size),
            "gradient_accumulation_steps": int(args.gradient_accumulation_steps),
            "lr": float(args.lr),
            "weight_decay": (
                0.0 if args.graph_encoder_type == MTS_ROUTE_INTERNAL
                else float(args.weight_decay)
            ),
            "adam_betas": [0.9, 0.98],
            "adam_eps": 1e-8,
            "warmup_steps": int(args.warmup_steps),
            "scheduler": str(args.mips_scheduler),
            "scheduler_power": float(args.scheduler_power),
            "end_lr": float(args.end_lr),
            "max_grad_norm": float(args.max_grad_norm),
            "amp_dtype": str(args.amp_dtype),
            "target_optimizer_steps": int(args.max_optimizer_steps),
            "epoch_cap": int(args.epochs),
            "loader_workers": int(args.loader_workers),
            "loader_prefetch_factor": int(args.loader_prefetch_factor),
            "sampler_type": type(sampler).__name__ if sampler is not None else "random",
            "batch_balance": str(args.batch_balance),
            "optimizer_impl": "adam",
            "matmul_precision": "high_tf32" if device.type == "cuda" else "default",
            "seed": int(args.seed),
            "world_size": int(world_size),
        }
        runtime_mismatches = {
            key: (resume_meta.get(key), value)
            for key, value in runtime_fields.items()
            if resume_meta.get(key) != value
        }
        # The maintenance resume gate intentionally exercises a short
        # 3-step checkpoint continuing to an 8-step target.  Scientific
        # Formal runs keep an exact fixed budget; only ``--resume_smoke`` may
        # extend its test target without changing optimizer/sampler identity.
        if args.resume_smoke and "target_optimizer_steps" in runtime_mismatches:
            previous_target = int(resume_meta.get("target_optimizer_steps", -1))
            current_target = int(runtime_fields["target_optimizer_steps"])
            if 0 < previous_target <= current_target:
                runtime_mismatches.pop("target_optimizer_steps")
        if runtime_mismatches:
            raise RuntimeError(f"resume state does not match runtime settings: {runtime_mismatches}")
        train_module.load_state_dict(resume_payload["train_module"], strict=True)
        if "aux_modules" in resume_payload:
            aux_modules.load_state_dict(resume_payload["aux_modules"], strict=True)
        optimizer.load_state_dict(resume_payload["optimizer"])
        scheduler.load_state_dict(resume_payload["scheduler"])
        resume_epoch = int(resume_payload.get("epoch", 0))
        resume_step_idx = int(resume_payload.get("next_step_idx", 0))
        global_step = int(resume_payload.get("global_step", 0))
        optimizer_steps_completed = int(
            resume_payload.get("optimizer_steps_completed", 0)
        )
        rng_by_rank = resume_payload.get("rng_state_by_rank")
        if rng_by_rank is None:
            # v1 checkpoints had only rank zero state.  They remain readable
            # for single-process diagnostics, but distributed resume is not
            # silently treated as exact.
            if distributed:
                raise RuntimeError(
                    "distributed resume requires rng_state_by_rank; "
                    "the checkpoint was created by the old single-RNG format"
                )
            _restore_rng_state(resume_payload.get("rng_state"))
        else:
            if int(rank) >= len(rng_by_rank):
                raise RuntimeError("checkpoint has no RNG state for this rank")
            resume_rng_state = rng_by_rank[int(rank)]
            _restore_rng_state(resume_rng_state)
        loader_states = resume_payload.get("loader_generator_state_by_rank")
        if loader_states is None:
            # A rank-zero loader state is not sufficient for a distributed
            # exact resume once workers are enabled: each rank has its own
            # worker-seeding stream.  Refuse the old format instead of
            # silently changing the masking/augmentation sequence.
            if distributed:
                raise RuntimeError(
                    "distributed resume requires loader_generator_state_by_rank"
                )
            loader_state = resume_payload.get("loader_generator_state")
        else:
            if int(rank) >= len(loader_states):
                raise RuntimeError(
                    "checkpoint has no DataLoader generator state for this rank"
                )
            loader_state = loader_states[int(rank)]
        if loader_state is not None:
            loader_generator.set_state(loader_state)
            resume_loader_state = loader_state
        sampler_states = resume_payload.get("sampler_state_by_rank")
        if distributed:
            if sampler_states is None or int(rank) >= len(sampler_states):
                raise RuntimeError(
                    "distributed resume requires sampler_state_by_rank"
                )
            saved_sampler = sampler_states[int(rank)] or {}
            if int(saved_sampler.get("world_size", -1)) != int(world_size):
                raise RuntimeError("resume sampler world_size mismatch")
            if int(saved_sampler.get("next_batch_index", -1)) != int(resume_step_idx):
                raise RuntimeError("resume sampler/batch position mismatch")
        if rank == 0:
            print(
                f"Resumed pretraining at epoch={resume_epoch}, "
                f"step={optimizer_steps_completed}"
            )

    train_state_path = (
        os.path.abspath(args.resume_state)
        if args.resume_state else os.path.abspath(args.save_path + ".last.pt")
    )

    if args.benchmark_only:
        if args.pretrain_stage != MTS_STAGE1_ID or not mips_ddp_route:
            raise ValueError("benchmark_only is currently supported for MIPS pretraining")
        benchmark_batches = int(args.benchmark_batches)
        if benchmark_batches <= 0:
            raise ValueError("--benchmark_batches must be positive")
        train_module.train()
        iterator = iter(dataloader)
        warmup_batches = 50
        total_batches = warmup_batches + benchmark_batches
        timings = []
        data_timings = []
        h2d_timings = []
        forward_timings = []
        backward_timings = []
        optimizer_timings = []
        sample_count = 0
        optimizer_steps = 0
        accumulation = max(1, int(args.gradient_accumulation_steps))
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        optimizer.zero_grad(set_to_none=True)
        finite_loss = True
        finite_gradient = True
        finite_parameters = True
        for batch_index in range(total_batches):
            data_started = time.monotonic()
            try:
                data = next(iterator)
            except StopIteration:
                if sampler is not None:
                    sampler.set_epoch(getattr(sampler, "epoch", 0) + 1)
                iterator = iter(dataloader)
                data = next(iterator)
            data_elapsed = time.monotonic() - data_started
            transfer_started = time.monotonic()
            data = data.to(device, non_blocking=True)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            transfer_elapsed = time.monotonic() - transfer_started
            started = time.monotonic()
            should_step = (batch_index + 1) % accumulation == 0
            sync_context = (
                train_module.no_sync()
                if distributed and not should_step else nullcontext()
            )
            with sync_context:
                forward_started = time.monotonic()
                with torch.autocast(
                    device_type="cuda", dtype=torch.bfloat16,
                    enabled=args.amp_dtype == "bf16" and device.type == "cuda",
                ):
                    payload = train_module(MTS_STAGE1_ID, data, args, batch_index)
                    priors = {
                        "masked_atom": args.scage_mips_mask_weight,
                        "angle": args.graph_angle_weight,
                    }
                    loss = (
                        priors["masked_atom"]
                        * payload["loss_terms"]["masked_atom_sum"]
                        / max(1, payload["counts"]["masked_atoms"])
                        + priors["angle"]
                        * payload["loss_terms"]["angle_sum"]
                        / max(1, payload["counts"]["angle_graphs"])
                    )
                    loss = loss + payload["zero_reference"]
                finite_loss = finite_loss and bool(torch.isfinite(loss.detach()))
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                forward_elapsed = time.monotonic() - forward_started
                backward_started = time.monotonic()
                (loss / accumulation).backward()
                if batch_index + 1 == total_batches:
                    finite_gradient = all(
                        parameter.grad is None
                        or bool(torch.isfinite(parameter.grad).all())
                        for parameter in train_module.parameters()
                    )
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                backward_elapsed = time.monotonic() - backward_started
            if should_step:
                optimizer_started = time.monotonic()
                if float(args.max_grad_norm) > 0:
                    torch.nn.utils.clip_grad_norm_(train_module.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                optimizer_elapsed = time.monotonic() - optimizer_started
                if batch_index + 1 == total_batches:
                    finite_parameters = all(
                        bool(torch.isfinite(parameter).all())
                        for parameter in train_module.parameters()
                    )
                optimizer_steps += int(batch_index >= warmup_batches)
            else:
                optimizer_elapsed = 0.0
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed = time.monotonic() - started
            if batch_index >= warmup_batches:
                timings.append(elapsed + data_elapsed + transfer_elapsed)
                data_timings.append(data_elapsed)
                h2d_timings.append(transfer_elapsed)
                forward_timings.append(forward_elapsed)
                backward_timings.append(backward_elapsed)
                optimizer_timings.append(optimizer_elapsed)
                sample_count += int(data.graph_available.numel())
        local_elapsed = float(sum(timings))
        finite_tensor = torch.tensor(
            [finite_loss, finite_gradient, finite_parameters],
            device=device, dtype=torch.int32,
        )
        if distributed:
            dist.all_reduce(finite_tensor, op=dist.ReduceOp.MIN)
        local_compute = torch.tensor(
            [local_elapsed, local_elapsed], device=device, dtype=torch.float64
        )
        if distributed:
            maximum = local_compute[:1].clone()
            minimum = local_compute[1:].clone()
            dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
            dist.all_reduce(minimum, op=dist.ReduceOp.MIN)
            elapsed = float(maximum.item())
            rank_wait_fraction = max(0.0, (elapsed - float(minimum.item())) / max(elapsed, 1e-9))
        else:
            elapsed = local_elapsed
            rank_wait_fraction = 0.0
        if rank == 0:
            effective_samples = sample_count * max(1, int(world_size))
            peak_allocated = (
                int(torch.cuda.max_memory_allocated(device))
                if device.type == "cuda" else 0
            )
            peak_reserved = (
                int(torch.cuda.max_memory_reserved(device))
                if device.type == "cuda" else 0
            )
            print(json.dumps({
                "benchmark_batches": benchmark_batches,
                "warmup_batches": warmup_batches,
                "batch_size_per_rank": int(args.batch_size),
                "world_size": int(world_size),
                "elapsed_seconds": elapsed,
                "samples_per_second": effective_samples / max(elapsed, 1e-9),
                "optimizer_steps_per_second": optimizer_steps / max(elapsed, 1e-9),
                "mean_data_seconds": float(np.mean(data_timings)),
                "mean_h2d_seconds": float(np.mean(h2d_timings)),
                "mean_forward_seconds": float(np.mean(forward_timings)),
                "mean_backward_ddp_seconds": float(np.mean(backward_timings)),
                "mean_optimizer_seconds": float(np.mean(optimizer_timings)),
                "rank_wait_fraction": rank_wait_fraction,
                "loss_finite": bool(finite_tensor[0].item()),
                "gradient_finite": bool(finite_tensor[1].item()),
                "parameters_finite": bool(finite_tensor[2].item()),
                "peak_memory_allocated_bytes": peak_allocated,
                "peak_memory_reserved_bytes": peak_reserved,
                "peak_memory_fraction": (
                    peak_reserved / float(torch.cuda.get_device_properties(device).total_memory)
                    if device.type == "cuda" else 0.0
                ),
                "loader_workers": int(args.loader_workers),
                "loader_prefetch_factor": int(args.loader_prefetch_factor),
                "batch_balance": str(args.batch_balance),
                "stage": args.checkpoint_stage,
            }, sort_keys=True))
        if distributed:
            dist.barrier()
            # The benchmark is an intentional short-lived DDP job.  Tear
            # down the process group explicitly so PyTorch does not emit a
            # misleading leaked-process-group warning at interpreter exit.
            dist.destroy_process_group()
        return

    def save_train_state(epoch_number, next_step_idx):
        local_rng_state = _capture_rng_state()
        local_loader_state = loader_generator.get_state()
        rng_state_by_rank = _gather_rng_states(
            local_rng_state, distributed, rank, world_size
        )
        loader_state_by_rank = _gather_rank_states(
            local_loader_state, distributed, rank, world_size
        )
        sampler_state_by_rank = _gather_rank_states(
            {
                "epoch": int(epoch_number),
                "next_batch_index": int(next_step_idx),
                "seed": int(args.seed),
                "world_size": int(world_size),
            },
            distributed,
            rank,
            world_size,
        )
        if rank == 0:
            payload = {
                "schema": PRETRAIN_TRAIN_STATE_SCHEMA,
                "meta": {
                    **resume_contract,
                    "target_optimizer_steps": int(args.max_optimizer_steps),
                    "epoch_cap": int(args.epochs),
                    "random_seed": int(args.seed),
                    "loader_workers": int(args.loader_workers),
                    "loader_prefetch_factor": int(args.loader_prefetch_factor),
                    "sampler_type": (
                        type(sampler).__name__ if sampler is not None else "random"
                    ),
                    "batch_balance": str(args.batch_balance),
                    "warmup_ratio": float(args.warmup_ratio),
                    "optimizer_impl": "adam",
                    "matmul_precision": (
                        "high_tf32" if device.type == "cuda" else "default"
                    ),
                },
                "train_module": {
                    key: value.detach().cpu()
                    for key, value in train_module.state_dict().items()
                },
                "aux_modules": {
                    key: value.detach().cpu()
                    for key, value in aux_modules.state_dict().items()
                },
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": int(epoch_number),
                "next_step_idx": int(next_step_idx),
                "global_step": int(global_step),
                "optimizer_steps_completed": int(optimizer_steps_completed),
                "rng_state_by_rank": rng_state_by_rank,
                "loader_generator_state_by_rank": loader_state_by_rank,
                "sampler_state_by_rank": sampler_state_by_rank,
            }
            _atomic_torch_save(payload, train_state_path)
        # Checkpointing must be transparent to every rank's stochastic stream.
        # Synchronize the atomic rank-zero write, then restore the exact local
        # states captured before object collectives and serialization.
        if distributed:
            dist.barrier()
        _restore_rng_state(local_rng_state)
        loader_generator.set_state(local_loader_state)

    if args.dynamic_pretrain_loss:
        active_loss_names = []
        task_priors = None
        if args.pretrain_stage == 'scage_m4p':
            active_loss_names = [
                name for name, weight in (
                    ('mips_mask', args.scage_mips_mask_weight),
                    ('ecfp', args.scage_ecfp_weight),
                    ('masked_spd', args.mips_spd_weight),
                    ('path_bond', args.mips_path_bond_weight),
                    ('finite_distance', args.mips_distance_weight),
                    ('repeat_consistency', args.mips_repeat_consistency_weight),
                ) if float(weight) > 0
            ]
            task_priors = {
                'mips_mask': args.scage_mips_mask_weight,
                'ecfp': args.scage_ecfp_weight,
                'masked_spd': args.mips_spd_weight,
                'path_bond': args.mips_path_bond_weight,
                'finite_distance': args.mips_distance_weight,
                'repeat_consistency': args.mips_repeat_consistency_weight,
            }
        elif args.pretrain_stage == 'graph_geom':
            if graph_atom_head is not None and loss_weights['graph'] > 0:
                if float(args.graph_mask_atom_weight) > 0:
                    active_loss_names.append('graph_mask')
                if float(args.graph_periodic_aug_weight) > 0:
                    active_loss_names.append('graph_periodic_aug')
                if graph_sp_head is not None and float(args.graph_shortest_path_weight) > 0:
                    active_loss_names.append('shortest_path')
                if graph_angle_head is not None and float(args.graph_angle_weight) > 0:
                    active_loss_names.append('angle')
            if geom_noise_head is not None and loss_weights['geom'] > 0:
                active_loss_names.append('geom')
        else:
            if loss_weights['contrastive'] > 0:
                active_loss_names.append('contrastive')
            if graph_atom_head is not None and loss_weights['graph'] > 0:
                active_loss_names.append('graph')
            if graph_sp_head is not None and float(args.graph_shortest_path_weight) > 0:
                active_loss_names.append('shortest_path')
            if graph_angle_head is not None and float(args.graph_angle_weight) > 0:
                active_loss_names.append('angle')
            if geom_noise_head is not None and loss_weights['geom'] > 0:
                active_loss_names.append('geom')
        dynamic_loss_weighter = DynamicPretrainLossWeighter(
            active_loss_names,
            init_window=args.dynamic_loss_warmup_steps,
            recent_window=args.dynamic_loss_recent_window,
            temperature=args.dynamic_loss_temperature,
            task_priors=task_priors,
            effective_caps={'ecfp': args.scage_ecfp_effective_max},
            device=device,
        )
        print(f"Dynamic pretraining loss terms: {active_loss_names}")

    model.train()
    angle_v2_validation_history = []
    angle_v2_validation_reference = None
    angle_v2_best_composite = float('inf')
    angle_v2_best_model_state = None
    joint_progress = None
    if args.pretrain_stage == MTS_STAGE1_ID and rank == 0:
        joint_progress = tqdm(
            total=int(args.max_optimizer_steps),
            initial=int(optimizer_steps_completed),
            desc="MTS Joint Pretraining",
            unit="step",
            disable=not (
                sys.stderr.isatty()
                and os.environ.get("MIPS_TQDM", "1") == "1"
            ),
        )
    structured_log_time = time.monotonic()
    structured_log_step = optimizer_steps_completed
    fixed_compare_path = os.environ.get("MTS_FIXED_BATCH_COMPARE", "").strip()
    fixed_compare_done = False
    for epoch in range(args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        if epoch < resume_epoch:
            continue
        first_step_idx = resume_step_idx if epoch == resume_epoch else 0
        epoch_loss = 0.0
        epoch_periodic_aug_stats = _empty_periodic_aug_stats()
        epoch_term_steps = 0
        show_progress = (
            rank == 0
            and sys.stderr.isatty()
            and os.environ.get("MIPS_TQDM", "1") == "1"
            and args.pretrain_stage != MTS_STAGE1_ID
        )
        progress_bar = tqdm(
            dataloader,
            desc=f"Pretraining Epoch {epoch + 1}/{args.epochs}",
            disable=not show_progress,
        )
        optimizer.zero_grad(set_to_none=True)
        # Reconstruct the iterator position without fetching the first
        # optimized batch under a stale RNG state.  The collate function may
        # sample a 3D conformer with torch.randint(); restoring only after
        # that batch has been fetched changes the input (and can consume a
        # different number of random draws inside the loss).  Consume the
        # skipped batches first, restore the checkpointed per-rank RNG, then
        # fetch the first actual optimization batch.
        direct_sampler_resume = bool(
            first_step_idx and hasattr(sampler, "set_start_batch")
        )
        if direct_sampler_resume:
            sampler.set_start_batch(first_step_idx)
        data_iterator = iter(progress_bar)
        if resume_loader_state is not None:
            # Creating a DataLoader iterator may advance its generator while
            # establishing worker seeds.  Restore the checkpointed stream
            # after iterator construction so resumed workers observe the same
            # generator state as the uninterrupted trajectory.
            loader_generator.set_state(resume_loader_state)
            resume_loader_state = None
        if first_step_idx and not direct_sampler_resume:
            for _ in range(first_step_idx):
                try:
                    next(data_iterator)
                except StopIteration:
                    break
        if resume_rng_state is not None:
            _restore_rng_state(resume_rng_state)
            resume_rng_state = None
        for step_idx, data in enumerate(data_iterator, start=first_step_idx):
            if (
                (args.max_steps > 0 and global_step >= args.max_steps)
                or (
                    loop_max_optimizer_steps > 0
                    and optimizer_steps_completed >= loop_max_optimizer_steps
                )
            ):
                break
            global_step += 1
            data = data.to(device, non_blocking=True)
            trace_record = None
            if (
                args.pretrain_stage == MTS_STAGE1_ID
                and rank == 0
                and os.environ.get("MTS_RESUME_TRACE")
            ):
                mask_preview = _joint_canonical_mask(
                    data, args.seed, global_step, args.graph_mask_ratio
                )
                trace_record = {
                    "global_step": int(global_step),
                    "optimizer_step": int(
                        optimizer_steps_completed + (1 if ((step_idx + 1) % accumulation_steps == 0) else 0)
                    ),
                    "sample_count": int(len(data.smiles)),
                    "masked_atom_count": int(mask_preview.sum().item()),
                }
            should_step = (
                (step_idx + 1) % accumulation_steps == 0
                or (step_idx + 1) == len(dataloader)
                or (args.max_steps > 0 and global_step >= args.max_steps)
            )
            base_model = _base_model(model)
            loss_terms = {}

            if args.pretrain_stage == MTS_STAGE1_ID:
                sync_context = (
                    train_module.no_sync()
                    if distributed and not should_step else nullcontext()
                )
                with sync_context:
                    with torch.autocast(
                        device_type='cuda', dtype=torch.bfloat16,
                        enabled=args.amp_dtype == 'bf16' and device.type == 'cuda',
                    ):
                        payload = train_module(
                            MTS_STAGE1_ID, data, args, global_step
                        )
                        global_counts = _distributed_joint_counts(
                            payload["counts"], device
                        )
                        global_atom_count = global_counts["masked_atoms"]
                        global_angle_count = global_counts["angle_graphs"]
                        global_masked_correct = global_counts["masked_correct"]
                        global_angle_correct = global_counts["angle_correct"]
                        global_angle_targets = global_counts["angle_targets"]
                        global_mcl_valid = global_counts["mcl_valid_graphs"]
                        global_graphs = global_counts["graphs"]
                        atom_mean = _global_mean_from_count(
                            payload["loss_terms"]["masked_atom_sum"],
                            global_atom_count,
                        )
                        angle_mean = _global_mean_from_count(
                            payload["loss_terms"]["angle_sum"],
                            global_angle_count,
                        )
                        loss = compose_joint_payload_loss(
                            payload,
                            args,
                            global_counts,
                        )
                        if fixed_compare_path and rank == 0 and not fixed_compare_done:
                            # Compare the extracted objective against the
                            # compatibility spelling on the first real
                            # production batch.  The comparison is made
                            # before backward, so it covers the exact loss
                            # terms, globally synchronized counts and both
                            # autograd paths without changing the training
                            # graph or RNG stream.
                            from src.training.pretrain.objective import (
                                joint_masked_atom_angle_loss,
                            )
                            reference_loss = joint_masked_atom_angle_loss(
                                payload,
                                args,
                                atom_count=global_atom_count,
                                angle_count=global_angle_count,
                            )
                            torch.testing.assert_close(
                                loss.detach(), reference_loss.detach(),
                                rtol=1e-6, atol=1e-7,
                            )
                            compare_params = [
                                parameter for parameter in train_module.parameters()
                                if parameter.requires_grad
                            ]
                            new_grads = torch.autograd.grad(
                                loss, compare_params, retain_graph=True,
                                allow_unused=True,
                            )
                            old_grads = torch.autograd.grad(
                                reference_loss, compare_params, retain_graph=True,
                                allow_unused=True,
                            )
                            grad_max_abs = 0.0
                            grad_cosine = 1.0
                            flat_new = []
                            flat_old = []
                            for new_grad, old_grad in zip(new_grads, old_grads):
                                if new_grad is None and old_grad is None:
                                    continue
                                if new_grad is None or old_grad is None:
                                    raise RuntimeError(
                                        "fixed-batch objective gradient presence mismatch"
                                    )
                                torch.testing.assert_close(
                                    new_grad, old_grad, rtol=1e-6, atol=1e-7
                                )
                                grad_max_abs = max(
                                    grad_max_abs,
                                    float((new_grad - old_grad).abs().max().item()),
                                )
                                flat_new.append(new_grad.detach().float().reshape(-1))
                                flat_old.append(old_grad.detach().float().reshape(-1))
                            if flat_new:
                                left = torch.cat(flat_new)
                                right = torch.cat(flat_old)
                                grad_cosine = float(
                                    torch.nn.functional.cosine_similarity(
                                        left, right, dim=0
                                    ).item()
                                )
                            fixed_compare_path_obj = Path(fixed_compare_path)
                            fixed_compare_path_obj.parent.mkdir(parents=True, exist_ok=True)
                            fixed_compare_path_obj.write_text(
                                json.dumps({
                                    "schema": "mts-real-fixed-batch-compare-v1",
                                    "sample_count": int(len(data.smiles)),
                                    "masked_atoms": int(global_atom_count),
                                    "angle_graphs": int(global_angle_count),
                                    "loss_new": float(loss.detach().float().item()),
                                    "loss_reference": float(reference_loss.detach().float().item()),
                                    "loss_abs_diff": float(
                                        (loss.detach() - reference_loss.detach()).abs().item()
                                    ),
                                    "gradient_max_abs_diff": grad_max_abs,
                                    "gradient_cosine": grad_cosine,
                                    "terms": {
                                        "masked_atom_sum": float(
                                            payload["loss_terms"]["masked_atom_sum"].detach().float().item()
                                        ),
                                        "angle_sum": float(
                                            payload["loss_terms"]["angle_sum"].detach().float().item()
                                        ),
                                        "zero_reference": float(
                                            payload["zero_reference"].detach().float().item()
                                        ),
                                    },
                                    "finite": bool(torch.isfinite(loss).item()),
                                    "optimizer_step_executed": True,
                                }, sort_keys=True, indent=2) + "\n",
                                encoding="utf-8",
                            )
                            fixed_compare_done = True
                    if not torch.isfinite(loss):
                        raise ValueError(
                            f"Non-finite MTS joint loss for batch smiles={data.smiles}"
                        )
                    (loss / accumulation_steps).backward()
                if trace_record is not None:
                    trace_record.update({
                        "loss": float(loss.detach().float().item()),
                        "lr": float(optimizer.param_groups[0]["lr"]),
                    })
                    with open(os.environ["MTS_RESUME_TRACE"], "a", encoding="utf-8") as trace_handle:
                        trace_handle.write(json.dumps(trace_record, sort_keys=True) + "\n")
                gradient_norm = loss.new_tensor(0.0)
                gradient_diag = None
                if should_step:
                    # Clipping is disabled for the fixed MTS runtime, but the
                    # diagnostic norm must still reflect the actual gradient.
                    gradient_norm = loss.new_tensor(_module_gradient_norm(train_module))
                    if (
                        optimizer_steps_completed < 3
                        or (optimizer_steps_completed + 1) % 50 == 0
                    ):
                        graph_encoder = _base_model(model).encoders["graph"].encoder
                        gradient_diag = {
                            "o8": _module_gradient_diagnostic(graph_encoder.layers),
                            "star_rbf": _module_gradient_diagnostic(graph_encoder.star_distance_bias),
                            "trimer_mcl": _module_gradient_diagnostic(graph_encoder.trimer_mcl),
                            "geometry_gate": _module_gradient_diagnostic(
                                getattr(graph_encoder.trimer_mcl, "geometry_gate", None)
                            ),
                            "masked_atom_head": _module_gradient_diagnostic(mips_atom_head),
                            "angle_head": _module_gradient_diagnostic(graph_angle_head),
                        }
                    if float(args.max_grad_norm) > 0:
                        gradient_norm = torch.nn.utils.clip_grad_norm_(
                            train_module.parameters(), args.max_grad_norm
                        )
                    optimizer.step()
                    scheduler.step()
                    optimizer_steps_completed += 1
                    if joint_progress is not None:
                        joint_progress.update(1)
                        joint_progress.set_postfix(
                            loss=f"{loss.item():.4f}",
                            atom=f"{atom_mean.item():.3f}",
                            angle=f"{angle_mean.item():.3f}",
                            acc=f"{global_masked_correct/max(1, global_atom_count):.2%}",
                            aacc=f"{global_angle_correct/max(1, global_angle_targets):.2%}",
                            mcl=f"{global_mcl_valid/max(1, global_graphs):.1%}",
                            lr=f"{optimizer.param_groups[0]['lr']:.2e}",
                        )
                    if (
                        rank == 0 and not show_progress and
                        (optimizer_steps_completed <= 3
                         or optimizer_steps_completed % 50 == 0
                         or optimizer_steps_completed >= loop_max_optimizer_steps)
                    ):
                        elapsed = max(1e-6, time.monotonic() - structured_log_time)
                        delta_steps = max(1, optimizer_steps_completed - structured_log_step)
                        effective_samples = (
                            delta_steps * int(args.batch_size)
                            * max(1, int(world_size)) * accumulation_steps
                        )
                        total_elapsed = max(1e-6, time.monotonic() - pretrain_started)
                        eta_seconds = total_elapsed * max(
                            0.0,
                            float(loop_max_optimizer_steps)
                            / max(1, optimizer_steps_completed) - 1.0,
                        )
                        peak_gib = (
                            torch.cuda.max_memory_allocated(device) / 2**30
                            if device.type == "cuda" else 0.0
                        )
                        print(
                            "[pretrain] "
                            f"stage={MTS_STAGE1_ID} "
                            f"step={optimizer_steps_completed}/{loop_max_optimizer_steps} "
                            f"samples_per_s={effective_samples / elapsed:.2f} "
                            f"loss={float(loss.detach().float().item()):.5f} "
                            f"atom={float(atom_mean.detach().float().item()):.5f} "
                            f"angle={float(angle_mean.detach().float().item()):.5f} "
                            f"atom_acc={global_masked_correct/max(1, global_atom_count):.4f} "
                            f"angle_acc={global_angle_correct/max(1, global_angle_targets):.4f} "
                            f"angle_targets={global_angle_targets} "
                            f"angle_majority={getattr(args, 'angle_majority_baseline', 0.0):.4f} "
                            f"mcl_rate={global_mcl_valid/max(1, global_graphs):.4f} "
                            f"grad_norm={float(gradient_norm.detach().float().item()):.6g} "
                            f"peak_mem_gib={peak_gib:.2f} "
                            f"eta_h={eta_seconds/3600.0:.2f}",
                            flush=True,
                        )
                        if gradient_diag is not None:
                            print(
                                "[pretrain-gradient] "
                                + json.dumps({
                                    "step": int(optimizer_steps_completed),
                                    "modules": gradient_diag,
                                }, sort_keys=True),
                                flush=True,
                            )
                        structured_log_time = time.monotonic()
                        structured_log_step = optimizer_steps_completed
                    if (
                        optimizer_steps_completed <= 3
                        or optimizer_steps_completed % 500 == 0
                        or optimizer_steps_completed >= loop_max_optimizer_steps
                    ):
                        _assert_ddp_parameters_synced(
                            train_module, optimizer_steps_completed
                        )
                    optimizer.zero_grad(set_to_none=True)
                    if (
                        args.checkpoint_interval_steps > 0
                        and (
                            optimizer_steps_completed % args.checkpoint_interval_steps == 0
                            or optimizer_steps_completed >= loop_max_optimizer_steps
                        )
                    ):
                        next_epoch = epoch + 1 if step_idx + 1 >= len(dataloader) else epoch
                        next_step = 0 if next_epoch > epoch else step_idx + 1
                        save_train_state(next_epoch, next_step)
                    if (
                        args.angle_objective == 'cosine'
                        and optimizer_steps_completed > 0
                        and optimizer_steps_completed % 2000 == 0
                    ):
                        atom_val, angle_val = _evaluate_angle_v2_validation(
                            train_module, angle_validation_loader, args, device
                        )
                        if angle_v2_validation_reference is None:
                            angle_v2_validation_reference = (
                                max(atom_val, 1e-12), max(angle_val, 1e-12)
                            )
                        composite = (
                            atom_val / angle_v2_validation_reference[0]
                            + angle_val / angle_v2_validation_reference[1]
                        )
                        if rank == 0:
                            record = {
                                'optimizer_step': int(optimizer_steps_completed),
                                'masked_atom_ce': float(atom_val),
                                'angle_cos_mae': float(angle_val),
                                'normalized_composite': float(composite),
                            }
                            angle_v2_validation_history.append(record)
                            if composite < angle_v2_best_composite:
                                angle_v2_best_composite = float(composite)
                                angle_v2_best_model_state = {
                                    key: value.detach().cpu().clone()
                                    for key, value in _base_model(model).state_dict().items()
                                }
                epoch_loss += float(loss.detach().item())
                epoch_term_steps += 1
                if show_progress:
                    progress_bar.set_postfix(
                        step=f"{optimizer_steps_completed}/{loop_max_optimizer_steps}",
                        loss=f"{loss.item():.4f}",
                        atom=f"{atom_mean.item():.3f}",
                        angle=f"{angle_mean.item():.3f}",
                    )
                continue

            if args.pretrain_stage == 'scage_m4p':
                sync_context = (
                    train_module.no_sync()
                    if distributed and not should_step else nullcontext()
                )
                with sync_context:
                    with torch.autocast(
                        device_type='cuda', dtype=torch.bfloat16,
                        enabled=args.amp_dtype == 'bf16' and device.type == 'cuda',
                    ):
                        payload = train_module(
                            (
                                "stage2_geometry_adapt"
                                if args.geometry_adapt else "mips_stage1"
                            ),
                            data, args, epoch,
                        )
                    loss_terms = payload["loss_terms"]
                    zero_reference = payload["zero_reference"]
                    if dynamic_loss_weighter is not None:
                        loss = dynamic_loss_weighter(
                            loss_terms, reference=zero_reference
                        )
                        dynamic_weights = dynamic_loss_weighter.last_weights
                    else:
                        priors = {
                            'mips_mask': args.scage_mips_mask_weight,
                            'masked_spd': args.mips_spd_weight,
                            'path_bond': args.mips_path_bond_weight,
                        }
                        loss = (
                            sum(
                                priors[name] * value
                                for name, value in loss_terms.items()
                            )
                            if loss_terms else zero_reference
                        )
                        dynamic_weights = {}
                    # Include inactive-head zero anchors in the scalar loss;
                    # otherwise DDP cannot see those parameters even though
                    # they are part of the fixed container.
                    loss = loss + payload["zero_reference"]
                    if not torch.isfinite(loss):
                        raise ValueError(
                            f"Non-finite MIPS Stage 1 loss for batch "
                            f"smiles={data.smiles}"
                        )
                    (loss / accumulation_steps).backward()
                gradient_norm = loss.new_tensor(0.0)
                if should_step:
                    gradient_norm = torch.nn.utils.clip_grad_norm_(
                        train_container.parameters(), args.max_grad_norm
                    )
                    optimizer.step()
                    scheduler.step()
                    optimizer_steps_completed += 1
                    if (
                        rank == 0
                        and not show_progress
                        and (
                            optimizer_steps_completed == 1
                            or optimizer_steps_completed % 50 == 0
                            or (
                                args.max_optimizer_steps > 0
                                and optimizer_steps_completed >= args.max_optimizer_steps
                            )
                        )
                    ):
                        elapsed = max(1e-6, time.monotonic() - structured_log_time)
                        delta_steps = optimizer_steps_completed - structured_log_step
                        effective_samples = (
                            delta_steps * int(args.batch_size)
                            * max(1, int(world_size)) * accumulation_steps
                        )
                        print(
                            "[pretrain] "
                            f"stage={args.checkpoint_stage} "
                            f"step={optimizer_steps_completed}/{args.max_optimizer_steps} "
                            f"samples_per_s={effective_samples / elapsed:.2f} "
                            f"loss={float(loss.detach().float().item()):.5f}",
                            flush=True,
                        )
                        structured_log_time = time.monotonic()
                        structured_log_step = optimizer_steps_completed
                    if (
                        optimizer_steps_completed <= 3
                        or optimizer_steps_completed % 500 == 0
                        or (
                            args.max_optimizer_steps > 0
                            and optimizer_steps_completed >= args.max_optimizer_steps
                        )
                    ):
                        _assert_ddp_parameters_synced(
                            train_module, optimizer_steps_completed
                        )
                    optimizer.zero_grad(set_to_none=True)
                    if (
                        args.checkpoint_interval_steps > 0
                        and (
                            optimizer_steps_completed
                            % args.checkpoint_interval_steps == 0
                            or (
                                args.max_optimizer_steps > 0
                                and optimizer_steps_completed
                                >= args.max_optimizer_steps
                            )
                        )
                    ):
                        next_epoch = epoch + 1 if step_idx + 1 >= len(dataloader) else epoch
                        next_step = 0 if next_epoch > epoch else step_idx + 1
                        save_train_state(next_epoch, next_step)
                epoch_loss += float(loss.item())
                epoch_term_steps += 1
                if show_progress and (
                    should_step or optimizer_steps_completed % 50 == 0
                ):
                    progress_bar.set_postfix(
                        loss=f"{loss.item():.4f}",
                        mask=f"{payload['mask_loss'].item():.3f}",
                        sp=f"{payload['spd_loss'].item():.3f}",
                        path=f"{payload['path_loss'].item():.3f}",
                    )
                continue

            if args.pretrain_stage == 'alignment' and args.graph_encoder_type == 'mips_trimer_scage':
                sync_context = (
                    train_module.no_sync()
                    if distributed and not should_step else nullcontext()
                )
                with sync_context:
                    with torch.autocast(
                        device_type='cuda', dtype=torch.bfloat16,
                        enabled=args.amp_dtype == 'bf16' and device.type == 'cuda',
                    ):
                        payload = train_module(
                            "alignment", data, args, epoch
                        )
                        alignment_losses = payload["losses"]
                        loss = _alignment_total(alignment_losses, args)
                    if not torch.isfinite(loss):
                        raise ValueError(
                            f"Non-finite parallel alignment loss for batch "
                            f"smiles={data.smiles}"
                        )
                    (loss / accumulation_steps).backward()
                gradient_norm = loss.new_tensor(0.0)
                if should_step:
                    gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer_steps_completed += 1
                    if (
                        optimizer_steps_completed <= 3
                        or optimizer_steps_completed % 500 == 0
                        or (
                            args.max_optimizer_steps > 0
                            and optimizer_steps_completed >= args.max_optimizer_steps
                        )
                    ):
                        _assert_ddp_parameters_synced(
                            train_module, optimizer_steps_completed
                        )
                    optimizer.zero_grad(set_to_none=True)
                epoch_loss += float(loss.item())
                epoch_term_steps += 1
                progress_bar.set_postfix(
                    loss=f"{loss.item():.4f}",
                    fusion=f"{alignment_losses['fused_view'].item():.3f}",
                    gs=f"{alignment_losses['graph_smiles'].item():.3f}",
                    gf=f"{alignment_losses['graph_fp'].item():.3f}",
                    lomo=f"{alignment_losses['lomo'].item():.3f}",
                )
                continue

            geom_loss = data.y.new_tensor(0.0)
            if geom_noise_head is not None:
                geom_loss = _geom_denoise_loss(
                    base_model,
                    data,
                    geom_noise_head,
                    noise_std=args.geom_noise_std,
                    noise_std_min=args.geom_noise_std_min,
                    noise_std_max=args.geom_noise_std_max,
                )
                loss_terms['geom'] = geom_loss

            contrastive_loss = data.y.new_tensor(0.0)
            if loss_weights['contrastive'] > 0:
                _, embeddings = model(data)  # embeddings: [batch_size, num_modalities, embedding_dim]
                contrastive_loss = compute_contrastive_loss(embeddings, temperature=args.temperature)
                loss_terms['contrastive'] = contrastive_loss

            graph_loss = data.y.new_tensor(0.0)
            if graph_atom_head is not None:
                graph_loss, graph_mask_loss, graph_paug_loss, graph_paug_stats = _graph_pretrain_loss(
                    base_model,
                    data,
                    graph_atom_head,
                    mask_ratio=args.graph_mask_ratio,
                    mask_weight=args.graph_mask_atom_weight,
                    periodic_aug_weight=args.graph_periodic_aug_weight,
                    graph_input=args.graph_input,
                    repeat_cut_max_mrus=args.repeat_cut_max_mrus,
                    repeat_cut_retry=args.repeat_cut_retry,
                    repeat_cut_temperature=args.repeat_cut_temperature,
                )
                _merge_periodic_aug_stats(epoch_periodic_aug_stats, graph_paug_stats)
                loss_terms['graph'] = graph_loss
            else:
                graph_mask_loss = data.y.new_tensor(0.0)
                graph_paug_loss = data.y.new_tensor(0.0)
                graph_paug_stats = _empty_periodic_aug_stats()

            graph_sp_loss = data.y.new_tensor(0.0)
            if graph_sp_head is not None and float(args.graph_shortest_path_weight) > 0:
                graph_sp_loss = _graph_shortest_path_loss(
                    base_model,
                    data,
                    graph_sp_head,
                    max_distance=args.scage_sp_max_distance,
                )

            graph_angle_loss = None
            if graph_angle_head is not None and float(args.graph_angle_weight) > 0:
                graph_angle_loss = _graph_angle_loss(
                    base_model,
                    data,
                    graph_angle_head,
                    angle_bins=args.scage_angle_bins,
                )
            graph_angle_log = graph_angle_loss if graph_angle_loss is not None else data.y.new_tensor(0.0)

            weighted_loss_terms = {}
            if args.pretrain_stage == 'graph_geom':
                if graph_atom_head is not None and loss_weights['graph'] > 0 and 'graph' in loss_terms:
                    if float(args.graph_mask_atom_weight) > 0:
                        weighted_loss_terms['graph_mask'] = float(args.graph_mask_atom_weight) * graph_mask_loss
                    if float(args.graph_periodic_aug_weight) > 0:
                        weighted_loss_terms['graph_periodic_aug'] = float(args.graph_periodic_aug_weight) * graph_paug_loss
                    if graph_sp_head is not None and float(args.graph_shortest_path_weight) > 0:
                        weighted_loss_terms['shortest_path'] = float(args.graph_shortest_path_weight) * graph_sp_loss
                    if graph_angle_loss is not None and float(args.graph_angle_weight) > 0:
                        weighted_loss_terms['angle'] = float(args.graph_angle_weight) * graph_angle_loss
                if geom_noise_head is not None and loss_weights['geom'] > 0 and 'geom' in loss_terms:
                    weighted_loss_terms['geom'] = loss_weights['geom'] * geom_loss
            else:
                if loss_weights['contrastive'] > 0 and 'contrastive' in loss_terms:
                    weighted_loss_terms['contrastive'] = loss_weights['contrastive'] * contrastive_loss
                if graph_atom_head is not None and loss_weights['graph'] > 0 and 'graph' in loss_terms:
                    weighted_loss_terms['graph'] = loss_weights['graph'] * graph_loss
                if graph_sp_head is not None and float(args.graph_shortest_path_weight) > 0:
                    weighted_loss_terms['shortest_path'] = float(args.graph_shortest_path_weight) * graph_sp_loss
                if graph_angle_loss is not None and float(args.graph_angle_weight) > 0:
                    weighted_loss_terms['angle'] = float(args.graph_angle_weight) * graph_angle_loss
                if geom_noise_head is not None and loss_weights['geom'] > 0 and 'geom' in loss_terms:
                    weighted_loss_terms['geom'] = loss_weights['geom'] * geom_loss

            if dynamic_loss_weighter is not None:
                loss = dynamic_loss_weighter(weighted_loss_terms)
                dynamic_weights = dynamic_loss_weighter.last_weights
            elif args.pretrain_stage == 'graph_geom' and weighted_loss_terms:
                loss = sum(weighted_loss_terms.values()) / len(weighted_loss_terms)
                dynamic_weights = {}
            else:
                loss = sum(weighted_loss_terms.values()) if weighted_loss_terms else data.y.new_tensor(0.0)
                dynamic_weights = {}


            if not torch.isfinite(loss):
                raise ValueError(f"Non-finite pretraining loss for batch smiles={getattr(data, 'smiles', [])}")
            (loss / accumulation_steps).backward()
            gradient_norm = loss.new_tensor(0.0)
            if should_step:
                _all_reduce_gradients((model, aux_modules))
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    list(model.parameters()) + list(aux_modules.parameters()), max_norm=args.max_grad_norm
                )
                optimizer.step()
                scheduler.step()
                optimizer_steps_completed += 1
                optimizer.zero_grad(set_to_none=True)
                if (
                    args.checkpoint_interval_steps > 0
                    and (
                        optimizer_steps_completed
                        % args.checkpoint_interval_steps == 0
                        or (
                            args.max_optimizer_steps > 0
                            and optimizer_steps_completed >= args.max_optimizer_steps
                        )
                    )
                ):
                    next_epoch = epoch + 1 if step_idx + 1 >= len(dataloader) else epoch
                    next_step = 0 if next_epoch > epoch else step_idx + 1
                    save_train_state(next_epoch, next_step)
            epoch_loss += loss.item()
            epoch_term_steps += 1
            paug_attempted = int(graph_paug_stats.get('attempted', 0))
            paug_success = int(graph_paug_stats.get('success', 0))
            paug_rate = (paug_success / paug_attempted) if paug_attempted else 0.0
            postfix = {
                'loss': f"{loss.item():.4f}",
                'con': f"{contrastive_loss.item():.4f}",
                'g_mask': f"{graph_mask_loss.item():.4f}",
                'g_paug': f"{graph_paug_loss.item():.4f}",
                'paug_rate': f"{paug_rate:.2f}",
                'paug_valid': int(graph_paug_stats.get('valid_contrastive', 0)),
                'paug_same': int(graph_paug_stats.get('same_smiles', 0)),
                'geom': f"{geom_loss.item():.4f}",
                'sp': f"{graph_sp_loss.item():.4f}",
                'angle': f"{graph_angle_log.item():.4f}",
            }
            for name, value in dynamic_weights.items():
                postfix[f'w_{name}'] = f"{value:.2f}"
            progress_bar.set_postfix(**postfix)
        epoch_summary = torch.tensor(
            [float(epoch_loss), float(epoch_term_steps)], dtype=torch.float64, device=device
        )
        if distributed:
            dist.all_reduce(epoch_summary, op=dist.ReduceOp.SUM)
        global_epoch_steps = max(1, int(round(epoch_summary[1].item())))
        avg_loss = float(epoch_summary[0].item()) / global_epoch_steps
        epoch_periodic_aug_stats = _distributed_sum_mapping(
            epoch_periodic_aug_stats, device
        )
        paug_summary = _finalize_periodic_aug_stats(epoch + 1, epoch_periodic_aug_stats)
        if rank == 0 and paug_summary['attempted'] > 0:
            print(
                f"Epoch [{epoch+1}/{args.epochs}] Total Loss: {avg_loss:.4f} | "
                f"PerioGT aug success: {paug_summary['success']}/{paug_summary['attempted']} "
                f"= {paug_summary['success_rate']:.2%}, "
                f"same_smiles={paug_summary['same_smiles']}, skipped={paug_summary['skipped']}"
            )
        elif rank == 0:
            print(f"Epoch [{epoch+1}/{args.epochs}] Total Loss: {avg_loss:.4f}")
        if (
            (args.max_steps > 0 and global_step >= args.max_steps)
            or (
                loop_max_optimizer_steps > 0
                and optimizer_steps_completed >= loop_max_optimizer_steps
            )
        ):
            break

    if joint_progress is not None:
        joint_progress.close()

    if distributed and rank != 0:
        dist.barrier()
        dist.destroy_process_group()
        return

    if args.angle_objective == 'cosine':
        if angle_v2_best_model_state is None:
            raise RuntimeError('Angle-v2 produced no 2,000-step validation checkpoint')
        _base_model(model).load_state_dict(angle_v2_best_model_state, strict=True)
        print(
            'Angle-v2 export selected the minimum validation composite: '
            f'{angle_v2_best_composite:.6f}'
        )

    state_dict = model.state_dict()
    # Formal checkpoints are deliberately minimal: the final file is for
    # downstream loading only, while .last.pt remains the sole resume state.
    # Publish the marker only after a round-trip and strict load against the
    # already-constructed model have succeeded.
    from src.training.common.checkpoint import save_final_state

    if rank == 0:
        def _strict_final_load(candidate_state):
            model.load_state_dict(candidate_state, strict=True)
            for value in candidate_state.values():
                if torch.is_tensor(value) and (
                    value.is_floating_point() or value.is_complex()
                ) and not torch.isfinite(value).all():
                    raise RuntimeError(
                        "final checkpoint contains non-finite parameters"
                    )

        save_final_state(
            state_dict,
            args.save_path,
            strict_load=_strict_final_load,
        )
    if distributed:
        dist.barrier()
        dist.destroy_process_group()
    print(f"Pretrained model saved at {args.save_path}")
    if distributed:
        dist.barrier()
        dist.destroy_process_group()

def main():
    return run_pretrain(parse_arguments())


if __name__ == "__main__":
    main()
