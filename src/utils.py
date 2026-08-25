import copy
import os
import random
import time
import torch
from contextlib import nullcontext
from functools import partial
from torch.utils.data import Subset
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import sklearn.metrics as metrics
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from src.dataset.dataloader import custom_collate, mips_trimer_collate


def set_global_seed(seed, deterministic=True):
    """Seed Python, NumPy, and PyTorch for reproducible experiment comparisons."""
    seed = int(seed)
    os.environ.setdefault('PYTHONHASHSEED', str(seed))
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)
    return seed


class TargetScaler:
    def __init__(self, task, scaler, eps=1e-4, transform_mode='auto'):
        self.task = task
        self.scaler = scaler
        self.eps = eps
        self.requested_transform_mode = transform_mode
        self.transform_mode = (
            'log' if transform_mode == 'recommended' and task in {'eps', 'nc'}
            else ('standard' if transform_mode == 'recommended' else transform_mode)
        )

    def _pre_transform(self, y):
        y = np.asarray(y, dtype=np.float64).reshape(-1, 1)
        if self.transform_mode == 'log':
            if np.any(y <= 0):
                raise ValueError('Log target transform requires strictly positive labels')
            return np.log(y)
        if self.transform_mode == 'auto' and self.task in ['er', 'iv']:
            return np.log10(y)
        if self.transform_mode == 'auto' and self.task == 'xc':
            # Xc is a bounded percentage target. A clipped logit maps it to an
            # unbounded space before standardization and avoids 0/100 singularities.
            p = np.clip(y / 100.0, self.eps, 1.0 - self.eps)
            return np.log(p / (1.0 - p))
        return y

    def _inverse_pre_transform(self, y):
        y = np.asarray(y, dtype=np.float64).reshape(-1, 1)
        if self.transform_mode == 'log':
            return np.exp(y)
        if self.transform_mode == 'auto' and self.task == 'xc':
            return 100.0 / (1.0 + np.exp(-y))
        return y

    def transform(self, y):
        return self.scaler.transform(self._pre_transform(y))

    def inverse_transform(self, y):
        y = np.asarray(y, dtype=np.float64).reshape(-1, 1)
        return self._inverse_pre_transform(self.scaler.inverse_transform(y))


def scale_targets(dataset, task, train_indices=None, raw_targets=None, transform_mode='auto'):
    scaler = StandardScaler()
    if raw_targets is None:
        if hasattr(dataset, "raw_targets"):
            raw_targets = np.asarray(dataset.raw_targets, dtype=np.float64)
        else:
            raw_targets = np.array(
                [data.y.item() for data in dataset], dtype=np.float64
            )
    else:
        raw_targets = np.asarray(raw_targets, dtype=np.float64)

    if train_indices is None:
        train_indices = np.arange(len(raw_targets))

    if transform_mode not in {'auto', 'standard', 'log', 'recommended'}:
        raise ValueError(f"Unsupported target transform: {transform_mode}")
    target_scaler = TargetScaler(task, scaler, transform_mode=transform_mode)
    train_values = raw_targets[train_indices].reshape(-1, 1)
    scaler.fit(target_scaler._pre_transform(train_values))

    print("Scaling y values with mean:", scaler.mean_[0], "and std:", scaler.scale_[0])
    resolved_mode = target_scaler.transform_mode
    if resolved_mode == 'auto' and task == 'xc':
        print("Using clipped logit transform for bounded percentage target: xc")
    elif resolved_mode == 'standard':
        print("Using direct target standardization without task-specific transform")
    elif resolved_mode == 'log':
        print("Using natural-log target transform before standardization")

    scaled_targets = target_scaler.transform(raw_targets.reshape(-1, 1))
    if hasattr(dataset, "set_target_override"):
        dataset.set_target_override(scaled_targets.reshape(-1))
    else:
        for data, value in zip(dataset, scaled_targets):
            data.y = torch.tensor(value, dtype=torch.float)

    return target_scaler

def get_data_loader(
    dataset, indices=None, batch_size=32, shuffle=False, drop_last=False,
    random_conformer=None, num_workers=0, pin_memory=None, persistent_workers=None,
    sampler=None, generator=None, prefetch_factor=2,
):
    loader_dataset = (
        dataset
        if indices is None
        else Subset(dataset, [int(index) for index in indices])
    )
    if random_conformer is None:
        random_conformer = bool(shuffle)

    # The graph-only production route must not allocate the legacy multimodal
    # collator's SMILES/FP/PBC/geometry tensors.  ``dataset`` is the original
    # UniDataset here (before Subset wrapping), so this branch is stable for
    # both pretraining and downstream loaders.
    collate = (
        mips_trimer_collate
        if bool(getattr(dataset, "is_mts_route", False))
        else partial(custom_collate, random_conformer=random_conformer)
    )
    loader_kwargs = dict(
        dataset=loader_dataset,
        batch_size=batch_size,
        collate_fn=collate,
        shuffle=bool(shuffle and sampler is None),
        sampler=sampler,
        drop_last=drop_last,
        num_workers=int(num_workers),
        pin_memory=torch.cuda.is_available() if pin_memory is None else bool(pin_memory),
        persistent_workers=(int(num_workers) > 0) if persistent_workers is None else bool(persistent_workers),
        generator=generator,
    )
    # PyTorch rejects prefetch_factor when num_workers=0.  Keep the public
    # parameter available for deterministic benchmark/config hashing while
    # omitting it from the single-process loader.
    if int(num_workers) > 0:
        loader_kwargs["prefetch_factor"] = max(1, int(prefetch_factor))
    loader = DataLoader(**loader_kwargs)

    print(f"Created dataloader with {len(loader_dataset)} samples")
    return loader


def _autocast_context(device, amp_dtype):
    enabled = str(amp_dtype) == 'bf16' and torch.device(device).type == 'cuda'
    return (
        torch.autocast(device_type='cuda', dtype=torch.bfloat16)
        if enabled else nullcontext()
    )


def finetune_bf16_parity_gate(model, batch, criterion, device):
    """Gate downstream BF16 on the complete graph forward and gradients."""
    if torch.device(device).type != 'cuda' or not torch.cuda.is_bf16_supported():
        return False, {
            'relative_loss_delta': float('inf'),
            'gradient_cosine': float('nan'),
            'finite': False,
            'reason': 'cuda_bf16_unavailable',
        }
    batch = batch.to(device, non_blocking=True)
    was_training = model.training
    model.eval()

    def evaluate(amp_dtype):
        model.zero_grad(set_to_none=True)
        with _autocast_context(device, amp_dtype):
            outputs, _ = model(batch)
            loss = criterion(outputs, batch.y)
        loss.backward()
        values = [
            parameter.grad.detach().float().reshape(-1)
            for parameter in model.parameters()
            if parameter.requires_grad and parameter.grad is not None
        ]
        gradient = torch.cat(values) if values else loss.new_zeros(1).float()
        return loss.detach().float(), gradient

    fp32_loss, fp32_gradient = evaluate('fp32')
    bf16_loss, bf16_gradient = evaluate('bf16')
    delta = float(
        (bf16_loss - fp32_loss).abs()
        / fp32_loss.abs().clamp_min(1e-8)
    )
    cosine = float(torch.nn.functional.cosine_similarity(
        fp32_gradient, bf16_gradient, dim=0
    ).item())
    fp32_finite = bool(
        torch.isfinite(fp32_loss) and torch.isfinite(fp32_gradient).all()
    )
    bf16_finite = bool(
        torch.isfinite(bf16_loss) and torch.isfinite(bf16_gradient).all()
    )
    finite = fp32_finite and bf16_finite
    model.zero_grad(set_to_none=True)
    model.train(was_training)
    result = {
        'relative_loss_delta': delta,
        'gradient_cosine': cosine,
        'finite': finite,
        'fp32_finite': fp32_finite,
        'bf16_finite': bf16_finite,
    }
    # The fine-tuning handoff gates BF16 on finite loss/gradients and a small
    # loss delta only.  Gradient cosine remains a useful diagnostic, but it is
    # not a required full-gradient parity claim for this speed experiment.
    return bool(finite and delta <= 0.02), result

def train_epoch(
    model, train_loader, criterion, optimizer, scheduler, device, epoch=None,
    max_grad_norm=1.0, unimodal_aux_weight=0.0,
    fusion_prior_kl_weight=0.0, fusion_prior=(0.30, 0.40, 0.30),
    cross_task_aux_weight=0.0,
    pcgrad=False,
    amp_dtype='fp32',
    return_timing=False,
    mts_glt_frozen_encoders_eval=False,
    mts_glt_frozen_encoder_policy=None,
):
    model.train()
    if mts_glt_frozen_encoder_policy is not None:
        _set_mts_glt_frozen_encoders_eval(
            model, mts_glt_frozen_encoder_policy
        )
    elif mts_glt_frozen_encoders_eval:
        _set_mts_glt_frozen_encoders_eval(model)
    epoch_started = time.perf_counter()
    optimizer_steps = 0
    legacy_sync = os.environ.get('MTS_BENCHMARK_LEGACY_SYNC', '0') == '1'
    train_losses = []
    fused_losses = []
    auxiliary_losses = []
    fusion_prior_losses = []
    cross_task_aux_losses = []
    train_preds = []
    train_targets = []

    progress_bar = tqdm(train_loader, desc="Training")
    for batch in progress_bar:
        batch = batch.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with _autocast_context(device, amp_dtype):
            outputs, embeddings = model(batch)
        multitask_component_losses = []
        if hasattr(batch, 'mts_task_index'):
            task_index = batch.mts_task_index.long().reshape(-1)
            auxiliary_predictions = _base_model(model).predict_cross_tasks(
                _base_model(model).unimodal_embedding
            )
            selected_outputs = outputs.clone()
            for index in range(1, auxiliary_predictions.size(1) + 1):
                selected_outputs[task_index == index] = auxiliary_predictions[
                    task_index == index, index - 1:index
                ]
            for index in torch.unique(task_index).tolist():
                valid = task_index == int(index)
                multitask_component_losses.append(
                    criterion(selected_outputs[valid], batch.y[valid])
                )
            outputs = selected_outputs
            fused_loss = torch.stack(multitask_component_losses).mean()
        else:
            fused_loss = criterion(outputs, batch.y)
        auxiliary_loss = fused_loss.new_zeros(())
        if float(unimodal_aux_weight) > 0:
            base_model = _base_model(model)
            modality_outputs = base_model.predict_modalities(embeddings)
            intrinsic_mask = base_model.intrinsic_availability_mask(
                batch, device=embeddings.device
            )
            modality_losses = []
            for idx in range(modality_outputs.size(1)):
                valid = intrinsic_mask[:, idx]
                if bool(valid.any()):
                    modality_losses.append(
                        criterion(modality_outputs[valid, idx], batch.y[valid])
                    )
            if modality_losses:
                auxiliary_loss = torch.stack(modality_losses).mean()
        fusion_prior_loss = fused_loss.new_zeros(())
        if float(fusion_prior_kl_weight) > 0:
            weights = getattr(_base_model(model), 'attention_visual_weights', None)
            if weights is None or weights.ndim != 2 or weights.size(1) != len(fusion_prior):
                raise ValueError(
                    "Fusion-prior KL requires parallel attention weights matching fusion_prior"
                )
            base_model = _base_model(model)
            intrinsic_mask = base_model.intrinsic_availability_mask(
                batch, device=weights.device
            )
            prior = weights.new_tensor(fusion_prior).clamp_min(1e-8)
            sample_prior = prior.unsqueeze(0) * intrinsic_mask.to(weights.dtype)
            sample_prior = sample_prior / sample_prior.sum(
                dim=1, keepdim=True
            ).clamp_min(1e-8)
            safe_weights = weights.clamp_min(1e-8)
            fusion_prior_loss = (
                safe_weights
                * (safe_weights.log() - sample_prior.clamp_min(1e-8).log())
                * intrinsic_mask
            ).sum(dim=1).mean()
        cross_task_aux_loss = fused_loss.new_zeros(())
        cross_task_component_losses = []
        if float(cross_task_aux_weight) > 0 and not hasattr(batch, 'mts_task_index'):
            auxiliary_targets = getattr(batch, 'cross_task_aux_y', None)
            auxiliary_mask = getattr(batch, 'cross_task_aux_mask', None)
            if auxiliary_targets is None or auxiliary_mask is None:
                raise ValueError(
                    'Cross-task auxiliary supervision requires targets and masks'
                )
            auxiliary_predictions = _base_model(model).predict_cross_tasks(
                _base_model(model).unimodal_embedding
            )
            if auxiliary_predictions.shape != auxiliary_targets.shape:
                raise ValueError(
                    'Cross-task prediction/target shape mismatch: '
                    f'{tuple(auxiliary_predictions.shape)} vs '
                    f'{tuple(auxiliary_targets.shape)}'
                )
            for task_index in range(auxiliary_predictions.size(1)):
                valid = auxiliary_mask[:, task_index]
                if bool(valid.any()):
                    cross_task_component_losses.append(criterion(
                        auxiliary_predictions[valid, task_index],
                        auxiliary_targets[valid, task_index],
                    ))
            if cross_task_component_losses:
                cross_task_aux_loss = torch.stack(cross_task_component_losses).mean()
        loss = (
            fused_loss
            + float(unimodal_aux_weight) * auxiliary_loss
            + float(fusion_prior_kl_weight) * fusion_prior_loss
            + float(cross_task_aux_weight) * cross_task_aux_loss
        )
        projected = None
        shared_parameters = None
        pcgrad_losses = (
            multitask_component_losses
            if multitask_component_losses else (
                [fused_loss] + [
                    float(cross_task_aux_weight) * value
                    for value in cross_task_component_losses
                ]
            )
        )
        if bool(pcgrad) and len(pcgrad_losses) > 1:
            base_model = _base_model(model)
            excluded = {
                id(parameter) for module in (
                    base_model.mlp, base_model.cross_task_aux_heads
                ) for parameter in module.parameters()
            }
            shared_parameters = [
                parameter for parameter in base_model.parameters()
                if parameter.requires_grad and id(parameter) not in excluded
            ]
            task_gradients = []
            for task_loss in pcgrad_losses:
                gradients = torch.autograd.grad(
                    task_loss, shared_parameters, retain_graph=True,
                    allow_unused=True,
                )
                task_gradients.append([
                    torch.zeros_like(parameter) if gradient is None else gradient
                    for parameter, gradient in zip(shared_parameters, gradients)
                ])
            projected = [[gradient.clone() for gradient in gradients]
                         for gradients in task_gradients]
            for left in range(len(projected)):
                for right in range(len(task_gradients)):
                    if left == right:
                        continue
                    dot = sum(
                        (a * b).sum() for a, b in
                        zip(projected[left], task_gradients[right])
                    )
                    norm = sum(
                        (value * value).sum()
                        for value in task_gradients[right]
                    ).clamp_min(1e-12)
                    if float(dot.detach()) < 0.0:
                        coefficient = dot / norm
                        projected[left] = [
                            a - coefficient * b for a, b in
                            zip(projected[left], task_gradients[right])
                        ]
        loss.backward()
        if projected is not None:
            for parameter, gradients in zip(shared_parameters, zip(*projected)):
                parameter.grad = torch.stack(list(gradients), dim=0).mean(dim=0)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
        optimizer.step()
        scheduler.step()
        optimizer_steps += 1
        if legacy_sync:
            train_losses.append(loss.item())
            fused_losses.append(fused_loss.item())
            auxiliary_losses.append(auxiliary_loss.item())
            fusion_prior_losses.append(fusion_prior_loss.item())
            cross_task_aux_losses.append(cross_task_aux_loss.item())
            train_preds.extend(outputs.detach().float().cpu().numpy())
            train_targets.extend(batch.y.detach().float().cpu().numpy())
        else:
            train_losses.append(loss.detach())
            fused_losses.append(fused_loss.detach())
            auxiliary_losses.append(auxiliary_loss.detach())
            fusion_prior_losses.append(fusion_prior_loss.detach())
            cross_task_aux_losses.append(cross_task_aux_loss.detach())
            train_preds.append(outputs.detach().float())
            train_targets.append(batch.y.detach().float())

    if legacy_sync:
        avg_train_loss = float(np.mean(train_losses))
        avg_fused_loss = float(np.mean(fused_losses))
        avg_auxiliary_loss = float(np.mean(auxiliary_losses))
        avg_fusion_prior_loss = float(np.mean(fusion_prior_losses))
        avg_cross_task_aux_loss = float(np.mean(cross_task_aux_losses))
        train_predictions = np.asarray(train_preds)
        train_target_values = np.asarray(train_targets)
    else:
        metric_tensors = torch.stack([
            torch.stack(train_losses).mean(),
            torch.stack(fused_losses).mean(),
            torch.stack(auxiliary_losses).mean(),
            torch.stack(fusion_prior_losses).mean(),
            torch.stack(cross_task_aux_losses).mean(),
        ]).float().cpu().numpy()
        train_predictions = torch.cat(train_preds).cpu().numpy()
        train_target_values = torch.cat(train_targets).cpu().numpy()
        avg_train_loss, avg_fused_loss, avg_auxiliary_loss, avg_fusion_prior_loss, avg_cross_task_aux_loss = (
            float(value) for value in metric_tensors
        )
    train_r2 = r2_score(train_target_values, train_predictions)

    result = (
        avg_train_loss, train_r2, avg_fused_loss,
        avg_auxiliary_loss, avg_fusion_prior_loss, avg_cross_task_aux_loss,
    )
    if not return_timing:
        return result
    elapsed = time.perf_counter() - epoch_started
    timing = {
        'training_steps': int(optimizer_steps),
        'training_seconds': float(elapsed),
        'optimizer_steps_per_second': (
            float(optimizer_steps / elapsed) if elapsed > 0 else 0.0
        ),
    }
    return result + (timing,)

def evaluate(model, data_loader, criterion, device, scaler=None, amp_dtype='fp32'):
    model.eval()
    legacy_sync = os.environ.get('MTS_BENCHMARK_LEGACY_SYNC', '0') == '1'
    losses = []
    preds = []
    targets = []

    with torch.inference_mode():
        for batch in tqdm(data_loader, desc="Evaluating"):
            batch = batch.to(device, non_blocking=True)
            with _autocast_context(device, amp_dtype):
                outputs,_ = model(batch)
                loss = criterion(outputs, batch.y)
            if legacy_sync:
                losses.append(loss.item())
                preds.extend(outputs.detach().float().cpu().numpy())
                targets.extend(batch.y.detach().float().cpu().numpy())
            else:
                losses.append(loss.detach().float())
                preds.append(outputs.detach().float())
                targets.append(batch.y.detach().float())

    # A single host transfer at loader end avoids three CUDA synchronizations
    # per validation batch while preserving the historical unweighted mean of
    # per-batch losses.
    if legacy_sync:
        avg_loss = float(np.mean(losses))
        targets = np.asarray(targets)
        preds = np.asarray(preds)
    else:
        avg_loss = float(torch.stack(losses).mean().cpu())
        targets = torch.cat(targets).cpu().numpy()
        preds = torch.cat(preds).cpu().numpy()
    if scaler is not None:
        r2 = r2_score(
            scaler.inverse_transform(np.asarray(targets)),
            scaler.inverse_transform(np.asarray(preds)),
        )
    else:
        r2 = r2_score(targets, preds)
    
    return avg_loss, r2, targets, preds

def test_model(
    model, test_loader, scaler, device, return_predictions=False,
    amp_dtype='fp32',
):
    model.eval()
    legacy_sync = os.environ.get('MTS_BENCHMARK_LEGACY_SYNC', '0') == '1'
    test_preds = []
    test_targets = []

    with torch.inference_mode():
        for batch in tqdm(test_loader, desc="Testing"):
            batch = batch.to(device, non_blocking=True)
            with _autocast_context(device, amp_dtype):
                outputs,_ = model(batch)
            if legacy_sync:
                test_preds.extend(outputs.detach().float().cpu().numpy())
                test_targets.extend(batch.y.detach().float().cpu().numpy())
            else:
                test_preds.append(outputs.detach().float())
                test_targets.append(batch.y.detach().float())

    # Keep prediction collection asynchronous until the complete loader has
    # finished, then synchronize once for metric computation and artifacts.
    if legacy_sync:
        y_true = np.asarray(test_targets)
        y_pred = np.asarray(test_preds)
    else:
        y_true = torch.cat(test_targets).cpu().numpy()
        y_pred = torch.cat(test_preds).cpu().numpy()
    y_true_unscaled = scaler.inverse_transform(y_true)
    y_pred_unscaled = scaler.inverse_transform(y_pred)
    test_r2 = r2_score(y_true_unscaled, y_pred_unscaled)
    test_mae = metrics.mean_absolute_error(y_true_unscaled, y_pred_unscaled)
    test_rmse = np.sqrt(metrics.mean_squared_error(y_true_unscaled, y_pred_unscaled))

    result = {
        'test_r2': float(test_r2),
        'test_mae': float(test_mae),
        'test_rmse': float(test_rmse),
    }
    if return_predictions:
        # Keep these private to the in-process result dictionary.  train.py
        # removes them before JSON/CSV serialization and writes the full-
        # precision arrays to an atomic NPZ prediction shard instead.
        result['_y_true'] = y_true_unscaled.reshape(-1).astype(np.float64)
        result['_y_pred'] = y_pred_unscaled.reshape(-1).astype(np.float64)
    return result


def _base_model(model):
    return model.module if isinstance(model, nn.DataParallel) else model


def _mts_glt_encoder(model):
    """Return the optional MTS-GLT graph encoder without changing forward."""
    base = _base_model(model)
    encoders = getattr(base, 'encoders', {})
    graph = encoders['graph'] if 'graph' in encoders else None
    encoder = getattr(graph, 'encoder', None)
    if getattr(encoder, 'downstream_mode', None) != 'o8_glt':
        return None
    if not callable(getattr(encoder, 'encode_views', None)):
        return None
    return encoder


def _mts_glt_fusion_components(model):
    """Return the concrete modules used by either supported GLT fusion path."""
    base = _base_model(model)
    encoders = getattr(base, 'encoders', {})
    graph = encoders['graph'] if 'graph' in encoders else None
    encoder = getattr(graph, 'encoder', None)
    mode = getattr(encoder, 'downstream_mode', None)
    if mode == 'o8_glt' and callable(getattr(encoder, 'encode_views', None)):
        return {
            'kind': 'scalar', 'base': base, 'graph': graph, 'encoder': encoder,
            'o8': encoder.o8, 'glt': encoder.glt,
            'fusion_norm': encoder.glt_fusion_norm,
            'fusion_projection': encoder.glt_fusion_projection,
            'gate': encoder.glt_gate, 'md_residual': encoder.o8.md_residual,
        }
    if (
        mode == 'o8_glt_graph'
        and getattr(encoder, 'architecture_name', '') == 'MTS-GLT-GraphGate-v1'
        and callable(getattr(encoder, 'encode_views', None))
    ):
        return {
            'kind': 'channel', 'base': base, 'graph': graph,
            'encoder': encoder, 'o8': encoder.o8_encoder,
            'glt': encoder.glt_line_encoder,
            'fusion_norm': encoder.fusion_norm,
            'fusion_projection': encoder.fusion_projection,
            'gate': encoder.channel_gate,
            'md_residual': encoder.o8_encoder.md_residual,
        }
    return None


def _mts_glt_gate_summary(components):
    gate = components['gate'].detach().float().cpu()
    alpha = torch.tanh(gate)
    absolute = alpha.abs()
    summary = {
        'mean_abs': float(absolute.mean()),
        'median_abs': float(absolute.median()),
        'p10': float(torch.quantile(alpha.flatten(), 0.10)),
        'p90': float(torch.quantile(alpha.flatten(), 0.90)),
        'fraction_abs_lt_0_01': float((absolute < 0.01).float().mean()),
        'fraction_positive': float((alpha > 0).float().mean()),
        'fraction_negative': float((alpha < 0).float().mean()),
    }
    if gate.numel() == 1:
        summary.update({
            'gate': float(gate), 'tanh_gate': float(alpha),
        })
    return summary


def _mts_glt_trajectory_fields(audit):
    alpha = audit['alpha']
    fields = {
        'rho_mean': audit['rho']['mean'],
        'rho_median': audit['rho']['median'],
        'rho_p10': audit['rho']['p10'],
        'rho_p90': audit['rho']['p90'],
        'alpha_mean_abs': alpha['mean_abs'],
        'alpha_median_abs': alpha['median_abs'],
        'alpha_p10': alpha['p10'],
        'alpha_p90': alpha['p90'],
        'alpha_fraction_abs_lt_0_01': alpha['fraction_abs_lt_0_01'],
        'alpha_fraction_positive': alpha['fraction_positive'],
        'alpha_fraction_negative': alpha['fraction_negative'],
    }
    if 'gate' in alpha:
        fields.update({'gate': alpha['gate'], 'tanh_gate': alpha['tanh_gate']})
    return fields


def initialize_mts_glt_fusion_warm(model, initial_alpha=0.1):
    """Set scalar or channel-wise fusion strength after strict checkpoint load."""
    components = _mts_glt_fusion_components(model)
    if components is None:
        raise ValueError('FusionWarm requires o8_glt or o8_glt_graph mode')
    alpha = float(initial_alpha)
    if not 0.0 < alpha < 1.0:
        raise ValueError('FusionWarm initial alpha must be between 0 and 1')
    with torch.no_grad():
        components['gate'].fill_(float(np.arctanh(alpha)))
    return float(torch.tanh(components['gate'].detach()).mean().cpu())


_MTS_GLT_STAGE2_TRAINABILITY = {
    'both_frozen', 'o8_only', 'glt_query_only', 'joint',
}


def _configure_mts_glt_fusion_stage(
    model, stage, stage2_trainability='joint',
):
    """Apply the exact trainability split for one FusionWarm stage."""
    stage = str(stage)
    if stage not in {'warm', 'joint'}:
        raise ValueError('FusionWarm stage must be warm or joint')
    components = _mts_glt_fusion_components(model)
    if components is None:
        raise ValueError('FusionWarm requires o8_glt or o8_glt_graph mode')
    encoder = components['encoder']
    base = components['base']
    graph_module = components['graph']
    stage2_trainability = str(stage2_trainability)
    if stage2_trainability not in _MTS_GLT_STAGE2_TRAINABILITY:
        raise ValueError(
            'unsupported FusionWarm Stage 2 trainability: '
            f'{stage2_trainability}'
        )
    if stage == 'joint':
        _configure_mts_trainability(model)
        if stage2_trainability in {'both_frozen', 'glt_query_only'}:
            for name, parameter in components['o8'].named_parameters():
                if not name.startswith('md_residual.'):
                    parameter.requires_grad = False
        if stage2_trainability in {'both_frozen', 'o8_only'}:
            for parameter in components['glt'].parameters():
                parameter.requires_grad = False
        return encoder

    for parameter in base.parameters():
        parameter.requires_grad = False
    for module in (
        components['fusion_norm'],
        components['fusion_projection'],
        components['md_residual'],
        graph_module.norm,
        graph_module.projection,
        base.mlp,
    ):
        _set_module_trainable(module, True)
    components['gate'].requires_grad = True
    return encoder


def _set_mts_glt_frozen_encoders_eval(model, policy='both_frozen'):
    """Keep frozen pretrained branches deterministic after top-level train()."""
    components = _mts_glt_fusion_components(model)
    if components is None:
        raise ValueError('FusionWarm requires o8_glt or o8_glt_graph mode')
    policy = str(policy)
    if policy not in _MTS_GLT_STAGE2_TRAINABILITY:
        raise ValueError(f'unsupported frozen-encoder policy: {policy}')
    if policy in {'both_frozen', 'glt_query_only'}:
        components['o8'].eval()
    if policy in {'both_frozen', 'o8_only'}:
        components['glt'].eval()
    # The MD200 residual belongs to the downstream adapter, not the frozen O8
    # body, and must remain in training mode during Stage 1.
    components['md_residual'].train()


def _finite_distribution(values):
    values = torch.cat(values) if values else torch.empty(0)
    values = values[torch.isfinite(values)]
    if not values.numel():
        return {
            'mean': None, 'median': None, 'p10': None, 'p90': None,
            'count': 0,
        }
    return {
        'mean': float(values.mean()),
        'median': float(values.median()),
        'p10': float(torch.quantile(values, 0.10)),
        'p90': float(torch.quantile(values, 0.90)),
        'count': int(values.numel()),
    }


def collect_mts_glt_fusion_audit(model, data_loader, device):
    """Measure the best-state GLT residual on one deterministic loader."""
    components = _mts_glt_fusion_components(model)
    if components is None:
        raise ValueError('fusion audit requires o8_glt or o8_glt_graph mode')
    encoder = components['encoder']
    model.eval()
    o8_norms, projected_norms, delta_norms, rhos, cosines = [], [], [], [], []
    valid_count = 0
    with torch.inference_mode():
        for batch in data_loader:
            batch = batch.to(device, non_blocking=True)
            views = encoder.encode_views(batch)
            valid = views['valid_3d']
            if not bool(valid.any()):
                continue
            z_o8 = views['z_o8'][valid].float()
            projected = views['projected_z_glt'][valid].float()
            delta = views['delta_z_3d'][valid].float()
            finite = (
                torch.isfinite(z_o8).all(dim=-1)
                & torch.isfinite(projected).all(dim=-1)
                & torch.isfinite(delta).all(dim=-1)
            )
            if not bool(finite.any()):
                continue
            z_o8, projected, delta = z_o8[finite], projected[finite], delta[finite]
            o8_norm = torch.linalg.vector_norm(z_o8, dim=-1)
            projected_norm = torch.linalg.vector_norm(projected, dim=-1)
            delta_norm = torch.linalg.vector_norm(delta, dim=-1)
            denom = o8_norm * delta_norm
            cosine_valid = denom > 1e-12
            if bool(cosine_valid.any()):
                cosines.append(
                    ((z_o8[cosine_valid] * delta[cosine_valid]).sum(dim=-1)
                     / denom[cosine_valid]).detach().cpu()
                )
            o8_norms.append(o8_norm.detach().cpu())
            projected_norms.append(projected_norm.detach().cpu())
            delta_norms.append(delta_norm.detach().cpu())
            rhos.append((delta_norm / (o8_norm + 1e-8)).detach().cpu())
            valid_count += int(finite.sum())
    if not valid_count:
        raise RuntimeError('fusion audit found no finite GLT-valid samples')
    cosine = _finite_distribution(cosines)
    return {
        'valid_count': valid_count,
        'o8_norm': _finite_distribution(o8_norms),
        'projected_glt_norm': _finite_distribution(projected_norms),
        'delta_3d_norm': _finite_distribution(delta_norms),
        'rho': _finite_distribution(rhos),
        'cosine_o8_delta': cosine,
        'alpha': _mts_glt_gate_summary(components),
    }


def collect_mts_graphgate_audit(model, data_loader, device):
    """Collect required channel-gate and residual statistics at the best state."""
    base = _base_model(model)
    graph = base.encoders['graph'] if 'graph' in getattr(base, 'encoders', {}) else None
    encoder = getattr(graph, 'encoder', None)
    if getattr(encoder, 'architecture_name', '') != 'MTS-GLT-GraphGate-v1':
        return None
    if getattr(encoder, 'downstream_mode', None) != 'o8_glt_graph':
        return None
    model.eval()
    rhos = []
    valid_count = 0
    with torch.inference_mode():
        for batch in data_loader:
            batch = batch.to(device, non_blocking=True)
            views = encoder.encode_views(batch)
            valid = views['valid_3d'].bool()
            if not bool(valid.any()):
                continue
            o8 = views['z_o8'][valid].float()
            delta = views['delta_z_3d'][valid].float()
            finite = torch.isfinite(o8).all(-1) & torch.isfinite(delta).all(-1)
            if bool(finite.any()):
                rhos.append((
                    torch.linalg.vector_norm(delta[finite], dim=-1)
                    / torch.linalg.vector_norm(o8[finite], dim=-1).clamp_min(1e-8)
                ).cpu())
                valid_count += int(finite.sum())
    alpha = torch.tanh(encoder.channel_gate.detach().float()).cpu()
    alpha_abs = alpha.abs()
    return {
        'valid_count': valid_count,
        'rho': _finite_distribution(rhos),
        'alpha': {
            'mean_abs': float(alpha_abs.mean()),
            'median_abs': float(alpha_abs.median()),
            'p10': float(torch.quantile(alpha, 0.10)),
            'p90': float(torch.quantile(alpha, 0.90)),
            'fraction_abs_lt_0_01': float((alpha_abs < 0.01).float().mean()),
            'fraction_positive': float((alpha > 0).float().mean()),
            'fraction_negative': float((alpha < 0).float().mean()),
        },
    }


def collect_mts_glt_initial_fusion_audit(model, data_loader, device):
    """Collect epoch-0 fusion statistics without advancing experiment RNG."""
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    cpu_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    generator = getattr(data_loader, 'generator', None)
    generator_state = generator.get_state() if generator is not None else None
    training_modes = [(module, bool(module.training)) for module in model.modules()]
    try:
        return collect_mts_glt_fusion_audit(model, data_loader, device)
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(cpu_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)
        if generator is not None and generator_state is not None:
            generator.set_state(generator_state)
        for module, training in training_modes:
            module.training = training


class _CpuStateAverager:
    """Incrementally average epoch checkpoints without duplicating the GPU model."""

    def __init__(self):
        self.state = None
        self.count = 0

    def update(self, model):
        current = model.state_dict()
        self.count += 1
        if self.state is None:
            self.state = {
                key: value.detach().cpu().clone()
                for key, value in current.items()
            }
            return

        weight = 1.0 / float(self.count)
        for key, value in current.items():
            value_cpu = value.detach().cpu()
            if value_cpu.is_floating_point() or value_cpu.is_complex():
                self.state[key].add_(value_cpu - self.state[key], alpha=weight)
            else:
                # Integer counters and other non-floating buffers cannot be averaged.
                self.state[key] = value_cpu.clone()

    def state_dict(self):
        if self.state is None:
            raise RuntimeError("Cannot read an empty weight average")
        return self.state


def _set_smiles_trainable(model, trainable, last_layers=4):
    base = _base_model(model)
    if 'smiles' not in base.encoders:
        return
    encoder = base.encoders['smiles'].encoder
    for parameter in encoder.parameters():
        parameter.requires_grad = False
    if not trainable:
        return
    layers = getattr(getattr(encoder, 'encoder', None), 'layer', [])
    for layer in list(layers)[-int(last_layers):]:
        for parameter in layer.parameters():
            parameter.requires_grad = True


def _set_module_trainable(module, trainable):
    if module is not None:
        for parameter in module.parameters():
            parameter.requires_grad = bool(trainable)


def _configure_mts_trainability(model):
    """Apply the explicit MTS trainability contract.

    The MTS downstream path trains the complete graph wrapper from epoch zero,
    but the optional V-stage modalities were deliberately low-capacity
    adapters.  Keeping this policy explicit is important: a blanket
    ``base.parameters()`` toggle would re-enable every RoBERTa parameter and
    silently turn the V experiments into full language-model fine-tuning.
    """
    base = _base_model(model)
    if not _is_mts_model(base):
        raise ValueError('MTS trainability requested for a non-MTS model')

    # Start from a frozen model and opt modules in explicitly.  This also
    # handles LoRALinear.base parameters, which are intentionally frozen by
    # inject_roberta_lora but must not be re-enabled here.
    for parameter in base.parameters():
        parameter.requires_grad = False

    graph_module = base.encoders['graph'] if 'graph' in base.encoders else None
    if graph_module is None:
        raise ValueError('MTS model is missing its graph encoder module')
    for parameter in graph_module.parameters():
        parameter.requires_grad = True

    # Disabled optional geometry modules remain frozen while staying in the
    # state dict for strict checkpoint loading.
    mts_encoder = getattr(graph_module, "encoder", None)
    if getattr(mts_encoder, "architecture_name", "") == "MIPS-Trimer-SCAGE":
        if not bool(getattr(mts_encoder, "use_star_rbf", True)):
            _set_module_trainable(
                getattr(mts_encoder, "star_distance_bias", None), False
            )
    if getattr(mts_encoder, "downstream_mode", None) == "o8_only":
        _set_module_trainable(getattr(mts_encoder, "glt", None), False)
        _set_module_trainable(getattr(mts_encoder, "glt_fusion_norm", None), False)
        _set_module_trainable(getattr(mts_encoder, "glt_fusion_projection", None), False)
        _set_module_trainable(getattr(mts_encoder, "atom_fusion_norm", None), False)
        _set_module_trainable(getattr(mts_encoder, "atom_fusion_projection", None), False)
        _set_module_trainable(getattr(mts_encoder, "compact19_residual", None), False)
        gate = getattr(mts_encoder, "glt_gate", None)
        if gate is not None:
            gate.requires_grad = False
        channel_gate = getattr(mts_encoder, "atom_channel_gate", None)
        if channel_gate is not None:
            channel_gate.requires_grad = False
        _set_module_trainable(getattr(mts_encoder, "glt_line_encoder", None), False)
        _set_module_trainable(getattr(mts_encoder, "fusion_norm", None), False)
        _set_module_trainable(getattr(mts_encoder, "fusion_projection", None), False)
        graph_gate = getattr(mts_encoder, "channel_gate", None)
        if graph_gate is not None:
            graph_gate.requires_grad = False
    if getattr(mts_encoder, "downstream_mode", None) in {
        "o8_only", "o8_glt", "o8_glt_atom", "o8_glt_atom_desc", "o8_glt_graph"
    }:
        _set_module_trainable(
            getattr(getattr(mts_encoder, "o8", None), "star_distance_bias", None),
            False,
        )
        _set_module_trainable(
            getattr(getattr(mts_encoder, "o8_encoder", None), "star_distance_bias", None),
            False,
        )

    for module in (
        getattr(base, 'mlp', None),
        getattr(base, 'modality_heads', None),
        getattr(base, 'cross_task_aux_heads', None),
        getattr(base, 'residual_modality_gates', None),
    ):
        _set_module_trainable(module, True)

    smiles_module = base.encoders['smiles'] if 'smiles' in base.encoders else None
    if smiles_module is not None:
        # The RoBERTa base remains frozen.  Only the rank-8 LoRA matrices in
        # the final four layers receive gradients; the external projection
        # and normalization are the low-capacity trainable adapter.
        for parameter_name, parameter in smiles_module.encoder.named_parameters():
            parameter.requires_grad = (
                parameter_name.endswith('lora_a')
                or parameter_name.endswith('lora_b')
            )
        _set_module_trainable(getattr(smiles_module, 'norm', None), True)
        _set_module_trainable(getattr(smiles_module, 'projection', None), True)

    fp_module = base.encoders['fp'] if 'fp' in base.encoders else None
    if fp_module is not None:
        # CountFP has no large pretrained backbone; its compact encoder and
        # projection are the adapter and are trainable at the FP learning
        # rate.
        _set_module_trainable(fp_module, True)


def _is_mts_model(model):
    base = _base_model(model)
    # MTS experiments may append SMILES and/or CountFP to the graph anchor.
    # Architecture detection must therefore inspect the graph encoder rather
    # than requiring the production graph-only modality tuple.
    if 'graph' not in tuple(getattr(base, 'modality_list', ())):
        return False
    encoders = getattr(base, 'encoders', {})
    graph_module = encoders['graph'] if 'graph' in encoders else None
    return graph_module is not None and getattr(
        graph_module.encoder, 'architecture_name', ''
    ) in {
        'MIPS-Trimer-SCAGE',
        'MIPS-Trimer-GLT-v2',
        'MTS-GLT-GraphGate-v1',
    }


def _build_downstream_optimizer(
    model, smiles_lr, graph_lr, geom_lr, fp_lr, fusion_lr, head_lr, weight_decay,
    mts_o8_lr=5e-6, mts_geometry_lr=1e-5, mts_adapter_lr=5e-5,
    mts_glt_fusion_stage=None,
):
    base = _base_model(model)
    groups = []
    used = set()

    def add_group(module, lr, name):
        if module is None:
            return
        decay, no_decay = [], []
        for parameter_name, param in module.named_parameters():
            if not param.requires_grad or id(param) in used:
                continue
            used.add(id(param))
            normalized_name = parameter_name.lower()
            if (
                param.ndim <= 1
                or normalized_name.endswith("bias")
                or "norm" in normalized_name
                or "gate" in normalized_name
            ):
                no_decay.append(param)
            else:
                decay.append(param)
        if decay:
            groups.append({
                'params': decay,
                'lr': float(lr),
                'weight_decay': float(weight_decay),
                'name': f'{name}/decay',
            })
        if no_decay:
            groups.append({
                'params': no_decay,
                'lr': float(lr),
                'weight_decay': 0.0,
                'name': f'{name}/no_decay',
            })

    encoders = base.encoders
    if _is_mts_model(base):
        # The graph optimizer has one group for the complete graph wrapper, so
        # norm/projection are intentionally at graph_lr as well. Optional
        # modality adapters are registered separately without unfreezing a
        # pretrained SMILES base.
        def add_mts_group(module, lr, name):
            if module is None:
                return
            params = [
                parameter for parameter in module.parameters()
                if parameter.requires_grad and id(parameter) not in used
            ]
            if not params:
                return
            used.update(id(parameter) for parameter in params)
            groups.append({
                'params': params,
                'lr': float(lr),
                'weight_decay': float(weight_decay),
                'name': name,
            })

        def add_mts_group_from_params(params, lr, name):
            params = [
                parameter for parameter in params
                if parameter.requires_grad and id(parameter) not in used
            ]
            if not params:
                return
            used.update(id(parameter) for parameter in params)
            groups.append({
                'params': params,
                'lr': float(lr),
                'weight_decay': float(weight_decay),
                'name': name,
            })

        graph_module = encoders['graph'] if 'graph' in encoders else None
        if mts_glt_fusion_stage is not None:
            stage = str(mts_glt_fusion_stage)
            if stage not in {'warm', 'joint'}:
                raise ValueError('mts_glt_fusion_stage must be warm or joint')
            components = _mts_glt_fusion_components(model)
            if components is None or graph_module is None:
                raise ValueError('FusionWarm optimizer requires O8+GLT graph mode')
            encoder = components['encoder']

            def add_named_parameters(named_parameters, lr, name):
                decay, no_decay = [], []
                for parameter_name, parameter in named_parameters:
                    if not parameter.requires_grad or id(parameter) in used:
                        continue
                    used.add(id(parameter))
                    normalized = str(parameter_name).lower()
                    target = (
                        no_decay if (
                            parameter.ndim <= 1
                            or normalized.endswith('bias')
                            or 'norm' in normalized
                            or 'gate' in normalized
                        ) else decay
                    )
                    target.append(parameter)
                if decay:
                    groups.append({
                        'params': decay, 'lr': float(lr),
                        'weight_decay': float(weight_decay),
                        'name': f'{name}/decay',
                    })
                if no_decay:
                    groups.append({
                        'params': no_decay, 'lr': float(lr),
                        'weight_decay': 0.0,
                        'name': f'{name}/no_decay',
                    })

            if stage == 'joint':
                o8_lr = (
                    float(mts_o8_lr)
                    if components['kind'] == 'channel' else float(graph_lr)
                )
                glt_lr = (
                    float(mts_geometry_lr)
                    if components['kind'] == 'channel' else float(graph_lr)
                )
                add_named_parameters(
                    (
                        (name, parameter)
                        for name, parameter in components['o8'].named_parameters()
                        if not name.startswith('md_residual.')
                    ),
                    o8_lr, 'o8_encoder',
                )
                add_named_parameters(
                    components['glt'].named_parameters(), glt_lr, 'glt_encoder'
                )
            fusion_prefix = (
                'graphgate_fusion'
                if components['kind'] == 'channel' else 'glt_fusion'
            )
            add_group(
                components['fusion_norm'], fusion_lr, f'{fusion_prefix}_norm'
            )
            add_group(
                components['fusion_projection'], fusion_lr,
                f'{fusion_prefix}_projection',
            )
            gate_group = (
                'graphgate_channel_gate'
                if components['kind'] == 'channel' else 'glt_gate'
            )
            add_named_parameters(
                [('gate', components['gate'])], fusion_lr,
                gate_group,
            )
            add_group(components['md_residual'], fusion_lr, 'md200_residual')
            add_group(
                nn.ModuleList([graph_module.norm, graph_module.projection]),
                head_lr, 'graph_output_adapter',
            )
            add_group(base.mlp, head_lr, 'regression_head')
            remaining = [
                name for name, parameter in base.named_parameters()
                if parameter.requires_grad and id(parameter) not in used
            ]
            if remaining:
                raise RuntimeError(
                    'FusionWarm optimizer omitted trainable parameters: '
                    + ', '.join(remaining[:10])
                )
            return torch.optim.AdamW(groups)

        if graph_module is not None:
            add_mts_group(graph_module, graph_lr, 'graph')

        smiles_module = encoders['smiles'] if 'smiles' in encoders else None
        if smiles_module is not None:
            lora_parameters = [
                parameter for name, parameter in
                smiles_module.encoder.named_parameters()
                if parameter.requires_grad and (
                    name.endswith('lora_a') or name.endswith('lora_b')
                )
            ]
            add_mts_group_from_params(lora_parameters, smiles_lr, 'smiles_lora')
            add_mts_group(
                nn.ModuleList([smiles_module.norm, smiles_module.projection]),
                fusion_lr, 'smiles_adapter'
            )
        add_mts_group(
            encoders['fp'] if 'fp' in encoders else None,
            fp_lr, 'fp_adapter'
        )
        add_mts_group(getattr(base, 'residual_modality_gates', None), fusion_lr, 'modality_gates')
        add_mts_group(base.mlp, head_lr, 'regression_head')
        add_mts_group(getattr(base, 'modality_heads', None), head_lr, 'modality_heads')
        add_mts_group(getattr(base, 'cross_task_aux_heads', None), head_lr, 'cross_task_aux_heads')
        remaining = [
            parameter for parameter in base.parameters()
            if parameter.requires_grad and id(parameter) not in used
        ]
        if remaining:
            groups.append({
                'params': remaining,
                'lr': float(fusion_lr),
                'weight_decay': float(weight_decay),
                'name': 'remaining',
            })
        return torch.optim.AdamW(groups)
    raise ValueError('MTS downstream optimizer requires an MTS graph model')


def fit_fixed_epochs(
    model,
    train_loader,
    device,
    num_epochs,
    max_grad_norm=1.0,
    smiles_lr=1e-5,
    graph_lr=5e-5,
    geom_lr=5e-5,
    fp_lr=1e-4,
    fusion_lr=3e-4,
    head_lr=3e-4,
    weight_decay=0.01,
    warmup_epochs=5,
    freeze_smiles_epochs=5,
    deep_unfreeze_epoch=10,
    fp_unfreeze_epoch=-1,
    regression_loss='mse',
    huber_beta=0.5,
    unimodal_aux_weight=0.0,
    fusion_prior_kl_weight=0.0,
    fusion_prior=(0.30, 0.40, 0.30),
    cross_task_aux_weight=0.0,
    mts_o8_lr=5e-6,
    mts_geometry_lr=1e-5,
    mts_adapter_lr=5e-5,
    amp_dtype='fp32',
):
    """Fit a fresh downstream model for a validation-selected epoch count."""
    num_epochs = int(num_epochs)
    if num_epochs <= 0:
        raise ValueError('Fixed-epoch refit requires a positive epoch count')
    criterion = (
        nn.SmoothL1Loss(beta=float(huber_beta))
        if regression_loss == 'huber' else nn.MSELoss()
    )
    base = _base_model(model)
    if not _is_mts_model(base):
        raise ValueError('MTS fixed-epoch refit requires an MTS graph model')
    _configure_mts_trainability(model)
    optimizer = _build_downstream_optimizer(
        model, smiles_lr, graph_lr, geom_lr, fp_lr,
        fusion_lr, head_lr, weight_decay,
        mts_o8_lr=mts_o8_lr,
        mts_geometry_lr=mts_geometry_lr,
        mts_adapter_lr=mts_adapter_lr,
    )
    total_steps = max(1, len(train_loader) * num_epochs)
    warmup_steps = min(
        total_steps - 1, max(0, int(warmup_epochs) * len(train_loader))
    )

    def lr_lambda(step):
        if warmup_steps and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (
            1.0 + np.cos(np.pi * min(1.0, max(0.0, progress)))
        )

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    final_train_r2 = float('nan')
    final_train_loss = float('nan')
    for epoch in range(num_epochs):
        (
            final_train_loss, final_train_r2, _, _, _, _,
        ) = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scheduler,
            device,
            epoch=epoch + 1,
            max_grad_norm=max_grad_norm,
            unimodal_aux_weight=unimodal_aux_weight,
            fusion_prior_kl_weight=fusion_prior_kl_weight,
            fusion_prior=fusion_prior,
            cross_task_aux_weight=cross_task_aux_weight,
            amp_dtype=amp_dtype,
        )
        print(
            f'Refit epoch {epoch + 1}/{num_epochs}: '
            f'loss={final_train_loss:.4f}, R2={final_train_r2:.4f}'
        )
    return {
        'refit_train_loss': float(final_train_loss),
        'refit_train_r2': float(final_train_r2),
        'refit_epochs': num_epochs,
    }
def train_and_evaluate(
    model,
    scaler,
    train_loader,
    val_loader,
    test_loader,
    device,
    num_epochs=100,
    patience=5,
    max_grad_norm=1.0,
    smiles_lr=1e-5,
    graph_lr=5e-5,
    geom_lr=5e-5,
    fp_lr=1e-4,
    fusion_lr=3e-4,
    head_lr=3e-4,
    weight_decay=0.01,
    warmup_epochs=5,
    freeze_smiles_epochs=5,
    deep_unfreeze_epoch=10,
    fp_unfreeze_epoch=-1,
    regression_loss='mse',
    huber_beta=0.5,
    unimodal_aux_weight=0.0,
    fusion_prior_kl_weight=0.0,
    fusion_prior=(0.30, 0.40, 0.30),
    cross_task_aux_weight=0.0,
    swa_start_epoch=-1,
    evaluate_test=True,
    return_predictions=False,
    mts_o8_lr=5e-6,
    mts_geometry_lr=1e-5,
    mts_adapter_lr=5e-5,
    pcgrad=False,
    amp_dtype='fp32',
    mts_glt_postmortem=False,
    mts_glt_fusion_strategy='legacy_zero',
    mts_glt_fusion_warm_epochs=5,
    mts_glt_initial_alpha=0.1,
    mts_glt_fusion_stage2_trainability='joint',
):
    # Define loss function and optimizer
    criterion = nn.SmoothL1Loss(beta=float(huber_beta)) if regression_loss == 'huber' else nn.MSELoss()
    base = _base_model(model)
    if not _is_mts_model(base):
        raise ValueError('MTS training requires an MTS graph model')
    fusion_strategy = str(mts_glt_fusion_strategy)
    if fusion_strategy not in {'legacy_zero', 'fusion_warm'}:
        raise ValueError('unsupported MTS-GLT fusion strategy')
    fusion_warm = fusion_strategy == 'fusion_warm'
    fusion_warm_epochs = int(mts_glt_fusion_warm_epochs)
    stage2_trainability = str(mts_glt_fusion_stage2_trainability)
    if stage2_trainability not in _MTS_GLT_STAGE2_TRAINABILITY:
        raise ValueError(
            'unsupported FusionWarm Stage 2 trainability: '
            f'{stage2_trainability}'
        )
    if fusion_warm:
        if not 0 <= fusion_warm_epochs < int(num_epochs):
            raise ValueError(
                'FusionWarm epochs must be non-negative and smaller than num_epochs'
            )
        if int(swa_start_epoch) >= 0:
            raise ValueError('FusionWarm does not use SWA')
        _configure_mts_glt_fusion_stage(model, 'warm')
        warm_components = _mts_glt_fusion_components(model)
        expected_alpha = float(mts_glt_initial_alpha)
        observed_alpha = torch.tanh(warm_components['gate'].detach()).float()
        if not bool(torch.allclose(
            observed_alpha,
            torch.full_like(observed_alpha, expected_alpha),
            atol=1e-7, rtol=0.0,
        )):
            raise ValueError(
                'FusionWarm gate must be initialized after checkpoint load: '
                f'expected alpha={expected_alpha}, '
                f'observed mean={float(observed_alpha.mean())}'
            )
    else:
        _configure_mts_trainability(model)
    if str(amp_dtype) not in {'fp32', 'bf16'}:
        raise ValueError("amp_dtype must be fp32 or bf16")
    if str(amp_dtype) == 'bf16':
        parity_batch = next(iter(train_loader))
        passed, parity = finetune_bf16_parity_gate(
            model, parity_batch, criterion, device
        )
        print(f"Fine-tune BF16 parity gate: {parity}, pass={passed}")
        if not passed:
            raise RuntimeError("Fine-tune BF16 parity gate failed")
    optimizer = _build_downstream_optimizer(
        model, smiles_lr, graph_lr, geom_lr, fp_lr, fusion_lr, head_lr,
        weight_decay, mts_o8_lr=mts_o8_lr,
        mts_geometry_lr=mts_geometry_lr,
        mts_adapter_lr=mts_adapter_lr,
        mts_glt_fusion_stage='warm' if fusion_warm else None,
    )
    postmortem_components = (
        _mts_glt_fusion_components(model) if mts_glt_postmortem else None
    )
    postmortem_encoder = (
        postmortem_components['encoder']
        if postmortem_components is not None else None
    )
    if mts_glt_postmortem and postmortem_components is None:
        raise ValueError(
            'MTS-GLT postmortem requires o8_glt or o8_glt_graph mode'
        )
    initial_fusion_weight = (
        postmortem_components['fusion_projection'].weight.detach().cpu().clone()
        if postmortem_components is not None else None
    )
    gate_trajectory = []
    initial_fusion_audit = None
    if fusion_warm:
        if postmortem_encoder is None:
            raise ValueError('FusionWarm requires fusion audit output')
        initial_fusion_audit = collect_mts_glt_initial_fusion_audit(
            model, val_loader, device
        )
        initial_row = {
            'epoch': 0,
            'stage': 'initial',
            'validation_r2': None,
            'is_best': False,
        }
        initial_row.update(_mts_glt_trajectory_fields(initial_fusion_audit))
        gate_trajectory.append(initial_row)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda _step: 1.0
        )
    else:
        total_steps = max(1, len(train_loader) * num_epochs)
        warmup_steps = min(
            total_steps - 1, max(0, int(warmup_epochs) * len(train_loader))
        )

        def lr_lambda(step):
            if warmup_steps and step < warmup_steps:
                return float(step + 1) / float(warmup_steps)
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (
                1.0 + np.cos(np.pi * min(1.0, max(0.0, progress)))
            )

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_val_r2 = -float('inf')
    best_val_rmse = float('inf')
    best_epoch = -1
    best_model_state = None
    epochs_no_improve = 0
    swa_start_epoch = int(swa_start_epoch)
    state_averager = _CpuStateAverager() if swa_start_epoch >= 0 else None
    training_timing = []

    for epoch in range(num_epochs):
        if fusion_warm and epoch == fusion_warm_epochs:
            _configure_mts_glt_fusion_stage(
                model, 'joint', stage2_trainability
            )
            optimizer = _build_downstream_optimizer(
                model, smiles_lr, graph_lr, geom_lr, fp_lr,
                fusion_lr, head_lr, weight_decay,
                mts_o8_lr=mts_o8_lr,
                mts_geometry_lr=mts_geometry_lr,
                mts_adapter_lr=mts_adapter_lr,
                mts_glt_fusion_stage='joint',
            )
            joint_steps = max(
                1, len(train_loader) * (int(num_epochs) - fusion_warm_epochs)
            )

            def joint_lr_lambda(step):
                progress = float(step) / float(max(1, joint_steps - 1))
                return 0.5 * (
                    1.0 + np.cos(np.pi * min(1.0, max(0.0, progress)))
                )

            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, joint_lr_lambda
            )
            epochs_no_improve = 0
            print(
                f'FusionWarm Stage 2 begins at epoch {epoch + 1}: '
                f'trainability={stage2_trainability}; optimizer and cosine '
                'scheduler rebuilt.'
            )
        epoch_stage = (
            'fusion_warm'
            if fusion_warm and epoch < fusion_warm_epochs
            else ('joint_finetune' if fusion_warm else 'legacy')
        )
        # Training phase
        (
            avg_train_loss, train_r2, avg_fused_loss,
            avg_auxiliary_loss, avg_fusion_prior_loss, avg_cross_task_aux_loss,
            epoch_timing,
        ) = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scheduler,
            device,
            epoch=epoch + 1,
            max_grad_norm=max_grad_norm,
            unimodal_aux_weight=unimodal_aux_weight,
            fusion_prior_kl_weight=fusion_prior_kl_weight,
            fusion_prior=fusion_prior,
            cross_task_aux_weight=cross_task_aux_weight,
            pcgrad=pcgrad,
            amp_dtype=amp_dtype,
            return_timing=True,
            mts_glt_frozen_encoders_eval=(epoch_stage == 'fusion_warm'),
            mts_glt_frozen_encoder_policy=(
                'both_frozen'
                if epoch_stage == 'fusion_warm'
                else (
                    stage2_trainability
                    if fusion_warm and stage2_trainability != 'joint'
                    else None
                )
            ),
        )
        training_timing.append(epoch_timing)

        if state_averager is not None and epoch >= swa_start_epoch:
            state_averager.update(model)

        # Validation phase
        avg_val_loss, val_r2, val_targets, val_predictions = evaluate(
            model, val_loader, criterion, device, scaler=scaler,
            amp_dtype=amp_dtype,
        )
        val_true_raw = scaler.inverse_transform(np.asarray(val_targets))
        val_pred_raw = scaler.inverse_transform(np.asarray(val_predictions))
        val_rmse = float(np.sqrt(metrics.mean_squared_error(
            val_true_raw, val_pred_raw
        )))
        epoch_audit = (
            collect_mts_glt_fusion_audit(model, val_loader, device)
            if fusion_warm else None
        )
        # Early stopping
        is_best = bool(val_r2 > best_val_r2)
        if is_best:
            best_val_r2 = val_r2
            best_val_rmse = val_rmse
            best_epoch = epoch + 1
            best_model_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
            print(f"Epoch {epoch+1}: Validation R2 improved to {val_r2:.4f}.")
        else:
            if epoch_stage == 'fusion_warm':
                print(
                    f"Epoch {epoch+1}: No validation improvement; "
                    "FusionWarm does not consume early-stopping patience."
                )
            else:
                epochs_no_improve += 1
                print(f"Epoch {epoch+1}: No improvement in Validation R2 for {epochs_no_improve} epoch(s).")

        if postmortem_encoder is not None:
            row = {
                'epoch': int(epoch + 1),
                'stage': epoch_stage,
                'validation_r2': float(val_r2),
                'is_best': is_best,
            }
            if epoch_audit is not None:
                row.update(_mts_glt_trajectory_fields(epoch_audit))
            gate_trajectory.append(row)

        if epoch_stage != 'fusion_warm' and epochs_no_improve >= patience:
            print(f"Early stopping after {patience} epochs with no improvement.")
            break

        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Training Loss: {avg_train_loss:.4f}, Training R2: {train_r2:.4f}")
        if float(unimodal_aux_weight) > 0:
            print(
                f"Fused Loss: {avg_fused_loss:.4f}, Unimodal Auxiliary Loss: "
                f"{avg_auxiliary_loss:.4f} (weight={float(unimodal_aux_weight):.3f})"
            )
        if float(fusion_prior_kl_weight) > 0:
            print(
                f"Fusion Prior KL: {avg_fusion_prior_loss:.6f} "
                f"(weight={float(fusion_prior_kl_weight):.3f})"
            )
        if float(cross_task_aux_weight) > 0:
            print(
                f"Cross-task Auxiliary Loss: {avg_cross_task_aux_loss:.4f} "
                f"(weight={float(cross_task_aux_weight):.3f})"
            )
        print(f"Validation Loss: {avg_val_loss:.4f}, Validation R2: {val_r2:.4f}")
        print("-" * 50)

    swa_selected = False
    swa_val_r2 = float('nan')
    swa_snapshots = state_averager.count if state_averager is not None else 0
    if state_averager is not None and swa_snapshots >= 2:
        model.load_state_dict(state_averager.state_dict())
        _, swa_val_r2, swa_targets, swa_predictions = evaluate(
            model, val_loader, criterion, device, scaler=scaler,
            amp_dtype=amp_dtype,
        )
        swa_val_rmse = float(np.sqrt(metrics.mean_squared_error(
            scaler.inverse_transform(np.asarray(swa_targets)),
            scaler.inverse_transform(np.asarray(swa_predictions)),
        )))
        if np.isfinite(swa_val_r2) and swa_val_r2 > best_val_r2:
            best_val_r2 = swa_val_r2
            best_val_rmse = swa_val_rmse
            best_model_state = copy.deepcopy(state_averager.state_dict())
            swa_selected = True
            best_epoch = -1
            print(
                f"SWA checkpoint selected: validation R2={swa_val_r2:.4f} "
                f"from {swa_snapshots} epoch snapshots."
            )
        else:
            print(
                f"SWA checkpoint rejected: validation R2={swa_val_r2:.4f}, "
                f"best raw R2={best_val_r2:.4f}, snapshots={swa_snapshots}."
            )

    # Load best model
    if best_model_state is None:
        # A non-finite validation metric must not turn a completed fold into a
        # NoneType crash; retain the final finite parameter state instead.
        best_model_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_model_state)

    postmortem = None
    if postmortem_components is not None:
        # load_state_dict updates the same encoder object in place.
        best_weight = (
            postmortem_components['fusion_projection'].weight.detach().cpu()
        )
        projection_relative_change = float(
            torch.linalg.vector_norm(best_weight - initial_fusion_weight)
            / (torch.linalg.vector_norm(initial_fusion_weight) + 1e-8)
        )
        final_alpha = _mts_glt_gate_summary(postmortem_components)
        postmortem = {
            'fusion_strategy': fusion_strategy,
            'fusion_kind': postmortem_components['kind'],
            'fusion_warm_epochs': fusion_warm_epochs if fusion_warm else 0,
            'stage2_trainability': stage2_trainability,
            'initial_alpha': float(mts_glt_initial_alpha) if fusion_warm else 0.0,
            'initial_validation': initial_fusion_audit,
            'gate_trajectory': gate_trajectory,
            'final_alpha': final_alpha,
            'projection_relative_change': projection_relative_change,
            'validation': collect_mts_glt_fusion_audit(
                model, val_loader, device
            ),
            'test': collect_mts_glt_fusion_audit(model, test_loader, device),
        }
        if 'gate' in final_alpha:
            postmortem.update({
                'final_gate': final_alpha['gate'],
                'final_tanh_gate': final_alpha['tanh_gate'],
            })

    # In refit mode, defer the only outer-test evaluation until after refitting.
    test_metrics = (
        test_model(
            model, test_loader, scaler, device,
            return_predictions=return_predictions,
            amp_dtype=amp_dtype,
        )
        if evaluate_test else {}
    )
    test_metrics.update({
        'best_val_r2': float(best_val_r2),
        'best_val_rmse': float(best_val_rmse),
        'best_epoch': int(best_epoch),
        'swa_selected': swa_selected,
        'swa_snapshots': swa_snapshots,
        'swa_val_r2': swa_val_r2,
        'training_steps': int(sum(
            item['training_steps'] for item in training_timing
        )),
        'training_seconds': float(sum(
            item['training_seconds'] for item in training_timing
        )),
        'optimizer_steps_per_second': (
            float(sum(item['training_steps'] for item in training_timing))
            / max(float(sum(item['training_seconds'] for item in training_timing)), 1e-12)
        ),
        'training_epoch_timing': training_timing,
    })
    if postmortem is not None:
        test_metrics['_mts_glt_postmortem'] = postmortem
    graphgate_validation = collect_mts_graphgate_audit(model, val_loader, device)
    if graphgate_validation is not None:
        test_metrics['_mts_glt_graphgate_audit'] = {
            'validation': graphgate_validation,
            'test': collect_mts_graphgate_audit(model, test_loader, device),
        }
    return test_metrics


def compute_contrastive_loss(embeddings, temperature=0.07):
    """
    Compute contrastive loss across modalities to make embeddings from same sample close and different samples far apart.
    
    Args:
        embeddings: Tensor of shape [batch_size, num_modalities, embedding_dim]
        temperature: Temperature parameter for scaling similarity scores
        
    Returns:
        Contrastive loss value
    """
    batch_size, num_modalities, embedding_dim = embeddings.shape
    device = embeddings.device
    total_loss = 0.0
    count = 0

    for i in range(num_modalities):
        for j in range(num_modalities):
            if i != j:
                embedding_i = embeddings[:, i, :]  # [batch_size, embedding_dim]
                embedding_j = embeddings[:, j, :]  # [batch_size, embedding_dim]

                # Normalize embeddings
                embedding_i = nn.functional.normalize(embedding_i, p=2, dim=1)
                embedding_j = nn.functional.normalize(embedding_j, p=2, dim=1)

                # Compute similarity matrix [batch_size, batch_size]
                logits = torch.matmul(embedding_i, embedding_j.T) / temperature
                labels = torch.arange(batch_size).to(device)
                loss_i = nn.functional.cross_entropy(logits, labels)
                total_loss += loss_i
                count += 1

    total_loss = total_loss / count
    return total_loss
