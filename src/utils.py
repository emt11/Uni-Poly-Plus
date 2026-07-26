import copy
import os
import random
import torch
from functools import partial
from torch.utils.data import Subset
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import sklearn.metrics as metrics
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from src.dataset.dataloader import custom_collate


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
        raw_targets = np.array([data.y.item() for data in dataset], dtype=np.float64)
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
    for data, value in zip(dataset, scaled_targets):
        data.y = torch.tensor(value, dtype=torch.float)

    return target_scaler

def get_data_loader(
    dataset, indices=None, batch_size=32, shuffle=False, drop_last=False,
    random_conformer=None, num_workers=0, pin_memory=None, persistent_workers=None,
    sampler=None,
):
    if indices is None:
        indices = range(len(dataset))
    subset_dataset = Subset(dataset, [int(index) for index in indices])
    if random_conformer is None:
        random_conformer = bool(shuffle)

    loader = DataLoader(
        subset_dataset,
        batch_size=batch_size,
        collate_fn=partial(custom_collate, random_conformer=random_conformer),
        shuffle=bool(shuffle and sampler is None),
        sampler=sampler,
        drop_last=drop_last,
        num_workers=int(num_workers),
        pin_memory=torch.cuda.is_available() if pin_memory is None else bool(pin_memory),
        persistent_workers=(int(num_workers) > 0) if persistent_workers is None else bool(persistent_workers),
    )

    print(f"Created dataloader with {len(subset_dataset)} samples")
    return loader

def train_epoch(
    model, train_loader, criterion, optimizer, scheduler, device, epoch=None,
    max_grad_norm=1.0, unimodal_aux_weight=0.0,
    fusion_prior_kl_weight=0.0, fusion_prior=(0.30, 0.40, 0.30),
    cross_task_aux_weight=0.0,
):
    model.train()
    train_losses = []
    fused_losses = []
    auxiliary_losses = []
    fusion_prior_losses = []
    cross_task_aux_losses = []
    train_preds = []
    train_targets = []

    progress_bar = tqdm(train_loader, desc="Training")
    for batch in progress_bar:
        batch = batch.to(device)
        optimizer.zero_grad()

        outputs, embeddings = model(batch)
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
        if float(cross_task_aux_weight) > 0:
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
            if bool(auxiliary_mask.any()):
                cross_task_aux_loss = criterion(
                    auxiliary_predictions[auxiliary_mask],
                    auxiliary_targets[auxiliary_mask],
                )
        loss = (
            fused_loss
            + float(unimodal_aux_weight) * auxiliary_loss
            + float(fusion_prior_kl_weight) * fusion_prior_loss
            + float(cross_task_aux_weight) * cross_task_aux_loss
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
        optimizer.step()
        scheduler.step()
        train_losses.append(loss.item())
        fused_losses.append(fused_loss.item())
        auxiliary_losses.append(auxiliary_loss.item())
        fusion_prior_losses.append(fusion_prior_loss.item())
        cross_task_aux_losses.append(cross_task_aux_loss.item())
        train_preds.extend(outputs.detach().cpu().numpy())
        train_targets.extend(batch.y.detach().cpu().numpy())

    avg_train_loss = sum(train_losses) / len(train_losses)
    train_r2 = r2_score(train_targets, train_preds)
    avg_fused_loss = sum(fused_losses) / len(fused_losses)
    avg_auxiliary_loss = sum(auxiliary_losses) / len(auxiliary_losses)
    avg_fusion_prior_loss = sum(fusion_prior_losses) / len(fusion_prior_losses)
    avg_cross_task_aux_loss = sum(cross_task_aux_losses) / len(cross_task_aux_losses)

    return (
        avg_train_loss, train_r2, avg_fused_loss,
        avg_auxiliary_loss, avg_fusion_prior_loss, avg_cross_task_aux_loss,
    )

def evaluate(model, data_loader, criterion, device):
    model.eval()
    losses = []
    preds = []
    targets = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            batch = batch.to(device)
            outputs,_ = model(batch)
            loss = criterion(outputs, batch.y)
            losses.append(loss.item())
            preds.extend(outputs.cpu().numpy())
            targets.extend(batch.y.cpu().numpy())

    avg_loss = sum(losses) / len(losses)
    r2 = r2_score(targets, preds)
    
    return avg_loss, r2, targets, preds

def test_model(model, test_loader, scaler, device):
    model.eval()
    test_preds = []
    test_targets = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            batch = batch.to(device)
            outputs,_ = model(batch)
            test_preds.extend(outputs.cpu().numpy())
            test_targets.extend(batch.y.cpu().numpy())

    y_true = np.array(test_targets)
    y_pred = np.array(test_preds)
    y_true_unscaled = scaler.inverse_transform(y_true)
    y_pred_unscaled = scaler.inverse_transform(y_pred)
    test_r2 = r2_score(y_true_unscaled, y_pred_unscaled)
    test_mae = metrics.mean_absolute_error(y_true_unscaled, y_pred_unscaled)
    test_rmse = np.sqrt(metrics.mean_squared_error(y_true_unscaled, y_pred_unscaled))

    return {'test_r2': test_r2, 'test_mae': test_mae, 'test_rmse': test_rmse}


def _base_model(model):
    return model.module if isinstance(model, nn.DataParallel) else model


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


def _configure_parallel_stage3_trainability(model, phase, fp_trainable=False):
    """Apply staged encoder unfreezing while keeping fusion/head trainable."""
    base = _base_model(model)
    for encoder in base.encoders.values():
        _set_module_trainable(encoder, False)
    _set_module_trainable(getattr(base, 'alignment_projections', None), False)
    _set_module_trainable(getattr(base, 'parallel_attention_fusion', None), True)
    _set_module_trainable(getattr(base, 'mlp', None), True)
    _set_module_trainable(getattr(base, 'modality_heads', None), True)
    if fp_trainable and 'fp' in base.encoders:
        _set_module_trainable(base.encoders['fp'], True)
    if phase <= 0:
        return

    if 'graph' in base.encoders:
        graph_module = base.encoders['graph']
        _set_module_trainable(graph_module.norm, True)
        _set_module_trainable(graph_module.projection, True)
        graph_layers = list(getattr(graph_module.encoder, 'layers', []))
        graph_count = 2 if phase == 1 else 3
        for layer in graph_layers[-graph_count:]:
            _set_module_trainable(layer, True)
    if 'smiles' in base.encoders:
        smiles_module = base.encoders['smiles']
        _set_module_trainable(smiles_module.norm, True)
        _set_module_trainable(smiles_module.projection, True)
        layers = list(getattr(getattr(smiles_module.encoder, 'encoder', None), 'layer', []))
        smiles_count = 2 if phase == 1 else 4
        for layer in layers[-smiles_count:]:
            _set_module_trainable(layer, True)


def _build_downstream_optimizer(
    model, smiles_lr, graph_lr, geom_lr, fp_lr, fusion_lr, head_lr, weight_decay
):
    base = _base_model(model)
    groups = []
    used = set()

    def add_group(module, lr, name):
        if module is None:
            return
        params = [param for param in module.parameters() if id(param) not in used]
        if params:
            used.update(id(param) for param in params)
            groups.append({
                'params': params,
                'lr': float(lr),
                'weight_decay': float(weight_decay),
                'name': name,
            })

    encoders = base.encoders
    add_group(encoders['smiles'] if 'smiles' in encoders else None, smiles_lr, 'smiles')
    add_group(encoders['graph'] if 'graph' in encoders else None, graph_lr, 'graph')
    add_group(encoders['geom'] if 'geom' in encoders else None, geom_lr, 'geom')
    add_group(encoders['fp'] if 'fp' in encoders else None, fp_lr, 'fp')
    fusion_module = (
        base.parallel_attention_fusion
        if getattr(base, 'fusion_type', '') == 'parallel_attention'
        else base.fusion_module
    )
    add_group(fusion_module, fusion_lr, 'fusion')
    add_group(base.mlp, head_lr, 'regression_head')
    add_group(getattr(base, 'modality_heads', None), head_lr, 'modality_heads')
    add_group(getattr(base, 'cross_task_aux_heads', None), head_lr, 'cross_task_aux_heads')
    remaining = [param for param in base.parameters() if id(param) not in used]
    if remaining:
        groups.append({
            'params': remaining,
            'lr': float(fusion_lr),
            'weight_decay': float(weight_decay),
            'name': 'remaining',
        })
    return torch.optim.AdamW(groups)


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
    parallel_staged = (
        getattr(base, 'graph_encoder_type', '') in {'scage', 'mips'}
        and getattr(base, 'fusion_type', '') == 'parallel_attention'
    )
    if parallel_staged:
        _configure_parallel_stage3_trainability(model, phase=0, fp_trainable=False)
    else:
        _set_smiles_trainable(model, trainable=False)
    optimizer = _build_downstream_optimizer(
        model, smiles_lr, graph_lr, geom_lr, fp_lr,
        fusion_lr, head_lr, weight_decay,
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
        fp_trainable = (
            int(fp_unfreeze_epoch) >= 0
            and epoch >= int(fp_unfreeze_epoch)
        )
        if parallel_staged and epoch == int(freeze_smiles_epochs):
            _configure_parallel_stage3_trainability(
                model, phase=1, fp_trainable=fp_trainable
            )
        elif parallel_staged and epoch == int(deep_unfreeze_epoch):
            _configure_parallel_stage3_trainability(
                model, phase=2, fp_trainable=fp_trainable
            )
        elif parallel_staged and epoch == int(fp_unfreeze_epoch):
            phase = 2 if epoch >= int(deep_unfreeze_epoch) else (
                1 if epoch >= int(freeze_smiles_epochs) else 0
            )
            _configure_parallel_stage3_trainability(
                model, phase=phase, fp_trainable=True
            )
        elif not parallel_staged and epoch == int(freeze_smiles_epochs):
            _set_smiles_trainable(model, trainable=True)

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
):
    # Define loss function and optimizer
    criterion = nn.SmoothL1Loss(beta=float(huber_beta)) if regression_loss == 'huber' else nn.MSELoss()
    base = _base_model(model)
    parallel_staged = (
        getattr(base, 'graph_encoder_type', '') in {'scage', 'mips'}
        and getattr(base, 'fusion_type', '') == 'parallel_attention'
    )
    if parallel_staged:
        _configure_parallel_stage3_trainability(model, phase=0, fp_trainable=False)
    else:
        _set_smiles_trainable(model, trainable=False)
    optimizer = _build_downstream_optimizer(
        model, smiles_lr, graph_lr, geom_lr, fp_lr, fusion_lr, head_lr, weight_decay
    )
    total_steps = max(1, len(train_loader) * num_epochs)
    warmup_steps = min(total_steps - 1, max(0, int(warmup_epochs) * len(train_loader)))
    def lr_lambda(step):
        if warmup_steps and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + np.cos(np.pi * min(1.0, max(0.0, progress))))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_val_r2 = -float('inf')
    best_epoch = -1
    best_model_state = None
    epochs_no_improve = 0
    swa_start_epoch = int(swa_start_epoch)
    state_averager = _CpuStateAverager() if swa_start_epoch >= 0 else None

    for epoch in range(num_epochs):
        fp_trainable = int(fp_unfreeze_epoch) >= 0 and epoch >= int(fp_unfreeze_epoch)
        if parallel_staged and epoch == int(freeze_smiles_epochs):
            _configure_parallel_stage3_trainability(
                model, phase=1, fp_trainable=fp_trainable
            )
            print(
                "Unfroze the last 2 graph and SMILES layers; "
                f"FP is {'trainable' if fp_trainable else 'frozen'}."
            )
        elif parallel_staged and epoch == int(deep_unfreeze_epoch):
            _configure_parallel_stage3_trainability(
                model, phase=2, fp_trainable=fp_trainable
            )
            print(
                "Unfroze the last 3 graph and last 4 SMILES layers; "
                f"FP is {'trainable' if fp_trainable else 'frozen'}."
            )
        elif parallel_staged and epoch == int(fp_unfreeze_epoch):
            phase = 2 if epoch >= int(deep_unfreeze_epoch) else (
                1 if epoch >= int(freeze_smiles_epochs) else 0
            )
            _configure_parallel_stage3_trainability(
                model, phase=phase, fp_trainable=True
            )
            print("Unfroze the FP encoder with its discriminative learning rate.")
        elif not parallel_staged and epoch == int(freeze_smiles_epochs):
            _set_smiles_trainable(model, trainable=True)
            print("Unfroze the last four SMILES Transformer layers.")
        # Training phase
        (
            avg_train_loss, train_r2, avg_fused_loss,
            avg_auxiliary_loss, avg_fusion_prior_loss, avg_cross_task_aux_loss,
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
        )

        if state_averager is not None and epoch >= swa_start_epoch:
            state_averager.update(model)

        # Validation phase
        avg_val_loss, val_r2, _, _ = evaluate(model, val_loader, criterion, device)
        # Early stopping
        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            best_epoch = epoch + 1
            best_model_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
            print(f"Epoch {epoch+1}: Validation R2 improved to {val_r2:.4f}.")
        else:
            epochs_no_improve += 1
            print(f"Epoch {epoch+1}: No improvement in Validation R2 for {epochs_no_improve} epoch(s).")

        if epochs_no_improve >= patience:
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
        _, swa_val_r2, _, _ = evaluate(model, val_loader, criterion, device)
        if np.isfinite(swa_val_r2) and swa_val_r2 > best_val_r2:
            best_val_r2 = swa_val_r2
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

    # In refit mode, defer the only outer-test evaluation until after refitting.
    test_metrics = (
        test_model(model, test_loader, scaler, device)
        if evaluate_test else {}
    )
    test_metrics.update({
        'best_val_r2': float(best_val_r2),
        'best_epoch': int(best_epoch),
        'swa_selected': swa_selected,
        'swa_snapshots': swa_snapshots,
        'swa_val_r2': swa_val_r2,
    })
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
