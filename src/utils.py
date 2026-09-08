"""Training utilities for the graph-only MTS-GLT-v2 baseline."""

from __future__ import annotations

import copy
import os
import random
import time
from contextlib import nullcontext

import numpy as np
import sklearn.metrics as metrics
import torch
import torch.nn as nn
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from src.dataset.dataloader import custom_collate, mips_trimer_collate


def set_global_seed(seed, deterministic=True):
    """Seed Python, NumPy and PyTorch for reproducible baseline runs."""
    seed = int(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
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
    def __init__(self, task, scaler, eps=1e-4, transform_mode="auto"):
        self.task = str(task)
        self.scaler = scaler
        self.eps = float(eps)
        self.requested_transform_mode = str(transform_mode)
        self.transform_mode = (
            "log" if transform_mode == "recommended" and self.task in {"eps", "nc"}
            else ("standard" if transform_mode == "recommended" else str(transform_mode))
        )

    def _pre_transform(self, values):
        values = np.asarray(values, dtype=np.float64).reshape(-1, 1)
        if self.transform_mode == "log":
            if np.any(values <= 0):
                raise ValueError("Log target transform requires strictly positive labels")
            return np.log(values)
        if self.transform_mode == "auto" and self.task in {"er", "iv"}:
            return np.log10(values)
        if self.transform_mode == "auto" and self.task == "xc":
            p = np.clip(values / 100.0, self.eps, 1.0 - self.eps)
            return np.log(p / (1.0 - p))
        return values

    def _inverse_pre_transform(self, values):
        values = np.asarray(values, dtype=np.float64).reshape(-1, 1)
        if self.transform_mode == "log":
            return np.exp(values)
        if self.transform_mode == "auto" and self.task == "xc":
            return 100.0 / (1.0 + np.exp(-values))
        return values

    def transform(self, values):
        return self.scaler.transform(self._pre_transform(values))

    def inverse_transform(self, values):
        values = np.asarray(values, dtype=np.float64).reshape(-1, 1)
        return self._inverse_pre_transform(self.scaler.inverse_transform(values))


def scale_targets(dataset, task, train_indices=None, raw_targets=None, transform_mode="auto"):
    if raw_targets is None:
        raw_targets = np.asarray(getattr(dataset, "raw_targets"), dtype=np.float64)
    else:
        raw_targets = np.asarray(raw_targets, dtype=np.float64)
    if train_indices is None:
        train_indices = np.arange(len(raw_targets))
    if transform_mode not in {"auto", "standard", "log", "recommended"}:
        raise ValueError(f"Unsupported target transform: {transform_mode}")
    target_scaler = TargetScaler(task, StandardScaler(), transform_mode=transform_mode)
    target_scaler.scaler.fit(target_scaler._pre_transform(raw_targets[train_indices]))
    scaled = target_scaler.transform(raw_targets)
    if hasattr(dataset, "set_target_override"):
        dataset.set_target_override(scaled.reshape(-1))
    else:
        for data, value in zip(dataset, scaled.reshape(-1)):
            data.y = torch.tensor([float(value)], dtype=torch.float)
    return target_scaler


def get_data_loader(
    dataset,
    indices=None,
    batch_size=32,
    shuffle=False,
    drop_last=False,
    num_workers=0,
    pin_memory=None,
    persistent_workers=None,
    sampler=None,
    generator=None,
    prefetch_factor=2,
):
    loader_dataset = dataset if indices is None else Subset(dataset, [int(i) for i in indices])
    collate = mips_trimer_collate if bool(getattr(dataset, "is_mts_route", False)) else custom_collate
    kwargs = {
        "dataset": loader_dataset,
        "batch_size": int(batch_size),
        "collate_fn": collate,
        "shuffle": bool(shuffle and sampler is None),
        "sampler": sampler,
        "drop_last": bool(drop_last),
        "num_workers": int(num_workers),
        "pin_memory": torch.cuda.is_available() if pin_memory is None else bool(pin_memory),
        "persistent_workers": int(num_workers) > 0 if persistent_workers is None else bool(persistent_workers),
        "generator": generator,
    }
    if int(num_workers) > 0:
        kwargs["prefetch_factor"] = max(1, int(prefetch_factor))
    loader = DataLoader(**kwargs)
    print(f"Created dataloader with {len(loader_dataset)} samples")
    return loader


def _autocast_context(device, amp_dtype):
    enabled = str(amp_dtype) == "bf16" and torch.device(device).type == "cuda"
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16) if enabled else nullcontext()


def finetune_bf16_parity_gate(model, batch, criterion, device):
    """Check finite BF16 loss/gradients before an explicitly requested BF16 run."""
    if torch.device(device).type != "cuda" or not torch.cuda.is_bf16_supported():
        return False, {"finite": False, "reason": "cuda_bf16_unavailable"}
    batch = batch.to(device, non_blocking=True)
    was_training = model.training
    model.eval()

    def evaluate(dtype):
        model.zero_grad(set_to_none=True)
        with _autocast_context(device, dtype):
            output, _ = model(batch)
            loss = criterion(output, batch.y)
        loss.backward()
        grads = [
            parameter.grad.detach().float().reshape(-1)
            for parameter in model.parameters()
            if parameter.requires_grad and parameter.grad is not None
        ]
        return loss.detach().float(), torch.cat(grads) if grads else loss.new_zeros(1).float()

    fp32_loss, fp32_gradient = evaluate("fp32")
    bf16_loss, bf16_gradient = evaluate("bf16")
    relative_delta = float((bf16_loss - fp32_loss).abs() / fp32_loss.abs().clamp_min(1e-8))
    finite = bool(
        torch.isfinite(fp32_loss)
        and torch.isfinite(bf16_loss)
        and torch.isfinite(fp32_gradient).all()
        and torch.isfinite(bf16_gradient).all()
    )
    model.zero_grad(set_to_none=True)
    model.train(was_training)
    return finite and relative_delta <= 0.02, {"finite": finite, "relative_loss_delta": relative_delta}


def _base_model(model):
    return model.module if isinstance(model, nn.DataParallel) else model


def _set_module_trainable(module, trainable):
    if module is not None:
        for parameter in module.parameters():
            parameter.requires_grad = bool(trainable)


def _mts_glt_encoder(model):
    base = _base_model(model)
    graph = base.encoders["graph"] if "graph" in base.encoders else None
    encoder = getattr(graph, "encoder", None)
    return encoder if getattr(encoder, "architecture_name", "") in {
        "MIPS-Trimer-GLT-v2", "MTS-GLT-v3-Galformer"
    } else None


def _configure_mts_trainability(model):
    """Enable only the graph O8/GLT-v2 encoder and downstream head."""
    base = _base_model(model)
    encoder = _mts_glt_encoder(base)
    if encoder is None:
        raise ValueError("MTS-GLT-v2 trainability requested for a non-baseline model")
    for parameter in base.parameters():
        parameter.requires_grad = False
    graph_module = base.encoders["graph"]
    _set_module_trainable(graph_module, True)
    _set_module_trainable(getattr(encoder.o8, "star_distance_bias", None), False)
    if getattr(encoder, "architecture_name", "") == "MTS-GLT-v3-Galformer":
        _set_module_trainable(getattr(encoder.o8, "md_residual", None), False)
        _set_module_trainable(base.mlp, True)
        return
    # The retained checkpoint contains a compatibility container for the
    # disabled Compact19 branch; it is never a baseline trainable parameter.
    _set_module_trainable(encoder.compact19_residual, False)
    if getattr(encoder, "downstream_mode", "o8_only") == "o8_only":
        _set_module_trainable(encoder.glt, False)
        _set_module_trainable(encoder.atom_fusion_norm, False)
        _set_module_trainable(encoder.atom_fusion_projection, False)
        encoder.atom_channel_gate.requires_grad = False
    _set_module_trainable(base.mlp, True)


def _is_mts_model(model):
    return _mts_glt_encoder(model) is not None


def _build_downstream_optimizer(
    model,
    graph_lr,
    head_lr,
    weight_decay,
    mts_o8_lr=1e-5,
    mts_geometry_lr=1e-5,
    mts_adapter_lr=1e-5,
):
    base = _base_model(model)
    if not _is_mts_model(base):
        raise ValueError("MTS-GLT-v2 optimizer requires the baseline graph encoder")
    encoder = _mts_glt_encoder(base)
    graph_module = base.encoders["graph"]
    groups, used = [], set()

    def add_module(module, lr, name):
        if module is None:
            return
        decay, no_decay = [], []
        for parameter_name, parameter in module.named_parameters():
            if not parameter.requires_grad or id(parameter) in used:
                continue
            used.add(id(parameter))
            normalized = parameter_name.lower()
            if parameter.ndim <= 1 or normalized.endswith("bias") or "norm" in normalized or "gate" in normalized:
                no_decay.append(parameter)
            else:
                decay.append(parameter)
        if decay:
            groups.append({"params": decay, "lr": float(lr), "weight_decay": float(weight_decay), "name": f"{name}/decay"})
        if no_decay:
            groups.append({"params": no_decay, "lr": float(lr), "weight_decay": 0.0, "name": f"{name}/no_decay"})

    add_module(encoder.o8, mts_o8_lr, "o8")
    add_module(getattr(encoder, "glt", None), mts_geometry_lr, "glt")
    if getattr(encoder, "architecture_name", "") == "MTS-GLT-v3-Galformer":
        add_module(getattr(encoder, "concat_norm", None), mts_adapter_lr, "concat_norm")
        add_module(getattr(encoder, "concat_projection", None), mts_adapter_lr, "concat_projection")
        add_module(encoder.md_residual, mts_adapter_lr, "md200_node_residual")
    else:
        add_module(encoder.atom_fusion_norm, mts_adapter_lr, "atom_fusion_norm")
        add_module(encoder.atom_fusion_projection, mts_adapter_lr, "atom_fusion_projection")
        add_module(encoder.compact19_residual, mts_adapter_lr, "compact19")
    add_module(graph_module, graph_lr, "graph_adapter")
    add_module(base.mlp, head_lr, "regression_head")
    remaining = [parameter for parameter in base.parameters() if parameter.requires_grad and id(parameter) not in used]
    if remaining:
        groups.append({"params": remaining, "lr": float(head_lr), "weight_decay": float(weight_decay), "name": "remaining"})
    if not groups:
        raise RuntimeError("MTS-GLT-v2 optimizer found no trainable parameters")
    return torch.optim.AdamW(groups)


def _safe_r2(targets, predictions):
    targets = np.asarray(targets).reshape(-1)
    predictions = np.asarray(predictions).reshape(-1)
    return float(r2_score(targets, predictions)) if targets.size > 1 else float("nan")


def train_epoch(
    model,
    train_loader,
    criterion,
    optimizer,
    scheduler,
    device,
    epoch=None,
    max_grad_norm=1.0,
    amp_dtype="fp32",
    return_timing=False,
):
    model.train()
    started = time.perf_counter()
    losses, predictions, targets = [], [], []
    steps = 0
    for batch in tqdm(train_loader, desc="Training"):
        batch = batch.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with _autocast_context(device, amp_dtype):
            output, _ = model(batch)
            loss = criterion(output, batch.y)
        loss.backward()
        if float(max_grad_norm) > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(max_grad_norm))
        optimizer.step()
        scheduler.step()
        steps += 1
        losses.append(loss.detach().float())
        predictions.append(output.detach().float())
        targets.append(batch.y.detach().float())
    if not losses:
        raise RuntimeError("MTS-GLT-v2 training loader is empty")
    mean_loss = float(torch.stack(losses).mean().cpu())
    result = (
        mean_loss,
        _safe_r2(torch.cat(targets).cpu().numpy(), torch.cat(predictions).cpu().numpy()),
        mean_loss,
        0.0,
        0.0,
        0.0,
    )
    if return_timing:
        elapsed = time.perf_counter() - started
        return result + ({
            "training_steps": int(steps),
            "training_seconds": float(elapsed),
            "optimizer_steps_per_second": float(steps / max(elapsed, 1e-12)),
        },)
    return result


def evaluate(model, data_loader, criterion, device, scaler=None, amp_dtype="fp32"):
    model.eval()
    losses, predictions, targets = [], [], []
    with torch.inference_mode():
        for batch in tqdm(data_loader, desc="Evaluating"):
            batch = batch.to(device, non_blocking=True)
            with _autocast_context(device, amp_dtype):
                output, _ = model(batch)
                loss = criterion(output, batch.y)
            losses.append(loss.detach().float())
            predictions.append(output.detach().float())
            targets.append(batch.y.detach().float())
    if not losses:
        raise RuntimeError("MTS-GLT-v2 evaluation loader is empty")
    target_values = torch.cat(targets).cpu().numpy()
    prediction_values = torch.cat(predictions).cpu().numpy()
    if scaler is not None:
        target_values = scaler.inverse_transform(target_values)
        prediction_values = scaler.inverse_transform(prediction_values)
    return float(torch.stack(losses).mean().cpu()), _safe_r2(target_values, prediction_values), target_values, prediction_values


def test_model(model, test_loader, scaler, device, return_predictions=False, amp_dtype="fp32"):
    model.eval()
    predictions, targets = [], []
    with torch.inference_mode():
        for batch in tqdm(test_loader, desc="Testing"):
            batch = batch.to(device, non_blocking=True)
            with _autocast_context(device, amp_dtype):
                output, _ = model(batch)
            predictions.append(output.detach().float())
            targets.append(batch.y.detach().float())
    if not predictions:
        raise RuntimeError("MTS-GLT-v2 test loader is empty")
    y_true = scaler.inverse_transform(torch.cat(targets).cpu().numpy())
    y_pred = scaler.inverse_transform(torch.cat(predictions).cpu().numpy())
    result = {
        "test_r2": _safe_r2(y_true, y_pred),
        "test_mae": float(metrics.mean_absolute_error(y_true, y_pred)),
        "test_rmse": float(np.sqrt(metrics.mean_squared_error(y_true, y_pred))),
    }
    if return_predictions:
        result["_y_true"] = y_true.reshape(-1).astype(np.float64)
        result["_y_pred"] = y_pred.reshape(-1).astype(np.float64)
    return result


def _cosine_scheduler(optimizer, total_steps, warmup_steps):
    total_steps = max(1, int(total_steps))
    warmup_steps = min(total_steps - 1, max(0, int(warmup_steps)))

    def scale(step):
        if warmup_steps and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + np.cos(np.pi * min(1.0, max(0.0, progress))))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, scale)


def train_and_evaluate(
    model,
    scaler,
    train_loader,
    val_loader,
    test_loader,
    device,
    num_epochs=100,
    patience=10,
    max_grad_norm=1.0,
    graph_lr=1e-5,
    head_lr=1e-4,
    weight_decay=0.02,
    warmup_epochs=5,
    regression_loss="huber",
    huber_beta=0.5,
    evaluate_test=True,
    return_predictions=False,
    mts_o8_lr=1e-5,
    mts_geometry_lr=1e-5,
    mts_adapter_lr=1e-5,
    amp_dtype="fp32",
    **_ignored,
):
    criterion = nn.SmoothL1Loss(beta=float(huber_beta)) if regression_loss == "huber" else nn.MSELoss()
    _configure_mts_trainability(model)
    if str(amp_dtype) == "bf16":
        passed, details = finetune_bf16_parity_gate(model, next(iter(train_loader)), criterion, device)
        print(f"Fine-tune BF16 parity gate: {details}, pass={passed}")
        if not passed:
            raise RuntimeError("Fine-tune BF16 parity gate failed")
    optimizer = _build_downstream_optimizer(
        model, graph_lr, head_lr, weight_decay,
        mts_o8_lr=mts_o8_lr, mts_geometry_lr=mts_geometry_lr, mts_adapter_lr=mts_adapter_lr,
    )
    scheduler = _cosine_scheduler(optimizer, int(num_epochs) * max(1, len(train_loader)), int(warmup_epochs) * len(train_loader))
    best_state, best_val_r2, best_val_rmse, best_epoch = None, -float("inf"), float("inf"), -1
    no_improve = 0
    timing = []
    for epoch in range(int(num_epochs)):
        train_result = train_epoch(
            model, train_loader, criterion, optimizer, scheduler, device,
            epoch=epoch + 1, max_grad_norm=max_grad_norm,
            amp_dtype=amp_dtype, return_timing=True,
        )
        train_loss, train_r2, _, _, _, _, epoch_timing = train_result
        timing.append(epoch_timing)
        val_loss, val_r2, val_true, val_pred = evaluate(model, val_loader, criterion, device, scaler=scaler, amp_dtype=amp_dtype)
        val_rmse = float(np.sqrt(metrics.mean_squared_error(val_true, val_pred)))
        if np.isfinite(val_r2) and val_r2 > best_val_r2:
            best_val_r2 = float(val_r2)
            best_val_rmse = val_rmse
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
            print(f"Epoch {epoch + 1}: Validation R2 improved to {val_r2:.4f}.")
        else:
            no_improve += 1
            print(f"Epoch {epoch + 1}: No improvement in Validation R2 for {no_improve} epoch(s).")
        print(f"Epoch {epoch + 1}/{num_epochs} - train={train_loss:.4f} val={val_loss:.4f} val_r2={val_r2:.4f}")
        if no_improve >= int(patience):
            print(f"Early stopping after {patience} epochs with no improvement.")
            break
    if best_state is None:
        best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state, strict=True)
    result = test_model(model, test_loader, scaler, device, return_predictions=return_predictions, amp_dtype=amp_dtype) if evaluate_test else {}
    total_steps = sum(item["training_steps"] for item in timing)
    total_seconds = sum(item["training_seconds"] for item in timing)
    result.update({
        "best_val_r2": float(best_val_r2),
        "best_val_rmse": float(best_val_rmse),
        "best_epoch": int(best_epoch),
        "training_steps": int(total_steps),
        "training_seconds": float(total_seconds),
        "optimizer_steps_per_second": float(total_steps / max(total_seconds, 1e-12)),
        "training_epoch_timing": timing,
        "swa_selected": False,
        "swa_snapshots": 0,
        "swa_val_r2": float("nan"),
    })
    return result


def fit_fixed_epochs(
    model,
    train_loader,
    device,
    num_epochs,
    max_grad_norm=1.0,
    graph_lr=1e-5,
    head_lr=1e-4,
    weight_decay=0.02,
    warmup_epochs=5,
    regression_loss="huber",
    huber_beta=0.5,
    mts_o8_lr=1e-5,
    mts_geometry_lr=1e-5,
    mts_adapter_lr=1e-5,
    amp_dtype="fp32",
):
    """Run a fixed-epoch baseline fit for a local smoke check."""
    criterion = nn.SmoothL1Loss(beta=float(huber_beta)) if regression_loss == "huber" else nn.MSELoss()
    _configure_mts_trainability(model)
    optimizer = _build_downstream_optimizer(
        model, graph_lr, head_lr, weight_decay,
        mts_o8_lr=mts_o8_lr, mts_geometry_lr=mts_geometry_lr, mts_adapter_lr=mts_adapter_lr,
    )
    scheduler = _cosine_scheduler(optimizer, int(num_epochs) * max(1, len(train_loader)), int(warmup_epochs) * len(train_loader))
    final_loss = float("nan")
    for epoch in range(int(num_epochs)):
        final_loss = float(train_epoch(model, train_loader, criterion, optimizer, scheduler, device, epoch=epoch + 1, max_grad_norm=max_grad_norm, amp_dtype=amp_dtype)[0])
    return {"refit_epochs": int(num_epochs), "refit_train_loss": final_loss}
