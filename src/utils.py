"""Small training utilities shared by the active GLT dual and MCL-PH routes."""
from __future__ import annotations
import os, random, time
from contextlib import nullcontext
import numpy as np
import sklearn.metrics as metrics
import torch
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


def set_global_seed(seed, deterministic=True):
    seed = int(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)
    return seed


class TargetScaler:
    def __init__(self, task, scaler, eps=1e-4, transform_mode="auto"):
        self.task = str(task); self.scaler = scaler; self.eps = float(eps)
        self.requested_transform_mode = str(transform_mode)
        self.transform_mode = ("log" if transform_mode == "recommended" and self.task in {"eps", "nc"}
                               else ("standard" if transform_mode == "recommended" else str(transform_mode)))
    def _pre_transform(self, values):
        values = np.asarray(values, dtype=np.float64).reshape(-1, 1)
        if self.transform_mode == "log":
            if np.any(values <= 0): raise ValueError("Log target transform requires strictly positive labels")
            return np.log(values)
        if self.transform_mode == "auto" and self.task in {"er", "iv"}: return np.log10(values)
        if self.transform_mode == "auto" and self.task == "xc":
            p = np.clip(values / 100.0, self.eps, 1.0 - self.eps); return np.log(p / (1.0 - p))
        return values
    def _inverse_pre_transform(self, values):
        values = np.asarray(values, dtype=np.float64).reshape(-1, 1)
        if self.transform_mode == "log": return np.exp(values)
        if self.transform_mode == "auto" and self.task == "xc": return 100.0 / (1.0 + np.exp(-values))
        return values
    def transform(self, values): return self.scaler.transform(self._pre_transform(values))
    def inverse_transform(self, values):
        values = np.asarray(values, dtype=np.float64).reshape(-1, 1)
        return self._inverse_pre_transform(self.scaler.inverse_transform(values))


class IdentityScaler:
    """Sklearn-shaped no-op scaler for targets kept in their original units."""
    mean_ = np.array([0.0])
    scale_ = np.array([1.0])

    def transform(self, values): return np.asarray(values, dtype=np.float64).reshape(-1, 1)
    def inverse_transform(self, values): return self.transform(values)


def scale_targets(dataset, task, train_indices=None, raw_targets=None, transform_mode="auto"):
    raw_targets = np.asarray(getattr(dataset, "raw_targets") if raw_targets is None else raw_targets, dtype=np.float64)
    if train_indices is None: train_indices = np.arange(len(raw_targets))
    if transform_mode not in {"auto", "standard", "log", "recommended", "identity"}: raise ValueError(f"Unsupported target transform: {transform_mode}")
    scaler = TargetScaler(task, IdentityScaler() if transform_mode == "identity" else StandardScaler(),
                          transform_mode=transform_mode)
    if transform_mode != "identity":
        scaler.scaler.fit(scaler._pre_transform(raw_targets[train_indices]))
    scaled = scaler.transform(raw_targets)
    if hasattr(dataset, "set_target_override"): dataset.set_target_override(scaled.reshape(-1))
    else:
        for data, value in zip(dataset, scaled.reshape(-1)): data.y = torch.tensor([float(value)], dtype=torch.float)
    return scaler


def _autocast_context(device, amp_dtype):
    enabled = str(amp_dtype) == "bf16" and torch.device(device).type == "cuda"
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16) if enabled else nullcontext()


def _safe_r2(targets, predictions):
    targets = np.asarray(targets).reshape(-1); predictions = np.asarray(predictions).reshape(-1)
    return float(r2_score(targets, predictions)) if targets.size > 1 else float("nan")


def _output(model, batch):
    output = model(batch)
    return output[0] if isinstance(output, tuple) else output


def train_epoch(model, train_loader, criterion, optimizer, scheduler, device, epoch=None,
                max_grad_norm=1.0, amp_dtype="fp32", return_timing=False, fail_nonfinite=False):
    model.train(); started = time.perf_counter(); losses=[]; predictions=[]; targets=[]; steps=0
    first_batch_ready_seconds = first_batch_seconds = None
    for batch in tqdm(train_loader, desc="Training"):
        if first_batch_ready_seconds is None: first_batch_ready_seconds = time.perf_counter() - started
        batch = batch.to(device, non_blocking=True); optimizer.zero_grad(set_to_none=True)
        with _autocast_context(device, amp_dtype):
            output = _output(model, batch); loss = criterion(output, batch.y)
        if fail_nonfinite and not torch.isfinite(loss): raise FloatingPointError("nonfinite training loss")
        loss.backward()
        if float(max_grad_norm) > 0: torch.nn.utils.clip_grad_norm_(model.parameters(), float(max_grad_norm), error_if_nonfinite=fail_nonfinite)
        optimizer.step()
        if scheduler is not None: scheduler.step()
        steps += 1
        if first_batch_seconds is None: first_batch_seconds = time.perf_counter() - started
        losses.append(loss.detach().float()); predictions.append(output.detach().float()); targets.append(batch.y.detach().float())
    if not losses: raise RuntimeError("training loader is empty")
    mean_loss = float(torch.stack(losses).mean().cpu())
    result = (mean_loss, _safe_r2(torch.cat(targets).cpu().numpy(), torch.cat(predictions).cpu().numpy()), mean_loss, 0.0, 0.0, 0.0, 0.0)
    if return_timing:
        elapsed = time.perf_counter() - started
        return result + ({"training_steps": int(steps), "training_seconds": float(elapsed), "optimizer_steps_per_second": float(steps/max(elapsed,1e-12)), "first_batch_ready_seconds": float(first_batch_ready_seconds), "first_batch_seconds": float(first_batch_seconds)},)
    return result


def evaluate(model, data_loader, criterion, device, scaler=None, amp_dtype="fp32"):
    model.eval(); losses=[]; predictions=[]; targets=[]
    with torch.inference_mode():
        for batch in tqdm(data_loader, desc="Evaluating"):
            batch = batch.to(device, non_blocking=True)
            with _autocast_context(device, amp_dtype):
                output = _output(model, batch); loss = criterion(output, batch.y)
            losses.append(loss.detach().float()); predictions.append(output.detach().float()); targets.append(batch.y.detach().float())
    if not losses: raise RuntimeError("evaluation loader is empty")
    target_values = torch.cat(targets).cpu().numpy(); prediction_values = torch.cat(predictions).cpu().numpy()
    if scaler is not None:
        target_values = scaler.inverse_transform(target_values); prediction_values = scaler.inverse_transform(prediction_values)
    return float(torch.stack(losses).mean().cpu()), _safe_r2(target_values, prediction_values), target_values, prediction_values


def test_model(model, test_loader, scaler, device, return_predictions=False, amp_dtype="fp32"):
    model.eval(); predictions=[]; targets=[]
    with torch.inference_mode():
        for batch in tqdm(test_loader, desc="Testing"):
            batch = batch.to(device, non_blocking=True)
            with _autocast_context(device, amp_dtype): output = _output(model, batch)
            predictions.append(output.detach().float()); targets.append(batch.y.detach().float())
    if not predictions: raise RuntimeError("test loader is empty")
    y_true = scaler.inverse_transform(torch.cat(targets).cpu().numpy()); y_pred = scaler.inverse_transform(torch.cat(predictions).cpu().numpy())
    result = {"test_r2": _safe_r2(y_true, y_pred), "test_mae": float(metrics.mean_absolute_error(y_true, y_pred)), "test_rmse": float(np.sqrt(metrics.mean_squared_error(y_true, y_pred)))}
    if return_predictions: result.update(_y_true=y_true.reshape(-1).astype(np.float64), _y_pred=y_pred.reshape(-1).astype(np.float64))
    return result


def _cosine_scheduler(optimizer, total_steps, warmup_steps):
    total_steps=max(1,int(total_steps)); warmup_steps=min(total_steps-1,max(0,int(warmup_steps)))
    def scale(step):
        if warmup_steps and step < warmup_steps: return float(step+1)/float(warmup_steps)
        progress=(step-warmup_steps)/max(1,total_steps-warmup_steps)
        return 0.5*(1.0+np.cos(np.pi*min(1.0,max(0.0,progress))))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, scale)
