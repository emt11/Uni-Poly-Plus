import copy
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import sklearn.metrics as metrics
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from src.dataset.dataloader import custom_collate


class TargetScaler:
    def __init__(self, task, scaler, eps=1e-4):
        self.task = task
        self.scaler = scaler
        self.eps = eps

    def _pre_transform(self, y):
        y = np.asarray(y, dtype=np.float64).reshape(-1, 1)
        if self.task in ['er', 'iv']:
            return np.log10(y)
        if self.task == 'xc':
            # Xc is a bounded percentage target. A clipped logit maps it to an
            # unbounded space before standardization and avoids 0/100 singularities.
            p = np.clip(y / 100.0, self.eps, 1.0 - self.eps)
            return np.log(p / (1.0 - p))
        return y

    def _inverse_pre_transform(self, y):
        y = np.asarray(y, dtype=np.float64).reshape(-1, 1)
        if self.task == 'xc':
            return 100.0 / (1.0 + np.exp(-y))
        return y

    def transform(self, y):
        return self.scaler.transform(self._pre_transform(y))

    def inverse_transform(self, y):
        y = np.asarray(y, dtype=np.float64).reshape(-1, 1)
        return self._inverse_pre_transform(self.scaler.inverse_transform(y))


def scale_targets(dataset, task, train_indices=None, raw_targets=None):
    scaler = StandardScaler()
    if raw_targets is None:
        raw_targets = np.array([data.y.item() for data in dataset], dtype=np.float64)
    else:
        raw_targets = np.asarray(raw_targets, dtype=np.float64)

    if train_indices is None:
        train_indices = np.arange(len(raw_targets))

    target_scaler = TargetScaler(task, scaler)
    train_values = raw_targets[train_indices].reshape(-1, 1)
    scaler.fit(target_scaler._pre_transform(train_values))

    print("Scaling y values with mean:", scaler.mean_[0], "and std:", scaler.scale_[0])
    if task == 'xc':
        print("Using clipped logit transform for bounded percentage target: xc")

    scaled_targets = target_scaler.transform(raw_targets.reshape(-1, 1))
    for data, value in zip(dataset, scaled_targets):
        data.y = torch.tensor(value, dtype=torch.float)

    return target_scaler

def get_data_loader(dataset, indices=None, batch_size=32, shuffle=False, drop_last=False):
    if indices is None:
        indices = range(len(dataset))
    subset_dataset = [dataset[i] for i in indices]

    loader = DataLoader(
        subset_dataset,
        batch_size=batch_size,
        collate_fn=custom_collate,
        shuffle=shuffle,
        drop_last=drop_last
    )

    print(f"Created dataloader with {len(subset_dataset)} samples")
    return loader

def train_epoch(model, train_loader, criterion, optimizer, scheduler, device, epoch=None, max_grad_norm=1.0):
    model.train()
    train_losses = []
    train_preds = []
    train_targets = []

    progress_bar = tqdm(train_loader, desc="Training")
    for batch in progress_bar:
        batch = batch.to(device)
        optimizer.zero_grad()

        outputs,_ = model(batch)
        loss = criterion(outputs, batch.y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
        optimizer.step()
        scheduler.step()
        train_losses.append(loss.item())
        train_preds.extend(outputs.detach().cpu().numpy())
        train_targets.extend(batch.y.detach().cpu().numpy())

    avg_train_loss = sum(train_losses) / len(train_losses)
    train_r2 = r2_score(train_targets, train_preds)
    
    return avg_train_loss, train_r2

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

def train_and_evaluate(
    model,
    scaler,
    train_loader,
    val_loader,
    test_loader,
    device,
    num_epochs=100,
    patience=5,
    max_grad_norm=1.0
):
    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    total_steps = max(1, len(train_loader) * num_epochs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    best_val_r2 = -float('inf')
    best_model_state = None
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        # Training phase
        avg_train_loss, train_r2 = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scheduler,
            device,
            epoch=epoch + 1,
            max_grad_norm=max_grad_norm
        )

        # Validation phase
        avg_val_loss, val_r2, _, _ = evaluate(model, val_loader, criterion, device)
        # Early stopping
        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
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
        print(f"Validation Loss: {avg_val_loss:.4f}, Validation R2: {val_r2:.4f}")
        print("-" * 50)

    # Load best model
    model.load_state_dict(best_model_state)

    # Testing phase
    test_metrics = test_model(model, test_loader, scaler, device)
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
