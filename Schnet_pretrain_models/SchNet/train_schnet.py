import os
import sys
import argparse
import copy
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from pretrain_models.SchNet.qm9_dataset import get_qm9_dataloaders, QM9_PROPERTIES
from src.modules.geom import SchNetEncoder


class SchNetPropertyPredictor(nn.Module):
    """
    SchNet 属性预测模型
    在 SchNetEncoder 基础上添加回归头
    """
    
    def __init__(self, args):
        """
        Args:
            args: 命令行参数
        """
        super().__init__()
        
        # 创建 SchNetEncoder（不加载预训练权重）
        self.encoder = SchNetEncoder(
            hidden_channels=128,
            num_filters=128,
            num_interactions=6,
            num_gaussians=50,
            cutoff=10.0,
            max_num_neighbors=32,
            readout='mean',
            load_from_pretrain=None  # 从头训练
        )
        
        # 添加回归头（用于属性预测任务）
        self.regression_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.Softplus(),
            nn.Linear(64, 1)
        )
    
    def forward(self, z, pos, batch=None):
        """
        前向传播
        
        Args:
            z: 原子序数 [num_atoms]
            pos: 3D 坐标 [num_atoms, 3]
            batch: 批次索引 [num_atoms]
            
        Returns:
            prediction: 属性预测值 [num_graphs, 1]
            embedding: 图级嵌入 [num_graphs, hidden_channels]
        """
        # 获取图级嵌入
        embedding = self.encoder(z, pos, batch)
        
        # 预测属性
        prediction = self.regression_head(embedding)
        
        return prediction, embedding
    
    def get_encoder_state_dict(self):
        """
        获取编码器的 state_dict（用于 Uni-Poly 加载）
        只返回 SchNetEncoder 部分的权重
        """
        return self.encoder.state_dict()


def get_target_normalization(train_loader):
    """Compute target normalization from the training split only."""
    targets = torch.cat([data.y.view(-1) for data in train_loader.dataset])
    mean = targets.mean().item()
    std = targets.std().item()
    if std < 1e-12:
        std = 1.0
    return mean, std


def train_epoch(model, train_loader, criterion, optimizer, device, target_mean, target_std, use_tqdm=True):
    """训练一个 epoch"""
    model.train()
    losses = []
    preds = []
    targets = []
    
    pbar = tqdm(train_loader, desc="Training", disable=not use_tqdm)
    for batch in pbar:
        batch = batch.to(device)
        optimizer.zero_grad()
        
        predictions, _ = model(batch.z, batch.pos, batch.batch)
        target_values = batch.y.view_as(predictions)
        targets_normalized = (target_values - target_mean) / target_std
        loss = criterion(predictions, targets_normalized)
        
        loss.backward()
        
        if args.clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
        
        optimizer.step()
        
        losses.append(loss.item())
        predictions_raw = predictions.detach() * target_std + target_mean
        preds.extend(predictions_raw.view(-1).cpu().numpy())
        targets.extend(target_values.detach().view(-1).cpu().numpy())
    
    avg_loss = np.mean(losses)
    r2 = r2_score(targets, preds)
    
    return avg_loss, r2


@torch.no_grad()
def evaluate(model, data_loader, criterion, device, target_mean, target_std, use_tqdm=True):
    """评估模型"""
    model.eval()
    losses = []
    preds = []
    targets = []
    
    pbar = tqdm(data_loader, desc="Evaluating", disable=not use_tqdm)
    for batch in pbar:
        batch = batch.to(device)
        
        predictions, _ = model(batch.z, batch.pos, batch.batch)
        targets_batch = batch.y.view_as(predictions)
        targets_normalized = (targets_batch - target_mean) / target_std
        loss = criterion(predictions, targets_normalized)
        
        losses.append(loss.item())
        predictions_raw = predictions * target_std + target_mean
        preds.extend(predictions_raw.view(-1).cpu().numpy())
        targets.extend(targets_batch.view(-1).cpu().numpy())
    
    avg_loss = np.mean(losses)
    r2 = r2_score(targets, preds)
    
    return avg_loss, r2, targets, preds


@torch.no_grad()
def test_model(model, test_loader, device, target_mean, target_std):
    """测试模型"""
    model.eval()
    preds = []
    targets = []
    
    pbar = tqdm(test_loader, desc="Testing", disable=False)
    for batch in pbar:
        batch = batch.to(device)
        predictions, _ = model(batch.z, batch.pos, batch.batch)
        predictions_raw = predictions * target_std + target_mean
        targets_batch = batch.y.view_as(predictions)
        
        preds.extend(predictions_raw.view(-1).cpu().numpy())
        targets.extend(targets_batch.view(-1).cpu().numpy())
    
    preds = np.array(preds)
    targets = np.array(targets)
    
    metrics = {
        'r2': r2_score(targets, preds),
        'mae': mean_absolute_error(targets, preds),
        'rmse': np.sqrt(mean_squared_error(targets, preds)),
    }
    
    return metrics


def plot_training_curves(metrics, save_path):
    """绘制训练曲线"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    axes[0].plot(metrics['train_losses'], label='Train Loss')
    axes[0].plot(metrics['val_losses'], label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    axes[1].plot(metrics['train_r2s'], label='Train R²')
    axes[1].plot(metrics['val_r2s'], label='Val R²')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('R² Score')
    axes[1].set_title('Training and Validation R²')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Training curves saved to {save_path}")


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Train SchNet on QM9 dataset')
    parser.add_argument('--property', type=str, default='homo',
                       choices=list(QM9_PROPERTIES.keys()),
                       help='QM9 property to predict')
    parser.add_argument('--epochs', type=int, default=1000,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--patience', type=int, default=100,
                       help='Early stopping patience')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['cuda', 'cpu', 'auto'],
                       help='Device to use')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--save_dir', type=str, default='./pretrained_models/encoders',
                       help='Directory to save models')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--scheduler', type=str, default='cosine',
                       choices=['cosine', 'step', 'reduce_on_plateau'],
                       help='Learning rate scheduler')
    parser.add_argument('--optimizer', type=str, default='adam',
                       choices=['adam', 'adamw', 'sgd'],
                       help='Optimizer')
    parser.add_argument('--loss', type=str, default='mse',
                       choices=['mse', 'mae', 'huber'],
                       help='Loss function')
    parser.add_argument('--clip_grad_norm', type=float, default=None,
                       help='Gradient clipping')
    parser.add_argument('--checkpoint_interval', type=int, default=10,
                       help='Checkpoint save interval')
    
    global args
    args = parser.parse_args()
    
    # 设置设备
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"SchNet Pretraining on QM9 - {args.property.upper()}")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Property: {QM9_PROPERTIES[args.property]['name']} ({args.property})")
    print(f"Model: hidden_channels=128, interactions=6")
    print(f"Training: epochs={args.epochs}, batch_size={args.batch_size}, lr={args.lr}")
    print(f"Output: {os.path.join(args.save_dir, f'schnet_qm9_{args.property}.pth')}")
    print(f"{'='*60}\n")
    
    # 加载数据
    print("Loading QM9 dataset...")
    train_loader, val_loader, test_loader, dataset = get_qm9_dataloaders(
        root='./pretrain_models/data/QM9',
        property_name=args.property,
        batch_size=args.batch_size,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        num_workers=args.num_workers,
        shuffle=True
    )
    
    # 打印统计信息
    stats = dataset.get_statistics()
    print(f"\nDataset statistics:")
    print(f"  Mean: {stats['mean']:.4f}, Std: {stats['std']:.4f}")
    print(f"  Min: {stats['min']:.4f}, Max: {stats['max']:.4f}")
    target_mean, target_std = get_target_normalization(train_loader)
    print(f"  Train target normalization: mean={target_mean:.4f}, std={target_std:.4f}")
    
    # 创建模型
    print("\nCreating SchNet model...")
    model = SchNetPropertyPredictor(args)
    model = model.to(device)
    
    # 创建优化器
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    elif args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    
    # 创建学习率调度器
    if args.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif args.scheduler == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    elif args.scheduler == 'reduce_on_plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # 创建损失函数
    if args.loss == 'mse':
        criterion = nn.MSELoss()
    elif args.loss == 'mae':
        criterion = nn.L1Loss()
    elif args.loss == 'huber':
        criterion = nn.HuberLoss(delta=1.0)
    
    # 训练循环
    print(f"\n{'='*60}")
    print("Starting training...")
    print(f"{'='*60}\n")
    
    best_val_r2 = -float('inf')
    best_model_state = None
    epochs_no_improve = 0
    
    metrics_history = {
        'train_losses': [],
        'val_losses': [],
        'train_r2s': [],
        'val_r2s': []
    }
    
    for epoch in range(args.epochs):
        # 训练
        train_loss, train_r2 = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            target_mean,
            target_std,
        )
        
        # 验证
        val_loss, val_r2, _, _ = evaluate(
            model,
            val_loader,
            criterion,
            device,
            target_mean,
            target_std,
        )

        if args.scheduler == 'reduce_on_plateau':
            scheduler.step(val_loss)
        else:
            scheduler.step()
        
        # 记录历史
        metrics_history['train_losses'].append(train_loss)
        metrics_history['val_losses'].append(val_loss)
        metrics_history['train_r2s'].append(train_r2)
        metrics_history['val_r2s'].append(val_r2)
        
        # 打印进度
        print(f"Epoch {epoch+1:3d}/{args.epochs} | "
              f"Train Loss: {train_loss:.6f} | Train R²: {train_r2:.4f} | "
              f"Val Loss: {val_loss:.6f} | Val R²: {val_r2:.4f}")
        
        # 早停检查
        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            best_model_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
            print(f"  ✓ New best model! (Val R²: {val_r2:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break
    
    # 加载最佳模型
    print(f"\n{'='*60}")
    print("Loading best model...")
    print(f"{'='*60}")
    model.load_state_dict(best_model_state)
    
    # 测试
    print("\nEvaluating on test set...")
    test_metrics = test_model(model, test_loader, device, target_mean, target_std)
    print(f"\nTest Results:")
    print(f"  R²:   {test_metrics['r2']:.4f}")
    print(f"  MAE:  {test_metrics['mae']:.4f}")
    print(f"  RMSE: {test_metrics['rmse']:.4f}")
    
    # 保存模型
    print(f"\n{'='*60}")
    print("Saving model...")
    print(f"{'='*60}")
    
    model_save_path = os.path.join(args.save_dir, f'schnet_qm9_{args.property}.pth')
    torch.save(model.get_encoder_state_dict(), model_save_path)
    print(f"✓ Encoder weights saved to: {model_save_path}")
    
    # 保存完整模型
    # full_model_path = os.path.join(args.save_dir, f'schnet_qm9_{args.property}_full.pth')
    # torch.save(model.state_dict(), full_model_path)
    # print(f"✓ Full model saved to: {full_model_path}")
    
    # 保存训练历史
    # metrics_save_path = os.path.join(args.save_dir, f'schnet_training_metrics_{args.property}.pt')
    # torch.save(metrics_history, metrics_save_path)
    # print(f"✓ Training metrics saved to: {metrics_save_path}")
    
    # 绘制训练曲线
    curve_save_path = os.path.join(args.save_dir, f'schnet_qm9_{args.property}_curves.png')
    plot_training_curves(metrics_history, curve_save_path)
    
    # 保存日志
    log_path = os.path.join(args.save_dir, f'schnet_qm9_{args.property}_log.txt')
    with open(log_path, 'w') as f:
        f.write(f"SchNet Pretraining Log - {args.property}\n")
        f.write(f"{'='*60}\n\n")
        f.write(f"Configuration:\n")
        f.write(f"  Property: {args.property}\n")
        f.write(f"  Epochs: {args.epochs}\n")
        f.write(f"  Batch size: {args.batch_size}\n")
        f.write(f"  Learning rate: {args.lr}\n")
        f.write(f"  Target mean: {target_mean:.6f}\n")
        f.write(f"  Target std: {target_std:.6f}\n")
        f.write(f"  Best Val R²: {best_val_r2:.4f}\n\n")
        f.write(f"Test Results:\n")
        f.write(f"  R²:   {test_metrics['r2']:.4f}\n")
        f.write(f"  MAE:  {test_metrics['mae']:.4f}\n")
        f.write(f"  RMSE: {test_metrics['rmse']:.4f}\n")
    print(f"✓ Training log saved to: {log_path}")
    
    print(f"\n{'='*60}")
    print("Training completed successfully!")
    print(f"{'='*60}")
    print(f"\nNext steps:")
    print(f"1. Update Uni-Poly config to use: {model_save_path}")
    print(f"2. In uni_encoder.py, set geom_model_name='{model_save_path}'")
    print(f"3. Run Uni-Poly training!")
    

if __name__ == '__main__':
    main()
