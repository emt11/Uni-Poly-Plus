# SchNet 预训练指南

本模块用于在 **QM9 数据集**上预训练 SchNet 模型，训练完成的权重可直接用于 **Uni-Poly** 项目的 `geom` 模态。

---

## 📁 文件结构

```
src/SchNet/
├── __init__.py              # 包初始化
├── qm9_dataset.py           # QM9 数据集加载器
├── config.py                # 训练配置
├── train_schnet.py          # 训练脚本
├── run_train.sh             # Linux/Mac 运行脚本
├── run_train.bat            # Windows 运行脚本
└── README.md                # 使用说明（本文件）
```

---

## 🚀 快速开始

### 方法 1：使用运行脚本（推荐）

**Windows:**
```bash
# 训练 HOMO 属性（默认）
src\SchNet\run_train.bat

# 训练其他属性
src\SchNet\run_train.bat homo
src\SchNet\run_train.bat lumo
src\SchNet\run_train.bat gap
src\SchNet\run_train.bat u0

# 自定义参数
src\SchNet\run_train.bat homo 150 64 1e-3 15
# 参数顺序: property epochs batch_size learning_rate patience
```

**Linux/Mac:**
```bash
chmod +x src/SchNet/run_train.sh

# 训练 HOMO 属性
bash src/SchNet/run_train.sh

# 自定义参数
bash src/SchNet/run_train.sh homo 150 64 1e-3 15
```

### 方法 2：直接使用 Python

```bash
# 训练 HOMO 属性
python src/SchNet/train_schnet.py --property homo

# 训练 LUMO 属性
python src/SchNet/train_schnet.py --property lumo --epochs 150

# 完整参数示例
python src/SchNet/train_schnet.py \
    --property gap \
    --epochs 200 \
    --batch_size 64 \
    --lr 1e-3 \
    --patience 15 \
    --device cuda \
    --num_workers 4 \
    --save_dir ./pretrained_models/encoders \
    --seed 42
```

---

## 📊 支持的 QM9 属性

QM9 数据集包含 **12 个分子属性**，可任意选择：

| 属性名 | 说明 | 单位 | 推荐用于预训练 |
|--------|------|------|----------------|
| `dipole` | 偶极矩 | D | ✅ |
| `alpha` | 极化率 | a_0^3 | ✅ |
| `homo` | 最高占据分子轨道 | eV | ⭐⭐⭐ |
| `lumo` | 最低未占据分子轨道 | eV | ⭐⭐⭐ |
| `gap` | HOMO-LUMO 能隙 | eV | ⭐⭐⭐ |
| `r2` | 电子空间范围 | a_0^2 | ✅ |
| `zpve` | 零点振动能 | eV | ✅ |
| `u0` | 0K 内能 | eV | ⭐⭐ |
| `u298` | 298K 内能 | eV | ⭐⭐ |
| `h298` | 298K 焓 | eV | ✅ |
| `g298` | 298K 自由能 | eV | ✅ |
| `cv` | 298K 热容 | cal/mol/K | ✅ |

**推荐**: `homo`, `lumo`, `gap` 是最常用的预训练任务。

---

## ⚙️ 配置参数

### 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--property` | `homo` | 预测的 QM9 属性 |
| `--epochs` | `100` | 训练轮数 |
| `--batch_size` | `32` | 批次大小 |
| `--lr` | `1e-4` | 学习率 |
| `--patience` | `10` | 早停耐心值 |
| `--device` | `auto` | 设备 (`cuda`, `cpu`, `auto`) |
| `--num_workers` | `4` | 数据加载线程数 |
| `--save_dir` | `./pretrained_models/encoders` | 模型保存目录 |
| `--seed` | `42` | 随机种子 |

### 模型架构参数（固定，不可修改）

这些参数**必须与 Uni-Poly 项目保持一致**：

```python
hidden_channels = 128      # 隐藏层维度
num_filters = 128          # 滤波器维度
num_interactions = 6       # 交互层数
num_gaussians = 50         # 高斯核数量
cutoff = 10.0              # 距离截断值
max_num_neighbors = 32     # 最大邻居数
readout = 'mean'           # 池化方式
```

⚠️ **不要修改这些参数**，否则权重无法在 Uni-Poly 中加载！

---

## 📤 输出文件

训练完成后会生成以下文件：

```
pretrained_models/encoders/
├── schnet_qm9_homo.pth              # 编码器权重（用于 Uni-Poly）⭐
├── schnet_qm9_homo_full.pth         # 完整模型（包含回归头）
├── schnet_qm9_homo_curves.png       # 训练曲线图
├── schnet_qm9_homo_metrics.pt       # 训练指标数据
└── schnet_qm9_homo_log.txt          # 训练日志
```

---

## 🔗 在 Uni-Poly 中使用预训练权重

### 步骤 1：训练 SchNet

```bash
python src/SchNet/train_schnet.py --property homo
```

### 步骤 2：更新 Uni-Poly 配置

在 `scripts/train.py` 或 `scripts/pretrain.py` 中修改：

```python
pre_trained_model_dict = {
    'smiles_model_name': "./pretrained_models/encoders/PubChem10M_SMILES_BPE_450k",
    'text_model_name': "./pretrained_models/encoders/multitask-text-and-chemistry-t5-base-augm",
    'gnn_model_name': "./pretrained_models/encoders/Mole-BERT.pth",
    'geom_model_name': "./pretrained_models/encoders/schnet_qm9_homo.pth"  # ← 修改这里
}
```

### 步骤 3：运行 Uni-Poly 训练

```bash
bash scripts/run_train.sh
```

---

## 💡 训练建议

### 快速测试
```bash
python src/SchNet/train_schnet.py --property homo --epochs 10 --batch_size 64
```

### 标准预训练
```bash
python src/SchNet/train_schnet.py --property homo --epochs 100 --batch_size 32 --patience 10
```

### 高质量预训练
```bash
python src/SchNet/train_schnet.py --property homo --epochs 200 --batch_size 64 --lr 1e-3 --patience 15
```

---

## 📈 预期性能

在 QM9 数据集上训练 100 个 epoch 的预期指标：

| 属性 | R² | MAE | RMSE |
|------|-----|-----|------|
| `homo` | ~0.96 | ~0.03 eV | ~0.05 eV |
| `lumo` | ~0.95 | ~0.04 eV | ~0.06 eV |
| `gap` | ~0.94 | ~0.05 eV | ~0.07 eV |

*实际性能可能因硬件、随机种子等因素略有差异。*

---

## 🐛 常见问题

### Q1: 数据集下载失败
**A**: QM9 数据集会自动从 PyG 下载。如果网络问题导致失败，可手动下载：
```python
from torch_geometric.datasets import QM9
dataset = QM9(root='./data/QM9', download=True)
```

### Q2: CUDA Out of Memory
**A**: 减小 batch_size：
```bash
python src/SchNet/train_schnet.py --property homo --batch_size 16
```

### Q3: 权重加载报错
**A**: 确保模型参数与 Uni-Poly 一致，检查 `config.py` 中的：
- `hidden_channels = 128`
- `num_interactions = 6`
- `num_gaussians = 50`

### Q4: 训练速度慢
**A**: 
- 增加 `num_workers`（Linux: 4-8, Windows: 0）
- 使用更大的 batch_size（如果显存允许）
- 确保使用 GPU：`--device cuda`

---

## 📚 参考资料

- [PyG SchNet 文档](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.models.SchNet)
- [QM9 数据集论文](https://arxiv.org/abs/1404.1020)
- [SchNet 原始论文](https://arxiv.org/abs/1706.08566)
- [Uni-Poly 项目](../README.MD)

---

## 📝 许可

本模块遵循 Uni-Poly 项目的许可协议。
