# Uni-Poly

Uni-Poly 是一个面向聚合物性质预测的多模态表征学习框架。项目将 SMILES、二维分子图、三维几何结构、分子指纹以及领域知识等信息统一到同一个聚合物表示中，用于提升不同聚合物性质任务上的预测效果。

## 项目简介

传统聚合物性质预测方法通常依赖单一表征，难以同时覆盖聚合物的序列、拓扑、空间结构和领域语义信息。Uni-Poly 通过多模态编码器和注意力融合机制，将以下信息组合起来：

- **SMILES 序列**：使用预训练 SMILES 编码器提取序列表示。
- **二维分子图**：使用图结构描述原子与化学键关系。
- **三维几何结构**：支持 `painn` 和 `schnet` 几何编码后端。
- **分子指纹**：引入传统化学指纹特征。
- **知识图谱/领域知识**：通过 `knowledge_graph/` 下的映射和嵌入文件增强聚合物语义信息。

当前仓库包含预训练、下游训练、注意力权重可视化、caption 生成示例以及部分已处理数据和模型权重。

## 目录结构

```text
Uni-Poly/
├── README.MD                         # 项目说明
├── requirements.txt                  # Python 依赖
├── LICENSE                           # Apache 2.0 许可证
├── data/
│   ├── raw/                          # 原始 CSV 数据，文件名形如 smi_tg.csv
│   ├── SchNet/                       # SchNet 相关中间数据
│   └── smiles_text_dict.json         # SMILES 到文本描述的映射
├── src/
│   ├── dataset/                      # 数据集、图数据、几何数据与 DataLoader
│   ├── modules/                      # 多模态编码器、图/几何/知识模块
│   ├── kg/                           # 知识图谱加载与映射工具
│   └── utils.py                      # 训练、评估、损失与通用工具函数
├── scripts/
│   ├── pretrain.py                   # 多模态对比预训练入口
│   ├── train.py                      # 下游性质预测训练入口
│   ├── run_pretrain.sh               # 预训练示例脚本
│   ├── run_train.sh                  # 下游训练示例脚本
│   └── plot_attention_heatmap.py     # 注意力权重热力图绘制脚本
├── pretrained_models/
│   ├── encoders/                     # 预训练编码器权重或 Hugging Face 模型目录
│   └── saved_pretrained_model.pth    # Uni-Poly 预训练权重
├── saved_models/                     # 各下游任务保存的最优模型
├── results/
│   └── results.csv                   # 训练评估结果
├── plots/
│   └── pretrain/                     # 预训练 loss 数据与曲线
├── knowledge_graph/                  # 知识图谱、本体、映射和嵌入文件
└── caption_generation/               # 聚合物文本描述生成 notebook 与示例输入
```

## 环境要求

- Python 3.8 或更高版本
- 推荐使用支持 CUDA 的 GPU
- PyTorch、PyTorch Geometric、RDKit、Transformers 等依赖

安装依赖：

```bash
pip install -r requirements.txt
```

如果安装 `torch-geometric`、`torch-scatter`、`torch-sparse` 或 `rdkit` 时遇到版本问题，请根据本机 CUDA、PyTorch 和 Python 版本安装匹配的 wheel 或 conda 包。

## 数据准备

### 1. 性质预测数据

下游任务数据放在 `data/raw/` 目录中，文件名约定为：

```text
smi_<task>.csv
```

例如：

- `smi_tg.csv`：玻璃化转变温度任务
- `smi_eat.csv`
- `smi_eea.csv`
- `smi_egb.csv`
- `smi_egc.csv`
- `smi_ei.csv`
- `smi_eps.csv`
- `smi_nc.csv`
- `smi_xc.csv`

训练脚本会根据任务名自动读取 `data/raw/smi_<task>.csv`。CSV 至少需要包含 SMILES 和对应性质标签，具体格式可参考仓库中已有的 `data/raw/smi_tg.csv`。

### 2. 预训练编码器与模型权重

默认配置会读取以下路径中的模型或权重：

```text
pretrained_models/encoders/PubChem10M_SMILES_BPE_450k
pretrained_models/encoders/multitask-text-and-chemistry-t5-base-augm
pretrained_models/saved_pretrained_model.pth
```

如果使用 SchNet 几何编码器，还需要确保相关 SchNet 权重和数据位于项目中对应目录，例如：

```text
pretrained_models/encoders/schnet_qm9_gap.pth
pretrained_models/encoders/schnet_qm9_cv.pth
data/Schnet/
```

### 3. 知识图谱数据

启用 `kg` 模态时，程序会使用 `knowledge_graph/` 下的映射、功能团、元素、本体和嵌入文件。当前示例脚本默认启用 `kg`：

```bash
--modalities smiles graph fp geom kg
```

## 使用方法

### 生成聚合物文本描述

caption 生成示例位于 `caption_generation/`：

```text
caption_generation/Poly-caption.ipynb
caption_generation/json_input.json
```

在 Jupyter Notebook 中打开并运行 `Poly-caption.ipynb`，输入 JSON 格式可参考 `json_input.json`。

### 多模态预训练

使用默认示例脚本：

```bash
bash scripts/run_pretrain.sh
```

该脚本默认执行：

```bash
python scripts/pretrain.py \
  --dataset_name smi_all \
  --modalities smiles graph fp geom kg \
  --geometry_encoder schnet
```

预训练输出包括：

- 模型权重：`pretrained_models/saved_pretrained_model.pth`
- loss 数据：`plots/pretrain/loss_data.json`
- loss 曲线：`plots/pretrain/loss_curve.png`
- 日志文件：`logs/pretrain.log`

可用参数包括：

- `--dataset_name`：预训练数据集名称，默认 `smi_all`
- `--modalities`：使用的模态列表
- `--geometry_encoder`：几何编码器，可选 `painn` 或 `schnet`
- `--batch_size`：批大小，默认 `32`
- `--epochs`：预训练轮数，默认 `20`
- `--lr`：学习率，默认 `1e-4`
- `--temperature`：对比学习温度系数，默认 `0.07`
- `--save_path`：预训练权重保存路径

### 下游性质预测训练

使用默认示例脚本：

```bash
bash scripts/run_train.sh
```

该脚本默认执行 9 个任务的训练：

```bash
python scripts/train.py \
  --modalities smiles graph fp geom kg \
  --geometry_encoder schnet \
  --tasks tg eat eea egb egc ei eps nc xc \
  --pretrained_model_path ./pretrained_models/saved_pretrained_model.pth
```

训练脚本使用 5 折交叉验证，并将结果保存到：

```text
results/results.csv
saved_models/<task>/UniEncoderAttention_best.pth
logs/train.log
```

常用参数包括：

- `--tasks`：需要训练的任务列表
- `--modalities`：使用的模态列表，例如 `smiles graph fp geom kg`
- `--geometry_encoder`：几何编码器，可选 `painn` 或 `schnet`
- `--freeze_encoder`：是否冻结编码器
- `--pretrained_model_path`：预训练 Uni-Poly 权重路径
- `--epochs`：训练轮数，默认 `100`
- `--patience`：早停耐心值，默认 `10`
- `--batch_size`：批大小，默认 `32`
- `--results_dir`：结果 CSV 保存路径
- `--models_dir`：模型保存目录

### 绘制注意力权重热力图

训练完成后，如果 `results/results.csv` 中包含 attention 相关列，可以运行：

```bash
python scripts/plot_attention_heatmap.py \
  --results_csv ./results/results.csv \
  --output ./results/attention_heatmap.png
```

输出文件：

```text
results/attention_heatmap.png
```

## 结果文件说明

- `results/results.csv`：下游任务的评估指标、模态组合和注意力权重等结果。
- `saved_models/`：每个任务对应的最优模型权重。
- `plots/pretrain/loss_curve.png`：预训练对比损失曲线。
- `logs/pretrain.log`：预训练日志。
- `logs/train.log`：下游训练日志。

## 注意事项

- `scripts/run_pretrain.sh` 和 `scripts/run_train.sh` 中默认设置了 `CUDA_VISIBLE_DEVICES=1`。如果机器上 GPU 编号不同，请按需修改。
- 示例脚本使用 `nohup ... &` 在后台运行任务，日志会写入 `logs/` 目录。
- 如果禁用知识图谱模态，请从 `--modalities` 中移除 `kg`。
- 如果使用 CPU 运行，训练速度会明显下降。

## 许可证

本项目使用 Apache License 2.0。详情请查看 [LICENSE](LICENSE)。

## 联系方式

如有问题或反馈，请联系：

**QI HUANG**  
Email: zzz1159600zz@gmail.com
