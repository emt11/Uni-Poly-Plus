# RUN.MD

本文是项目唯一运行手册。默认工作目录：

```bash
cd /root/workspace/DeepLearning/Uni-Poly-Plus-master
```

## 1. 安装依赖

```bash
pip install -r requirements.txt
```

## 2. 支持模态和 graph 输入

训练和预训练入口当前支持：

```text
smiles graph fp geom
```

`geom` 支持：

```bash
--geometry_encoder painn
--geometry_encoder schnet
```

`graph` 后端固定为 GIN，支持两种输入：

```bash
--graph_input repeat_unit    # 默认，保持原始 repeat-unit graph
--graph_input dimer          # 构造二聚体 graph；失败时自动 fallback 到 repeat-unit graph
```

`dimer` 只对恰好有两个 `*` attachment points 且 RDKit sanitize 成功的 SMILES 生效。`*` 数量不是 2，例如 3 个 `*` 的支化/交联候选结构，会保守回退到原始 repeat-unit graph。

## 3. 推荐运行流程：两步

当前项目有两类训练入口：

```text
Stage 1: scripts/pretrain.py          # Uni-Poly 多模态对齐预训练
Stage 2: scripts/train.py             # 下游属性训练
```

### 3.1 Stage 1：Uni-Poly 多模态对齐预训练

```bash
python scripts/pretrain.py \
  --dataset_name smi_all \
  --modalities smiles graph fp geom \
  --geometry_encoder schnet \
  --graph_input dimer \
  --epochs 20 \
  --batch_size 32 \
  --save_path ./pretrained_models/saved_pretrained_model_dimer_all.pth \
  --joint_embedding_dim 256
```

### 3.2 Stage 2：下游属性训练

```bash
python scripts/train.py \
  --modalities smiles graph fp geom \
  --geometry_encoder schnet \
  --graph_input dimer \
  --tasks tg eat eea egb egc ei eps nc xc \
  --pretrained_model_path ./pretrained_models/saved_pretrained_model_dimer_all.pth \
  --epochs 100 \
  --patience 10 \
  --results_dir ./results/results_dimer.csv \
  --models_dir ./saved_models_dimer \
  --joint_embedding_dim 256
```

只训练单个任务，例如 `ei`：

```bash
python scripts/train.py \
  --modalities smiles graph fp geom \
  --geometry_encoder schnet \
  --graph_input dimer \
  --tasks ei \
  --pretrained_model_path ./pretrained_models/saved_pretrained_model_dimer_all.pth \
  --results_dir ./results/results_dimer_ei.csv \
  --models_dir ./saved_models_dimer_ei
```

## 4. Dimer Graph 路线

Dimer graph 使用 GIN 后端。串行运行完整 Dimer 路线：

```bash
bash scripts/run.sh
```

总日志：

```text
logs/run_dimer.log
```

### 4.1 Dimer Graph 多模态对齐预训练

```bash
python scripts/pretrain.py \
  --dataset_name smi_all \
  --modalities smiles graph fp geom \
  --geometry_encoder schnet \
  --geom_model_name ./pretrained_models/encoders/schnet_qm9_gap.pth \
  --graph_input dimer \
  --epochs 20 \
  --batch_size 32 \
  --save_path ./pretrained_models/saved_pretrained_model_dimer_all.pth \
  --joint_embedding_dim 256
```

### 4.2 Dimer Graph 下游训练

```bash
python scripts/train.py \
  --modalities smiles graph fp geom \
  --geometry_encoder schnet \
  --geom_model_name ./pretrained_models/encoders/schnet_qm9_gap.pth \
  --graph_input dimer \
  --tasks eat eea egb egc ei eps nc tg xc \
  --pretrained_model_path ./pretrained_models/saved_pretrained_model_dimer_all.pth \
  --epochs 100 \
  --patience 10 \
  --results_dir ./results/results_dimer_all.csv \
  --models_dir ./saved_models_dimer \
  --joint_embedding_dim 256
```

## 5. 补充示例：预训练 baseline

默认使用 `data/raw/smi_all.csv`：

```bash
python scripts/pretrain.py \
  --dataset_name smi_all \
  --modalities smiles graph fp geom \
  --geometry_encoder schnet \
  --graph_input repeat_unit \
  --joint_embedding_dim 256
```

## 6. 补充示例：预训练 Dimer Graph

```bash
python scripts/pretrain.py \
  --dataset_name smi_all \
  --modalities smiles graph fp geom \
  --geometry_encoder schnet \
  --graph_input dimer \
  --joint_embedding_dim 256
```

Dimer graph 会写入独立缓存，例如 `data/Schnet/smi_all_gin-dimer.pt`，不会覆盖原始 `gin` 缓存。

## 7. 补充示例：下游训练 baseline

```bash
python scripts/train.py \
  --modalities smiles graph fp geom \
  --geometry_encoder schnet \
  --graph_input repeat_unit \
  --tasks tg eat eea egb egc ei eps nc xc \
  --pretrained_model_path ./pretrained_models/saved_pretrained_model.pth \
  --joint_embedding_dim 256
```

## 8. 补充示例：下游训练 Dimer Graph

```bash
python scripts/train.py \
  --modalities smiles graph fp geom \
  --geometry_encoder schnet \
  --graph_input dimer \
  --tasks tg eat eea egb egc ei eps nc xc \
  --pretrained_model_path ./pretrained_models/saved_pretrained_model.pth \
  --joint_embedding_dim 256
```

## 9. Shell 包装脚本

预训练：

```bash
bash scripts/run_pretrain.sh
```

下游训练：

```bash
bash scripts/run_train.sh
```

如需 Dimer graph，直接在命令中加入 `--graph_input dimer`，或使用 `scripts/run.sh` 直接运行完整 dimer 路线。

## 10. 常见输出

```text
pretrained_models/saved_pretrained_model.pth
saved_models/<task>/UniEncoderAttention_best.pth
results/results.csv
plots/pretrain/loss_curve.png
logs/
```

## 11. 缓存说明

Dataset cache 区分 graph 输入类型，例如：

```text
data/Schnet/smi_all_gin.pt
data/Schnet/smi_all_gin-dimer.pt
```

常用参数以 help 为准：

```bash
python scripts/pretrain.py --help
python scripts/train.py --help
```
