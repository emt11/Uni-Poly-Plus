# Uni-Poly-Plus

Uni-Poly-Plus 是面向聚合物性质预测的多模态学习项目。当前主模型支持 SMILES、二维分子图、分子指纹和三维几何信息四种模态。

## 支持模态

训练入口支持：

```text
smiles graph fp geom
```

说明：

- `smiles`：SMILES 序列编码。
- `graph`：二维分子图编码，后端固定为 PyG GIN。支持两种输入：
  - `repeat_unit`（默认）：原始重复单元 graph。
  - `dimer`：二聚体局部链段图，失败时自动 fallback 到 repeat_unit graph。
- `fp`：Morgan 分子指纹特征。
- `geom`：三维几何编码，支持 `painn` / `schnet`。

## 快速开始

安装依赖：

```bash
pip install -r requirements.txt
```

预训练四模态 baseline：

```bash
python scripts/pretrain.py \
  --dataset_name smi_all \
  --modalities smiles graph fp geom \
  --geometry_encoder schnet
```

下游训练四模态 baseline：

```bash
python scripts/train.py \
  --modalities smiles graph fp geom \
  --geometry_encoder schnet \
  --tasks tg eat eea egb egc ei eps nc xc \
  --pretrained_model_path ./pretrained_models/saved_pretrained_model.pth
```

启用 Dimer graph 输入时添加：

```bash
--graph_input dimer
```

也可以直接运行完整 Dimer 路线：

```bash
bash scripts/run.sh
```

## 文档导航

- [RUN.MD](RUN.MD)：唯一运行手册，包含预训练、下游训练和 Dimer graph 示例。
- [AGENT.MD](AGENT.MD)：开发和代码代理约束。
- [TODO.MD](TODO.MD)：已完成的重构计划（移除 PerioGT/pyg_periogt，保留 GIN+Dimer 路线）。

## 数据入口

下游任务数据位于：

```text
data/raw/smi_<task>.csv
```

预训练默认数据位于：

```text
data/raw/smi_all.csv
```

## 常见输出

```text
pretrained_models/saved_pretrained_model.pth
saved_models/<task>/UniEncoderAttention_best.pth
results/results.csv
plots/pretrain/loss_curve.png
logs/
```

## 源码模块

```text
src/dataset/       数据读取、图/几何/指纹构建和 batch collate
src/modules/       SMILES、Graph、Geometry encoder 与多模态融合
scripts/           预训练、下游训练和分析脚本
```
