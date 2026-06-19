# Uni-Poly-Plus

Uni-Poly-Plus 是面向聚合物性质预测的多模态学习项目。当前主模型支持结构、图、指纹、几何信息，并可选接入新的 Polymer KG embedding 作为第五模态。

## 支持模态

当前训练入口支持：

```text
smiles graph fp geom kg
```

说明：

- `smiles`：SMILES 序列编码。
- `graph`：二维分子图编码。
- `fp`：分子指纹特征。
- `geom`：三维几何编码，支持 `painn` / `schnet`。
- `kg`：新的 Polymer KG embedding，可选启用。

新 `kg` 模态读取：

```text
kg_work/features/kg_entity_mapping.csv
kg_work/features/kg_embedding.npy
```

## 快速开始

安装依赖：

```bash
pip install -r requirements.txt
```

预训练四模态：

```bash
python scripts/pretrain.py   --dataset_name smi_all   --modalities smiles graph fp geom   --geometry_encoder schnet
```

下游训练四模态：

```bash
python scripts/train.py   --modalities smiles graph fp geom   --geometry_encoder schnet   --tasks tg eat eea egb egc ei eps nc xc   --pretrained_model_path ./pretrained_models/saved_pretrained_model.pth
```

生成 KG embedding 后启用五模态训练：

```bash
python scripts/train.py   --modalities smiles graph fp geom kg   --kg_mapping_path kg_work/features/kg_entity_mapping.csv   --kg_embedding_path kg_work/features/kg_embedding.npy   --kg_embedding_dim 128   --kg_projection_dim 256   --kg_freeze_embedding true
```

## 文档导航

- [RUN.MD](RUN.MD)：唯一运行手册。包含 KG pipeline、online/local/both 文献模式、KG embedding、pretrain/train with kg。
- [LLM_KG.MD](LLM_KG.MD)：KG 语义与开发设计文档。包含 schema、label leakage、证据链、节点/边设计和 TODO。
- [AGENT.MD](AGENT.MD)：开发和代码代理约束。包含修改代码时必须遵守的边界和验证要求。

## 数据入口

下游任务数据位于：

```text
data/raw/smi_<task>.csv
```

KG pipeline 默认使用：

```text
data/raw/smi_all.csv
```

`data/raw/smi_all.csv` 至少包含：

```text
smiles,val,prop
```

KG pipeline 不允许将 `val` 输入 LLM、检索 query、KG triples 或 embedding。

## 常见输出

训练与评估：

```text
pretrained_models/saved_pretrained_model.pth
saved_models/<task>/UniEncoderAttention_best.pth
results/results.csv
plots/pretrain/loss_curve.png
logs/
```

KG pipeline：

```text
kg_work/records.csv
kg_work/repeat_units.csv
kg_work/polymer_class_candidates.jsonl
kg_work/retrieval/
kg_work/chunks/
kg_work/extractions/
kg_work/links/dataset_links.jsonl
kg_work/kg/
kg_work/features/kg_entity_mapping.csv
kg_work/features/kg_embedding.npy
kg_work/pilot_report.json
```


## 源码模块

KG pipeline 的核心实现集中在：

```text
src/kg_pipeline/io_utils.py
src/kg_pipeline/dataset.py
src/kg_pipeline/literature.py
src/kg_pipeline/llm_extract.py
src/kg_pipeline/article_validation.py
src/kg_pipeline/kg_build.py
src/kg_pipeline/kg_embedding.py
```

`scripts/kg/*.py` 保持为轻量 CLI wrapper。旧的细粒度模块文件已删除，新增代码应写入上述核心模块。

## 模型目录

当前不重命名模型目录，因为训练脚本和默认参数已经引用这些路径：

```text
pretrained_models/   外部或已训练权重，例如 encoder 和 saved_pretrained_model.pth
pretrain_models/     预训练过程输出、缓存或中间产物
```
