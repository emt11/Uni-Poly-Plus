# PIPELINE.MD

本文档是当前 Uni-Poly-Plus 项目的整体流程维护文档。任何涉及数据流、缓存、Graph/Geom 构造、预训练、训练脚本、输出路径或模型模态的改动，都必须同步更新本文档。

## 1. 总体目标

当前项目用于聚合物性质预测，主线是四模态学习：

```text
SMILES + Graph + FP + Geom -> UniEncoderAttention -> property prediction
```

当前推荐路线：

```text
data/raw/smi_all.csv
-> SMILES-level feature cache
-> Graph / FP / Geom feature cache
-> Graph + Geom 单模态预训练
-> 多模态 alignment 预训练
-> data/raw/smi_<task>.csv 下游 5-fold supervised training
-> results / saved_models
```

推荐完整入口：

```bash
bash scripts/run.sh
```

推荐只跑预训练入口：

```bash
bash scripts/run_pretrain.sh
```

## 2. 输入数据

预训练数据：

```text
data/raw/smi_all.csv
```

下游任务数据：

```text
data/raw/smi_<task>.csv
```

当前下游任务集合：

```text
eat eea egb egc ei eps nc xc
```

当前 `data/raw/smi_all.csv` 是 no-`tg` 主线数据，由 8 个单任务 CSV 合并生成，不包含 `prop=tg`。`data/raw/smi_tg.csv` 保留但不参与当前预训练、缓存构建或下游训练。恢复 9-task 时必须重新生成包含 `tg` 的 `smi_all.csv`，并同步修改脚本 `--tasks`、结果文件名和模型目录。

数据约束：

- `smi_all.csv` 用于预训练和全局缓存构建。
- `smi_<task>.csv` 用于单任务 supervised training。
- 下游训练使用 5-fold cross validation。
- 当前训练逻辑中 `KFold(n_splits=5, shuffle=True, random_state=1)`，即随机打乱后划分 fold。

## 3. Feature Cache

当前项目使用 SMILES 级别 feature cache，避免预训练和每个下游任务重复处理同一个 SMILES。

缓存默认目录：

```text
data/processed/Schnet/
```

缓存包含：

- SMILES token。
- Graph data。
- Fingerprint。
- Geom data。
- 相关 metadata。

如果修改以下内容，必须重建缓存：

- `graph_input`
- Graph 节点特征或边特征维度
- star-linking 逻辑
- backbone / attachment 特征
- Geom 构象生成逻辑
- conformer 数量
- SMILES token 长度
- cache tag

重建参数：

```bash
--rebuild_feature_cache
```

SMILES token 长度：

- 默认由 `smi_all.csv` 统计得到。
- 当前有 `--max_smiles_length_cap` 上限，默认 `256`。
- 如果出现：

```text
[token] truncated x / n SMILES at max_length=256
```

表示有 `x` 条 SMILES 超过最大 token 长度并被截断。少量截断通常可以接受；大量截断时需要提高 `--max_smiles_length_cap`。

## 4. Graph 流程

Graph 输入当前只支持：

```text
repeat_unit
star_linking
```

正式推荐：

```bash
--graph_input star_linking
```

### 4.1 repeat_unit

保留原始 repeat-unit graph。

用途：

- baseline。
- star-linking 失败 fallback。

### 4.2 star_linking

star-linking 只用于 Graph 模态。

处理逻辑：

1. 解析原始 P-SMILES。
2. 要求恰好存在两个 `*` attachment atoms。
3. 每个 `*` 必须只有一个邻居。
4. 两个 `*` 与邻居之间的 bond type 必须一致。
5. 删除两个 `*`。
6. 在两个 attachment neighbor atoms 之间添加同类型 bond。
7. RDKit sanitize 成功后得到 star-linked graph。
8. 失败时 fallback 到 `repeat_unit` graph。

语义：

```text
Graph star-linking 是拓扑周期近似，使 GNN 消息可以跨 repeat-unit 边界传播。
它不等价于真实 3D 闭环构象。
```

### 4.3 Backbone / Attachment Features

Graph 当前加入 backbone 相关特征：

- `is_backbone_atom`
- `is_attachment_neighbor`
- `is_side_chain_atom`
- `is_star_linking_edge`

当前 backbone 定义：

```text
两个 attachment neighbor atoms 之间的最短路径 = backbone
```

这些特征用于 Graph encoder，不直接进入 Geom 坐标构造。

## 5. FP 流程

FP 模态使用 Morgan fingerprint。

注意：

- RDKit 版本较新时可能提示 MorganGenerator deprecation warning。
- 这是 RDKit API 提示，不代表训练失败。
- 如需彻底消除 warning，应统一迁移到 MorganGenerator。

## 6. Geom 流程

Geom 当前支持：

```bash
--geometry_encoder painn
--geometry_encoder schnet
```

当前 `scripts/run.sh` 推荐主线：

```bash
--geometry_encoder schnet
```

但同一次完整流程必须保持预训练和下游训练的 `geometry_encoder` 一致。不要用 SchNet 预训练 checkpoint 去初始化 PaiNN 下游训练，除非明确是在做不兼容权重加载实验。

### 6.1 Geom 输入原则

Geom 不默认使用 star-linked closed graph。

原因：

```text
star-linking 是拓扑处理。
SchNet / PaiNN 使用 3D 坐标、欧氏距离和方向。
把 repeat unit 强行闭成环状 3D 构象，通常不等价于真实无限聚合物链。
```

### 6.2 当前 Geom 构象策略

当前 Geom 推荐使用：

- star-substitution capped repeat unit。
- 多构象生成。
- MMFF 优先。
- UFF fallback。
- 3D 失败时 fallback 到 2D。
- 训练时可随机选择 conformer。

star-substitution 逻辑：

```text
左侧 * 用右侧边界邻居元素替换
右侧 * 用左侧边界邻居元素替换
```

这样做的目标是给 open repeat unit 一个较合理的端基近似，而不是制造 star-linked 闭环 3D。

### 6.3 PaiNN 稳定性

PaiNN 对坐标、距离和异常构象敏感。修改 Geom 时必须检查：

- 是否出现 `nan`。
- 距离是否存在 0 或异常小值。
- batch index 是否正确。
- fallback 后 `pos3d` 是否仍可用。
- 多构象 shape 是否一致。

## 7. 模型结构

当前主模型：

```text
UniEncoderAttention
```

输入模态：

```text
smiles graph fp geom
```

Graph encoder：

```text
GINE / GIN-style PyG encoder
```

当前推荐 Graph 参数：

```bash
--graph_num_layers 6
--graph_emb_dim 256
--graph_dropout 0.1
--graph_pooling attention
```

Geom encoder：

```text
PaiNN 或 SchNet
```

当前融合方式：

```text
各模态 encoder 输出 embedding
-> projection 到 joint_embedding_dim
-> stack
-> attention fusion
-> prediction head
```

## 8. 预训练流程

`scripts/pretrain.py` 支持三种阶段：

```bash
--pretrain_stage graph_geom
--pretrain_stage alignment
--pretrain_stage joint
```

### 8.1 Stage 1：Graph + Geom 单模态预训练

目标：

```text
先让 Graph / Geom encoder 学到结构表征，不做跨模态强对齐。
```

典型 loss：

- Graph masked atom reconstruction。
- Graph repeat-unit vs star-linking consistency。
- Geom coordinate denoising。
- Geom pairwise distance reconstruction。

典型参数：

```bash
--pretrain_stage graph_geom
--geom_denoise_weight 1.0
--graph_pretrain_weight 1.0
--graph_starlink_consistency_weight 0.5
```

输出：

```text
pretrained_models/saved_pretrained_model_starlink_schnet_graph_geom_all.pth
```

### 8.2 Stage 2：多模态 Alignment 预训练

目标：

```text
加载 Stage 1 权重，再进行 SMILES / Graph / FP / Geom 的 contrastive alignment。
```

典型参数：

```bash
--pretrain_stage alignment
--pretrained_model_path pretrained_models/saved_pretrained_model_starlink_schnet_graph_geom_all.pth
--geom_denoise_weight 0.1
--graph_pretrain_weight 0.1
--graph_starlink_consistency_weight 0.1
```

输出：

```text
pretrained_models/saved_pretrained_model_starlink_schnet_alignment_all.pth
```

### 8.3 joint 阶段

`joint` 是兼容旧逻辑的 baseline：

```text
contrastive + graph auxiliary + geom auxiliary 直接相加。
```

不推荐作为最终主线。

## 9. 下游训练流程

下游训练入口：

```bash
python scripts/train.py
```

推荐加载 alignment checkpoint：

```bash
--pretrained_model_path ./pretrained_models/saved_pretrained_model_starlink_schnet_alignment_all.pth
```

当前训练方式：

```text
每个 task 独立训练
每个 task 做 5-fold cross validation
每个 fold 使用 4/5 训练，1/5 测试
当前代码中 test loader 同时作为 early-stopping validation loader
```

输出：

```text
results/results_starlink_schnet_multistage_8tasks.csv
results/results_starlink_schnet_multistage_8tasks_attention_heatmap.png
saved_models_starlink_schnet_multistage_8tasks/<task>/UniEncoderAttention_best.pth
```

### 9.1 Attention 可视化

`train.py` 会在所有 task 训练完成、results CSV 全部写入后生成一次 attention heatmap。默认输出路径由 `--results_dir` 推导：

```text
results/<results_csv_stem>_attention_heatmap.png
```

可选参数：

```bash
--attention_heatmap_path ./results/custom_attention_heatmap.png
--disable_attention_heatmap
```

热力图使用每个 task 的 5-fold mean attention，展示 SMILES / Graph / FP / Geom 在 fusion attention 中的相对权重。

注意：

```text
如果需要严格 train / val / test 三划分，需要改 scripts/train.py。
当前 5-fold 逻辑更接近 cross-validation benchmark。
```

## 10. 运行脚本

### 10.1 完整流程

```bash
bash scripts/run.sh
```

阶段：

```text
Stage 1/3 Graph + Geom 单模态预训练
Stage 2/3 多模态 alignment 预训练
Stage 3/3 下游 5-fold supervised training
```

可用环境变量控制 epoch：

```bash
GRAPH_GEOM_EPOCHS=5 ALIGN_EPOCHS=3 TRAIN_EPOCHS=20 bash scripts/run.sh
```

### 10.2 只跑预训练

```bash
bash scripts/run_pretrain.sh
```

阶段：

```text
Stage 1/2 Graph + Geom 单模态预训练
Stage 2/2 多模态 alignment 预训练
```

### 10.3 单独命令示例

Graph + Geom 单模态预训练：

```bash
python scripts/pretrain.py \
  --dataset_name smi_all \
  --modalities smiles graph fp geom \
  --geometry_encoder schnet \
  --graph_input star_linking \
  --pretrain_stage graph_geom \
  --epochs 20 \
  --geom_denoise_weight 1.0 \
  --graph_pretrain_weight 1.0 \
  --graph_starlink_consistency_weight 0.5 \
  --rebuild_feature_cache \
  --save_path ./pretrained_models/saved_pretrained_model_starlink_schnet_graph_geom_all.pth
```

Alignment 预训练：

```bash
python scripts/pretrain.py \
  --dataset_name smi_all \
  --modalities smiles graph fp geom \
  --geometry_encoder schnet \
  --graph_input star_linking \
  --pretrain_stage alignment \
  --pretrained_model_path ./pretrained_models/saved_pretrained_model_starlink_schnet_graph_geom_all.pth \
  --epochs 10 \
  --geom_denoise_weight 0.1 \
  --graph_pretrain_weight 0.1 \
  --graph_starlink_consistency_weight 0.1 \
  --save_path ./pretrained_models/saved_pretrained_model_starlink_schnet_alignment_all.pth
```

下游训练：

```bash
python scripts/train.py \
  --modalities smiles graph fp geom \
  --geometry_encoder schnet \
  --graph_input star_linking \
  --tasks eat eea egb egc ei eps nc xc \
  --pretrained_model_path ./pretrained_models/saved_pretrained_model_starlink_schnet_alignment_all.pth \
  --epochs 100 \
  --patience 10 \
  --results_dir ./results/results_starlink_schnet_multistage_8tasks.csv \
  --models_dir ./saved_models_starlink_schnet_multistage_8tasks
```

## 11. 输出文件

预训练权重：

```text
pretrained_models/saved_pretrained_model_starlink_schnet_graph_geom_all.pth
pretrained_models/saved_pretrained_model_starlink_schnet_alignment_all.pth
```

训练结果：

```text
results/results_starlink_schnet_multistage_8tasks.csv
results/results_starlink_schnet_multistage_8tasks_attention_heatmap.png
saved_models_starlink_schnet_multistage_8tasks/
```

日志：

```text
logs/run_starlink_schnet_multistage_8tasks.log
logs/pretrain_starlink_schnet_multistage_all.log
```

预训练曲线：

```text
plots/pretrain/loss_data.json
plots/pretrain/loss_curve.png
```

这些均属于运行产物，通常不应提交到 git。

## 12. 必须同步 PIPELINE.MD 的改动

以下改动必须同步本文档：

- 新增或删除模态。
- 修改 `scripts/run.sh` / `scripts/run_pretrain.sh` / `scripts/run_train.sh`。
- 修改 `scripts/pretrain.py` 的阶段、loss 或 checkpoint 逻辑。
- 修改 `scripts/train.py` 的 fold、validation、checkpoint 或输出逻辑。
- 修改 Graph 构造、Graph 特征、star-linking、backbone 逻辑。
- 修改 Geom 构象、star-substitution、多构象或 fallback 逻辑。
- 修改 feature cache 目录、命名、内容或 rebuild 条件。
- 修改默认数据文件或任务列表。
- 修改结果输出路径或模型保存路径。

如果代码和本文档不一致，应优先判断代码是否是误改；确认后同步修正文档或代码，不能让两者长期冲突。

