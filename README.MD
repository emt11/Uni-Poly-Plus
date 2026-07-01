# Uni-Poly-Plus

Uni-Poly-Plus 是面向聚合物性质预测的多模态学习项目。当前主线使用四种模态：

```text
SMILES + Graph + FP + Geom -> UniEncoderAttention -> property prediction
```

当前推荐路线是：

```text
data/raw/smi_all.csv
-> SMILES / Graph / FP / Geom feature cache
-> Graph + Geom 单模态预训练
-> 多模态 alignment 预训练
-> data/raw/smi_<task>.csv 下游 5-fold supervised training
-> results / saved_models
```

默认工作目录：

```bash
cd /root/workspace/DeepLearning/Uni-Poly-Plus-master
```

## 1. 安装依赖

```bash
pip install -r requirements.txt
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

当前 `data/raw/smi_all.csv` 是 no-`tg` 版本，由上述 8 个单任务 CSV 合并生成，不包含 `prop=tg`。`data/raw/smi_tg.csv` 保留在仓库中，但不进入当前主线预训练、缓存构建和下游任务列表。若需要恢复 9-task 实验，需要重新生成包含 `tg` 的 `smi_all.csv`，并同步修改脚本中的 `--tasks` 和输出路径。

## 3. 支持模态

训练和预训练入口支持：

```text
smiles graph fp geom
```

说明：

- `smiles`：SMILES 序列编码。
- `graph`：二维分子图编码，当前后端为 PyG GIN/GINE 路线。
- `fp`：Morgan fingerprint。
- `geom`：三维几何编码，支持 `painn` / `schnet`。

`graph_input` 当前支持：

```bash
--graph_input repeat_unit
--graph_input star_linking
```

推荐使用：

```bash
--graph_input star_linking
```

`star_linking` 只用于 Graph 模态：删除两个 `*` attachment atoms，并连接两个边界邻居原子。失败时 fallback 到原始 repeat-unit graph。

Geom 模态不使用 star-linked closed graph。当前 Geom 使用 star-substitution capped repeat unit 生成 3D 构象，避免把 repeat unit 强行闭成不合理 3D 环。

## 4. 推荐运行

完整串行流程：

```bash
bash scripts/run.sh
```

该脚本按顺序执行：

```text
Stage 1: Graph + Geom 单模态预训练
Stage 2: 多模态 alignment 预训练
Stage 3: 下游 5-fold supervised training
```

默认输出：

```text
logs/run_starlink_schnet_multistage_8tasks.log
pretrained_models/saved_pretrained_model_starlink_schnet_graph_geom_all.pth
pretrained_models/saved_pretrained_model_starlink_schnet_alignment_all.pth
results/results_starlink_schnet_multistage_8tasks.csv
results/results_starlink_schnet_multistage_8tasks_attention_heatmap.png
saved_models_starlink_schnet_multistage_8tasks/
```

可以通过环境变量临时调整 epoch：

```bash
GRAPH_GEOM_EPOCHS=5 ALIGN_EPOCHS=3 TRAIN_EPOCHS=20 bash scripts/run.sh
```

单独预训练入口：

```bash
bash scripts/run_pretrain.sh
```

单独训练入口：

```bash
bash scripts/run_train.sh
```

注意：分开运行脚本时，必须确认 `geometry_encoder`、`graph_input`、checkpoint 路径一致。不要用 SchNet 预训练权重初始化 PaiNN 下游训练，反之亦然。

## 5. 关键参数

常用模型参数：

```bash
--modalities smiles graph fp geom
--graph_input star_linking
--geometry_encoder schnet    # 当前 scripts/run.sh 默认路线
--geometry_encoder painn      # 可选路线，需保持预训练和训练一致
--joint_embedding_dim 256
--attention_heatmap_path ./results/custom_attention_heatmap.png  # 可选
--disable_attention_heatmap                              # 可选，关闭自动绘图
```

Graph 参数：

```bash
--graph_num_layers 6
--graph_emb_dim 256
--graph_dropout 0.1
--graph_pooling attention
```

预训练阶段参数：

```bash
--pretrain_stage graph_geom
--pretrain_stage alignment
--geom_denoise_weight 1.0
--geom_noise_std 0.2
--geom_distance_loss_weight 1.0
--graph_pretrain_weight 1.0
--graph_mask_atom_weight 1.0
--graph_mask_ratio 0.15
--graph_starlink_consistency_weight 0.5
```

实际可用参数以 help 为准：

```bash
python scripts/pretrain.py --help
python scripts/train.py --help
```

## 6. Feature Cache

当前项目使用 SMILES 级别 feature cache，避免预训练和每个下游任务重复处理同一个 SMILES。

默认缓存目录：

```text
data/processed/Schnet/
```

缓存内容包括：

```text
SMILES token
Graph data
Fingerprint
Geom data
metadata
```

修改以下内容后需要重建缓存：

- `graph_input`
- Graph 节点/边特征维度
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

## 7. 预训练任务

当前预训练包含：

```text
Graph masked atom reconstruction
Graph repeat-unit vs star-linking consistency
Geom coordinate denoising
Geom pairwise distance reconstruction
Multi-modal contrastive alignment
```

其中：

- Graph masked atom reconstruction：遮蔽 atom type 后重建。
- Graph repeat-unit vs star-linking consistency：让同一 SMILES 的 repeat-unit graph 和 star-linked graph 表示接近。
- Geom coordinate denoising：对 3D 坐标加噪声后恢复 clean position。
- Geom pairwise distance reconstruction：辅助恢复 clean pairwise distance。
- Multi-modal contrastive alignment：对齐 SMILES / Graph / FP / Geom 表示。

这些任务的详细语义和后续优化计划见 [PIPELINE.MD](PIPELINE.MD) 和 [TODO.MD](TODO.MD)。

## 8. 常见输出

训练完成后会自动生成 attention heatmap，默认路径为 `results/<results_csv_stem>_attention_heatmap.png`。

```text
logs/
pretrained_models/
results/
results/*_attention_heatmap.png
saved_models_starlink_schnet_multistage_8tasks/
plots/pretrain/
data/processed/Schnet/
```

常用查看命令：

```bash
tail -f logs/run_starlink_schnet_multistage_8tasks.log
```

## 9. 源码目录

```text
src/dataset/       数据读取、SMILES/Graph/FP/Geom 特征构建、cache 和 collate
src/modules/       SMILES、Graph、Geometry encoder 与多模态融合
scripts/           预训练、下游训练、绘图和运行脚本
```

## 10. 文档导航

- [PIPELINE.MD](PIPELINE.MD)：当前项目整体数据流、模型流、预训练和训练流程。涉及流程改动时必须同步更新。
- [AGENT.MD](AGENT.MD)：开发和代码代理约束。
- [TODO.MD](TODO.MD)：后续开发计划和论文方案记录。
- [PerioGT.MD](PerioGT.MD)：PerioGT 代码/论文分析。
- [MCMPP_CrossAttention.MD](MCMPP_CrossAttention.MD)：MCMPP 多模态 cross-attention 论文分析。
- [MFBERT_FGNN.MD](MFBERT_FGNN.MD)：MFBERT + FGNN 论文分析。
- [EDCL.MD](EDCL.MD)：Energy-Guided Denoising Contrastive Learning 论文分析。
- [Galformer.MD](Galformer.MD)：Geometry-Aware Line Graph Transformer 论文分析。

## 11. 维护约束

如果修改以下内容，必须同步更新 [PIPELINE.MD](PIPELINE.MD)：

- 数据流或输入文件。
- Feature cache 逻辑。
- Graph / Geom 构造。
- 预训练阶段或 loss。
- 训练脚本参数。
- 输出路径。
- 支持模态。
