# MCMPP 论文分析：Text / Sequence / Graph / Geometry Cross-Attention

分析对象：`multimodal-cross-attention-molecular-property-prediction-for-text-sequence-graph-and-geometry.pdf`

论文题目：Multimodal Cross-Attention Molecular Property Prediction for Text, Sequence, Graph, and Geometry

期刊：ACS Omega, 2025

本分析基于论文 PDF 正文阅读，重点关注模型结构、模态融合方式、实验设计，以及它对当前 Uni-Poly-Plus 项目的可借鉴内容。

## 1. 论文核心问题

论文关注普通分子属性预测中的多模态融合问题。

它认为单模态分子表示存在明显限制：

```text
SMILES       表达序列和语法信息
ECFP         表达局部子结构和拓扑环境
2D graph     表达原子-键拓扑关系
3D geometry  表达空间构象、距离、角度和相互作用
```

单独使用其中一种表示，无法充分覆盖分子性质所需的信息。因此论文提出 MCMPP：

```text
Multimodal Cross-Attention Molecular Property Prediction
```

核心思想：

```text
分别编码四种分子表示，再使用 cross-attention 进行模态间交互和融合。
```

## 2. 输入模态

论文使用四种分子表示。

### 2.1 SMILES

SMILES 被当作文本序列处理。

处理流程：

```text
SMILES
-> regex tokenizer
-> token dictionary
-> fixed-length integer sequence
-> token embedding
-> Transformer Encoder
-> SMILES feature
```

作用：

```text
捕获分子序列语法、环、支链、原子连接顺序等信息。
```

和当前项目对比：

```text
当前 Uni-Poly-Plus 已经有 SMILES encoder。
当前项目比论文更适合聚合物，因为输入是 P-SMILES / repeat-unit SMILES。
```

### 2.2 ECFP fingerprint

论文使用 ECFP4：

```text
radius = 2
bit length = 1024
```

处理方式不是简单 MLP，而是：

```text
ECFP binary sequence
-> TCN
-> BiLSTM
-> multi-head attention
-> ECFP feature
```

论文理由：

```text
ECFP 是固定长度 bit vector，但它仍然包含局部子结构激活模式。
TCN / BiLSTM / attention 被用于提取 bit 序列中的模式和依赖。
```

和当前项目对比：

```text
当前 Uni-Poly-Plus 的 fp 模态更简单，通常是 fingerprint vector -> projection。
论文提示：fp 模态不一定只能作为静态向量，可以考虑更强的 fp encoder。
```

### 2.3 Molecular graph

论文使用普通 2D molecular graph：

```text
node = atom
edge = bond
```

节点特征包括：

```text
元素类型
价态 / formal charge / hybridization
氢数量
bond environment
芳香性
```

模型：

```text
2-layer GCN + residual + GELU + BatchNorm + global pooling
```

公式上是标准 GCN：

```text
H_{l+1} = D^-1/2 A D^-1/2 H_l W_l
```

并加入 residual-enhanced GCN：

```text
H_{l+1} = GELU(BN(GCN(H_l) W)) + H_l
```

和当前项目对比：

```text
当前 Uni-Poly-Plus 使用 GINE / GIN-style graph encoder，比论文中的普通 GCN 更适合分子图。
当前项目还加入了 star-linking 和 backbone features，这是论文没有处理的聚合物周期信息。
```

### 2.4 3D molecular conformation

论文使用 RDKit 生成 3D：

```text
SMILES
-> MolFromSmiles
-> AddHs
-> ETKDG
-> MMFF94 energy minimization
-> 3D conformation
```

3D encoder 使用 reduced Uni-Mol+：

```text
去掉 Uni-Mol+ 原始 property prediction head
保留 3D feature extractor
4-layer architecture
hidden dim = 256
```

论文中的 Uni-Mol+ 是 dual-track Transformer，包含：

```text
atomic representation
pair representation
coordinate refinement / coordinate offset update
```

和当前项目对比：

```text
当前项目使用 PaiNN / SchNet，而不是 Uni-Mol+。
PaiNN 对 3D equivariance 更明确，适合当前 geometry 主线。
论文对当前项目最有价值的不是 Uni-Mol+ 本身，而是 3D encoder 应参与跨模态交互，而不是只在最后简单拼接。
```

## 3. 模型结构

论文 MCMPP 有四层：

```text
A. Feature representation layer
B. Feature extraction layer
C. Feature fusion layer
D. Prediction layer
```

### 3.1 Feature representation layer

输入四种表示：

```text
SMILES
ECFP
2D graph
3D conformation
```

### 3.2 Feature extraction layer

对应四个 encoder：

```text
SMILES      -> Transformer Encoder
ECFP        -> TCN + BiLSTM + multi-head attention
Graph       -> residual GCN + global pooling
3D geometry -> reduced Uni-Mol+
```

输出四个 embedding：

```text
H_s
H_e
H_g
H_3D
```

### 3.3 Feature fusion layer

论文比较两类融合方式。

第一类：decision-level late fusion

```text
每个模态单独预测
-> LASSO / ElasticNet / RF / GB / SGD 计算权重
-> 加权输出
```

第二类：cross-attention mechanism fusion，即 MCMPP_CAM。

```text
四个模态特征
-> cross-attention
-> fused representation
-> MLP prediction
```

论文结论：

```text
cross-attention 优于 late fusion。
late fusion 容易给弱模态过高权重，或低估强模态。
```

## 4. Cross-Attention 逻辑

论文将其中一个模态作为 target query，其余模态作为 source key/value。

公式形式：

```text
CrossAttn(Q_t, K_i, V_i) = softmax(Q_t K_i^T / sqrt(d_k)) V_i
```

其中：

```text
Q_t 来自目标模态
K_i, V_i 来自源模态 i
```

然后通过 multi-head attention 得到融合表示。

论文声称 cross-attention 的作用：

```text
动态对齐不同模态
建模模态间互补性
降低 late fusion 的权重偏差
提升可解释性，因为可以观察模态 attention 权重
```

## 5. 实验设计

数据集：

```text
Delaney
Lipophilicity
SAMPL
BACE regression
PDBbind v2020 refined
```

划分方式：

```text
多数数据集使用 train / validation / test = 8 / 1 / 1 random split
还额外比较 random split 和 scaffold split
```

评价指标：

```text
Pearson correlation
RMSE
MAE
```

重复实验：

```text
15 independent trials 用于稳定性分析
部分 baseline 对比使用多随机种子平均
```

## 6. 关键实验结论

### 6.1 多模态优于单模态

论文表格显示，单模态中 reduced Uni-Mol+ 往往很强，但 MCMPP_CAM 通常更好。

例如 Table 1 中 Pearson：

```text
Delaney:      MCMPP_CAM = 0.971
Lipophilicity: MCMPP_CAM = 0.913
SAMPL:        MCMPP_CAM = 0.978
BACE:         MCMPP_CAM = 0.841
```

RMSE / MAE 中，MCMPP_CAM 也通常最佳。

### 6.2 Cross-attention 优于 decision-level fusion

论文指出 late fusion 的问题：

```text
late fusion 可能给弱模态过高权重。
late fusion 难以建模早期模态间细粒度交互。
cross-attention 能在 feature level 进行动态模态对齐。
```

这点对当前项目很重要，因为当前 Uni-Poly-Plus 主要是：

```text
各模态 embedding stack -> attention fusion
```

它更接近模态级融合，而不是 node/feature-level cross-attention。

### 6.3 稳定性更好

论文进行了 15 次重复实验，观察 Pearson 分布。

结论：

```text
MCMPP_CAM 的 Pearson 分布更集中，稳定性优于单模态模型。
```

这对当前项目的 5-fold 波动问题有参考价值。

### 6.4 UMAP 用于分析模态互补性

论文使用 UMAP 可视化 SMILES / ECFP hidden vectors。

目的：

```text
观察 train/test 在不同模态空间中的重叠程度。
解释为什么某个模态在某个数据集上表现好或差。
```

这对当前项目可以直接借鉴，用来分析不同 task 的 fold 差异。

## 7. 论文不足

### 7.1 不是聚合物专用

论文处理普通小分子，不处理：

```text
P-SMILES
repeat unit
attachment point '*'
star-linking
聚合物周期性
链长 / 主链 / 侧链
```

因此不能直接照搬到当前聚合物任务。

### 7.2 Graph 模块较弱

论文 Graph encoder 是 2-layer GCN，表达能力有限。

当前项目的 GINE + star-linking 更适合聚合物拓扑。

### 7.3 3D 生成适用于小分子，不一定适用于聚合物 repeat unit

论文的 3D 流程：

```text
AddHs + ETKDG + MMFF94
```

对于普通小分子合理，但对带 `*` 的 repeat-unit SMILES 需要特殊处理。

当前项目的 star-substitution capped repeat unit 更贴近聚合物场景。

### 7.4 模型参数可能较大

论文自己也指出，四个独立 encoder + fusion 带来较长训练时间。

当前项目已有类似问题，应谨慎增加复杂 cross-attention。

## 8. 对当前 Uni-Poly-Plus 的帮助

### 8.1 最值得借鉴：Graph-Geom / 多模态 Cross-Attention

当前项目目前融合方式是模态级 attention：

```text
smiles_emb, graph_emb, fp_emb, geom_emb
-> stack
-> attention pooling
```

可以升级为：

```text
Graph query attends to Geom key/value
Geom query attends to Graph key/value
SMILES query attends to Graph/Geom/FP key/value
```

推荐优先做轻量版本：

```text
graph_emb, geom_emb -> cross-attention block -> fused_graph_geom
smiles_emb, fp_emb, fused_graph_geom -> modality attention
```

不要一开始做全 token / node / atom 级 cross-attention，工程风险太大。

### 8.2 可以引入模态可靠性分析

论文用 15 次实验和 UMAP 解释模态表现。

当前项目可以加：

```text
每个 task / fold 输出 modality attention
UMAP 可视化各模态 embedding
比较 train/test split 的 embedding overlap
分析为什么某些 fold 差异大
```

当前已有 `plot_attention_heatmap.py`，可以扩展为：

```text
attention heatmap + fold performance + embedding UMAP
```

### 8.3 FP encoder 可以增强，但不是最高优先级

论文用 TCN-BiLSTM-attention 编码 ECFP。

当前项目可以考虑：

```text
短期：保持 fp MLP projection
中期：fingerprint tokenization + lightweight Transformer / attention pooling
长期：借鉴 MFBERT，而不是 MCMPP 的 BiLSTM
```

对当前聚合物项目而言，优先级：

```text
Graph star-linking > Geom 稳定性 > Graph-Geom cross-attention > FP encoder 增强
```

### 8.4 cross-attention 可替代当前简单 stack fusion

当前融合问题：

```text
各模态只在最终 embedding 级别交互。
Graph 和 Geom 没有显式互相校正。
```

可借鉴 MCMPP：

```text
在 prediction 前加入 CrossModalFusionBlock。
```

建议实现路线：

```text
Stage A: 保留现有 encoder，不改数据。
Stage B: 新增 graph_geom_cross_attention。
Stage C: ablation 对比：原 attention fusion vs gated fusion vs cross-attention fusion。
```

## 9. 建议落地方案

### 9.1 短期方案

在当前 `UniEncoderAttention` 中新增一个可选参数：

```bash
--fusion_type attention
--fusion_type graph_geom_gated
--fusion_type cross_attention
```

初始只实现 embedding-level cross-attention：

```text
输入 shape: [batch, num_modalities, dim]
MultiheadAttention over modality tokens
输出 fused embedding
```

这比当前 attention pooling 更强，但改动较小。

### 9.2 中期方案

Graph / Geom 专门 cross-attention：

```text
graph node embedding <-> geom node embedding
```

前提：

```text
Graph 和 Geom atom 对齐必须可靠。
当前 star-linking graph 删除了两个 *，Geom 使用 capped repeat unit，两者节点不完全一致。
```

因此中期需要先建立 atom mapping。

### 9.3 长期方案

聚合物专用 cross-attention：

```text
backbone token
side-chain token
attachment token
geom local environment token
fp substructure token
```

这会比 MCMPP 更适合聚合物。

## 10. 对当前项目的优先级判断

推荐优先级：

```text
1. 保持当前 star-linking + PaiNN 多阶段预训练稳定。
2. 加入 embedding-level cross-attention fusion，替代或对比当前 modality attention。
3. 增加 UMAP / attention / fold stability 分析。
4. 再考虑 FP encoder 增强。
5. 暂不引入 reduced Uni-Mol+，避免和当前 PaiNN/SchNet 主线冲突。
```

## 11. 与当前项目差异总结

| 维度 | MCMPP | 当前 Uni-Poly-Plus | 建议 |
|---|---|---|---|
| 数据对象 | 小分子 | 聚合物 repeat unit | 不能直接照搬 3D 流程 |
| SMILES | Transformer | 已有 SMILES encoder | 保留 |
| FP | TCN-BiLSTM-attention | 简单 FP projection | 中期增强 |
| Graph | 2-layer residual GCN | GINE + star-linking + backbone | 当前更适合聚合物 |
| Geom | reduced Uni-Mol+ | PaiNN / SchNet | 保持 PaiNN 主线 |
| 融合 | cross-attention | modality attention | 借鉴 cross-attention |
| 实验 | random/scaffold split | 5-fold CV | 可增加 split/UMAP 稳定性分析 |

## 12. 结论

这篇论文对当前项目最有价值的不是具体 encoder，而是两个思想：

```text
1. 多模态 fusion 不应只做简单拼接或 late fusion，cross-attention 更适合建模模态互补。
2. 应该通过重复实验、模态 attention、UMAP 等方式分析模态贡献和稳定性。
```

对 Uni-Poly-Plus 的直接建议：

```text
保留当前 polymer-specific Graph/Geom 处理。
不要替换为论文的小分子 GCN / Uni-Mol+ 流程。
优先实现 embedding-level cross-attention fusion，并和当前 modality attention 做 ablation。
```
