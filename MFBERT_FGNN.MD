# MFBERT + FGNN 论文分析：Robust Fractal Graph Neural Network with Molecular Fingerprint BERT

分析对象：`Structure-Based_Robust_Fractal_Graph_Neural_Network_with_Molecular_Fingerprint_BERT_for_Molecular_Property_Prediction.pdf`

论文题目：Structure-Based Robust Fractal Graph Neural Network with Molecular Fingerprint BERT for Molecular Property Prediction

期刊：IEEE Transactions on Emerging Topics in Computational Intelligence, 2026

本分析基于论文 PDF 正文阅读，重点关注 MFBERT、Fractal Graph Neural Network、3D feature extraction、DRO 优化，以及它们对当前 Uni-Poly-Plus 项目的可借鉴内容。

## 1. 论文核心问题

论文关注普通分子属性预测中的三个问题：

```text
1. 标注数据不足。
2. 标签分布不均衡。
3. 很多 GNN 只使用 2D graph，忽略 3D conformation。
```

为解决这些问题，论文提出：

```text
MFBERT + FGNN + W-DRO
```

完整名称：

```text
Structure-Based Robust Fractal Graph Neural Network with Molecular Fingerprint BERT
```

核心流程：

```text
SMILES
-> ECFP sequence
-> MFBERT pretraining
-> ECFP contextual embeddings as graph node features
-> FGNN with 2D + 3D message passing
-> W-DRO robust optimization
-> property prediction
```

## 2. 整体框架

论文 Figure 1 描述了完整流程：

```text
Dataset partition
-> Data processing
-> Pre-training MFBERT
-> Pre-training FGNN
-> MFBERT embedding
-> FGNN feature extraction
-> Pooling
-> MLP prediction
-> W-DRO optimization
```

这不是简单的多模态拼接，而是把 fingerprint BERT embedding 注入 Graph 节点特征，再用 FGNN 同时处理 2D graph 和 3D geometry。

## 3. MFBERT：Molecular Fingerprint BERT

### 3.1 输入不是 SMILES token，而是 ECFP token

论文把 ECFP 子结构看成自然语言中的“词”。

处理逻辑：

```text
SMILES
-> Morgan algorithm
-> ECFP identifiers / substructure sequence
-> token embedding + position embedding
-> Transformer encoder
-> contextual ECFP embeddings
```

这和普通 SMILES-BERT 不同。

普通 SMILES-BERT：

```text
SMILES characters / tokens -> BERT
```

MFBERT：

```text
ECFP substructure identifiers -> BERT
```

### 3.2 ECFP radius

论文重点分析了 ECFP radius `r`。

结论：

```text
r = 1 最好
```

原因：

```text
r = 0 只有中心原子，缺少环境信息。
r = 1 包含中心原子 + 邻居原子 + bond type，能表达局部功能团。
r 更大时复杂度上升，可能引入噪声和过拟合。
```

这点对当前项目很重要，因为聚合物 repeat unit 经常存在局部片段重复，过大的 fingerprint radius 可能会产生噪声。

### 3.3 预训练任务

MFBERT 使用 MLM：

```text
Mask Language Model
```

预训练数据：

```text
ZINC > 90 million SMILES
```

目的：

```text
利用大量无标签分子，学习 ECFP 子结构的上下文表示。
```

### 3.4 Fine-tuning

论文分两阶段：

```text
1. 用无标签数据预训练 MFBERT。
2. 在下游任务中 fine-tune MFBERT + FGNN。
```

并在实验中证明 frozen MFBERT 性能低于 fine-tuned MFBERT。

## 4. FGNN：Fractal Graph Neural Network

### 4.1 Fractal 思想

论文中的 fractal 不是分形化学结构，而是一种网络结构设计。

含义：

```text
用不同深度的子路径提取不同尺度的分子特征。
浅路径响应快，深路径表达强。
多个路径 join 后形成最终表示。
```

FGNN block 包括：

```text
多个 GNN sub-path
join layer
drop-path regularization
```

drop-path 作用：

```text
随机关闭部分路径，至少保留一条路径。
降低过拟合，提高鲁棒性。
```

### 4.2 FGNN 输入

FGNN 输入包括：

```text
1. MFBERT node embedding
2. molecular edge features
3. 3D coordinates
4. Coulomb matrix
```

其中：

```text
node feature = MFBERT embedding of atom and local ECFP environments
edge feature = 18-dimensional molecular edge features
3D coordinate = RDKit conformer
Coulomb matrix = based on 3D coordinates and Gasteiger charges
```

### 4.3 2D feature extraction module

2D message passing 使用 molecular graph：

```text
m_ij^2D = phi_2D(h_i, h_j, e_ij)
m_i^2D = sum_{j in N_i} m_ij^2D * rbf_ij
h_i = h_i + m_i^2D
```

注意：2D message 也乘了 `rbf_ij`，也就是 3D 距离权重。

这说明论文不是把 2D 和 3D 完全分开，而是在 2D message passing 中也注入了几何距离。

### 4.4 3D feature extraction module

3D message passing 使用空间子集 `S_R`：

```text
m_ij^3D = phi_3D(h_i, h_j, c_i)
m_i^3D = sum_{j in S_R} m_ij^3D * rbf_ij
h_i = h_i + m_i^3D
```

其中：

```text
c_i = Coulomb matrix embedding
rbf_ij = distance RBF expansion
```

论文选择随机节点子集，而不是所有空间节点，理由：

```text
降低计算量
避免 node features 过度相似
```

### 4.5 Coordinate update module

论文加入坐标更新：

```text
p = p + Delta p = p + phi_p(h, p)
```

然后下一层继续基于新坐标计算：

```text
distance RBF
Coulomb matrix features
message passing weights
```

这点和当前项目的 Geom denoising / PaiNN 预训练有相通之处。

## 5. W-DRO：Weighted-gradient Distributionally Robust Optimization

论文认为分子属性数据存在标签分布不均衡，因此使用 W-DRO。

核心思想：

```text
把数据按 label 分成若干子集。
优化时不只最小化整体平均 loss，而是关注各子集中的最大平均 loss。
```

目标近似：

```text
min_theta max_k mean_loss(M_k)
```

然后通过加权梯度组合得到更新方向。

意义：

```text
避免模型只在多数分布区域表现好。
提高 minority label range 或 minority class 的性能。
```

对当前项目很有参考价值，因为聚合物属性数据经常分布不均衡，例如某些性质极端值样本少。

## 6. 实验设计

### 6.1 预训练数据

MFBERT 预训练：

```text
ZINC > 90 million sequences
```

FGNN 预训练：

```text
QM9 trusted conformations
```

### 6.2 下游数据集

论文使用 MoleculeNet 七个数据集。

回归：

```text
ESOL
FreeSolv
Lipophilicity
```

分类：

```text
BACE
MUV
Tox21
SIDER
```

### 6.3 评价指标

回归：

```text
RMSE
```

分类：

```text
ROC-AUC
```

实验报告：

```text
10 independent runs
```

### 6.4 训练策略

论文训练分两阶段：

```text
阶段 1：冻结 MFBERT，只训练 FGNN。
阶段 2：fine-tune MFBERT + FGNN。
```

这和当前项目最近改成多阶段预训练的思路一致。

## 7. 关键实验结论

### 7.1 整体性能

论文在七个 MoleculeNet 数据集上取得优于多个 baseline 的结果。

回归任务中，论文报告：

```text
ESOL RMSE = 0.563
FreeSolv RMSE = 1.157
Lipophilicity RMSE = 0.612
```

分类任务中，论文在 BACE / MUV / Tox21 / SIDER 上也报告较高 ROC-AUC。

### 7.2 MFBERT 有效

消融显示：

```text
MFBERT embedding 优于手工 67-dimensional node features。
Frozen MFBERT 差于 fine-tuned MFBERT。
```

说明：

```text
fingerprint contextual embedding 可以作为更强的 graph node feature。
```

### 7.3 3D module 有效

消融显示：

```text
FNN < 2D FGNN < 2D+3D FGNN
```

说明：

```text
3D feature extraction module 能提高预测性能。
```

### 7.4 FGNN 预训练有效

使用 QM9 预训练 FGNN 后：

```text
ESOL RMSE 进一步降低
BACE ROC-AUC 进一步提高
```

说明：

```text
3D/Graph encoder 预训练可以给下游更好初始化。
```

这与当前项目“先 Graph/Geom 单模态预训练，再 alignment”的方向一致。

### 7.5 W-DRO 有效

去掉 W-DRO 后性能下降：

```text
ESOL RMSE 上升
BACE ROC-AUC 下降
```

说明：

```text
针对 label imbalance 的 robust optimization 有实际价值。
```

## 8. 论文不足

### 8.1 不是聚合物专用

论文处理普通小分子，不处理：

```text
P-SMILES
repeat unit
attachment point '*'
star-linking
聚合物周期性
主链 / 侧链
```

因此不能直接照搬到当前聚合物任务。

### 8.2 MFBERT 预训练成本很大

论文使用 ZINC 9000 万级别数据预训练 MFBERT。

当前项目如果复现完整 MFBERT，成本较高。

### 8.3 FGNN 与当前 PaiNN / SchNet 主线存在重叠

FGNN 自己实现 2D+3D message passing 和坐标更新。

当前项目已经有：

```text
Graph encoder = GINE
Geom encoder = PaiNN / SchNet
Fusion = UniEncoderAttention
```

如果完整替换为 FGNN，会大幅改变项目结构。

更合理的是借鉴思想，而不是整体替换。

### 8.4 Coulomb matrix 对聚合物 repeat unit 未必稳定

论文使用 Coulomb matrix + Gasteiger charges。

对带 `*` 的 repeat unit，需要先 cap / substitute，否则电荷和 Coulomb matrix 可能不稳定。

## 9. 对当前 Uni-Poly-Plus 的帮助

### 9.1 最值得借鉴：Fingerprint contextual embedding

当前项目 fp 模态是独立模态。

论文提供另一种思路：

```text
把 fingerprint / local substructure embedding 作为 graph node feature。
```

对当前项目可以设计为：

```text
atom-level local environment token
-> lightweight fingerprint BERT / embedding table
-> concatenate to Graph node feature
```

短期不建议完整实现 MFBERT，但可以先做轻量版本：

```text
Morgan atom environment id
-> nn.Embedding
-> append to graph node features
```

这比直接上 BERT 更实际。

### 9.2 借鉴 2D message 中注入 3D 距离权重

论文的 2D message passing：

```text
m_ij^2D * rbf(distance_ij)
```

当前 Graph 和 Geom 是分开的。

可借鉴方案：

```text
Graph edge feature 增加 geometry distance RBF
```

但注意当前 Graph 使用 star-linking，Geom 使用 capped repeat unit，两者节点不完全一致。

可行前提：

```text
只对原始 repeat-unit 中真实化学键加入 3D edge distance。
star-linking 新增边不使用真实 3D distance，或单独标记为 periodic_edge。
```

### 9.3 借鉴 Geom 预训练和坐标恢复

论文 FGNN 通过坐标更新增强结构表征。

当前项目已经实现：

```text
Geom denoising
pairwise distance reconstruction
PaiNN encode_nodes
```

这和论文思路一致。

可以继续增强：

```text
多构象一致性 loss
coordinate denoising + force-like delta prediction
bond length / angle reconstruction
```

### 9.4 借鉴 W-DRO 处理标签分布不均衡

当前下游任务使用普通 loss。

聚合物属性通常存在长尾分布，例如极高 Tg / 极低 bandgap 样本少。

可以加一个轻量 W-DRO 版本：

```text
按 y 分位数分 bin
每个 batch 统计各 bin loss
loss = max_bin_loss 或 weighted_bin_loss
```

推荐先用于回归任务：

```text
bin_by_quantile(y_train, n_bins=5)
```

然后训练 loss：

```text
loss = mean_loss + lambda * max_bin_loss
```

这比完整求解论文中的 Lagrange multiplier 简单，也更容易稳定。

### 9.5 借鉴两阶段 fine-tuning

论文训练：

```text
先 freeze MFBERT，只训练 FGNN
再 fine-tune MFBERT + FGNN
```

当前项目可对应为：

```text
Stage 1: Graph/Geom 单模态预训练
Stage 2: Alignment 预训练
Stage 3a: 下游训练时先 freeze encoder，只训练 fusion/prediction head
Stage 3b: unfreeze all，低学习率 fine-tune
```

这比直接全模型 fine-tune 更稳。

## 10. 可落地改造路线

### 10.1 短期

不改大架构，加入两个分析/训练增强：

```text
1. label-bin robust loss / W-DRO-lite
2. downstream two-phase fine-tuning: freeze -> unfreeze
```

原因：

```text
这两个改动对当前代码侵入最小，且直接对应论文消融有效模块。
```

### 10.2 中期

增强 Graph node feature：

```text
Morgan local environment id embedding
append to graph node feature
```

可以先不训练 BERT，只训练 embedding table。

之后再升级：

```text
ECFP token sequence -> lightweight Transformer -> node/substructure embedding
```

### 10.3 中长期

实现 Graph-Geom 交互：

```text
Graph chemical bond message 加入真实 3D bond distance RBF
Geom branch 加入 graph shortest path / backbone distance bias
```

但必须处理 star-linking 和 Geom 节点不一致问题。

### 10.4 不建议立即做

不建议直接完整复现：

```text
完整 MFBERT on huge ZINC
完整 FGNN 替换 GINE + PaiNN
完整 W-DRO Lagrange multiplier solver
Coulomb matrix as mandatory feature
```

原因：

```text
工程量大，且当前项目已经有更聚合物专用的 star-linking / PaiNN 主线。
```

## 11. 与当前项目对照

| 维度 | MFBERT + FGNN | 当前 Uni-Poly-Plus | 建议 |
|---|---|---|---|
| 对象 | 小分子 | 聚合物 repeat unit | 不能直接照搬 |
| Fingerprint | ECFP token BERT | FP 独立模态 | 可借鉴 contextual FP |
| Graph | FGNN | GINE + star-linking + backbone | 保留当前 Graph 主线 |
| 3D | FGNN 内部 3D message + coordinate update | PaiNN / SchNet | 保留 PaiNN，借鉴 denoising / distance loss |
| 预训练 | MFBERT + FGNN 预训练 | Graph/Geom + alignment 多阶段 | 当前方向一致 |
| Robustness | W-DRO | 普通 supervised loss | 可加 W-DRO-lite |
| 融合 | 图内融合 2D/3D | 模态级融合 | 可做 Graph-Geom cross fusion |

## 12. 对当前项目的优先级判断

推荐优先级：

```text
1. 保持 star-linking + backbone + PaiNN 多阶段预训练主线。
2. 加 downstream freeze -> unfreeze fine-tuning。
3. 加 label-bin robust loss / W-DRO-lite。
4. 给 Graph node 增加 local Morgan environment embedding。
5. 再考虑 Graph edge 加 3D distance RBF。
6. 不建议短期完整复现 FGNN。
```

## 13. 结论

这篇论文对当前项目最有价值的三点：

```text
1. Fingerprint 不只可以作为全局向量，也可以作为局部子结构上下文 embedding。
2. 2D graph message passing 可以注入 3D distance / RBF 信息，实现更紧密的 2D-3D 耦合。
3. 标签分布不均衡应该在 loss / optimizer 层面处理，W-DRO 思路适合聚合物属性长尾分布。
```

当前项目不应直接替换为 FGNN，而应选择性吸收：

```text
W-DRO-lite
freeze -> unfreeze downstream fine-tuning
Morgan local environment embedding
Graph-Geom distance-aware interaction
```
