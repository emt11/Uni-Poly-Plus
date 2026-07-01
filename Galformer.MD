# Galformer 论文分析：Geometry-Aware Line Graph Transformer Pretraining

分析对象：`Geometry-Aware_Line_Graph_Transformer_Pretraining_for_Molecular_Property_Prediction.pdf`

论文题目：Geometry-Aware Line Graph Transformer Pretraining for Molecular Property Prediction

期刊：IEEE Transactions on Neural Networks and Learning Systems, 2026

代码：`https://github.com/peizhenbai/Galformer`

本分析基于论文 PDF 正文阅读，重点关注 line graph transformer、2D/3D 双模态预训练、masked line node prediction、dual-modality contrastive learning，以及这些设计对当前 Uni-Poly-Plus 项目的可借鉴价值。

## 1. 论文核心问题

论文关注分子属性预测中的 2D / 3D 自监督预训练。

作者认为已有方法有两个主要不足：

```text
1. Backbone capacity 不足。
   很多方法直接使用普通 2D GNN / 3D GNN，容易受到 message passing 表达力和 over-smoothing 限制。

2. 预训练任务不完整。
   很多 2D-3D 预训练只关注 inter-modality 对齐，忽略每个模态内部的结构学习。
```

因此论文提出：

```text
Geometry-aware line graph transformer, Galformer
```

核心思想：

```text
2D molecular graph -> 2D line graph -> 2D line graph transformer
3D molecular graph -> 3D line graph -> 3D line graph transformer
-> masked line node prediction
-> 2D/3D contrastive learning
-> 下游只使用 pretrained 2D encoder
```

## 2. 为什么使用 Line Graph

普通 molecular graph：

```text
node = atom
edge = bond
```

Line graph：

```text
node = 原图中的 bond / atom pair
edge = 原图中相邻 bond pair
```

对于分子来说，这个转换非常重要：

```text
原 graph 的 bond 变成 line graph 的 node。
原 graph 中两个共享原子的 bond pair 变成 line graph 的 edge。
```

这使模型能够更直接地表达：

```text
bond-level structure
atom-pair relation
bond-bond relation
bond angle
path between bonds
```

论文认为，line graph 比 atom graph 更适合捕获分子结构中的细粒度模式，尤其适合把 3D angle 信息自然放进图结构中。

## 3. 2D Topological Line Graph

给定 2D 分子图：

```text
G2d = (V2d, E2d)
```

其中：

```text
V2d = atoms
E2d = bonds
```

转换成 2D line graph：

```text
G_hat_2d = (V_hat_2d, E_hat_2d)
```

其中：

```text
V_hat_2d = original bonds
E_hat_2d = adjacent bond pairs sharing one atom
```

Line node 特征由原始 atom 和 bond 特征拼接得到：

```text
line_node(uv) = concat(atom_u + atom_v, bond_uv)
```

也就是说，一个 line node 表示一个化学键及其两端原子环境。

## 4. 3D Geometric Line Graph

3D 分子图包含：

```text
atom type
3D coordinate
atomic distance
bond angle
```

论文把 3D graph 也转换成 line graph。

3D line node 表示：

```text
atom pair / bond + pairwise distance encoding
```

3D line edge 表示：

```text
相邻 bond pair 对应的 geometric angle
```

距离和角度是连续值，论文使用 Gaussian basis kernel 映射成高维向量：

```text
distance -> Gaussian basis embedding
angle    -> Gaussian basis embedding
```

这点和 SchNet / DimeNet / PaiNN 类似：连续几何量先用径向基展开，再交给神经网络。

## 5. Dual-Modality Line Graph Transformer

Galformer 有两个结构一致但输入信息不同的 Transformer：

```text
2D line graph transformer
3D line graph transformer
```

一致性体现在：

```text
两者都是 line graph transformer backbone。
两者都以 line node 为 token。
两者都通过 structural bias 改造 self-attention。
```

差异体现在：

```text
2D branch 使用 topology structural encoding。
3D branch 使用 geometry structural encoding。
```

这种设计的目的：

```text
保持 2D / 3D 表示空间更可比，便于后续 contrastive alignment。
```

## 6. 2D Line Graph Transformer 的结构编码

普通 Transformer 不知道图结构，因为 self-attention 本身是 order-agnostic。

因此 Galformer 给 2D line graph 加了三种结构编码。

### 6.1 Laplacian Eigenvector Positional Encoding

对 2D line graph 的 normalized Laplacian 做特征分解，取最小的非平凡 eigenvectors 作为位置编码。

作用：

```text
给 line node 提供图中的全局/局部位置感知。
```

直觉：

```text
Transformer 需要知道每个 bond-token 在整个分子拓扑中的位置。
```

### 6.2 Path Length Encoding

对 line graph 中任意两个 line nodes，计算 shortest path length。

然后把路径长度作为 attention bias：

```text
attention(i, j) += bias(shortest_path_length(i, j))
```

作用：

```text
让 self-attention 感知两个 bond-token 在拓扑上的距离。
```

### 6.3 Path Node Encoding

不仅看 path length，还看 shortest path 上经过的 line node 特征。

作用：

```text
让 attention 知道两个 bond-token 之间经过了哪些结构片段。
```

### 6.4 2D attention 公式含义

论文中的 2D attention 分数可以理解为：

```text
QK similarity
+ shortest path length bias
+ shortest path node feature bias
```

这比当前项目中的 GIN/GINE message passing 更像 Graphormer 类模型：

```text
不是只在邻居之间传消息，而是全局 self-attention + 图结构 bias。
```

## 7. 3D Line Graph Transformer 的结构编码

3D branch 的核心是距离和角度。

### 7.1 Atomic Distance Encoding

每个 3D line node 中包含原子对距离：

```text
line_node(uv) = atom pair type + distance_uv embedding
```

距离通过 Gaussian basis 展开。

### 7.2 Geometric Angle Encoding

Line graph 的 edge 表示两个相邻 bond。

在原始 3D 分子中，两个相邻 bond 对应一个角度：

```text
u-v-w angle
```

Galformer 把 angle embedding 加入 3D self-attention bias：

```text
attention(i, j) += geometric angle bias along shortest path
```

这使 3D branch 不只看 pairwise distance，还能显式建模 bond angle。

## 8. 预训练任务

Galformer 使用两个互补任务。

### 8.1 Masked Line Node Prediction

这是模态内部任务。

流程：

```text
随机选取一部分 2D / 3D line nodes
80% 替换成 MASK token
10% 替换成随机 line node
10% 保持不变
Transformer 编码上下文
MLP 预测原始 line node type
```

损失：

```text
L_mask_2d + L_mask_3d
```

它类似 BERT 的 MLM，但 token 不是单词，而是 bond-level line node。

相比当前项目的 Graph masked atom reconstruction：

```text
当前项目 mask atom。
Galformer mask bond/line node。
```

Galformer 的粒度更接近：

```text
键、相邻键、角度、局部片段。
```

这对聚合物很重要，因为很多性质由局部键型和连接片段决定。

### 8.2 Dual-Modality Contrastive Learning

这是 2D / 3D 模态间任务。

Galformer 在每个 line graph 中加入 virtual node：

```text
virtual node connected to all line nodes
```

virtual node 的输出作为 graph-level embedding。

正样本：

```text
同一个分子的 2D embedding 和 3D embedding
```

负样本：

```text
batch 中其他分子的 2D / 3D embedding
```

损失是双向 InfoNCE：

```text
2D -> 3D
3D -> 2D
```

作用：

```text
让 2D 表示吸收 3D 几何知识。
让 3D 表示和 2D 拓扑语义对齐。
```

## 9. 总损失

总目标：

```text
L_obj = lambda1 * (L_mask_2d + L_mask_3d) + lambda2 * L_contrastive
```

论文中：

```text
lambda1 = 1
lambda2 = 1
mask ratio = 0.4
temperature = 0.1
```

## 10. 预训练和下游设置

预训练数据：

```text
GEOM dataset
约 304k molecules
每个 molecule 使用 lowest-energy conformer
```

模型配置：

```text
12-layer 2D line graph transformer
12-layer 3D line graph transformer
hidden size = 768
attention heads = 12
batch size = 256
pretrain epochs = 50
peak learning rate = 2e-4
weight decay = 1e-6
```

下游设置：

```text
MoleculeNet 等 15 个 benchmark
9 个 classification
6 个 regression
scaffold split 8:1:1
3 个 random seeds
```

重要细节：

```text
下游任务只使用 pretrained 2D encoder。
3D geometry 被视为 pretraining 阶段的 privileged information。
```

也就是说，Galformer 的目标不是下游每次都需要 3D 构象，而是：

```text
用 3D 信息增强 2D encoder。
```

## 11. 实验结果

论文结果表明：

```text
Galformer 在 classification 和 regression 任务上整体优于 10 个 baseline。
```

对比方法包括：

```text
GraphCL
JOAO
Mix-Key
MTSSMol
GROVER
KPGT
DMP
3DInformax
GraphMVP
3D-PGT
```

结论：

```text
1. 使用 2D+3D 预训练通常优于纯 2D 预训练。
2. Transformer backbone 通常强于普通 message passing backbone。
3. Galformer 的 line graph transformer + mask + contrastive 组合效果最好。
```

论文也指出：

```text
Galformer 不是每个单独任务都第一，但整体平均最好。
```

原因包括：

```text
1. 它是通用预训练框架，不是单任务定制模型。
2. GEOM 预训练规模约 304k，小于部分 baseline 的百万级预训练数据。
```

## 12. 消融实验

### 12.1 Temperature

InfoNCE temperature 对性能有明显影响。

论文结论：

```text
tau = 0.1 整体最佳。
```

太小：

```text
训练 loss 下降快，但可能降低表示多样性。
```

太大：

```text
对比区分度不足，收敛慢。
```

### 12.2 Mask Ratio

最佳 mask ratio 是：

```text
0.4
```

这比很多早期图预训练方法高。

论文解释：

```text
有 2D/3D contrastive learning 提供额外互补信息，因此模型可以承受更难的 masked reconstruction。
```

### 12.3 Pretraining 和 Backbone

论文比较：

```text
without pretraining
without line node
vanilla transformer
Graphormer
Galformer
```

结论：

```text
预训练显著提升性能。
line graph transformer 明显优于普通 transformer / Graphormer 变体。
```

原因：

```text
line graph transformer 能显式建模 bond-level 和 angle-level 结构。
```

### 12.4 Mask Loss 和 Contrastive Loss

去掉 mask loss 会下降。

去掉 contrastive loss 也会下降。

不同 contrastive loss 中：

```text
InfoNCE 最好。
```

说明：

```text
模态内部 masked prediction 和模态间 contrastive alignment 都有必要。
```

### 12.5 Pretraining Data Scale

性能随预训练数据规模增加接近线性提升。

这说明：

```text
Galformer 是 scalable pretraining framework。
```

## 13. 可解释性分析

Galformer 使用 virtual node attention 做子结构解释。

方法：

```text
提取最后一层中指向 virtual node 的 attention weights
选择 top 20% line nodes
映射回分子中的高关注 substructures
```

在 BBBP 数据集上，模型关注到：

```text
permeable molecules: hydrophobic regions / trifluoromethyl group
impermeable molecules: polar regions / carboxylic acid / guanidine group
```

这些高 attention 片段与化学知识一致。

对当前项目的启发：

```text
如果未来改成 line graph transformer，可以直接做 fragment / bond-level attention 可视化。
当前项目的模态 attention 只能说明不同模态的权重，不能解释具体子结构。
```

## 14. 和当前 Uni-Poly-Plus 的对比

当前项目：

```text
SMILES encoder: RoBERTa
Graph encoder: GIN/GINE + star-linking
FP encoder: Morgan fingerprint projection
Geom encoder: SchNet / PaiNN
Fusion: modality-level self-attention + attention pooling
```

Galformer：

```text
2D line graph transformer
3D line graph transformer
masked line node prediction
2D/3D InfoNCE
下游只用 2D encoder
```

主要差异：

```text
1. 当前 Graph 是 atom-level GIN/GINE；Galformer 是 bond-level line graph transformer。
2. 当前 Geom 是独立 SchNet/PaiNN；Galformer 把 3D 转成 line graph，用 distance + angle bias。
3. 当前 fusion 是 supervised training 阶段的多模态融合；Galformer 是 pretraining 阶段的 2D/3D 对齐。
4. 当前下游继续使用 Geom；Galformer 下游可以只用 2D encoder。
5. 当前 attention 解释是模态级；Galformer attention 解释可以到 bond/substructure 级。
```

## 15. 对当前项目最有价值的借鉴

### 15.1 Graph 模块可以从 atom graph 转向 line graph

当前 `graph` 模态基于 atom-level message passing。

可借鉴 Galformer：

```text
把 bond / atom-pair 当成 token。
构建 line graph。
用 transformer 或 graph transformer 编码 line nodes。
```

对聚合物的价值：

```text
聚合物性质常由局部键型、片段连接、主链柔性、芳香连接、杂原子连接决定。
这些信息在 line graph 中更直接。
```

### 15.2 Star-linking 可以自然作用在 line graph 上

当前项目已经有：

```text
star-linking graph
backbone / attachment features
```

如果转成 line graph：

```text
star-linking edge 本身会成为一个 line node。
```

这比只在 edge_attr 里标记 `is_star_linking_edge` 更强，因为模型可以直接对这个“跨 repeat-unit 边界 bond token”建模。

建议未来设计：

```text
repeat unit atom graph
-> star-linking
-> line graph
-> line node includes bond type + endpoint atom features + is_star_linking_edge
```

### 15.3 Graph masked atom reconstruction 可升级为 masked line node prediction

当前项目：

```text
mask atom type
predict atom type
```

Galformer 建议：

```text
mask line node
predict bond/atom-pair type
```

对聚合物更合理：

```text
很多性质不是单个 atom 决定，而是 bond connection / local fragment 决定。
```

可以设计：

```text
mask line node -> predict bond type
mask line node -> predict endpoint atom pair type
mask line node -> predict backbone / side-chain / star-linking edge label
```

### 15.4 Geom 可以显式加入 angle 信息

当前 SchNet 主要依赖 pairwise distance。

Galformer 说明 angle 很重要：

```text
bond angle 是 3D molecular geometry 的关键 invariant feature。
```

对当前项目的启发：

```text
如果继续用 SchNet，缺少 angle 建模。
如果使用 PaiNN，方向向量能补充部分角度信息。
如果设计新 Geom encoder，可以考虑 distance + angle bias。
```

### 15.5 当前 multi-modal alignment 可以改成 2D/3D batch-negative InfoNCE

当前项目已有多模态 contrastive alignment。

Galformer 的具体经验：

```text
2D graph embedding 与 3D graph embedding 做双向 InfoNCE。
temperature 可从 0.1 开始。
```

这比 positive-only consistency 更强。

尤其是当前项目中的：

```text
Graph repeat-unit vs star-linking consistency
```

目前更像正样本相似约束，没有 batch negatives。

Galformer 提醒：

```text
如果做 representation alignment，最好加入负样本。
```

### 15.6 下游是否必须使用 Geom，可以重新评估

Galformer 的思路是：

```text
3D 是 pretraining privileged information。
下游可以只用 pretrained 2D encoder。
```

对当前项目，这提供一个可选路线：

```text
预训练阶段使用 Geom。
下游训练阶段只使用 smiles + graph + fp。
```

优点：

```text
减少下游 3D 构象质量对结果的影响。
训练更快。
避免测试阶段构象生成噪声。
```

缺点：

```text
如果某些任务确实依赖 3D，可能损失直接几何信息。
```

建议做消融：

```text
A. pretrain with geom, train with geom
B. pretrain with geom, train without geom
C. no geom pretrain, train without geom
D. no geom pretrain, train with geom
```

## 16. 当前项目不应直接照搬的部分

### 16.1 不建议直接用 12 层 768 hidden Galformer

论文模型较大：

```text
12 layers
hidden size 768
12 heads
```

当前聚合物数据规模较小，直接照搬可能：

```text
训练慢
过拟合
cache 和 batch 复杂度升高
```

更适合先做轻量版：

```text
4-6 layers
hidden size 256
4-8 heads
```

### 16.2 3D Line Graph 对聚合物 repeat unit 需要重新定义

普通小分子中：

```text
3D conformer 是完整分子构象。
```

当前项目中：

```text
Geom 是 star-substitution capped repeat unit。
```

因此 3D angle 和 distance 只代表 capped repeat unit 的局部构象，不是真实无限聚合物链构象。

所以不能直接声称：

```text
3D line graph = 聚合物真实链构象
```

只能说：

```text
3D line graph 提供局部几何先验。
```

### 16.3 下游只用 2D encoder 未必适合当前多模态主线

Galformer 下游只用 2D encoder，是因为其目标是让 2D encoder 吸收 3D knowledge。

当前 Uni-Poly-Plus 的目标是多模态融合，因此不一定要放弃下游 Geom。

更合理做法是先做消融，而不是直接删除 Geom 模态。

## 17. 推荐落地路线

### Phase A：轻量 line graph 分支

新增一个 Graph 选项：

```bash
--graph_encoder line_transformer
```

输入：

```text
star-linked atom graph
-> line graph
-> line node feature = endpoint atom features + bond features
-> line edge feature = adjacent bond relation
```

输出：

```text
graph-level embedding
```

先不加入 3D line graph，只验证 2D line graph 是否优于 GIN/GINE。

### Phase B：Masked line node prediction

替换或补充当前 masked atom reconstruction：

```text
mask line node
predict bond type / endpoint atom pair / star-linking label
```

这可以直接服务于聚合物主链和局部连接模式学习。

### Phase C：2D Graph / Geom contrastive 改成 batch-negative InfoNCE

当前 alignment 可加入更明确的 2D/3D 对齐：

```text
graph embedding vs geom embedding
positive = same SMILES
negative = other SMILES in batch
```

这和 Galformer 的 2D/3D contrastive task 对应。

### Phase D：加入 angle-aware Geom

如果继续发展 Geom 模态：

```text
3D capped repeat unit
-> bond angle extraction
-> angle embedding
-> distance + angle Geom encoder
```

可以走两条路线：

```text
1. 在现有 PaiNN 上加入 angle/distance auxiliary loss。
2. 新增 3D line graph transformer 分支。
```

### Phase E：子结构级解释

如果采用 line graph transformer，可以实现：

```text
virtual node attention
-> top attention line nodes
-> 映射回 polymer repeat-unit 中的 bonds/fragments
```

这比当前模态 attention 更有化学解释价值。

## 18. 和当前已读论文的关系

### 和 PerioGT

PerioGT 关注聚合物周期增强、repeat unit shift/repeat、motif 和多尺度描述符。

Galformer 关注：

```text
bond-level line graph transformer
2D/3D paired pretraining
```

对当前项目：

```text
PerioGT 更贴近 polymer periodicity。
Galformer 更贴近 Graph/Geom backbone 设计和预训练任务。
```

### 和 EDCL

EDCL 关注 3D denoising、adaptive noise 和 energy-guided pretraining。

Galformer 关注：

```text
2D/3D line graph transformer + contrastive + masked prediction
```

对当前项目：

```text
EDCL 可优化 Geom denoising。
Galformer 可优化 Graph/Geom 结构编码和对齐。
```

### 和 MFBERT + FGNN

MFBERT + FGNN 强调用 fingerprint BERT 和 2D/3D message passing。

Galformer 强调：

```text
line graph + transformer + structural attention bias
```

对当前项目：

```text
MFBERT 提示 FP 模态可加强。
Galformer 提示 Graph 模态可从 atom-level GIN 升级为 bond-level transformer。
```

### 和 MCMPP

MCMPP 关注多模态 cross-attention 融合。

Galformer 关注预训练阶段 2D/3D 表示对齐。

对当前项目：

```text
MCMPP 对 fusion module 有启发。
Galformer 对 graph/geom encoder 和 pretraining task 有启发。
```

## 19. 总结

Galformer 对当前 Uni-Poly-Plus 最重要的启发是：

```text
1. Graph 不一定要停留在 atom-level GIN/GINE。
2. 对聚合物来说，bond/line-node 级别的建模可能更贴近局部结构和主链连接。
3. 2D/3D 预训练不应该只有模态对齐，还应该保留每个模态内部的 masked reconstruction。
4. Geometry 不只包含 distance，angle 也是重要的 invariant structural bias。
5. Attention 可解释性可以从模态级提升到 fragment/bond 级。
```

当前项目最建议优先尝试的方向：

```text
先做 2D star-linked line graph transformer + masked line node prediction，
再考虑 3D line graph 和 angle-aware Geom。
```

原因：

```text
它最贴近当前已有 star-linking graph 逻辑，
实现风险低于完整复现 Galformer，
并且能直接增强当前偏弱的 graph 模态。
```
