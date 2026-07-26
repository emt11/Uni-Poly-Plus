# MFBERT + FGNN 复现导向分析

分析对象：`Structure-Based_Robust_Fractal_Graph_Neural_Network_with_Molecular_Fingerprint_BERT_for_Molecular_Property_Prediction.pdf`

论文题目：Structure-Based Robust Fractal Graph Neural Network with Molecular Fingerprint BERT for Molecular Property Prediction

期刊：IEEE Transactions on Emerging Topics in Computational Intelligence, 2026

本文档按“复现该论文并作为 Uni-Poly-Plus 改进分支”的目标整理。重点不是只解释思想，而是拆出数据处理、MFBERT、FGNN、3D/Coulomb、W-DRO、预训练、fine-tune 和迁移到当前聚合物项目的实现路径。

## 1. 论文整体目标

论文要解决三个问题：

```text
1. 标注分子性质数据不足。
2. 分子性质标签分布不均衡。
3. 常规 GNN 主要依赖 2D graph，忽略 3D conformation。
```

因此提出组合模型：

```text
MFBERT + FGNN + W-DRO
```

完整流程：

```text
SMILES
-> ECFP substructure sequence
-> MFBERT pretraining on unlabeled molecules
-> ECFP contextual embeddings as atom/node features
-> FGNN with 2D + 3D message passing
-> coordinate update
-> graph pooling
-> MLP prediction
-> W-DRO robust optimization
```

这篇论文不是传统“多模态最后拼接”。它把 fingerprint 表示、2D graph、3D geometry 融入同一个 graph network 内部。

## 2. 复现时的模块拆分

如果按论文复现，应拆成以下模块：

```text
A. ECFP sequence generator
B. MFBERT pretraining module
C. MFBERT-to-node embedding mapping
D. 3D conformer + Coulomb matrix generator
E. Edge feature generator
F. FGNN block
G. W-DRO optimizer wrapper
H. Two-stage fine-tuning
```

对当前 Uni-Poly-Plus 来说，不建议一次性替换所有模块。更合理的是把它作为一个可控分支：

```text
baseline: current Uni-Poly-Plus GINE + SchNet/PaiNN + modality attention
branch:   MFBERT-local fingerprint embedding + FGNN-style 2D/3D interaction
```

## 3. 数据处理

### 3.1 原论文数据

MFBERT 预训练：

```text
ZINC > 90 million SMILES
```

FGNN 预训练：

```text
QM9 trusted conformations
```

下游测试数据：MoleculeNet 七个数据集：

```text
ESOL
FreeSolv
Lipophilicity
BACE
MUV
Tox21
SIDER
```

训练/验证/测试划分沿用 MoleculeNet setup，并报告 10 次独立运行结果。

### 3.2 当前项目数据差异

当前 Uni-Poly-Plus 使用聚合物 repeat-unit SMILES / P-SMILES：

```text
data/raw/smi_all.csv      当前 no-tg 8-task 预训练/缓存源
data/raw/smi_<task>.csv   下游单任务数据
```

当前任务：

```text
eat eea egb egc ei eps nc xc
```

差异：

```text
原论文：普通小分子 SMILES，不含 *。
当前项目：聚合物 repeat unit，可能含两个或多个 * attachment atoms。
```

因此复现时必须处理 `*`：

```text
Graph: 可以用 star-linking 或 repeat-unit graph。
Geom: 不建议 star-linked closed ring，当前更适合 star-substitution capped repeat unit。
ECFP/MFBERT: 需要决定 * 如何参与 Morgan local environment。
```

建议当前项目中：

```text
ECFP/Morgan local id 对 Graph 使用 star-linked graph 或 repeat-unit graph 分别做 ablation。
Geom 继续使用 capped repeat unit。
```

## 4. MFBERT：Molecular Fingerprint BERT

### 4.1 核心思想

普通 SMILES-BERT：

```text
SMILES tokens -> BERT
```

MFBERT：

```text
ECFP substructure identifiers -> BERT
```

也就是：先用 Morgan/ECFP 生成局部子结构 identifier，再把 identifier 当成 NLP 中的词 token 输入 BERT。

### 4.2 ECFP sequence generation

论文使用 Morgan algorithm。ECFP 定义为从 radius 0 到 radius r 的 atomic identifiers 集合。

Algorithm 1 可复现为：

```text
Input: molecule M, radius r
Output: ECFP identifier sequence/set

1. 给每个 atom 一个初始 32-bit identifier。
2. 把每个 atom 的 identifier 加入集合 S。
3. 对 i = 1..r:
   a. 对每个 atom 创建 feature list:
      [i, current_identifier, bond_order_1, neighbor_identifier_1, ...]
   b. hash feature list 得到新的 atom identifier。
   c. 如果新 identifier 不在 S，则加入 S。
4. 返回 S。
```

论文实验结论：

```text
r = 1 最好
```

原因：

```text
r = 0 只表达原子自身。
r = 1 表达原子 + 直接邻居 + bond type，足够捕获很多局部官能团。
r 更大时复杂度和噪声增加，可能过拟合。
```

### 4.3 MFBERT 输入

输入序列：

```text
[ecfp_id_1, ecfp_id_2, ..., ecfp_id_n]
```

Embedding：

```text
token embedding + position embedding
```

论文明确省略 segment embedding，因为分子 ECFP 序列没有类似句子 A/B 的结构。

Transformer：

```text
multi-head self-attention encoder layers
```

注意：PDF 中 Table III 给了 MFBERT 具体超参，但当前文本抽取无法可靠恢复表格单元。复现时需要从 PDF 原表人工核对：

```text
vocabulary size
hidden dimension
number of hidden layers
number of attention heads
maximum sequence length
MLM mask ratio
optimizer settings
```

已确认可复现设置：

```text
ECFP radius r = 1
optimizer = Adam
pretraining task = MLM
pretraining corpus = ZINC > 90M SMILES
```

### 4.4 MFBERT 预训练任务

论文只使用：

```text
Masked Language Modeling, MLM
```

复现逻辑：

```text
ECFP sequence
-> randomly mask some ECFP identifiers
-> Transformer encoder
-> predict masked identifiers
-> cross entropy loss
```

形式：

```text
loss_mfbert = CE(predicted_token_distribution, true_ecfp_id)
```

### 4.5 MFBERT 输出如何进入 FGNN

论文不是简单拿整分子的 CLS embedding，而是用 MFBERT embedding 构造 graph node feature。

文中描述：

```text
initial node features are extracted by averaging the ECFP embeddings of the atom and its environmental contexts across different radii
```

复现理解：

```text
对每个 atom，收集和该 atom 对应的 ECFP identifiers：radius 0..r。
用 MFBERT 得到这些 identifier 的 contextual embeddings。
对该 atom 的相关 embeddings 做平均。
得到 atom-level initial node feature h0_i。
```

实现要点：

```text
需要保存 atom -> ECFP token ids 的映射。
不能只保存全分子的去重 ECFP set，否则无法回填到 atom node。
```

当前项目如果实现轻量版，可以先做：

```text
Morgan atom environment id -> nn.Embedding -> append to graph node feature
```

完整 MFBERT 则需要：

```text
atom_to_ecfp_ids
batch ECFP token sequence
MFBERT contextual embedding
scatter/mean back to atom nodes
```

## 5. FGNN：Fractal Graph Neural Network

### 5.1 Fractal block 思想

论文中的 fractal 不是化学分形结构，而是网络结构：

```text
同一个 block 内有多条不同深度的 GNN sub-path。
浅路径响应快，深路径表达强。
多个路径输出通过 join layer 聚合。
训练时使用 drop-path 随机关闭部分路径。
```

复现结构：

```text
input node state
-> path_1: 1 layer GNN
-> path_2: 2 layer GNN
-> path_3: deeper GNN
-> drop-path
-> join by mean
-> output node state
```

论文 join layer 采用：

```text
mean operator
```

drop-path 约束：

```text
随机关闭若干路径，但至少保留一条路径。
```

### 5.2 FGNN 输入

FGNN 输入包括：

```text
h0_i: MFBERT atom/node embedding
e0_ij: predefined edge features
p_i: 3D coordinate
CM_i: Coulomb matrix related feature
```

Embedding layers：

```text
h_i  = phi_n(h0_i)
e_ij = phi_e(e0_ij)
c_i  = phi_c(CM_i)
```

这里 `phi_n`, `phi_e`, `phi_c` 都可以实现为 MLP / Linear + activation。

### 5.3 Edge features

论文使用 18-dimensional molecular edge features，见 PDF Table II。

文本抽取无法可靠恢复表 II 单元格。复现时需要从 PDF 原表核对。按分子 GNN 常见实现，可先覆盖：

```text
bond type one-hot
conjugated flag
ring flag
stereo
bond direction
```

但若目标是严格复现，必须按 Table II 的 18 维定义实现。

当前项目已经有 Graph edge feature，可以：

```text
先保留当前 edge_attr。
再对齐论文 18-dim edge feature 做 ablation。
```

## 6. 3D conformer 与 Coulomb matrix

### 6.1 3D conformer 生成

论文使用 RDKit：

```text
SMILES -> RDKit distance geometry
+ experimental torsion angle preferences
+ flat ring constraints / basic chemical knowledge
-> 3D conformation
```

这接近 RDKit ETKDG 系列流程。

当前项目已使用多构象、MMFF/UFF fallback 和 star-substitution capped repeat unit。对聚合物 repeat unit 更合适。

### 6.2 Coulomb matrix

Coulomb matrix 表达原子间类库仑作用：

```text
C_ij = Z_i Z_j / ||R_i - R_j||, i != j
C_ii = 0.5 * Z_i^2.4
```

论文还提到：

```text
Coulomb matrix based on 3D coordinates and Gasteiger charges
```

因此复现时可实现两种版本：

```text
A. 原始 Coulomb matrix: atomic number + distance
B. charge-aware Coulomb feature: Gasteiger charge + distance
```

当前项目中要注意：

```text
P-SMILES 中的 * 不能直接计算 Coulomb matrix。
必须先 cap/substitute。
```

推荐当前项目实现：

```text
Geom capped repeat unit -> compute pairwise distance / optional Coulomb feature
Graph star-linking edge 不使用虚假的 3D Coulomb distance
```

## 7. FGNN 的 2D message passing

论文 2D message passing 使用化学键邻居：

```text
m_ij^2D = phi_2D(h_i, h_j, e_ij)
m_i^2D = sum_{j in N_i} m_ij^2D * rbf_ij
h_i = h_i + m_i^2D
```

其中：

```text
N_i = 2D graph 中与 i 成键的邻居
rbf_ij = 由 3D 距离 d_ij 展开的 RBF 权重
```

关键点：

```text
2D message 沿化学键传播，但被 3D distance RBF 调制。
```

这说明 FGNN 不是 Graph encoder 和 Geom encoder 最后融合，而是在每层 2D message 中引入 3D 距离。

当前项目可借鉴：

```text
对真实化学键 edge，加入 bond distance RBF。
对 star-linking 新增周期边，只保留 is_star_linking_edge，不强行赋真实距离。
```

## 8. FGNN 的 3D message passing

3D message passing 使用空间节点子集 `S_R`：

```text
m_ij^3D = phi_3D(h_i, h_j, c_i)
m_i^3D = sum_{j in S_R} m_ij^3D * rbf_ij
h_i = h_i + m_i^3D
```

其中：

```text
S_R = 随机选择的空间节点子集
c_i = Coulomb matrix embedding
rbf_ij = 3D distance RBF
```

论文选择随机子集而不是所有空间节点，原因：

```text
降低计算量。
避免节点特征过度相似。
```

复现策略：

```text
for each batch/molecule:
  sample spatial neighbor subset S_R
  compute pairwise distances from i to selected nodes
  compute RBF(distance)
  compute 3D messages
```

当前项目如果迁移到 SchNet/PaiNN 主线，不必完整复现 `S_R`。如果做 FGNN 分支，则应实现它。

## 9. Coordinate update module

论文加入坐标更新：

```text
p = p + Delta p
Delta p = phi_p(h, p)
```

然后下一层重新计算：

```text
RBF distance
Coulomb matrix features
message passing weights
```

复现注意：

```text
phi_p 需要保证平移/旋转相关处理合理。
若直接用 MLP(h, p) 输出 delta，严格等变性可能不足。
论文声称保持 rotation/translation/permutation equivariance，但实际实现需要仔细设计。
```

当前项目已有：

```text
Geom coordinate denoising
pred_delta = coord_head(node_rep)
pred_pos = noisy_pos + pred_delta
```

因此这部分最接近当前 Geom 预训练任务。若借鉴 FGNN，可增强为：

```text
bond length reconstruction
angle reconstruction
force-like delta prediction
多构象一致性 loss
```

## 10. RBF distance expansion

论文对距离使用 RBF：

```text
rbf_ij = phi_rbf(exp(-gamma * ||d_ij - mu||^2))
```

其中：

```text
d_ij = ||p_i - p_j||
gamma, mu = RBF 参数
phi_rbf = transformation layer
```

复现建议：

```text
使用固定 centers mu: linspace(0, cutoff, num_rbf)
gamma = 1 / delta^2
RBF output -> Linear/MLP -> message weight
```

当前项目的 SchNet/PaiNN 已有 RBF 思想。Graph 分支若加入 3D edge distance，可复用类似 RBF expansion。

## 11. Readout 与预测

论文 readout：

```text
g = mean_pool(h_i)
y_hat = MLP(g)
```

即：

```text
node features -> mean pooling -> graph feature -> MLP prediction
```

当前项目使用多模态 fusion 后 MLP。若复现 FGNN 分支，可以：

```text
FGNN graph embedding -> UniEncoderAttention 作为 graph modality
```

或者：

```text
FGNN 单模态直接预测，用作 ablation baseline
```

## 12. W-DRO 优化

### 12.1 论文目标

论文把标签分布不均衡问题形式化为 DRO：

```text
min_theta max_k mean_loss(M_k)
```

其中 `M_k` 是按标签划分的子集。

直观含义：

```text
不要只优化全局平均 loss；
要关注表现最差的标签区间 / 子集。
```

### 12.2 Algorithm 2 复现流程

论文 Algorithm 2：

```text
1. Pre-train MFBERT on unlabeled data.
2. Dataset partition and data processing.
3. For each epoch:
   For each batch:
     a. Calculate ECFPs from SMILES.
     b. Embed ECFPs with pre-trained MFBERT.
     c. Forward FGNN and predict y_hat.
     d. Calculate subgroup losses L_k.
     e. Calculate gradients g_b.
     f. Solve Lagrange multiplier lambda.
     g. Use Delta theta = -g_b^T lambda.
     h. Update parameters.
4. Learning rate decay alpha = alpha * 0.95.
```

其中 Lagrange multiplier 求解：

```text
min_lambda 1/2 lambda^T g_b g_b^T lambda - L^T lambda
s.t. sum(lambda_k)=1, lambda_k >= 0
```

### 12.3 当前项目可实现的 W-DRO-lite

完整 W-DRO 需要每个 subgroup 的梯度并求约束优化，工程复杂。

建议当前项目先实现轻量版本：

```text
回归任务：按训练集 y 分位数分 K 个 bin。
分类任务：按 label class 分组。
每个 batch 计算各 bin mean loss。
最终 loss = mean_loss + lambda * max_bin_loss。
```

或者：

```text
loss = sum_k softmax(alpha * bin_loss_k) * bin_loss_k
```

这样不需要求 Lagrange multiplier，但保留“关注高损失标签区间”的核心思想。

## 13. 原论文训练流程

### 13.1 MFBERT pretraining

```text
Input: ZINC SMILES
Process: Morgan r=1 -> ECFP sequence
Objective: MLM
Optimizer: Adam
Output: MFBERT checkpoint
```

### 13.2 FGNN pretraining

```text
Input: QM9 trusted conformations
Process: 3D conformer + edge features + Coulomb matrix
Output: FGNN checkpoint
```

PDF 没有在正文中清晰给出 FGNN 预训练的具体 self-supervised loss；只明确说明使用 QM9 trusted conformations 预训练 FGNN 以获得良好初始化。复现时需要查作者代码或补充合理任务。

可选复现任务：

```text
coordinate denoising
pairwise distance reconstruction
QM9 property prediction pretraining
masked node / edge reconstruction
```

当前项目已经有：

```text
Geom coordinate denoising
pairwise distance reconstruction
Graph masked atom reconstruction
```

### 13.3 Downstream training two stages

论文下游训练分两阶段：

```text
Stage 1: freeze MFBERT, train FGNN with W-DRO.
Stage 2: fine-tune MFBERT + FGNN with W-DRO.
```

优化器：

```text
Adam
initial lr = 0.001
lr decay = 0.95
weight decay = 1e-5
```

硬件/环境：

```text
Python 3.8
PyTorch 1.8.1
5 Tesla T4 GPUs
```

## 14. 严格复现清单

若目标是尽可能复现论文，需要实现：

```text
1. Morgan/ECFP radius=1 token sequence generator。
2. atom -> ECFP token mapping。
3. MFBERT tokenizer/vocab/mask logic。
4. MFBERT MLM pretraining。
5. MFBERT contextual embedding 回填 atom node feature。
6. 18-dimensional edge features, matching PDF Table II。
7. RDKit 3D conformer generation with ETKDG-like settings。
8. Gasteiger charge + Coulomb matrix。
9. Distance RBF expansion。
10. FGNN fractal block with multi-depth paths。
11. Drop-path with at least one path kept。
12. 2D message passing weighted by RBF distance。
13. 3D message passing over sampled spatial subset S_R。
14. Coordinate update module。
15. Mean pooling + MLP prediction。
16. MFBERT pretraining on large unlabeled SMILES。
17. FGNN pretraining on QM9 conformations。
18. Downstream stage 1: frozen MFBERT。
19. Downstream stage 2: fine-tune MFBERT + FGNN。
20. W-DRO optimizer or W-DRO-lite。
```

## 15. 迁移到 Uni-Poly-Plus 的推荐实现路线

当前项目不是小分子，需要保留聚合物特性：

```text
P-SMILES
star-linking
backbone / attachment feature
star-substitution capped geom
multi-modal fusion
```

因此推荐分阶段改造。

### Phase A：低风险复现核心思想

目标：不替换当前主模型，只增强输入和 loss。

实现：

```text
1. Graph node 增加 Morgan local environment id embedding。
2. 下游训练增加 W-DRO-lite。
3. 下游 fine-tune 改为 freeze -> unfreeze 两阶段。
```

优点：

```text
代码侵入小。
能直接验证论文最有价值的两个点：contextual/local fingerprint + robust label optimization。
```

### Phase B：Graph-Geom 边级交互

目标：借鉴 FGNN 的 2D message 乘以 3D distance RBF。

实现：

```text
1. 为原始真实化学键计算 3D bond distance。
2. 对真实 edge 增加 RBF(distance) edge feature。
3. star-linking edge 只标记 periodic edge，不赋真实 3D 距离。
4. 修改 GINE edge encoder 适配新增 edge_attr。
```

注意：

```text
Graph 使用 star-linking，Geom 使用 capped repeat unit，两者节点可能不完全一致。
必须维护 atom mapping。
```

### Phase C：FGNN 分支

目标：复现论文网络作为新的 graph/geom joint encoder。

实现：

```text
1. 新增 src/modules/fgnn.py。
2. 新增 graph_backend 或 graph_encoder_type = fgnn。
3. 输入为 node features + edge features + pos3d + Coulomb/RBF。
4. 输出 graph embedding。
5. 可作为 graph modality，也可替代 graph+geom 两个模态。
```

建议先做：

```text
FGNN 单模态 baseline
```

再做：

```text
FGNN + SMILES + FP fusion
```

### Phase D：完整 MFBERT

目标：复现 fingerprint BERT。

实现：

```text
1. 构建 ECFP token vocab。
2. 实现 MLM dataset。
3. 预训练 MFBERT。
4. 将 MFBERT contextual embedding 回填到 atom node。
5. 下游支持 frozen/fine-tuned MFBERT。
```

当前不建议优先做完整 MFBERT，因为：

```text
ZINC 90M 预训练成本高。
当前聚合物数据规模小。
轻量 Morgan embedding 更适合作为第一步。
```

## 16. 对当前项目的具体建议

### 16.1 最应该先做

```text
1. Morgan local environment id embedding appended to Graph node feature。
2. W-DRO-lite for regression label imbalance。
3. freeze -> unfreeze downstream training。
```

原因：

```text
这三项和论文贡献直接对应，且不破坏当前 star-linking + SchNet/PaiNN 主线。
```

### 16.2 谨慎做

```text
Graph edge 加 3D distance RBF。
```

原因：

```text
这需要解决 star-linking graph 和 capped geom 的 atom mapping。
实现不严谨会引入错误空间距离。
```

### 16.3 暂不建议完整照搬

```text
完整 MFBERT on ZINC 90M。
完整 FGNN 替换 GINE + SchNet/PaiNN。
完整 Lagrange multiplier W-DRO。
```

原因：

```text
工程量大，且当前项目已有聚合物专用构图和几何策略。
直接替换会让实验变量过多，不利于判断收益来源。
```

## 17. 和当前 Uni-Poly-Plus 的对照

| 模块 | 原论文 | 当前项目 | 推荐迁移 |
|---|---|---|---|
| 分子输入 | 小分子 SMILES | 聚合物 P-SMILES / repeat unit | 保留 P-SMILES 特化处理 |
| Fingerprint | ECFP token -> MFBERT | Morgan FP 全局向量 | 先加 Morgan local id embedding |
| Graph | FGNN fractal block | GINE + star-linking + backbone | 中期可加 FGNN 分支 |
| 3D | FGNN 3D message + coordinate update | SchNet / PaiNN | 保留主线，借鉴 RBF/Coulomb/denoising |
| 2D/3D 融合 | message-passing 内部融合 | encoder 后 fusion | 可加 edge distance RBF 或 graph-geom cross fusion |
| 预训练 | MFBERT + QM9 FGNN | Graph/Geom + alignment | 当前方向一致，可补 fingerprint MLM |
| 优化 | W-DRO | 普通 loss | 先做 W-DRO-lite |
| Fine-tune | freeze MFBERT -> unfreeze all | 直接下游训练 | 可加 freeze -> unfreeze |

## 18. 复现风险点

```text
1. PDF 表 II / III / IV 的表格需要人工核对，文本抽取无法可靠恢复。
2. MFBERT 需要 atom-to-ECFP-token mapping，否则无法构造 node feature。
3. P-SMILES 的 * 会影响 Morgan id、3D conformer、Coulomb matrix。
4. star-linking edge 不是真实 3D 距离，不能直接套 FGNN 的 RBF edge weighting。
5. Coulomb matrix 对端基替换敏感。
6. 完整 W-DRO 的 Lagrange multiplier 求解可能训练不稳定。
7. 完整 FGNN 与当前 SchNet/PaiNN 功能重叠，需要做清晰 ablation。
```

## 19. 建议实验顺序

```text
Experiment 0: 当前 baseline，star-linking + SchNet/PaiNN + modality self-attention。
Experiment 1: + Morgan local environment embedding。
Experiment 2: + W-DRO-lite。
Experiment 3: + freeze -> unfreeze downstream fine-tuning。
Experiment 4: + real-bond 3D distance RBF edge feature。
Experiment 5: FGNN single-encoder branch。
Experiment 6: lightweight MFBERT / fingerprint Transformer。
```

每一步只改一个主要变量，避免无法判断收益来源。

## 20. 结论

如果以复现论文为目标，核心不是简单“加一个 3D 模型”，而是复现以下链条：

```text
ECFP tokenization
-> MFBERT contextual fingerprint embedding
-> atom-level node feature injection
-> fractal multi-path GNN
-> 2D bond message with 3D RBF weighting
-> 3D spatial message with Coulomb feature
-> coordinate update
-> W-DRO robust optimization
-> frozen-to-finetuned training
```

对当前 Uni-Poly-Plus 的最合理改进方向是选择性吸收，而不是一次性替换：

```text
短期：Morgan local embedding + W-DRO-lite + freeze/unfreeze。
中期：真实化学键距离 RBF 注入 Graph edge。
长期：实现 FGNN 分支和 lightweight MFBERT。
```
