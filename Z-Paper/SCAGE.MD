# SCAGE 论文分析：Self-Conformation-Aware Graph Transformer

分析对象：`s41467-025-59634-0.pdf`

论文题目：A self-conformation-aware pre-training framework for molecular property prediction with substructure interpretability

期刊：Nature Communications, 2025

代码：`https://github.com/KazeDog/scage`

本分析同时基于论文 PDF 和 `KazeDog/scage` 原仓库源码审计。源码基线为提交
`82bcbb4647e31bf0d413a317e69a2526df75ce01`。除论文方法外，本文明确区分：

```text
原仓库实际执行的逻辑
论文层面的设计目标
当前 Uni-Poly-Plus 的 Polymer-SCAGE 扩展
```

## 1. 论文核心问题

论文关注分子性质预测中的两个问题：

```text
1. 普通 2D graph model 缺少 3D conformational information。
2. 很多模型性能虽高，但难解释具体哪些 functional groups / substructures 影响性质。
```

作者提出：

```text
SCAGE = Self-Conformation-Aware Graph Transformer
```

核心思想是：

```text
用 2D molecular graph 作为主输入。
通过 MMFF 生成稳定 3D conformer。
在 Graph Transformer 中加入 multiscale conformational attention mask。
通过四个预训练任务学习结构到功能的多层语义。
用 attention 和 functional group 解释 molecular property / activity cliff。
```

## 2. SCAGE 的整体框架

SCAGE 遵循 pretraining-finetuning 范式：

```text
PubChem drug-like molecules
-> RDKit / MMFF 生成 stable conformation
-> Graph Transformer + MCL module
-> M4 multitask pretraining
-> downstream property prediction / activity cliff prediction
-> attention-based interpretability
```

整体由三个核心设计组成：

```text
1. MCL: Multiscale Conformational Learning
2. M4: four-task multitask pretraining
3. Atomic-level functional group annotation
```

## 3. 输入与构象处理

SCAGE 首先把分子表示成 2D molecular graph：

```text
node = atom
edge = chemical bond
```

然后使用 MMFF 获得稳定构象：

```text
SMILES / molecule
-> molecular graph
-> MMFF conformer generation
-> select lowest-energy conformation
```

论文说明：

```text
最低能量构象并不总是在所有任务上最优，
但多数情况下效果最好，因此作为主设置。
```

这一点和当前聚合物任务需要谨慎区分：

```text
小分子的 lowest-energy conformer 语义相对清晰。
聚合物 repeat unit / oligomer 的 lowest-energy conformer 不一定代表材料实际链构象。
```

## 4. Graph Transformer Backbone

SCAGE 使用 Graph Transformer 作为骨干。

每层包括：

```text
multi-head self-attention
feed-forward network
residual connection
GELU
normalization
```

基础 attention 和普通 Transformer 类似：

```text
Q = H W_Q
K = H W_K
V = H W_V
attention = softmax(QK^T / sqrt(d)) V
```

SCAGE 的关键不是这个 backbone 本身，而是 MCL 如何改造 attention。

## 5. Multiscale Conformational Learning

MCL 是 SCAGE 的核心结构模块。

问题背景：

```text
Transformer self-attention 让每个 atom 可以看全局。
但层数过多后，node representation 容易逐渐同质化。
同时，分子性质既依赖局部相互作用，也依赖中远程空间结构。
```

SCAGE 使用 3D conformational distance 生成多个 receptive field。

对每个 molecule：

```text
1. 计算 atom-atom distance matrix。
2. 根据距离百分位数生成 threshold。
3. 每个 threshold 对 attention score 做 mask。
4. 多个 threshold 对应多个尺度。
5. concat 多尺度 attention 输出。
```

形式上：

```text
receptive field = concat(threshold_1, threshold_2, ..., threshold_n)
```

每个 threshold 是分子内部距离分布的百分位数，而不是固定 Å 数。

这点很重要：

```text
小分子和大分子的空间尺度不同。
固定距离阈值会导致有些分子几乎全局可见，有些分子只剩局部。
百分位阈值可以自适应分子大小和构象尺度。
```

## 6. 2D Distance vs 3D Distance

论文比较了两类距离：

```text
2D topological distance:
  molecular graph shortest path

3D conformational distance:
  atom coordinates in conformer
```

实验显示：

```text
基于 3D conformational distance 的 MCL 通常优于 2D topological distance。
```

作者进一步比较单阈值、双阈值、三阈值、多尺度组合：

```text
多数数据集上，20% - 60% 左右的距离尺度组合效果较好。
过小尺度只看局部。
过大尺度更接近全局平均。
多尺度比固定阈值更稳。
```

## 7. M4 多任务预训练

SCAGE 的预训练框架叫 M4，包含四个任务：

```text
1. Molecular fingerprint prediction
2. Functional group prediction
3. 2D atomic distance prediction
4. 3D bond angle prediction
```

这四个任务分别覆盖：

```text
fingerprint:
  graph-level structural semantics

functional group:
  atom-level chemical prior

2D distance:
  global topology

3D bond angle:
  local geometric conformation
```

论文强调：

```text
单个任务不足以学到完整分子表示。
四个任务组合后效果最好。
```

## 8. 任务一：Molecular Fingerprint Prediction

SCAGE 使用 Morgan fingerprint：

```text
2048-bit Morgan fingerprint
```

任务形式：

```text
graph-level molecular representation
-> MLP
-> predict fingerprint bits
-> binary cross entropy
```

作用：

```text
把传统 cheminformatics structural descriptor 蒸馏进 neural representation。
```

和 MolGT 的 KGPC 不同：

```text
MolGT:
  fingerprint -> cluster prototype -> representation predicts prototype

SCAGE:
  representation -> directly predicts Morgan fingerprint bits
```

对当前 Uni-Poly-Plus 来说，两种都可借鉴。

## 9. 任务二：Functional Group Prediction

Functional group prediction 是 SCAGE 最重要的化学先验任务。

论文收集了 190 类 functional groups，来源包括：

```text
KANO
GROVER
DayLight
other motif databases
```

SCAGE 不只是预测 molecule-level functional group count，而是为每个原子构造 functional-group multi-hot target：

```text
一个 atom 可以同时属于多个 functional groups。
未匹配 functional group 的 atom 使用默认类别 0。
```

任务形式：

```text
atom representation
-> MLP
-> 190-dimensional logits
-> multi-label BCEWithLogits loss
```

为什么 atom-level 很重要：

```text
molecule-level functional group 只能告诉模型有哪些 group。
atom-level label 告诉模型 group 在哪里，哪些 atom 属于哪个 group。
```

论文的可解释性实验显示，atom-level FG task 比 molecule-level FG number task 更能让 attention 聚焦关键 substructure。

## 10. Functional Group Annotation Algorithm

论文提出一个 functional group annotation algorithm。

流程：

```text
1. 从 Wikipedia / Daylight / motif database 收集 functional group pattern。
2. 对每个 atom，匹配所有包含该 atom 的 functional groups。
3. 对严格子集关系做最大覆盖优先清理，避免较小 SMARTS 匹配重复覆盖较大功能团。
4. 对没有包含冲突的独立功能团继续保留多个标签。
5. 最终把每个 atom 的 functional-group ID 列表转换成 multi-hot vector。
```

示例：

```text
一个 atom 可能同时属于:
  carboxide
  ester group
  hexatomic ring

算法会移除被更大匹配严格包含的冗余标签，但不会强制所有情况只剩一个 label。
```

这一点对聚合物非常有价值，因为聚合物性质常常由功能团位置和 backbone / side-chain 关系决定。

## 11. 任务三：2D Atomic Distance Prediction

2D atomic distance 是 molecular graph 中任意两个 atom 的 shortest path。

SCAGE 预测完整 atom pair shortest path matrix：

```text
SP(u, v) = shortest path distance in 2D graph
cap distance > 20 to 20
```

任务形式：

```text
atom pair representation
-> predict shortest path bin
-> focal loss
```

作用：

```text
让模型显式学习全局拓扑连接关系。
```

这和 Graphormer / Uni-Mol 的 pair-distance bias 思想相近，但 SCAGE 把它作为预训练监督任务。

## 12. 任务四：3D Bond Angle Prediction

3D bond angle prediction 用构象中相邻化学键的角度作为监督。

流程：

```text
MMFF conformer
-> calculate bond angle
-> discretize angle [0, pi] into 20 bins
-> predict angle bin
-> focal loss
```

这个任务让模型从 2D graph 输入中学习 3D 局部几何规律。

与直接预测 atom coordinate 不同：

```text
bond angle 更局部、更稳定。
coordinate prediction 受旋转、平移、构象选择影响更大。
```

对于聚合物，角度任务可以扩展为：

```text
backbone bond angle
torsion angle
star-to-star direction angle
side-chain orientation angle
```

其中 torsion 可能比 bond angle 更贴近聚合物柔性和 Tg。

## 13. Dynamic Adaptive Multitask Learning

四个预训练任务的 loss scale 和难度不同。

SCAGE 使用 Dynamic Adaptive Multitask Learning 自动调整任务权重。

核心思想：

```text
根据每个任务 loss 的下降速度和数值范围调整权重。
下降慢 / 学习难的任务获得更合适的优化权重。
避免某个任务 loss 主导总目标。
```

这对多任务预训练很重要，因为：

```text
fingerprint BCE
functional group classification
2D distance focal loss
3D angle focal loss
```

数值尺度天然不同。

源码中的计算不是简单的可学习 uncertainty weight，而是基于历史 loss 的动态归一化：

```text
1. 保存每个任务初始阶段的平均 loss，作为归一化基线。
2. 当前任务 loss 除以该历史基线，得到 normalized loss。
3. 比较相邻历史窗口的 loss 变化率。
4. 对变化率做 softmax，得到 detached task weights。
5. total loss = sum(normalized_loss_i * dynamic_weight_i)。
6. 每个 step 更新滚动 loss buffer。
```

因此原实现的动态权重同时做了两件事：先消除任务量纲，再根据近期优化速度重新分配权重。它不是四个 raw loss 直接乘权重后求和。

对 Uni-Poly-Plus 的启发是：

```text
如果同时做 mask atom、mask motif、fingerprint、polymer class、geometry denoising，
不要简单手调固定 loss weights。
```

## 14. 预训练数据与下游任务

预训练数据：

```text
约 5M PubChem drug-like molecules
```

原仓库预训练配置的关键默认值为：

```text
mode = pretrain_bin
embedding_dim = 512
num_layers = 6
num_heads = 16
ffn_hidden = 256
dist_bar = [20, 50]
epochs = 100
learning_rate = 5e-5
weight_decay = 1e-4
tasks = finger, sp, angle, fg
```

代码配置默认 batch size 为 40；README 的硬件/时间说明中另有 batch size 16 的描述，两者并不完全一致。复现实验应以实际启动配置和 checkpoint 配套参数为准。

下游 molecular property：

```text
BBBP
Tox21
SIDER
ClinTox
ToxCast
BACE
FreeSolv
Lipophilicity
ESOL
```

数据划分：

```text
scaffold split
random scaffold split
train/valid/test = 8/1/1
10 random seeds
```

Activity cliff：

```text
30 activity cliff datasets from Tilborg et al.
metrics: RMSE and RMSE_cliff
```

## 15. 实验结果

SCAGE 对比的 baseline 包括：

```text
MolCLR
KANO
GEM
ImageMol
GROVER
Uni-Mol
MolAE
```

论文结论：

```text
scaffold split:
  SCAGE 在 9 个 property benchmark 中 8 个取得最佳。

random scaffold split:
  SCAGE 在所有方法中整体最优。

activity cliff:
  30 个 activity cliff 数据集上，SCAGE 的 RMSE / RMSE_cliff 整体优于 baseline。
```

几个论文强调的结果：

```text
BACE:
  scaffold split 下相对 ImageMol 有 10.8% AUC 提升，
  相对 MolAE 有 2.2% AUC 提升。

ClinTox:
  相对 Uni-Mol 和 MolAE 分别提升 6.2% 和 16.3%。

Regression:
  scaffold split 下 Lipophilicity 提升 3.4%，FreeSolv 提升 2.1%。
```

值得注意：

```text
SCAGE 使用约 5M 预训练分子，
少于 Uni-Mol 这类更大规模预训练方法。
```

论文将其归因于任务设计和 conformational learning 的有效性。

## 16. Ablation 结论

MCL ablation：

```text
3D conformational distance > 2D topological distance
multiscale threshold > fixed threshold
20% - 60% 左右的尺度组合常见较优
```

M4 task ablation：

```text
只使用单个 pretraining task 效果下降。
四个任务联合最好。
```

表示可视化：

```text
随机初始化时正负样本混杂。
使用不同预训练任务后区分能力增强。
四任务联合时 Davies-Bouldin Index 最低。
```

Functional group ablation：

```text
atom-level functional group prediction
优于 molecule-level functional group number prediction。
```

## 17. 可解释性分析

SCAGE 使用 attention score 做 atom-level sensitivity。

具体做法：

```text
1. 汇总 Graph Transformer 中 global attention scores。
2. 得到每个 atom 的 sensitivity score。
3. edge score 取两端 atom score 平均。
4. 在分子图上着色。
```

结果显示：

```text
Solubility:
  attention 集中在 polar groups，符合 like dissolves like。

Toxicity / drug safety:
  attention 可定位潜在 toxic substructure。

Activity cliff:
  attention 高亮结构相似分子中的差异 functional group。
```

作者还在 BACE case study 中用 docking 结果对照，说明 SCAGE 高亮区域与 protein-ligand interaction 的关键位点相符。

## 18. 对 Uni-Poly-Plus 的启发

### 18.1 可借鉴：Functional group atom-level label

聚合物性质高度依赖功能团：

```text
ester / amide / imide / ether / sulfone
aromatic ring
halogen
hydrogen-bond donor / acceptor
ionic group
side-chain polar group
```

当前项目如果只用 atom feature 或 fingerprint，模型可能知道有这些 group，但不知道：

```text
哪些 atom 属于同一个 functional group。
functional group 在 backbone 还是 side chain。
functional group 是否靠近 star connection。
```

因此可以借鉴 SCAGE：

```text
为 repeat unit 中每个 atom 标注 polymer functional group label。
作为 atom-level auxiliary task。
```

这比简单预测 functional group count 更有用。

### 18.2 可借鉴：Polymer M4 预训练

SCAGE 的 M4 可以改成 polymer-specific M4：

```text
1. Polymer fingerprint / descriptor prediction
   Morgan / RDKit / Mordred / custom polymer descriptors

2. Polymer functional group prediction
   atom-level functional group or motif label

3. 2D periodic distance prediction
   repeat-unit graph shortest path
   star-connected periodic shortest path

4. Geometry prediction
   bond angle / torsion angle / end-to-end distance / radius of gyration bin
```

其中最值得优先尝试的是：

```text
functional group prediction
2D periodic shortest path prediction
torsion / angle prediction
```

原因是这些任务与 polymer repeat unit 结构强相关。

### 18.3 可借鉴：MCL 但要改造

SCAGE 的 MCL 使用小分子 conformer 的 atom-atom 3D distance percentile。

聚合物中可以借鉴 percent threshold 的思想，但不应直接照搬 MMFF lowest-energy conformer。

更适合的 polymer MCL 输入可能是：

```text
capped repeat unit conformer
short oligomer conformer
periodic PBC geometry
MD sampled conformer statistics
```

MCL threshold 可以基于：

```text
3D distance percentile
backbone graph distance percentile
star-to-star distance
side-chain distance
periodic image distance
```

对于 polymer，建议把 MCL 分成两类：

```text
local scale:
  bond / angle / neighboring functional group

mesoscale proxy:
  backbone path distance
  star-to-star / repeat direction
  side-chain to backbone distance
```

### 18.4 可借鉴：Activity cliff 思想迁移到 polymer cliff

SCAGE 对 activity cliff 做了专门评估。

聚合物中也有类似问题：

```text
结构非常相似，但 Tg / dielectric / modulus / conductivity 差异很大。
```

可以构造 polymer property cliff：

```text
same / similar backbone scaffold
small functional group substitution
property difference above threshold
```

然后评估模型是否能解释：

```text
哪些 substituent / side-chain / functional group 导致 property cliff。
```

这对论文里的可解释性部分很有价值。

## 19. 不宜直接照搬的部分

### 19.1 MMFF lowest-energy conformer

SCAGE 使用 MMFF 选择 lowest-energy conformer。

对聚合物不一定成立：

```text
repeat unit 不是完整 molecule。
星号连接位点会影响构象。
capped group 会引入人工偏差。
材料性质来自 ensemble / chain packing，而不是单个小分子最低能构象。
```

如果使用 3D，建议明确构象来源：

```text
star-substitution capped unit
oligomer
periodic geometry
multiple conformer ensemble
```

### 19.2 190 类小分子 functional group

SCAGE 的 functional group taxonomy 面向 drug-like molecules。

聚合物需要增加：

```text
polymer backbone class
repeat connection motif
side-chain anchor
aromatic backbone
aliphatic backbone
siloxane / carbonate / urethane / imide 等聚合物特异 motif
```

否则 atom-level label 对聚合物结构不够完整。

### 19.3 Attention explanation 不能直接等于 causal explanation

SCAGE 用 attention score 解释 property。

这有实际展示价值，但需要谨慎：

```text
attention high 不一定表示 causal contribution。
```

在 Uni-Poly-Plus 中如果使用类似解释，最好配合：

```text
substructure masking
counterfactual substitution
gradient attribution
property cliff pair comparison
```

## 20. 可落地实现建议

对当前 Uni-Poly-Plus，可以按优先级拆解：

```text
优先级 1：atom-level polymer functional group annotation
  SMARTS library
  star-neighborhood rules
  backbone / side-chain rules
  auxiliary atom classification head

优先级 2：2D periodic shortest path prediction
  普通 repeat-unit graph shortest path
  加 star-star periodic virtual edge
  pair classification loss

优先级 3：fingerprint / descriptor prediction
  graph embedding -> polymer descriptors
  或 graph embedding -> fingerprint bits

优先级 4：geometry angle / torsion prediction
  capped repeat unit 或 oligomer
  discretized angle / torsion bins
```

如果只做一个最可能高收益模块：

```text
atom-level functional group / motif prediction
```

原因：

```text
1. 不需要昂贵 3D。
2. 与聚合物性质强相关。
3. 可解释性强。
4. 可以和 property cliff 分析结合。
```

## 21. 与 MolGT 的关系

SCAGE 和 MolGT 都试图把 3D 信息注入 2D graph representation，但方式不同：

```text
MolGT:
  2D branch + 3D branch
  modality-shared Transformer
  2D-3D contrastive learning
  目标是 implicit 3D inference

SCAGE:
  主体是 2D graph Transformer
  用 3D conformational distance 改 attention
  用 3D bond angle 作为 pretraining target
  目标是 conformation-aware representation + interpretability
```

对 Uni-Poly-Plus：

```text
如果目标是无 3D 下游部署，MolGT 思路更直接。
如果目标是增强结构解释和 functional group 定位，SCAGE 思路更直接。
```

两者可以组合：

```text
MolGT-style:
  2D-3D contrastive / implicit geometry

SCAGE-style:
  functional group annotation
  multiscale conformational attention
  angle / topology auxiliary tasks
```

## 22. 总结

SCAGE 的核心价值在于把化学先验、2D topology、3D conformation 和可解释性组织成一个统一预训练框架：

```text
MCL:
  基于 3D distance percentile 的多尺度 attention mask

M4:
  fingerprint prediction
  functional group prediction
  2D atomic distance prediction
  3D bond angle prediction

Interpretability:
  atom-level functional group label
  attention sensitivity
  activity cliff explanation
```

对 Uni-Poly-Plus 最值得借鉴的是：

```text
1. polymer functional group / motif 的 atom-level 预训练任务。
2. periodic / polymer-aware 2D distance prediction。
3. angle / torsion 几何辅助任务。
4. property cliff 风格的聚合物结构-性质解释评估。
```

SCAGE 不应该被理解成“再加一个 3D conformer encoder”，而应该被理解成：

```text
如何把可解释的化学功能团和多尺度构象关系，变成 graph Transformer 的预训练监督。
```

## 23. 原仓库源码执行链路

原 SCAGE 的核心模型位于：

```text
models/scage.py
models/layers/embedding.py
models/layers/encoder.py
utils/compound_tools.py
utils/functional_group_all.py
pretrain.py / finetune.py
```

真实前向流程为：

```text
离散 atom fields
  -> 每个字段独立 Embedding 后求和

mass / van der Waals radius / partial charge
  -> Gaussian kernel expansion
  -> Linear projection

两部分相加 + LayerNorm
  -> prepend graph token
  -> 6 层 Multiscale Transformer
  -> graph token / atom token / atom-pair heads
```

离散原子字段包括：

```text
atomic number
chirality
degree
explicit valence
formal charge
hybridization
aromaticity
total hydrogen count
ring membership
```

这里与当前项目“把拼接后的连续 node feature 一次性送入 Linear”并不相同。

## 24. 原仓库的 3D 构象与 MCL

### 24.1 构象生成

源码对不超过 400 个原子的分子执行：

```text
AddHs
-> RDKit EmbedMultipleConfs(numConfs=10)
-> MMFFOptimizeMoleculeConfs
-> 选择 MMFF energy 最低的 conformer
-> RemoveHs
```

嵌入或优化不可用时退回 RDKit 2D coordinates。原仓库没有聚合物重复单元、`*`、star-linking、三聚体、周期向量 T 或 PBC 最小像处理。

### 24.2 多尺度构象注意力

原仓库计算完整 Euclidean distance matrix，并通过：

```python
np.percentile(pair_distances, dist_bar)
```

得到每个样本的距离阈值。默认预训练尺度是 `[20, 50]` 百分位。各尺度共享 Q/K/V，但使用不同 hard attention mask；每个尺度输出单独投影，随后 concat 并线性融合。Graph token 始终可以访问全部 atom tokens。

必须明确：

```text
原 SCAGE 没有 Gaussian RBF distance attention bias。
原 SCAGE 没有逐层 bond-type attention bias。
原 SCAGE 没有 star-linking/backbone/attachment attention bias。
原 SCAGE 的距离主要用于 hard visibility mask。
```

虽然 attention API 支持通用 `attn_bias` 参数，但 `Scage.forward()` 的正常调用没有向各层传入化学边 bias。

## 25. M4 四任务的源码实现

### 25.1 Fingerprint reconstruction

```text
target = Morgan fingerprint, radius=2, 2048 bit
input representation = graph token
loss = binary cross entropy with logits
```

它是 graph encoder 的预训练监督，不是下游额外拼接一个 fingerprint 模态。

### 25.2 Functional-group prediction

```text
target = 每个 atom 的 190-dimensional multi-hot vector
representation = atom token
loss = balanced multi-label BCEWithLogits
```

这修正了“每个原子只有一个唯一功能团类别”的误解。

### 25.3 Shortest-path prediction

```text
target = Floyd-Warshall 2D graph shortest path
maximum class = 20
representation = atom-pair projection
loss = focal loss
```

### 25.4 Bond-angle prediction

```text
target = 3D conformer 中相邻键形成的角度
range = [0, pi]
bins = 20
loss = focal loss
```

Pair distance 也存在连续/分桶处理代码，但论文主 M4 与默认任务列表是 `finger / sp / angle / fg`。

## 26. 原仓库微调并非只优化性质损失

这是源码审计得到的一个重要细节。`finetune.py` 除了下游 property loss，仍计算：

```text
Morgan fingerprint reconstruction loss
atom functional-group multi-label loss
```

并通过动态多任务损失与 property loss 合并。也就是说，原方法在微调时继续使用化学先验任务作为正则，而不是只加载预训练权重后单独回归/分类。

原仓库微调的典型默认配置为：

```text
embedding_dim = 512
num_layers = 6
num_heads = 16
dist_bar = [20, 100]
batch_size = 32
learning_rate = 5e-5
weight_decay = 1e-4
```

模型预测使用 graph token；原代码没有当前项目新增的 atom attention pooling，也没有 PaiNN 或多模态融合。

## 27. 当前项目与原 SCAGE 的准确对比

| 组件 | 原 KazeDog/scage | 当前 Uni-Poly-Plus |
|---|---|---|
| 输入对象 | 普通小分子 | 聚合物重复单元 |
| 2D 图 | 原始 molecular graph | star-linked polymer graph |
| atom embedding | 分字段 Embedding + 连续属性 Gaussian expansion | 已采用相同输入与 embedding；保留额外 `data.x` 供 GIN 使用 |
| 3D 来源 | 最低 MMFF energy conformer | periodic-PBC 或 fallback 构象 |
| 距离 | 普通 Euclidean distance | PBC 成功时 minimum-image distance；PBC 失败但 3D 坐标有效时 Euclidean；仅无效坐标才 topology fallback |
| 多尺度 | 默认 `[20,50]` hard percentile masks | 聚合物扩展默认关闭 percentile hard mask，保留 Graph Token 全局可见 |
| 连续距离 | 无逐层 RBF bias | PBC minimum-image distance 经 32 Gaussian RBF 映射为每个 head 的 additive bias；无有效 3D/PBC 时 bias=0 |
| bond 信息 | 未逐层注入 attention logits | 相同；star-linking 只改变拓扑，不作为 attention bias |
| readout | graph token | 相同 graph-token-only readout |
| Geom encoder | 无独立 3D encoder | 独立 PaiNN branch |
| 原预训练 | fingerprint、FG、SP、angle | mask atom、PerioGT augmentation、SP、angle、Geom denoising、模态对齐 |
| FP 角色 | graph-token reconstruction target | 独立 ECFP/mixfp 输入模态 |
| 下游辅助任务 | 保留 fingerprint + FG | 当前未完整复现这两项正则 |

结论是：当前 `SCAGEGraphEncoder` 保留原仓库 atom input、AtomEmbedding、Graph Token、层数和维度设置；注意力的距离部分已刻意改为连续 RBF bias，以适配 PBC 聚合物距离并避免 percentile hard mask 截断。差异集中在输入对象与系统集成，即 star-linked polymer topology、PBC distance、独立 PaiNN 和多模态融合。M4 任务仍未完整复现。

## 28. 哪些差异是合理扩展，哪些仍需补齐

### 28.1 合理且应保留

```text
star-linking graph
PBC minimum-image distance
严格 graph-to-geometry atom mapping
mapping 失败时 topology fallback
独立 PaiNN branch
```

这些设计用于处理聚合物周期性，原小分子 SCAGE 没有对应问题，因此“不一致”不代表错误。

### 28.2 当前最值得从原源码补齐

优先级一：恢复 atom-level multi-label functional-group task。它是 SCAGE 可解释性与化学先验的核心，而当前 mask atom 不能等价替代。

优先级二：增加 graph-token fingerprint reconstruction，至少做 Morgan 2048-bit BCE。独立 FP 模态提供预测捷径，但不能保证 Graph encoder 学会重建化学子结构先验。

优先级三：下游微调保留低权重 FG/fingerprint auxiliary losses，验证其是否降低小数据集过拟合。

优先级四：对比原式 graph-token-only readout 与当前 graph-token + atom attention pooling，避免 readout 改动掩盖 encoder 本身效果。

### 28.3 不应机械照搬

```text
不应移除 PBC 而退回普通 Euclidean distance。
不应删除 star-linking/backbone 特征。
不应使用原普通小分子 FG SMARTS 而不检查其对含 * 的 polymer RU 的覆盖率。
```

## 29. 对“当前实现是否与原 SCAGE 一致”的修正回答

简短答案：**SCAGE 图主干和原子输入现已按原仓库实现，但距离来源和外部多模态系统仍是聚合物扩展。**

一致部分：

```text
Graph Transformer 主干
Graph token
基于构象距离百分位的多尺度 hard masks
shortest-path 与 angle 辅助任务的方向
动态多任务加权思想
```

不同部分：

```text
当前增加 PBC/star-linking/PaiNN/多模态融合，但没有在 attention 中保留额外 RBF/edge bias。
当前 atom embedding、6层/512维/16头、FFN 256 和 graph-token readout 与原仓库一致。
当前没有完整实现原 SCAGE 的 FG multi-label 和 graph-token fingerprint tasks。
当前动态 loss 的任务集合与阶段组织不同。
原 SCAGE 微调保留 FG/fingerprint 辅助损失，当前主线没有等价机制。
```

因此下一步不应以“让所有代码完全一致”为目标，而应进行两组清晰消融：

```text
A. 当前主线：原 SCAGE 输入/主干 + PBC + star-linking。
B. 在 A 上补充原 SCAGE 的 FG + fingerprint reconstruction，以及下游辅助正则。
```

这能直接判断原 SCAGE 真正关键的化学监督，是否能改善当前聚合物任务，而不会牺牲已经建立的周期结构表达。

## 30. 实验性 screw-periodic `(R,T)` 距离

项目的 `geom_input=screw_periodic` 现为前向构造的严格 screw 路线：从优化 capped RU 的 attachment vectors 初始化 `R,T`，在连接键轴上采样 dihedral，再由同一个解析 `(R,T)` 生成左右像。SCAGE 对中心、正向螺旋像和逆向螺旋像取最小原子间距离，再映射为连续 RBF attention bias。该逻辑不是原 SCAGE 的组成部分，而是聚合物周期几何扩展。

该实现参考 RadonPy 的连接向量同轴对齐、封端删除与新键二面角采样建链策略，以及 Hirata 等对无限螺旋聚合物中 twist/rise 结构变量的显式优化。它不再要求任意自由 RDKit 三聚体“碰巧”具有一步螺旋对称；解析构造保证所有成功样本的 screw residual 处于数值精度范围。文献：[RadonPy](https://www.nature.com/articles/s41524-022-00906-4)，[Helical Organic and Inorganic Polymers](https://pubs.acs.org/doi/10.1021/acs.jpcb.3c00620)。

当前固定 `seed=42` 的 500 unique-SMILES 审计结果为 `479/500 = 95.8%`。通过样本的 screw residual p95 约为 `9.7e-16 Å`；其余 21 个样本均标记为 fallback，而不是作为三聚体或普通欧氏几何参与 screw 成功统计：17 个为能量/原子超过 `5.0` 的高应变候选，3 个出现重原子碰撞，1 个不满足两个不同 attachment boundary 的线性链假设。
