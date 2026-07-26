# MIPS 论文分析：Multimodal Infinite Polymer Sequence Pre-training

分析对象：`/root/workspace/DeepLearning/Paper/3746027.3755729.pdf`

论文题目：**MIPS: A Multimodal Infinite Polymer Sequence Pre-training Framework for Polymer Property Prediction**

作者：Jiaxi Wang, Yaosen Min, Xun Zhu, Miao Li, Ji Wu

会议：ACM Multimedia 2025（MM '25），Dublin，2025-10-27 至 2025-10-31

页码：729-738

DOI：`10.1145/3746027.3755729`

论文代码地址：`https://github.com/wjxts/MIPS`

本文档基于论文正文、公式、算法、图表、参考文献及作者公开源码进行分析。源码审计基线为：

```text
repository: https://github.com/wjxts/MIPS
commit: 26aafe52926a3f33bf2d3d382ae263360319812d
commit date: 2025-07-28
```

以下内容严格区分：

```text
1. MIPS 论文明确提出的原始方法；
2. 论文没有说明、无法仅凭正文确定的工程细节；
3. 公开源码在该 commit 上实际执行的逻辑；
4. 当前 Uni-Poly-Plus 项目对 MIPS/SCAGE/PBC 的扩展。
```

## 1. 一句话总结

MIPS 将无限均聚物的拓扑压缩为一个 **star-linked repeating-unit quotient graph**，使用带 Graphormer 拓扑编码的 **Localized Graph Transformer** 建模无限链拓扑，再用单体构象的若干 **3D descriptor tokens** 通过单向 cross-attention 注入每个原子表示，最后用 masked-atom prediction 在约 100 万条 PI1M P-SMILES 上预训练。

MIPS 原论文的完整主线是：

```text
P-SMILES
├── 2D branch
│   ├── 删除 attachment stars，连接两个 boundary atoms
│   ├── backbone embedding
│   └── Localized Graph Transformer
│       ├── shortest-path distance attention
│       ├── shortest-path edge/path attention
│       └── localized graph attention
│
└── 3D branch
    ├── repeating monomer 3D coordinates
    ├── 3D molecular descriptors
    ├── atom-pair 3D descriptors
    └── descriptor-group projections

topological atom tokens <- cross-attention <- spatial descriptor tokens
                         |
                         ├── masked-atom pre-training
                         └── pooling + property prediction
```

最重要的辨析是：

```text
MIPS 没有构建无限链的原子级 PBC 坐标。
MIPS 没有 PaiNN、SchNet、EGNN 等等变 3D encoder。
MIPS 的 Localized Graph Attention 距离是 2D graph shortest-path distance。
MIPS 的 3D 信息来自 repeating monomer 的全局 descriptor groups。
MIPS 的预训练目标只有 masked-atom prediction。
```

## 2. 论文要解决的问题

论文认为常见 P-SMILES 处理存在三个主要问题。

### 2.1 表示不满足重复与平移不变性

同一个无限均聚物可以写成不同但等价的 P-SMILES，例如：

```text
translation:
  *CONO* -> *NOCO*

repetition:
  *CONO* -> *CONOCONO*
```

如果模型把字符串或有限重复单元直接当作普通分子处理，等价写法可能得到不同预测。

MIPS 的目标不是通过大量增强“学会”这种不变性，而是从图构造层面直接建立不变性。

### 2.2 无限链不能直接送入有限图网络

真正的均聚物图可视为沿链方向无限重复的图：

```text
... - RU[-1] - RU[0] - RU[1] - ...
```

其节点数无限，不能直接执行普通 message passing 或全局 self-attention。MIPS 因此寻找一个有限 quotient graph，使其局部计算等价于无限链计算。

### 2.3 单体 P-SMILES 缺少聚合后的空间信息

论文认为聚合物性质不仅依赖 2D 拓扑，也依赖空间结构。因此在拓扑分支外，增加由单体 3D 构象计算的 descriptor tokens，并与原子表示进行 cross-modal fusion。

需要注意：论文解决的是“补充空间先验”，并没有声称单体描述符等价于真实无限链、无定形堆积或晶体构象。

## 3. Repeat and Shift Invariance Test

MIPS 提出 RSIT（Repeat and Shift Invariance Test）评估模型对等价 P-SMILES 的鲁棒性。

### 3.1 随机增强

论文 Algorithm 1 的逻辑是：

```text
输入原 P-SMILES s
-> 以 0.5 概率先 repeat
-> 再随机平移 repeating sequence
-> 得到等价 P-SMILES s_hat
```

这里的增强不改变目标无限聚合物，只改变有限 P-SMILES 的起点或重复长度。

### 3.2 对抗式评估

Algorithm 2 对每个样本执行 `T` 次增强，保留损失最大的增强结果：

```text
for t in 1..T:
    x_adv = random_augment(x)
    prediction = model(x_adv)
    保留 loss 最大的一次
```

论文实验采用：

```text
T = 5
```

因此 RSIT 是一种 worst-case robustness test，不是论文主预训练任务，也不是论文明确采用的训练数据增强。

### 3.3 RSIT 实验结论

论文用 3-layer、512-hidden GIN 比较四种 P-SMILES 图处理：

| 方法 | RSIT 前平均表现 | RSIT Gap |
|---|---:|---:|
| Star Keep | 正常 | 0.233 |
| Star Remove | 正常 | 0.199 |
| Star Substitution | 正常 | 0.107 |
| Star Linking | 不变 | **0.000** |

这说明 star linking 对 repeat/shift 具有结构性不变性，而不是依赖模型碰巧学习到不变性。

## 4. 无限聚合物的 Star-Linking 表示

### 4.1 无限图定义

设一个 monomer graph 为：

$$
G=(V,E,X)
$$

其中两个 boundary atoms 是聚合时与相邻单元连接的原子。无限聚合物图 `G^p` 是该 monomer graph 的周期重复，并在相邻单元的 boundary atoms 之间增加化学键。

### 4.2 Induced star-linking graph

MIPS 构造有限图：

$$
G^*=(V^*,E^*,X^*)
$$

满足：

$$
V^*=V,\qquad X^*=X
$$

$$
E^*=E\cup\{(v_0,v_{|V|-1})\}
$$

直观上：

```text
P-SMILES: *-left_boundary ... right_boundary-*

删除两个 *
连接 left_boundary 与 right_boundary
得到一个闭合 quotient graph
```

这个闭合边代表“从当前 RU 穿过周期边界到相邻 RU”，不是同一个 RU 内真实存在的普通化学键。

### 4.3 Message passing 等价性

论文 Proposition 1 声称，在相同 message passing 规则下：

```text
无限链中第 i 类周期等价原子的表示
=
star-linked quotient graph 中第 i 个原子的表示
```

Theorem 1 进一步要求网络由以下部分组成：

```text
message-passing layers
node-wise transformations
final mean pooling
```

在这些条件下，网络对无限聚合物图和 star-linked graph 的输出相同。

这个结论为 star linking 提供了理论依据，但适用范围必须明确：

```text
它证明的是周期等价的局部 message passing 和最终 mean pooling。
它没有自动证明任意 global attention、Graph Token readout、PBC 3D bias
或任意复杂 pooling 仍保持相同等价性。
```

## 5. Localized Graph Attention

### 5.1 基础 Graphormer attention

MIPS 采用 Graphormer 风格的 topology-aware attention：

$$
Q=W^QX,\quad K=W^KX,\quad V=W^VX
$$

$$
A=\operatorname{Softmax}
\left(
\frac{K^TQ}{\sqrt d}+A^d+A^p
\right)
$$

其中：

```text
A^d_ij = f_dist(d_ij)
  d_ij 是节点 i、j 在图上的最短路径长度。

A^p_ij = f_path(p_ij)
  p_ij 是 i 到 j 的 shortest path，编码路径上的边信息。
```

这里的 `distance attention` 是拓扑 hop/SPD bias，不是 3D 欧氏距离。

### 5.2 为什么需要局部注意力

对无限图做 global attention 时，一个原子会关注无限多个周期镜像，计算不可行。MIPS 因此引入阈值 `d_thres`：

$$
\hat A_{ij}=A_{ij}\cdot \mathbf{1}\{d_{ij}<d_{thres}\}
$$

只允许拓扑距离小于阈值的节点对参与注意力。

### 5.3 无限链与 quotient graph 的注意力等价条件

Theorem 2 给出的关键条件是：

$$
d(boundary_{left},boundary_{right})>2d_{thres}-1
$$

如果单个 monomer 太短，不满足该条件，论文要求先重复 monomer，直到 boundary distance 超过条件，再应用 LGA。

这一点在复现中不能忽略。直接对所有单 RU 使用固定局部阈值，并不一定满足论文的理论前提。

### 5.4 公式实现上的歧义

论文公式先执行 Softmax，再将超阈值注意力乘零，但没有写重新归一化：

```text
Softmax -> multiply mask -> V @ masked_attention
```

而常见 Transformer 实现通常是：

```text
logits 上 masked_fill(-inf) -> Softmax
```

两者不完全等价：前者会让每个 query 的 attention mass 小于 1。论文正文没有解释是否代码实际做了重新归一化，因此严格复现时必须以原仓库源码为准，不能仅凭公式自行断言。

## 6. Twin Polymer Graph 与 Backbone Embedding

### 6.1 Star linking 的表达能力损失

不同无限聚合物可能在 star linking 后得到同构甚至相同的 quotient graph。论文把这种情况定义为 twin polymer graphs。

论文证明：

```text
WL test 无法区分 twin polymer graphs；
普通 message passing 无法区分；
localized graph attention 也无法区分。
```

问题在含环结构中尤其明显：同一个环可能属于真正的聚合主链，也可能只是侧链环，但 quotient graph 局部拓扑无法可靠恢复这种角色差异。

### 6.2 Backbone 定义

论文把 backbone 定义为：

```text
两个 boundary atoms 之间 shortest path 上的 atoms and rings
```

然后为 backbone atoms 加一个共享可学习向量：

$$
b_i=
\begin{cases}
b,& i\text{ 是 backbone atom}\cr
0,& \text{otherwise}
\end{cases}
$$

初始拓扑表示为：

$$
X_0^t=X^*+B
$$

“and rings”很重要：若 shortest path 穿过一个环，只标记最短路中恰好经过的几个原子，可能仍不能完整表达该环属于 backbone。当前项目已经显式扩展到 shortest path 实际穿过的完整环，这与论文文字意图一致。

### 6.3 Backbone embedding 的实际收益

GIN3-512 消融结果：

| Task | Star Linking | + Backbone Embedding | 增益 |
|---|---:|---:|---:|
| Egc | 0.870 | 0.882 | +0.012 |
| Egb | 0.918 | 0.920 | +0.002 |
| Eea | 0.916 | 0.917 | +0.001 |
| Ei | 0.785 | 0.785 | 0.000 |
| Xc | 0.279 | 0.346 | +0.067 |
| EPS | 0.772 | 0.772 | 0.000 |
| Nc | 0.847 | 0.849 | +0.002 |
| Eat | 0.966 | 0.967 | +0.001 |

收益主要集中在 `Egc` 和 `Xc`。论文进一步统计这两个数据集含环更多：

```text
Egc: 平均 1.81 个环，26% 样本超过 2 个环
Xc:  平均 1.28 个环，22% 样本超过 2 个环
```

因此 backbone embedding 是针对 ring/backbone ambiguity 的结构先验，不应被描述为对所有任务都有同等强度的普遍提升。

## 7. Localized Graph Transformer

MIPS 的 topology encoder 由 `L` 层 Localized Graph Transformer 构成：

$$
X^t_{l+1/2}=\operatorname{LayerNorm}
\left(
\operatorname{LocalAttn}(G^*,X_l^t)+X_l^t
\right)
$$

$$
X^t_{l+1}=\operatorname{LayerNorm}
\left(
\operatorname{FFN}(X^t_{l+1/2})+X^t_{l+1/2}
\right)
$$

FFN 是带 ReLU 的两层 MLP。论文主配置：

```text
layers = 6
hidden_dim = 512
```

论文没有在正文中给出：

```text
attention head 数量（公开源码默认值为 8）
d_thres 的具体值
dropout
weight decay
Graphormer path 最大 hop 数
padding/mask 的工程实现
```

因此这些参数不能从论文主文可靠还原。公开源码可以补充其中一部分，但源码与论文公式并不完全一致，详见第 20 节。

另一个重要区别是，论文的理论结论以 **mean pooling** 为条件；正文没有把 Graph Token 描述为核心 readout。当前项目使用 Graph Token，是来自 SCAGE/Transformer 工程体系的扩展，不是 MIPS 定理的一部分。

## 8. Spatial Structure Encoder

### 8.1 输入不是无限链 PBC 坐标

论文从 repeating monomer 的原子 3D coordinates 计算描述符：

```text
monomer coordinates
-> multiple 3D descriptor groups
-> each group independently projected to d dimensions
-> stack as spatial tokens
```

若第 `i` 个 descriptor group 为：

$$
x_i^s\in\mathbb{R}^{d_i}
$$

则独立线性投影为：

$$
\hat x_i^s=W_i x_i^s,\qquad W_i\in\mathbb{R}^{d\times d_i}
$$

最终堆叠得到：

$$
X^s\in\mathbb{R}^{d\times N_s}
$$

因此 descriptor group 被视作一组 spatial tokens，而不是先拼成一个长向量。

### 8.2 论文明确给出的描述符来源

正文只明确写到：

```text
3D molecular descriptors from Yang et al. [52]
atom-pair 3D descriptors from Awale and Reymond [3] / RDKit [20]
```

论文主文没有列出每个 descriptor group 的名称、维度、标准化方式、缺失值处理或构象生成参数。

因此，当前项目中的：

```text
shape-11
USRCAT-60
AUTOCORR3D-80
RDF-210
MORSE-224
WHIM-114
```

是一个合理且明确的工程扩展，但不是作者公开源码的输入。源码实际使用 `md-200 + atom_pair_3d-512`，详见第 20.4 节。

### 8.3 方法能力边界

这种设计的优点是：

```text
计算成本远低于原子级 3D Transformer；
能加入 shape、空间自相关、药效团和原子对信息；
descriptor token 数少，适合 batch=1024 的大规模预训练。
```

局限是：

```text
只描述单体构象，不是无限链周期构象；
不显式保留哪个空间关系来自左/右周期镜像；
对构象生成和 conformer selection 敏感；
不能表达无定形堆积、链间作用、分子量和工艺条件。
```

## 9. Cross-Modal Fusion

MIPS 使用单向 cross-attention，把 spatial descriptors 注入每个 topological atom token。

拓扑原子为 Query：


$$
Q=W^QX^t
$$

空间 descriptor tokens 为 Key/Value：

$$
K=W^KX^s,\qquad V^s=W^VX^s
$$

注意力：

$$
A^{ts} = \operatorname{Softmax}\left(\frac{K^TQ}{\sqrt{d}}\right)
$$

残差融合：

$$
X^{ts}=\operatorname{LayerNorm}
\left(
X^t+V^sA^{ts}
\right)
$$

其语义是：

```text
每个 atom token 查询全部 descriptor-group tokens；
不同原子可以选择不同的空间描述符组合；
最终输出仍保持 atom-token 序列结构；
拓扑表示通过 residual 保留。
```

这不是：

```text
Graph embedding 与 Geom embedding 的 gate；
SMILES/FP/Graph 三个 graph-level tokens 的平铺 attention；
Topology branch 与 Geometry branch 的 logits 相加；
双向 cross-attention。
```

因此它更像“descriptor-conditioned atom representation”，而不是普通 late fusion。

## 10. 预训练任务

MIPS 原论文只使用一个预训练任务：

```text
Masked Atom Prediction
```

具体流程：

```text
以 p_mask=0.30 选择原子
-> mask 该原子的完整 atom features
-> topology encoder
-> spatial descriptor cross-attention
-> linear classifier 预测 atom type
```

论文没有使用：

```text
ECFP reconstruction
periodic shortest-path prediction
angle/torsion prediction
coordinate denoising
PerioGT contrast
SCAGE-SMILES alignment
SCAGE-FP alignment
```

这些任务属于当前项目的后续扩展。

原方法的优势是目标简单、吞吐量高；缺点是监督信号单一，而且论文没有消融 masked-atom pretraining 相对随机初始化的独立收益，也没有比较其他预训练任务。

## 11. 数据集与训练参数

### 11.1 预训练数据

论文正文写作 `PL1M`，但引用 [27] 明确是：

```text
PI1M: A Benchmark Database for Polymer Informatics
```

因此这里应理解为约 100 万条无标签 PI1M polymer sequences，而不是另一个独立数据集。

### 11.2 下游数据

| Task | Property | Samples | Unit |
|---|---|---:|---|
| Egc | chain band gap | 3380 | eV |
| Egb | bulk band gap | 561 | eV |
| Eea | electron affinity | 368 | eV |
| Ei | ionization energy | 370 | eV |
| Eat | atomization energy | 390 | eV/atom |
| Xc | crystallization tendency | 432 | % |
| EPS | dielectric constant | 382 | dimensionless |
| Nc | refractive index | 382 | dimensionless |

论文使用 5-fold cross-validation，报告 mean ± standard deviation 的 RMSE 和 R²。

正文未充分说明：

```text
每个 fold 内 validation 如何划分；
best checkpoint 如何选择；
是否使用 early stopping；
随机种子与精确 fold 文件；
重复或相似 polymer 是否跨 fold；
```

因此与当前项目结果比较时，必须先确认数据划分和 checkpoint selection 完全一致。

### 11.3 预训练参数

```text
Localized Graph Transformer layers = 6
hidden_dim = 512
atom mask rate = 0.30
optimizer = Adam
beta1 = 0.9
beta2 = 0.999
learning_rate = 2e-4
batch_size = 1024
training_steps = 20,000
warmup_steps = 2,000
hardware = single NVIDIA RTX 4090
```

`20,000 x 1024 = 20.48 million` 次样本呈现，约等于对 100 万 PI1M 做 20 个 epoch 的数量级。当前项目若只使用 PI1M-50k，即使 epoch 数相同，数据多样性也远小于原论文。

### 11.4 微调参数

```text
epochs = 50
learning_rate = 3e-5
batch_size = 32
```

预实验 GIN3-512 使用：

```text
layers = 3
hidden_dim = 512
epochs = 50
learning_rate = 1e-3
batch_size = 32
```

## 12. 主实验结果

MIPS 报告的 R²：

| Task | MIPS R² | Std |
|---|---:|---:|
| Egc | 0.926 | 0.006 |
| Egb | 0.945 | 0.007 |
| Eea | 0.940 | 0.018 |
| Ei | 0.846 | 0.051 |
| Xc | 0.506 | 0.074 |
| EPS | 0.814 | 0.045 |
| Nc | 0.877 | 0.046 |
| Eat | 0.990 | 0.004 |

8 个任务的简单平均 R² 为：

$$
\bar R^2=0.8555
$$

相应 RMSE：

| Task | MIPS RMSE | Std |
|---|---:|---:|
| Egc | 0.429 | 0.014 |
| Egb | 0.460 | 0.037 |
| Eea | 0.267 | 0.018 |
| Ei | 0.384 | 0.048 |
| Xc | 16.435 | 0.576 |
| EPS | 0.484 | 0.049 |
| Nc | 0.083 | 0.010 |
| Eat | 0.038 | 0.006 |

论文在其比较协议下超过 ChemBERTa、MolCLR、3D Infomax、Uni-Mol、SML、PLM、polyBERT、TransPolymer 和 MMPolymer。

但不能仅根据这张表断言架构在任意当前项目协议中必然达到同样结果。预训练规模、fold、特征生成、归一化和模型选择策略都会显著影响结果。

## 13. 消融实验解读

Table 5 的结果为：

| Model | Egc | Egb | Eea | Ei | Xc | EPS | Nc | Eat | Mean R² |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2D | 0.904 | 0.932 | 0.931 | 0.835 | 0.409 | 0.803 | 0.870 | 0.980 | 0.8330 |
| 2D + BE | 0.915 | 0.934 | 0.933 | 0.836 | 0.457 | 0.806 | 0.873 | 0.988 | 0.8428 |
| 3D only | 0.886 | 0.904 | 0.868 | 0.778 | 0.419 | 0.725 | 0.822 | 0.969 | 0.7964 |
| 2D + BE + 3D | **0.926** | **0.945** | **0.940** | **0.846** | **0.506** | **0.814** | **0.877** | **0.990** | **0.8555** |

### 13.1 3D 不能替代 2D topology

`3D only` 平均 R² 明显低于 `2D`：

```text
3D only = 0.7964
2D only = 0.8330
```

这说明 descriptor branch 更适合作为补充，不适合作为唯一结构表示。

### 13.2 3D 对 2D+BE 的增益

加入 3D 后各任务增益：

| Task | 增益 |
|---|---:|
| Egc | +0.011 |
| Egb | +0.011 |
| Eea | +0.007 |
| Ei | +0.010 |
| Xc | +0.049 |
| EPS | +0.008 |
| Nc | +0.004 |
| Eat | +0.002 |

最大收益是 `Xc`，其次是 `Egc/Egb/Ei`。这与结晶倾向和电子结构对构象/空间分布较敏感的直觉一致。

### 13.3 论文没有完成的关键消融

论文没有单独比较：

```text
无预训练 vs masked-atom pretraining；
cross-attention vs concat/gate/late fusion；
不同 3D descriptor group；
不同 conformer 或 conformer ensemble；
不同 d_thres；
hard LGA vs global attention；
Graphormer distance bias/path bias 各自贡献；
单体 3D descriptor vs 周期链 3D 表示。
```

因此 Table 5 只能支持“BE 与 3D branch 整体有益”，不能证明其中每个工程组件都最优。

## 14. 可解释性分析

论文使用：

```text
principal subgraph mining
1-overlap fragment vocabulary
vocabulary size = 100
regression class activation mapping
```

fragment vocabulary只由每个任务的训练集建立，这一点避免了直接使用测试集片段统计。

以 Egc 为例，论文给出的高贡献片段包括：

```text
COC
CON
linear alkyl fragment
```

低贡献片段包括：

```text
thiophene-like conjugated ring
C#C
C=S
```

作者认为结果与局域电子结构、共轭和 pi-electron delocalization 的化学知识一致。

不过该解释是模型相关性解释，不是因果证明；fragment importance 也可能受数据集中共现模式影响。

## 15. 方法优势

### 15.1 用 quotient graph 表达无限拓扑

star linking 将无限重复图压缩为有限图，计算成本与单 RU 大小相关，而不是与人为选定的 oligomer 长度相关。

### 15.2 不变性来自构造而非增强

RSIT gap 为零是 star-linking 的直接结构性质，比依赖随机增强学习不变性更稳定。

### 15.3 Graphormer topology bias 比普通 GIN 更丰富

模型同时使用：

```text
atom features
shortest-path distance
shortest-path edge/path features
backbone role
```

比仅依赖相邻边传播更适合表达中程拓扑关系。

### 15.4 Descriptor-token cross-attention 成本低

空间 token 数远小于原子数，不需要在 3D 全原子图上执行昂贵的等变消息传递，因此可支持论文中的大 batch 预训练。

## 16. 局限与风险

### 16.1 Star linking 不是完整无限链的单射表示

twin polymer graph 已由论文自己证明存在。Backbone embedding 加入主链角色先验，但不能从理论上保证消除所有 quotient-graph collision。

### 16.2 LGA 理论有适用条件

只有 boundary distance 满足：

$$
d_{boundary}>2d_{thres}-1
$$

才可直接使用单 RU quotient graph。短 RU 必须扩展后再计算。论文没有在实验部分给出扩展比例、平均扩展长度或额外开销。

### 16.3 3D branch 不是周期聚合物构象

单体 3D descriptors 无法明确表达：

```text
相邻 RU 的边界二面角；
周期镜像原子关系；
链的螺旋/平移周期；
长链构象分布；
链间堆积与结晶形态。
```

因此应把它理解为低成本空间先验，而不是无限链 3D 模拟器。

### 16.4 构象与 descriptor 复现信息不足

正文没有给出：

```text
3D conformer generation algorithm；
候选 conformer 数；
MMFF/UFF 优化设置；
最低能量选择规则；
descriptor 完整清单与维度；
descriptor normalization；
构象失败处理。
```

这些都可能显著影响结果。

### 16.5 预训练消融不足

MIPS 叫“pre-training framework”，但论文没有清楚隔离 masked-atom pretraining 对最终结果的贡献，也没有与同架构随机初始化进行完整比较。

### 16.6 数据协议细节不足

下游样本很小，尤其 Eea/Ei/EPS/Nc 约 368-382 条。此时 validation、early stopping、fold 和超参数选择对 R² 影响很大。正文信息不足以排除协议差异造成的部分提升。

## 17. 与当前 Uni-Poly-Plus 项目的区别

当前项目已经吸收了 MIPS 的若干思想，但并非 MIPS 原样复现。

| 维度 | MIPS 原论文 | 当前项目主线 |
|---|---|---|
| Polymer topology | star-linked quotient graph | star-linked / adaptive m-RU periodic graph |
| Backbone | shortest path 上 atoms and rings 的共享 embedding | shortest path、完整穿越环、attachment/backbone role fields |
| Topology attention | Graphormer SPD/path bias + hard localized attention | `mips_dual` topology branch，SPD/path-bond bias，默认 hop locality |
| Readout | 理论以 mean pooling 为条件 | Graph Token readout |
| 3D input | monomer-level descriptor tokens | PolyGen-inspired periodic coordinates/explicit image distance；descriptor 默认关闭 |
| 3D attention | descriptor-to-atom cross-attention | topology/geometry dual attention branch 后 concat/projection |
| Infinite 3D | 未显式建模 | `(R=I,T=(0,0,L))` 或显式周期镜像 |
| Pretraining | masked atom only | masked atom + topology/SP + periodic geometry 等扩展任务 |
| Downstream modalities | topology + spatial descriptors | SCAGE + SMILES + FP parallel attention |

### 17.1 当前项目做对的部分

```text
star-linking 保留了 repeat/shift invariance；
补充完整 backbone ring role 更接近论文语义；
SPD 与 path-bond bias 对应 MIPS/Graphormer topology encoding；
将 topology distance 和 Å geometry distance 分成两个分支，避免单位混淆；
对周期镜像增加 image identity，解决 minimum-image 距离丢失来源的问题。
```

### 17.2 当前项目偏离 MIPS 的关键部分

```text
Graph Token 不在 MIPS 的等价性定理中；
PBC geometry branch 是扩展，不是 MIPS 原论文；
当前默认关闭显式 descriptor tokens，缺少 Table 5 中证据最直接的 3D 模块；
多任务预训练增加了复杂度，但原 MIPS 的强结果只依赖 masked atom + 大规模 PI1M；
PI1M-50k 数据规模远小于论文约 1M 的预训练数据。
```

## 18. 对当前项目最有价值的借鉴

### 18.1 首先建立一个忠实 MIPS baseline

不要直接把所有新模块叠加后与论文结果比较。应实现最小忠实基线：

```text
star linking
+ backbone embedding
+ Graphormer SPD/path bias
+ hard LGA
+ mean pooling
+ descriptor-token cross-attention
+ masked atom p=0.30
```

该基线用于回答：当前性能差距来自 MIPS 核心复现不足，还是新模块本身无效。

### 18.2 恢复显式 3D descriptor 消融

论文最一致的实验信号是：在 `2D+BE` 上加入 3D descriptors，8/8 任务均提升。

当前项目已有 descriptor cache，但默认关闭。建议做严格单变量消融：

```text
A. topology only
B. topology + MIPS descriptor cross-attention
C. topology + PolyGen periodic geometry branch
D. topology + descriptors + periodic geometry
```

这能判断：

```text
descriptor 是否提供稳定全局几何先验；
PBC branch 是否提供 descriptor 没有的局部周期信息；
二者是否互补，还是高度冗余。
```

### 18.3 RSIT 应作为固定鲁棒性指标

对每个下游 fold 同时报告：

```text
normal R²/RMSE
RSIT worst-case R²/RMSE, T=5
RSIT gap
```

先把 RSIT 用作测试指标，不要把它与 PerioGT 对比学习混为同一个任务。

### 18.4 验证 LGA 的短 RU 条件

缓存 diagnostics 应统计：

```text
boundary shortest-path distribution
满足 d_boundary > 2*d_thres-1 的比例
需要重复 RU 的样本数
重复后图大小和训练开销
```

若大量短 RU 不满足条件，当前固定单图 hard locality 并不是论文理论保证下的实现。

### 18.5 预训练规模优先于无控制地增加任务

论文的有效训练量约为：

```text
20,000 steps x global batch 1024
= 20.48M sample presentations
```

当前若使用 PI1M-50k，应优先验证：

```text
50k / 200k / 1M 数据规模曲线
固定 sample presentations 的公平比较
masked-atom-only 与多任务的吞吐量和下游收益
```

如果复杂几何任务使每步显著变慢，简单 masked-atom 在更多独特 polymer 上训练可能更有效。

### 18.6 不要把论文值直接当作当前协议阈值

必须先统一：

```text
相同原始 CSV
相同 fold
相同 target normalization
相同 validation/checkpoint selection
相同 R² 聚合方式
```

否则“未超过 MIPS”可能同时包含数据协议差异，而不只是模型差异。

## 19. 推荐消融矩阵

为了避免一次改动太多，推荐按以下顺序执行：

| ID | Topology | 3D | Pretraining | 目的 |
|---|---|---|---|---|
| A | star-link + BE + SPD/path + LGA | 无 | masked atom | 忠实 2D MIPS baseline |
| B | 同 A | descriptors cross-attention | masked atom | 复现论文完整 MIPS |
| C | 当前 mips_dual | PolyGen PBC bias | 当前主任务 | 当前项目 baseline |
| D | 当前 mips_dual | descriptors + PBC | masked atom | 检验两类 3D 是否互补 |
| E | D | descriptors + PBC | 当前多任务 | 检验附加任务净收益 |

每个实验至少记录：

```text
8-task mean R²
8 个 task 单独 R²/std
FP-only / SMILES-only / SCAGE-only
RSIT gap
geometry-valid/fallback 分组结果
训练吞吐量与总 GPU-hours
```

## 20. 作者公开源码审计

本节不是根据论文图示推断，而是对公开仓库 commit
`26aafe52926a3f33bf2d3d382ae263360319812d` 的实际代码路径进行核验。

源码审计得到的结论是：

```text
公开代码能验证 MIPS 的大部分拓扑思想，
但它不是论文方法与实验协议的完整、无歧义复现。
```

### 20.1 公开预训练命令

README 给出的主预训练命令确认：

```text
dataset = pl1m_aug
graph operation = star_link
model = kfuse_graph_transformer
layers = 6
hidden_dim = 512
global batch size = 1024
max steps = 20,000
warmup = 2,000
learning rate = 2e-4
knowledge vectors = [md, atom_pair_3d]
```

这与论文大部分超参数一致。源码默认：

```text
attention heads = 8
dropout = 0.1
attention dropout = 0.1
activation dropout = 0.1
readout = mean
```

优化器源码默认 `beta2=0.98`，而论文正文写 `beta2=0.999`。README 没有覆盖
`optimizer.beta2`，因此按公开命令执行时会使用 `0.98`。这是论文与源码的明确不一致。

### 20.2 “Complete graph”实际是 2-hop 局部图

数据配置使用：

```text
completion = True
max_length = 3
```

但 `generate_complete_graph()` 调用：

```python
all_pairs_shortest_path(cutoff=max_length - 1)
```

因此 `max_length=3` 时只创建：

```text
distance = 0: self loop
distance = 1: directly bonded atom
distance = 2: two-hop atom
```

拓扑距离大于 2 的 atom pair 根本不进入 DGL attention graph。也就是说源码中的 LGA 是一个稀疏 2-hop attention graph，而不是先构造全局 attention 再对远距离位置置零。

这也补出了论文未报告的实际局部阈值：

```text
d_thres = 3
即 d_ij < 3
```

### 20.3 短 repeating unit 的扩展实现

源码先计算两个 wildcard atoms 之间的 shortest path：

```python
path = nx.shortest_path(...)
if len(path) <= 5:
    graph = repeat_poly_graph(graph)
```

`5` 正好对应论文条件中的：

$$
2d_{thres}-1=2\times3-1=5
$$

说明源码确实尝试落实 Theorem 2。

但工程实现有两个风险：

```text
1. 只 repeat 一次，而不是循环到条件真正满足；
2. wildcard、boundary 和 path 索引在 repeat 前计算，repeat 后没有全部重新计算。
```

因此非常短 RU 的扩展是否严格满足论文条件，需要用人工图做回归测试，不能只根据代码注释认定正确。

### 20.4 源码的“3D descriptors”究竟是什么

公开命令只使用两个 knowledge vectors：

| 名称 | 维度 | 源码实际生成方式 |
|---|---:|---|
| `md` | 200 | `RDKit2DNormalized`，直接由 SMILES 计算 |
| `atom_pair_3d` | 512 | RDKit embedding + MMFF94 + hashed atom-pair fingerprint，`use2D=False` |

这里存在一个重要的论文/源码差异：

```text
论文把空间分支整体称为 3D descriptors；
但公开源码的 md-200 实际是 normalized RDKit 2D molecular descriptors，
只有 atom_pair_3d-512 明确依赖生成的 3D conformer。
```

`atom_pair_3d` 的构象逻辑为：

```text
MolFromSmiles
-> AddHs
-> EmbedMolecule（默认参数）
-> MMFFOptimizeMolecule(MMFF94)
-> RemoveHs
-> hashed atom-pair bit vector, 512 bits, use2D=False
```

源码没有：

```text
多构象搜索；
最低能量 conformer 选择；
ETKDG 参数配置；
MMFF 收敛质量门；
UFF fallback；
conformer ensemble。
```

3D fingerprint 失败时直接返回全零向量。因此原公开实现的 3D branch 比当前项目的 PolyGen periodic geometry 简单得多。

### 20.5 Descriptor disturbance

预训练 collator 不仅 mask 30% atom features，还以 `kvec_mask_rate=0.3` 扰动两个 descriptor：

```text
md continuous descriptor:
  随机选择 30% 位置，替换为 [0,1] 随机数

atom_pair_3d binary fingerprint:
  随机选择 30% bits，执行 0/1 翻转
```

但是 loss 只对 masked atoms 计算 cross entropy。源码虽然返回 descriptor labels/masks，`CrossEntropyMAE` 并没有计算 descriptor reconstruction loss。

因此真实目标是：

```text
在带噪 descriptor 条件下预测 masked atom type
```

而不是同时重建 descriptor。这项正则化在论文正文中没有说明。

### 20.6 Graph Transformer 实际实现

源码每层采用：

```text
pre-LayerNorm attention
residual
pre-LayerNorm FFN
GELU
FFN hidden = 4 * d_model
residual
```

而论文公式写成 post-LayerNorm，FFN 描述为 ReLU。二者存在实质差异。

源码 attention 还有以下细节：

```text
Q/K/V 共享一次 Linear(d, 3d)
heads = 8
query scale = d_model^(-0.5)
distance bias = 单个 scalar，跨 heads 共享
path bias = 单个 scalar，跨 heads 共享
distance/path bias 只在进入全部 Transformer layers 前计算一次
```

标准 multi-head attention 通常按 `head_dim^(-0.5)` 缩放；源码使用 `d_model^(-0.5)`，会让 logits 比标准缩放更小。

### 20.7 Path attention 不是 bond-path encoding

源码 `PathAttentionScore` 对 shortest path 中每个位置的 **node feature** 做线性映射并取平均：

```text
path node representations
-> position-specific Linear(d,1)
-> average over valid path positions
-> scalar path bias
```

它没有读取源码中定义的 14-dimensional bond features。因此：

```text
MIPS 公开源码 = node-path bias
当前项目 = multi-hop path-bond bias
```

当前项目的 bond-path encoding 是更接近 Graphormer edge encoding 的扩展，但不是公开 MIPS 源码的逐行复现。

### 20.8 Cross-modal fusion 的真实实现

公开源码把每个 knowledge vector 投影成一个 token：

```text
md-200 -> key 128 / value 512
atom_pair_3d-512 -> key 128 / value 512
atom token -> query 128
```

每个原子只在这两个 descriptor tokens 上执行 attention：

$$
a_i=\operatorname{Softmax}(q_iK^T)
$$

$$
x_i'=x_i+0.5\,a_iV
$$

与论文公式相比，源码有三项差异：

```text
1. 输出 residual 固定乘 0.5；
2. fusion 后没有 LayerNorm；
3. scale 使用 sqrt(d_model)，而 query/key 真实维度是 d_model/4。
```

此外，fusion 只在全部 Graph Transformer layers 结束后执行一次。Descriptor 不参与每一层 topology attention，而是只修正最终 atom representations。

### 20.9 Backbone embedding 的真实实现

源码没有单独定义：

```python
nn.Embedding(2, d_model)
```

而是把 137-dimensional atom feature 的第 40 位设为 1：

```python
graph.ndata['h'][backbone_indices, 40] = 1
```

之后通过输入 Linear 投影，间接形成可学习的 backbone vector。

该第 40 位位于 atomic-number one-hot 区域，不是显式新增的 feature slot。作者限定的元素集合不包含对应元素，因此正常数据中该位大概率原本为零，但这种实现依赖数据元素范围，不够稳健。

源码 backbone 还存在两点与论文文字不同：

```text
只标记 wildcard-to-wildcard shortest path；
没有把 shortest path 穿过的完整 rings 全部扩展为 backbone。
```

当前项目使用独立 backbone role embedding，并扩展完整穿越环，工程语义更明确。

### 20.10 Masked-atom loss

源码将被 mask 原子的完整 137-dimensional feature vector 乘零，标签取前 101 维 atomic-number one-hot 的 argmax。

Cross entropy 只在 mask 位置求平均：

$$
L=\frac{\sum_i m_i\operatorname{CE}(\hat y_i,y_i)}{\sum_i m_i}
$$

这确认论文的“mask complete atom features”和“predict atom type”描述准确。

### 20.11 Mean pooling 与 Graph Token

微调模型默认：

```text
readout = mean
```

源码没有 Graph Token。这与论文 Theorem 1/2 的 mean-pooling 前提一致。

因此当前项目中的 Graph Token readout 应作为独立扩展进行消融，不能直接认为它与 MIPS 理论严格等价。

### 20.12 公开微调脚本没有保留完整多模态分支

这是源码审计中最重要的复现问题。

README 的微调示例传入：

```text
--kvec_names=[md,atom_pair_3d]
```

但 `scripts/finetune.py` 的 argparse 并未定义这个参数，命令按当前 commit 原样执行会报告 unknown argument。

更关键的是，脚本内部固定：

```python
model = 'base_graph_transformer'
```

而不是预训练使用的：

```python
kfuse_graph_transformer
```

它也没有向下游 dataset 传入 `kvec_names`。Checkpoint 使用 `load_no_strict=True`，因此预训练 checkpoint 中的 descriptor fusion 权重会成为 unexpected keys，而 topology backbone 可以被加载。

所以该公开微调脚本实际执行的是：

```text
加载预训练 topology Transformer
-> 丢弃 k_fusion 模块
-> topology-only downstream fine-tuning
```

这与论文 Figure 1 和 Table 5 所描述的完整 `2D + BE + 3D` 模型不一致。可能存在未公开的实验脚本或 README 已过期，但仅凭当前公开仓库无法完整复现论文的多模态微调结果。

### 20.13 Backbone 在预训练与微调中的配置不一致

公开预训练命令没有设置：

```text
dataset.mol_graph_form_cfg.main_chain_embed=True
```

其默认值是 `False`。而微调脚本明确设置为 `True`。

按当前代码路径理解：

```text
预训练：没有 backbone indicator
微调：新增 backbone indicator
```

这与论文“完整 MIPS 预训练包含 backbone embedding”的直观理解不一致，也使 backbone feature 对应的输入投影参数无法在预训练中得到有效学习。

### 20.14 五折协议的公开源码实现

源码使用：

```python
KFold(n_splits=5, shuffle=True, random_state=1)
```

但 split 保存函数被调用为：

```python
save_scaffold_split(split_file, train_idx, valid_idx, valid_idx)
```

即同一个 held-out fold 同时作为 validation 和 test。Trainer 每个 epoch 同时计算二者，并按 validation RMSE 选择 best checkpoint。

由于 validation 与 test indices 相同，这相当于使用 test fold 选择 epoch，可能产生偏乐观结果。论文正文只写 5-fold cross-validation，没有披露这一细节。

这并不否定 MIPS 架构本身，但说明当前项目要与论文数值公平比较，必须明确区分：

```text
论文/公开源码协议结果
严格 train/validation/test 分离结果
```

### 20.15 公开源码复现可信度结论

可直接确认的部分：

```text
star linking
2-hop localized topology attention
distance/path scalar bias
mean pooling
masked atom p=0.30
PI1M 约 1M、20k steps、batch 1024
md + atom_pair_3d descriptor-conditioned pretraining
```

不能依赖公开脚本完整复现的部分：

```text
论文完整 2D+BE+3D downstream model
论文所称全部 3D descriptors 的严格定义
pretraining 中 backbone embedding
严格独立 validation/test 的五折结果
论文公式中的 post-LN/ReLU 版本
```

## 21. 基于源码审计修正后的复现建议

### 21.1 忠实论文 baseline 与忠实源码 baseline 要分开

建议同时定义：

```text
MIPS-paper:
  post-LN + ReLU + descriptor fusion LayerNorm
  按论文公式实现

MIPS-code:
  pre-LN + GELU + 2-hop sparse attention
  md-200 + atom_pair_3d-512
  residual fusion scale 0.5，无 fusion LayerNorm
```

否则出现差异时无法判断是论文公式还是公开代码带来的。

### 21.2 当前项目应优先做的三个对照

```text
1. Current topology vs MIPS-code topology
   固定相同 atom features、mean pooling 和 masked-atom task。

2. Descriptor off vs md+atom_pair_3d
   复现公开源码最小 descriptor branch，不先加入六组 descriptors。

3. Mean pooling vs Graph Token
   检验当前 Graph Token 是否破坏 repeat/shift 稳定性或产生过拟合。
```

### 21.3 不建议复制的源码问题

以下行为应保留为复现开关，但不应成为当前项目默认：

```text
validation 与 test 使用相同 fold；
在 atomic-number one-hot 内复用 backbone bit；
3D 构象失败静默返回全零；
repeat 后不重新计算 attachment/path indices；
以 d_model 而不是 head_dim 缩放 multi-head logits；
非严格加载后静默丢弃 multimodal fusion weights。
```

### 21.4 对当前性能问题的新判断

源码审计后，不能把当前结果未超过论文简单归因于 PolyGen PBC。还存在至少四个更直接的协议差异：

```text
1. 原论文训练数据约 1M，当前主实验通常只有 PI1M-50k；
2. 论文公开协议的 held-out fold 同时用于 validation/test；
3. 原源码以极简单 masked-atom task获得高吞吐量；
4. 当前 Graph Token、多任务和 PBC 双分支均不是原 MIPS baseline。
```

因此应先在当前严格协议下重跑 `MIPS-code topology-only`，再逐项加入 descriptor、PBC 和多模态融合。

## 22. 最终评价

MIPS 最有价值的创新是 **用 star-linked quotient graph 把无限重复拓扑转化为有限计算，并明确研究 repeat/shift invariance 与 quotient-graph 表达能力**。这一部分有清楚的理论动机，也得到 RSIT 实验支持。

其 3D 部分更保守：它通过单体 descriptor tokens 为拓扑原子提供空间先验，计算高效且消融结果稳定，但不能等价表示真实无限聚合物链的周期 3D 环境。

对当前项目而言，最合理的定位不是“用 PolyGen PBC 替换 MIPS”，而是：

```text
MIPS topology = 无限链拓扑基础
MIPS descriptors = 低成本全局空间先验
PolyGen periodic geometry = 显式局部周期环境扩展
SMILES/FP = 额外语义与结构先验
```

下一步最重要的不是继续堆叠新模块，而是建立忠实 MIPS baseline，并用单变量消融确认 descriptors、PBC、复杂预训练任务分别带来了什么。只有这样，才能解释当前模型为何尚未稳定超过 MIPS，而不是把所有性能差异归因于 PBC 构造。
