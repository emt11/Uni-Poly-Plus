# MolGT 论文分析：Multiview / Multimodal Graph Transformer Pretraining

分析对象：`1-s2.0-S1566253524005621-main.pdf`

论文题目：Pretraining graph transformer for molecular representation with fusion of multimodal information

期刊：Information Fusion, 2025

代码：`https://github.com/robbenplus/MolGT`

本分析基于论文 PDF 正文阅读，重点关注 MolGT 的 2D/3D 共享 Transformer、node-level 和 graph-level 预训练任务、implicit 3D geometry，以及这些设计对当前 Uni-Poly-Plus 项目的可借鉴价值。

## 1. 论文核心问题

论文关注分子表示学习中的多视角、多模态预训练。

作者认为已有 molecular representation learning 主要有三个不足：

```text
1. 很多方法只做 2D graph 预训练，缺少 3D geometry 约束。
2. 一些 2D-3D 方法只做粗粒度 contrastive alignment，没有充分利用 node-level 和 graph-level 视角。
3. 真实下游任务中，生成 3D conformer 有成本，推理阶段最好只用 2D 输入也能吸收 3D 知识。
```

因此论文提出：

```text
MolGT = Molecular Graph Transformer with multiview and multimodal pretraining
```

核心目标是：

```text
预训练阶段使用 2D topology + 3D geometry。
推理 / fine-tune 阶段可以只用 2D molecule。
通过 2D-3D contrastive learning 让 2D 表示学到 implicit 3D geometry。
```

这和当前 Uni-Poly-Plus 的问题高度相关：聚合物任务中 3D 构象通常更贵、更不稳定，尤其是 P-SMILES / repeat unit / periodic geometry 场景。

## 2. MolGT 的整体结构

MolGT 的整体流程是：

```text
2D molecule graph
  -> 2D atom embedding
  -> 2D topology attention bias
  -> modality-shared graph Transformer
  -> node representation + graph representation

3D molecule graph
  -> 3D atom embedding
  -> 3D distance attention bias
  -> same modality-shared graph Transformer
  -> node representation + graph representation

node-level pretext:
  2D InfoMotif
  3D coordinate denoising

graph-level pretext:
  knowledge-guided prototypical clustering
  2D/3D contrastive learning
```

它不是简单的两个 encoder 对齐，而是一个共享 backbone 加少量模态专用模块。

## 3. Modality-Shared Graph Transformer

MolGT 基于 Transformer-M / Graphormer 风格的 graph Transformer。

普通 self-attention：

```text
attention(i, j) = Q_i K_j / sqrt(d)
```

MolGT 加入分子结构 bias：

```text
attention(i, j) = Q_i K_j / sqrt(d)
                + Phi_2D(i, j) * I_2D
                + Phi_3D(i, j) * I_3D
```

其中：

```text
Phi_2D:
  shortest path distance encoding
  shortest path edge feature encoding

Phi_3D:
  Euclidean distance encoding
  Gaussian basis kernel
```

`I_2D` 和 `I_3D` 是输入模态开关。输入 2D 时只激活 2D bias，输入 3D 时只激活 3D bias。

## 4. 为什么不是两个完全独立 encoder

论文强调参数共享：

```text
shared:
  self-attention weights
  FFN weights
  main Transformer block

modality-specific:
  embedding layer
  layer normalization
  projection layer
  attention bias
```

这样设计有两个动机：

```text
1. 参数效率高。
   MolGT 约 16M 参数，小于 GROVER-base / GROVER-large。

2. 2D 和 3D 表示空间更容易共享知识。
   如果两个 encoder 完全独立，2D-3D 对齐更像普通双塔 contrastive learning。
```

论文还讨论了 Siamese dual encoder 容易在 Transformer 下发生 contrastive collapse，因为 2D 和 3D 输入除 bias 外过于相似，模型可能学 shortcut。

## 5. 预训练任务一：InfoMotif

InfoMotif 是 2D node-level 任务。

动机：

```text
分子性质通常由 motif / substructure / functional group 决定。
普通 atom masking 对 motif 级别结构感知不足。
```

MolGT 把 motif 内原子作为 positive pair，不同分子中的原子作为 negative pair。

正样本来源：

```text
同一 motif 中的 atom pair
```

负样本来源：

```text
minibatch 内其他 molecule 的 atom
```

motif 分解方式：

```text
1. ring motif 使用 RDKit GetSymmSSSR。
2. non-ring chemical bond 作为简单 motif。
```

InfoMotif 的作用不是直接预测 motif label，而是通过 node-level contrastive learning 让 atom representation 捕获局部化学结构。

论文 attention map 显示，加入 InfoMotif 后，模型高层 attention 不会过度集中到 graph token，而是更合理地聚合 substructure 信息。

## 6. 预训练任务二：3D Coordinate Denoising

Coordinate denoising 是 3D node-level 任务。

流程：

```text
3D coordinate r_i
-> add noise epsilon
-> Transformer with 3D distance bias
-> denoising head predicts epsilon
-> Huber loss
```

论文使用 15% atoms 加 uniform noise。

关键点：

```text
MolGT 的 backbone 不是严格 SE(3)-equivariant model。
它通过 distance bias 和 denoising head 学习几何扰动。
```

这和 PaiNN / SchNet / DimeNet 不同。MolGT 更像：

```text
Graphormer-style invariant geometry pretraining
```

而不是显式 equivariant message passing。

## 7. 预训练任务三：Knowledge-Guided Prototypical Clustering

KGPC 是 graph-level 任务，也是 MolGT 最有特色的设计之一。

作者认为普通 contrastive learning 缺少显式语义 anchor：

```text
图像-文本 contrastive learning 有自然语言作为语义监督。
分子 2D-3D contrastive learning 没有明确语义标签。
```

因此 MolGT 引入 fingerprint 作为先验：

```text
2D fingerprint: MACCS, 166 keys
3D fingerprint: USRCAT, shape + pharmacophore descriptor
```

流程：

```text
MACCS / USRCAT
-> K-Means clustering
-> 得到 prototype labels
-> graph representation 预测 prototype
-> cross-modal prototypical classification
```

论文使用多个 prototype 数：

```text
500
1000
3000
```

多个 cluster center 的融合效果最好。

直觉：

```text
fingerprint 提供 explicit chemical semantic anchors。
prototype learning 让 graph-level representation 不只是对齐同一个 molecule，
还要在 scaffold / shape / pharmacophore 层面形成有意义的簇。
```

## 8. 预训练任务四：Implicit 3D Geometry

MolGT 使用 InfoNCE 对齐同一分子的 2D graph representation 和 3D graph representation。

正样本：

```text
same molecule: z_2D and z_3D
```

负样本：

```text
minibatch 中其他 molecule 的 z_3D / z_2D
```

目标：

```text
让 2D 表示吸收 3D 表示中的几何知识。
```

论文重点不是让模型在推理时同时用 2D + 3D，而是：

```text
fine-tune / inference 时只用 2D channel，也能接近使用 3D channel 的结果。
```

这是 MolGT 对实际任务最有价值的地方。

## 9. 预训练数据和训练设置

预训练数据：

```text
GEOM-Drugs
QMugs
PCQM4Mv2
```

总计约：

```text
6M conformers
```

多构象处理：

```text
如果一个 molecule 有多个 conformer，选择较低能量的若干 conformer。
```

模型设置：

```text
8-layer graph Transformer
hidden dim = 512
FFN dim = 768
attention heads = 64
Adam
learning rate = 5e-5
weight decay = 0.001
```

训练策略：

```text
先在 <= 40 atoms 的分子上训练大部分 steps。
后续切到 <= 128 atoms 的较大分子，学习 2D / 3D bias。
```

这个分阶段策略对聚合物也有启发：可以先在短 repeat unit 上预训练，再扩展到更长 oligomer / segment。

## 10. 实验结果

MoleculeNet classification：

```text
BBBP, BACE, Tox21, ToxCast, SIDER, ClinTox, MUV, HIV
```

MolGT 平均 ROC-AUC：

```text
76.61
```

优于论文比较的多数 2D 和 implicit 3D baseline，包括：

```text
AttrMask
ContextPred
Mole-BERT
GROVER
GraphMVP
3D Infomax
MoleculeSDE
MoleBlend
```

Regression：

```text
ESOL
Lipo
Malaria
CEP
```

MolGT 平均 RMSE：

```text
0.9305
```

在 ESOL、Malaria、CEP 上优于 baseline，但 Lipo 不占优。

QM9 quantum property：

论文只用 2D topology 推理，MolGT 在多个 quantum properties 上优于普通 2D pretraining 和 3D Infomax，但明显不如 all-atom 3D 模型 GNS-TAT。

原因是：

```text
MolGT 预训练只用 heavy atoms，不包含 hydrogen。
QM9 quantum properties 对 all-atom conformation 很敏感。
```

这也是 MolGT 的一个明确边界。

## 11. Ablation 结论

论文 ablation 说明：

```text
1. 单独 node-level 任务不够。
2. 单独 graph-level 任务也不够。
3. 2D 和 3D modality 同时参与预训练更好。
4. KGPC 明显优于普通 prototypical clustering。
5. InfoMotif + DN + KGPC + CL 的完整组合最好。
```

分类任务平均 ROC-AUC：

```text
No pretrain: 67.67
CL only: 68.52
KGPC + CL: 72.23
IM + DN: 68.92
DN + KGPC + CL: 75.67
IM + KGPC + CL: 74.82
IM + DN + PC + CL: 73.17
IM + DN + KGPC + CL: 76.61
```

最关键的结论：

```text
普通 2D-3D contrastive learning 不够。
需要 node motif、3D denoising、fingerprint prototype 共同约束。
```

## 12. 对 Uni-Poly-Plus 的启发

### 12.1 可以借鉴：Implicit 3D for polymer

当前项目如果每个 repeat unit / oligomer 都生成 3D，会遇到：

```text
1. P-SMILES 星号处理复杂。
2. polymer conformer 不唯一。
3. 周期结构 / capped oligomer / repeat unit 的 3D 语义不一致。
4. 大规模预训练时构象生成成本高。
```

MolGT 提供了一个方向：

```text
预训练阶段使用可用的 3D / pseudo-3D。
下游阶段允许只用 2D graph / P-SMILES graph。
```

对 Uni-Poly-Plus 可以设计：

```text
2D polymer graph encoder
3D repeat-unit / capped oligomer geometry encoder
2D-3D contrastive alignment
fine-tune 时只用 2D 或 2D + optional geometry
```

### 12.2 可以借鉴：Polymer-specific InfoMotif

MolGT 的 InfoMotif 用 ring + non-ring bond。

聚合物中可以扩展为：

```text
aromatic ring
heteroatom functional group
backbone segment
side-chain segment
star-neighborhood motif
ester / amide / imide / ether / sulfone group
donor / acceptor motif
```

这比随机 mask atom 更贴近聚合物性质。

例如 Tg / Tm / modulus 相关结构可能不是单原子，而是：

```text
backbone rigidity motif
side-chain bulkiness motif
polar group motif
hydrogen-bond motif
```

### 12.3 可以借鉴：Fingerprint-guided prototype

Uni-Poly-Plus 已有 fingerprint 模态时，不一定只把 fingerprint 作为下游输入。

可以把 fingerprint 用作预训练先验：

```text
Morgan / MACCS / RDKit descriptor
polymer handcrafted descriptor
-> clustering
-> prototype label
-> graph-level representation 预测 prototype
```

对聚合物更合适的 prototype 先验可能包括：

```text
repeat-unit fingerprint
backbone fingerprint
functional group count
polymer class label
LLM/KG extracted polymer category
```

这与当前项目里的 KG / LLM mapping 方向可以结合。

### 12.4 可以借鉴：modality-shared backbone

当前项目如果有多模态：

```text
SMILES
fingerprint
2D graph
3D geometry
polymer KG features
```

一种常见做法是各自 encoder 后 concat。

MolGT 提醒：

```text
2D graph 和 3D graph 可以共享 Transformer backbone，
只把 embedding / bias / normalization 做成 modality-specific。
```

这有利于：

```text
参数效率
跨模态知识共享
避免下游小数据过拟合
```

### 12.5 不宜直接照搬：USRCAT

MolGT 的 3D prototype 使用 USRCAT。

对小分子这是合理的，但对聚合物 repeat unit 未必稳定：

```text
repeat unit 的 3D shape 不等于 polymer chain 的 3D shape。
capped oligomer 的 shape 受 capped group 和链长影响。
P-SMILES star atoms 会影响 RDKit conformer。
```

如果迁移到聚合物，建议先替换为更稳定的 polymer geometry descriptor：

```text
intra-repeat distance statistics
backbone end-to-end distance
radius of gyration for capped oligomer
periodic cell / PBC descriptors
torsion distribution
```

## 13. 可落地实现建议

对当前 Uni-Poly-Plus，更实际的第一步不是完整复刻 MolGT，而是拆成三个阶段：

```text
阶段 1：polymer prototype pretraining
  repeat-unit fingerprint / KG polymer class
  -> K-Means prototype
  -> graph embedding predicts prototype

阶段 2：polymer motif contrastive learning
  基于 SMARTS / RDKit / star-neighborhood 构造 motif
  同 motif 内 atom positive
  不同样本 atom negative

阶段 3：optional 2D-3D alignment
  2D repeat-unit graph
  3D capped repeat-unit / oligomer geometry
  InfoNCE alignment
  下游允许 no-3D inference
```

如果只选一个最值得尝试的模块：

```text
KGPC / fingerprint-guided prototype
```

原因：

```text
1. 不依赖昂贵 3D。
2. 与当前 fingerprint / KG pipeline 兼容。
3. 可以直接作为 graph-level SSL 任务。
4. 对小数据 polymer property prediction 可能收益明显。
```

## 14. 论文局限

MolGT 的主要局限：

```text
1. 主要面向小分子，不处理聚合物 repeat unit、P-SMILES、周期结构。
2. 只使用 heavy atoms，QM9 这类 all-atom quantum task 受限。
3. 3D geometry 是 distance-bias style，不是严格 equivariant architecture。
4. implicit 3D 能否泛化到构象高度多样的聚合物，需要重新验证。
5. prototype 依赖 MACCS / USRCAT，小分子先验不一定适合聚合物。
```

## 15. 总结

MolGT 的核心价值不是提出一个全新的 Transformer block，而是把预训练目标组织得比较完整：

```text
node view:
  InfoMotif
  3D denoising

graph view:
  fingerprint-guided prototype
  2D-3D contrastive learning

modality design:
  shared Transformer backbone
  modality-specific embedding / norm / projection / bias

deployment:
  pretrain with 3D
  infer with 2D
```

对 Uni-Poly-Plus 最值得借鉴的是：

```text
1. 用 fingerprint / KG label 做 graph-level prototype SSL。
2. 用 polymer motif 做 node-level contrastive SSL。
3. 预训练阶段吸收 3D，fine-tune 阶段允许无 3D。
```

相比直接增加一个 3D encoder，MolGT 更像是在回答一个实际问题：

```text
当 3D 有价值但不总是可靠可用时，如何把 3D 知识蒸馏进 2D 分子表示。
```
