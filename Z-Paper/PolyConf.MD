# PolyConf 论文与源码分析：层级聚合物构象生成

分析对象：

- 论文：[PolyConf: Unlocking Polymer Conformation Generation through Hierarchical Generative Models](https://openreview.net/forum?id=BsTLUx38qV)
- 会议：ICML 2025, PMLR 267, 63002-63015
- 代码：[FanmengWang/PolyConf](https://github.com/FanmengWang/PolyConf)
- 本次源码审计提交：`103d6655918288934b9108733a1d6914a721ed37`

本文同时基于论文、附录和公开源码，重点回答：

```text
PolyConf 如何从重复单元构象组装完整聚合物链？
它是否等价于 PBC？
它如何处理同一链中不同 RU 构象？
哪些设计可迁移到当前 Polymer-SCAGE？
```

## 1. 研究问题

PolyConf 面向有限长度线性均聚物的构象生成。

输入可以概括为：

```text
聚合物 2D 化学结构
重复单元定义
链中重复单元数量
```

输出为：

```text
整条有限聚合物链的原子级 3D 坐标
```

它不是性质预测模型，也不是无限周期晶胞生成器。

## 2. 核心判断：不能把聚合物视为同一刚性 RU 的重复

论文最重要的建模判断是：

```text
同一条聚合物中的不同重复单元可以具有不同局部构象。
```

因此不能简单地：

```text
生成一个 RU 坐标
-> 用同一个刚体变换重复复制
```

PolyConf 将完整构象分解成：

```text
局部变量：每个 RU 的内部构象 Cu_i
全局变量：每个 RU 的 orientation transformation O_i
```

形式上：

```text
C = Assemble({Cu_i}, {O_i})
O_i = (R_i, t_i) in SE(3)
```

## 3. 为什么采用层级生成

直接对约 2,000 个原子的完整坐标做扩散会遇到：

```text
原子数太多
构象空间维度太高
局部化学与全局链形态耦合
长链生成难以扩展
```

PolyConf 把问题拆成：

```text
Phase 1：生成所有 RU 的局部构象
Phase 2：生成 RU 的相对旋转并组装全链
```

这种分解是其可扩展到 20-100 个 RU 和约 2,000 个原子的主要原因。

## 4. 重复单元定义

PolyConf 的 RU 不只是化学意义上的最小 repeat unit。

为了保证相邻单元可以连续拼接，每个 RU 构象包含：

```text
本单元原子
两个与相邻单元重叠/连接的关键原子
```

论文用四个 key points 定义局部 frame 和拼接约束。

源码中的 `key_point_index` 用于：

```text
从 RU 坐标构造 frame
确定前后单元的重叠连接点
根据旋转结果推导平移
```

## 5. 局部构象与标准坐标系

每个 RU 的局部构象先变换到标准坐标系。

这样，Phase 1 只需要学习：

```text
RU 内部原子的相对几何
```

而不需要同时学习该 RU 在整条链中的全局位置和朝向。

全局朝向由 Phase 2 的 `R_i` 决定。

## 6. Phase 1：多模态 RU Encoder

每个 RU 的表示由 2D 和 3D 两部分组成：

```text
2D graph encoder
3D repeat-unit encoder
```

源码中全聚合物 2D encoder 可选：

```text
GIN / GCN
默认 GIN
5 layers
embedding 300
output 512
mean pooling
```

RU 3D encoder 采用 Uni-Mol 风格 pair-aware Transformer：

```text
atom token embedding
pairwise distance Gaussian basis
edge type conditioning
TransformerEncoderWithPair
```

源码默认值：

```text
encoder layers = 15
embedding dim = 512
FFN dim = 2048
attention heads = 64
Gaussian kernels = 128
```

## 7. Phase 1：Masked Autoregressive Modeling

假设链有 `Nu` 个 RU。

训练时随机打乱 RU 生成顺序，并随机 mask 一部分 RU：

```text
known RU conformations
-> MAR encoder
-> masked RU latent tokens
-> MAR decoder
-> local conformation diffusion model
```

mask ratio 从截断分布采样，源码默认下界为：

```text
mar_mask_ratio_lower_bound = 0.7
```

这意味着训练常常要求模型在只看到少量 RU 构象时生成大部分链的局部构象。

## 8. 为什么是 autoregressive + diffusion

单纯 autoregressive regression 容易输出均值构象，无法覆盖多模态分布。

单纯对所有 RU 同时扩散又难以建模不同 RU 构象之间的条件依赖。

PolyConf 的组合是：

```text
MAR 决定“基于哪些已知 RU 生成哪些未知 RU”
diffusion 决定“每个未知 RU 的连续 3D 构象分布”
```

这使同一 polymer identity 可以生成多个不同的局部构象组合。

## 9. Phase 1 的局部扩散

局部构象扩散参考分子 conformer diffusion。

公开参数包括：

```text
hidden scalar features ns = 32
vector features nv = 8
convolution layers = 4
radius cutoff = 5 Å
distance embedding dim = 50
sigma min = 0.01*pi
sigma max = pi
inference steps = 20
```

训练目标是 diffusion score matching，而不是直接坐标 MSE。

源码 Phase 1 最终 loss 主要是：

```text
L_phase1 = mean((score_target - score_pred)^2 / score_norm)
```

## 10. Phase 1 的条件信息

MAR 使用：

```text
RU multimodal embedding
whole-polymer graph embedding
polymer SMILES representation
RU sequence position embedding
```

源码中的 polymer-level condition 包含 768 维 `psmi_rep`。

训练时以一定概率丢弃 class condition：

```text
mar_label_drop_prob = 0.1
```

这用于 classifier-free guidance 风格的条件生成。

## 11. Phase 2：为什么主要生成旋转

论文将每个 RU 的 orientation 表示为：

```text
O_i = (R_i, t_i)
```

但相邻 RU 在连接原子处必须重合，因此平移不是完全自由变量。

给定：

```text
局部 RU 坐标
每个 RU 的旋转 R_i
相邻连接 key points
```

可以通过累计连接位移推导 `t_i`。

因此模型核心只扩散旋转：

```text
SO(3) diffusion for R_i
```

而不是独立学习不受约束的 SE(3) translation。

## 12. 源码中的平移递推

源码 `rot2trans()` 的逻辑是：

```text
1. 对每个 RU 的局部坐标应用 R_i。
2. 计算前一 RU 的右连接点与当前 RU 的左连接点差。
3. 对这些差做 cumulative sum。
4. 得到各 RU 的全局 translation。
```

概念上：

```text
t_1 = 0
t_i = t_{i-1} + x_right^(i-1) - x_left^i
```

这一设计保证相邻 RU 在共享连接点处连续。

## 13. Phase 2 的 SO(3) Diffusion

Phase 2 使用 FrameDiff/OpenFold 风格的 frame diffusion 和 IPA score model。

公开配置：

```text
node embedding = 256
edge embedding = 128
IPA heads = 8
IPA blocks = 4
query/key points = 8
value points = 12
SO(3) sigma: 0.1 -> 1.5, logarithmic schedule
inference diffusion steps = 500
```

Phase 2 从 Phase 1 checkpoint 初始化，并在 forward 中对 Phase 1 的多个 encoder 表示使用 `detach()`。

因此源码实际执行更接近：

```text
冻结/阻断 Phase 1 表示梯度
-> 专门训练 orientation assembly
```

## 14. Phase 2 损失

主要损失为：

```text
L_phase2 = L_rotation + L_distance_matrix
```

旋转损失可以拆成：

```text
rotation axis loss
rotation angle loss
```

公开配置：

```text
rotation weight = 0.5
rotation loss t threshold = 0.2
```

辅助距离矩阵损失：

```text
只监督参考距离 < 6 Å 的原子对
dist weight = 1.0
aux total weight = 0.25
只在 t < 0.25 时启用
```

这一点表明局部邻接距离用于稳定全链组装，但并不要求全局所有原子对逐点匹配。

## 15. 最终组装

推理流程：

```text
生成每个 RU 的局部构象
-> 生成每个 RU 的 R_i
-> 由连接点递推 t_i
-> 对局部原子应用 (R_i, t_i)
-> 合并重叠连接原子
-> 得到完整有限链坐标
```

这与当前 screw 的根本区别是：

```text
PolyConf: 每个 RU 有独立 R_i 和独立局部构象。
当前 screw: 所有周期像共享同一个 RU 构象和同一个 `(R,T)`。
```

## 16. PolyBench 数据集

论文构建了 force-field MD 数据集：

```text
总样本：超过 50,000 个 polymer conformations
每个构象：约 2,000 个原子
RU 数：多数约 20-100，少量超过 100
train / valid / test：约 46k / 5k / 2k
```

数据流程：

```text
1D polymer SMILES
-> RDKit monomer
-> define head/repeat/tail units
-> Antechamber + prepgen + GAFF
-> TLeap build chain
-> ACPYPE convert to GROMACS
-> steepest-descent minimization
-> 298 K, 1 atm, 5 ns MD
```

论文正文将其称为 NVT，但同时写了 298 K 和 1 atm；严格复现时需要核对脚本中的 ensemble/barostat 设置，不能只依据一句文字。

## 17. 数据含义的限制

PolyBench 表示的是：

```text
有限长、带端基、室温 force-field MD 链构象
```

不是：

```text
无限周期链
0 K DFT 最低能结构
体相多链 packing
```

因此它适合学习 conformational ensemble，不适合直接作为严格 PBC ground truth。

## 18. 训练配置

官方脚本的 Phase 1/2 都使用：

```text
8 GPUs
batch size per GPU = 1
workers = 8
lr = 1e-4
max epochs = 600
warmup ratio = 0.06
Adam beta = (0.9, 0.99)
gradient clip = 1.0
patience = 30
```

这说明完整复现成本很高，不适合直接嵌入当前 24 h 预训练预算。

## 19. 评价指标

论文使用结构和能量的 Matching：

```text
S-MAT-R: reference 到 generated set 的最近 RMSD
S-MAT-P: generated 到 reference set 的最近 RMSD
E-MAT-R/P: 将 RMSD 换成 potential-energy difference
```

作者认为小分子常用固定 RMSD threshold Coverage 不适合长链，原因是聚合物尺度和柔性差异太大。

附录仍报告 `delta = 25 Å` 的 Coverage 供参考。

## 20. 主要实验结果

PolyConf 的主要结果：

```text
S-MAT-R mean / median = 35.021 / 24.279 Å
S-MAT-P mean / median = 46.861 / 37.996 Å
E-MAT-R mean / median = 0.933 / 0.359
E-MAT-P mean / median = 6.191 / 4.122
```

平均生成时间：

```text
PolyConf 0.40 min
TorsionalDiff 0.45 min
MCF 1.12 min
GeoDiff 3.54 min
```

当推理链长翻倍、约 4,000 原子时：

```text
S-MAT-R mean / median = 65.040 / 41.992 Å
S-MAT-P mean / median = 84.626 / 64.445 Å
```

绝对 RMSD 仍然很大，但相对基线更好。不能将这些值解释为原子级精确复现。

## 21. 是否属于 PBC 方法

结论：不是。

PolyConf 生成：

```text
finite chain with per-RU variable conformations
```

它没有：

```text
周期晶胞
minimum-image distance
统一 screw generator g=(R,T)
首尾周期 closure
```

但它对 PBC 研究仍有关键意义：

```text
单个 RU screw 失败可能不是优化器问题，
而是“一步周期重复同一局部构象”的假设不适合该 polymer。
```

## 22. 与当前项目的对照

| 项目 | PolyConf | 当前 Polymer-SCAGE |
|---|---|---|
| 链类型 | 20-100 RU finite chain | one-RU infinite screw + fallback |
| RU 构象 | 每个 RU 可不同 | 周期像必须相同 |
| 变换 | 每个 RU 独立 `R_i`, `t_i` | 单一 `(R,T)` 递推 |
| 3D 来源 | 5 ns GAFF MD | RDKit + MMFF/UFF optimized trimer |
| 主模型 | MAR + local diffusion + SO(3) diffusion | SCAGE topology/geometry Transformer |
| 输出 | 全链坐标 | property embedding/prediction |
| 周期 | 无 | 有 |

## 23. 对当前项目最有价值的帮助

### 23.1 支持自适应多 RU 周期单元

PolyConf 表明相邻 RU 的局部构象可以变化。

因此严格周期模型应允许：

```text
m-RU conformational motif
-> repeat the motif periodically
```

而不是强制：

```text
1 RU -> same geometry forever
```

推荐的 `m`：

```text
1, 2, 3, 4, 6
```

只要 m-RU supercell 首尾严格 closure，它仍然是无限周期链。

### 23.2 `(R,T)` 不应只从整体 Kabsch 判断

PolyConf 用连接 key points 显式保证相邻单元连续。

当前项目可借鉴：

```text
先保证 boundary connection geometry，
再优化周期变换和内部 torsion，
而不是先自由优化三聚体后再拟合同一个 R,T。
```

### 23.3 几何监督应聚焦连接邻域

PolyConf 的 distance loss 只监督 `<6 Å` 的局部原子对。

当前 screw geometry task 可区分：

```text
boundary local geometry: 高权重
same-RU internal geometry: 中权重
long-range all-pair geometry: 低权重或仅作为 bias
```

### 23.4 构象系综应来自链上下文

当前 top-k conformers 若只来自单 RU 或三聚体，可能低估长链 torsion correlation。

可行的轻量替代是：

```text
构建 5-9 RU finite chain
短程 force-field relaxation
截取中心 m-RU motif
检验周期 closure 或作为 finite-chain context
```

### 23.5 预训练可增加相对 orientation 任务

当前 M4P 预测 screw operation，可以扩展为：

```text
预测 neighbor RU relative rotation class
预测 boundary translation/rise
预测 m-RU motif position index
```

这比直接生成全链便宜，也更适合 SCAGE encoder。

## 24. 不建议直接迁移的部分

### 24.1 不建议迁移完整生成模型

Phase 1/2 共包含大规模 Transformer、局部 diffusion、SO(3) diffusion 和 IPA，训练成本明显超过当前预算。

### 24.2 不建议把 PolyBench 当作严格 PBC 标签

有限链端部和室温 MD 会引入非周期效应；直接拟合 one-step `(R,T)` 很可能继续失败。

### 24.3 不建议将其 RMSD 作为当前 quality gate

长链整体 RMSD 随链长增长，不能与当前按 RU 对齐的 Kabsch RMSD 直接比较。

### 24.4 不建议复制“每个 RU 独立 orientation”到无限周期模型

无限链如果每个 RU 变换完全独立，就不再有有限周期晶胞。必须通过 m-RU motif 或统计链上下文保持模型定义清晰。

## 25. 推荐落地路线

### Phase A：诊断 one-RU 假设

对 strict screw 失败样本：

```text
构建 5-RU optimized chain
计算相邻 RU transform g1...g4
分析 rotation/rise 方差
```

如果相邻变换差异大，说明失败来自 one-RU symmetry mismatch，而不是简单优化失败。

### Phase B：自适应 m-RU supercell

```text
for m in [1,2,3,4,6]:
  构建连接的 m+2 RU context
  优化中心 m RU
  检验首尾 periodic closure
  选择最小 m 的合格周期单元
```

### Phase C：SCAGE 表示

```text
原子 token 保留 cell position id
周期像按 exact cell transform 生成
对应 RU 原子在 readout 前做 cell-average
增加 periodic_unit_size embedding
```

### Phase D：有限链 context 消融

保留一个不声称严格 PBC 的对照：

```text
explicit 5-RU finite context
center-RU readout
```

将其与 strict adaptive periodic 表示比较，判断性质预测是否真的需要周期性而非普通邻居上下文。

## 26. 对当前创新点的结论

PolyConf 最有价值的理论支持是：

```text
聚合物链不能普遍被建模为单一刚性 RU 的重复。
```

因此当前项目不应把 `one-RU screw >95%` 作为唯一成功目标。

更可辩护的创新点是：

```text
adaptive periodic-cell and finite-context geometry encoding
for polymer Graph Transformers
```

## 27. 总结

PolyConf 的核心不是“更强的 3D diffusion”，而是正确分解了长链构象：

```text
local RU conformation diversity
+ relative RU orientation sequence
= full-chain conformation
```

它对当前项目提出了一个必须正视的模型边界：

```text
严格 one-step screw 是一种特殊对称类别，
不是所有聚合物链的通用表示。
```

## 28. 参考链接

- [ICML/OpenReview 论文页面](https://openreview.net/forum?id=BsTLUx38qV)
- [PMLR 论文条目](https://proceedings.mlr.press/v267/wang25bo.html)
- [PolyConf 官方代码](https://github.com/FanmengWang/PolyConf)
- [项目主页](https://polyconf-icml25.github.io/)
