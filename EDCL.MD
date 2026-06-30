# EDCL 论文分析：Energy-Guided Denoising Contrastive Learning

分析对象：`energy-guided-denoising-contrastive-learning-for-molecular-property-prediction.pdf`

论文题目：Energy-Guided Denoising Contrastive Learning for Molecular Property Prediction

期刊：Journal of Chemical Information and Modeling, 2026, 66, 4245-4256

代码：`https://github.com/wangjx22/EDCL`

本分析基于论文 PDF 正文阅读，重点关注 3D 构象扰动、去噪预训练、对比学习、energy gap 监督，以及它们对当前 Uni-Poly-Plus 项目中 `geom` 模态和多阶段预训练的可借鉴内容。

## 1. 论文核心问题

论文关注 3D 分子表示学习中的一个关键问题：

```text
现有 3D molecular denoising pretraining 通常使用随机 Gaussian noise，
但随机噪声不考虑不同原子的化学环境，可能产生物理上不合理的扰动构象。
```

作者认为：

```text
合理的扰动构象应该位于原始构象附近的低能量区域。
```

因此，论文提出：

```text
Energy-Guided Denoising Contrastive Learning, EDCL
```

核心思想：

```text
clean 3D conformation
-> adaptive noise generator 生成原子级扰动
-> perturbed conformation
-> SE(3)-equivariant encoder
-> denoising + contrastive alignment + energy gap prediction
```

## 2. 整体框架

EDCL 分为两个阶段：

```text
1. Pretraining
2. Fine-tuning
```

预训练阶段：

```text
ground-truth 3D structure
-> learnable noise generator
-> perturbed 3D structure
-> shared SE(3)-equivariant Graph Transformer encoder
-> 三个任务：
   - atom-level denoising
   - clean / perturbed contrastive alignment
   - energy gap prediction
```

微调阶段：

```text
保留预训练后的 equivariant encoder
-> 接 task-specific MLP head
-> 下游分子属性预测
```

论文强调：预训练和微调中只考虑 heavy atoms。

## 3. 模型结构

EDCL 使用的是 SE(3)-equivariant Graph Transformer，基于 Equiformer V2。

主要模块：

```text
1. Embedding layer
   - atom embedding
   - edge embedding

2. Graph-based equivariant network
   - Equivariant Graph Attention
   - Feed-Forward Network
   - Separable LayerNorm

3. Output layer
   - equivariant head: 预测 atom-wise shift vectors
   - invariant head: 预测 molecule-level energy gap
```

3D 输入特征包括：

```text
atomic number
interatomic distance
unit direction vector
spherical harmonics
relative position
source / target atomic numbers
```

也就是说，它不是普通 SchNet/PaiNN 式距离消息传递，而是更强的等变 Transformer。

## 4. Adaptive Noise Generator

EDCL 最大的区别是：

```text
不用固定 Gaussian noise。
```

它训练一个 learnable noise generator：

```text
G_theta(clean_structure) -> atom-wise noise scale sigma_i
```

然后对每个原子采样：

```text
r_tilde_i = r_i + epsilon_i, epsilon_i ~ N(0, sigma_i^2 I)
```

这样每个原子的噪声大小不是固定的，而是由局部化学环境决定。

论文中的直觉是：

```text
柔性区域 / 单键 / 侧链：可以有更大扰动。
刚性芳香环 / 稳定骨架：扰动应该更小。
```

为了避免生成明显不合理的结构，论文还做了 validity check：

```text
如果任意两原子距离小于 r_min，则拒绝该扰动样本。
```

论文设置：

```text
r_min = 0.8
```

## 5. 预训练任务

EDCL 的预训练不是单一去噪任务，而是三个任务联合。

### 5.1 Equivariant Denoising

输入：

```text
perturbed structure
```

目标：

```text
预测每个原子从 perturbed position 回到 clean position 的 shift vector。
```

也就是：

```text
delta_i = r_i - r_tilde_i
```

论文强调，不直接预测绝对坐标，而预测 shift vector。

原因：

```text
shift vector 平移不变、旋转等变，比直接坐标回归更稳定。
```

这点和当前项目很相关，因为当前 `geom` denoising 也是：

```text
noisy_pos -> pred_delta -> pred_pos
```

但当前项目的噪声是固定 Gaussian noise，EDCL 的噪声是可学习的 atom-wise adaptive noise。

### 5.2 Contrastive Alignment

EDCL 把同一个分子的：

```text
clean conformation embedding
perturbed conformation embedding
```

作为正样本对。

同一个 batch 里的其他分子作为负样本。

损失是 InfoNCE：

```text
clean_i 与 perturbed_i 拉近
clean_i 与 perturbed_j, j != i 拉远
```

这比只做 positive-only consistency 更强，因为它有明确负样本。

### 5.3 Energy Gap Prediction

EDCL 还加入了一个 invariant energy head：

```text
clean structure embedding -> HOMO-LUMO energy gap
```

注意：

```text
energy loss 只作用在 clean branch，不直接作用在 perturbed branch。
```

论文这样做是为了避免 perturbation 噪声直接干扰 energy prediction 的稳定性。

这里的 energy gap 来自 DFT，是分子级全局量。

## 6. KL Regularization

Adaptive noise generator 有一个问题：

```text
如果只优化 denoising loss，noise scale sigma_i 可能塌缩到 0。
```

也就是模型学会“不加噪声”，任务变得太简单。

EDCL 使用 KL regularization 约束噪声分布接近一个合理 prior：

```text
N(0, sigma_i^2) 接近 N(0, omega^2)
```

论文中：

```text
omega = 0.1
```

作用：

```text
1. 防止 sigma_i -> 0。
2. 防止 sigma_i 过大，产生物理不合理扰动。
```

## 7. 总损失

EDCL 总损失是：

```text
L_total = L_denoise + alpha * L_contrast + beta * L_energy
```

其中 denoise loss 内部包含：

```text
denoising negative log likelihood
KL regularization
```

论文通过 grid search 选择权重。

重要结论：

```text
denoising、contrastive、energy supervision 三者不是互相替代，而是互补。
```

## 8. 实验结论

论文在 MoleculeNet、QM9、QM7 等任务上验证。

主要结论：

```text
1. EDCL 优于随机 Gaussian denoising。
2. adaptive noise generator 能生成更物理合理的扰动。
3. denoising + contrastive + energy gap 的组合优于单独任务。
4. 预训练后的模型在多个下游任务中优于 from-scratch。
```

论文特别强调：

```text
随机 noise 可能对芳香环、刚性片段产生不合理扰动，
导致模型把容量浪费在纠正无意义结构错误上。
```

Adaptive noise 的可解释现象：

```text
柔性片段噪声更大。
刚性片段噪声更小。
```

## 9. 消融实验结论

### 9.1 Adaptive noise vs random noise

替换成普通随机 Gaussian noise 后，性能下降。

原因：

```text
随机噪声没有化学环境感知能力。
```

### 9.2 KL regularization

没有 KL 时，noise generator 容易输出接近 0 的噪声。

加入 KL 后：

```text
noise scale 更稳定。
denoising 任务更有效。
```

### 9.3 Energy gap supervision

Energy gap 监督帮助模型学习：

```text
geometry <-> energy / electronic property
```

但这个监督信号必须谨慎使用，因为它如果来自下游标签，会造成 label leakage。

### 9.4 Contrastive learning

Contrastive alignment 提高模型对合理构象扰动的鲁棒性。

它和去噪任务不同：

```text
denoising 学局部几何恢复。
contrastive 学构象扰动下的分子级表示不变性。
```

## 10. 和当前 Uni-Poly-Plus 的对应关系

当前项目已经有类似方向：

```text
geom coordinate denoising
geom pairwise distance reconstruction
```

当前实现大致是：

```text
clean_pos
-> 加固定 Gaussian noise
-> geom encoder
-> coord head 预测 delta
-> pred_pos
-> coordinate loss + pairwise distance loss
```

这和 EDCL 的区别是：

```text
1. 当前 noise 是固定 std，不是 atom-wise adaptive。
2. 当前没有 noise generator。
3. 当前没有 KL regularization 防止 noise collapse，因为没有 learnable noise。
4. 当前没有 clean / perturbed geom embedding 的 batch-negative contrastive loss。
5. 当前没有 energy-gap supervision。
6. 当前使用 SchNet / PaiNN，而不是 Equiformer V2。
```

因此，当前 `geom` 预训练可以看成 EDCL 的简化 baseline。

## 11. 对当前项目最有价值的借鉴

### 11.1 将固定 Gaussian noise 改成 adaptive atom-wise noise

当前：

```text
noise = randn_like(pos) * geom_noise_std
```

可以改成：

```text
sigma_i = noise_generator(node_rep_i, local_geometry_i)
noise_i = randn(3) * sigma_i
```

其中 `noise_generator` 可以先做简单版本：

```text
atomic embedding + local degree + aromatic/ring flag + distance statistics
-> MLP
-> sigma_i
```

更强版本：

```text
用 PaiNN / SchNet 的 node representation 预测 sigma_i。
```

### 11.2 加 KL 或 scale regularization

如果引入 learnable sigma，必须防止两个极端：

```text
sigma_i -> 0：任务失效。
sigma_i 太大：构象不合理。
```

建议加入：

```text
KL / L2 scale prior
minimum distance validity check
sigma clamp
```

例如：

```text
sigma_i in [0.01, 0.3]
```

具体范围需要根据当前坐标单位和构象质量调参。

### 11.3 将 positive-only consistency 改成 batch-negative contrastive

当前项目已有多模态 contrastive，但 `geom` 内部的 clean/perturbed 对比可以单独做：

```text
clean geom embedding
perturbed geom embedding
```

正样本：

```text
同一个 SMILES 的 clean / perturbed
```

负样本：

```text
batch 中其他 SMILES 的 perturbed embedding
```

这比只要求两个 embedding 相似更合理。

### 11.4 保留 denoising，但重新审视 pairwise distance reconstruction

当前的 pairwise distance reconstruction 有价值，因为距离不依赖坐标系。

但它和 coordinate denoising 有重叠：

```text
coordinate denoising：恢复每个原子坐标。
pairwise distance reconstruction：恢复原子间距离矩阵。
```

如果保留两者，建议：

```text
1. pairwise distance 只作用在局部邻域或 cutoff 内。
2. 不要对所有 atom pair 等权重计算。
3. 对 bonded / near-neighbor pair 给更高权重。
```

理由：

```text
全 pair distance 对大分子成本高，也可能让模型过度拟合构象全局形状。
```

EDCL 本身没有把 all-pair distance reconstruction 作为主任务，而是用 denoising score / shift vector 学局部势能面。

### 11.5 Energy supervision 不能直接用下游标签

EDCL 使用 HOMO-LUMO energy gap 作为预训练监督。

当前项目如果直接使用：

```text
smi_ei.csv 的 val
smi_egc.csv 的 val
smi_tg.csv 的 val
```

作为预训练 energy supervision，会有 label leakage 风险。

更合理的方案：

```text
1. 使用外部独立数据集的 DFT energy / HOMO-LUMO gap。
2. 使用 RDKit / MMFF / UFF 计算的相对构象能量作为弱监督。
3. 只在不参与该 downstream task 的外部 pretraining set 上使用 property supervision。
4. 如果使用 smi_all 的 val，必须明确标记为 supervised pretraining，而不是无监督预训练，并且不能再声称无 label leakage。
```

对当前项目，最稳妥的是：

```text
先不用下游 property label 做 energy supervision。
优先实现 adaptive noise + contrastive + denoising。
```

## 12. 对 Graph / Geom 融合的启发

EDCL 主要是 3D 单模态预训练论文，不直接讨论 2D/3D 多模态融合。

但它给当前项目一个明确启发：

```text
Geom 模态不应该只是一个额外 encoder；
它应该通过物理合理的构象扰动预训练，学习稳定的几何势能面表示。
```

因此当前项目可以这样定位：

```text
Graph:
  star-linking graph 学聚合物拓扑周期性。

Geom:
  star-substitution capped repeat unit 学局部 3D 构象和势能面。

Fusion:
  不要求 Graph 和 Geom 使用完全相同 topology。
  Graph 解决周期拓扑消息传递。
  Geom 解决局部空间构象合理性。
```

## 13. 推荐落地路线

### Phase A：保守增强当前 Geom 预训练

不大改模型，只修改 pretraining objective：

```text
1. 保留 coordinate denoising。
2. pairwise distance loss 改成 local/cutoff distance loss。
3. 加 clean / noisy geom embedding 的 InfoNCE。
4. 不使用任何 downstream val。
```

### Phase B：加入 adaptive noise generator

新增小型模块：

```text
GeomNoiseGenerator
```

输入：

```text
atom type
degree
ring/aromatic flag
local geometry statistics
或 geom encoder 的 node representation
```

输出：

```text
sigma_i
```

加：

```text
sigma clamp
minimum distance validity check
KL / scale prior
```

### Phase C：考虑 energy proxy

如果要借鉴 EDCL 的 energy-guided：

```text
优先使用 force-field relative energy proxy。
```

例如：

```text
clean conformer UFF/MMFF energy
perturbed conformer UFF/MMFF energy
energy_delta = E_perturbed - E_clean
```

但要注意：

```text
RDKit force-field energy 对聚合物 repeat unit capped structure 只是弱近似。
不能把它当真实聚合物链能量。
```

### Phase D：更强 Geom encoder

EDCL 使用 Equiformer V2。

当前项目如果不考虑实现成本，贴近 EDCL 的优先级：

```text
Equiformer / TorchMD-Net / PaiNN > SchNet
```

原因：

```text
1. PaiNN 能处理 equivariant vector features。
2. SchNet 主要依赖距离，表达方向性和力场恢复不如 PaiNN / Equiformer。
3. EDCL 的 denoising head 需要预测向量 shift，等变模型更合适。
```

## 14. 当前项目应避免的误用

### 14.1 不要把 EDCL 的 energy gap 直接替换成下游 property

如果用 `val` 训练预训练 head，再在同一任务评估，会造成监督泄漏。

### 14.2 不要把随机大噪声当作物理扰动

随机噪声太大时，构象可能不再代表合理分子。

### 14.3 不要让 pairwise distance loss 主导训练

全局 pairwise distance 太强时，模型可能学习构象重建，而不是属性相关几何表示。

### 14.4 不要对 `*` 做 star-linking 后再生成 3D 闭环构象

当前项目中：

```text
Graph 使用 star-linking。
Geom 使用 star-substitution capped repeat unit。
```

这个区分仍然合理。

## 15. 总结

EDCL 对当前项目最重要的帮助是：

```text
它证明了 3D 预训练不能只依赖普通随机 Gaussian noise。
更合理的做法是让噪声感知化学环境，并用去噪、对比和能量信号共同约束几何表示。
```

对 Uni-Poly-Plus 的直接结论：

```text
1. 当前 geom denoising 是合理起点，但偏 baseline。
2. 下一步优先加入 clean/noisy contrastive。
3. 再加入 adaptive atom-wise noise。
4. energy supervision 必须避免 downstream label leakage。
5. PaiNN 比 SchNet 更适合做 denoising vector prediction。
```
