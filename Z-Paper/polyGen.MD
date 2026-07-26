# polyGen 论文与源码分析：周期单链的原子级生成模型

分析对象：

- 论文：[A Learning Framework for Atomic-Level Polymer Structure Generation](https://doi.org/10.1021/acs.chemmater.5c01644)
- 期刊：Chemistry of Materials, 2025, 37(18), 7337-7346
- 代码：[Ramprasad-Group/polyGen](https://github.com/Ramprasad-Group/polyGen)
- 本次源码审计提交：`07a88a8de1968a30bfdaa28962dfb6a4117f24fd`

本文同时基于论文和公开源码。以下内容明确区分：

```text
论文提出的方法
公开源码实际配置
对当前 Uni-Poly-Plus / Polymer-SCAGE 的可迁移部分
```

## 1. 研究目标

polyGen 解决的问题不是聚合物性质预测，而是：

```text
输入重复单元的原子连接关系
-> 生成原子级 3D 无限单链周期结构
```

作者认为聚合物 3D 结构生成的困难主要来自：

```text
1. 聚合物构象具有多解性，不能只输出唯一坐标。
2. 重复单元的化学连接必须在周期边界上保持连续。
3. 聚合物结构数据远少于小分子数据。
4. 大重复单元、芳香骨架和支链容易产生错误键与原子碰撞。
```

polyGen 因而采用生成模型，而不是从单个 RDKit 构象拟合一个确定的变换。

## 2. 论文中的“无限聚合物链”定义

polyGen 使用一维周期正交晶胞：

```text
x 方向盒长固定为 55 Å
y 方向盒长固定为 55 Å
z 方向盒长 bz 由模型预测
聚合物链跨 z 边界连续
```

原子位置表示为周期盒中的 fractional coordinates：

```text
fi = (fx, fy, fz)
pi = fi * (55 Å, 55 Å, bz)
```

这里的 `55 Å x 55 Å` 不是聚合物真实横向晶格常数，而是为了隔离单链，降低不同周期像之间的横向作用。

因此 polyGen 的周期表示本质上是：

```text
平移周期链，不是 screw/helical PBC。
```

它不显式预测当前项目中的旋转矩阵 `R`，只预测轴向周期长度 `bz` 和晶胞内坐标。

## 3. 数据集

核心聚合物数据集为：

```text
polyChainStructures
样本数：3,855
最大原子数：208，包含 H
结构：DFT 优化的无限周期单链
划分：3,084 / 386 / 385
```

小分子辅助数据为：

```text
QM9 training split
约 100,000 个 DFT 优化小分子
```

作者将聚合物样本每个 epoch 上采样约 30 倍，以缓解 QM9 与聚合物数据的数量失衡。

这里必须注意：

```text
QM9 主要提供局部键长和键角先验。
QM9 不提供聚合物轴向周期长度、长程扭转或链级构象先验。
```

实验也支持这一点：加入 QM9 后，局部 bond/angle 改进明显，但 dihedral 和 `bz` 改进较小。

## 4. 整体模型

polyGen 分为三个模块：

```text
repeat-unit graph
-> graph conditioning encoder
-> structural VAE
-> latent diffusion transformer / flow matching
-> periodic polymer structure
```

训练顺序为：

```text
Phase A：训练 structural VAE
Phase B：冻结/加载 VAE，在 latent space 训练 DiT
Phase C：采样并做 connectivity filtering
```

它不是端到端属性预测模型，也没有下游 regression head。

## 5. Graph Conditioning

输入图包含：

```text
node: atom type + graph positional encoding
edge: bond type
```

图位置编码包括：

```text
random-walk positional encoding: 16 dim
Laplacian positional encoding: 2 dim
```

图编码器采用 Interaction Network 风格的 edge/node 交替更新：

```text
edge_ij <- MLP(edge_ij, node_i, node_j) + residual
node_i  <- MLP(node_i, aggregate(edge_ji)) + residual
```

公开配置为：

```text
num_layers = 4
aggregation = sum
activation = GELU
dropout = 0
```

4 层的解释是覆盖到与原子最远 dihedral 相关的局部化学环境。

图级 sum pooling 再广播回每个原子，使 atom conditioning 同时包含：

```text
局部原子环境 + 全局重复单元信息
```

## 6. 周期 Graph 的关键意义

论文明确区分：

```text
polymer graph: periodic connectivity
QM9 graph: nonperiodic connectivity
```

这意味着跨晶胞的聚合键应作为真实 graph edge 存在，而不是仅由坐标距离推断。

对当前项目的直接启示是：

```text
star-linking quotient graph 是合理的拓扑条件输入，
但它不能单独决定真实 3D 构象。
```

polyGen 通过周期结构数据学习“同一拓扑对应的构象分布”，而当前项目主要通过三聚体拟合 `(R,T)`。两者解决的是不同层次的问题。

## 7. Structural VAE

VAE 将化学和结构映射到 atom-wise latent space。

聚合物输入：

```text
fractional coordinates fi
scaled Cartesian positions pi
graph condition Ci
```

小分子输入：

```text
Cartesian positions pi
fractional coordinates 为空
graph condition Ci
```

解码器预测：

```text
predicted fractional coordinates
predicted Cartesian positions
predicted z box height bz
```

公开配置：

```text
VAE encoder: 8 Transformer layers
VAE decoder: 8 Transformer layers
d_model: 512
heads: 8
FFN dim: 1024
latent dim: 8 per atom
```

## 8. VAE 损失

VAE 总损失包含：

```text
L_VAE =
  w_bbox * L_bbox
  + w_frac * L_fractional_coordinate
  + w_pos * L_cartesian_position
  + w_KL * L_KL
  + w_bond * L_bond
  + w_angle * L_angle
  + w_dihedral * L_dihedral
```

公开配置中聚合物权重为：

```text
bbox       1
fractional 10
bond       2
angle      2
dihedral   1
KL         1e-5
Cartesian  0
```

QM9 权重为：

```text
Cartesian  1
KL         1e-5
其他周期相关项为 0
```

结构损失的作用不是要求每个原子坐标逐点完全重合，而是强化：

```text
局部键长
三原子键角
四原子二面角
```

这是 polyGen 对当前项目最直接的帮助之一：3D 监督不能只依赖一个周期距离 bias，还需要显式结构量。

## 9. Latent Diffusion Transformer

VAE 学到 latent space 后，DiT 在 latent 中生成结构。

论文采用 Gaussian flow matching：

```text
z0 ~ N(0, I)
zt = (1 - t) z0 + t z1
target velocity = z1 - z0
```

DiT 学习速度场：

```text
v_theta(zt, t, graph_condition, dataset_condition, self_condition)
```

推理使用 Euler integration 从噪声推进到结构 latent。

公开主配置：

```text
d_model = 512
num_layers = 12
heads = 8
latent input dim = 8
sampling steps = 100
self-conditioning probability = 0.5
classifier-free guidance scale = 2.0
```

## 10. Relative Graph-Distance Attention Bias

polyGen 没有把 3D 距离直接加到 DiT attention，而是加入 graph-distance category bias。

原子对被分成：

```text
0-hop: same atom
1-hop: bonded
2-hop: angle relation
3-hop: dihedral relation
>=4-hop: longer-range
```

类别经线性层映射到每个 attention head 的 bias：

```text
score_ij^h = Qi^h Kj^h / sqrt(dh) + bias_ij^h
```

这与当前 SCAGE 的 Graphormer/MIPS topology bias 高度相关。

重要区别是：

```text
polyGen 的 bias 约束生成模型的 latent denoising。
当前 SCAGE 的 bias 约束属性编码器的 atom attention。
```

## 11. 数据增强与等变性

polyGen 主模型不是 SE(3)-equivariant 网络。

作者通过训练时随机旋转和平移来近似学习旋转/平移不变性。

这是一种数据驱动折中：

```text
优点：模型简单，适合统一 molecule/polymer latent space。
缺点：对仅 3,855 个聚合物样本，严格几何泛化能力受限。
```

当前项目的 SCAGE 主要读取距离 bias，距离天然具有旋转平移不变性，因此不需要照搬 atom-coordinate augmentation 才能获得这一性质。

## 12. 后处理过滤

采样后的结构并非全部接受。

过滤条件至少包括：

```text
1. 从预测坐标重建出的 graph 与目标 graph isomorphic。
2. 不允许任意 bond length < 0.8 Å。
```

因此论文中的 generation success 是：

```text
生成后通过 connectivity/collision filter 的比例
```

它不是：

```text
DFT 收敛率
真实最低能构象命中率
周期链物理正确率
```

## 13. 实验结果

关键结果：

```text
joint QM9 + polymer 训练后的总过滤通过率：36.1%
仅 polymer 训练：27.4%
```

relative graph-distance bias 对 graph isomorphism 的影响：

```text
36.9% -> 64.8%
```

加入 QM9 后：

```text
bond KL divergence 改进 41.0%
angle KL divergence 改进 29.0%
dihedral KL divergence 改进 16.1%
```

轴向周期长度预测：

```text
predicted bz vs DFT bz: R^2 = 0.854
```

能量验证：

```text
polyGen 中 53.8% 样本在 DFT 参考的 0.1 eV/atom 内
PSP 中 74.7% 样本在 0.1 eV/atom 内
```

速度：

```text
polyGen: 约 0.308 s / successful sample，batch=200
PSP: 约 30.2 s / successful sample
```

训练成本：

```text
VAE: 300 epochs
DiT: 最多 500 epochs
2 x NVIDIA L40S
约 140 h
```

## 14. 论文结论的正确解读

polyGen 证明的是：

```text
给定重复单元连接关系，生成模型可以学习周期单链结构分布。
```

它没有证明：

```text
任意 P-SMILES 都能高成功率生成严格周期链。
生成结构就是室温体相聚合物的真实构象。
单个重复单元必然是最小周期单元。
```

尤其是模型整体过滤通过率 36.1%，不能作为当前项目追求 `>95% strict screw` 的依据。

## 15. 源码与论文需要注意的细节

公开源码中 relative bias 层实际读取的 distogram channel 数需要以代码配置为准；代码中的线性层与论文“五类关系”的文字描述存在版本痕迹。

此外，仓库 README 明确说明数据需向作者联系获取，复现并非只 clone 代码即可完成。

当前公开仓库还包含不同实验分支，例如 `polymd`、其他 decoder/encoder，但论文主结论不应与这些未作为主实验报告的配置混淆。

## 16. 与当前项目的对照

| 项目 | polyGen | 当前 Polymer-SCAGE |
|---|---|---|
| 目标 | 生成周期 3D 结构 | 预测聚合物性质 |
| 周期形式 | z 轴平移晶胞 | screw `(R,T)` 或有限三聚体 |
| 图拓扑 | periodic graph | star-linked quotient graph |
| 几何来源 | DFT 数据学习生成 | RDKit + MMFF/UFF + trimer fit |
| 多构象 | latent sampling | top conformers / fallback hierarchy |
| 几何监督 | bbox/coords/bond/angle/dihedral | distance/angle/torsion/screw targets |
| 注意力结构先验 | graph-distance bias | topology branch + explicit-image geometry branch |
| 失败处理 | post-generation reject | strict screw -> finite trimer -> Euclidean -> topology |

## 17. 对当前项目最有价值的帮助

### 17.1 周期单链应表示为“周期晶胞 + 分数坐标”

当前 screw `(R,T)` 是一般化的一维空间群表示，但成功率低。

polyGen 提供了更稳定的替代类别：

```text
translation-only periodic cell
```

对于不能稳定拟合单 RU screw 的样本，可以尝试：

```text
m-RU translational supercell, m in {2,3,4,6}
```

这仍然是严格无限周期结构，不是有限三聚体 fallback。

### 17.2 拓扑距离 bias 应保留

polyGen 的消融显示 graph-distance bias 对 connectivity success 有显著帮助。

当前 SCAGE 已有 topology branch，因此不应为了突出 3D 而删除：

```text
shortest-path distance
path bond information
star-linking topology
```

### 17.3 3D 任务应覆盖 bond/angle/dihedral

当前 M4P 已包含 distance、angle、torsion、screw operation，方向是合理的。

可进一步改进为：

```text
使用连续值或周期分布监督二面角，
而不是只做粗粒度 angle/torsion bins。
```

### 17.4 小分子辅助数据只用于局部结构

如果未来引入大量小分子 3D 数据，应限制其作用：

```text
可训练 bond/angle/local geometry head
不可替代 polymer periodic/screw/supercell task
```

### 17.5 多构象是分布，不是单一 ground truth

下游属性模型不应默认一个 RDKit 最低能构象是唯一真实结构。

推荐：

```text
训练：从 top-k periodic conformers 采样
推理：对 top-k representation 做均值或 attention ensemble
```

## 18. 不建议直接迁移的部分

### 18.1 不建议直接训练 polyGen

当前项目目标是属性预测，且没有 3,855 个 DFT 周期链坐标的同分布训练集。

直接加入 12 层 DiT + 8/8 层 VAE 会显著增加训练成本，并偏离当前主任务。

### 18.2 不建议固定所有样本为 1-RU translation cell

柔性链可能需要多 RU 才形成周期；某些链根本不应被压缩到单 RU 平移周期。

### 18.3 不建议把生成过滤通过率当作物理正确率

graph isomorphism 和最小键长只能排除明显错误，不能证明热力学稳定性或体相代表性。

### 18.4 不建议只学习 `bz`

当前项目已有一般 screw 结构；若退化为只用 `bz`，会丢失螺旋旋转信息。正确做法是并列支持 symmetry class，而不是用一个表示覆盖全部样本。

## 19. 推荐落地方案

### Phase A：增加周期类型分类

```text
periodic_type = screw_1ru | translation_mru | finite_context | topology_only
periodic_unit_size = 1 | 2 | 3 | 4 | 6
```

### Phase B：增加 m-RU translation cell

```text
PSP-style candidate generation
-> build m-RU connected oligomer
-> optimize axial period Lz and internal torsions
-> exact periodic closure quality gate
```

### Phase C：SCAGE 统一周期像接口

```text
screw image: r_n = R^n r + sum R^k T
translation image: r_n = r + nL
```

二者统一生成 explicit-image distance bias，但保留 periodic type embedding。

### Phase D：构象系综

```text
每个 polymer 保存 3-4 个通过质量门的 periodic conformers
Stage 1 随机采样
Stage 3 ensemble
```

## 20. 对当前 PBC 创新点的评价

polyGen 支持当前项目继续研究严格周期几何，但也说明创新点应调整为：

```text
adaptive periodic symmetry representation for polymer property prediction
```

而不是：

```text
强制所有聚合物使用 one-RU screw PBC
```

更合理的验收指标是：

```text
strict periodic coverage >= 95%
= one-RU screw + valid m-RU translation/screw supercell

nonperiodic fallback <= 5%
```

## 21. 总结

polyGen 最重要的贡献是把聚合物 3D 生成定义为：

```text
periodic graph-conditioned atomic structure distribution generation
```

它对当前项目的直接价值不是替换 SCAGE，而是提供四条设计依据：

```text
1. 周期拓扑与周期坐标必须同时存在。
2. 局部 graph-distance bias 对维持化学连接非常关键。
3. bond/angle/dihedral 显式几何约束不能省略。
4. 多构象和多周期类型比单一 `(R,T)` 更符合聚合物结构多样性。
```

## 22. 参考链接

- [Chemistry of Materials 论文页面](https://pubs.acs.org/doi/10.1021/acs.chemmater.5c01644)
- [polyGen 官方代码](https://github.com/Ramprasad-Group/polyGen)
- [polyChainStructures 数据入口 Khazana](https://khazana.gatech.edu/dataset)
- [PSP 论文](https://doi.org/10.1021/acs.jctc.2c00022)
