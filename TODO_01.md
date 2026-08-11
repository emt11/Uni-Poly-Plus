结论是：下一步不应该继续在 G0 上堆更多 RBF，而应先把“Trimer 几何”和“MCL 额外容量”拆开。G2 目前说明的更可能是：

> MCL 作为额外 token mixer/正则化器有价值，但当前真实坐标的引入方式、Star-distance 或 Trimer token 语义存在问题。

## 一、G2 目前不是严格单变量对照

G2 同时做了两件事：

1. 将 Trimer 原子与坐标的对应关系反转；
2. 关闭 Star-distance bias。

坐标置乱见 [trimer_mcl.py](/root/workspace/DeepLearning/Uni-Poly-Plus-master/src/modules/trimer_mcl.py:260)，Star bias关闭见 [mips_local_graph.py](/root/workspace/DeepLearning/Uni-Poly-Plus-master/src/modules/mips_local_graph.py:386)。

因此：

```text
G0 = 正确坐标 + Star bias
G2 = 置乱坐标 + 无Star bias
```

G2 高于 G0 可能来自：

- 置乱坐标产生了类似随机图增强的正则化；
- 关闭 Star bias 带来了收益；
- 正确坐标确实包含噪声；
- 上述因素共同作用。

下一步首先应补齐严格对照：

| 对照 | MCL坐标 | Star bias | 用途 |
|---|---|---|---|
| C0 | 正确 | 开启 | 当前G0 |
| C1 | 正确 | 关闭 | 单独判断Star bias |
| C2 | 随机置乱 | 关闭 | 纯坐标负对照 |
| C3 | 随机置乱 | 开启 | 与C0只差坐标对应 |
| C4 | 不读取坐标 | 关闭 | MCL容量控制 |
| C5 | 不读取坐标 | 开启 | 判断Star本身 |

坐标置乱应改成由 `sample hash + shuffle seed` 生成的真正随机 permutation，不能继续使用确定性反转。至少使用3个shuffle seed，避免反转顺序偶然与canonical atom order相关。

## 二、Star-distance 很可能是主要噪声源之一

当前 Star-distance 是两条真实 inter-RU 共价键长度的平均：

$$
d_{\rm star}=(d_{\rm left}+d_{\rm right})/2
$$

但普通共价键长的变化范围很小，而且键类型已经被拓扑输入间接表达。把这个标量加到每一层 Star attention edge，可能产生冗余或过强扰动。

预训练 checkpoint 中：

```text
Star RBF projection L2 norm ≈ 7.53
mean absolute weight ≈ 0.275
```

它已经不是一个接近零的微小修正。

建议：

- 在完成 C1/C3/C5 前，默认关闭 Star-distance bias；
- 如果需要保留，改为一个 graph-level、小容量、sample-conditioned gate；
- 不再让同一个 `d_star` 直接进入六层 O8 attention；
- 更有价值的跨 RU 信息应是二面角、相邻片段取向和非键空间接触，而不是几乎固定的共价键长。

## 三、当前 Trimer token 的表达存在根本信息缺失

当前数据映射为：

```text
O8 node = (canonical atom a, MIPS copy c)
Trimer  = (canonical atom a, RU offset -1/0/+1)
```

构图时所有 O8 copies 都映射到中央 RU：

$$
(a,c)\rightarrow(a,0)
$$

见 [trimer_mcl.py](/root/workspace/DeepLearning/Uni-Poly-Plus-master/src/dataset/trimer_mcl.py:485)。

索引映射本身是正确的，缓存验证也没有发现 mapping failure。但模型随后做了：

```python
canonical = mean(O8 copies)
tokens = canonical[trimer_base_ru_atom_index]
```

见 [trimer_mcl.py](/root/workspace/DeepLearning/Uni-Poly-Plus-master/src/modules/trimer_mcl.py:348)。

这意味着：

```text
token(a,-1) = token(a,0) = token(a,+1)
```

三个 RU 中相同 canonical atom 的 value 完全一样。

因此，即使注意力根据距离给三个copy不同权重：

$$
w_{-1}V_a+w_0V_a+w_{+1}V_a
=(w_{-1}+w_0+w_{+1})V_a
=V_a
$$

也就是说，距离无法有效区分同一原子位于中央还是相邻 RU。它只能改变“哪些不同canonical atom可见”，几何表达能力非常受限。

### 正确调整

不要将 `mips_copy_id` 强行映射为 Trimer 的 `-1/0/+1`，因为两者没有唯一物理对应。继续在 canonical 层映射，但给 Trimer token增加反转不变的角色：

```text
central RU embedding
adjacent RU embedding
```

左右相邻 RU共享同一个 embedding，不引入方向。

同时加入关系类型：

```text
same-RU
cross-RU real bond
cross-RU nonbonded contact
```

新的 token 为：

$$
h_{a,q}=h_a^{\rm canonical}
+e_{\rm role}(|q|)
$$

这样 MCL 才能区分中央和相邻空间环境。

## 四、MIPS copies 的聚合需要保留更多信息

当前对同一 canonical atom 的全部 O8 copies直接做平均。这个操作安全，但可能过度丢失信息。

建议先统计：

$$
\operatorname{Var}_c(x_{a,c})
$$

如果相同 canonical atom 的不同copy表示差异很大，说明：

- Star-linking周期切口仍影响表示；
- 外侧copy和中央copy上下文不等价；
- 直接平均会掩盖构图问题。

新的聚合建议使用：

```text
mean(copy states)
std(copy states)
max(copy states)
→ Linear(1536→512)
```

或者更低容量：

```text
mean + gated std
```

但几何 residual仍应按 canonical atom广播给全部copy，以保持RU扩展不变性。

最终 graph readout也应先回到 canonical atoms，再做 pooling：

```text
O8 copies
→ canonical aggregation
→ canonical atom pooling
```

不再直接对全部扩展copy执行全局mean。

## 五、当前坐标进入MCL的方式过于粗糙

G0 中坐标不直接作为连续特征，只用于生成两个 hard mask：

```text
Trimer全部原子距离的20%分位数
Trimer全部原子距离的50%分位数
```

问题在于阈值是整个分子的全局分位数：

- 大RU和小RU对应的Å尺度不同；
- 紧凑构象和伸展构象阈值不同；
- 50% mask在大分子上可能接近全局attention；
- 共价近邻和真正的非键空间接触混在一起。

G2 保留了相同距离分布，只改变了距离对应的原子身份，因此它可能退化成一种“结构相关随机邻居采样器”。

### 推荐的空间邻接

将拓扑关系和空间关系分开：

```text
拓扑通道：
真实键、1-hop、2-hop由O8处理

几何通道：
只处理非键、跨RU空间接触
```

中央 RU query只关注：

- 相邻 RU 中的真实连接附近原子；
- 跨 RU 非键接触；
- 中央 RU 内拓扑距离大于2但空间接近的原子。

空间mask建议使用：

```text
每query top-k：8和16
```

或者：

```text
固定cutoff：4.5 Å和6.0 Å
```

优先推荐 per-query top-k，因为不同RU大小下邻居数量更稳定。

以下关系应从几何邻居中排除或单独编码：

```text
self
直接共价键
1–3拓扑邻居
```

避免MCL重复学习O8已经处理的局部拓扑。

## 六、MCL 的引入位置需要调整

当前流程是：

```text
六层O8全部完成
→ canonical scatter-mean
→ 两层MCL
→ geometry residual广播
→ mean pooling
```

见 [mips_local_graph.py](/root/workspace/DeepLearning/Uni-Poly-Plus-master/src/modules/mips_local_graph.py:403)。

问题是：

- MCL只能看到最后一层高度抽象、可能过平滑的节点；
- 几何更新发生得太晚，无法再经过拓扑传播；
- 两层MCL连续更新，但外侧RU memory始终主要来自同一份canonical topology token；
- 噪声geometry residual直接广播到全部O8 copies。

### 推荐位置：并行 canonical geometry branch

当前最稳妥的结构是：

```text
O8六层保持完全不变
├→ layer2/layer4/layer6 canonical states
│  → projection
│  → role-aware Trimer-MCL
│  → geometry graph token
└→ canonical hierarchical pooling
   → topology graph token

topology token
+ sample-confidence-gated geometry token
→ MD200
→ property
```

优点：

- O8的MIPS拓扑主干和证明不受影响；
- 3D噪声不再污染每个O8 node；
- geometry-invalid时精确退回topology token；
- 可以明确检查geometry gate是否真正打开；
- 可以同时利用浅层局部化学和深层语义。

不建议第一步就把MCL插进每一层O8。只有独立geometry token证明有效后，再考虑：

```text
O8 layer 1–3
→ 一层MCL residual
→ O8 layer 4–6
```

## 七、Trimer 构象质量应采用软置信度，而不是只看valid

当前协议是：

```text
ETKDGv3 4候选
→ MMFF94固定200步
→ 选择最低有限能量
→ 不要求完全收敛
```

见 [trimer_mcl.py](/root/workspace/DeepLearning/Uni-Poly-Plus-master/src/dataset/trimer_mcl.py:425)。

该策略适合百万规模缓存，但“坐标finite”不等于“几何对属性预测可靠”。

建议保留现有缓存，增加只读质量sidecar：

```text
bond-length异常率
非键碰撞数量/原子
MMFF energy/atom
star asymmetry
Rg、asphericity
end-to-end/contour ratio
局部接触密度
```

由这些量生成：

$$
q_{\rm geo}\in[0,1]
$$

最终：

$$
g=g_{\rm topology}
+q_{\rm geo}\tanh(\gamma)g_{\rm geometry}
$$

不要因低质量直接删除样本。

对于下游约数千个unique polymers，可进一步生成2–4个构象，用均值或confidence gate聚合；不需要重建百万级PI1M缓存。

## 八、当前预训练没有保证模型真正使用坐标

现有 angle任务可能通过原子类型、键型和拓扑环境预测出大部分角度先验，而不一定真正利用坐标。

建议增加一个直接任务：

```text
正确 canonical atom-coordinate mapping：正样本
同一Trimer内随机置换mapping：负样本
```

使用 binary matching 或 InfoNCE，权重建议 `0.05–0.10`。

另一个可用任务是跨 RU 非键 contact prediction：

- 隐藏目标pair对应的几何关系；
- 根据其余拓扑和空间上下文预测是否小于固定cutoff；
- 避免把答案作为同一次attention bias直接输入。

如果模型无法区分正确映射和随机映射，就不应该继续进行angle或torsion重预训练。

## 九、建议的优化顺序

### 第一步：修复对照

```text
正确/置乱坐标 × Star on/off
+ 真正随机permutation
+ 3个shuffle seeds
```

先确定主要噪声来自 Star bias还是MCL坐标。

### 第二步：修复表示语义

```text
central/adjacent role embedding
same/cross-RU relation
canonical mean/std聚合
canonical atom readout
```

不重新生成坐标。

### 第三步：重构几何邻域

```text
只保留跨RU非键空间接触
全局20/50%分位 → per-query top-k 8/16
sample geometry-confidence gate
```

### 第四步：移动MCL

将MCL改为并行canonical geometry branch，在graph readout处零门控融合。暂不污染O8 node stream。

### 第五步：重新验证3D

只有真实坐标稳定优于：

```text
MCL disabled
coordinate shuffled
topology-only stochastic mixer
```

才执行coordinate-mapping预训练、多构象或torsion任务。

## 推荐的下一版结构

```text
O8 MIPS 2-hop六层
→ layer2/4/6 canonical aggregation(mean+std)
→ canonical hierarchical topology pooling
                                  ┐
role-aware Trimer spatial branch  ├→ confidence-gated graph fusion
- central/adjacent role           │
- cross-RU nonbonded top-k        │
- one-layer MCL                   │
- no Star bond-length bias        ┘
→ MD200
→ residual regression head
```

这条路线保留MIPS的可靠二维主干，同时让Trimer成为可关闭、可解释、质量可控的几何增量。若严格对照后真实坐标仍不优于随机邻域，应将MCL重新定义为纯拓扑正则化器，并停止宣称当前Trimer提供了有效3D信息。