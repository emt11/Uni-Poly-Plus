结论：**有较高参考价值，但应借鉴 MolGT 的预训练组织方式，不能直接复制它的小分子 3D 表示或官方代码。**

对当前项目而言，我给出：

- 方法思想参考价值：`8/10`
- 聚合物 3D 直接适配性：`5/10`
- 官方代码直接复用价值：`3/10`

MolGT 最有价值的地方，是同时使用 node-level、graph-level、2D、3D 四种监督，让拓扑主干真正吸收几何知识，而不是单纯添加一个距离 RBF。论文的核心定位与官方摘要一致：共享 Transformer 处理 2D/3D 模态，并通过对比学习实现无 3D 构象的下游推理。[论文页面](https://www.sciencedirect.com/science/article/abs/pii/S1566253524005621)

## 与当前项目的契合度

| MolGT 机制 | 对当前项目的价值 | 建议 |
|---|---:|---|
| 3D coordinate denoising | 很高 | 优先迁移为 Trimer 周期几何去噪 |
| 2D–3D shared backbone | 很高 | 可构造 topology-only 与 Trimer-3D 两个共享权重视图 |
| InfoMotif | 高 | 可缓解 masked-atom-only 很快饱和 |
| KGPC prototype | 中等 | 可改善通用表示，但不能直接解决 3D 无效 |
| USRCAT 3D prototype | 低 | 不适合直接描述聚合物链 |
| 2D-only inference | 中等 | 可作为几何无效样本的鲁棒性方案，但不是当前首要目标 |

## 最值得借鉴：3D 去噪

历史 R2 预训练是 `masked_atom_only`，angle loss 为 0；其配置已在 R6 退役清理。原子分类约在 3,000 step 就达到 `99.3%`，说明任务很快饱和。

MolGT 的关键启示是：

```text
不能只把 3D 距离送进模型；
还要提供一个只有理解几何才能完成的预训练任务。
```

这正是当前 Star-RBF v2 缺少的部分。现在的 3D 距离只是 Attention bias，主要依靠 masked-atom loss 间接获得梯度。模型完全可能主要依靠原子类型、SPD 和路径特征完成任务，从而忽略 3D。

适合当前项目的迁移版本应当是：

```text
干净 open Trimer 坐标
→ 对部分原子加入噪声
→ 根据带噪坐标动态重算 periodic relation 距离
→ 共享 O8/MSTA 主干
→ 预测中央 RU 原子的坐标噪声
```

建议：

- 外侧 RU 作为几何上下文；
- 主要监督中央 RU 原子，避免三份相同化学原子重复计权；
- 噪声覆盖 RU 内和跨 RU 几何；
- 只使用真正有效的 3D 构象；
- geometry-invalid 样本保留 masked-atom loss，几何损失精确为零；
- 对整个 Trimer 随机旋转和平移后，预测应同步旋转但损失不变。

## 文档中有一个关键点需要修正

[MolGT.MD](/root/workspace/Uni-Poly-Plus-master/Z-Paper/MolGT.MD:166) 将其概括为“backbone 不是严格等变，通过 distance bias 和 denoising head 学习”，方向没错，但不够完整。

MolGT 官方实现的去噪头并不是普通的 `Linear(hidden, 3)`。它显式计算归一化相对方向：

```text
delta_pos = (r_i - r_j) / ||r_i - r_j||
```

然后将 Attention 权重与 `delta_pos` 相乘，聚合成三维向量，再预测坐标噪声。这使去噪头具备方向感知和等变结构。[MolGT 官方建模代码](https://raw.githubusercontent.com/robbenplus/MolGT/master/src/models/modeling.py)

而当前项目已有的通用 `_geom_denoise_loss` 使用的是：

```python
geom_noise_head = nn.Linear(geom_dim, 3)
pred_noise = geom_noise_head(node_rep)
```

见 [pretrain.py](/root/workspace/Uni-Poly-Plus-master/scripts/pretrain.py:4020) 和 [head 初始化](/root/workspace/Uni-Poly-Plus-master/scripts/pretrain.py:5064)。

这个普通线性头不能直接用于当前 Trimer 去噪：

1. MTS 已经拒绝旧的并行 coordinate encoder 路线，见 [dataset.py](/root/workspace/Uni-Poly-Plus-master/src/dataset/dataset.py:1077)。
2. 标量节点 embedding 直接回归 XYZ，不保证旋转等变。
3. 它没有利用 periodic relation、RU shift 和 Trimer 中的方向向量。

因此可以复用训练循环框架，但不能直接复用这个 head。

## 第二个高价值方向：2D–3D 共享视图

当前 MTS 是一次 forward 中同时加入拓扑和几何 bias。MolGT 则使用同一 Transformer 权重分别执行：

```text
2D view：Topology、SPD、path，没有 3D bias
3D view：Trimer geometry bias
```

然后对齐同一个聚合物的两个 graph embedding。

这个设计适合当前项目，因为可以定义：

```text
z_topology：
  canonical lifted graph
  Star-RBF/G1 geometry 全部关闭

z_geometry：
  同一个 canonical graph
  Star-RBF v2 开启
  使用带噪 Trimer geometry

loss：
  InfoNCE(z_topology, z_geometry)
```

优点是即使部分下游样本 3D 无效，topology-only 表示也能吸收预训练阶段的几何知识。

但成本明显：需要两次主干 forward。当前训练约 79% 时间都在 forward，因此吞吐可能接近下降一半。更现实的实现是：

- 只在部分 batch 执行双视图；
- 或者交替执行普通 batch 与 2D–3D alignment batch；
- 先用短 benchmark 确认成本；
- 不要一开始就同时加入 MolGT 全部四种 loss。

## InfoMotif 也非常适合当前项目

MolGT 指出 atom masking 对 motif 级化学结构监督不足，这与当前 masked-atom 快速饱和完全吻合。

但不能直接照搬“小分子 ring + non-ring bond”。聚合物版本应定义：

- attachment/star 邻域；
- 跨 RU backbone connection motif；
- backbone 连续片段；
- side-chain attachment motif；
- 芳香环；
- ester、amide、imide、ether、sulfone 等官能团；
- donor/acceptor 和极性基团。

需要特别注意：motif 定义不能重新引入已删除的“最短路径经过环后扩张整个环”Backbone 语义。Backbone annotation 和 motif decomposition 应是两个独立概念。

## 我不同意当前文档优先 KGPC 的结论

[MolGT.MD](/root/workspace/Uni-Poly-Plus-master/Z-Paper/MolGT.MD:538) 建议若只选一个模块，优先 KGPC。这个建议适用于“低风险提升通用分子表示”，但不适合你当前的核心问题——**为什么 Trimer 3D 没有被有效利用**。

KGPC 使用 fingerprint 聚类作为图级伪标签。它可能改善小数据任务，但模型完全可以在不理解 3D 的情况下预测 2D fingerprint prototype。

对当前项目更合理的顺序是：

1. Trimer 几何去噪；
2. topology-only / Trimer-3D shared-backbone alignment；
3. polymer-specific InfoMotif；
4. 最后再考虑 fingerprint prototype。

如果实现 prototype，也不应直接用 USRCAT。更合适的聚合物先验包括：

- MD200或独立 repeat-unit fingerprint；
- backbone/side-chain descriptor；
- 官能团计数；
- 按 `|shift|=0/1/2` 划分的距离分布；
- backbone end-to-end distance；
- 跨 RU 键角和二面角；
- 多构象距离方差。

## 不应直接复制 MolGT 官方代码

官方仓库可用于核对公式与实现，但工程成熟度较低：

- 仓库目前只有很少的提交历史；
- 使用旧式 `torch.distributed.launch`；
- README 示例使用 `strict=False` 加载；
- 代码中存在宽泛异常捕获。

这些都不符合当前项目的严格缓存、checkpoint 和身份合同。[MolGT 官方仓库](https://github.com/robbenplus/MolGT)

因此应借鉴公式和实验拆分，不建议把官方模块整体复制进当前代码。

## 历史研究草案（不执行）

下面的 M0–M5 仅保留为历史研究讨论。R2 已由 R6 退役，不是当前 baseline、配置或
可运行入口；不得继续 R2、恢复 R2、执行 milestone 或按此顺序启动新实验。未来新
预训练路线必须等待新的活动配置并重新审查。

不建议一次加入 MolGT 全套任务。最干净的因果拆分是：

```text
M0：历史 R2（已退役，不得执行）
    masked atom only
    Star-RBF v2

M1：R2 + periodic distance/RBF denoising
    不增加方向头

M2：R2 + Trimer coordinate denoising
    使用 relative-direction equivariant head

M3：M2 + topology/geometry graph-level alignment

M4：M3 + polymer InfoMotif

可选 M5：M4 + polymer prototype
```

其中：

- M1 是低风险验证：检查直接几何监督是否有效。
- M2 才是对 MolGT coordinate denoising 的较忠实迁移。
- M3 验证 implicit 3D 是否能帮助 geometry-disabled 推理。
- M4 解决 masked atom 任务过于简单的问题。
- M5 是通用语义增强，不应与 3D 效果混在一起解释。

最终判断：**MolGT 很值得参考，而且比继续单独修改 RBF 或 Attention scale更有价值；但应优先迁移“几何去噪 + 共享2D/3D视图”，而不是优先复制 KGPC。**目前没有修改任何文件或执行实验。
