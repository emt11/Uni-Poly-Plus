结论：当前项目最大的优化空间已经不在 DataLoader，而在“预训练目标是否真正迫使模型学习 3D”以及“3D 信息如何进入消息更新”。继续增加 workers、单独修改 Attention scale，优先级都较低。

## 当前审查结论

| 优先级 | 优化方向 | 当前问题 | 建议 |
|---|---|---|---|
| P0 | 实验状态与源码身份 | R2 未完成正式 20k/8×5，工作树存在大量未提交修改 | 先明确 R2 是继续还是归档，并冻结源码身份 |
| P1 | 预训练目标 | `masked_atom_only` 很快饱和，3D 只得到间接梯度 | 增加与 Trimer 几何直接相关的去噪/重建目标 |
| P1 | 3D 注入方式 | 主要作为标量 Attention bias，没有方向和几何消息 | 增加 shift-conditioned 几何消息或等变向量头 |
| P2 | 前向性能 | 约 79% 时间在 forward，存在 CUDA 同步和稀疏算子开销 | profiler 后消除热路径同步、优化 MSTA/Star-RBF |
| P2 | 构象表达 | 每个聚合物只保留一个最低能 Trimer 构象 | 后续加入小规模构象系综和可信度门控 |
| P3 | 下游质量 | XC、EI、EEA 仍是主要短板 | 多任务学习和图/SMILES融合比继续微调 workers 更有价值 |
| P3 | 工程维护 | 超大训练脚本、sidecar 启动审计重复、文档身份较复杂 | 当前实验结束后再模块化整理 |

## 1. 预训练目标是目前最大的瓶颈

R2 的实际配置是：

- `masked_atom_only`
- `angle_loss_weight=0`
- Graph-only
- Star-RBF v2 + G1 path-cosine
- 不使用 full-Trimer MCL

证据见 [R2 配置](/root/workspace/Uni-Poly-Plus-master/configs/mts/experiments/R2_g1_periodic_relation_rbf_v2_legacy_backbone_formal_v1.json:44)。

这个任务很快饱和：

- step 50：原子准确率约 `78.1%`
- step 2,000：约 `97.9%`
- step 3,000：约 `99.3%`
- step 6,000：约 `99.5%`

见 [R2 预训练日志](/root/workspace/Uni-Poly-Plus-master/logs/mts_star_rbf_v2/legacy_backbone_formal_v1/R2/formal20k_v3/mts_joint_pretraining_pi1m_v2.log:107)。

这说明后续大量训练主要是在继续压低一个已经接近解决的分类损失。模型并没有被直接要求理解：

- 原子坐标受到扰动后如何恢复；
- 跨 RU 距离是否正确；
- inter-RU 键角和二面角；
- 构象改变时哪些关系保持稳定。

最推荐的新预训练目标是先做“不改变主干架构的几何去噪”：

1. 给合法 Trimer 3D 坐标添加噪声。
2. 用带噪坐标重新计算 Star-RBF v2 relation distances。
3. 让模型预测干净距离、距离残差或干净 RBF 分布。
4. 只监督合法、非 self、`SPD≤2` relations。
5. 分别记录 `shift=0/1/2` 损失，防止模型只学习大量同 RU relation。
6. 增加跨 RU 键角、二面角重建作为次级目标。

这比继续增加 masked-atom 训练步数更可能真正改善 3D 表示。

## 2. 不建议直接用普通 MLP 预测 XYZ

当前模型主体是旋转不变的标量 Transformer。仅根据标量 embedding 直接回归 XYZ 噪声，会破坏旋转等变性：同一个 Trimer 旋转后，预测向量未必跟着旋转。

如果要做真正的坐标去噪，应增加等变向量头，例如：

$$
\hat{\epsilon}_i=
\sum_j
\alpha_{ij}
\frac{x_i-x_j}{\|x_i-x_j\|}
$$

其中 \(\alpha_{ij}\) 由节点表示、RBF、SPD 和 \(|shift|\) 产生。这样旋转坐标时，预测向量也同步旋转。

建议分两步：

- 第一阶段：距离/RBF、角度、二面角等不变量去噪，改动较小。
- 第二阶段：若第一阶段有效，再加入等变向量头做真正坐标去噪。

## 3. 当前 3D 信息仍然利用不足

Star-RBF v2 已经比旧版本合理：它在 unique periodic pairs 上编码真实 relation distance，并保证互逆 relation 共享结果，见 [实现](/root/workspace/Uni-Poly-Plus-master/src/modules/mips_local_graph.py:168)。

但仍有三个限制。

第一，距离主要作为 Attention logit bias，而不是几何消息。G2 的简单 endpoint-distance RBF 已经得到负结果：

- G2−G1 宏平均 `-0.002992`
- 只有 `3/8` 个任务为正

见 [G1/G2 正式报告](/root/workspace/Uni-Poly-Plus-master/results/mts_multiscale_topology/g_family_matched_v1/G1_vs_G2/final_report.md:21)。

这不能证明“3D 距离无效”，更准确的结论是：仅把单一距离加进 Attention bias 不足以形成有效几何表征。

第二，Star-RBF v2 对 `shift=0/1/2` 使用同一个 `32→8` 投影。代码只利用 `geometry_source` 排除 true-self，并没有让不同 shift class 使用不同的几何语义。后续可以采用：

- 共享 RBF centers；
- 按 `|shift|=0/1/2` 使用独立的零初始化 projection 或 FiLM gate；
- 仍然只使用绝对 shift，保证 inversion symmetry；
- 把距离同时用于 Attention bias 和 value/message gate。

第三，当前 R2 没有使用 full-Trimer MCL，而旧 MCL 默认也只是用距离决定 hard visibility mask，没有连续方向或坐标消息，见 [Trimer MCL](/root/workspace/Uni-Poly-Plus-master/src/modules/trimer_mcl.py:200)。因此“拥有 Trimer 坐标”并不等于模型已经充分学习了 Trimer 几何。

## 4. 构象系综比继续扩展单距离更重要

当前流程从最多四个候选中选择最低 finite MMFF 能量的单个构象。项目文档本身也明确：finite Trimer 只是局部代理，单构象不能代表完整构象系综，见 [PIPELINE.md](/root/workspace/Uni-Poly-Plus-master/PIPELINE.md:709)。

后续适合加入：

- 每个样本保留 `K=2–4` 个有效构象；
- 预训练时随机抽取一个；
- 微调或推理时对构象 embedding 做均值/注意力聚合；
- 使用能量差、构象间距离方差作为 confidence，而不是直接作为属性特征；
- 只池化中央 RU，外侧 RU 继续作为局部链环境。

这通常比把单构象 RBF 做得越来越复杂更稳健。

## 5. 训练速度已经不是 workers 问题

最新同配置测试结果：

- workers 6：`4849.81 samples/s`
- workers 8：`4794.61`
- workers 12：`4846.60`
- workers 16：`4838.82`

workers 6 仍然最快。其单步阶段时间中：

- forward：约 `164.9 ms`
- backward：约 `36.9 ms`
- H2D：约 `2.9 ms`
- optimizer：约 `2.7 ms`
- data wait：约 `0.37 ms`
- rank wait：约 `0.011%`

见 [workers=6 benchmark](/root/workspace/Uni-Poly-Plus-master/logs/mts_speed_optimization/r2_worker_sweep_20260813/w6/launcher.log:21)。

因此：

- 保持 `workers=6, prefetch=2`；
- 不再测试更高 workers；
- 不要增加 activation checkpointing，显存还有大量余量，而且会变慢；
- 应用 `torch.profiler` 定位 `scatter/softmax/RBF/repeat_interleave` 的真实占比。

代码层可优先检查两个同步热点：

- joint-pretraining 每 batch 执行 `loss.item()`：[pretrain.py](/root/workspace/Uni-Poly-Plus-master/scripts/pretrain.py:6242)
- MSTA 每次 forward 执行 `target.max().item()` 和 GPU `any()`：[mips_local_graph.py](/root/workspace/Uni-Poly-Plus-master/src/modules/mips_local_graph.py:427)
- Star-RBF v2 forward 中的 `min/max/any` 严格检查也会触发同步：[mips_local_graph.py](/root/workspace/Uni-Poly-Plus-master/src/modules/mips_local_graph.py:183)

建议把不变量检查前移到 collate/首批验证，正式热路径只保留必要检查；指标使用 GPU 累加器，到日志间隔再同步。是否真正提速必须用同一 benchmark 复测。

## 6. Sidecar 启动和存储也可以优化

当前 Star-RBF v2 artifact 约 `7.4 GB`。每次构造 sidecar 默认都会：

- 对全部数组重新计算 SHA256；
- 扫描全局 min/max/valid 条件；
- 打开所有 QC 数组。

见 [mts_star_rbf_v2.py](/root/workspace/Uni-Poly-Plus-master/src/dataset/mts_star_rbf_v2.py:250)。

可以在保持严格身份检查的前提下改成：

- rank 0 完成一次完整 hash/audit；
- barrier 后其他 ranks 只验证 `.done/.frozen/metadata` 并 mmap；
- 为训练提供只包含模型必需数组的轻量 read view；
- QC 数组继续留在原 artifact，由审计工具按需读取。

这主要优化启动时间和系统 I/O，不会显著改变稳态 samples/s。

## 7. 下游质量优化应优先覆盖 XC

当前 T1 的宏平均约 `0.840`，而 XC 约 `0.425`，与项目记录的 `0.579` 相差约 `0.154`。EI、EEA 也还有明显差距。报告见 [T1 下游结果](/root/workspace/Uni-Poly-Plus-master/results/mts_multiscale_topology/t0_t1_formal_v1/T1_msta/final_report.md:1) 和 [best_result.csv](/root/workspace/Uni-Poly-Plus-master/results/best_result.csv:1)。

因此建议在几何预训练之后开展：

1. 八任务统一的多任务微调，共享主干并使用 task embedding。
2. 使用 uncertainty weighting 或 PCGrad 处理任务梯度冲突。
3. 对 Graph-only 与 Graph+SMILES 做 matched 比较。
4. XC 只作为重点观察任务，不单独修改网络或超参数追分。
5. 保留独立预测与 task-level delta，避免把 40 folds 当作独立样本。

微调速度本身已经较成熟：4 GPU 单卡 slots、workers 2、eval batch 64、FP32；BF16 只有 `1.019x`，暂时没有继续优化价值。

## 建议的实际顺序

1. 先更新实验状态：当前 handoff 仍是 `ready_for_claude`，但正式训练已经停止；另一个对话可能误启动 R2。
2. 保存当前源码 hash/diff 快照，避免再次出现无法复原实验代码的问题。
3. 对现有 R2 做 20-step profiler，只优化 forward 热点。
4. 决定是完整跑完 R2，还是把它归档为未完成实验；不要在未完成 R2 上叠加新目标。
5. 新建 matched 预训练目标实验：`R2 masked-atom-only` 对比 `R2 + invariant geometry denoising`。
6. 先用 2k/5k/10k milestones 做明确标记的 screening，再决定是否执行正式 20k。
7. 若几何去噪有效，再加入 shift-conditioned message gate；最后才考虑等变坐标头和多构象。
8. 之后单独开展多任务/多模态微调，重点改善 XC、EI、EEA。

本轮是只读静态审查和已有实验证据核对；没有修改文件、运行测试或启动训练。